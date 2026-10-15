"""Bounded FIT-only linear probe of fixed teacher layers 1, 2, and 4.

One source-group-disjoint 75/25 internal FIT split and one fixed ridge alpha
are descriptive diagnostics. This is neither model selection on reserved data
nor a replacement for per-window richer-KD training evidence.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys

# Limit NumPy BLAS as well as torch; set before importing NumPy.
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
import torch
from models import masked_pool
from teacher_cache import load_cache, load_teacher
from train import load_prepared, setup

SEED = 20261015
RIDGE_ALPHA = 1.0
LAYERS = (1,2,4)
BATCH_WINDOWS = 32
PAIR_COUNT = 4096


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def stable_order(items, phase):
    return sorted(items,key=lambda value:hashlib.sha256(f"{SEED}:{phase}:{value}".encode()).digest())


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def quantiles(values):
    values=np.asarray(values,dtype=np.float64)
    if not len(values):return None
    return {key:float(np.quantile(values,q)) for key,q in
            (("min",0.),("p01",.01),("p05",.05),("p25",.25),("median",.5),
             ("p75",.75),("p95",.95),("p99",.99),("max",1.))}


def auc(scores,labels):
    scores=np.asarray(scores,dtype=np.float64); labels=np.asarray(labels,dtype=np.int8)
    pos=int(labels.sum()); neg=int(len(labels)-pos)
    if not pos or not neg:return None
    _,inverse=np.unique(scores,return_inverse=True)
    p=np.bincount(inverse,weights=labels,minlength=len(np.unique(inverse)))
    n=np.bincount(inverse,weights=1-labels,minlength=len(p))
    return float(np.sum(p*(np.cumsum(n)-n/2))/(pos*neg))


def membership(fitting, base_count, target_base=1000):
    docs=fitting["documents"]
    groups=defaultdict(list)
    for index,doc in enumerate(docs):
        groups[doc["source_group_id"]].append(index)
    base=defaultdict(list)
    transfer=defaultdict(list)
    for group,indices in groups.items():
        if max(indices)<base_count:
            base[docs[indices[0]]["source_dataset"]].append(group)
        elif min(indices)>=base_count:
            transfer[docs[indices[0]]["source_dataset"]].append(group)
        else:
            raise ValueError("Source ancestry crosses base/transfer boundary")
        if len({docs[i]["source_dataset"] for i in indices})!=1:
            raise ValueError("Source group crosses datasets")
    chosen=defaultdict(list)
    source_totals={dataset:sum(len(groups[group]) for group in names) for dataset,names in base.items()}
    for dataset,names in base.items():
        quota=round(target_base*source_totals[dataset]/base_count)
        total=0
        for group in stable_order(names,"sample-original"):
            chosen[("original_base_fit",dataset)].append(group)
            total+=len(groups[group])
            if total>=quota:break
    for dataset,names in transfer.items():
        chosen[("appended_fit_transfer",dataset)] = list(names)
    assignments={}
    for key,names in chosen.items():
        ordered=stable_order(names,"internal-split")
        n_train=max(1,min(len(ordered)-1,round(.75*len(ordered))))
        for position,group in enumerate(ordered):
            split="internal_fit_train" if position<n_train else "internal_fit_validation"
            for index in groups[group]:
                if index in assignments:raise ValueError("Duplicate sampled document")
                assignments[index]=(key[0],key[1],split)
    selected=sorted(assignments)
    rows=[{"document_index":i,"document_id":docs[i]["document_id"],
           "source_group_id":docs[i]["source_group_id"],"source_dataset":assignments[i][1],
           "cohort":assignments[i][0],"split":assignments[i][2],"label":docs[i]["label"],
           "window_start":int(fitting["starts"][i]),"window_end":int(fitting["starts"][i+1])}
          for i in selected]
    if any(row["label"] not in (0,1) for row in rows):raise ValueError("Untrusted FIT label")
    train={row["source_group_id"] for row in rows if row["split"]=="internal_fit_train"}
    validation={row["source_group_id"] for row in rows if row["split"]=="internal_fit_validation"}
    if train & validation:raise ValueError("Source group leaked across internal split")
    if sum(row["cohort"]=="appended_fit_transfer" for row in rows)!=len(docs)-base_count:
        raise ValueError("Not all appended transfer docs selected")
    return rows


def summarize(scores,labels,cohorts,split,decision_threshold=0.):
    scores=np.asarray(scores,dtype=np.float64);labels=np.asarray(labels,dtype=np.int8)
    return {cohort:subset(scores,labels,np.asarray([(s==split and (cohort=="all" or c==cohort))
        for s,c in cohorts],dtype=bool),decision_threshold)
        for cohort in ("all","original_base_fit","appended_fit_transfer")}


def subset(scores,labels,keep,threshold):
    values=scores[keep]; truth=labels[keep]; pred=values>threshold
    result={"documents":len(values),"auc":auc(values,truth),
            "accuracy":float(np.mean(pred==truth)) if len(values) else None,
            "predicted_positive":int(pred.sum()),
            "by_true_label":{str(label):{"documents":int(np.sum(truth==label)),
                "accuracy":float(np.mean(pred[truth==label]==label)) if np.any(truth==label) else None}
                for label in (0,1)}}
    return result


def saturation(margins,labels,cohorts):
    result={}
    for cohort in ("all","original_base_fit","appended_fit_transfer"):
        keep=np.asarray([cohort=="all" or value==cohort for value in cohorts],dtype=bool)
        values=np.asarray(margins)[keep];truth=np.asarray(labels)[keep]
        block={"documents":len(values),"margin_quantiles":quantiles(values),
               "margin_by_true_label":{str(label):quantiles(values[truth==label]) for label in (0,1)},
               "temperature":{}}
        for temperature in (1.,2.,4.):
            probabilities=1./(1.+np.exp(-np.clip(values/temperature,-80.,80.)))
            block["temperature"][str(int(temperature))]={
                "fraction_probability_outside_0_01_0_99":float(np.mean((probabilities<.01)|(probabilities>.99))),
                "fraction_probability_outside_0_001_0_999":float(np.mean((probabilities<.001)|(probabilities>.999))),
                "probability_quantiles":quantiles(probabilities)}
        result[cohort]=block
    return result


def pair_cosines(features,mean,validation_indices):
    X=np.asarray(features[validation_indices],dtype=np.float64)
    rng=np.random.default_rng(SEED)
    a=rng.integers(0,len(X),size=PAIR_COUNT)
    b=rng.integers(0,len(X)-1,size=PAIR_COUNT)
    b+=b>=a
    result={}
    for name,array in (("uncentered",X),("centered_on_internal_train_mean",X-mean)):
        norms=np.maximum(np.linalg.norm(array,axis=1),1e-12)
        cos=np.einsum("ij,ij->i",array[a],array[b])/(norms[a]*norms[b])
        result[name]={"quantiles":quantiles(cos),"mean":float(np.mean(cos)),
                      "fraction_above_0_99":float(np.mean(cos>.99))}
    result["distinct_pair_count"]=PAIR_COUNT
    result["pair_seed"]=SEED
    return result


def ridge(features,labels,train_indices):
    X=np.asarray(features,dtype=np.float64)
    train=X[train_indices]
    mean=train.mean(axis=0);std=train.std(axis=0)
    scale=np.maximum(std,1e-6)
    Z=(X-mean)/scale
    A=np.column_stack((np.ones(len(train_indices)),Z[train_indices]))
    y=np.where(np.asarray(labels)[train_indices]==1,1.,-1.)
    penalty=np.eye(A.shape[1],dtype=np.float64)*RIDGE_ALPHA
    penalty[0,0]=0.
    coefficients=np.linalg.solve(A.T@A+penalty,A.T@y)
    predictions=np.column_stack((np.ones(len(Z)),Z))@coefficients
    if not np.isfinite(predictions).all():raise ValueError("Nonfinite ridge scores")
    return predictions,mean,std


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root",type=Path,required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args();root=args.canonical_root.resolve()
    out=Path(args.out)
    out=(out if out.is_absolute() else root/out).resolve()
    if not out.is_relative_to((root/"results/kd_cnn_v2").resolve()) or out.exists():
        raise ValueError("Output must be a new task-owned directory")
    fit_root=root/"results/kd_cnn_v2/prepared/expanded_fit"
    cache_root=root/"results/kd_cnn_v2/teacher_cache/expanded_layer2_v1"
    old_fit=root/"results/cnn_adoption_v1/prepared/token_r2/fit/manifest.json"
    contract_path=root/"results/dual_track_v1/shared/contract.json"
    contract=load_json(contract_path)
    qualified=load_json(root/"configs/cnn_adoption_v1.json")["qualified_teacher"]
    teacher={"checkpoint":contract["teacher"]["path"],
             "checkpoint_sha256":contract["teacher"]["sha256"],
             "qualified_development_record":f"results/cnn_adoption_v1/evaluation/development/{qualified['run_id']}/fp32/complete.json",
             "qualified_development_record_sha256":qualified["qualified_development_record_sha256"]}
    if qualified["teacher_checkpoint_sha256"]!=teacher["checkpoint_sha256"]:
        raise ValueError("Common/qualified teacher differs")
    fitting=load_prepared(fit_root,"fit")
    cache=load_cache(cache_root,fitting,teacher)
    expanded_manifest=load_json(fit_root/"manifest.json")
    if expanded_manifest["components"]["base_fit_manifest_sha256"]!=digest(old_fit):
        raise ValueError("Base FIT component differs")
    base_count=int(load_json(old_fit)["documents"])
    rows=membership(fitting,base_count)
    if not rows:raise ValueError("Empty internal FIT sample")
    selected_windows=np.concatenate([np.arange(row["window_start"],row["window_end"],dtype=np.int64) for row in rows])
    window_offsets=np.r_[0,np.cumsum([row["window_end"]-row["window_start"] for row in rows])]
    if len(selected_windows)!=len(set(selected_windows.tolist())):
        raise ValueError("Duplicate selected window")
    labels=np.asarray([row["label"] for row in rows],dtype=np.int8)
    cohorts=[(row["split"],row["cohort"]) for row in rows]
    train_indices=np.asarray([i for i,row in enumerate(rows) if row["split"]=="internal_fit_train"])
    val_indices=np.asarray([i for i,row in enumerate(rows) if row["split"]=="internal_fit_validation"])
    if len(train_indices)<2 or len(val_indices)<2:raise ValueError("Invalid internal split")
    # Verify complete windows and masks before extracting features.
    starts=np.load(fit_root/"content_start.npy",mmap_mode="r")
    lengths=np.load(fit_root/"document_content_length.npy",mmap_mode="r")
    for row in rows:
        index=row["document_index"];lo,hi=row["window_start"],row["window_end"]
        expected=[];at=0;length=int(lengths[index])
        while True:
            end=min(at+254,length);expected.append(at)
            if end==length:break
            at=end-64
        if expected!=list(map(int,starts[lo:hi])):
            raise ValueError("Incomplete selected document tail")
    out.mkdir(parents=True)
    try:
        membership_path=out/"membership.json"
        membership_path.write_text(json.dumps({"schema":"kd_cnn_v2_fit_layer_probe_membership_v1",
            "seed":SEED,"sampling":"original FIT source groups hashed within dataset until proportional ~1000-document quota; all appended transfer groups",
            "split":"source-group-disjoint hash-ranked 75/25 within cohort and source dataset",
            "rows":rows},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        setup(42,"cpu",1)
        model=load_teacher(root,teacher,"cpu")
        features={layer:np.empty((len(selected_windows),512),dtype=np.float32) for layer in LAYERS}
        observed_margins=np.empty(len(selected_windows),dtype=np.float32)
        with torch.inference_mode():
            for lo in range(0,len(selected_windows),BATCH_WINDOWS):
                hi=min(lo+BATCH_WINDOWS,len(selected_windows));indices=selected_windows[lo:hi]
                ids=torch.as_tensor(np.array(fitting["ids"][indices],dtype=np.int64,copy=True))
                masks=torch.as_tensor(np.array(fitting["masks"][indices],dtype=np.int64,copy=True))
                output=model(input_ids=ids,attention_mask=masks,output_hidden_states=True)
                observed_margins[lo:hi]=(output.logits[:,1]-output.logits[:,0]).numpy()
                mask=masks.bool().unsqueeze(1)
                for layer in LAYERS:
                    features[layer][lo:hi]=masked_pool(output.hidden_states[layer].transpose(1,2),mask).numpy()
        cache_window=np.asarray(cache["window_margins"][selected_windows],dtype=np.float32)
        cache_pooled=np.asarray(cache["pooled"][selected_windows],dtype=np.float32)
        if any(not np.isfinite(X).all() for X in features.values()):raise ValueError("Nonfinite teacher pooled feature")
        margin_delta=float(np.max(np.abs(cache_window-observed_margins)))
        layer2_delta=float(np.max(np.abs(cache_pooled-features[2])))
        document_features={layer:np.stack([features[layer][int(window_offsets[i]):int(window_offsets[i+1])].max(axis=0)
                                            for i in range(len(rows))]) for layer in LAYERS}
        doc_margins=np.asarray(cache["document_margins"][[row["document_index"] for row in rows]],dtype=np.float64)
        if any(doc_margins[i]!=cache_window[int(window_offsets[i]):int(window_offsets[i+1])].max() for i in range(len(rows))):
            raise ValueError("Selected cached document max margin differs")
        majority=int(np.mean(labels[train_indices])>=.5)
        constant=np.full(len(rows),1. if majority else -1.)
        report={"schema":"kd_cnn_v2_fit_only_layer_probe_v1","complete":True,
            "purpose":"Bounded internal FIT diagnostic of layer-available label signal and shared components; no automatic recipe change.",
            "teacher_checkpoint_sha256":teacher["checkpoint_sha256"],
            "common_contract_sha256":digest(contract_path),
            "qualified_teacher_record_sha256":qualified["qualified_development_record_sha256"],
            "expanded_fit_manifest_sha256":digest(fit_root/"manifest.json"),
            "expanded_cache_manifest_sha256":digest(cache_root/"manifest.json"),
            "original_fit_manifest_sha256":digest(old_fit),
            "membership_sha256":digest(membership_path),
            "source_sha256":{name:digest(Path(__file__).resolve().parents[2]/name) for name in
                ("scripts/kd_cnn_v2/layer_probe.py","scripts/kd_cnn_v2/teacher_cache.py",
                 "scripts/kd_cnn_v2/train.py","scripts/kd_cnn_v2/models.py","scripts/cnn_models.py",
                 "configs/project.json","configs/cnn_adoption_v1.json")},
            "seed":SEED,"sampled_documents":len(rows),"sampled_windows":len(selected_windows),
            "internal_fit_train_documents":len(train_indices),"internal_fit_validation_documents":len(val_indices),
            "sample_counts":{cohort:dict(Counter(row["source_dataset"] for row in rows if row["cohort"]==cohort))
                              for cohort in ("original_base_fit","appended_fit_transfer")},
            "source_groups_by_split":{split:len({row["source_group_id"] for row in rows if row["split"]==split})
                                      for split in ("internal_fit_train","internal_fit_validation")},
            "teacher_cache_numeric_check":{"max_cpu_vs_cached_window_margin_absolute_delta":margin_delta,
                "max_cpu_layer2_vs_cached_fp16_pool_absolute_delta":layer2_delta,
                "original_cache_max_float16_conversion_error":load_json(cache_root/"manifest.json")["max_float16_pool_absolute_error"]},
            "probe":{"method":"ordinary least-squares ridge on standardized document-elementwise-max pooled coordinates",
                "fixed_alpha":RIDGE_ALPHA,"intercept_penalized":False,"standardization":"train-only mean and SD, floor 1e-6",
                "decision":"ridge score > 0","feature_dimensions":512,
                "representation_mismatch":"richer KD aligns each individual window's mean/max halves by cosine; this probe elementwise-maxes all pooled coordinates over a document"},
            "references":{"constant_majority_label_from_internal_train":majority,
                "constant_internal_validation":summarize(constant,labels,cohorts,"internal_fit_validation"),
                "teacher_margin_sign_internal_validation":summarize(doc_margins,labels,cohorts,"internal_fit_validation"),
                "teacher_document_margin_saturation":saturation(doc_margins,labels,[row["cohort"] for row in rows])},
            "layers":{},"environment":{"torch":torch.__version__,"numpy":np.__version__,
                "torch_threads":torch.get_num_threads(),"batch_windows":BATCH_WINDOWS,"device":"cpu",
                "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
                "cuda_matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32},
            "limits":"Teacher trained on original FIT, so internal original validation is not teacher-held-out. Appended transfer source groups were unused by old roles but use one-author response-redirection payloads and a teacher-exposed mechanism. Source-group split prevents document ancestry crossover within this probe only. Fixed ridge is a diagnostic, not a student or security predictor; label AUC/accuracy do not establish behavioral injection success. Centered cosine and linear separability do not establish security-specific information."}
        for layer in LAYERS:
            scores,mean,std=ridge(document_features[layer],labels,train_indices)
            report["layers"][str(layer)]={
                "internal_fit_train":summarize(scores,labels,cohorts,"internal_fit_train"),
                "internal_fit_validation":summarize(scores,labels,cohorts,"internal_fit_validation"),
                "train_coordinate_sd_quantiles":quantiles(std),
                "validation_document_pooled_pair_cosines":pair_cosines(document_features[layer],mean,val_indices),
            }
        report_path=out/"report.json"
        report_path.write_text(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({"report":str(report_path),"sha256":digest(report_path),
                          "documents":len(rows),"windows":len(selected_windows),
                          "validation_auc_by_layer":{key:value["internal_fit_validation"]["all"]["auc"]
                                                     for key,value in report["layers"].items()}}))
    except Exception as error:
        (out/"failure.json").write_text(json.dumps({"complete":False,"error":str(error),"command":sys.argv},indent=2)+"\n",encoding="utf-8")
        raise


if __name__=="__main__":main()
