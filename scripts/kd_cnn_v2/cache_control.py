"""Make a fit-only shuffled pooled-representation control from a pinned teacher cache.

The permutation is global across fitting windows, independent of labels, and
applies only to pooled features. Teacher logit margins, prepared inputs,
document IDs/order, and source cache bytes remain unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from train import identity, load_prepared, read, sha, write
from teacher_cache import load_cache

SEED = 20261015
CONTROL_ID = "shuffled_pooled_fit_only_v1"


def task_path(canonical, path):
    path = Path(path)
    path = (path if path.is_absolute() else canonical / path).resolve()
    if not path.is_relative_to((canonical / "results/kd_cnn_v2").resolve()):
        raise ValueError("Input/output must remain within task-owned results")
    return path


def deranged_permutation(size):
    if size < 2:
        raise ValueError("Permutation requires at least two windows")
    permutation = np.random.default_rng(SEED).permutation(size).astype(np.int64, copy=False)
    fixed = np.flatnonzero(permutation == np.arange(size))
    if len(fixed) > 1:
        permutation[fixed] = np.roll(permutation[fixed], 1)
    elif len(fixed) == 1:
        i = int(fixed[0]); j = (i + 1) % size
        permutation[i], permutation[j] = permutation[j], permutation[i]
    if np.any(permutation == np.arange(size)) or not np.array_equal(np.sort(permutation), np.arange(size)):
        raise ValueError("Global row permutation is invalid or contains aligned rows")
    return permutation


def quantiles(values):
    values = np.asarray(values,dtype=np.float64)
    if not len(values):
        return None
    return {name:float(np.quantile(values,q)) for name,q in
            (("min",0.),("p01",.01),("p05",.05),("p25",.25),("median",.5),
             ("p75",.75),("p95",.95),("p99",.99),("max",1.))}


def signal_diagnostic(canonical, fit_root, source_root, fitting, cached, source_manifest, source_manifest_sha):
    expanded_manifest = read(fit_root / "manifest.json")
    base_manifest_path = canonical / "results/cnn_adoption_v1/prepared/token_r2/fit/manifest.json"
    if sha(base_manifest_path) != expanded_manifest["components"]["base_fit_manifest_sha256"]:
        raise ValueError("Base FIT identity differs from expanded FIT component")
    base_count = int(read(base_manifest_path)["documents"])
    if not 0 < base_count < len(fitting["documents"]):
        raise ValueError("Invalid original FIT / appended transfer boundary")
    labels = np.asarray([d["label"] for d in fitting["documents"]],dtype=np.int8)
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("Fit-only diagnostic requires trusted binary labels")
    margins = np.asarray(cached["document_margins"],dtype=np.float64)
    def subset(indices):
        selected = margins[indices]
        truth = labels[indices]
        predicted = selected > 0.
        probabilities = {}
        for temperature in (1.,2.):
            p = 1. / (1. + np.exp(-np.clip(selected/temperature,-80.,80.)))
            probabilities[str(int(temperature))] = {
                "fraction_below_0_01_or_above_0_99":float(np.mean((p < .01) | (p > .99))),
                "fraction_below_0_001_or_above_0_999":float(np.mean((p < .001) | (p > .999))),
                "probability_quantiles":quantiles(p),
            }
        return {
            "documents":len(selected),"labels":{"benign":int((truth==0).sum()),"attack":int((truth==1).sum())},
            "margin_quantiles":quantiles(selected),
            "margin_quantiles_by_true_label":{
                "benign":quantiles(selected[truth==0]),"attack":quantiles(selected[truth==1])},
            "raw_sign_agreement":{"threshold_margin":0.,"strict_positive_decision":True,
                                  "false_positive_count":int(np.count_nonzero(predicted & (truth==0))),
                                  "false_negative_count":int(np.count_nonzero(~predicted & (truth==1))),
                                  "agreed_count":int(np.count_nonzero(predicted==truth))},
            "sigmoid_margin_saturation":probabilities,
        }
    rng = np.random.default_rng(SEED)
    pooled = cached["pooled"]
    sample_indices = rng.choice(len(pooled),size=min(1024,len(pooled)),replace=False)
    features = np.asarray(pooled[sample_indices],dtype=np.float32)
    variance = np.var(features,axis=0,dtype=np.float64)
    unit = features / np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)
    first = rng.integers(0,len(unit),size=4096)
    second = rng.integers(0,len(unit)-1,size=4096)
    second += second >= first
    cosines = np.einsum("ij,ij->i",unit[first],unit[second]).astype(np.float64)
    sample_digest = hashlib.sha256(sample_indices.astype(np.int64).tobytes()).hexdigest()
    pairs_digest = hashlib.sha256(np.stack((first,second),axis=1).astype(np.int64).tobytes()).hexdigest()
    return {
        "schema":"kd_cnn_v2_fit_only_teacher_signal_diagnostic_v1",
        "source_cache_manifest_sha256":source_manifest_sha,
        "expanded_fit_manifest_sha256":sha(fit_root / "manifest.json"),
        "base_fit_manifest_sha256":sha(base_manifest_path),
        "purpose":"Describe original teacher logit saturation and pooled target variability before interpreting richer KD; no candidate tuning.",
        "cohorts":{"expanded_fit":subset(np.arange(len(margins))),
                   "original_base_fit":subset(np.arange(base_count)),
                   "appended_fit_transfer":subset(np.arange(base_count,len(margins)))},
        "pooled_feature_summary":{
            "unit":"window, masked mean concatenated with max",
            "shape":list(pooled.shape),"dtype":str(pooled.dtype),
            "payload_bytes":int(pooled.nbytes),"source_file_bytes":int((source_root / "pooled.npy").stat().st_size),
            "coordinate_variance":variance.tolist(),"coordinate_variance_quantiles":quantiles(variance),
            "near_zero_variance_coordinate_count":int(np.count_nonzero(variance <= 1e-8)),
            "unlabeled_sample_seed":SEED,"unlabeled_sample_windows":len(sample_indices),
            "unlabeled_sample_indices_sha256":sample_digest,"cosine_distinct_pairs":len(cosines),
            "cosine_pair_indices_sha256":pairs_digest,"cosine_quantiles":quantiles(cosines),
            "cosine_mean":float(np.mean(cosines)),"cosine_std":float(np.std(cosines)),
            "cosine_fraction_above_0_99":float(np.mean(cosines > .99)),
            "max_float16_pool_absolute_conversion_error":source_manifest["max_float16_pool_absolute_error"],
        },
        "window_margin_shape":list(cached["window_margins"].shape),
        "document_margin_shape":list(cached["document_margins"].shape),
        "interpretation_limit":"Raw margin-sign agreement with constructed trusted labels is not behavioral correctness or injection success. Saturation/variance/cosine summaries do not establish that pooled coordinates carry security-relevant information; FP16 conversion error is separately recorded. All summaries are FIT-only.",
    }


def build(canonical, fit_root, source_root, output):
    canonical = Path(canonical).resolve()
    fit_root, source_root, output = (task_path(canonical, value) for value in (fit_root, source_root, output))
    if source_root == output or output.exists():
        raise FileExistsError("Control output must be new and separate from source cache")
    fitting = load_prepared(fit_root, "fit")
    source_manifest_path = source_root / "manifest.json"
    source_manifest_sha = sha(source_manifest_path)
    source_manifest = read(source_manifest_path)
    if source_manifest.get("intervention") or source_manifest.get("control_id"):
        raise ValueError("Source must be the original unshuffled teacher cache")
    if source_manifest.get("association",{}).get("teacher_layer") != 2:
        raise ValueError("Expected original pooled teacher layer 2")
    if source_manifest.get("fitting") != fitting["identity"]:
        raise ValueError("Source cache and expanded fitting material differ")
    expected_teacher = {"checkpoint_sha256": source_manifest["association"]["teacher_checkpoint_sha256"]}
    cached = load_cache(source_root, fitting, expected_teacher)
    source_files = dict(source_manifest["files_sha256"])
    permutation = deranged_permutation(len(fitting["ids"]))
    output.mkdir(parents=True)
    try:
        np.save(output / "permutation.npy", permutation, allow_pickle=False)
        original_pooled = cached["pooled"]
        pooled = np.lib.format.open_memmap(output / "pooled.npy", mode="w+", dtype=original_pooled.dtype,
                                           shape=original_pooled.shape)
        for lo in range(0,len(permutation),1024):
            hi = min(lo+1024,len(permutation))
            pooled[lo:hi] = original_pooled[permutation[lo:hi]]
        pooled.flush()
        del pooled
        for name in ("window_margins.npy","document_margins.npy"):
            shutil.copyfile(source_root / name, output / name)
            if sha(output / name) != source_files[name]:
                raise ValueError("Teacher margin copy differs from source")
        # A full row-wise equality check catches mapping/write errors while
        # preserving bitwise float16 pooled values and unchanged margins.
        result_pooled = np.load(output / "pooled.npy", mmap_mode="r", allow_pickle=False)
        for lo in range(0,len(permutation),1024):
            hi = min(lo+1024,len(permutation))
            if not np.array_equal(result_pooled[lo:hi],original_pooled[permutation[lo:hi]]):
                raise ValueError("Shuffled pooled rows differ from permutation map")
        starts = np.asarray(fitting["starts"])
        target_doc = np.searchsorted(starts[1:],np.arange(len(permutation)),side="right")
        source_doc = np.searchsorted(starts[1:],permutation,side="right")
        same_document = int(np.count_nonzero(target_doc == source_doc))
        source_groups = np.asarray([str(d["source_group_id"]) for d in fitting["documents"]])
        same_source_group = int(np.count_nonzero(source_groups[target_doc] == source_groups[source_doc]))
        diagnostic = signal_diagnostic(canonical,fit_root,source_root,fitting,cached,source_manifest,source_manifest_sha)
        write(output / "teacher_signal_diagnostic.json",diagnostic)
        association = dict(source_manifest["association"])
        association["intervention"] = {
            "id": CONTROL_ID,
            "kind": "global_fit_window_pooled_row_shuffle",
            "seed": SEED,
            "permutation_mapping": "output pooled row i equals original source pooled row permutation[i]",
            "permutation_sha256": sha(output / "permutation.npy"),
            "source_cache_manifest_sha256": source_manifest_sha,
            "source_pooled_sha256": source_files["pooled.npy"],
            "alignment": "pooled targets deliberately misaligned with token windows; logits remain aligned",
            "labels_used_for_permutation": False,
        }
        manifest = {
            "complete":True,"role":"fit","control_id":CONTROL_ID,"intervention":association["intervention"],
            "association":association,"association_sha256":identity(association),
            "fitting":fitting["identity"],"document_ids":source_manifest["document_ids"],
            "files_sha256":{name:sha(output/name) for name in
                            ("pooled.npy","window_margins.npy","document_margins.npy","permutation.npy")},
            "source_files_sha256":source_files,"source_cache_manifest_sha256":source_manifest_sha,
            "teacher_signal_diagnostic_sha256":sha(output / "teacher_signal_diagnostic.json"),
            "seed":SEED,"windows":len(permutation),"documents":len(fitting["documents"]),
            "fixed_window_pairs":0,"same_document_pairs":same_document,
            "same_source_group_pairs":same_source_group,
            "pooled_dtype":str(original_pooled.dtype),"margin_dtype":str(cached["window_margins"].dtype),
            "max_float16_pool_absolute_error":source_manifest["max_float16_pool_absolute_error"],
            "purpose":"Causal development control: hold the richer .1 auxiliary regularizer and training recipe fixed while breaking matched teacher representation content.",
            "limit":"A global shuffle preserves the pooled-target marginal distribution but creates artificial mismatched targets. It tests matched representation signal versus an auxiliary regularizer; it is not a deployment model or a new independent teacher.",
            "scope":"Expanded FIT only; no development, confirmation, final, or NotInject data used to construct targets.",
            "command":sys.argv,
        }
        if source_manifest["document_ids"] != [d["document_id"] for d in fitting["documents"]]:
            raise ValueError("Original teacher document order differs")
        if sha(source_manifest_path) != source_manifest_sha or any(sha(source_root/name) != digest for name,digest in source_files.items()):
            raise ValueError("Original cache changed during control construction")
        write(output / "manifest.json",manifest)
        validated = load_cache(output,fitting,expected_teacher)
        if validated["identity"]["association_sha256"] != manifest["association_sha256"]:
            raise ValueError("Control loader association differs")
        return manifest
    except Exception as error:
        write(output / "failure.json",{"complete":False,"error":str(error),"command":sys.argv})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root",type=Path,required=True)
    parser.add_argument("--fit-root",required=True)
    parser.add_argument("--source-cache",required=True)
    parser.add_argument("--out",required=True)
    args = parser.parse_args()
    result = build(args.canonical_root,args.fit_root,args.source_cache,args.out)
    print(json.dumps({"complete":result["complete"],"control_id":result["control_id"],
                      "windows":result["windows"],"documents":result["documents"],
                      "same_document_pairs":result["same_document_pairs"],
                      "association_sha256":result["association_sha256"]}))


if __name__ == "__main__":
    main()
