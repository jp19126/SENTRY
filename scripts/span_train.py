"""Serial, finite dispatches for approved span_coupled_pilot_v1.

Never reads protected partitions. Train/score/export are separate commands;
outputs must be fresh children of this study's results directory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gate_eval import fit_threshold,evaluate_documents
from r1_challengers import resource_check,utc,json_write,execution_settings

DEST=ROOT/"results"/"span_coupled_pilot_v1"
ALLOWED=("train","search_temporary_threshold","search_candidate_scoring","stress_probes")


class Budget:
    def __init__(self,seconds,folder):
        self.start=time.monotonic();self.seconds=seconds;self.last=self.start;self.folder=folder
        self.observations=[resource_check(True)];self.save()
    def save(self,error=None):
        json_write(self.folder/"resources.json",dict(observations=self.observations,max_seconds=self.seconds,
            elapsed_seconds=time.monotonic()-self.start,error=error,runtime_poll_seconds=5))
    def check(self):
        try:
            now=time.monotonic()
            if now-self.start>self.seconds:raise RuntimeError("Dispatch budget exhausted")
            if (DEST/"pause.request").exists():raise RuntimeError("Study pause requested")
            if now-self.last>=5:
                self.observations.append(resource_check());self.last=now;self.save()
        except Exception as e:
            self.save(str(e));raise


def setup(seed,device):
    import numpy as np
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device=="cuda":
        if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    return torch


def load_split(prepared,split):
    import numpy as np
    if split not in ALLOWED:raise ValueError("Closed/unknown split refused")
    path=prepared/split
    required=("ids","masks","labels","doc_indices","starts","content_start","document_content_length")
    arrays={name:np.load(path/(name+".npy"),mmap_mode="r",allow_pickle=False) for name in required}
    docs=[json.loads(line) for line in (path/"documents.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(arrays["starts"])!=len(docs)+1 or arrays["starts"][-1]!=len(arrays["ids"]):raise ValueError("Incomplete prepared windows")
    if any(len(arrays[k])!=len(arrays["ids"]) for k in required if k!="starts"):raise ValueError("Unmatched arrays")
    return arrays,docs


def batch(data,indices,device,torch):
    import numpy as np
    return {target:torch.from_numpy(np.array(data[source][indices],dtype=np.int64)).to(device)
        for source,target in (("ids","input_ids"),("masks","attention_mask"),
            ("content_start","content_start"),("document_content_length","document_content_length"))}


def checkpoint(path,model,epoch,seed,torch):
    from span_models import model_metadata
    torch.save(dict(state_dict=model.state_dict(),epoch=epoch,seed=seed,model=model_metadata(model),
        qat=model.qat,torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),path)


def load_checkpoint(path,device,torch):
    from span_models import SpanGuard
    saved=torch.load(path,map_location=device,weights_only=False)
    info=saved["model"]
    model=SpanGuard(info["arm"],info["vocab_size"],info["pad_id"]).to(device)
    model.load_state_dict(saved["state_dict"]);model.qat=False;model.eval()
    return model,saved


def train(args,folder,budget,torch):
    from span_models import SpanGuard,model_metadata
    from span_numerics import export_model
    summary=json.loads((args.prepared/"summary.json").read_text(encoding="utf-8-sig"))
    if not summary.get("complete"):raise RuntimeError("Preparation not complete")
    data,docs=load_split(args.prepared,"train")
    model=SpanGuard(args.arm,summary["vocab_size"],summary["pad_id"]).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    run=dict(command=sys.argv,model=model_metadata(model),seed=args.seed,epochs=10,fp32_epochs=8,qat_epochs=2,
        optimizer="AdamW",learning_rate=.001,weight_decay=.01,document_batch=32,
        microbatch_windows=args.microbatch,gradient_clip=5.,loss="mean within each document of span-label window CE; mean over document batch",
        prepared=str(args.prepared),execution=execution_settings(torch),complete=False,
        source_sha256={name:hashlib.sha256((ROOT/"scripts"/name).read_bytes()).hexdigest()
            for name in ("span_models.py","span_numerics.py","span_train.py")},
        prepared_contract_sha256=hashlib.sha256((args.prepared/"contract.json").read_bytes()).hexdigest())
    json_write(folder/"run.json",run)
    count=len(docs);history=[]
    with (folder/"training_log.jsonl").open("x",encoding="utf-8") as log:
        for epoch in range(1,11):
            model.train();model.qat=epoch>8;model.observe=epoch==8
            order=list(range(count));random.Random(args.seed+epoch).shuffle(order)
            start_epoch=time.monotonic();loss_sum=0.;seen=0;no_anchor_positive=0;no_anchor_total=0
            for bi,offset in enumerate(range(0,count,32)):
                budget.check();chosen=order[offset:offset+32];indices=[];weights=[]
                for d in chosen:
                    first,end=map(int,data["starts"][d:d+2]);indices.extend(range(first,end))
                    weights.extend([1/(len(chosen)*(end-first))]*(end-first))
                optimizer.zero_grad(set_to_none=True);batch_loss=0.
                for at in range(0,len(indices),args.microbatch):
                    current=indices[at:at+args.microbatch]
                    out=model(**batch(data,current,args.device,torch))
                    labels=torch.as_tensor(data["labels"][current].copy(),device=args.device,dtype=torch.long)
                    scales=torch.tensor(weights[at:at+args.microbatch],device=args.device)
                    loss=(torch.nn.functional.cross_entropy(out.logits,labels,reduction="none")*scales).sum()
                    if not bool(torch.isfinite(loss)):raise RuntimeError("Nonfinite loss")
                    loss.backward();batch_loss+=float(loss.detach())
                    no_anchor_total+=int(out.no_anchor.sum());no_anchor_positive+=int((out.no_anchor & labels.bool()).sum())
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                if not bool(torch.isfinite(norm)):raise RuntimeError("Nonfinite gradient")
                optimizer.step();loss_sum+=batch_loss*len(chosen);seen+=len(chosen)
                if bi%50==0:
                    record=dict(event="training",arm=args.arm,seed=args.seed,epoch=epoch,batch=bi,
                        documents=seen,loss=loss_sum/seen,elapsed_seconds=time.monotonic()-budget.start)
                    log.write(json.dumps(record)+"\n");log.flush();print(json.dumps(record),flush=True)
            record=dict(epoch=epoch,loss=loss_sum/seen,seconds=time.monotonic()-start_epoch,
                no_anchor_windows=no_anchor_total,no_anchor_positive_windows=no_anchor_positive)
            history.append(record);json_write(folder/"history.json",history)
            if epoch==8:
                model.observe=False;model.freeze_quantization()
                checkpoint(folder/"fp32_epoch8.pt",model,epoch,args.seed,torch)
            if epoch==10:checkpoint(folder/"qat_epoch10.pt",model,epoch,args.seed,torch)
            print(json.dumps(dict(event="epoch_complete",**record)),flush=True)
    model.eval();export_model(model,folder/"export")
    run["complete"]=True;run["elapsed_seconds"]=time.monotonic()-budget.start
    json_write(folder/"run.json",run);budget.save()


def score(args,folder,budget,torch):
    from span_numerics import IntegerSpan
    if not args.checkpoint and not args.export:raise ValueError("--checkpoint FP32 or --export fixed required")
    fixed=args.export is not None
    model=IntegerSpan(args.export,args.device) if fixed else load_checkpoint(args.checkpoint,args.device,torch)[0]
    thresholds=None;summary={};source_groups={}
    for split in ("search_temporary_threshold","search_candidate_scoring","stress_probes"):
        data,docs=load_split(args.prepared,split);source_groups[split]={d["source_group_id"] for d in docs}
        scores=[];no_anchor=0
        with torch.no_grad():
            for first in range(0,len(data["ids"]),args.microbatch):
                budget.check();indices=list(range(first,min(first+args.microbatch,len(data["ids"]))))
                output=model(**batch(data,indices,args.device,torch),head_swap=args.head_swap)
                risk=output["risk"] if fixed else output.risk
                absent=output["no_anchor"] if fixed else output.no_anchor
                scores.extend(risk.detach().cpu().tolist());no_anchor+=int(absent.sum())
                if first%512==0:print(json.dumps(dict(event="score_progress",split=split,arithmetic="integer" if fixed else "FP32",windows_done=len(scores))),flush=True)
        documents=[]
        for i,d in enumerate(docs):
            first,end=map(int,data["starts"][i:i+2]);s=scores[first:end]
            documents.append(dict(d,risk_score=max(s),window_scores=s))
        with (folder/(split+"_documents.jsonl")).open("x",encoding="utf-8") as stream:
            for d in documents:stream.write(json.dumps(d,ensure_ascii=False)+"\n")
        if split=="search_temporary_threshold":
            thresholds={str(alpha):fit_threshold([d["risk_score"] for d in documents if d["label"]==0],alpha) for alpha in (.005,.01,.02)}
        summary[split]={"documents":len(docs),"windows":len(scores),"no_anchor_windows":no_anchor,
            "metrics":{alpha:evaluate_documents([d["risk_score"] for d in documents],[d["label"] for d in documents],t) for alpha,t in thresholds.items()}}
    if source_groups["search_temporary_threshold"] & (source_groups["search_candidate_scoring"] | source_groups["stress_probes"]):raise ValueError("Calibration source leakage")
    json_write(folder/"metrics.json",dict(complete=True,arithmetic="integer own-reference" if fixed else "FP32",
        head_swap=args.head_swap,thresholds=thresholds,results=summary,command=sys.argv))
    budget.save();print(json.dumps(summary,indent=2),flush=True)


def score_bundle(args,folder,budget,torch):
    import copy
    specs=[("fp32_epoch8",args.run/"fp32_epoch8.pt",None,False),
           ("fp32_postqat",args.run/"qat_epoch10.pt",None,False),
           ("integer",None,args.run/"export",False)]
    info=torch.load(args.run/"fp32_epoch8.pt",map_location="cpu",weights_only=False)["model"]
    if info["arm"]=="coupled":
        specs.extend([("swap_fp32_postqat",args.run/"qat_epoch10.pt",None,True),
                      ("swap_integer",None,args.run/"export",True)])
    for name,cp,exp,swap in specs:
        budget.check();target=folder/name;target.mkdir()
        variant=copy.copy(args);variant.checkpoint=cp;variant.export=exp;variant.head_swap=swap
        score(variant,target,budget,torch)
    json_write(folder/"bundle.json",dict(complete=True,run=str(args.run),variants=[s[0] for s in specs],
        shared_budget_seconds=args.max_seconds,elapsed_seconds=time.monotonic()-budget.start))


def real_bringup(args,folder,budget,torch):
    import numpy as np
    from span_models import SpanGuard,model_metadata
    from span_numerics import export_model,IntegerSpan,np_rne_scalar
    data,docs=load_split(args.prepared,"train")
    selected=[]
    for label in (0,1):
        candidates=[i for i,d in enumerate(docs) if d["label"]==label]
        random.Random(args.seed+label).shuffle(candidates);selected.extend(candidates[:16])
    indices=[]
    for i in selected:
        first,end=map(int,data["starts"][i:i+2]);indices.extend(range(first,end))
    info=json.loads((args.prepared/"summary.json").read_text(encoding="utf-8-sig"))
    model=SpanGuard(args.arm,info["vocab_size"],info["pad_id"]).to(args.device)
    model.observe=True
    with torch.no_grad():
        for at in range(0,len(indices),args.microbatch):
            budget.check();model(**batch(data,indices[at:at+args.microbatch],args.device,torch))
    model.observe=False;model.freeze_quantization();export_model(model,folder/"export")
    cpu=IntegerSpan(folder/"export","cpu",False)
    accelerated=IntegerSpan(folder/"export",args.device,args.device=="cuda")
    fixture_count=0;scalar_state_checks=0;duplicates=0;seen={};no_anchor_positive=0
    with torch.no_grad():
        for at in range(0,len(indices),args.microbatch):
            budget.check();current=indices[at:at+args.microbatch]
            inp=batch(data,current,args.device,torch);ref=cpu(**{k:v.cpu() for k,v in inp.items()},trace=True)
            act=accelerated(**inp,trace=True)
            for name in ("risk","logit_q12","z_a8","projection_q12"):
                if not torch.equal(ref[name],act[name].cpu()):raise AssertionError("Exact route "+name)
            for n,w in enumerate(current):
                no_anchor_positive+=int(bool(ref["no_anchor"][n]) and bool(data["labels"][w]))
                anchors=ref["eligible"][n].nonzero().flatten().tolist()
                if ref["state_q12"] is not None:
                    for anchor in anchors[::max(1,len(anchors)//3)]:
                        state=[0]*64;lam=cpu.a["lambda_q8"].reshape(-1).tolist();p=ref["projection_q12"][n]
                        for position in range(anchor-28,anchor+30):
                            u=p[position].tolist() if 0<=position<len(p) else [0]*64
                            state=[min(32768,max(-32768,np_rne_scalar(v*l,-8)+int(u0))) for v,l,u0 in zip(state,lam,u)]
                        if state!=ref["state_q12"][n,anchor].tolist():raise AssertionError("Literal scalar recurrence")
                        scalar_state_checks+=1
                if ref["anchor_logits_q12"] is not None:
                    for anchor in anchors:
                        key=(int(data["doc_indices"][w]),int(data["content_start"][w])+anchor-1)
                        value=int(ref["anchor_logits_q12"][n,anchor])
                        if key in seen:
                            duplicates+=1
                            if seen[key]!=value:raise AssertionError("Real duplicate anchor")
                        seen[key]=value
            if fixture_count<4:
                torch.save(dict(inputs={k:v.cpu() for k,v in inp.items()},outputs=ref),folder/("fixture_%02d.pt"%fixture_count));fixture_count+=1
    json_write(folder/"bringup.json",dict(complete=True,synthetic_only=False,model=model_metadata(model),
        documents=[docs[i]["document_id"] for i in selected],windows=len(indices),
        scalar_state_checks=scalar_state_checks,duplicate_anchor_checks=duplicates,
        no_anchor_positive_windows=no_anchor_positive,execution=execution_settings(torch),
        scope="Untrained model, 32 train-only documents; exact CPU/CUDA/scalar arithmetic checks, not quality measurement"))
    budget.save();print((folder/"bringup.json").read_text(encoding="utf-8"),flush=True)


def bringup(args,folder,budget,torch):
    from span_models import SpanGuard,ARMS,model_metadata
    from span_numerics import export_model,IntegerSpan,rne_shift,np_rne_scalar
    # Synthetic contract checks; this is not a real-data quality experiment.
    values=list(range(-129,130))
    v=torch.tensor(values,dtype=torch.int64)
    for shift in range(-7,5):
        expected=torch.tensor([np_rne_scalar(x,shift) for x in values])
        if not torch.equal(rne_shift(v,shift),expected):raise AssertionError("Signed RNE")
    records=[]
    for arm in ARMS:
        budget.check();torch.manual_seed(args.seed)
        model=SpanGuard(arm,128,0).to(args.device)
        # Same 444-token synthetic document, two 254-token windows overlap64.
        seq=torch.randint(3,128,(444,),device=args.device)
        ids=torch.zeros((2,256),dtype=torch.long,device=args.device)
        ids[:,0]=1;ids[:,-1]=2;ids[0,1:255]=seq[:254];ids[1,1:255]=seq[190:444]
        inp=dict(input_ids=ids,attention_mask=torch.ones_like(ids),
            content_start=torch.tensor([0,190],device=args.device),document_content_length=torch.tensor([444,444],device=args.device))
        model.observe=True;out=model(**inp);model.observe=False
        loss=torch.nn.functional.cross_entropy(out.logits,torch.tensor([0,1],device=args.device));loss.backward()
        if not bool(torch.isfinite(loss)) or not any(p.grad is not None for p in model.parameters()):raise AssertionError("Gradients")
        if arm in ("coupled","span_mlp"):
            # global anchor221 is the duplicate complete footprint.
            if not torch.allclose(out.anchor_logits[0,222],out.anchor_logits[1,32],rtol=1e-6,atol=1e-6):raise AssertionError("Duplicate complete anchor FP32")
        model.freeze_quantization();model.qat=True
        qat=model(**inp)
        if not bool(torch.isfinite(qat.logits).all()):raise AssertionError("QAT finite")
        export_model(model,folder/(arm+"_export"))
        reference=IntegerSpan(folder/(arm+"_export"),"cpu",False)
        cpuinp={k:x.cpu() for k,x in inp.items()}
        exact=reference(**cpuinp,trace=True)
        if arm in ("coupled","span_mlp"):
            if exact["anchor_logits_q12"][0,222]!=exact["anchor_logits_q12"][1,32]:raise AssertionError("Duplicate complete anchor integer")
        if args.device=="cuda":
            accelerated=IntegerSpan(folder/(arm+"_export"),"cuda",True)(**inp,trace=True)
            for name in ("risk","logit_q12","z_a8","projection_q12"):
                if not torch.equal(accelerated[name].cpu(),exact[name]):raise AssertionError("CUDA integer mismatch "+name)
        # Explicit empty document and short internal, uncovered window.
        edgeids=torch.zeros((2,256),dtype=torch.long);edgeids[:,0]=1;edgeids[0,1]=2;edgeids[1,1]=9;edgeids[1,2]=2
        edgemask=torch.zeros_like(edgeids);edgemask[0,:2]=1;edgemask[1,:3]=1
        edges=reference(edgeids,edgemask,torch.tensor([0,190]),torch.tensor([0,500]))
        if arm!="sopa" and float(edges["risk"][1])!=0:raise AssertionError("No-eligible risk must be0")
        if float(edges["risk"][0])!=float(reference.lut[2048]):raise AssertionError("Empty head bias")
        records.append(dict(arm=arm,model=model_metadata(model),fp32_loss=float(loss.detach()),passed=True,
            fixed_risks=exact["risk"].tolist(),qat_risks=qat.risk.detach().cpu().tolist()))
    base=records[0]["model"]["nonembedding_parameters"]
    if any(abs(r["model"]["nonembedding_parameters"]-base)/base>.05 for r in records):raise AssertionError("Parameter parity")
    json_write(folder/"bringup.json",dict(complete=True,synthetic_only=True,checks=records,signed_rne_cases=len(values)*12,
        execution=execution_settings(torch),command=sys.argv))
    budget.save();print(json.dumps(records,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",required=True,choices=("bringup","train","score","export"))
    p.add_argument("--arm",default="coupled",choices=("coupled","independent","span_mlp","sopa"))
    p.add_argument("--seed",type=int,default=42,choices=(42,1337))
    p.add_argument("--prepared",type=Path,default=DEST/"prepared")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path);p.add_argument("--export",type=Path)
    p.add_argument("--run",type=Path,help="score all frozen variants as one bounded bundle")
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--microbatch",type=int,default=8)
    p.add_argument("--max-seconds",type=int,default=180)
    p.add_argument("--head-swap",action="store_true")
    p.add_argument("--real",action="store_true",help="bringup on fixed 32 train-only documents")
    args=p.parse_args();folder=args.output.resolve();args.prepared=args.prepared.resolve()
    limits={"train":2700,"score":1200,"bringup":600,"export":180}
    if args.max_seconds<=0 or args.max_seconds>limits[args.stage]:raise ValueError("Stage dispatch exceeds finite ceiling")
    if args.microbatch<1 or args.microbatch>64:raise ValueError("Microbatch outside1..64")
    if not args.prepared.is_relative_to(DEST.resolve()):raise ValueError("Prepared inputs outside approved study")
    if not folder.is_relative_to(DEST.resolve()) or folder==DEST.resolve():raise ValueError("Output outside study")
    if folder.exists():raise FileExistsError("Preserve existing output; choose fresh directory")
    folder.mkdir(parents=True);budget=Budget(args.max_seconds,folder)
    try:
        torch=setup(args.seed,args.device)
        if args.stage=="train":train(args,folder,budget,torch)
        elif args.stage=="score":
            if args.run:score_bundle(args,folder,budget,torch)
            else:score(args,folder,budget,torch)
        elif args.stage=="bringup":
            if args.real:real_bringup(args,folder,budget,torch)
            else:bringup(args,folder,budget,torch)
        else:
            from span_numerics import export_model
            if not args.checkpoint:raise ValueError("--checkpoint required")
            model,_=load_checkpoint(args.checkpoint,args.device,torch);export_model(model,folder/"export")
    except BaseException as error:
        budget.save(str(error));json_write(folder/"failure.json",dict(error=repr(error),command=sys.argv,complete=False));raise


if __name__=="__main__":main()
