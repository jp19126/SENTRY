"""One R1 BF16-operand/FP32-accumulation attention reference qualification.

Q/K and softmax-probability/V are RNE BF16 rounded then cast back to FP32.
QK and AV use native CUDA FP32 matmul, TF32 off. Scale/mask/softmax/output
remain FP32. This is neither an accelerated BF16 benchmark nor AIE identity.
Frozen saved W8A8 encoder linears retain actual native torch._int_mm kernels.
Only own development scores/threshold qualify this distinct arithmetic route.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gate_eval import load_config
from r1_challengers import (DEST,SPLITS,Guard,arrays,batch_inputs,contract,fresh,
    json_read,json_write,load_prepared,metrics_for,output_path,rows,setup_torch,utc)
from r1_bert_int8 import CHECKPOINT,backend_record,load_model,require_bringup

ROUTE = "r1_w8a8_bf16_qk_av_operands_fp32_accum_explicit_attention_v1"
U = 2.0**-24


def gamma(k):
    return k*U/(1-k*U)


def replace_attention(model,torch):
    """Single encoder-only eager path mirroring actual BERT attention semantics."""
    class BF16OperandsAttention(torch.nn.Module):
        def __init__(self,original):
            super().__init__()
            if original.position_embedding_type!="absolute" or original.is_decoder:
                raise RuntimeError("Only absolute-position encoder attention is qualified")
            self.query,self.key,self.value = original.query,original.key,original.value
            self.num_attention_heads = original.num_attention_heads
            self.attention_head_size = original.attention_head_size
            self.all_head_size = original.all_head_size
            self.capture = False
            self.observed = None

        def forward(self,hidden_states,attention_mask=None,head_mask=None,encoder_hidden_states=None,
                    past_key_values=None,output_attentions=False,cache_position=None,**kwargs):
            if self.training or head_mask is not None or encoder_hidden_states is not None or past_key_values is not None or output_attentions:
                raise RuntimeError("Reference requires eval encoder with no caches/head masks/attention output")
            batch,length,_ = hidden_states.shape
            def heads(linear):
                return linear(hidden_states).view(batch,length,self.num_attention_heads,self.attention_head_size).transpose(1,2)
            q,k,v = heads(self.query),heads(self.key),heads(self.value)
            qb,kb,vb = [x.to(torch.bfloat16).float() for x in (q,k,v)]
            qk = torch.matmul(qb,kb.transpose(-1,-2))
            scaled_qk = qk/math.sqrt(self.attention_head_size)
            scores = scaled_qk if attention_mask is None else scaled_qk+attention_mask
            probability = torch.softmax(scores,dim=-1)
            pb = probability.to(torch.bfloat16).float()
            av = torch.matmul(pb,vb)
            if self.capture:
                self.observed = {"q":q,"k":k,"v":v,"qb":qb,"kb":kb,"vb":vb,
                    "probability":probability,"pb":pb,"qk":qk,"scaled_qk":scaled_qk,"av":av}
            output = av.transpose(1,2).reshape(batch,length,self.all_head_size)
            return output,None

    for layer in model.bert.encoder.layer:
        layer.attention.self = BF16OperandsAttention(layer.attention.self)
    model.eval()
    return model


def metadata(model,modules,torch):
    result = backend_record(model,"cuda",modules,torch)
    result.update(route=ROUTE,attention_implementation="explicit eager FP32 scale/mask/softmax; BF16-rounded QK/AV operands",
        other_operations="FP32 embeddings/explicit attention scale and softmax/residual/LayerNorm/GELU/pooler/classifier/final softmax; BF16-rounded QK/AV operands",
        operand_rounding="PyTorch FP32→BF16 round-to-nearest-even cast→FP32",
        attention_accumulation="native CUDA FP32 matmul, TF32 disabled",
        attention_operand_dtype_during_matmul="FP32 storing exactly BF16-representable values",
        accelerated_bf16_benchmark=False,exact_AIE_identity=False,new_learning_runs=0,format_variants=1,
        bound_contract={"float32_unit_roundoff":U,"dot_bound":"gamma_K*sum(abs(products))",
            "gamma_K":"K*u/(1-K*u)","QK_K":64,"AV_K":256,
            "scaled_QK_bound":"dot_bound/sqrt64 + gamma_1*abs(reference_dot64/sqrt64)",
            "predeclared":True,"reference":"independent FP64 matmul/sum of BF16-representable operands"})
    return result


def create(prepared,config,native_proof,guard,torch):
    require_bringup(native_proof,"cuda")
    load_prepared(prepared,config)
    guard.check()
    model,_,modules = load_model("cuda",torch)
    model = replace_attention(model,torch)
    guard.check()
    return model,modules


def bound_check(actual,reference,bound,torch):
    error = (actual.double()-reference).abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all() and torch.isfinite(bound).all())
    passed = finite and bool((error<=bound).all())
    normalized = torch.where(bound>0,error/bound,torch.where(error==0,torch.zeros_like(error),torch.full_like(error,float("inf"))))
    return {"elements":actual.numel(),"finite":finite,"passed":passed,
            "violations":int((error>bound).sum()),"max_abs_error":float(error.max()),
            "max_normalized_error":float(normalized.max()),"max_bound":float(bound.max())}


def inspect_dot(left,right,actual,torch):
    # This reference uses FP64 products and reduction, independently of FP32 matmul.
    l64,r64 = left.double(),right.double()
    reference = torch.matmul(l64,r64)
    sum_abs = torch.matmul(l64.abs(),r64.abs())
    bound = gamma(left.shape[-1])*sum_abs
    return bound_check(actual,reference,bound,torch),reference,bound


def bringup(folder,prepared,native_proof,config,guard,torch):
    fresh(folder)
    guard.attach(folder)
    report = {"route":ROUTE,"device":"cuda","recorded_utc":utc(),"passed":False,
              "prospective_unit_roundoff":U,"bound_widening_permitted":False,
              "resource_preflight":guard.preflight}
    try:
        model,modules = create(prepared,config,native_proof,guard,torch)
        report["backend"] = metadata(model,modules,torch)
        data = arrays(prepared,SPLITS[2])
        docs = list(rows(prepared/SPLITS[2]/"documents.jsonl"))
        wanted = {"short":None,"full256":None}
        for index,mask in enumerate(data["masks"]):
            guard.check()
            key = "full256" if int(mask.sum())==256 else "short"
            if wanted[key] is None:
                wanted[key] = index
            if all(v is not None for v in wanted.values()):
                break
        if any(v is None for v in wanted.values()):
            raise RuntimeError("Actual short and full256 development windows required")
        for layer in model.bert.encoder.layer:
            layer.attention.self.capture = True
        checks = []
        with torch.inference_mode():
            for name,index in wanted.items():
                guard.check()
                logits = model(**batch_inputs(data,[index],"cnn","cuda",torch)).logits.float()
                case = {"case":name,"prepared_window_index":index,"valid_tokens":int(data["masks"][index].sum()),
                    "document_id":docs[int(data["doc_indices"][index])]["document_id"],
                    "logits":logits.cpu().tolist(),"risk":torch.softmax(logits,-1)[0,1].item(),"layers":[]}
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Nonfinite BF16-operand logits")
                for layer_index,layer in enumerate(model.bert.encoder.layer):
                    guard.check()
                    observed = layer.attention.self.observed
                    rounded = {raw:torch.equal(observed[rounded],observed[raw].to(torch.bfloat16).float())
                        for raw,rounded in (("q","qb"),("k","kb"),("v","vb"),("probability","pb"))}
                    qk,reference,bound = inspect_dot(observed["qb"],observed["kb"].transpose(-1,-2),observed["qk"],torch)
                    divisor = math.sqrt(layer.attention.self.attention_head_size)
                    scaled_reference = reference/divisor
                    scaled_bound = bound/divisor+gamma(1)*scaled_reference.abs()
                    scaled = bound_check(observed["scaled_qk"],scaled_reference,scaled_bound,torch)
                    av,_,_ = inspect_dot(observed["pb"],observed["vb"],observed["av"],torch)
                    result = {"layer":layer_index,"BF16_cast_values_exact":rounded,"QK":qk,"scaled_QK":scaled,"AV":av,
                              "probabilities_finite":bool(torch.isfinite(observed["probability"]).all())}
                    case["layers"].append(result)
                    if not all(rounded.values()) or not all(check["passed"] for check in (qk,scaled,av)) or not result["probabilities_finite"]:
                        checks.append(case)
                        report["checks"] = checks
                        raise RuntimeError("Predeclared BF16-operand numerical check failed: "+json.dumps(result))
                checks.append(case)
                report["checks"] = checks
        report.update(passed=True,operator_checks=24,rounded_operand_checks=32,
                      exact_AIE_identity=False,accelerated_bf16_benchmark=False)
        json_write(folder/"bringup.json",report)
    except Exception as error:
        report["failure"] = {"type":type(error).__name__,"message":str(error)}
        json_write(folder/"bringup.json",report)
        raise


def require_proof(path):
    proof = json_read(path/"bringup.json")
    if not proof["passed"] or proof["route"]!=ROUTE or proof["backend"]["checkpoint"]!=str(CHECKPOINT):
        raise RuntimeError("This exact BF16-operand route needs its own passed bringup")


def score(folder,prepared,native_proof,proof,config,guard,torch):
    require_proof(proof)
    fresh(folder)
    guard.attach(folder)
    model,modules = create(prepared,config,native_proof,guard,torch)
    json_write(folder/"backend.json",metadata(model,modules,torch))
    outputs = {}
    for split in SPLITS[1:]:
        data = arrays(prepared,split)
        docs = list(rows(prepared/split/"documents.jsonl"))
        risks = []
        began = time.perf_counter()
        with torch.inference_mode():
            for index in range(len(data["ids"])):
                guard.check()
                logits = model(**batch_inputs(data,[index],"cnn","cuda",torch)).logits.float()
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Nonfinite BF16-operand scoring logits")
                risks.append(torch.softmax(logits,-1)[0,1].item())
                if (index+1)%1000==0:
                    print(json.dumps({"event":"scoring","split":split,"windows":index+1}),flush=True)
        with (folder/(split+"_windows.jsonl")).open("x",encoding="utf-8") as handle:
            for index,risk in enumerate(risks):
                doc_index = int(data["doc_indices"][index])
                handle.write(json.dumps({"document_id":docs[doc_index]["document_id"],
                    "window_index":index-int(data["starts"][doc_index]),"risk_score":risk})+"\n")
        for index,doc in enumerate(docs):
            first,end = data["starts"][index:index+2]
            doc.update(risk_score=max(risks[first:end]),window_scores=risks[first:end])
        with (folder/(split+"_documents.jsonl")).open("x",encoding="utf-8") as handle:
            for doc in docs:
                handle.write(json.dumps(doc,ensure_ascii=False)+"\n")
        outputs[split] = docs
        print(json.dumps({"event":"scored","split":split,"documents":len(docs),"windows":len(risks),
                          "seconds":time.perf_counter()-began}),flush=True)
    quality = metrics_for(outputs[SPLITS[1]],outputs[SPLITS[2]],contract(config))
    quality.update(route=ROUTE,checkpoint=str(CHECKPOINT),device="cuda",batch_size=1,complete=True,
                   backend=metadata(model,modules,torch),bringup=str(proof/"bringup.json"),
                   exact_AIE_identity=False,accelerated_bf16_benchmark=False)
    json_write(folder/"metrics.json",quality)
    print(json.dumps(quality["candidate_scoring"]),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("bringup","score"),required=True)
    parser.add_argument("--prepared",default="results/goal_r1/prepared_v1")
    parser.add_argument("--output")
    parser.add_argument("--native-bringup-dir",default="results/goal_r1/bert_int8_cuda_bringup_v1")
    parser.add_argument("--bringup-dir",default="results/goal_r1/bf16_attention_bringup_v1")
    parser.add_argument("--max-seconds",type=int,default=600)
    args = parser.parse_args()
    if not 1<=args.max_seconds<=600:
        raise ValueError("BF16-operand dispatch ceiling is600seconds")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    os.environ.setdefault("OMP_NUM_THREADS","4")
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    config = load_config()
    contract(config)
    guard = Guard(args.max_seconds)
    guard.check()
    torch = setup_torch("cuda")
    folder = output_path(args.output,DEST/("bf16_attention_"+args.stage+"_v1"))
    prepared = output_path(args.prepared,DEST/"prepared_v1")
    native_proof = output_path(args.native_bringup_dir,DEST/"unused")
    proof = output_path(args.bringup_dir,DEST/"unused")
    if args.stage=="bringup":
        bringup(folder,prepared,native_proof,config,guard,torch)
    else:
        score(folder,prepared,native_proof,proof,config,guard,torch)


if __name__=="__main__":
    main()
