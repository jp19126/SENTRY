"""Final Goal R1 software optimization: one fixed B1/L256 native W8 CUDA Graph.

Always additive4D SDPA mask is a distinct numerical execution route. It needs
its own bringup, full development scores and threshold. No learning, compile,
new packages, fallback backend or shape/thread search is implemented.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gate_eval import load_config
from r1_challengers import (DEST,SPLITS,Guard,arrays,batch_inputs,contract,fresh,
    json_read,json_write,load_prepared,metrics_for,output_path,real_inputs,rows,
    setup_torch,tokenizer,utc)
from r1_bert_int8 import CHECKPOINT, backend_record, load_model, require_bringup

ROUTE = "r1_native_w8_a8_cuda_graph_always_additive4d_sdpa_b1_l256_v1"


def explicit_forward(model,inputs,torch):
    """Mirror BertModel's encoder path without tensor-to-Python mask reduction."""
    embeddings = model.bert.embeddings(input_ids=inputs["input_ids"],token_type_ids=inputs["token_type_ids"])
    expanded = inputs["attention_mask"][:,None,None,:].expand(1,1,256,256).to(embeddings.dtype)
    inverted = 1.0-expanded
    additive = inverted.masked_fill(inverted.to(torch.bool),torch.finfo(embeddings.dtype).min)
    output = model.bert.encoder(embeddings,attention_mask=additive,head_mask=[None]*4,
        use_cache=False,output_attentions=False,output_hidden_states=False,return_dict=False)
    pooled = model.bert.pooler(output[0])
    logits = model.classifier(model.dropout(pooled)).float()
    risk = torch.softmax(logits,-1)[:,1].max()
    return logits,risk


class GraphBackend:
    def __init__(self,model,first_inputs,torch):
        self.model,self.torch = model,torch
        began = time.perf_counter()
        self.inputs = {k:first_inputs[k].to("cuda").clone() for k in ("input_ids","attention_mask")}
        self.inputs["token_type_ids"] = first_inputs.get("token_type_ids",torch.zeros_like(first_inputs["input_ids"])).to("cuda").clone()
        if any(t.shape!=(1,256) for t in self.inputs.values()):
            raise RuntimeError("Only B1/L256 graph configuration is authorized")
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(),torch.cuda.stream(side):
            for _ in range(3):
                explicit_forward(model,self.inputs,torch)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        warmed = time.perf_counter()
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(),torch.cuda.graph(self.graph):
            self.logits,self.risk = explicit_forward(model,self.inputs,torch)
        torch.cuda.synchronize()
        captured = time.perf_counter()
        self.setup = {"warmup_for_capture_forwards":3,"configuration_count":1,
            "warmup_and_static_input_seconds":warmed-began,"capture_seconds":captured-warmed,
            "total_setup_seconds":captured-began,"static_input_shapes":{k:list(v.shape) for k,v in self.inputs.items()},
            "mask":"always additive4D constructed on device inside graph from2D mask",
            "graph_operations":"embeddings→encoder→pooler→evaldropout→classifier→softmax→max",
            "graph_live_allocated_bytes":torch.cuda.memory_allocated(),
            "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated()}

    def copy_inputs(self,inputs):
        for key in ("input_ids","attention_mask"):
            if inputs[key].shape!=(1,256):
                raise RuntimeError("Graph input shape must remain B1/L256")
            self.inputs[key].copy_(inputs[key])
        if "token_type_ids" in inputs:
            self.inputs["token_type_ids"].copy_(inputs["token_type_ids"])
        else:
            self.inputs["token_type_ids"].zero_()

    def replay(self):
        self.graph.replay()
        return self.logits,self.risk


def metadata(model,modules,backend,torch):
    value = backend_record(model,"cuda",modules,torch)
    value.update(route=ROUTE,cuda_graph=True,attention_mask_policy="always additive4D; own numerical scores required",
                 graph_setup=backend.setup,final_R1_software_optimization=True)
    return value


def initial_inputs(prepared,torch):
    data = arrays(prepared,SPLITS[2])
    return batch_inputs(data,[0],"cnn","cpu",torch)


def create_backend(prepared,config,guard,torch,native_proof):
    require_bringup(native_proof,"cuda")
    load_prepared(prepared,config)
    guard.check()
    model,_,modules = load_model("cuda",torch)
    guard.check()
    backend = GraphBackend(model,initial_inputs(prepared,torch),torch)
    guard.check()
    return model,modules,backend


def bringup(folder,prepared,native_proof,config,guard,torch):
    fresh(folder)
    guard.attach(folder)
    record = {"route":ROUTE,"device":"cuda","recorded_utc":utc(),"passed":False,
        "prospective_acceptance":"exact graph-vs-eager always4D logits/risk for real short and full256 inputs, then replay after input replacement",
        "original_2d_comparison":"differences observed only; no tolerance or inherited quality approval",
        "resource_preflight":guard.preflight}
    try:
        model,modules,backend = create_backend(prepared,config,guard,torch,native_proof)
        record["backend"] = metadata(model,modules,backend,torch)
        data = arrays(prepared,SPLITS[2])
        docs = list(rows(prepared/SPLITS[2]/"documents.jsonl"))
        selected = []
        wanted = {"short":None,"full256":None}
        for index,mask in enumerate(data["masks"]):
            guard.check()
            valid = int(mask.sum())
            name = "full256" if valid==256 else "short"
            if wanted[name] is None:
                wanted[name] = index
            if all(v is not None for v in wanted.values()):
                break
        if any(v is None for v in wanted.values()):
            raise RuntimeError("Real short and fully occupied256-token development windows are both required")
        checks = []
        with torch.inference_mode():
            for name in ("short","full256","short"):
                index = wanted[name]
                guard.check()
                current = batch_inputs(data,[index],"cnn","cpu",torch)
                backend.copy_inputs(current)
                graph_logits,graph_risk = backend.replay()
                graph_logits_cpu = graph_logits.cpu().clone()
                graph_risk_value = graph_risk.item()
                explicit_logits,explicit_risk = explicit_forward(model,backend.inputs,torch)
                original_logits = model(**backend.inputs).logits.float()
                original_risk = torch.softmax(original_logits,-1)[:,1].max()
                doc_index = int(data["doc_indices"][index])
                check = {"input_case":name,"prepared_window_index":index,"document_id":docs[doc_index]["document_id"],
                    "valid_tokens":int(data["masks"][index].sum()),"graph_logits":graph_logits_cpu.tolist(),
                    "eager_always4d_logits":explicit_logits.cpu().tolist(),"original_2d_logits":original_logits.cpu().tolist(),
                    "graph_risk":graph_risk_value,"eager_always4d_risk":explicit_risk.item(),"original_2d_risk":original_risk.item(),
                    "graph_eager_logits_exact":torch.equal(graph_logits_cpu,explicit_logits.cpu()),
                    "graph_eager_risk_exact":graph_risk_value==explicit_risk.item(),
                    "graph_original_2d_logits_exact":torch.equal(graph_logits_cpu,original_logits.cpu()),
                    "graph_original_2d_risk_abs_error":abs(graph_risk_value-original_risk.item())}
                checks.append(check)
                record["checks"] = checks
                if not torch.isfinite(graph_logits_cpu).all() or not check["graph_eager_logits_exact"] or not check["graph_eager_risk_exact"]:
                    raise RuntimeError("Graph/eager always4D exact real-input check failed")
        record.update(passed=True,masked_and_unmasked_inputs_exercised=True,input_replacement_replay_checked=True)
        json_write(folder/"bringup.json",record)
    except Exception as error:
        record["failure"] = {"type":type(error).__name__,"message":str(error)}
        json_write(folder/"bringup.json",record)
        raise


def require_graph_bringup(path):
    proof = json_read(path/"bringup.json")
    if not proof["passed"] or proof["route"]!=ROUTE or proof["backend"]["checkpoint"]!=str(CHECKPOINT):
        raise RuntimeError("Passed exact CUDA Graph bringup required")
    return proof


def score(folder,prepared,native_proof,graph_proof,config,guard,torch):
    require_graph_bringup(graph_proof)
    fresh(folder)
    guard.attach(folder)
    model,modules,backend = create_backend(prepared,config,guard,torch,native_proof)
    json_write(folder/"backend.json",metadata(model,modules,backend,torch))
    outputs = {}
    for split in SPLITS[1:]:
        data = arrays(prepared,split)
        docs = list(rows(prepared/split/"documents.jsonl"))
        risks = []
        began = time.perf_counter()
        with torch.inference_mode():
            for index in range(len(data["ids"])):
                guard.check()
                backend.copy_inputs(batch_inputs(data,[index],"cnn","cpu",torch))
                _,risk = backend.replay()
                value = risk.item()
                if not 0<=value<=1:
                    raise RuntimeError("Graph risk is nonfinite or outside[0,1]")
                risks.append(value)
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
        print(json.dumps({"event":"scored","split":split,"documents":len(docs),
                                      "windows":len(risks),"seconds":time.perf_counter()-began}),flush=True)
    quality = metrics_for(outputs[SPLITS[1]],outputs[SPLITS[2]],contract(config))
    quality.update(route=ROUTE,checkpoint=str(CHECKPOINT),device="cuda",batch_size=1,complete=True,
        backend=metadata(model,modules,backend,torch),bringup=str(graph_proof/"bringup.json"))
    json_write(folder/"metrics.json",quality)
    print(json.dumps(quality["candidate_scoring"]),flush=True)


def timing(folder,quality_path,prepared,native_proof,graph_proof,config,guard,torch):
    require_graph_bringup(graph_proof)
    quality = json_read(quality_path/"metrics.json")
    if (not quality["complete"] or not quality["passes_development_screen"] or quality["route"]!=ROUTE
            or quality["device"]!="cuda" or quality["batch_size"]!=1 or quality["checkpoint"]!=str(CHECKPOINT)):
        raise RuntimeError("Graph timing requires its own full passing B1 development screen")
    fresh(folder)
    guard.attach(folder)
    model,modules,backend = create_backend(prepared,config,guard,torch,native_proof)
    tok = tokenizer(config)
    requested = [r["document_id"] for r in json_read(ROOT/"results"/"goal2"/"timing_pilot.json")["raw"]]
    if len(requested)!=32 or len(set(requested))!=32:
        raise RuntimeError("Actual matched32 timing IDs required")
    lookup = {d["document_id"]:d for d in rows(ROOT/config["paths"]["data"]/"prepared"/(SPLITS[2]+".jsonl"))
              if d["document_id"] in requested}
    chosen = [lookup[key] for key in requested]
    raw = []
    with torch.inference_mode():
        for index,doc in enumerate(chosen[:8]+chosen):
            guard.check()
            torch.cuda.synchronize()
            began = time.perf_counter()
            windows,inputs = real_inputs(tok,doc,"cnn","cpu",config,torch)
            if len(windows)!=1:
                raise RuntimeError("Matched timing input no longer single window")
            prepared_at = time.perf_counter()
            backend.copy_inputs(inputs)
            torch.cuda.synchronize()
            transferred_at = time.perf_counter()
            _,risk = backend.replay()
            torch.cuda.synchronize()
            inferred_at = time.perf_counter()
            value = risk.item()
            decision = value>quality["threshold"]
            torch.cuda.synchronize()
            ended = time.perf_counter()
            if index>=8:
                raw.append({"document_id":doc["document_id"],"risk_score":value,"attack_decision":decision,
                    "preparation_ms":(prepared_at-began)*1000,"host_to_device_ms":(transferred_at-prepared_at)*1000,
                    "inference_and_aggregation_ms":(inferred_at-transferred_at)*1000,
                    "result_transfer_and_decision_ms":(ended-inferred_at)*1000,"total_ms":(ended-began)*1000})
    means = {k:sum(r[k] for r in raw)/len(raw) for k in raw[0] if k.endswith("_ms")}
    json_write(folder/"timing.json",{"backend":metadata(model,modules,backend,torch),"batch_size":1,
        "padded_total_length":256,"warmup_documents":8,"completed_measured_checks":32,"raw":raw,"means":means,
        "threshold":quality["threshold"],"attack_rule":"score > threshold",
        "matched_input_source":"results/goal2/timing_pilot.json raw32 IDs","quality_source":str(quality_path/"metrics.json"),
        "scope":"Warm rawtext→tokenizer/padding→static-input copies→graph(mask/inference/max)→riskreturn→strictdecision; graphsetup/model load separate",
        "resource_preflight":guard.preflight})
    print(json.dumps(means),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("bringup","score","timing"),required=True)
    parser.add_argument("--prepared",default="results/goal_r1/prepared_v1")
    parser.add_argument("--output")
    parser.add_argument("--native-bringup-dir",default="results/goal_r1/bert_int8_cuda_bringup_v1")
    parser.add_argument("--bringup-dir",default="results/goal_r1/bert_cuda_graph_bringup_v1")
    parser.add_argument("--score-dir",default="results/goal_r1/bert_cuda_graph_score_v1")
    parser.add_argument("--max-seconds",type=int,default=600)
    args = parser.parse_args()
    if not 1<=args.max_seconds<=600:
        raise ValueError("Final CUDA Graph dispatch ceiling is600seconds")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    os.environ.setdefault("OMP_NUM_THREADS","4")
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    config = load_config()
    contract(config)
    guard = Guard(args.max_seconds)
    guard.check()
    torch = setup_torch("cuda")
    folder = output_path(args.output,DEST/("bert_cuda_graph_"+args.stage+"_v1"))
    prepared = output_path(args.prepared,DEST/"prepared_v1")
    native_proof = output_path(args.native_bringup_dir,DEST/"unused")
    graph_proof = output_path(args.bringup_dir,DEST/"unused")
    if args.stage=="bringup":
        bringup(folder,prepared,native_proof,config,guard,torch)
    elif args.stage=="score":
        score(folder,prepared,native_proof,graph_proof,config,guard,torch)
    else:
        timing(folder,output_path(args.score_dir,DEST/"unused"),prepared,native_proof,graph_proof,config,guard,torch)


if __name__=="__main__":
    main()
