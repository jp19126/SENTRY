"""Goal R1 native signed INT8 encoder linear kernels, distinct from RTL reference.

Prequantized frozen W8 weights and runtime fixed A8 codes feed torch._int_mm.
Unsupported kernels fail explicitly; no floating emulation fallback is permitted.
Only train-independent development partitions are read. No learning occurs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gate_eval import load_config
from r1_challengers import (DEST, SPLITS, Guard, arrays, batch_inputs, contract,
    execution_settings, fresh, json_read, json_write, load_prepared, metrics_for,
    output_path, real_inputs, rows, setup_torch, tokenizer, utc)

CHECKPOINT = ROOT/"checkpoints"/"quantized"/"qat_w8_a8"
SELECTED = ("bert.encoder.layer.0.attention.self.query", "bert.encoder.layer.0.output.dense")
ROUTE = "r1_native_sdpa_frozen_w8_fixed_a8_torch_int_mm_v1"


def convert(model, torch):
    from gate_quant import GateQuantLinear, encoder_linears
    preparation_started = time.perf_counter()

    class FrozenInt8Linear(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            if original.weight_bits != 8:
                raise RuntimeError("This route only accepts uniform W8")
            self.in_features = original.in_features
            self.out_features = original.out_features
            scale = original.weight_scale_8.detach()
            codes = torch.round(original.weight.detach()/scale).clamp(-128,127).to(torch.int8)
            reference = original.quantizer.to_int(scale, original.zero_point, original.weight_width, original.weight).to(torch.int8)
            if not torch.equal(codes, reference):
                raise RuntimeError("Frozen weight codes differ from loaded Brevitas codes")
            self.register_buffer("weight_t", codes.T.contiguous())
            self.register_buffer("row_scale", scale.squeeze(1).clone())
            self.register_buffer("input_scale", original.input_scale.detach().clone())
            self.register_buffer("bias", original.bias.detach().clone() if original.bias is not None else None)

        def forward(self, x):
            if x.dtype != torch.float32 or torch.is_autocast_enabled(x.device.type):
                raise RuntimeError("Native route requires FP32 activation input without autocast")
            codes = torch.round(x/self.input_scale).clamp(-128,127).to(torch.int8)
            accumulator = torch._int_mm(codes.reshape(-1,self.in_features).contiguous(), self.weight_t)
            if accumulator.dtype != torch.int32:
                raise RuntimeError("Native kernel must return INT32")
            result = accumulator.float()*(self.input_scale*self.row_scale)
            if self.bias is not None:
                result = result+self.bias
            return result.reshape(*x.shape[:-1], self.out_features)

    linears = list(encoder_linears(model))
    if len(linears) != 24 or any(not isinstance(module, GateQuantLinear) for _,_,module in linears):
        raise RuntimeError("Expected all24 saved quantized encoder linears")
    for _,path,module in linears:
        parent, name = path.rsplit(".",1)
        setattr(model.get_submodule(parent),name,FrozenInt8Linear(module))
    model._r1_weight_preparation_seconds = time.perf_counter()-preparation_started
    return model, [{"path":path,"K":module.in_features,"N":module.out_features,
                    "weight_bits":8,"activation_bits":8} for _,path,module in linears]


def load_model(device, torch, original=False):
    loading_started = time.perf_counter()
    from gate_quant import configure_exact_fp32, load_quantized_checkpoint
    configure_exact_fp32()
    if not hasattr(torch,"_int_mm"):
        raise RuntimeError("Installed Torch has no _int_mm; native INT8 unavailable")
    model, metadata = load_quantized_checkpoint(CHECKPOINT,device)
    if set(metadata["precision_map"].values()) != {8}:
        raise RuntimeError("Uniform saved W8QAT checkpoint required")
    if model.config._attn_implementation != "sdpa":
        raise RuntimeError("R1 native software route requires actual SDPA attention")
    model.eval()
    model._r1_checkpoint_loading_seconds = time.perf_counter()-loading_started
    if original:
        return model,metadata
    native, modules = convert(model,torch)
    native.eval()
    return native,metadata,modules


def backend_record(model, device, modules, torch):
    return {"route":ROUTE,"checkpoint":str(CHECKPOINT),"device":device,
        "device_name":torch.cuda.get_device_name() if device=="cuda" else "CPU",
        "attention_implementation":model.config._attn_implementation,
        "kernel":"torch._int_mm (internal experimental Torch operator)",
        "linear_arithmetic":"signed W8/A8 INT32 dot; original FP32 scale product/rescale then bias",
        "weight_preparation":"loaded frozen scales; W8 codes and contiguous transposed weights precomputed before timing",
        "other_operations":"native FP32 embeddings/SDPA attention/residual/LayerNorm/GELU/pooler/classifier/softmax",
        "execution_settings":execution_settings(torch),"modules":modules,
        "checkpoint_loading_seconds":model._r1_checkpoint_loading_seconds,
        "frozen_weight_preparation_seconds":model._r1_weight_preparation_seconds,
        "torch_compile_used":False,
        "not_rtl_ordered_reference":True,"physical_measurement":False,"learning_runs":0}


def bringup(device, folder, config, guard, torch):
    fresh(folder)
    guard.attach(folder)
    report = {"route":ROUTE,"device":device,"recorded_utc":utc(),"passed":False,
              "acceptance":"exact accumulator, selected same-input linear outputs, full logits and risks",
              "resource_preflight":guard.preflight}
    try:
        original, metadata = load_model(device,torch,original=True)
        native, modules = convert(deepcopy(original),torch)
        native.eval()
        report["backend"] = backend_record(native,device,modules,torch)
        # Signed extremes specifically catch saturating/unsigned reinterpretation kernels.
        a = torch.full((32,1024),-128,dtype=torch.int8,device=device)
        b = torch.full((1024,32),-128,dtype=torch.int8,device=device)
        a[1::2,:] = 127
        a[2,1::2] = 127
        b[:,1::2] = 127
        b[1::2,2] = 127
        guard.check()
        boundary_actual = torch._int_mm(a.contiguous(),b.contiguous())
        boundary_expected = a.cpu().to(torch.int64)@b.cpu().to(torch.int64)
        report["signed_extreme_K1024"] = {"shape_A":list(a.shape),"shape_B":list(b.shape),
            "values":[-128,127],"exact":torch.equal(boundary_actual.cpu().to(torch.int64),boundary_expected),
            "maximum_abs_accumulator":int(boundary_expected.abs().max())}
        if not report["signed_extreme_K1024"]["exact"]:
            raise RuntimeError("Native signedINT8 kernel failed exact K1024 extreme accumulator check")
        tok = tokenizer(config)
        doc = next(rows(ROOT/config["paths"]["data"]/"prepared"/"search_candidate_scoring.jsonl"))
        windows, inputs = real_inputs(tok,doc,"cnn",device,config,torch)
        captures = {}
        hooks = []
        for path in SELECTED:
            def capture(module,args,output,path=path):
                captures[path] = {"input":args[0].detach().clone(),"original_output":output.detach().clone()}
            hooks.append(original.get_submodule(path).register_forward_hook(capture))
        original_logits, native_logits, original_risks, native_risks = [],[],[],[]
        selected_results = []
        with torch.inference_mode():
            for window_index in range(len(windows)):
                guard.check()
                current = {k:v[window_index:window_index+1].to(device) for k,v in inputs.items()}
                original_logit = original(**current).logits.float()
                native_logit = native(**current).logits.float()
                original_logits.extend(original_logit.cpu().tolist())
                native_logits.extend(native_logit.cpu().tolist())
                original_risks.extend(torch.softmax(original_logit,-1)[:,1].cpu().tolist())
                native_risks.extend(torch.softmax(native_logit,-1)[:,1].cpu().tolist())
                for path in SELECTED:
                    old = original.get_submodule(path)
                    new = native.get_submodule(path)
                    same_input = captures[path]["input"]
                    codes = torch.round(same_input/new.input_scale).clamp(-128,127).to(torch.int8)
                    old_codes = old.quantizer.to_int(old.input_scale,old.zero_point,old.activation_width,same_input).to(torch.int8)
                    actual = torch._int_mm(codes.reshape(-1,new.in_features).contiguous(),new.weight_t)
                    expected = codes.cpu().reshape(-1,new.in_features).to(torch.int64)@new.weight_t.cpu().to(torch.int64)
                    native_same_input = new(same_input)
                    same_output = captures[path]["original_output"]
                    check = {"path":path,"window_index":window_index,
                        "codes_exact":torch.equal(codes,old_codes),
                        "accumulator_exact":torch.equal(actual.cpu().to(torch.int64),expected),
                        "rescaled_output_exact":torch.equal(native_same_input,same_output)}
                    selected_results.append(check)
                    torch.save({"input_FP32":same_input.cpu(),"input_codes_A8":codes.cpu(),
                        "weight_t_W8":new.weight_t.cpu(),"accumulator_native_INT32":actual.cpu(),
                        "accumulator_reference_INT64":expected,"original_output_FP32":same_output.cpu(),
                        "native_same_input_output_FP32":native_same_input.cpu()},folder/(path.replace(".","_")+f"_w{window_index}.pt"))
                    if not all(check[k] for k in ("codes_exact","accumulator_exact","rescaled_output_exact")):
                        raise RuntimeError("Selected same-input linear exact check failed: "+json.dumps(check))
        for hook in hooks:
            hook.remove()
        report.update(document_id=doc["document_id"],windows=len(windows),selected_checks=selected_results,
            original_logits=original_logits,native_logits=native_logits,original_window_risks=original_risks,
            native_window_risks=native_risks,original_document_risk=max(original_risks),native_document_risk=max(native_risks),
            full_logits_exact=original_logits==native_logits,full_window_risks_exact=original_risks==native_risks)
        if not report["full_logits_exact"] or not report["full_window_risks_exact"]:
            raise RuntimeError("Full real-input original/native numerical check was not exact")
        report["passed"] = True
        json_write(folder/"bringup.json",report)
    except Exception as error:
        report["failure"] = {"type":type(error).__name__,"message":str(error)}
        json_write(folder/"bringup.json",report)
        raise


def require_bringup(path,device):
    proof = json_read(path/"bringup.json")
    if not proof["passed"] or proof["route"]!=ROUTE or proof["device"]!=device or proof["backend"]["checkpoint"]!=str(CHECKPOINT):
        raise RuntimeError("This exact backend needs its own passed native bringup")
    return proof


def score(device,folder,prepared,proof_path,config,guard,torch):
    proof = require_bringup(proof_path,device)
    load_prepared(prepared,config)
    model,metadata,modules = load_model(device,torch)
    fresh(folder)
    guard.attach(folder)
    json_write(folder/"backend.json",backend_record(model,device,modules,torch))
    outputs = {}
    for split in SPLITS[1:]:
        data = arrays(prepared,split)
        documents = list(rows(prepared/split/"documents.jsonl"))
        risks = []
        started = time.perf_counter()
        with torch.inference_mode():
            for index in range(len(data["ids"])):
                guard.check()
                logits = model(**batch_inputs(data,[index],"cnn",device,torch)).logits.float()
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Nonfinite nativeINT8 scoring logits")
                risks.append(torch.softmax(logits,-1)[0,1].item())
                if (index+1)%1000==0:
                    print(json.dumps({"event":"scoring","device":device,"split":split,"windows":index+1}),flush=True)
        with (folder/(split+"_windows.jsonl")).open("x",encoding="utf-8") as handle:
            for index,risk in enumerate(risks):
                doc_index = int(data["doc_indices"][index])
                handle.write(json.dumps({"document_id":documents[doc_index]["document_id"],
                    "window_index":index-int(data["starts"][doc_index]),"risk_score":risk})+"\n")
        for index,doc in enumerate(documents):
            first,end = data["starts"][index:index+2]
            doc.update(risk_score=max(risks[first:end]),window_scores=risks[first:end])
        with (folder/(split+"_documents.jsonl")).open("x",encoding="utf-8") as handle:
            for doc in documents:
                handle.write(json.dumps(doc,ensure_ascii=False)+"\n")
        outputs[split]=documents
        print(json.dumps({"event":"scored","device":device,"split":split,"documents":len(documents),
            "windows":len(risks),"seconds":time.perf_counter()-started}),flush=True)
    quality = metrics_for(outputs[SPLITS[1]],outputs[SPLITS[2]],contract(config))
    quality.update(route=ROUTE,checkpoint=str(CHECKPOINT),device=device,batch_size=1,complete=True,
        backend=backend_record(model,device,modules,torch),bringup=str(proof_path/"bringup.json"))
    json_write(folder/"metrics.json",quality)
    print(json.dumps(quality["candidate_scoring"]),flush=True)


def timing(device,folder,quality_path,proof_path,config,guard,torch):
    require_bringup(proof_path,device)
    quality = json_read(quality_path/"metrics.json")
    if (not quality["complete"] or not quality["passes_development_screen"] or quality["route"]!=ROUTE
            or quality["device"]!=device or quality["batch_size"]!=1 or quality["checkpoint"]!=str(CHECKPOINT)):
        raise RuntimeError("Timing requires this exact nativeINT8 backend's full passing B1 screen")
    model,metadata,modules = load_model(device,torch)
    tok = tokenizer(config)
    requested = [r["document_id"] for r in json_read(ROOT/"results"/"goal2"/"timing_pilot.json")["raw"]]
    if len(requested)!=32 or len(set(requested))!=32:
        raise RuntimeError("Actual matched32 timing IDs required")
    lookup = {d["document_id"]:d for d in rows(ROOT/config["paths"]["data"]/"prepared"/(SPLITS[2]+".jsonl"))
              if d["document_id"] in requested}
    chosen = [lookup[key] for key in requested]
    fresh(folder)
    guard.attach(folder)
    raw = []
    def synchronize():
        if device=="cuda":
            torch.cuda.synchronize()
    with torch.inference_mode():
        for index,doc in enumerate(chosen[:8]+chosen):
            guard.check()
            synchronize()
            began = time.perf_counter()
            windows,inputs = real_inputs(tok,doc,"cnn",device,config,torch)
            if len(windows)!=1:
                raise RuntimeError("Matched timing set no longer single window")
            preparation_end = time.perf_counter()
            inputs = {k:v.to(device) for k,v in inputs.items()}
            synchronize()
            transfer_end = time.perf_counter()
            risk = torch.softmax(model(**inputs).logits.float(),-1)[:,1].max()
            synchronize()
            inference_end = time.perf_counter()
            value = risk.item()
            decision = value>quality["threshold"]
            synchronize()
            end = time.perf_counter()
            if index>=8:
                raw.append({"document_id":doc["document_id"],"risk_score":value,"attack_decision":decision,
                    "preparation_ms":(preparation_end-began)*1000,"host_to_device_ms":(transfer_end-preparation_end)*1000,
                    "inference_and_aggregation_ms":(inference_end-transfer_end)*1000,
                    "result_transfer_and_decision_ms":(end-inference_end)*1000,"total_ms":(end-began)*1000})
    means = {k:sum(r[k] for r in raw)/len(raw) for k in raw[0] if k.endswith("_ms")}
    json_write(folder/"timing.json",{"backend":backend_record(model,device,modules,torch),
        "batch_size":1,"padded_total_length":256,"warmup_documents":8,"completed_measured_checks":32,
        "threshold":quality["threshold"],"attack_rule":"score > threshold","raw":raw,"means":means,
        "matched_input_source":"results/goal2/timing_pilot.json raw32 IDs",
        "scope":"Warm full raw-text to strict document decision; weights frozen before timing; no loading cost or physical measurements",
        "quality_source":str(quality_path/"metrics.json"),"resource_preflight":guard.preflight})
    print(json.dumps(means),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("bringup","score","timing"),required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),required=True)
    parser.add_argument("--prepared",default="results/goal_r1/prepared_v1")
    parser.add_argument("--output")
    parser.add_argument("--bringup-dir")
    parser.add_argument("--score-dir")
    parser.add_argument("--max-seconds",type=int,default=600)
    args = parser.parse_args()
    if not 1<=args.max_seconds<=600:
        raise ValueError("NativeINT8 dispatch ceiling is600seconds")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    os.environ.setdefault("OMP_NUM_THREADS","4")
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    config = load_config()
    contract(config)
    guard = Guard(args.max_seconds)
    guard.check()
    torch = setup_torch(args.device)
    folder = output_path(args.output,DEST/("bert_int8_"+args.device+"_"+args.stage+"_v1"))
    proof = output_path(args.bringup_dir,DEST/("bert_int8_"+args.device+"_bringup_v1"))
    prepared = output_path(args.prepared,DEST/"prepared_v1")
    if args.stage=="bringup":
        bringup(args.device,folder,config,guard,torch)
    elif args.stage=="score":
        score(args.device,folder,prepared,proof,config,guard,torch)
    else:
        quality_path = output_path(args.score_dir,DEST/("bert_int8_"+args.device+"_score_v1"))
        timing(args.device,folder,quality_path,proof,config,guard,torch)


if __name__=="__main__":
    main()
