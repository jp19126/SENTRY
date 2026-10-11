"""Diagnostic CPU timing only: native W8 INT8 full CPU quality is unresolved.

Uses the historical W8-QAT threshold solely to include strict decision cost.
This diagnostic cannot select CPU as a protection-matched quality winner and
does not weaken the full-quality timing gate in r1_bert_int8.py.
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
from r1_challengers import (DEST,Guard,contract,fresh,json_read,json_write,
    output_path,real_inputs,rows,setup_torch,tokenizer)
from r1_bert_int8 import backend_record,load_model,require_bringup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="results/goal_r1/bert_int8_cpu_diagnostic_timing_v1")
    parser.add_argument("--bringup-dir",default="results/goal_r1/bert_int8_cpu_bringup_v1")
    parser.add_argument("--max-seconds",type=int,default=180)
    args = parser.parse_args()
    if not 1<=args.max_seconds<=180:
        raise ValueError("Diagnostic CPU ceiling is180seconds")
    os.environ.setdefault("OMP_NUM_THREADS","4")
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    config = load_config()
    contract(config)
    guard = Guard(args.max_seconds)
    guard.check()
    proof = output_path(args.bringup_dir,DEST/"unused")
    require_bringup(proof,"cpu")
    folder = output_path(args.output,DEST/"unused")
    fresh(folder)
    guard.attach(folder)
    threshold_source = ROOT/"results"/"goal3"/"qat_w8_a8"/"candidate.json"
    historical = json_read(threshold_source)
    if not historical["identity"]["qat"] or set(historical["identity"]["precision_map"].values())!={8}:
        raise RuntimeError("Historical W8QAT diagnostic threshold identity mismatch")
    threshold = historical["metrics"]["primary"]["threshold"]
    torch = setup_torch("cpu")
    model,_,modules = load_model("cpu",torch)
    tok = tokenizer(config)
    requested = [r["document_id"] for r in json_read(ROOT/"results"/"goal2"/"timing_pilot.json")["raw"]]
    if len(requested)!=32 or len(set(requested))!=32:
        raise RuntimeError("Actual old matched32 timing IDs required")
    lookup = {d["document_id"]:d for d in rows(ROOT/config["paths"]["data"]/"prepared"/"search_candidate_scoring.jsonl")
              if d["document_id"] in requested}
    chosen = [lookup[key] for key in requested]
    raw = []
    record = {"diagnostic_only":True,"protection_matched":False,"CPU_full_scores_incomplete":True,
        "can_select_as_quality_winner":False,"threshold":threshold,"attack_rule":"score > threshold",
        "threshold_source":str(threshold_source),"threshold_status":"historical W8QAT value; not a completed native CPU calibration",
        "backend":backend_record(model,"cpu",modules,torch),"bringup_source":str(proof/"bringup.json"),
        "batch_size":1,"padded_total_length":256,"warmup_documents":8,
        "matched_input_source":"results/goal2/timing_pilot.json raw32 IDs",
        "scope":"Warm complete CPU rawtext→tokenizer/padding→nativeINT8 inference/max→riskreturn→strictdecision; model loading separate",
        "resource_preflight":guard.preflight,"complete":False}
    json_write(folder/"timing.json",record)
    try:
        with torch.inference_mode():
            for index,doc in enumerate(chosen[:8]+chosen):
                guard.check()
                began = time.perf_counter()
                windows,inputs = real_inputs(tok,doc,"cnn","cpu",config,torch)
                if len(windows)!=1:
                    raise RuntimeError("Matched CPU diagnostic input no longer single window")
                prepared_at = time.perf_counter()
                inputs = {k:v.to("cpu") for k,v in inputs.items()}
                transferred_at = time.perf_counter()
                risk = torch.softmax(model(**inputs).logits.float(),-1)[:,1].max()
                inferred_at = time.perf_counter()
                value = risk.item()
                if not 0<=value<=1:
                    raise RuntimeError("Diagnostic risk nonfinite or outside[0,1]")
                decision = value>threshold
                ended = time.perf_counter()
                if index>=8:
                    raw.append({"document_id":doc["document_id"],"risk_score":value,"attack_decision":decision,
                        "preparation_ms":(prepared_at-began)*1000,"host_to_device_ms":(transferred_at-prepared_at)*1000,
                        "inference_and_aggregation_ms":(inferred_at-transferred_at)*1000,
                        "result_transfer_and_decision_ms":(ended-inferred_at)*1000,"total_ms":(ended-began)*1000})
                print(json.dumps({"event":"diagnostic_check","check":index+1,"warmup":index<8}),flush=True)
        record.update(raw=raw,completed_measured_checks=len(raw),complete=True,
            means={k:sum(r[k] for r in raw)/len(raw) for k in raw[0] if k.endswith("_ms")})
        json_write(folder/"timing.json",record)
        print(json.dumps(record["means"]),flush=True)
    except Exception as error:
        record.update(raw=raw,completed_measured_checks=len(raw),failure={"type":type(error).__name__,"message":str(error)})
        json_write(folder/"timing.json",record)
        raise


if __name__=="__main__":
    main()
