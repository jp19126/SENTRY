"""CPU-only verification of the actual timing helper's raw input preparation.

Extracts the timing function's initialization and nested prepare() from its AST,
without running model inference, CUDA operations, warmups or timing loops.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gate_eval import evaluate_documents, load_config, window_document
from r1_challengers import tokenizer, rows, json_write
from cnn_prepare import digest


def extracted_helpers(source):
    tree = ast.parse(source)
    timing = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "timing")
    initialize = []
    for node in timing.body:
        initialize.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "prepare": break
    else: raise ValueError("Timing function no longer contains the expected nested prepare helper")
    prepare = initialize[-1]
    fields = {n.slice.value for n in ast.walk(prepare) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "d" and isinstance(n.slice, ast.Constant)}
    if fields != {"text", "document_id"}: raise ValueError(f"Timing preparation reads unexpected metadata: {fields}")
    # Allow CUDA calls only in the excluded network/timing section.
    if any(isinstance(n, ast.Attribute) and n.attr in ("cuda", "synchronize", "to") for node in initialize for n in ast.walk(node)):
        raise ValueError("Extracted initialization unexpectedly accesses device operations")
    strata = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "strata")
    return ast.Module(body=initialize, type_ignores=[]), ast.Module(body=[strata], type_ignores=[]), ast.get_source_segment(source, prepare)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/cnn_study_v1/timing_input_check.json")
    args = parser.parse_args()
    started = time.monotonic()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    evaluator = ROOT / "scripts/cnn_evaluate.py"
    source_bytes = evaluator.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    init_ast, strata_ast, prepare_source = extracted_helpers(source_bytes.decode("utf-8"))
    raw_path = ROOT / "data/prepared/search_candidate_scoring.jsonl"
    raw = list(rows(raw_path))
    clean = sorted((d for d in raw if d["label"] == 0), key=lambda d:(len(d["text"].encode("utf-8")), d["document_id"]))
    oracle_ids = [d["document_id"] for d in clean[:6]+clean[-6:]]
    result = {}
    strata_namespace = {"evaluate_documents":evaluate_documents}
    exec(compile(strata_ast,str(evaluator),"exec"),strata_namespace)
    for representation in ("token", "byte"):
        env = {"ROOT":ROOT,"json":json,"count":12,"c":{"representation":representation},"tokenizer":tokenizer,"load_config":load_config,"window_document":window_document,"torch":torch}
        exec(compile(init_ast,str(evaluator),"exec"),env)
        selected = env["selected"]
        assert [d["document_id"] for d in selected] == oracle_ids
        folder = ROOT / "results/cnn_study_v1/prepared" / representation / "search_candidate_scoring"
        docs = list(rows(folder / "documents.jsonl"))
        positions = {d["document_id"]:i for i,d in enumerate(docs)}
        arrays = {k:np.load(folder/(k+".npy"),mmap_mode="r",allow_pickle=False) for k in ("ids","masks","starts")}
        checked, stratum_rows = [], []
        for order,d in enumerate(selected):
            index = positions[d["document_id"]]; meta = docs[index]
            begin,end = map(int,arrays["starts"][index:index+2])
            pairs = env["prepare"](d)
            assert len(pairs) == end-begin == meta["window_count"]
            for local,(ids,mask) in enumerate(pairs):
                assert ids.device.type == mask.device.type == "cpu"
                assert ids.shape == mask.shape == (1,256 if representation == "token" else 1024)
                assert np.array_equal(ids.numpy()[0],arrays["ids"][begin+local]), f"Timing/prepared IDs differ: {representation} {d['document_id']} {local}"
                assert np.array_equal(mask.numpy()[0],arrays["masks"][begin+local]), f"Timing/prepared masks differ: {representation} {d['document_id']} {local}"
            utf8 = d["text"].encode("utf-8")
            assert meta["raw_utf8_bytes"] == len(utf8)
            assert meta["text_sha256"] == hashlib.sha256(utf8).hexdigest()
            if representation == "token":
                units = len(env["tok"](d["text"],add_special_tokens=False,verbose=False)["input_ids"])
                assert meta["content_token_count"] == meta["content_length"] == units
                unit_kind = "BERT content tokens"
            else:
                units = len(utf8); assert meta["content_length"] == units
                assert "content_token_count" not in meta, "Byte strata would incorrectly use a token count"
                unit_kind = "UTF-8 bytes"
            checked.append({"document_id":d["document_id"],"length_selection":"six_shortest" if order<6 else "six_longest","utf8_bytes":len(utf8),"content_units":units,"unit_kind":unit_kind,"windows":len(pairs),"ids_equal":True,"masks_equal":True})
            stratum_rows.append({**meta,"risk_score":.25,"window_scores":[.25]*len(pairs)})
        strata = strata_namespace["strata"](stratum_rows,.5)
        for prefix,value_key,bins in (("content_units:","content_units",((0,128),(128,256),(256,512),(512,10**9))),("windows:","windows",((1,2),(2,4),(4,8),(8,10**9)))):
            for low,high in bins:
                count = sum(low<=r[value_key]<high for r in checked)
                assert strata[f"{prefix}{low}-{high-1}"]["documents"] == count, "Representation length strata differ"
        result[representation] = {"checked_documents":12,"checked_windows":sum(r["windows"] for r in checked),"samples":checked,"length_strata_verified":True,"prepared_manifest_sha256":digest(folder/"manifest.json")}
    latest_data_check = json.loads((ROOT/"results/cnn_study_v1/prepared/check_token_byte_v2.json").read_text(encoding="utf-8"))
    assert latest_data_check["passed"] and not any(latest_data_check["diagnostic_whitespace_normalized_exact_text_overlap"].values())
    assert all(not any(v.values()) for v in latest_data_check["diagnostic_overlap_audit"].values())
    if evaluator.read_bytes() != source_bytes: raise RuntimeError("Evaluator changed during check; rerun against the current source")
    report = {"passed":True,"question":"Does the actual raw-document timing preparation process the same complete inputs as frozen scoring arrays?","evaluator_source_sha256":source_sha,"checker_source_sha256":digest(__file__),"timing_prepare_source":prepare_source,"canonical_development_source_sha256":digest(raw_path),"cpu_only":True,"gpu_calls":0,"model_inference_calls":0,"closed_partitions_opened":False,"same_short_long_document_ids_across_representations":True,"selection_rule":"Six shortest and six longest benign development documents ordered by UTF-8 byte length then document ID","representations":result,"diagnostic_overlap":latest_data_check["diagnostic_overlap_audit"],"diagnostic_whitespace_normalized_exact_text_overlap":latest_data_check["diagnostic_whitespace_normalized_exact_text_overlap"],"evidence_boundary":"Document-max aggregation is the common scoring baseline. Zero source/exact-text overlap supports authored diagnostic separation; these historically exposed development/timing documents and common attack concepts do not establish independent final or unseen-family evidence. Token and byte content-unit bins use different units and window context extents.","wall_seconds":time.monotonic()-started,"python":sys.version,"torch":torch.__version__,"numpy":np.__version__}
    output = (ROOT/args.output).resolve()
    if not output.is_relative_to((ROOT/"results/cnn_study_v1").resolve()) or output.exists(): raise ValueError("Use a fresh output under the CNN study results")
    json_write(output,report)
    print(json.dumps({"passed":True,"report":str(output),"wall_seconds":report["wall_seconds"],"representations":{k:{kk:vv for kk,vv in v.items() if kk != "samples"} for k,v in result.items()}},indent=2))


if __name__ == "__main__": main()
