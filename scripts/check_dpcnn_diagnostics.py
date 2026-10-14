"""CPU-only frozen diagnostic token reconstruction and two-document replay.

No fitting, new examples, GPU execution, threshold changes or probe changes.
The absolute CUDA/CPU risk tolerance is fixed at 1e-5 before replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gate_eval import evaluate_documents, fit_threshold, load_config
from cnn_models import build_model

DEST = ROOT / "results/cnn_study_v1"
RUN_ID = "dpcnn_05_final_s42"
RISK_TOLERANCE = 1e-5


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=ROOT / "configs/cnn_stage_c_v1.json")
    parser.add_argument("--output", type=Path, default=DEST / "dpcnn_diagnostic_check.json")
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    report = {"purpose": "Distinguish diagnostic input/association faults from a recorded DPCNN diagnostic limit",
              "run_id": RUN_ID, "device": "cpu", "threads": 1, "prospective_absolute_risk_tolerance": RISK_TOLERANCE,
              "new_examples": 0, "fitting_performed": False, "gpu_execution": False,
              "threshold_or_probe_changes": False, "protected_partitions_opened": False,
              "checks": [], "source_sha256": {}, "replays": []}
    def check(name, condition, detail=None):
        report["checks"].append({"check": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise ValueError(name)
    def identity(path):
        value = digest(path)
        report["source_sha256"][str(path.relative_to(ROOT))] = value
        return value
    try:
        freeze = read(args.freeze)
        freeze_hash = digest(args.freeze)
        report["freeze_sha256"] = freeze_hash
        raw_path = DEST / "prepared/diagnostics_raw.jsonl"
        folder = DEST / "prepared/token/diagnostics"
        manifest = read(folder / "manifest.json")
        summary = read(DEST / "prepared/token/summary.json")
        protocol = read(DEST / "prepared/data_protocol.json")
        confirmation_folder = DEST / "confirmation" / RUN_ID
        confirmation = read(confirmation_folder / "complete.json")
        run_folder = DEST / "runs" / RUN_ID
        done, meta = read(run_folder / "complete.json"), read(run_folder / "run.json")
        checkpoint = ROOT / done["checkpoint"]
        spec = next(s for s in freeze["runs"] if s["run_id"] == RUN_ID)
        check("freeze association", confirmation["freeze_sha256"] == freeze_hash)
        check("model source identity", identity(ROOT / "scripts/cnn_models.py") == freeze["model_source_sha256"])
        check("frozen raw diagnostic identity", identity(raw_path) == freeze["diagnostics_sha256"])
        check("prepared protocol identity", identity(DEST / "prepared/data_protocol.json") == freeze["data_protocol_sha256"])
        check("prepared token summary identity", identity(DEST / "prepared/token/summary.json") == freeze["prepared_representation_sha256"]["token"])
        check("prepared diagnostic manifest association", manifest["artifact_sha256"] == summary["diagnostics"]["artifact_sha256"])
        check("exact training spec", meta["spec"] == spec and meta["spec"]["fit_split"] == "full_train" and not meta["spec"].get("select_epoch", False))
        check("model/config/epoch/seed association", meta["config"] == confirmation["config"] == freeze["finalist_configs"]["dpcnn_05"] and done["epochs"] == spec["epochs"] == 6 and done["seed"] == confirmation["seed"] == 42)
        checkpoint_hash = identity(checkpoint)
        check("fixed checkpoint identity", checkpoint_hash == done["checkpoint_sha256"] == confirmation["checkpoint_sha256"])
        for path in (run_folder / "run.json", run_folder / "complete.json", confirmation_folder / "complete.json", folder / "manifest.json", ROOT / "configs/project.json", args.freeze, Path(__file__).resolve()):
            identity(path)
        for name, expected in manifest["artifact_sha256"].items():
            check("prepared artifact " + name, identity(folder / name) == expected)
        raw = rows(raw_path)
        prepared = rows(folder / "documents.jsonl")
        arrays = {name: np.load(folder / (name + ".npy"), allow_pickle=False) for name in (
            "ids", "masks", "labels", "doc_indices", "starts", "content_start", "document_content_length")}
        policy = load_config()
        detector = policy["models"]["detector"]
        check("pinned current tokenizer policy", detector == protocol["tokenizer"])
        check("current window policy", policy["windowing"]["primary_length"] == protocol["token"]["length"] == 256 and policy["windowing"]["content_token_overlap"] == protocol["token"]["overlap"] == 64)
        tokenizer = AutoTokenizer.from_pretrained(detector["id"], revision=detector["tokenizer_revision"],
            cache_dir=ROOT / policy["paths"]["checkpoints"] / "huggingface", use_fast=True, local_files_only=True)
        check("BERT vocabulary and special IDs", len(tokenizer) == 30522 and tokenizer.pad_token_id == 0 and tokenizer.cls_token_id == 101 and tokenizer.sep_token_id == 102 and tokenizer.is_fast)
        report["tokenizer"] = {"id": detector["id"], "revision": detector["tokenizer_revision"], "local_files_only": True,
                               "backend_serialization_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()}
        check("all 96 document and 187 window counts", len(raw) == len(prepared) == manifest["documents"] == 96 and len(arrays["ids"]) == manifest["windows"] == 187)
        expected = {key: [] for key in arrays}
        expected["starts"] = [0]
        document_checks = []
        for i, (doc, saved_doc) in enumerate(zip(raw, prepared, strict=True)):
            check("document raw/prepared identity " + str(i), all(doc[k] == saved_doc[k] for k in ("document_id", "source_group_id", "label")) and hashlib.sha256(doc["text"].encode("utf-8")).hexdigest() == saved_doc["text_sha256"])
            encoded = tokenizer(doc["text"], add_special_tokens=False, truncation=False, return_attention_mask=False,
                                return_token_type_ids=False, return_offsets_mapping=True, verbose=False)
            ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
            start = 0; windows = 0; last_end = 0
            while True:
                end = min(start + 254, len(ids))
                values = [101] + ids[start:end] + [102]
                mask = [1] * len(values) + [0] * (256 - len(values))
                label = int(any(te > a and ts < b for ts, te in offsets[start:end]
                                for a, b in doc.get("injection_spans", []))) if doc["label"] else 0
                expected["ids"].append(values + [0] * (256 - len(values)))
                expected["masks"].append(mask)
                expected["labels"].append(label)
                expected["doc_indices"].append(i)
                expected["content_start"].append(start)
                expected["document_content_length"].append(len(ids))
                windows += 1; last_end = end
                if end == len(ids):
                    break
                start = end - 64
            expected["starts"].append(len(expected["ids"]))
            check("complete tail and metadata " + str(i), last_end == len(ids) and windows == saved_doc["window_count"] and len(ids) == saved_doc["content_length"])
            document_checks.append({"document_id": doc["document_id"], "label": doc["label"], "windows": windows,
                                    "content_tokens": len(ids), "exact_raw_prepared_identity": True})
        for key, wanted in expected.items():
            check("independent tokenizer reconstruction " + key, np.array_equal(np.asarray(wanted), arrays[key]))
        mask_lengths = arrays["masks"].sum(axis=1)
        check("nonempty binary contiguous masks", np.all(mask_lengths >= 2) and np.all((arrays["masks"] == 0) | (arrays["masks"] == 1)) and np.all(np.diff(arrays["masks"].astype(np.int16), axis=1) <= 0))
        check("padding and CLS/SEP", np.all(arrays["ids"][arrays["masks"] == 0] == 0) and np.all(arrays["ids"][:, 0] == 101) and all(arrays["ids"][i, int(length) - 1] == 102 for i, length in enumerate(mask_lengths)))
        report["input_reproduction"] = {"documents": 96, "windows": 187, "valid_mask_length_range": [int(mask_lengths.min()), int(mask_lengths.max())],
                                       "method": "Independent complete pinned tokenization, explicit CLS/SEP and manual254/64 tail-preserving slicing; not cnn_prepare/window_document reuse.", "documents_checked": document_checks}
        variant = confirmation["variants"]["fp32"]
        saved_scores = {}
        for split, hash_key in (("calibration", "calibration_score_sha256"), ("development", "development_score_sha256"), ("diagnostics", "diagnostics_score_sha256")):
            path = confirmation_folder / ("fp32_" + split + ".jsonl")
            check("saved score hash " + split, identity(path) == variant[hash_key])
            saved_scores[split] = rows(path)
            check("finite score maxima " + split, all(r["window_scores"] and r["risk_score"] == max(r["window_scores"]) and all(math.isfinite(x) and 0 <= x <= 1 for x in r["window_scores"]) for r in saved_scores[split]))
        threshold = variant["metrics"]["0.01"]["threshold"]
        check("own frozen benign-calibrated threshold", threshold == fit_threshold([r["risk_score"] for r in saved_scores["calibration"] if r["label"] == 0], .01))
        diagnostic_rows = saved_scores["diagnostics"]
        check("diagnostic score/prepared document association", len(diagnostic_rows) == 96 and all(all(r[k] == d[k] for k in ("document_id", "source_group_id", "label", "text_sha256", "window_count", "content_length")) and len(r["window_scores"]) == d["window_count"] for r, d in zip(diagnostic_rows, prepared, strict=True)))
        observed = evaluate_documents([r["risk_score"] for r in diagnostic_rows], [r["label"] for r in diagnostic_rows], threshold)
        check("saved diagnostic confusion counts", observed == variant["diagnostics"])
        report["recorded_diagnostics"] = observed
        report["saved_positive_risk_range"] = [min(r["risk_score"] for r in diagnostic_rows if r["label"]), max(r["risk_score"] for r in diagnostic_rows if r["label"])]
        check("declared B1 scoring", variant["scoring_batch_windows"] == 1)
        selected = [max((i for i, r in enumerate(diagnostic_rows) if r["label"] == label), key=lambda i: (diagnostic_rows[i]["risk_score"], diagnostic_rows[i]["document_id"])) for label in (1, 0)]
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        check("checkpoint final epoch", state["epoch"] == spec["epochs"])
        model = build_model(confirmation["config"], 30522, 0).cpu().eval()
        model.load_state_dict(state["state_dict"], strict=True)
        check("CPU model finite FP32 parameters", all(p.device.type == "cpu" and p.dtype == torch.float32 and torch.isfinite(p).all() for p in model.parameters()))
        with torch.inference_mode():
            for index in selected:
                row = diagnostic_rows[index]
                lo, hi = map(int, arrays["starts"][index:index + 2])
                cpu_risks, logits = [], []
                for window in range(lo, hi):
                    ids = torch.tensor(arrays["ids"][window:window + 1], dtype=torch.long)
                    mask = torch.tensor(arrays["masks"][window:window + 1], dtype=torch.long)
                    values = model(ids, mask).logits
                    check("finite CPU replay " + str(window), bool(torch.isfinite(values).all()))
                    cpu_risks.append(float(torch.softmax(values, dim=1)[0, 1])); logits.append(values[0].tolist())
                risk = max(cpu_risks)
                document_delta = abs(risk - row["risk_score"])
                window_deltas = [abs(a - b) for a, b in zip(cpu_risks, row["window_scores"], strict=True)]
                passed = document_delta <= RISK_TOLERANCE and all(delta <= RISK_TOLERANCE for delta in window_deltas)
                report["replays"].append({"document_id": row["document_id"], "label": row["label"], "probe_category": row["probe_category"],
                    "subset_selection": "Highest recorded CUDA risk within this label, using frozen diagnostics only", "raw_text_sha256": prepared[index]["text_sha256"],
                    "windows": hi - lo, "cuda_window_risks": row["window_scores"], "cpu_window_risks": cpu_risks, "cpu_window_logits": logits,
                    "cuda_document_risk": row["risk_score"], "cpu_document_risk": risk, "absolute_document_delta": document_delta,
                    "absolute_window_deltas": window_deltas, "passes_prospective_tolerance": passed,
                    "threshold": threshold, "cuda_decision": row["risk_score"] > threshold, "cpu_decision": risk > threshold})
        report["input_and_association_checks_passed"] = True
        report["cpu_replay_tolerance_passed"] = all(r["passes_prospective_tolerance"] for r in report["replays"])
        report["interpretation"] = "No inspected diagnostic input, mask, tail, score-maximum, checkpoint or frozen-association fault found. Two CPU replays support the recorded low scores on those documents; this bounds a diagnostic limit, not its cause or all-case CPU/CUDA equivalence." if report["cpu_replay_tolerance_passed"] else "Prepared input and recorded identity checks passed, but the fixed CPU/CUDA tolerance was exceeded; arithmetic discrepancy remains unresolved without changing tolerance."
    except Exception as error:
        report["input_and_association_checks_passed"] = False
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        report["interpretation"] = "Check incomplete or a discrepancy identified; no repair, retuning or acceptance-tolerance change performed."
    finally:
        report["torch"] = torch.__version__; report["numpy"] = np.__version__; report["python"] = sys.version
        report["checks_after_import_seconds"] = time.perf_counter() - started
        report["scope_limitations"] = ["Only two recorded diagnostic documents receive CPU network replay; all96 token inputs are reconstructed without new network scoring.",
            "Constructed detector labels do not prove target-model compromise; topic/template/wording/length confounds remain.",
            "No model, checkpoint, threshold, data, recipe or probe is modified; no new fitting, GPU, protected final partition or target API execution.",
            "No architectural causation or superiority is inferred from this implementation check."]
        temporary = args.output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)
        print(json.dumps({"output": str(args.output), "input_and_association_checks_passed": report.get("input_and_association_checks_passed"),
                          "cpu_replay_tolerance_passed": report.get("cpu_replay_tolerance_passed"), "replays": report["replays"], "error": report.get("error"),
                          "checks_after_import_seconds": report["checks_after_import_seconds"]}))
    if not report.get("input_and_association_checks_passed") or not report.get("cpu_replay_tolerance_passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
