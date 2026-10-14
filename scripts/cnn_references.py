"""Reuse saved R1/Goal2 BERT and R1 CNN scores on canonical search data.

This is a CPU-only report reducer: it performs no model loading or inference.
Each reference fits its own benign-only threshold for each alpha, then reports
candidate-set metrics, strata and source-group uncertainty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

PROCESS_STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gate_eval import evaluate_documents, fit_threshold  # noqa: E402
from cnn_evaluate import strata, uncertainty  # noqa: E402


ALPHAS = (0.005, 0.01, 0.02)
DEFAULT_OUTPUT = ROOT / "results/cnn_study_v1/reused_references.json"
CANONICAL = {
    "search_temporary_threshold": ROOT / "data/prepared/search_temporary_threshold.jsonl",
    "search_candidate_scoring": ROOT / "data/prepared/search_candidate_scoring.jsonl",
}
R1_PREPARED = {
    "search_temporary_threshold": ROOT / "results/goal_r1/prepared_v1/search_temporary_threshold/documents.jsonl",
    "search_candidate_scoring": ROOT / "results/goal_r1/prepared_v1/search_candidate_scoring/documents.jsonl",
}
REFERENCES = {
    "r1_bert_w8_qat_cuda_graph": {
        "calibration": ROOT / "results/goal_r1/bert_cuda_graph_score_v1/search_temporary_threshold_documents.jsonl",
        "candidate": ROOT / "results/goal_r1/bert_cuda_graph_score_v1/search_candidate_scoring_documents.jsonl",
        "metadata": [
            ROOT / "results/goal_r1/bert_cuda_graph_score_v1/metrics.json",
            ROOT / "results/goal_r1/bert_cuda_graph_score_v1/backend.json",
            ROOT / "results/goal_r1/anchor_audit.json",
        ],
        "checkpoint": "checkpoints/quantized/qat_w8_a8",
        "training": "Previously trained BERT W8-QAT checkpoint; R1 adds no learning.",
        "pretraining": "google/bert_uncased_L-4_H-256_A-4 pretrained checkpoint; QAT checkpoint derived from it.",
        "scoring_backend": (
            "RTX 4070 CUDA Graph, B1/L256, torch._int_mm signed W8/A8 INT32 encoder linears "
            "with FP32 scale/rescale/bias; FP32 embeddings, SDPA attention, residuals, LayerNorm, "
            "GELU, pooler, classifier and softmax. Not ordered C, RTL or all-integer inference."
        ),
        "training_advantage": "Pretrained 4-layer BERT initialization; this R1 route performs no training.",
        "historical_exposure": "Its training data included the same 55 candidate attack templates across 11 attack families; not an unseen-template test.",
        "matched_timing": False,
        "timing_caveat": "R1 timing is not a matched comparison to the current CNN study's software timing boundary or hardware path.",
        "anchor_audit": "results/goal_r1/anchor_audit.json#bert_w8_qat",
    },
    "r1_scratch_textcnn": {
        "calibration": ROOT / "results/goal_r1/score_cnn_cuda_v1/search_temporary_threshold_documents.jsonl",
        "candidate": ROOT / "results/goal_r1/score_cnn_cuda_v1/search_candidate_scoring_documents.jsonl",
        "metadata": [
            ROOT / "results/goal_r1/score_cnn_cuda_v1/metrics.json",
            ROOT / "results/goal_r1/cnn_initial_v1/run.json",
            ROOT / "results/goal_r1/cnn_initial_v1/complete.json",
        ],
        "checkpoint": "results/goal_r1/cnn_initial_v1/checkpoint_latest.pt",
        "training": "Scratch FP32 TextCNN, seed 42, fixed 10 epochs AdamW at 0.001, weight decay 0.01; final epoch, no score-based selection.",
        "pretraining": "None; 30,522 by 64 embedding table learned from scratch.",
        "scoring_backend": "RTX 4070 CUDA, B1/L256, Torch 2.9.1+cu128 standard FP32 kernels; TF32 disabled; not an accelerated INT benchmark.",
        "training_advantage": "Small model and fixed short recipe; lacks BERT's pretrained initialization.",
        "historical_exposure": "Its training data included the same 55 candidate attack templates across 11 attack families; not an unseen-template test.",
        "matched_timing": False,
        "timing_caveat": "R1 CNN timing is not a matched comparison to the current CNN study's software timing boundary or hardware path.",
        "anchor_audit": "results/goal_r1/score_cnn_cuda_v1/metrics.json",
    },
    "goal2_fp32_bert_epoch3": {
        "calibration": ROOT / "results/goal2/epoch_3/search_temporary_threshold_documents.jsonl",
        "candidate": ROOT / "results/goal2/epoch_3/search_candidate_scoring_documents.jsonl",
        "metadata": [
            ROOT / "results/goal2/epoch_3/metrics.json",
            ROOT / "results/goal_r1/anchor_audit.json",
        ],
        "checkpoint": "checkpoints/floating/epoch_3",
        "training": "Previously fine-tuned FP32 BERT epoch 3; final epoch selected without search-score selection.",
        "pretraining": "google/bert_uncased_L-4_H-256_A-4 pretrained checkpoint.",
        "scoring_backend": "Historical native Torch CUDA FP32/SDPA, B1/L256; standard FP32 arithmetic.",
        "training_advantage": "Pretrained 4-layer BERT initialization.",
        "historical_exposure": "Its training data included the same 55 candidate attack templates across 11 attack families; not an unseen-template test.",
        "matched_timing": False,
        "timing_caveat": "Historical epoch-3 inference is not timed on the current CNN study's matched end-to-end software boundary.",
        "anchor_audit": "results/goal_r1/anchor_audit.json#bert_floating",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object rows at {path}:{number}")
            rows.append(row)
    return rows


def row_key(row: dict[str, Any]) -> tuple[str, int, str]:
    try:
        return (str(row["document_id"]), int(row["label"]), str(row["source_group_id"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Every row must contain document_id, binary label, and source_group_id") from exc


def merge_scores(
    canonical_rows: list[dict[str, Any]], prepared_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]], split: str, reference: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(canonical_rows) != len(prepared_rows) or len(canonical_rows) != len(score_rows):
        raise ValueError(f"{reference} {split}: score row count differs from canonical manifest")
    merged = []
    optional_text_hashes = True
    for index, (canonical, prepared, score) in enumerate(zip(canonical_rows, prepared_rows, score_rows)):
        if row_key(canonical) != row_key(prepared) or row_key(canonical) != row_key(score):
            raise ValueError(f"{reference} {split}: ordered identity mismatch at row {index}")
        for field in ("source_dataset", "attack_type", "attack_template_id", "injection_position"):
            if field in canonical and score.get(field) != canonical.get(field):
                raise ValueError(f"{reference} {split}: canonical {field} mismatch at row {index}")
        if "text_sha256" in canonical and "text_sha256" in score:
            if canonical["text_sha256"] != score["text_sha256"]:
                raise ValueError(f"{reference} {split}: text_sha256 mismatch at row {index}")
        else:
            optional_text_hashes = False
        risk = float(score["risk_score"])
        if not math.isfinite(risk):
            raise ValueError(f"{reference} {split}: nonfinite risk at row {index}")
        windows = score.get("window_scores")
        if not isinstance(windows, list) or not windows or not all(math.isfinite(float(x)) for x in windows):
            raise ValueError(f"{reference} {split}: missing/invalid window scores at row {index}")
        if "window_count" in score and int(score["window_count"]) != len(windows):
            raise ValueError(f"{reference} {split}: window_count differs from window_scores at row {index}")
        if "window_count" in canonical and int(canonical["window_count"]) != len(windows):
            raise ValueError(f"{reference} {split}: saved window count differs from canonical metadata at row {index}")
        if int(prepared["window_count"]) != len(windows):
            raise ValueError(f"{reference} {split}: window scores differ from R1 prepared manifest at row {index}")
        for field in ("content_token_count", "is_long_document", "window_count"):
            if field in score and score[field] != prepared.get(field):
                raise ValueError(f"{reference} {split}: prepared {field} mismatch at row {index}")
        # Canonical manifests supply labels and all strata; reference scores supply only risks.
        merged_row = dict(canonical)
        # R1 output metadata supplies derived token-length/window strata that
        # are not serialized in the canonical raw-text manifest.
        for field in ("window_count", "is_long_document", "content_token_count", "content_length"):
            if field in score:
                merged_row[field] = score[field]
        merged.append({**merged_row, "risk_score": risk, "window_scores": [float(x) for x in windows]})
    return merged, {
        "rows": len(merged),
        "ordered_document_id_label_source_group_match": True,
        "text_sha256_present_and_equal_for_all_rows": optional_text_hashes,
        "window_scores_match_r1_prepared_manifest": True,
    }


def summarize_reference(name: str, spec: dict[str, Any], cal: list[dict[str, Any]],
                        candidate: list[dict[str, Any]], input_hashes: dict[str, str]) -> dict[str, Any]:
    outputs = {}
    for alpha in ALPHAS:
        benign = [float(row["risk_score"]) for row in cal if int(row["label"]) == 0]
        threshold = fit_threshold(benign, alpha)
        overall = evaluate_documents(
            [row["risk_score"] for row in candidate],
            [row["label"] for row in candidate], threshold,
        )
        outputs[f"{alpha:.3f}"] = {
            "alpha": alpha,
            "threshold": threshold,
            "threshold_fit": {
                "partition": "search_temporary_threshold",
                "benign_documents_only": len(benign),
                "method": "gate_eval.fit_threshold; ascending benign score at n-floor(alpha*n), ties benign",
                "attack_rule": "score > threshold",
            },
            "candidate_scoring": overall,
            "strata": strata(candidate, threshold),
            "source_group_uncertainty": uncertainty(candidate, threshold, seed=424242, repeats=1000),
        }
    return {
        "training_and_pretraining": {
            key: spec[key] for key in ("checkpoint", "training", "pretraining", "training_advantage", "historical_exposure")
        },
        "scoring_backend": spec["scoring_backend"],
        "timing": {"matched_to_current_study": spec["matched_timing"], "caveat": spec["timing_caveat"]},
        "preserved_prior_evidence": spec["anchor_audit"],
        "inputs": {
            "canonical_calibration_manifest": "data/prepared/search_temporary_threshold.jsonl",
            "canonical_candidate_manifest": "data/prepared/search_candidate_scoring.jsonl",
            "calibration_scores": str(spec["calibration"].relative_to(ROOT)).replace("\\", "/"),
            "candidate_scores": str(spec["candidate"].relative_to(ROOT)).replace("\\", "/"),
            "sha256": input_hashes,
        },
        "partition_identity": {
            "search_temporary_threshold": cal_identity,
            "search_candidate_scoring": candidate_identity,
            "text_identity_limitation": (
                "Canonical input contains source text, but saved reference score rows contain no text_sha256; "
                "identity is established by exact ordered document_id/label/source_group_id, window counts, "
                "and the pre-existing anchor audit, not by a score-row text hash."
            ),
        },
        "operating_points": outputs,
    }


def main() -> int:
    global cal_identity, candidate_identity
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_repeats != 1000:
        raise ValueError("This report schema fixes bootstrap repetitions at 1000")

    started = PROCESS_STARTED
    canonical_cal = read_jsonl(CANONICAL["search_temporary_threshold"])
    canonical_candidate = read_jsonl(CANONICAL["search_candidate_scoring"])
    prepared_cal = read_jsonl(R1_PREPARED["search_temporary_threshold"])
    prepared_candidate = read_jsonl(R1_PREPARED["search_candidate_scoring"])
    cal_identity = {"documents": len(canonical_cal), "benign": sum(int(r["label"]) == 0 for r in canonical_cal),
                    "attacks": sum(int(r["label"]) == 1 for r in canonical_cal)}
    candidate_identity = {"documents": len(canonical_candidate), "benign": sum(int(r["label"]) == 0 for r in canonical_candidate),
                          "attacks": sum(int(r["label"]) == 1 for r in canonical_candidate)}
    if cal_identity != {"documents": 4065, "benign": 1355, "attacks": 2710}:
        raise ValueError(f"Unexpected canonical threshold manifest: {cal_identity}")
    if candidate_identity != {"documents": 4107, "benign": 1369, "attacks": 2738}:
        raise ValueError(f"Unexpected canonical candidate manifest: {candidate_identity}")
    canonical_hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
        for path in [*CANONICAL.values(), *R1_PREPARED.values()]
    }

    results = {}
    identity_checks = {}
    for name, spec in REFERENCES.items():
        cal_scores = read_jsonl(spec["calibration"])
        candidate_scores = read_jsonl(spec["candidate"])
        cal, cal_check = merge_scores(canonical_cal, prepared_cal, cal_scores, "search_temporary_threshold", name)
        candidate, candidate_check = merge_scores(canonical_candidate, prepared_candidate, candidate_scores, "search_candidate_scoring", name)
        identity_checks[name] = {
            "search_temporary_threshold": cal_check,
            "search_candidate_scoring": candidate_check,
        }
        hash_paths = [spec["calibration"], spec["candidate"], *spec["metadata"]]
        hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in hash_paths}
        results[name] = summarize_reference(name, spec, cal, candidate, hashes)

    text_hash_fields = {
        "canonical_calibration_manifest_has_text_sha256": all("text_sha256" in row for row in canonical_cal),
        "canonical_candidate_manifest_has_text_sha256": all("text_sha256" in row for row in canonical_candidate),
        "all_reference_score_rows_have_text_sha256": all(
            all("text_sha256" in row for row in read_jsonl(spec["candidate"]))
            and all("text_sha256" in row for row in read_jsonl(spec["calibration"]))
            for spec in REFERENCES.values()
        ),
    }
    wall_seconds = time.perf_counter() - started
    report = {
        "schema_version": "cnn-reused-references-v1",
        "purpose": "Saved-score reference comparison on canonical development search partitions; no inference or learning.",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "execution": {
            "command": "python scripts/cnn_references.py",
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "device": "CPU only; no model/GPU initialization or inference",
            "wall_seconds_processing_before_serialization": wall_seconds,
            "source_group_bootstrap": {"seed": 424242, "repeats": 1000, "threshold_uncertainty_included": False},
        },
        "protocol": {
            "canonical_inputs": {
                "search_temporary_threshold": "data/prepared/search_temporary_threshold.jsonl",
                "search_candidate_scoring": "data/prepared/search_candidate_scoring.jsonl",
            "R1_prepared_threshold_manifest": str(R1_PREPARED["search_temporary_threshold"].relative_to(ROOT)).replace("\\", "/"),
            "R1_prepared_candidate_manifest": str(R1_PREPARED["search_candidate_scoring"].relative_to(ROOT)).replace("\\", "/"),
                "sha256": canonical_hashes,
            },
            "identity_key": ["document_id", "label", "source_group_id"],
            "merge": "Exact ordered identity match required; canonical manifest supplies strata metadata, saved model scores supply risk_score/window_scores.",
            "threshold_fit": "Each model/alpha independently fits benign-only temporary-threshold risk scores using gate_eval.fit_threshold.",
            "candidate_rule": "Strict risk_score > model-specific threshold; no candidate scores used to fit threshold.",
            "alphas": list(ALPHAS),
            "text_hash_availability": text_hash_fields,
            "text_identity_limitation": "No score file contains per-document text_sha256, so exact text-byte equivalence cannot be re-established from score artifacts; existing R1 anchor audit is retained and exact ordered document/label/group/window identity is checked.",
            "protected_partitions_opened": False,
            "timing_claim": "None; historical route timings are not matched to the current CNN study's end-to-end timing boundary.",
        },
        "identity_checks": identity_checks,
        "references": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    temp_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp_path, args.output)
    report["execution"]["wall_seconds_including_first_serialization"] = time.perf_counter() - started
    temp_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp_path, args.output)
    print(json.dumps({"output": str(args.output.relative_to(ROOT)), "references": list(results),
                      "wall_seconds": report["execution"]["wall_seconds_including_first_serialization"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
