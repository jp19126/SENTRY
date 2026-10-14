"""Phase-A analysis of existing R1 scores only; never loads a model or final data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gate_eval import evaluate_documents, fit_threshold

SPLITS = ("search_temporary_threshold", "search_candidate_scoring")
ARMS = {
    "bert_w8_graph": "bert_cuda_graph_score_v1",
    "bert_bf16_reference": "bf16_attention_score_v1",
    "cnn": "score_cnn_cuda_v1",
    "feature": "score_feature_cpu_v1",
}


def read_scores(path, inputs):
    raw = path.read_bytes()
    inputs[str(path.relative_to(ROOT))] = {
        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()
    }
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    if len({row["document_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate document IDs: {path}")
    for row in rows:
        scores = row["window_scores"]
        if not scores or len(scores) != row["window_count"]:
            raise ValueError(f"Incomplete windows: {row['document_id']}")
        if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
            raise ValueError("Invalid saved risk")
        if max(scores) != row["risk_score"]:
            raise ValueError("Saved document maximum does not match its windows")
    return rows


def percentile(values, probability):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def window_work(rows, threshold):
    strata = defaultdict(lambda: Counter(documents=0, full_windows=0, decision_windows=0))
    for row in rows:
        scores = row["window_scores"]
        # Same strict binary decision, but not the complete numeric document maximum.
        needed = next((i + 1 for i, s in enumerate(scores) if s > threshold), len(scores))
        if any(s > threshold for s in scores[:needed]) != (row["risk_score"] > threshold):
            raise ValueError("Early-positive decision mismatch")
        for key in ("all", "attack" if row["label"] else "benign"):
            strata[key].update(documents=1, full_windows=len(scores), decision_windows=needed)
    for value in strata.values():
        value["saved_windows"] = value["full_windows"] - value["decision_windows"]
        value["saved_fraction"] = value["saved_windows"] / value["full_windows"]
    return dict(strata)


def grouped_intervals(rows, threshold):
    groups = defaultdict(lambda: [0, 0, 0, 0])
    for row in rows:
        values = groups[row["source_group_id"]]
        positive = row["risk_score"] > threshold
        values[0] += row["label"] == 1 and positive
        values[1] += row["label"] == 1
        values[2] += row["label"] == 0 and positive
        values[3] += row["label"] == 0
    values = list(groups.values())
    rng = random.Random(42)
    recalls, fprs = [], []
    for _ in range(2000):
        totals = [0, 0, 0, 0]
        for _ in values:
            sampled = values[rng.randrange(len(values))]
            for i in range(4):
                totals[i] += sampled[i]
        recalls.append(totals[0] / totals[1])
        fprs.append(totals[2] / totals[3])
    # A percentile bootstrap degenerates at zero misses; report that limitation.
    n = sum(v[1] for v in values)
    successes = sum(v[0] for v in values)
    z = 1.959963984540054
    p = successes / n
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    return {
        "source_groups": len(values), "resamples": 2000, "seed": 42,
        "recall_percentile_95": [percentile(recalls, .025), percentile(recalls, .975)],
        "fpr_percentile_95": [percentile(fprs, .025), percentile(fprs, .975)],
        "attack_document_wilson_95_descriptive_only": [center - radius, min(1., center + radius)],
        "limitations": "Grouped bootstrap resamples scoring source groups at the fixed fitted threshold; calibration groups are not resampled or recalibrated. It conditions on seen templates; zero observed misses gives a degenerate bootstrap, not proof of perfect population recall. Wilson treats attacks as independent and is descriptive only, not the grouped uncertainty estimate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "results"):
        parser.error("Output must be a fresh file under repository results/")
    if output.exists():
        parser.error("Refusing to overwrite saved evidence")
    inputs, records, saved, thresholds = {}, {}, {}, {}
    for name, folder in ARMS.items():
        saved[name] = {
            split: read_scores(ROOT / "results/goal_r1" / folder / (split + "_documents.jsonl"), inputs)
            for split in SPLITS
        }
        cal, score = (saved[name][split] for split in SPLITS)
        if {r["source_group_id"] for r in cal} & {r["source_group_id"] for r in score}:
            raise ValueError("Calibration/scoring source overlap")
        threshold = thresholds[name] = fit_threshold([r["risk_score"] for r in cal if r["label"] == 0], .01)
        metrics = evaluate_documents([r["risk_score"] for r in score], [r["label"] for r in score], threshold)
        # Independent direct decision counts, rather than trusting existing metrics.json.
        counts = Counter((r["label"], r["risk_score"] > threshold) for r in score)
        if [metrics[k] for k in ("tp", "fn", "fp", "tn")] != [counts[(1, True)], counts[(1, False)], counts[(0, True)], counts[(0, False)]]:
            raise ValueError("Independent count mismatch")
        records[name] = {
            "threshold": threshold, "metrics": metrics,
            "grouped_uncertainty": grouped_intervals(score, threshold),
            "window_work": window_work(score, threshold),
            "benign_absolute_threshold_margin": {
                str(q): percentile([abs(r["risk_score"] - threshold) for r in score if not r["label"]], q)
                for q in (0, .01, .05, .5)
            },
            "strata": {
                label: evaluate_documents([r["risk_score"] for r in subset], [r["label"] for r in subset], threshold)
                for label, subset in (("single_window", [r for r in score if r["window_count"] == 1]),
                                      ("multiple_windows", [r for r in score if r["window_count"] > 1]))
            },
        }
    pairs = {}
    anchor = {r["document_id"]: r for r in saved["bert_w8_graph"][SPLITS[1]]}
    for name in ARMS:
        if name == "bert_w8_graph":
            continue
        other = {r["document_id"]: r for r in saved[name][SPLITS[1]]}
        if set(anchor) != set(other):
            raise ValueError("Arms have different document coverage")
        disagreements = []
        cells = Counter()
        for doc_id, row in anchor.items():
            rhs = other[doc_id]
            if any(row[k] != rhs[k] for k in ("label", "source_group_id", "window_count", "content_token_count")):
                raise ValueError("Arms have unmatched document metadata")
            a, b = row["risk_score"] > thresholds["bert_w8_graph"], rhs["risk_score"] > thresholds[name]
            cells[f"label{row['label']}_bert{int(a)}_other{int(b)}"] += 1
            if a != b:
                disagreements.append({"document_id": doc_id, "label": row["label"], "bert_reject": a, "other_reject": b})
        pairs[name] = {"decision_cells": dict(cells), "disagreements": disagreements}
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Saved R1 development-score analysis only; zero new inference/training/simulation. Early-positive exit is a conventional comparison opportunity, not a proposed contribution or measured speedup. Complete numeric document maxima remain the official evaluator contract.",
        "inputs": inputs, "arms": records, "paired_against_bert_w8_graph": pairs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "arms": {k: {"metrics": v["metrics"], "window_work": v["window_work"], "uncertainty": v["grouped_uncertainty"]} for k, v in records.items()}, "paired_cells": {k: v["decision_cells"] for k, v in pairs.items()}}, indent=2))


if __name__ == "__main__":
    main()
