"""The one prescribed hand-calculable confusion/threshold check (no test framework)."""
from __future__ import annotations

import json
import sys
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gate_eval import aggregate_document_risk, evaluate_documents, fit_threshold, load_config


def main() -> None:
    benign = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.9]
    attacks = [0.8, 0.81, 0.2, 0.95]
    # Each document has one low-risk window and the hand-listed maximum.
    expected_scores = benign + attacks
    raw_windows = [value for score in expected_scores for value in (0.01, score)]
    indices = [index for index in range(len(expected_scores)) for _ in range(2)]
    scores = aggregate_document_risk(raw_windows, indices, len(expected_scores))
    alpha = 0.2
    threshold = fit_threshold(scores[:len(benign)], alpha)
    result = evaluate_documents(scores, [0] * len(benign) + [1] * len(attacks), threshold)
    expected = {"threshold": 0.8, "documents": 14, "attacks": 4, "benign": 10,
                "tp": 2, "fn": 2, "fp": 1, "tn": 9, "recall": 0.5, "fpr": 0.1,
                "accuracy": 11 / 14}
    if scores != expected_scores:
        raise AssertionError("Document maximum aggregation differs from the hand calculation.")
    if any(result[key] != value for key, value in expected.items()):
        raise AssertionError({"expected": expected, "actual": result})
    k = int((Decimal(str(alpha)) * len(benign)).to_integral_value(rounding=ROUND_FLOOR))
    evidence = {
        "check": "one hand-calculable document aggregation, threshold/tie and confusion example",
        "passed": True, "benign_scores": benign, "attack_scores": attacks,
        "alpha": alpha, "k_allowed": k, "threshold_index_1based": len(benign) - k,
        "expected": expected, "actual": result,
        "explanation": "n=10, k=2, index=8 => t=0.8. Ties at 0.8 are benign; only 0.9 exceeds among benign documents. Attacks 0.81 and 0.95 exceed; 0.8 and 0.2 do not.",
    }
    config = load_config()
    output = ROOT / config["paths"]["reports"] / "evaluator_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "report": str(output), "actual": result}, indent=2))


if __name__ == "__main__":
    main()
