"""CPU paired source-group analysis of frozen local/context confirmation.

Uses each architecture/seed's own benign-only calibration threshold. Positive
recall differences favor local/context; negative FPR differences favor it.
Source groups, including all document variants and fixed seed fits, are paired.
Authored diagnostics receive descriptive counts only, without population CIs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gate_eval import evaluate_documents, fit_threshold

DEST = ROOT / "results/cnn_study_v1"
SOURCE = "local_context_03"
REPLACEMENT = "local_context_03_residual_replace_c98"
IDENTITY_KEYS = ("document_id", "source_group_id", "label", "text_sha256", "window_count", "content_length")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def score_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    indexed = {}
    for row in rows:
        if row["document_id"] in indexed:
            raise ValueError(f"Duplicate document in {path}")
        if any(field not in row for field in IDENTITY_KEYS):
            raise ValueError("Required document identity fields are missing")
        if row["label"] not in (0, 1) or not isinstance(row["source_group_id"], str):
            raise ValueError("Invalid document label or source-group identity")
        scores = row["window_scores"]
        if len(scores) != row["window_count"] or not scores or any(not math.isfinite(x) or not 0 <= x <= 1 for x in scores):
            raise ValueError("Invalid or incomplete window scores")
        if row["risk_score"] != max(scores):
            raise ValueError("Document aggregation differs from window maximum")
        indexed[row["document_id"]] = row
    return indexed


def pair_rows(left, right):
    if left.keys() != right.keys():
        raise ValueError("Architectures do not have identical document memberships")
    paired = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if any(a.get(field) != b.get(field) for field in IDENTITY_KEYS):
            raise ValueError(f"Document identity differs: {key}")
        paired.append((a, b))
    return paired


def describe(paired, source_threshold, replacement_threshold):
    source = evaluate_documents([a["risk_score"] for a, _ in paired], [a["label"] for a, _ in paired], source_threshold)
    replacement = evaluate_documents([b["risk_score"] for _, b in paired], [b["label"] for _, b in paired], replacement_threshold)
    joint = {"attack": {"both_tp": 0, "source_tp_replacement_fn": 0, "source_fn_replacement_tp": 0, "both_fn": 0},
             "benign": {"both_tn": 0, "source_tn_replacement_fp": 0, "source_fp_replacement_tn": 0, "both_fp": 0}}
    disagreements = []
    groups = defaultdict(lambda: np.zeros(6, dtype=np.float64))
    for a, b in paired:
        label = a["label"]
        sp, rp = a["risk_score"] > source_threshold, b["risk_score"] > replacement_threshold
        if label:
            key = "both_tp" if sp and rp else "source_tp_replacement_fn" if sp else "source_fn_replacement_tp" if rp else "both_fn"
            joint["attack"][key] += 1
        else:
            key = "both_fp" if sp and rp else "source_fp_replacement_tn" if sp else "source_tn_replacement_fp" if rp else "both_tn"
            joint["benign"][key] += 1
        # [attack denominator, benign denominator, source TP, replacement TP,
        # source FP, replacement FP] keeps both models on the same sampled units.
        groups[a["source_group_id"]] += [label, 1 - label, label * sp, label * rp, (1 - label) * sp, (1 - label) * rp]
        if sp != rp:
            disagreements.append({**{k: a.get(k) for k in ("document_id", "source_group_id", "label", "attack_type", "attack_template_id", "injection_position", "probe_category", "probe_topic", "window_count", "content_length")},
                                  "source_prediction": int(sp), "replacement_prediction": int(rp),
                                  "source_score": a["risk_score"], "replacement_score": b["risk_score"],
                                  "source_margin": a["risk_score"] - source_threshold,
                                  "replacement_margin": b["risk_score"] - replacement_threshold})
    differences = {metric: source[metric] - replacement[metric] if source[metric] is not None else None for metric in ("recall", "fpr")}
    return {"source": source, "replacement": replacement, "differences_source_minus_replacement": differences,
            "paired_confusion": joint, "decision_disagreement_count": len(disagreements), "disagreements": disagreements}, groups


def bootstrap(groups, repeats, seed):
    array = np.stack([groups[g] for g in sorted(groups)])
    rng = np.random.default_rng(seed)
    samples = []
    invalid = 0
    for _ in range(repeats):
        attacks, benign, stp, rtp, sfp, rfp = array[rng.integers(0, len(array), len(array))].sum(0)
        if attacks == 0 or benign == 0:
            invalid += 1
            continue
        samples.append([(stp - rtp) / attacks, (sfp - rfp) / benign])
    if not samples:
        raise ValueError("No bootstrap sample contains both labels")
    quantiles = np.quantile(samples, [.025, .975], axis=0)
    return {"kind": "paired source-group percentile bootstrap", "source_groups": len(array), "repeats": repeats,
            "valid_repeats": len(samples), "discarded_single_label_repeats": invalid, "random_seed": seed,
            "recall_difference_95": quantiles[:, 0].tolist(), "fpr_difference_95": quantiles[:, 1].tolist(),
            "units": "proportions; source minus replacement", "thresholds_fixed": True,
            "calibration_uncertainty_included": False, "seed_population_uncertainty_included": False,
            "degenerate_sampled_recall_differences": bool(np.ptp(np.asarray(samples)[:, 0]) == 0),
            "degenerate_sampled_fpr_differences": bool(np.ptp(np.asarray(samples)[:, 1]) == 0)}


def load_run(spec, freeze, freeze_hash, variant):
    folder = DEST / "confirmation" / spec["run_id"]
    result = read(folder / "complete.json")
    if not result["complete"] or result["freeze_sha256"] != freeze_hash or result["config"]["id"] != spec["config_id"] or result["seed"] != spec["seed"]:
        raise ValueError("Confirmation does not match frozen run specification")
    if result["config"] != freeze["finalist_configs"][spec["config_id"]]:
        raise ValueError("Configuration differs from actual freeze")
    run = DEST / "runs" / spec["run_id"]
    meta, done = read(run / "run.json"), read(run / "complete.json")
    if meta["spec"] != spec:
        raise ValueError("Learning recipe differs from frozen specification")
    if meta["spec"]["fit_split"] != "full_train" or meta["spec"].get("select_epoch", False) or done["epochs"] != spec["epochs"]:
        raise ValueError("Pair requires fixed-epoch, full-training frozen fits")
    if digest(ROOT / done["checkpoint"]) != result["checkpoint_sha256"] or done["checkpoint_sha256"] != result["checkpoint_sha256"]:
        raise ValueError("Confirmation checkpoint identity changed")
    saved = result["variants"][variant]
    splits = {}
    identities = {}
    for split, key in (("calibration", "calibration_score_sha256"), ("development", "development_score_sha256"), ("diagnostics", "diagnostics_score_sha256")):
        path = folder / f"{variant}_{split}.jsonl"
        actual = digest(path)
        if actual != saved[key]:
            raise ValueError("Saved raw-score identity changed")
        splits[split] = score_rows(path)
        identities[split] = {"path": str(path.relative_to(ROOT)), "sha256": actual}
    threshold = fit_threshold([r["risk_score"] for r in splits["calibration"].values() if r["label"] == 0], .01)
    if threshold != saved["metrics"]["0.01"]["threshold"]:
        raise ValueError("Saved primary threshold differs from benign-only calibration rule")
    measured = evaluate_documents([r["risk_score"] for r in splits["development"].values()], [r["label"] for r in splits["development"].values()], threshold)
    if any(saved["metrics"]["0.01"][k] != v for k, v in measured.items()):
        raise ValueError("Saved primary metrics differ from raw decisions")
    return splits, threshold, {"run_id": spec["run_id"], "seed": spec["seed"], "checkpoint_sha256": result["checkpoint_sha256"],
                              "threshold": threshold, "score_identities": identities, "training_spec": meta["spec"], "cost": result["cost"],
                              "training_metadata": {key: meta.get(key) for key in ("loss", "batch_documents", "microbatch_windows", "arithmetic")},
                              "config_training_defaults": result["config"]["training"]}


def self_test():
    source, replacement = {}, {}
    definitions = [("g0", 0, .5, .8), ("g0", 1, .8, .2), ("g0", 1, .8, .8),
                   ("g1", 0, .8, .2), ("g1", 1, .2, .8), ("g1", 1, .8, .8)]
    for i, (group, label, left, right) in enumerate(definitions):
        identity = {"document_id": str(i), "source_group_id": group, "label": label, "text_sha256": str(i), "window_count": 1, "content_length": 1}
        source[str(i)] = {**identity, "risk_score": left, "window_scores": [left]}
        replacement[str(i)] = {**identity, "risk_score": right, "window_scores": [right]}
    measured, groups = describe(pair_rows(source, replacement), .5, .5)
    assert measured["source"]["tp"] == measured["replacement"]["tp"] == 3
    assert measured["source"]["fp"] == measured["replacement"]["fp"] == 1
    assert measured["decision_disagreement_count"] == 4
    assert measured["paired_confusion"]["attack"]["source_tp_replacement_fn"] == 1
    assert measured["paired_confusion"]["benign"]["source_tn_replacement_fp"] == 1
    interval = bootstrap(groups, 256, 42)
    assert interval["recall_difference_95"] == [-.5, .5]
    assert interval["fpr_difference_95"] == [-1., 1.]
    replacement["0"]["text_sha256"] = "changed"
    try:
        pair_rows(source, replacement)
    except ValueError:
        pass
    else:
        raise AssertionError("Identity mismatch not rejected")
    return {"passed": True, "purpose": "hand-computable paired-count, strict-threshold, group-bootstrap and identity-rejection synthetic checks", "bootstrap": interval}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--variant", default="fp32", choices=("fp32", "fake_w8a8"))
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=424242)
    parser.add_argument("--output", type=Path, default=DEST / "local_context_pair_analysis.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        write(DEST / "pair_analysis_self_check.json", result)
        print(json.dumps(result)); return
    if args.freeze is None or not 100 <= args.repeats <= 10000:
        parser.error("--freeze is required and bootstrap repeats must be 100..10000")
    started = time.perf_counter()
    freeze, freeze_hash = read(args.freeze), digest(args.freeze)
    by_config = {config: {} for config in (SOURCE, REPLACEMENT)}
    for spec in freeze["runs"]:
        if spec["config_id"] in by_config:
            if spec["seed"] in by_config[spec["config_id"]]:
                raise ValueError("Duplicate architecture/seed in freeze")
            by_config[spec["config_id"]][spec["seed"]] = spec
    expected = sorted(set(by_config[SOURCE]) | set(by_config[REPLACEMENT]))
    if not expected:
        raise ValueError("Actual freeze contains neither comparison architecture")
    seeds, pending, group_sets, document_identity = [], [], [], None
    for seed in expected:
        specs = [by_config[config].get(seed) for config in (SOURCE, REPLACEMENT)]
        unavailable = [config for config, spec in zip((SOURCE, REPLACEMENT), specs) if spec is None or not (DEST / "confirmation" / spec["run_id"] / "complete.json").exists()]
        if unavailable:
            pending.append({"seed": seed, "unavailable_configs": unavailable}); continue
        variant_missing = [config for config, spec in zip((SOURCE, REPLACEMENT), specs)
                           if args.variant not in read(DEST / "confirmation" / spec["run_id"] / "complete.json")["variants"]]
        if variant_missing:
            pending.append({"seed": seed, "variant_unavailable_configs": variant_missing}); continue
        loaded = [load_run(spec, freeze, freeze_hash, args.variant) for spec in specs]
        source, replacement = loaded
        pair_rows(source[0]["calibration"], replacement[0]["calibration"])
        pairs = pair_rows(source[0]["development"], replacement[0]["development"])
        identity = [tuple(a.get(k) for k in IDENTITY_KEYS) for a, _ in pairs]
        if document_identity is not None and document_identity != identity:
            raise ValueError("Development document identities differ across matched seeds")
        document_identity = identity
        counts, groups = describe(pairs, source[1], replacement[1])
        group_sets.append(groups)
        diagnostics = pair_rows(source[0]["diagnostics"], replacement[0]["diagnostics"])
        descriptive, _ = describe(diagnostics, source[1], replacement[1])
        strata = {}
        for field in ("probe_category", "probe_topic"):
            for value in sorted({str(a.get(field)) for a, _ in diagnostics}):
                selected = [(a, b) for a, b in diagnostics if str(a.get(field)) == value]
                strata[f"{field}:{value}"], _ = describe(selected, source[1], replacement[1])
        recipe_differences = {}
        for field in ("seed", "epochs", "fit_split", "cal_split", "selection_split", "select_epoch", "learning_rate"):
            left, right = source[2]["training_spec"].get(field), replacement[2]["training_spec"].get(field)
            if left != right:
                recipe_differences[field] = {"source": left, "replacement": right}
        for field in ("training_metadata", "config_training_defaults"):
            if source[2][field] != replacement[2][field]:
                recipe_differences[field] = {"source": source[2][field], "replacement": replacement[2][field]}
        seeds.append({"seed": seed, "source_identity": source[2], "replacement_identity": replacement[2],
                      "training_recipe_differences": recipe_differences,
                      "development": counts, "paired_uncertainty": bootstrap(groups, args.repeats, args.bootstrap_seed),
                      "authored_probes_descriptive": {"overall": descriptive, "strata": strata, "bootstrap_performed": False}})
    pooled = None
    if group_sets:
        if any(set(groups) != set(group_sets[0]) for groups in group_sets):
            raise ValueError("Source groups differ across matched seeds")
        mean_counts = {g: np.mean([groups[g] for groups in group_sets], axis=0) for g in group_sets[0]}
        pooled = {"fixed_seed_count": len(seeds), "differences_mean_over_fixed_matched_seeds": {
            metric: float(np.mean([s["development"]["differences_source_minus_replacement"][metric] for s in seeds])) for metric in ("recall", "fpr")},
            "seed_difference_ranges": {metric: [min(s["development"]["differences_source_minus_replacement"][metric] for s in seeds), max(s["development"]["differences_source_minus_replacement"][metric] for s in seeds)] for metric in ("recall", "fpr")},
            "paired_uncertainty": bootstrap(mean_counts, args.repeats, args.bootstrap_seed),
            "resampling": "One source-group draw shared across architectures and all available fixed seeds; seeds are not independent document replicates."}
    review_path = DEST / "ablations/local_context_03/capacity_review.json"
    report = {"complete": not pending and len(seeds) == len(expected), "source_config_id": SOURCE, "replacement_config_id": REPLACEMENT,
              "expected_seeds": expected, "matched_completed_seeds": [s["seed"] for s in seeds], "pending": pending,
              "variant": args.variant, "freeze_path": str(args.freeze), "freeze_sha256": freeze_hash,
              "difference_direction": "local/context minus ordinary residual replacement; positive recall and negative FPR favor local/context",
              "operating_point": "alpha=0.01, each model/seed's own designated benign-only calibration; strict score > threshold",
              "per_seed": seeds, "mean_fixed_seed_comparison": pooled,
              "scientific_confounds": read(review_path) if review_path.exists() else None,
              "limitations": ["Historically exposed post-search development data, not independent final evaluation or unseen attack-family evidence.",
                  "Source-group bootstrap conditions on fitted thresholds and fixed trained models; calibration, architecture selection and population seed uncertainty are excluded.",
                  "Repeated attack templates and semantic dependencies across source groups are unresolved; the bootstrap assumes independent original-source groups.",
                  "Degenerate zero-discordance or all-correct bootstrap tails are not proof of equal population performance or zero unseen errors.",
                  "Authored probes are few constructed examples with known concepts; only descriptive paired counts are reported, without target-model compromise or population generalization claims.",
                  "This replacement changes skip placement, normalization inputs, dropout placement/count and width; a difference cannot be uniquely attributed to concatenation/fusion.",
                  "A missing seed remains pending and is not a zero result; an available-seed mean is provisional until every prespecified matched seed completes.",
                  "No new training, threshold optimization on scoring labels, hypothesis change or protected-partition access occurs."],
              "helper_source_sha256": digest(Path(__file__)), "numpy": np.__version__, "wall_seconds": time.perf_counter() - started}
    write(args.output, report)
    print(json.dumps({"complete": report["complete"], "matched_seeds": report["matched_completed_seeds"], "pending": pending, "output": str(args.output)}))


if __name__ == "__main__":
    main()
