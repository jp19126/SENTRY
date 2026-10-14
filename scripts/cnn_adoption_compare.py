#!/usr/bin/env python3
"""Compare saved CNN/BERT evaluator scores without running inference.

Rows are paired only when complete evaluator associations and exact document,
text, window-input, role, label, and group identities agree. Confirmation score
rows are not decoded until every pair has passed the on-disk freeze and target
authorization checks. Bootstrap intervals are sensitivity summaries for the
fixed observed groups and seeds, not population guarantees or noninferiority
tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import gate_eval
PAIR_FIELDS = (
    "document_id", "text_sha256", "window_input_sha256", "label",
    "window_count", "source_group_id", "payload_group_id",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    raise ValueError(message)


def _pair_seed_agrees(pair: dict[str, Any], cnn: dict[str, Any], bert: dict[str, Any]) -> None:
    seed = pair.get("seed")
    if not isinstance(seed, int) or seed < 0:
        fail("Each pair needs a nonnegative integer seed label.")
    # Run checkpoint paths commonly encode the optimizer seed. Enforce it when
    # present; historical fixed references may have no run seed in their path.
    for side, complete in (("CNN", cnn), ("BERT", bert)):
        target = complete["association"]["target"]
        checkpoint_path = str(target.get("checkpoint_identity", {}).get("path", ""))
        match = re.search(r"(?:^|[_-])s(\d+)(?:[_./-]|$)", checkpoint_path)
        if match and int(match.group(1)) != seed:
            fail(f"{side} checkpoint path seed {match.group(1)} differs from pair seed {seed}.")


def _validate_freeze(plan: dict[str, Any], root: Path) -> tuple[dict[str, Any], str]:
    freeze_name = plan.get("confirmation_freeze")
    expected_hash = plan.get("confirmation_freeze_sha256")
    if not freeze_name or not expected_hash:
        fail("Confirmation comparison is unauthorized: plan has no complete freeze path and SHA-256.")
    freeze_path = Path(freeze_name)
    if not freeze_path.is_absolute():
        freeze_path = root / freeze_path
    if not freeze_path.is_file():
        fail("Confirmation freeze path does not resolve to a file.")
    actual = sha256(freeze_path)
    if actual != expected_hash:
        fail("Confirmation freeze file hash differs from the plan.")
    frozen = read_json(freeze_path)
    required_identities = (
        "nomination", "data_membership", "calibration_policy", "numerical_formats",
        "priority_order", "execution_policy",
    )
    for key in required_identities:
        if identity(frozen.get(key)) != frozen.get(key + "_sha256"):
            fail(f"Frozen {key} hash does not verify.")
    if not frozen.get("complete") or not frozen.get("nomination_before_confirmation"):
        fail("Confirmation requires a complete prospective nomination freeze.")
    if frozen.get("calibration_policy") != plan.get("calibration_policy"):
        fail("Frozen and planned calibration policies differ.")
    if frozen.get("quality_gates") != plan.get("quality_gates"):
        fail("Frozen and planned quality gates differ.")
    if identity(frozen.get("quality_gates")) != frozen.get("quality_gates_sha256"):
        fail("Frozen quality-gate identity does not verify.")
    if frozen.get("priority_order") != plan.get("priority_order"):
        fail("Frozen and planned priority order differ.")
    if frozen.get("prepared_root") != plan.get("prepared_root"):
        fail("Frozen and planned prepared-data root differs.")
    if frozen.get("execution_policy") != plan.get("execution_policy"):
        fail("Frozen and planned execution policy differs.")
    for filename, digest in frozen.get("source_sha256", {}).items():
        source = root / filename
        if not source.is_file() or sha256(source) != digest:
            fail(f"A frozen evaluator source changed or is missing: {filename}")
    if set(frozen.get("data_membership", {})) != {"benign_calibration", "confirmation"}:
        fail("Freeze does not identify both reserved evaluation roles.")
    return frozen, actual


def _expected_roles(stage: str) -> tuple[str, str]:
    if stage == "development":
        return "development_calibration", "development_selection"
    if stage == "confirmation":
        return "benign_calibration", "confirmation"
    fail(f"Unsupported evaluation stage: {stage}")


def _validate_scoring_dependencies(association: dict[str, Any], target: dict[str, Any],
                                  freeze: dict[str, Any] | None) -> None:
    sources = association.get("source_sha256", {})
    required = {
        "scripts/cnn_adoption_evaluate.py", "scripts/cnn_adoption_train.py",
        "scripts/cnn_models.py", "gate_eval.py",
    }
    fmt = target.get("numerical_format")
    kind = target.get("model_spec", {}).get("kind", "")
    if fmt == "w8a8_export":
        required |= {"scripts/cnn_adoption_export.py", "scripts/cnn_adoption_backend.py"}
    if kind == "historical_bert_w8":
        required |= {"gate_quant.py", "scripts/r1_bert_int8.py"}
    if not required.issubset(sources):
        fail("Evaluator association lacks required scoring dependencies: " + ", ".join(sorted(required - set(sources))))
    if freeze is not None:
        frozen_sources = freeze.get("source_sha256", {})
        for name, digest in sources.items():
            if frozen_sources.get(name) != digest:
                fail(f"Recorded evaluator scoring dependency differs from confirmation freeze: {name}")
        if not required.issubset(frozen_sources):
            fail("Confirmation freeze omits required scoring dependencies: " + ", ".join(sorted(required - set(frozen_sources))))


def _validate_pair_targets(pair: dict[str, Any], cnn: dict[str, Any], bert: dict[str, Any]) -> tuple[str, str]:
    ca, ba = cnn["association"], bert["association"]
    ct, bt = ca["target"], ba["target"]
    if ct.get("model_spec", {}).get("kind") != "cnn":
        fail("CNN side is not an associated CNN target.")
    if "bert" not in bt.get("model_spec", {}).get("kind", "").lower():
        fail("BERT side is not an associated BERT target.")
    cnn_format, bert_format = ct.get("numerical_format"), bt.get("numerical_format")
    if pair.get("cnn_format") and pair["cnn_format"] != cnn_format:
        fail("Pair manifest CNN format differs from nominated evaluator target.")
    if pair.get("bert_format") and pair["bert_format"] != bert_format:
        fail("Pair manifest BERT format differs from nominated evaluator target.")
    if ca["calibration_policy"] != ba["calibration_policy"]:
        fail("Paired evaluator calibration policies differ.")
    if ca["quality_gates"] != ba["quality_gates"]:
        fail("Paired evaluator quality gates differ.")
    for role in (cnn["calibration_role"], cnn["selection_role"]):
        if ca["data"][role] != ba["data"][role]:
            fail(f"Paired evaluator data role identities differ: {role}")
    for source in ("scripts/cnn_adoption_evaluate.py", "gate_eval.py"):
        if ca["source_sha256"].get(source) != ba["source_sha256"].get(source):
            fail(f"Paired evaluators used different {source} sources.")
    return cnn_format, bert_format


def _preflight_eval(path: Path, *, expected_stage: str, freeze: dict[str, Any] | None,
                    freeze_sha: str | None, plan: dict[str, Any], root: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "complete.json"
    if not path.is_file():
        fail(f"Missing evaluator complete.json: {path}")
    complete = read_json(path)
    if complete.get("complete") is not True:
        fail(f"Evaluator artifact is not complete: {path}")
    association = complete.get("association", {})
    if association.get("stage") != expected_stage or (complete.get("stage") is not None and complete.get("stage") != expected_stage):
        fail(f"Evaluator stage differs from requested comparison: {path}")
    target = association.get("target")
    if not isinstance(target, dict):
        fail(f"Missing target association: {path}")
    if not association.get("data") or not association.get("source_sha256"):
        fail(f"Missing data/source associations: {path}")
    if association.get("calibration_policy") != plan.get("calibration_policy"):
        fail(f"Evaluator calibration policy differs from plan: {path}")
    policy = association["calibration_policy"]
    if policy.get("procedure") != "gate_eval.fit_threshold" or policy.get("rule") != "score > threshold":
        fail(f"Evaluator does not use the frozen strict threshold procedure: {path}")
    if association.get("quality_gates") != plan.get("quality_gates"):
        fail(f"Evaluator quality gates differ from plan: {path}")
    calibration_role, selection_role = _expected_roles(expected_stage)
    expected_data_roles = {calibration_role, selection_role}
    if not expected_data_roles.issubset(set(association["data"])):
        fail(f"Evaluator lacks required role identities: {path}")
    if expected_stage == "confirmation":
        if freeze is None or freeze_sha is None:
            fail("Confirmation freeze must be verified before any score rows are read.")
        target_key = complete.get("target_key")
        if freeze.get("nomination", {}).get("targets", {}).get(target_key) != target:
            fail(f"Evaluator target was not exactly nominated in the freeze: {path}")
        if freeze.get("numerical_formats", {}).get(target_key) != target.get("numerical_format"):
            fail(f"Evaluator numerical format was not nominated: {path}")
        if association.get("confirmation_freeze_sha256") != freeze_sha:
            fail(f"Evaluator association names a different freeze: {path}")
        membership_roles = {"benign_calibration": calibration_role, "confirmation": selection_role}
        for freeze_role, association_role in membership_roles.items():
            expected = freeze["data_membership"][freeze_role]
            actual_data = association["data"][association_role]
            if actual_data.get("files") != expected.get("artifact_sha256"):
                fail(f"Frozen membership differs from evaluator association for {freeze_role}: {path}")
    _validate_scoring_dependencies(association, target, freeze)
    files = complete.get("score_files_sha256")
    if not isinstance(files, dict):
        fail(f"Missing score-file hashes: {path}")
    role_files = {
        calibration_role: calibration_role + "_scores.jsonl",
        selection_role: selection_role + "_scores.jsonl",
    }
    for role, filename in role_files.items():
        score_path = path.parent / filename
        if filename not in files or not score_path.is_file() or sha256(score_path) != files[filename]:
            fail(f"Score artifact is missing or its saved SHA-256 differs: {score_path}")
    return {"path": path, "complete": complete, "association": association,
            "calibration_role": calibration_role, "selection_role": selection_role,
            "role_files": role_files}


def preflight(pairs: list[dict[str, Any]], plan: dict[str, Any], stage: str, root: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str | None]:
    freeze = None
    freeze_sha = None
    if stage == "confirmation":
        freeze, freeze_sha = _validate_freeze(plan, root)
    checked = []
    seen_seeds = set()
    for pair in pairs:
        if not isinstance(pair, dict) or "cnn" not in pair or "bert" not in pair:
            fail("Each pair requires cnn and bert evaluator paths.")
        cnn_path = Path(pair["cnn"])
        bert_path = Path(pair["bert"])
        if not cnn_path.is_absolute(): cnn_path = root / cnn_path
        if not bert_path.is_absolute(): bert_path = root / bert_path
        cnn = _preflight_eval(cnn_path, expected_stage=stage, freeze=freeze, freeze_sha=freeze_sha, plan=plan, root=root)
        bert = _preflight_eval(bert_path, expected_stage=stage, freeze=freeze, freeze_sha=freeze_sha, plan=plan, root=root)
        _pair_seed_agrees(pair, cnn["complete"], bert["complete"])
        if pair["seed"] in seen_seeds:
            fail(f"Duplicate seed pair: {pair['seed']}")
        seen_seeds.add(pair["seed"])
        cnn_format, bert_format = _validate_pair_targets(pair, cnn, bert)
        checked.append({"seed": pair["seed"], "cnn_format": cnn_format,
                        "bert_format": bert_format, "cnn": cnn, "bert": bert})
    if not checked:
        fail("At least one CNN/BERT pair is required.")
    return checked, freeze, freeze_sha


def _read_rows(evaluation: dict[str, Any], role: str) -> list[dict[str, Any]]:
    path = evaluation["path"].parent / evaluation["role_files"][role]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        fail(f"No score rows in {path}")
    seen: set[str] = set()
    for row in rows:
        doc_id = str(row.get("document_id", ""))
        if not doc_id or doc_id in seen:
            fail(f"Missing/duplicate document ID in {path}")
        seen.add(doc_id)
        score = row.get("risk_score")
        windows = row.get("window_scores")
        if (not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1
                or not isinstance(windows, list) or not windows or len(windows) != row.get("window_count")):
            fail(f"Invalid risk/window scores in {path}: {doc_id}")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1 for value in windows):
            fail(f"Invalid window probability in {path}: {doc_id}")
        if max(windows) != score:
            fail(f"Document score is not the maximum saved window score: {path}: {doc_id}")
        if row.get("label") not in (0, 1):
            fail(f"Invalid document label in {path}: {doc_id}")
        if not row.get("text_sha256") or not row.get("window_input_sha256") or not row.get("source_group_id"):
            fail(f"Missing text/window/source identity in {path}: {doc_id}")
    return rows


def _pair_role(cnn_rows: list[dict[str, Any]], bert_rows: list[dict[str, Any]], context: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    cdict = {str(row["document_id"]): row for row in cnn_rows}
    bdict = {str(row["document_id"]): row for row in bert_rows}
    if cdict.keys() != bdict.keys():
        fail(f"Document ID sets differ in paired {context} rows.")
    paired = []
    for doc_id in sorted(cdict):
        c, b = cdict[doc_id], bdict[doc_id]
        for field in PAIR_FIELDS:
            if c.get(field) != b.get(field):
                fail(f"Exact paired identity differs for {field} in {context}: {doc_id}")
        if c["window_count"] != len(c["window_scores"]) or b["window_count"] != len(b["window_scores"]):
            fail(f"Window count differs from scores in {context}: {doc_id}")
        paired.append((c, b))
    return paired


def _metrics(rows: list[tuple[dict[str, Any], dict[str, Any]]], threshold_cnn: float, threshold_bert: float,
             indices: list[int] | None = None) -> dict[str, Any]:
    selected = [rows[i] for i in (range(len(rows)) if indices is None else indices)]
    if not selected:
        return {"documents": 0, "recall_cnn": None, "recall_bert": None, "delta_recall_cnn_minus_bert": None,
                "fpr_cnn": None, "fpr_bert": None, "delta_fpr_cnn_minus_bert": None}
    labels = [int(pair[0]["label"]) for pair in selected]
    cm = gate_eval.evaluate_documents([float(pair[0]["risk_score"]) for pair in selected], labels, threshold_cnn)
    bm = gate_eval.evaluate_documents([float(pair[1]["risk_score"]) for pair in selected], labels, threshold_bert)
    return {
        "documents": len(selected), "attacks": cm["attacks"], "benign": cm["benign"],
        "recall_cnn": cm["recall"], "recall_bert": bm["recall"],
        "delta_recall_cnn_minus_bert": None if cm["recall"] is None or bm["recall"] is None else cm["recall"] - bm["recall"],
        "fpr_cnn": cm["fpr"], "fpr_bert": bm["fpr"],
        "delta_fpr_cnn_minus_bert": None if cm["fpr"] is None or bm["fpr"] is None else cm["fpr"] - bm["fpr"],
        "tp_cnn": cm["tp"], "tp_bert": bm["tp"], "fp_cnn": cm["fp"], "fp_bert": bm["fp"],
    }


def _cluster_indices(rows: list[tuple[dict[str, Any], dict[str, Any]]], field: str,
                     *, attacks_only: bool = False) -> tuple[list[str], dict[str, list[int]], dict[str, Any]]:
    groups: dict[str, list[int]] = {}
    missing = 0
    for index, (cnn, _) in enumerate(rows):
        if attacks_only and cnn["label"] != 1:
            continue
        key = cnn.get(field)
        if key is None or key == "":
            missing += 1
            if field == "payload_group_id":
                if attacks_only:
                    # Attack records without payload ancestry remain excluded;
                    # do not fabricate a cluster for them.
                    continue
                fail("Payload-group bootstrap would need to invent benign clusters; use attacks-only mode.")
            key = "__missing__:" + str(cnn["document_id"])
        groups.setdefault(str(key), []).append(index)
    keys = sorted(groups)
    notes = {"group_field": field, "cluster_count": len(keys), "documents": sum(len(x) for x in groups.values()),
             "missing_group_documents": missing,
             "sparse_group_warning": (f"Only {len(keys)} clusters; percentile bootstrap intervals may be unstable."
                                      if len(keys) < 20 else None)}
    return keys, groups, notes


def _bootstrap_interval(draws: list[float | None]) -> dict[str, Any]:
    finite = sorted(value for value in draws if value is not None and math.isfinite(value))
    if not finite:
        return {"n": 0, "q025": None, "median": None, "q975": None}
    def quantile(p: float) -> float:
        pos = (len(finite) - 1) * p
        lo = math.floor(pos); hi = math.ceil(pos)
        return finite[lo] * (hi - pos) + finite[hi] * (pos - lo) if lo != hi else finite[lo]
    return {"n": len(finite), "q025": quantile(.025), "median": quantile(.5), "q975": quantile(.975)}


def _group_bootstrap(rows: list[tuple[dict[str, Any], dict[str, Any]]], thresholds: dict[str, float],
                     reps: int, rng: random.Random, group_field: str, attacks_only: bool = False) -> dict[str, Any]:
    keys, groups, notes = _cluster_indices(rows, group_field, attacks_only=attacks_only)
    if not keys:
        return {"groups": notes, "draws": [], "intervals": {}}
    draws: dict[str, list[float | None]] = {"delta_recall": [], "delta_fpr": []}
    raw = []
    for _ in range(reps):
        sampled = [rng.choice(keys) for __ in range(len(keys))]
        indices = [idx for key in sampled for idx in groups[key]]
        metric = _metrics(rows, thresholds["cnn"], thresholds["bert"], indices)
        dr = metric["delta_recall_cnn_minus_bert"]
        df = metric["delta_fpr_cnn_minus_bert"]
        if attacks_only:
            df = None
        draws["delta_recall"].append(dr)
        draws["delta_fpr"].append(df)
        raw.append({"delta_recall": dr, "delta_fpr": df, "documents": len(indices)})
    return {"groups": notes, "intervals": {name: _bootstrap_interval(values) for name, values in draws.items()}, "draws": raw,
            "interpretation": "Cluster-resampling sensitivity on this fixed observed dataset; not an IID-document or seed-population interval."}


def _calibration_bootstrap(cal_cnn: list[dict[str, Any]], cal_bert: list[dict[str, Any]],
                           selection: list[tuple[dict[str, Any], dict[str, Any]]], alphas: list[float],
                           reps: int, rng: random.Random) -> dict[str, Any]:
    calibration = _pair_role(cal_cnn, cal_bert, "calibration")
    keys, groups, notes = _cluster_indices(calibration, "source_group_id")
    results = {}
    for alpha in alphas:
        threshold_draws = []
        metric_draws = []
        for _ in range(reps):
            sampled = [rng.choice(keys) for __ in range(len(keys))]
            repeated = [idx for key in sampled for idx in groups[key]]
            scores_c = [float(calibration[i][0]["risk_score"]) for i in repeated]
            scores_b = [float(calibration[i][1]["risk_score"]) for i in repeated]
            tc = gate_eval.fit_threshold(scores_c, alpha)
            tb = gate_eval.fit_threshold(scores_b, alpha)
            metric = _metrics(selection, tc, tb)
            threshold_draws.append({"cnn": tc, "bert": tb, "delta_cnn_minus_bert": tc - tb})
            metric_draws.append({"delta_recall": metric["delta_recall_cnn_minus_bert"],
                                 "delta_fpr": metric["delta_fpr_cnn_minus_bert"],
                                 "recall_cnn": metric["recall_cnn"], "recall_bert": metric["recall_bert"],
                                 "fpr_cnn": metric["fpr_cnn"], "fpr_bert": metric["fpr_bert"]})
        results[str(alpha)] = {
            "threshold_intervals": {name: _bootstrap_interval([x[name] for x in threshold_draws])
                                     for name in ("cnn", "bert", "delta_cnn_minus_bert")},
            "selection_metric_intervals": {name: _bootstrap_interval([x[name] for x in metric_draws])
                                            for name in metric_draws[0]},
            "threshold_draws": threshold_draws, "selection_metric_draws": metric_draws,
        }
    return {"groups": notes, "by_alpha": results,
            "interpretation": "Paired calibration-source-group resampling; same sampled source groups feed each model, each threshold is refit by gate_eval.fit_threshold, and selection documents stay fixed."}


def _aggregate_seeds(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for alpha in per_seed[0]["by_alpha"]:
        rows = [item["by_alpha"][alpha]["observed"] for item in per_seed]
        result[alpha] = {
            "fixed_seed_count": len(rows),
            "cnn_recall_min_across_fixed_seeds": min(x["recall_cnn"] for x in rows if x["recall_cnn"] is not None),
            "bert_recall_min_across_fixed_seeds": min(x["recall_bert"] for x in rows if x["recall_bert"] is not None),
            "cnn_fpr_max_across_fixed_seeds": max(x["fpr_cnn"] for x in rows if x["fpr_cnn"] is not None),
            "bert_fpr_max_across_fixed_seeds": max(x["fpr_bert"] for x in rows if x["fpr_bert"] is not None),
            "delta_recall_min_across_fixed_seeds": min(x["delta_recall_cnn_minus_bert"] for x in rows if x["delta_recall_cnn_minus_bert"] is not None),
            "delta_fpr_max_across_fixed_seeds": max(x["delta_fpr_cnn_minus_bert"] for x in rows if x["delta_fpr_cnn_minus_bert"] is not None),
            "scope": "Descriptive extrema over the named fixed seeds; seeds are not treated as a random sample.",
        }
    return result


def compare(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = Path(args.plan)
    if not plan_path.is_absolute(): plan_path = ROOT / plan_path
    pairs_path = Path(args.pairs)
    if not pairs_path.is_absolute(): pairs_path = ROOT / pairs_path
    plan = read_json(plan_path)
    pair_manifest = read_json(pairs_path)
    pairs = pair_manifest.get("pairs")
    if not isinstance(pairs, list): fail("Pair manifest must contain a pairs list.")
    stages = {pair.get("stage", "development") for pair in pairs}
    if len(stages) != 1:
        fail("A pair manifest must use one stage; specify stage on each pair if non-development.")
    stage = stages.pop()
    if stage not in ("development", "confirmation"):
        fail("Pair stage must be development or confirmation.")
    checked, freeze, freeze_sha = preflight(pairs, plan, stage, ROOT)
    # All authorization, target, role, membership, and saved-score hashes have
    # now been verified. Only now is it safe to decode score rows.
    alphas = [float(plan["calibration_policy"]["primary_alpha"])] + [float(a) for a in plan["calibration_policy"].get("secondary_alphas", [])]
    rng = random.Random(args.seed)
    per_seed = []
    for pair in checked:
        cnn_cal = _read_rows(pair["cnn"], pair["cnn"]["calibration_role"])
        bert_cal = _read_rows(pair["bert"], pair["bert"]["calibration_role"])
        cnn_sel = _read_rows(pair["cnn"], pair["cnn"]["selection_role"])
        bert_sel = _read_rows(pair["bert"], pair["bert"]["selection_role"])
        paired_cal = _pair_role(cnn_cal, bert_cal, "calibration")
        paired_sel = _pair_role(cnn_sel, bert_sel, "selection")
        if any(row[0]["label"] != 0 for row in paired_cal):
            fail("Calibration role must be benign only.")
        seed_out = {"seed": pair["seed"], "cnn_format": pair["cnn_format"], "bert_format": pair["bert_format"],
                    "cnn_target_key": pair["cnn"]["complete"].get("target_key"),
                    "bert_target_key": pair["bert"]["complete"].get("target_key"), "by_alpha": {}}
        for alpha in alphas:
            alpha_key = str(alpha)
            thresholds = {
                "cnn": float(pair["cnn"]["complete"]["metrics"][alpha_key]["threshold"]),
                "bert": float(pair["bert"]["complete"]["metrics"][alpha_key]["threshold"]),
            }
            refit_cnn = gate_eval.fit_threshold([float(row[0]["risk_score"]) for row in paired_cal], alpha)
            refit_bert = gate_eval.fit_threshold([float(row[1]["risk_score"]) for row in paired_cal], alpha)
            if thresholds["cnn"] != refit_cnn or thresholds["bert"] != refit_bert:
                fail(f"Saved threshold differs from unchanged gate_eval.fit_threshold at alpha {alpha_key}.")
            observed = _metrics(paired_sel, thresholds["cnn"], thresholds["bert"])
            for side, evaluation in (("cnn", pair["cnn"]), ("bert", pair["bert"])):
                saved = evaluation["complete"]["metrics"][alpha_key]
                if observed["recall_" + side] != saved["recall"] or observed["fpr_" + side] != saved["fpr"]:
                    fail(f"Recomputed paired metrics disagree with saved evaluator metrics at alpha {alpha_key}.")
            seed_out["by_alpha"][alpha_key] = {
                "thresholds": thresholds, "observed": observed,
                "selection_source_group_bootstrap": _group_bootstrap(paired_sel, thresholds, args.reps, rng, "source_group_id"),
                "selection_payload_group_attack_recall_bootstrap": _group_bootstrap(paired_sel, thresholds, args.reps, rng, "payload_group_id", attacks_only=True),
            }
        seed_out["paired_calibration_source_group_bootstrap"] = _calibration_bootstrap(cnn_cal, bert_cal, paired_sel, alphas, args.reps, rng)
        per_seed.append(seed_out)
    out = {
        "schema": "cnn_adoption_paired_saved_score_comparison_v1", "stage": stage,
        "comparison": "CNN minus BERT; exact paired document identities; one independently calibrated threshold per model and seed/format.",
        "plan_path": str(plan_path), "pairs_path": str(pairs_path), "pairs_sha256": sha256(pairs_path),
        "freeze": {"path": plan.get("confirmation_freeze"), "sha256": freeze_sha} if stage == "confirmation" else None,
        "bootstrap": {"replicates": args.reps, "deterministic_seed": args.seed,
                      "interval": "percentile q2.5/q50/q97.5; sensitivity summaries, not formal noninferiority intervals"},
        "per_seed": per_seed, "fixed_seed_extrema": _aggregate_seeds(per_seed),
        "limitations": [
            "Only the named fixed seeds and observed development/confirmation membership are summarized; seed extrema are descriptive, not a sampled-seed population estimate.",
            "Source-group and payload-group bootstrap intervals are separate sensitivity analyses; they do not establish generalization under distribution shift.",
            "Payload-group resampling is restricted to attacks and reports attack recall only; full-panel FPR is undefined for this resampling because natural benigns lack payload ancestry, while controlled retrospective benigns may carry template ancestry. No synthetic benign clusters are added.",
            "Small or sparse group counts make percentile intervals unstable; report group counts and raw draws alongside quantiles.",
            "No universal noninferiority margin or pass/fail decision is imposed by this comparator.",
        ],
        "input_artifacts": [
            {"seed": pair["seed"], "side": side, "complete_json": str(ev["path"]), "complete_sha256": sha256(ev["path"]),
             "score_files": {name: digest for name, digest in ev["complete"]["score_files_sha256"].items()}}
            for pair in checked for side, ev in (("cnn", pair["cnn"]), ("bert", pair["bert"]))
        ],
    }
    output = Path(args.output)
    if not output.is_absolute(): output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return {"output": str(output), "stage": stage, "seeds": len(per_seed), "replicates": args.reps}


def _self_test() -> None:
    """Small development-only synthetic check; creates no data artifacts."""
    from copy import deepcopy
    calibration = [
        {"document_id": f"c{i}", "text_sha256": f"t{i}", "window_input_sha256": f"w{i}", "label": 0,
         "window_count": 1, "source_group_id": f"cs{i//2}", "payload_group_id": None,
         "risk_score": value, "window_scores": [value]}
        for i, value in enumerate((.1, .2, .3, .4))
    ]
    selection = []
    c_scores = (.9, .8, .2, .1); b_scores = (.2, .8, .3, .1); labels = (1, 1, 0, 0)
    for i, (cs, bs, label) in enumerate(zip(c_scores, b_scores, labels)):
        common = {"document_id": f"d{i}", "text_sha256": f"t{i}", "window_input_sha256": f"w{i}", "label": label,
                  "window_count": 1, "source_group_id": f"s{i//2}", "payload_group_id": f"p{i//2}" if label else None}
        selection.append((common | {"risk_score": cs, "window_scores": [cs]}, common | {"risk_score": bs, "window_scores": [bs]}))
    thresholds = {"cnn": gate_eval.fit_threshold([x["risk_score"] for x in calibration], .25),
                  "bert": gate_eval.fit_threshold([x["risk_score"] for x in calibration], .25)}
    observed = _metrics(selection, thresholds["cnn"], thresholds["bert"])
    assert observed["delta_recall_cnn_minus_bert"] == 0.5
    assert observed["delta_fpr_cnn_minus_bert"] == 0.0
    source_boot = _group_bootstrap(selection, thresholds, 12, random.Random(7), "source_group_id")
    payload_boot = _group_bootstrap(selection, thresholds, 12, random.Random(7), "payload_group_id", attacks_only=True)
    assert source_boot["groups"]["cluster_count"] == 2
    assert payload_boot["groups"]["cluster_count"] == 1
    assert payload_boot["intervals"]["delta_fpr"]["n"] == 0
    calibration_pairs = [(row, row | {"risk_score": row["risk_score"] + .01,
                                       "window_scores": [row["risk_score"] + .01]}) for row in calibration]
    cal_sensitivity = _calibration_bootstrap(calibration, [item[1] for item in calibration_pairs], selection,
                                               [.25], 8, random.Random(11))
    assert len(cal_sensitivity["by_alpha"]["0.25"]["threshold_draws"]) == 8
    keys, groups, _ = _cluster_indices(selection, "source_group_id")
    assert set(keys) == {"s0", "s1"} and all(len(group) == 2 for group in groups.values())
    altered = deepcopy(selection)
    altered[0][1]["text_sha256"] = "corrupted"
    try:
        _pair_role([x[0] for x in altered], [x[1] for x in altered], "synthetic corruption")
    except ValueError as error:
        assert "text_sha256" in str(error)
    else:
        raise AssertionError("Pair identity corruption was not rejected")
    try:
        _validate_freeze({"confirmation_freeze": None, "confirmation_freeze_sha256": None}, ROOT)
    except ValueError as error:
        assert "no complete freeze" in str(error)
    else:
        raise AssertionError("Unfrozen confirmation was accepted")
    role_identities = {"development_calibration": {"files": {"ids.npy": "cal"}},
                       "development_selection": {"files": {"ids.npy": "sel"}}}
    shared_sources = {"scripts/cnn_adoption_evaluate.py": "eval", "scripts/cnn_adoption_train.py": "train",
                      "scripts/cnn_models.py": "models", "gate_eval.py": "gate"}
    cnn_eval = {"association": {"target": {"model_spec": {"kind": "cnn"}, "numerical_format": "w8a8_export"},
                                 "calibration_policy": {"rule": "score > threshold"}, "quality_gates": {},
                                 "data": role_identities, "source_sha256": shared_sources | {
                                     "scripts/cnn_adoption_export.py": "export", "scripts/cnn_adoption_backend.py": "backend"}},
                "calibration_role": "development_calibration", "selection_role": "development_selection"}
    bert_eval = {"association": {"target": {"model_spec": {"kind": "historical_bert_fp32"}, "numerical_format": "fp32"},
                                  "calibration_policy": {"rule": "score > threshold"}, "quality_gates": {},
                                  "data": role_identities, "source_sha256": shared_sources.copy()},
                 "calibration_role": "development_calibration", "selection_role": "development_selection"}
    formats = _validate_pair_targets({"seed": 42, "cnn_format": "w8a8_export", "bert_format": "fp32"}, cnn_eval, bert_eval)
    assert formats == ("w8a8_export", "fp32")
    _validate_scoring_dependencies(cnn_eval["association"], cnn_eval["association"]["target"],
                                   {"source_sha256": shared_sources | {"scripts/cnn_adoption_export.py": "export",
                                                                       "scripts/cnn_adoption_backend.py": "backend",
                                                                       "scripts/unrelated_profiler.py": "profiler"}})
    incomplete = deepcopy(bert_eval["association"])
    del incomplete["source_sha256"]["gate_eval.py"]
    try:
        _validate_scoring_dependencies(incomplete, incomplete["target"], None)
    except ValueError as error:
        assert "gate_eval.py" in str(error)
    else:
        raise AssertionError("Missing scoring dependency was accepted")
    print("synthetic checks passed: paired metrics, source clustering, identity corruption, pre-row confirmation guard")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan")
    parser.add_argument("--pairs")
    parser.add_argument("--output")
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.plan or not args.pairs or not args.output:
        parser.error("--plan, --pairs, and --output are required unless --self-test is used")
    if args.reps < 1:
        parser.error("--reps must be positive")
    print(json.dumps(compare(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
