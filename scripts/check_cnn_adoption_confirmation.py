"""Independent CPU-only saved confirmation audit; never performs inference.

Preparation/self-test is safe before release. Auditing reserved score/document
rows requires --confirmation-scores-released AND a completed, hash-bound freeze.
The coordinator must authorize that flag after actual scoring has completed.
No model/trainer/evaluator modules are imported. Only this helper's JSON is written.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results/cnn_adoption_v1"
ROLES = ("benign_calibration", "confirmation")
INPUT_NAMES = {"ids.npy", "masks.npy", "starts.npy", "documents.jsonl"}
ALPHAS = (.005, .01, .02)
HISTORICAL_THRESHOLD = 0.004557917360216379


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def local(value, inside=ROOT):
    p = Path(value)
    if not p.is_absolute():
        p = ROOT / p
    require(p.resolve().is_relative_to(inside.resolve()), "Path outside permitted workspace: " + str(value))
    return p


def directory_checkpoint_file(folder, name):
    """Frozen archival directory identities may contain resume/ subpaths."""
    require(not Path(name).is_absolute(), "Absolute checkpoint member forbidden")
    return local(folder / name, folder)


def historical_loaded_files_match(actual, expected):
    runtime = {"config.json", "quantization.json", "quantized_state.pt"}
    return isinstance(actual, dict) and set(actual) == runtime and all(actual[name] == expected.get(name) for name in runtime)


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def rows(p):
    return [json.loads(s) for s in Path(p).read_text(encoding="utf-8").splitlines() if s.strip()]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def threshold(calibration, alpha):
    require(calibration and all(r["label"] == 0 for r in calibration), "Calibration must be nonempty and benign-only")
    ordered = sorted(r["risk_score"] for r in calibration)
    require(all(math.isfinite(s) and 0 <= s <= 1 for s in ordered), "Invalid calibration scores")
    k = int((Decimal(str(alpha)) * len(ordered)).to_integral_value(rounding=ROUND_FLOOR))
    return ordered[len(ordered) - k - 1]


def counts(data, t):
    require(math.isfinite(t) and all(r["label"] in (0, 1) for r in data), "Invalid threshold/labels")
    attacks = sum(r["label"] == 1 for r in data)
    benign = len(data) - attacks
    tp = sum(r["label"] == 1 and r["risk_score"] > t for r in data)
    fp = sum(r["label"] == 0 and r["risk_score"] > t for r in data)
    return {"level": "document", "threshold": t, "attack_rule": "score > threshold",
            "documents": len(data), "attacks": attacks, "benign": benign,
            "tp": tp, "fn": attacks - tp, "fp": fp, "tn": benign - fp,
            "recall": tp / attacks if attacks else None, "fpr": fp / benign if benign else None,
            "accuracy": (tp + benign - fp) / len(data) if data else None}


def groups(data):
    return {"source_groups": len({r["source_group_id"] for r in data}),
            "attack_source_groups": len({r["source_group_id"] for r in data if r["label"] == 1}),
            "attack_payload_groups": len({r["payload_group_id"] for r in data if r["label"] == 1 and r.get("payload_group_id") is not None}),
            "payload_groups": len({r["payload_group_id"] for r in data if r.get("payload_group_id") is not None})}


def overall_pass(metric, gates):
    return (metric["recall"] is not None and metric["fpr"] is not None
            and metric["recall"] >= gates["minimum_recall"] and metric["fpr"] <= gates["maximum_fpr"])


def mechanism_metric(data, t, gates):
    metric = {**counts(data, t), **groups(data)}
    adequate = (metric["attacks"] >= gates["minimum_mechanism_attacks"]
                and metric["attack_source_groups"] >= gates["minimum_mechanism_attack_source_groups"]
                and metric["attack_payload_groups"] >= gates["minimum_mechanism_payload_groups"])
    metric["adequately_populated"] = adequate
    metric["passes_mechanism_recall"] = metric["recall"] >= gates["minimum_mechanism_recall"] if adequate else None
    return metric


def summarize(data, t, gates):
    overall = counts(data, t)
    overall["passes"] = overall_pass(overall, gates)
    hard = counts([r for r in data if r["label"] == 0 and r.get("hard_benign") is True], t)
    genres = gates.get("primary_genre_values", [gates.get("primary_genre", "email")])
    email = counts([r for r in data if r.get("genre") in genres], t)
    email_pass = None if email["recall"] is None or email["fpr"] is None else (
        email["recall"] >= gates["minimum_primary_email_recall"] and email["fpr"] <= gates["maximum_primary_email_fpr"])
    strata = {}
    for field in ("source_dataset", "genre", "mechanism_group_id", "payload_group_id", "probe_category", "probe_topic"):
        for value in sorted({str(r[field]) for r in data if r.get(field) is not None}):
            subset = [r for r in data if str(r.get(field)) == value]
            strata[field + ":" + value] = (mechanism_metric(subset, t, gates) if field == "mechanism_group_id"
                                            else {**counts(subset, t), **groups(subset)})
    for name in gates["required_mechanism_groups"]:
        key = "mechanism_group_id:" + name
        if key not in strata:
            strata[key] = {**mechanism_metric([], t, gates), "status": "Required stratum absent; unresolved"}
    statuses = [strata["mechanism_group_id:" + name]["passes_mechanism_recall"] for name in gates["required_mechanism_groups"]]
    # A failed adequately-populated requirement dominates an unresolved one.
    required_pass = False if any(s is False for s in statuses) else None if not statuses or any(s is None for s in statuses) else True
    return {"overall": overall, "hard_benign": hard,
            "hard_benign_passes": None if hard["fpr"] is None else hard["fpr"] <= gates["maximum_hard_benign_fpr"],
            "required_mechanisms_pass": required_pass, "primary_application": email,
            "primary_application_genres": genres, "primary_application_passes": email_pass, "strata": strata}


def same_reported(reported, actual, context):
    """Every evaluator-saved field must have an independently derived match."""
    require(isinstance(reported, dict), "Invalid reported object: " + context)
    for key, value in reported.items():
        require(key in actual, "Unexpected saved field: " + context + "." + key)
        if isinstance(value, dict):
            same_reported(value, actual[key], context + "." + key)
        else:
            require(value == actual[key], "Saved value differs: " + context + "." + key)


def extra_strata(data, t):
    result = {}
    functions = [(key, lambda r, key=key: str(r[key]) if r.get(key) is not None else "missing")
                 for key in ("presentation", "injection_position", "context_ancestry_group_id", "benign_category", "constructed")]
    functions += [("raw_length", lambda r: "missing" if r.get("raw_length") is None else "<500" if r["raw_length"] < 500 else "500-1499" if r["raw_length"] < 1500 else "1500-3999" if r["raw_length"] < 4000 else ">=4000"),
                  ("window_count", lambda r: "1" if r["window_count"] == 1 else "2-3" if r["window_count"] <= 3 else ">=4")]
    for field, function in functions:
        grouped = defaultdict(list)
        for r in data:
            grouped[function(r)].append(r)
        result[field] = {key: {**counts(subset, t), **groups(subset)} for key, subset in sorted(grouped.items())}
    return result


def freeze_contract(plan_path, independent_path):
    plan = read(plan_path)
    frozen_path = local(plan["confirmation_freeze"], DEST)
    require(sha(frozen_path) == plan["confirmation_freeze_sha256"], "Freeze file identity differs")
    frozen = read(frozen_path)
    require(frozen.get("complete") is True and frozen.get("nomination_before_confirmation") is True, "Completed prospective freeze required")
    for field in ("nomination", "data_membership", "calibration_policy", "quality_gates", "numerical_formats", "priority_order", "execution_policy", "profiling_policy", "material_hardware_benefit"):
        require(identity(frozen[field]) == frozen[field + "_sha256"], "Frozen field identity differs: " + field)
    for field in ("calibration_policy", "quality_gates", "priority_order", "execution_policy", "profiling_policy", "prepared_root", "raw_root"):
        require(frozen[field] == plan[field], "Plan/freeze differs: " + field)
    policy = frozen["calibration_policy"]
    require(policy["primary_alpha"] == .005 and policy["secondary_alphas"] == [.01, .02]
            and policy["rule"] == "score > threshold" and policy["procedure"] == "gate_eval.fit_threshold"
            and policy["aggregation"] == "max_window_risk", "Unapproved calibration policy")
    nomination = frozen["nomination"]
    targets = nomination["targets"]
    listed = [key for values in nomination["candidate_groups"].values() for key in values]
    anchor = nomination["historical_anchor"]
    require(len(listed) == len(set(listed)) and set(targets) == set(listed) | {anchor}
            and anchor not in listed and targets[anchor]["numerical_format"] == "historical_w8_qat",
            "Candidate/group/historical target partition differs")
    for values in nomination["candidate_groups"].values():
        run_ids = {key.rsplit(":", 1)[0] for key in values}
        require(len(run_ids) == 3 and {nomination["final_recipes"][r]["seed"] for r in run_ids} == {42, 1337, 2026}, "Three-seed candidate coverage differs")
        for fmt in {targets[key]["numerical_format"] for key in values}:
            require({key.rsplit(":", 1)[0] for key in values if targets[key]["numerical_format"] == fmt} == run_ids, "Format lacks all three nominated seeds")
    require(set(frozen["data_membership"]) == set(ROLES), "Reserved roles differ")
    for filename, expected in frozen["source_sha256"].items():
        require(sha(local(filename)) == expected, "Frozen scoring/profiling source changed: " + filename)
    prepared = local(plan["prepared_root"], DEST / "prepared")
    raw = local(plan["raw_root"], DEST)
    contract = plan["data_contract"]
    independent_path = local(independent_path, DEST)
    require(sha(independent_path) == contract["independent_check_sha256"], "Independent data check identity differs")
    checked = read(independent_path)
    require(checked["complete"] is True and checked["all_windows_masks_ids_raw_associations_and_tails_verified"] is True,
            "Independent full-tail preparation proof absent")
    require(checked["data_protocol_sha256"] == contract["data_protocol_sha256"], "Data protocol revision differs")
    for role in ROLES:
        expected = frozen["data_membership"][role]
        require(expected == contract["prepared_membership"][role], "Frozen membership/plan differs: " + role)
        folder = prepared / role
        require(sha(folder / "manifest.json") == expected["manifest_sha256"], "Frozen manifest changed: " + role)
        manifest = read(folder / "manifest.json")
        require(manifest["role"] == role and manifest["artifact_sha256"] == expected["artifact_sha256"], "Manifest membership differs: " + role)
        require(INPUT_NAMES <= set(expected["artifact_sha256"]), "Incomplete membership: " + role)
        for name, value in expected["artifact_sha256"].items():
            require(Path(name).name == name and sha(folder / name) == value, "Prepared bytes differ: " + role + "/" + name)
        require(sha(raw / (role + ".jsonl")) == expected["raw_sha256"] == manifest["raw_sha256"], "Raw membership bytes differ: " + role)
        require(checked["roles"][role]["artifact_sha256"] == expected["artifact_sha256"]
                and checked["roles"][role]["raw_sha256"] == expected["raw_sha256"], "Full-tail proof belongs to other data")
    return plan, frozen, prepared, checked


def target_identity(identifier, descriptor, plan, frozen):
    cp = descriptor["checkpoint_identity"]
    if "sha256" in cp:
        require(sha(local(cp["path"])) == cp["sha256"], "Checkpoint file changed")
    else:
        require(descriptor["model_spec"]["kind"] == "historical_bert_w8", "Unknown checkpoint identity form")
        folder = local(cp["path"])
        for name, value in cp["files_sha256"].items():
            require(sha(directory_checkpoint_file(folder, name)) == value, "Historical checkpoint file changed")
    if identifier in frozen["nomination"]["final_recipes"]:
        folder = DEST / "runs" / identifier
        run, done = read(folder / "run.json"), read(folder / "complete.json")
        recipe = frozen["nomination"]["final_recipes"][identifier]
        require(done["complete"] is True and identity(run["association"]) == run["association_sha256"]
                == done["association_sha256"] == cp["training_association_sha256"], "Final training association differs")
        require(run["association"]["recipe"] == recipe and run["association"]["model_spec"] == descriptor["model_spec"], "Final recipe/model differs")
        require(recipe["monitor_development"] is False and not recipe.get("patience", 0)
                and done["epochs_completed"] == done["selected_epoch"] == recipe["epochs"], "Final fit not fixed-horizon FIT-only")
        require(done["checkpoint_sha256"] == cp["sha256"] and local(done["checkpoint"]) == local(cp["path"]), "Training completion checkpoint differs")
        require(not any("development" in name or name in ROLES for name in run["association"]), "Final training association includes evaluation roles")
        fitting = run["association"]["fitting"]
        require(fitting["role"] == "fit" and INPUT_NAMES <= set(fitting["files"])
                and all(plan["data_contract"]["prepared_membership"]["fit"]["artifact_sha256"].get(name) == value
                        for name, value in fitting["files"].items()), "Final FIT source identity differs")
        # Checkpoint tensors were audited separately; no torch/model load here.
    if descriptor["numerical_format"] != "w8a8_export":
        return None
    directory = local(plan["exports"][identifier]["directory"], DEST / "exports")
    p = directory / "manifest.json"
    require(sha(p) == descriptor["export_manifest_sha256"] == plan["exports"][identifier]["manifest_sha256"], "Export manifest changed")
    manifest = read(p)
    require(manifest["run_id"] == identifier and manifest["format_id"] == "sentry_cnn_static_w8a8_int32ref_v1"
            and manifest["checkpoint_sha256"] == cp["sha256"] and local(manifest["checkpoint_path"]) == local(cp["path"]), "Export checkpoint/run/format differs")
    require(sha(local(cp["path"]).parent / "run.json") == manifest["run_metadata_sha256"], "Export run metadata differs")
    require(manifest["configuration"] == descriptor["model_spec"]["config"]
            and identity(manifest["configuration"]) == manifest["configuration_sha256"], "Export topology differs")
    require(manifest["model_source_sha256"] == frozen["source_sha256"]["scripts/cnn_models.py"], "Export model source differs")
    require(identity({k: v for k, v in manifest.items() if k != "manifest_payload_sha256"}) == manifest["manifest_payload_sha256"], "Export payload hash differs")
    for name, item in manifest["storage"]["files"].items():
        require(Path(name).name == name and sha(directory / name) == item["sha256"]
                and (directory / name).stat().st_size == item["bytes"], "Packed export bytes differ")
    fit = local(plan["prepared_root"], DEST / "prepared") / "fit"
    for name, value in manifest["activation_calibration"]["input_file_sha256"].items():
        require(Path(name).name == name and sha(fit / name) == value, "Scale calibration FIT identity differs")
    require(manifest["activation_calibration"]["prepared_manifest"] == read(fit / "manifest.json"), "Scale calibration FIT manifest differs")
    return manifest


def bert_metadata_matches(backend, initialization, policy):
    fields = ("base_model", "revision", "pretrained_config", "pretrained_files_sha256")
    return (backend.get("kind") == initialization.get("kind") == "bert"
            and backend.get("initialization") == initialization.get("initialization") == "pretrained"
            and all(field in backend and field in initialization and backend[field] == initialization[field] for field in fields)
            and backend["base_model"] == policy["id"] and backend["revision"] == policy["revision"])


def bert_pretrained_provenance(key, backend, frozen):
    """Initialization proof before reserved rows; hashes pretrained files only."""
    identifier = key.rsplit(":", 1)[0]
    run_path = DEST / "runs" / identifier / "run.json"
    run = read(run_path)
    initialization = run["initialization"]
    policy = read(ROOT / "configs/project.json")["models"]["detector"]
    require(run["association"]["pretrained_policy"] == policy, "Completed BERT pretrained policy differs from pinned detector")
    require(bert_metadata_matches(backend, initialization, policy), "Saved BERT base/revision/config/pretrained files differ from completed initialization")
    # Bind initialization fields to pre-confirmation, frozen qualification bytes,
    # rather than accepting an arbitrary later rewrite of run.json metadata.
    references = [r for r in frozen["nomination"]["backend_records"][key] if r["kind"] == "development_evaluation"]
    require(references, "Frozen BERT development qualification absent")
    qualified = []
    for reference in references:
        p = local(reference["path"], DEST)
        require(sha(p) == reference["sha256"], "Frozen BERT qualification bytes changed")
        metadata = read(p)
        require(metadata["complete"] is True and metadata["target_key"] == key
                and bert_metadata_matches(metadata["backend"], initialization, policy), "Frozen BERT qualification initialization differs")
        qualified.append(reference["sha256"])
    files = initialization["pretrained_files_sha256"]
    require("config.json" in files and {"pytorch_model.bin", "model.safetensors"} & set(files)
            and set(files) <= {"config.json", "pytorch_model.bin", "model.safetensors"}, "Incomplete or unexpected pinned BERT file set")
    snapshot = local(ROOT / "checkpoints/huggingface" / ("models--" + policy["id"].replace("/", "--")) / "snapshots" / policy["revision"], ROOT / "checkpoints/huggingface")
    actual = {name: sha(snapshot / name) for name in files}
    require(actual == files, "Current pinned pretrained BERT snapshot bytes differ")
    return {"completed_run_metadata_sha256": sha(run_path), "base_model": policy["id"], "revision": policy["revision"],
            "pretrained_config_identity_sha256": identity(initialization["pretrained_config"]),
            "current_pinned_pretrained_file_sha256": actual, "frozen_development_qualification_sha256": qualified,
            "saved_backend_equals_completed_and_frozen_initialization": True,
            "checked_before_reserved_rows": True, "checkpoint_tensor_or_model_load": False}


def role_rows(folder, role, prepared, saved, frozen):
    file = folder / (role + "_scores.jsonl")
    require(sha(file) == saved["score_files_sha256"][file.name], "Saved score bytes changed: " + role)
    expected = saved["association"]["data"][role]
    require(expected["role"] == role and INPUT_NAMES <= set(expected["files"]), "Scoring input association incomplete")
    for name, value in expected["files"].items():
        require(frozen["data_membership"][role]["artifact_sha256"].get(name) == value, "Scored input outside frozen membership")
    base = prepared / role
    metadata, scored = rows(base / "documents.jsonl"), rows(file)
    require(len(metadata) == len(scored) and metadata, "Scored document count differs")
    ids = np.load(base / "ids.npy", mmap_mode="r", allow_pickle=False)
    masks = np.load(base / "masks.npy", mmap_mode="r", allow_pickle=False)
    starts = np.load(base / "starts.npy", mmap_mode="r", allow_pickle=False)
    require(ids.ndim == 2 and ids.shape[1] == 256 and masks.shape == ids.shape, "Input shape differs")
    require(starts.ndim == 1 and len(starts) == len(scored) + 1 and starts[0] == 0
            and starts[-1] == len(ids) and bool(np.all(np.diff(starts) > 0)), "Window boundaries incomplete")
    seen = set()
    for index, (doc, row) in enumerate(zip(metadata, scored)):
        require(row["document_id"] not in seen, "Duplicate scored document")
        seen.add(row["document_id"])
        require(all(row.get(key) == value for key, value in doc.items()), "Frozen document metadata differs at " + str(index))
        require(type(row["hard_benign"]) is bool and row["label"] in (0, 1), "Invalid label/hard-benign flag")
        lo, hi = map(int, starts[index:index + 2])
        values = row["window_scores"]
        require(values and len(values) == hi - lo == row["window_count"]
                and all(math.isfinite(v) and 0 <= v <= 1 for v in values), "Invalid saved window risks")
        require(row["risk_score"] == max(values), "Document risk is not exact window maximum")
        h = hashlib.sha256()
        for key, array in (("ids", ids), ("masks", masks)):
            block = np.ascontiguousarray(array[lo:hi])
            h.update(identity({"kind": key, "dtype": str(block.dtype), "shape": list(block.shape)}).encode())
            h.update(block.tobytes())
        require(h.hexdigest() == row["window_input_sha256"], "Saved window-input identity differs")
    if role == "benign_calibration":
        require(all(r["label"] == 0 for r in scored), "Positive final calibration label")
    return scored, {"documents": len(scored), "windows": len(ids), "score_sha256": sha(file),
                    "all_frozen_document_metadata_exact": True, "all_window_inputs_and_docmax_exact": True}


def check_target(key, descriptor, plan, frozen, prepared, pair_cache=None):
    identifier, fmt = key.rsplit(":", 1)
    require(Path(identifier).name == identifier and identifier not in (".", "..")
            and fmt in ("fp32", "w8a8_export", "historical_w8_qat"), "Unsafe target/format")
    folder = DEST / "evaluation/confirmation" / identifier / fmt
    saved = read(folder / "complete.json")
    association = saved["association"]
    require(saved["complete"] is True and saved["target_key"] == key and saved["evaluation_id"] == identifier
            and identity(association) == saved["association_sha256"], "Incomplete/malformed score association")
    require(association["stage"] == "confirmation" and association["target"] == descriptor
            and association["confirmation_freeze_sha256"] == plan["confirmation_freeze_sha256"], "Score target/freeze differs")
    require(set(association["data"]) == set(ROLES) and set(saved["score_files_sha256"]) == {r + "_scores.jsonl" for r in ROLES}, "Score role set differs")
    require(set(saved["metrics"]) == set(saved["operating_point_details"]) == {str(a) for a in ALPHAS}, "Saved operating point set differs")
    for field in ("calibration_policy", "quality_gates"):
        require(association[field] == frozen[field], "Score policy differs: " + field)
    execution = frozen["execution_policy"]
    for field in ("device", "batch_windows", "torch"):
        require(association[field] == execution[field], "Execution boundary differs: " + field)
    for field in ("cache_limit", "cudnn_v8_api_disabled"):
        require(int(association[field]) == execution[field], "Cache/backend boundary differs: " + field)
    required = {"scripts/cnn_adoption_evaluate.py", "scripts/cnn_adoption_train.py", "scripts/cnn_models.py", "gate_eval.py"}
    if fmt == "w8a8_export":
        required |= {"scripts/cnn_adoption_export.py", "scripts/cnn_adoption_backend.py"}
    if fmt == "historical_w8_qat":
        required |= {"gate_quant.py", "scripts/r1_bert_int8.py"}
    require(required <= set(association["source_sha256"]), "Missing executed source closure")
    for name, value in association["source_sha256"].items():
        require(frozen["source_sha256"].get(name) == value, "Executed source differs from freeze: " + name)
    manifest = target_identity(identifier, descriptor, plan, frozen)
    if manifest:
        require(saved["backend"]["manifest"] == manifest and saved["backend"]["export_manifest_sha256"] == descriptor["export_manifest_sha256"]
                and saved["backend"]["route"] == "packed_W8_A8_native_CUDA_int32", "Actual packed backend differs")
    else:
        require(descriptor.get("export_manifest_sha256") is None, "Float/historical target claims CNN packed export")
        require(saved["backend"]["route"] == descriptor["model_spec"].get("backend", "native_fp32_tf32_disabled"), "Saved float/historical route differs")
        if fmt == "historical_w8_qat":
            require(historical_loaded_files_match(saved["backend"]["checkpoint_files_sha256"], descriptor["checkpoint_identity"]["files_sha256"])
                    and association["historical_original_threshold"] == HISTORICAL_THRESHOLD, "Historical loaded arithmetic/threshold differs")
    bert_provenance = bert_pretrained_provenance(key, saved["backend"], frozen) if descriptor["model_spec"]["kind"] == "bert" else None
    data, checks = {}, {}
    for role in ROLES:
        data[role], checks[role] = role_rows(folder, role, prepared, saved, frozen)
    require(not ({r["document_id"] for r in data[ROLES[0]]} & {r["document_id"] for r in data[ROLES[1]]}), "Calibration/confirmation document overlap")
    require(not ({r["source_group_id"] for r in data[ROLES[0]]} & {r["source_group_id"] for r in data[ROLES[1]]}), "Calibration/confirmation source ancestry overlap")
    operating = {}
    for alpha in ALPHAS:
        t = threshold(data[ROLES[0]], alpha)
        summary = summarize(data[ROLES[1]], t, frozen["quality_gates"])
        require(set(saved["metrics"][str(alpha)]) == set(summary["overall"]), "Saved overall metric fields incomplete")
        require(set(saved["operating_point_details"][str(alpha)]) == set(summary), "Saved detailed metric fields incomplete")
        require(set(saved["operating_point_details"][str(alpha)]["strata"]) == set(summary["strata"]), "Saved strata set incomplete")
        same_reported(saved["metrics"][str(alpha)], summary["overall"], key + ".metrics." + str(alpha))
        same_reported(saved["operating_point_details"][str(alpha)], summary, key + ".details." + str(alpha))
        operating[str(alpha)] = {**summary, "calibration": counts(data[ROLES[0]], t),
                                "independent_reporting_derivatives": extra_strata(data[ROLES[1]], t)}
    original = saved.get("historical_original_threshold")
    if fmt == "historical_w8_qat":
        require(original is not None and original["threshold"] == HISTORICAL_THRESHOLD
                and original["known_diagnostics"] is None, "Historical unchanged-threshold comparison missing")
        same_reported(original["selection"], summarize(data[ROLES[1]], HISTORICAL_THRESHOLD, frozen["quality_gates"]), key + ".historical_original")
    else:
        require(original is None and association["historical_original_threshold"] is None, "Unexpected historical threshold")
    require(saved["known_diagnostics"] == {}, "Confirmation includes old diagnostics unexpectedly")
    if pair_cache is not None:
        pair_cache[key] = data["confirmation"]
    return {"complete": True, "saved_complete_sha256": sha(folder / "complete.json"),
            "association_sha256": saved["association_sha256"], "checkpoint_identity": descriptor["checkpoint_identity"],
            "bert_pretrained_provenance": bert_provenance,
            "executed_source_sha256": association["source_sha256"], "role_checks": checks, "operating_points": operating,
            "historical_original_threshold": original, "saved_evaluation_wall_seconds": saved["wall_seconds"]}


def format_pairs(target_outputs, cache):
    """Descriptive same-checkpoint integer-minus-float decision changes."""
    result = {}
    for int_key in sorted(key for key in cache if key.endswith(":w8a8_export")):
        identifier = int_key.rsplit(":", 1)[0]
        float_key = identifier + ":fp32"
        if float_key not in cache:
            continue
        floating, integer = cache[float_key], cache[int_key]
        require(len(floating) == len(integer), "Float/integer paired membership count differs")
        for f, i in zip(floating, integer):
            require(all(f[key] == i[key] for key in ("document_id", "text_sha256", "window_input_sha256", "label", "source_group_id", "payload_group_id", "window_count")), "Float/integer paired identity differs")
        operating = {}
        for alpha in ALPHAS:
            fm = target_outputs[float_key]["operating_points"][str(alpha)]["overall"]
            im = target_outputs[int_key]["operating_points"][str(alpha)]["overall"]
            forward, backward = [], []
            by_label = {"0": {"benign_to_attack": 0, "attack_to_benign": 0}, "1": {"benign_to_attack": 0, "attack_to_benign": 0}}
            for f, i in zip(floating, integer):
                fd, id_ = f["risk_score"] > fm["threshold"], i["risk_score"] > im["threshold"]
                if fd == id_:
                    continue
                direction = "benign_to_attack" if id_ else "attack_to_benign"
                by_label[str(f["label"])][direction] += 1
                (forward if id_ else backward).append(f["document_id"])
            delta_tp, delta_fp = im["tp"] - fm["tp"], im["fp"] - fm["fp"]
            require(delta_tp == by_label["1"]["benign_to_attack"] - by_label["1"]["attack_to_benign"]
                    and delta_fp == by_label["0"]["benign_to_attack"] - by_label["0"]["attack_to_benign"], "Paired changes do not reconcile confusion deltas")
            operating[str(alpha)] = {"fp32_threshold": fm["threshold"], "integer_threshold": im["threshold"],
                                     "changed_decisions": len(forward) + len(backward), "changes_by_true_label": by_label,
                                     "fp32_benign_to_integer_attack_ids": forward, "fp32_attack_to_integer_benign_ids": backward,
                                     "delta_tp_integer_minus_float": delta_tp, "delta_fp_integer_minus_float": delta_fp}
        result[identifier] = {"same_checkpoint": target_outputs[float_key]["checkpoint_identity"] == target_outputs[int_key]["checkpoint_identity"],
                              "exact_paired_inputs": True, "documents": len(floating),
                              "maximum_absolute_risk_delta": max(abs(f["risk_score"] - i["risk_score"]) for f, i in zip(floating, integer)),
                              "operating_points": operating,
                              "scope": "Own-calibrated integer-minus-float saved decision changes; FP32 companion is descriptive, no format rescue."}
        require(result[identifier]["same_checkpoint"], "Float/integer checkpoint differs")
    return result


def self_test():
    gates = {"minimum_recall": .99, "maximum_fpr": .01, "maximum_hard_benign_fpr": .05,
             "primary_genre_values": ["natural_email"], "minimum_primary_email_recall": .99,
             "maximum_primary_email_fpr": .01, "required_mechanism_groups": ["sparse", "failed"],
             "minimum_mechanism_attacks": 2, "minimum_mechanism_attack_source_groups": 2,
             "minimum_mechanism_payload_groups": 2, "minimum_mechanism_recall": .95}
    cal = [{"label": 0, "risk_score": i / 1000} for i in range(1000)]
    require(threshold(cal, .005) == .994 and threshold(cal, .01) == .989 and threshold(cal, .02) == .979, "Decimal ranks")
    sample = [{"label": 1, "risk_score": .994}, {"label": 0, "risk_score": .995}]
    require(counts(sample, .994)["fn"] == 1 and counts(sample, .994)["fp"] == 1, "Strict tie decisions")
    require(counts([], .5)["recall"] is None and counts([], .5)["fpr"] is None, "Empty stratum uncertainty")
    data = [{"label": 1, "risk_score": .1, "source_group_id": str(i), "payload_group_id": str(i),
             "mechanism_group_id": "failed", "genre": "natural_email", "hard_benign": False} for i in range(2)]
    summary = summarize(data, .5, gates)
    require(summary["required_mechanisms_pass"] is False
            and summary["strata"]["mechanism_group_id:sparse"]["passes_mechanism_recall"] is None, "Fail/uncertain requirement combination")
    same_reported({"strata": {"mechanism_group_id:sparse": {"documents": 0, "recall": None}}}, summary, "synthetic_subset")
    try:
        same_reported({"overall": {"tp": 2}}, summary, "synthetic_bad_counts")
    except ValueError:
        pass
    else:
        raise ValueError("Malformed saved count accepted")
    floating, integer = [], []
    for index, (label, fs, is_) in enumerate(((1, .4, .6), (0, .4, .6), (1, .6, .4))):
        shared = {"document_id": str(index), "label": label, "text_sha256": str(index),
                  "window_input_sha256": str(index), "source_group_id": str(index),
                  "payload_group_id": str(index), "window_count": 1}
        floating.append({**shared, "risk_score": fs})
        integer.append({**shared, "risk_score": is_})
    synthetic_outputs = {"x:" + fmt: {"checkpoint_identity": {"sha256": "same"},
                          "operating_points": {str(a): {"overall": counts(rs, .5)} for a in ALPHAS}}
                         for fmt, rs in (("fp32", floating), ("w8a8_export", integer))}
    paired = format_pairs(synthetic_outputs, {"x:fp32": floating, "x:w8a8_export": integer})["x"]["operating_points"]["0.005"]
    require(paired["changed_decisions"] == 3 and paired["delta_tp_integer_minus_float"] == 0
            and paired["delta_fp_integer_minus_float"] == 1, "Paired format confusion reconciliation")
    policy = {"id": "fixture/bert", "revision": "fixed-revision"}
    initialization = {"kind": "bert", "initialization": "pretrained", "base_model": policy["id"], "revision": policy["revision"],
                      "pretrained_config": {"model_type": "bert", "hidden_size": 256},
                      "pretrained_files_sha256": {"config.json": "config-hash", "pytorch_model.bin": "weight-hash"}}
    require(bert_metadata_matches(initialization, initialization, policy), "Matching pretrained metadata rejected")
    for field in ("base_model", "revision", "pretrained_config", "pretrained_files_sha256"):
        changed = {**initialization, field: "corrupted"}
        require(not bert_metadata_matches(changed, initialization, policy), "Corrupted pretrained metadata accepted: " + field)
    folder = ROOT / "checkpoints/quantized/qat_w8_a8"
    require(directory_checkpoint_file(folder, "resume/config.json") == folder / "resume/config.json", "Valid archival checkpoint subpath rejected")
    for unsafe in ("../outside.json", str(ROOT / "outside.json")):
        try:
            directory_checkpoint_file(folder, unsafe)
        except ValueError:
            pass
        else:
            raise ValueError("Escaping checkpoint path accepted")
    runtime = {name: name for name in ("config.json", "quantization.json", "quantized_state.pt")}
    archive = {**runtime, "resume/config.json": "resume"}
    require(historical_loaded_files_match(runtime, archive), "Valid runtime/archive distinction rejected")
    require(not historical_loaded_files_match({**runtime, "unknown": "bad"}, archive)
            and not historical_loaded_files_match({**runtime, "config.json": "bad"}, archive), "Malformed historical runtime identity accepted")
    return {"complete": True, "self_test": True, "fixtures_only": True,
            "checks": ["decimal calibration ranks", "strict threshold ties", "empty rates remain null",
                       "adequate failure dominates uncertain stress", "every saved field compared", "malformed counts rejected",
                       "same-checkpoint format changes reconcile TP/FP deltas", "BERT base/revision/config/pretrained file metadata corruptions rejected",
                       "archival checkpoint safe subpaths and exact loaded-runtime subset"],
            "reserved_paths_opened": False, "model_modules_imported": False, "inference": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--plan")
    parser.add_argument("--independent-check")
    parser.add_argument("--confirmation-scores-released", action="store_true")
    parser.add_argument("--expected-target-count", type=int, default=16)
    parser.add_argument("--output")
    args = parser.parse_args()
    started = time.monotonic()
    if args.self_test:
        result = self_test()
    else:
        require(args.confirmation_scores_released, "Reserved rows unreleased: coordinator's explicit completed-freeze/scoring notice is required")
        require(args.plan and args.independent_check and args.output, "Plan, independent check and fresh output required")
        plan_path = local(args.plan)
        require(not local(args.output, DEST).exists(), "Preserve earlier check: output already exists")
        plan, frozen, prepared, checked = freeze_contract(plan_path, args.independent_check)
        targets = frozen["nomination"]["targets"]
        require(len(targets) == args.expected_target_count and set(targets) == set(frozen["numerical_formats"]), "Exact frozen target count/format set differs")
        require(all(descriptor["numerical_format"] == frozen["numerical_formats"][key]
                    and key.rsplit(":", 1)[1] == descriptor["numerical_format"] for key, descriptor in targets.items()), "Target/format association differs")
        outputs, errors, pair_cache = {}, {}, {}
        for key, descriptor in targets.items():
            try:
                outputs[key] = check_target(key, descriptor, plan, frozen, prepared, pair_cache)
            except Exception as exc:
                errors[key] = {"type": type(exc).__name__, "message": str(exc)}
        result = {"complete": not errors and len(outputs) == len(targets), "targets_expected": len(targets),
                  "targets_passed": len(outputs), "errors": errors, "targets": outputs,
                  "same_checkpoint_format_pairs": format_pairs(outputs, pair_cache),
                  "plan_sha256": sha(plan_path), "freeze_sha256": plan["confirmation_freeze_sha256"],
                  "independent_data_check_sha256": sha(local(args.independent_check, DEST)),
                  "data_protocol_sha256": checked["data_protocol_sha256"],
                  "additional_strata_scope": "Presentation/position/context/benign category/raw length/windows are independent reporting derivatives from hash-bound prepared score metadata, not evaluator-saved fields.",
                  "evidence_boundary": "Saved reserved scores only after explicit coordinator release. No model load, inference, raw text decoding, tuning, threshold/format/seed rescue, protected project-final data or teacher use. Artifact/metadata association checks do not prove execution by syscall tracing."}
    result.update(cpu_wall_seconds=time.monotonic() - started, helper_sha256=sha(Path(__file__)), command=sys.argv)
    if args.output:
        output = local(args.output, DEST)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("complete", "cpu_wall_seconds", "targets_expected", "targets_passed", "errors", "self_test") if k in result}))
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
