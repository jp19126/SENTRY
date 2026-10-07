"""Goal 5 A/B evolutionary adapter core; no model execution or campaign CLI.

Default: report missing Goal 4 costs/caps. --score-bringup recomputes metrics
from the two existing PTQ anchor score files without consulting any search map.
A caller may use EvolutionAdapter only after real common costs/caps and an
approved arithmetic/execution score identity are ready. No training is dispatched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gate_eval import evaluate_documents, fit_threshold, load_config
from build_linear_hls import active_linear_profiles, active_linear_variants

SPLITS = ("search_temporary_threshold", "search_candidate_scoring")
SCHEME = "static_symmetric_encoder_linear_w4w8a8_fp32_other_v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def resolved(value):
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def groups(config):
    q = config["quantization"]
    names = [f"layer{layer}.{group}" for layer in range(q["encoder_layers"])
             for group in q["groups_per_layer"]]
    if len(names) != 16 or q["groups_per_layer"] != [
            "qkv", "attention_output", "ffn_input", "ffn_output"]:
        raise ValueError("This adapter is scoped to the declared sixteen fixed BERT groups.")
    return names


def precision_tuple(precision, config):
    names = groups(config)
    if set(precision) != set(names):
        raise ValueError("Precision map must contain exactly the sixteen declared groups.")
    result = tuple(precision[name] for name in names)
    if any(type(bit) is not int or bit not in (4, 8) for bit in result):
        raise ValueError("Only W4/W8 whole-group decisions are allowed.")
    return result


def protection_eligible(metrics, floating_recall, config):
    rule = config["evaluation"]
    return (metrics["fpr"] is not None and metrics["recall"] is not None
            and metrics["fpr"] <= rule["fpr_target"]
            and metrics["recall"] >= floating_recall - rule["max_recall_loss_absolute"])


def approved_score_identity(config, *, floating=False):
    """Separate arithmetic identities, with one explicitly approved execution backend."""
    keys = ("approved_score_identity", "approved_floating_score_identity")
    fields = ("arithmetic_identity", "execution_identity")
    records = {key: config["methods"].get(key) or {} for key in keys}
    for key, record in records.items():
        if any(not isinstance(record.get(field), str) or not record[field].strip() for field in fields):
            raise RuntimeError(f"Score reuse unavailable: methods.{key} arithmetic/execution identity is unresolved.")
    if records[keys[0]]["execution_identity"] != records[keys[1]]["execution_identity"]:
        raise ValueError("Approved quantized and floating scores must use the same explicit execution identity.")
    selected = records[keys[1] if floating else keys[0]]
    return {field: selected[field] for field in fields}


def require_execution_identity(identity, config, *, floating=False):
    approved = approved_score_identity(config, floating=floating)
    if not isinstance(identity, dict) or any(identity.get(k) != v for k, v in approved.items()):
        raise ValueError("Saved arithmetic/execution identity differs or is missing; historical SDPA scores cannot stand in for ordered scores.")


def validate_score_identity(identity, config, *, qat=False, floating=False):
    """Ordinary version/path associations; no model load, hashes or score relabelling."""
    require_execution_identity(identity, config, floating=floating)
    run = read_json(ROOT / "results/goal3/run.json")
    if resolved(identity.get("checkpoint", "")) != resolved(run["selected_checkpoint"]):
        raise ValueError("Scores must use the selected common floating-checkpoint source.")
    for field in ("windowing", "evaluation", "data"):
        if identity.get(field) != config[field]:
            raise ValueError(f"Cached {field} differs from the shared protocol.")
    if run["config"]["data"] != config["data"]:
        raise ValueError("Data configuration changed; existing split evidence is not reusable.")
    if floating:
        if identity.get("qat") is not False:
            raise ValueError("Floating recall reference must be explicitly untrained after selection.")
        return
    if identity.get("qat") is not qat:
        raise ValueError("Score stage differs from the requested PTQ/QAT stage.")
    if identity.get("scheme_id") != SCHEME or run["format"]["scheme_id"] != SCHEME:
        raise ValueError("Numerical format differs from the common search computation.")
    calibration_path = resolved(run["calibration_file"])
    if (resolved(identity.get("calibration_file", "")) != calibration_path
            or not calibration_path.is_file()):
        raise ValueError("Scores must identify the existing common train-only calibration file.")
    policy = {field: run["format"][field] for field in ("weight_scale", "activation_scale")}
    if identity.get("frozen_scale_policy") != policy:
        raise ValueError("Scores must retain the common frozen weight/activation scale policy.")
    precision_tuple(identity["precision_map"], config)
    if qat:
        if config["methods"]["qat_epochs"] != 1:
            raise ValueError("The common QAT allowance is exactly one epoch.")
        checkpoint = resolved(identity.get("scored_checkpoint", ""))
        metadata = read_json(checkpoint / "quantization.json")
        training = metadata["training"]
        if (metadata["scheme_id"] != SCHEME
                or metadata["precision_map"] != identity["precision_map"]
                or metadata["calibration"] != read_json(calibration_path)
                or metadata["format"] != run["format"]
                or training["epochs"] != 1
                or training["seed"] != config["seed"]
                or resolved(training["from_floating_checkpoint"]) != resolved(run["selected_checkpoint"])
                or training["scales"] != "frozen original FP-checkpoint W4/W8 scales and shared A8 calibration"):
            raise ValueError("QAT checkpoint must preserve the one-epoch common source, map and frozen scales.")


def provisional_rank(metrics, floating_recall, latency, key, config):
    """Nonnegative probability-point violations; ranking never grants feasibility."""
    fpr, recall = metrics["fpr"], metrics["recall"]
    if any(v is None or not math.isfinite(v) or not 0 <= v <= 1 for v in (fpr, recall)):
        raise ValueError("Provisional ranking requires finite development FPR and recall.")
    rule = config["evaluation"]
    excess = max(0.0, fpr - rule["fpr_target"])
    deficit = max(0.0, floating_recall - rule["max_recall_loss_absolute"] - recall)
    return (max(excess, deficit), excess + deficit, latency, -recall, key)


def post_qat_rank(arm, score, floating_recall, latency, key, config):
    """Return None for strict B rejection; A retains its conventional criterion."""
    if arm == "B" and not protection_eligible(score["protection"], floating_recall, config):
        return None
    return ((-score["ordinary_0_5"]["accuracy"], latency, key) if arm == "A"
            else (latency, -score["protection"]["recall"], key))


def ranked_pool(arm, consulted, cap, floating_recall, config, *, fill_provisional=False):
    """A accuracy; B strict first, with explicitly provisional physical/cap fits."""
    strict, provisional = [], []
    for key, candidate in consulted.items():
        score = candidate["score"]
        fits = [p for p in candidate["profiles"] if p["estimated_complete_checking_ms"] <= cap]
        if score is None or not fits:
            continue
        chosen = min(fits, key=lambda p: (p["estimated_complete_checking_ms"], p["profile_id"]))
        latency = chosen["estimated_complete_checking_ms"]
        eligible = protection_eligible(score["protection"], floating_recall, config)
        entry = {"precision_tuple": key, "candidate": candidate, **chosen,
                 "protection_eligible_pre_qat": eligible,
                 "provisional": arm == "B" and not eligible}
        if arm == "A":
            rank = (-score["ordinary_0_5"]["accuracy"], latency, key)
        elif eligible:
            rank = (latency, -score["protection"]["recall"], key)
        else:
            rank = provisional_rank(score["protection"], floating_recall, latency, key, config)
        entry["rank"] = rank
        (provisional if entry["provisional"] else strict).append(entry)
    strict.sort(key=lambda e: e["rank"])
    provisional.sort(key=lambda e: e["rank"])
    return strict + provisional if fill_provisional or not strict else strict


def recompute_metrics(folder, config):
    """Read document maxima already scored by the shared full-network evaluator."""
    threshold_rows, scoring_rows = [
        rows(folder / f"{split}_documents.jsonl") for split in SPLITS]
    benign = [row["risk_score"] for row in threshold_rows if row["label"] == 0]
    threshold = fit_threshold(benign, config["evaluation"]["fpr_target"])
    scores = [row["risk_score"] for row in scoring_rows]
    labels = [row["label"] for row in scoring_rows]
    return {
        "temporary_threshold": threshold,
        "threshold_benign_count": len(benign),
        "protection": evaluate_documents(scores, labels, threshold),
        "ordinary_0_5": evaluate_documents(scores, labels, 0.5),
    }


def load_score(folder, config, floating_recall, *, qat=False):
    """Reuse matching full-network PTQ or explicitly requested one-epoch QAT scores."""
    record = read_json(folder / "candidate.json")
    identity = record["identity"]
    validate_score_identity(identity, config, qat=qat)
    precision = precision_tuple(identity["precision_map"], config)
    metrics = recompute_metrics(folder, config)
    prior = record["metrics"]
    if (metrics["temporary_threshold"] != prior["primary"]["threshold"]
            or metrics["protection"] != prior["primary"]["candidate_scoring"]
            or metrics["ordinary_0_5"] != prior["ordinary_threshold_0_5"]):
        raise ValueError("Saved document rows and saved candidate metrics disagree.")
    return {
        "candidate_name": record["name"], "precision_map": identity["precision_map"],
        "precision_tuple": precision, "identity": identity, **metrics,
        "protection_eligible": protection_eligible(metrics["protection"], floating_recall, config),
        "source": str(folder.relative_to(ROOT)).replace("\\", "/"),
        "score_reused": True, "new_scoring_seconds": 0,
        "new_training_seconds": 0,
    }


def readiness(config):
    methods = config["methods"]
    settings = methods["published_baseline"]["project_search_defaults"]
    missing = []
    for key in ("approved_score_identity", "approved_floating_score_identity"):
        record = methods.get(key) or {}
        if any(not isinstance(record.get(field), str) or not record[field].strip()
               for field in ("arithmetic_identity", "execution_identity")):
            missing.append(f"methods.{key}: approved arithmetic_identity and execution_identity")
    # A mismatched pair is invalid rather than an implicit cross-backend approval.
    if not missing:
        approved_score_identity(config)
    value = methods.get("cost_table_path")
    if not value or not resolved(value).is_file():
        missing.append("methods.cost_table_path: existing validated Goal 4 complete-checking cost table")
    caps = settings.get("latency_cap_values")
    if not caps:
        missing.append("methods.published_baseline.project_search_defaults.latency_cap_values")
    elif (not isinstance(caps, list) or len(caps) > settings["latency_cap_count_max"]
          or any(not isinstance(cap, (int, float)) or isinstance(cap, bool)
                 or not math.isfinite(cap) or cap <= 0 for cap in caps)
          or caps != sorted(set(caps))):
        raise ValueError("Frozen caps must be positive, sorted, distinct and at most four.")
    if settings.get("latency_cap_unit") != "ms":
        missing.append("methods.published_baseline.project_search_defaults.latency_cap_unit = ms")
    for field in ("allocated_board", "part", "interface", "clock_objective_mhz",
                  "resource_limits", "tool_versions", "execution_host"):
        if not config["hardware"].get(field):
            missing.append("hardware." + field)
    if not active_linear_profiles(config):
        missing.append("hardware.hls: active common realizable profiles")
    return missing


class CostTable:
    """Lookup of actual Goal 4 estimator outputs, never invented per-bit costs."""

    def __init__(self, config):
        missing = readiness(config)
        if missing:
            raise RuntimeError("Search unavailable: " + "; ".join(missing))
        self.config = config
        self.record = table = read_json(resolved(config["methods"]["cost_table_path"]))
        if (table["latency_unit"] != "ms"
                or table["latency_scope"] != "full_detector_host_to_decision"
                or table["detector_execution"] != "fpga"):
            raise ValueError("Costs must cover full FPGA detector checking and host overhead.")
        hardware = config["hardware"]
        for field in ("part", "clock_objective_mhz", "interface", "resource_limits"):
            if table[field] != hardware[field]:
                raise ValueError(f"Cost-table {field} differs from the common allocation.")
        for field in ("primary_length", "primary_batch"):
            if table[field] != config["measurement"][field]:
                raise ValueError("Cost table uses a different primary design point.")
        evidence = table["source_reports"]
        if (not evidence or any(not resolved(path).is_file() for path in evidence)
                or not resolved(table["validation_report"]).is_file()):
            raise ValueError("Actual synthesis and estimator-validation reports are required.")
        if table["frozen_latency_caps_ms"] != config["methods"]["published_baseline"]["project_search_defaults"]["latency_cap_values"]:
            raise ValueError("Caps must match the values frozen with the common Goal 4 table.")
        profiles = table["profiles"]
        if not 1 <= len(profiles) <= config["methods"]["max_engine_profiles"]:
            raise ValueError("The common physical profile count is outside the allowance.")
        self.profiles = {item["id"]: item for item in profiles}
        if len(self.profiles) != len(profiles):
            raise ValueError("Duplicate physical profile IDs.")
        configured = {item["id"]: item for item in active_linear_profiles(config)}
        if table.get("profile_kernel_variants") != active_linear_variants(config):
            raise ValueError("Cost table must preserve each active physical profile source variant.")
        if set(self.profiles) != set(configured):
            raise ValueError("Cost table must expose the same common physical profiles to A/B.")
        if any(self.profiles[name]["physical_profile"] != value for name, value in configured.items()):
            raise ValueError("Physical lane/tile/buffer profiles differ from the costed implementation.")
        self.lookup = {}
        for row in table["candidates"]:
            key = precision_tuple(row["precision_map"], config)
            if key in self.lookup:
                raise ValueError("Duplicate precision-map cost rows.")
            self.lookup[key] = row["profiles"]
        self.lookups = 0

    def feasible_profiles(self, precision):
        key = precision_tuple(precision, self.config)
        if key not in self.lookup:
            raise RuntimeError("Goal 4 estimator has not supplied this proposed map's costs.")
        entries = self.lookup[key]
        if {item["profile_id"] for item in entries} != set(self.profiles) or len(entries) != len(self.profiles):
            raise ValueError("Every map must consider every common profile, including infeasible ones.")
        result = []
        for item in entries:
            self.lookups += 1
            profile = self.profiles[item["profile_id"]]
            limits = self.config["hardware"]["resource_limits"]
            used = profile["resources"]
            if set(used) != set(limits):
                raise ValueError("Profile resource accounting must cover every allocated limit.")
            values = list(used.values()) + list(limits.values())
            if any(not isinstance(value, (int, float)) or not math.isfinite(value)
                   or value < 0 for value in values):
                raise ValueError("Resources require actual finite nonnegative values.")
            resource_ok = all(used[name] <= limits[name] for name in limits)
            if (not resource_ok or profile["timing_met"] is not True
                    or profile["bandwidth_feasible"] is not True or item["feasible"] is not True):
                continue
            latency = item["estimated_complete_checking_ms"]
            if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0:
                raise ValueError("Complete-checking estimates must be actual positive table values.")
            result.append({"profile_id": item["profile_id"],
                           "estimated_complete_checking_ms": latency})
        return sorted(result, key=lambda item: (item["estimated_complete_checking_ms"], item["profile_id"]))


class EvolutionAdapter:
    """One A or B arm with explicit consult accounting; no scoring side effects."""

    def __init__(self, arm, config, costs, floating_recall, *, floating_reference_identity=None):
        if arm not in ("A", "B"):
            raise ValueError("Only A/B exist; C remains unresolved.")
        # Recheck the gate even if callers pass an existing CostTable instance.
        if readiness(config) or not isinstance(costs, CostTable):
            raise RuntimeError("Real common cost table and frozen latency caps are required.")
        validate_score_identity(floating_reference_identity, config, floating=True)
        if not isinstance(floating_recall, (int, float)) or not math.isfinite(floating_recall) or not 0 <= floating_recall <= 1:
            raise ValueError("Floating reference recall must be a finite development rate.")
        self.arm, self.config, self.costs = arm, config, costs
        self.floating_recall = floating_recall
        self.settings = config["methods"]["published_baseline"]["project_search_defaults"]
        self.names = groups(config)
        self.rng = random.Random(config["seed"])
        self.consulted = {}
        self.initialized = False
        self.generations = 0
        self.hardware_lookups = 0
        self.duplicate_only_batches = 0
        self._qat_shortlist = None

    def as_map(self, precision):
        return dict(zip(self.names, precision))

    def random_map(self):
        return tuple(self.rng.choice((4, 8)) for _ in self.names)

    def crossover(self, first, second):
        child = []
        for start in range(0, 16, 4):
            parent = self.rng.choice((first, second))
            child.extend(parent[start:start + 4])
        return tuple(child)

    def mutate(self, parent):
        probability = self.settings["mutation_probability_per_group"]
        if not 0 < probability <= 1:
            raise ValueError("Mutation probability must be positive and at most one.")
        while True:
            child = tuple(12 - bit if self.rng.random() < probability else bit for bit in parent)
            if child != parent:
                return child

    def elite_pools(self):
        return {cap: ranked_pool(self.arm, self.consulted, cap, self.floating_recall, self.config)
                [:self.settings["elites_per_latency_cap"]]
                for cap in self.settings["latency_cap_values"]}

    def qat_shortlist(self):
        """Freeze at most two unique maps; no training or new logical consults.

        Strict B maps rank first; remaining slots may hold explicitly provisional
        cap/resource-fitting maps. Uniform anchors may reuse their common QAT run.
        """
        if self._qat_shortlist is None:
            if self.config["methods"]["qat_epochs"] != 1:
                raise ValueError("The common QAT allowance is exactly one epoch.")
            limit = self.config["methods"]["max_shortlist_per_method"]
            if type(limit) is not int or not 0 <= limit <= 2:
                raise ValueError("At most two unique maps may be shortlisted per arm.")
            ranked = ranked_pool(self.arm, self.consulted, max(self.settings["latency_cap_values"]),
                                 self.floating_recall, self.config, fill_provisional=True)
            self._qat_shortlist = tuple(entry["precision_tuple"] for entry in ranked[:limit])
        ranked = {e["precision_tuple"]: e for e in ranked_pool(
            self.arm, self.consulted, max(self.settings["latency_cap_values"]),
            self.floating_recall, self.config, fill_provisional=True)}
        return [dict(ranked[key], common_qat_epochs=1,
                     uniform_anchor=key in ((4,) * 16, (8,) * 16)) for key in self._qat_shortlist]

    def post_qat_selection(self, scores):
        """Select only after every frozen shortlisted map has matching QAT evidence.

        A follows accuracy and reports protection. B's final eligibility is strict;
        provisional ranking never makes an ineligible trained result a finalist.
        Common uniform-anchor evaluations outside this shortlist remain separate.
        """
        shortlist = {e["precision_tuple"]: e for e in self.qat_shortlist()}
        seen, evaluated, accepted = set(), [], []
        for score in scores:
            key = precision_tuple(score["precision_map"], self.config)
            if key not in shortlist or key in seen:
                raise ValueError("Post-QAT scores must be unique maps from the frozen shortlist.")
            validate_score_identity(score["identity"], self.config, qat=True)
            if precision_tuple(score["identity"]["precision_map"], self.config) != key or tuple(score["precision_tuple"]) != key:
                raise ValueError("QAT score and identity precision maps disagree.")
            seen.add(key)
            eligible = protection_eligible(score["protection"], self.floating_recall, self.config)
            latency = shortlist[key]["estimated_complete_checking_ms"]
            rank = post_qat_rank(self.arm, score, self.floating_recall, latency, key, self.config)
            entry = {"precision_tuple": key, "precision_map": score["precision_map"],
                     "profile_id": shortlist[key]["profile_id"],
                     "estimated_complete_checking_ms": shortlist[key]["estimated_complete_checking_ms"],
                     "score": score, "protection_eligible_post_qat": eligible,
                     "accepted": rank is not None, "provisional": False}
            evaluated.append(entry)
            if entry["accepted"]:
                accepted.append((rank, entry))
        missing = sorted(set(shortlist) - seen)
        accepted.sort(key=lambda item: item[0])
        return {"arm": self.arm, "state": "awaiting_qat_scores" if missing else "complete",
                "missing_precision_tuples": missing, "evaluated": evaluated,
                "selected": accepted[0][1] if accepted and not missing else None,
                "protection_rule_relaxed": False, "new_training_calls": 0}

    def propose(self):
        """Return new maps, or [] only when the logical allowance is exhausted.

        Duplicate-only draws retry the same proposal operators; they neither
        consult maps nor consume additional scoring allowance.
        """
        if self._qat_shortlist is not None:
            raise RuntimeError("QAT shortlist is frozen; further search proposals are closed.")
        remaining = self.config["methods"]["max_unique_maps_per_method"] - len(self.consulted)
        if remaining <= 0:
            return []
        while True:
            if (ROOT / "results/goal2/pause.request").exists():
                raise RuntimeError("User pause marker is present.")
            if not self.initialized:
                proposed = [(8,) * 16, (4,) * 16]
                while len(proposed) < self.settings["initial_population"]:
                    value = self.random_map()
                    if value not in proposed:
                        proposed.append(value)
                self.initialized = True
            else:
                self.generations += 1
                elites = sorted({item["precision_tuple"]
                                 for pool in self.elite_pools().values() for item in pool})
                if elites:
                    proposed = [self.crossover(self.rng.choice(elites), self.rng.choice(elites))
                                for _ in range(self.settings["crossover_children_per_generation"])]
                    proposed += [self.mutate(self.rng.choice(elites))
                                 for _ in range(self.settings["mutation_children_per_generation"])]
                else:
                    count = (self.settings["crossover_children_per_generation"]
                             + self.settings["mutation_children_per_generation"])
                    proposed = [self.random_map() for _ in range(count)]
            unique = list(dict.fromkeys(value for value in proposed if value not in self.consulted))
            if unique:
                return [self.as_map(value) for value in unique[:remaining]]
            self.duplicate_only_batches += 1

    def consider(self, precision, score):
        """Charge an arm when it consults a map, including a reused cached score.

        Physically rejected maps may use score=None. Missing cost rows raise
        before consulting/scoring; they must be supplied by the real estimator.
        """
        key = precision_tuple(precision, self.config)
        if key in self.consulted:
            return self.consulted[key]
        if self._qat_shortlist is not None:
            raise RuntimeError("QAT shortlist is frozen; further logical consults are closed.")
        if score is not None:
            validate_score_identity(score["identity"], self.config)
            if precision_tuple(score["identity"]["precision_map"], self.config) != key:
                raise ValueError("PTQ score and identity precision maps disagree.")
        if len(self.consulted) >= self.config["methods"]["max_unique_maps_per_method"]:
            raise RuntimeError("This arm's unique-map allowance is exhausted.")
        before = self.costs.lookups
        profiles = self.costs.feasible_profiles(precision)
        self.hardware_lookups += self.costs.lookups - before
        if profiles and (score is None or tuple(score["precision_tuple"]) != key):
            raise ValueError("A realizable map requires its complete-network saved score.")
        candidate = {"precision_map": precision, "profiles": profiles, "score": score,
                     "protection_feasible": bool(score and protection_eligible(
                         score["protection"], self.floating_recall, self.config)),
                     "physical_rejection": not profiles}
        self.consulted[key] = candidate
        return candidate

    def accounting(self):
        return {"arm": self.arm, "logical_unique_maps": len(self.consulted),
                "cheap_hardware_profile_lookups": self.hardware_lookups,
                "duplicate_only_proposal_batches": self.duplicate_only_batches,
                "new_model_scoring_calls": 0, "new_training_calls": 0,
                "note": "Adapter reuses saved scores only; caller must record any future actual scoring/training."}


def bringup(config):
    floating = read_json(ROOT / "results/goal3/floating_reference.json")
    validate_score_identity(floating.get("identity"), config, floating=True)
    floating_metrics = recompute_metrics(resolved(floating["source_scores"]), config)
    if floating_metrics["protection"] != floating["metrics"]["primary"]["candidate_scoring"]:
        raise ValueError("Floating-reference scores and saved metrics disagree.")
    recall = floating_metrics["protection"]["recall"]
    anchors = [load_score(ROOT / "results/goal3" / name, config, recall)
               for name in ("ptq_w8_a8", "ptq_w4_a8")]
    return {"state": "saved_score_adapter_bringup_only", "search_run": False,
            "floating_reference": floating_metrics, "candidates": anchors,
            "logical_comparison_maps_consulted": {"A": 0, "B": 0, "C": 0},
            "new_model_scoring_calls": 0, "new_training_calls": 0,
            "hardware_lookup_count": 0, "selected_candidates": None, "C": None,
            "note": "No evolutionary arm constructed, proposals generated or comparison outcomes selected."}


def adapter_check(config):
    """One deterministic in-memory protocol check using existing local metric records.

    Fixture costs are ordering sentinels, never a hardware table or campaign score.
    Historical records are NOT upgraded to an approved identity or saved anew.
    """
    from copy import deepcopy
    records = {name: read_json(ROOT / "results/goal3" / name / "candidate.json")
               for name in ("ptq_w8_a8", "ptq_w4_a8", "qat_w8_a8", "qat_w4_a8")}
    def metric_record(record):
        return {"protection": record["metrics"]["primary"]["candidate_scoring"],
                "ordinary_0_5": record["metrics"]["ordinary_threshold_0_5"]}
    recall = read_json(ROOT / "results/goal3/floating_reference.json")["metrics"]["primary"]["candidate_scoring"]["recall"]
    fixture = deepcopy(config)
    fixture["methods"]["approved_score_identity"] = {
        "arithmetic_identity": "fixture_ordered_arithmetic_only",
        "execution_identity": "fixture_ordered_cpu_only"}
    fixture["methods"]["approved_floating_score_identity"] = {
        "arithmetic_identity": "fixture_ordered_floating_arithmetic_only",
        "execution_identity": "fixture_ordered_cpu_only"}
    floating_identity = dict(fixture["methods"]["approved_floating_score_identity"])
    unresolved_floating = deepcopy(fixture)
    unresolved_floating["methods"]["approved_floating_score_identity"] = None
    mismatched_backends = deepcopy(fixture)
    mismatched_backends["methods"]["approved_floating_score_identity"]["execution_identity"] = "fixture_cuda_only"
    rejected = []
    for label, identity, settings, expected, floating in (
        ("unresolved_approval", records["ptq_w8_a8"]["identity"],
         dict(config, methods=dict(config["methods"], approved_score_identity=None)), RuntimeError, False),
        ("missing_historical_identity", records["ptq_w8_a8"]["identity"], fixture, ValueError, False),
        ("explicit_sdpa_mismatch", dict(records["ptq_w8_a8"]["identity"],
         arithmetic_identity="historical_sdpa", execution_identity="fixture_ordered_cpu_only"), fixture, ValueError, False),
        ("backend_mismatch", {"arithmetic_identity": "fixture_ordered_arithmetic_only",
         "execution_identity": "fixture_cuda_only"}, fixture, ValueError, False),
        ("unresolved_floating_approval", floating_identity, unresolved_floating, RuntimeError, True),
        ("wrong_floating_arithmetic", fixture["methods"]["approved_score_identity"], fixture, ValueError, True),
        ("historical_floating_identity_missing", {}, fixture, ValueError, True),
        ("cross_backend_approval_pair", fixture["methods"]["approved_score_identity"], mismatched_backends, ValueError, False)):
        try:
            require_execution_identity(identity, settings, floating=floating)
        except expected:
            rejected.append(label)
        else:
            raise AssertionError("Identity check did not reject " + label)
    require_execution_identity(floating_identity, fixture, floating=True)
    require_execution_identity(fixture["methods"]["approved_score_identity"], fixture)
    consulted = {}
    for name, latency in (("ptq_w8_a8", 1.0), ("ptq_w4_a8", 2.0)):
        record = records[name]
        key = precision_tuple(record["identity"]["precision_map"], config)
        consulted[key] = {"precision_map": record["identity"]["precision_map"],
                          "score": metric_record(record), "profiles": [{
                              "profile_id": "fixture_only_not_measured",
                              "estimated_complete_checking_ms": latency}]}
    b = ranked_pool("B", consulted, 2.0, recall, config)
    a = ranked_pool("A", consulted, 2.0, recall, config)
    assert [e["precision_tuple"][0] for e in b] == [4, 8] and all(e["provisional"] for e in b)
    assert [e["precision_tuple"][0] for e in a] == [8, 4]
    assert [e["precision_tuple"][0] for e in ranked_pool("B", consulted, 1.0, recall, config)] == [8]
    # Exercise the actual bounded shortlist method without constructing a CostTable.
    probe = object.__new__(EvolutionAdapter)
    probe.arm, probe.config, probe.floating_recall, probe.consulted = "B", config, recall, consulted
    probe.settings = {"latency_cap_values": [2.0]}
    probe._qat_shortlist = None
    short = probe.qat_shortlist()
    assert len(short) == 2 and len({e["precision_tuple"] for e in short}) == 2
    assert [e["precision_tuple"] for e in probe.qat_shortlist()] == [e["precision_tuple"] for e in short]
    strict_fixture = deepcopy(consulted)
    strict_fixture[(8,) * 16]["score"] = metric_record(records["qat_w8_a8"])
    strict_pool = ranked_pool("B", strict_fixture, 2.0, recall, config)
    assert len(strict_pool) == 1 and strict_pool[0]["precision_tuple"] == (8,) * 16
    mixed_short = ranked_pool("B", strict_fixture, 2.0, recall, config, fill_provisional=True)
    assert [e["provisional"] for e in mixed_short] == [False, True]
    post = {name: post_qat_rank("B", metric_record(records[name]), recall, 1.0,
                               precision_tuple(records[name]["identity"]["precision_map"], config), config) is not None
            for name in ("qat_w8_a8", "qat_w4_a8")}
    assert post == {"qat_w8_a8": True, "qat_w4_a8": False}
    return {"state": "offline_adapter_check_passed", "identity_rejections": rejected,
            "distinct_arithmetic_same_execution_fixture_accepted": True,
            "A_fixture_order": [8, 4], "B_provisional_fixture_order": [4, 8],
            "unique_shortlist_maps": len(short), "strict_pool_excludes_provisional": True,
            "post_qat_strict_rule_on_historical_metric_fixtures": post,
            "fixture_hardware_values_are_measurements": False,
            "approved_historical_scores": 0, "campaign_maps": 0,
            "new_model_scoring_calls": 0, "new_training_calls": 0,
            "note": "Uses four saved candidate metric summaries only; no score rows/models, CostTable, campaign, config or historical artifact writes."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-bringup", action="store_true",
                        help="Recompute metrics from completed PTQ anchors only; no search.")
    parser.add_argument("--adapter-check", action="store_true",
                        help="Run the one bounded local-record protocol/identity check; stdout only.")
    args = parser.parse_args()
    if args.adapter_check and args.score_bringup:
        parser.error("Choose one offline check.")
    config = load_config()
    if args.adapter_check:
        print(json.dumps(adapter_check(config), indent=2))
        return 0
    record = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
              "state": "blocked" if readiness(config) else "configured_not_run",
              "missing": readiness(config), "search_run": False, "C": None}
    if args.score_bringup:
        record["score_adapter"] = bringup(config)
    output = ROOT / "results/goal5" / ("score_adapter_bringup.json" if args.score_bringup else "readiness.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": record["state"], "missing": record["missing"],
                      "search_run": False, "report": str(output)}, indent=2))
    return 0 if args.score_bringup else (2 if record["missing"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
