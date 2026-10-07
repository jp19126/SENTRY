"""Combine existing Goal 4 evidence at L256/B1 without inventing integration costs.

Default emits the two uniform anchors for every active physical profile. An
optional sixteen-group precision JSON uses the same estimator and runs no model.
This is a component ledger, not an eligible Goal 5 complete-checking cost table.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from analyze_cached_linear import prediction
from build_linear_hls import active_linear_variants

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def relative(path):
    return str(path.relative_to(ROOT)).replace("\\", "/")


def names(config):
    return [f"layer{layer}.{group}" for layer in range(4)
            for group in config["quantization"]["groups_per_layer"]]


def matrices(precision):
    for layer in range(4):
        for group, k, n, count in (("qkv", 256, 256, 3),
                                  ("attention_output", 256, 256, 1),
                                  ("ffn_input", 256, 1024, 1),
                                  ("ffn_output", 1024, 256, 1)):
            for index in range(count):
                yield {"id": f"layer{layer}.{group}.{index}", "rows": 256,
                       "inner": k, "outputs": n,
                       "weight_bits": precision[f"layer{layer}.{group}"]}


def fixed_evidence():
    build = read(ROOT / "results/goal4/fixed_fp32/run.json")
    result = {"synthesis_recorded_utc": build["recorded_utc"],
              "service_schedule_revision": build["service_schedule_revision"],
              "service_calls": 411, "cycles": None, "source": None}
    for path in (ROOT / "results/goal4/fixed_fp32_rtl").glob("*/primary_l256/primary_costs.json"):
        item = read(path)
        if (item["synthesis_recorded_utc"] == build["recorded_utc"]
                and item["service_schedule_revision"] == build["service_schedule_revision"]):
            if item["call_count"] != 411 or len(item["terms"]) != 15:
                raise ValueError("FP32 primary schedule changed")
            observed = item["fp32_service_sum_cycles_by_rep"]
            if len(observed) != 2 or observed[0] != observed[1]:
                raise ValueError("Repeated full-call costs differ; inspect before aggregation")
            result.update(cycles=observed[0], source=relative(path),
                          memory_model=item["memory_model"],
                          boundary=item["cycle_boundary"],
                          axi_traffic_by_rep=item.get("fp32_service_axi_by_rep"))
    return result


def integer_evidence(profile, precision, variant):
    suffix = ("_static_bank_" + profile["id"] if variant == "weight_cache_static_bank"
              else "" if profile["id"] == "lanes16" else "_" + profile["id"])
    path = ROOT / f"results/goal4/cached_characterization{suffix}.json"
    result = {"profile": profile, "kernel_variant": variant, "cycles": None, "source": None,
              "reserved_validation_passed": False, "calls": []}
    if not path.is_file():
        return result
    item = read(path)
    synthesis = item["synthesis"]
    if item["kernel_variant"] != variant:
        raise ValueError("Selected source variant differs from frozen evidence")
    freeze = item.get("source_derived_complete_freeze")
    if not freeze:
        return result
    if (synthesis["profile"] != profile or freeze["synthesis_recorded_utc"]
            != synthesis["synthesis_recorded_utc"]):
        raise ValueError("Physical profile or synthesis/freeze association changed")
    boundaries = {p["boundary_cycles_relative_to_report_components"]
                  for p in freeze["predictions"].values()}
    if len(boundaries) != 1:
        raise ValueError("No common reviewed transaction-boundary correction")
    boundary = next(iter(boundaries))
    # Reproduce every frozen point with the existing source-derived predictor.
    for entry in item["points"]:
        p = prediction(entry["point"], synthesis)
        component = p["bounded_axi_adjusted_component_cycles"]
        if component is None:
            return result
        if component + boundary != freeze["predictions"][entry["point"]["id"]]["cycles"]:
            raise ValueError("Predictor no longer reproduces its frozen evidence")
    calls = []
    for point in matrices(precision):
        pred = prediction(point, synthesis)
        calls.append({**point, "cycles": pred["bounded_axi_adjusted_component_cycles"] + boundary,
                      "traffic": pred["traffic"]})
    reserved = [p for p in item["points"] if p["point"]["role"] == "reserved"]
    reserved_ok = bool(reserved) and all(
        p["observation"].get("status") == "measured_rtl"
        and p["observation"].get("complete_prediction_error_cycles") == [0, 0]
        and all(t.get("exact_output_passed") for t in p["observation"].get("transactions", []))
        and len(p["observation"].get("transactions", [])) == 2 for p in reserved)
    result.update(cycles=sum(p["cycles"] for p in calls), calls=calls,
                  source=relative(path), reserved_validation_passed=reserved_ok,
                  hls_resources=synthesis["estimated_resources"],
                  hls_estimated_period_ns=synthesis["estimated_clock_ns"],
                  synthesis_recorded_utc=synthesis["synthesis_recorded_utc"],
                  boundary="24 serialized full-tile formula evaluations using the frozen bounded-memory schedule; not a whole-detector measurement",
                  traffic={
                      "logical_activation_bytes": sum(p["traffic"]["logical_activation_bytes"] for p in calls),
                      "activation_axi_read_beats": sum(p["traffic"]["logical_activation_bytes"] for p in calls),
                      "activation_axi_full_beat_bytes": 4 * sum(p["traffic"]["logical_activation_bytes"] for p in calls),
                      "weight_preload_bytes": sum(p["traffic"]["source_weight_read_bytes"] for p in calls),
                      "output_bytes": sum(p["traffic"]["output_write_bytes"] for p in calls),
                      "basis": "Source-derived full-window counts using the observed single-byte activation requests on32-bit AXI and contiguous packed weight preload; no physical DDR observation or extra byte/bandwidth time added"})
    return result



def planned_integration(config, fixed):
    """Read the concrete fixed plan; these are counts, never measured latency."""
    path = ROOT / "results/goal4/internal_integration/command_plan.json"
    plan = read(path)
    count = plan["counts"]
    if (plan["identity"] != "bert_mini_l256_internal_plan_v1"
            or count["integer_calls"] != 24 or count["fp32_calls"] != 411
            or plan["fp32_build"]["synthesis_recorded_utc"] != fixed["synthesis_recorded_utc"]):
        raise ValueError("Integration plan no longer matches the measured fixed services")
    selected = active_linear_variants(config)
    if {p["profile"]: p["kernel_variant"] for p in plan["integer_builds"]} != selected:
        raise ValueError("Integration plan does not match the selected physical profiles")
    return {"source": relative(path), "identity": plan["identity"],
            "status": plan["status"], "arena": plan["arena"],
            "counts": count, "cycles": None,
            "memory_boundary": plan["memory_model"],
            "scope": "Relative buffer/command plan; no physical allocation or measured control/layout latency",
            "register_transactions": {
                "configuration_and_start_writes": count["service_argument_start_writes"],
                "completion_status_reads": count["completion_reads"],
                "interrupt_clear_writes": count["interrupt_clear_writes"],
                "initial_irq_enable_writes": count["interrupt_setup_writes_once_per_reset"],
                "basis": "Actual generated offsets: full argument programming plus start, AP_CTRL and status reads, one ISR clear per call; IRQ completion, no polling loop"}}



def measured_integration(config, planned):
    path = ROOT / config["hardware"]["internal_integration_result"]
    run = read(path)
    if not run.get("passed") or run["memory_mode"] != {
            "STALL": 0, "PIPELINED_READ": 1, "extra_DDR_delay": False}:
        raise ValueError("A passing nominal-memory integration record is required")
    if run["clock_ns"] != 1000 / config["hardware"]["clock_objective_mhz"]:
        raise ValueError("Integration and service clock objectives differ")
    plan = read(ROOT / planned["source"])
    executed_plan = read(path.parent / "command_plan.json")
    for field in ("identity", "arena", "precision", "actual_register_offsets", "buffers",
                  "matrices", "commands", "counts", "integer_builds", "fp32_build"):
        if plan[field] != executed_plan[field]:
            raise ValueError(f"Current integration plan differs from its executed {field}")
    for source in run["actual_control_sources"]:
        if (ROOT / source).read_text() != (path.parent / Path(source).name).read_text():
            raise ValueError("Generated control RTL differs from its executed source snapshot")
    for name in run["source_files"]:
        if (ROOT / "hardware" / name).read_text() != (path.parent / name).read_text():
            raise ValueError("Controller/mover source differs from its executed snapshot")
    # All selected profiles use the same generated control logic. No profile's
    # datapath latency is borrowed; only the common control overhead is shared.
    controls = [ROOT / entry["control_source"] for entry in plan["integer_builds"]]
    selected_control = ROOT / run["actual_control_sources"][0]
    if any(p.read_text() != selected_control.read_text() for p in controls):
        raise ValueError("Common integration overhead requires identical integer control RTL")
    if run["actual_control_sources"][1] != plan["fp32_build"]["control_source"]:
        raise ValueError("Integration FP32 control source differs from the plan")
    count = planned["counts"]
    valid = [(i, item) for i, item in enumerate(run["cases"])
             if item["complete_window"] and item["status"] == 0]
    if len(valid) < 2:
        raise ValueError("Repeated successful full control schedules are required")
    for index, item in valid:
        sampled = [p for p in run["sampled_stub_intervals"] if p["case_index"] == index]
        if (len(sampled) != 435 or {p["call_index"] for p in sampled} != set(range(435))
                or sum(p["latency_cycles"] for p in sampled) != item["stub_interval_cycles"]
                or any(p["done_cycle"]-p["start_cycle"] != p["latency_cycles"] for p in sampled)):
            raise ValueError("Incomplete or inconsistent sampled service substitution boundary")
        if (item["integer_calls"] != 24 or item["fp32_calls"] != 411
                or item["control_writes"] != count["service_argument_start_writes"] + count["interrupt_clear_writes"]
                or item["control_reads"] != count["completion_reads"]
                or item["read_beats"]*4 != count["movement_useful_read_bytes"] + count["control_data_read_bytes"]
                or item["write_beats"]*4 != count["movement_useful_write_bytes"]
                or item["inclusive_cycles"]-item["stub_interval_cycles"] != item["control_layout_cycles"]):
            raise ValueError("Observed control/layout counts differ from the fixed plan")
    cycles = {item["control_layout_cycles"] for _, item in valid}
    if len(cycles) != 1:
        raise ValueError("Control overhead differs across the successful fixed schedules")
    return {"source": relative(path), "control_layout_cycles": cycles.pop(),
            "successful_windows": len(valid), "memory_mode": run["memory_mode"],
            "read_bytes": valid[0][1]["read_beats"]*4,
            "write_bytes": valid[0][1]["write_beats"]*4,
            "source_identity": "fixed L256 controller/mover with actual generated control registers and explicit arithmetic completion stubs",
            "boundary": "sampled window accept-to-first-result edge difference minus every sampled accepted-service-start-to-done interval; legacy inclusive_cycles field adds no extra cycle",
            "scope": "Conditional sequential control/layout term; source-control equality supports all active profiles. Replacement with real service costs is analytical, not a full-detector measurement. Cold IRQ setup, physical stalls and host costs remain separate."}


def estimate(config, label, precision, profile, fixed, control):
    integer = integer_evidence(profile, precision, active_linear_variants(config)[profile["id"]])
    calls = list(matrices(precision))
    wide_elements = sum(p["inner"] * p["outputs"] for p in calls if p["weight_bits"] == 8)
    clock = config["hardware"]["clock_objective_mhz"]
    known_sum = (integer["cycles"] + fixed["cycles"]
                 if integer["cycles"] is not None and fixed["cycles"] is not None else None)
    internal_sum = known_sum + control["control_layout_cycles"] if known_sum is not None else None
    return {"candidate": label, "precision_map": precision, "profile_id": profile["id"],
            "w8_matrix_elements": wide_elements,
            "packed_encoder_weight_bytes": 1572864 + wide_elements // 2,
            "integer": integer, "fixed_fp32": fixed,
            "known_service_cycles": known_sum,
            "known_service_ms_at_clock_objective": known_sum / (clock * 1000) if known_sum is not None else None,
            "control_layout": control,
            "modeled_internal_cycles": internal_sum,
            "modeled_internal_ms_at_clock_objective": internal_sum / (clock * 1000) if internal_sum is not None else None,
            "estimated_complete_checking_ms": None, "feasible": None,
            "equation": "T_ms = host_preprocessing_ms + transport_sync_ms + (integer_cycles + fp32_service_cycles + measured_conditional_control_layout_cycles + additional_memory_wait_cycles)/(clock_MHz*1000)",
            "scope": "Known service subtotal under bounded AXI memory; all missing costs remain explicit, no measured board result"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision-map", type=Path)
    args = parser.parse_args()
    config = read(ROOT / "configs/project.json")
    if (config["measurement"]["primary_length"], config["measurement"]["primary_batch"]) != (256, 1):
        raise ValueError("This scoped ledger is for L256/B1 only")
    keys = names(config)
    if args.precision_map:
        source = args.precision_map if args.precision_map.is_absolute() else ROOT / args.precision_map
        candidates = [(source.stem, read(source))]
    else:
        candidates = [(f"uniform_w{bits}a8", dict.fromkeys(keys, bits)) for bits in (4, 8)]
    for _, precision in candidates:
        if set(precision) != set(keys) or any(type(b) is not int or b not in (4, 8) for b in precision.values()):
            raise ValueError("Expected exactly the sixteen configured W4/W8 group decisions")
    fixed = fixed_evidence()
    integration = planned_integration(config, fixed)
    control = measured_integration(config, integration)
    schedule = read(ROOT / "results/goal4/fixed_fp32/schedule_summary.json")
    primary = next(p for p in schedule["windows"] if p["length"] == 256)
    profiles = config["hardware"]["hls"]["cached_linear"]["profiles"]
    output = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
              "status": "component_ledger_incomplete_integration", "search_ready": False,
              "primary_length": 256, "primary_batch": 1,
              "clock_objective_mhz": config["hardware"]["clock_objective_mhz"],
              "rows": [estimate(config, label, precision, p, fixed, control)
                       for label, precision in candidates for p in profiles],
              "fp32_useful_service_bytes": primary["useful_service_source_bytes"],
              "previous_analytical_layout": primary["analytical_layout"],
              "planned_integration": integration,
              "measured_control_layout": control,
              "planned_layout_useful_bytes": (integration["counts"]["movement_useful_read_bytes"]
                                                + integration["counts"]["movement_useful_write_bytes"]),
              "planned_control_input_useful_bytes": integration["counts"]["control_data_read_bytes"],
              "register_transactions": integration["register_transactions"],
              "missing_terms": ["additional physical DDR/NoC latency and arbitration relative to the existing bounded memory model",
                                "deployed host preprocessing, transport and synchronization",
                                "integrated system resources, timing and bandwidth eligibility",
                                "hardware-matched numerical reference and development protection scores"],
              "accounting_note": "Do not add bytes/bandwidth on top of already charged nominal memory service. Useful bytes and AXI beats differ. Partial HLS loop spans are not substituted for complete-call cycles. No zero-cost assumption is made for any missing term. Concrete plan counts supersede the earlier abstract layout assumptions (32-bit packed IDs and cached mask instead of repeated FP32 mask reads); host packing remains a missing cost."}
    path = ROOT / "results/goal4/detector_cost_ledger.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": relative(path), "search_ready": False,
                      "rows": [{"candidate": p["candidate"], "profile": p["profile_id"],
                                "integer_cycles": p["integer"]["cycles"],
                                "fp32_cycles": p["fixed_fp32"]["cycles"],
                                "known_service_ms": p["known_service_ms_at_clock_objective"],
                                "modeled_internal_ms": p["modeled_internal_ms_at_clock_objective"]}
                               for p in output["rows"]]}, indent=2))


if __name__ == "__main__":
    main()
