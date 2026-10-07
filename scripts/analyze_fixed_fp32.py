"""Read this study's actual FP32 HLS reports; emit partial analytical accounting.
No vendor execution, coefficient fitting, DDR assumption or full-runtime claim.
"""
from pathlib import Path
import json
import math
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "fixed_fp32_service_Pipeline_"
REVISION = "panel_cache_batched_attention_v1"
CURRENT = {
    "dot_row": "dot_load_row",
    "dot_panel": ("dot_load_panel", "dot_panel_outputs_dot_load_panel"),
    "dot_compute": "dot_reduce",
    "ln_load": "ln_load_sum", "ln_variance": "ln_variance", "ln_affine": "ln_normalize",
    "softmax_load": "softmax_scale_max", "softmax_exp": "softmax_exp_sum",
    "softmax_store": "softmax_normalize", "quantize": "quantize_elements",
    "rescale": "rescale_rows_rescale_channels", "embedding": "embedding_elements",
    "residual": "residual_elements", "gelu": "gelu_elements", "tanh": "tanh_elements",
}
VECTOR = {
    "dot_row": "dot_load_row", "dot_weights": "dot_load_weights",
    "dot_zero": "dot_zero_tail", "dot_compute": "VITIS_LOOP_187_4",
    "ln_load": "VITIS_LOOP_216_8", "ln_variance": "VITIS_LOOP_225_9",
    "ln_affine": "VITIS_LOOP_235_10", "softmax_load": "VITIS_LOOP_250_12",
    "softmax_exp": "VITIS_LOOP_259_13", "softmax_store": "VITIS_LOOP_264_14",
    "quantize": "quantize_elements", "rescale": "rescale_rows_rescale_channels",
    "embedding": "embedding_elements", "residual": "residual_elements",
    "gelu": "gelu_elements", "tanh": "tanh_elements",
}
SCHEDULED = dict(VECTOR)
SCHEDULED["rescale"] = "rescale_channels"
BEFORE = {
    "dot_row": "VITIS_LOOP_150_2", "dot_weights_four": "VITIS_LOOP_156_4",
    "dot_compute": "VITIS_LOOP_169_7", "ln_load": "VITIS_LOOP_198_11",
    "ln_variance": "VITIS_LOOP_207_12", "ln_affine": "VITIS_LOOP_217_13",
    "softmax_load": "VITIS_LOOP_232_15", "softmax_exp": "VITIS_LOOP_241_16",
    "softmax_store": "VITIS_LOOP_246_17", "mixed_vectors": "VITIS_LOOP_252_18",
}


def number(text):
    try:
        value = int(text)
        return value if value >= 0 else None
    except (TypeError, ValueError):
        return None

def bounds(element):
    if element is None:
        return None
    if element.find("range") is not None:
        return [number(element.findtext("range/min")), number(element.findtext("range/max"))]
    value = number(element.text)
    return [value, value] if value is not None else None

def report(path):
    root = ET.parse(path).getroot()
    perf = root.find("PerformanceEstimates")
    overall = perf.find("SummaryOfOverallLatency")
    loops = []
    parent = perf.find("SummaryOfLoopLatency")
    if parent is not None:
        for node in parent.iter():
            if node.find("PipelineII") is None:
                continue
            ii = number(node.findtext("PipelineII"))
            trip = bounds(node.find("TripCount"))
            latency = bounds(node.find("Latency"))
            function_latency = [number(overall.findtext("Best-caseLatency")),
                                number(overall.findtext("Worst-caseLatency"))]
            constant = None
            if trip and all(v is not None for v in trip + function_latency):
                candidate = function_latency[0] - ii * trip[0]
                if function_latency[1] == ii * trip[1] + candidate:
                    constant = candidate
            loops.append({
                "name": node.tag, "ii": ii,
                "pipeline_depth": number(node.findtext("PipelineDepth")),
                "trip_count_range": trip, "loop_latency_range": latency,
                "submodule_affine_constant": constant,
                "affine_basis": "Both reported trip-count/function-latency endpoints"
                    if constant is not None else None,
            })
    return {
        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "module": root.findtext("UserAssignments/TopModelName"),
        "estimated_period_ns": float(perf.findtext("SummaryOfTimingAnalysis/EstimatedClockPeriod")),
        "reported_latency": {k: number(overall.findtext(v)) for k, v in (
            ("min", "Best-caseLatency"), ("avg", "Average-caseLatency"), ("max", "Worst-caseLatency"))},
        "reported_interval": {k: number(overall.findtext(v)) for k, v in (
            ("min", "Interval-min"), ("max", "Interval-max"))},
        "resources": {e.tag: int(e.text) for e in root.findall("AreaEstimates/Resources/*")},
        "loops": loops,
    }

def load_build(relative, role_map, required_revision=None):
    directory = ROOT / relative
    run = json.loads((directory / "run.json").read_text(encoding="utf-8-sig"))
    if required_revision and run.get("service_schedule_revision") != required_revision:
        raise ValueError("Current schedule metadata mismatch; stale source reports are not panel-cache evidence")
    log = (directory / "vendor.log").read_text(errors="replace")
    if not (run.get("returncode") == 0 and run.get("c_simulation_passed")
            and run.get("synthesis_passed") and "Finished Command csynth_design" in log):
        raise ValueError("Successful actual Csim/synthesis required: " + relative)
    # Results directories retain old reports; only this log's modules are admissible.
    names = list(dict.fromkeys(re.findall(r"-- Implementing module '([^']+)'", log)))
    modules = {name: report(directory / (name + "_csynth.xml")) for name in names}
    top = modules["gate_fixed_fp32_top"]
    if not math.isclose(top["estimated_period_ns"], run["estimated_clock_ns"], abs_tol=1e-9):
        raise ValueError("Run/top XML period mismatch")
    stages = {}
    for role, suffix in role_map.items():
        candidates = suffix if isinstance(suffix, tuple) else (suffix,)
        matches = [modules[PREFIX + name] for name in candidates if PREFIX + name in modules]
        if len(matches) != 1 or len(matches[0]["loops"]) != 1:
            raise ValueError("Actual stage missing/ambiguous; review fixed-study map: " + repr(suffix))
        module = matches[0]
        stages[role] = dict(module["loops"][0], report=module["file"],
                            reported_latency=module["reported_latency"])
    return {
        "directory": relative, "recorded_utc": run["recorded_utc"],
        "service_schedule_revision": run.get("service_schedule_revision"),
        "clock_target_ns": run["clock_ns"], "uncertainty_ns": run["clock_uncertainty_ns"],
        "requested_fdiv_fabric_latency": run.get("fdiv_fabric_latency_request"),
        "part": run["part"], "tool_versions": run["tool_versions"], "top": top,
        "hls_estimate_meets_effective_budget":
            top["estimated_period_ns"] <= run["clock_ns"] - run["clock_uncertainty_ns"],
        "stages": stages,
        "leaf_reports": {name: item for name, item in modules.items() if name.startswith("fp32_")},
        "excluded_stale_reports": sorted(p.name for p in directory.glob("*_csynth.xml")
                                         if p.stem[:-7] not in modules),
        "warnings": [line for line in log.splitlines() if "WARNING:" in line],
        "c_simulation_checks": [line for line in log.splitlines() if line.startswith("CHECK ")],
        "c_simulation_pass_marker": "PASS fixed_fp32 service" in log,
        "runtime_cycles_measured": None, "postroute_frequency_mhz": None,
    }

def loop_component(build, role, count):
    """Affine submodule if identifiable; otherwise only its pipeline launch span."""
    if count < 1:
        raise ValueError("Positive valid trip count required")
    stage = build["stages"][role]
    ii, constant = stage["ii"], stage["submodule_affine_constant"]
    if constant is not None:
        trip = stage["trip_count_range"]
        if not trip[0] <= count <= trip[1]:
            raise ValueError("Trip count outside reported affine support")
        return ii * count + constant
    return ii * (count - 1)

def segments(rows, width):
    maximum = 32768 // width
    return [min(maximum, rows - first) for first in range(0, rows, maximum)]

def fixed_window(build, length):
    """Declared panel-cache schedule; partial HLS terms and structural traffic."""
    if length not in (128, 256, 512) or build["service_schedule_revision"] != REVISION:
        raise ValueError("Only the declared lengths and actual panel-cache revision are supported")
    layers, hidden, heads, intermediate = 4, 256, 4, 1024
    stages = []
    def add(name, calls, elements, component, useful_bytes, missing):
        stages.append({"stage": name, "calls": calls, "elements_or_dot_pairs": elements,
                       "partial_report_component_cycles": component,
                       "useful_source_bytes": useful_bytes,
                       "remaining_terms": missing, "complete_cycles": None})
    def dot_component(rows, k, outputs):
        panel = build["stages"]["dot_panel"]
        if panel["name"] == "dot_panel_outputs_dot_load_panel":
            loads = loop_component(build, "dot_panel", outputs * k)
        elif panel["name"] == "dot_load_panel":
            loads = outputs * loop_component(build, "dot_panel", k)
        else:
            raise ValueError("Review actual panel-loop shape before assigning trip counts")
        return loads + rows * (loop_component(build, "dot_row", k)
                    + math.ceil(outputs / 4) * loop_component(build, "dot_compute", k))
    dot_missing = ["top/panel/row/block transitions", "AXI request setup and stalls",
                   "output stores and optional bias loads/adds", "dispatch"]
    maximum_rows = min(length, 32768 // length)
    query_chunks = [min(maximum_rows, length-first) for first in range(0, length, maximum_rows)]
    blocks = math.ceil(length / 64)
    qk_calls = layers * heads * len(query_chunks) * blocks
    av_calls = layers * heads * len(query_chunks)
    pairs = layers * hidden * length * length
    add("attention_qk", qk_calls, pairs,
        layers * heads * blocks * sum(dot_component(m, 64, 64) for m in query_chunks),
        layers * heads * blocks * sum(4 * (m*64 + 64*64 + m*64) for m in query_chunks),
        dot_missing)
    add("attention_av", av_calls, pairs,
        layers * heads * sum(dot_component(m, length, 64) for m in query_chunks),
        layers * heads * sum(4 * (m*length + 64*length + m*64) for m in query_chunks),
        dot_missing)
    soft_component = lambda width: sum(loop_component(build, key, width) for key in
                                      ("softmax_load", "softmax_exp", "softmax_store"))
    incomplete = ["unidentified fill/drain/control", "AXI waits", "dispatch"]
    add("attention_softmax", av_calls, layers * heads * length * length,
        layers * heads * length * soft_component(length),
        layers * heads * length * length * 12, incomplete)
    leaves = build["leaf_reports"]
    leaf = lambda name: leaves["fp32_" + name]["reported_latency"]["min"]
    ln_row = (loop_component(build, "ln_load", hidden)
              + loop_component(build, "ln_variance", hidden)
              + loop_component(build, "ln_affine", hidden)
              + 2 * leaf("div") + leaf("add") + leaf("sqrt"))
    ln_rows = (2 * layers + 1) * length
    add("layer_norm_all", (2 * layers + 1) * len(segments(length, hidden)),
        ln_rows * hidden, ln_rows * ln_row, 16 * ln_rows * hidden, incomplete)
    for role, bytes_per_element in (("quantize", 5), ("rescale", 16)):
        calls = count = component = 0
        for width, matrices in ((hidden, 20), (intermediate, 4)):
            chunk_rows = segments(length, width)
            calls += matrices * len(chunk_rows)
            count += matrices * length * width
            component += matrices * sum(loop_component(build, role, rows * width) for rows in chunk_rows)
        add(role, calls, count, component, bytes_per_element * count, incomplete)
    for role, width, repeats, byte_rate in (
        ("embedding", hidden, 1, 16), ("residual", hidden, 8, 12),
        ("gelu", intermediate, 4, 8)):
        chunks = segments(length, width)
        add(role, repeats * len(chunks), repeats * length * width,
            repeats * sum(loop_component(build, role, rows * width) for rows in chunks),
            repeats * length * width * byte_rate, incomplete)
    add("pooler_dot", 4, hidden * hidden, 4 * dot_component(1, hidden, 64),
        4 * 4 * (hidden + 64 * hidden + 64 + 64), dot_missing)
    add("pooler_tanh", 1, hidden, loop_component(build, "tanh", hidden), 8 * hidden, incomplete)
    add("classifier_dot", 1, hidden * 2, dot_component(1, hidden, 2),
        4 * (hidden + 2 * hidden + 2 + 2), dot_missing)
    add("classifier_softmax", 1, 2, soft_component(2), 24, incomplete)
    layout = [
        {"stage": "k_gather_v_transpose", "useful_bytes": layers * 16 * length * hidden,
         "basis": "read+write each K and V head once per layer"},
        {"stage": "q_gather_context_scatter", "useful_bytes": layers * 16 * length * hidden,
         "basis": "read+write Q query slabs and resulting context head slabs"},
        {"stage": "qk_tile_to_score_slab", "useful_bytes": layers * 8 * heads * length * length,
         "basis": "read contiguous DOT tiles and scatter into complete score rows"},
        {"stage": "replicated_mask_slab", "useful_bytes": 8 * maximum_rows * length,
         "basis": "once per window, uncached base-mask read+slab write; reused across calls"},
        {"stage": "word_type_embedding_gather", "category": "embedding",
         "useful_bytes": 16 * length * hidden,
         "basis": "read+write word and token-type embedding vectors into two contiguous L*256 FP32 slabs"},
        {"stage": "word_type_embedding_id_reads", "category": "embedding",
         "useful_bytes": 16 * length,
         "basis": "read two L-element64-bit ID arrays, retaining software integer width; no unmeasured32-bit packing"},

    ]
    for item in layout:
        item.setdefault("category", "attention")
        item["cycles"] = None
        item["implemented"] = False
    layout_bytes = sum(item["useful_bytes"] for item in layout)
    attention_bytes = sum(s["useful_source_bytes"] for s in stages if s["stage"].startswith("attention_"))
    return {
        "length": length, "batch": 1, "service_schedule_revision": REVISION,
        "attention_query_chunks": query_chunks, "attention_key_blocks": blocks,
        "segmentation": {"width256_max_rows": 128, "width1024_max_rows": 32,
                         "width256_calls_per_tensor": len(segments(length, 256)),
                         "width1024_calls_per_tensor": len(segments(length, 1024))},
        "stages": stages, "service_calls": sum(s["calls"] for s in stages),
        "partial_report_component_cycles": sum(s["partial_report_component_cycles"] for s in stages),
        "useful_service_source_bytes": sum(s["useful_source_bytes"] for s in stages),
        "analytical_layout": layout, "analytical_layout_bytes": layout_bytes,
        "attention_source_bytes_including_layout": attention_bytes + sum(item["useful_bytes"] for item in layout if item["category"] == "attention"),
        "embedding_staging_workspace_bytes": 2 * length * hidden * 4,
        "layout_cycles": None, "integer_cycles": None, "dispatch_cycles": None,
        "memory_wait_cycles": None, "host_and_transfer_ms": None,
        "complete_fp32_cycles": None, "complete_check_latency_ms": None, "feasible": None,
    }

def runtime_evidence(build):
    """Admit only completed same-revision evidence; never turn missing costs into zero."""
    implementation_path = ROOT / "results/goal4/fixed_fp32_implementation/run.json"
    implementation = None
    if implementation_path.is_file():
        item = json.loads(implementation_path.read_text(encoding="utf-8-sig"))
        if (item.get("synthesis_recorded_utc") == build["recorded_utc"]
                and item.get("service_schedule_revision") == REVISION
                and item.get("returncode") == 0 and item.get("implementation_report_parsed")):
            implementation = {"run_record": str(implementation_path.relative_to(ROOT)),
                              "report": item["implementation_report"],
                              "metrics": item["implementation_metrics"],
                              "integrated_system": False, "board_programmed": False}
    tag = re.sub(r"[^A-Za-z0-9_-]", "_", build["recorded_utc"])
    directory = ROOT / "results/goal4/fixed_fp32_rtl" / tag
    small_path = directory / "run.json"
    small = None
    if small_path.is_file():
        run = json.loads(small_path.read_text(encoding="utf-8-sig"))
        if (run.get("passed") and run.get("all_nine_modes_checked")
                and run.get("synthesis_recorded_utc") == build["recorded_utc"]
                and run.get("service_schedule_revision") == REVISION):
            cases = []
            for reference in run["cases"]:
                row = json.loads(Path(reference["run_record"]).read_text(encoding="utf-8-sig"))
                if not row.get("passed") or len(row.get("numerical_checks", [])) != 2:
                    raise ValueError("Incomplete selected small RTL record")
                cases.append({"id": reference["id"], "run_record": reference["run_record"],
                              "observed_call_cycles": [v["latency_cycles"] for v in row["transactions"]],
                              "max_absolute_error": max(v["maximum_absolute_error"] for v in row["numerical_checks"])})
            if len(cases) != 11:
                raise ValueError("Expected unchanged11-case small RTL suite")
            small = {"run_record": str(small_path.relative_to(ROOT)), "cases": cases,
                     "checked_calls": 22, "memory_model": run["memory_model"], "clock_ns": run["clock_ns"]}
    primary_path = directory / "primary_l256/primary_costs.json"
    primary_run_path = directory / "primary_l256/run.json"
    primary = None
    status = "absent"
    if primary_run_path.is_file():
        run = json.loads(primary_run_path.read_text(encoding="utf-8-sig"))
        status = run["state"]
        if run.get("passed") and primary_path.is_file():
            from run_fixed_fp32_rtl import primary_cases, primary_record_matches
            value = json.loads(primary_path.read_text(encoding="utf-8-sig"))
            if (not small or value.get("service_schedule_revision") != REVISION
                    or value.get("synthesis_recorded_utc") != build["recorded_utc"]
                    or value.get("clock_ns") != build["clock_target_ns"]
                    or value.get("memory_model") != small["memory_model"]
                    or value.get("call_count") != 411 or len(value.get("terms", [])) != 15):
                raise ValueError("Primary runtime summary differs from current schedule/evidence")
            expected = {c["id"]: c for c in primary_cases()}
            totals = [0, 0]
            synthesis = {"recorded_utc": build["recorded_utc"], "service_schedule_revision": REVISION,
                         "clock_ns": build["clock_target_ns"]}
            seen = set()
            for term in value["terms"]:
                case = expected.get(term["id"])
                observed = json.loads(Path(term["run_record"]).read_text(encoding="utf-8-sig"))
                if (not case or term["id"] in seen or not primary_record_matches(observed, case, synthesis)
                        or term["calls_per_window"] != case["multiplicity"]):
                    raise ValueError("Primary full-call evidence mismatch")
                cycles = [v["latency_cycles"] for v in observed["transactions"]]
                if cycles != term["observed_call_cycles"]:
                    raise ValueError("Primary aggregate differs from raw observed cycles")
                totals = [a + case["multiplicity"] * b for a, b in zip(totals, cycles)]
                seen.add(term["id"])
            if seen != set(expected) or totals != value["fp32_service_sum_cycles_by_rep"]:
                raise ValueError("Primary weighted sum differs from declared411-call schedule")
            primary = dict(value, summary_record=str(primary_path.relative_to(ROOT)))
            status = "complete"
    return {"implementation": implementation, "small_rtl": small,
            "primary_runtime": primary, "primary_status": status}



def main():
    model = json.loads((ROOT / "checkpoints/floating/epoch_3/config.json").read_text())
    expected = {"num_hidden_layers": 4, "hidden_size": 256,
                "num_attention_heads": 4, "intermediate_size": 1024}
    if any(model[key] != value for key, value in expected.items()):
        raise ValueError("Analyzer is restricted to the declared fixed BERT model")
    current = load_build("results/goal4/fixed_fp32", CURRENT, REVISION)
    before = load_build("results/goal4/fixed_fp32_timing_corrected", BEFORE)
    scheduled = load_build("results/goal4/fixed_fp32_scheduled", SCHEDULED)
    vector = load_build("results/goal4/fixed_fp32_vector_scheduled", VECTOR)
    windows = [fixed_window(current, length) for length in (128, 256, 512)]
    runtime = runtime_evidence(current)
    primary = runtime["primary_runtime"]
    windows[1]["observed_fp32_service_sum_cycles_by_rep"] = primary["fp32_service_sum_cycles_by_rep"] if primary else None
    windows[1]["observed_fp32_service_sum_ms_at_simulated_clock_by_rep"] = primary["fp32_service_sum_ms_at_simulated_clock_by_rep"] if primary else None
    windows[1]["observed_fp32_axi_by_rep"] = primary["fp32_service_axi_by_rep"] if primary else None
    batch_grid = [
        {"length": w["length"], "batch": batch, "execution": "serialized examples",
         "service_calls": batch * w["service_calls"],
         "partial_report_component_cycles": batch * w["partial_report_component_cycles"],
         "useful_source_bytes_including_layout":
             batch * (w["useful_service_source_bytes"] + w["analytical_layout_bytes"]),
         "complete_check_latency_ms": None, "feasible": None}
        for w in windows for batch in (1, 8, 32)]
    comparison = []
    for role in CURRENT:
        older_role = "dot_weights" if role == "dot_panel" else role
        initial_role = ("mixed_vectors" if role in ("quantize", "rescale", "embedding",
                         "residual", "gelu", "tanh") else
                        "dot_weights_four" if role == "dot_panel" else role)
        baseline = before["stages"].get(initial_role)
        comparison.append({
            "stage": role, "timing_corrected_ii": baseline["ii"] if baseline else None,
            "scheduled_ii": scheduled["stages"][older_role]["ii"],
            "vector_scheduled_ii": vector["stages"][older_role]["ii"],
            "current_ii": current["stages"][role]["ii"],
            "current_depth": current["stages"][role]["pipeline_depth"],
            "basis_note": "Current panel load occurs once per call; historical weight loads repeat per input row"
                if role == "dot_panel" else None,
            "not_a_measured_speedup": True,
        })
    output = {
        "scope": "Fixed-study HLS extraction, observed primary service-call accounting and remaining analytical terms",
        "service_schedule_revision": REVISION,
        "runtime_evidence": runtime,
        "builds": {"timing_corrected": before, "scheduled": scheduled,
                   "vector_scheduled": vector, "current": current},
        "comparison": comparison, "windows": windows, "serialized_batch_grid": batch_grid,
        "historical_row_reload_accounting": "results/goal4/fixed_fp32_vector_scheduled/schedule_summary.json",
        "component_definition": {
            "affine": "II*n+constant only when both actual trip/function-latency endpoints support it",
            "otherwise": "II*(n-1) launch-span component, not complete latency or a bound",
            "whole_cost": "Historical HLS components remain partial; completed primary evidence supplies weighted full-call observations only; missing integration terms remain null",
            "memory": "Source bytes are not AXI beats; later DDR service must not be double-counted"},
        "unresolved": ["integer-engine revision/runtime evidence", "FPGA sequencer and call overhead",
                       "all layout/gather/transpose/mask implementations",
                       "DDR/NoC mapping, effective bandwidth and arbitration", "host transport/synchronization",
                       "document max/strict-threshold control", "post-route and board latency/energy",
                       "end-to-end numerical/protection agreement"],
    }
    target = ROOT / "results/goal4/fixed_fp32/schedule_summary.json"
    target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(target), "revision": REVISION,
                     "period_ns": current["top"]["estimated_period_ns"],
                     "resources": current["top"]["resources"], "primary_calls": windows[1]["service_calls"],
                     "primary_attention_bytes_including_layout": windows[1]["attention_source_bytes_including_layout"],
                     "partial_cycles_not_latency": windows[1]["partial_report_component_cycles"],
                     "full_latency": None}, indent=2))

if __name__ == "__main__":
    main()

