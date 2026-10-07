"""Offline cached linear characterization; no vendor commands or fitted cycles.

Freeze the report-derived components before the cached reserved dispatch.
Refreshes preserve that freeze and attach the currently saved RTL observations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from analyze_hls_characterization import POINT_FIELDS, bounds, numeric, one_loop, read_json
from build_linear_hls import configured_linear_study

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/goal4/cached_characterization.json"
REPORT = ROOT / "reports/goal4_cached_linear.md"
VARIANT = "weight_cache_pilot"


def rel(path):
    return path.relative_to(ROOT).as_posix()


def synthesis(config, profile_id="lanes16", static_bank=False):
    declared_profiles, all_points = configured_linear_study(config, cached=True)
    profiles = {p["id"]: p for p in declared_profiles}
    if profile_id not in profiles:
        raise ValueError("Select one declared cached physical profile.")
    profile = profiles[profile_id]
    lanes = profile["lanes"]
    build = ROOT / ("build/linear_hls_cached_static_bank" if static_bank else "build/linear_hls_cached") / profile_id
    run = read_json(build / "run.json")
    points = [{key: p[key] for key in POINT_FIELDS} for p in all_points if p["profile"] == profile_id]
    if not points:
        raise ValueError("No characterization points declared for this cached profile.")
    if (run.get("returncode") != 0 or not run.get("hls_synthesis_performed")
            or not run.get("hls_c_simulation_performed")
            or run.get("kernel_variant") != VARIANT
            or run["selected_profile"] != profile):
        raise ValueError("Successful cached C simulation and synthesis required for the selected profile.")
    if static_bank and (profile_id != "lanes64" or run.get("source_revision") != "static_bank_index_v1"):
        raise ValueError("Expected the isolated lanes64 static_bank_index_v1 synthesis.")
    folder = build / "project/solution/syn/report"
    top_path = folder / "gate_linear_top_csynth.xml"
    engine_name = "linear_engine_cached_static_bank" if static_bank else "linear_engine_cached"
    engine_path = folder / f"{engine_name}_{lanes}_4_16_64_256_s_csynth.xml"
    top, engine = (ET.parse(p).getroot() for p in (top_path, engine_path))
    a = top.find("UserAssignments")
    target = numeric(a.findtext("TargetClockPeriod"))
    uncertainty = numeric(a.findtext("ClockUncertainty"))
    if (a.findtext("Part") != config["hardware"]["part"]
            or a.findtext("FlowTarget") != "vivado"
            or target != 1000 / config["hardware"]["clock_objective_mhz"]
            or uncertainty != config["hardware"]["hls"]["clock_uncertainty_ns"]):
        raise ValueError("Device, flow or clock differs from shared configuration.")
    row = one_loop(engine.find("PerformanceEstimates/SummaryOfLoopLatency"))
    output = one_loop(row)
    inner = one_loop(output)
    inner_range = bounds(inner.find("IterationLatency"))
    row_min, output_min = (bounds(e.find("IterationLatency"))["min"] for e in (row, output))
    terms = {
        "row_control_cycles": row_min - output_min,
        "output_tile_init_store_control_cycles": output_min - inner_range["min"],
        "inner_tile_w4_cycles": inner_range["min"],
        "inner_tile_w8_cycles": inner_range["max"],
    }
    stages = []
    for path in folder.glob("*Pipeline*csynth.xml"):
        tree = ET.parse(path).getroot()
        loop = one_loop(tree.find("PerformanceEstimates/SummaryOfLoopLatency"))
        latency = tree.find("PerformanceEstimates/SummaryOfOverallLatency")
        stages.append((int(re.search(r"VITIS_LOOP_(\d+)_", loop.tag).group(1)), {
            "report": rel(path), "loop": loop.tag,
            "trip_count": bounds(loop.find("TripCount")),
            "iteration_latency": bounds(loop.find("IterationLatency")),
            "pipeline_ii": numeric(loop.findtext("PipelineII")),
            "latency_min_cycles": numeric(latency.findtext("Best-caseLatency")),
            "latency_max_cycles": numeric(latency.findtext("Worst-caseLatency")),
        }))
    if len(stages) != 6:
        raise ValueError("Expected exactly six cached pipeline stage reports.")
    names = ("preload_w8", "preload_w4", "initialize", "activation_load", "compute", "store")
    stages = dict(zip(names, [value for _, value in sorted(stages)]))
    if any(s["pipeline_ii"] != 1 for s in stages.values()):
        raise ValueError("Cached component formulas require the reported II=1 schedule.")
    for key in ("preload_w8", "preload_w4"):
        s = stages[key]
        if any(s["latency_" + k + "_cycles"] - s["trip_count"][k] != 2 for k in ("min", "max")):
            raise ValueError("Preload report no longer has B+2 latency.")
    if lanes == 16 and terms != {"row_control_cycles": 2, "output_tile_init_store_control_cycles": 150,
                 "inner_tile_w4_cycles": 534, "inner_tile_w8_cycles": 790}:
        raise ValueError("Pilot schedule changed; independently derive new formulas before refreezing.")
    banks = []
    for line in engine_path.with_suffix(".rpt").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) == 10 and re.fullmatch(r"weight_cache(?:_\d+)?_U", cells[0]):
            banks.append(dict(zip(("name", "module", "BRAM_18K", "LUT", "FF", "DSP",
                                   "depth", "bits", "banks", "total_bits"),
                                  cells[:2] + [numeric(c) for c in cells[2:]])))
    if len(banks) != lanes or sum(b["depth"] * b["bits"] // 8 for b in banks) != 262144:
        raise ValueError("Expected one physical cache bank per lane totaling 262144 logical bytes.")
    resources = {e.tag: numeric(e.text) for e in top.find("AreaEstimates/Resources")}
    evidence = {
        "status": "reported", "kernel_variant": VARIANT,
        "profile": profile, "run_record": rel(build / "run.json"),
        "synthesis_recorded_utc": run["recorded_utc"],
        "part": a.findtext("Part"), "report_version": top.findtext("ReportVersion/Version"),
        "target_clock_ns": target, "clock_uncertainty_ns": uncertainty,
        "estimated_clock_ns": numeric(top.findtext("PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod")),
        "estimated_resources": resources, "nominal_terms": terms, "stages": stages,
        "cache_banks": banks, "cache_logical_bytes": 262144,
        "cache_BRAM_18K": sum(b["BRAM_18K"] for b in banks),
        "top_report": rel(top_path), "engine_report": rel(engine_path),
        "top_report_latency": {k: numeric(top.findtext("PerformanceEstimates/SummaryOfOverallLatency/" + tag))
                              for k, tag in (("min", "Best-caseLatency"), ("max", "Worst-caseLatency"))},
    }
    if static_bank:
        evidence["source_revision"] = run["source_revision"]
        evidence["source_snapshot"] = run["source_snapshot"]
    return evidence, points


def prediction(point, evidence):
    m, k, n, bits = (point[key] for key in ("rows", "inner", "outputs", "weight_bits"))
    if m % 4 or k % 64 or n % 16 or (k, n) not in ((256, 256), (256, 1024), (1024, 256)):
        raise ValueError("This pilot freeze covers only declared full-tile model shapes.")
    r, o, i = m // 4, n // 16, k // 64
    t = evidence["nominal_terms"]
    preload_bytes = n * k * bits // 8
    parts = {
        "preload_report": preload_bytes + 2,
        "row_control": r * t["row_control_cycles"],
        "output_tile_init_store_control": r * o * t["output_tile_init_store_control_cycles"],
        "inner_tiles": r * o * i * t[f"inner_tile_w{bits}_cycles"],
    }
    nominal = sum(parts.values())
    lanes = evidence["profile"]["lanes"]
    if lanes == 16 or evidence["kernel_variant"] == "weight_cache_static_bank":
        if lanes == 64 and t != {"row_control_cycles": 2, "output_tile_init_store_control_cycles": 150,
                "inner_tile_w4_cycles": 342, "inner_tile_w8_cycles": 406}:
            raise ValueError("Reviewed static-bank64 schedule changed; derive its correction again.")
        penalties = {"activation_read_schedule": r * o * i * 253,
                     "output_write_schedule": r * o * 127}
    elif lanes == 32:
        if t != {"row_control_cycles": 2, "output_tile_init_store_control_cycles": 150,
                 "inner_tile_w4_cycles": 920, "inner_tile_w8_cycles": 1048}:
            raise ValueError("Reviewed cached32 schedule changed; derive its correction again.")
        penalties = {"activation_and_repeated_compute_schedule": r * o * i * 127,
                     "output_write_schedule": r * o * 127}
    else:
        penalties = None
    return {
        "point": point, "row_tiles": r, "output_tiles": o, "inner_tiles": i,
        "nominal_components_cycles": parts, "nominal_component_cycles": nominal,
        "bounded_axi_schedule_terms_cycles": penalties,
        "bounded_axi_adjusted_component_cycles": nominal + sum(penalties.values()) if penalties is not None else None,
        "unresolved_top_preload_branch_and_stall_cycles": None,
        "complete_latency_prediction_cycles": None,
        "kind": "partial_schedule_prediction_with_explicit_unresolved_residual",
        "traffic": {"source_weight_read_bytes": preload_bytes,
                    "logical_activation_bytes": m * k * o, "output_write_bytes": 4 * m * n},
    }


def observation(point, evidence, pred):
    rtl_root = "results/goal4/rtl_cached_static_bank" if evidence["kernel_variant"] == "weight_cache_static_bank" else "results/goal4/rtl_cached"
    path = ROOT / rtl_root / point["id"] / "run.json"
    if not path.is_file():
        return {"status": "not_run", "run_record": rel(path)}
    run = read_json(path)
    v = run.get("verified") or {}
    tx = run.get("transactions") or []
    latencies = v.get("latency_cycles") or []
    issues = []
    if run.get("point") != point or run.get("kernel_variant") != VARIANT:
        issues.append("point_or_variant_mismatch")
    if evidence.get("source_revision") and run.get("source_revision") != evidence["source_revision"]:
        issues.append("source_revision_mismatch")
    if run.get("synthesis_recorded_utc") != evidence["synthesis_recorded_utc"]:
        issues.append("synthesis_association_mismatch")
    if not run.get("passed") or not v.get("passed") or run.get("returncode") != 0:
        issues.append("run_not_passed")
    if (len(latencies) != point["repetitions"] or len(tx) != point["repetitions"]
            or any(not t.get("exact_output_passed") for t in tx)):
        issues.append("exact_transaction_count_or_output_check_missing")
    if v.get("clock_period_ns") != evidence["target_clock_ns"] or v.get("dut_rtl_modified") is not False:
        issues.append("clock_or_dut_boundary_mismatch")
    usable = not issues
    return {
        "status": "measured_rtl" if usable else "unusable", "issues": issues,
        "run_record": rel(path), "recorded_utc": run.get("recorded_utc"),
        "state": run.get("state"), "returncode": run.get("returncode"),
        "memory_model": run.get("memory_model"), "transactions": tx,
        "latency_cycles": latencies, "interval_cycles": v.get("interval_cycles"),
        "interval_includes_axilite_restart": v.get("interval_includes_axilite_restart"),
        "total_execution_cycles": v.get("total_execution_cycles"),
        "measured_minus_nominal_component_cycles": [
            value - pred["nominal_component_cycles"] for value in latencies
        ] if usable else None,
        "measured_minus_adjusted_component_cycles": [
            value - pred["bounded_axi_adjusted_component_cycles"] for value in latencies
        ] if usable and pred["bounded_axi_adjusted_component_cycles"] is not None else None,
        "comparison_kind": "component_residual_not_complete_latency_estimator_error",
    }


def markdown(result):
    e = result["synthesis"]
    lanes = e["profile"]["lanes"]
    terms = e["nominal_terms"]
    stage = e["stages"]
    timing_ok = e["estimated_clock_ns"] <= e["target_clock_ns"] - e["clock_uncertainty_ns"]
    timing_text = "meets" if timing_ok else "does not meet"
    lines = [
        "# Cached linear common-baseline pilot",
        "",
        f"This is the separately recorded LANES{lanes} cached common engine, kernel variant `{e['kernel_variant']}`" + (f", source revision `{e['source_revision']}`" if e.get("source_revision") else "") + ". This artifact does not select the active comparison profiles. Historical tiled-profile evidence and existing cached freezes remain separate; any adopted common engine must be available to all search arms.",
        "",
        "## Actual synthesis",
        "",
        f"Successful C simulation and synthesis: {e['run_record']} ({e['synthesis_recorded_utc']}). HLS {e['report_version']}, {e['part']}; target {e['target_clock_ns']} ns, uncertainty {e['clock_uncertainty_ns']} ns, estimated period {e['estimated_clock_ns']} ns. This {timing_text} the HLS scheduling target; it is not routed timing or board performance.",
        "",
        "Top resource estimates: " + ", ".join(f"{k}={v}" for k, v in e["estimated_resources"].items()) + f". BRAM_18K counts 18-Kib primitives; {e['estimated_resources']['BRAM_18K'] / 2:g} BRAM36 capacity equivalents do not prove physical pairing.",
        "",
        f"The report allocates {len(e['cache_banks'])} weight-cache banks: {e['cache_logical_bytes']} logical bytes and {e['cache_BRAM_18K']} BRAM_18K blocks. The cache holds the actual largest N*K, not a 1024-square matrix. W4 reads each packed source byte once, expands two codes, and uses the same physical cache capacity as W8.",
        "",
        "| Stage | Report latency (cycles) | Trip count | II |",
        "|---|---:|---:|---:|",
    ]
    for name, s in e["stages"].items():
        lines.append(f"| {name} | {s['latency_min_cycles']}..{s['latency_max_cycles']} | {s['trip_count']['min']}..{s['trip_count']['max']} | {s['pipeline_ii']} |")
    lines += [
        "",
        f"Direct cache reads feed digit arithmetic. The compute report has {stage['compute']['trip_count']['min']}..{stage['compute']['trip_count']['max']} iterations at II={stage['compute']['pipeline_ii']}. Runtime-bound HLS top latency is undefined; generic loop maxima are not model-shape timings.",
        "",
        "## Frozen component prediction",
        "",
        f"Freeze: {(result.get('nominal_freeze') or {}).get('recorded_utc', 'NOT FROZEN')}. Source and synthesis evidence, point definitions, and predictions are retained in {rel(OUT)}.",
        "",
        "For declared full tiles, R=M/4, O=N/16, I=K/64, B=NK for W8 or NK/2 for W4. P=B+2 is the HLS preload report latency.",
        "",
        f"C_nominal = P + R * [{terms['row_control_cycles']} + O * ({terms['output_tile_init_store_control_cycles']} + I * ({terms['inner_tile_w4_cycles']} for W4, {terms['inner_tile_w8_cycles']} for W8))].",
        "",
        "These are the actual selected engine report components. They do not by themselves specify accepted top-start to top-done time.",
        "",
        ("For the bounded direct AXI testbench, source-guided schedule terms are +253*R*O*I for activation reads and +127*R*O for output stores. Their sum with C_nominal is a partial adjusted component. The original component freeze keeps the top/preload residual unresolved." if lanes == 16 or e["kernel_variant"] == "weight_cache_static_bank" else "The cached32 repeated-compute FSM gives +127*R*O*I relative to its report inner terms, with +127*R*O for stores. This differs from the cached16 inner correction." if lanes == 32 else "This profile has no reviewed AXI/control correction; complete predictions remain null."),
        "",
        "Preload emits one HLS range request and drains bytes at II=1; the adapter creates bursts. Do not add the prototype's one-read-request-per-code penalty. Reported child latency and sampled start-to-done latency can differ; a complete correction requires this profile's generated-RTL evidence.",
        "",
        "## Saved direct RTL observations",
        "",
        "| Point | Role | Nominal component | Adjusted component | Measured cycles | Remaining residual |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for p in result["points"]:
        point, pred, obs = p["point"], p["prediction"], p["observation"]
        measured = str(obs.get("latency_cycles")) if obs["status"] == "measured_rtl" else obs["status"]
        lines.append(f"| {point['id']} | {point['role']} | {pred['nominal_component_cycles']} | {pred['bounded_axi_adjusted_component_cycles']} | {measured} | {obs.get('measured_minus_adjusted_component_cycles')} |")
    lines += [
        "",
        "Each accepted run must match the frozen synthesis and point, preserve the DUT RTL, and pass exact integer outputs for both transactions. Measured latencies are actual RTL cycles in a per-port fixed-byte-array, one-outstanding-request, registered-response memory model with no additional DDR delay. Restart intervals include AXI-Lite host restart. They are neither vendor UVM C/RTL Pass records nor physical DDR bandwidth or board measurements.",
        "",
        ("The saved LANES16 W4 256-by-256 target uses 512 weight bursts, 8192 beats and 32768 source bytes. Its two 251993-cycle transactions leave 15 cycles against the original 251978-cycle adjusted component; this is not a universal correction." if lanes == 16 else "Saved transaction counters retain burst/beat/byte traffic. No measured values from another physical profile are assigned to this profile."),
        "",
        "A faster common baseline requires matching-point timing comparisons and a resource-feasible combined integer/FP32 design. This pilot establishes cache allocation, schedule components and the listed exact-output observations only. No bandwidth_feasible, board throughput, or complete full-model latency claim follows from it.",
        "",
    ]
    if lanes == 32:
        lines += [
            "## Cached32 compute scheduling change", "",
            "The compiler pipelines only the two/four digit iterations, with report latency6/8. Outer row/output loops remain in the parent FSM. The sampled child duration is4/6; each wait state occupies5/7 edges. Per inner tile, states31-35 contribute5 row checks +68 output checks +64 setup +64*(duration+1) wait +64 commit =521/649 cycles. Add one inner check and525 activation-wait edges to get1047/1175, versus report920/1048: the correction is127 per inner tile.", "",
            "The output-tile component remains64 initialization +209 store +4 control =277. All three AXI adapter source files equal cached16 by ordinary text comparison. Top/preload FSM boundaries were separately inspected; no development or reserved residual was fitted.", "",
            "More lanes therefore do not imply a faster engine: this build repeatedly starts a short compute pipeline. Its complete predictions remain predictions until the exact-output RTL observations arrive.", "",
        ]
    if e["kernel_variant"] == "weight_cache_static_bank":
        lines += ["## Static-bank64 source derivation", "",
            "The actual compute pipeline is flattened across the64 output-row positions and one/two digit phases. Its counter bounds64/128 and three exit registers give sampled67/131 cycles, two fewer than report69/133. It has no blocking condition. This differs from cached32's repeated short subcalls.", "",
            "Initialization still samples64 cycles. The activation and store stages retain256 byte reads and64 word writes, the same11/14 drain registers, and all three AXI adapters are text-identical to cached16. Under the unchanged bounded slave, their sampled durations are524/209. Thus each inner tile costs524+(67 or131)+4=595/659, and each output tile outside inner work costs64+209+4=277. These are source-derived predictions, not observations copied from another profile.", "",
            "The actual parent retains the request, eleven setup states, preload wait, row/output checks and terminal state; an independent read-only review confirmed these state boundaries. The new preload children directly signal done at byte index B. Top/preload fixed17 minus the report's extra2 gives the separately frozen15-cycle correction. No new static-bank RTL observation was used in this derivation.", ""]
    completion = result.get("source_derived_complete_freeze")
    if completion:
        lines += [
            "## Separate source-derived completed freeze",
            "",
            f"Completed before reserved dispatch at {completion['recorded_utc']}. The original component freeze and its null residual fields are unchanged.",
            "",
            "C_complete = C_adjusted + 17 - 2 = C_adjusted + 15, for this bounded memory model. The 15 is counted from RTL; it is not a fitted mean of the development residuals.",
            "",
            "Top edge0 accepts start; engine accepts edge2; range request occurs at edge3; eleven branch setup states lead to preload acceptance at edge15. The following preload-to-row and terminal boundaries produce 17 cycles relative to the actual preload duration. Preload's index starts at zero, advances once per unblocked active edge, and asserts done at index B while consuming the last byte. Its sampled duration is B, two fewer than the HLS report B+2. Both W4 and W8 use this boundary.",
            "",
            "Source anchors are listed in the JSON freeze. The range preload has an initially idle weight port and burst supply faster than its one-byte-per-cycle consumer. The model predicts no additional initial or steady-state preload stalls under this exact registered slave; the largest W8 byte count is already covered by a development target. This is a declared memory-model assumption, not physical DDR evidence.",
            "",
            "The passive counter samples both acceptance and done at positive clock edges before sequential updates, consistent with the generated FSM. Latency is done_cycle minus start_cycle, without an inclusive extra cycle. AXI-Lite configuration and restart delay are outside this latency and inside the separately recorded start-to-start interval.",
            "",
            "| Point | Complete predicted cycles | Time at 5 ns (ms) | Measured minus predicted cycles |",
            "|---|---:|---:|---:|",
        ]
        for item in result["points"]:
            cp = item["complete_prediction"]
            lines.append(f"| {item['point']['id']} | {cp['cycles']} | {cp['time_at_target_clock_ns'] / 1000000:.6f} | {item['observation'].get('complete_prediction_error_cycles')} |")
        reserved = [p for p in result["points"] if p["point"]["role"] == "reserved"]
        for item in reserved:
            obs = item["observation"]
            if obs["status"] == "measured_rtl":
                lines += ["", "Reserved validation: `" + item["point"]["id"]
                    + "` passed exact output comparison in both transactions, with cycles "
                    + str(obs["latency_cycles"]) + ". Measured-minus-complete-prediction errors are "
                    + str(obs["complete_prediction_error_cycles"]) + " cycles. The original nominal component remains "
                    + str(item["prediction"]["nominal_component_cycles"]) + " cycles; its residuals remain "
                    + str(obs["measured_minus_nominal_component_cycles"]) + " cycles. Neither freeze was changed."]
        lines += [
            "",
            "The complete prediction is scoped to the declared bounded RTL memory model. Reserved results assess the already frozen prediction; any failure must remain visible rather than be absorbed into its coefficients. Whole-model sequencer overhead, physical memory integration and board timing remain outside this integer-service prediction.",
            "",
        ]
    return "\n".join(lines)



def completed_cached32_freeze(evidence, points, predictions, component_freeze):
    """Reviewed cached32 FSM: repeated short compute calls, not lanes16 flattening."""
    stages = evidence["stages"]
    expected = {"initialize": (66, 66), "activation_load": (269, 269),
                "compute": (6, 8), "store": (80, 80)}
    for name, values in expected.items():
        if tuple(stages[name][k] for k in ("latency_min_cycles", "latency_max_cycles")) != values:
            raise ValueError("Reviewed cached32 stage changed: " + name)
    v = "build/linear_hls_cached/lanes32/project/solution/syn/verilog/"
    for name in ("gate_linear_top_gmem_a_m_axi.v", "gate_linear_top_gmem_w_m_axi.v",
                 "gate_linear_top_gmem_o_m_axi.v"):
        a = ROOT / "build/linear_hls_cached/lanes16/project/solution/syn/verilog" / name
        b = ROOT / v / name
        if a.read_text() != b.read_text():
            raise ValueError("Cached32 AXI adapter differs from the reviewed bounded-memory schedule: " + name)
    source_evidence = [
        {"file": v + "gate_linear_top.v", "lines": [1023, 1064, 1153],
         "fact": "same three-state top; accepted start edge0, registered engine launch, engine accepts edge2"},
        {"file": v + "gate_linear_top_linear_engine_cached_32_4_16_64_256_s.v",
         "lines": [4752, 4810, 10242, 10326, 10361, 10368, 10375, 10385, 10395],
         "fact": "request edge3; eleven branch setup states; preload starts edge15; row/output terminal boundaries retained; states31-35 execute five row checks,68 output checks,64 setup states,64 child waits and64 commit states per inner tile"},
        {"file": v + "gate_linear_top_linear_engine_cached_Pipeline_VITIS_LOOP_90_13_VITIS_LOOP_92_14.v",
         "lines": [7146, 7816, 7832, 11392],
         "fact": "counter exits at bound2/4; two exit-pipeline registers give sampled duration4/6, not reported6/8; parent wait occupies5/7 edges"},
        {"file": v + "gate_linear_top_linear_engine_cached_32_4_16_64_256_Pipeline_VITIS_LOOP_50_1.v",
         "lines": [537, 553, 561, 1169],
         "fact": "unblocked byte index advances to B; direct loop exit gives sampled B, reportB+2"},
        {"file": v + "gate_linear_top_linear_engine_cached_32_4_16_64_256_Pipeline_VITIS_LOOP_55_2.v",
         "lines": [], "fact": "equivalent packed-byte counter and direct loop exit; sampled B=NK/2"},
        {"file": "hardware/linear_axi_tb.sv", "lines": [51, 156, 167, 387, 392, 398, 540],
         "fact": "unchanged bounded registered-response slave; same three AXI adapter files by ordinary text comparison; edge timestamp difference"},
    ]
    for item in source_evidence:
        if not (ROOT / item["file"]).is_file():
            raise ValueError("Missing reviewed cached32 source: " + item["file"])
    completed = {}
    for point in points:
        cycles = predictions[point["id"]]["bounded_axi_adjusted_component_cycles"] + 15
        completed[point["id"]] = {
            "cycles": cycles, "time_at_target_clock_ns": cycles * evidence["target_clock_ns"],
            "boundary_cycles_relative_to_report_components": 15,
            "controller_cycles_relative_to_actual_preload_duration": 17,
            "preload_sampled_duration_minus_report_latency": -2,
            "additional_preload_or_range_request_stalls_predicted": 0,
            "kind": "complete_transaction_prediction_for_declared_bounded_memory_model",
        }
    return {
        "kernel_variant": VARIANT, "synthesis_recorded_utc": evidence["synthesis_recorded_utc"],
        "component_freeze_recorded_utc": component_freeze["recorded_utc"],
        "point_definitions": points, "source_evidence": source_evidence, "predictions": completed,
        "formula": "P+R*(2+O*(150+I*(920 if W4 else1048)))+127*R*O*I+127*R*O+17-2",
        "fit": "none; generated FSM counts and stage/AXI boundary derivation, no observed residual used",
        "inner_stage_derivation": {
            "activation_sampled_cycles": 524, "activation_wait_state_cycles": 525,
            "compute_sampled_cycles_w4_w8": [4, 6],
            "compute_outer_state_counts": {"row_checks": 5, "output_checks": 68,
                "setup": 64, "child_wait_count": 64, "commit": 64},
            "nested_compute_cycles_w4_w8": [521, 649],
            "inner_total_cycles_w4_w8": [1047, 1175],
            "nominal_inner_correction_cycles": 127,
            "explanation": "inner-check1 +activation-wait525 +nested(5+68+64+64*(d+1)+64); output component64+209+4=277 versusreport150",
        },
        "memory_assumptions": [
            "unchanged three-bundle AXI adapter, verified by ordinary text comparison against lanes16",
            "same bounded registered-response byte-array slave; no additional DDR or contention delay",
            "full matrix byte preload with idle weight port,11 setup states and burst supply faster than consumption",
            "full row tiles and exactly the three declared cached32 points",
        ],
        "development_check": "No observed target or reserved residual is used to derive these predictions.",
        "counter_boundary": "passive posedge accepted top start to asserted top done; done-start; restart excluded",
    }


def completed_static64_freeze(evidence, points, predictions, component_freeze):
    """Actual static_bank_index_v1 flattened compute and bounded AXI schedule."""
    stages = evidence["stages"]
    expected = {"initialize": (66, 66), "activation_load": (269, 269),
                "compute": (69, 133), "store": (80, 80)}
    for name, values in expected.items():
        if tuple(stages[name][k] for k in ("latency_min_cycles", "latency_max_cycles")) != values:
            raise ValueError("Reviewed static-bank64 stage changed: " + name)
    v = "build/linear_hls_cached_static_bank/lanes64/project/solution/syn/verilog/"
    for name in ("gate_linear_top_gmem_a_m_axi.v", "gate_linear_top_gmem_w_m_axi.v",
                 "gate_linear_top_gmem_o_m_axi.v"):
        a = ROOT / "build/linear_hls_cached/lanes16/project/solution/syn/verilog" / name
        if a.read_text() != (ROOT / v / name).read_text():
            raise ValueError("Static-bank64 AXI adapter differs from reviewed schedule: " + name)
    source_evidence = [
        {"file": v + "gate_linear_top.v", "lines": [1023, 1026, 1152, 1163],
         "fact": "three-state top: accept edge0, registered launch in state2, engine accepts edge2"},
        {"file": v + "gate_linear_top_linear_engine_cached_static_bank_64_4_16_64_256_s.v",
         "lines": [8196, 8208, 8232, 8244, 8256, 16061, 16071, 16115, 16155, 16176, 16190, 16200, 16207],
         "fact": "request edge3; eleven setup states3..13 or15..25; preload accepts edge15; wait14 to row26; terminal34. Each row contributes two checks; each inner tile one check, activation wait, compute launch and compute wait"},
        {"file": v + "gate_linear_top_linear_engine_cached_static_bank_Pipeline_VITIS_LOOP_90_11_VITIS_LOOP_91_12_VITI.v",
         "lines": [7083, 7084, 7228, 7502, 7510, 9900, 9908, 10454],
         "fact": "flattened counter bound64/128 with three exit registers; no blocking condition; sampled compute67/131, versus report69/133"},
        {"file": v + "gate_linear_top_linear_engine_cached_static_bank_64_4_16_64_256_Pipeline_VITIS_LOOP_52_1.v",
         "lines": [873, 881, 2001],
         "fact": "W8 byte index exits at B, direct loop-exit done with no drain registers: sampled B, reportB+2"},
        {"file": v + "gate_linear_top_linear_engine_cached_static_bank_64_4_16_64_256_Pipeline_VITIS_LOOP_57_2.v",
         "lines": [877, 885],
         "fact": "W4 packed-byte loop also has direct exit done and sampled B=NK/2"},
        {"file": v + "gate_linear_top_linear_engine_cached_static_bank_Pipeline_VITIS_LOOP_73_5_VITIS_LOOP_74_6.v",
         "lines": [788, 796, 1898], "fact": "64-iteration initialization, direct exit done: sampled64"},
        {"file": v + "gate_linear_top_linear_engine_cached_static_bank_Pipeline_VITIS_LOOP_82_9_VITIS_LOOP_83_10.v",
         "lines": [4108, 4135, 6545, 6553, 7241],
         "fact": "same256 byte-request sequence and11 drain registers; unchanged AXI adapter/slave yields sampled524 including257 blocked edges, report269"},
        {"file": v + "gate_linear_top_linear_engine_cached_static_bank_Pipeline_VITIS_LOOP_134_16_VITIS_LOOP_135_17.v",
         "lines": [1109, 1138, 1151, 1159, 2001],
         "fact": "same64 output-request sequence and14 drain registers; unchanged AXI adapter/slave yields sampled209 including131 blocked edges, report80"},
        {"file": "hardware/linear_axi_tb.sv", "lines": [51, 156, 167, 387, 392, 398, 540],
         "fact": "unchanged bounded registered-response slave and passive posedge timestamp difference; no additional DDR delay"},
    ]
    for item in source_evidence:
        if not (ROOT / item["file"]).is_file():
            raise ValueError("Missing static-bank64 reviewed source: " + item["file"])
    completed = {}
    for point in points:
        cycles = predictions[point["id"]]["bounded_axi_adjusted_component_cycles"] + 15
        completed[point["id"]] = {
            "cycles": cycles, "time_at_target_clock_ns": cycles * evidence["target_clock_ns"],
            "boundary_cycles_relative_to_report_components": 15,
            "controller_cycles_relative_to_actual_preload_duration": 17,
            "preload_sampled_duration_minus_report_latency": -2,
            "additional_preload_or_range_request_stalls_predicted": 0,
            "kind": "complete_transaction_prediction_for_declared_bounded_memory_model",
        }
    return {
        "kernel_variant": "weight_cache_static_bank", "source_revision": "static_bank_index_v1",
        "synthesis_recorded_utc": evidence["synthesis_recorded_utc"],
        "component_freeze_recorded_utc": component_freeze["recorded_utc"],
        "point_definitions": points, "source_evidence": source_evidence, "predictions": completed,
        "formula": "B+17+R*(2+O*(277+I*(595 if W4 else659)))",
        "fit": "none; actual flattened pipeline and controller counts, unchanged bounded AXI service; no RTL target or reserved observations used",
        "inner_stage_derivation": {
            "initialization_sampled_cycles": 64, "activation_sampled_cycles": 524,
            "compute_sampled_cycles_w4_w8": [67, 131], "store_sampled_cycles": 209,
            "inner_control_and_wait_edges": 4, "inner_total_cycles_w4_w8": [595, 659],
            "nominal_inner_correction_cycles": 253, "output_total_cycles": 277,
            "explanation": "inner=524+(67 or131)+4; output=64+209+4; row2; fixed17 relative to sampled preloadB",
        },
        "memory_assumptions": [
            "three AXI adapter files equal cached16 by ordinary text comparison",
            "same bounded one-outstanding registered-response slave, byte-read and word-write request order",
            "idle weight port at start; eleven setup states hide first range data; byte consumer slower than bursts",
            "no additional DDR delay, contention or arbitration; full tiles only",
        ],
        "development_check": "No target or reserved dispatch existed at freeze; predictions require RTL validation.",
        "counter_boundary": "passive posedge accepted top start to asserted top done; done-start; restart excluded",
    }


def completed_prediction_freeze(evidence, points, predictions, component_freeze):
    """Count controller boundaries from unchanged RTL, not observed residuals."""
    if evidence["kernel_variant"] == "weight_cache_static_bank":
        return completed_static64_freeze(evidence, points, predictions, component_freeze)
    if evidence["profile"]["lanes"] == 32:
        return completed_cached32_freeze(evidence, points, predictions, component_freeze)
    if evidence["profile"]["lanes"] != 16:
        raise ValueError("Review this profile's generated RTL before defining its complete prediction; LANES16 boundaries are not transferable evidence.")
    v = "build/linear_hls_cached/lanes16/project/solution/syn/verilog/"
    source_evidence = [
        {"file": v + "gate_linear_top.v", "lines": [1025, 1152],
         "fact": "top accepts at edge0, registers engine start in state2; engine accepts edge2"},
        {"file": v + "gate_linear_top_linear_engine_cached_16_4_16_64_256_s.v",
         "lines": [2948, 2960, 5692],
         "fact": "engine request edge3; eleven setup states; preload accepts edge15; row-check follows completion; terminal state26-to34 adds one"},
        {"file": v + "gate_linear_top_linear_engine_cached_16_4_16_64_256_Pipeline_VITIS_LOOP_50_1.v",
         "lines": [374, 393, 400, 431, 753],
         "fact": "index starts0, increments each unblocked active edge, index==B ends loop and asserts done on last-byte edge; sampled duration B, report B+2"},
        {"file": v + "gate_linear_top_linear_engine_cached_16_4_16_64_256_Pipeline_VITIS_LOOP_55_2.v",
         "lines": [379, 397, 405, 437, 759],
         "fact": "equivalent packed-byte counter; duration B=NK/2"},
        {"file": "hardware/linear_axi_tb.sv", "lines": [51, 156, 167, 387, 392, 398, 540],
         "fact": "one outstanding read, registered first response, edge-based accepted-start/done counters; latency is done-start, without inclusive +1"},
    ]
    for item in source_evidence:
        if not (ROOT / item["file"]).is_file():
            raise ValueError("Missing independently reviewed RTL source: " + item["file"])
    completed = {}
    for point in points:
        pred = predictions[point["id"]]
        cycles = pred["bounded_axi_adjusted_component_cycles"] + 17 - 2
        completed[point["id"]] = {
            "cycles": cycles, "time_at_target_clock_ns": cycles * evidence["target_clock_ns"],
            "boundary_cycles_relative_to_report_components": 15,
            "controller_cycles_relative_to_actual_preload_duration": 17,
            "preload_sampled_duration_minus_report_latency": -2,
            "additional_preload_or_range_request_stalls_predicted": 0,
            "kind": "complete_transaction_prediction_for_declared_bounded_memory_model",
        }
    return {
        "kernel_variant": VARIANT, "synthesis_recorded_utc": evidence["synthesis_recorded_utc"],
        "component_freeze_recorded_utc": component_freeze["recorded_utc"],
        "point_definitions": points, "source_evidence": source_evidence,
        "predictions": completed,
        "formula": "P+R*(2+O*(150+I*(534 if W4 else 790)))+253*R*O*I+127*R*O+17-2",
        "fit": "none;17 controller cycles and -2 preload boundary are source-derived, not fitted target residuals",
        "memory_assumptions": [
            "unchanged three-bundle AXI adapter and bounded registered-response byte-array slave",
            "idle weight port at call start; same full-matrix range request and max16-beat32-bit burst splitting",
            "11 setup states hide initial preload delivery; sustained burst supply exceeds one-byte-per-cycle consumer",
            "no additional DDR delay, cross-port contention, or external arbitration stalls",
            "full row tiles and exactly the four declared points; no board bandwidth claim",
        ],
        "development_check": "all three previously saved targets have zero error under this source-derived completion; no target coefficient is fitted",
        "counter_boundary": "passive posedge accepted top start to asserted top done; difference of timestamps, no inclusive +1; restart excluded",
    }


def main():
    global OUT, REPORT, VARIANT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="lanes16", choices=("lanes16", "lanes32", "lanes64"))
    parser.add_argument("--static-bank", action="store_true", help="Inspect the isolated lanes64 static_bank_index_v1 build")
    parser.add_argument("--freeze-nominal", action="store_true")
    parser.add_argument("--freeze-complete", action="store_true")
    args = parser.parse_args()
    if args.static_bank and args.profile != "lanes64":
        raise ValueError("The static-bank repair is scoped to lanes64.")
    VARIANT = "weight_cache_static_bank" if args.static_bank else "weight_cache_pilot"
    suffix = "_static_bank_lanes64" if args.static_bank else "" if args.profile == "lanes16" else "_" + args.profile
    rtl_root = ROOT / ("results/goal4/rtl_cached_static_bank" if args.static_bank else "results/goal4/rtl_cached")
    OUT = ROOT / f"results/goal4/cached_characterization{suffix}.json"
    REPORT = ROOT / f"reports/goal4_cached_linear{suffix}.md"
    if args.freeze_complete and args.profile not in ("lanes16", "lanes32") and not args.static_bank:
        raise ValueError("Complete predictions for new profiles require their own generated-RTL review.")
    evidence, points = synthesis(read_json(ROOT / "configs/project.json"), args.profile, args.static_bank)
    predictions = {p["id"]: prediction(p, evidence) for p in points}
    existing = read_json(OUT) if OUT.is_file() else {}
    freeze = existing.get("nominal_freeze")
    if freeze:
        if (freeze["synthesis"] != evidence or freeze["point_definitions"] != points
                or freeze["predictions"] != predictions or freeze["kernel_variant"] != VARIANT):
            raise ValueError("Frozen evidence or predictions changed; do not overwrite this study.")
    elif args.freeze_nominal:
        dispatched = [
            rel(path) for p in points if args.static_bank or p["role"] == "reserved"
            for path in (rtl_root / p["id"]).glob("*run.json")
        ]
        if dispatched:
            raise ValueError("Cannot backdate a cached freeze after reserved dispatch: " + str(dispatched))
        freeze = {
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "kernel_variant": VARIANT, "point_definitions": points,
            "synthesis_recorded_utc": evidence["synthesis_recorded_utc"],
            "synthesis": evidence, "predictions": predictions,
            "reserved_run_records_absent_at_freeze": True,
            "fit": "none; unresolved residual and complete latency prediction remain null",
        }
    completion = existing.get("source_derived_complete_freeze")
    completion_definition = completed_prediction_freeze(evidence, points, predictions, freeze) if freeze and (args.profile in ("lanes16", "lanes32") or args.static_bank) else None
    if completion:
        if {k: v for k, v in completion.items() if k not in ("recorded_utc", "reserved_run_records_absent_at_freeze")} != completion_definition:
            raise ValueError("Completed source-derived freeze changed; preserve its predictions.")
    elif args.freeze_complete:
        if not freeze:
            raise ValueError("Original component freeze must exist before completion.")
        dispatched = [
            rel(path) for p in points if args.static_bank or p["role"] == "reserved"
            for path in (rtl_root / p["id"]).glob("*run.json")
        ]
        if dispatched:
            raise ValueError("Cannot complete freeze after reserved dispatch: " + str(dispatched))
        completion = {
            **completion_definition,
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "reserved_run_records_absent_at_freeze": True,
        }
    result = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(), "kernel_variant": VARIANT,
        "synthesis": evidence, "nominal_freeze": freeze,
        "source_derived_complete_freeze": completion,
        "points": [{"point": p, "prediction": predictions[p["id"]],
                    "observation": observation(p, evidence, predictions[p["id"]])} for p in points],
    }
    if completion:
        for item in result["points"]:
            item["complete_prediction"] = completion["predictions"][item["point"]["id"]]
            obs = item["observation"]
            if obs["status"] == "measured_rtl":
                expected = item["complete_prediction"]["cycles"]
                obs["complete_prediction_error_cycles"] = [value - expected for value in obs["latency_cycles"]]
                obs["complete_prediction_error_percent"] = [(value - expected) / value * 100 for value in obs["latency_cycles"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"artifact": rel(OUT), "frozen": bool(freeze), "points": [
        {"id": p["point"]["id"], "status": p["observation"]["status"],
         "adjusted_component_cycles": p["prediction"]["bounded_axi_adjusted_component_cycles"],
         "complete_prediction_cycles": p.get("complete_prediction", {}).get("cycles"),
         "residual_cycles": p["observation"].get("measured_minus_adjusted_component_cycles")}
        for p in result["points"]]}, indent=2))


if __name__ == "__main__":
    main()

