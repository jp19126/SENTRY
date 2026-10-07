"""Offline analysis of the fixed three-profile, twelve-point Goal 4 study.

No vendor tools, training, regression fitting, or board measurements are performed.
Freeze nominal reports first; --freeze-calibrated records the targeted/passive
diagnostic correction before reserved dispatch. Later refreshes preserve both.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/goal4"
SUMMARY = OUT / "characterization_summary.json"
UNKNOWN = {"", "?", "-", "NA", "N/A", "undef"}
POINT_FIELDS = ("id", "profile", "role", "rows", "inner", "outputs", "weight_bits", "repetitions")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def relative(path):
    return path.relative_to(ROOT).as_posix()


def numeric(value):
    value = str(value or "").strip().replace(",", "")
    if value in UNKNOWN:
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"Unexpected report number: {value!r}") from exc
    if number < 0:  # HLS uses negative sentinels for unknown bounds.
        return None
    return int(number) if number.is_integer() else number


def bounds(element):
    if element is None:
        return {"min": None, "max": None}
    nested = element.find("range")
    if nested is not None:
        return {key: numeric(nested.findtext(key)) for key in ("min", "max")}
    value = numeric(element.text)
    return {"min": value, "max": value}


def one_loop(parent):
    loops = [child for child in parent if child.tag.startswith("VITIS_LOOP_")]
    if len(loops) != 1:
        raise ValueError("Expected this study's single nested row/output/inner loop.")
    return loops[0]


def checked_study(config):
    hls = config["hardware"]["hls"]
    profiles = {p["id"]: p for p in hls["profiles"]}
    if set(profiles) != {"lanes4", "lanes8", "lanes16"}:
        raise ValueError("Analyzer is scoped to lanes4/8/16 only.")
    for name, p in profiles.items():
        expected = (int(name[5:]), 4, 16, 64, 256)
        actual = tuple(p[k] for k in ("lanes", "tile_rows", "tile_outputs", "tile_inner", "max_rows"))
        if actual != expected:
            raise ValueError(f"Physical profile changed: {name}")
    points = [{key: p[key] for key in POINT_FIELDS} for p in hls["characterization_points"]]
    expected = set()
    for lanes, bits in ((4, (4, 8, 4)), (8, (8, 4, 8)), (16, (4, 8, 4))):
        for (inner, outputs), width in zip(((256, 256), (256, 1024), (1024, 256)), bits):
            expected.add((f"target_l{lanes}_k{inner}_n{outputs}_w{width}",
                          f"lanes{lanes}", "target", 16, inner, outputs, width, 2))
    for lanes, inner, outputs in ((4, 256, 256), (8, 256, 1024), (16, 1024, 256)):
        expected.add((f"reserved_l{lanes}_k{inner}_n{outputs}_w8",
                      f"lanes{lanes}", "reserved", 256, inner, outputs, 8, 2))
    if len(points) != 12 or {tuple(p[k] for k in POINT_FIELDS) for p in points} != expected:
        raise ValueError("Declared points differ from the fixed nine-target/three-reserved plan.")
    return profiles, points


def profile_evidence(name, physical, hardware):
    build = ROOT / "build/linear_hls" / name
    report_dir = build / "project/solution/syn/report"
    run_path = build / "run.json"
    top_path = report_dir / "gate_linear_top_csynth.xml"
    if not run_path.is_file() or not top_path.is_file():
        return {"status": "missing_synthesis", "profile": physical}
    run = read_json(run_path)
    if run.get("returncode") != 0 or not run.get("hls_synthesis_performed"):
        return {"status": "synthesis_not_successful", "profile": physical,
                "run_record": relative(run_path)}
    if run["selected_profile"] != physical or run["hardware"]["part"] != hardware["part"]:
        raise ValueError(f"Synthesis/config mismatch for {name}")
    top = ET.parse(top_path).getroot()
    assignments = top.find("UserAssignments")
    target_ns = numeric(assignments.findtext("TargetClockPeriod"))
    if (assignments.findtext("Part") != hardware["part"]
            or assignments.findtext("FlowTarget") != "vivado"
            or target_ns != 1000.0 / hardware["clock_objective_mhz"]
            or numeric(assignments.findtext("ClockUncertainty")) != hardware["hls"]["clock_uncertainty_ns"]):
        raise ValueError(f"Report/config device, flow or clock mismatch: {name}")
    engine_path = report_dir / f"linear_engine_{physical['lanes']}_4_16_64_256_s_csynth.xml"
    engine = ET.parse(engine_path).getroot()
    row = one_loop(engine.find("PerformanceEstimates/SummaryOfLoopLatency"))
    output = one_loop(row)
    inner = one_loop(output)
    if bounds(output.find("TripCount"))["min"] != 1 or bounds(inner.find("TripCount"))["min"] != 1:
        raise ValueError("Nominal derivation requires the report's minimum tile trip count of one.")
    row_min = bounds(row.find("IterationLatency"))["min"]
    output_min = bounds(output.find("IterationLatency"))["min"]
    inner_bounds = bounds(inner.find("IterationLatency"))
    if None in (row_min, output_min, inner_bounds["min"], inner_bounds["max"]):
        raise ValueError(f"Missing report-defined loop iteration bounds: {name}")
    terms = {"row_control_cycles": row_min - output_min,
             "output_tile_init_store_control_cycles": output_min - inner_bounds["min"],
             "inner_tile_w4_cycles": inner_bounds["min"],
             "inner_tile_w8_cycles": inner_bounds["max"]}
    if min(terms.values()) < 0:
        raise ValueError(f"Unexpected nested-loop accounting: {name}")
    # Exactly five pipelined stages occur in this unchanged fixed-study source.
    stages = []
    for path in report_dir.glob("*Pipeline*csynth.xml"):
        tree = ET.parse(path).getroot()
        loop = one_loop(tree.find("PerformanceEstimates/SummaryOfLoopLatency"))
        order = int(re.search(r"VITIS_LOOP_(\d+)_", loop.tag).group(1))
        latency = tree.find("PerformanceEstimates/SummaryOfOverallLatency")
        stages.append((order, {
            "report": relative(path), "loop": loop.tag,
            "latency_min_cycles": numeric(latency.findtext("Best-caseLatency")),
            "latency_max_cycles": numeric(latency.findtext("Worst-caseLatency")),
            "iteration_latency": bounds(loop.find("IterationLatency")),
            "trip_count": bounds(loop.find("TripCount")),
            "pipeline_ii": numeric(loop.findtext("PipelineII")),
            "resources": {e.tag: numeric(e.text) for e in tree.find("AreaEstimates/Resources")},
        }))
    if len(stages) != 5:
        raise ValueError(f"Expected five stage reports, found {len(stages)} for {name}")
    stage_rows = dict(zip(("initialize", "activation_load", "weight_load", "compute", "store"),
                         [stage for _, stage in sorted(stages)]))
    mac_trips = stage_rows["compute"]["trip_count"]
    if mac_trips != {"min": 4096 // physical["lanes"], "max": 8192 // physical["lanes"]}:
        raise ValueError(f"Compute loop no longer matches the one/two digit-phase schedule: {name}")
    resources = {e.tag: numeric(e.text) for e in top.find("AreaEstimates/Resources")}
    return {
        "status": "reported", "profile": physical, "part": assignments.findtext("Part"),
        "report_version": top.findtext("ReportVersion/Version"),
        "synthesis_recorded_utc": run["recorded_utc"], "run_record": relative(run_path),
        "top_report": relative(top_path), "engine_report": relative(engine_path),
        "target_clock_ns": target_ns,
        "clock_uncertainty_ns": numeric(assignments.findtext("ClockUncertainty")),
        "estimated_clock_ns": numeric(top.findtext("PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod")),
        "estimated_resources": resources, "nominal_terms": terms, "stages": stage_rows,
        "top_report_latency": {key: numeric(top.findtext("PerformanceEstimates/SummaryOfOverallLatency/" + tag))
                               for key, tag in (("min", "Best-caseLatency"), ("avg", "Average-caseLatency"),
                                                ("max", "Worst-caseLatency"))},
        "unknown_residual": "Top/control boundary and memory-system stalls; not set to zero or fitted.",
    }


def nominal(point, profile):
    if profile["status"] != "reported":
        return None
    rows = (point["rows"] + 3) // 4
    outputs = point["outputs"] // 16
    inner = point["inner"] // 64
    t = profile["nominal_terms"]
    inner_cycles = t[f"inner_tile_w{point['weight_bits']}_cycles"]
    components = {
        "row_control": rows * t["row_control_cycles"],
        "output_tile_init_store_control": rows * outputs * t["output_tile_init_store_control_cycles"],
        "inner_tiles": rows * outputs * inner * inner_cycles,
    }
    cycles = sum(components.values())
    return {"cycles": cycles, "components_cycles": components,
            "row_tiles": rows, "output_tiles": outputs, "inner_tiles": inner,
            "nominal_time_at_target_clock_ns": cycles * profile["target_clock_ns"],
            "kind": "report_derived_schedule_component_not_measured_total_latency",
            "top_control_memory_residual_cycles": None}


def reserved_dispatched(points):
    paths = []
    for point in points:
        if point["role"] != "reserved":
            continue
        folder = OUT / "cosim" / point["id"]
        paths.extend(relative(p) for p in folder.glob("*run.json"))
        paths.extend(relative(p) for p in (OUT / "rtl" / point["id"]).glob("*run.json"))
        active = ROOT / "build/linear_hls" / point["profile"] / "cosim_run.json"
        if active.is_file() and read_json(active).get("point", {}).get("id") == point["id"]:
            paths.append(relative(active))
    return paths


def cosim_evidence(point, profile):
    direct_path = OUT / "rtl" / point["id"] / "run.json"
    if direct_path.is_file():
        run = read_json(direct_path)
        v = run.get("verified") or {}
        if run.get("point") != point:
            raise ValueError("Direct RTL point mismatch: " + point["id"])
        issues = []
        if not run.get("passed") or not v.get("passed") or run.get("returncode") != 0:
            issues.append("direct_rtl_not_passed")
        if run.get("synthesis_recorded_utc") != profile.get("synthesis_recorded_utc"):
            issues.append("synthesis_association_unverified")
        latencies = v.get("latency_cycles") or []
        if len(latencies) != point["repetitions"]:
            issues.append("transaction_count_mismatch")
        return {"status":"measured_rtl" if not issues else "unusable", "issues":issues,
                "report_status":"direct_rtl_verified" if not issues else "failed",
                "report":relative(direct_path), "run_record":relative(direct_path),
                "recorded_utc":run.get("recorded_utc"),
                "latency_cycles":({"min":min(latencies),"avg":sum(latencies)/len(latencies),"max":max(latencies)} if latencies and not issues else None),
                "interval_cycles":({k:v["interval_cycles"] for k in ("min","avg","max")} if not issues else None),
                "total_execution_cycles":v.get("total_execution_cycles") if not issues else None,
                "transactions":run.get("transactions"),
                "verification_method":run.get("verification_method"),
                "memory_model":run.get("memory_model"),
                "boundary":"Direct RTL accepted-start to done cycles under bounded AXI model; exact outputs checked twice. Start-start includes AXI-Lite restart. Not vendor UVM or board timing."}
    folder = OUT / "cosim" / point["id"]
    path, run_path = folder / "cosim.rpt", folder / "run.json"
    verification_path = folder / "gate_cosim_verified.json"
    if verification_path.is_file() and run_path.is_file():
        verified, run = read_json(verification_path), read_json(run_path)
        if run.get("point") != point:
            raise ValueError(f"Saved point differs from declaration: {point['id']}")
        issues = []
        if (verified.get("passed") is not True or run.get("returncode") != 0
                or not run.get("hls_rtl_cosimulation_passed")):
            issues.append("run_or_verification_not_passed")
        if (profile["status"] != "reported"
                or run.get("synthesis_recorded_utc") != profile.get("synthesis_recorded_utc")):
            issues.append("synthesis_association_unverified")
        return {"status": "measured_rtl" if not issues else "unusable", "issues": issues,
                "report_status": "verified_custom_launch" if not issues else "failed",
                "report": relative(verification_path), "run_record": relative(run_path),
                "recorded_utc": run.get("recorded_utc"),
                "latency_cycles": verified.get("latency_cycles"),
                "interval_cycles": verified.get("interval_cycles"),
                "total_execution_cycles": verified.get("total_execution_cycles"),
                "verification_method": run.get("verification_method"),
                "boundary": "Vendor-generated RTL/testbench with detailed loop profiling suppressed; top-level cycle logic and exact output checks retained. Simulator AXI model, not board timing."}
    if not path.is_file() or not run_path.is_file():
        return {"status": "not_recorded", "latency_cycles": None, "interval_cycles": None}
    run = read_json(run_path)
    if run.get("point") != point:
        raise ValueError(f"Saved point differs from declaration: {point['id']}")
    row = next((line for line in path.read_text(encoding="utf-8").splitlines()
                if re.match(r"\s*\|\s*Verilog\s*\|", line)), None)
    if row is None:
        raise ValueError(f"No Verilog data row in {path}")
    cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
    if len(cells) != 9:
        raise ValueError(f"Unexpected co-simulation report columns: {path}")
    reported = {"latency_cycles": dict(zip(("min", "avg", "max"), map(numeric, cells[2:5]))),
                "interval_cycles": dict(zip(("min", "avg", "max"), map(numeric, cells[5:8]))),
                "total_execution_cycles": numeric(cells[8])}
    issues = []
    if cells[1] != "Pass" or run.get("returncode") != 0 or not run.get("hls_rtl_cosimulation_passed"):
        issues.append("run_or_report_not_passed")
    if profile["status"] != "reported" or run.get("synthesis_recorded_utc") != profile.get("synthesis_recorded_utc"):
        issues.append("synthesis_association_unverified")
    return {"status": "measured_rtl" if not issues else "unusable", "issues": issues,
            "report_status": cells[1], "report": relative(path), "run_record": relative(run_path),
            "recorded_utc": run.get("recorded_utc"), **reported,
            "boundary": "Vendor RTL transaction cycles under the simulator AXI model, not board timing."}


def freeze_calibrated(points, profiles, predictions, nominal_freeze, now):
    """Derive this engine's correction from one passive diagnostic, then check nine targets."""
    if not nominal_freeze:
        raise ValueError("Freeze the original nominal predictions before calibration.")
    dispatched = reserved_dispatched(points)
    if dispatched:
        raise ValueError("Reserved work already dispatched; cannot claim prior calibration: " + ", ".join(dispatched))
    point = next(p for p in points if p["id"] == "target_l4_k256_n256_w4")
    folder = OUT / "rtl_diagnostic" / point["id"]
    path = folder / "run.json"
    run = read_json(path)
    diagnostic = run.get("diagnostics", [])
    if (run.get("point") != point or run.get("passed") is not True or run.get("returncode") != 0
            or run.get("verification_method") != "direct_xsim_bounded_axi_testbench"
            or run.get("synthesis_recorded_utc") != profiles["lanes4"]["synthesis_recorded_utc"]
            or len(diagnostic) != 2 or len(run.get("transactions", [])) != 2):
        raise ValueError("A successful associated two-transaction passive diagnostic is required.")
    R, O, I = 4, 16, 4
    stage_durations = []
    for index, (d, transaction) in enumerate(zip(diagnostic, run["transactions"])):
        if (d.get("checks_passed") is not True or d.get("diagnostic_errors") != 0
                or d.get("passive") is not True or d.get("index") != index
                or transaction.get("exact_output_passed") is not True
                or any(d.get(k) != transaction.get(k) for k in ("start_cycle", "done_cycle", "latency_cycles"))):
            raise ValueError("Passive diagnostic checks or exact-output association failed.")
        duration = {}
        for name, stage in d["stages"].items():
            calls = R * O * (I if name in ("activation_load", "weight_load", "compute") else 1)
            timing = stage["duration_cycles"]
            if (stage["calls"] != calls or stage["finishes"] != calls
                    or timing["min"] != timing["max"] or timing["sum"] != timing["min"] * calls):
                raise ValueError("Unexpected stage calls or varying duration in the fixed diagnostic.")
            duration[name] = timing["min"]
        expected_states = [2, R+1, R*(O+1), R*O*(duration["initialize"]+1), R*O*(I+1),
                           R*O*I*(max(duration["activation_load"], duration["weight_load"])+1),
                           R*O*I, R*O*I*(duration["compute"]+1), R*O*(duration["store"]+1), 0]
        if d["engine_fsm_state_cycles"] != expected_states or sum(expected_states) != d["latency_cycles"]:
            raise ValueError("Diagnostic FSM occupancy does not support the structural formula.")
        stage_durations.append(duration)
    if stage_durations[0] != stage_durations[1]:
        raise ValueError("Repeated diagnostic stage durations disagree.")
    duration = stage_durations[0]
    reference = profiles["lanes4"]["nominal_terms"]
    correction = {
        "fixed_top_terminal_cycles": 3,
        "extra_row_control_cycles": 0,
        "extra_output_tile_cycles": duration["initialize"] + duration["store"] + 4
                                    - reference["output_tile_init_store_control_cycles"],
        "extra_inner_tile_cycles": max(duration["activation_load"], duration["weight_load"])
                                   + duration["compute"] + 4 - reference["inner_tile_w4_cycles"],
    }
    calibrated = {}
    targeted_evidence = []
    for p in points:
        profile = profiles[p["profile"]]
        if profile["nominal_terms"]["row_control_cycles"] != 2:
            raise ValueError("The verified row-control structure changed.")
        base = predictions[p["id"]]
        R, O, I = base["row_tiles"], base["output_tiles"], base["inner_tiles"]
        delta = correction["fixed_top_terminal_cycles"] + R*O*(correction["extra_output_tile_cycles"]
                                                              + I*correction["extra_inner_tile_cycles"])
        calibrated[p["id"]] = {
            "cycles": base["cycles"] + delta, "correction_to_frozen_nominal_cycles": delta,
            "time_at_target_clock_ns": (base["cycles"]+delta)*profile["target_clock_ns"],
            "kind": "target_calibrated_direct_rtl_bounded_axi_prediction",
        }
        if p["role"] == "target":
            measured = cosim_evidence(p, profile)
            if (measured["status"] != "measured_rtl"
                    or measured.get("verification_method") != "direct_xsim_bounded_axi_testbench"
                    or any(v != calibrated[p["id"]]["cycles"] for v in measured["latency_cycles"].values())):
                raise ValueError("Structural correction must explain every completed targeted observation: " + p["id"])
            targeted_evidence.append({"point_id": p["id"], "run_record": measured["run_record"],
                                      "latency_cycles": measured["latency_cycles"]})
    if len(targeted_evidence) != 9:
        raise ValueError("All nine targeted cases are required.")
    return {
        "recorded_utc": now, "nominal_freeze_recorded_utc": nominal_freeze["recorded_utc"],
        "point_definitions": points, "predictions": calibrated,
        "method": "Passive stage/FSM decomposition, checked against nine targeted cases; no regression fit",
        "correction": correction,
        "formula": "C_nominal + 3 + R*O*(127 + 1021*I)",
        "diagnostic_run_record": relative(path), "diagnostic_log": relative(folder/"simulation.log"),
        "diagnostic_source": relative(folder/"linear_rtl_diagnostic.sv"),
        "diagnostic_checks_passed": True, "diagnostic_transactions": diagnostic,
        "stage_durations_cycles": duration, "targeted_evidence": targeted_evidence,
        "reserved_run_records_absent_at_freeze": True,
        "boundary": "Initial tiled engine only, complete row tiles, unchanged one-outstanding registered AXI model. No board timing, row-tail validation, cached-engine transfer, or minimum-II claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-nominal", action="store_true",
                        help="Save all 12 report-derived predictions before any reserved dispatch.")
    parser.add_argument("--freeze-calibrated", action="store_true",
                        help="Freeze the passive-diagnostic correction after nine targets, before reserved dispatch.")
    args = parser.parse_args()
    config = read_json(ROOT / "configs/project.json")
    profiles, points = checked_study(config)
    evidence = {name: profile_evidence(name, p, config["hardware"]) for name, p in profiles.items()}
    predictions = {p["id"]: nominal(p, evidence[p["profile"]]) for p in points}
    previous = read_json(SUMMARY) if SUMMARY.is_file() else {}
    frozen = previous.get("nominal_freeze")
    now = datetime.now(timezone.utc).isoformat()
    if frozen:
        if frozen["point_definitions"] != points:
            raise ValueError("Frozen point definitions changed; preserve the original comparison.")
        for name, old in frozen["profiles"].items():
            current = evidence[name]
            if current != old:
                raise ValueError(f"Synthesis evidence changed after nominal freeze: {name}")
        predictions = frozen["predictions"]
    elif args.freeze_nominal:
        if any(p is None for p in predictions.values()):
            raise ValueError("All three successful synthesis/submodule reports are needed before freezing.")
        dispatched = reserved_dispatched(points)
        if dispatched:
            raise ValueError("Reserved work already dispatched; cannot claim a prior freeze: " + ", ".join(dispatched))
        frozen = {"recorded_utc": now, "point_definitions": points, "profiles": evidence,
                  "predictions": predictions, "fit": "none",
                  "reserved_run_records_absent_at_freeze": True}
    calibrated_freeze = previous.get("calibrated_freeze")
    if calibrated_freeze:
        if (calibrated_freeze["point_definitions"] != points or not frozen
                or calibrated_freeze["nominal_freeze_recorded_utc"] != frozen["recorded_utc"]):
            raise ValueError("Frozen calibration association changed.")
    elif args.freeze_calibrated:
        calibrated_freeze = freeze_calibrated(points, evidence, predictions, frozen, now)
    rows = []
    for point in points:
        prediction = predictions[point["id"]]
        measured = cosim_evidence(point, evidence[point["profile"]])
        errors = None
        if prediction and measured["status"] == "measured_rtl":
            errors = {}
            for statistic, observed in measured["latency_cycles"].items():
                if observed is not None:
                    delta = prediction["cycles"] - observed
                    errors[statistic] = {"nominal_minus_measured_cycles": delta,
                                         "relative_error": delta / observed if observed else None,
                                         "measured_minus_nominal_residual_cycles": -delta}
        corrected = calibrated_freeze["predictions"][point["id"]] if calibrated_freeze else None
        corrected_errors = None
        if corrected and measured["status"] == "measured_rtl":
            corrected_errors = {stat: {"calibrated_minus_measured_cycles": corrected["cycles"]-value,
                                      "relative_error": (corrected["cycles"]-value)/value if value else None}
                                for stat, value in measured["latency_cycles"].items() if value is not None}
        rows.append({"point": point, "nominal": prediction, "cosimulation": measured,
                     "calibrated": corrected, "calibrated_latency_errors": corrected_errors,
                     "calibrated_reserved_comparison_frozen_before_dispatch": bool(calibrated_freeze) if point["role"] == "reserved" else None,
                     "nominal_latency_errors": errors,
                     "reserved_comparison_frozen_before_dispatch": bool(frozen) if point["role"] == "reserved" else None})
    result = {
        "recorded_utc": now, "scope": "Fixed three-profile/twelve-point integer-linear-kernel characterization",
        "nominal_freeze": frozen, "profiles": evidence, "points": rows,
        "nominal_formula": "R * (row_control + O * (output_tile_init_store_control + I * inner_tile_cycles[bits]))",
        "calibrated_freeze": calibrated_freeze,
        "calibration": calibrated_freeze["correction"] if calibrated_freeze else None,
        "boundary": "Original nominal errors retained. Optional targeted/passive calibration applies only to initial tiled RTL under its bounded AXI model. No board timing, cached-engine transfer, complete-detector latency, or feasibility claim.",
        "measured_rtl_points": sum(r["cosimulation"]["status"] == "measured_rtl" for r in rows),
        "reserved_predictions_ready": bool(frozen),
        "calibrated_reserved_predictions_ready": bool(calibrated_freeze),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    fields = [*POINT_FIELDS, "nominal_cycles", "nominal_frozen", "rtl_status",
              "latency_min", "latency_avg", "latency_max", "interval_min", "interval_avg",
              "interval_max", "total_execution_cycles", "nominal_minus_measured_avg_cycles",
              "relative_error_avg", "calibrated_cycles", "calibrated_frozen",
              "calibrated_minus_measured_avg_cycles", "calibrated_relative_error_avg",
              "target_clock_ns", "estimated_clock_ns"]
    with (OUT / "characterization_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            p, measured = row["point"], row["cosimulation"]
            profile = evidence[p["profile"]]
            values = dict(p, nominal_cycles=(row["nominal"] or {}).get("cycles"),
                          nominal_frozen=bool(frozen), rtl_status=measured["status"],
                          target_clock_ns=profile.get("target_clock_ns"),
                          estimated_clock_ns=profile.get("estimated_clock_ns"),
                          total_execution_cycles=measured.get("total_execution_cycles"))
            for kind in ("latency", "interval"):
                for statistic in ("min", "avg", "max"):
                    values[f"{kind}_{statistic}"] = (measured.get(kind + "_cycles") or {}).get(statistic)
            error = (row["nominal_latency_errors"] or {}).get("avg", {})
            values["nominal_minus_measured_avg_cycles"] = error.get("nominal_minus_measured_cycles")
            values["relative_error_avg"] = error.get("relative_error")
            values["calibrated_cycles"] = (row["calibrated"] or {}).get("cycles")
            values["calibrated_frozen"] = bool(calibrated_freeze)
            calibrated_error = (row["calibrated_latency_errors"] or {}).get("avg", {})
            values["calibrated_minus_measured_avg_cycles"] = calibrated_error.get("calibrated_minus_measured_cycles")
            values["calibrated_relative_error_avg"] = calibrated_error.get("relative_error")
            writer.writerow(values)
    print(json.dumps({"summary": relative(SUMMARY), "measured_rtl_points": result["measured_rtl_points"],
                      "nominal_frozen": bool(frozen), "calibrated_frozen": bool(calibrated_freeze),
                      "full_detector_complete": False}, indent=2))


if __name__ == "__main__":
    main()
