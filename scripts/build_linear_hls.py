"""Preflight or build the source-only Goal 4 fixed-BERT integer HLS engine.

Default invocation only checks the shared config. Unknown hardware is a blocker,
never replaced by an example FPGA/clock/profile. No tools are installed or boards
programmed. --csim/--synthesize are explicit vendor invocations once configured.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import shutil

ROOT = Path(__file__).resolve().parents[1]
PROFILE_FIELDS = ("lanes", "tile_rows", "tile_outputs", "tile_inner", "max_rows")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def configured_linear_study(config, cached=False):
    """Return one declared engine family; historical profiles stay reproducible."""
    hls = config["hardware"]["hls"]
    if not cached:
        return hls.get("profiles") or [], hls.get("characterization_points") or []
    block = hls.get("cached_linear")
    if block is None:  # Existing lanes16 pilot/config compatibility.
        profiles = [p for p in hls.get("profiles", []) if p["id"] == "lanes16"]
        points = [p for p in hls.get("characterization_points", []) if p["profile"] == "lanes16"]
    else:
        profiles = block.get("profiles") or []
        points = block.get("characterization_points") or []
    if not 1 <= len(profiles) <= config["methods"]["max_engine_profiles"]:
        raise ValueError("Configure one to at most three cached physical profiles.")
    ids = [p["id"] for p in profiles]
    if len(set(ids)) != len(ids):
        raise ValueError("Cached physical profile names must be unique.")
    for profile in profiles:
        lanes = profile.get("lanes")
        if (lanes not in (16, 32, 64) or profile["id"] != f"lanes{lanes}"
                or tuple(profile.get(k) for k in PROFILE_FIELDS) != (lanes, 4, 16, 64, 256)):
            raise ValueError("Cached profiles support lanes16/32/64 with fixed 4/16/64 tiles and maxrows 256.")
    point_ids = [p["id"] for p in points]
    if len(point_ids) != len(set(point_ids)):
        raise ValueError("Cached point names must be unique.")
    for point in points:
        if (point["profile"] not in ids or not re.fullmatch(r"[a-z0-9_-]+", point["id"])
                or point.get("role") not in ("target", "reserved")
                or point.get("rows") not in (16, 256)
                or (point.get("inner"), point.get("outputs")) not in ((256, 256), (256, 1024), (1024, 256))
                or point.get("weight_bits") not in (4, 8) or point.get("repetitions") != 2):
            raise ValueError("Invalid bounded cached characterization point.")
    return profiles, points


def active_linear_profiles(config):
    """The single current comparison space, without historical-profile union."""
    variant = config["hardware"]["hls"].get("active_kernel_variant", "initial_tiled")
    if variant not in ("initial_tiled", "weight_cache_pilot"):
        raise ValueError("Unknown active linear engine variant.")
    return configured_linear_study(config, cached=variant == "weight_cache_pilot")[0]



def active_linear_variants(config):
    """Source identity separately keyed by profile; numeric frozen shapes stay intact."""
    hls = config["hardware"]["hls"]
    profiles = active_linear_profiles(config)
    family = hls.get("active_kernel_variant", "initial_tiled")
    if family == "initial_tiled":
        return {p["id"]: "initial_tiled" for p in profiles}
    declared = (hls.get("cached_linear") or {}).get("profile_kernel_variants")
    variants = ({p["id"]: "weight_cache_pilot" for p in profiles}
                if declared is None else dict(declared))
    if set(variants) != {p["id"] for p in profiles}:
        raise ValueError("Cached source variant mapping must name exactly the active profiles.")
    for name, variant in variants.items():
        if variant not in ("weight_cache_pilot", "weight_cache_static_bank"):
            raise ValueError("Unknown cached source variant for " + name)
        if variant == "weight_cache_static_bank" and name != "lanes64":
            raise ValueError("static_bank_index_v1 was synthesized only for lanes64.")
    return variants


def path_from_root(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def preflight(config, selected_profile, cached_pilot=False):
    hardware = config["hardware"]
    hls = hardware.get("hls") or {}
    missing = []
    for field in ("allocated_board", "part", "interface", "clock_objective_mhz",
                  "resource_limits", "tool_versions", "execution_host"):
        if hardware.get(field) is None or hardware.get(field) == "":
            missing.append("hardware." + field)
    for field in ("tool_executable", "driver", "flow_target", "interface_tcl"):
        if not hls.get(field):
            missing.append("hardware.hls." + field)
    profiles, _ = configured_linear_study(config, cached_pilot)
    if not profiles:
        missing.append("hardware.hls.profiles")
    profile = None
    if profiles:
        if not isinstance(profiles, list) or not 1 <= len(profiles) <= config["methods"]["max_engine_profiles"]:
            raise ValueError("Configure one to at most three actual physical profiles.")
        ids = [value.get("id") for value in profiles]
        if any(not isinstance(value, str) or not re.fullmatch(r"[a-z0-9_-]+", value) for value in ids):
            raise ValueError("Physical profiles need unique lowercase ordinary names.")
        if len(set(ids)) != len(ids):
            raise ValueError("Physical profile names must be unique.")
        if selected_profile is None and len(profiles) == 1:
            profile = profiles[0]
        elif selected_profile in ids:
            profile = profiles[ids.index(selected_profile)]
        elif selected_profile is not None:
            raise ValueError("The requested physical profile is not configured.")
        else:
            missing.append("--profile selecting one actual profile")
        if profile:
            for field in PROFILE_FIELDS:
                value = profile.get(field)
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    missing.append(f"hardware.hls.profiles[{profile['id']}].{field}")
            if all(isinstance(profile.get(field), int) and profile[field] > 0 for field in PROFILE_FIELDS):
                if profile["tile_inner"] % profile["lanes"]:
                    raise ValueError("tile_inner must be divisible by lanes.")
                if profile["tile_inner"] > 1024 or profile["tile_outputs"] > 1024:
                    raise ValueError("Tile exceeds the observed fixed BERT encoder dimension.")
                if profile["tile_rows"] > profile["max_rows"]:
                    raise ValueError("tile_rows exceeds the profile's max_rows.")
    if hls.get("driver") and hls["driver"] not in ("vitis_hls", "vitis-run"):
        raise ValueError("Supported documented launchers are vitis_hls or vitis-run.")
    if hls.get("flow_target") and hls["flow_target"] not in ("vivado", "vitis"):
        raise ValueError("Explicit flow_target must be vivado or vitis.")
    if hardware.get("clock_objective_mhz") is not None:
        if not isinstance(hardware["clock_objective_mhz"], (int, float)) or hardware["clock_objective_mhz"] <= 0:
            raise ValueError("Actual clock objective must be positive MHz.")
    if hardware.get("resource_limits") is not None and not hardware["resource_limits"]:
        missing.append("hardware.resource_limits (usable resource allocation)")
    if hardware.get("tool_versions") is not None:
        versions = hardware["tool_versions"]
        if not isinstance(versions, dict) or not versions.get(hls.get("driver")):
            missing.append("hardware.tool_versions[selected HLS driver]")
    for field in ("tool_executable", "interface_tcl"):
        if hls.get(field) and not path_from_root(hls[field]).is_file():
            missing.append("existing file for hardware.hls." + field)
    inventory = ROOT / "results/goal2/training_run.json"
    if not inventory.exists():
        missing.append("results/goal2/training_run.json actual shape inventory")
        counts = {}
    else:
        entries = read_json(inventory)["encoder_matrix_shapes"]
        counts = Counter(tuple(value["shape"]) for value in entries)
        expected = {(256, 256): 16, (1024, 256): 4, (256, 1024): 4}
        if dict(counts) != expected:
            raise ValueError("Observed encoder shapes differ from this scoped HLS source.")
        counts = {f"{inner}->{output}": count for (output, inner), count in counts.items()}
    arithmetic = ROOT / "reports/quant_arithmetic_check.json"
    if not arithmetic.exists() or not read_json(arithmetic).get("passed"):
        missing.append("passed existing reports/quant_arithmetic_check.json software contract")
    return {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "state": "blocked_missing_hardware" if missing else "configured_not_built",
        "missing": missing, "actual_encoder_shapes": counts,
        "selected_profile": profile, "hardware": hardware,
        "numerical_contract": "gate_quant.py signed W4/W8 x A8, int32 accumulation; FP32 rescale/bias outside this integer kernel",
        "software_arithmetic_evidence": "reports/quant_arithmetic_check.json (reused; not HLS validation)",
        "hls_c_simulation_performed": False, "hls_synthesis_performed": False,
        "hardware_measurements": None,
    }


def tcl_literal(value):
    text = str(value).replace("\\", "/")
    if any(character in text for character in "{}\n\r"):
        raise ValueError("Unsupported Tcl path/config character.")
    return "{" + text + "}"


def emit_build(config, readiness, action, point=None, cached_pilot=False, static_bank=False):
    hls = config["hardware"]["hls"]
    profile = readiness["selected_profile"]
    build_root = ("build/linear_hls_cached_static_bank" if static_bank else
                  "build/linear_hls_cached" if cached_pilot else "build/linear_hls")
    build = ROOT / build_root / profile["id"]
    if static_bank and (build / "run.json").exists():
        raise ValueError("Preserve the recorded static-bank attempt; no automatic overwrite or extra synthesis.")
    build.mkdir(parents=True, exist_ok=True)
    defines = "\n".join(
        "#define GATE_" + field.upper() + " " + str(profile[field])
        for field in PROFILE_FIELDS)
    profile_path = build / "selected_profile.hpp"
    profile_text = defines + "\n"
    if action == "cosim":
        if not profile_path.is_file() or profile_path.read_text(encoding="utf-8") != profile_text:
            raise ValueError("Synthesize this physical profile before RTL co-simulation.")
    else:
        profile_path.write_text(profile_text, encoding="utf-8")
    top = """#include "linear_engine.hpp"
#include "selected_profile.hpp"
extern "C" void gate_linear_top(const ap_int<8>* activations,
                                const ap_uint<8>* weights, ap_int<32>* output,
                                int rows, int inner, int outputs, int weight_bits,
                                int* status) {
    const bool valid = gate::linear_engine<GATE_LANES, GATE_TILE_ROWS,
        GATE_TILE_OUTPUTS, GATE_TILE_INNER, GATE_MAX_ROWS>(
            activations, weights, output, rows, inner, outputs, weight_bits);
    *status = valid ? 0 : 1;
}
"""
    if static_bank:
        top = top.replace('"linear_engine.hpp"', '"linear_engine_cached_static_bank.hpp"').replace("gate::linear_engine<", "gate::linear_engine_cached_static_bank<")
    elif cached_pilot:
        top = top.replace('"linear_engine.hpp"', '"linear_engine_cached.hpp"').replace("gate::linear_engine<", "gate::linear_engine_cached<")
    top_path = build / "gate_linear_top.cpp"
    if action == "cosim":
        if not top_path.is_file() or top_path.read_text(encoding="utf-8") != top:
            raise ValueError("Existing synthesized top differs from the configured engine.")
    else:
        top_path.write_text(top, encoding="utf-8")
    hardware_dir = (ROOT / config["paths"]["hardware"]).resolve()
    testbench = hardware_dir / "linear_engine_tb.cpp"
    interface_tcl = path_from_root(hls["interface_tcl"]).resolve()
    build_tcl = hardware_dir / "build_linear.tcl"
    if static_bank:
        for name in ("linear_engine_cached_static_bank.hpp", "linear_engine.hpp",
                     "linear_engine_tb.cpp", "build_linear.tcl"):
            shutil.copyfile(hardware_dir / name, build / name)
        shutil.copyfile(interface_tcl, build / interface_tcl.name)
        shutil.copyfile(Path(__file__), build / "build_linear_hls.py")
        testbench = build / "linear_engine_tb.cpp"
        interface_tcl = build / interface_tcl.name
        build_tcl = build / "build_linear.tcl"
        readiness["source_snapshot"] = str((build / "linear_engine_cached_static_bank.hpp").relative_to(ROOT))
        readiness["testbench_snapshot"] = str(testbench.relative_to(ROOT))
    # Vitis 2025.2 treats quotes after -I as part of a relative path.
    include_paths = ((build.resolve().as_posix(),) if static_bank else
                     (hardware_dir.as_posix(), build.resolve().as_posix()))
    if any(any(char.isspace() for char in path) for path in include_paths):
        raise ValueError("The HLS build requires include paths without whitespace.")
    cflags = "-std=c++14 " + " ".join("-I" + path for path in include_paths)
    values = {
        "gate_project": (build / "project").resolve().as_posix(),
        "gate_top": top_path.resolve().as_posix(),
        "gate_testbench": testbench.as_posix(),
        "gate_cflags": cflags, "gate_flow_target": hls["flow_target"],
        "gate_part": config["hardware"]["part"],
        "gate_clock_ns": 1000.0 / config["hardware"]["clock_objective_mhz"],
        "gate_clock_uncertainty_ns": hls["clock_uncertainty_ns"],
        "gate_interface_tcl": interface_tcl.as_posix(),
        "gate_action": action,
        "gate_cosim_argv": " ".join(str(point[field]) for field in
            ("rows", "inner", "outputs", "weight_bits", "repetitions")) if point else "",
    }
    settings = build / "settings.tcl"
    settings.write_text("\n".join("set " + key + " " + tcl_literal(value)
                                  for key, value in values.items()) + "\n", encoding="utf-8")
    return build, settings, build_tcl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="Name of one actual configured physical profile.")
    parser.add_argument("--cached-pilot", action="store_true", help="Build one declared cached linear profile separately from the historical tiled study")
    parser.add_argument("--static-bank", action="store_true", help="Prepare/build the isolated static_bank_index_v1 repair at lanes64 only; implies --cached-pilot")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--emit", action="store_true", help="Emit top/Tcl only after all hardware fields are resolved.")
    actions.add_argument("--csim", action="store_true", help="Run the configured vendor C simulation.")
    actions.add_argument("--synthesize", action="store_true", help="Run C simulation, then one vendor HLS synthesis.")
    actions.add_argument("--cosim-point", help="Run one declared runtime point against existing RTL.")
    args = parser.parse_args()
    if args.static_bank:
        args.cached_pilot = True
        if args.profile != "lanes64" or args.cosim_point:
            raise ValueError("The final static-bank repair is scoped to --profile lanes64; no vendor co-sim route.")
    config = read_json(ROOT / "configs/project.json")
    point = None
    _, declared_points = configured_linear_study(config, args.cached_pilot)
    if args.cosim_point:
        matches = [p for p in declared_points
                   if p["id"] == args.cosim_point]
        if len(matches) != 1:
            raise ValueError("Select one declared bounded characterization point.")
        point = matches[0]
        if args.profile and args.profile != point["profile"]:
            raise ValueError("Point and selected physical profile disagree.")
        args.profile = point["profile"]
    if args.cached_pilot and args.cosim_point:
        raise ValueError("Use the direct RTL launcher for cached runtime validation")
    readiness = preflight(config, args.profile, args.cached_pilot)
    readiness["kernel_variant"] = ("weight_cache_static_bank" if args.static_bank else
                                    "weight_cache_pilot" if args.cached_pilot else "initial_tiled")
    if args.static_bank:
        header = ROOT / "hardware/linear_engine_cached_static_bank.hpp"
        revision = re.search(r'^#define GATE_LINEAR_STATIC_BANK_REVISION "([^"\n]+)"',
                             header.read_text(encoding="utf-8"), re.M)
        if revision is None or revision.group(1) != "static_bank_index_v1":
            raise ValueError("Static-bank source revision is missing or changed.")
        readiness["source_revision"] = revision.group(1)
        readiness["source_header"] = str(header.relative_to(ROOT))
    report_path = ROOT / ("reports/hls_static_bank_build_readiness.json" if args.static_bank else
                          "reports/hls_build_readiness.json")
    save(report_path, readiness)
    print(json.dumps({key: readiness[key] for key in
                      ("state", "missing", "actual_encoder_shapes", "selected_profile")}, indent=2))
    if readiness["missing"]:
        return 2
    if not (args.emit or args.csim or args.synthesize or args.cosim_point):
        return 0
    if (ROOT / "results/goal2/pause.request").exists():
        raise RuntimeError("User pause marker is present; no hardware tool will be launched.")
    action = "cosim" if point else ("synthesize" if args.synthesize else "csim")
    point_dir = ROOT / "results/goal4/cosim" / point["id"] if point else None
    if point_dir and (point_dir / "run.json").exists():
        previous = read_json(point_dir / "run.json")
        synthesis_record = ROOT / "build/linear_hls" / point["profile"] / "run.json"
        current_synthesis = read_json(synthesis_record) if synthesis_record.is_file() else {}
        if (previous.get("hls_rtl_cosimulation_passed") and previous.get("point") == point
                and previous.get("synthesis_recorded_utc") == current_synthesis.get("recorded_utc")
                and previous.get("synthesis_recorded_utc") is not None):
            print(json.dumps({"reused": str(point_dir), "point": point["id"]}))
            return 0
    build, settings, script = emit_build(config, readiness, action, point, args.cached_pilot, args.static_bank)
    if point and not (build / "project/solution/syn/verilog/gate_linear_top.v").is_file():
        raise ValueError("RTL is missing; synthesize the physical profile first.")
    if args.emit:
        save(report_path, readiness)
        print("Generated source and Tcl only; no vendor compilation.")
        return 0
    hls = config["hardware"]["hls"]
    executable = str(path_from_root(hls["tool_executable"]).resolve())
    command = ([executable, "-f", str(script)] if hls["driver"] == "vitis_hls" else
               [executable, "--mode", "hls", "--tcl", str(script)])
    environment = os.environ.copy()
    environment["GATE_HLS_SETTINGS"] = str(settings.resolve())
    record = dict(readiness, command=command, action=action, point=point, state="vendor_tool_running")
    save(build / ("cosim_run.json" if point else "run.json"), record)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    log_path = build / ("cosim_vendor.log" if point else "vendor.log")
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=build, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    record.update(returncode=completed.returncode, state="vendor_tool_returned",
                  note="A zero tool exit is not timing closure or complete-model cost evidence; inspect actual reports.")
    tool_log = log_path.read_text(encoding="utf-8", errors="replace")
    csim_passed = completed.returncode == 0 and "CSim done with 0 errors." in tool_log
    synthesis_report = build / "project/solution/syn/report/gate_linear_top_csynth.xml"
    synthesis_passed = (csim_passed and action == "synthesize"
                        and "Finished Command csynth_design" in tool_log
                        and synthesis_report.is_file())
    record.update(hls_c_simulation_performed=csim_passed,
                  hls_synthesis_performed=synthesis_passed,
                  synthesis_report=str(synthesis_report) if synthesis_passed else None)
    if point:
        sim_dir = build / "project/solution/sim/verilog"
        verification_path = sim_dir / "gate_cosim_verified.json"
        verification = read_json(verification_path) if verification_path.is_file() else {}
        passed = (completed.returncode == 0 and "GATE COSIM VERIFIED" in tool_log
                  and verification.get("passed") is True)
        synthesis_record = read_json(build / "run.json")
        record.update(hls_rtl_cosimulation_passed=passed,
                      hls_synthesis_performed=False,
                      verification_method="vendor_generated_cosim_with_detailed_profiling_disabled",
                      synthesis_reused_from=str(build / "run.json"),
                      synthesis_recorded_utc=synthesis_record["recorded_utc"])
        point_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(log_path, point_dir / "vendor.log")
        evidence_names = ("gate_cosim_verified.json", "gate_linear_top.result.lat.rb",
            "gate_linear_top.performance.result.transaction.xml", "gate_linear_top.autotb.v",
            "run_sim.tcl", "gate_linear_top.autotb.v.before_profile_disable",
            "run_sim.tcl.before_profile_disable")
        for name in evidence_names:
            if (sim_dir / name).is_file():
                shutil.copyfile(sim_dir / name, point_dir / name)
        postcheck = sim_dir.parent / "wrapc_pc/temp0.log"
        if postcheck.is_file():
            shutil.copyfile(postcheck, point_dir / "postcheck.log")
        original_dir = sim_dir / "verification_repair" / "_".join(str(point[field]) for field in
            ("rows", "inner", "outputs", "weight_bits", "repetitions"))
        if original_dir.is_dir():
            for original in original_dir.iterdir():
                if original.is_file():
                    shutil.copyfile(original, point_dir / ("generated_original_" + original.name))
        save(point_dir / "run.json", record)
    save(build / ("cosim_run.json" if point else "run.json"), record)
    save(report_path, record)
    print(json.dumps({"returncode": completed.returncode, "log": str(log_path)}, indent=2))
    return completed.returncode or (1 if point and not record.get("hls_rtl_cosimulation_passed") else 0)


if __name__ == "__main__":
    raise SystemExit(main())

