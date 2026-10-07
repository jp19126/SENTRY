"""Run the fixed one-document numerical bridge with C simulation only.

Default is a read-only plan. Requires the explicitly prepared one-case export
and the current successful FP32 synthesis/source association. No new synthesis,
hardware timing claim, final calibration/test access, or target API calls.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/goal4/numerical_bridge"
BUILD = ROOT / "build/numerical_bridge"
PAUSE = ROOT / "results/goal2/pause.request"
CASE = "aeslc:train:panus-s_inbox_1.subject:clean"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def literal(value):
    value = str(value)
    if any(c in value for c in "{}\n\r"):
        raise ValueError("Unsupported Tcl path.")
    return "{" + value + "}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--diagnostic-first-divergence", action="store_true")
    args = parser.parse_args()
    out = DATA / "diagnostic_first_divergence_v1" if args.diagnostic_first_divergence else DATA
    if not args.execute:
        print(json.dumps({"vendor_tool_started": False, "case": CASE,
                          "export_required": str(DATA / "manifest.json"),
                          "action": "C simulation only, optimized ordinary CPU integer reference"}, indent=2))
        return 0
    if PAUSE.exists():
        raise RuntimeError("User pause marker present.")
    if os.name == "nt":
        raise RuntimeError("Execute C simulation using WSL Python and installed Linux Vitis.")
    config = read(ROOT / "configs/project.json")
    manifest = read(DATA / "manifest.json")
    if (manifest["case"] != CASE or manifest["checkpoint"].replace("\\", "/") != "checkpoints/quantized/qat_w8_a8"
            or manifest["shape"] != {"length": 256, "layers": 4, "hidden": 256, "heads": 4, "ffn": 1024}
            or not manifest["cpu_reference"]["integer_reference_exact"]):
        raise RuntimeError("Fixed case/checkpoint/CPU reference mismatch.")
    source = ROOT / "build/fixed_fp32_hls"
    synthesis = read(source / "run.json")
    header = (ROOT / "hardware/fixed_fp32_service.hpp").read_bytes()
    revision = re.search(rb'^#define GATE_FIXED_FP32_SCHEDULE_REVISION "([^"\n]+)"', header, re.M)
    if (not revision or revision.group(1).decode() != manifest["hardware_source_revision"]
            or synthesis.get("service_schedule_revision") != manifest["hardware_source_revision"]
            or synthesis.get("returncode") != 0 or not synthesis.get("c_simulation_passed")
            or not synthesis.get("synthesis_passed") or synthesis.get("status") != "returned"
            or header != (source / "fixed_fp32_service.hpp").read_bytes()):
        raise RuntimeError("Current FP32 source/export/successful synthesis association differs.")
    for item in manifest["files"]:
        size = item["elements"] * (1 if item["dtype"] == "int8" else 4)
        if (DATA / item["file"]).stat().st_size != size:
            raise RuntimeError("Export extent mismatch: " + item["file"])
    if (out / "bridge_run.json").exists():
        raise RuntimeError("Prior numerical bridge run must remain available for review; no automatic overwrite.")
    out.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    for name in ("fixed_fp32_service.hpp", "numerical_bridge_tb.cpp"):
        shutil.copyfile(ROOT / "hardware" / name, BUILD / name)
    shutil.copyfile(source / "gate_fixed_fp32_top.cpp", BUILD / "gate_fixed_fp32_top.cpp")
    include = BUILD.as_posix()
    if any(c.isspace() for c in include):
        raise ValueError("HLS include path must not contain whitespace.")
    flags = "-std=c++14 -O2 -ffp-contract=off -I" + include
    h = config["hardware"]
    commands = [
        "open_project " + literal(BUILD / "project"),
        "set_top gate_fixed_fp32_top",
        "add_files " + literal(BUILD / "gate_fixed_fp32_top.cpp") + " -cflags " + literal(flags),
        "add_files -tb " + literal(BUILD / "numerical_bridge_tb.cpp") + " -cflags " + literal(flags),
        "open_solution solution -flow_target " + h["hls"]["flow_target"],
        "set_part " + literal(h["part"]),
        "create_clock -period " + str(1000 / h["clock_objective_mhz"]),
        "config_compile -unsafe_math_optimizations=false",
        "csim_design", "exit",
    ]
    tcl = BUILD / "bridge.tcl"
    tcl.write_text("\n".join(commands) + "\n")
    command = [h["hls"]["tool_executable"], "--mode", "hls", "--tcl", str(tcl)]
    record = {"recorded_utc": datetime.now(timezone.utc).isoformat(), "case": CASE,
              "source_revision": manifest["hardware_source_revision"],
              "synthesis_recorded_utc": synthesis["recorded_utc"], "command": command,
              "action": "csim_only", "state": "running", "completed": False,
              "compiler_flags": flags, "unsafe_math_optimizations": False,
              "boundary": "Numerical diagnostic, CPU exact integer reference plus actual common FP32 C service; no hardware timing/full-RTL equivalence."}
    save(out / "bridge_run.json", record)
    for name in ("fixed_fp32_service.hpp", "numerical_bridge_tb.cpp", "gate_fixed_fp32_top.cpp", "bridge.tcl"):
        shutil.copyfile(BUILD / name, out / name)
    shutil.copyfile(Path(__file__), out / "run_numerical_bridge.py")
    env = os.environ.copy()
    env["GATE_NUMERICAL_BRIDGE_DIR"] = str(DATA)
    if args.diagnostic_first_divergence:
        env["GATE_NUMERICAL_BRIDGE_DIAGNOSTIC_DIR"] = str(out)
        record["diagnostic"] = "first layer1 attention-output A8 boundary; unchanged arithmetic"
        record["original_run"] = str(DATA / "bridge_run.json")
    if PAUSE.exists():
        record.update(state="paused_before_launch")
        save(out / "bridge_run.json", record)
        raise RuntimeError("User pause marker present.")
    try:
        with (out / "bridge_vendor.log").open("w") as log:
            process = subprocess.Popen(command, cwd=BUILD, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            start = time.monotonic()
            while process.poll() is None:
                if PAUSE.exists() or time.monotonic() - start > 1800:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise RuntimeError("Paused or30-minute bounded C-sim timeout; partial evidence retained.")
                time.sleep(0.5)
        text = (out / "bridge_vendor.log").read_text(errors="replace")
        summary_path = out / "bridge_summary.json"
        summary = read(summary_path) if summary_path.is_file() else None
        completed = (process.returncode == 0 and "CSim done with 0 errors." in text
                     and text.count("GATE_NUMERICAL_BRIDGE_COMPLETE calls=411 ") == 1
                     and "GATE_NUMERICAL_BRIDGE_FAIL" not in text and summary
                     and summary.get("completed") is True and summary.get("case") == CASE
                     and summary.get("source_revision") == manifest["hardware_source_revision"]
                     and summary.get("same_input_quantization_exact") is True
                     and summary.get("integer_accumulators_exact") is True
                     and summary.get("strict_decision_agrees") is True
                     and sum(summary.get("service_calls_by_op", [])) == 411)
        record.update(state="returned", returncode=process.returncode, completed=bool(completed),
                      summary_file=str(summary_path) if summary else None,
                      end_to_end_numerical_acceptance="unresolved; no global error tolerance assigned")
    except Exception as exc:
        record.update(state="stopped", completed=False, error=str(exc))
        save(out / "bridge_run.json", record)
        raise
    save(out / "bridge_run.json", record)
    print(json.dumps(record, indent=2))
    return 0 if record["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

