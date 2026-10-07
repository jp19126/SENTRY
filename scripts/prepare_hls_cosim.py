"""Repair only generated co-simulation profiling and require checked completion.

Run after cosim_design -setup. The synthesized DUT, AXI agents, top-level cycle
monitor and output comparisons are not edited. This is specific to the observed
Vitis 2025.2 generated launcher, not a general simulator patching framework.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import re

MONITOR = """dataflow_monitor U_dataflow_monitor(
    .clock(AESL_clock),
    .reset(~rst),
    .finish(all_finish));"""
LAUNCH = """if {[file isfile run_xsim.sh]} {
\tset ret [catch {eval exec "sh ./run_xsim.sh | tee temp2.log" >&@ stdout} err]
}
 set ::env(RDI_USE_JDK11) true
    df_record_move"""
STRICT_LAUNCH = """if {![file isfile run_xsim.sh]} {
    error "Missing generated XSIM launcher"
}
set ret [catch {exec sh -e ./run_xsim.sh | tee temp2.log >&@ stdout} err]
set ::env(RDI_USE_JDK11) true
if {$ret != 0} {
    error "XSIM launcher failed: $err"
}
rtl_sim_check
# GATE: detailed profiling was omitted; no module/loop CSV archive is expected."""
POSTCHECK = """set ret [catch {exec ./cosim.pc.exe {*}$ap_argv | tee temp0.log >&@ stdout} err]

sc_sim_check $ret $err "temp3.log"""
STRICT_POSTCHECK = """set ret [catch {exec ./cosim.pc.exe {*}$ap_argv | tee temp0.log >&@ stdout} err]
if {$ret != 0} {
    error "C postcheck failed: $err"
}
sc_sim_check $ret $err "temp0.log"""
LATENCY_NAME = "gate_linear_top.result.lat.rb"
TRANSACTION_NAME = "gate_linear_top.performance.result.transaction.xml"
MARKER_NAME = "gate_cosim_verified.json"
LATENCY_KEYS = {
    "MAX_LATENCY", "MIN_LATENCY", "AVER_LATENCY", "MAX_THROUGHPUT",
    "MIN_THROUGHPUT", "AVER_THROUGHPUT", "TOTAL_EXECUTE_TIME",
}


def replace_once(text, old, new, description):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one {description}; generated layout changed.")
    return text.replace(old, new, 1)


def point_arguments(launcher):
    matches = re.findall(r"^set ap_argv \{([0-9 ]+)\}$", launcher, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError("Expected one generated numeric point argument list.")
    values = [int(value) for value in matches[0].split()]
    if len(values) != 5 or values[-1] != 2:
        raise ValueError("Expected rows, inner, outputs, bits and two repetitions.")
    return values


def repaired_contents(testbench, launcher):
    testbench = replace_once(
        testbench, MONITOR,
        "// GATE: omit only the generated detailed module/loop profiling monitor.",
        "detailed profiling monitor instance")
    launcher = replace_once(launcher, LAUNCH, STRICT_LAUNCH, "XSIM launcher/profiling archive block")
    launcher = replace_once(launcher, POSTCHECK, STRICT_POSTCHECK, "C postcheck block")
    return testbench, launcher


def prepare(sim_dir):
    paths = [sim_dir / "gate_linear_top.autotb.v", sim_dir / "run_sim.tcl"]
    originals = [path.read_text(encoding="utf-8") for path in paths]
    argv = point_arguments(originals[1])
    repaired = repaired_contents(*originals)  # Validate every expected block before writing.
    archive = sim_dir / "verification_repair" / "_".join(map(str, argv))
    for path, original in zip(paths, originals):
        previous = archive / path.name
        if previous.exists() and previous.read_text(encoding="utf-8") != original:
            raise ValueError(f"Different original already retained at {previous}; archive it before retry.")
    archive.mkdir(parents=True, exist_ok=True)
    diff = []
    for path, original, replacement in zip(paths, originals, repaired):
        (archive / path.name).write_text(original, encoding="utf-8")
        diff.extend(difflib.unified_diff(
            original.splitlines(keepends=True), replacement.splitlines(keepends=True),
            fromfile=path.name + " (generated)", tofile=path.name + " (verification repair)"))
    (archive / "changes.diff").write_text("".join(diff), encoding="utf-8")
    for path, replacement in zip(paths, repaired):
        path.write_text(replacement, encoding="utf-8")
    # These exact generated outputs must be fresh; previous attempts are archived by the caller.
    for path in [sim_dir / MARKER_NAME, sim_dir / LATENCY_NAME, sim_dir / TRANSACTION_NAME,
                 sim_dir.parent / "wrapc_pc" / "temp0.log", sim_dir.parent / "wrapc_pc" / "err.log",
                 sim_dir.parent / "tv" / "rtldatafile" / "rtl.gate_linear_top.autotvout_gmem_o.dat",
                 sim_dir.parent / "tv" / "rtldatafile" / "rtl.gate_linear_top.autotvout_status.dat"]:
        path.unlink(missing_ok=True)
    print(f"Prepared verification-only repair; original files and diff: {archive}")


def verify(sim_dir):
    """Called by the Tcl build only after the complete launcher returned successfully."""
    argv = point_arguments((sim_dir / "run_sim.tcl").read_text(encoding="utf-8"))
    for name in (".exit.err", ".aesl_error", "err.log"):
        path = sim_dir / name
        if path.exists() and (name == ".exit.err" or path.read_text(errors="replace").strip() not in ("", "0")):
            raise ValueError(f"RTL simulation error artifact: {path}")
    simulation_log = (sim_dir / "temp2.log").read_text(errors="replace")
    for severity in ("UVM_ERROR", "UVM_FATAL"):
        counts = re.findall(r"^\s*" + severity + r"\s*:\s*(\d+)\s*$", simulation_log, flags=re.MULTILINE)
        if not counts or any(int(count) != 0 for count in counts):
            raise ValueError(f"Fresh simulator log must report {severity}: 0.")
    post_dir = sim_dir.parent / "wrapc_pc"
    error_file = post_dir / "err.log"
    if error_file.exists():
        for line in error_file.read_text(errors="replace").splitlines():
            match = re.search(r"AESL_mErrNo\s*=\s*(\d+)", line)
            if match and int(match[1]) != 0:
                raise ValueError("C postcheck reported output mismatches.")
    post_log = (post_dir / "temp0.log").read_text(errors="replace")
    expected_prefix = f"PASS case rows={argv[0]} inner={argv[1]} outputs={argv[2]} bits={argv[3]}"
    pass_lines = [line.strip() for line in post_log.splitlines() if line.startswith("PASS case ")]
    expected_lines = [f"{expected_prefix} repetition={index} repetitions=2" for index in (1, 2)]
    if pass_lines != expected_lines:
        raise ValueError("Fresh C postcheck must contain exactly the two expected reference PASS lines.")
    latency_text = (sim_dir / LATENCY_NAME).read_text(encoding="utf-8")
    matches = re.findall(r'^\$(\w+) = "(\d+)"$', latency_text, flags=re.MULTILINE)
    if len(matches) != len(LATENCY_KEYS) or {key for key, _ in matches} != LATENCY_KEYS:
        raise ValueError("Incomplete or unexpected top-level latency assignments.")
    cycles = {key: int(value) for key, value in matches}
    transaction_text = (sim_dir / TRANSACTION_NAME).read_text(encoding="utf-8")
    transactions = re.findall(r"^transaction\s+(\d+):\s+(\d+)\s+(\d+|x)\s*$", transaction_text, flags=re.MULTILINE)
    if len(transactions) != 2 or [row[0] for row in transactions] != ["0", "1"] or transactions[1][2] != "x":
        raise ValueError("Expected two fresh top-level transaction timing rows.")
    if transactions[0][2] == "x" or min(cycles.values()) <= 0:
        raise ValueError("Missing positive latency/interval cycle observations.")
    transaction_rows = [
        {"transaction": int(index), "latency_cycles": int(latency),
         "interval_cycles": None if interval == "x" else int(interval)}
        for index, latency, interval in transactions]
    latencies = [row["latency_cycles"] for row in transaction_rows]
    interval = transaction_rows[0]["interval_cycles"]
    if (cycles["MIN_LATENCY"], cycles["AVER_LATENCY"], cycles["MAX_LATENCY"]) != (
            min(latencies), sum(latencies) // 2, max(latencies)):
        raise ValueError("Top-level latency summary disagrees with transaction rows.")
    if any(cycles[key] != interval for key in ("MIN_THROUGHPUT", "AVER_THROUGHPUT", "MAX_THROUGHPUT")):
        raise ValueError("Top-level interval summary disagrees with transaction rows.")
    archive = "verification_repair/" + "_".join(map(str, argv))
    marker = {
        "passed": True,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "argv": argv,
        "latency_cycles": {"min": cycles["MIN_LATENCY"], "avg": cycles["AVER_LATENCY"], "max": cycles["MAX_LATENCY"]},
        "interval_cycles": {"min": cycles["MIN_THROUGHPUT"], "avg": cycles["AVER_THROUGHPUT"], "max": cycles["MAX_THROUGHPUT"]},
        "total_execution_cycles": cycles["TOTAL_EXECUTE_TIME"],
        "transactions": transaction_rows,
        "verification_method": "vendor_generated_cosim_with_detailed_profiling_disabled",
        "detailed_profiling_enabled": False,
        "dut_rtl_modified": False,
        "vendor_cosim_pass_report_generated": False,
        "evidence": [LATENCY_NAME, TRANSACTION_NAME, "temp2.log", "../wrapc_pc/temp0.log",
                     archive + "/gate_linear_top.autotb.v", archive + "/run_sim.tcl", archive + "/changes.diff"],
    }
    (sim_dir / MARKER_NAME).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    print("GATE COSIM VERIFIED " + str(sim_dir / MARKER_NAME))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    sim_dir = args.sim_dir.resolve()
    root = Path(__file__).resolve().parents[1]
    sim_dir.relative_to((root / "build/linear_hls").resolve())
    if sim_dir.name != "verilog" or sim_dir.parent.name != "sim":
        raise ValueError("Expected this repository's generated sim/verilog directory.")
    (verify if args.verify else prepare)(sim_dir)


if __name__ == "__main__":
    main()
