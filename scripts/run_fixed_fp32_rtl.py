"""Bounded direct RTL checks for all nine fixed FP32 service modes.

Default only lists cases. --execute uses existing successful HLS RTL, creates
only its reachable vendor IP simulation models, and runs two calls per case.
No HLS synthesis, UVM, board execution, or full-model validity is claimed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PORTS = ("x", "y", "z", "accumulators", "output", "codes")
PAUSE = ROOT / "results/goal2/pause.request"
MEMORY_MODEL = "Six independent byte-array ports; one outstanding burst per port; registered response; no additional DDR delay or cross-port contention."


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def cases():
    # Existing C TB tolerances are copied without relaxation.
    return [
        dict(id="dot_panel_tail", op=0, rows=3, width=64, outputs=9, bias=True, scale=1.0, atol=3e-5, rtol=3e-5),
        dict(id="dot_nonfused", op=0, rows=1, width=2, outputs=1, bias=False, scale=1.0, atol=0.0, rtol=0.0),
        dict(id="quantize_ties_clip", op=1, rows=1, width=17, outputs=0, bias=False, scale=0.25, atol=0.0, rtol=0.0),
        dict(id="quantize_nonbinary", op=1, rows=1, width=33, outputs=0, bias=False, scale=f32(0.073), atol=0.0, rtol=0.0),
        dict(id="rescale_2x1024", op=2, rows=2, width=1024, outputs=0, bias=False, scale=0.25, atol=0.0, rtol=0.0),
        dict(id="embedding_order", op=3, rows=1, width=19, outputs=0, bias=False, scale=1.0, atol=0.0, rtol=0.0),
        dict(id="residual_signed", op=4, rows=1, width=19, outputs=0, bias=False, scale=1.0, atol=0.0, rtol=0.0),
        dict(id="layer_norm_3x256", op=5, rows=3, width=256, outputs=0, bias=False, scale=1.0, atol=2e-4, rtol=2e-4),
        dict(id="softmax_2x256", op=6, rows=2, width=256, outputs=0, bias=False, scale=8.0, atol=3e-6, rtol=3e-5),
        dict(id="gelu_signed", op=7, rows=1, width=10, outputs=0, bias=False, scale=1.0, atol=3e-6, rtol=3e-5),
        dict(id="tanh_signed", op=8, rows=1, width=10, outputs=0, bias=False, scale=1.0, atol=3e-6, rtol=3e-5),
    ]


def primary_cases():
    # One actual L256 invocation shape per distinct mode/shape/bias contract.
    # Multiplicities are the common panel-cache schedule, not fitted weights.
    return [
        dict(id="primary_qk", op=0, rows=128, width=64, outputs=64, bias=False, scale=1.0, atol=3e-5, rtol=3e-5, multiplicity=128),
        dict(id="primary_av", op=0, rows=128, width=256, outputs=64, bias=False, scale=1.0, atol=3e-5, rtol=3e-5, multiplicity=32),
        dict(id="primary_layer_norm", op=5, rows=128, width=256, outputs=0, bias=False, scale=1.0, atol=2e-4, rtol=2e-4, multiplicity=18),
        dict(id="primary_attention_softmax", op=6, rows=128, width=256, outputs=0, bias=False, scale=8.0, atol=3e-6, rtol=3e-5, multiplicity=32),
        dict(id="primary_quantize_h256", op=1, rows=128, width=256, outputs=0, bias=False, scale=f32(0.073), atol=0.0, rtol=0.0, multiplicity=40),
        dict(id="primary_quantize_f1024", op=1, rows=32, width=1024, outputs=0, bias=False, scale=f32(0.073), atol=0.0, rtol=0.0, multiplicity=32),
        dict(id="primary_rescale_h256", op=2, rows=128, width=256, outputs=0, bias=False, scale=0.25, atol=0.0, rtol=0.0, multiplicity=40),
        dict(id="primary_rescale_f1024", op=2, rows=32, width=1024, outputs=0, bias=False, scale=0.25, atol=0.0, rtol=0.0, multiplicity=32),
        dict(id="primary_embedding", op=3, rows=128, width=256, outputs=0, bias=False, scale=1.0, atol=0.0, rtol=0.0, multiplicity=2),
        dict(id="primary_residual", op=4, rows=128, width=256, outputs=0, bias=False, scale=1.0, atol=0.0, rtol=0.0, multiplicity=16),
        dict(id="primary_gelu", op=7, rows=32, width=1024, outputs=0, bias=False, scale=1.0, atol=3e-6, rtol=3e-5, multiplicity=32),
        dict(id="primary_pooler", op=0, rows=1, width=256, outputs=64, bias=True, scale=1.0, atol=3e-5, rtol=3e-5, multiplicity=4),
        dict(id="primary_classifier", op=0, rows=1, width=256, outputs=2, bias=True, scale=1.0, atol=3e-5, rtol=3e-5, multiplicity=1),
        dict(id="primary_tanh", op=8, rows=1, width=256, outputs=0, bias=False, scale=1.0, atol=3e-6, rtol=3e-5, multiplicity=1),
        dict(id="primary_classifier_softmax", op=6, rows=1, width=2, outputs=0, bias=False, scale=1.0, atol=3e-6, rtol=3e-5, multiplicity=1),
    ]



def fixture(case, rep):
    """Independent formula references; actual shapes, representative test data."""
    op, rows, width, outputs = (case[k] for k in ("op", "rows", "width", "outputs"))
    count = rows * width
    x, y, z, acc = [0.0], [0.0], [0.0], [0]
    if case["id"] == "dot_nonfused":
        x, y = [-1.0, f32(1 + 2**-23)], [1.0, f32(1 - 2**-23)]
        expected = [0.0]
    elif op == 0:
        x = [((i * 7 + rep * 3) % 17 - 8) / 16 for i in range(count)]
        y = [((i * 11 + rep * 7) % 23 - 11) / 32 for i in range(outputs * width)]
        z = [(o - 2 + rep) / 8 for o in range(outputs)] if case["bias"] else [0.0]
        expected = [(z[o] if case["bias"] else 0.0)
                    + math.fsum(x[r * width + k] * y[o * width + k] for k in range(width))
                    for r in range(rows) for o in range(outputs)]
    elif op == 1:
        if case["id"] == "quantize_ties_clip":
            probes = [-1000, -128.5, -128, -127.5, -126.5, -2.5, -1.5, -0.5, -0.0,
                      0.5, 1.5, 2.5, 125.5, 126.5, 127, 127.5, 1000]
            x = [v * case["scale"] for v in (probes if rep == 0 else list(reversed(probes)))]
        else:
            x = [f32((i % 33 - 16) * f32(0.037)) for i in range(count)]
            if rep: x.reverse()
        expected = [max(-128, min(127, round(f32(v / case["scale"])))) for v in x]
    elif op == 2:
        acc = [((i + rep * 3) % 37 - 18) * 17 for i in range(count)]
        y = [(c + 1) / 2048 for c in range(width)]
        z = [(c - 511 + rep) / 1024 for c in range(width)]
        expected = [f32(f32(f32(a) * f32(case["scale"] * y[i % width])) + z[i % width])
                    for i, a in enumerate(acc)]
    elif op in (3, 4):
        x = [(i % 19 - 9 + rep) / 8 for i in range(count)]
        y = [(i % 5 - 2) / 4 for i in range(count)]
        z = [(i % 3 - 1) / 2 for i in range(count)]
        if op == 3: x[0], y[0], z[0] = 16777216.0, -16777216.0, 1.0 + rep
        expected = [f32(f32(a + b) + c) if op == 3 else f32(a + b) for a, b, c in zip(x, y, z)]
    elif op == 5:
        # Repeat the same three C-bring-up row patterns for the full invocation.
        patterns = ([1.25] * width, [(c % 13 - 6 + rep) / 8 for c in range(width)],
                    [128 + (c % 9 - 4 + rep) / 4 for c in range(width)])
        x = [v for r in range(rows) for v in patterns[r % 3]]
        y = [1 + (c % 5 - 2) / 16 for c in range(width)]
        z = [(c % 7 - 3 + rep) / 32 for c in range(width)]
        expected = []
        for r in range(rows):
            row = x[r * width:(r + 1) * width]
            mean = math.fsum(row) / width
            variance = math.fsum((v - mean)**2 for v in row) / width
            expected += [(v - mean) / math.sqrt(variance + 1e-12) * y[c] + z[c] for c, v in enumerate(row)]
    elif op == 6:
        largest = struct.unpack("<f", bytes.fromhex("ffff7f7f"))[0]
        x = [1000.0 + ((i + rep * 3) % 17 - 8) for i in range(count)]
        y = [-largest if i % 11 == 0 and i % width != 0 else 0.0 for i in range(count)]
        expected = []
        for r in range(rows):
            scores = [x[r * width + c] / case["scale"] + y[r * width + c] for c in range(width)]
            maximum = max(scores)
            exps = [math.exp(v - maximum) for v in scores]
            denominator = math.fsum(exps)
            expected += [v / denominator for v in exps]
    else:
        probes = [-10, -3, -1, -0.125, -0.0, 0.0, 0.125, 1, 3, 10]
        x = [probes[i % len(probes)] for i in range(count)]
        if rep: x.reverse()
        expected = [0.5 * v * (1 + math.erf(v / math.sqrt(2))) if op == 7 else math.tanh(v) for v in x]
    arrays = {"x": [f32(v) for v in x], "y": [f32(v) for v in y],
              "z": [f32(v) for v in z], "accumulators": acc}
    return arrays, expected



def prepare_fixture(folder, case):
    references = []
    extents = None
    for rep in range(2):
        arrays, expected = fixture(case, rep)
        current = {}
        for name, values in arrays.items():
            raw = b"".join(struct.pack("<i" if name == "accumulators" else "<f", value) for value in values)
            (folder / f"{name}_{rep}.hex").write_text("".join(f"{byte:02x}\n" for byte in raw))
            current[name] = len(raw)
        # AXI writes use32-bit beats; final WSTRB may contain only the payload tail.
        # Allocate complete bus words while checking exactly len(expected) codes.
        current.update(output=4 if case["op"] == 1 else len(expected) * 4,
                       codes=((len(expected) + 3) // 4) * 4 if case["op"] == 1 else 4)
        if extents is not None and extents != current:
            raise ValueError("Repetition extents differ.")
        extents = current
        references.append(expected)
    save(folder / "reference.json", {"case": case, "expected": references,
                                    "reference": "Independent Python math formula with FP32 input packing and explicit nearest-even quantization."})
    return references, extents


def check_output(folder, case, references):
    checks = []
    for rep, expected in enumerate(references):
        path = folder / f"actual_{rep}.hex"
        if not path.is_file():
            checks.append({"index": rep, "passed": False, "error": "missing_raw_output"})
            continue
        tokens = path.read_text().split()
        if any(re.fullmatch(r"[0-9a-fA-F]{2}", token) is None for token in tokens):
            checks.append({"index": rep, "passed": False, "error": "invalid_or_unknown_raw_byte"})
            continue
        raw = bytes(int(token, 16) for token in tokens)
        if len(raw) != len(expected) * (1 if case["op"] == 1 else 4):
            checks.append({"index": rep, "passed": False, "error": "wrong_output_extent"})
            continue
        actual = list(struct.unpack("<" + ("b" if case["op"] == 1 else "f") * len(expected), raw))
        failures, max_abs, max_rel = [], 0.0, 0.0
        for i, (value, reference) in enumerate(zip(actual, expected)):
            error = abs(value - reference)
            relative = error / abs(reference) if abs(reference) > 1e-30 else 0.0
            if math.isfinite(error):
                max_abs, max_rel = max(max_abs, error), max(max_rel, relative)
            if (not math.isfinite(value) or error > case["atol"] + case["rtol"] * abs(reference)):
                failures.append({"index": i, "actual": value if math.isfinite(value) else str(value),
                                 "expected": reference, "absolute_error": error if math.isfinite(error) else str(error)})
        sums = [math.fsum(actual[r * case["width"]:(r + 1) * case["width"]]) for r in range(case["rows"])] if case["op"] == 6 else []
        normalization = all(math.isfinite(v) and abs(v - 1) <= 2e-5 for v in sums)
        checks.append({"index": rep, "passed": not failures and normalization,
                       "values": len(expected), "maximum_absolute_error": max_abs,
                       "maximum_relative_error": max_rel, "atol": case["atol"], "rtol": case["rtol"],
                       "exact_codes_required": case["op"] == 1, "softmax_row_sums": sums,
                       "mismatch_count": len(failures), "first_mismatches": failures[:8]})
    return checks


def paused():
    if PAUSE.exists():
        raise RuntimeError("User pause marker present; no further vendor commands.")


def command_run(command, cwd, log_path, timeout=900):
    paused()
    environment = os.environ.copy()
    tool_bin = Path(read(ROOT / "configs/project.json")["hardware"]["hls"]["vivado_executable"]).parent
    environment["PATH"] = str(tool_bin) + os.pathsep + environment.get("PATH", "")
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        while process.poll() is None:
            if PAUSE.exists() or time.monotonic() - started > timeout:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError("Paused or bounded vendor-command timeout; see " + str(log_path))
            time.sleep(0.5)
        return process.returncode


def source_evidence(config):
    source = ROOT / "build/fixed_fp32_hls"
    record = read(source / "run.json")
    if not (record.get("returncode") == 0 and record.get("status") == "returned"
            and record.get("action") == "synthesize" and record.get("c_simulation_passed")
            and record.get("synthesis_passed")):
        raise RuntimeError("Current successful FP32 C simulation and synthesis required.")
    for name in ("fixed_fp32_service.hpp", "fixed_fp32_service_tb.cpp"):
        if (source / name).read_bytes() != (ROOT / "hardware" / name).read_bytes():
            raise RuntimeError("Current source differs from synthesized copy: " + name)
    revision = re.search(r'^#define GATE_FIXED_FP32_SCHEDULE_REVISION "([^"\n]+)"',
                         (source / "fixed_fp32_service.hpp").read_text(), re.M)
    if not revision or record.get("service_schedule_revision") != revision.group(1):
        raise RuntimeError("Current schedule revision must match successful synthesis.")
    top = source / "project/solution/syn/verilog/gate_fixed_fp32_top.v"
    report = source / "project/solution/syn/report/gate_fixed_fp32_top_csynth.xml"
    tree = ET.parse(report).getroot()
    start = datetime.fromisoformat(record["recorded_utc"]).timestamp()
    if min(top.stat().st_mtime, report.stat().st_mtime) + 1 < start:
        raise RuntimeError("RTL/report predates current synthesis attempt.")
    if (tree.findtext("UserAssignments/Part") != config["hardware"]["part"]
            or float(tree.findtext("UserAssignments/TargetClockPeriod")) != record["clock_ns"]):
        raise RuntimeError("Actual report device/clock mismatch.")
    control = (top.parent / "gate_fixed_fp32_top_control_s_axi.v").read_text()
    for name, address in (("OP", 0x10), ("X", 0x18), ("Y", 0x24), ("Z", 0x30),
                          ("ACCUMULATORS", 0x3c), ("OUTPUT_R", 0x48), ("CODES", 0x54),
                          ("ROWS", 0x60), ("WIDTH", 0x68), ("OUTPUTS", 0x70),
                          ("SCALE", 0x78), ("DOT_BIAS", 0x80), ("STATUS", 0x88)):
        match = re.search(r"ADDR_" + name + r"_DATA_0\s*=\s*8'h([0-9a-fA-F]+)", control)
        if not match or int(match.group(1), 16) != address:
            raise RuntimeError("Harness register mapping differs: " + name)
    return source, record


def reachable_rtl(folder):
    modules = {}
    for path in folder.glob("*.v"):
        text = path.read_text()
        for name in re.findall(r"\bmodule\s+(\w+)", text):
            modules[name] = path
    # Stem strips .tcl only; an IP module name already ends in _ip.
    ip_scripts = {p.stem: p for p in folder.glob("*_ip.tcl")}
    wanted, seen, files, ips = ["gate_fixed_fp32_top"], set(), set(), set()
    while wanted:
        module = wanted.pop()
        if module in seen:
            continue
        seen.add(module)
        if module in ip_scripts:
            ips.add(ip_scripts[module])
            continue
        if module not in modules:
            raise RuntimeError("Missing generated dependency: " + module)
        path = modules[module]
        files.add(path)
        text = path.read_text()
        referenced = re.findall(r"\b(gate_fixed_fp32_top\w*)\s+(?:#\s*\([^;]*?\)\s*)?\w+\s*\(", text, re.S)
        wanted.extend(name for name in referenced if name != module)
    return sorted(files), sorted(ips)


def tcl_literal(value):
    value = str(value)
    if any(ch in value for ch in "{}\n\r"):
        raise ValueError("Unsupported Tcl path.")
    return "{" + value + "}"


def compile_simulation(config, source, synthesis, build):
    """Generate official IP simulation products; no synth_design or implementation."""
    build.mkdir(parents=True, exist_ok=True)
    names = ("fixed_fp32_axi_tb.sv", "linear_axi_tb.sv")
    signature = {"synthesis_recorded_utc": synthesis["recorded_utc"],
                 "service_schedule_revision": synthesis["service_schedule_revision"]}
    saved = build / "compiled.json"
    simdir = build / "ip_project/fixed_fp32_sim.sim/sim_1/behav/xsim"
    if (saved.is_file() and read(saved).get("signature") == signature
            and all((build / n).is_file() and (build / n).read_bytes() == (ROOT / "hardware" / n).read_bytes() for n in names)
            and (simdir / "xsim.dir" / read(saved)["snapshot"] / "xsimk").is_file()):
        return simdir, read(saved)["snapshot"]
    for name in names:
        shutil.copyfile(ROOT / "hardware" / name, build / name)
    files, ips = reachable_rtl(source / "project/solution/syn/verilog")
    rtlcopy = build / "rtl"
    rtlcopy.mkdir(exist_ok=True)
    for path in files + list((source / "project/solution/syn/verilog").glob("*.dat")):
        shutil.copyfile(path, rtlcopy / path.name)
    copied_ips = []
    for path in ips:
        text = path.read_text()
        # Only product-selection changes; all generated IP parameters are exact.
        text = text.replace("generate_target {synthesis simulation}", "generate_target {simulation}")
        destination = rtlcopy / path.name
        destination.write_text(text)
        copied_ips.append(destination)
    lines = [
        "create_project -force fixed_fp32_sim " + tcl_literal(build / "ip_project") + " -part " + config["hardware"]["part"],
        "set_property target_simulator XSim [current_project]",
    ]
    lines += ["source " + tcl_literal(p) for p in copied_ips]
    lines += ["add_files -norecurse [list " + " ".join(tcl_literal(rtlcopy / p.name) for p in files) + "]",
              "add_files -fileset sim_1 -norecurse [list " + " ".join(tcl_literal(build / n) for n in names) + "]",
              "set_property top fixed_fp32_axi_tb [get_filesets sim_1]",
              "set_property top_lib xil_defaultlib [get_filesets sim_1]",
              "set_property xsim.elaborate.debug_level off [get_filesets sim_1]",
              "update_compile_order -fileset sim_1",
              "launch_simulation -scripts_only",
              "close_project", "exit"]
    tcl = build / "prepare_simulation.tcl"
    tcl.write_text("\n".join(lines) + "\n")
    vivado = config["hardware"]["hls"]["vivado_executable"]
    rc = command_run([vivado, "-mode", "batch", "-source", str(tcl)], build, build / "prepare.log")
    if rc:
        raise RuntimeError(f"Simulation-product preparation failed ({rc}): {build / 'prepare.log'}")
    for path in rtlcopy.glob("*.dat"):
        shutil.copyfile(path, simdir / path.name)
    for script in ("compile.sh", "elaborate.sh"):
        rc = command_run(["bash", str(simdir / script)], simdir, build / (script + ".log"))
        if rc:
            raise RuntimeError(f"{script} failed ({rc}): {build / (script + '.log')}")
    elaborate = (simdir / "elaborate.sh").read_text()
    match = re.search(r"(?:^|\s)(?:--snapshot|-s)\s+([A-Za-z0-9_.-]+)", elaborate)
    if not match:
        raise RuntimeError("Cannot establish generated XSIM snapshot name.")
    snapshot = match.group(1)
    save(saved, {"signature": signature, "snapshot": snapshot, "reachable_rtl_files": [p.name for p in files],
                 "generated_ip_scripts": [p.name for p in ips], "synthesis_performed": False})
    return simdir, snapshot


def parse_capture(log, case, clock_ns):
    def records(prefix):
        return [json.loads(line[len(prefix):]) for line in log.splitlines() if line.startswith(prefix)]
    finals = records("GATE_FP32_CAPTURED ")
    tx = records("GATE_FP32_TRANSACTION ")
    traffic = records("GATE_FP32_TRAFFIC ")
    issues = []
    if "GATE_FP32_FAIL" in log or "GATE_RTL_FAIL" in log:
        issues.append("testbench_failure")
    if len(finals) != 1 or len(tx) != 2 or len(traffic) != 12:
        issues.append("missing_or_duplicate_capture_markers")
        return tx, traffic, finals[0] if len(finals) == 1 else None, issues
    final = finals[0]
    expected = {k: case[k] for k in ("op", "rows", "width", "outputs")}
    count = case["rows"] * (case["outputs"] if case["op"] == 0 else case["width"])
    expected.update(output_count=count, repetitions=2, clock_period_ns=clock_ns, capture_complete=True)
    if any(final.get(k) != value for k, value in expected.items()):
        issues.append("capture_definition_mismatch")
    if ([t.get("index") for t in tx] != [0, 1] or
            any(t.get("status") != 0 or t.get("status_valid") is not True or
                t.get("all_output_bytes_written") is not True or t["latency_cycles"] <= 0 or
                t["latency_cycles"] != t["done_cycle"] - t["start_cycle"] for t in tx)):
        issues.append("invalid_transaction")
    if (final.get("latency_cycles") != [t["latency_cycles"] for t in tx]
            or final.get("interval_cycles") != tx[1]["start_cycle"] - tx[0]["start_cycle"]
            or final.get("total_execution_cycles") != tx[1]["done_cycle"] - tx[0]["start_cycle"]):
        issues.append("cycle_accounting_mismatch")
    if {(t.get("index"), t.get("port")) for t in traffic} != {(r, p) for r in range(2) for p in PORTS}:
        issues.append("traffic_coverage_mismatch")
    return tx, traffic, final, issues


def primary_record_matches(item, case, synthesis):
    return (item.get("passed") is True and item.get("state") == "returned"
            and item.get("case") == case
            and item.get("synthesis_recorded_utc") == synthesis["recorded_utc"]
            and item.get("service_schedule_revision") == synthesis["service_schedule_revision"]
            and item.get("memory_model") == MEMORY_MODEL
            and item.get("clock_ns") == synthesis["clock_ns"]
            and len(item.get("numerical_checks", [])) == 2
            and all(c.get("passed") for c in item["numerical_checks"])
            and len(item.get("transactions", [])) == 2)


def aggregate_primary(out, record, synthesis):
    """Weighted measured calls and AXI traffic; no vendor command or DDR inference."""
    runs = {c["id"]: Path(c["run_record"]) for c in record["cases"]}
    required = primary_cases()
    if set(runs) != {c["id"] for c in required}:
        return None
    harness = (out / "linear_axi_tb.sv").read_text()
    if not all(re.search(r"logic\s+\[31:0\]\s+" + name + r"\s*;", harness)
               for name in ("wdata", "rdata")):
        raise RuntimeError("Review changed AXI data width before aggregating full-beat occupancy")
    counters = ("read_bursts", "read_beats", "read_bytes",
                "write_bursts", "write_beats", "write_bytes")
    traffic_totals = [{port: {key: 0 for key in counters} for port in PORTS} for _ in range(2)]
    def traffic_view(ports):
        totals = {key: sum(ports[p][key] for p in PORTS) for key in counters}
        def with_occupancy(values):
            result = dict(values)
            result.update(read_full_beat_bytes=4 * values["read_beats"],
                          write_full_beat_bytes=4 * values["write_beats"],
                          transfer_or_strobe_bytes=values["read_bytes"] + values["write_bytes"],
                          full_beat_bytes=4 * (values["read_beats"] + values["write_beats"]))
            return result
        return {"ports": {p: with_occupancy(ports[p]) for p in PORTS},
                "totals": with_occupancy(totals)}
    terms, totals = [], [0, 0]
    for case in required:
        item = read(runs[case["id"]])
        if not primary_record_matches(item, case, synthesis):
            raise RuntimeError("Primary evidence mismatch: " + case["id"])
        cycles = [t["latency_cycles"] for t in item["transactions"]]
        if any(not isinstance(c, int) or c <= 0 for c in cycles):
            raise RuntimeError("Invalid observed cycles: " + case["id"])
        weighted = [case["multiplicity"] * c for c in cycles]
        totals = [a + b for a, b in zip(totals, weighted)]
        raw = item.get("traffic", [])
        keys = [(r.get("index"), r.get("port")) for r in raw]
        if len(raw) != 12 or set(keys) != {(rep, port) for rep in range(2) for port in PORTS}:
            raise RuntimeError("Incomplete six-port/two-repetition AXI traffic: " + case["id"])
        observed_traffic, weighted_traffic = [], []
        for rep in range(2):
            ports = {r["port"]: {key: r[key] for key in counters} for r in raw if r["index"] == rep}
            for port in PORTS:
                values = ports[port]
                if any(type(values[k]) is not int or values[k] < 0 for k in counters):
                    raise RuntimeError("Invalid AXI counter: " + case["id"])
                if values["read_bytes"] > 4 * values["read_beats"] or values["write_bytes"] > 4 * values["write_beats"]:
                    raise RuntimeError("Transfer/strobe bytes exceed32-bit beat occupancy")
            weighted_ports = {p: {k: case["multiplicity"] * ports[p][k] for k in counters} for p in PORTS}
            observed_traffic.append(traffic_view(ports))
            weighted_traffic.append(traffic_view(weighted_ports))
            for p in PORTS:
                for key in counters:
                    traffic_totals[rep][p][key] += weighted_ports[p][key]
        terms.append({"id": case["id"], "op": case["op"],
                      "shape": {k: case[k] for k in ("rows", "width", "outputs", "bias")},
                      "calls_per_window": case["multiplicity"],
                      "observed_call_cycles": cycles, "weighted_cycles_by_rep": weighted,
                      "observed_interval_cycles": item["captured"]["interval_cycles"],
                      "observed_axi_by_rep": observed_traffic, "weighted_axi_by_rep": weighted_traffic,
                      "run_record": str(runs[case["id"]])})
    result = {
        "scope": "L256,batch1 serialized sum of representative full-call RTL observations",
        "service_schedule_revision": synthesis["service_schedule_revision"],
        "synthesis_recorded_utc": synthesis["recorded_utc"],
        "clock_ns": synthesis["clock_ns"], "memory_model": MEMORY_MODEL,
        "cycle_boundary": "DUT accepted start to done; excludes configuration and inter-call dispatch gaps",
        "inputs": "Declared signed formula-check fixtures at real invocation shapes; not checkpoint activation/weight traces",
        "call_count": sum(c["multiplicity"] for c in required), "terms": terms,
        "fp32_service_sum_cycles_by_rep": totals,
        "fp32_service_sum_cycles_mean": sum(totals) / 2,
        "fp32_service_sum_ms_at_simulated_clock_by_rep":
            [c * synthesis["clock_ns"] / 1e6 for c in totals],
        "fp32_service_axi_by_rep": [traffic_view(p) for p in traffic_totals],
        "axi_data_bus_width_bits": 32,
        "axi_traffic_definition": {
            "read_bytes": "Captured transfer bytes from ARSIZE/address, including repeated reads; not deduplicated source bytes",
            "write_bytes": "Captured asserted WSTRB bytes, excluding inactive byte lanes",
            "full_beat_bytes": "4 times accepted read+write beats on the same32-bit buses; occupancy, not extra payload",
            "boundary": "Sum of actual six-port simulated transactions; not physical DDR traffic, bandwidth or board observation"},
        "analytical_layout_cycles": None, "fpga_sequencer_and_intercall_cycles": None,
        "physical_ddr_adjustment_cycles": None, "integer_engine_cycles": None,
        "host_and_transport_ms": None, "complete_detector_latency_ms": None,
        "board_latency_ms": None, "feasible": None,
        "boundary": "Complete service-call cost in the declared simulation memory model, not integrated scheduling, physical DDR, detector accuracy, routed frequency or board measurement",
    }
    if result["call_count"] != 411:
        raise RuntimeError("Primary schedule multiplicity changed")
    save(out / "primary_costs.json", result)
    return str(out / "primary_costs.json")



def run_case(case, folder, synthesis, simdir, snapshot, tcl, xsim):
    """Unchanged two-call fixture, capture and numerical checks for one case."""
    folder.mkdir(parents=True, exist_ok=True)
    references, extents = prepare_fixture(folder, case)
    values = {"OP": case["op"], "ROWS": case["rows"], "WIDTH": case["width"],
              "OUTPUTS": case["outputs"], "BIAS": int(case["bias"]),
              "SCALE": struct.pack(">f", case["scale"]).hex(), "COUNT": len(references[0]),
              "CASE_DIR": str(folder), "CLOCK_NS": synthesis["clock_ns"], "TIMEOUT_CYCLES": 10000000}
    values.update({p.upper() + "_BYTES": n for p, n in extents.items()})
    command = [xsim, snapshot, "-tclbatch", str(tcl)]
    for key, value in values.items():
        command += ["-testplusarg", str(key) + "=" + str(value)]
    item = {"case": case, "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "state": "running", "passed": False, "command": command, "extents": extents,
            "synthesis_recorded_utc": synthesis["recorded_utc"],
            "service_schedule_revision": synthesis["service_schedule_revision"],
            "memory_model": MEMORY_MODEL, "clock_ns": synthesis["clock_ns"]}
    save(folder / "run.json", item)
    rc = command_run(command, simdir, folder / "simulation.log")
    tx, traffic, final, issues = parse_capture((folder / "simulation.log").read_text(errors="replace"), case, synthesis["clock_ns"])
    checks = check_output(folder, case, references)
    passed = rc == 0 and not issues and len(checks) == 2 and all(c["passed"] for c in checks)
    item.update(state="returned", returncode=rc, passed=passed, transactions=tx, traffic=traffic,
                captured=final, numerical_checks=checks, issues=issues)
    save(folder / "run.json", item)
    return item


def isolated_primary_worker(number, build, out, config, source, synthesis):
    """Two fixed disjoint queues; no compilation or shared suite-record writes."""
    root = build / "isolated_primary"
    plan = read(root / "dispatch.json")
    parent = Path("/proc") / str(plan["serial_parent_pid"])
    expected = [b"/usr/bin/python3", str(Path(__file__).resolve()).encode(),
                b"--characterize-primary", b"--execute", b""]
    if ((parent / "cmdline").read_bytes().split(bytes([0])) != expected
            or not re.search(r"^State:\s+T", (parent / "status").read_text(), re.M)):
        raise RuntimeError("Verified serial parent must remain SIGSTOPped")
    signature = {"synthesis_recorded_utc": synthesis["recorded_utc"],
                 "service_schedule_revision": synthesis["service_schedule_revision"]}
    compiled = read(build / "compiled.json")
    if plan["signature"] != signature or compiled["signature"] != signature:
        raise RuntimeError("Private workers require the existing exact compiled snapshot")
    for name in ("fixed_fp32_axi_tb.sv", "linear_axi_tb.sv"):
        if (build / name).read_bytes() != (ROOT / "hardware" / name).read_bytes():
            raise RuntimeError("Compiled harness changed")
    queues = plan["queues"]
    flattened = [c for queue in queues.values() for c in queue]
    allowed = {c["id"]: c for c in primary_cases()}
    if len(flattened) != len(set(flattened)) or not set(flattened) <= set(allowed):
        raise RuntimeError("Worker queues overlap or contain undeclared cases")
    destination = root / ("worker_" + str(number))
    destination.mkdir()  # Preserve any previous invocation rather than overwrite.
    state = {"pid": os.getpid(), "serial_parent_pid": plan["serial_parent_pid"],
             "state": "copying", "queue": queues[str(number)], "cases": [],
             "signature": signature, "compilation_performed": False,
             "shared_suite_record_modified": False}
    save(destination / "run.json", state)
    simdir = destination / "sim"
    simdir.mkdir()
    original = build / "ip_project/fixed_fp32_sim.sim/sim_1/behav/xsim"
    snapshot = compiled["snapshot"]
    if not (original / "xsim.dir" / snapshot / "xsimk").is_file():
        raise RuntimeError("Existing simulation executable absent")
    mutable = ("xsim_script.tcl", "xsimSettings.ini", "TempBreakPointFile.txt",
               "xsimkernel.log", "xsimcrash.log", "*.wdb", "*.jou", "*.log")
    shutil.copytree(original / "xsim.dir", simdir / "xsim.dir",
                    ignore=shutil.ignore_patterns(*mutable))
    for path in list(original.glob("*.dat")) + [original / "xsim.ini"]:
        shutil.copyfile(path, simdir / path.name)
    tcl = simdir / "run.tcl"
    tcl.write_text("run all\nquit\n")
    xsim = str(Path(config["hardware"]["hls"]["vivado_executable"]).parent / "xsim")
    state.update(state="running", simulation_cwd=str(simdir), snapshot=snapshot)
    save(destination / "run.json", state)
    print(json.dumps({"isolated_worker": number, "pid": os.getpid(),
                      "serial_parent_pid": plan["serial_parent_pid"],
                      "simulation_cwd": str(simdir), "queue": state["queue"]}), flush=True)
    try:
        for identifier in state["queue"]:
            paused()
            if read(source / "run.json") != synthesis:
                raise RuntimeError("Synthesis record changed")
            case = allowed[identifier]
            folder = out / identifier
            # Never touch the active serial case, a failed attempt or completed data.
            if (folder / "run.json").exists():
                old = read(folder / "run.json")
                if not primary_record_matches(old, case, synthesis):
                    raise RuntimeError("Unexpected occupied worker case: " + identifier)
                item = old
            else:
                item = run_case(case, folder, synthesis, simdir, snapshot, tcl, xsim)
            state["cases"].append({"id": identifier, "passed": item["passed"],
                                   "run_record": str(folder / "run.json")})
            save(destination / "run.json", state)
            print(json.dumps({"worker": number, "case": identifier,
                              "passed": item["passed"],
                              "latencies": item["captured"].get("latency_cycles") if item.get("captured") else None}), flush=True)
            if not item["passed"]:
                raise RuntimeError("Isolated numerical/capture check failed: " + identifier)
        state.update(state="returned", passed=True)
    except Exception as exc:
        state.update(state="stopped", passed=False, error=str(exc))
        save(destination / "run.json", state)
        raise
    save(destination / "run.json", state)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--analyze-only", action="store_true", help="Reaggregate completed primary RTL records; never launch vendor tools")
    parser.add_argument("--characterize-primary", "--primary", dest="primary", action="store_true",
                        help="Use15 L256 cost shapes only; preserve/reuse the separate11 small checks")
    parser.add_argument("--case", choices=["all"] + [c["id"] for c in cases() + primary_cases()], default="all")
    parser.add_argument("--isolated-worker", type=int, choices=(1, 2),
                        help="Run one preassigned private primary queue; no suite-record writes")
    args = parser.parse_args()
    if args.isolated_worker and (not args.execute or not args.primary or args.case != "all"):
        parser.error("--isolated-worker requires --execute --primary and its preassigned queue")
    if args.analyze_only and (not args.primary or args.case != "all"):
        parser.error("--analyze-only requires --primary and all completed primary cases")
    suite = primary_cases() if args.primary else cases()
    selected = [c for c in suite if args.case == "all" or c["id"] == args.case]
    if not selected:
        parser.error("--case must belong to the selected small or primary suite")
    if not args.execute and not args.analyze_only:
        print(json.dumps({"vendor_tool_started": False,
                          "suite": "primary_l256" if args.primary else "small_checks",
                          "cases": selected, "calls_per_case": 2,
                          "primary_schedule_calls": sum(c["multiplicity"] for c in primary_cases()) if args.primary else None}, indent=2))
        return 0
    paused()
    if os.name == "nt":
        raise RuntimeError("Execute using WSL Python and configured Linux Vivado.")
    config = read(ROOT / "configs/project.json")
    source, synthesis = source_evidence(config)
    if args.primary and synthesis["service_schedule_revision"] != "panel_cache_batched_attention_v1":
        raise RuntimeError("Primary multiplicities require the declared panel-cache schedule")
    tag = re.sub(r"[^A-Za-z0-9_-]", "_", synthesis["recorded_utc"])
    build = ROOT / "build/fixed_fp32_rtl" / tag
    small_out = ROOT / "results/goal4/fixed_fp32_rtl" / tag
    if args.primary:
        small = read(small_out / "run.json")
        if not (small.get("passed") and small.get("all_nine_modes_checked")
                and small.get("synthesis_recorded_utc") == synthesis["recorded_utc"]
                and small.get("service_schedule_revision") == synthesis["service_schedule_revision"]
                and {c["id"] for c in small.get("cases", []) if c.get("passed")} == {c["id"] for c in cases()}):
            raise RuntimeError("Reuse requires all11 small checks passed for this exact synthesis")
    out = small_out / "primary_l256" if args.primary else small_out
    if args.isolated_worker:
        return isolated_primary_worker(args.isolated_worker, build, out, config, source, synthesis)
    if args.analyze_only:
        record = read(out / "run.json")
        if not (record.get("state") == "returned" and record.get("passed")
                and record.get("suite") == "primary_l256"
                and record.get("synthesis_recorded_utc") == synthesis["recorded_utc"]
                and record.get("service_schedule_revision") == synthesis["service_schedule_revision"]):
            raise RuntimeError("Completed same-synthesis primary suite required for offline aggregation")
        result = aggregate_primary(out, record, synthesis)
        if not result:
            raise RuntimeError("All15 primary cases required")
        print(json.dumps({"vendor_tool_started": False, "analysis_only": True,
                          "primary_cost_summary": result}, indent=2))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    record = {"recorded_utc": datetime.now(timezone.utc).isoformat(), "state": "preparing", "passed": False,
              "suite": "primary_l256" if args.primary else "small_checks",
              "synthesis_recorded_utc": synthesis["recorded_utc"],
              "service_schedule_revision": synthesis["service_schedule_revision"],
              "source_run": str(source / "run.json"), "source_matches_synthesized_copy": True,
              "memory_model": MEMORY_MODEL, "clock_ns": synthesis["clock_ns"], "cases": [],
              "reused_small_suite": str(small_out / "run.json") if args.primary else None,
              "boundary": "Unchanged service RTL, official arithmetic IP simulation models, independent output checks; not full-model, vendor UVM, DDR or board validation."}
    save(out / "run.json", record)
    for name in ("fixed_fp32_axi_tb.sv", "linear_axi_tb.sv", "fixed_fp32_service.hpp", "fixed_fp32_service_tb.cpp"):
        shutil.copyfile(ROOT / "hardware" / name, out / name)
    shutil.copyfile(Path(__file__), out / "run_fixed_fp32_rtl.py")
    try:
        simdir, snapshot = compile_simulation(config, source, synthesis, build)
        tcl = build / "run.tcl"
        tcl.write_text("run all\nquit\n")
        xsim = str(Path(config["hardware"]["hls"]["vivado_executable"]).parent / "xsim")
        for case in selected:
            paused()
            if read(source / "run.json") != synthesis:
                raise RuntimeError("Synthesis record changed during validation.")
            folder = out / case["id"]
            # Reuse matching checked observations, including a later successful attempt.
            candidates = [folder / "run.json"] + sorted(out.glob(case["id"] + "_attempt*/run.json"))
            reusable = next((p for p in reversed(candidates) if p.is_file()
                             and primary_record_matches(read(p), case, synthesis)), None)
            if reusable:
                record["cases"].append({"id": case["id"], "passed": True,
                                        "run_record": str(reusable), "reused": True})
                record["state"] = "running"
                save(out / "run.json", record)
                print(json.dumps({"case": case["id"], "passed": True, "reused": True}), flush=True)
                continue
            attempt = 1
            while (folder / "run.json").exists():
                attempt += 1
                folder = out / (case["id"] + f"_attempt{attempt}")
            item = run_case(case, folder, synthesis, simdir, snapshot, tcl, xsim)
            passed, final = item["passed"], item["captured"]
            record["cases"].append({"id": case["id"], "passed": passed, "run_record": str(folder / "run.json")})
            record["state"] = "running"
            save(out / "run.json", record)
            print(json.dumps({"case": case["id"], "passed": passed, "latencies": final.get("latency_cycles") if final else None}), flush=True)
            if not passed:
                raise RuntimeError("Bounded FP32 validation failed; stop before remaining cases: " + case["id"])
        record.update(state="returned", passed=True, all_nine_modes_checked={c["op"] for c in selected} == set(range(9)))
        if args.primary:
            record["primary_cost_summary"] = aggregate_primary(out, record, synthesis)
    except Exception as exc:
        record.update(state="stopped", passed=False, error=str(exc))
        save(out / "run.json", record)
        raise
    save(out / "run.json", record)
    print(json.dumps({"passed": record["passed"], "evidence": str(out / "run.json"),
                      "cases": len(record["cases"]), "calls": 2 * len(record["cases"]),
                      "primary_cost_summary": record.get("primary_cost_summary")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

