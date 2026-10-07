"""Bounded shared FP32-service C simulation / HLS synthesis for Goal 4."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

def literal(value):
    value = str(value)
    if any(c in value for c in "{}\n\r"):
        raise ValueError("Unsupported Tcl value")
    return "{" + value + "}"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthesize", action="store_true", help="C-sim then one shared-service synthesis")
    parser.add_argument("--csim", action="store_true", help="C simulation only")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/project.json").read_text(encoding="utf-8-sig"))
    h = config["hardware"]
    hls = h["hls"]
    service = hls["fixed_fp32_service"]
    for key, expected in (("lanes",4),("max_elements",32768),("max_reduction",512),("dot_output_limit",64)):
        if service[key] != expected:
            raise ValueError("Source service contract differs from configuration: " + key)
    if not args.csim and not args.synthesize:
        print(json.dumps({"service":service,"vendor_tool_started":False},indent=2))
        return 0
    if (ROOT / "results/goal2/pause.request").exists():
        raise RuntimeError("User pause marker present")
    if os.name == "nt":
        raise RuntimeError("Run using WSL Python and the configured Linux Vitis executable")
    executable = Path(hls["tool_executable"])
    if not executable.is_file() or not h["part"] or not h["clock_objective_mhz"]:
        raise RuntimeError("Actual tool/part/clock required")
    build = ROOT / "build/fixed_fp32_hls"
    build.mkdir(parents=True, exist_ok=True)
    top = build / "gate_fixed_fp32_top.cpp"
    top.write_text('''#include "fixed_fp32_service.hpp"
extern "C" void gate_fixed_fp32_top(int op, const float* x, const float* y,
    const float* z, const ap_int<32>* accumulators, float* output, ap_int<8>* codes,
    int rows, int width, int outputs, float scale, bool dot_bias, int* status) {
    const bool valid = gate::fixed_fp32_service(op,x,y,z,accumulators,output,codes,
        rows,width,outputs,scale,dot_bias);
    *status = valid ? 0 : 1;
}
''', encoding="utf-8")
    for name in ("fixed_fp32_service.hpp", "fixed_fp32_service_tb.cpp"):
        shutil.copyfile(ROOT / "hardware" / name, build / name)
    revision = re.search(r'^#define GATE_FIXED_FP32_SCHEDULE_REVISION "([^"\n]+)"',
                         (build / "fixed_fp32_service.hpp").read_text(), re.MULTILINE)
    if revision is None:
        raise ValueError("FP32 header must declare the schedule revision before a new build")
    include = build.as_posix()
    if any(c.isspace() for c in include):
        raise ValueError("No whitespace permitted in HLS include path")
    flags = "-std=c++14 -ffp-contract=off -I" + include
    lines = [("open_project -reset " if args.synthesize else "open_project ") + literal(build / "project"), "set_top gate_fixed_fp32_top",
        "add_files " + literal(top) + " -cflags " + literal(flags),
        "add_files -tb " + literal(build / "fixed_fp32_service_tb.cpp") + " -cflags " + literal(flags),
        "open_solution solution -flow_target " + hls["flow_target"],
        "set_part " + literal(h["part"]),
        "create_clock -period " + str(1000 / h["clock_objective_mhz"]),
        "set_clock_uncertainty " + str(hls["clock_uncertainty_ns"]),
        "config_compile -unsafe_math_optimizations=false",
        "config_op fdiv -impl fabric -latency 12"]
    for port in ("x","y","z","accumulators","output","codes"):
        lines.append("set_directive_interface -mode m_axi -offset slave -bundle gmem_" + port + " -depth 32768 gate_fixed_fp32_top " + port)
    for port in ("op","x","y","z","accumulators","output","codes","rows","width","outputs","scale","dot_bias","status","return"):
        lines.append("set_directive_interface -mode s_axilite -bundle control gate_fixed_fp32_top " + port)
    lines += ["csim_design"] + (["csynth_design"] if args.synthesize else []) + ["exit"]
    tcl = build / "build.tcl"
    tcl.write_text("\n".join(lines)+"\n",encoding="utf-8")
    command = [str(executable),"--mode","hls","--tcl",str(tcl)]
    record = {"recorded_utc":datetime.now(timezone.utc).isoformat(),"service":service,
        "service_schedule_revision":revision.group(1),
        "fdiv_fabric_latency_request":12, "part":h["part"],"clock_ns":1000/h["clock_objective_mhz"],
        "clock_uncertainty_ns":hls["clock_uncertainty_ns"],"tool_versions":h["tool_versions"],
        "command":command,"action":"synthesize" if args.synthesize else "csim",
        "status":"running","c_simulation_passed":False,"synthesis_passed":False,
        "boundary":"Shared FP32 service characterization, not full-detector numerical/RTL or board validation"}
    save(build / "run.json",record)
    log = build / "vendor.log"
    with log.open("w",encoding="utf-8") as handle:
        result = subprocess.run(command,cwd=build,stdout=handle,stderr=subprocess.STDOUT)
    text = log.read_text(errors="replace")
    passed = result.returncode == 0 and "CSim done with 0 errors." in text and "PASS fixed_fp32 service" in text
    report = build / "project/solution/syn/report/gate_fixed_fp32_top_csynth.xml"
    synthesized = passed and args.synthesize and report.is_file() and "Finished Command csynth_design" in text
    record.update(status="returned",returncode=result.returncode,c_simulation_passed=passed,synthesis_passed=synthesized)
    if synthesized:
        tree = ET.parse(report).getroot()
        record["estimated_resources"] = {e.tag:int(e.text) for e in tree.find("AreaEstimates/Resources")}
        record["estimated_clock_ns"] = float(tree.findtext("PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod"))
    save(build / "run.json",record)
    out = ROOT / "results/goal4/fixed_fp32"
    out.mkdir(parents=True,exist_ok=True)
    for name in ("run.json","vendor.log","fixed_fp32_service.hpp","fixed_fp32_service_tb.cpp","build.tcl"):
        shutil.copyfile(build/name,out/name)
    if synthesized:
        for source in report.parent.iterdir():
            if source.suffix in (".xml",".rpt"):
                shutil.copyfile(source,out/source.name)
    print(json.dumps(record,indent=2))
    return result.returncode or (0 if passed and (not args.synthesize or synthesized) else 1)

if __name__ == "__main__":
    raise SystemExit(main())
