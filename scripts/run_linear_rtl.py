"""Run a declared Goal 4 point against unchanged HLS RTL in bounded-memory XSIM."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import shutil
import subprocess

from build_linear_hls import configured_linear_study

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+"\n",encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", required=True)
    parser.add_argument("--cached-pilot", action="store_true", help="Validate one declared cached linear profile separately from the historical tiled study")
    parser.add_argument("--static-bank", action="store_true", help="Validate the isolated lanes64 static_bank_index_v1 repair; implies --cached-pilot")
    parser.add_argument("--diagnostic", action="store_true", help="Passive lanes4 target-stage counters in a separate build")
    args = parser.parse_args()
    if args.static_bank:
        args.cached_pilot = True
    if os.name == "nt":
        raise RuntimeError("Use WSL Python with the configured Linux XSIM tools")
    if (ROOT / "results/goal2/pause.request").exists():
        raise RuntimeError("User pause marker present")
    config = read(ROOT / "configs/project.json")
    h = config["hardware"]
    profiles, points = configured_linear_study(config, args.cached_pilot)
    point = next((p for p in points if p["id"] == args.point),None)
    if point is None:
        raise ValueError("Select one declared characterization point")
    if args.static_bank and point["profile"] != "lanes64":
        raise ValueError("Static-bank validation is scoped to the separate lanes64 revision")
    if args.diagnostic and (args.cached_pilot or point["profile"] != "lanes4" or point["role"] != "target"):
        raise ValueError("Passive diagnostic is only for initial lanes4 targets")
    if point["role"] == "reserved":
        suffix = "_static_bank_lanes64" if args.static_bank else "" if point["profile"] == "lanes16" else "_" + point["profile"]
        summary = read(ROOT / (f"results/goal4/cached_characterization{suffix}.json" if args.cached_pilot else "results/goal4/characterization_summary.json"))
        if not summary.get("nominal_freeze"):
            raise RuntimeError("Freeze predictions before any reserved dispatch")
    source = ROOT / ("build/linear_hls_cached_static_bank" if args.static_bank else "build/linear_hls_cached" if args.cached_pilot else "build/linear_hls") / point["profile"]
    synthesis = read(source / "run.json")
    expected_profile = next(p for p in profiles if p["id"] == point["profile"])
    expected_variant = "weight_cache_static_bank" if args.static_bank else "weight_cache_pilot" if args.cached_pilot else "initial_tiled"
    if (not synthesis.get("hls_synthesis_performed")
            or synthesis.get("returncode") != 0
            or synthesis.get("selected_profile") != expected_profile
            or synthesis.get("kernel_variant", "initial_tiled") != expected_variant):
        raise RuntimeError("Successful unchanged runtime-engine synthesis for this profile/variant required")
    if args.static_bank and synthesis.get("source_revision") != "static_bank_index_v1":
        raise RuntimeError("Expected the recorded static_bank_index_v1 synthesis")
    if args.cached_pilot and point["role"] == "reserved":
        freeze = summary["nominal_freeze"]
        if (freeze.get("kernel_variant") != expected_variant
                or freeze.get("synthesis_recorded_utc") != synthesis["recorded_utc"]
                or point not in freeze.get("point_definitions", [])):
            raise RuntimeError("Cached reserved point requires its own matching synthesis/point freeze")
    tb = ROOT / "hardware/linear_axi_tb.sv"
    build = ROOT / ("build/linear_rtl_diagnostic" if args.diagnostic else "build/linear_rtl_cached_static_bank" if args.static_bank else "build/linear_rtl_cached" if args.cached_pilot else "build/linear_rtl") / point["profile"]
    build.mkdir(parents=True,exist_ok=True)
    out = ROOT / ("results/goal4/rtl_diagnostic" if args.diagnostic else "results/goal4/rtl_cached_static_bank" if args.static_bank else "results/goal4/rtl_cached" if args.cached_pilot else "results/goal4/rtl") / point["id"]
    out.mkdir(parents=True,exist_ok=True)
    if (out / "run.json").exists():
        old = read(out / "run.json")
        if (not args.diagnostic and old.get("passed") and old.get("point") == point
                and old.get("synthesis_recorded_utc") == synthesis["recorded_utc"]
                and (out / "linear_axi_tb.sv").read_text() == tb.read_text()):
            print(json.dumps({"reused":str(out)}))
            return 0
    vivado_bin = Path(h["hls"]["vivado_executable"]).parent
    signature = {"synthesis_recorded_utc":synthesis["recorded_utc"],"profile":point["profile"]}
    compiled = build / "compiled.json"
    source_copy = build / "linear_axi_tb.sv"
    reuse = (not args.diagnostic and compiled.is_file() and read(compiled)==signature
             and source_copy.is_file() and source_copy.read_text()==tb.read_text()
             and (build / "xsim.dir/linear_axi_tb/xsimk").is_file())
    if not reuse:
        shutil.copyfile(tb,source_copy)
        files = sorted((source / "project/solution/syn/verilog").glob("*.v"))
        if not files:
            raise RuntimeError("No actual synthesized Verilog files")
        prj = build / "rtl.prj"
        entries = ['verilog xil_defaultlib "'+str(p)+'"' for p in files]
        entries.append('sv xil_defaultlib "'+str(source_copy)+'"')
        if args.diagnostic:
            shutil.copyfile(ROOT / "hardware/linear_rtl_diagnostic.sv", build / "linear_rtl_diagnostic.sv")
            entries.append('sv xil_defaultlib "'+str(build / "linear_rtl_diagnostic.sv")+'"')
        prj.write_text("\n".join(entries)+"\n",encoding="utf-8")
        command = [str(vivado_bin/"xelab"),"xil_defaultlib.linear_axi_tb","-prj",str(prj),
                   "-L","unisims_ver","-L","xpm","-relax","-s","linear_axi_tb"]
        if args.diagnostic:
            command.insert(2,"xil_defaultlib.linear_diag")
        with (build / "compile.log").open("w") as log:
            result = subprocess.run(command,cwd=build,stdout=log,stderr=subprocess.STDOUT)
        shutil.copyfile(build/"compile.log",out/"compile.log")
        if result.returncode:
            print(json.dumps({"compile_returncode":result.returncode,"log":str(out/"compile.log")}))
            return result.returncode
        save(compiled,signature)
    tcl = build / "run.tcl"
    tcl.write_text("run all\nquit\n")
    command = [str(vivado_bin/"xsim"),"linear_axi_tb","-tclbatch",str(tcl)]
    values = {"ROWS":point["rows"],"INNER":point["inner"],"OUTPUTS":point["outputs"],
              "BITS":point["weight_bits"],"REPS":point["repetitions"],
              "CLOCK_NS":1000/h["clock_objective_mhz"],"TIMEOUT_CYCLES":500000000}
    for key,value in values.items():
        command += ["-testplusarg",str(key)+"="+str(value)]
    record = {"recorded_utc":datetime.now(timezone.utc).isoformat(),"point":point,
              "synthesis_recorded_utc":synthesis["recorded_utc"],"synthesis_reused_from":str(source/"run.json"),
              "command":command,"state":"running","passed":False,
              "verification_method":"direct_xsim_bounded_axi_testbench",
              "kernel_variant":expected_variant,
              "memory_model":"Per-port fixed byte arrays, one outstanding request, registered response, no additional DDR delay; distinct from vendor UVM timing",
              "boundary":"Exact integer output checks and RTL cycles; not vendor C/RTL Pass report, physical memory bandwidth, or board measurement"}
    if args.static_bank:
        record["source_revision"] = synthesis["source_revision"]
    save(out/"run.json",record)
    shutil.copyfile(tb,out/"linear_axi_tb.sv")
    with (out/"simulation.log").open("w") as log:
        result = subprocess.run(command,cwd=build,stdout=log,stderr=subprocess.STDOUT)
    lines=(out/"simulation.log").read_text(errors="replace").splitlines()
    finals=[json.loads(line.split(" ",1)[1]) for line in lines if line.startswith("GATE_RTL_VERIFIED ")]
    transactions=[json.loads(line.split(" ",1)[1]) for line in lines if line.startswith("GATE_RTL_TRANSACTION ")]
    passed=result.returncode==0 and len(finals)==1 and len(transactions)==point["repetitions"]
    if passed:
        final = finals[0]
        expected = {"rows":point["rows"],"inner":point["inner"],"outputs":point["outputs"],
                    "bits":point["weight_bits"],"repetitions":point["repetitions"]}
        passed = (final.get("passed") is True and all(final.get(k)==v for k,v in expected.items())
                  and final.get("clock_period_ns")==1000/h["clock_objective_mhz"]
                  and [t.get("index") for t in transactions]==[0,1]
                  and all(t.get("exact_output_passed") is True and t["latency_cycles"]>0
                          and t["latency_cycles"]==t["done_cycle"]-t["start_cycle"] for t in transactions)
                  and final.get("latency_cycles")==[t["latency_cycles"] for t in transactions]
                  and final.get("interval_cycles")==transactions[1]["start_cycle"]-transactions[0]["start_cycle"]
                  and final.get("total_execution_cycles")==transactions[1]["done_cycle"]-transactions[0]["start_cycle"])
    diagnostics=[json.loads(line.split(" ",1)[1]) for line in lines if line.startswith("GATE_RTL_DIAGNOSTIC ")]
    if args.diagnostic:
        shutil.copyfile(build / "linear_rtl_diagnostic.sv", out / "linear_rtl_diagnostic.sv")
        passed = passed and len(diagnostics)==point["repetitions"] and all(d.get("checks_passed") is True for d in diagnostics)
    record.update(state="returned",returncode=result.returncode,passed=passed,transactions=transactions,diagnostics=diagnostics,
                  verified=finals[0] if len(finals)==1 else None)
    save(out/"run.json",record)
    print(json.dumps({"passed":passed,"returncode":result.returncode,"evidence":str(out),
                      "verified":record["verified"]},indent=2))
    return result.returncode or (0 if passed else 1)

if __name__ == "__main__":
    raise SystemExit(main())
