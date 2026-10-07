"""One representative out-of-context implementation of the current FP32 service.

Uses the completed HLS solution; no C simulation or HLS synthesis is repeated.
This is not the integrated detector/DDR system or a board programming command.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import math
import xml.etree.ElementTree as ET
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, record):
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def parse_implementation_report(path, expected_clock_ns):
    """Parse only the two short header sections, never the huge RTL hierarchy."""
    sections, active, lines = {}, None, []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= 256 or sum(map(len, lines)) > 65536:
                raise ValueError("Expected small implementation report sections not found")
            stripped = line.strip()
            if stripped in ("<TimingReport>", "<AreaReport>"):
                active, lines = stripped[1:-1], [line]
            elif active:
                lines.append(line)
            if active and stripped == "</" + active + ">":
                sections[active] = ET.fromstring("".join(lines))
                active, lines = None, []
                if len(sections) == 2:
                    break
    if set(sections) != {"TimingReport", "AreaReport"}:
        raise ValueError("Implementation TimingReport/AreaReport missing")
    timing, area = sections["TimingReport"], sections["AreaReport"]
    def finite(tag):
        value = float(timing.findtext(tag))
        if not math.isfinite(value):
            raise ValueError("Nonfinite implementation metric: " + tag)
        return value
    target = finite("TargetClockPeriod")
    if not math.isclose(target, expected_clock_ns, abs_tol=1e-9):
        raise ValueError("Implementation target differs from recorded clock")
    flag = timing.findtext("TIMING_MET")
    if flag not in ("TRUE", "FALSE"):
        raise ValueError("Unrecognized explicit implementation timing result")
    resources = {node.tag: int(node.text) for node in area.findall("Resources/*")}
    available = {node.tag: int(node.text) for node in area.findall("AvailableResources/*")}
    # This exact VEK280 export reports1200 BRAM units, i.e.18Kb equivalents.
    if available.get("BRAM") != 1200 or available.get("DSP") != 1312:
        raise ValueError("Unexpected selected-part resource units; review report")
    result = {
        "timing_met": flag == "TRUE", "clock_name": timing.findtext("CLOCK_NAME"),
        "target_clock_ns": target, "achieved_clock_ns": finite("AchievedClockPeriod"),
        "routed_clock_ns": finite("CP_ROUTE"), "post_synthesis_clock_ns": finite("CP_SYNTH"),
        "wns_ns": finite("WNS_FINAL"), "tns_ns": finite("TNS_FINAL"),
        "resources": {"BRAM_18K_equivalent": resources["BRAM"], "DSP": resources["DSP"],
                      "FF": resources["FF"], "LUT_including_SRL": resources["LUT"],
                      "SRL_included_in_LUT": resources["SRL"],
                      "LUT_excluding_SRL": resources["LUT"] - resources["SRL"],
                      "URAM": resources["URAM"]},
        "resource_basis": "AreaReport totals; LUT excluding SRL is total LUT minus SRL, not an additional resource",
        "available_resources_as_reported": available,
        "scope": "Representative HLS IP out-of-context implementation; no integrated detector, DDR or board measurement",
    }
    if result["timing_met"] and (result["wns_ns"] < 0 or result["tns_ns"] < 0
                                or result["achieved_clock_ns"] > target):
        raise ValueError("Implementation timing fields contradict TIMING_MET")
    return result


def record_implementation_metrics(out, record):
    if record.get("status") != "returned" or record.get("returncode") != 0:
        raise ValueError("Completed successful implementation command required")
    report = out / "reports/verilog/export_impl.xml"
    metrics = parse_implementation_report(report, record["clock_ns"])
    record.update(timing_met=metrics["timing_met"], implementation_metrics=metrics,
                  implementation_report=str(report.relative_to(ROOT)).replace("\\", "/"),
                  implementation_report_parsed=True,
                  note="Actual report parsed; timing result is for this out-of-context service only")
    save(out / "run.json", record)
    return record



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--implement", action="store_true")
    mode.add_argument("--parse-report", action="store_true", help="Refresh existing run metrics without a vendor command")
    args = parser.parse_args()
    config = read(ROOT / "configs/project.json")
    build = ROOT / "build/fixed_fp32_hls"
    synthesis = read(build / "run.json")
    if not synthesis.get("synthesis_passed"):
        raise RuntimeError("Successful current FP32 synthesis required")
    if synthesis.get("part") != config["hardware"]["part"]:
        raise RuntimeError("Part differs from completed synthesis")
    source = build / "fixed_fp32_service.hpp"
    if not source.is_file() or source.read_text() != (ROOT / "hardware/fixed_fp32_service.hpp").read_text():
        raise RuntimeError("Current source differs from the completed synthesis snapshot")
    if args.parse_report:
        out = ROOT / "results/goal4/fixed_fp32_implementation"
        record = read(out / "run.json")
        if record.get("synthesis_recorded_utc") != synthesis["recorded_utc"]:
            raise RuntimeError("Saved implementation belongs to a different synthesis")
        print(json.dumps(record_implementation_metrics(out, record), indent=2))
        return 0
    if not args.implement:
        print(json.dumps({"synthesis_recorded_utc": synthesis["recorded_utc"],
                          "service_schedule_revision": synthesis.get("service_schedule_revision"),
                          "vendor_tool_started": False}, indent=2))
        return 0
    if os.name == "nt":
        raise RuntimeError("Use WSL Python with the configured Linux tools")
    if (ROOT / "results/goal2/pause.request").exists():
        raise RuntimeError("User pause marker present")
    out = ROOT / "results/goal4/fixed_fp32_implementation"
    out.mkdir(parents=True, exist_ok=True)
    record_path = out / "run.json"
    if record_path.exists():
        old = read(record_path)
        if old.get("synthesis_recorded_utc") == synthesis["recorded_utc"]:
            raise RuntimeError("Implementation already dispatched for this synthesis; inspect saved evidence")
        raise RuntimeError("Archive prior implementation before a changed-source run")
    tcl = build / "implement.tcl"
    project = (build / "project").as_posix()
    if any(c in project for c in "{}\n\r"):
        raise ValueError("Unsupported Tcl path")
    tcl.write_text("open_project {" + project + "}\nopen_solution solution\n"
                   "config_export -vivado_report_level 2\n"
                   "export_design -flow impl -rtl verilog -format ip_catalog\nexit\n")
    command = [config["hardware"]["hls"]["tool_executable"], "--mode", "hls", "--tcl", str(tcl)]
    record = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
              "synthesis_recorded_utc": synthesis["recorded_utc"],
              "service_schedule_revision": synthesis.get("service_schedule_revision"),
              "part": synthesis["part"], "clock_ns": synthesis["clock_ns"],
              "command": command, "status": "running", "returncode": None,
              "timing_met": None, "integrated_system": False, "board_programmed": False,
              "boundary": "Representative FP32 HLS IP out-of-context synthesis/place/route only"}
    save(record_path, record)
    log = out / "vendor.log"
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=build, stdout=handle, stderr=subprocess.STDOUT)
    record.update(status="returned", returncode=result.returncode,
                  note="Inspect archived implementation/timing reports; zero exit alone is not timing closure")
    save(record_path, record)
    shutil.copyfile(tcl, out / tcl.name)
    reports = build / "project/solution/impl/report"
    if reports.is_dir():
        shutil.copytree(reports, out / "reports", dirs_exist_ok=True)
    if result.returncode == 0:
        record_implementation_metrics(out, record)
    print(json.dumps(record, indent=2))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
