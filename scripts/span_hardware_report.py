"""Read actual span-pilot HLS/RTL artifacts without launching vendor tools.

Top estimates, completed submodules and actual cosimulation are separate evidence.
Missing full-region or host/transport terms stay null; hierarchy costs are never
summed as though independent pipelined stages were sequential.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/"results/span_coupled_pilot_v1"
ENVELOPE={"LUT":100000,"FF":200000,"DSP":256,"BRAM_36K_equivalent":64,"URAM":96}
MODULES=("load_model","encode_z","encode_stream","project","projection_stream","update","head_score","state_stream","pipeline","window_streamed","window_reference","window_sopa")

def number(text):
    if text is None:return None
    text=text.strip()
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?",text):return None
    return float(text) if "." in text else int(text)

def identity(path,raw):return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def synthesis(path):
    if not path.exists():return dict(present=False,path=str(path),evidence_class="unresolved")
    raw=path.read_bytes();record=dict(present=True,identity=identity(path,raw),evidence_class="HLS synthesis estimate; not routed timing or measured latency")
    try:root=ET.fromstring(raw)
    except ET.ParseError as exc:return dict(record,valid=False,error=str(exc))
    get=lambda key:root.findtext(key)
    record.update(valid=True,version=get("ReportVersion/Version"),module=get("UserAssignments/TopModelName"),part=get("UserAssignments/Part"),
      target_period_ns=number(get("UserAssignments/TargetClockPeriod")),uncertainty_ns=number(get("UserAssignments/ClockUncertainty")),
      estimated_period_ns=number(get("PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod")))
    latency=root.find("PerformanceEstimates/SummaryOfOverallLatency")
    record["latency_cycles"]={label:number(latency.findtext(tag)) if latency is not None else None for label,tag in
      (("minimum","Best-caseLatency"),("average","Average-caseLatency"),("maximum","Worst-caseLatency"),("interval_minimum","Interval-min"),("interval_maximum","Interval-max"))}
    record["latency_raw"]={child.tag:child.text for child in latency} if latency is not None else {}
    resource=root.find("AreaEstimates/Resources")
    resources={child.tag:number(child.text) for child in resource} if resource is not None else {}
    if resources.get("BRAM_18K") is not None:resources["BRAM_36K_equivalent"]=resources["BRAM_18K"]/2
    record["resources"]=resources
    loops=root.find("PerformanceEstimates/SummaryOfLoopLatency")
    record["loop_estimates"]=[dict(name=loop.tag,tripcount=loop.findtext("TripCount"),pipeline_ii=number(loop.findtext("PipelineII")),latency_raw=ET.tostring(loop.find("Latency"),encoding="unicode") if loop.find("Latency") is not None else None) for loop in loops] if loops is not None else []
    return record

def cosimulation(path):
    if not path.exists():return dict(present=False,path=str(path),evidence_class="unresolved",warm_cycles=None,cold_cycles=None)
    raw=path.read_bytes();record=dict(present=True,identity=identity(path,raw),evidence_class="RTL cosimulation report; bounded simulated memory, not physical device")
    for line in raw.decode("utf-8",errors="replace").splitlines():
        cells=[x.strip() for x in line.strip().strip("|").split("|")]
        if cells and cells[0].lower()=="verilog" and len(cells)>=8:
            record.update(status=cells[1],aggregate_cycles={k:number(v) for k,v in zip(("latency_minimum","latency_average","latency_maximum","interval_minimum","interval_average","interval_maximum","total_execution"),cells[2:])})
    record["warm_cycles"]=None;record["cold_cycles"]=None
    record["limitation"]="Aggregate latency includes the separate cold preload and warm cases; transaction-resolved evidence is required before assigning a warm case cost. NA is unresolved."
    return record

def json_record(path):return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

def transaction_evidence(sim,rows,rtl):
    """Read the actual Vitis 2025.2 text table (its suffix is .xml)."""
    path=sim/"verilog/span_guard_top.performance.result.transaction.xml"
    record=dict(present=path.exists(),path=str(path),valid=False,
      evidence_class="Transaction-resolved RTL clock-cycle counts; simulated memory, not physical device",
      definition="Generated testbench latency = finish_timestamp - start_timestamp; interval = next start - current start",
      transactions=[])
    if not path.exists():return record
    raw=path.read_bytes();record["identity"]=identity(path,raw)
    try:
        lines=raw.decode("utf-8").splitlines()
        values=[]
        for line in lines:
            match=re.fullmatch(r"\s*transaction\s+(\d+):\s*(\d+)\s+(\d+|x)\s*",line)
            if match:values.append(dict(transaction=int(match[1]),latency_cycles=int(match[2]),interval_cycles=None if match[3]=="x" else int(match[3])))
            elif line.strip() and line.split()!=["latency","interval"]:raise ValueError("Unrecognized transaction table line")
        record["transactions"]=values
        if [v["transaction"] for v in values]!=list(range(len(rows)+1)):raise ValueError("Incomplete or unexpected transaction indices")
        tb=sim/"verilog/span_guard_top.autotb.v";tbraw=tb.read_bytes();text=tbraw.decode("utf-8")
        record["generated_testbench_identity"]=identity(tb,tbraw)
        count=re.search(r"parameter AUTOTB_TRANSACTION_NUM = (\d+);",text)
        if not count or int(count[1])!=len(values) or "latency[i] = finish_timestamp[i] - start_timestamp[i];" not in text:raise ValueError("Unverified generated cycle definition/count")
        inputs={}
        for field in ("load","count","empty","previous","threshold"):
            inp=sim/("tv/cdatafile/c.span_guard_top.autotvin_"+field+".dat");inpraw=inp.read_bytes()
            scalar=re.findall(r"\[\[transaction\]\]\s+(\d+)\s+(0x[0-9a-fA-F]+)\s+\[\[/transaction\]\]",inpraw.decode("utf-8"))
            if [int(i) for i,_ in scalar]!=list(range(len(values))):raise ValueError("Missing scalar input transactions: "+field)
            inputs[field]=[int(v,16) for _,v in scalar]
            record.setdefault("scalar_input_identities",{})[field]=identity(inp,inpraw)
        if inputs["load"]!=[1]+[0]*len(rows) or inputs["count"]!=[0]+[len(row["windows"]) for row in rows]:raise ValueError("Cold/warm load or window count mismatch")
        for field,key in (("empty","empty_document"),("previous","previous_risk_bits"),("threshold","threshold_bits")):
            if inputs[field][1:]!=[int(row.get(key,0)) for row in rows]:raise ValueError("Warm scalar case identity mismatch: "+field)
        record["input_order_verified"]=True
        if rtl.get("status","").lower()!="pass":raise ValueError("Completed Verilog Pass required before verified cycle attribution")
        record["valid"]=True
    except (ValueError,OSError,UnicodeError) as exc:record["error"]=str(exc)
    return record

def complete_document(windows,previous):
    if not windows or previous:return False
    lengths={w["document_content_length"] for w in windows}
    if len(lengths)!=1:return False
    end=0
    for lo,hi in sorted((w["content_start"],w["content_start"]+len(w["ids"])) for w in windows):
        if lo>end:return False
        end=max(end,hi)
    return end==next(iter(lengths))

def inspect(run):
    run=Path(run).resolve();prepared=json_record(run/"prepared.json")
    if prepared is None:raise FileNotFoundError("Prepared pilot identity required")
    syn=run/"hls_project/solution/syn/report";sim=run/"hls_project/solution/sim/report"
    top=synthesis(syn/"span_guard_top_csynth.xml");modules={name:synthesis(syn/(name+"_csynth.xml")) for name in MODULES}
    rtl=cosimulation(sim/"span_guard_top_cosim.rpt")
    vendor={}
    for action in ("csim","synthesize","cosim","full-flow"):
        value=json_record(run/("vendor_"+action+".json"))
        if value is not None:
            if "Resource floor failed:" in value.get("error",""):
                try:value["failed_memory_observation"]=json.loads(value["error"].split("Resource floor failed:",1)[1].strip())
                except json.JSONDecodeError:pass
            vendor[action]=value
    resources=top.get("resources",{})
    envelope={k:dict(limit=limit,actual=resources.get(k),within=resources[k]<=limit if resources.get(k) is not None else None) for k,limit in ENVELOPE.items()}
    known=all(v["within"] is not None for v in envelope.values())
    period=None
    if top.get("valid") and top.get("target_period_ns") is not None and top.get("estimated_period_ns") is not None:
        period=max(top["target_period_ns"],top["estimated_period_ns"])/.8
    selection=prepared.get("rtl_cases_source_indices",{});indices=list(dict.fromkeys(selection.values()))
    rows=json_record(run/"cases_rtl.json") or [];transactions=[]
    resolved=transaction_evidence(run/"hls_project/solution/sim",rows,rtl)
    if resolved["valid"]:
        try:
            from span_hardware import pack_cases
            path=run/"cases_rtl.bin";raw=path.read_bytes()
            resolved["prepared_case_identity"]=identity(path,raw)
            if hashlib.sha256(raw).hexdigest()!=prepared["rtl_cases_sha256"]:raise ValueError("Prepared binary case identity changed")
            if len(rows)!=prepared["rtl_case_count"] or pack_cases(rows)!=raw:raise ValueError("Case JSON order/content differs from prepared binary inputs")
            resolved["complete_case_order_verified"]=True
        except (OSError,ValueError,KeyError,struct.error) as exc:
            resolved["valid"]=False;resolved["error"]=str(exc)
    rtl["transaction_evidence"]=resolved
    cycles={v["transaction"]:v for v in resolved["transactions"]} if resolved["valid"] else {}
    if cycles:
        rtl["cold_cycles"]=cycles[0]["latency_cycles"]
        rtl["warm_cycles"]=[cycles[i]["latency_cycles"] for i in range(1,len(rows)+1)]
        rtl["limitation"]="Cold/warm cycles are transaction-resolved and input-order verified. Durations use a separately stated estimated clock; memory is simulated and host/transport costs remain unresolved."
    for at,row in enumerate(rows):
        source_index=indices[at] if at<len(indices) else None
        windows=row["windows"];tokens=sum(len(w["ids"]) for w in windows)
        transactions.append(dict(transaction=at+1,labels=[k for k,v in selection.items() if v==source_index],source_case_index=source_index,
          windows=len(windows),content_tokens=tokens,complete_document_case=complete_document(windows,row.get("previous_risk_bits",0)),
          continuation=bool(row.get("previous_risk_bits",0)),expected_risk_bits=row["expected_risk_bits"],
          minimum_logical_bytes=dict(token_ids=2*tokens,descriptors=16*len(windows),result=20),
          actual_axi_bytes=None,observed_latency_cycles=cycles.get(at+1,{}).get("latency_cycles"),
          observed_start_interval_cycles=cycles.get(at+1,{}).get("interval_cycles"),
          clock_derated_estimate_ms=cycles[at+1]["latency_cycles"]*period/1e6 if at+1 in cycles and period is not None else None))
    anchor=json_record(ROOT/"results/goal_r1/architecture_facts.json")
    software=anchor.get("current_best_available_software_path") if anchor else None
    worksheet=dict(primary_metric="warm batch-one raw document through complete strict decision, all windows",
      evidence_boundary="Dependency-aware full-chain simulation plus separately evidenced host costs; no physical measurement",
      equation="preparation + packing + ingress/control + scheduled complete region + incremental physical memory delay + return/decision",
      terms_ms={k:None for k in ("same_workload_preparation","packing","ingress_and_launch","warm_complete_region","incremental_physical_memory_delay","return_and_decision")},
      full_check_ms=None,same_candidate_software_ms=None,quality_qualified_baseline_software_ms=None,
      historical_software_anchor=software,
      historical_anchor_limitation="R1 is32 measured single-window inputs with W8 BERT arithmetic; never attach its timing or preparation directly to another workload/candidate.",
      clock_derated_period_ns=period,clock_derating=.8,clock_class="HLS estimate with20% frequency derating; routed frequency unresolved",
      scheduled_overlap="Do not add encoder/projection/state module latencies or duplicate nominal AXI waits already inside a complete RTL transaction.",
      transport=dict(verified_fast_host_path=False,bandwidth_bytes_per_second=None,launch_seconds=None,
        dependency="A defensible bound on physical copy/control/DDR/NoC is required; missing terms are not zero.",
        historical_uart=dict(baud=115200,bits_per_byte=10,framing_bytes=128,framing_only_ms=128*10/115200*1000,evidence_class="Exact analytical wire serialization; historical interface model")),
      cold=dict(model_blob_bytes=prepared["model_blob_bytes"],observed_latency_cycles=rtl.get("cold_cycles"),
        clock_derated_estimate_ms=rtl["cold_cycles"]*period/1e6 if rtl.get("cold_cycles") is not None and period is not None else None,
        tokenizer_initialization_ms=None,physical_configuration_ms=None,physical_model_transport_ms=None))
    worksheet["complete_document_rtl_cases"]=[dict(transaction=t["transaction"],labels=t["labels"],latency_cycles=t["observed_latency_cycles"],clock_derated_estimate_ms=t["clock_derated_estimate_ms"]) for t in transactions if t["complete_document_case"]]
    return dict(design=prepared["design"],algorithm=prepared["algorithm"],mapping=prepared["mapping"],run=str(run),
      source_hashes=(json_record(run/"c_build.json") or {}).get("source_hashes"),model_blob_sha256=prepared["model_blob_sha256"],
      vendor=vendor,top_synthesis=top,completed_module_estimates=modules,rtl_cosimulation=rtl,
      common_envelope=dict(checks=envelope,part_verified=top.get("part")=="xcve2802-vsvh1760-2MP-e-S",passed=all(v["within"] for v in envelope.values()) and top.get("part")=="xcve2802-vsvh1760-2MP-e-S" if known else None,
        limitation="Only complete top resource report can establish this gate; submodules cannot be summed or substituted."),
      rtl_transaction_order=[dict(transaction=0,phase="cold preload only",windows=0,observed_latency_cycles=rtl.get("cold_cycles"),clock_derated_estimate_ms=worksheet["cold"]["clock_derated_estimate_ms"])]+transactions,
      full_check_worksheet=worksheet,hardware_value_gate_passed=False if known and not all(v["within"] for v in envelope.values()) else None,protected_partitions_opened=False,
      unresolved=(["transaction-resolved warm/cold RTL cycles"] if not resolved["valid"] else [])+["physical/defensibly bounded host ingress and DDR/NoC","same-workload host preparation","same-candidate competent software timing","routed timing and physical measurements"])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    run,out=a.run.resolve(),a.output.resolve()
    if not run.is_relative_to(STUDY.resolve()) or not out.is_relative_to(STUDY.resolve()) or out.exists():raise ValueError("Named study input and fresh report output required")
    result=inspect(run);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x",encoding="utf-8") as f:json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps(dict(design=result["design"],top_report=result["top_synthesis"]["present"],envelope=result["common_envelope"],cosim=result["rtl_cosimulation"].get("status")),indent=2))
if __name__=="__main__":main()
