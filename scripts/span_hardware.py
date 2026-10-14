"""Prepare/replay approved span-guard HLS inputs; vendor dispatch stays explicit.

No model inference, training, dataset discovery or board access is performed by
prepare. Input expected decisions must come from the independent integer path.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import signal
import struct
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DESIGNS={"coupled_stream":(0,0),"coupled_buffered":(0,1),"span_mlp":(2,1),"sopa":(3,1)}
FIELDS=[("emb_w","i1",(30522,64)),("conv_w","i1",(61440,)),("conv_bias","i8",(192,)),
 ("conv_shift","i4",(192,)),("proj_w","i1",(64,192)),("proj_bias","i8",(64,)),
 ("proj_shift","i4",(64,)),("lambda_q8","u2",(2,32)),("head_a","i4",(32,)),
 ("head_b","i4",(32,)),("head_c","i4",(32,)),("head_anchor","i4",(192,)),
 ("head_bias","i4",()),("z_exp","i4",()),("sigmoid_bits","u4",(4097,)),
 ("mlp1_w","i4",(32,64)),("mlp1_b","i4",(32,)),("mlp2_w","i4",(32,)),
 ("epsilon_q12","i4",(2,16)),("pattern_head","i4",(16,)),
 ("tanh_q12","i4",(4097,)),("logsigmoid_q12","i4",(4097,))]

def write_new(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as f:f.write(data)

def pack_cases(case_rows):
    packed=[struct.pack("<3I",0x53505431,1,len(case_rows))]
    for row in case_rows:
        windows=row["windows"]
        if len(windows)>64:raise ValueError("Use continued <=64-window chunks; never truncate")
        threshold=int(row["threshold_bits"]);previous=int(row.get("previous_risk_bits",0));empty=int(row.get("empty_document",False))
        if not 0<=threshold<=0x3f800000 or not 0<=previous<=0x3f800000:raise ValueError("Risk bits must be positive finite probability")
        if empty and (windows or previous):raise ValueError("Empty document cannot contain windows or prior risk")
        packed.append(struct.pack("<6I",len(windows),empty,previous,threshold,int(row["expected_risk_bits"]),int(row["expected_anchors"])))
        for w in windows:
            ids=w["ids"];start=int(w["content_start"]);length=int(w["document_content_length"])
            if len(ids)>254 or any(not 0<=int(i)<30522 for i in ids):raise ValueError("Invalid content IDs")
            if not 0<=start<=length or start+len(ids)>length:raise ValueError("Invalid content interval")
            packed.append(struct.pack("<4H2I",len(ids),start==0,start+len(ids)==length,0,start,length))
            packed.append(struct.pack("<254H",*ids,*([0]*(254-len(ids)))))
    return b"".join(packed)

def compact_cases(rows):
    # Select by declared size/boundary, never model scores: one short, one full,
    # one multiwindow and one continued document. Missing classes remain explicit.
    single=[i for i,r in enumerate(rows) if len(r["windows"])==1 and r["windows"][0]["ids"]]
    complete=[i for i in single if rows[i]["windows"][0]["content_start"]==0
              and len(rows[i]["windows"][0]["ids"])==rows[i]["windows"][0]["document_content_length"]]
    real_complete=[i for i in complete if rows[i].get("fixture")!="fixture_corners.pt" and not rows[i].get("corner")]
    multi=[i for i,r in enumerate(rows) if len(r["windows"])>=2]
    continued=[i for i,r in enumerate(rows) if r.get("previous_risk_bits",0)>0 and r["windows"]]
    chosen={}
    if single:
        chosen["full"]=max(single,key=lambda i:(len(rows[i]["windows"][0]["ids"]),-i))
    if complete:
        chosen["short_complete_document"]=min(real_complete or complete,key=lambda i:(len(rows[i]["windows"][0]["ids"]),i))
    if multi:chosen["multiwindow"]=min(multi,key=lambda i:(sum(len(w["ids"]) for w in rows[i]["windows"]),i))
    if continued:chosen["continued"]=min(continued,key=lambda i:(sum(len(w["ids"]) for w in rows[i]["windows"]),i))
    indices=list(dict.fromkeys(chosen.values()))
    if not indices:raise ValueError("No real nonempty RTL cycle case")
    return [rows[i] for i in indices],chosen

def prepare(export,cases,output,design):
    import numpy as np
    export,output=Path(export),Path(output)
    if output.exists():raise FileExistsError(output)
    source=json.loads((export/"manifest.json").read_text(encoding="utf-8"))
    canonical=(export/"model.bin").read_bytes()
    if hashlib.sha256(canonical).hexdigest()!=source["binary_sha256"]:raise ValueError("Export model.bin hash mismatch")
    kind,mapping=DESIGNS[design];arm=source["model"]["arm"]
    if arm!={0:"coupled",2:"span_mlp",3:"sopa"}[kind]:raise ValueError("Wrong export arm for design")
    output.mkdir(parents=True);offset=32;parts=[];meta={}
    optional={"lambda_q8","head_a","head_b","head_c","mlp1_w","mlp1_b","mlp2_w","epsilon_q12","pattern_head"}
    with np.load(export/"model.npz",allow_pickle=False) as arrays:
        for name,dtype,shape in FIELDS:
            value=arrays[name] if name in arrays else None
            if value is not None:
                original=source["arrays"][name]
                raw=np.ascontiguousarray(value,dtype=value.dtype.newbyteorder("<")).tobytes()
                if raw!=canonical[original["offset"]:original["offset"]+original["bytes"]]:raise ValueError("NPZ/bin identity mismatch: "+name)
            if value is None and name not in optional:raise ValueError("Missing "+name)
            if value is None:value=np.zeros(shape,dtype=dtype)
            if name=="mlp2_w":value=value.reshape(-1)
            if tuple(value.shape)!=shape:raise ValueError((name,value.shape,shape))
            value=np.ascontiguousarray(value,dtype="<"+dtype)
            raw=value.tobytes();parts.append(raw);meta[name]={"offset":offset,"bytes":len(raw),"shape":shape,"dtype":"<"+dtype};offset+=len(raw)
    payload=b"".join(parts);blob=struct.pack("<8I",0x53504731,1,kind,58,32,192,30522,len(payload))+payload
    write_new(output/"model.hls.bin",blob)
    case_rows=json.loads(Path(cases).read_text(encoding="utf-8"));packed=pack_cases(case_rows)
    compact,selection=compact_cases(case_rows);rtl=pack_cases(compact)
    write_new(output/"cases.bin",packed);write_new(output/"cases_rtl.bin",rtl)
    write_new(output/"cases_rtl.json",(json.dumps(compact,indent=2)+"\n").encode())
    record={"design":design,"algorithm":kind,"mapping":mapping,"export":str(export.resolve()),"source_manifest":source,
      "model_blob_sha256":hashlib.sha256(blob).hexdigest(),"model_blob_bytes":len(blob),"fields":meta,
      "cases_source":str(Path(cases).resolve()),"cases_sha256":hashlib.sha256(packed).hexdigest(),
      "rtl_cases_source_indices":selection,"rtl_cases_sha256":hashlib.sha256(rtl).hexdigest(),
      "rtl_case_count":len(compact),"latency_annotations":"LOOP_TRIPCOUNT bounds are analytical estimates, not observed cycles",
      "boundaries":"content-only IDs; true document edges; <=64-window chunks with exact previous maximum",
      "evidence":"prepared only; no compiler/simulation execution"}
    write_new(output/"prepared.json",(json.dumps(record,indent=2)+"\n").encode())
    return record

def build_c(output,design):
    kind,mapping=DESIGNS[design];output=Path(output)
    if (output/"span_reference").exists() or (output/"c_build.json").exists():raise FileExistsError("Use a fresh correction output directory")
    sources=snapshot_sources(output)
    cmd=["g++","-std=c++17","-O2","-Wno-unknown-pragmas",f"-DSPAN_ALGORITHM={kind}",f"-DSPAN_MAPPING={mapping}",
      str(output/"source/span_guard.cpp"),str(output/"source/reference_main.cpp"),"-o",str(output/"span_reference")]
    subprocess.run(cmd,check=True,timeout=120)
    write_new(output/"c_build.json",(json.dumps({"command":cmd,"exit_code":0,"source_hashes":sources,"executable_sha256":hashlib.sha256((output/"span_reference").read_bytes()).hexdigest()})+"\n").encode())

def snapshot_sources(output):
    output=Path(output);record={}
    for name in ("span_guard.cpp","span_guard.hpp","reference_main.cpp"):
        dest=output/"source"/name
        if not dest.exists():write_new(dest,(ROOT/"hardware/span_guard"/name).read_bytes())
        record[name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    return record

def prepare_hls(output,design,action):
    output=Path(output).resolve();kind,mapping=DESIGNS[design]
    if action not in ("csim","synthesize","cosim","full-flow"):raise ValueError(action)
    def tclpath(path):
        s=str(path)
        if any(c in s for c in "{}\n\r"):raise ValueError("Unsupported Tcl path")
        return "{"+s+"}"
    snapshot_sources(output);source=output/"source/span_guard.cpp";tb=output/"source/reference_main.cpp"
    if action=="cosim" and not (output/"hls_project/solution/syn/report/span_guard_top_csynth.xml").exists():raise FileNotFoundError("Cosim requires the matching existing synthesis report")
    flags=f"-std=c++17 -DSPAN_ALGORITHM={kind} -DSPAN_MAPPING={mapping}"
    tcl=[f"open_project {tclpath(output/'hls_project')}","set_top span_guard_top",
      f"add_files {tclpath(source)} -cflags {{{flags}}}",f"add_files -tb {tclpath(tb)} -cflags {{{flags}}}",
      "open_solution solution -flow_target vivado","set_part xcve2802-vsvh1760-2MP-e-S",
      "create_clock -period 5.0 -name default","set_clock_uncertainty 0.5"]
    for arg in ("blob","bytes","ids","windows","count","load","empty","previous","threshold","output","return"):
        tcl.append(f"set_directive_interface -mode s_axilite -bundle control span_guard_top {arg}")
    argv=tclpath(str(output/"model.hls.bin")+" "+str(output/"cases.bin"))
    tcl.append(f"csim_design -argv {argv}")
    if action in ("synthesize","full-flow"):tcl.append("csynth_design")
    if action in ("cosim","full-flow"):
        rtl_argv=tclpath(str(output/"model.hls.bin")+" "+str(output/"cases_rtl.bin"))
        tcl.append(f"cosim_design -rtl verilog -tool xsim -trace_level port -enable_dataflow_profiling -argv {rtl_argv}")
    tcl.append("exit")
    write_new(output/(action+".tcl"),("\n".join(tcl)+"\n").encode())
    return {"prepared_tcl":str(output/(action+".tcl")),"command":["/home/jp19126/Xilinx/2025.2/Vitis/bin/vitis-run","--mode","hls","--tcl",str(output/(action+".tcl"))],"dispatched":False}

def run_vendor(output,design,action,max_seconds):
    """One explicitly invoked, coordinator-owned, bounded vendor process group."""
    from r1_challengers import resource_check
    output=Path(output).resolve()
    if os.name!="posix" or not hasattr(os,"sched_setaffinity"):raise RuntimeError("Run the vendor wrapper in Ubuntu WSL")
    if not output.is_relative_to(ROOT/"results/span_coupled_pilot_v1"):raise ValueError("Vendor evidence must remain in the approved study")
    if not 0<max_seconds<=7200:raise ValueError("Per-design vendor ceiling is7200seconds")
    prepared=json.loads((output/"prepared.json").read_text(encoding="utf-8"))
    if prepared["design"]!=design:raise ValueError("Prepared design mismatch")
    path=output/("vendor_"+action+".json")
    if path.exists():raise FileExistsError("Retain the prior attempt; choose a fresh correction directory")
    prior=[]
    for phase in ("csim","synthesize","cosim","full-flow"):
        old=output/("vendor_"+phase+".json")
        if old.exists():prior.append(json.loads(old.read_text(encoding="utf-8")))
    spent=sum(float(r.get("elapsed_seconds",0)) for r in prior)
    remaining=min(max_seconds,7200-spent)
    if remaining<=0 or (action=="full-flow" and prior):raise RuntimeError("Per-design vendor allowance exhausted or full-flow would repeat a phase")
    script=output/(action+".tcl")
    if not script.exists():raise FileNotFoundError("Prepare the exact Tcl before dispatch")
    for name,key in (("model.hls.bin","model_blob_sha256"),("cases.bin","cases_sha256"),("cases_rtl.bin","rtl_cases_sha256")):
        if hashlib.sha256((output/name).read_bytes()).hexdigest()!=prepared[key]:raise ValueError("Prepared artifact changed: "+name)
    for name,digest in json.loads((output/"c_build.json").read_text(encoding="utf-8"))["source_hashes"].items():
        if hashlib.sha256((output/"source"/name).read_bytes()).hexdigest()!=digest:raise ValueError("C-accepted source snapshot changed")
    if json.loads((output/"c_run.json").read_text(encoding="utf-8"))["exit_code"]!=0:raise RuntimeError("Actual C replay must pass before vendor dispatch")
    settings=Path("/home/jp19126/Xilinx/2025.2/Vitis/settings64.sh")
    executable=Path("/home/jp19126/Xilinx/2025.2/Vitis/bin/vitis-run")
    if not settings.is_file() or not executable.is_file():raise FileNotFoundError("Configured installed Vitis tool environment missing")
    command=["/bin/bash","-c",'source "$1" && exec "$2" --mode hls --tcl "$3"',"span-vendor",str(settings),str(executable),str(script)]
    affinity=sorted(os.sched_getaffinity(0))[:4]
    if len(affinity)!=4:raise RuntimeError("Four assigned CPU cores unavailable")
    started=time.monotonic();proc=None;last=0.;original_affinity=os.sched_getaffinity(0)
    record={"design":design,"action":action,"started_utc":datetime.now(timezone.utc).isoformat(),"command":command,
      "tcl_sha256":hashlib.sha256(script.read_bytes()).hexdigest(),"budget_seconds":remaining,"prior_design_vendor_seconds":spent,
      "affinity":affinity,"thread_environment":{k:"4" for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")},
      "state":"preflight","passed":False,"evidence_boundary":"Tool outcome only; clock estimates are not routed timing or physical performance"}
    def save():path.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    def group_present():
        if proc is None:return False
        try:os.killpg(proc.pid,0);return True
        except ProcessLookupError:return False
    def cleanup():
        if not group_present():return
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:return
        deadline=time.monotonic()+10
        while group_present() and time.monotonic()<deadline:time.sleep(.25)
        if group_present():
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        if proc.poll() is None:proc.wait(timeout=10)
    save()
    try:
        record["preflight"]=resource_check(True);save()
        os.sched_setaffinity(0,set(affinity));env=dict(os.environ);env.update(record["thread_environment"])
        with (output/("vendor_"+action+".stdout.log")).open("x",encoding="utf-8") as log:
            if (ROOT/"results/span_coupled_pilot_v1/pause.request").exists():raise RuntimeError("Study pause requested before dispatch")
            proc=subprocess.Popen(command,cwd=output,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record.update(state="running",pid=proc.pid,actual_affinity=sorted(os.sched_getaffinity(proc.pid)));save()
            while proc.poll() is None:
                if time.monotonic()-started>remaining:raise TimeoutError("Shared per-design vendor time ceiling reached")
                if (ROOT/"results/span_coupled_pilot_v1/pause.request").exists():raise RuntimeError("Study pause requested")
                if time.monotonic()-last>=5:
                    observation=resource_check()
                    with (output/("vendor_"+action+".resources.jsonl")).open("a",encoding="utf-8") as f:f.write(json.dumps(observation)+"\n")
                    record["latest_memory"]=observation;save();last=time.monotonic()
                time.sleep(.25)
        if proc.returncode:raise RuntimeError("Vendor tool returned nonzero; retain its stdout log")
        record.update(state="completed",passed=True)
    except BaseException as exc:
        record.update(state="failed",error=str(exc));cleanup();raise
    finally:
        # Terminate only descendants remaining in this invocation's process group.
        cleanup();os.sched_setaffinity(0,original_affinity)
        record.update(elapsed_seconds=time.monotonic()-started,returncode=proc.returncode if proc else None,assigned_group_present=group_present())
        save()
    return record

def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=("prepare","build-c","run-c","prepare-hls","run-vendor"));p.add_argument("--design",choices=DESIGNS,required=True);p.add_argument("--output",required=True);p.add_argument("--export");p.add_argument("--cases");p.add_argument("--vendor-action",default="full-flow",choices=("csim","synthesize","cosim","full-flow"));p.add_argument("--max-seconds",type=int,default=7200);p.add_argument("--trace",action="store_true");a=p.parse_args()
    if a.action=="prepare":result=prepare(a.export,a.cases,a.output,a.design)
    elif a.action=="build-c":build_c(a.output,a.design);result={"compiled":True}
    elif a.action=="prepare-hls":result=prepare_hls(a.output,a.design,a.vendor_action)
    elif a.action=="run-vendor":result=run_vendor(a.output,a.design,a.vendor_action,a.max_seconds)
    else:
        out=Path(a.output);cmd=[str(out/"span_reference"),str(out/"model.hls.bin"),str(out/"cases.bin")]
        if (out/"c_run.json").exists() or list(out.glob("trace_c*_w*.bin")):raise FileExistsError("Replay output already exists")
        if a.trace:cmd.append(str(out/"trace"))
        done=subprocess.run(cmd,text=True,capture_output=True,timeout=1800)
        write_new(out/"c_run.json",(json.dumps({"command":cmd,"exit_code":done.returncode,"stdout":done.stdout,"stderr":done.stderr},indent=2)+"\n").encode())
        if done.returncode:raise RuntimeError(done.stderr or done.stdout)
        result={"exit_code":done.returncode,"stdout":done.stdout}
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
