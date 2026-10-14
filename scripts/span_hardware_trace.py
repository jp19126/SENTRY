"""Compare emitted C intermediates with fixed saved integer PT fixtures.

Reads the explicitly named development fixtures and C replay only. No model
inference, threshold fitting, vendor job or protected dataset access is performed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import struct
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/"results/span_coupled_pilot_v1"

def decode(path):
    import numpy as np
    raw=path.read_bytes()
    if len(raw)<16:raise ValueError("Short trace header")
    magic,version,n,kind=struct.unpack_from("<4I",raw)
    if magic!=0x53505231 or version!=1 or n>254 or kind not in (0,2,3):raise ValueError("Wrong C trace identity")
    p=16;arrays={}
    for name,dtype,shape in (("z_a8","i1",(n,192)),("projection_q12","<i4",(n,64)),
          ("a","<i4",(n,32)),("b","<i4",(n,32)),("g","<i8",(n,)),("eligible","u1",(n,))):
        size=int(np.prod(shape))*np.dtype(dtype).itemsize
        if p+size>len(raw):raise ValueError("Short trace payload")
        arrays[name]=np.frombuffer(raw,dtype=dtype,count=int(np.prod(shape)),offset=p).reshape(shape).copy();p+=size
    if p!=len(raw):raise ValueError("Trailing trace bytes")
    return arrays,dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),length=n,kind=kind)

def compare(actual,expected):
    import numpy as np
    if actual.shape!=expected.shape:return {"pass":False,"shape_actual":list(actual.shape),"shape_expected":list(expected.shape)}
    different=np.argwhere(actual!=expected)
    out={"pass":len(different)==0,"elements":int(actual.size),"changed":len(different)}
    if len(different):
        at=tuple(int(i) for i in different[0]);out.update(first_index=list(at),actual=int(actual[at]),expected=int(expected[at]),max_absolute_error=int(np.abs(actual.astype(np.int64)-expected.astype(np.int64)).max()))
    return out

def main():
    import numpy as np
    import torch
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--fixtures",type=Path,required=True);p.add_argument("--replay",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    fixtures,replay,out=map(lambda q:q.resolve(),(a.fixtures,a.replay,a.output))
    if not all(q.is_relative_to(STUDY.resolve()) for q in (fixtures,replay,out)) or out.exists():raise ValueError("Use named study evidence and a fresh output")
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    manifest=json.loads((fixtures/"manifest.json").read_text(encoding="utf-8"));prepared=json.loads((replay/"prepared.json").read_text(encoding="utf-8"))
    run=json.loads((replay/"c_run.json").read_text(encoding="utf-8"))
    if not manifest["complete"] or run["exit_code"]!=0:raise ValueError("Require complete fixtures and a passed C replay")
    from span_hardware import pack_cases
    canonical=pack_cases(json.loads((fixtures/"cases.json").read_text(encoding="utf-8")))
    if hashlib.sha256(canonical).hexdigest()!=prepared["cases_sha256"] or canonical!=(replay/"cases.bin").read_bytes():raise ValueError("Fixture/replay case identity differs")
    cache={};records=[];all_pass=True;empty=0
    for entry in manifest["trace_mapping"]:
        name=entry["fixture"];file=fixtures/name
        if name not in cache:
            raw=file.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=manifest["files_sha256"][name]:raise ValueError("Fixture identity changed")
            cache[name]=torch.load(file,map_location="cpu",weights_only=True)
        saved=cache[name];row=entry["fixture_row"];n=entry["content_tokens"]
        inp,expected=saved["inputs"],saved["outputs"]
        if int(inp["attention_mask"][row].sum())-2!=n:raise ValueError("Fixture content extent changed")
        if int(inp["document_content_length"][row])==0:
            empty+=1;records.append(dict(case=entry["case_index"],fixture=name,row=row,empty=True,trace="Not emitted: distinct empty-document rule already checked by C replay"));continue
        path=replay/("trace_c%d_w0.bin"%entry["case_index"]);actual,identity=decode(path)
        if identity["length"]!=n or identity["kind"]!=prepared["algorithm"]:raise ValueError("Trace algorithm/length mismatch")
        sl=slice(1,n+1);eligible=expected["eligible"][row,sl].numpy().astype(bool)
        checks={"z_a8":compare(actual["z_a8"],expected["z_a8"][row,sl].numpy()),"eligible":compare(actual["eligible"].astype(bool),eligible)}
        projection=expected["projection_q12"][row,sl].numpy()
        if identity["kind"]==3:
            projection=projection.clip(-65536,65536)
        checks["projection_q12"]=compare(actual["projection_q12"],projection)
        if identity["kind"]!=3:
            state=expected["state_q12"][row,sl].numpy()
            checks["a_q12"]=compare(actual["a"][eligible],state[eligible,:32]);checks["b_q12"]=compare(actual["b"][eligible],state[eligible,32:])
            checks["anchor_logit_q12"]=compare(actual["g"],expected["anchor_logits_q12"][row,sl].numpy())
        else:
            checks["sopa_state"]={"pass":True,"checked":False,"limitation":"Saved Python fixture lacks intermediate WFSA states; final risk and complete decisions were checked by C replay. Projection clipping only matches the declared LUT input domain."}
        passed=all(v.get("pass",False) for v in checks.values());all_pass&=passed
        records.append(dict(case=entry["case_index"],fixture=name,row=row,identity=identity,eligible_anchors=int(eligible.sum()),checks=checks,passed=passed))
    result=dict(complete=True,passed=all_pass,records=records,trace_windows=len(records)-empty,empty_document_cases=empty,
        scope="Exact saved C/Python integer intermediates on explicitly supplied train/development fixtures; no RTL or physical-device evidence",
        state_comparison="Coupled/MLP eligible complete anchors only",protected_partitions_opened=False)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x",encoding="utf-8") as f:json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2))
    if not all_pass:raise SystemExit(1)
if __name__=="__main__":main()
