"""Stream complete frozen corpora through loaded-once portable full BERT C++.

No GPU, vendor tool or board access. Scalar FP32 CpuTiles are a different
floating accumulation order from Torch/AIE. Freeze required before corpus reads.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from queue import Queue,Empty
import subprocess
import threading
import time
import numpy as np
from common import ROOT,read,rows,sha,write

def unix_path(path):
    value=str(Path(path).absolute()).replace('\\','/')
    for prefix in ('//wsl.localhost/Ubuntu/','//wsl$/Ubuntu/'):
        if value.lower().startswith(prefix.lower()):return '/'+value[len(prefix):]
    return str(path)

class CppStream:
    def __init__(self,functional,export,deadline,wsl_distro,stderr_path):
        command=[unix_path(functional),unix_path(export),'--stdin','both']
        if os.name=='nt':command=['wsl.exe','-d',wsl_distro,'--exec']+command
        self.stderr=Path(stderr_path).open('w',encoding='utf-8');self.deadline=deadline
        self.proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr)
        self.queue=Queue();self.count=0;self.command=command
        def pump():
            for line in self.proc.stdout:self.queue.put(line)
            self.queue.put(None)
        threading.Thread(target=pump,daemon=True).start()
    def window(self,ids,mask,types=None):
        if types is None:types=np.zeros(256,dtype='<i4')
        array=np.asarray([ids,mask,types],dtype='<i4')
        if array.shape!=(3,256) or np.any(array[0]<0) or np.any(array[0]>=30522) or np.any((array[1]!=0)&(array[1]!=1)) or not array[1].any():raise ValueError('Invalid transport window')
        self.proc.stdin.write(array.tobytes());self.proc.stdin.flush()
        while True:
            remaining=self.deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Bounded C++ functional dispatch exhausted')
            try:line=self.queue.get(timeout=min(remaining,5))
            except Empty:continue
            if line is None:raise RuntimeError('C++ process ended: inspect stderr log')
            row=json.loads(line)
            if row['window_index']!=self.count:raise ValueError('C++ stream window association changed')
            self.count+=1
            for mapping in ('resident','external'):
                if row[mapping]['status']!=0 or len(row[mapping]['logits'])!=2 or not all(math.isfinite(v) for v in row[mapping]['logits']):raise ValueError('C++ rejected input or nonfinite output')
            if not row['mapping_bit_identical']:raise ValueError('Resident/external arithmetic differ')
            return row
    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:self.proc.wait(timeout=max(1,min(10,self.deadline-time.monotonic())))
            except subprocess.TimeoutExpired:self.proc.terminate();self.proc.wait(timeout=10)
        self.stderr.close()
        if self.proc.returncode:raise RuntimeError('C++ functional process failed')
    def terminate(self):
        if self.proc.poll() is None:self.proc.terminate();self.proc.wait(timeout=10)
        self.stderr.close()

def verify_export(folder):
    m=read(Path(folder)/'manifest.json')
    if m['variant']!='w8a8_bf16_attention':raise ValueError('Current C++ implements only the frozen BF16 attention format')
    for name,desc in m['files'].items():
        if Path(name).name!=name or sha(Path(folder)/name)!=desc['sha256']:raise ValueError('Export identity changed')
    return sha(Path(folder)/'manifest.json')

def smoke(args,stream,export_hash):
    expected=read(args.smoke_fixtures);records=[]
    for name,record in expected.items():
        if not name.startswith('fit_'):raise ValueError('Smoke must use FIT fixtures only')
        path=args.smoke_fixtures.parent/(name+'.bin')
        if sha(path)!=record['input_sha256']:raise ValueError('FIT fixture bytes changed')
        data=np.fromfile(path,dtype='<i4').reshape(3,256)
        result=stream.window(*data);gold=record['candidate_logits']['w8a8_bf16_attention'][0]
        result.update(case=name,python_logits=gold,max_logit_error=max(abs(a-b) for a,b in zip(gold,result['resident']['logits'])))
        records.append(result)
    return {'schema':'sentry_bert_cpp_smoke_v1','complete':True,'data_role':'fit','export_manifest_sha256':export_hash,'cases':records,
            'mapping_bit_identical':True,'max_logit_error':max(r['max_logit_error'] for r in records),'source_fixture_sha256':sha(args.smoke_fixtures),
            'evidence_boundary':'Portable C++ whole-model FIT differential; no corpus quality/AIE/RTL/board timing claim'}

def frozen_corpus(args):
    # All source/binary authorization precedes opening reference scores or arrays.
    if not args.freeze or not args.freeze_sha256 or sha(args.freeze)!=args.freeze_sha256:raise ValueError('Exact prospective freeze required before corpus access')
    freeze=read(args.freeze)
    needed=[Path(__file__),ROOT/'hardware/bert_aie_v1/model.hpp',ROOT/'hardware/bert_aie_v1/functional.cpp',args.functional]
    for path in needed:
        key=str(Path(path).resolve().relative_to(ROOT.resolve())).replace('\\','/')
        if freeze['source_sha256'].get(key)!=sha(path):raise ValueError('C++ runner/source/binary not frozen: '+key)
    from analyze import load_verified
    complete,reference=load_verified(args.reference,args.freeze,args.freeze_sha256)
    if complete['target']['variant']!='w8a8_bf16_attention':raise ValueError('Expected matched frozen Python BF16 candidate')
    return complete,reference

def corpus(args,stream,complete,reference,export_hash):
    import hashlib
    from evaluate import load_prepared
    from analyze import fit_threshold,metrics,ALPHAS
    if complete['target']['export_manifest_sha256']!=export_hash:raise ValueError('Reference export mismatch')
    outputs={};details={};max_logit_error=None
    with (args.out/'window_execution.jsonl').open('w',encoding='utf-8') as log:
        for role in ('calibration','selection'):
            folder=Path(complete['data'][role]['directory']);data=load_prepared(folder);refs=reference[role]
            if [r['document_id'] for r in data['documents']]!=[r['document_id'] for r in refs]:raise ValueError('Python/C++ document membership differs')
            scored=[];role_error=0.;logit_count=0
            for di,(document,ref) in enumerate(zip(data['documents'],refs)):
                lo,hi=map(int,data['starts'][di:di+2]);margins=[];logits=[];mapping_seconds={'resident':0.,'external':0.}
                digest=hashlib.sha256()
                for key in ('ids','masks'):
                    value=np.ascontiguousarray(data[key][lo:hi]);digest.update(key.encode());digest.update(str(value.dtype).encode());digest.update(str(value.shape).encode());digest.update(value.tobytes())
                if digest.hexdigest()!=ref['window_input_sha256']:raise ValueError('Python/C++ input bytes differ')
                for offset,wi in enumerate(range(lo,hi)):
                    execution=stream.window(data['ids'][wi],data['masks'][wi]);value=execution['resident']['logits'];margin=float(np.float32(value[1])-np.float32(value[0]))
                    logits.append(value);margins.append(margin);role_error=max(role_error,abs(margin-ref['window_margins'][offset]))
                    for mapping in mapping_seconds:mapping_seconds[mapping]+=execution[mapping]['seconds']
                    execution.update(role=role,document_id=document['document_id'],document_window_index=offset,python_margin=ref['window_margins'][offset],cpp_margin=margin)
                    if 'window_logits' in ref:
                        execution['python_logits']=ref['window_logits'][offset]
                        err=max(abs(a-b) for a,b in zip(value,execution['python_logits']));max_logit_error=max(max_logit_error or 0.,err);logit_count+=1
                    log.write(json.dumps(execution,allow_nan=False)+'\n')
                scored.append({**document,'window_count':hi-lo,'window_margins':margins,'window_logits':logits,'margin':max(margins),
                               'window_input_sha256':digest.hexdigest(),'portable_functional_seconds':mapping_seconds})
            outputs[role]=scored;details[role]={'documents':len(scored),'windows':sum(r['window_count'] for r in scored),'max_window_margin_delta':role_error,'windows_with_python_logits':logit_count}
            (args.out/(role+'_scores.jsonl')).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in scored),encoding='utf-8')
    own={str(a):fit_threshold([r['margin'] for r in outputs['calibration'] if r['label']==0],a) for a in ALPHAS};comparisons={}
    for role in outputs:
        comparisons[role]={}
        for key,threshold in complete['thresholds'].items():
            changed=[];near=[];own_changed=[]
            for ref,current in zip(reference[role],outputs[role]):
                delta=current['margin']-ref['margin'];record={'document_id':ref['document_id'],'label':ref['label'],'python_margin':ref['margin'],'cpp_margin':current['margin'],'delta':delta,'distance_to_python_threshold':ref['margin']-threshold}
                if (current['margin']>threshold)!=(ref['margin']>threshold):changed.append(record)
                if (current['margin']>own[key])!=(ref['margin']>threshold):own_changed.append(record)
                if abs(ref['margin']-threshold)<=abs(delta):near.append(record)
            comparisons[role][key]={'python_threshold':threshold,'cpp_own_threshold':own[key],'fixed_threshold_disagreements':changed,'own_calibrated_disagreements':own_changed,'near_threshold':near,
                                    'cpp_fixed_threshold_metrics':metrics(outputs[role],threshold),'cpp_own_threshold_metrics':metrics(outputs[role],own[key])}
    return {'schema':'sentry_bert_cpp_corpus_comparison_v1','complete':True,'reference_complete_sha256':sha(args.reference/'complete.json'),'freeze_sha256':args.freeze_sha256,
            'target':complete['target'],'data':complete['data'],'cpp_own_thresholds':own,'python_thresholds':complete['thresholds'],'comparisons':comparisons,'roles':details,
            'mapping_bit_identical':True,'max_logit_error_when_python_logits_saved':max_logit_error,
            'missing_logit_limit':'Saved Python margins can be compared for every window; direct logits only where window_logits exists.',
            'score_files':{role:{'name':role+'_scores.jsonl','sha256':sha(args.out/(role+'_scores.jsonl'))} for role in outputs},
            'evidence_boundary':'All four BERT layers and all document windows executed through portable CpuTiles. Resident/external scheduling shares exact scalar arithmetic. Native AIE accumulator ordering, HLS/RTL/board execution and latency remain separate evidence.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--functional',type=Path,default=ROOT/'build/bert_aie_v1/common/bert_functional_stream')
    p.add_argument('--export',required=True,type=Path);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke-fixtures',type=Path);mode.add_argument('--reference',type=Path)
    p.add_argument('--freeze',type=Path);p.add_argument('--freeze-sha256');p.add_argument('--out',required=True,type=Path)
    p.add_argument('--max-seconds',type=int,default=1800);p.add_argument('--wsl-distro',default='Ubuntu');a=p.parse_args()
    if not 1<=a.max_seconds<=7200 or a.smoke_fixtures and a.max_seconds>30:p.error('Smoke <=30seconds, corpus <=7200seconds')
    for path in reversed([a.out.absolute(),*a.out.absolute().parents]):
        if str(path)!=path.anchor:path.mkdir(exist_ok=True)
    if (a.out/'complete.json').exists():p.error('Preserve completed C++ executions; choose new output')
    started=time.monotonic();export_hash=verify_export(a.export)
    authorized=frozen_corpus(a) if a.reference else None
    stream=CppStream(a.functional,a.export,started+a.max_seconds,a.wsl_distro,a.out/'stderr.log')
    try:
        result=smoke(a,stream,export_hash) if a.smoke_fixtures else corpus(a,stream,*authorized,export_hash)
        stream.close();result.update(elapsed_seconds=time.monotonic()-started,command=stream.command,functional_binary_sha256=sha(a.functional),
                                    source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in (Path(__file__),ROOT/'hardware/bert_aie_v1/model.hpp',ROOT/'hardware/bert_aie_v1/functional.cpp')})
        write(a.out/'complete.json',result);print(json.dumps({'complete':True,'out':str(a.out),'windows':stream.count,'elapsed_seconds':result['elapsed_seconds'],'max_logit_error':result.get('max_logit_error')},indent=2))
    except BaseException as exc:
        stream.terminate();write(a.out/'failure.json',{'complete':False,'error':str(exc),'windows_executed':stream.count,'elapsed_seconds':time.monotonic()-started});raise

if __name__=='__main__':main()
