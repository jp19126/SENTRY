"""Package complete-document board inputs offline; physical execution is opt-in.

--execute --board-authorized may be used only after explicit human permission to
program/use the board. Default operation performs no XRT/device operation.
"""
from __future__ import annotations
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import hashlib
import json
import math
from pathlib import Path
from queue import Queue,Empty
import subprocess
import threading
import time
import numpy as np
from transformers import AutoTokenizer
from common import ROOT,SHARED,read,rows,sha,write

def input_windows(tokenizer,text):
    """Independent transport packing: all content tails, key mask, zero types."""
    began=time.perf_counter()
    content=tokenizer(text,add_special_tokens=False,truncation=False,return_attention_mask=False,return_token_type_ids=False,verbose=False)['input_ids']
    tokenization=time.perf_counter()-began;began=time.perf_counter();records=[];at=0
    while True:
        tokens=[101]+content[at:at+254]+[102];n=len(tokens)
        records.append(np.asarray([tokens+[0]*(256-n),[1]*n+[0]*(256-n),[0]*256],dtype='<i4'))
        if at+254>=len(content):break
        at+=190
    return np.stack(records),{'tokenization_seconds':tokenization,'window_assembly_seconds':time.perf_counter()-began,'content_tokens':len(content),'windows':len(records)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timing-manifest',required=True,type=Path);p.add_argument('--raw',type=Path,help='Required only for timing manifests that contain IDs rather than raw documents')
    p.add_argument('--export',type=Path,default=ROOT/'results/bert_aie_v1/export/w8a8_bf16_attention')
    p.add_argument('--evaluation',type=Path,help='Completed frozen-format score record supplying the calibrated margin threshold')
    p.add_argument('--alpha',type=float,choices=(.01,.005,.02,.05),default=.01)
    p.add_argument('--host',type=Path);p.add_argument('--xclbin',type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--repeats',type=int);p.add_argument('--warmups',type=int)
    p.add_argument('--max-seconds',type=int,default=3600);p.add_argument('--execute',action='store_true');p.add_argument('--board-authorized',action='store_true')
    a=p.parse_args()
    if a.execute and not a.board_authorized:p.error('--execute requires explicit prior human board permission and --board-authorized')
    if not 1<=a.max_seconds<=7200:p.error('Finite time bound required')
    if a.execute and not all((a.host,a.xclbin,a.evaluation)):p.error('Physical execution requires --host --xclbin --evaluation')
    for path in reversed([a.out.absolute(),*a.out.absolute().parents]):
        if str(path)!=path.anchor:path.mkdir(exist_ok=True)
    if (a.out/'package.json').exists():p.error('Preserve previous packages; choose a new --out directory')
    began=time.perf_counter();deadline=time.monotonic()+a.max_seconds
    timing=read(a.timing_manifest);contract=read(SHARED/'contract.json')
    for key,default in (('repeats',3),('warmups',1)):
        supplied=getattr(a,key)
        if key in timing and supplied is not None and supplied!=timing[key]:raise ValueError('Shared timing '+key+' cannot change')
        setattr(a,key,timing.get(key,default) if supplied is None else supplied)
    if not 1<=a.repeats<=20 or not 0<=a.warmups<=5:p.error('Finite repeats/warmups bounds required')
    if timing.get('common_contract_sha256') is not None:
        if timing['common_contract_sha256']!=sha(SHARED/'contract.json'):raise ValueError('Timing/shared contract mismatch')
    elif a.timing_manifest.resolve()!=(SHARED/contract['timing']['manifest']).resolve():
        raise ValueError('Legacy timing manifest must be the exact contract-owned path')
    if 'documents' in timing:
        documents=timing['documents']
    else:
        raw=a.raw or ROOT/timing['raw_source']
        lookup={r['document_id']:r for r in rows(raw)};documents=[lookup[k] for k in timing['document_ids']]
        if 'raw_sha256' in timing and sha(raw)!=timing['raw_sha256']:raise ValueError('Timing raw identity changed')
    if not 1<=len(documents)<=128 or len({d['document_id'] for d in documents})!=len(documents):raise ValueError('Bounded unique timing documents required')
    for d in documents:
        if hashlib.sha256(d['text'].encode('utf-8')).hexdigest()!=d['text_sha256']:raise ValueError('Timing document text changed')
    manifest=read(a.export/'manifest.json');export_hash=sha(a.export/'manifest.json')
    if manifest['variant']!='w8a8_bf16_attention':raise ValueError('Current hardware implements BF16 attention candidate')
    for filename,info in manifest['files'].items():
        if Path(filename).name!=filename or sha(a.export/filename)!=info['sha256']:raise ValueError('Export stream identity changed')
    threshold=None;evaluation_hash=None
    if a.evaluation:
        evaluation=read(a.evaluation);evaluation_hash=sha(a.evaluation)
        if not evaluation.get('complete') or evaluation['target']['export_manifest_sha256']!=export_hash:raise ValueError('Threshold/model export identities differ')
        if evaluation['target']['score']!='max_window_logit1_minus_logit0':raise ValueError('Expected margin score domain')
        threshold=evaluation['thresholds'][str(a.alpha)]
        if not math.isfinite(threshold):raise ValueError('Finite frozen threshold required')
        if 'source_role' in timing:
            selection=Path(evaluation['data']['selection']['directory'])
            if selection.name!=timing['source_role'] or sha(selection/'manifest.json')!=timing['source_manifest_sha256']:
                raise ValueError('Timing/evaluation prepared selection identity mismatch')
        else:
            expected_role={d.get('construction_role') for d in documents if not d.get('timing_only')}
            # Supplemental timings use new_selection, while the matched primary
            # threshold is from new_calibration; this is an explicit timing-only
            # workload, not re-evaluation of its labels or threshold fitting.
            if expected_role!={'new_selection'} or Path(evaluation['data']['calibration']['directory']).name!='new_calibration':
                raise ValueError('Supplemental timing needs the primary calibrated candidate')
    tokpath=ROOT/contract['tokenizer']['path']
    for filename,digest in contract['tokenizer']['files_sha256'].items():
        if sha(tokpath/filename)!=digest:raise ValueError('Tokenizer identity changed')
    tokbegan=time.perf_counter();tokenizer=AutoTokenizer.from_pretrained(tokpath,local_files_only=True,use_fast=True)
    tokenizer_load=time.perf_counter()-tokbegan
    if (tokenizer.cls_token_id,tokenizer.sep_token_id,tokenizer.pad_token_id)!=(101,102,0):raise ValueError('Unexpected tokenizer special IDs')
    package={'schema':'sentry_bert_aie_board_package_v1','physical_executed':a.execute,'timing_manifest_sha256':sha(a.timing_manifest),
             'export_manifest_sha256':export_hash,'evaluation_sha256':evaluation_hash,'threshold_margin':threshold,'alpha':a.alpha,
             'repeats':a.repeats,'warmups':a.warmups,
             'decision':'max(logit1-logit0) > threshold_margin','tokenizer':contract['tokenizer'],'tokenizer_load_seconds':tokenizer_load,
             'host_source_sha256':sha(ROOT/'hardware/bert_aie_v1/host.cpp'),'wrapper_sha256':sha(Path(__file__)),
             'documents':[],'window_record':'3072 bytes little-endian int32 ids256 then key_mask256 then token_types256; consecutive complete windows',
             'boundary':'All raw-document windows retained until decision. Cold host setup includes device programming and parameter transfer; warm contiguous timings include real tokenization, assembly, file I/O, IPC, host transfer/launch/return/aggregation.',
             'limitations':['No physical execution or performance result is established by offline packaging. No energy is inferred from TDP.',
                            'Warm latency includes Python/file/stdin transport overhead actually used by this harness.',
                            'Tokenization setup is separate from warm per-document input-to-decision latency.']}
    proc=None;observations=[];cold_setup=None;stderr_handle=None
    if a.execute:
        # This is the sole physical action in this script, guarded above.
        package['host_binary_sha256']=sha(a.host);package['xclbin_sha256']=sha(a.xclbin)
        stderr_handle=(a.out/'host_stderr.log').open('w',encoding='utf-8')
        host_start=time.perf_counter()
        proc=subprocess.Popen([str(a.host),str(a.xclbin),str(a.export),repr(threshold),'--stdin'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr_handle,text=True,bufsize=1)
        q=Queue()
        def pump():
            for line in proc.stdout:q.put(line)
            q.put(None)
        threading.Thread(target=pump,daemon=True).start()
        def event():
            while True:
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('Bounded board run exhausted')
                try:line=q.get(timeout=min(remaining,5))
                except Empty:continue
                if line is None:raise RuntimeError('Physical host ended unexpectedly; inspect host_stderr.log')
                with (a.out/'host_stdout.jsonl').open('a',encoding='utf-8') as log:log.write(line)
                return json.loads(line)
        try:
            cold_setup=event()
            if 'cold_setup_seconds' not in cold_setup:raise ValueError('Expected persistent host cold-start handshake')
        except BaseException:
            proc.terminate();proc.wait(timeout=10);stderr_handle.close()
            raise
        package['host_start_to_ready_seconds']=time.perf_counter()-host_start;package['host_cold_setup']=cold_setup
    try:
        loops=a.warmups+a.repeats if a.execute else 1
        for repeat in range(loops):
            for index,d in enumerate(documents):
                if time.monotonic()>deadline:raise TimeoutError('Bounded packaging/run exhausted')
                document_start=time.perf_counter();array,detail=input_windows(tokenizer,d['text'])
                if 'windows' in d and detail['windows']!=d['windows']:raise ValueError('Timing document window count differs')
                if 'content_tokens' in d and detail['content_tokens']!=d['content_tokens']:raise ValueError('Timing document token length differs')
                path=(a.out/f'document_{index:03d}.bin').absolute();pack_start=time.perf_counter();path.write_bytes(array.tobytes())
                detail['file_write_seconds']=time.perf_counter()-pack_start
                if repeat==0:package['documents'].append({'document_id':d['document_id'],'text_sha256':d['text_sha256'],'timing_only':d.get('timing_only',False),'input_file':path.name,'input_sha256':sha(path),**detail})
                if not a.execute:continue
                if '\n' in str(path):raise ValueError('Input path cannot contain newline')
                proc.stdin.write(str(path)+'\n');proc.stdin.flush();window_events=[]
                while True:
                    record=event()
                    if 'max_margin' in record:break
                    if 'window_index' not in record:raise ValueError('Unexpected host protocol record')
                    window_events.append(record)
                elapsed=time.perf_counter()-document_start
                if len(window_events)!=len(array) or record['windows']!=len(array):raise ValueError('Host omitted complete-document windows')
                calculated=max(float(r['logits'][1])-float(r['logits'][0]) for r in window_events)
                if abs(calculated-record['max_margin'])>1e-5:raise ValueError('Host document margin inconsistent')
                if record['blocked']!=(record['max_margin']>threshold):raise ValueError('Host threshold rule inconsistent')
                observations.append({'repeat':repeat,'warmup':repeat<a.warmups,'document_id':d['document_id'],**detail,'input_to_decision_seconds':elapsed,'host':record,'window_events':window_events})
        if proc:
            proc.stdin.close();proc.wait(timeout=max(1,deadline-time.monotonic()))
            if proc.returncode:raise RuntimeError('Host failed on shutdown')
        package['elapsed_seconds']=time.perf_counter()-began;package['physical_observations']=observations
        if observations:
            values=[r['input_to_decision_seconds'] for r in observations if not r['warmup']]
            package['warm_summary_seconds']={'mean':float(np.mean(values)),'p50':float(np.quantile(values,.5)),'p95':float(np.quantile(values,.95)),'p99':float(np.quantile(values,.99)),'documents':len(values)}
            package['sustained_documents_per_second']=len(values)/sum(values)
            package['cold_first_result_seconds']=package['host_start_to_ready_seconds']+tokenizer_load+observations[0]['input_to_decision_seconds']
            package['cold_first_result_boundary']='Sum of measured sequential tokenizer loading, host startup-to-ready and first raw-document-to-decision; excludes preflight hashing/raw manifest parsing.'
        write(a.out/'package.json',package)
        print(json.dumps({'out':str(a.out),'physical_executed':a.execute,'documents':len(package['documents']),'threshold_margin':threshold,'elapsed_seconds':package['elapsed_seconds']},indent=2))
    except BaseException as exc:
        package['failure']=str(exc);package['physical_observations']=observations
        write(a.out/'failed_package.json',package)
        raise
    finally:
        if proc and proc.poll() is None:proc.terminate();proc.wait(timeout=10)
        if stderr_handle:stderr_handle.close()

if __name__=='__main__':main()
