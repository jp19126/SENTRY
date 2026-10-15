"""CUDA-event graph replay timing on the exact supplemental 18/77 inputs.

This standalone supplemental measurement does not change frozen scoring sources,
model/export identities, calibration or quality. Root dispatches it only after
the qualified final profiles, while holding the task's GPU and vendor leases.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,OUT,SHARED,read,sha,write
from evaluate import ensure_directory,load_target,target_descriptor
from profile_detector import GraphRunner,graph_execution_identity

MANIFEST_SHA='8d141711fcee46d65394c37460ef4bb377bd7346592ce11afa240ed6cf254162'
VARIANTS=('source','w8a8_bf16_attention')

def pinned_profiles(args,frozen):
    profiles={}
    for variant in VARIANTS:
        path=args.profile_root/('supplemental_confirmed_'+variant+'_cuda_cuda_graph.json')
        profile=read(path)
        if not profile.get('complete') or profile['backend']!='cuda_graph' or profile['device']!='cuda':raise ValueError('Completed final CUDA graph profile required')
        if profile['target']!=frozen['models'][variant] or profile['execution_identity']!=frozen['execution_routes'][variant]:raise ValueError('Final profile model/execution identity differs')
        if profile['execution_identity']!=graph_execution_identity(variant):raise ValueError('Frozen graph implementation changed')
        if profile['timing_manifest_sha256']!=MANIFEST_SHA:raise ValueError('Final profile timing membership differs')
        if profile['execution_quality']['freeze_sha256']!=args.freeze_sha256:raise ValueError('Final profile confirmation freeze differs')
        if len(profile['samples'])!=18 or sum(r['windows'] for r in profile['samples'])!=77:raise ValueError('Expected exact supplemental18/77 final timing inputs')
        profiles[variant]={'path':path,'metadata':profile,'sha256':sha(path)}
    return profiles

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze',type=Path,required=True);p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--profile-root',type=Path,default=OUT/'profiling/final')
    p.add_argument('--timing-manifest',type=Path,default=SHARED/'data/timing_manifest.json')
    p.add_argument('--calibration',type=Path,default=OUT/'numerics/calibration_fit64.json')
    p.add_argument('--export',type=Path,default=OUT/'export/w8a8_bf16_attention')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--max-seconds',type=int,default=60)
    args=p.parse_args();began=time.monotonic();process_started=time.perf_counter()
    if not 1<=args.max_seconds<=120:raise ValueError('CUDA-event supplemental dispatch must remain bounded to120seconds')
    ensure_directory(args.out.parent)
    if args.out.exists():raise ValueError('Preserve prior CUDA-event measurements; choose fresh output')
    if sha(args.freeze)!=args.freeze_sha256:raise ValueError('Wrong final freeze digest')
    frozen=read(args.freeze)
    if not frozen['complete'] or not frozen['nomination_before_confirmation']:raise ValueError('Final prospective freeze required')
    for name,digest in frozen['source_sha256'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha(path)!=digest:raise ValueError('Frozen numerical/scoring source changed: '+name)
    for resource in ('vendor','gpu0'):
        if read(SHARED/'locks'/(resource+'.lock')/'owner.json')['task']!='bert_aie_v1':raise ValueError('CUDA-event profiling requires task GPU and vendor leases')
    if sha(args.timing_manifest)!=MANIFEST_SHA:raise ValueError('Exact supplemental timing manifest required')
    timing=read(args.timing_manifest);profiles=pinned_profiles(args,frozen)
    documents=timing['documents'];ids=[d['document_id'] for d in documents]
    if len(ids)!=18 or len(set(ids))!=18:raise ValueError('Exact18 distinct timing documents required')
    from tokenizers import Tokenizer
    tokenizer_path=ROOT/'checkpoints/floating/epoch_3/tokenizer.json'
    tokenizer=Tokenizer.from_file(str(tokenizer_path));tokenizer.no_truncation();tokenizer.no_padding()
    if [tokenizer.token_to_id(t) for t in ('[CLS]','[SEP]','[PAD]')]!=[101,102,0]:raise ValueError('Tokenizer special IDs differ')
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise ValueError('CUDA is required; no fallback timing')
    prepared={};input_hashes={}
    for document in documents:
        if hashlib.sha256(document['text'].encode()).hexdigest()!=document['text_sha256']:raise ValueError('Manifest raw text changed')
        content=tokenizer.encode(document['text'],add_special_tokens=False).ids
        if len(content)!=document['content_tokens']:raise ValueError('Exact tokenizer content length differs')
        pairs=[];digest=hashlib.sha256()
        for at in range(0,max(1,len(content)),190):
            values=[101]+content[at:at+254]+[102];n=len(values)
            pair={'input_ids':torch.tensor([values+[0]*(256-n)],dtype=torch.long),
                  'attention_mask':torch.tensor([[1]*n+[0]*(256-n)],dtype=torch.long)}
            for key in ('input_ids','attention_mask'):digest.update(pair[key].numpy().tobytes())
            pairs.append(pair)
            if at+254>=len(content):break
        if len(pairs)!=document['windows']:raise ValueError('Exact tokenizer window/tail count differs')
        prepared[document['document_id']]=pairs;input_hashes[document['document_id']]=digest.hexdigest()
    if sum(map(len,prepared.values()))!=77:raise ValueError('Exact77 windows required')
    for variant,info in profiles.items():
        profile=info['metadata'];observed=[r['document_id'] for r in profile['samples']]
        if observed!=ids:raise ValueError('Final timing order differs from supplemental manifest')
        qualification={r['document_id']:r for r in profile['timing_input_qualification']}
        for document in documents:
            row=qualification[document['document_id']]
            if row['window_input_sha256']!=input_hashes[document['document_id']] or row['text_sha256']!=document['text_sha256']:
                raise ValueError('Device-only inputs differ from qualified final raw profile')
    result={'complete':False,'freeze_sha256':args.freeze_sha256,'profiler_source_sha256':sha(Path(__file__)),
        'timing_manifest_sha256':MANIFEST_SHA,'tokenizer_sha256':sha(tokenizer_path),'documents':18,'windows':77,
        'warmups_per_document':3,'repetitions_per_document':20,'batch_windows':1,'torch':torch.__version__,
        'cuda_device':torch.cuda.get_device_name(0),'routes':{},
        'boundary':'CUDA-event stream elapsed time bracketing one frozen B1/L256 graph replay. Static input copies and synchronization finish before the start event; output return, margin checking, tokenization, loading and capture are outside events. Document device time sums its serial window replay events. Event brackets can include GPU stream/launch gaps; this is not a profiler sum of individual kernel active cycles. GPU only; no FPGA/energy or complete input-to-decision claim.'}
    def check():
        if time.monotonic()-began>args.max_seconds:raise TimeoutError('Supplemental CUDA-event dispatch elapsed ceiling')
    routes_started=time.perf_counter()
    try:
        with torch.inference_mode():
            for variant in VARIANTS:
                check();load_start=time.perf_counter()
                model_args=SimpleNamespace(variant=variant,device='cuda',calibration=args.calibration,
                    export=args.export if variant.startswith('w8') else None)
                if target_descriptor(model_args)!=frozen['models'][variant]:raise ValueError('Frozen model/export descriptor differs')
                model=load_target(model_args);torch.cuda.synchronize();loaded=time.perf_counter()
                graph=GraphRunner(model,prepared[ids[0]][0],torch);captured=time.perf_counter()
                if graph.execution_identity!=frozen['execution_routes'][variant]:raise ValueError('Captured execution identity differs')
                start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True)
                reference={r['document_id']:r for r in profiles[variant]['metadata']['timing_input_qualification']}
                samples=[];max_margin_delta=0.
                for document in documents:
                    check();key=document['document_id'];pairs=prepared[key]
                    for _ in range(3):
                        for pair in pairs:graph.copy(pair);graph.replay()
                    torch.cuda.synchronize();repetitions=[]
                    for _ in range(20):
                        check();window_times=[]
                        for index,pair in enumerate(pairs):
                            graph.copy(pair);torch.cuda.synchronize() # outside CUDA events
                            start_event.record();logits=graph.replay();end_event.record();end_event.synchronize()
                            window_times.append(float(start_event.elapsed_time(end_event)))
                            margin=float((logits[0,1]-logits[0,0]).cpu()) # outside CUDA events
                            delta=abs(margin-reference[key]['execution_window_margins'][index]);max_margin_delta=max(max_margin_delta,delta)
                            if delta>1e-4:raise ValueError('Event-profile graph output differs from qualified execution')
                        repetitions.append({'window_cuda_event_ms':window_times,'document_cuda_event_ms':sum(window_times)})
                    values=[r['document_cuda_event_ms'] for r in repetitions]
                    samples.append({'document_id':key,'text_sha256':document['text_sha256'],'window_input_sha256':input_hashes[key],
                        'timing_only':document.get('timing_only',False),'windows':len(pairs),'mean_document_cuda_event_ms':float(np.mean(values)),
                        'p50_document_cuda_event_ms':float(np.quantile(values,.5)),'p95_document_cuda_event_ms':float(np.quantile(values,.95)),
                        'repetitions':repetitions})
                result['routes'][variant]={'target':frozen['models'][variant],'execution_identity':graph.execution_identity,
                    'qualified_final_profile_sha256':profiles[variant]['sha256'],'model_load_seconds':loaded-load_start,
                    'graph_setup_seconds':captured-loaded,'graph_reported_setup_seconds':graph.setup_seconds,
                    'max_qualified_margin_delta':max_margin_delta,'samples':samples,
                    'mean_document_cuda_event_ms':float(np.mean([r['mean_document_cuda_event_ms'] for r in samples]))}
                del graph,model,start_event,end_event;torch.cuda.empty_cache()
        result['complete']=True
    except BaseException as error:
        result['error']={'type':type(error).__name__,'message':str(error)}
        raise
    finally:
        result['elapsed_seconds']=time.monotonic()-began;result['validated_preparation_before_routes_seconds']=routes_started-process_started
        result['setup_boundary']='Model load and capture are separately timed in one process; OS/file/CUDA caches are not flushed. These are not fresh-process cold-boot measurements.'
        write(args.out,result)
    print(json.dumps({'complete':True,'out':str(args.out),'elapsed_seconds':result['elapsed_seconds'],
        'route_mean_document_cuda_event_ms':{k:v['mean_document_cuda_event_ms'] for k,v in result['routes'].items()}},indent=2))

if __name__=='__main__':main()
