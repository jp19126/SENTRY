"""Bounded raw-document B1 software timing, qualified against saved margins.

Warm input-to-decision includes tokenization, all windows, transfers and return.
Model/tokenizer setup and optional CUDA capture are separately reported.
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
from common import ROOT,read,rows,sha,write
from evaluate import load_target,load_prepared,target_descriptor,ensure_directory,score_role
from analyze import load_verified,window_bin,fit_threshold,metrics,roc_report

def graph_execution_identity(variant):
    source_hf=variant=='source'
    return {'backend':'cuda_graph','batch_windows':1,'window_length':256,
        'attention_mask_policy':'always additive4D constructed inside graph' if source_hf else 'readable model original additive4D',
        'operations':'embeddings→encoder→pooler→eval dropout→classifier' if source_hf else 'readable complete model.forward',
        'adapted_from_sha256':sha(ROOT/'scripts/r1_bert_cuda_graph.py') if source_hf else None,
        'distinct_from':'ordinary HF 2D-mask SDPA dispatch' if source_hf else 'eager execution scheduling',
        'profile_detector_source_sha256':sha(Path(__file__))}

class GraphRunner:
    def __init__(self,model,first,torch):
        self.model=model
        self.source_hf=hasattr(model,'bert') and hasattr(model.bert,'encoder')
        self.torch=torch; self.ids=first['input_ids'].to('cuda'); self.mask=first['attention_mask'].to('cuda')
        began=time.perf_counter(); stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): self.forward()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph): self.logits=self.forward()
        torch.cuda.synchronize();self.setup_seconds=time.perf_counter()-began
        self.execution_identity=graph_execution_identity('source' if self.source_hf else model.variant)
    def forward(self):
        if not self.source_hf:return self.model(input_ids=self.ids,attention_mask=self.mask).logits
        # Adapted from r1_bert_cuda_graph.explicit_forward. Avoid importing that
        # historical campaign module and its unrelated training/data dependencies.
        # The always-additive mask removes HF SDPA's tensor-to-Python checks.
        torch=self.torch;model=self.model
        embeddings=model.bert.embeddings(input_ids=self.ids,token_type_ids=torch.zeros_like(self.ids))
        expanded=self.mask[:,None,None,:].expand(1,1,256,256).to(embeddings.dtype)
        inverted=1.0-expanded
        additive=inverted.masked_fill(inverted.to(torch.bool),torch.finfo(embeddings.dtype).min)
        hidden=model.bert.encoder(embeddings,attention_mask=additive,head_mask=[None]*4,use_cache=False,
            output_attentions=False,output_hidden_states=False,return_dict=False)[0]
        pooled=model.bert.pooler(hidden)
        return model.classifier(model.dropout(pooled)).float()
    def copy(self,pair):self.ids.copy_(pair['input_ids']);self.mask.copy_(pair['attention_mask'])
    def replay(self):self.graph.replay();return self.logits
    def bringup(self,pairs,atol):
        records=[]
        for pair in pairs:
            self.copy(pair);captured=self.replay().clone();expected=self.forward()
            self.torch.cuda.synchronize()
            delta=float((captured-expected).abs().max())
            records.append({'valid_tokens':int(pair['attention_mask'].sum()),'max_absolute_logit_delta':delta,
                'exact':self.torch.equal(captured,expected),'passed':delta<=atol})
        return {'passed':all(r['passed'] for r in records),'cases':records,'engineering_logit_atol':atol,
                'boundary':'Graph replay after input replacement versus eager execution of the identical explicit graph arithmetic, separate from original HF 2D-mask comparison'}

def execution_comparison(reference,execution,thresholds,atol):
    """Report numerical/order/decision transfer separately from own calibration."""
    alphas=tuple(float(k) for k in thresholds)
    benign=lambda records:[r for r in records if r['label']==0]
    before=benign(reference['calibration']);after=benign(execution['calibration'])
    own={str(a):fit_threshold([r['margin'] for r in after],a) for a in alphas}
    order=lambda records:[r['document_id'] for r in sorted(records,key=lambda r:(r['margin'],r['document_id']))]
    old_order=order(before);new_order=order(after)
    comparison={'execution_thresholds':own,'reference_thresholds':thresholds,
        'calibration_benign_order_unchanged':old_order==new_order,
        'calibration_benign_changed_rank_positions':sum(a!=b for a,b in zip(old_order,new_order)),
        'reference_benign_order':old_order,'execution_benign_order':new_order,'roles':{}}
    qualified=old_order==new_order
    for role in reference:
        refs=reference[role];current=execution[role]
        if [r['document_id'] for r in refs]!=[r['document_id'] for r in current]:raise ValueError('Execution membership/order differs')
        maximum=max(abs(a-b) for ref,row in zip(refs,current) for a,b in zip(ref['window_margins'],row['window_margins']))
        qualified &= maximum<=atol
        detail={'documents':len(refs),'windows':sum(r['window_count'] for r in current),'max_window_margin_delta':maximum,'by_alpha':{}}
        for key,threshold in thresholds.items():
            changes=[];near=[];own_changes=[]
            for ref,row in zip(refs,current):
                delta=row['margin']-ref['margin'];record={'document_id':row['document_id'],'label':row['label'],
                    'reference_margin':ref['margin'],'execution_margin':row['margin'],'delta':delta,
                    'reference_distance_to_threshold':ref['margin']-threshold}
                if (row['margin']>threshold)!=(ref['margin']>threshold):changes.append(record)
                if (row['margin']>own[key])!=(ref['margin']>threshold):own_changes.append(record)
                if abs(ref['margin']-threshold)<=max(atol,abs(delta)):near.append(record)
            qualified &= not changes
            detail['by_alpha'][key]={'reference_fixed_threshold_metrics':metrics(current,threshold),'own_calibrated_metrics':metrics(current,own[key]),
                'fixed_threshold_disagreements':changes,'own_calibrated_disagreements':own_changes,'near_reference_threshold':near,
                'threshold_delta':own[key]-threshold}
        comparison['roles'][role]=detail
    comparison['reference_transfer_qualified']=bool(qualified)
    comparison['interpretation']='Transfer requires unchanged benign calibration rank, margin error within the engineering tolerance, and no decisions changed at any reference threshold on either role. Own calibration is always reported separately; label-informed ROC is descriptive only.'
    return comparison

def select(records,per_stratum,max_documents):
    strata={}
    for row in records:
        key=(row.get('genre','unknown'),row['label'],window_bin(row['window_count']))
        strata.setdefault(key,[]).append(row)
    chosen={}
    for key,group in sorted(strata.items()):
        for row in sorted(group,key=lambda r:hashlib.sha256(('bert_aie_v1|'+r['document_id']).encode()).hexdigest())[:per_stratum]:
            chosen[row['document_id']]=row
    # All models choose the same score-independent sample. Preserve shortest and
    # longest complete documents even when the budget caps the stratum coverage.
    ranked=sorted(records,key=lambda r:(r['window_count'],r['document_id']))
    ordered=[ranked[0],ranked[-1]]+[chosen[k] for k in sorted(chosen)]
    unique={}
    for row in ordered:
        unique.setdefault(row['document_id'],row)
        if len(unique)>=max_documents:break
    return list(unique.values()),{'|'.join(map(str,k)):len(v) for k,v in sorted(strata.items())}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',required=True,type=Path);p.add_argument('--raw',required=True,type=Path)
    p.add_argument('--timing-manifest',required=True,type=Path)
    p.add_argument('--timing-manifest-sha256')
    p.add_argument('--mode',choices=('qualify','profile'),default='profile')
    p.add_argument('--threshold-policy',choices=('reference','execution_calibrated'),default='reference')
    p.add_argument('--calibration',type=Path);p.add_argument('--export',type=Path)
    p.add_argument('--freeze',type=Path);p.add_argument('--freeze-sha256')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--backend',choices=('eager','cuda_graph'),default='eager')
    p.add_argument('--cpu-threads',type=int,default=1);p.add_argument('--repeats',type=int);p.add_argument('--warmups',type=int)
    p.add_argument('--max-seconds',type=int,default=600);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--logit-atol',type=float,default=1e-4)
    args=p.parse_args();start=time.monotonic()
    if not 1<=args.max_seconds<=600 or args.cpu_threads<1:
        raise ValueError('Invalid bounded profile settings')
    if args.backend=='cuda_graph' and args.device!='cuda':raise ValueError('CUDA graph needs CUDA')
    ensure_directory(args.out.parent)
    if args.out.exists():raise ValueError('Preserve prior profile; choose new output')
    if args.backend=='cuda_graph':
        for suffix in ('_graph_bringup.json','_execution_quality.json','_calibration_execution_scores.jsonl','_selection_execution_scores.jsonl'):
            if args.out.with_name(args.out.stem+suffix).exists():raise ValueError('Preserve prior graph attempt artifacts; choose new output name')
    # A changed execution route must be nominated before confirmation scores or
    # data are decoded, in addition to the underlying checkpoint/format freeze.
    evaluation_metadata=read(args.evaluation/'complete.json')
    if evaluation_metadata['stage']=='confirmation' and args.backend=='cuda_graph':
        if not args.freeze or not args.freeze_sha256 or sha(args.freeze)!=args.freeze_sha256:
            raise ValueError('Confirmation graph route needs exact prospective freeze')
        variant=evaluation_metadata['target']['variant']
        if read(args.freeze).get('execution_routes',{}).get(variant)!=graph_execution_identity(variant):
            raise ValueError('Distinct graph execution identity was not prospectively nominated')
    complete,scored=load_verified(args.evaluation,args.freeze,args.freeze_sha256)
    folder=Path(complete['data']['selection']['directory']); data=load_prepared(folder)
    timing=read(args.timing_manifest)
    embedded=timing.get('schema')=='kd_cnn_v2_development_timing_v1'
    timing_hash=sha(args.timing_manifest)
    if args.timing_manifest_sha256 and timing_hash!=args.timing_manifest_sha256:raise ValueError('Requested timing manifest hash differs')
    if embedded:
        expected_hash=args.timing_manifest_sha256 or '8d141711fcee46d65394c37460ef4bb377bd7346592ce11afa240ed6cf254162'
        if timing_hash!=expected_hash or not timing.get('immutable'):raise ValueError('Unapproved embedded development timing manifest')
        if timing['windowing']!={'length':256,'capacity':254,'overlap':64,'stride':190,'all_tails':True}:raise ValueError('Shared timing window policy differs')
        requested_ids=[r['document_id'] for r in timing['documents']]
        repetitions={'repeats':20,'warmups':3}
    else:
        if timing['source_manifest_sha256']!=sha(folder/'manifest.json') or timing['source_role']!=folder.name:
            raise ValueError('Shared timing manifest names different prepared source data')
        if timing['batch_windows']!=1:raise ValueError('Shared timing manifest needs batch one')
        requested_ids=timing['document_ids'];repetitions=timing
    if len(set(requested_ids))!=len(requested_ids):raise ValueError('Timing IDs must be unique')
    for key in ('repeats','warmups'):
        if getattr(args,key) is not None and getattr(args,key)!=repetitions[key]:raise ValueError('Shared timing repetitions/warmups cannot change')
        setattr(args,key,repetitions[key])
    if args.repeats<1 or args.warmups<0:raise ValueError('Invalid shared timing counts')
    raw_path=args.raw.resolve();raw_hash=sha(raw_path)
    expected_raw=Path(timing['raw_development_path']) if embedded else ROOT/timing['raw_source']
    if raw_path!=expected_raw.resolve():raise ValueError('Raw profiling path differs from shared timing manifest')
    if embedded and raw_hash!=timing['raw_development_sha256']:raise ValueError('Shared development timing raw source hash differs')
    if complete['stage']=='confirmation' and not embedded:
        frozen=read(args.freeze)['data']['selection']
        if raw_path.parent!=folder.resolve() or frozen['files'].get(raw_path.name)!=raw_hash:
            raise ValueError('Raw confirmation bytes must be inside the prospectively frozen role')
    else:
        lowered=str(raw_path).lower()
        if 'confirmation' in lowered or 'notinject' in lowered or 'final' in lowered:raise ValueError('Only development raw profiling input is permitted')
    raw={r['document_id']:r for r in rows(raw_path)}
    if embedded:
        if len(timing['documents'])!=18 or sum(bool(r.get('timing_only')) for r in timing['documents'])!=6:raise ValueError('Expected exact shared 12+6 timing membership')
        for row in timing['documents']:
            if hashlib.sha256(row['text'].encode()).hexdigest()!=row['text_sha256']:raise ValueError('Embedded timing text hash differs')
            if row.get('timing_only'):
                if row['label'] is not None or not row.get('not_quality_evaluation'):raise ValueError('Timing stress must have no quality label')
                origin=raw[row['origin_document_id']]
                if origin['label']!=0 or origin['text_sha256']!=row['origin_text_sha256']:raise ValueError('Timing stress origin differs')
                prefix=origin['text'][:row['source_prefix_end_char']]
                if '\n\n'.join([prefix]*row['whole_prefix_repeats'])!=row['text']:raise ValueError('Stress repetition construction differs')
            else:
                origin=raw[row['document_id']]
                if origin['text_sha256']!=row['text_sha256'] or origin['text']!=row['text']:raise ValueError('Embedded natural timing source differs')
        raw={r['document_id']:r for r in timing['documents']}
    else:
        if len(raw)!=len(data['documents']):raise ValueError('Raw/prepared membership count differs')
        for row in data['documents']:
            r=raw[row['document_id']]
            if r['text_sha256']!=row['text_sha256'] or hashlib.sha256(r['text'].encode()).hexdigest()!=row['text_sha256']:
                raise ValueError('Raw/prepared text differs')
    import torch
    torch.set_num_threads(args.cpu_threads);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.variant=complete['target']['variant']
    if target_descriptor(args)!=complete['target']:raise ValueError('Profile target/calibration/export differs from scored route')
    load_start=time.perf_counter();model=load_target(args)
    if args.device=='cuda':torch.cuda.synchronize()
    loaded=time.perf_counter()
    config=read(ROOT/'configs/project.json')
    if config['windowing']['primary_length']!=256 or config['windowing']['content_token_overlap']!=64:raise ValueError('Pinned window policy differs')
    from tokenizers import Tokenizer
    tokenizer_path=ROOT/'checkpoints/floating/epoch_3/tokenizer.json'
    tok=Tokenizer.from_file(str(tokenizer_path))
    if (tok.token_to_id('[CLS]'),tok.token_to_id('[SEP]'),tok.token_to_id('[PAD]'))!=(101,102,0):raise ValueError('Tokenizer special IDs differ')
    # The serialized tokenizer contains its normalizer, pretokenizer and model;
    # window assembly follows the project L256/content254/overlap64 contract.
    tok.no_truncation();tok.no_padding();ready=time.perf_counter()
    def prepare(document):
        content=tok.encode(document['text'],add_special_tokens=False).ids
        if embedded and len(content)!=document['content_tokens']:raise ValueError('Shared timing content token count differs')
        starts=list(range(0,max(1,len(content)),190));pairs=[]
        for offset in starts:
            values=[101]+content[offset:offset+254]+[102];count=len(values)
            pairs.append({'input_ids':torch.tensor([values+[0]*(256-count)],dtype=torch.long),
                          'attention_mask':torch.tensor([[1]*count+[0]*(256-count)],dtype=torch.long)})
            if offset+254>=len(content):break
        if embedded and len(pairs)!=document['windows']:raise ValueError('Shared timing window/tail count differs')
        return pairs
    def synchronize():
        if args.device=='cuda':torch.cuda.synchronize()
    deadline=start+args.max_seconds
    def check():
        if time.monotonic()>deadline:raise TimeoutError('Raw-document profile elapsed-time budget exhausted')
    score_index={r['document_id']:r for r in scored['selection']}
    if not requested_ids or (not embedded and any(k not in score_index for k in requested_ids)):raise ValueError('Shared timing document is absent from scored role')
    selected=[{**raw[k],'window_count':raw[k]['windows']} for k in requested_ids] if embedded else [score_index[k] for k in requested_ids]
    population={}
    for row in scored['selection']:
        key='|'.join(map(str,(row.get('genre','unknown'),row['label'],window_bin(row['window_count']))))
        population[key]=population.get(key,0)+1
    document_index={d['document_id']:i for i,d in enumerate(data['documents'])}
    graph=None;qualification=[];execution_scores=None;execution_quality=None;route_thresholds=complete['thresholds'];score_hashes={};bringup=None
    with torch.inference_mode():
        if args.backend=='cuda_graph':graph=GraphRunner(model,prepare(raw[selected[0]['document_id']])[0],torch)
        if graph:
            valid_counts=np.asarray(data['masks'].sum(axis=1))
            fixture_indices=list(dict.fromkeys([int(valid_counts.argmin()),int(valid_counts.argmax()),int(valid_counts.argmin())]))
            fixtures=[{'input_ids':torch.tensor(np.array(data['ids'][i:i+1]),dtype=torch.long),
                       'attention_mask':torch.tensor(np.array(data['masks'][i:i+1]),dtype=torch.long)} for i in fixture_indices]
            # Short→full→short exercises changed inputs; repeat rather than keep
            # fixture indices unique for this runtime replay check.
            fixtures=fixtures+[fixtures[0]]
            bringup=graph.bringup(fixtures,args.logit_atol)
            write(args.out.with_name(args.out.stem+'_graph_bringup.json'),bringup)
            if not bringup['passed']:raise ValueError('Graph replay differs from identical eager explicit arithmetic; bringup evidence preserved')
            class CapturedModel:
                def __call__(self,input_ids,attention_mask):
                    graph.copy({'input_ids':input_ids,'attention_mask':attention_mask})
                    return SimpleNamespace(logits=graph.replay())
            execution_scores={}
            for role in ('calibration','selection'):
                check();role_data=load_prepared(Path(complete['data'][role]['directory']))
                execution_scores[role]=score_role(CapturedModel(),role_data,args.device,1,deadline)
                destination=args.out.with_name(args.out.stem+'_'+role+'_execution_scores.jsonl')
                destination.write_text(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in execution_scores[role]),encoding='utf-8')
                score_hashes[role]={'name':destination.name,'sha256':sha(destination)}
            execution_quality=execution_comparison(scored,execution_scores,complete['thresholds'],args.logit_atol)
            execution_quality.update(execution_identity=graph.execution_identity,target=complete['target'],data=complete['data'],
                score_files=score_hashes,reference_evaluation_complete_sha256=sha(args.evaluation/'complete.json'),
                selection_roc=roc_report(execution_scores['selection']),graph_bringup=bringup,freeze_sha256=complete['freeze_sha256'],elapsed_seconds=time.monotonic()-start)
            write(args.out.with_name(args.out.stem+'_execution_quality.json'),execution_quality)
            if args.mode=='qualify':
                write(args.out,{'complete':True,'mode':'qualify','execution_quality':execution_quality,'elapsed_seconds':time.monotonic()-start})
                print(json.dumps({'complete':True,'mode':'qualify','reference_transfer_qualified':execution_quality['reference_transfer_qualified'],'out':str(args.out)}));return
            if args.threshold_policy=='reference' and not execution_quality['reference_transfer_qualified']:
                raise ValueError('Graph numerical/order/decision transfer failed; exact route scores and own-calibration comparison preserved. Explicit execution_calibrated policy required to time this distinct route.')
            if args.threshold_policy=='execution_calibrated':route_thresholds=execution_quality['execution_thresholds']
            score_index={r['document_id']:r for r in execution_scores['selection']}
            if not embedded:selected=[score_index[k] for k in requested_ids]
            qualification=[{'role':role,**detail} for role,detail in execution_quality['roles'].items()]
        if not graph and args.mode=='qualify':raise ValueError('Scoring-only qualification mode is for a distinct CUDA graph execution route')
        # Eager CPU/GPU verifies every window of the exact timing sample. Graph
        # routes already scored both complete calibration and selection roles.
        for row in ([] if graph or embedded else selected):
            check();i=document_index[row['document_id']];lo,hi=map(int,data['starts'][i:i+2]);observed=[]
            for at in range(lo,hi):
                pair={'input_ids':torch.tensor(np.array(data['ids'][at:at+1]),dtype=torch.long),'attention_mask':torch.tensor(np.array(data['masks'][at:at+1]),dtype=torch.long)}
                if graph:graph.copy(pair);logits=graph.replay()
                else:logits=model(**{k:v.to(args.device) for k,v in pair.items()}).logits
                observed.append(float((logits[0,1]-logits[0,0]).cpu()))
            delta=max(abs(a-b) for a,b in zip(observed,row['window_margins']))
            if delta>args.logit_atol or any((max(observed)>t)!=(row['margin']>t) for t in complete['thresholds'].values()):
                raise ValueError('Profile route fails saved margin/decision qualification: '+row['document_id'])
            qualification.append({'document_id':row['document_id'],'max_margin_delta':delta})
        def observe(document):
            check();began=time.perf_counter();pairs=prepare(document);prepared=time.perf_counter();transfer=network=returned=0.;margin=-float('inf')
            for pair in pairs:
                before=time.perf_counter()
                if graph:graph.copy(pair)
                else:pair={k:v.to(args.device) for k,v in pair.items()}
                synchronize();transferred=time.perf_counter()
                logits=graph.replay() if graph else model(**pair).logits
                synchronize();forwarded=time.perf_counter()
                margin=max(margin,float((logits[0,1]-logits[0,0]).cpu()));finished=time.perf_counter()
                transfer+=transferred-before;network+=forwarded-transferred;returned+=finished-forwarded
            decision=margin>route_thresholds['0.01'];end=time.perf_counter()
            return {'preprocessing_ms':1000*(prepared-began),'transfer_ms':1000*transfer,'network_ms':1000*network,
                    'return_aggregation_ms':1000*returned,'complete_ms':1000*(end-began),'windows':len(pairs),'margin':margin,'decision':decision}
        timing_qualification=[]
        if embedded:
            qualified=[]
            for row in selected:
                check();pairs=prepare(raw[row['document_id']]);eager_values=[];route_values=[];input_digest=hashlib.sha256()
                for pair in pairs:
                    for key in ('input_ids','attention_mask'):input_digest.update(pair[key].numpy().tobytes())
                    eager_logits=model(**{k:v.to(args.device) for k,v in pair.items()}).logits
                    eager_values.append(float((eager_logits[0,1]-eager_logits[0,0]).cpu()))
                    if graph:graph.copy(pair);route_logits=graph.replay();route_values.append(float((route_logits[0,1]-route_logits[0,0]).cpu()))
                    else:route_values.append(eager_values[-1])
                delta=max(abs(a-b) for a,b in zip(eager_values,route_values))
                timing_qualification.append({'document_id':row['document_id'],'text_sha256':row['text_sha256'],'timing_only':row.get('timing_only',False),
                    'windows':len(pairs),'window_input_sha256':input_digest.hexdigest(),'eager_window_margins':eager_values,'execution_window_margins':route_values,
                    'max_window_margin_delta':delta,'fixed_threshold_disagreements':{k:(max(eager_values)>t)!=(max(route_values)>t) for k,t in route_thresholds.items()}})
                if delta>args.logit_atol:raise ValueError('Timing execution differs from same-model eager margins beyond the engineering tolerance')
                qualified.append({**row,'window_margins':route_values,'margin':max(route_values)})
            selected=qualified
            write(args.out.with_name(args.out.stem+'_timing_input_qualification.json'),{'timing_manifest_sha256':timing_hash,'records':timing_qualification,
                'boundary':'All exact 18 timing inputs checked against same-model eager margins. Timing-only stress has no label and enters no confusion, ROC or threshold fitting.'})
        observations=[]
        for row in selected:
            document=raw[row['document_id']];pairs=prepare(document)
            if embedded:lo,hi=0,row['window_count']
            else:i=document_index[row['document_id']];lo,hi=map(int,data['starts'][i:i+2])
            if len(pairs)!=hi-lo:raise ValueError('Raw profile lost windows/tails')
            for at,pair in ([] if embedded else enumerate(pairs)):
                if not np.array_equal(pair['input_ids'].numpy()[0],data['ids'][lo+at]) or not np.array_equal(pair['attention_mask'].numpy()[0],data['masks'][lo+at]):
                    raise ValueError('Raw tokenizer/padding differs from saved scoring')
            for _ in range(args.warmups):observe(document)
            repeated=[]
            for _ in range(args.repeats):
                value=observe(document)
                if abs(value['margin']-row['margin'])>args.logit_atol or any((value['margin']>t)!=(row['margin']>t) for t in route_thresholds.values()):raise ValueError('Timed raw decision differs')
                repeated.append(value)
            observations.append({'document_id':row['document_id'],'text_sha256':row['text_sha256'],'genre':row.get('genre'),'label':row['label'],
                'windows':hi-lo,'timing_only':row.get('timing_only',False),'utf8_bytes':len(document['text'].encode()),'mean_ms':{k:float(np.mean([r[k] for r in repeated])) for k in ('preprocessing_ms','transfer_ms','network_ms','return_aggregation_ms','complete_ms')},
                'complete_p50_ms':float(np.quantile([r['complete_ms'] for r in repeated],.5)),'complete_p95_ms':float(np.quantile([r['complete_ms'] for r in repeated],.95)),
                'repetitions':repeated})
    result={'complete':True,'evaluation_complete_sha256':sha(args.evaluation/'complete.json'),'raw_sha256':raw_hash,'profile_source_sha256':sha(Path(__file__)),
            'target':complete['target'],'device':args.device,'backend':args.backend,'cpu_threads':args.cpu_threads,'torch':torch.__version__,
            'model_load_seconds':loaded-load_start,'tokenizer_load_seconds':ready-loaded,'tokenizer_sha256':sha(tokenizer_path),
            'attention_implementation':getattr(model.config,'_attn_implementation',None),'graph_setup_seconds':graph.setup_seconds if graph else None,
            'execution_identity':graph.execution_identity if graph else {'backend':'eager','variant':args.variant,'profile_detector_source_sha256':sha(Path(__file__))},
            'threshold_policy':args.threshold_policy,'thresholds':route_thresholds,'execution_quality':execution_quality,
            'qualification':{'scope':'all calibration and selection windows' if graph else 'all selected raw document windows','margin_atol':args.logit_atol,'records':qualification},
            'timing_manifest_sha256':timing_hash,'timing_manifest':timing,'timing_input_qualification':timing_qualification,
            'sample_policy':'Exact shared timing manifest document IDs/order; no independent sample cap',
            'population_strata':population,'samples':observations,'elapsed_seconds':time.monotonic()-start,
            'boundary':'Warm raw text to strict-threshold decision, serial batch-one windows; stage synchronization, transfers/static copies, margin return and all tails included. Model/tokenizer load and graph setup excluded from warm latency and reported separately. Sample is not workload-frequency weighted; no physical device/energy claim.'}
    write(args.out,result);print(json.dumps({'complete':True,'out':str(args.out),'documents':len(observations),'elapsed_seconds':result['elapsed_seconds']}))

if __name__=='__main__':main()
