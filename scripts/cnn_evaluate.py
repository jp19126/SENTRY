"""Frozen CNN confirmation, group uncertainty, FP32/fake-A8W8 scoring and B1 timing."""
from __future__ import annotations
import argparse, copy, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gate_eval import evaluate_documents, window_document
from cnn_study import DEST, Budget, read, write, sha, setup, data, model_for, score, metrics
from cnn_models import model_cost
from r1_challengers import tokenizer
from gate_eval import load_config
from cnn_artifacts import validate_prepared

def validate_freeze(freeze):
    if sha(DEST/'prepared/data_protocol.json')!=freeze['data_protocol_sha256'] or sha(ROOT/'scripts/cnn_models.py')!=freeze['model_source_sha256'] or sha(DEST/'prepared/diagnostics_raw.jsonl')!=freeze['diagnostics_sha256']:raise ValueError('Frozen model/data/probe identity changed')
    if any(sha(DEST/'prepared'/rep/'summary.json')!=h for rep,h in freeze['prepared_representation_sha256'].items()):raise ValueError('Frozen prepared representation changed')
    if sha(ROOT/'data/prepared/search_candidate_scoring.jsonl')!=freeze['raw_timing_source_sha256']:raise ValueError('Frozen raw timing documents changed')
    validate_preprocessing_policy()

def validate_preprocessing_policy(policy=None):
    policy=load_config() if policy is None else policy
    protocol=read(DEST/'prepared/data_protocol.json')
    if policy['models']['detector']!=protocol['tokenizer']:
        raise ValueError('Current tokenizer policy differs from prepared inputs')
    if (policy['windowing']['primary_length'],policy['windowing']['content_token_overlap'])!=(protocol['token']['length'],protocol['token']['overlap']):
        raise ValueError('Current token window policy differs from prepared inputs')
    return policy

def validate_saved(folder,existing,freeze_hash):
    if existing['freeze_sha256']!=freeze_hash:raise ValueError('Existing confirmation belongs to another freeze')
    done=read(DEST/'runs'/existing['run_id']/'complete.json')
    if sha(ROOT/done['checkpoint'])!=existing['checkpoint_sha256']:raise ValueError('Saved checkpoint changed')
    for variant,v in existing['variants'].items():
        for split,key in (('calibration','calibration_score_sha256'),('development','development_score_sha256'),('diagnostics','diagnostics_score_sha256')):
            if sha(folder/(variant+'_'+split+'.jsonl'))!=v[key]:raise ValueError('Saved confirmation scores changed')

def uncertainty(rows,t,seed=424242,repeats=1000):
    # Resample original source groups, retaining each clean source and its variants.
    groups={}
    for r in rows:
        g=r['source_group_id']; counts=groups.setdefault(g,np.zeros(4,dtype=np.int64))
        pred=r['risk_score']>t; lab=r['label']
        counts[0 if lab and pred else 1 if lab else 2 if pred else 3]+=1
    a=np.stack(list(groups.values())); rng=np.random.default_rng(seed); samples=[]
    for _ in range(repeats):
        tp,fn,fp,tn=a[rng.integers(0,len(a),len(a))].sum(0)
        samples.append([tp/(tp+fn),fp/(fp+tn)])
    q=np.quantile(samples,[.025,.975],axis=0)
    def wilson(success,n):
        z=1.959963984540054;ph=success/n;den=1+z*z/n
        middle=(ph+z*z/(2*n))/den;half=z*((ph*(1-ph)/n+z*z/(4*n*n))**.5)/den
        return [max(0.,middle-half),min(1.,middle+half)]
    benign=a[a[:,2:4].sum(1)>0];attack=a[a[:,:2].sum(1)>0]
    return {'kind':'source-group bootstrap percentile','groups':len(a),'repeats':repeats,'seed':seed,
            'recall_95':[float(q[0,0]),float(q[1,0])],'fpr_95':[float(q[0,1]),float(q[1,1])],
            'conditional_on_fitted_threshold':True,'calibration_uncertainty_included':False,
            'group_any_false_positive_wilson_95':wilson(int((benign[:,2]>0).sum()),len(benign)),
            'group_any_missed_attack_wilson_95':wilson(int((attack[:,1]>0).sum()),len(attack)),
            'group_event_note':'Wilson source-group event intervals supplement degenerate all-zero bootstrap tails; benign events equal document FPR only with one benign per group. Any attack miss is a group event, not document recall. Independence assumes the documented source-group unit; template/semantic dependence is unresolved.'}

def strata(rows,t):
    result={}
    keys=('source_dataset','attack_type','attack_template_id','injection_position','probe_category','probe_topic')
    for key in keys:
        for value in sorted(set(str(r.get(key)) for r in rows)):
            group=[r for r in rows if str(r.get(key))==value]
            result[key+':'+value]=evaluate_documents([r['risk_score'] for r in group],[r['label'] for r in group],t)
    for lo,hi in ((1,2),(2,4),(4,8),(8,10**9)):
        group=[r for r in rows if lo<=len(r['window_scores'])<hi]
        result[f'windows:{lo}-{hi-1}']=evaluate_documents([r['risk_score'] for r in group],[r['label'] for r in group],t)
    for lo,hi in ((0,128),(128,256),(256,512),(512,10**9)):
        group=[r for r in rows if lo<=r.get('content_token_count',r.get('content_length',0))<hi]
        result[f'content_units:{lo}-{hi-1}']=evaluate_documents([r['risk_score'] for r in group],[r['label'] for r in group],t)
    return result

def quantized(model,fit,budget):
    """Static symmetric fake W8/input-A8; FP32 norm, gates, bias and sums.

    Per-output-channel conv/linear and per-row embedding W8, scale=maxabs/127.
    A8 per-tensor input maxabs calibrated on first 512 canonical-fit windows.
    Emulated FP32 convolution accumulates rounded inputs; not native INT8 timing
    or guaranteed INT32 accumulation order. No training or evaluation-label use.
    """
    q=copy.deepcopy(model); maxima={}; handles=[]
    for name,m in q.named_modules():
        if isinstance(m,(torch.nn.Conv1d,torch.nn.Linear)):
            def observe(module,inputs,key=name):
                maxima[key]=max(maxima.get(key,0.),float(inputs[0].detach().abs().amax()))
            handles.append(m.register_forward_pre_hook(observe))
    q.eval()
    with torch.inference_mode():
        for at in range(0,min(512,len(fit['x'])),64):
            budget.check(); q(fit['x'][at:at+64],fit['mask'][at:at+64])
    for h in handles:h.remove()
    formats={};weight_elements=weight_scale_elements=activation_scale_elements=0
    with torch.no_grad():
        for name,m in q.named_modules():
            if isinstance(m,(torch.nn.Conv1d,torch.nn.Linear,torch.nn.Embedding)):
                axes=tuple(range(1,m.weight.ndim)); ws=m.weight.abs().amax(dim=axes,keepdim=True).clamp_min(1e-12)/127
                m.weight.copy_((m.weight/ws).round().clamp(-127,127)*ws)
                formats[name]={'weight_scale_min':float(ws.min()),'weight_scale_max':float(ws.max())}
                weight_elements+=m.weight.numel();weight_scale_elements+=ws.numel()
            if isinstance(m,(torch.nn.Conv1d,torch.nn.Linear)):
                scale=max(maxima[name]/127,1e-12); formats[name]['activation_scale']=scale
                activation_scale_elements+=1
                def rounding(module,inputs,s=scale):
                    return ((inputs[0]/s).round().clamp(-127,127)*s,)+inputs[1:]
                m.register_forward_pre_hook(rounding)
    remaining=sum(p.numel() for p in q.parameters())-weight_elements
    formats['_analytical_storage']={'int8_weight_elements':weight_elements,
        'fp32_remaining_parameter_elements':remaining,'fp32_weight_scale_elements':weight_scale_elements,
        'fp32_activation_scale_elements':activation_scale_elements,
        'mixed_parameter_scale_payload_bytes':weight_elements+4*(remaining+weight_scale_elements+activation_scale_elements),
        'boundary':'Ideal packed W8 weights with FP32 remaining parameters/scales; excludes alignment, activation buffers and instructions. Fake-quantization tensors actually remain FP32; this is not measured native INT8 memory.'}
    return q,formats

def timing(model,c,threshold,budget,count=12,repeats=20):
    source=ROOT/'data/prepared/search_candidate_scoring.jsonl'
    docs=[json.loads(s) for s in source.read_text(encoding='utf-8').splitlines() if s.strip()]
    clean=[d for d in docs if d['label']==0]
    clean.sort(key=lambda d:(len(d['text'].encode('utf-8')),d['document_id']))
    selected=clean[:count//2]+clean[-count//2:]
    policy=validate_preprocessing_policy()
    tok=tokenizer(policy) if c['representation']=='token' else None
    def prepare(d):
        if tok:
            windows=window_document(tok,d['text'],d['document_id'],config=policy)
            pairs=[(w['input_ids'],w['attention_mask']) for w in windows]; length=256
        else:
            values=[b+1 for b in d['text'].encode('utf-8')]; length=1024; pairs=[]; at=0
            while True:
                end=min(at+1024,len(values)); pairs.append((values[at:end],[1]*(end-at)))
                if end==len(values):break
                at=end-256
        return [(torch.tensor([ids+[0]*(length-len(ids))]),torch.tensor([mask+[0]*(length-len(mask))])) for ids,mask in pairs]
    output=[]; model.eval()
    with torch.inference_mode():
        for d in selected:
            warm=prepare(d)
            for _ in range(5):
                for ids,mask in warm:model(ids.cuda(),mask.cuda())
            torch.cuda.synchronize(); vals=[]; repetition_risks=[]; repetition_decisions=[]
            for _ in range(repeats):
                budget.check(); t0=time.perf_counter(); pairs=prepare(d); t1=time.perf_counter(); risk=0.; h2d=network=ret=0.
                for ids,mask in pairs:
                    a=time.perf_counter(); ids=ids.cuda(); mask=mask.cuda(); torch.cuda.synchronize(); b=time.perf_counter()
                    out=model(ids,mask).logits; value=torch.softmax(out,1)[0,1]; torch.cuda.synchronize(); cc=time.perf_counter()
                    risk=max(risk,float(value)); e=time.perf_counter(); h2d+=b-a;network+=cc-b;ret+=e-cc
                decision=risk>threshold; t2=time.perf_counter()
                vals.append([t1-t0,h2d,network,ret,t2-t0])
                repetition_risks.append(risk); repetition_decisions.append(decision)
            a=np.asarray(vals)*1000
            output.append({'document_id':d['document_id'],'utf8_bytes':len(d['text'].encode('utf-8')),'windows':len(warm),
                'repeats':repeats,'warmups':5,'mean_ms':dict(zip(('preprocessing','h2d','network','return_aggregation','complete'),a.mean(0).tolist())),
                'complete_p50_ms':float(np.quantile(a[:,4],.5)),'complete_p95_ms':float(np.quantile(a[:,4],.95)),
                'repetition_ms':[dict(zip(('preprocessing','h2d','network','return_aggregation','complete'),row.tolist())) for row in a],
                'repetition_risk_scores':repetition_risks,'repetition_decisions':repetition_decisions,
                'observed_decision':decision,'observed_risk_score':risk})
    return {'backend':'CUDA eager native FP32; TF32 disabled; B1 windows serial','samples':output,
        'boundary':'raw complete document to strict threshold decision; CPU tensors plus explicit transfer synchronization',
        'cold_process_startup_excluded':True,'cold_model_loading_measured_separately':True}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',required=True);p.add_argument('--max-seconds',type=int,default=10800);p.add_argument('--run-id')
    args=p.parse_args();setup(42);budget=Budget(args.max_seconds);freeze=read(Path(args.freeze));cache={};results=[]
    validate_freeze(freeze)
    selected=freeze['runs'] if not args.run_id else [r for r in freeze['runs'] if r['run_id']==args.run_id]
    if not selected:raise ValueError('Run ID is absent from the actual freeze')
    for rep in {freeze['finalist_configs'][s['config_id']]['representation'] for s in selected}:
        validate_prepared(rep,('full_train','search_temporary_threshold','search_candidate_scoring','diagnostics'))
    for spec in selected:
        if int(os.environ.get('TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT','64'))!=freeze['cudnn_cache_limits'][spec['config_id']] or int(os.environ.get('TORCH_CUDNN_V8_API_DISABLED','0'))!=freeze['cudnn_v8_disabled'][spec['config_id']]:raise ValueError('Scoring CUDA runtime options differ from freeze; dispatch through the serial confirmation queue')
        folder=DEST/'confirmation'/spec['run_id']; folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists():
            existing=read(folder/'complete.json')
            validate_saved(folder,existing,sha(Path(args.freeze)))
            results.append(existing);continue
        run=DEST/'runs'/spec['run_id']; meta=read(run/'run.json');done=read(run/'complete.json'); c=meta['config']
        if meta['spec']!=spec:raise ValueError('Full training recipe differs from frozen specification')
        if int(meta['cudnn_plan_cache_limit'])!=freeze['cudnn_cache_limits'][spec['config_id']] or int(meta.get('cudnn_v8_api_disabled','0'))!=freeze['cudnn_v8_disabled'][spec['config_id']]:raise ValueError('Fitted CUDA runtime options differ from freeze')
        if c!=freeze['finalist_configs'][spec['config_id']]:raise ValueError('Frozen finalist configuration changed')
        if done['epochs']!=spec['epochs'] or done['seed']!=spec['seed'] or c['id']!=spec['config_id'] or meta['spec'].get('fit_split')!='full_train' or meta['spec'].get('select_epoch',False):raise ValueError('Frozen run mismatch')
        if sha(ROOT/done['checkpoint'])!=done['checkpoint_sha256']:raise ValueError('Frozen checkpoint identity changed')
        load_start=time.perf_counter();model=model_for(c);model.load_state_dict(torch.load(ROOT/done['checkpoint'],map_location='cpu',weights_only=False)['state_dict']);model.eval()
        torch.cuda.synchronize();cold=time.perf_counter()-load_start
        rep=c['representation']
        for split in ('search_temporary_threshold','search_candidate_scoring','diagnostics','full_train'):
            if (rep,split) not in cache:cache[(rep,split)]=data(rep,split)
        scored={};qformats=None; variants=['fp32']+(['fake_w8a8'] if c['id'] in freeze['quantization_config_ids'] else [])
        for variant in variants:
            active=model
            if variant=='fake_w8a8':active,qformats=quantized(model,cache[(rep,'full_train')],budget)
            cal=score(active,cache[(rep,'search_temporary_threshold')],budget,batch=1);sel=score(active,cache[(rep,'search_candidate_scoring')],budget,batch=1);diag=score(active,cache[(rep,'diagnostics')],budget,batch=1)
            m=metrics(cal,sel);t=m['0.01']['threshold']; dm=evaluate_documents([r['risk_score'] for r in diag],[r['label'] for r in diag],t)
            for split,rows in [('calibration',cal),('development',sel),('diagnostics',diag)]:
                with (folder/(variant+'_'+split+'.jsonl')).open('w',encoding='utf-8') as f:
                    for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            scored[variant]={'metrics':m,'diagnostics':dm,'strata':strata(sel,t),'diagnostic_strata':strata(diag,t),
                'uncertainty':uncertainty(sel,t),'calibration_score_sha256':sha(folder/(variant+'_calibration.jsonl')),
                'development_score_sha256':sha(folder/(variant+'_development.jsonl')),'diagnostics_score_sha256':sha(folder/(variant+'_diagnostics.jsonl')),
                'scoring_batch_windows':1}
            if variant=='fake_w8a8':
                floating=[json.loads(s) for s in (folder/'fp32_development.jsonl').read_text(encoding='utf-8').splitlines()]
                ft=scored['fp32']['metrics']['0.01']['threshold']
                changed=[r['document_id'] for r,f in zip(sel,floating) if (r['risk_score']>t)!=(f['risk_score']>ft)]
                scored[variant]['changed_decision_ids']=changed;scored[variant]['formats']=qformats
                scored[variant]['note']='fake quantization arithmetic sensitivity; FP32 accumulations/norms/nonlinear/bias; no native INT8 timing'
                del active
        timed=timing(model,c,scored['fp32']['metrics']['0.01']['threshold'],budget) if spec.get('time',False) else None
        if timed:
            expected={r['document_id']:r for r in sel} if variants==['fp32'] else {r['document_id']:r for r in [json.loads(s) for s in (folder/'fp32_development.jsonl').read_text(encoding='utf-8').splitlines()]}
            for sample in timed['samples']:
                row=expected[sample['document_id']];sample['prepared_quality_batch1_risk_score']=row['risk_score']
                sample['batch_one_score_delta']=sample['observed_risk_score']-row['risk_score']
                sample['batch_one_decision_changed']=sample['observed_decision']!=(row['risk_score']>scored['fp32']['metrics']['0.01']['threshold'])
        result={'complete':True,'run_id':spec['run_id'],'config':c,'seed':spec['seed'],'checkpoint_sha256':done['checkpoint_sha256'],
            'freeze_sha256':sha(Path(args.freeze)),'cold_model_load_seconds':cold,'variants':scored,
            'evaluator_source_sha256':sha(Path(__file__)),'training_runner_source_sha256':meta['runner_source_sha256'],
            'environment':{'python':sys.version,'torch':torch.__version__,'device':torch.cuda.get_device_name(),
                'arithmetic':'native CUDA FP32 with TF32 disabled; fake W8/input-A8 is emulated FP32'},
            'cudnn_plan_cache_limit':os.environ.get('TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT'),
            'cudnn_v8_api_disabled':os.environ.get('TORCH_CUDNN_V8_API_DISABLED','0'),
            'cost':model_cost(model,1024 if rep=='byte' else 256),
            'timing':timed}
        write(folder/'complete.json',result);results.append(result);write(DEST/'confirmation_table.json',results)
        print(json.dumps({'event':'confirmation','run':spec['run_id'],'metrics':scored['fp32']['metrics']['0.01'],'diagnostics':scored['fp32']['diagnostics']}),flush=True)
        del model
    dispatch_name='confirmation_dispatch.json' if not args.run_id else 'confirmation_dispatch_'+args.run_id+'.json'
    write(DEST/dispatch_name,{'wall_seconds':time.monotonic()-budget.started,'command':sys.argv,'freeze_sha256':sha(Path(args.freeze)), 'complete_count':len(results),'memory_observations':budget.observations})
if __name__=='__main__':main()
