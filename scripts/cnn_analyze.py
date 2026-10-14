"""Reconcile actual CNN records and evidence boundaries; no model fitting."""
from __future__ import annotations
import csv,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cnn_study import DEST,read,write
from cnn_evaluate import strata,uncertainty

def rows(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]

def nondominated(records,keys):
    result=[]
    for r in records:
        v=[f(r) for f in keys]
        dominated=False
        for other in records:
            if other is r:continue
            u=[f(other) for f in keys]
            if all(a<=b for a,b in zip(u,v)) and any(a<b for a,b in zip(u,v)):dominated=True;break
        if not dominated:result.append(r['run_id'])
    return result

def main():
    output={'evidence_boundary':'Development detector labels; final partitions/target calls/physical FPGA remain unexecuted'}
    allruns=[]
    for stage in ('stage_a','stage_b','stage_c','ablations'):
        path=DEST/(stage+'_run_table.json')
        if not path.exists():continue
        a=read(path);output[stage]={'attempted':len(a),'complete':sum(r['complete'] for r in a),'families':{}}
        for r in a:
            allruns.append({**r,'stage':stage})
        families=defaultdict(list)
        for r in a:
            if r['complete']:families[r['family']].append(r)
        for f,rs in families.items():
            if stage=='stage_c':
                output[stage]['families'][f]={'completed_frozen_fits':len(rs),'quality_evidence':'See separate confirmation; no in-fit monitoring is treated as held-out quality'}
                continue
            output[stage]['families'][f]={'configs':len(rs),'empirical_inner_passes':sum(r['metrics']['0.01']['passes'] for r in rs),
                'recall_range':[min(r['metrics']['0.01']['recall'] for r in rs),max(r['metrics']['0.01']['recall'] for r in rs)],
                'fpr_range':[min(r['metrics']['0.01']['fpr'] for r in rs),max(r['metrics']['0.01']['fpr'] for r in rs)]}
        if stage in ('stage_b','ablations'):
            output[stage]['strata']={}
            output[stage]['learning_trajectories']={}
            for r in a:
                if not r['complete']:continue
                saved=rows(DEST/'runs'/r['run_id']/'selection_scores.jsonl');t=r['metrics']['0.01']['threshold']
                output[stage]['strata'][r['run_id']]={'strata':strata(saved,t),'uncertainty':uncertainty(saved,t)}
                history=read(DEST/'runs'/r['run_id']/'history.json')
                points=[{'epoch':h['epoch'],'loss':h['loss'],**{k:h['selection_metrics']['0.01'][k] for k in ('tp','fn','fp','tn','recall','fpr','passes')}} for h in history]
                output[stage]['learning_trajectories'][r['run_id']]={'points':points,'selected_epoch':r['selected_epoch'],
                    'first':points[0],'final':points[-1],
                    'passing_epochs':[x['epoch'] for x in points if x['passes']],
                    'boundary':'Repeated inner development observations used for selection; epoch changes do not isolate architecture or establish independent improvement'}
    failures=[];failure_identities=set()
    for path in sorted((DEST/'runs').glob('*/failure*.json')):
        failure=read(path)
        identity=(failure.get('run_id'),failure.get('error'),failure.get('traceback'))
        if identity in failure_identities:continue
        failure_identities.add(identity)
        meta_path=ROOT/failure['attempt_metadata_path'] if failure.get('attempt_metadata_path') else path.with_name('run.json')
        meta=read(meta_path) if meta_path.exists() else {}
        failure.update(stage='failed_attempt',config_id=meta.get('config',{}).get('id'),family=meta.get('config',{}).get('family'),
            seed=meta.get('spec',{}).get('seed'),failure_record=str(path.relative_to(ROOT)))
        failures.append(failure)
    output['preserved_failed_attempts']=failures
    fields=('stage','run_id','complete','config_id','family','representation','seed','epochs','selected_epoch','tp','fn','fp','tn','recall','fpr','passes','parameters','training_scoring_seconds','error')
    with (DEST/'run_table.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in allruns+failures:
            values={k:r.get(k) for k in fields};values.update({k:r.get('metrics',{}).get('0.01',{}).get(k) for k in ('tp','fn','fp','tn','recall','fpr','passes')});w.writerow(values)
    # Count each checkpoint-bearing run once; stage-A timing reuse is not extra fitting.
    unique={r['run_id']:r for r in allruns}
    output['executed_training_scoring_seconds_completed_runs']=sum(r.get('training_scoring_seconds',0) for r in unique.values())
    output['complete_learning_runs']=sum(r['complete'] for r in unique.values())
    output['dispatches']={}
    for path in DEST.glob('*_dispatch.json'):
        output['dispatches'][path.stem]=read(path)
    confirmation=DEST/'confirmation_table.json'
    if confirmation.exists():
        confirmed=read(confirmation);byconfig=defaultdict(list)
        for r in confirmed:byconfig[r['config']['id']].append(r)
        summary={}; error_inspection={}
        for cid,rs in byconfig.items():
            ms=[r['variants']['fp32']['metrics']['0.01'] for r in rs]
            qs=[r['variants'].get('fake_w8a8') for r in rs]
            manifest=read(DEST/'prepared'/rs[0]['config']['representation']/'search_candidate_scoring'/'manifest.json')
            prepared=rows(DEST/'prepared'/rs[0]['config']['representation']/'search_candidate_scoring'/'documents.jsonl')
            window_counts=np.asarray([r['window_count'] for r in prepared])
            document_macs=window_counts*rs[0]['cost']['conv_linear_macs']
            summary[cid]={'family':rs[0]['config']['family'],'seeds':[r['seed'] for r in rs],'primary':ms,'passes_seeds':sum(m['passes'] for m in ms),
                'recall_range':[min(m['recall'] for m in ms),max(m['recall'] for m in ms)],'fpr_range':[min(m['fpr'] for m in ms),max(m['fpr'] for m in ms)],
                'diagnostics':[r['variants']['fp32']['diagnostics'] for r in rs],
                'quantization':[q['metrics']['0.01'] if q else None for q in qs],
                'cost':rs[0]['cost'],'complete_document_cost':{'documents':manifest['documents'],'windows':manifest['windows'],
                    'mean_conv_linear_macs':float(document_macs.mean()),'p50_conv_linear_macs':float(np.quantile(document_macs,.5)),
                    'p95_conv_linear_macs':float(np.quantile(document_macs,.95)),'max_conv_linear_macs':int(document_macs.max()),
                    'boundary':'Complete identical development raw documents, actual representation-specific padded window chains; excludes normalization/nonlinear/pooling/transport'},
                'timing':[r['timing'] for r in rs if r['timing']],
                'bootstrap':[r['variants']['fp32']['uncertainty'] for r in rs]}
            for r,m in zip(rs,ms):
                folder=DEST/'confirmation'/r['run_id'];a=rows(folder/'fp32_development.jsonl'); t=m['threshold']
                fp=[x for x in a if x['label']==0 and x['risk_score']>t];fn=[x for x in a if x['label']==1 and x['risk_score']<=t]
                details={'false_positive_ids':[x['document_id'] for x in fp],'false_negative_ids':[x['document_id'] for x in fn],
                    'false_positive_window_counts':dict(Counter(len(x['window_scores']) for x in fp)),
                    'missed_attack_types':dict(Counter(x.get('attack_type') for x in fn)),
                    'benign_tail':[{k:x.get(k) for k in ('document_id','risk_score','source_group_id','source_dataset','window_count','content_length')} for x in sorted([x for x in a if x['label']==0],key=lambda x:x['risk_score'],reverse=True)[:40]]}
                q=r['variants'].get('fake_w8a8')
                if q:
                    b=rows(folder/'fake_w8a8_development.jsonl'); qt=q['metrics']['0.01']['threshold']; changes=[]
                    floating_by_id={x['document_id']:x for x in a}
                    details['quantized_benign_tail']=[{**{k:y.get(k) for k in ('document_id','risk_score','source_group_id','source_dataset','window_count','content_length')},
                        'own_quantized_threshold':qt,'own_threshold_margin':y['risk_score']-qt,
                        'fp32_score':floating_by_id[y['document_id']]['risk_score'],'fp32_threshold':t}
                        for y in sorted([x for x in b if x['label']==0],key=lambda x:x['risk_score'],reverse=True)[:40]]
                    for x,y in zip(a,b):
                        before=x['risk_score']>t;after=y['risk_score']>qt
                        if before!=after:changes.append({'document_id':x['document_id'],'label':x['label'],'fp32_score':x['risk_score'],'quantized_score':y['risk_score'],'before':before,'after':after})
                    details['quantized_changes']=changes
                error_inspection[r['run_id']]=details
        output['confirmation']=summary
        confirmed_fields=('config_id','family','representation','seed','variant','cudnn_v8_api_disabled','cudnn_plan_cache_limit','threshold','tp','fn','fp','tn','recall','fpr','passes',
            'parameter_bytes_fp32','ideal_mixed_quantized_parameter_scale_bytes','mean_complete_document_conv_linear_macs',
            'measured_native_fp32_complete_ms_extreme_sample_mean','measured_native_fp32_timing_document_count')
        with (DEST/'confirmation_table.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=confirmed_fields);writer.writeheader()
            for r in confirmed:
                c=r['config'];tm=r['timing']['samples'] if r['timing'] else []
                for variant,v in r['variants'].items():
                    primary=v['metrics']['0.01'];storage=v.get('formats',{}).get('_analytical_storage',{})
                    writer.writerow({'config_id':c['id'],'family':c['family'],'representation':c['representation'],'seed':r['seed'],'variant':variant,
                        'cudnn_v8_api_disabled':r.get('cudnn_v8_api_disabled'),'cudnn_plan_cache_limit':r.get('cudnn_plan_cache_limit'),
                        **{key:primary[key] for key in ('threshold','tp','fn','fp','tn','recall','fpr','passes')},
                        'parameter_bytes_fp32':r['cost']['parameter_bytes_fp32'],
                        'ideal_mixed_quantized_parameter_scale_bytes':storage.get('mixed_parameter_scale_payload_bytes'),
                        'mean_complete_document_conv_linear_macs':summary[c['id']]['complete_document_cost']['mean_conv_linear_macs'],
                        'measured_native_fp32_complete_ms_extreme_sample_mean':float(np.mean([x['mean_ms']['complete'] for x in tm])) if tm and variant=='fp32' else None,
                        'measured_native_fp32_timing_document_count':len(tm) if variant=='fp32' else None})
        architecture_rows=[]
        for cid,s in summary.items():
            tm=s['timing'][0]['samples'] if s['timing'] else []
            architecture_rows.append({'run_id':cid,'recall':s['recall_range'][0],'fpr':s['fpr_range'][1],
                'parameters':s['cost']['parameters'],'macs':s['complete_document_cost']['mean_conv_linear_macs'],
                'three_seed_complete':len(s['seeds'])==3,'all_seeds_pass':len(s['seeds'])==3 and s['passes_seeds']==3,
                'worst_seed_excess_benign_rejections':max(0,max(m['fp']-math.floor(.01*m['benign']) for m in s['primary'])),
                'worst_seed_excess_attack_misses':max(0,max(m['fn']-math.floor(.01*m['attacks']) for m in s['primary'])),
                'mean_measured_complete_ms':float(np.mean([r['mean_ms']['complete'] for r in tm])) if tm else None})
        output['architecture_worst_seed_frontier']=nondominated(architecture_rows,(lambda r:1-r['recall'],lambda r:r['fpr'],lambda r:r['parameters'],lambda r:r['macs']))
        output['architecture_all_seed_feasible_frontier']=nondominated([r for r in architecture_rows if r['all_seeds_pass']],(lambda r:r['parameters'],lambda r:r['macs'],lambda r:r['fpr'],lambda r:1-r['recall']))
        output['architecture_summary']=architecture_rows
        output['nonpassing_boundary_distances']=[{k:r[k] for k in ('run_id','recall','fpr','worst_seed_excess_benign_rejections','worst_seed_excess_attack_misses')} for r in architecture_rows if r['three_seed_complete'] and not r['all_seeds_pass']]
        output['boundary_distance_note']='Separate counts beyond the unchanged 99%-recall/1%-FPR empirical limits; no weighted score or new passing/near-miss tolerance is introduced.'
        timed_architectures=[r for r in architecture_rows if r['three_seed_complete'] and r['mean_measured_complete_ms'] is not None]
        output['quality_complete_latency_storage_frontier']=nondominated(timed_architectures,(lambda r:1-r['recall'],lambda r:r['fpr'],lambda r:r['parameters'],lambda r:r['mean_measured_complete_ms']))
        output['feasible_complete_latency_storage_frontier']=nondominated([r for r in timed_architectures if r['all_seeds_pass']],(lambda r:r['parameters'],lambda r:r['fpr'],lambda r:1-r['recall'],lambda r:r['mean_measured_complete_ms']))
        flat=[]
        for r in confirmed:
            m=r['variants']['fp32']['metrics']['0.01'];flat.append({'run_id':r['run_id'],'recall':m['recall'],'fpr':m['fpr'],'parameters':r['cost']['parameters'],'macs':summary[r['config']['id']]['complete_document_cost']['mean_conv_linear_macs'],'passes':m['passes']})
        output['quality_storage_macs_frontier']=nondominated(flat,(lambda r:1-r['recall'],lambda r:r['fpr'],lambda r:r['parameters'],lambda r:r['macs']))
        output['passing_frontier']=nondominated([r for r in flat if r['passes']],(lambda r:r['parameters'],lambda r:r['macs'],lambda r:r['fpr'],lambda r:1-r['recall']))
        write(DEST/'error_inspection.json',error_inspection)
    write(DEST/'analysis.json',output)
    print(json.dumps({'learning_runs':output['complete_learning_runs'],'completed_run_seconds':output['executed_training_scoring_seconds_completed_runs'],'confirmed_configs':list(output.get('confirmation',{}))}))
if __name__=='__main__':main()
