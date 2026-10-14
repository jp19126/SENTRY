"""Independent saved-score calibration/aggregation/checkpoint checks; no inference."""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gate_eval import fit_threshold,evaluate_documents
DEST=ROOT/'results/cnn_study_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def score_check(a,expected):
    assert len(a)==len(expected)
    for actual,want in zip(a,expected):
        for key in ('document_id','source_group_id','label','text_sha256'):assert actual[key]==want[key],key
        values=actual['window_scores'];assert len(values)==want['window_count'];assert all(math.isfinite(x) and 0<=x<=1 for x in values)
        assert actual['risk_score']==max(values)
def metric_check(cal,sel,saved):
    for alpha in (.005,.01,.02):
        t=fit_threshold([r['risk_score'] for r in cal if r['label']==0],alpha)
        m=evaluate_documents([r['risk_score'] for r in sel],[r['label'] for r in sel],t);s=saved[str(alpha)]
        for key in m:assert m[key]==s[key],(alpha,key)
        assert s['passes']==(m['recall']>=.99 and m['fpr']<=.01)
def main():
    started=time.perf_counter()
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('stage_a','stage_b','stage_c','ablations','confirmation'));a=p.parse_args()
    records=read(DEST/(a.stage+'_table.json' if a.stage=='confirmation' else a.stage+'_run_table.json'));checks=[];cache={}
    frozen=read(ROOT/'configs/cnn_stage_c_v1.json') if a.stage in ('stage_c','confirmation') else None
    if frozen:
        expected={s['run_id']:s for s in frozen['runs']}
        assert len(records)==len(expected) and {r['run_id'] for r in records}==set(expected)
        assert all(r['complete'] for r in records)
    for r in records:
        if not r['complete']:continue
        run=DEST/'runs'/r['run_id'];meta=read(run/'run.json');c=meta['config'];rep=c['representation'];done=read(run/'complete.json')
        assert sha(ROOT/done['checkpoint'])==done['checkpoint_sha256']
        if frozen:
            assert meta['spec']==expected[r['run_id']] and c==frozen['finalist_configs'][c['id']]
            assert int(meta['cudnn_plan_cache_limit'])==frozen['cudnn_cache_limits'][c['id']]
            assert int(meta.get('cudnn_v8_api_disabled','0'))==frozen['cudnn_v8_disabled'][c['id']]
        if a.stage=='stage_c':
            assert meta['spec']['fit_split']=='full_train' and not meta['spec'].get('select_epoch',False)
            assert done['metrics']=={} and done['selected_epoch']==meta['spec']['epochs']
            checks.append({'run_id':r['run_id'],'passed':True,'checkpoint_sha256':done['checkpoint_sha256'],'scope':'Fixed-recipe fit completion and checkpoint identity only; no quality assertion'})
            continue
        if a.stage=='confirmation':
            assert r['freeze_sha256']==sha(ROOT/'configs/cnn_stage_c_v1.json') and r['checkpoint_sha256']==done['checkpoint_sha256']
            assert r['seed']==expected[r['run_id']]['seed']
            assert set(r['variants'])==({'fp32','fake_w8a8'} if c['id'] in frozen['quantization_config_ids'] else {'fp32'})
            assert int(r['cudnn_plan_cache_limit'])==frozen['cudnn_cache_limits'][c['id']]
            assert int(r['cudnn_v8_api_disabled'])==frozen['cudnn_v8_disabled'][c['id']]
            folder=DEST/'confirmation'/r['run_id'];splits=('search_temporary_threshold','search_candidate_scoring','diagnostics')
            for split in splits:
                if (rep,split) not in cache:cache[(rep,split)]=rows(DEST/'prepared'/rep/split/'documents.jsonl')
            for variant,v in r['variants'].items():
                assert v['scoring_batch_windows']==1
                cal=rows(folder/(variant+'_calibration.jsonl'));sel=rows(folder/(variant+'_development.jsonl'));diag=rows(folder/(variant+'_diagnostics.jsonl'))
                for rs,split in zip((cal,sel,diag),splits):score_check(rs,cache[(rep,split)])
                metric_check(cal,sel,v['metrics']);t=v['metrics']['0.01']['threshold']
                dm=evaluate_documents([d['risk_score'] for d in diag],[d['label'] for d in diag],t)
                for k in dm:assert dm[k]==v['diagnostics'][k]
                for split,key in (('calibration','calibration_score_sha256'),('development','development_score_sha256'),('diagnostics','diagnostics_score_sha256')):assert sha(folder/(variant+'_'+split+'.jsonl'))==v[key]
            assert bool(r['timing'])==expected[r['run_id']]['time']
            if r['timing']:
                floating={x['document_id']:x for x in rows(folder/'fp32_development.jsonl')}
                assert len(r['timing']['samples'])==12
                for sample in r['timing']['samples']:
                    assert sample['repeats']==20 and sample['warmups']==5 and len(sample['repetition_ms'])==20
                    stages=('preprocessing','h2d','network','return_aggregation','complete')
                    values=np.asarray([[x[k] for k in stages] for x in sample['repetition_ms']])
                    assert np.isfinite(values).all() and (values>=0).all()
                    assert np.array_equal(values.mean(0),[sample['mean_ms'][k] for k in stages])
                    assert float(np.quantile(values[:,4],.5))==sample['complete_p50_ms']
                    assert float(np.quantile(values[:,4],.95))==sample['complete_p95_ms']
                    row=floating[sample['document_id']];threshold=r['variants']['fp32']['metrics']['0.01']['threshold']
                    assert sample['windows']==row['window_count']
                    assert len(sample['repetition_risk_scores'])==len(sample['repetition_decisions'])==20
                    assert all(math.isfinite(x) and 0<=x<=1 for x in sample['repetition_risk_scores'])
                    assert all(d==(score>threshold) for score,d in zip(sample['repetition_risk_scores'],sample['repetition_decisions']))
                    assert sample['observed_risk_score']==sample['repetition_risk_scores'][-1]
                    assert sample['observed_decision']==sample['repetition_decisions'][-1]
                    assert sample['batch_one_score_delta']==sample['observed_risk_score']-row['risk_score']
                    assert sample['batch_one_decision_changed']==(sample['observed_decision']!=(row['risk_score']>threshold))
        else:
            for split in ('inner_calibration','inner_selection'):
                if (rep,split) not in cache:cache[(rep,split)]=rows(DEST/'prepared'/rep/split/'documents.jsonl')
            cal=rows(run/'calibration_scores.jsonl');sel=rows(run/'selection_scores.jsonl')
            score_check(cal,cache[(rep,'inner_calibration')]);score_check(sel,cache[(rep,'inner_selection')]);metric_check(cal,sel,r['metrics'])
        checks.append({'run_id':r['run_id'],'passed':True,'checkpoint_sha256':done['checkpoint_sha256']})
    scope='Fixed full-training recipe, final epoch, empty quality fields and checkpoint identity only; no scoring, inference or quality assertion' if a.stage=='stage_c' else 'Saved permitted rows; exact IDs/labels/text identities/window maxima, benign-only strict calibration, counts and checkpoint hashes; no inference or protected data'
    output={'stage':a.stage,'passed':True,'completed_runs_checked':len(checks),'checks':checks,'wall_seconds':time.perf_counter()-started,'scope':scope}
    (DEST/('saved_score_check_'+a.stage+'.json')).write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in output.items() if k!='checks'}))
if __name__=='__main__':main()
