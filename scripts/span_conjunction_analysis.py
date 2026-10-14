"""Analyze the frozen BERT AND CNN policy without treating BERT risk as AND risk.

Refits both component calibration thresholds within each paired source bootstrap.
Exports explicit 0/1 decisions for the existing unchanged-input application replay.
No inference, model fitting, protected data, target calls or vendor operations.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gate_eval import fit_threshold,evaluate_documents
from span_analyze import read,aligned,bootstrap_plan,CAL,NAT,PROBE


def decision(left,right,tl,tr):
    return (np.asarray(left)>tl)&(np.asarray(right)>tr)


def counts(labels,positive):
    value=evaluate_documents(np.asarray(positive,dtype=float).tolist(),[int(label) for label in labels],.5)
    value.pop('threshold');value['attack_rule']='BERT score > own threshold AND CNN score > own threshold'
    value['balanced_error']=(1-value['recall']+value['fpr'])/2 if value['attacks'] and value['benign'] else None
    return value


def strata_masks(rows):
    groups=defaultdict(list)
    for i,d in enumerate(rows):
        for key,value in (('windows','single' if d['window_count']==1 else 'multiple'),
                          ('position',d.get('injection_position')),('recipe',d.get('probe_recipe')),
                          ('attack_type',d.get('attack_type'))):
            if value is not None:groups[key+':'+str(value)].append(i)
    return {'all':np.arange(len(rows)),**{k:np.asarray(v) for k,v in sorted(groups.items())}}


def interval(values):
    finite=np.asarray(values)[np.isfinite(values)]
    return dict(percentile_95=np.quantile(finite,[.025,.975]).tolist() if len(finite) else None,
                defined_replicates=len(finite))


def rates(positive,labels,weights):
    attack=weights*labels;benign=weights*(~labels)
    with np.errstate(divide='ignore',invalid='ignore'):
        recall=(attack*positive).sum(1)/attack.sum(1)
        fpr=(benign*positive).sum(1)/benign.sum(1)
    return dict(recall=recall,fpr=fpr,balanced_error=(1-recall+fpr)/2)


def threshold_draws(rows,plan):
    lookup={d['document_id']:d['risk_score'] for d in rows if d['label']==0}
    scores=np.asarray([lookup[k] for k in plan['calibration_ids']])
    # Exactly n-floor(.01*n)-1, the gate_eval strict quantile rule.
    rank=len(scores)-len(scores)//100-1
    return np.partition(scores[plan['calibration_draws']],rank,axis=1)[:,rank]


def run(args):
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'results/span_coupled_pilot_v1') or output.exists():raise ValueError('Fresh study output required')
    left,right,hashes={},{},{}
    for split in (CAL,NAT,PROBE):
        lpath=args.bert/(split+'_documents.jsonl');rpath=args.cnn/(split+'_documents.jsonl')
        left[split],hashes[str(lpath)]=read(lpath);right[split],hashes[str(rpath)]=read(rpath)
        right[split]=aligned(right[split],left[split],same_windows=True)
    for arm in (left,right):
        if {d['source_group_id'] for d in arm[CAL]}&{d['source_group_id'] for s in (NAT,PROBE) for d in arm[s]}:raise ValueError('Calibration leakage')
    tl=fit_threshold([d['risk_score'] for d in left[CAL] if not d['label']],.01)
    tr=fit_threshold([d['risk_score'] for d in right[CAL] if not d['label']],.01)
    plan=bootstrap_plan(left,2000);draw_l=threshold_draws(left[CAL],plan);draw_r=threshold_draws(right[CAL],plan)
    report={}
    aliases={CAL:'calibration',NAT:'natural',PROBE:'stress'}
    for split in (CAL,NAT,PROBE):
        rows=left[split];labels=np.asarray([d['label'] for d in rows],dtype=bool)
        lrisks=np.asarray([d['risk_score'] for d in rows]);rrisks=np.asarray([d['risk_score'] for d in right[split]])
        positive=decision(lrisks,rrisks,tl,tr);masks=strata_masks(rows)
        saved_path=args.conjunction/(aliases[split]+'_documents.jsonl')
        saved_raw=saved_path.read_bytes();hashes[str(saved_path)]=hashlib.sha256(saved_raw).hexdigest()
        saved={d['document_id']:d for d in (json.loads(line) for line in saved_raw.decode('utf-8').splitlines())}
        if set(saved)!={d['document_id'] for d in rows}:raise ValueError('Saved conjunction identity differs')
        for d,p in zip(rows,positive):
            if bool(saved[d['document_id']]['attack_decisions']['0.01'])!=bool(p):raise ValueError('Saved AND decision differs')
        result={k:dict(point=counts(labels[ids],positive[ids])) for k,ids in masks.items()}
        if split!=CAL:
            samples={k:{m:[] for m in ('recall','fpr','balanced_error')} for k in masks}
            deltas={name:[] for name in ('and_minus_bert_balanced_error','and_minus_cnn_balanced_error')}
            index=plan[split]['indices']
            for start in range(0,2000,50):
                stop=start+50;weights=plan[split]['weights'][start:stop,index]
                pl=lrisks[None,:]>draw_l[start:stop,None];pr=rrisks[None,:]>draw_r[start:stop,None]
                both=pl&pr
                for key,ids in masks.items():
                    rr=rates(both[:,ids],labels[ids],weights[:,ids])
                    for metric,value in rr.items():samples[key][metric].extend(value.tolist())
                conjunction=rates(both,labels,weights)['balanced_error']
                for name,pred in (('bert',pl),('cnn',pr)):
                    deltas['and_minus_'+name+'_balanced_error'].extend((conjunction-rates(pred,labels,weights)['balanced_error']).tolist())
            for key in result:result[key]['source_and_calibration_bootstrap']={m:interval(v) for m,v in samples[key].items()}
            result['paired_component_differences']={name:interval(100*np.asarray(values)) for name,values in deltas.items()}
        report[split]=result
    # The application replay accepts scalar risks. These are explicit decisions,
    # not probabilities and not inherited component scores. Threshold .5 merely
    # encodes the already fixed two-threshold AND policy without recalibration.
    source=args.conjunction/'application_pilot_documents.jsonl';raw=source.read_bytes()
    hashes[str(source)]=hashlib.sha256(raw).hexdigest();pilot=[]
    for d in (json.loads(line) for line in raw.decode('utf-8').splitlines()):
        reject=bool(d['attack_decisions']['0.01'])
        if reject!=bool(d['bert_risk']>tl and d['cnn_risk']>tr):raise ValueError('Pilot AND decision differs')
        record={k:v for k,v in d.items() if k not in ('risk_score','window_scores','window_count','content_token_count','is_long_document')}
        record.update(risk_score=float(reject),score_semantics='Encoded fixed AND decision, not a model probability')
        pilot.append(record)
    if len(pilot)!=120 or sum(d['label']==0 for d in pilot)!=20:raise ValueError('Pilot population differs')
    output.mkdir(parents=True)
    with (output/'application_pilot_decisions.jsonl').open('x',encoding='utf-8') as stream:
        for row in pilot:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    threshold_record=dict(complete=True,thresholds={'0.01':.5},component_thresholds={'bert':tl,'cnn':tr},
        score_semantics='Threshold .5 decodes fixed 0/1 AND policy; not a newly fitted risk threshold')
    (output/'application_threshold.json').write_text(json.dumps(threshold_record,indent=2)+'\n',encoding='utf-8')
    report['application_pilot']=counts([d['label'] for d in pilot],[bool(d['risk_score']) for d in pilot])
    result=dict(complete=True,component_thresholds={'bert':tl,'cnn':tr},results=report,input_sha256=hashes,
        bootstrap=dict(repetitions=2000,seed=20261013,calibration='Shared benign source draws; both component thresholds refitted independently within each draw',
            scoring='Shared source-group cluster draws preserving recipe and attack/benign dependence',difference_units='percentage points'),
        limitations='Exploratory conventional policy on exposed development scores; no cascade execution or latency measurement. Bootstrap conditions on fixed checkpoints and authored recipes; not a training-seed or attack-population generalization interval.',
        application_replay='Use application_pilot_decisions.jsonl with application_threshold.json in span_application_replay.py; unchanged saved responses only')
    (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


def self_test():
    # Strict threshold equality is benign, and AND cannot act like OR.
    assert decision([.5,.6,.4,.6],[.6,.5,.9,.7],.5,.5).tolist()==[False,False,False,True]
    # At n=100, duplicated upper-tail samples change each refitted threshold.
    rows=[dict(document_id=str(i),risk_score=i/100,label=0) for i in range(100)]
    draws=np.asarray([list(range(100)),[0]*98+[99,99]])
    plan=dict(calibration_ids=[str(i) for i in range(100)],calibration_draws=draws)
    actual=threshold_draws(rows,plan)
    expected=[fit_threshold([rows[i]['risk_score'] for i in draw],.01) for draw in draws]
    assert actual.tolist()==expected==[.98,.99]
    rr=rates(np.asarray([[True,False,True,False]]),np.asarray([True,True,False,False]),np.asarray([[2,1,2,1]]))
    assert rr['recall'][0]==2/3 and rr['fpr'][0]==2/3
    report=dict(results={'natural':{'all':dict(point=counts(np.asarray([1,1,0,0]),np.asarray([True,False,True,False])),
        source_and_calibration_bootstrap={key:interval(value) for key,value in rr.items()})}})
    restored=json.loads(json.dumps(report))
    assert restored['results']['natural']['all']['point']['tp']==1
    print('strict threshold, AND, calibration refit, cluster weighting and report JSON checks passed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--bert',type=Path);p.add_argument('--cnn',type=Path);p.add_argument('--conjunction',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.self_test:self_test()
    else:
        if not all((args.bert,args.cnn,args.conjunction,args.output)):p.error('All three input directories and output required')
        run(args)
