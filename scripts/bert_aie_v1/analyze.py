"""Verify saved margin scores and report paired fixed/own-threshold sensitivity.

Source-cluster and crossed source/payload bootstrap intervals describe the fixed
observed panel, not deployment guarantees or variability over training seeds.
"""
from __future__ import annotations
import argparse
from decimal import Decimal, ROUND_FLOOR
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import read, rows, sha, write

ALPHAS=(.01,.005,.02,.05)

def fit_threshold(scores,alpha):
    values=np.sort(np.asarray(scores,dtype=np.float64))
    if not len(values) or not np.isfinite(values).all() or not 0<=alpha<1: raise ValueError('Finite benign calibration scores required')
    count=int((Decimal(str(alpha))*len(values)).to_integral_value(rounding=ROUND_FLOOR))
    return float(values[len(values)-count-1])

def metrics(records,threshold):
    labels=np.asarray([r['label'] for r in records]); pred=np.asarray([r['margin'] for r in records])>threshold
    attack=labels==1; benign=labels==0
    tp=int((pred&attack).sum()); fp=int((pred&benign).sum()); na=int(attack.sum()); nb=int(benign.sum())
    return {'documents':len(records),'attacks':na,'benign':nb,'tp':tp,'fn':na-tp,'fp':fp,'tn':nb-fp,
            'recall':tp/na if na else None,'fpr':fp/nb if nb else None,'threshold':float(threshold),
            'source_groups':len({r.get('source_group_id') for r in records if r.get('source_group_id')}),
            'attack_payload_groups':len({r.get('payload_group_id') for r in records if r['label']==1 and r.get('payload_group_id')}),
            'rule':'margin > threshold'}

def roc_report(records):
    labels=np.asarray([r['label'] for r in records],dtype=int); scores=np.asarray([r['margin'] for r in records],dtype=float)
    na=int(labels.sum()); nb=len(labels)-na
    if not na or not nb: return {'auc':None,'points':[],'reason':'Both labels required'}
    order=np.argsort(-scores,kind='stable'); labels=labels[order]; scores=scores[order]
    endpoints=np.r_[np.flatnonzero(np.diff(scores)!=0),len(scores)-1]
    tp=np.cumsum(labels)[endpoints]; fp=(endpoints+1)-tp
    tpr=np.r_[0,tp/na]; fpr=np.r_[0,fp/nb]
    auc=float(np.sum((tpr[1:]+tpr[:-1])*np.diff(fpr)/2))
    points=[{'threshold':None,'fpr':0.,'recall':0.,'kind':'reject_all'}]
    points.extend({'threshold':float(np.nextafter(scores[i],-np.inf)),'fpr':float(f),'recall':float(t)} for i,f,t in zip(endpoints,fpr[1:],tpr[1:]))
    return {'auc':auc,'points':points,'descriptive_at_observed_fpr':{str(a):max(p['recall'] for p in points if p['fpr']<=a) for a in ALPHAS},
            'boundary':'Empirical label-informed ROC; no deployable threshold and no interpolation at unobserved low-FPR points'}

def strata(records,threshold):
    output={}
    fields=('source_dataset','genre','mechanism_group_id','benign_category')
    for field in fields:
        for value in sorted({str(r[field]) for r in records if r.get(field) is not None}):
            group=[r for r in records if str(r.get(field))==value]
            output[field+':'+value]=metrics(group,threshold)
    for label in ('1','2-3','4-7','8+'):
        group=[r for r in records if window_bin(r['window_count'])==label]
        output['window_count:'+label]=metrics(group,threshold)
    output['hard_benign']=metrics([r for r in records if r['label']==0 and r.get('hard_benign') is True],threshold)
    return output

def window_bin(count): return '1' if count==1 else '2-3' if count<4 else '4-7' if count<8 else '8+'

def load_verified(folder,freeze=None,freeze_sha256=None):
    """Check authorization, score bytes and reaggregation before returning rows."""
    folder=Path(folder); complete=read(folder/'complete.json')
    if not complete.get('complete') or complete['schema']!='sentry_bert_aie_v1_scores_v1': raise ValueError('Completed task scoring required')
    if complete['stage']=='confirmation':
        if not freeze or not freeze_sha256 or complete['freeze_sha256']!=freeze_sha256 or sha(freeze)!=freeze_sha256:
            raise ValueError('Frozen confirmation authorization required before saved rows are decoded')
        from evaluate import authorize
        authorize(SimpleNamespace(stage='confirmation',calibration_role=Path(complete['data']['calibration']['directory']),
                    selection_role=Path(complete['data']['selection']['directory']),freeze=freeze,freeze_sha256=freeze_sha256,
                    variant=complete['target']['variant']),complete['target'])
    result={}
    for role,desc in complete['score_files'].items():
        if Path(desc['name']).name!=desc['name'] or sha(folder/desc['name'])!=desc['sha256']: raise ValueError('Saved score bytes changed')
        scored=list(rows(folder/desc['name'])); seen=set()
        for row in scored:
            vals=row['window_margins']
            if row['document_id'] in seen or row['label'] not in (0,1) or not row.get('source_group_id') or not row.get('text_sha256') or not row.get('window_input_sha256'):
                raise ValueError('Missing/duplicate row association')
            seen.add(row['document_id'])
            if not vals or len(vals)!=row['window_count'] or not all(math.isfinite(v) for v in vals) or row['margin']!=max(vals): raise ValueError('Saved document max margin invalid')
        result[role]=scored
    thresholds={str(a):fit_threshold([r['margin'] for r in result['calibration'] if r['label']==0],a) for a in ALPHAS}
    if thresholds!=complete['thresholds']: raise ValueError('Saved thresholds fail independent refit')
    for key,t in thresholds.items():
        if metrics(result['selection'],t)!=complete['metrics'][key]: raise ValueError('Saved confusion metrics differ')
    return complete,result

def pair(left,right):
    fields=('document_id','label','text_sha256','window_input_sha256','window_count','source_group_id','payload_group_id')
    if len(left)!=len(right): raise ValueError('Paired membership differs')
    index={r['document_id']:r for r in right}
    aligned=[]
    for row in left:
        other=index.get(row['document_id'])
        if other is None or any(row.get(k)!=other.get(k) for k in fields): raise ValueError('Paired row inputs/ancestry differ')
        aligned.append(other)
    return aligned

def interval(values):
    finite=np.asarray([x for x in values if x is not None and math.isfinite(x)])
    if not len(finite):return {'draws':0,'q025':None,'median':None,'q975':None}
    lo,med,hi=np.quantile(finite,[.025,.5,.975]);return {'draws':len(finite),'q025':float(lo),'median':float(med),'q975':float(hi)}

def encoded_groups(records,field):
    keys=sorted({str(r[field]) for r in records if r.get(field) is not None})
    lookup={key:i for i,key in enumerate(keys)}
    return keys,np.asarray([lookup[str(r[field])] if r.get(field) is not None else -1 for r in records],dtype=int)

def resample_weights(rng,codes,count):
    if count==0:return np.ones(len(codes))
    sampled=np.bincount(rng.integers(0,count,size=count),minlength=count)
    weights=np.ones(len(codes),dtype=float); valid=codes>=0; weights[valid]=sampled[codes[valid]]
    return weights

def paired_values(candidate,reference,tc,tr,weights=None):
    labels=np.asarray([r['label'] for r in candidate]); attack=labels==1; benign=~attack
    cp=np.asarray([r['margin'] for r in candidate])>tc; rp=np.asarray([r['margin'] for r in reference])>tr
    w=np.ones(len(labels)) if weights is None else weights
    values={}
    for name,mask in (('recall',attack),('fpr',benign)):
        denom=float(w[mask].sum()); c=float(w[mask&cp].sum()/denom) if denom else None; r=float(w[mask&rp].sum()/denom) if denom else None
        values['candidate_'+name]=c; values['reference_'+name]=r; values['delta_'+name]=c-r if c is not None and r is not None else None
    values.update(disagreements=int(np.count_nonzero(cp!=rp)),candidate_only=int(np.count_nonzero(cp&~rp)),reference_only=int(np.count_nonzero(rp&~cp)))
    return values

def bootstrap(candidate,reference,cal_candidate,cal_reference,tc,tr,alpha,reps,seed):
    rng=np.random.default_rng(seed)
    sk,sc=encoded_groups(candidate,'source_group_id'); pk,pc=encoded_groups(candidate,'payload_group_id')
    benign=[i for i,r in enumerate(cal_candidate) if r['label']==0]
    cc=[cal_candidate[i] for i in benign]; cr=[cal_reference[i] for i in benign]
    ck,codes=encoded_groups(cc,'source_group_id')
    draws={name:{metric:[] for metric in ('delta_recall','delta_fpr')} for name in ('source','crossed_source_payload','calibration_source')}
    for _ in range(reps):
        sw=resample_weights(rng,sc,len(sk)); pw=resample_weights(rng,pc,len(pk))
        for name,weight in (('source',sw),('crossed_source_payload',sw*pw)):
            value=paired_values(candidate,reference,tc,tr,weight)
            for metric in draws[name]:draws[name][metric].append(value[metric])
        cw=resample_weights(rng,codes,len(ck)).astype(int)
        selected=np.repeat(np.arange(len(cc)),cw)
        if len(selected):
            ct=fit_threshold([cc[i]['margin'] for i in selected],alpha); rt=fit_threshold([cr[i]['margin'] for i in selected],alpha)
            value=paired_values(candidate,reference,ct,rt)
            for metric in draws['calibration_source']:draws['calibration_source'][metric].append(value[metric])
    return {'repetitions':reps,'seed':seed,'selection_source_groups':len(sk),'selection_payload_groups':len(pk),'calibration_source_groups':len(ck),
            'missing_payload_documents':int((pc<0).sum()),'missing_attack_payload_documents':sum(r['label']==1 and not r.get('payload_group_id') for r in candidate),
            'intervals':{name:{key:interval(value) for key,value in result.items()} for name,result in draws.items()},
            'interpretation':'Paired observed-group percentile sensitivity. Crossed product weights independently resample sources and nonmissing payload ancestries; missing payload rows retain source weights. Calibration sensitivity refits both benign thresholds on identical source resamples, with selection fixed. Not independent-population or seed reliability intervals.',
            'sparse_caveat':'Few shared payload/context families and authored construction can make intervals unstable or incomplete; no universal minimum-cluster pass is imposed.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--reference',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--freeze',type=Path);p.add_argument('--freeze-sha256')
    p.add_argument('--reps',type=int,default=1000);p.add_argument('--seed',type=int,default=424242);args=p.parse_args()
    if not 1<=args.reps<=10000:raise ValueError('Bootstrap repetitions must be1..10000')
    c,cd=load_verified(args.candidate,args.freeze,args.freeze_sha256)
    output={'candidate_complete_sha256':sha(args.candidate/'complete.json'),'verified_saved_scores':True,'stage':c['stage'],'target':c['target'],
            'metrics':c['metrics'],'roc':roc_report(cd['selection']), 'strata':{key:strata(cd['selection'],t) for key,t in c['thresholds'].items()}}
    if args.reference:
        r,rd=load_verified(args.reference,args.freeze,args.freeze_sha256)
        if c['stage']!=r['stage'] or c['data']!=r['data'] or c['freeze_sha256']!=r['freeze_sha256']:raise ValueError('Paired stage/data/freeze differ')
        rd={role:pair(cd[role],rd[role]) for role in cd}; comparisons={}
        for alpha in ALPHAS:
            key=str(alpha);ct=c['thresholds'][key];rt=r['thresholds'][key]
            comparisons[key]={'own_calibration':paired_values(cd['selection'],rd['selection'],ct,rt),
                'fixed_reference_threshold':paired_values(cd['selection'],rd['selection'],rt,rt),
                'candidate_threshold':ct,'reference_threshold':rt,
                'candidate_strata_fixed_reference_threshold':strata(cd['selection'],rt),
                'bootstrap_own_calibration':bootstrap(cd['selection'],rd['selection'],cd['calibration'],rd['calibration'],ct,rt,alpha,args.reps,args.seed)}
        output.update(reference_complete_sha256=sha(args.reference/'complete.json'),comparisons=comparisons)
    from evaluate import ensure_directory
    ensure_directory(args.out.parent)
    output['analysis_source_sha256']=sha(Path(__file__));write(args.out,output)
    print(json.dumps({'out':str(args.out),'verified_saved_scores':True,'comparisons':bool(args.reference)}))

if __name__=='__main__':main()
