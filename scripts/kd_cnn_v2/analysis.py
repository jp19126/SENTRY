"""Margin-based descriptive metrics and paired cluster sensitivity, no tuning."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gate_eval import fit_threshold, evaluate_documents


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def auc(scores, labels):
    score, label = np.asarray(scores), np.asarray(labels)
    na, nb = int(label.sum()), int((1-label).sum())
    if not na or not nb:
        return None
    _, groups = np.unique(score, return_inverse=True)
    p = np.bincount(groups, weights=label)
    n = np.bincount(groups, weights=1-label)
    return float(np.sum(p*(np.cumsum(n)-n/2))/(na*nb))


def roc(scores, labels):
    """Tied margins enter together; label-derived ROC is descriptive only."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    na, nb = int(labels.sum()), int((1-labels).sum())
    if not na or not nb:
        return []
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    endpoints = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores)-1]
    tp = np.cumsum(sorted_labels)[endpoints]
    fp = endpoints+1-tp
    return [{"tpr":0.,"fpr":0.,"included_minimum_margin":None}] + [
        {"tpr":float(t/na),"fpr":float(f/nb),"included_minimum_margin":float(sorted_scores[i])}
        for i,t,f in zip(endpoints,tp,fp)]


def counts(data, threshold):
    return evaluate_documents([r['margin'] for r in data], [r['label'] for r in data], threshold)


def describe(calibration, data):
    if not calibration or any(r['label'] != 0 for r in calibration):
        raise ValueError('Calibration must contain only benign documents')
    if len({r['document_id'] for r in data}) != len(data):
        raise ValueError('Duplicate score rows')
    for row in calibration + data:
        if not math.isfinite(row['margin']) or row['margin'] != max(row['window_margins']):
            raise ValueError('Nonfinite margin or incomplete max aggregation')
    result = {'auc':auc([r['margin'] for r in data],[r['label'] for r in data]),
              'roc':roc([r['margin'] for r in data],[r['label'] for r in data]),
              'source_groups':len({r['source_group_id'] for r in data}),
              'attack_payload_groups':len({r.get('payload_group_id') for r in data if r['label']==1}),
              'calibration_documents':len(calibration),'operating_points':{}}
    for alpha in (.005,.01,.02,.05):
        threshold = fit_threshold([r['margin'] for r in calibration],alpha)
        hard = [r for r in data if r['label']==0 and r.get('hard_benign')]
        strata = {}
        for name in ('mechanism_group_id','genre','source_dataset'):
            for value in sorted({str(r.get(name)) for r in data}):
                group = [r for r in data if str(r.get(name))==value]
                strata[name+':'+value] = counts(group,threshold)
        for lo,hi,label in ((1,1,'1'),(2,3,'2-3'),(4,7,'4-7'),(8,10**9,'8+')):
            group=[r for r in data if lo<=len(r['window_margins'])<=hi]
            strata['windows:'+label]=counts(group,threshold)
        result['operating_points'][str(alpha)] = {'overall':counts(data,threshold),
            'calibration':counts(calibration,threshold),'hard_benign':counts(hard,threshold),'strata':strata}
    return result


def paired(left, right, left_cal, right_cal, repeats=1000, seed=20261015):
    """Separate crossed-dependency sensitivity intervals, not joint coverage."""
    if [r['document_id'] for r in left] != [r['document_id'] for r in right]:
        raise ValueError('Pair memberships/order differ')
    if [r['document_id'] for r in left_cal] != [r['document_id'] for r in right_cal]:
        raise ValueError('Calibration memberships/order differ')
    for a,b in zip(left,right):
        if any(a.get(k)!=b.get(k) for k in ('label','source_group_id','payload_group_id','text_sha256')):
            raise ValueError('Paired identities differ')
    rng=np.random.default_rng(seed)
    y=np.asarray([r['label'] for r in left]); a=np.asarray([r['margin'] for r in left]); b=np.asarray([r['margin'] for r in right])
    ca=np.asarray([r['margin'] for r in left_cal]); cb=np.asarray([r['margin'] for r in right_cal])
    ta,tb=fit_threshold(ca,.01),fit_threshold(cb,.01)
    pa,pb=a>ta,b>tb
    result={'left_minus_right':{},'decision_disagreements':int(np.sum(pa!=pb)),
            'both_wrong':int(np.sum((pa!=y)&(pb!=y))), 'left_only_correct':int(np.sum((pa==y)&(pb!=y))),
            'right_only_correct':int(np.sum((pa!=y)&(pb==y))), 'repeats':repeats,'seed':seed,
            'scope':'Separate source/payload/calibration sensitivity; not a joint crossed-dependency interval or seed-as-data bootstrap.'}
    def interval(values):
        values=np.asarray(values); values=values[np.isfinite(values)]
        return {'interval_95':np.quantile(values,[.025,.975]).tolist() if len(values) else None,'valid_repeats':len(values)}
    for metric,label in (('recall',1),('fpr',0)):
        keep=y==label; diff=pa[keep].astype(float)-pb[keep].astype(float)
        result['left_minus_right'][metric]={'estimate':float(diff.mean()) if len(diff) else None}
        for axis in (('source_group_id','payload_group_id') if label else ('source_group_id',)):
            groups=[str(r.get(axis)) for r in left if r['label']==label]
            unique,inverse=np.unique(groups,return_inverse=True); n=len(unique)
            sums=np.bincount(inverse,weights=diff); sizes=np.bincount(inverse)
            samples=rng.integers(0,n,size=(repeats,n)) if n else np.zeros((repeats,0),dtype=int)
            values=sums[samples].sum(1)/sizes[samples].sum(1) if n else []
            result['left_minus_right'][metric][axis]={**interval(values),'groups':n}
    groups=np.asarray([r['source_group_id'] for r in left_cal]); unique,inverse=np.unique(groups,return_inverse=True)
    indexes=[np.flatnonzero(inverse==i) for i in range(len(unique))]
    curves={'recall':[],'fpr':[]}
    for _ in range(repeats):
        ix=np.concatenate([indexes[i] for i in rng.integers(0,len(unique),len(unique))])
        qa,qb=a>fit_threshold(ca[ix],.01),b>fit_threshold(cb[ix],.01)
        for metric,label in (('recall',1),('fpr',0)):
            curves[metric].append(float((qa[y==label].astype(float)-qb[y==label]).mean()))
    for metric in curves: result['left_minus_right'][metric]['calibration_source_resampling']=interval(curves[metric])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores',required=True);p.add_argument('--calibration',required=True);p.add_argument('--out',required=True)
    p.add_argument('--reference-scores');p.add_argument('--reference-calibration')
    a=p.parse_args(); data,cal=rows(a.scores),rows(a.calibration); result=describe(cal,data)
    if a.reference_scores: result['paired']=paired(data,rows(a.reference_scores),cal,rows(a.reference_calibration))
    path=Path(a.out)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'out':str(path),'auc':result['auc'],'primary':result['operating_points']['0.01']['overall']}))


if __name__=='__main__':main()
