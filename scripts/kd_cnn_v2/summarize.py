"""Summarize explicitly nominated score artifacts, with no model selection."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from analysis import describe, paired, rows
from data import file_sha, load, save
from gate_eval import fit_threshold


def checked(path, stage, role, ident):
    side=load(path.with_suffix('.complete.json'))
    expected={'complete':True,'stage':stage,'role':role,'target_id':ident,'scores_sha256':file_sha(path)}
    if any(side.get(k)!=v for k,v in expected.items()): raise ValueError('Score identity differs: '+str(path))
    data=rows(path)
    if len(data)!=side['documents'] or any(r['target_id']!=ident or r['role']!=role or r['stage']!=stage for r in data):
        raise ValueError('Score rows differ from sidecar')
    return data,side


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--score-root',type=Path,required=True)
    p.add_argument('--target-list',type=Path,required=True)
    p.add_argument('--stage',choices=('development','confirmation'),required=True)
    p.add_argument('--freeze',type=Path,help='Required immutable confirmation freeze')
    p.add_argument('--pairs',type=Path,help='JSON list: {left,right,question}')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    calrole,datarole=('new_calibration','new_confirmation') if a.stage=='confirmation' else ('new_development_calibration','new_selection')
    targets=load(a.target_list)['targets']; scored={}; summary={}; sidecars={}
    frozen=load(a.freeze) if a.freeze else None
    if a.stage=='confirmation' and (not frozen or frozen.get('schema')!='kd_cnn_v2_confirmation_freeze_v1'):
        raise ValueError('Confirmation summary requires completed candidate freeze')
    if frozen and file_sha(a.target_list)!=frozen['target_list_sha256']:
        raise ValueError('Summary target list differs from frozen list')
    for target in targets:
        ident=target['id']; base=a.score_root/ident
        cal,cs=checked(base/(calrole+'.jsonl'),a.stage,calrole,ident)
        data,ds=checked(base/(datarole+'.jsonl'),a.stage,datarole,ident)
        if cs['target']!=ds['target'] or cs['source_sha256']!=ds['source_sha256'] or cs['freeze_sha256']!=ds['freeze_sha256']:
            raise ValueError('Calibration/score identity mismatch')
        if frozen:
            for side,role in ((cs,calrole),(ds,datarole)):
                expected={'freeze_sha256':file_sha(a.freeze),'target':frozen['targets'][ident],
                          'source_sha256':frozen['source_sha256'],
                          'prepared_manifest_sha256':frozen['reserved_roles'][role]['manifest_sha256']}
                if any(side.get(k)!=v for k,v in expected.items()):raise ValueError('Score does not match candidate/data freeze')
        if {r['source_group_id'] for r in cal}&{r['source_group_id'] for r in data}:
            raise ValueError('Calibration and selection source ancestry overlap')
        if {r['document_id'] for r in cal}&{r['document_id'] for r in data}:
            raise ValueError('Calibration and selection document overlap')
        scored[ident]=(cal,data);sidecars[ident]={'calibration':cs,'data':ds}
        stats=describe(cal,data);stats['target']=target
        if target['kind']=='cnn_integer':
            t=stats['operating_points']['0.01']['overall']['threshold']; code=math.floor(t*4096)
            if any(any(abs(q)>=2**31 for q in row['window_margin_q12']) for row in data):raise ValueError('Integer margin range')
            if any((r['margin']>t)!=(max(r['window_margin_q12'])>code) for r in data):raise ValueError('Host threshold conversion mismatch')
            stats['integer_threshold_q12']=code
        summary[ident]=stats
    comparisons=[]
    for edge in load(a.pairs) if a.pairs else []:
        left,right=edge['left'],edge['right']; lc,ld=scored[left];rc,rd=scored[right]
        item={**edge,'paired':paired(ld,rd,lc,rc)}
        # This common-cut sensitivity is descriptive; each model's calibrated
        # cut remains the principal comparison. Window correspondence is exact.
        t=fit_threshold([r['margin'] for r in rc],.01)
        deltas=[]
        for x,y in zip(ld,rd):
            if len(x['window_margins'])!=len(y['window_margins']):raise ValueError('Window correspondence differs')
            deltas.extend(np.subtract(x['window_margins'],y['window_margins']).tolist())
        item['right_threshold_sensitivity']={'threshold':t,
            'document_disagreements':sum((x['margin']>t)!=(y['margin']>t) for x,y in zip(ld,rd)),
            'max_abs_window_margin_difference':float(np.max(np.abs(deltas))),
            'mean_abs_window_margin_difference':float(np.mean(np.abs(deltas)))}
        comparisons.append(item)
    groups={}
    for target in targets:
        if target.get('report_group'):groups.setdefault(target['report_group'],[]).append(target['id'])
    aggregates={}
    for group,ids in groups.items():
        values={key:[summary[i]['operating_points']['0.01']['overall'][key] for i in ids] for key in ('recall','fpr')}
        values['auc']=[summary[i]['auc'] for i in ids]
        aggregates[group]={'targets':ids,'seeds':[summary[i]['target'].get('seed') for i in ids],
            'metrics':{k:{'values':v,'mean':float(np.mean(v)),'sample_sd':float(np.std(v,ddof=1)) if len(v)>1 else None} for k,v in values.items()},
            'interpretation':'Descriptive variability over fixed training seeds on the same documents; not independent data replications.'}
    save(a.out,{'stage':a.stage,'target_list_sha256':file_sha(a.target_list),
                'score_root':str(a.score_root.resolve()),'source_sha256':file_sha(__file__),
                'targets':summary,'comparisons':comparisons,'seed_groups':aggregates,
                'score_evidence':sidecars})
    print(json.dumps({'out':str(a.out),'targets':len(targets),'pairs':len(comparisons),'groups':list(groups)}))


if __name__=='__main__':main()
