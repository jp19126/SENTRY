"""Tabulate completed development runs at selected and matched terminal steps."""
import argparse
import json
from pathlib import Path
from data import file_sha,load,save


def compact(metrics):
    p=metrics['0.01']
    return {'auc':metrics['auc'],'recall':p['recall'],'fpr':p['fpr'],
            'tp':p['tp'],'fp':p['fp'],'threshold_margin':p['threshold_margin']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--canonical-root',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--require-all',action='store_true')
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    cfg=load(a.config); records=[]; pending=[]
    for recipe in cfg['recipes']:
        if not recipe.get('monitor_development',True):continue
        ident=recipe['run_id']; folder=a.canonical_root/'results/kd_cnn_v2/runs'/ident
        if not (folder/'complete.json').exists():pending.append(ident);continue
        run=load(folder/'run.json');done=load(folder/'complete.json');history=load(folder/'history.json')
        if run['association']['recipe']!=recipe or done['association_sha256']!=run['association_sha256']:
            raise ValueError('Run identity differs: '+ident)
        if done['steps_completed']!=recipe['total_steps'] or history[-1]['step']!=recipe['total_steps']:
            raise ValueError('Terminal exposure missing: '+ident)
        if file_sha(folder/'checkpoint_selected.pt')!=done['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed')
        records.append({'run_id':ident,'recipe':recipe,'selected_step':done['selected_step'],
            'selected':compact(done['best']['development']),'terminal':compact(history[-1]['development']),
            'selected_checkpoint_sha256':done['checkpoint_sha256'],
            'terminal_checkpoint_sha256':file_sha(folder/'checkpoint_latest.pt'),
            'training_seconds':done['training_seconds'],
            'history':[{'step':h['step'],'fit_passes':h['fit_passes'],'loss':h['mean_batch_loss'],
                        'quality':compact(h['development'])} for h in history]})
    if pending and a.require_all:raise ValueError('Incomplete runs: '+', '.join(pending))
    save(a.out,{'config_sha256':file_sha(a.config),'records':records,'pending':pending,
                'boundary':'Development evidence only. Terminal points support matched-exposure control edges; selected points nominate a principal recipe. No confirmation or automatic model choice.'})
    print(json.dumps({'completed':len(records),'pending':pending,
        'selected':[{'run':r['run_id'],'step':r['selected_step'],**r['selected']} for r in records],
        'terminal':[{'run':r['run_id'],**r['terminal']} for r in records]}))


if __name__=='__main__':main()
