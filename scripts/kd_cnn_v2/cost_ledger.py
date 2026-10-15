"""Reconcile recorded occupied-device wall time without treating it as energy."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from data import file_sha,load,save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--canonical-root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); root=a.canonical_root.resolve(); folder=root/'results/kd_cnn_v2'
    if a.out.exists():raise FileExistsError(a.out)
    records=[]; totals={'gpu0':0.,'vendor':0.}; reservations={'gpu0':0.,'vendor':0.}; pending=[]
    candidates=list((folder/'dispatch').glob('*.json'))+list(folder.glob('*dispatch.json'))+list(folder.glob('*_ram_lease.json'))
    for path in sorted(set(candidates)):
        d=load(path)
        if d.get('task')!='kd_cnn_v2' or d.get('resource') not in totals:
            raise ValueError('Unknown dispatch format: '+str(path))
        ram_exclusion=path.name.endswith('_ram_lease.json')
        item={'path':str(path.relative_to(root)),'sha256':file_sha(path),**d,
              'occupation_kind':'host_ram_exclusion' if ram_exclusion else 'compute_dispatch'}
        if d.get('wall_seconds') is None:pending.append(item['path'])
        else:(reservations if ram_exclusion else totals)[d['resource']]+=d['wall_seconds']
        records.append(item)
    fits=[]
    for run in sorted((folder/'runs').iterdir()):
        if not run.is_dir() or not (run/'run.json').exists():continue
        finished=load(run/'complete.json') if (run/'complete.json').exists() else None
        failure=load(run/'failure.json') if (run/'failure.json').exists() else None
        fits.append({'run_id':run.name,'completed':bool(finished),'completed_training_seconds':finished.get('training_seconds') if finished else None,
                     'selected_step':finished.get('selected_step') if finished else None,
                     'failure_record_preserved':failure is not None})
    limits=load(a.config)['resource_limits']; count={r:sum(x['resource']==r and x['occupation_kind']=='compute_dispatch' for x in records) for r in totals}
    occupied={r:totals[r]+reservations[r] for r in totals}
    report={'as_of_utc':datetime.now(timezone.utc).isoformat(),'config_sha256':file_sha(a.config),
        'completed_dispatch_wall_seconds':totals,'dispatch_counts_including_failed_or_running':count,
        'completed_host_ram_exclusion_seconds':reservations,'completed_resource_lease_wall_seconds':occupied,
        'started_unique_learning_runs':len(fits),'completed_learning_runs':sum(x['completed'] for x in fits),
        'learning_runs':fits,'pending_dispatch_records':pending,'dispatch_records':records,'limits':limits,
        'limit_checks':{'learning_runs':len(fits)<=limits['max_learning_runs'],
                        'gpu_recorded_time':occupied['gpu0']<=limits['gpu_seconds'],
                        'vendor_recorded_time':occupied['vendor']<=limits['vendor_seconds'],
                        'vendor_dispatches':count['vendor']<=limits['max_vendor_dispatches']},
        'accounting_boundary':'Occupied-resource subprocess wall time includes setup and failed attempts. Host-RAM exclusion leases are separately identified: they reserve GPU availability but perform zero GPU computation, and their duration is conservatively included in resource-time caps. Overlapping vendor/GPU leases are not summed as elapsed study time. Training runtime is cumulative across resumes and is not added again. Running records and uninstrumented CPU preparation/analysis are unresolved overhead, not zero. No energy, physical latency, dollar cost or account-token usage inferred.'}
    save(a.out,report);print(json.dumps({k:report[k] for k in ('completed_dispatch_wall_seconds','started_unique_learning_runs','completed_learning_runs','pending_dispatch_records','limit_checks')}))


if __name__=='__main__':main()
