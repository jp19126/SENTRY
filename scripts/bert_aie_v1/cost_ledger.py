"""Reconstruct recorded job costs without treating lock waits as computation."""
import json
from pathlib import Path
from common import ROOT, OUT, read, sha, write


def main():
    dispatch=[]
    for p in sorted((OUT/'dispatch').glob('*/run.json')):
        d=read(p);acquired=d.get('acquired_unix');child=d.get('child_pid')
        dispatch.append({'name':p.parent.name,'record_sha256':sha(p),'resource':d['resource'],
            'started_unix':d['started_unix'],'lock_wait_seconds':None if acquired is None else acquired-d['started_unix'],
            'child_started':child is not None,'complete':d.get('complete'),
            'exit_code':d.get('exit_code'),'error':d.get('error'),
            'recorded_elapsed_seconds':d.get('elapsed_seconds'),
            'reserved_child_wall_seconds':d.get('elapsed_seconds') if child is not None else 0})
    builds=[]
    for p in sorted((ROOT/'build/bert_aie_v1').glob('*/*attempt*.json')):
        d=read(p)
        if not all(k in d for k in ('command','elapsed_seconds')):continue
        builds.append({'name':str(p.relative_to(ROOT)),'record_sha256':sha(p),
            'command':d['command'],'started_utc':d.get('started_utc'),
            'elapsed_seconds':d['elapsed_seconds'],'returncode':d.get('returncode'),
            'resource_floor_abort':d.get('resource_floor_abort'),'timeout':d.get('timeout',False)})
    totals={resource:sum(r['reserved_child_wall_seconds'] or 0 for r in dispatch if r['resource']==resource)
            for resource in ('cpu','gpu0','gpu_vendor')}
    vendor=sum(r['elapsed_seconds'] for r in builds if any(tool in ' '.join(r['command']) for tool in ('aiecompiler','aiesimulator','vitis-run','v++','vivado')))
    result={'dispatch':dispatch,'builds':builds,'reserved_child_wall_seconds_by_dispatch_resource':totals,
            'recorded_vendor_stage_wall_seconds':vendor,
            'boundary':'Elapsed wall-clock records, not GPU-active time, CPU core-seconds, energy or billed cost. Outer lock waiting is separate. gpu_vendor final_profiles_v1 includes its child waiting on the CPU lock; reserve totals are conservative and exceed actual model runtime. Preflight/lock failures have no child computation. Incomplete or manually interrupted records are unresolved, never counted as successful execution. Direct short preparation/smoke work outside these dispatch records is not exhaustively metered.',
            'limits':read(ROOT/'configs/bert_aie_v1.json')['bounded_matrix']}
    write(OUT/'cost_ledger.json',result)
    print(json.dumps({'dispatch_resource_seconds':totals,'vendor_stage_seconds':vendor,'records':len(dispatch)+len(builds)},indent=2))


if __name__=='__main__':main()
