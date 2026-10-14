"""Reconcile recorded CNN job walls without double-counting nested dispatches."""
import hashlib,json,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'results/cnn_study_v1'

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    jobs=[];seen=set()
    def add(path,seconds,kind,boundary):
        key=str(path.relative_to(ROOT))
        if key in seen:raise ValueError('Duplicate runtime evidence')
        seen.add(key);jobs.append({'evidence':key,'recorded_wall_seconds':seconds,'kind':kind,'boundary':boundary})
    # Root queue records already include their subprocess walls and startup.
    # Do not sum *_group*_dispatch.json or training_seconds again.
    for path in sorted(DEST.glob('*_queue_*.json')):
        record=read(path)
        if 'dispatches' not in record:continue
        seconds=sum(d['wall_seconds'] for d in record['dispatches'])
        add(path,seconds,'accelerator_queue','Sum of terminal child process walls; includes interrupted jobs/startup/teardown, excludes any still live child')
    for name in ('timing_dispatch.json','stage_a_dispatch_initial_memory_stop.json','cudnn_cache_dispatch.json','pyramid_cache8_dispatch.json','pyramid_backend_dispatch.json'):
        path=DEST/name
        if path.exists():add(path,read(path)['wall_seconds'],'accelerator_support','Recorded guarded dispatch wall; initial/cache support included once')
    path=DEST/'confirmation_queue.json'
    if path.exists():add(path,sum(j['wall_seconds'] for j in read(path)['jobs']),'accelerator_confirmation','Terminal scoring/profile child processes including startup/teardown')
    path=DEST/'cold_process_timings.json'
    if path.exists():add(path,sum(r['parent_spawn_to_process_exit_seconds'] for r in read(path)),'accelerator_cold_profile','Fresh child process spawn through exit; system caches not flushed')
    complete={}
    for path in DEST.glob('*_run_table.json'):
        if path.name=='timing_run_table.json':continue # Equivalent timing fits are in stage A.
        for row in read(path):
            if row.get('complete'):complete[row['run_id']]=row
    accelerator=sum(j['recorded_wall_seconds'] for j in jobs)
    result={'observed_unix':time.time(),'recorded_accelerator_job_seconds':accelerator,
        'completed_unique_fit_scoring_seconds':sum(r.get('training_scoring_seconds',0) for r in complete.values()),
        'completed_unique_learning_runs':len(complete),'jobs':jobs,
        'ceiling_seconds':48*3600,'recorded_seconds_below_ceiling':accelerator<48*3600,
        'budget_boundary':'Process wall for local accelerator learning/scoring/profiling, not active device utilization. Nested times are descriptive and not summed again.',
        'limitations':['A still-live child is excluded until its parent records terminal wall; this is an as-of ledger, not a live remaining-budget guarantee.',
            'The first three timing refusals occurred before optimizer steps; their wall was not preserved separately after correction and remains unquantified, not zero.',
            'Initial direct guarded dispatch records start after process imports/setup; their small unrecorded startup/teardown is excluded.',
            'CPU preparation/checks/analysis are separate; see execution log and check artifacts. Vendor attempts remain zero.'],
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (DEST/'runtime_ledger.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('recorded_accelerator_job_seconds','completed_unique_fit_scoring_seconds','completed_unique_learning_runs','recorded_seconds_below_ceiling')}))
if __name__=='__main__':main()
