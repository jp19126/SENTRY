"""Serial frozen confirmation with immutable attempts and cumulative child walls."""
import argparse
from datetime import datetime, timezone
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'results/cnn_study_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');os.replace(tmp,p)
def write_new(p,v):
    with p.open('x',encoding='utf-8') as f:f.write(json.dumps(v,indent=2)+'\n')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def saved_confirmation(spec,freeze,h):
    folder=DEST/'confirmation'/spec['run_id'];path=folder/'complete.json'
    # Windows UNC sandbox lookup can report access denied for a missing ancestor.
    if not (DEST/'confirmation').exists() or not folder.exists():return None
    if not path.exists():return None
    saved=read(path)
    if saved.get('complete') is not True or saved['run_id']!=spec['run_id'] or saved['seed']!=spec['seed']:raise ValueError('Saved confirmation has another run/seed or is incomplete')
    if saved['freeze_sha256']!=h:raise ValueError('Old confirmation belongs to another freeze')
    run=read(DEST/'runs'/spec['run_id']/'complete.json');meta=read(DEST/'runs'/spec['run_id']/'run.json')
    if run.get('complete') is not True or meta['spec']!=spec or run['seed']!=spec['seed'] or run['epochs']!=spec['epochs']:raise ValueError('Saved training completion/recipe differs from freeze')
    cp=digest(ROOT/run['checkpoint'])
    if cp!=saved['checkpoint_sha256'] or cp!=run['checkpoint_sha256']:raise ValueError('Saved checkpoint changed')
    if saved['config']!=freeze['finalist_configs'][spec['config_id']] or meta['config']!=saved['config']:raise ValueError('Saved config changed')
    if any(int(r['cudnn_plan_cache_limit'])!=freeze['cudnn_cache_limits'][spec['config_id']] or int(r.get('cudnn_v8_api_disabled','0'))!=freeze['cudnn_v8_disabled'][spec['config_id']] for r in (saved,meta)):raise ValueError('Saved CUDA backend/cache differs from freeze')
    variants={'fp32'}|({'fake_w8a8'} if spec['config_id'] in freeze['quantization_config_ids'] else set())
    if set(saved['variants'])!=variants:raise ValueError('Saved arithmetic variants differ from freeze')
    for variant,v in saved['variants'].items():
        if v['scoring_batch_windows']!=1:raise ValueError('Saved quality is not batch one')
        for split,key in (('calibration','calibration_score_sha256'),('development','development_score_sha256'),('diagnostics','diagnostics_score_sha256')):
            if digest(folder/(variant+'_'+split+'.jsonl'))!=v[key]:raise ValueError('Saved raw scores changed')
    if bool(saved.get('timing'))!=bool(spec.get('time',False)):raise ValueError('Saved timing coverage differs from freeze')
    if saved.get('timing') and saved['timing']['backend']!='CUDA eager native FP32; TF32 disabled; B1 windows serial':raise ValueError('Saved timing backend differs from freeze')
    return saved

def execute(frozen_path,max_seconds):
    frozen_path=Path(frozen_path).resolve();freeze=read(frozen_path);h=digest(frozen_path)
    expected=[s['run_id'] for s in freeze['runs']]
    if not expected or len(set(expected))!=len(expected):raise ValueError('Distinct nonempty frozen run IDs required')
    if digest(ROOT/'scripts/cnn_models.py')!=freeze['model_source_sha256'] or digest(DEST/'prepared/data_protocol.json')!=freeze['data_protocol_sha256'] or digest(DEST/'prepared/diagnostics_raw.jsonl')!=freeze['diagnostics_sha256']:raise ValueError('Frozen source/data/probes changed')
    ledger_path=DEST/'confirmation_queue.json';ledger=read(ledger_path) if ledger_path.exists() else {'freeze_sha256':h,'jobs':[],'attempts':[]}
    if ledger['freeze_sha256']!=h:raise ValueError('Existing ledger belongs to another freeze')
    if ledger_path.exists() and ledger.get('schema_version')!=2:
        snapshot=DEST/('confirmation_queue_legacy_'+digest(ledger_path)[:16]+'.json')
        if not snapshot.exists():
            with snapshot.open('xb') as f:f.write(ledger_path.read_bytes())
    ledger.setdefault('attempts',[])
    for i,j in enumerate(ledger['jobs']):j.setdefault('job_id',f'legacy_{i:04d}');j.setdefault('attempt_id','legacy')
    if len({j['job_id'] for j in ledger['jobs']})!=len(ledger['jobs']):raise ValueError('Duplicate cumulative child job IDs')
    parent=DEST/'confirmation_attempts';parent.mkdir(parents=True,exist_ok=True);number=1
    while True:
        attempt=parent/f'{number:04d}'
        try:attempt.mkdir();break
        except FileExistsError:number+=1
    aid=attempt.name;started=time.monotonic()
    write_new(attempt/'started.json',{'attempt_id':aid,'freeze_sha256':h,'expected_run_ids':expected,'max_seconds':max_seconds,'started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':digest(Path(__file__)),'command':sys.argv})
    ledger.update(schema_version=2,active_attempt_id=aid);reason=error=None;child_exit=0;job_ids=[]
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
    def save_ledger(status='running'):
        ledger.update(status=status,freeze_sha256=h,wall_seconds=sum(j['wall_seconds'] for j in ledger['jobs']));write(ledger_path,ledger)
    save_ledger()
    try:
        for spec in freeze['runs']:
            if saved_confirmation(spec,freeze,h) is not None:continue
            if (DEST/'stop.request').exists():reason='stop_requested';break
            remaining=max_seconds-(time.monotonic()-started)
            if remaining<1:reason='time_limit';break
            job_env=dict(env,TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=str(freeze['cudnn_cache_limits'][spec['config_id']]),TORCH_CUDNN_V8_API_DISABLED=str(freeze['cudnn_v8_disabled'][spec['config_id']]))
            log=attempt/(spec['run_id']+'.log');command=[sys.executable,'-u',str(ROOT/'scripts/cnn_evaluate.py'),'--freeze',str(frozen_path),'--run-id',spec['run_id'],'--max-seconds',str(int(remaining))]
            t=time.monotonic();returncode=None;launch_error=None
            try:
                with log.open('x',encoding='utf-8') as f:r=subprocess.run(command,cwd=ROOT,env=job_env,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                returncode=r.returncode
            except (Exception,KeyboardInterrupt) as exc:
                launch_error=str(exc) or type(exc).__name__;returncode=130 if isinstance(exc,KeyboardInterrupt) else 1
            job={'job_id':aid+':'+spec['run_id'],'attempt_id':aid,'run_id':spec['run_id'],'returncode':returncode,'wall_seconds':time.monotonic()-t,'command':command,'log':str(log.relative_to(ROOT)),'log_sha256':digest(log) if log.exists() else None,'error':launch_error}
            ledger['jobs'].append(job);job_ids.append(job['job_id']);write_new(attempt/(spec['run_id']+'_job.json'),job);save_ledger()
            if returncode:child_exit=returncode if 0<returncode<256 else 1;reason='child_failed';error=launch_error;break
            if saved_confirmation(spec,freeze,h) is None:raise ValueError('Child exited zero without saved confirmation')
            print(json.dumps({'confirmation_complete':spec['run_id'],'attempt_id':aid}),flush=True)
    except (Exception,KeyboardInterrupt) as exc:
        reason='validation_or_queue_error';error=str(exc) or type(exc).__name__;child_exit=130 if isinstance(exc,KeyboardInterrupt) else 1
    results=[];invalid={}
    for spec in freeze['runs']:
        try:
            saved=saved_confirmation(spec,freeze,h)
            if saved is not None:results.append(saved)
        except Exception as exc:invalid[spec['run_id']]=str(exc)
    missing=[rid for rid in expected if rid not in {r['run_id'] for r in results}]
    complete=not missing and not invalid and not child_exit;exit_code=0 if complete else child_exit or (1 if invalid else 2);status='complete' if complete else 'incomplete'
    terminal={'attempt_id':aid,'freeze_sha256':h,'status':status,'complete':complete,'exit_code':exit_code,'reason':reason or ('all_frozen_runs_verified' if complete else 'missing_confirmations'),'error':error,'expected_run_ids':expected,'verified_run_ids':[r['run_id'] for r in results],'missing_run_ids':missing,'invalid_saved_runs':invalid,'job_ids':job_ids,'attempt_wall_seconds':time.monotonic()-started,'terminal_utc':datetime.now(timezone.utc).isoformat()}
    write_new(attempt/'terminal.json',terminal);ledger['attempts'].append({'attempt_id':aid,'status':status,'exit_code':exit_code,'terminal':str((attempt/'terminal.json').relative_to(ROOT))})
    ledger.update(complete=complete,exit_code=exit_code,missing_run_ids=missing,invalid_saved_runs=invalid,active_attempt_id=None,latest_attempt_id=aid);save_ledger(status)
    write(DEST/'confirmation_table.json',results);print(json.dumps(terminal),flush=True);return exit_code

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',required=True);p.add_argument('--max-seconds',type=int,default=10800);a=p.parse_args();return execute(a.freeze,a.max_seconds)
if __name__=='__main__':raise SystemExit(main())
