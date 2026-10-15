"""Serial bounded fit dispatch; shared GPU lock and one log per run."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from resource_lock import ResourceLock


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--canonical-root',required=True)
    p.add_argument('--run-ids',nargs='+',required=True);p.add_argument('--max-seconds-per-run',type=int,default=1800)
    a=p.parse_args();canonical=Path(a.canonical_root);out=canonical/'results/kd_cnn_v2';shared=canonical/'results/dual_track_v1/shared'
    cfg=json.loads(Path(a.config).read_text(encoding='utf-8'));known={r['run_id'] for r in cfg['recipes']}
    if not set(a.run_ids)<=known:raise ValueError('Unknown run')
    if len(set(a.run_ids))!=len(a.run_ids):raise ValueError('Duplicate run')
    dispatch=out/'dispatch';dispatch.mkdir(parents=True,exist_ok=True)
    for ident in a.run_ids:
        if (out/'stop.request').exists():raise RuntimeError('Study stop requested')
        args=[sys.executable,'-B',str(Path(__file__).with_name('train.py')),'--config',str(Path(a.config).resolve()),'--run-id',ident,
              '--canonical-root',str(canonical),'--output-root',str(out),'--device','cuda','--max-seconds',str(a.max_seconds_per_run)]
        attempt=1
        while (dispatch/f'{ident}_attempt{attempt}.json').exists():attempt+=1
        record=dispatch/f'{ident}_attempt{attempt}.json';log=record.with_suffix('.log')
        with ResourceLock(shared,'gpu0') as lock:
            value={**lock.owner,'command':args,'state':'running','log':str(log)};record.write_text(json.dumps(value,indent=2),encoding='utf-8')
            start=time.monotonic()
            with log.open('x',encoding='utf-8') as handle:
                result=subprocess.run(args,stdout=handle,stderr=subprocess.STDOUT,check=False)
            value.update(exit_code=result.returncode,wall_seconds=time.monotonic()-start,state='completed' if result.returncode==0 else 'failed')
            record.write_text(json.dumps(value,indent=2),encoding='utf-8')
        print(json.dumps({'run_id':ident,'exit_code':result.returncode,'wall_seconds':value['wall_seconds'],'log':str(log)}),flush=True)
        if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':main()
