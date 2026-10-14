"""Bounded serial subprocess groups release native caches between CNN jobs."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'results/cnn_study_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');os.replace(tmp,p)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--group-size',type=int,default=4);p.add_argument('--max-seconds',type=int,required=True);p.add_argument('--cache-override',action='append',default=[]);p.add_argument('--v7-config',action='append',default=[]);args=p.parse_args()
    plan_path=Path(args.plan);plan=read(plan_path);stage=plan['stage'];started=time.monotonic();dispatches=[]
    configs={c['id']:c for c in read(ROOT/'configs/cnn_search_v1.json')['configurations']}
    configs.update({c['id']:c for c in plan.get('extra_configs',[])})
    def completed(spec):
        folder=DEST/'runs'/spec['run_id'];f=folder/'complete.json'
        if not f.exists():return False
        m=read(folder/'run.json');d=read(f)
        if m['spec']!=spec or m['config']!=configs[spec['config_id']]:raise ValueError('Completed job specification changed')
        cp=ROOT/d['checkpoint']
        if hashlib.sha256(cp.read_bytes()).hexdigest()!=d['checkpoint_sha256']:raise ValueError('Completed checkpoint changed')
        return True
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT='64')
    limits={cid:int(value) for cid,value in plan.get('cudnn_cache_limits',{}).items()}
    v7=set(cid for cid,value in plan.get('cudnn_v8_disabled',{}).items() if int(value))|set(args.v7_config)
    if not v7<=set(configs):raise ValueError('Unknown v7 configuration')
    for item in args.cache_override:
        cid,value=item.split('=',1)
        if cid not in configs or not 1<=int(value)<=64:raise ValueError('Invalid cache override')
        limits[cid]=int(value)
    serial=1
    while (DEST/f'{stage}_queue_{serial}.json').exists():serial+=1
    for at in range(0,len(plan['runs']),args.group_size):
        if (DEST/'stop.request').exists():break
        group=plan['runs'][at:at+args.group_size]
        if all(completed(s) for s in group):continue
        remaining=args.max_seconds-(time.monotonic()-started)
        if remaining<=0:break
        name=f'{stage}_q{serial}_group{at//args.group_size:03d}'
        chunk=DEST/'dispatch_plans'/(name+'.json');write(chunk,{**plan,'stage':name,'runs':group})
        log=DEST/'dispatch_logs'/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u',str(ROOT/'scripts/cnn_study.py'),'--plan',str(chunk),'--max-seconds',str(int(remaining))]
        group_limits={limits.get(s['config_id'],64) for s in group}
        if len(group_limits)!=1:raise ValueError('Use isolated jobs for different cache limits')
        job_env=dict(env,TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=str(next(iter(group_limits))))
        group_api={int(s['config_id'] in v7) for s in group}
        if len(group_api)!=1:raise ValueError('Use isolated jobs for different cuDNN APIs')
        job_env['TORCH_CUDNN_V8_API_DISABLED']=str(next(iter(group_api)))
        job_start=time.monotonic()
        with log.open('w',encoding='utf-8') as out:
            proc=subprocess.Popen(command,cwd=ROOT,env=job_env,stdout=out,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            write(DEST/(stage+'_process.json'),{'pid':proc.pid,'parent_pid':os.getpid(),'command':command,'state':'running','started_unix':time.time(),'log':str(log.relative_to(ROOT))})
            proc.wait()
        write(DEST/(stage+'_process.json'),{'pid':proc.pid,'parent_pid':os.getpid(),'command':command,'state':'terminal','returncode':proc.returncode,'finished_unix':time.time(),'log':str(log.relative_to(ROOT))})
        rec={'command':command,'returncode':proc.returncode,'wall_seconds':time.monotonic()-job_start,'log':str(log.relative_to(ROOT)),
             'cudnn_plan_cache_limit':job_env['TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT']};dispatches.append(rec)
        rec['cudnn_v8_api_disabled']=job_env['TORCH_CUDNN_V8_API_DISABLED']
        records=[]
        for spec in plan['runs']:
            f=DEST/'runs'/spec['run_id']/'complete.json';fail=f.with_name('failure.json')
            if f.exists():records.append(read(f))
            elif fail.exists():records.append(read(fail))
        write(DEST/(stage+'_run_table.json'),records)
        write(DEST/f'{stage}_queue_{serial}.json',{'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),'wall_seconds':time.monotonic()-started,'dispatches':dispatches})
        print(json.dumps({'group':name,'returncode':proc.returncode,'seconds':rec['wall_seconds'],'complete':sum(r['complete'] for r in records),'attempted':len(records)}),flush=True)
        if proc.returncode or any(not completed(s) for s in group):
            print('Group incomplete; preserved records, inspect before resume',flush=True);break
    print(json.dumps({'event':'serial_queue_terminal','stage':stage,'wall_seconds':time.monotonic()-started}),flush=True)
if __name__=='__main__':main()
