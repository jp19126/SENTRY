"""Bounded DP03 CUDA cache64 versus cache8 arithmetic/memory diagnostic.

Reuses the existing real-fit forward/full-gradient worker in fresh processes.
No optimizer, learning, scoring folds, registry edits or floor relaxation.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import check_cnn_cache as base

ROOT=base.ROOT;DEST=base.DEST
ART=DEST/'pyramid_cache8_artifacts'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--child',choices=('worker','compare'));p.add_argument('--folder',type=Path);p.add_argument('--worker-seconds',type=int,default=45);a=p.parse_args()
    if a.child=='worker':
        base.CONFIG_IDS=('dpcnn_03',);base.BATCH_SIZES=(32,45)
        base.worker(a);return
    if a.child=='compare':
        import numpy as np
        left,right=[json.loads((ART/m/'manifest.json').read_text(encoding='utf-8')) for m in ('limit64','limit8')]
        if not left['complete'] or not right['complete']:raise ValueError('Incomplete cache diagnostic')
        checks=[]
        for x,y in zip(left['results'],right['results'],strict=True):
            if (x['config_id'],x['batch_size'],x['input_identity'],x['parameter_names'])!=(y['config_id'],y['batch_size'],y['input_identity'],y['parameter_names']):raise ValueError('Different cache comparison inputs')
            deltas={}
            with np.load(ROOT/x['tensor_path']) as xx,np.load(ROOT/y['tensor_path']) as yy:
                if set(xx.files)!=set(yy.files):raise ValueError('Different saved tensors')
                for key in xx.files:
                    if not np.array_equal(xx[key],yy[key]):deltas[key]=float(np.max(np.abs(xx[key].astype(np.float64)-yy[key].astype(np.float64))))
            checks.append({'config_id':x['config_id'],'batch_size':x['batch_size'],'all_tensors_bitwise_equal':not deltas,
                'max_absolute_differences':deltas,'limit64_memory':x['memory'],'limit8_memory':y['memory']})
        result={'complete':True,'all_tensors_bitwise_equal':all(c['all_tensors_bitwise_equal'] for c in checks),'checks':checks,'children':[left,right],
            'source_sha256':base.digest(Path(__file__)),'worker_source_sha256':base.digest(Path(base.__file__)),
            'scope':'Two DP03 same-weight real-fit B32/B45 forward/CE/full gradients, dropout disabled. No optimizer or quality evidence; not all possible shapes.',
            'memory_boundary':'Fresh-process working set/physical availability; no causal attribution of all host growth to cuDNN cache'}
        base.save_json(DEST/'pyramid_cache8_check.json',result);print(json.dumps({'complete':True,'all_tensors_bitwise_equal':result['all_tensors_bitwise_equal']}))
        if not result['all_tensors_bitwise_equal']:raise SystemExit(1)
        return
    started=time.monotonic();logs=[]
    try:
        for cache in (64,8):
            folder=ART/('limit'+str(cache));folder.mkdir(parents=True,exist_ok=True)
            env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=str(cache))
            command=[sys.executable,'-B',str(Path(__file__).resolve()),'--child','worker','--folder',str(folder),'--worker-seconds','45']
            r=subprocess.run(command,env=env,capture_output=True,text=True,timeout=max(1,120-(time.monotonic()-started)),creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            logs.append({'cache':cache,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'command':command})
            if r.returncode:raise RuntimeError('Cache child failed; preserved dispatch and partial manifest')
        r=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--child','compare'],capture_output=True,text=True,timeout=max(1,120-(time.monotonic()-started)),creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        logs.append({'phase':'cpu_comparison','returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
        if r.returncode:raise RuntimeError('Cache arithmetic comparison failed')
        print(r.stdout.strip())
    finally:
        base.save_json(DEST/'pyramid_cache8_dispatch.json',{'wall_seconds':time.monotonic()-started,'command':sys.argv,'children':logs})
if __name__=='__main__':main()
