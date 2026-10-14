"""Bounded same-input DP03 cuDNN v8/v7 CUDA arithmetic diagnostic."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import check_cnn_cache as base
DEST=base.DEST;ROOT=base.ROOT;ART=DEST/'pyramid_backend_artifacts'

def main():
    p=argparse.ArgumentParser();p.add_argument('--child',choices=('worker','compare'));p.add_argument('--folder',type=Path);p.add_argument('--worker-seconds',type=int,default=45);a=p.parse_args()
    if a.child=='worker':
        base.CONFIG_IDS=('dpcnn_03',);base.BATCH_SIZES=(32,45);base.worker(a)
        path=a.folder/'manifest.json';v=json.loads(path.read_text(encoding='utf-8'));v['cudnn_v8_api_disabled']=os.environ['TORCH_CUDNN_V8_API_DISABLED'];base.save_json(path,v);return
    if a.child=='compare':
        import numpy as np
        children=[json.loads((ART/m/'manifest.json').read_text(encoding='utf-8')) for m in ('v8','v7')];checks=[]
        if not all(v['complete'] for v in children):raise ValueError('Incomplete backend case')
        for x,y in zip(children[0]['results'],children[1]['results'],strict=True):
            if (x['config_id'],x['batch_size'],x['input_identity'],x['parameter_names'])!=(y['config_id'],y['batch_size'],y['input_identity'],y['parameter_names']):raise ValueError('Backend inputs differ')
            comparisons={}
            with np.load(ROOT/x['tensor_path']) as xx,np.load(ROOT/y['tensor_path']) as yy:
                for k in xx.files:
                    if xx[k].shape!=yy[k].shape:raise ValueError('Tensor shapes differ')
                    delta=xx[k].astype(np.float64)-yy[k].astype(np.float64);absolute=float(np.max(np.abs(delta)))
                    norm=float(np.linalg.norm(xx[k].astype(np.float64).ravel()));relative=float(np.linalg.norm(delta.ravel())/max(norm,1e-12))
                    passed=absolute<=1e-5 if k in ('logits','loss') else (absolute<=1e-6 or relative<=1e-4)
                    comparisons[k]={'max_absolute_delta':absolute,'relative_l2_delta':relative,'bitwise_equal':bool(np.array_equal(xx[k],yy[k])),'passes':passed}
            checks.append({'batch_size':x['batch_size'],'tensor_comparisons':comparisons,'passes':all(v['passes'] for v in comparisons.values()),'v8_memory':x['memory'],'v7_memory':y['memory']})
        result={'complete':True,'passed':all(c['passes'] for c in checks),'checks':checks,'children':children,
            'prospective_tolerances':'logit/loss maximum absolute <=1e-5; each full gradient absolute <=1e-6 or relative L2 <=1e-4. Finite/pad-gradient checks in shared worker.',
            'scope':'Same weights and two permitted real-fit B32/B45 cases, dropout off, CUDA FP32/TF32off/deterministic; no optimizer or quality evidence. Not bitwise equivalence on all possible shapes.',
            'source_sha256':base.digest(Path(__file__)),'worker_source_sha256':base.digest(Path(base.__file__))}
        base.save_json(DEST/'pyramid_backend_check.json',result);print(json.dumps({'complete':True,'passed':result['passed']}))
        if not result['passed']:raise SystemExit(1)
        return
    start=time.monotonic();logs=[]
    try:
        for name,disabled in (('v8','0'),('v7','1')):
            folder=ART/name;folder.mkdir(parents=True,exist_ok=True)
            env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT='8',TORCH_CUDNN_V8_API_DISABLED=disabled)
            cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--child','worker','--folder',str(folder)]
            r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=max(1,120-(time.monotonic()-start)),creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            logs.append({'backend':name,'returncode':r.returncode,'command':cmd,'stdout':r.stdout,'stderr':r.stderr})
            if r.returncode:raise RuntimeError('Backend worker failed')
        r=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--child','compare'],capture_output=True,text=True,timeout=max(1,120-(time.monotonic()-start)),creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        logs.append({'phase':'cpu_comparison','returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
        if r.returncode:raise RuntimeError('Backend numerical qualification failed')
        print(r.stdout.strip())
    finally:base.save_json(DEST/'pyramid_backend_dispatch.json',{'wall_seconds':time.monotonic()-start,'command':sys.argv,'children':logs})
if __name__=='__main__':main()
