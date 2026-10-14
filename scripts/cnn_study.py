"""Serial, resumable CNN experiment queue; only explicitly permitted prepared folds."""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, time, traceback
from pathlib import Path
import numpy as np
os.environ.setdefault('TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT','64')
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gate_eval import fit_threshold, evaluate_documents
from cnn_models import build_model, load_search_configs
from cnn_artifacts import validate_prepared

DEST = ROOT / 'results/cnn_study_v1'
ALLOWED = {'screen_fit','fit','inner_calibration','inner_selection','full_train',
           'search_temporary_threshold','search_candidate_scoring','diagnostics'}

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    os.replace(temp,path)

def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''): h.update(part)
    return h.hexdigest()

def save_torch(path,value):
    temp=path.with_suffix(path.suffix+'.tmp'); torch.save(value,temp); os.replace(temp,path)

def setup(seed):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)

class Budget:
    def __init__(self, limit):
        from r1_challengers import resource_check
        self.check_memory=resource_check
        self.started=time.monotonic(); self.limit=limit; self.last=self.started
        self.observations=[resource_check(True)]
    def check(self):
        if (DEST/'stop.request').exists(): raise RuntimeError('CNN study stop requested')
        now=time.monotonic()
        if now-self.started>self.limit: raise RuntimeError('Dispatch wall-time ceiling exhausted')
        if now-self.last>=60:
            self.observations.append(self.check_memory()); self.last=now

def data(rep, split):
    if split not in ALLOWED: raise ValueError('Unapproved fold')
    p=DEST/'prepared'/rep/split
    a={n:np.load(p/(n+'.npy'),allow_pickle=False) for n in ('ids','masks','labels','starts')}
    if np.any(a['labels']<0): raise ValueError('Unresolved window labels require document loss')
    a['documents']=[json.loads(s) for s in (p/'documents.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(a['documents']) != len(a['starts'])-1: raise ValueError('Document boundaries differ')
    a['x']=torch.as_tensor(a['ids'].astype(np.int64),device='cuda')
    a['mask']=torch.as_tensor(a['masks'].astype(np.int64),device='cuda')
    a['y']=torch.as_tensor(a['labels'],dtype=torch.long,device='cuda')
    return a

def model_for(c):
    rep=c['representation']
    return build_model(c,257 if rep=='byte' else 30522,0).cuda()

def score(model,a,budget,batch=256):
    model.eval(); risks=[]
    with torch.inference_mode():
        for at in range(0,len(a['x']),batch):
            budget.check()
            logits=model(a['x'][at:at+batch],a['mask'][at:at+batch]).logits
            if not torch.isfinite(logits).all(): raise RuntimeError('Nonfinite logits')
            risks.extend(torch.softmax(logits,dim=1)[:,1].cpu().tolist())
    rows=[]
    for i,d in enumerate(a['documents']):
        start,end=map(int,a['starts'][i:i+2])
        rows.append({**d,'risk_score':max(risks[start:end]),'window_scores':risks[start:end]})
    return rows

def metrics(cal,rows):
    benign=[r['risk_score'] for r in cal if r['label']==0]
    result={}
    for alpha in (.005,.01,.02):
        t=fit_threshold(benign,alpha)
        m=evaluate_documents([r['risk_score'] for r in rows],[r['label'] for r in rows],t)
        m['passes']=m['recall'] is not None and m['fpr'] is not None and m['recall']>=.99 and m['fpr']<=.01
        result[str(alpha)]=m
    return result

def selection_key(m):
    # Prioritize the fixed recall floor, then lowest benign rejection rate.
    return (m['recall']<.99, m['fpr'] if m['recall']>=.99 else -m['recall'],
            -m['recall'] if m['recall']>=.99 else m['fpr'])

def train_one(c,spec,budget,cache):
    seed=spec.get('seed',42); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rep=c['representation']; splits=(spec['fit_split'],spec['cal_split'],spec['selection_split'])
    if spec['fit_split'] not in {'screen_fit','fit','full_train'}:raise ValueError('Calibration/selection data cannot fit weights')
    if splits[1:] != ('inner_calibration','inner_selection'):raise ValueError('Training queue must use inner calibration/selection')
    if spec['fit_split']=='full_train' and spec.get('select_epoch'):raise ValueError('Frozen finalists require fixed epoch')
    fitting_only=spec['fit_split']=='full_train'
    needed=splits[:1] if fitting_only else splits
    for split in needed:
        if (rep,split) not in cache: cache[(rep,split)]=data(rep,split)
    a=cache[(rep,splits[0])]
    cal,sel=(None,None) if fitting_only else [cache[(rep,s)] for s in splits[1:]]
    folder=DEST/'runs'/spec['run_id']; folder.mkdir(parents=True,exist_ok=True)
    identity={'config':c,'protocol_sha256':sha(DEST/'prepared/data_protocol.json'),
              'representation_sha256':sha(DEST/'prepared'/rep/'summary.json')}
    if (folder/'run.json').exists():
        previous=read(folder/'run.json')
        if previous['spec']!=spec or previous['config']!=c: raise ValueError('Existing run has different frozen specification/config')
        if previous['prepared_protocol_sha256']!=identity['protocol_sha256'] or previous['prepared_representation_sha256']!=identity['representation_sha256']:
            raise ValueError('Prepared identities changed')
    if (folder/'complete.json').exists():
        done=read(folder/'complete.json')
        if sha(ROOT/done['checkpoint'])!=done['checkpoint_sha256']: raise ValueError('Completed checkpoint changed')
        return done
    torch.cuda.reset_peak_memory_stats()
    model=model_for(c); opt=torch.optim.AdamW(model.parameters(),lr=spec.get('learning_rate',c['training']['learning_rate']),weight_decay=c['training'].get('weight_decay',.01))
    start_epoch=0; history=[]; prior_seconds=0; best_key=None; best_epoch=None
    latest=folder/'checkpoint_latest.pt'
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['state_dict']); opt.load_state_dict(saved['optimizer'])
        start_epoch=saved['epoch']; history=saved['history']; prior_seconds=saved['training_seconds']
        best_key=saved['best_key']; best_epoch=saved['best_epoch']
        torch.set_rng_state(saved['rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    write(folder/'run.json',{'spec':spec,'config':c,'parameters':sum(p.numel() for p in model.parameters()),
        'loss':'Span-window CE averaged within each document then across document batch',
        'batch_documents':32,'microbatch_windows':128,'arithmetic':'native CUDA FP32; TF32 disabled',
        'optimizer_defaults':opt.defaults,
        'torch':torch.__version__,'python':sys.version,'device':torch.cuda.get_device_name(),
        'cudnn_plan_cache_limit':os.environ.get('TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT','default'),
        'cudnn_v8_api_disabled':os.environ.get('TORCH_CUDNN_V8_API_DISABLED','0'),
        'prepared_protocol_sha256':sha(DEST/'prepared/data_protocol.json'),
        'prepared_representation_sha256':sha(DEST/'prepared'/rep/'summary.json'),
        'model_source_sha256':sha(ROOT/'scripts/cnn_models.py'),
        'runner_source_sha256':sha(Path(__file__)),
        'selection':'fixed final epoch' if not spec.get('select_epoch') else 'recall floor then lowest inner-selection FPR; inner-calibration benign thresholds',
        'training_evaluation_boundary':'No in-fit source scoring for frozen full-training fits; quality evaluated separately on prescribed development partitions' if fitting_only else 'Unexposed inner source groups used for development selection',
        'command':sys.argv})
    started=time.perf_counter()
    n=len(a['starts'])-1
    for epoch in range(start_epoch+1,spec['epochs']+1):
        model.train(); epoch_start=time.perf_counter(); total=0
        order=np.random.default_rng(seed+epoch).permutation(n)
        for at in range(0,n,32):
            budget.check(); docs=order[at:at+32]; idx=[]; weights=[]
            for d in docs:
                lo,hi=map(int,a['starts'][d:d+2]); idx.extend(range(lo,hi)); weights.extend([1/(len(docs)*(hi-lo))]*(hi-lo))
            opt.zero_grad(set_to_none=True); value=0
            for pos in range(0,len(idx),128):
                ii=torch.tensor(idx[pos:pos+128],device='cuda')
                logits=model(a['x'][ii],a['mask'][ii]).logits
                loss=(F.cross_entropy(logits,a['y'][ii],reduction='none')*torch.tensor(weights[pos:pos+128],device='cuda')).sum()
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                loss.backward(); value+=loss.item()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            if not torch.isfinite(norm): raise RuntimeError('Nonfinite gradient')
            opt.step(); total+=value*len(docs)
        rec={'epoch':epoch,'loss':total/n,'fit_seconds':time.perf_counter()-epoch_start}
        if not fitting_only and (spec.get('select_epoch') or epoch==spec['epochs']):
            m=metrics(score(model,cal,budget),score(model,sel,budget)); rec['selection_metrics']=m
            key=selection_key(m['0.01'])
            if best_key is None or key<tuple(best_key):
                best_key=key; best_epoch=epoch
                save_torch(folder/'checkpoint_selected.pt',{'state_dict':model.state_dict(),'epoch':epoch,'config':c})
        history.append(rec)
        save_torch(latest,{'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'history':history,
            'training_seconds':prior_seconds+time.perf_counter()-started,'best_key':best_key,'best_epoch':best_epoch,
            'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()})
        write(folder/'history.json',history)
        print(json.dumps({'event':'epoch','run':spec['run_id'],**rec}),flush=True)
    selected=folder/('checkpoint_selected.pt' if spec.get('select_epoch') else 'checkpoint_latest.pt')
    if spec.get('select_epoch'): model.load_state_dict(torch.load(selected,map_location='cpu',weights_only=False)['state_dict'])
    cal_rows=[] if fitting_only else score(model,cal,budget)
    sel_rows=[] if fitting_only else score(model,sel,budget)
    for name,rows in [('calibration',cal_rows),('selection',sel_rows)]:
        with (folder/(name+'_scores.jsonl')).open('w',encoding='utf-8') as f:
            for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    done={'complete':True,'run_id':spec['run_id'],'config_id':c['id'],'family':c['family'],'representation':rep,
        'seed':seed,'epochs':spec['epochs'],'selected_epoch':best_epoch if spec.get('select_epoch') else spec['epochs'],
        'checkpoint':str(selected.relative_to(ROOT)),'checkpoint_sha256':sha(selected),
        'training_scoring_seconds':prior_seconds+time.perf_counter()-started,
        'metrics':{} if fitting_only else metrics(cal_rows,sel_rows),'parameters':sum(p.numel() for p in model.parameters()),
        'quality_evidence':'Pending separate frozen confirmation' if fitting_only else 'Inner development selection only',
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
    write(folder/'complete.json',done)
    del model,opt
    return done

def queue(args):
    plan=read(Path(args.plan)); configs={c['id']:c for c in load_search_configs()}; cache={}; records=[]
    for c in plan.get('extra_configs',[]):
        if c['id'] in configs and configs[c['id']]!=c:raise ValueError('Extra config overwrites a registry identity')
        configs[c['id']]=c
    if plan.get('finalist_configs'):
        if sha(DEST/'prepared/data_protocol.json')!=plan['data_protocol_sha256'] or sha(ROOT/'scripts/cnn_models.py')!=plan['model_source_sha256'] or sha(DEST/'prepared/diagnostics_raw.jsonl')!=plan['diagnostics_sha256']:raise ValueError('Frozen model/data/probes changed')
        if any(configs[cid]!=c for cid,c in plan['finalist_configs'].items()):raise ValueError('Frozen finalist configuration changed')
        if any(sha(DEST/'prepared'/rep/'summary.json')!=h for rep,h in plan['prepared_representation_sha256'].items()):raise ValueError('Frozen prepared representation changed')
        for rep in {c['representation'] for c in plan['finalist_configs'].values()}:
            validate_prepared(rep,('full_train',))
    budget=Budget(args.max_seconds)
    for spec in plan['runs']:
        try: done=train_one(configs[spec['config_id']],spec,budget,cache); records.append(done)
        except Exception as e:
            failure={'run_id':spec['run_id'],'complete':False,'error':str(e),'traceback':traceback.format_exc()}
            write(DEST/'runs'/spec['run_id']/'failure.json',failure); records.append(failure)
            print(json.dumps(failure),flush=True)
            if 'stop requested' in str(e) or 'ceiling' in str(e) or 'Resource floor' in str(e):
                write(DEST/(plan['stage']+'_run_table.json'),records); break
        write(DEST/(plan['stage']+'_run_table.json'),records)
    write(DEST/(plan['stage']+'_dispatch.json'),{'plan_sha256':sha(Path(args.plan)),'command':sys.argv,
        'wall_seconds':time.monotonic()-budget.started,'memory_observations':budget.observations,'run_count':len(records)})
    print(json.dumps({'event':'queue_complete','stage':plan['stage'],'runs':len(records)}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--plan',required=True); p.add_argument('--max-seconds',type=int,default=14400)
    args=p.parse_args(); setup(42); queue(args)
if __name__=='__main__': main()
