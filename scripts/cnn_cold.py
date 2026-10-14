"""One process-start-to-document-decision observation per frozen CNN architecture."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'results/cnn_study_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze');p.add_argument('--child');args=p.parse_args()
    if args.child:
        started=time.perf_counter()
        sys.path.insert(0,str(ROOT));import torch
        from scripts.cnn_models import build_model
        from gate_eval import load_config,window_document
        sys.path.insert(0,str(ROOT/'scripts'));from r1_challengers import tokenizer
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        from cnn_evaluate import validate_freeze,validate_saved,validate_preprocessing_policy,sha
        imports_finished=time.perf_counter()
        freeze=read(Path(args.freeze));validate_freeze(freeze)
        meta=read(DEST/'runs'/args.child/'run.json');done=read(DEST/'runs'/args.child/'complete.json');c=meta['config']
        spec=next(s for s in freeze['runs'] if s['run_id']==args.child)
        if meta['spec']!=spec:raise ValueError('Cold timing training recipe differs from freeze')
        if c!=freeze['finalist_configs'][c['id']]:raise ValueError('Cold timing config differs from freeze')
        for field,environment,expected in (
            ('cudnn_plan_cache_limit','TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT',freeze['cudnn_cache_limits'][c['id']]),
            ('cudnn_v8_api_disabled','TORCH_CUDNN_V8_API_DISABLED',freeze['cudnn_v8_disabled'][c['id']])):
            if int(meta.get(field,'0'))!=expected or int(os.environ[environment])!=expected:raise ValueError('Cold timing backend/cache differs from freeze')
        validate_saved(DEST/'confirmation'/args.child,read(DEST/'confirmation'/args.child/'complete.json'),sha(Path(args.freeze)))
        load_at=time.perf_counter()
        model=build_model(c,257 if c['representation']=='byte' else 30522,0).cuda().eval()
        model.load_state_dict(torch.load(ROOT/done['checkpoint'],map_location='cpu',weights_only=False)['state_dict']);torch.cuda.synchronize();loaded=time.perf_counter()
        confirmed=read(DEST/'confirmation'/args.child/'complete.json');t=confirmed['variants']['fp32']['metrics']['0.01']['threshold']
        expected=confirmed['timing']['samples'][0];docid=expected['document_id']
        docs=[json.loads(s) for s in (ROOT/'data/prepared/search_candidate_scoring.jsonl').read_text(encoding='utf-8').splitlines()];d=next(x for x in docs if x['document_id']==docid)
        if c['representation']=='token':
            policy=validate_preprocessing_policy();tok=tokenizer(policy);windows=window_document(tok,d['text'],docid,config=policy);pairs=[(w['input_ids'],w['attention_mask']) for w in windows];length=256
        else:
            values=[b+1 for b in d['text'].encode('utf-8')];pairs=[];at=0;length=1024
            while True:
                end=min(at+1024,len(values));pairs.append((values[at:end],[1]*(end-at)))
                if end==len(values):break
                at=end-256
        ready=time.perf_counter();risk=0.
        with torch.inference_mode():
            for ids,mask in pairs:
                ids=torch.tensor([ids+[0]*(length-len(ids))],device='cuda');mask=torch.tensor([mask+[0]*(length-len(mask))],device='cuda')
                risk=max(risk,float(torch.softmax(model(ids,mask).logits,1)[0,1]))
        decision=risk>t;torch.cuda.synchronize();finished=time.perf_counter()
        print(json.dumps({'run_id':args.child,'document_id':docid,'windows':len(pairs),'risk':risk,'decision':decision,
            'profiler_source_sha256':sha(Path(__file__)),'freeze_sha256':sha(Path(args.freeze)),
            'cudnn_plan_cache_limit':os.environ['TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT'],
            'cudnn_v8_api_disabled':os.environ['TORCH_CUDNN_V8_API_DISABLED'],
            'imports_and_setup_seconds':imports_finished-started,'artifact_validation_seconds':load_at-imports_finished,
            'model_load_seconds':loaded-load_at,'data_tokenizer_and_preparation_seconds':ready-loaded,
            'first_document_execution_seconds':finished-ready,'child_start_to_decision_seconds':finished-started,
            'expected_warm_fp32_risk':expected['observed_risk_score'],'cold_warm_risk_delta':risk-expected['observed_risk_score'],
            'cold_warm_decision_changed':decision!=expected['observed_decision']}));return
    freeze=read(Path(args.freeze));output=[]
    for spec in freeze['runs']:
        if spec['seed']!=42:continue
        env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=str(freeze['cudnn_cache_limits'][spec['config_id']]));started=time.perf_counter()
        env['TORCH_CUDNN_V8_API_DISABLED']=str(freeze['cudnn_v8_disabled'][spec['config_id']])
        result=subprocess.run([sys.executable,str(Path(__file__)),'--child',spec['run_id'],'--freeze',args.freeze],capture_output=True,text=True,env=env,cwd=ROOT,check=True)
        elapsed=time.perf_counter()-started;r=json.loads(result.stdout.strip().splitlines()[-1]);r['parent_spawn_to_process_exit_seconds']=elapsed
        r['note']='one fresh process; system/file cache not flushed; first short timing document; not a cold-boot distribution'
        output.append(r);(DEST/'cold_process_timings.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8');print(json.dumps(r),flush=True)
if __name__=='__main__':main()
