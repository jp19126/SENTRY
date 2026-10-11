"""Bounded B8/B32 ordered CPU W8-QAT or floating equivalence against saved B1.

Preview by default. Captured forward elapsed is instrumented research work,
not a detector/board benchmark. Qualification never approves a score identity.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/goal4/numerical_bridge/development_qualification_w8_v1'
OUT=ROOT/'results/goal4/numerical_bridge/ordered_batching_w8_v1'
FLOATING=False
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from validate_qualification_exports import validate_exports,LINEARS

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,value):
 p.parent.mkdir(parents=True,exist_ok=True);temporary=p.with_suffix(p.suffix+'.tmp')
 temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temporary.replace(p)
def raw(tensor,integer=False):
 value=tensor.detach().cpu().contiguous().numpy()
 return value.astype('i1' if integer else '<f4',copy=False).tobytes()
def tensors():
 a8=[] if FLOATING else [(f'layer{i}_{name}_a8',f'layer{i}_{name}_reference_a8.bin',True) for i in range(4) for name in LINEARS]
 return a8+[(name,'reference_'+name+'.bin',False) for name in ('embedding_ln','layer0','layer1','layer2','layer3','logits','risk')]
def extents(name):
 if name.endswith('_a8'):return 256*LINEARS[name.split('_',1)[1][:-3]][0]
 return 2 if name=='logits' else 1 if name=='risk' else 256*256

def validate_model_parameters(model,items):
 """Bind current loaded checkpoint tensors to the already exported parameters."""
 import torch
 from gate_quant import encoder_linears,quantize_codes
 common=DATA/'parameters';underlying=model.model;emb=underlying.bert.embeddings
 suffix={'attention.self.query':'query','attention.self.key':'key','attention.self.value':'value','attention.output.dense':'attention_output','intermediate.dense':'ffn_input','output.dense':'ffn_output'}
 def check(name,value,integer=False,folder=common):
  if raw(value,integer)!=(folder/(name+'.bin')).read_bytes():raise ValueError('Loaded parameter differs from saved B1: '+name)
 for _,path,module in encoder_linears(underlying):
  name='layer'+path.split('.')[3]+'_'+suffix['.'.join(path.split('.')[4:])]
  if FLOATING:
   check(name+'_float_weight',module.weight);check(name+'_float_bias',module.bias)
  else:
   check(name+'_w8',quantize_codes(module.weight,module.weight_scale_8,8),True)
   check(name+'_params',torch.cat((module.input_scale.reshape(1),module.weight_scale_8.reshape(-1),module.bias.reshape(-1))))
 check('embedding_ln',torch.cat((emb.LayerNorm.weight,emb.LayerNorm.bias)))
 check('embedding_position',emb.position_embeddings.weight[:256])
 for i,block in enumerate(underlying.bert.encoder.layer):
  for name,module in [('attention_ln',block.attention.output.LayerNorm),('output_ln',block.output.LayerNorm)]:check(f'layer{i}_'+name,torch.cat((module.weight,module.bias)))
 for name,module in [('pooler',underlying.bert.pooler.dense),('classifier',underlying.classifier)]:check(name+'_weight',module.weight);check(name+'_bias',module.bias)
 for item in items:
  folder=ROOT/item['folder'];inputs=item['inputs']
  check('embedding_word',emb.word_embeddings.weight[torch.tensor(inputs['input_ids'])],folder=folder)
  check('embedding_type',emb.token_type_embeddings.weight[torch.tensor(inputs['token_type_ids'])],folder=folder)

def member(item):return {k:item[k] for k in ('case_index','window_index','document_id','source_group_id','partition','label','folder')}
def check_record(record,items,batch_size,start):
 """Recompute saved checks from saved output bytes before reusing a batch."""
 expected=items[start:start+batch_size]
 if record['start_window']!=start or record['configured_batch_size']!=batch_size or record['actual_batch_size']!=len(expected) or record['members']!=[member(v) for v in expected]:raise ValueError('Saved batching membership differs')
 folder=(OUT/record['attempt_folder']).resolve()
 if not folder.is_relative_to(OUT.resolve()):raise ValueError('Saved attempt escaped output tree')
 checks=[]
 for wi,item in enumerate(expected):
  reference=ROOT/item['folder']
  for name,filename,integer in tensors():
   actual=(folder/f'window{wi:03d}'/(name+'.bin')).read_bytes();saved=(reference/filename).read_bytes();size=extents(name)*(1 if integer else 4)
   if len(actual)!=size or len(saved)!=size:raise ValueError('Captured tensor extent differs: '+name)
   checks.append(dict(window_index=wi,tensor=name,bytes=size,exact=actual==saved,unequal_bytes=sum(a!=b for a,b in zip(actual,saved))))
 if record['checks']!=checks or record['passed']!=all(v['exact'] for v in checks):raise ValueError('Saved batch checks differ from actual bytes')
 if not record['passed']:raise RuntimeError('Saved first batching mismatch requires review; do not skip on resume')
 if not isinstance(record['forward_elapsed_seconds'],(int,float)) or record['forward_elapsed_seconds']<0:raise ValueError('Malformed forward elapsed')
 return record

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');ap.add_argument('--floating',action='store_true');ap.add_argument('--batch-size',choices=('8','32','both'),default='both');ap.add_argument('--max-new-batches',type=int,default=1);args=ap.parse_args()
 global DATA,OUT,FLOATING
 FLOATING=args.floating
 if FLOATING:
  DATA=ROOT/'results/goal4/numerical_bridge/development_qualification_floating_v1'
  OUT=ROOT/'results/goal4/numerical_bridge/ordered_batching_floating_v1'
 sizes=[8,32] if args.batch_size=='both' else [int(args.batch_size)]
 exports=read(DATA/'export.json');selection=read(ROOT/'results/goal4/numerical_bridge/development_cases_v1/selection.json')
 if exports['selection']!=selection or exports.get('floating',False)!=FLOATING:raise ValueError('Frozen B1 selection/reference mode differs')
 ready=exports['complete'] and len(exports['completed_windows'])==66
 if not args.execute:
  print(json.dumps(dict(execute=False,floating=FLOATING,windows=66,exported_b1_windows=len(exports['completed_windows']),ready=ready,blockers=[] if ready else ['Complete 66-window B1 exports required'],batch_sizes=sizes,max_new_batches=args.max_new_batches,required_forwards={str(b):(66+b-1)//b for b in sizes},output=str(OUT),scope='instrumented forward with captures; no board benchmark',score_acceptance=None)));return
 if not ready:raise ValueError('Complete frozen 66-window B1 exports required')
 # Canonical selection order, independent of exporter dispatch/commit order.
 bykey={(v['case_index'],v['window_index']):v for v in exports['completed_windows']}
 items=[bykey[c['case_index'],wi] for c in selection['cases'] for wi in range(c['window_count'])]
 if args.max_new_batches<1:raise ValueError('Positive bounded batch count required')
 from score_qualified_development import load_model,paused
 from gate_eval import load_config
 paused()
 # This prerequisite binds saved B1 exports; C-service diagnostics are a
 # separate experiment and are neither required nor relabelled by batching.
 validate_exports(DATA,qualification=dict(identity=exports['identity'],selection=selection,windows=[],complete=False))
 if (DATA/'gate_fpga_reference.py').read_bytes()!=(ROOT/'gate_fpga_reference.py').read_bytes():raise ValueError('Ordered arithmetic source differs from saved B1 export')
 config=load_config();precision=None if FLOATING else exports['identity']['precision_map']
 if not FLOATING and set(precision.values())!={8}:raise ValueError('Only frozen uniform W8 QAT supported')
 model,tokenizer,identity=load_model(config,precision,None if FLOATING else 'checkpoints/quantized/qat_w8_a8',floating=FLOATING)
 if identity!=exports['identity']:raise ValueError('Loaded model/settings differ from saved B1 identity')
 import torch
 settings=dict(torch_version=torch.__version__,device=str(next(model.model.parameters()).device),threads=torch.get_num_threads(),default_dtype=str(torch.get_default_dtype()),cpu_autocast_enabled=torch.is_autocast_enabled('cpu'),cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,flush_denormal_requested=False,model_training=model.training,wrapped_model_training=model.model.training)
 if settings['threads']!=4 or settings['device']!='cpu' or settings['default_dtype']!='torch.float32' or any(settings[k] for k in ('cpu_autocast_enabled','cuda_matmul_tf32','cudnn_tf32','model_training','wrapped_model_training')):raise ValueError('Ordered CPU backend/settings differ')
 with torch.inference_mode():validate_model_parameters(model,items)
 statepath=OUT/'qualification.json';state=read(statepath) if statepath.exists() else dict(recorded_utc=datetime.now(timezone.utc).isoformat(),b1_identity=identity,selection=selection,batch_sizes={},score_acceptance=None,timing_scope='CPU instrumented forward elapsed including captures; excludes setup/comparison/storage; no board benchmark')
 if state['b1_identity']!=identity or state['selection']!=selection:raise ValueError('Batching resume source identity differs')
 if state.get('settings',settings)!=settings:raise ValueError('Saved batching CPU settings differ')
 state['settings']=settings
 count=0
 for size in sizes:
  execution=identity['execution_identity']+f'__qualification_batch_policy{size}_tail_actual'
  policy=dict(identity={**identity,'source_execution_identity':identity['execution_identity'],'execution_identity':execution,'configured_batch_size':size},batches=[],active=None,next_attempt=0,complete=False,passed=None,score_acceptance=None)
  policy=state['batch_sizes'].setdefault(str(size),policy)
  if policy['identity']!={**identity,'source_execution_identity':identity['execution_identity'],'execution_identity':execution,'configured_batch_size':size}:raise ValueError('Batching policy identity differs')
  start=0
  for record in policy['batches']:
   check_record(record,items,size,start);start+=record['actual_batch_size']
  active=policy['active']
  if active:
   if active['start_window']!=start:raise ValueError('Pending batching position differs')
   path=OUT/active['attempt_folder']/'batch.json'
   if path.exists():
    record=check_record(read(path),items,size,start);policy['batches'].append(record);start+=record['actual_batch_size']
   else:
    # Preserve partial output and unknown interrupted work, then retry into a new
    # attempt directory. Never overwrite or claim a partial forward completed.
    policy.setdefault('interrupted_attempts',[]).append(dict(**active,state='interrupted_uncommitted',forward_elapsed_seconds=None))
   policy['active']=None;save(statepath,state)
  if start>len(items) or (policy['complete'] and start!=len(items)):raise ValueError('Saved batching completion differs')
  if policy.get('state')=='stopped_first_mismatch':raise RuntimeError('Saved batching mismatch requires review')
  while start<len(items) and count<args.max_new_batches:
   paused();current=items[start:start+size];attempt=policy['next_attempt'];policy['next_attempt']+=1
   relative=Path(f'b{size}')/f'batch{start//size:03d}'/f'attempt{attempt:03d}';folder=OUT/relative
   if folder.exists():raise ValueError('Preserve existing unassigned batching attempt')
   active=dict(start_window=start,actual_batch_size=len(current),attempt_folder=relative.as_posix())
   policy['active']=active;save(statepath,state);folder.mkdir(parents=True)
   batch={k:torch.tensor([item['inputs'][k] for item in current],dtype=torch.long) for k in ('input_ids','attention_mask','token_type_ids')}
   begun=time.perf_counter()
   with torch.inference_mode():result=model(**batch,capture_intermediates=True)
   elapsed=time.perf_counter()-begun
   checks=[]
   for wi,item in enumerate(current):
    destination=folder/f'window{wi:03d}';destination.mkdir()
    for name,filename,integer in tensors():
     actual=raw(result.intermediates[name][wi],integer);expected=(ROOT/item['folder']/filename).read_bytes();extent=extents(name)*(1 if integer else 4)
     (destination/(name+'.bin')).write_bytes(actual)
     if len(actual)!=extent or len(expected)!=extent:raise ValueError('Actual captured extent differs: '+name)
     checks.append(dict(window_index=wi,tensor=name,bytes=extent,exact=actual==expected,unequal_bytes=sum(a!=b for a,b in zip(actual,expected))))
   passed=all(v['exact'] for v in checks)
   record=dict(**active,configured_batch_size=size,members=[member(v) for v in current],forward_elapsed_seconds=elapsed,checks=checks,passed=passed,first_mismatch=next((v for v in checks if not v['exact']),None),numerical_acceptance=None)
   save(folder/'batch.json',record);policy['batches'].append(record);policy['active']=None;count+=1;start+=len(current)
   if not passed:policy['state']='stopped_first_mismatch';save(statepath,state);raise RuntimeError('First batching bit mismatch saved; inspect before any further forward')
   save(statepath,state);print(json.dumps(dict(batch_size=size,actual_batch_size=len(current),completed_windows=start,forward_elapsed_seconds=elapsed,all_captured_bits_exact=True)),flush=True)
  policy['complete']=start==len(items);policy['passed']=True if policy['complete'] else None
  policy['completed_windows']=start;policy['forward_elapsed_seconds']=sum(v['forward_elapsed_seconds'] for v in policy['batches'])
  policy['state']='complete_pending_review' if policy['complete'] else 'bounded_handoff';save(statepath,state)
 state['complete']=all(state['batch_sizes'].get(str(size),{}).get('complete',False) for size in (8,32));save(statepath,state)
if __name__=='__main__':main()
