"""Export the frozen32development cases for connected actual FP32 C-service checks.
Default preview; bounded resumable CPU B1 execution, no approval/training/held-out data.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, json, shutil, struct, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from score_qualified_development import load_model, paused, save
from gate_eval import load_config, window_document
DATA=ROOT/'results/goal4/numerical_bridge'
OUT=DATA/'development_qualification_w8_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def validate_selection(selected,cfg,tokenizer):
 if len(selected['cases'])!=32 or len({c['document_id'] for c in selected['cases']})!=32 or len({c['source_group_id'] for c in selected['cases']})!=32:raise ValueError('Selection must contain32unique documents/groups')
 assigned={}
 for split in ('search_temporary_threshold','search_candidate_scoring'):
  with (ROOT/'data/prepared'/(split+'.jsonl')).open(encoding='utf-8') as f:
   for line in f:
    doc=json.loads(line)
    if doc['document_id'] in assigned:raise ValueError('Duplicate prepared ID')
    assigned[doc['document_id']]=(split,doc)
 for case in selected['cases']:
  split,doc=assigned[case['document_id']]
  if split!=case['partition'] or doc['label']!=case['label'] or doc['source_group_id']!=case['source_group_id'] or doc['split']!='search' or doc['search_subset']!=split.removeprefix('search_'):raise ValueError('Selected prepared membership differs')
  windows=window_document(tokenizer,doc['text'],doc['document_id'],doc['label'],doc.get('injection_spans'),cfg)
  padded=[]
  for window in windows:
   fields={k:window[k] for k in ('input_ids','attention_mask','token_type_ids')}
   batch=tokenizer.pad([fields],padding='max_length',max_length=256,return_tensors=None)
   padded.append({k:[int(v) for v in batch[k][0]] for k in fields})
  if padded!=case['windows'] or len(padded)!=case['window_count']:raise ValueError('Regenerated selected window differs')
 return {(case['case_index'],wi):(case,window) for case in selected['cases'] for wi,window in enumerate(case['windows'])}

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true')
 ap.add_argument('--max-new-windows',type=int,default=1);ap.add_argument('--floating',action='store_true');args=ap.parse_args()
 global OUT
 if args.floating:OUT=DATA/'development_qualification_floating_v1'
 selected=read(DATA/'development_cases_v1/selection.json')
 if selected['case_count']!=32 or sum(c['window_count'] for c in selected['cases'])!=66:raise ValueError('Frozen selection differs')
 if not args.execute:
  print(json.dumps(dict(execute=False,cases=32,windows=66,max_new_windows=args.max_new_windows,output=str(OUT),score_approval=None)));return
 if args.max_new_windows<1:raise ValueError('Positive window limit required')
 paused();cfg=load_config();candidate=read(ROOT/'results/goal3/qat_w8_a8/candidate.json')
 precision=candidate['identity']['precision_map']
 model,tokenizer,identity=load_model(cfg,precision if not args.floating else None,'checkpoints/quantized/qat_w8_a8' if not args.floating else None,floating=args.floating)
 import torch
 from gate_quant import encoder_linears, quantize_codes
 assigned=validate_selection(selected,cfg,tokenizer)
 if set(precision.values())!={8}:raise ValueError('W8 qualification only')
 OUT.mkdir(parents=True,exist_ok=True);common=OUT/'parameters';common.mkdir(exist_ok=True)
 statepath=OUT/'export.json'
 if statepath.exists():
  state=read(statepath)
  if state['identity']!=identity or state['selection']!=selected or state.get('floating',False)!=args.floating:raise ValueError('Existing qualification identity differs')
 else:
  state=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),identity=identity,selection=selected,
    completed_windows=[],complete=False,floating=args.floating,source='actual unchanged HLS FP C service plus checked integer contract; no full RTL claim')
  shutil.copyfile(Path(__file__),OUT/Path(__file__).name);shutil.copyfile(ROOT/'gate_fpga_reference.py',OUT/'gate_fpga_reference.py')
 def write(folder,name,tensor,integer=False,existing=None):
  value=tensor.detach().cpu().to(torch.int8 if integer else torch.float32).contiguous()
  if not integer and not bool(torch.isfinite(value).all()):raise ValueError('Nonfinite '+name)
  raw=value.numpy().astype('i1' if integer else '<f4',copy=False).tobytes();path=folder/(name+'.bin')
  if existing is not None and not args.floating and (existing/(name+'.bin')).read_bytes()!=raw:raise ValueError('Original frozen parameter differs: '+name)
  if path.exists() and path.read_bytes()!=raw:raise ValueError('Existing binary differs: '+name)
  if not path.exists():path.write_bytes(raw)
 suffix={'attention.self.query':'query','attention.self.key':'key','attention.self.value':'value',
  'attention.output.dense':'attention_output','intermediate.dense':'ffn_input','output.dense':'ffn_output'}
 underlying=model.model;emb=underlying.bert.embeddings
 with torch.inference_mode():
  for _,path,module in encoder_linears(underlying):
   name='layer'+path.split('.')[3]+'_'+suffix['.'.join(path.split('.')[4:])]
   if args.floating:
    write(common,name+'_float_weight',module.weight);write(common,name+'_float_bias',module.bias)
   else:
    write(common,name+'_w8',quantize_codes(module.weight,module.weight_scale_8,8),True,DATA)
    write(common,name+'_params',torch.cat((module.input_scale.reshape(1),module.weight_scale_8.reshape(-1),module.bias.reshape(-1))),existing=DATA)
  write(common,'embedding_ln',torch.cat((emb.LayerNorm.weight,emb.LayerNorm.bias)),existing=DATA)
  write(common,'embedding_position',emb.position_embeddings(torch.arange(256).reshape(1,-1)),existing=DATA)
  for i,block in enumerate(underlying.bert.encoder.layer):
   for name,module in [('attention_ln',block.attention.output.LayerNorm),('output_ln',block.output.LayerNorm)]:
    write(common,f'layer{i}_'+name,torch.cat((module.weight,module.bias)),existing=DATA)
  for name,module in [('pooler',underlying.bert.pooler.dense),('classifier',underlying.classifier)]:
   write(common,name+'_weight',module.weight,existing=DATA);write(common,name+'_bias',module.bias,existing=DATA)
  write(common,'threshold',torch.tensor([selected['sampling']['historical_operating_threshold']]),existing=DATA)
  done=set()
  for v in state['completed_windows']:
   key=(v['case_index'],v['window_index'])
   if key in done or key not in assigned:raise ValueError('Duplicate or unassigned completed window')
   case,window=assigned[key]
   if v['inputs']!=window or any(v[k]!=case[k] for k in ('document_id','source_group_id','label','partition')):raise ValueError('Committed selected window differs')
   folder=ROOT/v['folder']
   if not folder.is_relative_to(OUT):raise ValueError('Committed output path escaped selected tree')
   if read(folder/'window.json')!=v:raise ValueError('Committed window record differs')
   if (folder/'reference_logits.bin').read_bytes()!=struct.pack('<2f',*v['logits']) or (folder/'reference_risk.bin').read_bytes()!=struct.pack('<f',v['risk']):raise ValueError('Committed terminal bytes differ')
   done.add(key)
  new=0
  for case in selected['cases']:
   for wi,window in enumerate(case['windows']):
    key=(case['case_index'],wi)
    if key in done:continue
    if new>=args.max_new_windows:save(statepath,state);return
    paused();folder=OUT/f"case{case['case_index']:02d}"/f'window{wi:03d}';folder.mkdir(parents=True,exist_ok=True)
    batch={k:torch.tensor([window[k]],dtype=torch.long) for k in ('input_ids','attention_mask','token_type_ids')}
    if any(v.shape!=(1,256) for v in batch.values()):raise ValueError('Frozen input shape differs')
    write(folder,'embedding_word',emb.word_embeddings(batch['input_ids']))
    write(folder,'embedding_type',emb.token_type_embeddings(batch['token_type_ids']))
    write(folder,'key_mask',(1.-batch['attention_mask'].float())*torch.finfo(torch.float32).min)
    result=model(**batch,capture_intermediates=True)
    for name,tensor in result.intermediates.items():
     if name.endswith('_a8'):write(folder,name[:-3]+'_reference_a8',tensor,True)
     elif name in ('embedding_ln','layer0','layer1','layer2','layer3','logits','risk'):write(folder,'reference_'+name,tensor)
    record=dict(case_index=case['case_index'],window_index=wi,document_id=case['document_id'],source_group_id=case['source_group_id'],
      partition=case['partition'],label=case['label'],inputs=window,folder=folder.relative_to(ROOT).as_posix(),
      logits=result.logits.tolist()[0],risk=float(result.risk.item()),score_acceptance=None)
    save(folder/'window.json',record);state['completed_windows'].append(record);save(statepath,state);new+=1
    print(json.dumps(dict(exported=key,risk=record['risk'],score_approval=None)),flush=True)
  state['complete']=len(state['completed_windows'])==66;save(statepath,state)
if __name__=='__main__':main()
