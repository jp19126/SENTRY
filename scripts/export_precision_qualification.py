"""Explicit supplied W4/W8-map B1 exports on the frozen32/66 development set.
Preview by default; no map search, training, calibration, scores or approval.
Decoded signed-i8 codes and packed weight bytes are separate representations.
"""
from pathlib import Path
import argparse,ast,json,math,re,struct,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
BASE=ROOT/'results/goal4/numerical_bridge/precision_qualification_v1'
GROUPS=[f'layer{i}.{g}' for i in range(4) for g in ('qkv','attention_output','ffn_input','ffn_output')]
SITES={'query':(256,256),'key':(256,256),'value':(256,256),'attention_output':(256,256),'ffn_input':(256,1024),'ffn_output':(1024,256)}
SCHEME='static_symmetric_encoder_linear_w4w8a8_fp32_other_v1'

def need(ok,why):
 if not ok:raise ValueError(why)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def path(value):
 p=Path(str(value).replace('\\','/'));return p if p.is_absolute() else ROOT/p
def outdir(name):
 need(re.fullmatch(r'[a-z][a-z0-9_-]{0,63}',name) is not None,'Explicit lowercase map/run name required');return BASE/name
def map_from(value):
 record=read(path(value));p=record.get('precision_map',record.get('identity',{}).get('precision_map',record));need(set(p)==set(GROUPS) and all(type(p[k]) is int and p[k] in (4,8) for k in p),'Exactly16 explicit integer W4/W8 group decisions required');return {k:p[k] for k in GROUPS}
def decoded_codes(raw,bits):
 need(bits in (4,8),'W4/W8 only');values=[v if v<128 else v-256 for v in raw];lo,hi=-(1<<(bits-1)),(1<<(bits-1))-1;need(all(lo<=v<=hi for v in values),'Decoded signed codes exceed declared W'+str(bits));return values
def pack_codes(raw,bits):
 values=decoded_codes(raw,bits)
 if bits==8:return bytes(v&255 for v in values)
 need(len(values)%2==0,'Even row-major W4 code count required');return bytes((values[i]&15)|((values[i+1]&15)<<4) for i in range(0,len(values),2))
def unpack_codes(packed,bits):
 if bits==8:return bytes(packed)
 need(bits==4,'W4/W8 only');result=[]
 for value in packed:
  for code in (value&15,value>>4):result.append((code if code<8 else code-16)&255)
 return bytes(result)
def write(p,raw):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():need(p.read_bytes()==raw,'Preserve different actual bytes: '+str(p))
 else:p.write_bytes(raw)
def function_ast(source,name):return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name))
def scientific_sources(out,first=False):
 sources={'gate_quant.py':ROOT/'gate_quant.py','gate_fpga_reference.py':ROOT/'gate_fpga_reference.py','gate_eval.py':ROOT/'gate_eval.py','score_qualified_development.py':ROOT/'scripts/score_qualified_development.py','export_qualification_cases.py':ROOT/'scripts/export_qualification_cases.py'}
 for name,current in sources.items():
  stored=out/'sources'/name
  if first:write(stored,current.read_bytes())
  elif name=='score_qualified_development.py':need(function_ast(stored.read_text(),'load_model')==function_ast(current.read_text(),'load_model'),'Scientific model-loader function changed')
  elif name=='export_qualification_cases.py':need(function_ast(stored.read_text(),'validate_selection')==function_ast(current.read_text(),'validate_selection'),'Scientific selection checker changed')
  else:need(stored.read_bytes()==current.read_bytes(),'Scientific primitive/window source changed: '+name)

def request(args):
 precision=map_from(args.precision_map);out=outdir(args.name);run=read(ROOT/'results/goal3/run.json');calibration=read(path(run['calibration_file']));need(calibration['scheme_id']==SCHEME,'Frozen training calibration scheme differs')
 need((args.source=='qat')==bool(args.qat_checkpoint),'Supply matching saved QAT checkpoint only for --source qat')
 qat=None
 if args.qat_checkpoint:
  qp=path(args.qat_checkpoint);qat=read(qp/'quantization.json');need(qat['precision_map']==precision and qat['scheme_id']==SCHEME,'Saved QAT precision map/scheme mismatch');need(qat['calibration']==calibration,'QAT did not retain the existing training calibration');need((qp/'quantized_state.pt').is_file(),'Missing saved actual QAT state')
 selection=read(ROOT/'results/goal4/numerical_bridge/development_cases_v1/selection.json');need(selection['case_count']==32 and sum(c['window_count'] for c in selection['cases'])==66,'Frozen32/66 selection required')
 return precision,out,run,calibration,qat,selection

def validate_export(out,complete=False):
 """Pure-stdlib association, code-range and exact packed-byte checks."""
 out=Path(out);state=read(out/'export.json');selection=read(ROOT/'results/goal4/numerical_bridge/development_cases_v1/selection.json');need(state['selection']==selection and state['identity']['precision_map']==state['precision_map'],'Selection/map identity differs');scientific_sources(out)
 need(state['calibration']==read(path(read(ROOT/'results/goal3/run.json')['calibration_file'])),'Frozen training scale metadata changed');params=out/'parameters';need(len(state['layers'])==24,'Exactly24 real encoder sites required');common=set()
 for layer in state['layers']:
  name,k,n,bits=layer['name'],layer['k'],layer['n'],layer['bits'];need(bits==state['precision_map'][layer['group']] and (k,n)==SITES[name.split('_',1)[1]],'Site precision/shape metadata differs')
  codes=(params/(name+'_code_i8.bin')).read_bytes();need(len(codes)==k*n,'Decoded weight extent');values=decoded_codes(codes,bits);packed=(params/layer['packed_file']).read_bytes();need(packed==pack_codes(codes,bits) and unpack_codes(packed,bits)==codes,'Actual packed/decoded weight mismatch')
  need(layer['code_min']==min(values) and layer['code_max']==max(values) and layer['packed_bytes']==len(packed),'Actual range/packed report differs');common.update((name+'_code_i8.bin',name+'_params.bin'))
  raw=(params/(name+'_params.bin')).read_bytes();need(len(raw)==(1+2*n)*4,'Scale/bias extent');p=struct.unpack('<'+str(1+2*n)+'f',raw);need(all(math.isfinite(v) for v in p) and all(v>0 for v in p[:1+n]),'Actual scale/bias validity');need(raw[4:4+4*n]==(params/(name+f'_weight_scale_{bits}.bin')).read_bytes(),'Selected actual frozen weight scale differs');need(raw[:4]==struct.pack('<f',state['calibration']['modules'][layer['module_path']]['input_scale']),'Actual A8 training scale differs')
  weights_raw=(params/(name+'_float_weight.bin')).read_bytes();need(len(weights_raw)==k*n*4,'Genuine FP weight snapshot extent');weights=struct.unpack('<'+str(k*n)+'f',weights_raw);need(all(math.isfinite(v) for v in weights),'Nonfinite genuine FP weights')
  for width in (4,8):
   scales_raw=(params/(name+f'_weight_scale_{width}.bin')).read_bytes();need(len(scales_raw)==n*4,'Genuine frozen scale extent');scales=struct.unpack('<'+str(n)+'f',scales_raw);need(all(math.isfinite(v) and v>0 for v in scales),'Invalid genuine frozen scale')
  low,high=-(1<<(bits-1)),(1<<(bits-1))-1
  for index,value in enumerate(weights):
   quotient=struct.unpack('<f',struct.pack('<f',value/p[1+index//k]))[0];expected=max(low,min(high,round(quotient)));need(values[index]==expected,'Actual decoded code differs from genuine saved FP weight/frozen selected scale')
  common.update((layer['packed_file'],name+'_float_weight.bin',name+'_weight_scale_4.bin',name+'_weight_scale_8.bin'))
 from validate_qualification_exports import common_parameters,terminal
 common.update(common_parameters());common.update(common_parameters(True))
 fixed=common_parameters();fixed={k:v for k,v in fixed.items() if not k.endswith(('_w8.bin','_params.bin'))}
 for name,size in fixed.items():need(len((params/name).read_bytes())==size,'Common FP32 extent: '+name)
 snapshot=out/'reference_parameters';need(snapshot.is_dir() and {p.name for p in snapshot.iterdir()}=={p.name for p in params.iterdir()},'Genuine loaded parameter snapshot membership differs')
 for original in params.iterdir():need(original.is_file() and original.read_bytes()==(snapshot/original.name).read_bytes(),'C parameter bytes differ from genuine loaded B1 parameter snapshot: '+original.name)
 need((params/'threshold.bin').read_bytes()==struct.pack('<f',selection['sampling']['historical_operating_threshold']),'Frozen threshold changed')
 assigned={(c['case_index'],i):(c,w) for c in selection['cases'] for i,w in enumerate(c['windows'])};done=set()
 for item in state['completed_windows']:
  key=item['case_index'],item['window_index'];need(key in assigned and key not in done,'Unassigned/duplicate exported window');case,window=assigned[key];folder=ROOT/item['folder'];need(folder.resolve()==(out/f'case{key[0]:02d}'/f'window{key[1]:03d}').resolve(),'Assigned output folder differs');need(item['inputs']==window and all(item[k]==case[k] for k in ('document_id','source_group_id','partition','label')),'Frozen input membership differs');need(read(folder/'window.json')==item,'Saved actual B1 window record differs');need(not any((folder/name).exists() for name in common),'Window file shadows common parameter');terminal(folder,'reference',item['logits'],item['risk'])
  for name,size in [('embedding_word',65536*4),('embedding_type',65536*4),('key_mask',256*4),('reference_embedding_ln',65536*4),*[(f'reference_layer{i}',65536*4) for i in range(4)]]:need(len((folder/(name+'.bin')).read_bytes())==size,'B1 tensor extent '+name)
  for layer in state['layers']:need(len((folder/(layer['name']+'_reference_a8.bin')).read_bytes())==256*layer['k'],'Actual A8 B1 extent')
  done.add(key)
 need(not state['complete'] or done==set(assigned),'Incomplete claimed export');need(not complete or state['complete'] and done==set(assigned),'All66 genuine B1 exports required before C dispatch')
 return state

def execute(args):
 precision,out,run,calibration,qat,selection=request(args)
 from score_qualified_development import load_model,paused
 from export_qualification_cases import validate_selection
 from gate_eval import load_config
 from gate_quant import GateQuantLinear,encoder_linears,quantize_codes
 import torch
 paused();config=load_config();model,tokenizer,identity=load_model(config,precision,args.qat_checkpoint if args.source=='qat' else None)
 need(torch.get_num_threads()==4 and not torch.is_autocast_enabled('cpu') and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,'Established CPU settings differ');assigned=validate_selection(selection,config,tokenizer)
 out.mkdir(parents=True,exist_ok=True);p=out/'parameters';p.mkdir(exist_ok=True);statepath=out/'export.json';first=not statepath.exists();scientific_sources(out,first)
 dispatch=out/f'export_dispatch_{len(list(out.glob("export_dispatch_*.py")))+1:03d}.py';write(dispatch,Path(__file__).read_bytes());write(out/'sources/calibration.json',json.dumps(calibration,sort_keys=True).encode());
 if qat is not None:write(out/'sources/qat_quantization.json',path(args.qat_checkpoint).joinpath('quantization.json').read_bytes())
 extended=dict(identity,precision_source=args.source,qat_checkpoint=None if args.source=='ptq' else path(args.qat_checkpoint).relative_to(ROOT).as_posix(),storage_contract='decoded_signed_i8_codes_plus_real_packed_w4w8_v1')
 state=read(statepath) if not first else dict(identity=extended,precision_map=precision,source=args.source,map_source=str(path(args.precision_map).relative_to(ROOT)),calibration=calibration,selection=selection,layers=[],completed_windows=[],complete=False,score_acceptance=None)
 need(state['identity']==extended and state['precision_map']==precision and state['selection']==selection and state['calibration']==calibration,'Actual map/checkpoint/protocol identity changed')
 def raw(t,integer=False):
  value=t.detach().cpu().to(torch.int8 if integer else torch.float32).contiguous();need(integer or bool(torch.isfinite(value).all()),'Nonfinite export');return value.numpy().astype('i1' if integer else '<f4',copy=False).tobytes()
 suffix={'attention.self.query':'query','attention.self.key':'key','attention.self.value':'value','attention.output.dense':'attention_output','intermediate.dense':'ffn_input','output.dense':'ffn_output'};layers=[];underlying=model.model
 with torch.inference_mode():
  for group,module_path,module in encoder_linears(underlying):
   need(isinstance(module,GateQuantLinear) and module.weight_bits==precision[group] and not module.collect_saturation,'Genuine matching GateQuantLinear required');name='layer'+module_path.split('.')[3]+'_'+suffix['.'.join(module_path.split('.')[4:])];bits=module.weight_bits;scale=getattr(module,f'weight_scale_{bits}');codes=raw(quantize_codes(module.weight,scale,bits),True);packed=pack_codes(codes,bits);need(unpack_codes(packed,bits)==codes,'Packed roundtrip');values=decoded_codes(codes,bits)
   write(p/(name+'_code_i8.bin'),codes);write(p/(name+'_float_weight.bin'),raw(module.weight));write(p/(name+'_params.bin'),raw(torch.cat((module.input_scale.reshape(1),scale.reshape(-1),module.bias.reshape(-1)))))
   for width in (4,8):write(p/(name+f'_weight_scale_{width}.bin'),raw(getattr(module,f'weight_scale_{width}')))
   packed_file=name+f'_packed_w{bits}.bin';write(p/packed_file,packed);layers.append(dict(name=name,group=group,module_path=module_path,bits=bits,k=module.in_features,n=module.out_features,decoded_file=name+'_code_i8.bin',decoded_format='row-major[N,K] signed two-complement i8; decoded values use actual declared bit range',code_min=min(values),code_max=max(values),packed_file=packed_file,packed_bytes=len(packed),scale_source='Genuine loaded GateQuantLinear frozen buffers; both W4/W8 tables retained; no scale recomputation',packed_format='row-major[N,K], earlier K in low nibble, signed two-complement nibbles' if bits==4 else 'row-major[N,K] signed two-complement bytes',physical_scope='Actual packed storage validated against decoded codes; C arithmetic consumes decoded_i8, not a physical full-detector RTL execution'))
  need(not state['layers'] or state['layers']==layers,'Actual precision/range/storage metadata changed');state['layers']=layers
  emb=underlying.bert.embeddings
  for name,t in [('embedding_ln',torch.cat((emb.LayerNorm.weight,emb.LayerNorm.bias))),('embedding_position',emb.position_embeddings(torch.arange(256).reshape(1,-1)))]:write(p/(name+'.bin'),raw(t))
  for i,block in enumerate(underlying.bert.encoder.layer):
   for name,module in [('attention_ln',block.attention.output.LayerNorm),('output_ln',block.output.LayerNorm)]:write(p/(f'layer{i}_{name}.bin'),raw(torch.cat((module.weight,module.bias))))
  for name,module in [('pooler',underlying.bert.pooler.dense),('classifier',underlying.classifier)]:write(p/(name+'_weight.bin'),raw(module.weight));write(p/(name+'_bias.bin'),raw(module.bias))
  write(p/'threshold.bin',raw(torch.tensor([selection['sampling']['historical_operating_threshold']])));
  for original in p.iterdir():write(out/'reference_parameters'/original.name,original.read_bytes())
  save(statepath,state);validate_export(out)
  done={(v['case_index'],v['window_index']) for v in state['completed_windows']};count=0
  for key,(case,window) in assigned.items():
   if key in done:continue
   if count>=args.max_new_windows:break
   paused();folder=out/f'case{key[0]:02d}'/f'window{key[1]:03d}';need(not folder.exists(),'Preserve interrupted/uncommitted actual B1 output');folder.mkdir(parents=True);batch={k:torch.tensor([window[k]],dtype=torch.long) for k in ('input_ids','attention_mask','token_type_ids')}
   write(folder/'embedding_word.bin',raw(emb.word_embeddings(batch['input_ids'])));write(folder/'embedding_type.bin',raw(emb.token_type_embeddings(batch['token_type_ids'])));write(folder/'key_mask.bin',raw((1.-batch['attention_mask'].float())*torch.finfo(torch.float32).min));result=model(**batch,capture_intermediates=True)
   for name,tensor in result.intermediates.items():
    if name.endswith('_a8'):write(folder/(name[:-3]+'_reference_a8.bin'),raw(tensor,True))
    elif name in ('embedding_ln','layer0','layer1','layer2','layer3','logits','risk'):write(folder/('reference_'+name+'.bin'),raw(tensor))
   item=dict(case_index=key[0],window_index=key[1],document_id=case['document_id'],source_group_id=case['source_group_id'],partition=case['partition'],label=case['label'],inputs=window,folder=folder.relative_to(ROOT).as_posix(),logits=result.logits.tolist()[0],risk=float(result.risk.item()),score_acceptance=None);save(folder/'window.json',item);state['completed_windows'].append(item);count+=1;save(statepath,state)
 state['complete']=len(state['completed_windows'])==66;save(statepath,state);validate_export(out);print(json.dumps(dict(new_B1_windows=count,complete=state['complete'],output=str(out),approval=None)))

def batch_tensors():
 return [(f'layer{i}_{name}_a8',f'layer{i}_{name}_reference_a8.bin',True,256*k) for i in range(4) for name,(k,n) in SITES.items()]+[(name,'reference_'+name+'.bin',False,2 if name=='logits' else 1 if name=='risk' else 65536) for name in ('embedding_ln','layer0','layer1','layer2','layer3','logits','risk')]

def batch_capture_inventory(actual_names):
 required=sorted(v[0] for v in batch_tensors());extras=sorted(['layer1_query_output','layer1_key_output','layer1_value_output','layer1_attention_context']);actual=sorted(actual_names)
 return dict(required_names=required,declared_diagnostic_names=extras,actual_names=actual,exact_membership=set(actual)==set(required)|set(extras),compared_tensors=len(required))

def batch_members(items):return [{k:v[k] for k in ('case_index','window_index','document_id','source_group_id','partition','label','folder')} for v in items]

def batch_checks(folder,items):
 checks=[]
 for wi,item in enumerate(items):
  for name,reference,integer,extent in batch_tensors():
   actual=(folder/f'window{wi:03d}'/(name+'.bin')).read_bytes();expected=(ROOT/item['folder']/reference).read_bytes();size=extent*(1 if integer else 4)
   need(len(actual)==len(expected)==size,'Actual batch/B1 capture extent differs: '+name)
   checks.append(dict(member=wi,tensor=name,bytes=size,exact=actual==expected,unequal_bytes=sum(a!=b for a,b in zip(actual,expected))))
 return checks

def batching_state(out,exports):
 selection=exports['selection'];bykey={(v['case_index'],v['window_index']):v for v in exports['completed_windows']};items=[bykey[c['case_index'],wi] for c in selection['cases'] for wi in range(c['window_count'])]
 folder=out/'batching';file=folder/'qualification.json';state=read(file) if file.exists() else dict(b1_identity=exports['identity'],selection=selection,batches=[],complete_b32=False,score_acceptance=None,scope='Actual captured B8/B32 versus genuine B1 on identical66 frozen windows; no scores or board timing')
 need(state['b1_identity']==exports['identity'] and state['selection']==selection and state['score_acceptance'] is None,'Batch scientific identity/selection/approval differs')
 positions={8:0,32:0}
 for record in state['batches']:
  size,start=record['configured_batch_size'],record['start_window'];need(size in positions and positions[size]==start,'Saved batch order/gap differs');current=items[start:start+size]
  need(record['members']==batch_members(current) and record['actual_batch_size']==len(current),'Actual batch/frozen membership differs')
  target=(folder/record['attempt_folder']).resolve();need(target.is_relative_to(folder.resolve()),'Batch capture escaped own output')
  need(read(target/'batch.json')==record,'Saved batch record association differs');inventory=read(target/'capture_names.json');need(inventory==record['capture_inventory']==batch_capture_inventory(inventory['actual_names']) and inventory['exact_membership'],'Saved exact required31/declared4 capture inventory differs');checks=batch_checks(target,current);need(record['checks']==checks and record['passed']==all(v['exact'] for v in checks),'Saved batch checks differ from actual bytes')
  need(record['passed'],'Saved first batching mismatch cannot resume automatically');positions[size]+=len(current)
 need(state['complete_b32']==(positions[32]==66),'B32 completion declaration differs')
 need(not state.get('state','').startswith('stopped_'),'Saved batching process/mismatch stop requires explicit inspection')
 return folder,file,state,items,positions

def execute_batches(args,validate_only=False):
 """Bounded genuine new-map batch forwards; never inherit W8 qualification."""
 precision,out,run,calibration,qat,selection=request(args);exports=validate_export(out,complete=True);folder,file,state,items,positions=batching_state(out,exports)
 if validate_only:print(json.dumps(dict(validated_batches=len(state['batches']),complete_b32=state['complete_b32'],approval=None)));return
 from score_qualified_development import load_model,paused
 from gate_eval import load_config
 from gate_quant import GateQuantLinear,encoder_linears,quantize_codes
 import torch,time
 paused();model,tokenizer,identity=load_model(load_config(),precision,args.qat_checkpoint if args.source=='qat' else None)
 expected=dict(identity,precision_source=args.source,qat_checkpoint=None if args.source=='ptq' else path(args.qat_checkpoint).relative_to(ROOT).as_posix(),storage_contract='decoded_signed_i8_codes_plus_real_packed_w4w8_v1');need(expected==exports['identity'],'Batch loaded model differs from actual B1 identity')
 need(torch.get_num_threads()==4 and str(next(model.model.parameters()).device)=='cpu' and torch.get_default_dtype()==torch.float32 and not model.training and not model.model.training and not torch.is_autocast_enabled('cpu') and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,'Actual batch CPU/backend settings differ')
 def raw(t,integer=False):
  value=t.detach().cpu().to(torch.int8 if integer else torch.float32).contiguous();need(integer or bool(torch.isfinite(value).all()),'Nonfinite actual batch output');return value.numpy().astype('i1' if integer else '<f4',copy=False).tobytes()
 def check(name,t,integer=False):need(raw(t,integer)==(out/'parameters'/(name+'.bin')).read_bytes(),'Loaded batch parameter differs from genuine saved B1: '+name)
 suffix={'attention.self.query':'query','attention.self.key':'key','attention.self.value':'value','attention.output.dense':'attention_output','intermediate.dense':'ffn_input','output.dense':'ffn_output'};underlying=model.model;emb=underlying.bert.embeddings
 with torch.inference_mode():
  for group,module_path,module in encoder_linears(underlying):
   name='layer'+module_path.split('.')[3]+'_'+suffix['.'.join(module_path.split('.')[4:])];need(isinstance(module,GateQuantLinear) and module.weight_bits==precision[group],'Batch actual GateQuantLinear precision differs');bits=module.weight_bits
   check(name+'_float_weight',module.weight);check(name+'_code_i8',quantize_codes(module.weight,getattr(module,f'weight_scale_{bits}'),bits),True);check(name+'_params',torch.cat((module.input_scale.reshape(1),getattr(module,f'weight_scale_{bits}').reshape(-1),module.bias.reshape(-1))))
   for width in (4,8):check(name+f'_weight_scale_{width}',getattr(module,f'weight_scale_{width}'))
  check('embedding_ln',torch.cat((emb.LayerNorm.weight,emb.LayerNorm.bias)));check('embedding_position',emb.position_embeddings.weight[:256])
  for i,block in enumerate(underlying.bert.encoder.layer):
   for name,module in [('attention_ln',block.attention.output.LayerNorm),('output_ln',block.output.LayerNorm)]:check(f'layer{i}_'+name,torch.cat((module.weight,module.bias)))
  for name,module in [('pooler',underlying.bert.pooler.dense),('classifier',underlying.classifier)]:check(name+'_weight',module.weight);check(name+'_bias',module.bias)
  for item in items:
   for name,table,key in [('embedding_word',emb.word_embeddings.weight,'input_ids'),('embedding_type',emb.token_type_embeddings.weight,'token_type_ids')]:need(raw(table[torch.tensor(item['inputs'][key])])==(ROOT/item['folder']/(name+'.bin')).read_bytes(),'Actual batch input embedding/B1 association differs')
 folder.mkdir(exist_ok=True);dispatch=folder/f'batch_dispatch_{len(list(folder.glob("batch_dispatch_*.py")))+1:03d}.py';write(dispatch,Path(__file__).read_bytes());settings=dict(device='cpu',dtype='float32',threads=4,autocast=False,tf32=False,torch_version=torch.__version__);need(state.get('settings',settings)==settings,'Saved actual batch settings changed');state['settings']=settings;save(file,state);count=0
 for size in ([8,32] if args.batch_size=='both' else [int(args.batch_size)]):
  start=positions[size]
  while start<66 and count<args.max_new_batches:
   paused();current=items[start:start+size];relative=Path(f'b{size}')/f'batch{start//size:03d}';target=folder/relative;need(not target.exists(),'Preserve interrupted/actual batch evidence');target.mkdir(parents=True)
   active=dict(configured_batch_size=size,start_window=start,actual_batch_size=len(current),attempt_folder=relative.as_posix(),members=batch_members(current));state['active']=active;save(file,state)
   try:
    batch={k:torch.tensor([item['inputs'][k] for item in current],dtype=torch.long) for k in ('input_ids','attention_mask','token_type_ids')};begun=time.perf_counter()
    with torch.inference_mode():result=model(**batch,capture_intermediates=True)
    elapsed=time.perf_counter()-begun;inventory=batch_capture_inventory(result.intermediates);save(target/'capture_names.json',inventory);need(inventory['exact_membership'],'Actual required31 plus declared4 diagnostic capture membership differs')
    for wi,item in enumerate(current):
     for name,reference,integer,extent in batch_tensors():write(target/f'window{wi:03d}'/(name+'.bin'),raw(result.intermediates[name][wi],integer))
    checks=batch_checks(target,current);record=dict(**active,capture_inventory=inventory,checks=checks,passed=all(v['exact'] for v in checks),forward_elapsed_seconds=elapsed,first_mismatch=next((v for v in checks if not v['exact']),None),numerical_acceptance=None);save(target/'batch.json',record);state['batches'].append(record);state['active']=None;save(file,state)
    need(record['passed'],'First actual new-map batch/B1 mismatch; preserved all captured bytes')
   except Exception as error:state.update(state='stopped_first_mismatch' if (target/'batch.json').exists() else 'stopped_process_capture_failure',error=str(error));save(file,state);raise
   start+=len(current);positions[size]=start;count+=1;state['complete_b32']=positions[32]==66;state['state']='complete_b32_pending_review' if state['complete_b32'] else 'bounded_handoff';save(file,state)
 print(json.dumps(dict(new_actual_batches=count,completed_windows=positions,complete_b32=state['complete_b32'],score_acceptance=None)))

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--name',required=True);ap.add_argument('--precision-map',required=True);ap.add_argument('--source',choices=('ptq','qat'),required=True);ap.add_argument('--qat-checkpoint');ap.add_argument('--execute',action='store_true');ap.add_argument('--stage',choices=('export','batches','validate-batches'),default='export');ap.add_argument('--batch-size',choices=('8','32','both'),default='32');ap.add_argument('--max-new-batches',type=int,default=1);ap.add_argument('--max-new-windows',type=int,default=1);a=ap.parse_args();need(a.max_new_windows>0 and a.max_new_batches>0,'Positive bounded B1/batch count');precision,out,run,calibration,qat,selection=request(a)
 if not a.execute:print(json.dumps(dict(execute=False,stage=a.stage,batch_size=a.batch_size,actual_batch_forwards_required={str(b):(66+b-1)//b for b in ([8,32] if a.batch_size=='both' else [int(a.batch_size)])},output=str(out),precision_map=precision,source=a.source,matching_QAT_metadata=qat is not None,cases=32,windows=66,source_checkpoint=run['selected_checkpoint'],frozen_training_calibration=run['calibration_file'],existing_export=(out/'export.json').exists(),model_imported=False,prerequisite='Explicit coordinator release; no concurrent Torch or occupied C resource slots; same frozen inputs/settings',score_acceptance=None),indent=2));return
 if a.stage=='export':execute(a)
 else:execute_batches(a,validate_only=a.stage=='validate-batches')
if __name__=='__main__':main()