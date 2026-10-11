"""Two real frozen development documents at true L128/L512; preview by default.

CPU stages share existing W8-QAT/ordinary-floating checkpoints and settings.
B8/B32 use explicitly repeated real window rows and compare all required saved
B1 tensors. C stages own new build/data trees; canonical L256 files stay intact.
No score acceptance, broad shape qualification, training, board or latency claim.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,copy,json,math,os,re,shutil,struct,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
DATA=ROOT/'results/goal4/numerical_bridge/real_length_bringup_v1'
TIE='aeslc:train:panus-s_inbox_1.subject:clean'
LINEARS={'query':256,'key':256,'value':256,'attention_output':256,'ffn_input':256,'ffn_output':1024}
NORMS=('embedding_ln','layer0','layer1','layer2','layer3','logits','risk')

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def need(ok,text):
 if not ok:raise RuntimeError(text)
def paused():
 need(not any((ROOT/p).exists() for p in ('results/goal2/pause.request','results/goal3/pause.request')),'Pause marker present')
def linux(p):
 value=str(p).replace('\\','/');prefix='//wsl.localhost/Ubuntu';return value[len(prefix):] if value.startswith(prefix) else value
def spec(mode):
 return ([] if mode=='floating' else [(f'layer{i}_{n}_a8',f'layer{i}_{n}_reference_a8',True) for i in range(4) for n in LINEARS])+[(n,'reference_'+n,False) for n in NORMS]
def raw(t,integer=False):
 import torch
 t=t.detach().cpu().to(torch.int8 if integer else torch.float32).contiguous()
 need(integer or bool(torch.isfinite(t).all()),'Nonfinite captured tensor');return t.numpy().astype('i1' if integer else '<f4',copy=False).tobytes()
def write(p,value):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():need(p.read_bytes()==value,'Existing source/output bytes differ: '+str(p))
 else:p.write_bytes(value)
def documents():
 selection=read(ROOT/'results/goal4/numerical_bridge/development_cases_v1/selection.json')
 chosen=[next(c for c in selection['cases'] if c['document_id']==TIE),next(c for c in selection['cases'] if c['case_index']==18)]
 need(chosen[1]['label']==1 and chosen[1]['window_count']>=3,'Frozen attacked long case18 required')
 wanted={c['document_id']:c for c in chosen};found={}
 for partition in sorted({c['partition'] for c in chosen}):
  with (ROOT/'data/prepared'/(partition+'.jsonl')).open(encoding='utf-8') as f:
   for line in f:
    doc=json.loads(line)
    if doc['document_id'] not in wanted:continue
    c=wanted[doc['document_id']]
    need(doc['label']==c['label'] and doc['source_group_id']==c['source_group_id'] and doc['split']=='search' and c['partition']==partition,'Frozen prepared document membership changed')
    need(doc['document_id'] not in found,'Duplicate selected prepared document');found[doc['document_id']]=doc
 need(set(found)==set(wanted),'Missing selected real development document')
 return selection,chosen,[found[c['document_id']] for c in chosen]
def plan_path(length):return DATA/f'plan_l{length}.json'
def mode_dir(length,mode):return DATA/f'l{length}_{mode}'
def prepare(args):
 paused();selection,chosen,docs=documents()
 from gate_eval import load_config,window_document
 from transformers import AutoTokenizer
 config=load_config();shape=copy.deepcopy(config);shape['windowing']['primary_length']=args.length
 source=read(ROOT/'results/goal3/run.json')['selected_checkpoint'];source=ROOT/Path(source.replace('\\','/'))
 tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True,use_fast=True)
 windows=[];document_rows=[]
 for case,doc in zip(chosen,docs):
  real=window_document(tokenizer,doc['text'],doc['document_id'],doc['label'],doc.get('injection_spans'),shape)
  members=[]
  for window in real:
   fields={k:window[k] for k in ('input_ids','attention_mask','token_type_ids')}
   padded=tokenizer.pad([fields],padding='max_length',max_length=args.length,return_tensors=None)
   inputs={k:[int(v) for v in padded[k][0]] for k in fields};key=f"c{case['case_index']:02d}_w{window['window_index']:03d}"
   item=dict(key=key,case_index=case['case_index'],document_id=case['document_id'],source_group_id=case['source_group_id'],partition=case['partition'],label=case['label'],window_index=window['window_index'],inputs=inputs,
    token_span={k:window[k] for k in ('content_token_start','content_token_end','content_token_count','total_token_count','char_start','char_end','window_label')},last_valid_position=max(i for i,v in enumerate(inputs['attention_mask']) if v))
   windows.append(item);members.append(key)
  document_rows.append(dict(case_index=case['case_index'],document_id=case['document_id'],label=case['label'],windows=members))
 if args.length==512:need(any(w['case_index']==18 and w['inputs']['attention_mask'][511]==1 for w in windows),'Real attacked512 window must genuinely reach position511')
 batches={str(b):[[windows[(start+row)%len(windows)]['key'] for row in range(b)] for start in range(0,len(windows),b)] for b in (8,32)}
 plan=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),length=args.length,selection=selection,selected_cases=chosen,prepared_documents=docs,
  original_config=config,shape_windowing=shape['windowing'],tokenizer_checkpoint=str(source),documents=document_rows,windows=windows,batches=batches,
  batch_policy='Full B8/B32; final batch cycles the declared real window list to fill all rows. Every selected window receives B1; repeats are comparisons only.',
  scope='Two real documents at true requested length; per-document max resets across their regenerated full window lists',model_forward_executed=False,score_acceptance=None)
 path=plan_path(args.length);need(not path.exists(),'Preserve existing real-length plan');save(path,plan);print(json.dumps(dict(prepared=True,plan=str(path),documents=document_rows,windows=len(windows),last_valid_positions=[w['last_valid_position'] for w in windows],batch_rows=batches),indent=2))
def load_plan(length):
 plan=read(plan_path(length));selection,chosen,docs=documents()
 need(plan['length']==length and plan['selection']==selection and plan['selected_cases']==chosen and plan['prepared_documents']==docs,'Real length plan/document source association changed')
 return plan

def model_and_parameters(args,plan,out):
 from score_qualified_development import load_model
 from gate_fpga_shapes import OrderedBertMiniShapes
 from gate_eval import load_config
 from gate_quant import encoder_linears,quantize_codes
 import torch
 config=load_config();need(config['models']['detector']==plan['original_config']['models']['detector'] and all(config[k]==plan['original_config'][k] for k in ('windowing','evaluation','data')),'Relevant scientific protocol/model config changed since length plan')
 floating=args.mode=='floating';candidate=read(ROOT/'results/goal3/qat_w8_a8/candidate.json');precision=None if floating else candidate['identity']['precision_map']
 need(floating or set(precision.values())=={8},'Require frozen W8-QAT map')
 base,tokenizer,identity=load_model(config,precision,None if floating else 'checkpoints/quantized/qat_w8_a8',floating=floating)
 model=OrderedBertMiniShapes(base.model,sequence_length=args.length,encoder_mode='floating' if floating else 'quantized')
 identity=dict(identity,source_arithmetic_identity=identity['arithmetic_identity'],arithmetic_identity=model.arithmetic_identity,windowing=plan['shape_windowing'])
 settings=dict(torch_version=torch.__version__,threads=torch.get_num_threads(),device=str(next(model.model.parameters()).device),default_dtype=str(torch.get_default_dtype()),autocast=torch.is_autocast_enabled('cpu'),cuda_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,flush_denormal_requested=False)
 need(settings['threads']==4 and settings['device']=='cpu' and settings['default_dtype']=='torch.float32' and not any(settings[k] for k in ('autocast','cuda_tf32','cudnn_tf32')),'Established CPU settings changed')
 old=ROOT/'results/goal4/numerical_bridge'/('development_qualification_floating_v1' if floating else 'development_qualification_w8_v1')/'parameters';common=out/'parameters'
 def parameter(name,t,integer=False):
  value=raw(t,integer)
  if name!='embedding_position':need(value==(old/(name+'.bin')).read_bytes(),'Loaded frozen parameter differs: '+name)
  else:
   prefix=min(args.length,256)*256*4;need(value[:prefix]==(old/'embedding_position.bin').read_bytes()[:prefix],'Loaded frozen position prefix differs')
  write(common/(name+'.bin'),value)
 suffix={'attention.self.query':'query','attention.self.key':'key','attention.self.value':'value','attention.output.dense':'attention_output','intermediate.dense':'ffn_input','output.dense':'ffn_output'}
 with torch.inference_mode():
  for _,path,module in encoder_linears(model.model):
   name='layer'+path.split('.')[3]+'_'+suffix['.'.join(path.split('.')[4:])]
   if floating:parameter(name+'_float_weight',module.weight);parameter(name+'_float_bias',module.bias)
   else:
    parameter(name+'_w8',quantize_codes(module.weight,module.weight_scale_8,8),True)
    parameter(name+'_params',torch.cat((module.input_scale.reshape(1),module.weight_scale_8.reshape(-1),module.bias.reshape(-1))))
  emb=model.model.bert.embeddings;parameter('embedding_ln',torch.cat((emb.LayerNorm.weight,emb.LayerNorm.bias)))
  parameter('embedding_position',emb.position_embeddings.weight[:args.length])
  for i,block in enumerate(model.model.bert.encoder.layer):
   for name,module in [('attention_ln',block.attention.output.LayerNorm),('output_ln',block.output.LayerNorm)]:parameter(f'layer{i}_'+name,torch.cat((module.weight,module.bias)))
  for name,module in [('pooler',model.model.bert.pooler.dense),('classifier',model.model.classifier)]:parameter(name+'_weight',module.weight);parameter(name+'_bias',module.bias)
  parameter('threshold',torch.tensor([plan['selection']['sampling']['historical_operating_threshold']]))
 return model,identity,settings

def cpu_stage(args):
 paused();plan=load_plan(args.length);out=mode_dir(args.length,args.mode);out.mkdir(parents=True,exist_ok=True)
 for source in (Path(__file__),ROOT/'gate_fpga_shapes.py',ROOT/'gate_fpga_reference.py',ROOT/'gate_quant.py'):write(out/source.name,source.read_bytes())
 statepath=out/'cpu.json';state=read(statepath) if statepath.exists() else dict(length=args.length,mode=args.mode,plan=plan,b1=[],batches=[],score_acceptance=None,complete=False)
 need(state['plan']==plan and not state.get('stopped'),'Saved shape discrepancy or plan change requires review')
 model,identity,settings=model_and_parameters(args,plan,out)
 need(state.get('identity',identity)==identity and state.get('settings',settings)==settings,'Shape model/settings changed');state.update(identity=identity,settings=settings)
 import torch
 bykey={w['key']:w for w in plan['windows']};count=0
 def batch(keys):return {k:torch.tensor([bykey[key]['inputs'][k] for key in keys],dtype=torch.long) for k in ('input_ids','attention_mask','token_type_ids')}
 if args.stage=='b1':
  done={v['key'] for v in state['b1']}
  for item in plan['windows']:
   if item['key'] in done:continue
   if count>=args.max_forwards:break
   paused();folder=out/'b1'/item['key'];need(not folder.exists(),'Preserve interrupted/uncommitted B1 output');folder.mkdir(parents=True)
   tensors=batch([item['key']]);emb=model.model.bert.embeddings
   with torch.inference_mode():
    write(folder/'embedding_word.bin',raw(emb.word_embeddings(tensors['input_ids'])));write(folder/'embedding_type.bin',raw(emb.token_type_embeddings(tensors['token_type_ids'])));write(folder/'key_mask.bin',raw((1.-tensors['attention_mask'].float())*torch.finfo(torch.float32).min))
    start=time.perf_counter();result=model(**tensors,capture_intermediates=True);elapsed=time.perf_counter()-start
   for name,filename,integer in spec(args.mode):write(folder/(filename+'.bin'),raw(result.intermediates[name][0],integer))
   entry=dict(key=item['key'],folder=folder.relative_to(ROOT).as_posix(),inputs=item['inputs'],document_id=item['document_id'],case_index=item['case_index'],window_index=item['window_index'],label=item['label'],risk=float(result.risk.item()),logits=result.logits.tolist()[0],forward_elapsed_seconds=elapsed,acceptance=None)
   save(folder/'window.json',entry);state['b1'].append(entry);count+=1;save(statepath,state)
 else:
  baseline={v['key']:v for v in state['b1']};need(set(baseline)==set(bykey),'Complete real-window B1 outputs required before B8/B32')
  done={(v['size'],v['index']) for v in state['batches']}
  for size in (8,32):
   for index,keys in enumerate(plan['batches'][str(size)]):
    if (size,index) in done:continue
    if count>=args.max_forwards:break
    paused();folder=out/f'b{size}'/f'batch{index:03d}';need(not folder.exists(),'Preserve interrupted/uncommitted batch');folder.mkdir(parents=True)
    start=time.perf_counter()
    with torch.inference_mode():result=model(**batch(keys),capture_intermediates=True)
    elapsed=time.perf_counter()-start;checks=[];risks=[]
    for row,key in enumerate(keys):
     destination=folder/f'row{row:03d}';risks.append(float(result.risk[row]))
     for name,filename,integer in spec(args.mode):
      actual=raw(result.intermediates[name][row],integer);expected=(ROOT/baseline[key]['folder']/(filename+'.bin')).read_bytes();write(destination/(name+'.bin'),actual)
      checks.append(dict(row=row,key=key,tensor=name,bytes=len(actual),exact=actual==expected,unequal_bytes=sum(a!=b for a,b in zip(actual,expected)),extent_equal=len(actual)==len(expected)))
    entry=dict(size=size,index=index,members=keys,repeated_rows=[row for row in range(size) if index*size+row>=len(bykey)],risks=risks,checks=checks,passed=all(v['exact'] and v['extent_equal'] for v in checks),forward_elapsed_seconds=elapsed,acceptance=None)
    save(folder/'batch.json',entry);state['batches'].append(entry);count+=1
    if not entry['passed']:state['stopped']='First unexplained batch tensor discrepancy';save(statepath,state);raise RuntimeError(state['stopped'])
    save(statepath,state)
   if count>=args.max_forwards:break
 state['b1_complete']=len(state['b1'])==len(bykey);state['batches_complete']=len(state['batches'])==sum(len(plan['batches'][str(b)]) for b in (8,32));state['complete']=state['b1_complete'] and state['batches_complete']
 if state['batches_complete']:
  state['batch_document_max']={}
  for size in (8,32):
   actual={}
   for entry in state['batches']:
    if entry['size']==size:
     for key,risk in zip(entry['members'],entry['risks']):
      need(key not in actual or actual[key]==risk,'Repeated actual row risk changed');actual[key]=risk
   state['batch_document_max'][str(size)]=[dict(document_id=d['document_id'],windows=d['windows'],actual_max_risk=max(actual[k] for k in d['windows']),aggregation='Fresh actual captured row max per document') for d in plan['documents']]
 if state['b1_complete']:
  baseline={v['key']:v for v in state['b1']};state['documents']=[dict(document_id=d['document_id'],windows=d['windows'],b1_max_risk=max(baseline[k]['risk'] for k in d['windows']),aggregation='Fresh max per document across all genuine regenerated windows') for d in plan['documents']]
 save(statepath,state);print(json.dumps(dict(stage=args.stage,new_forwards=count,b1_complete=state['b1_complete'],batches_complete=state['batches_complete'],acceptance=None)))

# These finite caps use actual primary L256 C timings (source-review evidence),
# not measured L128/L512 timings. Compilation includes its first C invocation.
C_WINDOW_TIMEOUT=1800
C_COMPILE_TIMEOUT=3600

def shape_replacements(length):
 need(length in (128,512),'Only the two bounded real-length routes are supported')
 m='std::min(128,CAP/L)'
 return [('constexpr int L=256,',f'constexpr int L={length},'),('constexpr int M=CAP/L;',f'constexpr int M={m};'),('(floating_reference?267:411)',f'(34LL*(L/128)+16LL*((L+{m}-1)/{m})*(L/64+2)+7+(floating_reference?0LL:72LL*(L/128)))')]

def generate_shape_tb(source,length):
 """Three declared shape substitutions, five bounds insertions, two writes."""
 from run_qualification_cases import generate_bounds_tb
 for old,new in shape_replacements(length):
  need(source.count(old)==1,'Canonical C template seam changed: '+old);source=source.replace(old,new,1)
 source=generate_bounds_tb(source)
 before='        pooled=fp(gate::FP32_TANH,pooled,{},{},1,D);'
 need(source.count(before)==1,'Actual TANH output-capture seam changed')
 return source.replace(before,'        write_binary("diagnostic_pooler_pre_tanh",pooled);\n'+before+'\n        write_binary("diagnostic_pooler_post_tanh",pooled);',1)

def shape_tcl(build,config):
 q=lambda p:'{'+linux(p)+'}';flags='-std=c++14 -O2 -ffp-contract=off -I'+linux(build)
 return '\n'.join(['open_project '+q(build/'project'),'set_top gate_fixed_fp32_top','add_files '+q(build/'gate_fixed_fp32_top.cpp')+' -cflags {'+flags+'}','add_files -tb '+q(build/'numerical_bridge_tb.cpp')+' -cflags {'+flags+'}','open_solution solution -flow_target '+config['hardware']['hls']['flow_target'],'set_part {'+config['hardware']['part']+'}','create_clock -period 5','config_compile -unsafe_math_optimizations=false','csim_design','exit'])+'\n'

def same_functions(copied,current,names):
 def selected(path):
  tree=ast.parse(path.read_text(encoding='utf-8-sig'))
  return {name:ast.dump(next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)) for name in names}
 need(selected(copied)==selected(current),'Declared generation/arithmetic function source changed')

def same_files(original,snapshot):
 names=lambda path:sorted(p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file())
 need(names(original)==names(snapshot),'Frozen scientific input/parameter file membership changed')
 for name in names(original):need((original/name).read_bytes()==(snapshot/name).read_bytes(),'Frozen scientific input/parameter bytes changed: '+name)

def validate_shape_inputs(args,plan,cpu,out,build=None):
 from validate_qualification_exports import common_parameters,terminal
 need(cpu.get('b1_complete') is True and cpu['length']==args.length and cpu['mode']==args.mode and cpu['plan']==plan,'Complete matching real B1 exports required')
 expected={w['key']:w for w in plan['windows']}
 need(len(cpu['b1'])==len(expected) and [v['key'] for v in cpu['b1']]==list(expected),'B1 membership/order differs from regenerated real windows')
 need(len(plan['documents'])==2 and [d['case_index'] for d in plan['documents']]==[1,18],'Only the two frozen development documents are allowed')
 # These source snapshots establish the actual B1 capture route; dispatch-only
 # changes to this CLI do not invalidate historical CPU capture arithmetic.
 same_functions(out/'bringup_detector_shapes.py',Path(__file__),('model_and_parameters','cpu_stage','raw','spec'))
 for name in ('gate_fpga_shapes.py','gate_fpga_reference.py','gate_quant.py'):
  need((out/name).read_bytes()==(ROOT/name).read_bytes(),'Frozen B1 arithmetic source changed: '+name)
 sizes=common_parameters(args.mode=='floating');sizes['embedding_position.bin']=args.length*256*4
 parameters=out/'parameters'
 need({p.name for p in parameters.iterdir() if p.is_file()}==set(sizes),'Common parameter membership differs')
 for name,size in sizes.items():need((parameters/name).stat().st_size==size,'Common parameter extent differs: '+name)
 threshold=struct.pack('<f',plan['selection']['sampling']['historical_operating_threshold'])
 need((parameters/'threshold.bin').read_bytes()==threshold,'Frozen historical threshold changed')
 for entry in cpu['b1']:
  item=expected[entry['key']];folder=ROOT/entry['folder']
  need(folder.resolve()==(out/'b1'/entry['key']).resolve() and read(folder/'window.json')==entry,'B1 output path/record differs')
  need(all(entry[k]==item[k] for k in ('document_id','case_index','window_index','label','inputs')),'B1 prepared window association differs')
  need(all(len(item['inputs'][k])==args.length for k in ('input_ids','attention_mask','token_type_ids')),'Real shape input extent differs')
  terminal(folder,'reference',entry['logits'],entry['risk'])
  wanted={'window.json','embedding_word.bin','embedding_type.bin','key_mask.bin'}
  for name,filename,integer in spec(args.mode):
   wanted.add(filename+'.bin');size=args.length*LINEARS[name.split('_',1)[1].removesuffix('_a8')] if integer else (8 if name=='logits' else 4 if name=='risk' else args.length*256*4)
   need((folder/(filename+'.bin')).stat().st_size==size,'Real B1 capture extent differs: '+filename)
  need({p.name for p in folder.iterdir() if p.is_file()}==wanted,'B1 scientific input membership differs')
  for name,size in [('embedding_word',args.length*256*4),('embedding_type',args.length*256*4),('key_mask',args.length*4)]:
   need((folder/(name+'.bin')).stat().st_size==size,'B1 embedding/mask extent differs')
  mask=b''.join(struct.pack('<f',-0.0 if v else -3.4028234663852886e38) for v in item['inputs']['attention_mask'])
  need((folder/'key_mask.bin').read_bytes()==mask,'B1 real attention mask bytes differ')
  need(not any((folder/name).exists() for name in sizes),'Window shadows shared parameter file')
  if build is not None:same_files(folder,build/'reference_inputs'/entry['key'])
 if build is not None:same_files(parameters,build/'parameters')
 return dict(windows=len(expected),documents=2,parameter_files=len(sizes),input_source_association_checked=True)

def c_template(args):
 plan=load_plan(args.length);out=mode_dir(args.length,args.mode);cpu=read(out/'cpu.json')
 validate_shape_inputs(args,plan,cpu,out)
 build=ROOT/'build'/f'detector_shape_bringup_l{args.length}_{args.mode}_v1';need(not build.exists(),'Preserve existing fresh C build')
 canonical=ROOT/'hardware/numerical_bridge_tb.cpp';text=generate_shape_tb(canonical.read_text(),args.length)
 source=ROOT/'build/fixed_fp32_hls';synthesis=read(source/'run.json');need(synthesis['synthesis_passed'],'Actual successful existing common FP C service required')
 build.mkdir();write(build/'canonical_numerical_bridge_tb.cpp',canonical.read_bytes());write(build/'numerical_bridge_tb.cpp',text.encode())
 write(build/'checker_generator.py',(ROOT/'scripts/run_qualification_cases.py').read_bytes());write(build/'shape_generator.py',Path(__file__).read_bytes());write(build/'build_fixed_fp32_hls.py',(ROOT/'scripts/build_fixed_fp32_hls.py').read_bytes())
 headers=re.findall(r'^#include "([A-Za-z0-9_.-]+)"',text,re.M)
 for name in headers:write(build/name,(ROOT/'hardware'/name).read_bytes())
 write(build/'gate_fixed_fp32_top.cpp',(source/'gate_fixed_fp32_top.cpp').read_bytes())
 shutil.copytree(out/'parameters',build/'parameters')
 for entry in cpu['b1']:shutil.copytree(ROOT/entry['folder'],build/'reference_inputs'/entry['key'])
 config=read(ROOT/'configs/project.json');write(build/'shape_csim.tcl',shape_tcl(build,config).encode())
 save(build/'template.json',dict(length=args.length,mode=args.mode,canonical_source='hardware/numerical_bridge_tb.cpp',replacements=[dict(old=a,new=b) for a,b in shape_replacements(args.length)],headers=headers,
  expected_calls=({128:177,512:1711} if args.mode=='w8' else {128:105,512:1423})[args.length],identity=cpu['identity'],b1=cpu['b1'],synthesis_recorded_utc=synthesis['recorded_utc'],
  window_timeout_seconds=C_WINDOW_TIMEOUT,compile_timeout_seconds=C_COMPILE_TIMEOUT,tail_capture_writes=['diagnostic_pooler_pre_tanh','diagnostic_pooler_post_tanh'],
  scope='Actual common FP C service; true requested attention length; not full RTL',operator_checker_binding='Unconditional connected_fp32_bounds.hpp; every actual call validated',
  input_snapshots='Only the two real documents regenerated at this length: their committed B1 files plus one common parameter copy',new_synthesis=False,acceptance=None))
 print(json.dumps(dict(staged=True,build=str(build),compile_blocker='Explicit root review and resource release required; taskset -c 6 or 7',acceptance=None),indent=2))

def validate_shape_build(args,plan,cpu,out,build):
 template=read(build/'template.json');config=read(ROOT/'configs/project.json');source=ROOT/'build/fixed_fp32_hls';synthesis=read(source/'run.json')
 need(template['identity']==cpu['identity'] and template['b1']==cpu['b1'] and template['length']==args.length and template['mode']==args.mode,'C template/B1 identity differs')
 need(template['expected_calls']==({128:177,512:1711} if args.mode=='w8' else {128:105,512:1423})[args.length] and template['tail_capture_writes']==['diagnostic_pooler_pre_tanh','diagnostic_pooler_post_tanh'],'Shape call count/capture contract differs')
 need((template['window_timeout_seconds'],template['compile_timeout_seconds'])==(C_WINDOW_TIMEOUT,C_COMPILE_TIMEOUT),'Finite C timeout contract differs')
 canonical=ROOT/'hardware/numerical_bridge_tb.cpp'
 need((build/'canonical_numerical_bridge_tb.cpp').read_bytes()==canonical.read_bytes(),'Canonical source changed')
 need(template['replacements']==[dict(old=a,new=b) for a,b in shape_replacements(args.length)] and (build/'numerical_bridge_tb.cpp').read_bytes()==generate_shape_tb(canonical.read_text(),args.length).encode(),'Generated true-length bounds/capture source changed')
 need(template['headers']==re.findall(r'^#include "([A-Za-z0-9_.-]+)"',(build/'numerical_bridge_tb.cpp').read_text(),re.M),'Generated include membership changed')
 for name in template['headers']:need((build/name).read_bytes()==(ROOT/'hardware'/name).read_bytes(),'Checker/arithmetic header changed: '+name)
 same_functions(build/'checker_generator.py',ROOT/'scripts/run_qualification_cases.py',('generate_bounds_tb',))
 same_functions(build/'shape_generator.py',Path(__file__),('shape_replacements','generate_shape_tb','shape_tcl'))
 need((build/'build_fixed_fp32_hls.py').read_bytes()==(ROOT/'scripts/build_fixed_fp32_hls.py').read_bytes(),'FP service generator changed')
 tree=ast.parse((ROOT/'scripts/build_fixed_fp32_hls.py').read_text(encoding='utf-8-sig'))
 tops=[n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='top' and n.func.attr=='write_text' and n.args and isinstance(n.args[0],ast.Constant)]
 need(len(tops)==1 and (build/'gate_fixed_fp32_top.cpp').read_bytes()==tops[0].encode()==(source/'gate_fixed_fp32_top.cpp').read_bytes(),'Actual generated/synthesized FP top changed')
 need(synthesis['synthesis_passed'] and template['synthesis_recorded_utc']==synthesis['recorded_utc'] and (source/'fixed_fp32_service.hpp').read_bytes()==(ROOT/'hardware/fixed_fp32_service.hpp').read_bytes(),'Existing synthesized FP service association changed')
 need((build/'shape_csim.tcl').read_bytes()==shape_tcl(build,config).encode(),'C compilation flags/part/clock changed')
 validate_shape_inputs(args,plan,cpu,out,build)
 return template

def validate_shape_process(args,item,folder,build,template,process):
 from run_qualification_cases import ConnectedCMemoryGuard
 config=read(ROOT/'configs/project.json');out=mode_dir(args.length,args.mode)
 need(process['identity']==template['identity'] and process['length']==args.length and process['mode']==args.mode and process['window']==item,'Saved process identity/input window differs')
 need(process['build']==build.relative_to(ROOT).as_posix() and process['input_folder']==item['folder'] and process['frozen_input_folder']==(build/'reference_inputs'/item['key']).relative_to(ROOT).as_posix() and process['parameter_folder']==(build/'parameters').relative_to(ROOT).as_posix(),'Saved process build/input/parameter association differs')
 need(process['cpu_affinity'] in ([6],[7]) and process['resource_partition']==config['execution_control']['board_free_continuation']['resource_partition'],'Saved process resource partition differs')
 guard=ConnectedCMemoryGuard(config,out/'c','shape_validation',active=False)
 path=(ROOT/process['memory_guard']).resolve();need(path.is_relative_to((out/'c').resolve()) and path.is_file(),'Actual memory guard record absent/outside mode tree')
 observations=process['guard_observations'];all_records=[json.loads(line) for line in path.read_text().splitlines()]
 start=process['guard_start_line'];need(all_records[start:start+len(observations)]==observations,'Actual guard observations changed')
 need(observations and all(v.get('passed') is not False for v in observations) and not any(v['stage']=='assigned_child_stop' for v in observations),'Failed actual memory/process guard cannot be resolved')
 children=[v for v in observations if v['stage']=='child_started']
 need(len(children)==1 and children[0]['child_pid']==children[0]['child_group']==process['child_pid']==process['child_group'],'Assigned child group association differs')
 need(any(v['stage']=='postflight' and v.get('passed') is True and v['child_pid']==process['child_pid'] for v in observations),'Actual postflight memory observation absent')
 for record in observations:
  if 'passed' in record:
   need(record['limits']==guard.limits and record['cpu_affinity']==process['cpu_affinity'] and record['wsl_available_kib']>=guard.limits['minimum_wsl_available_kib'] and record['windows_free_kib']>=guard.limits['minimum_windows_free_kib'],'Actual memory observation violates original floors')
 command=process['command']
 normal=[str(build/'project/solution/csim/build/csim.exe')]
 compile_command=[config['hardware']['hls']['tool_executable'],'--mode','hls','--tcl',linux(build/'shape_csim.tcl')]
 need(command in (normal,compile_command) and process['timeout_seconds']==(C_COMPILE_TIMEOUT if command==compile_command else C_WINDOW_TIMEOUT),'Saved C command/cap differs')
 same_functions(folder/'dispatch_connected_runner.py',ROOT/'scripts/run_qualification_cases.py',('run','ConnectedCMemoryGuard','generate_bounds_tb','paused','validate_operator_bounds'))
 same_functions(folder/'dispatch_shape_runner.py',Path(__file__),('generate_shape_tb','shape_replacements','shape_tcl','shape_case','validate_shape_process','c_stage_locked','c_stage','shape_discrepancy','shape_documents','shape_compilation_completed'))
 return process

def shape_case(args,item,folder,template):
 """Read-only actual saved case verification; no dispatch or auto-resolution."""
 from run_qualification_cases import validate_operator_bounds
 from validate_qualification_exports import terminal
 process=read(folder/'csim_run.json');need(process['state']=='returned' and process['returncode']==0,'Process/service failure cannot be resolved')
 build=ROOT/process['build'];validate_shape_process(args,item,folder,build,template,process)
 summary=read(folder/'bridge_summary.json');reference=ROOT/item['folder']
 need(summary['completed'] is True and summary['case']==item['document_id'],'Actual C completion identity/status differs')
 need(summary['encoder_arithmetic']==('ordinary_floating_reference' if args.mode=='floating' else 'w8_a8_integer_contract'),'Actual C arithmetic mode differs')
 need(sum(summary['service_calls_by_op'])==template['expected_calls'],'Actual true-length service count differs')
 bounds=validate_operator_bounds(folder,summary)
 need(summary['same_input_quantization_exact']==(None if args.mode=='floating' else True) and summary['integer_accumulators_exact']==(None if args.mode=='floating' else True),'Same-input numerical failure cannot be reviewed away')
 need(summary['same_input_codes_checked']==bounds['checked_values_by_op'][1] and summary['exact_accumulators_checked']==bounds['checked_values_by_op'][2],'Same-input code/accumulator coverage differs')
 terminal(folder,'bridge',summary['logits'],summary['risk']);terminal(reference,'reference',summary['reference_logits'],summary['reference_risk'])
 need(summary['reference_logits']==item['logits'] and summary['reference_risk']==item['risk'],'Saved B1 terminal association differs')
 need(summary['risk_absolute_error']==abs(summary['risk']-summary['reference_risk']) and summary['end_to_end_score_tolerance'] is None,'Saved terminal error/tolerance record differs')
 threshold=struct.unpack('<f',(folder.parents[1]/'parameters/threshold.bin').read_bytes())[0]
 need(summary['threshold']==threshold and summary['strict_reject']==(summary['risk']>threshold) and summary['reference_strict_reject']==(summary['reference_risk']>threshold),'Frozen threshold/strict decisions differ')
 need(summary['strict_decision_agrees'] is True and summary['strict_reject']==summary['reference_strict_reject'],'Strict decision disagreement cannot be resumed')
 need(summary['source_revision']==read(ROOT/'build/fixed_fp32_hls/run.json')['service_schedule_revision'],'Actual service schedule revision differs')
 checks=[]
 for layer in summary['integer_layers']:
  name=layer['name'];actual=(folder/('actual_'+name+'_a8.bin')).read_bytes();expected=(reference/(name+'_reference_a8.bin')).read_bytes()
  size=args.length*LINEARS[name.split('_',1)[1]]
  need(len(actual)==len(expected)==size==layer['input_codes'],'Actual A8 extent differs')
  count=sum(a!=b for a,b in zip(actual,expected));need(count==layer['original_software_code_differences'],'Actual A8 summary differs')
  checks.append(dict(name=name,values=len(actual),differences=count))
 need([v['name'] for v in checks]==([] if args.mode=='floating' else [f'layer{i}_{name}' for i in range(4) for name in LINEARS]),'Actual integer layer membership/order differs')
 need(sum(v['accumulators_checked'] for v in summary['integer_layers'])==summary['exact_accumulators_checked'],'Actual accumulator summary count differs')
 need(summary['same_input_codes_checked']==summary['exact_accumulators_checked']==(0 if args.mode=='floating' else args.length*9216),'True-length same-input integer coverage differs')
 need([v['name'] for v in summary['intermediate_errors']]==list(NORMS),'Seven actual boundaries required')
 for metric in summary['intermediate_errors']:
  name=metric['name'];a=(folder/('bridge_'+name+'.bin')).read_bytes();b=(reference/('reference_'+name+'.bin')).read_bytes();size=8 if name=='logits' else 4 if name=='risk' else args.length*256*4
  need(len(a)==len(b)==size,'Actual boundary extent differs')
  av=struct.unpack('<'+'f'*(size//4),a);bv=struct.unpack('<'+'f'*(size//4),b)
  need(all(math.isfinite(v) for v in (*av,*bv)),'Actual/reference boundary nonfinite')
  maximum=total=squared=0.
  for x,y in zip(av,bv):
   error=abs(float(x)-float(y));maximum=max(maximum,error);total+=error;squared+=error*error
  need(metric['values']==len(av) and [metric[k] for k in ('maximum_absolute_error','mean_absolute_error','rms_error')]==[maximum,total/len(av),math.sqrt(squared/len(av))],'Saved boundary metrics differ from canonical ascending binary64 reduction')
 for name in ('diagnostic_pooler_pre_tanh','diagnostic_pooler_post_tanh'):
  value=(folder/(name+'.bin')).read_bytes();need(len(value)==1024 and all(math.isfinite(v) for v in struct.unpack('<256f',value)),'Own actual TANH capture missing/nonfinite')
 return dict(key=item['key'],case_index=item['case_index'],window_index=item['window_index'],document_id=item['document_id'],input_folder=item['folder'],folder=folder.relative_to(ROOT).as_posix(),returncode=0,elapsed_seconds=process['elapsed_seconds'],summary=summary,operator_bounds=bounds,a8=checks,
  process_record=process,logits_equal=(folder/'bridge_logits.bin').read_bytes()==(reference/'reference_logits.bin').read_bytes(),risk_equal=(folder/'bridge_risk.bin').read_bytes()==(reference/'reference_risk.bin').read_bytes(),acceptance=None)

def shape_discrepancy(entry):
 return any(v['differences'] for v in entry['a8']) or not entry['logits_equal'] or not entry['risk_equal']

def shape_documents(plan,windows):
 bykey={v['key']:v for v in windows};documents=[]
 for doc in plan['documents']:
  if not all(key in bykey for key in doc['windows']):continue
  values=[bykey[k]['summary'] for k in doc['windows']];threshold=values[0]['threshold'];need(all(v['threshold']==threshold for v in values),'Document threshold differs')
  actual=max(v['risk'] for v in values);reference=max(v['reference_risk'] for v in values)
  need((actual>threshold)==(reference>threshold),'Complete document max strict decision differs')
  documents.append(dict(document_id=doc['document_id'],windows=doc['windows'],c_max_risk=actual,reference_max_risk=reference,c_winners=[k for k in doc['windows'] if bykey[k]['summary']['risk']==actual],reference_winners=[k for k in doc['windows'] if bykey[k]['summary']['reference_risk']==reference],threshold=threshold,c_signed_margin=actual-threshold,reference_signed_margin=reference-threshold,strict_reject=actual>threshold,reference_strict_reject=reference>threshold,strict_decision_agrees=True,aggregation='Independent full real-window maximum per document; strict >',acceptance=None))
 return documents

def shape_compilation_completed(executable,folder,log):
 # A fresh file alone could be an interrupted link. Successful actual C
 # execution (at least one complete checker record) establishes a loaded build.
 if not executable.is_file():return False
 if (folder/'bridge_summary.json').is_file():return True
 records=folder/'operator_bounds.jsonl'
 if records.is_file():
  lines=records.read_text().splitlines()
  if lines:
   try:
    first=json.loads(lines[0])
    if first.get('call_index')==1:return True
   except (ValueError,AttributeError):pass
 return log.is_file() and 'CSim done with 0 errors.' in log.read_text(errors='replace')

def c_stage_locked(args,plan,cpu,out,build,template,guard):
 from run_qualification_cases import run as bounded_command,validate_reviewed_resolution
 statepath=out/'c.json';state=read(statepath) if statepath.exists() else dict(identity=cpu['identity'],plan=plan,windows=[],complete=False,numerical_acceptance=None)
 need(state['identity']==cpu['identity'] and state['plan']==plan,'Saved C identity/plan differs')
 need('numerical_acceptance' in state and state['numerical_acceptance'] is None,'Diagnostic shape C numerical acceptance must remain explicitly null')
 expected={v['key']:v for v in cpu['b1']};need(len({v['key'] for v in state['windows']})==len(state['windows']),'Duplicate committed C window')
 compiledpath=build/'compiled.json';executable=build/'project/solution/csim/build/csim.exe'
 if state['windows']:
  compiled=read(compiledpath);need(compiled['passed'] is True and executable.is_file() and compiled['template']==template and compiled['synthesis_recorded_utc']==template['synthesis_recorded_utc'] and compiled['executable_stat']==dict(size=executable.stat().st_size,mtime_ns=executable.stat().st_mtime_ns),'Saved actual windows require unchanged successful compilation')
 unresolved=[]
 for entry in state['windows']:
  key=entry['key'];need(key in expected and (ROOT/entry['folder']).resolve()==(out/'c'/key).resolve(),'Committed C window association differs')
  current=shape_case(args,expected[key],ROOT/entry['folder'],template)
  need({k:v for k,v in entry.items() if k!='reviewed_resolution'}==current,'Committed actual C record/bytes differ')
  if shape_discrepancy(entry):
   if entry.get('reviewed_resolution'):
    need(validate_reviewed_resolution(entry['reviewed_resolution']['path'],entry,cpu['identity'],args.mode)==entry['reviewed_resolution'],'Stored explicit review changed')
   else:unresolved.append(entry)
 if args.reviewed_resolution:
  need(args.stage=='c-run' and len(unresolved)==1 and state.get('stop_kind')=='numerical_discrepancy' and state.get('active_window')==unresolved[0]['key'],'Review only the one saved active numerical stop')
  entry=unresolved[0];review=validate_reviewed_resolution(args.reviewed_resolution,entry,cpu['identity'],args.mode)
  state.setdefault('stopped_history',[]).append(dict(stopped=state['stopped'],stop_kind=state['stop_kind'],active_window=state['active_window'],reviewed_resolution=review))
  entry['reviewed_resolution']=review;state.pop('stopped',None);state.pop('stop_kind',None);state['active_window']=None;unresolved=[];save(statepath,state)
 derived=shape_documents(plan,state['windows']);need(not state.get('documents') or state['documents']==derived,'Saved document maxima changed')
 if args.stage=='c-validate':
  need(not args.reviewed_resolution,'Read-only validation cannot attach a review')
  print(json.dumps(dict(read_only=True,committed_windows=len(state['windows']),complete=len(state['windows'])==len(plan['windows']),complete_documents=derived,blocked=bool(unresolved or state.get('stopped') or state.get('active_window')),unresolved_windows=[v['key'] for v in unresolved],saved_stop=state.get('stopped'),numerical_acceptance=None)));return
 need(not unresolved,'Saved A8/terminal discrepancy requires explicit exact root casewise resolution')
 need(not state.get('stopped') and not state.get('active_window'),'Saved process/operator/noncompletion failure remains blocked; preserve diagnostics')
 def dispatch(item,compile_first=False):
  folder=out/'c'/item['key'];need(not folder.exists(),'Preserve uncommitted actual C attempt');folder.mkdir(parents=True)
  env=os.environ.copy();env.update(GATE_NUMERICAL_BRIDGE_DIR=linux(build/'reference_inputs'/item['key']),GATE_NUMERICAL_BRIDGE_PARAMETER_DIR=linux(build/'parameters'),GATE_NUMERICAL_BRIDGE_CASE=item['document_id'],GATE_NUMERICAL_BRIDGE_DIAGNOSTIC_DIR=linux(folder))
  if args.mode=='floating':env['GATE_NUMERICAL_BRIDGE_FLOATING']='1'
  else:env.pop('GATE_NUMERICAL_BRIDGE_FLOATING',None)
  command=[read(ROOT/'configs/project.json')['hardware']['hls']['tool_executable'],'--mode','hls','--tcl',linux(build/'shape_csim.tcl')] if compile_first else [str(executable)]
  timeout=C_COMPILE_TIMEOUT if compile_first else C_WINDOW_TIMEOUT;log=build/'compile.log' if compile_first else folder/'csim.log'
  process=dict(state='running',command=command,identity=cpu['identity'],length=args.length,mode=args.mode,input_folder=item['folder'],frozen_input_folder=(build/'reference_inputs'/item['key']).relative_to(ROOT).as_posix(),parameter_folder=(build/'parameters').relative_to(ROOT).as_posix(),build=build.relative_to(ROOT).as_posix(),window=item,timeout_seconds=timeout,memory_guard=guard.path.relative_to(ROOT).as_posix(),guard_start_line=len(guard.path.read_text().splitlines()) if guard.path.exists() else 0,cpu_affinity=sorted(os.sched_getaffinity(0)),resource_partition=guard.partition,recorded_utc=datetime.now(timezone.utc).isoformat(),acceptance=None)
  # Retain actual dispatch source separately; build generation stays immutable.
  write(folder/'dispatch_shape_runner.py',Path(__file__).read_bytes());write(folder/'dispatch_connected_runner.py',(ROOT/'scripts/run_qualification_cases.py').read_bytes())
  save(folder/'csim_run.json',process);state.update(active_window=item['key'],last_dispatch=process);save(statepath,state)
  compiled=None
  if compile_first:
   compiled=dict(state='running',passed=False,command=command,length=args.length,mode=args.mode,first_window=item['key'],timeout_seconds=timeout,synthesis_recorded_utc=template['synthesis_recorded_utc'],template=template,new_synthesis=False,acceptance=None);save(compiledpath,compiled)
  try:
   rc,elapsed=bounded_command(command,build,env,log,timeout,memory_guard=guard)
   process.update(state='returned',returncode=rc,elapsed_seconds=elapsed)
   observations=[json.loads(line) for line in guard.path.read_text().splitlines()][process['guard_start_line']:];children=[v for v in observations if v['stage']=='child_started']
   need(len(children)==1,'Assigned actual child group record missing');process.update(guard_observations=observations,child_pid=children[0]['child_pid'],child_group=children[0]['child_group']);save(folder/'csim_run.json',process)
   if compiled is not None:
    compiled.update(state='returned',returncode=rc,elapsed_seconds=elapsed,executable_stat=dict(size=executable.stat().st_size,mtime_ns=executable.stat().st_mtime_ns) if executable.is_file() else None,passed=shape_compilation_completed(executable,folder,log),compilation_evidence='Fresh executable plus actual C checker/summary or successful CSim completion; process/numerical outcome is separate');save(compiledpath,compiled)
   need(rc==0 and executable.is_file(),'Actual C process failed; retain uncommitted diagnostics')
   entry=shape_case(args,item,folder,template)
  except BaseException as error:
   if process['state']=='running':
    observations=[json.loads(line) for line in guard.path.read_text().splitlines()][process['guard_start_line']:] if guard.path.exists() else []
    process.update(state='stopped',error=str(error),guard_observations=observations);save(folder/'csim_run.json',process)
   if compiled is not None and compiled['state']=='running':
    compiled.update(state='stopped',error=str(error),passed=shape_compilation_completed(executable,folder,log),executable_stat=dict(size=executable.stat().st_size,mtime_ns=executable.stat().st_mtime_ns) if executable.is_file() else None,compilation_evidence='Actual loaded-C evidence required; interrupted link alone is unresolved; process/numerical outcome remains separate');save(compiledpath,compiled)
   state.update(stopped=str(error),stop_kind='process_or_operator_failure');save(statepath,state);raise
  state['windows'].append(entry);state['documents']=shape_documents(plan,state['windows'])
  if shape_discrepancy(entry):
   state.update(stopped='First actual A8/terminal propagation discrepancy requires exact root causal review',stop_kind='numerical_discrepancy')
   save(statepath,state);raise RuntimeError(state['stopped'])
  state['active_window']=None;state['documents']=shape_documents(plan,state['windows']);save(statepath,state)
 if args.stage=='c-compile':
  need(not compiledpath.exists() and not executable.exists() and not state['windows'],'Preserve existing C compilation/executable');dispatch(cpu['b1'][0],True)
 else:
  need(read(compiledpath)['passed'] is True and executable.is_file(),'Successful fresh checked C compilation required')
  done={v['key'] for v in state['windows']};count=0
  for item in cpu['b1']:
   if item['key'] in done:continue
   if count>=args.max_forwards:break
   paused();dispatch(item);count+=1
 state['complete']=len(state['windows'])==len(plan['windows']);state['documents']=shape_documents(plan,state['windows']);need(not state['complete'] or len(state['documents'])==2,'Both complete document maxima required')
 save(statepath,state);print(json.dumps(dict(stage=args.stage,completed_windows=len(state['windows']),complete=state['complete'],documents=state['documents'],numerical_acceptance=None)))

def c_stage(args):
 """Guarded actual C dispatch, or explicitly read-only saved-case validation."""
 need(os.name!='nt','Use WSL Python for C stages')
 if args.stage!='c-validate':paused()
 plan=load_plan(args.length);out=mode_dir(args.length,args.mode);cpu=read(out/'cpu.json')
 build=ROOT/'build'/f'detector_shape_bringup_l{args.length}_{args.mode}_v1';template=validate_shape_build(args,plan,cpu,out,build)
 from run_qualification_cases import ConnectedCMemoryGuard
 if args.stage=='c-validate':return c_stage_locked(args,plan,cpu,out,build,template,None)
 with ConnectedCMemoryGuard(read(ROOT/'configs/project.json'),out/'c',f'shape_l{args.length}_{args.mode}',active=args.max_forwards>0) as guard:
  return c_stage_locked(args,plan,cpu,out,build,template,guard)

def preview(args):
 path=plan_path(args.length);exists=path.exists();plan=read(path) if exists else None
 native=r'C:\research\GATE\.venv\Scripts\python.exe';script=linux(Path(__file__));windows_script='\\\\wsl.localhost\\Ubuntu'+script.replace('/','\\')
 commands={stage:[native,'-B',windows_script,'--length',str(args.length),'--mode',args.mode,'--stage',stage,'--execute','--max-forwards','1'] for stage in ('prepare','b1','batch')}
 print(json.dumps(dict(execute=False,length=args.length,mode=args.mode,plan=str(path),plan_prepared=exists,selected_case_indices=[1,18],real_windows=None if plan is None else len(plan['windows']),documents=None if plan is None else plan['documents'],batch_rows=None if plan is None else plan['batches'],commands=commands,
  c_stage_commands={stage:['wsl.exe','-d','Ubuntu','--exec','taskset','-c','7','/usr/bin/python3','-B',script,'--length',str(args.length),'--mode',args.mode,'--stage',stage,'--execute','--max-forwards','1'] for stage in ('c-template','c-compile','c-run')},
  read_only_saved_case_command=['wsl.exe','-d','Ubuntu','--exec','/usr/bin/python3','-B',script,'--length',str(args.length),'--mode',args.mode,'--stage','c-validate','--max-forwards','0'],assigned_cpu_choices=[6,7],window_timeout_seconds=C_WINDOW_TIMEOUT,compile_timeout_seconds=C_COMPILE_TIMEOUT,c_blocker='Root resource release; taskset CPU6 or7; exact casewise review on any new discrepancy; 1800s window / 3600s compile caps; no prospective acceptance',score_acceptance=None),indent=2))
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--length',type=int,choices=(128,512),required=True);ap.add_argument('--mode',choices=('w8','floating'),default='w8');ap.add_argument('--stage',choices=('preview','prepare','b1','batch','c-template','c-compile','c-run','c-validate'),default='preview');ap.add_argument('--execute',action='store_true');ap.add_argument('--max-forwards',type=int,default=1);ap.add_argument('--reviewed-resolution',metavar='PATH');args=ap.parse_args()
 need(args.max_forwards>=0 and (args.max_forwards>0 or args.stage=='c-validate' or (args.stage=='c-run' and args.reviewed_resolution)),'Positive bounded count, or zero for read-only validation/explicit review, required')
 need(not args.reviewed_resolution or args.stage=='c-run','Explicit root review only attaches through c-run')
 if args.stage=='c-validate':c_stage(args)
 elif not args.execute or args.stage=='preview':preview(args)
 elif args.stage=='prepare':prepare(args)
 elif args.stage in ('b1','batch'):cpu_stage(args)
 elif args.stage=='c-template':c_template(args)
 elif args.stage in ('c-compile','c-run'):c_stage(args)
if __name__=='__main__':main()