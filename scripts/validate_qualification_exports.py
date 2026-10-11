"""Read-only qualification associations; no inference, manifests or acceptance."""
from pathlib import Path
import argparse,ast,json,math,re,struct
ROOT=Path(__file__).resolve().parents[1]
LINEARS={'query':(256,256),'key':(256,256),'value':(256,256),'attention_output':(256,256),'ffn_input':(256,1024),'ffn_output':(1024,256)}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def common_parameters(floating=False):
 """Exact common filenames and byte extents consumed by the C harness."""
 sizes={'embedding_ln.bin':2048,'embedding_position.bin':256*256*4,'pooler_weight.bin':256*256*4,'pooler_bias.bin':1024,'classifier_weight.bin':2048,'classifier_bias.bin':8,'threshold.bin':4}
 for i in range(4):
  for name in ('attention_ln','output_ln'):sizes[f'layer{i}_{name}.bin']=2048
  for name,(k,n) in LINEARS.items():
   prefix=f'layer{i}_{name}'
   if floating:sizes[prefix+'_float_weight.bin']=k*n*4;sizes[prefix+'_float_bias.bin']=n*4
   else:sizes[prefix+'_w8.bin']=k*n;sizes[prefix+'_params.bin']=(1+2*n)*4
 return sizes

def binary(path,size):
 value=path.read_bytes()
 if len(value)!=size:raise ValueError('Binary extent differs: '+str(path))
 return value

def terminal(folder,prefix,logits,risk):
 if len(logits)!=2 or not all(math.isfinite(v) for v in [*logits,risk]):raise ValueError('Malformed terminal values')
 if binary(folder/(prefix+'_logits.bin'),8)!=struct.pack('<2f',*logits) or binary(folder/(prefix+'_risk.bin'),4)!=struct.pack('<f',risk):raise ValueError('Recorded terminal values differ from binary output')


def validate_reference_inputs(data_dir,*,root=ROOT):
 """Read frozen reference/source/parameter/input associations without Torch."""
 root=Path(root).resolve();data_dir=Path(data_dir);data_dir=(data_dir if data_dir.is_absolute() else root/data_dir).resolve();exports=read(data_dir/'export.json')
 floating=exports.get('floating',False)
 if (data_dir/'gate_fpga_reference.py').read_bytes()!=(root/'gate_fpga_reference.py').read_bytes():raise ValueError('Frozen arithmetic reference source changed')
 captured=(data_dir/'export_qualification_cases.py').read_bytes();current=(root/'scripts/export_qualification_cases.py').read_bytes()
 exporter_current=captured==current
 if floating and not exporter_current:raise ValueError('Floating exporter source changed')
 if not exporter_current:
  # The W8 export predates added membership/resume checks. Preserve its actual
  # snapshot; require all model/parameter-export calls and input batch creation
  # to match the current exporter, rather than relabel it as the newer source.
  def calls(raw):
   tree=ast.parse(raw.decode('utf-8-sig'));main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
   return [ast.dump(n) for n in ast.walk(main) if (isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('load_model','model','write')) or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='batch' for t in n.targets))]
  if calls(captured)!=calls(current):raise ValueError('Historical W8 model/export arithmetic calls changed')
 parameters=data_dir/'parameters';threshold=struct.pack('<f',exports['selection']['sampling']['historical_operating_threshold'])
 if binary(parameters/'threshold.bin',4)!=threshold:raise ValueError('Frozen historical threshold bytes differ')
 floating=exports.get('floating',False);verified=[]
 if floating:
  checkpoint=root/str(exports['identity']['checkpoint']).replace('\\','/')/'model.safetensors'
  with checkpoint.open('rb') as ck:
   header_size=struct.unpack('<Q',ck.read(8))[0];header=json.loads(ck.read(header_size));base=8+header_size
   def tensor(name):
    item=header[name]
    if item['dtype']!='F32':raise ValueError('Floating checkpoint parameter format differs')
    start,end=item['data_offsets'];ck.seek(base+start);raw=ck.read(end-start)
    if len(raw)!=end-start:raise ValueError('Truncated checkpoint parameter')
    return raw
   def equal(name,value):
    if (parameters/(name+'.bin')).read_bytes()!=value:raise ValueError('Floating checkpoint/common parameter differs: '+name)
    verified.append(name)
   suffix={'query':'attention.self.query','key':'attention.self.key','value':'attention.self.value','attention_output':'attention.output.dense','ffn_input':'intermediate.dense','ffn_output':'output.dense'}
   for i in range(4):
    for name,path in suffix.items():
     equal(f'layer{i}_{name}_float_weight',tensor(f'bert.encoder.layer.{i}.{path}.weight'));equal(f'layer{i}_{name}_float_bias',tensor(f'bert.encoder.layer.{i}.{path}.bias'))
    for name,path in [('attention_ln','attention.output.LayerNorm'),('output_ln','output.LayerNorm')]:equal(f'layer{i}_{name}',tensor(f'bert.encoder.layer.{i}.{path}.weight')+tensor(f'bert.encoder.layer.{i}.{path}.bias'))
   equal('embedding_ln',tensor('bert.embeddings.LayerNorm.weight')+tensor('bert.embeddings.LayerNorm.bias'));equal('embedding_position',tensor('bert.embeddings.position_embeddings.weight')[:256*256*4])
   for name,path in [('pooler','bert.pooler.dense'),('classifier','classifier')]:equal(name+'_weight',tensor(path+'.weight'));equal(name+'_bias',tensor(path+'.bias'))
   word=tensor('bert.embeddings.word_embeddings.weight');types=tensor('bert.embeddings.token_type_embeddings.weight')
   for item in exports['completed_windows']:
    folder=root/item['folder']
    for name,table,ids in [('embedding_word',word,item['inputs']['input_ids']),('embedding_type',types,item['inputs']['token_type_ids'])]:
     if (folder/(name+'.bin')).read_bytes()!=b''.join(table[i*1024:(i+1)*1024] for i in ids):raise ValueError('Floating frozen embedding gather differs: '+name)
 for item in exports['completed_windows']:
  mask=b''.join(struct.pack('<f',-0.0 if v else -3.4028234663852886e38) for v in item['inputs']['attention_mask'])
  if (root/item['folder']/'key_mask.bin').read_bytes()!=mask:raise ValueError('Frozen key mask differs')
 return dict(floating=floating,windows=len(exports['completed_windows']),floating_parameter_files=len(verified),reference_sources_checked=True,exporter_source_current=exporter_current,historical_exporter_arithmetic_calls_checked=not exporter_current)

def validate_exports(data_dir,*,root=ROOT,compiled_dir=None,qualification=None,bounds_dir=None,require_complete_bounds=True):
 """Check committed exports and optional saved C work against actual files.

 compiled_dir enables successful compiled/synthesized/current source checks.
 qualification accepts a loaded state or path, otherwise an existing saved state
 is checked. Saved discrepancies reject resume without granting score acceptance.
 """
 if bounds_dir is not None:
  if qualification is not None:raise ValueError('Choose explicit bounds evidence or historical v1 state')
  return validate_bounds_qualification(data_dir,bounds_dir=bounds_dir,compiled_dir=compiled_dir,root=root,require_complete=require_complete_bounds)
 root,data_dir=Path(root).resolve(),Path(data_dir).resolve();exports=read(data_dir/'export.json')
 selection=read(root/'results/goal4/numerical_bridge/development_cases_v1/selection.json')
 if exports['selection']!=selection:raise ValueError('Export frozen selection differs')
 cases=selection['cases']
 if len(cases)!=32 or selection['case_count']!=32 or any(len({c[k] for c in cases})!=32 for k in ('case_index','document_id','source_group_id')):raise ValueError('Frozen case membership differs')
 assigned={}
 for case in cases:
  if case['window_count']!=len(case['windows']):raise ValueError('Frozen window count differs')
  for wi,window in enumerate(case['windows']):
   if set(window)!={'input_ids','attention_mask','token_type_ids'} or any(len(v)!=256 for v in window.values()):raise ValueError('Frozen input shape differs')
   assigned[case['case_index'],wi]=(case,window)
 if len(assigned)!=66:raise ValueError('Frozen window membership differs')
 floating=exports.get('floating',False)
 if type(floating) is not bool:raise ValueError('Malformed reference mode')
 for name,size in common_parameters(floating).items():
  actual=binary(data_dir/'parameters'/name,size)
  if not floating and actual!=(root/'results/goal4/numerical_bridge'/name).read_bytes():raise ValueError('Original frozen common parameter differs: '+name)
 # C reads window-local names before the common directory. Forbid all predefined
 # common names locally, including the other mode, to prevent silent shadowing.
 names=set(common_parameters())|set(common_parameters(True));exported={}
 for item in exports['completed_windows']:
  key=item['case_index'],item['window_index']
  if key in exported or key not in assigned:raise ValueError('Duplicate/unassigned exported window')
  case,window=assigned[key]
  if item['inputs']!=window or any(item[k]!=case[k] for k in ('document_id','source_group_id','label','partition')):raise ValueError('Export membership differs')
  folder=(root/item['folder']).resolve();expected=data_dir/f'case{key[0]:02d}'/f'window{key[1]:03d}'
  if folder!=expected or not folder.is_relative_to(data_dir):raise ValueError('Export path differs from assigned window')
  if read(folder/'window.json')!=item:raise ValueError('Exported window record differs')
  if any((folder/name).exists() for name in names):raise ValueError('Window-local file shadows a common parameter')
  terminal(folder,'reference',item['logits'],item['risk']);exported[key]=(item,folder)
 if exports['complete'] and set(exported)!=set(assigned):raise ValueError('Complete export lacks assigned windows')
 if compiled_dir is not None:
  build=Path(compiled_dir).resolve();source=root/'build/fixed_fp32_hls'
  synthesis,compiled=read(source/'run.json'),read(build/'compiled.json')
  if not synthesis['synthesis_passed'] or not compiled['passed'] or compiled['synthesis_recorded_utc']!=synthesis['recorded_utc'] or not (build/'project/solution/csim/build/csim.exe').is_file():raise ValueError('Successful compiled/synthesized source association missing')
  for name in ('fixed_fp32_service.hpp','numerical_bridge_tb.cpp'):
   if (build/name).read_bytes()!=(root/'hardware'/name).read_bytes():raise ValueError('Compiled current source differs: '+name)
  generator=ast.parse((root/'scripts/build_fixed_fp32_hls.py').read_text(encoding='utf-8-sig'))
  tops=[call.args[0].value for call in ast.walk(generator) if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name) and call.func.value.id=='top' and call.func.attr=='write_text' and call.args and isinstance(call.args[0],ast.Constant)]
  if len(tops)!=1 or (source/'gate_fixed_fp32_top.cpp').read_bytes()!=tops[0].encode('utf-8'):raise ValueError('Current generated FP top differs from actual synthesis source')
  if (source/'fixed_fp32_service.hpp').read_bytes()!=(root/'hardware/fixed_fp32_service.hpp').read_bytes() or (build/'gate_fixed_fp32_top.cpp').read_bytes()!=(source/'gate_fixed_fp32_top.cpp').read_bytes():raise ValueError('Generated FP top/current synthesis source differs')
 statepath=data_dir/'qualification.json'
 state=(read(statepath) if statepath.exists() else None) if qualification is None else qualification
 if isinstance(state,(str,Path)):state=read(Path(state))
 saved=set()
 if state is not None:
  if state['identity']!=exports['identity'] or state['selection']!=selection:raise ValueError('Saved qualification identity/selection differs')
  for output in state['windows']:
   key=output['case_index'],output['window_index']
   if key in saved or key not in exported:raise ValueError('Duplicate/unexported saved C window')
   item,folder=exported[key]
   if any(output[k]!=item[k] for k in ('document_id','label','folder')):raise ValueError('Saved C window membership differs')
   summary=read(folder/'bridge_summary.json')
   if output['summary']!=summary or not summary['completed'] or summary['case']!=item['document_id']:raise ValueError('Saved C summary differs')
   mode='ordinary_floating_reference' if floating else 'w8_a8_integer_contract'
   if summary['encoder_arithmetic']!=mode or summary['service_calls_by_op']!=([165,0,0,2,16,18,33,32,1] if floating else [165,72,72,2,16,18,33,32,1]):raise ValueError('Saved C arithmetic mode differs')
   layers=summary['integer_layers'];expected_names=[f'layer{i}_{name}' for i in range(4) for name in LINEARS] if not floating else []
   if [v['name'] for v in layers]!=expected_names or len(output['a8'])!=len(layers):raise ValueError('Saved integer layer membership differs')
   for layer,check in zip(layers,output['a8']):
    name=layer['name'];size=256*LINEARS[name.split('_',1)[1]][0]
    actual=binary(folder/('actual_'+name+'_a8.bin'),size);reference=binary(folder/(name+'_reference_a8.bin'),size)
    differences=sum(a!=b for a,b in zip(actual,reference))
    if check!=dict(name=name,values=size,differences=differences) or layer['input_codes']!=size or layer['original_software_code_differences']!=differences:raise ValueError('Saved A8 checks differ from actual bytes')
   terminal(folder,'bridge',summary['logits'],summary['risk']);terminal(folder,'reference',summary['reference_logits'],summary['reference_risk'])
   logits_equal=(folder/'bridge_logits.bin').read_bytes()==(folder/'reference_logits.bin').read_bytes();risk_equal=(folder/'bridge_risk.bin').read_bytes()==(folder/'reference_risk.bin').read_bytes()
   if output['logits_equal']!=logits_equal or output['risk_equal']!=risk_equal:raise ValueError('Saved terminal checks differ from actual bytes')
   if output['returncode'] or any(v['differences'] for v in output['a8']) or not logits_equal or not risk_equal:raise RuntimeError('Saved unexplained discrepancy requires review before resume')
   saved.add(key)
  if state.get('state')=='stopped_first_unexplained_discrepancy':raise RuntimeError('Saved stopped discrepancy requires review')
  if state['complete'] and saved!=set(assigned):raise ValueError('Complete qualification lacks assigned windows')
 return dict(cases=len(cases),assigned_windows=len(assigned),exported_windows=len(exported),saved_c_windows=len(saved),floating=floating,compiled_sources_checked=compiled_dir is not None)


def _same_path(value,expected,root):
 candidate=Path(str(value).replace('\\','/'))
 if not candidate.is_absolute():candidate=root/candidate
 def portable(path):return re.sub(r'^//wsl[.]localhost/[^/]+','',Path(path).as_posix()).rstrip('/')
 return portable(candidate)==portable(expected)


def validate_bounds_qualification(data_dir,*,bounds_dir=None,compiled_dir=None,root=ROOT,require_complete=True):
 """Explicit checked-C evidence route; never grants an arithmetic/score approval.

 Default requires all frozen66 windows and32 exact reported document maxima.
 Partial mode only audits already committed evidence and reports ineligibility.
 Historical validate_exports semantics remain unchanged without bounds_dir.
 """
 root=Path(root).resolve();data_dir=Path(data_dir);data_dir=(data_dir if data_dir.is_absolute() else root/data_dir).resolve()
 bounds=(Path(bounds_dir) if Path(bounds_dir).is_absolute() else root/Path(bounds_dir)).resolve() if bounds_dir is not None else data_dir/'bounds_v2'
 if bounds!=data_dir/'bounds_v2':raise ValueError('Only the assigned fresh bounds_v2 tree is permitted')
 build=(Path(compiled_dir) if Path(compiled_dir).is_absolute() else root/Path(compiled_dir)).resolve() if compiled_dir is not None else root/'build/development_qualification_bounds_v2'
 exports=read(data_dir/'export.json');state=read(bounds/'qualification.json');selection=exports['selection']
 assigned={(c['case_index'],wi):(c,window) for c in selection['cases'] for wi,window in enumerate(c['windows'])}
 if require_complete and (state.get('complete') is not True or len(state.get('windows',[]))!=66):raise RuntimeError('Checked C qualification incomplete: '+str(len(state.get('windows',[])))+'/66 committed windows; numerical acceptance remains null')
 if not exports['complete'] or len(exports['completed_windows'])!=66:raise ValueError('All frozen B1 exports required')
 base=validate_exports(data_dir,root=root,qualification=dict(identity=exports['identity'],selection=selection,windows=[],complete=False))
 reference=validate_reference_inputs(data_dir,root=root)
 import run_qualification_cases as runner
 compiled=(runner.validate_captured_tail_build(build,read(root/'build/fixed_fp32_hls/run.json'),root=root) if build==root/runner.TAIL_BUILD else runner.validate_bounds_build(build,read(root/'build/fixed_fp32_hls/run.json')))
 if '-ffp-contract=off' not in compiled['flags'] or 'config_compile -unsafe_math_optimizations=false' not in (build/'qualify.tcl').read_text():raise ValueError('Actual C arithmetic compilation flags differ')
 revisions=re.findall(r'^#define GATE_FIXED_FP32_SCHEDULE_REVISION "([^"]+)"$',(build/'fixed_fp32_service.hpp').read_text(),re.M)
 if len(revisions)!=1:raise ValueError('Compiled service revision missing')
 if state['identity']!=exports['identity'] or state['selection']!=selection or state['bounds'] is not True or state['numerical_acceptance'] is not None or not _same_path(state['input_export'],data_dir/'export.json',root):raise ValueError('Bounds identity/selection/null-approval association differs')
 if state.get('state','').startswith('stopped_'):raise RuntimeError('Saved unresolved bounds stop cannot qualify')
 floating=exports['floating'];mode='floating' if floating else 'w8'
 items={(v['case_index'],v['window_index']):v for v in exports['completed_windows']}
 build_records={build:compiled};bykey={};reviews=0;recognized=0;recognized_w8=0;recognized_captured=0;calls=0;guarded=0;maximum_risk_error=0.0
 threshold=struct.unpack('<f',(data_dir/'parameters/threshold.bin').read_bytes())[0]
 for entry in state['windows']:
  key=entry['case_index'],entry['window_index']
  if key in bykey or key not in assigned or key not in items:raise ValueError('Duplicate/unassigned bounds window')
  item=items[key];case,window=assigned[key];folder=bounds/f'case{key[0]:02d}'/f'window{key[1]:03d}';input_folder=root/item['folder']
  assigned_build=entry.get('compiled_build','build/development_qualification_bounds_v2')
  if assigned_build not in ('build/development_qualification_bounds_v2',runner.TAIL_BUILD) or (floating and assigned_build==runner.TAIL_BUILD):raise ValueError('Unapproved actual per-entry compiled build')
  entry_build=root/assigned_build
  if entry_build not in build_records:
   build_records[entry_build]=(runner.validate_captured_tail_build(entry_build,read(root/'build/fixed_fp32_hls/run.json'),root=root) if assigned_build==runner.TAIL_BUILD else runner.validate_bounds_build(entry_build,read(root/'build/fixed_fp32_hls/run.json')))
  entry_compiled=build_records[entry_build]
  if not _same_path(entry['folder'],folder,root) or not _same_path(entry['input_folder'],input_folder,root) or any(entry[k]!=item[k] for k in ('document_id','label')) or entry['returncode']!=0 or entry['acceptance'] is not None:raise ValueError('Bounds actual case/status/null-acceptance differs')
  summary=read(folder/'bridge_summary.json')
  expected_calls=[165,0,0,2,16,18,33,32,1] if floating else [165,72,72,2,16,18,33,32,1]
  if summary!=entry['summary'] or summary['completed'] is not True or summary['case']!=item['document_id'] or summary['source_revision']!=revisions[0] or summary['encoder_arithmetic']!=('ordinary_floating_reference' if floating else 'w8_a8_integer_contract') or summary['service_calls_by_op']!=expected_calls:raise ValueError('Actual checked C summary/source/count/mode differs')
  actual_bounds=runner.validate_operator_bounds(folder,summary)
  if actual_bounds!=entry['operator_bounds'] or summary['same_input_quantization_exact']!=(None if floating else True) or summary['integer_accumulators_exact']!=(None if floating else True) or summary['same_input_codes_checked']!=actual_bounds['checked_values_by_op'][1] or summary['exact_accumulators_checked']!=actual_bounds['checked_values_by_op'][2]:raise ValueError('Actual original bounds/exact-arithmetic coverage differs')
  calls+=actual_bounds['calls']
  terminal(folder,'bridge',summary['logits'],summary['risk']);terminal(input_folder,'reference',summary['reference_logits'],summary['reference_risk'])
  logits_equal=(folder/'bridge_logits.bin').read_bytes()==(input_folder/'reference_logits.bin').read_bytes();risk_equal=(folder/'bridge_risk.bin').read_bytes()==(input_folder/'reference_risk.bin').read_bytes()
  if entry['logits_equal']!=logits_equal or entry['risk_equal']!=risk_equal:raise ValueError('Actual terminal equality report differs')
  risk,reference_risk=summary['risk'],summary['reference_risk']
  if not (0<=risk<=1 and 0<=reference_risk<=1) or summary['threshold']!=threshold or summary['strict_reject']!=(risk>threshold) or summary['reference_strict_reject']!=(reference_risk>threshold) or summary['strict_decision_agrees']!=((risk>threshold)==(reference_risk>threshold)) or summary['strict_decision_agrees'] is not True or summary['end_to_end_score_tolerance'] is not None:raise RuntimeError('Frozen strict decision/threshold/no-epsilon contract differs')
  error=abs(risk-reference_risk);maximum_risk_error=max(maximum_risk_error,error)
  if summary['risk_absolute_error']!=error:raise ValueError('Actual terminal error report differs')
  checks=[];names=[f'layer{i}_{name}' for i in range(4) for name in LINEARS] if not floating else []
  if [v['name'] for v in summary['integer_layers']]!=names:raise ValueError('Actual integer layer membership differs')
  for layer in summary['integer_layers']:
   name=layer['name'];k,n=LINEARS[name.split('_',1)[1]];codes=binary(folder/('actual_'+name+'_a8.bin'),256*k);ref=binary(input_folder/(name+'_reference_a8.bin'),256*k)
   differences=sum(a!=b for a,b in zip(codes,ref));checks.append(dict(name=name,values=len(codes),differences=differences))
   if layer['input_codes']!=256*k or layer['accumulators_checked']!=256*n or layer['original_software_code_differences']!=differences:raise ValueError('Actual A8/accumulator record differs')
  if entry['a8']!=checks:raise ValueError('Actual A8 discrepancy record differs')
  stages=['embedding_ln','layer0','layer1','layer2','layer3','logits','risk']
  if [v['name'] for v in summary['intermediate_errors']]!=stages:raise ValueError('Captured stage membership differs')
  for stage in summary['intermediate_errors']:
   name=stage['name'];size=2 if name=='logits' else 1 if name=='risk' else 65536
   actual=struct.unpack('<'+str(size)+'f',binary(folder/('bridge_'+name+'.bin'),size*4));reference_values=struct.unpack('<'+str(size)+'f',binary(input_folder/('reference_'+name+'.bin'),size*4))
   if stage['values']!=size or not all(math.isfinite(v) for v in (*actual,*reference_values)):raise ValueError('Captured stage extent/finiteness differs')
   if stage['maximum_absolute_error']!=max(abs(a-b) for a,b in zip(actual,reference_values)) or any(not math.isfinite(stage[k]) or stage[k]<0 for k in ('maximum_absolute_error','mean_absolute_error','rms_error')):raise ValueError('Captured stage error report differs')
  process=folder/'csim_run.json'
  if process.exists():
   observed=read(process)
   if observed['state']!='returned' or observed['returncode']!=0 or observed['identity']!=exports['identity'] or tuple(observed['window'])!=key or len(observed['command'])!=1 or not _same_path(observed['command'][0],entry_build/'project/solution/csim/build/csim.exe',root):raise RuntimeError('Actual C command/status/window association differs')
   if not (bounds/observed['source']).is_file():raise ValueError('Actual dispatch source snapshot missing')
   if entry.get('memory_guard_file'):
    if observed.get('memory_guard_file')!=entry['memory_guard_file'] or not (root/entry['memory_guard_file']).is_file():raise ValueError('Actual memory guard association missing')
    guarded+=1
  elif not _same_path(entry_compiled['compiled_first_window'],folder,root):raise ValueError('Actual C process/compile-first association missing')
  discrepancy=any(v['differences'] for v in checks) or not logits_equal or not risk_equal
  reviewed=entry.get('reviewed_resolution');known=entry.get('recognized_floating_propagation');known_w8=entry.get('recognized_w8_propagation');known_captured=entry.get('recognized_w8_captured_tail')
  if reviewed:
   if runner.validate_reviewed_resolution(reviewed['path'],entry,exports['identity'],mode,root=root)!=reviewed:raise ValueError('Original casewise resolution changed')
   reviews+=1
  if assigned_build==runner.TAIL_BUILD and (reviewed or not discrepancy):
   if not process.is_file() or observed.get('bounds') is not True or observed.get('compiled_build')!=runner.TAIL_BUILD:raise ValueError('Exact/reviewed actual capture process/build association missing')
   for captured in ('diagnostic_pooler_pre_tanh.bin','diagnostic_pooler_post_tanh.bin'):
    if not all(math.isfinite(v) for v in struct.unpack('<256f',binary(folder/captured,256*4))):raise ValueError('Exact/reviewed actual capture finite values differ')
  if known:
   if not floating or any(v['differences'] for v in checks):raise RuntimeError('Recognized floating policy cannot waive A8/mode')
   if runner.recognized_floating_entry(entry,exports['identity'],selection,data_dir/'parameters',root=root)!=known:raise ValueError('Recognized floating classification changed')
   recognized+=1
  if known_w8:
   if floating or any(v['differences'] for v in checks):raise RuntimeError('Recognized W8 policy cannot waive A8/mode')
   if runner.recognized_w8_entry(entry,exports['identity'],selection,data_dir/'parameters',root=root)!=known_w8:raise ValueError('Recognized W8 classification changed')
   recognized_w8+=1
  if known_captured:
   if floating or assigned_build!=runner.TAIL_BUILD or any(v['differences'] for v in checks):raise RuntimeError('Captured-tail policy cannot waive mode/build/A8')
   if runner.recognized_w8_captured_tail_entry(entry,exports['identity'],selection,data_dir/'parameters',root=root)!=known_captured:raise ValueError('Captured-tail classification/source changed')
   recognized_captured+=1
  if assigned_build==runner.TAIL_BUILD and discrepancy and not (known_captured or reviewed):raise RuntimeError('Actual capture-build entry lacks an explicit prospective or casewise record')
  if discrepancy and not (reviewed or known or known_w8 or known_captured):raise RuntimeError('Unreviewed actual numerical discrepancy remains')
  bykey[key]=entry
 complete=state['complete'] is True and set(bykey)==set(assigned) and len(bykey)==66
 if state['complete'] is True and not complete:raise ValueError('Claimed complete bounds set differs')
 documents=[]
 if complete:
  if state.get('active_window') is not None:raise ValueError('Complete state retains an unfinished window')
  for case in selection['cases']:
   windows=[bykey[case['case_index'],wi] for wi in range(case['window_count'])]
   actual=max(v['summary']['risk'] for v in windows);reference_max=max(v['summary']['reference_risk'] for v in windows)
   documents.append(dict(document_id=case['document_id'],label=case['label'],windows=len(windows),actual_max=actual,reference_max=reference_max,threshold=threshold,actual_reject=actual>threshold,reference_reject=reference_max>threshold,decision_agrees=(actual>threshold)==(reference_max>threshold)))
  if len(documents)!=32 or state['documents']!=documents or any(v['decision_agrees'] is not True for v in documents):raise RuntimeError('Exact document-max/strict-decision evidence differs')
 if require_complete and not complete:raise RuntimeError('Complete frozen66-window/32-document evidence required')
 return dict(cases=32,assigned_windows=66,exported_windows=base['exported_windows'],saved_c_windows=len(bykey),floating=floating,bounds_v2=True,complete=complete,documents_checked=len(documents),actual_operator_calls=calls,casewise_reviews=reviews,recognized_floating_windows=recognized,recognized_w8_windows=recognized_w8,recognized_w8_captured_tail_windows=recognized_captured,compiled_builds_checked=[p.relative_to(root).as_posix() for p in build_records],memory_guard_recorded_windows=guarded,maximum_window_risk_error=maximum_risk_error,reference_association=reference,compiled_sources_checked=True,numerical_acceptance=None,score_approval=None)

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--floating',action='store_true');ap.add_argument('--compiled',action='store_true');ap.add_argument('--bounds',action='store_true');ap.add_argument('--allow-partial',action='store_true');args=ap.parse_args()
 if args.allow_partial and not args.bounds:ap.error('--allow-partial requires --bounds')
 mode='floating' if args.floating else 'w8';data=ROOT/f'results/goal4/numerical_bridge/development_qualification_{mode}_v1'
 if args.bounds:result=validate_bounds_qualification(data,require_complete=not args.allow_partial)
 else:result=validate_exports(data,compiled_dir=ROOT/'build/development_qualification_w8_v1' if args.compiled else None)
 print(json.dumps(result))
