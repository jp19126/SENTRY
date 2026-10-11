"""Bounded connected FP32 C-service qualification of the frozen development windows.
Preview by default. Compile once; inspect actual A8/logits/risk before continuing.
No synthesis, board access, scores approval or replacement of failed actual outputs.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse,ast,importlib.util,json,math,os,shutil,signal,struct,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/goal4/numerical_bridge/development_qualification_w8_v1'
BUILD=ROOT/'build/development_qualification_w8_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,r):
 temporary=p.with_suffix(p.suffix+'.tmp');temporary.write_text(json.dumps(r,indent=2)+'\n');temporary.replace(p)
def paused():
 if any((ROOT/p).exists() for p in ['results/goal2/pause.request','results/goal3/pause.request']):raise RuntimeError('Pause marker present')
def run(cmd,cwd,env,log,timeout,memory_guard=None):
 paused()
 if memory_guard:memory_guard.observe('preflight')
 with log.open('w') as f:
  p=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);start=time.monotonic();next_memory=start+5
  if memory_guard:memory_guard.event(stage='child_started',child_pid=p.pid,child_group=p.pid,command=cmd)
  try:
   while p.poll() is None:
    paused()
    if memory_guard and time.monotonic()>=next_memory:
     memory_guard.observe('runtime',p.pid);next_memory=time.monotonic()+5
    if time.monotonic()-start>timeout:raise RuntimeError('Bounded command timeout')
    time.sleep(.5)
  except BaseException as error:
   if memory_guard:memory_guard.event(stage='assigned_child_stop',child_pid=p.pid,child_group=p.pid,error=str(error))
   if p.poll() is None:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise
 if memory_guard:memory_guard.observe('postflight',p.pid)
 return p.returncode,time.monotonic()-start

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true')
 ap.add_argument('--compile',action='store_true');ap.add_argument('--max-new-windows',type=int,default=1);ap.add_argument('--floating',action='store_true');ap.add_argument('--bounds',action='store_true');ap.add_argument('--reviewed-resolution',metavar='PATH');ap.add_argument('--recognized-floating-policy',metavar='PATH');ap.add_argument('--recognized-w8-policy',metavar='PATH');ap.add_argument('--recognized-w8-captured-tail-policy',metavar='PATH');args=ap.parse_args()
 if args.recognized_w8_captured_tail_policy and (not args.bounds or args.floating):ap.error('--recognized-w8-captured-tail-policy requires W8 --bounds')
 if args.recognized_w8_captured_tail_policy and (args.recognized_w8_policy or args.compile):ap.error('Captured-tail policy reuses its existing build; no --compile or original W8 policy')
 if args.recognized_w8_policy and (not args.bounds or args.floating):ap.error('--recognized-w8-policy requires --bounds without --floating')
 if args.recognized_floating_policy and not (args.bounds and args.floating):ap.error('--recognized-floating-policy requires --bounds --floating')
 if args.reviewed_resolution and not args.bounds:ap.error('--reviewed-resolution requires --bounds')
 if args.reviewed_resolution and args.compile:ap.error('Review an existing stopped run without --compile')
 if args.bounds and (args.max_new_windows<0 or (args.max_new_windows==0 and not args.reviewed_resolution)):ap.error('Bounds count must be positive, or zero with --reviewed-resolution')
 if args.bounds:return bounds_main(args)
 global DATA
 if args.floating:DATA=ROOT/'results/goal4/numerical_bridge/development_qualification_floating_v1'
 if not args.execute:
  print(json.dumps(dict(execute=False,compile=args.compile,max_new_windows=args.max_new_windows,
   input=str(DATA/'export.json'),timeout_seconds=1800,scope='Actual unchanged FP32 C service with exact integer contract; not full RTL')));return
 if os.name=='nt':raise RuntimeError('Run with WSL Python and installed Vitis')
 if args.max_new_windows<1:raise ValueError('Positive bounded window count required')
 paused();exports=read(DATA/'export.json')
 if exports['selection']!=read(ROOT/'results/goal4/numerical_bridge/development_cases_v1/selection.json'):raise ValueError('Frozen selection changed')
 assigned={(case['case_index'],wi):(case,window) for case in exports['selection']['cases'] for wi,window in enumerate(case['windows'])}
 keys=set()
 for item in exports['completed_windows']:
  key=(item['case_index'],item['window_index'])
  if key in keys or key not in assigned:raise ValueError('Duplicate/unassigned exported window')
  case,window=assigned[key]
  if item['inputs']!=window or any(item[k]!=case[k] for k in ('document_id','source_group_id','label','partition')):raise ValueError('Export membership differs')
  folder=ROOT/item['folder']
  if not folder.is_relative_to(DATA) or read(folder/'window.json')!=item:raise ValueError('Exported window record/path differs')
  keys.add(key)
 config=read(ROOT/'configs/project.json')
 source=ROOT/'build/fixed_fp32_hls';synthesis=read(source/'run.json')
 if not synthesis['synthesis_passed'] or (source/'fixed_fp32_service.hpp').read_bytes()!=(ROOT/'hardware/fixed_fp32_service.hpp').read_bytes():raise ValueError('Current successful FP32 source association differs')
 if exports.get('floating',False)!=args.floating:raise ValueError('Requested reference mode differs')
 if not exports['completed_windows']:raise ValueError('At least one real frozen export required')
 executable=BUILD/'project/solution/csim/build/csim.exe';compiledpath=BUILD/'compiled.json'
 def env_for(item):
  env=os.environ.copy();env.update(GATE_NUMERICAL_BRIDGE_DIR=str(ROOT/item['folder']),
   GATE_NUMERICAL_BRIDGE_PARAMETER_DIR=str(DATA/'parameters'),GATE_NUMERICAL_BRIDGE_CASE=item['document_id'])
  env.pop('GATE_NUMERICAL_BRIDGE_DIAGNOSTIC_DIR',None)
  if args.floating:env['GATE_NUMERICAL_BRIDGE_FLOATING']='1'
  else:env.pop('GATE_NUMERICAL_BRIDGE_FLOATING',None)
  return env
 first=exports['completed_windows'][0]
 if args.compile:
  if compiledpath.exists():raise ValueError('Compiled record exists; reuse instead of overwriting')
  BUILD.mkdir(parents=True,exist_ok=True)
  for name in ['fixed_fp32_service.hpp','numerical_bridge_tb.cpp']:shutil.copyfile(ROOT/'hardware'/name,BUILD/name)
  shutil.copyfile(source/'gate_fixed_fp32_top.cpp',BUILD/'gate_fixed_fp32_top.cpp')
  q=lambda p:'{'+str(p)+'}'
  flags='-std=c++14 -O2 -ffp-contract=off -I'+str(BUILD)
  tcl=BUILD/'qualify.tcl';tcl.write_text('\n'.join([
   'open_project '+q(BUILD/'project'),'set_top gate_fixed_fp32_top',
   'add_files '+q(BUILD/'gate_fixed_fp32_top.cpp')+' -cflags '+q(flags),
   'add_files -tb '+q(BUILD/'numerical_bridge_tb.cpp')+' -cflags '+q(flags),
   'open_solution solution -flow_target '+config['hardware']['hls']['flow_target'],
   'set_part '+q(config['hardware']['part']),'create_clock -period 5',
   'config_compile -unsafe_math_optimizations=false','csim_design','exit'])+'\n')
  command=[config['hardware']['hls']['tool_executable'],'--mode','hls','--tcl',str(tcl)]
  record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),state='running',passed=False,
   command=command,synthesis_recorded_utc=synthesis['recorded_utc'],flags=flags,
   compiled_first_window=first['folder'],new_synthesis=False,score_approval=None)
  save(compiledpath,record)
  try:
   rc,elapsed=run(command,BUILD,env_for(first),BUILD/'compile.log',1800)
   log=(BUILD/'compile.log').read_text(errors='replace')
   record.update(state='returned',returncode=rc,elapsed_seconds=elapsed,passed=rc==0 and executable.is_file() and 'CSim done with 0 errors.' in log)
  except Exception as e:record.update(state='stopped',error=str(e));save(compiledpath,record);raise
  save(compiledpath,record)
  if not record['passed']:raise RuntimeError('Actual C-service compilation/first window failed')
 else:
  record=read(compiledpath)
  if not record['passed'] or record['synthesis_recorded_utc']!=synthesis['recorded_utc']:raise ValueError('Compiled source identity differs')
  for name in ['fixed_fp32_service.hpp','numerical_bridge_tb.cpp']:
   if (BUILD/name).read_bytes()!=(ROOT/'hardware'/name).read_bytes():raise ValueError('Compiled testbench/source changed')
 from validate_qualification_exports import validate_exports
 validate_exports(DATA,compiled_dir=BUILD)
 dispatch=DATA/f'driver_dispatch_{len(list(DATA.glob("driver_dispatch_*.py")))+1:03d}.py';shutil.copyfile(Path(__file__),dispatch)
 statepath=DATA/'qualification.json'
 state=read(statepath) if statepath.exists() else dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
  identity=exports['identity'],selection=exports['selection'],windows=[],complete=False,numerical_acceptance=None,
  evidence_kind='C simulation actual FP32 service; direct/decomposed signed integer contract; not synthesized full detector')
 if state['identity']!=exports['identity'] or state['selection']!=exports['selection']:raise ValueError('Qualification identity differs')
 if any(v['returncode'] or any(c['differences'] for c in v['a8']) or not v['logits_equal'] or not v['risk_equal'] for v in state['windows']):raise RuntimeError('Saved unresolved numerical discrepancy cannot be skipped on resume')
 if len({(v['case_index'],v['window_index']) for v in state['windows']})!=len(state['windows']):raise ValueError('Duplicate saved C window')
 if state.get('state')=='stopped_first_unexplained_discrepancy':raise RuntimeError('Saved unresolved discrepancy requires cause review; do not skip it on resume')
 state['last_dispatch']=dict(source=dispatch.name,mode='floating' if args.floating else 'w8',max_new_windows=args.max_new_windows,cpu_affinity=sorted(os.sched_getaffinity(0)))
 save(statepath,state)
 completed={(v['case_index'],v['window_index']) for v in state['windows']};count=0
 for item in exports['completed_windows']:
  key=(item['case_index'],item['window_index'])
  if key in completed:continue
  if count>=args.max_new_windows:break
  paused();folder=ROOT/item['folder'];summarypath=folder/'bridge_summary.json'
  state['active_window']=list(key);save(statepath,state)
  if summarypath.exists():
   if item['folder']!=record['compiled_first_window']:raise ValueError('Uncommitted existing C run requires manual inspection')
   rc=0;elapsed=None
  else:
   observation=dict(state='running',command=[str(executable)],window=key,source=dispatch.name,identity=exports['identity'],cpu_affinity=sorted(os.sched_getaffinity(0)),board_access=False)
   save(folder/'csim_run.json',observation)
   try:
    rc,elapsed=run([str(executable)],BUILD,env_for(item),folder/'csim.log',1800)
    observation.update(state='returned',returncode=rc,elapsed_seconds=elapsed);save(folder/'csim_run.json',observation)
   except Exception as error:
    observation.update(state='stopped',error=str(error));save(folder/'csim_run.json',observation);raise
  summary=read(summarypath)
  if not summary['completed'] or summary['case']!=item['document_id'] or sum(summary['service_calls_by_op'])!=(267 if args.floating else 411):raise ValueError('Cservice completion identity differs')
  checks=[]
  if len(summary['integer_layers'])!=(0 if args.floating else 24):raise ValueError('Encoder arithmetic mode differs')
  for linear in summary['integer_layers']:
   name=linear['name'];actual=(folder/('actual_'+name+'_a8.bin')).read_bytes();expected=(folder/(name+'_reference_a8.bin')).read_bytes()
   if len(actual)!=len(expected) or len(actual)!=linear['input_codes']:raise ValueError('A8 extent differs')
   different=sum(a!=b for a,b in zip(actual,expected));checks.append(dict(name=name,values=len(actual),differences=different))
   if different!=linear['original_software_code_differences']:raise ValueError('Reported A8 discrepancies differ')
  output=dict(case_index=key[0],window_index=key[1],document_id=item['document_id'],label=item['label'],
   returncode=rc,elapsed_seconds=elapsed,folder=item['folder'],a8=checks,
   logits_equal=(folder/'bridge_logits.bin').read_bytes()==(folder/'reference_logits.bin').read_bytes(),
   risk_equal=(folder/'bridge_risk.bin').read_bytes()==(folder/'reference_risk.bin').read_bytes(),
   summary=summary,acceptance=None)
  state['windows'].append(output);count+=1
  print(json.dumps(dict(case=key,a8_differences=sum(v['differences'] for v in checks),
   logits_equal=output['logits_equal'],risk_equal=output['risk_equal'])),flush=True)
  if rc or any(v['differences'] for v in checks) or not output['logits_equal'] or not output['risk_equal']:
   state['state']='stopped_first_unexplained_discrepancy';save(statepath,state);raise RuntimeError('Inspect saved first connected discrepancy before further windows; diagnostic stop does not adopt universal bit equality')
  state['active_window']=None;save(statepath,state)
 state['complete']=len(state['windows'])==66
 if state['complete']:
  import struct
  bykey={(v['case_index'],v['window_index']):v for v in state['windows']}
  if set(bykey)!=set(assigned):raise ValueError('Complete case membership differs')
  documents=[];threshold=struct.unpack('<f',(DATA/'parameters/threshold.bin').read_bytes())[0]
  for case in exports['selection']['cases']:
   windows=[bykey[(case['case_index'],wi)] for wi in range(case['window_count'])]
   actual=max(v['summary']['risk'] for v in windows);expected=max(v['summary']['reference_risk'] for v in windows)
   documents.append(dict(document_id=case['document_id'],label=case['label'],windows=len(windows),actual_max=actual,reference_max=expected,threshold=threshold,actual_reject=actual>threshold,reference_reject=expected>threshold,decision_agrees=(actual>threshold)==(expected>threshold)))
  state['documents']=documents
 state['state']='complete_pending_acceptance' if state['complete'] else 'bounded_handoff';save(statepath,state)


def generate_bounds_tb(source):
 """Insert checks only; canonical arithmetic and its v1 source remain unchanged."""
 substitutions=[
  ('#include "fixed_fp32_service.hpp"', '#include "fixed_fp32_service.hpp"\n#include "connected_fp32_bounds.hpp"'),
  ('    finite(result,"service output");', '    finite(result,"service output");\n    connected_bounds::check(op,x,y,z,acc,result,rows,width,outputs,scale,bias,output_directory);'),
  ('            if(actual!=expected) throw std::runtime_error("same-input quantize mismatch index="+std::to_string(begin+i));', '            if(actual!=expected) {connected_bounds::quantization_failed(count_rows,width,i,actual,expected,output_directory); throw std::runtime_error("same-input quantize mismatch index="+std::to_string(begin+i));}'),
  ('            ++same_input_codes_checked;\n        }\n        row+=count_rows;', '            ++same_input_codes_checked;\n        }\n        connected_bounds::quantization_checked(count_rows,width,output_directory);\n        row+=count_rows;'),
  ('        <<"  \\"service_calls_by_op\\":[";', '        <<"  \\"connected_fp32_bounds\\":";\n    connected_bounds::write_summary(file,calls);\n    file<<",\\n  \\"service_calls_by_op\\":[";')]
 for before,after in substitutions:
  if source.count(before)!=1:raise ValueError('Canonical bounds insertion sentinel differs: '+repr(before))
  source=source.replace(before,after)
 return source


def validate_operator_bounds(folder,summary):
 """Read-only verification of complete per-call records, reusable at other lengths."""
 records=[json.loads(line) for line in (folder/'operator_bounds.jsonl').read_text().splitlines()]
 counts=[0]*9;values=[0]*9
 for index,item in enumerate(records,1):
  op=item['op']
  if type(op) is not int or op not in range(9):raise ValueError('Malformed checked operator')
  counts[op]+=1
  rows,width,outputs=(item[k] for k in ('rows','width','outputs'))
  if any(type(v) is not int or v<1 for v in (rows,width)):raise ValueError('Malformed checked shape')
  expected=rows*(outputs if op==0 else width)
  atol,rtol=(3e-5,3e-5) if op==0 else (2e-4,2e-4) if op==5 else (3e-6,3e-5) if op>=6 else (0,0)
  if (item['call_index'],item['op_call_index'],item['values_checked'],item['atol'],item['rtol'])!=(index,counts[op],expected,atol,rtol):raise ValueError('Checked operator coverage/bounds differ')
  if item['passed'] is not True or 'first_violation' in item or item['exact_formula_required']!=(op in (1,2,3,4)):raise RuntimeError('Saved operator violation requires review')
  for key in ('maximum_absolute_error','maximum_relative_error','maximum_bound_ratio','maximum_softmax_row_sum_error'):
   if not math.isfinite(item[key]) or item[key]<0:raise ValueError('Malformed operator error')
  if item['maximum_bound_ratio']>1 or (not atol and item['maximum_absolute_error']!=0):raise RuntimeError('Saved operator exceeds unchanged bound')
  if item['softmax_rows_checked']!=(rows if op==6 else 0) or item['softmax_row_sum_atol']!=(2e-5 if op==6 else None) or item['maximum_softmax_row_sum_error']>2e-5:raise RuntimeError('Saved softmax normalization/coverage differs')
  values[op]+=expected
 bounded=summary['connected_fp32_bounds']
 if bounded['schema']!='sentry_connected_fp32_bounds_v2' or bounded['all_service_calls_checked'] is not True or bounded['record']!='operator_bounds.jsonl':raise ValueError('Connected bound summary differs')
 if bounded['fp32_mantissa_bits']!=24 or bounded['reference_mantissa_bits']<53:raise ValueError('Connected reference precision differs')
 if counts!=summary['service_calls_by_op'] or counts!=bounded['checked_calls_by_op'] or values!=bounded['checked_values_by_op']:raise ValueError('Connected bound summary coverage differs')
 return dict(calls=len(records),checked_calls_by_op=counts,checked_values_by_op=values)


def validate_bounds_build(build,synthesis):
 source=ROOT/'build/fixed_fp32_hls';record=read(build/'compiled.json')
 if not record['passed'] or record['synthesis_recorded_utc']!=synthesis['recorded_utc']:raise ValueError('Bounds compiled/synthesis association differs')
 if not (build/'project/solution/csim/build/csim.exe').is_file():raise ValueError('Bounds C executable missing')
 for local,current in [('fixed_fp32_service.hpp',ROOT/'hardware/fixed_fp32_service.hpp'),('connected_fp32_bounds.hpp',ROOT/'hardware/connected_fp32_bounds.hpp'),('canonical_numerical_bridge_tb.cpp',ROOT/'hardware/numerical_bridge_tb.cpp'),('build_fixed_fp32_hls.py',ROOT/'scripts/build_fixed_fp32_hls.py')]:
  if (build/local).read_bytes()!=current.read_bytes():raise ValueError('Bounds copied current source differs: '+local)
 if (build/'numerical_bridge_tb.cpp').read_text()!=generate_bounds_tb((ROOT/'hardware/numerical_bridge_tb.cpp').read_text()):raise ValueError('Generated bounds TB differs')
 tree=ast.parse((ROOT/'scripts/build_fixed_fp32_hls.py').read_text(encoding='utf-8-sig'))
 tops=[call.args[0].value for call in ast.walk(tree) if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name) and call.func.value.id=='top' and call.func.attr=='write_text' and call.args and isinstance(call.args[0],ast.Constant)]
 if len(tops)!=1 or (source/'gate_fixed_fp32_top.cpp').read_bytes()!=tops[0].encode('utf-8') or (build/'gate_fixed_fp32_top.cpp').read_bytes()!=(source/'gate_fixed_fp32_top.cpp').read_bytes():raise ValueError('Bounds actual generator/current synthesized wrapper differs')
 if (source/'fixed_fp32_service.hpp').read_bytes()!=(ROOT/'hardware/fixed_fp32_service.hpp').read_bytes():raise ValueError('Bounds synthesized service differs')
 # Preserve and verify the actual generator, rather than requiring an unchanged
 # dispatch driver forever. Only its declared pure insertion function is used.
 copied=ast.parse((build/'run_qualification_cases.py').read_text())
 current=ast.parse(Path(__file__).read_text())
 find=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate_bounds_tb')
 if ast.dump(find(copied))!=ast.dump(find(current)):raise ValueError('Bounds generation source differs')
 return record



def validate_reviewed_resolution(path,entry,identity,mode,root=ROOT):
 """Validate an explicitly supplied coordinator review; never create a review.

 The caller must first validate its frozen input/window/compiled associations.
 This helper additionally rechecks actual output, per-call bounds and A8 bytes.
 It is shape-independent for the separately generated length testbenches.
 """
 root=Path(root).resolve();path=Path(path)
 path=(path if path.is_absolute() else root/path).resolve()
 if not path.is_relative_to(root) or not path.is_file():raise ValueError('Resolution must be an existing repository file')
 folder=(root/entry['folder']).resolve();input_folder=(root/entry['input_folder']).resolve()
 if not folder.is_relative_to(root) or not input_folder.is_relative_to(root):raise ValueError('Reviewed output/input path differs')
 summary=read(folder/'bridge_summary.json')
 if summary!=entry['summary'] or not summary['completed'] or entry['returncode']!=0:raise RuntimeError('Service/process/noncompletion failures cannot be reviewed away')
 expected_mode={'w8':'w8_a8_integer_contract','floating':'ordinary_floating_reference'}
 if mode not in expected_mode or summary['encoder_arithmetic']!=expected_mode[mode]:raise ValueError('Reviewed numerical mode differs')
 if summary['strict_decision_agrees'] is not True:raise RuntimeError('Strict decision disagreement remains unresolved')
 actual=validate_operator_bounds(folder,summary)
 if actual!=entry['operator_bounds']:raise ValueError('Reviewed actual operator coverage differs')
 if summary['same_input_quantization_exact']!=(None if mode=='floating' else True) or summary['integer_accumulators_exact']!=(None if mode=='floating' else True) or summary['same_input_codes_checked']!=actual['checked_values_by_op'][1] or summary['exact_accumulators_checked']!=actual['checked_values_by_op'][2]:raise RuntimeError('Same-input arithmetic failure cannot be resolved')
 process=folder/'csim_run.json'
 if process.exists():
  observed=read(process)
  if observed['state']!='returned' or observed['returncode']!=0:raise RuntimeError('Process failure cannot be reviewed away')
 from validate_qualification_exports import terminal
 terminal(folder,'bridge',summary['logits'],summary['risk']);terminal(input_folder,'reference',summary['reference_logits'],summary['reference_risk'])
 checks=[]
 for layer in summary['integer_layers']:
  name=layer['name'];codes=(folder/('actual_'+name+'_a8.bin')).read_bytes();reference=(input_folder/(name+'_reference_a8.bin')).read_bytes()
  if len(codes)!=len(reference) or len(codes)!=layer['input_codes']:raise ValueError('Reviewed actual A8 extent differs')
  differences=sum(a!=b for a,b in zip(codes,reference));checks.append(dict(name=name,values=len(codes),differences=differences))
  if differences!=layer['original_software_code_differences']:raise ValueError('Reviewed actual A8 summary differs')
 logits_equal=(folder/'bridge_logits.bin').read_bytes()==(input_folder/'reference_logits.bin').read_bytes()
 risk_equal=(folder/'bridge_risk.bin').read_bytes()==(input_folder/'reference_risk.bin').read_bytes()
 if checks!=entry['a8'] or logits_equal!=entry['logits_equal'] or risk_equal!=entry['risk_equal']:raise ValueError('Reviewed actual discrepancy differs')
 if logits_equal and risk_equal and not any(v['differences'] for v in checks):raise ValueError('No saved A8/terminal discrepancy to resolve')
 resolution=read(path)
 if resolution.get('policy_id')!='bounded_connected_development_v2' or resolution.get('coordinator_review') is not True or resolution.get('allow_resume') is not True or resolution.get('same_input_operator_checks_valid') is not True:raise ValueError('Explicit coordinator bounded-policy review required')
 expected=dict(mode=mode,case_index=entry['case_index'],window_index=entry['window_index'],document_id=entry['document_id'],identity=identity,summary=summary,a8=checks,operator_bounds=actual)
 if any(resolution.get(k)!=v for k,v in expected.items()):raise ValueError('Resolution is not bound to this actual case/window/identity/discrepancy')
 if not isinstance(resolution.get('reason'),str) or not resolution['reason'].strip():raise ValueError('Causal review reason required')
 evidence=resolution.get('evidence_paths')
 if not isinstance(evidence,list) or not evidence or any(not isinstance(v,str) or not v.strip() for v in evidence):raise ValueError('Actual causal evidence paths required')
 for name in evidence:
  candidate=Path(name)
  if candidate.is_absolute():raise ValueError('Use repository-relative causal evidence paths')
  candidate=(root/candidate).resolve()
  if not candidate.is_relative_to(root) or not candidate.is_file() or candidate==path:raise ValueError('Causal evidence file missing or outside repository')
 return dict(path=path.relative_to(root).as_posix(),record=resolution)


class ConnectedCMemoryGuard:
 """Two assigned C cores, independent mode trees, observed memory; no inference."""
 def __init__(self,config,output,mode,active=True):
  self.output,self.mode,self.active=output,mode,active;self.locks=[]
  limits=config['execution_control']['board_free_continuation']['resource_partition']
  self.partition=limits;self.limits=limits['connected_c_memory_guard']
  if self.limits['minimum_wsl_available_kib']<4*1024*1024 or self.limits['minimum_windows_free_kib']<2*1024*1024:raise ValueError('Do not relax connected C memory floors')
  if limits['connected_c_max_workers']!=2 or set(limits['connected_c_cpu_affinity'])!={6,7} or set(limits['vendor_cpu_affinity'])!=set(range(6)) or limits['vendor_jobs']!=1 or limits['vendor_max_threads']!=2 or limits['native_torch_concurrent'] is not False:raise ValueError('Configured connected C/vendor resource partition differs')
  self.path=output/('memory_guard_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')+'_'+str(os.getpid())+'.jsonl')
 def event(self,**item):
  item.update(observed_utc=datetime.now(timezone.utc).isoformat(),mode=self.mode,coordinator_pid=os.getpid())
  with self.path.open('a') as f:f.write(json.dumps(item,allow_nan=False)+'\n');f.flush()
 def __enter__(self):
  import fcntl
  self.output.mkdir(parents=True,exist_ok=True)
  try:
   paths=[self.output/'.worker.lock']
   if self.active:
    affinity=sorted(os.sched_getaffinity(0))
    if len(affinity)!=1 or affinity[0] not in (6,7):raise RuntimeError('Assign each connected C worker exactly one of CPUs6/7')
    paths.append(ROOT/f'results/goal4/numerical_bridge/connected_c_cpu{affinity[0]}.lock')
   for path in paths:
    handle=path.open('a');self.locks.append(handle);fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
   if self.active:self.observe('dispatch_preflight')
   return self
  except BaseException:
   self.__exit__(None,None,None);raise
 def __exit__(self,*unused):
  for handle in reversed(self.locks):handle.close()
  self.locks=[]
 def observe(self,stage,child_pid=None):
  observation=dict(stage=stage,child_pid=child_pid,child_group=child_pid,limits=self.limits,cpu_affinity=sorted(os.sched_getaffinity(0)),wsl_available_kib=None,windows_free_kib=None)
  try:
   memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','SwapFree:'))}
   observation.update(wsl_available_kib=memory['MemAvailable'],wsl_swap_free_kib=memory.get('SwapFree'))
   host=subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-Command','(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'],capture_output=True,text=True,timeout=10)
   if host.returncode!=0:raise RuntimeError('Windows free-memory counter failed: '+host.stderr.strip())
   observation['windows_free_kib']=int(host.stdout.strip())
   if observation['wsl_available_kib']<self.limits['minimum_wsl_available_kib'] or observation['windows_free_kib']<self.limits['minimum_windows_free_kib']:raise RuntimeError('Connected C available-memory guard tripped')
   observation['passed']=True;self.event(**observation)
  except BaseException as error:
   observation.update(passed=False,error=str(error));self.event(**observation);raise


def validate_recognized_floating_policy(path,identity,selection,root=ROOT):
 root=Path(root).resolve();path=Path(path);path=(path if path.is_absolute() else root/path).resolve()
 if not path.is_relative_to(root) or not path.is_file():raise ValueError('Existing repository floating policy required')
 policy=read(path)
 required=dict(schema='sentry_recognized_floating_dispatch_v1',policy_id='bounded_connected_development_v2',dispatch_class='floating_embedding_ln_neighbor_denominator_v1',mode='floating',coordinator_review=True,allow_prospective_dispatch=True,identity=identity,selection=selection,numerical_acceptance=None)
 if any(policy.get(k)!=v for k,v in required.items()) or policy.get('coordinator_review') is not True or policy.get('allow_prospective_dispatch') is not True:raise ValueError('Recognized floating policy/identity/selection differs')
 helper=(root/policy['classifier_source_path']).resolve()
 if not helper.is_relative_to(root) or helper.read_text()!=policy['classifier_source']:raise ValueError('Recognized floating classifier source differs')
 if helper!=root/'results/goal4/numerical_bridge/connected_bounds_review_v2/saved_ln_boundary.py':raise ValueError('Only the explicitly reviewed classifier is permitted')
 if not isinstance(policy.get('evidence_paths'),list) or not policy['evidence_paths']:raise ValueError('Reviewed floating evidence paths required')
 for value in policy['evidence_paths']:
  evidence=(root/value).resolve()
  if Path(value).is_absolute() or not evidence.is_relative_to(root) or not evidence.is_file():raise ValueError('Reviewed floating evidence file missing')
 return dict(path=path.relative_to(root).as_posix(),record=policy)


def recognized_floating_entry(entry,identity,selection,parameters,*,policy=None,write=False,root=ROOT):
 """Recompute all-row recognition; never infer acceptance or relax A8 gates."""
 root=Path(root).resolve();stored=entry.get('recognized_floating_propagation')
 if stored:
  validated=validate_recognized_floating_policy(stored['policy_path'],identity,selection,root)
  if policy is not None and policy!=validated:raise ValueError('Stored prospective floating policy differs')
  policy=validated
 if policy is None:raise RuntimeError('Explicit prospective floating policy required')
 summary=entry['summary'];folder=root/entry['folder'];reference=root/entry['input_folder']
 if summary['encoder_arithmetic']!='ordinary_floating_reference' or entry['a8'] or summary['integer_layers'] or entry['returncode'] or summary['strict_decision_agrees'] is not True or entry['acceptance'] is not None:raise RuntimeError('Recognized class cannot waive mode/A8/service/decision failures')
 actual=validate_operator_bounds(folder,summary)
 if actual!=entry['operator_bounds'] or summary['same_input_quantization_exact'] is not None or summary['integer_accumulators_exact'] is not None:raise RuntimeError('Recognized class requires all original same-input checks')
 helper=root/policy['record']['classifier_source_path'];spec=importlib.util.spec_from_file_location('sentry_saved_ln_boundary',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 classified=module.classify_embedding_ln(folder,reference,parameters,expected_length=256)
 recognized=(classified['schema']=='sentry_saved_embedding_ln_denominator_class_v1' and classified['length']==256 and classified['width']==256 and classified['rows_checked']==256 and classified['values_checked']==65536 and classified['all_rows_recognized'] is True and classified['first_unrecognized'] is None and classified['numerical_acceptance'] is None)
 artifact=folder/'recognized_floating_rows.json';policy_copy=folder.parent.parent/'recognized_floating_policy.json';helper_copy=folder.parent.parent/'recognized_floating_classifier.py'
 record=dict(schema='sentry_recognized_floating_entry_v1',policy_path=policy['path'],policy_snapshot=policy_copy.relative_to(root).as_posix(),classifier_copy=helper_copy.relative_to(root).as_posix(),classification_path=artifact.relative_to(root).as_posix(),classification_summary={k:v for k,v in classified.items() if k!='rows'},numerical_acceptance=None)
 if stored:
  if stored!=record or read(artifact)!=classified or policy_copy.read_bytes()!=(root/policy['path']).read_bytes() or helper_copy.read_bytes()!=helper.read_bytes():raise ValueError('Saved floating classification/policy/source association differs')
 elif write:
  for target,raw in [(policy_copy,(root/policy['path']).read_bytes()),(helper_copy,helper.read_bytes())]:
   if target.exists() and target.read_bytes()!=raw:raise ValueError('Preserve original floating policy/classifier snapshot')
   if not target.exists():target.write_bytes(raw)
  if artifact.exists():raise ValueError('Preserve earlier uncommitted floating classification')
  save(artifact,classified)
 else:raise RuntimeError('Saved recognized floating record absent')
 if write:entry['floating_classification_attempt']=dict(classification_path=record['classification_path'],recognized=recognized,policy_path=record['policy_path'])
 if not recognized:raise RuntimeError('Floating propagation row class is not recognized; full row evidence retained')
 return record

def validate_recognized_w8_policy(path,identity,selection,root=ROOT):
 root=Path(root).resolve();path=Path(path);path=(path if path.is_absolute() else root/path).resolve()
 if not path.is_relative_to(root) or not path.is_file():raise ValueError('Existing repository W8 policy required')
 policy=read(path)
 required=dict(schema='sentry_recognized_w8_dispatch_v1',policy_id='bounded_connected_development_v2',dispatch_class='w8_exact_a8_embedding_ln_same_tail_v1',mode='w8',coordinator_review=True,allow_prospective_dispatch=True,identity=identity,selection=selection,numerical_acceptance=None)
 if any(policy.get(k)!=v for k,v in required.items()) or policy.get('coordinator_review') is not True or policy.get('allow_prospective_dispatch') is not True:raise ValueError('Recognized W8 policy/identity/selection differs')
 if identity['windowing']['primary_length']!=256 or len(identity['precision_map'])!=16 or any(type(v) is not int or v!=8 for v in identity['precision_map'].values()):raise ValueError('Recognized W8 policy is current W8/L256 only')
 base=root/'results/goal4/numerical_bridge/connected_bounds_review_v2'
 for key,name in [('classifier','saved_w8_propagation.py'),('dependent_classifier','saved_ln_boundary.py')]:
  helper=(root/policy[key+'_source_path']).resolve()
  if helper!=base/name or helper.read_text(encoding='utf-8-sig')!=policy[key+'_source'].lstrip('\ufeff'):raise ValueError('Reviewed W8 classifier/dependency source differs')
 if not isinstance(policy.get('evidence_paths'),list) or not policy['evidence_paths']:raise ValueError('Reviewed W8 evidence paths required')
 for value in policy['evidence_paths']:
  evidence=(root/value).resolve()
  if Path(value).is_absolute() or not evidence.is_relative_to(root) or not evidence.is_file():raise ValueError('Reviewed W8 evidence missing')
 return dict(path=path.relative_to(root).as_posix(),record=policy)


def recognized_w8_entry(entry,identity,selection,parameters,*,policy=None,write=False,root=ROOT):
 """Explicit prospective exact-A8/all-row-LN/both-tail class; acceptance stays null."""
 root=Path(root).resolve();stored=entry.get('recognized_w8_propagation')
 if stored:
  validated=validate_recognized_w8_policy(stored['policy_path'],identity,selection,root)
  if policy is not None and policy!=validated:raise ValueError('Stored prospective W8 policy differs')
  policy=validated
 if policy is None:raise RuntimeError('Explicit prospective W8 policy required')
 summary=entry['summary'];folder=root/entry['folder'];reference=root/entry['input_folder'];parameters=Path(parameters)
 if summary['encoder_arithmetic']!='w8_a8_integer_contract' or entry['returncode']!=0 or entry['acceptance'] is not None or summary['completed'] is not True:raise RuntimeError('W8 class cannot waive mode/service/completion failure')
 actual=validate_operator_bounds(folder,summary)
 if actual!=entry['operator_bounds'] or actual['calls']!=411 or actual['checked_calls_by_op']!=[165,72,72,2,16,18,33,32,1] or summary['same_input_quantization_exact'] is not True or summary['integer_accumulators_exact'] is not True or summary['same_input_codes_checked']!=2359296 or summary['exact_accumulators_checked']!=2359296:raise RuntimeError('W8 class requires all original exact/bounded operations')
 from validate_qualification_exports import LINEARS,terminal,_same_path
 names=[f'layer{i}_{name}' for i in range(4) for name in LINEARS];checks=[]
 if [v['name'] for v in summary['integer_layers']]!=names:raise ValueError('W8 class integer membership differs')
 for layer in summary['integer_layers']:
  name=layer['name'];k,n=LINEARS[name.split('_',1)[1]];a=(folder/('actual_'+name+'_a8.bin')).read_bytes();b=(reference/(name+'_reference_a8.bin')).read_bytes()
  if len(a)!=256*k or len(b)!=256*k or a!=b or layer['input_codes']!=256*k or layer['accumulators_checked']!=256*n or layer['original_software_code_differences']!=0:raise RuntimeError('W8 class cannot waive any A8 difference')
  checks.append(dict(name=name,values=256*k,differences=0))
 if entry['a8']!=checks:raise ValueError('W8 class original A8 records differ')
 terminal(folder,'bridge',summary['logits'],summary['risk']);terminal(reference,'reference',summary['reference_logits'],summary['reference_risk'])
 threshold=struct.unpack('<f',(parameters/'threshold.bin').read_bytes())[0];risk=summary['risk'];ref=summary['reference_risk']
 if not(0<=risk<=1 and 0<=ref<=1) or summary['threshold']!=threshold or summary['strict_reject']!=(risk>threshold) or summary['reference_strict_reject']!=(ref>threshold) or summary['strict_decision_agrees'] is not True or (risk>threshold)!=(ref>threshold) or summary['end_to_end_score_tolerance'] is not None or summary['risk_absolute_error']!=abs(risk-ref):raise RuntimeError('W8 class cannot waive frozen strict decisions/no epsilon')
 observed=read(folder/'csim_run.json');build=root/'build/development_qualification_bounds_v2'
 if observed['state']!='returned' or observed['returncode']!=0 or observed['identity']!=identity or tuple(observed['window'])!=(entry['case_index'],entry['window_index']) or len(observed['command'])!=1 or not _same_path(observed['command'][0],build/'project/solution/csim/build/csim.exe',root) or observed.get('bounds') is not True or not (folder.parent.parent/observed['source']).is_file():raise RuntimeError('W8 class requires actual completed associated process')
 revisions=__import__('re').findall(r'^#define GATE_FIXED_FP32_SCHEDULE_REVISION "([^"]+)"$',(build/'fixed_fp32_service.hpp').read_text(),__import__('re').M)
 if revisions!=[summary['source_revision']]:raise ValueError('W8 class current compiled source revision differs')
 helper=root/policy['record']['classifier_source_path'];dependent=root/policy['record']['dependent_classifier_source_path'];artifact=folder/'recognized_w8_mechanism.json';base=folder.parent.parent
 policy_copy=base/'recognized_w8_policy.json';helper_copy=base/'recognized_w8_classifier.py';dependent_copy=base/'recognized_w8_ln_classifier.py'
 snapshots=[(policy_copy,(root/policy['path']).read_bytes()),(helper_copy,helper.read_bytes()),(dependent_copy,dependent.read_bytes())]
 if write and not stored:
  for target,raw in snapshots:
   if target.exists() and target.read_bytes()!=raw:raise ValueError('Preserve original W8 policy/classifier snapshot')
   if not target.exists():target.write_bytes(raw)
  if artifact.exists():raise ValueError('Preserve earlier uncommitted W8 classification')
 try:
  spec=importlib.util.spec_from_file_location('sentry_saved_w8_propagation',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  classified=module.classify_w8_propagation(folder,reference,parameters,expected_length=256)
 except Exception as error:
  if write and not stored:
   save(artifact,dict(schema='sentry_w8_classification_failure_v1',error=repr(error),numerical_acceptance=None));entry['w8_classification_attempt']=dict(classification_path=artifact.relative_to(root).as_posix(),recognized=False,policy_path=policy['path'])
  raise
 recognized=(classified['schema']=='sentry_saved_w8_exact_a8_propagation_class_v1' and classified['length']==256 and classified['width']==256 and classified['recognized_mechanism'] is True and classified['all_embedding_ln_rows_recognized'] is True and classified['all_a8_equal'] is True and classified['both_backend_terminal_bits_reproduced'] is True and classified['first_changed_stage']=='embedding_ln' and classified['numerical_acceptance'] is None)
 record=dict(schema='sentry_recognized_w8_entry_v1',policy_path=policy['path'],policy_snapshot=policy_copy.relative_to(root).as_posix(),classifier_copy=helper_copy.relative_to(root).as_posix(),dependent_classifier_copy=dependent_copy.relative_to(root).as_posix(),classification_path=artifact.relative_to(root).as_posix(),classification_summary={k:classified[k] for k in ('schema','length','width','recognized_mechanism','all_embedding_ln_rows_recognized','all_a8_equal','both_backend_terminal_bits_reproduced','first_changed_stage','first_changed_cls_stage','numerical_acceptance')},numerical_acceptance=None)
 if stored:
  if stored!=record or read(artifact)!=classified or any(target.read_bytes()!=raw for target,raw in snapshots):raise ValueError('Saved W8 classification/policy/both-source association differs')
 elif write:
  save(artifact,classified);entry['w8_classification_attempt']=dict(classification_path=record['classification_path'],recognized=recognized,policy_path=policy['path'])
 else:raise RuntimeError('Saved recognized W8 record absent')
 if [v['name'] for v in summary['intermediate_errors']]!=['embedding_ln','layer0','layer1','layer2','layer3','logits','risk']:raise ValueError('W8 class captured stage membership differs')
 for stage,measured in zip(summary['intermediate_errors'],classified['stages']):
  if stage['name']!=measured['name'] or stage['values']!=measured['values'] or stage['maximum_absolute_error']!=measured['maximum_absolute_error']:raise ValueError('W8 class captured error report differs')
 if not recognized:raise RuntimeError('W8 propagation class not recognized; diagnostic retained')
 return record


TAIL_BUILD='build/development_w8_tail_diagnostic_v1'
TAIL_DIAGNOSTIC='results/goal4/numerical_bridge/connected_bounds_review_v2/w8_case04_window000_tail_diagnostic_v1'


def validate_captured_tail_build(build,synthesis,root=ROOT):
 """Reuse the actually completed output-only diagnostic; never recompile it."""
 root=Path(root).resolve();build=Path(build).resolve()
 if build!=root/TAIL_BUILD:raise ValueError('Only the completed captured-tail build is permitted')
 original_build=root/'build/development_qualification_bounds_v2';base=validate_bounds_build(original_build,synthesis)
 actual=root/TAIL_DIAGNOSTIC;record=read(actual/'run.json');prepared=read(actual/'prepared.json')
 if record['state']!='returned' or record['passed'] is not True or record['returncode']!=0 or record['case']!=[4,0] or record['expected_service_calls']!=411 or record['actual_tail_captured'] is not True or record['all_original_operator_records_exact'] is not True or record['new_synthesis'] is not False or record['numerical_acceptance'] is not None:raise ValueError('Successful original captured diagnostic required')
 if prepared['compile_run'] is not False or prepared['identity']!=record['identity'] or prepared['expected_service_calls']!=411:raise ValueError('Original diagnostic preparation differs')
 for name,current in [('fixed_fp32_service.hpp',root/'hardware/fixed_fp32_service.hpp'),('connected_fp32_bounds.hpp',root/'hardware/connected_fp32_bounds.hpp'),('canonical_numerical_bridge_tb.cpp',root/'hardware/numerical_bridge_tb.cpp'),('gate_fixed_fp32_top.cpp',root/'build/fixed_fp32_hls/gate_fixed_fp32_top.cpp'),('build_fixed_fp32_hls.py',root/'scripts/build_fixed_fp32_hls.py'),('prepare_w8_tail_diagnostic.py',root/'results/goal4/numerical_bridge/connected_bounds_review_v2/prepare_w8_tail_diagnostic.py')]:
  if (build/name).read_bytes()!=current.read_bytes():raise ValueError('Captured diagnostic source differs: '+name)
 # Dispatch changes may follow the diagnostic; retain its actual old driver,
 # requiring only the unchanged exact bounds-generation function association.
 find=lambda raw:next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='generate_bounds_tb')
 if ast.dump(find((build/'bounds_generator_driver.py').read_text()))!=ast.dump(find(Path(__file__).read_text())):raise ValueError('Captured diagnostic bounds generator changed')
 sentinel='        pooled=fp(gate::FP32_TANH,pooled,{},{},1,D);'
 insert='        write_binary("diagnostic_pooler_pre_tanh",pooled);\n'+sentinel+'\n        write_binary("diagnostic_pooler_post_tanh",pooled);'
 canonical=(root/'hardware/numerical_bridge_tb.cpp').read_text();bounded=generate_bounds_tb(canonical)
 if bounded.count(sentinel)!=1 or (build/'numerical_bridge_tb.cpp').read_text()!=bounded.replace(sentinel,insert):raise ValueError('Captured source is not exactly original bounds plus two output writes')
 if '-ffp-contract=off' not in record['flags'] or record['flags']!=prepared['flags'] or 'config_compile -unsafe_math_optimizations=false' not in (build/'qualify.tcl').read_text() or not (build/'project/solution/csim/build/csim.exe').is_file() or 'CSim done with 0 errors.' not in (actual/'compile_and_case.log').read_text(errors='replace'):raise ValueError('Actual captured C executable/flags missing')
 if record['command'][-1].replace('\\','/')!='/home/jp19126/Projects/SENTRY/'+TAIL_BUILD+'/qualify.tcl':raise ValueError('Actual diagnostic compile command differs')
 original=root/record['original_actual_folder'];summary=read(actual/'bridge_summary.json')
 if summary!=read(original/'bridge_summary.json') or validate_operator_bounds(actual,summary)!=record['operator_bounds'] or record['operator_bounds']['calls']!=411 or (actual/'operator_bounds.jsonl').read_bytes()!=(original/'operator_bounds.jsonl').read_bytes():raise ValueError('Original diagnostic summary/operator evidence changed')
 originals={v.name for v in original.glob('*.bin')}
 if set(record['all_original_numerical_bin_files_exact'])!=originals or any((actual/name).read_bytes()!=(original/name).read_bytes() for name in originals):raise ValueError('Original diagnostic numerical bytes changed')
 for name in ('diagnostic_pooler_pre_tanh','diagnostic_pooler_post_tanh'):
  raw=(actual/(name+'.bin')).read_bytes()
  if len(raw)!=1024 or not all(math.isfinite(v) for v in struct.unpack('<256f',raw)):raise ValueError('Original finite diagnostic captures missing')
 return dict(record,compiled_first_window=TAIL_DIAGNOSTIC,synthesis_recorded_utc=base['synthesis_recorded_utc'])


def validate_recognized_w8_captured_tail_policy(path,identity,selection,root=ROOT):
 root=Path(root).resolve();path=Path(path);path=(path if path.is_absolute() else root/path).resolve()
 if not path.is_relative_to(root) or not path.is_file():raise ValueError('Existing repository captured-tail policy required')
 policy=read(path)
 required=dict(schema='sentry_recognized_w8_captured_tail_dispatch_v1',policy_id='bounded_connected_development_v2',dispatch_class='w8_exact_a8_embedding_ln_actual_bounded_tanh_tail_v1',mode='w8',coordinator_review=True,allow_prospective_dispatch=True,identity=identity,selection=selection,compiled_build=TAIL_BUILD,diagnostic_run_path=TAIL_DIAGNOSTIC+'/run.json',numerical_acceptance=None)
 if any(policy.get(k)!=v for k,v in required.items()) or policy.get('coordinator_review') is not True or policy.get('allow_prospective_dispatch') is not True:raise ValueError('Captured-tail policy/identity/selection/build differs')
 if identity['windowing']['primary_length']!=256 or len(identity['precision_map'])!=16 or any(type(v) is not int or v!=8 for v in identity['precision_map'].values()):raise ValueError('Captured policy is current W8/L256 only')
 base=root/'results/goal4/numerical_bridge/connected_bounds_review_v2'
 for key,name in [('classifier','saved_w8_captured_tail.py'),('w8_classifier','saved_w8_propagation.py'),('ln_classifier','saved_ln_boundary.py')]:
  helper=(root/policy[key+'_source_path']).resolve()
  if helper!=base/name or helper.read_text(encoding='utf-8-sig')!=policy[key+'_source'].lstrip('\ufeff'):raise ValueError('Captured-tail classifier/dependency source differs')
 if not isinstance(policy.get('evidence_paths'),list) or not policy['evidence_paths']:raise ValueError('Captured-tail reviewed evidence required')
 for value in policy['evidence_paths']:
  evidence=(root/value).resolve()
  if Path(value).is_absolute() or not evidence.is_relative_to(root) or not evidence.is_file():raise ValueError('Captured-tail evidence missing')
 return dict(path=path.relative_to(root).as_posix(),record=policy)


def recognized_w8_captured_tail_entry(entry,identity,selection,parameters,*,policy=None,write=False,root=ROOT):
 """Explicit prospective exact-A8/all-row-LN/both-tail class; acceptance stays null."""
 root=Path(root).resolve();stored=entry.get('recognized_w8_captured_tail')
 if stored:
  validated=validate_recognized_w8_captured_tail_policy(stored['policy_path'],identity,selection,root)
  if policy is not None and policy!=validated:raise ValueError('Stored prospective W8 policy differs')
  policy=validated
 if policy is None:raise RuntimeError('Explicit prospective W8 policy required')
 if entry.get('compiled_build')!=TAIL_BUILD:raise ValueError('Captured-tail record requires its actual per-entry build')
 summary=entry['summary'];folder=root/entry['folder'];reference=root/entry['input_folder'];parameters=Path(parameters)
 if summary['encoder_arithmetic']!='w8_a8_integer_contract' or entry['returncode']!=0 or entry['acceptance'] is not None or summary['completed'] is not True:raise RuntimeError('W8 class cannot waive mode/service/completion failure')
 actual=validate_operator_bounds(folder,summary)
 if actual!=entry['operator_bounds'] or actual['calls']!=411 or actual['checked_calls_by_op']!=[165,72,72,2,16,18,33,32,1] or summary['same_input_quantization_exact'] is not True or summary['integer_accumulators_exact'] is not True or summary['same_input_codes_checked']!=2359296 or summary['exact_accumulators_checked']!=2359296:raise RuntimeError('W8 class requires all original exact/bounded operations')
 from validate_qualification_exports import LINEARS,terminal,_same_path
 names=[f'layer{i}_{name}' for i in range(4) for name in LINEARS];checks=[]
 if [v['name'] for v in summary['integer_layers']]!=names:raise ValueError('W8 class integer membership differs')
 for layer in summary['integer_layers']:
  name=layer['name'];k,n=LINEARS[name.split('_',1)[1]];a=(folder/('actual_'+name+'_a8.bin')).read_bytes();b=(reference/(name+'_reference_a8.bin')).read_bytes()
  if len(a)!=256*k or len(b)!=256*k or a!=b or layer['input_codes']!=256*k or layer['accumulators_checked']!=256*n or layer['original_software_code_differences']!=0:raise RuntimeError('W8 class cannot waive any A8 difference')
  checks.append(dict(name=name,values=256*k,differences=0))
 if entry['a8']!=checks:raise ValueError('W8 class original A8 records differ')
 terminal(folder,'bridge',summary['logits'],summary['risk']);terminal(reference,'reference',summary['reference_logits'],summary['reference_risk'])
 threshold=struct.unpack('<f',(parameters/'threshold.bin').read_bytes())[0];risk=summary['risk'];ref=summary['reference_risk']
 if not(0<=risk<=1 and 0<=ref<=1) or summary['threshold']!=threshold or summary['strict_reject']!=(risk>threshold) or summary['reference_strict_reject']!=(ref>threshold) or summary['strict_decision_agrees'] is not True or (risk>threshold)!=(ref>threshold) or summary['end_to_end_score_tolerance'] is not None or summary['risk_absolute_error']!=abs(risk-ref):raise RuntimeError('W8 class cannot waive frozen strict decisions/no epsilon')
 observed=read(folder/'csim_run.json');build=root/TAIL_BUILD
 if observed['state']!='returned' or observed['returncode']!=0 or observed['identity']!=identity or tuple(observed['window'])!=(entry['case_index'],entry['window_index']) or len(observed['command'])!=1 or not _same_path(observed['command'][0],build/'project/solution/csim/build/csim.exe',root) or observed.get('bounds') is not True or observed.get('compiled_build')!=TAIL_BUILD or not (folder.parent.parent/observed['source']).is_file():raise RuntimeError('W8 class requires actual completed associated process')
 revisions=__import__('re').findall(r'^#define GATE_FIXED_FP32_SCHEDULE_REVISION "([^"]+)"$',(build/'fixed_fp32_service.hpp').read_text(),__import__('re').M)
 if revisions!=[summary['source_revision']]:raise ValueError('W8 class current compiled source revision differs')
 helper=root/policy['record']['classifier_source_path'];dependent=root/policy['record']['w8_classifier_source_path'];ln_helper=root/policy['record']['ln_classifier_source_path'];artifact=folder/'recognized_w8_captured_tail_mechanism.json';base=folder.parent.parent
 policy_copy=base/'recognized_w8_captured_tail_policy.json';helper_copy=base/'recognized_w8_captured_tail_classifier.py';dependent_copy=base/'recognized_w8_captured_tail_w8_classifier.py';ln_copy=base/'recognized_w8_captured_tail_ln_classifier.py'
 snapshots=[(policy_copy,(root/policy['path']).read_bytes()),(helper_copy,helper.read_bytes()),(dependent_copy,dependent.read_bytes()),(ln_copy,ln_helper.read_bytes())]
 if write and not stored:
  for target,raw in snapshots:
   if target.exists() and target.read_bytes()!=raw:raise ValueError('Preserve original W8 policy/classifier snapshot')
   if not target.exists():target.write_bytes(raw)
  if artifact.exists():raise ValueError('Preserve earlier uncommitted W8 classification')
 try:
  spec=importlib.util.spec_from_file_location('sentry_saved_w8_propagation',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  classified=module.classify_captured_w8_tail(folder,reference,parameters,expected_length=256)
 except Exception as error:
  if write and not stored:
   save(artifact,dict(schema='sentry_w8_classification_failure_v1',error=repr(error),numerical_acceptance=None));entry['w8_captured_tail_classification_attempt']=dict(classification_path=artifact.relative_to(root).as_posix(),recognized=False,policy_path=policy['path'])
  raise
 recognized=(classified['schema']=='sentry_saved_w8_captured_bounded_tanh_class_v1' and classified['length']==256 and classified['width']==256 and classified['recognized_mechanism'] is True and classified['all_embedding_ln_rows_recognized'] is True and classified['all_a8_equal'] is True and classified['actual_c_pooler_pre_dot_exact'] is True and classified['actual_tanh_original_bounds_valid'] is True and classified['both_terminal_bits_reproduced'] is True and classified['strict_frozen_decision_agrees'] is True and classified['first_changed_stage']=='embedding_ln' and classified['numerical_acceptance'] is None)
 record=dict(schema='sentry_recognized_w8_captured_tail_entry_v1',compiled_build=TAIL_BUILD,policy_path=policy['path'],policy_snapshot=policy_copy.relative_to(root).as_posix(),classifier_copy=helper_copy.relative_to(root).as_posix(),dependent_classifier_copy=dependent_copy.relative_to(root).as_posix(),ln_classifier_copy=ln_copy.relative_to(root).as_posix(),classification_path=artifact.relative_to(root).as_posix(),classification_summary={k:classified[k] for k in ('schema','length','width','recognized_mechanism','all_embedding_ln_rows_recognized','all_a8_equal','actual_c_pooler_pre_dot_exact','actual_tanh_original_bounds_valid','both_terminal_bits_reproduced','strict_frozen_decision_agrees','first_changed_stage','first_changed_cls_stage','numerical_acceptance')},numerical_acceptance=None)
 if stored:
  if stored!=record or read(artifact)!=classified or any(target.read_bytes()!=raw for target,raw in snapshots):raise ValueError('Saved W8 classification/policy/both-source association differs')
 elif write:
  save(artifact,classified);entry['w8_captured_tail_classification_attempt']=dict(classification_path=record['classification_path'],recognized=recognized,policy_path=policy['path'])
 else:raise RuntimeError('Saved recognized W8 record absent')
 if [v['name'] for v in summary['intermediate_errors']]!=['embedding_ln','layer0','layer1','layer2','layer3','logits','risk']:raise ValueError('W8 class captured stage membership differs')
 for stage in summary['intermediate_errors']:
  name=stage['name'];count=2 if name=='logits' else 1 if name=='risk' else 65536
  a=(folder/('bridge_'+name+'.bin')).read_bytes();b=(reference/('reference_'+name+'.bin')).read_bytes()
  if len(a)!=count*4 or len(b)!=count*4:raise ValueError('Captured stage extent differs')
  av=struct.unpack('<'+str(count)+'f',a);bv=struct.unpack('<'+str(count)+'f',b)
  if stage['values']!=count or not all(math.isfinite(v) for v in (*av,*bv)) or stage['maximum_absolute_error']!=max(abs(x-y) for x,y in zip(av,bv)):raise ValueError('Captured stage/error differs')
 if not recognized:raise RuntimeError('W8 propagation class not recognized; diagnostic retained')
 return record


def bounds_main(args):
 driver_source=Path(__file__).read_bytes()
 mode='floating' if args.floating else 'w8'
 inputs=ROOT/f'results/goal4/numerical_bridge/development_qualification_{mode}_v1'
 output=inputs/'bounds_v2';build=ROOT/(TAIL_BUILD if args.recognized_w8_captured_tail_policy else 'build/development_qualification_bounds_v2')
 generated=generate_bounds_tb((ROOT/'hardware/numerical_bridge_tb.cpp').read_text())
 if not args.execute:
  print(json.dumps(dict(execute=False,bounds=True,compile=args.compile,max_new_windows=args.max_new_windows,reviewed_resolution=args.reviewed_resolution,recognized_floating_policy=args.recognized_floating_policy,recognized_w8_policy=args.recognized_w8_policy,recognized_w8_captured_tail_policy=args.recognized_w8_captured_tail_policy,
   input=str(inputs/'export.json'),output=str(output),build=str(build),source_generation_checked=True,
   requires_complete_b1_exports=True,timeout_seconds=1800,numerical_acceptance=None,
   scope='Fresh actual C outputs with independent same-input original operator bounds; no synthesis or scoring')));return
 if os.name=='nt':raise RuntimeError('Run bounds with WSL Python and installed Vitis')
 if args.max_new_windows<0 or (args.max_new_windows==0 and not args.reviewed_resolution):raise ValueError('Positive work bound or explicit zero-work review required')
 paused();exports=read(inputs/'export.json');selection=read(ROOT/'results/goal4/numerical_bridge/development_cases_v1/selection.json')
 if exports['selection']!=selection or exports.get('floating',False)!=args.floating:raise ValueError('Bounds input selection/mode differs')
 if not exports['complete'] or len(exports['completed_windows'])!=66:raise ValueError('Bounds mode requires all 66 frozen B1 exports')
 from validate_qualification_exports import validate_exports,validate_reference_inputs,terminal
 # B1 export association is independent of unresolved historical v1 C results.
 validate_exports(inputs,qualification=dict(identity=exports['identity'],selection=selection,windows=[],complete=False))
 reference_association=validate_reference_inputs(inputs)
 prospective_policy=validate_recognized_floating_policy(args.recognized_floating_policy,exports['identity'],selection) if args.recognized_floating_policy else None
 w8_policy=validate_recognized_w8_policy(args.recognized_w8_policy,exports['identity'],selection) if args.recognized_w8_policy else None
 captured_policy=validate_recognized_w8_captured_tail_policy(args.recognized_w8_captured_tail_policy,exports['identity'],selection) if args.recognized_w8_captured_tail_policy else None
 config=read(ROOT/'configs/project.json');source=ROOT/'build/fixed_fp32_hls';synthesis=read(source/'run.json')
 if not synthesis['synthesis_passed']:raise ValueError('Successful common FP32 synthesis required')
 tree=ast.parse((ROOT/'scripts/build_fixed_fp32_hls.py').read_text(encoding='utf-8-sig'))
 tops=[call.args[0].value for call in ast.walk(tree) if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name) and call.func.value.id=='top' and call.func.attr=='write_text' and call.args and isinstance(call.args[0],ast.Constant)]
 if len(tops)!=1 or (source/'gate_fixed_fp32_top.cpp').read_bytes()!=tops[0].encode('utf-8') or (source/'fixed_fp32_service.hpp').read_bytes()!=(ROOT/'hardware/fixed_fp32_service.hpp').read_bytes():raise ValueError('Current successful FP source/generator association differs')
 assigned={(c['case_index'],wi) for c in selection['cases'] for wi in range(c['window_count'])}
 def folder_for(item):return output/f"case{item['case_index']:02d}"/f"window{item['window_index']:03d}"
 def env_for(item):
  env=os.environ.copy();env.update(GATE_NUMERICAL_BRIDGE_DIR=str(ROOT/item['folder']),
   GATE_NUMERICAL_BRIDGE_PARAMETER_DIR=str(inputs/'parameters'),GATE_NUMERICAL_BRIDGE_CASE=item['document_id'],
   GATE_NUMERICAL_BRIDGE_DIAGNOSTIC_DIR=str(folder_for(item)))
  if args.floating:env['GATE_NUMERICAL_BRIDGE_FLOATING']='1'
  else:env.pop('GATE_NUMERICAL_BRIDGE_FLOATING',None)
  return env
 with ConnectedCMemoryGuard(config,output,mode,active=args.max_new_windows>0 or args.compile) as memory_guard:
  dispatch=output/f'driver_dispatch_{len(list(output.glob("driver_dispatch_*.py")))+1:03d}.py';dispatch.write_bytes(driver_source)
  executable=build/'project/solution/csim/build/csim.exe';compiledpath=build/'compiled.json';first=exports['completed_windows'][0]
  if args.compile:
   if build.exists():raise ValueError('Fresh bounds build already exists; preserve and inspect/reuse it')
   build.mkdir(parents=True);folder_for(first).mkdir(parents=True,exist_ok=False)
   for name in ('fixed_fp32_service.hpp','connected_fp32_bounds.hpp'):shutil.copyfile(ROOT/'hardware'/name,build/name)
   shutil.copyfile(ROOT/'hardware/numerical_bridge_tb.cpp',build/'canonical_numerical_bridge_tb.cpp')
   shutil.copyfile(source/'gate_fixed_fp32_top.cpp',build/'gate_fixed_fp32_top.cpp')
   shutil.copyfile(ROOT/'scripts/build_fixed_fp32_hls.py',build/'build_fixed_fp32_hls.py')
   (build/'run_qualification_cases.py').write_bytes(driver_source)
   (build/'numerical_bridge_tb.cpp').write_text(generated)
   q=lambda p:'{'+str(p)+'}';flags='-std=c++14 -O2 -ffp-contract=off -I'+str(build)
   tcl=build/'qualify.tcl';tcl.write_text('\n'.join([
    'open_project '+q(build/'project'),'set_top gate_fixed_fp32_top',
    'add_files '+q(build/'gate_fixed_fp32_top.cpp')+' -cflags '+q(flags),
    'add_files -tb '+q(build/'numerical_bridge_tb.cpp')+' -cflags '+q(flags),
    'open_solution solution -flow_target '+config['hardware']['hls']['flow_target'],
    'set_part '+q(config['hardware']['part']),'create_clock -period 5',
    'config_compile -unsafe_math_optimizations=false','csim_design','exit'])+'\n')
   command=[config['hardware']['hls']['tool_executable'],'--mode','hls','--tcl',str(tcl)]
   record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),state='running',passed=False,bounds=True,
    command=command,synthesis_recorded_utc=synthesis['recorded_utc'],flags=flags,
    compiled_first_window=str(folder_for(first).relative_to(ROOT)),compiled_first_input_window=first['folder'],
    cpu_affinity=sorted(os.sched_getaffinity(0)),memory_guard_file=str(memory_guard.path.relative_to(ROOT)),new_synthesis=False,score_approval=None)
   save(compiledpath,record)
   try:
    rc,elapsed=run(command,build,env_for(first),build/'compile.log',1800,memory_guard)
    log=(build/'compile.log').read_text(errors='replace')
    record.update(state='returned',returncode=rc,elapsed_seconds=elapsed,passed=rc==0 and executable.is_file() and 'CSim done with 0 errors.' in log)
   except Exception as error:record.update(state='stopped',error=str(error));save(compiledpath,record);raise
   save(compiledpath,record)
   if not record['passed']:raise RuntimeError('Bounds compile/first actual window failed; preserve first diagnostic')
  record=validate_captured_tail_build(build,synthesis) if captured_policy else validate_bounds_build(build,synthesis)
  statepath=output/'qualification.json'
  state=read(statepath) if statepath.exists() else dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
   identity=exports['identity'],selection=selection,input_export=str((inputs/'export.json').relative_to(ROOT)),
   windows=[],complete=False,numerical_acceptance=None,bounds=True,
   evidence_kind='Fresh C simulation; independent same-input operator bounds plus checked signed integer contract; not full RTL')
  if state['identity']!=exports['identity'] or state['selection']!=selection or state['input_export']!=str((inputs/'export.json').relative_to(ROOT)) or state['bounds'] is not True:raise ValueError('Saved bounds association differs')
  if state['numerical_acceptance'] is not None:raise ValueError('A reviewed resolution cannot change global numerical acceptance')
  if state.get('state','').startswith('stopped_') and state['state']!='stopped_first_unexplained_discrepancy':raise RuntimeError('Operator/service/process failure cannot be resolved by an A8/terminal review')
  bykey={(v['case_index'],v['window_index']):v for v in state['windows']}
  if len(bykey)!=len(state['windows']) or not set(bykey)<=assigned:raise ValueError('Saved bounds membership differs')
  historical_keys=set(bykey)
  items={(v['case_index'],v['window_index']):v for v in exports['completed_windows']}
  requested_key=None
  if args.reviewed_resolution:
   requested=read(Path(args.reviewed_resolution) if Path(args.reviewed_resolution).is_absolute() else ROOT/args.reviewed_resolution)
   requested_key=requested.get('case_index'),requested.get('window_index')
   if requested_key not in bykey:raise ValueError('Only an already saved stopped discrepancy may be reviewed')
   if state.get('state')!='stopped_first_unexplained_discrepancy' and not bykey[requested_key].get('reviewed_resolution'):raise ValueError('Requested entry was not a stopped unexplained discrepancy')
  def check_window(item,entry):
   folder=folder_for(item);summary=read(folder/'bridge_summary.json')
   if entry['summary']!=summary or any(entry[k]!=item[k] for k in ('document_id','label')) or entry['folder']!=str(folder.relative_to(ROOT)) or entry['input_folder']!=item['folder']:raise ValueError('Bounds saved case association differs')
   expected_calls=[165,0,0,2,16,18,33,32,1] if args.floating else [165,72,72,2,16,18,33,32,1]
   expected_mode='ordinary_floating_reference' if args.floating else 'w8_a8_integer_contract'
   if not summary['completed'] or summary['case']!=item['document_id'] or summary['encoder_arithmetic']!=expected_mode or summary['service_calls_by_op']!=expected_calls:raise ValueError('Bounds actual mode/call completion differs')
   checked=validate_operator_bounds(folder,summary)
   if summary['same_input_quantization_exact']!=(None if args.floating else True) or summary['integer_accumulators_exact']!=(None if args.floating else True) or summary['same_input_codes_checked']!=checked['checked_values_by_op'][1] or summary['exact_accumulators_checked']!=checked['checked_values_by_op'][2]:raise ValueError('Bounds exact integer/quantization coverage differs')
   if entry['operator_bounds']!=checked:raise ValueError('Saved operator coverage differs')
   terminal(folder,'bridge',summary['logits'],summary['risk'])
   terminal(ROOT/item['folder'],'reference',summary['reference_logits'],summary['reference_risk'])
   checks=[]
   from validate_qualification_exports import LINEARS
   names=[f'layer{i}_{name}' for i in range(4) for name in LINEARS] if not args.floating else []
   if [v['name'] for v in summary['integer_layers']]!=names:raise ValueError('Bounds actual integer membership differs')
   for layer in summary['integer_layers']:
    name=layer['name'];actual=(folder/('actual_'+name+'_a8.bin')).read_bytes();reference=(ROOT/item['folder']/(name+'_reference_a8.bin')).read_bytes()
    size=256*LINEARS[name.split('_',1)[1]][0]
    if len(actual)!=len(reference) or len(actual)!=size or layer['input_codes']!=size or layer['accumulators_checked']!=256*LINEARS[name.split('_',1)[1]][1]:raise ValueError('Bounds actual A8/accumulator extent differs')
    differences=sum(a!=b for a,b in zip(actual,reference));checks.append(dict(name=name,values=size,differences=differences))
    if differences!=layer['original_software_code_differences']:raise ValueError('Bounds actual A8 discrepancy record differs')
   logits_equal=(folder/'bridge_logits.bin').read_bytes()==(ROOT/item['folder']/'reference_logits.bin').read_bytes()
   risk_equal=(folder/'bridge_risk.bin').read_bytes()==(ROOT/item['folder']/'reference_risk.bin').read_bytes()
   if entry['a8']!=checks or entry['logits_equal']!=logits_equal or entry['risk_equal']!=risk_equal:raise ValueError('Bounds saved actual terminal/A8 checks differ')
   if entry['returncode']:raise RuntimeError('Service/process return failure cannot be resolved')
   if entry['acceptance'] is not None:raise ValueError('Reviewed entry acceptance must remain null')
   discrepancy=any(v['differences'] for v in checks) or not logits_equal or not risk_equal
   stored=entry.get('reviewed_resolution')
   if stored:
    validated=validate_reviewed_resolution(stored['path'],entry,exports['identity'],mode)
    if validated!=stored:raise ValueError('Saved coordinator resolution changed')
   if requested_key==(entry['case_index'],entry['window_index']):
    validated=validate_reviewed_resolution(args.reviewed_resolution,entry,exports['identity'],mode)
    if stored and stored!=validated:raise ValueError('Do not replace an existing saved review')
    entry['reviewed_resolution']=validated;stored=validated
   recognized=entry.get('recognized_floating_propagation');known_w8=entry.get('recognized_w8_propagation');known_captured=entry.get('recognized_w8_captured_tail')
   if recognized:recognized_floating_entry(entry,exports['identity'],selection,inputs/'parameters')
   if known_w8:recognized_w8_entry(entry,exports['identity'],selection,inputs/'parameters')
   if known_captured:recognized_w8_captured_tail_entry(entry,exports['identity'],selection,inputs/'parameters')
   if entry.get('compiled_build')==TAIL_BUILD and (stored or not discrepancy):
    # Exact A8+terminal outcomes and casewise reviews keep actual capture provenance.
    validate_captured_tail_build(ROOT/TAIL_BUILD,synthesis)
    from validate_qualification_exports import _same_path
    observed=read(folder/'csim_run.json')
    if observed.get('state')!='returned' or observed.get('returncode')!=0 or observed.get('identity')!=exports['identity'] or tuple(observed.get('window',()))!=(entry['case_index'],entry['window_index']) or observed.get('bounds') is not True or observed.get('compiled_build')!=TAIL_BUILD or len(observed.get('command',[]))!=1 or not _same_path(observed['command'][0],ROOT/TAIL_BUILD/'project/solution/csim/build/csim.exe',ROOT) or not (output/observed['source']).is_file():raise ValueError('Reviewed capture entry lacks its actual source/build/process association')
    for captured in ('diagnostic_pooler_pre_tanh.bin','diagnostic_pooler_post_tanh.bin'):
     raw=(folder/captured).read_bytes()
     if len(raw)!=256*4 or not all(math.isfinite(v) for v in struct.unpack('<256f',raw)):raise ValueError('Reviewed actual capture extent/finite values differ')
   if entry.get('compiled_build')==TAIL_BUILD and discrepancy and not known_captured and not stored:
    key=entry['case_index'],entry['window_index']
    if captured_policy is None or key in historical_keys:raise RuntimeError('An unclassified historical capture entry requires explicit review; never adopt diagnostic captures as original')
    entry['recognized_w8_captured_tail']=recognized_w8_captured_tail_entry(entry,exports['identity'],selection,inputs/'parameters',policy=captured_policy,write=True);known_captured=entry['recognized_w8_captured_tail']
   if discrepancy and not stored and not recognized and not known_w8 and not known_captured:
    key=entry['case_index'],entry['window_index']
    if prospective_policy is not None and key not in historical_keys:
     entry['recognized_floating_propagation']=recognized_floating_entry(entry,exports['identity'],selection,inputs/'parameters',policy=prospective_policy,write=True)
    elif w8_policy is not None and key not in historical_keys:
     entry['recognized_w8_propagation']=recognized_w8_entry(entry,exports['identity'],selection,inputs/'parameters',policy=w8_policy,write=True)
    else:raise RuntimeError('Bounds unexplained A8/terminal discrepancy requires explicit coordinator review')
  for key,entry in bykey.items():check_window(items[key],entry)
  if state.get('state')=='stopped_first_unexplained_discrepancy' and not args.reviewed_resolution:raise RuntimeError('Saved discrepancy requires an explicit reviewed resolution')
  if args.reviewed_resolution:
   state['state']='reviewed_resolution_handoff';state['active_window']=None;save(statepath,state)
  if state['complete'] and set(bykey)!=assigned:raise ValueError('Complete bounds qualification lacks assigned windows')
  state['last_dispatch']=dict(source=dispatch.name,mode=mode,max_new_windows=args.max_new_windows,reviewed_resolution=args.reviewed_resolution,recognized_floating_policy=args.recognized_floating_policy,recognized_w8_policy=args.recognized_w8_policy,recognized_w8_captured_tail_policy=args.recognized_w8_captured_tail_policy,compiled_build=build.relative_to(ROOT).as_posix(),memory_guard_file=str(memory_guard.path.relative_to(ROOT)) if memory_guard.active else None,reference_association=reference_association,cpu_affinity=sorted(os.sched_getaffinity(0)));save(statepath,state)
  count=0
  for item in exports['completed_windows']:
   key=item['case_index'],item['window_index']
   if key in bykey:continue
   if count>=args.max_new_windows:break
   paused();folder=folder_for(item);summarypath=folder/'bridge_summary.json';state['active_window']=list(key);save(statepath,state)
   if folder.exists() and any(folder.iterdir()):
    if str(folder.relative_to(ROOT))!=record['compiled_first_window'] or not summarypath.exists():raise RuntimeError('Uncommitted actual bounds output requires inspection; never overwrite')
    rc=0;elapsed=None
   else:
    folder.mkdir(parents=True,exist_ok=True)
    observation=dict(state='running',command=[str(executable)],window=key,source=dispatch.name,identity=exports['identity'],cpu_affinity=sorted(os.sched_getaffinity(0)),board_access=False,bounds=True,memory_guard_file=str(memory_guard.path.relative_to(ROOT)))
    if captured_policy:observation['compiled_build']=TAIL_BUILD
    save(folder/'csim_run.json',observation)
    try:
     rc,elapsed=run([str(executable)],build,env_for(item),folder/'csim.log',1800,memory_guard)
     observation.update(state='returned',returncode=rc,elapsed_seconds=elapsed);save(folder/'csim_run.json',observation)
    except Exception as error:
     observation.update(state='stopped',error=str(error));save(folder/'csim_run.json',observation)
     state['state']='stopped_process_failure';save(statepath,state);raise
   if not summarypath.exists():
    state['state']='stopped_first_operator_or_service_failure';save(statepath,state)
    raise RuntimeError('No completed bounds summary; inspect preserved operator_bounds.jsonl and C log')
   summary=read(summarypath)
   entry=dict(case_index=key[0],window_index=key[1],document_id=item['document_id'],label=item['label'],returncode=rc,elapsed_seconds=elapsed,
    folder=str(folder.relative_to(ROOT)),input_folder=item['folder'],summary=summary,a8=[],acceptance=None,memory_guard_file=str(memory_guard.path.relative_to(ROOT)),
    operator_bounds=validate_operator_bounds(folder,summary),
    logits_equal=(folder/'bridge_logits.bin').read_bytes()==(ROOT/item['folder']/'reference_logits.bin').read_bytes(),
    risk_equal=(folder/'bridge_risk.bin').read_bytes()==(ROOT/item['folder']/'reference_risk.bin').read_bytes())
   if captured_policy:entry['compiled_build']=TAIL_BUILD
   for layer in summary['integer_layers']:
    name=layer['name'];actual=(folder/('actual_'+name+'_a8.bin')).read_bytes();reference=(ROOT/item['folder']/(name+'_reference_a8.bin')).read_bytes()
    entry['a8'].append(dict(name=name,values=len(actual),differences=sum(a!=b for a,b in zip(actual,reference))))
   state['windows'].append(entry);save(statepath,state)
   try:check_window(item,entry)
   except Exception:
    state['state']='stopped_first_unexplained_discrepancy';save(statepath,state);raise
   bykey[key]=entry;count+=1;state['active_window']=None;save(statepath,state)
   print(json.dumps(dict(case=key,operator_calls=entry['operator_bounds']['calls'],a8_differences=0,logits_equal=entry['logits_equal'],risk_equal=entry['risk_equal'],acceptance=None)),flush=True)
  state['complete']=set(bykey)==assigned
  if state['complete']:
   threshold=struct.unpack('<f',(inputs/'parameters/threshold.bin').read_bytes())[0];documents=[]
   for case in selection['cases']:
    windows=[bykey[case['case_index'],wi] for wi in range(case['window_count'])]
    actual=max(v['summary']['risk'] for v in windows);reference=max(v['summary']['reference_risk'] for v in windows)
    documents.append(dict(document_id=case['document_id'],label=case['label'],windows=len(windows),actual_max=actual,reference_max=reference,
     threshold=threshold,actual_reject=actual>threshold,reference_reject=reference>threshold,decision_agrees=(actual>threshold)==(reference>threshold)))
   state['documents']=documents
  state['state']='complete_pending_acceptance' if state['complete'] else 'bounded_handoff';save(statepath,state)

if __name__=='__main__':main()
