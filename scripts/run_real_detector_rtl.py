"""Validate connected saved-W8 RTL captures and diagnose the next real prefix stage.

Default previews the next numerical work without writing or launching tools.
--check-connected inspects the actual shared arena at a declared service stop.
--execute-next runs only rows 0..15 of layer0-query integer arithmetic, using
actual completed prefix A8 bytes. It compiles a capture-only harness against
unchanged static64 RTL. --replay links the complete saved L256 arithmetic path
with resumable bounded row partitions; --preview-replay is read-only.
--prepare-probes reconstructs small inputs from the completed frozen C case;
--probe executes one named diagnostic through the unchanged FP RTL snapshot.
--max-services counts new required partitions (a singleton uses two identical
testbench slots). Python movement and handoffs are numerical orchestration,
not connected-control timing. No synthesis, inference, dataset or board access.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import json
import math
import re
import shutil
import struct
import run_fixed_fp32_rtl as rtl
from run_real_prefix_rtl import compare_codes, compare_float, floats, need

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/goal4/numerical_bridge'
PREFIX = DATA / 'rtl_prefix_v1'
REFERENCE = DATA / 'ordered_model_v1'
PLAN = ROOT / 'results/goal4/internal_integration/command_plan.json'
PREPARED = ROOT / 'results/goal4/internal_integration/arena_w8_qat_real_input_v1'
POLICY = {
 'identity': 'connected_saved_w8_diagnostic_v1',
 'integer': 'exact signed A8 x W8 accumulator bytes with int32 range check',
 'quantization': 'exact nearest-even/clipped codes for identical input and frozen FP32 scale',
 'fp_local_bounds': {'DOT':[3e-5,3e-5], 'LAYER_NORM':[2e-4,2e-4],
   'SOFTMAX':[3e-6,3e-5], 'GELU':[3e-6,3e-5], 'TANH':[3e-6,3e-5], 'other':[0,0]},
 'softmax_row_sum_absolute_bound': 2e-5,
 'propagation': 'pause diagnosis at the first differing saved ordered A8 code; retain input/quotient/scale; local FP propagation is not automatically a detector acceptance failure',
 'terminal': 'report logits/risk bits and errors, document max and strict threshold; detector score acceptance remains null',
 'qualification': 'bounded diagnostic; no universal terminal-bit rule or score-identity approval',
 'fp_reference': 'separate FP32-rounded host arithmetic; host math transcendentals are local references under existing bounds'
}

def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def pack(values, kind='f'): return struct.pack('<'+kind*len(values),*values)
def f32(value):
 try: return struct.unpack('<f',struct.pack('<f',value))[0]
 except OverflowError: return math.copysign(math.inf,value)
def hex_write(path,raw): path.write_text(''.join(f'{v:02x}\n' for v in raw))
def capture(path,count):
 words=path.read_text().split()
 need(len(words)==count and all(re.fullmatch('[0-9a-fA-F]{2}',v) for v in words),'Wrong/unknown capture bytes: '+str(path))
 return bytes(int(v,16) for v in words)
def signed(raw): return struct.unpack('<'+'b'*len(raw),raw)
def clip_a8(quotient):
 need(not math.isnan(quotient),'NaN quantization quotient')
 return 127 if quotient>=127 else -128 if quotient<=-128 else round(quotient)

def integer_reference(activations,weights,rows,inner,outputs):
 """Independent unbounded signed sum; output-major weights; no float product."""
 a,w=signed(activations),signed(weights); result=[]
 need(len(a)==rows*inner and len(w)==outputs*inner,'Integer input extent differs')
 for r in range(rows):
  row=a[r*inner:(r+1)*inner]
  for o in range(outputs):
   value=sum(v*code for v,code in zip(row,w[o*inner:(o+1)*inner]))
   need(-(1<<31)<=value<(1<<31),'Accumulator overflow')
   result.append(value)
 return pack(result,'i')

def fp_reference(arguments,arrays,scale):
 """Same-input operation reference, retaining every rounded basic FP32 step."""
 a=arguments;op,rows,width,outputs=[a[k] for k in ('OP','ROWS','WIDTH','OUTPUTS')]
 x,y,z=[floats(arrays.get(p,b'\0'*4)) for p in ('x','y','z')]
 add=lambda v,w:f32(v+w)
 mul=lambda v,w:f32(v*w)
 div=lambda v,w:f32(v/w)
 if op==1:
  return bytes(clip_a8(div(v,scale))&255 for v in x)
 if op==2:
  acc=struct.unpack('<'+'i'*(len(arrays['accumulators'])//4),arrays['accumulators'])
  return pack([add(mul(f32(v),mul(scale,y[i%width])),z[i%width]) for i,v in enumerate(acc)])
 if op==3:return pack([add(add(v,y[i]),z[i]) for i,v in enumerate(x)])
 if op==4:return pack([add(v,y[i]) for i,v in enumerate(x)])
 if op==7:return pack([mul(mul(v,.5),add(1.,f32(math.erf(div(v,f32(math.sqrt(2))))))) for v in x])
 if op==8:return pack([f32(math.tanh(v)) for v in x])
 result=[]
 for r in range(rows):
  row=x[r*width:(r+1)*width]
  if op==0:
   for o in range(outputs):
    total=0.
    for k,value in enumerate(row):total=add(total,mul(value,y[o*width+k]))
    result.append(add(total,z[o]) if a['DOT_BIAS'] else total)
  elif op==5:
   total=0.
   for value in row:total=add(total,value)
   mean=div(total,256.);total=0.
   for value in row:
    centered=add(value,-mean);total=add(total,mul(centered,centered))
   denominator=f32(math.sqrt(add(div(total,256.),f32(1e-12))))
   result.extend(add(mul(div(add(value,-mean),denominator),y[k]),z[k]) for k,value in enumerate(row))
  elif op==6:
   scores=[add(div(value,scale),y[r*width+k]) for k,value in enumerate(row)]
   maximum=max(scores);exponents=[f32(math.exp(add(value,-maximum))) for value in scores];total=0.
   for value in exponents:total=add(total,value)
   result.extend(div(value,total) for value in exponents)
  else:raise ValueError('Unsupported FP op')
 return pack(result)

def associations():
 config=read(ROOT/'configs/project.json');plan=read(PLAN);prep=read(PREPARED/'preparation.json')
 prefix=read(PREFIX/'run.json');manifest=read(DATA/'manifest.json');reference=read(REFERENCE/'summary.json')
 source,synthesis=rtl.source_evidence(config)
 need(prefix['passed'] and prefix['executed_calls']==6 and prefix['source_association_checked'],'Completed prefix required')
 need(prefix['synthesis_recorded_utc']==synthesis['recorded_utc']==plan['fp32_build']['synthesis_recorded_utc'],'Prefix/FP plan synthesis differs')
 need(prep['complete'] and prep['plan_recorded_utc']==plan['recorded_utc'],'Arena/plan association differs')
 need(prep['case']==prefix['case']==manifest['case']==reference['case'],'Saved input differs')
 need(prep['config_w8_mask']==65535 and manifest['window_count']==1 and reference['validation_device']=='cpu','Require saved CPU W8 single-window case')
 need(prep['service_revision']==synthesis['service_schedule_revision']=='panel_cache_batched_attention_v1','FP revision differs')
 integer=ROOT/'build/linear_hls_cached_static_bank/lanes64';integer_run=read(integer/'run.json')
 need(integer_run['hls_synthesis_performed'] and integer_run['returncode']==0 and integer_run['source_revision']=='static_bank_index_v1','Successful static64 source required')
 identity=next(v for v in plan['integer_builds'] if v['profile']=='lanes64')
 need(identity['synthesis_recorded_utc']==integer_run['recorded_utc'],'Integer plan association differs')
 return config,plan,prep,prefix,source,synthesis,integer,integer_run

def codes_comparison(actual,expected,inputs,scale,start=0):
 result=compare_codes(actual,expected);examples=[]
 for i in result['first_failure_indices']:
  q=f32(floats(inputs)[i]/scale);boundary=math.floor(q)+.5
  examples.append(dict(index=start+i,input=floats(inputs)[i],scale=scale,quotient=q,
    saved_code=signed(expected[i:i+1])[0],actual_code=signed(actual[i:i+1])[0],
    nearest_half_integer=boundary,distance_to_half=q-boundary))
 result['first_code_difference_details']=examples
 return result

def connected_capture(folder):
 """Accept direct execute and fresh run-only records without reading build output."""
 record_folder=folder if (folder/'run.json').is_file() else folder.parent
 run=read(record_folder/'run.json')
 need(run['passed'] and run['state']=='returned','Successful stopped/complete actual RTL run required')
 compiled=run.get('compiled_record',run)
 need(compiled.get('board_access') is False and 'actual unchanged synthesized' in compiled['arithmetic'],
      'Actual connected arithmetic/no-board scope differs')
 if 'compiled_record' in run:
  need(compiled['state']=='compiled' and compiled['passed'],'Successful compiled snapshot association required')
 observations=run.get('observations')
 if observations is not None:
  need(len(observations)==1,'One actual connected completion required');observation=observations[0]
 else:observation=run['observation']
 capture_folder=record_folder/'capture' if (record_folder/'capture').is_dir() else record_folder
 return run,compiled,observation,capture_folder,record_folder

def check_connected(folder,output):
 config,plan,prep,prefix,source,synthesis,integer,integer_run=associations()
 run,compiled,observation,folder,record_folder=connected_capture(folder);raw=(folder/'arena_after.bin').read_bytes()
 need(compiled['source_plan']==plan['identity'],'Connected plan differs')
 need(len(raw)==67108864,'Connected arena extent differs')
 count=observation.get('completed_services',435 if run['complete_detector'] else 0)
 services=[c for c in plan['commands'] if c['kind']=='service']
 need(1<=count<=435,'Unknown completed service count')
 def buffer(name,size=None):
  b=plan['buffers'][name];return raw[b['offset']:b['offset']+(size or b['bytes'])]
 record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),policy=POLICY,case=prep['case'],
  connected_record=str(record_folder/'run.json'),completed_services=count,checks=[],detector_score_acceptance=None)
 # Hidden remains the completed embedding LN until later layers overwrite it.
 if 4<=count<=9:
  record['checks'].append(dict(boundary='embedding_ln',comparison=compare_float(buffer('hidden'),(REFERENCE/'model_embedding_ln.bin').read_bytes(),2e-4,2e-4)))
 # The first matrix's live A8/accumulator buffers remain until the next matrix.
 if 6<=count<=9:
  actual=buffer('a8',65536);expected=(REFERENCE/'model_layer0_query_a8.bin').read_bytes()
  params=floats((DATA/'layer0_query_params.bin').read_bytes());hidden=buffer('hidden')
  expected_same=bytes(clip_a8(f32(v/params[0]))&255 for v in floats(hidden))
  record['checks'].append(dict(boundary='layer0_query_a8_same_input',comparison=compare_codes(actual,expected_same)))
  record['checks'].append(dict(boundary='layer0_query_a8_saved_ordered',comparison=codes_comparison(actual,expected,hidden,params[0])))
  if count>=7:
   weights=(DATA/'layer0_query_w8.bin').read_bytes();acc=buffer('accumulator',65536*4)
   expected_acc=integer_reference(actual,weights,256,256,256)
   record['checks'].append(dict(boundary='layer0_query_accumulator',comparison=compare_codes(acc,expected_acc)))
   if count==9:
    a=dict(OP=2,ROWS=256,WIDTH=256,OUTPUTS=0,DOT_BIAS=0)
    expected_query=fp_reference(a,dict(accumulators=acc,y=pack(params[1:257]),z=pack(params[257:])),params[0])
    record['checks'].append(dict(boundary='layer0_query_rescale_same_input',comparison=compare_float(buffer('query'),expected_query,0,0)))
 if count==435 and run['complete_detector']:
  logits,risk=buffer('logits'),buffer('class_probabilities')[4:8]
  expected_logits=(REFERENCE/'model_logits.bin').read_bytes();expected_risk=(REFERENCE/'model_risk.bin').read_bytes()
  threshold_raw=(DATA/'threshold.bin').read_bytes();threshold=floats(threshold_raw)[0];value=floats(risk)[0]
  need(threshold_raw==bytes.fromhex(prep['threshold_bits_hex'])[::-1],'Threshold bits differ')
  need(math.isfinite(value) and 0<=value<=1,'Invalid risk')
  record['terminal']=dict(logits=list(floats(logits)),risk=value,logits_bits=logits.hex(),risk_bits=risk.hex(),
   logits_bit_equal=logits==expected_logits,risk_bit_equal=risk==expected_risk,
   logits_absolute_errors=[abs(v-e) for v,e in zip(floats(logits),floats(expected_logits))],
   risk_absolute_error=abs(value-floats(expected_risk)[0]),threshold_bits=threshold_raw.hex(),
   document_window_count=1,document_max_risk=value,strict_reject=value>threshold,
   saved_strict_reject=floats(expected_risk)[0]>threshold,
   decision_agrees=(value>threshold)==(floats(expected_risk)[0]>threshold),detector_score_acceptance=None)
 record['observed_checks_passed']=all(v['comparison']['passed'] for v in record['checks'])
 record['full_boundary_coverage']=False
 record['coverage_limitation']='Single final arena cannot retain overwritten A8/accumulator/FP traces; use per-service captures for full qualification'
 need(record['checks'] or 'terminal' in record,'This stop point has no implemented numerical boundary check')
 if output:
  need(not output.exists(),'Preserve existing numerical check evidence')
  output.parent.mkdir(parents=True,exist_ok=True);rtl.save(output,record)
 print(json.dumps(record,indent=2))
 need(record['observed_checks_passed'],'Earliest unexplained connected arithmetic discrepancy; see saved check')

def execute_next(args):
 config,plan,prep,prefix,source,synthesis,integer,integer_run=associations();rtl.paused()
 out=args.output.resolve();need(out.is_relative_to(ROOT/'results/goal4'),'Output must stay within results/goal4')
 need(not out.exists(),'Preserve existing/failed evidence; select a fresh output directory');out.mkdir()
 rows=args.rows;need(rows in (1,4,16),'Bounded first matrix rows must be 1,4,16')
 a=capture(PREFIX/'layer0_query_a8/actual_0.hex',32768)[:rows*256]
 w=(DATA/'layer0_query_w8.bin').read_bytes();need(len(w)==65536,'Query weight extent differs')
 expected=integer_reference(a,w,rows,256,256)
 hex_write(out/'activations.hex',a);hex_write(out/'weights.hex',w);(out/'expected_accumulators.bin').write_bytes(expected)
 record=dict(state='preparing',passed=False,case=prep['case'],recorded_utc=datetime.now(timezone.utc).isoformat(),
  policy=POLICY,scope='Actual static64 layer0 query rows0..'+str(rows-1)+' using completed prefix A8; not full detector',
  fp_synthesis_recorded_utc=synthesis['recorded_utc'],integer_synthesis_recorded_utc=integer_run['recorded_utc'],
  prefix_reused=str(PREFIX/'run.json'),rows=rows,integer_input_from_actual_prefix=True,new_synthesis=False,
  detector_score_acceptance=None,memory_model='Three independent bounded byte-array ports; registered responses; no added DDR delay')
 rtl.save(out/'run.json',record)
 shutil.copyfile(Path(__file__),out/'run_real_detector_rtl.py')
 sim=out/'sim';sim.mkdir();shutil.copytree(integer/'project/solution/syn/verilog',sim/'rtl')
 shutil.copyfile(ROOT/'hardware/real_linear_axi_tb.sv',sim/'real_linear_axi_tb.sv')
 files=sorted((sim/'rtl').glob('*.v'))+[sim/'real_linear_axi_tb.sv']
 (sim/'rtl.prj').write_text('\n'.join(('sv' if p.suffix=='.sv' else 'verilog')+' xil_defaultlib "'+str(p)+'"' for p in files)+'\n')
 tools=Path(config['hardware']['hls']['vivado_executable']).parent
 compile_command=[str(tools/'xelab'),'xil_defaultlib.real_linear_axi_tb','-prj',str(sim/'rtl.prj'),'-L','unisims_ver','-L','xpm','-relax','-s','real_linear_axi_tb']
 (sim/'run.tcl').write_text('run all\nquit\n')
 values=dict(ROWS=rows,INNER=256,OUTPUTS=256,BITS=8,REPS=2,CLOCK_NS=synthesis['clock_ns'],TIMEOUT_CYCLES=10000000,CASE_DIR=str(out))
 command=[str(tools/'xsim'),'real_linear_axi_tb','-tclbatch',str(sim/'run.tcl')]
 for k,v in values.items():command+=['-testplusarg',str(k)+'='+str(v)]
 record.update(compile_command=compile_command,command=command);rtl.save(out/'run.json',record)
 try:
  rc=rtl.command_run(compile_command,sim,out/'compile.log',timeout=900);need(rc==0,'Integer capture compile failed')
  record['state']='running';rtl.save(out/'run.json',record)
  need(read(integer/'run.json')==integer_run,'Synthesis changed before replay')
  rc=rtl.command_run(command,sim,out/'simulation.log',timeout=900)
  log=(out/'simulation.log').read_text(errors='replace')
  tx=[json.loads(v.split(' ',1)[1]) for v in log.splitlines() if v.startswith('GATE_RTL_TRANSACTION ')]
  final=[json.loads(v.split(' ',1)[1]) for v in log.splitlines() if v.startswith('GATE_RTL_VERIFIED ')]
  record.update(returncode=rc,transactions=tx,verified=final[0] if len(final)==1 else None)
  need(rc==0 and 'GATE_RTL_FAIL' not in log and len(tx)==2 and len(final)==1,'Capture status/count failed')
  need(final[0]['passed'] and all(final[0].get(k)==v for k,v in dict(rows=rows,inner=256,outputs=256,bits=8,repetitions=2,clock_period_ns=synthesis['clock_ns']).items()),'Capture definition differs')
  need([v['index'] for v in tx]==[0,1] and all(v.get('output_captured') and v['latency_cycles']>0 and v['latency_cycles']==v['done_cycle']-v['start_cycle'] for v in tx),'Transaction boundary differs')
  comparisons=[]
  for slot in range(2):
   actual=capture(out/f'actual_{slot}.hex',len(expected));(out/f'actual_{slot}.bin').write_bytes(actual)
   comparisons.append(compare_codes(actual,expected))
  record.update(state='returned',passed=all(v['passed'] for v in comparisons),comparisons=comparisons)
  rtl.save(out/'run.json',record);need(record['passed'],'Exact integer mismatch')
 except Exception as error:
  record.update(state='stopped',passed=False,error=str(error));rtl.save(out/'run.json',record);raise
 print(json.dumps(record,indent=2))

def approved_prefix16_plan(original):
 """Independent fixed transformation; no generic partial-plan acceptance."""
 selected=[copy.deepcopy(c) for c in original['commands'][:4]]
 names=['embedding_sum_rows0','embedding_ln_rows0','layer0_query_quantize_rows0','layer0_query','layer0_query_rescale_rows0']
 for name in names:
  matches=[c for c in original['commands'] if c['kind']=='service' and c['name']==name]
  need(len(matches)==1,'Original approved prefix service missing/ambiguous: '+name)
  selected.append(copy.deepcopy(matches[0]))
 for index,c in enumerate(selected):
  c['original_command_index']=c['index'];c['index']=index
  if c['kind']=='movement':
   c['rows']=16;c['useful_read_bytes']=c['useful_write_bytes']=16*c['columns']*4
   c['destination']['extent_bytes']=16*c['columns']*4
  elif c['kind']=='service':
   c['arguments']['ROWS']=16
   for write in c['register_writes']:
    if write['argument']=='ROWS':write['value']=16
 return selected

def trace_manifest_entries(plan):
 """Derive every required input/output byte extent independently of capture files."""
 entries=[]
 for c in plan['commands']:
  if c['kind']!='service':continue
  a=c['arguments'];integer=c['engine']=='integer';quant=not integer and a['OP']==1
  if integer:sizes={'ACTIVATIONS':a['ROWS']*a['INNER'],'WEIGHTS':a['INNER']*a['OUTPUTS']}
  else:
   rows,width,n,op=[a[k] for k in ('ROWS','WIDTH','OUTPUTS','OP')];sizes={'X':rows*width*4}
   if op==0:sizes.update(Y=n*width*4,**({'Z':n*4} if a['DOT_BIAS'] else {}))
   elif op==2:sizes={'ACCUMULATORS':rows*width*4,'Y':width*4,'Z':width*4}
   elif op==3:sizes.update(Y=rows*width*4,Z=rows*width*4)
   elif op in (4,6):sizes.update(Y=rows*width*4)
   elif op==5:sizes.update(Y=width*4,Z=width*4)
  index=len(entries)+1;inputs=[]
  for argument,size in sizes.items():
   ref=a[argument];ref=ref['alternatives']['8'] if 'select_by_frozen_precision_group' in ref else ref
   inputs.append(dict(argument=argument,offset=ref['arena_offset'],bytes=size,file=f'service_{index:04d}_{argument.lower()}.bin'))
  output=a['CODES' if quant else 'OUTPUT_R'];count=a['ROWS']*(a['OUTPUTS'] if integer or a['OP']==0 else a['WIDTH'])
  entries.append(dict(service_index=index,command_index=c['index'],name=c['name'],engine=c['engine'],op=a.get('OP'),arguments=a,inputs=inputs,offset=output['arena_offset'],bytes=count*(1 if quant else 4),file=f'service_{index:04d}.bin'))
 return entries

def trace_descriptor_rom(plan):
 """Reviewed fixed descriptor layout; verify the ROM actually used by this run."""
 records=[]
 for c in plan['commands']:
  w=[0]*16
  if c['kind']=='service':
   a=c['arguments']
   if c['engine']=='integer':w[:8]=[1,a['ACTIVATIONS']['arena_offset'],a['WEIGHTS']['arena_offset'],a['OUTPUT_R']['arena_offset'],a['ROWS'],a['INNER'],a['OUTPUTS'],a['WEIGHT_BITS']['frozen_precision_group']]
   else:
    y=a['Y'];group=y.get('select_by_frozen_precision_group',16);y4=y['alternatives']['4'] if group<16 else y;y8=y['alternatives']['8'] if group<16 else y
    w=[2,a['OP'],a['X']['arena_offset'],y4['arena_offset'],a['Z']['arena_offset'],a['ACCUMULATORS']['arena_offset'],a['OUTPUT_R']['arena_offset'],a['CODES']['arena_offset'],a['ROWS'],a['WIDTH'],a['OUTPUTS'],a['SCALE'].get('f32_bits',0),a['SCALE'].get('local_input_scale_index',255),a['DOT_BIAS'],group,y8['arena_offset']]
  elif c['kind']=='movement':
   op={'rectangle':0,'indexed_embedding_rows':1,'transpose_f32_tiles16':2,'binary_mask_to_fp32_repeated_rows':3}[c['mode']]
   w[:10]=[3,op,c['source']['arena_offset'] if isinstance(c['source'],dict) else 0,c['destination']['arena_offset'],c['rows'],c['columns'],c.get('source_stride_bytes') or 0,c.get('destination_stride_bytes') or c['columns']*4,c.get('index_limit',0),int(c.get('index_source')=='local_type_ids')]
  elif c['name']=='read_risk_and_update_document':w[:2]=[4,c['source']['arena_offset']]
  records.append(''.join(f'{v:08x}' for v in reversed(w)))
 return records

def bind_trace_plan(original,compiled,run,folder,record_folder,diagnostic_path):
 """Reduced scope must be requested explicitly and be the reviewed exact schedule."""
 if diagnostic_path is None:
  need(not compiled.get('diagnostic') and not run.get('diagnostic'),'Reduced diagnostic requires explicit --diagnostic-plan; cannot qualify original435 schedule')
  need(compiled['source_plan']==original['identity'],'Connected full trace plan differs')
  return original,435,None
 declared=read(diagnostic_path)
 need(declared['identity']==original['identity']+'_connected_prefix16_v1','Only approved prefix16 diagnostic is supported; seeded tails require a separate explicit binding')
 need(declared['commands']==approved_prefix16_plan(original),'Reduced command schedule differs from exact approved original-plan transformation')
 need({k:v for k,v in declared.items() if k not in ('identity','commands','status','diagnostic','counts','recorded_utc')}=={k:v for k,v in original.items() if k not in ('identity','commands','status','diagnostic','counts','recorded_utc')},'Reduced diagnostic changed buffers, parameters, source builds or original scientific metadata')
 d=declared['diagnostic']
 need(d['rows']==16 and d['original_plan_identity']==original['identity'] and d['integer_calls']==1 and d['fp32_calls']==4 and d['stop_after']==5 and d['complete_detector'] is False,'Reduced diagnostic scope/call declaration differs')
 need(declared['counts']==dict(commands=9,integer_calls=1,fp32_calls=4),'Reduced diagnostic count differs')
 need(compiled['source_plan']==declared['identity'] and compiled.get('diagnostic')==d and run.get('diagnostic')==d,'Compiled/run reduced-plan association differs')
 need(compiled.get('host_control')=='actual AXI-Lite mailbox configure/submit','Reduced diagnostic requires actual mailbox configuration/submission')
 need(run['stop_after']==5 and run['complete_detector'] is False,'Reduced prefix must stop after5 and cannot declare complete detector')
 need(read(folder/'diagnostic_plan.json')==declared,'Actual capture diagnostic differs from explicitly supplied reviewed plan')
 def local(path):
  value=str(path).replace('\\','/');prefix='/home/jp19126/Projects/SENTRY/'
  return ROOT/value[len(prefix):] if value.startswith(prefix) else Path(path)
 build=local(run.get('compiled_build',record_folder))
 need(read(build/'diagnostic_plan.json')==declared,'Compiled diagnostic source plan differs from actual capture')
 need(read(build/'run.json')==compiled,'Nested compiled snapshot record differs from preserved build')
 need(read(build/'service_manifest.json')==read(folder/'service_manifest.json'),'Frozen compiled/capture sparse manifest differs')
 expected_rom=trace_descriptor_rom(declared)
 need((build/'command_plan.mem').read_text().split()==expected_rom and (folder/'command_plan.mem').read_text().split()==expected_rom,'Compiled/capture descriptor ROM differs from approved reduced schedule')
 expected_roots=[ROOT/'build/linear_hls_cached_static_bank/lanes64/project/solution/syn/verilog',(ROOT/original['fp32_build']['control_source']).parent]
 need([local(p).resolve() for p in compiled['source_roots']]==[p.resolve() for p in expected_roots],'Compiled actual synthesized arithmetic source roots differ')
 commands=compiled['commands'];vlog=[c for c in commands if Path(c[0]).name=='xvlog']
 need(len(vlog)==1 and any(str(v).endswith('/gate_fixed_fp32_top.v') for v in vlog[0]) and any(str(v).endswith('/gate_linear_top.v') for v in vlog[0]) and not any('stub' in str(v) for v in vlog[0]),'Actual synthesized arithmetic compile association differs')
 for name in ('gate_host_mailbox.sv','gate_detector_sequencer.sv','gate_data_mover.sv','gate_detector_arithmetic_tb.sv'):
  need((build/name).is_file() and any(str(v).endswith('/'+name) for v in vlog[0]),'Preserved actual integration source absent from compile: '+name)
 need(local(compiled['prepared_arena']).resolve()==PREPARED.resolve(),'Compiled diagnostic prepared arena differs')
 return declared,5,dict(identity=declared['identity'],rows=16,scope=d['scope'],complete_detector=False,checked_schedule_only=True,mailbox_scope='actual configure/submit; prefix stops before terminal mailbox completion',detector_acceptance=None)

def check_trace(folder,output,diagnostic_path=None):
 """Check sparse complete service captures, including exact arena dataflow."""
 config,plan,prep,prefix,source,synthesis,integer,integer_run=associations()
 run,compiled,observation,folder,record_folder=connected_capture(folder)
 plan,total,diagnostic=bind_trace_plan(plan,compiled,run,folder,record_folder,diagnostic_path)
 manifest=read(folder/'service_manifest.json')
 need(manifest['plan_identity']==compiled['source_plan']==plan['identity'],'Connected trace plan differs')
 count=observation.get('completed_services',total if run['complete_detector'] else 0)
 need(1<=count<=total and (diagnostic is None or count==5),'Invalid completed service count')
 services=[c for c in plan['commands'] if c['kind']=='service']
 need(len(services)==total and manifest['captures']==trace_manifest_entries(plan),'Sparse descriptor/input/output declaration differs from exact source plan')
 arena=bytearray((PREPARED/'arena.bin').read_bytes());need(len(arena)==67108864,'Prepared arena extent differs')
 def location(ref):return ref['alternatives']['8'] if 'select_by_frozen_precision_group' in ref else ref
 def take(ref,size=None):
  ref=location(ref);n=ref['extent_bytes'] if size is None else size;b=plan['buffers'][ref['buffer']]
  need(ref['arena_offset']==b['offset']+ref['buffer_offset'] and ref['buffer_offset']+n<=b['bytes'],'Invalid trace reference extent')
  return bytes(arena[ref['arena_offset']:ref['arena_offset']+n])
 def put(ref,raw):
  ref=location(ref);b=plan['buffers'][ref['buffer']]
  need(ref['arena_offset']==b['offset']+ref['buffer_offset'] and ref['buffer_offset']+len(raw)<=b['bytes'],'Invalid trace output extent')
  arena[ref['arena_offset']:ref['arena_offset']+len(raw)]=raw
 def move(c):
  rows,cols=c['rows'],c['columns'];mode=c['mode']
  if mode=='binary_mask_to_fp32_repeated_rows':
   b=plan['buffers']['attention_mask'];mask=arena[b['offset']:b['offset']+b['bytes']]
   need(all(v in (0,1) for v in mask),'Invalid mask byte')
   put(c['destination'],b''.join(struct.pack('<I',0x80000000 if v else 0xff7fffff) for v in mask)*rows);return
  src=take(c['source']);target=bytearray(take(c['destination']));ss=c['source_stride_bytes'];ds=c['destination_stride_bytes']
  if mode=='indexed_embedding_rows':
   b=plan['buffers']['token_ids' if c['index_source']=='local_token_ids' else 'type_ids']
   indices=struct.unpack('<'+'I'*rows,arena[b['offset']:b['offset']+rows*4]);need(all(i<c['index_limit'] for i in indices),'Invalid gather ID')
  else:indices=range(rows)
  for row,source_row in enumerate(indices):
   if mode=='transpose_f32_tiles16':
    for col in range(cols):target[col*ds+row*4:col*ds+row*4+4]=src[source_row*ss+col*4:source_row*ss+col*4+4]
   else:target[row*ds:row*ds+cols*4]=src[source_row*ss:source_row*ss+cols*4]
  put(c['destination'],target)
 def scale_value(a):
  v=a['SCALE']
  if 'f32_bits' in v:return struct.unpack('<f',struct.pack('<I',v['f32_bits']))[0]
  b=plan['buffers']['input_scales'];off=b['offset']+4*v['local_input_scale_index'];return floats(arena[off:off+4])[0]
 record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),policy=POLICY,case=prep['case'],
  connected_record=str(record_folder/'run.json'),completed_services=count,checks=[],completed=False,
  full_boundary_coverage=False,detector_score_acceptance=None,diagnostic=diagnostic,
  full_detector_completed=False,source_plan=plan['identity'],approved_service_count=total)
 if output:need(not output.exists(),'Preserve existing numerical trace check');output.parent.mkdir(parents=True,exist_ok=True)
 def save():
  if output:rtl.save(output,record)
 index=0
 try:
  for c in plan['commands']:
   if index>=count:break
   if c['kind']=='movement':move(c);continue
   if c['kind']!='service':continue
   e=manifest['captures'][index];a=c['arguments'];arrays={}
   item=dict(service_index=index+1,name=c['name'],engine=c['engine'],input_dataflow_exact=True)
   record['checks'].append(item);save()
   for inp in e['inputs']:
    raw=(folder/inp['file']).read_bytes()
    need(len(raw)==inp['bytes'],'Captured input extent differs: '+inp['file'])
    reference=take(a[inp['argument']],inp['bytes'])
    need(inp['offset']==location(a[inp['argument']])['arena_offset'] and raw==reference,'Captured input differs from preceding actual outputs/movement: '+inp['file'])
    arrays[inp['argument'].lower()]=raw
   actual=(folder/e['file']).read_bytes();need(len(actual)==e['bytes'],'Captured output extent differs')
   if c['engine']=='integer':
    need(e['bytes']==a['ROWS']*a['OUTPUTS']*4,'Integer output extent differs')
    expected=integer_reference(arrays['activations'],arrays['weights'],a['ROWS'],a['INNER'],a['OUTPUTS'])
    comparison=compare_codes(actual,expected);comparison['semantic']='exact signed int32 accumulator bytes'
   else:
    scale=scale_value(a);expected=fp_reference(a,arrays,scale)
    if a['OP']==1:comparison=compare_codes(actual,expected)
    else:
     bounds=POLICY['fp_local_bounds'].get(c['operation'],POLICY['fp_local_bounds']['other'])
     comparison=compare_float(actual,expected,*bounds)
     if a['OP']==6:
      values=floats(actual);error=max(abs(math.fsum(values[r*a['WIDTH']:(r+1)*a['WIDTH']])-1) for r in range(a['ROWS']))
      comparison['maximum_row_sum_error']=error;comparison['passed'] &= error<=2e-5
   need(len(actual)==len(expected),'Same-input reference output extent differs')
   item['same_input_comparison']=comparison;save()
   need(comparison['passed'],'Same-input arithmetic failure at '+c['name'])
   if c['engine']=='fp32' and a['OP']==1:
    matrix=c['name'].split('_quantize_rows')[0];off=a['CODES']['buffer_offset']
    ref=(REFERENCE/('model_'+matrix+'_a8.bin')).read_bytes()[off:off+len(actual)]
    item['saved_ordered_a8']=codes_comparison(actual,ref,arrays['x'],scale,off);save()
    need(item['saved_ordered_a8']['passed'],'Propagation into saved ordered A8 codes at '+c['name'])
   output_ref=a['CODES'] if a.get('OP')==1 else a['OUTPUT_R']
   need(e['offset']==output_ref['arena_offset'],'Capture output offset differs')
   put(output_ref,actual)
   if c['name'].endswith('_output_ln_rows128'):
    layer=c['name'].split('_')[0];b=plan['buffers']['hidden'];values=bytes(arena[b['offset']:b['offset']+b['bytes']])
    item['saved_layer_error']=compare_float(values,(REFERENCE/f'model_{layer}.bin').read_bytes(),2e-4,2e-4)
    item['saved_layer_error']['local_ln_bounds_are_not_detector_acceptance']=True
   item['passed']=True;index+=1;save()
  final_arena=(folder/'arena_after.bin').read_bytes();need(bytes(arena)==final_arena,'Reconstructed actual arena differs from captured connected arena')
  record.update(completed=True,observed_checks_passed=True,full_boundary_coverage=diagnostic is None and count==435,
   diagnostic_boundary_coverage=diagnostic is not None and count==total,
   checked_a8_calls=sum(c['engine']=='fp32' and c['arguments']['OP']==1 for c in services[:count]),
   checked_integer_calls=sum(c['engine']=='integer' for c in services[:count]),
   checked_fp32_calls=sum(c['engine']=='fp32' for c in services[:count]))
  if diagnostic is None and count==435:
   record['full_detector_completed']=True
   final=(folder/'arena_after.bin').read_bytes();need(bytes(arena)==final,'Reconstructed actual arena differs from complete connected arena')
   b=plan['buffers']['logits'];logits=final[b['offset']:b['offset']+8]
   b=plan['buffers']['class_probabilities'];risk=final[b['offset']+4:b['offset']+8]
   expected_logits=(REFERENCE/'model_logits.bin').read_bytes();expected_risk=(REFERENCE/'model_risk.bin').read_bytes()
   threshold_raw=(DATA/'threshold.bin').read_bytes();need(threshold_raw==bytes.fromhex(prep['threshold_bits_hex'])[::-1],'Threshold bits differ')
   value=floats(risk)[0];threshold=floats(threshold_raw)[0];need(math.isfinite(value) and 0<=value<=1,'Invalid terminal risk')
   obs=observation;need(obs['risk_bits'].lower()==risk[::-1].hex() and bool(obs['reject'])==(value>threshold),'Connected result/strict threshold differs')
   record['terminal']=dict(logits=list(floats(logits)),risk=value,logits_bits=logits.hex(),risk_bits=risk.hex(),
    logits_bit_equal=logits==expected_logits,risk_bit_equal=risk==expected_risk,
    logits_absolute_errors=[abs(v-e) for v,e in zip(floats(logits),floats(expected_logits))],
    risk_absolute_error=abs(value-floats(expected_risk)[0]),threshold_bits=threshold_raw.hex(),
    document_window_count=1,document_max_risk=value,strict_reject=value>threshold,
    saved_strict_reject=floats(expected_risk)[0]>threshold,decision_agrees=(value>threshold)==(floats(expected_risk)[0]>threshold),
    detector_score_acceptance=None)
  save();print(json.dumps(record,indent=2))
 except Exception as error:
  record.update(completed=False,observed_checks_passed=False,error=str(error),checked_services=index);save();raise

class ArenaReplay:
 """Resumable numerical service replay; Python movement has no timing claim."""
 def __init__(self,args):
  self.args=args
  self.config,self.plan,self.prep,self.prefix,self.fp_source,self.fp_run,self.int_source,self.int_run=associations()
  self.out=args.output.resolve();need(self.out.is_relative_to(ROOT/'results/goal4'),'Replay output must stay within results/goal4')
  need(args.integer_rows in (1,4,16,32,64,128) and 1<=args.fp_rows<=128,'Invalid row partition bounds')
  self.initial=(PREPARED/'arena.bin').read_bytes();need(len(self.initial)==67108864,'Initial arena extent differs')
  self.arena=bytearray(self.initial)
  self.tools=Path(self.config['hardware']['hls']['vivado_executable']).parent
  self.identity=dict(case=self.prep['case'],checkpoint=self.prep['checkpoint'],
   plan_recorded_utc=self.plan['recorded_utc'],fp_synthesis=self.fp_run['recorded_utc'],
   integer_synthesis=self.int_run['recorded_utc'],policy=POLICY['identity'],
   fp_rows=args.fp_rows,integer_rows=args.integer_rows)
  self.jobs=self.expand_jobs()
  self.record=None
 def ref(self,name,offset,size):
  b=self.plan['buffers'][name];need(0<=offset and offset+size<=b['bytes'],'Partition exceeds planned buffer')
  return dict(buffer=name,buffer_offset=offset,arena_offset=b['offset']+offset,extent_bytes=size)
 def shifted(self,ref,offset,size):return self.ref(ref['buffer'],ref['buffer_offset']+offset,size)
 def location(self,ref):return ref['alternatives']['8'] if 'select_by_frozen_precision_group' in ref else ref
 def take(self,ref,size=None):
  ref=self.location(ref);size=ref['extent_bytes'] if size is None else size
  checked=self.ref(ref['buffer'],ref['buffer_offset'],size);need(checked['arena_offset']==ref['arena_offset'],'Arena pointer association differs')
  start=ref['arena_offset'];return bytes(self.arena[start:start+size])
 def put(self,ref,raw):
  checked=self.ref(ref['buffer'],ref['buffer_offset'],len(raw));need(checked['arena_offset']==ref['arena_offset'],'Arena output pointer differs')
  self.arena[ref['arena_offset']:ref['arena_offset']+len(raw)]=raw
 def scale(self,a):
  v=a['SCALE']
  if 'f32_bits' in v:return struct.unpack('<f',struct.pack('<I',v['f32_bits']))[0]
  b=self.plan['buffers']['input_scales'];off=b['offset']+4*v['local_input_scale_index'];return floats(self.arena[off:off+4])[0]
 def output_ref(self,job):
  a=job['arguments'];return a['CODES'] if a.get('OP')==1 else a['OUTPUT_R']
 def expand_jobs(self):
  jobs=[];commands=self.plan['commands'];index=0
  prefix_names=['embedding_sum_rows0','embedding_sum_rows128','embedding_ln_rows0','embedding_ln_rows128',
                'layer0_query_quantize_rows0','layer0_query_quantize_rows128']
  while index<len(commands):
   c=commands[index];index+=1
   if c['kind']!='service':jobs.append(dict(c));continue
   if c['name'] in prefix_names:
    jobs.append(dict(c,reused_prefix=True));continue
   if c['engine']=='integer':
    a=c['arguments'];rescale=[]
    while index<len(commands) and commands[index].get('engine')=='fp32' and commands[index]['arguments']['OP']==2:
     rescale.append(commands[index]);index+=1
    need(rescale and all(v['name'].startswith(c['name']+'_rescale_rows') for v in rescale),'Expected adjacent matrix rescale commands')
    base=rescale[0]['arguments'];k,n=a['INNER'],a['OUTPUTS']
    first=0
    limit=min(self.args.integer_rows,128 if (k,n)==(256,256) else 64)
    while first<a['ROWS']:
     rows=min(limit,a['ROWS']-first)
     # Preserve the existing observed first16rows, then use larger safe chunks.
     if c['name']=='layer0_query' and first==0 and limit>=16:rows=16
     aa=dict(a,ROWS=rows,ACTIVATIONS=self.shifted(a['ACTIVATIONS'],first*k,rows*k),
             OUTPUT_R=self.shifted(a['OUTPUT_R'],first*n*4,rows*n*4))
     jobs.append(dict(c,name=c['name']+f'_integer_rows{first}',arguments=aa,partition_first_row=first,
                      original_command_index=c['index'],numerical_partition=True))
     # The FP service has a32768-element limit even when an integer chunk is larger.
     for within in range(0,rows,min(self.args.fp_rows,32768//n)):
      current_rows=min(rows-within,self.args.fp_rows,32768//n);row=first+within
      bb=dict(base,ROWS=current_rows,ACCUMULATORS=self.shifted(a['OUTPUT_R'],row*n*4,current_rows*n*4),
              OUTPUT_R=self.shifted(base['OUTPUT_R'],row*n*4,current_rows*n*4))
      jobs.append(dict(rescale[0],name=c['name']+f'_rescale_rows{row}',arguments=bb,
                       partition_first_row=row,original_command_indices=[v['index'] for v in rescale],numerical_partition=True))
     first+=rows
   else:
    a=c['arguments'];op=a['OP'];width=a['WIDTH'];outputs=a['OUTPUTS']
    for first in range(0,a['ROWS'],self.args.fp_rows):
     rows=min(self.args.fp_rows,a['ROWS']-first);aa=dict(a,ROWS=rows)
     for field in ('X','Y','Z','ACCUMULATORS'):
      per_row=(field=='X' or (field=='Y' and op in (3,4,6)) or (field=='Z' and op==3) or (field=='ACCUMULATORS' and op==2))
      if per_row and a[field]['extent_bytes']:aa[field]=self.shifted(a[field],first*width*4,rows*width*4)
     field='CODES' if op==1 else 'OUTPUT_R';factor=1 if op==1 else 4
     outwidth=outputs if op==0 else width
     aa[field]=self.shifted(a[field],first*outwidth*factor,rows*outwidth*factor)
     jobs.append(dict(c,name=c['name']+f'_part{first}',arguments=aa,partition_first_row=first,
                      original_command_index=c['index'],numerical_partition=True))
  for i,job in enumerate(jobs):
   job['replay_index']=i
   if job.get('numerical_partition'):
    job.pop('register_writes',None);job.pop('completion',None)
    job['dispatch_contract']='Actual component harness controlled by recorded xsim arguments; numerical row partition, not original whole-call control timing'
  need(sum(v.get('reused_prefix',False) for v in jobs)==6,'Prefix task coverage differs')
  return jobs
 def movement(self,c):
  rows,cols=c['rows'],c['columns'];mode=c['mode']
  if mode=='binary_mask_to_fp32_repeated_rows':
   b=self.plan['buffers']['attention_mask'];mask=self.arena[b['offset']:b['offset']+b['bytes']]
   need(all(v in (0,1) for v in mask),'Invalid mask byte')
   self.put(c['destination'],b''.join(struct.pack('<I',0x80000000 if v else 0xff7fffff) for v in mask)*rows);return
  source=self.take(c['source']);target=bytearray(self.take(c['destination']));ss=c['source_stride_bytes'];ds=c['destination_stride_bytes']
  if mode=='indexed_embedding_rows':
   b=self.plan['buffers']['token_ids' if c['index_source']=='local_token_ids' else 'type_ids']
   indices=struct.unpack('<'+'I'*rows,self.arena[b['offset']:b['offset']+rows*4]);need(all(v<c['index_limit'] for v in indices),'Invalid embedding ID')
  else:indices=range(rows)
  for row,srow in enumerate(indices):
   if mode=='transpose_f32_tiles16':
    for col in range(cols):target[col*ds+row*4:col*ds+row*4+4]=source[srow*ss+col*4:srow*ss+col*4+4]
   else:target[row*ds:row*ds+cols*4]=source[srow*ss:srow*ss+cols*4]
  self.put(c['destination'],target)
 def control(self,c):
  if c['name']=='validate_window_and_load_inputs':
   for name,limit in [('token_ids',30522),('type_ids',2)]:
    b=self.plan['buffers'][name];values=struct.unpack('<256I',self.arena[b['offset']:b['offset']+1024]);need(all(v<limit for v in values),'Invalid saved input IDs')
   b=self.plan['buffers']['attention_mask'];mask=self.arena[b['offset']:b['offset']+256]
   need(all(v in (0,1) for v in mask) and sum(mask)==self.prep['valid_length'],'Saved mask/length differs')
   b=self.plan['buffers']['input_scales'];need(all(math.isfinite(v) and v>0 for v in floats(self.arena[b['offset']:b['offset']+96])),'Invalid frozen input scale')
  else:
   risk_raw=self.take(c['source']);risk=floats(risk_raw)[0];need(math.isfinite(risk) and 0<=risk<=1,'Invalid terminal risk')
   threshold_raw=(DATA/'threshold.bin').read_bytes();need(threshold_raw==bytes.fromhex(self.prep['threshold_bits_hex'])[::-1],'Threshold bits differ')
   threshold=floats(threshold_raw)[0];b=self.plan['buffers']['logits'];logits=bytes(self.arena[b['offset']:b['offset']+8])
   expected_logits=(REFERENCE/'model_logits.bin').read_bytes();expected_risk=(REFERENCE/'model_risk.bin').read_bytes()
   self.record['terminal']=dict(logits=list(floats(logits)),risk=risk,logits_bits=logits.hex(),risk_bits=risk_raw.hex(),
    logits_bit_equal=logits==expected_logits,risk_bit_equal=risk_raw==expected_risk,
    logits_absolute_errors=[abs(v-e) for v,e in zip(floats(logits),floats(expected_logits))],
    risk_absolute_error=abs(risk-floats(expected_risk)[0]),threshold_bits=threshold_raw.hex(),
    document_window_count=1,document_max_risk=risk,strict_reject=risk>threshold,
    saved_strict_reject=floats(expected_risk)[0]>threshold,decision_agrees=(risk>threshold)==(floats(expected_risk)[0]>threshold),
    detector_score_acceptance=None)
   (self.out/'actual_logits.bin').write_bytes(logits);(self.out/'actual_risk.bin').write_bytes(risk_raw)
 def save(self):rtl.save(self.out/'run.json',self.record)
 def arrays(self,job):
  a=job['arguments']
  if job['engine']=='integer':return dict(activations=self.take(a['ACTIVATIONS']),weights=self.take(a['WEIGHTS']))
  return {p:self.take(a[f]) if self.location(a[f])['extent_bytes'] else b'\0'*4
          for p,f in [('x','X'),('y','Y'),('z','Z'),('accumulators','ACCUMULATORS')]}
 def prefix_output(self,job):
  stage='embedding_add' if job['name'].startswith('embedding_sum') else 'embedding_ln' if job['name'].startswith('embedding_ln') else 'layer0_query_a8'
  slot=0 if job['name'].endswith('rows0') else 1
  return capture(PREFIX/stage/f'actual_{slot}.hex',self.output_ref(job)['extent_bytes'])
 def cached_integer(self,job):
  if job['name']!='layer0_query_integer_rows0' or job['arguments']['ROWS']!=16:return None
  old=DATA/'rtl_query_integer_rows16_v1';run=read(old/'run.json');a=job['arguments'];arrays=self.arrays(job)
  need(run['passed'] and run['state']=='returned' and run['case']==self.prep['case']
       and run['integer_synthesis_recorded_utc']==self.int_run['recorded_utc'],'Saved actual integer source association differs')
  need(capture(old/'activations.hex',4096)==arrays['activations'] and capture(old/'weights.hex',65536)==arrays['weights'],'Saved actual integer inputs differ')
  actual=(old/'actual_0.bin').read_bytes();expected=integer_reference(arrays['activations'],arrays['weights'],16,256,256)
  need(actual==expected and len(actual)==16384,'Saved actual integer output differs')
  return actual
 def initialize(self):
  self.out.mkdir(parents=True,exist_ok=True)
  if (self.out/'run.json').exists():
   need(self.args.resume,'Existing replay requires --resume');self.record=read(self.out/'run.json')
   need(self.record['identity']==self.identity and self.record['policy']==POLICY,'Resume arithmetic/partition identity changed')
   need(self.record['state']!='returned','Complete replay already exists')
  else:
   self.record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),identity=self.identity,policy=POLICY,
    state='preparing',completed=False,progress=[],new_vendor_invocations=0,full_detector_acceptance=None,
    scope='Actual synthesized arithmetic linked by relative-arena Python movement; row partitions are numerical only, not full-call/device/board timing')
   shutil.copyfile(PLAN,self.out/'command_plan.json');shutil.copyfile(PREPARED/'preparation.json',self.out/'preparation.json')
  dispatch=len(list(self.out.glob('driver_dispatch_*.py')))+1
  shutil.copyfile(Path(__file__),self.out/f'driver_dispatch_{dispatch:04d}.py')
  self.record['last_dispatch_source']=f'driver_dispatch_{dispatch:04d}.py'
  # Rebuild from the original arena and observed outputs. Expected tensors are never injected.
  for index,item in enumerate(self.record['progress']):
   job=self.jobs[index];need(item['job']==job and item['passed'],'Saved completed task differs')
   if job['kind']=='movement':self.movement(job)
   elif job['kind']=='control':self.control(job)
   else:
    if item.get('reused_prefix'):actual=self.prefix_output(job)
    elif item.get('reused_integer'):actual=self.cached_integer(job)
    else:
     folder=Path(item['folder']);arrays=self.arrays(job)
     for port,raw in arrays.items():need((folder/f'{port}_{item["slot"]}.bin').read_bytes()==raw,'Resume actual input association differs')
     actual=(folder/f'actual_{item["slot"]}.bin').read_bytes()
    need(len(actual)==self.output_ref(job)['extent_bytes'],'Resume output extent differs');self.put(self.output_ref(job),actual)
  if self.record.get('arena_checkpoint_task_count')==len(self.record['progress']):
   need((self.out/'arena_checkpoint.bin').read_bytes()==bytes(self.arena),'Reconstructed replay arena differs from saved checkpoint')
  self.save()
 def private_snapshot(self,kind):
  sim=self.out/(kind+'_sim')
  if kind=='fp':
   tag=re.sub('[^A-Za-z0-9_-]','_',self.fp_run['recorded_utc']);build=ROOT/'build/fixed_fp32_rtl'/tag
   compiled=read(build/'compiled.json');need(compiled['signature']==dict(synthesis_recorded_utc=self.fp_run['recorded_utc'],service_schedule_revision=self.fp_run['service_schedule_revision']),'FP snapshot signature differs')
   for name in ('fixed_fp32_axi_tb.sv','linear_axi_tb.sv'):need((build/name).read_bytes()==(ROOT/'hardware'/name).read_bytes(),'FP compiled harness differs')
   for name in compiled['reachable_rtl_files']:need((build/'rtl'/name).read_bytes()==(self.fp_source/'project/solution/syn/verilog'/name).read_bytes(),'FP compiled source differs')
   original=build/'ip_project/fixed_fp32_sim.sim/sim_1/behav/xsim';snapshot=compiled['snapshot']
  else:
   original=DATA/'rtl_query_integer_rows16_v1/sim';snapshot='real_linear_axi_tb'
   need((original/'real_linear_axi_tb.sv').read_bytes()==(ROOT/'hardware/real_linear_axi_tb.sv').read_bytes(),'Integer compiled harness differs')
   for source in (self.int_source/'project/solution/syn/verilog').glob('*.v'):need((original/'rtl'/source.name).read_bytes()==source.read_bytes(),'Integer compiled source differs')
  need((original/'xsim.dir'/snapshot/'xsimk').is_file(),'Existing compiled executable missing')
  if not sim.exists():
   sim.mkdir();shutil.copytree(original/'xsim.dir',sim/'xsim.dir',ignore=shutil.ignore_patterns('*.log','*.jou','*.wdb','xsim_script.tcl','xsimSettings.ini','TempBreakPointFile.txt'))
   for p in [*original.glob('*.dat'),original/'xsim.ini']:
    if p.is_file():shutil.copyfile(p,sim/p.name)
   (sim/'run.tcl').write_text('run all\nquit\n')
  return sim,snapshot
 def compatible(self,job,peer):
  if job.get('engine')!='fp32' or peer.get('engine')!='fp32' or peer.get('reused_prefix'):return False
  a,b=job['arguments'],peer['arguments']
  if any(a[k]!=b[k] for k in ('OP','ROWS','WIDTH','OUTPUTS','DOT_BIAS')) or self.scale(a)!=self.scale(b):return False
  aa,bb=self.arrays(job),self.arrays(peer)
  if {p:len(v) for p,v in aa.items()}!={p:len(v) for p,v in bb.items()}:return False
  out=self.output_ref(job);begin,end=out['arena_offset'],out['arena_offset']+out['extent_bytes']
  for field in ('X','Y','Z','ACCUMULATORS'):
   ref=self.location(b[field])
   if ref['extent_bytes'] and max(begin,ref['arena_offset'])<min(end,ref['arena_offset']+ref['extent_bytes']):return False
  return True
 def compare(self,job,actual,expected,arrays):
  a=job['arguments'];need(len(actual)==len(expected)==self.output_ref(job)['extent_bytes'],'Actual/reference/output extent differs')
  if job['engine']=='integer' or a['OP']==1:result=compare_codes(actual,expected)
  else:
   bounds=POLICY['fp_local_bounds'].get(job['operation'],POLICY['fp_local_bounds']['other']);result=compare_float(actual,expected,*bounds)
   if a['OP']==6:
    values=floats(actual);error=max(abs(math.fsum(values[r*a['WIDTH']:(r+1)*a['WIDTH']])-1) for r in range(a['ROWS']))
    result['maximum_row_sum_error']=error;result['passed'] &= error<=2e-5
  return result
 def propagation(self,job,actual,arrays):
  a=job['arguments']
  if job['engine']!='fp32' or a['OP']!=1:return None
  matrix=job['name'].split('_quantize_rows')[0];off=a['CODES']['buffer_offset']
  expected=(REFERENCE/('model_'+matrix+'_a8.bin')).read_bytes()[off:off+len(actual)]
  return codes_comparison(actual,expected,arrays['x'],self.scale(a),off)
 def invoke(self,job,peer=None):
  # A failed/partial invocation is retained; a later explicit resume uses a new attempt.
  base=self.out/f'task_{job["replay_index"]:04d}';folder=base;attempt=1
  while folder.exists():attempt+=1;folder=self.out/(base.name+f'_attempt{attempt}')
  folder.mkdir();stages=[job,peer or job];arrays=[self.arrays(v) for v in stages]
  item=dict(state='preparing',passed=False,jobs=[v['replay_index'] for v in stages],
   slot_semantics='two independent required row partitions' if peer else 'same required partition twice; second slot diagnostic repeat',
   synthesis_reused=True,new_synthesis=False,command_timeout_seconds=900)
  rtl.save(folder/'run.json',item);expected=[]
  for slot,(stage,values) in enumerate(zip(stages,arrays)):
   for port,raw in values.items():
    (folder/f'{port}_{slot}.bin').write_bytes(raw);hex_write(folder/f'{port}_{slot}.hex',raw)
   a=stage['arguments']
   current=integer_reference(values['activations'],values['weights'],a['ROWS'],a['INNER'],a['OUTPUTS']) if stage['engine']=='integer' else fp_reference(a,values,self.scale(a))
   expected.append(current);(folder/f'expected_{slot}.bin').write_bytes(current)
  a=job['arguments']
  if job['engine']=='integer':
   sim,snapshot=self.private_snapshot('integer');hex_write(folder/'activations.hex',arrays[0]['activations']);hex_write(folder/'weights.hex',arrays[0]['weights'])
   values=dict(ROWS=a['ROWS'],INNER=a['INNER'],OUTPUTS=a['OUTPUTS'],BITS=8,REPS=2,CLOCK_NS=self.fp_run['clock_ns'],TIMEOUT_CYCLES=10000000,CASE_DIR=str(folder))
  else:
   sim,snapshot=self.private_snapshot('fp');count=a['ROWS']*(a['OUTPUTS'] if a['OP']==0 else a['WIDTH'])
   values=dict(OP=a['OP'],ROWS=a['ROWS'],WIDTH=a['WIDTH'],OUTPUTS=a['OUTPUTS'],BIAS=a['DOT_BIAS'],SCALE=struct.pack('>f',self.scale(a)).hex(),
    COUNT=count,CASE_DIR=str(folder),CLOCK_NS=self.fp_run['clock_ns'],TIMEOUT_CYCLES=10000000,
    OUTPUT_BYTES=4 if a['OP']==1 else count*4,CODES_BYTES=((count+3)//4)*4 if a['OP']==1 else 4)
   values.update({port.upper()+'_BYTES':len(raw) for port,raw in arrays[0].items()})
  command=[str(self.tools/'xsim'),snapshot,'-tclbatch',str(sim/'run.tcl')]
  for key,value in values.items():command+=['-testplusarg',str(key)+'='+str(value)]
  item.update(state='running',command=command);rtl.save(folder/'run.json',item)
  try:
   rc=rtl.command_run(command,sim,folder/'simulation.log',timeout=900);log=(folder/'simulation.log').read_text(errors='replace')
   if job['engine']=='fp32':
    case=dict(op=a['OP'],rows=a['ROWS'],width=a['WIDTH'],outputs=a['OUTPUTS'])
    tx,traffic,final,issues=rtl.parse_capture(log,case,self.fp_run['clock_ns'])
   else:
    tx=[json.loads(v.split(' ',1)[1]) for v in log.splitlines() if v.startswith('GATE_RTL_TRANSACTION ')];finals=[json.loads(v.split(' ',1)[1]) for v in log.splitlines() if v.startswith('GATE_RTL_VERIFIED ')]
    final=finals[0] if len(finals)==1 else None;traffic=[];issues=[]
    if rc or 'GATE_RTL_FAIL' in log or len(tx)!=2 or len(finals)!=1:issues.append('integer status/capture count')
    if final and (not final['passed'] or any(final.get(k)!=v for k,v in dict(rows=a['ROWS'],inner=a['INNER'],outputs=a['OUTPUTS'],bits=8,repetitions=2,clock_period_ns=self.fp_run['clock_ns']).items())):issues.append('integer capture definition')
    if [v.get('index') for v in tx]!=[0,1] or any(not v.get('output_captured') or v['latency_cycles']<=0 or v['latency_cycles']!=v['done_cycle']-v['start_cycle'] for v in tx):issues.append('integer transaction definition')
   item.update(returncode=rc,transactions=tx,traffic=traffic,captured=final,issues=issues);rtl.save(folder/'run.json',item)
   need(rc==0 and not issues,'RTL capture/status failure at '+job['name'])
   outputs=[];checks=[]
   for slot in range(2):
    raw=capture(folder/f'actual_{slot}.hex',len(expected[slot]));(folder/f'actual_{slot}.bin').write_bytes(raw);outputs.append(raw)
    # Store comparison diagnostics before propagating any required output.
    check=self.compare(stages[slot],raw,expected[slot],arrays[slot]);checks.append(check)
    item['same_input_comparisons']=checks;rtl.save(folder/'run.json',item)
    need(check['passed'],'Same-input numerical mismatch at '+stages[slot]['name'])
   item.update(same_input_comparisons=checks,propagation=[],singleton_slot_bytes_equal=outputs[0]==outputs[1] if not peer else None);rtl.save(folder/'run.json',item)
   need(peer is not None or item['singleton_slot_bytes_equal'],'Identical RTL inputs produced differing repeated output bytes')
   for slot,stage in enumerate(stages[:2 if peer else 1]):
    propagation=self.propagation(stage,outputs[slot],arrays[slot]);item['propagation'].append(propagation);rtl.save(folder/'run.json',item)
    need(propagation is None or propagation['passed'],'Diagnostic pause: actual A8 differs from saved ordered reference at '+stage['name']+'; detector-level acceptance remains unresolved')
   item.update(state='returned',passed=True);rtl.save(folder/'run.json',item)
   return folder,outputs,item
  except Exception as error:
   item.update(state='stopped',passed=False,error=str(error));rtl.save(folder/'run.json',item);raise
 def execute(self):
  rtl.paused();self.initialize();new_services=0
  try:
   i=len(self.record['progress'])
   while i<len(self.jobs):
    rtl.paused();need(read(self.fp_source/'run.json')==self.fp_run and read(self.int_source/'run.json')==self.int_run,'Synthesis changed during replay')
    job=self.jobs[i]
    if job['kind']=='movement':self.movement(job);entry=dict(job=job,passed=True)
    elif job['kind']=='control':self.control(job);entry=dict(job=job,passed=True)
    elif job.get('reused_prefix'):
     self.put(self.output_ref(job),self.prefix_output(job));entry=dict(job=job,passed=True,reused_prefix=True)
    elif self.cached_integer(job) is not None:
     self.put(self.output_ref(job),self.cached_integer(job));entry=dict(job=job,passed=True,reused_integer=True)
    else:
     if new_services>=self.args.max_services:
      (self.out/'arena_checkpoint.bin').write_bytes(self.arena);self.record['arena_checkpoint_task_count']=len(self.record['progress'])
      self.record.update(state='bounded_handoff',completed=False,next_task=i);self.save();return
     peer=self.jobs[i+1] if i+1<len(self.jobs) and new_services+1<self.args.max_services and self.compatible(job,self.jobs[i+1]) else None
     self.record.update(state='running',active_task=i);self.save()
     folder,outputs,observation=self.invoke(job,peer)
     stages=[job,peer] if peer else [job]
     for slot,stage in enumerate(stages):
      self.put(self.output_ref(stage),outputs[slot]);self.record['progress'].append(dict(job=stage,passed=True,folder=str(folder),slot=slot))
     count=len(stages);new_services+=count;i+=count;self.record['new_vendor_invocations']+=1
     self.record.update(next_task=i,active_task=None);self.save()
     print(json.dumps(dict(passed=True,task=job['name'],required_partitions=len(stages),next_task=i)),flush=True);continue
    self.record['progress'].append(entry);i+=1;self.record['next_task']=i;self.save()
   (self.out/'arena_checkpoint.bin').write_bytes(self.arena);self.record['arena_checkpoint_task_count']=len(self.record['progress'])
   self.record.update(state='returned',completed=True,active_task=None);self.save()
  except Exception as error:
   self.record.update(state='stopped',completed=False,error=str(error));self.save();raise
 def preview(self):
  next_job=next(v for v in self.jobs if v.get('engine')=='fp32' and v['name']=='layer0_query_rescale_rows0')
  print(json.dumps(dict(execute=False,identity=self.identity,policy=POLICY,expanded_tasks=len(self.jobs),
    next_new_service=next_job['name'],arguments=next_job['arguments'],prefix_calls_reused=6,
    reused_actual_integer_rows=16,new_synthesis=False,new_compilation=False,
    max_required_new_partitions=self.args.max_services,command_watchdog_seconds=900,
    scalar_arithmetic='unchanged generated integer/FP32 RTL; ascending reduction retained within each row',
    timing_boundary='Numerical row partitions and Python movement/file handoffs; no full detector/device timing claim'),indent=2))

def prepare_probes(args):
 """Reconstruct selected C-stage inputs from saved actual codes, never inference."""
 import ctypes,os,sys
 case=args.case_directory.resolve();data=case.parent.parent;params=data/'parameters'
 summary=read(case/'bridge_summary.json');window=read(case/'window.json')
 need(0<=args.probe_layer<4 and 0<=args.probe_row<256 and 0<=args.probe_head<4,'Bounded layer/row/head required')
 prefix='layer'+str(args.probe_layer);first=args.probe_row;head=args.probe_head*64
 need(summary['completed'] and summary['case']==window['document_id'] and summary['encoder_arithmetic']=='w8_a8_integer_contract','Completed matching W8 C case required')
 need(sum(summary['service_calls_by_op'])==411 and len(summary['integer_layers'])==24,'C service schedule differs')
 need(len(window['inputs']['input_ids'])==256,'Probes require saved L256 case')
 compiled=read(ROOT/'build/development_qualification_w8_v1/compiled.json')
 need(compiled['passed'],'Successful unchanged C compilation required')
 out=args.output.resolve();need(out.is_relative_to(DATA) and not out.exists(),'Preserve existing probe evidence; output must be new under numerical_bridge')
 config=read(ROOT/'configs/project.json');source,synthesis=rtl.source_evidence(config)
 need(compiled['synthesis_recorded_utc']==synthesis['recorded_utc'],'C/RTL synthesis association differs')
 build=ROOT/'build/development_qualification_w8_v1'
 need((build/'fixed_fp32_service.hpp').read_bytes()==(source/'fixed_fp32_service.hpp').read_bytes(),'C/RTL arithmetic source differs')
 mk=(build/'project/solution/csim/build/csim.mk').read_text()
 vendor=Path(re.search(r'^AUTOPILOT_ROOT := (.+)$',mk,re.M).group(1))
 fpo=vendor/'lnx64/tools/fpo_v7_1/libIp_floating_point_v7_1_bitacc_cmodel.so'
 mathlib=vendor/'lnx64/lib/csim/libhlsmc++-GCC46.so'
 # The installed FPO library has no SONAME; its dependents require loader search.
 if str(fpo.parent) not in os.environ.get('LD_LIBRARY_PATH','').split(':'):
  env=os.environ.copy();env['LD_LIBRARY_PATH']=str(fpo.parent)+':'+env.get('LD_LIBRARY_PATH','')
  os.execvpe(sys.executable,[sys.executable,*sys.argv],env)
 for dependency in ('libgmp.so.11','libmpfr.so.4'):ctypes.CDLL(str(fpo.parent/dependency),mode=ctypes.RTLD_GLOBAL)
 ctypes.CDLL(str(fpo),mode=ctypes.RTLD_GLOBAL);library=ctypes.CDLL(str(mathlib))
 funcs={}
 for name in ('expf','erff','tanhf'):
  fn=getattr(library,'_ZN3hls'+str(len(name))+name+'Ef');fn.argtypes=[ctypes.c_float];fn.restype=ctypes.c_float;funcs[name]=fn
 out.mkdir();(out/'inputs').mkdir();shutil.copyfile(Path(__file__),out/'driver_prepare.py')
 record=dict(state='preparing',case=window,source_case=str(case),parameters=str(params),c_compiled_record=compiled,
  fp_synthesis=synthesis['recorded_utc'],math_library=str(mathlib),fpo_library=str(fpo),new_synthesis=False,new_compilation=False,
  inputs_kind='Reconstructed C-stage inputs from saved actual C A8/layer bytes, exact integer sums and separate ascending FP32 arithmetic; not captured pre-input trace',
  scope='Selected actual-input C-to-RTL arithmetic diagnostics; no full RTL detector or score acceptance',policy=POLICY,
  full_detector_acceptance=None,probes=[],source_inputs=[])
 def load(p,size=None):
  raw=p.read_bytes();need(size is None or len(raw)==size,'Probe source extent differs: '+str(p))
  target=out/'inputs'/p.name
  if target.exists():need(target.read_bytes()==raw,'Ambiguous probe source filename')
  else:target.write_bytes(raw);record['source_inputs'].append(dict(source=str(p),file=str(target.relative_to(out)),bytes=len(raw)))
  return raw
 def projection(name,rows,outputs=256,first_row=0):
  codes=load(case/('actual_'+name+'_a8.bin'));need(len(codes)==65536,'Hidden A8 extent differs')
  need(codes==load(case/(name+'_reference_a8.bin')),'Saved actual/ordered A8 differs; diagnose before probing')
  weight=load(params/(name+'_w8.bin'),256*outputs);p=floats(load(params/(name+'_params.bin'),4*(1+2*outputs)))
  acc=integer_reference(codes[first_row*256:(first_row+rows)*256],weight,rows,256,outputs)
  values=fp_reference(dict(OP=2,ROWS=rows,WIDTH=outputs,OUTPUTS=0),dict(accumulators=acc,y=pack(p[1:1+outputs]),z=pack(p[1+outputs:])),p[0])
  (out/'inputs'/(name+'_reconstructed_acc.bin')).write_bytes(acc)
  (out/'inputs'/(name+'_reconstructed_fp.bin')).write_bytes(values)
  return values
 q=floats(projection(prefix+'_query',1,first_row=first));k=floats(projection(prefix+'_key',256));v=floats(projection(prefix+'_value',256))
 queries=pack(q[head:head+64]);keys=pack([k[t*256+head+d] for t in range(256) for d in range(64)])
 scores=fp_reference(dict(OP=0,ROWS=1,WIDTH=64,OUTPUTS=256,DOT_BIAS=False),dict(x=queries,y=keys),1.)
 # The service invokes four N64 panels; each output keeps the identical ascending K order.
 mask=load(case/'key_mask.bin',1024);scaled=[f32(f32(x/8.)+m) for x,m in zip(floats(scores),floats(mask))]
 maximum=max(scaled);exps=[funcs['expf'](f32(x-maximum)) for x in scaled];total=0.
 for x in exps:total=f32(total+x)
 probabilities=pack([f32(x/total) for x in exps])
 transposed=pack([v[t*256+head+d] for d in range(64) for t in range(256)])
 av=fp_reference(dict(OP=0,ROWS=1,WIDTH=256,OUTPUTS=64,DOT_BIAS=False),dict(x=probabilities,y=transposed),1.)
 intermediate=projection(prefix+'_ffn_input',1,1024,first) if not args.attention_only else b''
 gelu=pack([f32(f32(x*.5)*f32(1.+funcs['erff'](f32(x/f32(1.4142135623730951))))) for x in floats(intermediate)])
 hidden=load(case/'bridge_layer3.bin',262144)[:1024] if not args.attention_only else b''
 pool_weight=load(params/'pooler_weight.bin',262144);pool_bias=load(params/'pooler_bias.bin',1024)
 pooled=fp_reference(dict(OP=0,ROWS=1,WIDTH=256,OUTPUTS=256,DOT_BIAS=True),dict(x=hidden,y=pool_weight,z=pool_bias),1.) if not args.attention_only else b''
 tanh=pack([funcs['tanhf'](x) for x in floats(pooled)])
 clsweight=load(params/'classifier_weight.bin',2048);clsbias=load(params/'classifier_bias.bin',8)
 reconstructed_logits=fp_reference(dict(OP=0,ROWS=1,WIDTH=256,OUTPUTS=2,DOT_BIAS=True),dict(x=tanh,y=clsweight,z=clsbias),1.) if not args.attention_only else b''
 if not args.attention_only:need(reconstructed_logits==load(case/'bridge_logits.bin',8),'Reconstructed C tanh/pooler path does not reproduce actual C logits')
 record['reconstructed_terminal_logits_exact']=True if not args.attention_only else None
 valid=[i for i,m in enumerate(window['inputs']['attention_mask']) if m]
 record['attention_selection']=dict(layer=args.probe_layer,query_token=first,head=args.probe_head,context_channels=[head,head+63],first_valid_key=valid[0],last_valid_key=valid[-1],valid_keys=len(valid),query_valid=bool(window['inputs']['attention_mask'][first]))
 def add(name,op,width,outputs,arrays,expected,operation,scale=1.,downstream=None):
  folder=out/name;folder.mkdir();arrays={p:arrays.get(p,b'\0'*4) for p in ('x','y','z','accumulators')}
  for port,raw in arrays.items():(folder/(port+'.bin')).write_bytes(raw)
  (folder/'expected_c.bin').write_bytes(expected)
  item=dict(name=name,op=op,rows=1,width=width,outputs=outputs,bias=False,scale=scale,operation=operation,
   output_bytes=len(expected),directory=name,expected_kind='Reconstructed actual vendor C-service arithmetic on saved actual preceding C values',
   acceptance=None)
  if downstream:
   matrix,values,offset=downstream;p=floats(load(params/(matrix+'_params.bin')));saved=load(case/('actual_'+matrix+'_a8.bin'))[offset:offset+len(values)//4]
   need(saved==load(case/(matrix+'_reference_a8.bin'))[offset:offset+len(saved)],'Saved A8 propagation association differs')
   predicted=fp_reference(dict(OP=1,ROWS=1,WIDTH=len(saved),OUTPUTS=0),dict(x=values),p[0])
   need(saved==predicted,'Reconstructed C stage fails saved downstream A8 association: '+name)
   (folder/'expected_downstream_a8.bin').write_bytes(saved)
   item['downstream_a8']=dict(matrix=matrix,scale=p[0],values=len(saved),source_code_offset=offset,reconstructed_codes_equal=True,
    scope='Host nearest-even/clip on selected actual RTL output; no new quantization RTL call')
  record['probes'].append(item)
 add('softmax',6,256,0,dict(x=scores,y=mask),probabilities,'SOFTMAX',8.)
 add('av',0,256,64,dict(x=probabilities,y=transposed),av,'DOT',downstream=(prefix+'_attention_output',av,first*256+head))
 if not args.attention_only:
  add('gelu',7,1024,0,dict(x=intermediate),gelu,'GELU',downstream=(prefix+'_ffn_output',gelu,first*1024))
  add('tanh',8,256,0,dict(x=pooled),tanh,'TANH')
 if (args.probe_layer,first,args.probe_head)==(1,131,1):
  expected=signed((out/'av/expected_downstream_a8.bin').read_bytes())[50]
  need(expected==43,'Known tie coordinate expected saved C A8 code43')
  record['known_tie_boundary']=dict(layer=1,head=1,query_token=131,context_channel=114,head_channel=50,saved_c_a8=expected,reconstructed_c_context=floats(av)[50],scope='Probe prior attention quantization tie; no acceptance rule adopted')
 record.update(state='prepared',passed=True);rtl.save(out/'probes.json',record)
 print(json.dumps(dict(prepared=True,output=str(out),probes=record['probes'],full_detector_acceptance=None),indent=2))

def prepare_quant_length_probes(args):
 """One real RTL-to-A8 check plus four explicitly synthetic length shapes."""
 import ctypes,os,sys
 prior_dir=DATA/'rtl_known_tie_probes_v1';prior=read(prior_dir/'probes.json')
 avrun=read(prior_dir/'av/rtl/run.json');need(avrun['passed'],'Completed known-tie actual AV required')
 config=read(ROOT/'configs/project.json');source,synthesis=rtl.source_evidence(config)
 need(prior['fp_synthesis']==synthesis['recorded_utc'],'Known-tie source association changed')
 for item in prior['source_inputs']:need(Path(item['source']).read_bytes()==(prior_dir/item['file']).read_bytes(),'Known-tie actual C source association changed')
 out=args.output.resolve();need(out.is_relative_to(DATA) and not out.exists(),'New probe evidence directory required')
 mk=(ROOT/'build/development_qualification_w8_v1/project/solution/csim/build/csim.mk').read_text()
 vendor=Path(re.search(r'^AUTOPILOT_ROOT := (.+)$',mk,re.M).group(1));fpopath=vendor/'lnx64/tools/fpo_v7_1'
 if str(fpopath) not in os.environ.get('LD_LIBRARY_PATH','').split(':'):
  env=os.environ.copy();env['LD_LIBRARY_PATH']=str(fpopath)+':'+env.get('LD_LIBRARY_PATH','');os.execvpe(sys.executable,[sys.executable,*sys.argv],env)
 for name in ('libgmp.so.11','libmpfr.so.4','libIp_floating_point_v7_1_bitacc_cmodel.so'):ctypes.CDLL(str(fpopath/name),mode=ctypes.RTLD_GLOBAL)
 mathlib=vendor/'lnx64/lib/csim/libhlsmc++-GCC46.so';library=ctypes.CDLL(str(mathlib));expfn=getattr(library,'_ZN3hls4expfEf');expfn.argtypes=[ctypes.c_float];expfn.restype=ctypes.c_float
 out.mkdir();(out/'inputs').mkdir();shutil.copyfile(Path(__file__),out/'driver_prepare.py')
 record=dict(state='preparing',passed=False,fp_synthesis=synthesis['recorded_utc'],prior_known_tie=str(prior_dir/'probes.json'),
  source_case=prior['source_case'],c_compiled_record=prior['c_compiled_record'],math_library=str(mathlib),
  inputs_kind='Mixed bounded diagnostics; each probe declares actual-input versus synthetic cropped/padded shape scope',
  scope='One actual RTL quantization continuation; four arithmetic shape checks only, no L128/L512 model forward or hardware timing qualification',
  policy=POLICY,known_tie_boundary=prior['known_tie_boundary'],new_compilation=False,new_synthesis=False,full_detector_acceptance=None,probes=[],source_inputs=[])
 def load(path,name):
  raw=path.read_bytes();target=out/'inputs'/name;target.write_bytes(raw)
  record['source_inputs'].append(dict(source=str(path),file=str(target.relative_to(out)),bytes=len(raw)));return raw
 def add(name,op,width,outputs,arrays,expected,operation,scale,kind,source_info,producer=None):
  folder=out/name;folder.mkdir()
  for port in ('x','y','z','accumulators'):(folder/(port+'.bin')).write_bytes(arrays.get(port,b'\0'*4))
  (folder/'expected_c.bin').write_bytes(expected)
  item=dict(name=name,op=op,rows=1,width=width,outputs=outputs,bias=False,scale=scale,operation=operation,
   directory=name,output_bytes=len(expected),inputs_kind=kind,input_source=source_info,acceptance=None,
   expected_kind='Exact independent nearest-even/clip from actual RTL AV floats and frozen scale' if op==1 else 'Vendor C exp plus explicit separate ascending FP32 reduction on declared synthetic shape operands')
  if producer:item['input_from_probe']=producer
  record['probes'].append(item)
 actual=load(prior_dir/'av/rtl/actual_0.bin','known_tie_actual_av64.bin')
 need(actual==(prior_dir/'av/rtl/actual_1.bin').read_bytes() and len(actual)==256,'Known actual AV capture extent/repeat differs')
 avprobe=next(v for v in prior['probes'] if v['name']=='av');scale=avprobe['downstream_a8']['scale']
 expected=fp_reference(dict(OP=1,ROWS=1,WIDTH=64,OUTPUTS=0),dict(x=actual),scale)
 need(expected==load(prior_dir/'av/expected_downstream_a8.bin','known_tie_saved_c_a8.bin') and signed(expected)[50]==43,'Known-tie A8 reference association differs')
 add('quantize',1,64,0,dict(x=actual),expected,'QUANTIZE',scale,
  'Actual preceding known-tie AV RTL output bytes with frozen layer1_attention_output scale',
  dict(actual_av_record=str(prior_dir/'av/rtl/run.json'),layer=1,head=1,query_token=131,context_channels=[64,127],target_head_channel=50,expected_captured_rtl_code=43))
 scores=floats(load(prior_dir/'softmax/x.bin','known_tie_scores256.bin'));mask=floats(load(prior_dir/'softmax/y.bin','known_tie_mask256.bin'))
 transposed=floats(load(prior_dir/'av/y.bin','known_tie_v64x256.bin'));need(len(scores)==len(mask)==256 and len(transposed)==64*256,'Source attention shape differs')
 largest=floats(bytes.fromhex('ffff7f7f'))[0]
 for length in (128,512):
  score=list(scores[:length]) if length==128 else list(scores)+[f32(scores[k%256]+((k*7)%17-8)/16.) for k in range(256)]
  masks=list(mask[:length]) if length==128 else list(mask)+[0.]*256
  weights=[transposed[d*256+k] if k<256 else f32(transposed[d*256+(k-256)]+((d*13+k*7)%17-8)/32.) for d in range(64) for k in range(length)]
  if length==512:
   score[511]=f32(8.*(max(f32(x/8.) for x,m in zip(scores,mask) if m==0.)+8.))
   for d in range(64):weights[d*512+511]=f32((d%13+1)/8.)
  values=[f32(f32(x/8.)+m) for x,m in zip(score,masks)];maximum=max(values);exps=[expfn(f32(x-maximum)) for x in values];total=0.
  for x in exps:total=f32(total+x)
  probabilities=pack([f32(x/total) for x in exps]);x=pack(score);y=pack(masks);w=pack(weights)
  valid=[k for k,m in enumerate(masks) if m==0.]
  shape=dict(source_layer=1,source_head=1,source_query_token=131,synthetic_query_row=0,
   source_key_tokens=[0,min(length,256)-1],synthetic_appended_key_tokens=[256,511] if length==512 else None,key_length=length,first_valid_key=valid[0],last_valid_key=valid[-1],valid_keys=len(valid),
   crop='Keep source key tokens0..127; no second model window' if length==128 else None,
   extension='Append256 deterministic nonzero keys/V with unmasked extension; key511 score=max(source valid scaled scores)+8, V[d]=(d%13+1)/8' if length==512 else None,
   original_padding='Source padding keys177..255 are excluded by crop' if length==128 else 'Original source padding keys177..255 remain masked; no second window',
   qualification_limitation='Synthetic arithmetic shape only; one joint full-length softmax and AV reduction; no length-specific detector or batch qualification')
  kind='Synthetic L'+str(length)+' arithmetic shape from declared frozen L256 operands; not an actual L'+str(length)+' model trace'
  add('softmax'+str(length),6,length,0,dict(x=x,y=y),probabilities,'SOFTMAX',8.,kind,shape)
  av=fp_reference(dict(OP=0,ROWS=1,WIDTH=length,OUTPUTS=64,DOT_BIAS=False),dict(x=probabilities,y=w),1.)
  if length==512:
   omitted=fp_reference(dict(OP=0,ROWS=1,WIDTH=length,OUTPUTS=64,DOT_BIAS=False),dict(x=probabilities[:-4]+pack([0.]),y=w),1.)
   omission=compare_float(omitted,av,*POLICY['fp_local_bounds']['DOT']);need(not omission['passed'],'Last key must contribute above unchanged local bounds')
   shape['last_key_probability']=floats(probabilities)[511];shape['omit_key511_comparison']=omission
   (out/'inputs/av512_omit_key511.bin').write_bytes(omitted)
  add('av'+str(length),0,length,64,dict(x=probabilities,y=w),av,'DOT',1.,kind,shape,producer='softmax'+str(length))
 record.update(state='prepared',passed=True);rtl.save(out/'probes.json',record)
 print(json.dumps(dict(prepared=True,output=str(out),new_partitions=5,actual_calls_if_executed=10,probes=record['probes'],acceptance=None),indent=2))
def execute_probe(args):
 """One small probe, using the unchanged compiled FP service snapshot."""
 out=args.output.resolve();record=read(out/'probes.json');need(record['passed'] and record['state']=='prepared','Prepared actual-C probes required')
 probe=next(v for v in record['probes'] if v['name']==args.probe);folder=out/probe['directory']/'rtl'
 need(not folder.exists(),'Preserve existing probe attempt; inspect it before redispatch')
 rtl.paused();config=read(ROOT/'configs/project.json');source,synthesis=rtl.source_evidence(config)
 need(synthesis['recorded_utc']==record['fp_synthesis'],'Probe synthesis changed')
 for item in record['source_inputs']:need(Path(item['source']).read_bytes()==(out/item['file']).read_bytes(),'Probe saved actual source changed')
 a=dict(OP=probe['op'],ROWS=1,WIDTH=probe['width'],OUTPUTS=probe['outputs'],DOT_BIAS=False)
 arrays={p:(out/probe['directory']/(p+'.bin')).read_bytes() for p in ('x','y','z','accumulators')}
 c_expected=(out/probe['directory']/'expected_c.bin').read_bytes();input_kind=probe.get('inputs_kind',record['inputs_kind'])
 if args.probe=='av' or probe.get('input_from_probe'):
  producer=probe.get('input_from_probe','softmax');soft=out/producer/'rtl';prior=read(soft/'run.json');need(prior['passed'],'Complete actual softmax probe before AV')
  arrays['x']=(soft/'actual_0.bin').read_bytes();input_kind='Actual preceding '+producer+' RTL output bytes; '+probe.get('inputs_kind',record['inputs_kind'])
 expected=fp_reference(a,arrays,probe['scale'])
 harness=ArenaReplay.__new__(ArenaReplay);harness.out=out;harness.fp_run=synthesis;harness.fp_source=source
 sim,snapshot=harness.private_snapshot('fp');folder.mkdir();shutil.copyfile(Path(__file__),folder/'driver_execute.py')
 run=dict(state='preparing',passed=False,probe=probe,policy=POLICY,inputs_kind=input_kind,
  slot_semantics='Identical required partition twice; second slot diagnostic repeat',command_timeout_seconds=900,
  synthesis_reused=True,new_compilation=False,new_synthesis=False,full_detector_acceptance=None)
 rtl.save(folder/'run.json',run)
 for port,raw in arrays.items():
  (folder/(port+'.bin')).write_bytes(raw)
  for slot in range(2):hex_write(folder/f'{port}_{slot}.hex',raw)
 (folder/'expected_same_input_host.bin').write_bytes(expected);(folder/'expected_c.bin').write_bytes(c_expected)
 count=len(expected) if a['OP']==1 else len(expected)//4;values=dict(OP=a['OP'],ROWS=1,WIDTH=a['WIDTH'],OUTPUTS=a['OUTPUTS'],BIAS=0,
  SCALE=struct.pack('>f',probe['scale']).hex(),COUNT=count,CASE_DIR=str(folder),CLOCK_NS=synthesis['clock_ns'],TIMEOUT_CYCLES=10000000,
  OUTPUT_BYTES=4 if a['OP']==1 else count*4,CODES_BYTES=((count+3)//4)*4 if a['OP']==1 else 4)
 values.update({p.upper()+'_BYTES':len(v) for p,v in arrays.items()})
 command=[str(Path(config['hardware']['hls']['vivado_executable']).parent/'xsim'),snapshot,'-tclbatch',str(sim/'run.tcl')]
 for k,v in values.items():command+=['-testplusarg',str(k)+'='+str(v)]
 run.update(state='running',command=command);rtl.save(folder/'run.json',run)
 try:
  rc=rtl.command_run(command,sim,folder/'simulation.log',timeout=900)
  log=(folder/'simulation.log').read_text(errors='replace');case=dict(op=a['OP'],rows=1,width=a['WIDTH'],outputs=a['OUTPUTS'])
  tx,traffic,final,issues=rtl.parse_capture(log,case,synthesis['clock_ns'])
  run.update(returncode=rc,transactions=tx,traffic=traffic,captured=final,issues=issues);rtl.save(folder/'run.json',run)
  need(rc==0 and not issues,'Probe RTL capture/status failed')
  finish_probe_checks(out,probe,record,folder,run,expected,c_expected,folder/'run.json')
 except Exception as error:
  run.update(state='stopped',passed=False,error=str(error));rtl.save(folder/'run.json',run);raise
def finish_probe_checks(out,probe,record,folder,run,expected,c_expected,savepath):
 bounds=POLICY['fp_local_bounds'].get(probe['operation'],[0,0]);run['checks']=[];outputs=[];count=len(expected) if probe['op']==1 else len(expected)//4
 for slot in range(2):
  raw=capture(folder/f'actual_{slot}.hex',len(expected));outputs.append(raw);binary=folder/f'actual_{slot}.bin'
  if binary.exists():need(binary.read_bytes()==raw,'Saved actual output differs from hex capture')
  else:binary.write_bytes(raw)
  compare=compare_codes if probe['op']==1 else lambda a,b:compare_float(a,b,*bounds)
  check=dict(slot=slot,same_input_host=compare(raw,expected),c_stage=compare(raw,c_expected))
  if probe['op']==1 and record.get('known_tie_boundary'):
   tie=record['known_tie_boundary'];check['known_tie_boundary']=dict(tie,captured_rtl_a8=signed(raw)[tie['head_channel']],input_kind=probe['inputs_kind'])
  if probe['op']==6:check['row_sum_error']=abs(math.fsum(floats(raw))-1.)
  if probe.get('downstream_a8'):
   downstream=probe['downstream_a8'];codes=fp_reference(dict(OP=1,ROWS=1,WIDTH=count,OUTPUTS=0),dict(x=raw),downstream['scale'])
   check['downstream_a8']=codes_comparison(codes,(out/probe['directory']/'expected_downstream_a8.bin').read_bytes(),raw,downstream['scale'],downstream.get('source_code_offset',0))
   if probe['name']=='av' and record.get('known_tie_boundary'):
    tie=record['known_tie_boundary'];channel=tie['head_channel'];value=floats(raw)[channel]
    check['known_tie_boundary']=dict(tie,actual_rtl_context=value,actual_rtl_scaled=f32(value/downstream['scale']),same_input_host_a8_from_rtl_context=signed(codes)[channel],c_scaled=f32(tie['reconstructed_c_context']/downstream['scale']),half_boundary=42.5)
  run['checks'].append(check);rtl.save(savepath,run)
  need(check.get('row_sum_error',0)<=2e-5,'Softmax normalization discrepancy')
  need(check['same_input_host']['passed'] and check['c_stage']['passed'],'Probe local FP bound exceeded; diagnose before further probes')
  need(not check.get('downstream_a8') or check['downstream_a8']['passed'],'Diagnostic pause: saved C downstream A8 differs; acceptance remains unresolved')
 run['singleton_slot_bytes_equal']=outputs[0]==outputs[1];need(run['singleton_slot_bytes_equal'],'Repeated probe output differs')
 run.update(state='returned',passed=True);rtl.save(savepath,run)
 print(json.dumps(dict(passed=True,probe=probe['name'],record=str(savepath),cycles=[v['latency_cycles'] for v in run['transactions']],checks=run['checks']),indent=2))

def check_saved_probe(args):
 """Repair/check preserved successful captures without a simulator relaunch."""
 out=args.output.resolve();record=read(out/'probes.json');probe=next(v for v in record['probes'] if v['name']==args.probe)
 folder=out/probe['directory']/'rtl';saved=read(folder/'run.json');savepath=folder/'postcheck.json'
 need(not savepath.exists(),'Preserve existing postcheck')
 need(saved['returncode']==0,'Successful actual simulator return required')
 for item in record['source_inputs']:need(Path(item['source']).read_bytes()==(out/item['file']).read_bytes(),'Probe source changed')
 config=read(ROOT/'configs/project.json');source,synthesis=rtl.source_evidence(config)
 need(synthesis['recorded_utc']==record['fp_synthesis'],'Probe synthesis association changed')
 case=dict(op=probe['op'],rows=1,width=probe['width'],outputs=probe['outputs'])
 tx,traffic,final,issues=rtl.parse_capture((folder/'simulation.log').read_text(errors='replace'),case,synthesis['clock_ns'])
 need(not issues,'Preserved capture status failed')
 arrays={p:(folder/(p+'.bin')).read_bytes() for p in ('x','y','z','accumulators')}
 a=dict(OP=probe['op'],ROWS=1,WIDTH=probe['width'],OUTPUTS=probe['outputs'],DOT_BIAS=False)
 expected=fp_reference(a,arrays,probe['scale']);need(expected==(folder/'expected_same_input_host.bin').read_bytes(),'Preserved same-input reference differs')
 run=dict(saved,capture_run=str(folder/'run.json'),prior_checker_error=saved.get('error'),checker_only=True)
 run.pop('error',None);run.update(state='checking',passed=False,transactions=tx,traffic=traffic,captured=final,issues=issues)
 rtl.save(savepath,run)
 try:finish_probe_checks(out,probe,record,folder,run,expected,(folder/'expected_c.bin').read_bytes(),savepath)
 except Exception as error:
  run.update(state='stopped',passed=False,error=str(error));rtl.save(savepath,run);raise
def main():
 ap=argparse.ArgumentParser(description=__doc__);mode=ap.add_mutually_exclusive_group()
 mode.add_argument('--execute-next',action='store_true');mode.add_argument('--check-connected',type=Path);mode.add_argument('--check-trace',type=Path)
 mode.add_argument('--replay',action='store_true');mode.add_argument('--preview-replay',action='store_true')
 mode.add_argument('--prepare-probes',action='store_true');mode.add_argument('--prepare-quant-lengths',action='store_true');mode.add_argument('--probe',choices=['softmax','av','gelu','tanh','quantize','softmax128','av128','softmax512','av512'])
 ap.add_argument('--case-directory',type=Path,default=DATA/'development_qualification_w8_v1/case01/window000')
 ap.add_argument('--probe-layer',type=int,default=0);ap.add_argument('--probe-row',type=int,default=0);ap.add_argument('--probe-head',type=int,default=0)
 ap.add_argument('--attention-only',action='store_true');ap.add_argument('--check-probe',action='store_true')
 ap.add_argument('--resume',action='store_true');ap.add_argument('--max-services',type=int,default=1)
 ap.add_argument('--fp-rows',type=int,default=128);ap.add_argument('--integer-rows',type=int,default=128)
 ap.add_argument('--output',type=Path)
 ap.add_argument('--rows',type=int,default=16);ap.add_argument('--check-output',type=Path)
 ap.add_argument('--diagnostic-plan',type=Path,help='Explicit reviewed reduced prefix16 plan; valid only with --check-trace')
 args=ap.parse_args()
 need(args.diagnostic_plan is None or args.check_trace is not None,'--diagnostic-plan requires --check-trace')
 if args.output is None:args.output=DATA/('rtl_quant_and_length_probes_v2' if args.prepare_quant_lengths or args.probe in ('quantize','softmax128','av128','softmax512','av512') else 'rtl_c_stage_probes_v1' if args.prepare_probes or args.probe else 'rtl_detector_replay_v1' if args.replay or args.preview_replay else 'rtl_query_integer_rows16_v1')
 if args.prepare_quant_lengths:prepare_quant_length_probes(args)
 elif args.prepare_probes:prepare_probes(args)
 elif args.probe:check_saved_probe(args) if args.check_probe else execute_probe(args)
 elif args.replay or args.preview_replay:
  need(args.max_services>0,'max-services must be positive');replay=ArenaReplay(args)
  if args.replay:replay.execute()
  else:replay.preview()
 elif args.execute_next:execute_next(args)
 elif args.check_connected:check_connected(args.check_connected,args.check_output)
 elif args.check_trace:check_trace(args.check_trace,args.check_output,args.diagnostic_plan)
 else:
  _,plan,prep,_,_,synthesis,_,integer=associations()
  print(json.dumps(dict(execute=False,case=prep['case'],policy=POLICY,next_stage='layer0_query_integer_rows0..15',
   prefix_calls_reused=6,new_synthesis=False,compile_capture_harness=True,command_watchdog_seconds=900,
   fp_synthesis=synthesis['recorded_utc'],integer_synthesis=integer['recorded_utc'],
   complete_connected_runner='scripts/run_detector_arithmetic.py',
   complete_replay_estimate='Historical FP-only host simulation estimate ~18.3 CPU hours; integer/integration extra',
   full_replay_completion=False),indent=2))
if __name__=='__main__':main()