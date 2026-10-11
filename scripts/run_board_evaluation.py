"""Later-board execution with a default offline preview and two explicit hardware gates.
Uniform W8 bring-up -> final U/B/C agreement -> repeated completion timing -> matched load.
This script never writes SD media. Pack/provision the documented reserved model slot first.
"""
from pathlib import Path
import argparse,copy,csv,json,math,statistics,struct,subprocess,time
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
HEADER=struct.Struct('<8sIIIIQQQIIII')
def resolve(value):
 p=Path(value);return p if p.is_absolute() else ROOT/p
def json_file(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def bits_float(n):return struct.unpack('<f',struct.pack('<I',n))[0]
def response(raw,model,doc,sample,index,last,maximum,threshold):
 if len(raw)!=64 or raw[:8]!=b'SENTRES1':raise ValueError('Invalid/truncated response')
 version,status=struct.unpack_from('<II',raw,8);rs,rd,rm=struct.unpack_from('<QQQ',raw,16)
 ri,risk,mx,flags=struct.unpack_from('<IIII',raw,40);kernel_ns=struct.unpack_from('<Q',raw,56)[0]
 if version!=1 or status or (rs,rd,rm,ri)!=(sample,doc,model,index) or flags>3 or flags&1!=int(last) or risk>0x3f800000:raise ValueError('Error or identity-mismatched response')
 maximum=max(maximum,risk);reject=last and maximum>threshold
 if mx!=maximum or bool(flags&2)!=reject or not kernel_ns:raise ValueError('Maximum/strict threshold/timestamp mismatch')
 return maximum,dict(risk_bits=risk,maximum_bits=mx,reject=reject,supervisor_submit_to_completion_ns=kernel_ns)
def ready_response(raw,package):
 if len(raw)!=64 or raw[:8]!=b'SENTRYR1':raise ValueError('Supervisor initialization handshake failed')
 version,length,model,threshold,freq=struct.unpack_from('<IIQII',raw,8)
 if version!=1 or length!=package['length'] or model!=package['model_id'] or threshold!=package['threshold_bits'] or not freq:raise ValueError('Ready model/length/threshold/timer mismatch')
 if package.get('timer_frequency_hz') and freq!=package['timer_frequency_hz']:raise ValueError('Ready frequency differs from compiled package')
 return freq
def read_exact(port,n):
 data=bytearray()
 while len(data)<n:
  part=port.read(n-len(data))
  if not part:raise TimeoutError('Serial completion timeout')
  data.extend(part)
 return bytes(data)
def percentile(values,p):
 values=sorted(values);return values[max(0,math.ceil(len(values)*p)-1)] if values else None
def energy(csv_path,begin,end,completed):
 # Relative seconds must share the measurement epoch in run.json; no TDP inputs.
 rows=list(csv.DictReader(resolve(csv_path).open(newline='')))
 if len(rows)<2:raise ValueError('Measured power trace needs at least2 samples')
 result={}
 times=[float(r['time_seconds']) for r in rows]
 if any(not math.isfinite(x) for x in times) or any(b<=a for a,b in zip(times,times[1:])) or times[0]>begin or times[-1]<end:raise ValueError('Power samples must span the matched completion interval in the same epoch')
 for field in ['board_watts','whole_system_watts']:
  if not all(r.get(field,'')!='' for r in rows):result[field.replace('_watts','_joules_per_check')]=None;continue
  powers=[float(r[field]) for r in rows]
  if any(not math.isfinite(p) or p<0 for p in powers):raise ValueError('Invalid measured power')
  joules=0.
  for t0,t1,p0,p1 in zip(times,times[1:],powers,powers[1:]):
   left=max(t0,begin);right=min(t1,end)
   if right>left:
    pl=p0+(p1-p0)*(left-t0)/(t1-t0);pr=p0+(p1-p0)*(right-t0)/(t1-t0);joules+=(pl+pr)*.5*(right-left)
  result[field.replace('_watts','_joules_per_check')]=joules/completed
 return result
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--package',required=True);ap.add_argument('--stage',choices=['program','bringup','agreement','timing','matched-load','prepare-arrivals'],default='bringup')
 ap.add_argument('--workload',help='JSONL document_id/text; references separately frozen from development/search')
 ap.add_argument('--references',help='JSONL document_id,risk_bits[list],maximum_bits,reject from qualified ordered CPU arithmetic')
 ap.add_argument('--length',type=int,choices=[128,256,512],default=256);ap.add_argument('--batch',type=int,choices=[1,8,32],default=1)
 ap.add_argument('--boundary',choices=['pretokenized','raw-text'],default='pretokenized')
 ap.add_argument('--examples',type=int,default=1000);ap.add_argument('--checks',type=int);ap.add_argument('--repeats',type=int);ap.add_argument('--warmup',type=int,default=50)
 ap.add_argument('--port');ap.add_argument('--output');ap.add_argument('--execute-hardware',action='store_true')
 ap.add_argument('--arrival-schedule',help='One of the three common arrivals_25/60/85.csv siblings, checked against actual capacities')
 ap.add_argument('--host-workload',default='results/goal4/host_preparation/l256_b1/run.json',help='Actual32-document packet source shared with native software timing')
 ap.add_argument('--power-csv',help='Measured CSV time_seconds,board_watts,whole_system_watts in run epoch')
 ap.add_argument('--comparison-policy',help='Explicit adopted frozen physical comparison contract; no terminal epsilon default')
 ap.add_argument('--runtime-settings',help='Approved actual unit/media/boot/arena/port/profile settings JSON')
 ap.add_argument('--agreement-run',help='Actual successful1000-example board agreement for this frozen model before timing/load')
 ap.add_argument('--prepare',action='store_true',help='Write non-hardware matched arrival CSVs from supplied measured capacities')
 ap.add_argument('--capacities',help='Actual software+FPGA timing capacities with full_document_checks_per_second/raw_text_to_decision units and evidence')
 ap.add_argument('--hw-server-url');ap.add_argument('--target-filter');ap.add_argument('--xsdb',default='/home/jp19126/Xilinx/2025.2/Vitis/bin/xsdb')
 a=ap.parse_args();package=json_file(resolve(a.package));config=json_file(ROOT/'configs/project.json')
 if package.get('passed') is not True or package['length']!=a.length:raise ValueError('Matching actual packed model required')
 primary=(a.length==256 and a.batch==1)
 if a.checks is None:a.checks=10000 if primary or a.stage in ('timing','matched-load','prepare-arrivals') else 1
 if a.repeats is None:a.repeats=3 if a.stage=='timing' else 1
 if a.examples<1 or a.checks<1 or a.repeats<1 or a.warmup<0:raise ValueError('Invalid measurement extent')
 policy=json_file(resolve(a.comparison_policy)) if a.comparison_policy else None
 policy_ready=False
 if policy:
  if policy.get('adopted') is not True or policy.get('model_id')!=package['model_id'] or policy.get('length')!=a.length or not policy.get('policy_id') or not resolve(policy['qualification_record']).is_file():raise ValueError('Comparison policy must explicitly adopt this frozen model/length and cite actual qualification')
  for rule_name in ['risk_rule','reference_maximum_rule']:
   rule=policy[rule_name]
   if rule['mode'] not in ('exact_bits','qualified_absolute_bound'):raise ValueError('Unsupported adopted comparison rule')
   if rule['mode']=='qualified_absolute_bound' and (type(rule.get('absolute_bound')) not in (int,float) or not math.isfinite(rule['absolute_bound']) or rule['absolute_bound']<0):raise ValueError('Invalid adopted qualified bound')
  policy_ready=True
 if a.stage=='bringup' and package['w8_mask']!=65535:raise ValueError('Bring-up requires uniform W8')
 if a.stage=='agreement' and a.examples<1000:raise ValueError('Final agreement needs approximately1000 distinct development/search examples')
 if a.stage=='timing' and (a.checks<10000 or a.repeats<3):raise ValueError('Every final timing cell requires at least10000 completed checks in each of at least3 repeats')
 if a.stage=='timing':a.checks=((a.checks+a.batch-1)//a.batch)*a.batch
 if a.stage=='matched-load' and (a.length!=256 or not a.arrival_schedule or not a.capacities or a.boundary!='raw-text'):raise ValueError('Matched load requires L256 raw-text boundary, common arrival CSV and actual measured capacities')
 if a.stage in ('matched-load','prepare-arrivals') and a.checks!=10000:raise ValueError('Common matched-load recipe requires exactly10000 ordered document checks')
 rows=[];refs={}
 if a.workload:
  rows=[json.loads(line) for line in resolve(a.workload).read_text(encoding='utf-8').splitlines() if line.strip()]
  if not rows or any(not isinstance(r.get('text'),str) or not isinstance(r.get('document_id'),str) for r in rows):raise ValueError('Expected document_id/text JSONL')
  if len({r['document_id'] for r in rows})!=len(rows):raise ValueError('Source documents must have unique identifiers')
 if a.references:
  reference_rows=[json.loads(line) for line in resolve(a.references).read_text().splitlines() if line.strip()]
  refs={r['document_id']:r for r in reference_rows}
  if len(refs)!=len(reference_rows):raise ValueError('Duplicate reference document identifier')
  if any(not isinstance(r.get('risk_bits'),list) or not r['risk_bits'] or any(type(v) is not int or not 0<=v<=0x3f800000 for v in r['risk_bits']) or type(r.get('maximum_bits')) is not int or not 0<=r['maximum_bits']<=0x3f800000 or type(r.get('reject')) is not bool for r in reference_rows):raise ValueError('Invalid frozen reference risk/maximum/decision format')
 host_docs=None
 if a.stage in ('timing','matched-load'):
  from measure_frozen_software import load_host_docs
  host_docs=load_host_docs(resolve(a.host_workload))
  if a.workload and resolve(a.workload).resolve()!=(ROOT/'data/prepared/search_candidate_scoring.jsonl').resolve():raise ValueError('Timing/load must use the same actual host-preparation development source as native software')
  rows=host_docs
 capacity_evidence=None;validated_schedule=None
 if a.stage in ('prepare-arrivals','matched-load'):
  if not a.capacities:raise ValueError('Actual measured software and FPGA capacities required')
  from measure_frozen_software import validate_capacities,read_arrivals
  capacities,rates,slowest=validate_capacities(resolve(a.capacities))
  capacity_evidence=dict(source_capacities=str(resolve(a.capacities)),measured_rates=rates,slowest_measured_capacity_checks_per_second=slowest,capacity_unit='full_document_checks_per_second',boundary='raw_text_to_decision')
  if a.stage=='matched-load':
   chosen=resolve(a.arrival_schedule).resolve()
   paths=read_arrivals([chosen.parent/f'arrivals_{percent}.csv' for percent in (25,60,85)],slowest)
   if chosen not in [p.resolve() for p in paths]:raise ValueError('Selected common schedule must be one validated25/60/85 sibling')
   validated_schedule=chosen
 required=['physical_board_activity_authorized=true in configs/project.json','separate explicit --execute-hardware request','reserved/provisioned SD model slot; matching frozen model/threshold','actual allocated UART port; correct booted length/profile package']
 preview=dict(stage=a.stage,execute_hardware=a.execute_hardware,physical_board_activity=False,package=str(resolve(a.package)),model_id=package['model_id'],length=a.length,batch=a.batch,batch_semantics='sequential single-core service; batch formation waits for the group, no parallel speedup claimed',boundary=a.boundary,supervisor_interval='A72 submit/poll through completed PL check; input copy/cache/UART excluded, includes A72 control overhead',workload_documents=len(rows),agreement_examples=a.examples,completed_checks=a.checks,repeats=a.repeats,warmup=a.warmup,boot_packaged=package.get('boot_packaged',False),integrated_eligible=package.get('integrated_eligible',False),primary_configuration=primary,final_timing_cell_scope="Conservative adopted scope: at least10000 completed checks per repeat, three repeats, all nine length/batch cells and both boundaries",numerical_prerequisite='adopted frozen comparison policy supplied' if policy_ready else 'unresolved: no adopted physical comparison contract',comparison_policy=str(resolve(a.comparison_policy)) if a.comparison_policy else None,runtime_settings=str(resolve(a.runtime_settings)) if a.runtime_settings else None,agreement_run=str(resolve(a.agreement_run)) if a.agreement_run else None,requirements=required,sequence=['uniform-W8 bringup','U/B/C per arithmetic-changing build agreement on distinct development/search examples','warmup then all nine L128/256/512 x B1/8/32 cells >=10000 completed checks x3 per boundary; both boundaries required','L256 matched arrivals at25/60/85 percent slowest measured capacity; measured power integration'])
 if a.stage=='timing':
  preview['completed_checks_per_repeat']=a.checks;preview['requested_total_completed_checks']=a.checks*a.repeats
  preview['final_timing_grid']=[dict(length=L,batch=b,completed_checks_per_repeat=((10000+b-1)//b)*b,repeats=3,boundaries=['pretokenized','raw-text']) for L in (128,256,512) for b in (1,8,32)]
  preview['future_timing_commands']=[f"python scripts/run_board_evaluation.py --stage timing --package <approved-packed-model-L{L}.json> --length {L} --batch {b} --boundary {boundary} --checks {((10000+b-1)//b)*b} --repeats 3 --warmup {a.warmup} --host-workload {a.host_workload} --runtime-settings <approved-runtime-L{L}.json> --agreement-run <actual-agreement-L{L}.json> --output <fresh-L{L}-B{b}-{boundary}> --execute-hardware" for L in (128,256,512) for b in (1,8,32) for boundary in ('pretokenized','raw-text')]
 if a.stage in ('bringup','agreement'):preview.update(completed_checks=min(len(rows),1 if a.stage=='bringup' else a.examples),repeats=1)
 if host_docs is not None:preview.update(host_workload=str(resolve(a.host_workload)),source_document_ids_in_preparation_order=[r['document_id'] for r in host_docs],workload_selection='same32host-preparation source/order as software, then complete single-window subset at requested length; payload revalidated before physical execution')
 if capacity_evidence:preview.update(capacity_evidence)
 if a.stage=='prepare-arrivals':
  if a.execute_hardware:raise ValueError('Arrival preparation is offline')
  preview.update(load_fractions=[.25,.60,.85])
  if a.prepare:
   if not a.output:raise ValueError('Arrival preparation requires fresh output')
   out=resolve(a.output)
   if out.exists():raise ValueError('Preserve prior schedules')
   out.mkdir(parents=True)
   for fraction in [.25,.60,.85]:
    with (out/f'arrivals_{int(fraction*100)}.csv').open('w',newline='') as f:
     writer=csv.writer(f);writer.writerow(['check_index','arrival_seconds'])
     writer.writerows((i,format(i/(slowest*fraction),'.12f')) for i in range(a.checks))
   (out/'preparation.json').write_text(json.dumps(dict(preview,physical_board_activity=False),indent=2)+'\n')
  print(json.dumps(preview,indent=2));return
 if a.prepare:raise ValueError('--prepare applies only to prepare-arrivals; model packaging has its own offline command')
 if not a.execute_hardware:print(json.dumps(preview,indent=2));return
 # This check occurs before serial import/open, XSDB or any device operation.
 if config['execution_control']['board_free_continuation'].get('physical_board_activity_authorized') is not True:raise PermissionError('Physical board activity is not authorized in shared configuration')
 if a.stage=='program' and config['hardware'].get('programming_authorized') is not True:raise PermissionError('Programming is not authorized in shared hardware configuration')
 if not package.get('boot_packaged'):raise ValueError('Actual implemented+host BOOT.BIN package required')
 if package.get('integrated_eligible') is not True or not package.get('integrated_closure') or not resolve(package['integrated_closure']).is_file():raise ValueError('Actual routed timing/resource/bandwidth eligibility is unresolved')
 if a.stage in ('bringup','agreement') and not policy_ready:raise ValueError('Numerical prerequisite unresolved: no adopted frozen physical comparison contract')
 if not a.runtime_settings:raise ValueError('Approved actual physical runtime settings required')
 runtime=json_file(resolve(a.runtime_settings))
 if runtime.get('approved') is not True or not runtime.get('unit_id') or not runtime.get('board_revision') or not runtime.get('sd_media_identity') or runtime.get('sd_slot_reserved') is not True or runtime.get('arena_allocation_confirmed') is not True or runtime.get('arena_base')!=package['deployment']['arena_base'] or runtime.get('profile_id')!=package.get('profile_id') or resolve(runtime['boot_package']).resolve()!=resolve(a.package).resolve() or runtime.get('timer_frequency_verified') is not True or runtime.get('timer_frequency_hz')!=package.get('timer_frequency_hz'):raise ValueError('Actual unit/media/allocation/boot/profile/timer settings are unresolved or differ from compiled package')
 if a.port and a.port!=runtime.get('uart_port'):raise ValueError('Port differs from approved allocated UART')
 a.port=runtime.get('uart_port')
 if not a.port:raise ValueError('Approved allocated UART missing')
 if a.stage=='program' and (a.hw_server_url!=runtime.get('hw_server_url') or a.target_filter!=runtime.get('target_filter')):raise ValueError('Programming target differs from approved allocated target')
 agreement=None
 if a.stage in ('timing','matched-load'):
  if not a.agreement_run:raise ValueError('Actual successful approximately1000-example board agreement required before timing/load')
  agreement=json_file(resolve(a.agreement_run))
  if agreement.get('passed') is not True or agreement.get('stage')!='agreement' or agreement.get('physical_board_activity') is not True or agreement.get('model_id')!=package['model_id'] or agreement.get('length')!=a.length or agreement.get('completed',0)<1000 or agreement.get('unique_source_documents',0)<1000 or not agreement.get('adopted_comparison_policy'):raise ValueError('Agreement record does not qualify this frozen model/length')
 if not a.output:raise ValueError('Hardware run requires a fresh output directory')
 out=resolve(a.output)
 if out.exists():raise ValueError('Preserve earlier hardware evidence; choose fresh output')
 out.mkdir(parents=True);record=dict(preview,prior_board_agreement=str(resolve(a.agreement_run)) if agreement else None,approved_runtime_settings=runtime,adopted_comparison_policy=policy,physical_board_activity=True,passed=False,state='running',recorded_utc=datetime.now(timezone.utc).isoformat())
 try:
  if a.stage=='program':
   if not a.hw_server_url or not a.target_filter:raise ValueError('Programming requires the allocated hw_server URL and exact target filter')
   if any(c in a.hw_server_url+a.target_filter for c in '{}\r\n'):raise ValueError('Invalid Tcl literal')
   boot=resolve(a.package).parent/'BOOT.BIN'
   if any(c in str(boot) for c in '{}\r\n'):raise ValueError('Invalid boot-image Tcl literal')
   if not boot.is_file():raise ValueError('Actual BOOT.BIN missing')
   tcl=out/'program.tcl';tcl.write_text('connect -url {'+a.hw_server_url+'}\ntargets -set -filter {'+a.target_filter+'}\ndevice program {'+str(boot).replace('\\','/')+'}\nexit\n')
   r=subprocess.run([a.xsdb,str(tcl)],text=True,capture_output=True,timeout=300);(out/'program.log').write_text(r.stdout+r.stderr)
   if r.returncode:raise RuntimeError('Programming failed; inspect preserved log')
   record.update(passed=True,state='programmed');return
  if not a.port or not rows:raise ValueError('Explicit allocated port and workload required')
  if a.stage in ('bringup','agreement') and (not refs or any(r['document_id'] not in refs for r in rows[:a.examples if a.stage=='agreement' else 1])):raise ValueError('Frozen qualified numerical references required')
  if a.stage=='agreement' and len(rows)<a.examples:raise ValueError('Insufficient distinct development/search examples; do not duplicate agreement evidence')
  from transformers import AutoTokenizer
  import serial
  from gate_eval import window_document
  from prepare_host_workload import packet
  cfg=copy.deepcopy(config);cfg['windowing']['primary_length']=a.length
  tokenizer=AutoTokenizer.from_pretrained(ROOT/'checkpoints/floating/epoch_3',local_files_only=True,use_fast=True)
  def tokenize(row):return window_document(tokenizer,row['text'],row['document_id'],config=cfg)
  if a.stage in ('timing','matched-load'):
   from measure_frozen_software import validate_host_payload,workload
   record['host_payload_validation']=validate_host_payload(tokenizer,resolve(a.host_workload),host_docs,config)
   selected,selected_windows=workload(tokenizer,resolve(a.host_workload),host_docs,config,a.length)
   cached={r['document_id']: [window] for r,window in selected_windows} if a.boundary=='pretokenized' else {}
   record['source_document_ids_in_cycle']=[r['document_id'] for r in selected]
  else:
   selected=rows[:a.examples if a.stage=='agreement' else 1]
   cached={r['document_id']:tokenize(r) for r in selected} if a.boundary=='pretokenized' else {}
  schedule=[]
  if validated_schedule:schedule=list(csv.DictReader(validated_schedule.open(newline='')))
  sample=0;numeric_doc=0;latencies=[];kernel=[];completed=0
  with serial.Serial(a.port,115200,bytesize=8,parity='N',stopbits=1,timeout=120,write_timeout=120) as port,(out/'checks.jsonl').open('w') as trace:
   cold=time.perf_counter();port.write(bytes([0xa5]));port.flush();record['ready_timer_frequency_hz']=ready_response(read_exact(port,64),package);record['hello_to_ready_seconds']=time.perf_counter()-cold
   def check(row,arrival=None,warm=False):
    nonlocal sample,numeric_doc,completed
    numeric_doc+=1;begin=time.perf_counter();windows=cached.get(row['document_id']) or tokenize(row);maximum=0;window_results=[]
    for i,w in enumerate(windows):
     last=i==len(windows)-1;blob=packet(w,a.length,numeric_doc,sample,last,package['model_id'],package['threshold_bits']);port.write(blob);port.flush();raw=read_exact(port,64)
     try:maximum,res=response(raw,package['model_id'],numeric_doc,sample,i,last,maximum,package['threshold_bits'])
     except BaseException:
      (out/'invalid_response.bin').write_bytes(raw);(out/'invalid_response_context.json').write_text(json.dumps(dict(source_document_id=row['document_id'],numeric_document_id=numeric_doc,sample=sample,window=i,expected_last=last,actual_response_hex=raw.hex()),indent=2)+'\n');raise
     window_results.append(res);sample+=1
    finish=time.perf_counter()
    if a.stage in ('bringup','agreement'):
     ref=refs[row['document_id']]
     def adopted_match(actual,expected,rule):
      return actual==expected if rule['mode']=='exact_bits' else abs(bits_float(actual)-bits_float(expected))<=rule['absolute_bound']
     differences=[dict(window=i,actual_risk_bits=v['risk_bits'],reference_risk_bits=expected,absolute_error=abs(bits_float(v['risk_bits'])-bits_float(expected)),accepted=adopted_match(v['risk_bits'],expected,policy['risk_rule'])) for i,(v,expected) in enumerate(zip(window_results,ref['risk_bits']))]
     max_accepted=adopted_match(maximum,ref['maximum_bits'],policy['reference_maximum_rule']);decision_accepted=(window_results[-1]['reject'] is ref['reject'])
     comparison=dict(warmup=warm,source_document_id=row['document_id'],actual_windows=window_results,reference_risk_bits=ref['risk_bits'],window_count_matches=(len(ref['risk_bits'])==len(window_results)),risk_differences=differences,actual_maximum_bits=maximum,reference_maximum_bits=ref['maximum_bits'],maximum_absolute_error=abs(bits_float(maximum)-bits_float(ref['maximum_bits'])),maximum_accepted=max_accepted,actual_reject=window_results[-1]['reject'],reference_reject=ref['reject'],strict_decision_matches=decision_accepted)
     with (out/'agreement.jsonl').open('a') as evidence:evidence.write(json.dumps(comparison)+'\n')
     if not comparison['window_count_matches'] or not all(v['accepted'] for v in differences) or not max_accepted or not decision_accepted:raise ValueError('First unexplained board/reference difference; preserved actual risks/maxima and strict decisions')
    if warm:return
    completed+=1;latency=finish-(begin if arrival is None else arrival);latencies.append(latency*1000);kernel.append(sum(v['supervisor_submit_to_completion_ns'] for v in window_results)/1e6)
    trace.write(json.dumps(dict(check_index=completed-1,source_document_id=row['document_id'],numeric_document_id=numeric_doc,arrival_seconds=(arrival or begin)-epoch,begin_seconds=begin-epoch,completion_seconds=finish-epoch,decision_latency_ms=latency*1000,windows=window_results))+'\n');trace.flush()
   for i in range(a.warmup):check(selected[i%len(selected)],warm=True)
   epoch=time.perf_counter();record['measurement_epoch_utc']=datetime.now(timezone.utc).isoformat();first_begin=0.
   count=len(selected) if a.stage in ('bringup','agreement') else a.checks
   repeats=a.repeats if a.stage=='timing' else 1;record['repeat_summaries']=[]
   for repeat in range(repeats):
    rep_start=time.perf_counter();before=completed
    for group in range(0,count,a.batch):
     items=list(range(group,min(count,group+a.batch)))
     if schedule:
      arrivals=[epoch+float(schedule[i]['arrival_seconds']) for i in items];wait=arrivals[-1]-time.perf_counter()
      if wait>0:time.sleep(wait)
     else:arrivals=[time.perf_counter()]*len(items)
     for i,arrival in zip(items,arrivals):check(selected[i%len(selected)],arrival=arrival)
    duration=time.perf_counter()-rep_start;record['repeat_summaries'].append(dict(repeat=repeat,completed=completed-before,seconds=duration,checks_per_second=(completed-before)/duration,median_decision_latency_ms=statistics.median(latencies[before:]),p95_decision_latency_ms=percentile(latencies[before:],.95),median_supervisor_submit_to_completion_ms=statistics.median(kernel[before:])))
   end=time.perf_counter()-epoch
  record.update(passed=True,state='completed',final_timing_cell_qualified=(a.stage=='timing' and len(record['repeat_summaries'])>=3 and all(x['completed']>=10000 for x in record['repeat_summaries'])),completed=completed,median_decision_latency_ms=statistics.median(latencies),p95_decision_latency_ms=percentile(latencies,.95),median_supervisor_submit_to_completion_ms=statistics.median(kernel),energy=None,unique_source_documents=len(selected),timing_resolution_ns=package.get('timing_resolution_ns'),cold_model_load_included=False)
  if a.power_csv:
   if a.stage!='matched-load':raise ValueError('Energy input is accepted only for matched arrival runs')
   record['energy']=energy(a.power_csv,first_begin,end,completed)
 except BaseException as exc:record.update(passed=False,state='failed',error=str(exc));raise
 finally:(out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':main()
