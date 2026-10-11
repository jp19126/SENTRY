"""Opt-in true-L128/L512 floating/W8 diagnostic adapter; preview by default.

No compile/export/model path. Only new guarded dispatches can acquire classes.
The unchanged shape inspector runs first; failed candidates remain numerical stops.
"""
import argparse,ast,copy,importlib.util,json,math,os,struct
from contextlib import contextmanager
from decimal import Decimal,localcontext
from pathlib import Path
import bringup_detector_shapes as shapes
import run_qualification_cases as guards
from run_precision_qualification import w4_padding_literal_functions as literals
ROOT=shapes.ROOT
REVIEW=ROOT/'results/goal4/numerical_bridge/connected_bounds_review_v2'
HOME=shapes.DATA/'shape_adapter_source_v1'
SCHEMA='sentry_true_length_shape_adapter_policy_v1'
CLASS='true_l128_l512_floating_w8_exact_discrete_captured_tail_v1'
SCOPES=((128,'floating'),(128,'w8'),(512,'floating'),(512,'w8'))
SHAPE_FUNCTIONS=('documents','load_plan','raw','spec','model_and_parameters','cpu_stage','shape_replacements','generate_shape_tb','shape_tcl','same_functions','same_files','validate_shape_inputs','validate_shape_build','validate_shape_process','shape_case','shape_discrepancy','shape_documents','shape_compilation_completed','c_stage_locked','c_stage')
GUARD_FUNCTIONS=('run','paused','ConnectedCMemoryGuard','generate_bounds_tb','validate_operator_bounds','validate_reviewed_resolution')
_LOADED_SOURCE=Path(__file__).read_text(encoding='utf-8-sig')
_context=None

def need(ok,why):
 if not ok:raise RuntimeError(why)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def relative(p):return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()
def save(p,v):
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def load_math(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def source_binding():
 return dict(adapter_source=_LOADED_SOURCE,shape_functions=literals((ROOT/'scripts/bringup_detector_shapes.py').read_text(encoding='utf-8-sig'),SHAPE_FUNCTIONS),guard_functions=literals((ROOT/'scripts/run_qualification_cases.py').read_text(encoding='utf-8-sig'),GUARD_FUNCTIONS),math_helper_sources={p.name:p.read_text(encoding='utf-8-sig') for p in (REVIEW/'saved_ln_boundary.py',REVIEW/'saved_w8_propagation.py')})

def scope_records():
 records={}
 for length,mode in SCOPES:
  out=shapes.mode_dir(length,mode);cpu=read(out/'cpu.json');plan=read(shapes.plan_path(length));build=ROOT/'build'/f'detector_shape_bringup_l{length}_{mode}_v1';template=read(build/'template.json');compiled=read(build/'compiled.json')
  need(cpu['length']==length and cpu['mode']==mode and cpu['plan']==plan and cpu['b1_complete'] is True and cpu['batches_complete'] is True and cpu['score_acceptance'] is None,'Own completed CPU identity/plan/batch scope required')
  need(template['identity']==cpu['identity'] and template['b1']==cpu['b1'] and compiled['passed'] is True and compiled['template']==template,'Own successful compiled-template association required')
  records[f'l{length}_{mode}']=dict(length=length,mode=mode,out=relative(out),cpu={k:cpu[k] for k in ('length','mode','identity','settings','b1_complete','batches_complete','score_acceptance')},plan=plan,template=template,compiled={k:v for k,v in compiled.items() if k!='template'})
 return records

def policy_expected():
 return dict(schema=SCHEMA,dispatch_class=CLASS,source_binding=source_binding(),scopes=scope_records(),numerical_acceptance=None,score_approval=None)

def validate_policy(path):
 p=Path(path);p=p if p.is_absolute() else ROOT/p;relative(p);record=read(p);expected=policy_expected()
 need(all(k in record and record[k]==v for k,v in expected.items()),'Exact four-scope policy/source/CPU/compiled association differs')
 need(record.get('coordinator_adopted') is True and record.get('allow_prospective_dispatch') is True,'Separate explicit root policy adoption required')
 need(isinstance(record.get('reason'),str) and record['reason'].strip() and isinstance(record.get('evidence_paths'),list) and record['evidence_paths'],'Actual root rationale/evidence required')
 for value in record['evidence_paths']:
  pth=ROOT/value;need(not Path(value).is_absolute() and pth.is_file() and pth.resolve()!=p.resolve(),'Actual repo-relative root evidence required');relative(pth)
 need(Path(__file__).read_text(encoding='utf-8-sig')==_LOADED_SOURCE,'Loaded adapter source changed')
 return dict(path=relative(p),record=record)

def classify(args,item,entry,out):
 """Called only after unchanged original shape_case verified all original gates."""
 need((args.length,args.mode) in SCOPES and Path(out).resolve()==shapes.mode_dir(args.length,args.mode).resolve(),'Only exact four declared shapes permitted')
 a,p,q=ROOT/entry['folder'],ROOT/item['folder'],out/'parameters';length=args.length
 need(entry['returncode']==0 and entry['acceptance'] is None and entry['summary']['strict_decision_agrees'] is True,'Original process/strict/null-acceptance required')
 need([v['name'] for v in entry['a8']]==([] if args.mode=='floating' else [f'layer{i}_{name}' for i in range(4) for name in shapes.LINEARS]),'Exact A8 membership required')
 raw=[]
 for v in entry['a8']:
  name=v['name'];ab=(a/('actual_'+name+'_a8.bin')).read_bytes();rb=(p/(name+'_reference_a8.bin')).read_bytes();need(len(ab)==len(rb)==length*shapes.LINEARS[name.split('_',1)[1]],'Full A8 extent required');raw.append(dict(name=name,exact=ab==rb,differences=v['differences']))
 discrete=all(v['exact'] and v['differences']==0 for v in raw)
 m=load_math('_shape_adapter_saved_math',REVIEW/'saved_w8_propagation.py');ln=load_math('_shape_adapter_saved_ln',REVIEW/'saved_ln_boundary.py').classify_embedding_ln(a,p,q,expected_length=length)
 first=next((n for n in ('embedding_ln','layer0','layer1','layer2','layer3') if (a/('bridge_'+n+'.bin')).read_bytes()!=(p/('reference_'+n+'.bin')).read_bytes()),None)
 pre=m.floats(a/'diagnostic_pooler_pre_tanh.bin',256);post=m.floats(a/'diagnostic_pooler_post_tanh.bin',256);ct=m.tail(m.floats(a/'bridge_layer3.bin',length*256)[:256],q);rt=m.tail(m.floats(p/'reference_layer3.bin',length*256)[:256],q);dot=m.differences(pre,ct['pooler_pre'],all_outputs=True)
 records=[json.loads(s) for s in (a/'operator_bounds.jsonl').read_text().splitlines()];tr=[v for v in records if v['op']==8];need(len(tr)==1,'One actual TANH call required');tr=tr[0]
 need(all(tr.get(k)==v for k,v in dict(op_call_index=1,rows=1,width=256,outputs=0,values_checked=256,atol=3e-6,rtol=3e-5,passed=True,exact_formula_required=False).items()) and 'first_violation' not in tr,'Original TANH bounds/shape required')
 tanh=[]
 with localcontext() as ctx:
  ctx.prec=80
  for index,(x,y) in enumerate(zip(pre,post)):
   ex=(2*Decimal.from_float(x)).exp();v=(ex-1)/(ex+1);error=abs(Decimal.from_float(y)-v);bound=Decimal.from_float(tr['atol'])+Decimal.from_float(tr['rtol'])*abs(v);tanh.append(dict(index=index,input=x,actual=y,mathematical=str(v),error=str(error),original_bound=str(bound),passed=error<=bound))
 logits=m.ordered_dot(post,m.floats(q/'classifier_weight.bin',512),m.floats(q/'classifier_bias.bin',2));maximum=max(logits);exps=[m.f32(math.exp(m.f32(v-maximum))) for v in logits];total=0.
 for value in exps:total=m.f32(total+value)
 computed=dict(logits=logits,risk=[m.f32(exps[1]/total)]);tails=[]
 for backend,values,folder,prefix in [('actual_captured_tanh',computed,a,'bridge'),('reference_host',rt,p,'reference')]:
  v=dict(backend=backend)
  for name,count in [('logits',2),('risk',1)]:v[name]=m.differences(values[name],m.floats(folder/(prefix+'_'+name+'.bin'),count),all_outputs=True)
  v['exact']=all(v[n]['changed_values']==0 for n in ('logits','risk'));tails.append(v)
 threshold=m.floats(q/'threshold.bin',1)[0];s=entry['summary'];strict=s['threshold']==threshold and s['strict_decision_agrees'] is True and (s['risk']>threshold)==(s['reference_risk']>threshold)
 predicates=dict(all_raw_discrete_exact=discrete,first_changed_embedding_ln=first=='embedding_ln',all_length_ln_rows_recognized=ln['all_rows_recognized'] is True and ln['rows_checked']==length and ln['values_checked']==length*256 and ln['first_unrecognized'] is None,actual_pre_dot_exact=dot['changed_values']==0,all256_actual_tanh_original_bounds=all(v['passed'] for v in tanh),both_own_terminal_recipes_exact=all(v['exact'] for v in tails),strict_window_decision=strict)
 return dict(schema='sentry_true_length_shape_captured_tail_classification_v1',dispatch_class=CLASS,key=entry['key'],length=length,mode=args.mode,predicates=predicates,raw_a8=raw,first_changed_stage=first,embedding_ln=ln,actual_pre_dot=dot,actual_tanh_original_record=tr,actual_tanh_rows=tanh,actual_host_tanh=m.differences(ct['pooler_tanh'],post,all_outputs=True),tail_reproduction=tails,threshold=threshold,actual_margin=s['risk']-threshold,reference_margin=s['reference_risk']-threshold,recognized=all(predicates.values()),inference_limit='LN mechanism recognized; internal variance/sqrt remains inference. Reference pooler/TANH reconstructed, exp/denominator not observed. Host TANH/exp failures retained; no fitting or epsilon.',numerical_acceptance=None,score_approval=None)

def policy_reference(args,policy):
 scope=policy['record']['scopes'][f'l{args.length}_{args.mode}']
 return dict(path=policy['path'],schema=policy['record']['schema'],dispatch_class=policy['record']['dispatch_class'],length=args.length,mode=args.mode,identity=scope['cpu']['identity'])

def provenance_record(args,item,folder,policy):
 binding=source_binding()
 return dict(schema='sentry_true_length_loaded_shape_adapter_dispatch_v1',length=args.length,mode=args.mode,key=item['key'],identity=policy['record']['scopes'][f'l{args.length}_{args.mode}']['cpu']['identity'],policy=policy_reference(args,policy),source_binding=binding,adapter_source=relative(folder/'dispatch_shape_adapter.py'),policy_snapshot=relative(folder/'dispatch_shape_adapter_policy.json'),helper_snapshots={name:relative(folder/('dispatch_shape_adapter_'+name)) for name in binding['math_helper_sources']},canonical_shape_source=relative(folder/'dispatch_shape_runner.py'),canonical_guard_source=relative(folder/'dispatch_connected_runner.py'),callback_functions=['shape_case_adapter','shape_discrepancy_adapter','run_adapter'],arithmetic_changed=False)

def dispatch_provenance(args,item,folder,policy):
 path=folder/'shape_adapter_dispatch.json';record=read(path);need(record==provenance_record(args,item,folder,policy),'Actual shape adapter provenance differs')
 need((ROOT/record['adapter_source']).read_text(encoding='utf-8-sig')==_LOADED_SOURCE and read(ROOT/record['policy_snapshot'])==policy,'Actual adapter/policy snapshot differs')
 need(literals((ROOT/record['canonical_shape_source']).read_text(encoding='utf-8-sig'),SHAPE_FUNCTIONS)==record['source_binding']['shape_functions'] and literals((ROOT/record['canonical_guard_source']).read_text(encoding='utf-8-sig'),GUARD_FUNCTIONS)==record['source_binding']['guard_functions'],'Actual canonical shape/guard seams differ')
 for name,p in record['helper_snapshots'].items():need((ROOT/p).read_text(encoding='utf-8-sig')==record['source_binding']['math_helper_sources'][name],'Actual helper snapshot differs')
 return dict(path=relative(path),record=record)

def run_adapter(command,cwd,env,log,timeout,memory_guard=None):
 c=_context;need(c is not None and c['args'].stage=='c-run','Actual opted-in shape dispatch only');args=c['args'];out=c['out'];build=c['build']
 need(isinstance(memory_guard,guards.ConnectedCMemoryGuard) and memory_guard.active and memory_guard.output.resolve()==(out/'c').resolve() and len(memory_guard.locks)==2 and all(not f.closed for f in memory_guard.locks),'Actual held active2-lock canonical guard required')
 need(list(command)==[str(build/'project/solution/csim/build/csim.exe')] and Path(cwd).resolve()==build.resolve() and timeout==shapes.C_WINDOW_TIMEOUT==1800,'Existing compiled1800second run only; no compile')
 policy=validate_policy(c['policy']['path']);need(policy==c['policy'],'Policy changed before actual dispatch');folder=Path(log).parent;process=read(folder/'csim_run.json');item=process['window']
 need(item in c['cpu']['b1'] and item['key'] not in c['initial_keys'] and folder.resolve()==(out/'c'/item['key']).resolve(),'Fresh own real window required')
 need(process['state']=='running' and process['identity']==c['cpu']['identity'] and process['length']==args.length and process['mode']==args.mode and process['cpu_affinity'] in ([6],[7]),'Actual original process scope differs')
 need(env.get('GATE_NUMERICAL_BRIDGE_DIR')==shapes.linux(build/'reference_inputs'/item['key']) and env.get('GATE_NUMERICAL_BRIDGE_PARAMETER_DIR')==shapes.linux(build/'parameters') and env.get('GATE_NUMERICAL_BRIDGE_DIAGNOSTIC_DIR')==shapes.linux(folder) and env.get('GATE_NUMERICAL_BRIDGE_CASE')==item['document_id'] and ((env.get('GATE_NUMERICAL_BRIDGE_FLOATING')=='1') if args.mode=='floating' else ('GATE_NUMERICAL_BRIDGE_FLOATING' not in env)),'Original actual C environment differs')
 record=provenance_record(args,item,folder,policy);path=folder/'shape_adapter_dispatch.json';files={ROOT/record['adapter_source']:_LOADED_SOURCE,ROOT/record['policy_snapshot']:json.dumps(policy,indent=2)+'\n'};files.update({ROOT/p:record['source_binding']['math_helper_sources'][n] for n,p in record['helper_snapshots'].items()})
 need(not path.exists() and not any(p.exists() for p in files),'Preserve prior/partial actual adapter provenance')
 for p,text in files.items():
  with p.open('x',encoding='utf-8') as f:f.write(text)
 save(path,record)
 return c['original_run'](command,cwd,env,log,timeout,memory_guard=memory_guard)

def shape_case_adapter(args,item,folder,template):
 c=_context;need(c is not None,'Explicit installed shape adapter required')
 entry=c['original_case'](args,item,folder,template) # ALL original checks first.
 path=folder/'shape_adapter_dispatch.json'
 if not path.exists():return entry # Historical, never retrospectively classified.
 provenance=dispatch_provenance(args,item,folder,c['policy']);entry['shape_adapter_dispatch']=provenance
 if not c['original_discrepancy'](entry):return entry
 artifact=folder/'recognized_shape_exemplar.json'
 try:classified=classify(args,item,entry,c['out'])
 except Exception as error:classified=dict(schema='sentry_true_length_shape_classification_failure_v1',key=item['key'],length=args.length,mode=args.mode,recognized=False,error=str(error),numerical_acceptance=None,score_approval=None)
 if artifact.exists():need(read(artifact)==classified,'Actual saved shape classification changed')
 else:
  need(args.stage=='c-run' and item['key'] not in c['initial_keys'],'Read-only/historical classification cannot create artifacts');save(artifact,classified)
 record=dict(policy=policy_reference(args,c['policy']),classification_path=relative(artifact),provenance=dict(path=provenance['path']),key=item['key'],identity=c['cpu']['identity'],classification=classified,acceptance=None)
 # A false candidate is returned and committed before original numerical stop.
 entry['recognized_shape_exemplar' if classified['recognized'] is True else 'shape_classification_attempt']=record
 return entry

def shape_discrepancy_adapter(entry):
 c=_context;need(c is not None,'Explicit installed shape adapter required');original=c['original_discrepancy'](entry)
 if entry.get('recognized_shape_exemplar'):
  record=entry['recognized_shape_exemplar'];need(original and 'shape_classification_attempt' not in entry and record['classification']['recognized'] is True and record['policy']==policy_reference(c['args'],c['policy']),'Valid current prospective class required')
  return False
 return original

@contextmanager
def installed(args,policy):
 global _context
 need(_context is None and (args.length,args.mode) in SCOPES,'No nested/other scope adapter');out=shapes.mode_dir(args.length,args.mode);cpu=read(out/'cpu.json');build=ROOT/'build'/f'detector_shape_bringup_l{args.length}_{args.mode}_v1';state=read(out/'c.json') if (out/'c.json').exists() else dict(windows=[])
 old_case,old_discrepancy,old_run=shapes.shape_case,shapes.shape_discrepancy,guards.run
 _context=dict(args=args,policy=policy,out=out,cpu=cpu,build=build,initial_keys={v['key'] for v in state['windows']},original_case=old_case,original_discrepancy=old_discrepancy,original_run=old_run)
 shapes.shape_case,shapes.shape_discrepancy,guards.run=shape_case_adapter,shape_discrepancy_adapter,run_adapter
 try:yield
 finally:shapes.shape_case,shapes.shape_discrepancy,guards.run=old_case,old_discrepancy,old_run;_context=None

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--length',type=int,choices=(128,512),required=True);ap.add_argument('--mode',choices=('floating','w8'),required=True);ap.add_argument('--stage',choices=('preview','c-run','c-validate'),default='preview');ap.add_argument('--policy');ap.add_argument('--execute',action='store_true');ap.add_argument('--max-forwards',type=int,default=1);ap.add_argument('--reviewed-resolution');args=ap.parse_args()
 need((args.stage=='c-validate' and 0<=args.max_forwards<=6) or 1<=args.max_forwards<=6,'Original bounded1..6 run or read-only0..6 count required');need(not args.reviewed_resolution or args.stage=='c-run','Casewise attachment only through original c-run')
 if args.stage=='preview' or (args.stage=='c-run' and not args.execute):
  print(json.dumps(dict(execute=False,length=args.length,mode=args.mode,stage=args.stage,policy=args.policy,dispatch_class=CLASS,required='Separate root source review/adopted policy/resource release; original completed compiled shape path',CPU=[6,7],max_workers=2,window_timeout_seconds=1800,compile_allowed=False,historical_classes=False,numerical_acceptance=None,score_approval=None),indent=2));return
 need(os.name!='nt' and args.policy,'WSL and explicit separately adopted policy required');policy=validate_policy(args.policy)
 with installed(args,policy):shapes.c_stage(args)

if __name__=='__main__':main()