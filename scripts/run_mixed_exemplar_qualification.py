"""Prospective saved-byte diagnostics for ONE frozen mixed PTQ exemplar.
Canonical precision execution/guards remain unchanged; no policy is authored here.
"""
from pathlib import Path
from contextlib import contextmanager
from decimal import Decimal, localcontext
import argparse, importlib.util, json, math, os, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_precision_qualification as canonical
import run_qualification_cases as guards
from export_precision_qualification import need,read,save,validate_export
NAME='campaign_ptq_p4488884884844448'
TUPLE='4488884884844448'
OUT=ROOT/'results/goal4/numerical_bridge/precision_qualification_v1'/NAME
REVIEW=ROOT/'results/goal4/numerical_bridge/connected_bounds_review_v2'
POLICY_SCHEMA='sentry_mixed_ptq_captured_tail_dispatch_policy_v1'
CLASS='campaign_ptq_p4488884884844448_l256_all_a8_exact_captured_tail_v1'
FINAL_STOP='First propagated A8/terminal difference requires explicit causal review; diagnostic stop is not universal terminal-bit acceptance'
CANONICAL_FUNCTIONS=('inspect','execute','execute_locked','prepared','generate_precision_tb','review_resolution','document_maxima')
GUARD_FUNCTIONS=('run','paused','ConnectedCMemoryGuard','generate_bounds_tb','validate_operator_bounds')
_LOADED_SOURCE=Path(__file__).read_text(encoding='utf-8-sig')
_context=None


def relative(path):
 return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def load_math(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def source_binding():
 literal=canonical.w4_padding_literal_functions
 return dict(adapter_source=_LOADED_SOURCE,
  canonical_functions=literal((ROOT/'scripts/run_precision_qualification.py').read_text(encoding='utf-8-sig'),CANONICAL_FUNCTIONS),
  guard_functions=literal((ROOT/'scripts/run_qualification_cases.py').read_text(encoding='utf-8-sig'),GUARD_FUNCTIONS),
  exporter_source=(ROOT/'scripts/export_precision_qualification.py').read_text(encoding='utf-8-sig'),
  math_helper_sources={p.name:p.read_text(encoding='utf-8-sig') for p in (REVIEW/'saved_ln_boundary.py',REVIEW/'saved_w8_propagation.py')})


def scope(exports,out=OUT):
 need(Path(out).resolve()==OUT.resolve(),'Only the declared mixed exemplar directory is permitted')
 i=exports['identity'];names=[f'layer{d}.{g}' for d in range(4) for g in ('qkv','attention_output','ffn_input','ffn_output')]
 need(all(type(v) is int for v in i['precision_map'].values()) and i['precision_map']==dict(zip(names,map(int,TUPLE))) and exports['precision_map']==i['precision_map'],'Exact mixed map required')
 need(i['qat'] is False and i['precision_source']=='ptq' and i['qat_checkpoint'] is None and 'scored_checkpoint' not in i,'PTQ only; no QAT coverage')
 need(i['windowing']['primary_length']==256 and i['arithmetic_identity']=='bert_mini_ordered_fp32_frozen_quant_v1' and i['storage_contract']=='decoded_signed_i8_codes_plus_real_packed_w4w8_v1','Actual L256 arithmetic/storage differs')
 need(exports['complete'] is True and len(exports['completed_windows'])==66 and exports['score_acceptance'] is None,'Own complete66 B1 exports/null approval required')
 selection=exports['selection'];need(len(selection['cases'])==32 and sum(c['window_count'] for c in selection['cases'])==66 and selection==read(ROOT/'results/goal4/numerical_bridge/development_cases_v1/selection.json'),'Frozen32/66 selection differs')
 need(exports['parameter_folder']==relative(OUT/'parameters') and len(exports['layers'])==24,'Own parameters/24 sites required')
 run=read(ROOT/'results/goal3/run.json');need(i['checkpoint']==run['selected_checkpoint'] and exports['calibration']==read(ROOT/i['calibration_file']),'Common selected checkpoint/calibration differs')
 return exports


def validate_policy(path,exports):
 path=Path(path);path=path if path.is_absolute() else ROOT/path;relative(path);record=read(path);scope(exports)
 expected=dict(schema=POLICY_SCHEMA,policy_id='bounded_connected_development_v2',dispatch_class=CLASS,mode='mixed_ptq_l256',name=NAME,precision_tuple=TUPLE,
  identity=exports['identity'],selection=exports['selection'],precision_layers=exports['layers'],reference_export=relative(OUT/'export.json'),parameter_folder=exports['parameter_folder'],
  map_source=exports['map_source'],precision_map=exports['precision_map'],calibration=exports['calibration'],frozen_scale_policy=exports['identity']['frozen_scale_policy'],source_binding=source_binding(),numerical_acceptance=None,score_approval=None)
 need(all(k in record and record[k]==v for k,v in expected.items()),'Mixed policy exact source/input/map association differs')
 need(record.get('coordinator_adopted') is True and record.get('allow_prospective_dispatch') is True,'Explicit root prospective adoption required')
 need(isinstance(record.get('reason'),str) and record['reason'].strip() and isinstance(record.get('evidence_paths'),list) and record['evidence_paths'],'Actual root rationale/evidence required')
 for value in record['evidence_paths']:
  p=ROOT/value;need(not Path(value).is_absolute() and p.is_file() and p.resolve()!=path.resolve(),'Missing/outside mixed evidence');relative(p)
 need(Path(__file__).read_text(encoding='utf-8-sig')==_LOADED_SOURCE,'Actually loaded adapter source changed')
 return dict(path=relative(path),record=record)


def classify(entry,exports):
 """Original inspect must pass first. No mode relabel, alternate recipe or fitting."""
 scope(exports);a,p,q=ROOT/entry['folder'],ROOT/entry['input_folder'],ROOT/exports['parameter_folder']
 need(len(entry['a8'])==24 and all(v['differences']==0 for v in entry['a8']),'Any propagated A8 change requires exact-case review')
 for meta in exports['layers']:
  name=meta['name'];ab=(a/('actual_'+name+'_a8.bin')).read_bytes();rb=(p/(name+'_reference_a8.bin')).read_bytes();need(len(ab)==len(rb)==256*meta['k'] and ab==rb,'Actual raw A8 must be byte-exact: '+name)
 need(not entry['logits_equal'] or not entry['risk_equal'],'Exact outcomes require no mechanism class')
 need(entry['acceptance'] is None and entry['returncode']==0,'No process/acceptance waiver')
 m=load_math('_mixed_saved_math',REVIEW/'saved_w8_propagation.py');ln=load_math('_mixed_saved_ln',REVIEW/'saved_ln_boundary.py').classify_embedding_ln(a,p,q,expected_length=256)
 first=next((n for n in ('embedding_ln','layer0','layer1','layer2','layer3') if (a/('bridge_'+n+'.bin')).read_bytes()!=(p/('reference_'+n+'.bin')).read_bytes()),None)
 pre=m.floats(a/'diagnostic_pooler_pre_tanh.bin',256);post=m.floats(a/'diagnostic_pooler_post_tanh.bin',256)
 ct=m.tail(m.floats(a/'bridge_layer3.bin',65536)[:256],q);rt=m.tail(m.floats(p/'reference_layer3.bin',65536)[:256],q);dot=m.differences(pre,ct['pooler_pre'],all_outputs=True)
 records=[json.loads(s) for s in (a/'operator_bounds.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()];tr=[v for v in records if v['op']==8];need(len(tr)==1,'Exactly one actual TANH call');tr=tr[0]
 need(all(tr.get(k)==v for k,v in dict(call_index=409,op_call_index=1,rows=1,width=256,outputs=0,values_checked=256,atol=3e-6,rtol=3e-5,passed=True,exact_formula_required=False).items()) and 'first_violation' not in tr,'Original TANH record differs')
 tanh=[]
 with localcontext() as ctx:
  ctx.prec=80
  for j,(x,y) in enumerate(zip(pre,post)):
   ex=(2*Decimal.from_float(x)).exp();value=(ex-1)/(ex+1);error=abs(Decimal.from_float(y)-value);bound=Decimal.from_float(tr['atol'])+Decimal.from_float(tr['rtol'])*abs(value)
   tanh.append(dict(index=j,input=x,actual=y,mathematical=str(value),error=str(error),original_bound=str(bound),passed=error<=bound))
 logits=m.ordered_dot(post,m.floats(q/'classifier_weight.bin',512),m.floats(q/'classifier_bias.bin',2));maximum=max(logits);exps=[m.f32(math.exp(m.f32(v-maximum))) for v in logits];total=0.
 for value in exps:total=m.f32(total+value)
 computed=dict(logits=logits,risk=[m.f32(exps[1]/total)]);tails=[]
 for backend,values,folder,prefix in [('actual_captured_tanh',computed,a,'bridge'),('reference_host',rt,p,'reference')]:
  item=dict(backend=backend)
  for name,count in [('logits',2),('risk',1)]:item[name]=m.differences(values[name],m.floats(folder/(prefix+'_'+name+'.bin'),count),all_outputs=True)
  item['exact']=all(item[n]['changed_values']==0 for n in ('logits','risk'));tails.append(item)
 s=entry['summary'];threshold=m.floats(q/'threshold.bin',1)[0];strict=s['strict_decision_agrees'] is True and s['threshold']==threshold and (s['risk']>threshold)==(s['reference_risk']>threshold)
 recognized=(ln['all_rows_recognized'] is True and ln['rows_checked']==256 and ln['values_checked']==65536 and ln['first_unrecognized'] is None and first=='embedding_ln' and dot['changed_values']==0 and all(v['passed'] for v in tanh) and all(v['exact'] for v in tails) and strict)
 return dict(schema='sentry_mixed_exemplar_captured_tail_classification_v1',dispatch_class=CLASS,key=[entry['case_index'],entry['window_index']],identity=exports['identity'],all24_a8_exact=True,first_changed_stage=first,embedding_ln=ln,actual_pre_dot=dot,actual_tanh_original_record=tr,actual_tanh_rows=tanh,tail_reproduction=tails,strict_decision_agrees=strict,recognized=recognized,
  inference_limit='LN saved rows recognized; internal variance/sqrt attribution remains inference. Reference pooler/TANH reconstructed from own CLS, not observed. No fitted recipe/epsilon.',numerical_acceptance=None,score_approval=None)


def provenance_record(dispatch,observed,exports,policy):
 prefix=dispatch.with_name('mixed_adapter_'+dispatch.stem);binding=source_binding()
 return dict(schema='sentry_mixed_exemplar_loaded_adapter_dispatch_v1',canonical_dispatch_source=relative(dispatch),canonical_guard_source=relative(dispatch.parent/observed['guard_source']),adapter_source=relative(prefix.with_suffix('.py')),policy_snapshot=relative(prefix.with_name(prefix.name+'_policy.json')),
  helper_snapshots={n:relative(prefix.with_name(prefix.name+'_'+n)) for n in binding['math_helper_sources']},policy=policy,source_binding=binding,identity=exports['identity'],callback_functions=['inspect_adapter','run_adapter'],arithmetic_changed=False)


def dispatch_provenance(entry,exports,policy):
 folder=ROOT/entry['folder'];observed=read(folder/'csim_run.json');dispatch=folder.parent.parent/observed['source'];path=dispatch.with_name(dispatch.stem+'_mixed_adapter.json');record=read(path)
 need(record==provenance_record(dispatch,observed,exports,policy),'Actual adapter dispatch provenance differs')
 need((ROOT/record['adapter_source']).read_text(encoding='utf-8-sig')==_LOADED_SOURCE and read(ROOT/record['policy_snapshot'])==policy,'Actual loaded adapter/policy snapshots differ')
 literal=canonical.w4_padding_literal_functions
 need(literal(dispatch.read_text(encoding='utf-8-sig'),CANONICAL_FUNCTIONS)==record['source_binding']['canonical_functions'],'Actual canonical dispatch seams differ')
 need(literal((ROOT/record['canonical_guard_source']).read_text(encoding='utf-8-sig'),GUARD_FUNCTIONS)==record['source_binding']['guard_functions'],'Actual guard seams differ')
 for n,p in record['helper_snapshots'].items():need((ROOT/p).read_text(encoding='utf-8-sig')==record['source_binding']['math_helper_sources'][n],'Actual math helper snapshot differs')
 return dict(path=relative(path),record=record)


def run_adapter(command,cwd,env,log,timeout,memory_guard=None):
 """Additional callback snapshots before C, inside canonical active advisory guard."""
 c=_context;need(c is not None and isinstance(memory_guard,guards.ConnectedCMemoryGuard) and memory_guard.active and memory_guard.output.resolve()==(OUT/'bounds_v1').resolve(),'Canonical held active guard required')
 need(len(memory_guard.locks)==2 and all(not h.closed for h in memory_guard.locks),'Existing canonical advisory locks must remain held')
 need(timeout==900 and list(command)==[str(canonical.BUILD/'project/solution/csim/build/csim.exe')],'Existing compiled run only; no compile')
 e=c['exports'];policy=validate_policy(c['policy']['path'],e);need(policy==c['policy'],'Policy changed before dispatch')
 folder=Path(log).parent;observed=read(folder/'csim_run.json');dispatch=folder.parent.parent/observed['source']
 need(observed['state']=='running' and observed['identity']==e['identity'] and observed['parameter_folder']==e['parameter_folder'],'Canonical predispatch association differs')
 need(folder.resolve().is_relative_to((OUT/'bounds_v1').resolve()),'Actual callback output must remain in own bounds tree')
 frozen=next((v for v in e['completed_windows'] if [v['case_index'],v['window_index']]==observed['window']),None);need(frozen is not None and frozen['folder']==observed['input_folder'] and env.get('GATE_NUMERICAL_BRIDGE_CASE')==frozen['document_id'],'Actual callback frozen case/window differs')
 need(env.get('GATE_NUMERICAL_BRIDGE_DIR')==str(ROOT/observed['input_folder']) and env.get('GATE_NUMERICAL_BRIDGE_PARAMETER_DIR')==str(OUT/'parameters') and env.get('GATE_NUMERICAL_BRIDGE_DIAGNOSTIC_DIR')==str(folder) and 'GATE_NUMERICAL_BRIDGE_FLOATING' not in env,'Canonical actual environment differs')
 record=provenance_record(dispatch,observed,e,policy);path=dispatch.with_name(dispatch.stem+'_mixed_adapter.json')
 files={ROOT/record['adapter_source']:_LOADED_SOURCE,ROOT/record['policy_snapshot']:json.dumps(policy,indent=2)+'\n'};files.update({ROOT/p:record['source_binding']['math_helper_sources'][n] for n,p in record['helper_snapshots'].items()})
 if path.exists():need(read(path)==record and all(p.read_text(encoding='utf-8-sig')==v for p,v in files.items()),'Do not overwrite actual adapter provenance')
 else:
  need(not any(p.exists() for p in files),'Partial predispatch provenance requires inspection')
  for p,text in files.items():
   with p.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
  save(path,record)
 return c['original_run'](command,cwd,env,log,timeout,memory_guard=memory_guard)


def inspect_adapter(item,entry,exports,review=None,*,padding_policy=None,write_classification=False):
 c=_context;need(c is not None and padding_policy is None,'No inherited uniform class');stopped=False
 try:result=c['original_inspect'](item,entry,exports,review,padding_policy=None,write_classification=False)
 except canonical.PropagationDiscrepancy as error:
  if str(error)!=FINAL_STOP:raise
  stopped=True;result=entry
 stored=entry.get('recognized_mixed_exemplar');bound=entry.get('mixed_adapter_dispatch');observed=read(ROOT/entry['folder']/'csim_run.json');dispatch=(ROOT/entry['folder']).parent.parent/observed['source'];has=(dispatch.with_name(dispatch.stem+'_mixed_adapter.json')).exists();provenance=None
 if write_classification or stored or bound or has:
  provenance=dispatch_provenance(entry,exports,c['policy']);need(not bound or bound==provenance,'Saved callback provenance changed')
  if write_classification:entry['mixed_adapter_dispatch']=provenance
  else:need(bound==provenance,'Historical uncommitted adapter entry cannot be adopted')
 attempt=entry.get('mixed_classification_attempt')
 if attempt:
  expected_path=relative(ROOT/entry['folder']/'recognized_mixed_exemplar.json')
  need(attempt['path']==expected_path and attempt['policy_path']==c['policy']['path'] and attempt['recognized'] is False and read(ROOT/attempt['path'])==attempt['record'],'Preserved failed classification artifact differs')
 if not stopped:
  need(not stored,'Class must not coexist with historical resolution/exact terminal');return result
 if not stored and not write_classification:raise canonical.PropagationDiscrepancy(FINAL_STOP)
 need(provenance is not None and not entry.get('reviewed_resolution'),'Prospective unreviewed actual adapter dispatch required')
 artifact=ROOT/entry['folder']/'recognized_mixed_exemplar.json'
 try:
  classified=classify(entry,exports)
  if write_classification and not stored:
   need(not artifact.exists(),'Preserve previous classification attempt');save(artifact,classified)
  need(classified['recognized'] is True,'Mixed saved-row/own-tail classification failed')
  need(validate_policy(c['policy']['path'],exports)==c['policy'],'Scientific source/policy changed during classification')
 except Exception as error:
  if write_classification and not stored:
   if not artifact.exists():save(artifact,dict(recognized=False,error=str(error),numerical_acceptance=None,score_approval=None))
   entry['mixed_classification_attempt']=dict(path=relative(artifact),recognized=False,policy_path=c['policy']['path'],record=read(artifact))
  raise canonical.PropagationDiscrepancy('Mixed exemplar classification failed: '+str(error)) from error
 record=dict(policy=c['policy'],classification_path=relative(artifact),provenance=provenance,key=[entry['case_index'],entry['window_index']],identity=exports['identity'],acceptance=None)
 if stored:need(stored==record and read(artifact)==classified,'Stored class/policy/artifact changed')
 elif write_classification:entry['recognized_mixed_exemplar']=record
 return entry


@contextmanager
def installed(exports,policy):
 global _context
 need(_context is None,'No nested mixed adapter');old_inspect,old_run=canonical.inspect,guards.run
 _context=dict(exports=exports,policy=policy,original_inspect=old_inspect,original_run=old_run);canonical.inspect,guards.run=inspect_adapter,run_adapter
 try:yield
 finally:canonical.inspect,guards.run=old_inspect,old_run;_context=None


def validate_saved(out,policy_path=None):
 """Read-only canonical saved-state revalidation with exact stored callback/class."""
 from types import SimpleNamespace
 out=Path(out);e=validate_export(out,complete=True);e=dict(e,parameter_folder=relative(out/'parameters'));scope(e,out);state=read(out/'bounds_v1/qualification.json')
 paths={v['recognized_mixed_exemplar']['policy']['path'] for v in state['windows'] if v.get('recognized_mixed_exemplar')};paths.update(v['mixed_adapter_dispatch']['record']['policy']['path'] for v in state['windows'] if v.get('mixed_adapter_dispatch'))
 if policy_path is not None:paths.add(relative(Path(policy_path) if Path(policy_path).is_absolute() else ROOT/policy_path))
 need(len(paths)==1,'One actual adopted policy must bind this state');policy=validate_policy(next(iter(paths)),e)
 with installed(e,policy):canonical.execute_locked(SimpleNamespace(stage='validate',reviewed_resolution=None),out,e,out/'bounds_v1',read(ROOT/'configs/project.json'),None)


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--name',default=NAME,choices=[NAME]);ap.add_argument('--stage',choices=['preview','run','validate'],default='preview');ap.add_argument('--policy');ap.add_argument('--execute',action='store_true');ap.add_argument('--max-new-windows',type=int,default=1);ap.add_argument('--reviewed-resolution');args=ap.parse_args()
 need(0<=args.max_new_windows<=6,'Original bounded allowance0..6');need(args.max_new_windows>0 or args.reviewed_resolution or args.stage=='validate','Zero-work only for explicit review/validation')
 if not args.execute or args.stage=='preview':
  print(json.dumps(dict(execute=False,name=NAME,policy_schema=POLICY_SCHEMA,dispatch_class=CLASS,policy=args.policy,policy_exists=bool(args.policy and (ROOT/args.policy).is_file()),required='Explicit root policy/slot release; exact PTQ exemplar only',canonical_executor='scripts/run_precision_qualification.py',max_new_windows=args.max_new_windows,CPU=[6,7],max_workers=2,WSL_GiB_floor=4,Windows_GiB_floor=2,timeout_seconds=900,prospective_only=True,numerical_acceptance=None,score_approval=None),indent=2));return
 need(args.policy,'Explicit --policy required; no inferred adoption');need(os.name!='nt','Use genuine WSL canonical process paths')
 e=validate_export(OUT,complete=True);e=dict(e,parameter_folder=relative(OUT/'parameters'));scope(e);policy=validate_policy(args.policy,e)
 if args.stage=='validate':validate_saved(OUT,args.policy);return
 with installed(e,policy):canonical.execute(args)


if __name__=='__main__':main()
