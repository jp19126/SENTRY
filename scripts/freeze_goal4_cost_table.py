"""Freeze real Goal4 analytical cost rows; default read-only preview.

No cap selection, model/config writes, inference, synthesis or board activity.
The coordinator supplies caps and actual closure paths for every active profile.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, copy, itertools, json, math, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import estimate_detector_cost as estimator
from build_linear_hls import active_linear_profiles,active_linear_variants
_RUNTIME_CONTEXT_TOKEN=object()
# The three concrete TBs below supply actual port/demux maps to emit_core.
SOURCES=('scripts/freeze_goal4_cost_table.py','scripts/estimate_detector_cost.py','scripts/analyze_cached_linear.py','scripts/build_linear_hls.py','scripts/inspect_vek280_system.py','scripts/plan_detector_commands.py','scripts/build_vek280_system.py','scripts/run_fixed_fp32_rtl.py','hardware/gate_data_mover.sv','hardware/gate_detector_sequencer.sv','hardware/gate_host_mailbox.sv','hardware/gate_detector_control_tb.sv','hardware/linear_axi_tb.sv','hardware/fixed_fp32_axi_tb.sv')
SENSITIVITY=('additional_read_response_cycles','additional_write_response_cycles','supervisor_sync_ms')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def path(value):
 value=Path(str(value).replace('\\','/'))
 candidate=(value if value.is_absolute() else ROOT/value).resolve()
 if not candidate.is_relative_to(ROOT.resolve()):raise ValueError('Evidence/output must remain within this repository')
 return candidate

def relative(value):return Path(value).resolve().relative_to(ROOT.resolve()).as_posix()
def finite(value,positive=False):return type(value) in (int,float) and math.isfinite(value) and (value>0 if positive else value>=0)
def caps_valid(values,maximum=4):
 if not values or len(values)>maximum or any(not finite(v,True) for v in values) or values!=sorted(set(values)):raise ValueError('Root-supplied caps must be sorted, distinct, positive and at most four, in ms')

def stable_inspection(record):
 result={k:v for k,v in record.items() if k!='recorded_utc'}
 for key in ('system_directory','run_record','source_snapshot','pdi','xsa'):
  if key in result:result[key]=result[key].replace('\\','/')
 result['reports']=[v.replace('\\','/') for v in result['reports']]
 return result

class GeneratedText:
 """Capture only the existing two writers' concrete text, without disk writes."""
 def __init__(self,records,name=''):self.records,self.name=records,name
 def __truediv__(self,name):return GeneratedText(self.records,str(name))
 def mkdir(self,**kwargs):pass
 def write_text(self,text,**kwargs):self.records[self.name]=text

def current_system_sources(system,config,profile):
 from build_vek280_system import emit_core
 from plan_detector_commands import create_plan,write_rtl_plan
 from run_fixed_fp32_rtl import reachable_rtl
 snapshot=system/'sources';variant=active_linear_variants(config)[profile['id']]
 parent='linear_hls_cached_static_bank' if variant=='weight_cache_static_bank' else 'linear_hls_cached'
 integer=ROOT/'build'/parent/profile['id']/'project/solution/syn/verilog';fp=ROOT/'build/fixed_fp32_hls/project/solution/syn/verilog'
 fpfiles,ips=reachable_rtl(fp)
 expected=[ROOT/'hardware'/name for name in ('gate_data_mover.sv','gate_detector_sequencer.sv','gate_host_mailbox.sv')]+sorted(integer.glob('*.v'))+fpfiles+sorted(fp.glob('*.dat'))
 if not (integer/'gate_linear_top.v').is_file():raise ValueError('Current selected integer synthesis missing')
 expected_rel={p.relative_to(ROOT).as_posix() for p in expected}
 actual_rel={p.relative_to(snapshot).as_posix() for p in snapshot.rglob('*') if p.is_file()}
 if expected_rel!=actual_rel:raise ValueError('Actual integrated source inventory differs from current physical profile')
 for source in expected:
  if source.read_bytes()!=(snapshot/source.relative_to(ROOT)).read_bytes():raise ValueError('Actual integrated source is stale: '+relative(source))
 for source in ips:
  if (system/'arithmetic_ip'/source.name).read_text()!=source.read_text().replace('generate_target {synthesis simulation}','generate_target {synthesis}'):raise ValueError('Actual arithmetic IP parameters/source differ')
 generated={};emit_core(GeneratedText(generated),256)
 for name,text in generated.items():
  if (system/name).read_text()!=text:raise ValueError('Actual generated platform wrapper differs: '+name)
 plan=create_plan(config,256);saved=read(system/'command_plan.json')
 if {k:v for k,v in plan.items() if k!='recorded_utc'}!={k:v for k,v in saved.items() if k!='recorded_utc'}:raise ValueError('Actual integrated command plan differs')
 generated={};write_rtl_plan(plan,GeneratedText(generated))
 for name,text in generated.items():
  if (system/name).read_text()!=text:raise ValueError('Actual integrated command ROM/constants differ: '+name)
 run=read(system/'run.json')
 if run['integer_kernel_variant']!=variant or run['length']!=256 or run['lanes']!=profile['lanes']:raise ValueError('Actual integrated physical profile/variant differs')
 return dict(source_files=len(expected),generated_wrappers=3,command_plan_current=True,kernel_variant=variant)

def inspect_closure(closure,profile,config):
 import inspect_vek280_system as inspector
 if not hasattr(inspector,'inspect_system'):raise RuntimeError('Read-only inspect_system API not yet available')
 saved=read(closure);system=path(saved['system_directory']);implementation=path(saved['run_record'])
 actual,solution,traffic=inspector.inspect_system(system,implementation_record=None if implementation==(system/'run.json').resolve() else implementation,config=config)
 required=('part','profile_id','length','clock_objective_mhz','resources','resource_limits','resources_within_limits','timing_met','bandwidth_feasible','integrated_eligible','drc_errors_or_critical_warnings')
 if actual['profile_id']!=profile['id'] or actual['length']!=256:raise ValueError('Closure belongs to another actual profile/length')
 if any(type(actual[k]) is not bool for k in ('timing_met','bandwidth_feasible','integrated_eligible','resources_within_limits')):raise ValueError('Closure eligibility flags remain unresolved')
 disagreements=[k for k in required if k not in saved or saved[k]!=actual[k]]
 if disagreements:raise ValueError('Saved closure differs from reparsed actual reports: '+', '.join(disagreements))
 current=current_system_sources(system,config,profile)
 return dict(path=relative(closure),record=saved,reparsed=stable_inspection(actual),current_sources=current,noc_solution=solution,noc_traffic=traffic)

def model_sensitivity(model):
 values=model['sensitivity']
 if set(values)!=set(SENSITIVITY):raise ValueError('Unknown/missing cost sensitivity dimensions')
 for key in SENSITIVITY:
  choices=values[key]
  if not isinstance(choices,list) or not choices or choices!=sorted(set(choices)) or any(not finite(v,key=='supervisor_sync_ms') for v in choices) or model['nominal'][key] not in choices:raise ValueError('Sensitivity must preserve finite declared nominal and sorted distinct values: '+key)
 return [dict(zip(SENSITIVITY,v)) for v in itertools.product(*(values[k] for k in SENSITIVITY))]

def consumer_contract():
 source=(ROOT/'scripts/search_precision.py').read_text(encoding='utf-8-sig')
 selected={node.name:ast.get_source_segment(source,node) for node in ast.parse(source).body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in ('CostTable','groups','precision_tuple')}
 if set(selected)!={'CostTable','groups','precision_tuple'}:raise ValueError('Actual CostTable consumer contract unavailable')
 return selected

def cost_header(config,model_path,model,closures,caps):
 hardware=config['hardware']
 return dict(schema='sentry_goal4_frozen_complete_cost_table_v1',latency_unit='ms',latency_scope='full_detector_host_to_decision',detector_execution='fpga',primary_length=256,primary_batch=1,part=hardware['part'],clock_objective_mhz=hardware['clock_objective_mhz'],interface=hardware['interface'],tool_versions=hardware['tool_versions'],execution_host=hardware['execution_host'],allocated_board=hardware['allocated_board'],resource_limits=hardware['resource_limits'],frozen_latency_caps_ms=caps,profile_kernel_variants=active_linear_variants(config),cost_consumer_contract=consumer_contract(),whole_check_cost_model_path=relative(model_path),whole_check_cost_model=model,closure_associations=closures,cost_source_text={name:(ROOT/name).read_text(encoding='utf-8-sig') for name in SOURCES},scope='Analytical warm complete-window checking at selected nominal waits/host/UART assumptions; physical calibration pending; cold boundary separate',cold_boundary=model['cold_boundary'],physical_board_measurement=False)

def validate_frozen_cost_sources(table,config,*,context=None):
 """Full source/build check once per execute/resume; cheap preappend checks later.

 The returned context is process-local, never saved as evidence or acceptance.
 Campaign sources/compiled RTL are immutable during that execution. Nominal
 assumptions, consumed JSON evidence and the frozen header are rechecked for
 every appended map. A new process/resume always repeats the full source check.
 """
 header={k:v for k,v in table.items() if k!='candidates'}
 if context is not None:
  if context.get('_process_token') is not _RUNTIME_CONTEXT_TOKEN or context['frozen_header']!=header:raise ValueError('Unvalidated/changed frozen cost runtime context')
 model_path=path(config['hardware']['whole_check_cost_model'])
 if relative(model_path)!=table['whole_check_cost_model_path'] or read(model_path)!=table['whole_check_cost_model']:raise ValueError('Current nominal cost model/closure mapping differs from frozen table')
 profiles=active_linear_profiles(config)
 if table['profile_kernel_variants']!=active_linear_variants(config) or {v['id']:v['physical_profile'] for v in table['profiles']}!={v['id']:v for v in profiles}:raise ValueError('Current common physical profiles/variants differ')
 for name,record in table['estimator_evidence_records'].items():
  if read(path(name))!=record:raise ValueError('Current estimator evidence changed: '+name)
 for key in ('part','clock_objective_mhz','interface','resource_limits','tool_versions','execution_host','allocated_board'):
  if table[key]!=config['hardware'][key]:raise ValueError('Frozen common hardware changed: '+key)
 if context is not None:return context
 if table['cost_consumer_contract']!=consumer_contract():raise ValueError('Actual CostTable consumer/map contract changed')
 for name,text in table['cost_source_text'].items():
  if (ROOT/name).read_text(encoding='utf-8-sig')!=text:raise ValueError('Current cost/source contract differs from frozen table: '+name)
 if set(table['cost_source_text'])!=set(SOURCES):raise ValueError('Frozen cost source scope differs')
 for profile in profiles:
  association=table['closure_associations'][profile['id']]
  if inspect_closure(path(association['path']),profile,config)!=association:raise ValueError('Actual closure/source association changed: '+profile['id'])
 return dict(_process_token=_RUNTIME_CONTEXT_TOKEN,frozen_header=copy.deepcopy(header),current_frozen_sources_checked=True,profiles=len(profiles),nominal_model_unchanged=True)


def validate_table_rows(table,config):
 # Use the actual consumer row validator without fabricating numerical approvals
 # to bypass CostTable.__init__'s independent score-readiness prerequisite.
 from search_precision import CostTable,precision_tuple
 consumer=CostTable.__new__(CostTable);consumer.config=config;consumer.profiles={v['id']:v for v in table['profiles']};consumer.lookup={};consumer.lookups=0
 for candidate in table['candidates']:
  key=precision_tuple(candidate['precision_map'],config)
  if key in consumer.lookup:raise ValueError('Duplicate estimator anchor')
  consumer.lookup[key]=candidate['profiles']
 feasible=[consumer.feasible_profiles(v['precision_map']) for v in table['candidates']]
 if not any(feasible):raise RuntimeError('No genuinely eligible actual profile for any anchor')
 return dict(anchor_rows=len(feasible),feasible_anchor_profiles=feasible,consumer_row_checks=True)

def prepare(config,caps,requested):
 blockers=[];profiles=active_linear_profiles(config)
 if caps is not None:caps_valid(caps,min(4,config['methods']['published_baseline']['project_search_defaults']['latency_cap_count_max']))
 model_path=path(config['hardware']['whole_check_cost_model']);model=read(model_path);scenarios=model_sensitivity(model)
 configured={k:relative(path(v)) for k,v in model['integrated_profiles'].items()};selected={**configured,**{k:relative(path(v)) for k,v in requested.items()}};closures={};observations={}
 if set(selected)-{v['id'] for v in profiles}:raise ValueError('Unknown closure profile')
 if caps is None:blockers.append('Root-supplied --caps-ms required; no caps selected')
 if config['methods']['published_baseline']['project_search_defaults']['latency_cap_unit']!='ms':blockers.append('Common configured latency cap unit must be ms')
 for profile in profiles:
  name=profile['id'];closure=selected.get(name)
  if not closure:blockers.append('Missing current actual closure: '+name);continue
  saved={}
  try:
   saved=read(path(closure));closures[name]=inspect_closure(path(closure),profile,config);observations[name]=dict(validated=True,integrated_eligible=closures[name]['reparsed']['integrated_eligible'])
  except (ValueError,RuntimeError,FileNotFoundError,KeyError) as exc:
   blockers.append(name+': '+str(exc));observations[name]=dict(validated=False,error=str(exc),reported_timing_met=saved.get('timing_met'),reported_integrated_eligible=saved.get('integrated_eligible'))
 if selected!=configured:blockers.append('Selected closures must equal hardware whole-check model integrated_profiles before freezing/dynamic estimator use')
 if len(closures)==len(profiles) and not any(v['reparsed']['integrated_eligible'] for v in closures.values()):blockers.append('No genuinely eligible current actual integrated profile')
 if (config['measurement']['primary_length'],config['measurement']['primary_batch'])!=(256,1):blockers.append('Only scoped L256/B1 cost freeze supported')
 anchors=[];reports=set(model['source_reports'])|{relative(model_path),model['host_preprocessing_record']};evidence={};sensitivities=[]
 try:
  for bits in (4,8):
   row=estimator.estimate_candidate_cost(dict.fromkeys(estimator.names(config),bits),config);row['anchor']=f'uniform_w{bits}a8';anchors.append(row)
   for item in row['profiles']:
    reports.update((item['integer']['source'],item['fixed_fp32']['source'],item['control_layout']['source']))
    if not item['integer']['reserved_validation_passed']:raise ValueError('Actual reserved estimator validation failed: '+item['profile_id'])
    for scenario in scenarios:
     changed=copy.deepcopy(model);changed['nominal'].update(scenario);terms=estimator.full_check_terms(config,item,changed)
     sensitivities.append(dict(anchor=row['anchor'],profile_id=item['profile_id'],assumptions=scenario,estimated_complete_checking_ms=terms['estimated_complete_checking_ms']))
 except (ValueError,RuntimeError,FileNotFoundError,KeyError,TypeError) as exc:blockers.append('Actual estimator prerequisite: '+str(exc))
 reports.update(('results/goal4/fixed_fp32/run.json','results/goal4/internal_integration/command_plan.json','build/fixed_fp32_hls/run.json'))
 for name in sorted(v for v in reports if v):
  if Path(name).suffix=='.json':evidence[name]=read(path(name))
 if blockers:return None,dict(ready=False,blockers=blockers,profiles=observations,anchor_rows=len(anchors),anchor_estimates=[dict(anchor=r['anchor'],profiles=[dict(profile_id=v['profile_id'],estimated_complete_checking_ms=v['estimated_complete_checking_ms'],feasible=v['feasible']) for v in r['profiles']]) for r in anchors],caps_ms=caps,sensitivity_scenarios=len(scenarios),cost_approval=None)
 reports.update(v['reparsed']['run_record'].replace('\\','/') for v in closures.values())
 for association in closures.values():reports.update(name.replace('\\','/') for name in association['reparsed']['reports'])
 table=cost_header(config,model_path,model,closures,caps)
 table['profiles']=[dict(id=p['id'],physical_profile=p,resources=closures[p['id']]['reparsed']['resources'],timing_met=closures[p['id']]['reparsed']['timing_met'],bandwidth_feasible=closures[p['id']]['reparsed']['bandwidth_feasible'],integrated_eligible=closures[p['id']]['reparsed']['integrated_eligible'],closure_report=closures[p['id']]['path']) for p in profiles]
 table.update(candidates=anchors,source_reports=sorted(reports|{v['path'] for v in closures.values()}),estimator_evidence_records=evidence,nominal_sensitivity=sensitivities)
 checks=validate_table_rows(table,config);validate_frozen_cost_sources(table,config)
 return table,dict(ready=True,blockers=[],profiles=observations,caps_ms=caps,sensitivity_scenarios=len(scenarios),checks=checks,cost_approval=None)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--freeze',action='store_true');ap.add_argument('--caps-ms',type=float,nargs='+');ap.add_argument('--closure',action='append',default=[],metavar='PROFILE=PATH');ap.add_argument('--output');ap.add_argument('--validation-report');args=ap.parse_args()
 if args.caps_ms is not None:
  try:caps_valid(args.caps_ms)
  except ValueError as exc:ap.error(str(exc))
 if args.freeze and (not args.output or not args.validation_report):ap.error('--freeze requires explicit --output and --validation-report')
 requested={}
 for value in args.closure:
  if '=' not in value:ap.error('--closure requires PROFILE=PATH')
  name,filename=value.split('=',1)
  if not filename or name in requested:ap.error('Nonempty unique --closure assignment required')
  requested[name]=filename.replace('\\','/')
 config=read(ROOT/'configs/project.json');table,preview=prepare(config,args.caps_ms,requested)
 preview.update(freeze_requested=args.freeze,physical_board_activity=False,numerical_identity_approval=None,config_changed=False)
 if not args.freeze:print(json.dumps(preview,indent=2));return
 if table is None:raise RuntimeError('Freeze blocked: '+'; '.join(preview['blockers']))
 output,report=path(args.output),path(args.validation_report)
 if output==report or output.exists() or report.exists():raise ValueError('Preserve existing evidence; choose two distinct fresh output paths')
 table['validation_report']=relative(report);table['recorded_utc']=datetime.now(timezone.utc).isoformat()
 validation=dict(recorded_utc=table['recorded_utc'],state='actual_cost_sources_validated',table=relative(output),**preview,scope=table['scope'],nominal_model=table['whole_check_cost_model'],source_reports=table['source_reports'])
 # Two explicit fresh files only. A partial write cannot be selected as a valid
 # common table: publish the completed validation report before the table.
 for destination,value in ((report,validation),(output,table)):
  destination.parent.mkdir(parents=True,exist_ok=True)
  with destination.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(output=relative(output),validation_report=relative(report),caps_ms=args.caps_ms,numerical_identity_approval=None,config_changed=False)))
if __name__=='__main__':main()
