"""Separate checked-C continuation for explicit decoded W4/W8-map exports.
Preview by default. One fresh compatible C build; no HLS synthesis. All discrete
or unexplained terminal changes stop with original actual bytes preserved.
"""
from pathlib import Path
import argparse,ast,contextlib,json,math,os,struct,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from export_precision_qualification import SITES,need,read,save,outdir,validate_export,write,function_ast
BUILD=ROOT/'build/precision_qualification_bounds_tail_v2'
CONTRACT='decoded_w4w8_a8_integer_contract'
class PropagationDiscrepancy(RuntimeError):pass
SUBSTITUTIONS=[('checked native int64 reference of the already-validated W8/A8 integer contract.','checked native int64 reference of the decoded W4/W8/A8 integer contract.'),('name+"_w8"','name+"_code_i8"'),('"w8_a8_integer_contract"','"decoded_w4w8_a8_integer_contract"')]

def generate_precision_tb(source):
 """Three checked storage labels, five bounds insertions, two output-only tail captures."""
 for before,after in SUBSTITUTIONS:need(source.count(before)==1,'Canonical precision substitution seam changed: '+before);source=source.replace(before,after,1)
 from run_qualification_cases import generate_bounds_tb
 source=generate_bounds_tb(source)
 before='        pooled=fp(gate::FP32_TANH,pooled,{},{},1,D);'
 need(source.count(before)==1,'Actual TANH capture seam changed')
 return source.replace(before,'        write_binary("diagnostic_pooler_pre_tanh",pooled);\n'+before+'\n        write_binary("diagnostic_pooler_post_tanh",pooled);',1)

def prepared(build=BUILD):
 from run_qualification_cases import generate_bounds_tb
 need((build/'prepared.json').is_file(),'Run separate --stage prepare first');record=read(build/'prepared.json');source=ROOT/'build/fixed_fp32_hls';need(read(source/'run.json')['synthesis_passed'],'Existing common FP synthesis failed/missing')
 need(record['substitutions']==[dict(before=a,after=b) for a,b in SUBSTITUTIONS] and record['tail_capture_writes']==['diagnostic_pooler_pre_tanh','diagnostic_pooler_post_tanh'],'Recorded precision substitutions/output-only captures changed')
 for name in ('fixed_fp32_service.hpp','connected_fp32_bounds.hpp'):need((build/name).read_bytes()==(ROOT/'hardware'/name).read_bytes(),'Current service/checker source changed: '+name)
 need((build/'canonical_numerical_bridge_tb.cpp').read_bytes()==(ROOT/'hardware/numerical_bridge_tb.cpp').read_bytes(),'Canonical C source changed');need((build/'numerical_bridge_tb.cpp').read_text()==generate_precision_tb((ROOT/'hardware/numerical_bridge_tb.cpp').read_text()),'Generated precision C source changed')
 need((build/'gate_fixed_fp32_top.cpp').read_bytes()==(source/'gate_fixed_fp32_top.cpp').read_bytes(),'Current FP top changed');need((source/'fixed_fp32_service.hpp').read_bytes()==(ROOT/'hardware/fixed_fp32_service.hpp').read_bytes(),'Synthesized current FP source changed')
 generator=ast.parse((ROOT/'scripts/build_fixed_fp32_hls.py').read_text(encoding='utf-8-sig'));tops=[n.args[0].value for n in ast.walk(generator) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='top' and n.func.attr=='write_text' and n.args and isinstance(n.args[0],ast.Constant)];need(len(tops)==1 and (source/'gate_fixed_fp32_top.cpp').read_bytes()==tops[0].encode(),'Actual synthesized wrapper/generator association changed')
 need(function_ast((build/'run_qualification_cases.py').read_text(),'generate_bounds_tb')==function_ast((ROOT/'scripts/run_qualification_cases.py').read_text(),'generate_bounds_tb'),'Checker insertion function changed')
 need(function_ast((build/'run_precision_qualification.py').read_text(),'generate_precision_tb')==function_ast(Path(__file__).read_text(),'generate_precision_tb'),'Precision generation function changed')
 tcl=(build/'qualify.tcl').read_text();need(tcl.count('-std=c++14 -O2 -ffp-contract=off')==2 and tcl.count('config_compile -unsafe_math_optimizations=false')==1 and 'csim_design' in tcl and 'csynth_design' not in tcl,'Separate C compile flags/no-synthesis contract changed')
 need(record['synthesis_recorded_utc']==read(source/'run.json')['recorded_utc'],'Existing synthesized source timestamp association differs');return record

def prepare():
 if BUILD.exists():return dict(prepared=True,reused=True,record=prepared())
 source=ROOT/'build/fixed_fp32_hls';synthesis=read(source/'run.json');need(synthesis['synthesis_passed'],'Require existing successful common FP32 synthesis');text=generate_precision_tb((ROOT/'hardware/numerical_bridge_tb.cpp').read_text());BUILD.mkdir()
 for name in ('fixed_fp32_service.hpp','connected_fp32_bounds.hpp'):write(BUILD/name,(ROOT/'hardware'/name).read_bytes())
 for name,p in [('canonical_numerical_bridge_tb.cpp',ROOT/'hardware/numerical_bridge_tb.cpp'),('gate_fixed_fp32_top.cpp',source/'gate_fixed_fp32_top.cpp'),('run_qualification_cases.py',ROOT/'scripts/run_qualification_cases.py'),('run_precision_qualification.py',Path(__file__))]:write(BUILD/name,p.read_bytes())
 write(BUILD/'numerical_bridge_tb.cpp',text.encode());config=read(ROOT/'configs/project.json');q=lambda p:'{'+str(p).replace('\\','/').replace('//wsl.localhost/Ubuntu','')+'}';flags='-std=c++14 -O2 -ffp-contract=off -I'+str(BUILD).replace('\\','/').replace('//wsl.localhost/Ubuntu','')
 tcl='\n'.join(['open_project '+q(BUILD/'project'),'set_top gate_fixed_fp32_top','add_files '+q(BUILD/'gate_fixed_fp32_top.cpp')+' -cflags {'+flags+'}','add_files -tb '+q(BUILD/'numerical_bridge_tb.cpp')+' -cflags {'+flags+'}','open_solution solution -flow_target '+config['hardware']['hls']['flow_target'],'set_part {'+config['hardware']['part']+'}','create_clock -period 5','config_compile -unsafe_math_optimizations=false','csim_design','exit'])+'\n';write(BUILD/'qualify.tcl',tcl.encode())
 record=dict(contract=CONTRACT,sequence_length=256,substitutions=[dict(before=a,after=b) for a,b in SUBSTITUTIONS],bounds_insertions=5,tail_capture_writes=['diagnostic_pooler_pre_tanh','diagnostic_pooler_post_tanh'],tail_scope='Two output-only writes around unchanged actual TANH service; no arithmetic/call count changes',synthesis_recorded_utc=synthesis['recorded_utc'],new_synthesis=False,compatible_maps='Explicit exported16-group W4/W8 PTQ or matching saved QAT; decoded signed-i8 file storage, separately validated real packed bytes',approval=None);save(BUILD/'prepared.json',record);prepared();return dict(prepared=True,reused=False,record=record)

def review_resolution(path,entry,exports):
 """Only explicit coordinator records; never manufacture a resolution or epsilon."""
 p=Path(path);p=p if p.is_absolute() else ROOT/p;p=p.resolve();need(p.is_relative_to(ROOT.resolve()),'Review must be existing repository file');r=read(p);s=entry['summary'];expected=dict(policy_id='bounded_connected_development_v2',mode='decoded_w4w8',coordinator_review=True,allow_resume=True,same_input_operator_checks_valid=True,identity=exports['identity'],case_index=entry['case_index'],window_index=entry['window_index'],document_id=entry['document_id'],summary=s,a8=entry['a8'],operator_bounds=entry['operator_bounds'])
 need(all(r.get(k)==v for k,v in expected.items()) and all(r.get(k) is True for k in ('coordinator_review','allow_resume','same_input_operator_checks_valid')),'Explicit exact saved-case coordinator review differs');need(entry['returncode']==0 and s['completed'] and s['strict_decision_agrees'] is True,'Cannot resolve service/process/strict-decision failure');need(isinstance(r.get('reason'),str) and r['reason'].strip() and isinstance(r.get('evidence_paths'),list) and r['evidence_paths'],'Concrete causal review evidence required')
 for value in r['evidence_paths']:
  e=(ROOT/value).resolve();need(not Path(value).is_absolute() and e.is_relative_to(ROOT.resolve()) and e.is_file() and e!=p,'Missing/outside causal evidence')
 return dict(path=p.relative_to(ROOT.resolve()).as_posix(),record=r)

# This optional class never writes an adopted policy. The coordinator must supply one.
W4_PADDING_DATA=ROOT/'results/goal4/numerical_bridge/precision_qualification_v1/uniform_w4_qat'
W4_PADDING_REVIEW=ROOT/'results/goal4/numerical_bridge/connected_bounds_review_v2'
W4_PADDING_CLASS='uniform_w4_qat_l256_valid_a8_exact_masked_padding_v1'
W4_PADDING_FUNCTIONS=('w4_padding_scope','w4_padding_literal_functions','w4_padding_sources','classify_w4_padding','validate_w4_padding_policy_record','validate_w4_padding_policy','w4_padding_compare_sources','w4_padding_provenance','w4_padding_entry_record','recognized_w4_padding_entry','inspect','execute','execute_locked','prepared','generate_precision_tb')
W4_PADDING_GUARD_FUNCTIONS=('paused','run','generate_bounds_tb','validate_operator_bounds','ConnectedCMemoryGuard')

def w4_padding_scope(exports):
 anchor=read(W4_PADDING_DATA/'export.json');identity=exports['identity'];selection=exports['selection']
 need(identity==anchor['identity'] and selection==anchor['selection'] and exports['layers']==anchor['layers'],'Padding class requires own frozen uniform W4 identity/selection/layers')
 need(identity['windowing']['primary_length']==256 and identity.get('precision_source')=='qat' and identity.get('qat_checkpoint')=='checkpoints/quantized/qat_w4_a8' and identity.get('scored_checkpoint')=='checkpoints/quantized/qat_w4_a8','Padding class is matching uniform W4-QAT/L256 only')
 need(len(identity['precision_map'])==16 and all(type(v) is int and v==4 for v in identity['precision_map'].values()) and len(exports['layers'])==24 and all(v['bits']==4 and v['decoded_file']==v['name']+'_code_i8.bin' for v in exports['layers']),'Padding class excludes W8/mixed/storage substitutes')
 need(len(selection['cases'])==32 and sum(v['window_count'] for v in selection['cases'])==66 and exports['complete'] is True and exports['parameter_folder']==(W4_PADDING_DATA/'parameters').relative_to(ROOT).as_posix(),'Padding class requires own complete frozen32/66 exports/parameters')
 return anchor

def w4_padding_literal_functions(source,names):
 tree=ast.parse(source);result={}
 for name in names:
  nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name];need(len(nodes)==1,'Exactly one associated scientific function required: '+name);result[name]=ast.get_source_segment(source,nodes[0])
 return result

def w4_padding_sources():
 helpers={p.relative_to(ROOT).as_posix():p.read_text(encoding='utf-8-sig') for p in (W4_PADDING_REVIEW/'saved_ln_boundary.py',W4_PADDING_REVIEW/'saved_w8_propagation.py')}
 functions=w4_padding_literal_functions(Path(__file__).read_text(encoding='utf-8-sig'),W4_PADDING_FUNCTIONS);guards=w4_padding_literal_functions((ROOT/'scripts/run_qualification_cases.py').read_text(encoding='utf-8-sig'),W4_PADDING_GUARD_FUNCTIONS)
 return dict(helper_sources=helpers,classifier_functions=functions,guard_functions=guards)

def classify_w4_padding(entry,exports):
 """Pure saved-byte diagnostic. Caller validates full export/build/process/operators.

 The unchanged W8 classifiers retain their separate mode/zero-A8 rejection.
 No reference-input model, elementary neighbor fitting or terminal epsilon.
 """
 import importlib.util
 from decimal import Decimal,localcontext
 from validate_qualification_exports import binary
 w4_padding_scope(exports);a,p,parameters=ROOT/entry['folder'],ROOT/entry['input_folder'],ROOT/exports['parameter_folder'];s=entry['summary'];item=next(v for v in exports['completed_windows'] if (v['case_index'],v['window_index'])==(entry['case_index'],entry['window_index']));window=read(p/'window.json');case=next(v for v in exports['selection']['cases'] if v['case_index']==entry['case_index']);mask=window['inputs']['attention_mask']
 need(item['folder']==entry['input_folder'] and window['inputs']==case['windows'][entry['window_index']] and len(mask)==256 and set(mask)<={0,1} and mask[0]==1 and sum(mask)>0,'Padding class frozen window/mask differs')
 need(s['completed'] is True and s['encoder_arithmetic']==CONTRACT and entry['returncode']==0 and entry['acceptance'] is None and s['strict_decision_agrees'] is True,'Padding class cannot waive process/mode/completion/decision failure')
 need(s==read(a/'bridge_summary.json'),'Own actual summary changed');bounds=__import__('run_qualification_cases').validate_operator_bounds(a,s);need(bounds==entry['operator_bounds'] and bounds['calls']==411 and bounds['checked_calls_by_op']==[165,72,72,2,16,18,33,32,1] and s['same_input_quantization_exact'] is True and s['integer_accumulators_exact'] is True and s['same_input_codes_checked']==s['exact_accumulators_checked']==2359296,'Padding class requires all411 original same-input contracts')
 def load(name,path):
  spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
 m=load('w4_padding_saved_math',W4_PADDING_REVIEW/'saved_w8_propagation.py');lnm=load('w4_padding_saved_ln',W4_PADDING_REVIEW/'saved_ln_boundary.py');f32=m.f32
 def signed(raw):return [v if v<128 else v-256 for v in raw]
 def code(x,scale):return max(-128,min(127,round(f32(x/scale))))
 def stats(av,rv,indices):
  d=[abs(av[i]-rv[i]) for i in indices];total=squared=0.0
  # Match canonical ascending binary64 additions, including Python3.12+.
  for value in d:total+=value;squared+=value*value
  return dict(values=len(d),maximum_absolute_error=max(d,default=0),mean_absolute_error=total/len(d) if d else 0,rms_error=math.sqrt(squared/len(d)) if d else 0,changed_fp32_encodings=sum(m.encoding(av[i])!=m.encoding(rv[i]) for i in indices))
 km=binary(p/'key_mask.bin',1024);mv=struct.unpack('<256f',km);maximum=struct.unpack('<f',struct.pack('<I',0x7f7fffff))[0];need(all(mv[i]==(0.0 if mask[i] else -maximum) for i in range(256)),'Actual/reference process-bound key-mask bytes differ')
 sites=[];valid_changes=0;retained=set();need(len(s['integer_layers'])==24,'Exactly24 sites required')
 for record,meta in zip(s['integer_layers'],exports['layers']):
  name,k,n=meta['name'],meta['k'],meta['n'];need(record['name']==name,'Padding class24-site order differs');ab=binary(a/('actual_'+name+'_a8.bin'),256*k);rb=binary(p/(name+'_reference_a8.bin'),256*k);av,rv=signed(ab),signed(rb);changed=[i for i,(x,y) in enumerate(zip(av,rv)) if x!=y];vc=sum(mask[i//k] for i in changed);valid_changes+=vc;need(vc==0,'Padding class rejects any valid-token propagated A8: '+name)
  need(record['original_software_code_differences']==len(changed),'Padding class A8 count differs');params=m.floats(parameters/(name+'_params.bin'),1+2*n);scale=params[0];need(scale>0,'Positive frozen input scale required');x=refx=None;saved=a/('changed_'+name+'_input.bin')
  if changed:
   x=m.floats(saved,256*k);retained.add(saved.name);need(av==[code(v,scale) for v in x],'Actual retained input does not reproduce A8')
  if name.split('_',1)[1] in ('query','key','value'):
   depth=int(name[5]);stage='embedding_ln' if depth==0 else 'layer'+str(depth-1);staged=m.floats(a/('bridge_'+stage+'.bin'),256*k);need(x is None or x==staged,'Retained QKV input differs from preceding stage');x=staged;refx=m.floats(p/('reference_'+stage+'.bin'),256*k);need(av==[code(v,scale) for v in x] and rv==[code(v,scale) for v in refx],'Available paired QKV quantization reconstruction failed')
  sites.append(dict(name=name,values=len(ab),differences=len(changed),valid_row_differences=vc,changed_rows=sorted({i//k for i in changed}),all_index_actual_reference=[[i,av[i],rv[i]] for i in changed],scale=scale,retained_actual_input=saved.relative_to(ROOT).as_posix() if changed else None,actual_saved_input_reconstructed=x is not None,reference_saved_input_reconstructed=refx is not None,reference_input_gap=None if refx is not None else 'Reference context/LN/GELU pre-input floats not captured.'))
 need({v.name for v in a.glob('changed_*_input.bin')}==retained,'Retained changed-input membership differs');stages=[]
 for error in s['intermediate_errors']:
  name=error['name'];count=2 if name=='logits' else 1 if name=='risk' else 65536;av=m.floats(a/('bridge_'+name+'.bin'),count);rv=m.floats(p/('reference_'+name+'.bin'),count);allstats=stats(av,rv,range(count));need(all(error[k]==allstats[k] for k in ('values','maximum_absolute_error','mean_absolute_error','rms_error')),'Saved stage statistics changed')
  stages.append(dict(name=name,all_rows=allstats,valid_rows=stats(av,rv,[i for i in range(count) if mask[i//256]]) if count==65536 else None,padding_rows=stats(av,rv,[i for i in range(count) if not mask[i//256]]) if count==65536 else None,cls=stats(av,rv,range(256)) if count==65536 else None))
 first=next((v['name'] for v in stages if v['all_rows']['changed_fp32_encodings']),None);need(first=='embedding_ln','First changed saved stage must be embedding-LN');ln=lnm.classify_embedding_ln(a,p,parameters,expected_length=256);need(ln['all_rows_recognized'] is True and ln['rows_checked']==256 and ln['values_checked']==65536 and ln['first_unrecognized'] is None,'All256 LN rows must be recognized')
 pre=m.floats(a/'diagnostic_pooler_pre_tanh.bin',256);post=m.floats(a/'diagnostic_pooler_post_tanh.bin',256);ct=m.tail(m.floats(a/'bridge_layer3.bin',65536)[:256],parameters);rt=m.tail(m.floats(p/'reference_layer3.bin',65536)[:256],parameters);precheck=m.differences(pre,ct['pooler_pre'],all_outputs=True);need(precheck['changed_values']==0,'Actual captured pooler pre-dot mismatch')
 records=[json.loads(v) for v in (a/'operator_bounds.jsonl').read_text().splitlines()];trs=[v for v in records if v['op']==8];need(len(trs)==1,'Exactly one actual TANH record required');tr=trs[0];need(tr['call_index']==409 and tr['rows']==1 and tr['width']==256 and tr['values_checked']==256 and tr['atol']==3e-6 and tr['rtol']==3e-5 and tr['passed'] is True and tr['exact_formula_required'] is False,'Original TANH call/bounds differ');tanh=[]
 with localcontext() as ctx:
  ctx.prec=80
  for i,(x,y) in enumerate(zip(pre,post)):
   exp=(2*Decimal.from_float(x)).exp();mat=(exp-1)/(exp+1);error=abs(Decimal.from_float(y)-mat);bound=Decimal.from_float(tr['atol'])+Decimal.from_float(tr['rtol'])*abs(mat);tanh.append(dict(index=i,input=x,actual=y,mathematical_tanh=str(mat),absolute_error=str(error),original_bound=str(bound),passed=error<=bound))
 need(all(v['passed'] for v in tanh),'Captured TANH violates original bound');logits=m.ordered_dot(post,m.floats(parameters/'classifier_weight.bin',512),m.floats(parameters/'classifier_bias.bin',2));exps=[f32(math.exp(f32(v-max(logits)))) for v in logits];total=0.
 for v in exps:total=f32(total+v)
 actualtail=dict(logits=logits,risk=[f32(exps[1]/total)]);tails=[]
 for name,computed,folder,prefix in [('actual_captured_tanh',actualtail,a,'bridge'),('reference_host_recipe',rt,p,'reference')]:
  checks={n:m.differences(computed[n],m.floats(folder/(prefix+'_'+n+'.bin'),c),all_outputs=True) for n,c in [('logits',2),('risk',1)]};need(all(v['changed_values']==0 for v in checks.values()),'Own '+name+' terminal bits do not reproduce');tails.append(dict(backend=name,checks=checks))
 scorebounds=[]
 for depth in range(4):
  projected=[]
  for op in ('query','key'):
   name=f'layer{depth}_{op}';params=m.floats(parameters/(name+'_params.bin'),513);w=signed(binary(parameters/(name+'_code_i8.bin'),65536));need(all(-8<=v<=7 for v in w),'Declared W4 range differs');projected.append(max(2*(128*sum(abs(v) for v in w[o*256:(o+1)*256])*abs(f32(params[0]*params[1+o]))+abs(params[257+o]))+1 for o in range(256)))
  bound=2*64*projected[0]*projected[1]/8+1;need(math.isfinite(bound) and bound<1e20 and f32(-maximum+bound)==-maximum and f32(math.exp(f32(f32(-maximum+bound)-bound)))==0.0,'Frozen finite-QK masking/underflow argument failed');scorebounds.append(dict(layer=depth,conservative_absolute_scaled_qk_bound=bound,masked_score_host_fp32=-maximum,masked_exponential_host_fp32=0.0))
 threshold=m.floats(parameters/'threshold.bin',1)[0];risk,rr=s['risk'],s['reference_risk'];need(s['threshold']==threshold and s['strict_reject']==(risk>threshold) and s['reference_strict_reject']==(rr>threshold) and (risk>threshold)==(rr>threshold) and s['end_to_end_score_tolerance'] is None,'Frozen strict decision/no-epsilon differs')
 return dict(schema='sentry_w4_padding_classification_v1',dispatch_class=W4_PADDING_CLASS,length=256,recognized=True,first_changed_saved_stage=first,embedding_ln=ln,all_a8_equal=not any(v['differences'] for v in sites),all_valid_token_a8_equal=valid_changes==0,all24_a8=sites,stage_statistics=stages,mask_evidence=dict(actual_process_input_folder=entry['input_folder'],actual_and_reference_key_mask_source=(p/'key_mask.bin').relative_to(ROOT).as_posix(),key_mask_hex=km.hex(),valid_tokens=sum(mask),padding_tokens=256-sum(mask),conservative_frozen_parameter_bounds=scorebounds,probabilities_captured=False,scope='Source inference: identical broadcast additive -FLT_MAX key mask, finite bounded QK and exp underflow; no probability capture'),captured_tail=dict(actual_pre_dot=precheck,actual_tanh_math=tanh,actual_tanh_vs_same_input_host=m.differences(post,ct['pooler_tanh'],all_outputs=True),both_terminal_reconstructions=tails,reference_pre_post_directly_captured=False),terminal=dict(logits=s['logits'],reference_logits=s['reference_logits'],risk=risk,reference_risk=rr,threshold=threshold,actual_margin=risk-threshold,reference_margin=rr-threshold,strict_decision_agrees=True),numerical_acceptance=None,score_approval=None)

def validate_w4_padding_policy_record(record,exports):
 """Pure read-only record check; adopted flags are never assigned or saved."""
 w4_padding_scope(exports)
 expected=dict(schema='sentry_w4_padding_dispatch_policy_v1',policy_id='bounded_connected_development_v2',dispatch_class=W4_PADDING_CLASS,mode='uniform_w4_qat_l256',identity=exports['identity'],selection=exports['selection'],precision_layers=exports['layers'],parameter_folder=exports['parameter_folder'],reference_export=(W4_PADDING_DATA/'export.json').relative_to(ROOT).as_posix(),source_binding=w4_padding_sources(),numerical_acceptance=None,score_approval=None)
 need(record.get('coordinator_adopted') is True and record.get('allow_prospective_dispatch') is True and all(record.get(k)==v for k,v in expected.items()),'Explicit independently adopted W4 padding policy/identity/selection/literal sources required')
 need(isinstance(record.get('evidence_paths'),list) and record['evidence_paths'],'W4 padding reviewed evidence required')
 for value in record['evidence_paths']:
  evidence=(ROOT/value).resolve();need(not Path(value).is_absolute() and evidence.is_relative_to(ROOT.resolve()) and evidence.is_file(),'W4 policy evidence missing/outside repository')
 return record

def validate_w4_padding_policy(path,exports):
 """READ an independently authored adopted policy; never create/adopt one."""
 p=Path(path);p=(p if p.is_absolute() else ROOT/p).resolve();need(p.is_relative_to(ROOT.resolve()) and p.is_file(),'Existing repository W4 padding policy required');record=validate_w4_padding_policy_record(read(p),exports)
 return dict(path=p.relative_to(ROOT.resolve()).as_posix(),record=record)

def w4_padding_compare_sources(dispatch_source,guard_source,policy):
 """No output writes: require actual scientific snapshot literals for new class."""
 binding=policy['record']['source_binding'];dispatch=w4_padding_literal_functions(dispatch_source,W4_PADDING_FUNCTIONS);guard=w4_padding_literal_functions(guard_source,W4_PADDING_GUARD_FUNCTIONS)
 need(dispatch==binding['classifier_functions'],'Actual W4 dispatch scientific source differs from independently bound policy/current source');need(guard==binding['guard_functions'],'Actual W4 guard/generator/operator-validation source differs')
 return dict(dispatch_functions=dispatch,guard_functions=guard)

def w4_padding_provenance(entry,policy):
 folder=ROOT/entry['folder'];observed=read(folder/'csim_run.json');base=folder.parent.parent;dispatch=(base/observed['source']).resolve();guard=(base/observed['guard_source']).resolve()
 need(dispatch.is_relative_to(base.resolve()) and guard.is_relative_to(base.resolve()) and dispatch.is_file() and guard.is_file(),'Associated actual W4 dispatch/guard snapshot missing/outside own run tree');bound=w4_padding_compare_sources(dispatch.read_text(encoding='utf-8-sig'),guard.read_text(encoding='utf-8-sig'),policy)
 return dict(dispatch_source=dispatch.relative_to(ROOT.resolve()).as_posix(),guard_source=guard.relative_to(ROOT.resolve()).as_posix(),scientific_source_functions=bound,scope='Literal actual per-window scientific dispatch/guard functions, current independently bound policy; stronger association applies only to new stored classified entries')

def w4_padding_entry_record(entry,policy,classified,provenance):
 """Pure record assembly, also permits an explicitly in-memory test roundtrip."""
 artifact=ROOT/entry['folder']/'recognized_w4_padding.json';base=artifact.parent.parent.parent
 return dict(schema='sentry_recognized_w4_padding_entry_v1',policy_path=policy['path'],policy_snapshot=(base/'recognized_w4_padding_policy.json').relative_to(ROOT).as_posix(),classification_path=artifact.relative_to(ROOT).as_posix(),helper_snapshots=[(base/('recognized_w4_padding_'+Path(name).name)).relative_to(ROOT).as_posix() for name in policy['record']['source_binding']['helper_sources']],classification_summary={k:classified[k] for k in ('schema','dispatch_class','length','recognized','first_changed_saved_stage','all_a8_equal','all_valid_token_a8_equal','numerical_acceptance','score_approval')},source_provenance=provenance,numerical_acceptance=None)

def recognized_w4_padding_entry(entry,exports,*,policy=None,write=False):
 """Historical entries need their own stored class or explicit casewise review."""
 stored=entry.get('recognized_w4_padding')
 if stored:
  validated=validate_w4_padding_policy(stored['policy_path'],exports);need(policy is None or policy==validated,'Stored W4 padding policy differs');policy=validated
 need(policy is not None,'Explicit prospective W4 padding opt-in required');need(policy==validate_w4_padding_policy(policy['path'],exports),'Provided W4 padding policy must independently validate');provenance=w4_padding_provenance(entry,policy);artifact=ROOT/entry['folder']/'recognized_w4_padding.json';base=artifact.parent.parent.parent;policycopy=base/'recognized_w4_padding_policy.json';snapshots=[(policycopy,(ROOT/policy['path']).read_bytes())]
 for name,text in policy['record']['source_binding']['helper_sources'].items():snapshots.append((base/('recognized_w4_padding_'+Path(name).name),text.encode('utf-8')))
 if write and not stored:
  for target,raw in snapshots:
   need(not target.exists() or target.read_bytes()==raw,'Preserve original W4 padding policy/helper snapshot')
   if not target.exists():target.write_bytes(raw)
  need(not artifact.exists(),'Preserve earlier uncommitted W4 classification')
 try:classified=classify_w4_padding(entry,exports)
 except Exception as error:
  if write and not stored:
   save(artifact,dict(schema='sentry_w4_padding_classification_failure_v1',recognized=False,error=repr(error),numerical_acceptance=None));entry['w4_padding_classification_attempt']=dict(path=artifact.relative_to(ROOT).as_posix(),recognized=False,policy_path=policy['path'])
  raise PropagationDiscrepancy('W4 padding diagnostic classification failed: '+str(error)) from error
 record=w4_padding_entry_record(entry,policy,classified,provenance)
 if stored:need(stored==record and read(artifact)==classified and all(p.read_bytes()==raw for p,raw in snapshots),'Saved W4 classification/source/policy changed')
 elif write:save(artifact,classified)
 else:raise PropagationDiscrepancy('Historical W4 discrepancy has no stored prospective classification')
 return record

def inspect(item,entry,exports,review=None,*,padding_policy=None,write_classification=False):
 from run_qualification_cases import validate_operator_bounds
 a,p=ROOT/entry['folder'],ROOT/item['folder'];s=read(a/'bridge_summary.json');observed=read(a/'csim_run.json');expected_command=[str(BUILD/'project/solution/csim/build/csim.exe')]
 compiled=read(BUILD/'compiled.json');source=prepared();first=ROOT/compiled['compiled_first_folder'];first_run=read(first/'csim_run.json');expected_compile=[read(ROOT/'configs/project.json')['hardware']['hls']['tool_executable'],'--mode','hls','--tcl',str(BUILD/'qualify.tcl')]
 need(compiled['state']=='returned' and compiled['passed'] is True and compiled['returncode']==0 and compiled['command']==expected_compile and compiled['source_preparation']==source and compiled['synthesis_recorded_utc']==source['synthesis_recorded_utc'] and Path(expected_command[0]).is_file(),'Successful actual compatible C compile/current source association required')
 need(first.is_relative_to((ROOT/'results/goal4/numerical_bridge/precision_qualification_v1').resolve()) and compiled['compiled_first_log']==(first/'csim.log').relative_to(ROOT).as_posix() and (first/'csim.log').is_file() and (first/'csim.log').stat().st_size>0,'Actual first compile log association missing')
 need(first_run['state']=='returned' and first_run['returncode']==0 and first_run['command']==compiled['command'] and first_run['identity']==compiled['first_identity'] and first_run['elapsed_seconds']==compiled['elapsed_seconds'] and read(first/'bridge_summary.json')['completed'] is True,'Actual successful compile-first process/summary association differs')
 need((a/'csim.log').is_file() and 'GATE_NUMERICAL_BRIDGE_COMPLETE calls=411 ' in (a/'csim.log').read_text(errors='replace') and 'GATE_NUMERICAL_BRIDGE_COMPLETE calls=411 ' in (first/'csim.log').read_text(errors='replace'),'Actual successful411-call completion log association missing')
 if observed['command']!=expected_command:
  need(compiled['compiled_first_folder']==entry['folder'] and observed['command']==compiled['command'],'Actual compile-first command association differs')
 need(observed['state']=='returned' and observed['returncode']==entry['returncode']==0 and observed['identity']==exports['identity'] and observed['window']==[item['case_index'],item['window_index']] and observed['input_folder']==item['folder'] and observed['parameter_folder']==exports['parameter_folder'],'Actual returned process/input/parameter association differs');need((a.parent.parent/observed['source']).is_file() and (a.parent.parent/observed['guard_source']).is_file(),'Actual dispatch/guard source snapshot missing');need(observed['memory_guard_file']==entry['memory_guard_file'] and (ROOT/observed['memory_guard_file']).is_file(),'Actual resource guard association missing')
 need(entry['summary']==s and entry['case_index']==item['case_index'] and entry['window_index']==item['window_index'] and entry['document_id']==item['document_id'] and entry['input_folder']==item['folder'] and entry['acceptance'] is None,'Actual case/source association differs')
 revision=next(line.split(' ',2)[2].strip().strip(chr(34)) for line in (BUILD/'fixed_fp32_service.hpp').read_text().splitlines() if line.startswith('#define GATE_FIXED_FP32_SCHEDULE_REVISION '));need(s['source_revision']==revision,'Actual service source revision differs')
 need(entry['returncode']==0 and s['completed'] is True and s['case']==item['document_id'] and s['encoder_arithmetic']==CONTRACT and s['service_calls_by_op']==[165,72,72,2,16,18,33,32,1],'Actual completion/mode/call count differs');bounds=validate_operator_bounds(a,s);need(bounds==entry['operator_bounds'] and bounds['calls']==411 and bounds['checked_calls_by_op']==[165,72,72,2,16,18,33,32,1] and s['same_input_codes_checked']==s['exact_accumulators_checked']==2359296 and s['same_input_quantization_exact'] is True and s['integer_accumulators_exact'] is True and s['same_input_codes_checked']==bounds['checked_values_by_op'][1] and s['exact_accumulators_checked']==bounds['checked_values_by_op'][2],'Original bounds/exact discrete coverage differs')
 from validate_qualification_exports import terminal
 terminal(a,'bridge',s['logits'],s['risk']);terminal(p,'reference',s['reference_logits'],s['reference_risk']);threshold=struct.unpack('<f',(ROOT/exports['parameter_folder']/'threshold.bin').read_bytes())[0];need(0<=s['risk']<=1 and 0<=s['reference_risk']<=1 and s['risk_absolute_error']==abs(s['risk']-s['reference_risk']) and s['threshold']==threshold and s['strict_reject']==(s['risk']>threshold) and s['reference_strict_reject']==(s['reference_risk']>threshold) and s['strict_decision_agrees'] is True and (s['risk']>threshold)==(s['reference_risk']>threshold) and s['end_to_end_score_tolerance'] is None,'Frozen strict-decision/no-terminal-epsilon contract differs')
 checks=[];need([v['name'] for v in s['integer_layers']]==[v['name'] for v in exports['layers']],'Actual24 integer layer membership differs')
 for layer,meta in zip(s['integer_layers'],exports['layers']):
  name=layer['name'];actual=(a/('actual_'+name+'_a8.bin')).read_bytes();reference=(p/(name+'_reference_a8.bin')).read_bytes();need(len(actual)==len(reference)==256*meta['k'] and layer['input_codes']==256*meta['k'] and layer['accumulators_checked']==256*meta['n'],'Actual A8/accumulator extent differs');differences=sum(x!=y for x,y in zip(actual,reference));checks.append(dict(name=name,values=len(actual),differences=differences));need(differences==layer['original_software_code_differences'],'Actual A8 discrepancy count differs')
 need(entry['a8']==checks,'Saved A8 checks changed')
 stages=['embedding_ln','layer0','layer1','layer2','layer3','logits','risk'];need([v['name'] for v in s['intermediate_errors']]==stages,'Actual captured stage membership differs')
 for error in s['intermediate_errors']:
  name=error['name'];count=2 if name=='logits' else 1 if name=='risk' else 65536;ab=(a/('bridge_'+name+'.bin')).read_bytes();rb=(p/('reference_'+name+'.bin')).read_bytes();need(len(ab)==len(rb)==count*4,'Actual stage extent differs');av=struct.unpack('<'+str(count)+'f',ab);rv=struct.unpack('<'+str(count)+'f',rb);need(all(math.isfinite(x) for x in (*av,*rv)) and error['values']==count and error['maximum_absolute_error']==max(abs(x-y) for x,y in zip(av,rv)),'Actual finite stage/error record differs');ds=[abs(x-y) for x,y in zip(av,rv)];total=squared=0.0
  for value in ds:total+=value;squared+=value*value
  need(error['mean_absolute_error']==total/count and error['rms_error']==math.sqrt(squared/count),'Actual ascending stage mean/RMS differs')
 for name in ('diagnostic_pooler_pre_tanh','diagnostic_pooler_post_tanh'):
  raw=(a/(name+'.bin')).read_bytes();need(len(raw)==1024 and all(math.isfinite(v) for v in struct.unpack('<256f',raw)),'Actual preserved tail capture missing/nonfinite: '+name)
 le=(a/'bridge_logits.bin').read_bytes()==(p/'reference_logits.bin').read_bytes();re=(a/'bridge_risk.bin').read_bytes()==(p/'reference_risk.bin').read_bytes();need(entry['logits_equal']==le and entry['risk_equal']==re,'Saved terminal bit checks differ');stored=entry.get('reviewed_resolution')
 if stored:need(review_resolution(stored['path'],entry,exports)==stored,'Stored exact coordinator review changed')
 if review:
  validated=review_resolution(review,entry,exports);need(not stored or stored==validated,'Do not replace earlier actual review');entry['reviewed_resolution']=validated;stored=validated
 padding=entry.get('recognized_w4_padding')
 if padding:need(recognized_w4_padding_entry(entry,exports,policy=padding_policy)==padding,'Stored W4 padding classification differs')
 elif padding_policy is not None and write_classification and not stored and (any(v['differences'] for v in checks) or not le or not re):
  padding=recognized_w4_padding_entry(entry,exports,policy=padding_policy,write=True);entry['recognized_w4_padding']=padding
 if (any(v['differences'] for v in checks) or not le or not re) and not stored and not padding:raise PropagationDiscrepancy('First propagated A8/terminal difference requires explicit causal review; diagnostic stop is not universal terminal-bit acceptance')
 return entry

def document_maxima(exports,bykey):
 threshold=struct.unpack('<f',(ROOT/exports['parameter_folder']/'threshold.bin').read_bytes())[0];documents=[]
 for case in exports['selection']['cases']:
  windows=[bykey[case['case_index'],i] for i in range(case['window_count'])];actual=max(v['summary']['risk'] for v in windows);reference=max(v['summary']['reference_risk'] for v in windows);need((actual>threshold)==(reference>threshold),'Actual document maximum strict decision differs');documents.append(dict(document_id=case['document_id'],windows=case['window_count'],actual_max=actual,reference_max=reference,threshold=threshold,actual_reject=actual>threshold,reference_reject=reference>threshold,decision_agrees=True,aggregation='Fresh maximum over this document actual windows'))
 return documents

def execute(args):
 from run_qualification_cases import ConnectedCMemoryGuard,paused
 need(os.name!='nt','C execution requires WSL Python with one assigned CPU6/7');paused();out=outdir(args.name);exports=validate_export(out,complete=True);exports=dict(exports,parameter_folder=(out/'parameters').relative_to(ROOT).as_posix());prepared();config=read(ROOT/'configs/project.json');output=out/'bounds_v1'
 # Resolution and every saved-state mutation hold the mode advisory lock too.
 with ConnectedCMemoryGuard(config,output,'precision_'+args.name,active=args.stage!='validate') as guard:
  execute_locked(args,out,exports,output,config,guard)

def execute_locked(args,out,exports,output,config,guard):
 from run_qualification_cases import run,validate_operator_bounds,paused
 statepath=output/'qualification.json';state=read(statepath) if statepath.exists() else dict(identity=exports['identity'],selection=exports['selection'],precision_layers=exports['layers'],windows=[],complete=False,numerical_acceptance=None,contract=CONTRACT)
 need(state.get('state')!='stopped_operator_service_process','Saved operator/service/process failure cannot be cleared by a propagation review')
 need(state['identity']==exports['identity'] and state['selection']==exports['selection'] and state['precision_layers']==exports['layers'] and state['contract']==CONTRACT and state['numerical_acceptance'] is None,'Saved actual map/source/approval association differs');items={(v['case_index'],v['window_index']):v for v in exports['completed_windows']};requested=None
 if args.reviewed_resolution:
  r=read(Path(args.reviewed_resolution) if Path(args.reviewed_resolution).is_absolute() else ROOT/args.reviewed_resolution);requested=r['case_index'],r['window_index'];need(state.get('state')=='stopped_first_unexplained_discrepancy' and tuple(state.get('active_window',[]))==requested,'Only the current saved A8/terminal discrepancy can be reviewed');need(any((e['case_index'],e['window_index'])==requested for e in state['windows']),'Only existing actual discrepancy can be reviewed')
 padding_path=getattr(args,'recognized_w4_padding_policy',None)
 padding_policy=validate_w4_padding_policy(padding_path,exports) if padding_path else None
 bykey={}
 for entry in state['windows']:
  key=entry['case_index'],entry['window_index'];need(key not in bykey and key in items,'Duplicate/unassigned saved C window');inspect(items[key],entry,exports,args.reviewed_resolution if requested==key else None,padding_policy=padding_policy);bykey[key]=entry
 need(not state.get('state','').startswith('stopped_') or requested is not None,'Saved stop requires explicit review')
 if args.stage!='validate' or requested is not None:state['state']='bounded_handoff';save(statepath,state)
 if state['complete']:need(set(bykey)==set(items) and state['documents']==document_maxima(exports,bykey),'Complete actual window/document-max reset association differs')
 if args.stage=='validate':print(json.dumps(dict(validated_windows=len(bykey),complete=state['complete'],numerical_acceptance=None)));return
 if args.stage!='validate':
  dispatch=output/f'c_dispatch_{len([p for p in output.glob("c_dispatch_*.py") if not p.stem.endswith("_guard")])+1:03d}.py';write(dispatch,Path(__file__).read_bytes());guard_source=output/(dispatch.stem+'_guard.py');write(guard_source,(ROOT/'scripts/run_qualification_cases.py').read_bytes());exe=BUILD/'project/solution/csim/build/csim.exe';compiledpath=BUILD/'compiled.json';targets=[v for v in exports['completed_windows'] if (v['case_index'],v['window_index']) not in bykey]
  if args.stage=='compile':need(not compiledpath.exists() and targets,'Preserve existing C compile or no unrun windows');targets=targets[:1]
  else:need(compiledpath.is_file() and read(compiledpath)['passed'] and exe.is_file() and read(compiledpath)['synthesis_recorded_utc']==prepared()['synthesis_recorded_utc'],'Successful separate compatible precision C compile/source association required');targets=targets[:args.max_new_windows]
  for item in targets:
   paused();key=item['case_index'],item['window_index'];folder=output/f'case{key[0]:02d}'/f'window{key[1]:03d}';need(not folder.exists(),'Preserve uncommitted actual C output');folder.mkdir(parents=True);env=os.environ.copy();env.update(GATE_NUMERICAL_BRIDGE_DIR=str(ROOT/item['folder']),GATE_NUMERICAL_BRIDGE_PARAMETER_DIR=str(out/'parameters'),GATE_NUMERICAL_BRIDGE_CASE=item['document_id'],GATE_NUMERICAL_BRIDGE_DIAGNOSTIC_DIR=str(folder));env.pop('GATE_NUMERICAL_BRIDGE_FLOATING',None)
   command=[config['hardware']['hls']['tool_executable'],'--mode','hls','--tcl',str(BUILD/'qualify.tcl')] if args.stage=='compile' else [str(exe)];observed=dict(state='running',command=command,window=list(key),identity=exports['identity'],source=dispatch.name,guard_source=guard_source.name,input_folder=item['folder'],parameter_folder=(out/'parameters').relative_to(ROOT).as_posix(),timeout_seconds=900,memory_guard_file=guard.path.relative_to(ROOT).as_posix(),board_access=False,new_synthesis=False);save(folder/'csim_run.json',observed);state['active_window']=list(key);save(statepath,state)
   try:
    if args.stage=='compile':
     import fcntl
     with (BUILD/'.compile.lock').open('a') as lock:
      fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);need(not compiledpath.exists(),'Another actual precision compile exists');save(compiledpath,dict(state='running',passed=False,command=command,new_synthesis=False));rc,elapsed=run(command,BUILD,env,folder/'csim.log',900,memory_guard=guard)
    else:rc,elapsed=run(command,BUILD,env,folder/'csim.log',900,memory_guard=guard)
    observed.update(state='returned',returncode=rc,elapsed_seconds=elapsed);save(folder/'csim_run.json',observed)
    if args.stage=='compile':save(compiledpath,dict(state='returned',passed=rc==0 and exe.is_file(),command=command,compiled_first_folder=folder.relative_to(ROOT).as_posix(),compiled_first_log=(folder/'csim.log').relative_to(ROOT).as_posix(),source_preparation=prepared(),first_identity=exports['identity'],synthesis_recorded_utc=prepared()['synthesis_recorded_utc'],new_synthesis=False,returncode=rc,elapsed_seconds=elapsed))
    need((folder/'bridge_summary.json').exists(),'Actual operator/service failure; inspect preserved log and per-call records');summary=read(folder/'bridge_summary.json');checks=[]
    for layer in summary['integer_layers']:
     name=layer['name'];ab=(folder/('actual_'+name+'_a8.bin')).read_bytes();rb=(ROOT/item['folder']/(name+'_reference_a8.bin')).read_bytes();checks.append(dict(name=name,values=len(ab),differences=sum(a!=b for a,b in zip(ab,rb))))
    entry=dict(case_index=key[0],window_index=key[1],document_id=item['document_id'],label=item['label'],returncode=rc,elapsed_seconds=elapsed,folder=folder.relative_to(ROOT).as_posix(),input_folder=item['folder'],summary=summary,operator_bounds=validate_operator_bounds(folder,summary),a8=checks,logits_equal=(folder/'bridge_logits.bin').read_bytes()==(ROOT/item['folder']/'reference_logits.bin').read_bytes(),risk_equal=(folder/'bridge_risk.bin').read_bytes()==(ROOT/item['folder']/'reference_risk.bin').read_bytes(),memory_guard_file=guard.path.relative_to(ROOT).as_posix(),acceptance=None);state['windows'].append(entry);save(statepath,state);inspect(item,entry,exports,padding_policy=padding_policy,write_classification=True);bykey[key]=entry;state['active_window']=None;save(statepath,state)
   except Exception as error:
    observed.update(diagnostic_stop=str(error));
    if observed['state']=='running':observed['state']='stopped'
    save(folder/'csim_run.json',observed);state.update(state='stopped_first_unexplained_discrepancy' if isinstance(error,PropagationDiscrepancy) else 'stopped_operator_service_process',error=str(error));observed['diagnostic_stop_state']=state['state'];save(folder/'csim_run.json',observed);save(statepath,state);raise
 state['complete']=set(bykey)==set(items)
 if state['complete']:state['documents']=document_maxima(exports,bykey)
 state['state']='complete_pending_acceptance' if state['complete'] else 'bounded_handoff';save(statepath,state);print(json.dumps(dict(saved_windows=len(bykey),complete=state['complete'],numerical_acceptance=None)))

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--name',required=True);ap.add_argument('--stage',choices=('preview','prepare','compile','run','validate'),default='preview');ap.add_argument('--execute',action='store_true');ap.add_argument('--max-new-windows',type=int,default=1);ap.add_argument('--reviewed-resolution');ap.add_argument('--recognized-w4-padding-policy',metavar='PATH');args=ap.parse_args();need(not args.recognized_w4_padding_policy or (args.name=='uniform_w4_qat' and args.stage in ('preview','run','validate')),'Optional W4 padding policy is uniform_w4_qat run/validate only; no compile');need(args.max_new_windows>0,'Positive bounded actual window count');out=outdir(args.name);generated=generate_precision_tb((ROOT/'hardware/numerical_bridge_tb.cpp').read_text())
 if not args.execute or args.stage=='preview':print(json.dumps(dict(execute=False,name=args.name,input_export=str(out/'export.json'),input_ready=(out/'export.json').exists(),build=str(BUILD),build_prepared=(BUILD/'prepared.json').exists(),compiled=(BUILD/'compiled.json').exists(),checked_substitutions=[dict(before=a,after=b) for a,b in SUBSTITUTIONS],generated_source_characters=len(generated),actual_expected_calls=411,timeout_seconds=900,resource_guard='Current ConnectedCMemoryGuard, one assigned CPU6/7 and common2-worker locks, observed WSL4GiB/Windows2GiB floors',prerequisites=['Explicit root slot release','All66 actual B1 exports for supplied map/checkpoint','Separate source prepare and one necessary C compile'],scope='Actual common FP C service + checked decoded W4/W8 integer contract; packed bytes validated separately; not full RTL',recognized_w4_padding_policy=args.recognized_w4_padding_policy,prospective_only=True,numerical_acceptance=None),indent=2));return
 if args.stage=='prepare':print(json.dumps(prepare(),indent=2));return
 execute(args)
if __name__=='__main__':main()