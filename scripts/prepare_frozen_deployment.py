"""Prepare a real frozen QAT deployment arena/job at true L128/256/512.

Preview is stdlib-only. --execute reads one actual checkpoint and packs bytes;
it performs no tokenizer/model forward, compiler/vendor run or physical access.
Finalist U/B/C requires genuine validated development freeze/candidate evidence.
Explicit pre-freeze bringup-w8 uses only the existing saved uniformW8 checkpoint.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,math,struct,sys,shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT))
from export_precision_qualification import GROUPS,SCHEME,pack_codes,unpack_codes
from plan_detector_commands import create_plan,write_rtl_plan,exact_threshold_bits
WORKSPACE=('embedding_word','embedding_type','hidden','query','key','value','context','projected','residual','normalized','ffn_pre_gelu','ffn_activated','a8','accumulator','key_head','value_head_transposed','query_slab','attention_tile','scores','probabilities','mask_slab','pooler_output','pooled','logits','class_probabilities')
SUFFIX={'query':'attention.self.query','key':'attention.self.key','value':'attention.self.value','attention_output':'attention.output.dense','ffn_input':'intermediate.dense','ffn_output':'output.dense'}

def need(ok,why):
 if not ok:raise ValueError(why)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,value):Path(p).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def path(value):
 text=str(value).replace('\\','/');prefix='/home/jp19126/Projects/SENTRY/'
 if text.startswith(prefix):return ROOT/text[len(prefix):]
 p=Path(text);return p if p.is_absolute() else ROOT/p

def rel(p):return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()
def linux(p):return '/home/jp19126/Projects/SENTRY/'+rel(p)

def selected_window(length,key):
 selection=read(ROOT/'results/goal4/numerical_bridge/development_cases_v1/selection.json')
 need(selection['case_count']==32 and sum(c['window_count'] for c in selection['cases'])==66,'Frozen32/66 selection required')
 if length==256:
  found=[dict(key=f"c{c['case_index']:02d}_w{i:03d}",case_index=c['case_index'],window_index=i,document_id=c['document_id'],source_group_id=c['source_group_id'],partition=c['partition'],label=c['label'],inputs=w) for c in selection['cases'] for i,w in enumerate(c['windows']) if f"c{c['case_index']:02d}_w{i:03d}"==key]
  source=ROOT/'results/goal4/numerical_bridge/development_cases_v1/selection.json';snapshot=selection
 else:
  from bringup_detector_shapes import load_plan,plan_path
  snapshot=load_plan(length);found=[w for w in snapshot['windows'] if w['key']==key];source=plan_path(length)
 need(len(found)==1,'Select exactly one genuine prepared frozen window: '+key);window=found[0];fields=window['inputs']
 for field,limit in [('input_ids',30522),('token_type_ids',2),('attention_mask',2)]:need(len(fields[field])==length and all(type(v) is int and 0<=v<limit for v in fields[field]),'Genuine input extent/range differs: '+field)
 valid=sum(fields['attention_mask']);need(1<=valid<=length,'Invalid frozen valid length');need(fields['attention_mask']==[1]*valid+[0]*(length-valid),'Only actual tokenizer right-padding is supported')
 return window,source,snapshot,valid

def frozen_role(cfg,role):
 """Use the canonical freeze validator; unrelated held-out dispatch flags do not authorize packing."""
 target=path(cfg['final_evaluation']['development_freeze_path']);need(target.is_file(),'Actual standard development freeze is missing; no handwritten role/map substitutes')
 from evaluate_frozen_finalists import prerequisites
 preview,frozen=prerequisites(cfg,target)
 fatal=[v for v in preview['blockers'] if v.startswith('Development freeze:') or v.startswith('methods.') or 'identity' in v or 'batch' in v or 'unresolved' in v]
 need(not fatal,'Actual freeze/development evidence failed: '+'; '.join(fatal))
 need(frozen is not None and frozen['state']=='frozen_development' and set(frozen['finalists'])=={'floating','U','B','C'},'Actual floating/U/B/C freeze required');entry=frozen['finalists'][role]
 need(entry['identity']['qat'] is True and entry['precision_map']==entry['identity']['precision_map'],'Actual post-QAT finalist precision identity required')
 return target,frozen,entry

def final_calibration_threshold(entry,freeze,role,reported):
 """Recompute only saved actual calibration-score chunks; no held-out text/model read."""
 folder=ROOT/'results/goal7/final_quality'/role/'final_calibration';progress=read(folder/'progress.json')
 need(progress['identity']==entry['identity'] and progress['freeze']==freeze and progress['complete'] is True,'Actual complete final calibration score identity/freeze missing')
 values=[];groups=set();seen=set();position=0
 for commit in progress['chunks']:
  need(commit['start']==position and Path(commit['file']).name==commit['file'],'Saved calibration commit order/path differs');chunk=read(folder/commit['file'])
  need(chunk['identity']==entry['identity'] and chunk['start']==position and chunk['count']==commit['count']==len(chunk['documents']) and chunk['count']>0,'Saved actual calibration chunk association differs')
  for row in chunk['documents']:
   need(row['document_id'] not in seen and row['window_count']==len(row['window_scores']) and row['window_count']>0 and row['risk_score']==max(row['window_scores']),'Actual saved calibration document max/reset differs');seen.add(row['document_id'])
   for value in row['window_scores']:exact_threshold_bits(value)
   if row['label']==0:values.append(row['risk_score']);groups.add(row['source_group_id'])
  position+=chunk['count']
 need(position==progress['next_document'] and len(values)==reported['calibration_benign_documents'] and len(groups)==reported['calibration_source_groups'],'Actual calibration document/source-group count differs')
 from gate_eval import fit_threshold
 need(fit_threshold(values,freeze['evaluation']['fpr_target'])==reported['primary'],'Recorded final threshold differs from actual observed FP32 benign calibration scores')
 return dict(progress=rel(folder/'progress.json'),actual_documents=position,benign_documents=len(values),benign_source_groups=len(groups),threshold_refitted_for_validation_only=True)

def threshold_record(args,selection,entry,freeze,checkpoint,precision):
 """Only genuine recorded thresholds; purpose/length transfer is never implicit."""
 source=path(args.threshold_record);record=read(source)
 if args.mode=='bringup-w8':
  need(args.purpose=='bringup','Pre-freeze uniformW8 only permits bringup purpose')
  expected=ROOT/'results/goal4/numerical_bridge/development_qualification_w8_v1/export.json';need(source.resolve()==expected.resolve(),'Bringup threshold must name the actual ordered W8 export.json')
  identity=record['identity'];need(record['complete'] and len(record['completed_windows'])==66 and record['selection']==selection,'Complete actual ordered W8 B1/selection association required')
  need(identity['arithmetic_identity']=='bert_mini_ordered_fp32_frozen_quant_v1' and identity['qat'] is True and identity['precision_map']==precision and path(identity['scored_checkpoint']).resolve()==checkpoint.resolve(),'Bringup actual ordered threshold computation/checkpoint/map identity differs')
  need((source.parent/'gate_fpga_reference.py').read_bytes()==(ROOT/'gate_fpga_reference.py').read_bytes(),'Bringup ordered arithmetic source changed')
  raw=(source.parent/'parameters/threshold.bin').read_bytes();need(len(raw)==4,'Actual ordered threshold.bin missing');value=struct.unpack('<f',raw)[0];need(value==selection['sampling']['historical_operating_threshold'],'Ordered threshold bytes/frozen selection differ');source_role='uniformW8_bringup';scope='Actual ordered export threshold; no historical score reuse or pre-freeze score approval'
 elif args.purpose=='development_agreement':
  expected=path(entry['development_scores'])/'candidate.json';need(source.resolve()==expected.resolve() and record['identity']==entry['identity'],'Actual frozen finalist development candidate/threshold identity differs');identity=entry['identity'];value=record['metrics']['primary']['threshold'];source_role=args.role;scope='Actual finalist development temporary threshold; no final-calibration quality claim'
  documents=expected.parent/'search_temporary_threshold_documents.jsonl';observed=False
  with documents.open(encoding='utf-8') as stream:
   for line in stream:
    row=json.loads(line)
    if row['label']==0 and row['risk_score']==value:observed=True
  need(observed,'Development threshold is not an actual observed benign document FP32 score')
 else:
  need(args.purpose=='final_evaluation','Finalist purpose must be development_agreement or final_evaluation');expected=ROOT/'results/goal7/final_quality/thresholds.json';need(source.resolve()==expected.resolve() and record['freeze']==freeze and args.role in record['methods'],'Actual final-calibration threshold freeze/role differs');identity=freeze['finalists'][args.role]['identity'];value=record['methods'][args.role]['primary'];source_role=args.role;scope='Existing actual final-calibration threshold with exact frozen length/window identity'
  need(record['methods'][args.role]['attack_rule']=='score > threshold' and record['methods'][args.role]['fit_policy']==freeze['evaluation']['threshold_rule'],'Actual final threshold fit/strict rule differs')
 need(identity['evaluation']['attack_rule']=='score > threshold','Threshold strict rule differs');source_length=identity['windowing']['primary_length'];transfer=source_length!=args.length
 need(not transfer or args.allow_threshold_length_transfer and args.purpose in ('bringup','development_agreement'),'Threshold length differs; explicit non-final length transfer is required')
 need(args.purpose!='final_evaluation' or identity['windowing']==freeze['windowing'] and source_length==args.length and not args.allow_threshold_length_transfer,'Final evaluation requires exact frozen threshold/window policy/length; transfer prohibited')
 bits=exact_threshold_bits(value)
 calibration_validation=final_calibration_threshold(entry,freeze,args.role,record['methods'][args.role]) if args.purpose=='final_evaluation' else None
 association=dict(calibration_validation=calibration_validation,scope=scope,record=record,threshold_identity=identity,threshold_role=source_role,purpose=args.purpose,source=rel(source),source_length=source_length,deployed_length=args.length,threshold_transfer=transfer,transfer_explicitly_requested=args.allow_threshold_length_transfer,threshold_bits_hex=f'{bits:08x}',attack_rule='score > threshold',shape_numerical_approval=None)
 return value,bits,source,association

def prepare_request(args):
 cfg=read(ROOT/'configs/project.json');plan=create_plan(cfg,length=args.length);window,input_source,input_record,valid=selected_window(args.length,args.window_key);run=read(ROOT/'results/goal3/run.json');calibration=read(path(run['calibration_file']));need(calibration['scheme_id']==SCHEME,'Existing training calibration scheme differs')
 freeze_source=freeze=entry=None
 if args.mode=='bringup-w8':
  need(args.role is None and args.purpose=='bringup','Pre-freeze bringup has no selected finalist role/final threshold');checkpoint=ROOT/'checkpoints/quantized/qat_w8_a8';precision={g:8 for g in GROUPS};identity=dict(scope='Explicit uniformW8 pre-freeze true-length bring-up only',checkpoint=rel(checkpoint),precision_map=precision,approved=False)
 else:
  need(args.role in ('U','B','C'),'Finalist requires explicit actual frozen U/B/C role');freeze_source,freeze,entry=frozen_role(cfg,args.role);checkpoint=path(entry['checkpoint']);precision=entry['precision_map'];identity=entry['identity']
 quant=read(checkpoint/'quantization.json');need(path(quant['training']['from_floating_checkpoint']).resolve()==path(run['selected_checkpoint']).resolve(),'Actual QAT floating source checkpoint differs');need(quant['scheme_id']==SCHEME and quant['precision_map']==precision and quant['calibration']==calibration,'Actual QAT metadata/map/frozen original scales differ');need(set(precision)==set(GROUPS) and all(type(v) is int and v in (4,8) for v in precision.values()),'Real16group W4/W8 map required');need((checkpoint/'quantized_state.pt').is_file(),'Actual saved QAT state missing')
 model=read(checkpoint/'config.json');need(tuple(model[k] for k in ('vocab_size','type_vocab_size','max_position_embeddings','hidden_size','num_hidden_layers','num_attention_heads','intermediate_size'))==(30522,2,512,256,4,4,1024),'Actual finalist model shape differs from existing fixed plan')
 selection=read(ROOT/'results/goal4/numerical_bridge/development_cases_v1/selection.json');threshold,bits,threshold_source,threshold_association=threshold_record(args,selection,entry,freeze,checkpoint,precision)
 template=read(path(args.template_job));base=template['selected_deployment'];need(base['arena_bytes']==67108864 and 0<=base['mailbox_base']<2**64 and base['arena_base']%4096==0 and 0<base['arena_sd_lba']<2**32,'Explicit template aperture/SD choices invalid');need(0<=args.model_id<2**64,'Explicit model-ID binding out of range')
 mask=sum((precision[g]==8)<<i for i,g in enumerate(plan['precision']['groups']));buffers=plan['buffers'];deployment={k:base[k] for k in ('mailbox_base','arena_base','mailbox_poll_limit','arena_sd_lba')};deployment.update(model_id=args.model_id,tokens=args.length,arena_bytes=67108864,token_offset=buffers['token_ids']['offset'],type_offset=buffers['type_ids']['offset'],mask_offset=buffers['attention_mask']['offset'],threshold_bits=bits,w8_mask=mask)
 out=path(args.output);need(out.resolve().is_relative_to((ROOT/'results/goal4').resolve()),'Fresh deployment output must stay in results/goal4')
 return dict(config=cfg,plan=plan,window=window,input_source=input_source,input_record=input_record,valid=valid,checkpoint=checkpoint,quant=quant,calibration=calibration,identity=identity,precision=precision,freeze_source=freeze_source,freeze=freeze,entry=entry,threshold=threshold,threshold_source=threshold_source,threshold_association=threshold_association,deployment=deployment,template=template,out=out)

def header(deployment):
 result='#include "xparameters.h"\n#include "sentry_supervisor.h"\nconst SentryDeployment deployment={\n'
 for key,value in deployment.items():result+=f' .{key}={value}ULL,\n'
 return result+' .uart_base=XPAR_XUARTPSV_0_BASEADDR,\n .sd_base=XPAR_XSDPS_0_BASEADDR\n};\n'

def execute(args,r):
 need(not any((ROOT/p).exists() for p in ('results/goal2/pause.request','results/goal3/pause.request')),'Pause marker present');out=r['out'];need(not out.exists(),'Preserve existing/failed deployment output; choose fresh directory')
 import torch
 out.mkdir(parents=True);sources=out/'sources';sources.mkdir();shutil.copyfile(r['checkpoint']/'quantized_state.pt',sources/'quantized_state.pt');torch.set_num_threads(1);state=torch.load(sources/'quantized_state.pt',map_location='cpu',weights_only=True);plan=r['plan'];buffers=plan['buffers'];arena=bytearray(plan['arena']['planned_bytes']);need(len(arena)==67108864 and plan['arena']['physical_base'] is None,'No physical allocation is represented')
 intervals=sorted((b['offset'],b['offset']+b['bytes'],name) for name,b in buffers.items());need(all(begin>=0 and begin%4096==0 and end<=len(arena) and begin<end for begin,end,name in intervals) and all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])),'Invalid/overlapping true-length plan buffers');written={};weights=[];consumed=set()
 def tensor(key,shape):
  need(key in state,'Missing real checkpoint parameter; no zero substitution: '+key);t=state[key];need(tuple(t.shape)==tuple(shape) and t.dtype==torch.float32 and bool(torch.isfinite(t).all()),'Invalid actual saved tensor: '+key);consumed.add(key);return t.detach().contiguous()
 def raw(t):return t.numpy().astype('<f4',copy=False).tobytes()
 def put(name,payload,source,padding=0):
  b=buffers[name];need(name not in written and len(payload)+padding==b['bytes'],'Payload/slot extent or duplicate: '+name);value=payload+bytes(padding);arena[b['offset']:b['offset']+b['bytes']]=value;written[name]=dict(offset=b['offset'],bytes=b['bytes'],payload_bytes=len(payload),padding_bytes=padding,padding_value=0 if padding else None,source=source)
 fields=r['window']['inputs']
 for field,name,kind in [('input_ids','token_ids','I'),('token_type_ids','type_ids','I'),('attention_mask','attention_mask','B')]:put(name,struct.pack('<'+kind*args.length,*fields[field]),rel(r['input_source'])+' '+args.window_key+' '+field)
 for name,key,shape in [('word_table','bert.embeddings.word_embeddings.weight',(30522,256)),('type_table','bert.embeddings.token_type_embeddings.weight',(2,256)),('position_table','bert.embeddings.position_embeddings.weight',(512,256))]:put(name,raw(tensor(key,shape)),key)
 norms=[('embedding_ln','bert.embeddings.LayerNorm')]+[(f'layer{i}_{name}_ln',f'bert.encoder.layer.{i}.'+suffix) for i in range(4) for name,suffix in [('attention','attention.output.LayerNorm'),('output','output.LayerNorm')]]
 for name,prefix in norms:
  for suffix,key in [('gamma','weight'),('beta','bias')]:put(name+'_'+suffix,raw(tensor(prefix+'.'+key,(256,))),prefix+'.'+key)
 scales=[None]*24;need({m['input_scale_index'] for m in plan['matrices']}==set(range(24)),'Actual24matrix/scales schedule differs')
 for m in plan['matrices']:
  name=m['name'];layer,suffix=name.split('_',1);prefix=f"bert.encoder.layer.{int(layer[5:])}.{SUFFIX[suffix]}";n,k=m['outputs'],m['inner'];group=plan['precision']['groups'][m['precision_group']];bits=r['precision'][group];weight=tensor(prefix+'.weight',(n,k));bias=tensor(prefix+'.bias',(n,));scale=tensor(prefix+'.input_scale',())
  need(float(scale)>0 and raw(scale)==struct.pack('<f',r['calibration']['modules'][prefix]['input_scale']),'Actual frozen training A8 scale differs');need(float(tensor(prefix+'.weight_width',()))==bits and float(tensor(prefix+'.activation_width',()))==8 and float(tensor(prefix+'.zero_point',()))==0,'Actual saved precision/zero-point differs')
  row_scales={width:tensor(prefix+f'.weight_scale_{width}',(n,1)) for width in (4,8)}
  for width,table in row_scales.items():need(bool((table>0).all()),'Invalid frozen row scale');put(name+f'_weight_scale{width}',raw(table),prefix+f'.weight_scale_{width}')
  lo,hi=-(1<<(bits-1)),(1<<(bits-1))-1;codes=torch.clamp(torch.round(weight/row_scales[bits]),lo,hi).to(torch.int8).contiguous().numpy().tobytes();packed=pack_codes(codes,bits);need(unpack_codes(packed,bits)==codes and len(packed)==n*k*bits//8,'Actual W4/W8 packing roundtrip differs');put(name+'_weights',packed,prefix+'.weight / actual frozen selected scale; nearest-even FP32 quotient',n*k-len(packed));put(name+'_bias',raw(bias),prefix+'.bias');scales[m['input_scale_index']]=raw(scale)
  weights.append(dict(name=name,group=group,bits=bits,k=k,n=n,decoded_bytes=len(codes),packed_bytes=len(packed),slot_bytes=n*k,padding_bytes=n*k-len(packed),code_min=min(v if v<128 else v-256 for v in codes),code_max=max(v if v<128 else v-256 for v in codes),packing='row-major[N,K], earlier K low nibble, signed two-complement W4' if bits==4 else 'row-major[N,K], signed two-complement W8',frozen_scale_tables=[4,8]))
 put('input_scales',b''.join(scales),'Actual24 frozen QAT input scales in exact true-length matrix order')
 for name,key,shape in [('pooler_weight','bert.pooler.dense.weight',(256,256)),('pooler_bias','bert.pooler.dense.bias',(256,)),('classifier_weight','classifier.weight',(2,256)),('classifier_bias','classifier.bias',(2,))]:put(name,raw(tensor(key,shape)),key)
 zero=set(WORKSPACE)|{'dummy_zero'};need(set(buffers)==set(written)|zero and not set(written)&zero,'Every actual parameter buffer must be supplied; zero only declared workspace')
 (out/'arena.bin').write_bytes(arena);save(out/'command_plan.json',plan);write_rtl_plan(plan,out);(sources/'prepare_frozen_deployment.py').write_bytes(Path(__file__).read_bytes());shutil.copyfile(ROOT/'scripts/plan_detector_commands.py',sources/'plan_detector_commands.py');shutil.copyfile(ROOT/'gate_quant.py',sources/'gate_quant.py');save(sources/'quantization.json',r['quant']);save(sources/'input_plan.json',r['input_record']);save(sources/'threshold_record.json',r['threshold_association']);save(sources/'template_job.json',r['template']);save(sources/'model_identity.json',r['identity']);
 if r['freeze'] is not None:save(sources/'development_freeze.json',r['freeze'])
 (sources/'deployment.h').write_text(header(r['deployment']),encoding='utf-8')
 with (out/'arena.bin').open('rb') as f:
  for begin,end,name in intervals:
   f.seek(begin);payload=f.read(end-begin);need(payload==arena[begin:end],'Actual arena readback differs: '+name)
   if name in zero:need(not any(payload),'Nonzero declared workspace: '+name)
   elif written[name]['padding_bytes']:need(not any(payload[written[name]['payload_bytes']:]),'Nonzero fixed-slot padding: '+name)
 result=dict(complete=True,scope='Actual saved QAT parameter/input packing only; no forward, numerical qualification or physical allocation',mode=args.mode,role=args.role,length=args.length,plan_identity=plan['identity'],plan_recorded_utc=plan['recorded_utc'],checkpoint=rel(r['checkpoint']),checkpoint_snapshot='sources/quantized_state.pt',model_identity=r['identity'],precision_map=r['precision'],config_w8_mask=r['deployment']['w8_mask'],threshold_bits_hex=f"{r['deployment']['threshold_bits']:08x}",threshold_source=rel(r['threshold_source']),threshold_scope=r['threshold_association']['scope'],threshold_role=r['threshold_association']['threshold_role'],purpose=args.purpose,threshold_transfer=r['threshold_association']['threshold_transfer'],threshold_source_length=r['threshold_association']['source_length'],threshold_deployed_length=args.length,attack_rule='score > threshold',input_source=rel(r['input_source']),window=r['window'],valid_length=r['valid'],image='arena.bin',image_bytes=len(arena),physical_base=None,allocation_confirmed=False,physical_allocation_confirmed=False,runtime_mailbox_numeric_ids=None,written_buffers=written,zero_buffers=sorted(zero),weights=weights,readback_buffers_checked=len(buffers),consumed_checkpoint_keys=sorted(consumed),frozen_scale_policy='Both actual saved W4/W8 row tables and original training A8 scales copied without recalibration',unused_alignment_and_tail='Zero padding; never substituted model parameters',numerical_acceptance=None,source_script='scripts/prepare_frozen_deployment.py')
 save(out/'preparation.json',result);job=dict(output=linux(out),sources=linux(sources),arena_record=linux(out/'preparation.json'),selected_deployment=r['deployment'],model_identity=r['identity'],development_freeze=None if r['freeze_source'] is None else rel(r['freeze_source']),explicit_template_choices=rel(path(args.template_job)),physical_allocation_confirmed=False,physical_board_activity=False,role=args.role,mode=args.mode);save(out/'job.json',job);validate_saved(out);print(json.dumps(dict(complete=True,output=rel(out),length=args.length,window=args.window_key,w8_mask=result['config_w8_mask'],weights=len(weights),readback_buffers=len(buffers),model_forward=False,physical_allocation_confirmed=False,numerical_acceptance=None),indent=2))

def validate_saved(out):
 """Read-only stdlib readback of concrete packed image, job and declared slots."""
 out=path(out);p=read(out/'preparation.json');plan=read(out/'command_plan.json');job=read(out/'job.json');d=job['selected_deployment'];need(p['complete'] and p['image_bytes']==67108864 and p['numerical_acceptance'] is None and p['physical_base'] is None and p['allocation_confirmed'] is False,'Packed scope/approval/allocation differs');need(p['plan_identity']==plan['identity'] and p['plan_recorded_utc']==plan['recorded_utc'],'True-length packed plan association differs');need(d['tokens']==p['length'] and d['w8_mask']==p['config_w8_mask'] and d['threshold_bits']==int(p['threshold_bits_hex'],16) and job['model_identity']==p['model_identity'],'Job/model/map/threshold differs');need((out/'sources/deployment.h').read_text()==header(d),'Header/job deployment differs');need((out/p['image']).stat().st_size==67108864,'Actual packed arena size differs')
 intervals=sorted((b['offset'],b['offset']+b['bytes'],name) for name,b in plan['buffers'].items());need(all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])),'Saved arena buffers overlap')
 need(p['image']=='arena.bin' and p['checkpoint_snapshot']=='sources/quantized_state.pt' and (out/p['checkpoint_snapshot']).is_file(),'Actual saved checkpoint/image provenance missing')
 need(read(out/'sources/quantization.json')==read(path(p['checkpoint'])/'quantization.json') and read(out/'sources/model_identity.json')==p['model_identity'],'Actual QAT metadata/model identity snapshot differs')
 need((out/'sources/gate_quant.py').read_bytes()==(ROOT/'gate_quant.py').read_bytes(),'Actual frozen quantization scientific source changed')
 ta=read(out/'sources/threshold_record.json');need(ta['record']==read(path(p['threshold_source'])) and ta['threshold_bits_hex']==p['threshold_bits_hex'] and ta['threshold_role']==p['threshold_role'] and ta['purpose']==p['purpose'] and ta['source_length']==p['threshold_source_length'] and ta['deployed_length']==p['length'] and ta['threshold_transfer']==p['threshold_transfer'] and ta['attack_rule']==p['attack_rule']=='score > threshold','Exact actual threshold/source/purpose/length association differs')
 need(p['precision_map']==read(out/'sources/quantization.json')['precision_map'] and p['config_w8_mask']==sum((p['precision_map'][g]==8)<<i for i,g in enumerate(plan['precision']['groups'])),'Actual precision bits/map differs')
 need(set(plan['buffers'])==set(p['written_buffers'])|set(p['zero_buffers']) and not set(p['written_buffers'])&set(p['zero_buffers']),'Saved missing/ambiguous parameter mapping')
 with (out/p['image']).open('rb') as f:
  for begin,end,name in intervals:
   f.seek(begin);raw=f.read(end-begin);need(len(raw)==end-begin,'Saved buffer readback extent')
   if name in p['zero_buffers']:need(not any(raw),'Saved nonzero workspace')
   else:
    written=p['written_buffers'][name];need(written['offset']==begin and written['bytes']==end-begin and written['payload_bytes']+written['padding_bytes']==len(raw),'Saved payload metadata differs')
    if written['padding_bytes']:need(not any(raw[written['payload_bytes']:]),'Saved nonzero fixed-slot padding')
  need(len(p['weights'])==24 and [v['name'] for v in p['weights']]==[v['name'] for v in plan['matrices']],'Saved actual24matrix packing membership differs')
  for weight in p['weights']:
   name,bits=weight['name'],weight['bits'];n,k=weight['n'],weight['k'];b=plan['buffers'][name+'_weights'];f.seek(b['offset']);slot=f.read(b['bytes']);packed=slot[:weight['packed_bytes']];codes=unpack_codes(packed,bits);values=[v if v<128 else v-256 for v in codes]
   need(bits==p['precision_map'][weight['group']] and weight['slot_bytes']==b['bytes']==n*k and weight['packed_bytes']==n*k*bits//8 and len(codes)==n*k and pack_codes(codes,bits)==packed and not any(slot[weight['packed_bytes']:]) and min(values)==weight['code_min'] and max(values)==weight['code_max'],'Actual saved precision/packed-range/padding readback differs')
   for width in (4,8):
    b=plan['buffers'][name+f'_weight_scale{width}'];f.seek(b['offset']);scales=struct.unpack('<'+str(n)+'f',f.read(b['bytes']));need(all(math.isfinite(v) and v>0 for v in scales),'Saved genuine frozen row scales invalid')
  for name,b in plan['buffers'].items():
   if b['dtype']=='f32' and name in p['written_buffers']:
    f.seek(b['offset'])
    for start in range(0,b['bytes'],65536):
     chunk=f.read(min(65536,b['bytes']-start));need(len(chunk)%4==0 and all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',chunk)),'Saved nonfinite model parameter: '+name)
  for field,name,kind in [('input_ids','token_ids','I'),('token_type_ids','type_ids','I'),('attention_mask','attention_mask','B')]:
   b=plan['buffers'][name];f.seek(b['offset']);need(f.read(b['bytes'])==struct.pack('<'+kind*p['length'],*p['window']['inputs'][field]),'Saved frozen IDs/mask differ');need(d[{'token_ids':'token_offset','type_ids':'type_offset','attention_mask':'mask_offset'}[name]]==b['offset'],'Job input address differs')
 return p

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--mode',choices=('bringup-w8','finalist'),default='bringup-w8');ap.add_argument('--role',choices=('U','B','C'));ap.add_argument('--length',type=int,choices=(128,256,512),required=True);ap.add_argument('--window-key',required=True);ap.add_argument('--template-job',required=True);ap.add_argument('--model-id',type=lambda v:int(v,0),required=True);ap.add_argument('--threshold-record',required=True);ap.add_argument('--purpose',choices=('bringup','development_agreement','final_evaluation'),required=True);ap.add_argument('--allow-threshold-length-transfer',action='store_true');ap.add_argument('--output',required=True);ap.add_argument('--execute',action='store_true');ap.add_argument('--validate-only',action='store_true');a=ap.parse_args()
 if a.validate_only:print(json.dumps(dict(validated=True,length=validate_saved(a.output)['length'],numerical_acceptance=None)));return
 try:r=prepare_request(a)
 except (ValueError,FileNotFoundError,KeyError) as error:
  if a.execute:raise
  print(json.dumps(dict(execute=False,ready=False,blockers=[str(error)],torch_imported='torch' in sys.modules,numerical_acceptance=None),indent=2));return
 if not a.execute:print(json.dumps(dict(execute=False,ready=True,output=rel(r['out']),mode=a.mode,role=a.role,length=a.length,window=a.window_key,valid_length=r['valid'],last_valid_position=r['valid']-1,checkpoint=rel(r['checkpoint']),precision_map=r['precision'],deployment=r['deployment'],threshold_scope=r['threshold_association']['scope'],threshold_identity=r['threshold_association']['threshold_identity'],threshold_source=rel(r['threshold_source']),threshold_role=r['threshold_association']['threshold_role'],purpose=a.purpose,threshold_transfer=r['threshold_association']['threshold_transfer'],threshold_source_length=r['threshold_association']['source_length'],deployed_length=a.length,attack_rule='score > threshold',plan_identity=r['plan']['identity'],integer_calls=r['plan']['counts']['integer_calls'],fp32_calls=r['plan']['counts']['fp32_calls'],torch_imported='torch' in sys.modules,model_forward=False,physical_allocation_confirmed=False,numerical_acceptance=None,execution_prerequisite='Explicit root resource release for checkpoint-only Torch read/packing; no inference/compiler/vendor activity'),indent=2));return
 execute(a,r)
if __name__=='__main__':main()