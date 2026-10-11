"""Frozen Goal 7 software timing; default is a source-only prerequisite preview.

The backend descriptor separates frozen_model_identity provenance from the
actual native_scoring_identity and observed backend_identity, and binds backend/format,
checkpoint and precision map, CPU thread settings,
and backend-specific development_scores_dir plus same-backend
floating_reference_scores_dir. development_evidence must be that candidate.json;
its real per-document rows are membership-checked and rescored before timing.
Timing reports strict-decision raw-text full-document checks separately from
pre-padded, device-resident single-window checks. No output is a measurement
until --execute passes all frozen, quality and workload prerequisites.
"""
import argparse,csv,json,math,statistics,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
LENGTHS=(128,256,512);BATCHES=(1,8,32);ROLES=("floating","U","B","C");OUT=ROOT/"results/goal7/software_measurements"
NATIVE_SCORING_IDENTITY="transformers_native_logits_fp32_softmax_class1_max_window_v1"
FROZEN_IDENTITY_SCOPE="checkpoint/finalist provenance association; native scores use the separately recorded native computation"
NATIVE_QAT_LOAD_IDENTITY="qat_trained_fp32_parameters_native_hf_no_static_fakequant_v1"
QUANT_BUFFER_NAMES=("weight_scale_4","weight_scale_8","input_scale","zero_point","activation_width","weight_width")
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def rp(v):
 p=Path(str(v).replace(chr(92),"/"));return p if p.is_absolute() else ROOT/p
def finite(x):return isinstance(x,(int,float)) and math.isfinite(x)
def native_checkpoint_spec(role,entry):
 """Stdlib-only frozen provenance checks before importing Torch or loading a model."""
 identity=entry["identity"];cp=rp(entry["checkpoint"])
 if not cp.is_dir():raise ValueError("Frozen checkpoint missing: "+str(cp))
 if role=="floating":
  if entry.get("precision_map") is not None or rp(identity["checkpoint"]).resolve()!=cp.resolve():raise ValueError("Frozen floating checkpoint/map association differs.")
  return {"role":role,"checkpoint":str(cp),"tokenizer_checkpoint":str(cp)}
 if role not in ("U","B","C"):raise ValueError("Only the actual frozen floating/U/B/C roles are supported.")
 groups=[f"layer{i}.{g}" for i in range(4) for g in ("qkv","attention_output","ffn_input","ffn_output")];precision=entry["precision_map"]
 if not isinstance(precision,dict) or set(precision)!=set(groups) or any(type(precision[g]) is not int or precision[g] not in (4,8) for g in groups):raise ValueError("Frozen QAT map must contain exactly16 integer W4/W8 group choices.")
 if identity.get("qat") is not True or identity.get("precision_map")!=precision or rp(identity["scored_checkpoint"]).resolve()!=cp.resolve():raise ValueError("Frozen QAT identity/checkpoint/map association differs.")
 metadata=read(cp/"quantization.json");run=read(ROOT/"results/goal3/run.json");calibration=read(rp(identity["calibration_file"]))
 if metadata.get("scheme_id")!=identity.get("scheme_id") or metadata.get("scheme_id")!="static_symmetric_encoder_linear_w4w8a8_fp32_other_v1" or metadata.get("precision_map")!=precision:raise ValueError("Saved QAT scheme/map differs from the frozen finalist.")
 if metadata.get("calibration")!=calibration or rp(identity["calibration_file"]).resolve()!=rp(run["calibration_file"]).resolve():raise ValueError("Original frozen training calibration differs.")
 if any(metadata.get("format",{}).get(k)!=v for k,v in identity["frozen_scale_policy"].items()):raise ValueError("Saved static scale provenance differs from the frozen identity.")
 tokenizer=rp(identity["checkpoint"])
 if not tokenizer.is_dir() or tokenizer.resolve()!=rp(run["selected_checkpoint"]).resolve() or rp(metadata["training"]["from_floating_checkpoint"]).resolve()!=tokenizer.resolve():raise ValueError("QAT trained/tokenizer floating-checkpoint provenance differs.")
 if not(cp/"quantized_state.pt").is_file() or not(cp/"config.json").is_file():raise ValueError("Actual saved QAT state/config missing.")
 return {"role":role,"checkpoint":str(cp),"tokenizer_checkpoint":str(tokenizer),"precision_map":precision,"quantization_metadata":metadata,"conversion_identity":NATIVE_QAT_LOAD_IDENTITY}

def validate_native_qat_report(report,spec):
 """Require the completed loader evidence before consuming QAT backend scores."""
 if not isinstance(report,dict):raise ValueError("Actual native QAT conversion report missing.")
 if report.get('conversion_identity')!=NATIVE_QAT_LOAD_IDENTITY or report.get('finalist_role')!=spec['role'] or report.get('checkpoint')!=spec['checkpoint'] or report.get('precision_map')!=spec['precision_map'] or report.get('source_quantization_metadata')!=spec['quantization_metadata']:raise ValueError("Native QAT conversion provenance differs from the frozen checkpoint/map.")
 if report.get('all_original_trained_parameters_byte_value_exact') is not True or report.get('static_fake_quantization_used') is not False or report.get('native_int4_kernel_claimed') is not False:raise ValueError("Actual trained-parameter verification and native-format declaration required.")
 rows=report.get('trained_state_inventory',[]);count=report.get('trained_parameters_checked');omitted={f"bert.encoder.layer.{i}.{site}.{name}" for i in range(4) for site in ('attention.self.query','attention.self.key','attention.self.value','attention.output.dense','intermediate.dense','output.dense') for name in QUANT_BUFFER_NAMES}
 if type(count) is not int or count<1 or len({v['name'] for v in rows})!=len(rows) or sum(v.get('trained_parameter') is True for v in rows)!=count or any(v.get('dtype')!='torch.float32' for v in rows if v.get('trained_parameter')) or report.get('omitted_static_quantizer_buffers')!=sorted(omitted):raise ValueError("Full trained-state inventory or exactly144 omitted quantizer buffers missing.")
 return report

def native_qat_model(torch,spec):
 """Strict trained-state conversion, before any native precision optimization."""
 from transformers import AutoConfig,AutoModelForSequenceClassification
 cp=Path(spec["checkpoint"]);model=AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(cp,local_files_only=True)).float()
 state=torch.load(cp/"quantized_state.pt",map_location="cpu",weights_only=True)
 if not isinstance(state,dict):raise ValueError("Saved QAT state must be a named tensor dictionary.")
 modules={n:m for n,m in model.named_modules() if n.startswith("bert.encoder.layer.") and isinstance(m,torch.nn.Linear)}
 sites={f"bert.encoder.layer.{i}.{tail}" for i in range(4) for tail in ("attention.self.query","attention.self.key","attention.self.value","attention.output.dense","intermediate.dense","output.dense")}
 if set(modules)!=sites:raise ValueError("Native QAT target must contain exactly the24 original HF encoder linears.")
 expected=model.state_dict();parameters=dict(model.named_parameters());omitted={site+"."+name for site in sites for name in QUANT_BUFFER_NAMES}
 if set(state)!=set(expected)|omitted or set(expected)&omitted or not set(parameters)<=set(expected):raise ValueError("QAT state inventory differs: only the enumerated144 quantizer buffers may be omitted.")
 trained={}
 for name,target in expected.items():
  value=state[name]
  if not isinstance(value,torch.Tensor) or value.dtype!=target.dtype or value.shape!=target.shape or (name in parameters and value.dtype!=torch.float32) or not bool(torch.isfinite(value).all()):raise ValueError("Original trained/native state dtype,shape or finite-value mismatch: "+name)
  trained[name]=value
 for site,module in modules.items():
  layer=int(site.split('.')[3]);tail='.'.join(site.split('.')[4:]);group=f"layer{layer}."+("qkv" if tail.startswith('attention.self.') else "attention_output" if tail=='attention.output.dense' else "ffn_input" if tail=='intermediate.dense' else "ffn_output")
  for name in QUANT_BUFFER_NAMES:
   value=state[site+'.'+name];shape=(module.out_features,1) if name.startswith('weight_scale_') else ()
   if not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or tuple(value.shape)!=shape or not bool(torch.isfinite(value).all()):raise ValueError("Original quantizer buffer dtype/shape differs: "+site+'.'+name)
   if name in ('weight_scale_4','weight_scale_8','input_scale') and not bool((value>0).all()):raise ValueError("Saved frozen scale must remain positive.")
  if float(state[site+'.weight_width'])!=spec['precision_map'][group] or float(state[site+'.activation_width'])!=8 or float(state[site+'.zero_point'])!=0:raise ValueError("Saved per-site QAT width/zero-point differs from exact frozen group map.")
  if float(state[site+'.input_scale'])!=float(torch.tensor(spec['quantization_metadata']['calibration']['modules'][site]['input_scale'],dtype=torch.float32)):raise ValueError("Saved A8 scale differs from frozen training calibration.")
 model.load_state_dict(trained,strict=True)
 verified=[]
 for name,value in model.state_dict().items():
  original=trained[name]
  if value.dtype!=original.dtype or value.shape!=original.shape or not torch.equal(value,original) or value.detach().cpu().contiguous().numpy().tobytes()!=original.detach().cpu().contiguous().numpy().tobytes():raise ValueError("Native loaded state is not byte/value exact: "+name)
  verified.append(dict(name=name,shape=list(value.shape),dtype=str(value.dtype),bytes=value.numel()*value.element_size(),trained_parameter=name in parameters))
 report={"conversion_identity":NATIVE_QAT_LOAD_IDENTITY,"finalist_role":spec['role'],"checkpoint":spec['checkpoint'],"precision_map":spec['precision_map'],"source_quantization_metadata":spec['quantization_metadata'],"all_original_trained_parameters_byte_value_exact":True,"trained_parameters_checked":len(parameters),"trained_state_inventory":verified,"omitted_static_quantizer_buffers":sorted(omitted),"static_fake_quantization_used":False,"native_int4_kernel_claimed":False,"map_usage":"Frozen QAT training/checkpoint provenance; native backend ignores static W4/W8/A8 fake quantization. Exact trained FP32 parameters verified BEFORE native format conversion."}
 validate_native_qat_report(report,spec)
 return model.eval(),report

def backend_runtime_identity(torch,model,dev,fmt,threads,engine=None):
 from contextlib import nullcontext
 import platform,transformers
 context=precision_context(torch,dev,fmt)
 with context:
  autocast_enabled=torch.is_autocast_enabled(dev.type)
  try:autocast_dtype=str(torch.get_autocast_dtype(dev.type))
  except (AttributeError,RuntimeError):autocast_dtype=None
 device={"backend":dev.type,"torch_version":str(torch.__version__),"torch_git_version":getattr(torch.version,"git_version",None),
  "cuda_runtime":torch.version.cuda,"transformers_version":transformers.__version__,"python_version":platform.python_version(),"device_name":torch.cuda.get_device_name(dev) if dev.type=="cuda" else platform.processor(),
  "cuda_device_index":torch.cuda.current_device() if dev.type=="cuda" else None,"cuda_device_capability":list(torch.cuda.get_device_capability(dev)) if dev.type=="cuda" else None,"platform":platform.platform(),"intraop_threads":torch.get_num_threads(),"inter_op_threads":torch.get_num_interop_threads(),
  "float32_matmul_precision":torch.get_float32_matmul_precision(),"cuda_matmul_allow_tf32":bool(torch.backends.cuda.matmul.allow_tf32),
  "cudnn_allow_tf32":bool(torch.backends.cudnn.allow_tf32),"cudnn_deterministic":bool(torch.backends.cudnn.deterministic),"cudnn_benchmark":bool(torch.backends.cudnn.benchmark),"attention_implementation":getattr(model.config,"_attn_implementation",None),
  "quantized_engine":torch.backends.quantized.engine if dev.type=="cpu" else None}
 operation={"format":fmt,"autocast_enabled":bool(autocast_enabled),"autocast_dtype":autocast_dtype,
  "native_scoring_identity":NATIVE_SCORING_IDENTITY,
  "model_class":type(model).__module__+"."+type(model).__qualname__,
  "dynamic_qint8_modules":sorted(n for n,m in model.named_modules() if isinstance(m,torch.ao.nn.quantized.dynamic.Linear))}
 if hasattr(model,"_sentry_native_qat_conversion"):operation["native_qat_conversion"]=model._sentry_native_qat_conversion
 return {"device_identity":device,"backend_identity":dict(device,**operation)}
def save_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp")
 tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");tmp.replace(path)
def frozen_gate(fp):
 from gate_eval import load_config
 from evaluate_frozen_finalists import prerequisites
 c=load_config();pr,f=prerequisites(c,str(fp));return c,pr,f
def gate(dp,fp):
 c,pr,f=frozen_gate(fp);e=[]
 try:d=read(dp)
 except (OSError,ValueError,TypeError) as x:d={};e.append("Backend descriptor unreadable: "+str(x))
 if dp is None:e.append("Pass --backend-descriptor from a completed qualify-backend run.")
 if d.get("state")!="qualified_backend_development":e.append("Backend descriptor state must be qualified_backend_development.")
 be,fmt,role=d.get("backend"),d.get("format"),d.get("finalist_role")
 if be not in ("cpu","cuda"):e.append("Backend must be cpu or cuda.")
 if fmt not in ("float32","dynamic_qint8","float16","bfloat16"):e.append("Native format unsupported by this driver.")
 if be=="cuda" and fmt=="dynamic_qint8":e.append("dynamic_qint8 is CPU-only.")
 if type(d.get("intraop_threads")) is not int or d.get("intraop_threads",0)<1:e.append("Qualified descriptor needs positive intraop_threads.")
 if type(d.get("inter_op_threads")) is not int or d.get("inter_op_threads",0)<1:e.append("Qualified descriptor needs positive inter_op_threads.")
 fs=f.get("finalists",{}) if isinstance(f,dict) else {}
 if role not in ROLES or role not in fs:e.append("Select an actual frozen floating/U/B/C finalist.")
 else:
  try:
   spec=native_checkpoint_spec(role,fs[role])
   if role!="floating":
    backend=d.get("backend_identity")
    if not isinstance(backend,dict):raise ValueError("Native QAT backend identity must contain actual conversion evidence.")
    validate_native_qat_report(backend.get("native_qat_conversion",{}),spec)
  except (OSError,ValueError,KeyError,TypeError) as x:e.append("Native frozen checkpoint prerequisites: "+str(x))
  for descriptor_key,finalist_key in (("frozen_model_identity","identity"),("checkpoint","checkpoint"),("precision_map","precision_map")):
   if d.get(descriptor_key)!=fs[role].get(finalist_key):e.append("Descriptor "+descriptor_key+" differs from frozen finalist field "+finalist_key+".")
 if d.get("backend") not in ("cpu","cuda") or not d.get("backend_identity"):e.append("Backend identity is absent.")
 if d.get("frozen_model_identity_scope")!=FROZEN_IDENTITY_SCOPE:e.append("Descriptor does not limit frozen identity to checkpoint/finalist provenance.")
 if d.get("native_scoring_identity")!=NATIVE_SCORING_IDENTITY:e.append("Descriptor does not identify the actual native HF logits/softmax/max-window computation.")
 derived=None
 try:
  from search_precision import recompute_metrics
  if not d.get("development_scores_dir") or not d.get("floating_reference_scores_dir"):raise ValueError("Both actual backend score directories are required.")
  folder=rp(d["development_scores_dir"]);ref_folder=rp(d["floating_reference_scores_dir"])
  cand=read(folder/"candidate.json");ref=read(ref_folder/"candidate.json")
  if rp(d.get("development_evidence",""))!=(folder/"candidate.json"):raise ValueError("development_evidence must point to the actual candidate.json score record.")
  if cand.get("frozen_model_identity")!=d.get("frozen_model_identity") or cand.get("backend_identity")!=d.get("backend_identity") or cand.get("backend")!=be or cand.get("format")!=fmt or cand.get("qualification_run")!=d.get("qualification_run"):raise ValueError("Backend candidate record is not bound to this exact frozen identity/backend/format/run.")
  if ref.get("frozen_model_identity")!=fs.get("floating",{}).get("identity") or ref.get("backend_identity")!=d.get("floating_reference_backend_identity") or ref.get("backend")!=be or ref.get("format")!="float32":raise ValueError("Matching same-backend floating reference score record is required.")
  if cand.get("device_identity")!=d.get("device_identity") or ref.get("device_identity")!=d.get("device_identity"):raise ValueError("Backend and FP32 reference must share the same observed host/runtime device identity.")
  for rec,score_folder in ((cand,folder),(ref,ref_folder)):
   run=read(rec["qualification_run"])
   if run.get("state")!="completed_backend_qualification" or run.get("native_scoring_identity")!=NATIVE_SCORING_IDENTITY or rec.get("native_scoring_identity")!=NATIVE_SCORING_IDENTITY or rec.get("frozen_model_identity_scope")!=FROZEN_IDENTITY_SCOPE or run.get("frozen_model_identity_scope")!=FROZEN_IDENTITY_SCOPE or type(run.get("quality_passed")) is not bool or run.get("quality_passed")!=rec.get("quality_passed") or run.get("backend_identity")!=rec.get("backend_identity") or run.get("device_identity")!=rec.get("device_identity") or run.get("frozen_model_identity")!=rec.get("frozen_model_identity") or run.get("development_scores_dir")!=str(score_folder) or run.get("backend")!=rec.get("backend") or run.get("format")!=rec.get("format"):raise ValueError("Score metadata is not linked to its completed observed-runtime qualification run.")
  if cand.get("quality_passed") is not True or d.get("qualification_run")!=cand.get("qualification_run"):raise ValueError("Measured candidate must pass actual development protection; the same-backend FP32 reference supplies recall, with completed score/run links required.")
  _,actual,_=validate_native_scores(folder,d.get("frozen_model_identity"),c,be,fmt,d.get("device_identity"),str(fp))
  _,baseline,_=validate_native_scores(ref_folder,fs["floating"]["identity"],c,be,"float32",d.get("device_identity"),str(fp),require_eligible=False)
  for record,metrics in ((cand,actual),(ref,baseline)):
   saved=record.get("metrics",{})
   if saved.get("primary",{}).get("threshold")!=metrics["temporary_threshold"] or saved.get("primary",{}).get("candidate_scoring")!=metrics["protection"] or saved.get("ordinary_threshold_0_5")!=metrics["ordinary_0_5"]:raise ValueError("Saved backend quality summary differs from recomputed assigned document rows.")
  prot=actual["protection"];refrec=baseline["protection"]["recall"]
  derived=dict(recall=prot["recall"],floating_reference_recall=refrec,fpr=prot["fpr"],threshold=actual["temporary_threshold"],
               candidate_documents=prot["documents"],threshold_benign_documents=actual["threshold_benign_count"])
  if d.get("decision_threshold")!=derived["threshold"]:raise ValueError("Descriptor threshold differs from recomputed backend threshold.")
  if prot["fpr"]>c["evaluation"]["fpr_target"] or prot["recall"]<refrec-c["evaluation"]["max_recall_loss_absolute"]:raise ValueError("Backend-specific recomputed development protection fails.")
 except (OSError,ValueError,KeyError,TypeError,RuntimeError) as x:e.append("Backend development quality not proven: "+str(x))
 blockers=list(dict.fromkeys(pr["blockers"]+e))
 return c,d,{"ready":not blockers,"blockers":blockers,"freeze_prerequisites":pr,"backend_development_protection":derived,
   "heldout_content_read":False,"model_loaded":False,"benchmark_started":False}
def sync(torch,dev):
 if dev.type=="cuda":torch.cuda.synchronize(dev)
def load(d,dev):
 import torch
 from transformers import AutoModelForSequenceClassification,AutoTokenizer
 if dev.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable.")
 if d["format"] in ("float16","bfloat16") and dev.type!="cuda":raise RuntimeError(d["format"]+" timing requires CUDA; CPU autocast is not part of this driver.")
 if d["format"]=="bfloat16" and not torch.cuda.is_bf16_supported():raise RuntimeError("CUDA device does not support bfloat16 autocast.")
 spec=native_checkpoint_spec(d["finalist_role"],dict(identity=d["frozen_model_identity"],checkpoint=d["checkpoint"],precision_map=d["precision_map"]))
 tok=AutoTokenizer.from_pretrained(spec["tokenizer_checkpoint"],local_files_only=True,use_fast=True);conversion=None
 if d["finalist_role"]=="floating":m=AutoModelForSequenceClassification.from_pretrained(spec["checkpoint"],local_files_only=True).float().eval()
 else:m,conversion=native_qat_model(torch,spec)
 if d["format"]=="dynamic_qint8":
  if dev.type!="cpu":raise RuntimeError("dynamic_qint8 is CPU-only.")
  names=[n for n,x in m.named_modules() if n.startswith("bert.encoder.layer.") and isinstance(x,torch.nn.Linear)]
  if len(names)!=24:raise RuntimeError("Expected 24 BERT-Mini encoder linears.")
  eng=d.get("quantized_engine")
  if eng not in torch.backends.quantized.supported_engines:raise RuntimeError("Quantized engine unavailable.")
  torch.backends.quantized.engine=eng;m=torch.ao.quantization.quantize_dynamic(m,qconfig_spec={n:torch.ao.quantization.default_dynamic_qconfig for n in names},dtype=torch.qint8,inplace=False).eval()
  if {n for n,x in m.named_modules() if isinstance(x,torch.ao.nn.quantized.dynamic.Linear)}!=set(names):raise RuntimeError("Quantized module scope differs.")
 if conversion is not None:m._sentry_native_qat_conversion=conversion
 return m.to(dev).eval(),tok
def precision_context(torch,dev,fmt):
 from contextlib import nullcontext
 if fmt=="float16":return torch.autocast(device_type=dev.type,dtype=torch.float16)
 if fmt=="bfloat16":return torch.autocast(device_type=dev.type,dtype=torch.bfloat16)
 return nullcontext()
def workload(tok,host_path,docs,cfg,L):
 from gate_eval import window_document
 host=read(host_path)
 if host.get("passed") is not True or host.get("documents")!=32 or host.get("data")!="data/prepared/search_candidate_scoring.jsonl":raise ValueError("Need the actual 32-document host-preparation record.")
 expected=[]
 for label in (0,1):expected += [x for x in docs if x["label"]==label][:16]
 records=host.get("records",[])
 if len(expected)!=32 or [x["document_id"] for x in records]!=[x["document_id"] for x in expected] or [x["label"] for x in records]!=[x["label"] for x in expected]:raise ValueError("Host workload IDs/order differ from the board-preparation source.")
 selected=[];windows=[]
 local=json.loads(json.dumps(cfg));local["windowing"]["primary_length"]=L
 for doc in expected:
  ww=window_document(tok,doc["text"],doc["document_id"],doc["label"],doc.get("injection_spans"),local)
  if len(ww)==1:selected.append(doc);windows.append((doc,ww[0]))
 if not selected:raise ValueError("Board-aligned workload has no single-window documents at requested length.")
 return selected,windows
def device_tensors(torch,tok,windows,L,dev):
 batch=tok.pad([{k:w[k] for k in ("input_ids","attention_mask","token_type_ids")} for _,w in windows],
   padding="max_length",max_length=L,return_tensors="pt")
 return {k:v.to(dev) for k,v in batch.items()}
def infer_prepared(torch,model,tensors,indices,dev,fmt):
 batch={k:v[indices] for k,v in tensors.items()}
 with torch.inference_mode(),precision_context(torch,dev,fmt):
  logits=model(**batch).logits.float()
  return torch.softmax(logits,dim=-1)[:,1]
def stats(x):
 x=sorted(x);return {"p50_ms":x[int((len(x)-1)*.5)],"p95_ms":x[int((len(x)-1)*.95)],"p99_ms":x[int((len(x)-1)*.99)],"mean_ms":statistics.fmean(x)}
def primary(model,tok,docs,cfg,dev,fmt,threshold,warm,host_path):
 import torch
 from gate_eval import score_documents
 selected,windows=workload(tok,host_path,docs,cfg,256);tensors=device_tensors(torch,tok,windows,256,dev)
 local=json.loads(json.dumps(cfg));local["windowing"]["primary_length"]=256;out=[]
 for rep in range(3):
  for i in range(warm):infer_prepared(torch,model,tensors,[i%len(selected)],dev,fmt)
  sync(torch,dev);pre=[];raw=[];pre_batches=[];raw_batches=[]
  for i in range(10000):
   idx=i%len(selected);sync(torch,dev);q=time.perf_counter();score=infer_prepared(torch,model,tensors,[idx],dev,fmt);decision=bool((score>threshold).detach().cpu()[0]);sync(torch,dev)
   dt=(time.perf_counter()-q)*1000;pre.append(dt);pre_batches.append(dt)
   doc=dict(selected[idx]);doc["document_id"]=f"timing-{rep}-{i}"
   q=time.perf_counter()
   with torch.inference_mode(),precision_context(torch,dev,fmt):
    scored=score_documents(model,tok,[doc],batch_size=1,device=dev,config=local)
    decision=scored["documents"][0]["risk_score"]>threshold
   sync(torch,dev);dt=(time.perf_counter()-q)*1000;raw.append(dt);raw_batches.append(dt)
  out.append({"repeat":rep+1,"completed_checks":10000,"documents":10000,"pretokenized_unit":"one already padded device-resident window through strict risk-score decision","raw_text_unit":"one full single-window source document through tokenization/windowing to strict decision",
   "pretokenized_latency_ms":stats(pre),"raw_text_to_decision_latency_ms":stats(raw),
   "pretokenized_throughput_checks_per_second":10000/(sum(pre_batches)/1000),
   "raw_text_throughput_checks_per_second":10000/(sum(raw_batches)/1000),
   "source_document_ids_in_cycle":[d["document_id"] for d in selected]})
 return out
def secondary(model,tok,docs,cfg,dev,fmt,threshold,count,warm,host_path):
 import torch
 from gate_eval import score_documents
 out=[]
 for L in LENGTHS:
  selected,windows=workload(tok,host_path,docs,cfg,L);tensors=device_tensors(torch,tok,windows,L,dev)
  local=json.loads(json.dumps(cfg));local["windowing"]["primary_length"]=L
  for b in (1,8,32):
   if (L,b)==(256,1):continue
   n=((count+b-1)//b)*b;nb=n//b;wc=max(b,warm//b*b);runs=[]
   for rep in range(3):
    for i in range(wc//b):infer_prepared(torch,model,tensors,[(i*b+j)%len(selected) for j in range(b)],dev,fmt)
    sync(torch,dev);pre_lat=[];raw_lat=[];pre_batches=[];raw_batches=[]
    for i in range(nb):
     ix=[(i*b+j)%len(selected) for j in range(b)]
     sync(torch,dev);q=time.perf_counter();scores=infer_prepared(torch,model,tensors,ix,dev,fmt);decisions_pre=(scores>threshold).detach().cpu().tolist();sync(torch,dev)
     dt=(time.perf_counter()-q)*1000;pre_batches.append(dt);pre_lat.extend([dt]*b)
     ds=[]
     for j,index in enumerate(ix):
      doc=dict(selected[index]);doc["document_id"]=f"sec-{L}-{b}-{rep}-{i}-{j}";ds.append(doc)
     q=time.perf_counter()
     with torch.inference_mode(),precision_context(torch,dev,fmt):
      scored=score_documents(model,tok,ds,batch_size=b,device=dev,config=local)
      decisions=[x["risk_score"]>threshold for x in scored["documents"]]
     sync(torch,dev);dt=(time.perf_counter()-q)*1000;raw_batches.append(dt);raw_lat.extend([dt]*b)
    runs.append({"repeat":rep+1,"length":L,"batch_size":b,"completed_checks":n,"document_checks":n,"batches":nb,"warmup_checks":wc,
      "pretokenized_unit":"already padded device-resident windows","raw_text_unit":"full source documents with window aggregation and strict decision",
      "pretokenized":{"per_check_decision_latency_ms":stats(pre_lat),"throughput_checks_per_second":n/(sum(pre_batches)/1000)},
      "raw_text_to_decision":{"per_check_decision_latency_ms":stats(raw_lat),"throughput_checks_per_second":n/(sum(raw_batches)/1000)},
      "batch_latency_rule":"Each of B checks receives measured batch completion time; throughput divides checks by elapsed batch-service time, never latency by B.",
      "source_document_ids_in_cycle":[d["document_id"] for d in selected]})
   out.append({"length":L,"batch_size":b,"completed_checks_per_repeat":n,"repeats":3,"completed_checks":sum(x["completed_checks"] for x in runs),"repeat_summaries":runs,
    "final_timing_cell_qualified":n>=10000 and len(runs)==3 and all(x["completed_checks"]>=10000 for x in runs)})
 return out
def energy(path,completed_checks=None,system_id="software"):
 if not path:return None
 groups={}
 with Path(path).open(newline="",encoding="utf-8-sig") as f:
  r=csv.DictReader(f);fields=set(r.fieldnames or [])
  generic={"system_id","scope","elapsed_seconds","power_w"}.issubset(fields)
  wide={"time_seconds","board_watts","whole_system_watts"}.issubset(fields)
  if not generic and not wide:raise ValueError("Energy CSV requires generic system/scope/time/power fields or time_seconds,board_watts,whole_system_watts.")
  for row in r:
   if generic:
    scope=row["scope"]
    if scope not in ("board_only","whole_system"):raise ValueError("Energy scope must be board_only or whole_system.")
    groups.setdefault((row["system_id"],scope),[])
    if row.get("power_w") not in ("",None):groups[(row["system_id"],scope)].append((float(row["elapsed_seconds"]),float(row["power_w"]),float(row["completed_checks"]) if row.get("completed_checks") else None))
   else:
    t=float(row["time_seconds"])
    for scope,key in (("board_only","board_watts"),("whole_system","whole_system_watts")):
     groups.setdefault((system_id,scope),[])
     if row.get(key) not in ("",None):groups[(system_id,scope)].append((t,float(row[key]),float(row["completed_checks"]) if row.get("completed_checks") else None))
 if groups:
  for sid,_ in list(groups):
   for scope in ("board_only","whole_system"):groups.setdefault((sid,scope),[])
 out=[]
 for (sid,scope),samples in groups.items():
  if len(samples)<2:
   out.append({"system_id":sid,"scope":scope,"measured_energy_j":None,"completed_checks":completed_checks,"joules_per_check":None,"source":str(path),"method":"measured power unavailable or fewer than two samples; no TDP substitution"});continue
  samples.sort(key=lambda x:x[0]);joules=0.
  for left,right in zip(samples,samples[1:]):
   dt=right[0]-left[0]
   if dt<=0 or not all(math.isfinite(x) and x>=0 for x in (left[1],right[1],dt)):raise ValueError("Energy times and power must be finite, nonnegative, and ordered.")
   joules+=(left[1]+right[1])*.5*dt
  counts=[x[2] for x in samples if x[2] is not None]
  count=(counts[-1]-counts[0]) if len(counts)==len(samples) else completed_checks
  if count is not None and (not isinstance(count,(int,float)) or count<=0):raise ValueError("Completed-check denominator must be a positive measured count.")
  out.append({"system_id":sid,"scope":scope,"measured_energy_j":joules,"completed_checks":count,"joules_per_check":joules/count if count else None,"source":str(path),"method":"trapezoidal integration of measured power; TDP never substituted"})
 return out
def replay_arrivals(model,tok,docs,cfg,dev,csv_path,batch_size,threshold,fmt):
 import torch
 from gate_eval import score_documents
 with Path(csv_path).open(newline="",encoding="utf-8-sig") as f:
  rows=[(int(row["check_index"]),float(row["arrival_seconds"])) for row in csv.DictReader(f)]
 if not rows or [i for i,_ in rows]!=list(range(len(rows))) or any(not math.isfinite(t) or t<0 for _,t in rows) or any(rows[i][1]>rows[i+1][1] for i in range(len(rows)-1)):
  raise ValueError("Arrival CSV needs contiguous checks and nondecreasing finite arrival_seconds.")
 if not finite(threshold) or not 0<=threshold<=1:raise ValueError("Frozen decision threshold missing/invalid.")
 c=json.loads(json.dumps(cfg));c["windowing"]["primary_length"]=256
 done=[];service=[];epoch=time.perf_counter();group_size=batch_size
 with torch.inference_mode(),precision_context(torch,dev,fmt):
  for group in range(0,len(rows),group_size):
   members=rows[group:group+group_size]
   target=epoch+members[-1][1];delay=target-time.perf_counter()
   if delay>0:time.sleep(delay)
   inputs=[]
   for i,_ in members:
    doc=docs[i%len(docs)];inputs.append({"document_id":"arrival-"+str(i),"text":doc["text"],"label":0,"injection_spans":[]})
   start=time.perf_counter();scored=score_documents(model,tok,inputs,batch_size=batch_size,device=dev,config=c);sync(torch,dev);finish=time.perf_counter()
   service.append((finish-start)*1000)
   for (i,arrival),result in zip(members,scored["documents"]):
    done.append({"check_index":i,"source_document_id":docs[i%len(docs)]["document_id"],"arrival_to_decision_ms":(finish-(epoch+arrival))*1000,"risk_score":result["risk_score"],"decision":result["risk_score"]>threshold})
 elapsed=time.perf_counter()-epoch
 return {"arrival_file":str(csv_path),"arrival_checks":len(rows),"completed_checks":len(done),"batch_size":batch_size,"batch_policy":"fixed contiguous groups; wait for each group's final scheduled arrival; sequential model service","throughput_checks_per_second":len(done)/elapsed if elapsed else None,"decision_latency_ms":stats([x["arrival_to_decision_ms"] for x in done]),"batch_service_ms":stats(service),"checks":done,"queue_and_batch_formation_included":True,"measurement_elapsed_seconds":elapsed}

def validate_capacities(path):
 data=read(path);devices=data.get("devices",[])
 if len(devices)<2:raise ValueError("Capacity evidence needs software and FPGA devices.")
 kinds=set();rates={};ids=set()
 for item in devices:
  if not isinstance(item.get("id"),str) or not item["id"] or item["id"] in ids:raise ValueError("Measured device IDs must be nonempty and unique.")
  ids.add(item["id"])
  kind=item.get("device_kind");kinds.add(kind)
  if kind not in ("software","fpga") or item.get("capacity_unit")!="full_document_checks_per_second" or item.get("boundary")!="raw_text_to_decision":raise ValueError("Capacity records must identify device_kind, full-document checks/s, and raw-text-to-decision boundary.")
  evp=rp(item.get("evidence",""))
  if not evp.is_file():raise ValueError("Capacity evidence source file is missing.")
  record=read(evp)
  if kind=="software":
   if record.get("state")!="measured" or record.get("stage")!="timing" or record.get("configuration")!={"length":256,"batch_size":1}:raise ValueError("Software capacity source must be this driver's completed L256/B1 timing record.")
   cap=record.get("capacity",{})
   if cap.get("capacity_unit")!="full_document_checks_per_second" or cap.get("boundary")!="raw_text_to_decision" or cap.get("completed_checks_per_repeat")!=10000 or cap.get("repeats")!=3:raise ValueError("Software source does not establish the full-document raw-text capacity scope and unit.")
   runs=record.get("primary_L256_B1",[])
   if len(runs)!=3 or any(x.get("completed_checks")!=10000 for x in runs):raise ValueError("Software capacity requires 10k completed raw-text document checks x3.")
   measured=statistics.median(x["raw_text_throughput_checks_per_second"] for x in runs)
   if not finite(cap.get("checks_per_second")) or abs(cap["checks_per_second"]-measured)>max(1e-9,measured*1e-6):raise ValueError("Software run capacity value differs from its three measured full-document repeats.")
  else:
   if record.get("passed") is not True or record.get("state")!="completed" or record.get("stage")!="timing" or record.get("physical_board_activity") is not True or record.get("length")!=256 or record.get("batch")!=1 or record.get("boundary")!="raw-text" or record.get("completed",0)<10000 or record.get("workload_documents",0)<1:raise ValueError("FPGA capacity source must be a completed actual L256/B1 raw-text full-document timing run.")
   repeats=record.get("repeat_summaries",[])
   if len(repeats)<3 or any(x.get("completed",0)<10000 for x in repeats[:3]):raise ValueError("FPGA capacity needs at least 10k completed document checks in each of three repeats.")
   measured=statistics.median(x["checks_per_second"] for x in repeats[:3])
  supplied=item.get("checks_per_second")
  if not finite(supplied) or supplied<=0 or abs(supplied-measured)>max(1e-9,measured*1e-6):raise ValueError("Capacity JSON value differs from the cited actual timing evidence.")
  rates[item["id"]]=measured
 if not {"software","fpga"}.issubset(kinds):raise ValueError("Need at least one software and one FPGA capacity.")
 return data,rates,min(rates.values())
def read_arrivals(paths,slow):
 if len(paths)!=3:raise ValueError("Load stage needs exactly three common 25/60/85% arrival schedules.")
 schedules={}
 for path in paths:
  p=rp(path);match=__import__("re").search(r"arrivals_(25|60|85)",p.stem)
  if not match:raise ValueError("Arrival filename must identify arrivals_25, arrivals_60, or arrivals_85.")
  percent=int(match.group(1))
  if percent in schedules:raise ValueError("Duplicate arrival fraction.")
  with p.open(newline="",encoding="utf-8-sig") as f:
   rows=[(int(x["check_index"]),float(x["arrival_seconds"])) for x in csv.DictReader(f)]
  if len(rows)!=10000 or [i for i,_ in rows]!=list(range(10000)):raise ValueError("Each common arrival schedule must contain exactly10,000 ordered checks.")
  rate=slow*(percent/100)
  if any(not math.isfinite(t) or abs(t-i/rate)>1e-8 for i,t in rows):raise ValueError("Arrival schedule differs from the measured slowest-system rate/fraction.")
  schedules[percent]=p
 if set(schedules)!={25,60,85}:raise ValueError("Need exactly25/60/85 percent common schedules.")
 return [schedules[x] for x in (25,60,85)]
def load_host_docs(host_path):
 host=read(host_path)
 if host.get("passed") is not True or host.get("length")!=256 or host.get("documents")!=32 or host.get("data")!="data/prepared/search_candidate_scoring.jsonl":raise ValueError("Actual L256/B1 32-document host-preparation evidence required.")
 records=host.get("records",[])
 if len(records)!=32:raise ValueError("Host-preparation record must preserve32 source documents.")
 all_docs=[];labels={0:[],1:[]};wanted={r["document_id"] for r in records}
 with (ROOT/"data/prepared/search_candidate_scoring.jsonl").open(encoding="utf-8-sig") as f:
  for line in f:
   d=json.loads(line)
   if len(labels[d["label"]])<16:labels[d["label"]].append(d["document_id"])
   if d["document_id"] in wanted:all_docs.append(d)
 expected=labels[0]+labels[1]
 if [x["document_id"] for x in records]!=expected:raise ValueError("Host-prepared source IDs/order differ from first16 benign then first16 attacked development documents.")
 by_id={x["document_id"]:x for x in all_docs}
 if set(by_id)!=wanted:raise ValueError("Host-prepared workload members missing from search candidate partition.")
 return [by_id[r["document_id"]] for r in records]
def validate_host_payload(tok,host_path,docs,cfg):
 import struct
 from gate_eval import window_document
 from prepare_host_workload import HEADER,packet
 host=read(host_path);binary=Path(host_path).with_name("workload.bin").read_bytes();length=host.get("length")
 if length!=256 or host.get("batch")!=1 or host.get("passed") is not True:raise ValueError("Only the actual passed L256/B1 host packet source is accepted.")
 if host.get("tokenizer_checkpoint")!="checkpoints/floating/epoch_3" or host.get("tokenizer_revision")!=cfg["models"]["detector"]["tokenizer_revision"]:raise ValueError("Host packet tokenizer source/revision differs from the frozen protocol.")
 expected=[]
 for label in (0,1):expected += [d for d in docs if d["label"]==label][:16]
 if [r.get("document_id") for r in host.get("records",[])]!=[d["document_id"] for d in expected]:raise ValueError("Host packet document IDs/order differ from the prepared source.")
 if len(binary)<HEADER.size:raise ValueError("Host workload binary is missing packet headers.")
 first=HEADER.unpack_from(binary,0)
 if first[0]!=b"SENTRY01" or first[1]!=1 or first[2]!=length:raise ValueError("Host workload header format differs.")
 model_id,threshold=first[7],first[10];offset=0;sample=0;counts=[]
 local=json.loads(json.dumps(cfg));local["windowing"]["primary_length"]=length
 for doc_index,doc in enumerate(expected,1):
  actual_count=host["records"][doc_index-1].get("windows")
  windows=window_document(tok,doc["text"],doc["document_id"],doc["label"],doc.get("injection_spans"),local)
  if len(windows)!=actual_count:raise ValueError("Host payload window count differs for "+doc["document_id"])
  for window_index,window in enumerate(windows):
   expected_packet=packet(window,length,doc_index,sample,window_index==len(windows)-1,model_id,threshold)
   if offset+len(expected_packet)>len(binary) or binary[offset:offset+len(expected_packet)]!=expected_packet:raise ValueError("Regenerated native token IDs/types/masks differ from workload.bin for "+doc["document_id"]+" window "+str(window_index))
   header=HEADER.unpack_from(binary,offset)
   if header[4]!=window_index or header[5]!=doc_index or header[6]!=sample or header[7]!=model_id or header[9]!=len(expected_packet)-HEADER.size or header[10]!=threshold:raise ValueError("Host payload packet sequence/association differs.")
   offset+=len(expected_packet);sample+=1
  counts.append(len(windows))
 if offset!=len(binary) or sample!=host.get("windows"):raise ValueError("Host payload has trailing/missing window packets.")
 return {"ready":True,"run_record":str(host_path),"payload":str(Path(host_path).with_name("workload.bin")),"document_ids":[d["document_id"] for d in expected],"windows":sample,"window_counts":counts,"token_ids_token_types_attention_masks_exact":True}
def existing_float32_candidates(backend,identity,threads,inter_op_threads,engine):
 root=ROOT/"results/goal7/backend_qualification";found=[]
 if not root.is_dir():return found
 for path in root.glob(f"{backend}_float32_*/candidate.json"):
  try:
   candidate=read(path);device=candidate.get("device_identity",{})
   if (candidate.get("generator")=="measure_frozen_software.py:qualify-backend" and candidate.get("frozen_model_identity")==identity and candidate.get("backend")==backend and candidate.get("format")=="float32" and candidate.get("state")=="completed_backend_qualification" and type(candidate.get("quality_passed")) is bool and device.get("intraop_threads")==threads and device.get("inter_op_threads")==inter_op_threads and device.get("quantized_engine")==engine):found.append(path.parent)
  except (OSError,ValueError,TypeError):continue
 return found

def native_candidate(folder):
 candidate=read(Path(folder)/"candidate.json")
 if candidate.get("generator")!="measure_frozen_software.py:qualify-backend":raise ValueError("Native FP32 reference must come from this actual qualification stage.")
 runpath=rp(candidate.get("qualification_run",""));run=read(runpath)
 if candidate.get("state")!="completed_backend_qualification" or run.get("state")!="completed_backend_qualification" or run.get("backend_identity")!=candidate.get("backend_identity") or run.get("device_identity")!=candidate.get("device_identity") or run.get("frozen_model_identity")!=candidate.get("frozen_model_identity") or run.get("native_scoring_identity")!=NATIVE_SCORING_IDENTITY or candidate.get("native_scoring_identity")!=NATIVE_SCORING_IDENTITY or run.get("frozen_model_identity_scope")!=FROZEN_IDENTITY_SCOPE or candidate.get("frozen_model_identity_scope")!=FROZEN_IDENTITY_SCOPE or run.get("development_scores_dir")!=str(Path(folder)):raise ValueError("Native reference is not associated with its completed observed-runtime run.")
 return candidate,run

def validate_window_rows(documents,document_rows,window_rows,index_offset=0):
 if len(documents)!=len(document_rows):raise ValueError("Window/document evidence length differs.")
 by_id={doc["document_id"]:doc for doc in documents};position=0
 for doc_index,(doc,scored) in enumerate(zip(documents,document_rows)):
  if doc["document_id"]!=scored.get("document_id"):raise ValueError("Window score document order differs.")
  expected=scored.get("window_scores",[]);actual=[]
  while position<len(window_rows) and window_rows[position].get("document_index")==doc_index+index_offset:
   row=window_rows[position];wi=len(actual)
   if row.get("document_id")!=doc["document_id"] or row.get("window_index")!=wi or any(row.get(field)!=doc.get(field) for field in ("source_group_id","source_ids","source_dataset","attack_type","attack_template_id","injection_position")):raise ValueError("Enriched window row identity/order differs from assigned development document.")
   actual.append(row.get("risk_score"));position+=1
  if actual!=expected or len(actual)!=scored.get("window_count"):raise ValueError("Per-document actual window risks/counts differ from the saved document aggregate.")
 if position!=len(window_rows):raise ValueError("Extra or misordered actual window score rows.")

def validate_native_scores(folder,identity,cfg,backend,fmt,device_identity,freeze_path,*,require_eligible=True):
 from search_precision import SPLITS,rows as score_rows,validate_document_rows,recompute_metrics
 candidate,run=native_candidate(folder)
 if identity.get("qat") is True:
  report=candidate.get("backend_identity",{}).get("native_qat_conversion",{});spec=native_checkpoint_spec(report.get("finalist_role"),dict(identity=identity,checkpoint=identity["scored_checkpoint"],precision_map=identity["precision_map"]));validate_native_qat_report(report,spec)
 if candidate.get("native_scoring_identity")!=NATIVE_SCORING_IDENTITY or candidate.get("frozen_model_identity_scope")!=FROZEN_IDENTITY_SCOPE or candidate.get("frozen_model_identity")!=identity or candidate.get("backend")!=backend or candidate.get("format")!=fmt or candidate.get("device_identity")!=device_identity or candidate.get("freeze")!=str(freeze_path) or type(candidate.get("quality_passed")) is not bool or candidate.get("quality_passed")!=run.get("quality_passed"):raise ValueError("Native score artifact source/freeze/backend/completed quality-record association differs.")
 if require_eligible and candidate["quality_passed"] is not True:raise ValueError("Native measurement candidate fails the actual development protection gate.")
 if not require_eligible and fmt!="float32":raise ValueError("Only an actual completed FP32 score record may supply reference recall without candidate eligibility.")
 actual={}
 for split in SPLITS:
  docs=score_rows(ROOT/"data/prepared"/(split+".jsonl"));scored=score_rows(Path(folder)/(split+"_documents.jsonl"));windows=score_rows(Path(folder)/(split+"_windows.jsonl"));validate_document_rows(scored,docs,split);validate_window_rows(docs,scored,windows);actual[split]=len(scored)
 if candidate.get("document_counts")!=actual or candidate.get("data")!=cfg["data"] or candidate.get("windowing")!=cfg["windowing"] or candidate.get("evaluation")!=cfg["evaluation"]:raise ValueError("Native score split counts or shared data/window/evaluation protocol association differs.")
 metrics=recompute_metrics(Path(folder),cfg);saved=candidate.get("metrics",{})
 if saved.get("primary",{}).get("threshold")!=metrics["temporary_threshold"] or saved.get("primary",{}).get("candidate_scoring")!=metrics["protection"] or saved.get("ordinary_threshold_0_5")!=metrics["ordinary_0_5"]:raise ValueError("Native candidate metrics differ from assigned document rows.")
 return candidate,metrics,actual

def qualify_backend(args,fp,c,pr,f):
 if not pr["ready"]:raise RuntimeError("Frozen method/finalist prerequisites: "+"; ".join(pr["blockers"]))
 if args.backend not in ("cpu","cuda"):raise ValueError("Choose the actual cpu or cuda backend.")
 if args.format not in ("float32","dynamic_qint8","float16","bfloat16"):raise ValueError("Unsupported native format.")
 if args.backend=="cuda" and args.format=="dynamic_qint8":raise ValueError("dynamic_qint8 is CPU-only.")
 if args.backend=="cpu" and args.format in ("float16","bfloat16"):raise ValueError("CPU autocast formats are unsupported by this driver.")
 if type(args.intraop_threads)is not int or args.intraop_threads<1 or type(args.inter_op_threads)is not int or args.inter_op_threads<1:raise ValueError("Positive actual thread counts are required.")
 role=args.finalist_role;entry=f["finalists"][role];identity=entry["identity"];checkpoint=entry["checkpoint"];native_checkpoint_spec(role,entry);reference_identity=f["finalists"]["floating"]["identity"]
 base={"state":"qualifying_backend","backend":args.backend,"format":args.format,"finalist_role":role,"frozen_model_identity":identity,"native_scoring_identity":NATIVE_SCORING_IDENTITY,"frozen_model_identity_scope":FROZEN_IDENTITY_SCOPE,"checkpoint":checkpoint,"precision_map":entry["precision_map"],"intraop_threads":args.intraop_threads,"inter_op_threads":args.inter_op_threads,"quantized_engine":args.quantized_engine}
 reference=None
 if role=="floating" and args.format=="float32":
  if args.floating_reference:
   ref_folder=rp(args.floating_reference);fc=read(ref_folder/"candidate.json");reference,_,_=validate_native_scores(ref_folder,identity,c,args.backend,"float32",fc.get("device_identity"),str(fp),require_eligible=False)
   descriptor_path=rp(reference["backend_descriptor"]) if reference.get("backend_descriptor") else None
   if reference["quality_passed"] and (descriptor_path is None or not descriptor_path.is_file()):
    descriptor={**base,"state":"qualified_backend_development","backend_identity":reference["backend_identity"],"device_identity":reference["device_identity"],"development_scores_dir":str(ref_folder),"development_evidence":str(ref_folder/"candidate.json"),"floating_reference_scores_dir":str(ref_folder),"floating_reference_backend_identity":reference["backend_identity"],"decision_threshold":reference["decision_threshold"],"qualification_run":reference["qualification_run"]}
    descriptor_path=rp(args.descriptor_output) if args.descriptor_output else ROOT/"results/goal7/backend_descriptors"/f"{args.backend}_float32_reused_{ref_folder.name}.json"
    if descriptor_path.exists():raise ValueError("Preserve existing backend descriptor; choose a new --descriptor-output.")
    save_json(descriptor_path,descriptor);reference["backend_descriptor"]=str(descriptor_path);save_json(ref_folder/"candidate.json",reference)
   print(json.dumps({"state":"reused_native_fp32_reference","directory":str(ref_folder),"backend_descriptor":str(descriptor_path) if descriptor_path else None,"frozen_model_identity":identity,"quality_passed":reference["quality_passed"],"model_loaded":False},indent=2));return None
  existing=existing_float32_candidates(args.backend,identity,args.intraop_threads,args.inter_op_threads,args.quantized_engine if args.backend=="cpu" else None)
  if existing:raise ValueError("A completed native FP32 baseline already exists; reuse it with --floating-reference "+str(existing[0])+" ; it remains valid as recall reference even if its candidate protection gate failed.")
 else:
  if not args.floating_reference:raise ValueError("QAT or optimized backend qualification requires a completed same-backend floating native FP32 reference directory.")
  ref_candidate=read(rp(args.floating_reference)/"candidate.json")
  reference,_,_=validate_native_scores(rp(args.floating_reference),reference_identity,c,args.backend,"float32",ref_candidate.get("device_identity"),str(fp),require_eligible=False)
 if args.resume_attempt:
  outdir=rp(args.resume_attempt)
  if not outdir.is_dir():raise ValueError("Resume attempt directory does not exist.")
 else:
  stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
  outdir=ROOT/"results/goal7/backend_qualification"/f"{args.backend}_{args.format}_{stamp}"
 if outdir.exists() and not args.resume_attempt:raise ValueError("Attempt path exists; preserve it and choose a fresh unique attempt.")
 if not args.execute:raise RuntimeError("Qualification preview must be handled before calling execution.")
 if (outdir/"candidate.json").exists():raise ValueError("This attempt is already complete; preserve it and create a new attempt.")
 outdir.mkdir(parents=True,exist_ok=True)
 import torch
 dev=torch.device(args.backend)
 if dev.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable for requested backend.")
 if args.format=="bfloat16" and not torch.cuda.is_bf16_supported():raise RuntimeError("CUDA device does not support bfloat16 autocast.")
 torch.set_num_threads(args.intraop_threads);torch.set_num_interop_threads(args.inter_op_threads)
 if dev.type=="cpu":
  if args.quantized_engine not in torch.backends.quantized.supported_engines:raise RuntimeError("Requested CPU quantized engine is unavailable.")
  torch.backends.quantized.engine=args.quantized_engine
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision("highest")
 descriptor=dict(base)
 model,tok=load(descriptor,dev);observed=backend_runtime_identity(torch,model,dev,args.format,args.intraop_threads,args.quantized_engine)
 if reference:
  if reference.get("device_identity")!=observed["device_identity"]:raise ValueError("FP32 reference host/runtime identity differs from this optimized backend run.")
 batch_size,_=__import__("score_qualified_development").scoring_batch_policy(c,floating=True)
 from score_qualified_development import paused
 from search_precision import SPLITS,rows as score_rows,validate_document_rows,recompute_metrics
 from gate_eval import score_documents
 from train_detector import enrich,summarize
 runpath=outdir/"run.json";progresspath=outdir/"progress.json"
 association={"frozen_model_identity":identity,"native_scoring_identity":NATIVE_SCORING_IDENTITY,"frozen_model_identity_scope":FROZEN_IDENTITY_SCOPE,"freeze":str(fp),"backend":args.backend,"format":args.format,"backend_identity":observed["backend_identity"],"device_identity":observed["device_identity"],"batch_size":batch_size,"thread_settings":{"intraop":args.intraop_threads,"inter_op":args.inter_op_threads}}
 if progresspath.exists():
  progress=read(progresspath)
  if progress.get("association")!=association:raise ValueError("Resume attempt runtime/freeze association differs; preserve evidence.")
 else:progress={"association":association,"splits":{},"scoring_seconds":0.0};save_json(progresspath,progress)
 run={"state":"running_backend_qualification","finalist_role":role,"frozen_model_identity":identity,"native_scoring_identity":NATIVE_SCORING_IDENTITY,"frozen_model_identity_scope":FROZEN_IDENTITY_SCOPE,"freeze":str(fp),"backend":args.backend,"format":args.format,"backend_identity":observed["backend_identity"],"device_identity":observed["device_identity"],"development_scores_dir":str(outdir),"device":str(dev),"device_name":torch.cuda.get_device_name(dev) if dev.type=="cuda" else __import__("platform").processor(),"torch_version":torch.__version__,"cuda_runtime":torch.version.cuda,"thread_settings":{"intraop":torch.get_num_threads(),"inter_op":torch.get_num_interop_threads()},"batch_size":batch_size,"heldout_content_read":False,"model_loaded":True,"benchmark_started":False,"recorded_utc":datetime.now(timezone.utc).isoformat()}
 save_json(runpath,run)
 for split in SPLITS:
  documents=score_rows(ROOT/"data/prepared"/(split+".jsonl"));validate_document_rows([],documents,split,complete=False)
  state=progress["splits"].setdefault(split,{"next_document":0,"chunks":[]});position=state["next_document"];scored=[];committed_position=0
  if type(position)is not int or not 0<=position<=len(documents):raise ValueError("Resumed position exceeds the prepared development partition.")
  for chunk_meta in state["chunks"]:
   chunk=read(outdir/chunk_meta["file"]);start=chunk_meta["start"];current=documents[start:start+chunk_meta["count"]]
   if start!=committed_position:raise ValueError("Resumed chunk order has a gap or overlap.")
   if chunk["frozen_model_identity"]!=identity or chunk.get("native_scoring_identity")!=NATIVE_SCORING_IDENTITY or chunk.get("frozen_model_identity_scope")!=FROZEN_IDENTITY_SCOPE or chunk["backend_identity"]!=observed["backend_identity"] or chunk.get("device_identity")!=observed["device_identity"] or chunk.get("split")!=split or chunk.get("start")!=start or chunk.get("count")!=len(current):raise ValueError("Resumed chunk runtime identity differs.")
   validate_document_rows(chunk["documents"],current,split);validate_window_rows(current,chunk["documents"],chunk["windows"],index_offset=start);scored.extend(chunk["documents"]);committed_position+=len(current)
  if len(scored)!=position or committed_position!=position:raise ValueError("Resumed chunk cursor differs from complete committed rows.")
  for start in range(position,len(documents),args.chunk_documents):
   paused()
   if (ROOT/"results/goal7/pause.request").exists():raise RuntimeError("Goal7 pause marker present; preserve qualification progress.")
   current=documents[start:start+args.chunk_documents];chunkpath=outdir/f"{split}_chunk_{start:06d}.json"
   if chunkpath.exists():
    chunk=read(chunkpath)
    if chunk.get("frozen_model_identity")!=identity or chunk.get("native_scoring_identity")!=NATIVE_SCORING_IDENTITY or chunk.get("frozen_model_identity_scope")!=FROZEN_IDENTITY_SCOPE or chunk.get("backend_identity")!=observed["backend_identity"] or chunk.get("device_identity")!=observed["device_identity"] or chunk.get("split")!=split or chunk.get("start")!=start or chunk.get("count")!=len(current):raise ValueError("Uncommitted chunk identity/extent differs; preserve this attempt.")
    validate_document_rows(chunk["documents"],current,split);validate_window_rows(current,chunk["documents"],chunk["windows"],index_offset=start);actual={"documents":chunk["documents"]};elapsed=chunk["scoring_seconds"]
   else:
    begun=time.perf_counter()
    with torch.inference_mode(),precision_context(torch,dev,args.format):actual=enrich(score_documents(model,tok,current,batch_size=batch_size,device=dev,config=c),current)
    elapsed=time.perf_counter()-begun;source_by_id={doc["document_id"]:doc for doc in current}
    for row in actual["windows"]:
     row["document_index"]+=start;source=source_by_id[row["document_id"]]
     for field in ("source_group_id","source_ids","source_dataset","attack_type","attack_template_id","injection_position"):row[field]=source.get(field)
    chunk={"frozen_model_identity":identity,"native_scoring_identity":NATIVE_SCORING_IDENTITY,"frozen_model_identity_scope":FROZEN_IDENTITY_SCOPE,"backend_identity":observed["backend_identity"],"device_identity":observed["device_identity"],"split":split,"start":start,"count":len(current),"documents":actual["documents"],"windows":actual["windows"],"scoring_seconds":elapsed};save_json(chunkpath,chunk)
   state["chunks"].append({"start":start,"count":len(current),"file":chunkpath.name});state["next_document"]=start+len(current);progress["scoring_seconds"]+=elapsed;save_json(progresspath,progress);scored.extend(actual["documents"])
  if len(scored)!=len(documents):raise ValueError("Qualification split is incomplete.")
  validate_document_rows(scored,documents,split)
  docs_path=outdir/(split+"_documents.jsonl")
  if docs_path.exists() and score_rows(docs_path)!=scored:raise ValueError("Existing aggregate score rows differ; preserve the attempt.")
  save_jsonl(docs_path,scored)
  windows=[row for chunk_meta in state["chunks"] for row in read(outdir/chunk_meta["file"])["windows"]]
  windows_path=outdir/(split+"_windows.jsonl")
  if windows_path.exists() and score_rows(windows_path)!=windows:raise ValueError("Existing aggregate window rows differ; preserve the attempt.")
  save_jsonl(windows_path,windows)
 metrics=summarize(score_rows(outdir/(SPLITS[0]+"_documents.jsonl")),score_rows(outdir/(SPLITS[1]+"_documents.jsonl")),c)
 if reference:
  ref,ref_metrics,ref_counts=validate_native_scores(rp(args.floating_reference),reference_identity,c,args.backend,"float32",observed["device_identity"],str(fp),require_eligible=False);ref_recall=ref_metrics["protection"]["recall"]
 else:ref_recall=metrics["primary"]["candidate_scoring"]["recall"]
 protection=metrics["primary"]["candidate_scoring"];quality_passed=(finite(protection.get("fpr")) and finite(protection.get("recall")) and finite(ref_recall) and protection["fpr"]<=c["evaluation"]["fpr_target"] and protection["recall"]>=ref_recall-c["evaluation"]["max_recall_loss_absolute"])
 run.update(state="completed_backend_qualification",benchmark_started=False,quality_passed=quality_passed,score_directories={s:str(outdir) for s in SPLITS},scoring_seconds=progress["scoring_seconds"],document_counts={s:len(score_rows(outdir/(s+"_documents.jsonl"))) for s in SPLITS},metrics=metrics,floating_reference_recall=ref_recall,recorded_finished_utc=datetime.now(timezone.utc).isoformat());save_json(runpath,run)
 candidate={"generator":"measure_frozen_software.py:qualify-backend","state":"completed_backend_qualification","finalist_role":role,"frozen_model_identity":identity,"native_scoring_identity":NATIVE_SCORING_IDENTITY,"frozen_model_identity_scope":FROZEN_IDENTITY_SCOPE,"backend":args.backend,"format":args.format,"backend_identity":observed["backend_identity"],"device_identity":observed["device_identity"],"metrics":metrics,"decision_threshold":metrics["primary"]["threshold"],"floating_reference_recall":ref_recall,"quality_passed":quality_passed,"document_counts":run["document_counts"],"freeze":str(fp),"development_scores_dir":str(outdir),"qualification_run":str(runpath),"data":c["data"],"windowing":c["windowing"],"evaluation":c["evaluation"]};save_json(outdir/"candidate.json",candidate)
 descriptor_out=None
 if quality_passed:
  if role=="floating" and args.format=="float32":ref_folder=outdir;ref_identity=observed["backend_identity"]
  else:ref_folder=rp(args.floating_reference);ref_identity=ref["backend_identity"]
  descriptor={**base,"state":"qualified_backend_development","backend_identity":observed["backend_identity"],"device_identity":observed["device_identity"],"development_scores_dir":str(outdir),"development_evidence":str(outdir/"candidate.json"),"floating_reference_scores_dir":str(ref_folder),"floating_reference_backend_identity":ref_identity,"decision_threshold":metrics["primary"]["threshold"],"qualification_run":str(runpath)}
  descriptor_out=rp(args.descriptor_output) if args.descriptor_output else ROOT/"results/goal7/backend_descriptors"/f"{args.backend}_{args.format}_{outdir.name}.json"
  if descriptor_out.exists():raise ValueError("Preserve existing backend descriptor; choose a new --descriptor-output.")
  save_json(descriptor_out,descriptor)
  candidate["backend_descriptor"]=str(descriptor_out);save_json(outdir/"candidate.json",candidate)
  run["backend_descriptor"]=str(descriptor_out);save_json(runpath,run)
 print(json.dumps({"state":"complete" if quality_passed else "quality_rejected","attempt":str(outdir),"candidate":str(outdir/"candidate.json"),"backend_descriptor":str(descriptor_out) if descriptor_out else None,"metrics":metrics,"heldout_content_read":False},indent=2))
 return descriptor_out

def save_jsonl(path,rows):
 path=Path(path);tmp=path.with_suffix(path.suffix+".tmp")
 with tmp.open("w",encoding="utf-8") as stream:
  for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+"\n")
 tmp.replace(path)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=("qualify-backend","timing","load"),default="timing")
 p.add_argument("--backend-descriptor");p.add_argument("--backend",choices=("cpu","cuda"));p.add_argument("--format",choices=("float32","dynamic_qint8","float16","bfloat16"))
 p.add_argument("--finalist-role",choices=ROLES,default="floating");p.add_argument("--intraop-threads",type=int,default=4);p.add_argument("--inter-op-threads",type=int,default=1);p.add_argument("--quantized-engine",default="fbgemm")
 p.add_argument("--floating-reference");p.add_argument("--resume-attempt", "--attempt-dir", dest="resume_attempt");p.add_argument("--descriptor-output");p.add_argument("--chunk-documents",type=int,default=64)
 p.add_argument("--freeze");p.add_argument("--host-workload",default="results/goal4/host_preparation/l256_b1/run.json")
 p.add_argument("--execute",action="store_true");p.add_argument("--warmup-checks",type=int,default=100);p.add_argument("--secondary-checks",type=int,default=10000,help="Checks per repeat in each non-primary cell; rounded up to full batches. Below10000 is diagnostic coverage only.")
 p.add_argument("--energy-csv");p.add_argument("--energy-completed-checks",type=int);p.add_argument("--capacity-evidence")
 p.add_argument("--arrival-csv",action="append",default=[]);p.add_argument("--arrival-batch-size",type=int,default=1)
 a=p.parse_args()
 if a.energy_csv and a.stage!="load":p.error("Measured energy is accepted only with the matched-load stage.")
 if a.energy_completed_checks is not None and a.energy_completed_checks<1:p.error("Measured completed-check denominator must be positive.")
 if a.warmup_checks<1 or a.secondary_checks<32:p.error("Require positive warmups and at least32 secondary checks.")
 if a.arrival_batch_size not in (1,8,32):p.error("Invalid arrival batch size.")
 if a.chunk_documents<1:p.error("chunk-documents must be positive.")
 from gate_eval import load_config
 config=load_config();fp=rp(a.freeze or config.get("final_evaluation",{}).get("development_freeze_path","results/goal5/development_freeze.json"))
 if a.stage=="qualify-backend":
  c,pr,f=frozen_gate(fp);blockers=list(pr["blockers"]);compatible=True
  if a.backend not in ("cpu","cuda"):blockers.append("Qualification needs an explicit cpu or cuda backend.");compatible=False
  if a.format not in ("float32","dynamic_qint8","float16","bfloat16"):blockers.append("Qualification needs explicit float32/dynamic_qint8/float16/bfloat16 format.");compatible=False
  if a.backend=="cuda" and a.format=="dynamic_qint8":blockers.append("dynamic_qint8 is CPU-only.");compatible=False
  if a.backend=="cpu" and a.format in ("float16","bfloat16"):blockers.append("float16/bfloat16 autocast is CUDA-only in this driver.");compatible=False
  if a.intraop_threads<1 or a.inter_op_threads<1:blockers.append("Positive actual intraop/inter-op thread settings are required.");compatible=False
  if (a.finalist_role!="floating" or (a.format and a.format!="float32")) and not a.floating_reference:blockers.append("QAT or optimized qualification requires an actual same-backend floating native FP32 reference directory.");compatible=False
  if a.finalist_role=="floating" and a.format=="float32" and a.backend and not a.floating_reference:
   found=existing_float32_candidates(a.backend,(f or {}).get("finalists",{}).get("floating",{}).get("identity"),a.intraop_threads,a.inter_op_threads,a.quantized_engine if a.backend=="cpu" else None)
   if found:blockers.append("A completed native FP32 baseline already exists; reuse it with --floating-reference "+str(found[0])+".");compatible=False
  try:native_checkpoint_spec(a.finalist_role,(f or {})["finalists"][a.finalist_role])
  except (OSError,ValueError,KeyError,TypeError) as x:blockers.append("Native frozen checkpoint prerequisites: "+str(x));compatible=False
  reference=None
  if compatible and a.finalist_role=="floating" and a.format=="float32" and a.floating_reference:
   try:
    fc=read(rp(a.floating_reference)/"candidate.json");reference,_,_=validate_native_scores(rp(a.floating_reference),(f or {})["finalists"]["floating"]["identity"],c,a.backend,"float32",fc.get("device_identity"),str(fp),require_eligible=False)
   except (OSError,ValueError,KeyError,TypeError,RuntimeError) as x:blockers.append("Native FP32 reuse prerequisite: "+str(x))
  if compatible and (a.finalist_role!="floating" or a.format!="float32") and a.floating_reference:
   try:
    fc=read(rp(a.floating_reference)/"candidate.json");reference,_,_=validate_native_scores(rp(a.floating_reference),f["finalists"]["floating"]["identity"],c,a.backend,"float32",fc.get("device_identity"),str(fp),require_eligible=False)
   except (OSError,ValueError,KeyError,TypeError,RuntimeError) as x:blockers.append("Native FP32 reference prerequisite: "+str(x))
  preview={"stage":a.stage,"ready":not blockers,"execute":a.execute,"blockers":list(dict.fromkeys(blockers)),"freeze":str(fp),"finalist_role":a.finalist_role,"backend":a.backend,"format":a.format,"same_backend_fp32_reference":str(rp(a.floating_reference)) if a.floating_reference else None,"reusing_existing_fp32":bool(a.finalist_role=="floating" and a.format=="float32" and a.floating_reference and reference),"partitions":["search_temporary_threshold","search_candidate_scoring"],"chunks_resume":True,"model_loaded":False,"heldout_content_read":False,"benchmark_started":False}
  if not a.execute or blockers:print(json.dumps(preview,indent=2));return 2 if a.execute and blockers else 0
  qualify_backend(a,fp,c,pr,f);return 0
 dp=rp(a.backend_descriptor) if a.backend_descriptor else None;c,d,r=gate(dp,fp)
 r.update(stage=a.stage,configuration={"length":256,"batch_size":1},grid=[{"length":L,"batch_size":b,"primary":(L,b)==(256,1),"checks":10000 if (L,b)==(256,1) else ((a.secondary_checks+b-1)//b)*b,"repeats":3} for L in LENGTHS for b in (1,8,32)],warmup_checks=a.warmup_checks,heldout_content_read=False,model_loaded=False,benchmark_started=False)
 r["final_timing_grid_scope"]="Conservative adopted scope: all nine L128/256/512 x B1/8/32 cells, at least10000 completed checks per repeat, three repeats, both boundaries."
 r["final_timing_grid_coverage_requested"]=a.secondary_checks>=10000
 r["timing_coverage_kind"]="full_final_grid" if a.secondary_checks>=10000 else "diagnostic_secondary_coverage_only"
 capacity=None;arrivals=[]
 if a.stage=="load":
  r["matched_arrivals"]={"ready":False,"reason":"Actual software+FPGA capacity evidence and all three common schedules are required."}
  if not a.capacity_evidence:r["blockers"].append("Load stage requires actual software and FPGA capacity evidence.")
  else:
   try:
    capdoc,rates,slow=validate_capacities(rp(a.capacity_evidence));capacity={"evidence":str(rp(a.capacity_evidence)),"devices":rates,"slowest_checks_per_second":slow}
    arrivals=read_arrivals(a.arrival_csv,slow);r["matched_arrivals"]={"ready":True,"fractions":[.25,.60,.85],"slowest_actual_capacity_checks_per_second":slow,"arrival_paths":[str(x) for x in arrivals]}
   except (OSError,ValueError,KeyError,TypeError,statistics.StatisticsError) as x:r["blockers"].append("Matched-load prerequisites: "+str(x))
  if not r.get("matched_arrivals",{}).get("ready",False):r["blockers"].append("Matched arrivals are not ready: valid actual capacities and all three schedules are required.")
  if a.energy_csv:r["energy"]=energy(rp(a.energy_csv),a.energy_completed_checks,d.get("backend","software")+"_"+str(d.get("format","unknown")))
  else:r["energy"]=None
 else:r["matched_arrivals"]={"ready":False,"reason":"Timing stage creates actual software capacity; load waits for actual software+FPGA evidence."};r["energy"]=None
 try:
  load_host_docs(rp(a.host_workload));r["host_workload"]={"ready":True,"path":str(rp(a.host_workload)),"ordered_documents":32,"selection":"first16 benign then first16 attacked; verified against source partition"}
 except (OSError,ValueError,KeyError,TypeError) as x:r["host_workload"]={"ready":False,"path":str(rp(a.host_workload))};r["blockers"].append("Board-aligned host workload prerequisite: "+str(x))
 r["blockers"]=list(dict.fromkeys(r["blockers"]));r["ready"]=not r["blockers"]
 if not a.execute:print(json.dumps(r,indent=2));return 0
 if not r["ready"]:print(json.dumps(r,indent=2));raise SystemExit("Blocked by frozen/backend/workload prerequisites.")
 docs=load_host_docs(rp(a.host_workload));import torch
 dev=torch.device("cuda" if d["backend"]=="cuda" else "cpu")
 torch.set_num_threads(d["intraop_threads"]);torch.set_num_interop_threads(d["inter_op_threads"])
 if dev.type=="cpu":
  if d.get("quantized_engine") not in torch.backends.quantized.supported_engines:raise RuntimeError("Qualified CPU quantized engine is unavailable.")
  torch.backends.quantized.engine=d["quantized_engine"]
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision("highest")
 model,tok=load(d,dev);observed=backend_runtime_identity(torch,model,dev,d["format"],d["intraop_threads"],d.get("quantized_engine"))
 if observed["backend_identity"]!=d.get("backend_identity") or observed["device_identity"]!=d.get("device_identity"):raise RuntimeError("Loaded software runtime identity differs from the actual qualified backend evidence; no measurement started.")
 payload_association=validate_host_payload(tok,rp(a.host_workload),docs,c)
 if a.stage=="load":docs,_=workload(tok,rp(a.host_workload),docs,c,256)
 stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ");outdir=OUT/f"{d['backend']}_{d['format']}_{stamp}";outdir.mkdir(parents=True,exist_ok=False)
 result={"state":"running","stage":a.stage,"recorded_utc":datetime.now(timezone.utc).isoformat(),"descriptor":d,"freeze":str(fp),"backend_identity":observed["backend_identity"],"device_identity":observed["device_identity"],
  "device":str(dev),"device_name":torch.cuda.get_device_name(dev) if dev.type=="cuda" else __import__("platform").processor(),"torch_version":torch.__version__,"cuda_runtime":torch.version.cuda,"intraop_threads":torch.get_num_threads(),"inter_op_threads":torch.get_num_interop_threads(),
  "format":d["format"],"configuration":{"length":256,"batch_size":1},"source":"same32 ordered host-preparation development docs; board timing/load uses only single-window documents","host_workload":str(rp(a.host_workload)),"host_payload_association":payload_association,"unit":"completed full document checks/second","heldout_content_read":False}
 try:
  if a.stage=="timing":
   result["primary_L256_B1"]=primary(model,tok,docs,c,dev,d["format"],r["backend_development_protection"]["threshold"],a.warmup_checks,rp(a.host_workload))
   result["secondary_grid"]=secondary(model,tok,docs,c,dev,d["format"],r["backend_development_protection"]["threshold"],a.secondary_checks,a.warmup_checks,rp(a.host_workload))
   result["final_timing_grid_qualified"]=len(result["secondary_grid"])==8 and all(x["final_timing_cell_qualified"] for x in result["secondary_grid"]) and len(result["primary_L256_B1"])==3 and all(x["completed_checks"]>=10000 for x in result["primary_L256_B1"])
   result["timing_coverage_kind"]="full_final_grid" if result["final_timing_grid_qualified"] else "diagnostic_secondary_coverage_only"
   result["capacity"]={"device_kind":"software","boundary":"raw_text_to_decision","capacity_unit":"full_document_checks_per_second","checks_per_second":statistics.median(x["raw_text_throughput_checks_per_second"] for x in result["primary_L256_B1"]),"completed_checks_per_repeat":10000,"repeats":3};result["state"]="measured"
  else:
   result["capacity_evidence"]=capacity;result["arrival_replays"]=[replay_arrivals(model,tok,docs,c,dev,x,a.arrival_batch_size,r["backend_development_protection"]["threshold"],d["format"]) for x in arrivals];result["energy"]=r["energy"];result["state"]="measured"
  runpath=outdir/"run.json";save_json(runpath,result)
  if a.stage=="timing":save_json(outdir/"capacity.json",{"devices":[dict(result["capacity"],id=d["backend"]+"_"+d["format"],evidence=str(runpath))]})
  print(json.dumps({"state":"complete","output":str(runpath),"capacity":str(outdir/"capacity.json") if a.stage=="timing" else None},indent=2))
 except BaseException as x:result.update(state="failed",error=str(x));save_json(outdir/"run.json",result);raise
 return 0
if __name__=="__main__":raise SystemExit(main())


















