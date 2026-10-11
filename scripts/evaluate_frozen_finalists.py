"""Frozen Goal 7 quality, nonadaptive challenge and one pinned TableQA extension.

Preview never reads held-out content or loads a model. Execution requires a real
validated development freeze and explicit quality readiness. No target API.
"""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import argparse,csv,json,math,random,re,string,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
ROLES=('floating','U','B','C')
SPLITS={'final_calibration':'final_calibration.jsonl','test':'test.jsonl','notinject':'notinject.jsonl'}
OUT=ROOT/'results/goal7/final_quality'
TABLEQA_REVISION='a004b69ec0dd446e0afd461d98cb5e96e120a5d0'
TABLEQA_TASK='BIPIA TableQA'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def rows(path):
 with Path(path).open(encoding='utf-8-sig') as stream:return [json.loads(line) for line in stream if line.strip()]
def path(value):
 result=Path(str(value).replace(chr(92),"/"));return result if result.is_absolute() else ROOT/result
def save(destination,value):
 destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_suffix(destination.suffix+'.tmp')
 temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');temporary.replace(destination)
def config():return read(ROOT/'configs/project.json')

def prerequisites(cfg,freeze_path=None):
 """Only configuration, development evidence and file existence are inspected."""
 from search_precision import approved_score_identity,validate_score_identity,precision_tuple,recompute_metrics,protection_eligible
 from score_qualified_development import scoring_batch_policy
 target=path(freeze_path or cfg.get('final_evaluation',{}).get('development_freeze_path','results/goal5/development_freeze.json'))
 missing=[];frozen=None
 if cfg.get('final_evaluation',{}).get('quality_evaluation_ready') is not True:missing.append('final_evaluation.quality_evaluation_ready must be true after development freeze')
 unresolved=False
 for key in ('approved_score_identity','approved_floating_score_identity'):
  for field in ('arithmetic_identity','execution_identity'):
   value=(cfg['methods'].get(key) or {}).get(field)
   if not isinstance(value,str) or not value.strip():missing.append('methods.'+key+'.'+field+' unresolved');unresolved=True
 if not unresolved:
  for floating in (True,):
   try:
    approved=approved_score_identity(cfg,floating=floating);batch_size,_=scoring_batch_policy(cfg,floating=floating)
    if batch_size!=1 and not approved['execution_identity'].endswith(f'__document_score_batch{batch_size}_tail_actual'):raise ValueError('Approved execution identity differs from configured qualified batching')
   except (RuntimeError,ValueError,KeyError,OSError) as error:missing.append(str(error))
 if not target.is_file():missing.append('Real development freeze missing: '+str(target))
 else:
  try:
   frozen=read(target)
   if frozen['state']!='frozen_development':raise ValueError('Development freeze is not final')
   for key in ('evaluation','windowing'):
    if frozen[key]!=cfg[key]:raise ValueError('Frozen '+key+' policy differs')
   if set(frozen['finalists'])!=set(ROLES):raise ValueError('Freeze must assign floating/U/B/C')
   if not cfg['methods'].get('proposed_mechanism') or frozen.get('mechanism')!=cfg['methods']['proposed_mechanism']:raise ValueError('Actual C mechanism remains unresolved or differs')
   actual_by_role={}
   for role,entry in frozen['finalists'].items():
    identity=entry['identity'];floating=role=='floating'
    validate_score_identity(identity,cfg,qat=identity.get('qat') is True,floating=floating)
    if not floating and (entry['precision_map']!=identity['precision_map'] or identity.get('qat') is not True):raise ValueError(role+' must identify its actual post-QAT precision/checkpoint')
    if not floating:precision_tuple(entry['precision_map'],cfg)
    if floating and entry['precision_map'] is not None:raise ValueError('Floating reference cannot have a precision map')
    expected=identity['checkpoint'] if floating else identity['scored_checkpoint']
    if path(entry['checkpoint']).resolve()!=path(expected).resolve() or not path(entry['checkpoint']).is_dir():raise ValueError(role+' checkpoint association missing/different')
    if not floating:
     scoped=dict(cfg,_ordered_scoring_binding=dict(precision_map=entry['precision_map'],qat_checkpoint=entry['checkpoint']))
     batch_size,_=scoring_batch_policy(scoped,floating=False)
     if batch_size!=1 and not identity['execution_identity'].endswith(f'__document_score_batch{batch_size}_tail_actual'):raise ValueError(role+' frozen execution identity differs from its actual qualified batch policy')
    candidate=path(entry['development_scores'])/'candidate.json'
    if not candidate.is_file() or read(candidate)['identity']!=identity:raise ValueError(role+' lacks actual matching development candidate scores')
    saved=read(candidate)['metrics'];actual=recompute_metrics(candidate.parent,cfg)
    if actual['temporary_threshold']!=saved['primary']['threshold'] or actual['protection']!=saved['primary']['candidate_scoring'] or actual['ordinary_0_5']!=saved['ordinary_threshold_0_5']:raise ValueError(role+' actual development score rows differ from selected metrics')
    actual_by_role[role]=actual
   floating_recall=actual_by_role['floating']['protection']['recall']
   if not isinstance(floating_recall,(int,float)) or not math.isfinite(floating_recall) or not 0<=floating_recall<=1:raise ValueError('Floating reference has no valid recomputed candidate-scoring recall')
   for role in ('U','B','C'):
    protection=actual_by_role[role]['protection']
    if not isinstance(protection.get('fpr'),(int,float)) or not math.isfinite(protection['fpr']) or not isinstance(protection.get('recall'),(int,float)) or not math.isfinite(protection['recall']):raise ValueError(role+' has incomplete recomputed development protection metrics')
    if not protection_eligible(protection,floating_recall,cfg):raise ValueError(role+' fails frozen development protection against the recomputed floating reference')
   if len(set(frozen['finalists']['U']['precision_map'].values()))!=1:raise ValueError('U must be a genuine uniform anchor')
  except (RuntimeError,ValueError,KeyError,TypeError,OSError) as error:missing.append('Development freeze: '+str(error))
 for split,filename in SPLITS.items():
  if not (ROOT/'data/prepared'/filename).is_file():missing.append('Prepared '+split+' file missing')
 for marker in ('results/goal2/pause.request','results/goal3/pause.request','results/goal7/pause.request'):
  if (ROOT/marker).exists():missing.append('Pause marker: '+marker)
 if cfg.get('execution_control',{}).get('current_pause'):missing.append('Current execution pause is active')
 return dict(ready=not missing,blockers=list(dict.fromkeys(missing)),freeze_path=str(target),heldout_content_read=False,model_loaded=False),frozen

def require_ready(cfg,freeze_path=None):
 preview,frozen=prerequisites(cfg,freeze_path)
 if not preview['ready']:raise RuntimeError('No held-out read/model work: '+'; '.join(preview['blockers']))
 return frozen

def validate_membership(scored,documents):
 if len(scored)!=len(documents):raise ValueError('Cached document count differs')
 for actual,expected in zip(scored,documents):
  if any(actual.get(k)!=expected.get(k) for k in ('document_id','source_group_id','label','source_dataset')):raise ValueError('Cached document membership/order differs')
  if not actual.get('window_scores') or actual['window_count']!=len(actual['window_scores']) or actual['risk_score']!=max(actual['window_scores']):raise ValueError('Cached document maximum differs')
  if not math.isfinite(actual['risk_score']):raise ValueError('Nonfinite cached risk')

def score_partition(folder,documents,entry,frozen,cfg,allowance,chunk_size,loaded):
 """Chunk commit precedes progress; recover an already finished next chunk."""
 from gate_eval import score_documents
 from train_detector import enrich
 from score_qualified_development import load_model,paused,scoring_batch_policy
 from search_precision import validate_score_identity,precision_tuple
 floating=entry['precision_map'] is None;identity=entry['identity']
 validate_score_identity(identity,cfg,qat=identity.get('qat') is True,floating=floating)
 if not floating:
  if entry['precision_map']!=identity['precision_map'] or identity.get('qat') is not True:raise ValueError('Final scoring requires actual post-QAT map/checkpoint association')
  precision_tuple(entry['precision_map'],cfg)
 expected=identity['checkpoint'] if floating else identity['scored_checkpoint']
 if path(entry['checkpoint']).resolve()!=path(expected).resolve() or not path(entry['checkpoint']).is_dir():raise ValueError('Final scoring actual checkpoint association missing/different')
 scoped=dict(cfg,_ordered_scoring_binding=dict(precision_map=entry['precision_map'],qat_checkpoint=None if floating else entry['checkpoint']))
 batch_size,_=scoring_batch_policy(scoped,floating=floating)
 if batch_size!=1 and not identity['execution_identity'].endswith(f'__document_score_batch{batch_size}_tail_actual'):raise ValueError('Frozen execution identity differs from its verified configured batch policy')
 progresspath=folder/'progress.json';progress=read(progresspath) if progresspath.exists() else dict(identity=entry['identity'],freeze=frozen,chunks=[],next_document=0,complete=False,scoring_seconds=0.)
 if progress['identity']!=entry['identity'] or progress['freeze']!=frozen:raise ValueError('Saved final scoring freeze/identity differs')
 scored=[];position=0;new=0
 for committed in progress['chunks']:
  if committed['start']!=position:raise ValueError('Cached final chunk order differs')
  chunk=read(folder/committed['file'])
  if chunk['identity']!=entry['identity'] or chunk['start']!=position or chunk['count']!=committed['count']:raise ValueError('Cached final chunk association differs')
  current=documents[position:position+chunk['count']];validate_membership(chunk['documents'],current)
  scored.extend(chunk['documents']);position+=len(current)
 if position!=progress['next_document']:raise ValueError('Committed final resume position differs')
 while position<len(documents):
  paused()
  if (ROOT/'results/goal7/pause.request').exists():raise RuntimeError('Goal7 pause marker present; preserve scoring state')
  chunkpath=folder/f'chunk{position:06d}.json'
  if chunkpath.exists():
   chunk=read(chunkpath)
   if chunk['identity']!=entry['identity'] or chunk['start']!=position:raise ValueError('Uncommitted chunk identity differs')
   current=documents[position:position+chunk['count']];validate_membership(chunk['documents'],current)
  else:
   if new>=allowance:break
   current=documents[position:position+min(chunk_size,allowance-new)]
   if not current:break
   if not loaded:
    floating=entry['precision_map'] is None
    model,tokenizer,identity=load_model(scoped,entry['precision_map'],None if floating else entry['checkpoint'],floating=floating,scoring=False)
    if batch_size!=1:identity=dict(identity,execution_identity=identity['execution_identity']+f'__document_score_batch{batch_size}_tail_actual')
    if identity!=entry['identity']:raise ValueError('Loaded final model/settings differ from frozen identity')
    loaded.extend((model,tokenizer))
   begun=time.perf_counter();actual=enrich(score_documents(loaded[0],loaded[1],current,batch_size=batch_size,device='cpu',config=cfg,risk_function=loaded[0].predict_risk),current);elapsed=time.perf_counter()-begun
   chunk=dict(identity=entry['identity'],batch_size=batch_size,start=position,count=len(current),documents=actual['documents'],windows=actual['windows'],scoring_seconds=elapsed,timing_scope='research scoring work; not a detector performance benchmark')
   validate_membership(chunk['documents'],current);save(chunkpath,chunk);new+=len(current)
  if chunk['count']<1 or position+chunk['count']>len(documents):raise ValueError('Invalid committed final chunk size')
  progress['chunks'].append(dict(start=position,count=chunk['count'],file=chunkpath.name));progress['scoring_seconds']+=chunk['scoring_seconds']
  scored.extend(chunk['documents']);position+=chunk['count'];progress['next_document']=position;progress['complete']=position==len(documents);save(progresspath,progress)
 return scored,progress['complete'],new

def metrics(scored,threshold):
 from gate_eval import evaluate_documents
 return evaluate_documents([r['risk_score'] for r in scored],[r['label'] for r in scored],threshold)

def quantile(values,q):
 values=sorted(values);return values[int((len(values)-1)*q)] if values else None

def uncertainty(scored_by_role,thresholds,seed,repeats=500):
 """Paired group bootstrap, conditional on already frozen calibration thresholds."""
 ids=[[r['document_id'] for r in scored_by_role[role]] for role in ROLES]
 if any(v!=ids[0] for v in ids):raise ValueError('Paired methods must use identical documents/order')
 groups=defaultdict(list)
 for index,row in enumerate(scored_by_role['B']):groups[row['source_group_id']].append(index)
 keys=sorted(groups);rng=random.Random(seed);samples={role:{'recall':[],'fpr':[]} for role in ROLES};paired={name:{'recall':[],'fpr':[]} for name in ('B_minus_U','C_minus_B')}
 # Summarize counts once per source group; never treat windows as independent.
 counts={role:{} for role in ROLES}
 for role in ROLES:
  for key,indexes in groups.items():counts[role][key]=metrics([scored_by_role[role][i] for i in indexes],thresholds[role])
 for _ in range(repeats):
  chosen=rng.choices(keys,k=len(keys));rates={}
  for role in ROLES:
   totals={name:sum(counts[role][key][name] for key in chosen) for name in ('tp','fn','fp','tn')}
   rates[role]={'recall':totals['tp']/(totals['tp']+totals['fn']) if totals['tp']+totals['fn'] else None,'fpr':totals['fp']/(totals['fp']+totals['tn']) if totals['fp']+totals['tn'] else None}
   for metric,value in rates[role].items():
    if value is not None:samples[role][metric].append(value)
  for name,left,right in [('B_minus_U','B','U'),('C_minus_B','C','B')]:
   for metric in ('recall','fpr'):
    if rates[left][metric] is not None and rates[right][metric] is not None:paired[name][metric].append(rates[left][metric]-rates[right][metric])
 def intervals(values):return {metric:dict(low=quantile(v,.025),high=quantile(v,.975),valid_resamples=len(v)) for metric,v in values.items()}
 return dict(unit='original source group',source_groups=len(keys),resamples=repeats,seed=seed,confidence=.95,thresholds_refitted=False,method={role:intervals(value) for role,value in samples.items()},paired_changes={name:intervals(value) for name,value in paired.items()})

def summarize_pool(scored_by_role,thresholds,cfg):
 primary={role:thresholds[role]['primary'] for role in ROLES}
 result=dict(actual_documents=len(scored_by_role['B']),actual_source_groups=len({r['source_group_id'] for r in scored_by_role['B']}),methods={})
 for role,scored in scored_by_role.items():
  strata={name:metrics([r for r in scored if r['source_dataset']==name],primary[role]) for name in sorted({r['source_dataset'] for r in scored})}
  result['methods'][role]=dict(primary=metrics(scored,primary[role]),nearby={alpha:metrics(scored,t) for alpha,t in thresholds[role]['nearby'].items()},source_strata=strata,single_window=metrics([r for r in scored if r['window_count']==1],primary[role]),multiple_windows=metrics([r for r in scored if r['window_count']>1],primary[role]))
 result['uncertainty']=uncertainty(scored_by_role,primary,cfg['seed']+700)
 result['paired_point_changes']={name:{key:result['methods'][left]['primary'][key]-result['methods'][right]['primary'][key] if result['methods'][left]['primary'][key] is not None and result['methods'][right]['primary'][key] is not None else None for key in ('recall','fpr')} for name,left,right in [('B_minus_U','B','U'),('C_minus_B','C','B')]}
 return result

def challenge_pool(test,cfg):
 groups=defaultdict(list)
 for doc in test:
  if doc.get('source_dataset')=='BIPIA EmailQA' or doc.get('task_records'):groups[doc['source_group_id']].append(doc)
 number=cfg['final_evaluation']['challenge_tasks'];limit=cfg['final_evaluation']['max_variants_per_task']
 if number!=20 or not 1<=limit<=3 or len(groups)<number:raise ValueError('Bounded20-group challenge cannot be prepared from actual held-out pool')
 selected=random.Random(cfg['seed']+701).sample(sorted(groups),number);pool=[];description=[]
 for group in selected:
  variants=sorted((r for r in groups[group] if r['label']==1),key=lambda r:r['document_id'])[:limit]
  if not variants:raise ValueError('Challenge group lacks existing attack variants')
  pool.extend(variants);description.append(dict(source_group_id=group,document_ids=[r['document_id'] for r in variants],positions=[r.get('injection_position') for r in variants],attack_templates=[r.get('attack_template_id') for r in variants]))
 return pool,dict(tasks=number,max_variants=limit,actual_variants=len(pool),groups=description,adaptive=False,detector_targeted=None,scope='Existing held-out wording/position variants shared by all methods; no detector-feedback generation or API; repeated-attempt group outcome is any variant passing')

def prepare_newsqa(csv_path,cfg):
 """Pinned BIPIA test selection from the locally assembled licensed NewsQA CSV.

 Reproduce BIPIA qa/process.py with the reviewed HF combined-csv row decoder;
 parse character ranges directly, without eval, datasets dependency or hashes.
 """
 from prepare_data import download
 csv_path=path(csv_path)
 if not csv_path.is_file():raise ValueError('Locally assembled combined-newsqa-data-v1.csv is required')
 revision=cfg['data']['preparation']['bipia_revision'];folder=ROOT/'data/raw'/('bipia-'+revision)/'benchmark/qa'
 url=f'https://raw.githubusercontent.com/microsoft/BIPIA/{revision}/benchmark/qa/'
 indexes=read(download(url+'index.json',folder/'index.json'))['test']
 if not indexes or any(type(i) is not int or i<0 for i in indexes) or len(set(indexes))!=len(indexes):raise ValueError('Pinned NewsQA test index is malformed')
 selected={};wanted=set(indexes);position=0;csv.field_size_limit(16*1024*1024)
 with csv_path.open(encoding='utf-8-sig',newline='') as stream:
  reader=csv.reader(stream,quotechar='"',delimiter=',',quoting=csv.QUOTE_ALL,skipinitialspace=True);next(reader)
  for row in reader:
   if not row:continue
   if position in wanted:
    if len(row)<5:raise ValueError('NewsQA combined CSV row shape differs')
    story=row[-1];answers=set()
    for group in row[2].split(','):
     for value in group.split('|'):
      if value=='None':continue
      match=re.fullmatch(r'(\d+):(\d+)',value)
      if not match:raise ValueError('NewsQA character range differs')
      start,end=map(int,match.groups())
      if not 0<=start<=end<=len(story):raise ValueError('NewsQA answer span outside original story')
      answers.add(story[start:end].strip(string.punctuation+string.whitespace))
    context=re.sub(r'\n\s*\n','\n\n',story);context=re.sub(r'\n{3,}','\n\n',context)
    selected[position]=dict(context=context,question=row[1],ideal=sorted(answers))
   position+=1
   if len(selected)==len(wanted):break
 if set(selected)!=wanted:raise ValueError('Licensed CSV lacks pinned NewsQA test rows')
 result=[selected[i] for i in indexes]
 if len(result)<=87:raise ValueError('Pinned BIPIA NewsQA test selection changed')
 # The upstream pinned processor explicitly corrects this answer annotation.
 result[87]['ideal']=['Janine Sligar']
 target=folder/'test.jsonl';raw=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in result)
 if target.exists() and rows(target)!=result:raise ValueError('Preserve existing NewsQA prepared data; source differs')
 if not target.exists():
  temporary=target.with_suffix('.jsonl.tmp');temporary.write_text(raw,encoding='utf-8');temporary.replace(target)
 record=dict(task='BIPIA WebQA (NewsQA)',bipia_revision=revision,processor_source=url+'process.py',index_source=url+'index.json',csv_decoder_source='https://github.com/huggingface/datasets/blob/1.18.4/datasets/newsqa/newsqa.py',licensed_input=str(csv_path),output=str(target),actual_test_rows=len(result),new_model_forwards=0)
 save(folder/'preparation.json',record);return record

def extension_pool(source,cfg):
 """Parse actual pinned BIPIA WebQA preparation output, never invent examples."""
 from prepare_data import normalized,attack_document,TEST_CATEGORIES
 source=path(source)
 if not source.is_file():raise RuntimeError('NewsQA requires prepared BIPIA benchmark/qa/test.jsonl; see reports/final_evaluation.md')
 data=rows(source);groups={}
 for index,row in enumerate(data,1):
  if not isinstance(row.get('context'),str) or not isinstance(row.get('question'),str) or not isinstance(row.get('ideal'),list) or not all(isinstance(v,str) for v in row['ideal']):raise ValueError('NewsQA source must use pinned BIPIA context/question/ideal-list schema')
  body=normalized(row['context'])
  if body:groups.setdefault(body,(index,row))
 if len(groups)<20:raise ValueError('NewsQA extension needs20actual distinct source bodies')
 forbidden=set()
 for filename in ('train','search_temporary_threshold','search_candidate_scoring','final_calibration','test'):
  forbidden.update(normalized(d['original_context']) for d in rows(ROOT/'data/prepared'/(filename+'.jsonl')))
 candidates=[value for body,value in sorted(groups.items()) if body not in forbidden]
 if len(candidates)<20:raise ValueError('NewsQA has fewer than20 source-disjoint bodies')
 selected=random.Random(cfg['seed']+702).sample(candidates,20)
 revision=cfg['data']['preparation']['bipia_revision'];attacks=read(ROOT/'data/raw'/('bipia-'+revision)/'benchmark/text_attack_test.json')
 payloads=[(category,index,payload) for category in TEST_CATEGORIES for index,payload in enumerate(attacks[category])];rng=random.Random(cfg['seed']+703);pool=[]
 for index,row in selected:
  group=f'bipia:qa:test:line-{index:04d}'
  clean=dict(document_id=group+':clean',source_group_id=group,source_ids=[group],source_dataset='BIPIA WebQA (NewsQA)',source_revision=revision,source_path=str(source),split='extension_test',text=row['context'],context=row['context'],original_context=row['context'],label=0,injection_spans=[],question=row['question'],reference_answer=row['ideal'],attack_type=None,attack_template_id=None,injection_position=None)
  pool.append(clean)
  for position,(category,attack_index,payload) in zip(('start','end'),rng.sample(payloads,2)):pool.append(attack_document(clean,category,attack_index,payload,position,'test'))
 return pool,dict(task='BIPIA WebQA (NewsQA)',source=str(source),bipia_revision=revision,source_groups=20,documents=len(pool),adaptation='Frozen EmailQA U/B/C and thresholds transferred without refitting or training; two original harmless held-out payloads per source',additional_model=False,additional_hardware=False)

def tableqa_row(row):
 """Validate published table schema without rewriting cells or row breaks."""
 if not isinstance(row,dict) or set(row)!={'context','question','ideal'} or any(not isinstance(row[k],str) or not row[k].strip() for k in ('context','question','ideal')):raise ValueError('Pinned TableQA requires nonempty context/question/ideal strings')
 return row

def tableqa_extension_pool(source,cfg,frozen,freeze_path=None):
 """One post-freeze document extension; question/ideal remain metadata."""
 actual_frozen=require_ready(cfg,freeze_path)
 if frozen!=actual_frozen:raise ValueError('Supplied TableQA freeze differs from the actual validated development freeze')
 from prepare_data import normalized,attack_document,TEST_CATEGORIES
 if not isinstance(frozen,dict) or frozen.get('state')!='frozen_development' or frozen.get('evaluation')!=cfg['evaluation'] or frozen.get('windowing')!=cfg['windowing']:raise ValueError('Actual validated development freeze required before TableQA selection')
 revision=cfg['data']['preparation']['bipia_revision'];source=path(source)
 expected=ROOT/'data/raw'/('bipia-'+TABLEQA_REVISION)/'benchmark/table/test.jsonl'
 if revision!=TABLEQA_REVISION or cfg['final_evaluation'].get('single_extension_task')!=TABLEQA_TASK or source.resolve()!=expected.resolve() or source.is_symlink():raise ValueError('Only the selected pinned BIPIA TableQA source is active')
 if not source.is_file():raise ValueError('Acquire pinned public TableQA test.jsonl before post-freeze preparation')
 grouped=defaultdict(list)
 for index,raw in enumerate(rows(source),1):
  row=tableqa_row(raw);grouped[normalized(row['context'])].append((index,row))
 forbidden=set();exclusion_counts={}
 for split in ('train','search_temporary_threshold','search_candidate_scoring','final_calibration','test'):
  existing=rows(ROOT/'data/prepared'/(split+'.jsonl'));exclusion_counts[split]=len(existing)
  for doc in existing:
   if not isinstance(doc.get('original_context'),str) or not normalized(doc['original_context']):raise ValueError('Original source body missing from exclusion split '+split)
   forbidden.add(normalized(doc['original_context']))
   for task in doc.get('task_records',[]):
    if not isinstance(task.get('original_context'),str):raise ValueError('Original task source missing from exclusion split '+split)
    forbidden.add(normalized(task['original_context']))
 candidates=[grouped[body] for body in sorted(grouped) if body not in forbidden]
 if len(candidates)<20:raise ValueError('TableQA requires20 distinct normalized source-disjoint groups')
 selected=random.Random(cfg['seed']+702).sample(candidates,20)
 attacks=read(ROOT/'data/raw'/('bipia-'+revision)/'benchmark/text_attack_test.json');payloads=[]
 for category in TEST_CATEGORIES:
  values=attacks[category]
  if not isinstance(values,list) or not values or any(not isinstance(v,str) or not v for v in values):raise ValueError('Existing held-out BIPIA payload schema differs')
  payloads.extend((category,index,payload) for index,payload in enumerate(values))
 if len(payloads)<2:raise ValueError('Two existing held-out payloads required')
 rng=random.Random(cfg['seed']+703);pool=[];selected_groups=[]
 for members in selected:
  first,row=members[0];group=f'bipia:table:test:line-{first:04d}';ids=[f'bipia:table:test:line-{i:04d}' for i,_ in members]
  tasks=[dict(source_id=source_id,source_row=i,question=v['question'],reference_answer=v['ideal'],original_context=v['context']) for source_id,(i,v) in zip(ids,members)]
  clean=dict(document_id=group+':clean',source_group_id=group,source_ids=ids,source_dataset=TABLEQA_TASK,source_revision=revision,source_path=str(source),source_rows=[i for i,_ in members],split='extension_test',text=row['context'],context=row['context'],original_context=row['context'],label=0,injection_spans=[],question=row['question'],reference_answer=row['ideal'],task_records=tasks,attack_type=None,attack_template_id=None,injection_position=None,source_license='CC-BY-SA-4.0; WikiTableQuestions via pinned BIPIA',llm_attack_success=None)
  pool.append(clean);variant_ids=[]
  for position,(category,index,payload) in zip(('start','end'),rng.sample(payloads,2)):
   attacked=attack_document(clean,category,index,payload,position,'test');pool.append(attacked);variant_ids.append(attacked['document_id'])
  selected_groups.append(dict(source_group_id=group,source_ids=ids,source_rows=clean['source_rows'],clean_document_id=clean['document_id'],attack_document_ids=variant_ids))
 if len(pool)!=60 or len({d['source_group_id'] for d in pool})!=20 or any(normalized(d['original_context']) in forbidden for d in pool):raise ValueError('TableQA group/variant/disjointness bound differs')
 description=dict(task=TABLEQA_TASK,source=str(source),source_url=f'https://raw.githubusercontent.com/microsoft/BIPIA/{revision}/benchmark/table/test.jsonl',bipia_revision=revision,source_groups=20,max_attack_variants_per_group=2,documents=len(pool),selection_seed=cfg['seed']+702,payload_seed=cfg['seed']+703,normalized_source_groups_available=len(grouped),excluded_normalized_source_groups=sum(body in forbidden for body in grouped),exclusion_split_document_counts=exclusion_counts,groups=selected_groups,detector_input='Exact original table context; row/column serialization unchanged; upstream start/end newline insertion. Questions/answers remain metadata.',adaptation='Frozen EmailQA floating/U/B/C checkpoints, thresholds and window policy transferred without refitting, training, map selection or target generation',selection_after_development_freeze=True,selection_uses_outcomes=False,additional_model=False,additional_hardware=False,source_attribution='Microsoft BIPIA TableQA, derived from WikiTableQuestions (Pasupat and Liang); pinned source/licenses retained in results/goal7/tableqa_extension_source_v1/sources',source_license='WikiTableQuestions CC-BY-SA-4.0; BIPIA code MIT with dataset exception',adaptation_license='CC-BY-SA-4.0; original contexts retained and injected start/end payload changes identified')
 return pool,description

def save_tableqa_pool(source,cfg,frozen,freeze_path=None):
 pool,description=tableqa_extension_pool(source,cfg,frozen,freeze_path);destination=OUT/'tableqa_pool.json';record=dict(freeze=frozen,description=description,documents=pool)
 if destination.exists() and read(destination)!=record:raise ValueError('Preserve existing TableQA selection; frozen pool differs')
 save(destination,record);return pool,description

def cached_scores(folder,documents,entry,frozen):
 progress=read(folder/'progress.json')
 if not progress['complete'] or progress['identity']!=entry['identity'] or progress['freeze']!=frozen:raise ValueError('Complete frozen cached scores required')
 actual=[row for chunk in progress['chunks'] for row in read(folder/chunk['file'])['documents']];validate_membership(actual,documents);return actual

def main():
 ap=argparse.ArgumentParser(description=__doc__);actions=ap.add_mutually_exclusive_group();actions.add_argument('--execute',action='store_true');actions.add_argument('--prepare',action='store_true');actions.add_argument('--prepare-newsqa',action='store_true');actions.add_argument('--prepare-tableqa',action='store_true');ap.add_argument('--newsqa-csv',type=Path);ap.add_argument('--check-prerequisites',action='store_true');ap.add_argument('--freeze',type=Path);ap.add_argument('--phase',choices=('quality','challenge','extension','all'),default='quality');ap.add_argument('--max-new-documents',type=int,default=32);ap.add_argument('--chunk-documents',type=int,default=16);ap.add_argument('--newsqa-source',type=Path);ap.add_argument('--tableqa-source',type=Path);args=ap.parse_args();
 if args.prepare_newsqa or args.newsqa_csv or args.newsqa_source:raise ValueError('Historical NewsQA preparation remains source-only; the single active extension is pinned BIPIA TableQA')
 cfg=config();preview,frozen=prerequisites(cfg,args.freeze)
 revision=cfg['data']['preparation']['bipia_revision'];tableqa=args.tableqa_source or ROOT/'data/raw'/('bipia-'+TABLEQA_REVISION)/'benchmark/table/test.jsonl'
 preview.update(execute=args.execute,prepare=args.prepare,phase=args.phase,source_files={key:str(ROOT/'data/prepared'/value) for key,value in SPLITS.items()},extension_task=TABLEQA_TASK,tableqa_source=str(tableqa),tableqa_source_exists=path(tableqa).is_file(),max_new_documents=args.max_new_documents,batch_size=cfg['methods'].get('ordered_scoring_batch_size',1),score_acceptance=None)
 if args.prepare or args.prepare_tableqa or args.phase in ('extension','all'):
  if cfg['final_evaluation'].get('single_extension_task')!=TABLEQA_TASK or revision!=TABLEQA_REVISION:preview['blockers'].append('Shared single_extension_task must explicitly select pinned BIPIA TableQA')
  if path(tableqa).resolve()!=(ROOT/'data/raw'/('bipia-'+TABLEQA_REVISION)/'benchmark/table/test.jsonl').resolve():preview['blockers'].append('Only the pinned TableQA source path is permitted')
  preview['ready']=not preview['blockers']
 if not args.execute and not args.prepare and not args.prepare_tableqa:

  print(json.dumps(preview,indent=2));return 0 if not args.check_prerequisites or preview['ready'] else 2
 frozen=require_ready(cfg,args.freeze)
 if not preview['ready']:raise ValueError('; '.join(preview['blockers']))
 if args.prepare_tableqa:
  _,description=save_tableqa_pool(tableqa,cfg,frozen,args.freeze);print(json.dumps(dict(task=TABLEQA_TASK,source_groups=description['source_groups'],documents=description['documents'],selection_after_development_freeze=True,model_loaded=False),indent=2));return 0
 if args.max_new_documents<1 or args.chunk_documents<1:raise ValueError('Positive bounded document/chunk allowances required')
 # No held-out file is opened before all gates above succeed.
 snapshot=OUT/'freeze.json'
 if snapshot.exists() and read(snapshot)!=frozen:raise ValueError('Existing final run belongs to a different freeze')
 save(snapshot,frozen)
 if args.prepare:
  save(OUT/'execution_plan.json',dict(freeze=frozen,source_files=preview['source_files'],tableqa_source=str(tableqa),batch_size=cfg['methods'].get('ordered_scoring_batch_size',1),calibration='benign final-calibration only',nearby='reuse cached scores',challenge='20 held-out BIPIA groups, up to3 existing variants, nonadaptive',extension='one pinned BIPIA TableQA task only',heldout_content_read=False));print('Prepared frozen execution plan; no held-out reads or inference');return 0
 remaining=args.max_new_documents;thresholdpath=OUT/'thresholds.json';thresholds=read(thresholdpath) if thresholdpath.exists() else dict(freeze=frozen,methods={})
 if thresholds['freeze']!=frozen:raise ValueError('Threshold freeze differs')
 if args.phase in ('quality','all'):
  from gate_eval import fit_threshold
  documents={split:rows(ROOT/'data/prepared'/filename) for split,filename in SPLITS.items()}
  for split,docs in documents.items():
   expected='notinject_challenge' if split=='notinject' else split
   if any(d.get('split')!=expected for d in docs) or len({d['document_id'] for d in docs})!=len(docs):raise ValueError('Held-out partition membership differs')
  documents['final_calibration']=[d for d in documents['final_calibration'] if d['label']==0]
  testgroups={d['source_group_id'] for d in documents['test']}
  if testgroups & {d['source_group_id'] for d in documents['final_calibration']}:raise ValueError('Calibration/test source groups overlap')
  for role,entry in frozen['finalists'].items():
   loaded=[]
   calibration,complete,new=score_partition(OUT/role/'final_calibration',documents['final_calibration'],entry,frozen,cfg,remaining,args.chunk_documents,loaded);remaining-=new
   if not complete:print(json.dumps(dict(state='bounded_handoff',role=role,partition='final_calibration',new_documents=args.max_new_documents-remaining)));return 0
   benign=[d['risk_score'] for d in calibration if d['label']==0]
   fitted=dict(primary=fit_threshold(benign,cfg['evaluation']['fpr_target']),nearby={str(alpha):fit_threshold(benign,alpha) for alpha in cfg['evaluation']['nearby_fpr']},calibration_benign_documents=len(benign),calibration_source_groups=len({d['source_group_id'] for d in calibration if d['label']==0}),attack_rule='score > threshold',fit_policy=cfg['evaluation']['threshold_rule'])
   if role in thresholds['methods'] and thresholds['methods'][role]!=fitted:raise ValueError('Previously frozen final thresholds differ')
   thresholds['methods'][role]=fitted;save(thresholdpath,thresholds)
   for split in ('test','notinject'):
    _,complete,new=score_partition(OUT/role/split,documents[split],entry,frozen,cfg,remaining,args.chunk_documents,loaded);remaining-=new
    if not complete:print(json.dumps(dict(state='bounded_handoff',role=role,partition=split,new_documents=args.max_new_documents-remaining)));return 0
   del loaded
  pools={split:{role:cached_scores(OUT/role/split,documents[split],entry,frozen) for role,entry in frozen['finalists'].items()} for split in ('test','notinject')}
  save(OUT/'quality_summary.json',dict(freeze=frozen,test=summarize_pool(pools['test'],thresholds['methods'],cfg),notinject_separate=summarize_pool(pools['notinject'],thresholds['methods'],cfg),notinject_category_counts={category:sum(d.get('category')==category for d in documents['notinject']) for category in sorted({d.get('category') for d in documents['notinject']})},benign_test_target=3000,actual_distinct_benign_test=len({d['source_group_id'] for d in documents['test'] if d['label']==0}),no_test_tuning=True))
 if args.phase in ('challenge','all'):
  test=rows(ROOT/'data/prepared/test.jsonl');pool,description=challenge_pool(test,cfg)
  challengepath=OUT/'challenge_pool.json'
  if challengepath.exists() and read(challengepath)!=description:raise ValueError('Existing shared challenge pool differs')
  save(challengepath,description)
  selected={d['document_id'] for d in pool};scored={role:[r for r in cached_scores(OUT/role/'test',test,entry,frozen) if r['document_id'] in selected] for role,entry in frozen['finalists'].items()}
  summary=summarize_pool(scored,thresholds['methods'],cfg)
  summary['group_any_variant_passes']={role:sum(any(r['risk_score']<=thresholds['methods'][role]['primary'] for r in actual if r['source_group_id']==group['source_group_id']) for group in description['groups']) for role,actual in scored.items()}
  save(OUT/'challenge_summary.json',dict(pool=description,results=summary))
 if args.phase in ('extension','all'):
  if set(thresholds['methods'])!=set(ROLES):raise ValueError('Main final-calibration thresholds must already be frozen')
  pool,description=save_tableqa_pool(tableqa,cfg,frozen,args.freeze);scored={}
  for role,entry in frozen['finalists'].items():
   loaded=[];actual,complete,new=score_partition(OUT/role/'tableqa',pool,entry,frozen,cfg,remaining,args.chunk_documents,loaded);remaining-=new;del loaded
   if not complete:print(json.dumps(dict(state='bounded_handoff',role=role,partition='tableqa',new_documents=args.max_new_documents-remaining)));return 0
   scored[role]=actual
  save(OUT/'tableqa_summary.json',dict(extension=description,results=summarize_pool(scored,thresholds['methods'],cfg)))
 print(json.dumps(dict(state='requested_phase_complete',phase=args.phase,new_documents=args.max_new_documents-remaining,board_measurements=None)));return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except (RuntimeError,ValueError,KeyError,OSError) as error:print(str(error),file=sys.stderr);raise SystemExit(2)
