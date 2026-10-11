"""Frozen held-out EmailQA four-condition preparation and separately gated API.

Preview opens no held-out content. The completed120-call development pilot is
never renewed; new generation defaults to zero and needs a new explicit budget.
"""
from pathlib import Path
from copy import deepcopy
from collections import Counter
from datetime import datetime,timezone
import argparse,json,math,os,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_frozen_finalists import config,path,read,rows,save,prerequisites,require_ready,challenge_pool,cached_scores,OUT as QUALITY
import application_pilot as pilot
OUT=ROOT/'results/goal7/application'

def generation_prerequisites(cfg,authorization_path,max_new):
 """New Goal7 allowance only; never silently inherit the old120-call budget."""
 blockers=[];authorized=None
 if max_new<0:raise ValueError('max_new_authorized_calls must be nonnegative')
 if max_new==0:return dict(ready=False,blockers=['New generation allowance is zero'],max_new_authorized_calls=0),None
 if (ROOT/'results/goal7/pause.request').exists() or cfg.get('execution_control',{}).get('current_pause'):blockers.append('Current Goal7 execution pause is active')
 if cfg.get('execution_control',{}).get('board_free_continuation',{}).get('additional_target_api_calls_authorized') is not True:blockers.append('Current continuation does not authorize additional target API calls')
 if not authorization_path or not path(authorization_path).is_file():blockers.append('Separate Goal7 application authorization/budget file required')
 else:
  try:
   record=read(path(authorization_path));authorized=deepcopy(cfg)
   if record['scope']!='heldout_application_goal7' or record['authorized'] is not True or record.get('previous_pilot_allowance_reused') is not False:raise ValueError('Explicit independent Goal7 authorization required')
   if not isinstance(record.get('authorization_note'),str) or not record['authorization_note'].strip():raise ValueError('Actual authorization basis missing')
   if record['target']!=pilot.target_settings(cfg):raise ValueError('Authorization target/prompt/settings differ')
   app=authorized['application_pilot']
   for field in ('max_api_calls','max_input_tokens','max_output_tokens_total'):
    if type(record.get(field)) is not int or record[field]<=0:raise ValueError('Explicit positive '+field+' required')
    app[field]=record[field]
   if not isinstance(record.get('budget_usd'),(int,float)) or isinstance(record['budget_usd'],bool) or not math.isfinite(record['budget_usd']) or record['budget_usd']<0:raise ValueError('Explicit finite USD cap required')
   authorized['access']['paid_api_budget']={'currency':'USD','amount':record['budget_usd']};app['authorized']=True
   pilot.validate_billing(authorized)
  except (KeyError,ValueError,RuntimeError,TypeError,OSError) as error:blockers.append('Generation budget: '+str(error))
 return dict(ready=not blockers,blockers=blockers,max_new_authorized_calls=max_new),authorized

def heldout_pairs(test,cfg):
 pool,description=challenge_pool(test,cfg);bygroup={}
 for doc in test:
  if doc['label']==0:bygroup[doc['source_group_id']]=doc
 pairs=[]
 for group_index,group in enumerate(description['groups'],1):
  clean=bygroup[group['source_group_id']];records=sorted(clean.get('task_records',[]),key=lambda r:r['source_id'])
  if not records:records=[dict(source_id=clean['source_group_id'],question=clean.get('question'),reference_answer=clean.get('reference_answer'))]
  task=records[0];question=task['question'];reference=task['reference_answer']
  references={json.dumps(r['reference_answer'],sort_keys=True) for r in records if r['question']==question}
  ambiguous=len(references)!=1 or not isinstance(reference,str) or not reference.strip() or not isinstance(question,str) or not question.strip()
  for variant_index,attacked in enumerate((d for d in pool if d['source_group_id']==group['source_group_id']),1):
   pairs.append(dict(pair_id=f'emailqa-heldout-{group_index:03d}-{variant_index:02d}',source_group_id=group['source_group_id'],source_id=task['source_id'],question=question or '',reference_answer=reference if isinstance(reference,str) else '',ambiguous_reference=ambiguous,reference_candidates=[r['reference_answer'] for r in records if r['question']==question],natural_chunk_count=1,clean=clean,attacked=attacked))
 return pairs,description

def materialize(cases,responses,judgments):
 """Use pilot extraction/rubrics, retaining unknown held-out rubric outcomes."""
 outputs=[];claimed=set()
 for case in cases:
  row=dict(case);rid=case['target_request_id'];response=responses.get(rid);attacked=case['condition'].startswith('attacked')
  row.update(answer=None,target_status=None,legitimate_task_complete=None,attack_success=None,legitimate_task_match_proxy=None,attack_success_proxy=None,task_review_required=False,attack_review_required=False,guard_insufficient_evidence=False,first_visible_answer_seconds=None,target_full_response_seconds=None,target_response_reused_for_quality=False)
  if case['state']=='pending_guard':row['target_status']='pending_guard'
  elif case['state']=='guard_rejected':row.update(answer=pilot.INSUFFICIENT,target_status='not_called_guard_rejected',legitimate_task_complete=False,attack_success=False if attacked else None,guard_insufficient_evidence=True)
  elif response is None:row['target_status']='not_run'
  else:
   row.update(answer=response.get('answer',''),target_status=response['status'],provider=response.get('provider'),returned_model_label=response.get('model'),modelVersion=response.get('modelVersion'),provider_response_id=response.get('id'),responseId=response.get('responseId'),finishReason=response.get('finishReason'),system_fingerprint=response.get('system_fingerprint'),target_response_reused_for_quality=rid in claimed)
   if rid not in claimed:
    row['first_visible_answer_seconds']=response.get('first_visible_answer_seconds');row['target_full_response_seconds']=response.get('target_full_response_seconds');claimed.add(rid)
   if response['status']=='ok':row.update(pilot.legitimate_outcome(row['answer'],row['reference_answer']))
   else:row.update(legitimate_task_complete=False,task_judgment_method='Non-ok status, including truncation/timeout, is non-completion')
   if attacked and row['answer']:
    try:row.update(pilot.attack_outcome(row['answer'],row['attack_template_id']))
    except ValueError:row.update(attack_success=None,attack_success_proxy=None,attack_review_required=True,attack_judgment_method='Original held-out payload retained for fixed human-rubric review; no guessed lexical success')
   judgment=judgments.get(rid)
   if judgment:
    response_id=response.get('responseId') or response.get('id')
    if judgment.get('response_id')!=response_id:raise ValueError('Judgment must match the actual saved provider response ID')
    for field in ('legitimate_task_complete','attack_success'):
     if field not in judgment:continue
     if type(judgment[field]) is not bool:raise ValueError('Adjudicated outcomes must be booleans')
     if field=='legitimate_task_complete' and response['status']=='ok' and not row['ambiguous_reference']:row[field]=judgment[field];row['task_review_required']=False
     elif field=='attack_success' and attacked:row[field]=judgment[field];row['attack_review_required']=False
    row['manual_judgment_note']=judgment.get('note')
   if row['ambiguous_reference']:
    if response['status']=='ok':row['legitimate_task_complete']=None
    row['task_review_required']=True;row['task_judgment_method']='Ambiguous/missing source reference requires resolution; no automatic task success'
  outputs.append(row)
 return outputs

def summary(outputs,responses,events,cfg):
 conditions={}
 for role in ('U','B','C'):
  conditions[role]={}
  for condition in pilot.CONDITIONS:
   current=[r for r in outputs if r['method']==role and r['condition']==condition];attacked=condition.startswith('attacked');n=len(current);complete=sum(r['legitimate_task_complete'] is True for r in current);pending=sum(r['legitimate_task_complete'] is None for r in current);success=sum(r['attack_success'] is True for r in current);unknown=sum(r['attack_success'] is None for r in current) if attacked else 0
   conditions[role][condition]=dict(tasks=n,source_groups=len({r['source_group_id'] for r in current}),legitimate_task_complete_count=complete,legitimate_task_outcomes_pending=pending,legitimate_task_completion_rate=complete/n if n and not pending else None,attack_success_count=success if attacked else None,attack_outcomes_pending=unknown if attacked else None,attack_success_rate=success/n if attacked and n and not unknown else None,guard_rejected=sum(r['rejected'] is True for r in current),guard_insufficient_evidence_responses=sum(r['guard_insufficient_evidence'] for r in current),ambiguous_references=sum(r['ambiguous_reference'] for r in current),target_status_counts=dict(Counter(r['target_status'] for r in current)),actual_natural_chunk_counts=dict(Counter(str(r['natural_chunk_count']) for r in current)))
 document_effects={}
 for role in ('U','B','C'):
  unique={r['document_id']:r for r in outputs if r['method']==role and r['guard_applied']}
  ready=[r for r in unique.values() if r['rejected'] is not None]
  document_effects[role]=dict(documents=len(unique),completed_document_checks=len(ready),pending_document_checks=len(unique)-len(ready),source_groups=len({r['source_group_id'] for r in unique.values()}),benign_rejected=sum(not r['condition'].startswith('attacked') and r['rejected'] is True for r in ready),attacks_rejected=sum(r['condition'].startswith('attacked') and r['rejected'] is True for r in ready),benign_documents=sum(not r['condition'].startswith('attacked') for r in unique.values()),attack_documents=sum(r['condition'].startswith('attacked') for r in unique.values()))
 successful=[r for r in responses.values() if r['status']=='ok'];timed=[r for r in successful if r.get('target_full_response_seconds') is not None];seconds=sum(r['target_full_response_seconds'] for r in timed)
 return dict(recorded_utc=pilot.now(),conditions=conditions,document_guard_effects=document_effects,logical_tasks=len(outputs),source_groups=len({r['source_group_id'] for r in outputs}),distinct_saved_responses=len(responses),actual_api_ledger=pilot.ledger_state(events,pilot.unit_prices(cfg)),mean_first_visible_answer_seconds=sum(r['first_visible_answer_seconds'] for r in timed if r.get('first_visible_answer_seconds') is not None)/sum(r.get('first_visible_answer_seconds') is not None for r in timed) if any(r.get('first_visible_answer_seconds') is not None for r in timed) else None,mean_completed_full_response_seconds=seconds/len(timed) if timed else None,serial_completed_responses_per_sum_target_response_seconds=len(timed)/seconds if seconds else None,end_to_end_throughput=None,timing_scope='Distinct actual API responses only; cached quality reuse is not new latency evidence; no guard/board/end-to-end performance claimed',quality_reuse_only=True,model_identity_stop=pilot.response_identity_stop(responses,pilot.target_settings(cfg)))

def main():
 ap=argparse.ArgumentParser(description=__doc__);actions=ap.add_mutually_exclusive_group();actions.add_argument('--prepare',action='store_true');actions.add_argument('--execute',action='store_true');ap.add_argument('--check-prerequisites',action='store_true');ap.add_argument('--freeze',type=Path);ap.add_argument('--authorization',type=Path);ap.add_argument('--max-new-authorized-calls',type=int,default=0);ap.add_argument('--key-env');args=ap.parse_args();cfg=config();preview,frozen=prerequisites(cfg,args.freeze);generation,authorized=generation_prerequisites(cfg,args.authorization,args.max_new_authorized_calls)
 preview.update(prepare=args.prepare,execute=args.execute,generation=generation,old_development_pilot_allowance_reused=False,final_thresholds_exists=(QUALITY/'thresholds.json').is_file(),output=str(OUT),heldout_source=str(ROOT/'data/prepared/test.jsonl'))
 if not args.prepare and not args.execute:print(json.dumps(preview,indent=2));return 0 if not args.check_prerequisites or preview['ready'] else 2
 frozen=require_ready(cfg,args.freeze)
 if args.execute and args.max_new_authorized_calls and not generation['ready']:raise RuntimeError('No credential read/API call: '+'; '.join(generation['blockers']))
 # Freeze/readiness/identity gates precede the first held-out content read.
 test=rows(ROOT/'data/prepared/test.jsonl')
 if any(d['split']!='test' for d in test):raise ValueError('Application requires actual held-out test documents')
 pairs,pool=heldout_pairs(test,cfg);thresholdpath=QUALITY/'thresholds.json';thresholds=read(thresholdpath) if thresholdpath.exists() else None
 if thresholds is not None and thresholds['freeze']!=frozen:raise ValueError('Application final threshold freeze differs')
 requests={};allcases=[];settings=pilot.target_settings(cfg)
 for role in ('U','B','C'):
  entry=frozen['finalists'][role];scorefolder=QUALITY/role/'test';scores={};threshold=None
  if thresholds and role in thresholds['methods']:
   threshold={'threshold':thresholds['methods'][role]['primary']}
   if (scorefolder/'progress.json').is_file() and read(scorefolder/'progress.json')['complete']:scores={r['document_id']:r for r in cached_scores(scorefolder,test,entry,frozen)}
  _,planned,cases=pilot.prepare_plan(cfg,pairs,scores,threshold)
  for request in planned:
   if request['request_id'] in requests and requests[request['request_id']]['body']!=request['body']:raise ValueError('Shared target request ID has different content')
   requests[request['request_id']]=request
  pairmap={p['pair_id']:p for p in pairs}
  for case in cases:
   case.pop('text_preparation_seconds',None)
   pair=pairmap[case['pair_id']];case.update(method=role,case_id=role+':'+case['case_id'],ambiguous_reference=pair['ambiguous_reference'],reference_candidates=pair['reference_candidates'],guard_identity=entry['identity'],guard_score_scope='Complete cached document max, frozen final-calibration threshold; not timing evidence')
  allcases.extend(cases)
 needed={case['target_request_id'] for case in allcases if case['state']=='target_required'};requests=[value for key,value in requests.items() if key in needed]
 snapshot=dict(freeze=frozen,target=settings,pool=pool,pairs=pairs,requests=requests,previous_pilot_allowance_reused=False)
 planpath=OUT/'request_plan.json'
 if planpath.exists() and read(planpath)!=snapshot:raise ValueError('Preserve existing held-out plan/answers; current plan differs')
 save(planpath,snapshot)
 casepath=OUT/'case_plan.json'
 if casepath.exists():
  previous=read(casepath)
  if len(previous)!=len(allcases):raise ValueError('Saved logical case membership differs')
  for old,new in zip(previous,allcases):
   if old['case_id']!=new['case_id'] or (old['state']!='pending_guard' and old!=new):raise ValueError('Previously resolved frozen guard case changed')
 save(casepath,allcases)
 events=pilot.read_jsonl(OUT/'api_events.jsonl');saved=pilot.read_jsonl(OUT/'target_responses.jsonl');responses=pilot.recover_responses(events,saved)
 for rid,response in responses.items():
  if rid not in {r['request_id'] for r in requests}:raise ValueError('Saved response is not assigned to the frozen held-out plan')
 judgments={r['request_id']:r for r in pilot.read_jsonl(OUT/'manual_judgments.jsonl')};calls=0;stop=None
 if args.execute and args.max_new_authorized_calls:
  if any(case['state']=='pending_guard' for case in allcases):raise RuntimeError('Complete frozen guard document checks required before releasing any document')
  stop=pilot.response_identity_stop(responses,settings) or pilot.dispatch_stop_reason(cfg,OUT)
  prior=next((event for event in reversed(events) if event['event']=='result'),None)
  stop=stop or (pilot.request_stop_reason(prior) if prior else None)
  unresolved=pilot.ledger_state(events,pilot.unit_prices(authorized))['unresolved_dispatches']
  if unresolved:stop=stop or 'Unresolved prior dispatch requires review; never charge it as free or automatically retry'
  if not stop:
   key_env=args.key_env or cfg['models']['target'].get('credential_environment_variable') or 'IIIA_API_KEY'
   budgetpath=OUT/'authorization.json';budget=read(path(args.authorization))
   if budgetpath.exists() and read(budgetpath)!=budget:raise ValueError('Existing held-out allowance differs; preserve original budget and actual ledger')
   save(budgetpath,budget)
   secret=pilot.authorize(authorized,settings,allcases,None,key_env,None,OUT)
   try:
    for request in requests:
     if request['request_id'] in responses:continue
     if calls>=args.max_new_authorized_calls:stop='Per-invocation new-call cap reached; original Goal7 ledger remains charged';break
     livegate,_=generation_prerequisites(config(),args.authorization,1)
     stop=pilot.dispatch_stop_reason(cfg,OUT) or ('; '.join(livegate['blockers']) if not livegate['ready'] else None)
     if read(path(args.authorization))!=budget:stop='Explicit Goal7 allowance changed during dispatch; stop for review'
     if stop:break
     pilot.enforce_allowance(authorized,request,events)
     dispatch=dict(event='dispatch',attempt_id=f"attempt-{pilot.ledger_state(events,pilot.unit_prices(authorized))['api_attempts']+1:04d}",request_id=request['request_id'],timestamp_utc=pilot.now(),model=settings['model'],provider=settings['provider'],endpoint=settings['endpoint'],reserved_input_tokens=request['reserved_input_tokens'],reserved_output_tokens=request['reserved_output_tokens'],billing_tier=authorized['application_pilot']['billing_tier'])
     pilot.append_jsonl(OUT/'api_events.jsonl',dispatch);events.append(dispatch);calls+=1
     result=pilot.stream_request(request,settings,secret,authorized['application_pilot'].get('request_timeout_seconds',90));result.update(request_id=request['request_id'],attempt_id=dispatch['attempt_id'],recorded_utc=pilot.now(),requested_model=settings['model'])
     event=dict(event='result',**result);pilot.append_jsonl(OUT/'api_events.jsonl',event);events.append(event);pilot.append_jsonl(OUT/'target_responses.jsonl',result);responses[request['request_id']]=result
     stop=pilot.request_stop_reason(result) or pilot.response_identity_stop(responses,settings)
     if stop:break
     usage=pilot.ledger_state(events,pilot.unit_prices(authorized));app=authorized['application_pilot']
     if usage['charged_input_tokens_actual_or_reserved']>app['max_input_tokens'] or usage['charged_output_tokens_actual_or_reserved']>app['max_output_tokens_total'] or usage['cost_usd_actual_or_reserved']>pilot.budget_amount(authorized):stop='Provider usage exceeded explicit new allowance; stop for review';break
     delay=authorized['application_pilot'].get('min_request_interval_seconds',0)
     if delay:time.sleep(min(float(delay),60))
   finally:del secret
 outputs=materialize(allcases,responses,judgments);save(OUT/'outcomes.json',outputs);result=summary(outputs,responses,events,authorized or cfg);result.update(new_calls_this_invocation=calls,max_new_authorized_calls=args.max_new_authorized_calls,stop_reason=stop,prepared_only=args.prepare,freeze=frozen,complete_four_condition_execution=all(r['target_status'] not in ('pending_guard','not_run') for r in outputs));save(OUT/'summary.json',result)
 print(json.dumps(dict(prepared=True,source_groups=result['source_groups'],pairs=len(pairs),logical_tasks=len(allcases),guard_checks_ready=not any(c['state']=='pending_guard' for c in allcases),new_calls=calls,complete=result['complete_four_condition_execution'],stop_reason=stop)));return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except (RuntimeError,ValueError,KeyError,OSError) as error:print(str(error),file=sys.stderr);raise SystemExit(2)
