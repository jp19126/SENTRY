"""Bounded frozen references and unchanged application guard scoring.

No fitting, target API calls, remote code, protected partitions or hardware.
Historical BERT/CNN development scores are reused with identity checks.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gate_eval import load_config,window_document,fit_threshold,evaluate_documents
from r1_challengers import json_write,rows,trained_model,execution_settings
from span_train import Budget,setup,load_split,batch,load_checkpoint

STUDY=ROOT/'results/span_coupled_pilot_v1'
PREPARED=STUDY/'prepared'
SPLITS=('search_temporary_threshold','search_candidate_scoring','stress_probes')
ALIASES=dict(zip(SPLITS,('calibration','natural','stress')))
SAVED={'bert':'bert_cuda_graph_score_v1','cnn':'score_cnn_cuda_v1'}
REVISION='dd78b24e330193a22d2293ac66922dd4f982f563'


def write_rows(path,documents):
    with path.open('x',encoding='utf-8') as stream:
        for d in documents:stream.write(json.dumps(d,ensure_ascii=False)+'\n')


def pilot_documents():
    # Request IDs differ from R1 IDs; deduplicate EXACT external document text.
    unique={}
    for pair in rows(ROOT/'data/prepared/development_pairs.jsonl'):
        for side in ('clean','attacked'):
            doc=pair[side];key=doc['text']
            if key not in unique:
                unique[key]=dict(doc,application_side=side,application_document_ids=[],application_pair_ids=[])
            record=unique[key]
            if record['label']!=doc['label']:raise ValueError('Inconsistent pilot text labels')
            record['application_document_ids'].append(doc['document_id'])
            record['application_pair_ids'].append(pair['pair_id'])
    output=list(unique.values())
    if len(output)!=120 or sum(d['label']==0 for d in output)!=20:raise ValueError('Unchanged pilot population differs')
    return output


def metadata_only(doc):
    excluded={'text','context','original_context','question','reference_answer','task_records'}
    return {k:v for k,v in doc.items() if k not in excluded}


def local_piguard(folder,device,torch):
    from safetensors.torch import load_file
    from transformers import AutoTokenizer,DebertaV2Config,DebertaV2ForSequenceClassification
    retrieval=json.loads((folder/'retrieval.json').read_text(encoding='utf-8'))
    if not retrieval['complete'] or retrieval['revision']!=REVISION:raise ValueError('Pinned completed retrieval required')
    for name,record in retrieval['files'].items():
        file=folder/name
        if file.stat().st_size!=record['bytes']:raise ValueError('Reference size changed')
        digest=hashlib.sha256()
        with file.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024**2),b''):digest.update(chunk)
        if digest.hexdigest()!=record['sha256']:raise ValueError('Reference identity changed')
    config=DebertaV2Config.from_pretrained(folder,local_files_only=True)
    if config.num_labels!=2:raise ValueError('Expected binary PIGuard head')
    base=DebertaV2ForSequenceClassification(config)
    # Upstream replaces the classifier but retains the superclass pooler in
    # the state dict. Load that unused module too; never silently discard keys.
    base.classifier=torch.nn.Linear(config.hidden_size,2)
    weights=load_file(str(folder/'model.safetensors'),device='cpu')
    base.load_state_dict(weights,strict=True)
    class PIGuard(torch.nn.Module):
        def __init__(self,base):
            super().__init__();self.base=base
        def forward(self,input_ids,attention_mask,**unused):
            hidden=self.base.deberta(input_ids=input_ids,attention_mask=attention_mask,return_dict=True).last_hidden_state
            return SimpleNamespace(logits=self.base.classifier(hidden[:,0]))
    model=PIGuard(base).float().eval().to(device)
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False,use_fast=True)
    return model,tok,dict(repository='leolee99/PIGuard',revision=REVISION,
        forward='Installed DebertaV2Model last_hidden_state[:,0] followed by loaded classifier; eval dropout identity',
        strict_state_dict=True,loaded_keys=len(weights),unused_pooler_loaded=True,
        remote_code_executed=False,labels={'0':'benign','1':'injection'},
        pretrained_exposure='External guard training provenance from model card; overlap with this development population is not ruled out',
        files=retrieval['files'])


def reference(args,torch,budget):
    config=load_config();budget.check()
    if args.baseline=='piguard':
        return local_piguard(args.reference,args.device,torch)
    from r1_challengers import tokenizer
    tok=tokenizer(config)
    if args.baseline=='cnn':
        model=trained_model('cnn',ROOT/'results/goal_r1/cnn_initial_v1',ROOT/'results/goal_r1/prepared_v1',config,args.device,torch)
        return model,tok,dict(checkpoint='results/goal_r1/cnn_initial_v1/checkpoint_latest.pt',epoch=10,arithmetic='FP32 batch1 historical route')
    if args.device!='cuda':raise ValueError('Historical BERT graph requires CUDA')
    from r1_bert_cuda_graph import GraphBackend,require_graph_bringup,metadata
    from r1_bert_int8 import load_model,require_bringup
    require_graph_bringup(ROOT/'results/goal_r1/bert_cuda_graph_bringup_v1')
    require_bringup(ROOT/'results/goal_r1/bert_int8_cuda_bringup_v1','cuda')
    model,_,modules=load_model('cuda',torch)
    data,_=load_split(PREPARED,SPLITS[1]);inp=batch(data,[0],'cuda',torch)
    first={k:inp[k] for k in ('input_ids','attention_mask')}
    graph=GraphBackend(model,first,torch)
    class Adapter:
        def __call__(self,**inputs):
            graph.copy_inputs(inputs);logits,_=graph.replay()
            return SimpleNamespace(logits=logits)
    return Adapter(),tok,metadata(model,modules,graph,torch)


def infer_windows(model,inputs,torch):
    logits=model(**inputs).logits.float()
    if not bool(torch.isfinite(logits).all()):raise ValueError('Nonfinite reference logits')
    return torch.softmax(logits,-1)[:,1].detach().cpu().tolist()


def prepared_score(model,split,device,torch,budget,microbatch):
    data,documents=load_split(PREPARED,split);scores=[]
    with torch.inference_mode():
        for start in range(0,len(data['ids']),microbatch):
            budget.check();inp=batch(data,list(range(start,min(start+microbatch,len(data['ids'])))),device,torch)
            scores.extend(infer_windows(model,{k:inp[k] for k in ('input_ids','attention_mask')},torch))
    result=[]
    for i,d in enumerate(documents):
        first,end=map(int,data['starts'][i:i+2]);values=scores[first:end]
        result.append(dict(d,risk_score=max(values),window_scores=values))
    return result


def raw_score(model,tok,documents,device,torch,budget,microbatch):
    result=[];config=load_config()
    with torch.inference_mode():
        for i,d in enumerate(documents):
            budget.check();windows=window_document(tok,d['text'],d['document_id'],config=config);values=[]
            for at in range(0,len(windows),microbatch):
                encoded=tok.pad([{k:w[k] for k in ('input_ids','attention_mask')} for w in windows[at:at+microbatch]],
                    padding='max_length',max_length=256,return_tensors='pt')
                values.extend(infer_windows(model,{k:v.to(device) for k,v in encoded.items()},torch))
            result.append(dict(metadata_only(d),risk_score=max(values),window_scores=values,window_count=len(values),
                               content_token_count=sum(w['content_token_count'] for w in windows)-64*(len(windows)-1)))
            if i%128==0:print(json.dumps(dict(event='reference_progress',documents=i+1)),flush=True)
    return result


def saved_scores(name,split,model,device,torch,budget):
    file=ROOT/'results/goal_r1'/SAVED[name]/(split+'_documents.jsonl')
    saved=list(rows(file));data,docs=load_split(PREPARED,split)
    if len(saved)!=len(docs):raise ValueError('Historical document count differs')
    for a,b in zip(saved,docs):
        if any(a[k]!=b[k] for k in ('document_id','label','source_group_id','window_count')):raise ValueError('Historical identity mismatch')
        if len(a['window_scores'])!=a['window_count'] or max(a['window_scores'])!=a['risk_score']:raise ValueError('Historical aggregation mismatch')
    # One short and one full window, selected without score inspection; exact risk.
    selected={}
    for w,mask in enumerate(data['masks']):
        kind='full' if int(mask.sum())==256 else 'short'
        if kind not in selected:selected[kind]=w
        if len(selected)==2:break
    with torch.inference_mode():
        for w in selected.values():
            budget.check();inp=batch(data,[w],device,torch)
            risk=infer_windows(model,{k:inp[k] for k in ('input_ids','attention_mask')},torch)[0]
            d=int(data['doc_indices'][w]);local=w-int(data['starts'][d])
            if risk!=saved[d]['window_scores'][local]:raise AssertionError('Frozen route differs from historical score')
    return saved,dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),exact_recheck_windows=list(selected.values()))


def metrics(outputs):
    thresholds={str(a):fit_threshold([d['risk_score'] for d in outputs[SPLITS[0]] if d['label']==0],a) for a in (.005,.01,.02)}
    result={split:dict(documents=len(docs),windows=sum(len(d['window_scores']) for d in docs),
        metrics={a:evaluate_documents([d['risk_score'] for d in docs],[d['label'] for d in docs],t) for a,t in thresholds.items()}) for split,docs in outputs.items()}
    return thresholds,result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True,choices=('bert','cnn','piguard','conjunction','candidate_pilot'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda',choices=('cpu','cuda'))
    p.add_argument('--reference',type=Path,default=STUDY/'public_reference')
    p.add_argument('--bert-results',type=Path);p.add_argument('--cnn-results',type=Path)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--export',type=Path)
    p.add_argument('--microbatch',type=int,default=8);p.add_argument('--max-seconds',type=int,default=1200)
    args=p.parse_args();folder=args.output.resolve()
    if not folder.is_relative_to(STUDY.resolve()) or folder.exists():raise ValueError('Fresh study output required')
    if not 0<args.max_seconds<=1200 or not 1<=args.microbatch<=8:raise ValueError('Finite dispatch bound')
    folder.mkdir(parents=True);budget=Budget(args.max_seconds,folder)
    try:
        if args.baseline=='conjunction':
            conjunction(args,folder);budget.save();return
        torch=setup(42,args.device)
        if args.baseline=='candidate_pilot':
            candidate_pilot(args,folder,budget,torch);return
        model,tok,identity=reference(args,torch,budget);outputs={};reuse={}
        microbatch=1 if args.baseline in SAVED else args.microbatch
        for split in SPLITS:
            if args.baseline in SAVED and split in SPLITS[:2]:
                outputs[split],reuse[split]=saved_scores(args.baseline,split,model,args.device,torch,budget)
            elif args.baseline in SAVED:
                outputs[split]=prepared_score(model,split,args.device,torch,budget,microbatch)
            else:
                raw=PREPARED/'stress_probes_constructed.jsonl' if split=='stress_probes' else ROOT/'data/prepared'/(split+'.jsonl')
                documents=list(rows(raw));_,expected=load_split(PREPARED,split)
                if [(d['document_id'],d['label']) for d in documents]!=[(d['document_id'],d['label']) for d in expected]:raise ValueError('Raw/metadata identity mismatch')
                outputs[split]=raw_score(model,tok,documents,args.device,torch,budget,microbatch)
            write_rows(folder/(ALIASES[split]+'_documents.jsonl'),outputs[split])
            write_rows(folder/(split+'_documents.jsonl'),outputs[split])
        outputs['application_pilot']=raw_score(model,tok,pilot_documents(),args.device,torch,budget,microbatch)
        write_rows(folder/'application_pilot_documents.jsonl',outputs['application_pilot'])
        thresholds,results=metrics(outputs)
        if {d['source_group_id'] for d in outputs[SPLITS[0]]}&{d['source_group_id'] for s in (SPLITS[1],SPLITS[2],'application_pilot') for d in outputs[s]}:raise ValueError('Calibration leakage')
        json_write(folder/'metrics.json',dict(complete=True,baseline=args.baseline,thresholds=thresholds,results=results,
            identity=identity,reused=reuse,execution=execution_settings(torch),microbatch=microbatch,command=sys.argv,
            pretrained_exposure='PIGuard external training overlap unresolved; BERT/CNN original templates previously exposed',
            application='Guard inference only on unchanged120 texts; no target calls or claims about missing responses'))
        json_write(folder/'bundle.json',dict(complete=True,variants=['.'],baseline=args.baseline,shared_budget_seconds=args.max_seconds))
        budget.save()
    except BaseException as error:
        budget.save(str(error));json_write(folder/'failure.json',dict(complete=False,error=repr(error)));raise


def conjunction(args,folder):
    if not args.bert_results or not args.cnn_results:raise ValueError('Both completed reference bundles required')
    summaries=[json.loads((p/'metrics.json').read_text(encoding='utf-8')) for p in (args.bert_results,args.cnn_results)]
    if not all(s['complete'] for s in summaries):raise ValueError('Incomplete reference')
    results={}
    for name in ('calibration','natural','stress','application_pilot'):
        left,right=(list(rows(p/(name+'_documents.jsonl'))) for p in (args.bert_results,args.cnn_results))
        if [(d['document_id'],d['label']) for d in left]!=[(d['document_id'],d['label']) for d in right]:raise ValueError('Unpaired conjunction')
        records=[]
        for a,b in zip(left,right):
            decisions={alpha:bool(a['risk_score']>summaries[0]['thresholds'][alpha] and b['risk_score']>summaries[1]['thresholds'][alpha]) for alpha in summaries[0]['thresholds']}
            records.append(dict(metadata_only(a),bert_risk=a['risk_score'],cnn_risk=b['risk_score'],attack_decisions=decisions))
        write_rows(folder/(name+'_documents.jsonl'),records)
        results[name]=dict(documents=len(records),metrics={alpha:evaluate_documents([float(d['attack_decisions'][alpha]) for d in records],[d['label'] for d in records],.5) for alpha in summaries[0]['thresholds']})
    json_write(folder/'metrics.json',dict(complete=True,baseline='conjunction',thresholds={key:s['thresholds'] for key,s in zip(('bert','cnn'),summaries)},results=results,
        interpretation='Each component uses its own benign calibration threshold; AND decisions. Exploratory conventional comparator, no validated cascade latency claim.'))


def candidate_pilot(args,folder,budget,torch):
    from r1_challengers import tokenizer
    from span_numerics import IntegerSpan
    if bool(args.checkpoint)==bool(args.export):raise ValueError('Choose exactly checkpoint or export')
    model=IntegerSpan(args.export,args.device) if args.export else load_checkpoint(args.checkpoint,args.device,torch)[0]
    tok=tokenizer(load_config());results=[]
    with torch.inference_mode():
        for d in pilot_documents():
            budget.check();windows=window_document(tok,d['text'],d['document_id'],config=load_config());scores=[]
            length=windows[-1]['content_token_end']
            for at in range(0,len(windows),args.microbatch):
                selected=windows[at:at+args.microbatch]
                encoded=tok.pad([{k:w[k] for k in ('input_ids','attention_mask')} for w in selected],padding='max_length',max_length=256,return_tensors='pt')
                inp={k:v.to(args.device) for k,v in encoded.items()}
                inp.update(content_start=torch.tensor([w['content_token_start'] for w in selected],device=args.device),document_content_length=torch.full((len(selected),),length,device=args.device))
                output=model(**inp);risk=output['risk'] if args.export else output.risk;scores.extend(risk.cpu().tolist())
            results.append(dict(metadata_only(d),risk_score=max(scores),window_scores=scores,window_count=len(scores)))
    write_rows(folder/'application_pilot_documents.jsonl',results)
    json_write(folder/'metrics.json',dict(complete=True,checkpoint=str(args.checkpoint),export=str(args.export),documents=120,calibration='Reuse matching arithmetic scoring-bundle thresholds; none fitted here',target_calls=0))
    budget.save()


if __name__=='__main__':main()
