"""Goal 3 short CPU software profile; no final-test data or fake-INT4 timing.

Run only while the training/quantization worker is idle. The saved Goal 2 CUDA
pilot is reused when its checkpoint and input documents match this experiment.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gate_eval import load_config, window_document, score_documents
from train_detector import read_jsonl, save_json, save_jsonl, enrich, summarize


def check_pause(destination):
    if (destination.parent/'pause.request').exists():
        raise RuntimeError('Goal 3 pause requested; saved completed measurements remain available.')


def matched_documents(tokenizer, documents, config):
    selected = []
    for doc in documents:
        windows = window_document(tokenizer,doc['text'],doc['document_id'],doc['label'],doc['injection_spans'],config)
        if len(windows)==1:
            selected.append(doc)
        if len(selected)==40:
            return selected
    raise ValueError('Need 40 permitted single-window development documents for the short timing pilot.')


def cpu_pilot(model,tokenizer,documents,config,threads):
    import torch
    torch.set_num_threads(threads)
    model.eval()
    fields=('input_ids','attention_mask','token_type_ids')
    raw=[]
    with torch.inference_mode():
        for index,doc in enumerate(documents):
            start=time.perf_counter()
            windows=window_document(tokenizer,doc['text'],doc['document_id'],doc['label'],doc['injection_spans'],config)
            batch=tokenizer.pad([{key:w[key] for key in fields} for w in windows],padding='max_length',max_length=config['windowing']['primary_length'],return_tensors='pt')
            prepared=time.perf_counter()
            risk=torch.softmax(model(**batch).logits.float(),dim=-1)[:,1].max()
            inference_end=time.perf_counter()
            score=risk.item()
            end=time.perf_counter()
            if index>=8:
                raw.append({'document_id':doc['document_id'],'risk_score':score,
                    'preparation_ms':(prepared-start)*1000,'inference_and_aggregation_ms':(inference_end-prepared)*1000,
                    'host_to_device_ms':0.,'result_transfer_ms':(end-inference_end)*1000,'total_ms':(end-start)*1000})
    means={key:sum(row[key] for row in raw)/len(raw) for key in raw[0] if key.endswith('_ms')}
    return {'threads':threads,'inter_op_threads':torch.get_num_interop_threads(),'device':'CPU','batch_size':1,
            'padded_total_length':config['windowing']['primary_length'],'warmup_documents':8,'completed_measured_checks':len(raw),
            'means':means,'raw':raw,'scope':'Short warm development pilot only; no final workload or energy claim.'}


def source_quality(model,tokenizer,sets,config,destination):
    outputs={}
    for name,docs in sets.items():
        check_pause(destination.parent)
        rows_path=destination/(name+'_documents.jsonl')
        windows_path=destination/(name+'_windows.jsonl')
        if rows_path.exists() and windows_path.exists():
            outputs[name]={'documents':read_jsonl(rows_path),'windows':read_jsonl(windows_path)}
        else:
            started=time.perf_counter()
            outputs[name]=enrich(score_documents(model,tokenizer,docs,batch_size=1,device='cpu',config=config),docs)
            save_jsonl(rows_path,outputs[name]['documents'])
            save_jsonl(windows_path,outputs[name]['windows'])
            print(json.dumps({'event':'cpu_search_scored','format':destination.name,'split':name,'seconds':time.perf_counter()-started}),flush=True)
    metrics=summarize(outputs['search_temporary_threshold']['documents'],outputs['search_candidate_scoring']['documents'],config)
    save_json(destination/'metrics.json',metrics)
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-dynamic',action='store_true')
    args=parser.parse_args()
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    config=load_config()
    goal2=ROOT/config['paths']['results']/'goal2'
    selection=json.loads((goal2/'selection.json').read_text())
    checkpoint=ROOT/selection['checkpoint']
    destination=ROOT/config['paths']['results']/'goal3'/'software_profile'
    destination.mkdir(parents=True,exist_ok=True)
    check_pause(destination)
    if (destination/'profile.json').exists():
        old=json.loads((destination/'profile.json').read_text())
        if old.get('status')=='complete':
            print(json.dumps({'event':'completed_profile_reused','path':str(destination/'profile.json')}),flush=True)
            return
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,use_fast=True,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(checkpoint,local_files_only=True).float().cpu().eval()
    data=ROOT/config['paths']['data']/'prepared'
    sets={name:read_jsonl(data/(name+'.jsonl')) for name in ('search_temporary_threshold','search_candidate_scoring')}
    documents=matched_documents(tokenizer,sets['search_candidate_scoring'],config)
    profile={'recorded_utc':datetime.now(timezone.utc).isoformat(),'command':'python scripts/profile_software.py',
        'checkpoint':selection['checkpoint'],'config':config,'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,
        'reference_recall':selection['metrics']['primary']['candidate_scoring']['recall'],
        'thread_candidates':[1,4,8],'selection':'Lowest mean raw-text-to-decision latency among the three fixed CPU thread counts.',
        'quality_rule':'Own temporary threshold fitted to assigned benign search groups; candidate FPR <= 1% and recall loss <= 1 percentage point versus selected floating reference.',
        'formats':{}}
    def measure(name,current_model,description):
        check_pause(destination)
        results=[]
        for threads in profile['thread_candidates']:
            check_pause(destination)
            saved=destination/(name+f'_threads_{threads}.json')
            if saved.exists():
                timing=json.loads(saved.read_text())
            else:
                timing=cpu_pilot(current_model,tokenizer,documents,config,threads)
                save_json(saved,timing)
            results.append(timing)
            print(json.dumps({'event':'cpu_timing','format':name,'threads':threads,'means':timing['means']}),flush=True)
        best=min(results,key=lambda row:row['means']['total_ms'])
        torch.set_num_threads(best['threads'])
        metrics=source_quality(current_model,tokenizer,sets,config,destination/name)
        primary=metrics['primary']['candidate_scoring']
        feasible=(primary['fpr']<=config['evaluation']['fpr_target'] and primary['recall']>=profile['reference_recall']-config['evaluation']['max_recall_loss_absolute'])
        record={'description':description,'status':'measured','best_threads':best['threads'],'best_timing':best,
                'quality':metrics,'quality_window_batch_size':1,'meets_development_protection':feasible}
        profile['formats'][name]=record
        save_json(destination/'profile.json',profile)
        return record
    measure('float32',model,'CPU float32 checkpoint; selected encoder weights, embeddings, classifier, activations and nonlinear operations remain float32.')
    if not args.skip_dynamic:
        check_pause(destination)
        names=[name for name,module in model.named_modules() if name.startswith('bert.encoder.layer.') and isinstance(module,torch.nn.Linear)]
        if len(names)!=24:
            raise ValueError(f'Expected the fixed BERT-Mini 24 encoder linear modules, got {len(names)}.')
        supported=list(torch.backends.quantized.supported_engines)
        preferred=next((engine for engine in ('x86','fbgemm','onednn') if engine in supported),None)
        if preferred is None:
            profile['formats']['dynamic_qint8']={'status':'unsupported','supported_engines':supported}
        else:
            torch.backends.quantized.engine=preferred
            qconfig={name:torch.ao.quantization.default_dynamic_qconfig for name in names}
            quantized=torch.ao.quantization.quantize_dynamic(model,qconfig_spec=qconfig,dtype=torch.qint8,inplace=False).eval()
            quantized_modules={name:module for name,module in quantized.named_modules() if isinstance(module,torch.ao.nn.quantized.dynamic.Linear)}
            if set(quantized_modules)!=set(names):
                raise RuntimeError('Actual dynamic-int8 module scope differs from the declared encoder-linear scope.')
            record=measure('dynamic_qint8',quantized,
                'Actual PyTorch dynamic quantized CPU Linear kernels: selected 24 encoder linears use signed qint8 weights and runtime activation quantization. Embeddings/classifier/nonlinear operations stay float32. This is a different numerical policy from the static W8A8 FPGA reference and is not INT4.')
            record.update({'quantized_engine':preferred,'supported_engines':supported,
                'activation_quantization_wrapper_policy':'Installed PyTorch dynamic Linear version 4 calls quantized.linear_dynamic with reduce_range=True; activation quantization occurs per runtime input tensor.',
                'quantized_modules':[{'name':name,'weight_dtype':str(module.weight().dtype),'weight_shape':list(module.weight().shape),'weight_qscheme':str(module.weight().qscheme())} for name,module in quantized_modules.items()]})
    gpu=json.loads((goal2/'timing_pilot.json').read_text())
    measured_ids=[doc['document_id'] for doc in documents[8:]]
    gpu_ids=[row['document_id'] for row in gpu['raw']]
    if measured_ids==gpu_ids and gpu['padded_total_length']==config['windowing']['primary_length']:
        profile['cuda_float32_reused']={'source':'results/goal2/timing_pilot.json','same_selected_checkpoint':selection['checkpoint'],
            'same_input_ids_verified':True,'timing':gpu,'quality':selection['metrics'],
            'meets_development_protection':selection['metrics']['primary']['candidate_scoring']['fpr']<=config['evaluation']['fpr_target']}
    else:
        profile['cuda_float32_reused']={'status':'not_reused','reason':'Saved GPU timing inputs/window length differ; no new GPU benchmark was run.'}
    profile['status']='complete'
    save_json(destination/'profile.json',profile)
    print(json.dumps({'event':'complete','formats':{name:{'status':row['status'],'threads':row.get('best_threads'),'protection':row.get('meets_development_protection'),'total_ms':row.get('best_timing',{}).get('means',{}).get('total_ms')} for name,row in profile['formats'].items()}},indent=2),flush=True)


if __name__=='__main__':
    main()

