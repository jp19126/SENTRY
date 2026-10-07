"""Goal 2: budgeted floating BERT-Mini training, search selection and short timing.

Only train/search/development-pair files are opened. No final-calibration or test
records are read. Create results/goal2/pause.request to checkpoint and pause.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gate_eval import load_config, window_document, score_documents, fit_threshold, evaluate_documents


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def save_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(value, ensure_ascii=False) + '\n' for value in values), encoding='utf-8')


def mean_doc_loss(rows):
    return sum(-math.log(max(1e-12, min(1-1e-12, row['risk_score'] if row['label'] else 1-row['risk_score']))) for row in rows) / len(rows)


def enrich(scored, documents):
    source = {d['document_id']: d for d in documents}
    for row in scored['documents']:
        original = source[row['document_id']]
        for key in ('source_group_id', 'source_ids', 'source_dataset', 'attack_type', 'attack_template_id', 'injection_position'):
            row[key] = original.get(key)
    return scored


def summarize(threshold_rows, scoring_rows, config):
    benign = [r['risk_score'] for r in threshold_rows if r['label'] == 0]
    operating = {}
    for alpha in [config['evaluation']['fpr_target']] + config['evaluation']['nearby_fpr']:
        threshold = fit_threshold(benign, alpha)
        metrics = evaluate_documents([r['risk_score'] for r in scoring_rows], [r['label'] for r in scoring_rows], threshold)
        calibration = evaluate_documents(benign, [0] * len(benign), threshold)
        operating[str(alpha)] = {'threshold': threshold, 'temporary_threshold_benign': calibration, 'candidate_scoring': metrics}
    primary = operating[str(config['evaluation']['fpr_target'])]
    threshold = primary['threshold']
    strata = {}
    for name, rows in [('single_window', [r for r in scoring_rows if not r['is_long_document']]),
                       ('long_document', [r for r in scoring_rows if r['is_long_document']])]:
        strata[name] = evaluate_documents([r['risk_score'] for r in rows], [r['label'] for r in rows], threshold)
    for source in sorted({r['source_dataset'] for r in scoring_rows}):
        rows = [r for r in scoring_rows if r['source_dataset'] == source]
        strata['source:' + source] = evaluate_documents([r['risk_score'] for r in rows], [r['label'] for r in rows], threshold)
    ordinary = evaluate_documents([r['risk_score'] for r in scoring_rows], [r['label'] for r in scoring_rows], .5)
    return {'primary': primary, 'operating_points': operating, 'strata': strata,
            'candidate_mean_document_bce': mean_doc_loss(scoring_rows), 'ordinary_threshold_0_5': ordinary}


def infer_search(model, tokenizer, sets, config, destination, batch_size):
    outputs = {}
    for name, docs in sets.items():
        started = time.perf_counter()
        if (destination / (name + '_documents.jsonl')).exists() and (destination / (name + '_windows.jsonl')).exists():
            outputs[name] = enrich({'documents':read_jsonl(destination / (name + '_documents.jsonl')), 'windows':read_jsonl(destination / (name + '_windows.jsonl'))},docs)
        else:
            outputs[name] = enrich(score_documents(model, tokenizer, docs, batch_size=batch_size, device='cuda', config=config), docs)
        save_jsonl(destination / (name + '_documents.jsonl'), outputs[name]['documents'])
        save_jsonl(destination / (name + '_windows.jsonl'), outputs[name]['windows'])
        print(json.dumps({'event': 'search_scored', 'split': name, 'documents': len(docs), 'seconds': time.perf_counter()-started}), flush=True)
    metrics = summarize(outputs['search_temporary_threshold']['documents'], outputs['search_candidate_scoring']['documents'], config)
    save_json(destination / 'metrics.json', metrics)
    return metrics


def checkpoint(model, tokenizer, optimizer, folder, state):
    import torch
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    state.update({'optimizer': optimizer.state_dict(), 'python_rng': random.getstate(),
                  'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()})
    torch.save(state, folder / 'training_state.pt')


def timing_pilot(model, tokenizer, docs, config):
    import torch
    model.eval()
    model_fields = ('input_ids', 'attention_mask', 'token_type_ids')
    chosen = []
    for doc in docs:
        if len(window_document(tokenizer, doc['text'], doc['document_id'], doc['label'], doc['injection_spans'], config)) == 1:
            chosen.append(doc)
        if len(chosen) == 40:
            break
    measured = []
    with torch.inference_mode():
        for index, doc in enumerate(chosen):
            torch.cuda.synchronize()
            total_start = time.perf_counter()
            windows = window_document(tokenizer, doc['text'], doc['document_id'], doc['label'], doc['injection_spans'], config)
            batch = tokenizer.pad([{key: w[key] for key in model_fields} for w in windows], padding='max_length', max_length=config['windowing']['primary_length'], return_tensors='pt')
            preparation_end = time.perf_counter()
            batch = {key: value.to('cuda') for key, value in batch.items()}
            torch.cuda.synchronize()
            transfer_end = time.perf_counter()
            risk = torch.softmax(model(**batch).logits.float(), dim=-1)[:,1].max()
            torch.cuda.synchronize()
            inference_end = time.perf_counter()
            score = risk.item()
            torch.cuda.synchronize()
            total_end = time.perf_counter()
            if index >= 8:
                measured.append({'document_id':doc['document_id'], 'risk_score':score,
                    'preparation_ms':(preparation_end-total_start)*1000, 'host_to_device_ms':(transfer_end-preparation_end)*1000,
                    'inference_and_aggregation_ms':(inference_end-transfer_end)*1000,
                    'result_transfer_ms':(total_end-inference_end)*1000, 'total_ms':(total_end-total_start)*1000})
    averages = {key:sum(row[key] for row in measured)/len(measured) for key in measured[0] if key.endswith('_ms')}
    return {'scope':'Short Goal 2 single-window GPU pilot; not the final repeated workload benchmark.',
            'batch_size':1,'padded_total_length':config['windowing']['primary_length'],'warmup_documents':8,
            'completed_measured_checks':len(measured),'synchronization':'CUDA synchronized at boundaries', 'means':averages,'raw':measured}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--microbatch-windows', type=int, default=32)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    config = load_config()
    training = config['training']
    if not 1 <= args.microbatch_windows <= 32:
        raise ValueError('Use a measured microbatch of 1..32 windows; effective document batch comes from shared config.')
    if not torch.cuda.is_available():
        raise RuntimeError('Authorized CUDA training device is unavailable.')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed = config['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    data = ROOT / config['paths']['data'] / 'prepared'
    destination = ROOT / config['paths']['results'] / 'goal2'
    destination.mkdir(parents=True, exist_ok=True)
    pause_file = destination / 'pause.request'
    progress_file = destination / 'training_progress.json'
    history_file = destination / 'epoch_history.json'
    if pause_file.exists():
        raise RuntimeError('Pause requested. Remove results/goal2/pause.request only after authorization to resume.')
    if history_file.exists() and not args.resume:
        raise RuntimeError('Existing Goal 2 epoch history: use --resume to reuse completed training.')
    detector = config['models']['detector']
    cache = ROOT / config['paths']['checkpoints'] / 'huggingface'
    tokenizer = AutoTokenizer.from_pretrained(detector['id'], revision=detector['tokenizer_revision'], cache_dir=cache, use_fast=True, local_files_only=True)
    progress = json.loads(progress_file.read_text()) if args.resume and progress_file.exists() else None
    if progress and progress.get('status') == 'complete':
        print(json.dumps({'event':'completed_training_reused','selection':str(destination / 'selection.json')}),flush=True)
        return
    model_source = ROOT / progress['checkpoint'] if progress else detector['id']
    model_args = {'local_files_only':True} if progress else {'revision':detector['revision'], 'cache_dir':cache, 'num_labels':2, 'local_files_only':True}
    model = AutoModelForSequenceClassification.from_pretrained(model_source, **model_args).float().to('cuda')
    model.config.id2label = {0:'benign',1:'injection'}
    model.config.label2id = {'benign':0,'injection':1}
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'], weight_decay=training['weight_decay'])
    history = json.loads(history_file.read_text()) if history_file.exists() else []
    start_epoch, start_batch, partial_loss, partial_documents = 1, 0, 0., 0
    if progress:
        saved = torch.load(ROOT / progress['checkpoint'] / 'training_state.pt', map_location='cpu', weights_only=False)
        optimizer.load_state_dict(saved['optimizer'])
        random.setstate(saved['python_rng'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start_epoch, start_batch = saved['next_epoch'], saved['next_batch']
        partial_loss, partial_documents = saved['epoch_loss_sum'], saved['epoch_documents_seen']
    train_docs = read_jsonl(data / 'train.jsonl')
    search = {name: read_jsonl(data / (name + '.jsonl')) for name in ('search_temporary_threshold','search_candidate_scoring')}
    print(json.dumps({'event':'tokenization_started','train_documents':len(train_docs),'search_documents':{name:len(rows) for name,rows in search.items()}}), flush=True)
    started = time.perf_counter()
    train_windows = []
    for doc in train_docs:
        windows = window_document(tokenizer, doc['text'], doc['document_id'], doc['label'], doc.get('injection_spans'), config)
        if any(w['window_label'] is None for w in windows):
            raise ValueError('This prepared source has known spans; document-aggregate training is required for an unannotated source and cannot be silently replaced by window labels.')
        train_windows.append(windows)
    tokenize_seconds = time.perf_counter()-started
    parameter_details = [{'name':name,'shape':list(p.shape),'parameters':p.numel(),'bytes':p.numel()*p.element_size(),'dtype':str(p.dtype)} for name,p in model.named_parameters()]
    run = {'recorded_utc':datetime.now(timezone.utc).isoformat(),'command':f'python scripts/train_detector.py --microbatch-windows {args.microbatch_windows}',
        'config':config,'seed':seed,'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,
        'device':torch.cuda.get_device_name(),'cuda_runtime':torch.version.cuda,'numerics':'float32 parameters, activations and optimizer; TF32 disabled',
        'loss':'Window cross-entropy with known span labels; average windows within each document, then average documents within effective batch.',
        'selection':'Highest candidate-scoring document recall at threshold fitted to assigned temporary-threshold benign documents; tie: lower candidate document BCE; final tie: earlier epoch.',
        'effective_batch_documents':training['effective_batch'],'microbatch_windows':args.microbatch_windows,'gradient_accumulation':'All window microbatches belonging to one 32-document batch; variable with document lengths; final batch averages its actual document count.',
        'learning_rate_schedule':'constant','gradient_clipping':None,'train_documents':len(train_docs),'train_windows':sum(map(len,train_windows)),
        'train_long_documents':sum(len(w)>1 for w in train_windows),'tokenization_seconds':tokenize_seconds,
        'parameter_count':sum(p.numel() for p in model.parameters()),'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
        'embedding_parameter_count':sum(p.numel() for name,p in model.named_parameters() if name.startswith('bert.embeddings.')),
        'encoder_matrix_shapes':[row for row in parameter_details if row['name'].startswith('bert.encoder.') and len(row['shape'])==2],
        'all_parameter_shapes':parameter_details}
    if args.resume and (destination / 'training_run.json').exists():
        previous_run = json.loads((destination / 'training_run.json').read_text())
        if 'first_effective_batch_memory_pilot' in previous_run:
            run['first_effective_batch_memory_pilot'] = previous_run['first_effective_batch_memory_pilot']
        run['initial_tokenization_seconds'] = previous_run.get('initial_tokenization_seconds',previous_run['tokenization_seconds'])
        run['resume_command'] = run['command']+' --resume'
    save_json(destination / 'training_run.json',run)
    if progress and progress.get('status') == 'needs_search_evaluation':
        pending = progress['epoch_record']
        pending['metrics'] = infer_search(model,tokenizer,search,config,destination / f"epoch_{pending['epoch']}",args.microbatch_windows)
        history = [record for record in history if record['epoch'] != pending['epoch']] + [pending]
        save_json(history_file,history)
        save_json(progress_file,{'status':'epoch_complete','checkpoint':pending['checkpoint'],'next_epoch':pending['epoch']+1,'next_batch':0})
        print(json.dumps({'event':'recovered_epoch_evaluation','epoch':pending['epoch'],'reused_saved_training':True,'primary':pending['metrics']['primary']}),flush=True)
    model_fields = ('input_ids','attention_mask','token_type_ids')
    effective = int(training['effective_batch'])
    for epoch in range(start_epoch, int(training['epochs'])+1):
        order = list(range(len(train_docs)))
        random.Random(seed+epoch).shuffle(order)
        epoch_loss = partial_loss if epoch == start_epoch else 0.
        epoch_seen = partial_documents if epoch == start_epoch else 0
        batch_begin = start_batch if epoch == start_epoch else 0
        model.train()
        torch.cuda.reset_peak_memory_stats()
        epoch_start = time.perf_counter()
        batches = math.ceil(len(order)/effective)
        for batch_index in range(batch_begin, batches):
            doc_indices = order[batch_index*effective:(batch_index+1)*effective]
            flattened, weights = [], []
            for doc_index in doc_indices:
                windows = train_windows[doc_index]
                flattened.extend(windows)
                weights.extend([1/(len(windows)*len(doc_indices))]*len(windows))
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.
            for offset in range(0,len(flattened),args.microbatch_windows):
                micro = flattened[offset:offset+args.microbatch_windows]
                batch = tokenizer.pad([{key:w[key] for key in model_fields} for w in micro], padding='max_length', max_length=config['windowing']['primary_length'],return_tensors='pt')
                batch = {key:value.to('cuda') for key,value in batch.items()}
                labels = torch.tensor([w['window_label'] for w in micro],device='cuda')
                scaling = torch.tensor(weights[offset:offset+args.microbatch_windows],device='cuda')
                logits = model(**batch).logits
                loss = (F.cross_entropy(logits,labels,reduction='none')*scaling).sum()
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss; no successful epoch is claimed.')
                loss.backward()
                batch_loss += loss.item()
            optimizer.step()
            epoch_loss += batch_loss*len(doc_indices)
            epoch_seen += len(doc_indices)
            if batch_index == 0 and epoch == 1:
                run['first_effective_batch_memory_pilot']={'documents':len(doc_indices),'windows':len(flattened),
                    'microbatches':math.ceil(len(flattened)/args.microbatch_windows),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'counted_within_epoch':True}
                save_json(destination / 'training_run.json',run)
            if batch_index % 50 == 0 or batch_index+1 == batches:
                print(json.dumps({'event':'training','epoch':epoch,'batch':batch_index+1,'batches':batches,
                    'mean_document_balanced_window_ce':epoch_loss/epoch_seen,'elapsed_seconds':time.perf_counter()-epoch_start}),flush=True)
            if pause_file.exists():
                folder = ROOT / config['paths']['checkpoints'] / 'floating' / 'paused'
                checkpoint(model,tokenizer,optimizer,folder,{'next_epoch':epoch,'next_batch':batch_index+1,'epoch_loss_sum':epoch_loss,'epoch_documents_seen':epoch_seen})
                save_json(progress_file,{'status':'paused','checkpoint':str(folder.relative_to(ROOT)),'next_epoch':epoch,'next_batch':batch_index+1})
                print(json.dumps({'event':'paused','checkpoint':str(folder)}),flush=True)
                return
        epoch_seconds = time.perf_counter()-epoch_start
        folder = ROOT / config['paths']['checkpoints'] / 'floating' / f'epoch_{epoch}'
        checkpoint(model,tokenizer,optimizer,folder,{'next_epoch':epoch+1,'next_batch':0,'epoch_loss_sum':0.,'epoch_documents_seen':0})
        record = {'epoch':epoch,'checkpoint':str(folder.relative_to(ROOT)),'training_mean_document_balanced_window_ce':epoch_loss/epoch_seen,
            'training_seconds_this_process':epoch_seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'checkpoint_model_bytes':sum(path.stat().st_size for path in folder.glob('*.safetensors'))}
        save_json(progress_file,{'status':'needs_search_evaluation','checkpoint':record['checkpoint'],'next_epoch':epoch+1,'next_batch':0,'epoch_record':record})
        metrics = infer_search(model,tokenizer,search,config,destination / f'epoch_{epoch}',args.microbatch_windows)
        record['metrics'] = metrics
        history.append(record)
        save_json(history_file,history)
        save_json(progress_file,{'status':'epoch_complete','checkpoint':str(folder.relative_to(ROOT)),'next_epoch':epoch+1,'next_batch':0})
        print(json.dumps({'event':'epoch_complete','epoch':epoch,'primary':metrics['primary'],'mean_doc_bce':metrics['candidate_mean_document_bce']}),flush=True)
    selected = min(history,key=lambda row:(-row['metrics']['primary']['candidate_scoring']['recall'],row['metrics']['candidate_mean_document_bce'],row['epoch']))
    selection = {'selected_epoch':selected['epoch'],'checkpoint':selected['checkpoint'],'selection_rule':run['selection'],'metrics':selected['metrics']}
    save_json(destination / 'selection.json',selection)
    threshold = selected['metrics']['primary']['threshold']
    save_json(destination / 'search_threshold.json',{'threshold':threshold,'checkpoint':selected['checkpoint'],'alpha':config['evaluation']['fpr_target'],
        'fit_split':'search_temporary_threshold benign documents only','windowing':config['windowing'],'final_threshold':False})
    model = AutoModelForSequenceClassification.from_pretrained(ROOT/selected['checkpoint'],local_files_only=True).float().to('cuda').eval()
    pairs = read_jsonl(data / 'development_pairs.jsonl')
    pilot_docs = [pair[side] for pair in pairs for side in ('clean','attacked')]
    scored_pilot = score_documents(model,tokenizer,pilot_docs,batch_size=args.microbatch_windows,device='cuda',config=config)
    save_jsonl(destination / 'pilot_guard_scores.jsonl',scored_pilot['documents'])
    save_jsonl(destination / 'pilot_guard_windows.jsonl',scored_pilot['windows'])
    scoring_rows = read_jsonl(destination / f"epoch_{selected['epoch']}" / 'search_candidate_scoring_documents.jsonl')
    original = {doc['document_id']:doc for doc in search['search_candidate_scoring']}
    errors = {'false_positives':[],'false_negatives':[]}
    for row in scoring_rows:
        kind = 'false_positives' if row['label']==0 and row['risk_score']>threshold else 'false_negatives' if row['label']==1 and row['risk_score']<=threshold else None
        if kind and len(errors[kind])<5:
            doc = original[row['document_id']]
            errors[kind].append(row|{'text':doc['text'],'injection_spans':doc.get('injection_spans'),'reference_answer':doc.get('reference_answer')})
    save_json(destination / 'search_error_examples.json',errors)
    timing = timing_pilot(model,tokenizer,search['search_candidate_scoring'],config)
    save_json(destination / 'timing_pilot.json',timing)
    save_json(progress_file,{'status':'complete','checkpoint':selected['checkpoint'],'epochs_completed':len(history)})
    print(json.dumps({'event':'complete','selected_epoch':selected['epoch'],'checkpoint':selected['checkpoint'],'threshold':threshold,'primary':selected['metrics']['primary'],'timing_means':timing['means']},indent=2),flush=True)


if __name__ == '__main__':
    main()
