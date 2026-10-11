"""Explicit ordered development scoring; default preview, no historical relabelling.

--bringup executes one existing input through the shared document evaluator.
Full scoring requires the coordinator's separately qualified score identities.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gate_eval import load_config, score_documents, window_document
from search_precision import (SPLITS, SCHEME, approved_score_identity, groups,
                              read_json, rows, resolved, validate_score_identity, validate_document_rows)

CASE = 'aeslc:train:panus-s_inbox_1.subject:clean'
DATA = ROOT / 'results/goal4/numerical_bridge'


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n', encoding='utf8')
    temporary.replace(path)


def paused():
    if any((ROOT / rel).exists() for rel in ('results/goal2/pause.request', 'results/goal3/pause.request')):
        raise RuntimeError('Pause marker present; preserve scoring state.')



def scoring_batch_policy(config, *, floating=False):
    """Read actual mode-specific B1 equivalence bytes before batched scoring."""
    size = config['methods'].get('ordered_scoring_batch_size', 1)
    if type(size) is not int or size not in (1, 8, 32):
        raise ValueError('Use an explicitly qualified ordered scoring batch size.')
    if size == 1:
        return size, None
    run = read_json(ROOT / 'results/goal3/run.json')
    binding = config.get('_ordered_scoring_binding')
    if not floating and (not isinstance(binding, dict) or set(binding) != {'precision_map', 'qat_checkpoint'}):
        raise ValueError('Batched quantized scoring requires an explicit actual map/checkpoint binding.')
    precision = None if floating else binding['precision_map']
    checkpoint = None if floating else binding['qat_checkpoint']
    if not floating:
        from search_precision import precision_tuple
        precision_tuple(precision, config)
    checkpoint = None if checkpoint is None else str(resolved(checkpoint).relative_to(ROOT)).replace(chr(92), '/')
    expected = dict(checkpoint=run['selected_checkpoint'], qat=bool(checkpoint),
                    windowing=config['windowing'], evaluation=config['evaluation'], data=config['data'],
                    arithmetic_identity='bert_mini_ordered_fp32_floating_v1' if floating else 'bert_mini_ordered_fp32_frozen_quant_v1')
    if not floating:
        expected.update(scheme_id=SCHEME, precision_map=precision, calibration_file=run['calibration_file'],
                        frozen_scale_policy={k:run['format'][k] for k in ('weight_scale', 'activation_scale')})
        if checkpoint:
            expected['scored_checkpoint'] = checkpoint
    # Package metadata reads the installed backend version without importing Torch.
    from importlib.metadata import version
    execution = f'torch_{version("torch")}__cpu_eager_fp32_threads4_tf32off_autocastoff_flushdenormalfalse'
    expected['execution_identity'] = execution
    selection = read_json(DATA / 'development_cases_v1/selection.json')
    if selection['case_count'] != 32 or sum(c['window_count'] for c in selection['cases']) != 66:
        raise ValueError('Actual frozen32/66 batching inputs are required.')
    canonical = floating or (set(precision.values()) == {8} and checkpoint == 'checkpoints/quantized/qat_w8_a8')
    if canonical:
        mode = 'floating' if floating else 'w8'
        target = DATA / ('ordered_batching_' + mode + '_v1') / 'qualification.json'
        evidence = read_json(target)
        source = evidence['b1_identity']
        if source != expected or evidence['selection'] != selection:
            raise ValueError('Batch proof does not bind the actual checkpoint/map/backend/frozen inputs.')
        settings = dict(torch_version=version('torch'),device='cpu',threads=4,default_dtype='torch.float32',
                        cpu_autocast_enabled=False,cuda_matmul_tf32=False,cudnn_tf32=False,
                        flush_denormal_requested=False,model_training=False,wrapped_model_training=False)
        if evidence.get('settings') != settings:
            raise ValueError('Actual batch proof backend settings differ.')
        policy = evidence['batch_sizes'].get(str(size), {})
        expected_policy = dict(source,source_execution_identity=execution,
                               execution_identity=execution+f'__qualification_batch_policy{size}_tail_actual',configured_batch_size=size)
        if (not evidence.get('complete') or not policy.get('complete') or policy.get('passed') is not True
                or policy.get('completed_windows') != 66 or policy.get('active') is not None
                or policy.get('identity') != expected_policy or evidence.get('score_acceptance') is not None):
            raise ValueError('Complete actual B1 equivalence evidence is required for ordered batching.')
        from validate_qualification_exports import validate_exports, validate_reference_inputs
        data = DATA / ('development_qualification_' + mode + '_v1')
        exports = read_json(data / 'export.json')
        if exports['identity'] != source or not exports['complete'] or len(exports['completed_windows']) != 66:
            raise ValueError('Batch source must be all66 actual B1 exports for this identity.')
        validate_exports(data,qualification=dict(identity=source,selection=selection,windows=[],complete=False))
        validate_reference_inputs(data)
        # A private validator module instance retains the existing saved-byte checks.
        import importlib.util
        spec = importlib.util.spec_from_file_location('_score_batch_saved_validator', ROOT/'scripts/qualify_ordered_batching.py')
        checker = importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
        checker.OUT,checker.DATA,checker.FLOATING = target.parent,data,floating
        bykey = {(v['case_index'],v['window_index']):v for v in exports['completed_windows']}
        items = [bykey[c['case_index'],wi] for c in selection['cases'] for wi in range(c['window_count'])]
        start = 0
        for record in policy['batches']:
            checker.check_record(record,items,size,start)
            if read_json(target.parent/record['attempt_folder']/'batch.json') != record:
                raise ValueError('Actual batch record differs from saved policy.')
            start += record['actual_batch_size']
        if start != 66:
            raise ValueError('Actual batch proof must cover every frozen window exactly once.')
        return size,target
    from export_precision_qualification import validate_export, batching_state
    matches = []
    for target in sorted((DATA/'precision_qualification_v1').glob('*/batching/qualification.json')):
        evidence = read_json(target);source = evidence.get('b1_identity', {})
        if all(source.get(k) == v for k,v in expected.items()) and set(source) == set(expected)|{'precision_source','qat_checkpoint','storage_contract'}:
            matches.append(target)
    if len(matches) != 1:
        raise ValueError('Exactly one own actual precision-map/checkpoint batch proof is required; W8 fallback is forbidden.')
    target = matches[0];out = target.parent.parent;exports = validate_export(out,complete=True)
    _,_,evidence,_,positions = batching_state(out,exports)
    source = evidence['b1_identity']
    if (evidence['selection'] != selection or positions[size] != 66 or evidence.get('active') is not None
            or evidence.get('settings') != dict(device='cpu',dtype='float32',threads=4,autocast=False,tf32=False,torch_version=version('torch'))
            or source.get('storage_contract') != 'decoded_signed_i8_codes_plus_real_packed_w4w8_v1'):
        raise ValueError('Own actual precision batch proof inputs/settings/completion differ.')
    # Only the specifically matched source may establish the selected mode.
    source = evidence['b1_identity']
    if (any(source.get(k) != v for k,v in expected.items())
            or source['qat_checkpoint'] != checkpoint or source['precision_source'] != ('qat' if checkpoint else 'ptq')):
        raise ValueError('Own batch proof actual precision source differs.')
    return size,target


def load_model(config, precision=None, qat_checkpoint=None, *, floating=False, scoring=False):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from gate_fpga_reference import OrderedBertMini
    from gate_quant import configure_exact_fp32, install_quantized_linears, load_quantized_checkpoint
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    torch.set_flush_denormal(False)
    configure_exact_fp32()
    run = read_json(ROOT / 'results/goal3/run.json')
    source = resolved(run['selected_checkpoint'])
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, use_fast=True)
    if qat_checkpoint:
        if floating:
            raise ValueError('Floating recall reference cannot use a QAT checkpoint.')
        model, metadata = load_quantized_checkpoint(resolved(qat_checkpoint), 'cpu')
        if metadata['precision_map'] != precision:
            raise ValueError('QAT checkpoint map differs from the requested precision map.')
    else:
        model = AutoModelForSequenceClassification.from_pretrained(source, local_files_only=True).float()
        if not floating:
            model = install_quantized_linears(model, read_json(resolved(run['calibration_file'])), precision)
    model.eval()
    wrapped = OrderedBertMini(model, encoder_mode='floating' if floating else 'quantized')
    execution = f'torch_{torch.__version__}__cpu_eager_fp32_threads4_tf32off_autocastoff_flushdenormalfalse'
    if scoring:
        size, _ = scoring_batch_policy(config, floating=floating)
        if size != 1:
            execution += f'__document_score_batch{size}_tail_actual'
    identity = dict(checkpoint=run['selected_checkpoint'], qat=bool(qat_checkpoint),
                    windowing=config['windowing'], evaluation=config['evaluation'], data=config['data'],
                    arithmetic_identity=wrapped.arithmetic_identity, execution_identity=execution)
    if not floating:
        identity.update(scheme_id=SCHEME, precision_map=precision,
                        calibration_file=run['calibration_file'],
                        frozen_scale_policy={k: run['format'][k] for k in ('weight_scale', 'activation_scale')})
        if qat_checkpoint:
            identity['scored_checkpoint'] = str(resolved(qat_checkpoint).relative_to(ROOT)).replace('\\', '/')
    return wrapped, tokenizer, identity


def score_to_folder(folder, config, precision=None, qat_checkpoint=None, *, floating=False, chunk_documents=64):
    """Full permitted search partitions only; resume from committed JSONL byte offsets."""
    paused()
    if not floating and precision is not None and set(precision.values()) == {4, 8}:
        # Own-map batching and generic quantized approval do not grant mixed numerical scope.
        from run_precision_campaign import validate_mixed_review
        validate_mixed_review(config, precision, qat_checkpoint)
    approved = approved_score_identity(config, floating=floating)
    if chunk_documents < 1:
        raise ValueError('chunk_documents must be positive.')
    config = dict(config, _ordered_scoring_binding=dict(precision_map=precision, qat_checkpoint=qat_checkpoint))
    batch_size, batch_path = scoring_batch_policy(config, floating=floating)
    model, tokenizer, identity = load_model(config, precision, qat_checkpoint, floating=floating, scoring=False)
    if batch_size != 1:
        identity = dict(identity, execution_identity=identity['execution_identity']+f'__document_score_batch{batch_size}_tail_actual')
    if any(identity[k] != v for k, v in approved.items()):
        raise ValueError('This scorer backend/arithmetic does not match approved identities.')
    validate_score_identity(identity, config, qat=bool(qat_checkpoint), floating=floating)
    candidate = folder / 'candidate.json'
    if candidate.exists():
        if read_json(candidate)['identity'] != identity:
            raise ValueError('Completed score identity differs; preserve existing evidence.')
        from search_precision import recompute_metrics
        actual = recompute_metrics(folder, config)
        prior = read_json(candidate)['metrics']
        if (actual['temporary_threshold'] != prior['primary']['threshold']
                or actual['protection'] != prior['primary']['candidate_scoring']
                or actual['ordinary_0_5'] != prior['ordinary_threshold_0_5']):
            raise ValueError('Completed score rows and metrics disagree.')
        return read_json(candidate)
    folder.mkdir(parents=True, exist_ok=True)
    progress_path = folder / 'progress.json'
    progress = read_json(progress_path) if progress_path.exists() else dict(identity=identity, splits={}, scoring_seconds=0.0, batch_size=batch_size)
    if progress['identity'] != identity or progress.get('batch_size', 1) != batch_size:
        raise ValueError('Partial scoring identity/batch policy differs; preserve existing evidence.')
    from train_detector import enrich, summarize
    for split in SPLITS:
        documents = rows(ROOT / 'data/prepared' / (split + '.jsonl'))
        if any(doc.get('split') != 'search' or doc.get('search_subset') != split.removeprefix('search_') for doc in documents):
            raise ValueError('Prepared partition label differs.')
        state = progress['splits'].setdefault(split, dict(next_document=0, windows_bytes=0, documents_bytes=0))
        if not 0 <= state['next_document'] <= len(documents):
            raise ValueError('Resume index exceeds the assigned partition.')
        streams = {}
        try:
            for level in ('windows', 'documents'):
                path = folder / f'{split}_{level}.jsonl'
                stream = path.open('r+b' if path.exists() else 'w+b')
                if stream.seek(0, 2) < state[level + '_bytes']:
                    stream.close()
                    raise ValueError('Score file is shorter than the committed resume position.')
                stream.truncate(state[level + '_bytes'])
                stream.seek(state[level + '_bytes'])
                streams[level] = stream
            committed = rows(folder / f'{split}_documents.jsonl')
            validate_document_rows(committed, documents, split, complete=False)
            if len(committed) != state['next_document']:
                raise ValueError('Committed document count differs from resume position.')
            for start in range(state['next_document'], len(documents), chunk_documents):
                paused()
                begun = time.perf_counter()
                current = documents[start:start + chunk_documents]
                scored = enrich(score_documents(model, tokenizer, current, batch_size=batch_size, device='cpu',
                                                  config=config, risk_function=model.predict_risk), current)
                for level, stream in streams.items():
                    for row in scored[level]:
                        if level == 'windows':
                            row['document_index'] += start
                        stream.write((json.dumps(row) + '\n').encode('utf8'))
                    stream.flush()
                    state[level + '_bytes'] = stream.tell()
                state['next_document'] = start + len(current)
                progress['scoring_seconds'] += time.perf_counter() - begun
                save(progress_path, progress)
                print(json.dumps(dict(event='scored_chunk', split=split, completed_documents=state['next_document'],
                                      total_documents=len(documents))), flush=True)
        finally:
            for stream in streams.values():
                stream.close()
    output_rows = [rows(folder / f'{split}_documents.jsonl') for split in SPLITS]
    for split, scored in zip(SPLITS, output_rows):
        validate_document_rows(scored, rows(ROOT / 'data/prepared' / (split + '.jsonl')), split)
    metrics = summarize(*output_rows, config)
    record = dict(name=folder.name, identity=identity, metrics=metrics,
                  scoring_elapsed_seconds=progress['scoring_seconds'],
                  scoring_batch_size=batch_size, batching_evidence=None if batch_path is None else str(batch_path.relative_to(ROOT)).replace(chr(92), '/'),
                  timing_scope='research scoring cost; not a detector performance benchmark',
                  recorded_utc=datetime.now(timezone.utc).isoformat())
    save(candidate, record)
    if floating:
        save(folder / 'floating_reference.json', dict(identity=identity, metrics=metrics,
             source_scores=str(folder.relative_to(ROOT)).replace('\\', '/')))
    return record


def bringup(config):
    """One unchanged W8-QAT case, no numerical approval or threshold fit."""
    paused()
    folder = DATA / 'ordered_document_scorer_v1'
    if folder.exists():
        raise RuntimeError('Preserve existing scorer bring-up evidence; inspect rather than overwrite.')
    manifest = read_json(DATA / 'manifest.json')
    if manifest['case'] != CASE:
        raise ValueError('Saved bring-up case differs.')
    selected = [d for d in rows(ROOT / 'data/prepared/search_candidate_scoring.jsonl') if d['document_id'] == CASE]
    if len(selected) != 1:
        raise ValueError('Saved case must occur exactly once in development scoring.')
    precision = dict.fromkeys(groups(config), 8)
    model, tokenizer, identity = load_model(config, precision, 'checkpoints/quantized/qat_w8_a8')
    windows = window_document(tokenizer, selected[0]['text'], CASE, selected[0]['label'], selected[0].get('injection_spans'), config)
    if len(windows) != 1:
        raise ValueError('Saved bring-up case must remain a single window.')
    batch = tokenizer.pad([{k: windows[0][k] for k in ('input_ids', 'attention_mask', 'token_type_ids')}],
                          padding='max_length', max_length=256)
    if any(batch[k][0] != manifest[k] for k in ('input_ids', 'attention_mask', 'token_type_ids')):
        raise ValueError('Current shared windowing/tokenizer differs from saved input IDs.')
    begun = time.perf_counter()
    scored = score_documents(model, tokenizer, selected, batch_size=1, device='cpu', config=config,
                             risk_function=model.predict_risk)
    elapsed = time.perf_counter() - begun
    actual = scored['documents'][0]
    expected_logits = (DATA / 'ordered_model_v1/model_logits.bin').read_bytes()
    expected_risk = (DATA / 'ordered_model_v1/model_risk.bin').read_bytes()
    logits_exact = struct.pack('<2f', *scored['windows'][0]['logits']) == expected_logits
    risk_exact = struct.pack('<f', actual['risk_score']) == expected_risk
    record = dict(completed=True, passed=logits_exact and risk_exact, case=CASE, identity=identity,
                  saved_ids_exact=True, logits_exact=logits_exact, risk_exact=risk_exact,
                  document_max_exact=actual['risk_score'] == max(actual['window_scores']),
                  forward_seconds=elapsed, numerical_acceptance=None, campaign_maps_consulted=0,
                  new_model_forwards=1, scored=scored, recorded_utc=datetime.now(timezone.utc).isoformat())
    save(folder / 'summary.json', record)
    if not record['passed']:
        raise RuntimeError('Shared scoring path differs from saved ordered reference; see evidence.')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bringup', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--floating', action='store_true')
    parser.add_argument('--precision-map', help='JSON file containing exactly the sixteen W4/W8 group choices')
    parser.add_argument('--qat-checkpoint')
    parser.add_argument('--output', default='results/goal5/qualified_scores/floating')
    args = parser.parse_args()
    config = load_config()
    if args.bringup:
        if args.execute:
            result = bringup(config)
            print(json.dumps({k:v for k,v in result.items() if k != 'scored'}, indent=2))
        else:
            print(json.dumps(dict(execute=False, case=CASE, new_forwards=1, campaign_maps_consulted=0)))
        return
    if not args.execute:
        from search_precision import readiness
        print(json.dumps(dict(execute=False, missing_prerequisites=readiness(config),
                              partitions=SPLITS, output=args.output), indent=2))
        return
    if not args.floating and not args.precision_map:
        parser.error('Quantized scoring requires --precision-map.')
    precision = None if args.floating else read_json(resolved(args.precision_map))
    if precision is not None:
        from search_precision import precision_tuple
        precision_tuple(precision, config)
    score_to_folder(resolved(args.output), config, precision, args.qat_checkpoint, floating=args.floating)


if __name__ == '__main__':
    main()
