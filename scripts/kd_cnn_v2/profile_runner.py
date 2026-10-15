"""Matched complete-document software timings; coordinator holds shared GPU lock.

Warm latency includes local tokenization, all tails, padding, serial B1 transfers,
inference, returned margins, maximum aggregation and fixed-threshold decision.
Device-only and fresh-process observations are separate software measurements.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ENTRY = time.perf_counter()
SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT))
STAGES = ('tokenization_padding_ms', 'h2d_validation_ms', 'inference_ms', 'd2h_ms', 'aggregation_decision_ms', 'complete_ms')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def allowed_path(path):
    path = Path(path)
    if any(part.lower() in {'notinject', 'final_test', 'final_calibration', 'new_confirmation', 'confirmation', 'new_calibration'} for part in path.parts):
        raise ValueError('Timing accepts development inputs; protected/final paths are closed')
    return path


def read_rows(path):
    path = allowed_path(path)
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def timing_documents(path, canonical):
    """Full raw JSONL, or JSON {documents:[...],raw_path:...} membership manifest."""
    path = allowed_path(path)
    sources = {str(path.resolve()): digest(path)}
    if path.suffix.lower() == '.jsonl':
        selected = read_rows(path)
    else:
        manifest = read(path)
        selected = manifest.get('documents', manifest.get('samples', manifest.get('document_ids')))
        if not isinstance(selected, list): raise ValueError('Timing manifest needs explicit documents membership')
        raw_source = manifest.get('raw_path', manifest.get('raw_source'))
        if raw_source:
            raw_path = Path(raw_source)
            raw_path = allowed_path(raw_path if raw_path.is_absolute() else canonical / raw_path)
            sources[str(raw_path.resolve())] = digest(raw_path)
            if manifest.get('raw_sha256') and manifest['raw_sha256'] != sources[str(raw_path.resolve())]:
                raise ValueError('Timing raw source digest differs')
            raw = {row['document_id']: row for row in read_rows(raw_path)}
            merged = []
            for item in selected:
                item = {'document_id': item} if isinstance(item, str) else item
                row = raw[item['document_id']]
                if item.get('text_sha256', row.get('text_sha256')) != row.get('text_sha256'):
                    raise ValueError('Timing membership text digest differs')
                merged.append({**row, **item})
            selected = merged
    if not selected or len({row['document_id'] for row in selected}) != len(selected):
        raise ValueError('Nonempty unique timing document membership required')
    for row in selected:
        if not isinstance(row.get('text'), str): raise ValueError('Timing rows require complete raw text')
        actual = hashlib.sha256(row['text'].encode('utf-8')).hexdigest()
        if row.get('text_sha256') != actual: raise ValueError('Timing text digest differs/missing')
    return selected, sources


def load_tokenizer(canonical, contract):
    from transformers import AutoTokenizer
    policy = contract['tokenizer']; folder = canonical / policy['path']
    for name, expected in policy['files_sha256'].items():
        if digest(folder / name) != expected: raise ValueError('Common tokenizer bytes changed: ' + name)
    tokenizer = AutoTokenizer.from_pretrained(folder, use_fast=True, local_files_only=True)
    if tokenizer.num_special_tokens_to_add(pair=False) != 2 or tokenizer.pad_token_id != 0:
        raise ValueError('Expected shared BERT CLS/SEP and PAD0 tokenizer')
    return tokenizer


def make_timing(canonical, output):
    """Freeze development-only natural sample plus timing-only repeated text."""
    canonical, output = Path(canonical), Path(output)
    shared = canonical / 'results/dual_track_v1/shared'
    if output.exists():
        existing = read(output)
        if existing.get('schema') != 'kd_cnn_v2_development_timing_v1':
            raise ValueError('Existing timing artifact is owned by another protocol; preserve it')
        timing_documents(output, canonical)
        return existing
    # Preserve another track's prospective timing contract, if supplied. A
    # separate artifact can be reconciled by the coordinator without mutation.
    sibling = shared / 'prospective_v1/timing_manifest.json'
    common_path = shared / 'contract.json'; contract = read(common_path)
    tokenizer = load_tokenizer(canonical, contract)
    source = shared / 'data/raw/new_selection.jsonl'; rows = read_rows(source)
    measured = []
    for row in rows:
        pairs, count, _ = encode_document(tokenizer, row['text'])
        measured.append((count, len(pairs), row))
    measured.sort(key=lambda item: (item[0], item[2]['document_id']))
    selected, used_sources = [], set()
    for quartile in range(4):
        low, high = len(measured) * quartile // 4, len(measured) * (quartile + 1) // 4
        candidates = sorted(measured[low:high], key=lambda item: hashlib.sha256(('424242|' + item[2]['document_id']).encode()).hexdigest())
        chosen = []
        for unique_only in (True, False):
            for count, windows, row in candidates:
                if any(item['document_id'] == row['document_id'] for item in chosen): continue
                group = row.get('source_group_id', row.get('source_id', row['document_id']))
                if unique_only and group in used_sources: continue
                chosen.append({**row, 'content_tokens': count, 'windows': windows,
                    'timing_only': False, 'selection_reason': 'development content-token quartile' + str(quartile + 1) + ', deterministic seed424242, source diversity'})
                used_sources.add(group)
                if len(chosen) == 3: break
            if len(chosen) == 3: break
        if len(chosen) != 3: raise ValueError('Insufficient development sources for twelve-document timing sample')
        selected.extend(chosen)
    benign = [item for item in measured if item[2].get('label') == 0 and not item[2].get('hard_benign')]
    if not benign: benign = [item for item in measured if item[2].get('label') == 0]
    if not benign: raise ValueError('Benign development text required for timing-only stress documents')
    _, _, base = min(benign, key=lambda item: (item[0], item[2]['document_id']))
    encoded = tokenizer(base['text'], add_special_tokens=False, return_offsets_mapping=True, truncation=False, verbose=False)
    if not encoded['input_ids']: raise ValueError('Empty benign stress source')
    # A source prefix <=128 tokens ensures each whole-text concatenation adds
    # fewer than one stride, allowing every requested window count exactly.
    source_end = encoded['offset_mapping'][min(128, len(encoded['input_ids'])) - 1][1]
    base_text = base['text'][:source_end]
    stress = []
    for target_windows in (1, 2, 4, 8, 16, 32):
        text = base_text; repeats = 1
        while True:
            pairs, tokens, _ = encode_document(tokenizer, text)
            if len(pairs) >= target_windows: break
            repeats += 1; text = '\n\n'.join([base_text] * repeats)
            if repeats > 10000: raise ValueError('Stress concatenation bound exceeded')
        if len(pairs) != target_windows: raise ValueError('Stress text failed exact requested window count')
        stress.append({'document_id': 'stress' + str(target_windows), 'text': text,
            'text_sha256': hashlib.sha256(text.encode()).hexdigest(), 'label': None, 'genre': 'timing_stress_repeated_benign',
            'source_group_id': base.get('source_group_id'), 'content_tokens': tokens, 'windows': len(pairs),
            'timing_only': True, 'not_quality_evaluation': True, 'origin_document_id': base['document_id'],
            'origin_text_sha256': base['text_sha256'], 'source_prefix_end_char': source_end, 'whole_prefix_repeats': repeats,
            'selection_reason': 'Explicit synthetic length stress; repeated benign development prefix, no new security labels'})
    result = {'schema': 'kd_cnn_v2_development_timing_v1', 'immutable': True, 'seed': 424242,
        'common_contract_sha256': digest(common_path), 'raw_development_path': str(source), 'raw_development_sha256': digest(source),
        'documents': selected + stress, 'natural_sample_documents': len(selected), 'timing_only_stress_documents': len(stress),
        'unique_natural_source_groups': len({row.get('source_group_id', row.get('source_id')) for row in selected}),
        'windowing': {'length': 256, 'capacity': 254, 'overlap': 64, 'stride': 190, 'all_tails': True},
        'boundary': 'Complete raw document retained until fixed-threshold decision; natural development sample plus explicit synthetic length stress, not a deployment-frequency distribution',
        'sibling_manifest_present': sibling.is_file(), 'sibling_manifest_path': str(sibling),
        'reconciliation': 'Task-local prospective timing membership; compare/reconcile with sibling membership before cross-track timing claims',
        'profiler_sha256': digest(Path(__file__))}
    write(output, result)
    ready = shared / 'timing_ready.json'
    pointer = {'owner': 'kd_cnn_v2', 'timing_manifest': str(output.resolve()), 'timing_manifest_sha256': digest(output),
               'common_contract_sha256': digest(common_path), 'natural_documents': len(selected), 'timing_stress_windows': [1, 2, 4, 8, 16, 32],
               'source_role': 'new_selection only; no confirmation access'}
    if ready.exists() and read(ready) != pointer: raise ValueError('Existing timing-ready pointer differs; preserve it')
    write(ready, pointer)
    return result


def encode_document(tokenizer, text):
    """L256 includes two specials: capacity254, overlap64, content stride190."""
    import torch
    content = tokenizer(text, add_special_tokens=False, truncation=False, return_attention_mask=False,
                        return_token_type_ids=False, verbose=False)['input_ids']
    pairs, spans, start = [], [], 0
    while True:
        end = min(start + 254, len(content))
        ids = tokenizer.build_inputs_with_special_tokens(content[start:end])
        if len(ids) != end - start + 2: raise ValueError('Unexpected special-token construction')
        mask = [1] * len(ids)
        pairs.append({'input_ids': torch.tensor([ids + [0] * (256 - len(ids))], dtype=torch.long),
                      'attention_mask': torch.tensor([mask + [0] * (256 - len(mask))], dtype=torch.long)})
        spans.append((start, end))
        if end == len(content): break
        start += 190
    return pairs, len(content), spans


def synchronize(device):
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.synchronize(device)


def infer_margin(model, pair, integer=False, explicit_bert=False):
    import torch
    if integer:
        logits = model.integer_logits(pair['input_ids'], pair['attention_mask'], validate=False)
        return logits[:, 1] - logits[:, 0]
    if explicit_bert:
        if model.config.num_hidden_layers != 4: raise ValueError('Explicit graph route requires common four-layer BERT')
        embeddings = model.bert.embeddings(input_ids=pair['input_ids'], token_type_ids=torch.zeros_like(pair['input_ids']))
        expanded = pair['attention_mask'][:, None, None, :].expand(1, 1, 256, 256).to(embeddings.dtype)
        inverted = 1. - expanded
        additive = inverted.masked_fill(inverted.bool(), torch.finfo(embeddings.dtype).min)
        encoded = model.bert.encoder(embeddings, attention_mask=additive, head_mask=[None] * 4,
            use_cache=False, output_attentions=False, output_hidden_states=False, return_dict=False)
        logits = model.classifier(model.dropout(model.bert.pooler(encoded[0])))
    else:
        logits = model(**pair).logits
    if logits.shape != (1, 2): raise ValueError('Expected fixed B1 binary logits')
    return logits[:, 1] - logits[:, 0]


class GraphRoute:
    def __init__(self, model, first, integer, teacher):
        import torch
        began = time.perf_counter(); self.model = model
        self.integer, self.teacher = integer, teacher
        self.static = {name: value.to('cuda').clone() for name, value in first.items()}
        if integer: model.validate_inputs(self.static['input_ids'], self.static['attention_mask'])
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(side):
            for _ in range(3): infer_margin(model, self.static, integer, teacher)
        torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.value = infer_margin(model, self.static, integer, teacher)
        torch.cuda.synchronize(); self.setup_seconds = time.perf_counter() - began

    def copy(self, pair):
        for name, value in pair.items(): self.static[name].copy_(value)

    def run(self):
        self.graph.replay(); return self.value


def to_number(value, integer):
    scalar = value.detach().cpu().item()
    if not math.isfinite(scalar): raise RuntimeError('Nonfinite timing margin')
    return float(scalar) / 4096. if integer else float(scalar)


def observe(model, tokenizer, document, device, integer, threshold, graph=None):
    import torch
    synchronize(device); began = time.perf_counter()
    pairs, content_tokens, _ = encode_document(tokenizer, document['text'])
    prepared = time.perf_counter(); h2d = network = d2h = 0.; margins = []
    with torch.inference_mode():
        for pair in pairs:
            at = time.perf_counter()
            # Validation is charged for every new input on either route.
            if graph:
                graph.copy(pair); target = graph.static
            else:
                target = {name: value.to(device) for name, value in pair.items()}
            if integer: model.validate_inputs(target['input_ids'], target['attention_mask'])
            synchronize(device); moved = time.perf_counter()
            value = graph.run() if graph else infer_margin(model, target, integer)
            synchronize(device); inferred = time.perf_counter()
            margins.append(to_number(value, integer)); returned = time.perf_counter()
            h2d += moved - at; network += inferred - moved; d2h += returned - inferred
    aggregate_at = time.perf_counter(); margin = max(margins); decision = margin > threshold; ended = time.perf_counter()
    return {'ms': {'tokenization_padding_ms': 1000 * (prepared - began), 'h2d_validation_ms': 1000 * h2d,
        'inference_ms': 1000 * network, 'd2h_ms': 1000 * d2h, 'aggregation_decision_ms': 1000 * (ended - aggregate_at),
        'complete_ms': 1000 * (ended - began)}, 'margin': margin, 'decision': decision,
        'window_margins': margins, 'windows': len(pairs), 'content_tokens': content_tokens}


def device_observe(model, pairs, device, integer, graph=None):
    """Prepared device tensors; excludes tokenization, H2D, D2H, host aggregation."""
    import torch
    synchronize(device); began = time.perf_counter(); outputs = []
    if str(device).startswith('cuda'):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
    with torch.inference_mode():
        for pair in pairs:
            if graph:
                # Device-to-device static copies are part of this graph route.
                graph.copy(pair); outputs.append(graph.run().clone())
            else: outputs.append(infer_margin(model, pair, integer))
    if str(device).startswith('cuda'): end.record()
    synchronize(device); ended = time.perf_counter()
    return {'inference_host_interval_ms': 1000 * (ended - began),
            'cuda_event_interval_ms': float(begin.elapsed_time(end)) if str(device).startswith('cuda') else None}


def summary(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    return {'mean': float(values.mean()), 'p50': float(np.quantile(values, .5)),
            'p90': float(np.quantile(values, .9)), 'p95': float(np.quantile(values, .95)),
            'minimum': float(values.min()), 'maximum': float(values.max()), 'observations': len(values)}


def qualify(model, tokenizer, documents, device, integer, threshold, reference=None, graph=None):
    import torch
    from gate_eval import window_document
    checks = []; predictions = {}; largest_delta = 0.
    for document in documents:
        pairs, count, spans = encode_document(tokenizer, document['text'])
        expected = window_document(tokenizer, document['text'], document['document_id'], config={
            'windowing': {'primary_length': 256, 'content_token_overlap': 64}})
        if len(expected) != len(pairs): raise ValueError('Tail/window count differs from protocol')
        for pair, window in zip(pairs, expected):
            padded = window['input_ids'] + [0] * (256 - len(window['input_ids']))
            if pair['input_ids'].tolist()[0] != padded: raise ValueError('Raw timing token IDs differ from protocol')
        eager, current = [], []
        with torch.inference_mode():
            for pair in pairs:
                target = {name: value.to(device) for name, value in pair.items()}
                if integer: model.validate_inputs(target['input_ids'], target['attention_mask'])
                eager.append(to_number(infer_margin(model, target, integer), integer))
                if graph:
                    graph.copy(target); current.append(to_number(graph.run(), integer))
                else: current.append(eager[-1])
        saved = reference.get(document['document_id']) if reference is not None else None
        if reference is not None and saved is None and not document.get('timing_only'):
            raise ValueError('Reference score missing natural timing document')
        if saved and saved.get('text_sha256') != document['text_sha256']: raise ValueError('Saved score text digest differs')
        expected_margins = saved['window_margins'] if saved else eager
        if len(expected_margins) != len(current): raise ValueError('Saved-score timing membership differs')
        tolerance = 0. if integer else 1e-5
        delta = max(abs(a - b) for a, b in zip(current, expected_margins)); largest_delta = max(largest_delta, delta)
        if any(abs(a - b) > tolerance + (0 if integer else 1e-5 * abs(b)) for a, b in zip(current, expected_margins)):
            raise ValueError('Timing route differs from reference document margins')
        if (max(current) > threshold) != (max(expected_margins) > threshold): raise ValueError('Timing route changes fixed-threshold decision')
        if graph and any(abs(a - b) > tolerance + (0 if integer else 1e-5 * abs(b)) for a, b in zip(current, eager)):
            raise ValueError('Graph differs from ordinary eager B1 margins')
        predictions[document['document_id']] = current
        checks.append({'document_id': document['document_id'], 'content_tokens': count, 'windows': len(pairs),
                       'last_content_span': spans[-1], 'max_margin_absolute_delta': delta, 'decision_preserved': True,
                       'reference': 'saved evaluation' if saved else 'direct eager B1; synthetic timing stress has no quality labels'})
    return {'passed': True, 'reference': 'saved matching evaluation where present; timing-only stress uses direct eager B1' if reference is not None else 'direct eager B1 ordinary-mask route',
            'checks': checks, 'maximum_absolute_margin_delta': largest_delta,
            'scope': 'Every timing document/window; graph qualification covers this workload, not an untested population'}, predictions


def profile_target(args, target_id, descriptor, contract, documents, threshold, reference):
    import torch
    from evaluate import load_target
    integer = descriptor['kind'] == 'cnn_integer'
    model_began = time.perf_counter(); model = load_target(Path(args.canonical_root), descriptor, args.device, contract)
    synchronize(args.device); model_loaded = time.perf_counter()
    tokenizer = load_tokenizer(Path(args.canonical_root), contract); tokenizer_loaded = time.perf_counter()
    if args.cold_child:
        document = next(d for d in documents if d['document_id'] == args.cold_child)
        measured = observe(model, tokenizer, document, args.device, integer, threshold)
        return {'complete': True, 'target_id': target_id, 'document_id': document['document_id'], 'observation': measured,
            'model_load_seconds': model_loaded - model_began, 'tokenizer_load_seconds': tokenizer_loaded - model_loaded,
            'entry_to_first_decision_seconds': time.perf_counter() - ENTRY,
            'scope': 'Fresh Python process, warm OS/filesystem cache state uncontrolled; eager first decision, not power-on cold hardware'}
    qualified, predictions = qualify(model, tokenizer, documents, args.device, integer, threshold, reference)
    graph = None; graph_record = {'requested': args.graph, 'selected': False}
    if args.graph and str(args.device).startswith('cuda'):
        try:
            first = encode_document(tokenizer, documents[0]['text'])[0][0]
            candidate = GraphRoute(model, first, integer, descriptor['kind'] == 'teacher')
            proof, graph_predictions = qualify(model, tokenizer, documents, args.device, integer, threshold, reference, candidate)
            # Three matched complete-document pilot observations per route. Same
            # fixed sample; never use quality outcomes or a new shape search.
            eager_ms, graph_ms = [], []
            for document in documents:
                for _ in range(3):
                    eager_ms.append(observe(model, tokenizer, document, args.device, integer, threshold)['ms']['complete_ms'])
                    graph_ms.append(observe(model, tokenizer, document, args.device, integer, threshold, candidate)['ms']['complete_ms'])
            selected = statistics.mean(graph_ms) < statistics.mean(eager_ms)
            graph_record = {'requested': True, 'selected': selected, 'setup_seconds': candidate.setup_seconds,
                'qualification': proof, 'pilot_eager_complete_ms': eager_ms, 'pilot_graph_complete_ms': graph_ms,
                'selection': 'lower matched mean complete-document pilot latency; one static B1/L256 graph',
                'bert_mask_route': 'always additive4D, compared against ordinary2D eager' if descriptor['kind'] == 'teacher' else None}
            if selected: graph, qualified, predictions = candidate, proof, graph_predictions
        except Exception as error:
            graph_record = {'requested': True, 'selected': False, 'failure': str(error), 'fallback': 'qualified eager baseline'}
            synchronize(args.device)
    samples = []
    for document in documents:
        args.budget_check()
        for _ in range(args.warmups): observe(model, tokenizer, document, args.device, integer, threshold, graph)
        repetitions = []
        for _ in range(args.repeats):
            args.budget_check()
            measured = observe(model, tokenizer, document, args.device, integer, threshold, graph)
            expected = predictions[document['document_id']]
            if measured['window_margins'] != expected:
                if integer or any(abs(a - b) > 1e-5 + 1e-5 * abs(b) for a, b in zip(measured['window_margins'], expected)):
                    raise ValueError('Timed route changed qualified margin')
            if measured['decision'] != (max(expected) > threshold): raise ValueError('Timed route changed decision')
            repetitions.append(measured)
        pairs, count, _ = encode_document(tokenizer, document['text'])
        device_pairs = [{name: value.to(args.device) for name, value in pair.items()} for pair in pairs]
        if integer:
            for pair in device_pairs: model.validate_inputs(pair['input_ids'], pair['attention_mask'])
        for _ in range(args.warmups): device_observe(model, device_pairs, args.device, integer, graph)
        device_times = []
        for _ in range(args.repeats):
            args.budget_check(); device_times.append(device_observe(model, device_pairs, args.device, integer, graph))
        samples.append({'document_id': document['document_id'], 'text_sha256': document['text_sha256'],
            'genre': document.get('genre'), 'label': document.get('label'), 'utf8_bytes': len(document['text'].encode()),
            'content_tokens': count, 'windows': len(pairs), 'threshold_margin': threshold,
            'summary_ms': {name: summary([r['ms'][name] for r in repetitions]) for name in STAGES},
            'device_only_ms': {name: summary([r[name] for r in device_times]) for name in device_times[0] if device_times[0][name] is not None},
            'repetitions': repetitions, 'device_only_repetitions': device_times})
    return {'complete': True, 'target_id': target_id, 'descriptor': descriptor, 'device': args.device,
        'route': 'cuda_graph' if graph else 'eager', 'graph': graph_record, 'qualification': qualified, 'samples': samples,
        'model_load_seconds': model_loaded - model_began, 'tokenizer_load_seconds': tokenizer_loaded - model_loaded,
        'device_name': torch.cuda.get_device_name(args.device) if str(args.device).startswith('cuda') else 'CPU',
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith('cuda') else None,
        'boundary': 'Retain complete raw document through decision; tokenize, pad, B1 all windows/tails, H2D/validation, inference, D2H margins, max and strict threshold. Loading, graph setup/qualification and warmups excluded from warm latency.',
        'device_only_boundary': 'Inputs already resident/validated; B1 inference host interval and CUDA event interval include launch gaps, graph static D2D copies and output clone. No tokenization/H2D/D2H/aggregation.',
        'claim_boundary': 'Measured software route only; sample-stratified observations are not deployment-weighted latency or FPGA measurements'}


def cold_observations(args, target_id, documents, out, warm_result):
    results = []
    selected = [min(documents, key=lambda d: len(d['text'])), max(documents, key=lambda d: len(d['text']))]
    for index in range(args.cold_repeats):
        args.budget_check(); document = selected[index % len(selected)]
        destination = out / ('cold_' + target_id + '_' + str(index) + '.json')
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--canonical-root', args.canonical_root,
            '--targets', args.targets, '--target-id', target_id, '--timing-manifest', args.timing_manifest,
            '--thresholds', args.thresholds, '--out', str(destination), '--device', args.device,
            '--cpu-threads', str(args.cpu_threads), '--cold-child', document['document_id'], '--max-seconds', str(args.max_seconds)]
        began = time.perf_counter()
        child = subprocess.run(command, capture_output=True, text=True, timeout=max(1, args.max_seconds - (time.monotonic() - args.started)),
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        elapsed = time.perf_counter() - began
        if child.returncode or not destination.is_file():
            raise RuntimeError('Cold child failed: ' + child.stderr[-2000:])
        result = read(destination); result['process_start_to_exit_seconds'] = elapsed
        expected = next(sample['repetitions'][0] for sample in warm_result['samples'] if sample['document_id'] == document['document_id'])
        actual = result['observation']; integer = warm_result['descriptor']['kind'] == 'cnn_integer'
        if len(actual['window_margins']) != len(expected['window_margins']): raise ValueError('Cold child window membership differs')
        if any(abs(a - b) > (0. if integer else 1e-5 + 1e-5 * abs(b))
               for a, b in zip(actual['window_margins'], expected['window_margins'])) or actual['decision'] != expected['decision']:
            raise ValueError('Cold child changes qualified margins/decision')
        result['qualified_warm_correspondence'] = True
        result['subprocess_command'] = command; results.append(result)
    return results


def self_test():
    import torch
    class Tokenizer:
        is_fast = False
        def __call__(self, text, **kwargs): return {'input_ids': list(range(10, 10 + int(text)))}
        def build_inputs_with_special_tokens(self, content): return [101] + content + [102]
        def num_special_tokens_to_add(self, pair=False): return 2
        def create_token_type_ids_from_sequences(self, content): return [0] * (len(content) + 2)
    cases = []
    for count in (0, 1, 190, 254, 255, 444, 445, 634, 635, 1024):
        pairs, actual_count, spans = encode_document(Tokenizer(), str(count))
        expected = max(1, math.ceil(max(0, count - 254) / 190) + 1)
        assert actual_count == count and len(pairs) == expected and spans[-1][1] == count
        for index, (pair, (lo, hi)) in enumerate(zip(pairs, spans)):
            assert pair['input_ids'].shape == pair['attention_mask'].shape == (1, 256)
            assert pair['input_ids'][0, 1:1 + hi - lo].tolist() == list(range(10 + lo, 10 + hi))
            assert int(pair['attention_mask'].sum()) == hi - lo + 2
            assert not index or lo == spans[index - 1][0] + 190
        cases.append({'content_tokens': count, 'windows': len(pairs), 'last_span': spans[-1]})
    stats = summary([1, 2, 3, 4, 5]); assert stats['mean'] == stats['p50'] == 3.
    for value in ('x/new_confirmation/a', 'x/final_test/a', 'x/NotInject/a'):
        try: allowed_path(value)
        except ValueError: pass
        else: raise AssertionError('Closed input path accepted')
    class Toy:
        def __call__(self, input_ids, attention_mask):
            from types import SimpleNamespace
            margin = (input_ids * attention_mask).float().sum(1) / attention_mask.sum(1)
            return SimpleNamespace(logits=torch.stack((margin * 0, margin), dim=1))
    doc = {'document_id': 'stress2', 'text': '255', 'text_sha256': hashlib.sha256(b'255').hexdigest(), 'timing_only': True}
    proof, predictions = qualify(Toy(), Tokenizer(), [doc], 'cpu', False, 120., reference={})
    observation = observe(Toy(), Tokenizer(), doc, 'cpu', False, 120.)
    assert observation['windows'] == 2 and observation['window_margins'] == predictions['stress2']
    assert observation['decision'] == (max(predictions['stress2']) > 120.) and all(value >= 0 for value in observation['ms'].values())
    return {'passed': True, 'window_cases': cases, 'quantile_check': stats, 'complete_observation_check': observation,
            'direct_reference_qualification': proof, 'cuda_initialized': torch.cuda.is_initialized()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canonical-root'); parser.add_argument('--targets'); parser.add_argument('--target-id')
    parser.add_argument('--timing-manifest'); parser.add_argument('--thresholds'); parser.add_argument('--out')
    parser.add_argument('--device', default='cuda'); parser.add_argument('--cpu-threads', type=int, default=1)
    parser.add_argument('--warmups', type=int, default=3); parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--graph', action='store_true'); parser.add_argument('--reference-scores', action='append', default=[])
    parser.add_argument('--cold-repeats', type=int, default=0); parser.add_argument('--max-seconds', type=float, default=1200)
    parser.add_argument('--cold-child'); parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--make-timing', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        if args.out: write(args.out, result)
        print(json.dumps(result)); return
    if args.make_timing:
        if not args.canonical_root or not args.out: parser.error('--make-timing needs --canonical-root and immutable --out manifest path')
        result = make_timing(args.canonical_root, args.out)
        print(json.dumps({'complete': True, 'manifest': args.out, 'documents': len(result['documents']),
                          'natural_documents': result['natural_sample_documents'], 'stress_documents': result['timing_only_stress_documents']})); return
    if any(getattr(args, key) is None for key in ('canonical_root', 'targets', 'timing_manifest', 'thresholds', 'out')):
        parser.error('--canonical-root, --targets, --timing-manifest, --thresholds and --out required')
    if args.repeats < 1 or args.warmups < 0 or not 0 <= args.cold_repeats <= 3: parser.error('Invalid bounded repetition counts')
    import torch
    from train import setup
    from evaluate import common, target_descriptors
    args.started = time.monotonic()
    def budget_check():
        if time.monotonic() - args.started > args.max_seconds: raise RuntimeError('Profiling wall-time ceiling exhausted')
    args.budget_check = budget_check
    canonical = Path(args.canonical_root); out = Path(args.out)
    if not args.cold_child and out.exists() and any(out.iterdir()):
        raise ValueError('Profiling output is nonempty; preserve previous observations and choose a new dispatch ID')
    setup(42, args.device, args.cpu_threads)
    contract, contract_sha = common(canonical); documents, sources = timing_documents(args.timing_manifest, canonical)
    target_list = read(args.targets)
    if args.target_id: target_list = {'targets': [item for item in target_list['targets'] if item['id'] == args.target_id]}
    descriptors = target_descriptors(canonical, target_list, contract)
    thresholds = read(args.thresholds); thresholds = thresholds.get('targets', thresholds)
    references, reference_sha = {}, {}
    for binding in args.reference_scores:
        ident, path = binding.split('=', 1); rows = read_rows(path)
        if any(row.get('target_id', ident) != ident for row in rows): raise ValueError('Reference target mismatch')
        references[ident] = {row['document_id']: row for row in rows}; reference_sha[ident] = digest(path)
    results = {}
    for ident, descriptor in descriptors.items():
        args.budget_check()
        threshold = thresholds[ident]
        if isinstance(threshold, dict): threshold = threshold['threshold_margin']
        if not isinstance(threshold, (float, int)) or not math.isfinite(threshold): raise ValueError('Frozen finite margin threshold required')
        if str(args.device).startswith('cuda'): torch.cuda.reset_peak_memory_stats(args.device)
        results[ident] = profile_target(args, ident, descriptor, contract, documents, float(threshold), references.get(ident))
        if args.cold_child:
            write(out, results[ident]); print(json.dumps({'complete': True, 'cold_child': args.cold_child})); return
        # Release model ownership inside profile_target before fresh-process loads.
        if str(args.device).startswith('cuda'): torch.cuda.empty_cache()
        if args.cold_repeats: results[ident]['cold_observations'] = cold_observations(args, ident, documents, out, results[ident])
        write(out / (ident + '.json'), results[ident])
    result = {'complete': True, 'targets': results, 'common_contract_sha256': contract_sha,
        'timing_sources_sha256': sources, 'targets_sha256': digest(args.targets), 'thresholds_sha256': digest(args.thresholds),
        'reference_scores_sha256': reference_sha, 'profiler_sha256': digest(Path(__file__)),
        'source_sha256': {name: digest(Path(__file__).with_name(name)) for name in ('evaluate.py', 'models.py', 'numerical.py', 'teacher_cache.py')},
        'policy': {'warmups': args.warmups, 'repeats': args.repeats, 'batch_windows': 1, 'cold_repeats': args.cold_repeats,
                   'cpu_threads': args.cpu_threads, 'graph_requested': args.graph, 'retains_document_until_decision': True,
                   'gpu_exclusivity': 'Coordinator must hold shared gpu0.lock throughout; no self-scheduling or concurrent sibling timings'},
        'wall_seconds': time.monotonic() - args.started, 'python': sys.version, 'torch': torch.__version__, 'command': sys.argv}
    write(out / 'complete.json', result); print(json.dumps({'complete': True, 'out': str(out), 'targets': list(results)}))


if __name__ == '__main__': main()
