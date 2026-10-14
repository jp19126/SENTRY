"""Matched complete-raw-document profiles; root alone dispatches inference.

Warm profiles include tokenization/padding, transfers/static copies, all windows,
network/softmax, return/max and decision. CUDA graphs require full saved-role
decision qualification. Cold mode uses fresh child processes without cache flush.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

PROCESS_STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'results/cnn_adoption_v1'
STAGES = ('preprocessing', 'h2d', 'network', 'return_aggregation', 'complete')
LOGIT_ATOL = LOGIT_RTOL = RISK_ATOL = 1e-5


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): result.update(block)
    return result.hexdigest()


def select_documents(documents, per_stratum=2, stress_per_label=2):
    """Score-independent fixed sampling, shared by every model and numerical route."""
    strata = {}
    for row in documents:
        windows = row.get('windows', row.get('window_count'))
        if windows is None: raise ValueError('Prepared window count missing')
        window_bin = '1' if windows == 1 else '2-3' if windows < 4 else '4-7' if windows < 8 else '8+'
        key = (row['genre'], row['label'], window_bin)
        strata.setdefault(key, []).append(row)
    chosen, reasons = {}, {}
    for key, rows in sorted(strata.items()):
        ranked = sorted(rows, key=lambda r: hashlib.sha256(('424242|' + r['document_id']).encode()).hexdigest())
        for row in ranked[:per_stratum]:
            chosen[row['document_id']] = row
            reasons.setdefault(row['document_id'], []).append('stratum:' + '|'.join(map(str, key)))
    for label in (0, 1):
        ranked = sorted([r for r in documents if r['label'] == label], key=lambda r: (-r.get('utf8_bytes', 0), r['document_id']))
        for row in ranked[:stress_per_label]:
            chosen[row['document_id']] = row; reasons.setdefault(row['document_id'], []).append('long_stress_label:' + str(label))
    selected = [chosen[key] for key in sorted(chosen)]
    if not selected: raise ValueError('Empty profiling sample')
    return selected, reasons, {'|'.join(map(str, k)): len(v) for k, v in sorted(strata.items())}


def policy_for(args):
    return {'seed': 424242, 'strata': ['genre', 'label', 'window_count:1/2-3/4-7/8+'],
            'sample_per_stratum': args.sample_per_stratum, 'long_stress_per_label': args.long_stress_per_label,
            'warmups': args.warmups, 'repeats': args.repeats, 'batch_windows': 1,
            'graph_logit_atol': LOGIT_ATOL, 'graph_logit_rtol': LOGIT_RTOL, 'risk_atol': RISK_ATOL,
            'decision_changes_allowed': 0, 'cpu_threads': args.cpu_threads,
            'cold_documents': args.cold_documents, 'cold_selection': 'shortest and longest shared warm-sample documents'}


def cpu_checkpoint(path):
    """Lazy ZIP storage mapping: unused optimizer tensors are never materialized.

    Older non-ZIP checkpoints use the explicit legacy CPU reader. File identity
    verification still reads every byte and is measured separately.
    """
    import torch
    mapped = zipfile.is_zipfile(path)
    began = time.perf_counter()
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=mapped)
    return state, {'storage_reader': 'cpu_zip_mmap' if mapped else 'legacy_cpu_eager',
                   'mmap': mapped, 'metadata_parse_seconds': time.perf_counter() - began}


def resolve_profile_target(plan, run_id=None, reference_id=None, numerical_format=None):
    """Same qualified target guards; FP32 checkpoint metadata uses lazy mapping."""
    from cnn_adoption_train import identity
    from cnn_adoption_evaluate import resolve_target, path, relative
    if bool(run_id) == bool(reference_id): raise ValueError('Exactly one run/reference required')
    identifier = run_id or reference_id
    if Path(identifier).name != identifier or identifier in ('.', '..'): raise ValueError('Unsafe profiling ID')
    specs = plan.get('model_specs', plan.get('models', {}))
    if run_id:
        choices = [r for r in plan['runs'] if r['run_id'] == run_id]
        if len(choices) != 1: raise ValueError('Run absent/duplicated in plan')
        recipe = choices[0]; spec = specs[recipe['model_id']]
    else:
        reference = plan['references'][reference_id]
        spec = reference.get('model_spec') or specs[reference['model_id']]
    fmt = numerical_format or ('historical_w8_qat' if spec['kind'] == 'historical_bert_w8' else 'fp32')
    if fmt != 'fp32' or spec['kind'] not in ('cnn', 'bert'):
        return resolve_target(plan, run_id, reference_id, numerical_format), {'resolver': 'existing_evaluator'}
    began = time.perf_counter()
    if run_id:
        folder = DEST / 'runs' / identifier
        run, done = read_json(folder / 'run.json'), read_json(folder / 'complete.json')
        if not done['complete'] or run['association']['recipe'] != recipe or run['association']['model_spec'] != spec:
            raise ValueError('Completed training differs from exact planned recipe/model')
        if done['association_sha256'] != run['association_sha256'] or identity(run['association']) != run['association_sha256']:
            raise ValueError('Training association changed')
        checkpoint = path(done['checkpoint'])
        hash_began = time.perf_counter(); checkpoint_hash = digest(checkpoint); hash_seconds = time.perf_counter() - hash_began
        if checkpoint_hash != done['checkpoint_sha256']: raise ValueError('Checkpoint bytes changed')
        checkpoint_identity = {'path': relative(checkpoint), 'sha256': checkpoint_hash,
                               'training_association_sha256': done['association_sha256']}
        stored, storage = cpu_checkpoint(checkpoint)
        if stored['association_sha256'] != done['association_sha256'] or stored['recipe'] != recipe or stored['model_spec'] != spec or stored['epoch'] != done['selected_epoch']:
            raise ValueError('Checkpoint payload differs from recipe/model/epoch association')
        del stored
        if run['association']['sources']['cnn_models'] != digest(ROOT / 'scripts/cnn_models.py'):
            raise ValueError('Model implementation changed since training')
        if spec['kind'] == 'bert' and run['association']['pretrained_policy'] != read_json(ROOT / 'configs/project.json')['models']['detector']:
            raise ValueError('Pretrained model policy changed since training')
    else:
        if spec['kind'] != 'cnn' or not reference.get('checkpoint'): raise ValueError('Explicit frozen CNN checkpoint required')
        checkpoint = path(reference['checkpoint'])
        hash_began = time.perf_counter(); checkpoint_hash = digest(checkpoint); hash_seconds = time.perf_counter() - hash_began
        if checkpoint_hash != reference['checkpoint_sha256']: raise ValueError('Frozen reference checkpoint changed')
        checkpoint_identity = {'path': relative(checkpoint), 'sha256': checkpoint_hash}
        storage = {'storage_reader': 'metadata_validated_during_restore'}
    descriptor = {'model_spec': spec, 'checkpoint_identity': checkpoint_identity,
                  'numerical_format': 'fp32', 'export_manifest_sha256': None}
    return (identifier, descriptor, checkpoint, None, None), {'resolver': 'profiler_fp32_lazy_checkpoint_metadata',
        'checkpoint_hash_seconds': hash_seconds, 'total_validation_seconds': time.perf_counter() - began, **storage}


def load_profile_target(descriptor, checkpoint, exported, device):
    """Direct FP32 inference restore; all other numerical routes retain loaders."""
    from cnn_adoption_evaluate import load_target
    from cnn_adoption_train import cnn_config, identity
    import torch
    spec = descriptor['model_spec']; kind = spec['kind']
    if descriptor['numerical_format'] != 'fp32' or kind not in ('cnn', 'bert'):
        return load_target(descriptor, checkpoint, exported, device)
    if exported is not None or checkpoint is None: raise ValueError('Qualified FP32 checkpoint required')
    began = time.perf_counter(); hash_began = began
    actual_hash = digest(checkpoint); hash_seconds = time.perf_counter() - hash_began
    if actual_hash != descriptor['checkpoint_identity']['sha256'] or spec.get('initial_checkpoint_sha256', actual_hash) != actual_hash:
        raise ValueError('Qualified FP32 checkpoint bytes changed')
    stored, storage = cpu_checkpoint(checkpoint)
    if stored.get('model_spec') and stored['model_spec'] != spec: raise ValueError('Checkpoint model specification differs')
    expected_association = descriptor['checkpoint_identity'].get('training_association_sha256')
    if expected_association and stored['association_sha256'] != expected_association: raise ValueError('Checkpoint training association differs')
    construction = time.perf_counter()
    if kind == 'cnn':
        from cnn_models import build_model
        config = spec.get('config') or cnn_config(spec['config_id'])
        if stored.get('config') and stored['config'] != config: raise ValueError('Checkpoint CNN topology differs')
        model = build_model(config, spec.get('vocab_size', 30522), spec.get('pad_id', 0))
        config_identity = {'configuration_sha256': identity(config)}
        attention = None; library = None
    else:
        from transformers import BertConfig, BertForSequenceClassification, __version__ as library
        from transformers.models.bert.modeling_bert import BertSdpaSelfAttention
        policy = read_json(ROOT / 'configs/project.json')['models']['detector']
        snapshot = ROOT / 'checkpoints/huggingface' / ('models--' + policy['id'].replace('/', '--')) / 'snapshots' / policy['revision']
        config_path = snapshot / 'config.json'
        run_metadata = read_json(checkpoint.parent / 'run.json')
        if run_metadata['association_sha256'] != expected_association or digest(config_path) != run_metadata['initialization']['pretrained_files_sha256']['config.json']:
            raise ValueError('Pinned local BERT configuration differs from qualified training provenance')
        config = BertConfig.from_pretrained(snapshot, local_files_only=True, num_labels=2)
        # Current qualified AutoModel loader uses SDPA, confirmed by CPU proof.
        config._attn_implementation = 'sdpa'
        model = BertForSequenceClassification(config)
        if model.config._attn_implementation != 'sdpa' or not all(isinstance(layer.attention.self, BertSdpaSelfAttention) for layer in model.bert.encoder.layer):
            raise ValueError('Direct BERT constructor did not retain qualified SDPA attention')
        attention = {'implementation': model.config._attn_implementation,
                     'classes': [type(layer.attention.self).__name__ for layer in model.bert.encoder.layer]}
        config_identity = {'model_id': policy['id'], 'revision': policy['revision'],
            'path': str(config_path.relative_to(ROOT)).replace('\\', '/'), 'file_sha256': digest(config_path),
            'effective_configuration_sha256': identity(config.to_dict()), 'num_labels': model.config.num_labels}
    constructed = time.perf_counter()
    weights = stored['state_dict']; del stored
    model.load_state_dict(weights, strict=True)
    del weights
    restored = time.perf_counter(); model = model.float().to(device).eval()
    if str(device).startswith('cuda'): torch.cuda.synchronize()
    finished = time.perf_counter()
    metadata = {'route': 'native_fp32_tf32_disabled', 'inference_loader': 'direct_exact_state_no_pretrained_weight_load_v1',
        'checkpoint_sha256': actual_hash, 'configuration_identity': config_identity, 'attention': attention,
        'transformers': library, **storage, 'optimizer_tensors_transferred': False, 'pretrained_base_weights_loaded': False,
        'strict_state_restore': True, 'timings': {'checkpoint_hash_seconds': hash_seconds,
            'construct_seconds': constructed - construction, 'state_restore_seconds': restored - constructed,
            'device_transfer_seconds': finished - restored, 'total_restore_seconds': finished - began}}
    return model, metadata


class CNNGraph:
    def __init__(self, model, first, torch):
        self.torch = torch; began = time.perf_counter()
        self.inputs = {key: value.to('cuda').clone() for key, value in first.items()}
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(side):
            for _ in range(3): model(**self.inputs).logits.softmax(1)[:, 1]
        torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.logits = model(**self.inputs).logits; self.risk = self.logits.softmax(1)[:, 1].max()
        torch.cuda.synchronize()
        self.setup = {'seconds': time.perf_counter() - began, 'capture_warmup_forwards': 3,
                      'shape': [1, 256], 'mask': 'unchanged CNN two-dimensional mask', 'configuration_count': 1}

    def copy_inputs(self, inputs):
        for key, value in inputs.items():
            if tuple(value.shape) != (1, 256): raise ValueError('Only static B1/L256 graph shape supported')
            self.inputs[key].copy_(value)

    def replay(self):
        self.graph.replay(); return self.logits, self.risk


def context(args):
    """Deferred heavy imports let cold child startup include framework imports."""
    import numpy as np
    import torch
    from cnn_adoption_train import Budget, identity, load_prepared, setup, write
    from cnn_adoption_evaluate import (path, confirmation_authorization, verify_membership)
    from gate_eval import load_config, window_document
    from r1_challengers import tokenizer
    imports_finished = time.perf_counter()
    setup(42, args.device, args.cpu_threads); budget = Budget(args.max_seconds)
    if args.backend == 'cuda_graph' and args.device != 'cuda': raise ValueError('CUDA graph requires CUDA device')
    plan = read_json(args.plan)
    resolved, checkpoint_validation = resolve_profile_target(plan, args.run_id, args.reference_id, args.format)
    identifier, descriptor, checkpoint, exported, _ = resolved
    fmt = descriptor['numerical_format']; target_key = identifier + ':' + fmt
    policy = policy_for(args)
    frozen = confirmation_authorization(plan, target_key, descriptor) if args.stage == 'confirmation' else None
    prepared = path(plan['prepared_root']); role = 'confirmation' if frozen else 'development_selection'
    if not prepared.resolve().is_relative_to(DEST.resolve()): raise ValueError('Prepared revision must remain inside adoption results')
    if frozen:
        if frozen.get('profiling_policy') != policy or frozen.get('profiling_policy_sha256') != identity(policy):
            raise ValueError('Profiling policy was not fixed before confirmation')
        if frozen['source_sha256'].get('scripts/cnn_adoption_profile.py') != digest(Path(__file__)):
            raise ValueError('Profiler source was not frozen')
        if args.backend == 'cuda_graph' and descriptor['model_spec']['kind'] != 'cnn' and frozen['source_sha256'].get('scripts/r1_bert_cuda_graph.py') != digest(ROOT / 'scripts/r1_bert_cuda_graph.py'):
            raise ValueError('BERT graph helper source was not frozen')
        for name in ('benign_calibration', 'confirmation'): verify_membership(prepared / name, frozen['data_membership'][name])
    evaluated_folder = DEST / 'evaluation' / args.stage / identifier / fmt
    evaluated = read_json(evaluated_folder / 'complete.json')
    if not evaluated['complete'] or evaluated['association']['target'] != descriptor:
        raise ValueError('Matching completed evaluation required before profiling')
    if evaluated['association']['batch_windows'] != 1: raise ValueError('Reference scoring must be B1')
    if frozen and evaluated['association']['confirmation_freeze_sha256'] != plan['confirmation_freeze_sha256']:
        raise ValueError('Reference confirmation has another freeze')
    for filename, expected in evaluated['score_files_sha256'].items():
        if digest(evaluated_folder / filename) != expected: raise ValueError('Reference score bytes changed')
    threshold = evaluated['metrics']['0.005']['threshold']
    data = load_prepared(prepared / role, role, release_confirmation=bool(frozen))
    if data['identity'] != evaluated['association']['data'][role]: raise ValueError('Reference prepared identity changed')
    sample, reasons, population = select_documents(data['documents'], args.sample_per_stratum, args.long_stress_per_label)
    manifest = read_json(prepared / role / 'manifest.json'); raw_path = path(plan.get('raw_root', 'results/cnn_adoption_v1/raw')) / (role + '.jsonl')
    if not raw_path.resolve().is_relative_to(DEST.resolve()): raise ValueError('Raw revision must remain inside adoption results')
    if digest(raw_path) != manifest['raw_sha256']: raise ValueError('Raw profiling source differs from prepared data')
    raw = {r['document_id']: r for r in [json.loads(line) for line in raw_path.read_text(encoding='utf-8').splitlines() if line.strip()]}
    if len(raw) != len(data['documents']): raise ValueError('Raw/prepared role membership differs')
    for row in data['documents']:
        original = raw[row['document_id']]
        if original['text_sha256'] != row['text_sha256'] or hashlib.sha256(original['text'].encode()).hexdigest() != row['text_sha256']:
            raise ValueError('Raw/prepared text identity differs')
    config = load_config()
    if config['windowing']['primary_length'] != 256 or config['windowing']['content_token_overlap'] != 64:
        raise ValueError('Current raw preprocessing policy changed')
    if config['models']['detector']['id'] != manifest['tokenizer']['name'] or config['models']['detector']['tokenizer_revision'] != manifest['tokenizer']['revision']:
        raise ValueError('Current tokenizer policy differs from prepared reference')
    validated = time.perf_counter(); load_started = validated
    model, backend_record = load_profile_target(descriptor, checkpoint, exported, args.device)
    if args.device == 'cuda': torch.cuda.synchronize()
    loaded = time.perf_counter(); tok = tokenizer(config); tokenizer_ready = time.perf_counter()

    def prepare(document):
        # Trusted span positions and labels are never passed into preprocessing.
        windows = window_document(tok, document['text'], document['document_id'], config=config)
        return [{'input_ids': torch.tensor([w['input_ids'] + [0] * (256 - len(w['input_ids']))], dtype=torch.long),
                 'attention_mask': torch.tensor([w['attention_mask'] + [0] * (256 - len(w['attention_mask']))], dtype=torch.long)} for w in windows]

    graph = None; graph_setup = None
    if args.backend == 'cuda_graph':
        first_id = args.cold_child_document or sample[0]['document_id']
        if first_id not in {r['document_id'] for r in sample}: raise ValueError('Graph initialization must use fixed sample')
        first = prepare(raw[first_id])[0]
        if hasattr(model, 'bert'):
            if model.config.num_hidden_layers != 4: raise ValueError('BERT graph helper is fixed to four-layer BERT')
            from r1_bert_cuda_graph import GraphBackend
            graph = GraphBackend(model, first, torch)
            graph_setup = {**graph.setup, 'numerical_route_note': 'BERT always-additive4D SDPA mask differs from ordinary 2D-mask dispatch; full saved-role decisions must match'}
        else:
            graph = CNNGraph(model, first, torch); graph_setup = graph.setup
    association = {'target': descriptor, 'stage': args.stage, 'backend': args.backend, 'device': args.device,
                   'torch': torch.__version__, 'policy': policy, 'raw_role_sha256': digest(raw_path),
                   'prepared_identity': data['identity'], 'evaluation_sha256': digest(evaluated_folder / 'complete.json'),
                   'threshold': threshold, 'sample_document_ids': [r['document_id'] for r in sample],
                   'profiler_source_sha256': digest(Path(__file__)),
                   'graph_helper_sha256': digest(ROOT / 'scripts/r1_bert_cuda_graph.py') if graph and hasattr(model, 'bert') else None,
                   'inference_loader': backend_record.get('inference_loader', 'existing_numerical_loader'),
                   'loader_configuration_identity': backend_record.get('configuration_identity'),
                   'cpu_threads': args.cpu_threads,
                   'cache_limit': os.environ.get('TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT'),
                   'cudnn_v8_api_disabled': os.environ.get('TORCH_CUDNN_V8_API_DISABLED', '0')}
    folder = DEST / 'profiling' / args.stage / identifier / fmt / args.backend / ('threads' + str(args.cpu_threads) if args.device == 'cpu' else 'cuda')
    return locals()


def synchronize(c):
    if c['args'].device == 'cuda': c['torch'].cuda.synchronize()


def qualify(c):
    """All saved cal+scoring windows; no new thresholds or label-dependent inputs."""
    torch, np = c['torch'], c['np']; output = []; maximum_logit_delta = maximum_risk_delta = 0.
    roles = ('benign_calibration', 'confirmation') if c['frozen'] else ('development_calibration', 'development_selection')
    for role in roles:
        data = c['load_prepared'](c['prepared'] / role, role, release_confirmation=bool(c['frozen']))
        if data['identity'] != c['evaluated']['association']['data'][role]: raise ValueError('Qualification data differs from saved evaluation')
        saved = [json.loads(line) for line in (c['evaluated_folder'] / (role + '_scores.jsonl')).read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(saved) != len(data['documents']): raise ValueError('Qualification membership differs')
        risks = []; failures = []
        with torch.inference_mode():
            for index in range(len(data['ids'])):
                c['budget'].check()
                pair = {key: torch.as_tensor(np.array(data[array][index:index + 1], dtype=np.int64), device=c['args'].device)
                        for key, array in (('input_ids', 'ids'), ('attention_mask', 'masks'))}
                eager = c['model'](**pair).logits
                actual = eager
                if c['graph']:
                    c['graph'].copy_inputs(pair); actual, _ = c['graph'].replay()
                    delta = float((actual - eager).abs().max()); maximum_logit_delta = max(maximum_logit_delta, delta)
                    if not torch.allclose(actual, eager, atol=LOGIT_ATOL, rtol=LOGIT_RTOL): failures.append({'window': index, 'kind': 'graph_logits', 'absolute_delta': delta})
                if not torch.isfinite(actual).all(): raise ValueError('Nonfinite profiling logits')
                risks.append(float(actual.softmax(1)[0, 1]))
        changes = []
        for di, (row, document) in enumerate(zip(saved, data['documents'])):
            if row['document_id'] != document['document_id'] or row['text_sha256'] != document['text_sha256']:
                raise ValueError('Saved qualification association differs')
            lo, hi = map(int, data['starts'][di:di + 2]); current = risks[lo:hi]
            if len(current) != len(row['window_scores']): raise ValueError('Window membership differs')
            delta = max(abs(left - right) for left, right in zip(current, row['window_scores']))
            maximum_risk_delta = max(maximum_risk_delta, delta)
            if delta > RISK_ATOL: failures.append({'document_id': row['document_id'], 'kind': 'saved_window_risk', 'absolute_delta': delta})
            if (max(current) > c['threshold']) != (row['risk_score'] > c['threshold']): changes.append(row['document_id'])
        output.append({'role': role, 'windows': len(risks), 'documents': len(saved), 'failures': failures, 'changed_decision_ids': changes})
    return {'passed': all(not r['failures'] and not r['changed_decision_ids'] for r in output),
            'logit_atol': LOGIT_ATOL, 'logit_rtol': LOGIT_RTOL, 'saved_risk_atol': RISK_ATOL,
            'maximum_logit_absolute_delta': maximum_logit_delta, 'maximum_saved_risk_absolute_delta': maximum_risk_delta, 'roles': output,
            'scope': 'Fixed primary threshold decisions and bounded window risks on all existing scored calibration/development roles; no recalibration or universal arithmetic equivalence'}


def observe(c, document):
    torch = c['torch']; began = time.perf_counter(); pairs = c['prepare'](document); prepared_at = time.perf_counter()
    transfer = network = returned = 0.; risk = 0.
    with torch.inference_mode():
        for pair in pairs:
            at = time.perf_counter()
            if c['graph']:
                c['graph'].copy_inputs(pair)
            else: pair = {key: value.to(c['args'].device) for key, value in pair.items()}
            synchronize(c); transferred = time.perf_counter()
            if c['graph']: _, value = c['graph'].replay()
            else: value = c['model'](**pair).logits.softmax(1)[0, 1]
            synchronize(c); forwarded = time.perf_counter()
            risk = max(risk, float(value)); finished = time.perf_counter()
            transfer += transferred - at; network += forwarded - transferred; returned += finished - forwarded
    decision = risk > c['threshold']; end = time.perf_counter()
    return {'ms': dict(zip(STAGES, [1000 * value for value in (prepared_at - began, transfer, network, returned, end - began)])),
            'risk': risk, 'decision': decision, 'windows': len(pairs)}


def warm(c):
    file = c['folder'] / 'warm_complete.json'
    if file.exists():
        prior = read_json(file)
        if prior['association'] != c['association']: raise ValueError('Existing profile identity differs; preserve it')
        return prior
    qualification = qualify(c)
    c['folder'].mkdir(parents=True, exist_ok=True)
    if not qualification['passed']:
        c['write'](c['folder'] / 'qualification_failure.json', {'association': c['association'], 'qualification': qualification})
        raise ValueError('Profiling route fails saved decision/logit qualification; no fallback')
    saved_role = [json.loads(line) for line in (c['evaluated_folder'] / (c['role'] + '_scores.jsonl')).read_text(encoding='utf-8').splitlines() if line.strip()]
    reference = {r['document_id']: r for r in saved_role}; observations = []
    for selected in c['sample']:
        document = c['raw'][selected['document_id']]
        # One untimed preparation audit ensures every raw tail matches the stored IDs/masks.
        di = next(i for i, r in enumerate(c['data']['documents']) if r['document_id'] == selected['document_id'])
        lo, hi = map(int, c['data']['starts'][di:di + 2]); pairs = c['prepare'](document)
        if len(pairs) != hi - lo: raise ValueError('Raw profiling tail coverage differs')
        for offset, pair in enumerate(pairs):
            if not c['np'].array_equal(pair['input_ids'].numpy()[0], c['data']['ids'][lo + offset]) or not c['np'].array_equal(pair['attention_mask'].numpy()[0], c['data']['masks'][lo + offset]):
                raise ValueError('Raw profiling token inputs differ from saved quality inputs')
        for _ in range(c['args'].warmups): observe(c, document)
        repetitions = []
        for _ in range(c['args'].repeats):
            c['budget'].check(); measured = observe(c, document); expected = reference[document['document_id']]['risk_score']
            if abs(measured['risk'] - expected) > RISK_ATOL or measured['decision'] != (expected > c['threshold']):
                raise ValueError('Timed raw decision differs from quality reference')
            repetitions.append(measured)
        array = c['np'].asarray([[r['ms'][key] for key in STAGES] for r in repetitions])
        observations.append({'document_id': document['document_id'], 'text_sha256': document['text_sha256'],
            'genre': selected['genre'], 'label': selected['label'], 'utf8_bytes': len(document['text'].encode()),
            'windows': len(pairs), 'selection_reasons': c['reasons'][document['document_id']],
            'mean_ms': dict(zip(STAGES, array.mean(0).tolist())), 'complete_p50_ms': float(c['np'].quantile(array[:, 4], .5)),
            'complete_p95_ms': float(c['np'].quantile(array[:, 4], .95)), 'repetitions': repetitions})
    result = {'complete': True, 'association': c['association'], 'backend': c['backend_record'], 'graph_setup': c['graph_setup'],
              'qualification': qualification, 'population_strata': c['population'], 'samples': observations,
              'sample_count': len(observations), 'timed_document_repetitions': len(observations) * c['args'].repeats,
              'model_load_seconds': c['loaded'] - c['load_started'], 'tokenizer_load_seconds': c['tokenizer_ready'] - c['loaded'],
              'checkpoint_validation': c['checkpoint_validation'],
              'wall_seconds': time.monotonic() - c['budget'].started, 'memory_observations': c['budget'].observations,
              'boundary': 'Complete raw text to fixed-threshold document decision; serial B1 windows, explicit stage synchronization. Loading/capture/qualification excluded from warm latency. Sample is stratified/stress-selected, not a workload-frequency estimate.'}
    c['write'](file, result); return result


def cold_parent(args):
    # No Torch/framework import in parent; each observation pays fresh child setup.
    plan = read_json(args.plan); identifier = args.run_id or args.reference_id
    specs = plan['model_specs']
    if args.reference_id: entry = plan['references'][identifier]; spec = entry.get('model_spec') or specs[entry['model_id']]
    else: spec = specs[next(r for r in plan['runs'] if r['run_id'] == identifier)['model_id']]
    fmt = args.format or ('historical_w8_qat' if spec['kind'] == 'historical_bert_w8' else 'fp32')
    folder = DEST / 'profiling' / args.stage / identifier / fmt / args.backend / ('threads' + str(args.cpu_threads) if args.device == 'cpu' else 'cuda')
    warmed = read_json(folder / 'warm_complete.json'); output = folder / 'cold_process_complete.json'
    if output.exists():
        previous = read_json(output)
        if previous['warm_profile_sha256'] != digest(folder / 'warm_complete.json'): raise ValueError('Existing cold profile has another warm identity')
        return previous
    ranked = sorted(warmed['samples'], key=lambda r: (r['utf8_bytes'], r['document_id']))
    selected = [ranked[0]] if args.cold_documents == 1 else [ranked[0], ranked[-1]]
    records = []
    for sample in selected:
        argv = [sys.executable, '-B', str(Path(__file__)), '--plan', args.plan,
                '--run-id' if args.run_id else '--reference-id', identifier, '--stage', args.stage,
                '--format', fmt, '--device', args.device, '--backend', args.backend, '--mode', 'cold',
                '--sample-per-stratum', str(args.sample_per_stratum), '--long-stress-per-label', str(args.long_stress_per_label),
                '--warmups', str(args.warmups), '--repeats', str(args.repeats), '--cpu-threads', str(args.cpu_threads),
                '--cold-documents', str(args.cold_documents), '--max-seconds', str(args.max_seconds), '--cold-child-document', sample['document_id']]
        start = time.perf_counter(); child = subprocess.run(argv, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True, timeout=args.max_seconds)
        if child.returncode: raise RuntimeError('Cold child failed: ' + child.stderr[-3000:])
        record = json.loads(child.stdout.strip().splitlines()[-1]); record['parent_spawn_to_exit_seconds'] = time.perf_counter() - start
        if record['decision'] != sample['repetitions'][-1]['decision'] or abs(record['risk'] - sample['repetitions'][-1]['risk']) > RISK_ATOL:
            raise ValueError('Cold decision differs from warm reference')
        record['command'] = argv; records.append(record)
    result = {'complete': True, 'warm_profile_sha256': digest(folder / 'warm_complete.json'), 'records': records,
              'note': 'Fresh process startup observations, shortest/longest common sample only; system/file caches not flushed; no cold-boot distribution or deployment startup guarantee.'}
    temporary = output.with_suffix('.json.tmp'); temporary.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8'); os.replace(temporary, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan', required=True)
    group = parser.add_mutually_exclusive_group(required=True); group.add_argument('--run-id'); group.add_argument('--reference-id')
    parser.add_argument('--stage', choices=('development', 'confirmation'), default='development')
    parser.add_argument('--format', choices=('fp32', 'w8a8_export', 'historical_w8_qat'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda'); parser.add_argument('--backend', choices=('eager', 'cuda_graph'), default='eager')
    parser.add_argument('--mode', choices=('warm', 'cold'), default='warm'); parser.add_argument('--cpu-threads', type=int, choices=(1, 4), default=1)
    parser.add_argument('--sample-per-stratum', type=int, default=2); parser.add_argument('--long-stress-per-label', type=int, default=2)
    parser.add_argument('--warmups', type=int, default=3); parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--cold-documents', type=int, choices=(1, 2), default=2); parser.add_argument('--cold-child-document')
    parser.add_argument('--max-seconds', type=int, default=1800); args = parser.parse_args()
    if min(args.sample_per_stratum, args.repeats) < 1 or min(args.warmups, args.long_stress_per_label) < 0:
        raise ValueError('Invalid profiling counts')
    if args.mode == 'cold' and not args.cold_child_document:
        result = cold_parent(args); print(json.dumps({'complete': True, 'cold_processes': len(result['records'])})); return
    try:
        c = context(args)
    except Exception as error:
        # Preserve bounded backend bring-up failures without an eager fallback.
        failure_root = DEST / 'profiling' / 'attempts'; failure_root.mkdir(parents=True, exist_ok=True)
        filename = hashlib.sha256(json.dumps(vars(args), sort_keys=True).encode()).hexdigest()[:16] + '.json'
        target = failure_root / filename
        if not target.exists(): target.write_text(json.dumps({'complete': False, 'arguments': vars(args), 'error': str(error),
            'profiler_source_sha256': digest(Path(__file__)), 'wall_seconds': time.perf_counter() - PROCESS_STARTED}, indent=2) + '\n', encoding='utf-8')
        raise
    if args.cold_child_document:
        if args.cold_child_document not in c['association']['sample_document_ids']: raise ValueError('Cold document absent from fixed warm sample')
        warm_record = read_json(c['folder'] / 'warm_complete.json')
        if warm_record['association'] != c['association']: raise ValueError('Cold/warm target association differs')
        measured = observe(c, c['raw'][args.cold_child_document]); finished = time.perf_counter()
        print(json.dumps({'document_id': args.cold_child_document, **measured,
                          'imports_seconds': c['imports_finished'] - PROCESS_STARTED,
                          'validation_and_setup_seconds': c['validated'] - c['imports_finished'],
                          'model_load_seconds': c['loaded'] - c['load_started'],
                          'checkpoint_validation': c['checkpoint_validation'], 'inference_loader': c['backend_record'],
                          'tokenizer_load_seconds': c['tokenizer_ready'] - c['loaded'],
                          'graph_setup': c['graph_setup'], 'child_start_to_decision_seconds': finished - PROCESS_STARTED})); return
    result = warm(c); print(json.dumps({'complete': True, 'sample_count': result['sample_count'], 'timed_document_repetitions': result['timed_document_repetitions']}))


if __name__ == '__main__': main()
