"""Shared CNN/small-pretrained-BERT adoption training; serial dispatch by coordinator.

Baseline: BCE on the maximum binary logit margin over ALL document windows.
Two-pass replay retains only one microbatch graph while preserving dropout RNG.
Known-span window CE is an explicitly separate objective, never a label fallback.
Historical saved W8-QAT inference is separately identified and never trained here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault('TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT', '64')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cnn_models import build_model, load_search_configs
from gate_eval import evaluate_documents, fit_threshold

DEST = ROOT / 'results/cnn_adoption_v1'
TRAIN_ROLES = {'fit'}
DEVELOPMENT_ROLES = {'development_calibration', 'development_selection'}
READ_ROLES = TRAIN_ROLES | DEVELOPMENT_ROLES | {'benign_calibration', 'confirmation', 'diagnostics', 'regression'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def save(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + '.tmp')
    torch.save(value, temporary); os.replace(temporary, path)


def setup(seed, device='cpu', threads=1):
    torch.set_num_threads(threads)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class Budget:
    def __init__(self, seconds, check_memory=True):
        self.started = time.monotonic(); self.seconds = seconds; self.last = self.started
        self.observations = []
        self.memory_check = None
        if check_memory:
            from r1_challengers import resource_check
            self.memory_check = resource_check
            self.observations.append(resource_check(True))

    def check(self):
        now = time.monotonic()
        if (DEST / 'stop.request').exists():
            raise RuntimeError('Adoption stop requested')
        if now - self.started > self.seconds:
            raise RuntimeError('Adoption dispatch wall-time ceiling exhausted')
        if self.memory_check and now - self.last >= 60:
            self.observations.append(self.memory_check()); self.last = now


def cnn_config(config_id):
    registry = {c['id']: c for c in load_search_configs()}
    registry.update(read(ROOT / 'configs/cnn_stage_c_v1.json')['finalist_configs'])
    if config_id not in registry:
        raise ValueError('CNN absent from existing registry/frozen replacement')
    return registry[config_id]


def model_for(spec, device='cpu', checkpoint=None):
    """No network fallback. Returned metadata identifies pretrained/historical roles."""
    kind = spec['kind']
    metadata = {'kind': kind, 'initialization': spec.get('initialization', 'scratch' if kind == 'cnn' else 'pretrained')}
    if kind == 'cnn':
        config = spec.get('config') or cnn_config(spec['config_id'])
        model = build_model(config, spec.get('vocab_size', 30522), spec.get('pad_id', 0))
        metadata['config'] = config
    elif kind == 'bert':
        from transformers import AutoModelForSequenceClassification
        policy = read(ROOT / 'configs/project.json')['models']['detector']
        if metadata['initialization'] != 'pretrained':
            raise ValueError('Matched BERT requires pretrained initialization')
        model, loading = AutoModelForSequenceClassification.from_pretrained(
            policy['id'], revision=policy['revision'], cache_dir=ROOT / 'checkpoints/huggingface',
            num_labels=2, local_files_only=True, output_loading_info=True)
        if set(loading.get('missing_keys', ())) - {'classifier.weight', 'classifier.bias'}:
            raise ValueError('Pretrained encoder has missing weights')
        if loading.get('mismatched_keys') or loading.get('error_msgs'):
            raise ValueError('Pretrained loader reported incompatible weights')
        metadata.update(base_model=policy['id'], revision=policy['revision'], loading=loading,
                        pretrained_config=model.config.to_dict())
        snapshot = ROOT / 'checkpoints/huggingface' / ('models--' + policy['id'].replace('/', '--')) / 'snapshots' / policy['revision']
        metadata['pretrained_files_sha256'] = {path.name: sha(path) for path in snapshot.iterdir() if path.name in ('config.json', 'pytorch_model.bin', 'model.safetensors')}
    elif kind == 'historical_bert_w8':
        if checkpoint is not None:
            raise ValueError('Historical W8 loader owns its original checkpoint')
        from r1_bert_int8 import load_model
        backend = spec.get('backend', 'native_int8')
        if backend not in {'native_int8', 'frozen_quant_emulation'}:
            raise ValueError('Explicit historical backend required')
        loaded = load_model(device, torch, original=backend == 'frozen_quant_emulation')
        model, quantization = loaded[:2]
        metadata.update(checkpoint='checkpoints/quantized/qat_w8_a8', backend=backend,
                        quantization=quantization, training_allowed=False)
        metadata['checkpoint_files_sha256'] = {name: sha(ROOT / metadata['checkpoint'] / name)
                                              for name in ('config.json', 'quantization.json', 'quantized_state.pt')}
        return model.eval(), metadata
    elif kind == 'historical_bert_fp32':
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(ROOT / 'checkpoints/floating/epoch_3', local_files_only=True)
        metadata.update(checkpoint='checkpoints/floating/epoch_3', training_allowed=False)
    else:
        raise ValueError('Unknown model kind')
    initial = checkpoint if checkpoint is not None else spec.get('initial_checkpoint')
    if initial is not None:
        initial = Path(initial); initial = initial if initial.is_absolute() else ROOT / initial
        expected = spec.get('initial_checkpoint_sha256')
        digest = sha(initial)
        if expected and digest != expected:
            raise ValueError('Initial checkpoint bytes changed')
        stored = torch.load(initial, map_location='cpu', weights_only=False)
        if stored.get('model_spec') and stored['model_spec'] != spec:
            raise ValueError('Checkpoint model specification differs')
        if kind == 'cnn' and stored.get('config') and stored['config'] != metadata['config']:
            raise ValueError('Legacy CNN checkpoint config differs')
        model.load_state_dict(stored['state_dict'], strict=True)
        metadata.update(loaded_checkpoint=str(initial), checkpoint_sha256=digest)
    return model.float().to(device), metadata


def load_prepared(folder, role, *, release_confirmation=False):
    if role not in READ_ROLES or role == 'confirmation' and not release_confirmation:
        raise ValueError('Unapproved/unreleased data role')
    folder = Path(folder)
    names = ('ids', 'masks', 'starts')
    data = {name: np.load(folder / (name + '.npy'), mmap_mode='r', allow_pickle=False) for name in names}
    data['documents'] = [json.loads(line) for line in (folder / 'documents.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    ids, masks, starts, documents = (data[key] for key in ('ids', 'masks', 'starts', 'documents'))
    if ids.ndim != 2 or masks.shape != ids.shape or ids.shape[1] != 256:
        raise ValueError('Shared adoption input requires fixed L256 IDs/masks')
    if starts.ndim != 1 or len(starts) != len(documents) + 1 or starts[0] != 0 or starts[-1] != len(ids) or np.any(np.diff(starts) <= 0):
        raise ValueError('Incomplete or empty document window boundaries')
    if np.any((masks != 0) & (masks != 1)) or np.any(masks.sum(1) == 0):
        raise ValueError('Nonbinary or empty token attention masks')
    if not documents or len({d['document_id'] for d in documents}) != len(documents) or any(d['label'] not in (0, 1) for d in documents):
        raise ValueError('Duplicate document IDs or invalid document labels')
    hashes = {name: sha(folder / name) for name in ('ids.npy', 'masks.npy', 'starts.npy', 'documents.jsonl')}
    if (folder / 'labels.npy').exists():
        data['window_labels'] = np.load(folder / 'labels.npy', mmap_mode='r', allow_pickle=False)
        hashes['labels.npy'] = sha(folder / 'labels.npy')
    data.update(folder=folder, role=role, identity={'role': role, 'files': hashes})
    return data


def inputs(data, indices, device):
    return {'input_ids': torch.as_tensor(np.array(data['ids'][indices], dtype=np.int64), device=device),
            'attention_mask': torch.as_tensor(np.array(data['masks'][indices], dtype=np.int64), device=device)}


def score(model, data, budget=None, batch=128, device=None):
    model.eval(); risk = []
    if device is None:
        representative = next(model.parameters(), None)
        if representative is None: representative = next(model.buffers(), None)
        device = representative.device if representative is not None else torch.device('cpu')
    with torch.inference_mode():
        for at in range(0, len(data['ids']), batch):
            if budget: budget.check()
            logits = model(**inputs(data, slice(at, at + batch), device)).logits
            if not torch.isfinite(logits).all(): raise RuntimeError('Nonfinite scoring logits')
            risk.extend(logits.softmax(1)[:, 1].cpu().tolist())
    output = []
    for index, document in enumerate(data['documents']):
        low, high = map(int, data['starts'][index:index + 2])
        output.append({**document, 'risk_score': max(risk[low:high]), 'window_scores': risk[low:high]})
    return output


def metrics(calibration, selection):
    benign = [d['risk_score'] for d in calibration if d['label'] == 0]
    if not benign: raise ValueError('No benign development calibration documents')
    result = {}
    for alpha in (.005, .01, .02):
        threshold = fit_threshold(benign, alpha)
        metric = evaluate_documents([d['risk_score'] for d in selection], [d['label'] for d in selection], threshold)
        metric['passes'] = metric['recall'] is not None and metric['fpr'] is not None and metric['recall'] >= .99 and metric['fpr'] <= .01
        result[str(alpha)] = metric
    return result


def rng_state(device):
    return (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def restore_rng(state, device):
    torch.set_rng_state(state[0])
    if state[1] is not None: torch.cuda.set_rng_state(state[1], device)


def document_max_backward(model, data, documents, microbatch_windows, *, teacher_margins=None, distillation_weight=0., temperature=2., budget=None):
    """Exact piecewise max-BCE gradient with replayed stochastic microbatches.

    Every window is scouted in train mode. Only winner-containing microbatches
    replay with the same RNG; global RNG is restored after the scout. No model
    state may mutate during forward (the existing CNNs/BERT have no BatchNorm).
    Ties choose the first window. Dropout is never disabled for this objective.
    """
    if not len(documents) or microbatch_windows < 1: raise ValueError('Empty batch/microbatch')
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError('Replay objective requires no mutable BatchNorm forward state')
    if not 0 <= distillation_weight < 1 or temperature <= 0:
        raise ValueError('Invalid distillation weight/temperature')
    device = next(model.parameters()).device
    window_indices, owners = [], []
    for owner, document in enumerate(documents):
        lo, hi = map(int, data['starts'][document:document + 2])
        window_indices.extend(range(lo, hi)); owners.extend([owner] * (hi - lo))
    chunks, best, winners = [], [-math.inf] * len(documents), [None] * len(documents)
    # Use the replay's grad-mode dispatch, but release every scout graph before
    # the next microbatch. No backward occurs until winning chunks replay.
    with torch.enable_grad():
        for offset in range(0, len(window_indices), microbatch_windows):
            if budget: budget.check()
            indices = window_indices[offset:offset + microbatch_windows]
            state = rng_state(device)
            logits = model(**inputs(data, indices, device)).logits
            margins = logits[:, 1] - logits[:, 0]
            if not torch.isfinite(margins).all(): raise RuntimeError('Nonfinite document scout')
            for position, margin in enumerate(margins.detach().cpu().tolist()):
                owner = owners[offset + position]
                if margin > best[owner]: best[owner], winners[owner] = margin, (len(chunks), position)
            chunks.append((indices, state))
            del logits, margins
    terminal_rng = rng_state(device)
    maximum = torch.tensor(best, dtype=torch.float32, device=device, requires_grad=True)
    labels = torch.tensor([data['documents'][d]['label'] for d in documents], dtype=torch.float32, device=device)
    loss = F.binary_cross_entropy_with_logits(maximum, labels)
    if distillation_weight:
        if teacher_margins is None: raise ValueError('Qualified fitting teacher cache required')
        teacher = torch.tensor([teacher_margins[d] for d in documents], dtype=torch.float32, device=device)
        soft = F.binary_cross_entropy_with_logits(maximum / temperature, torch.sigmoid(teacher / temperature)) * temperature ** 2
        loss = (1 - distillation_weight) * loss + distillation_weight * soft
    if not torch.isfinite(loss): raise RuntimeError('Nonfinite document loss')
    coefficients = torch.autograd.grad(loss, maximum)[0].detach()
    selected = {}
    for owner, (chunk, position) in enumerate(winners): selected.setdefault(chunk, []).append((owner, position))
    try:
        for chunk, assignments in selected.items():
            if budget: budget.check()
            indices, state = chunks[chunk]; restore_rng(state, device)
            logits = model(**inputs(data, indices, device)).logits
            margin = logits[:, 1] - logits[:, 0]
            owner_indices = torch.tensor([pair[0] for pair in assignments], device=device)
            positions = torch.tensor([pair[1] for pair in assignments], device=device)
            if not torch.equal(margin.detach()[positions], maximum.detach()[owner_indices]):
                raise RuntimeError('Scout/replay margins differ; exact stochastic gradient not established')
            (margin[positions] * coefficients[owner_indices]).sum().backward()
    finally:
        restore_rng(terminal_rng, device)
    return float(loss.detach()), len(window_indices)


def span_backward(model, data, documents, microbatch_windows, budget=None):
    """Separate reliable-span objective, explicitly selected and validated."""
    if 'window_labels' not in data or np.any((data['window_labels'] != 0) & (data['window_labels'] != 1)):
        raise ValueError('Reliable complete window labels required; unresolved spans cannot use window CE')
    indices, weights = [], []
    for document in documents:
        lo, hi = map(int, data['starts'][document:document + 2])
        indices.extend(range(lo, hi)); weights.extend([1 / (len(documents) * (hi - lo))] * (hi - lo))
    device = next(model.parameters()).device; total = 0.
    for offset in range(0, len(indices), microbatch_windows):
        if budget: budget.check()
        chosen = indices[offset:offset + microbatch_windows]
        logits = model(**inputs(data, chosen, device)).logits
        labels = torch.as_tensor(np.array(data['window_labels'][chosen], dtype=np.int64), device=device)
        weighted = torch.tensor(weights[offset:offset + microbatch_windows], device=device)
        loss = (F.cross_entropy(logits, labels, reduction='none') * weighted).sum()
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite span loss')
        loss.backward(); total += float(loss.detach())
    return total, len(indices)


def teacher_cache(path, fitting, recipe):
    cached = read(path)
    if cached['role'] != 'fit' or cached['fit_documents_sha256'] != fitting['identity']['files']['documents.jsonl']:
        raise ValueError('Teacher cache is not associated with this fitting population')
    for key in ('teacher_checkpoint_sha256', 'qualified_development_record_sha256'):
        if cached[key] != recipe[key]: raise ValueError('Qualified teacher identity changed')
    if cached['document_ids'] != [d['document_id'] for d in fitting['documents']]:
        raise ValueError('Teacher fitting document order differs')
    margins = np.asarray(cached['document_margins'], dtype=np.float32)
    if len(margins) != len(fitting['documents']) or not np.isfinite(margins).all():
        raise ValueError('Teacher cache has missing/nonfinite margins')
    return margins


def train_one(model_spec, recipe, prepared_root, output_root=DEST, budget=None, device='cpu'):
    """Development selection or fixed-epoch final fit; never loads confirmation."""
    recipe = dict(recipe); run_id = recipe['run_id']
    if Path(run_id).name != run_id or run_id in ('.', '..'): raise ValueError('Unsafe run ID')
    if model_spec['kind'] not in {'cnn', 'bert'}: raise ValueError('Historical anchors are inference-only')
    if recipe.get('fit_split', 'fit') not in TRAIN_ROLES: raise ValueError('Only fitting role can train')
    objective = recipe.get('objective', 'document_max_bce')
    if objective not in {'document_max_bce', 'reliable_span_ce'}: raise ValueError('Explicit legal objective required')
    monitoring = recipe.get('monitor_development', True)
    if not monitoring and recipe.get('patience', 0): raise ValueError('Fixed final fit cannot early stop')
    prepared_root, output_root = Path(prepared_root), Path(output_root)
    if not output_root.resolve().is_relative_to(DEST.resolve()): raise ValueError('New artifacts must stay under adoption results')
    if not prepared_root.resolve().is_relative_to((DEST / 'prepared').resolve()):
        raise ValueError('Training input must be explicitly prepared under adoption results')
    fit = load_prepared(prepared_root / recipe.get('fit_split', 'fit'), 'fit')
    calibration = selection = None
    if monitoring:
        calibration = load_prepared(prepared_root / 'development_calibration', 'development_calibration')
        selection = load_prepared(prepared_root / 'development_selection', 'development_selection')
    source_hashes = {'trainer': sha(Path(__file__)), 'cnn_models': sha(ROOT / 'scripts/cnn_models.py'),
                     'gate_eval': sha(ROOT / 'gate_eval.py')}
    association = {'recipe': recipe, 'model_spec': model_spec, 'fitting': fit['identity'], 'sources': source_hashes,
                   'backend': {'device': str(device), 'torch': torch.__version__, 'tf32': False,
                               'cache_limit': os.environ.get('TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT'),
                               'cudnn_v8_api_disabled': os.environ.get('TORCH_CUDNN_V8_API_DISABLED', '0')}}
    if model_spec['kind'] == 'bert':
        association['pretrained_policy'] = read(ROOT / 'configs/project.json')['models']['detector']
    for path in (prepared_root / 'summary.json', prepared_root.parent / 'summary.json', prepared_root.parent / 'data_protocol.json'):
        if path.exists(): association.setdefault('prepared_manifests', {})[str(path.relative_to(ROOT))] = sha(path)
    if monitoring: association.update(development_calibration=calibration['identity'], development_selection=selection['identity'])
    if objective == 'reliable_span_ce' and not recipe.get('all_span_annotations_reliable', False):
        raise ValueError('Reliable-span CE requires explicit audited annotation declaration')
    teacher = None
    if recipe.get('distillation_weight', 0):
        if objective != 'document_max_bce': raise ValueError('Distillation intervention requires document objective')
        teacher = teacher_cache(recipe['teacher_cache'], fit, recipe)
        association['teacher_cache_sha256'] = sha(recipe['teacher_cache'])
    folder = output_root / 'runs' / run_id; folder.mkdir(parents=True, exist_ok=True)
    association_hash = identity(association)
    if (folder / 'run.json').exists() and read(folder / 'run.json')['association'] != association:
        raise ValueError('Existing run recipe/model/data/source identity differs')
    if (folder / 'complete.json').exists():
        done = read(folder / 'complete.json')
        if done['association_sha256'] != association_hash or sha(ROOT / done['checkpoint']) != done['checkpoint_sha256']:
            raise ValueError('Completed checkpoint association changed')
        return done
    seed = recipe.get('seed', 42); setup(seed, device, recipe.get('cpu_threads', 1))
    model, initialization = model_for(model_spec, device)
    parameters = list(model.named_parameters())
    if recipe.get('exclude_bias_norm_weight_decay', model_spec['kind'] == 'bert'):
        groups = [{'params': [p for name, p in parameters if p.ndim > 1 and 'LayerNorm' not in name], 'weight_decay': recipe.get('weight_decay', .01)},
                  {'params': [p for name, p in parameters if p.ndim <= 1 or 'LayerNorm' in name], 'weight_decay': 0.}]
    else: groups = model.parameters()
    optimizer = torch.optim.AdamW(groups, lr=recipe['learning_rate'], weight_decay=recipe.get('weight_decay', .01))
    documents_per_batch, micro = recipe.get('batch_documents', 32), recipe.get('microbatch_windows', 128)
    epochs = recipe['epochs']; steps_per_epoch = math.ceil(len(fit['documents']) / documents_per_batch)
    if epochs < 1 or documents_per_batch < 1 or micro < 1 or recipe['learning_rate'] <= 0:
        raise ValueError('Epochs, document/window batch sizes and learning rate must be positive')
    horizon = steps_per_epoch * recipe.get('schedule_total_epochs', epochs)
    warmup = int(horizon * recipe.get('warmup_fraction', 0.))
    schedule = recipe.get('scheduler', 'constant')
    if schedule not in {'constant', 'linear'}: raise ValueError('Unsupported schedule')
    def multiplier(step):
        if warmup and step < warmup: return (step + 1) / warmup
        return 1. if schedule == 'constant' else max(0., (horizon - step) / max(1, horizon - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    latest = folder / 'checkpoint_latest.pt'; history = []; best_key = None; best_epoch = None; first = 1; prior_seconds = 0.
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=False)
        if state['association_sha256'] != association_hash: raise ValueError('Resume checkpoint association differs')
        model.load_state_dict(state['state_dict']); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        history, best_key, best_epoch = state['history'], state['best_key'], state['best_epoch']
        first = state['epoch'] + 1; prior_seconds = state['training_seconds']; restore_rng(state['rng_state'], next(model.parameters()).device)
    write(folder / 'run.json', {'association': association, 'association_sha256': association_hash, 'initialization': initialization,
                              'parameters': sum(p.numel() for p in model.parameters()), 'objective': objective,
                              'gradient_method': 'dropout-preserving two-pass max-margin replay' if objective == 'document_max_bce' else 'within-document mean known-span CE',
                              'python': sys.version, 'torch': torch.__version__, 'device': str(device),
                              'cudnn_plan_cache_limit': os.environ.get('TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT'),
                              'cudnn_v8_api_disabled': os.environ.get('TORCH_CUDNN_V8_API_DISABLED', '0'), 'command': sys.argv})
    started = time.perf_counter()
    for epoch in range(first, epochs + 1):
        model.train(); order = np.random.default_rng(seed + epoch).permutation(len(fit['documents'])); total = 0.; window_count = 0
        epoch_started = time.perf_counter()
        for offset in range(0, len(order), documents_per_batch):
            if budget: budget.check()
            chosen = order[offset:offset + documents_per_batch]; optimizer.zero_grad(set_to_none=True)
            if objective == 'document_max_bce':
                loss, windows = document_max_backward(model, fit, chosen, micro, teacher_margins=teacher,
                    distillation_weight=recipe.get('distillation_weight', 0.), temperature=recipe.get('temperature', 2.), budget=budget)
            else: loss, windows = span_backward(model, fit, chosen, micro, budget)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.get('gradient_clip', 1. if model_spec['kind'] == 'bert' else 5.))
            if not torch.isfinite(norm): raise RuntimeError('Nonfinite gradient norm')
            optimizer.step(); scheduler.step(); total += loss * len(chosen); window_count += windows
        record = {'epoch': epoch, 'loss': total / len(order), 'windows_scouted': window_count,
                  'fit_seconds': time.perf_counter() - epoch_started, 'learning_rate_next_step': optimizer.param_groups[0]['lr']}
        if monitoring:
            cal_rows = score(model, calibration, budget, recipe.get('scoring_batch_windows', 128))
            selection_rows = score(model, selection, budget, recipe.get('scoring_batch_windows', 128))
            measured = metrics(cal_rows, selection_rows); record['selection_metrics'] = measured
            primary = measured['0.005']
            if primary['recall'] is None or primary['fpr'] is None: raise ValueError('Development selection needs both labels')
            key = (primary['recall'] < .99, primary['fpr'] if primary['recall'] >= .99 else -primary['recall'], -primary['recall'] if primary['recall'] >= .99 else primary['fpr'])
            if best_key is None or key < tuple(best_key):
                best_key, best_epoch = key, epoch
                save(folder / 'checkpoint_selected.pt', {'state_dict': model.state_dict(), 'model_spec': model_spec,
                     'config': initialization.get('config'), 'epoch': epoch, 'recipe': recipe,
                     'data_identity': fit['identity'], 'source_sha256': source_hashes, 'association_sha256': association_hash})
                for name, values in (('calibration', cal_rows), ('selection', selection_rows)):
                    path = folder / (name + '_scores.jsonl'); temporary = path.with_suffix('.jsonl.tmp')
                    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in values), encoding='utf-8'); os.replace(temporary, path)
        history.append(record)
        save(latest, {'state_dict': model.state_dict(), 'model_spec': model_spec, 'config': initialization.get('config'),
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch, 'history': history,
             'association_sha256': association_hash, 'recipe': recipe, 'data_identity': fit['identity'], 'source_sha256': source_hashes,
             'best_key': best_key, 'best_epoch': best_epoch,
             'rng_state': rng_state(next(model.parameters()).device), 'training_seconds': prior_seconds + time.perf_counter() - started})
        write(folder / 'history.json', history); print(json.dumps({'event': 'epoch', 'run_id': run_id, **record}), flush=True)
        if monitoring and recipe.get('patience', 0) and epoch >= recipe.get('minimum_epochs', 1) and epoch - best_epoch >= recipe['patience']:
            break
    selected = folder / ('checkpoint_selected.pt' if monitoring else 'checkpoint_latest.pt')
    done = {'complete': True, 'run_id': run_id, 'model_spec': model_spec, 'seed': seed, 'epochs_completed': history[-1]['epoch'],
            'selected_epoch': best_epoch if monitoring else history[-1]['epoch'], 'association_sha256': association_hash,
            'checkpoint': str(selected.relative_to(ROOT)), 'checkpoint_sha256': sha(selected),
            'training_scoring_seconds': prior_seconds + time.perf_counter() - started,
            'metrics': history[best_epoch - 1]['selection_metrics'] if monitoring else {},
            'quality_scope': 'Development selection only' if monitoring else 'Fixed final fit; no development/confirmation loaded'}
    write(folder / 'complete.json', done)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True); parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', default='cuda'); parser.add_argument('--max-seconds', type=int, default=14400)
    args = parser.parse_args(); plan = read(args.plan)
    candidates = [r for r in plan['runs'] if r['run_id'] == args.run_id]
    if len(candidates) != 1: raise ValueError('Exactly one named run must be present in plan')
    recipe = candidates[0]; spec = plan.get('model_specs', plan.get('models', {}))[recipe['model_id']]
    budget = Budget(args.max_seconds)
    try:
        done = train_one(spec, recipe, ROOT / plan['prepared_root'], budget=budget, device=args.device)
        write(DEST / 'runs' / args.run_id / 'dispatch.json', {'plan_sha256': sha(args.plan), 'command': sys.argv,
              'wall_seconds': time.monotonic() - budget.started, 'memory_observations': budget.observations})
        print(json.dumps(done))
    except Exception as error:
        write(DEST / 'runs' / args.run_id / 'failure.json', {'complete': False, 'error': str(error), 'command': sys.argv,
              'wall_seconds': time.monotonic() - budget.started, 'memory_observations': budget.observations})
        raise


if __name__ == '__main__':
    main()
