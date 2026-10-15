"""Bounded, resumable fit-only CNN training; coordinator schedules heavy jobs.

Label/output KD use the deployed max-window document margin. Representation
loss uses every window with equal document weights. Exact stochastic replay
keeps one microbatch graph; no unknown document label is copied onto windows.
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
from models import SOURCE_ROOT, RepresentationHead, build_student, forward_training
sys.path.insert(0, str(SOURCE_ROOT))
from gate_eval import fit_threshold


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def save(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
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


def rng_state(device):
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
            'numpy': np.random.get_state(), 'python': random.getstate()}


def restore_rng(state, device):
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device)
    np.random.set_state(state['numpy']); random.setstate(state['python'])


class Budget:
    def __init__(self, seconds, output_root=None):
        if seconds <= 0:
            raise ValueError('Positive finite wall-time budget required')
        self.started = time.monotonic(); self.seconds = seconds
        self.output_root = Path(output_root) if output_root else None

    def check(self):
        if time.monotonic() - self.started > self.seconds:
            raise RuntimeError('Learning dispatch wall-time budget exhausted')
        if self.output_root and (self.output_root / 'stop.request').exists():
            raise RuntimeError('Study stop requested')


def load_prepared(folder, role):
    """This API never releases confirmation/final/diagnostic inputs for training."""
    if role not in {'fit', 'development_calibration', 'development_selection'}:
        raise ValueError('Only fit and development roles are permitted by learning code')
    folder = Path(folder)
    if any(part.lower() in {'notinject', 'final_test', 'final_calibration', 'new_confirmation', 'confirmation', 'new_calibration'} for part in folder.parts):
        raise ValueError('Protected/final path is forbidden in learning code')
    aliases = {'fit': {'fit', 'fit_transfer', 'expanded_fit'},
               'development_selection': {'development_selection', 'new_selection'},
               'development_calibration': {'development_calibration', 'new_development_calibration'}}
    if not (folder / 'manifest.json').is_file():
        raise ValueError('Learning input requires an explicit prepared-role manifest')
    if read(folder / 'manifest.json')['role'] not in aliases[role]:
        raise ValueError('Prepared manifest role differs')
    files = ['ids.npy', 'masks.npy', 'starts.npy', 'documents.jsonl']
    data = {name: np.load(folder / (name + '.npy'), mmap_mode='r', allow_pickle=False) for name in ('ids', 'masks', 'starts')}
    data['documents'] = [json.loads(line) for line in (folder / 'documents.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    ids, masks, starts, docs = (data[k] for k in ('ids', 'masks', 'starts', 'documents'))
    if ids.ndim != 2 or ids.shape != masks.shape or ids.shape[1] != 256:
        raise ValueError('Expected shared L256 IDs/masks')
    if not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) or np.any(ids >= 30522):
        raise ValueError('Invalid shared WordPiece IDs')
    if starts.ndim != 1 or not np.issubdtype(starts.dtype, np.integer) or len(starts) != len(docs) + 1 or starts[0] != 0 or starts[-1] != len(ids) or np.any(np.diff(starts) <= 0):
        raise ValueError('Incomplete document windows/tails')
    if np.any((masks != 0) & (masks != 1)) or np.any(masks.sum(1) == 0):
        raise ValueError('Nonbinary/empty window masks')
    if not docs or len({d['document_id'] for d in docs}) != len(docs) or any(d.get('label') not in (None, 0, 1) for d in docs):
        raise ValueError('Invalid labels or document identity')
    if role != 'fit' and any(d.get('label') is None for d in docs):
        raise ValueError('Development requires trusted labels')
    hashes = {name: sha(folder / name) for name in files}
    manifest = read(folder / 'manifest.json')
    if any(manifest['artifact_sha256'].get(name) != digest for name, digest in hashes.items()):
        raise ValueError('Prepared artifacts differ from manifest')
    data.update(role=role, folder=str(folder.resolve()), identity={'role': role, 'files': hashes})
    return data


def inputs(data, indices, device):
    return {'input_ids': torch.as_tensor(np.array(data['ids'][indices], dtype=np.int64), device=device),
            'attention_mask': torch.as_tensor(np.array(data['masks'][indices], dtype=np.int64), device=device)}


def document_loss(maximum, labels, teacher, *, label_weight=1., logit_weight=0., temperature=2.):
    if temperature <= 0 or min(label_weight, logit_weight) < 0 or not math.isfinite(temperature + label_weight + logit_weight):
        raise ValueError('Invalid loss weights/temperature')
    valid = torch.tensor([label is not None for label in labels], device=maximum.device)
    label_loss = maximum.sum() * 0
    if valid.any() and label_weight:
        target = torch.tensor([label if label is not None else 0 for label in labels], dtype=maximum.dtype, device=maximum.device)
        label_loss = F.binary_cross_entropy_with_logits(maximum[valid], target[valid])
    kd_loss = maximum.sum() * 0
    if logit_weight:
        if teacher is None:
            raise ValueError('Output KD requires fit-only teacher targets')
        q = torch.sigmoid(teacher.to(maximum) / temperature)
        logp, log1p = F.logsigmoid(maximum / temperature), F.logsigmoid(-maximum / temperature)
        # Actual Bernoulli KL, avoiding probability-log(0) under saturated teachers.
        t = teacher.to(maximum) / temperature
        kd_loss = (q * (F.logsigmoid(t) - logp) + (1 - q) * (F.logsigmoid(-t) - log1p)).mean() * temperature ** 2
    return label_weight * label_loss + logit_weight * kd_loss, label_loss, kd_loss


def document_max_backward(model, data, documents, microbatch_windows, *, teacher=None, representation_head=None,
                          label_weight=1., logit_weight=0., representation_weight=0., temperature=2., budget=None):
    """Exact dropout replay, first-tie max; all-window repr is document balanced."""
    if not len(documents) or microbatch_windows < 1 or representation_weight < 0 or not math.isfinite(representation_weight):
        raise ValueError('Invalid document/microbatch/representation weight')
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError('Replay forbids mutable BatchNorm state')
    if representation_weight and (teacher is None or representation_head is None or 'pooled' not in teacher):
        raise ValueError('Richer KD requires fit-only pooled teacher and training head')
    device = next(model.parameters()).device
    indices, owners, window_weights = [], [], []
    for owner, document in enumerate(documents):
        low, high = map(int, data['starts'][document:document + 2])
        indices.extend(range(low, high)); owners.extend([owner] * (high - low))
        window_weights.extend([1 / (len(documents) * (high - low))] * (high - low))
    chunks, best, winners = [], [-math.inf] * len(documents), [None] * len(documents)
    # enable_grad preserves train-mode dispatch. Graphs are freed after each scout.
    with torch.enable_grad():
        for offset in range(0, len(indices), microbatch_windows):
            if budget: budget.check()
            chosen = indices[offset:offset + microbatch_windows]; state = rng_state(device)
            result = forward_training(model, **inputs(data, chosen, device))
            margins = result.logits[:, 1] - result.logits[:, 0]
            if not torch.isfinite(margins).all():
                raise RuntimeError('Nonfinite scout margins')
            for pos, value in enumerate(margins.detach().cpu().tolist()):
                owner = owners[offset + pos]
                if value > best[owner]:
                    best[owner], winners[owner] = value, (len(chunks), pos)
            chunks.append((chosen, state, offset))
            del result, margins
    terminal = rng_state(device)
    maximum = torch.tensor(best, device=device, dtype=torch.float32, requires_grad=True)
    margins = None if teacher is None else torch.as_tensor(np.array(teacher['document_margins'][documents]), device=device)
    objective, label, kd = document_loss(maximum, [data['documents'][d].get('label') for d in documents], margins,
        label_weight=label_weight, logit_weight=logit_weight, temperature=temperature)
    coefficients = torch.autograd.grad(objective, maximum)[0].detach()
    selected = {}
    for owner, (chunk, position) in enumerate(winners):
        selected.setdefault(chunk, []).append((owner, position))
    representation_total = 0.
    try:
        for chunk, (chosen, state, offset) in enumerate(chunks):
            if chunk not in selected and not representation_weight:
                continue
            if budget: budget.check()
            restore_rng(state, device)
            result = forward_training(model, **inputs(data, chosen, device))
            margin = result.logits[:, 1] - result.logits[:, 0]
            replay_loss = margin.sum() * 0
            if chunk in selected:
                assignments = selected[chunk]
                owner_ids = torch.tensor([p[0] for p in assignments], device=device)
                positions = torch.tensor([p[1] for p in assignments], device=device)
                if not torch.equal(margin.detach()[positions], maximum.detach()[owner_ids]):
                    raise RuntimeError('Scout/replay differ; exact stochastic gradient unqualified')
                replay_loss = (margin[positions] * coefficients[owner_ids]).sum()
            if representation_weight:
                target = torch.as_tensor(np.array(teacher['pooled'][chosen], dtype=np.float32), device=device)
                per_window = representation_head(result.representation, target)
                weights = torch.tensor(window_weights[offset:offset + len(chosen)], device=device)
                repr_loss = (per_window * weights).sum()
                representation_total += float(repr_loss.detach())
                replay_loss = replay_loss + representation_weight * repr_loss
            if not torch.isfinite(replay_loss):
                raise RuntimeError('Nonfinite replay loss')
            replay_loss.backward()
    finally:
        restore_rng(terminal, device)
    values = {'loss': float(objective.detach()) + representation_weight * representation_total,
              'label_loss': float(label.detach()), 'logit_kl': float(kd.detach()), 'representation_loss': representation_total,
              'windows': len(indices), 'labelled_documents': sum(data['documents'][d].get('label') is not None for d in documents)}
    if not all(math.isfinite(v) for v in values.values()):
        raise RuntimeError('Nonfinite document objective')
    return values


def auc(margins, labels):
    margins, labels = np.asarray(margins), np.asarray(labels)
    positives, negatives = int(labels.sum()), int((1 - labels).sum())
    if not positives or not negatives:
        return None
    _, groups = np.unique(margins, return_inverse=True)
    p = np.bincount(groups, weights=labels); n = np.bincount(groups, weights=1 - labels)
    return float((p * (np.cumsum(n) - .5 * n)).sum() / (positives * negatives))


def score(model, data, budget=None, batch=128):
    model.eval(); values = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for at in range(0, len(data['ids']), batch):
            if budget: budget.check()
            logits = model(**inputs(data, slice(at, at + batch), device)).logits
            margins = logits[:, 1] - logits[:, 0]
            if not torch.isfinite(margins).all():
                raise RuntimeError('Nonfinite evaluation margin')
            values.extend(margins.cpu().tolist())
    return [{**d, 'margin': max(values[int(lo):int(hi)]), 'window_margins': values[int(lo):int(hi)]}
            for d, lo, hi in zip(data['documents'], data['starts'][:-1], data['starts'][1:])]


def metrics(calibration, selection):
    benign = [d['margin'] for d in calibration if d['label'] == 0]
    if not benign:
        raise ValueError('Benign calibration required')
    labels = np.array([d['label'] for d in selection]); margins = np.array([d['margin'] for d in selection])
    result = {'auc': auc(margins, labels)}
    for alpha in (.005, .01, .02, .05):
        threshold = float(fit_threshold(benign, alpha)); pred = margins > threshold
        tp, fp = int(np.sum(pred & (labels == 1))), int(np.sum(pred & (labels == 0)))
        na, nb = int(labels.sum()), int((1 - labels).sum())
        result[str(alpha)] = {'threshold_margin': threshold, 'attack_rule': 'margin > threshold_margin',
            'tp': tp, 'fp': fp, 'attacks': na, 'benign': nb, 'recall': tp / na if na else None, 'fpr': fp / nb if nb else None}
    return result


def resolve(path, canonical):
    path = Path(path)
    return path if path.is_absolute() else canonical / path


def train_one(config, run_id, canonical, output_root, device, budget):
    recipe = next(r for r in config['recipes'] if r['run_id'] == run_id)
    if Path(run_id).name != run_id or any(c in run_id for c in '/\\:'):
        raise ValueError('Unsafe run ID')
    canonical, output_root = Path(canonical).resolve(), Path(output_root).resolve()
    seed = recipe.get('seed', 42); setup(seed, device, recipe.get('cpu_threads', 1))
    raw_spec = config['model_specs'][recipe['model_id']]
    model_config = raw_spec.get('config', raw_spec)
    model = build_student(model_config, raw_spec.get('vocab_size', 30522), raw_spec.get('pad_id', 0))
    pca = recipe.get('pca_artifact', config.get('pca_artifact'))
    initialization = {'method': 'seeded scratch'}
    if pca:
        from cnn_adoption_embedding_init import apply_projected_embedding
        artifact = resolve(pca, canonical)
        stored = torch.load(artifact, map_location='cpu', weights_only=False)
        projected = stored['projected']
        initialization = {'method': 'raw-pretrained PCA prefix with seeded RMS matching', 'artifact': str(artifact),
                          'artifact_sha256': sha(artifact), 'applied': apply_projected_embedding(model, projected)}
    model.embedding.weight.requires_grad_(not recipe.get('freeze_embedding', False))
    if recipe.get('initial_checkpoint'):
        initial = resolve(recipe['initial_checkpoint'], canonical)
        if not recipe.get('initial_checkpoint_sha256') or sha(initial) != recipe['initial_checkpoint_sha256']:
            raise ValueError('Adaptation parent checkpoint digest differs/missing')
        stored = torch.load(initial, map_location='cpu', weights_only=False)
        if stored.get('model_config', stored.get('model_spec', {}).get('config')) != model_config:
            raise ValueError('Adaptation parent architecture differs')
        model.load_state_dict(stored['state_dict'], strict=True)
        initialization['adaptation'] = {'checkpoint': str(initial), 'checkpoint_sha256': recipe['initial_checkpoint_sha256'],
                                      'parent_run': recipe.get('parent_run'), 'student_only': True}
    model = model.to(device)
    prepared = resolve(recipe.get('development_root', config['prepared_root']), canonical)
    fit = load_prepared(resolve(recipe.get('fit_root', str(resolve(config['prepared_root'], canonical) / 'fit')), canonical), 'fit')
    cal = selection = None
    if recipe.get('monitor_development', True):
        cal = load_prepared(resolve(recipe.get('development_calibration_root', str(prepared / 'development_calibration')), canonical), 'development_calibration')
        selection = load_prepared(resolve(recipe.get('development_selection_root', str(prepared / 'development_selection')), canonical), 'development_selection')
    label_weight = float(recipe.get('label_weight', 1.))
    logit_weight = float(recipe.get('logit_weight', 0.))
    repr_weight = float(recipe.get('representation_weight', 0.))
    teacher = None; head = None
    if logit_weight or repr_weight:
        from teacher_cache import load_cache
        teacher = load_cache(resolve(recipe['teacher_cache'], canonical), fit, config.get('teacher'))
    if repr_weight:
        width = model.classifier.in_features // (2 if model_config.get('pooling') == 'max_mean' else 1)
        # Model construction and PCA are complete before this training-only RNG use.
        # Reset after construction so its RNG draw cannot change first student dropout.
        student_rng = rng_state(next(model.parameters()).device)
        head = RepresentationHead(width, teacher['pooled'].shape[1] // 2).to(device)
        restore_rng(student_rng, next(model.parameters()).device)
    qat = None; pure_student = model
    if recipe.get('qat'):
        from numerical import fake_quant_model
        qat = read(resolve(recipe['qat_exponents'], canonical))
        model = fake_quant_model(model, qat)
    folder = output_root / 'runs' / run_id; folder.mkdir(parents=True, exist_ok=True)
    batch, micro = recipe.get('batch_documents', 32), recipe.get('microbatch_windows', 128)
    per_epoch = math.ceil(len(fit['documents']) / batch)
    total_steps = recipe.get('total_steps', per_epoch * recipe['epochs'])
    eval_every = recipe.get('eval_every_steps', per_epoch)
    if min(batch, micro, total_steps, eval_every) < 1 or recipe['learning_rate'] <= 0:
        raise ValueError('Invalid training schedule')
    association = {'recipe': recipe, 'model_config': model_config, 'fitting': fit['identity'], 'initialization': initialization,
        'total_steps': total_steps, 'steps_per_fit_pass': per_epoch, 'canonical_root': str(canonical),
        'source_sha256': {name: sha(Path(__file__).with_name(name)) for name in ('train.py', 'models.py', 'teacher_cache.py')},
        'cnn_models_sha256': sha(SOURCE_ROOT / 'scripts/cnn_models.py'),
        'backend': {'torch': torch.__version__, 'device': str(device), 'tf32': False}}
    if teacher is not None: association['teacher_cache'] = teacher['identity']
    if cal is not None: association.update(calibration=cal['identity'], selection=selection['identity'])
    if qat is not None: association['qat_exponents'] = qat
    digest = identity(association)
    if (folder / 'run.json').exists() and read(folder / 'run.json')['association'] != association:
        raise ValueError('Existing run association differs; preserve original artifacts')
    if (folder / 'complete.json').exists():
        done = read(folder / 'complete.json')
        if done['association_sha256'] != digest or sha(done['checkpoint']) != done['checkpoint_sha256']:
            raise ValueError('Completed checkpoint identity differs')
        return done
    parameters = list(model.parameters()) + (list(head.parameters()) if head else [])
    optimizer = torch.optim.AdamW(parameters, lr=recipe['learning_rate'], weight_decay=recipe.get('weight_decay', .01))
    schedule = recipe.get('scheduler', 'constant'); schedule_steps = recipe.get('scheduler_steps', total_steps)
    warmup = int(schedule_steps * recipe.get('warmup_fraction', 0.))
    if schedule not in {'constant', 'linear'}:
        raise ValueError('Unsupported scheduler')
    def multiplier(step):
        if warmup and step < warmup: return (step + 1) / warmup
        return 1. if schedule == 'constant' else max(0., (schedule_steps - step) / max(1, schedule_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    latest = folder / 'checkpoint_latest.pt'; history = []; best = None; first = 0; prior_seconds = 0.; pending = []
    if latest.exists():
        stored = torch.load(latest, map_location='cpu', weights_only=False)
        if stored['association_sha256'] != digest: raise ValueError('Resume association differs')
        pure_student.load_state_dict(stored['state_dict']); optimizer.load_state_dict(stored['optimizer']); scheduler.load_state_dict(stored['scheduler'])
        if head: head.load_state_dict(stored['training_head_state'])
        history, best, first, prior_seconds = stored['history'], stored['best'], stored['step'], stored['training_seconds']
        pending = stored.get('pending', [])
        restore_rng(stored['rng_state'], next(model.parameters()).device)
    write(folder / 'run.json', {'association': association, 'association_sha256': digest, 'command': sys.argv,
        'student_parameters': sum(p.numel() for p in model.parameters()), 'training_only_head_parameters': sum(p.numel() for p in head.parameters()) if head else 0,
        'selection_rule': 'development recall at nominal1% calibration FPR, then AUC, then earlier checkpoint; no hard quality veto',
        'objective': 'document max label BCE + T2 Bernoulli KL + document-balanced per-window mean/max cosine'})
    started = time.monotonic(); order = None; current_pass = None; completed_steps = first
    safe_rng = rng_state(next(model.parameters()).device)
    def checkpoint(step):
        return {'state_dict': pure_student.state_dict(), 'model_config': model_config, 'model_spec': {'kind': 'cnn', 'config': model_config},
            'training_head_state': head.state_dict() if head else None, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'step': step, 'history': history, 'pending': pending, 'best': best, 'association_sha256': digest, 'recipe': recipe,
            'rng_state': rng_state(next(model.parameters()).device), 'training_seconds': prior_seconds + time.monotonic() - started}
    def evaluate_checkpoint(step):
        nonlocal best, pending
        record = {'step': step, 'fit_passes': step / per_epoch,
            'training_seconds': prior_seconds + time.monotonic() - started,
            'mean_batch_loss': {key: float(np.mean([v[key] for v in pending])) for key in ('loss', 'label_loss', 'logit_kl', 'representation_loss')},
            'windows': sum(v['windows'] for v in pending), 'learning_rate_next': optimizer.param_groups[0]['lr']}
        if cal is not None:
            cal_rows = score(model, cal, budget, recipe.get('scoring_batch_windows', 128))
            rows = score(model, selection, budget, recipe.get('scoring_batch_windows', 128))
            measured = metrics(cal_rows, rows); record['development'] = measured
            primary = measured['0.01']
            if primary['recall'] is None or primary['fpr'] is None or measured['auc'] is None: raise ValueError('Both dev labels required')
            key = (-primary['recall'], -measured['auc'])
            if best is None or key < tuple(best['key']):
                best = {'key': key, 'step': step, 'development': measured}
                save(folder / 'checkpoint_selected.pt', checkpoint(step))
                for name, rows_at in (('calibration', cal_rows), ('selection', rows)):
                    target = folder / (name + '_scores.jsonl')
                    target.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows_at), encoding='utf-8')
        history.append(record); pending = []
        save(latest, checkpoint(step)); write(folder / 'history.json', history)
        print(json.dumps({'event': 'learning', 'run_id': run_id, **record}), flush=True)
    try:
        if first and pending and (first % eval_every == 0 or first == total_steps):
            evaluate_checkpoint(first)
        for step in range(first, total_steps):
            budget.check(); fit_pass, batch_at = divmod(step, per_epoch)
            if fit_pass != current_pass:
                order = np.random.default_rng(seed + fit_pass + 1).permutation(len(fit['documents'])); current_pass = fit_pass
            chosen = order[batch_at * batch:(batch_at + 1) * batch]
            model.train()
            if head: head.train()
            optimizer.zero_grad(set_to_none=True)
            safe_rng = rng_state(next(model.parameters()).device)
            values = document_max_backward(model, fit, chosen, micro, teacher=teacher, representation_head=head,
                label_weight=label_weight, logit_weight=logit_weight, representation_weight=repr_weight,
                temperature=recipe.get('temperature', 2.), budget=budget)
            norm = torch.nn.utils.clip_grad_norm_(parameters, recipe.get('gradient_clip', 5.))
            if not torch.isfinite(norm): raise RuntimeError('Nonfinite gradient norm')
            optimizer.step(); scheduler.step(); pending.append(values); completed_steps = step + 1
            safe_rng = rng_state(next(model.parameters()).device)
            if (step + 1) % eval_every == 0 or step + 1 == total_steps:
                evaluate_checkpoint(step + 1)
        selected = folder / ('checkpoint_selected.pt' if cal is not None else 'checkpoint_latest.pt')
        done = {'complete': True, 'run_id': run_id, 'association_sha256': digest, 'checkpoint': str(selected), 'checkpoint_sha256': sha(selected),
            'selected_step': best['step'] if best else total_steps, 'steps_completed': total_steps, 'seed': seed,
            'training_seconds': prior_seconds + time.monotonic() - started, 'best': best, 'quality_scope': 'development only; no confirmation loaded'}
        write(folder / 'complete.json', done)
        return done
    except Exception as error:
        # Partial backward has no model mutation. Restore the last optimizer-step
        # RNG and preserve pending metrics so a budget boundary resumes exactly.
        restore_rng(safe_rng, next(model.parameters()).device)
        optimizer.zero_grad(set_to_none=True)
        save(latest, checkpoint(completed_steps))
        write(folder / 'failure.json', {'complete': False, 'error': str(error), 'resume_checkpoint': str(latest) if latest.exists() else None,
            'wall_seconds_this_dispatch': time.monotonic() - started, 'command': sys.argv})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--run-id', required=True)
    parser.add_argument('--canonical-root', required=True); parser.add_argument('--output-root', required=True)
    parser.add_argument('--device', default='cpu'); parser.add_argument('--max-seconds', type=float, default=3600)
    args = parser.parse_args()
    result = train_one(read(args.config), args.run_id, args.canonical_root, args.output_root, args.device, Budget(args.max_seconds, args.output_root))
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
