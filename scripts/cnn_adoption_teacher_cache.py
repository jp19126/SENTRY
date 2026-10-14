"""Qualified matched-BERT fitting-only margin cache; coordinator dispatches inference.

No calibration, confirmation, unknown-span labels or runtime teacher are used.
plan.qualified_teacher binds run_id, teacher_checkpoint_sha256,
qualified_development_record_sha256, fit_documents_sha256, numerical_format
('fp32') and a nonempty rationale based on completed development evidence.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

from cnn_adoption_train import (ROOT, DEST, Budget, identity, inputs, load_prepared,
    model_for, read, setup, sha, teacher_cache, write)
from cnn_adoption_evaluate import resolve_target, relative
import numpy as np
import torch


def safe_id(value):
    if not value or Path(value).name != value or value in ('.', '..') or any(c in value for c in '/\\:'):
        raise ValueError('Unsafe cache/run identifier')
    return value


def qualification(qualified, descriptor, development, fitting_identity):
    """Pure association guard, also used in small synthetic rejection tests."""
    if descriptor['model_spec']['kind'] != 'bert' or descriptor['model_spec'].get('initialization', 'pretrained') != 'pretrained':
        raise ValueError('Teacher must be completed matched pretrained BERT')
    if qualified['numerical_format'] != 'fp32' or descriptor['numerical_format'] != 'fp32':
        raise ValueError('Explicit FP32 teacher arithmetic required')
    if not isinstance(qualified.get('rationale'), str) or not qualified['rationale'].strip():
        raise ValueError('Explicit completed-development teacher usefulness rationale required')
    if descriptor['checkpoint_identity']['sha256'] != qualified['teacher_checkpoint_sha256']:
        raise ValueError('Qualified teacher checkpoint differs')
    association = development['association']
    if not development['complete'] or association['stage'] != 'development' or association['target'] != descriptor:
        raise ValueError('Completed development record belongs to a different teacher/format/stage')
    if identity(association) != development['association_sha256']:
        raise ValueError('Development association changed')
    if development['evaluation_id'] != qualified['run_id'] or development['target_key'] != qualified['run_id'] + ':fp32':
        raise ValueError('Development run identifier differs')
    if fitting_identity['role'] != 'fit' or fitting_identity['files']['documents.jsonl'] != qualified['fit_documents_sha256']:
        raise ValueError('Qualified teacher fitting documents differ')
    # Score artifacts must come from exactly these roles; none are decoded here.
    if set(association['data']) != {'development_calibration', 'development_selection', 'known_diagnostics'}:
        raise ValueError('Teacher development record has unexpected data roles')


def document_margins(window_margins, starts):
    values = np.asarray(window_margins, dtype=np.float32)
    bounds = np.asarray(starts)
    if values.ndim != 1 or not np.isfinite(values).all() or bounds.ndim != 1 or len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != len(values) or np.any(np.diff(bounds) <= 0):
        raise ValueError('Incomplete/nonfinite document window margin association')
    return [float(values[int(lo):int(hi)].max()) for lo, hi in zip(bounds[:-1], bounds[1:])]


def reuse(folder, association, fitting, qualified):
    """Completed exact identities only. A partial cache is never overwritten."""
    complete = folder / 'complete.json'
    if complete.exists():
        saved = read(complete)
        if not saved['complete'] or saved['association'] != association or saved['association_sha256'] != identity(association):
            raise ValueError('Existing cache association differs')
        cache = folder / 'teacher_cache.json'
        if sha(cache) != saved['cache_sha256']:
            raise ValueError('Saved teacher cache bytes changed')
        cached = read(cache)
        if cached.get('association_sha256') != saved['association_sha256']:
            raise ValueError('Cache payload association differs')
        teacher_cache(cache, fitting, qualified)
        return saved
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Partial teacher cache exists; preserve it before retry')
    return None


def build_cache(plan, cache_id, device, batch_windows, budget):
    qualified = plan['qualified_teacher']
    run_id = safe_id(qualified['run_id']); cache_id = safe_id(cache_id)
    identifier, descriptor, checkpoint, export, _ = resolve_target(plan, run_id=run_id, numerical_format='fp32')
    if export is not None or checkpoint is None or not checkpoint.resolve().is_relative_to((DEST / 'runs' / run_id).resolve()):
        raise ValueError('Teacher checkpoint must be local to its completed adoption run')
    folder = DEST / 'teacher_cache' / cache_id
    evaluation_folder = DEST / 'evaluation/development' / identifier / 'fp32'
    evaluation_path = evaluation_folder / 'complete.json'
    if sha(evaluation_path) != qualified['qualified_development_record_sha256']:
        raise ValueError('Qualified development record bytes changed')
    development = read(evaluation_path)
    prepared = Path(plan['prepared_root']); prepared = prepared if prepared.is_absolute() else ROOT / prepared
    if not prepared.resolve().is_relative_to((DEST / 'prepared').resolve()):
        raise ValueError('Only adoption-prepared fitting inputs allowed')
    # Metadata-only guard precedes all fitting row decoding or teacher loading.
    fit_manifest = prepared / 'fit/manifest.json'
    manifest = read(fit_manifest)
    if manifest['role'] != 'fit': raise ValueError('Manifest is not fitting-only')
    fitting_identity = {'role': 'fit', 'files': {name: sha(prepared / 'fit' / name)
        for name in ('ids.npy', 'masks.npy', 'starts.npy', 'documents.jsonl')}}
    if not set(fitting_identity['files']) <= set(manifest['artifact_sha256']) or any(manifest['artifact_sha256'][name] != digest for name, digest in fitting_identity['files'].items()):
        raise ValueError('Fitting bytes differ from prepared manifest')
    if (prepared / 'fit/labels.npy').exists():
        fitting_identity['files']['labels.npy'] = sha(prepared / 'fit/labels.npy')
    qualification(qualified, descriptor, development, fitting_identity)
    training = read(DEST / 'runs' / run_id / 'run.json')
    if training['association']['fitting'] != fitting_identity:
        raise ValueError('Teacher was trained on a different fitting identity')
    for filename, digest in development['association']['source_sha256'].items():
        if sha(ROOT / filename) != digest: raise ValueError('Qualified development implementation changed: ' + filename)
    expected_scores = {role + '_scores.jsonl' for role in ('development_calibration', 'development_selection', 'known_diagnostics')}
    if set(development['score_files_sha256']) != expected_scores:
        raise ValueError('Qualified development score artifact set differs')
    for name, digest in development['score_files_sha256'].items():
        if sha(evaluation_folder / name) != digest: raise ValueError('Qualified development score bytes changed')
    fitting = load_prepared(prepared / 'fit', 'fit')
    if fitting['identity'] != fitting_identity: raise ValueError('Fitting input changed during validation')
    sources = {name: sha(ROOT / name) for name in ('scripts/cnn_adoption_teacher_cache.py',
        'scripts/cnn_adoption_train.py', 'scripts/cnn_adoption_evaluate.py', 'scripts/cnn_models.py', 'gate_eval.py')}
    association = {'qualified_teacher': qualified, 'teacher': descriptor, 'fitting': fitting_identity,
        'fit_manifest_sha256': sha(fit_manifest), 'sources_sha256': sources,
        'format': {'dtype': 'float32', 'score': 'max_window(logit_1-logit_0)',
                   'role': 'fit', 'dropout': 'disabled', 'unknown_annotations': 'no window labels'},
        'backend': {'device': str(device), 'torch': torch.__version__, 'batch_windows': batch_windows,
                    'tf32': False, 'cache_limit': os.environ.get('TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT'),
                    'cudnn_v8_api_disabled': os.environ.get('TORCH_CUDNN_V8_API_DISABLED', '0')}}
    # Materialize the empty leaf before stat: Windows UNC can deny stat of
    # nonexistent nested parents. Existing nonempty leaves still refuse reuse.
    folder.mkdir(parents=True, exist_ok=True)
    previous = reuse(folder, association, fitting, qualified)
    if previous: return previous
    budget.check(); folder.mkdir(parents=True, exist_ok=True)
    write(folder / 'run.json', {'association': association, 'association_sha256': identity(association),
        'command': sys.argv, 'max_seconds': budget.seconds, 'python': sys.version})
    try:
        model, loader = model_for(descriptor['model_spec'], device, checkpoint)
        model.eval(); values = []
        with torch.inference_mode():
            for at in range(0, len(fitting['ids']), batch_windows):
                budget.check()
                logits = model(**inputs(fitting, slice(at, at + batch_windows), device)).logits
                if logits.shape[1] != 2 or not torch.isfinite(logits).all(): raise ValueError('Teacher output is not finite binary logits')
                margin = logits[:, 1] - logits[:, 0]
                if margin.dtype != torch.float32 or not torch.isfinite(margin).all(): raise ValueError('Teacher margin format is not finite FP32')
                values.extend(margin.cpu().tolist())
        margins = document_margins(values, fitting['starts']); budget.check()
        # Rehash live inputs and checkpoint before committing a completed cache.
        if sha(checkpoint) != qualified['teacher_checkpoint_sha256'] or sha(evaluation_path) != qualified['qualified_development_record_sha256'] or any(sha(prepared / 'fit' / name) != digest for name, digest in fitting_identity['files'].items()):
            raise ValueError('Teacher/fitting/qualification bytes changed while caching')
        buffer_info = {'model_class': type(model).__module__ + '.' + type(model).__qualname__,
            'module_classes': sorted({type(m).__module__ + '.' + type(m).__qualname__ for m in model.modules()}),
            'parameters': sum(p.numel() for p in model.parameters()),
            'buffers': [{'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype)} for name, value in model.named_buffers()],
            'loader': loader, 'runtime_teacher_required': False}
        cache = {'role': 'fit', 'fit_documents_sha256': qualified['fit_documents_sha256'],
            'teacher_checkpoint_sha256': qualified['teacher_checkpoint_sha256'],
            'qualified_development_record_sha256': qualified['qualified_development_record_sha256'],
            'document_ids': [d['document_id'] for d in fitting['documents']], 'document_margins': margins,
            'association_sha256': identity(association)}
        write(folder / 'teacher_cache.json', cache)
        teacher_cache(folder / 'teacher_cache.json', fitting, qualified)
        result = {'complete': True, 'cache_path': relative(folder / 'teacher_cache.json'),
            'cache_sha256': sha(folder / 'teacher_cache.json'), 'association': association,
            'association_sha256': identity(association), 'teacher_buffer_class_info': buffer_info,
            'documents': len(margins), 'windows': len(values), 'command': sys.argv, 'python': sys.version,
            'wall_seconds': time.monotonic() - budget.started, 'memory_observations': budget.observations,
            'scope': 'Fitting-only cached supervision; ground-truth labels retained by trainer. No runtime teacher, no calibration/confirmation rows, no inferred window labels.'}
        write(folder / 'complete.json', result); return result
    except Exception as error:
        write(folder / 'failure.json', {'complete': False, 'error': str(error), 'command': sys.argv,
            'wall_seconds': time.monotonic() - budget.started, 'memory_observations': budget.observations})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True); parser.add_argument('--cache-id', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--batch-windows', type=int, default=32)
    parser.add_argument('--max-seconds', type=int, default=3600)
    args = parser.parse_args()
    if args.batch_windows < 1 or not 0 < args.max_seconds <= 10800: raise ValueError('Positive batch and bounded <=10800s cache budget required')
    budget = Budget(args.max_seconds); setup(42, args.device, 1)
    result = build_cache(read(args.plan), args.cache_id, args.device, args.batch_windows, budget)
    print(json.dumps({key: result[key] for key in ('complete', 'cache_path', 'cache_sha256', 'documents', 'windows', 'wall_seconds')}))


if __name__ == '__main__': main()
