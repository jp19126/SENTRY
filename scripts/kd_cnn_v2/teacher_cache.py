"""Fit-only pinned BERT layer-2 pooled cache; never consumes confirmation data."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from train import Budget, identity, inputs, load_prepared, read, resolve, setup, sha, write
from models import masked_pool


def validate_teacher(teacher, canonical):
    checkpoint = resolve(teacher['checkpoint'], canonical)
    expected = teacher.get('checkpoint_sha256', teacher.get('teacher_checkpoint_sha256'))
    if not expected or sha(checkpoint) != expected:
        raise ValueError('Common teacher checkpoint digest differs/missing')
    development = resolve(teacher['qualified_development_record'], canonical)
    if sha(development) != teacher['qualified_development_record_sha256']:
        raise ValueError('Development qualification digest differs')
    record = read(development)
    if not record.get('complete') or record['association']['stage'] != 'development':
        raise ValueError('Teacher must have completed development qualification')
    if record['association']['target']['checkpoint_identity']['sha256'] != expected:
        raise ValueError('Teacher development belongs to different checkpoint')
    return checkpoint, expected


def load_teacher(canonical, teacher, device='cpu'):
    """Reusable pinned common teacher loader for coordinator evaluation."""
    canonical = Path(canonical)
    checkpoint, _ = validate_teacher(teacher, canonical)
    policy = read(canonical / 'configs/project.json')['models']['detector']
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(policy['id'], revision=policy['revision'],
        cache_dir=canonical / 'checkpoints/huggingface', num_labels=2, local_files_only=True)
    stored = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(stored['state_dict'], strict=True)
    return model.float().to(device).eval()


def load_cache(folder, fitting, expected_teacher=None):
    folder = Path(folder); manifest = read(folder / 'manifest.json')
    if not manifest.get('complete') or manifest['role'] != 'fit' or manifest['fitting'] != fitting['identity']:
        raise ValueError('Teacher cache belongs to different/uncompleted fitting material')
    if manifest['document_ids'] != [d['document_id'] for d in fitting['documents']]:
        raise ValueError('Teacher fitting document order differs')
    if identity(manifest['association']) != manifest['association_sha256']:
        raise ValueError('Teacher cache association changed')
    if expected_teacher:
        expected = expected_teacher.get('checkpoint_sha256', expected_teacher.get('teacher_checkpoint_sha256'))
        if manifest['association']['teacher_checkpoint_sha256'] != expected:
            raise ValueError('Cache is not the common teacher')
    arrays = {}
    for name in ('pooled', 'window_margins', 'document_margins'):
        path = folder / (name + '.npy')
        if sha(path) != manifest['files_sha256'][path.name]:
            raise ValueError('Cached teacher array changed')
        arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
        if not np.isfinite(arrays[name]).all():
            raise ValueError('Nonfinite teacher targets')
    if arrays['pooled'].shape != (len(fitting['ids']), 512) or arrays['window_margins'].shape != (len(fitting['ids']),) or arrays['document_margins'].shape != (len(fitting['documents']),):
        raise ValueError('Incomplete teacher targets')
    for doc, (lo, hi) in enumerate(zip(fitting['starts'][:-1], fitting['starts'][1:])):
        if arrays['document_margins'][doc] != arrays['window_margins'][int(lo):int(hi)].max():
            raise ValueError('Teacher max aggregation differs')
    arrays['identity'] = {'manifest': str((folder / 'manifest.json').resolve()), 'manifest_sha256': sha(folder / 'manifest.json'),
                          'association_sha256': manifest['association_sha256']}
    return arrays


def build_cache(canonical, fit_root, teacher, out, layer=2, device='cpu', batch_windows=128, budget=None,
                reuse_expanded_cache=None, expanded_fit_root=None):
    canonical, out = Path(canonical), Path(out)
    checkpoint, teacher_sha = validate_teacher(teacher, canonical)
    fitting = load_prepared(fit_root, 'fit')
    policy = read(canonical / 'configs/project.json')['models']['detector']
    association = {'teacher_checkpoint_sha256': teacher_sha,
        'qualified_development_record_sha256': teacher['qualified_development_record_sha256'],
        'fitting': fitting['identity'], 'tokenizer': {'id': policy['id'], 'revision': policy['tokenizer_revision']},
        'teacher_layer': layer, 'pooling': 'masked mean then max, padding excluded; CLS/SEP retained',
        'pooling_unit': 'window', 'representation_dtype': 'float16', 'margin_dtype': 'float32',
        'dropout': 'disabled', 'aggregation': 'document max window binary margin',
        'source_sha256': {name: sha(Path(__file__).with_name(name)) for name in ('teacher_cache.py', 'models.py', 'train.py')}}
    if layer not in (1, 2, 3, 4) or batch_windows < 1:
        raise ValueError('Explicit BERT layer1..4 and positive batch required')
    if (out / 'manifest.json').exists():
        existing = read(out / 'manifest.json')
        if existing['association'] != association:
            raise ValueError('Existing cache association differs')
        load_cache(out, fitting, teacher)
        return existing
    if out.exists() and any(out.iterdir()):
        raise ValueError('Partial cache exists; preserve it before retrying under a new ID')
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'run.json', {'association': association, 'command': sys.argv, 'complete': False})
    started = time.monotonic(); setup(42, device, 1)
    pooled = np.lib.format.open_memmap(out / 'pooled.npy', mode='w+', dtype=np.float16, shape=(len(fitting['ids']), 512))
    windows = np.lib.format.open_memmap(out / 'window_margins.npy', mode='w+', dtype=np.float32, shape=(len(fitting['ids']),))
    max_pool_conversion_error = 0.
    try:
        if reuse_expanded_cache:
            if not expanded_fit_root: raise ValueError('Prefix reuse requires expanded fitting identity')
            expanded = load_prepared(expanded_fit_root, 'fit')
            cached = load_cache(reuse_expanded_cache, expanded, teacher)
            source_manifest = read(Path(reuse_expanded_cache) / 'manifest.json')
            if source_manifest['association']['teacher_layer'] != layer:
                raise ValueError('Expanded cache layer differs')
            count, docs = len(fitting['ids']), len(fitting['documents'])
            if [d['document_id'] for d in expanded['documents'][:docs]] != [d['document_id'] for d in fitting['documents']]:
                raise ValueError('Base document order is not exact expanded prefix')
            for key in ('ids', 'masks', 'starts'):
                expected = fitting[key]
                if not np.array_equal(expanded[key][:len(expected)], expected):
                    raise ValueError('Expanded prefix differs: ' + key)
            pooled[:] = cached['pooled'][:count]; windows[:] = cached['window_margins'][:count]
            max_pool_conversion_error = source_manifest['max_float16_pool_absolute_error']
        else:
            model = load_teacher(canonical, teacher, device)
            with torch.inference_mode():
                for at in range(0, len(fitting['ids']), batch_windows):
                    if budget: budget.check()
                    result = model(**inputs(fitting, slice(at, at + batch_windows), device), output_hidden_states=True)
                    hidden = result.hidden_states[layer]
                    mask = torch.as_tensor(np.array(fitting['masks'][at:at + batch_windows], dtype=np.bool_), device=device).unsqueeze(1)
                    summary = masked_pool(hidden.transpose(1, 2), mask).cpu().numpy()
                    margin = (result.logits[:, 1] - result.logits[:, 0]).cpu().numpy()
                    converted = summary.astype(np.float16)
                    if not np.isfinite(converted).all() or not np.isfinite(margin).all():
                        raise RuntimeError('Nonfinite cached teacher target')
                    max_pool_conversion_error = max(max_pool_conversion_error, float(np.abs(summary - converted.astype(np.float32)).max()))
                    pooled[at:at + len(summary)] = converted; windows[at:at + len(summary)] = margin
        pooled.flush(); windows.flush()
        document_margins = np.array([windows[int(lo):int(hi)].max() for lo, hi in zip(fitting['starts'][:-1], fitting['starts'][1:])], dtype=np.float32)
        np.save(out / 'document_margins.npy', document_margins, allow_pickle=False)
        # Check original data/teacher once more before marking completed.
        if sha(checkpoint) != teacher_sha or load_prepared(fit_root, 'fit')['identity'] != fitting['identity']:
            raise ValueError('Teacher/fitting bytes changed during inference')
        manifest = {'complete': True, 'role': 'fit', 'association': association, 'association_sha256': identity(association),
            'fitting': fitting['identity'], 'document_ids': [d['document_id'] for d in fitting['documents']],
            'files_sha256': {name + '.npy': sha(out / (name + '.npy')) for name in ('pooled', 'window_margins', 'document_margins')},
            'max_float16_pool_absolute_error': max_pool_conversion_error, 'windows': len(windows), 'documents': len(document_margins),
            'bytes_pooled_payload': int(pooled.nbytes), 'wall_seconds': time.monotonic() - started, 'python': sys.version,
            'torch': torch.__version__, 'device': device, 'batch_windows': batch_windows, 'command': sys.argv,
            'reused_expanded_cache': str(reuse_expanded_cache) if reuse_expanded_cache else None,
            'scope': 'Teacher targets on fitting only; no runtime teacher, no invented labels, no confirmation access'}
        write(out / 'manifest.json', manifest); load_cache(out, fitting, teacher)
        return manifest
    except Exception as error:
        write(out / 'failure.json', {'complete': False, 'error': str(error), 'wall_seconds': time.monotonic() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canonical-root', required=True); parser.add_argument('--fit-root', required=True)
    parser.add_argument('--teacher-config', required=True); parser.add_argument('--out', required=True)
    parser.add_argument('--layer', type=int, default=2); parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-windows', type=int, default=128); parser.add_argument('--max-seconds', type=float, default=1800)
    parser.add_argument('--reuse-expanded-cache'); parser.add_argument('--expanded-fit-root')
    args = parser.parse_args(); config = read(args.teacher_config); teacher = config.get('teacher', config)
    result = build_cache(args.canonical_root, args.fit_root, teacher, args.out, args.layer, args.device,
        args.batch_windows, Budget(args.max_seconds), args.reuse_expanded_cache, args.expanded_fit_root)
    print(json.dumps({key: result[key] for key in ('complete', 'windows', 'documents', 'wall_seconds', 'association_sha256')}))


if __name__ == '__main__': main()
