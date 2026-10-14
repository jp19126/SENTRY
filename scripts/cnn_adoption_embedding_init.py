"""Isolated raw-pretrained word-embedding PCA initializer; not used by trainer.

Standard transfer initialization, no novelty or quality claim. No dataset rows,
teacher scores, task-finetuned weights or encoder/position/type embeddings enter
the projection. Model construction must retain ordinary seeded initialization;
then call apply_projected_embedding before constructing its optimizer.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''): digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def raw_pretrained_embeddings():
    """Read only the actual pinned raw pretrained checkpoint and its vocabulary."""
    policy = json.loads((ROOT / 'configs/project.json').read_text(encoding='utf-8-sig'))['models']['detector']
    if policy['tokenizer_revision'] != policy['revision']: raise ValueError('Tokenizer and embedding revision differ')
    folder = ROOT / 'checkpoints/huggingface' / ('models--' + policy['id'].replace('/', '--')) / 'snapshots' / policy['revision']
    checkpoint = folder / 'pytorch_model.bin'
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    vocabulary = (folder / 'vocab.txt').read_text(encoding='utf-8').splitlines()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    weight = state['bert.embeddings.word_embeddings.weight'].detach().clone()
    del state
    if tuple(weight.shape) != (30522, 256) or config['vocab_size'] != len(vocabulary) or vocabulary[0] != '[PAD]' or config['pad_token_id'] != 0:
        raise ValueError('Pinned shared token/PAD vocabulary or pretrained embedding shape differs')
    if not torch.isfinite(weight).all(): raise ValueError('Nonfinite pretrained embeddings')
    metadata = {'source': 'raw language-pretrained checkpoint, not project-finetuned teacher',
        'model_id': policy['id'], 'revision': policy['revision'], 'tensor_key': 'bert.embeddings.word_embeddings.weight',
        'shape': list(weight.shape), 'tensor_sha256': tensor_sha(weight),
        'files_sha256': {name: file_sha(folder / name) for name in ('pytorch_model.bin', 'config.json', 'vocab.txt')},
        'vocabulary_size': len(vocabulary), 'pad_id': 0}
    return weight, metadata


def project_vocabulary(weight, dimensions=96, pad_id=0):
    """Full deterministic covariance PCA, with uniform non-PAD token weighting.

    Axis signs use the first maximum-absolute loading, made positive. Exact
    repeated eigenvalues can still admit rotated bases across LAPACK versions;
    artifact hashes/environment and eigenvalue gaps preserve that limitation.
    """
    if weight.device.type != 'cpu' or weight.ndim != 2 or not torch.isfinite(weight).all():
        raise ValueError('Finite CPU vocabulary matrix required')
    if not 0 <= pad_id < len(weight) or not 0 < dimensions <= weight.shape[1]:
        raise ValueError('Invalid PCA width/PAD row')
    retained = torch.arange(len(weight)) != pad_id
    rows = weight[retained].double(); mean = rows.mean(0); centered = rows - mean
    covariance = centered.T @ centered / len(rows)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.flip(0); basis = eigenvectors.flip(1)[:, :dimensions]
    anchor = basis.abs().argmax(0)
    signs = torch.sign(basis[anchor, torch.arange(dimensions)])
    basis = basis * signs
    projected = ((weight.double() - mean) @ basis).float(); projected[pad_id].zero_()
    gaps = eigenvalues[:-1] - eigenvalues[1:]
    kept_gaps = gaps[:dimensions]
    gaps_relative = kept_gaps / eigenvalues[:len(kept_gaps)].abs().clamp_min(torch.finfo(torch.float64).tiny)
    metadata = {'method': 'uniform non-PAD centered covariance PCA, FP64 eigendecomposition, FP32 projected rows',
        'dimensions': dimensions, 'pad_id': pad_id, 'rows': len(rows),
        'mean_sha256': tensor_sha(mean), 'basis_sha256': tensor_sha(basis),
        'projected_sha256': tensor_sha(projected), 'eigenvalues': eigenvalues.tolist(),
        'retained_variance_fraction': float(eigenvalues[:dimensions].sum() / eigenvalues.sum()),
        'minimum_relative_eigenvalue_gap_through_retained_boundary': float(gaps_relative.min()) if len(gaps_relative) else None,
        'axis_sign_rule': 'first maximum-absolute loading coordinate positive',
        'portability': 'Pinned artifacts/environment; eigenvector rotation at repeated eigenvalues is not universally bitwise portable.'}
    return projected, metadata


def apply_projected_embedding(model, projected, *, pad_id=0):
    """In-place trainable replacement, preserving all other weights and RNG.

    Prefixes of the saved maximal PCA basis support both 64 and 96 dimensions.
    Global RMS matches THIS seeded scratch model, not a tuned scaling target.
    """
    if projected.device.type != 'cpu' or projected.ndim != 2 or not torch.isfinite(projected).all():
        raise ValueError('Finite CPU projected table required')
    table = model.embedding.weight
    if table.device.type != 'cpu' or table.dtype != torch.float32 or table.shape[0] != len(projected) or projected.shape[1] < table.shape[1] or model.embedding.padding_idx != pad_id:
        raise ValueError('CNN embedding shape/device/PAD is incompatible')
    retained = torch.arange(len(table)) != pad_id
    target_rms = table.detach()[retained].double().square().mean().sqrt()
    replacement = projected[:, :table.shape[1]].clone(); replacement[pad_id].zero_()
    original_rms = replacement[retained].double().square().mean().sqrt()
    if original_rms <= 0 or target_rms <= 0: raise ValueError('Zero RMS initializer')
    factor = float(target_rms / original_rms)
    replacement.mul_(factor)
    with torch.no_grad(): table.copy_(replacement)
    table.requires_grad_(True)
    return {'dimensions': table.shape[1], 'global_nonpad_scale': factor,
        'scratch_nonpad_rms': float(target_rms), 'projected_nonpad_rms_before_scale': float(original_rms),
        'initialized_nonpad_rms': float(table.detach()[retained].double().square().mean().sqrt()),
        'initialized_sha256': tensor_sha(table), 'trainable': table.requires_grad, 'pad_zero': bool((table[pad_id] == 0).all()),
        'remaining_confounds': 'PCA anisotropy/centering and lexical pretraining change together; RMS matching does not isolate semantic knowledge.'}
