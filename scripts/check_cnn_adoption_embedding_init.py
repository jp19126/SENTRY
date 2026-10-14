"""Bounded CPU feasibility check: raw-pretrained PCA, four unchanged CNNs."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from cnn_adoption_embedding_init import (ROOT, raw_pretrained_embeddings,
    project_vocabulary, apply_projected_embedding, tensor_sha)
from cnn_adoption_train import DEST, read, write, sha
from cnn_models import build_model


def main():
    started = time.monotonic(); torch.set_num_threads(1)
    weight, source = raw_pretrained_embeddings()
    rng = torch.get_rng_state().clone()
    projected, pca = project_vocabulary(weight, 96)
    again, repeated = project_vocabulary(weight, 96)
    assert torch.equal(projected, again) and pca == repeated and torch.equal(rng, torch.get_rng_state())
    plan = read(ROOT / 'configs/cnn_adoption_v1.json')
    identifiers = ('textcnn_01', 'stacked_01', 'local_context_03', 'local_context_03_residual_replace_c98')
    ids = torch.tensor([[101, 2023, 2003, 1037, 7099, 102, 0, 0], [101, 7954, 3947, 102, 0, 0, 0, 0]])
    mask = ids.ne(0)
    observations = []
    for identifier in identifiers:
        spec = plan['model_specs'][identifier]; config = spec['config']
        if config['representation'] != 'token' or spec['vocab_size'] != source['vocabulary_size'] or spec['pad_id'] != source['pad_id']:
            raise ValueError('CNN does not share raw pretrained token vocabulary')
        torch.manual_seed(42); baseline = build_model(config, spec['vocab_size'], spec['pad_id'])
        baseline_rng = torch.get_rng_state().clone()
        torch.manual_seed(42); initialized = build_model(config, spec['vocab_size'], spec['pad_id'])
        assert torch.equal(baseline_rng, torch.get_rng_state())
        details = apply_projected_embedding(initialized, projected, pad_id=0)
        assert torch.equal(baseline_rng, torch.get_rng_state())
        assert all(torch.equal(value, initialized.state_dict()[key]) for key, value in baseline.state_dict().items() if key != 'embedding.weight')
        assert initialized.embedding.weight.requires_grad and details['pad_zero']
        assert abs(details['initialized_nonpad_rms'] - details['scratch_nonpad_rms']) <= 2e-7
        initialized.eval()
        logits = initialized(ids, mask).logits
        perturbed = ids.clone(); perturbed[~mask] = 1044
        altered = initialized(perturbed, mask).logits
        assert torch.equal(logits, altered) and torch.isfinite(logits).all()
        loss = F.cross_entropy(logits, torch.tensor([0, 1])); loss.backward()
        grad = initialized.embedding.weight.grad
        assert torch.isfinite(grad).all() and torch.equal(grad[0], torch.zeros_like(grad[0])) and grad[ids[mask]].abs().sum() > 0
        observations.append({'id': identifier, 'seed': 42, 'initialization': details,
            'nonembedding_state_bitwise_equal': True, 'post_initialization_rng_bitwise_equal': True,
            'finite_synthetic_forward_backward': True, 'padding_perturbation_bitwise_equal': True,
            'pad_gradient_zero': True, 'nonpad_gradient_nonzero': True, 'synthetic_loss': float(loss.detach()),
            'nonembedding_state_sha256': {key: tensor_sha(value) for key, value in baseline.state_dict().items() if key != 'embedding.weight'}})
        del baseline, initialized
    assert not torch.cuda.is_initialized()
    out = DEST / 'embedding_feasibility'; out.mkdir(parents=True, exist_ok=True)
    artifact = out / 'raw_pretrained_pca96.pt'
    if artifact.exists():
        prior = torch.load(artifact, map_location='cpu', weights_only=True)
        if prior['source'] != source or prior['pca'] != pca or not torch.equal(prior['projected'], projected):
            raise ValueError('Preserve prior projection: identity differs')
    else:
        torch.save({'projected': projected, 'source': source, 'pca': pca}, artifact)
    result = {'complete': True, 'scope': 'Supporting feasibility engineering only. No fit/dev/reserved rows, label-frequency weighting, task-finetuned embeddings, GPU, teacher inference or fits.',
        'source': source, 'projection': pca, 'projection_artifact': str(artifact.relative_to(ROOT)),
        'projection_artifact_sha256': sha(artifact), 'repeat_projection_bitwise_equal': True,
        'pca_rng_unchanged': True, 'models': observations, 'torch': torch.__version__, 'threads': torch.get_num_threads(),
        'command': __import__('sys').argv, 'python': __import__('sys').version, 'cuda_initialized': False,
        'source_sha256': {name: sha(ROOT / 'scripts' / name) for name in ('cnn_adoption_embedding_init.py', 'check_cnn_adoption_embedding_init.py', 'cnn_models.py', 'cnn_adoption_train.py')},
        'wall_seconds': time.monotonic() - started,
        'proposed_discriminator': 'Prospectively one strongest architecture: exact paired recipe/seed/KD-cache/schedule/selection versus existing KD-only control, with only trainable embedding initialization replaced. No threshold or representation change.',
        'limits': ['Raw pretrained exposure may include unknown overlap; vocab-only PCA reads no held-out text but cannot certify language-pretraining exclusions.',
            'Lexical pretraining, covariance anisotropy, centering and conditioning change together; no isolated semantic-transfer attribution.',
            'RMS is globally matched to each seeded scratch table, not per-channel whitened. CNN normalization may change the effect of this scale.',
            'PCA loses contextual/position/type information and some lexical variance; practical quality could worsen.',
            'Exact repeated eigenvalues admit cross-environment basis rotations; use the frozen projection artifact and hashes.',
            'Two requested widths are feasible; no chosen architecture, new run, superiority, novelty or adoption claim is established.']}
    write(out / 'checks.json', result)
    print(json.dumps({'complete': True, 'wall_seconds': result['wall_seconds'],
        'variance_96': pca['retained_variance_fraction'], 'minimum_relative_gap': pca['minimum_relative_eigenvalue_gap_through_retained_boundary'],
        'models': [{k: item[k] for k in ('id', 'seed', 'initialization')} for item in observations],
        'report': str((out / 'checks.json').relative_to(ROOT))}))


if __name__ == '__main__': main()
