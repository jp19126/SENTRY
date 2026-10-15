"""Four fixed initial-model FIT batches; no optimizer step or model selection."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from models import RepresentationHead, build_student, forward_training
from train import (document_loss, identity, inputs, load_prepared, read, resolve,
                   restore_rng, rng_state, setup, sha, write)
from teacher_cache import load_cache
from cnn_adoption_embedding_init import apply_projected_embedding


def dot(left, right):
    return sum(float((a.double() * b.double()).sum()) for a, b in zip(left, right))


def norm(values): return math.sqrt(max(0., dot(values, values)))


def cosine(left, right):
    denominator = norm(left) * norm(right)
    return dot(left, right) / denominator if denominator else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canonical-root', type=Path, required=True); parser.add_argument('--config', required=True)
    parser.add_argument('--out', type=Path, required=True); parser.add_argument('--max-seconds', type=float, default=120)
    args = parser.parse_args(); began = time.monotonic()
    if args.out.exists(): raise FileExistsError(args.out)
    if not 0 < args.max_seconds <= 120: raise ValueError('Bounded diagnostic<=120seconds required')
    setup(42, 'cpu', 2); root = args.canonical_root; config = read(args.config)
    model_config = config['model_specs']['dilated64_rf127']; model_config = model_config.get('config', model_config)
    model = build_student(model_config, 30522)
    pca_path = resolve(config['pca_artifact'], root)
    pca = torch.load(pca_path, map_location='cpu', weights_only=False)['projected']
    initialization = apply_projected_embedding(model, pca); del pca
    original_rng = rng_state(torch.device('cpu'))
    head = RepresentationHead(64, 256); restore_rng(original_rng, torch.device('cpu'))
    model.train(); head.train()
    recipe = next(r for r in config['recipes'] if r['run_id'] == 'richer_expanded_s42_dev')
    data = load_prepared(resolve(recipe['fit_root'], root), 'fit')
    teacher = load_cache(resolve(recipe['teacher_cache'], root), data, config['teacher'])
    base_count = read(root / 'results/cnn_adoption_v1/prepared/token_r2/fit/manifest.json')['documents']
    counts = np.diff(data['starts']); specifications = []
    for cohort, low, high in (('base', 0, base_count), ('transfer', base_count, len(data['documents']))):
        by_label = {}
        for label in (0, 1):
            candidates = [i for i in range(low, high) if data['documents'][i].get('label') == label and counts[i] <= 3]
            candidates.sort(key=lambda i: hashlib.sha256(('20261015|' + data['documents'][i]['document_id']).encode()).hexdigest())
            if len(candidates) < 16: raise ValueError('Insufficient bounded diagnostic cohort')
            by_label[label] = candidates[:16]
        for batch in range(2):
            chosen = by_label[0][batch * 8:(batch + 1) * 8] + by_label[1][batch * 8:(batch + 1) * 8]
            specifications.append((cohort + str(batch), chosen))
    parameters = [(name, value) for name, value in model.named_parameters() if not name.startswith('classifier.')]
    values = [value for _, value in parameters]; results = []
    for batch_index, (batch_name, documents) in enumerate(specifications):
        if time.monotonic() - began >= args.max_seconds: raise RuntimeError('FIT gradient diagnostic time ceiling reached')
        # Same fixed initial parameters and deterministic dropout seed for every
        # component; all three gradients come from this one shared forward graph.
        torch.manual_seed(42 + batch_index)
        indices, window_counts = [], []
        for doc in documents:
            low, high = map(int, data['starts'][doc:doc + 2]); indices.extend(range(low, high)); window_counts.append(high - low)
        result = forward_training(model, **inputs(data, indices, 'cpu'))
        margins = result.logits[:, 1] - result.logits[:, 0]; maxima = []; start = 0
        for count in window_counts:
            maxima.append(margins[start:start + count].max(0).values); start += count
        target_margins = torch.tensor(np.array(teacher['document_margins'][documents]), dtype=torch.float32)
        _, label, logit = document_loss(torch.stack(maxima), [data['documents'][doc]['label'] for doc in documents], target_margins,
            label_weight=1., logit_weight=1., temperature=2.)
        targets = torch.tensor(np.array(teacher['pooled'][indices], dtype=np.float32))
        per_window = head(result.representation, targets); per_document = []; start = 0
        for count in window_counts:
            per_document.append(per_window[start:start + count].mean()); start += count
        representation = torch.stack(per_document).mean()
        components = {}
        for name, loss in (('label', label), ('logit', logit), ('representation', representation)):
            raw = torch.autograd.grad(loss, values, retain_graph=True, allow_unused=True)
            components[name] = [gradient.detach() if gradient is not None else torch.zeros_like(parameter)
                                for gradient, parameter in zip(raw, values)]
        weighted = {name: [gradient * weight for gradient in components[name]]
                    for name, weight in (('label', .7), ('logit', .3), ('representation', .1))}
        task = [a + b for a, b in zip(weighted['label'], weighted['logit'])]
        results.append({'batch': batch_name, 'documents': [data['documents'][doc]['document_id'] for doc in documents],
            'windows': len(indices), 'labels': {'benign': 8, 'attack': 8}, 'dropout_seed': 42 + batch_index,
            'losses': {'label': float(label.detach()), 'logit_kl_T2': float(logit.detach()), 'representation_cosine': float(representation.detach())},
            'raw_gradient_norm': {name: norm(gradients) for name, gradients in components.items()},
            'weighted_gradient_norm': {name: norm(gradients) for name, gradients in weighted.items()},
            'weighted_task_gradient_norm': norm(task),
            'weighted_repr_to_task_norm_ratio': norm(weighted['representation']) / norm(task) if norm(task) else None,
            'gradient_cosines': {'label_logit': cosine(components['label'], components['logit']),
                'label_representation': cosine(components['label'], components['representation']),
                'logit_representation': cosine(components['logit'], components['representation']),
                'weighted_task_representation': cosine(task, weighted['representation'])}})
        del result, margins, maxima, label, logit, representation, per_window, components, weighted, task
    output = {'schema': 'kd_cnn_v2_initial_fit_gradient_diagnostic_v1', 'complete': True, 'seed': 42,
        'initialization': initialization, 'pca_artifact_sha256': sha(pca_path), 'config_sha256': sha(args.config),
        'fit_identity': data['identity'], 'cache_identity': teacher['identity'], 'teacher_layer': 2,
        'weights': {'label': .7, 'logit': .3, 'representation': .1}, 'temperature': 2.,
        'shared_parameters': [name for name, _ in parameters], 'excluded_parameters': 'Output classifier and training-only representation head; report gradients on their shared CNN backbone only',
        'batches': results, 'wall_seconds': time.monotonic() - began, 'cuda_initialized': torch.cuda.is_initialized(),
        'script_sha256': sha(Path(__file__)), 'optimizer_steps': 0,
        'limits': 'Four fixed balanced FIT batches with<=3windows per document, initial common PCA model only. Local gradient scale/alignment does not measure security information, converged training influence or held-out utility.'}
    if output['wall_seconds'] > args.max_seconds: raise RuntimeError('Diagnostic exceeded declared wall budget; do not label it complete')
    write(args.out, output)
    print(json.dumps({'out': str(args.out), 'wall_seconds': output['wall_seconds'],
        'batches': [{key: row[key] for key in ('batch', 'losses', 'raw_gradient_norm', 'weighted_gradient_norm',
            'weighted_task_gradient_norm', 'weighted_repr_to_task_norm_ratio', 'gradient_cosines')} for row in results]}), flush=True)


if __name__ == '__main__': main()
