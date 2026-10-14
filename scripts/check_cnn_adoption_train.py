"""Bounded CPU/CUDA proof of adoption stochastic replay on old fit inputs."""
from __future__ import annotations
import copy
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from types import SimpleNamespace

from cnn_adoption_train import (ROOT, DEST, cnn_config, document_max_backward,
    inputs, load_prepared, model_for, rng_state, restore_rng, setup, sha, write)

# Declared before execution; no adaptation in response to outcomes.
LOSS_ATOL, LOSS_RTOL = 1e-6, 1e-6
GRAD_ATOL, GRAD_RTOL = 2e-5, 2e-4
MASK_ATOL, MASK_RTOL = 2e-6, 2e-6


def direct(model, data, documents, micro):
    device = next(model.parameters()).device
    indices = [i for doc in documents for i in range(int(data['starts'][doc]), int(data['starts'][doc + 1]))]
    logits = torch.cat([model(**inputs(data, indices[offset:offset + micro], device)).logits
                        for offset in range(0, len(indices), micro)])
    margins = logits[:, 1] - logits[:, 0]
    maximum, position = [], 0
    for doc in documents:
        count = int(data['starts'][doc + 1] - data['starts'][doc])
        maximum.append(margins[position:position + count].max(0).values); position += count
    labels = torch.tensor([data['documents'][doc]['label'] for doc in documents], dtype=torch.float32, device=device)
    loss = F.binary_cross_entropy_with_logits(torch.stack(maximum), labels)
    loss.backward()
    return float(loss.detach())


def compare(spec, data, documents, device='cpu'):
    setup(424242, device, 1)
    model, initialization = model_for(spec, device)
    actual_device = next(model.parameters()).device
    model.train(); reference = copy.deepcopy(model)
    initial_rng = rng_state(actual_device)
    expected_loss = direct(reference, data, documents, 4)
    expected_rng = rng_state(actual_device)
    restore_rng(initial_rng, actual_device)
    observed_loss, windows = document_max_backward(model, data, documents, 4)
    observed_rng = rng_state(actual_device)
    failures, maximum_delta = [], 0.
    for (name, left), (right_name, right) in zip(reference.named_parameters(), model.named_parameters()):
        assert name == right_name
        if left.grad is None or right.grad is None:
            assert left.grad is None and right.grad is None
            continue
        if not torch.isfinite(right.grad).all(): failures.append(name + ':nonfinite')
        delta = float((left.grad - right.grad).abs().max()); maximum_delta = max(maximum_delta, delta)
        if not torch.allclose(left.grad, right.grad, atol=GRAD_ATOL, rtol=GRAD_RTOL): failures.append(name)
    loss_passed = bool(np.isclose(expected_loss, observed_loss, atol=LOSS_ATOL, rtol=LOSS_RTOL))
    rng_passed = torch.equal(expected_rng[0], observed_rng[0]) and (expected_rng[1] is None or torch.equal(expected_rng[1], observed_rng[1]))
    model.eval()
    chosen = [int(data['starts'][d + 1]) - 1 for d in documents]
    original = inputs(data, chosen, device); changed = {key: value.clone() for key, value in original.items()}
    changed['input_ids'][changed['attention_mask'] == 0] = 30521
    with torch.inference_mode():
        original_logits = model(**original).logits; altered_logits = model(**changed).logits
    masking_passed = torch.allclose(original_logits, altered_logits, atol=MASK_ATOL, rtol=MASK_RTOL)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5 if spec['kind'] == 'bert' else .001)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1. if spec['kind'] == 'bert' else 5.)
    optimizer.step()
    update_passed = bool(torch.isfinite(norm)) and all(torch.isfinite(parameter).all() for parameter in model.parameters())
    return {'model_spec': spec, 'initialization': initialization, 'windows': windows,
            'direct_loss': expected_loss, 'replay_loss': observed_loss,
            'loss_difference': abs(expected_loss - observed_loss), 'maximum_gradient_absolute_difference': maximum_delta,
            'gradient_failures': failures, 'loss_passed': loss_passed, 'terminal_rng_equal': rng_passed,
            'masked_id_perturbation_max_difference': float((original_logits - altered_logits).abs().max()),
            'masking_passed': bool(masking_passed), 'finite_one_update': update_passed,
            'passed': not failures and loss_passed and rng_passed and bool(masking_passed) and update_passed}


def ties(device='cpu'):
    class TieModel(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.values = torch.nn.Parameter(torch.zeros(4))
        def forward(self, input_ids, attention_mask):
            margin = self.values[input_ids[:, 0]]
            return SimpleNamespace(logits=torch.stack((margin * 0, margin), 1))
    data = {'ids': np.arange(4).reshape(4, 1), 'masks': np.ones((4, 1), dtype=np.int64),
            'starts': np.array([0, 2, 4]), 'documents': [{'label': 1}, {'label': 0}]}
    model = TieModel().to(device); loss, _ = document_max_backward(model, data, [0, 1], 1)
    expected = torch.tensor([-.25, 0., .25, 0.], device=device)
    passed = torch.equal(model.values.grad, expected)
    return {'loss': loss, 'gradient': model.values.grad.tolist(), 'expected': expected.tolist(), 'passed': passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args(); device = args.device
    started = time.perf_counter(); setup(424242, device, 1)
    if device == 'cuda': torch.cuda.reset_peak_memory_stats()
    data = load_prepared(ROOT / 'results/cnn_study_v1/prepared/token/fit', 'fit')
    choices = []
    lengths = np.diff(data['starts'])
    for label, long in ((0, False), (1, False), (1, True)):
        candidates = [i for i, d in enumerate(data['documents']) if d['label'] == label]
        choices.append((max if long else min)(candidates, key=lambda i: int(lengths[i])))
    observed = []
    for config_id in ('textcnn_01', 'stacked_01', 'local_context_03', 'local_context_03_residual_replace_c98'):
        observed.append(compare({'kind': 'cnn', 'config_id': config_id, 'config': cnn_config(config_id)}, data, choices, device))
    observed.append(compare({'kind': 'bert', 'initialization': 'pretrained'}, data, choices, device))
    tie_check = ties(device)
    output = {'purpose': 'Same-dropout direct document-max BCE versus memory-bounded replay, full gradients, tail masks and one finite update. Permitted old fitting documents only; no new-stage outcomes or teacher inference.',
              'prospective_tolerances': {'loss_atol': LOSS_ATOL, 'loss_rtol': LOSS_RTOL, 'gradient_atol': GRAD_ATOL,
                                        'gradient_rtol': GRAD_RTOL, 'mask_atol': MASK_ATOL, 'mask_rtol': MASK_RTOL},
              'command': sys.argv, 'python': sys.version, 'torch': torch.__version__, 'threads': 1, 'device': device,
              'device_name': torch.cuda.get_device_name() if device == 'cuda' else 'CPU',
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
              'cuda_initialized': torch.cuda.is_initialized(), 'fitting_identity': data['identity'],
              'selected_documents': [{'document_id': data['documents'][d]['document_id'], 'label': data['documents'][d]['label'],
                                      'windows': int(lengths[d]), 'last_window_valid_tokens': int(data['masks'][int(data['starts'][d + 1]) - 1].sum())} for d in choices],
              'source_sha256': {'trainer': sha(ROOT / 'scripts/cnn_adoption_train.py'), 'check': sha(Path(__file__))},
              'models': observed, 'first_tied_maximum': tie_check, 'wall_seconds': time.perf_counter() - started,
              'passed': all(r['passed'] for r in observed) and tie_check['passed'] and (torch.cuda.is_initialized() == (device == 'cuda'))}
    write(DEST / ('training_contract_checks_' + device + '.json'), output)
    print(json.dumps({key: value for key, value in output.items() if key not in ('models', 'fitting_identity')}))
    for result in observed:
        print(json.dumps({'model': result['model_spec'].get('config_id', result['model_spec']['kind']),
                          'passed': result['passed'], 'gradient_delta': result['maximum_gradient_absolute_difference'],
                          'loss_delta': result['loss_difference'], 'gradient_failures': result['gradient_failures']}))
    if not output['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
