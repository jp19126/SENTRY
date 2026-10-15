"""Small CPU proofs for pooled transfer, exact replay and deployment separation."""
import argparse
import copy
import json
import tempfile
import io
import contextlib
import time
from pathlib import Path
import numpy as np
import torch
from train import (Budget, auc, document_loss, document_max_backward, inputs, load_prepared,
                   restore_rng, rng_state, setup, sha, train_one, write)
from models import RepresentationHead, build_student, forward_training, masked_pool, receptive_field


def direct(model, head, data, documents, micro, teacher, label_weight, logit_weight, representation_weight):
    device = next(model.parameters()).device
    indices, counts = [], []
    for doc in documents:
        low, high = map(int, data['starts'][doc:doc + 2]); counts.append(high - low)
        indices.extend(range(low, high))
    results = [forward_training(model, **inputs(data, indices[at:at + micro], device)) for at in range(0, len(indices), micro)]
    margins = torch.cat([r.logits[:, 1] - r.logits[:, 0] for r in results])
    offset = 0; maximum = []
    for count in counts:
        maximum.append(margins[offset:offset + count].max(0).values); offset += count
    objective, _, _ = document_loss(torch.stack(maximum), [data['documents'][d].get('label') for d in documents],
        torch.as_tensor(teacher['document_margins'][documents], device=device), label_weight=label_weight, logit_weight=logit_weight)
    if representation_weight:
        summary = torch.cat([r.representation for r in results])
        losses = head(summary, torch.as_tensor(teacher['pooled'][indices], device=device)); offset = 0; mean = []
        for count in counts:
            mean.append(losses[offset:offset + count].mean()); offset += count
        objective = objective + representation_weight * torch.stack(mean).mean()
    objective.backward()
    return float(objective.detach())


def comparison(data, teacher, *, label_weight, logit_weight, representation_weight, tie=False):
    setup(2718, 'cpu', 1)
    config = {'id': 'tiny', 'family': 'dilated', 'representation': 'token', 'embedding_dim': 4,
              'channels': 4, 'depth': 2, 'kernel_sizes': [3], 'dilations': [1, 2], 'normalization': 'none',
              'dropout': .25, 'pooling': 'max_mean'}
    model = build_student(config, 32); head = RepresentationHead(4, 256)
    if tie:
        with torch.no_grad(): model.classifier.weight.zero_(); model.classifier.bias.zero_()
    other, other_head = copy.deepcopy(model), copy.deepcopy(head)
    model.train(); other.train(); head.train(); other_head.train()
    initial = rng_state(torch.device('cpu'))
    expected = direct(other, other_head, data, [0, 1, 2], 2, teacher, label_weight, logit_weight, representation_weight)
    expected_rng = rng_state(torch.device('cpu'))
    restore_rng(initial, torch.device('cpu'))
    measured = document_max_backward(model, data, [0, 1, 2], 2, teacher=teacher, representation_head=head,
        label_weight=label_weight, logit_weight=logit_weight, representation_weight=representation_weight)
    actual_rng = rng_state(torch.device('cpu')); max_error = 0.; failures = []
    for prefix, left, right in (('student', other, model), ('head', other_head, head)):
        for (name, a), (name2, b) in zip(left.named_parameters(), right.named_parameters()):
            assert name == name2
            if a.grad is None or b.grad is None:
                if a.grad is not None or b.grad is not None: failures.append(prefix + '.' + name + ':missing')
                continue
            error = float((a.grad - b.grad).abs().max()); max_error = max(max_error, error)
            if not torch.allclose(a.grad, b.grad, atol=2e-5, rtol=2e-4) or not torch.isfinite(b.grad).all():
                failures.append(prefix + '.' + name)
    model.eval(); original = inputs(data, [1, 4, 6], 'cpu'); modified = {k: v.clone() for k, v in original.items()}
    modified['input_ids'][modified['attention_mask'] == 0] = 31
    with torch.inference_mode():
        first = forward_training(model, **original); second = forward_training(model, **modified)
    mask_error = max(float((first.logits - second.logits).abs().max()), float((first.representation - second.representation).abs().max()))
    passed = not failures and abs(measured['loss'] - expected) <= 2e-6 and torch.equal(expected_rng['torch'], actual_rng['torch']) and mask_error == 0.
    return {'label_weight': label_weight, 'logit_weight': logit_weight, 'representation_weight': representation_weight,
        'ties': tie, 'direct_loss': expected, 'replay_loss': measured['loss'], 'max_gradient_error': max_error,
        'gradient_failures': failures, 'terminal_torch_rng_equal': torch.equal(expected_rng['torch'], actual_rng['torch']),
        'masked_id_logits_and_representation_error': mask_error, 'passed': passed}


def resume_check(fixture_root):
    """Budget interruption inside an unfinished backward must resume exact weights."""
    # Preserve focused fixtures in the caller's writable results tree. Windows
    # sandbox temporary-directory permissions are not assumed to be recursive.
    fixture_root = Path(fixture_root) / ('resume_' + str(time.time_ns()))
    fixture_root.mkdir(parents=True, exist_ok=True)
    with contextlib.nullcontext(str(fixture_root)) as temporary:
        root = Path(temporary); prepared = root / 'prepared'
        for role in ('fit', 'development_calibration', 'development_selection'):
            folder = prepared / role; folder.mkdir(parents=True)
            ids = np.tile(np.arange(1, 257) % 29 + 1, (3, 1)); ids[1] = ids[1, ::-1]
            masks = np.ones_like(ids); masks[:, 10:] = 0
            np.save(folder / 'ids.npy', ids); np.save(folder / 'masks.npy', masks); np.save(folder / 'starts.npy', np.arange(4))
            documents = [{'document_id': str(i), 'label': label} for i, label in enumerate((1, 0, 1))]
            (folder / 'documents.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in documents), encoding='utf-8')
            write(folder / 'manifest.json', {'role': role, 'artifact_sha256': {name: sha(folder / name)
                for name in ('ids.npy', 'masks.npy', 'starts.npy', 'documents.jsonl')}})
        config = {'prepared_root': str(prepared), 'model_specs': {'tiny': {'config': {'id': 'tiny', 'family': 'dilated',
            'representation': 'token', 'embedding_dim': 4, 'channels': 4, 'depth': 2, 'kernel_sizes': [3], 'dilations': [1, 2],
            'normalization': 'none', 'dropout': .25, 'pooling': 'max_mean'}, 'vocab_size': 32}},
            'recipes': [{'run_id': 'tiny', 'model_id': 'tiny', 'seed': 42, 'epochs': 2, 'learning_rate': .001,
                         'batch_documents': 1, 'microbatch_windows': 1, 'scoring_batch_windows': 3,
                         'scheduler': 'constant', 'warmup_fraction': 0.}]}
        class InterruptBudget(Budget):
            def __init__(self): super().__init__(120); self.calls = 0
            def check(self):
                self.calls += 1
                if self.calls == 16: raise RuntimeError('intentional synthetic resource boundary')
        with contextlib.redirect_stdout(io.StringIO()):
            baseline = train_one(config, 'tiny', root, root / 'baseline', 'cpu', Budget(120))
            try: train_one(config, 'tiny', root, root / 'resumed', 'cpu', InterruptBudget())
            except RuntimeError as error:
                assert 'intentional synthetic resource boundary' in str(error)
            else: raise AssertionError('Interruption fixture did not interrupt')
            interrupted = torch.load(root / 'resumed/runs/tiny/checkpoint_latest.pt', map_location='cpu', weights_only=False)
            resumed = train_one(config, 'tiny', root, root / 'resumed', 'cpu', Budget(120))
        a = torch.load(root / 'baseline/runs/tiny/checkpoint_latest.pt', map_location='cpu', weights_only=False)
        b = torch.load(root / 'resumed/runs/tiny/checkpoint_latest.pt', map_location='cpu', weights_only=False)
        weights_equal = all(torch.equal(a['state_dict'][name], b['state_dict'][name]) for name in a['state_dict'])
        losses_equal = [r['mean_batch_loss'] for r in a['history']] == [r['mean_batch_loss'] for r in b['history']]
        selection_equal = a['best'] == b['best']
        return {'interrupted_completed_step': interrupted['step'], 'final_student_weights_exact': weights_equal,
                'epoch_losses_exact': losses_equal, 'development_selection_exact': selection_equal,
                'optimizer_steps_completed': b['step'], 'passed': weights_equal and losses_equal and selection_equal and b['step'] == 6}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out'); parser.add_argument('--fixture-root')
    args = parser.parse_args(); setup(42, 'cpu', 1)
    ids = np.random.default_rng(42).integers(1, 30, size=(7, 9)); masks = np.ones_like(ids)
    masks[1, 4:] = 0; masks[4, 2:] = 0; masks[6, 7:] = 0
    data = {'ids': ids, 'masks': masks, 'starts': np.array([0, 2, 5, 7]),
            'documents': [{'document_id': 'a', 'label': 1}, {'document_id': 'b', 'label': None}, {'document_id': 'c', 'label': 0}]}
    windows = np.array([1., -2., 4., .5, 2., -.1, -.5], dtype=np.float32)
    teacher = {'window_margins': windows, 'document_margins': np.array([1., 4., -.1], dtype=np.float32),
               'pooled': np.random.default_rng(5).normal(size=(7, 512)).astype(np.float32)}
    comparisons = [comparison(data, teacher, label_weight=1., logit_weight=kd, representation_weight=rep)
                   for kd, rep in ((0., 0.), (.3, 0.), (.3, .1), (0., .1))]
    comparisons.append(comparison(data, teacher, label_weight=1., logit_weight=.3, representation_weight=.1, tie=True))
    empty = masked_pool(torch.ones(2, 4, 5), torch.zeros(2, 1, 5, dtype=torch.bool))
    pool = masked_pool(torch.tensor([[[1., 3., 1e9]]]), torch.tensor([[[True, True, False]]]))
    pool_ok = bool((empty == 0).all()) and torch.equal(pool, torch.tensor([[2., 3.]]))
    auc_ok = auc([0, 1, 1, 2], [0, 0, 1, 1]) == .875
    rejected_roles = []
    for role in ('confirmation', 'new_confirmation', 'final_test', 'new_calibration'):
        try: load_prepared('never-opened', role)
        except ValueError: rejected_roles.append(role)
    config = {'family': 'dilated', 'depth': 6, 'kernel_sizes': [3], 'dilations': [1, 2, 4, 8, 16, 32]}
    fixture_root = args.fixture_root or str(Path(args.out).parent / 'learning_cpu_fixtures' if args.out else Path.cwd() / 'results/kd_cnn_v2/learning_cpu_fixtures')
    resume = resume_check(fixture_root)
    result = {'purpose': 'CPU synthetic direct-autograd versus stochastic replay, including unlabelled docs, ties, padding, all-window representation gradients',
        'comparisons': comparisons, 'masked_pool_empty_and_padding': pool_ok, 'auc_tie_check': auc_ok,
        'closed_roles_rejected_before_file_open': rejected_roles, 'receptive_field': receptive_field(config),
        'cuda_initialized': torch.cuda.is_initialized(), 'torch': torch.__version__,
        'resource_boundary_resume': resume,
        'passed': all(r['passed'] for r in comparisons) and pool_ok and auc_ok and len(rejected_roles) == 4 and receptive_field(config) == 127 and resume['passed'] and not torch.cuda.is_initialized()}
    if args.out: write(args.out, result)
    print(json.dumps(result, indent=2))
    if not result['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
