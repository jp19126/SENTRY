"""Light synthetic CPU cache construction/association checks; no real inference."""
from __future__ import annotations
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import cnn_adoption_teacher_cache as cache
from cnn_adoption_train import DEST, ROOT, identity, read, sha, teacher_cache, write


def rejects(function):
    try: function()
    except (ValueError, KeyError): return
    raise AssertionError('Expected invalid identity to be rejected')


def main():
    started = time.monotonic(); torch.set_num_threads(1)
    directory = DEST / 'teacher_cache_guard_fixture'
    prepared = directory / 'prepared/synthetic'; fit_folder = prepared / 'fit'
    fit_folder.mkdir(parents=True, exist_ok=True)
    ids = np.zeros((5, 256), dtype=np.int64); ids[:, 0] = [2, 7, 5, 5, 1]
    masks = np.zeros_like(ids, dtype=np.uint8); masks[:, 0] = 1
    starts = np.array([0, 2, 5], dtype=np.int64)
    for name, value in (('ids', ids), ('masks', masks), ('starts', starts)):
        np.save(fit_folder / (name + '.npy'), value)
    (fit_folder / 'documents.jsonl').write_text('\n'.join(json.dumps({'document_id': value, 'label': label})
        for value, label in (('synthetic:a', 0), ('synthetic:b', 1))) + '\n', encoding='utf-8')
    files = {name: sha(fit_folder / name) for name in ('ids.npy', 'masks.npy', 'starts.npy', 'documents.jsonl')}
    fitting_identity = {'role': 'fit', 'files': files}
    write(fit_folder / 'manifest.json', {'role': 'fit', 'artifact_sha256': files})
    run_id = 'synthetic_bert'; run_folder = directory / 'runs' / run_id
    run_folder.mkdir(parents=True, exist_ok=True); checkpoint = run_folder / 'selected.pt'
    checkpoint.write_bytes(b'synthetic checkpoint identity, never deserialized')
    descriptor = {'model_spec': {'kind': 'bert', 'initialization': 'pretrained'},
        'checkpoint_identity': {'path': str(checkpoint.relative_to(ROOT)), 'sha256': sha(checkpoint)},
        'numerical_format': 'fp32', 'export_manifest_sha256': None}
    write(run_folder / 'run.json', {'association': {'fitting': fitting_identity}})
    eval_folder = directory / 'evaluation/development' / run_id / 'fp32'; eval_folder.mkdir(parents=True, exist_ok=True)
    roles = ('development_calibration', 'development_selection', 'known_diagnostics')
    score_hashes = {}
    for role in roles:
        name = role + '_scores.jsonl'; (eval_folder / name).write_text('opaque synthetic bytes\n', encoding='utf-8')
        score_hashes[name] = sha(eval_folder / name)
    association = {'stage': 'development', 'target': descriptor, 'data': {role: {} for role in roles},
        'source_sha256': {'scripts/cnn_adoption_train.py': sha(ROOT / 'scripts/cnn_adoption_train.py')}}
    development = {'complete': True, 'evaluation_id': run_id, 'target_key': run_id + ':fp32',
        'association': association, 'association_sha256': identity(association), 'score_files_sha256': score_hashes}
    write(eval_folder / 'complete.json', development)
    qualified = {'run_id': run_id, 'teacher_checkpoint_sha256': sha(checkpoint),
        'qualified_development_record_sha256': sha(eval_folder / 'complete.json'),
        'fit_documents_sha256': files['documents.jsonl'], 'numerical_format': 'fp32',
        'rationale': 'Synthetic association test, no utility claim.'}
    cache.qualification(qualified, descriptor, development, fitting_identity)
    for key, value in (('teacher_checkpoint_sha256', 'bad'), ('fit_documents_sha256', 'bad'),
                       ('numerical_format', 'historical_w8_qat'), ('rationale', '')):
        changed = {**qualified, key: value}
        rejects(lambda: cache.qualification(changed, descriptor, development, fitting_identity))
    for change in ('checkpoint', 'stage', 'kind', 'role'):
        altered = copy.deepcopy(development); target = copy.deepcopy(descriptor); fit = copy.deepcopy(fitting_identity)
        if change == 'checkpoint': altered['association']['target']['checkpoint_identity']['sha256'] = 'bad'
        elif change == 'stage': altered['association']['stage'] = 'confirmation'
        elif change == 'kind': target['model_spec']['kind'] = 'cnn'
        else: fit['role'] = 'confirmation'
        altered['association_sha256'] = identity(altered['association'])
        rejects(lambda: cache.qualification(qualified, target, altered, fit))
    assert cache.document_margins([2, 7, 5, 5, 1], starts) == [7., 5.]
    for margins, bounds in (([1, float('nan')], [0, 2]), ([1, 2], [0, 1]), ([1, 2], [0, 0, 2])):
        rejects(lambda: cache.document_margins(margins, bounds))

    class TinyTeacher(torch.nn.Module):
        calls = 0
        def forward(self, input_ids, attention_mask):
            self.calls += 1
            margin = input_ids[:, 0].float() * attention_mask[:, 0]
            return SimpleNamespace(logits=torch.stack([torch.zeros_like(margin), margin], 1))

    model = TinyTeacher(); loaded_roles = []
    original_loader = cache.load_prepared
    def only_fit(folder, role, **kwargs):
        assert role == 'fit'; loaded_roles.append(role)
        return original_loader(folder, role, **kwargs)
    budget = SimpleNamespace(started=time.monotonic(), seconds=30, observations=[], check=lambda: None)
    plan = {'qualified_teacher': qualified, 'prepared_root': str(prepared)}
    # Unique fixture output preserves completed earlier tests if rerun.
    cache_id = 'test_' + str(time.time_ns())
    with patch.object(cache, 'DEST', directory), patch.object(cache, 'resolve_target',
            return_value=(run_id, descriptor, checkpoint, None, None)), \
            patch.object(cache, 'model_for', return_value=(model, {'synthetic': True})), \
            patch.object(cache, 'load_prepared', side_effect=only_fit):
        saved = cache.build_cache(plan, cache_id, 'cpu', 2, budget)
        assert saved['documents'] == 2 and saved['windows'] == 5 and model.calls == 3
        assert cache.build_cache(plan, cache_id, 'cpu', 2, budget) == saved and model.calls == 3
        rejects(lambda: cache.build_cache(plan, cache_id, 'cpu', 3, budget))
        altered = copy.deepcopy(plan); altered['qualified_teacher']['teacher_checkpoint_sha256'] = 'bad'
        rejects(lambda: cache.build_cache(altered, cache_id, 'cpu', 2, budget))
    cache_path = directory / 'teacher_cache' / cache_id / 'teacher_cache.json'
    fitting = original_loader(fit_folder, 'fit')
    assert teacher_cache(cache_path, fitting, qualified).tolist() == [7., 5.]
    valid = read(cache_path)
    for key, value in (('role', 'confirmation'), ('document_ids', ['synthetic:b', 'synthetic:a']),
                       ('teacher_checkpoint_sha256', 'bad'), ('document_margins', [1.])):
        write(cache_path, {**valid, key: value})
        rejects(lambda: teacher_cache(cache_path, fitting, qualified))
    write(cache_path, valid)
    rejects(lambda: cache.reuse(cache_path.parent, {**saved['association'], 'format': 'bad'}, fitting, qualified))
    partial = directory / 'teacher_cache' / ('partial_' + str(time.time_ns())); partial.mkdir(parents=True)
    (partial / 'run.json').write_text('{}\n', encoding='utf-8')
    rejects(lambda: cache.reuse(partial, saved['association'], fitting, qualified))
    assert not torch.cuda.is_initialized()
    result = {'complete': True, 'scope': 'Synthetic CPU only; mocked local completed teacher and development records; no real model inference, training, reserved data or GPU.',
        'checks': ['qualified CP/fit/format/stage/model/rationale identity rejection',
            'max-margin all windows including tails and tied maxima', 'actual trainer cache-schema document order/role/CP/count guards',
            'synthetic end-to-end inference batching crossing document boundaries', 'exact completed reuse without inference and changed batch/CP rejection',
            'partial output preservation', 'only fit loaded; CUDA never initialized'],
        'loaded_roles': loaded_roles, 'synthetic_cache_complete_sha256': sha(cache_path.parent / 'complete.json'),
        'fixture_path': str(directory.relative_to(ROOT)), 'cuda_initialized': False,
        'source_sha256': {name: sha(ROOT / 'scripts' / name) for name in ('cnn_adoption_teacher_cache.py', 'check_cnn_adoption_teacher_cache.py', 'cnn_adoption_train.py')},
        'wall_seconds': time.monotonic() - started}
    write(DEST / 'teacher_cache_contract_checks.json', result)
    print(json.dumps(result))


if __name__ == '__main__': main()
