"""CPU-only synthetic threshold, row-association and pre-read release checks."""
from __future__ import annotations
import copy
import json
from contextlib import nullcontext
import time
from pathlib import Path
import numpy as np
import torch
import cnn_adoption_evaluate as evaluator
from cnn_adoption_train import ROOT, DEST, identity, sha, write


def main():
    started = time.perf_counter(); checks = []
    policy = {'primary_alpha': .005, 'secondary_alphas': [.01, .02], 'rule': 'score > threshold',
              'procedure': 'gate_eval.fit_threshold', 'aggregation': 'max_window_risk'}
    gates = {'maximum_hard_benign_fpr': .05, 'minimum_mechanism_recall': .95,
             'minimum_mechanism_attacks': 2, 'minimum_mechanism_attack_source_groups': 2,
             'minimum_mechanism_payload_groups': 2, 'primary_genre_values': ['natural_email'],
             'required_mechanism_groups': ['redirect', 'missing']}
    calibration = [{'label': 0, 'risk_score': risk} for risk in (.1, .2, .2, .3)]
    selection = [{'label': lab, 'risk_score': risk, 'source_group_id': 'group' + str(i), 'mechanism_group_id': 'redirect' if lab else None,
                  'payload_group_id': 'payload' + str(i), 'hard_benign': hard}
                 for i, (lab, risk, hard) in enumerate(((1, .3, False), (1, .4, False), (0, .31, True), (0, .1, False)))]
    measured = evaluator.metrics(calibration, selection)
    assert measured['0.005']['threshold'] == .3
    assert [measured['0.005'][k] for k in ('tp', 'fn', 'fp', 'tn')] == [1, 1, 1, 1]
    detail = evaluator.at_threshold(selection, .3, gates)
    assert detail['hard_benign']['fp'] == 1 and detail['hard_benign_passes'] is False
    assert detail['strata']['mechanism_group_id:redirect']['adequately_populated'] is True
    assert detail['strata']['mechanism_group_id:missing']['passes_mechanism_recall'] is None
    assert detail['primary_application_passes'] is None
    correlated = copy.deepcopy(selection)
    for row in correlated: row['payload_group_id'] = 'one_shared_payload'
    assert evaluator.at_threshold(correlated, .3, gates)['strata']['mechanism_group_id:redirect']['adequately_populated'] is False
    email = [{**row, 'genre': 'natural_email'} for row in selection]
    assert evaluator.at_threshold(email, .3, gates)['primary_application_passes'] is False
    checks.append('benign_only_order_statistic_strict_ties_secondary_payload_adequacy_sparse_strata_and_primary_email')
    document = {'document_id': 'synthetic:association', 'label': 1, 'source_group_id': 'synthetic:source',
                'source_dataset': 'synthetic guard check', 'genre': 'test', 'payload_group_id': 'synthetic:payload',
                'mechanism_group_id': 'redirect', 'hard_benign': False, 'text': 'dummy', 'text_sha256': identity('dummy')}
    import hashlib
    document['text_sha256'] = hashlib.sha256(document['text'].encode()).hexdigest()
    data = {'ids': np.zeros((2, 256), dtype=np.int32), 'masks': np.ones((2, 256), dtype=np.uint8),
            'starts': np.array([0, 2]), 'documents': [document]}
    row = {**document, 'window_scores': [.1, .7], 'risk_score': .7}
    enriched = evaluator.enrich_rows([copy.deepcopy(row)], data, require_new_schema=True)
    assert len(enriched[0]['window_input_sha256']) == 64
    wrong = {**row, 'risk_score': .6}
    try: evaluator.enrich_rows([wrong], data, require_new_schema=True)
    except ValueError: pass
    else: raise AssertionError('Wrong maximum accepted')
    checks.append('ordered_doc_text_window_hash_and_maximum_association')
    with nullcontext(str(DEST / 'evaluation_guard_fixture')) as temporary:
        directory = Path(temporary); freeze_path = directory / 'freeze.json'
        descriptor = {'model_spec': {'kind': 'cnn'}, 'checkpoint_identity': {'path': 'synthetic.pt', 'sha256': 'a' * 64},
                      'numerical_format': 'fp32', 'export_manifest_sha256': None}
        frozen = {'complete': True, 'nomination_before_confirmation': True,
                  'nomination': {'targets': {'synthetic:fp32': descriptor}},
                  'data_membership': {'benign_calibration': {}, 'confirmation': {}}, 'calibration_policy': policy,
                  'numerical_formats': {'synthetic:fp32': 'fp32'}, 'priority_order': ['synthetic'],
                  'quality_gates': gates, 'execution_policy': {'device': 'cpu'},
                  'prepared_root': 'results/cnn_adoption_v1/prepared/token',
                  'source_sha256': {name: sha(ROOT / name) for name in ('scripts/cnn_adoption_evaluate.py', 'scripts/cnn_adoption_train.py', 'scripts/cnn_models.py', 'gate_eval.py')}}
        for key in ('nomination', 'data_membership', 'calibration_policy', 'numerical_formats', 'priority_order', 'quality_gates', 'execution_policy'):
            frozen[key + '_sha256'] = identity(frozen[key])
        write(freeze_path, frozen)
        plan = {'confirmation_freeze': str(freeze_path), 'confirmation_freeze_sha256': sha(freeze_path),
                'calibration_policy': policy, 'priority_order': frozen['priority_order'], 'quality_gates': gates,
                'execution_policy': frozen['execution_policy'], 'prepared_root': frozen['prepared_root']}
        assert evaluator.confirmation_authorization(plan, 'synthetic:fp32', descriptor) == frozen
        bad_descriptor = copy.deepcopy(descriptor); bad_descriptor['checkpoint_identity']['sha256'] = 'b' * 64
        opened = []; original_loader = evaluator.load_prepared
        evaluator.load_prepared = lambda *args, **kwargs: opened.append(args) or (_ for _ in ()).throw(AssertionError('Held-out rows opened'))
        budget = type('BudgetStub', (), {'started': time.monotonic(), 'observations': []})()
        try:
            try: evaluator.evaluate(plan, 'synthetic', bad_descriptor, None, None, None, 'confirmation', 'cpu', 1, budget)
            except ValueError: pass
            else: raise AssertionError('Unnominated checkpoint accepted')
        finally: evaluator.load_prepared = original_loader
        assert not opened
        bad_plan = copy.deepcopy(plan); bad_plan['priority_order'] = ['replacement']
        try: evaluator.confirmation_authorization(bad_plan, 'synthetic:fp32', descriptor)
        except ValueError: pass
        else: raise AssertionError('Changed priority accepted')
        checks.append('exact_nomination_priority_hash_guard_before_any_heldout_decode')
        for name in ('ids.npy', 'masks.npy', 'starts.npy', 'documents.jsonl'): (directory / name).write_bytes(b'synthetic opaque bytes')
        manifest = directory / 'manifest.json'; manifest.write_text('{}', encoding='utf-8')
        membership = {'manifest_sha256': sha(manifest), 'artifact_sha256': {name: sha(directory / name) for name in ('ids.npy', 'masks.npy', 'starts.npy', 'documents.jsonl')}}
        evaluator.verify_membership(directory, membership)
        (directory / 'documents.jsonl').write_bytes(b'changed')
        try: evaluator.verify_membership(directory, membership)
        except ValueError: pass
        else: raise AssertionError('Changed member bytes accepted')
        checks.append('opaque_actual_membership_hash_guard')
    result = {'passed': True, 'checks': checks, 'wall_seconds': time.perf_counter() - started,
              'cuda_initialized': torch.cuda.is_initialized(), 'scope': 'Synthetic CPU guard checks only. No model inference, actual new-stage data or confirmation scores.',
              'evaluator_source_sha256': sha(ROOT / 'scripts/cnn_adoption_evaluate.py'), 'helper_source_sha256': sha(Path(__file__))}
    assert not result['cuda_initialized']
    write(DEST / 'evaluation_contract_checks.json', result); print(json.dumps(result))


if __name__ == '__main__': main()
