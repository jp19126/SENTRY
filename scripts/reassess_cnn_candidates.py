"""Retrospective policy analysis of existing, released CNN-study scores.

No model execution, fitting, threshold deployment, protected inputs or score
modification. Frozen primary outcomes are preserved. Empirical ROC points use
evaluation labels and are descriptive only; they are not calibrated policies.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/cnn_adoption_v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def counts(data, threshold):
    a = sum(r['label'] == 1 for r in data)
    b = len(data) - a
    tp = sum(r['label'] == 1 and r['risk_score'] > threshold for r in data)
    fp = sum(r['label'] == 0 and r['risk_score'] > threshold for r in data)
    return dict(attacks=a, benign=b, tp=tp, fp=fp,
                recall=tp/a if a else None, fpr=fp/b if b else None)


def roc(data):
    """Exact score-tier ROC; do not interpolate through unattainable ties."""
    tiers = defaultdict(lambda: [0, 0])
    for row in data:
        score = row['risk_score']
        assert math.isfinite(score) and 0 <= score <= 1 and row['label'] in (0, 1)
        tiers[score][1 - row['label']] += 1
    a = sum(v[0] for v in tiers.values())
    b = sum(v[1] for v in tiers.values())
    assert a and b
    tp = fp = 0
    curve = [{'tp': 0, 'fp': 0}]
    wins = 0.0
    # Descending: higher attack scores beat all as-yet-unseen lower benign scores.
    for score in sorted(tiers, reverse=True):
        attacks, benign = tiers[score]
        wins += attacks * (b - fp - benign + benign / 2)
        tp += attacks
        fp += benign
        curve.append({'tp': tp, 'fp': fp})
    area = sum((q['fp']-p['fp']) * (q['tp']+p['tp']) / 2
               for p, q in zip(curve, curve[1:])) / (a*b)
    assert abs(area - wins/(a*b)) < 1e-12
    points = {}
    for rate in ('0.005', '0.01', '0.02', '0.05'):
        cap = int(Decimal(rate) * b)
        reachable = [p for p in curve if p['fp'] <= cap]
        best = max(reachable, key=lambda p: (p['tp'], -p['fp']))
        points[rate] = {**best, 'recall': best['tp']/a, 'fpr': best['fp']/b,
                        'maximum_false_positives': cap}
    return dict(auc=area, attacks=a, benign=b, distinct_score_tiers=len(tiers),
                empirical_roc_points=points, curve_counts=curve,
                scope='Same-sample label-informed descriptive ROC, not a fitted/deployable threshold or out-of-sample guarantee.')


def self_check():
    fixtures = [
        ([{'label': 1, 'risk_score': .9}, {'label': 0, 'risk_score': .1}], 1),
        ([{'label': 1, 'risk_score': .1}, {'label': 0, 'risk_score': .9}], 0),
        ([{'label': 1, 'risk_score': .9}, {'label': 0, 'risk_score': .9}], .5),
        ([{'label': 1, 'risk_score': 1.}, {'label': 0, 'risk_score': 1.},
          {'label': 1, 'risk_score': .5}, {'label': 0, 'risk_score': 0.}], .625),
    ]
    for data, expected in fixtures:
        result = roc(data)
        assert result['auc'] == expected
        for rate, point in result['empirical_roc_points'].items():
            # Exhaustive strict-threshold enumeration is independent of tier accumulation.
            cuts = {r['risk_score'] for r in data} | {-1., 1.}
            metrics = [counts(data, cut) for cut in cuts]
            valid = [m for m in metrics if m['fp'] <= point['maximum_false_positives']]
            assert point['tp'] == max(m['tp'] for m in valid)
    assert counts(fixtures[2][0], .9)['tp'] == 0
    return {'complete': True, 'fixtures': 4, 'coverage': 'perfect/reversed/tied/saturated score ROC, rank-vs-area AUC, brute-force strict cuts, no random tie-breaking'}


def summarize(data, threshold):
    fields = ['genre', 'mechanism_group_id', 'benign_category', 'presentation', 'position', 'context_family_id']
    groups = {}
    for field in fields:
        values = {str(r[field]) for r in data if r.get(field) is not None}
        if values:
            groups[field] = {v: counts([r for r in data if str(r.get(field)) == v], threshold) for v in sorted(values)}
    groups['window_count_bin'] = {}
    for name, lo, hi in [('1', 1, 1), ('2-3', 2, 3), ('4-7', 4, 7), ('8+', 8, math.inf)]:
        subset = [r for r in data if lo <= r['window_count'] <= hi]
        if subset: groups['window_count_bin'][name] = counts(subset, threshold)
    return {**counts(data, threshold), 'hard_benign': counts([r for r in data if r.get('hard_benign')], threshold),
            'natural_benign': counts([r for r in data if r['label'] == 0 and not r.get('constructed')], threshold),
            'constructed_benign': counts([r for r in data if r['label'] == 0 and r.get('constructed')], threshold),
            'strata': groups}


def analyze_complete(path, manifest, audited=None):
    meta = read(path)
    assert meta['complete']
    manifest[str(path.relative_to(ROOT))] = sha(path)
    if audited:
        assert manifest[str(path.relative_to(ROOT))] == audited['saved_complete_sha256']
    stage = meta['association']['stage']
    role = 'confirmation' if stage == 'confirmation' else 'development_selection'
    cal_role = 'benign_calibration' if stage == 'confirmation' else 'development_calibration'
    data_path = path.parent / (role + '_scores.jsonl')
    cal_path = path.parent / (cal_role + '_scores.jsonl')
    for p in [data_path, cal_path]:
        digest = sha(p)
        assert digest == meta['score_files_sha256'][p.name]
        manifest[str(p.relative_to(ROOT))] = digest
    data, cal = rows(data_path), rows(cal_path)
    assert all(r['label'] == 0 for r in cal)
    assert len({r['document_id'] for r in data}) == len(data)
    result = {'stage': stage, 'target': meta['target_key'], 'checkpoint_identity': meta['association']['target']['checkpoint_identity'],
              'source_groups': len({r['source_group_id'] for r in data}),
              'payload_groups': len({r['payload_group_id'] for r in data if r['label'] == 1}),
              'registered_operating_points': {}}
    ordered = sorted(r['risk_score'] for r in cal)
    for alpha, saved in meta['metrics'].items():
        threshold = ordered[len(ordered)-int(Decimal(alpha)*len(ordered))-1]
        assert threshold == saved['threshold']
        metrics = summarize(data, threshold)
        for key in ['tp', 'fp', 'recall', 'fpr']:
            assert metrics[key] == saved[key]
            if audited: assert metrics[key] == audited['operating_points'][alpha]['overall'][key]
        result['registered_operating_points'][alpha] = {'threshold': threshold,
            'calibration': counts(cal, threshold), **metrics}
    result['score_discrimination'] = roc(data)
    for name, predicate in [
        ('hard_benign', lambda r: r.get('hard_benign')),
        ('constructed_benign', lambda r: r.get('constructed')),
        ('natural_benign', lambda r: not r.get('constructed')),
    ]:
        subset = [r for r in data if r['label'] == 1 or predicate(r)]
        if any(r['label'] == 0 for r in subset):
            result['score_discrimination_attacks_vs_' + name] = roc(subset)
    result['score_discrimination_by_genre'] = {}
    for genre in sorted({r['genre'] for r in data}):
        subset = [r for r in data if r['genre'] == genre]
        if {r['label'] for r in subset} == {0, 1}:
            result['score_discrimination_by_genre'][genre] = roc(subset)
    return result, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='Fresh JSON output inside results/cnn_adoption_v1')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    assert output.is_relative_to(BASE.resolve()) and not output.exists()
    started = time.perf_counter()
    manifest = {}
    context = ['EVALUATION_POLICY.md', 'SENTRY_CNN_adoption_prompt.md', 'configs/cnn_adoption_v1.json',
               'reports/cnn_adoption_study.md', 'scripts/reassess_cnn_candidates.py',
               'results/cnn_adoption_v1/confirmation_freeze.json',
               'results/cnn_adoption_v1/independent_confirmation_score_check_r2.json',
               'results/cnn_adoption_v1/final_confirmation_comparison_summary_r2.json',
               'results/cnn_adoption_v1/final_profiling_audit_r2.json',
               'results/cnn_adoption_v1/final_hardware_adoption_audit.json']
    for name in context: manifest[name] = sha(ROOT / name)
    audit = read(BASE / 'independent_confirmation_score_check_r2.json')
    assert audit['complete'] and audit['targets_passed'] == 16 and not audit['errors']
    freeze = read(BASE / 'confirmation_freeze.json')
    for name, digest in freeze['source_sha256'].items():
        assert sha(ROOT / name) == digest
    checks = self_check()
    confirmation = {}
    dataset_ids = None
    by_target = {}
    for key, audited in audit['targets'].items():
        run_id, fmt = key.split(':')
        path = BASE / f'evaluation/confirmation/{run_id}/{fmt}/complete.json'
        result, data = analyze_complete(path, manifest, audited)
        confirmation[key] = result
        ids = [(r['document_id'],r['label'],r['source_group_id'],r['payload_group_id']) for r in data]
        if dataset_ids is None: dataset_ids = ids
        else: assert dataset_ids == ids
        by_target[key] = data
    development = {}
    for path in sorted((BASE / 'evaluation/development').glob('*_r2/*/complete.json')):
        result, _ = analyze_complete(path, manifest)
        development[result['target']] = result
    pair_differences = {}
    for seed in [42, 1337, 2026]:
        local = f'local_context_03_final_s{seed}_r2:w8a8_export'
        c98 = f'local_context_03_residual_replace_c98_final_s{seed}_r2:w8a8_export'
        ld, cd = by_target[local], by_target[c98]
        pair_differences[str(seed)] = {}
        for alpha in ['0.005', '0.01', '0.02']:
            lt = confirmation[local]['registered_operating_points'][alpha]['threshold']
            ct = confirmation[c98]['registered_operating_points'][alpha]['threshold']
            attacks = {'both_detect': 0, 'local_only': 0, 'c98_only': 0, 'both_miss': 0}
            benign = {'both_reject': 0, 'local_only': 0, 'c98_only': 0, 'both_accept': 0}
            for l, c in zip(ld, cd):
                lp, cp = l['risk_score'] > lt, c['risk_score'] > ct
                if l['label']:
                    k = 'both_detect' if lp and cp else 'local_only' if lp else 'c98_only' if cp else 'both_miss'
                    attacks[k] += 1
                else:
                    k = 'both_reject' if lp and cp else 'local_only' if lp else 'c98_only' if cp else 'both_accept'
                    benign[k] += 1
            pair_differences[str(seed)][alpha] = {'attacks': attacks, 'benign': benign,
                'scope': 'Paired counts at separately benign-calibrated registered policies; not equal-FPR or causal topology effect.'}
    result = {'complete': True, 'kind': 'secondary_retrospective_evaluation_policy_reassessment',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'authorization': 'User requested detailed reevaluation following the new repository evaluation policy, after original results were known.',
              'decision_stage': 'Explore / advance to bounded FPGA co-design research; not original-protocol ADOPT or deployment.',
              'no_new_hard_numeric_acceptance_threshold': True, 'no_training_or_inference': True,
              'original_frozen_protocol_and_outcomes_unchanged': True,
              'empirical_roc_points_use_evaluation_labels': True,
              'empirical_roc_cutoffs_not_selected_for_deployment': True,
              'self_check': checks, 'input_sha256': manifest,
              'confirmation': confirmation, 'development': development,
              'local_vs_c98_paired_decisions': pair_differences,
              'existing_uncertainty_artifact': 'results/cnn_adoption_v1/final_confirmation_comparison_summary_r2.json',
              'limitations': ['Retrospective selection and known confirmation panel cannot supply independent prospective validation.',
                  'AUC and empirical ROC points are descriptive; no population/seed inference or corrected multiple-selection claim.',
                  'Same 1000 attacks/2000 benign reused for each final seed; no independence inflation.',
                  'One-author hard-benign constructions, operational ancestry groups, sparse mechanisms and unknown model exposure remain.',
                  'No score-calibrated-on-confirmation threshold is proposed; registered calibration thresholds remain frozen.',
                  'Hardware costs distinguish actual packed artifacts, software measurements and conditional analytical mapping.'],
              'cpu_wall_seconds': time.perf_counter()-started}
    # Re-read all associations at completion to detect concurrent source/evidence edits.
    for name, digest in manifest.items(): assert sha(ROOT / name) == digest
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(output), 'sha256': sha(output), 'confirmation_targets': len(confirmation),
                      'development_targets': len(development), 'cpu_wall_seconds': result['cpu_wall_seconds']}))


if __name__ == '__main__':
    main()
