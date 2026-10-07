"""Summarize saved Goal 3 scores; performs no model inference or fitting."""
from pathlib import Path
import csv
import json
import math

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'results' / 'goal3'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def ranks(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    output = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[ordered[start]] == values[ordered[end]]:
            end += 1
        value = (start + 1 + end) / 2
        for index in ordered[start:end]:
            output[index] = value
        start = end
    return output


def spearman(left, right):
    x, y = ranks(left), ranks(right)
    mx, my = sum(x)/len(x), sum(y)/len(y)
    numerator = sum((a-mx)*(b-my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
    return numerator/denominator if denominator else None


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    floating = read(DEST / 'floating_reference.json')
    uniform = [('FP32', floating['metrics'], None)]
    for name in ['ptq_w8_a8', 'ptq_w4_a8', 'qat_w8_a8', 'qat_w4_a8']:
        record = read(DEST / name / 'candidate.json')
        uniform.append((name, record['metrics'], record))
    rows = []
    nearby = []
    for name, metrics, record in uniform:
        primary = metrics['primary']['candidate_scoring']
        fixed = metrics.get('at_floating_threshold', primary)
        row = {'name': name, 'refit_threshold': metrics['primary']['threshold'],
               'refit_recall': primary['recall'], 'refit_fpr': primary['fpr'],
               'refit_tp': primary['tp'], 'refit_fn': primary['fn'], 'refit_fp': primary['fp'],
               'fixed_fp_threshold_recall': fixed['recall'], 'fixed_fp_threshold_fpr': fixed['fpr'],
               'fixed_fp_threshold_fp': fixed['fp'],
               'ordinary_accuracy': metrics['ordinary_threshold_0_5']['accuracy'],
               'ordinary_fp': metrics['ordinary_threshold_0_5']['fp'],
               'mean_document_bce': metrics['candidate_mean_document_bce'],
               'logical_payload_bytes': record['storage_estimate']['logical_payload_bytes'] if record else read(DEST/'ptq_w8_a8'/'candidate.json')['storage_estimate']['saved_parameter_representation_bytes'],
               'actual_saved_weight_state_bytes': record['training_record']['actual_saved_quantized_state_bytes'] if record and record['training_record'] else ((ROOT/floating['checkpoint']/'model.safetensors').stat().st_size if not record else None)}
        rows.append(row)
        for target, operation in metrics['operating_points'].items():
            m = operation['candidate_scoring']
            nearby.append({'name': name, 'calibration_fpr_target': float(target),
                           'threshold': operation['threshold'],
                           'empirical_calibration_fpr': operation['temporary_threshold_benign']['fpr'],
                           'scoring_recall': m['recall'], 'scoring_fpr': m['fpr'],
                           'tp': m['tp'], 'fn': m['fn'], 'fp': m['fp'], 'tn': m['tn']})
    write_csv(DEST/'uniform_summary.csv', rows)
    write_csv(DEST/'nearby_operating_points.csv', nearby)
    singles = read(DEST/'single_group_sensitivity.json')
    details = []
    for entry in singles:
        record = read(DEST/entry['name']/'candidate.json')
        m = record['metrics']; q = m['primary']['candidate_scoring']
        fixed_changes = entry['at_reference_threshold']
        refit_changes = entry['after_threshold_refit']
        details.append({'group': entry['group'], 'name': entry['name'],
                        'ordinary_accuracy': m['ordinary_threshold_0_5']['accuracy'],
                        'ordinary_error': 1-m['ordinary_threshold_0_5']['accuracy'],
                        'bce_change': entry['mean_document_bce_change'],
                        'window_logit_mse': entry['window_logit_mse'],
                        'window_logit_max_abs_error': entry['window_logit_max_abs_error'],
                        'recall_loss_after_refit': entry['recall_loss_after_refit'],
                        'refit_fp': q['fp'], 'refit_fpr': q['fpr'],
                        'fpr_change_after_refit': entry['fpr_change_after_refit'],
                        'new_misses_fixed_w8': len(fixed_changes['new_misses']),
                        'new_misses_refit': len(refit_changes['new_misses']),
                        'new_fp_fixed_w8': len(fixed_changes['new_false_positives']),
                        'removed_fp_fixed_w8': len(fixed_changes['removed_false_positives']),
                        'new_fp_refit': len(refit_changes['new_false_positives']),
                        'removed_fp_refit': len(refit_changes['removed_false_positives']),
                        'mean_score_change': refit_changes['mean_score_change'],
                        'mean_absolute_score_change': refit_changes['mean_absolute_score_change']})
    for column in ['ordinary_error','bce_change','window_logit_mse','refit_fp']:
        for row, rank in zip(details, ranks([v[column] for v in details])):
            row[column+'_ascending_rank'] = rank
    write_csv(DEST/'single_group_rank_comparison.csv', details)
    summary = {'maps': len(details), 'comparison_unit': '16 dependent interventions on one selected checkpoint; seed42; fixed development groups',
               'rank_method': 'Spearman correlation computed as Pearson correlation of average tied ranks; lower numerical error/loss and lower FPR are aligned',
               'rank_correlations_with_refit_false_positive_count': {
                   column: spearman([v[column] for v in details], [v['refit_fp'] for v in details])
                   for column in ['ordinary_error','bce_change','window_logit_mse']},
               'recall_rank_correlation': None,
               'recall_rank_reason': 'All16 recall-loss values equal zero, so a recall ranking is undefined.',
               'inference': 'Descriptive only: no p-values or seed/generalization confidence intervals from these dependent maps.',
               'single_group_fpr_range': [min(v['refit_fpr'] for v in details),max(v['refit_fpr'] for v in details)],
               'all_single_group_misses_fixed_w8': sum(v['new_misses_fixed_w8'] for v in details),
               'all_single_group_misses_refit': sum(v['new_misses_refit'] for v in details),
               'uniform_results': rows,
               'pair_results': read(DEST/'pair_sensitivity.json'),
               'source_files': ['floating_reference.json','*/candidate.json','single_group_sensitivity.json','pair_sensitivity.json']}
    (DEST/'sensitivity_analysis.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('pair_results',)}, indent=2))


if __name__ == '__main__':
    main()
