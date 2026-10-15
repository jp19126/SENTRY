"""Compact descriptive summary from immutable C++/Python saved scores only."""
from pathlib import Path
from common import ROOT,read,rows,sha,write
from analyze import strata

def main():
    folder=ROOT/'results/bert_aie_v1/evaluation/cpp_confirmation';complete=read(folder/'complete.json')
    reference=ROOT/'results/bert_aie_v1/evaluation/confirmation/w8a8_bf16_attention'
    if sha(reference/'complete.json')!=complete['reference_complete_sha256']:raise ValueError('Reference result identity changed')
    original=read(reference/'complete.json');scores={}
    for role,info in complete['score_files'].items():
        if sha(folder/info['name'])!=info['sha256']:raise ValueError('C++ scores changed')
        scores[role]=list(rows(folder/info['name']))
    points={}
    for alpha,threshold in complete['python_thresholds'].items():
        c=complete['comparisons']['selection'][alpha]
        points[alpha]={'python_threshold':threshold,'cpp_own_threshold':complete['cpp_own_thresholds'][alpha],
                       'python_confirmation_metrics':original['metrics'][alpha],
                       'cpp_fixed_threshold_metrics':c['cpp_fixed_threshold_metrics'],'cpp_own_threshold_metrics':c['cpp_own_threshold_metrics'],
                       'fixed_decision_changes':len(c['fixed_threshold_disagreements']),'own_decision_changes':len(c['own_calibrated_disagreements']),
                       'near_threshold_documents':len(c['near_threshold'])}
    result={'schema':'bert_cpp_execution_summary_v1','source_complete_sha256':sha(folder/'complete.json'),'roles':complete['roles'],
            'max_absolute_logit_difference':complete['max_logit_error_when_python_logits_saved'],'mapping_bit_identical':complete['mapping_bit_identical'],
            'operating_points':points,'primary_strata_cpp_fixed':strata(scores['selection'],complete['python_thresholds']['0.01']),
            'primary_strata_cpp_own':strata(scores['selection'],complete['cpp_own_thresholds']['0.01']),
            'decision_transfer_qualified':False,
            'interpretation':'Portable C++ is not numerically or decision-identical to Python BF16. Own calibration is separately reported and does not establish native AIE/RTL decision transfer.',
            'limits':['Strata are descriptive point estimates; no new cluster confidence intervals computed for C++ here.',
                      'One-author constructed response-redirection panel; no downstream attack-success or natural-population claim.',
                      'Native AIE accumulation and HLS nonlinear approximations remain independently unqualified.']}
    write(folder/'summary.json',result)
    print({'summary':str(folder/'summary.json'),'points':points,'hard_benign_cpp_own':result['primary_strata_cpp_own']['hard_benign']})

if __name__=='__main__':main()
