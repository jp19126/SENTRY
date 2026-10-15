"""Summarize saved final timings and paired quality; performs no model execution."""
from pathlib import Path
import json
import numpy as np
from common import OUT, read, sha, write


def distribution(values):
    a=np.asarray(values,dtype=float)
    return {'n':len(a),'mean':float(a.mean()),'p50':float(np.quantile(a,.5)),
            'p95':float(np.quantile(a,.95))}


def main():
    profiles={}
    for scope in ('common_contract','supplemental_confirmed'):
        for route in ('source_cuda_cuda_graph','w8a8_bf16_attention_cuda_cuda_graph','source_cpu_eager'):
            p=OUT/'profiling/final'/f'{scope}_{route}.json';d=read(p)
            if not d['complete']:raise ValueError(str(p))
            samples=d['samples'];repetitions=[r for s in samples for r in s['repetitions']]
            warm=[r['complete_ms'] for r in repetitions]
            by_windows={str(w):distribution([r['complete_ms'] for s in samples if s['windows']==w for r in s['repetitions']])
                        for w in sorted({s['windows'] for s in samples})}
            groups={kind:distribution([r['complete_ms'] for s in samples if bool(s.get('timing_only'))==stress for r in s['repetitions']])
                    for kind,stress in [('natural',False),('stress',True)] if any(bool(s.get('timing_only'))==stress for s in samples)}
            profiles[f'{scope}_{route}']={'source_sha256':sha(p),'documents':len(samples),
                'windows_per_workload':sum(s['windows'] for s in samples),'warm_document_ms':distribution(warm),
                'serial_documents_per_second':1000*len(warm)/sum(warm),
                'by_window_count_ms':by_windows,'natural_stress_ms':groups,
                'mean_stages_ms':{k:float(np.mean([r[k] for r in repetitions])) for k in ('preprocessing_ms','transfer_ms','network_ms','return_aggregation_ms')},
                'model_load_seconds':d['model_load_seconds'],'tokenizer_load_seconds':d['tokenizer_load_seconds'],
                'graph_setup_seconds':d['graph_setup_seconds'],'thresholds':d['thresholds'],
                'profile_job_seconds':d['elapsed_seconds'],
                'boundary':d['boundary']}
    quality={}
    for name in ('bf16','fp32'):
        p=OUT/'analysis'/f'confirmation_{name}.json';d=read(p)
        paired=d['comparisons']['0.01']
        quality[name]={'source_sha256':sha(p),'metrics':d['metrics'],
            'auc':d['roc']['auc'],'principal_paired':{k:v for k,v in paired.items() if k!='candidate_strata_fixed_reference_threshold'},
            'principal_strata':d['strata']['0.01']}
    write(OUT/'final_software_summary.json',{'profiles':profiles,'quality':quality,
        'interpretation':'Profile quantiles pool 20 repeats of a deliberately stratified fixed document sample; they are not traffic-frequency-weighted production quantiles. Serial throughput is the inverse summed measured document service time, not a concurrent throughput experiment. network_ms is synchronized host elapsed time, not device-only GPU events. Loading/capture were measured in fresh worker processes but OS/file/GPU caches were not flushed.'})
    print(json.dumps({'profiles':{k:{'warm':v['warm_document_ms'],'docs_s':v['serial_documents_per_second'],'cold_load_s':v['model_load_seconds'],'windows':v['windows_per_workload']} for k,v in profiles.items()}},indent=2))


if __name__=='__main__':main()
