"""Final shared raw-document profiles; wrap with run_job --resource gpu_vendor."""
import argparse
from pathlib import Path
import subprocess
import sys
from common import ROOT, OUT, SHARED, read, sha, write, lock


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze',type=Path,required=True);p.add_argument('--freeze-sha256',required=True)
    a=p.parse_args()
    if sha(a.freeze)!=a.freeze_sha256:raise ValueError('Wrong freeze')
    frozen=read(a.freeze)
    if not frozen['complete']:raise ValueError('Final freeze required')
    for resource in ('vendor','gpu0'):
        if read(SHARED/'locks'/(resource+'.lock')/'owner.json')['task']!='bert_aie_v1':raise ValueError('Final profiles require both task leases')
    routes=[('source','cuda','cuda_graph'),('w8a8_bf16_attention','cuda','cuda_graph'),('source','cpu','eager')]
    jobs=[(scope,*route) for scope in ('common_contract','supplemental_confirmed') for route in routes]
    directory=OUT/'profiling/final';directory.mkdir(parents=True,exist_ok=True)
    runs=[]
    for scope,variant,device,backend in jobs:
        common=scope=='common_contract'
        evaluation=OUT/('evaluation/development' if common else 'evaluation/confirmation')/('source_cpu' if device=='cpu' else variant)
        complete=read(evaluation/'complete.json')
        if not common and complete['freeze_sha256']!=a.freeze_sha256:raise ValueError('Evaluation freeze differs')
        destination=directory/(scope+'_'+variant+'_'+device+'_'+backend+'.json')
        if destination.exists():raise ValueError('Preserve prior final profiles; inspect before resuming')
        command=[sys.executable,'-B','scripts/bert_aie_v1/profile_detector.py','--evaluation',str(evaluation),
            '--raw',str(ROOT/'results/cnn_adoption_v1/revision_r2/raw/development_selection.jsonl' if common else SHARED/'data/raw/new_selection.jsonl'),
            '--timing-manifest',str(SHARED/('prospective_v1/timing_manifest.json' if common else 'data/timing_manifest.json')),
            '--timing-manifest-sha256','6380d3ac38079ba5ec829a4e8cdef56452f306d4e281e7108ae9e133e08b1392' if common else '8d141711fcee46d65394c37460ef4bb377bd7346592ce11afa240ed6cf254162',
            '--device',device,'--backend',backend,'--threshold-policy','execution_calibrated' if backend=='cuda_graph' else 'reference',
            '--out',str(destination),'--max-seconds','600','--repeats','20','--warmups','3']
        if not common:command+=['--freeze',str(a.freeze),'--freeze-sha256',a.freeze_sha256]
        if variant.startswith('w8'):command+=['--calibration',str(OUT/'numerics/calibration_fit64.json'),'--export',str(OUT/'export/w8a8_bf16_attention')]
        with destination.with_suffix('.log').open('w',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=650)
        runs.append({'command':command,'returncode':result.returncode,'output':str(destination)})
        write(directory/'dispatch.json',{'runs':runs,'boundary':'Serial profiles with vendor and GPU leases and one CPU thread. Common-contract25developmentdocuments retain development-calibrated thresholds; supplemental18natural/stress documents use newly confirmed execution thresholds. No timing data enters threshold fitting or model selection; no board claim.'})
        if result.returncode:raise RuntimeError('Inspect '+str(destination.with_suffix('.log')))
        print('COMPLETED',scope,variant,device,backend,flush=True)


if __name__=='__main__':
    # Exclude the task's corpus comparison/bootstrap CPU work too. Vendor/GPU
    # leases alone do not prevent those host measurements from being disturbed.
    with lock('cpu',timeout=1800):main()
