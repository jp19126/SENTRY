"""Execute only the prospectively frozen four-route primary scoring matrix."""
import argparse
from pathlib import Path
import subprocess
import sys
from common import CANONICAL, OUT, read, sha, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--freeze-sha256',required=True)
    a=p.parse_args()
    if sha(a.freeze)!=a.freeze_sha256:raise ValueError('Freeze digest differs')
    frozen=read(a.freeze)
    if not frozen['complete'] or not frozen['nomination_before_confirmation']:raise ValueError('Final nomination required')
    logs=OUT/'confirmation_logs';logs.mkdir(parents=True,exist_ok=True)
    result=[]
    for variant in ('source','fp32','w8a8_fp32_attention','w8a8_bf16_attention'):
        target=OUT/'evaluation/confirmation'/variant
        if target.parent.exists() and target.exists() and (target/'complete.json').exists():
            saved=read(target/'complete.json')
            if not saved['complete'] or saved['freeze_sha256']!=a.freeze_sha256 or saved['target']!=frozen['models'][variant]:raise ValueError('Existing result has a different freeze/model')
            for descriptor in saved['score_files'].values():
                if sha(target/descriptor['name'])!=descriptor['sha256']:raise ValueError('Existing scores changed')
            result.append({'variant':variant,'reused_complete_sha256':sha(target/'complete.json')});continue
        command=[sys.executable,'-B','scripts/bert_aie_v1/evaluate.py','--variant',variant,'--stage','confirmation',
            '--calibration-role',frozen['data']['calibration']['directory'],'--selection-role',frozen['data']['selection']['directory'],
            '--freeze',str(a.freeze),'--freeze-sha256',a.freeze_sha256,'--out',str(target),
            '--device','cuda','--batch-windows','16','--max-seconds','600']
        if variant.startswith('w8'):command+=['--calibration','results/bert_aie_v1/numerics/calibration_fit64.json']
        if variant=='w8a8_bf16_attention':command+=['--export','results/bert_aie_v1/export/w8a8_bf16_attention']
        with (logs/(variant+'.log')).open('a',encoding='utf-8') as log:
            completed=subprocess.run(command,cwd=CANONICAL,stdout=log,stderr=subprocess.STDOUT,timeout=650)
        result.append({'variant':variant,'command':command,'returncode':completed.returncode})
        write(OUT/'confirmation_matrix.json',result)
        if completed.returncode:raise RuntimeError('Inspect '+str(logs/(variant+'.log')))
        print('COMPLETED',variant,flush=True)
    write(OUT/'confirmation_matrix.json',result)


if __name__=='__main__':main()
