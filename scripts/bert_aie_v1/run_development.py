"""Finite three-format development matrix; root wraps one exclusive GPU lease."""
from pathlib import Path
import subprocess
import sys
from common import CANONICAL, OUT, read, write, sha

def main():
    logs=OUT/'development_logs';logs.mkdir(parents=True,exist_ok=True)
    variants=('source','fp32','w8a8_fp32_attention','w8a8_bf16_attention')
    result=[]
    for variant in variants:
        target=OUT/'evaluation/development'/variant
        if target.exists() and (target/'complete.json').exists():
            saved=read(target/'complete.json')
            assert saved['complete'] and saved['target']['variant']==variant
            assert saved['source_sha256']['evaluate.py']==sha(Path(__file__).with_name('evaluate.py'))
            result.append({'variant':variant,'reused_complete_sha256':sha(target/'complete.json')});continue
        command=[sys.executable,'-B','scripts/bert_aie_v1/evaluate.py','--variant',variant,'--stage','development',
          '--calibration-role','results/cnn_adoption_v1/prepared/token_r2/development_calibration',
          '--selection-role','results/cnn_adoption_v1/prepared/token_r2/development_selection',
          '--out',str(target),'--device','cuda','--batch-windows','16','--max-seconds','600']
        if variant.startswith('w8'):
            command+=['--calibration','results/bert_aie_v1/numerics/calibration_fit64.json']
        if variant=='w8a8_bf16_attention':command+=['--export','results/bert_aie_v1/export/w8a8_bf16_attention']
        with (logs/(variant+'.log')).open('a',encoding='utf-8') as log:
            completed=subprocess.run(command,cwd=CANONICAL,stdout=log,stderr=subprocess.STDOUT,timeout=650)
        result.append({'variant':variant,'command':command,'returncode':completed.returncode})
        write(OUT/'development_matrix.json',result)
        if completed.returncode:raise RuntimeError('Inspect '+str(logs/(variant+'.log')))
        print('COMPLETED',variant,flush=True)
    write(OUT/'development_matrix.json',result)

if __name__=='__main__':main()
