"""Bounded scoring of an explicit frozen target list; never discovers inputs."""
import argparse
import contextlib
import json
import subprocess
import sys
from pathlib import Path

from data import file_sha, load, save
from evaluate import (ROLES, common, confirm_freeze, numerical_settings,
                      prepared_identity, source_hashes, target_descriptors, within)
from resource_lock import ResourceLock
from train import setup


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--canonical-root',type=Path,required=True)
    p.add_argument('--target-list',type=Path,required=True)
    p.add_argument('--stage',choices=('development','confirmation'),required=True)
    p.add_argument('--freeze',type=Path)
    p.add_argument('--target-ids',nargs='+')
    p.add_argument('--out-root',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--timeout-per-score',type=int,default=1200)
    a=p.parse_args(); root=a.canonical_root.resolve()
    out=within(root,a.out_root); contract,_=common(root)
    descriptors=target_descriptors(root,load(a.target_list),contract)
    selected=a.target_ids or list(descriptors)
    if len(set(selected))!=len(selected) or not set(selected)<=set(descriptors):
        raise ValueError('Duplicate or unknown targets')
    setup(42,a.device,1); settings=numerical_settings(a.device)
    roles=sorted(ROLES[a.stage]); jobs=[]
    for ident in selected:
        for role in roles:
            output=out/ident/(role+'.jsonl'); side=output.with_suffix('.complete.json')
            if a.stage=='confirmation':
                if not a.freeze: raise ValueError('Confirmation needs prior freeze')
                frozen=confirm_freeze(root,a.freeze,ident,role)
                if descriptors[ident]!=frozen['targets'][ident]:
                    raise ValueError('Provided target list differs from frozen target')
            if output.exists() or side.exists():
                if not output.exists() or not side.exists(): raise ValueError('Incomplete prior score; preserve and inspect: '+str(output))
                done=load(side); identity=prepared_identity(root,role)
                checks={'complete':True,'stage':a.stage,'role':role,'target_id':ident,
                        'target':descriptors[ident],'scores_sha256':file_sha(output),
                        'prepared_manifest_sha256':identity['manifest_sha256'],
                        'source_sha256':source_hashes(),'numerical_settings':settings,
                        'freeze_sha256':file_sha(a.freeze) if a.freeze else None}
                for key,value in checks.items():
                    if done.get(key)!=value: raise ValueError('Prior score association differs '+key+': '+str(output))
                print(json.dumps({'reused':str(output)}),flush=True)
                continue
            jobs.append((ident,role,output))
    guard=ResourceLock(root/'results/dual_track_v1/shared','gpu0') if a.device.startswith('cuda') else contextlib.nullcontext()
    with guard:
        for ident,role,output in jobs:
            if (root/'results/kd_cnn_v2/stop.request').exists(): raise RuntimeError('Study stop requested')
            command=[sys.executable,'-B',str(Path(__file__).with_name('evaluate.py')),
                     '--canonical-root',str(root),'--stage',a.stage,'--target-list',str(a.target_list.resolve()),
                     '--target-id',ident,'--role',role,'--out',str(output),'--device',a.device]
            if a.freeze: command+=['--freeze',str(a.freeze.resolve())]
            subprocess.run(command,check=True,timeout=a.timeout_per_score)
    save(out/'queue_complete.json',{'stage':a.stage,'targets':selected,'roles':roles,
                                  'device':a.device,'source_sha256':source_hashes()})


if __name__=='__main__': main()
