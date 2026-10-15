"""Freeze common identities before training; no confirmation scores are read."""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timezone
from resource_lock import ResourceLock


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def freeze(canonical,shared,config):
    canonical,shared=Path(canonical),Path(shared)
    with ResourceLock(shared,'setup'):
        out=shared/'contract.json'
        if out.exists():
            prior=read(out)
            if prior['teacher'].get('checkpoint_sha256',prior['teacher'].get('sha256'))!=config['teacher']['checkpoint_sha256']:
                raise ValueError('Existing common teacher differs; do not replace contract')
            if prior.get('input',{}).get('stride_content_tokens',190)!=190:
                raise ValueError('Common window contract differs')
            print(json.dumps({'reused':str(out),'sha256':sha(out)}));return
        base=canonical/'results/cnn_adoption_v1/prepared/token_r2/fit/manifest.json'
        protocol=shared/'data/protocol.json'
        roles=('fit_transfer','new_selection','new_development_calibration','new_calibration','new_confirmation')
        manifests={role:read(shared/'data/prepared'/role/'manifest.json') for role in roles}
        checkpoint=canonical/config['teacher']['checkpoint']
        if sha(checkpoint)!=config['teacher']['checkpoint_sha256']:raise ValueError('Teacher identity changed')
        # The timing panel is development material selected solely by token length.
        metadata=[json.loads(s) for s in (shared/'data/prepared/new_selection/documents.jsonl').read_text(encoding='utf-8').splitlines()]
        raw={r['document_id']:r for r in [json.loads(s) for s in (shared/'data/raw/new_selection.jsonl').read_text(encoding='utf-8').splitlines()]}
        import numpy as np
        starts=np.load(shared/'data/prepared/new_selection/starts.npy',allow_pickle=False)
        candidates=sorted(zip(metadata,np.diff(starts).tolist()),key=lambda r:(r[1],r[0]['document_id']))
        indices=sorted(set(np.linspace(0,len(candidates)-1,24,dtype=int).tolist()))
        timing=[]
        for index in indices:
            row,n=candidates[index];r=raw[row['document_id']]
            timing.append({'document_id':row['document_id'],'text':r['text'],'text_sha256':r['text_sha256'],
                           'expected_window_count':n,'kind':'length-stratified development document'})
        timing_path=shared/'timing_documents.jsonl'
        with timing_path.open('x',encoding='utf-8') as f:
            for r in timing:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        value={'version':'dual_track_v1','created_utc':datetime.now(timezone.utc).isoformat(),'creator':'kd_cnn_v2',
            'canonical_root_linux':'/home/jp19126/Projects/SENTRY',
            'teacher':{**config['teacher'],'checkpoint_sha256':sha(checkpoint),'selection':'prior matched-data development only; fixed seed42'},
            'tokenizer':{'id':config['teacher']['backbone'],'revision':config['teacher']['revision'],'local_files_only':True},
            'base_fit':{'path':'results/cnn_adoption_v1/prepared/token_r2/fit','manifest_sha256':sha(base),'manifest':read(base)},
            'new_data':{'protocol_relative_path':'data/protocol.json','protocol_sha256':sha(protocol),
                'roles':{role:{'relative_path':'data/prepared/'+role,'manifest_sha256':sha(shared/'data/prepared'/role/'manifest.json'),
                              'manifest':manifests[role]} for role in roles},
                'boundary':'New authored response-redirection/discussion constructions and unused public WikiText TRAIN sources; no unseen-mechanism or natural population claim'},
            'windowing':config['windowing'],'reporting_policy':config['reporting_policy'],
            'timing':{'relative_path':'timing_documents.jsonl','sha256':sha(timing_path),'documents':len(timing),
                'boundary':'raw document tokenization, every window/tail, copies, model, aggregation, strict decision; hold document until decision',
                'warm_repetitions':20,'cold_process_observations':3,'device_batch':1,'sampling':'length stratified; not deployment frequency'},
            'target':{'board':'VEK280','part':'xcve2802-vsvh1760-2MP-e-S','objective_period_ns':5.0,'physical_authorized':False},
            'locks':{'setup':'locks/setup.lock','gpu0':'locks/gpu0.lock','vendor':'locks/vendor.lock','protocol':'atomic mkdir; inspect owner.json; no automatic stale-lock eviction'},
            'release':'Candidates/exports freeze before each task reads own confirmation scores; do not read sibling score reports while tuning',
            'closed_inputs':['project final-calibration','project final-test','NotInject and copies'],
            'supplemental_fit':'Versioned addition, never mutation of base membership or teacher'}
        with out.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False)
        print(json.dumps({'created':str(out),'sha256':sha(out),'timing_documents':len(timing)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--canonical-root',required=True);p.add_argument('--shared-root',required=True);p.add_argument('--config',required=True)
    a=p.parse_args();freeze(a.canonical_root,a.shared_root,read(a.config))
