"""Independent score-free membership and primary token transport audit.

Never imports model/scoring code or opens old confirmation rows. The only old
study material read is frozen membership metadata. No classifier is fitted.
"""
from __future__ import annotations
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from transformers import AutoTokenizer
from common import ROOT,SHARED,sha,read,rows,write

PIN='614549f8a1eeac21cc4168f551812fe6398c3da2e856529f81bb55910ce1b398'

def audit(output):
    started=time.monotonic();data=SHARED/'data';protocol=read(data/'protocol.json');summary=read(data/'prepared/summary.json')
    if sha(data/'protocol.json')!=PIN:raise ValueError('Primary protocol identity changed')
    contract=read(SHARED/'contract.json');oldpath=ROOT/contract['base_data']['protocol'];old=read(oldpath)
    assert sha(oldpath)==protocol['source_provenance']['old_protocol_sha256']
    excluded={v for values in old['source_role_membership'].values() for v in values}
    tokpath=ROOT/contract['tokenizer']['path']
    for name,digest in summary['tokenizer_files_sha256'].items():assert sha(tokpath/name)==digest
    tok=AutoTokenizer.from_pretrained(tokpath,local_files_only=True,use_fast=True)
    assert (tok.cls_token_id,tok.sep_token_id,tok.pad_token_id)==(101,102,0)
    errors=[];members={};roles={};global_documents=set();text_roles={};label_diagnostics={};window_total=0;primary_docs=0
    def check(ok,msg):
        if not ok:errors.append(msg)
    for role in protocol['role_plan']:
        rawpath=data/'raw'/(role+'.jsonl');folder=data/'prepared'/role
        raw=list(rows(rawpath));docs=list(rows(folder/'documents.jsonl'));manifest=read(folder/'manifest.json')
        check(sha(rawpath)==protocol['raw_role_sha256'][role],role+': raw hash')
        check(sha(folder/'manifest.json')==summary['roles'][role]['manifest_sha256'],role+': manifest hash')
        for filename,digest in manifest['artifact_sha256'].items():check(sha(folder/filename)==digest,role+': '+filename+' hash')
        check(len(raw)==len(docs)==protocol['role_counts'][role]['documents'],role+': document count')
        source={d['source_group_id'] for d in raw};payload={d['payload_group_id'] for d in raw if d.get('payload_group_id')};context={d['context_ancestry_group_id'] for d in raw if d.get('context_ancestry_group_id')}
        members[role]={'source':source,'payload':payload,'context':context}
        for kind,values in members[role].items():
            check(values==set(protocol[kind+'_role_membership'][role]),role+': '+kind+' exact membership')
        check(not source&excluded,role+': overlaps old source memberships')
        labels=Counter(str(d['label']) for d in raw);hard=sum(bool(d.get('hard_benign')) for d in raw)
        check(labels.get('1',0)==protocol['role_counts'][role]['positive'],role+': positive count')
        check(hard==protocol['role_counts'][role]['hard_benign'],role+': hard benign count')
        if 'calibration' in role:check(labels.get('1',0)==0,role+': calibration not benign-only')
        family_labels=defaultdict(Counter);source_labels=defaultdict(Counter);prefix=defaultdict(Counter);backgrounds=defaultdict(set)
        for d,m in zip(raw,docs):
            did=d['document_id'];check(did not in global_documents,role+': duplicate document identity');global_documents.add(did)
            text=d['text'];digest=hashlib.sha256(text.encode('utf-8')).hexdigest()
            check(digest==d['text_sha256'],role+': text hash '+did)
            if digest in text_roles:check(text_roles[digest]==role,role+': exact text crosses role')
            text_roles[digest]=role
            check(all(m.get(k)==v for k,v in d.items() if k!='text'),role+': prepared metadata '+did)
            check(d['label'] in (0,1),role+': invalid label')
            check(d.get('construction_role')==role,role+': wrong construction role')
            check(not d.get('prior_project_training_exposed'),role+': recorded prior project training exposure')
            if d.get('payload_group_id'):family_labels[d['payload_group_id']][str(d['label'])]+=1
            source_labels[d['source_group_id']][str(d['label'])]+=1;backgrounds[d['source_group_id']].add(d['background_text_sha256'])
            # Actual text cues are described, never used to select model/threshold.
            prefix[' '.join(text.strip().split()[:8])][str(d['label'])]+=1
            for lo,hi in d.get('injection_spans',[]):check(d['label']==1 and 0<=lo<hi<=len(text),role+': span bounds')
        check(all(len(v)==1 for v in backgrounds.values()),role+': background hash changes inside source group')
        paired=sum(v.get('0',0)>0 and v.get('1',0)>0 for v in source_labels.values())
        # A positive's source also occurs with a benign document; natural genre alone
        # cannot explain class1. Hard benign wrappers remain an authored distinction.
        check(all(v.get('0',0)>0 for v in source_labels.values() if v.get('1',0)),role+': positive-only source background')
        if labels.get('1',0):check(all(v.get('0',0)>0 and v.get('1',0)>0 for v in family_labels.values()),role+': label-exclusive payload family')
        label_diagnostics[role]={'paired_source_groups':paired,'payload_family_label_counts':{k:dict(v) for k,v in family_labels.items()},
                                 'most_common_first8word_prefixes':[{'prefix':p,'labels':dict(v)} for p,v in sorted(prefix.items(),key=lambda kv:-sum(kv[1].values()))[:5]]}
        role_result={'documents':len(raw),'labels':dict(labels),'hard_benign':hard,'source_groups':len(source),'payload_families':len(payload),'context_families':len(context),'raw_sha256':sha(rawpath)}
        if role in {'new_calibration','new_confirmation'}:
            ids=np.load(folder/'ids.npy',mmap_mode='r');masks=np.load(folder/'masks.npy',mmap_mode='r');starts=np.load(folder/'starts.npy',mmap_mode='r')
            cstarts=np.load(folder/'content_start.npy',mmap_mode='r');lengths=np.load(folder/'document_content_length.npy',mmap_mode='r')
            check(starts[0]==0 and starts[-1]==len(ids) and np.all(np.diff(starts)>0),role+': document starts')
            check(ids.shape==masks.shape and ids.shape[1]==256,role+': ID/mask dimensions')
            checked=0;tail_windows=0;multi=0
            for di,d in enumerate(raw):
                content=tok(d['text'],add_special_tokens=False,truncation=False,return_attention_mask=False,return_token_type_ids=False,verbose=False)['input_ids']
                expected_starts=[];at=0
                while True:
                    expected_starts.append(at)
                    if at+254>=len(content):break
                    at+=190
                lo,hi=int(starts[di]),int(starts[di+1]);multi+=len(expected_starts)>1
                check(hi-lo==len(expected_starts),role+': all window count '+d['document_id'])
                check(int(lengths[di])==len(content)==docs[di]['content_tokens'],role+': content length')
                for wi,at in enumerate(expected_starts):
                    w=lo+wi;tokens=[101]+content[at:at+254]+[102];valid=len(tokens)
                    expected=np.asarray(tokens+[0]*(256-valid),dtype=ids.dtype)
                    expected_mask=np.asarray([1]*valid+[0]*(256-valid),dtype=masks.dtype)
                    check(np.array_equal(ids[w],expected),role+': token transport '+str(w))
                    check(np.array_equal(masks[w],expected_mask),role+': masks/padding '+str(w))
                    check(int(cstarts[w])==at,role+': content offset '+str(w))
                    checked+=1;tail_windows+=valid<256
            primary_docs+=len(raw);window_total+=checked
            role_result.update(exactly_reconstructed_documents=len(raw),exactly_reconstructed_windows=checked,padded_tail_windows=tail_windows,multiwindow_documents=multi)
        roles[role]=role_result
    intersections={}
    names=list(members)
    for kind in ('source','payload','context'):
        intersections[kind]={a+' / '+b:len(members[a][kind]&members[b][kind]) for i,a in enumerate(names) for b in names[i+1:]}
        check(not any(intersections[kind].values()),kind+': cross-role intersection')
    result={'schema':'bert_aie_primary_data_check_v1','passed':not errors,'errors':errors,'protocol_sha256':PIN,'audit_script_sha256':sha(Path(__file__)),
            'old_protocol_sha256':sha(oldpath),'old_source_groups_excluded':len(excluded),'roles':roles,'role_intersections':intersections,
            'primary_reconstructed_documents':primary_docs,'primary_reconstructed_windows':window_total,'label_diagnostics':label_diagnostics,
            'model_scores_read':False,'old_confirmation_rows_read':False,'sibling_outcome_reports_read':False,'model_training_or_inference':False,
            'seconds':time.monotonic()-started,'threads_policy':'OMP_NUM_THREADS1, tokenizer parallelism disabled',
            'findings':['Source/payload/context identities and exact text are disjoint across declared roles, including sibling FIT transfer, conditional on zero reported errors.',
                        'All positive source backgrounds also have benign examples; payload families contain both classes where positives occur.',
                        'Model arrays contain only text-derived WordPiece IDs and masks; metadata/labels/IDs/URLs are absent from model input.',
                        'Hard benign contexts are authored discussion wrappers, while positives are operative redirection; frequent wrapper prefixes are a potential learnable construction cue, not evidence of natural attack generalization.'],
            'limits':['Role IDs are author-assigned ancestry, not proof of semantic independence. Exact set/text checks cannot detect paraphrase or semantic overlap.',
                      'One-author response-redirection payloads/contexts and Wikipedia backgrounds; no natural-population or unseen-mechanism claim.',
                      'Old source exclusion verified from frozen role metadata only; old confirmation rows/scores and protected data remain unopened.',
                      'No target-LLM outcomes inspected; labels describe authored intent. Prefix/source checks cannot rule out all label shortcuts.',
                      'Primary calibration/confirmation data were inspected score-free for this audit; this is not a blind reannotation.']}
    write(output,result);print(json.dumps({k:result[k] for k in ('passed','errors','primary_reconstructed_documents','primary_reconstructed_windows','seconds')},indent=2))
    if errors:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'results/bert_aie_v1/primary_data_check.json')
    audit(p.parse_args().output)
