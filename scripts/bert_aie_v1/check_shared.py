"""Independently reconstruct every shared raw document and complete token tail."""
from collections import Counter
import numpy as np
from tokenizers import Tokenizer
from common import CANONICAL, SHARED, OUT, read, rows, sha, write

def main():
    dest=SHARED/'prospective_v1';manifest=read(dest/'manifest.json')
    tok=Tokenizer.from_file(str(CANONICAL/'checkpoints/floating/epoch_3/tokenizer.json'))
    tok.no_padding();tok.no_truncation()
    old=read(CANONICAL/'results/cnn_adoption_v1/revision_r2/data_protocol.json')
    old_groups={x for v in old['source_role_membership'].values() for x in v}
    groups={};payloads={};report={}
    for role,info in manifest['roles'].items():
        folder=dest/role
        assert all(sha(folder/n)==h for n,h in info['files'].items())
        raw=list(rows(folder/'raw.jsonl'));docs=list(rows(folder/'documents.jsonl'))
        ids=np.load(folder/'ids.npy');masks=np.load(folder/'masks.npy');starts=np.load(folder/'starts.npy')
        assert len(raw)==len(docs)==info['documents']
        assert starts.shape==(len(docs)+1,) and starts[0]==0 and starts[-1]==len(ids)
        assert ids.shape==masks.shape==(info['windows'],256)
        assert np.all(np.diff(starts)>0)
        ids_seen=set()
        for n,(r,d) in enumerate(zip(raw,docs)):
            assert r['document_id']==d['document_id'] and d['document_id'] not in ids_seen
            ids_seen.add(d['document_id'])
            tokens=tok.encode(r['text'],add_special_tokens=False).ids
            lo,hi=map(int,starts[n:n+2]);reconstructed=[]
            for j,w in enumerate(range(lo,hi)):
                used=int(masks[w].sum())
                assert 2<=used<=256 and np.all(masks[w,:used]==1) and np.all(masks[w,used:]==0)
                assert ids[w,0]==101 and ids[w,used-1]==102 and np.all(ids[w,used:]==0)
                content=ids[w,1:used-1].tolist()
                assert content==tokens[j*190:j*190+254]
                if j:
                    assert reconstructed[-64:]==content[:64]
                    reconstructed.extend(content[64:])
                else:reconstructed.extend(content)
            assert reconstructed==tokens and d['content_tokens']==len(tokens) and d['windows']==hi-lo
            assert r['label']==int(r['construction_kind']=='attack')
        groups[role]={r['source_group_id'] for r in docs}
        payloads[role]={r['payload_group_id'] for r in docs if r.get('payload_group_id')}
        assert not groups[role]&old_groups
        report[role]={'documents':len(docs),'windows':len(ids),'source_groups':len(groups[role]),'payload_groups':len(payloads[role]),'maximum_windows':max(d['windows'] for d in docs),'labels':dict(Counter(d['label'] for d in docs))}
    assert not groups['benign_calibration']&groups['confirmation']
    assert not payloads['benign_calibration']&payloads['confirmation']
    assert report['benign_calibration']['labels'].get(1,0)==0
    # Serialized tokenizer path must reproduce actual old prepared development
    # windows too, not only agree with this task's newly generated arrays.
    dev=CANONICAL/'results/cnn_adoption_v1/prepared/token_r2/development_selection'
    olddocs=list(rows(dev/'documents.jsonl'));raw={r['document_id']:r for r in rows(CANONICAL/'results/cnn_adoption_v1/revision_r2/raw/development_selection.jsonl')}
    ids=np.load(dev/'ids.npy',mmap_mode='r');masks=np.load(dev/'masks.npy',mmap_mode='r');starts=np.load(dev/'starts.npy')
    for n,d in enumerate(olddocs):
        tokens=tok.encode(raw[d['document_id']]['text'],add_special_tokens=False).ids
        lo,hi=map(int,starts[n:n+2]);assert hi-lo==max(1,(max(0,len(tokens)-254)+189)//190+1)
        for j,w in enumerate(range(lo,hi)):
            used=int(masks[w].sum());assert ids[w,1:used-1].tolist()==tokens[j*190:j*190+254]
    result={'complete':True,'manifest_sha256':sha(dest/'manifest.json'),'roles':report,'all_raw_tails_reconstructed':True,
            'old_development_tokenizer_regression_documents':len(olddocs),'shared_old_source_intersections':0,'new_source_and_payload_role_intersections':0,
            'limits':'Operational source groups/exact templates; no semantic-unseen or independent natural attack claim.'}
    write(OUT/'shared_data_check.json',result);print(result)

if __name__=='__main__':main()
