"""Produce portable C cases from saved tensors or frozen integer own-reference.

Trained mode is bounded to fixed32 train documents and synthetic corners.
No protected-data access, vendor tool or hardware operation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import random

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'results/span_coupled_pilot_v1'


def trained_fixtures(args,out,torch):
    from span_train import Budget,load_split,batch
    from span_numerics import IntegerSpan
    budget=Budget(args.max_seconds,out)
    data,docs=load_split(args.prepared.resolve(),'train')
    selected=[]
    for label in (0,1):
        candidates=[i for i,d in enumerate(docs) if d['label']==label]
        random.Random(42+label).shuffle(candidates);selected.extend(candidates[:16])
    indices=[]
    for d in selected:
        first,end=map(int,data['starts'][d:d+2]);indices.extend(range(first,end))
    model=IntegerSpan(args.export.resolve(),'cpu',False);records=[]
    with torch.inference_mode():
        for at in range(0,len(indices),8):
            budget.check();inp=batch(data,indices[at:at+8],'cpu',torch)
            result=model(**inp,trace=True)
            file=out/('fixture_%02d.pt'%(at//8));torch.save(dict(inputs=inp,outputs=result),file)
            records.append(file)
        # Synthetic boundary inputs borrow only IDs from the fixed train subset.
        first=batch(data,[indices[0]],'cpu',torch)
        tokens=first['input_ids'][0,1:int(first['attention_mask'][0].sum())-1].tolist()
        if not tokens:raise ValueError('First fixed training fixture unexpectedly empty')
        seq=(tokens*((444+len(tokens)-1)//len(tokens)))[:444]
        specs=[([],0,0,'empty'),(seq[:1],190,500,'no_eligible_internal'),
               (seq[:30],0,30,'short_true_document_edges'),(seq[:254],0,444,'duplicate_left'),
               (seq[190:],190,444,'duplicate_right'),(seq[:65],190,255,'full_tail')]
        summary=json.loads((args.prepared/'summary.json').read_text(encoding='utf-8-sig'))
        ids=torch.full((len(specs),256),summary['pad_id'],dtype=torch.long);masks=torch.zeros_like(ids)
        for i,(content,start,length,name) in enumerate(specs):
            values=[summary['cls_id']]+content+[summary['sep_id']]
            ids[i,:len(values)]=torch.tensor(values);masks[i,:len(values)]=1
        inp=dict(input_ids=ids,attention_mask=masks,content_start=torch.tensor([s[1] for s in specs]),document_content_length=torch.tensor([s[2] for s in specs]))
        result=model(**inp,trace=True)
        if model.arm!='sopa' and float(result['risk'][1])!=0:raise AssertionError('No eligible anchor risk')
        if result['anchor_logits_q12'] is not None and result['anchor_logits_q12'][3,222]!=result['anchor_logits_q12'][4,32]:raise AssertionError('Corner duplicate anchor')
        file=out/'fixture_corners.pt';torch.save(dict(inputs=inp,outputs=result,corner_names=[s[3] for s in specs]),file);records.append(file)
    budget.check();budget.save()
    return records,dict(complete=True,synthetic_only=False,model=model.manifest['model'],
        selected_document_ids=[docs[i]['document_id'] for i in selected],windows=len(indices),
        scope='Frozen trained-weight integer own-reference; fixed32 train documents and synthetic boundary cases')


def bits(value):
    return struct.unpack('<I',struct.pack('<f',float(value)))[0]


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bringup',type=Path)
    p.add_argument('--export',type=Path)
    p.add_argument('--prepared',type=Path,default=STUDY/'prepared')
    p.add_argument('--max-seconds',type=int,default=600)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if bool(args.bringup)==bool(args.export):raise ValueError('Choose saved bringup or frozen export')
    if not 0<args.max_seconds<=600:raise ValueError('Finite fixture ceiling')
    source=(args.bringup or args.export).resolve()
    if not source.is_relative_to(STUDY.resolve()) or not out.is_relative_to(STUDY.resolve()):raise ValueError('Outside study')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);cases=[];sources={};traces=[]
    if args.export:
        if not args.prepared.resolve().is_relative_to(STUDY.resolve()):raise ValueError('Prepared outside study')
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        files,info=trained_fixtures(args,out,torch)
    else:
        info=json.loads((source/'bringup.json').read_text(encoding='utf-8'))
        if not info['complete'] or info.get('synthetic_only',True):raise ValueError('Passed real train-only bringup required')
        files=sorted(source.glob('fixture_*.pt'))
    continuation=[]
    for file in files:
        sources[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
        saved=torch.load(file,map_location='cpu',weights_only=True)
        inp,result=saved['inputs'],saved['outputs']
        for i in range(len(inp['input_ids'])):
            n=int(inp['attention_mask'][i].sum())-2;length=int(inp['document_content_length'][i])
            risk=bits(result['risk'][i]);count=int(result['eligible'][i,1:n+1].sum())
            if info['model']['arm']=='sopa':count=n
            window=dict(ids=inp['input_ids'][i,1:n+1].tolist(),content_start=int(inp['content_start'][i]),document_content_length=length)
            base=dict(windows=[] if length==0 else [window],expected_risk_bits=risk,expected_anchors=count,
                      empty_document=length==0,previous_risk_bits=0,fixture=file.name,fixture_row=i)
            # Exact equality, just below and just above exercise strict comparison.
            for threshold in sorted({max(0,risk-1),risk,min(bits(1.),risk+1)}):
                cases.append(dict(base,threshold_bits=threshold))
            traces.append(dict(case_index=len(cases)-1,fixture=file.name,fixture_row=i,content_tokens=n,
                               state_comparison='eligible anchors only; SoPa states unavailable'))
            if file.name=='fixture_corners.pt' and i in (3,4):continuation.append(base)
    if continuation:
        left,right=continuation;combined=max(left['expected_risk_bits'],right['expected_risk_bits'])
        cases.append(dict(windows=left['windows']+right['windows'],threshold_bits=combined,expected_risk_bits=combined,
                          expected_anchors=left['expected_anchors']+right['expected_anchors'],empty_document=False,previous_risk_bits=0,corner='two_window_document'))
        cases.append(dict(right,threshold_bits=combined,previous_risk_bits=left['expected_risk_bits'],expected_risk_bits=combined,corner='continued_document_max'))
        cases.append(dict(right,threshold_bits=bits(1.),previous_risk_bits=bits(1.),expected_risk_bits=bits(1.),corner='previous_max_preserved'))
    if not cases:raise ValueError('No saved fixtures')
    (out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n',encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(dict(complete=True,source=str(source),files_sha256=sources,
        cases=len(cases),trace_mapping=traces,scope=info.get('scope','Saved untrained train-only bringup outputs; no new inference'),
        selection=info.get('selected_document_ids'),export=str(args.export.resolve() if args.export else source/'export')),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(cases=len(cases),output=str(out))))


if __name__=='__main__':main()
