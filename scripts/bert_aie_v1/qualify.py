"""Bounded calibration and numerical differential checks; never held-out fitting."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model import ROOT, LINEARS, load_model, inventory, sha

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def fit_calibration(folder,output,windows=64,device='cpu'):
    folder=Path(folder)
    if folder.name!='fit': raise ValueError('Only the explicitly approved FIT prepared folder is permitted')
    if windows<1 or windows>256: raise ValueError('Calibration budget must be 1..256 windows')
    ids=np.load(folder/'ids.npy',mmap_mode='r');masks=np.load(folder/'masks.npy',mmap_mode='r')
    # Deterministic, label-free coverage across the FIT window stream.
    indices=np.unique(np.linspace(0,len(ids)-1,min(windows,len(ids)),dtype=np.int64))
    model,meta=load_model(observe=True,device=device)
    started=time.monotonic()
    with torch.inference_mode():
        for at in indices:
            model(input_ids=torch.tensor(np.array(ids[at:at+1]),dtype=torch.long,device=device),
                  attention_mask=torch.tensor(np.array(masks[at:at+1]),dtype=torch.long,device=device))
    scales={name:max(model.observed[name],torch.finfo(torch.float32).tiny)/127 for name in LINEARS}
    result={'schema':'sentry_bert_aie_v1_calibration_v1','data_role':'fit','policy':'fixed evenly spaced FIT windows, per-linear absmax/127, no labels',
            'data_identity':{'folder':str(folder),'ids_sha256':sha(folder/'ids.npy'),'masks_sha256':sha(folder/'masks.npy')},
            'window_indices':indices.tolist(),'input_scales':scales,'observed_absmax':model.observed,'model':meta,
            'seconds':time.monotonic()-started,'device':device,'threads':torch.get_num_threads()}
    write(output,result);return result

def qualification(calibration,output,device='cpu',export_folder=None):
    from transformers import BertConfig, BertForSequenceClassification
    from model import CHECKPOINT, CONFIG
    c=json.loads(Path(calibration).read_text(encoding='utf-8'))
    if c['data_role']!='fit': raise ValueError('FIT scale identity required')
    reference,meta=load_model(device=device)
    source=BertForSequenceClassification(BertConfig.from_json_file(str(CONFIG))).eval()
    source.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=False)['state_dict']);source.to(device)
    # Real FIT inputs plus varying tails, nonzero token type and masked-key mutation.
    folder=Path(c['data_identity']['folder']);ids=np.load(folder/'ids.npy',mmap_mode='r');masks=np.load(folder/'masks.npy',mmap_mode='r')
    examples=[]
    first_full=next((i for i in range(len(masks)) if int(masks[i].sum())==256),c['window_indices'][0])
    for at in dict.fromkeys((c['window_indices'][0],c['window_indices'][-1],first_full)):
        examples.append((f'fit_{at}',torch.tensor(np.array(ids[at:at+1]),dtype=torch.long,device=device),torch.tensor(np.array(masks[at:at+1]),dtype=torch.long,device=device),None))
    gen=torch.Generator().manual_seed(42)
    for n in (7,65,255,256):
        x=torch.randint(100,30000,(1,256),generator=gen).to(device);x[:,0]=101;x[:,n-1]=102
        mask=(torch.arange(256,device=device)[None,:]<n).long();types=torch.zeros_like(x);types[:,max(1,n//2):]=1
        examples.append((f'tail_{n}_token_type1',x,mask,types))
    candidates={v:load_model(v,c['input_scales'],device=device,observe=True)[0] for v in ('w8a8_fp32_attention','w8a8_bf16_attention')}
    if export_folder:
        from export import load_export
        candidates['packed_reload']=load_export(export_folder,device=device)[0]
    records=[];fixtures={}
    with torch.inference_mode():
        for name,x,mask,types in examples:
            kwargs=dict(input_ids=x,attention_mask=mask,token_type_ids=types)
            original=source(**kwargs,output_hidden_states=True)
            explicit=reference(**kwargs,return_hidden=True)
            tensor_errors=[float((a-b).abs().max()) for a,b in zip(original.hidden_states,explicit.hidden_states)]
            rec={'case':name,'source_reference_max_logit_error':float((original.logits-explicit.logits).abs().max()),
                 'source_reference_hidden_max_errors':tensor_errors,'reference_logits':explicit.logits.cpu().tolist(),'variants':{}}
            for variant,model in candidates.items():
                value=model(**kwargs).logits
                rec['variants'][variant]={'logits':value.cpu().tolist(),'max_logit_error':float((value-explicit.logits).abs().max()),
                    'positive_probability_error':float((value.softmax(-1)[:,1]-explicit.logits.softmax(-1)[:,1]).abs().max())}
                # Masked keys must have no influence on CLS, even if IDs change.
                mutated=x.clone();mutated[mask==0]=(mutated[mask==0]+59)%30522
                changed=model(input_ids=mutated,attention_mask=mask,token_type_ids=types).logits
                rec['variants'][variant]['masked_key_mutation_max_logit_error']=float((value-changed).abs().max())
            fixtures[name]={'input_ids':x.cpu(),'attention_mask':mask.cpu(),'token_type_ids':types.cpu() if types is not None else torch.zeros_like(x.cpu()),
                            'reference_logits':explicit.logits.cpu(),'candidate_logits':{v:m(**kwargs).logits.cpu() for v,m in candidates.items()}}
            records.append(rec)
    result={'model':meta,'data_role':'fit_and_synthetic','calibration_sha256':sha(calibration),'cases':records,
            'inventory':inventory(reference),'device':device,'threads':torch.get_num_threads(),
            'source_differential_qualified':max(r['source_reference_max_logit_error'] for r in records)<1e-4,
            'source_differential_tolerance':'1e-4 absolute logits engineering regression tolerance; not a scientific quality acceptance gate',
            'saturation':{v:m.saturation for v,m in candidates.items()},'evidence_boundary':'software numerical differential; no device throughput or corpus quality claim'}
    write(output,result)
    torch.save(fixtures,Path(output).with_suffix('.fixtures.pt'))
    # Small directly consumable whole-model C++ fixtures (no Torch dependency).
    fixture_dir=Path(output).parent/'fixtures';fixture_dir.mkdir(exist_ok=True)
    plain={}
    for name,entry in fixtures.items():
        if not name.startswith('fit_'):continue
        path=fixture_dir/(name+'.bin')
        with path.open('wb') as handle:
            for key in ('input_ids','attention_mask','token_type_ids'):
                handle.write(entry[key].numpy().astype('<i4').tobytes())
        plain[name]={'input':str(path),'input_sha256':sha(path),'valid_tokens':int(entry['attention_mask'].sum()),
                     'reference_logits':entry['reference_logits'].tolist(),
                     'candidate_logits':{v:t.tolist() for v,t in entry['candidate_logits'].items()}}
    write(fixture_dir/'expected_logits.json',plain)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    c=sub.add_parser('calibrate');c.add_argument('--fit',type=Path,default=ROOT/'results/cnn_adoption_v1/prepared/token_r2/fit');c.add_argument('--windows',type=int,default=64)
    q=sub.add_parser('differential');q.add_argument('--calibration',required=True,type=Path);q.add_argument('--export',type=Path)
    for command in (c,q):command.add_argument('--output',required=True,type=Path);command.add_argument('--device',default='cpu')
    args=p.parse_args();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic()
    result=fit_calibration(args.fit,args.output,args.windows,args.device) if args.action=='calibrate' else qualification(args.calibration,args.output,args.device,args.export)
    print(json.dumps({'output':str(args.output),'seconds':time.monotonic()-started,'source_differential_qualified':result.get('source_differential_qualified')},indent=2))
    if args.action=='differential' and not result['source_differential_qualified']:raise SystemExit(1)

if __name__=='__main__':main()
