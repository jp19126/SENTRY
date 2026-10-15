"""FIT0 real first-layer AIE service fixtures; CPU only and no fitting."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import ROOT, quantize, bf16_matmul, sha
from export import load_export

def main():
    torch.set_num_threads(1)
    model,manifest=load_export(ROOT/'results/bert_aie_v1/export/w8a8_bf16_attention')
    out=ROOT/'results/bert_aie_v1/numerics/real_tiles';out.mkdir(parents=True,exist_ok=True)
    fixtures=json.loads((ROOT/'results/bert_aie_v1/numerics/fixtures/expected_logits.json').read_text())
    raw=np.fromfile(ROOT/'results/bert_aie_v1/numerics/fixtures/fit_0.bin',dtype='<i4').reshape(3,256)
    ids,mask,types=[torch.tensor(x[None,:],dtype=torch.long) for x in raw]
    with torch.inference_mode():
        x=F.embedding(ids,model.w('bert.embeddings.word_embeddings.weight'))
        x=x+F.embedding(types,model.w('bert.embeddings.token_type_embeddings.weight'))
        x=x+F.embedding(torch.arange(256)[None,:],model.w('bert.embeddings.position_embeddings.weight'))
        x=model.norm(x,'bert.embeddings.LayerNorm')
        p='bert.encoder.layer.0'
        q,k,v=[model.linear(x,p+'.attention.self.'+n).view(1,256,4,64).transpose(1,2) for n in ('query','key','value')]
        scores=bf16_matmul(q,k.transpose(-1,-2))/8+(1-mask[:,None,None,:].float())*torch.finfo(torch.float32).min
        probabilities=scores.softmax(-1)
        pairs={'qk':(q[0,0,:4,:64],k[0,0,:8,:64].t()),
               'pv':(probabilities[0,0,:4,:64],v[0,0,:64,:8]),
               'qk32':(q[0,0,:32,:64],k[0,0,:32,:64].t()),
               'pv32':(probabilities[0,0,:32,:64],v[0,0,:64,:32])}
        metadata={'data_role':'fit','source_fixture_sha256':fixtures['fit_0']['input_sha256'],'tiles':{},'model_manifest_sha256':sha(ROOT/'results/bert_aie_v1/export/w8a8_bf16_attention/manifest.json')}
        for name,(a,b) in pairs.items():
            a=a.bfloat16().float().contiguous();b=b.bfloat16().float().contiguous()
            golden=a.double()@b.double();sumabs=a.double().abs()@b.double().abs()
            eps=2.**-24;bound=(64*eps/(1-64*eps))*sumabs
            record={'A':a.tolist(),'B':b.tolist(),'expected_float64':golden.tolist(),'fp32_absolute_error_bound':bound.tolist(),
                    'sum_abs_products':sumabs.tolist(),'shape_A':list(a.shape),'shape_B':list(b.shape),
                    'bound':'gamma64*sumabs FP32 sequential dot, finite normal values; no integer ULP arbitrary gate'}
            path=out/(name+'.json');path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
            np.save(out/(name+'_A.npy'),a.numpy());np.save(out/(name+'_B.npy'),b.numpy())
            metadata['tiles'][name]={'json_sha256':sha(path),'max_absolute_bound':float(bound.max()),'non_integer_operands':bool(((a-a.round()).abs()>0).any() or ((b-b.round()).abs()>0).any())}
        name=p+'.attention.self.query'
        a=quantize(x[0,:32],model.scales[name]);w=getattr(model,name.replace('.','__')+'__codes')[:32].t().contiguous()
        gold=a.to(torch.int32)@w.to(torch.int32)
        np.save(out/'int8_A.npy',a.numpy());np.save(out/'int8_B.npy',w.numpy());np.save(out/'int8_expected.npy',gold.numpy())
        (out/'int8.json').write_text(json.dumps({'A':a.tolist(),'B':w.tolist(),'expected':gold.tolist(),'shape_A':[32,256],'shape_B':[256,32]},indent=2)+'\n',encoding='utf-8')
        metadata['int8']={'source':'FIT0 embedding LN first32 tokens/query first32 outputs','json_sha256':sha(out/'int8.json')}
        (out/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
