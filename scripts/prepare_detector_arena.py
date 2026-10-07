"""Pack only the saved single-input W8-QAT arena; no model forward or new data."""
from pathlib import Path
import argparse, json, struct
ROOT=Path(__file__).resolve().parents[1]
BRIDGE=ROOT/'results/goal4/numerical_bridge'
CHECKPOINT=ROOT/'checkpoints/quantized/qat_w8_a8'
PLAN=ROOT/'results/goal4/internal_integration/command_plan.json'
OUT=ROOT/'results/goal4/internal_integration/arena_w8_qat_real_input_v1'
CASE='aeslc:train:panus-s_inbox_1.subject:clean'
WORKSPACE=('embedding_word','embedding_type','hidden','query','key','value','context',
 'projected','residual','normalized','ffn_pre_gelu','ffn_activated','a8','accumulator',
 'key_head','value_head_transposed','query_slab','attention_tile','scores','probabilities',
 'mask_slab','pooler_output','pooled','logits','class_probabilities')

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def require(test,message):
    if not test:raise ValueError(message)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    if not args.execute:
        print(json.dumps({'execute':False,'checkpoint':str(CHECKPOINT),'case':CASE,'precision':'uniform W8 only','output':str(OUT),'inference':False}));return
    require(not (ROOT/'results/goal2/pause.request').exists(),'Pause marker present')
    require(not OUT.exists(),'Preserve existing arena; no overwrite')
    manifest,plan,quant=read(BRIDGE/'manifest.json'),read(PLAN),read(CHECKPOINT/'quantization.json')
    require(manifest['case']==CASE and manifest['checkpoint'].replace('\\','/')==str(CHECKPOINT.relative_to(ROOT)).replace('\\','/'),'Saved case/checkpoint differs')
    require(manifest['hardware_source_revision']=='panel_cache_batched_attention_v1','Unsupported service revision')
    require(manifest['shape']==dict(length=256,layers=4,hidden=256,heads=4,ffn=1024) and manifest['window_count']==1,'Only existing L256 input supported')
    require(plan['identity']=='bert_mini_l256_internal_plan_v1' and plan['arena']['planned_bytes']==67108864 and plan['arena']['physical_base'] is None,'Plan/allocation differs')
    groups=plan['precision']['groups']
    require(len(groups)==16 and set(quant['precision_map'])==set(groups) and all(quant['precision_map'][g]==8 for g in groups),'Only saved uniform W8 precision supported')
    require(quant['scheme_id']==manifest['format']=='static_symmetric_encoder_linear_w4w8a8_fp32_other_v1','Numerical scheme differs')
    import numpy as np
    import torch
    state=torch.load(CHECKPOINT/'quantized_state.pt',map_location='cpu',weights_only=True)
    buffers=plan['buffers'];arena=bytearray(plan['arena']['planned_bytes']);written={};checks=[]
    intervals=[]
    for name,b in buffers.items():
        begin,end=b['offset'],b['offset']+b['bytes']
        require(begin>=0 and begin%4096==0 and end<=len(arena) and b['bytes']>0,'Invalid buffer extent '+name)
        intervals.append((begin,end,name))
    ordered=sorted(intervals)
    require(all(a[1]<=b[0] for a,b in zip(ordered,ordered[1:])),'Overlapping arena buffers')
    def tensor(key,shape):
        require(key in state,'Missing checkpoint parameter '+key)
        x=state[key];require(tuple(x.shape)==tuple(shape) and x.dtype==torch.float32 and bool(torch.isfinite(x).all()),'Invalid tensor '+key)
        return x.detach().contiguous()
    def fbytes(x):return x.numpy().astype('<f4',copy=False).tobytes()
    def put(name,data,source):
        b=buffers[name];require(name not in written and len(data)==b['bytes'],'Payload extent/duplicate '+name)
        arena[b['offset']:b['offset']+b['bytes']]=data
        written[name]={'offset':b['offset'],'bytes':len(data),'source':source}
    exports={v['file']:v for v in manifest['files']}
    def same_export(name,data):
        filename=name+'.bin';require(filename in exports,'Export not in saved manifest '+name)
        expected=(BRIDGE/filename).read_bytes()
        require(len(expected)==exports[filename]['elements']*(1 if exports[filename]['dtype']=='int8' else 4),'Saved export extent '+name)
        require(data==expected,'Exact saved export mismatch '+name);checks.append(filename)
    ids={}
    for field,name,limit,dtype in [('input_ids','token_ids',30522,'<u4'),('token_type_ids','type_ids',2,'<u4'),('attention_mask','attention_mask',2,'u1')]:
        values=manifest[field];require(len(values)==256 and all(type(v) is int and 0<=v<limit for v in values),'Input range '+field)
        ids[field]=np.array(values,dtype=dtype);put(name,ids[field].tobytes(),'saved bridge manifest '+field)
    require(int(ids['attention_mask'].sum())==manifest['total_unpadded_tokens'],'Saved valid length differs')
    tables=[('word_table','bert.embeddings.word_embeddings.weight',(30522,256),'embedding_word',ids['input_ids']),
            ('type_table','bert.embeddings.token_type_embeddings.weight',(2,256),'embedding_type',ids['token_type_ids']),
            ('position_table','bert.embeddings.position_embeddings.weight',(512,256),'embedding_position',np.arange(256))]
    for name,key,shape,export,index in tables:
        value=tensor(key,shape);put(name,fbytes(value),key)
        same_export(export,fbytes(value[torch.from_numpy(index.astype(np.int64))]))
    mask_bits=np.where(ids['attention_mask']==1,np.uint32(0x80000000),np.uint32(0xff7fffff)).astype('<u4')
    same_export('key_mask',mask_bits.tobytes())
    norms=[('embedding_ln','bert.embeddings.LayerNorm')]
    for layer in range(4):norms += [(f'layer{layer}_attention_ln',f'bert.encoder.layer.{layer}.attention.output.LayerNorm'),(f'layer{layer}_output_ln',f'bert.encoder.layer.{layer}.output.LayerNorm')]
    for name,prefix in norms:
        gamma,beta=tensor(prefix+'.weight',(256,)),tensor(prefix+'.bias',(256,))
        put(name+'_gamma',fbytes(gamma),prefix+'.weight');put(name+'_beta',fbytes(beta),prefix+'.bias')
        same_export(name,fbytes(torch.cat((gamma,beta))))
    suffixes={'query':'attention.self.query','key':'attention.self.key','value':'attention.self.value','attention_output':'attention.output.dense','ffn_input':'intermediate.dense','ffn_output':'output.dense'}
    matrices=plan['matrices'];require(len(matrices)==24 and {m['input_scale_index'] for m in matrices}==set(range(24)),'Matrix/scales order differs')
    scales=[None]*24
    for m in matrices:
        name=m['name'];layer,suffix=name.split('_',1);prefix=f'bert.encoder.layer.{int(layer[5:])}.{suffixes[suffix]}'
        n,k=m['outputs'],m['inner'];weight=tensor(prefix+'.weight',(n,k));bias=tensor(prefix+'.bias',(n,))
        input_scale=tensor(prefix+'.input_scale',());require(float(input_scale)>0,'Invalid input scale')
        require(float(tensor(prefix+'.weight_width',()))==8 and float(tensor(prefix+'.activation_width',()))==8 and float(tensor(prefix+'.zero_point',()))==0,'Unexpected saved widths/zero point')
        row_scales={bits:tensor(prefix+f'.weight_scale_{bits}',(n,1)) for bits in (4,8)}
        for bits,scale in row_scales.items():
            require(bool((scale>0).all()),'Nonpositive frozen row scale')
            put(name+f'_weight_scale{bits}',fbytes(scale),prefix+f'.weight_scale_{bits}')
        codes=torch.clamp(torch.round(weight/row_scales[8]),-128,127).to(torch.int8).contiguous().numpy().tobytes()
        require(len(codes)==n*k,'Code payload size differs');same_export(name+'_w8',codes)
        put(name+'_weights',codes,'Existing W8 export checked against QAT weights / saved frozen W8 scales')
        put(name+'_bias',fbytes(bias),prefix+'.bias')
        params=torch.cat((input_scale.reshape(1),row_scales[8].reshape(n),bias))
        same_export(name+'_params',fbytes(params));scales[m['input_scale_index']]=fbytes(input_scale)
    put('input_scales',b''.join(scales),'24 saved frozen checkpoint input_scale buffers in plan matrix order')
    for name,key,shape in [('pooler_weight','bert.pooler.dense.weight',(256,256)),('pooler_bias','bert.pooler.dense.bias',(256,)),('classifier_weight','classifier.weight',(2,256)),('classifier_bias','classifier.bias',(2,))]:
        payload=fbytes(tensor(key,shape));same_export(name,payload);put(name,payload,key)
    zero_buffers=set(WORKSPACE)|{'dummy_zero'}
    require(set(buffers)==set(written)|zero_buffers and not(set(written)&zero_buffers),'Unmapped parameter buffer; no missing data may be zero-filled')
    threshold=(BRIDGE/'threshold.bin').read_bytes();require(len(threshold)==4,'Missing saved threshold')
    threshold_bits=struct.unpack('<I',threshold)[0];require(threshold_bits<=0x3f800000,'Invalid saved threshold')
    OUT.mkdir(parents=True);image=OUT/'arena.bin';image.write_bytes(arena)
    with image.open('rb') as f:
        for begin,end,name in ordered:
            f.seek(begin);actual=f.read(end-begin)
            require(actual==arena[begin:end],'Readback mismatch '+name)
            if name in zero_buffers:require(not any(actual),'Nonzero workspace '+name)
    require(image.stat().st_size==67108864,'Arena file extent differs')
    result={'complete':True,'scope':'Packed existing single-input uniformW8QAT arena only; no inference/integration execution/numerical acceptance',
        'plan_identity':plan['identity'],'plan_recorded_utc':plan['recorded_utc'],'case':CASE,'checkpoint':str(CHECKPOINT.relative_to(ROOT)),
        'saved_bridge_recorded_utc':manifest['recorded_utc'],'service_revision':manifest['hardware_source_revision'],
        'image':'arena.bin','image_bytes':len(arena),'physical_base':None,'allocation_confirmed':False,
        'precision_map':quant['precision_map'],'config_w8_mask':65535,'threshold_bits_hex':f'{threshold_bits:08x}',
        'valid_length':int(ids['attention_mask'].sum()),'runtime_mailbox_numeric_ids':None,
        'written_buffers':written,'zero_buffers':sorted(zero_buffers),'unused_alignment_and_tail':'zero-filled padding, not model parameters',
        'exact_saved_export_checks':checks,'readback_buffers_checked':len(buffers),
        'weight_slot_bytes':sum(written[m['name']+'_weights']['bytes'] for m in matrices),
        'frozen_scale_policy':'Both W4/W8 row tables copied exactly from saved checkpoint buffers; no scale recalibration or regeneration',
        'unsupported':'Mixed/W4 maps, other inputs/checkpoints and missing parameters rejected; no fallback parameters',
        'source_script':str(Path(__file__).relative_to(ROOT))}
    (OUT/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'complete':True,'output':str(OUT),'image_bytes':len(arena),'exact_saved_export_checks':len(checks),'buffers_checked':len(buffers)},indent=2))
if __name__=='__main__':main()
