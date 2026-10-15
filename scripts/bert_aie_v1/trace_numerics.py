"""Bounded diagnostic on frozen calibration+FIT inputs; no tuning or selection.

Instrumentation copies and dumps live in build/results. Canonical model, export,
hardware header and stream binary are read-only. Root alone dispatches CUDA.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from common import ROOT,read,rows,sha,write

WORST='kd_cnn_v2:new_calibration:5bc518e8a468e7b1ac30:benign'
BUILD=ROOT/'build/bert_aie_v1/numerical_trace'
OUT=ROOT/'results/bert_aie_v1/numerical_trace'
EXPORT=ROOT/'results/bert_aie_v1/export/w8a8_bf16_attention'

def directory(path):
    for p in reversed([path,*path.parents]):
        if str(p)!=p.anchor:p.mkdir(exist_ok=True)

def prepare():
    directory(BUILD);directory(OUT)
    for case in ('fit','calibration'):directory(OUT/(case+'_cpp'))
    folder=ROOT/'results/dual_track_v1/shared/data/prepared/new_calibration'
    docs=list(rows(folder/'documents.jsonl'));di=next(i for i,d in enumerate(docs) if d['document_id']==WORST)
    starts=np.load(folder/'starts.npy',mmap_mode='r');wi=int(starts[di])
    ids=np.load(folder/'ids.npy',mmap_mode='r');mask=np.load(folder/'masks.npy',mmap_mode='r')
    np.asarray([ids[wi],mask[wi],np.zeros(256)],dtype='<i4').tofile(OUT/'calibration.bin')
    (OUT/'fit.bin').write_bytes((ROOT/'results/bert_aie_v1/numerics/fixtures/fit_0.bin').read_bytes())
    source=(ROOT/'hardware/bert_aie_v1/model.hpp').read_text(encoding='utf-8')
    helper='''
#include <fstream>
#include <string>
#include <stdexcept>
namespace bert_trace {
std::string destination;
int linear_index=0,attention_index=0;
template<class T>void dump(const std::string& name,const T* data,int count) {
 std::ofstream f(destination+"/"+name+".bin",std::ios::binary);
 if(!f)throw std::runtime_error("trace output directory missing");
 f.write(reinterpret_cast<const char*>(data),count*sizeof(T));
}
}
'''
    source=source.replace('namespace bert_aie {',helper+'\nnamespace bert_aie {',1)
    replacements=[
      ('    for(int i=0;i<N*K;++i) codes[i]=quant(x[i],input_scale);','    int trace_linear=bert_trace::linear_index++;\n    for(int i=0;i<N*K;++i) codes[i]=quant(x[i],input_scale);\n    bert_trace::dump("linear_"+std::to_string(trace_linear)+"_codes",codes,N*K);'),
      ('scales[o+j]+bias[o+j];\n    }\n}', 'scales[o+j]+bias[o+j];\n    }\n    bert_trace::dump("linear_"+std::to_string(trace_linear)+"_output",out,N*O);\n}'),
      ('    for(int h=0;h<4;++h)for(int r=0;r<N;r+=AM) {','    int trace_attention=bert_trace::attention_index++;\n    for(int h=0;h<4;++h)for(int r=0;r<N;r+=AM) {'),
      ('        for(int i=0;i<AM;++i) {\n            float m=-FLT_MAX,sum=0.f;', '        std::string trace_name="attention_"+std::to_string(trace_attention)+"_h"+std::to_string(h)+"_r"+std::to_string(r);\n        bert_trace::dump(trace_name+"_scores",scores,AM*N);\n        for(int i=0;i<AM;++i) {\n            float m=-FLT_MAX,sum=0.f;'),
      ('        for(int col=0;col<HEAD;col+=AN) {','        bert_trace::dump(trace_name+"_probabilities",scores,AM*N);\n        for(int col=0;col<HEAD;col+=AN) {'),
      ('    norm(x,p+EMB_G,p+EMB_B,N);','    bert_trace::dump("embedding_sum",x,HIDDEN);\n    norm(x,p+EMB_G,p+EMB_B,N);\n    bert_trace::dump("embedding_norm",x,HIDDEN);'),
      ('        const float* b=p+LAYER_P+l*LP;', '        std::string trace_prefix="layer_"+std::to_string(l)+"_";\n        const float* b=p+LAYER_P+l*LP;'),
      ('        attention(e,q,k,v,mask,a);','        attention(e,q,k,v,mask,a);\n        bert_trace::dump(trace_prefix+"context",a,HIDDEN);'),
      ('        norm(y,b+4*D,b+5*D,N);','        bert_trace::dump(trace_prefix+"attention_residual",y,HIDDEN);\n        norm(y,b+4*D,b+5*D,N);\n        bert_trace::dump(trace_prefix+"attention_norm",y,HIDDEN);'),
      ('        linear(e,ff,lw+4*D*D+D*F', '        bert_trace::dump(trace_prefix+"gelu",ff,FF);\n        linear(e,ff,lw+4*D*D+D*F'),
      ('        norm(x,b+7*D+F,b+8*D+F,N);','        bert_trace::dump(trace_prefix+"ff_residual",x,HIDDEN);\n        norm(x,b+7*D+F,b+8*D+F,N);\n        bert_trace::dump(trace_prefix+"block_output",x,HIDDEN);'),
      ('    for(int o=0;o<2;++o) {','    bert_trace::dump("pooled",pooled,D);\n    for(int o=0;o<2;++o) {'),
      ('    return 0;\n}', '    bert_trace::dump("logits",logits,2);\n    return 0;\n}')]
    for before,after in replacements:
        if source.count(before)!=1:raise ValueError('Instrumentation replacement is not unique: '+before)
        source=source.replace(before,after,1)
    (BUILD/'model_trace.hpp').write_text(source,encoding='utf-8')
    functional=(ROOT/'hardware/bert_aie_v1/functional.cpp').read_text(encoding='utf-8')
    functional=functional.replace('#include "model.hpp"','#include "model_trace.hpp"')
    functional=functional.replace('int main(int argc,char** argv) {','int main(int argc,char** argv) {\n    if(argc!=5)return 3;\n    bert_trace::destination=argv[4];--argc;')
    (BUILD/'functional_trace.cpp').write_text(functional,encoding='utf-8')
    write(OUT/'inputs.json',{'calibration_document_id':WORST,'calibration_window_index':wi,'fit_fixture':'fit_0','inputs':{n:sha(OUT/(n+'.bin')) for n in ('calibration','fit')},
        'source_sha256':{n:sha(ROOT/n) for n in ('hardware/bert_aie_v1/model.hpp','hardware/bert_aie_v1/functional.cpp','scripts/bert_aie_v1/model.py')},'purpose':'locate execution divergence; no threshold/model choice'})
    print(json.dumps({'prepared':True,'build':str(BUILD),'out':str(OUT)}))

def python_trace(device,cases):
    import torch
    import model as module
    from export import load_export
    from torch.nn import functional as F
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,_=load_export(EXPORT,device=device);started=time.monotonic()
    for case in cases:
        target=OUT/(case+'_python_'+device);directory(target)
        raw=np.fromfile(OUT/(case+'.bin'),dtype='<i4').reshape(3,256)
        ids,mask,types=[torch.tensor(a[None,:],device=device,dtype=torch.long) for a in raw]
        dump_data={};linear_index=0;attention_index=0
        def dump(name,value):dump_data[name]=value.detach().cpu().numpy().copy()
        original_linear=model.linear;original_mm=module.bf16_matmul
        def linear(x,name):
            nonlocal linear_index
            result=original_linear(x,name)
            if name.endswith('.attention.output.dense'):dump('layer_'+name.split('.')[3]+'_context',x)
            if name.endswith('.output.dense') and '.attention.' not in name:dump('layer_'+name.split('.')[3]+'_gelu',x)
            if name=='classifier':dump('pooled',x)
            if name in module.LINEARS:
                dump('linear_'+str(linear_index)+'_codes',module.quantize(x,model.scales[name]));dump('linear_'+str(linear_index)+'_output',result);linear_index+=1
            return result
        def mm(a,b,chunk=64):
            nonlocal attention_index
            result=original_mm(a,b,chunk)
            if attention_index%2==0:
                layer=attention_index//2;scores=result/8+(1-mask[:,None,None,:].float())*torch.finfo(torch.float32).min;prob=scores.softmax(-1)
                for h in range(4):
                    for r in range(0,256,32):
                        prefix=f'attention_{layer}_h{h}_r{r}'
                        dump(prefix+'_scores',scores[0,h,r:r+32]);dump(prefix+'_probabilities',prob[0,h,r:r+32])
            attention_index+=1
            return result
        model.linear=linear;module.bf16_matmul=mm
        original_norm=model.norm
        def norm(x,name):
            result=original_norm(x,name)
            if name=='bert.embeddings.LayerNorm':dump('embedding_sum',x);dump('embedding_norm',result)
            else:
                layer=int(name.split('.')[3]);attention='.attention.' in name
                dump(f'layer_{layer}_'+('attention_residual' if attention else 'ff_residual'),x)
                dump(f'layer_{layer}_'+('attention_norm' if attention else 'block_output'),result)
            return result
        model.norm=norm
        # Hooks only copy operands/results; verify terminal logits against
        # unchanged original forward before interpreting the trace.
        with torch.inference_mode():
            instrumented=model(input_ids=ids,attention_mask=mask,token_type_ids=types).logits
        model.linear=original_linear;model.norm=original_norm;module.bf16_matmul=original_mm
        with torch.inference_mode():baseline=model(input_ids=ids,attention_mask=mask,token_type_ids=types).logits
        if not torch.equal(instrumented,baseline):raise ValueError('Instrumentation changed Python output')
        dump('logits',instrumented)
        for name,array in dump_data.items():array.astype('i1' if array.dtype==np.int8 else '<f4').tofile(target/(name+'.bin'))
        write(target/'manifest.json',{'case':case,'device':device,'baseline_logits':baseline.cpu().tolist(),'instrumentation_bit_identical':True,'files':{n:sha(target/(n+'.bin')) for n in dump_data},'seconds_total':time.monotonic()-started})
    print(json.dumps({'complete':True,'device':device,'cases':cases,'seconds':time.monotonic()-started}))

def compare(case,device):
    cpp=OUT/(case+'_cpp');py=OUT/(case+'_python_'+device);records=[]
    order=['embedding_sum','embedding_norm']
    for l in range(4):
        for j in range(3):order += [f'linear_{l*6+j}_codes',f'linear_{l*6+j}_output']
        order += [f'attention_{l}_h{h}_r{r}_{kind}' for kind in ('scores','probabilities') for h in range(4) for r in range(0,256,32)]
        order += [f'layer_{l}_context']
        order += [f'linear_{l*6+3}_codes',f'linear_{l*6+3}_output',f'layer_{l}_attention_residual',f'layer_{l}_attention_norm']
        order += [f'linear_{l*6+4}_codes',f'linear_{l*6+4}_output',f'layer_{l}_gelu',f'linear_{l*6+5}_codes',f'linear_{l*6+5}_output']
        order += [f'layer_{l}_ff_residual',f'layer_{l}_block_output']
    order += ['pooled','logits']
    for name in order:
        if not (cpp/(name+'.bin')).exists() or not (py/(name+'.bin')).exists():continue
        dtype='i1' if name.endswith('_codes') else '<f4';a=np.fromfile(cpp/(name+'.bin'),dtype=dtype);b=np.fromfile(py/(name+'.bin'),dtype=dtype)
        if a.shape!=b.shape:raise ValueError('Trace shape mismatch '+name)
        delta=a.astype(np.float64)-b.astype(np.float64);row={'stage':name,'elements':len(a),'changed_elements':int(np.count_nonzero(delta)),'max_abs_error':float(np.max(np.abs(delta))),'mean_abs_error':float(np.mean(np.abs(delta)))}
        if dtype=='i1':row['fraction_changed']=row['changed_elements']/len(a)
        records.append(row)
    result={'case':case,'python_device':device,'stages':records,'first_numerical_difference':next((r for r in records if r['changed_elements']),None),
            'first_quantized_code_difference':next((r for r in records if r['stage'].endswith('_codes') and r['changed_elements']),None)}
    write(OUT/(case+'_comparison_'+device+'.json'),result)
    print(json.dumps({'case':case,'device':device,'first_numerical_difference':result['first_numerical_difference'],
                      'first_quantized_code_difference':result['first_quantized_code_difference'],'logits':records[-1]},indent=2))

def run():
    """Root-dispatched, lock-guarded one-shot after profiling releases GPU/CPU."""
    from evaluate_cpp import unix_path
    started=time.monotonic();identities={}
    for case in ('fit','calibration'):
        commands=[]
        for instrumented in (False,True):
            binary=BUILD/'functional_trace' if instrumented else ROOT/'build/bert_aie_v1/common/bert_functional_stream'
            command=[unix_path(binary),unix_path(EXPORT),unix_path(OUT/(case+'.bin')),'resident']
            if instrumented:command += [unix_path(OUT/(case+'_cpp'))]
            if os.name=='nt':command=['wsl.exe','-d','Ubuntu','--exec']+command
            result=subprocess.run(command,capture_output=True,text=True,check=True,timeout=15)
            commands.append(json.loads(result.stdout))
        same=np.array_equal(np.asarray(commands[0]['logits'],dtype=np.float32),np.asarray(commands[1]['logits'],dtype=np.float32))
        if not same:raise ValueError('C++ instrumentation altered output: '+case)
        identities[case]={'instrumentation_bit_identical':same,'unmodified':commands[0],'instrumented':commands[1]}
    write(OUT/'cpp_instrumentation_check.json',identities)
    python_trace('cuda',['fit','calibration'])
    for case in ('fit','calibration'):compare(case,'cuda')
    write(OUT/'run_complete.json',{'complete':True,'seconds':time.monotonic()-started,'data_roles':['fit','one frozen calibration window'],
        'script_sha256':sha(Path(__file__)),'canonical_sources_unchanged':read(OUT/'inputs.json')['source_sha256']=={n:sha(ROOT/n) for n in read(OUT/'inputs.json')['source_sha256']},
        'scope':'execution diagnosis only; no model/threshold selection or retuning'})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','python','compare','run'));p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--cases',nargs='+',choices=('fit','calibration'),default=['fit','calibration'])
    a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='run':run()
    elif a.action=='python':python_trace(a.device,a.cases)
    else:
        for case in a.cases:compare(case,a.device)
