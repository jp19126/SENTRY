"""Concrete UART guard preparation, native verification and GNU A72 link only.
No serial/device APIs, Vitis/Vivado dispatch or board operations.
"""
import argparse,hashlib,json,os,shutil,struct,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
NAMES=['cnn.hpp','host_driver.hpp','host_driver.cpp','host_a72.hpp','host_a72.cpp','host_sha256.hpp','host_sha256.cpp','host_service.hpp','host_service.cpp','host_main.cpp','host_test.cpp','host_service_test.cpp']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def packet(raw,ids,masks,model_sha,threshold,seq=1):
 if len(raw)>1048576 or len(ids)%512 or len(ids)//512!=len(masks)//256 or len(masks)%256 or not 1<=len(ids)//512<=4096 or not 1<=seq<2**32:raise ValueError('Bounded complete document required')
 h=struct.pack('<4sIII',b'KDC2',seq,len(raw),len(ids)//512)+bytes.fromhex(model_sha)+struct.pack('<i',threshold);payload=raw+ids+masks
 return h+hashlib.sha256(h+payload).digest()+payload

def prepare(out,export,cases,registers,bsp,linker,threshold):
 out=Path(out);out.mkdir(exist_ok=False,parents=True);source=out/'source';source.mkdir()
 for n in NAMES:shutil.copyfile(ROOT/'hardware/kd_cnn_v2'/n,source/n)
 shutil.copyfile(ROOT/'hardware/kd_cnn_v2/cnn.cpp',source/'cnn.cpp');shutil.copyfile(registers,source/'registers.hpp');shutil.copyfile(linker,out/'lscript.ld')
 model=Path(export)/'model.bin';manifest=json.loads((Path(export)/'manifest.json').read_text())
 if model.stat().st_size!=2036944:raise ValueError('Wrong model extent')
 # The export manifest schema stores model SHA at top level.
 model_sha=sha(model)
 if manifest.get('model_sha256')!=model_sha:raise ValueError('Packed export identity differs')
 if os.name=='nt':raise RuntimeError('Prepare GNU host build from WSL to bind usable BSP paths')
 if not -2147483648<=threshold<=2147483647:raise ValueError('Integer threshold required')
 (source/'deployment.hpp').write_text('#include "registers.hpp"\nconstexpr int32_t deployment_threshold='+str(threshold)+';\nconstexpr uint8_t deployment_model_sha[32]={'+','.join(str(v) for v in bytes.fromhex(model_sha))+'};\n')
 c=Path(cases).read_bytes();count,n,first,final,prev,prior,t,margin,blocked=struct.unpack_from('<IIIIiIiii',c)
 if n!=1 or not first or not final:raise ValueError('First fixture must be one complete document')
 ids=c[36:36+512];masks=c[548:804];frame=packet(b'retained',ids*17,masks*17,model_sha,threshold)
 (out/'fixture.bin').write_bytes(model.read_bytes()+frame)
 save(out/'prepared.json',{'source_sha256':{p.name:sha(p) for p in source.iterdir()},'model_sha256':model_sha,'export_manifest_sha256':sha(Path(export)/'manifest.json'),'cases_sha256':sha(cases),'fixture_sha256':sha(out/'fixture.bin'),'fixture_margin':margin,'threshold':threshold,'bsp':str(Path(bsp).resolve()),'bsp_xparameters_sha256':sha(Path(bsp)/'include/xparameters.h'),'linker_sha256':sha(out/'lscript.ld'),'boundary':'Prepared sources/frame; threshold is supplied qualification input until immutable nomination; existing BSP compatibility only'})

def execute(out,action):
 if os.name!='posix':raise RuntimeError('Use WSL GNU toolchain')
 out=Path(out).resolve();p=json.loads((out/'prepared.json').read_text());source=out/'source';record=out/(action+'.json')
 if record.exists():raise FileExistsError('Preserve previous result')
 for name,digest in p['source_sha256'].items():
  if sha(source/name)!=digest:raise ValueError('Frozen source changed')
 rows=[];started=time.monotonic()
 def run(args):
  start=time.monotonic();r=subprocess.run(args,capture_output=True,text=True,timeout=180);rows.append({'command':args,'seconds':time.monotonic()-start,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr});save(record,{'state':'running','commands':rows});r.check_returncode();return r
 try:
  if action=='test':
   args=['g++','-O2','-std=c++17','-DCNN_LANES=16','-DCNN_REGISTER_MAP_HEADER="registers.hpp"','-I'+str(source),*[str(source/n) for n in ['cnn.cpp','host_driver.cpp','host_sha256.cpp','host_service.cpp','host_service_test.cpp']],'-o',str(out/'service_test')];run(args);run([str(out/'service_test'),str(out/'fixture.bin'),str(p['fixture_margin'])]);artifact=out/'service_test'
  else:
   bsp=Path(p['bsp']);compiler='/home/jp19126/Xilinx/2025.2/gnu/aarch64/lin/aarch64-none/bin/aarch64-none-elf-g++'
   flags=['-std=c++17','-O2','-fno-exceptions','-fno-rtti','-DSDT','-DARMA72_EL3','-I'+str(source),'-I'+str(bsp/'include'),'-specs='+str(bsp/'Xilinx.spec')]
   objects=[]
   for name in ['host_driver.cpp','host_a72.cpp','host_sha256.cpp','host_service.cpp','host_main.cpp']:
    obj=out/(name+'.o');run([compiler,*flags,'-c',str(source/name),'-o',str(obj)]);objects.append(str(obj))
   artifact=out/'kd_cnn_guard.elf';run([compiler.replace('g++','gcc'),*flags,*objects,'-Wl,-T,'+str(out/'lscript.ld'),'-Wl,-Map,'+str(out/'kd_cnn_guard.map'),'-L'+str(bsp/'lib'),'-Wl,--start-group','-lxilstandalone','-lxiltimer','-lxil','-lgcc','-lc','-Wl,--end-group','-o',str(artifact)])
   run([compiler.replace('g++','size'),str(artifact)])
  save(record,{'state':'completed','commands':rows,'seconds':time.monotonic()-started,'artifact':str(artifact),'artifact_sha256':sha(artifact),'prepared_sha256':sha(out/'prepared.json'),'physical_execution':False,'boundary':'Native service verification' if action=='test' else 'Cross-linked existing-BSP compatibility ELF; matching routed XSA/BSP and boot package still pending'})
 except BaseException as error:save(record,{'state':'failed','commands':rows,'error':repr(error),'seconds':time.monotonic()-started});raise

def frame(out,raw,canonical,contract,seq,export,threshold):
 from profile_runner import load_tokenizer,encode_document,allowed_path
 import numpy as np
 raw=allowed_path(raw);model=Path(export)/'model.bin';model_sha=sha(model);manifest=Path(export)/'manifest.json';
 if json.loads(manifest.read_text())['model_sha256']!=model_sha:raise ValueError('Export model digest changed')
 body=raw.read_bytes();text=body.decode('utf-8');policy=json.loads(Path(contract).read_text());tokenizer=load_tokenizer(Path(canonical),policy);pairs,_,spans=encode_document(tokenizer,text)
 ids=np.concatenate([p['input_ids'].numpy().astype('<u2') for p in pairs]).tobytes();masks=np.concatenate([p['attention_mask'].numpy().astype('u1') for p in pairs]).tobytes()
 out=Path(out);out.mkdir(exist_ok=False,parents=True);(out/'request.bin').write_bytes(packet(body,ids,masks,model_sha,threshold,seq));save(out/'request.json',{'model_sha256':model_sha,'threshold_q12':threshold,'export_manifest_sha256':sha(manifest),'authentication':False,'trusted_tokenizer_host':True,'raw_sha256':sha(raw),'request_sha256':sha(out/'request.bin'),'contract_sha256':sha(contract),'tokenizer':policy['tokenizer'],'windows':len(pairs),'content_spans':spans,'raw_bytes':len(body),'sequence':seq,'scope':'Offline exact raw/token frame; no serial transport'})

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','test','link','frame']);p.add_argument('--out',type=Path,required=True);p.add_argument('--export',type=Path);p.add_argument('--cases',type=Path);p.add_argument('--registers',type=Path);p.add_argument('--bsp',type=Path);p.add_argument('--linker',type=Path);p.add_argument('--threshold',type=int,default=0);p.add_argument('--raw',type=Path);p.add_argument('--canonical',type=Path);p.add_argument('--contract',type=Path);p.add_argument('--sequence',type=int,default=1);a=p.parse_args()
 if a.action=='prepare':prepare(a.out,a.export,a.cases,a.registers,a.bsp,a.linker,a.threshold)
 elif a.action=='frame':frame(a.out,a.raw,a.canonical,a.contract,a.sequence,a.export,a.threshold)
 else:execute(a.out,a.action)
if __name__=='__main__':main()
