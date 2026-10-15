"""Narrow, complete W8/A8 power-of-two realization of dilated64/rf127.

Integer semantics are shared with hardware/kd_cnn_v2/cnn.cpp. This numerical
treatment is distinct from the historical mixed FP32 CNN export. No held-out
inputs are discovered or opened. Calibration paths must be supplied explicitly.
"""
from __future__ import annotations
import argparse, hashlib, json, math, struct, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from cnn_models import build_model
from cnn_adoption_backend import integer_dot

CONFIG=dict(family='dilated',representation='token',embedding_dim=64,channels=64,
 depth=6,kernel_sizes=[3],dilations=[1,2,4,8,16,32],normalization='none',
 gated=False,pooling='max_mean',dropout=.1)
MAGIC=0x4b444332
CAP=1<<30

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fit_manifest(directory,files=('ids.npy','masks.npy')):
 """Validate declared role and immutable FIT identities before loading arrays."""
 directory=Path(directory).resolve();name=str(directory).lower()
 if any(word in name for word in ['notinject','final-test','final_test','final-calibration','final_calibration','confirmation']):raise ValueError('Protected/evaluation path is not FIT calibration')
 p=directory/'manifest.json';manifest=json.loads(p.read_text(encoding='utf-8-sig'))
 if manifest.get('role') not in {'fit','full_train','screen_fit','expanded_fit','fit_transfer'}:raise ValueError('Manifest does not declare an allowed FIT role')
 hashes=manifest.get('artifact_sha256',{})
 for filename in files:
  if filename not in hashes or sha(directory/filename)!=hashes[filename]:raise ValueError('FIT artifact identity mismatch: '+filename)
 if manifest.get('representation')!='token' or manifest.get('windowing',{}).get('length')!=256:raise ValueError('FIT manifest must declare token length256')
 return {'manifest_sha256':sha(p),'role':manifest['role'],'artifact_sha256':{n:hashes[n] for n in files}}
def exponent(value):return int(math.ceil(math.log2(float(value)/127))) if float(value)>0 else -16
def check_model(model):
 for k,v in CONFIG.items():
  if k not in ('dropout','representation') and model.study_config.get(k,False if k=='gated' else None)!=v:
   raise ValueError(f'Unsupported graph field {k}: {model.study_config.get(k)} != {v}')
 if tuple(model.embedding.weight.shape)!=(30522,64):raise ValueError('embedding shape')

def calibrate(model,ids,masks,batch=32):
 """Return exponents from explicitly supplied FIT tensors, never labels."""
 check_model(model);maxima={};handles=[]
 for name,module in [('projection',model.projection)]+[(f'block{i}',b) for i,b in enumerate(model.blocks)]:
  def hook(mod,inp,out,key=name):maxima[key]=max(maxima.get(key,0.),float(out[0].detach().abs().max()))
  handles.append(module.register_forward_hook(hook))
 was=model.training;model.eval();device=model.embedding.weight.device
 try:
  with torch.inference_mode():
   for p in range(0,len(ids),batch):model(ids[p:p+batch].to(device),masks[p:p+batch].to(device))
 finally:
  for h in handles:h.remove()
  model.train(was)
 if not maxima:raise ValueError('Calibration needs at least one FIT window')
 return {'embedding':exponent(model.embedding.weight.detach().abs().max()),
  'projection':exponent(maxima['projection']),
  'blocks':[exponent(maxima[f'block{i}']) for i in range(6)],'head_output':-12,
  'calibration_maxima':maxima,'calibration_windows':len(ids)}

def weight_codes(weight):
 w=weight.detach().cpu().double().numpy();axes=tuple(range(1,w.ndim))
 maximum=np.max(np.abs(w),axis=axes);ex=np.array([exponent(v) for v in maximum],dtype=np.int32)
 scales=np.exp2(ex).reshape((-1,)+(1,)*(w.ndim-1))
 return np.rint(w/scales).clip(-127,127).astype(np.int8),ex

def export_model(model,exponents,out,provenance=None):
 check_model(model);out=Path(out);out.mkdir(parents=True,exist_ok=False)
 e=exponents;arrays={};fields=[]
 if len(e['blocks'])!=6 or e['head_output']!=-12:raise ValueError('Invalid exponent schema')
 if any(type(v) is not int or not -30<=v<=16 for v in [e['embedding'],e['projection'],*e['blocks']]):raise ValueError('Activation exponents must be integers in [-30,16]')
 def field(name,value,dtype):
  arr=np.ascontiguousarray(value,dtype=dtype);arrays[name]=arr;fields.append((name,arr));return arr
 emb=np.rint(model.embedding.weight.detach().cpu().double().numpy()/2.**e['embedding']).clip(-127,127).astype('i1')
 emb[0]=0;field('embedding',emb,'i1')
 def linear(prefix,module,ei,eo):
  q,we=weight_codes(module.weight);field(prefix+'_w',q,'i1')
  b=np.rint(module.bias.detach().cpu().double().numpy()/np.exp2(ei+we))
  if not np.isfinite(b).all() or np.any(np.abs(b)>(1<<55)):raise OverflowError('bias exceeds exact bounded INT64 contract')
  field(prefix+'_b',b,'<i8');sh=field(prefix+'_s',ei+we-eo,'<i4')
  if np.any((sh< -62)|(sh>31)):raise ValueError('shift outside supported range')
 linear('projection',model.projection.conv,e['embedding'],e['projection'])
 previous=e['projection']
 for i,b in enumerate(model.blocks):
  linear(f'block{i}',b.temporal.conv,previous,e['blocks'][i]);skip=previous-e['blocks'][i]
  if not -62<=skip<=31:raise ValueError('Residual shift outside supported range')
  field(f'skip{i}',[skip],'<i4');previous=e['blocks'][i]
 linear('head',model.classifier,previous,-12)
 # Projection is k=1: serialized flattened exactly as HLS [out][in].
 payload=b''.join(a.tobytes() for _,a in fields);size=32+len(payload)
 blob=struct.pack('<8I',MAGIC,1,30522,64,256,6,12,size)+payload
 (out/'model.bin').write_bytes(blob);np.savez(out/'model.npz',**arrays)
 offset=32;descriptors={}
 for name,a in fields:descriptors[name]={'offset':offset,'bytes':a.nbytes,'shape':a.shape,'dtype':a.dtype.str};offset+=a.nbytes
 manifest={'format':'kd_cnn_pow2_w8a8_v1','config':model.study_config,'exponents':e,
  'model_sha256':sha(out/'model.bin'),'npz_sha256':sha(out/'model.npz'),'bytes':size,'arrays':descriptors,
  'source_sha256':sha(__file__),'provenance':provenance or {},
  'semantics':'RNE; signed symmetric127; INT32 dots; INT64 bias; residual branch cap2^30+ReLU plus aligned skip then sat127; max and RNE mean; Q12 logits cap2^30-1; strict margin threshold; masked outputs zero',
  'evidence':'packed numerical treatment, not quality or implementation qualification'}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
 return manifest

def shift(x,s,cap=CAP):
 """Exact INT64 nearest-even power-of-two scaling with bounded saturation."""
 s=torch.as_tensor(s,device=x.device,dtype=torch.int64);x=x.to(torch.int64)
 right=(-s).clamp(0,62);div=torch.bitwise_left_shift(torch.ones_like(s),right)
 a=x.abs();q=torch.div(a,div,rounding_mode='floor');r=a-q*div
 q=q+((r>div//2)|((r==div//2)&((q&1)!=0)&(right>0))).to(torch.int64)
 signed=torch.where(x<0,-q,q)
 factor=torch.bitwise_left_shift(torch.ones_like(s),s.clamp(0,31));bound=cap//factor
 left=x.clamp(-bound,bound)*factor
 left=torch.where(x>bound,cap,torch.where(x< -bound,-cap,left))
 return torch.where(s>=0,left,signed).clamp(-cap,cap)

def divide(x,d):
 a=x.abs();q=torch.div(a,d,rounding_mode='floor');r=a-q*d
 q=q+((2*r>d)|((2*r==d)&((q&1)!=0))).to(q.dtype)
 return torch.where(x<0,-q,q)

class IntegerCNN(nn.Module):
 def __init__(self,export):
  super().__init__();p=Path(export);self.manifest=json.loads((p/'manifest.json').read_text())
  if sha(p/'model.bin')!=self.manifest['model_sha256'] or sha(p/'model.npz')!=self.manifest['npz_sha256']:raise ValueError('export hash mismatch')
  blob=(p/'model.bin').read_bytes()
  with np.load(p/'model.npz',allow_pickle=False) as f:
   for name in f.files:
    a=f[name];desc=self.manifest['arrays'][name]
    if a.tobytes()!=blob[desc['offset']:desc['offset']+desc['bytes']]:raise ValueError('packed data mismatch')
    self.register_buffer(name,torch.from_numpy(a.copy()))
  self.study_config=self.manifest['config']
 def _conv(self,x,prefix,dilation=1):
  w=getattr(self,prefix+'_w');k=1 if w.ndim==2 else w.shape[-1]
  extent=(k-1)*dilation;x=F.pad(x.transpose(1,2),(extent//2,extent-extent//2))
  patch=x.unfold(-1,extent+1,1)[...,::dilation].permute(0,2,1,3).reshape(-1,64*k)
  out=integer_dot(patch.contiguous(),w.reshape(64,-1).T.contiguous()).to(torch.int64)
  return (out+getattr(self,prefix+'_b')).reshape(-1,256,64)
 def validate_inputs(self,ids,mask):
  if ids.shape!=mask.shape or ids.ndim!=2 or ids.shape[1]!=256:raise ValueError('Expected Bx256 ids/masks')
  if bool(((mask!=0)&(mask!=1)).any()) or bool(((ids<0)|(ids>=30522))[mask.bool()].any()):raise ValueError('Invalid mask/token')
 def codes(self,ids,mask,validate=True):
  if validate:self.validate_inputs(ids,mask)
  valid=mask.bool().unsqueeze(-1);x=self.embedding[ids.masked_fill(~mask.bool(),0)].masked_fill(~valid,0)
  x=shift(self._conv(x,'projection'),self.projection_s,127).to(torch.int8).masked_fill(~valid,0)
  for i in range(6):
   branch=shift(self._conv(x,f'block{i}',1<<i),getattr(self,f'block{i}_s')).clamp_min(0)
   skip=shift(x,getattr(self,f'skip{i}'))
   x=(branch+skip).clamp(-127,127).to(torch.int8).masked_fill(~valid,0)
  return x
 def integer_logits(self,ids,mask,validate=True):
  x=self.codes(ids,mask,validate);valid=mask.bool().unsqueeze(-1);count=mask.sum(1,keepdim=True).to(torch.int64)
  mx=x.masked_fill(~valid,-127).amax(1).masked_fill(count==0,0)
  mean=divide(x.to(torch.int64).sum(1),count.clamp_min(1)).to(torch.int8)
  pool=torch.cat((mx,mean),1);dots=integer_dot(pool.contiguous(),self.head_w.T.contiguous()).to(torch.int64)
  return shift(dots+self.head_b,self.head_s,CAP-1).to(torch.int32)
 def forward(self,input_ids,attention_mask=None,**kwargs):
  mask=input_ids.ne(0) if attention_mask is None else attention_mask
  return SimpleNamespace(logits=self.integer_logits(input_ids,mask).float()/4096.)
 def margin(self,ids,mask):
  y=self.integer_logits(ids,mask);return y[:,1]-y[:,0]
 def forward_validated(self,ids,mask):
  """Capture-safe arithmetic for inputs already checked by validate_inputs.

  Call validation for each newly supplied input outside a CUDA Graph and charge
  its host time. Skipping that prerequisite is not a qualified timing route.
  """
  return SimpleNamespace(logits=self.integer_logits(ids,mask,False).float()/4096.)

def ste_round(x):return x+(torch.round(x)-x).detach()
def fq(x,scale,cap=127):return ste_round(x/scale).clamp(-cap,cap)*scale

class FakeQuantCNN(nn.Module):
 """Training-only STE wrapper; checkpoint the original base model's state_dict.

 Floating convolution reductions can differ near integer rounding ties; final
 export/reference checks are mandatory. Fixed calibrated activation exponents,
 dynamically recomputed per-output weight exponents, no teacher inference.
 """
 def __init__(self,model,exponents):
  super().__init__();check_model(model);self.base=model;self.exponents=exponents
  self.study_config=model.study_config;self.dropout=model.dropout
 def _conv(self,x,module,ei,eo,cap=CAP):
  w=module.weight;mx=w.detach().abs().flatten(1).amax(1);we=torch.where(mx>0,torch.ceil(torch.log2(mx/127)),torch.full_like(mx,-16))
  ws=torch.pow(2.,we).view(-1,1,1);wi=ste_round(w/ws).clamp(-127,127)
  bi=ste_round(module.bias/torch.pow(2.,ei+we));xi=x/(2.**ei)
  a=F.conv1d(xi,wi,bi,dilation=module.dilation,groups=module.groups)
  return ste_round(a*torch.pow(2.,ei+we-eo).view(1,-1,1)).clamp(-cap,cap)*(2.**eo)
 def forward_features(self,input_ids,attention_mask=None):
  mask=(input_ids.ne(0) if attention_mask is None else attention_mask.bool()).unsqueeze(1);e=self.exponents
  w=fq(self.base.embedding.weight,2.**e['embedding'])
  x=F.embedding(input_ids.masked_fill(~mask.squeeze(1),0),w).transpose(1,2).masked_fill(~mask,0)
  x=self._conv(x,self.base.projection.conv,e['embedding'],e['projection'],127).masked_fill(~mask,0)
  previous=e['projection']
  for i,block in enumerate(self.base.blocks):
   eo=e['blocks'][i];extent=1<<i
   branch=self._conv(F.pad(x,(extent,extent)),block.temporal.conv,previous,eo).relu().masked_fill(~mask,0)
   # Preserve the base student's training dropout call/order. Requantize the
   # dropped branch to its output grid; eval dropout is identity as in hardware.
   branch=fq(block.dropout(branch),2.**eo,CAP)
   skip=fq(x,2.**eo,CAP);x=(branch+skip).clamp(-127*(2.**eo),127*(2.**eo)).masked_fill(~mask,0);previous=eo
  return x,mask
 def pool(self,x,mask):
  e=self.exponents['blocks'][-1];return fq(self.base.pool(x,mask),2.**e)
 def classifier(self,x):
  m=self.base.classifier;ei=self.exponents['blocks'][-1];x=fq(x,2.**ei);mx=m.weight.detach().abs().amax(1)
  we=torch.where(mx>0,torch.ceil(torch.log2(mx/127)),torch.full_like(mx,-16));ws=torch.pow(2.,we).view(-1,1)
  wi=ste_round(m.weight/ws).clamp(-127,127);bi=ste_round(m.bias/torch.pow(2.,ei+we))
  dot=F.linear(x/(2.**ei),wi,bi)
  return ste_round(dot*torch.pow(2.,ei+we+12)).clamp(-CAP+1,CAP-1)/4096.
 def forward(self,input_ids,attention_mask=None,**kwargs):
  x,mask=self.forward_features(input_ids,attention_mask);return SimpleNamespace(logits=self.classifier(self.dropout(self.pool(x,mask))))

def fake_quant_model(model,exponents):return FakeQuantCNN(model,exponents)

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True);p.add_argument('--max-windows',type=int,default=128);p.add_argument('--exponents',type=Path)
 a=p.parse_args();torch.set_num_threads(2)
 fit_identity=fit_manifest(a.calibration)
 state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);config=state.get('model_config') or state.get('model_spec',{}).get('config')
 model=build_model(config,30522);model.load_state_dict(state['state_dict']);model.eval()
 allids=np.load(a.calibration/'ids.npy',mmap_mode='r');allmask=np.load(a.calibration/'masks.npy',mmap_mode='r')
 indices=np.unique(np.linspace(0,len(allids)-1,min(a.max_windows,len(allids)),dtype=np.int64))
 ids=torch.tensor(np.asarray(allids[indices]),dtype=torch.long);mask=torch.tensor(np.asarray(allmask[indices]),dtype=torch.long)
 ex=json.loads(a.exponents.read_text()) if a.exponents else calibrate(model,ids,mask)
 manifest=export_model(model,ex,a.out,{'checkpoint':str(a.checkpoint.resolve()),'checkpoint_sha256':sha(a.checkpoint),'calibration':str(a.calibration.resolve()),'calibration_indices':indices.tolist(),'fit_identity':fit_identity})
 with torch.inference_mode():
  ref=IntegerCNN(a.out);y=ref(ids[:8],mask[:8]).logits;f=model(ids[:8],mask[:8]).logits
 check={'windows':len(y),'max_abs_logit_error':float((y-f).abs().max()),'mean_abs_logit_error':float((y-f).abs().mean()),'boundary':'FIT numerical comparison only; no labels or thresholds'}
 (a.out/'float_comparison.json').write_text(json.dumps(check,indent=2)+'\n');print(json.dumps({'export':str(a.out),'bytes':manifest['bytes'],'check':check}))
if __name__=='__main__':main()
