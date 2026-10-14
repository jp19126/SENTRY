"""Frozen integer execution/export for the span pilot, independent of fake QAT.

All multiply/accumulate and rescale operations use integer tensors. CPU int64
matmul is the literal reference; CUDA may use exact int8/int32 _int_mm with
proved K*127*127 accumulator bounds. No hardware timing claim is made.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from span_models import H, LEFT, RIGHT, R, masks_for, model_metadata


def rne_shift(x, shift):
    """Exact signed nearest-even x*2**shift, with bounded integer shifts."""
    if not isinstance(shift, torch.Tensor):
        shift = torch.as_tensor(shift, device=x.device)
    if bool(((shift < -62) | (shift > 30)).any()):
        raise OverflowError("Unsupported exported rescale shift")
    shift = shift.long()
    right = (-shift).clamp(min=0)
    denominator = torch.bitwise_left_shift(torch.ones_like(right), right)
    q = torch.div(x, denominator, rounding_mode="floor")
    remainder = x - q*denominator
    half = torch.div(denominator, 2, rounding_mode="floor")
    up = (right > 0) & ((remainder > half) | ((remainder == half) & ((q & 1) != 0)))
    rounded = q + up.long()
    return torch.bitwise_left_shift(rounded, shift.clamp(min=0))


def np_rne_scalar(value, shift):
    value, shift = int(value), int(shift)
    if shift >= 0:
        return value << shift
    den = 1 << -shift
    q, remainder = divmod(value, den)
    return q + int(2*remainder > den or (2*remainder == den and q & 1))


def export_model(model, folder):
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(folder)
    if not bool(model.quant_frozen):
        raise RuntimeError("Freeze training-calibrated formats before export")
    folder.mkdir(parents=True)
    def arr(t, dtype):
        values=t.detach().cpu().numpy()
        if not np.isfinite(values).all():raise ValueError("Nonfinite export")
        if np.issubdtype(dtype,np.integer):
            limits=np.iinfo(dtype)
            if values.min()<limits.min or values.max()>limits.max:raise OverflowError("Export integer cast")
        return values.astype(dtype)
    def codes(w,e):
        step = torch.pow(2., e.float())
        if step.ndim:
            step = step.reshape([-1]+[1]*(w.ndim-1))
        return arr(torch.round(w/step).clamp(-127,127), np.int8)
    a = {"emb_w":codes(model.embedding.weight,model.emb_exp),
         "emb_exp":arr(model.emb_exp,np.int32), "z_exp":arr(model.z_exp,np.int32)}
    convs, biases, shifts = [], [], []
    for n,conv in enumerate(model.convs):
        ew=model.conv_exp[n*64:(n+1)*64]
        convs.append(codes(conv.weight,ew).transpose(0,2,1).reshape(-1))
        scale=torch.pow(2.,(model.emb_exp+ew).float())
        biases.append(arr(torch.round(conv.bias/scale),np.int64))
        shifts.append(arr(model.emb_exp+ew-model.z_exp,np.int32))
    a.update(conv_w=np.concatenate(convs),conv_bias=np.concatenate(biases),conv_shift=np.concatenate(shifts),
        proj_w=codes(model.proj.weight,model.proj_exp),
        proj_bias=arr(torch.round(model.proj.bias/torch.pow(2.,(model.z_exp+model.proj_exp).float())),np.int64),
        proj_shift=arr(model.z_exp+model.proj_exp+12,np.int32),
        head_anchor=arr(torch.round(model.anchor*4096),np.int32),head_bias=arr(torch.round(model.bias*4096),np.int32))
    if model.arm == "sopa":
        a["epsilon_q12"]=arr(torch.round(F.logsigmoid(model.epsilon)*4096),np.int32)
        a["pattern_head"]=arr(torch.round(model.pattern_head*4096),np.int32)
    else:
        a["lambda_q8"]=arr(torch.round(model.lambdas.clamp(0,1)*256),np.uint16)
        for name in ("head_a","head_b","head_c"):
            if hasattr(model,name): a[name]=arr(torch.round(getattr(model,name)*4096),np.int32)
        if model.arm=="span_mlp":
            for name,t in (("mlp1_w",model.mlp1.weight),("mlp1_b",model.mlp1.bias),("mlp2_w",model.mlp2.weight)):
                a[name]=arr(torch.round(t*4096),np.int32)
    domain=np.linspace(-16,16,4097,dtype=np.float64)
    a["sigmoid_bits"]=(1/(1+np.exp(-domain))).astype(np.float32).view(np.uint32)
    a["tanh_q12"]=np.rint(np.tanh(domain)*4096).astype(np.int32)
    a["logsigmoid_q12"]=np.rint(-np.logaddexp(0,-domain)*4096).astype(np.int32)
    # Bounds are specific to the exported model, before loading actual scoring inputs.
    bounds={"conv_int32_raw":448*127*127,"proj_int32_raw":192*127*127}
    for name in ("conv_bias","proj_bias"):
        peak=max(abs(int(v)) for v in a[name].flat)
        bounds[name+"_absolute"]=peak
        if peak+10_000_000 >= 2**62: raise OverflowError(name)
    for name in ("head_a","head_b","head_c","head_anchor","mlp1_w","mlp1_b","mlp2_w","pattern_head","epsilon_q12"):
        if name in a and np.max(np.abs(a[name].astype(np.int64))) >= 2**30:
            raise OverflowError("Head code exceeds prespecified bound: "+name)
    # a*b/4096<=262144; conservative head bound includes all signed terms.
    bounds["head_sum_abs"]=sum(int(np.abs(a.get(k,np.zeros(1,dtype=np.int64))).astype(np.int64).sum())*factor
        for k,factor in (("head_a",32768),("head_b",32768),("head_c",262144)))
    if bounds["head_sum_abs"]>=2**62: raise OverflowError("head accumulator")
    anchor_raw=sum(abs(int(v)) for v in a["head_anchor"])*127
    anchor_exp=int(a["z_exp"])
    if anchor_exp not in range(-62,31):raise OverflowError("Anchor rescale shift")
    bounds["anchor_raw_abs"]=anchor_raw
    bounds["anchor_q12_abs"]=abs(np_rne_scalar(anchor_raw,anchor_exp))+1
    bounds["head_bias_abs"]=abs(int(a["head_bias"]))
    core_bound=bounds["head_sum_abs"]//4096+1
    if model.arm=="span_mlp":
        hidden=[sum(abs(int(w)) for w in row)*32768 for row in a["mlp1_w"]]
        bounds["mlp1_acc_abs"]=max(hidden)
        hidden=[h//4096+1+abs(int(b)) for h,b in zip(hidden,a["mlp1_b"])]
        bounds["mlp_hidden_abs"]=max(hidden)
        bounds["mlp2_acc_abs"]=sum(h*abs(int(w)) for h,w in zip(hidden,a["mlp2_w"].flat))
        core_bound=bounds["mlp2_acc_abs"]//4096+1
    if model.arm=="sopa":
        # Main transitions are tanh in[-1,1]; loops/eps are nonpositive.
        # Two main transitions bound positive state by8192; epsilon handles init.
        state_bound=max(8192,sum(max(abs(int(x)) for x in row) for row in a["epsilon_q12"]))
        bounds["sopa_state_abs"]=state_bound
        bounds["sopa_head_acc_abs"]=state_bound*sum(abs(int(x)) for x in a["pattern_head"])
        core_bound=bounds["sopa_head_acc_abs"]//4096+1
    bounds["final_logit_abs"]=core_bound+bounds["anchor_q12_abs"]+bounds["head_bias_abs"]
    for name,value in bounds.items():
        if value>=2**62:raise OverflowError("Static bound failed: "+name)
    # Bound the bias-inclusive sums and any left rescaling BEFORE saturation.
    offset=0
    for layer in range(3):
        k=(3,5,7)[layer];weights=a["conv_w"][offset:offset+64*k*64].reshape(64,k*64);offset+=64*k*64
        for row,bias,shift in zip(weights,a["conv_bias"][layer*64:(layer+1)*64],a["conv_shift"][layer*64:(layer+1)*64]):
            peak=sum(abs(int(w)) for w in row)*127+abs(int(bias))
            if int(shift) not in range(-62,31) or peak*(1<<max(0,int(shift)))>=2**62:raise OverflowError("Convolution rescale")
    for row,bias,shift in zip(a["proj_w"],a["proj_bias"],a["proj_shift"]):
        peak=sum(abs(int(w)) for w in row)*127+abs(int(bias))
        if int(shift) not in range(-62,31) or peak*(1<<max(0,int(shift)))>=2**62:raise OverflowError("Projection rescale")
    np.savez(folder/"model.npz",**a)
    manifest={"version":"span_integer_v1","model":model_metadata(model),"arrays":{},"bounds":bounds,
        "state_format":"INT32 Q12 clipped [-32768,+32768]","lambda":"uint16 Q0.8 including256",
        "risk":"binary32 lookup bits, logitQ12 index=RNE(logit/32)+2048",
        "rounding":"signed nearest ties to even after each specified rescale; no wrap",
        "virtual_padding":"embedding,z,u,v zero including projection bias; real conv padding only at document edges"}
    offset=0
    with (folder/"model.bin").open("xb") as stream:
        for name,x in a.items():
            x=np.ascontiguousarray(x.astype(x.dtype.newbyteorder("<")))
            raw=x.tobytes()
            manifest["arrays"][name]={"offset":offset,"bytes":len(raw),"dtype":x.dtype.str,"shape":list(a[name].shape)}
            stream.write(raw); offset+=len(raw)
    manifest["binary_sha256"]=hashlib.sha256((folder/"model.bin").read_bytes()).hexdigest()
    (folder/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return manifest


class IntegerSpan:
    def __init__(self, export, device="cpu", accelerated=True):
        self.folder=Path(export)
        self.manifest=json.loads((self.folder/"manifest.json").read_text(encoding="utf-8"))
        self.arm=self.manifest["model"]["arm"]
        self.device=torch.device(device)
        with np.load(self.folder/"model.npz",allow_pickle=False) as saved:
            self.a={k:torch.from_numpy(saved[k].astype(np.int64)).to(device) for k in saved.files if k!="sigmoid_bits"}
            self.lut=torch.from_numpy(saved["sigmoid_bits"].view(np.float32).copy()).to(device)
        self.accelerated=accelerated

    def mm(self,x,w):
        shape=x.shape[:-1]
        x=x.reshape(-1,x.shape[-1]).contiguous()
        if self.device.type=="cuda" and self.accelerated:
            # int8 GEMM accumulation cannot overflow: max K=448, |operand|<=127.
            if x.shape[1]*127*127 >= 2**31: raise OverflowError("int32 GEMM")
            if bool((x.abs()>127).any()) or bool((w.abs()>127).any()): raise ValueError("int8 GEMM operand")
            # _int_mm alignment requires multiples of8; all model dimensions satisfy it.
            y=torch._int_mm(x.to(torch.int8),w.T.contiguous().to(torch.int8)).long()
        else:
            y=x.long()@w.T.long()
        return y.reshape(*shape,w.shape[0])

    def encode(self,ids,content):
        a=self.a
        x=a["emb_w"][ids]*content[...,None]
        outputs=[]; at=0
        for n,k in enumerate((3,5,7)):
            w=a["conv_w"][at:at+64*k*64].reshape(64,k*64);at+=64*k*64
            padded=F.pad(x,(0,0,k//2,k//2))
            patches=padded.unfold(1,k,1).permute(0,1,3,2).contiguous().reshape(*ids.shape,k*64)
            accum=self.mm(patches,w)+a["conv_bias"][n*64:(n+1)*64]
            outputs.append(rne_shift(accum,a["conv_shift"][n*64:(n+1)*64]).clamp(0,127))
        return torch.cat(outputs,-1)*content[...,None]

    def project(self,z,content):
        a=self.a
        p=rne_shift(self.mm(z,a["proj_w"])+a["proj_bias"],a["proj_shift"])
        if self.arm!="sopa":p=p.clamp(-32768,32768)
        return p*content[...,None]

    def head(self,s,t,z):
        a=self.a
        anchor=rne_shift((z*a["head_anchor"]).sum(-1),a["z_exp"])
        if self.arm=="span_mlp":
            features=torch.cat((s,t),-1)
            # CUDA has no Long GEMM. Each row uses exact int64 products/reduction;
            # export bounds prove every signed partial sum fits without wrap.
            accum=torch.stack([(features*row).sum(-1) for row in a["mlp1_w"]],-1)
            hidden=rne_shift(accum,-12)+a["mlp1_b"]
            hidden=hidden.clamp(min=0)
            core=rne_shift((hidden*a["mlp2_w"].reshape(-1)).sum(-1),-12)
        else:
            product=rne_shift(s*t,-12)
            core=rne_shift((s*a["head_a"]+t*a["head_b"]+product*a["head_c"]).sum(-1),-12)
        return core+anchor+a["head_bias"]

    def __call__(self,input_ids,attention_mask,content_start,document_content_length,head_swap=False,trace=False):
        a=self.a
        content,eligible,lengths=masks_for(input_ids,attention_mask,content_start,document_content_length)
        z=self.encode(input_ids,content);p=self.project(z,content)
        anchors=None;state=None
        if self.arm=="sopa":
            index=rne_shift(p.clamp(-65536,65536),-5)+2048
            main=a["tanh_q12"][index[...,:32]].reshape(*p.shape[:2],2,16)
            loops=a["logsigmoid_q12"][index[...,32:]].reshape(*p.shape[:2],2,16)
            eps=a["epsilon_q12"]
            s1=eps[0].expand(p.shape[0],-1);s2=(eps[0]+eps[1]).expand(p.shape[0],-1);pooled=s2
            for i in range(p.shape[1]):
                n1=torch.maximum(torch.maximum(s1+loops[:,i,0],main[:,i,0]),eps[0])
                n2=torch.maximum(torch.maximum(s2+loops[:,i,1],s1+main[:,i,1]),n1+eps[1])
                s1=torch.where(content[:,i,None],n1,s1);s2=torch.where(content[:,i,None],n2,s2)
                pooled=torch.maximum(pooled,s2)
            global_z=z.masked_fill(~content[...,None],-2**40).max(1).values
            global_z=torch.where(content.any(1)[:,None],global_z,torch.zeros_like(global_z))
            logit=rne_shift((pooled*a["pattern_head"]).sum(-1),-12)+rne_shift((global_z*a["head_anchor"]).sum(-1),a["z_exp"])+a["head_bias"]
        else:
            padded=F.pad(p,(0,0,LEFT,RIGHT));state=torch.zeros_like(p)
            for age in range(H):
                state=(rne_shift(state*a["lambda_q8"].reshape(-1),-8)+padded[:,age:age+p.shape[1]]).clamp(-32768,32768)
            s,t=state[...,:R],state[...,R:]
            if self.arm=="independent" or head_swap:
                def pool(x):
                    y=x.masked_fill(~eligible[...,None],-2**40).max(1).values
                    return torch.where(eligible.any(1)[:,None],y,torch.zeros_like(y))
                logit=self.head(pool(s),pool(t),pool(z))
            else:
                anchors=self.head(s,t,z).masked_fill(~eligible,-2**40)
                logit=anchors.max(1).values
        no_anchor=~eligible.any(1) if self.arm!="sopa" else ~content.any(1)
        empty=document_content_length==0
        logit=torch.where(empty,a["head_bias"].expand_as(logit),logit)
        logit=logit.clamp(-65536,65536)
        risk=self.lut[rne_shift(logit,-5)+2048]
        risk=torch.where(no_anchor & ~empty,torch.zeros_like(risk),risk)
        result={"risk":risk,"logit_q12":logit,"eligible":eligible,"anchor_logits_q12":anchors,"no_anchor":no_anchor,"empty":empty}
        if trace:result.update(z_a8=z,projection_q12=p,state_q12=state)
        return result
