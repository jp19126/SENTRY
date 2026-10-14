"""Native CUDA integer CNN dots using the exporter's exact W8/A8 format.

CPU qualification uses the exported integer reference; FP32 exceptions remain
explicit. Matrix padding is execution overhead, not a change to the model.
"""
from __future__ import annotations
import copy
import torch
from torch import nn
from torch.nn import functional as F
from cnn_adoption_export import IntegerConv1d, IntegerLinear


def integer_dot(a, b):
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise TypeError('Signed int8 dot inputs required')
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError('Incompatible integer matrices')
    m, k = a.shape; n = b.shape[1]
    if a.device.type == 'cpu':
        result = a.to(torch.int64) @ b.to(torch.int64)
        if bool((result.abs() > 2147483647).any()):
            raise OverflowError('Signed int32 dot overflow')
        return result.to(torch.int32)
    if a.device.type != 'cuda' or b.device != a.device:
        raise ValueError('CPU or same-device CUDA matrices required')
    # torch._int_mm requires suitably aligned matrix sizes on this backend.
    mm, kk, nn_ = (max(32,(size + 7) // 8 * 8) for size in (m, k, n))
    ap = F.pad(a, (0, kk-k, 0, mm-m)).contiguous()
    bp = F.pad(b, (0, nn_-n, 0, kk-k)).contiguous()
    result = torch._int_mm(ap, bp)
    if result.dtype != torch.int32:
        raise TypeError('Native CUDA dot must return int32')
    return result[:m, :n]


class NativeConv(IntegerConv1d):
    def forward(self, x):
        q = torch.round(x.float()/self.input_scale).clamp(-127,127).to(torch.int8)
        extent = (self.kernel_size-1)*self.dilation+1
        patches = q.unfold(-1, extent, 1)[..., ::self.dilation]
        batch, _, positions, _ = patches.shape
        cg, og = self.in_channels//self.groups, self.out_channels//self.groups
        outputs = []
        for g in range(self.groups):
            a = patches[:, g*cg:(g+1)*cg].permute(0,2,1,3).reshape(-1,cg*self.kernel_size)
            b = self.weight_codes[g*og:(g+1)*og].reshape(og,-1).T.contiguous()
            accum = integer_dot(a,b).reshape(batch,positions,og).transpose(1,2)
            outputs.append(accum.float())
        out = torch.cat(outputs,1)*(self.input_scale*self.weight_scales).view(1,-1,1)
        if self.has_bias: out = out+self.bias_fp32.view(1,-1,1)
        return out


class NativeLinear(IntegerLinear):
    def forward(self, x):
        q = torch.round(x.float()/self.input_scale).clamp(-127,127).to(torch.int8)
        accum = integer_dot(q.reshape(-1,self.in_features),self.weight_codes.T.contiguous())
        out = accum.float()*(self.input_scale*self.weight_scales)
        if self.has_bias: out = out+self.bias_fp32
        return out.reshape(*x.shape[:-1],self.out_features)


def native_from_export_reference(model, device='cuda'):
    """Adapt the packed-file CPU runtime, preserving every loaded buffer."""
    result = copy.deepcopy(model)
    for name,module in list(result.named_modules()):
        if type(module) not in (IntegerConv1d,IntegerLinear): continue
        cls = NativeConv if isinstance(module,IntegerConv1d) else NativeLinear
        replacement = copy.deepcopy(module)
        replacement.__class__ = cls
        parent,key = name.rsplit('.',1) if '.' in name else ('',name)
        owner = result.get_submodule(parent) if parent else result
        setattr(owner,key,replacement)
    return result.to(device).eval()


def record(model):
    return {'route':'native_signed_w8a8_int32_cuda_dot_v1',
            'kernel':'torch._int_mm with zero matrix padding to multiples of8, minimum32',
            'accumulators':'signed INT32; conservative K*127^2 bound checked by exporter',
            'weights':'signed per-output/per-embedding-row W8; [-127,127]',
            'activations':'static per-module A8; nearest-even rounding/saturation',
            'exceptions':'FP32 embedding dequantization, scale products/rescale, biases, norms, gates, nonlinearities, pooling and classifier-result softmax',
            'integer_modules':sum(isinstance(m,(NativeConv,NativeLinear)) for m in model.modules()),
            'actual_tensor_payload_bytes':sum(t.numel()*t.element_size() for t in list(model.parameters())+list(model.buffers())),
            'physical_fpga':False,'fallback':False}
