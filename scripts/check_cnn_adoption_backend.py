"""Qualify the native integer dot route against packed CPU reference on old FIT."""
import json,time,hashlib,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import cnn_adoption_backend as backend
from cnn_adoption_export import load_export_reference,calibration_indices
from cnn_adoption_train import Budget,setup,write,sha


def main():
    started=time.perf_counter();setup(42,'cuda',1);guard=Budget(600)
    manifest_path=ROOT/'results/cnn_adoption_v1/exports/stacked_01_final_s42/manifest.json'
    model,manifest=load_export_reference(manifest_path)
    x,mask,cal=calibration_indices(ROOT/'results/cnn_study_v1/prepared/token/full_train',512)
    x,mask=x[:4],mask[:4]
    native=backend.native_from_export_reference(model,'cuda')
    checks=[];original_dot=backend.integer_dot
    def observed_dot(a,b):
        actual=original_dot(a,b)
        expected=a.cpu().to(torch.int64)@b.cpu().to(torch.int64)
        same=torch.equal(actual.cpu().to(torch.int64),expected)
        checks.append({'shape_a':list(a.shape),'shape_b':list(b.shape),'exact_accumulator':same})
        if not same:raise ValueError('Native integer accumulation differs from int64 oracle')
        return actual
    backend.integer_dot=observed_dot
    # Include signed extremes and unaligned classifier/input dimensions.
    for m,k,n in ((1,7,2),(8,192,64),(17,1470,98)):
        a=torch.full((m,k),-127,dtype=torch.int8,device='cuda');a[:,1::2]=127
        b=torch.full((k,n),127,dtype=torch.int8,device='cuda');b[1::2,:]=-127
        observed_dot(a,b);guard.check()
    errors=[]
    with torch.inference_mode():
        for i in range(len(x)):
            reference=model(x[i:i+1],mask[i:i+1]).logits
            actual=native(x[i:i+1].cuda(),mask[i:i+1].cuda()).logits.cpu()
            delta=(actual-reference).abs();riskdelta=(actual.softmax(1)-reference.softmax(1)).abs()
            passed=torch.allclose(actual,reference,atol=1e-5,rtol=1e-5) and float(riskdelta.max())<=1e-6
            errors.append({'window':i,'max_logit_abs':float(delta.max()),'max_risk_abs':float(riskdelta.max()),'passed':bool(passed)})
            if not passed:raise ValueError('Packed CPU/native CUDA output tolerance failed')
            guard.check()
    torch.cuda.synchronize()
    result={'passed':True,'prospective_tolerances':{'logit_atol':1e-5,'logit_rtol':1e-5,'risk_atol':1e-6},
            'manifest_sha256':sha(manifest_path),'checkpoint_sha256':manifest['checkpoint_sha256'],
            'fit_input_ids_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest(),
            'fit_masks_sha256':hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
            'dot_checks':checks,'output_checks':errors,'backend':backend.record(native),
            'source_sha256':sha(Path(__file__)),'backend_source_sha256':sha(Path(backend.__file__)),
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'resources':guard.observations,
            'scope':'Four old FIT windows and signed/unaligned integer cases only; no adoption quality/confirmation or population numerical assertion',
            'wall_seconds':time.perf_counter()-started}
    write(ROOT/'results/cnn_adoption_v1/native_backend_check.json',result)
    print(json.dumps({'passed':True,'dot_checks':len(checks),'fit_windows':len(errors),'wall_seconds':result['wall_seconds']}))


if __name__=='__main__':main()
