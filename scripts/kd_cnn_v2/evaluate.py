"""Score kd_cnn_v2 prepared documents; freeze targets before confirmation.

No task here chooses a winner. The coordinator supplies target-list.json after
development, then uses --stage freeze before any new_confirmation scoring.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import time
import numpy as np
import torch
from data import file_sha, load, save
from models import build_student
from numerical import IntegerCNN
from teacher_cache import load_teacher
from train import setup

WORKTREE = Path(__file__).resolve().parents[2]
TASK = "results/kd_cnn_v2"
SHARED = "results/dual_track_v1/shared"
ROLES = {
    "development": {"new_selection", "new_development_calibration"},
    "confirmation": {"new_calibration", "new_confirmation"},
}
SOURCES = (
    "scripts/kd_cnn_v2/evaluate.py", "scripts/kd_cnn_v2/models.py",
    "scripts/kd_cnn_v2/numerical.py", "scripts/kd_cnn_v2/teacher_cache.py",
    "scripts/kd_cnn_v2/train.py", "scripts/kd_cnn_v2/analysis.py",
    "scripts/kd_cnn_v2/data.py", "scripts/cnn_models.py",
    "scripts/cnn_adoption_backend.py", "gate_eval.py",
    "configs/project.json", "configs/cnn_adoption_v1.json",
)

def safe_id(value):
    if not isinstance(value,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",value) or value in {".",".."}:
        raise ValueError("Unsafe target id")
    return value

def within(canonical, value, section=TASK):
    path = Path(value)
    path = path if path.is_absolute() else canonical/path
    path = path.resolve()
    if not path.is_relative_to((canonical/section).resolve()):
        raise ValueError("Target path outside task-owned results: "+str(path))
    return path

def source_hashes():
    return {name: file_sha(WORKTREE/name) for name in SOURCES}

def canonical_config_hashes(canonical):
    # teacher_config reads the canonical registry. Bind those actual bytes in
    # addition to the isolated source checkout, which may use other line endings.
    return {name:file_sha(canonical/name) for name in
            ('configs/cnn_adoption_v1.json','configs/project.json')}

def numerical_settings(device):
    requested = str(device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return {"seed": 42, "requested_device": requested, "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(torch.device(requested)) if requested.startswith("cuda") else "cpu",
            "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark}

def common(canonical):
    path = canonical/SHARED/"contract.json"
    return load(path), file_sha(path)

def prepared_identity(canonical, role):
    folder = canonical/SHARED/"data/prepared"/role
    manifest_path = folder/"manifest.json"
    manifest = load(manifest_path)
    if manifest["role"] != role or manifest["representation"] != "token" or manifest["windowing"]["stride"] != 190:
        raise ValueError("Prepared role/window policy differs")
    required = {"ids.npy","masks.npy","starts.npy","documents.jsonl"}
    if not required <= set(manifest["artifact_sha256"]): raise ValueError("Incomplete prepared manifest")
    return {"folder": str(folder), "manifest_sha256": file_sha(manifest_path),
            "artifact_sha256": manifest["artifact_sha256"],
            "raw_sha256": manifest["raw_sha256"],
            "data_protocol_sha256": manifest["data_protocol_sha256"],
            "documents": manifest["documents"], "windows": manifest["windows"]}

def verify_prepared_bytes(identity, role):
    """Opaque hash check; does not decode reserved rows before candidate freeze."""
    folder = Path(identity["folder"])
    if file_sha(folder/"manifest.json") != identity["manifest_sha256"]: raise ValueError("Manifest changed")
    manifest = load(folder/"manifest.json")
    if manifest["role"] != role or manifest["artifact_sha256"] != identity["artifact_sha256"]:
        raise ValueError("Prepared manifest association changed")
    for name, expected in identity["artifact_sha256"].items():
        if file_sha(folder/name) != expected: raise ValueError("Prepared artifact changed: "+name)
    if manifest["raw_sha256"] != identity["raw_sha256"] or manifest["data_protocol_sha256"] != identity["data_protocol_sha256"]:
        raise ValueError("Prepared raw/protocol identity changed")
    return manifest

def check_prepared(identity, role):
    folder = Path(identity["folder"])
    manifest = verify_prepared_bytes(identity,role)
    ids = np.load(folder/"ids.npy", mmap_mode="r", allow_pickle=False)
    masks = np.load(folder/"masks.npy", mmap_mode="r", allow_pickle=False)
    starts = np.load(folder/"starts.npy", mmap_mode="r", allow_pickle=False)
    with (folder/"documents.jsonl").open(encoding="utf-8") as handle:
        docs = [json.loads(line) for line in handle if line.strip()]
    if ids.shape != masks.shape or ids.ndim != 2 or ids.shape[1] != 256:
        raise ValueError("Prepared tensor shape differs")
    if ids.dtype != np.int64 or masks.dtype != np.uint8 or len(ids) != manifest["windows"]:
        raise ValueError("Prepared tensor dtype/count differs")
    if len(docs) != manifest["documents"] or len(starts) != len(docs)+1 or starts[0] != 0 or starts[-1] != len(ids) or np.any(np.diff(starts)<=0):
        raise ValueError("Prepared document association differs")
    if len({d["document_id"] for d in docs}) != len(docs): raise ValueError("Duplicate document ids")
    if role in {"new_development_calibration","new_calibration"} and any(d["label"]!=0 for d in docs):
        raise ValueError("Calibration includes positive label")
    return {"folder": folder, "manifest": manifest, "ids": ids, "masks": masks, "starts": starts, "documents": docs}

def target_descriptors(canonical, target_list, contract):
    if not isinstance(target_list,dict) or not isinstance(target_list.get("targets"),list) or not target_list["targets"]:
        raise ValueError("Nonempty target-list object required")
    result = {}
    teacher_name = contract["teacher"]["path"]
    teacher_sha = contract["teacher"]["sha256"]
    for item in target_list["targets"]:
        ident = safe_id(item["id"])
        if ident in result: raise ValueError("Duplicate target id")
        kind = item["kind"]
        if kind == "teacher":
            if item.get("checkpoint", teacher_name) != teacher_name: raise ValueError("Only common teacher allowed")
            checkpoint = canonical/teacher_name
            if file_sha(checkpoint) != teacher_sha: raise ValueError("Teacher checkpoint changed")
            result[ident] = {"kind":kind, "checkpoint":teacher_name, "checkpoint_sha256":teacher_sha,
                             "numerical_format":"fp32", "seed":contract["teacher"]["seed"]}
        elif kind == "cnn_fp32":
            checkpoint = within(canonical, item["checkpoint"])
            expected = item.get("checkpoint_sha256")
            if not expected or file_sha(checkpoint) != expected: raise ValueError("CNN checkpoint digest differs")
            state = torch.load(checkpoint,map_location="cpu",weights_only=False)
            config = state.get("model_config") or state.get("model_spec",{}).get("config")
            if not isinstance(config,dict) or "state_dict" not in state: raise ValueError("Invalid CNN checkpoint")
            result[ident] = {"kind":kind, "checkpoint":str(checkpoint.relative_to(canonical)).replace("\\","/"),
                             "checkpoint_sha256":expected, "model_config":config,
                             "numerical_format":"fp32", "seed":item.get("seed")}
        elif kind == "cnn_integer":
            folder = within(canonical, item["export_dir"])
            expected = item.get("export_manifest_sha256")
            if not expected or file_sha(folder/"manifest.json") != expected: raise ValueError("Integer export digest differs")
            manifest = load(folder/"manifest.json")
            if manifest.get("format") != "kd_cnn_pow2_w8a8_v1": raise ValueError("Incorrect integer format")
            for name,key in (("model.bin","model_sha256"),("model.npz","npz_sha256")):
                if file_sha(folder/name) != manifest[key]: raise ValueError("Packed integer artifact changed")
            provenance = manifest.get("provenance",{})
            if not provenance.get("checkpoint") or not provenance.get("checkpoint_sha256"):
                raise ValueError("Integer export lacks exact source checkpoint")
            parent_checkpoint = within(canonical,provenance["checkpoint"])
            if file_sha(parent_checkpoint) != provenance["checkpoint_sha256"]:
                raise ValueError("Integer export source checkpoint changed")
            result[ident] = {"kind":kind, "export_dir":str(folder.relative_to(canonical)).replace("\\","/"),
                             "export_manifest_sha256":expected, "model_sha256":manifest["model_sha256"],
                             "npz_sha256":manifest["npz_sha256"], "numerical_format":"w8a8_pow2_q12",
                             "checkpoint":str(parent_checkpoint.relative_to(canonical)).replace("\\","/"),
                             "checkpoint_sha256":provenance["checkpoint_sha256"],
                             "provenance":provenance, "seed":item.get("seed")}
        else: raise ValueError("Unknown target kind")
    return result

def freeze(canonical, list_path, output, device):
    output = within(canonical, output)
    if output.exists(): raise FileExistsError(output)
    target_list = load(list_path)
    contract, contract_sha = common(canonical)
    data_protocol = canonical/SHARED/"data/protocol.json"
    addendum = canonical/TASK/"data_addendum.json"
    if load(addendum)["common_contract_sha256"] != contract_sha: raise ValueError("Task addendum/common contract differs")
    if load(addendum)["prospective_protocol_sha256"] != file_sha(data_protocol): raise ValueError("Prospective protocol differs")
    targets = target_descriptors(canonical,target_list,contract)
    reserved = {role: prepared_identity(canonical,role) for role in ROLES["confirmation"]}
    for role, identity in reserved.items():
        verify_prepared_bytes(identity,role)
    payload = {"schema":"kd_cnn_v2_confirmation_freeze_v1", "frozen_utc":datetime.now(timezone.utc).isoformat(),
               "before_any_new_confirmation_scoring":True, "target_list_path":str(Path(list_path).resolve()),
               "target_list_sha256":file_sha(list_path), "targets":targets,
               "common_contract_sha256":contract_sha, "data_protocol_sha256":file_sha(data_protocol),
               "data_addendum_sha256":file_sha(addendum), "source_sha256":source_hashes(),
               "canonical_runtime_config_sha256":canonical_config_hashes(canonical),
               "numerical_settings":numerical_settings(device),
               "reserved_roles":reserved,
               "calibration_policy":{"principal_alpha":.01,"secondary_alphas":[.005,.02,.05],
                                     "threshold":"gate_eval.fit_threshold benign-only conservative order statistic",
                                     "decision":"strict document max-window margin > threshold"},
               "result_release":"task-private until sibling candidate freeze; no sibling scores read"}
    output.parent.mkdir(parents=True,exist_ok=True)
    save(output,payload)
    return payload

def confirm_freeze(canonical, path, target_id, role):
    frozen = load(path)
    if frozen.get("schema") != "kd_cnn_v2_confirmation_freeze_v1" or not frozen.get("before_any_new_confirmation_scoring"):
        raise ValueError("Completed preconfirmation freeze required")
    if target_id not in frozen["targets"] or role not in frozen["reserved_roles"]:
        raise ValueError("Target or role not nominated")
    if file_sha(canonical/SHARED/"contract.json") != frozen["common_contract_sha256"]:
        raise ValueError("Common contract changed")
    if file_sha(canonical/SHARED/"data/protocol.json") != frozen["data_protocol_sha256"]:
        raise ValueError("Data protocol changed")
    if file_sha(canonical/TASK/"data_addendum.json") != frozen["data_addendum_sha256"]:
        raise ValueError("Data addendum changed")
    if canonical_config_hashes(canonical) != frozen.get('canonical_runtime_config_sha256'):
        raise ValueError('Canonical runtime configuration changed')
    if source_hashes() != frozen["source_sha256"]: raise ValueError("Scoring source changed")
    if file_sha(frozen["target_list_path"]) != frozen["target_list_sha256"]:
        raise ValueError("Target list changed")
    contract,_ = common(canonical)
    current = target_descriptors(canonical,load(frozen["target_list_path"]),contract)
    if current != frozen["targets"]: raise ValueError("Target bytes or identity changed")
    now = prepared_identity(canonical,role)
    if now != frozen["reserved_roles"][role]: raise ValueError("Reserved data bytes changed")
    return frozen

def teacher_config(canonical,contract):
    old = load(canonical/"configs/cnn_adoption_v1.json")["qualified_teacher"]
    if old["teacher_checkpoint_sha256"] != contract["teacher"]["sha256"]:
        raise ValueError("Qualified teacher differs from common teacher")
    return {"checkpoint":contract["teacher"]["path"],"checkpoint_sha256":contract["teacher"]["sha256"],
            "qualified_development_record":f"results/cnn_adoption_v1/evaluation/development/{old['run_id']}/fp32/complete.json",
            "qualified_development_record_sha256":old["qualified_development_record_sha256"]}

def load_target(canonical, descriptor, device, contract):
    kind = descriptor["kind"]
    if kind == "teacher":
        model = load_teacher(canonical,teacher_config(canonical,contract),device)
    elif kind == "cnn_fp32":
        checkpoint = canonical/descriptor["checkpoint"]
        state = torch.load(checkpoint,map_location="cpu",weights_only=False)
        config = state.get("model_config") or state.get("model_spec",{}).get("config")
        if config != descriptor["model_config"]: raise ValueError("Checkpoint config changed")
        model = build_student(descriptor["model_config"],30522,0)
        model.load_state_dict(state["state_dict"],strict=True)
        model = model.float().to(device).eval()
    else:
        model = IntegerCNN(canonical/descriptor["export_dir"]).to(device).eval()
    return model

def score(canonical, stage, role, target_id, descriptor, data, model, device, freeze_sha=None):
    result = []
    started = time.monotonic()
    with torch.inference_mode():
        for index, doc in enumerate(data["documents"]):
            lo,hi = map(int,data["starts"][index:index+2])
            window_margins, integer_codes = [], []
            for j in range(lo,hi):
                # Prepared arrays are read-only memmaps. Give torch writable
                # window copies so neither backend aliases protected storage.
                ids = torch.as_tensor(np.array(data["ids"][j:j+1],dtype=np.int64,copy=True),device=device)
                mask = torch.as_tensor(np.array(data["masks"][j:j+1],dtype=np.int64,copy=True),device=device)
                if descriptor["kind"] == "cnn_integer":
                    code = int(model.margin(ids,mask).item())
                    value = code/4096.
                    integer_codes.append(code)
                else:
                    logits = model(input_ids=ids,attention_mask=mask).logits
                    if logits.shape != (1,2): raise ValueError("Binary model logits shape differs")
                    value = float((logits[0,1]-logits[0,0]).item())
                if not math.isfinite(value): raise ValueError("Nonfinite margin")
                window_margins.append(value)
            row = {key:doc.get(key) for key in
                   ("document_id","label","source_id","source_group_id","source_dataset","genre",
                    "payload_group_id","context_ancestry_group_id","mechanism_group_id","hard_benign",
                    "constructed","presentation","injection_position","text_sha256","background_text_sha256",
                    "content_tokens","windows","original_article_title")}
            row.update(margin=max(window_margins),window_margins=window_margins,
                       target_id=target_id,numerical_format=descriptor["numerical_format"],
                       stage=stage,role=role,prepared_manifest_sha256=file_sha(data["folder"]/"manifest.json"))
            if integer_codes: row["window_margin_q12"]=integer_codes
            result.append(row)
    return result,time.monotonic()-started

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--canonical-root",type=Path,required=True)
    p.add_argument("--stage",choices=("freeze","development","confirmation"),required=True)
    p.add_argument("--target-list",type=Path)
    p.add_argument("--freeze",type=Path)
    p.add_argument("--target-id")
    p.add_argument("--role")
    p.add_argument("--out",type=Path)
    p.add_argument("--device",default="cpu")
    a=p.parse_args()
    canonical=a.canonical_root.resolve()
    setup(42,a.device,1)
    if a.stage=="freeze":
        if not a.target_list or not a.out: raise ValueError("Freeze needs target list and output")
        frozen=freeze(canonical,a.target_list,a.out,a.device)
        print(json.dumps({"freeze":str(a.out),"targets":list(frozen["targets"]),
                          "reserved_roles":list(frozen["reserved_roles"])}))
        return
    if not a.target_id or not a.role or not a.out: raise ValueError("Scoring needs target, role and output")
    if a.role not in ROLES[a.stage]: raise ValueError("Role inappropriate for stage")
    target_id=safe_id(a.target_id)
    output=within(canonical,a.out)
    if output.exists(): raise FileExistsError(output)
    if a.stage=="confirmation":
        if not a.freeze: raise ValueError("Confirmation requires prior freeze")
        frozen=confirm_freeze(canonical,a.freeze,target_id,a.role)
        descriptor=frozen["targets"][target_id]
        identity=frozen["reserved_roles"][a.role]
        freeze_sha=file_sha(a.freeze)
    else:
        if not a.target_list: raise ValueError("Development needs target list")
        contract,_=common(canonical)
        descriptor=target_descriptors(canonical,load(a.target_list),contract)[target_id]
        identity=prepared_identity(canonical,a.role)
        freeze_sha=None
    data=check_prepared(identity,a.role)
    contract,_=common(canonical)
    model=load_target(canonical,descriptor,a.device,contract)
    scored,elapsed=score(canonical,a.stage,a.role,target_id,descriptor,data,model,a.device,freeze_sha)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8",newline="\n") as handle:
        for row in scored: handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
    save(output.with_suffix(".complete.json"),
         {"complete":True,"stage":a.stage,"role":a.role,"target_id":target_id,"target":descriptor,
          "scores_sha256":file_sha(output),"prepared_manifest_sha256":identity["manifest_sha256"],
          "freeze_sha256":freeze_sha,"documents":len(scored),"windows":sum(len(r["window_margins"]) for r in scored),
          "batch_windows":1,"device":a.device,"wall_seconds":elapsed,"score":"max_window(logit1-logit0)",
          "integer_margin_units":"Q12 divided by 4096" if descriptor["kind"]=="cnn_integer" else None,
          "no_threshold_fitted_here":True,"numerical_settings":numerical_settings(a.device),
          "source_sha256":source_hashes(),"command":sys.argv})
    print(json.dumps({"scores":str(output),"documents":len(scored),"seconds":elapsed}))

if __name__=="__main__": main()
