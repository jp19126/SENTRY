"""Independent saved-score audit; permitted development + old96 only, CPU-only.

Never loads a model, invokes inference, or opens new final calibration/confirmation.
Writes only its own evidence JSON. Counts use the frozen strict score>threshold.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from decimal import Decimal,ROUND_FLOOR
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/"results/cnn_adoption_v1"
ROLES=("development_calibration","development_selection","known_diagnostics")

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()

def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def rows(path):return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]

def counts(data,threshold):
    tp=sum(r["label"]==1 and r["risk_score"]>threshold for r in data)
    fp=sum(r["label"]==0 and r["risk_score"]>threshold for r in data)
    attacks=sum(r["label"]==1 for r in data);benign=len(data)-attacks
    return {"documents":len(data),"attacks":attacks,"benign":benign,"tp":tp,"fn":attacks-tp,"fp":fp,"tn":benign-fp,"recall":tp/attacks if attacks else None,"fpr":fp/benign if benign else None,"accuracy":(tp+benign-fp)/len(data) if data else None}

def distribution(data):
    values=sorted(r["risk_score"] for r in data)
    return {"documents":len(values),"minimum":values[0] if values else None,"median":values[(len(values)-1)//2] if values else None,"p90":values[min(len(values)-1,math.floor(.9*(len(values)-1)))] if values else None,"p99":values[min(len(values)-1,math.floor(.99*(len(values)-1)))] if values else None,"maximum":values[-1] if values else None}

def strata(data,threshold):
    result={}
    for key in ("genre","source_dataset","benign_category","presentation","injection_position","mechanism_group_id","payload_group_id","context_ancestry_group_id","probe_category"):
        groups=defaultdict(list)
        for r in data:
            if r.get(key) is not None:groups[str(r[key])].append(r)
        result[key]={g:{**counts(rs,threshold),"score_distribution":distribution(rs),"source_groups":len({r["source_group_id"] for r in rs})} for g,rs in sorted(groups.items())}
    for key,fn in (("raw_length",lambda r: "missing" if r.get("raw_length") is None else "<500" if r["raw_length"]<500 else "500-1499" if r["raw_length"]<1500 else "1500-3999" if r["raw_length"]<4000 else ">=4000"),("window_count",lambda r:"1" if r["window_count"]==1 else "2-3" if r["window_count"]<=3 else ">=4")):
        groups=defaultdict(list)
        for r in data:groups[fn(r)].append(r)
        result[key]={g:{**counts(rs,threshold),"score_distribution":distribution(rs)} for g,rs in sorted(groups.items())}
    return result

def check(evaluation_id,numerical_format,plan_path=None,independent_check_path=None):
    if Path(evaluation_id).name!=evaluation_id or numerical_format not in ("fp32","historical_w8_qat","w8a8_export"):raise ValueError("Unsafe evaluation identifier/format")
    started=time.monotonic();folder=DEST/"evaluation/development"/evaluation_id/numerical_format
    prepared_root=DEST/"prepared/token";data_check_path=DEST/"data_check.json";plan=None
    if independent_check_path and not plan_path:raise ValueError("Explicit independent check requires a binding plan/hash")
    if plan_path:
        plan=read(plan_path);prepared_root=Path(plan["prepared_root"])
        if not prepared_root.is_absolute():prepared_root=ROOT/prepared_root
        if not prepared_root.resolve().is_relative_to((DEST/"prepared").resolve()):raise ValueError("Prepared path outside permitted adoption inputs")
        contract=plan["data_contract"]
        entry=independent_check_path or contract.get("independent_check",contract.get("independent_check_path"))
        if not entry:raise ValueError("Plan must explicitly name data_contract.independent_check path")
        data_check_path=Path(entry["path"] if isinstance(entry,dict) else entry)
        if not data_check_path.is_absolute():data_check_path=ROOT/data_check_path
        if not data_check_path.resolve().is_relative_to(DEST.resolve()):raise ValueError("Independent check outside adoption artifacts")
        expected=entry.get("sha256") if isinstance(entry,dict) else contract.get("independent_check_sha256")
        if not expected or digest(data_check_path)!=expected:raise ValueError("Independent data check identity differs")
    saved=read(folder/"complete.json");assert saved["complete"] is True
    association=saved["association"];assert association["stage"]=="development"
    assert set(association["data"])==set(ROLES)
    assert identity(association)==saved["association_sha256"]
    assert saved["evaluation_id"]==evaluation_id and saved["target_key"]==evaluation_id+":"+numerical_format
    assert association["target"]["numerical_format"]==numerical_format
    policy=association["calibration_policy"]
    assert policy["primary_alpha"]==.005 and policy["secondary_alphas"]==[.01,.02] and policy["rule"]=="score > threshold" and policy["aggregation"]=="max_window_risk"
    checkpoint=association["target"]["checkpoint_identity"]
    if "sha256" in checkpoint:assert digest(ROOT/checkpoint["path"])==checkpoint["sha256"]
    elif "files_sha256" in checkpoint:
        for name,value in checkpoint["files_sha256"].items():assert digest(ROOT/checkpoint["path"]/name)==value
    source_status={name:{"recorded":expected,"current":digest(ROOT/name),"matches_current":digest(ROOT/name)==expected} for name,expected in association["source_sha256"].items()}
    # Later source changes are recorded, not silently assigned to saved outcomes.
    export_check=None
    if numerical_format=="w8a8_export":
        export_base=DEST/"exports"/evaluation_id;manifest_path=export_base/"manifest.json";manifest=read(manifest_path)
        assert manifest==saved["backend"]["manifest"]
        assert digest(manifest_path)==association["target"]["export_manifest_sha256"]==saved["backend"]["export_manifest_sha256"]
        assert saved["backend"]["route"] in ("packed_W8_A8_checked_int32_CPU_reference","packed_W8_A8_native_CUDA_int32")
        assert manifest["format_id"]=="sentry_cnn_static_w8a8_int32ref_v1" and manifest["run_id"]==evaluation_id
        assert manifest["checkpoint_sha256"]==checkpoint["sha256"] and digest(ROOT/manifest["checkpoint_path"])==checkpoint["sha256"]
        assert manifest["model_source_sha256"]==association["source_sha256"]["scripts/cnn_models.py"]
        assert manifest["configuration"]==association["target"]["model_spec"]["config"] and identity(manifest["configuration"])==manifest["configuration_sha256"]
        assert digest(ROOT/manifest["checkpoint_path"].replace("checkpoint_selected.pt","run.json"))==manifest["run_metadata_sha256"]
        payload={k:v for k,v in manifest.items() if k!="manifest_payload_sha256"};assert identity(payload)==manifest["manifest_payload_sha256"]
        for name,expected in manifest["storage"]["files"].items():
            assert Path(name).name==name and digest(export_base/name)==expected["sha256"] and (export_base/name).stat().st_size==expected["bytes"]
        cal_identity=manifest["activation_calibration"]["input_file_sha256"]
        for name,expected in cal_identity.items():assert digest(prepared_root/"fit"/name)==expected
        export_check={"manifest_sha256":digest(manifest_path),"packed_files":manifest["storage"]["files"],"checkpoint_configuration_run_fit_associations_exact":True,"actual_saved_route":saved["backend"]["route"],"source_identity_policy":"Exact recorded source hashes retained; later current-file changes are separately reported, never substituted for executed versions."}
    elif numerical_format=="fp32":
        assert association["target"].get("export_manifest_sha256") is None
    score_data={};role_checks={}
    for role in ROLES:
        filename=role+"_scores.jsonl";assert digest(folder/filename)==saved["score_files_sha256"][filename]
        base=ROOT/"results/cnn_study_v1/prepared/token/diagnostics" if role=="known_diagnostics" else prepared_root/role
        expected=association["data"][role]["files"]
        for name,value in expected.items():assert digest(base/name)==value,(role,name)
        metadata=rows(base/"documents.jsonl");scored=rows(folder/filename)
        assert len(metadata)==len(scored)
        ids=np.load(base/"ids.npy",mmap_mode="r",allow_pickle=False);masks=np.load(base/"masks.npy",mmap_mode="r",allow_pickle=False);starts=np.load(base/"starts.npy",mmap_mode="r",allow_pickle=False)
        assert starts[0]==0 and starts[-1]==len(ids) and len(starts)==len(scored)+1
        seen=set();tail_winners=0;nonfirst_winners=0
        for i,(doc,r) in enumerate(zip(metadata,scored)):
            assert r["document_id"] not in seen;seen.add(r["document_id"])
            for field,value in doc.items():assert r[field]==value,(role,i,field)
            low,high=map(int,starts[i:i+2]);values=r["window_scores"]
            assert len(values)==high-low==r["window_count"] and values
            assert all(math.isfinite(v) and 0<=v<=1 for v in values)
            assert r["risk_score"]==max(values)
            h=hashlib.sha256()
            for key,array in (("ids",ids),("masks",masks)):
                value=np.ascontiguousarray(array[low:high]);h.update(identity({"kind":key,"dtype":str(value.dtype),"shape":list(value.shape)}).encode());h.update(value.tobytes())
            assert h.hexdigest()==r["window_input_sha256"]
            winner=values.index(max(values));tail_winners+=winner==len(values)-1;nonfirst_winners+=winner>0
        if role=="development_calibration":assert all(r["label"]==0 for r in scored)
        role_checks[role]={"documents":len(scored),"windows":len(ids),"all_max_window_risks_exact":True,"all_window_input_associations_exact":True,"last_window_risk_winners":tail_winners,"nonfirst_window_risk_winners":nonfirst_winners,"score_sha256":digest(folder/filename)}
        score_data[role]=scored
    cal=score_data["development_calibration"];selection=score_data["development_selection"];diagnostics=score_data["known_diagnostics"]
    ordered=sorted(r["risk_score"] for r in cal);operating={};primary_threshold=None
    for alpha in (.005,.01,.02):
        k=int((Decimal(str(alpha))*len(ordered)).to_integral_value(rounding=ROUND_FLOOR));threshold=ordered[len(ordered)-k-1]
        actual=counts(selection,threshold);reported=saved["metrics"][str(alpha)]
        assert reported["threshold"]==threshold and reported["attack_rule"]=="score > threshold"
        for key,value in actual.items():assert reported[key]==value,(alpha,key)
        detailed=saved["operating_point_details"][str(alpha)]
        for key,value in actual.items():assert detailed["overall"][key]==value
        hard=[r for r in selection if r["label"]==0 and r.get("hard_benign") is True]
        for key,value in counts(hard,threshold).items():assert detailed["hard_benign"][key]==value
        genres=association["quality_gates"].get("primary_genre_values",["email"])
        email_rows=[r for r in selection if r.get("genre") in genres]
        for key,value in counts(email_rows,threshold).items():assert detailed["primary_application"][key]==value
        for key,value in counts(diagnostics,threshold).items():assert saved["known_diagnostics"][str(alpha)]["overall"][key]==value
        gates=association["quality_gates"]
        overall_pass=None if actual["recall"] is None or actual["fpr"] is None else actual["recall"]>=gates["minimum_recall"] and actual["fpr"]<=gates["maximum_fpr"]
        hard_metric=counts(hard,threshold);email_metric=counts(email_rows,threshold)
        hard_pass=None if hard_metric["fpr"] is None else hard_metric["fpr"]<=gates["maximum_hard_benign_fpr"]
        email_pass=None if email_metric["recall"] is None or email_metric["fpr"] is None else email_metric["recall"]>=gates["minimum_primary_email_recall"] and email_metric["fpr"]<=gates["maximum_primary_email_fpr"]
        assert detailed["overall"]["passes"] is overall_pass and detailed["hard_benign_passes"] is hard_pass and detailed["primary_application_passes"] is email_pass
        mechanism_checks={}
        for name in sorted(set(gates["required_mechanism_groups"])|{r["mechanism_group_id"] for r in selection if r["label"]==1 and r.get("mechanism_group_id")}):
            attack_rows=[r for r in selection if r["label"]==1 and r.get("mechanism_group_id")==name]
            source_n=len({r["source_group_id"] for r in attack_rows});payload_n=len({r["payload_group_id"] for r in attack_rows if r.get("payload_group_id") is not None})
            adequate=len(attack_rows)>=gates["minimum_mechanism_attacks"] and source_n>=gates["minimum_mechanism_attack_source_groups"] and payload_n>=gates["minimum_mechanism_payload_groups"]
            recall=counts(attack_rows,threshold)["recall"];passes=None if not adequate else recall>=gates["minimum_mechanism_recall"]
            recorded=detailed["strata"]["mechanism_group_id:"+name]
            assert recorded["attacks"]==len(attack_rows) and recorded["attack_source_groups"]==source_n and recorded["attack_payload_groups"]==payload_n
            assert recorded["adequately_populated"] is adequate and recorded["passes_mechanism_recall"] is passes
            mechanism_checks[name]={"attacks":len(attack_rows),"attack_source_groups":source_n,"attack_payload_groups":payload_n,"adequately_populated":adequate,"passes_mechanism_recall":passes}
        required_checks={name:mechanism_checks[name] for name in gates["required_mechanism_groups"]}
        required_pass=None if any(v["passes_mechanism_recall"] is None for v in required_checks.values()) else all(v["passes_mechanism_recall"] for v in required_checks.values())
        assert detailed["required_mechanisms_pass"] is required_pass
        operating[str(alpha)]={"threshold":threshold,"calibration":counts(cal,threshold),"selection":actual,"hard_benign":hard_metric,"primary_email":email_metric,"known96":counts(diagnostics,threshold),"gates":{"overall":overall_pass,"hard_benign":hard_pass,"primary_email":email_pass,"required_mechanisms":required_pass},"required_mechanism_checks":required_checks,"all_mechanism_checks":mechanism_checks}
        if alpha==.005:primary_threshold=threshold
    # Inspect calibration tail composition and paired background score gaps,
    # without introducing another threshold candidate or selecting anything.
    calibration_tail=sorted(cal,key=lambda r:r["risk_score"],reverse=True)[:20]
    backgrounds=defaultdict(list)
    for r in selection:backgrounds[r["source_group_id"]].append(r)
    pairs=[]
    for group,rs in backgrounds.items():
        benign=[r for r in rs if r["label"]==0];attack=[r for r in rs if r["label"]==1]
        if len(benign)==1 and attack:
            for a in attack:pairs.append({"genre":a["genre"],"benign_category":benign[0]["benign_category"],"benign_context_ancestry_group_id":benign[0].get("context_ancestry_group_id"),"mechanism":a["mechanism_group_id"],"gap":a["risk_score"]-benign[0]["risk_score"]})
    pair_summary={"pairs":len(pairs),"attack_score_above_matched_benign":sum(p["gap"]>0 for p in pairs),"attack_score_equal_matched_benign":sum(p["gap"]==0 for p in pairs),"attack_score_below_matched_benign":sum(p["gap"]<0 for p in pairs),"median_gap":sorted(p["gap"] for p in pairs)[(len(pairs)-1)//2] if pairs else None,"by_benign_category":{k:{"pairs":sum(p["benign_category"]==k for p in pairs),"attack_above":sum(p["benign_category"]==k and p["gap"]>0 for p in pairs)} for k in sorted({p["benign_category"] for p in pairs})}}
    pair_summary["by_benign_context_ancestry_group"]={k:{"pairs":sum(p["benign_context_ancestry_group_id"]==k for p in pairs),"attack_above":sum(p["benign_context_ancestry_group_id"]==k and p["gap"]>0 for p in pairs)} for k in sorted({p["benign_context_ancestry_group_id"] for p in pairs if p["benign_context_ancestry_group_id"]})}
    # Prepared full-tail evidence is tied to these exact new role file hashes.
    data_check=read(data_check_path);assert data_check["complete"] and data_check["all_windows_masks_ids_raw_associations_and_tails_verified"]
    if plan:assert data_check["data_protocol_sha256"]==plan["data_contract"]["data_protocol_sha256"]
    for role in ROLES[:2]:
        for name in ("ids.npy","masks.npy","starts.npy","documents.jsonl"):assert data_check["roles"][role]["artifact_sha256"][name]==association["data"][role]["files"][name]
    result={"complete":True,"evaluation_id":evaluation_id,"format":numerical_format,"cpu_wall_seconds":time.monotonic()-started,"prepared_root":str(prepared_root.relative_to(ROOT)),"plan_sha256":digest(plan_path) if plan_path else None,"saved_complete_sha256":digest(folder/"complete.json"),"association_sha256":saved["association_sha256"],"checkpoint_identity":checkpoint,"source_identity_status":source_status,"role_checks":role_checks,"operating_points":operating,"primary_development_strata":strata(selection,primary_threshold),"primary_calibration_strata":strata(cal,primary_threshold),"calibration_top20_categories":dict(Counter(r.get("benign_category") for r in calibration_tail)),"calibration_top20_genres":dict(Counter(r.get("genre") for r in calibration_tail)),"matched_background_pairs":pair_summary,"primary_error_ids":{"fp":[r["document_id"] for r in selection if r["label"]==0 and r["risk_score"]>primary_threshold],"fn":[r["document_id"] for r in selection if r["label"]==1 and r["risk_score"]<=primary_threshold]},"verified_new_role_tail_evidence_sha256":digest(data_check_path),"scope":"Saved development and old96 diagnostic scores only; descriptive error diagnosis, no inference, threshold search, selection decision, finalcalibration or confirmation score access; length/payload/context associations are not causal mechanisms. r2 has six authored context families perrole from one construction author, not30independent authors/discoursepopulations."}
    result["export_identity_check"]=export_check
    target=DEST/(evaluation_id+("" if numerical_format=="fp32" else "_"+numerical_format)+"_development_score_check.json");target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"complete":True,"evidence":str(target.relative_to(ROOT)),"cpu_wall_seconds":result["cpu_wall_seconds"],"operating_points":operating,"calibration_top20_categories":result["calibration_top20_categories"],"matched_background_pairs":pair_summary},indent=2))

def audit_selected(run_id,plan_path):
    """Bounded saved-run/context-window diagnosis; FIT/dev only, no inference."""
    if Path(run_id).name!=run_id:raise ValueError("Unsafe run ID")
    begin=time.monotonic();plan=read(plan_path)
    prepared=ROOT/plan["prepared_root"];raw_root=ROOT/plan["raw_root"]
    if not prepared.resolve().is_relative_to((DEST/"prepared").resolve()) or not raw_root.resolve().is_relative_to(DEST.resolve()):raise ValueError("Input paths outside study")
    folder=DEST/"runs"/run_id;run=read(folder/"run.json");assert identity(run["association"])==run["association_sha256"]
    paths=[folder/"calibration_scores.jsonl",folder/"selection_scores.jsonl",folder/"checkpoint_selected.pt"]
    before={p.name:digest(p) for p in paths};cal=rows(paths[0]);selected=rows(paths[1]);history=read(folder/"history.json")
    snapshot={"run_id":run_id,"score_and_checkpoint_sha256":before,"training_terminal":(folder/"complete.json").exists(),"history_epochs":len(history)}
    operating={};primary=None
    for alpha in (.005,.01,.02):
        ordered=sorted(r["risk_score"] for r in cal);k=int((Decimal(str(alpha))*len(ordered)).to_integral_value(rounding=ROUND_FLOOR));threshold=ordered[len(ordered)-k-1]
        operating[str(alpha)]={"threshold":threshold,"selection":counts(selected,threshold),"calibration":counts(cal,threshold)}
        if alpha==.005:primary=threshold
    matching=[]
    for record in history:
        if all(record["selection_metrics"][alpha]["threshold"]==value["threshold"] and all(record["selection_metrics"][alpha][k]==v for k,v in value["selection"].items()) for alpha,value in operating.items()):matching.append(record["epoch"])
    assert matching,"Saved selected scores do not match any recorded epoch"
    snapshot["matching_selected_score_epochs"]=matching
    inputs={};score_maps={"development_calibration":{r["document_id"]:r for r in cal},"development_selection":{r["document_id"]:r for r in selected}}
    for role in ("fit","development_calibration","development_selection"):
        base=prepared/role;docs=rows(base/"documents.jsonl")
        association=run["association"]["fitting" if role=="fit" else role]
        for name,expected in association["files"].items():assert digest(base/name)==expected
        inputs[role]={"documents":docs,"ids":np.load(base/"ids.npy",mmap_mode="r",allow_pickle=False),"masks":np.load(base/"masks.npy",mmap_mode="r",allow_pickle=False),"starts":np.load(base/"starts.npy",mmap_mode="r",allow_pickle=False),"content_start":np.load(base/"content_start.npy",mmap_mode="r",allow_pickle=False)}
        if role!="fit":
            scored=cal if role=="development_calibration" else selected;assert len(scored)==len(docs)
            for i,(d,r) in enumerate(zip(docs,scored)):
                assert all(r[k]==v for k,v in d.items());a,b=map(int,inputs[role]["starts"][i:i+2])
                assert len(r["window_scores"])==b-a and r["risk_score"]==max(r["window_scores"])
                r["window_count"]=b-a  # Training saves metadata.windows; evaluator adds this alias.
    # Source construction code is inspected, not reserved role documents. Use
    # only the three permitted role keys of its authored discourse definitions.
    sys.path.insert(0,str(ROOT/"scripts"));sys.path.insert(0,str(ROOT))
    from cnn_adoption_data import R2_CONTEXTS
    from transformers import AutoTokenizer
    project=read(ROOT/"configs/project.json");policy=project["models"]["detector"]
    tok=AutoTokenizer.from_pretrained(policy["id"],revision=policy["tokenizer_revision"],cache_dir=ROOT/project["paths"]["checkpoints"] / "huggingface",use_fast=True,local_files_only=True)
    summaries={};examples=[];isolated=[];context_windows={}
    for role in ("fit","development_calibration","development_selection"):
        data=inputs[role];summary=Counter();family_counts=defaultdict(Counter)
        for index,(r,d) in enumerate(zip(rows(raw_root/(role+".jsonl")),data["documents"])):
            assert r["document_id"]==d["document_id"] and r["text_sha256"]==d["text_sha256"]
            if r["label"] or not r.get("context_ancestry_group_id"):continue
            summary["constructed_benign_documents"]+=1
            key=r["context_ancestry_group_id"].split(":")[-1]
            template=R2_CONTEXTS[role][key][r["context_realization_id"]]
            for a,b in (("a fictional ","a "),("an earlier fictional ","an earlier "),("the fictional ","the "),("fictional ",""),("a inscription","an inscription"),("A oral-history","An oral-history"),("a oral-history","an oral-history"),("a artwork","an artwork")):template=template.replace(a,b)
            prefix,suffix=template.split("{payload}");text=r["text"]
            assert text.count(prefix)==1
            intro_start=text.index(prefix);payload_start=intro_start+len(prefix);suffix_start=text.index(suffix,payload_start);suffix_end=suffix_start+len(suffix)
            encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True,return_attention_mask=False,return_token_type_ids=False,truncation=False,verbose=False)
            a,b=map(int,data["starts"][index:index+2]);scores=score_maps.get(role,{}).get(r["document_id"])
            winner=scores["window_scores"].index(scores["risk_score"]) if scores else None
            statuses=[];doc_isolated=[]
            for w in range(a,b):
                start=int(data["content_start"][w]);count=int(data["masks"][w].sum())-2
                assert data["ids"][w,1:count+1].tolist()==encoded["input_ids"][start:start+count]
                low=encoded["offset_mapping"][start][0];high=encoded["offset_mapping"][start+count-1][1]
                overlap=lambda x,y:high>x and low<y
                touches_payload=overlap(payload_start,suffix_start);touches_intro=overlap(intro_start,payload_start);touches_post=overlap(suffix_start,suffix_end)
                status="both" if touches_intro and touches_post else "intro_only" if touches_intro else "post_only" if touches_post else "neither"
                if touches_payload:
                    summary["payload_windows"]+=1;summary["payload_windows_context_"+status]+=1
                    if status=="neither":
                        assert low>=payload_start and high<=suffix_start
                        h=hashlib.sha256(data["ids"][w].tobytes()+data["masks"][w].tobytes()).hexdigest();doc_isolated.append(w-a);isolated.append({"role":role,"document_id":r["document_id"],"window_index":w-a,"input_hash":h})
                statuses.append({"payload":touches_payload,"context":status})
            family=r["context_ancestry_group_id"];family_counts[family]["documents"]+=1
            if doc_isolated:
                summary["documents_with_isolated_payload_window"]+=1;family_counts[family]["isolated_documents"]+=1
            if scores:
                state=statuses[winner];summary["winner_payload_"+str(state["payload"]).lower()+"_context_"+state["context"]]+=1
                if state["payload"] and state["context"]=="neither":summary["isolated_payload_risk_winners"]+=1
                if scores["risk_score"]>primary:
                    summary["primary_fp"]+=1;family_counts[family]["primary_fp"]+=1
                    if len(examples)<20:examples.append({"role":role,"document_id":r["document_id"],"risk_score":scores["risk_score"],"context_family":family,"payload_group_id":r["payload_group_id"],"windows":b-a,"winner_window":winner,"winning_window_payload":state["payload"],"winning_window_context":state["context"],"isolated_payload_windows":doc_isolated})
                context_windows[(role,r["document_id"])]=statuses
        summaries[role]={"counts":dict(summary),"families":{k:dict(v) for k,v in family_counts.items()}}
    # Exact cross-label input collisions are distinguished from merely lacking
    # framing. A single identical window does not prove an irreducible document
    # contradiction: another positive window may still distinguish the label.
    isolated_hashes={r["input_hash"] for r in isolated};matches=[]
    if isolated_hashes:
        for role,data in inputs.items():
            for i,d in enumerate(data["documents"]):
                if not d["label"]:continue
                a,b=map(int,data["starts"][i:i+2])
                for w in range(a,b):
                    h=hashlib.sha256(data["ids"][w].tobytes()+data["masks"][w].tobytes()).hexdigest()
                    if h in isolated_hashes:matches.append({"role":role,"document_id":d["document_id"],"window_index":w-a,"input_hash":h})
    after={p.name:digest(p) for p in paths}
    snapshot["stable_during_audit"]=before==after
    assert before==after,"Selected score/checkpoint changed during audit; retry consistent snapshot"
    tail=sorted(cal,key=lambda r:r["risk_score"],reverse=True)[:20]
    pairs=defaultdict(list)
    for r in selected:pairs[r["source_group_id"]].append(r)
    paired=[]
    for rs in pairs.values():
        benign=[r for r in rs if r["label"]==0];attack=[r for r in rs if r["label"]==1]
        if len(benign)==1:
            paired.extend({"gap":a["risk_score"]-benign[0]["risk_score"],"benign_context":benign[0].get("context_ancestry_group_id")} for a in attack)
    result={"complete":True,"cpu_wall_seconds":time.monotonic()-begin,"snapshot":snapshot,"operating_points":operating,"calibration_top20_context_families":dict(Counter(r.get("context_ancestry_group_id","natural") for r in tail)),"primary_selection_strata":strata(selected,primary),"primary_calibration_strata":strata(cal,primary),"context_window_inspection":summaries,"isolated_payload_windows":isolated,"exact_isolated_benign_window_matching_positive_windows":matches,"informative_error_ids":examples,"paired_backgrounds":{"pairs":len(paired),"attack_above_benign":sum(p["gap"]>0 for p in paired),"attack_below_benign":sum(p["gap"]<0 for p in paired),"attack_equal_benign":sum(p["gap"]==0 for p in paired)},"limits":"FIT/development rawdocuments and saved selected scores only; no reserved roles or inference. Prefix/postscript loss is inspected token coverage, not a causal model explanation. Identical window alone does not prove unavoidable document-level contradiction. CNN local receptive fields may limit contextual interpretation even when a window contains framing. Contextfamilies remain controlled and one-author."}
    target=DEST/(run_id+"_context_window_diagnosis.json");target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"complete":True,"cpu_wall_seconds":result["cpu_wall_seconds"],"snapshot":snapshot,"primary":operating["0.005"],"context_windows":summaries,"exact_collision_positive_windows":len(matches),"paired_backgrounds":result["paired_backgrounds"],"evidence":str(target.relative_to(ROOT))},indent=2))

def common_selected(plan_path):
    """Four completed CNN selected-score audits; reuse frozen token/tail proof."""
    begin=time.monotonic();plan=read(plan_path);prepared=ROOT/plan["prepared_root"]
    if not prepared.resolve().is_relative_to((DEST/"prepared").resolve()):raise ValueError("Prepared path outside study")
    data_check_path=DEST/"data_check_r2.json";proof=read(data_check_path)
    assert digest(data_check_path)==plan["data_contract"]["independent_check_sha256"]
    assert proof["complete"] and proof["data_protocol_sha256"]==plan["data_contract"]["data_protocol_sha256"]
    inputs={}
    for role in ("development_calibration","development_selection"):
        base=prepared/role;expected=proof["roles"][role]["artifact_sha256"]
        for name in ("ids.npy","masks.npy","starts.npy","documents.jsonl"):assert digest(base/name)==expected[name]
        inputs[role]={"metadata":rows(base/"documents.jsonl"),"starts":np.load(base/"starts.npy",mmap_mode="r",allow_pickle=False),"files":{name:expected[name] for name in ("ids.npy","masks.npy","starts.npy","documents.jsonl")}}
    desired=("textcnn_01","stacked_01","local_context_03","local_context_03_residual_replace_c98")
    recipes=[r for r in plan["runs"] if r.get("stage")=="development_supervised" and r["model_id"] in desired]
    assert len(recipes)==4 and {r["model_id"] for r in recipes}==set(desired)
    results={};table=[]
    for recipe in recipes:
        folder=DEST/"runs"/recipe["run_id"];done=read(folder/"complete.json");run=read(folder/"run.json");history=read(folder/"history.json")
        association=run["association"];assert done["complete"] and done["run_id"]==recipe["run_id"]
        assert association["recipe"]==recipe and association["model_spec"]==done["model_spec"]
        assert identity(association)==run["association_sha256"]==done["association_sha256"]
        checkpoint=ROOT/done["checkpoint"];assert digest(checkpoint)==done["checkpoint_sha256"]
        epoch=done["selected_epoch"];record=next(r for r in history if r["epoch"]==epoch)
        assert history[-1]["epoch"]==done["epochs_completed"]
        scores={};hashes={}
        for short,role in (("calibration","development_calibration"),("selection","development_selection")):
            filename=folder/(short+"_scores.jsonl");hashes[short]=digest(filename);values=rows(filename);data=inputs[role]
            assert association[role]["files"]==data["files"] and len(values)==len(data["metadata"])
            for i,(row,metadata) in enumerate(zip(values,data["metadata"])):
                assert all(row[k]==v for k,v in metadata.items())
                n=int(data["starts"][i+1]-data["starts"][i]);assert len(row["window_scores"])==n
                assert all(math.isfinite(v) and 0<=v<=1 for v in row["window_scores"])
                assert row["risk_score"]==max(row["window_scores"]);row["window_count"]=n
            if role=="development_calibration":assert all(r["label"]==0 for r in values)
            scores[short]=values
        for name,expected in association["fitting"]["files"].items():assert digest(prepared/"fit"/name)==expected
        operating={}
        cal=scores["calibration"];sel=scores["selection"]
        for alpha in (.005,.01,.02):
            ordered=sorted(r["risk_score"] for r in cal);k=int((Decimal(str(alpha))*len(ordered)).to_integral_value(rounding=ROUND_FLOOR));threshold=ordered[len(ordered)-k-1]
            overall=counts(sel,threshold);hard=counts([r for r in sel if r["label"]==0 and r.get("hard_benign")],threshold);email=counts([r for r in sel if r["genre"]=="natural_email"],threshold)
            for measured in (done["metrics"][str(alpha)],record["selection_metrics"][str(alpha)]):
                assert measured["threshold"]==threshold and measured["attack_rule"]=="score > threshold"
                assert all(measured[key]==value for key,value in overall.items())
            operating[str(alpha)]={"threshold":threshold,"overall":overall,"hard_benign":hard,"primary_email":email,"overall_gate_pass":overall["recall"]>=.99 and overall["fpr"]<=.01,"hard_benign_gate_pass":hard["fpr"]<=.05,"email_gate_pass":email["recall"]>=.99 and email["fpr"]<=.01}
        primary=operating["0.005"];threshold=primary["threshold"]
        false_positive=[r for r in sel if r["label"]==0 and r["risk_score"]>threshold]
        tail=sorted(cal,key=lambda r:r["risk_score"],reverse=True)[:20]
        fp_groups=Counter(r.get("payload_group_id") or "natural" for r in false_positive)
        fp_contexts=Counter(r.get("context_ancestry_group_id") or "natural" for r in false_positive)
        pairs=defaultdict(list)
        for r in sel:pairs[r["source_group_id"]].append(r)
        gaps=[]
        for group in pairs.values():
            benign=[r for r in group if r["label"]==0]
            if len(benign)==1:gaps.extend(r["risk_score"]-benign[0]["risk_score"] for r in group if r["label"]==1)
        model=recipe["model_id"]
        results[model]={"run_id":recipe["run_id"],"seed":done["seed"],"epochs_completed":done["epochs_completed"],"selected_epoch":epoch,"checkpoint_sha256":done["checkpoint_sha256"],"association_sha256":done["association_sha256"],"score_sha256":hashes,"operating_points":operating,"primary_selection_strata":strata(sel,threshold),"primary_calibration_strata":strata(cal,threshold),"primary_fp_context_family_counts":dict(fp_contexts),"primary_fp_payload_group_counts":dict(fp_groups),"primary_fp_ids":[r["document_id"] for r in false_positive],"calibration_top20_context_family_counts":dict(Counter(r.get("context_ancestry_group_id") or "natural" for r in tail)),"calibration_top20_payload_group_counts":dict(Counter(r.get("payload_group_id") or "natural" for r in tail)),"paired_backgrounds":{"pairs":len(gaps),"attack_above_benign":sum(g>0 for g in gaps),"attack_below_benign":sum(g<0 for g in gaps),"attack_equal_benign":sum(g==0 for g in gaps)},"all_saved_metadata_and_window_max_associations_exact":True,"all_three_thresholds_and_saved_counts_exact":True}
        table.append({"model":model,"selected_epoch":epoch,"tp":primary["overall"]["tp"],"fn":primary["overall"]["fn"],"fp":primary["overall"]["fp"],"tn":primary["overall"]["tn"],"recall":primary["overall"]["recall"],"fpr":primary["overall"]["fpr"],"hard_benign_fp":primary["hard_benign"]["fp"],"hard_benign_n":primary["hard_benign"]["benign"],"hard_benign_fpr":primary["hard_benign"]["fpr"],"email_tp":primary["primary_email"]["tp"],"email_fp":primary["primary_email"]["fp"],"email_recall":primary["primary_email"]["recall"],"email_fpr":primary["primary_email"]["fpr"],"overall_gate_pass":primary["overall_gate_pass"],"hard_benign_gate_pass":primary["hard_benign_gate_pass"],"email_gate_pass":primary["email_gate_pass"]})
    result={"complete":True,"cpu_wall_seconds":time.monotonic()-begin,"plan_sha256":digest(plan_path),"data_check_r2_sha256":digest(data_check_path),"prepared_root":str(prepared.relative_to(ROOT)),"common_primary_table":table,"models":results,"scope":"Completed commonCNN selectedseed42 floating development only. Reused independently verified r2 token/tail inputproof, no repeated tokenization or inference. Hard/email/overall gates are selected-development empirical screens, not three-seed confirmation or population guarantees. Six controlled contextfamilies perrole, one author. No reservedrolecontent/scores or newgates/data/trainer/config/mainreport changes."}
    target=DEST/"common_supervised_development_check_r2.json";target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"complete":True,"cpu_wall_seconds":result["cpu_wall_seconds"],"common_primary_table":table,"evidence":str(target.relative_to(ROOT))},indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--evaluation-id",default="original_textcnn_01");parser.add_argument("--format",default="fp32");parser.add_argument("--plan");parser.add_argument("--independent-check");parser.add_argument("--audit-run-id");parser.add_argument("--common-supervised-r2",action="store_true");args=parser.parse_args()
    if args.common_supervised_r2:
        if not args.plan:parser.error("--common-supervised-r2 requires --plan")
        common_selected(args.plan)
    elif args.audit_run_id:
        if not args.plan:parser.error("--audit-run-id requires --plan")
        audit_selected(args.audit_run_id,args.plan)
    else:check(args.evaluation_id,args.format,args.plan,args.independent_check)
