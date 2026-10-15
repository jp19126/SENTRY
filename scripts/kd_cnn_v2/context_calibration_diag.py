"""Development-only RF13/RF127 calibration and context diagnostic from saved rows.

Matched selection-FP thresholds are post hoc descriptive ROC points. They do
not replace the benign-only calibration or the fixed scientific selection rule.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from analysis import auc, paired
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from gate_eval import fit_threshold

MODELS={
    "rf13_selected_terminal":"rf13_richer_expanded_s42_dev",
    "rf127_selected_step1168":"richer_expanded_s42_dev",
}
ALPHA=.01
SEED=20261015


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def quantiles(values):
    values=np.asarray(values,dtype=np.float64)
    if not len(values):return None
    return {name:float(np.quantile(values,q)) for name,q in
            (("min",0.),("p10",.1),("median",.5),("p90",.9),("p95",.95),("p99",.99),("max",1.))}


def check_rows(values,prepared,role):
    if len(values)!=len(prepared):raise ValueError("Development role count differs")
    for scored,doc in zip(values,prepared):
        if scored.get("construction_role")!=role:
            raise ValueError("Unexpected score construction role")
        if any(scored.get(key)!=doc.get(key) for key in
               ("document_id","label","text_sha256","source_group_id","payload_group_id",
                "context_ancestry_group_id","hard_benign","content_tokens")):
            raise ValueError("Score/prepared membership or ancestry differs")
        if not scored.get("window_margins") or not math.isfinite(scored["margin"]):
            raise ValueError("Missing or nonfinite score")
        if scored["margin"]!=max(scored["window_margins"]) or len(scored["window_margins"])!=doc["windows"]:
            raise ValueError("Score/document max-window association differs")
    if role=="new_development_calibration" and any(row["label"]!=0 for row in values):
        raise ValueError("Development calibration contains attack")


def counts(values,threshold):
    positive=[row for row in values if row["label"]==1]
    benign=[row for row in values if row["label"]==0]
    tp=sum(row["margin"]>threshold for row in positive)
    fp=sum(row["margin"]>threshold for row in benign)
    return {"documents":len(values),"attacks":len(positive),"benign":len(benign),
            "tp":tp,"fn":len(positive)-tp,"fp":fp,"tn":len(benign)-fp,
            "recall":tp/len(positive) if positive else None,
            "fpr":fp/len(benign) if benign else None}


def stratum(values,threshold,key):
    groups={}
    for name in sorted({str(key(row)) for row in values}):
        subset=[row for row in values if str(key(row))==name]
        groups[name]=counts(subset,threshold)
    return groups


def length_bin(row):
    n=int(row["content_tokens"])
    if n<=128:return "0_128"
    if n<=254:return "129_254"
    if n<=508:return "255_508"
    return "509_plus"


def selection_fp_budget(values,budget):
    """Lowest selection threshold that allows at most budget benign FPs."""
    benign=sorted((row["margin"] for row in values if row["label"]==0),reverse=True)
    if not 0<=budget<len(benign):raise ValueError("Invalid descriptive FP budget")
    threshold=float(benign[budget])
    outcome=counts(values,threshold)
    if outcome["fp"]>budget:raise ValueError("Descriptive FP budget exceeded")
    return {"selection_label_fitted_threshold":threshold,"budget_fp":budget,
            "posthoc_selection_outcome":outcome,
            "use":"Descriptive ROC comparison only; invalid as a deployment/calibration threshold"}


def characterize(selection,calibration,threshold):
    benign=[row for row in selection if row["label"]==0]
    attack=[row for row in selection if row["label"]==1]
    hard=[row for row in benign if row.get("hard_benign")]
    plain=[row for row in benign if not row.get("hard_benign")]
    cal_hard=[row for row in calibration if row.get("hard_benign")]
    cal_plain=[row for row in calibration if not row.get("hard_benign")]
    false_positive=[row for row in benign if row["margin"]>threshold]
    positive_by_source={row["source_group_id"]:row for row in attack}
    fp_pairs=[(row,positive_by_source.get(row["source_group_id"])) for row in false_positive]
    return {
        "selection_auc":auc([row["margin"] for row in selection],[row["label"] for row in selection]),
        "selection_overall":counts(selection,threshold),
        "selection_hard_benign":counts(hard,threshold),
        "selection_other_benign":counts(plain,threshold),
        "calibration_all":counts(calibration,threshold),
        "calibration_hard_benign":counts(cal_hard,threshold),
        "calibration_other_benign":counts(cal_plain,threshold),
        "benign_score_distributions":{
            "calibration_hard":quantiles([row["margin"] for row in cal_hard]),
            "selection_hard":quantiles([row["margin"] for row in hard]),
            "calibration_other":quantiles([row["margin"] for row in cal_plain]),
            "selection_other":quantiles([row["margin"] for row in plain])},
        "source":{"source_dataset":stratum(selection,threshold,lambda row:row.get("source_dataset")),
                  "selection_source_groups":len({row["source_group_id"] for row in selection}),
                  "calibration_source_groups":len({row["source_group_id"] for row in calibration}),
                  "false_positive_source_groups":len({row["source_group_id"] for row in false_positive}),
                  "false_positives_with_paired_attack":sum(other is not None for _,other in fp_pairs),
                  "paired_attack_detected_for_false_positive":sum(other is not None and other["margin"]>threshold for _,other in fp_pairs)},
        "hard_discussion_family":stratum(hard,threshold,lambda row:row.get("context_ancestry_group_id")),
        "attack_payload_family":stratum(attack,threshold,lambda row:row.get("payload_group_id")),
        "attack_presentation":stratum(attack,threshold,lambda row:row.get("presentation")),
        "attack_injection_position":stratum(attack,threshold,lambda row:row.get("injection_position")),
        "windows":stratum(selection,threshold,lambda row:"1" if len(row["window_margins"])==1 else "2_plus"),
        "content_tokens":stratum(selection,threshold,length_bin),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root",type=Path,required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args();root=args.canonical_root.resolve()
    output=Path(args.out)
    output=(output if output.is_absolute() else root/output).resolve()
    if not output.is_relative_to((root/"results/kd_cnn_v2").resolve()) or output.exists():
        raise ValueError("Output must be a new task-owned diagnostic")
    prepared=root/"results/dual_track_v1/shared/data/prepared"
    selection_prepared=rows(prepared/"new_selection/documents.jsonl")
    calibration_prepared=rows(prepared/"new_development_calibration/documents.jsonl")
    if {row["source_group_id"] for row in selection_prepared}&{row["source_group_id"] for row in calibration_prepared}:
        raise ValueError("Development source groups cross roles")
    inputs={};summaries={}
    for name,run in MODELS.items():
        directory=root/"results/kd_cnn_v2/runs"/run
        complete=read_json(directory/"complete.json")
        run_record=read_json(directory/"run.json")
        if not complete.get("complete") or complete["run_id"]!=run or run_record["association_sha256"]!=complete["association_sha256"]:
            raise ValueError("Run qualification/association differs")
        selection=rows(directory/"selection_scores.jsonl")
        calibration=rows(directory/"calibration_scores.jsonl")
        check_rows(selection,selection_prepared,"new_selection")
        check_rows(calibration,calibration_prepared,"new_development_calibration")
        threshold=fit_threshold([row["margin"] for row in calibration],ALPHA)
        if abs(threshold-complete["best"]["development"]["0.01"]["threshold_margin"])>1e-4:
            raise ValueError("Independent benign threshold differs from saved selected run")
        summary=characterize(selection,calibration,threshold)
        expected=complete["best"]["development"]["0.01"]
        if (summary["selection_overall"]["tp"],summary["selection_overall"]["fp"])!=(expected["tp"],expected["fp"]):
            raise ValueError("Saved selected development counts differ")
        inputs[name]=(selection,calibration)
        summaries[name]={"run_id":run,"selected_step":complete["selected_step"],
            "terminal_step":complete["steps_completed"],
            "selected_equals_terminal":complete["selected_step"]==complete["steps_completed"],
            "selected_checkpoint_sha256":complete["checkpoint_sha256"],
            "selected_score_sha256":{"new_selection":sha(directory/"selection_scores.jsonl"),
                                     "new_development_calibration":sha(directory/"calibration_scores.jsonl")},
            "selected_calibration_threshold":float(threshold),
            "selected_at_benign_calibrated_alpha_0_01":summary}
    for name,(selection,_) in inputs.items():
        summaries[name]["selected_posthoc_matched_selection_fp_budgets"]={str(budget):selection_fp_budget(selection,budget)
            for budget in (3,29)}
    left,left_cal=inputs["rf13_selected_terminal"]
    right,right_cal=inputs["rf127_selected_step1168"]
    comparison=paired(left,right,left_cal,right_cal,repeats=1000,seed=SEED)
    teacher_root=root/"results/kd_cnn_v2/development/anchors"
    teacher_selection=rows(teacher_root/"teacher_common_new_selection.jsonl")
    teacher_calibration=rows(teacher_root/"teacher_common_new_development_calibration.jsonl")
    if [row["document_id"] for row in teacher_selection]!=[row["document_id"] for row in left]:
        raise ValueError("Teacher anchor selection membership differs")
    if [row["document_id"] for row in teacher_calibration]!=[row["document_id"] for row in left_cal]:
        raise ValueError("Teacher anchor calibration membership differs")
    teacher_threshold=fit_threshold([row["margin"] for row in teacher_calibration],ALPHA)
    terminal_history=read_json(root/"results/kd_cnn_v2/runs/richer_expanded_s42_dev/history.json")[-1]
    if terminal_history["step"]!=summaries["rf127_selected_step1168"]["terminal_step"]:
        raise ValueError("RF127 terminal history step differs")
    terminal=terminal_history["development"]
    output.parent.mkdir(parents=True,exist_ok=True)
    payload={"schema":"kd_cnn_v2_rf_context_calibration_development_v1","complete":True,
        "scope":"Saved new_selection and new_development_calibration only; no new scoring, no confirmation/final access, no selection-rule change.",
        "source_sha256":sha(Path(__file__)),
        "prepared_manifest_sha256":{"new_selection":sha(prepared/"new_selection/manifest.json"),
                                    "new_development_calibration":sha(prepared/"new_development_calibration/manifest.json")},
        "alpha":ALPHA,"threshold_policy":"benign-only conservative order statistic, strict margin > threshold",
        "models":summaries,"selected_rf13_minus_selected_rf127_paired":comparison,
        "teacher_anchor":{"selection_score_sha256":sha(teacher_root/"teacher_common_new_selection.jsonl"),
                          "calibration_score_sha256":sha(teacher_root/"teacher_common_new_development_calibration.jsonl"),
                          "threshold":teacher_threshold,"calibrated_selection":counts(teacher_selection,teacher_threshold),
                          "posthoc_matched_selection_fp_budgets":{str(budget):selection_fp_budget(teacher_selection,budget)
                                                                 for budget in (3,29)}},
        "rf127_terminal_summary_only":{"step":terminal_history["step"],
            "auc":terminal["auc"],"benign_calibrated_alpha_0_01":terminal["0.01"],
            "history_sha256":sha(root/"results/kd_cnn_v2/runs/richer_expanded_s42_dev/history.json"),
            "limitation":"No saved terminal per-document score rows; no terminal strata, paired row test, or matched selection-FP ROC inferred."},
        "interpretation_limits":"Matched selection-FP thresholds use selection labels and are post hoc ROC descriptions, not deployable or model-selection thresholds. 500 benign calibration docs resolve 1% in about five tail examples; 200 selection benign docs yield 0.5% FPR steps. Hard discussion and 12 attack payload families have small dependent counts. The paired source/payload/calibration sensitivity intervals are separate, not joint coverage. All data are constructed one-author development material."}
    output.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"out":str(output),"sha256":sha(output),
        "rf13":summaries["rf13_selected_terminal"]["selected_at_benign_calibrated_alpha_0_01"]["selection_overall"],
        "rf127":summaries["rf127_selected_step1168"]["selected_at_benign_calibrated_alpha_0_01"]["selection_overall"]}))


if __name__=="__main__":main()
