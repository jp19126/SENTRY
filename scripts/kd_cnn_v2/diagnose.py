"""Descriptive teacher/student context gaps on frozen development score rows.

Reads only task-owned new_selection and new_development_calibration margins.
Does not score models, tune candidates, or access any confirmation role.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import numpy as np
from analysis import auc
from gate_eval import fit_threshold


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inside(root, value):
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to((root / "results/kd_cnn_v2").resolve()):
        raise ValueError("Score/diagnostic path outside task-owned results")
    return path


def scored_rows(root, value, role, expected_kind):
    path = inside(root, value)
    complete_path = path.with_suffix(".complete.json")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if not complete.get("complete") or complete.get("stage") != "development" or complete.get("role") != role:
        raise ValueError("Expected completed development-only scoring association")
    if complete.get("target",{}).get("kind") != expected_kind or complete["scores_sha256"] != sha(path):
        raise ValueError("Target type or score bytes differ")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != complete["documents"] or len(rows) != len({r["document_id"] for r in rows}):
        raise ValueError("Score count or document identity differs")
    for row in rows:
        if row["stage"] != "development" or row["role"] != role or row["target_id"] != complete["target_id"]:
            raise ValueError("Mixed score stage/role/target")
        windows = row["window_margins"]
        if not windows or not math.isfinite(row["margin"]) or row["margin"] != max(windows):
            raise ValueError("Incomplete or nonfinite document max score")
        if row["prepared_manifest_sha256"] != complete["prepared_manifest_sha256"]:
            raise ValueError("Prepared data identity differs")
    if role == "new_development_calibration" and any(r["label"] != 0 for r in rows):
        raise ValueError("Development calibration contains positive label")
    return rows, complete


def match(left, right):
    if len(left) != len(right):
        raise ValueError("Paired row count differs")
    keys = ("document_id", "label", "source_group_id", "payload_group_id",
            "context_ancestry_group_id", "text_sha256", "content_tokens", "hard_benign")
    for a, b in zip(left, right):
        if any(a.get(k) != b.get(k) for k in keys):
            raise ValueError("Paired document ancestry or order differs")


def interval_count(rows, teacher, student, threshold_teacher, threshold_student, indices):
    selected = [rows[i] for i in indices]
    y = np.asarray([r["label"] for r in selected], dtype=int)
    td = np.asarray([teacher[i]["margin"] > threshold_teacher for i in indices], dtype=bool)
    sd = np.asarray([student[i]["margin"] > threshold_student for i in indices], dtype=bool)
    pos, benign = y == 1, y == 0
    teacher_correct, student_correct = td == y, sd == y
    return {
        "documents": len(indices), "positive": int(pos.sum()), "benign": int(benign.sum()),
        "teacher": {"tp": int(td[pos].sum()), "fn": int((~td[pos]).sum()),
                    "fp": int(td[benign].sum()), "tn": int((~td[benign]).sum()),
                    "recall": float(td[pos].mean()) if pos.any() else None,
                    "fpr": float(td[benign].mean()) if benign.any() else None},
        "student": {"tp": int(sd[pos].sum()), "fn": int((~sd[pos]).sum()),
                    "fp": int(sd[benign].sum()), "tn": int((~sd[benign]).sum()),
                    "recall": float(sd[pos].mean()) if pos.any() else None,
                    "fpr": float(sd[benign].mean()) if benign.any() else None},
        "paired": {"different_decisions": int((td != sd).sum()),
                   "both_wrong": int((~teacher_correct & ~student_correct).sum()),
                   "teacher_only_correct": int((teacher_correct & ~student_correct).sum()),
                   "student_only_correct": int((~teacher_correct & student_correct).sum())},
    }


def lengths(rows, decisions, label):
    out = {}
    for model_name, pred in decisions.items():
        failed = [int(r["content_tokens"]) for r, p in zip(rows, pred)
                  if r["label"] == label and int(p) != label]
        correct = [int(r["content_tokens"]) for r, p in zip(rows, pred)
                   if r["label"] == label and int(p) == label]
        out[model_name] = {"incorrect_n":len(failed), "correct_n":len(correct),
                           "incorrect_median_content_tokens":median(failed) if failed else None,
                           "correct_median_content_tokens":median(correct) if correct else None}
    return out


def diagnose(root, teacher_path, student_path, teacher_cal_path, student_cal_path):
    role, calibration_role = "new_selection", "new_development_calibration"
    teacher, teacher_meta = scored_rows(root,teacher_path,role,"teacher")
    student, student_meta = scored_rows(root,student_path,role,"cnn_fp32")
    teacher_cal, teacher_cal_meta = scored_rows(root,teacher_cal_path,calibration_role,"teacher")
    student_cal, student_cal_meta = scored_rows(root,student_cal_path,calibration_role,"cnn_fp32")
    match(teacher,student); match(teacher_cal,student_cal)
    if teacher_meta["target_id"] != teacher_cal_meta["target_id"] or student_meta["target_id"] != student_cal_meta["target_id"]:
        raise ValueError("Selection/calibration model targets differ")
    if teacher_meta["prepared_manifest_sha256"] != student_meta["prepared_manifest_sha256"] or teacher_cal_meta["prepared_manifest_sha256"] != student_cal_meta["prepared_manifest_sha256"]:
        raise ValueError("Paired prepared panel differs")
    threshold_teacher = fit_threshold([r["margin"] for r in teacher_cal],.01)
    threshold_student = fit_threshold([r["margin"] for r in student_cal],.01)
    all_indices = list(range(len(teacher)))
    strata = {
        "attack": [i for i,r in enumerate(teacher) if r["label"] == 1],
        "hard_benign": [i for i,r in enumerate(teacher) if r["label"] == 0 and r.get("hard_benign")],
        "other_benign": [i for i,r in enumerate(teacher) if r["label"] == 0 and not r.get("hard_benign")],
        "one_window": [i for i,r in enumerate(teacher) if len(r["window_margins"]) == 1],
        "two_or_more_windows": [i for i,r in enumerate(teacher) if len(r["window_margins"]) >= 2],
    }
    for low, high, name in ((0,128,"content_tokens_0_128"),(129,254,"content_tokens_129_254"),
                            (255,508,"content_tokens_255_508"),(509,10**9,"content_tokens_509_plus")):
        strata[name] = [i for i,r in enumerate(teacher) if low <= int(r["content_tokens"]) <= high]
    decisions = {"teacher": [r["margin"] > threshold_teacher for r in teacher],
                 "student": [r["margin"] > threshold_student for r in student]}
    overall = interval_count(teacher,teacher,student,threshold_teacher,threshold_student,all_indices)
    return {
        "schema":"kd_cnn_v2_development_context_diagnostic_v1",
        "scope":"New development selection only; 1% benign-calibrated operating point, descriptive counts, no candidate optimization or confirmation access.",
        "teacher_target":teacher_meta["target_id"],"student_target":student_meta["target_id"],
        "selection_manifest_sha256":teacher_meta["prepared_manifest_sha256"],
        "development_calibration_manifest_sha256":teacher_cal_meta["prepared_manifest_sha256"],
        "teacher_selection_scores_sha256":teacher_meta["scores_sha256"],
        "student_selection_scores_sha256":student_meta["scores_sha256"],
        "teacher_calibration_scores_sha256":teacher_cal_meta["scores_sha256"],
        "student_calibration_scores_sha256":student_cal_meta["scores_sha256"],
        "calibration_n":len(teacher_cal),"alpha":.01,
        "auc_descriptive":{"teacher":auc([r["margin"] for r in teacher],[r["label"] for r in teacher]),
                           "student":auc([r["margin"] for r in student],[r["label"] for r in student])},
        "overall":overall,
        "teacher_minus_student":{"recall":overall["teacher"]["recall"]-overall["student"]["recall"],
                                 "fpr":overall["teacher"]["fpr"]-overall["student"]["fpr"]},
        "strata":{name:interval_count(teacher,teacher,student,threshold_teacher,threshold_student,index)
                  for name,index in strata.items()},
        "content_length_error_medians":{"positive":lengths(teacher,decisions,1),
                                         "benign":lengths(teacher,decisions,0)},
        "interpretation_limit":"Source/payload clusters and constructed context constrain independence; small hard-benign and length strata are descriptive, with no confidence or natural-population claim. Teacher/student score scales are not directly subtracted.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root",type=Path,required=True)
    parser.add_argument("--teacher-scores",type=Path,required=True)
    parser.add_argument("--student-scores",type=Path,required=True)
    parser.add_argument("--teacher-calibration",type=Path,required=True)
    parser.add_argument("--student-calibration",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    root = args.canonical_root.resolve()
    output = inside(root,args.out)
    if output.exists():
        raise FileExistsError(output)
    result = diagnose(root,args.teacher_scores,args.student_scores,args.teacher_calibration,args.student_calibration)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"out":str(output),"teacher_minus_student":result["teacher_minus_student"]}))


if __name__ == "__main__":
    main()
