"""Recompute Goal R1's saved development anchors without model execution."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gate_eval import fit_threshold, evaluate_documents, load_config


def rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    config = load_config()
    names = ("search_temporary_threshold", "search_candidate_scoring")
    data = {n: rows(ROOT / "data/prepared" / (n + ".jsonl")) for n in names}
    groups = {n: {r["source_group_id"] for r in rr} for n, rr in data.items()}
    assert groups[names[0]].isdisjoint(groups[names[1]])
    train_groups = {r["source_group_id"] for r in rows(ROOT / "data/prepared/train.jsonl")}
    assert all(train_groups.isdisjoint(g) for g in groups.values())
    outputs = {}
    for name, rel in (("bert_floating", "results/goal2/epoch_3"),
                      ("bert_w8_qat", "results/goal3/qat_w8_a8")):
        saved = {}
        for split in names:
            docs = rows(ROOT / rel / (split + "_documents.jsonl"))
            windows = rows(ROOT / rel / (split + "_windows.jsonl"))
            lookup = {r["document_id"]: r for r in data[split]}
            assert len(docs) == len(lookup) == len({r["document_id"] for r in docs})
            assert {r["document_id"] for r in docs} == set(lookup)
            by_doc = {}
            for w in windows:
                by_doc.setdefault(w["document_id"], []).append(w)
            for d in docs:
                assert d["label"] == lookup[d["document_id"]]["label"]
                ww = by_doc[d["document_id"]]
                assert len(ww) == d["window_count"]
                assert max(w["risk_score"] for w in ww) == d["risk_score"]
                assert [w["window_index"] for w in ww] == list(range(len(ww)))
                assert ww[-1]["content_token_end"] == d["content_token_count"]
                assert ww[0]["content_token_start"] == 0
                for a, b in zip(ww, ww[1:]):
                    assert a["content_token_end"] - b["content_token_start"] == 64
                d.update({k: lookup[d["document_id"]].get(k) for k in
                          ("source_dataset", "source_group_id", "attack_type", "attack_template_id")})
            saved[split] = docs
        threshold = fit_threshold([r["risk_score"] for r in saved[names[0]] if r["label"] == 0], .01)
        def metrics(rr):
            return evaluate_documents([r["risk_score"] for r in rr], [r["label"] for r in rr], threshold)
        scoring = saved[names[1]]
        strata = {}
        for field in ("source_dataset", "is_long_document", "attack_type", "attack_template_id"):
            for value in sorted({str(r.get(field)) for r in scoring}):
                strata[field + ":" + value] = metrics([r for r in scoring if str(r.get(field)) == value])
        outputs[name] = {"source": rel, "threshold": threshold, "metrics": metrics(scoring),
                         "strata": strata, "coverage_windows": {n: sum(r["window_count"] for r in saved[n]) for n in names},
                         "semantics": "Historical native Torch CUDA FP32/SDPA; W8 uses fixed-scale emulated encoder linears. Not ordered C or accelerated INT8."}
    floor = outputs["bert_floating"]["metrics"]["recall"] - config["evaluation"]["max_recall_loss_absolute"]
    for v in outputs.values():
        v["feasible"] = v["metrics"]["recall"] >= floor and v["metrics"]["fpr"] <= .01
    record = {"scope": "Saved development data only; no inference or final partitions opened", "recall_floor": floor,
              "split_groups": {n: len(g) for n, g in groups.items()}, "train_groups": len(train_groups),
              "anchors": outputs,
              "coverage": {n: {key: dict(Counter(str(r.get(key)) for r in rr)) for key in
                                   ("source_dataset", "attack_type", "attack_template_id")} for n, rr in data.items()}}
    dest = ROOT / "results/goal_r1/anchor_audit.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"path": str(dest), "recall_floor": floor, "anchors": {k: v["metrics"] for k, v in outputs.items()}}))


if __name__ == "__main__":
    main()
