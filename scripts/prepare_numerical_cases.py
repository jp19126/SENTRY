"""Freeze a prospective Goal 4 numerical-replay input set from development evidence."""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gate_eval import load_config, window_document

DEST = ROOT / "results/goal4/numerical_bridge/development_cases_v1/selection.json"
MANIFEST = ROOT / "results/goal4/numerical_bridge/manifest.json"
PARTITIONS = ("search_temporary_threshold", "search_candidate_scoring")
SCORES = ROOT / "results/goal3/qat_w8_a8"
CASE1 = "aeslc:train:panus-s_inbox_1.subject:clean"
FIELDS = ("input_ids", "attention_mask", "token_type_ids")

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]

def qtile(values, q):
    values = sorted(values)
    return values[min(len(values)-1, max(0, round((len(values)-1)*q)))]

def main():
    if DEST.exists():
        old = read_json(DEST)
        if old.get("case_count") != 32 or len(old.get("cases", [])) != 32:
            raise RuntimeError(f"Existing destination is malformed: {DEST}")
        print(json.dumps({"event":"existing_selection_reused","path":str(DEST),"case_count":32}))
        return

    cfg = load_config()
    if read_json(MANIFEST).get("case") != CASE1:
        raise ValueError("The existing bridge manifest does not identify the required first case.")
    tokdir = ROOT / "checkpoints/quantized/qat_w8_a8"
    if not tokdir.is_dir():
        raise FileNotFoundError(f"Local tokenizer checkpoint missing: {tokdir}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokdir, use_fast=True, local_files_only=True)

    raw, scores = {}, {}
    for split in PARTITIONS:
        raw[split] = read_jsonl(ROOT / "data/prepared" / f"{split}.jsonl")
        sr = read_jsonl(SCORES / f"{split}_documents.jsonl")
        if len(sr) != len(raw[split]):
            raise ValueError(f"Saved W8-QAT scoring coverage mismatch for {split}")
        for row in sr:
            if row["document_id"] in scores:
                raise ValueError(f"Duplicate document score {row['document_id']}")
            scores[row["document_id"]] = row

    records = {}
    for split, rows in raw.items():
        for row in rows:
            did = row["document_id"]
            if row.get("search_subset") != ("temporary_threshold" if split == "search_temporary_threshold" else "candidate_scoring") or did not in scores:
                raise ValueError(f"Invalid partition or missing historical score for {did}")
            sc = scores[did]
            if int(row["label"]) != int(sc["label"]):
                raise ValueError(f"Label differs in saved score record: {did}")
            wins = window_document(tok, row["text"], did, row["label"], row.get("injection_spans"), cfg)
            frozen = []
            for win in wins:
                fields = {k: win[k] for k in FIELDS if k in win}
                pad = tok.pad([fields], padding="max_length",
                              max_length=cfg["windowing"]["primary_length"], return_tensors=None)
                frozen.append({k: [int(x) for x in pad[k][0]] for k in fields})
            if any(len(w["input_ids"]) != cfg["windowing"]["primary_length"] for w in frozen):
                raise ValueError(f"Window has unexpected length: {did}")
            records[did] = {
                "row": row, "split": split, "score": float(sc["risk_score"]),
                "tokens": int(sc["content_token_count"]), "windows": frozen,
                "pads": [cfg["windowing"]["primary_length"] - sum(w["attention_mask"]) for w in frozen],
            }

    threshold = float(read_json(SCORES / "candidate.json")["metrics"]["primary"]["threshold"])
    if threshold != 0.004557917360216379:
        raise ValueError("Saved W8-QAT operating threshold differs from recorded bridge identity.")
    thresh = [x for x in records.values() if x["split"] == PARTITIONS[0] and x["row"]["label"] == 0]
    cand = [x for x in records.values() if x["split"] == PARTITIONS[1]]
    chosen, used_ids, used_groups = [], set(), set()

    def available(pool, label=None, pred=lambda x: True):
        return [x for x in pool if x["row"]["document_id"] not in used_ids
                and x["row"].get("source_group_id", x["row"]["document_id"]) not in used_groups
                and (label is None or int(x["row"]["label"]) == label) and pred(x)]

    def choose(pool, reason, label=None, pred=lambda x: True, key=None):
        av = available(pool, label, pred)
        if not av:
            raise ValueError(f"Empty predeclared selection stratum: {reason}")
        x = min(av, key=key or (lambda z: z["row"]["document_id"]))
        did = x["row"]["document_id"]
        used_ids.add(did); used_groups.add(x["row"].get("source_group_id", did))
        chosen.append((x, reason))
        return x

    first = records[CASE1]
    did = first["row"]["document_id"]
    used_ids.add(did); used_groups.add(first["row"].get("source_group_id", did))
    chosen.append((first, "case1 existing bridge input; retain the known attention/A8 tie case"))
    choose(thresh, "search_threshold benign nearest historical W8-QAT threshold",
           key=lambda x:(abs(x["score"]-threshold),x["row"]["document_id"]))
    t25 = qtile([x["score"] for x in thresh], .25)
    choose(thresh, "search_threshold benign lower-score range",
           key=lambda x:(abs(x["score"]-t25),x["row"]["document_id"]))

    benign = [x for x in cand if int(x["row"]["label"]) == 0]
    near = lambda x:(abs(x["score"]-threshold),x["row"]["document_id"])
    choose(benign, "candidate_scoring benign nearest operating-point neighbor", key=near)
    choose(benign, "candidate_scoring benign next operating-point neighbor", key=near)
    for title, pred in (
        ("short <128 content tokens", lambda x:x["tokens"]<128),
        ("single-window 128-255 content tokens", lambda x:128<=x["tokens"]<=255 and len(x["windows"])==1),
        ("long 256-511 content tokens", lambda x:256<=x["tokens"]<512),
        ("very long >=512 content tokens", lambda x:x["tokens"]>=512),
    ):
        choose(benign, f"candidate_scoring benign length coverage: {title}", pred=pred, key=near)
    choose(benign, "candidate_scoring benign high-padding single-window input",
           pred=lambda x:len(x["windows"])==1, key=lambda x:(-x["pads"][0],x["row"]["document_id"]))
    choose(benign, "candidate_scoring benign low-padding single-window input",
           pred=lambda x:len(x["windows"])==1, key=lambda x:(x["pads"][0],x["row"]["document_id"]))
    for q in (.20,.50,.80):
        target=qtile([x["score"] for x in benign],q)
        choose(benign,f"candidate_scoring benign historical score quantile {q:.2f}",
               key=lambda x:(abs(x["score"]-target),x["row"]["document_id"]))
    while sum(int(x["row"]["label"])==0 for x,_ in chosen)<16:
        choose(benign,"candidate_scoring benign deterministic score/length fill",
               key=lambda x:(abs(x["score"]-threshold),-x["tokens"],x["row"]["document_id"]))

    attacks = [x for x in cand if int(x["row"]["label"]) == 1]
    for _ in range(4):
        choose(attacks,"candidate_scoring attacked nearest operating-point range",key=near)
    for title,pred in (
        ("short <128 content tokens",lambda x:x["tokens"]<128),
        ("single-window 128-255 content tokens",lambda x:128<=x["tokens"]<=255 and len(x["windows"])==1),
        ("long 256-511 content tokens",lambda x:256<=x["tokens"]<512),
        ("very long >=512 content tokens",lambda x:x["tokens"]>=512),
    ):
        choose(attacks,f"candidate_scoring attacked length coverage: {title}",pred=pred,key=near)
    for q in (.10,.35,.65,.90):
        target=qtile([x["score"] for x in attacks],q)
        choose(attacks,f"candidate_scoring attacked historical score quantile {q:.2f}",
               key=lambda x:(abs(x["score"]-target),x["row"]["document_id"]))
    types=sorted({str(x["row"].get("attack_type")) for x in attacks if x["row"].get("attack_type") is not None})
    for typ in types:
        if sum(int(x["row"]["label"])==1 for x,_ in chosen)>=16: break
        choose(attacks,f"candidate_scoring attacked injection type {typ}",
               pred=lambda x,typ=typ:str(x["row"].get("attack_type"))==typ,key=near)
    while sum(int(x["row"]["label"])==1 for x,_ in chosen)<16:
        choose(attacks,"candidate_scoring attacked deterministic score/length fill",
               key=lambda x:(abs(x["score"]-threshold),-x["tokens"],x["row"]["document_id"]))

    if len(chosen)!=32 or Counter(int(x["row"]["label"]) for x,_ in chosen)!=Counter({0:16,1:16}):
        raise ValueError("Selection did not satisfy frozen 32-case 16/16 quotas.")
    cases=[]
    for i,(x,reason) in enumerate(chosen,1):
        row=x["row"]
        cases.append({"case_index":i,"document_id":row["document_id"],
            "source_group_id":row.get("source_group_id"),"source_ids":row.get("source_ids",[]),
            "partition":x["split"],"label":int(row["label"]),
            "attack_type":row.get("attack_type"),"attack_template_id":row.get("attack_template_id"),
            "injection_position":row.get("injection_position"),
            "historical_w8_qat_risk":x["score"],"historical_threshold":threshold,
            "historical_threshold_distance":abs(x["score"]-threshold),
            "content_token_count":x["tokens"],"window_count":len(x["windows"]),
            "padding_tokens_per_window":x["pads"],"selection_reason":reason,"windows":x["windows"]})
    result={"schema":"sentry_goal4_numerical_development_selection_v1",
        "selection_status":"prospectively frozen before new numerical replay","case_count":32,
        "sampling":{"allowed_partitions":list(PARTITIONS),"strata":{"benign":16,"attacked":16},
            "unit":"unique source_group_id","first_case":CASE1,
            "historical_operating_threshold":threshold,
            "selection_sources":["prepared records from approved search partitions",
                "saved uniform W8-QAT scores in results/goal3/qat_w8_a8",
                "existing numerical bridge manifest for case 1"],
            "selection_rule":"fixed class quotas, threshold proximity, historical score quantiles, padding and token-length strata; deterministic document-ID tie breaks",
            "new_numerical_discrepancies_used_for_selection":False},
        "tokenizer":{"model":cfg["models"]["detector"]["id"],
            "revision":cfg["models"]["detector"]["tokenizer_revision"],
            "local_source":"checkpoints/quantized/qat_w8_a8; local_files_only=true",
            "fast":bool(tok.is_fast),"padding_length":cfg["windowing"]["primary_length"],
            "overlap_content_tokens":cfg["windowing"]["content_token_overlap"]},
        "input_scope":"document text only; labels and source IDs are metadata, not model inputs",
        "inference_performed":False,"cases":cases}
    DEST.parent.mkdir(parents=True,exist_ok=True)
    DEST.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"event":"selection_written","path":str(DEST),"case_count":32,
        "class_counts":dict(Counter(c["label"] for c in cases)),
        "partition_counts":dict(Counter(c["partition"] for c in cases)),
        "window_count_distribution":dict(Counter(str(c["window_count"]) for c in cases)),
        "maximum_content_tokens":max(c["content_token_count"] for c in cases),
        "frozen_token_windows":sum(c["window_count"] for c in cases)},indent=2))

if __name__=="__main__":
    main()
