"""Independent structural, complete-tail, identity and overlap checks for CNN data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cnn_prepare import digest, rows, json_write, prepare_byte


def check_split(path, representation, np):
    docs = list(rows(path/"documents.jsonl"))
    data = {k:np.load(path/(k+".npy"),mmap_mode="r",allow_pickle=False) for k in ("ids","masks","labels","doc_indices","starts","content_start","document_content_length")}
    manifest = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["artifact_sha256"].items():
        assert digest(path/name) == expected, f"Artifact changed: {path/name}"
    ids, masks = data["ids"], data["masks"]
    assert ids.shape == masks.shape and ids.ndim == 2
    assert ids.shape[1] == (256 if representation == "token" else 1024)
    assert len(data["starts"]) == len(docs)+1 and data["starts"][0] == 0 and data["starts"][-1] == len(ids)
    assert len({d["document_id"] for d in docs}) == len(docs)
    assert bool(np.all((masks == 0)|(masks == 1)))
    assert bool(np.all(np.diff(masks.astype(np.int16),axis=1) <= 0)), "Mask has valid entries after padding"
    assert bool(np.all(ids[masks == 0] == 0)), "Padding IDs differ"
    assert bool(np.all(ids >= 0)) and bool(np.all(ids < (30522 if representation == "token" else 257)))
    assert bool(np.all((data["labels"] == 0)|(data["labels"] == 1)))
    complete_tails, overlap_checks = 0, 0
    for i,d in enumerate(docs):
        a,b = map(int,data["starts"][i:i+2]); assert b>a and b-a == d["window_count"]
        assert bool(np.all(data["doc_indices"][a:b] == i))
        assert bool(np.all(data["document_content_length"][a:b] == d["content_length"]))
        starts = data["content_start"][a:b]
        valid = masks[a:b].sum(axis=1).astype(np.int64)
        if representation == "token":
            assert bool(np.all(ids[a:b,0] == 101))
            assert all(int(ids[j,int(masks[j].sum())-1]) == 102 for j in range(a,b))
            valid -= 2
        assert starts[0] == 0
        assert int(starts[-1]+valid[-1]) == d["content_length"]
        assert all(int(starts[j]) <= int(starts[j-1]+valid[j-1]) for j in range(1,b-a))
        if d["label"] == 0: assert bool(np.all(data["labels"][a:b] == 0))
        else: assert bool(np.any(data["labels"][a:b] == 1)), "Attack has no labeled windows"
        reconstructed = []
        for j in range(b-a):
            if j:
                overlap = int(starts[j-1]+valid[j-1]-starts[j]); assert overlap == (64 if representation == "token" else 256)
                previous = ids[a+j-1,1:1+valid[j-1]] if representation == "token" else ids[a+j-1,:valid[j-1]]
                current = ids[a+j,1:1+valid[j]] if representation == "token" else ids[a+j,:valid[j]]
                assert np.array_equal(previous[-overlap:],current[:overlap]), "Adjacent window content differs"
                overlap_checks += 1
            if representation == "byte":
                skip = 0 if j == 0 else 256
                reconstructed.extend((ids[a+j,skip:valid[j]]-1).tolist())
        if representation == "byte":
            raw = bytes(reconstructed)
            assert len(raw) == d["raw_utf8_bytes"]
            assert hashlib.sha256(raw).hexdigest() == d["text_sha256"], "Full raw-document reconstruction differs"
            raw.decode("utf-8",errors="strict")
        complete_tails += 1
    return {"documents":len(docs),"windows":len(ids),"complete_tail_checks":complete_tails,"adjacent_overlap_checks":overlap_checks,"artifact_identities_verified":len(manifest["artifact_sha256"]),"source_groups":{d["source_group_id"] for d in docs},"document_identity":[(d["document_id"],d["label"],d["text_sha256"]) for d in docs]}


def byte_unit_checks(np):
    class Guard:
        def check(self): pass
    # Span offsets are Unicode characters, byte windows are actual UTF-8 units.
    text = "é"*510 + "ATTACK" + "漢"*450
    doc = {"document_id":"unicode", "source_group_id":"unit", "label":1, "text":text,"injection_spans":[[510,516]]}
    arrays,meta = prepare_byte([doc],Guard(),np)
    assert arrays["content_start"].tolist() == [0,768,1536]
    assert arrays["labels"].tolist() == [1,1,0], "Byte/Unicode span mismatch"
    assert arrays["document_content_length"].tolist() == [2376]*3
    benign = {**doc,"label":0,"injection_spans":[]}
    b,_ = prepare_byte([benign],Guard(),np); assert not b["labels"].any()
    empty = {**benign,"text":""}; e,m = prepare_byte([empty],Guard(),np)
    assert e["ids"].shape == (1,1024) and not e["masks"].any() and m[0]["window_count"] == 1
    return {"unicode_byte_span_crosses_boundary":True,"benign_windows_zero":True,"empty_document_one_masked_window":True}


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations",nargs="+",choices=("token","byte"),default=["token","byte"])
    parser.add_argument("--output", help="Fresh report filename under prepared; default automatically preserves earlier checks")
    args = parser.parse_args()
    import numpy as np
    prepared = ROOT/"results/cnn_study_v1/prepared"
    protocol = json.loads((prepared/"data_protocol.json").read_text(encoding="utf-8"))
    assert digest(prepared/"split_membership.json") == protocol["split_membership_sha256"]
    assert digest(prepared/"diagnostics_raw.jsonl") == protocol["diagnostics_raw_sha256"]
    for info in protocol["source_files"].values(): assert digest(ROOT/info["path"]) == info["sha256"]
    normalized_probes = {" ".join(d["text"].split()) for d in rows(prepared/"diagnostics_raw.jsonl")}
    normalized_overlap = {s:sum(" ".join(d["text"].split()) in normalized_probes for d in rows(ROOT/info["path"])) for s,info in protocol["source_files"].items()}
    assert not any(normalized_overlap.values()), "Diagnostic normalized exact text overlaps permitted source data"
    checks,identities = {},{}
    for rep in args.representations:
        repchecks = {}
        for path in sorted((prepared/rep).iterdir()):
            if not path.is_dir(): continue
            repchecks[path.name] = check_split(path,rep,np)
        for i,a in enumerate(("fit","inner_calibration","inner_selection")):
            for b in ("fit","inner_calibration","inner_selection")[i+1:]: assert not repchecks[a]["source_groups"] & repchecks[b]["source_groups"]
        full = repchecks["full_train"]["source_groups"]
        assert set.union(*(repchecks[k]["source_groups"] for k in ("fit","inner_calibration","inner_selection"))) == full
        assert repchecks["screen_fit"]["source_groups"] <= repchecks["fit"]["source_groups"]
        for a in ("fit","inner_calibration","inner_selection","search_temporary_threshold","search_candidate_scoring"):
            assert not repchecks[a]["source_groups"] & repchecks["diagnostics"]["source_groups"]
        assert not full & (repchecks["search_temporary_threshold"]["source_groups"]|repchecks["search_candidate_scoring"]["source_groups"])
        identities[rep] = {k:v["document_identity"] for k,v in repchecks.items()}
        checks[rep] = {k:{kk:vv for kk,vv in v.items() if kk not in ("source_groups","document_identity")} for k,v in repchecks.items()}
    if len(identities) == 2: assert identities["token"] == identities["byte"], "Representations process different raw documents"
    result = {"passed":True,"closed_partitions_opened":False,"source_group_folds_disjoint":True,"diagnostic_overlap_audit":protocol["diagnostic_overlap"],"diagnostic_whitespace_normalized_exact_text_overlap":normalized_overlap,"representation_raw_document_identity_equal": len(identities) == 2,"byte_unit_checks":byte_unit_checks(np),"checks":checks,"checker_source_sha256":digest(__file__),"python":sys.version,"numpy":np.__version__,"wall_seconds":time.monotonic()-started}
    stem = "check_"+"_".join(args.representations)
    path = prepared/(args.output or (stem+".json"))
    if args.output:
        if path.exists() or not path.resolve().is_relative_to(prepared.resolve()): raise ValueError("Use a fresh report under prepared")
    else:
        attempt = 1
        while path.exists():
            attempt += 1; path = prepared/(stem+f"_{attempt:03d}.json")
    json_write(path,result)
    print(json.dumps({"passed":True,"report":str(path),"checks":checks},indent=2))


if __name__ == "__main__": main()
