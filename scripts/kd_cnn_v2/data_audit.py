"""Score-free audit of frozen prospective data and expanded training input."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
import numpy as np
from transformers import AutoTokenizer
from data import ROLES, file_sha, load, rows, save, sha

def grams(text):
    words = re.findall(r"\w+", text.lower())
    return set(zip(*(words[i:] for i in range(4)))) if len(words) >= 4 else set()

def similarity(a, b):
    return len(a & b) / len(a | b) if a or b else 1.0

def audit(canonical, worktree):
    shared = canonical/"results/dual_track_v1/shared/data"
    protocol = load(shared/"protocol.json")
    protocol_sha = file_sha(shared/"protocol.json")
    old_protocol = load(canonical/"results/cnn_adoption_v1/revision_r2/data_protocol.json")
    old_source = set().union(*(old_protocol["source_role_membership"][r]
                               for r in old_protocol["source_role_membership"]))
    old_payload_texts = set()
    for role in ("fit", "development_selection", "development_calibration", "benign_calibration"):
        for row in rows(canonical/"results/cnn_adoption_v1/raw"/(role+".jsonl")):
            if row["label"] != 1: continue
            for a, b in row.get("injection_spans", []):
                old_payload_texts.add(re.sub(r"\s+", " ", row["text"][a:b]).strip().lower())
    old_grams = [(s, grams(s)) for s in old_payload_texts]
    tok = AutoTokenizer.from_pretrained(canonical/"checkpoints/floating/epoch_3", local_files_only=True, use_fast=True)
    role_summary, source_sets, payload_sets, context_sets = {}, {}, {}, {}
    max_old_match = (0.0, None, None)
    for role in ROLES:
        raw_path = shared/"raw"/(role+".jsonl")
        folder = shared/"prepared"/role
        manifest = load(folder/"manifest.json")
        if file_sha(raw_path) != protocol["raw_role_sha256"][role]: raise AssertionError("Raw role changed")
        if manifest["data_protocol_sha256"] != protocol_sha: raise AssertionError("Protocol mismatch")
        for name, expected in manifest["artifact_sha256"].items():
            if file_sha(folder/name) != expected: raise AssertionError(name)
        raw = list(rows(raw_path))
        docs = list(rows(folder/"documents.jsonl"))
        ids = np.load(folder/"ids.npy", mmap_mode="r", allow_pickle=False)
        masks = np.load(folder/"masks.npy", mmap_mode="r", allow_pickle=False)
        starts = np.load(folder/"starts.npy", mmap_mode="r", allow_pickle=False)
        content_starts = np.load(folder/"content_start.npy", mmap_mode="r", allow_pickle=False)
        lengths = np.load(folder/"document_content_length.npy", mmap_mode="r", allow_pickle=False)
        if len(raw) != len(docs) or len(raw) != manifest["documents"]: raise AssertionError("Document count")
        if ids.shape != masks.shape or ids.shape[1] != 256 or len(ids) != manifest["windows"]: raise AssertionError("Windows")
        if len(starts) != len(raw)+1 or starts[0] != 0 or starts[-1] != len(ids) or not np.all(np.diff(starts)>0): raise AssertionError("Starts")
        if np.any((masks != 0)&(masks != 1)): raise AssertionError("Nonbinary mask")
        source_sets[role] = {r["source_group_id"] for r in raw}
        payload_sets[role] = {r["payload_group_id"] for r in raw if r["payload_group_id"]}
        context_sets[role] = {r["context_ancestry_group_id"] for r in raw if r["context_ancestry_group_id"]}
        if source_sets[role] & old_source: raise AssertionError("Prior source group reused")
        by_family = defaultdict(set)
        for index, (r, d) in enumerate(zip(raw, docs)):
            if sha(r["text"]) != r["text_sha256"]: raise AssertionError("Text changed")
            for key, value in r.items():
                if key != "text" and d[key] != value: raise AssertionError((role,index,key))
            tokens = tok(r["text"], add_special_tokens=False, truncation=False,
                         return_attention_mask=False, return_token_type_ids=False, verbose=False)["input_ids"]
            lo, hi = int(starts[index]), int(starts[index+1])
            rebuilt = []
            for j in range(lo,hi):
                used = int(masks[j].sum())
                if not 2 <= used <= 256 or ids[j,0] != tok.cls_token_id or ids[j,used-1] != tok.sep_token_id:
                    raise AssertionError("Special tokens")
                if np.any(masks[j,:used] != 1) or np.any(masks[j,used:] != 0) or np.any(ids[j,used:] != tok.pad_token_id):
                    raise AssertionError("Mask/padding")
                piece = ids[j,1:used-1].tolist()
                if content_starts[j] != (j-lo)*190: raise AssertionError("Stride")
                if j == lo: rebuilt.extend(piece)
                else:
                    if rebuilt[-64:] != piece[:64]: raise AssertionError("Overlap")
                    rebuilt.extend(piece[64:])
            if rebuilt != tokens or lengths[index] != len(tokens) or d["content_tokens"] != len(tokens):
                raise AssertionError("Tail reconstruction")
            if r["label"] == 1:
                a,b = r["injection_spans"][0]
                candidate = re.sub(r"\s+", " ", r["text"][a:b]).strip().lower()
                by_family[r["payload_group_id"]].add(candidate)
        for family, variants in by_family.items():
            for candidate in variants:
                g = grams(candidate)
                for old_text, old_g in old_grams:
                    score = similarity(g, old_g)
                    if score > max_old_match[0]:
                        max_old_match = (score, family, sha(old_text))
        role_summary[role] = {"documents": len(raw), "windows": len(ids),
                              "labels": dict(sorted({str(x):sum(r["label"]==x for r in raw) for x in (0,1)}.items())),
                              "source_groups": len(source_sets[role]), "payload_families": len(payload_sets[role]),
                              "context_families": len(context_sets[role]), "all_document_tails_reconstructed": True,
                              "manifest_sha256": file_sha(folder/"manifest.json")}
    for i,left in enumerate(ROLES):
        for right in ROLES[i+1:]:
            if source_sets[left]&source_sets[right] or payload_sets[left]&payload_sets[right] or context_sets[left]&context_sets[right]:
                raise AssertionError("Cross-role ancestry overlap")
    expanded = canonical/"results/kd_cnn_v2/prepared/expanded_fit"
    em = load(expanded/"manifest.json")
    base = canonical/"results/cnn_adoption_v1/prepared/token_r2/fit"
    transfer = shared/"prepared/fit_transfer"
    for name, expected in em["artifact_sha256"].items():
        if file_sha(expanded/name) != expected: raise AssertionError("Expanded artifact changed: "+name)
    for name in ("ids", "masks", "content_start", "document_content_length"):
        full = np.load(expanded/(name+".npy"), mmap_mode="r", allow_pickle=False)
        a = np.load(base/(name+".npy"), mmap_mode="r", allow_pickle=False)
        b = np.load(transfer/(name+".npy"), mmap_mode="r", allow_pickle=False)
        if not np.array_equal(full[:len(a)], a) or not np.array_equal(full[len(a):], b):
            raise AssertionError("Expanded array source mismatch: "+name)
    expanded_starts = np.load(expanded/"starts.npy", mmap_mode="r", allow_pickle=False)
    a = np.load(base/"starts.npy", mmap_mode="r", allow_pickle=False)
    b = np.load(transfer/"starts.npy", mmap_mode="r", allow_pickle=False)
    if not np.array_equal(expanded_starts[:len(a)],a) or not np.array_equal(expanded_starts[len(a):], b[1:]+a[-1]):
        raise AssertionError("Expanded starts mismatch")
    report = {"complete": True, "protocol_sha256": protocol_sha,
              "roles": role_summary, "old_source_overlap": 0, "cross_new_role_source_payload_context_overlap": 0,
              "old_permitted_positive_span_strings": len(old_payload_texts),
              "maximum_old_fit_development_4word_jaccard": max_old_match[0],
              "maximum_match_new_family": max_old_match[1],
              "maximum_match_old_text_sha256": max_old_match[2],
              "expanded_fit": {"manifest_sha256": file_sha(expanded/"manifest.json"),
                               "documents": em["documents"], "windows": em["windows"],
                               "source_arrays_exact_concatenation": True},
              "no_scores_read": True, "old_confirmation_content_read": False,
              "protected_data_read": False,
              "limits": "Lexical shingle check and exact group disjointness do not prove semantic independence. All new payloads use response redirection, a mechanism previously seen by the teacher. One author wrote all payload/context families."}
    destination = canonical/"results/kd_cnn_v2/data_audit.json"
    save(destination, report)
    print(json.dumps({"audit": str(destination), "roles": role_summary,
                      "max_old_4word_jaccard": max_old_match[0]}, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, required=True)
    args = parser.parse_args()
    audit(args.canonical_root.resolve(), Path(__file__).resolve().parents[2])

if __name__ == "__main__": main()
