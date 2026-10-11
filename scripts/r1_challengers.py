"""Bounded Goal R1 challengers. Only train and the two search partitions are read.

Stages are separate dispatches. Learning recipes are fixed before execution;
selection is the final epoch, with no search-based epoch/hyperparameter tuning.
All outputs must be under results/goal_r1. Existing learning runs are refused.
"""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gate_eval import load_config, window_document, fit_threshold, evaluate_documents

DEST = ROOT / "results" / "goal_r1"
SPLITS = ("train", "search_temporary_threshold", "search_candidate_scoring")
META = ("document_id", "label", "source_group_id", "source_ids", "source_dataset",
        "attack_type", "attack_template_id", "injection_position")
RECIPES = {
    "feature": {"epochs": 20, "optimizer": "SGD", "learning_rate": .5,
                "momentum": .9, "weight_decay": 1e-5, "bins": 262144,
                "features": "hashed raw-text word unigram/bigram term counts; L2 normalized"},
    "cnn": {"epochs": 10, "optimizer": "AdamW", "learning_rate": 1e-3,
            "weight_decay": .01, "embedding_dim": 64, "kernels": [3, 5, 7],
            "channels_each": 64, "pooling": "global max at valid token centers",
            "dropout": .2, "pretrained_features": False},
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def output_path(value, default):
    path = Path(value) if value else default
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(DEST.resolve()) or path == DEST.resolve():
        raise ValueError("Output must be a child of results/goal_r1")
    return path


def fresh(path):
    if path.exists():
        raise RuntimeError(f"Preserve existing artifact/run; choose a fresh output: {path}")
    path.mkdir(parents=True)


def resource_check(preflight=False):
    """Actual memory floors on both hosts; no missing telemetry means pass."""
    gib = 1024**3
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                    "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
        memory = MEMORYSTATUSEX()
        memory.dwLength = ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise RuntimeError("Windows memory telemetry unavailable")
        windows = memory.ullAvailPhys
        result = subprocess.run(["wsl.exe", "-d", "Ubuntu", "--exec", "/bin/cat", "/proc/meminfo"],
                                capture_output=True, text=True, timeout=20, check=True)
        linux = result.stdout
    else:
        linux = Path("/proc/meminfo").read_text(encoding="utf-8")
        result = subprocess.run(["powershell.exe", "-NoProfile", "-Command",
            "[Console]::Write((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory*1024)"],
            capture_output=True, text=True, timeout=20, check=True)
        windows = int(result.stdout.strip())
    wsl = int(next(line.split()[1] for line in linux.splitlines() if line.startswith("MemAvailable:"))) * 1024
    record = {"observed_utc": utc(), "windows_available_bytes": windows, "wsl_available_bytes": wsl,
              "windows_floor_bytes": (4 if preflight else 2)*gib,
              "wsl_floor_bytes": (8 if preflight else 4)*gib}
    if windows < record["windows_floor_bytes"] or wsl < record["wsl_floor_bytes"]:
        raise RuntimeError("Resource floor failed: " + json.dumps(record))
    return record


class Guard:
    def __init__(self, seconds):
        self.started = time.monotonic()
        self.limit = seconds
        self.last_check = self.started
        self.preflight = resource_check(True)
        self.observations = [self.preflight]
        self.output = None

    def attach(self, output):
        self.output = output
        self.save()

    def save(self, failure=None):
        if self.output:
            json_write(self.output/"resources.json", {"observations":self.observations,
                "failure":failure, "runtime_poll_seconds":5, "max_seconds":self.limit})

    def check(self):
        now = time.monotonic()
        try:
            if now-self.started > self.limit:
                raise RuntimeError("Finite dispatch time budget exhausted; no completed run claimed")
            if (DEST / "pause.request").exists():
                raise RuntimeError("Goal R1 pause requested")
            if now-self.last_check >= 5:
                self.observations.append(resource_check())
                self.last_check = now
                self.save()
        except Exception as error:
            self.save({"observed_utc":utc(), "error":str(error)})
            raise


def setup_torch(device):
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    random.seed(42)
    torch.manual_seed(42)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return torch


def execution_settings(torch):
    return {"python":sys.version, "torch":torch.__version__, "cuda_runtime":torch.version.cuda,
            "intraop_threads":torch.get_num_threads(), "interop_threads":torch.get_num_interop_threads(),
            "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark":torch.backends.cudnn.benchmark,
            "cudnn_deterministic":torch.backends.cudnn.deterministic,
            "cuda_matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32":torch.backends.cudnn.allow_tf32,
            "cublas_workspace_config":os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def tokenizer(config):
    from transformers import AutoTokenizer
    detector = config["models"]["detector"]
    return AutoTokenizer.from_pretrained(detector["id"], revision=detector["tokenizer_revision"],
        cache_dir=ROOT/config["paths"]["checkpoints"]/"huggingface", use_fast=True, local_files_only=True)


def contract(config):
    if (config["windowing"]["primary_length"], config["windowing"]["content_token_overlap"]) != (256, 64):
        raise RuntimeError("R1 requires the frozen L256 / overlap64 policy")
    if config["seed"] != 42 or config["evaluation"]["fpr_target"] != .01:
        raise RuntimeError("R1 requires seed42 and primary alpha .01")
    anchor_path = ROOT/"results"/"goal2"/"selection.json"
    anchor = json_read(anchor_path)
    metric = anchor["metrics"]["primary"]["candidate_scoring"]
    if anchor["selected_epoch"] != 3 or metric["recall"] is None:
        raise RuntimeError("Actual fixed floating reference is not the expected saved epoch3 anchor")
    return {"seed": 42, "windowing": config["windowing"], "tokenizer": config["models"]["detector"],
            "alpha": .01, "attack_rule": "score > threshold", "fixed_bert_anchor_path": str(anchor_path),
            "fixed_bert_checkpoint": anchor["checkpoint"], "fixed_bert_counts": metric,
            "fixed_bert_recall": metric["recall"], "recall_floor": metric["recall"]-.01,
            "selection": "fixed final epoch; no search-based training selection"}


def prepare(config, folder, guard):
    import numpy as np
    from r1_models import word_features
    tok = tokenizer(config)
    shared = contract(config)
    fresh(folder)
    guard.attach(folder)
    json_write(folder/"contract.json", shared)
    groups = {}
    summary = {}
    for split in SPLITS:
        source = ROOT/config["paths"]["data"]/"prepared"/(split+".jsonl")
        target = folder/split
        target.mkdir()
        ids, masks, labels, doc_indices, features, values, offsets = [], [], [], [], [], [], [0]
        docs, starts = [], [0]
        split_groups = set()
        seen_ids = set()
        started = time.perf_counter()
        for doc_index, doc in enumerate(rows(source)):
            guard.check()
            doc_id = doc["document_id"]
            if doc_id in seen_ids:
                raise ValueError("Duplicate document ID")
            seen_ids.add(doc_id)
            split_groups.add(doc["source_group_id"])
            windows = window_document(tok, doc["text"], doc_id, doc["label"], doc.get("injection_spans"), config)
            meta = {key: doc.get(key) for key in META}
            meta.update(window_count=len(windows), is_long_document=len(windows)>1,
                        content_token_count=windows[-1]["content_token_end"])
            docs.append(meta)
            for window in windows:
                if window["window_label"] is None:
                    raise ValueError("Unannotated positive needs aggregate-document training; cannot assign fake span labels")
                padding = 256-len(window["input_ids"])
                ids.append(np.asarray(window["input_ids"]+[tok.pad_token_id]*padding, dtype=np.int32))
                masks.append(np.asarray(window["attention_mask"]+[0]*padding, dtype=np.uint8))
                labels.append(window["window_label"])
                doc_indices.append(doc_index)
                text = doc["text"][window["char_start"]:window["char_end"]] if window["char_start"] is not None else ""
                fi, fv = word_features(text)
                features.append(np.asarray(fi, dtype=np.int32))
                values.append(np.asarray(fv, dtype=np.float32))
                offsets.append(offsets[-1]+len(fi))
            starts.append(len(ids))
            if (doc_index+1)%1000 == 0:
                print(json.dumps({"event":"prepare", "split":split, "documents":doc_index+1, "windows":len(ids)}), flush=True)
        arrays = {"ids": np.stack(ids), "masks": np.stack(masks), "labels": np.asarray(labels, dtype=np.int64),
                  "doc_indices": np.asarray(doc_indices, dtype=np.int32), "starts": np.asarray(starts, dtype=np.int64),
                  "feature_indices": np.concatenate(features), "feature_values": np.concatenate(values),
                  "feature_offsets": np.asarray(offsets, dtype=np.int64)}
        for name, array in arrays.items():
            np.save(target/(name+".npy"), array, allow_pickle=False)
        with (target/"documents.jsonl").open("w", encoding="utf-8") as handle:
            for doc in docs:
                handle.write(json.dumps(doc, ensure_ascii=False)+"\n")
        groups[split] = split_groups
        summary[split] = {"source":str(source), "documents":len(docs), "groups":len(split_groups),
                          "windows":len(ids), "array_bytes":sum(v.nbytes for v in arrays.values()),
                          "seconds":time.perf_counter()-started,
                          "window_labels":dict(Counter(labels)),
                          "source_documents":dict(Counter(d["source_dataset"] for d in docs))}
        del arrays, ids, masks, features, values
    for i, split in enumerate(SPLITS):
        for other in SPLITS[i+1:]:
            if groups[split] & groups[other]:
                raise ValueError("Source group crosses permitted partitions")
    summary.update(vocab_size=len(tok), pad_id=tok.pad_token_id, resource_preflight=guard.preflight,
                   grouped_split_check="passed", closed_partitions_opened=False, complete=True)
    json_write(folder/"summary.json", summary)


def load_prepared(folder, config):
    import numpy as np
    summary = json_read(folder/"summary.json")
    if not summary["complete"] or json_read(folder/"contract.json") != contract(config):
        raise RuntimeError("Prepared contract differs from the actual fixed anchor/config")
    return summary


def arrays(folder, split):
    import numpy as np
    target = folder/split
    return {p.stem:np.load(p, mmap_mode="r", allow_pickle=False) for p in target.glob("*.npy")}


def batch_inputs(data, indices, model_name, device, torch):
    import numpy as np
    if model_name == "cnn":
        return {"input_ids":torch.from_numpy(np.array(data["ids"][indices], dtype=np.int64)).to(device),
                "attention_mask":torch.from_numpy(np.array(data["masks"][indices], dtype=np.int64)).to(device)}
    pieces, weights, offsets, at = [], [], [], 0
    for index in indices:
        start, end = data["feature_offsets"][index:index+2]
        pieces.append(data["feature_indices"][start:end])
        weights.append(data["feature_values"][start:end])
        offsets.append(at)
        at += end-start
    return {"indices":torch.from_numpy(np.concatenate(pieces).astype(np.int64)).to(device),
            "values":torch.from_numpy(np.concatenate(weights).copy()).to(device),
            "offsets":torch.tensor(offsets, dtype=torch.long, device=device)}


def new_model(name, summary):
    from r1_models import FeatureLogistic, TokenCNN
    return FeatureLogistic() if name == "feature" else TokenCNN(summary["vocab_size"], summary["pad_id"])


def save_checkpoint(path, model, optimizer, epoch, torch):
    torch.save({"state_dict":model.state_dict(), "optimizer":optimizer.state_dict(), "epoch":epoch,
                "torch_rng":torch.get_rng_state(), "python_rng":random.getstate(),
                "cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, path)


def train(name, prepared, folder, config, device, guard):
    torch = setup_torch(device)
    import torch.nn.functional as F
    summary = load_prepared(prepared, config)
    data = arrays(prepared, "train")
    fresh(folder)
    guard.attach(folder)
    model = new_model(name, summary).to(device)
    recipe = RECIPES[name]
    options = {"lr":recipe["learning_rate"], "weight_decay":recipe["weight_decay"]}
    optimizer = torch.optim.SGD(model.parameters(), momentum=recipe["momentum"], **options) if name == "feature" else torch.optim.AdamW(model.parameters(), **options)
    run = {"recorded_utc":utc(), "command":sys.argv, "model":name, "recipe":recipe,
           "contract":contract(config), "prepared":str(prepared), "device":device,
           "device_name":torch.cuda.get_device_name() if device=="cuda" else "CPU",
           "python":sys.version, "torch":torch.__version__, "threads":4, "parameter_count":sum(p.numel() for p in model.parameters()),
           "effective_document_batch":32, "microbatch_windows":64, "gradient_clip_norm":5.0,
           "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
           "cudnn_benchmark":torch.backends.cudnn.benchmark,
           "cudnn_deterministic":torch.backends.cudnn.deterministic,
           "cuda_matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32,
           "cudnn_allow_tf32":torch.backends.cudnn.allow_tf32,
           "cublas_workspace_config":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
           "loss":"span-label window CE averaged within document then across 32 documents",
           "arithmetic":"FP32; TF32 disabled; standard accelerated kernels", "resource_preflight":guard.preflight}
    json_write(folder/"run.json", run)
    history = []
    count = len(data["starts"])-1
    all_started = time.perf_counter()
    with (folder/"training_log.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(1, recipe["epochs"]+1):
            model.train()
            order = list(range(count))
            random.Random(42+epoch).shuffle(order)
            epoch_started = time.perf_counter()
            loss_sum = 0.
            documents_seen = 0
            for batch_number, start in enumerate(range(0, count, 32)):
                guard.check()
                docs = order[start:start+32]
                indices, weights = [], []
                for doc in docs:
                    first, end = data["starts"][doc:doc+2]
                    indices.extend(range(first, end))
                    weights.extend([1/(len(docs)*(end-first))]*(end-first))
                optimizer.zero_grad(set_to_none=True)
                loss_value = 0.
                for at in range(0, len(indices), 64):
                    current = indices[at:at+64]
                    logits = model(**batch_inputs(data, current, name, device, torch)).logits
                    target = torch.tensor(data["labels"][current].tolist(), dtype=torch.long, device=device)
                    scale = torch.tensor(weights[at:at+64], dtype=torch.float32, device=device)
                    loss = (F.cross_entropy(logits, target, reduction="none")*scale).sum()
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss")
                    loss.backward()
                    loss_value += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                loss_sum += loss_value*len(docs)
                documents_seen += len(docs)
                if batch_number%100 == 0:
                    row = {"event":"training", "epoch":epoch, "batch":batch_number,
                           "documents_seen":documents_seen, "mean_document_balanced_window_ce":loss_sum/documents_seen,
                           "elapsed_seconds":time.perf_counter()-epoch_started}
                    log.write(json.dumps(row)+"\n")
                    log.flush()
                    print(json.dumps(row), flush=True)
            record = {"epoch":epoch, "mean_document_balanced_window_ce":loss_sum/documents_seen,
                      "training_seconds":time.perf_counter()-epoch_started, "documents":documents_seen}
            history.append(record)
            save_checkpoint(folder/"checkpoint_latest.pt", model, optimizer, epoch, torch)
            json_write(folder/"history.json", history)
            print(json.dumps({"event":"epoch_complete", **record}), flush=True)
    json_write(folder/"complete.json", {"complete":True, "final_epoch":recipe["epochs"],
        "training_seconds":time.perf_counter()-all_started, "parameter_count":run["parameter_count"],
        "selected_checkpoint":"checkpoint_latest.pt", "selection":"fixed final epoch",
        "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated() if device=="cuda" else None})


def trained_model(name, run, prepared, config, device, torch):
    summary = load_prepared(prepared, config)
    metadata = json_read(run/"run.json")
    done = json_read(run/"complete.json")
    if metadata["contract"] != contract(config) or metadata["recipe"] != RECIPES[name] or metadata["model"] != name or not done["complete"]:
        raise RuntimeError("Completed matching fixed-recipe training run required")
    model = new_model(name, summary)
    saved = torch.load(run/"checkpoint_latest.pt", map_location="cpu", weights_only=False)
    if saved["epoch"] != RECIPES[name]["epochs"]:
        raise RuntimeError("Only fixed final epoch can be scored")
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval()


def metrics_for(threshold_rows, scoring_rows, shared):
    benign = [r["risk_score"] for r in threshold_rows if r["label"]==0]
    threshold = fit_threshold(benign, .01)
    def metric(subset):
        return evaluate_documents([r["risk_score"] for r in subset], [r["label"] for r in subset], threshold)
    overall = metric(scoring_rows)
    strata = {"single_window":metric([r for r in scoring_rows if not r["is_long_document"]]),
              "long_document":metric([r for r in scoring_rows if r["is_long_document"]])}
    for key in ("source_dataset", "attack_type", "attack_template_id", "injection_position"):
        for value in sorted({str(r.get(key)) for r in scoring_rows}):
            strata[key+":"+value] = metric([r for r in scoring_rows if str(r.get(key))==value])
    return {"threshold":threshold, "fit_split":"search_temporary_threshold benign only",
            "temporary_threshold_benign":evaluate_documents(benign, [0]*len(benign), threshold),
            "candidate_scoring":overall, "strata":strata, "fixed_bert_reference":shared,
            "passes_development_screen":overall["fpr"] <= .01 and overall["recall"] >= shared["recall_floor"],
            "coverage_note":"Existing development attacks only; no new difficult-benign benchmark; NotInject/final partitions remain closed"}


def score(name, run, prepared, folder, config, device, guard):
    torch = setup_torch(device)
    model = trained_model(name, run, prepared, config, device, torch)
    fresh(folder)
    guard.attach(folder)
    outputs = {}
    for split in SPLITS[1:]:
        data = arrays(prepared, split)
        documents = list(rows(prepared/split/"documents.jsonl"))
        risks = []
        started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(data["ids"])):
                guard.check()
                indices = [start]
                logits = model(**batch_inputs(data, indices, name, device, torch)).logits
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Nonfinite scoring logits")
                risks.extend(torch.softmax(logits.float(), -1)[:,1].cpu().tolist())
        with (folder/(split+"_windows.jsonl")).open("x", encoding="utf-8") as handle:
            for index, risk in enumerate(risks):
                doc_index = int(data["doc_indices"][index])
                handle.write(json.dumps({"document_id":documents[doc_index]["document_id"],
                    "window_index":index-int(data["starts"][doc_index]), "window_label":int(data["labels"][index]),
                    "risk_score":risk})+"\n")
        for index, doc in enumerate(documents):
            first, end = data["starts"][index:index+2]
            doc.update(risk_score=max(risks[first:end]), window_scores=risks[first:end])
        with (folder/(split+"_documents.jsonl")).open("x", encoding="utf-8") as handle:
            for doc in documents:
                handle.write(json.dumps(doc, ensure_ascii=False)+"\n")
        outputs[split] = documents
        print(json.dumps({"event":"scored", "split":split, "documents":len(documents), "windows":len(risks),
                          "seconds":time.perf_counter()-started}), flush=True)
    metrics = metrics_for(outputs[SPLITS[1]], outputs[SPLITS[2]], contract(config))
    metrics.update(model=name, checkpoint=str(run/"checkpoint_latest.pt"), device=device, batch_size=1,
                   execution_settings=execution_settings(torch),
                   arithmetic="FP32 standard kernels; not an accelerated INT benchmark", complete=True)
    json_write(folder/"metrics.json", metrics)
    print(json.dumps(metrics["candidate_scoring"]), flush=True)


def real_inputs(tok, doc, name, device, config, torch):
    from r1_models import word_features
    windows = window_document(tok, doc["text"], doc["document_id"], doc["label"], doc.get("injection_spans"), config)
    if name == "cnn":
        return windows, tok.pad([{k:w[k] for k in ("input_ids", "attention_mask", "token_type_ids")} for w in windows],
            padding="max_length", max_length=256, return_tensors="pt")
    indices, values, offsets, at = [], [], [], 0
    for window in windows:
        text = doc["text"][window["char_start"]:window["char_end"]] if window["char_start"] is not None else ""
        fi, fv = word_features(text)
        indices.extend(fi)
        values.extend(fv)
        offsets.append(at)
        at += len(fi)
    return windows, {"indices":torch.tensor(indices, dtype=torch.long), "values":torch.tensor(values, dtype=torch.float32),
                     "offsets":torch.tensor(offsets, dtype=torch.long)}


def bringup(name, folder, config, device, guard):
    torch = setup_torch(device)
    tok = tokenizer(config)
    model = new_model(name, {"vocab_size":len(tok), "pad_id":tok.pad_token_id}).to(device)
    fresh(folder)
    guard.attach(folder)
    selected = []
    for doc in rows(ROOT/config["paths"]["data"]/"prepared"/"train.jsonl"):
        if doc["label"] not in {r["label"] for r in selected}:
            selected.append(doc)
        if len(selected)==2:
            break
    observations = []
    for doc in selected:
        guard.check()
        windows, inputs = real_inputs(tok, doc, name, device, config, torch)
        if any(w["window_label"] is None for w in windows):
            raise RuntimeError("Bringup needs actual span labels")
        logits = model(**{k:v.to(device) for k,v in inputs.items()}).logits
        labels = torch.tensor([w["window_label"] for w in windows], device=device)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Bringup loss nonfinite")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Bringup gradient nonfinite")
        observations.append({"document_id":doc["document_id"], "label":doc["label"], "windows":len(windows),
                             "window_labels":[w["window_label"] for w in windows], "loss":loss.item(),
                             "logits":logits.detach().cpu().tolist()})
        model.zero_grad(set_to_none=True)
    json_write(folder/"bringup.json", {"model":name, "device":device, "observations":observations,
        "passed":True, "optimizer_steps":0, "learning_run":False, "resource_preflight":guard.preflight})


def timing(name, run, prepared, folder, config, device, guard, score_path):
    torch = setup_torch(device)
    model = trained_model(name, run, prepared, config, device, torch)
    quality = json_read(score_path/"metrics.json")
    if (not quality["complete"] or not quality["passes_development_screen"]
            or quality["checkpoint"] != str(run/"checkpoint_latest.pt")
            or quality["device"] != device or quality["batch_size"] != 1):
        raise RuntimeError("Timing requires this backend's full passing B1 development screen")
    tok = tokenizer(config)
    old = json_read(ROOT/"results"/"goal2"/"timing_pilot.json")
    requested = [r["document_id"] for r in old["raw"]]
    if len(requested) != 32 or len(set(requested)) != 32:
        raise RuntimeError("Expected actual old matched32 timing inputs")
    lookup = {}
    for doc in rows(ROOT/config["paths"]["data"]/"prepared"/"search_candidate_scoring.jsonl"):
        if doc["document_id"] in requested:
            lookup[doc["document_id"]] = doc
    if set(lookup) != set(requested):
        raise RuntimeError("Old matched timing document unavailable")
    chosen = [lookup[i] for i in requested]
    fresh(folder)
    guard.attach(folder)
    raw = []
    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()
    with torch.inference_mode():
        for index, doc in enumerate(chosen[:8]+chosen):
            guard.check()
            synchronize()
            began = time.perf_counter()
            windows, inputs = real_inputs(tok, doc, name, device, config, torch)
            if len(windows) != 1:
                raise RuntimeError("Matched timing input no longer single window")
            prepared_at = time.perf_counter()
            inputs = {k:v.to(device) for k,v in inputs.items()}
            synchronize()
            transferred_at = time.perf_counter()
            risk = torch.softmax(model(**inputs).logits.float(), -1)[:,1].max()
            synchronize()
            inferred_at = time.perf_counter()
            value = risk.item()
            decision = value > quality["threshold"]
            synchronize()
            ended = time.perf_counter()
            if index >= 8:
                raw.append({"document_id":doc["document_id"], "risk_score":value, "attack_decision":decision,
                    "preparation_ms":(prepared_at-began)*1000, "host_to_device_ms":(transferred_at-prepared_at)*1000,
                    "inference_and_aggregation_ms":(inferred_at-transferred_at)*1000,
                    "result_transfer_ms":(ended-inferred_at)*1000, "total_ms":(ended-began)*1000})
    means = {k:sum(r[k] for r in raw)/len(raw) for k in raw[0] if k.endswith("_ms")}
    json_write(folder/"timing.json", {"model":name, "checkpoint":str(run/"checkpoint_latest.pt"),
        "device":device, "device_name":torch.cuda.get_device_name() if device=="cuda" else "CPU",
        "threads":4, "batch_size":1, "padded_total_length":256, "warmups":8, "measured":len(raw),
        "threshold":quality["threshold"], "attack_rule":"score > threshold",
        "execution_settings":execution_settings(torch),
        "matched_input_source":"results/goal2/timing_pilot.json raw32 IDs", "raw":raw, "means":means,
        "scope":"Warm complete document check includes tokenizer, feature extraction, padding, transfers, inference, max and result return; model loading excluded",
        "arithmetic":"FP32 standard accelerated kernels; no emulated INT speed claim", "resource_preflight":guard.preflight})
    print(json.dumps(means), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "bringup", "train-feature", "train-cnn", "score", "timing"), required=True)
    parser.add_argument("--model", choices=("feature", "cnn"), default="cnn")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--prepared", default="results/goal_r1/prepared_v1")
    parser.add_argument("--run-dir", help="Fresh learning output, or completed run for score/timing")
    parser.add_argument("--output", help="Fresh output under results/goal_r1")
    parser.add_argument("--score-dir", help="Full passing score output required for timing")
    parser.add_argument("--max-seconds", type=int, default=1800)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 3600:
        raise ValueError("Dispatch time budget must be 1..3600 seconds")
    if args.stage.startswith("train-"):
        args.model = args.stage.removeprefix("train-")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = load_config()
    contract(config)
    if (DEST/"pause.request").exists():
        raise RuntimeError("Goal R1 pause requested")
    guard = Guard(args.max_seconds)
    prepared = output_path(args.prepared, DEST/"prepared_v1")
    run = output_path(args.run_dir, DEST/(args.model+"_initial_v1"))
    default = DEST/(args.stage+"_"+args.model+"_"+args.device+"_v1")
    folder = output_path(args.output, default)
    if args.stage == "prepare":
        prepare(config, output_path(args.output, prepared), guard)
    elif args.stage == "bringup":
        bringup(args.model, folder, config, args.device, guard)
    elif args.stage.startswith("train-"):
        train(args.model, prepared, run, config, args.device, guard)
    elif args.stage == "score":
        score(args.model, run, prepared, folder, config, args.device, guard)
    else:
        if not args.score_dir:
            parser.error("--score-dir is required for timing")
        timing(args.model, run, prepared, folder, config, args.device, guard,
               output_path(args.score_dir, DEST/"unused"))


if __name__ == "__main__":
    main()
