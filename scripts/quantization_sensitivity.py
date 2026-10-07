"""Goal 3 train-only calibration, fixed-map sensitivity, and two uniform QAT runs.

Uses only train and the two assigned search subsets. This driver never benchmarks
fake quantization as INT4. Execute CUDA stages only after the training GPU is free.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gate_eval import load_config, window_document, score_documents, evaluate_documents
from gate_quant import (SCHEME_ID, GateQuantLinear, calibrate_inputs, configure_exact_fp32,
                        encoder_linears, format_record, group_names, install_quantized_linears,
                        load_quantized_checkpoint, logical_storage_estimate, normalize_precision_map,
                        save_quantized_checkpoint, set_execution_mode, set_precision_map,
                        weight_numerical_error)
from train_detector import read_jsonl, enrich, summarize, save_json, save_jsonl

FIELDS = ("input_ids", "attention_mask", "token_type_ids")
SPLITS = ("search_temporary_threshold", "search_candidate_scoring")


def pause_requested(destination):
    return (destination / "pause.request").exists()


def stop_if_paused(destination):
    if pause_requested(destination):
        print(json.dumps({"event": "paused", "boundary": "completed inference split/map",
                          "resume": "remove pause.request when authorized; rerun identical command"}), flush=True)
        raise SystemExit(0)



def device_batch(tokenizer, windows, config, device):
    return {key: value.to(device) for key, value in tokenizer.pad(
        [{key: row[key] for key in FIELDS} for row in windows],
        padding="max_length", max_length=config["windowing"]["primary_length"],
        return_tensors="pt").items()}


def model_source(selection):
    return ROOT / selection["checkpoint"]


def load_floating(selection, device):
    from transformers import AutoModelForSequenceClassification
    return AutoModelForSequenceClassification.from_pretrained(
        model_source(selection), local_files_only=True).float().to(device).eval()


def calibration_windows(tokenizer, train, config, maximum):
    order = list(range(len(train)))
    random.Random(config["seed"]).shuffle(order)
    selected = []
    sources = []
    for index in order:
        document = train[index]
        if document["split"] != "train":
            raise ValueError("Activation calibration must use training documents only.")
        windows = window_document(tokenizer, document["text"], document["document_id"],
                                  document["label"], document.get("injection_spans"), config)
        for window in windows:
            selected.append(window)
            sources.append({"document_id": document["document_id"],
                            "source_group_id": document["source_group_id"],
                            "window_index": window["window_index"]})
            if len(selected) == maximum:
                return selected, sources
    return selected, sources


def get_calibration(selection, tokenizer, train, config, args, destination):
    path = destination / "calibration.json"
    identity = {"scheme_id": SCHEME_ID, "checkpoint": selection["checkpoint"],
                "seed": config["seed"], "requested_windows": args.calibration_windows,
                "windowing": config["windowing"]}
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["identity"] != identity:
            raise ValueError("Existing calibration identity differs; do not silently recalibrate.")
        return record
    windows, sources = calibration_windows(tokenizer, train, config, args.calibration_windows)
    model = load_floating(selection, args.device)
    batches = (device_batch(tokenizer, windows[start:start + args.batch_size], config, args.device)
               for start in range(0, len(windows), args.batch_size))
    record = calibrate_inputs(model, batches)
    record.update({"identity": identity, "source_partition": "train", "source_windows": sources,
                   "created_utc": datetime.now(timezone.utc).isoformat()})
    save_json(path, record)
    del model
    print(json.dumps({"event": "calibrated", "windows": len(windows), "modules": len(record["modules"])}), flush=True)
    return record


def load_existing_candidate(folder, identity):
    meta = folder / "candidate.json"
    if not meta.exists():
        return None
    record = json.loads(meta.read_text(encoding="utf-8"))
    if record["identity"] != identity:
        raise ValueError(f"Candidate identity differs at {folder}; refusing stale-score reuse.")
    if not all((folder / f"{split}_documents.jsonl").exists() and
               (folder / f"{split}_windows.jsonl").exists() for split in SPLITS):
        return None
    return record


def score_candidate(name, model, tokenizer, sets, config, selection, args, destination,
                    precision, qat=False, training_record=None):
    folder = destination / name
    identity = {"checkpoint": selection["checkpoint"], "scheme_id": SCHEME_ID,
                "precision_map": dict(normalize_precision_map(precision)), "qat": bool(qat),
                "windowing": config["windowing"], "evaluation": config["evaluation"]}
    prior = load_existing_candidate(folder, identity)
    if prior is not None:
        print(json.dumps({"event": "reuse_candidate", "name": name}), flush=True)
        return prior
    folder.mkdir(parents=True, exist_ok=True)
    progress_path = folder / "scoring_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"identity": identity, "completed_splits": []}
    if progress["identity"] != identity:
        raise ValueError("Partial candidate identity differs; refusing stale-score reuse.")
    outputs = {}
    start = time.perf_counter()
    for split in SPLITS:
        stop_if_paused(destination)
        if split in progress["completed_splits"]:
            scored = {level: read_jsonl(folder / f"{split}_{level}.jsonl")
                      for level in ("documents", "windows")}
        else:
            scored = enrich(score_documents(model, tokenizer, sets[split],
                                            batch_size=args.batch_size, device=args.device, config=config), sets[split])
            for level in ("documents", "windows"):
                save_jsonl(folder / f"{split}_{level}.jsonl", scored[level])
            progress["completed_splits"].append(split)
            save_json(progress_path, progress)
        outputs[split] = scored
        print(json.dumps({"event": "scored", "name": name, "split": split,
                          "documents": len(scored["documents"])}), flush=True)
    metrics = summarize(outputs[SPLITS[0]]["documents"], outputs[SPLITS[1]]["documents"], config)
    fp_threshold = selection["metrics"]["primary"]["threshold"]
    rows = outputs[SPLITS[1]]["documents"]
    metrics["at_floating_threshold"] = evaluate_documents(
        [row["risk_score"] for row in rows], [row["label"] for row in rows], fp_threshold)
    record = {"name": name, "identity": identity, "metrics": metrics,
              "weight_error": weight_numerical_error(model), "storage_estimate": logical_storage_estimate(model),
              "scoring_elapsed_seconds": time.perf_counter() - start,
              "timing_scope": "research scoring cost; not accelerated-format inference latency",
              "training_record": training_record}
    save_json(folder / "candidate.json", record)
    return record


def read_candidate_rows(destination, name, level="documents"):
    return read_jsonl(destination / name / f"search_candidate_scoring_{level}.jsonl")


def changes(reference_rows, rows, reference_threshold, threshold):
    baseline = {row["document_id"]: row for row in reference_rows}
    if set(baseline) != {row["document_id"] for row in rows}:
        raise ValueError("Paired comparison requires identical document IDs.")
    new_misses, recovered, new_fp, removed_fp = [], [], [], []
    differences = []
    for row in rows:
        before = baseline[row["document_id"]]
        old = before["risk_score"] > reference_threshold
        new = row["risk_score"] > threshold
        differences.append(row["risk_score"] - before["risk_score"])
        if row["label"] == 1:
            if old and not new:
                new_misses.append(row["document_id"])
            if new and not old:
                recovered.append(row["document_id"])
        else:
            if new and not old:
                new_fp.append(row["document_id"])
            if old and not new:
                removed_fp.append(row["document_id"])
    return {"new_misses": new_misses, "recovered_attacks": recovered,
            "new_false_positives": new_fp, "removed_false_positives": removed_fp,
            "mean_score_change": sum(differences) / len(differences),
            "mean_absolute_score_change": sum(abs(value) for value in differences) / len(differences)}


def compare_record(name, record, base_name, base_record, destination):
    base_rows = read_candidate_rows(destination, base_name)
    rows = read_candidate_rows(destination, name)
    primary = record["metrics"]["primary"]
    base_primary = base_record["metrics"]["primary"]
    reference_windows = {(row["document_id"], row["window_index"]): row
                         for row in read_candidate_rows(destination, base_name, "windows")}
    errors = []
    for row in read_candidate_rows(destination, name, "windows"):
        before = reference_windows[(row["document_id"], row["window_index"])]
        errors.extend(a-b for a, b in zip(row["logits"], before["logits"]))
    return {
        "name": name, "reference": base_name,
        "recall_loss_after_refit": base_primary["candidate_scoring"]["recall"] - primary["candidate_scoring"]["recall"],
        "fpr_change_after_refit": primary["candidate_scoring"]["fpr"] - base_primary["candidate_scoring"]["fpr"],
        "mean_document_bce_change": record["metrics"]["candidate_mean_document_bce"] - base_record["metrics"]["candidate_mean_document_bce"],
        "ordinary_accuracy_change": record["metrics"]["ordinary_threshold_0_5"]["accuracy"] - base_record["metrics"]["ordinary_threshold_0_5"]["accuracy"],
        "window_logit_mse": sum(value*value for value in errors) / len(errors),
        "window_logit_max_abs_error": max(abs(value) for value in errors),
        "at_reference_threshold": changes(base_rows, rows, base_primary["threshold"], base_primary["threshold"]),
        "after_threshold_refit": changes(base_rows, rows, base_primary["threshold"], primary["threshold"]),
    }


def real_input_agreement(model, tokenizer, train, config, destination):
    """One actual short training email through both CPU arithmetic paths."""
    import torch
    path = destination / "real_input_agreement.json"
    if path.exists():
        return
    chosen = None
    for doc in train:
        windows = window_document(tokenizer, doc["text"], doc["document_id"], doc["label"],
                                  doc.get("injection_spans"), config)
        if len(windows) == 1 and windows[0]["total_token_count"] <= 64:
            chosen = (doc, windows[0])
            break
    if chosen is None:
        raise ValueError("No short real training document was found for arithmetic bring-up.")
    model.cpu().eval()
    # Pad only this actual short input to its natural length; this is not quality scoring.
    batch = tokenizer.pad([{key: chosen[1][key] for key in FIELDS}], return_tensors="pt")
    with torch.inference_mode():
        set_execution_mode(model, "ste_emulation")
        emulated = model(**batch).logits
        set_execution_mode(model, "integer_reference")
        reference = model(**batch).logits
    set_execution_mode(model, "ste_emulation")
    if not torch.equal(emulated, reference):
        raise AssertionError(f"Whole-path arithmetic mismatch: {(emulated-reference).abs().max().item()}")
    save_json(path, {"passed": True, "document_id": chosen[0]["document_id"],
                     "tokens": chosen[1]["total_token_count"], "device": "CPU",
                     "emulated_logits": emulated.tolist(), "integer_reference_logits": reference.tolist(),
                     "precision": "uniform W8A8", "other_operations": "FP32",
                     "max_absolute_difference": float((emulated-reference).abs().max())})


def choose_pairs(single_records):
    """At most three development pairs; no exhaustive subset search."""
    by_recall = sorted(single_records, key=lambda row: (-row["recall_loss_after_refit"], row["group"]))
    by_loss = sorted(single_records, key=lambda row: (-row["mean_document_bce_change"], row["group"]))
    proposed = [(by_recall[0]["group"], by_recall[1]["group"]),
                (by_loss[0]["group"], by_loss[1]["group"])]
    first = by_loss[0]["group"]
    other = next(row["group"] for row in by_loss if row["group"].split(".")[0] != first.split(".")[0])
    proposed.append((first, other))
    unique = []
    for pair in proposed:
        pair = sorted(pair)
        if pair not in unique:
            unique.append(pair)
    return unique[:3]


def qat_one_epoch(model, tokenizer, train_windows, config, args, name, destination,
                  checkpoint_dir, calibration, selection):
    import torch
    import torch.nn.functional as F
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                 weight_decay=config["training"]["weight_decay"])
    effective = int(config["training"]["effective_batch"])
    order = list(range(len(train_windows)))
    random.Random(config["seed"] + 1).shuffle(order)
    torch.manual_seed(config["seed"])
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(config["seed"])
    model.train()
    start = time.perf_counter()
    loss_sum, seen, first_batch, prior_seconds = 0.0, 0, 0, 0.0
    resume_dir = checkpoint_dir / "resume"
    if (resume_dir / "training_state.pt").exists():
        model, saved = load_quantized_checkpoint(resume_dir, args.device)
        if saved["from_floating_checkpoint"] != selection["checkpoint"]:
            raise ValueError("QAT resume belongs to a different floating checkpoint.")
        state = torch.load(resume_dir / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                     weight_decay=config["training"]["weight_decay"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        first_batch, seen, loss_sum = state["next_batch"], state["seen"], state["loss_sum"]
        prior_seconds = state["elapsed_seconds"]
        model.train()
        print(json.dumps({"event": "qat_resumed", "name": name, "next_batch": first_batch,
                          "reused_documents": seen}), flush=True)
    total_batches = math.ceil(len(order) / effective)

    def save_resume(next_batch):
        save_quantized_checkpoint(model, resume_dir, calibration,
                                  {"from_floating_checkpoint": selection["checkpoint"],
                                   "completed_epoch": False, "next_batch": next_batch})
        torch.save({"optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else [],
                    "next_batch": next_batch, "seen": seen, "loss_sum": loss_sum,
                    "elapsed_seconds": prior_seconds + time.perf_counter() - start},
                   resume_dir / "training_state.pt")

    for batch_index in range(first_batch, total_batches):
        if pause_requested(destination):
            save_resume(batch_index)
            print(json.dumps({"event": "paused", "name": name, "next_batch": batch_index}), flush=True)
            raise SystemExit(0)
        indices = order[batch_index*effective:(batch_index+1)*effective]
        flattened, weights = [], []
        for index in indices:
            windows = train_windows[index]
            flattened.extend(windows)
            weights.extend([1.0 / (len(windows)*len(indices))] * len(windows))
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        for offset in range(0, len(flattened), args.batch_size):
            micro = flattened[offset:offset+args.batch_size]
            batch = device_batch(tokenizer, micro, config, args.device)
            labels = torch.tensor([w["window_label"] for w in micro], device=args.device)
            factors = torch.tensor(weights[offset:offset+args.batch_size], device=args.device)
            loss = (F.cross_entropy(model(**batch).logits, labels, reduction="none") * factors).sum()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite QAT loss; no successful epoch is claimed.")
            loss.backward()
            batch_loss += float(loss.detach())
        optimizer.step()
        seen += len(indices)
        loss_sum += batch_loss * len(indices)
        if (batch_index + 1) % 100 == 0:
            save_resume(batch_index + 1)
        if batch_index % 50 == 0 or batch_index + 1 == total_batches:
            print(json.dumps({"event": "qat", "name": name, "batch": batch_index+1,
                              "batches": total_batches, "mean_loss": loss_sum/seen}), flush=True)
    record = {"epochs": 1, "from_floating_checkpoint": selection["checkpoint"], "documents": seen,
              "mean_document_balanced_window_ce": loss_sum/seen,
              "seconds": prior_seconds + time.perf_counter()-start,
              "optimizer": "AdamW", "learning_rate": config["training"]["learning_rate"],
              "weight_decay": config["training"]["weight_decay"], "seed": config["seed"],
              "scales": "frozen original FP-checkpoint W4/W8 scales and shared A8 calibration",
              "effective_batch_documents": effective, "microbatch_windows": args.batch_size,
              "loss": "window CE averaged within document then across actual document batch",
              "storage_note": "Saved FP32 parameters/scales; not a packed W4/W8 runtime checkpoint."}
    save_quantized_checkpoint(model, checkpoint_dir, calibration, {"training": record})
    tokenizer.save_pretrained(checkpoint_dir)
    record["actual_saved_quantized_state_bytes"] = (checkpoint_dir / "quantized_state.pt").stat().st_size
    save_json(destination / name / "training.json", record)
    return model, record

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibrate", "scan", "qat", "all"), default="scan")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--calibration-windows", type=int, default=256)
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import AutoTokenizer
    torch.set_num_threads(4)
    configure_exact_fp32()
    config = load_config()
    if not 1 <= args.batch_size <= 32 or args.calibration_windows < 1:
        raise ValueError("Use a positive calibration count and microbatch1..32.")
    selection = json.loads((ROOT / "results" / "goal2" / "selection.json").read_text(encoding="utf-8"))
    destination = ROOT / config["paths"]["results"] / "goal3"
    destination.mkdir(parents=True, exist_ok=True)
    stop_if_paused(destination)
    save_json(destination / "floating_reference.json",
              {"checkpoint": selection["checkpoint"], "metrics": selection["metrics"],
               "source_scores": f"results/goal2/epoch_{selection['selected_epoch']}/",
               "reuse": "No floating inference repeated; original search thresholds retained."})
    source = ROOT / config["paths"]["data"] / "prepared"
    train = read_jsonl(source / "train.jsonl")
    sets = {name: read_jsonl(source / f"{name}.jsonl") for name in SPLITS}
    tokenizer = AutoTokenizer.from_pretrained(model_source(selection), local_files_only=True, use_fast=True)
    calibration = get_calibration(selection, tokenizer, train, config, args, destination)
    save_json(destination / "run.json", {"command": " ".join(sys.argv), "config": config,
                                        "selected_checkpoint": selection["checkpoint"],
                                        "torch": torch.__version__, "transformers": transformers.__version__,
                                        "device": args.device, "format": format_record(),
                                        "calibration_file": "results/goal3/calibration.json"})
    if args.stage == "calibrate":
        return
    if args.stage in ("scan", "all"):
        model = install_quantized_linears(load_floating(selection, "cpu"), calibration, 8)
        real_input_agreement(model, tokenizer, train, config, destination)
        model.to(args.device)
        records = {}
        for bits in (8, 4):
            name = f"ptq_w{bits}_a8"
            set_precision_map(model, bits)
            records[name] = score_candidate(name, model, tokenizer, sets, config, selection, args,
                                            destination, bits)
        base_name = "ptq_w8_a8"
        singles = []
        for group in group_names():
            mapping = normalize_precision_map(8)
            mapping[group] = 4
            name = "single_" + group.replace(".", "_")
            set_precision_map(model, mapping)
            record = score_candidate(name, model, tokenizer, sets, config, selection, args,
                                     destination, mapping)
            comparison = compare_record(name, record, base_name, records[base_name], destination)
            comparison["group"] = group
            singles.append(comparison)
        save_json(destination / "single_group_sensitivity.json", singles)
        pair_list = choose_pairs(singles)
        save_json(destination / "pair_selection.json",
                  {"rule": "top-two recall-harm; top-two BCE-harm; greatest-BCE-harm with strongest distinct-layer partner; deduplicated, maximum3",
                   "pairs": pair_list, "selected_from": "single-group search-scoring results only"})
        pairs = []
        by_group = {record["group"]: record for record in singles}
        for groups in pair_list:
            mapping = normalize_precision_map(8)
            for group in groups:
                mapping[group] = 4
            name = "pair_" + "__".join(group.replace(".", "_") for group in groups)
            set_precision_map(model, mapping)
            record = score_candidate(name, model, tokenizer, sets, config, selection, args,
                                     destination, mapping)
            comparison = compare_record(name, record, base_name, records[base_name], destination)
            comparison["groups"] = groups
            comparison["recall_interaction_vs_additive_singles"] = (
                comparison["recall_loss_after_refit"]
                - sum(by_group[group]["recall_loss_after_refit"] for group in groups))
            comparison["bce_interaction_vs_additive_singles"] = (
                comparison["mean_document_bce_change"]
                - sum(by_group[group]["mean_document_bce_change"] for group in groups))
            pairs.append(comparison)
        save_json(destination / "pair_sensitivity.json", pairs)
        save_json(destination / "uniform_comparison.json",
                  compare_record("ptq_w4_a8", records["ptq_w4_a8"], base_name, records[base_name], destination))
        del model
    if args.stage in ("qat", "all"):
        train_windows = []
        for doc in train:
            windows = window_document(tokenizer, doc["text"], doc["document_id"], doc["label"],
                                      doc.get("injection_spans"), config)
            if any(window["window_label"] is None for window in windows):
                raise ValueError("Unknown spans need document-aggregate supervision, not copied labels.")
            train_windows.append(windows)
        for bits in (8, 4):
            name = f"qat_w{bits}_a8"
            folder = ROOT / config["paths"]["checkpoints"] / "quantized" / name
            if (folder / "quantization.json").exists():
                model, saved = load_quantized_checkpoint(folder, args.device)
                if saved["training"]["from_floating_checkpoint"] != selection["checkpoint"]:
                    raise ValueError("Existing QAT started from a different floating checkpoint.")
                training_record = dict(saved["training"])
                training_record["actual_saved_quantized_state_bytes"] = (folder / "quantized_state.pt").stat().st_size
            else:
                model = install_quantized_linears(load_floating(selection, args.device), calibration, bits)
                model, training_record = qat_one_epoch(model, tokenizer, train_windows, config, args, name,
                                                       destination, folder, calibration, selection)
            score_candidate(name, model, tokenizer, sets, config, selection, args,
                            destination, bits, qat=True, training_record=training_record)
            del model
    print(json.dumps({"event": "goal3_stage_complete", "stage": args.stage,
                      "remaining": "Analyze saved scores and measure actual supported CPU/GPU inference formats."}), flush=True)


if __name__ == "__main__":
    main()
