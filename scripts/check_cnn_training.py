"""CPU checks of real-fit document weighting, microbatch gradients and updates.

No fitting campaign, scoring fold, accelerator or protected data is used.
Dropout is disabled to isolate objective/gradient equivalence across batches.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from cnn_models import build_model, load_search_configs

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results/cnn_study_v1"


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def real_inputs(representation):
    folder = DEST / "prepared" / representation / "fit"
    docs = [json.loads(line) for line in (folder / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
    arrays = {name: np.load(folder / (name + ".npy"), mmap_mode="r", allow_pickle=False)
              for name in ("ids", "masks", "labels", "starts")}
    def choose(label, longest):
        candidates = [i for i, d in enumerate(docs) if d["label"] == label]
        ordered = sorted(candidates, key=lambda i: (docs[i]["content_length"], docs[i]["document_id"]))
        return ordered[-1] if longest else ordered[0]
    chosen = [choose(0, False), choose(1, True), choose(0, True), choose(1, False)]
    assert len(set(chosen)) == 4
    indices, lengths, selected = [], [], []
    for index in chosen:
        lo, hi = map(int, arrays["starts"][index:index + 2])
        assert hi > lo
        indices.extend(range(lo, hi))
        lengths.append(hi - lo)
        selected.append({"document_id": docs[index]["document_id"], "source_group_id": docs[index]["source_group_id"],
                         "label": docs[index]["label"], "content_length": docs[index]["content_length"],
                         "text_sha256": docs[index]["text_sha256"], "windows": hi - lo})
    assert min(lengths) == 1 and max(lengths) > 1
    values = {name: np.asarray(arrays[name][indices], dtype=np.int64).copy() for name in ("ids", "masks", "labels")}
    identity = {name: digest_bytes(a.tobytes()) for name, a in values.items()}
    return {name: torch.from_numpy(a) for name, a in values.items()}, lengths, {
        "representation": representation, "fold": "fit", "selection": "shortest and longest of each document label by content length, then document ID",
        "documents": selected, "windows": len(indices), "selected_array_sha256_int64": identity,
        "prepared_manifest_sha256": digest_bytes((folder / "manifest.json").read_bytes())}


def check(config, inputs, lengths):
    torch.manual_seed(907)
    vocab = 257 if config["representation"] == "byte" else 30522
    reference = build_model(config, vocab, 0).cpu().double().eval()
    microbatched = copy.deepcopy(reference).eval()
    x, mask, labels = inputs["ids"], inputs["masks"], inputs["labels"]
    # Independent explicit objective: calculate a mean window loss for each
    # original document, then give those document losses equal weight.
    logits = reference(x, mask).logits
    assert logits.shape == (len(x), 2) and torch.isfinite(logits).all()
    losses = F.cross_entropy(logits, labels, reduction="none")
    document_losses = []
    offset = 0
    for length in lengths:
        document_losses.append(losses[offset:offset + length].mean())
        offset += length
    explicit = torch.stack(document_losses).mean()
    assert torch.isfinite(explicit)
    explicit.backward()
    # Queue formula: each window gets 1/(documents*its_document_windows),
    # with a single optimizer update after all microbatches accumulate.
    weights = torch.tensor([1 / (len(lengths) * length) for length in lengths for _ in range(length)], dtype=torch.float64)
    objective_value = 0.
    microbatch = 7
    assert len(x) > microbatch
    for start in range(0, len(x), microbatch):
        end = start + microbatch
        out = microbatched(x[start:end], mask[start:end]).logits
        loss = (F.cross_entropy(out, labels[start:end], reduction="none") * weights[start:end]).sum()
        assert torch.isfinite(loss)
        loss.backward()
        objective_value += loss.item()
    loss_delta = abs(explicit.item() - objective_value)
    assert loss_delta <= 1e-10, (config["id"], loss_delta)
    max_gradient_delta, worst_parameter = 0., None
    for (name, p), (other_name, q) in zip(reference.named_parameters(), microbatched.named_parameters()):
        assert name == other_name
        assert p.grad is not None and q.grad is not None, name
        assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all(), name
        delta = float((p.grad - q.grad).abs().amax())
        if delta > max_gradient_delta:
            max_gradient_delta, worst_parameter = delta, name
        try:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-9, rtol=1e-7)
        except AssertionError:
            print(json.dumps({"failed_gradient": name, "config_id": config["id"], "max_abs_difference": delta,
                              "reference_gradient_max": float(p.grad.abs().amax()),
                              "loss_abs_difference": loss_delta}), flush=True)
            raise
    assert torch.count_nonzero(microbatched.embedding.weight.grad[0]) == 0
    # Separate real-input FP32 bring-up, including a fresh FP32 backward.
    microbatched.float().zero_grad(set_to_none=True)
    fp32_logits = microbatched(x, mask).logits
    assert torch.isfinite(fp32_logits).all()
    fp32_loss = (F.cross_entropy(fp32_logits, labels, reduction="none") * weights.float()).sum()
    assert torch.isfinite(fp32_loss)
    fp32_loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in microbatched.parameters())
    assert torch.count_nonzero(microbatched.embedding.weight.grad[0]) == 0
    before = microbatched.classifier.weight.detach().clone()
    norm = torch.nn.utils.clip_grad_norm_(microbatched.parameters(), 5.)
    assert torch.isfinite(norm)
    optimizer = torch.optim.AdamW(microbatched.parameters(), lr=config["training"]["learning_rate"],
                                 weight_decay=config["training"]["weight_decay"])
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in microbatched.parameters())
    assert not torch.equal(before, microbatched.classifier.weight)
    with torch.no_grad():
        assert torch.isfinite(microbatched(x[:2], mask[:2]).logits).all()
    return {"config_id": config["id"], "passed": True, "document_windows": lengths,
            "microbatch_windows": microbatch, "explicit_loss": explicit.item(),
            "weighted_microbatch_loss": objective_value, "loss_absolute_difference": loss_delta,
            "largest_gradient_absolute_difference": max_gradient_delta, "worst_parameter": worst_parameter,
            "objective_gradient_precision": "FP64", "fp32_loss": fp32_loss.item(),
            "gradient_norm_before_clipping": float(norm), "finite_fp32_forward_backward_update": True}


def main():
    started = time.perf_counter()
    torch.set_num_threads(1)
    # Use the native CPU convolution implementation for an objective equality
    # check: shape-dependent oneDNN roundoff may select different max-pool
    # subgradients for repeated byte patterns across microbatch shapes.
    torch.backends.mkldnn.enabled = False
    registry = {c["id"]: c for c in load_search_configs()}
    cache = {rep: real_inputs(rep) for rep in ("token", "byte")}
    results = []
    for config_id in ("textcnn_01", "residual_01", "dpcnn_01", "byte_hierarchy_01"):
        config = registry[config_id]
        arrays, lengths, _ = cache[config["representation"]]
        results.append(check(config, arrays, lengths))
    result = {"passed": True, "purpose": "focused CPU real-fit implementation checks, not training-quality or GPU evidence",
              "device": "cpu", "threads": 1, "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
              "cpu_convolution_backend": "native; oneDNN disabled to isolate objective equality from tied-max subgradient roundoff",
              "dropout": "disabled with eval() in both compared models",
              "loss_tolerance_absolute": 1e-10, "gradient_tolerance_absolute_relative": [1e-9, 1e-7],
              "numerical_attempt_history": "Initial FP32 equality check missed 1e-5 gradient tolerance for byte_hierarchy_01: one element delta1.0960735380649567e-5 with native CPU conv, loss delta3.166496753692627e-8. FP64 isolates objective equality; separate fresh FP32 backward/update checks finite behavior.",
              "inputs": [value[2] for value in cache.values()], "results": results,
              "runner_source_sha256": digest_bytes((ROOT / "scripts/cnn_study.py").read_bytes()),
              "model_source_sha256": digest_bytes((ROOT / "scripts/cnn_models.py").read_bytes()),
              "checker_source_sha256": digest_bytes(Path(__file__).read_bytes()),
              "wall_seconds": time.perf_counter() - started, "protected_partitions_opened": False}
    output = DEST / "training_contract_checks.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"passed": True, "wall_seconds": result["wall_seconds"], "results": results}))


if __name__ == "__main__":
    main()
