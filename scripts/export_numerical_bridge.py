"""Export ONE fixed search-development input for the numerical C-sim bridge.

Default prints the plan. --execute performs CPU reference inference and exports
only the selected W8-QAT weights/scales and this one input. No calibration/test
reads, training, target APIs, generic model conversion, or hardware timing.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = "aeslc:train:panus-s_inbox_1.subject:clean"
CHECKPOINT = ROOT / "checkpoints/quantized/qat_w8_a8"
OUT = ROOT / "results/goal4/numerical_bridge"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def select_row(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["document_id"] == CASE:
                return row
    raise ValueError("Fixed search-development case missing: " + str(path))


def paused():
    if (ROOT / "results/goal2/pause.request").exists():
        raise RuntimeError("User pause marker present.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"checkpoint": str(CHECKPOINT), "case": CASE,
                          "output": str(OUT), "inference_started": False}, indent=2))
        return
    paused()
    import torch
    from transformers import AutoTokenizer
    from gate_eval import window_document, score_documents
    from gate_quant import (load_quantized_checkpoint, configure_exact_fp32,
                            quantize_codes, encoder_linears, set_execution_mode)
    config = read(ROOT / "configs/project.json")
    document = select_row(ROOT / "data/prepared/search_candidate_scoring.jsonl")
    saved_window = select_row(ROOT / "results/goal3/qat_w8_a8/search_candidate_scoring_windows.jsonl")
    saved_document = select_row(ROOT / "results/goal3/qat_w8_a8/search_candidate_scoring_documents.jsonl")
    candidate = read(ROOT / "results/goal3/qat_w8_a8/candidate.json")
    threshold = float(candidate["metrics"]["primary"]["threshold"])
    runtime = configure_exact_fp32()
    torch.set_num_threads(4)
    model, quantization = load_quantized_checkpoint(CHECKPOINT, "cpu")
    model.eval()
    if set(quantization["precision_map"].values()) != {8}:
        raise ValueError("This bridge is fixed to selected uniform W8 QAT.")
    shape = model.config
    if (shape.num_hidden_layers, shape.hidden_size, shape.num_attention_heads,
            shape.intermediate_size, shape.layer_norm_eps) != (4, 256, 4, 1024, 1e-12):
        raise ValueError("Fixed bridge architecture differs.")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True, use_fast=True)
    windows = window_document(tokenizer, document["text"], CASE, document["label"],
                              document.get("injection_spans"), config)
    if len(windows) != 1 or windows[0]["total_token_count"] != 177 or config["windowing"]["primary_length"] != 256:
        raise ValueError("Expected one 177-token window padded to primary length256.")
    batch = tokenizer.pad([{k: windows[0][k] for k in ("input_ids", "attention_mask", "token_type_ids")}],
                          padding="max_length", max_length=256, return_tensors="pt")
    source_text = (ROOT / "hardware/fixed_fp32_service.hpp").read_text()
    revision = re.search(r'^#define GATE_FIXED_FP32_SCHEDULE_REVISION "([^"\n]+)"', source_text, re.M)
    if revision is None:
        raise ValueError("FP32 source schedule revision is required.")
    if OUT.exists() and (OUT / "manifest.json").exists():
        raise RuntimeError("Existing one-case export must be retained; review it before rerunning.")
    OUT.mkdir(parents=True, exist_ok=True)
    files = []

    def floats(name, tensor):
        value = tensor.detach().cpu().float().contiguous()
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite export tensor: " + name)
        path = OUT / (name + ".bin")
        path.write_bytes(value.numpy().astype("<f4", copy=False).tobytes())
        files.append({"file": path.name, "dtype": "float32_le", "elements": value.numel(), "shape": list(value.shape)})

    def codes(name, tensor):
        value = tensor.detach().cpu().to(torch.int8).contiguous()
        path = OUT / (name + ".bin")
        path.write_bytes(value.numpy().tobytes())
        files.append({"file": path.name, "dtype": "int8", "elements": value.numel(), "shape": list(value.shape)})

    suffix = {"attention.self.query": "query", "attention.self.key": "key",
              "attention.self.value": "value", "attention.output.dense": "attention_output",
              "intermediate.dense": "ffn_input", "output.dense": "ffn_output"}
    linear_names = {}
    for _, path, module in encoder_linears(model):
        layer = int(path.split(".")[3])
        tail = ".".join(path.split(".")[4:])
        name = f"layer{layer}_{suffix[tail]}"
        linear_names[path] = name
        codes(name + "_w8", quantize_codes(module.weight, module.weight_scale_8, 8))
        floats(name + "_params", torch.cat((module.input_scale.reshape(1),
                                             module.weight_scale_8.reshape(-1), module.bias.reshape(-1))))
    if len(linear_names) != 24:
        raise ValueError("Expected the actual24 encoder matrices.")

    with torch.inference_mode():
        embeddings = model.bert.embeddings
        floats("embedding_word", embeddings.word_embeddings(batch["input_ids"]))
        floats("embedding_type", embeddings.token_type_embeddings(batch["token_type_ids"]))
        floats("embedding_position", embeddings.position_embeddings(torch.arange(256).reshape(1, -1)))
        floats("embedding_ln", torch.cat((embeddings.LayerNorm.weight, embeddings.LayerNorm.bias)))
        mask = (1.0 - batch["attention_mask"].float()) * torch.finfo(torch.float32).min
        floats("key_mask", mask)
        for layer, block in enumerate(model.bert.encoder.layer):
            floats(f"layer{layer}_attention_ln", torch.cat((block.attention.output.LayerNorm.weight,
                                                           block.attention.output.LayerNorm.bias)))
            floats(f"layer{layer}_output_ln", torch.cat((block.output.LayerNorm.weight, block.output.LayerNorm.bias)))
        floats("pooler_weight", model.bert.pooler.dense.weight)
        floats("pooler_bias", model.bert.pooler.dense.bias)
        floats("classifier_weight", model.classifier.weight)
        floats("classifier_bias", model.classifier.bias)
    floats("threshold", torch.tensor([threshold], dtype=torch.float32))
    if struct.unpack("<f", (OUT / "threshold.bin").read_bytes())[0] != threshold:
        raise ValueError("Stored search threshold is not representable exactly as FP32.")

    handles = []
    def capture_output(name):
        def hook(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            floats(name, tensor)
        return hook
    handles.append(model.bert.embeddings.LayerNorm.register_forward_hook(capture_output("reference_embedding_ln")))
    for layer, block in enumerate(model.bert.encoder.layer):
        handles.append(block.register_forward_hook(capture_output(f"reference_layer{layer}")))
    for _, path, module in encoder_linears(model):
        name = linear_names[path]
        def capture_input(mod, inputs, name=name):
            codes(name + "_reference_a8", quantize_codes(inputs[0], mod.input_scale, 8))
        handles.append(module.register_forward_pre_hook(capture_input))
    paused()
    try:
        # Existing software path and attention backend are intentionally retained.
        scored = score_documents(model, tokenizer, [document], config=config, batch_size=1, device="cpu")
    finally:
        for handle in handles:
            handle.remove()
    logits = scored["windows"][0]["logits"]
    risk = scored["documents"][0]["risk_score"]
    floats("reference_logits", torch.tensor(logits))
    floats("reference_risk", torch.tensor([risk]))
    # One additional CPU arithmetic-path check uses exactly the same padded input.
    paused()
    with torch.inference_mode():
        set_execution_mode(model, "integer_reference")
        integer_logits = model(**batch).logits
    set_execution_mode(model, "ste_emulation")
    same_integer_logits = torch.equal(integer_logits, torch.tensor([logits]))
    if not same_integer_logits:
        raise AssertionError("CPU integer reference differs from current W8 emulation on the selected input.")
    manifest = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(), "case": CASE,
        "source_file": "data/prepared/search_candidate_scoring.jsonl", "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "format": quantization["scheme_id"], "weight_scales": "saved frozen weight_scale_8; never regenerated from QAT weights",
        "input_scales": "saved frozen input_scale; no new calibration",
        "hardware_source_revision": revision.group(1), "shape": {"length": 256, "layers": 4, "hidden": 256, "heads": 4, "ffn": 1024},
        "window_count": 1, "total_unpadded_tokens": 177, "window_policy": config["windowing"],
        "input_ids": batch["input_ids"].tolist()[0], "attention_mask": batch["attention_mask"].tolist()[0],
        "token_type_ids": batch["token_type_ids"].tolist()[0], "runtime": runtime,
        "cpu_reference": {"logits": logits, "risk": risk, "threshold": threshold,
                          "strict_reject": risk > threshold, "integer_reference_logits": integer_logits.tolist()[0],
                          "integer_reference_exact": same_integer_logits,
                          "attention_implementation": getattr(model.config, "_attn_implementation", None)},
        "saved_gpu_context": {"window": saved_window, "document": saved_document,
                              "source": "results/goal3/qat_w8_a8/search_candidate_scoring_*"},
        "tolerance_policy": "No end-to-end score/logit tolerance invented. Record deviations and strict decision agreement; operator tolerances do not certify detector equivalence.",
        "scope": "One numerical bring-up; selected embedding-row gathers are exported from software, not a validated FPGA gather engine.",
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"export_complete": True, "case": CASE, "files": len(files),
                      "cpu_reference": manifest["cpu_reference"], "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()

