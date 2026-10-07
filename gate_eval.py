"""Goal 1 document-level windowing and evaluation shared by later experiments.

Only the external document is tokenized. Character spans are half-open [start,end)
positions in that exact text. No question, reference answer, or source label enters
model inputs. The class-1 softmax probability is the window risk.
"""
from __future__ import annotations

import json
import math
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent


def load_config() -> dict[str, Any]:
    """Resolve the shared config relative to this repository, never the shell CWD."""
    return json.loads((ROOT / "configs" / "project.json").read_text(encoding="utf-8-sig"))


def _finite(values: Iterable[float]) -> list[float]:
    values = [float(value) for value in values]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Scores must be finite.")
    return values


def fit_threshold(benign_scores: Sequence[float], alpha: float) -> float:
    """Ascending benign score at 1-based n-floor(alpha*n); attack iff score > t.

    Decimal multiplication preserves the stated decimal operating point at integer
    boundaries. Ties at the threshold are benign and can make observed FPR smaller.
    """
    scores = sorted(_finite(benign_scores))
    if not scores:
        raise ValueError("Threshold fitting needs at least one benign document.")
    if not math.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0, 1).")
    k = int((Decimal(str(alpha)) * len(scores)).to_integral_value(rounding=ROUND_FLOOR))
    return scores[len(scores) - k - 1]


def evaluate_documents(
    scores: Sequence[float], labels: Sequence[int], threshold: float
) -> dict[str, Any]:
    """Compute document confusion counts; undefined rates remain null (None)."""
    scores = _finite(scores)
    labels = list(labels)
    if len(scores) != len(labels):
        raise ValueError("Exactly one score and binary label per document is required.")
    if not math.isfinite(threshold) or any(label not in (0, 1) for label in labels):
        raise ValueError("Threshold must be finite and labels must be 0 or 1.")
    tp = sum(label == 1 and score > threshold for label, score in zip(labels, scores))
    fn = sum(label == 1 and score <= threshold for label, score in zip(labels, scores))
    fp = sum(label == 0 and score > threshold for label, score in zip(labels, scores))
    tn = sum(label == 0 and score <= threshold for label, score in zip(labels, scores))
    return {
        "level": "document", "threshold": float(threshold), "attack_rule": "score > threshold",
        "documents": len(labels), "attacks": tp + fn, "benign": fp + tn,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": tp / (tp + fn) if tp + fn else None,
        "fpr": fp / (fp + tn) if fp + tn else None,
        "accuracy": (tp + tn) / len(labels) if labels else None,
    }


def aggregate_document_risk(
    window_scores: Any, document_indices: Sequence[int], document_count: int
) -> Any:
    """Maximum window risk for each document (indices 0..document_count-1).

    A 1D PyTorch Tensor produces a differentiable Tensor via stack/max, suitable
    for document-level loss when positive injection spans are unavailable. A list
    produces a list. At least one window per document is required in both cases.
    """
    indices = list(document_indices)
    if document_count < 1 or len(window_scores) != len(indices):
        raise ValueError("Need matching scores/indices and at least one document.")
    if any(not isinstance(index, int) or not 0 <= index < document_count for index in indices):
        raise ValueError("Window document index is out of bounds.")
    positions = [[] for _ in range(document_count)]
    for i, index in enumerate(indices):
        positions[index].append(i)
    if any(not group for group in positions):
        raise ValueError("Every document must have at least one window.")
    if hasattr(window_scores, "requires_grad"):
        import torch
        if window_scores.ndim != 1:
            raise ValueError("Window score tensor must be one-dimensional.")
        return torch.stack([window_scores[group].max() for group in positions])
    scores = _finite(window_scores)
    return [max(scores[i] for i in group) for group in positions]


def window_document(
    tokenizer: Any,
    text: str,
    document_id: str,
    label: int | None = None,
    injection_spans: Sequence[Sequence[int]] | None = None,
    config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Cover every content token with overlapping BERT windows, including the tail.

    Length includes [CLS]/[SEP]; overlap counts only content tokens. When spans
    exist, positive labels are assigned only to windows containing tokens whose
    character intervals intersect a span. Without positive spans, all positive
    window labels are null: train using aggregate_document_risk and the document
    label, never by copying the malicious label to unrelated clean windows.
    """
    config = load_config() if config is None else config
    policy = config["windowing"]
    max_length = int(policy["primary_length"])
    overlap = int(policy["content_token_overlap"])
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    capacity = max_length - special_count
    if special_count != 2:
        raise ValueError("This protocol requires BERT single-text [CLS]/[SEP] tokenization.")
    if capacity <= 0 or not 0 <= overlap < capacity:
        raise ValueError("Window length must permit content with overlap smaller than capacity.")
    if label not in (None, 0, 1):
        raise ValueError("Document label must be 0, 1, or None.")
    if not isinstance(text, str):
        raise TypeError("Document text must be a string.")
    spans = [tuple(span) for span in injection_spans] if injection_spans else None
    if spans is not None:
        if label != 1:
            raise ValueError("Known injection spans require a positive document.")
        if any(len(span) != 2 or any(not isinstance(x, int) for x in span)
               or not 0 <= span[0] < span[1] <= len(text) for span in spans):
            raise ValueError("Injection spans must be valid half-open character offsets in the document.")
        if not tokenizer.is_fast:
            raise ValueError("Injection-span supervision requires a fast tokenizer with offsets.")
    encoded = tokenizer(
        text, add_special_tokens=False, truncation=False, return_attention_mask=False,
        return_token_type_ids=False, return_offsets_mapping=bool(tokenizer.is_fast), verbose=False,
    )
    content_ids = list(encoded["input_ids"])
    offsets = encoded.get("offset_mapping")
    if spans is not None:
        for span_start, span_end in spans:
            if not any(end > span_start and start < span_end for start, end in offsets):
                raise ValueError("An injection span contains no tokens; cannot assign trustworthy window labels.")
    windows = []
    start = 0
    while True:
        end = min(start + capacity, len(content_ids))
        content = content_ids[start:end]
        input_ids = tokenizer.build_inputs_with_special_tokens(content)
        types = tokenizer.create_token_type_ids_from_sequences(content)
        if len(input_ids) > max_length:
            raise ValueError("Tokenizer special-token construction exceeded configured window length.")
        window_offsets = offsets[start:end] if offsets is not None else None
        if label == 0:
            window_label = 0
        elif spans is not None:
            window_label = int(any(
                token_end > span_start and token_start < span_end
                for token_start, token_end in window_offsets
                for span_start, span_end in spans
            ))
        else:
            window_label = None
        windows.append({
            "document_id": str(document_id), "window_index": len(windows),
            "input_ids": input_ids, "attention_mask": [1] * len(input_ids),
            "token_type_ids": types, "content_token_start": start, "content_token_end": end,
            "content_token_count": end - start, "total_token_count": len(input_ids),
            "char_start": window_offsets[0][0] if window_offsets else None,
            "char_end": window_offsets[-1][1] if window_offsets else None,
            "document_label": label, "window_label": window_label,
            "supervision": "unlabeled" if label is None else ("document_aggregate" if label == 1 and spans is None else "window"),
        })
        if end == len(content_ids):
            break
        start = end - overlap
    return windows


def score_documents(
    model: Any,
    tokenizer: Any,
    documents: Sequence[Mapping[str, Any]],
    batch_size: int = 8,
    device: Any = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Batched binary inference, returning raw window and aggregated document scores.

    Inputs use document_id (or id), text (or context), label, optional injection_spans.
    The caller owns model loading, model revision, seed and saved experiment records.
    This helper neither calibrates a threshold nor opens another data partition.
    """
    import torch
    config = load_config() if config is None else config
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if int(config["models"]["detector"]["labels"]["injection"]) != 1:
        raise ValueError("This scorer requires class 1 to mean injection.")
    docs = list(documents)
    windows = []
    ids = []
    for index, doc in enumerate(docs):
        doc_id = doc.get("document_id", doc.get("id"))
        text = doc.get("text", doc.get("context"))
        if doc_id is None or text is None:
            raise ValueError("Each document requires document_id/id and text/context.")
        ids.append(str(doc_id))
        current = window_document(tokenizer, text, str(doc_id), doc.get("label"), doc.get("injection_spans"), config)
        for window in current:
            window["document_index"] = index
        windows.extend(current)
    if len(set(ids)) != len(ids):
        raise ValueError("Document IDs must be unique within an inference call.")
    if not docs:
        return {"windows": [], "documents": []}
    device = next(model.parameters()).device if device is None else torch.device(device)
    model.to(device)
    was_training = model.training
    model.eval()
    output_windows = []
    try:
        with torch.inference_mode():
            for start in range(0, len(windows), batch_size):
                current = windows[start:start + batch_size]
                model_inputs = [{name: window[name] for name in ("input_ids", "attention_mask", "token_type_ids")}
                                for window in current]
                batch = tokenizer.pad(model_inputs, padding="max_length", max_length=int(config["windowing"]["primary_length"]), return_tensors="pt")
                batch = {key: tensor.to(device) for key, tensor in batch.items()}
                logits = model(**batch).logits
                if logits.ndim != 2 or logits.shape[1] != 2:
                    raise ValueError("Expected a binary classification head with two logits.")
                if not torch.isfinite(logits).all():
                    raise ValueError("Model produced nonfinite logits.")
                risks = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist()
                raw_logits = logits.float().cpu().tolist()
                for window, risk, pair in zip(current, risks, raw_logits):
                    output_windows.append({
                        key: value for key, value in window.items()
                        if key not in ("input_ids", "attention_mask", "token_type_ids")
                    } | {"risk_score": risk, "logits": pair})
    finally:
        model.train(was_training)
    scores = aggregate_document_risk([window["risk_score"] for window in output_windows],
                                    [window["document_index"] for window in output_windows], len(docs))
    output_docs = []
    grouped_windows = [[] for _ in docs]
    for window in output_windows:
        grouped_windows[window["document_index"]].append(window)
    for index, (doc, score) in enumerate(zip(docs, scores)):
        current = grouped_windows[index]
        output_docs.append({
            "document_id": ids[index], "label": doc.get("label"), "risk_score": score,
            "window_count": len(current), "is_long_document": len(current) > 1,
            "content_token_count": current[-1]["content_token_end"],
            "window_scores": [window["risk_score"] for window in current],
        })
    return {
        "score_definition": "softmax(binary_logits)[1]", "aggregation": "max_window_risk",
        "max_length": int(config["windowing"]["primary_length"]),
        "content_token_overlap": int(config["windowing"]["content_token_overlap"]),
        "windows": output_windows, "documents": output_docs,
    }


