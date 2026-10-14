"""Focused CPU contract checks; no datasets, accelerator jobs or learning fits.

Run from the repository: python -B scripts/check_cnn_models.py [--output PATH].
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from cnn_models import FAMILIES, MaskedConv, build_model, load_search_configs, model_cost


def check(config):
    torch.manual_seed(173)
    model = build_model(config, vocab_size=257, pad_id=0).cpu().eval()
    ids = torch.randint(1, 257, (3, 17))
    mask = torch.ones_like(ids)
    mask[1, 8:] = 0
    mask[2] = 0
    # Internal holes distinguish mask propagation from mere length cropping.
    mask[0, 5:7] = 0
    observations = []
    def inspect(module, inputs, output):
        x, m = output
        assert torch.isfinite(x).all(), config["id"]
        assert torch.count_nonzero(x.masked_select(~m.expand_as(x))) == 0, config["id"]
        observations.append(1)
    handles = [m.register_forward_hook(inspect) for m in model.modules() if isinstance(m, MaskedConv)]
    try:
        with torch.no_grad():
            logits = model(ids, mask, token_type_ids=torch.zeros_like(ids)).logits
            assert logits.shape == (3, 2)
            assert torch.isfinite(logits).all()
            changed = ids.clone()
            changed[mask == 0] = 256
            torch.testing.assert_close(model(changed, mask).logits, logits, atol=2e-6, rtol=2e-6)
            for extension in (1, 8, 19):
                padded = F.pad(ids, (0, extension), value=256)
                padded_mask = F.pad(mask, (0, extension))
                torch.testing.assert_close(model(padded, padded_mask).logits, logits, atol=3e-6, rtol=3e-6)
            # A mixed-length batch must agree with isolated rows.
            for row, length in ((0, 17), (1, 8), (2, 0)):
                individual = model(ids[row:row+1, :length], mask[row:row+1, :length]).logits
                torch.testing.assert_close(individual, logits[row:row+1], atol=1e-5, rtol=1e-5,
                                           msg=f"{config['id']} isolated row {row}")
            empty = model(torch.empty(2, 0, dtype=torch.long)).logits
            torch.testing.assert_close(empty, model.classifier.bias.unsqueeze(0).expand(2, -1))
            assert torch.isfinite(empty).all()
            inferred_ids = ids.masked_fill(mask == 0, 0)
            torch.testing.assert_close(model(inferred_ids).logits, logits, atol=3e-6, rtol=3e-6)
            if config["family"] == "tcn" and config["direction"] == "causal":
                full_x, _ = model.forward_features(ids[:1], torch.ones_like(mask[:1]))
                prefix_x, _ = model.forward_features(ids[:1, :7], torch.ones_like(mask[:1, :7]))
                torch.testing.assert_close(full_x[:, :, :7], prefix_x, atol=1e-5, rtol=1e-5)
        model.train()
        loss = F.cross_entropy(model(ids, mask).logits, torch.tensor([1, 0, 0]))
        loss.backward()
        missing = [name for name, p in model.named_parameters() if p.grad is None]
        assert not missing, (config["id"], missing)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        assert torch.count_nonzero(model.embedding.weight.grad[0]) == 0
        # An actual parameter update must remain finite, including empty row.
        torch.optim.AdamW(model.parameters(), lr=1e-4).step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert observations
    finally:
        for handle in handles:
            handle.remove()
    costs = model_cost(model, sequence_length=17, batch_size=3)
    assert costs["conv_linear_macs"] > 0
    return {"id": config["id"], "family": config["family"], "passed": True,
            "masked_conv_calls_checked": len(observations), "parameters": costs["parameters"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", action="append", help="Check selected IDs; default all configurations")
    args = parser.parse_args()
    torch.set_num_threads(1)
    start = time.perf_counter()
    configs = load_search_configs()
    assert len({c["family"] for c in configs}) == len(FAMILIES)
    if args.config:
        wanted = set(args.config)
        configs = [c for c in configs if c["id"] in wanted]
        assert {c["id"] for c in configs} == wanted
    results = [check(c) for c in configs]
    anchor_equivalence = None
    if any(c["id"] == "textcnn_01" for c in configs):
        from r1_models import TokenCNN
        torch.manual_seed(73)
        reference = TokenCNN(257, 0).eval()
        anchor = build_model(next(c for c in configs if c["id"] == "textcnn_01"), 257, 0).eval()
        anchor.embedding.load_state_dict(reference.embedding.state_dict())
        for target, source in zip(anchor.blocks, reference.convs):
            target.conv.load_state_dict(source.state_dict())
        anchor.classifier.load_state_dict(reference.classifier.state_dict())
        ids = torch.randint(1, 257, (2, 23))
        mask = torch.ones_like(ids)
        mask[1, 11:] = 0
        with torch.no_grad():
            torch.testing.assert_close(anchor(ids, mask).logits, reference(ids, mask).logits,
                                       atol=1e-6, rtol=1e-6)
        anchor_equivalence = "same-weight R1 TokenCNN nonempty inputs pass atol/rtol 1e-6"
    report = {"purpose": "synthetic CPU implementation checks, not trained quality evidence",
              "torch": torch.__version__, "device": "cpu", "threads": 1,
              "elapsed_seconds": time.perf_counter() - start,
              "configuration_count": len(results), "family_count": len({c["family"] for c in configs}),
              "anchor_equivalence": anchor_equivalence,
              "tolerances": {"masked_values": [2e-6, 2e-6], "padding_extent": [3e-6, 3e-6],
                             "isolated_batch_and_causal_prefix": [1e-5, 1e-5]},
              "checks": ["finite binary logits", "masked value invariance", "right-padding extent invariance",
                         "isolated versus mixed-length batch", "internal masked holes", "zero-length input",
                         "pad-inferred mask", "masked intermediate convolution outputs", "causal prefix independence",
                         "finite all-parameter gradients", "zero padding embedding gradient", "finite AdamW step"],
              "results": results}
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}))


if __name__ == "__main__":
    main()
