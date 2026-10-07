"""Run the prescribed small Goal 3 signed-arithmetic and STE check on CPU."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "reports" / "quant_arithmetic_check.json"))
    args = parser.parse_args()
    import torch
    from torch import nn
    from gate_quant import (GateQuantLinear, configure_exact_fp32, format_record,
                            int_linear_reference, quantize_codes, w8_atomic_reference)

    torch.set_num_threads(2)
    settings = configure_exact_fp32()
    checks = []

    # Hand-calculable tie-to-even, signed clipping and zero-point tests.
    linear = nn.Linear(4, 2, bias=True)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[-8., -1., 0., 7.], [7., -8., 1., -1.]]))
        linear.bias.copy_(torch.tensor([0.5, -0.25]))
    module = GateQuantLinear(linear, input_scale=1.0, bits=4)
    values = torch.tensor([-200., -128.5, -128., -127.5, -2.5, -1.5, -0.5,
                           0.5, 1.5, 2.5, 126.5, 127., 127.5, 200.])
    expected8 = torch.tensor([-128, -128, -128, -128, -2, -2, 0, 0, 2, 2, 126, 127, 127, 127])
    got8 = quantize_codes(values, torch.tensor(1.), 8)
    brevitas8 = module.quantizer.to_int(torch.tensor(1.), torch.tensor(0.), torch.tensor(8.), values)
    assert torch.equal(got8, expected8)
    assert torch.equal(brevitas8.to(torch.int64), expected8)
    values4 = torch.tensor([-9., -8.5, -8., -7.5, -1.5, -0.5, 0.5, 1.5, 6.5, 7., 7.5, 8.])
    expected4 = torch.tensor([-8, -8, -8, -8, -2, 0, 0, 2, 6, 7, 7, 7])
    assert torch.equal(quantize_codes(values4, torch.tensor(1.), 4), expected4)
    assert torch.equal(module.quantizer.to_int(torch.tensor(1.), torch.tensor(0.),
                                               torch.tensor(4.), values4).to(torch.int64), expected4)
    checks.append({"name": "signed_limits_rounding_saturation", "passed": True,
                   "w4_expected": expected4.tolist(), "a8_expected": expected8.tolist()})

    a = torch.tensor([[-128, -1, 0, 127], [127, 1, -128, -1]], dtype=torch.int64)
    w4 = torch.tensor([[-8, -1, 0, 7], [7, -8, 1, -1]], dtype=torch.int64)
    w8 = torch.tensor([[-128, -1, 0, 127], [127, -128, 1, -1]], dtype=torch.int64)
    expected_dot4 = torch.tensor([[1914, -1015], [-1024, 754]], dtype=torch.int32)
    expected_dot8 = torch.tensor([[32514, -16255], [-16384, 15874]], dtype=torch.int32)
    assert torch.equal(int_linear_reference(a, w4, 4), expected_dot4)
    assert torch.equal(int_linear_reference(a, w8, 8), expected_dot8)
    assert torch.equal(torch.nn.functional.linear(a.float(), w4.float()).to(torch.int32), expected_dot4)
    assert torch.equal(torch.nn.functional.linear(a.float(), w8.float()).to(torch.int32), expected_dot8)
    checks.append({"name": "hand_calculated_signed_dot_products", "passed": True,
                   "w4_expected": expected_dot4.tolist(), "w8_expected": expected_dot8.tolist()})

    # Every W8 value, including -128, -1, 0 and 127; low nibble must be unsigned.
    all_weights = torch.arange(-128, 128, dtype=torch.int64).reshape(256, 1)
    representative_inputs = torch.tensor([-128, -1, 0, 1, 127], dtype=torch.int64).reshape(5, 1)
    atomic = w8_atomic_reference(representative_inputs, all_weights)
    full = int_linear_reference(representative_inputs, all_weights, 8).to(torch.int64)
    assert torch.equal(atomic, full)
    checks.append({"name": "signed_high_unsigned_low_w8_decomposition", "passed": True,
                   "w8_codes": 256, "activation_values": 5})

    limits = torch.full((1, 1024), -128, dtype=torch.int64)
    exact = int_linear_reference(limits, limits, 8)
    emulated = torch.nn.functional.linear(limits.float(), limits.float())
    assert exact.item() == (1 << 24)
    assert emulated.item() == exact.item()
    overflow_guard = False
    try:
        GateQuantLinear(nn.Linear(1025, 1, bias=False), 1., 8)
    except ValueError:
        overflow_guard = True
    assert overflow_guard
    checks.append({"name": "K1024_exact_accumulator_bound_and_guard", "passed": True,
                   "maximum_absolute_sum": 1 << 24, "declared_accumulator": "int32"})

    # Same frozen scales and operation order in Brevitas STE and CPU reference.
    x = torch.tensor([[-2.5, -1., 0.5, 3.], [0.25, 2.5, -1.5, 1.]], requires_grad=True)
    scale4 = module.weight_scale_4.detach().clone()
    scale8 = module.weight_scale_8.detach().clone()
    max_difference = 0.
    for bits in (4, 8):
        module.set_bits(bits)
        module.execution_mode = "ste_emulation"
        emulated = module(x)
        module.execution_mode = "integer_reference"
        integer = module(x.detach())
        assert torch.equal(emulated.detach(), integer)
        max_difference = max(max_difference, float((emulated.detach() - integer.detach()).abs().max()))
    module.execution_mode = "ste_emulation"
    optimizer = torch.optim.SGD(module.parameters(), lr=0.001)
    module(x).square().mean().backward()
    assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all()
    assert float(module.weight.grad.abs().sum()) > 0
    assert x.grad is not None and torch.isfinite(x.grad).all()
    optimizer.step()
    assert torch.equal(scale4, module.weight_scale_4) and torch.equal(scale8, module.weight_scale_8)
    checks.append({"name": "emulation_reference_equality_and_fixed_scale_STE_step", "passed": True,
                   "max_absolute_difference": max_difference})

    record = {"scope": "small prescribed CPU arithmetic check; no detector quality or hardware timing",
              "device": "CPU", "torch": torch.__version__, "settings": settings,
              "format": format_record(), "checks": checks, "passed": all(c["passed"] for c in checks),
              "command": "python scripts/check_quant_arithmetic.py"}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": record["passed"], "checks": len(checks), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
