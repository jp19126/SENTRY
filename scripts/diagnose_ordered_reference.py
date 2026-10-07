"""One saved-input CPU ordered-FP32 diagnostic; not a model adapter or acceptance test.

Reads the existing 97 numerical-bridge exports and saved C-service outputs only.
No checkpoint/dataset load, network, training, calibration or vendor invocation.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/goal4/numerical_bridge'
DIAGNOSTIC = DATA / 'diagnostic_first_divergence_v1'
OUT = DATA / 'ordered_torch_v1'
CASE = 'aeslc:train:panus-s_inbox_1.subject:clean'
REVISION = 'panel_cache_batched_attention_v1'
L, D, H, HD, FF, CAP = 256, 256, 4, 64, 1024, 32768
DOT, QUANT, RESCALE, EMBED, RESIDUAL, LN, SOFTMAX, GELU, TANH = range(9)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def paused():
    if (ROOT / 'results/goal2/pause.request').exists():
        raise RuntimeError('User pause marker present.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'execute': False, 'case': CASE, 'input': str(DATA),
                          'output': str(OUT), 'device': 'cpu', 'new_exports': 0}, indent=2))
        return
    paused()
    if OUT.exists():
        raise RuntimeError('Preserve the existing ordered diagnostic; no overwrite.')
    manifest = read_json(DATA / 'manifest.json')
    bridge = read_json(DATA / 'bridge_summary.json')
    if (manifest['case'] != CASE or manifest['hardware_source_revision'] != REVISION
            or manifest['shape'] != dict(length=L, layers=4, hidden=D, heads=H, ffn=FF)
            or manifest['checkpoint'].replace('\\', '/') != 'checkpoints/quantized/qat_w8_a8'
            or not bridge['completed'] or bridge['source_revision'] != REVISION):
        raise RuntimeError('Fixed saved-input/source association differs.')
    entries = {row['file']: row for row in manifest['files']}
    if len(entries) != 97 or len(manifest['files']) != 97:
        raise RuntimeError('Expected exactly the existing 97 exports.')
    header = (DATA / 'fixed_fp32_service.hpp').read_text(encoding='utf-8')
    if '#define GATE_FIXED_FP32_SCHEDULE_REVISION "' + REVISION + '"' not in header:
        raise RuntimeError('Saved service revision differs.')
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    denormal_control_supported = torch.set_flush_denormal(False)
    OUT.mkdir()
    shutil.copyfile(Path(__file__), OUT / Path(__file__).name)
    calls = [0] * 9
    call_shapes = []
    integer_layers, boundaries = [], {}
    total_accumulators = 0
    total_quantized = 0
    loaded_exports = set()
    original_integer = {row['name']: row for row in bridge['integer_layers']}

    def f32(value):
        return torch.tensor(value, dtype=torch.float32, device='cpu')

    def finite(value, name):
        if value.dtype != torch.float32 or value.device.type != 'cpu' or not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite/non-CPU-FP32 ' + name)

    def binary(path, count, dtype='float32_le'):
        np_type = '<f4' if dtype == 'float32_le' else 'i1'
        raw = path.read_bytes()
        if len(raw) != count * (4 if dtype == 'float32_le' else 1):
            raise ValueError('Saved binary extent differs: ' + str(path))
        value = torch.from_numpy(np.frombuffer(raw, dtype=np_type).copy())
        if dtype == 'float32_le':
            finite(value, path.name)
        return value

    def load(name, shape):
        entry = entries[name + '.bin']
        count = int(np.prod(shape))
        if count != entry['elements']:
            raise ValueError('Manifest extent differs: ' + name)
        loaded_exports.add(name + '.bin')
        return binary(DATA / (name + '.bin'), count, entry['dtype']).reshape(shape)

    def saved(name, shape, directory=DATA):
        return binary(directory / (name + '.bin'), int(np.prod(shape))).reshape(shape)

    def write(name, value):
        array = value.detach().contiguous().numpy()
        (OUT / (name + '.bin')).write_bytes(array.tobytes())

    def compare(actual, expected):
        a, b = actual.contiguous(), expected.contiguous()
        delta = (a.double() - b.double()).abs().reshape(-1)
        return {'values': a.numel(), 'bit_differences': int((a.view(torch.int32) != b.view(torch.int32)).sum()),
                'maximum_absolute_error': float(delta.max()), 'maximum_error_flat_index': int(delta.argmax()),
                'mean_absolute_error': float(delta.mean()), 'rms_error': float(delta.square().mean().sqrt())}

    def record(name, actual):
        reference = load('reference_' + name, actual.shape)
        service = saved('bridge_' + name, actual.shape)
        boundaries[name] = {'versus_original_torch': compare(actual, reference),
                            'versus_c_service': compare(actual, service)}
        write('ordered_' + name, actual)
        print('ORDERED_BOUNDARY', name, json.dumps(boundaries[name]), flush=True)

    def quant_codes(x, scale):
        quotient = torch.div(x, scale)
        if bool(torch.isnan(quotient).any()):
            raise ValueError('NaN quantization quotient.')
        # Clip before bounded floor/cast; exact service nearest-even behavior.
        bounded = torch.clamp(quotient, -128.0, 127.0)
        lower = torch.floor(bounded)
        integer = lower.to(torch.int32)
        fraction = torch.sub(bounded, lower)
        increment = torch.logical_or(fraction > f32(.5),
                                     torch.logical_and(fraction == f32(.5), integer.remainder(2) != 0))
        codes = (integer + increment.to(torch.int32)).to(torch.int8)
        if not torch.equal(codes, torch.clamp(torch.round(quotient), -128, 127).to(torch.int8)):
            raise AssertionError('Same-input nearest-even rules disagree.')
        return codes

    def fp(op, x, y=None, z=None, scale=None, bias=False):
        paused()
        rows, width = x.shape
        if not (0 < rows <= 512 and 0 < width <= 1024 and rows * width <= CAP):
            raise ValueError('Service array extent differs.')
        outputs = y.shape[0] if op == DOT else 0
        if op == DOT and not (0 < outputs <= 64 and width <= 512 and outputs * width <= CAP and rows * outputs <= CAP):
            raise ValueError('DOT extent differs.')
        if op == SOFTMAX and width > 512:
            raise ValueError('Softmax extent differs.')
        if op == LN and width != D:
            raise ValueError('LayerNorm width differs.')
        if op != RESCALE:
            finite(x, 'service input')
        if scale is None:
            scale = f32(1)
        if op in (QUANT, RESCALE, SOFTMAX) and (not bool(torch.isfinite(scale)) or float(scale) <= 0):
            raise ValueError('Invalid scale.')
        calls[op] += 1
        call_shapes.append([op, rows, width, outputs])
        if op == DOT:
            result = torch.zeros((rows, outputs), dtype=torch.float32)
            for k in range(width):
                product = torch.mul(x[:, k:k+1], y[:, k].reshape(1, outputs))
                result = torch.add(result, product)
            if bias:
                result = torch.add(result, z.reshape(1, outputs))
        elif op == QUANT:
            return quant_codes(x, scale)
        elif op == RESCALE:
            factor = torch.mul(scale, y)
            if not bool((factor > 0).all()):
                raise ValueError('Invalid combined scale.')
            product = torch.mul(x.to(torch.float32), factor.reshape(1, width))
            result = torch.add(product, z.reshape(1, width))
        elif op == EMBED:
            first = torch.add(x, y)
            result = torch.add(first, z)
        elif op == RESIDUAL:
            result = torch.add(x, y)
        elif op == LN:
            total = torch.zeros(rows, dtype=torch.float32)
            for k in range(width):
                total = torch.add(total, x[:, k])
            mean = torch.div(total, f32(256))
            square_sum = torch.zeros(rows, dtype=torch.float32)
            for k in range(width):
                centered = torch.add(x[:, k], torch.neg(mean))
                product = torch.mul(centered, centered)
                square_sum = torch.add(square_sum, product)
            variance = torch.div(square_sum, f32(256))
            denominator = torch.sqrt(torch.add(variance, f32(1e-12)))
            centered = torch.add(x, torch.neg(mean.reshape(rows, 1)))
            normalized = torch.div(centered, denominator.reshape(rows, 1))
            product = torch.mul(normalized, y.reshape(1, width))
            result = torch.add(product, z.reshape(1, width))
        elif op == SOFTMAX:
            divided = torch.div(x, scale)
            scores = torch.add(divided, y)
            finite(scores, 'softmax scaled scores')
            maximum = torch.full((rows,), -torch.finfo(torch.float32).max)
            for k in range(width):
                maximum = torch.where(scores[:, k] > maximum, scores[:, k], maximum)
            exponentials = torch.exp(torch.add(scores, torch.neg(maximum.reshape(rows, 1))))
            total = torch.zeros(rows, dtype=torch.float32)
            for k in range(width):
                total = torch.add(total, exponentials[:, k])
            if not bool((total > 0).all()):
                raise ValueError('Invalid softmax sum.')
            result = torch.div(exponentials, total.reshape(rows, 1))
        elif op == GELU:
            argument = torch.div(x, f32(1.4142135623730951))
            half = torch.mul(x, f32(.5))
            nonlinear = torch.erf(argument)
            result = torch.mul(half, torch.add(f32(1), nonlinear))
        elif op == TANH:
            result = torch.tanh(x)
        else:
            raise ValueError(op)
        finite(result, 'service output')
        return result

    def elements(op, x, y=None, z=None, per_channel=False, scale=None):
        rows, width = x.shape
        blocks = []
        for start in range(0, rows, CAP // width):
            stop = min(rows, start + CAP // width)
            yy = y if y is None or per_channel else y[start:stop]
            zz = z if z is None or per_channel else z[start:stop]
            blocks.append(fp(op, x[start:stop], yy, zz, scale))
        return torch.cat(blocks, dim=0)

    def norm(x, name):
        parameters = load(name, (2 * D,))
        return elements(LN, x, parameters[:D], parameters[D:], True)

    def service_input(name, k):
        prefix, suffix = name.split('_', 1)
        layer = int(prefix[5:])
        if suffix in ('query', 'key', 'value'):
            boundary = 'embedding_ln' if layer == 0 else 'layer' + str(layer - 1)
            return saved('bridge_' + boundary, (L, k))
        if name == 'layer1_attention_output':
            return saved('diagnostic_layer1_attention_context', (L, k), DIAGNOSTIC)
        return None  # Other service pre-quantization arrays were not saved.

    def linear(name, x, k, n):
        nonlocal total_accumulators, total_quantized
        weights = load(name + '_w8', (n, k))
        parameters = load(name + '_params', (1 + 2 * n,))
        codes = elements(QUANT, x, scale=parameters[0])
        original = load(name + '_reference_a8', (L, k))
        total_quantized += codes.numel()
        # Products and every possible partial sum are exact FP32 integers under
        # this absolute-sum bound. Still compare all results to independent int64.
        bound = k * int(codes.to(torch.int32).abs().max()) * int(weights.to(torch.int32).abs().max())
        if bound > 2**24:
            raise ValueError('FP32 integer dot exactness bound exceeded.')
        float_acc = torch.mm(codes.float(), weights.float().T)
        exact_acc = torch.mm(codes.to(torch.int64), weights.to(torch.int64).T)
        if not torch.equal(float_acc.to(torch.int64), exact_acc):
            raise AssertionError('Integer FP32 products disagree with int64 reference: ' + name)
        if bool((exact_acc < -2**31).any()) or bool((exact_acc >= 2**31).any()):
            raise AssertionError('Integer accumulator overflow.')
        total_accumulators += exact_acc.numel()
        result = elements(RESCALE, exact_acc.to(torch.int32), parameters[1:1+n], parameters[1+n:], True, parameters[0])
        service_x = service_input(name, k)
        service_codes = None if service_x is None else quant_codes(service_x, parameters[0])
        row = {'name': name, 'input_codes': codes.numel(),
               'original_software_code_differences': int((codes != original).sum()),
               'versus_torch_quantized_saved_service_input_code_differences': None if service_codes is None else int((codes != service_codes).sum()),
               'saved_service_input_comparison_available': service_codes is not None,
               'service_code_basis': None if service_codes is None else 'Torch quantization of saved C-service inputs; division identity with HLS unverified',
               'service_reported_original_software_code_differences': original_integer[name]['original_software_code_differences'],
               'integer_absolute_sum_bound': bound, 'accumulators_checked_exactly': exact_acc.numel()}
        integer_layers.append(row)
        write(name + '_ordered_a8', codes)
        if name in ('layer1_query', 'layer1_key', 'layer1_value'):
            service_y = saved('diagnostic_' + name + '_output', (L, n), DIAGNOSTIC)
            boundaries[name + '_output'] = {'versus_c_service': compare(result, service_y)}
            write(name + '_ordered_output', result)
        print('ORDERED_LINEAR', json.dumps(row), flush=True)
        return result

    def attention(q, k, v, key_mask):
        context = torch.empty((L, D), dtype=torch.float32)
        for head in range(H):
            keys = k[:, head*HD:(head+1)*HD].contiguous()
            transposed_v = v[:, head*HD:(head+1)*HD].T.contiguous()
            for start in range(0, L, CAP // L):
                stop = min(L, start + CAP // L)
                queries = q[start:stop, head*HD:(head+1)*HD].contiguous()
                score_tiles = [fp(DOT, queries, keys[key:key+64]) for key in range(0, L, 64)]
                scores = torch.cat(score_tiles, dim=1)
                mask = key_mask.reshape(1, L).expand(stop-start, L).contiguous()
                probability = fp(SOFTMAX, scores, mask, scale=f32(8))
                context[start:stop, head*HD:(head+1)*HD] = fp(DOT, probability, transposed_v)
        return context

    with torch.inference_mode():
        word, token_type, position = [load('embedding_' + name, (L, D)) for name in ('word', 'type', 'position')]
        mask = load('key_mask', (L,))
        hidden = norm(elements(EMBED, word, token_type, position), 'embedding_ln')
        record('embedding_ln', hidden)
        context_summary = None
        for layer in range(4):
            prefix = 'layer' + str(layer)
            q, k, v = [linear(prefix + '_' + name, hidden, D, D) for name in ('query', 'key', 'value')]
            context = attention(q, k, v, mask)
            if layer == 1:
                service_context = saved('diagnostic_layer1_attention_context', (L, D), DIAGNOSTIC)
                original_context = saved('analysis_torch_layer1_attention_context', (L, D), DIAGNOSTIC)
                scale = load('layer1_attention_output_params', (1 + 2 * D,))[0]
                index = (131, 114)
                context_summary = {'versus_c_service': compare(context, service_context),
                                   'versus_original_sdpa': compare(context, original_context),
                                   'known_boundary': {'token': 131, 'channel': 114, 'scale': float(scale)}}
                for label, values in [('ordered', context), ('c_service', service_context), ('original_sdpa', original_context)]:
                    context_summary['known_boundary'][label] = {'input': float(values[index]),
                        'quotient': float(torch.div(values[index], scale)), 'code': int(quant_codes(values, scale)[index])}
                write('ordered_layer1_attention_context', context)
            projected = linear(prefix + '_attention_output', context, D, D)
            normalized = norm(elements(RESIDUAL, projected, hidden), prefix + '_attention_ln')
            intermediate = linear(prefix + '_ffn_input', normalized, D, FF)
            activated = elements(GELU, intermediate)
            output = linear(prefix + '_ffn_output', activated, FF, D)
            hidden = norm(elements(RESIDUAL, output, normalized), prefix + '_output_ln')
            record(prefix, hidden)
        pool_weights, pool_bias = load('pooler_weight', (D, D)), load('pooler_bias', (D,))
        pool_tiles = [fp(DOT, hidden[:1], pool_weights[o:o+64], pool_bias[o:o+64], bias=True) for o in range(0, D, 64)]
        pooled = fp(TANH, torch.cat(pool_tiles, dim=1))
        logits = fp(DOT, pooled, load('classifier_weight', (2, D)), load('classifier_bias', (2,)), bias=True)
        record('logits', logits)
        probabilities = fp(SOFTMAX, logits, torch.zeros_like(logits))
        risk = probabilities[:, 1]
        record('risk', risk)
        threshold = float(load('threshold', (1,))[0])
    if calls != bridge['service_calls_by_op'] or sum(calls) != 411 or len(integer_layers) != 24:
        raise AssertionError('Actual service segmentation/count differs.')
    if loaded_exports != set(entries):
        raise AssertionError('Export coverage differs: ' + str(set(entries) - loaded_exports))
    summary = {
        'completed': True, 'pre_execution_review': 'baseline_research: no arithmetic blocker; clarify reconstructed service codes and Torch-only quantizer crosscheck', 'recorded_utc': datetime.now(timezone.utc).isoformat(), 'case': CASE,
        'arithmetic_identity': 'ordered_cpu_eager_fp32_saved_service_order_v1',
        'saved_service_revision': REVISION, 'source_exports': str(DATA.relative_to(ROOT)),
        'torch_version': torch.__version__, 'numpy_version': np.__version__, 'python_version': sys.version,
        'cpu_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
        'flush_denormal': False, 'denormal_control_supported': denormal_control_supported,
        'device': 'cpu', 'autocast': False, 'tf32': False, 'graph_compilation': False,
        'reduction_order': 'ascending k, separate FP32 multiply/add tensors; independent output dimensions batched',
        'nonlinear_identity': 'Torch CPU exp/erf/tanh/sqrt/div; unverified versus HLS C library and RTL IP',
        'integer_identity': 'bounded FP32 code-only matmul checked elementwise against int64 matmul',
        'export_file_count': len(loaded_exports), 'service_calls_by_op': calls, 'total_service_calls': sum(calls),
        'actual_call_shapes': call_shapes, 'torch_nearest_even_crosscheck_values': total_quantized,
        'quantizer_check_identity': 'Two Torch nearest-even implementations on the same Torch FP32 quotient; not observed HLS A8 output',
        'integer_accumulators_checked': total_accumulators, 'integer_layers': integer_layers,
        'boundaries': boundaries, 'layer1_attention_context': context_summary,
        'logits': logits.reshape(-1).tolist(), 'risk': float(risk[0]), 'threshold': threshold,
        'strict_reject': float(risk[0]) > threshold, 'c_service_strict_reject': bridge['strict_reject'],
        'original_torch_strict_reject': manifest['cpu_reference']['strict_reject'],
        'end_to_end_tolerance': None, 'numerical_acceptance': None,
        'scope': 'One saved development input, not dataset scoring, FPGA timing or detector validation.'}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print('ORDERED_REFERENCE_COMPLETE', json.dumps({'output': str(OUT), 'calls': sum(calls),
          'logits': summary['logits'], 'risk': summary['risk'], 'numerical_acceptance': None}), flush=True)


if __name__ == '__main__':
    main()
