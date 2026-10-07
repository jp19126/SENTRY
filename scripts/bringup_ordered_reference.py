"""Bring up the optional loaded-model ordered path on the SAME saved token IDs.

One existing W8-QAT checkpoint and exported development input only. No tokenizer,
new dataset, model download, training, threshold fitting or hardware execution.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/goal4/numerical_bridge'
ORDERED = DATA / 'ordered_torch_v1'
DIAGNOSTIC = DATA / 'diagnostic_first_divergence_v1'
CASE = 'aeslc:train:panus-s_inbox_1.subject:clean'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def paused():
    if (ROOT / 'results/goal2/pause.request').exists():
        raise RuntimeError('User pause marker present.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    out = DATA / ('ordered_model_v1' if args.device == 'cpu' else 'ordered_model_cuda_v1')
    if not args.execute:
        print(json.dumps({'execute': False, 'case': CASE, 'output': str(out),
                          'validation_shape': [1, 256], 'device': args.device}, indent=2))
        return
    paused()
    if out.exists():
        raise RuntimeError('Preserve existing loaded-model bring-up; no automatic overwrite.')
    manifest, ordered, bridge = read(DATA / 'manifest.json'), read(ORDERED / 'summary.json'), read(DATA / 'bridge_summary.json')
    if (manifest['case'] != CASE or ordered['case'] != CASE or not ordered['completed']
            or manifest['shape'] != dict(length=256, layers=4, hidden=256, heads=4, ffn=1024)
            or manifest['checkpoint'].replace('\\', '/') != 'checkpoints/quantized/qat_w8_a8'):
        raise ValueError('Fixed input/checkpoint association differs.')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from gate_quant import load_quantized_checkpoint, configure_exact_fp32, encoder_linears, quantize_codes
    from gate_fpga_reference import OrderedBertMini, ARITHMETIC_IDENTITY, PRIMITIVE_IDENTITY, SERVICE_REVISION, LIBRARY_CAVEAT
    if (ordered['arithmetic_identity'] != PRIMITIVE_IDENTITY or manifest['hardware_source_revision'] != SERVICE_REVISION
            or bridge['source_revision'] != SERVICE_REVISION):
        raise ValueError('Reference primitive/source identity differs.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        print(json.dumps({'completed': False, 'cuda_available': False, 'setup_changed': False, 'case': CASE}))
        return
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:0')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    configure_exact_fp32()
    torch.set_flush_denormal(False)
    checkpoint = ROOT / 'checkpoints/quantized/qat_w8_a8'
    model, metadata = load_quantized_checkpoint(checkpoint, 'cpu')
    model.eval()
    if set(metadata['precision_map'].values()) != {8}:
        raise ValueError('Bring-up is restricted to the saved uniform W8-QAT model.')
    batch = {name: torch.tensor([manifest[name]], dtype=torch.long)
             for name in ('input_ids', 'attention_mask', 'token_type_ids')}
    if any(value.shape != (1, 256) for value in batch.values()):
        raise ValueError('Saved token IDs/masks must describe exactly B1/L256.')
    entries = {row['file']: row for row in manifest['files']}
    loaded_exports = set()

    def binary(directory, name, shape, integer=False):
        dtype = 'i1' if integer else '<f4'
        raw = (directory / (name + '.bin')).read_bytes()
        if len(raw) != int(np.prod(shape)) * (1 if integer else 4):
            raise ValueError('Saved extent differs: ' + name)
        return torch.from_numpy(np.frombuffer(raw, dtype=dtype).copy()).reshape(shape)

    def exported(name, like):
        entry = entries[name + '.bin']
        if entry['elements'] != like.numel():
            raise ValueError('Manifest extent differs: ' + name)
        loaded_exports.add(name + '.bin')
        return binary(DATA, name, like.shape, entry['dtype'] == 'int8')

    def exact_export(name, tensor):
        expected = exported(name, tensor)
        a, b = tensor.detach().cpu().contiguous(), expected.contiguous()
        exact = torch.equal(a, b) if a.dtype == torch.int8 else torch.equal(a.view(torch.int32), b.view(torch.int32))
        if not exact:
            raise AssertionError('Loaded parameter or input differs from original export: ' + name)

    suffix = {'attention.self.query': 'query', 'attention.self.key': 'key', 'attention.self.value': 'value',
              'attention.output.dense': 'attention_output', 'intermediate.dense': 'ffn_input', 'output.dense': 'ffn_output'}
    modules = []
    original_modes = []
    with torch.inference_mode():
        for _, path, module in encoder_linears(model):
            layer = int(path.split('.')[3])
            name = 'layer' + str(layer) + '_' + suffix['.'.join(path.split('.')[4:])]
            exact_export(name + '_w8', quantize_codes(module.weight, module.weight_scale_8, 8).to(torch.int8))
            exact_export(name + '_params', torch.cat((module.input_scale.reshape(1), module.weight_scale_8.reshape(-1), module.bias.reshape(-1))))
            modules.append((name, module))
            original_modes.append(module.execution_mode)
        embedding = model.bert.embeddings
        positions = torch.arange(256).reshape(1, 256)
        exact_export('embedding_word', embedding.word_embeddings(batch['input_ids']))
        exact_export('embedding_type', embedding.token_type_embeddings(batch['token_type_ids']))
        exact_export('embedding_position', embedding.position_embeddings(positions))
        exact_export('embedding_ln', torch.cat((embedding.LayerNorm.weight, embedding.LayerNorm.bias)))
        for layer, block in enumerate(model.bert.encoder.layer):
            exact_export('layer' + str(layer) + '_attention_ln', torch.cat((block.attention.output.LayerNorm.weight, block.attention.output.LayerNorm.bias)))
            exact_export('layer' + str(layer) + '_output_ln', torch.cat((block.output.LayerNorm.weight, block.output.LayerNorm.bias)))
        for prefix, module in [('pooler', model.bert.pooler.dense), ('classifier', model.classifier)]:
            exact_export(prefix + '_weight', module.weight)
            exact_export(prefix + '_bias', module.bias)
        exact_export('key_mask', (1.0 - batch['attention_mask'].float()) * torch.finfo(torch.float32).min)
    parameter_input_exports_checked = len(loaded_exports)
    model.to(device)
    batch = {name: value.to(device) for name, value in batch.items()}
    wrapper = OrderedBertMini(model)
    device_record = {'device': str(device), 'backend': 'PyTorch eager FP32', 'cuda_runtime': torch.version.cuda}
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        device_record.update(name=properties.name, capability=list(torch.cuda.get_device_capability(device)),
                             total_memory_bytes=properties.total_memory)
    paused()
    out.mkdir()
    for path in (Path(__file__), ROOT / 'gate_fpga_reference.py'):
        shutil.copyfile(path, out / path.name)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    forward_start = time.perf_counter()
    result = wrapper(**batch, capture_intermediates=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    forward_elapsed = time.perf_counter() - forward_start
    paused()
    if original_modes != [module.execution_mode for _, module in modules]:
        raise AssertionError('Original quantized execution modes changed.')
    for name, module in modules:
        exact_export(name + '_params', torch.cat((module.input_scale.reshape(1), module.weight_scale_8.reshape(-1), module.bias.reshape(-1))))

    cuda_weight_code_checks = []
    if device.type == 'cuda':
        with torch.inference_mode():
            for name, module in modules:
                actual_codes = quantize_codes(module.weight, module.weight_scale_8, 8).to(torch.int8).cpu()
                expected_codes = exported(name + '_w8', actual_codes)
                cuda_weight_code_checks.append({'name': name, 'values': actual_codes.numel(),
                    'differences_from_saved_cpu_weight_codes': int((actual_codes != expected_codes).sum())})

    def comparison(a, b):
        a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
        delta = (a.double() - b.double()).abs()
        bits = (a != b) if a.dtype == torch.int8 else (a.view(torch.int32) != b.view(torch.int32))
        return {'values': a.numel(), 'bit_differences': int(bits.sum()),
                'maximum_absolute_error': float(delta.max()), 'rms_error': float(delta.square().mean().sqrt())}

    boundaries, codes = {}, []
    ordered_bit_differences = 0
    for name, device_actual in result.intermediates.items():
        actual = device_actual.detach().cpu()
        (out / ('model_' + name + '.bin')).write_bytes(actual.contiguous().numpy().tobytes())
        if name.endswith('_a8'):
            linear_name = name[:-3]
            expected = binary(ORDERED, linear_name + '_ordered_a8', actual.shape, True)
            stats = comparison(actual, expected)
            original = exported(linear_name + '_reference_a8', actual)
            codes.append({'name': linear_name, 'versus_saved_ordered_codes': stats,
                          'original_software_code_differences': int((actual != original).sum())})
        else:
            ordered_name = name.replace('_output', '_ordered_output') if name in ('layer1_query_output', 'layer1_key_output', 'layer1_value_output') else 'ordered_' + name
            expected = binary(ORDERED, ordered_name, actual.shape)
            stats = comparison(actual, expected)
            row = {'versus_saved_ordered': stats}
            if name in ('embedding_ln', 'layer0', 'layer1', 'layer2', 'layer3', 'logits', 'risk'):
                row['versus_c_service'] = comparison(actual, binary(DATA, 'bridge_' + name, actual.shape))
                row['versus_original_torch'] = comparison(actual, exported('reference_' + name, actual))
            else:
                row['versus_c_service'] = comparison(actual, binary(DIAGNOSTIC, 'diagnostic_' + name, actual.shape))
            boundaries[name] = row
        ordered_bit_differences += stats['bit_differences']
    threshold_tensor = exported('threshold', torch.empty(1))
    threshold = float(threshold_tensor[0])
    if loaded_exports != set(entries):
        raise AssertionError('Saved97-export comparison incomplete: ' + str(set(entries) - loaded_exports))
    summary = {'completed': True, 'recorded_utc': datetime.now(timezone.utc).isoformat(), 'case': CASE,
        'arithmetic_identity': ARITHMETIC_IDENTITY, 'primitive_identity': PRIMITIVE_IDENTITY,
        'execution_identity': ARITHMETIC_IDENTITY + '__' + device.type + '_eager_fp32',
        'primitive_identity_role': 'Saved ordered CPU comparison reference; actual execution device/backend recorded separately',
        'service_revision': SERVICE_REVISION, 'library_caveat': LIBRARY_CAVEAT,
        'checkpoint': str(checkpoint.relative_to(ROOT)), 'precision_map': metadata['precision_map'],
        'validation_device': str(device), 'device_record': device_record, 'validation_shape': [1, 256],
        'other_devices_and_batches_validated': False, 'backend_general_eligibility': None,
        'forward_elapsed_seconds': forward_elapsed, 'forward_synchronization': 'before and after forward' if device.type == 'cuda' else 'CPU synchronous',
        'timing_scope': 'One first forward with intermediate/code capture; includes Python/kernel launch and diagnostic capture overhead; excludes checkpoint load and post-forward CPU comparisons; no warmup/repetition or benchmark claim',
        'cuda_weight_code_checks': cuda_weight_code_checks,
        'torch_version': torch.__version__, 'cpu_threads': torch.get_num_threads(), 'autocast': False,
        'tf32': torch.backends.cuda.matmul.allow_tf32, 'float32_matmul_precision': torch.get_float32_matmul_precision(), 'graph_compilation': False, 'frozen_parameter_input_exports_exact': parameter_input_exports_checked,
        'all_export_files_compared': len(loaded_exports), 'frozen_quantization_params_unchanged': True,
        'quantized_execution_modes_unchanged': True, 'saved_ordered_bit_differences': ordered_bit_differences,
        'saved_ordered_reference_reproduced_exactly': ordered_bit_differences == 0,
        'boundaries': boundaries, 'linear_codes': codes, 'logits': result.logits.reshape(-1).tolist(),
        'risk': float(result.risk[0]), 'threshold': threshold, 'strict_reject': float(result.risk[0]) > threshold,
        'historical_evaluator_changed': False, 'risk_integration': 'Explicit wrapper.predict_risk(logits), never implicit historical evaluator softmax',
        'numerical_acceptance': None, 'end_to_end_tolerance': None,
        'scope': 'One loaded-checkpoint/saved-token integration check; no broader detector, HLS nonlinear or board validation.'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: summary[key] for key in ('completed', 'arithmetic_identity', 'saved_ordered_reference_reproduced_exactly',
          'saved_ordered_bit_differences', 'logits', 'risk', 'strict_reject', 'forward_elapsed_seconds', 'validation_device', 'numerical_acceptance')}, indent=2))
    if ordered_bit_differences or any(row['differences_from_saved_cpu_weight_codes'] for row in cuda_weight_code_checks):
        raise SystemExit('Loaded-model implementation differs from the reviewed ordered reference; inspect saved comparisons.')


if __name__ == '__main__':
    main()
