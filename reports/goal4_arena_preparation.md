# Existing-input W8-QAT arena preparation

Completed a fixed **67,108,864-byte (64 MiB)** arena image from the already exported numerical-bridge input and its existing W8-QAT checkpoint. No model was constructed or evaluated; no new data, inference, scoring, API, synthesis, allocation or board execution occurred.

Files are `results/goal4/internal_integration/arena_w8_qat_real_input_v1/arena.bin` and `preparation.json`. The image has only relative offsets; physical base and allocation remain null. This prepares a later real-input integration run and does not establish numerical acceptance or measured full-detector latency.

The fixed case is `aeslc:train:panus-s_inbox_1.subject:clean`, length256 with177 valid tokens. Its token IDs, type IDs and binary mask come directly from the saved bridge manifest. The prepared fixture uses the existing uniform-W8 map (`config_w8_mask=0xffff`); threshold bits are `0x3b955a95` from the saved bridge threshold. This is the historical search threshold carried for traceability, not a newly calibrated or approved deployment threshold. Numeric host mailbox IDs remain unassigned rather than invented.

The packer loads only `checkpoints/quantized/qat_w8_a8/quantized_state.pt` with PyTorch `weights_only=True`, plus its existing quantization metadata and the validated fixed command plan. It places full word/type/position embedding tables, all24 output-major signed W8 code matrices, biases, both saved frozen W4/W8 per-channel scale tables,24 frozen A8 input scales,9 LayerNorm affine parameter pairs, FP32 pooler and classifier parameters into the plan's exact slots. Both scale tables are copied from saved buffers; neither is recalibrated or regenerated from QAT weights. The W8 payload occupies the full3,145,728-byte maximum-weight-slot allocation. Other precision maps are rejected.

All151 buffer extents/alignment/nonoverlap and source tensor shapes/ranges/finiteness were checked. **65 saved exports matched bit-for-bit**: embedding rows gathered using the saved input IDs, the literal -0/-FLT_MAX key mask,9 normalization parameter arrays,24 W8 code arrays,24 input/row-scale/bias arrays, and pooler/classifier weights/biases. W8 codes recomputed solely for packing verification from saved QAT weights and frozen W8 scales exactly matched the existing exported code bytes. All151 image buffers were read back and checked; working buffers and dummy-zero space were verified zero.

No missing model parameter was replaced with zero. The script requires every plan buffer to be either an explicitly populated parameter/input buffer or an enumerated workspace/dummy buffer. Unused alignment gaps and the unallocated tail are zero padding. `preparation.json` records ordinary source paths, the plan/bridge timestamps, exact mapped buffers and checks; no hashing system was added.

Reproduce in a fresh output location/checkout with the already installed native environment:

```powershell
& 'C:\research\GATE\.venv\Scripts\python.exe' scripts/prepare_detector_arena.py --execute
```

The default command without `--execute` only previews scope. The script refuses to overwrite the existing completed output, use unsupported maps/input shapes/checkpoints, or continue with missing/inconsistent parameters. Existing completed evidence should be reused. Physical cold transfer/setup, numeric mailbox assignment, execution of the actual arithmetic engines against this arena, integrated resource/timing closure and broader numerical/protection validation remain undone.
