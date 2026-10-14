# Initial CNN export and target mapping evidence

**Scope.** This is an initial exporter bring-up and analytical mapping for the
existing `stacked_01` seed-42 checkpoint. It exercises the new W8/A8 reference
path on old FIT data only. It is not the new adoption-stage candidate, quality
evidence, a confirmation result, or FPGA implementation. No GPU, vendor build,
or physical device was used.

## CPU W8/A8 reference and export

[`cnn_adoption_export.py`](../scripts/cnn_adoption_export.py) loads either the
legacy study run record or the adoption trainer's `model_spec.config` plus
`state_dict`, and exports a CNN-specific packed manifest, tensors and operation
graph. This initial execution used
`results/cnn_study_v1/runs/stacked_01_final_s42/checkpoint_latest.pt`
(SHA-256 `bcb3b0d7eb5dc28da5b6d285e5c861d7cf6100f7529572aac2ee3369feadac5b`).

Activation scales use up to 512 windows selected without labels from old
`full_train`: deterministic round-robin across source, available genre and
content-length strata, with one window per source group before additional
windows. This execution selected 512 windows from 512 documents/groups. The
exact ordered IDs, source-group IDs, available text hashes, window indices,
and hashes of prepared inputs are in
[`activation_calibration.json`](../results/cnn_adoption_v1/exports/stacked_01_final_s42/activation_calibration.json).
That mixture is only an exporter test; the adoption-stage activation calibration
must be rebuilt from the new FIT information and frozen before confirmation.

The reference uses symmetric W8 weights (`[-127,127]`, per output channel and
per embedding row) and static symmetric A8 inputs per Conv/Linear module
(`max_abs/127`, nearest-even rounding, same signed range). CPU dot products are
computed in int64, checked against the signed int32 worst-case and observed
bounds, then narrowed to int32. FP32 rescale and bias follow each dot product.
LayerNorm, gates, ReLU, residual operations, pooling and document aggregation
remain explicit FP32/reference operations. `stacked_01` has no LayerNorm, gate,
residual or downsampling; its two centered width-3 convolutions feed masked
ReLU, global max pooling and a two-logit classifier. It therefore has two-token
feature lookahead. Replacing FP32 rescale/bias or other exceptions with a fixed
integer approximation changes the reference and needs its own numerical check.

The first implementation's FP32-derived integer-dot check on 512 selected
FIT windows had maximum absolute logit error 0.130992, mean absolute logit
error 0.026466, and matching argmax on 512/512 windows. That historical
record is preserved under
[`history/pre_packed_loader`](../results/cnn_adoption_v1/exports/stacked_01_final_s42/history/pre_packed_loader/).
The current reference loader instead reconstructs its runtime weights, scales,
biases and FP32 exceptions from the packed export files, verifies file hashes
and tensor descriptors, and does not requantize the checkpoint. Its separate
four-window FIT check had maximum absolute logit error 0.115667, mean absolute
logit error 0.039136, and matching argmax on 4/4 windows. Neither check used
thresholds or labels. These are FIT-only numerical path checks, not
decision-equivalence at the calibrated operating point or numerical acceptance
for an adoption candidate.

The packed export occupies 2,106,124 bytes: 1,982,208 bytes of W8 weights,
123,140 bytes of FP32 scales and 776 bytes of remaining FP32 parameters, with
64-byte tensor alignment. Against the 7,929,608-byte FP32 parameter payload,
that is a 73.4% size reduction for this checkpoint. Files, tensor offsets,
scale values, operation list, identities and source hashes are recorded in
[`manifest.json`](../results/cnn_adoption_v1/exports/stacked_01_final_s42/manifest.json),
[`op_graph.json`](../results/cnn_adoption_v1/exports/stacked_01_final_s42/op_graph.json),
and [`reference_check.json`](../results/cnn_adoption_v1/exports/stacked_01_final_s42/reference_check.json).
Per-module A8 input scales are explicitly recorded in the manifest and repeated
in the scale payload; the loader checks they agree. Execution environment and
packed-loader use are in [`execution.json`](../results/cnn_adoption_v1/exports/stacked_01_final_s42/execution.json).
The corrected scale-serialization mismatch from the first loader attempt is
recorded in [`execution_attempts.json`](../results/cnn_adoption_v1/exports/stacked_01_final_s42/execution_attempts.json).

## Precision-matched BERT shape comparison

Using the saved four-layer, hidden-256 BERT checkpoint only for shapes, an
analytical uniform format that stores every two-dimensional matrix W8 with
per-row FP32 scales, all vector/bias parameters FP32, and static A8 input scales
for each of its 26 linear layers requires 11,375,464 bytes. The initial
`stacked_01` export is 81.5% smaller than this same-storage-format estimate,
above the prompt's prospective 50% parameter-and-scale reduction example.
The BERT estimate includes its embedding matrices at W8, matching the CNN's
embedding storage; it does not establish numerical quality for that broader
BERT quantization route. By comparison, the existing BERT W8-QAT scope quantizes
24 encoder linears while leaving embeddings and other parameters FP32; its
ideal packed payload is 35,284,072 bytes, but that is not an embedding-precision
matched comparison. The underlying FP32 checkpoint shape identity and both
payload calculations are in
[`mapping_assessment.json`](../results/cnn_adoption_v1/exports/stacked_01_final_s42/mapping_assessment.json).

For the same L=256 token window, counted linear plus attention QK/AV work is
7.340M CNN Conv/Linear MACs versus 939.590M BERT MACs (about 128:1). The BERT
count includes four layers of Q/K/V, attention output and feed-forward linears,
plus QK and AV products; it excludes embedding lookup, LayerNorm, softmax and
other operations. The CNN excludes embedding lookup, masks, pooling and
nonlinear work. These unlike operation graphs and counts do not predict latency
or speedup. In the old FIT partition, stacked averages 1.263 L=256 windows per
document (40,883 windows / 32,370 documents), or 9.271M counted Conv/Linear MACs
per document. For a tokenized document with `n` content tokens, the existing
254-content, overlap-64, stride-190 policy gives
`max(1, ceil(max(0, n - 254) / 190) + 1)` windows; work grows with that count.
These old-FIT distribution values must be replaced by the adoption workload's
document lengths before confirmation.

## VEK280 route and costs still open

The recorded target is AMD VEK280, part `xcve2802-vsvh1760-2MP-e-S`, with a
200 MHz PL objective. The configured 70% resource budget is 918 DSP, 364,492
LUT, 728,985 flip-flops, 420 BRAM36 and 184 URAM. The selected offline system
has shared PS/PL DDR_LOW0 through the NoC and a 64 MiB arena. Its PL IP contract
has separate AXI4 memory-mapped activation, weight and output bundles plus
AXI4-Lite control. The existing host sends a 64-byte header, uint32 IDs,
uint32 token types, uint8 prefix mask, and 64-byte response; it has no CNN model
descriptor or command. Prior BERT closure/resource results describe the BERT
design, not available CNN headroom. These details are recorded in
[`goal4_platform_architecture.md`](goal4_platform_architecture.md),
[`fpga_toolchain.md`](fpga_toolchain.md), and `configs/project.json`.

A plausible PL schedule streams one 256-token window from the activation
bundle, looks up the W8 embedding table and per-row scales from the weight
bundle, converts the gathered vectors to the first layer's A8 scale, then
streams two width-3 channel-64 convolutions. The logical A8 line-buffer lower
bound is 2 stages × 2 retained positions × 64 channels = 256 bytes, with
two-token lookahead. A full A8 feature map is 16 KiB; two ping-pong maps are
32 KiB. If int32 maps are materialized, one is 64 KiB and two are 128 KiB.
Streaming can avoid those full maps only if accumulator/rescale/activation
boundaries are scheduled as assumed. The final per-channel max needs 256 bytes
of FP32 state; the document maximum can be a 4-byte scalar if window scores are
consumed in order. The detector still cannot decide until all windows for the
document have completed. The embedding table is the dominant 1.95 MB W8 tensor;
its random lookup, scale access, banking and contention with convolution weights
need a concrete memory schedule. These are source-derived storage bounds and a
schedule hypothesis, not measured live memory or resource use.

If the existing UART request/response shape were reused once per CNN window,
omitting BERT's token-type array would reduce an L=256 exchange from 2,432 to
1,408 bytes, or 42.1%. At 115,200 baud, 8N1 serialization would still be
122.22 ms per window; a three-window document would be 366.67 ms before
tokenization, copies, cache maintenance, mailbox work, CNN compute and
aggregation. The existing BERT exchange is 211.11 ms per window under the same
serialization calculation. No CNN packet or batching contract exists, so this
is a conditional transport arithmetic comparison rather than measured
end-to-end performance. Reducing compute alone does not solve host transport.

The export's 2.11 MB payload is smaller than both the nominal aggregate BRAM36
plus URAM capacity under the declared budget (8.72 MB) and the selected 64 MiB
DDR arena. Neither comparison establishes that those resources are free or
that the tensors fit the required bank widths, ports, placement and bandwidth.
The exporter has not established CNN LUT/DSP/BRAM use, 200 MHz closure, power,
NoC traffic, UART integration or physical-device behavior. No vendor build was
made because this is a mapping artifact for the root's selection work, not an
approved hardware dispatch. A later discriminator should first fix a named
candidate and full document-chain boundary, then determine whether the
embedding-plus-convolution schedule, FP32 exceptions and host transport fit
the declared envelope. If that schedule cannot be bounded, the current 81.5%
payload scenario alone is insufficient for hardware adoption.

Reproduction (CPU only; no evaluation or confirmation partition):

```powershell
& 'C:/research/GATE/.venv/Scripts/python.exe' -B scripts/cnn_adoption_export.py `
  --run-dir results/cnn_study_v1/runs/stacked_01_final_s42 `
  --calibration-dir results/cnn_study_v1/prepared/token/full_train `
  --max-calibration-windows 512 --reference-windows 4
```
