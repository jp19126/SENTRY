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

## Four original structures: exact shape costs

[`original_static_mapping.json`](../results/cnn_adoption_v1/original_static_mapping.json)
is a shape-only audit of the four frozen original checkpoints named in
`configs/cnn_adoption_v1.json`. It verifies each checkpoint hash and loads the
state dict only to confirm parameter shapes; it executes no model forward. The
payload sizes below count every weight as signed W8, per-output/per-embedding-
row FP32 weight scales, one FP32 static A8 input scale per Conv/Linear, FP32
biases and all other trainable parameters, with the export's actual 64-byte
per-tensor alignment. These are exact byte counts for that format and shape;
the three non-Stacked values are not exported numerical references and their
activation scales still depend on their future FIT calibration.

| Frozen original | Parameters | FP32 bytes | Aligned W8/A8 payload | Payload reduction | Conv/Linear MACs per L=256 window | Feature RF / centered lookahead |
|---|---:|---:|---:|---:|---:|---:|
| TextCNN-01 | 2,015,426 | 8,061,704 | 2,139,148 B | 73.5% | 15,729,024 | 7 / 3 tokens |
| Stacked-01 | 1,982,402 | 7,929,608 | 2,106,124 B | 73.4% | 7,340,160 | 5 / 2 tokens |
| Local-context-03 | 3,309,986 | 13,239,944 | 3,445,444 B | 74.0% | 96,731,328 | 33 / 16 tokens |
| Residual replacement C98 | 3,306,532 | 13,226,128 | 3,444,560 B | 74.0% | 95,836,356 | 33 / 16 tokens |

The matched BERT route is hypothetical shape accounting on the saved
four-layer H=256 checkpoint: every 2-D matrix, including all embeddings, is
W8; each matrix has a per-row FP32 scale; each of 26 non-embedding matrices
has one FP32 A8 input scale; and all 14,082 non-matrix parameters (biases,
normalization vectors and other vectors) remain FP32. Applying the same
per-tensor alignment gives 11,156,992 bytes W8, 163,780 bytes of scales and
56,328 bytes FP32 parameters, 11,377,100 bytes total. The candidate payloads
are 69.7–81.5% smaller under this matched storage rule. This is above the
prompt's prospective 50% payload-reduction example, but it does not establish
equivalent BERT W8/A8 quality or any speed benefit. The existing BERT W8-QAT
scope instead leaves embeddings and other parameters FP32; its ideal packed
payload is 35,284,072 bytes and is not embedding-precision matched. All shapes,
checkpoint hashes, tensor counts, exact per-file payload arithmetic, and
boundaries are in the JSON record.

Per-document work is explicitly all-window work. For `n` content tokens the
frozen tokenizer/window policy gives
`N(n)=max(1, ceil(max(0,n-254)/190)+1)`, so each model costs its table MAC count
times `N(n)`; no model stops after the first suspicious window. At content
lengths 254, 255, 444, 445, 634 and 635, the complete window count is 1, 2, 2,
3, 3 and 4 respectively. Every further 190 content tokens adds one full
L=256 window, including the 64-token overlap and repeated special tokens.
The JSON lists corresponding MACs and payload bytes for each architecture.

For scale, the matched-format BERT count at L=256 is 939,590,144 MACs under the
same narrow linear plus QK/AV boundary, compared with the four CNN counts above.
CNN counts exclude embedding lookup, masks, norms, activations, pooling and
aggregation; BERT excludes its embedding lookup, masks, norms and softmax.
These operation counts are not latency or speedup evidence.

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

### Candidate-specific PL dataflow and retention

The existing integrated design is a desktop host → UART → standalone A72
supervisor → mailbox/PL path. The desktop host performs tokenization and window
construction; A72 validates/copies already-windowed IDs and submits PL work.
The existing PL contract has activation, weight and output AXI4 master bundles
plus AXI4-Lite, but no CNN command, CNN input packet, or CNN graph. The selected
shared DDR_LOW0 arena is 64 MiB; the configured 70%-budget aggregate BRAM36 plus
URAM capacity is 8.72 MB. Neither capacity number proves free or correctly
banked space. There is no recorded AIE/XRT route in this project, so the
hypotheses below remain PL/DDR/NoC routes rather than assumptions about AIE.
The existing A72 supervisor ELF has a 241,472-byte OCM load; each complete CNN
payload is 8.7–14.3 times larger, so an implementation that retains all weights
cannot assume the supervisor OCM as its backing store. The shared DDR arena is
the recorded host/PL memory route, subject to allocation and bandwidth proof.

| Original topology | Source-derived PL schedule question | Useful materialization sizes at L=256 | Main numerical/dataflow burden |
|---|---|---|---|
| TextCNN-01 | Three independent centered kernel-3/5/7 branches at C=64. Compare parallel branches with one time-multiplexed Conv unit; the former holds more branch data, the latter serializes weights/work. | Embedding W8+row scales 2,075,496 B; three branch maps total 192 KiB FP32 (48 KiB if represented as A8 codes), plus concatenated 192 KiB FP32 map (48 KiB as A8) at the join; pooled FP32 summary 768 B. | No LayerNorm or residual, but global max needs all token positions; a fused branch-wise max could reduce retention to three C-wide summaries. That fusion is a proposed schedule, not the current graph. |
| Stacked-01 | Two centered width-3 stages at C=64; test a fused stream from embedding/projection through both stages. | Embedding W8+row scales 2,075,496 B; one FP32 feature map 64 KiB (16 KiB if saved after A8 requantization); one int32 accumulator map 64 KiB; pooled FP32 summary 256 B. | Lowest source-counted CNN MACs; two-token local feature lookahead. Global max still holds a full-window barrier; FP32 bias/rescale then static A8 requantization sits between Conv stages. |
| Local-context-03 | Two blocks; each has local k=3, context k=15, concatenation and 2C→C fusion at C=96. Compare branch scheduling and fusion without losing the local/context values needed at the join. | Embedding W8+row scales 3,052,200 B; one FP32 map 96 KiB (24 KiB as A8); both branch outputs plus the concatenated map are 384 KiB FP32 total if simultaneously materialized (96 KiB as A8); pooled FP32 summary 384 B. | Six tokenwise LayerNorms; long 16-token centered feature lookahead. Local and context outputs must coexist until fusion under the direct source graph. |
| Residual replacement C98 | Six sequential residual Conv stages with kernels 3/15/1/3/15/1. Compare token-streaming skip alignment against materialized residual tensors. | Embedding W8+row scales 3,052,200 B; one C=98 FP32 map 98 KiB (24.5 KiB as A8); skip plus output maps 196 KiB FP32 (49 KiB as A8); pooled FP32 summary 392 B. | Six tokenwise LayerNorms and residual adds; 16-token centered feature lookahead. k=15 stages and skip/output alignment complicate streaming even though this is a single chain. |

Every model has an embedding gather followed by a static-scale A8 Conv input.
The reference exports per-row FP32 embedding scales, dequantizes each gathered
row to FP32, then quantizes it to the first Conv's A8 input scale. The C-wide
first Conv/projection input map is 65,536 FP32 bytes for TextCNN/Stacked and
98,304 bytes for the two width-96 embedding models if fully materialized.
Fusing that row-scale to A8 conversion, Conv rescale/bias, tokenwise LayerNorm,
ReLU and residual updates into narrower fixed-point operators would alter the
CPU reference; it needs its own exact numerical qualification. The manifest's
W8/A8 name therefore describes Conv/Linear dot inputs and weights, not an
all-integer execution graph.

The current integer-dot reference rescales each Conv/Linear output to FP32,
applies FP32 bias, and requantizes the next operator's input to A8. Thus the
FP32 sizes are the direct materialization requirement for that reference path;
A8 sizes describe a narrower hypothetical schedule that fuses or preserves
quantized boundaries and needs separate numerical qualification. The profile
sizes are individual buffers, not the simultaneous live peak. A streamed PL
design might retain only line history plus a pooled summary, while
a tiled schedule could need full feature maps, int32 accumulators, concatenated
branch data and residual skips. TextCNN's parallel branches and Local-context's
fusion impose explicit joins; Stacked is the clearest serial pipeline; C98 is
sequential but has the longest receptive field and residual retention. A
complete allocation must also place the random-access embedding table and
scales alongside kernel weights, with port/bank arbitration on the target's
weight bundle. None of those choices has an HLS schedule, allocation, or
resource report yet.

The host can either retain the complete `n`-token ID array (at least `4*n`
bytes, excluding raw text/tokenizer storage) or produce windows incrementally
while retaining 254 IDs for the active 256-position window. The latter assumes
tokenizer chunk equivalence that has not been implemented or tested. Windows
overlap by 64 tokens and are all scored; one FP32 running document risk scalar
(4 bytes) is sufficient if outputs arrive in order, but the document decision
cannot be returned until every window completes. Per-window CNN exchange size
under the existing packet framing is 1,408 bytes versus BERT's 2,432 bytes;
at 115,200-baud 8N1 that is 122.22 ms versus 211.11 ms of wire time per window.
For `N(n)` windows, separate exchanges cost `1,408*N(n)` bytes, while an
unimplemented single batched request would cost `128+1,280*N(n)` bytes before
responses. These are exact serialization calculations under the stated
packet assumption, not a CNN protocol or end-to-end time.

The detailed per-model work, aligned tensor payload, 64-byte storage
accounting, window-count examples, target envelope, packet arithmetic and
checkpoint/config hashes are in
[`original_static_mapping.json`](../results/cnn_adoption_v1/original_static_mapping.json).

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

The 2.11–3.45 MB candidate payloads are smaller than both nominal aggregate
BRAM36 plus URAM capacity under the declared budget (8.72 MB) and the selected
64 MiB DDR arena. Neither comparison establishes free resources, valid banking,
ports, placement or bandwidth. No candidate has a CNN build, allocated buffer
map, CNN UART integration, or physical measurement in this mapping. Zero
optional vendor builds were chosen for this bounded task: its purpose is
shape-based mapping across the four originals, and the current FP32-exception
W8/A8 route is not yet a fully qualified CNN execution chain. Tool availability
is not the blocker; the installed VEK280 flow has been used for the existing PL
design. A useful next discriminator is one exact candidate checkpoint and FIT
calibration, followed by complete-window packed/reference equivalence and a
named activation/embedding/host-transport schedule. Only then can a small
candidate-specific PL build answer whether that route fits the declared target
envelope. Payload reduction by itself is insufficient for adoption.

Reproduction (CPU only; no evaluation or confirmation partition):

Shape-only profiles for all four frozen originals, without a model forward:

```powershell
& 'C:/research/GATE/.venv/Scripts/python.exe' -B scripts/cnn_adoption_export.py `
  --static-originals-only
```

The initial Stacked packed-reference bring-up (which does run calibration and
the requested FIT-only reference vectors) is:

```powershell
& 'C:/research/GATE/.venv/Scripts/python.exe' -B scripts/cnn_adoption_export.py `
  --run-dir results/cnn_study_v1/runs/stacked_01_final_s42 `
  --calibration-dir results/cnn_study_v1/prepared/token/full_train `
  --max-calibration-windows 512 --reference-windows 4
```
