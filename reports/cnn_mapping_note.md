# Preliminary mapping questions for the CNN finalists

**Evidence state, 14 October 2026.** This note translates the eight selected
inner-development roles into possible implementation shapes on the recorded
VEK280 architecture, then reconciles the completed C confirmation and software
timing evidence. It is a supporting engineering comparison, not a method claim
or a deployment recommendation. The outcomes are from historically exposed
post-search development data; final partitions and physical FPGA measurements
remain unexecuted.

## What the current counts say

The numbers below come from executed CPU hooks on the actual configured
single-window tensor shapes and the frozen inner-selection manifests. MACs count
Conv1d/Linear calls, including repeated calls to shared weights. FP32 weight
bytes count unique parameters once, including the embedding table. “Inner MACs”
multiplies the per-window count by the actual padded windows across 4,860 inner
selection documents; it is a workload total, not a latency. Token windows are
L=256 (254 content positions, overlap 64, stride 190); byte windows are L=1024
(overlap 256, stride 768). The selected inner set contains 6,055 token windows
and 6,828 byte windows. The selected roles include a capacity-matched residual control paired with
`local_context_03`, rather than eight independent architecture claims. The
broader 96-configuration registry has 94 unique executed forward computations;
its only exact duplicates are `dilated_02`/`tcn_02` and
`dilated_06`/`tcn_06`. The `stacked_01`/`residual_01` signature match is a
parameter-layout match, not an identical forward graph: only the residual
family adds a skip connection.

| Selected role | Local structure / evidence hypothesis | Unique parameters; FP32 weights (embedding table) | Conv/Linear MACs per window; all inner windows | Local RF / feature lookahead; longest conv delay* |
|---|---|---:|---:|---:|
| `textcnn_01` | Parallel widths 3/5/7; conventional shallow lexical-pattern anchor | 2.015M; 8.06 MB (7.81 MB) | 15.73M; 95.24B | 7 / 3 tokens; 1,536 B |
| `stacked_01` | Two sequential width-3 convolutions; short-context conventional comparator | 1.982M; 7.93 MB (7.81 MB) | 7.34M; 44.44B | 5 / 2 tokens; 512 B |
| `dpcnn_05` | Repeated residual blocks with max-3 and pairwise max-2 pyramid downsampling; tests coarsened context | 3.537M; 14.15 MB (11.72 MB) | 52.69M; 319.05B | 346 / 188 tokens; 1,024 B |
| `tcn_04` | Causal dilations 1/2/4/8/16/32 with max/mean summary; tests long past context without model lookahead | 2.128M; 8.51 MB (7.81 MB) | 44.04M; 266.67B | 127 / 0 tokens; 24,576 B |
| `pooling_04` | Four width-3 blocks with per-channel top-4 pooling; tests retaining several high-evidence positions | 2.008M; 8.03 MB (7.81 MB) | 13.63M; 82.54B | 9 / 4 tokens; 512 B |
| `byte_hierarchy_03` | Raw UTF-8 byte embedding plus residual pyramid; tests a tokenizer-free, finer-grained representation | 0.553M; 2.21 MB (0.099 MB) | 257.16M; 1,755.91B | 188 / 97 bytes; 2,304 B |
| `local_context_03` | Local and contextual branches, concatenation/fusion, global max; tests explicit context mixing | 3.310M; 13.24 MB (11.72 MB) | 96.73M; 585.71B | 33 / 16 tokens; 5,376 B |
| `local_context_03_residual_replace_c98` | Six ordinary residual convolutions, kernels 3/15/1/3/15/1; capacity/receptive-field control | 3.307M; 13.23 MB (11.72 MB) | 95.84M; 580.29B | 33 / 16 tokens; 5,488 B |

MB is decimal. The delay values marked * use the registry’s logical tap-span
formula at FP32; they are not allocated RAM, simultaneous activation storage,
or device resource estimates. Receptive field and lookahead describe local
feature dependence only. Global pooling and the study’s document maximum still
require processing the complete window/document decision path.

The counts support design questions, not performance conclusions. The
embedding gather is 65,536 logical FP32 bytes for a 64-dimensional token window,
98,304 bytes for a 96-dimensional token window, and 393,216 bytes for the
byte window. These are logical gathered tensor sizes, not proven DDR traffic.
Token embedding tables are 7.81 or 11.72 MB FP32; the byte table is 98,688 B.
Ideal INT8 payload sizes in the machine-readable profile omit scales, packing,
numerical effects, and implementation support, so they are not deployment
storage claims. The registry’s largest observed module output and output sum
also do not reveal concurrent liveness or peak memory. In particular, the
local/context model’s functional concatenation is 196,608 FP32 bytes and is not
represented by the module-output hooks; its matched replacement changes the
convolution MACs and nonembedding parameter count by under 1%, but still differs
in normalization, residual placement, dropout and channel layout.

`pooling_04` retains 4 values for each of 64 channels, a 1,024-byte FP32 final
summary. That is only the result payload: comparison/update operations and a
working-state or full-sort buffer are not costed. `idcnn_01` is not a finalist
role, but its mapping profile illustrates why parameter and compute counts are
separate: it calls three temporal weight sets twice, counting their weights once
while counting both executions in MACs. `textcnn_01` does not apply
normalization in its parallel-filter forward path; its normalization config
field is inactive. `stacked_01` is configured with no normalization. The other
six selected architectures use per-token LayerNorm. The
Conv/Linear MAC totals exclude normalization reductions and affine operations,
pooling/ranking, masks, activations, residual additions, gates and data movement.

## Conditional route on the recorded platform

The existing design record describes a standalone Cortex-A72 supervisor in OCM,
the current PL BERT detector, shared PS/PL DDR_LOW0 through the NoC, and an
AXI-Lite mailbox. The offline L256 host packet carries a 64-byte header,
uint32 token IDs, uint32 token types, uint8 prefix masks and a 64-byte response:
2,432 bytes total. At 115,200 baud, 8N1, serialization alone is 211.11 ms per
such request/response exchange. This is a recorded BERT packet path, not a CNN
measurement or complete-document latency. The current platform exposes L=128,
256 and 512 command profiles; no CNN engine, candidate descriptor, embedding
table placement, or CNN host command has been integrated.

For the token finalists, a plausible first PL question is whether the embedding
gather can be co-located economically with a time-multiplexed temporal engine
and how its weights and activation buffers cross the existing DDR/NoC boundary.
`textcnn_01`, `stacked_01`, and `pooling_04` preserve L=256 and offer short local
windows; their structural contrast is parallel filter widths, sequential local
filters, and top-k evidence retention. The pyramidal `dpcnn_05` trades sequence
extent for repeated blocks and has stride-32 output; its RF exceeds the 256-token
window and its lookahead is 188 tokens. The causal `tcn_04` avoids future feature
lookahead, but the maximum score over a document’s windows cannot be final
until all windows have been observed; its longest dilation implies a much
larger logical delay span than the short-kernel models. A zero-lookahead label
therefore does not establish early document enforcement or remove the need to
retain/aggregate a complete document decision.

`local_context_03` and the capacity-matched residual replacement are the useful
structural pair: compare explicit branch/context fusion against a conventional
serial residual route at similar parameter and MAC counts. The pair may expose
different concat/fusion versus residual activation schedules, but no buffer
benefit follows from the hook counts. The byte model changes both representation
and window geometry: L=1024, stride 768, up to 47 prepared windows per inner
document. Its byte IDs do not match the recorded token-ID/type/mask packet, and
L=1024 is outside the recorded command profiles. It would need a separately
specified host format and engine contract before comparing transport or
complete-document cost. Its lower parameter count does not compensate by itself
for the larger per-window MAC count or establish a system-level trade-off.

The transport boundary is material for every route. The present UART number is
per L256 exchange. It must not be treated as a document total: CNN packet
granularity, whether token types are omitted, mask handling, where window
maxima are accumulated, and the number of host/PL round trips are all
unvalidated. A causal feature engine may emit intermediate scores early while
the end-to-end detector still waits for final document aggregation and policy
action. Embedding placement, tensor precision, activation liveness, pooling
implementation, host/PL synchronization, and tokenizer/window preparation need
to be included in any later complete-chain comparison.

No candidate CNN vendor build was dispatched. The CNN study made synthesis
optional and reserved its bounded resources for fit, confirmation and complete
software timing; it had not yet established C-quality finalists or a frozen CNN
chain whose synthesis would answer a decision-critical question. This is a
bounded-study choice, not tool unavailability: the repository records working
toolchain probes and prior platform/RTL build results. Those results concern the
existing BERT/arithmetic design, not any of these CNN graphs. Catalog capacities,
prior routed utilization, estimated frequency, and numerical service checks do
not predict CNN LUTs, DSP use, BRAM/URAM, clock closure, power, throughput or
physical-device performance.

## Confirmation and route implications

The C confirmation completed all 24 frozen fits and independent saved-score
checks. At the prespecified document operating point (each seed calibrated on
its own benign calibration examples; `score > threshold`; 99% recall and at most
1% FPR), all eight models had zero false negatives on the 2,738 development
attacks, but **no architecture passed in all three seeds**. Pass counts were:
TextCNN 0/3, stacked 0/3, DPCNN 0/3, causal TCN 0/3, pooling 1/3, byte hierarchy
1/3, local-context 2/3, and capacity-matched residual replacement 1/3. The
worst-seed observed FPRs ranged from 1.10% (local-context) to 1.68% (DPCNN);
all exceeded the frozen 1% bound. These are development-set outcomes, not
independent final-set evidence. The diagnostic set is also small and authored
(96 documents, one attack intent/template across eight topics and 12 recipes),
so it does not establish natural-family generalization.

The local-context versus replacement comparison is the most useful remaining
architecture discriminator, not a winner claim. Across the three paired seeds,
recall differences were zero and local-context mean FPR was 0.341 percentage
points lower; the paired source-group bootstrap 95% interval was -0.731 to
+0.024 percentage points, including zero. It conditions on fitted thresholds
and fixed models and omits calibration and seed-population uncertainty. The two
are within 1% on parameter and Conv/Linear MAC counts, but differ in width,
skip placement, normalization inputs, dropout and feature fusion. Therefore the
observed contrast cannot isolate concatenation/fusion as a cause. A falsifiable
next question is whether a prospectively specified hardware-aware local/context
fusion can reduce complete-chain buffering or transfer cost while preserving a
predeclared detector-utility bound against this capacity-matched residual
control. If the method and comparison do not need the FPGA, this is not an
FPGA-centered contribution.

Timing reinforces the need to measure the whole document path. On one RTX 4070
software setup (eager FP32, TF32 disabled, batch-one serial windows), the mean
warm complete-document time over 12 benign development examples was TextCNN
17.76 ms, stacked 18.17 ms, DPCNN 61.94 ms, TCN 38.43 ms, pooling 25.57 ms,
byte hierarchy 205.61 ms, local-context 31.53 ms, and replacement 35.52 ms.
These include preparation, host-to-device transfer, network, return and
aggregation for the sampled documents. They are not matched BERT timings,
throughput, all-seed timing distributions, or FPGA results. For context, the
corresponding analytical full-development mean Conv/Linear MACs per document are
(respectively 20.33M, 9.49M, 68.10M, 56.92M, 17.62M, 383.84M, 125.02M and
123.86M), but neither quantity substitutes for the other: the hook count omits
normalization, pooling, activations, masks, copies and synchronization, and
software kernel/transfer overhead affects latency. The byte representation also
uses L=1024 windows, up to 47 windows per prepared document, and lacks a matching
recorded host packet contract.

The separate one-process cold-start check covered all eight on one short,
single-window benign example. It observed zero risk delta and no changed
decision versus the saved warm result. It did not flush system/file caches or
estimate a cold-boot distribution; it does not characterize long-document cold
latency or a startup distribution. Emulated W8A8 fake-quant scoring was independently checked for TextCNN,
byte and local-context across three seeds (nine scored lists). These still used
FP32 arithmetic, not native integer kernels or RTL. Thresholded decisions
changed on examples, so fake-quant cannot be described as decision-equivalent;
the sampled FPR/pass counts did not reverse the conclusions above.

Given these results, keep three bounded research routes in view, with no route
authorized as deployment:

1. **Primary paired discriminator: local-context versus its residual
   replacement.** This has the strongest observed quality evidence (2/3 versus
   1/3 passes) and similar analytical compute and measured software latency,
   while remaining inconclusive under paired uncertainty. A small next study
   should test one explicit fusion/buffering hypothesis end-to-end, count all
   host and PL transfers, and retain the same threshold/utility boundary. It
   should stop if the candidate cannot meet a predeclared quality bound or if
   the hardware boundary does not change cost materially.
2. **Secondary small-engine discriminator: pooling_04.** Its analytical mean
   document Conv/Linear work (17.62M) and warm software timing (25.57 ms) are
   modest, but it passed only 1/3 seeds and is not quality-qualified. A bounded
   test could compare streaming top-k accumulation with a conventional max
   summary on the actual masks and complete document aggregator. Include ranking
   comparisons, retained state and all boundaries; stop absent a decision-relevant
   advantage over the simpler summary route.
3. **Negative/parked representation route: byte_hierarchy_03.** Its 1/3 pass
   record, 383.84M mean document MACs, 205.61 ms warm software time, up to 47
   windows and unmatched L=1024 transport make it a poor first hardware target.
   Reopen only if tokenizer-free byte input is made central to an approved
   question and a concrete host/engine packet and complete-chain comparison are
   specified.

TextCNN and stacked remain useful low-cost conventional software comparators,
but neither passed any of the three confirmation seeds. DPCNN and causal TCN
also passed none and had higher measured warm times and work than these two
small comparators. There is no evidence-based reason to map DPCNN, TCN, TextCNN
or stacked first.
The current evidence therefore supports **no CNN candidate confirmed across
the three fixed seed runs**; the routes above are falsifiable follow-up
hypotheses only.

If a bounded implementation discriminator is later authorized, the conditional
PL route is to test one time-multiplexed temporal engine with explicit embedding
placement, activation retention, normalization/fusion and document-maximum
handling across the existing PS/PL DDR_LOW0 boundary. This is a mapping question,
not a resource or speed prediction. An AIE or heterogeneous partition remains
conditional on verifying the actual device/flow support for the needed operators
and tensor boundary; there is no candidate CNN AIE mapping or compile evidence
here. For either route, first define one complete raw-document-to-decision chain
and count packets, bytes, copies/cache steps, buffering, aggregation and
synchronization. Include the unchanged tokenizer/windowing and threshold policy.
The recorded UART figure remains only for the BERT L256 packet and cannot be
used as a CNN document total.

No candidate CNN vendor build was dispatched. This was a bounded-study choice:
no architecture met the unchanged empirical screen in all three fixed
confirmations, and no frozen CNN PL/AIE chain existed whose synthesis could
answer a decision-critical question.
It does not reflect tool unavailability. Existing toolchain probes and prior
platform/RTL build results concern the BERT/arithmetic design only; they do not
predict CNN LUTs, DSP, BRAM/URAM, timing closure, power, throughput or physical
device behavior. Any eventual build should be gated on a concrete falsifiable
complete-chain hypothesis rather than treated as evidence of detector quality.

## Reproduction sources

- Executed-hook and complete-inner-window costs, formulas and compatibility:
  [`shortlist_mapping_inputs.json`](../results/cnn_study_v1/shortlist_mapping_inputs.json).
- Frozen input/overlap/window rules and partition identities:
  [`data_protocol.json`](../results/cnn_study_v1/prepared/data_protocol.json).
- Candidate code and registry:
  [`cnn_models.py`](../scripts/cnn_models.py),
  [`cnn_search_v1.json`](../configs/cnn_search_v1.json), and
  [`cost_registry.json`](../results/cnn_study_v1/cost_registry.json).
- Recorded target, host protocol and prior platform evidence:
  [`goal4_platform_architecture.md`](goal4_platform_architecture.md),
  [`goal4_numerical_bridge.md`](goal4_numerical_bridge.md), and
  [`fpga_toolchain.md`](fpga_toolchain.md).
- Scientific boundary and active candidate evidence:
  [`RESEARCH_SCOPE.md`](../RESEARCH_SCOPE.md), [`STATUS.md`](../STATUS.md),
  and [`cnn_study.md`](cnn_study.md).
- Final C analysis, seed confirmation, complete-document warm/cold timing,
  error/quantization diagnostics and paired comparison:
  [`analysis.json`](../results/cnn_study_v1/analysis.json),
  [`confirmation_table.json`](../results/cnn_study_v1/confirmation_table.json),
  [`cold_process_timings.json`](../results/cnn_study_v1/cold_process_timings.json),
  [`final_error_inspection.json`](../results/cnn_study_v1/final_error_inspection.json),
  and [`local_context_pair_analysis.json`](../results/cnn_study_v1/local_context_pair_analysis.json).
