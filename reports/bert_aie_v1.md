# Whole-BERT AIE-ML/PL study — execution record

This is the task-owned report for the user-authorized 15 October 2026 Path 2
execution. It preserves the historical adoption decision and does not authorize
another research direction. **Execution is in progress. Physical validation is
not authorized. A scientific contribution is not established.** The final
offline completion state will distinguish successful builds from blocked stages.

## Question, controls and evidence boundary

The fixed computation is the matched-data four-layer BERT document guard. The
experiment asks whether retaining whole-model intermediates within PL storage
reduces external-memory traffic and complete latency compared with a conventional
external-scratch mapping. Both mappings use the same INT8 projection/FFN engine,
BF16 attention engine, PL nonlinear operations, parameters and document semantics.
The experiment develops supporting implementation evidence. Generic tiling,
residency and spatial/sequential engine sharing already occur in CAT/SSR and are
not themselves a new contribution. A useful distinction and complete measured
benefit beyond those methods remain to be demonstrated.

The prospectively specified falsifier is failure to fit/connect, or lack of a
complete latency/traffic improvement after storage, graph crossings and achieved
clock are accounted for. The numerical comparison retains FP32, W8/A8 with FP32
attention, and W8/A8 with BF16 attention. Nominal 1% calibration FPR is the
principal reporting point; 0.5%, 2% and 5% are secondary. These are reporting
points, not universal adoption requirements. There is no post-hoc non-inferiority
margin. Detector labels do not establish downstream LLM protection.

## Bound inputs and data exposure

The checkpoint is
`results/cnn_adoption_v1/runs/bert_matched_lr5e5_supervised_s42_r2/checkpoint_selected.pt`,
SHA256 `95025bfc0e29163efea613d556d7da0b516ec1bb6c6fac9d9c4c9c6f1f373525`.
It was selected using existing development evidence (seed 42, epoch 4), not the
best historical confirmation seed. This study performs no learning comparison;
three newly trained seeds would not test the fixed numerical/mapping question.
WordPiece tokenizer SHA256 is
`d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66`.

The common immutable contract is
`results/dual_track_v1/shared/contract.json`, SHA256
`1dc93c47389aec8e823defebc578ed5d0d553c3df3c2045b5d9f6f781f1933b8`.
Before any new BERT scores, the task adopted the sibling-prepared shared panel
under `shared/data/prepared/`: `new_calibration` has 1,000 benign documents and
1,146 windows; `new_confirmation` has 600 documents (200 attack, 400 benign,
including 100 hard benign), 703 windows and 400 source groups. Protocol SHA256:
`614549f8a1eeac21cc4168f551812fe6398c3da2e856529f81bb55910ce1b398`.
Five shared roles have disjoint source, payload and context groups; the 8,800
previous source groups are excluded. Independent exact token/tail/mask checking
of all 1,600 primary calibration/confirmation documents passed in 5.36 seconds
(`results/bert_aie_v1/primary_data_check.json`).

The authored response-redirection and quote/discussion constructions have one
author and potentially learnable wrapper cues. Group disjointness does not make
them natural attacks or prove unseen-mechanism generalization. The independently
prepared `shared/prospective_v1/` panel is preserved unused for quality, training,
threshold or model choices. Final-calibration, final-test, NotInject and sibling
confirmation outcomes remain unopened by this task.

The common timing workload is `shared/data/timing_manifest.json`, SHA256
`8d141711fcee46d65394c37460ef4bb377bd7346592ce11afa240ed6cf254162`:
12 natural development documents and six explicit unlabeled repeated-benign
stress documents with 1, 2, 4, 8, 16 and 32 windows (18 documents, 77 windows).
The stress documents supply timing evidence only. The initial alternative timing
manifest remains historical; this shared manifest is the final comparison input.

## Actual model and numerical specification

The model has four layers, width 256, four heads of width 64, and FFN width 1,024.
It retains word, position and token-type embeddings, embedding normalization,
all residuals and layer normalizations (epsilon 1e-12), exact-erf GELU, tanh pooler
and classifier. Sequence length is 256; each content window holds 254 tokens,
overlap is 64 content tokens and stride is 190. Tails are complete, attention is
bidirectional with key padding masks, and ordinary token types are zero. The
document score is maximum window `logit[1] - logit[0]`; rejection uses strict `>`.

There are 11,171,074 parameters, occupying 44,684,296 bytes in FP32. Matrix work
per window is `4 * (4*N*d*d + 2*N*d*1024 + 2*N*N*d)` = 939,524,096 encoder MACs,
plus 66,048 pooler/classifier MACs. Of the encoder work, 805,306,368 MACs use
quantized dense layers and 134,217,728 use attention matrices. These are operation
counts, not measured throughput.

Weights use signed INT8 [-127,127], per-output-channel scales and nearest-even
rounding. Activations use fixed symmetric per-module absmax scales obtained from
64 evenly sampled FIT windows, with INT32 accumulation. Dequantization is
`(acc.float() * input_scale) * weight_scale + bias`, preserving operation order.
BF16 attention rounds Q/K and probability/V operands, accumulates in FP32 in
64-inner-dimension chunks, and retains surrounding FP32 operations. All three
formats are separate numerical execution variants of the fixed checkpoint.

The actual packed payload is 35,284,072 bytes: 32,101,384 fixed FP32 bytes,
3,145,728 INT8 weights, 36,864 weight-scale bytes and 96 input-scale bytes. It is
not an 11 MB integer-only BERT. Export manifest SHA256:
`c4fe164a48051ca3834e112f5efd6564c559d8d62604dfe716cb2c1c097d92a3`.
The scale calibration identity is
`6bda96b7d50f55299523e9bde8e3ee3339cebd1619837670a17879e4a55cad73`.

## Executed development evidence

Three integer/cast/tail/BF16 primitive tests passed. Seven full padded cases
(three real FIT and four synthetic, including nonzero token types) gave maximum
source-versus-explicit FP32 logit error 1.669e-6, exact zero change under masked
key mutation, and exact export/reload agreement. BF16 candidate versus reference
maximum logit change was 0.02639 on the real FIT subset and 1.662 on synthetic
random sequences; these populations must not be merged.

The existing development comparison used 400 attacks and 800 benign documents,
with each route calibrated separately on the existing benign calibration role.
At the principal nominal 1% point:

| Execution | Threshold (margin) | Recall | Observed FPR | Descriptive AUC |
|---|---:|---:|---:|---:|
| Source FP32 CUDA | 8.4258651733 | 336/400 = 84.0% | 0/800 | 0.993666 |
| Explicit FP32 CUDA | 8.4258613586 | 336/400 = 84.0% | 0/800 | 0.993666 |
| W8/A8, FP32 attention | 7.5797929764 | 321/400 = 80.25% | 0/800 | 0.991128 |
| W8/A8, BF16 attention | 7.7387905121 | 316/400 = 79.0% | 0/800 | 0.991209 |

The 5 percentage point candidate gap is retained, not rescued by additional fit
or a changed criterion. The attention-format control contributes a further
1.25 point gap relative to W8/A8 FP32 attention. These results are development
evidence, not fresh confirmation. A CPU source pass took 86.25 seconds and gave
the same principal counts. The complete four-route CUDA scoring dispatch took
34.36 seconds; this batch scoring duration is not document latency.
Paired source/payload-aware development analyses are in
`results/bert_aie_v1/analysis/`. Confirmation is pending a frozen execution record.

## Connected implementation and build state

`hardware/bert_aie_v1/model.hpp` implements the complete four-layer path, with
resident and external-scratch template mappings. `pl.cpp` supplies AXI memory
interfaces and six 128-bit streams to native INT8/BF16 AIE kernels;
`graph.cpp` and `system.cfg` define their actual connection. `host.cpp` implements
the persistent complete-document protocol and raw-margin aggregation. It has
not been run on a physical device.

The actual target is `xcve2802-vsvh1760-2MP-e-S`, VEK280 platform
`xilinx_vek280_base_202520_1`, Vitis/Vivado 2025.2. Native AIE-ML support and the
AIEML build licence were exercised by compilation. The first build reached
kernel compilation, placement/routing and host generation, then failed because
an ELF named `functional` shadowed the C++ standard `<functional>` header. The
executable was renamed and the failed build preserved. A subsequent build was
stopped by the resource guard at 130.73 seconds during parallel elf generation
(Windows available memory 1.09 GiB); this is not a numerical or licence failure.
The documented `--Xelfgen=-j1` option is the focused next resource correction.

The current competent tile refinement is INT8 32x256x32 and BF16 32x64x32;
inner BF16 accumulation remains 64, so refinement does not change the numerical
model. It reduces calls to 3,072 INT8 and 2,048 BF16 services per window. Aggregate
PLIO operand/result payload is 62,914,560 + 25,165,824 = 88,080,384 bytes/window,
excluding weights, external scratch and host transport. At 128 bits and an
assumed 250 MHz PL port, the INT8 A port alone implies 6.29 ms of ideal transfer
time; this is a bound for this narrow two-engine graph, not for BERT or FPGA
architectures generally. Achieved clocks and actual overlap remain to be built.
Resident scratch requests 720,896 FP32 values (2,883,584 bytes) and 262,144
quantized bytes; synthesis must establish physical URAM/BRAM placement.

Portable full-model FIT checks agree exactly between the two scratch layouts.
The maximum inspected Python-versus-C++ BF16 logit difference is approximately
0.00358; different floating reduction and quantization boundaries remain an
explicit execution distinction. Actual signed INT8/BF16 stream packing tests
pass. One full FIT0 run with the installed bit-accurate HLS math library took
138.81 CPU seconds and matched the portable C++ logits exactly. It is functional
simulation, not a device latency measurement, and does not qualify every input.

Native XRT host linking requires the matching target XRT headers/libraries or
board sysroot; these were absent from the inspected installed Vitis/XRT paths.
No physical access, probing, programming, power or energy measurement occurred.

## Software timing and board continuation

The measured reference uses a CUDA Graph with explicit embedding/mask/encoder/
pooler execution. Its tiny accumulation differences change two calibration rank
positions, so it is recorded as a separate execution identity with separately
calibrated thresholds. The original eager route is retained. The first graph
profile is diagnostic because CPU analysis overlapped part of its measurement;
the definitive matched profile will run without competing task-owned heavy work.
A candidate graph attempt failed memory preflight before inference and is retained.

`scripts/bert_aie_v1/board_measure.py` defaults to offline packaging. Its first
18-document/77-window package passed transport checking. The final package will
bind the frozen evaluation threshold and export identity. Physical execution
requires a separate explicit permission and `--execute --board-authorized`, a
compiled persistent host, a linked xclbin and the matching runtime. Warm online
measurement includes tokenization, all windows/tails, packing, transfers, compute,
return and strict decision; cold initialization is reported separately.

## Resources, failures and reproducibility

Windows Python 3.11.16 (`C:/research/GATE/.venv/Scripts/python.exe`), PyTorch
2.9.1+cu128, Transformers 4.57.3 and RTX 4070 provide the software route. Vendor
tools run under Ubuntu WSL. Exact commands, hashes, memory snapshots, durations,
stdout/stderr and exit states are retained in `results/bert_aie_v1/dispatch/` and
`build/bert_aie_v1/`. Atomic canonical `gpu0`/`vendor` locks coordinate the sibling
task; no lock is stolen. Heavy builds hold the GPU lease too when necessary for
host-memory isolation. Preflight floors are 4 GiB Windows/8 GiB WSL; runtime
floors are 2/4 GiB. Failed preflights are not counted as successful experiments.

Other diagnosed failures preserved in the dispatch record are a `profile.py`
filename shadowing Python's stdlib module (renamed `profile_detector.py`) and
resource-busy admission. No dependency reinstall or result fabrication was used.
`configs/bert_aie_v1.json` records the bounded matrix and cumulative ceilings.
Weekly usage is checked periodically; work must pause at 90% used (10% remaining),
without consuming a reset. The last checkpoint observed 35% used.

## Prior work and serialized handoff

The source-verified methods/device/evidence comparison is retained at
`results/bert_aie_v1/literature.md`. CAT uses original-AIE VCK5000 BERT at N=256;
SSR's spatial/sequential mapping is established prior art. GAMA's VEK280 GEMM
simulation does not validate a complete BERT system. Current AIE4ML requires a
newer toolchain and does not support the complete imported graph/interconnect
needed here. The hls4ml physics transformer is a different small model, I-BERT
is a numerical method with GPU evidence, and HCCS is an optional learned surrogate,
not an exact softmax replacement. None supplies our missing system measurement.

Top-level STATUS and board handoff remain read-only during the concurrent track.
Proposed serialized update: add this task's actual offline completion/build state,
quality findings, artifact identities, explicit physical-validation pending state,
and one exact next command after final execution. Use `scripts/update_handoff.py`
and its read-only checks for any later shared-file update. No scope change is
proposed. Git synchronization uses the isolated `codex/bert-aie-v1` branch;
only task-owned sources/config/report are to be committed, with ignored evidence
retained in the canonical checkout and selected compact evidence archived as needed.
