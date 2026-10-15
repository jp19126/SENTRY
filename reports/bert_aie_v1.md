# Git release view — confirmation results withheld

This is the score-free publication view of the canonical task report. The full
local report and saved calibration/confirmation outcomes remain task-private
until the CNN candidate freeze, as required by the shared execution contract.
This view is not a claim that private evidence is missing or passed.

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

The immutable common timing workload is
`shared/prospective_v1/timing_manifest.json`, SHA256
`6380d3ac38079ba5ec829a4e8cdef56452f306d4e281e7108ae9e133e08b1392`:
25 existing development documents selected without scores across observed
genre/label/window-count bins. A later score-free reconciliation with the sibling
acknowledgment retained this original contract workload for definitive comparison.
The additional timing workload is `shared/data/timing_manifest.json`, SHA256
`8d141711fcee46d65394c37460ef4bb377bd7346592ce11afa240ed6cf254162`:
12 natural development documents and six explicit unlabeled repeated-benign
stress documents with 1, 2, 4, 8, 16 and 32 windows (18 documents, 77 windows).
The stress documents supply timing evidence only. Initial graph qualification
used this 18-document supplement. The scoring freeze preserves that preliminary
timing entry; the final dispatch explicitly distinguishes the original common
25-document workload from the supplement. No model, numerical operation,
calibration rule or quality population changes with this timing reconciliation.

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
The independent final-claim review found one stale descriptive field in that
frozen export manifest: `semantics.document_score` says positive softmax. The
implemented and prospectively frozen score is raw logit margin throughout.
`export/w8a8_bf16_attention/manifest_semantic_erratum.json` corrects the statement
against the exact frozen manifest hash without rewriting evidence. This does
not alter any payload, threshold, score or model choice.

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
`results/bert_aie_v1/analysis/`.
At the principal point there are 22 separately calibrated decision disagreements
(one candidate-only, 21 reference-only). The paired source bootstrap recall-gap
percentiles are [-7.46, -2.92] percentage points; crossed source/payload resampling
gives [-11.36, -1.17], and refitting both thresholds on paired calibration-source
resamples gives [-11.50, +2.25]. These are distinct observed-group sensitivity
analyses, not independent-population reliability intervals. Reusing the FP32
threshold for the candidate instead gives 67% recall, underscoring the importance
of explicitly binding the threshold to the numerical route.

## Prospective evaluation release state

All five numerical routes, the complete scalar C++ corpus comparison, paired
uncertainty analyses, six matched document profiles and CUDA event timings have
executed. Outcomes and derived thresholds are deliberately withheld here until
the sibling candidate freeze; they remain in the canonical report and results.
No new model, scale or threshold was selected from confirmation outcomes.



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
The documented `--Xelfgen=-j1` correction succeeded on attempt 2 in 213.27 seconds,
producing the 6,418,148-byte native `libadf.a`. This attempt held the vendor lease
and permitted compatible training overlap; Windows/WSL minima were 4.25/16.61 GiB,
above the unchanged 2/4-GiB runtime floors. The build driver records that scheduling
decision. Its initial process-group RSS sampler missed vendor descendants that
change groups, so that limited sample is not presented as total peak memory.
The first aiesimulator invocation returned zero for an unrecognized input option;
the output-file check caught the non-execution. The documented `--input-dir=.`
option enabled actual execution. All 2,048 INT8 outputs were exact; 2,048 real
BF16 QK/PV outputs passed the prospective gamma64 accumulation bounds (maximum
absolute error 3.63564e-6, maximum bound ratio 0.06706). This is two iterations
per engine, including FIT tiles and an INT8 boundary tile. The simulator reported
cores complete but stalled during teardown; its unclean termination is recorded
separately from the validated outputs. These are actual native tile results,
not a connected whole-model hardware simulation.

Compiler allocation uses two compute cores at (10,0) and (11,0), and five AIE
tiles when supporting tile-memory groups are included. Graph buffers occupy
65,536 bytes and reported system allocations 2,844 bytes; program memory use
is 3,264 and 1,040 bytes, with 32 bytes of reported stack use per core. These
are compiler allocations, not measured physical resource or power results.
The configured AIE clock is 1,250 MHz and the requested PLIO clock 250 MHz;
neither establishes the routed PL system frequency. Over the two native output
frames, first-beat intervals were 2,048 ns for INT8 and 2,480 ns for BF16, with
1,020 ns first-to-last spans for each 256-beat frame. Simulator sources fed the
tiles independently: these short intervals do not establish connected service
rate, complete-model latency or steady-state throughput. Stall counters were
not collected. Source hashes and allocations are in
`build/bert_aie_v1/common/native_costs.json`.

Resident full-model HLS attempt 0 completed in 269.51 seconds and produced an
XO. HLS estimated 124 BRAM18, 129 DSP, 81,199 FF, 320,294 LUT and 113 URAM.
The estimated 3.505-ns path is not timing closure: the requested period is 4 ns
and the HLS uncertainty is 1.08 ns. The generated multiply-add IP specifies
unfused operation, preserving the intended arithmetic separation. The frozen
model source was unchanged. Connected builds remain a separate evidence stage.

The external-scratch HLS attempt also completed, in 446.59 seconds, producing
its XO. Its estimates are 139 BRAM18, 110 DSP, 168,290 FF, 839,976 LUT and
22 URAM. The LUT estimate exceeds the part's 520,704-LUT capacity; this is a
synthesis-estimate warning, not a demonstrated placement failure. Hierarchical
reports attribute large costs to wide AXI data carried through long pipelines.
The scalar pooler inner loop alone is estimated at 144,216 LUT, dominated by
512-bit data mux/register structures. A focused paired correction caps automatic
parameters/scratch interface widening at 128 bits, without changing model
arithmetic, burst policy or memory-placement comparison. Both original XOs and
reports are preserved, and the original resident connected link runs first.
The original external schedule estimate is 412,058,913 actual-shape cycles,
or 1.648 seconds at 4 ns under ready-interface assumptions. This includes a
diagnosed conservative AXI dependence: the independent LayerNorm output loop
has II=154, while its reduction passes have II=1. It is therefore not used as a
competent-baseline speedup claim. The paired retry also targets only that output
loop's inter-iteration dependence. Each iteration reads/writes a distinct
`x[r*256+c]`; gamma/beta occupy a separate immutable parameters buffer, and the
mean/variance reductions complete before this loop. Their arithmetic order stays
unchanged. Corrected-build outcomes remain pending; neither narrower ports nor
the scoped dependence correction is yet a demonstrated improvement.

Original resident connected link attempt 0 completed block-design creation,
update and target generation, then was stopped during Vivado synthesis at the
required memory floor after 241.57 seconds. Windows available memory reached
1,655,484,416 bytes, below 2 GiB; WSL still had 20,769,845,248 bytes available.
Sampled owned-descendant RSS peaked at 5.21 GB. The aborted attempt and generated
design are preserved; it produced no qualified routed XSA. The focused retry
uses one Vivado internal thread, exclusive GPU/vendor admission and an 8-GiB
Windows launch floor for link, retaining the existing runtime floors. This
extra headroom follows the observed build peak; it is a host safety requirement,
not a scientific acceptance threshold. The driver persists owned process IDs
and treats resource/monitor/timeout aborts as failure even when a vendor wrapper
returns zero. No global cache or WSL setting was changed.

The generic HLS maximum of 284,672,201 cycles independently combines dynamic
matrix bounds, including K=1,024/O=1,024, which is not an actual encoder layer.
Substituting the actual shapes into the reported loop schedule gives 57,340,361
cycles, or 229.361 ms at the requested 4-ns period. This still assumes always-ready
AXI and stream interfaces and is an estimate, not executed full-model latency.
Quantization, GELU and residual loops already achieve approximately one element
per cycle. Each 8,192-byte tile gather instead requires about 8,200 cycles;
the weight AXI interface is only 32 bits despite a requested maximum width of
512. The dynamic generic access pattern prevents that requested widening from
being inferred. Thus scalar tile preparation remains a major implementation
bottleneck even though native matrix tiles execute quickly. A wider accessor
and banked gather would be a further implementation variant; it is not silently
substituted for the frozen mapping during this connected-build campaign.

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
`scripts/bert_aie_v1/cost_model.py` derives these counts from the actual source
dimensions. Its result (`results/bert_aie_v1/source_cost_model.json`) also counts
69,730,304 bytes of logical external intermediate accesses per window. This is
not observed AXI traffic: compiler reuse, burst formation and transaction
overheads remain unmeasured. The implementation author independently reconciled
the arithmetic against the source loops. No advantage over a PL-only design is
claimed because no matched PL-only implementation is measured.

Portable full-model FIT checks agree exactly between the two scratch layouts.
The initial FIT Python-CPU-versus-C++ BF16 logit difference was approximately
0.00358; different floating reduction and quantization boundaries remain an
explicit execution distinction. Actual signed INT8/BF16 stream packing tests
pass. One full FIT0 run with the installed bit-accurate HLS math library took
138.81 CPU seconds and matched the portable C++ logits exactly. It is functional
simulation, not a device latency measurement, and does not qualify every input.
The final streaming C++ harness loads parameters once and completed a three-FIT-
window smoke in 2.19 seconds; both mappings remained bit-identical, with maximum
Python logit error 0.003582. This harness uses scalar CpuTiles. Physical HLS uses
the stream adapter and native AIE `mmul` accumulators, whose floating reduction
order requires separate simulation evidence. A corpus C++ result must not be
mislabelled as complete hardware simulation.

Native XRT host linking requires the matching target XRT headers/libraries or
board sysroot; these were absent from the inspected installed Vitis/XRT paths.
The selected platform contains hardware-emulation XSA/ISD/PDI and QEMU templates,
but its XRT image directory contains only `boot.scr`, with no matching Linux
Image/root filesystem, XRT headers or `libxrt_coreutil`. Thus a connected emulation
smoke also requires missing target runtime assets; simulator templates alone do
not establish whole-system functional execution.
No physical access, probing, programming, power or energy measurement occurred.

## Offline board continuation

The raw-document wrapper and immutable 25-document/18-document input packages
are prepared locally. The threshold-free prerequisites and exact offline
command templates are archived under results/bert_aie_v1/public_evidence/.
Actual link outputs are build/bert_aie_v1/{resident,external}/bert.xsa; future
packaged outputs are each mapping/package/bert.xclbin. Linux Image, rootfs and a
matching Cortex-A72 Linux/XRT SDK are external prerequisites. Physical execution
requires separate authorization. No board operation or energy measurement ran.

## Resources, failures and reproducibility

All commands run from the canonical repository. The native Python interpreter is
`C:/research/GATE/.venv/Scripts/python.exe`; vendor stages use WSL `python3` after
`source /home/jp19126/Xilinx/2025.2/Vitis/settings64.sh`. Set
`SENTRY_CANONICAL_ROOT` and `SENTRY_SHARED_ROOT` explicitly when executing from a
different worktree; do not create independent locks or substitute missing data.
The checkpoint, prepared data, packed binary weights and vendor products remain
local artifacts, identified by the frozen manifests rather than embedded in Git.

Executed entry points include the following (the fixed freeze argument is
`--freeze results/bert_aie_v1/confirmation_freeze_v1.json --freeze-sha256
dbb8bf7b114b49bf18e4ca9c2dd0b5ca2e3072cea242859061ff8fd60824264d`):

- `run_confirmation.py` with the fixed freeze, through `run_job.py` and the GPU
  lease: reuses verified complete outcomes instead of rescoring them.
- `evaluate_cpp.py --export results/bert_aie_v1/export/w8a8_bf16_attention
  --reference results/bert_aie_v1/evaluation/confirmation/w8a8_bf16_attention`
  with the freeze, `--out results/bert_aie_v1/evaluation/cpp_confirmation`
  and `--max-seconds 3500`, through the CPU lease.
- `run_profiles.py` with the fixed freeze, through `run_job.py --resource
  gpu_vendor`: six serial profiles, internally reserving CPU too. It refuses to
  overwrite completed profiles; inspect existing results before any continuation.
- `analyze.py --candidate results/bert_aie_v1/evaluation/confirmation/VARIANT
  --reference results/bert_aie_v1/evaluation/confirmation/source` with the freeze,
  `--reps 1000` and a distinct `--out`; both numerical controls were analyzed.
- `summarize_results.py`, `summarize_cpp.py` and `cost_ledger.py` derive compact
  tables from saved evidence, without additional inference.

Each filename above is under `scripts/bert_aie_v1/`. Exact argument arrays,
interpreter identities and exit records are preserved in each dispatch `run.json`.
Numerical qualification/export commands are retained in the initial dispatch and
source CLI; preserve the frozen export and its adjacent semantic erratum instead
of overwriting it. The hardware build driver documents finite `aie`, `aiesim`,
`check-sim`, paired `hls --mapping resident|external` and paired `link` stages.
The board continuation is in `board_packages/resume.json` and remains separate
from these offline commands.

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
without consuming a reset. The latest checkpoint observed 83% used.
`results/bert_aie_v1/cost_ledger.json` separates outer lock waiting, admitted
child wall time and vendor stages. The profile dispatch reserved resources for
902.27 seconds, much of it waiting for the C++ CPU lease; that is not 902 seconds
of GPU computation. Initial recorded totals before the remaining native builds
are 951.28 seconds of CPU-dispatched work, 171.42 seconds of GPU-only dispatch,
918.58 seconds of combined reservations, and 1,709.58 seconds of native vendor
attempts through the original connected-link abort.
These are conservative wall-time records, not CPU core-seconds, energy or bills;
short direct preparation work is not exhaustively metered.

## Prior work and serialized handoff

The source-verified methods/device/evidence comparison is retained at
`results/bert_aie_v1/literature.md`. CAT uses original-AIE VCK5000 BERT at N=256;
SSR's spatial/sequential mapping is established prior art. GAMA's VEK280 GEMM
simulation does not validate a complete BERT system. Current AIE4ML requires a
newer toolchain and does not support the complete imported graph/interconnect
needed here. The hls4ml physics transformer is a different small model, I-BERT
is a numerical method with GPU evidence, and HCCS is an optional learned surrogate,
not an exact softmax replacement. None supplies our missing system measurement.
Primary sources: [CAT](https://arxiv.org/html/2409.09689v1),
[SSR](https://arxiv.org/html/2401.10417v2),
[GAMA](https://arxiv.org/html/2504.09688v3),
[AIE4ML](https://arxiv.org/html/2512.15946v2),
[hls4ml physics transformer](https://arxiv.org/html/2409.05207v1),
[I-BERT](https://proceedings.mlr.press/v139/kim21d.html), and
[HCCS](https://arxiv.org/html/2604.02292v1). The detailed comparison records
device generation, model/sequence, numerical format, code limitations and whether
the reported boundary is simulation, board execution or an analytical estimate.

Top-level STATUS and board handoff remain read-only during the concurrent track.
Proposed serialized update: add this task's actual offline completion/build state,
quality findings, artifact identities, explicit physical-validation pending state,
and one exact next command after final execution. Use `scripts/update_handoff.py`
and its read-only checks for any later shared-file update. No scope change is
proposed. Git synchronization uses the isolated `codex/bert-aie-v1` branch;
only task-owned sources/config/report are to be committed, with ignored evidence
retained in the canonical checkout and selected compact evidence archived as needed.
The first implementation checkpoint, `ad7eead`, was committed and successfully
pushed to `origin/codex/bert-aie-v1`; it explicitly records execution as in progress.
The second implementation checkpoint, `90aff0a`, was also pushed normally. It
adds corpus/profiling/trace/board tools without releasing confirmation outcomes.
The Windows worktree used the existing WSL-configured author identity via
per-command Git options. No main-branch switch, force push or shared-file edit
was performed. Later evidence and focused fixes require another ordinary commit.
