# Goal R1: detector and architecture decision

Started 2026-10-11 (Europe/Madrid), under `SENTRY_next_goal_prompt.md` and the
user's explicit instruction to execute it with no usage limit. That instruction
supersedes the historical usage-percentage stops for this run. The experiment
ceilings, resource floors, closed final partitions and prohibition on board/API
work remain. The old Goal 5 campaign is paused; the canonical BERT model and
historical arithmetic identities are unchanged.

## Bounded plan fixed before challenger training

Reuse epoch-3 floating BERT and uniform W8-QAT development scores, after checking
their actual saved documents, windows and identities. The fixed floating recall
is 1.0; every challenger must meet recall >=0.99 and empirical FPR <=0.01 on the
separate 4,107-document scoring partition. Fit thresholds only on the 1,355
benign threshold-partition documents, using strict score > threshold. Preserve
L256, 64-content-token overlap, complete tails and maximum window risk.

Add exactly one hashed word-unigram/bigram linear classifier and one small
scratch-trained token CNN. Seed 42; one fixed initial recipe each, detailed in
their saved training configurations. At most one additional learning run may
address a diagnosed training problem, distillation or quantization adaptation.
No repeated development-score tuning. Run one small real-input bring-up for each
path. Inspect learning once after initial training. Time viable paths on the
same saved single-window documents at batch one, including raw-text preparation,
inference, aggregation and applicable transfers.

Use two native workers with non-overlapping files: `gpt-6.1-sol` for challenger
implementation and `gpt-6-luna` for focused architecture/evidence extraction.
The coordinator alone dispatches heavy jobs and owns the checkpoint decisions.
No nested delegation or concurrent heavy jobs. Preserve Windows/WSL preflight
floors of 4/8 GiB and runtime floors of 2/4 GiB.

Compare at most two mappings (buffered PL and AIE-ML/PL), conditional on the
selected detector. Prototype at most one route, at most two configurations and
one cause-identified corrective rebuild per configuration. These allowances
do not renew the old 12-synthesis budget. No whole-device place-and-route.

The prospective practical objective is complete warm batch-one checking latency
below the strongest available protection-matched software execution of the same
candidate on these inputs. Arithmetic-only speed, unmeasured energy and changed
quality cannot substitute for this objective. A prototype requires both a
credible complete-cost scenario and a specific falsifiable research question.

## Saved starting evidence

`results/goal_r1/anchor_audit.json` recomputes thresholds/counts from saved raw
scores and verifies complete document IDs, window maxima/tails and train/search
source-group separation. No model was rerun. Both anchors use historical native
Torch CUDA FP32/SDPA execution; W8-QAT emulates fixed-scale encoder linears.
Neither is relabelled as ordered-C or accelerated INT8.

| Anchor | TP | FN | FP | TN | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Floating BERT epoch 3 | 2738 | 0 | 18 | 1351 | 100% | 1.3148% |
| BERT W8-QAT | 2738 | 0 | 13 | 1356 | 100% | 0.9496% |

## Actual challenger recipes and data scope

Both real-input forward/backward checks passed on one clean/injected training
pair, with finite losses/gradients and zero optimizer steps. Preparation produced
40,883 training windows from 32,370 documents, 5,143 threshold windows from 4,065
documents and 5,308 scoring windows from 4,107 documents. Prepared arrays occupy
138,404,904 bytes. Source-group separation passed.

The feature classifier has 262,144 hashed word unigram/bigram coefficients and
one bias, L2-normalized term counts and a logistic output. It uses 20 fixed
epochs, SGD learning rate 0.5, momentum 0.9 and weight decay 1e-5. Hashing is a
feature representation, not an artifact-integrity system.

The CNN uses the shared WordPiece IDs, randomly initialized 64-dimensional
embeddings, parallel width-3/5/7 convolutions with 64 channels each, ReLU, masked
maximum pooling, dropout 0.2 and a two-class linear head. It uses 10 fixed epochs
of AdamW at 0.001, weight decay 0.01. Convolution/max-pooling follows the standard
[text CNN structure](https://aclanthology.org/D14-1181.pdf); these smaller widths,
optimizer and training limits are project choices, not a paper reproduction.
Its embeddings are learned from scratch; BERT benefited from prior pretraining.

Both use seed 42, document batches of 32, window microbatches up to 64, span-label
cross-entropy averaged first within documents, gradient norm clipping at 5 and
the fixed final epoch. Native Torch 2.9.1+cu128 supplies real FP32 CPU/CUDA kernels;
TF32 is disabled. No INT8 or INT4 acceleration is implied.

`development_coverage.json` shows all 55 scoring attack templates also occur in
training, across 11 attack families. The scoring pool includes only 20 benign
EmailQA sources (the other 1,349 are AESLC), and 187 benign long documents. There
is no separate difficult-benign development subset in the opened files. These
data can test transfer across email backgrounds, not across unseen templates.
Final calibration/test and NotInject remain closed.

## Checkpoint 1: retain W8-QAT BERT provisionally

Expected: the linear classifier or CNN might satisfy the fixed protection screen
with much less computation. Observed: both completed their fixed recipes and
full batch-one development scoring, and both failed the screen.

| Candidate and actual scoring backend | Threshold | TP | FN | FP | TN | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Feature classifier, CPU FP32 | 0.63498354 | 2639 | 99 | 20 | 1349 | 96.3842% | 1.4609% |
| CNN, CUDA FP32 | 0.14351106 | 2738 | 0 | 24 | 1345 | 100% | 1.7531% |

Feature fitting took 64.289 seconds and CNN training 75.615 seconds; CNN peak
allocated CUDA memory was 132,050,944 bytes. Training cross-entropy fell from
0.46012 to 0.10977 for the feature model and 0.11615 to 0.000881 for the CNN.
No nonfinite values, resource failure or obvious training implementation failure
occurred. The fixed final checkpoints were selected without looking at search
scores. Scoring took 6.486 seconds for the feature model and 24.449 seconds for
the CNN, excluding process/model startup and original preparation.

Feature misses span all 11 attack families: 52/400 long-document attacks and
47/2,338 single-window attacks. The CNN detects all attacks but rejects 12/187
long benign documents and 12/1,182 single-window benign documents; its EmailQA
stratum has one false positive among 20 benign sources. These are recipe-specific
negative results, not proof that CNNs or linear methods are intrinsically
inadequate. The small benign tail and seen attack templates also prevent a
general claim that BERT is necessary. Detailed raw counts and strata are in the
two `score_*_v1/metrics.json` records.

Implication: keep the existing W8-QAT BERT anchor as the only presently eligible
candidate. Its 13 false positives sit immediately below the 14-example failure
boundary, so this is a provisional development choice. Do not call its advantage
conclusive robustness evidence. The independent worker's short challenge is in
`results/goal_r1/checkpoint1_review.md`.

Next permitted action: establish a genuine accelerated software execution of
this same W8 arithmetic before deciding whether hardware is worth pursuing. The
single extra learning allowance is unused: distillation is permitted, but no
training-only teacher/student diagnosis currently supports spending it to repair
the observed benign tail. No threshold, seed or epoch is changed to rescue either
challenger. Failed candidates receive no performance-winner claim.

## Same-arithmetic software execution

Replaced the 24 fixed-scale W8/A8 encoder linears with genuine signed INT8
`torch._int_mm` products and INT32 accumulators. Frozen codes/scales are unchanged;
the original FP32 rescale product and bias order are preserved. Other operations
remain native FP32. CPU and CUDA passed exact code/accumulator/rescale checks
against INT64, including K=1,024 signed boundary inputs and real first-layer
query/FFN operands. This internal experimental Torch kernel is an available
implementation, not a claim that all optimized CPU libraries were exhausted.

CUDA eager and a single B1/L256 CUDA Graph each scored all 10,451 development
windows and independently fit the same threshold, 0.004557917360216379. Both give
TP/FN/FP/TN = 2,738/0/13/1,356. The graph always constructs an additive 4-D SDPA
mask; its own exact short/full-window bring-up and complete scoring qualify this
execution route. It is not an ordered-C/RTL identity.

| Native W8 execution | Preparation | H2D | Inference/max | Return/decision | Complete mean |
|---|---:|---:|---:|---:|---:|
| CUDA eager | 0.516 ms | 0.230 ms | 13.350 ms | 0.047 ms | 14.144 ms |
| CUDA Graph | 0.516 ms | 0.125 ms | 1.013 ms | 0.054 ms | **1.707 ms** |
| CPU, diagnostic only | 0.752 ms | 0.011 ms | 503.144 ms | 0.005 ms | 503.913 ms |

All pilots used the old 32 matched single-window documents after eight warmups,
batch one and complete raw-text-to-strict-decision boundaries. CUDA measurements
synchronize transfers/inference. Checkpoint loading, frozen weight preparation
and graph capture are separate cold costs. These are short warm means, not tail,
energy or general multi-window measurements. CUDA Graph is the strongest
available passing implementation measured here, not a globally optimal claim.

CPU full scoring was safely cancelled after 151.330 seconds with fewer than
1,000 windows completed: the observed rate could not finish within its fixed
600-second ceiling. Its exact bring-up passed, but full CPU quality is unresolved.
The separate 32-check diagnostic uses the historical threshold only to include
decision cost and explicitly cannot qualify a protection-matched CPU winner.
Partial failure/resources and the cancellation are preserved. No resource floor
was lowered. The historical 5.836-ms floating CUDA pilot has 18 false positives
at this operating rule and is not the protection-matched baseline.

## Checkpoint 2, intermediate decision before format qualification

Expected: replacing slow integer services might make AIE-ML attractive. Observed:
the new passing software target is 1.707 ms, while the old serial FP32 service
schedule alone is 474.325 ms, including 326.734 ms in QK/AV. Thus moving only W8
linears to AIE cannot rescue that schedule. Those are implementation-specific
simulation sums, not lower bounds on a newly buffered/vectorized FPGA design.

The independent architecture challenge identified one cheaper discriminator
before a hardware decision: BF16 attention operands with FP32 accumulation match
native AIE-ML capability. Authorize exactly one fixed, no-learning format check:
round Q/K and probability/V inputs to BF16 for all four layers' QK and AV products;
keep W8 linears, FP32 scaling/softmax and all other operations unchanged. Full
threshold and scoring partitions are required. Native FP32 matrix multiplication
of BF16-rounded values is a numerical reference, not a BF16 speed benchmark or
proof of AIE accumulation order. Bring-up must check the independent FP64 result
within the prospectively fixed FP32 dot-product bound gamma_K times the sum of
absolute products; exact BF16 rounding remains mandatory. No tolerance may be
expanded after seeing an error. No other precision variant is authorized here.

This is the one scoped format pivot; it does not change the canonical model or
select C. Its outcome, plus the realistic complete-path cost worksheet, decides
whether any hardware pilot is justified. Hardware prototype configurations and
additional learning runs remain zero at this intermediate checkpoint.

## Two hardware mappings and complete-cost worksheet

The common baseline is buffered/tiled PL with resident or prefetched W8 matrices,
activation reuse, streamed QK-softmax-AV row tiles, connected feed-forward stages
and reusable engines across layers. The preferred alternative is AIE-ML W8
projection engines with PL buffering/surrounding arithmetic, potentially native
BF16 attention only if qualified. Both receive common buffering and vectorization;
the slow historical service decomposition is not the sole baseline for a C claim.
Fully replicating four layers is not assumed to minimize batch-one latency.

Official [VEK280 device information](https://www.amd.com/en/products/adaptive-socs-and-fpgas/evaluation-boards/vek280.html)
and the [portfolio guide](https://docs.amd.com/api/khub/documents/9uMn0S~A2dGpuc~Nr~9cmA/content)
identify VE2802 with 304 first-generation AIE-ML tiles. The
[AM020 format table](https://docs.amd.com/r/en-US/am020-versal-aie-ml/Functional-Overview)
specifies 256 INT8 MAC/tile/cycle with INT32 accumulation and 128 BF16 MACs with
FP32 accumulation. INT8-by-INT4 is emulated at INT8 throughput; FP32 is emulated.
Stored W4 therefore provides no automatic twofold compute gain. These are
capabilities, not achieved rates or verified clock frequencies for this design.
Local Vitis 2025.2 AIE compiler/simulator/API headers and the VEK280 base platform
exist; their presence alone does not establish a licensed successful build.

| Item, L256 batch one | Actual count or evidence | Implication / unresolved term |
|---|---|---|
| 24 encoder W8 linears | 805,306,368 MACs; 3,145,728 weight bytes | Resident weights must be placed with buffers, or prefetched through measured bandwidth |
| Four layers' QK/AV | 134,217,728 MACs | FP32 contract initially; BF16 operand qualification is separate |
| Whole-model logical W8 payload | 35,284,072 bytes, including 32,101,384 fixed FP32 bytes | Embeddings dominate; this is not container size or proof of on-chip residency |
| FP32 hidden / FFN tensors | 256 KiB / 1 MiB | Tiling and producer-consumer storage are necessary |
| AIE local data memory | 64 KiB/tile; 19 MiB distributed over 304 tiles | Not one shared memory; routing and bank placement constrain useful capacity |
| AIE memory tiles | 512 KiB each | Exact usable tile count/placement not established here |
| One BF16 attention head | Each Q/K/V 32 KiB; full FP32 score matrix 256 KiB | A full head does not fit one local data RAM; row/panel tiling is mandatory |
| Linear A8 input / INT32 output payload | 2,359,296 / 9,437,184 bytes per window | Conditional AIE-PL traffic if every projection output crosses for PL rescaling; streaming/fusion must prove any reduction |
| Host IDs and mask | Two int64 arrays total 4,096 bytes; type IDs/layout additional if used | int32 packing would reduce these two to 2,048 bytes, with packing and matching input checks required |
| Nonlinear work | 1,048,576 attention softmax elements; 1,048,576 GELU elements | Matrix throughput alone omits substantial dependent work |
| Startup and synchronization | Cold load/capture separated in GPU evidence; hardware unmeasured | No free deletion of transfers or dependent stages |

Memory facts come from [AIE-ML tile architecture](https://docs.amd.com/r/en-US/am020-versal-aie-ml/AIE-ML-Tile-Architecture)
and [memory-tile documentation](https://docs.amd.com/r/en-US/am020-versal-aie-ml/AIE-ML-Memory-Tile-Memory).
The counts and saved sources are in `results/goal_r1/architecture_facts.json` and
`architecture_notes.md`; those extraction notes predate the new software timing.

The saved static-bank64 W8 integer subtotal is 142,925,208 cycles (714.626 ms).
Its 411-call FP32 service sum is 94,864,971 cycles (474.325 ms), and conditional
control/layout is 5,399,345 cycles (26.997 ms), giving 1,215.948 ms known internal
cost at the 200-MHz objective. Within the FP32 sum, QK/AV costs 326.734 ms;
everything else totals 147.591 ms, including softmax 32.789, quantization 35.454,
rescale/bias 35.447, LayerNorm 18.778 and GELU 15.773 ms. These are weighted
component RTL observations with formula fixtures and bounded one-outstanding AXI
traffic, plus a control simulation using completion stubs. They are not a full
detector or board run. Nominal transfer waits already appear in these observations;
the 105,393,192 old service-transfer bytes must not be charged again. Physical
DDR/NoC, host transport and startup remain additional unknowns.

For a redesigned route, the decision equation is preparation + packing/transport
+ scheduled dependent compute/transfers + result/decision < 1.707225 ms.
Keeping the measured 0.515797-ms preparation term leaves less than 1.191429 ms
for every other cost. Even granting zero transfer/nonlinear overhead, W8 linears
alone require more than 676 GMAC/s of useful work. AIE's peak capacity does not
rule this out, but neither distributed memory nor TOPS establishes achievable
occupancy, overlap or latency. The old W8-only AIE substitution fails; a new
attention/surrounding-operator design remains a different, unmeasured scenario.

## Checkpoint 2 outcome and final recommendation

The single BF16-operand reference passed all 32 exact cast checks and 24 operator
checks; the largest observed error was 0.114 of its predeclared bound. Full
development scoring took 79.990 + 85.099 seconds. Its independently fitted
threshold is **0.005079554859548807**, with **TP/FN/FP/TN 2,738/0/13/1,356**,
recall 100% and FPR 0.9496%. It has the same scoring decisions and length/source
strata as native W8 with FP32 attention: 8/1,182 single-window benign false
positives, 5/187 long benign false positives, 13/1,349 AESLC and 0/20 EmailQA.
No learning or threshold adjustment was used to obtain this result.

Numerical identity did change: 5,160/5,308 scoring-window risks differ, with
maximum absolute difference 0.13810. On the threshold partition, two benign
decisions exchange membership after each route fits its own threshold, despite
the same 13 calibration false positives. This supports investigating sensitivity
of the benign score tail; equal aggregate counts do not mean equal arithmetic.
The unchanged scoring decisions do not demonstrate a method improvement.
`root_bf16_result_review.json` preserves the comparison. The software reference
does not prove the actual AIE reduction order or its full-detector decisions.
No accelerated BF16 software timing was measured, and 1.707 ms must not be
attached to this changed arithmetic as though it were that route's timing.

**Final recommendation: INCONCLUSIVE-BLOCKED for a redesigned FPGA/Versal
investment; STOP-CURRENT-ROUTE for the legacy serial PL path and the variant
that replaces only its W8 linears with AIE.** Keep native W8-QAT BERT with CUDA
Graph as the provisional measured software choice. Keep the BF16-attention
reference as a numerically feasible future mapping candidate, not a selected C
or qualified accelerator. BERT necessity is unresolved beyond these two tested
challenger recipes and the limited development coverage.

Expected: a useful hardware pilot would discriminate complete-check value.
Observed: native AIE format support and BF16 development quality are feasible,
but there is no supported complete-cost scenario below the measured software
target. An attention-only prototype would leave FP32 normalization/GELU/rescale,
projection handoff, host transport and a common-engineered PL comparison
unresolved. The remaining legacy 147.591-ms service sum is not a fundamental
floor, and the 9.4-MB accumulator transfer can be reduced by a different placement;
neither reduction has been implemented or costed here. Peak AIE arithmetic alone
is insufficient to justify that investment. The existing architecture worker's
final challenge agreed with this bounded inconclusive outcome.

Implication: stop R1 at Checkpoint 2. Stage C and Checkpoint 3 hardware acceptance
were not reached; there is no AIE/HLS compile, simulation, physical measurement
or hardware speedup claim from R1. This is a completed decision study with an
unresolved architecture recommendation, not an access failure or a universal
negative result for FPGA. No pending R1 job remains. The old campaign is paused.

The contribution hypothesis remains explicitly unselected: **does preserving
benign threshold-tail margins change the useful operator-format/placement choice
under a complete-check cost budget, beyond a precision policy based on aggregate
quality/error?** The closest project baseline is the restricted Quasar-derived B
with its low-FPR criterion, receiving the same candidate formats, common tiling,
buffering, budgets and independently fitted thresholds. A future C would need
one defined additional mechanism and a matched B-to-C result. AIE adoption,
uniform BF16, buffering and this single development pass are not that result.

**Single most useful next action:** a separately authorized, bounded representative
encoder-block feasibility experiment for W8 projections plus BF16 attention,
including actual softmax, LayerNorm, GELU, rescale/quantize and realistic producer/
consumer transfers. Prerequisites are a concrete tile/buffer/format schedule,
successful tool access, an equally engineered PL baseline cost, and accelerated
software qualification/timing for the exact BF16 arithmetic. It should establish
numerical bounds before seeing errors and exercise real L256 model tensors.
Acceptance requires measured useful rates/stalls/resources and a dependency-aware
four-layer projection plus embedding/pooler/host costs that credibly beat the
strongest passing same-candidate software path. A simulator-only kernel time or
an unexplained optimistic allowance for the remaining work is a failure to
establish this case. Do not start full integration, board work or Goal 5 from this
recommendation. A negative or still incomplete case stops the route.

## Reuse, execution record and ceilings

Actual checkout HEAD was `a81928a2c6d68942826f1a932ce9119253af4224`, already dirty.
Native Python 3.11.16/Torch 2.9.1+cu128/Transformers 4.57.3 ran on an AMD Ryzen 7
5700X3D and RTX 4070, with four intra-op and one inter-op thread. No package was
installed. All heavy jobs were serial. Resource observations and prescribed
ceilings are in each run's `resources.json`. Preparation took 82.393 seconds
across the three partitions. Graph timing additionally recorded cold loading
5.382 s, weight preparation 0.108 s and graph setup 0.192 s; peak allocated CUDA
memory was 108,460,544 bytes. These costs are not hidden in the warm mean.

`root_saved_result_review.json` independently rereads complete document coverage,
window maxima, raw confusion counts, the fixed threshold order statistic and all
32 timing decisions. CUDA eager and graph risks are identical on all 10,451
development windows. A descriptive inherited SDPA metadata label was corrected
after the BF16 run; `bf16_metadata_erratum.json` records the old/new label, with
all scores and arithmetic unchanged. The cancelled CPU attempt is preserved in
`cpu_scoring_cancellation.json` and its run directory.

These are the executed entry points; all paths are relative to this checkout.
Use the native interpreter in README. Existing output directories are deliberately
refused: reuse results rather than rerunning completed experiments. For a justified
future reproduction, provide new output/run/score/bring-up paths explicitly.

| Script | Executed stages and limits |
|---|---|
| `r1_anchor_audit.py` | Saved-anchor inspection; no model execution |
| `r1_challengers.py` | `--stage prepare --max-seconds 1800`; `--stage bringup --model feature --device cpu --max-seconds 180` and CNN/CUDA equivalent; `--stage train-feature --device cpu --max-seconds 1800`; `--stage train-cnn --device cuda --max-seconds 1800`; each `--stage score --model NAME --device DEVICE --max-seconds 600` |
| `r1_bert_int8.py` | CPU/CUDA `--stage bringup --device DEVICE --max-seconds 180`; CUDA `--stage score --device cuda --max-seconds 600` and `--stage timing --device cuda --max-seconds 180`; CPU score attempted with 600 s ceiling, safely cancelled |
| `r1_bert_cuda_graph.py` | `--stage bringup --max-seconds 180`, `--stage score --max-seconds 600`, `--stage timing --max-seconds 180` |
| `r1_cpu_int8_pilot.py` | `--max-seconds 180`; diagnostic timing only |
| `r1_bert_bf16_attention.py` | `--stage bringup --max-seconds 180`, `--stage score --max-seconds 600` |

Ceilings consumed: two initial learning runs, **zero** additional learning runs;
two analytic hardware mappings; **zero** prototype routes/configurations/rebuilds;
one scoped format pivot. No final calibration/test/NotInject data were opened,
no new target API call or paid service was used, and no board action occurred.
Saved anchors, trained challenger checkpoints, prepared arrays, native INT8 and
graph implementations, BF16 reference and raw measurements are reusable.
They do not complete the historical numerical bridge, A/B/C campaign or deployment.
