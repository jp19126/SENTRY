# Broad CNN development study

Started 14 October 2026 under the explicitly submitted
`SENTRY_CNN_development_prompt.md`. This authorizes local board-free CNN
exploration; the earlier span-coupled pilot remains a separate negative result.
The account usage ceiling was increased by the user from 40% to **60% used**;
the initial observation was 28%. No protected final partitions, new target
calls, paid services, physical hardware or Git synchronization are included.

**Offline exploration completed on 14 October 2026.** The study executed
144 learning runs, confirmed eight frozen architectures with three seeds each,
and checked 33 numerical variants. No architecture meets the primary empirical
screen in all three seeds. The useful result is a bounded quality/cost comparison
and a clear diagnostic limitation, rather than a robust guard or demonstrated
FPGA-aware contribution. Final findings and next-stage choices appear below.

## Question and prospective comparisons

Which convolutional representations and evidence-combination structures improve
document-level low-FPR injection detection, and which have credible economical
FPGA mappings? This is supporting architecture discovery, not a demonstrated
new FPGA-aware contribution. The discriminator is recall >=99% and observed
benign FPR <=1%, using benign-only calibration and strict score > threshold.
Nominal .5% and 2% calibration points are secondary views. Full document tails
and maximum-over-windows risk are retained. Token and byte window policies are
separate representation/context factors, not equivalent per-unit computations.

Historical R1 TextCNN found 2,738/2,738 attacks but rejected 24/1,369 benign
documents. Long benign FPR was 12/187 (6.42%), versus 12/1,182 (1.02%) for
single-window documents. This motivates context, pooling and length inspection;
it does not diagnose the cause. Its small ten-epoch recipe is not a general
rejection of CNNs. The new inner comparison retrains it without inner-holdout
exposure. Historical BERT pretraining and canonical-training exposure differ
from this architecture-only comparison.

## Execution protocol fixed before screening

Source-stratified grouped canonical-training folds: 70% fitting, 15% benign
calibration/associated variants, 15% selection, split seed 424242. Memberships
and a 1,200-fit-source screening subset freeze before fitting. No vocabulary or
model is learned on inner calibration. The pinned WordPiece tokenizer is reused
as an external fixed representation; byte IDs are raw UTF-8+1, pad zero.
Train loss averages span-labeled window cross entropy within each document,
then across 32 documents; max 128 windows per microbatch, AdamW, gradient norm
clip 5. CUDA FP32 kernels have TF32 disabled and deterministic algorithms enabled.

Registry: 96 configurations in 16 implemented structural branches. A source and
cost audit finds 94 unique evaluation computations: two centered TCN variants
duplicate the corresponding dilated variants. They count once for structural
coverage. TextCNN normalization/depth/dilation fields are inactive in its current
parallel-filter implementation; its true differences are widths, kernels and
pooling. The pyramid branch is a DPCNN-style adaptation, not paper reproduction.
Stage A
uses three epochs on the screening subset; these are screening results, not
adequate-training family rejections. Representative timing precedes the main
allocation. Stage B will give one representative of every viable family a
12-epoch full-fit opportunity, plus up to eight justified refinements, within
24 promotions. Inner checkpoint selection prioritizes the fixed recall floor,
then lowest benign FPR, then recall; no scoring-fold threshold optimization.
Selection checks use inner-calibration thresholds. Both feasible and near-feasible
models, structural diversity and cost inform promotion.

Stage C will freeze up to eight architectures, normally three seeds
(42, 1337, 2026), training on canonical full training at a fixed final epoch.
Frozen choices include representation, learning rate, epoch count and diagnostic
comparisons before canonical post-search development scoring. No tuning loop
will follow those scores. The byte representative, strong conventional contextual
CNN and shallow competence anchor are retained when viable. New authored probes
are labeled synthetic detector diagnostics, not verified target compromises or
an independent natural attack benchmark. Existing templates have historical
exposure; final calibration/test/NotInject remain closed.

Confirmation includes source-group bootstrap intervals conditional on fitted
thresholds, seed variability, source/family/position/window-count/length strata,
missed-attack and false-positive inspection. The shortlist receives complete
raw-document-to-decision batch-one eager CUDA FP32 timings on six shortest and
six longest benign development documents, five warmups and twenty repetitions.
Preparation, H2D, network, return/aggregation and cold model load are recorded.
Historical BERT timings are separately bounded evidence with a different input
sample; they are not a matched full-document latency superiority comparison.

For three diverse frozen architectures, practical eight-bit sensitivity uses
per-output-channel convolution/linear and per-row embedding symmetric W8,
static per-tensor input A8 calibrated on 512 full-training windows, maxabs/127,
nearest-even rounding and saturation [-127,127]. Norms, gates, nonlinearities,
biases and emulated accumulation remain FP32. Each variant fits its own benign
threshold; changed decisions and the score tail are inspected. This is fake
quantization, not native INT8 timing or RTL INT32 numerical equivalence.

The fixed 512-window calibration prefix is label-free but not randomized or
source-stratified. It is all AESLC: token input touches 447 complete documents
from 149 complete source triplets (149 clean/298 injected), byte input touches
419 complete documents from 140 source groups (140 clean/279 injected). Byte
window counts make the prefix stop before the last source's end variant. Both
prefixes contain the 55 known templates. Their contents differ by representation,
and neither establishes representative deployment activation tails. A failed
fake-quantized variant will concern this frozen calibration/arithmetic recipe,
not demonstrate that the architecture cannot support practical INT8 or QAT.

Ceilings remain 48 local accelerator job-hours, 24 heavy CPU job-hours and six
optional vendor attempts/12 vendor hours. Allocation after actual representative
timing will preserve confirmation capacity. Vendor synthesis is optional and
will only be dispatched if it answers a useful complete-chain question within
remaining resources. Analytical MAC/storage/mapping figures are labeled as
estimates and cannot establish LUTs, power, clocks, speedup or physical protection.

Detailed records: `results/cnn_study_v1/`; preparation and queue specifications,
run metadata, histories, checkpoints and raw scores preserve unsuccessful runs.
Reproduction uses the existing native Python environment
`C:/research/GATE/.venv/Scripts/python.exe`; WSL is used only for necessary
resource telemetry and repository reads. Supporting literature/mapping:
[cnn_prior_hardware.md](cnn_prior_hardware.md).

## Executed search and repairs

All 96 configurations passed synthetic CPU contracts; representative real-fit
document loss, gradients and updates passed separately. Token/byte preparation
and independent whole-document checks are complete. Frozen fit/calibration/
selection source counts are 7,552/1,618/1,620; screening uses 1,200 fit groups.
The 96 authored diagnostics have eight topics and exact/group/normalized-text
overlap zero; this does not establish semantic or attack-family independence.

Three representative 3-epoch timing jobs completed in 25.594 dispatch seconds,
with approximately 1.2-2.6 seconds/epoch for the small initial representatives.
The frozen stage allocation is A 3h, B 10h, C 20h, targeted learning 6h,
confirmation scoring/profiling 3h and reserve 6h, totaling the 48h ceiling.
Larger structures actually take longer; allocation is a ceiling, not predicted
runtime or a spending target. CPU preparation took 133.422 seconds; the combined
checks took 44.078 and 50.937 seconds and timing input verification 27.829 seconds.

The first timing dispatch had three implementation refusals before optimizer
steps (incorrect preparation manifest path); corrected and preserved. The first
main screen process completed 20 fits then stopped on the 2 GiB Windows memory
floor (1,869,197,312 bytes free). After actual exit, preflight recovered to
14,812,319,744 bytes free. Serial subprocess groups now release native allocations
between groups and resume saved epochs. Floors and recipes are unchanged.
Initial failure/dispatch records are preserved separately. A later grouped
byte-model attempt also stopped at the floor; isolated single jobs completed
the remaining screens. A bounded cuDNN cache is now used for longer fits.
[PyTorch 2.9.1 source](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/native/cudnn/Conv_v8.cpp)
documents the cache control; cache growth as the cause of all observed host
memory growth remains an inference. A focused actual-CUDA default-versus-64
cache comparison passed bitwise logits, losses and full gradients on 28
recorded real-fit cases in 17.265 dispatch seconds. It does not prove equality
on all conceivable shapes or attribute all working-set differences to the cache.

Stage A is complete: 96 runs, 94 distinct evaluation computations, all 16
branches. Twenty-three runs (21 distinct computations) meet the inner empirical
screen. Completed fit/scoring time totals 1,420.619 seconds; stopped-job wall
times and preparation are accounted separately. Independent saved-score checks
passed all 96 runs: identities, window maxima, thresholds, counts and checkpoint
hashes. These are short subset screens and are not confirmed candidates.

The frozen longer-fit queue contains 23 configurations: one representative per
branch, six additional configurations from different branches, and the exact
R1-style shallow TextCNN anchor for a competent adequately trained comparison.
Each receives all 12 full-fit epochs, with inner checkpoint selection. Centered
TCN computations already covered by dilated models are not promoted twice.
One of 24 promotion slots remains for the capacity-aware replacement. The
initial unexecuted 22-fit proposal is archived as superseded; the competence
anchor was added before any Stage B dispatch. All 23 Stage B fits are complete;
all 16 branches received all 12 epochs. Independent saved-score, strict
calibration, document/window identity, confusion and checkpoint checks pass all
23 fits in 7.157s. Twenty-one selected checkpoints pass the primary inner
screen. `large_kernel_05` and the exact `textcnn_01` anchor remain empirical
near-misses (19 and 17/1620 benign FP respectively); all 23 selected checkpoints
miss zero of 3240 attacks. Completed Stage B fit/scoring totals 6432.404s.
These selected inner observations do not establish superiority or seed stability.
Canonical post-search/diagnostic outcomes are reported in the completed frozen
confirmation below; the inner comparisons alone do not establish them.

Selected-checkpoint inner error inspection compares stacked_01, pooling_04,
dilated_06, local_context_03 and byte_hierarchy_03: each misses 0/3240 attacks;
benign FP counts are 12/11/7/14/9. The 53 model-document false alarms span
40 unique AESLC documents: 31 are unique to one model, six shared by two, two
by three and one by four; none is rejected by all five. Ordinary contract or
administrative questions, repair/editorial requests, procedural correspondence,
technical lists/newsletters and promotional text appear in the errors. These
are inspected examples, not a demonstrated causal taxonomy. Near-copy seminar
invitations differ by dates/source groups, illustrating unresolved semantic
dependence beyond exact source identity. All 4860 identities/text hashes,
own benign calibration cutoffs, maxima and checkpoint digests were rechecked
in 7.593s CPU work; `inner_error_inspection.json` contains concise paraphrases.
Token multiwindow FPs are 2/208, 4/208, 3/208 and 4/208 respectively; byte is
2/262 under its different window definition. No model was tuned on this
inspection or probe examples added to fitting.

After 21 completed fits, `dpcnn_03` stopped before its first epoch checkpoint
at the unchanged 2 GiB Windows floor (993,656,832 bytes available). After
process exit, availability recovered to 16,795,693,056 bytes. Cache 64 versus 8
gave bitwise-equal logits/loss/all gradients on two recorded fit shapes in
10.813s, but the cache-8 full retry still stopped at 1,253,363,712 bytes. These
two failed attempts and their original metadata are preserved; neither is a
measured quality rejection. A 10.765s comparison of default cuDNN v8 versus
the legacy API route passed the prospectively declared FP32 output/gradient
tolerances on the same two inputs. The
[PyTorch 2.9.1 implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/native/ConvUtils.h)
defines the API switch. The final bounded retry completed all 12 DP03 epochs
in 359.562s child wall (354.127s fit/scoring); observed free Windows memory
stayed between 15.50 and 16.56 GB at minute checks. Its selected epoch 10 has
15/1620 benign FP and zero attack misses, but only one of 12 epochs passes.
This is a
backend workaround, not demonstrated cause attribution or broad bitwise
equivalence. Architecture, masks, fit data, loss, optimizer and 12-epoch
opportunity are unchanged. Any finalist carries its recorded API/cache policy
through fit, confirmation and profiling; numerical/backend differences remain
separate from structural differences.

Final confirmation scoring used batch-one windows for both float and
quantized references, matching the timed network shape. Complete document
coverage and risk/calibration rules are unchanged. Cold-process observations
are separate from warm complete-document timings; system caches are not flushed.

The [analytical registry](../results/cnn_study_v1/cost_registry.json) contains
actual-shaped CPU hook counts for all 96 configurations and per-document
window scaling. Exact timing inputs match prepared tensors for the same six
shortest and six longest benign documents (49 token windows, 237 byte windows).
The capacity-aware local/context replacement helper and independent saved-score
checks passed; the learned inner ablation is complete.

The controlled replacement is frozen as `local_context_03_residual_replace_c98`:
six ordinary residual convolutions with kernels 3/15/1/3/15/1, width 98, the
same token embedding and receptive field 33/lookahead 16 as `local_context_03`.
Nonembedding parameters differ by -0.909%, padded-window Conv/Linear MACs by
-0.925%, and full FP32 weight storage by -13,816 bytes. This controls approximate
capacity and receptive extent; residual/normalization placement, nonlinearities,
dropout count and channel packing remain confounds. Functional concatenation
storage in the source is not captured by the largest module-output hook, so the
pair does not establish equal simultaneous activation memory. Its CPU checks
passed in 6.84s; the 12-epoch inner fit completed in 236.514s fit/scoring
(242.125s child wall). Selected epoch 10 has eight benign FP and zero attack
misses, versus the source's 14 FP at selected epoch 4. Both pass only one of
12 inner epochs. Their final epochs have 18 and 21 FP respectively. Distinct
selected epochs and one seed cannot isolate a fusion benefit; frozen final
confirmation uses the same 12-epoch opportunity for both across three seeds.

An independent byte-identity check passed all 64 files for token/byte full
training, search calibration, search scoring and diagnostics in 3.985s. The
freezer and confirmation now check actual prepared bytes against manifest
hashes and compare the complete fitted recipe to the frozen specification.
Full-training fits do not monitor the training-exposed inner folds; their
quality fields remain empty until separate development confirmation.

## Frozen finalist decision and completed confirmation

The inner-only selection rationale is saved with source/score/checkpoint hashes
in `results/cnn_study_v1/finalist_selection.json`; Stage C freeze records its
digest and creation time in `configs/cnn_stage_c_v1.json`. The freezer validated
all actual prepared files before writing. All 24 run IDs, three seeds, 162
total epochs and API0/cache64 settings were independently checked before dispatch.
Serial queue `stage_c_q1` completed all 24 fixed full-training fits in
4761.516s parent wall. Independent verification passed all 24 recipes, final
epochs, empty quality fields and checkpoint hashes (5.556s initially; 4.385s
after making the check's output scope explicitly fit-only). No canonical CNN
post-search or diagnostic scoring preceded the freeze. Serial confirmation
completed all 24 children in 3286.219s parent wall. The independent saved-score
check passed all 24 frozen runs in 13.133s, covering 33 variants and their three
score partitions. A separate ID-joined review verified all nine fake-quantized
changed-decision lists and table/per-run associations without discrepancies.
Eight warm profiles and eight fresh-process cold observations are complete.

| Finalist | Fixed epochs per seed | Inner-selected benign FP /1620 | Role |
|---|---:|---:|---|
| textcnn_01 | 8 | 17 | Exact shallow competence anchor, retained despite inner near-miss |
| stacked_01 | 1 | 12 | Cheap ordinary stack; 9/12 inner epochs pass |
| dpcnn_05 | 6 | 10 | Stable pyramid representative; 7/12 pass |
| tcn_04 | 4 | 10 | Causal long-context/cost diversity; 5/12 pass |
| pooling_04 | 4 | 11 | Top-k evidence retention; 7/12 pass |
| byte_hierarchy_03 | 7 | 9 | Byte representation/storage alternative; 3/12 pass |
| local_context_03 | 12 | 14 | Local/context mechanism source; common epoch with replacement |
| local_context_03_residual_replace_c98 | 12 | 8 | Conventional capacity/receptive-field replacement |

All selected inner checkpoints have zero attack misses. Ordinary architectures
transfer their inner-selected epoch count to full training, without checkpoint
selection on training-exposed folds. The last two deliberately use common epoch
12, whose inner endpoints are FP21 and FP18; their best selected endpoints are
not the frozen pair recipe. Every finalist uses seeds 42, 1337 and 2026.
Quantization targets are TextCNN01, byte03 and local/context03, with the
previously declared static W8/input-A8 recipe.

DP05 is retained over cheaper DP03 because DP03 passes only one inner epoch
and required a distinct backend repair. TCN04 is retained over centered dilated06
for causal dependency and lower counted MAC/storage cost; three additional
selected benign errors do not establish statistically worse quality. Pooling04
adds evidence retention, while the cheaper stack is an economical conventional
reference. These decisions preserve hardware diversity rather than assuming
an uncalibrated FPGA cost model ranks implementation performance. See the
[preliminary mapping note](cnn_mapping_note.md) for measured source counts,
conditional routes and unresolved buffering/normalization/transport costs.

## Diagnostic evidence limits

The 96 authored texts comprise eight background topics crossed with 12 recipes,
32 benign and 64 attack-labeled examples. They contain one attack intent/template
and four payload surface forms, with no independently sampled natural documents.
They can reveal sensitivity to quotation, ordinary imperatives, position,
long repeated context and typography. Their counts are dependent template/topic
variants and cannot establish attack-family generalization. Explicit historical
or analyst wrappers may help benign discrimination; several positive recipes
use 18 repetitions of a topic background while typography perturbations use one.
Typography and length/context therefore change together. Uncased tokenization
can remove case perturbations. The shared target phrase also appears in benign
quotations, but that alone does not rule out other benchmark shortcuts.

## Recoverable execution and comparison inputs

The frozen search registry is `configs/cnn_search_v1.json`; Stage A/B plans and
the controlled replacement plan are `configs/cnn_stage_a_v1.json`,
`configs/cnn_stage_b_v1.json` and `configs/cnn_ablation_v1.json`. Completed run
records contain exact configuration, seed, optimizer, loss, source/data hashes,
selected epoch, checkpoint digest and raw score files. Per-epoch histories and
dispatch plans/logs retain interrupted attempts. `run_table.csv` includes
preserved failures; it does not label full-training completion as held-out
quality. `runtime_ledger.json` avoids counting parent and child walls twice and
keeps unrecorded startup/refusal time unresolved.

Before finalist scoring, 38 current source/configuration/protocol files were
saved as immutable hash-named byte copies under `source_snapshots`, with source
set identity `bce33b79281e4e3bda3f52570c158979a798606fd9094cc14d862f8ea09ba138`.
The CPU-only copy took 0.248s. This preserves the current implementation; it does
not reconstruct older runner bytes merely from their recorded hashes. Final
evaluation records include the actual evaluator/training-runner source hashes
and environment. The independent final error helper refuses partial confirmation
and rechecked raw/prepared identities, common UTF-8-length strata, seed/error
overlaps, diagnostic failures and numerical decision changes after all 24 saved
confirmations passed independent checks. Its descriptions are anonymous rules,
not exhaustive manual causal annotations or additional training examples.
The pre-confirmation manifest `3147a262532037e063e3e74975f6fadcc20d851e056deb74281e58c890a78c6a`
adds the fit-checker's explicit output-scope wording (0.270s CPU); original
hash-named source copies and the first manifest remain preserved. Model, fitting,
data, operating-point and quantization computation are unchanged.

Saved canonical-development references are reduced without new inference in
`reused_references.json`: R1 W8-QAT BERT/CUDA Graph has TP/FN/FP/TN
2738/0/13/1356; R1 scratch TextCNN 2738/0/24/1345; FP32 epoch-3 BERT
2738/0/18/1351 at their own nominal 1% thresholds. Their ordered document,
label, group and window identities match the canonical partitions. Historical
score rows lack text hashes, so that limitation remains explicit alongside
the original R1 anchor audit. BERT's pretrained initialization and mixed
integer-linear/FP32 backend differ from the scratch CNNs; old timings use a
different input sample/boundary. No BERT optimization campaign was restarted.

The new `textcnn_01` anchor matches historical `TokenCNN`: all four matched-weight
CPU comparisons produced bitwise-identical logits, including controlled-dropout
training mode, in 7.91s process wall. Empty-input handling is extended. Both
runners already use document-averaged span-window CE, batch 32, AdamW and clip 5;
the objective has not changed. Document shuffling, microbatch 128 versus 64,
data exposure, checkpoint/epoch policy and cache handling differ, so matching
the computation does not reproduce a historical learning trajectory.
`anchor_equivalence_check.json` and saved tensors preserve this distinction.

The following commands document the frozen reproduction sequence using the
existing native environment and preserved local raw data/tokenizer. Valid saved
fits are reused. Run one accelerator queue at a time. All listed stages have
executed; the confirmation and cold processes are terminal. Subsequent commands
document reproduction rather than authorization to restart this study.
Protected final data are never inputs to these commands. Initial contracts,
preparation and historical interrupted dispatches have their commands in the
execution log and machine-readable artifacts.

```powershell
& C:/research/GATE/.venv/Scripts/python.exe -B scripts/cnn_prepare.py --representations token byte --max-seconds 1200
& C:/research/GATE/.venv/Scripts/python.exe -B scripts/check_cnn_data.py --representations token byte
& C:/research/GATE/.venv/Scripts/python.exe -B scripts/cnn_queue.py --plan configs/cnn_stage_a_v1.json --group-size 1 --max-seconds 10800
& C:/research/GATE/.venv/Scripts/python.exe -B scripts/cnn_queue.py --plan configs/cnn_stage_b_v1.json --group-size 1 --max-seconds 36000 --cache-override dpcnn_03=8 --v7-config dpcnn_03
& C:/research/GATE/.venv/Scripts/python.exe -B scripts/check_cnn_scores.py --stage stage_b
& C:/research/GATE/.venv/Scripts/python.exe -B scripts/cnn_queue.py --plan configs/cnn_ablation_v1.json --group-size 1 --max-seconds 21600
& C:/research/GATE/.venv/Scripts/python.exe -B scripts/check_cnn_scores.py --stage ablations
& C:/research/GATE/.venv/Scripts/python.exe -B scripts/cnn_freeze.py --ids textcnn_01 stacked_01 dpcnn_05 tcn_04 pooling_04 byte_hierarchy_03 local_context_03 local_context_03_residual_replace_c98 --quant-ids textcnn_01 byte_hierarchy_03 local_context_03 --extra-plan configs/cnn_ablation_v1.json --common-epoch-pair local_context_03 local_context_03_residual_replace_c98 --epochs 12
& C:/research/GATE/.venv/Scripts/python.exe -B scripts/cnn_queue.py --plan configs/cnn_stage_c_v1.json --group-size 1 --max-seconds 72000
& C:/research/GATE/.venv/Scripts/python.exe -B scripts/check_cnn_scores.py --stage stage_c
& C:/research/GATE/.venv/Scripts/python.exe -B scripts/cnn_confirm_queue.py --freeze configs/cnn_stage_c_v1.json --max-seconds 10800
& C:/research/GATE/.venv/Scripts/python.exe -B scripts/check_cnn_scores.py --stage confirmation
& C:/research/GATE/.venv/Scripts/python.exe -B scripts/cnn_final_errors.py
& C:/research/GATE/.venv/Scripts/python.exe -B scripts/cnn_pair_analysis.py --freeze configs/cnn_stage_c_v1.json
& C:/research/GATE/.venv/Scripts/python.exe -B scripts/cnn_cold.py --freeze configs/cnn_stage_c_v1.json
& C:/research/GATE/.venv/Scripts/python.exe -B scripts/cnn_references.py
& C:/research/GATE/.venv/Scripts/python.exe -B scripts/cnn_analyze.py
& C:/research/GATE/.venv/Scripts/python.exe -B scripts/cnn_runtime.py
```

## Final detection results and seed variability

Each entry below follows seeds **42 / 1337 / 2026**. Canonical post-search
development contains 2,738 attack-labeled and 1,369 benign documents. All 24
FP32 confirmations have TP=2738 and FN=0; TN is 1369 minus the listed FP.
The unchanged primary screen permits at most 13 benign FP. Each model/seed uses
its own designated benign-only nominal 1% threshold and strict `score > threshold`.

| Architecture | Canonical benign FP | Passing seeds | Authored TP /64 | Authored FP /32 |
|---|---|---:|---|---|
| TextCNN01 | 16 / 14 / 17 | 0/3 | 62 / 63 / 61 | 17 / 18 / 16 |
| Stacked01 | 14 / 14 / 18 | 0/3 | 1 / 1 / 0 | 0 / 0 / 8 |
| DPCNN05 | 22 / 16 / 23 | 0/3 | 0 / 0 / 0 | 0 / 0 / 0 |
| TCN04 | 18 / 18 / 21 | 0/3 | 28 / 59 / 0 | 0 / 16 / 0 |
| Pooling04 | 18 / 11 / 22 | 1/3 | 7 / 26 / 7 | 4 / 0 / 1 |
| Byte03 | 21 / 13 / 21 | 1/3 | 12 / 0 / 13 | 0 / 0 / 7 |
| Local/context03 | 11 / 15 / 11 | 2/3 | 1 / 3 / 6 | 0 / 0 / 0 |
| Residual replacement98 | 12 / 14 / 25 | 1/3 | 2 / 58 / 5 | 0 / 16 / 0 |

Five individual FP32 runs pass; **no architecture passes all three fixed seeds**.
Local/context has the lowest worst-seed FPR (15/1369=1.096%), but its authored
attack recall is only 1.56%-9.38%. It cannot be selected as a broadly reliable
guard from its canonical scores. Stacked01's 14-FP seeds miss the boundary by
one document; they remain failures, not statistically established inferiority
or an excuse to relax the criterion. Three fixed seeds describe this execution,
not the probability that a future seed or deployment succeeds.

Source-group bootstrap intervals are saved for every variant, conditional on
its fitted threshold. For example, Local/context42 FPR=0.804% has a 95% interval
0.365%-1.315%, crossing the empirical boundary. Zero observed canonical misses
give degenerate bootstrap recall tails; the supplementary 95% Wilson upper
bound for any missed-attack source-group event is 0.280%. This is a group event
bound, not a document recall interval. Calibration, selection and population
seed uncertainty are excluded; shared templates/semantic dependencies across
source groups remain unresolved. The R1 BERT W8-QAT saved single-run reference
passes with 13 FP, but does not establish a three-seed or matched-training advantage.

The prespecified secondary nominal calibration points remain separate views:

| Architecture | FP at nominal 0.5%, seeds 42/1337/2026 | FP at nominal 2%, same seeds |
|---|---|---|
| TextCNN01 | 7 / 4 / 9 | 34 / 24 / 43 |
| Stacked01 | 9 / 8 / 7 | 24 / 30 / 32 |
| DPCNN05 | 15 / 6 / 8 | 38 / 31 / 41 |
| TCN04 | 10 / 9 / 6 | 35 / 32 / 28 |
| Pooling04 | 5 / 7 / 17 | 33 / 22 / 37 |
| Byte03 | 5 / 3 / 12 | 40 / 21 / 29 |
| Local/context03 | 8 / 5 / 8 | 21 / 27 / 29 |
| Replacement98 | 8 / 8 / 9 | 34 / 30 / 34 |

Only Pooling2026 and Byte42 miss one canonical attack each at nominal 0.5%; all
other secondary FP32 entries have zero misses. A calibration target is not an
observed scoring FPR guarantee. These secondary outcomes do not replace the
frozen primary decision or authorize another tuning loop. Secondary thresholds/
counts and primary-point uncertainty/strata are recoverable in `confirmation_table.json`,
the 33-row `confirmation_table.csv` and per-run raw score files.

## Errors, diagnostic behavior and competing explanations

The independent final error reconciliation checked all 24 runs/33 variants,
raw/prepared text identities and common UTF-8-length bins in 19.016s. Across the
24 FP32 runs, 405 benign rejection incidents concern 195 distinct canonical
documents; 119 are rejected by only one run and none by all 24. Canonical attack
misses remain zero. Across the authored probes, 21 distinct benign texts are
rejected and all 64 positives are missed by at least one run. These overlap
counts are descriptive; no ensemble was selected or evaluated from them.

For seed42, single-/multi-window benign FP are TextCNN 8/1182 versus 8/187,
Stacked 9/1182 versus 5/187, DPCNN 7/1182 versus 15/187, Local/context 5/1182
versus 6/187 and replacement 3/1182 versus 9/187. Byte uses different units:
9/1150 versus 12/219. The common raw-byte bins remove this unit mismatch for
length inspection. Only nine benign documents exceed 8191 UTF-8 bytes; seed42
replacement rejects six and Local/context two. Tiny tail strata and changed
thresholds/models do not establish a universal length effect or its cause.
All recorded source, 55 historically exposed template, insertion-position and
length/window strata remain available in the final error artifact.

The coordinator inspected six illustrative raw benign errors selected by the
saved inspection helper: an outage newsletter ending in unsubscribe commands,
a quoted news discussion, a scheduling/reply-all request, a long corporate
events bulletin, business modeling questions and a social puzzle exchange.
These refine the helper's automatic paraphrases; they are not exhaustive causal
annotations of 195 errors. Ordinary recipient instructions and reported speech
are meaningful hard-benign cases, and their presence alone does not establish
which features caused a model's score.

TextCNN42 detects 62/64 authored positives, but rejects all eight quoted-attack
and all eight code-quoted examples, plus one long benign text. Its two misses
are start/middle attacks in repeated travel background. Local/context42 detects
only one byte-boundary example and misses all positives in the other seven
attack recipe categories. DPCNN misses all 64 positives in every seed, despite
zero canonical misses. Its focused check reconstructed all 96 raw/token inputs
and 187 windows exactly, then reproduced two saved risks on CPU within 4.7e-10
(fixed tolerance 1e-5). No inspected input/checkpoint association fault explains
those scores; two replays do not prove all-case equivalence or diagnose the cause.

Known-template fitting, wording/background dependence, genre shift and calibration
are competing explanations for high canonical recall and weak authored behavior.
The probes are newly authored texts, but share one intent/template and dependent
topic/recipe construction; exact non-overlap is not unseen-family or independent
natural generalization. The shallow model's quotation errors suggest that strong
instruction sensitivity alone is insufficient. Conversely, low benign rejection
with nearly all authored attacks missed is not successful context handling.

## Capacity-aware mechanism comparison

The common-12-epoch, matched-seed pair preserves source groups, document identities,
loss/optimizer and approximately matched parameters/MACs/receptive extent. Its
own-threshold FP differences (Local/context minus replacement) are -1 / +1 / -14,
with identical canonical attack decisions. Mean fixed-seed FPR difference is
-0.341 percentage points; the paired source-group 95% interval is
[-0.730, +0.024] percentage points and includes zero. The seed2026 difference
favors Local/context, while the other two seeds provide no clear separation.
The bootstrap uses common source draws across the fixed seeds and excludes
calibration and population seed uncertainty. Norm inputs, skip locations, dropout
placement/count and width remain confounds. This does not establish that local/
context fusion is superior or identify a novel co-design mechanism. Detailed
paired decisions/probe strata are in `local_context_pair_analysis.json`.

## Numerical sensitivity and measured complete-document costs

All nine fake-W8/input-A8 canonical variants retain zero primary attack misses.
Their FP triples are TextCNN 18/14/16, Byte 22/13/20 and Local/context 10/15/12;
their passing-seed counts remain 0/3, 1/3 and 2/3. Independently reconstructed
own-threshold transitions total five new benign FP and four recovered FP, with
no canonical attack transitions. Authored changes include one lost TextCNN2026
TP and one gained Byte42 TP. Arithmetic and recalibrated threshold movement are
both involved. Full changed IDs and each quantized benign tail relative to its
own floating reference are saved; the frozen AESLC-prefix calibration remains
a representativeness limitation. No native INT8 performance is measured.

The following measurements use seed42, eager CUDA FP32, batch-one windows,
the same six shortest/six longest raw benign documents, five warmups and twenty
recorded repetitions per document. Columns are averages of each document's
twenty complete raw-to-decision observations, split into the two six-document
sets. This extreme sample is not a population latency distribution. Per-document
p50/p95 and raw stage/repetition/risk/decision records passed independent checking.

| Architecture | FP32 parameter payload, MB | Mean complete-document Conv/Linear MACs, millions | Warm short / long mean, ms | All-12 mean, ms |
|---|---:|---:|---:|---:|
| TextCNN01 | 8.062 | 20.329 | 3.615 / 31.903 | 17.759 |
| Stacked01 | 7.930 | 9.487 | 4.196 / 32.152 | 18.174 |
| DPCNN05 | 14.149 | 68.100 | 15.507 / 108.376 | 61.941 |
| TCN04 | 8.511 | 56.919 | 9.698 / 67.168 | 38.433 |
| Pooling04 | 8.032 | 17.618 | 7.348 / 43.789 | 25.569 |
| Byte03 | 2.210 | 383.835 | 12.792 / 398.434 | 205.613 |
| Local/context03 | 13.240 | 125.018 | 8.265 / 54.804 | 31.535 |
| Replacement98 | 13.226 | 123.862 | 8.989 / 62.042 | 35.515 |

MB means decimal 10^6 bytes. MACs are analytical Conv/Linear counts averaged
over all 4,107 canonical raw documents, including all padded representation
windows (5,308 token versus 6,130 byte). They exclude norm, pooling, nonlinear,
mask, preprocessing, transfer and aggregation costs; the timing includes the
recorded raw preparation/H2D/network/return boundary. TextCNN's all-12 stage
means are 3.748/1.408/12.410/0.176ms respectively. Local/context's are
3.783/1.115/26.476/0.144ms. These costs should not be collapsed into kernel-only
speedup. Byte storage is small, but its complete-document MAC and measured long
cost are large; fewer embedding parameters do not imply lower end-to-end cost.

Ideal packed W8 weights plus remaining FP32 parameters/scales would require
2,138,888 bytes for TextCNN, 564,316 for Byte and 3,440,280 for Local/context.
These exclude alignment, activations and instructions. Actual fake-quantized
tensors remain FP32; these are analytical payloads, not measured device memory.

The worst-seed quality/storage/MAC frontier and the measured complete-latency/
storage frontier both retain TextCNN, Stacked, Byte and Local/context. Neither
frontier has an all-three-seed feasible member. Individual passing-seed frontiers
remain descriptive and do not justify choosing a lucky seed. Cold profiling
observed eight fresh processes: spawn-to-exit ranges 3.776-10.829s and child
start-to-decision 3.208-10.034s. All eight risks exactly match the corresponding
warm reference and no decision changes. Imports, artifact validation, model load,
data/tokenizer preparation and first execution are separately recorded; caches
were not flushed and each uses one short document. These are not cold-boot
distributions or expected deployment startup guarantees.

## Coverage, evidence totals and completion boundary

All 16 implemented branches were screened and received a representative full
12-epoch inner opportunity: TextCNN, stacked, residual, pyramid, byte hierarchy,
dilated, iterated dilated, TCN, depthwise, grouped, large kernel, gated, multiscale,
local/context, dense-connected and pooling. The final
eight choices preserve conventional, contextual, causal, pooling and byte
diversity; the other branches were not given three-seed full-training confirmation
because the eight-finalist allowance was allocated using frozen inner evidence,
trajectory and costs. Cross-window/sentence hierarchical document convolution,
conditional/multi-representation fusion, true character inputs and separate task-specific pretraining,
distillation/augmentation, changed document aggregation and independent withheld-
family training remain untested. They were not necessary to finish this bounded
first stage, and no post-confirmation adaptation was introduced.

Actual learning is 96 short screens +23 full inner fits +one learned replacement
+24 fixed full-training fits =144 unique runs; equivalent timing fits are reused.
Confirmation covers 24 FP32 +nine fake8 variants, each with calibration,
development and diagnostic scores (99 raw-score files). There are eight warm
profiles (1,920 timed document repetitions) and eight cold observations. The
complete table preserves seven distinct unsuccessful implementation/resource
attempts in addition to successful runs; archival copies are deduplicated without
discarding distinct failures. Recorded accelerator process walls total
16,675.152s (4.632h), including confirmation/cold/support/stopped jobs once;
completed learning fit/scoring sums 12,674.951s. Both are below the 48h ceiling.
Unknown initial refusal walls and small unrecorded startup overhead remain
unquantified rather than zero. Substantial CPU preparation/check/analysis times
are recorded separately in the execution log and artifacts; no long CPU learning
campaign, vendor build, target API call, paid service or physical operation ran.

Final evidence: `analysis.json`, both CSV tables, `final_error_inspection.json`,
`local_context_pair_analysis.json`, `saved_score_check_confirmation.json`,
`final_confirmation_review.json`, `cold_process_timings.json`,
`cold_process_check.json` and `runtime_ledger.json`. Current 40-file source bytes
are additionally preserved by manifest
`2ed6f630bfadb0b83bd02d78805a0a851383a442d85ff0684aa14183e7cb3f27`;
the three earlier manifests remain, including the 39-file diagnostic snapshot
`10b361ddb2b46262a9a4316caa15b4a1cefb6b91af7511818913a4f4eea894f4`.
Hashes do not reconstruct unsnapshotted older
runner versions. Ignored local data/checkpoints/results and the pinned tokenizer/
native environment remain required for reproduction; a fresh Git clone is
insufficient. The protected final partitions remain closed.

## Next FPGA-centered research decision

The recommendation is **a bounded negative result for a confirmed guard, with
useful conventional comparators**. No deployment or large accelerator build is
justified by this study. Three roles merit retention for a later explicitly
defined research question; this ordering reflects complementary comparisons,
not a statistically established quality ranking or predicted FPGA performance:

| Retained role | Design and reason | Condition before a next hardware study |
|---|---|---|
| Economical conventional contextual baseline | Stacked01: the smallest counted token Conv/Linear workload, 9/12 passing inner epochs, and 14/14/18 canonical FP. TextCNN stays as the shallow/probe-sensitive anchor. | Resolve weak authored attack detection; any proposed mechanism must compare against competent conventional training with the same information. |
| Quality-leading contextual candidate and conventional control | Local/context03 plus the residual replacement98: 2/3 versus 1/3 canonical passes, common 12 epochs and near-matched capacity/RF. The replacement is retained as the stronger capacity-matched conventional control even though its worst-seed frontier is dominated. | Specify one fusion/buffering benefit that could be falsified, preserve all input/decision/utility boundaries and account for remaining skip/norm/dropout confounds. Severe probe misses and paired CI crossing zero prevent a fusion-benefit claim. |
| Parked storage/representation alternative | Byte03: 2.210MB FP32 and 0.564MB ideal mixed W8 payload, with tokenizer-free raw byte encoding; only 1/3 canonical passes and large complete-document work/latency. | Reopen only if byte representation becomes central to an approved question, with a complete L1024 transport/retention chain and comparable raw documents. It is not the first hardware-efficiency choice from these results. |

Pooling04 remains a secondary evidence-retention discriminator: streaming top-k
versus max should count comparison/state costs and preserve localized attacks.
It is not promoted over the retained roles by its one passing seed. DPCNN and
TCN receive no first-build preference from their current quality/probe/cost
results. These decisions preserve negative evidence rather than selecting an
easy model solely to obtain a synthesis result.

The [mapping note](cnn_mapping_note.md) reconciles the complete results with
embedding access, convolution, normalization/fusion, alignment buffers,
top-k/max pooling, document retention, decision logic and host transport. A
time-multiplexed PL chain is a conditional implementation starting point, not
a resource prediction; AIE/heterogeneous routing requires actual operator,
device and flow validation. The previous BERT UART/RTL/platform evidence is
not a CNN result. Vendor attempts remain zero by a bounded-study choice,
without claiming tool unavailability or physical performance.

The closest competing explanation is learning historically repeated instruction
and corpus cues rather than a transferable distinction between operative
injection and legitimate quoted/recipient instructions. The decisive next
scientific experiment would first fix an independently sourced development
comparison with withheld mechanisms/near-duplicates, matched benign quotations,
ordinary instructions and controlled length/positions, then compare the
conventional anchors and contextual pair under equal information/effort with
prospective utility and numerical bounds. It must separately establish target
task effects if protection is claimed. Those new inputs/training choices are a
proposal, not a post-confirmation extension of this frozen study.

Only if that comparison establishes a useful detector boundary should one
explicit FPGA-aware evidence/retention or fusion hypothesis receive a small
complete-chain implementation discriminator. Count raw preparation, transfers,
memory/state, compute, all-window aggregation and enforcement/retention, compare
the conventional path, and stop if the proposed mechanism provides no material
predeclared benefit. Standard CNN fusion, top-k pooling or byte encoding alone
does not establish novelty. The present evidence establishes an executed
architecture discovery stage and an unresolved generalization/co-design
limitation; it does not complete SENTRY's new-method and FPGA-realization purpose.
