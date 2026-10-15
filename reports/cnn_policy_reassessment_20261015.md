# CNN candidate reassessment under the new evaluation policy

**LocalContext PCA + KD merits a provisional role as the leading CNN backbone
for a bounded FPGA co-design investigation. C98 should remain a conventional
control and a challenger at strict false-positive operating points. Matched
BERT remains the strongest detection reference.** This is a retrospective
research-selection recommendation, not a pass of the original adoption study
or a claim that any candidate is suitable for deployment.

The user requested this reassessment on 15 October 2026 after the original
results were known. The [new evaluation policy](../EVALUATION_POLICY.md)
distinguishes exploratory advancement, a scientific contribution and deployment.
It supplies no replacement universal recall/FPR threshold. The original
[adoption report](cnn_adoption_study.md), its frozen thresholds and negative
primary verdict remain unchanged. A recommendation to investigate a backbone
under a different decision stage does not reverse that result.

## Decision and comparison

The supporting question is whether an existing CNN has enough meaningful
detection behavior and a plausible quality/resource trade-off to justify one
bounded co-design investigation. The comparison includes all four focused CNN
structures, supervised and training-intervention controls, both BERT reference
roles, all final seeds and each finalist's own FP32 companion. The earlier broad
architecture sweep uses different selection data and cannot rank candidates
on this new panel.

No new numerical acceptance cutoff is introduced. A useful discriminator is
whether a candidate offers reproducible discrimination, a tractable exact
numerical reference and a specific hardware question worth testing against
competent alternatives. A poor trade-off or a failure that the proposed
hardware work cannot address can falsify the case for further investment.
This selection work supports the FPGA-centered contribution; it does not
establish novelty. Standard CNNs, distillation, PCA initialization and a
conventional port remain supporting methods.

| Item | Status in this reassessment |
|---|---|
| Historical 99% recall / 1% FPR conjunction | Preserved original-study requirement; not a new universal research veto |
| Nominal .005 / .01 / .02 calibration | Existing reporting policies; observed FPR is reported separately |
| Observed .5% / 1% / 2% / 5% FPR ROC points | Descriptive comparison points, not deployed thresholds |
| Hard-benign and withheld mechanisms | Failure analysis with denominators and ancestry limits; no replacement fixed recall floor |
| Three training seeds | Variability evidence; no rule that a narrow one-seed miss excludes research |
| One-percentage-point BERT margin and 50% storage target | Historical study choices; no universal admission condition |
| Exact export and arithmetic qualification | Required to reuse a claimed numerical implementation; already available for the final CNNs |
| Physical FPGA speed, energy or deployment protection | Separate claims requiring corresponding evidence |

## All final seeds at unchanged calibration policies

Each row predicts the same 1,000 attacks and 2,000 benign documents. Each format
has a global threshold fitted to the same separate 3,000 benign calibration
documents. Entries are recall / observed FPR. Nominal calibration rate is not
the measured confirmation FPR.

| Candidate | Seed | Nominal .5% | Nominal 1% | Nominal 2% |
| --- | --- | --- | --- | --- |
| LocalContext PCA + KD | 42 | 41.30% / 2.05% | 46.00% / 2.40% | 56.00% / 3.15% |
| LocalContext PCA + KD | 1337 | 50.30% / 2.45% | 56.50% / 3.10% | 68.40% / 4.90% |
| LocalContext PCA + KD | 2026 | 41.70% / 2.20% | 49.50% / 2.70% | 56.00% / 3.25% |
| C98 KD | 42 | 34.30% / 1.45% | 40.30% / 1.90% | 49.70% / 2.55% |
| C98 KD | 1337 | 32.30% / 2.90% | 38.90% / 3.55% | 52.90% / 5.85% |
| C98 KD | 2026 | 21.80% / 1.50% | 30.70% / 1.95% | 40.00% / 2.30% |
| Matched BERT | 42 | 78.20% / 1.75% | 85.70% / 2.50% | 93.40% / 4.15% |
| Matched BERT | 1337 | 45.50% / 1.15% | 69.00% / 2.40% | 82.60% / 3.90% |
| Matched BERT | 2026 | 74.50% / 1.10% | 83.30% / 1.80% | 90.20% / 2.95% |
| Historical W8 BERT | fixed | 6.90% / 1.10% | 9.70% / 1.40% | 12.90% / 1.90% |

LocalContext's primary recall spans 41.3–50.3%, C98's 21.8–34.3%, and BERT's
45.5–78.2%. BERT's 32.7 percentage-point range at this calibrated point is
larger than LocalContext's 9.0-point range, but that alone does not make
LocalContext the better detector. Score ranking and low-FPR behavior must
also be compared. Secondary settings expose continuous trade-offs rather
than furnishing alternative chances to pass the original protocol.

## Score discrimination at matched observed FPR

These empirical ROC points enumerate complete tied-score tiers on the already
released confirmation labels. Each is the maximum observed recall reachable
without exceeding the stated FPR cap. There is no interpolation through a tie,
and the script does not propose these label-informed cutoffs for deployment.
They describe this panel's score separation; they are not independent
validation of a newly selected threshold. AUC measures ranking over the full
FPR range and can conceal weak performance in the low-FPR tail.

| Candidate | Seed | ROC AUC | Recall at .5% FPR | At 1% | At 2% | At 5% |
| --- | --- | --- | --- | --- | --- | --- |
| LocalContext PCA + KD | 42 | 0.9427 | 12.10% | 22.40% | 40.50% | 74.40% |
| LocalContext PCA + KD | 1337 | 0.9395 | 17.40% | 24.90% | 42.90% | 68.60% |
| LocalContext PCA + KD | 2026 | 0.9390 | 8.50% | 21.50% | 37.60% | 68.00% |
| C98 KD | 42 | 0.9270 | 17.40% | 24.90% | 43.10% | 64.30% |
| C98 KD | 1337 | 0.9155 | 5.70% | 15.00% | 25.40% | 48.00% |
| C98 KD | 2026 | 0.9158 | 9.40% | 13.40% | 31.60% | 58.50% |
| Matched BERT | 42 | 0.9809 | 37.60% | 60.30% | 82.90% | 94.70% |
| Matched BERT | 1337 | 0.9729 | 26.00% | 43.50% | 63.00% | 90.70% |
| Matched BERT | 2026 | 0.9847 | 66.40% | 74.00% | 85.30% | 93.80% |
| Historical W8 BERT | fixed | 0.7625 | 2.60% | 6.20% | 13.60% | 25.60% |

Matched BERT is stronger at every tabulated matched-FPR point in every seed.
The apparent LocalContext seed-1337 advantage at the registered primary
threshold (+4.8 recall points) disappears at matched observed FPR: at 1%,
BERT reaches 43.5% recall versus LocalContext's 24.9%. That primary-point
advantage therefore is not evidence of superior ranking.

LocalContext has higher overall AUC than C98 in all three seeds and better
1%, 2% and 5% FPR points in seeds 1337 and 2026. At .5% FPR in seed 2026,
C98 also has a small advantage, 9.4% versus 8.5%. Seed-42 C98 reaches 24.9%
recall at 1% FPR versus LocalContext's 22.4%; at .5% the comparison is
17.4% versus 12.1%. C98 cannot be declared uniformly dominated. At 5% FPR,
LocalContext is stronger in every seed, but that false-alarm rate may be
unacceptable in an actual application. No deployment base rate or accepted
error cost has been specified.

The CNNs' AUCs around .92–.94 establish nontrivial ranking on this constructed
task. They do not imply strong low-FPR detection: at 1% observed FPR,
LocalContext detects only 21.5–24.9% and C98 13.4–24.9%. This limitation
persists after removing the old 99% requirement.

## Hard benign and application strata

The benign panel contains 1,600 natural documents, including 57 human-recipient
requests, and 400 authored retrospective discussion examples. Hard benign is
those 400 plus the 57 requests. The following AUC comparison always uses the
same 1,000 attacks while changing the benign comparison population. The
constructed-only column exposes how easy natural backgrounds affect pooled AUC.
False positives use the unchanged primary calibration policy.

| Candidate | Seed | AUC vs constructed benign | AUC vs hard benign | Constructed FP | Natural FP | Hard-benign FPR |
| --- | --- | --- | --- | --- | --- | --- |
| LocalContext PCA + KD | 42 | 0.7859 | 0.8078 | 41/400 | 0/1600 | 8.97% |
| LocalContext PCA + KD | 1337 | 0.7583 | 0.7857 | 48/400 | 1/1600 | 10.50% |
| LocalContext PCA + KD | 2026 | 0.7520 | 0.7809 | 44/400 | 0/1600 | 9.63% |
| C98 KD | 42 | 0.7277 | 0.7581 | 29/400 | 0/1600 | 6.35% |
| C98 KD | 1337 | 0.6364 | 0.6779 | 58/400 | 0/1600 | 12.69% |
| C98 KD | 2026 | 0.6761 | 0.7127 | 30/400 | 0/1600 | 6.56% |
| Matched BERT | 42 | 0.9201 | 0.9279 | 31/400 | 4/1600 | 7.00% |
| Matched BERT | 1337 | 0.8939 | 0.9048 | 23/400 | 0/1600 | 5.03% |
| Matched BERT | 2026 | 0.9402 | 0.9467 | 21/400 | 1/1600 | 4.60% |

LocalContext's constructed-only AUC is .7520–.7859, versus .6364–.7277 for
C98 and .8939–.9402 for BERT. All three families distinguish more than a
constant or reversed classifier on these examples, but the CNNs struggle with
the intended distinction between discussing an instruction and issuing an
operative assistant redirection. LocalContext rejects 10.25–12% of the
400 discussions; C98 rejects 7.25–14.5%. BERT rejects 5.25–7.75%.
These are one-author constructed discourse families, not population rates.
Near-zero rejection of the natural subset cannot substitute for this stress,
and the 57 natural requests alone do not precisely measure over-defense.

The primary email subset has 200 attacks and 400 benign documents.
LocalContext's primary recall/FPR is 42.5%/2.5%, 47%/3%, 40%/2.25%; C98's
is 32.5%/1.5%, 28%/2.5%, 16.5%/1.5%; BERT's is 76.5%/2.5%, 40%/1%,
61%/1%. Email-only AUC remains stronger for BERT (.9716–.9795) than
LocalContext (.9293–.9333) or C98 (.9054–.9212). Most of the pooled panel is
article material, so pooled values alone do not establish email performance.

Among seed-42 LocalContext benign inputs, FP counts are 1/1,425 for one window,
21/455 for two or three, 16/85 for four to seven, and 3/35 for eight or more.
That association motivates attention to full-document aggregation and context;
it does not isolate a causal length effect because genre, construction and
wording also vary. All tails remain covered with L=256, 254 content tokens,
64-token overlap, stride 190, and maximum risk across all windows. No quote
exemption or trusted span is supplied at inference.

At the original primary policy, code-mechanism recall spans 32.8–66.4% for
LocalContext, 16.8–32% for C98, and 49.6–87.2% for BERT. Each uses 125 attacks
on 125 backgrounds with 21 payload ancestries. Role and encoding also have
125 attacks each but only 4 and 13 ancestries. Their low observed recalls are
warnings; the repeated variants do not make the mechanism evidence precise.
Earlier teacher/fine-tuning semantic exposure remains possible. These strata
are project-held lexical groups, not mechanisms proven unseen to every model.

## Adaptation and simpler CNN alternatives

The following same-panel development evidence explains the shortlist rather
than inventing missing confirmations. Each row has 400 attacks and 800 benign
documents, with thresholds from 1,000 separate development benign documents.
These examples influenced selection and are not fresh validation.

| Development candidate | Primary recall | Observed FPR | AUC | Descriptive recall at 1% FPR |
| --- | --- | --- | --- | --- |
| Original TextCNN | 0.25% | 0.38% | 0.5935 | 1.00% |
| Original Stacked | 0.25% | 0.38% | 0.4683 | 0.50% |
| Original LocalContext | 0.00% | 0.00% | 0.6608 | 6.25% |
| Original C98 | 0.25% | 0.00% | 0.6567 | 7.00% |
| Supervised TextCNN | 27.00% | 1.12% | 0.9316 | 21.00% |
| Supervised Stacked | 36.00% | 1.12% | 0.9303 | 35.50% |
| Supervised LocalContext | 47.00% | 2.00% | 0.9368 | 37.75% |
| Supervised C98 | 41.25% | 0.38% | 0.9311 | 51.50% |
| LocalContext KD | 48.00% | 0.75% | 0.9172 | 51.00% |
| C98 KD | 42.75% | 0.88% | 0.9366 | 47.25% |
| LocalContext PCA + KD | 51.75% | 0.88% | 0.9450 | 54.50% |
| Matched BERT LR 2e-5 | 68.75% | 0.25% | 0.9942 | 90.25% |
| Matched BERT LR 5e-5 | 79.00% | 0.00% | 0.9937 | 94.00% |

The enriched supervised stage yields substantial gains over the original
CNNs. Its changed fitting mixture and legal document-level loss were introduced
together, so the gain cannot be attributed solely to data diversity. TextCNN
and Stacked show worthwhile learning, not architectural impossibility. Stacked
is a useful economical alternative: two width-3 convolutions at C=64, no
LayerNorm/residual branches and a simpler streaming question. It has no new
three-seed confirmation or final-route timing on this task, so it cannot be
declared a better deployed trade-off than the two finalists. TextCNN has three
parallel branches and lower primary recall here; it remains a shallow control.

Distillation does not improve every discrimination metric. LocalContext KD
raises primary recall from 47% to 48% and lowers FPR from 2% to .75%, yet
overall AUC falls from .9368 to .9172. C98 KD raises primary recall from
41.25% to 42.75% while descriptive recall at 1% FPR falls from 51.5% to
47.25%. These are selected development observations, not a clean universal
benefit from KD. LocalContext PCA + KD improves AUC to .9450 and the primary
point to 51.75%/.875%; vocabulary pretraining, scaling and conditioning change
together, so PCA's individual causal contribution is unresolved.

The teacher is the same-FIT refined pretrained BERT, useful on development
but not security-qualified. Teacher outputs were cached only on FIT; the
teacher is absent at inference. The LocalContext final recipe uses the
pretrained WordPiece PCA-96 initializer plus KD for seven epochs; C98 uses
KD for four. BERT uses four epochs at LR 5e-5 with its preserved five-epoch
scheduler horizon. All final recipes repeat seeds 42/1337/2026. Different
pretraining and training interventions prevent interpreting the final
LocalContext–C98 contrast as a topology-only ablation.

## Calibration uncertainty and precision

The existing paired source/payload/calibration sensitivity analysis is reused;
there is no need to resample the same experiment again. LocalContext-minus-BERT
primary recall differences are -36.9, +4.8 and -32.8 percentage points. For
seed 1337 the source interval is [1.10, 8.69], but the payload interval is
[-5.94, 13.83] and the calibration-refit interval [-11.20, 15.80]. The apparent
advantage is sensitive to grouping and the calibration tail. C98's differences
are -43.9, -13.2 and -52.7 points. Separate sensitivity intervals are not a
joint dependence model, noninferiority proof or multiple-selection correction.

Each seed repeats the same documents, not an independent draw of 1,000 new
attacks. The confirmation panel has 2,000 source groups and 52 attack payload
groups; calibration has 3,000 source groups. Source and payload grouping do
not remove dependence from the one-author discussion families. New AUC/ROC
numbers are descriptive and have no newly claimed confidence intervals.

Every CNN's FP32 companion also shows substantial limitations. At separately
calibrated primary thresholds, W8/A8 changes LocalContext TP by +31/+4/+11
and C98 by -6/+5/0; decision changes are 35/9/15 and 13/15/7 respectively.
These include threshold changes and are not pure arithmetic causal effects.
There is no evidence that simply switching to FP32 resolves the detection
gap. W8/A8 dot products use native INT8 inputs/weights and INT32 accumulation,
with explicit FP32 scales, bias, embedding dequantization, LayerNorm, nonlinear,
pooling and softmax work. This is a concrete mixed arithmetic reference,
not an all-integer FPGA design.

## Hardware trade offs and measured software timing

| Quantity | LocalContext | C98 | Matched BERT or boundary |
|---|---|---|---|
| Actual packed parameters and scales | 3,440,780 B | 3,439,184 B | Hypothetical precision-matched W8 BERT: 11,377,100 B; not numerically qualified or packed |
| W8 embedding plus row scales | 3,052,200 B | 3,052,200 B | About 88.7% of each CNN payload; gather/banking matters |
| Conv/Linear MACs per L=256 window | 96.731 M | 95.836 M | BERT 939.590 M; excludes other operators and is not a speed prediction |
| Receptive field and lookahead | 33 tokens, 16 centered | 33 tokens, 16 centered | Full-window pooling and full-document maximum still apply |
| Dataflow complexity | Local/context branches and fusion; six tokenwise LayerNorms | Six residual stages; six tokenwise LayerNorms and skip alignment | Neither has a complete implemented CNN PL chain |
| Qualified complete-document CUDA Graph mean | 5.211 ms W8/A8; 4.415 ms FP32 | 5.602 ms W8/A8; 4.557 ms FP32 | BERT FP32 5.106 ms |
| Qualified FP32 CPU mean, 1 / 4 threads | 23.414 / 21.257 ms | 25.570 / 22.668 ms | BERT CPU failed numerical qualification; no qualified CPU timing |

C98 saves only 1,596 bytes (about .046%) relative to LocalContext and about
.93% of counted Conv/Linear work; it has no compelling storage advantage
between these final candidates. Its seed-42 stricter-FPR advantage still
justifies retaining it as a quality/control comparison. LocalContext is about
2.1% slower than BERT's measured Graph mean in its intended format; C98 about
9.7% slower. That neither proves an FPGA disadvantage nor establishes an
FPGA win. FP32 CNN routes are separate numerical/performance alternatives,
not native INT8 evidence.

Timing covers identical 29 raw documents with 20 retained repetitions per
route and 1–17 windows, including tokenization, copies, all windows, aggregation
and decision. The sample is length-stratified, not a deployment-frequency
estimate, and startup is separate. Ten cold child observations over five
routes are retained in the original profiling audit. No historical timing is
assigned to a new checkpoint.

The target is VEK280 xcve2802-vsvh1760-2MP-e-S with a 200 MHz objective,
a recorded 64 MiB DDR arena and nominal 8.72 MB aggregate on-chip memory budget.
These are setup constraints, not allocated CNN capacity or achieved clock.
The candidate payloads' conditional approximately 69.8% reduction against a
hypothetical all-matrix-W8 BERT is a plausible research motivation, not a
mandatory 50% gate or a demonstrated complete-system benefit. Branch/skip
liveness, embedding ports, normalization, host document retention, packet
transport and all operator costs remain part of the proposed schedule.
Whole-document work scales with N(n)=1+ceil(max(0,n-254)/190) windows.
The current UART path can dominate per-window network latency; lowering MACs
alone may therefore produce little complete-document gain.

## Candidate roles and next discriminating action

| Candidate | Research role under the new policy | Reason and condition |
|---|---|---|
| LocalContext PCA + KD | Provisional leading CNN for one bounded co-design investigation | Reproducible mixed-arithmetic export, meaningful discrimination and strongest overall CNN ranking; hard-benign and low-FPR weaknesses constrain its claims |
| C98 KD | Conventional capacity control and strict-FPR challenger | Similar footprint/work, weaker overall ranking, but seed-42 tail advantage means it is not uniformly dominated |
| Stacked supervised | Reserve economical baseline | Simpler source graph and learned discrimination; lacks comparable repeated confirmation/timing on this panel |
| TextCNN supervised | Shallow diagnostic baseline, lower investment priority | Learns from richer supervision, but lower measured development recall and branching costs give no current reason to lead hardware work |
| Matched BERT | Required strong detection reference and teacher | Best tabulated equal-FPR discrimination and hard-benign ranking; larger conditional storage/work, variable calibration behavior, no deployment qualification |
| Historical W8 BERT | Regression anchor | New-panel AUC .7625 and primary recall 6.9%; insufficient task transfer for the principal quality reference |

If one concrete CNN is needed for a bounded feasibility investigation, retain
the already prespecified seed-42 LocalContext artifact, rather than selecting
seed 1337 because its primary point is highest. Its checkpoint is
`results/cnn_adoption_v1/runs/local_context_03_final_s42_r2/checkpoint_latest.pt`,
SHA256 `50eefbb95597519ef268d0e7262c05f470b947076c167c7603f762a5060a08eb`;
its W8/A8 manifest SHA256 is
`e6b79b363f74c23bdefd4a796084ff770782aab0d55f7668e6ae38ddbf547744`.
The old alpha-.005 threshold 0.9998701810836792 remains a recorded comparison
point, not a newly endorsed deployment setting. The complete teacher,
initializer and all-seed checkpoint provenance remains in the original report.

The smallest hardware discriminator is an exact schedule/allocation and
complete-chain simulation comparison of frozen LocalContext branch fusion
against C98 residual scheduling, including embedding gathers, FP32 exceptions,
host/transport and full-document retention. Compare peak simultaneously live
storage and complete-document cost at the same arithmetic and workload;
retain numerical decisions. If LocalContext's joins/normalization erase its
plausible benefit, stop treating its small parameter payload as sufficient
reason to invest. This is supporting feasibility work, not itself the novel
FPGA-aware security contribution. It was not executed as part of this
reevaluation.

For learning rather than mapping, the already identified attribution/context
versus representation question remains the smallest useful discriminator.
Any new examples, training, threshold selection or claimed validation need a
separate prospective design: this confirmation panel is now known. Reusing it
to tune a more favorable policy cannot create new independent evidence.

## Reproduction and evidence

The local analysis reads existing released score JSONL and checks every saved
score hash, all sixteen independent-audit summary hashes, fixed calibration
counts and the unchanged frozen scoring sources. It covers 16 confirmation
targets and 36 development target/format records, including aliases rather
than 36 independent trained models. Perfect/reversed/tied/saturated score
fixtures compare ROC integration with rank AUC and brute-force strict cuts.

```powershell
& 'C:/research/GATE/.venv/Scripts/python.exe' -B scripts/reassess_cnn_candidates.py --output results/cnn_adoption_v1/policy_reassessment_20261015/reproduction.json
& 'C:/research/GATE/.venv/Scripts/python.exe' -B scripts/update_handoff.py --check
```

Use a fresh output name; the analyzer refuses to overwrite an artifact. The
first execution was denied access to the WSL network path before analysis;
the approved local-path execution completed. The initial full analysis took
9.961 CPU-process wall seconds; the expanded hard-benign/genre analysis took
10.989 seconds. Both are retained. An independent check of all 16 confirmation targets passed
using direct attack/benign pairwise AUC and benign order statistics for the
four ROC caps, taking another 2.392 seconds. Its first attempt stopped at a
Windows-path metadata lookup before producing output; the corrected check
changed no score or analysis metric. There were zero new learning jobs,
accelerator jobs, vendor builds, target-LLM calls or physical-device operations.
Review/documentation time is additional and untimed. Original resource records
are preserved, including their previously declared unknown intervals.

Authoritative evidence:

- [Retrospective metrics and all input hashes](../results/cnn_adoption_v1/policy_reassessment_20261015/metrics_v2.json).
- [Independent new ROC calculation check](../results/cnn_adoption_v1/policy_reassessment_20261015/independent_roc_check.json).
- [Independent final score check](../results/cnn_adoption_v1/independent_confirmation_score_check_r2.json).
- [Existing paired source, payload and calibration sensitivity](../results/cnn_adoption_v1/final_confirmation_comparison_summary_r2.json).
- [Qualified software profiling](../results/cnn_adoption_v1/final_profiling_audit_r2.json).
- [Actual exports and target mapping audit](../results/cnn_adoption_v1/final_hardware_adoption_audit.json).
- [Original preserved adoption outcome](cnn_adoption_study.md) and [new standing policy](../EVALUATION_POLICY.md).
