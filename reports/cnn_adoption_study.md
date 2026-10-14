# CNN adoption decision

**DO NOT ADOPT THESE CNNs YET — measured quality limitation.**

The frozen LocalContext PCA + KD and C98 KD candidates fail the 99% recall /
1% FPR primary requirement in every seed in their intended W8/A8 format.
All six also fail the hard-benign bound and adequately populated code-mechanism
recall. Matched BERT is usually stronger but also fails the primary bounds
in all three seeds. No CNN is selected for the next FPGA implementation stage.
The complete sixteen-target confirmation set passed independent checks; all
six declared paired sensitivity analyses completed. This closes this bounded
backbone decision, while SENTRY's new FPGA-aware method remains unresolved.

Study executed 14–15 October 2026 (local time); recorded process timestamps use UTC.

## Confirmation quality

Each seed uses the same 1,000 attacks and 2,000 benign documents, a separate
3,000-document benign calibration population and the frozen alpha 0.005 rule.
Email denominators are 200 attacks / 400 benign; hard benign is 457 documents.

| Intended candidate | Seed | Recall | FPR | Hard-benign FP/457 | Email recall/FPR | Code recall |
|---|---:|---:|---:|---:|---:|---:|
| LocalContext PCA + KD | 42 | 41.30% | 2.05% | 41/457 | 42.50% / 2.50% | 66.40% |
| LocalContext PCA + KD | 1337 | 50.30% | 2.45% | 48/457 | 47.00% / 3.00% | 59.20% |
| LocalContext PCA + KD | 2026 | 41.70% | 2.20% | 44/457 | 40.00% / 2.25% | 32.80% |
| C98 KD | 42 | 34.30% | 1.45% | 29/457 | 32.50% / 1.50% | 32.00% |
| C98 KD | 1337 | 32.30% | 2.90% | 58/457 | 28.00% / 2.50% | 31.20% |
| C98 KD | 2026 | 21.80% | 1.50% | 30/457 | 16.50% / 1.50% | 16.80% |
| Matched BERT | 42 | 78.20% | 1.75% | 32/457 | 76.50% / 2.50% | 77.60% |
| Matched BERT | 1337 | 45.50% | 1.15% | 23/457 | 40.00% / 1.00% | 49.60% |
| Matched BERT | 2026 | 74.50% | 1.10% | 21/457 | 61.00% / 1.00% | 87.20% |

| Descriptive fixed-seed summary | Pooled recall/FPR | Worst recall | Worst FPR |
|---|---:|---:|---:|
| LocalContext PCA + KD | 44.43% / 2.23% | 41.30% | 2.45% |
| C98 KD | 29.47% / 1.95% | 21.80% | 2.90% |
| Matched BERT | 66.07% / 1.33% | 45.50% | 1.75% |

Pooled values describe three predictions on the same documents. They are not
3,000 independent attacks, 6,000 independent benign units or a sampled-seed
population; every individual seed still controls readiness.

## Decision requirements

| Requirement | LocalContext | C98 | Evidence/implication |
|---|---|---|---|
| Primary 99% recall / 1% FPR, every seed | Fail | Fail | All six intended-format rows fail both bounds |
| Primary email 99% recall / 1% FPR | Fail | Fail | All seeds fail recall; all intended CNN email FPRs exceed 1% |
| Hard-benign FPR at most 5% | Fail | Fail | Every intended CNN seed exceeds 22/457 allowable false positives |
| Adequate withheld code recall at least 95% | Fail | Fail | 125 attacks / 125 backgrounds / 21 payload groups, every seed below 95% |
| Role-delimiter / encoding mechanism readiness | Uncertain | Uncertain | Only 4 / 13 payload groups, below the frozen minimum 20; never passed by default |
| At most 1 pp observed recall loss versus matched BERT | Fail across seeds | Fail across seeds | Local loses 36.9 / -4.8 / 32.8 pp; C98 loses 43.9 / 13.2 / 52.7 pp |
| Reproducible intended export and actual arithmetic | Qualified | Qualified | Six actual packed exports, native checks, full intended-format scoring |
| Bounded competent software route | Qualified | Qualified | Matched complete-document CUDA Graph and CPU routes; no intended-format GPU latency win |
| Material target-specific hardware benefit | Conditional; unproven | Conditional; unproven | >69.7% hypothetical precision-matched payload saving; target allocation/closure/transport remain unmeasured |
| Physical FPGA/energy/protection or novel-method claim | Not established | Not established | Separate later claims, not prerequisites for this negative backbone decision |

A measured primary failure is sufficient for this negative decision. Sparse
mechanisms and unresolved hardware feasibility remain explicit; they do not
turn the completed quality comparison into an unexecuted or positive result.

## Exact frozen checkpoint, format and threshold identities

Priority was LocalContext then C98, with deployment seed 42 fixed beforehand.
The following rejected candidate references are retained for reproducibility:

- **LocalContext PCA + KD, seed 42:** `results/cnn_adoption_v1/runs/local_context_03_final_s42_r2/checkpoint_latest.pt`;
  checkpoint SHA256 `50eefbb95597519ef268d0e7262c05f470b947076c167c7603f762a5060a08eb`;
  format `w8a8_export`, global threshold `0.9998701810836792`.
  Export-manifest SHA256 `e6b79b363f74c23bdefd4a796084ff770782aab0d55f7668e6ae38ddbf547744`.
- **C98 KD, seed 42:** `results/cnn_adoption_v1/runs/local_context_03_residual_replace_c98_final_s42_r2/checkpoint_latest.pt`;
  checkpoint SHA256 `ecf5a25b99bd810ce624cad9eac95682af2db5e7de6d8623b1c6a0d67183ce72`;
  format `w8a8_export`, global threshold `0.9997296929359436`.
  Export-manifest SHA256 `58137128fe2652a037e4d6f20bd6ff59579268c87331bd83e4b4b98cd43d7adf`.
- **Matched BERT, seed 42:** `results/cnn_adoption_v1/runs/bert_matched_final_s42_r2/checkpoint_latest.pt`;
  checkpoint SHA256 `3b395574f25722ecb9b96954fec21bbd8e5b56cb1b9d7735e4e2170c34de64ce`;
  format `fp32`, global threshold `0.9996341466903687`.

| Frozen target | Primary threshold |
|---|---:|
| `local_context_03_final_s42_r2:fp32` | `0.9998902082443237` |
| `local_context_03_final_s42_r2:w8a8_export` | `0.9998701810836792` |
| `local_context_03_final_s1337_r2:fp32` | `0.9997723698616028` |
| `local_context_03_final_s1337_r2:w8a8_export` | `0.9997743964195251` |
| `local_context_03_final_s2026_r2:fp32` | `0.999937891960144` |
| `local_context_03_final_s2026_r2:w8a8_export` | `0.9999349117279053` |
| `local_context_03_residual_replace_c98_final_s42_r2:fp32` | `0.9997135996818542` |
| `local_context_03_residual_replace_c98_final_s42_r2:w8a8_export` | `0.9997296929359436` |
| `local_context_03_residual_replace_c98_final_s1337_r2:fp32` | `0.999708354473114` |
| `local_context_03_residual_replace_c98_final_s1337_r2:w8a8_export` | `0.9997016787528992` |
| `local_context_03_residual_replace_c98_final_s2026_r2:fp32` | `0.9999077320098877` |
| `local_context_03_residual_replace_c98_final_s2026_r2:w8a8_export` | `0.9999077320098877` |
| `bert_matched_final_s42_r2:fp32` | `0.9996341466903687` |
| `bert_matched_final_s1337_r2:fp32` | `0.9998801946640015` |
| `bert_matched_final_s2026_r2:fp32` | `0.9998262524604797` |
| `historical_bert_w8_final_r2:historical_w8_qat` | `0.9984118938446045` |

All nine final checkpoint digests and all six manifest/blob identities are
also bound in the prospective freeze and independent audit, without relying
on mutable aliases. The teacher checkpoint SHA256 is
`95025bfc0e29163efea613d556d7da0b516ec1bb6c6fac9d9c4c9c6f1f373525`;
FIT-only teacher-cache SHA256 is
`3e97743cb216568d31d7f463b4428d25dca72e89afc92cceb2ed64578896c182`.
The raw-pretrained PCA initializer SHA256 is
`e76ae0743036d41cfebf706f060264742d6628ee92e7706692248ca3248f5100`.

## Secondary operating points and own-format effects

Nominal 1% and 2% are secondary calibration settings, not alternative passes.
FP32 CNN rows are diagnostic companions; they cannot replace intended W8/A8.

| Target | Alpha .005 recall/FPR | Alpha .01 recall/FPR | Alpha .02 recall/FPR |
|---|---:|---:|---:|
| `local_context_03_final_s42_r2:fp32` | 38.20% / 1.85% | 45.70% / 2.45% | 55.00% / 3.05% |
| `local_context_03_final_s42_r2:w8a8_export` | 41.30% / 2.05% | 46.00% / 2.40% | 56.00% / 3.15% |
| `local_context_03_final_s1337_r2:fp32` | 49.90% / 2.50% | 55.60% / 3.05% | 67.90% / 5.05% |
| `local_context_03_final_s1337_r2:w8a8_export` | 50.30% / 2.45% | 56.50% / 3.10% | 68.40% / 4.90% |
| `local_context_03_final_s2026_r2:fp32` | 40.60% / 2.20% | 49.50% / 2.70% | 56.00% / 3.20% |
| `local_context_03_final_s2026_r2:w8a8_export` | 41.70% / 2.20% | 49.50% / 2.70% | 56.00% / 3.25% |
| `local_context_03_residual_replace_c98_final_s42_r2:fp32` | 34.90% / 1.60% | 40.50% / 1.90% | 49.40% / 2.40% |
| `local_context_03_residual_replace_c98_final_s42_r2:w8a8_export` | 34.30% / 1.45% | 40.30% / 1.90% | 49.70% / 2.55% |
| `local_context_03_residual_replace_c98_final_s1337_r2:fp32` | 31.80% / 3.00% | 38.10% / 3.35% | 52.40% / 5.90% |
| `local_context_03_residual_replace_c98_final_s1337_r2:w8a8_export` | 32.30% / 2.90% | 38.90% / 3.55% | 52.90% / 5.85% |
| `local_context_03_residual_replace_c98_final_s2026_r2:fp32` | 21.80% / 1.55% | 30.10% / 1.90% | 39.70% / 2.30% |
| `local_context_03_residual_replace_c98_final_s2026_r2:w8a8_export` | 21.80% / 1.50% | 30.70% / 1.95% | 40.00% / 2.30% |
| `bert_matched_final_s42_r2:fp32` | 78.20% / 1.75% | 85.70% / 2.50% | 93.40% / 4.15% |
| `bert_matched_final_s1337_r2:fp32` | 45.50% / 1.15% | 69.00% / 2.40% | 82.60% / 3.90% |
| `bert_matched_final_s2026_r2:fp32` | 74.50% / 1.10% | 83.30% / 1.80% | 90.20% / 2.95% |
| `historical_bert_w8_final_r2:historical_w8_qat` | 6.90% / 1.10% | 9.70% / 1.40% | 12.90% / 1.90% |

At the primary own-format thresholds, float-to-integer LocalContext changes
TP by +31/+4/+11 and FP by +4/-1/0 across seeds; C98 changes TP by -6/+5/0
and FP by -3/-2/-1. Changed document decisions number 35/9/15 and 13/15/7.
These differences include each format's separately fitted benign threshold.
All FP32 companions also fail, so conversion is not the dominant security
deficit. Representation, attribution/context and fixed max-window aggregation
were not causally separated by this bounded study.

## Failure strata and withheld mechanisms

| Candidate/seed | Role-delimiter recall (sparse) | Encoding recall (sparse) | Code recall (adequate) |
|---|---:|---:|---:|
| LocalContext PCA + KD / 42 | 40.00% | 17.60% | 66.40% |
| LocalContext PCA + KD / 1337 | 68.00% | 34.40% | 59.20% |
| LocalContext PCA + KD / 2026 | 55.20% | 24.80% | 32.80% |
| C98 KD / 42 | 33.60% | 26.40% | 32.00% |
| C98 KD / 1337 | 35.20% | 16.80% | 31.20% |
| C98 KD / 2026 | 25.60% | 16.80% | 16.80% |
| Matched BERT / 42 | 78.40% | 67.20% | 77.60% |
| Matched BERT / 1337 | 48.80% | 41.60% | 49.60% |
| Matched BERT / 2026 | 70.40% | 63.20% | 87.20% |

Each required mechanism has 125 attack documents and backgrounds, but role
and encoding have only four and thirteen attacker ancestries. Their values
remain descriptive and uncertain. Code has twenty-one; all sixteen targets
fail its 95% recall gate, including the best BERT result 109/125 (87.2%).

Seed-42 intended CNN false positives are all controlled retrospective
constructions: LocalContext 41/400 and C98 29/400; neither rejects the 1,600
raw-natural benign documents in that seed. This limits any claim about
natural-mail rejection while documenting failure on the predeclared meaningful
quotation/discussion stress. Across plain/code/quotation attack carriers,
LocalContext detects 123/174/116 of 334/333/333; C98 detects 68/217/58.
Long benign documents with at least four windows have 19/120 Local false
positives and 9/120 C98 false positives. Length association does not establish
a causal aggregation mechanism. All seeds, source/genre, presentation, position,
context, benign category, length/window and payload summaries are retained in
`confirmation_primary_quality_summary_r2.json`, with input identities verified.

## Paired uncertainty

Effects are CNN minus the same-seed, development-selected matched BERT.
Intervals are the 2.5th/97.5th percentiles of 1,000 fixed-seed sensitivity
draws. Source resampling preserves paired backgrounds; payload resampling
uses attack ancestry only (no FPR estimate); calibration resampling refits
each global threshold on the same resampled benign source groups.

| Intended CNN | Seed | Recall difference, pp | Source interval | Payload interval | Calibration-refit interval |
|---|---:|---:|---:|---:|---:|
| LocalContext PCA + KD | 42 | -36.90 | [-40.61, -33.30] | [-41.98, -28.01] | [-47.50, -27.50] |
| LocalContext PCA + KD | 1337 | +4.80 | [1.10, 8.69] | [-5.94, 13.83] | [-11.20, 15.80] |
| LocalContext PCA + KD | 2026 | -32.80 | [-36.16, -29.27] | [-45.18, -24.75] | [-39.20, -20.70] |
| C98 KD | 42 | -43.90 | [-47.54, -40.37] | [-50.99, -39.58] | [-50.00, -32.60] |
| C98 KD | 1337 | -13.20 | [-17.06, -9.34] | [-19.55, -0.83] | [-27.90, -1.10] |
| C98 KD | 2026 | -52.70 | [-55.93, -49.33] | [-60.06, -49.50] | [-58.40, -39.10] |

Local seed 1337 has a positive point difference, but its excessive FPR
fails the absolute requirement and its payload/calibration intervals include
zero. Calibration-tail sensitivity remains material. Full FPR intervals,
all secondary settings, FP32 companions and historical-anchor comparisons
are retained in the six complete comparison artifacts and original draws.
The historical anchor is one fixed checkpoint reused across comparisons,
not three reference fits. These are separate dependence sensitivities, not
a joint source/payload/context bootstrap, multiple-candidate correction,
population guarantee or proof of statistical noninferiority. Fixed seeds and
one-author negative context families remain additional limits.

The analysis-only adaptor corrected a four-core-input versus six-frozen-file
metadata comparison. It verifies all frozen bytes before rows and preserves
the original comparator, exact patch and corruption tests. Independent
verification also corrected final-checkpoint filename and historical archive
path/runtime-subset schema faults. Failed checks and original sources remain
preserved. No frozen scoring source, score, model, format or policy changed.

## Question and evidence boundary

This study selects a possible backbone for SENTRY's next bounded FPGA
development stage. It asks whether focused CNN adaptation can transfer to
source/payload-disjoint indirect-injection examples and justify development
against a competent pretrained BERT reference. It supports the eventual
FPGA-aware contribution; neither detector selection nor conventional transfer,
distillation or quantization establishes that contribution.

The task is English external email/document content for document QA. Model
inputs contain document text alone. Labels describe constructed assistant
redirection intent, not successful compromise or preservation of a legitimate
task. No target-LLM API, paid service, physical hardware, payload execution or
protected project final-calibration/final-test/NotInject access was used.

## Data, independence and exposure

The natural backgrounds are a pinned SpamAssassin ham redistribution and the
official WikiText-103 TRAIN release. The inventory retains 3,667 usable English mail units in
1,568 thread/ancestry groups and 28,938 complete article paragraphs in 28,598
groups. Mail uses the documented CC0/PDDL distribution; Wikipedia text retains
CC BY-SA provenance. Article units are whole naturally bounded paragraphs,
selected by a fixed hash; they are not claims to full Wikipedia-article coverage.
The SpamAssassin package is stdlib-js revision
`02b1f0de3ee4ec2320fc0b23035ff994996e81c3`; WikiText is the official Salesforce
revision `b08601e04326c79dfdd32d625aee71d232d685c3`. License assertions are
release documentation, not an independent legal adjudication. Exact download
URLs, byte counts and hashes are retained in `source_cache/acquisition.json`.
SpamAssassin ancestry joins message/reply/subject identity, substantial shared
paragraphs and candidate token-shingle near duplicates. Generic footers were
excluded from the ancestry rule. This is a documented operational grouping,
not exhaustive semantic paraphrase detection.

Operative attacker text comes from the official Tensor Trust original release:
36,108 eligible strings from 417 players form 192 author/near-template union
groups. The giant 33,977-string component was quarantined rather than counted
as independent templates. Source rights and research-use consent are recorded;
no unrestricted SPDX license is invented. Only attacker input and grouping
metadata were used, without target-response/success supervision. The permitted
old canonical TRAIN contributes 32,370 documents in 10,790 source groups and
55 known payload groups. PIGuard/BIPIA leads were checked for provenance and
protected-family overlap; uncertain imports were quarantined rather than
opening their validation/test partitions.

The official Tensor Trust input revision is
`747a75e096761ebc01bd3970158827326b4add23`; raw download SHA256
`c3af2cb1b812fed72d879e130f2478d22915a2f4834494972418c57a60a2d4f6`.
The saved consent terms and acquisition manifest bind the research-use basis.

| Role | Benign | Attack | Source groups | Windows |
|---|---:|---:|---:|---:|
| FIT, including permitted canonical TRAIN | 12,790 | 23,580 | 12,790 | 47,283 |
| Development selection | 800 | 400 | 800 | 1,884 |
| Development benign calibration | 1,000 | 0 | 1,000 | 1,451 |
| Final benign calibration | 3,000 | 0 | 3,000 | 4,057 |
| New confirmation | 2,000 | 1,000 | 2,000 | 5,551 |

The confirmation panel pairs attacks with new backgrounds and contains 52
attacker ancestry groups. Its primary natural-email subset has 400 benign and
200 attack documents; the article generalization stratum has 1,600/800. The
final calibration mix has 350 mail and 2,650 article documents. Weights were
fixed before scoring; one global threshold is used despite differing genre
weights. Hard benign contains 457 confirmation documents: 400 controlled
retrospective examples and 57 natural human-recipient requests. Those labels
do not make every constructed variant an independent discourse population.

An initial construction fault reused class-specific positive introductions
and negative framing families across roles. The affected TextCNN fit stopped
after eight saved epochs; its favorable scores and all original r1 artifacts
are retained but excluded from adoption evidence. The focused r2 correction
removed every positive introduction and assigned each negative context family
to one role. Source/attacker memberships, labels, positions, targets and quotas
were preserved. R2 membership and construction froze at
2026-10-14T19:20:11.012077 UTC before any r2 fitting, not before the invalid r1
fit. The protocol explicitly records that distinction.

Independent checks verified 44,570 documents, 60,226 windows and every tail,
with zero cross-role source/payload/context ancestry intersections or exact
normalized document matches. The 30 negative context families and their 60
realizations are authored by one person; related themes and a maximum
cross-role token-trigram Jaccard of 0.0625 do not prove semantic novelty.
Foundation pretraining exposure remains unknown. Old canonical training and
teacher exposure are explicit; lexical exclusions do not establish mechanisms
unseen to all checkpoints. The old 96 probes are now known diagnostics.

Historical BERT canonical metrics were reused from their saved summary, without
reopening protected inputs: 2,738 attacks and 1,369 benign documents, all attacks
detected at alpha 0.005/0.01/0.02, with 4/13/43 false positives respectively.
The saved summary SHA256 is
`b05b6664129286bac67639fad24f5524b68ce54548cfcb9f77d5e41fadedd9e8`.
On the new development population, its unchanged original threshold detects
95/400 attacks and rejects 56/800 benign documents (23.75% recall, 7% FPR).
New alpha-0.005 calibration detects only 16/400 with 3/800 false positives.
Canonical performance therefore does not transfer to this new population;
recalibration alone does not establish generalization. At the original threshold,
the known 96 probes give 50/64 attack detections and 16/32 benign rejections,
which remain descriptive regression evidence.

Three lexical-mechanism strata were withheld from project fitting/selection.
Each has 125 attack documents on 125 backgrounds. Code simulation has 21
payload ancestries and meets the prospective adequacy rule of at least
100 attacks, 100 attack-source groups and 20 payload groups. Encoding has
13 ancestries and role-delimiter spoofing four, so their readiness remains
uncertain even when a descriptive recall is reported. These are mechanical
strata, not a validated new attack taxonomy or semantic-unseen guarantee.

## Shared computation and bounded adaptation

The pinned BERT WordPiece representation uses length 256, 254 content tokens,
64-token overlap and stride 190. Every tail is included and document risk is
the maximum attack-class probability over all windows. No model receives
trusted span locations, source identifiers, target responses or quotation
exemptions. A document-level max-margin binary cross-entropy objective avoids
fabricating labels for unknown individual windows. Focused same-dropout replay
and gradient checks verify the bounded-memory implementation.

The original TextCNN, Stacked CNN, LocalContext and C98 replacement were first
evaluated on repaired development data, alongside historical BERT W8 and FP32.
All originals failed the new primary recall requirement. The common enriched
study then trained all four CNNs and a pretrained four-layer, 256-hidden,
four-head BERT on the same FIT information. CNNs received twelve development
epochs with their recorded architecture learning rates; BERT received five,
AdamW, 10% warmup and linear decay. One bounded BERT refinement raised its
learning rate from 2e-5 to 5e-5. Different pretraining access is acknowledged.

Only two additional CNN training interventions were used, with no new
architecture sweep. First, the two strongest families received ground-truth
plus training-only BERT distillation: weight 0.3, temperature 2, document-level
Bernoulli teacher margins and a same-data supervised control. The refined
seed-42 BERT teacher had useful development behavior (79% recall, zero FPR)
but did not satisfy the security gate. Its cache contains FIT outputs only.
No teacher remains at inference or supplies confirmation supervision.

Second, LocalContext received raw language-pretrained WordPiece embeddings
projected by a FIT-independent, deterministic FP64 PCA to 96 dimensions,
with PAD fixed, RMS matched and every non-embedding initial tensor/RNG state
preserved. This is standard transfer initialization, not a novel method.
It retained the same distillation objective and architecture. The intervention
improved selected development recall but was not a uniform mechanism benefit;
its practical comparison also includes the prespecified epoch selector.

Development-only selection fixed LocalContext PCA + KD at seven epochs,
C98 KD at four, and matched BERT at four epochs of its five-epoch LR schedule.
Fresh final fits used seeds 42, 1337 and 2026, with no development monitoring,
early stopping or seed selection. Seed 42 was fixed for the deployment
checkpoint, and candidate priority was LocalContext then C98. Exact seed-42
state tensors reproduce their selected parents; file digests differ because
the completed final artifacts have distinct metadata and optimizer states.

## Development behavior and score separation

Common enriched seed-42 supervision produced recall/FPR of 27%/1.125% for
TextCNN, 36%/1.125% for Stacked, 47%/2% for LocalContext and 41.25%/0.375%
for C98. Matched BERT improved from 68.75%/0.25% at LR 2e-5 to 79%/0% at
5e-5. Distillation gave LocalContext 48%/0.75% and C98 42.75%/0.875%; the
PCA-initialized LocalContext selected parent reached 51.75%/0.875% in FP32
and 52.75%/1% after conversion. These are practical selected-recipe effects,
not a claim that every error category improved or that data enrichment alone
caused the difference from the original span-trained models.

| Final intended format | Seed 42 recall/FPR | Seed 1337 recall/FPR | Seed 2026 recall/FPR |
|---|---:|---:|---:|
| LocalContext PCA + KD, W8/A8 | 52.75% / 1% | 63.75% / 2.25% | 44.5% / 1% |
| C98 KD, W8/A8 | 43% / 0.875% | 40.5% / 2% | 28.25% / 0.875% |
| Matched BERT FP32 | 79% / 0% | 47% / 0.375% | 75.75% / 0.125% |

All three seeds of each model fail development recall. LocalContext seed 1337
has higher recall than its BERT counterpart but excessive FPR; BERT is not
uniformly stronger across seeds. Four paired three-seed comparisons preserve
source-group, attack-payload-group and benign-calibration-refit sensitivity
draws. Their thresholds remain the actual benign-calibrated policy.

A separate retrospective development-label diagnostic enumerates every strict
score tier, preserving ties. At most 8/800 false positives gives maximum
intended LocalContext recall 53.5%/41.75%/44.5% across seeds; C98 gives
47.75%/27.75%/29.25%; matched BERT gives 94%/69.25%/94.5%. At least 396/400
attack detections instead requires minimum BERT FPR 19.625%/14.375%/7.625%.
No target has a single strict threshold attaining both 99% recall and 1% FPR
on this observed development panel, even with its labels available. Independent
whole-tier enumeration and mixed-class/saturated-score tie tests confirmed the
rank calculation. These oracle cuts are neither calibration choices nor
confirmation/population bounds. They show a score-separation limitation in
addition to calibration transfer; global threshold movement alone cannot solve
the observed development failure.

## Frozen numerical and operating policies

Primary calibration is nominal alpha 0.005 on 3,000 separate benign documents,
with strict `risk > threshold`; required observed recall is at least 99% and
FPR at most 1%. Natural-email recall/FPR has the same bound. Hard-benign FPR
must be at most 5%; adequately populated required mechanisms must reach 95%
recall. Alpha 0.01 and 0.02 are secondary descriptions, never rescue chances.
All three fixed seeds must qualify; pooled or best-seed results cannot pass
for a failed seed. Relative to the development-selected matched BERT, observed
recall loss may not exceed one percentage point while meeting absolute bounds.

The prospective freeze binds all sixteen exact targets, recipes, formats,
source identities, data membership, policies, priority and seed rule before
confirmation. Its SHA256 is
`6252a2f357edc9fe62690da3dd375ea1c795fd9d28c5c4accfd7c8e1e07a5aaa`.
Both CNNs are assessed in intended W8/A8 plus explicit FP32 exceptions; FP32
companions are diagnostic. BERT is FP32. The historical W8 anchor remains
separate, with its original threshold 0.004557917360216379 and prospective
recalibration reported independently. No post-confirmation tuning is allowed.

The CNN export uses actual signed W8 codes, A8 inputs, INT32 accumulation,
round-to-nearest-even/saturation, and FP32 scales, biases, LayerNorm,
nonlinearities, pooling and softmax. Representative FIT calibration selects
512 windows from 500 source groups across twelve source/genre/length strata.
Actual packed CPU references and native signed CUDA integer products are
checked per checkpoint. Odd-shape/edge tests and module comparisons total
35 exact accumulator checks per final export; full intended-format development
scores and decisions are separately verified. No fake-quantized runtime is
reported as native integer performance.

The initial length sampler queried an absent metadata key; corrected exports
use the actual `content_tokens` field. Old exports/outcomes remain preserved.
A stale family mapping and the aligned/un-aligned BERT byte subtotal were
also corrected with their historical evidence retained. Post-training integer
conversion changed development recall modestly; floating models already
failed severely. Additional QAT was therefore not a quality-critical rescue.

## Software timing and hardware interpretation

All matched timing uses exact final seed-42 checkpoints and 29 shared raw
development documents, stratified by genre, label and window counts 1/2-3/4-7/8+,
plus the two longest documents per label. Each route has three warmups and
twenty retained repetitions per document. Complete latency includes raw-text
tokenization, padding/copies, every serial batch-one window, network, device
return, max aggregation and fixed-threshold decision. Loading/capture/qualification
are separate. The stratified sample is not a workload-frequency estimate.

Fourteen warm routes passed all 3,335 calibration/development qualification
windows and 2,200 document decisions, with zero decision changes. Five cold
routes retain ten fresh-process shortest/longest observations. OS/file caches
were not flushed. BERT CPU1 failed the unchanged 1e-5 risk tolerance on two
windows, maximum delta 2.31266e-5, despite zero decision changes. Its CPU timing
is unqualified; CPU4/cold follow-ups were omitted and no fallback was hidden.
Memory-floor interruptions are retained separately from numerical failures.

| Complete-document mean, ms | LocalContext | C98 | Matched BERT |
|---|---:|---:|---:|
| FP32 CPU, one thread | 23.414 | 25.570 | Unqualified |
| FP32 CPU, four threads | 21.257 | 22.668 | Not executed |
| FP32 CUDA eager | 34.993 | 36.203 | 25.755 |
| FP32 CUDA Graph | 4.415 | 4.557 | 5.106 |
| Intended W8/A8 CUDA eager | 55.656 | 56.412 | Not this format |
| Intended W8/A8 CUDA Graph | 5.211 | 5.602 | Not this format |

The intended integer CNN routes show no measured GPU latency advantage over
qualified BERT FP32 Graph. Their float companions are faster on this sample
but cannot replace the frozen intended arithmetic. Measurements are sequential
on the shared Windows/RTX 4070 host; concurrent system load is not controlled,
and this is not a physical FPGA or population-latency claim. Exact per-document,
length-bin, long-stress, stage and startup observations are retained in
`final_profiling_audit_r2.json` and its bound profiles.

The intended FPGA target is VEK280, `xcve2802-vsvh1760-2MP-e-S`, with a 200 MHz
objective, 64 MiB DDR arena and explicit host/PS tokenization/windowing/document
retention. The PL route proposes three AXI4 memory bundles and AXI-Lite control.
LocalContext and C98 both need receptive field 33/lookahead 16, per-token
channel LayerNorm and branch work, whole-window pooling and document-wide
decision retention. The ideal
3,072-byte A8 line-history subtotal is neither simultaneous peak nor allocated
RAM. Whole-document work scales with
`N(n)=max(1,ceil(max(0,n-254)/190)+1)`; no tail or host transport is free.

Actual final payloads are 3,440,780 bytes for LocalContext and 3,439,184 for C98,
identical within each family across seeds. The hypothetical, equally aligned
all-matrix-W8 BERT shape subtotal is 11,377,100 bytes, giving conditional
69.75697%/69.770996% storage reductions. That BERT precision scope has no
qualified packed artifact. The comparison passes the arithmetic example of
a 50% reduction conditionally; it does not establish the complete prospective
target-feasible memory/dataflow/retention/transport benefit.

No CNN allocated-buffer map, CNN command/packet integration, HLS/Vivado closure,
achieved clock, bandwidth, power, energy or physical protection measurement
exists. Existing BERT FPGA resources/timings cannot be inherited. Analytical
MAC counts exclude some operators and cannot establish speedup. Zero optional
vendor builds were used: the observed quality question was the first
discriminator, and builds are not required to explain a measured quality failure.

## Reproduction and retained limitations

The principal configuration is `configs/cnn_adoption_v1.json`; its freeze and
checkpoint/export identities bind the study. Python 3.11.16, Torch 2.9.1+cu128,
Transformers 4.57.3 and the pinned Google small-BERT revision are recorded.
Results, source downloads, checkpoints and build artifacts are Git-ignored;
a fresh clone does not contain the measurements. Reuse these bound artifacts
rather than rerunning confirmation or selecting a new seed after seeing results.

The exact process commands and logs are in `final_training_dispatch/`,
`qualification_dispatch/` and `confirmation_dispatch/`. Once-only scoring uses
`scripts/cnn_adoption_evaluate.py --stage confirmation` with the bound plan.
Independent verification uses `scripts/check_cnn_adoption_confirmation.py`,
and profiling uses `scripts/cnn_adoption_profile.py`. Resource accounting
separates completed process walls, nested timers, waiting/bookkeeping,
interrupted upper charges and unknown intervals; no missing total is zero.

This bounded panel does not establish impossibility for all CNNs or justify
abandoning FPGA. It does not demonstrate target compromise, downstream task
utility, universal injection protection, energy/performance on a physical
device or publication-level novelty. Sparse held-mechanism ancestry, controlled
one-author benign contexts, operational grouping and unknown foundation
exposure limit the breadth of any empirical adoption decision.

## Historical reference on confirmation

The original threshold `0.004557917360216379` gives 288/1000 attack
detections and 122/2000 benign rejections (28.80% recall,
6.10% FPR). Prospective alpha-0.005 recalibration gives 69/1000
and 22/2000 (6.9% recall, 1.1% FPR). Both are distinct from the saved canonical
and known-probe regression metrics; neither is silently substituted for the other.

## Resource closure and next action

The final ledger records 19 charged fitting jobs (18 completed,
one interrupted invalid-r1 fit). Fifteen transient gradient qualification
updates are separately disclosed; counting every one conservatively as a
learning job yields 34, still below the 60-job ceiling.
Recorded accelerator process/internal lower bound is 3.462178 hours;
the interrupted-r1 upper charge is 323.747091 seconds.
The compatibility terminal charge is 3.552107 hours,
while recorded CPU lower bound is 0.669796 hours.
These are disjoint recorded process intervals, not exact device-busy time or
a strict total-runtime upper bound; unknown intervals remain explicit.
Nested timers are not added twice. Export waiting/bookkeeping is distinct
from its active subprocess walls. Failed preflights/checks and successful
retries are charged separately. Prior ledger bytes and hashes are retained.
Account usage was observed at 53% against the user-raised 60% cap,
initially 36%, with no reset or purchase. Zero vendor attempts, paid/API calls
or physical-device operations were performed. Recorded charges are below the
48 accelerator-hour / 12 heavy-CPU-hour / four optional vendor-attempt ceilings.
Eight explicit unknown intervals prevent an exact study-wide total or a strict
resource-accounting upper-bound claim; they are not zero or silently passed.

The smallest proposed next discriminator is a prospectively labeled, grouped
attribution-distance test using the frozen models: keep background and quoted
payload fixed while moving the retrospective attribution within/across the
256-token window boundary. Treat related variants as one source/payload group.
A boundary-specific loss would implicate missing document-context coupling;
failure even in the short, same-window control would instead motivate a
representation/intent-boundary investigation. This is a proposed separate
work package, not post-confirmation probe wording, rescue training or an
authorized hardware launch. It precedes a larger build or another broad search.

## Reproduction commands and history

Use the retained commands and immutable artifacts; do not rerun confirmation
as a new selection opportunity. Representative completed entry points:

```powershell
& 'C:/research/GATE/.venv/Scripts/python.exe' -B scripts/cnn_adoption_train.py --plan configs/cnn_adoption_v1.json --run-id local_context_03_final_s42_r2
& 'C:/research/GATE/.venv/Scripts/python.exe' -B scripts/cnn_adoption_evaluate.py --plan configs/cnn_adoption_v1.json --run-id local_context_03_final_s42_r2 --stage confirmation --format w8a8_export --device cuda --batch-windows 1
& 'C:/research/GATE/.venv/Scripts/python.exe' -B scripts/check_cnn_adoption_confirmation.py --plan configs/cnn_adoption_v1.json --independent-check results/cnn_adoption_v1/data_check_r2.json --confirmation-scores-released --output results/cnn_adoption_v1/independent_confirmation_recheck.json
& 'C:/research/GATE/.venv/Scripts/python.exe' -B scripts/update_handoff.py --check
```

Existing completed training/scoring entry points reuse only matching saved
associations; the independent recheck writes a fresh analysis artifact.
Profiling command variants and exact matched policies are recorded in
`qualification_dispatch/`; comparisons and helper versions are recorded in
`comparison_dispatch/` and `confirmation_comparison_adapter/`.
The preceding principal-report chronology is preserved verbatim in
`reports/cnn_adoption_execution_history_20261015.md`. Its old pending plans
are historical and non-authorizing. The execution log preserves failed
attempts and corrections. Older broad-exploration and method-study reports
remain unchanged. Git commits 38c294b and 6e566a0 exist locally; the requested
push to origin/main was rejected by automatic approval review and remains
pending explicit destination/payload approval. This final study handoff
does not imply GitHub synchronization.
