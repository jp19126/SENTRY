# CNN adoption study

Started 14 October 2026 under the explicitly submitted
`SENTRY_CNN_adoption_prompt.md`. This is local, board-free backbone selection
supporting SENTRY's FPGA-centered purpose, not a production guard or completed
novel contribution. Account usage ceiling: **60% used** (raised explicitly by the user), initially observed 36%.
The previous exploration and its failed primary outcomes remain unchanged.

## Current decision state

**Executing; no adoption verdict.** New confirmation is unopened. Account usage
last observed43% of the user-raised60% ceiling. The primary operating point is
alpha .005, strict max-window-risk > fitted benign threshold. Results below are
seed42 development observations on400 attacks/800 benign documents, not a
confirmation or population guarantee.

| Completed recipe/format | Recall | FPR | Primary gate |
|---|---:|---:|---|
| TextCNN enriched FP32 |27%|1.125%|Fail|
| Stacked enriched FP32 |36%|1.125%|Fail|
| LocalContext enriched FP32 |47%|2%|Fail|
| C98 replacement enriched FP32 |41.25%|.375%|Fail|
| Matched BERT 2e-5 FP32 |68.75%|.25%|Fail|
| Matched BERT 5e-5 FP32 |79%|0%|Fail|
| LocalContext distilled FP32 |48%|.75%|Fail|
| C98 replacement distilled FP32 |42.75%|.875%|Fail|
| LocalContext distilled W8/A8 |47.25%|.625%|Fail|
| LocalContext PCA+KD FP32 |51.75%|.875%|Fail|
| LocalContext PCA+KD W8/A8 |52.75%|1%|Fail|
| C98 replacement distilled W8/A8 |43%|.875%|Fail|

Refined BERT is the qualified useful teacher, not a security-qualified guard.
Its FIT-only cache completed; LocalContext KD completed12epochs/selected2.
All bounded interventions completed. Nine fixed final three-seed fits are running.
Ground-truth labels remain in both KD objectives; runtime
CNN inference has no teacher. Independent data/score/cache checks and concrete
corrected FIT-calibrated exports exist. The old length-sampler bug is disclosed
below; corrected exports require fresh numerical/quality/timing qualification.
Three-seed final training, numerical/profile qualification, prospective
candidate/rule freeze and once-only new confirmation remain incomplete.

## Prospective contract before new fitting

Question: can focused CNNs learn transferable indirect-injection discrimination
from richer, comparable supervision, and merit a concrete FPGA development
route against historical and matched-data pretrained BERT?

The primary workload is English external email/document content for QA. Only
document text enters inference. Newly licensed natural mail/document sources
and controlled matched examples will form source/payload-ancestry-disjoint
fit, development-calibration, development-selection, final benign-calibration
and new confirmation roles. Membership, labels, genre weights and construction
rules will be fixed before fitting. Confirmation scores/errors stay unopened
until candidate, numerical and decision freeze. Old authored probes are known
diagnostics; canonical results are regression/exposure evidence. Project final
calibration/test and NotInject, including redistributed copies, stay closed.

Data preparation is executing separately. SpamAssassin mail has a documented
CC0/PDDL distribution; exact download identities, deduplicated/thread group
counts, English/ambiguous exclusions and an independent licensed article source
are being established. PIGuard validation/test mixtures include protected
benchmark families and are quarantined. Imported uncertain material is excluded.
The numerical configuration will bind actual prepared memberships/provenance
before any optimizer job. Planning targets are thousands of independent benign
calibration rows and at least 2,000 benign/1,000 attack-labelled confirmation
rows, without counting template multiplication as independent mechanisms.

Shared initial representation: pinned BERT WordPiece, L256, 254 content tokens,
overlap64/stride190, complete tails and document max-risk aggregation. Natural
article document units will be defined before fitting; all models process the
same complete units, not different truncated inputs. Actual whole mail lengths
and deliberately prespecified long-document stresses will be retained.

Four initial CNNs: TextCNN01, Stacked01, Local/context03 and residual
replacement98. Historical BERT W8-QAT retains its original checkpoint/threshold
and prospective recalibration separately; its canonical .5/1/2% saved scores
already exist. Matched BERT starts from the pinned small pretrained encoder,
with a fresh task head and the same fitting information. Original checkpoints
remain unchanged. Pretraining advantages and controllable project exposure
will be reported rather than relabelled as equal initialization.

The baseline supervised objective is shared document-level max-logit-margin
binary cross entropy. No unknown annotation becomes a fabricated window label;
short malicious evidence is not averaged away. Memory-bounded replay, if used,
must reproduce a direct same-dropout objective/gradient on focused real inputs.
The change from old span-window training is explicit, not attributed solely to
architecture. AdamW schedules may differ competently by model; comparable
information and recorded tuning opportunity matter more than identical epochs.

Initial bounds: CNNs at their existing architecture learning rates, at most12
development epochs; pretrained BERT at2e-5, at most5, with10% warmup/decay.
One BERT5e-5 refinement is conditional on inadequate learning. At most two
additional CNN structures and two named interventions are allowed, not required
spending targets. Qualified teacher distillation must use fit-only cached
outputs plus labels and retain a same-data no-distillation control; no runtime
teacher or confirmation-derived supervision is allowed.

Primary threshold policy is nominal benign calibration alpha=.005 and strict
score>threshold. Observed primary requirements stay recall>=99%, FPR<=1%; hard-
benign rejection<=5%; adequately populated prespecified withheld-mechanism
recall>=95%. Secondary alpha=.01/.02 apply to CNN and BERT alike. Adequacy and
application/stress strata will be fixed with actual independent source/payload
counts before evaluation; sparse evidence cannot silently pass a gate.

At most two CNN finalists plus matched BERT and the historical anchor will be
frozen using development only. Repeated training seeds42/1337/2026; choose a
deployment seed/checkpoint rule and candidate priority before confirmation.
All three seeds must meet adoption requirements; pooled/best-seed results cannot
rescue a failed seed. Compare against the stronger development-selected BERT,
allowing at most one absolute percentage point observed recall loss while
retaining absolute bounds. Paired source/payload dependence and calibration
uncertainty will remain distinct from empirical gates and fixed-seed variability.

Numerical route must export actual codes, scales, exceptions and accumulation
semantics, initially signedW8/inputA8 with INT32 reference accumulation and
explicit FP32 rescale/bias/norm/nonlinear exceptions where needed. Representative
fit-only activation calibration replaces the old ordered512-window prefix.
Prospective hardware benefit: at least50% precision-matched parameter/scale
payload reduction versus the selected BERT route, together with a feasible
target weight/activation/context/document-retention and transport schedule.
This is not a MAC-derived speed or physical resource claim. Any alternative
material-benefit contract must be fixed on development before confirmation.

Match raw-document batch-one timing, length-stratified samples and long stresses,
preparation/copies/all windows/max/decision, startup and warm repetitions.
Give both sides bounded competent backend optimization and validate decisions;
CUDA Graph/native INT routes cannot inherit old checkpoint qualification.

Ceilings:60 learning jobs,48 accelerator process-hours,12 heavy CPU hours,
four optional vendor attempts/eight hours. Reserve final three-seed BERT/CNN,
numerical qualification and confirmation before optional tuning. Root alone
dispatches serial accelerator/vendor work; workers own data, training interface
and export/mapping files. No paid services, target API, physical board, payload
execution. The user subsequently authorized Git synchronization. A new-stage negative/unresolved decision
requires actual completed competent available comparisons, not old failures.

## Actual execution

Read current repository/protocol/reference implementations and original prompt;
working tree initially contains only the user adoption prompt/download metadata.
Native Python3.11.16, Torch2.9.1+cu128 and RTX4070 are available. One-time resource
preflight passes:7.206GB Windows/24.127GB WSL available against4/8GiB floors.
Runtime floors remain2/4GiB. Preflight process is terminal; no learning dispatched.
Data preparation, shared objective/loader implementation and concrete export
reference are active independent work. Exact source choices/counts and allocation
will be entered in the single study configuration before fitting.

## Progress at user-requested synchronization

The new data roles are frozen and independently checked: 44,570 documents and
60,453 windows, with no source/payload ancestry crossing roles. The data protocol
identity is 8370d8455b9d43b160c9a699bec9f50222b05a39139f8fb441eab1dbdf2e43aa.
Confirmation contains 2,000 benign and 1,000 attack-labelled documents; encoding
and role-delimiter payload ancestry remain underpowered. Confirmation scores
remain unopened. Exact membership and checks are local ignored artifacts in
results/cnn_adoption_v1/data_protocol.json and data_check.json.

CUDA document-loss/gradient/RNG qualification passed for all five trainable
models. The packed integer backend passed 19 exact dot-product checks and
bounded CPU/GPU logit comparisons. These are implementation checks. The first
frozen original TextCNN development evaluation completed: primary alpha=.005,
threshold .9999271631240845, TP1/FN399/FP1/TN799. This fails the development
quality gate; it is not a final adoption result. No new candidate fit or
confirmation scoring has run. Remaining original baselines and common supervised
adaptation are pending. Git synchronization preserves this incomplete state.

## Frozen-original development comparison completed

All six saved references finished before new supervised fitting. The following
counts use the same 1,000 benign development-calibration and 1,200 selection
documents (400 attack/800 benign), strict score>threshold, alpha=.005.

| Reference | Threshold | TP/FN | FP/TN |
|---|---:|---:|---:|
| historical_bert_fp32 | 0.999423384666 | 5/395 | 1/799 |
| historical_bert_w8 | 0.999802529812 | 7/393 | 3/797 |
| original_local_context_03 | 0.995144784451 | 17/383 | 2/798 |
| original_local_context_03_residual_replace_c98 | 0.98744904995 | 10/390 | 0/800 |
| original_stacked_01 | 0.998128354549 | 1/399 | 3/797 |
| original_textcnn_01 | 0.999927163124 | 1/399 | 1/799 |

All fail the primary development gate. Historical W8 unchanged threshold
.004557917360216379 gives TP118/FN282/FP62/TN738, separately from recalibration.
Independent TextCNN saved-score audit passed all 2,296 rows, input/window and
checkpoint/source identities, exact max aggregation and threshold/count
recomputation (3.047 CPU seconds). Primary natural-email counts TP0/FN100/FP0/TN200.
The top20 calibration risks include16 retrospective instruction discussions;
4of5 primary calibration rejections are retrospective examples. All239
single-window attacks and all21 attacks with >=4windows are missed. This does
not implicate missing tails. These descriptive associations motivate enriched
supervision without changing labels, construction or thresholds.

The fixed five supervised recipes are dispatched serially by
results/cnn_adoption_v1/serial_development_queue.py, stopping on any error.
Per-job exact commands, terminal codes and walls are in queue_logs. TextCNN
fitting has started; final outcomes are pending. Actual account usage39% used
against the explicitly updated60% ceiling; no reset consumed.

## Verified construction fault: original new panel superseded for adoption

Independent inspection found six repeated POSITIVE_CONTEXT and six disjoint
BENIGN_CONTEXT framing sentences in scripts/cnn_adoption_data.py. These
label-exclusive framings cross fit/development roles and can identify class
without interpreting the human-authored payload. Transport balancing does not
remove this shortcut. All reported original-panel scores remain preserved but
cannot establish the requested beyond-template generalization. Supervised
TextCNN early perfect development scores are affected and do not qualify a
candidate. The serial queue was stopped by verified process identity; last
saved epoch and interruption record are in construction_fault_stop.json.
One interrupted learning job counts against the60-job allowance; measured
process time will be charged rather than treating the interruption as free.
No confirmation scores were opened. A focused construction repair will preserve
source/payload role memberships and scientific gates, use distinct revision
paths, and freeze before new fitting. The previously recorded data check verified
token/ancestry mechanics, not the absence of semantic construction shortcuts.

Focused repair authorized within the original fault-correction allowance:
remove positive intros; use common neutral transports and30 reviewed authored
benign discourse-ancestry families allocated6perrole, with grouped wording
realizations. Retain original source/TT memberships, labels, counts, quotas and
gates. Old files remain untouched; revised raw/protocol and packed inputs will
use revision_r2/ and prepared/token_r2/. The label evidence remains controlled
context recognition on natural backgrounds, not natural historical incident
records or evidence of broad population generalization. Revised ancestry and
shortcut checks must pass before scoring/fitting. Resource accounting currently
charges one interrupted learning job and a conservative323.747s fit-process
upper bound; scoring/qualification intervals are explicitly distinguished from
whole-process walls in results/cnn_adoption_v1/resource_accounting.json.

CUDA Graph complete-raw-document profiling pipeline bring-up passed on the
preserved original TextCNN development inputs:15 stratified/stress documents,
45 repetitions; all-window/eager/saved-risk and fixed-decision checks passed.
This is implementation qualification on the superseded panel, not adoption
quality or a latency claim for a future checkpoint. CLI: python -B
scripts/cnn_adoption_profile.py --plan
results/cnn_adoption_v1/config_r1_before_repair.json --reference-id
original_textcnn_01 --stage development --format fp32 --device cuda --backend
cuda_graph --mode warm --sample-per-stratum 1 --long-stress-per-label 1
--warmups 1 --repeats 3 --max-seconds 600. SourceSHA256
9b6fc7f710246a6073c661052596a5657a2168bd1ab4c88d4d8a507b2fdaa3b9.
Four-original analytical target mappings now record actual shape/identity checks,
2.139/2.106/3.445/3.445MB proposed packed payloads and complete-window scaling.
The hypothetical precision-matched BERT payload11.377MB is unqualified
numerically; reductions69.7–81.5% do not establish measured resources or speed.

The same bounded CUDA Graph profiler check also passed historical FP32 BERT
(15documents/45timedobservations). This qualifies the explicit-mask graph
route only for that checkpoint/input panel, with all saved calibration and
development windows/decisions checked; a new matched-data checkpoint must be
qualified again. The repaired context bank was inspected and released before
new raw construction:30 controlled families/60realizations, zero exact cross-
role templates, maximum token3shingle Jaccard.0625. Semantic themes and one
construction author remain shared limitations, not independent natural sources.

Fresh-process cold profiling bring-up also completed for the shortest/longest
shared original TextCNN warm-sample documents (two children), with saved warm
risks/decisions matched. Model/tokenizer loading and graph capture remain
separate from the warm raw-document interval; operating-system caches were
not flushed. Source and matched-warm identities are recorded in
profiling/development/original_textcnn_01/fp32/cuda_graph/cuda/
cold_process_complete.json. No latency result is assigned to a revised fit.

## Repaired r2 execution released

R2 token preparation completed47.000s; core independent token/association/tail
check59.469s, with additional raw-block/context checks outside that core timer.
44,570documents/60,226windows verified. All3,400 newly constructed positive
blocks equal the original transformed TT text with common transport only;
all1,760 benign discussion blocks match the released role-exclusive bank;
removing each of5,160 constructed blocks recovers its exact original background.
Source/payload/context role intersections and exact normalized-document role
matches are zero. R2 source/TT memberships and labels equal the original panel.
The protocol freeze precedes every r2 fit; the earlier interrupted r1 fit and
its known development outcomes remain disclosed, not erased by this repair.

Protocol:c84c7c0ca42b3099a95e7f48471788e221873ef694bf4e7cf96fe6d3f7a62cb7.
Independentcheck:0844e039d7b3216a27936a07661cc5dd2c86abefbc07d46c06205488858fabe2.
Pretrainingcontract:e47f997b1bcc53f6fd079da74d29ef169ebd741bd20399e0b12a4082fa30b785.
All are under results/cnn_adoption_v1, with revised raw/protocol inrevision_r2
and prepared inputs inprepared/token_r2. Config now binds exact manifest/raw/
artifact hashes. Required-stratum display aliases were corrected to actual
role_delimiter_spoof/encoded_instruction/code_simulation metadata keys before
r2 scoring; the same prespecified minima apply. Payloadgroup counts4/13/21
respectively retain sparse evidence for the first two.

The guarded serial_development_queue_r2.py was dispatched after terminalcheck
exit0; session17071 starts original_textcnn_01_r2, then the other five saved
references, followed by allfive unchanged supervised recipes with_r2 IDs.
All prior r1 models/scores/data are retained; no r2 confirmation scored.

## Repaired frozen-original development results

All six original-checkpoint r2 evaluations completed exit0 before supervised
r2 fitting. Same1,000 benign development calibration and1,200 selection rows.

| Reference | Primary TP/FN | Primary FP/TN | Threshold |
|---|---:|---:|---:|
| historical_bert_fp32_r2 | 13/387 | 1/799 | 0.9994353652 |
| historical_bert_w8_r2 | 16/384 | 3/797 | 0.999746978283 |
| original_local_context_03_r2 | 0/400 | 0/800 | 0.999974727631 |
| original_local_context_03_residual_replace_c98_r2 | 1/399 | 0/800 | 0.999627590179 |
| original_stacked_01_r2 | 1/399 | 3/797 | 0.998424172401 |
| original_textcnn_01_r2 | 1/399 | 3/797 | 0.99991953373 |

Independent TextCNN/Stacked saved-score checks passed3.547/3.359 CPU seconds.
TextCNN natural-email TP0/FN100/FP1/TN199. Both saved own-threshold/max/input
associations match the revised data. Original TextCNN vs historical W8 primary
paired recall difference is-.0375 (CNN.0025 vs BERT.04), equal FPR.00375.
Separate95% sensitivity intervals: sourcegroups[-.0595,-.0179], attackpayload
groups[-.0765,0], paired calibration-source refit[-.06,-.0125]. This is one
fixed seed, not a sampled seed population or distribution-shift guarantee.
1000 raw resampling draws and group counts are retained in the comparison JSON.
The initial comparator integration rejected a nonexistent redundant top-level
stage before reading rows; it now uses the actual association.stage schema.
The saved output caption incorrectly said all benigns lack payload ancestry;
its numeric draws remain valid: payload bootstrap concerns attack recall only,
and full-panel FPR is undefined because natural benigns lack that ancestry.
Controlled retrospective benigns can carry TT ancestry. The caption correction
is preserved in cnn_adoption_compare_execution_attempts.json; source corrected
for future outputs without overwriting the original comparison.

Supervised r2 TextCNN remains live. Epoch7 primary TP108/400,FP9/800 at the
last inspection; this is interim development evidence, not finalist selection.
Other fixed supervised recipes continue serially after this run.

R2 supervised TextCNN completed12epochs, development-selectedepoch7.
CheckpointSHA256:b068fcc77e25946b904c1d4abf4b61e2ce538a8b4e18eba4324b78f86c1f7056.
Primarythreshold.9999972581863403;TP108/FN292/FP9/TN791, recall27%,FPR1.125%.
Training+scoring361.778s (whole subprocesswall in queue_logs_r2). This fails
the primary gate. Stacked fitting follows. Some late TextCNN FP32-softmax
calibration scores round to1 (epoch12threshold1/TP0/FP0); a separate
development-only raw-margin/F64 diagnostic is underway to distinguish rounding
from actual score overlap. It changes neither primary scores nor thresholds.

Independent selected-TextCNN context/window diagnosis completed34.219 CPU
seconds. All9 selection false-positive winning windows retain BOTH discourse
framing sides; no retrospective selection/calibration document has an isolated
payload window. Thus disappearance of both framing sides does not explain
current development false positives. All20 highest calibration scores are
retrospective examples;7of9 selection false positives share one payload group
across five context families. Matched backgrounds:376/400 attack risks exceed
the paired benign risk,22are lower,2tie. Improved relative ranking does not
satisfy the absolute low-FPR requirement. FIT separately contains19/400
retrospective documents with28 isolated payload windows, matching54positive
windows in24positive fitting documents. This is a local supervision tension,
not proof of an unavoidable whole-document contradiction. The diagnosis
artifact retains exact IDs/input hashes and did not access reserved scores.

## Probability saturation discriminated from low-FPR overlap

Bounded CPU-only TextCNN margin probe completed14.015s, one CPU thread,
CUDA not initialized, devcal/devselection only. Selectedepoch7 primary counts
are identical for nativeFP32probability, FP64sigmoid of FP32logits, and raw
FP32/FP64margin diagnostics:TP108/FN292/FP9/TN791. Its selected result is
therefore not rescued by preserving more score-order precision. Latestepoch12
FP32threshold1 producesTP0/FP0; diagnostic FP64probability or rawmargin
producesTP64/FN336/FP11/TN789 (recall16%,FPR1.375%). Rounding explains the
lateepoch collapse but not satisfactory discrimination. This does not change
primary outcomes, model selection, calibration or numerical policy. CPU/CUDA
comparison limitations and frozen checkpoint hashes are retained in
margin_probe/textcnn_01_supervised_s42_r2/complete.json.

StackedCNN completed12epochs, selectedepoch9:primaryTP144/FN256/FP9/TN791,
recall36%,FPR1.125%, threshold.9998946189880371, also fails.
CheckpointSHA882cde263e5b6b0c706e268fc73894e5564517eb96f0f04c493beab2edacd201.
Training+scoring391.034s; subprocesswall separately saved. LocalContext fitting
is live; no intervention is selected until the common panel and matchedBERT
finish.

Actual enriched Stacked epoch9 W8/A8 export completed CPU1thread7.593s
(invocationwall~17.1s), CUDA/vendor/physicalfalse.512representativeFITwindows
from512sourcegroups calibrate activations; codes/scales are hashed and loaded
from packed bytes, not recomputed weights. Payload:1,982,208W8-codebytes +
123,140FP32-scalebytes +776FP32parameter/biasbytes =2,106,124bytes. FourFIT
referencewindows show max/mean absolute logit error.07037/.04206 and4/4
argmaxagreement. This small reference check is NOT low-FPR quality or
threshold qualification; own-format development scoring follows whenGPUqueue
finishes. No finalist is nominated by this bring-up. Output:
exports/stacked_01_supervised_s42_r2/ under the adoption resultroot.

LocalContext03 supervised_r2 completed12epochs, selectedepoch4. Primary
threshold.9990829229354858;TP188/FN212/FP16/TN784, recall47%,FPR2%,
failingboth absolutebounds. CP SHA256
ce775d06de3b15b728b604c59e9ef5f660809463371a67c31b0019f8308172f0.
Training+scoring792.277s; wholeprocesswall separately inqueue_logs_r2.
Residualreplacement98 fitting is now live, followed bymatched-data BERT.
Train-only teacher-cache entrypoint is being implemented against the existing
FIT/cache schema; no teacher has been qualified or queried yet. Any such
intervention depends on the actual completed matchedBERT development result.

Residualreplacement98 supervised_r2 completed12epochs, selectedepoch2.
Primarythreshold.998331606388092;TP165/FN235/FP3/TN797, recall41.25%,
FPR.375%; meetsFPR butfailsrecall. CheckpointSHA256
8b5f53c9791b1312a0bdb529aecc9fd220967971bac494a179a4be7c6ffce185.
Training+scoring816.655s, subprocesswall separately inqueue_logs_r2.
Allfour common CNN fits are complete; matchedBERT nowruns.
LocalContext numericalexport produced3,440,780 packedbytes andfourFIT
referencevectors withmax/mean |logitdelta|.107903/.026414,4/4argmaxagreement,
CPU1thread11.969s. Its generatedmapping sidecar erroneously reused a
Stacked-specific schedule; topology/liveness claims are quarantined pending
focused correction, preserving oldbytes. Packedcodes/scales/config/opgraph
and theirhashes remain checkpoint-specific. No GPUqualityqualification yet.
The nativeinteger check now accepts explicitmanifest/FIT/output CLIargs with
unique-output/preservation guards; help passes, actualnewformatGPUchecks
wait for theserialtrainingqueue. Teacher FITcache helper andsyntheticchecks
are ready, but no teacher has beenqualified orqueried.

## Mapping estimate correction

The LocalContext sidecar schedule is now family-specific, preserving the
invalid oldsidecar andpre-fix exporter inits historydirectory. Packedmanifest,
codes/scales/reference/opgraph bytes are unchanged. The staticpayload estimator
also double-counted LayerNorm affine parameters: corrected LocalContext
3,440,780B (prior3,445,444B), C98replacement3,439,184B (prior3,444,560B).
TextCNN2,139,148B andStacked2,106,124B unchanged. These corrections supersede
previous preliminary storage figures; equalconfig/shapes fororiginal and
enrichedLocalContext nowmatchtheactual3,440,780B payload. The discrepancy
was accounting, not a silent architecture change. BERTshape-derived storage
comparison remains hypothetical andnumericallyunqualified; no physicalclaim.

Common matchedBERT completed5epochs, selected4:TP275/FN125/FP2/TN798,
recall68.75%,FPR.25%, threshold.9993425011634827. It improves the practical
comparison butfailsrecall99%. CP SHA
13c6814362b39eee0d4a58ae8e6aef51d4e4fb12ae728368cd452042d08f2d4c.
Training+scoring521.877s. Allfive common selectedCPs nowhavecompleted full
development+known96 evaluations. The prospectively allowed oneBERT5e-5
refinement was dispatched afternewformatchecks; no extraBERTsweep.

Nativeinteger qualification of actual enrichedStacked/LocalContext passed
19/35exactINT32-dotchecks againstCPUINT64oracles, respectively, plusfourFIT
packedCPU/nativeCUDA outputchecks underunchanged1e-5logit/1e-6risk tolerances.
Checkintervals4.627/6.019s; fullprocesswalls separatelylogged. BothW8/A8
formats completedown-calibration developmentscoring; these are actualmixed
integer/FP32-exception routes, not fakequant runtime. Qualityresults below:
stacked_01_supervised_s42_r2 W8/A8:TP142/FN258/FP9/TN791, threshold0.999895572662.
local_context_03_supervised_s42_r2 W8/A8:TP188/FN212/FP16/TN784, threshold0.999074459076.

These remaindevelopment-only andnotfinalnominations. Accountusage43% used
againstuser-raised60%; noresetconsumed. Confirmationunopened.


### Git synchronization checkpoint, 14 October 2026

Stage 2 terminated successfully. BERT 5e-5 refinement full development primary
evaluation: TP316/FN84/FP0/TN800, 79% recall/0% FPR, below fixed 99% recall.
Checkpoint SHA256: 95025bfc0e29163efea613d556d7da0b516ec1bb6c6fac9d9c4c9c6f1f373525.
Seven independent common-recipe/format audits passed in 28.000 CPU seconds.
C98 CPU export: 512 FIT calibration windows, four reference vectors,
3,439,184 bytes, maximum logit error .059832; no FPGA measurements.
Independent inspection found export calibration queries content_token_count/
content_length while R2 supplies content_tokens. Existing exports actually
balance source/genre with collapsed length strata; advertised length balancing
is unproven. Saved artifacts/results remain tied to that sampler. Repair needs
new qualification and preservation of existing artifacts. Confirmation remains
unopened; distillation and frozen three-seed confirmation are pending.
The user requested synchronization of this incomplete study state to Git.


### Qualified teacher and bounded distillation dispatch

Independent refined-BERT score audit passed (2.906 CPU seconds): primary
316/400 attacks, 0/800 benign false positives, hard 0/182, natural-email
72/100 attacks and 0/200 benign. Common BERT gives 275/400 and 2/800.
The refined FP32 checkpoint is selected as a useful same-FIT teacher, not a
security-qualified guard. FIT-only cache binds its exact checkpoint, completed
development record and FIT identities. Before any distillation fit, weight .3
and temperature2 were fixed for LocalContext and C98 replacement, with .7
ground-truth BCE and .3 temperature-squared soft Bernoulli BCE of document-max
margins. Corresponding supervised runs are same-architecture/data/seed/optimizer
controls. No inference teacher, new architecture or confirmation access.

FIT-only refined-BERT cache completed: 36,370 documents/47,283 windows,
26.407 seconds; cache SHA256
3e97743cb216568d31d7f463b4428d25dca72e89afc92cceb2ed64578896c182.
Serial distillation queue started with LocalContext; no confirmation access.


### Corrected calibration and optional transfer feasibility

The corrected exporter resolves R2 content_tokens and checks aliases explicitly.
Its actual FIT-only sampler check used 36,370 metadata rows and selected512
windows/500 source groups, spanning12 source/genre/length strata and all four
length bins. Two common contextual controls were re-exported into separate
_length_r3 directories; old manifests/scales/scores were untouched. Shared
calibration identity:5a11a1c912c7eb2f50ab4763a348348fd327d9e8f76a61b74040913d90c7624e.
Exporter SHA256:403017af3fae3e95d77edc0538e02101800fd5a2eb566c83b30289ec219c241a.
Local/C98 packed bytes remain3,440,780/3,439,184; four-vector CPU maximum
absolute logit deltas are .102971/.102071. These exports do not inherit old
CUDA qualification or development quality. Their records are in
results/cnn_adoption_v1/exports/length_r3_control_pair_summary.json.

An isolated transfer-initialization feasibility check completed in2.421 CPU
seconds without changing the live trainer: raw language-pretrained BERT
WordPiece embeddings were projected via uniform non-PAD FP64 covariance PCA
to existingCNN dimensions, signed deterministically and scaled to each seeded
scratch embedding RMS. Repeated projection and nonembedding weights/RNG were
bitwise equal; PADzero/trainability/finite backward/mask checks passed for all
fourCNNs. This standard technique has no demonstrated task-quality benefit.
No fitting/development text, frequencies or task-finetuned weights enter PCA;
pretraining exposure remains unknown. The raw96-dimensional projection is
e76ae0743036d41cfebf706f060264742d6628ee92e7706692248ca3248f5100.
No integration or second training intervention has been executed/adopted yet.


### First bounded distillation result

LocalContext KD completed12epochs, selectedepoch2 by the unchanged development
rule. Checkpoint SHA256:3361e55b6cb16dc6b770baf89ad8d65b2a9e0e5ae21a19b222aef3b72d41a2b7.
Training/scoring interval784.1493201seconds. Full FP32 development primary
TP192/FN208/FP6/TN794, threshold .9986555576324463 (48%recall/.75%FPR).
Same-data/seed/architecture supervised control188/212/16/784 (47%/2%).
KD improves observed FPR but fails fixed99% recall; no confirmation claim.
The serial C98 replacement KD job is live. Independent score audit and corrected
FIT-calibrated numerical export of the completed first checkpoint are underway.

Independent Local KD saved-score audit passed3.281CPU seconds. Hard-benign FP
6/182=3.30% versus16/182=8.79% supervised; natural-email TP48/100 andFP1/200
versus47/100 and2/200. Exact-control audit .469CPU seconds binds same model,
FIT/development inputs,seed,optimizer,scheduler andepochs. Own-threshold paired
changes:28attackgains/24losses;13benignFPremoved/3added. Old96TP0/FN64/FP0/TN32.
Thus KD's observed benefit primarily reduces FP-tail risk without closing the
recall gap to useful BERT. Required withheld strata remain absent ondevelopment.
New corrected Local KD export uses512FITwindows/500groups/12strata and shared
calibration5a11a1c9...c7624e. Packedbytes3,440,780; fourFITrefmax/mean logitdelta
.127228/.040494, argmax4/4,CPU10.140seconds. ManifestSHA256
616cc048b8bb6ad7032c7f700fd7c1b96c23263983e804c734def3d003c7cb29.
GPU packed qualification and actualformat full quality are still pending.


### Prospective second-intervention selection rule

Before C98 KD finishes, the remaining intervention is bounded to one existing
contextualCNN: if completed KD still fails primary recall, choose the stronger
of the two KD-only recipes by the existing checkpoint quality key (highest
primary recall while below99%, then FPR; fixed tie priority LocalContext).
Test raw-pretrained PCA embedding initialization plus the identical KD recipe
against that KD-only control, with the same12epoch horizon, seed42, architecture,
FIT/cache, optimizer and decision policy. Pretraining representation is a named
remaining difference from BERT after KD's limited recall improvement. This is
standard transfer learning, not a new security contribution; PCA/centering/
anisotropy and lexical knowledge change together. No runtime BERT or new encoder
computation enters the CNN. All nonembedding weights and RNG must remain equal.
No improvement of the prespecified development primary key would falsify its
usefulness for selection. No further training intervention or architecture
search follows this bounded test. Final recipes/formats/priority must then freeze
on development evidence, with all3seed comparison and once-only confirmation.
Integration is prepared separately and cannot replace the trainer until the
live KD queue terminates; original executed source/artifacts stay preserved.

Paired Local KD FP32 vs refined BERT FP32 development comparison executed
1,000 draws, seed20261014. Nominal CNN-minus-BERT recall -31pp andFPR+.75pp.
Selection-source800group percentile95% sensitivity interval: recall[-36.041,
-25.309]pp,FPR[+.25,+1.375]pp; attackpayload20group recall[-47.727,-11.744]pp,
FPRundefined. Paired1000benigncalibration-group refits withselectionfixed yield
recall[-41.25,+15.75]pp,FPR[+.25,+1.125]pp. These distinct dependence/tail
sensitivities neither rescue fixed nominal gates nor prove noninferiority.
OutputSHA256:947aa7f83ea3167d456c6b7ca331a271576e8623a6a438d94239a9b7ff28fcb7;
results/cnn_adoption_v1/local_kd_vs_refined_bert_comparison_r2.json.
No inference/GPU/reserved access. Whole-process CPU duration was not captured
by the comparator; recorded as unknown, not zero or inferred from wait timeout.


### Second KD completed; final bounded intervention released

C98 KD completed12epochs, selected4, CPd17e8e63c177a8a2ae6d21d02ae7159c8ad07a3117b6e5fab1365ceeaa9196b6,
816.9545773training/scoring seconds. Full FP32 primary TP171/FN229/FP7/TN793,
threshold .9994576573371887 (42.75%/.875%), vs supervised165/3.
Local KD192/6 wins the prospectively recorded rule. The isolated reviewed
PCA integration was applied only after serial KD queue terminal0; original
trainer689391...14d04 preserved, newtrainer a9589bd633b67f413be2cafdf3b79f3f4d4f1bdbcb18c48fdb724ee434f34e4f.
The Local PCA+KD recipe retains allarchitecture/data/loss/seed/optimizer/horizon
fields and binds frozen projection/source/provenance hashes. No further training
intervention or newarchitecture search remains. Serial queue first confirms
actual LocalKD W8/A8 development quality then fits/evaluates Local PCA+KD.
LocalKD native signed integer qualification passed35dots/4FITwindows5.1374104s.
C98KD corrected CPU export:3,439,184bytes, max/mean4FITdelta .076938/.035418,
9.922CPU seconds; manifest6d57a4abab0e457351fe9ced9252e1ee5f3b146a08445926cf648c2bddbc63e7.
All existing scores preserve executed oldsources; no re-association or overwrite.
Confirmation remains unopened; final3seed/format/priority freeze still pending.

Local KD W8/A8 independent audit passed5.657CPU seconds; ownthreshold
.998722493648529 gives189TP/5FP vsFP32192/6. Exactly3attacklosses and1FPremoved;
emailunchanged48/100TP/1/200FP, hard5/182=2.747%, old96zeroTP/FP.
NineW8packed tensors and8A8scales matched codes/manifest and35dotnativeproof;
actualCPtrainingtrainer689391... andcurrentW8scoringtrainera9589... are retained
as separate identities. No inherited floating quality or newconfirmation claim.
Actual PCA training association confirms96dims, PADzero/trainable, nonembedding
state/RNGunchanged, scratchRMS .9997781367 vs initialized .9997781573.
Initializerartifact/source/provenance/teacher-cache hashes are bound inrun.json.
Accountusage actual45%, cap60; noresetused.


### Fixed three-seed stage released before confirmation

All additional training interventions are complete. Final Local PCA+KD selected7
CP6f34e8c...63f342d,12epoch fit/scoring850.8515419seconds; fullFP32207TP/7FP.
Independent audit verifies51/100emailTP/1FP,hard6/182,old96knownTP16/64.
PCA gain is concentratedtranslation; response-redirectionTP49→31/per80.
Corrected512FIT-window CPUexport3,440,780B,13.187seconds, fourFITmaxlogitdelta
.059180; native35dotcheck4.9811459seconds passed. FullW8/A8TP211/FP8
(52.75%/1%),threshold .9997921586036682. C98KD native35dotcheck4.7931423s
passed; fullW8/A8TP172/FP7 (43%/.875%),threshold .9994614720344543.
Both intendedroutes fail99%recall; noQAT is warranted to rescue a floating
securityfailure. No numerical quality/timing from a parent is inherited by
a new final checkpoint.

Before final fits, existing primary development key selected LocalPCA+KD7,
C98KD4, and matchedBERT5e-5 epoch4 withschedule_total_epochs5. Nine newfixed
fits for42/1337/2026 nowrun serially without development monitoring; eachnewCP
then gets full development qualification. Priority LocalContext→C98, deployment
seed42. IntendedCNNadoptionformatW8/A8 with explicitFP32exceptions; matched
BERTFP32. CNNFP32confirmation companions diagnose conversion and cannotrescue
failed intendedformat/seed. These rules precede newconfirmation scoring.
Currentconfig/archiveparentrecord preserves exactrecipes and allsources.
FreshhistoricalW8 development alias will qualify the current scoring-source
closure without altering its oldcheckpoint/threshold orcanonicalevidence.
Actualfinal exports, arithmetic/profile qualification and prospective freeze
remain required before once-only heldout scoring. No further tuning follows.


### LocalContext final development replication

All three fixed seven-epoch LocalContext fits and FP32 development evaluations
completed. Independent saved-score audits passed. Seed42: TP207/400, FP7/800;
seed1337: TP256/400, FP19/800; seed2026: TP170/400, FP6/800. Natural-email
TP counts are51/69/36 per100 andFP1/3/1 per200. Hard-benign FP counts are
6/19/6 per182. All fail pooled/email99% recall; seed1337 additionally fails
pooled/email FPR and hard-benign bounds. These are development observations,
not a confirmation verdict or sampled-seed population claim.

The seed42 fixed FIT-only final checkpoint50eefbb95597519ef268d0e7262c05f470b947076c167c7603f762a5060a08eb
reproduces all29 model-state tensors of its selected parent exactly. The newly
completed C98 seed42 checkpointecf5a25b99bd810ce624cad9eac95682af2db5e7de6d8623b1c6a0d67183ce72
also reproduces all29 parent tensors despite the recorded trainer-source change.
Final-fit associations/history contain no development monitoring. New checkpoint
files retain their own digests and still require fresh numerical/profile checks.
Three Local CPU exports completed; a watcher association-field lookup error
delayed the first export, was corrected, and its interrupted record preserved.
No training was restarted and no partial export was overwritten.


### Prospective final timing routes and loader qualification

Before final profiling, the FP32 benchmark loader was optimized to restore the
exact fitted state directly, with lazy ZIP mapping and no redundant pretrained
base-weight load. CPU checks passed in42.078seconds: exact descriptors, model
states and nonpersistent buffers; identical logits onfour actual FIT windows;
matching BERT SDPA classes, padding invariance and strict missing-key rejection.
Profiler SHA25620da10034392fb4e4661af29886e756755d389e3c90b9048be51d03dca0e2849.
Original profiler and proof are preserved underprofile_loader_check. Trainer,
evaluator, checkpoint and decision policy were unchanged. New final checkpoints
still require full saved-score route qualification before timing.

Fixed seed42 comparative routes: FP32CNNs/BERT onCPU1/4threads andCUDA eager/
graph; W8/A8CNNs onactualCUDA integer eager/graph. Coldstartup uses two matched
fresh-child documents onCPU1 FP32 andCUDAgraph intendedformats, with noOS-cache
flush. The CPU INT64integer oracle is a correctness implementation, so its
nonoptimized latency is omitted from the practical speed comparison. Sampling
remains seed424242, two pergenre/label/window-count stratum plus two longest
perlabel, three warmups and20retained repetitions; all rawtext, tokenization,
copies, complete windows, aggregation and fixed-threshold decisions are timed.
Scope is deploymentseed42; otherseeds inherit no timing claim.
Accountusage last observed48% against60%; noresetused.


### Final training completed; numerical and timing qualification live

All nine fixed fits completed:19 terminal learning jobs including the initial
interrupted fit. Independent FP32development audits passed for allnine. BERT
recall/FPR for42/1337/2026:79%/0%,47%/.375%,75.75%/.125%; emailTP72/33/70
per100 andFP0/0/0 per200; hardFP0/3/1 per182. BERT is not uniformly stronger
acrossseeds. Every finalmodel still fails pooled/email99%recall. All3seed42
parent reproductions are bitwise proven (Local29, C9829, BERT73state tensors).
All6 finalCPUexports completed; driverelapsed990.766seconds includeswaiting
and is distinct from activeexport processwalls. Finalsource/manifest bindings
and prospective profiling/intended-format policies were recorded after training
terminal and before qualification. Serial native/fullW8-A8/profile dispatch is
live. Newconfirmation remains unopened and noverdict is claimed.
Savedhistorical canonicalmetrics were reused without reopening protectedinputs:
results/span_coupled_pilot_v1/reference_bert_v1/metrics.json SHA256
b05b6664129286bac67639fad24f5524b68ce54548cfcb9f77d5e41fadedd9e8.
SavedcanonicalN4107, TP2738/FN0; alpha .005/.01/.02 FP4/13/43 per1369benign.
Thresholds .41116243600845337/.004557917360216379/.0004980998346582055.
Originalalpha.01threshold is preserved separately from new-panel recalibration.
