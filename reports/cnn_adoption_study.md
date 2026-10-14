# CNN adoption study

Started 14 October 2026 under the explicitly submitted
`SENTRY_CNN_adoption_prompt.md`. This is local, board-free backbone selection
supporting SENTRY's FPGA-centered purpose, not a production guard or completed
novel contribution. Account usage ceiling: **60% used** (raised explicitly by the user), initially observed 36%.
The previous exploration and its failed primary outcomes remain unchanged.

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
