# SENTRY — Task 1: BERT-to-CNN distillation and FPGA co-design

**Standalone, resumable execution prompt.** Research direction fixed by the researcher on 15 October 2026. Task namespace: `kd_cnn_v2`. This task may run concurrently with `bert_aie_v1`; neither is a prerequisite for the other.

## Research objective

Conduct a thorough study of whether security-relevant behaviour learned by SENTRY's BERT teacher can be transferred into a convolutional student whose actual FPGA implementation offers a worthwhile quality–hardware trade-off. Deliver the learning comparison, causal evidence about the transfer mechanism, qualified numerical implementation, complete FPGA inference path, end-to-end evaluation and a defensible conclusion. Do not finish after literature review, training, model export or a parameter-count comparison.

The deployed candidate must remain a CNN-based guard: no BERT call, hidden attention block, teacher-generated embedding service or transformer fallback at inference. Teacher computation and training-only projection heads must disappear from deployment. Fixed or learned embeddings are allowed; count their storage and access cost.

Treat these as hypotheses, not established findings:

1. Richer, task-relevant teacher supervision and appropriate transfer data can improve on the ordinary logit distillation already attempted.
2. A deliberately chosen convolutional representation can retain enough contextual discrimination without reproducing all of BERT.
3. A student designed with actual arithmetic, receptive-field and memory constraints can offer an FPGA advantage that survives complete-document execution.

The prior work was not exclusively scratch CNN training. Inspect `reports/cnn_adoption_study.md`, `reports/cnn_policy_reassessment_20261015.md`, `reports/cnn_study.md`, `configs/cnn_adoption_v1.json` and their source implementations. Existing LocalContext/C98 experiments already include matched-BERT output distillation with temperature 2 and weight 0.3; LocalContext also has a PCA-initialized pretrained-embedding variant. Reuse these as prior evidence and controls, not as a newly proposed mechanism. The original negative verdict is preserved. [Repository anchors]

## Execution, independence and shared comparison contract

### Authority and project context

When submitted by the researcher as an execution task, this prompt authorizes the described local data preparation, training, evaluation, numerical adaptation, FPGA/AIE implementation, simulation, synthesis, offline integration and place-and-route with available tools/licences. Execute the study; do not stop at another plan. The file's mere presence is not authorization.

Read README.md, AGENTS.md, RESEARCH_SCOPE.md, EVALUATION_POLICY.md and STATUS.md, then the relevant source, reports and available local artifacts. Follow their standing rules without rebuilding the documentation system. Reuse working code and valid results; a fresh clone does not include ignored checkpoints/data/builds. These are new studies, not modifications that turn the previous adoption failure into a success. Keep historical artifacts and decisions unchanged.

The application remains indirect prompt injection in external documents for document question answering. FPGA realization is required for the intended research output. Do not substitute a software-only security project or equate detector labels with downstream protection or training/model-weight poisoning prevention. No paid services, external model APIs, credential use, protected final-data access or physical-device operations are authorized here. Physical measurement is a separately authorized continuation. Git commits, pushes, merges and branch creation/deletion also require separate authorization; local research edits do not grant it.

### Parallel ownership and resources

This task does not depend on the other track succeeding or finishing. Use the task-owned paths below and any already supplied isolated worktree; otherwise keep edits non-overlapping in the current checkout. Do not switch the branch beneath another session. Shared helpers and top-level documentation are read-only during concurrent execution: retain needed shared-file patches and proposed STATUS/board-handoff changes in the task report for serialized integration. Keep task goals out of AGENTS.md.

Resolve a canonical project location, normally `/home/jp19126/Projects/SENTRY`, from the actual environment. Both tasks use the same absolute `SENTRY_SHARED_ROOT`, defaulting to `<canonical-checkout>/results/dual_track_v1/shared`; separate worktrees must not silently create separate shared roots. Create/reuse an immutable `contract.json` there containing the common teacher/tokenizer identity, base data memberships/grouping/exposure, window/mask/aggregation rules, calibration/reporting policy, timing workload/boundary, target and resource-lock locations. The first task may prepare it; the other reuses it. Select inputs from approved existing artifacts and development-only evidence, not new confirmation scores. Supplemental fit-only transfer data are versioned additions, not mutations of the base split or teacher.

Use an existing scheduler or simple OS locks for shared setup and each shared GPU, not a new orchestration service. GPU training and definitive timings are exclusive on a shared device; compatible CPU/AIE work may overlap within memory limits. Avoid timing under uncontrolled sibling-job contention. Use available subagents for separable learning, literature, numerical or hardware work, with one coordinator and explicit file ownership per task. Only the coordinator schedules heavy jobs. Do not assume named agents or tools exist.

If a shared artifact is unavailable, identify the missing dependency and continue independent work. Do not silently substitute a teacher. A necessary replacement must be versioned and development-selected, with the original retained. Exchange artifact/interface identities before final evaluation, not results that would influence sibling model selection.

### Shared model and prospective data

Bind the existing matched-data BERT, not the historical seed with the best exposed confirmation score. The expected backbone is `google/bert_uncased_L-4_H-256_A-4`: four layers, hidden width 256, four heads, feed-forward width 1024. Preserve WordPiece tokenization, primary length 256, 64-content-token overlap, complete tails, padding/masks, strict score-versus-threshold decisions and maximum-window document risk unless an explicit controlled method comparison changes them. Derive the exact stride including special tokens from the code.

Use separate fit, model-selection, benign-calibration and new prospective-confirmation data, grouped by original source and relevant attack-payload ancestry rather than random windows. Preserve mechanism withholding where claimed. Keep natural benign examples, hard benign cases and authored constructions identifiable. Extra teacher-transfer data must not introduce evaluation sources or payload ancestors into training.

The previously examined adoption confirmation panel is not a fresh holdout: keep it out of new training, and label any permitted re-evaluation retrospective/regression evidence. Project final-calibration/final-test and NotInject, including copies, remain closed. Use permitted material for new confirmation; disclose data/construction and statistical limitations rather than claiming universal generalization.

Use the same new confirmation and timing manifests across tracks where feasible. Each task can freeze its candidates and finish its evaluation independently. Keep its confirmation results task-private until the sibling's candidates are frozen; do not read sibling confirmation reports while tuning. If those outcomes have influenced choices, disclose the exposure and use genuinely unexposed material for a new independent claim. A simple file-ownership/result-release convention is sufficient; do not build an access-control system.

### Comparable evaluation

Apply EVALUATION_POLICY.md prospectively. Specify the hypothesis, candidate-selection rule, controls, operating points and claim before confirmation. Separate actual device/interface constraints from reporting points and aspirations. Do not reinstate universal-looking gates of 99% recall, a fixed recall margin, 50% storage reduction or every seed individually passing. A formal non-inferiority claim requires a prospectively justified margin; otherwise report the quality gap and uncertainty without inventing a pass/fail boundary.

Report document recall and observed FPR at separately calibrated frozen thresholds, recall–FPR curves, hard-benign rejection, mechanism/source/genre/length strata and reference–candidate disagreements. Unless an existing justified protocol supersedes it, use nominal 1% calibration FPR as the principal shared reporting point and 0.5%, 2%, 5% as secondary points where sample size supports them. These are not safety guarantees. Label evaluation-label-derived ROC points descriptive, not deployable thresholds. Save unsaturated logits/margins where possible and handle ties explicitly.

Use fixed multiple seeds, normally three, for final learning comparisons; do not count seeds as independent copies of the data. Reuse sound paired source/payload-aware uncertainty analysis from saved scores, explaining sparse strata and crossed dependencies. Distinguish teacher/model errors, calibration shift, quantization effects and implementation bugs. A narrow exploratory boundary miss is not a reason to abandon a family, while a large repeatable failure must not be averaged away.

Compare the quality–hardware frontier, not parameter count alone. Keep software measurements, analytical estimates, AIE simulation, synthesis, routed implementation and board measurements separately labelled. Include complete raw-document-to-decision latency: tokenization, all windows/tails, padding, transfers, inference, aggregation and host work. Also report device-only costs separately. Distinguish batch-one first-result latency from throughput/batching, cold from warm operation, and representative from stress workloads; include length/window-count effects and supported latency quantiles. Retain the document until the decision rather than forwarding an uncounted unchecked prefix.

### Multi-session execution and checks

Record a finite staged experiment/build matrix, estimated resources and applicable account/host limits before heavy execution; reserve capacity for confirmation and full integration. Respect current explicit caps and do not assume a reset. Historical counters are evidence, not renewable allocations. Track cumulative costs across sessions and expand the matrix only for a concrete reason within authorized resources.

At a session/resource boundary save usable checkpoints, optimizer/RNG state where needed, exact commands, completed artifacts, the current conclusion and one precise next action in the task report/configuration. Resume rather than restart. Do not imply automatic background continuation. Missing tools/access or exhausted resources make a stage pending, not scientifically negative. Continue independent authorized work without repeated approval requests.

Use focused learning bring-up, arithmetic/mask/tail checks, representative software-versus-generated-implementation comparisons and final-table verification. Exact specified integer operations should agree; different floating-point accumulation orders need not be bit-identical. Quantify numerical and decision changes and reassess the actual format instead of imposing an arbitrary probability-error veto. Fix and rerun affected evidence, not every experiment. Avoid broad CI, dashboards, repeated environment scans, custom integrity frameworks and multiple review committees; use one limited independent review of important final claims and arithmetic.

### Board continuation and completion

Without physical access/authorization, finish all feasible offline implementation, system builds, functional evaluation and cost analysis. Leave exact deployment/measurement commands, artifact identities, connections and unresolved prerequisites. A visible board is not permission to program or probe it.

After explicit permission, resume on VEK280 with representative functional checks and full-document latency/throughput measurement. Measure power/energy when instrumentation permits, stating idle/incremental/device/host boundaries and uncertainty; never infer measured energy from TDP. Downstream LLM protection/utility evaluation requires separately authorized target access and remains distinct from detector/hardware results.

Report `offline study completed`, `physical validation pending/completed`, and `scientific contribution supported/not established/negative within the tested scope` separately. Useful engineering or a justified negative result is legitimate; an unmeasured system is not fully hardware-validated.

## Study ownership and required outputs

Own `reports/kd_cnn_v2.md`, `configs/kd_cnn_v2.json`, `scripts/kd_cnn_v2/`, `hardware/kd_cnn_v2/`, `results/kd_cnn_v2/` and `build/kd_cnn_v2/`, using existing equivalent paths when appropriate. Keep one principal report and one experiment configuration; put generated details in the results/build trees. Do not create a documentation hierarchy for its own sake.

The report must ultimately contain a literature-to-design table, prospective hypotheses and comparison contract, teacher/student/data identities, learning and distillation results, representation/context analysis, numerical-format specification, allocated memory/dataflow design, executed implementation evidence, complete checking costs, uncertainty, limitations, contribution assessment and a precise resume/board handoff. Keep machine-readable per-document scores, per-run metrics, checkpoint/export manifests and reproduction commands.

## Stage 1 — Establish the teacher and the transfer problem

Verify the existing matched-BERT reference and its actual development performance using available artifacts. Use the common anchor for cross-track comparisons. Do not choose a teacher by the best exposed confirmation seed, and do not treat the old easy-development result as evidence of general security robustness.

Identify where the current student differs from the teacher: prediction ranking near low-FPR decisions, hard-benign rejection, attack-versus-discussion distinctions, contextual distance, document length and mechanism/source transfer. Use a compact, train/development-only diagnostic with controlled instruction-versus-quotation/background pairs where needed. Measure the existing students' receptive fields and the context a proposed target depends on; do not assert a whole-family impossibility from one local model.

A better teacher may be considered only as a bounded extension of the same BERT task: development-selected retraining/continued adaptation with the permitted training data, not a new foundation-model project. Preserve the common teacher comparison and record any new teacher as a separate treatment. Evaluate teacher errors against trusted labels. Agreement with a mistaken teacher is not improved security. Do not use an ensemble at deployment to rescue a nominally compact student.

Check the closest relevant sources, their current versions and available code. Use [K1] for direct BERT-to-CNN precedent; [K2] for student/embedding/transfer-data choices; [K3–K4] for richer representation supervision; and [H1] for real CNN FPGA trade-offs. Read only additional sources that change the design or novelty comparison. Reproduce the needed ideas on our task, not every paper's experiments. Do not import sentiment-accuracy or software-speedup numbers as SENTRY expectations.

## Stage 2 — Freeze a purposeful student/transfer comparison

Select a small, deliberate set of student families rather than restarting the previous broad architecture sweep. Include the existing LocalContext and C98 where they provide reusable controls, and examine a wider-context residual/dilated or separable/multiscale alternative only with a clear representational and hardware rationale. Prefer approximately three distinct computational hypotheses, with limited capacity variants, over many cosmetic width/depth permutations. Derive widths, receptive fields and target arithmetic from memory/compute estimates, then update the estimates using actual synthesis evidence.

Create a new fit-only transfer corpus with adequate coverage of the actual application. Separate natural benign text, difficult benign discussion/quotation, allowed attack-bearing examples and controlled constructions. Additional unlabelled data may receive teacher soft targets but must not be presented as human ground truth. Keep licensed/public provenance, grouping, source/payload ancestry and held-out mechanisms explicit. Preserve paired benign/attack context where meaningful; avoid label-specific introductions and other construction shortcuts. Estimate cache/storage size before collecting all intermediate teacher states.

Include the following causal comparisons, reusing exact valid runs where possible:

| Comparison | What it isolates |
|---|---|
| Supervised student versus conventional output-KD student on the same labelled material | Whether ordinary teacher predictions help beyond labels |
| Output-KD on original versus expanded fit-only transfer material | Contribution of transfer coverage, with a repeated-original-data/exposure control where extra training would confound the conclusion |
| Output-KD versus richer distillation using the same student, data and training opportunity | Contribution of the richer teacher signal |
| Hardware-informed student/numerics versus an appropriate conventional student/mapping | Whether the co-design choice adds value beyond a larger model, extra data or routine compression |

This is a targeted comparison graph, not a requirement to cross every architecture, dataset, loss, precision and seed. Screen on development data, then confirm the important differences. Retain the supervised and logit-KD controls for at least the selected student, even if some weaker architecture screens are stopped for a diagnosed reason.

Before confirmation, select the principal student/recipe and an informative alternative only when it represents a genuinely different point on the quality–hardware frontier. Use fixed final seeds, normally 42, 1337 and 2026, without selecting a deployment seed using confirmation results. Keep optimizer schedules, training exposure, checkpoint selection and parameter budgets explicit. Resource limits are engineering limits, not academic acceptance thresholds.

## Stage 3 — Implement and train richer, architecture-compatible distillation

Implement a stable supervised-plus-distillation objective. A starting family is:

`L = lambda_label * L_label + lambda_logit * T^2 * KL(p_teacher(T) || p_student(T)) + lambda_repr * L_representation`.

Do not copy a published loss mechanically. Specify the normalization, temperature convention, weighting, padding exclusion and unit of supervision. A binary output has little dimensional information; test whether a selected intermediate or relational teacher signal supplies useful information beyond saturated final probabilities.

Choose one primary richer transfer mechanism and, if justified, one alternative. Suitable candidates include selected teacher hidden states with training-only projections, dimension-tolerant CKA/relation matching, or pooled task representations. Explain why the student can represent the target. A short-receptive-field token feature should not be forced to reproduce arbitrarily global final-layer BERT features. Local teacher layers/targets, appropriately pooled representations or an explicit wider-context student are preferable hypotheses to blindly matching every layer and head. Training-only annotation masks or teacher features must never be required at inference.

Consider a staged representation-transfer phase followed by task-specific supervised/KD fine-tuning when supported by the diagnosis. Keep it comparable to a matched training-exposure control. Use a compact development search for temperature, relative loss scale and teacher-layer selection; do not make an enormous grid. Cache only the fit-only logits/features actually consumed and reuse them across compatible runs.

Make the window/document objectives consistent with available labels and deployed aggregation. Use span labels only where genuinely available. Do not label every window of an injected document as malicious by default. A smooth training aggregation or different representation target is an explicit method choice, not a silent change to the frozen inference contract. Analyse whether max aggregation and score saturation, rather than missing contextual features alone, explain some errors.

Train through a credible convergence schedule with development-only model selection. Diagnose finite losses, learning progress and teacher–student gaps with focused checks. Do not dismiss a promising family because one seed narrowly misses an exploratory target, or continue training indefinitely when the supposed transfer signal is demonstrably uninformative.

## Stage 4 — Quantization and hardware design must influence each other

Begin hardware characterization before final student selection. For representative candidate shapes, implement the actual convolution/normalization/pooling path sufficiently to measure resource and schedule behaviour; use this evidence to constrain the student. Do not wait until training is completely frozen to discover that the chosen operators cannot be supported.

Use a practical initial integer format, normally W8/A8 with specified accumulator width and per-operator exceptions. Investigate lower or mixed precision only where it addresses a measured resource/latency bottleneck. Use quantization-aware adaptation when needed, comparing the original float model, its direct numerical conversion and its adapted implementation. Attribute quality changes separately to architecture, transfer and arithmetic.

Document actual embedding precision and placement; weight/scale/bias payloads; integer accumulator range; rescaling/rounding/saturation; padding values; residual alignment; normalization, activation and pooling implementations; and classifier output/threshold representation. LayerNorm cannot simply be folded into fixed weights like a fixed affine map. Removing/replacing it or a nonlinear activation is a learned-model variant requiring adaptation and a control.

Derive the real line buffers, channel buffers, branch alignment, lookahead, live activations, weight residency, DDR accesses and document-retention costs. Use allocated memory banks/ports/replication, not just an ideal byte subtotal. Quantify the complete receptive field and streaming delay. Count any host embedding lookup, preprocessing, pooling or aggregation; a deployed BERT embedding service is not allowed.

Use hls4ml/FINN/HLS or a supported AIE path when they fit the selected graph. Verify current operator/shape support and implement a focused missing operator if needed, rather than writing a generic compiler. The default scientific attraction is a compact FPGA convolutional datapath; AIE is available but not mandatory for this track. All claimed deployment computation must be accounted for.

## Stage 5 — Build a complete CNN guard, not merely convolution kernels

Implement the full selected student from declared inputs through embeddings, convolutions, all normalization/activation/residual stages, pooling, classifier and document decision interface. Produce a packed parameter artifact and functional reference for its actual numerical semantics. Establish software-to-export-to-generated-implementation correspondence using representative real, padded, boundary and multi-window cases.

Run the available C/functional simulations, synthesis, representative RTL co-simulation and system-level builds. Complete a connected design and place-and-route for the primary candidate where the installed platform/tools permit it. Record achieved timing, resources, buffers, schedule/II and integration constraints. An isolated synthesized convolution does not complete this stage. A hardware build that is unavailable must be explicitly pending, not replaced with an analytical number.

Implement the best informative student even when it remains below BERT quality; the whole purpose is to establish the actual trade-off. A measured, decisive infeasibility or genuinely degenerate detector can justify terminating the affected hardware branch, but document the evidence and the unanswered hardware question. Failing the historical arbitrary quality gates alone is not such a reason.

Include a conventional FPGA implementation or a focused mechanism-removal configuration sufficient to isolate the proposed hardware-informed choice. Compare against the shared BERT software reference immediately. Incorporate the BERT track's qualified hardware point when it becomes available, using the same data and measurement boundary; its absence must not block this task's own result. Do not inherit that track's resources or timing by assumption.

## Stage 6 — Confirmation, end-to-end costs and scientific interpretation

Freeze final recipes, checkpoints, formats and threshold-fitting policy before the new confirmation evaluation. Score the teacher, selected students and necessary controls on the same panel. Reuse saved raw scores for uncertainty and alternative descriptive operating points. Confirm the inference export actually evaluated, not an unexported float companion. Report float/quantized decision disagreements and distinguish teacher mimicry from correctness against labels.

Obtain matched complete-document software timings and all available implemented-hardware evidence. Compare quality with latency, throughput, storage/bandwidth, allocated resources and measured energy when available. Include length/window-count scaling and hard-benign/withheld-mechanism behaviour. Do not claim an FPGA advantage from the fact that the CNN has fewer parameters or that a GPU route was inefficient.

Carry out the separately authorized board continuation under the common contract. A later downstream evaluation, when target access is authorized, should compare no guard, BERT guard and CNN guard on benign-task utility, utility under attack and attack success; distinguish these from classifier recall. Until then, those claims remain open, not prerequisites for reporting the bounded detector–hardware result.

Close with an explicit answer to each hypothesis. Explain whether the limiting factor was teacher quality, transfer coverage, student context/capacity, the transfer objective, numerical implementation or hardware cost; use the ablations to distinguish them rather than guessing. Identify the closest existing method and the experimentally isolated difference. Standard KD plus standard FPGA export may yield useful engineering but not a demonstrated new method.

The expected endpoint is a complete, reproducible offline learning-and-FPGA study plus physical results when authorized, not a declaration that one promising training run deserves another unspecified project. Recommend the best supported student/implementation or a justified negative conclusion. State separately whether a new contribution has been established.

## Final handoff

Return the selected model and hardware configuration; the principal paired quality/hardware table; what richer transfer changed versus existing output KD; what the co-design changed versus a conventional control; artifact/command locations; physical/downstream evidence still pending; the contribution assessment; and exactly how a later session resumes any remaining authorized stage. Preserve cumulative costs and all negative results. Do not silently select another model after seeing confirmation.

## Literature anchors

These are starting points, not guaranteed recipes or performance promises. Verify the versions and relevant claims before implementation; distinguish peer-reviewed work from preprints and code capability from published evaluation.

- **[K1]** Chang et al., *One-Teacher and Multiple-Student Knowledge Distillation on Sentiment Classification*, COLING 2022. Direct shallow-CNN student and additional-transfer-data precedent; not a security benchmark. https://aclanthology.org/2022.coling-1.614/
- **[K2]** Nityasya et al., *Which Student is Best? A Comprehensive Knowledge Distillation Exam for Task-Specific BERT Models*, arXiv:2201.00558. Student/embedding/transfer-data comparison. https://arxiv.org/abs/2201.00558
- **[K3]** Jiao et al., *TinyBERT: Distilling BERT for Natural Language Understanding*, Findings of EMNLP 2020. Rich and staged distillation; its transformer student does not establish CNN equivalence. https://aclanthology.org/2020.findings-emnlp.372/
- **[K4]** Dasgupta and Cohn, *Improving Language Model Distillation through Hidden State Matching*, ICLR 2025. Dimension-tolerant hidden-state matching with CKA. https://openreview.net/forum?id=IcVSKhVpKu
- **[H1]** Aarrestad et al., *Fast convolutional neural networks on FPGAs with hls4ml*, Machine Learning: Science and Technology 2, 045015 (2021). Implementation/quantization/resource trade-offs on a different workload. https://arxiv.org/abs/2101.05108
- **[Repository anchors]** `reports/cnn_adoption_study.md`, `reports/cnn_policy_reassessment_20261015.md`, `configs/cnn_adoption_v1.json`, `STATUS.md`, `EVALUATION_POLICY.md`, and the source/artifacts they identify. Repository context was inspected on 15 October 2026; resolve actual local identities at execution time.
