# SENTRY — Task 2: Whole-BERT co-design on VEK280 AIE-ML and programmable logic

**Standalone, resumable execution prompt.** Research direction fixed by the researcher on 15 October 2026. Task namespace: `bert_aie_v1`. This task may run concurrently with `kd_cnn_v2`; neither is a prerequisite for the other.

## Research objective

Conduct a thorough implementation and algorithm–hardware co-design study of SENTRY's complete BERT guard on the AMD VEK280/VE2802, making effective use of AIE-ML and programmable logic. Deliver a qualified complete model, competent baselines, isolated evidence for the proposed co-design choices, a connected hardware implementation, complete-document performance analysis and a defensible contribution assessment. Do not finish after a faster GEMM, a single attention block, a model exporter or an architecture proposal.

Retain the BERT encoder as the principal computation. The primary path must include all four encoder layers, embeddings, residuals, normalization, activation functions, pooler/classifier as actually used, and the document-window interface. A quantization or nonlinear approximation is an explicit numerical/model variant; it is not silently equivalent to the original checkpoint. This task is not authorized to switch to a CNN, an unrelated guard, a generative LLM accelerator, or a smaller transformer and call it the same BERT result.

Investigate three questions:

1. Can a complete, competently tiled and buffered AIE-ML/PL implementation remove the implementation-specific bottlenecks of the earlier design?
2. Which choices of operator placement, intermediate residency, engine sharing, precision and scheduling actually determine the end-to-end cost at our shapes and batch-one workload?
3. Can a specific, experimentally isolated co-design choice improve the quality–hardware trade-off beyond conventional mapping and numerical compression?

Read `reports/goal_r1_decision.md`, relevant numerical/hardware reports, `configs/project.json`, and actual HLS/AIE/host code. Historical slow integer/FP32 services are implementation evidence, not lower bounds on transformer latency. Likewise, published tiny physics-transformer latencies are not predictions for this BERT. Do not present either comparison as the main scientific result. [Repository anchors]

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

Own `reports/bert_aie_v1.md`, `configs/bert_aie_v1.json`, `scripts/bert_aie_v1/`, `hardware/bert_aie_v1/`, `results/bert_aie_v1/` and `build/bert_aie_v1/`, using existing equivalents where appropriate. Keep one principal report and one experiment configuration, with generated artifacts in the corresponding results/build directories.

Required outputs include a verified workload/operator inventory; source-to-hardware numerical specification; competent software and conventional hardware references; the co-design hypothesis and controls; AIE kernels/graph plus required PL/host components; parameter packaging; connected system/build scripts; functional and implementation reports; full-document cost and quality tables; a board-measurement harness; and a reproducible conclusion/resume record. Reuse available libraries rather than creating an accelerator framework as a substitute for the study.

## Stage 1 — Fix the actual computation and establish competent references

Bind the common matched-BERT checkpoint, tokenizer, masks, embeddings and head. Inspect whether biases, pooler, GELU variant, LayerNorm epsilon, attention masking and residual ordering match the assumed implementation. List tensor shapes for each operator, all parameters, current numerical formats, weight/activation storage and window/document behaviour. Preserve token-type and positional embeddings where used.

Calculate the matrix-operation count from the verified shapes. For the expected N=256, d=256, f=1024 and four layers, the principal matrix count should be approximately 0.94 billion MACs per window; derive it rather than treating that figure as measured throughput. Feed-forward and projection work must be included. Batch-one encoder inference still processes a token matrix; do not design it as autoregressive single-token GEMV by default.

Establish at least one competent measured software execution of the shared reference, reusing existing qualified paths where possible. Compare actual FP32 and any relevant accelerated numerical route at matched quality and boundaries. Use an installed efficient backend/CUDA Graph or equivalent without embarking on an open-ended software-infrastructure project. Report a CPU path when qualified and informative; do not falsely call a slow or numerically unqualified historical service the strongest available competitor.

Reproduce the earlier implementation's important bottleneck breakdown only from relevant artifacts or a focused measurement. Separate algorithmic arithmetic, memory traffic, serialized service overhead, transfers and nonlinear costs. Do not rerun all historical campaigns.

Review [B1–B7] and the currently usable implementations. For every borrowed technique, record source device/generation, model, sequence/batch size, numerical format and evidence boundary. Distinguish original AIE, AIE-ML and AIE-MLv2. Verify actual VEK280 part/platform, clocks, bank capacities, native arithmetic, interfaces, tool version and licence support for the selected path. Do not assume that a framework's support for MatMul or softmax means the complete current BERT graph can be imported.

## Stage 2 — Select a whole-model architecture and a falsifiable co-design hypothesis

Build an initial cost/traffic model for our actual workload, including embeddings, weight loading, activations, score tiles, softmax/normalization, inter-engine communication, DDR/NoC/PLIO or other used interfaces, and host launch/return. Use the real graph cut and allocation, not just the sum of parameter bytes. At least compare the plausibility of buffered PL and AIE-ML/PL before committing to the preferred implementation.

Plan a competent conventional hardware baseline and a proposed co-designed variant. The conventional baseline may be a well-buffered PL implementation or a straightforward, competently tiled AIE-ML/PL mapping; choose the one that isolates the claimed advance with feasible resources. The old pathologically slow service is historical context, not the sole baseline. If the claim specifically concerns advantage over PL-only, actually implement a comparable PL reference instead of inferring its performance from a paper or a few kernels.

Use the workload and measured bottlenecks to choose a concrete mechanism, for example intermediate-residency-aware operator placement and engine sharing, or a task-sensitive precision/placement combination that reduces costly graph crossings. Do not equate generic fusion, tiling or AIE use with novelty; compare against CAT/SSR and the closest quantization/mapping work. A new hardware architecture can be a contribution if the difference and benefit are genuine, but merely porting a published design is not.

Before tuning, specify what the conventional control changes, which quantity the proposed mechanism is expected to improve, its numerical/quality implications, and what result would falsify its usefulness. Keep implementation feasibility and novelty as separate questions: finish a competent complete reference even if novelty remains uncertain; do not manufacture novelty to justify the implementation.

Use analytical pruning to choose a small set of plausible mappings, then calibrate that model with actual kernels and integrated-block evidence. Avoid an exhaustive matrix of every tile size, precision, device and sequence length. Optimise the measured workload rather than maximum AIE tile occupancy or headline peak TOPS.

## Stage 3 — Define and qualify the numerical path

Implement a readable software reference for each hardware numerical variant. Specify weight/activation precision, accumulators, scales, bias ordering, rounding/saturation, residual alignment, attention scaling and masking, softmax, LayerNorm, GELU and output scores. Keep trained-model identity and execution identity separate.

The existing W8 scope is not a wholly INT8 BERT: inspect every remaining floating-point operation and embedding payload. Do not estimate the memory requirement as total parameters times one byte. Plan embedding residency or lookup/transfer explicitly. Report actual stored formats, scales, alignment, replication and live activations.

Use practical native formats as starting candidates, such as INT8 projections/feed-forward computation and BF16 attention with appropriate accumulation, only after checking target support and numerical suitability. A conventional reference retaining the original operators comes first; differing floating-point summation order need not reproduce an unnecessarily serialized historical sequence.

Qualify quantization and operator approximations on permitted calibration/development data, then use limited QAT/adaptation when necessary. Preserve the original floating reference and direct-conversion result. Evaluate rounding effects near the decision boundary, held-out quality, saturation and mask/tail behaviour, not only average tensor MSE. Distinguish applying a calibration-frozen threshold from separately calibrating the deployed format; both can be informative, but they answer different questions.

Consider I-BERT-style nonlinear arithmetic or the head-calibrated softmax surrogate in [B7] only when the conventional implementation identifies an important bottleneck. Replacing softmax/GELU/LayerNorm or changing precision is a named variant requiring its own qualification and, where appropriate, retraining. Do not remove normalization or attention because a physics model omitted them. Do not substitute causal attention for bidirectional attention. Do not adopt a new compressed-attention model merely to rescue a poor mapping.

If a numerical variant needs training, freeze its adaptation recipe on development data and assess final variability with fixed seeds as appropriate. If a mapping preserves the same qualified model/numerics and changes only scheduling, repeated training is unnecessary; use differential execution evidence and repeated performance measurements instead.

## Stage 4 — Implement a complete encoder block, then all four layers

Use [B1–B5] as implementation foundations, verifying current source support and keeping any missing-operator extension focused. AIE4ML or GAMA-derived kernels may be components; the research is not a generic rewrite of those projects.

Implement matrix kernels at the actual shapes and integrate them with the surrounding computation. Investigate Q/K/V projection packing, tile reuse, cascade accumulation, bank placement, DMA scheduling and double buffering when they improve the connected pipeline. Include data layout conversion and padding overhead. Streaming an attention matrix must not be confused with making a correct bidirectional decision before the required window is available.

For attention, preserve scaling and masks and account for QK, normalization and probability–V multiplication. Avoid unnecessary external materialization of attention scores where a qualified tiled/streamed algorithm allows it. Count all required row/tile state and passes. For the feed-forward network, include both large projections, activation and residual/normalization stages. Place nonlinear operations where their compute-plus-transfer cost is best; do not insist that every operation run on AIE or every nonlinear run in PL.

Make one complete encoder block the first integrated milestone. Its reference includes normalization, residuals, attention, GELU and feed-forward work. Validate representative real/padded/masked inputs against the specified numerical reference; inspect stalls and the actual critical path. A complete-block result does not complete the task: use it to fix the mapping and continue to all four layers plus embedding/head and complete-document control.

Decide between cross-layer engine reuse, spatial replication or a hybrid using measured block costs, storage and target constraints. Demonstrate whether additional engines reduce complete latency or just move the bottleneck. Separate steady-state initiation interval from first-result latency, and check that the schedule is feasible under actual producer/consumer buffer limits.

Generate the packed parameters, AIE graph and kernels, PL wrappers/buffers/nonlinear units, host/PS input/output path and system integration. Verify real interface compatibility, supported link widths, clock-domain crossings and board platform. Do not invent a working memory-mapped-to-AIE bridge from a block diagram. Include the required DDR/NoC connection and host transport in the actual connected design or state exactly which build dependency prevents it.

## Stage 5 — Complete offline implementation and isolate the mechanism

Run available functional simulation, AIE compiler/simulator, HLS synthesis and representative RTL checks; then perform the linked system build and place-and-route for the final mapping where tools/platform permit. Record achieved clocks, actual resource allocations, AIE tile/memory use, FIFO/buffer state, stalls, routing and bandwidth constraints. AIE simulation assuming operands are already supplied is not end-to-end hardware measurement. Estimated clock goals are not achieved frequency.

Maintain a cost table with separate analytical, simulator, synthesis, routed and physical columns. Model overlap correctly: neither summing every pipelined stage nor summing peak kernel throughput gives complete latency. Include weight/model startup, host tokenization, window assembly, each transfer, graph/kernel launch, return, aggregation and final decision. Distinguish persistent-resident weights from per-document loading.

Execute the controls needed for the actual claim. When both scheduling and arithmetic change, use a compact factorial/paired comparison where feasible: conventional mapping with reference numerics, proposed mapping with the same numerics, and the selected numerical adaptation on the relevant mapping(s). Do not attribute a format-induced speedup to a new schedule or compare unequal-quality models without showing the difference. A precision-allocation mechanism needs a sensible uniform or conventional mixed-precision control; a residency mechanism needs a same-model placement/reuse control. Do not deliberately cripple the baseline.

The endpoint is a connected full-BERT implementation, not a kernel extrapolation. If a build fails, diagnose and fix the responsible integration/timing issue within the campaign; avoid unrelated toolchain reconstruction. If a genuine unavailable licence/platform, proven capacity limit or exhausted resource allocation blocks completion, preserve the exact artifacts and remaining command/dependency, keep the achieved evidence level explicit, and do not claim implementation closure.

## Stage 6 — Full-detector confirmation and end-to-end evaluation

Freeze final model/numerical variants, parameters, threshold policies and the primary implementation before the prospective confirmation evaluation. Evaluate complete documents with the same tokenizer, windows, tails and aggregation as the common reference. Use the qualified functional numerical model for corpus-scale evaluation, clearly distinguishing it from the smaller differential tests actually run through RTL/AIE or a board. Re-run affected quality evidence if implementation changes alter numerical semantics.

Report floating-reference and final-format quality, low-FPR behaviour, hard-benign rejection, mechanism and length strata, and numerical decision changes with uncertainty. Keep ROC-derived oracle points separate from calibration-fixed operation. Any security-aware precision optimisation must have used fit/development data, not the confirmation labels.

Measure competent matched software baselines and obtain all available complete-hardware costs. Report batch-one input-to-decision latency, first-result versus steady-state performance, sustained document throughput, realistic window-count dependence, cold loading and warm resident-weight cases. Include necessary host fallback work and transfers. A single-window number is not a long-document result; a throughput-optimised batch is not a batch-one latency comparison.

Execute the physical continuation only after explicit permission. Check the frozen implementation against real-input references, measure the full connected path, and collect power/energy when possible with the measurement boundary stated. Compare against the conventional FPGA/AIE control on matched hardware resources/clock policies where relevant; report both fixed-clock and achieved-clock interpretations if they differ. Do not infer measured energy efficiency from device specifications.

The CNN track is not a prerequisite. Conclude this track against its own software/hardware/numerical controls. After both candidate freezes, add the other track's qualified point using shared workload/quality boundaries; label unavailable physical results as pending rather than supplying estimates as measurements. No new CNN experiment or unrelated model is needed to complete this task.

## Stage 7 — Scientific conclusion and complete handoff

Answer whether the whole-BERT implementation is feasible on the actual target, which bottlenecks were removed and which remain, whether AIE-ML improves the complete guard rather than only matrix kernels, and what quality/latency/resources/energy trade-off was demonstrated. Separate measured gains from projected ones and give a source-backed comparison to the nearest existing accelerator/co-design method.

State what is new, what was adapted from the literature and what is routine integration. Use the mechanism-removal evidence to support causality. A competent BERT implementation with no demonstrated novel mechanism is a useful engineering result, not the completed FPGA-centered scientific contribution by relabelling. Conversely, do not discard a valid contribution because it fails to beat a strong GPU on every metric.

Deliver the actual runnable software/export/build/measurement commands, configuration/parameter identities, the final paired result table, a concise architecture/dataflow description, the contribution assessment, and a precise physical/downstream handoff. The authorized downstream stage, if later available, compares guard protection and legitimate-task utility; until then report a prompt-injection detector/guard implementation, not proven universal LLM protection.

Finish all feasible offline stages before declaring `offline study completed`. A later invocation should need to resume the documented board stage or a specific external blocker, not redesign the system. Do not silently abandon whole-model integration after an encouraging microbenchmark.

## Literature anchors

Verify current versions, code support and measurement scope. Use these sources to make concrete design decisions, not as guaranteed performance targets.

- **[B1]** Zhang et al., *CAT: Customized Transformer Accelerator Framework on Versal ACAP*, arXiv:2409.09689. Whole-transformer heterogeneous mapping; source platform/generation differs from VEK280. https://arxiv.org/abs/2409.09689
- **[B2]** Zhuang et al., *SSR: Spatial Sequential Hybrid Architecture for Latency Throughput Tradeoff in Transformer Acceleration*, FPGA 2024, arXiv:2401.10417. Spatial/sequential allocation and complete scheduling trade-offs. https://arxiv.org/abs/2401.10417
- **[B3]** Mhatre et al., *GAMA: High-Performance GEMM Acceleration on AMD Versal ML-Optimized AI Engines*, arXiv:2504.09688. AIE-ML matrix implementation and placement; establish the exact simulator/interface boundary before reusing results. https://arxiv.org/abs/2504.09688
- **[B4]** Danopoulos et al., *AIE4ML: An End-to-End Framework for Compiling Neural Networks for the Next Generation of AMD AI Engines*, arXiv:2512.15946. Reusable compiler/kernel path; inspect the current official implementation and operator constraints. https://arxiv.org/abs/2512.15946
- **[B5]** Jiang et al., *Low Latency Transformer Inference on FPGAs for Physics Applications with hls4ml*, arXiv:2409.05207. Attention/nonlinear pipeline techniques; the small physics models do not establish BERT latency. https://arxiv.org/abs/2409.05207
- **[B6]** Kim et al., *I-BERT: Integer-only BERT Quantization*, ICML 2021. Integer BERT numerical methods, not itself proof of this FPGA implementation. https://proceedings.mlr.press/v139/kim21d.html
- **[B7]** Danopoulos et al., *Taming the Exponential: A Fast Softmax Surrogate for Integer-Native Edge Inference*, arXiv:2604.02292. Optional operator/training co-design candidate, not an exact softmax substitution or established security result. https://arxiv.org/abs/2604.02292
- **[Repository anchors]** `reports/goal_r1_decision.md`, `configs/project.json`, relevant numerical/hardware reports and source code, `STATUS.md`, `RESEARCH_SCOPE.md`, `EVALUATION_POLICY.md`. Repository context was inspected on 15 October 2026; resolve actual local identities at execution time.
