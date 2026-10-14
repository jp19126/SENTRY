# SENTRY: broad CNN architecture exploration for an FPGA-oriented document guard

Work in the existing SENTRY checkout. Execute a substantial first-stage CNN study for indirect prompt-injection detection in external documents. Develop models, run experiments, analyze results and leave reproducible implementations—not just another research plan.

## 1. Purpose and authority of this execution request

Investigate which convolutional representations, architectures and decision mechanisms can distinguish injected instructions from legitimate document content while offering a credible route to efficient FPGA implementation.

Explore broadly. Do not preselect the previously suggested local/context CNN, assume BERT is necessary, or stop after one unsuccessful CNN recipe. Established CNNs, substantial adaptations and new combinations are all legitimate candidates. Prefer genuinely different computational structures over many nearly identical width settings.

This is preliminary architecture discovery. A successful first stage can identify promising models, explain their behavior and establish a useful security–cost frontier without demonstrating a novel algorithm or completing hardware/software co-design. Do not impose publication-level novelty as an entry requirement for each experiment. Equally, do not describe a conventional CNN that passes the benchmark as SENTRY's completed scientific contribution.

The overall project remains FPGA-centered. Software training and evaluation are instruments for selecting and understanding an eventual FPGA method; they are not authorization to turn SENTRY into a software-only security project. Full co-design is welcome where it answers a concrete question, but full FPGA integration is not required at this stage.

When submitted for execution, this request authorizes the local, board-free study and allowances below. It replaces the earlier requirement to select one specific new method before starting this CNN exploration. Do not repeatedly request permission for model choices, training, ordinary corrections or progression within this study. More recent explicit user decisions take precedence.

It does not authorize physical-board operations, paid services/cloud compute, new target-LLM API calls, protected final evaluation, unrelated campaigns or GitHub synchronization.

## 2. Establish the actual starting point, then proceed

Read README.md, AGENTS.md, RESEARCH_SCOPE.md and STATUS.md. Next inspect the current execution prompt, reports/protocol.md, reports/goal_r1_decision.md, scripts/r1_models.py, scripts/r1_challengers.py, the shared evaluator and relevant saved artifacts. Read any newer local CNN/method-study report before assuming R1 is the latest work.

Distinguish committed reports from locally available checkpoints, raw scores and datasets. Check the working tree and necessary compute/tool access once; preserve unrelated changes and reuse valid artifacts. Do not rerun historical campaigns merely to reconstruct their documentation.

Verify the previous CNN's actual architecture, training, scoring and error strata. The reported R1 outcome—high attack recall but excessive benign false positives—is a starting observation, not a diagnosis or a rejection of CNNs. Inspect available false positives, false negatives, length dependence and learning curves before proposing causes. Do not assume a newer quoted “negative candidate result” refers to R1.

Use the previous literature review as a set of leads. Verify the original papers, accessible implementations and relevant newer work for the families actually tested. Distinguish direct prompt-injection evidence from language, time-series or hardware analogies. Do not reproduce unverified bibliographic or performance claims from chat. Keep this focused literature check subordinate to implementation and experiments.

## 3. Establish data discipline suitable for a broad search

Keep the current task: classify external document content for injected instructions. Do not quietly introduce the user question, target response, attack-template identifier, label-derived span marker or other privileged metadata as an inference input. A document label does not establish that an attack succeeded against a target LLM.

Preserve the canonical datasets and historical split files. Create study-specific, source-group-disjoint fit, inner-calibration and inner-selection folds inside the permitted training partition for architecture search. A practical starting split is approximately 70/15/15 by original source group; adjust once before fitting if group counts or composition require it, and record the choice. Keep each clean source and all its variants together. Inner-calibration benign examples fit thresholds; inner-selection examples guide architecture/hyperparameter choices. Never call either fold an independent final test.

Freeze these memberships before the search. Fit any learned vocabulary, normalization statistics or task-specific pretraining only on permitted fit data. Inner-selection may support checkpoint selection, but inner-calibration must not become extra optimization data during this phase. A historical checkpoint trained on these inner folds is not an unexposed comparator there: retrain the shallow CNN on the study fit data for the inner comparison.

After architecture and training choices are frozen, finalists may train on the full canonical training partition. Use the existing search-threshold benign partition for each finalist's threshold and the separate search-scoring partition for post-search development evaluation. Do not use those results to launch another tuning loop within this study. These partitions have historical exposure and remain development evidence, not a newly untouched test. Keep final calibration, final test and NotInject closed.

Freeze separate development-only diagnostic probes before candidate evaluation where suitable permitted data exist: benign descriptions/quotations of attacks, ordinary imperative language, long documents, insertion positions, chunk boundaries, varied topics and text perturbations. Keep diagnostic inspection distinct from training augmentation. Check overlaps with all fitting and selection sources.

Use genuinely withheld templates/families or new independent development examples for generalization claims. Existing templates do not become unseen by changing source emails or split names. Any leave-template/family-out experiment needs separately trained models with those mechanisms and near-duplicate payloads absent from fitting, tuning and teacher exposure. Synthetic inserted attacks demonstrate labeled detection behavior, not target-model compromise. If independent probes are unavailable, finish the usable comparisons and state the limitation.

For token/character/byte comparisons, process the same complete raw documents and preserve tails. Make overlap, span alignment, representation length and preprocessing explicit; do not compare a full-document model against a truncated-prefix model. Character and byte inputs are different. Label any change to context extent, windowing or document aggregation as its own experimental factor.

## 4. Explore a broad structural search space

Aim for 12–16 meaningful architectural families or structural branches and roughly 80–120 distinct configurations, subject to the execution budget. The following are a starting map, not a whitelist or a requirement to force every possible combination:

| Family or structural branch | Main variation/question |
|---|---|
| Original-style TextCNN | Parallel short kernels; reproduce a competent shallow reference. |
| Stacked token CNN | Build context through ordinary successive convolutions without early pooling. |
| Residual token-preserving CNN | Increase depth while retaining token resolution. |
| Deep pyramid CNN | Progressive downsampling with residual blocks. |
| Very deep/hierarchical character or byte CNN | Alternative input units and deeper local-to-global processing. |
| Dilated and iterated dilated CNN | Receptive-field schedules; shared versus independent blocks. |
| Temporal convolutional network | Causal, delayed/centered and bidirectional context where meaningful. |
| Depthwise-separable/grouped CNN | Separate temporal processing from channel mixing. |
| Large-kernel CNN | Compare contiguous wide context against dilated context. |
| Gated CNN | GLU-style or simpler gates versus capacity-matched ungated computation. |
| Multiscale/Inception-style CNN | Parallel receptive fields and fusion locations. |
| Local-and-context CNN | Contextualize local evidence before pooling; test rather than assume its benefit. |
| Hierarchical document CNN | Convolutional evidence combination across sentences/chunks. |
| Dense-connected or alternative skip-connected CNN | Feature reuse, activation storage and trainability. |
| Pooling/evidence-retention variants | Max, max-plus-mean, k-max/top-k and bounded learned summaries. |
| Conditional or multi-representation CNN | Lightweight dynamic kernels or token/character fusion where simpler results justify them. |

Favor a strong token-preserving contextual branch, a competent DPCNN-style branch and a compact character/byte branch, without making any the presumed winner. Some rows overlap: record the implemented topology and count genuine distinctions, not labels.

Explore depth, channel width, embedding dimension, kernel sizes, dilation, residual paths, normalization, pooling, shared weights and fusion position. Include small, medium and larger reference capacities. Compare equal-resource or equal-receptive-field pairs where they answer a question. Use representative sampling and evidence-guided expansion rather than a huge Cartesian grid.

Do not make every family inherit one unsuitable optimizer or learning rate. Give each a modest, comparable adaptation allowance using inner data. Record tuning effort. Conversely, avoid creating a different data/training advantage for every architecture.

Implement shared components and a small configuration-driven model registry using the repository's conventions. Do not build a general neural-architecture-search framework when ordinary scripts and a run table suffice.

## 5. Execute a broad-to-deep search, not a one-recipe screen

Use these proposed default allowances unless a newer explicit resource limit is smaller:

- Up to 128 distinct architecture configurations, including structurally motivated refinements.
- Up to 24 configurations promoted to full inner-development training.
- Up to 8 frozen finalist architectures, normally trained with three prespecified seeds each.
- Up to 24 additional learning runs for targeted ablations, generalization experiments, quantization adaptation or controlled training comparisons.
- A total ceiling of 48 hours of local accelerator job runtime for learning, scoring and profiling; record substantial CPU work separately and cap heavy CPU experimentation at 24 job-hours. Do not substitute a much longer CPU campaign if acceleration is unavailable. Existing explicit usage/resource limits still apply.

These are ceilings, not spending targets. Establish per-stage allocations after a few representative timing runs and record them before dispatching the main search. Reallocate unused allowance inside the total budget toward informative breadth or confirmation. Preserve enough budget for adequately trained baselines, finalist repeats and analysis. Do not spend everything on screening.

**Stage A—bring-up and broad screening.** Check small real inputs, masks, losses and gradients, then train a broad initial set with affordable, comparable exposure. Screening on a representative training subset is acceptable when sampling is source-group consistent and documented. Give architectural diversity priority over minor hyperparameter variants.

**Stage B—deepen and refine.** Promote by a combination of low-FPR detection quality, learning trajectory, structural diversity and cost. Before rejecting a viable family, give at least one representative a meaningful training opportunity; a slow-learning deep model must not lose solely because a shallow model converges in fewer epochs. Apply successive-halving or similar scheduling with a grace period, not blind early elimination. Promote strong candidates and informative near-misses, not only those already crossing the nominal quality threshold.

**Stage C—freeze and confirm.** Select up to eight diverse finalists using inner evidence. Freeze architecture, representation, training recipe, selection rule, intended arithmetic variants and diagnostic comparisons before post-search development scoring. Train the finalists with three fixed seeds rather than selecting a lucky seed. Reuse genuinely equivalent completed runs; do not repeat them simply to fill a count. Evaluate each frozen finalist on the prescribed common partitions and probes.

A short diagnostic run is not a completed candidate result. Distinguish optimization failure, insufficient training, runtime/resource failure and a measured quality–cost limitation. Correct actual bugs and allow modest family-specific training repairs within budget.

Do not close the whole study after one candidate fails, or stop exploring merely because the first model passes. Finish the informative planned coverage, or explain the concrete budget/access limit. Continue useful independent tasks when one route is blocked.

## 6. Select for the task, not headline accuracy

Retain the inherited empirical operating screen—attack recall at least 99% and document-level benign false-positive rate at most 1%—unless a newer explicit decision changes it. Fit thresholds only on the designated benign calibration subset using the current strict decision rule. Do not optimize thresholds on the scoring labels.

Report confusion counts, recall and false positives at the primary operating point, with 0.5% and 2% nominal calibration points as secondary trade-off views. Distinguish calibration targets from observed scoring FPR. Include uncertainty that respects source groups and separately report variability across seeds. A one-example crossing of the empirical boundary is not a statistically established superiority claim.

Break down results by document length, window count, attack family/template exposure, source/topic, insertion position and relevant hard-benign categories. Inspect missed attacks as seriously as benign rejections. Do not use pooling that reduces false alarms by silently hiding localized attacks.

Keep maximum-over-windows aggregation as the initial comparison baseline. Analyze alternative document aggregation separately, with its own training/calibration identity and ablation. Determine whether improvements arise from contextual features, document-length treatment, training information or changed operating points.

Maintain feasible and near-feasible security–cost frontiers. A near-miss is worth understanding but is not a passing model. Do not lower the primary threshold requirement after seeing results, and do not collapse all outcomes into one arbitrary weighted score.

## 7. Isolate architecture from training and representation

The core architecture comparison should share training information, source groups and broadly comparable fitting effort. Reproduce the shallow CNN under this study's conditions; reuse the historical BERT reference only on compatible partitions it was not fine-tuned on. Report its pretraining advantage rather than pretending it is an architecture-only comparison. Do not restart a BERT optimization campaign.

Use a small number of controlled follow-ups for learned versus pretrained embeddings, hard-benign augmentation, teacher distillation, span-supervised auxiliary losses or document-level objectives where evidence justifies them. Give the relevant baseline comparable access. Teacher exposure must respect the evaluation boundary; a teacher fine-tuned on the inner holdout contaminates that comparison even if the student's optimizer never sees those rows.

Span supervision may use legitimate training annotations, but annotations and attack-specific metadata must not enter inference. A quoted instruction or code block is context, not a trusted exemption. Preserve the distinction between detecting, locating, blocking and sanitizing an attack.

For the strongest custom mechanism, run a capacity/cost-aware removal or replacement ablation. Include a competent conventional contextual CNN. Establish whether any advantage comes from the mechanism rather than more parameters, more examples, extra tuning or a weaker comparator.

## 8. Keep hardware and software in the same decision process

Hardware awareness is required; complete co-design and FPGA integration are optional in this stage.

For all serious candidates, record parameter and embedding storage, operation counts, intermediate activation sizes, receptive field, lookahead, buffering and implementation-sensitive operations. Normalize costs per complete document and report length scaling. Character/byte models cannot claim efficiency from per-token numbers alone.

For the shortlist, measure actual batch-one software inference and raw-document-to-decision cost, with representative short and long documents. Separate preprocessing, transfers, network execution, aggregation, cold startup and warm operation. Use enough recorded repetitions for the reported statistics. Compare like execution boundaries and disclose backend/precision; fake-quantization timing is not native INT8 timing.

Investigate quantization sensitivity for a few diverse finalists or their inner-selected precursors, beginning with a practical 8-bit target. Declare weight, embedding, activation, accumulator and nonlinear formats. Use calibration/QAT only on the appropriate permitted data and freeze these choices before confirmation. Check changed decisions and the low-FPR score tail against each model's own floating reference. Do not inherit historical BERT bitwise-equivalence requirements for a new CNN.

Develop a preliminary FPGA mapping for the most promising design styles. Include embedding access, convolution, normalization/gating, alignment buffers, pooling, decision logic and host transport. Distinguish a continuously pipelined design from a time-multiplexed engine; account for the latency/storage implications of weight sharing, dilation and context lookahead. Do not assume depthwise convolution, fewer MACs or fewer parameters guarantees lower hardware cost.

Choose PL, AIE or a heterogeneous route from actual target capabilities and the candidate's computation—not from an a priori requirement to use AIE. Reuse suitable existing tooling and components rather than constructing another generic accelerator framework.

If installed tools and available budget permit, test a small representative complete chain for a conventional baseline and one or two promising candidates. Up to six local synthesis/build attempts, including corrections, are allowed within a separate maximum of 12 hours of vendor-job runtime. Validate relevant numerical behavior and record measured cycles/resources and unresolved surrounding costs. Do not require full-device integration or board-ready packaging, and do not let vendor builds crowd out the architecture study.

Without builds, provide transparent analytical estimates and implementation risks rather than invented LUT, BRAM, clock, power or speedup numbers. An uncalibrated cost model is not accurate enough to eliminate otherwise promising structures automatically. Retain a hardware-diverse shortlist when ranking uncertainty is material.

Streaming inspection does not authorize forwarding unapproved content to the target LLM. Keep any document-retention/enforcement cost visible. No physical performance, energy or downstream protection claim follows from software scores or synthesis.

## 9. Execute autonomously with lightweight coordination

Use available subagents when they help: architecture/literature, implementation/experiments, and hardware/cost analysis are separable roles. Prefer at most three workers with non-overlapping ownership. Use actually configured agents; do not invent model names. The coordinator owns common evaluation, result reconciliation and the heavy-job queue. Run GPU-heavy jobs serially on a shared accelerator unless measured resources justify concurrency.

Use existing environments, cache reusable preparation, checkpoint long work and keep ordinary machine-readable run records. Run focused checks for masking, variable lengths, complete coverage, span alignment, aggregation, calibration, numerical changes and actual interfaces. Avoid elaborate approval machinery, repeated environment audits, unrelated CI or large dependency churn.

Report meaningful findings as they emerge: families covered, major error patterns, promising trade-offs, resource constraints and the next active experiment. Routine research decisions within this scope do not require another planning handoff.

## 10. Deliver evidence and a clear next research decision

Maintain one principal study report, preferably reports/cnn_study.md or an appropriate existing equivalent, with the verified references, search space, data protocol, budget, results, decisions and limitations. Keep detailed execution history in the existing log and machine-readable results under a new study-specific results directory. Preserve all historical evidence and make raw-score/checkpoint identities recoverable.

Deliver executable model/training/scoring code, configurations, a complete run table including unsuccessful attempts, saved finalist scores/checkpoints, quality–cost comparisons and reproduction commands. For finalist comparisons, separate structural differences, training differences and actual numerical/backend differences.

The closing report must answer:

1. Which families were implemented, screened, adequately trained and confirmed? Which remain untested, and why?
2. Which CNNs meet the current empirical screen, how stable are they across seeds, and what do the independent or reused-development probes actually establish?
3. What explains the main gains and failures: context, representation, pooling, training, precision or another factor?
4. Which designs occupy useful quality/latency/storage trade-offs, and which costs are measured versus estimated?
5. What are the best two or three candidates for the next FPGA-centered stage, including a stronger conventional baseline and a hardware-efficient option where supported?
6. Does the evidence suggest a defensible new co-design mechanism, only a promising conventional backbone, or an unresolved limitation? State the nearest competing explanation and the next decisive experiment.

This stage is complete when the broad exploration and meaningful confirmation yield an evidence-backed recommendation or a precisely bounded limitation—not when every CNN has been exhausted or a model has been forced to win. A successful exploratory study need not already be a publishable FPGA contribution.

Keep STATUS.md concise and factual. Preserve the responsibilities of README.md, AGENTS.md and RESEARCH_SCOPE.md; do not insert this study's model list, budgets or goals into standing instructions. Record the active task in the existing execution-prompt location rather than adding competing goal files. Preserve any superseded prompt as history where needed. Use the repository's existing safe-update helper for protected handoff files.

Keep changes local. Do not automatically commit, push, merge, delete branches or overwrite unrelated work. Finish with the scientific findings, changed files, actual execution totals, unresolved claims and a concise diff summary. GitHub synchronization requires a separate explicit user request.