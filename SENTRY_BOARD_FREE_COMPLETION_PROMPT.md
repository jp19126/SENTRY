# SENTRY — complete the board-free research and prepare physical deployment

Continuation of `CODEX_RESEARCH_GOALS_v2_1.md`, based on the SENTRY handoff inspected on 2026-10-08. This is an execution prompt for Codex, not a claim that the work below has already been completed.

## 1. Goal and authority

Complete SENTRY's remaining essential work that can be performed without a physical FPGA board. Implement, execute, inspect and fix the required software, numerical validation, hardware integration, cost modelling and bounded method experiments. Leave a usable deployment package and an exact continuation procedure for the physical-board stage. Do not stop after producing a plan, scaffolding, another audit, or instructions for routine work you can execute yourself.

The scientific objective remains: determine whether one concrete improvement to precision allocation or FPGA execution reduces complete online document-checking cost at the same protection requirement. A is the restricted published-method adaptation; B integrates the protection criterion; C adds one evidence-supported change. **B-to-C is the main method test.** Uniform precision is an anchor, not a replacement for B.

When the user starts this prompt in Codex, it explicitly resumes the paused research **for this board-free scope only**, and replaces the historical instruction to execute just one numbered goal. Continue across dependent goals without requesting permission at each handoff. Preserve the original scientific protocol and budgets except for the explicitly scoped offline integration/build authorization below. Keep the original goals document and historical evidence; do not rewrite history to match new results.

Physical-board activity is excluded even if a board becomes reachable: do not connect to, program, reset, boot, reimage or perform live device/memory I/O on it. Do not run board benchmarks or power measurements. Keep `hardware.programming_authorized` false. Preparing and compiling code for those later actions is in scope; executing it against hardware is not.

## 2. Establish the real starting point once

Read `README.md`, `AGENTS.md`, `STATUS.md`, `configs/project.json`, and the relevant sections of `CODEX_RESEARCH_GOALS_v2_1.md`. Then read the current numerical, integration, cost and method reports, particularly:

- `reports/unfinished_measurements.md`, `reports/goal5_adapter_audit.md`, and `reports/method.md`;
- `reports/goal4_numerical_bridge.md`, `reports/goal4_real_prefix_rtl.md`, and `reports/goal4_arena_preparation.md`;
- `reports/goal4_internal_integration.md`, `reports/hardware_cost_model.md`, and `reports/goal4_symbolic_cost.md`.

Read relevant source and saved evidence, not every repository file. Consult the proposal PDF when a scientific requirement needs resolving, and the manuscript when updating claims. Prefer current evidence over stale status prose. Record the actual starting revision and preserve unrelated local changes. Do not reset the checkout or launch old commands blindly.

Use the existing SENTRY checkout and environments. The recorded software environment is native Windows Python; vendor tools execute under Ubuntu WSL. Historical GATE and current SENTRY are different directories. Establish only the path corrections needed to reuse the existing checkpoints, results and compiled vendor artifacts. A successful preview is not proof that a relocated compiled build executes correctly.

`data/`, `checkpoints/`, `results/` and `build/` are ignored by Git. Verify the specific artifacts needed for the next work package are present. Do not rerun completed experiments just because a new clone lacks their outputs. Report a genuinely missing artifact/access item once and continue independent work. Do not add a general environment scanner or provenance framework.

## 3. Preserve the completed work and its lessons

Verify these starting facts against newer local evidence, then reuse them:

- Goals 1–2 and the available Goal 3 software work are complete: prepared grouped data, floating epoch 3, the 120-request institutional application pilot, 20 PTQ maps, two one-epoch QAT anchors, arithmetic checks and short software timing pilots.
- All 12 initial micro-syntheses and the recorded active integer/FP32 component RTL cases are complete. Cached16, cached32 and static-bank64 are the active common profiles. Their existing common repairs are not C novelty.
- The controller/mover integration used arithmetic completion stubs. The six-call real-input RTL prefix used actual arithmetic but is not a complete detector execution. The prepared relative arena is not allocated board memory.
- Floating and quantized score approvals remain unresolved. One-input ordered CPU agreement with C simulation does not qualify the full synthesized detector. Small FP32 attention differences have crossed an A8 rounding boundary; local floating tolerances alone do not establish downstream agreement.
- Historical development results show false-positive/calibration changes, not quantization-induced attack-recall loss. W8 QAT passed the empirical 1% FPR condition; W4 QAT and the floating reference did not. Extra QAT training prevents attributing the W8 improvement to quantization alone.
- The current static64 internal model is approximately 1.145–1.216 seconds at L256/B1, before additional physical-memory and host costs. Precision alone offers only about 6% reduction in the known serialized service subtotal. Investigate dominant execution costs rather than assuming bit reduction produces a large system speedup. This is not a protection-matched board/software benchmark.
- The A/B adapter exists, but the campaign runner, actual comparison and C are unfinished. The application pilot is complete but limited by truncation and loss of task evidence after whole-document rejection; do not repeat it to improve the story.

Preserve the fixed model, grouped splits, W4/W8 choices in 16 encoder groups, A8 policy, declared FP32 operations, training-only frozen scales and selected checkpoint. Keep document-max aggregation and strict `score > threshold` semantics. B/C require development FPR <= 1% and at most one absolute percentage point of recall loss against the qualified matching floating reference. Keep temporary-threshold and candidate-scoring groups separate.

Do not switch the detector, add an AIE/HBM/second-board project, move the target LLM onto the FPGA, introduce a cascade or production gateway, or expand attack generation. Necessary common correctness fixes remain available to every comparison arm.

## 4. Use native multi-agent delegation deliberately

Use the installed Codex multi-agent capability when available. Inspect available agent types, models and supported reasoning settings once. Prefer the following division; these are task assignments, not assumed installed aliases:

| Role | Preferred model, when available | Responsibility |
|---|---|---|
| Coordinator / integrator | Current strongest suitable parent; `gpt-6-astra` or `gpt-6.1-sol` | Scientific decisions, dependency ordering, budgets, shared configuration, integration and acceptance of evidence. |
| Numerical specialist | `gpt-6.1-sol`, with stronger reasoning as supported | Synthesized replay, floating-point/quantization discrepancies and the matching reference. |
| Hardware / cost specialist | `gpt-6.1-sol` | Actual service integration, memory/host design, vendor builds and complete-cost accounting. |
| Experiment specialist | `gpt-6.1-sol` for method logic; delegate narrow implementation tasks | Campaign runner, fair A/B/C execution, QAT reuse and method attribution. |
| Focused implementation / evidence assistant | `gpt-6-luna`, normally high reasoning | Targeted source tracing, parsers, log extraction, focused code edits, tables, figures and handoff documentation. |

Use `luna6` only if the runtime actually exposes that agent/model alias; otherwise select the supported model identifier. Do not invent model settings or claim a model was used when it was not. Do not force identical reasoning levels across models. If per-agent model selection is unavailable, use the supported agents and report the limitation. If subagents are unavailable, execute the same workflow sequentially; do not build a replacement orchestration framework.

Keep at most three subagents active at once. Spawn them for bounded deliverables, not permanent titles. Start with numerical work and hardware/cost work in parallel; use Luna for a small evidence or implementation task. Bring in the experiment specialist when its inputs are ready. Use a separate capable reviewer briefly at numerical qualification and method/cost freeze, not an endless review loop.

Give each worker an exact task, relevant paths, owned files, required evidence and completion condition. The coordinator alone edits shared configuration, protocol approvals and global status. Avoid overlapping writes. Serialize expensive work that competes for the same GPU, simulator, vendor-build memory or benchmark host; parallelize independent analysis and coding. Workers must not independently launch duplicate training or synthesis. No nested agent teams.

A worker's report should state what changed, what actually ran, the evidence paths and unresolved issues. The coordinator must inspect the relevant implementation/results before accepting the work. Delegate routine implementation, but retain difficult numerical acceptance and scientific attribution with the stronger agents.

## 5. Execute the dependency chain

### A. Finish the numerical bridge, then qualify development scoring

Implement the missing replay beyond `scripts/run_real_prefix_rtl.py`, reusing the saved input, completed prefix, frozen W8-QAT checkpoint/scales and unchanged service evidence. Feed each stage the preceding stage's actual outputs. Independently replaying historical software traces does not establish a connected numerical path.

Trace all selected-linear A8 boundaries, integer accumulations, relevant floating boundaries, logits, final risk and document decision. Diagnose the earliest unexplained discrepancy and fix its cause before paying for the rest of a known-failing run. Do not force expected codes, substitute reference outputs, change scales or refit thresholds to hide an arithmetic mismatch.

Before broader comparison, resolve and record the numerical acceptance policy. The stricter terminal-bit-equality rule in the adapter audit was a proposal, not an adopted requirement. Do not silently promote it into a mandatory universal rule, and do not silently relax it into a convenient tolerance. Preserve exact integer arithmetic, packing, quantization for identical inputs/scales, and aggregation/threshold semantics. Use the existing local FP bounds where applicable, but also evaluate propagation into A8 codes and final detector outputs. Match the software reference to the synthesized arithmetic wherever possible. A substantive change to the detector-level acceptance contract requires an explicit research decision, not a hidden code patch.

Use the prescribed approximately 32 representative development inputs, beginning with the existing saved case. Preselect the remaining cases to cover benign/attacked content, padding, operating-point proximity and relevant length coverage; retain the known tie case. Use the least expensive adequate validation route and state which evidence is C simulation and which is actual synthesized RTL. Do not label C-only checks as RTL agreement, claim universal equality from a small sample, or multiply full RTL replay by every search map. Validate an introduced W4/mixed precision path on the same bounded set when it becomes relevant.

Qualify both the quantized computation and the ordinary floating recall reference, including their explicit arithmetic relationship and common execution backend/settings. Approve score identities only after the required evidence exists. Generate the necessary new matching development scores; reuse compatible weights and QAT checkpoints rather than retraining them. Historical SDPA score files remain historical and must not be relabelled.

If the qualified ordered reference is too slow for the bounded search, make an equivalence-preserving implementation improvement and check the affected arithmetic. Do not replace it silently with a faster unqualified backend or relax the model-evaluation budget.

**Done when:** the complete numerical path has bounded, documented qualification, the matching floating/candidate reference identities are justified, and the development-scoring path executes correctly. Record a real unresolved numerical issue separately rather than calling the reference approved.

### B. Complete the baseline integration and a usable whole-check estimator

In parallel with A, connect the real integer and FP32 services, sequencer and mover. Retain the existing stub simulation as control evidence, but add actual-arithmetic integration evidence. The checkpoint, packing, addresses, scales, IDs, masks and output semantics must agree across host code, generated command plans and hardware.

Select one implementable VEK280 host/memory architecture using the installed platform files, existing environment and authoritative vendor documentation. Make ordinary design choices autonomously. Distinguish a selected deployment configuration from an observed fact about the physical unit. Specify memory placement, port sharing, synchronization/coherency responsibilities and the host transport rather than leaving an unspecified 'DDR overhead'. Do not assume Linux, XRT, PCIe or a board revision merely because another project used it.

Measure available host preprocessing/packing work on the intended accessible host. Model unavailable transport and physical-memory effects from the actual chosen design and defensible assumptions. Include embedding access, attention, fixed FP32 operations, movement, control, synchronization, warm/cold boundaries and complete document/window work. Do not add nominal memory service twice on top of RTL cycles that already include it; do not credit unimplemented overlap or CPU offload of detector arithmetic.

Extend `scripts/estimate_detector_cost.py` into the complete estimator needed by the method interface. Distinguish measured component/host quantities, analytical or simulated estimates, and quantities requiring future board calibration. An offline full-check estimate is legitimate when supported; it need not pretend to be a board measurement. Validate integrated resources/timing using actual implementation reports and justify bandwidth feasibility. Inspect the reports, not just tool exit codes.

Freeze a supported nominal cost model and the common millisecond caps before comparison. Check whether plausible unresolved memory/host variation changes feasibility or method ranking; keep sensitive conclusions conditional. Never fill unknown fields with zero or weaken the existing eligibility checks just to make `search_ready` true. A symbolic-cap substitution is a protocol change, not completion of the current cost-table contract.

Use the completed cost breakdown to identify the dominant bottleneck. Apply necessary common engineering fixes to all arms. Do not give C routine caching, more resources or a corrected baseline implementation and call that novelty.

**Done when:** the integrated baseline, justified physical search space and documented full host-to-decision estimator support the actual campaign contract. If an essential platform fact cannot be established offline, identify the exact missing input, finish the independent code and do not falsely certify a finalist.

### C. Finish and run the bounded A/B/C experiment

Implement the missing campaign execution/resume path around `scripts/search_precision.py`; use the existing adapter rather than a new framework. Connect actual score generation, common cost lookup, result reuse, logical candidate accounting, the frozen QAT shortlist and post-QAT selection. Verify this new execution path on the smallest real permitted work, then continue it; a fixture-only adapter check is not the experiment.

Retain the existing Quasar-derived operators, common physical profiles and declared A/B criterion hooks. B's provisional pre-QAT candidates may enter the bounded shortlist, but final B/C eligibility remains strict. Reuse identical scores across methods only when all numerical/checkpoint/data associations match, while charging each arm for its logically consulted maps.

Run A and B when A/B's numerical and cost prerequisites are satisfied. Use their results and the existing sensitivity/cost evidence to choose one defensible C change. Record the specific weakness, proposed mechanism, expected cost effect, closest prior mechanism and controlled comparison before C's comparative execution. Consult the nearest relevant primary sources once at this decision; do not start another broad literature campaign.

Implement and test that one change within the original allocation/selection or FPGA-execution scope. Preserve equal baseline tuning and resource access. Count its real conversion, storage, movement, buffer and control costs. Any common arithmetic change requires affected reference qualification and development rescoring, not selective updating of C.

Stay within 64 logically consulted maps per A/B/C arm, at most three active engine profiles, and at most two shortlisted maps per arm receiving the common one-epoch QAT allowance. Reuse compatible completed uniform QAT runs. Keep diagnostic method-specific candidates inside their arm's allowance. Perform one mechanism-removal comparison; reuse B when removal exactly yields B.

Freeze precision maps, physical configurations and qualified score computations from development evidence. Select the uniform comparison by the same protection requirement; retain W4/W8 as software anchors. Prepare U/B/C, not an automatic fourth A board design. A tested C that adds no benefit is a valid result; do not expand the search until C wins. An untested idea or unexplained implementation failure is not a negative scientific result.

**Done when:** actual bounded comparisons, attribution, candidate records and figures exist, with an implemented/tested C outcome or the explicit null outcome justified under the original goals. Distinguish predicted hardware benefit from measured hardware benefit. Do not claim Goal 5 complete from scaffolding or leave the researcher to invent the mechanism.

### D. Prepare a deployable hardware and host package

Prepare uniform W8 bring-up first, while the method work proceeds. Once finalists are frozen, prepare their distinct implementations. Reuse a build when only compatible weights, scales, thresholds or precision configuration change; method labels do not justify separate synthesis.

Complete the actual host-facing software, parameter/checkpoint packing, service connection, command generation, build scripts and configuration for the chosen platform. Compile the host code and run its offline functional path. A fake device may validate host control/error handling, but is not detector numerical or performance evidence. Build the appropriate deployment artifacts for the actual flow; do not require unrelated image formats or a new platform stack.

Cover the original required lengths 128/256/512 and batches 1/8/32 in the executable workload path. Start with L256/B1, then implement the missing schedule/shape handling. Do not claim L512 support by treating two independent L256 attention windows as one L512 inference. Sequential batch execution is acceptable when honestly reported; do not claim parallel batch acceleration without an implemented schedule. Ensure long-document windowing, maximum aggregation and reset between documents remain correct.

Run the permitted offline synthesis/implementation required to establish resource/timing closure and produce deployment files. Reuse completed component validation; rerun only checks affected by a source/arithmetic/interface change. Keep any unresolved platform/runtime values as explicit configuration inputs, not buried TODOs.

Prepare a later board runner with a non-hardware preview/prepare mode. Actual programming and measurements must require a separate explicit execution request and approved hardware settings. Its later sequence must cover uniform W8 bring-up; U/B/C execution; approximately 1,000 development-example agreement checks per final arithmetic-changing build; warm-up and completion-aware timing; kernel versus raw-text-to-decision boundaries; the original repeated workload and matched-load/energy measurements. Do not execute that hardware branch now.

**Done when:** another session can begin physical bring-up from real build artifacts, compiled host software, packed models, documented required runtime inputs and exact commands, without first having to implement the missing software. Actual on-device success remains unclaimed.

### E. Finish eligible evaluation work and the evidence handoff

Do board-independent Goal 7 work when its actual prerequisites are satisfied; do not classify work as board-blocked merely because it appears in Goal 7. Final calibration/test, NotInject, the bounded challenge and the single extension must not influence method development. Execute eligible quality evaluation only after the relevant methods, numerical computation and threshold/window policy are frozen under the original protocol. If freeze still depends on board-stage information, leave those evaluations explicitly deferred and finish their runnable preparation instead.

Keep the original challenge and one-extension limits. Do not create a new target-model benchmark. The completed 120-request application allowance is not renewed by this prompt. Prepare the held-out application runner and an explicit policy for truncations, insufficient evidence and task-reference ambiguities; preserve the earlier answers and judgments. Any genuinely required new target generation needs separate authorization and must be identified as an API-budget dependency, not a board dependency.

Generate plots/tables from real saved results. Update `paper/draft.md` and `reports/research_summary.md` with the actual H1 and B-to-C findings, the integrated architecture, and the evidence boundary. Keep board latency/energy and other unavailable results absent; do not create placeholder measurements or imply the paper is final.

Update the existing `STATUS.md`, execution log, method/cost reports and README at meaningful handoffs. Put the physical-board continuation procedure in an existing suitable report, or create a single `reports/board_handoff.md` if none fits. Avoid redundant progress files, dashboards, CI, hashes, integrity manifests, generic test suites and unrelated refactoring.

## 6. Resources, pauses and necessary checks

The initial 12-characterization allowance remains consumed. This continuation authorizes the missing offline full-system builds required for the existing U/B/C plan, necessary compilation, and cause-identified corrective rebuilds, including required generation for the single selected implementation mechanism. It does **not** renew the 12-run profile sweep or authorize open-ended architecture tuning. The coordinator records a bounded, purpose-specific build list before dispatch and reuses identical hardware designs. Respect available local compute and any stricter current execution limits.

Retain the current Codex usage policy across parent and workers: normally pause at 85% used, allow only a bounded task expected to finish within seven further percentage points, and stop by 92%; pause on a newly observed reset during this resumed run. Establish a fresh live starting observation when available; the historical 89% and reset timestamp are not current readings. Do not treat the passage from an old paused cycle into a newly authorized session as an instruction to pause immediately. Never invent usage readings. If usage visibility is unavailable, state that once, work in bounded checkpoints and do not assume unlimited allowance.

Record the board-free resumption and reconcile only the relevant old pause markers. Respect subsequent pauses and preserve resumable state; workers must stop dispatching new work when the coordinator pauses. Do not buy compute, extend paid/API budgets, change credentials or launch target generation.

Use only necessary checks: affected data/split checks, evaluator arithmetic, quantization/packing arithmetic, small real-path bring-up, relevant numerical/interface agreement and actual build/cost validation. Reuse completed checks. Documentation edits do not trigger training, synthesis or benchmarks. Fix routine problems autonomously; ask only for genuinely missing access/budget, physical action or a substantive scientific change. Continue independent work when one branch is blocked.

## 7. Completion and final response

Aim for **OFFLINE BOARD-READY**, meaning the required numerical qualification, complete offline cost model, bounded method results, integrated deployment artifacts, host code and later measurement drivers are ready. Final calibration/testing may remain deliberately staged behind physical qualification where the protocol requires it. Keep that planned deferral separate from unfinished implementation.

Do not label the result board-ready merely because an unconnected board is mentioned in the reports. If an essential non-board prerequisite remains unresolved, finish everything independent and report **PARTIAL — exact blocker**, with the working deliverables and smallest concrete resolution. No unsupported C result, fabricated cost, placeholder host integration or unqualified model approval may be used to satisfy a milestone.

Finish with:

1. Completed work and actual scientific conclusions, with evidence paths.
2. Checks/experiments/builds actually executed versus reused; actual models/subagents used; consumed and remaining relevant budgets.
3. A compact readiness table for numerical reference, estimator, A/B/C, integrated builds, host software and evaluation drivers.
4. Exact commands for offline reproduction and later physical-board continuation; identify which later commands were not executed.
5. Remaining physical actions/runtime inputs and board measurements; separately list any non-board access, budget, numerical or method blockers.

Leave the project ready for the next authorized physical stage, not silently running it. Begin with the focused state check, delegate independent work, and execute the first unblocked deliverable.
