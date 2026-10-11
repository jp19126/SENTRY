# SENTRY — Decision-first detector and hardware study (Goal R1)

Execute this bounded research goal in the existing jp19126/SENTRY checkout,
not the separate historical GATE checkout.
The objective is to decide which detector and hardware architecture deserve the
next substantial investment, using small executed experiments—not to finish the
old implementation at any cost. A well-supported stop or pivot is a successful
outcome. A positive outcome authorizes only a recommended next goal, not its
automatic execution.

## 1. Establish the starting point and scope

Read README.md, AGENTS.md and STATUS.md first. Then consult the relevant portions
of CODEX_RESEARCH_GOALS_v2_1.md, configs/project.json, reports/protocol.md,
reports/quantization_sensitivity.md, reports/software_profile.md,
reports/hardware_cost_model.md, reports/goal4_symbolic_cost.md,
reports/goal4_numerical_bridge.md and reports/baseline_selection.md.
Inspect additional source or evidence only when it affects a decision.

Use the actual local checkout and saved artifacts as the authority. Do not assume
that every result mentioned in a report is available in a fresh clone. Preserve
existing results and historical arithmetic identities. Do not repeat completed
training, synthesis, RTL cases or application requests merely to reconfirm them.

This prompt authorizes a limited detector challenge and alternative-architecture
study, superseding the previous instruction to prioritize completing the old
Goal 4 path. It does not abandon the original project, change its canonical
primary model, declare old milestones complete, or authorize a full redesign.
Resume only the work covered here; unrelated research remains paused.

No board programming, boot changes, physical measurements, new target-LLM API
calls, paid services, full Goal 5 campaign, final evaluation or full-detector
hardware implementation. Do not turn this into another literature survey.

## 2. Questions to answer and scientific rules

Answer these three questions in order:
1. Is BERT-Mini necessary at our protection requirement, or can a substantially
   simpler detector do the job on the available development evidence?
2. For the selected candidate, is there a credible complete-checking advantage
   from FPGA/Versal execution over running that same candidate well on CPU/GPU?
3. Is there a specific research question beyond implementing known acceleration
   techniques that makes the next investment worthwhile?

Do not assume either that FPGA is unsuitable or that it must win. Treat an
AIE-ML–PL architecture as a serious preferred candidate when it fits the selected
model, not as a compulsory component. Model suitability comes before hardware
preference. Keep BERT available as a reference or teacher even if a simpler
candidate becomes the provisional deployment choice.

Preserve source-group separation, complete document coverage, the shared
window/max-aggregation policy and strict score > threshold decision rule.
Fit each candidate's threshold only on the existing search-threshold partition;
evaluate on the separate search-scoring partition. Keep the 1% empirical FPR
requirement and the recall floor of the fixed BERT floating reference minus
one percentage point. Do not redefine the floor from a weaker new model.
Final calibration, final test and NotInject remain closed.

Report raw TP/FN/FP/TN counts and available source/length strata, not only accuracy.
A one-example difference near the FPR boundary is not conclusive evidence of a
superior model. Passing this development screen is not general robustness.
Use available development-only attack-family or difficult-benign subsets for
interpretation; disclose absent coverage without building a new benchmark.

Software model comparison does not require completing the old RTL reference.
Reuse old scores only when their checkpoint, preprocessing and actual execution
semantics match. New hardware arithmetic needs its own reference and matching
scores; never attach a new arithmetic label to old results.

## 3. Bounded execution and delegation

Use these ceilings, not targets to exhaust:
- Reuse the BERT anchors. Add one cheap feature-based classifier and one compact
  1D text CNN. No model zoo or automated architecture/hyperparameter search.
- One initial fitting/training recipe and seed per challenger. Permit at most one
  additional learning run in the whole goal, for a diagnosed training problem,
  a justified distillation attempt, or necessary quantization adaptation—not all
  three. Fix finite training limits before launching; use training-only
  validation or a fixed schedule, not repeated search-score tuning.
- Compare at most two hardware mappings analytically. Prototype only one route.
  Allow at most two distinct prototype configurations, each with at most one
  cause-identified corrective rebuild. Necessary AIE/HLS steps form a recorded
  build pipeline; do not evade the ceiling through renamed scripts or runs.
- Start at L256, batch one. Reuse the old matched-input timing set where possible.
  No full workload grid, 10,000-check campaign or whole-device place-and-route.

These are new, scoped prototype allowances, not a renewal of the exhausted
12-synthesis characterization budget. Existing stricter resource/usage limits
still apply. If the work cannot fit, report the missing experiment instead of
silently increasing the allowance.

Use one coordinator and at most two concurrent subagents when independent work
justifies them. Delegate software challenger work to one and architecture/cost
analysis to the other. Use an available lower-cost agent such as the configured
luna6 for focused coding or evidence extraction; keep numerical and research
judgments with a capable coordinator/worker. Inspect available aliases once;
do not invent model names or install an agent framework. Give agents narrow
context and non-overlapping files. No nested delegation or competing heavy jobs.
Only the coordinator dispatches expensive training/builds after the checkpoints.

Preserve the current usage policy: normally stop at 85%; a bounded task may finish
within the existing allowance of at most seven further percentage points, never
past 92%. Stop on a newly observed usage reset during this run. Establish the
current reading rather than treating the historical 89% as current. Check at
natural checkpoints and before expensive dispatches, not continuously. If usage
telemetry is unavailable, disclose that and enforce the explicit job ceilings.

## 4. Stage A — Test the detector assumption before redesigning hardware

Implement and run two small challengers using the existing evaluation machinery:
- A cheap text-feature baseline, such as character/word n-grams with a linear
  classifier or small MLP. Choose one implementation, not several variants.
- A compact 1D CNN with inexpensive token embeddings, suitable local convolutions
  and pooling. It must not require BERT to produce features at deployment.

Choose sensible training settings from existing implementations or a narrowly
relevant source. A CNN trained from scratch and a pretrained Transformer have
different starting advantages: disclose this. Do not declare CNNs intrinsically
inadequate because one obviously undertrained or broken run fails. Inspect the
learning behaviour once; use the single extra learning allowance only when there
is a concrete reason. Distillation may use training data and the existing teacher,
but the deployed student's timing must include every component it actually needs.

Compare against the saved BERT floating and W8-QAT anchors where legitimately
reusable. Score new candidates on the full existing development partitions, not
just a tiny convenient subset. Keep training and model-selection costs visible.

Run short, warmed-up CPU/GPU timing pilots for viable candidates. Include feature
extraction/tokenization, inference, aggregation and applicable transfers. Do not
compare a CNN's FPGA estimate only against BERT's software time. Use genuine
accelerated software kernels where available; fake-quantization or intentionally
ordered reference timing is not an optimized INT8/INT4 benchmark.

CHECKPOINT 1 — Select a provisional detector direction.
Prefer a simpler candidate when protection and cost evidence support it; retain
BERT when the added capability appears necessary. Near-threshold or inadequately
trained results can be inconclusive. Neither one failed CNN nor one successful
benchmark settles the model family generally. If no candidate supports further
hardware work, stop with the exact quality/data issue. Do not manufacture a winner.

## 5. Stage B — Select the execution architecture and prospective benefit

For the provisional detector, compare at most two appropriate mappings:
- A locally buffered streaming/tiled PL design.
- An AIE-ML–PL design using AIE-ML for suitable matrix/convolution operations and
  PL for the dataflow, buffering and selected surrounding arithmetic.

For a retained Transformer, assess activation reuse, resident or prefetched
weights, tiled QK–softmax–AV execution, feed-forward stage connections and reuse
of appropriately sized engines across layers. Do not assume a fully replicated
network minimizes batch-one latency. For a small CNN, assess native convolution
streaming before adding an unnecessary heterogeneous interface.

Check only decision-critical published precedents and official device/tool
support. Relevant starting leads include SSR, EQ-ViT, TATAA and spatial BERT
acceleration; hls4ml/FINN/da4ml may be more relevant for a simpler model. Inspect
at most a few sources needed to choose the mapping. Verify batch, shape, device
and measurement boundary before using a published result. Prior conversation
claims are leads, not measurements for this project.

Verify actual AIE-ML generation, operand formats, accumulation, packing and kernel
availability. AIE-ML is not interchangeable with original AIE or AIE-ML v2. Do not
assume FP32 or stored W4 automatically maps to efficient native execution.
Begin with a credible W8 path; evaluate BF16 or other changes only where they
address a demonstrated bottleneck and can be numerically qualified.

Create a small, explicit cost worksheet: arithmetic workload and useful capacity,
local-memory placement, required external traffic, layout/conversion costs,
nonlinear work, startup, synchronization and host-to-decision costs. Distinguish
arithmetic floors, simulated observations, analytical scenarios and unknowns.
Peak TOPS or total on-chip memory alone is not an implementation feasibility proof.
Account for valid dependencies; overlapping stages cannot simply have all their
costs deleted. Do not double-count interface waits already present in RTL evidence.

Before observing new hardware results, state the primary practical benefit and
what evidence would make it worth pursuing. Default to complete batch-one
checking latency against the strongest available protection-matched software
path for the same model. A meaningful latency/energy/throughput trade-off may be
selected prospectively, but unmeasured energy or tail latency cannot rescue a
failed latency result after the fact. Do not invent favorable fixed latency caps.

Identify one plausible, falsifiable research question and its closest baseline.
Known buffering, streaming, AIE adoption, lower precision and model distillation
are not automatically novel. A quality-matched baseline must receive common
engineering improvements. C can remain unselected; do not invent novelty to pass
this checkpoint, and do not demand a completed novelty proof before a small pilot.

CHECKPOINT 2 — Is a targeted prototype worth its cost?
Continue only when the analysis identifies a credible benefit, a plausible
research question and a specific uncertainty that the pilot can resolve. If even
an optimistic arithmetic/data-movement floor rules out the declared objective,
reject that mapping without building it. A favorable floor alone is not enough.
If assumptions dominate, choose the smallest discriminating experiment or pause
inconclusively; do not launch a full system build to find out.

## 6. Stage C — Execute one decision-relevant hardware pilot

Implement, compile and simulate the selected route within the build ceiling.
Reuse vendor/library kernels and existing qualified components when appropriate.
Do not build a general accelerator framework or complete both alternative routes.

First establish the selected arithmetic in software on existing real inputs.
For a format-changing candidate, score its actual proposed arithmetic on the
development partitions before describing it as protection-feasible. Keep exact
integer/code checks where the contract is exact. Define justified floating or
approximation checks independently of the errors observed; never widen a tolerance
to make a failure disappear. Preserve failures and earlier reference results.

The pilot must exercise the claimed bottleneck and realistic data movement:
for example, a buffered linear-plus-surrounding-operations chain, an attention
chain, or a representative CNN block. Use actual model dimensions and realistic
producer/consumer interfaces, not an isolated ideal MAC loop. A complete encoder
block is preferable when it fits the allowance; a smaller chain is acceptable
only with its remaining uncertainty stated. Do not label it full-detector proof.

Record useful compute rate, stalls or transfer costs available from the tools,
local memory/resources and numerical agreement. Refresh the complete-cost
scenario with this evidence. Keep remaining host, physical DDR/NoC and integrated
system effects explicit. An AIE simulation is not board latency or energy.

CHECKPOINT 3 — Recommend the next investment, then stop this goal.
Recommend proceeding only when protection, architecture and contribution evidence
justify one named next experiment or implementation stage. Recommend stopping the
current route when the evidence argues against its value. If a single missing
measurement could reverse the decision, report INCONCLUSIVE/BLOCKED and name it.
Do not automatically start Goal 5, full-detector integration or board work.

## 7. Lightweight error correction and early termination

At the three checkpoints, write a short decision: what was expected, what was
observed, the implication, and the next permitted action. The coordinator owns
this judgment; another existing agent may challenge the reasoning briefly.
No recurring review committees or review after every edit.

Distinguish:
- IMPLEMENTATION ERROR: fix the identified cause and rerun only affected work,
  within the existing allowance. A broken run is not a negative scientific result.
- NEGATIVE EVIDENCE: abandon that candidate or mapping. Allow one local pivot
  across the whole goal only if it has a stated reason to work and fits the
  remaining ceilings; do not restart the search under a new name.
- INSUFFICIENT EVIDENCE / MISSING ACCESS: state the unresolved fact and pause the
  dependent work. Missing hardware/tool access does not show FPGA is unsuitable.

Selecting among the initially planned candidates is not a pivot; changing a
committed direction after negative evidence is. The local pivot is not an extra
budget. Do not keep changing thresholds, seeds,
models, precision or performance objectives until something wins. Stop as soon
as the decision is supported; completing every stage is not mandatory. On a stop,
cancel pending work and safely stop/checkpoint only jobs owned by this goal.
Keep the evidence and a concise resumption point.

Use only essential checks: relevant split/evaluator checks when changed, one small
real-input bring-up per new path, arithmetic checks for changed operations, and
comparisons required to support the research decision. Do not add broad CI,
coverage/fuzzing suites, custom hashes, audit frameworks, repeated environment
scans, dashboards or tests unrelated to a concrete issue.

## 8. Deliverables

Keep documentation small. Use one report, reports/goal_r1_decision.md, with:
- The bounded plan/ceilings and three checkpoint decisions, as far as reached.
- Model quality and complete software timing comparisons with raw counts.
- The chosen/rejected mappings, cost breakdown and any actual pilot evidence.
- What is reusable, what remains unmeasured, and the contribution hypothesis.
- A final GO / STOP-CURRENT-ROUTE / INCONCLUSIVE-BLOCKED recommendation and the
  single most useful next action, including prerequisites and acceptance criteria.

Save necessary configurations, scores, measurements, logs and reproducible code
under the existing repository conventions, using a results/goal_r1/ directory
only as needed. Update STATUS.md and append substantive execution facts to
reports/execution_log.md. Briefly record the scoped override in the goals document;
do not rewrite completed history or relabel the old milestones as complete.

Finish with a direct researcher-facing summary: what you actually executed, what
changed our understanding, whether further FPGA investment is justified, which
model/architecture is recommended, and why the next expense is or is not warranted.
Do not return only a plan or unexecuted scripts. A supported negative or explicitly
inconclusive result is preferable to an expensive, unjustified continuation.
