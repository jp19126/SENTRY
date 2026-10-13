# SENTRY: method discovery to board-free research execution

Work in the existing SENTRY checkout. Carry out this as one continuous research workflow, with one planned scientific approval checkpoint before adopting a method and starting substantial experiments. After that approval, execute the agreed work rather than returning another plan at each phase.

## 1. Scientific purpose

Develop and test a substantive new FPGA-aware guard algorithm or algorithm–hardware co-design method for protecting LLM applications against malicious influence in external documents. FPGA must be central to the proposed method and its implementation. The current application is indirect prompt-injection detection; do not silently broaden the claims to training/model poisoning or general misinformation.

Existing BERT, other transformers, CNNs and guard models may be baselines, teachers or reusable components. Implementing an unchanged existing detector on FPGA, replacing BERT with another off-the-shelf model, or moving transformer kernels to AIE-ML is not the contribution. Conversely, a completely new foundation model or unrelated theory of LLM security is not required. Identify an actual methodological change in computation, representation, decision-making or joint algorithm–hardware design.

The objective is a better useful security–hardware trade-off, not necessarily the highest standalone accuracy. Bounded quality degradation may be acceptable when justified by substantial hardware benefits. Specify protection and legitimate-use limits before selecting results. Do not invent novelty, force an FPGA win, or substitute a software-only defense when an FPGA hypothesis fails.

## 2. Start from the real project; keep execution lightweight

Read README.md, AGENTS.md, RESEARCH_SCOPE.md and STATUS.md first. Then consult reports/protocol.md, reports/goal_r1_decision.md, configs/project.json and only the evidence/source needed for this task. Inspect the actual checkout, local changes and required saved artifacts once. Distinguish committed reports from locally available raw results and builds. A Git clone does not imply that ignored checkpoints or data are present.

Reuse valid work. Do not resume old Goal 4/5, R1, A/B/C searches or the legacy BERT hardware path merely because they are unfinished. Preserve their evidence, configurations and numerical identities. Keep research-specific tasks and decisions in the study report and STATUS.md, not embedded as permanent goals in AGENTS.md.

Use existing environments and vendor tools. Verify relevant access before dispatch, not repeatedly. Preserve memory/resource limits and other active permissions. No physical-board operation, paid service, new target-LLM API call or opening of protected final calibration/test/NotInject partitions is authorized by this prompt.

Use at most two concurrent subagents when available and useful: one for algorithm/security/prior-art work and one for FPGA/AIE-ML architecture and implementation. Select actually configured agents; do not invent model names or capabilities. Give them independent tasks and non-overlapping file ownership. The coordinator owns the scientific question and dispatches resource-intensive jobs serially on shared hardware. Reuse agents for implementation or focused review; no nested review teams. Work sequentially if delegation is unavailable.

Run only checks tied to changed evaluator logic, arithmetic, interfaces and small real-input bring-up. Do not add CI, approval databases, integrity frameworks, unrelated tests or repeated environment audits.

## 3. Phase A: identify a concrete method and propose its decisive experiment

This phase authorizes focused literature research, source inspection, saved-result analysis and lightweight analytical calculations. It does not authorize new training, inference campaigns, vendor builds or simulations.

First establish what the existing evidence actually says. Separate detector limitations, numerical differences, implementation bottlenecks and missing measurements. Do not infer that a transformer is necessary from a small number of failed compact-model recipes. Do not turn the saved CUDA timing into the scientific objective or a universal FPGA feasibility threshold.

Verify the nearest relevant literature as of the execution date. Prioritize journal articles and early-access publications, but include directly relevant conference papers and preprints with their status clearly labeled. Treat prior chat literature reviews as search leads, not verified evidence. Check bibliographic identity, publication dates, accessible method details and measurement boundaries. An abstract-only source cannot support detailed claims about an uninspected method. A search that finds no match is not proof of novelty.

Cover the closest work on guard detection and overdefense, generalization across attack distributions, hardware-aware compression/co-design, and the particular FPGA/AIE computation being proposed. Include alternatives that avoid redundant guard computation when relevant to the deployment. Do not attempt to reproduce every paper or expand into a general survey.

Propose at most two concrete methodological candidates and recommend one. For each, specify:

- The security limitation and FPGA constraint being addressed, with supporting evidence.
- The mechanism in sufficient detail to implement: algorithm steps or pseudocode, learned/static quantities, inputs/outputs and its difference from the nearest existing method.
- Why this changes the security–hardware trade-off rather than merely renaming standard quantization, pruning, distillation, loss reweighting or accelerator scheduling.
- A preliminary computation, memory and communication model; intended PL/AIE-ML partition if relevant; and the condition under which the FPGA realization adds value.
- The smallest experiment that distinguishes the claim, an essential ablation, and results that would support, contradict or leave the hypothesis unresolved.

Do not select BERT, a non-transformer or AIE-ML in advance. Retain existing components when useful. For a transformer candidate, jointly consider native arithmetic, nonlinear operations, activation/weight storage and AIE–PL transfers. Verify device-specific capabilities; fewer bits, parameters or operations alone do not establish lower latency.

Write one recommended experiment contract containing the method, scope, comparators, data protocol, primary hardware metric/workload, quality limits and finite execution budget. Use the current workload and protection limits as defaults; explicitly propose any changes before observing candidate outcomes. Distinguish absolute percentage-point quality losses from relative changes in missed-attack rates.

Separate a pilot allowance from a conditional integration allowance, so the user can approve the whole board-free workflow once, with progression tied to stated pilot results. The budget must cover necessary baseline training/adaptation, the candidate, its central ablation, numerical checks, representative hardware builds and informative repeats. Specify run/build counts and compute limits appropriate to available resources. Include a bounded correction allowance for diagnosed implementation failures, not unlimited model or parameter search. Do not charge fresh work against an exhausted historical allowance.

Preserve protected final partitions. Explain how development-only hard-benign and generalization probes will be obtained without leaking them. Track source groups and attack-template exposure, including teacher/checkpoint exposure; relabeling an existing split cannot make it unseen. State limitations when suitable independent data are unavailable.

Save the proposal and verified related-work comparison in reports/method_study.md, reusing that file if present. Keep a short pointer and decision in STATUS.md. Return the recommended method, its strongest competing explanation, the experiment contract and the specific approval requested. Pause here for the user's selection and budget approval. Do not claim that drafting the proposal completes the research.

## 4. Phase B: execute a coupled algorithm-and-hardware feasibility study

After the user explicitly approves the selected method and experiment contract, record the approval and execute that study autonomously. The approval covers the named method and allowance, not a generic license to change the research direction. Missing access should block only dependent work.

Implement the proposed computation and its software reference, the nearest appropriate baseline/adaptation, and the essential ablation. Use one study-specific configuration where needed; preserve frozen historical configurations. Reuse compatible checkpoints and results rather than retraining by default. A new co-design procedure does not require retraining unless its hypothesis requires it.

Make training, pretraining/distillation access, data, search effort and threshold calibration comparable. A teacher-trained student must not receive undeclared information advantages. Fit operating thresholds only on the designated development-calibration subset; assess candidates on the separate scoring subset. Keep comparison arms named descriptively; do not silently redefine the historical A/B/C labels.

Run the approved experiments and inspect real results. Report recall and false positives at the declared operating points, relevant attack/benign strata and uncertainty respecting shared source groups. Inspect changed decisions and legitimate-task implications, not overall accuracy alone. Report development generalization honestly and distinguish it from a final held-out claim.

In parallel with algorithm development, develop the hardware cost model and representative design. Once the proposed arithmetic is sufficiently defined, test the smallest complete computational region that distinguishes the method from its baseline. Do not defer hardware evidence until after a large software search.

Choose that region from the hypothesis: it might be a complete encoder block, aggregation stage or another method-specific chain. Include the buffering, conversions, nonlinear operations and producer–consumer transfers necessary for that chain. An isolated matrix kernel or softmax benchmark is insufficient when the claimed advantage depends on surrounding work.

Compile, simulate and synthesize the approved representative designs with the installed tools. Compare against a competent conventional implementation at comparable constraints. Reuse valid baseline components; do not start a second large generic accelerator project merely to create a comparator. Clearly label any unimplemented baseline or analytically projected cost.

Validate the new hardware against its own intended algorithm/numerical reference, not against compulsory bitwise identity with historical BERT. Set relevant numerical tolerances before evaluating discrepancies. Record meaningful quality effects of changed arithmetic rather than masking them with retuned tolerances or thresholds.

Measure useful cycles/rates, storage, transfers, stalls and resource requirements where the tools support them. Construct a dependency-aware full-check cost estimate with unresolved terms visible. Include tokenization/windowing, model parameters/embeddings, data transport and decision costs when applicable. Separate startup from steady-state cost. Do not sum overlapping costs twice or equate peak arithmetic throughput with achieved end-to-end performance.

Make two distinct judgments: does the method appear differentiated and supported, and does the FPGA realization offer a credible useful trade-off? Use the approved criteria to continue or stop. A negative or inconclusive result is legitimate. Do not automatically switch models, metrics or research questions until something wins. Reconsider the method with the user only when a substantive change is needed; routine fixes remain autonomous.

## 5. Phase C: complete justified board-free work and prepare the board handoff

If Phase B meets its declared continuation criteria and the approved allowance includes integration, continue without another planning-only handoff. Complete the board-free implementation and comparison needed for the selected contribution, not every historical milestone.

Integrate the method's FPGA or heterogeneous AIE-ML/PL path with model export, required host interfaces and numerical evaluation. Run supported simulation and implementation stages within budget. Use meaningful raw-document-to-decision checks where available; identify modeled or unvalidated host/transport components explicitly. Report achieved synthesis/routed timing and resources only when actual reports exist.

Isolate algorithmic and hardware effects using the agreed comparisons: established versus proposed computation on competent software; proposed computation on conventional versus co-designed FPGA mappings; and the central ablation. Compare complete deployment costs where evidence supports them. Do not claim FPGA value solely from comparing a smaller new model on FPGA against a larger old model on GPU.

Complete approved development evaluation, robustness probes and sensitivity checks without repeatedly tuning to the evaluation set. Keep protected final data closed. Prepare final-evaluation entry points, but do not execute them without later explicit authorization. Reuse saved target-LLM responses only where the inputs/actions match; do not invent downstream outcomes for new blocked, modified or sanitized inputs. Mark new target-model studies unresolved when calls are not authorized.

Prepare a concise board handoff: selected method/checkpoint and formats, target/platform/tool versions, actual build status, host interface, reproduction commands, initial real-input check and remaining physical measurements. Commands must correspond to existing artifacts and implemented scripts. Missing tool access or failed builds remain explicit blockers, not placeholders presented as deployable outputs.

Do not program a board or claim measured energy, physical throughput, tail latency or complete on-board correctness from simulation. Physical deployment is outside this prompt.

## 6. Reporting, completion and Git

Keep STATUS.md short. Use reports/method_study.md for the hypothesis, verified prior art, approved contract, findings and decisions; reports/execution_log.md for detailed attempts; and reports/board_handoff.md for deployment readiness. Use ordinary study/run identifiers and fresh result directories. Update Markdown handoffs with the repository's existing safe-update helper. Avoid duplicate planning documents or a new management framework.

At meaningful checkpoints, report: what was predicted, what was observed, what the result says about the new method, what remains unmeasured and the next authorized action. Distinguish method evidence, supporting implementation and security/application claims.

Finish this workflow with one of: a supported method and verified board-free handoff; a completed negative study explaining which hypothesis failed; or a precise evidence/access blocker with independent authorized work completed. Source code, a literature report, passing tests or a successful unchanged-model port alone is not research completion. Board-free completion is not final experimental validation.

Keep modifications local. Preserve unrelated changes and do not automatically pull, reset, push, open a PR or merge. Present the changed files, completed evidence, unresolved claims and Git diff summary, and obtain user confirmation before GitHub synchronization.
