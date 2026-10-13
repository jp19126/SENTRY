# SENTRY — Method-first FPGA research entry point

**2026-10-13 strengthened purpose clarification.** Read AGENTS.md and STATUS.md
first. SENTRY's scientific goal is **a substantive new FPGA-aware LLM-security
guard algorithm or algorithm–hardware co-design method**, implemented and tested
on FPGA. The goal is not just to deploy an established successful CPU/GPU
detector such as BERT on another platform. This supersedes contradictory
priorities in old prompts and handoffs. R1 remains historical, with executed
evidence in reports/goal_r1_decision.md and its old prompt archived in
reports/goal_r1_prompt.md. No previously used budget is renewed.

## Fixed scope: the new method must precede the implementation campaign

The proposed contribution must change the guard's computation, representation,
decision procedure, or method for jointly selecting algorithm and FPGA hardware.
A completely new pretrained language model or new theory of prompt injection is
NOT required. Existing BERT, CNN and other guards may serve as baselines,
teachers or components; their unchanged FPGA implementation is not the
scientific contribution. Inventing a standalone accelerator for an unchanged
detector is not the current approved direction and would require a user decision.

Hardware innovation matters only when tied to a genuinely differentiated method
with a falsifiable security–hardware consequence. Generic mixed precision,
pruning, dataflow scheduling, off-the-shelf model conversion, CUDA comparison,
buffering and AIE-ML deployment are not inherently new. A transformer and AIE-ML
can play central roles **if the proposed method justifies them**, but neither
technology is a predetermined final solution.

The pilot studies indirect prompt injection in attacker-controlled external
documents, not demonstrated general model/training poisoning. Classifier labels
do not prove utility-preserving LLM protection. Lower detection quality may
be a legitimate co-design trade-off only under prospectively defined bounds on
missed attacks, false positives and legitimate-task utility, justified by
measured hardware benefit; do not tune the bounds to make a candidate win.

## Current task: select the method, not the next porting milestone

1. Use existing STATUS.md, R1 findings, relevant hardware and numerical
   evidence, reports/protocol.md and focused relevant prior art. Identify where
   existing guard computation and FPGA execution constraints create a specific
   unsolved security–hardware trade-off. Keep R1's 1.707-ms CUDA Graph result as
   a useful measured comparator, not the project objective or a reason to stop
   FPGA research. Do not automatically restart unfinished Goal 4/5 or A/B/C.

2. Propose **at most two method-level hypotheses**, recommending one. For each,
   write in plain terms: "We change X in the guard's computation or joint
   algorithm–hardware procedure to exploit FPGA property Y, while controlling
   security consequence Z." Specify the mechanism, its nearest algorithmic AND
   FPGA/co-design prior art, and what observation would refute novelty or
   practical advantage. Distinguish a real method from a renamed standard
   quantizer, smaller unchanged model or ordinary accelerator schedule.

3. Propose matched comparisons that isolate the contribution: an established
   guard on competent software and conventional FPGA implementations; the
   proposed guard on software where feasible; and an ablation/conventional FPGA
   mapping of the proposed method. Comparison need not demand identical model
   accuracy, but must apply prospectively fixed protection and legitimate-use
   constraints, account for complete checking cost and attribute any gain to
   algorithm change, hardware realization or their interaction.

4. Identify the **smallest bounded discriminating experiment** and name its
   endpoint, tools, compute allowance, metrics, positive/negative/inconclusive
   interpretations and physical-board dependence. A limited baseline port,
   CUDA profile, BERT/AIE-ML kernel or HLS bring-up can be a prerequisite only
   if it answers a named scientific question and has a stopping point. Do not
   let its implementation tasks grow into the main research campaign.

5. Present the recommended new method and experiment for user approval. This
   handoff permits only reasoning, evidence review and focused prior-art checks;
   it does NOT authorize a new model-training/scoring campaign, HLS/AIE build,
   simulation, physical-board programming or target-LLM API call. No method
   proposal should be represented as already approved or already novel.

## After explicit approval: test the method, then implement as justified

Record the approved method hypothesis and budget in STATUS.md. Execute
authorized experiments autonomously, beginning with the smallest useful
algorithm/security and FPGA discriminator, then proceeding to end-to-end FPGA
or heterogeneous AIE-ML/PL implementation as results warrant. Do not require a
guaranteed positive result before an exploratory test.

Measure detector security and legitimate-task utility, algorithmic and numerical
quality, total checking latency/throughput, transfers, memory/resources and
other actually supported costs. Separate the contribution of the new method
from ordinary porting or optimization, and distinguish board evidence from
simulation/synthesis. Neither a faster unchanged model nor a GPU-only guard can
finish SENTRY.

Negative evidence rules out the unsupported method as claimed, not the
FPGA-centered scientific purpose. No automatic software-only, unchanged-model
accelerator or new attack-scope pivot without user approval. Preserve completed
evidence, frozen partitions, numerical identities, budget limits and board/API
permissions. Work within existing AGENTS.md, STATUS.md and reports/execution_log.md;
no extra CI, audit frameworks or review databases.
