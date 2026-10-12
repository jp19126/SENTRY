# SENTRY — FPGA-centered next-goal entry point

**2026-10-13 purpose correction.** Read AGENTS.md and STATUS.md before acting.
The user has reaffirmed that SENTRY is about **developing an original FPGA
architecture or algorithm–hardware co-design method for LLM security**.
FPGA is a mandatory part of the research, not an optional accelerator.
This clarification supersedes conflicting priorities in older run prompts.
Goal R1 is complete (see reports/goal_r1_decision.md); its prompt is archived at
reports/goal_r1_prompt.md and does not renew allowances.

## Fixed scientific purpose

Develop and evaluate an FPGA-based guard implementation or co-design method
with a differentiated hardware contribution. The security algorithm may be
established, including BERT-Mini or another justified detector. **Inventing a
new LLM-security algorithm is not required.** Novelty may lie in original FPGA
dataflow, scheduling, storage/streaming, numerical representation, precision
allocation, or security-constrained hardware–algorithm integration.

An off-the-shelf BERT-to-FPGA port and a generic CUDA comparison are not, by
themselves, this project's intended research contribution. CPU/GPU references
are deployment controls. The primary hardware claim needs relevant FPGA prior
art and an equally engineered FPGA comparison. Do not claim an automatic FPGA
win, but do not abandon FPGA because one implementation loses to CUDA.

The current pilot concerns indirect prompt injection in external documents.
Do not assert protection against general training/model poisoning without a
separate threat-model decision. Detector scores alone do not demonstrate
end-to-end application protection or retained legitimate usefulness.

No specific new FPGA hypothesis, experiment, threat expansion, or budget was
authorized by the documentation correction.

## Current task: select a specific FPGA contribution

1. Reuse STATUS.md, reports/goal_r1_decision.md, prior hardware evidence,
   reports/protocol.md and relevant existing results. Inspect focused FPGA
   accelerator/co-design literature and guard prior art where needed. R1's
   1.707-ms CUDA Graph timing is a measured control, not the objective or an
   automatic reason to terminate FPGA research.

2. Propose **at most two FPGA-centered, falsifiable research questions** and
   recommend one. For each, explain the security task, the proposed concrete
   architecture/co-design change, what FPGA-specific problem it solves, the
   nearest known hardware method and the difference to be tested. Existing
   detectors are allowed; a new security algorithm must not be invented solely
   to satisfy a mistaken novelty requirement.

3. Give a serious same-security-requirement FPGA baseline with common
   implementation improvements, a separate practical software comparator,
   measurable protection/utility requirements and complete checking-cost
   boundaries. State what would falsify the contribution. Examples such as
   security-aware mixed precision, streaming or data placement are **candidate
   ideas**, not accepted novel mechanisms.

4. Identify the smallest bounded experiment, using existing evidence or
   board-free HLS/RTL/AIE design where it can discriminate the hypothesis.
   State tools, budget, measurements, physical-board dependencies and the
   meaning of positive, negative or inconclusive results. No generic FPGA
   port merely to complete an obsolete milestone.

5. Submit the recommended specific scientific question and bounded experiment
   for user approval. Research planning and focused source inspection are
   permitted here; this file does **not** authorize a new training/scoring
   campaign, HLS/AIE build, simulation, target-LLM call or board programming.

## After approval

Record the approved research contract and budget in STATUS.md. Then execute
routine authorized work autonomously: software/security reference, original
FPGA design, numerical qualification, synth/simulation, matched architecture
comparison and actual board evaluation once access and permission exist.

Hardware novelty and deployment competitiveness are separate conclusions.
Assess equal-quality FPGA-to-FPGA differences for the former, and competent
CPU/GPU references for the latter; FPGA is not required to beat GPU in every
scenario. Measure guard quality and legitimate-task impact as required, and
report timing/throughput, utilization, startup, transfers and full cost where
actually supported. Never label simulation a physical measurement.

A failed FPGA route is a legitimate negative result, not permission for an
automatic software-only pivot or unbounded attempts to secure a win. Scientific
scope, primary objective, threat, or FPGA centrality changes require explicit
approval. Preserve R1 evidence, frozen datasets, security thresholds, numerical
identities, used allowances and board/API restrictions. Do not automatically
resume the older A/B/C campaign or Goal 4/5 simply because it is unfinished.

Keep execution lightweight: use AGENTS.md, STATUS.md and
reports/execution_log.md. No new audits, review machinery, CI or repeated
environment scans. Delegate routine work under the same FPGA research question.
