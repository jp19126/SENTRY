# Status

Updated: **2026-10-13 — FPGA-centrality correction; documentation only.**

**FPGA-CENTERED QUESTION TO SELECT.** The 2026-10-13 user clarification
reaffirms that SENTRY is FPGA architecture/co-design research for LLM security,
not a search for a new software-only guard. This updates purpose, not experiment
authorization. [AGENTS.md](AGENTS.md) governs; the current entry point is
[SENTRY_next_goal_prompt.md](SENTRY_next_goal_prompt.md).

## Research contract

- **Question and authority:** FPGA-centered protection is fixed by the user's
  2026-10-13 clarification. A specific new FPGA hypothesis/experiment is
  **unselected and not authorized**.
- **Closest prior approach:** the original Quasar-derived B and R1 guards are
  prior security/method references. Closest FPGA hardware/co-design prior art
  depends on the proposed design; a new scientific gap is not yet established.
- **Hypothesis/difference:** unselected. Security-constrained mixed precision,
  FPGA placement or streaming architecture are candidates, not adopted novelty.
  No new LLM-security algorithm is required.
- **Scientific outcome:** a differentiated FPGA design/co-design contribution
  that preserves an appropriate protection/utility requirement and accounts for
  hardware resources, full costs and security claim limits. A software result
  or routine port alone cannot complete the research.
- **Hardware role:** FPGA implementation is required. Match an equally engineered
  FPGA baseline and use CPU/GPU implementations as practical comparators, not
  replacement objectives. Actual board claims remain blocked by physical access.
- **Next discriminator:** propose at most two FPGA-specific questions using saved
  R1/hardware evidence and focused prior art, then seek approval for a bounded
  hardware-centered test before substantial execution.

## Completed evidence to reuse

[Goal R1](reports/goal_r1_decision.md) completed its bounded decision study.
Its measurements and source remain unchanged:

| Finding | Evidence boundary |
|---|---|
| Feature classifier: TP/FN/FP/TN = 2639/99/20/1349 | Fails the existing recall and FPR screen; one recipe. |
| Compact CNN: 2738/0/24/1345 | Passes recall but fails FPR; not a general CNN rejection. |
| Native W8-QAT BERT/CUDA Graph: 2738/0/13/1356; warm mean 1.707225 ms | Development screen and 32 matched single-window checks, not a scientific contribution or final deployment result. |
| BF16-attention reference: 2738/0/13/1356 after its own calibration | Changed arithmetic/scores; no AIE equivalence or accelerated BF16 timing claim. |
| R1 hardware work | Two mappings analysed; zero AIE/HLS prototypes. A redesigned accelerator remains untested. |

All 55 development attack templates were seen in training. Final calibration,
final test and NotInject remain closed. R1 added no downstream target-LLM study;
its detector results do not establish utility-preserving protection.

The legacy serial PL route remains stopped. Earlier board-free implementation
and numerical evidence is preserved, not rerun or discarded. The original
Goal 5 campaign, board work and further target-LLM calls remain paused.

## What is no longer the automatic next step

R1's recommended encoder-block experiment is a **historical recommendation**,
not the current research priority or execution authorization. It can be proposed
again only as a bounded test of an explicit scientific question. Do not rerun
R1, automatically finish Goal 4, optimise CUDA further, or start a hardware
redesign merely to fill a milestone or beat a baseline latency.

The next agent should follow the FPGA-centered entry point and return a
specific hardware/co-design scientific decision for user approval. This change approves no new budgets,
models, threat scopes, methods or experiments and invents no current usage reading.

## History and reproduction

The complete previous STATUS.md is preserved verbatim in
[the pre-correction handoff](reports/status_before_purpose_lock_20261011.md).
The completed R1 prompt is preserved verbatim in
[the R1 prompt archive](reports/goal_r1_prompt.md). These copies come from commit
`6c06973e5a568ad484ec0e9b779876bca1ec9626`; their old next-step commands are
historical, not instructions to resume work.

Use reports/goal_r1_decision.md and reports/execution_log.md for measurements,
source paths, exact numerical scopes and historical allowances. No experimental
code, configuration, data, checkpoint or result was changed by this correction.
