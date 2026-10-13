# Status

Updated: **2026-10-13 — FPGA-aware methodological contribution clarified; documentation only.**

**METHOD-LEVEL RESEARCH QUESTION STILL TO SELECT.** The user reaffirmed on
2026-10-13 that SENTRY must contribute a **new FPGA-aware guard algorithm or
algorithm–hardware co-design method**, not merely implement established BERT
or another successful CPU/GPU detector on FPGA. The strengthened
[AGENTS.md](AGENTS.md) is binding; [SENTRY_next_goal_prompt.md](SENTRY_next_goal_prompt.md)
is the active decision handoff. This change authorizes no new experiment.

## Research contract

- **Question and authority:** substantive methodological novelty in FPGA-aware
  LLM protection is now fixed. A specific new method, experiment, budget and
  scientific hypothesis remain **unselected and not approved**.
- **Closest prior approaches:** Quasar-derived B, existing guard baselines and
  relevant FPGA architecture/co-design literature must be compared for the
  actual proposed mechanism. No novel gap has yet been established.
- **Hypothesis/difference:** unselected. A proposed method must change the
  guard's computation, representation, decision rule or joint hardware-aware
  design procedure; a routine existing-model port or new kernel alone is not it.
- **Scientific outcome:** demonstrate a method-attributable, useful
  security–hardware trade-off with prospectively bounded protection and
  legitimate-utility loss if quality is traded for cost. Report complete
  latency/throughput/resources and claim limits; speed alone is insufficient.
- **Hardware role:** implement the new method on FPGA/PL or heterogeneous
  FPGA–AIE-ML as appropriate. Compare established and proposed algorithms on
  competent implementations, conventional FPGA mappings and method ablations.
  Board measurements remain blocked until physical access/authorization.
- **Next discriminator:** identify at most two method-level candidates from
  saved evidence and focused prior art; recommend one falsifiable, bounded
  experiment for approval. Existing BERT/HLS work is supporting baseline
  evidence, not the automatic continuation goal.

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

The next agent should follow the method-first FPGA entry point and return
a concrete scientific hypothesis and bounded test for user approval. Do not
automatically complete the existing BERT FPGA port, expand CUDA studies, or
relabel routine accelerator work as methodological novelty. This change
approves no new budgets, models, threats, methods or experiments.

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
