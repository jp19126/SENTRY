# Status

Updated: **2026-10-11 — research-purpose correction; documentation only.**

**RESEARCH-DIRECTION REVIEW REQUIRED.** The user's instruction is to prevent
purpose drift, not to launch another experiment. The purpose and decision rules
in [AGENTS.md](AGENTS.md) govern the next goal. The current entry point is
[SENTRY_next_goal_prompt.md](SENTRY_next_goal_prompt.md).

## Research contract

- **Question and authority:** retain the original high-novelty FPGA-enabled LLM
  protection objective, reinforced by the user's 2026-10-11 correction. No
  specific new scientific hypothesis or next experiment has been approved.
- **Closest prior approach:** the original safety-matched Quasar-derived B and
  relevant prior-art reports are existing references, not proof of a new gap.
  The appropriate nearest comparator depends on the proposed contribution.
- **Hypothesis/difference:** unselected. Recent suggestions about numerical
  vulnerabilities or threshold-tail protection are proposals, not adopted goals.
- **Scientific outcome:** a differentiated, falsifiable contribution relevant to
  protection and legitimate-task usefulness; supporting cost metrics remain
  separate. No central new method or architecture has yet been established.
- **Hardware role:** FPGA-enabled protection remains the research focus. BERT,
  CNN, CUDA and AIE-ML are possible instruments; 1.707 ms is a measured control,
  not the definition of scientific success or a reason to abandon the mission.
- **Next discriminator:** prepare a concise contribution proposal using existing
  evidence and focused prior-art checks; request approval of the question and
  bounded experiment before launching new research runs.

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

The next agent should follow the contribution-first entry point and return the
scientific decision for user approval. This change approves no new budgets,
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
