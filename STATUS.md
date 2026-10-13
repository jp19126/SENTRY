# SENTRY status

**Updated: 2026-10-13 — documentation reorganization; no new experiment.**

**Scientific purpose approved; specific new method NOT selected.** The
approved direction is described in [RESEARCH_SCOPE.md](RESEARCH_SCOPE.md):
SENTRY requires a substantive new FPGA-aware guard algorithm or
algorithm–hardware co-design method. An unchanged BERT or other established
detector on FPGA is supporting work, not the scientific contribution.
[AGENTS.md](AGENTS.md) defines execution rules, not research goals.

No new method, model-training campaign, API spending, build campaign or
physical-board action is approved by this status update. The staged
[Codex execution prompt](SENTRY_codex_execution_prompt.md) is available,
but its substantive experiments require a separate method/budget approval.

## Current research decision

| Question | State |
|---|---|
| Method-level scientific hypothesis | **Unselected**; a differentiated FPGA-aware mechanism and falsifier are needed |
| Closest competing methods and hardware co-design | Must be checked for the actual candidate; [existing review](reports/prior_art.md) is bounded, not an exhaustive novelty clearance |
| Security/utility versus full hardware cost | Must be compared with suitable baselines and prospective quality bounds; no method-attributable benefit established |
| FPGA realization | Required by scope; offline hardware components exist, but end-to-end board validation is not available |
| Next research decision | **Unresolved:** selection of a defensibly new method and a bounded experiment; neither has execution authorization |

The old precision-allocation A/B/C campaign, further CUDA optimization, an
encoder-block prototype, and an unchanged-model FPGA port are **not automatic
next steps**. They can be proposed only when they answer the approved method
question and have an explicit endpoint.

## Completed evidence to reuse

The bounded [Goal R1 study](reports/goal_r1_decision.md) is complete. Its
source, inputs and recorded results are unchanged:

| Finding | Evidence boundary |
|---|---|
| Feature classifier: TP/FN/FP/TN = 2639/99/20/1349 | Fails the recorded recall/FPR screen; one recipe |
| Compact CNN: 2738/0/24/1345 | Passes recall, fails FPR; not a general CNN rejection |
| Native W8-QAT BERT/CUDA Graph: 2738/0/13/1356; warm mean 1.707225 ms | Development screen and 32 matched single-window checks, not a new scientific method or full deployment cost |
| BF16-attention reference: 2738/0/13/1356 after separate calibration | Changed arithmetic/scores; no AIE equivalence or accelerated BF16 timing claim |
| R1 hardware work | Two analytic mappings; zero new AIE/HLS prototypes and no validated redesigned accelerator |

All 55 development attack templates appeared in training. Final calibration,
final test and NotInject remain closed. R1 did not add a downstream target-LLM
protection/utility study. Its classification measurements alone do not prove
end-to-end safe and useful operation.

The legacy serial PL approach was stopped. Its saved software, numerical and
offline hardware evidence should be reused where valid, not discarded or
misrepresented as a functioning deployed detector. The earlier Goal 5 campaign,
board evaluation and additional target-LLM calls remain paused.

## Current offline FPGA evidence

The earlier Goal 4 work includes synthesis, partial RTL/component integration,
routed closures and conditional cost modeling, described in
[reports/goal4_platform_architecture.md](reports/goal4_platform_architecture.md),
[reports/goal4_numerical_bridge.md](reports/goal4_numerical_bridge.md) and
[reports/unfinished_measurements.md](reports/unfinished_measurements.md).
These are not equivalent to complete numerical acceptance, physical board
execution or measured end-to-end protection. The detailed
[board handoff](reports/board_handoff.md) records historical readiness checks
and access limits; its October 8 first-A plan does **not** override today's
method-selection decision or restart paused work.

## Unresolved decision and authorization

The next scientific decision concerns a specific FPGA-aware method and a
prospectively bounded test that can distinguish its contribution from an
unchanged existing guard or conventional accelerator. A method hypothesis,
nearest-prior-art distinction, comparisons, prospective security/utility
limits and resource requirements have **not yet been selected or approved**.
The completed R1 study and earlier hardware reports are evidence for a future
decision, not a standing execution order.

The earlier pause and budget/access restrictions remain in effect. Prior
goal numbers, prepared work, available machines and promising baselines do not
independently authorize continuation.

## History and reproduction

The previous full status is preserved verbatim in
[reports/status_before_purpose_lock_20261011.md](reports/status_before_purpose_lock_20261011.md).
The executed R1 prompt is preserved in
[reports/goal_r1_prompt.md](reports/goal_r1_prompt.md); the later method-first
handoff is in [docs/archive/](docs/archive/). The pre-correction status and R1
prompt were copied from commit `6c06973e5a568ad484ec0e9b779876bca1ec9626`.
Consult [reports/execution_log.md](reports/execution_log.md) and individual
reports for exact commands, numerical identities, limits and artifact paths.
No experimental code, datasets, checkpoints, configuration or measurements are
changed by the documentation cleanup.
