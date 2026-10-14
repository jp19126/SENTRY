# SENTRY status

**Updated: 2026-10-13 — bounded span-coupled pilot closed with a negative candidate result.**

The approved study was implemented and executed. The candidate failed its
prospective quality, mechanism and FPGA resource gates. Conditional Phase C
integration is not justified. The SoPa comparison and RTL timing remain
incomplete; their absence is not evidence against SoPa or proof of a speedup.
See [method study](reports/method_study.md) for the contract, results and limits,
and [execution log](reports/execution_log.md) for preserved attempts.

## Current decision and evidence

| Question | Observed result |
|---|---|
| Useful coupled guard | Both fixed seeds fail authored-stress quality; seed 42 also fails natural FPR |
| Coupling versus independent pooling | Integer balanced-error gains are -10.2539 and +1.1719 percentage points; both miss the required +5, and the pooled paired 95% interval includes zero |
| Ordinary span comparator | Both span-MLP fits completed and fail stress quality; the stronger MLP/SoPa discriminator remains unresolved because SoPa did not complete |
| Numerical implementation | All six completed fits meet the same-weight integer degradation limits; this does not repair failed absolute quality |
| Fused FPGA mapping | HLS uses 104,176 LUTs against a 100,000 limit; RTL stopped at the memory floor before completing a transaction |
| Conventional FPGA mappings | Buffered candidate failed synthesis; MLP synthesized with 292 DSPs against a 256 limit, then RTL stopped at the memory floor |
| Full-check speedup | Unresolved: no observed complete-region RTL cycles, qualified timing campaign, validated transport or physical measurement |
| Next action | Retain this negative study. A revised method, implementation campaign or additional retry requires a new explicit decision and budget |

Six of eight planned fits completed: coupled, independent pooling and span MLP,
each at seeds 42 and 1337. Frozen BERT, CNN, PIGuard and a saved-score BERT/CNN
conjunction also failed the stress screen. Final combined saved-score analysis:
`results/span_coupled_pilot_v1/analysis_final_available_v1.json`.

SoPa seed 42 was stopped safely after 318 seconds. Two observed 50-batch
intervals projected approximately 5.5 hours for the fixed ten-epoch recipe,
versus its 45-minute ceiling. This was a coordinator early stop on projected
budget infeasibility, not an actual timeout or usage-cap pause. No partial model
was scored. Seed 1337 and the trained-export-dependent SoPa hardware attempt
were not run. The full comparative protocol therefore remains incomplete.

Both coupled hardware mappings and span MLP pass 143 full C cases and 46
nonempty-window trace checks each. These are software arithmetic checks, not
RTL or board validation. The two completed top-level HLS reports estimate
5.992 ns against a 5 ns target. Both software correction slots and both hardware
correction slots are consumed; no retry or integration was launched after the
failed continuation gates. No deployable image exists.

Saved-response replay uses unchanged historical inputs: the six learned fits
add no rejection among 20 distinct clean tasks, preserve 9 saved task completions,
and block all 100 saved attacks. This is a small offline replay, not new target
execution or evidence of task success on the authored stress probes. Those
probes have documented wording, length and padding confounds; no unseen attack
distribution or final held-out claim is made.

## Execution boundary

The user approved `span_coupled_pilot_v1` and its conditional integration contract.
The bounded candidate decision is now negative. The scientific purpose in
[RESEARCH_SCOPE.md](RESEARCH_SCOPE.md) is unchanged: supporting ports and passing
C checks are not a new research contribution. [AGENTS.md](AGENTS.md) governs work.
No target API calls, paid services, protected final partitions, physical board
actions or GitHub synchronization were performed. Changes remain local.

The global usage reset was observed at 05:06 UTC on 13 October, with reset
boundary 1792472301. The user's post-reset **50%-used cap** is active; latest
observed usage is **7%**. Evidence is in
`results/span_coupled_pilot_v1/usage_reset_observed_v1.json`. The temporary SoPa
stop marker was removed after process exit; no training or vendor job remains.

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

## History and reproduction

The previous full status is preserved verbatim in
[reports/status_before_purpose_lock_20261011.md](reports/status_before_purpose_lock_20261011.md).
The executed R1 prompt is preserved in
[reports/goal_r1_prompt.md](reports/goal_r1_prompt.md); the later method-first
handoff is in [docs/archive/](docs/archive/). The pre-correction status and R1
prompt were copied from commit `6c06973e5a568ad484ec0e9b779876bca1ec9626`.
Consult [reports/execution_log.md](reports/execution_log.md) and individual
reports for exact commands, numerical identities, limits and artifact paths.
The historical documentation cleanup did not change experimental artifacts; the
subsequent approved span pilot is recorded separately above.
