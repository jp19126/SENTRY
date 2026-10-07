# Unfinished measurements and continuation

Updated 2026-10-07 from existing evidence; no experiments rerun. Research remains
paused at the handoff in [STATUS.md](../STATUS.md). The table records established
remaining work, not authorization to dispatch it. Detailed completed attempts are
in [execution_log.md](execution_log.md).

Goal 2's detector and 120-request DeepSeek development pilot are complete under
the fixed 256-token cap; no further pilot generation is needed. Preserve historical
Gemini results separately. The initial 12 HLS syntheses and the active integer/FP32
component RTL campaigns are complete. The mover and controller checks and six-call
real-input prefix also passed within their stated boundaries. These are not
full-detector or physical-board measurements.

| Work | Missing prerequisite / next established action | Existing evidence to reuse |
|---|---|---|
| Goal 4 numerical qualification | Implement the remaining actual synthesized-arithmetic replay on the saved W8-QAT input, then the prescribed broader development bring-up. No complete replay driver exists. Floating and quantized reference identities remain unapproved; do not relabel historical SDPA scores. | [Numerical bridge](goal4_numerical_bridge.md), [completed prefix](goal4_real_prefix_rtl.md), [prepared arena](goal4_arena_preparation.md), [adapter audit](goal5_adapter_audit.md). Its stricter proposed numerical rule remains unadopted. |
| Goal 4 complete checking costs | Resolve physical DDR/NoC contention, host preprocessing/transport/synchronization and integrated resources/timing. `estimate_detector_cost.py` exists, but produces a component ledger with `search_ready: false` and null full latency/feasibility. | [Hardware cost model](hardware_cost_model.md), [control integration](goal4_internal_integration.md), `results/goal4/detector_cost_ledger.json`. The conditional static64 internal model is 1,145.16874–1,215.94762 ms at 200 MHz, excluding unresolved physical-memory/host costs. |
| Goal 5 A/B/C comparison | Obtain common complete costs, frozen caps and qualified matching scores; finish campaign orchestration and then identify one evidenced B weakness for C. `search_precision.py` exposes only readiness, saved-score bring-up and an offline adapter check. Zero campaign maps have run; C remains undefined. | [Method](method.md), [adapter audit](goal5_adapter_audit.md). The provisional-QAT and cache-identity corrections are complete. Retain at most 64 logically consulted maps and two one-epoch-QAT shortlist entries per method. |
| Goal 6 U/B/C board evidence | Complete the detector/host integration, freeze feasible candidates and establish board connection, physical unit revision, memory allocation and programming authorization. Bring up uniform W8 first; then run the prescribed agreement and same-board measurements. | [Command plan](goal4_command_plan.md), [arena preparation](goal4_arena_preparation.md). The relative 64 MiB fixture is not allocated memory; FP32-only routed timing is not a routed full system. |
| Goal 7 final evaluation | Frozen methods/models/threshold policy and board finalists. Final calibration/test, separate NotInject, bounded held-out challenge and one extension remain unexecuted. Final workload is 128/256/512 × 1/8/32, with at least 10,000 checks × three repeats at main configurations; matched-load energy and held-out application work also remain. | [Protocol](protocol.md) and [research goals](../CODEX_RESEARCH_GOALS_v2_1.md). Short pilots do not replace these measurements. |
| Goal 8 complete paper | Update the interim manuscript from actual method, hardware and final evaluation results once available. Do not launch a campaign merely for writing. | [Draft](../paper/draft.md), [research summary](research_summary.md) and saved development figures. |
| Optional Prompt Guard reference | Gated model access and exact revision remain unavailable; this optional reference does not block ordinary detector work. | [Baseline selection](baseline_selection.md), `models.quality_reference` in shared configuration. |

On research resumption, use [STATUS.md's next steps](../STATUS.md#next-step-and-resumption)
and [README commands](../README.md#commands-and-checks). Native software uses the
existing Windows environment; vendor execution uses WSL Python. Current scripts
are in SENTRY, while historical absolute GATE paths in saved builds still need
review before reuse. Do not repeat successful synthesis, packing, prefix replay
or application requests merely to refresh documentation.
