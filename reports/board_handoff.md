# Board handoff

**2026-10-13: span_coupled_pilot_v1 closed with a negative candidate result; NOT board-ready.**

The candidate failed the approved quality, coupling-benefit and FPGA resource
gates. Phase C integration was not entered. Six completed fits and all final
available comparisons are in [method study](method_study.md). The stronger
MLP/SoPa comparison remains incomplete: SoPa training was stopped at 318 seconds
on observed projected budget infeasibility, without scoring a partial checkpoint.
No trained SoPa export or SoPa hardware result exists.

## Actual implementation and build state

Vitis/Vivado 2025.2 targets `xcve2802-vsvh1760-2MP-e-S`; requested clock 5 ns.
All paths below are under `results/span_coupled_pilot_v1/`.

| Design / preserved build | Complete C / trace checks | Actual vendor outcome |
|---|---|---|
| Fused coupled / `hls_coupled_stream_v4` | 143 cases; unchanged arithmetic trace covered by v3, 46 windows | 104,176 LUT, 50,671 FF, 167 DSP, 99 BRAM18, 65 URAM; LUT limit failed; RTL memory-floor stop |
| Buffered coupled / `hls_coupled_buffered_v1` | 143 cases, 46 windows | SYNCHK unsupported pointer selection; no complete synthesis or RTL cost |
| Span MLP / `hls_span_mlp_v1` | 143 cases, 46 windows | 95,120 LUT, 37,954 FF, 292 DSP, 119 BRAM18, 65 URAM; DSP limit failed; RTL memory-floor stop |
| SoPa | No trained full-chain C replay | Not dispatched: completed trained export unavailable |

Both successful top-level HLS reports estimate 5.992 ns. Applying the approved
20% frequency derating gives a modeled 7.490 ns cycle, not achieved routed timing.
Neither RTL attempt completed a transaction or returned a pass. HLS average or
worst-case latency is not substituted for observed cold/warm cycles. Terminal
reports are `hardware_terminal_report_v1.json` inside each of the three attempted
build directories. Both corrective rebuild slots are consumed; no retry is authorized.

The old buffered source snapshot is preserved. Current source fixes the
unsupported pointer selection, and the first MLP build accepted that correction.
This does not retroactively repair the failed buffered build. Exact source/model
hashes, test fixtures, generated Tcl and vendor reports are retained per attempt.
No device implementation, routed image, board run or physical result exists.

## Artifacts and implemented entry points

Candidate exports are `coupled_seed42_v1/export/` and
`coupled_seed1337_v3/export/`; the hardware pilot uses seed 42. The MLP hardware
export is `span_mlp_seed42_v1/export/`. Fixtures are
`coupled_seed42_fixtures_v1/` and `span_mlp_seed42_fixtures_v1/`, from the same
32 approved training documents plus corners, not protected evaluation data.
The Python integer model is `scripts/span_numerics.py`; the full C region is in
`hardware/span_guard/`. All executed commands are in [execution log](execution_log.md).

The executed C replay entry point, shown for traceability rather than dispatch,
was:

```sh
python3 -B scripts/span_hardware.py run-c --design span_mlp --output results/span_coupled_pilot_v1/hls_span_mlp_v1 --trace
```

Native Python then ran `scripts/span_hardware_trace.py --fixtures
results/span_coupled_pilot_v1/span_mlp_seed42_fixtures_v1 --replay
results/span_coupled_pilot_v1/hls_span_mlp_v1 --output
results/span_coupled_pilot_v1/span_mlp_seed42_trace_v1.json`. These outputs already
exist; preserve them. `span_hardware_report.py` reads preserved vendor evidence
and requires a fresh output filename. Existing paths and reproduction boundaries
are further described in [hardware README](../hardware/span_guard/README.md).

The implemented interface carries uint16 content-token IDs, window/document
boundaries and a continued document maximum; model load is a separate cold
transaction. Warm checking includes embedding, CNN, projection, recurrence/head,
LUT and strict decision. Host tokenization/packing, transport and result return
remain required dependencies. No measured warm-region cost is available, and
unknown host/transport terms are not zero. The historical UART route has at least
11.111 ms framing serialization; no faster complete transport was validated.

A future board campaign would require a newly approved viable method/build,
complete RTL and routed checks, platform/transport validation, and physical
measurements of correctness, latency, throughput and energy. It is outside this
completed pilot. Protected final data, new target calls and physical actions
remain excluded. No GitHub synchronization occurred.

## Historical handoff — retained, non-authorizing

The following October 8 state and deferred commands are preserved for evidence.
They do not resume the old Goal 4/5 or A/B/C campaign, and their old usage limit
is superseded by the current user's post-reset 50%-used cap.

**Paused at the user weekly usage limit (2026-10-08T22:05:20.790425+00:00).** Actual usage tool reports90% used/10% remaining. No reset, scoring release/dispatch or physical-board action. OFFLINE BOARD-READY remains unfinished. Attempt002 is preserved; after explicit user resume, recheck usage and actual resources before any release. Last actual Windows RAM3,632,664,576B is below the unchanged4,294,967,296B preflight floor. Pause evidence: `results/goal5/campaign_next_execution_review_v2/guarded_campaign_dispatch_v1/first_A_attempt002/root_weekly_usage_pause_v1.json`. Earlier observations below are historical.


Checkpoint read: 2026-10-08 21:55 UTC. The latest supplied weekly usage observation was 89% used (11% remaining) at 21:50:11 UTC; pause at 90%. The user decision between freeing 1 GiB and queuing scoring remains pending. Counts below are from saved records read for this checkpoint. This is an offline handoff: no target API, physical board, or new vendor/model run was used.

## Current numerical state

- True-length bounded review: root accepted 50 windows across four routes and eight document-route outcomes on the same two development documents. Three benign false positives remain; large propagated A8 differences remain documented. The independent floating L512 full-checkpoint reread is unperformed. This bounded numerical acceptance is not full RTL, board agreement, population quality, or generic score approval. Evidence: `results/goal4/numerical_bridge/real_length_bringup_v1/final_saved_review_source_v1/actual_complete_review_v1/root_bounded_acceptance_v1.json`.
- Mixed PTQ tuple `4488884884844448`: root accepted the exact saved development exemplar after all 32 raw-document rechecks. The main and independent peer reviews cover 66 windows/32 documents, 27,126 original calls, 1,386 metrics, and 155,713,536 same-input codes and accumulators. They retain 1,482,297 propagated A8 differences (1,339,206 valid, 143,091 padding, 4,998 CLS), maximum valid hidden error 1.098734498 at layer2 (case25/window2), separate maximum CLS hidden error 0.31253311038, and scalar window-risk error 0.000581837259; every document max/winner/strict decision agrees. Five benign false positives remain as diagnostics, not population FPR. The shared fixed16 PTQ numerical scope and matching development scoring identity are separately adopted; this permits matching development scoring under the approved scope. Each future PTQ map still needs its own proof and full matching quality scores. Future mixed QAT remains unapproved and has no reviewed scope. Case30/window1 timeout recovery preserved 59 committed entries and 120 partial calls; the exact same-input memory query passed on retry. Evidence: `results/goal4/numerical_bridge/precision_qualification_v1/campaign_ptq_p4488884884844448/bounds_v1/qualification.json`; `results/goal4/numerical_bridge/mixed_campaign_review_v1/final_saved_review_source_v1/actual_complete_review_v1/`; `results/goal4/numerical_bridge/mixed_campaign_review_v1/final_saved_review_source_v1/actual_complete_review_v1/root_bounded_acceptance_v1.json`; `results/goal4/numerical_bridge/mixed_campaign_review_v1/final_saved_review_source_v1/actual_complete_review_v1/root_score_identity_decision_v1.json`; `results/goal4/numerical_bridge/mixed_campaign_review_v1/reviewed_scope.json`; `results/goal4/numerical_bridge/mixed_campaign_review_v1/reviewed_qat_scope.json` (absent); `results/goal4/numerical_bridge/mixed_campaign_review_v1/final_saved_review_source_v1/root_terminal_bookkeeping_v1/root_record.json`; `results/goal4/numerical_bridge/mixed_campaign_review_v1/case30_window001_operational_recovery_v1/`.
- Matching development score identity and shared fixed16 mixed PTQ scope are adopted for bounded development scoring only. Own-map numerical proofs and full matching development quality scores remain mandatory; no A/B/C comparison or final method freeze exists. Future mixed QAT remains unapproved and `reviewed_qat_scope.json` is absent. The fixed external C decision path is `results/goal5/campaign_ab_v1/c_decision.json`; it is absent, and the C executor is missing.
- Guard and first-A state: probe003 remains operational evidence only. Compact approval-plan binding was accepted after 18 independent checks, retaining full-scope snapshots and exact byte comparisons. Native scope validation passed in 44.453 seconds after serialization-only WSL 3.12/native Python 3.11 AST `type_params` correction; the original failure is preserved. Extended-UNC first-A preview passed with `missing=[]`; ordinary UNC lost 17 long snapshot files (77 native versus 94 WSL), and a 281-character path test passed. Attempt002 preparation passed with a 5,040,280-byte plan/69 snapshots, `missing=[]`, `allocation_adopted=false`, max A maps 1 and C null. Attempt001 is preserved as superseded/unconsumed. Windows memory remains below the 4 GiB preflight floor; no release, dispatch or scoring exists. The PTQ/QAT selector review is pending. Evidence: `results/goal5/campaign_next_execution_review_v2/guarded_campaign_dispatch_v1/job_probe_attempt003/root_operational_acceptance_v1.json`; `results/goal5/campaign_next_execution_review_v2/guarded_campaign_dispatch_v1/small_approval_plan_binding_v1/root_source_acceptance_v1.json`; `results/goal4/numerical_bridge/mixed_campaign_review_v1/final_saved_review_source_v1/actual_complete_review_v1/native_scope_validation_v1.json`; `results/goal5/campaign_next_execution_review_v2/guarded_campaign_dispatch_v1/first_A_attempt002/root_plan_review_v1.json`; `results/goal5/campaign_next_execution_review_v2/guarded_campaign_dispatch_v1/first_A_attempt002/root_exclusive_handoff_v1.json`; `results/goal5/campaign_next_execution_review_v2/guarded_campaign_dispatch_v1/first_A_attempt002/root_idle_memory_observation_v1.json`; `results/goal5/campaign_next_execution_review_v2/guarded_campaign_dispatch_v1/first_A_attempt001/superseded_unconsumed_plan_v1.json`.
- Existing routed L128/L256/L512 builds, cost records, and deployment packages remain available; the connected-service RTL still stops before terminal completion. Final quality remains deferred behind the actual method freeze. The accepted NativeQT loader, timing-grid, and TableQA source extensions have not been evaluated. Physical measurements and the remaining non-board work are separate gates.

## First-A guarded preparation and deferred attempt

The exact mixed PTQ exemplar has bounded numerical acceptance and the shared fixed16 PTQ development scope/score identity is adopted. Each future PTQ map needs its own proof and full matching quality scores; future mixed QAT has no approved scope. Native scope validation passed. No A/B/C comparison, C decision, or final freeze exists.

The ordinary UNC preview failed on 17 long snapshot files (native inventory found 77 files where WSL expected 94); the extended-UNC preview passed with `missing=[]`, and a 281-character path test passed. Attempt001's 242 MB plan is retained as superseded/unconsumed. Attempt002 preparation passed with a 5,040,280-byte plan and 69 full snapshots; the exact current full-scope snapshot bytes validate. The plan records `allocation_adopted=false`, `max_new_maps=1`, and `C=null`. No release, dispatch, owned scoring job, or score call exists. The native Job probe and compact approval-binding review are source/operational evidence only.

The attempt002 full-handoff inventory at 21:42:54 showed Windows available memory of 3,787,337,728 B, below the 4 GiB preflight floor by 507,629,568 B; WSL had 22,110,208,000 B and no blocking processes. The 300-second Windows-only idle observer exited 0 after 15 samples (minimum 3,704,627,200 B; maximum 3,903,287,296 B; last 3,792,723,968 B), with no passing full handoff, model, C, vendor, or board dispatch. This is the actual RAM resource gate for non-board scoring; physical-board work remains a separate, unexecuted gate. Evidence: `results/goal5/campaign_next_execution_review_v2/guarded_campaign_dispatch_v1/first_A_attempt002/root_plan_review_v1.json`; `results/goal5/campaign_next_execution_review_v2/guarded_campaign_dispatch_v1/first_A_attempt002/root_exclusive_handoff_v1.json`; `results/goal5/campaign_next_execution_review_v2/guarded_campaign_dispatch_v1/first_A_attempt002/root_idle_memory_observation_v1.json`.

Attempt002 preparation is complete. Its command is preserved here for traceability; do not launch a duplicate:

```powershell
& 'C:\research\GATE\.venv\Scripts\python.exe' -B '\\?\UNC\wsl.localhost\Ubuntu\home\jp19126\Projects\SENTRY\results\goal5\campaign_next_execution_review_v2\guarded_campaign_dispatch_v1\guard_native_campaign.py' --arm A --prepare-dir '\\?\UNC\wsl.localhost\Ubuntu\home\jp19126\Projects\SENTRY\results\goal5\campaign_next_execution_review_v2\guarded_campaign_dispatch_v1\first_A_attempt002'
```

After Windows available memory meets the 4 GiB preflight floor and a separate one-job resource-release record is explicitly adopted, use this deferred guarded attempt command:

```powershell
& 'C:\research\GATE\.venv\Scripts\python.exe' -B '\\?\UNC\wsl.localhost\Ubuntu\home\jp19126\Projects\SENTRY\results\goal5\campaign_next_execution_review_v2\guarded_campaign_dispatch_v1\guard_native_campaign.py' --execute-plan '\\?\UNC\wsl.localhost\Ubuntu\home\jp19126\Projects\SENTRY\results\goal5\campaign_next_execution_review_v2\guarded_campaign_dispatch_v1\first_A_attempt002\plan.json' --release '\\?\UNC\wsl.localhost\Ubuntu\home\jp19126\Projects\SENTRY\results\goal5\campaign_next_execution_review_v2\guarded_campaign_dispatch_v1\first_A_attempt002\root_release.json'
```

All deferred guard commands target attempt002. Attempt001 is superseded, unconsumed, and must not be executed under current bindings. The guard owns the child job; do not invoke the campaign driver directly. No release or scoring has been authorized.

## Physical-board continuation: UNEXECUTED

No physical-board command has run. Board programming, boot, UART capture, timing, arena, and target API actions remain unexecuted. The memory-floor hold above is a non-board scoring resource gate; it does not authorize or block physical work. Any physical continuation needs a separate explicit user request plus actual unit, boot/media, UART, timer, and arena settings. `hardware.programming_authorized` remains false.

L256 board preview example — **UNEXECUTED**:

```bash
python3 -B scripts/run_board_evaluation.py --package build/deployment/l256_uniform_w8_boot_v1/package.json --stage bringup
```

Physical program template — **DO NOT EXECUTE** until separately approved and all runtime settings are supplied; `$RUNTIME_JSON`, `$HW_SERVER`, and `$TARGET` remain unresolved:

```bash
python3 -B scripts/run_board_evaluation.py --package build/deployment/l256_uniform_w8_boot_v1/package.json --stage program --runtime-settings "$RUNTIME_JSON" --hw-server-url "$HW_SERVER" --target-filter "$TARGET" --execute-hardware
```

Numerical root acceptance does not imply physical agreement. Keep these physical runtime inputs separate from non-board scope, quality, and RAM-release gates.


### 2026-10-08T22:00:00.868886+00:00

Freshroot inventory at2026-10-08T21:59:09.700200Z still has only3,632,664,576B Windowsfree (<4,294,967,296B preflightfloor), WSL22,100,799,488B andzero native/C/vendor blockers. Evidence: results/goal5/campaign_next_execution_review_v2/guarded_campaign_dispatch_v1/first_A_attempt002/root_exclusive_handoff_v2.json. Latestactualweekly89%used/11%remaining@21:59:03UTC;90%pause nottriggered. UserRAM/queueanswerpending. No release, dispatch, scoring orphysicalactivity.
