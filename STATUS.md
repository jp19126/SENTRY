# Status

Last updated: **2026-10-07** (documentation review; no new research measurements).
Latest research handoff: 2026-09-26 22:30 UTC (2026-09-27 in Europe/Madrid).
Current objective: complete Goal 4 numerical qualification and full checking-cost
accounting before the Goal 5 campaign. Research remains paused.

Research paused after completing the authorized bounded extension at a recorded **89% weekly usage used** (11% remaining at that handoff, not a current account reading). The extension used 4 percentage points beyond 85%, below the 7-point allowance; reset `1791049884` remained unchanged. Resume only on a new user instruction. Standing rules remain: pause on any new Codex usage reset; normally pause at 85%, allowing completion of a bounded task only when expected within 7 further points, with a hard stop at 92%.

The target remains institutional `deepseek-ai/DeepSeek-V4-Flash-0731`. Model-selection benchmarking is stopped; the 120-request application cap is exhausted and no new API calls are planned.

| Goal | Evidence and state |
|---|---|
| 1 | Complete. Grouped data, real model bring-up, evaluator and Quasar-ViT adaptation. See `reports/protocol.md`, `reports/data.md`, `reports/baseline_selection.md`. |
| 2 | Required work complete: 3 detector epochs and 120 DeepSeek requests/judgments. See `reports/floating_baseline.md`, `reports/deepseek_pilot_findings.md`. Optional Prompt Guard remains unavailable. |
| 3 | Available software work complete: 20 PTQ maps, 2 one-epoch QAT anchors, arithmetic checks and CPU/CUDA timing pilots. M2 still needs Goal 4. |
| 4 | Paused during offline integration. All 12 initial HLS syntheses and all dispatched integer/FP32 RTL cases are complete. The fixed sequencer/control integration passed 13 RTL cases with explicit arithmetic stubs; whole-checking latency, system feasibility and general numerical acceptance remain unresolved. See below. |
| 5 | A/B adapter exists; 0 actual comparison maps. Provisional QAT shortlist and numerical cache-identity corrections passed their bounded offline check. No C selected, no hardware-grounded campaign. |
| 6-7 | Not executed. VEK280 was reported disconnected at the research handoff; current connection is unverified. No programming, routed full system, final-test scoring, board latency or energy. |
| 8 | Interim draft only. See `paper/draft.md` and `reports/research_summary.md`; current detailed hardware evidence is in the Goal 4 reports. |

## Current offline FPGA evidence

- VEK280 part `xcve2802-vsvh1760-2MP-e-S` and WSL Vitis/Vivado 2025.2 verified. The objective is 200 MHz; programming is unauthorized. All 12 initial HLS syntheses are used.
- Active common profiles are cached 16/32 and static-bank 64. Every targeted and reserved case for these profiles passed exact outputs twice. The last reserved static64 case measured **11,342,993 cycles twice**, exactly matching its pre-dispatch freeze. Original tiled-profile results remain historical evidence.
- Static64 HLS uses 28,595 LUT versus the original cached64's 87,741, with a 4.441 ns estimate. This common repair is available to A/B/C and is not C novelty.
- All 15 primary FP32 call shapes passed twice. The weighted 411-call subtotal is **94,864,971 cycles**, or **474.324855 ms** at 200 MHz. All 11 smaller FP32 checks also passed twice. Representative FP32-only implementation meets 200 MHz: 4.932 ns routed period, +0.068 ns slack.
- The known static64 integer-plus-FP32 subtotal is **1,118.172-1,188.951 ms** for uniform W4/W8. It excludes unresolved integration/physical-memory/host costs and shows no software-speed advantage. `scripts/estimate_detector_cost.py` preserves null full latency and feasibility.
- The fixed L256 plan has 664 commands and uses 42,225,672 bytes of a planned 64 MiB relative arena. No physical memory is allocated. Independent order/layout review passed; document reset, exact threshold bits and full-width ID checks are explicit. The bounded mover passed 13 focused RTL cases. Sequencer/control integration passed 13 cases, including four full schedules. Conditional control/layout overhead is 5,399,345 cycles (26.996725 ms), giving a static64 internal-path model of 1,145.16874–1,215.94762 ms. See `reports/goal4_internal_integration.md`. See `reports/goal4_command_plan.md`, `reports/goal4_data_mover.md`.
- A 64 MiB arena image for the existing W8-QAT real input is prepared. All 151 buffers passed readback and 65 saved exports matched exactly. Reuse `results/goal4/internal_integration/arena_w8_qat_real_input_v1/`; see `reports/goal4_arena_preparation.md`. It is a relative fixture, not allocated/programmed hardware or an approved deployment.
- One real-input C-sim bridge confirms integer arithmetic but exposes a quantization tie crossed by FP32 attention differences. The ordered CPU reference reproduces C-service logits/risk on this input. CUDA differs by 1 ULP in risk and retains its exact-comparison failure. This is bring-up evidence, not general numerical acceptance or a dataset rescore. See `reports/goal4_numerical_bridge.md`.

- The actual synthesized FP32 prefix passed all six saved-input calls: embedding addition is bit-exact; embedding LayerNorm meets the existing local bounds despite 12,415 bit differences (maximum absolute error 9.5367431640625e-7); all 65,536 first-query A8 codes match both the saved ordered CPU reference and same-input quantization. Actual RTL outputs feed each next operation. The 811,174-cycle subtotal covers only this prefix under simulated memory, excluding Python/file handoffs. See `reports/goal4_real_prefix_rtl.md` and `results/goal4/numerical_bridge/rtl_prefix_v1/run.json`. Full-detector agreement remains unresolved.

## Scientific boundaries

All uniform development references recalled 2,738/2,738 attacks. FP32 FPR 18/1,369 fails 1%; W8 QAT 13/1,369 passes; W4 QAT 20/1,369 fails. The extra QAT epoch is a confound. Final calibration/test and NotInject remain closed.

The DeepSeek pilot has 43 normal completions and 77 truncations (64 empty). Clean completion is 9/20; attacked completion 24/100 and attack success 11/100 without guard. Rejecting all 100 attacked chunks yields 0 attack successes and 0 task completions. These apply only to the recorded pilot configuration.

## Next step and resumption

After explicit research resumption:

1. Implement the remaining synthesized-arithmetic replay on the same saved input,
   reusing the completed prefix, arena and service records. No complete replay
   command exists yet. See [the adapter audit](reports/goal5_adapter_audit.md).
2. Complete the prescribed broader development bring-up and establish the
   hardware-matched reference. The narrow ordered floating path passed one CPU
   input; both floating and quantized identity gates exist, but approvals remain
   null. Historical SDPA scores cannot stand in for qualified matching scores.
3. Establish the common physical-memory/host model and complete costs/caps before
   Goal 5. Its campaign CLI is unfinished. See
   [unfinished measurements](reports/unfinished_measurements.md) for dependencies.

Reuse completed synthesis, packing and API evidence; none needs repeating for
this documentation review.

Read-only preview of the completed prefix (verified 2026-10-07):

```powershell
wsl.exe -d Ubuntu --exec /usr/bin/python3 -B /home/jp19126/Projects/SENTRY/scripts/run_real_prefix_rtl.py
```

Offline ledger refresh after relevant evidence changes (writes
`results/goal4/detector_cost_ledger.json`; inspected, not rerun in this review):

```powershell
wsl.exe -d Ubuntu --exec /usr/bin/python3 /home/jp19126/Projects/SENTRY/scripts/estimate_detector_cost.py
```

This checkout is `/home/jp19126/Projects/SENTRY`; the historical `GATE` checkout is a separate directory. Use [README paths](README.md#environment-and-paths) for current commands. Saved vendor projects contain historical absolute paths; their reuse from SENTRY has not been validated. Native Python is `C:\research\GATE\.venv\Scripts\python.exe`; vendor tools are in `/home/jp19126/Xilinx/2025.2/{Vitis,Vivado}/bin/`. Physical board revision, boot/host connection and memory/coherency allocation remain unresolved. Research changes are uncommitted, based on `df904fcefe6566650490f7476e13914524db6907`. Detailed attempts, failures and progress are in `reports/execution_log.md`.

Documentation/configuration discrepancy: `configs/project.json` still has an earlier
`configuration_state` string saying to continue remaining profiles. Its structured
`execution_control.current_pause`, `conditional_usage_extension.state` and
`results/goal2/pause.request` record the later pause. The completed 12 initial
syntheses are not a new allowance. Configuration and pause markers were preserved
during this documentation-only review.
