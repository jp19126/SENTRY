# Goal 4: VEK280 cost-model evidence

Goal4 remains active. Target-specific HLS data and integer RTL calibration are available; whole-model runtime, integrated-system feasibility and board measurements remain unfinished. The board need not be connected for this offline phase.

## Target and current implementation

The allocated device is AMD VEK280, installed board definition `xilinx.com:vek280:part0:1.2`, part `xcve2802-vsvh1760-2MP-e-S`. WSL Vitis/Vivado2025.2 execute at the user-supplied paths. The PL-IP objective is200MHz with0.5ns HLS uncertainty; the declared budget is70% of device PL resources. These are design settings. Physical unit revision and DDR/NoC/host integration remain unresolved; programming is not authorized. See `reports/fpga_toolchain.md`.

The common design comprises one reused W4/W8xA8 integer engine and one fixed FP32 service. All24 encoder linear matrices use the selected physical profile; other detector arithmetic stays on FPGA in the declared schedule. No independent per-layer datapaths, CPU nonlinear offload or unimplemented overlap is credited.

Initial tiled profiles4/8/16 remain historical characterization evidence. The intended active comparison space is the separate cached16/32/64 family in `hardware.hls.cached_linear`, equally available to A/B/C. Final feasible profiles have not been selected. Increasing lane count does not by itself establish a speedup.

## Actual component evidence

| Component | BRAM18K | DSP | FF | LUT | HLS period |
|---|---:|---:|---:|---:|---:|
| Initial tiled4 |2|3|6291|6809|4.422ns|
| Initial tiled8 |2|3|6892|8249|4.422ns|
| Initial tiled16 |2|3|8048|11214|4.422ns|
| Cached16 |130|7|8149|15443|4.258ns|
| Cached32 |130|7|10930|30644|4.258ns|
| Cached64 original |130|9|19744|87741|4.258ns|
| Cached64 static-bank repair |130|7|19317|28595|4.441ns|
| Common FP32 panel cache |73|95|24808|29830|4.499ns|

Cached64 synthesis completed. All 12 initial HLS syntheses have completed:3 original integer,5 FP32 revisions,3 cached integer profiles and1 equivalent static-bank64 repair. No initial allowance remains. The repair reduced estimated LUTs from87,741 to28,595 and Csim passed; its two targets and reserved case have passed twice with exact pre-dispatch predictions. The reserved latency is11,342,993cycles in each repetition. Pre-synthesis setup/C-linker failures are preserved separately. Earlier FP32 builds and source are archived; current schedule revision is `panel_cache_batched_attention_v1`. HLS resource counts do not establish routed or integrated-system costs. BRAM18K is18Kib; compare consistently with the configured36Kib budget, without assuming all banks pair perfectly.

The representative current FP32 component implementation completed using the documented [AMD export_design implementation flow](https://docs.amd.com/r/en-US/ug1399-vitis-hls/export_design). Actual routed period is4.932ns, slack+0.068ns and TNS0 at5ns objective. It uses76BRAM18K-equivalent,91DSP,22246FF and25625LUT (including1703SRL). It is an out-of-context IP timing/resource check, not a DDR-connected detector or board result.

## Integer arithmetic and runtime calibration

Signed codes are W4[-8,7],W8/A8[-128,127], zero point0, exact int32 accumulation. W8 uses signed high nibble and unsigned low nibble: w=16*h+l. W4 is packed low-nibble-first; the cached kernel reads each packed byte once and stores two unpacked codes. Both precisions use the same physical cache capacity. Conversion/rescale/bias occurs in the fixed FP32 service and is charged separately.

All9 targeted original RTL cases pass two exact-output transactions. The calibrated bounded-memory prediction is

`C = C_nominal + 3 + R*O*(127 + 1021*I)`.

It was frozen before the three longer reserved cases. All three reserved cases passed twice with exact frozen predictions:17,130,627 /51,744,899 /42,505,347cycles for lanes4/8/16. Original nominal predictions/errors remain separate. See `reports/goal4_cost_accounting.md` and `results/goal4/characterization_summary.json`.

The separate cached16 validation has passed all3 targets and its reserved case twice. Reserved latency is17,634,449cycles, exactly matching its independently frozen source-derived prediction. The matched targeted speedups over original16 are1.93-2.51x. For the reserved W8 case, actual weight traffic is262,144bytes in4096bursts, versus67,108,864bytes for the original tiled engine's first reserved transaction. Activation reloads remain; they must not disappear from the estimator.

Cached32 has a different generated nested-loop schedule. Its predicted complete primary integer cost is worse than16 throughout the allowed map space; both targets and its reserved case passed twice exactly. This is a compiler scheduling result, not evidence that doubling lanes halves latency. Its own freeze is `results/goal4/cached_characterization_lanes32.json`.

These direct RTL observations use unchanged synthesized DUTs and fixed byte-array memories: one outstanding burst per port, registered responses, no additional DDR delay or cross-port contention. They are accepted-start-to-done cycles, distinct from vendor UVM results and board timing. Start-to-start includes the harness restart. Failed UVM attempts and the bounded-harness transition are preserved in `reports/execution_log.md`.

## Whole-model accounting

The current FP32 service contains DOT, quantization/rescale, embedding/residual addition, centered LayerNorm, stable softmax, erf GELU and tanh. The panel cache reuses one N<=64,K<=512 weight panel across query rows. All15 mapped data loops reportII1; ascending reductions and separate dot/rescale FP32 operations are preserved. Actual C-simulation checks passed. Standalone formula tolerances do not establish detector-score agreement.

AtL256,B1, the explicit batched schedule has411FP32 calls and24integer calls. Useful service traffic is105,393,192bytes. The earlier abstract layout estimate was18,092,032bytes using64-bit IDs and uncached repeated mask reads. The concrete command plan instead specifies32-bit IDs and a cached binary mask:17,956,864 useful movement bytes plus2,404 input/control read bytes. Host packing and checking of IDs must be charged separately; these planned counts are not measured bandwidth or latency. Historical attention/layout accounting is retained in the schedule report for traceability. All15 primary FP32 RTL shapes now passed two repetitions. Their411-call weighted full-service sum is94,864,971cycles (474.324855ms at5ns), replacing the49,742,799-cycle partial HLS report component. This is a sum of representative service observations, not a measured full-detector execution. `reports/goal4_fixed_fp32_plan.md` and `reports/goal4_fixed_fp32_results.md` give the schedule and boundaries.

The implemented FPGA sequencer performs complete per-call programming and completion handling; the successful control/layout simulation is detailed below. Full programming of these actual control maps entails8,073argument/start writes,870completion/status reads,435interrupt clears and4initial interrupt-enable writes. The concrete plan reads both AP_CTRL and the returned status after each interrupt. Those are transaction counts, not zero-cost operations. Layout/gather/scatter and controller transitions now have a measured conditional term. Additional physical DDR/NoC waits, host/device transport and synchronization remain explicit missing terms. Do not add nominal memory service twice when extending bounded-memory RTL costs to another memory model.

For cached16 alone, the exact frozen integer model atL256 is `166,518,168 + 16.5*S` cycles, where S is the number of matrix elements assigned W8. This useful affine result does not eliminate unknown checking terms or certify feasibility. `reports/goal4_symbolic_cost.md` states when common offsets cancel rankings and why numerical caps and cross-profile choices still need care.

Before a hardware-grounded A/B/C campaign: establish the common physical/control/layout/interface model and numerical bridge; then publish a usable complete-checking estimator and freeze supported latency caps. Final calibration/test and NotInject stay closed. A/B/C receive all common optimizations. Ordinary caching is not claimed as C novelty.

## Reproduction and current work

Reuse successful evidence. Offline analyses:

```powershell
wsl.exe -d Ubuntu --exec /usr/bin/python3 /home/jp19126/Projects/SENTRY/scripts/analyze_hls_characterization.py
wsl.exe -d Ubuntu --exec /usr/bin/python3 /home/jp19126/Projects/SENTRY/scripts/analyze_cached_linear.py
wsl.exe -d Ubuntu --exec /usr/bin/python3 /home/jp19126/Projects/SENTRY/scripts/analyze_fixed_fp32.py
```

Actual dispatches, failures, active sessions, usage monitoring and remaining allowances are in `reports/execution_log.md`. The selected DeepSeek pilot is already complete; this work makes no API calls. No board latency, sustained throughput, energy, full-model hardware accuracy or final method improvement has been measured.


## Current executable ledger and numerical boundary

`python scripts/estimate_detector_cost.py` reuses the frozen cached formulas and the completed same-build15-shape FP32 RTL aggregate. It reproduces uniform integer anchors (cached16:166,518,168/218,422,680cycles; cached32:217,636,248/244,374,936cycles) without scoring a candidate. `results/goal4/detector_cost_ledger.json` retains null complete-check latency and feasibility until integration costs/eligibility are established. Partial HLS loop spans are never substituted for full-call costs. These integer times alone are0.833 to1.092s at200MHz for cached16, already much longer than the earlier CPU/GPU timing pilots; the present design does not establish a software-speed advantage.

The one-real-input411-call C-sim bridge passed exact same-input A8/integer contract checks but differs from Torch risk by0.000296016. The first changed code is layer1 attention-output token131/channel114: identical Q/K/V projections feed attention implementations with slightly different FP32 reductions. Torch quotient42.5 rounds to42, while the service quotient42.500011444 rounds to43. Later quantization amplifies this difference. The unchanged strict decision agrees on this one input; detector numerical acceptance remains unresolved. Preserve prior software results as historical evidence and align the development reference before hardware-grounded method scoring. See `reports/goal4_numerical_bridge.md`.


## Completed primary service ledger

All15 FP32 shapes passed twice with identical cycles. Weighted bus observations are105,393,192 transfer/strobe bytes and112,471,080 full32-bit-beat bytes; the7,077,888 difference is unused lanes on2,359,296 one-byte A8 output writes. Nominal memory service is already included in the measured cycles. Do not add a second complete byte/bandwidth term on top.

| Common profile | W4 integer cycles | W8 integer cycles | W4 known service subtotal | W8 known service subtotal |
|---|---:|---:|---:|---:|
| Cached16 |166,518,168|218,422,680|1306.916ms|1566.438ms|
| Cached32 |217,636,248|244,374,936|1562.506ms|1696.200ms|
| Static-bank64 |128,769,432|142,925,208|1118.172ms|1188.951ms|

Each subtotal adds the frozen integer formula to94,864,971 observed-weighted FP32 cycles at the200MHz objective. Static64 has two passing targets and a passing long reserved point, all repeated twice with zero error against the prior freeze. These are bounded-memory service subtotals, not full checking/board latency. No overlap is credited. The static64 map-dependent integer term is4.5*S; all other declared missing terms stay explicit in `results/goal4/detector_cost_ledger.json`.

An installed-board-file audit confirms that internal command/layout integration can be prepared offline without Linux or a connected board. The fixed L256 plan in `reports/goal4_command_plan.md` contains435 service calls,227 movement commands and2 input/final-control commands. It uses42,225,672 bytes in a proposed64MiB relative arena, which is not an allocated physical DDR region. The bounded mover, control master and sequencer passed offline RTL validation. Control-register completion stubs validated integration control only, not whole-detector arithmetic. Boot image preservation, physical unit revision, host transport and memory/coherency allocation remain deployment facts to establish.


## Completed control/layout integration

The fixed sequencer and data mover passed13 cases with actual generated service-control registers and explicit arithmetic completion stubs. Three valid windows and one rejected nonfinite-risk window each traversed all24 integer+411 FP32 calls. Actual full-window elapsed time was5,402,825 sampled clock intervals; subtracting the435 observed stub intervals totaling3,480 gives5,399,345 conditional control/layout cycles. The legacy field named `inclusive_cycles` is the sampled edge difference with no extra cycle added. Register writes/reads and17,959,268 mover-port bytes match the fixed plan. Error handling, strict threshold equality, max aggregation and fresh-document reset were checked. See `reports/goal4_internal_integration.md`.

The generated integer control RTL is text-identical for all three active profiles. Under the explicitly matched nominal memory-response timing and unchanged fixed schedule, the common control term may therefore be added to each profile's own saved arithmetic latency. This is conditional analytical substitution, not observed complete arithmetic execution. At200MHz it adds26.996725ms:

| Profile | W4 modeled internal path | W8 modeled internal path |
|---|---:|---:|
| Cached16 |1333.912420ms|1593.434980ms|
| Cached32 |1589.502820ms|1723.196260ms|
| Static-bank64 |1145.168740ms|1215.947620ms|

`estimate_detector_cost.py` ingests the saved record, validates its sampled service intervals and traffic/counts, and keeps complete checking latency and feasibility null. Physical DDR/NoC stalls, deployment host preparation/packing/transport, cold setup, integrated resource/timing closure and a qualified numerical reference remain unresolved. No additional HLS/Vivado synthesis or board run was made for this integration.
