# Bounded VEK280 characterization plan

Target: installed production VEK280 1.2 board definition, part xcve2802-vsvh1760-2MP-e-S. The Vivado catalog query succeeded; physical revision verification is deferred until connection. Use Vivado PL IP flow, 200 MHz target and explicit 0.5 ns clock uncertainty. These are declared design objectives, not achieved frequency. Initial maximum PL allocation is 70 percent of catalog resources, reserving 30 percent for integration; exact totals and budgets are in shared config.

Three physical engine profiles use 4, 8 or 16 arithmetic lanes with common tile rows/outputs/inner = 4/16/64 and maximum 256 rows. They preserve one shared runtime-programmable engine per profile, including runtime W4/W8. AXI4 activation/weight/output master bundles share an AXI4-Lite control bundle. Port separation is not an assertion of independent DDR channels or overlapped traffic. Actual system integration and host transport remain unresolved.

First execute the existing exact-reference vendor C-simulation at lanes4. It covers W4 and W8 for each of the three real encoder matrix shapes, a full row tile plus tail, and one invalid dimension. Then synthesize the shared engines. There is no need to repeatedly synthesize unchanged RTL for different runtime arguments.

| Profile | Target 256->256 | Target 256->1024 | Target 1024->256 | Reserved |
|---|---|---|---|---|
| lanes4 | W4 | W8 | W4 | 256->256 W8 |
| lanes8 | W8 | W4 | W8 | 256->1024 W8 |
| lanes16 | W4 | W8 | W4 | 1024->256 W8 |

Nine targeted runtime cases use 16 rows; three reserved cases use 256 rows. Reserve the latter for assessing the estimator, not initial fitting. Only three shared-engine syntheses are planned, below the original at-most-12 synthesis allowance; failures and necessary reruns must be recorded. Runtime-dependent latency may be unknown in HLS reports. Point-selected RTL co-simulation is needed for per-case cycle observations; do not constant-specialize the engine to manufacture distinct physical costs. HLS implementation estimates and simulated cycles are separate from physical board measurements.

This plan covers only the integer linear kernel. A complete-model estimator still needs fixed FP32 operations, embeddings/attention, memory/conversion work, actual system integration and host costs. No complete-model latency or Goal 4 completion is implied by the first successful kernel run.
