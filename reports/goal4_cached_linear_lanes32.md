# Cached linear common-baseline pilot

This is the separately recorded LANES32 cached common engine, kernel variant `weight_cache_pilot`. This artifact does not select the active comparison profiles. Historical tiled-profile evidence and existing cached freezes remain separate; any adopted common engine must be available to all search arms.

## Actual synthesis

Successful C simulation and synthesis: build/linear_hls_cached/lanes32/run.json (2026-09-26T19:41:27.299233+00:00). HLS 2025.2, xcve2802-vsvh1760-2MP-e-S; target 5 ns, uncertainty 0.5 ns, estimated period 4.258 ns. This meets the HLS scheduling target; it is not routed timing or board performance.

Top resource estimates: BRAM_18K=130, DSP=7, FF=10930, LUT=30644, URAM=0. BRAM_18K counts 18-Kib primitives; 65 BRAM36 capacity equivalents do not prove physical pairing.

The report allocates 32 weight-cache banks: 262144 logical bytes and 128 BRAM_18K blocks. The cache holds the actual largest N*K, not a 1024-square matrix. W4 reads each packed source byte once, expands two codes, and uses the same physical cache capacity as W8.

| Stage | Report latency (cycles) | Trip count | II |
|---|---:|---:|---:|
| preload_w8 | 65538..262146 | 65536..262144 | 1 |
| preload_w4 | 32770..131074 | 32768..131072 | 1 |
| initialize | 66..66 | 64..64 | 1 |
| activation_load | 269..269 | 256..256 | 1 |
| compute | 6..8 | 2..4 | 1 |
| store | 80..80 | 64..64 | 1 |

Direct cache reads feed digit arithmetic. The compute report has 2..4 iterations at II=1. Runtime-bound HLS top latency is undefined; generic loop maxima are not model-shape timings.

## Frozen component prediction

Freeze: 2026-09-26T19:52:25.558000+00:00. Source and synthesis evidence, point definitions, and predictions are retained in results/goal4/cached_characterization_lanes32.json.

For declared full tiles, R=M/4, O=N/16, I=K/64, B=NK for W8 or NK/2 for W4. P=B+2 is the HLS preload report latency.

C_nominal = P + R * [2 + O * (150 + I * (920 for W4, 1048 for W8))].

These are the actual selected engine report components. They do not by themselves specify accepted top-start to top-done time.

The cached32 repeated-compute FSM gives +127*R*O*I relative to its report inner terms, with +127*R*O for stores. This differs from the cached16 inner correction.

Preload emits one HLS range request and drains bytes at II=1; the adapter creates bursts. Do not add the prototype's one-read-request-per-code penalty. Reported child latency and sampled start-to-done latency can differ; a complete correction requires this profile's generated-RTL evidence.

## Saved direct RTL observations

| Point | Role | Nominal component | Adjusted component | Measured cycles | Remaining residual |
|---|---|---:|---:|---:|---:|
| target_l32_k256_n256_w4 | target | 277898 | 318538 | [318553, 318553] | [15, 15] |
| target_l32_k256_n1024_w8 | target | 1373706 | 1536266 | [1536281, 1536281] | [15, 15] |
| reserved_l32_k1024_n256_w8 | reserved | 17586306 | 19797122 | [19797137, 19797137] | [15, 15] |

Each accepted run must match the frozen synthesis and point, preserve the DUT RTL, and pass exact integer outputs for both transactions. Measured latencies are actual RTL cycles in a per-port fixed-byte-array, one-outstanding-request, registered-response memory model with no additional DDR delay. Restart intervals include AXI-Lite host restart. They are neither vendor UVM C/RTL Pass records nor physical DDR bandwidth or board measurements.

Saved transaction counters retain burst/beat/byte traffic. No measured values from another physical profile are assigned to this profile.

A faster common baseline requires matching-point timing comparisons and a resource-feasible combined integer/FP32 design. This pilot establishes cache allocation, schedule components and the listed exact-output observations only. No bandwidth_feasible, board throughput, or complete full-model latency claim follows from it.

## Cached32 compute scheduling change

The compiler pipelines only the two/four digit iterations, with report latency6/8. Outer row/output loops remain in the parent FSM. The sampled child duration is4/6; each wait state occupies5/7 edges. Per inner tile, states31-35 contribute5 row checks +68 output checks +64 setup +64*(duration+1) wait +64 commit =521/649 cycles. Add one inner check and525 activation-wait edges to get1047/1175, versus report920/1048: the correction is127 per inner tile.

The output-tile component remains64 initialization +209 store +4 control =277. All three AXI adapter source files equal cached16 by ordinary text comparison. Top/preload FSM boundaries were separately inspected; no development or reserved residual was fitted.

More lanes therefore do not imply a faster engine: this build repeatedly starts a short compute pipeline. Its complete predictions remain predictions until the exact-output RTL observations arrive.

## Separate source-derived completed freeze

Completed before reserved dispatch at 2026-09-26T19:52:25.587587+00:00. The original component freeze and its null residual fields are unchanged.

C_complete = C_adjusted + 17 - 2 = C_adjusted + 15, for this bounded memory model. The 15 is counted from RTL; it is not a fitted mean of the development residuals.

Top edge0 accepts start; engine accepts edge2; range request occurs at edge3; eleven branch setup states lead to preload acceptance at edge15. The following preload-to-row and terminal boundaries produce 17 cycles relative to the actual preload duration. Preload's index starts at zero, advances once per unblocked active edge, and asserts done at index B while consuming the last byte. Its sampled duration is B, two fewer than the HLS report B+2. Both W4 and W8 use this boundary.

Source anchors are listed in the JSON freeze. The range preload has an initially idle weight port and burst supply faster than its one-byte-per-cycle consumer. The model predicts no additional initial or steady-state preload stalls under this exact registered slave; the largest W8 byte count is already covered by a development target. This is a declared memory-model assumption, not physical DDR evidence.

The passive counter samples both acceptance and done at positive clock edges before sequential updates, consistent with the generated FSM. Latency is done_cycle minus start_cycle, without an inclusive extra cycle. AXI-Lite configuration and restart delay are outside this latency and inside the separately recorded start-to-start interval.

| Point | Complete predicted cycles | Time at 5 ns (ms) | Measured minus predicted cycles |
|---|---:|---:|---:|
| target_l32_k256_n256_w4 | 318553 | 1.592765 | [0, 0] |
| target_l32_k256_n1024_w8 | 1536281 | 7.681405 | [0, 0] |
| reserved_l32_k1024_n256_w8 | 19797137 | 98.985685 | [0, 0] |

Reserved validation: `reserved_l32_k1024_n256_w8` passed exact output comparison in both transactions, with cycles [19797137, 19797137]. Measured-minus-complete-prediction errors are [0, 0] cycles. The original nominal component remains 17586306 cycles; its residuals remain [2210831, 2210831] cycles. Neither freeze was changed.

The complete prediction is scoped to the declared bounded RTL memory model. Reserved results assess the already frozen prediction; any failure must remain visible rather than be absorbed into its coefficients. Whole-model sequencer overhead, physical memory integration and board timing remain outside this integer-service prediction.
