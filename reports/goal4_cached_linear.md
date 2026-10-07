# Cached linear common-baseline pilot

This is the separately recorded LANES16 cached common engine, not a new C method. All search arms receive the same active hardware space. Historical tiled-profile evidence and the existing LANES16 freezes remain separate.

## Actual synthesis

Successful C simulation and synthesis: build/linear_hls_cached/lanes16/run.json (2026-09-26T19:05:29.956550+00:00). HLS 2025.2, xcve2802-vsvh1760-2MP-e-S; target 5 ns, uncertainty 0.5 ns, estimated period 4.258 ns. This meets the HLS scheduling target; it is not routed timing or board performance.

Top resource estimates: BRAM_18K=130, DSP=7, FF=8149, LUT=15443, URAM=0. BRAM_18K counts 18-Kib primitives; 65 BRAM36 capacity equivalents do not prove physical pairing.

The report allocates 16 weight-cache banks: 262144 logical bytes and 128 BRAM_18K blocks. The cache holds the actual largest N*K, not a 1024-square matrix. W4 reads each packed source byte once, expands two codes, and uses the same physical cache capacity as W8.

| Stage | Report latency (cycles) | Trip count | II |
|---|---:|---:|---:|
| preload_w8 | 65538..262146 | 65536..262144 | 1 |
| preload_w4 | 32770..131074 | 32768..131072 | 1 |
| initialize | 66..66 | 64..64 | 1 |
| activation_load | 269..269 | 256..256 | 1 |
| compute | 261..517 | 256..512 | 1 |
| store | 80..80 | 64..64 | 1 |

Direct cache reads feed digit arithmetic. The compute report has 256..512 iterations at II=1. Runtime-bound HLS top latency is undefined; generic loop maxima are not model-shape timings.

## Frozen component prediction

Freeze: 2026-09-26T19:20:13.647345+00:00. Source and synthesis evidence, point definitions, and predictions are retained in results/goal4/cached_characterization.json.

For declared full tiles, R=M/4, O=N/16, I=K/64, B=NK for W8 or NK/2 for W4. P=B+2 is the HLS preload report latency.

C_nominal = P + R * [2 + O * (150 + I * (534 for W4, 790 for W8))].

These are the actual selected engine report components. They do not by themselves specify accepted top-start to top-done time.

For the bounded direct AXI testbench, source-guided schedule terms are +253*R*O*I for activation reads and +127*R*O for output stores. Their sum with C_nominal is a partial adjusted component. The original component freeze keeps the top/preload residual unresolved.

Preload emits one HLS range request and drains bytes at II=1; the adapter creates bursts. Do not add the prototype's one-read-request-per-code penalty. Reported child latency and sampled start-to-done latency can differ; a complete correction requires this profile's generated-RTL evidence.

## Saved direct RTL observations

| Point | Role | Nominal component | Adjusted component | Measured cycles | Remaining residual |
|---|---|---:|---:|---:|---:|
| target_l16_k256_n256_w4 | target | 179082 | 251978 | [251993, 251993] | [15, 15] |
| target_l16_k256_n1024_w8 | target | 1109514 | 1401098 | [1401113, 1401113] | [15, 15] |
| target_l16_k1024_n256_w4 | target | 687498 | 954698 | [954713, 954713] | [15, 15] |
| reserved_l16_k1024_n256_w8 | reserved | 13359234 | 17634434 | [17634449, 17634449] | [15, 15] |

Each accepted run must match the frozen synthesis and point, preserve the DUT RTL, and pass exact integer outputs for both transactions. Measured latencies are actual RTL cycles in a per-port fixed-byte-array, one-outstanding-request, registered-response memory model with no additional DDR delay. Restart intervals include AXI-Lite host restart. They are neither vendor UVM C/RTL Pass records nor physical DDR bandwidth or board measurements.

The saved LANES16 W4 256-by-256 target uses 512 weight bursts, 8192 beats and 32768 source bytes. Its two 251993-cycle transactions leave 15 cycles against the original 251978-cycle adjusted component; this is not a universal correction.

A faster common baseline requires matching-point timing comparisons and a resource-feasible combined integer/FP32 design. This pilot establishes cache allocation, schedule components and the listed exact-output observations only. No bandwidth_feasible, board throughput, or complete full-model latency claim follows from it.

## Separate source-derived completed freeze

Completed before reserved dispatch at 2026-09-26T19:27:33.017027+00:00. The original component freeze and its null residual fields are unchanged.

C_complete = C_adjusted + 17 - 2 = C_adjusted + 15, for this bounded memory model. The 15 is counted from RTL; it is not a fitted mean of the development residuals.

Top edge0 accepts start; engine accepts edge2; range request occurs at edge3; eleven branch setup states lead to preload acceptance at edge15. The following preload-to-row and terminal boundaries produce 17 cycles relative to the actual preload duration. Preload's index starts at zero, advances once per unblocked active edge, and asserts done at index B while consuming the last byte. Its sampled duration is B, two fewer than the HLS report B+2. Both W4 and W8 use this boundary.

Source anchors are listed in the JSON freeze. The range preload has an initially idle weight port and burst supply faster than its one-byte-per-cycle consumer. The model predicts no additional initial or steady-state preload stalls under this exact registered slave; the largest W8 byte count is already covered by a development target. This is a declared memory-model assumption, not physical DDR evidence.

The passive counter samples both acceptance and done at positive clock edges before sequential updates, consistent with the generated FSM. Latency is done_cycle minus start_cycle, without an inclusive extra cycle. AXI-Lite configuration and restart delay are outside this latency and inside the separately recorded start-to-start interval.

| Point | Complete predicted cycles | Time at 5 ns (ms) | Measured minus predicted cycles |
|---|---:|---:|---:|
| target_l16_k256_n256_w4 | 251993 | 1.259965 | [0, 0] |
| target_l16_k256_n1024_w8 | 1401113 | 7.005565 | [0, 0] |
| target_l16_k1024_n256_w4 | 954713 | 4.773565 | [0, 0] |
| reserved_l16_k1024_n256_w8 | 17634449 | 88.172245 | [0, 0] |

Reserved validation: `reserved_l16_k1024_n256_w8` passed exact output comparison in both transactions, with cycles [17634449, 17634449]. Measured-minus-complete-prediction errors are [0, 0] cycles. The original nominal component remains 13359234 cycles; its residuals remain [4275215, 4275215] cycles. Neither freeze was changed.

The complete prediction is scoped to the declared bounded RTL memory model. Reserved results assess the already frozen prediction; any failure must remain visible rather than be absorbed into its coefficients. Whole-model sequencer overhead, physical memory integration and board timing remain outside this integer-service prediction.
