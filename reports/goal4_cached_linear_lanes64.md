# Cached linear common-baseline pilot

This is the separately recorded LANES64 cached common engine, not a new C method. All search arms receive the same active hardware space. Historical tiled-profile evidence and the existing LANES16 freezes remain separate.

## Actual synthesis

Successful C simulation and synthesis: build/linear_hls_cached/lanes64/run.json (2026-09-26T19:44:10.796226+00:00). HLS 2025.2, xcve2802-vsvh1760-2MP-e-S; target 5 ns, uncertainty 0.5 ns, estimated period 4.258 ns. This meets the HLS scheduling target; it is not routed timing or board performance.

Top resource estimates: BRAM_18K=130, DSP=9, FF=19744, LUT=87741, URAM=0. BRAM_18K counts 18-Kib primitives; 65 BRAM36 capacity equivalents do not prove physical pairing.

The report allocates 64 weight-cache banks: 262144 logical bytes and 128 BRAM_18K blocks. The cache holds the actual largest N*K, not a 1024-square matrix. W4 reads each packed source byte once, expands two codes, and uses the same physical cache capacity as W8.

| Stage | Report latency (cycles) | Trip count | II |
|---|---:|---:|---:|
| preload_w8 | 65538..262146 | 65536..262144 | 1 |
| preload_w4 | 32770..131074 | 32768..131072 | 1 |
| initialize | 66..66 | 64..64 | 1 |
| activation_load | 269..269 | 256..256 | 1 |
| compute | 69..133 | 64..128 | 1 |
| store | 80..80 | 64..64 | 1 |

Direct cache reads feed digit arithmetic. The compute report has 64..128 iterations at II=1. Runtime-bound HLS top latency is undefined; generic loop maxima are not model-shape timings.

## Frozen component prediction

Freeze: NOT FROZEN. Source and synthesis evidence, point definitions, and predictions are retained in results/goal4/cached_characterization_lanes64.json.

For declared full tiles, R=M/4, O=N/16, I=K/64, B=NK for W8 or NK/2 for W4. P=B+2 is the HLS preload report latency.

C_nominal = P + R * [2 + O * (150 + I * (342 for W4, 406 for W8))].

These are the actual selected engine report components. They do not by themselves specify accepted top-start to top-done time.

This profile has no reviewed AXI/control correction; complete predictions remain null.

Preload emits one HLS range request and drains bytes at II=1; the adapter creates bursts. Do not add the prototype's one-read-request-per-code penalty. Reported child latency and sampled start-to-done latency can differ; a complete correction requires this profile's generated-RTL evidence.

## Saved direct RTL observations

| Point | Role | Nominal component | Adjusted component | Measured cycles | Remaining residual |
|---|---|---:|---:|---:|---:|
| target_l64_k256_n256_w4 | target | 129930 | None | not_run | None |
| target_l64_k256_n1024_w8 | target | 716298 | None | not_run | None |
| reserved_l64_k1024_n256_w8 | reserved | 7067778 | None | not_run | None |

Each accepted run must match the frozen synthesis and point, preserve the DUT RTL, and pass exact integer outputs for both transactions. Measured latencies are actual RTL cycles in a per-port fixed-byte-array, one-outstanding-request, registered-response memory model with no additional DDR delay. Restart intervals include AXI-Lite host restart. They are neither vendor UVM C/RTL Pass records nor physical DDR bandwidth or board measurements.

Saved transaction counters retain burst/beat/byte traffic. No measured values from another physical profile are assigned to this profile.

A faster common baseline requires matching-point timing comparisons and a resource-feasible combined integer/FP32 design. This pilot establishes cache allocation, schedule components and the listed exact-output observations only. No bandwidth_feasible, board throughput, or complete full-model latency claim follows from it.
