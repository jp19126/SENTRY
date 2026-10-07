# Cached linear common-baseline pilot

This is the separately recorded LANES64 cached common engine, kernel variant `weight_cache_static_bank`, source revision `static_bank_index_v1`. This artifact does not select the active comparison profiles. Historical tiled-profile evidence and existing cached freezes remain separate; any adopted common engine must be available to all search arms.

## Actual synthesis

Successful C simulation and synthesis: build/linear_hls_cached_static_bank/lanes64/run.json (2026-09-26T20:28:41.307689+00:00). HLS 2025.2, xcve2802-vsvh1760-2MP-e-S; target 5 ns, uncertainty 0.5 ns, estimated period 4.441 ns. This meets the HLS scheduling target; it is not routed timing or board performance.

Top resource estimates: BRAM_18K=130, DSP=7, FF=19317, LUT=28595, URAM=0. BRAM_18K counts 18-Kib primitives; 65 BRAM36 capacity equivalents do not prove physical pairing.

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

Freeze: 2026-09-26T20:35:49.688420+00:00. Source and synthesis evidence, point definitions, and predictions are retained in results/goal4/cached_characterization_static_bank_lanes64.json.

For declared full tiles, R=M/4, O=N/16, I=K/64, B=NK for W8 or NK/2 for W4. P=B+2 is the HLS preload report latency.

C_nominal = P + R * [2 + O * (150 + I * (342 for W4, 406 for W8))].

These are the actual selected engine report components. They do not by themselves specify accepted top-start to top-done time.

For the bounded direct AXI testbench, source-guided schedule terms are +253*R*O*I for activation reads and +127*R*O for output stores. Their sum with C_nominal is a partial adjusted component. The original component freeze keeps the top/preload residual unresolved.

Preload emits one HLS range request and drains bytes at II=1; the adapter creates bursts. Do not add the prototype's one-read-request-per-code penalty. Reported child latency and sampled start-to-done latency can differ; a complete correction requires this profile's generated-RTL evidence.

## Saved direct RTL observations

| Point | Role | Nominal component | Adjusted component | Measured cycles | Remaining residual |
|---|---|---:|---:|---:|---:|
| target_l64_k256_n256_w4 | target | 129930 | 202826 | [202841, 202841] | [15, 15] |
| target_l64_k256_n1024_w8 | target | 716298 | 1007882 | [1007897, 1007897] | [15, 15] |
| reserved_l64_k1024_n256_w8 | reserved | 7067778 | 11342978 | [11342993, 11342993] | [15, 15] |

Each accepted run must match the frozen synthesis and point, preserve the DUT RTL, and pass exact integer outputs for both transactions. Measured latencies are actual RTL cycles in a per-port fixed-byte-array, one-outstanding-request, registered-response memory model with no additional DDR delay. Restart intervals include AXI-Lite host restart. They are neither vendor UVM C/RTL Pass records nor physical DDR bandwidth or board measurements.

Saved transaction counters retain burst/beat/byte traffic. No measured values from another physical profile are assigned to this profile.

A faster common baseline requires matching-point timing comparisons and a resource-feasible combined integer/FP32 design. This pilot establishes cache allocation, schedule components and the listed exact-output observations only. No bandwidth_feasible, board throughput, or complete full-model latency claim follows from it.

## Static-bank64 source derivation

The actual compute pipeline is flattened across the64 output-row positions and one/two digit phases. Its counter bounds64/128 and three exit registers give sampled67/131 cycles, two fewer than report69/133. It has no blocking condition. This differs from cached32's repeated short subcalls.

Initialization still samples64 cycles. The activation and store stages retain256 byte reads and64 word writes, the same11/14 drain registers, and all three AXI adapters are text-identical to cached16. Under the unchanged bounded slave, their sampled durations are524/209. Thus each inner tile costs524+(67 or131)+4=595/659, and each output tile outside inner work costs64+209+4=277. These are source-derived predictions, not observations copied from another profile.

The actual parent retains the request, eleven setup states, preload wait, row/output checks and terminal state; an independent read-only review confirmed these state boundaries. The new preload children directly signal done at byte index B. Top/preload fixed17 minus the report's extra2 gives the separately frozen15-cycle correction. No new static-bank RTL observation was used in this derivation.

## Separate source-derived completed freeze

Completed before reserved dispatch at 2026-09-26T20:35:49.689479+00:00. The original component freeze and its null residual fields are unchanged.

C_complete = C_adjusted + 17 - 2 = C_adjusted + 15, for this bounded memory model. The 15 is counted from RTL; it is not a fitted mean of the development residuals.

Top edge0 accepts start; engine accepts edge2; range request occurs at edge3; eleven branch setup states lead to preload acceptance at edge15. The following preload-to-row and terminal boundaries produce 17 cycles relative to the actual preload duration. Preload's index starts at zero, advances once per unblocked active edge, and asserts done at index B while consuming the last byte. Its sampled duration is B, two fewer than the HLS report B+2. Both W4 and W8 use this boundary.

Source anchors are listed in the JSON freeze. The range preload has an initially idle weight port and burst supply faster than its one-byte-per-cycle consumer. The model predicts no additional initial or steady-state preload stalls under this exact registered slave; the largest W8 byte count is already covered by a development target. This is a declared memory-model assumption, not physical DDR evidence.

The passive counter samples both acceptance and done at positive clock edges before sequential updates, consistent with the generated FSM. Latency is done_cycle minus start_cycle, without an inclusive extra cycle. AXI-Lite configuration and restart delay are outside this latency and inside the separately recorded start-to-start interval.

| Point | Complete predicted cycles | Time at 5 ns (ms) | Measured minus predicted cycles |
|---|---:|---:|---:|
| target_l64_k256_n256_w4 | 202841 | 1.014205 | [0, 0] |
| target_l64_k256_n1024_w8 | 1007897 | 5.039485 | [0, 0] |
| reserved_l64_k1024_n256_w8 | 11342993 | 56.714965 | [0, 0] |

Reserved validation: `reserved_l64_k1024_n256_w8` passed exact output comparison in both transactions, with cycles [11342993, 11342993]. Measured-minus-complete-prediction errors are [0, 0] cycles. The original nominal component remains 7067778 cycles; its residuals remain [4275215, 4275215] cycles. Neither freeze was changed.

The complete prediction is scoped to the declared bounded RTL memory model. Reserved results assess the already frozen prediction; any failure must remain visible rather than be absorbed into its coefficients. Whole-model sequencer overhead, physical memory integration and board timing remain outside this integer-service prediction.
