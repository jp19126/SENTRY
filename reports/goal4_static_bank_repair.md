# Static-bank indexing repair and isolated evidence

Initially prepared 2026-09-26T20:15:58.448042+00:00 as an inactive candidate. Root subsequently authorized and ran the twelfth and final initial synthesis. The original cached source, numeric profile configuration and historical evidence remain separate; this report does not select the active comparison profile.

The separate header is `hardware/linear_engine_cached_static_bank.hpp`; its callable template is `gate::linear_engine_cached_static_bank<LANES,4,16,64,256>`, and its compile-time identity is `GATE_LINEAR_STATIC_BANK_REVISION="static_bank_index_v1"`. It preserves the full-matrix byte-code cache and all arithmetic/preload/activation/store loops. Only the header/function identity and the cache-address expression differ from the active source. Reversing those declared edits reproduced the original source exactly by ordinary text comparison.

## Exact address equivalence

For P=LANES in{16,32,64}, accepted K in{256,1024} is divisible by P. Every inner_base is a multiple of64, and k_base is a multiple of P. For a valid unrolled lane:

```text
new = [(out_base+o)*(K/P)+(inner_base+k_base)/P]*P+lane
    = (out_base+o)*K+inner_base+k_base+lane
    = old.
new modulo P = lane.
```

All divisions are exact and all operands nonnegative. The largest index is 262143; the largest intermediate cache_row is 16383 at P16. No signed 32-bit overflow is introduced. Allowed row counts 1..256 and either weight precision use the same address expression, so neither changes this proof. The existing shape guards and fixed tile assertions are retained. W4 nibble expansion, W8 signed-high/unsigned-low digit reconstruction and accumulation order are unchanged.

An actual bounded Python integer check enumerated every cache access position in all three accepted matrix shapes for all three profiles: 1,769,472 address pairs. Every pair agreed, every bank remainder equalled its lane, and every address stayed in range. This checks indexing, not newly synthesized arithmetic or timing.

| Lanes | Inner K | Outputs N | Addresses checked | Largest index |
|---:|---:|---:|---:|---:|
| 16 | 256 | 256 | 65536 | 65535 |
| 16 | 256 | 1024 | 262144 | 262143 |
| 16 | 1024 | 256 | 262144 | 262143 |
| 32 | 256 | 256 | 65536 | 65535 |
| 32 | 256 | 1024 | 262144 | 262143 |
| 32 | 1024 | 256 | 262144 | 262143 |
| 64 | 256 | 256 | 65536 | 65535 |
| 64 | 256 | 1024 | 262144 | 262143 |
| 64 | 1024 | 256 | 262144 | 262143 |

## Evidence motivating this repair

Actual cached32 compute RTL carries the runtime low five product bits into the child: parent `gate_linear_top_linear_engine_cached_32_4_16_64_256_s.v` lines 4640 and 10775 pass `(out_base+o)*inner` modulo32 as `empty`. Child `...Pipeline_VITIS_LOOP_90_13_VITIS_LOOP_92_14.v` lines 2923-2960 select bank outputs with that signal; lines 10584 onward select bank addresses. Those bits are mathematically zero for every accepted shape, yet they remain dynamic in the generated implementation.

Its actual HLS compute report lists 32 address muxes, each 33 inputs, 13 bits and 336 LUT, plus 32 read selectors of 88 LUT each: 13,568 estimated LUT for those identified components. This establishes unnecessary general bank routing in that build. It does not prove the cause of cached64 scheduling time or predict the repaired area/clock/cycles. Cached32's separate repeated compute-subcall overhead may remain.

At the latest read-only check, cached64 HLS process 164089 was alive at about 29 minutes, 88.6% average CPU and about 499 MiB RSS. WSL reported about 20 GiB available. A one-second vmstat sample showed 4 KiB/s swap-in and 0 swap-out, with no I/O wait. The last log event remained compute-loop scheduling; no compute schedule/report or error had appeared. Active CPU does not establish convergence. Nothing was interrupted.

## Isolated build and actual synthesis

`scripts/build_linear_hls.py --profile lanes64 --static-bank --synthesize` uses `build/linear_hls_cached_static_bank/lanes64`, the external runtime `gate_linear_top` interface, and source identity `weight_cache_static_bank` / `static_bank_index_v1`. It snapshots the header, integer helpers, testbench, top, interface Tcl and build driver. The ordinary cached build and its source remain reproducible. An existing static-bank run record prevents an accidental additional synthesis.

The agent's offline `--emit` check passed before root's vendor launch: Linux paths, exact source copy, source identity and absence of vendor run/log files were checked. The 470 original build/archive/config/source files retained their size and modification time. Root's actual attempt12 then passed C simulation and synthesis, recorded at2026-09-26T20:28:41.307689+00:00. HLS reports130 BRAM18K,7 DSP,19317 FF,28595 LUT and4.441 ns estimated period, against5 ns target and0.5 ns uncertainty. Original cached64 reported87741 LUT and4.258 ns; these are HLS estimates, not routed timing or board measurements. Root collected synthesis durations1m04s for the repair and33m26s for the original.

The actual source, settings, top, testbench, driver, vendor log and reports are archived in `results/goal4/lanes64_cached_static_bank/`. The original archive remains `results/goal4/lanes64_cached/`.

## Predictions and runtime boundary

`scripts/analyze_cached_linear.py --profile lanes64 --static-bank --freeze-nominal --freeze-complete` saved separate freezes in `results/goal4/cached_characterization_static_bank_lanes64.json` before any target or reserved dispatch. The report is `reports/goal4_cached_linear_static_bank_lanes64.md`. Actual report components are row2, output150, inner342/406 for W4/W8. The new flattened compute pipeline has64/128 iterations and three exit registers, giving sampled67/131 cycles. The reviewed bounded AXI stages give initialization64, activation524 and store209, so the complete prediction is:

```text
B + 17 + R * [2 + O * (277 + I * (595 for W4 or659 for W8))]
R=M/4, O=N/16, I=K/64; B=NK/2 for W4, NK for W8.
```

All three AXI adapter files equal cached16 by ordinary text comparison. Actual new top/preload state boundaries were separately inspected, with an independent read-only parent-FSM review. The15-cycle correction relative to the report components is17 controller cycles minus the preload report's extra2. No static-bank observed residual was used. This predicts202841 cycles for the small W4 target,1007897 for the wide W8 target and11342993 for the tall W8 reserved point. Runtime observations and any errors are retained separately in the generated characterization report.

`scripts/run_linear_rtl.py --static-bank --point <declared lanes64 point>` uses separate `build/linear_rtl_cached_static_bank/lanes64` and `results/goal4/rtl_cached_static_bank/<point>` paths and requires the matching source revision and synthesis. No source-derived complete prediction is a physical DDR, full detector, routed timing or board result.

The new `active_linear_variants(config)` helper reads an optional `hardware.hls.cached_linear.profile_kernel_variants` mapping, separate from numeric profile dictionaries. Missing mapping preserves the original cached variants; selecting static-bank64 requires root's explicit configuration change after validation. There are still at most three active physical profiles. Original lanes16/32 frozen definitions were reproduced exactly in memory after adding the isolated routes. No vendor command was launched by this agent.
