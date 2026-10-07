# Goal 4 cost-accounting boundary

The scoped analyzer in `scripts/analyze_hls_characterization.py` reads the fixed three physical profiles and twelve points in shared configuration. It launches no vendor tools and performs no regression fit. The ordinary invocation refreshes saved results; `--freeze-nominal` freezes all twelve report-derived predictions after all three syntheses and before reserved dispatch. `--freeze-calibrated` adds the targeted, passive-diagnostic correction below before reserved dispatch. Later refreshes preserve both freezes and the original nominal errors, and refuse changed point definitions or synthesis evidence.

```powershell
wsl.exe -d Ubuntu --exec /usr/bin/python3 /home/jp19126/Projects/GATE/scripts/analyze_hls_characterization.py --freeze-nominal
```

Use `results/goal4/characterization_summary.{json,csv}` for actual outputs. No measurement or prediction is fabricated for missing reports. An NA/failed simulation remains unusable; successful rows must agree with the point definition and synthesis record. Latency and interval minimum/average/maximum are separate, and total execution cycles are not divided into invented per-call timings.

## Report-derived kernel prediction

For M rows, K inputs and N outputs, R=ceil(M/4), O=N/16 and I=K/64. The supported K/N dimensions divide these tiles exactly. The unchanged runtime engine performs one arithmetic digit phase for W4 and two for W8.

The nested engine synthesis XML defines the minimum row-loop iteration r, minimum output-loop iteration o, and inner-loop iteration range [i4,i8]. Both nested minimum trip counts must be one. Derive h_row=r-o and h_output=o-i4. The nominal schedule component is:

```text
C_nominal = R * [h_row + O * (h_output + I * i_bits)]
C_RTL = C_nominal + top/control/memory-system residual
```

This original expression is not measured total latency and does not assume a zero residual. Lanes4 reports r=2220, o=2218, i4=2068 and i8=3092, giving `R * {2 + O * [150 + I * (1044 + 1024*phases)]}`. Other profiles supply their own reports; total cycles are not divided by the lane ratio. The analyzer snapshots actual stage reports and the source synthesis record, without a hashing or integrity system.

The lanes4 reported stage latencies are initialization 66, activation load 269, weight load 1037, arithmetic 1027/2051 and reduction/store 80 cycles. Generated RTL starts activation and weight loads together and waits for both; inner iteration costs match max(loads)+arithmetic+4. Load/compute/store phases remain sequential. This RTL scheduling evidence permits overlapping those two loads in the nominal model; it does not prove independent physical DDR channels or no contention. The arithmetic loop reports achieved II=1. Its four small multipliers map to LUT logic; the three top-level DSPs occur in load/store address/support modules, not packed arithmetic.

At the declared 200 MHz objective, use 5 ns per cycle for simulated-time accounting. The 4.422 ns HLS period estimate is not a measured operating clock. Synthesis resources are one profile's shared physical cost, not per-layer resources to sum or average by precision. Keep BRAM18K counts distinct from the catalog's BRAM36K units.

Nine 16-row targeted cases support the correction below. Both prediction series are frozen before the three 256-row reserved cases; original signed/relative errors remain. Total RTL cycles alone cannot identify independent bandwidth, packing, fixed-control and compute coefficients. The reserved cases jointly test longer-row scaling and held-out shape/precision combinations, not arbitrary row tails or board memory.

## Initial tiled engine: targeted calibration under bounded AXI

All nine targeted cases passed two exact integer-output comparisons under `hardware/linear_axi_tb.sv`. Its memory model has fixed byte arrays, one outstanding burst per port, a registered first response and no additional DDR delay. These observations are distinct from vendor UVM timings and physical board bandwidth.

One repeated targeted case, lanes4 / 16 rows / 256 inputs / 256 outputs / W4, was observed by the separately compiled passive `hardware/linear_rtl_diagnostic.sv`. Both transactions passed diagnostic call-count/FSM-occupancy checks and retained the original latency of 808,523 cycles. No DUT source, synthesized RTL or testbench handshakes changed. The archived diagnostic source, `run.json` and `simulation.log` are in `results/goal4/rtl_diagnostic/target_l4_k256_n256_w4/`. The analyzer requires `checks_passed: true`, zero diagnostic errors, two exact-output transactions and matching synthesis provenance.

| Stage | HLS overall report cycles | Accepted-start to done cycles | Blocked cycles per call |
|---|---:|---:|---:|
| Initialize | 66 | 64 | 0 |
| Activation load | 269 | 524 | 257 |
| Weight load | 1,037 | 2,060 | 1,025 |
| W4 compute, lanes4 | 1,027 | 1,025 | 0 |
| Reduction/store | 80 | 209 | 131 |

Each observed stage duration equals its HLS overall report latency minus two plus measured stalls. This is an observed boundary convention for these generated stages, not a universal HLS timing rule. Weight stalls came from the read-response condition; store stalls came from the write-response condition. Cause counters can overlap, so they are not added as independent delays. Activation and weight stages launch together; the longer weight stage determines the load wait.

The generated FSM supplies four control cycles per inner iteration and four per output tile when expressed using sampled stage durations. The lanes4 W4 inner term is `2060 + 1025 + 4 = 3089`, versus frozen report term 2068. The output term is `64 + 209 + 4 = 277`, versus frozen report term 150. Corrections are therefore 1021 per inner tile and 127 per output tile. All nine targeted cases confirm the same corrections across the three profiles and both tested precisions; original per-profile arithmetic terms remain in use.

Each row tile contributes one row check and one final output-exit check, already represented by the nominal two cycles. The accepted top start launches the engine through a register; two initial engine-state1 cycles plus the final failing row check give three fixed cycles outside the nominal sum. No additional row-dependent coefficient is inferred from the sixteen-row targets. Diagnostic FSM occupancies sum exactly to accepted-start/done latency.

```text
C_calibrated = C_nominal + 3 + R*O*(127 + 1021*I)
             = 3 + R*[2 + O*(277 + I*(report_inner_bits + 1021))]
```

All nine targeted latencies match exactly. This is calibration evidence; the longer-row reserved cases assess prediction. The correction is restricted to the initial tiled engine, full row tiles, existing physical profiles and unchanged bounded AXI model. It does not transfer to the separate cached engine, arbitrary row tails, board memory or the full detector. Start-to-start intervals include AXI-Lite restart/status reads and are not the engine's minimum initiation interval.

Calibrated predictions were frozen at **2026-09-26T19:18:00.703110+00:00**, while all reserved run records were absent. The original nominal freeze remains at **2026-09-26T18:27:12.811839+00:00**. `results/goal4/characterization_summary.json` stores separate `calibrated_freeze`, diagnostic transactions, nine target evidence paths and both prediction/error series; CSV adds calibrated columns without replacing nominal columns.

| Reserved case, 256 rows | Original nominal cycles | Frozen calibrated cycles |
|---|---:|---:|
| lanes4, 256->256, W8 | 12,818,560 | 17,130,627 |
| lanes8, 256->1024, W8 | 34,496,640 | 51,744,899 |
| lanes16, 1024->256, W8 | 25,647,232 | 42,505,347 |

The offline freeze command completed once with nine measured targets and both freeze flags true:

```powershell
& 'C:\research\GATE\.venv\Scripts\python.exe' 'scripts/analyze_hls_characterization.py' --freeze-calibrated
```

Ordinary invocations refresh saved observations while retaining both freezes. A first calibrated freeze refuses any prior reserved dispatch. Original and calibrated signed/relative errors remain separate when reserved observations are recorded.

## Packing, traffic and local storage

Source-level valid activation loads per call total M*K*O bytes because activation tiles are reloaded for every output tile. Weight helper accesses total R*N*K bytes in either mode. W4's distinct packed-byte footprint is R*N*K/2, but adjacent codes can request the same byte twice. Neither footprint nor port separation establishes half the AXI beats or bandwidth time. Output writes contain 4*M*N bytes of int32 results.

The first direct target measured 65,536 activation and 262,144 weight single-beat 32-bit reads, plus 4,096 int32 writes. Thus a byte request can transfer a full 32-bit bus beat; W4 packing alone does not remove the duplicated helper requests in this initial engine.

Logical scratch per profile is 256 activation bytes + 1024 unpacked weight-code bytes + 256*lanes partial-sum bytes. Local weight scratch does not shrink for W4. At one 256-token window across the 24 selected linears, source accesses total 48 MiB activation loads, 192 MiB W8 weight loads and 9 MiB int32 writes. W4's distinct weight footprint per repeated traversal is 96 MiB, while helper accesses remain duplicated. The generated 32-bit ports are not a measured DDR rate.

The max_rows=256 engine covers the primary point. Larger sequences/batches need calls of at most 256 rows and their actual call costs; attention still spans the full intended sequence. Do not infer the final 512-token/batch-32 workload from this row limit.

## Complete-detector ledger

The fixed inventory is 16 matrices 256->256, four 256->1024 and four 1024->256. For one padded 256-token window these are analytical operation/storage counts, not FPGA timings:

| Component | Known quantity | Hardware term still needed |
|---|---|---|
| Selected integer linears | 24 calls for this unfused schedule; 805,306,368 mathematical MACs | Runtime profile cycles, actual call/transfer schedule |
| FP32 attention QK and AV | 134,217,728 MACs | Realizable FP32 engines, buffering, cycles/resources |
| Attention softmax | 1,048,576 score elements across four layers/four heads | Exp/reduction/division implementation and cycles |
| GELU | 1,048,576 elements | FP32 implementation, accuracy, cycles/resources |
| LayerNorm | 9 normalizations of 256x256; 589,824 elements | Reduction and reciprocal-square-root schedule |
| Residual addition | 524,288 FP32 additions | Buffering and execution cost |
| Linear output rescale/bias | 2,359,296 output elements | Conversion, scale multiplication and bias |
| Selected-linear input quantization | 2,359,296 input elements for separate wrappers | Round/clip/conversion; explicit validation if shared QKV conversion is reused |
| Embeddings | 7,945,728 embedding-module parameters, including normalization | Lookup/add/normalization placement and traffic |
| Pooler/classifier | 65,536 + 512 MACs per window; tanh and two-class decision | Fixed FP32 logic and cycles |
| Fixed FP32 parameter payload | 32,101,384 bytes | Resident placement versus reload schedule |
| Selected packed weights | W8 3,145,728 bytes; W4 1,572,864 bytes | Actual packing, alignment and transfer behavior |
| Active FP32 scales | 36,960 bytes | Placement and access costs |

Parameter evidence is `results/goal2/training_run.json`; arithmetic/packing evidence is `gate_quant.py`, `results/goal3/calibration.json` and `reports/quantization_sensitivity.md`. The saved QAT checkpoint is FP32 software state, not a packed hardware image.

A full checking estimate must include host preprocessing + transport/control + selected-linear kernels + FPGA conversion/fixed-FP32 work + synchronization/aggregation. Goal 4 permits labelled analytical estimates for unimplemented blocks, but they require a realizable full-FPGA schedule, declared arithmetic/resources and justified rates. Unknown terms remain unknown. CPU execution of these detector blocks would not meet the proposed full-FPGA detector.

`results/goal2/timing_pilot.json` measures 32 single-window CUDA checks after eight warm-ups: preparation 0.500547 ms, CUDA input transfer 0.181972 ms, complete inference/aggregation 5.100134 ms and result transfer 0.053753 ms. `reports/software_profile.md` records CPU FP32/8-thread preparation 0.611644 ms and full inference/aggregation 7.863903 ms. Preparation includes tokenization/windowing/padding and may be used only as a provisional host term for the same host/path/workload. Native-Windows timing does not establish the WSL/deployed host cost. CUDA transfers are not FPGA transfers; full CPU/GPU inference timings do not isolate the remaining FP32 blocks.

No board bandwidth, achieved frequency, post-route resource use, sustained workload, energy or complete-detector latency is established here. Kernel-only fit against PL resource limits cannot certify full-system feasibility. Goal 4 and the hardware-grounded method comparison remain incomplete.
