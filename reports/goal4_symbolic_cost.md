# Symbolic precision-cost audit

Read-only derivation, 2026-09-26. No search, candidate scoring, vendor run or feasibility-gate change was performed. This report derives properties of the separately frozen cached16, cached32 and static-bank64 models; it does not replace complete checking latency with integer-kernel time. Runtime validation retains its own evidence in each characterization report. The static-bank64 predictions below were frozen before dispatch; both targets now passed exact outputs and matched their predictions in both repetitions, and the reserved point has also passed twice at11,342,993cycles. The original cached64 synthesis remains historical and is not substituted for the repaired source.

## Exact cached16 dependence at length256

Let x_j=1 for a W8 matrix and 0 for W4, and let S=sum_j(x_j*N_j*K_j). Q/K/V share one precision decision but remain three matrix calls. There are 24 calls: 16 matrices of 256x256, four of 1024x256 and four of 256x1024. Therefore sum(NK)=3,145,728 and sum(N)=9,216, from `results/goal2/training_run.json`.

The source-derived complete formula in `results/goal4/cached_characterization.json` and `reports/goal4_cached_linear.md` is, with R=M/4, O=N/16 and I=K/64:

```text
C_j(x) = NK*(1+x)/2 +17 +R*[2+O*(277+I*(787+256*x))].
```

Here the extra W8 preload is NK/2 cycles, and the second digit phase adds256 cycles per inner tile. The reviewed top/preload boundary is the same for both bits. For M=256, R=64:

```text
C_j(x) = (795/16)*NK +1108*N +145 +(33/2)*NK*x.
C_integer(S) =166,518,168 +16.5*S cycles.
```

Thus the model's all-W4 integer total is 166,518,168 cycles and all-W8 is 218,422,680 cycles. These are 832.59084ms and 1,092.1134ms at the 5ns objective, not complete-detector or measured board latencies. This is an algebraic application of the frozen formula to the 24-call schedule, not an observed whole-window run. The saved targeted observations and reserved validation retain their original roles.

The sixteen group decisions have these contributions:

| One group changed W4 to W8 | Added S | Added cached16 cycles |
|---|---:|---:|
| One layer's Q/K/V |196,608|3,244,032|
| Attention output |65,536|1,081,344|
| Feed-forward input |262,144|4,325,376|
| Feed-forward output |262,144|4,325,376|

Locations do not affect this cost model: maps with equal S tie in integer cost while their protection scores can differ. Matrix count alone is insufficient; QKV and feed-forward decisions have different weights.

For another profile with unchanged tile sizes and reviewed bit-independent corrections, let d_p be its **actual** reported inner-tile W8-minus-W4 latency. The coefficient is:

```text
beta_p =1/2 +d_p/16;
C_integer,p(S) =A_p +beta_p*S.
```

Source scheduling would give d_p=4096/p if its digit loop keeps II1 and all phase boundaries remain unchanged. The original conditional expectation was beta32=8.5 and beta64=4.5. Actual cached32 reports establish the 8.5 slope but expose a larger fixed cost. The separate static-bank64 synthesis and generated RTL now establish the 4.5 slope and its own intercept, as derived below; both targets and the reserved case passed twice with exact pre-dispatch predictions. These are not complete-checking cost-table entries. Different initialization/store costs change A_p; a bit-dependent branch or stall can also change the slope or break this two-parameter form.

## Actual cached32 counterexample to simple lane scaling

After the initial audit, actual cached32 synthesis became available and `results/goal4/cached_characterization_lanes32.json` froze its predictions before any target/reserved dispatch. Its report inner terms are920/1048, with reviewed correction127 per inner tile, yielding actual-model inner terms1047/1175. Unlike cached16, HLS invokes a short compute pipeline64 times per inner tile. More lanes therefore reduce the precision slope while increasing the intercept.

The complete source-derived integer predictions at length256 give:

```text
C_32(S) =217,636,248 +8.5*S.
C_32(S)-C_16(S) =51,118,080 -8*S.
```

Over0<=S<=3,145,728, this difference stays positive:51,118,080 cycles at all-W4 and25,952,256 at all-W8. The formal crossover S=6,389,760 is outside the permitted map space. Thus cached16 dominates cached32 in this bounded-memory integer model at the same clock. Unknown profile-specific full-system costs still cannot be silently cancelled. Cached32 now has two exact-output target observations at 318,553 and 1,536,281 cycles and a reserved observation at 19,797,137 cycles, each repeated twice and exactly matching the prior complete freeze. Those saved RTL observations validate the scoped prediction; they are not full-check measurements.

## Source-derived static-bank64 formula

Attempt12 uses the separate `weight_cache_static_bank` / `static_bank_index_v1` source. Its nominal and complete freezes are in `results/goal4/cached_characterization_static_bank_lanes64.json`; source, controller and memory-model evidence are described in `reports/goal4_cached_linear_static_bank_lanes64.md`. The original cached64 source is archived separately and has no borrowed complete prediction.

The new flattened compute loop gives sampled 67/131 cycles for W4/W8, rather than cached32's repeated short calls. Under the same bounded AXI model, the complete per-matrix formula is:

```text
C_j,64(x) = NK*(1+x)/2 +17 +R*[2+O*(277+I*(595+64*x))].
```

At M=256, the 24-call integer total is exactly:

```text
C_64_static(S) =128,769,432 +4.5*S cycles.
C_16(S)-C_64_static(S) =37,748,736 +12*S.
C_32(S)-C_64_static(S) =88,866,816 +4*S.
```

The all-W4 and all-W8 static-bank64 totals are 128,769,432 and 142,925,208 cycles, respectively: 643.84716 ms and 714.62604 ms at the common 5 ns objective. These are algebraic integer-service totals, not a complete detector timing or a measured whole-window run. The 4.5 coefficient comprises 0.5 cycles per additional packed weight byte element and 4 cycles per element from the second digit phase across 64 row tiles.

Both differences are strictly positive throughout the allowed 0 <= S <= 3,145,728 range. Static-bank64 therefore conditionally dominates cached16 and cached32 for every identical precision map in this frozen integer-service model at the common clock; neither difference has a nonnegative crossover. Both static-bank64 targets now pass exact outputs and match the prior frozen 202,841 and 1,007,897 cycles in both repetitions. Its reserved case now passes exact outputs in both repetitions at11,342,993cycles, exactly matching the prior prediction. It does not establish resource, bandwidth or complete-system feasibility. A profile-specific unresolved offset or achieved clock can still change full-check ranking.

The selected active set is three physical shapes: cached16, cached32 and static-bank64. Root selected static-bank64 in the explicit profile-to-source mapping after both target cases passed; its reserved check has now also passed twice. Source identity is a separate profile-keyed mapping; no fourth active profile is implied. The root-owned configuration remains the authority.

## What can cancel

Suppose a valid fixed-profile full check really satisfies:

```text
T_p(m) =U_p +a_p +b_p*S(m),   b_p>0,
```

where a_p and b_p convert accepted integer cycles to time, and U_p is all genuinely map-independent FP32, layout, host, controller and other work. U_p remains unknown and is not assigned zero.

Within this fixed feasible profile, U_p cancels from pairwise latency differences. Ranking by S and latency tie-breaking are then exact. It does **not** cancel from a previously fixed numerical cap tau: membership requires a_p+b_p*S <=tau-U_p. Unknown U_p can change membership and even whether any map fits. Latency ratios and claimed percentage speedups also depend on U_p.

An anchor-interpolated symbolic cap is different. For a prospectively fixed lambda in [0,1], define tau_p(lambda)=(1-lambda)*T_p(W4)+lambda*T_p(W8). Under the stated affine model:

```text
T_p(m)<=tau_p(lambda)  iff  S(m)<=lambda*3,145,728.
```

Both the unknown offset and any common positive slope cancel. This is a truthful normalized position between two full-check anchors, not a known millisecond cap. Defining a separate such cap per profile does not generally produce one common absolute cap across profiles. Similarly, a cap written symbolically as U+c cancels a **single shared** U but leaves its absolute time unknown.

Across profiles, offsets cancel only if they are demonstrably the same. In general:

```text
T_p-T_q =(U_p-U_q)+(a_p-a_q)+(b_p-b_q)*S.
S_cross =(U_q+a_q-U_p-a_p)/(b_p-b_q), when b_p!=b_q.
```

Unknown profile-specific offsets or achieved clocks can move this crossover. If a_p<=a_q and b_p<=b_q with a common U and at least one strict inequality, p dominates over S>=0; there is no positive crossover. The three source-derived integer relations above are now specified. Static-bank64 reserved validation is now complete; all complete-system relations remain separate requirements. Resource/timing/bandwidth eligibility must be established independently; additive cancellation cannot make an unavailable profile feasible.

## Conditions and exceptions

- **The frozen memory boundary matters.** Cached16 assumes an initially idle weight port, registered bounded responses and no physical DDR or cross-port contention. The preload consumes one byte per cycle; its burst supplier is fast enough in this model. Physical weight-read work changes by S/2 bytes, so unknown DDR waits are not automatically part of U.
- **Burst counts can remain affine under a specified layout.** All present matrix sizes give W4/W8 byte counts divisible by 64 and 4096. With suitably aligned bases and the observed maximum 16-beat 32-bit bursts, the additional full bursts would be S/128. A constant cost per byte/burst could alter the positive slope while preserving S ranking. Real startup, page/row mapping, alignment, arbitration, cache state and overlapping service must first satisfy that model; they are not established by useful-byte counts. Bit-dependent fixed setup per W8 call would depend on the number of W8 calls as well as S, breaking the stated single-variable formula.
- **No hidden reuse or tails.** Every integer call reloads its weight matrix. M256,N256/1024,K256/1024 are full tiles with even packed sizes. Tail rows, other shapes, persistent cache hits or a changed schedule need a new derivation. Length512 uses two integer row calls, not this coefficient; batch/examples and multiple document windows must use their actual fixed counts.
- **Common work is a conditional property.** The revised 411-call FP32 schedule, 24 integer dispatches and declared layout operations are structurally bit-independent. Full register programming gives fixed write counts, but a compact precision-dependent address layout or shadow-register optimization needs its own accounting. Warm checking can treat static weight packing/loading separately; cold per-map upload is precision-dependent. Fixed-length valid arithmetic and no early-exit/window-skipping schedule are also required. Nonfinite failures or changed detector execution cannot be silently treated as equal work.
- **Resources differ from costs.** W4 and W8 use the same instantiated expanded cache and arithmetic engine within one profile, while packed external storage varies with S. Combined system resource fit, external allocation and timing remain separate requirements. The new common FP32 panel's resources cannot be borrowed from its archived smaller-panel build.

## Consequence for Goal5

This audit establishes conditional ranking invariants for the frozen integer models, including the new static-bank64 prediction. A normalized anchor-cap protocol could preserve cap membership mathematically if complete-cost separability, a common cap construction and physical feasibility were established beforehand. It would still need clear symbolic units and could not claim measured/full-check milliseconds.

The current `reports/method.md`, `scripts/search_precision.py` and shared config explicitly require a validated complete host-to-decision cost table, numerical millisecond caps and actual resource/timing/bandwidth gates. Replacing that contract with symbolic caps is a methodological change, not completion of the missing measurements. This report neither changes that contract nor authorizes a comparison. Remaining blockers are full service/control/layout/interface accounting, the validity of map/profile-common offsets and precision-dependent memory treatment, complete new-profile runtime evidence, physical eligibility and a prospectively frozen cap rule. No final-test data is needed to resolve this algebra.

Evidence checked: actual 24 matrix shapes; cached16, cached32 and static-bank64 nominal and source-derived complete freezes; `hardware/linear_engine_cached.hpp` and separate `hardware/linear_engine_cached_static_bank.hpp`; revised `reports/goal4_fixed_fp32_plan.md`; current Goal5 cap and elite-selection contract. Exact rational arithmetic reproduced the coefficient, both uniform totals and all four group increments. Only this Markdown report was written for the audit.


## Conditional precision-only improvement envelope

Adding the completed common FP32 service aggregate to static-bank64 gives223,634,403 cycles at all-W4 and237,790,179 cycles at all-W8. Their difference is14,155,776 cycles, or70.778880ms at200MHz. Thus even the full W8-to-W4 change reduces this known serialized service subtotal by only5.953053% relative to W8. An intermediate map cannot exceed that precision-only reduction in the frozen bounded-memory model. This is an algebraic limit within the current design, not a measured FPGA speedup or a universal hardware bound.

If the remaining costs were a nonnegative map-independent offset, the percentage reduction in complete latency would be smaller. Physical DDR costs may themselves depend on packed weight bytes; no such cancellation is claimed before the memory model is specified. Changing the physical engine, layout or execution schedule is a separate mechanism with its own costs and common-baseline requirements. A lower bit count alone therefore cannot justify a large end-to-end speedup claim.

This constraint helps interpret the future B-to-C comparison: precision allocation still needs actual protection scores, but its possible benefit is bounded by the part of execution that changes. No method or cap is selected from this calculation, and no additional candidate was evaluated.
