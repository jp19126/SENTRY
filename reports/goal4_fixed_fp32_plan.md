# Common fixed-FP32 service and qualified whole-model accounting plan

Current source schedule: **panel_cache_batched_attention_v1**, 2026-09-26. This is a common baseline cleanup available equally to A/B/C and uniform anchors, not a proposed C. The previous eighth-build evidence is archived in `results/goal4/fixed_fp32_vector_scheduled/`. Its timing and resources do not describe this changed cache. Current build evidence belongs in `results/goal4/fixed_fp32/` and is admissible for this schedule only when `run.json.service_schedule_revision` matches the header macro. Synthesis, RTL observations, analytical counts and board results remain distinct.

## 1. Fixed model and scope

One common FP32 service complements one selected integer engine. All detector arithmetic remains on FPGA; host tokenization, transfers and synchronization remain explicit costs. The selected `checkpoints/floating/epoch_3` has four encoder layers, hidden width D=256, four heads, d=64, FFN width F=1024, vocabulary 30,522, 512 positions, two token types and two classes. Inference dropout is off, GELU uses erf and LayerNorm epsilon is 1e-12. Primary L=256,batch=1; L=128/256/512 and batches1/8/32 are the declared grid. Examples are serialized without cross-example cache credit.

The fixed W4/W8×A8 encoder linears retain their exact integer contract. Attention products, embeddings, bias/rescale, normalization, nonlinear functions, pooler and classifier use FP32. Sources are `configs/project.json`, checkpoint configuration, `gate_quant.py`, `gate_eval.py` and installed Transformers4.57.3 BERT. Different FP32 reductions or transcendental implementations are not automatically equivalent to the software backend.

The part is `xcve2802-vsvh1760-2MP-e-S`; objective200MHz with0.5ns uncertainty. Vitis HLS2025.2 estimates do not establish routed or board frequency.

## 2. Service API and unchanged arithmetic

`hardware/fixed_fp32_service.hpp` defines:

~~~cpp
enum FixedFp32Op {
    FP32_DOT=0, FP32_QUANTIZE=1, FP32_RESCALE_BIAS=2,
    FP32_EMBED_ADD=3, FP32_RESIDUAL_ADD=4, FP32_LAYER_NORM=5,
    FP32_SOFTMAX=6, FP32_GELU=7, FP32_TANH=8
};
bool fixed_fp32_service(int op, const float* x, const float* y, const float* z,
    const ap_int<32>* accumulators, float* output, ap_int<8>* codes,
    int rows, int width, int outputs, float scale, bool dot_bias);
~~~

The wrapper `gate_fixed_fp32_top` adds status. Every external pointer depth is32,768 elements of its type, not bytes or an implied on-chip buffer. Invalid control/extent is rejected before memory access. Invalid data returns failure; outputs already written by a failing call are unusable. Unused ports do not supply arithmetic inputs.

DOT uses X[M,K],Y[N,K],output[M,N],optional biasZ[N]; M<=512,K<=512,N<=64 and each accessed flat span<=32,768. Each of four output lanes starts at+0 and accumulates in ascending k:

~~~text
product = FP32(X[r,k]*Y[o,k])
sum[o] = FP32(sum[o]+product)
result = FP32(sum[o]+bias[o]) only when requested
~~~

No tree reassociation or interleaved reduction partial sums is introduced. Explicit wrapper multiply/add remains separate; compile with `-ffp-contract=off` and unsafe-math disabled. AMD nonlinear library internals can contain fused primitives, so this is not a claim that the entire math library is nonfused. Actual operator sharing, recurrence II and binding come from the full multimode synthesis.

For other modes rows*width<=32,768,rows<=512. Elementwise width<=1024 is required for FFN; SOFTMAX width<=512; LayerNorm width=256.

| Mode | Formula and ports |
|---|---|
| QUANTIZE | clip(nearest_even(FP32(X/scale)),-128,127), finite positive scale. Division, not reciprocal multiply. Signed infinities arising from finite division saturate; NaN never reaches floor/int conversion. |
| RESCALE_BIAS | factor=FP32(scale*Y[channel]); FP32(FP32(float(int32)*factor)+Z[channel]). Channel scales/bias cover the full width, including1024. |
| EMBED_ADD | FP32(FP32(word+token_type)+position), supplied in X,Y,Z. |
| RESIDUAL_ADD | FP32(X+Y). |
| LAYER_NORM | Ascending mean, ascending centered-square sum divided by256; add1e-12, sqrt, divide each centered element by that denominator, multiply gammaY then add betaZ. |
| SOFTMAX | score=FP32(FP32(X/scale)+Ymask), row maximum, exp(score-max), ascending exp sum, divide each exp by sum. Attention scale8; classifier scale1/zero mask. |
| GELU | argument=FP32(x/1.4142135623730951f); half=FP32(x*0.5f); FP32(half*FP32(1+erf(argument))). |
| TANH | FP32 tanh of pooler outputs. |

The finite additive key mask must match software. Masked keys and padded query rows remain computed; valid windows have unmasked CLS/SEP. Quant codes and bounded integer accumulation require exact agreement. Rescale order, LN sqrt/divide and library division/exp/erf/tanh/sqrt may differ from PyTorch, including denormal behavior. End-to-end agreement remains required.

Validity reductions now finish after each vector pass, LN load/normalize pass and softmax scale/mask/max pass. Invalid operands use safe substitutes only on failing paths. Valid-input operation grouping and reduction order are unchanged. Checks between reduction passes reject overflowing sums/variance/denominators before dependent work.

## 3. Actual buffers and proposed workspace

The updated service contains row[512], weights[64][512] and four partial accumulators: **133,136 logical bytes**. The panel is cyclically partitioned into four output banks and bound to dual-port BRAM. It is loaded once per DOT call, reused across M rows and replaced for the next call. It is not simultaneously a K and V cache or a cache across calls. Actual ninth-build HLS uses73BRAM18K versus13 previously, with95DSP,24,808FF,29,830LUT and estimated4.499ns. The measured HLS increase is60BRAM18K; it does not establish integrated resource fit or routed timing.

The row buffer is reused by inactive-relative-to-DOT LN/softmax modes. Four math lanes and existing sharing requests are unchanged. Whole-service synthesis includes nonlinear internals, FIFOs, AXI/control and bank padding; logical bytes alone are not a physical resource report.

External device-visible workspace is explicit:

| Tensor/staging | Maximum logical extent |
|---|---|
| Hidden ping-pong | 2*L*256 FP32 |
| Word/type embedding staging | 2*L*256 FP32, distinct from hidden workspace |
| Q,K,V,context | 4*L*256 FP32, separate without proven lifetime reuse |
| Staged K head / transposed V head | L*64 and64*L FP32 |
| Q query slab / context slab | M*64 each FP32 |
| QK output tile | M*64 FP32 |
| Score slab / probability slab | 2*M*L FP32 |
| Replicated additive mask slab | M*L FP32, once per window, reusable across heads/layers/query slabs |
| FFN intermediate | L*1024 FP32 |
| Integer output/code staging | up to256*1024 int32 and A8 per integer row chunk |
| Token/type IDs and base additive mask | explicit per-window arrays |

Here M=min(L,floor(32768/L)). All service spans fit their depth. Width256 streaming modes segment into<=128rows; width1024 into<=32rows. Integer L512 requires two<=256row calls. Full length512 attention is preserved. The external workspace and layout engines are not yet implemented or included in service BRAM. Full embedding/weight tables remain external unless an implemented cache says otherwise. Initialization is separate from warm-check traffic.

## 4. Batched sequential FPGA schedule

An FPGA-side sequencer must generate these calls, addresses and layout operations. There is no free host-per-call assumption and no claimed overlap between integer, FP32 or layout work.

1. Read two L-element64-bit word/type ID arrays and gather those vectors into two contiguous L*256 FP32 slabs; charge both vector reads and writes. Absolute position rows0..L-1 are contiguous and can be addressed directly. EMBED_ADD then embedding LayerNorm; no unimplemented token-type cache is assumed.
2. In each encoder, quantize/project/rescale Q,K,V separately. No QKV code reuse is credited without equal frozen scales and an implemented common reuse schedule.
3. Per head, gather K[L,64] and transpose V[64,L] in device memory.
4. For each query slab of M rows, gather Q[M,64]. For each of g=ceil(L/64) key blocks, call DOT(M,K64,N64); scatter the resulting [M,64] tile into score[M,L]. All declared lengths are multiples of64.
5. Call SOFTMAX(rowsM,widthL) using the replicated mask slab. Call AV DOT(M,K=L,N64) using transposed V and probability[M,L]. Scatter context[M,64] into the correct head columns of the global context tensor.
6. Quantize/project/rescale context; residual add; LayerNorm. Quantize/project/rescale FFN input; GELU; quantize/project/rescale FFN output; residual add; LayerNorm.
7. Pooler K256,N256 uses four N64 DOT calls with bias, then TANH. Classifier K256,N2 with bias then two-class SOFTMAX yields class1 risk.
8. FPGA control forms document maximum and applies strict risk>threshold; include control and result transfer.

Mask materialization reads the base length-L mask and writes each of M rows once per window. Conservative uncached useful traffic is8*M*L bytes; subsequent reads by every SOFTMAX are separately charged. A replicated slab can be reused because all heads/layers use the same key mask. A later different mask/cache layout needs changed accounting.

Let q=ceil(L/M). Across4layers×4heads, QK calls=16*q*g, AV calls=16*q and attention SOFTMAX calls=16*q. Attention rows/reductions and arithmetic counts are unchanged.

| L | M / q / g | QK / AV / attention-softmax calls | Total FP32 service calls |
|---|---|---|---:|
|128|128 /1 /2|32 /16 /16|177|
|256|128 /2 /4|128 /32 /32|411|
|512|64 /8 /8|1024 /128 /128|1711|

The former L256 schedule made16,384 QK calls because it issued one query row per call:16heads*256queries*4keyblocks. It also reloaded panel weights inside each row. Batching the API alone would not eliminate those reads; the new source explicitly hoists the full panel load outside the row loop. The 411-call schedule still requires positive dispatch/layout costs.

## 5. Structural arithmetic and byte ledger

These are calculated counts, not timings. At L256 across all four layers, QK and AV each perform67,108,864 scalar multiply/add pairs; attention has1,048,576 scores and4,096 softmax rows. There are2,304 LayerNorm rows/589,824 elements,524,288 residual adds,1,048,576 GELU elements,2,359,296 A8 input elements and the same rescale count. Embeddings use131,072 additions. Pooler has65,536 multiply/add pairs plus256 tanh outputs; classifier has512 pairs. Rescale recomputes the FP32 scale product per element.

For a cached DOT call:

~~~text
X reads = 4*M*K bytes
Y reads = 4*N*K bytes, once per call
output writes = 4*M*N bytes
optional bias reads = 4*M*N bytes
~~~

At each layer, with D256,h4,g=ceil(L/64),q=ceil(L/M):

~~~text
QK: Q reads4*D*L*g; K-panel reads4*D*L*q; score-tile writes4*h*L^2.
Softmax: score+mask reads and probability writes12*h*L^2.
AV: probability reads4*h*L^2; V-panel reads4*D*L*q; context writes4*D*L.
K gather + V transpose:16*L*D.
Q gather + context scatter:16*L*D.
QK tile-to-score-slab read + write:8*h*L^2.
Attention total per layer =
    8*D*L*q + 28*h*L^2 + 4*D*L*(g+1) + 32*L*D.
Once-per-window mask materialization adds8*M*L.
~~~

These explicit extra copies permit the unchanged contiguous DOT API. They cannot be credited as free stride reinterpretation. The primary attention useful traffic is **45.25MiB/window**, versus541MiB in the archived row-reload schedule. The structural totals are13.625/45.25/178.25MiB for L128/256/512. They are useful source bytes, not AXI beat counts or measured DDR traffic/time.

At L256, remaining direct-service useful traffic remains quantization11.25MiB, rescale36MiB, embedding-add plus embedding-LN2MiB, residual-add plus encoder-LN14MiB and GELU8MiB. Pooler/classifier and final nonlinear traffic are added explicitly by the parser. Layout costs include the mask slab, Q/context staging and QK assembly in addition to K/V transforms. Conversion/layout controllers, actual bursts/arbitration and external memory timing remain unresolved.

### Embedding staging bookkeeping correction

EMBED_ADD requires three contiguous L*256 FP32 inputs. Word lookup and token-type expansion therefore need two explicit staging slabs, totaling2*L*256*4 workspace bytes. With no implemented type-vector cache, gathering their vectors reads and writes16*L*256 useful bytes. The current software IDs are64-bit; reading two L-element ID arrays adds16*L bytes. Any later32-bit packing must declare and charge the host conversion/transport change. Absolute positions0..L-1 already form contiguous table rows and can be addressed directly, so no additional position slab is credited or charged.

At L256 this adds1MiB vector traffic plus4KiB ID reads and0.5MiB workspace, beyond the service's own embedding input reads. These are previously omitted analytical bookkeeping terms, not a new HLS revision, implemented gather kernel or timing measurement. All corresponding layout/controller costs remain unknown.


## 6. Estimator and evidence boundaries

The analyzer `scripts/analyze_fixed_fp32.py` extracts only modules named by the successful run's vendor log, avoiding stale reports. Current panel accounting requires matching schedule metadata and actual new reports. Archived row-reload reports retain their original role maps and schedule assumptions.

For actual per-stage report function C, the panel DOT component has structure:

~~~text
C_panel(N,K) + M * [C_row(K) + ceil(N/4)*C_reduce(K)].
~~~

If the compiler flattens the panel load, its actual loop trip is N*K; otherwise it is N calls of the K loop. Use the reported form. Identifiable affine constants require both report trip/function-latency endpoints; otherwise retain only the II-based launch-span component and unresolved fill/drain/control. No old cache coefficients, expected II1 or measured runtime is substituted.

Complete checking cost must add the selected integer-engine revision, layout/controller cycles, per-call dispatch, source-to-AXI/DDR service, host transport/synchronization and document reduction/threshold work. Do not double-count nominal memory service already present in a runtime observation. No bandwidth, overlap, complete latency or feasibility is invented.

Resource totals count one whole FP32 service, one selected integer engine and future integration/layout buffers/control. Preserve BRAM18K units; two18Kb capacities are one36Kb but banking can prevent perfect pairing. Project70% budgets are DSP918,LUT364492,FF728985,BRAM36K420,URAM184. The reserved share does not prove integration fits. The actual representative out-of-context service route now meets5ns at4.932ns withWNS+.068/TNS0; it does not establish integrated-system timing or board measurements.

## 7. Checks and bounded build handoff

Reuse all existing `hardware/fixed_fp32_service_tb.cpp` checks and fixed tolerances. The only additional vector is DOT M2,N64,K512, which exercises the last panel address, all64 outputs and second-row cache reuse. Existing cases cover K64/256/512, output tails, nonfused cancellation, exact signed ties/clipping, int32/rescale grouping including width1024, embeddings/residuals, LayerNorm, masked softmax, erf-GELU/tanh and invalid input/overflow rejection. Print `PASS fixed_fp32 service` only if all checks pass.

Criterion is abs_error<=atol+rtol*abs(reference): DOT(3e-5,3e-5), LN(2e-4,2e-4), softmax/GELU/tanh(3e-6,3e-5). Declared conversion/rescale/embedding/residual formulas and A8 codes use exact checks. These remain operator bring-up tolerances, not allowed detector-score error. End-to-end logits, risks, document maxima and strict decisions need the prescribed later agreement check; held-out calibration/test remains closed.

Root completed the ninth synthesis within the allowance; all existing C checks plus the one cache endpoint passed. Actual reports and extracted II1 data-loop schedules are documented in `reports/goal4_fixed_fp32_results.md`. No vendor tool was run by the offline analyzer. The header's revision must be saved in run metadata. A failed or stale run cannot support new cycle/area claims. Csim/HLS/RTL can advance Goal4 while the board is disconnected; they do not complete full detector validation or board deployment.

AMD2025.2 [operation binding](https://docs.amd.com/r/en-US/ug1399-vitis-hls/set_directive_bind_op?contentId=7v4LpZC4PLBM7PBulnDrAg) and the installed HLS headers define supported controls/math functions; only actual selected-part reports establish achieved binding and schedules.

