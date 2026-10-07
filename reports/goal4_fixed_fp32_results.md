# Fixed FP32 service: synthesis, implementation and primary RTL accounting

The ninth actual synthesis and C-simulation passed for **panel_cache_batched_attention_v1** on `xcve2802-vsvh1760-2MP-e-S`. Vitis HLS2025.2 estimates **4.499ns**, **73BRAM18K,95DSP,24,808FF,29,830LUT,0URAM**. All named data loops report II1, including the previously serialized LayerNorm and softmax passes. The 5ns objective minus0.5ns uncertainty leaves only0.001ns HLS margin; this is not routed timing. Top-level runtime latency and interval remain undefined.

Evidence is `results/goal4/fixed_fp32/{run.json,vendor.log,gate_fixed_fp32_top_csynth.xml,schedule_summary.json}`. The driver snapshots the header/TB before compiling and records the revision. The offline analyzer `scripts/analyze_fixed_fp32.py` requires that revision, reads only modules listed in that successful run's log and excludes30 stale subreports. It ran once successfully after the ninth build, without fitting coefficients or invoking a vendor tool. Current source is `hardware/fixed_fp32_service.hpp`; its complete planned schedule is `reports/goal4_fixed_fp32_plan.md`.

## Actual representative out-of-context implementation

The completed implementation report `results/goal4/fixed_fp32_implementation/reports/verilog/export_impl.xml` gives target5.000ns, achieved/routed4.932ns, WNS+0.068ns, TNS0 and TIMING_MET=TRUE. Its post-synthesis estimate was5.555ns; place/route improved that result. The representative routed service therefore meets this5ns target. The HLS4.499ns estimate and its4.5ns effective scheduling budget are distinct from this routed report.

Actual implementation area is76BRAM18K-equivalent,91DSP,22,246FF,25,625LUT and0URAM. The LUT total already includes1,703SRLs;23,922 is the remaining logic-LUT count, not another resource to add. Compare these routed/IP totals with HLS estimates as different stages. The implemented object is the exported service IP and its generated implementation context, not the integrated integer+FP32 detector, external memory, FPGA sequencer or board design.

`scripts/implement_fixed_fp32.py` now parses only the small TimingReport and AreaReport sections on successful completion; `--parse-report` refreshes existing metadata without a vendor command. The actual metrics and report path are recorded in its `run.json`. It does not load the huge RTL hierarchy or infer success from returncode alone. No implementation was rerun for this update.

## Actual small RTL suite

`results/goal4/fixed_fp32_rtl/2026-09-26T19_39_34_372157_00_00/run.json` selects11 passed cases/22 checked calls covering all9 modes. Both repetitions passed the unchanged local references/tolerances, including DOT with output tail, nonfused cancellation, signed A8 ties/clipping, nonbinary quantization,2x1024 rescale channels, embedding order, residuals, LayerNorm, masked softmax, erf-GELU and tanh. Earlier launch/harness failures remain saved; the summary points to the passing attempts.

| Small case | Observed cycles, repetition0 /1 | Maximum absolute error |
|---|---:|---:|
| DOT3x64,N9 with bias |2397 /2397|0|
| Nonfused cancellation |79 /79|0|
| A8 ties/clips |231 /231|0|
| A8 nonbinary scale |279 /279|0|
| Rescale2x1024 |6303 /6303|0|
| Embedding19 elements |208 /208|0|
| Residual19 elements |206 /206|0|
| LayerNorm3x256 |4898 /4898|3.17636e-6|
| Softmax2x256 |3208 /3208|7.62618e-9|
| GELU10 elements |252 /252|8.38897e-8|
| Tanh10 elements |177 /177|2.77502e-8|

These are start-to-done service cycles at5ns with six independent byte-array memory ports, one outstanding burst per port, registered response, no additional DDR delay and no cross-port contention. They include this simulated interface's waits but exclude control-register setup and gaps between calls. Small cases are not extrapolated to complete detector latency. The separate15-shape primary characterization is now complete, as detailed below. Its checked weighted full-call summary uses the same synthesis revision, clock, memory model and411-call multiplicities.


## Completed primary RTL accounting

All15 declared L256 invocation shapes passed their unchanged formula references/tolerances twice:30 checked calls. The resulting411-call weighted sum is **94,864,971cycles**, or **474.324855ms at the simulated5ns clock**, identically for both repetitions. This is a weighted sum of representative full-call observations, not a measured end-to-end411-call execution or board latency. Test operands have the actual invocation shapes but are signed formula-check fixtures, not checkpoint activation traces.

Evidence is `results/goal4/fixed_fp32_rtl/2026-09-26T19_39_34_372157_00_00/primary_l256/{run.json,primary_costs.json}` and the referenced per-case records. The previous11 small cases were reused. Ten pending primary cases ran through two disjoint workers using private copies of the same compiled snapshot, ROM files and XSIM runtime directories; the verified serial parent alone was SIGSTOPped while its current simulation finished. Both workers exited0, the parent was SIGCONTed and it reused those ten successful records. `build/fixed_fp32_rtl/2026-09-26T19_39_34_372157_00_00/isolated_primary/dispatch.json` records that completed recovery. No synthesis, elaboration, arithmetic or memory-model change was made for parallel execution.

| Primary service shape | Calls/window | Observed cycles/call, both repetitions | Weighted cycles/window |
|---|---:|---:|---:|
|QK128x64,N64|128|321,941|41,208,448|
|AV128x256,N64|32|754,325|24,138,400|
|LayerNorm128x256|18|208,648|3,755,664|
|Attention softmax128x256|32|204,934|6,557,888|
|Quantize128x256|40|98,484|3,939,360|
|Quantize32x1024|32|98,484|3,151,488|
|Rescale128x256|40|98,463|3,938,520|
|Rescale32x1024|32|98,463|3,150,816|
|Embedding128x256|2|98,455|196,910|
|Residual128x256|16|98,453|1,575,248|
|GELU32x1024|32|98,584|3,154,688|
|Pooler1x256,N64|4|23,954|95,816|
|Classifier1x256,N2|1|1,182|1,182|
|Tanh1x256|1|460|460|
|Final softmax1x2|1|83|83|
|Total|411|—|94,864,971|

QK and AV contribute65,346,848cycles (68.8841% of this service sum). Attention softmax adds6,557,888cycles and all LayerNorm calls3,755,664cycles. Actual complete-call accounting is substantially larger than the earlier49,742,799 identified HLS components; that earlier quantity was explicitly partial and is retained below without being relabeled a latency prediction.

The cycle boundary is accepted DUT start to done. The sum includes the declared simulated interfaces' waits, but excludes configuration-register writes, sequencer dispatch/inter-call gaps, layout/gather/transpose/mask work, integer projections, physical DDR/NoC effects, host transport and document aggregation. These remain separate unresolved terms;411 calls require an FPGA-side sequencer, and no zero-cost host-per-call assumption is made.

### Observed six-port AXI traffic

Weighting the actual six-port counters by the same411-call schedule gives **105,393,192 transfer/strobe bytes**, matching the source-level service byte count, but **112,471,080bytes of full32-bit-beat occupancy**. Both repetitions give identical traffic. The extra7,077,888 occupancy bytes arise from2,359,296 one-byte A8 writes on four-byte buses; they are inactive lane capacity, not additional useful payload.

| AXI port | Read transfer bytes | Written strobe bytes | Full-beat occupancy bytes |
|---|---:|---:|---:|
|x|30,939,144|0|30,939,144|
|y|22,808,584|0|22,808,584|
|z|12,059,656|0|12,059,656|
|accumulators|9,437,184|0|9,437,184|
|output|0|27,789,328|27,789,328|
|codes|0|2,359,296|9,437,184|
|Total|75,244,568|30,148,624|112,471,080|

There are18,811,142 accepted read beats in15,798,662 bursts and9,306,628 accepted write beats in9,306,388 bursts. Read bytes use observed transfer size/address; written bytes count asserted WSTRB lanes. These are simulated AXI transactions including repeated accesses, not physical DDR traffic or measured sustained bandwidth. The additional18,092,032 analytical layout bytes below are outside these service ports and retain unknown timing; do not add this traffic as an extra latency term on top of already charged nominal AXI waits without an explicit replacement/adjustment model.

`scripts/run_fixed_fp32_rtl.py --primary --analyze-only` reproduced the weighted cycle/traffic file offline, without a vendor process. `scripts/analyze_fixed_fp32.py` admits the completed same-revision records and embeds these observations in `schedule_summary.json`; full checking latency and feasibility remain null.

## What changed and what passed

One N<=64,K<=512 DOT panel is cached before processing the M input rows. Four cyclic BRAM banks feed the unchanged four output lanes, each retaining separate FP32 multiply/add and ascending k order. Unused tail lanes receive zero without reading unwritten cache rows. LN load/normalize and softmax scale/mask/max now defer invalid-data rejection until the end of their passes; safe substitutes are used only on failing paths. Vector modes retain the earlier equivalent treatment.

All previous C checks passed with unchanged references/tolerances. The sole new test is M2,N64,K512, covering the final cache address, every output and second-row reuse. DOT now checks158 values with zero formula error; separate multiply/add cancellation remains exactly0. A8 signed ties/clipping, int32 conversion, rescale order including2x1024channel parameters, embeddings/residuals and invalid/overflow rejection pass.

| Formula check | Values | Maximum absolute / relative error |
|---|---:|---|
| LayerNorm |768|3.17636e-6 /3.46768e-6|
| Stable softmax |531|1.89347e-8 /1.93752e-6|
| erf GELU |10|8.38897e-8 /1.36238e-6|
| Tanh |10|2.77502e-8 /3.78803e-8|

These are local C-vector checks, not full-detector agreement, RTL timing or protection evidence. The separate small RTL evidence above is now complete. Floating reduction/library differences from PyTorch still require later end-to-end validation.

## Actual scheduling and historical comparisons

Earlier source/report snapshots remain at `fixed_fp32_timing_corrected` (fifth build), `fixed_fp32_scheduled` (sixth) and `fixed_fp32_vector_scheduled` (eighth), under `results/goal4/`. Their historical schedules remain distinct. The original5.653ns build is outside this scheduling comparison.

| Stage | Fifth II | Sixth II | Eighth II | Ninth II / XML depth |
|---|---:|---:|---:|---:|
| DOT row load |12|1|1|1 /2|
| DOT weights |46 per four strided reads|1 per weight row|1 per weight row|1 /14, flattened full panel|
| DOT four-output reduction |1|1|1|1 /4|
| Quantize |82 mixed-mode|26|1|1 /40|
| Rescale/bias |82 mixed-mode|30|1|1 /29|
| Embedding add |82 mixed-mode|15|1|1 /26|
| Residual add |82 mixed-mode|14|1|1 /25|
| erf GELU |82 mixed-mode|82|1|1 /92|
| Tanh |82 mixed-mode|48|1|1 /38|
| LN load/sum |12|12|12|1 /14|
| LN centered variance |1|1|1|1 /5|
| LN normalize/affine/store |40|40|40|1 /29|
| Softmax scale/mask/max |38|38|38|1 /27|
| Softmax exp/sum |1|1|1|1 /9|
| Softmax normalize/store |1|1|1|1 /26|

A panel load occurs once per call; historical weight loads occurred once per input row. Comparing those IIs alone is not a speedup calculation. The current loop `dot_panel_outputs_dot_load_panel` is the compiler's flattening of the output-row and reduction-width source loops, with N*K iterations. RESCALE remains flattened across row/channel dimensions.

Compared with the eighth build, the current design adds60BRAM18K,934FF and534LUT, with DSP unchanged95 and period unchanged4.499ns. This matches the larger logical panel but does not prove integrated device fit. The fifth/sixth period was4.426ns. Count the entire shared service once; nonlinear-library internals and interface logic are included, not just four DOT lanes.

Compiler warnings include conservative loop-bound updates to1024/1023, module renaming, unused/dangling duplicate AXI signals and synchronous active-low reset. API control still restricts DOT/SOFTMAX width<=512 and LN width256. No ignored allocation pragma appears in this run. Synthesis warnings and a C pass do not replace actual RTL checks.

## Report-supported partial expressions

For these submodules, both reported trip-count/function-latency endpoints support the listed affine expression. K remains restricted by the actual API, even when the report gives a broader bound.

| Submodule | Actual report-derived cycles |
|---|---|
| DOT row load |K+2|
| Four-output DOT reduction |K+4|
| LN load/sum |K+14|
| LN centered variance |K+5|
| LN normalize/affine |K+29|
| Softmax scale/mask/max |K+27|
| Softmax exp/sum |K+9|

The panel-load, softmax-store and six elementwise loops have no sufficient complete range to identify a full affine expression. Their partial term is the observed II times n-1; pipeline depth is retained separately. Undefined overhead is not filled with zero or inferred from one minimum endpoint.

Consequently, one current DOT call contributes the identifiable/launch-span components:

~~~text
(N*K - 1) + M * [(K+2) + ceil(N/4)*(K+4)].
~~~

Panel fill/drain/control, row/block transitions, output stores, optional bias, request setup/stalls and dispatch are unresolved. There is no zero-tail fill in the new cache. A per-row softmax component is `(K+27)+(K+9)+(K-1)`; its store fill/drain and parent control remain unresolved.

LayerNorm's partial per-row component is `(256+14)+(256+5)+(256+29)+2*13+1+8 =851` cycles, including the explicit mean/variance divisions, epsilon add and square root. Actual wrapper latencies remain add1,multiply1,divide13,sqrt8,exp4,erf51,tanh33. Square-root interval is8; others report1. The divider request is12, while the wrapper reports13.

These expressions are partial accounting terms, not measured call latency, complete predictions or validated bounds. No expected II is substituted where the actual report is missing.

## Batched attention, calls and traffic

For padded length L, choose M=min(L,floor(32768/L)), q=ceil(L/M),g=ceil(L/64). Per head and query slab: gatherQ[M,64]; call one QK DOT per key block; scatter its contiguous tile into score[M,L]; SOFTMAX all L keys per query; AV DOT with cached transposedV[64,L]; scatter context[M,64]. The mask slab[M,L] is materialized once per window and reused across heads/layers. All call spans fit32,768 elements.

This is a concrete analytical sequencer/layout schedule. Those components are not yet implemented. They require explicit positive costs, and host dispatch per call is not free. Full length512 attention remains intact. Width1024 rescale/GELU/quantization uses<=32rows per call; width256 streaming uses<=128rows. Examples are serialized.

| L | FP32 calls/window | Partial report components | Attention useful bytes including layout |
|---|---:|---:|---:|
|128|177|14,681,888cycles|13.625MiB|
|256|411|49,742,799cycles|45.25MiB|
|512|1,711|185,449,005cycles|178.25MiB|

At L256, QK/AV still each perform67,108,864 multiply/add pairs. Calls become128QK,32AV and32attention-softmax, plus219 other service calls. The old24,795-call total and253,337,989 partial components remain in the eighth-build archived JSON. Their comparison is structural/report-based, not a measured acceleration ratio.

Current L256 partial components are:

| Stage | Partial cycles |
|---|---:|
| QK DOT |19,431,296|
| AV DOT |18,620,384|
| Attention softmax |3,289,088|
| All LayerNorm |1,960,704|
| Quantization |2,359,224|
| Rescale/bias |2,359,224|
| Embedding + residual + GELU |1,638,350|
| Pooler/classifier + final nonlinear |84,529|
| Total |49,742,799|

Useful direct-service traffic is105,393,192bytes at L256, plus18,092,032bytes of analytical layout. The latter contains4MiB K/V staging,4MiB Q/context staging,8MiB QK tile assembly,0.25MiB mask-slab materialization,1MiB word/type staging and4KiB ID reads. Attention including only its own layouts is47,448,064bytes=45.25MiB, compared with541MiB for the old row-reload schedule. These are source bytes, not measured AXI transfers or sustained DDR bandwidth. The mask materialization and repeated softmax mask reads are both charged.

The JSON includes L128/256/512 and serialized batches1/8/32. Only L256,batch1 now has observed service-call and AXI totals; the other shapes retain analytical components. Integer cycles, layout cycles, dispatch, physical memory adjustment, host/transport, integrated FP32/checking latency and feasibility remain null. The current integer-engine revision must provide its own traffic/cycle evidence; old uncached integer counts cannot silently describe a cached design.

### Embedding staging bookkeeping correction

EMBED_ADD requires three contiguous L*256 FP32 inputs. Word lookup and token-type expansion therefore need two explicit staging slabs, totaling2*L*256*4 workspace bytes. With no implemented type-vector cache, gathering their vectors reads and writes16*L*256 useful bytes. The current software IDs are64-bit; reading two L-element ID arrays adds16*L bytes. Any later32-bit packing must declare and charge the host conversion/transport change. Absolute positions0..L-1 already form contiguous table rows and can be addressed directly, so no additional position slab is credited or charged.

At L256 this adds1MiB vector traffic plus4KiB ID reads and0.5MiB workspace, beyond the service's own embedding input reads. These are previously omitted analytical bookkeeping terms, not a new HLS revision, implemented gather kernel or timing measurement. All corresponding layout/controller costs remain unknown.


## Remaining work and interpretation

The new cache and validity scheduling are common baseline corrections. They make the shared schedule more practical but do not define C or establish H2/H3. Complete checking cost still needs the chosen integer engine, FPGA sequencer/layout/controller, byte-to-memory service beyond the declared simulated service, host transport/synchronization and strict document-max/threshold control. Unknown terms are not a passed feasibility gate, and nominal simulated interface service must not be double-counted when later DDR terms are added.

Integrated-system routed timing, bandwidth/resources and physical board latency/energy remain unexecuted; detector numerical agreement and protection remain unestablished; the standalone IP route above is a separate completed result. CPU/GPU inference or CUDA transfers cannot substitute for FPGA arithmetic or transport. The board is not required for these completed offline C/HLS steps, but it is required for later board measurements.

