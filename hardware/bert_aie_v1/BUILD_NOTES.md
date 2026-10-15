# Executed evidence and remaining integration

This file records supporting engineering evidence. It does not authorize new
runs or declare the offline study complete. All paths are canonical-checkout
relative. No physical device was opened or programmed.

## Binding and implemented computation

Matched seed42 checkpoint SHA256:
`95025bfc0e29163efea613d556d7da0b516ec1bb6c6fac9d9c4c9c6f1f373525`.
Packed numerical manifest SHA256:
`c4fe164a48051ca3834e112f5efd6564c559d8d62604dfe716cb2c1c097d92a3`.
The four payloads total35,284,072 bytes, including32,101,384 FP32 parameter bytes,
3,145,728 INT8 encoder weights,36,864 FP32 scale bytes and96 input-scale bytes.
`model.hpp` implements all four layers and embeddings/pooler/classifier.
No floating head or embedding parameters were omitted from hardware storage.

Both mapping controls use the same two matrix engines and surrounding operators;
they differ in external versus local intermediate tensor storage. The final
tile choices and source-derived traffic are in README.md. The local PL array
requests URAM, and contiguous weight/K source loads transpose into local tiles.
These are implementation choices awaiting synthesis, not measured resource use.

## Completed focused checks

- Portable whole-model C++ execution on FIT0 (130 valid tokens) and FIT6 (256)
  gave bit-identical resident/external outputs. FIT0 logits were
  `[4.807548523,-4.444448471]`, FIT6 `[4.57810545,-4.23065424]`.
  Maximum difference from the vectorized Python candidate over these outputs
  was approximately0.003582. These are numerical observations, not a declared
  universal tolerance or proof of matching threshold-tail decisions.
- Increasing W8 and BF16 independent output-tile dimensions preserved the
  checked FIT0 outputs exactly. The final portable mapping reports3,072 integer
  and2,048 BF16 tile calls per full window. Its roughly0.215-second software
  runtime is only diagnostic; it is not the qualified optimized software timing.
- A complete FIT0 inference using the installed HLS bit-accurate math C library
  completed in138.8068818 seconds and gave exactly the same logits as the portable
  scalar path. Evidence: `build/bert_aie_v1/common/cpu-hls_attempt0.json` and
  `hls_math_fit0_resident.json`. Matrix services in that run were portable CPU
  implementations; this was not an AIE/RTL/physical whole-model run.
- Actual PL AXIS adapter packing and result reordering passed integer boundary
  and BF16 checks with vendor `ap_axiu`/`hls::stream` types. Final evidence:
  `build/bert_aie_v1/common/wire_attempt2_run.log`. Returned arithmetic values
  were injected from the CPU matrix oracle, so this validates the interface
  layout rather than claiming actual AIE execution.
- The AIE simulator fixture generator uses independently exported real FIT QK,
  probability/V and W8 tiles; BF16 comparison uses prospectively specified
  `gamma_64 * sum(abs(products))` bounds against FP64. A separate INT8 signed
  boundary tile is exact. Fixture sources are in
  `results/bert_aie_v1/numerics/real_tiles/`.

## Build attempts and bounded corrections

1. Initial portable compile failed because GCC does not expose all C `*f`
   functions in namespace`std`. Selecting their global C names fixed the
   focused compile. Original logs remain `cpu_attempt0.log` and`cpu_attempt1.log`.
2. AIE diagnostic attempt0 ran154.85 seconds. It found the `AIEMLbuild` licence,
   compiled both kernels, mapped and routed the graph, then failed CDO host
   generation. Root cause: a local executable named`functional` shadowed the
   C++ standard header`<functional>` through the compiler's include path.
   Renamed it to`bert_functional`; the old binary, source/data and Work directory
   are preserved, not rewritten as a successful run.
3. AIE attempt1 (larger W8 tile) was stopped after130.73 seconds by the required
   runtime memory guard. Lowest sampled available memory was1,170,616,320 bytes
   on Windows and10,676,981,760 bytes in WSL. This is a host-resource abort, not
   a failed numerical test or a demonstrated device-capacity limit. Its output
   shows both kernel compilation and graph mapping/routing progress, but no
   complete archive/simulator/link success is claimed. Logs and resource record:
   `build/bert_aie_v1/common/aie_attempt1.{log,json}`.
4. The focused resource correction uses documented`aiecompiler --Xelfgen=-j1`
   instead of the generated default parallel ELF compilation. The installed
   2025.2 `--help` explicitly describes`--Xelfgen` as a Makefile option, with
   `-j2` as its example. The same finite timeout and memory floors remain.

The HLS configuration places the device part in the general section and explicitly sets `package.output.syn=true` with `package.output.format=xo`, following the installed Vitis help and [2025.2 package options](https://docs.amd.com/r/2025.2-English/ug1702-vitis-accelerated-reference/Package-Options). Link implementation is restricted to one synthesis and implementation job using the documented `vivado.synth.jobs` and `vivado.impl.jobs` options.

The shared build driver acquires the vendor lock per vendor stage and also the
GPU lock during AIE compilation. It never steals a sibling lock. Memory checks
require4GiB Windows/8GiB WSL available before launch and abort at2GiB/4GiB runtime
floors. Compilation, simulation, synthesis, routed implementation and board
measurements must remain separately labelled in the final report.

The coordinator subsequently authorized `--aie-vendor-only` for the single-ELF
retry to avoid starvation by an unrelated GPU fit queue. This explicit option
permits CPU/AIE compilation alongside GPU training only while the same memory
floors hold. The driver records the compatibility assumption, its own SHA256,
minimum available Windows/WSL memory and sampled maximum owned process-group
RSS. Summed RSS can double-count shared pages and is not a physical/PSS peak.
Attempt2's process-group RSS excludes vendor children that create separate
process groups and must not be treated as total compiler memory. Subsequent
stages follow the owned descendant tree across process groups instead, still
subject to sampling, shared-page and detached-child limitations.
Default AIE behavior still excludes GPU work; no frozen numerical source changes
are involved in this scheduling option.

AIE attempt2 completed successfully in213.2686seconds with `libadf.a`6,418,148B.
It used the explicit vendor-only overlap option. Minimum sampled available
memory was4,561,444,864B Windows and17,834,987,520B WSL, above runtime floors.
The archived failed attempts remain intact. Native tile simulation and complete
system HLS/link are separate subsequent qualification stages.

Native simulation attempt0 rejected an invalid input-directory option despite
returning zero; it produced no output and is not a passed simulation. Attempt1
used documented `--input-dir=.` and runtime memory checks. All2048 INT8 outputs
were exact; all2048 BF16 outputs passed the prospective bounds, with maximum
absolute error3.63563919e-6 and maximum error/bound ratio0.0670609. Cores reported
completion, but teardown stalled after repeated SIGINT messages. Only its known
simulator/debugger processes were terminated after over two minutes without log
progress. The wrapper again returned zero; explicit `completion_adjudication`
in its JSON records **unclean simulator termination**, independently of the
arithmetic pass. No clean tool exit or full-model hardware run is inferred.

`native_costs.json`, generated by `build_hardware_costs.py`, preserves allocation,
program/stack use and two-frame timestamp facts with source hashes. The graph
uses two compute cores and five AIE tiles including supporting tile-memory
groups,65536B graph buffers and2844B system allocation. Configured AIE clock is
1250MHz; requested PLIO clock250MHz. Output-frame first-beat intervals are2048ns
INT8 and2480ns BF16 over only two independent simulator-fed tiles. These are not
whole-model latency or steady-state connected service measurements. Routed
system clock and stall counters are unavailable at this stage.

Resident full-model HLS attempt0 completed in269.5103seconds and exported its
own`bert_pl.xo`. Estimates are124BRAM18,129DSP,81199FF,320294LUT and113URAM;
target availability is1200BRAM18,1312DSP,1041408FF,520704LUT and264URAM. The
3.505ns estimated clock exceeds the effective2.920ns budget (4ns target minus
1.08ns uncertainty) by0.585ns. This is not proof of meeting250MHz.

The generic top's284672201-cycle maximum combines independent K/O ranges.
`build_hardware_hls_costs.py` reconstructs actual call shapes from loop reports:
linear cycles=`2+(256*K+28)+8*(2+(O/32)*(3157+9825*K/256))`. Calls therefore take
896430,3388974 and2979438 estimated cycles for256x256,256x1024 and1024x256,
respectively. Four actual layers plus conservative other overhead yield57340361
cycles, or229.361444ms at requested4ns. This assumes ready interfaces and omits
actual external-memory/AIE stalls; it is neither measured runtime nor a routed
latency guarantee. No GPU-versus-FPGA measurement claim follows.

Long quantization, GELU and copy loops already have automatic pipelining, mostly
II1; scalar tile gathering still takes8201/8271 cycles per8192-byte operand.
The inferred weight AXI port is32bits despite a maximum-width request of512;
all six AIE streams are128bits. The generated multiply/add IP explicitly uses
`Unfused_Multiply_Add`, so its name alone does not indicate FMA contraction.
The exact reports, loop IIs, source hashes and warnings are in the resident
`hls_costs.json`. No frozen numerical source was edited to tune these findings.

## Unresolved items

Both HLS mappings, linked system and routed implementation require actual
successful reports before being marked complete. Native AIE arithmetic is
validated as described above, with an unclean simulator teardown qualification.
The two-engine design's configured W8 A-port alone has an ideal6.291456ms
transfer lower bound; that is evidence about this mapping, not all BERT designs.
Whether additional engines/residency help the complete path remains to be
measured against dependent PL operators, external memory and resources.

The board host source provides batch and persistent stdin protocols, complete
window aggregation using raw logit margins, strict frozen-threshold decisions,
finite-result checks, cold setup timing and per-document timing excluding
tokenization. The raw-document wrapper must time preparation through final
return contiguously. No installed XRT host headers or `libxrt_coreutil.so` were
found in Vitis, `/opt/xilinx/xrt` or `/usr/include/xrt`; a verified board Linux/XRT
sysroot is therefore required for host compilation/package continuation.
Source availability does not establish a compiled board application.

Focused connected-emulation inspection found `hw_emu.xsa`, `.isd`, presynthesis
PDI and QEMU templates in the selected platform; its SPFM declares a Cortex-A72
Linux/XRT domain. Installed Vivado provides `xsim`. The platform's XRT image
directory contains only `boot.scr`, with no Linux `Image`, root filesystem or
matching XRT host sysroot located in the inspected installation. This blocks the
normal platform Linux/XRT connected emulation continuation independently of
achievable hardware HLS/link. No emulation framework, runtime installation or
physical-device access was attempted.

Paired HLS correction prepared after original synthesis (2026-10-15 UTC):
the original external estimate was 839,976 LUT, exceeding 520,704 available;
this is an HLS estimate, not a proven physical implementation failure. Wide
512-bit interface muxes dominate several modules. Both mappings now cap the
parameters and scratch AXI interfaces at128 bits. Original `pl.cpp` is preserved
as `versions/pl_original_axi.cpp`; original XOs and synthesis directories are
archived automatically before retry. This changes interface scheduling, not
the frozen numerical source or format.

The sole dependence directive is
`syn.directive.dependence=variable=x type=inter dependent=false bert_aie::norm/VITIS_LOOP_49_4`.
The pointwise output iteration c reads/writes only x[r*256+c], disjoint across c;
gamma and beta reside in the separate immutable parameters BO. Mean and
variance reductions finish before this output loop, and their dependence/order
is unchanged. Original external output-loop II154 was attributed by HLS to a
scratch AXI memory dependence. Directive acceptance and resulting II must be
verified from retry reports; original slow external estimates are excluded from
a competence-based speedup claim. AMD UG1399 2025.2 documents this config syntax
and warns that false dependencies must be proved; the proof above is scoped to
this output loop only.

Original resident connected link attempt0 completed block-design creation,
update and target generation, then was aborted during Vivado synthesis after
241.568s by the runtime Windows memory floor:1,655,484,416 bytes available;
WSL20,769,845,248 bytes; sampled owned-descendant RSS5,210,300,416 bytes (RSS
can double-count shared pages). Return-15 and explicit aborted status are
preserved, along with terminated owned PIDs. No routed XSA was produced. The
next link uses both vendor/GPU leases, an8GiB Windows launch floor based on
this observed footprint plus2GiB headroom, and general.maxThreads=1 in addition
to synth.jobs=impl.jobs=1. Runtime floors remain Windows2GiB/WSL4GiB. No global
WSL/cache settings or unrelated processes were changed. A waiting lease or
failed preflight is not vendor execution. Numerical sources remain frozen.
