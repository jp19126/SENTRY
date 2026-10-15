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

The shared build driver acquires the vendor lock per vendor stage and also the
GPU lock during AIE compilation. It never steals a sibling lock. Memory checks
require4GiB Windows/8GiB WSL available before launch and abort at2GiB/4GiB runtime
floors. Compilation, simulation, synthesis, routed implementation and board
measurements must remain separately labelled in the final report.

## Unresolved items

Final AIE compilation/simulation, both HLS mappings, linked system and routed
implementation require actual successful reports before being marked complete.
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
