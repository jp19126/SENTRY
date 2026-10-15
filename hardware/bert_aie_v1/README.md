# Whole-model hardware route

This is a supporting engineering implementation of the matched four-layer
BERT checkpoint. No scientific novelty or achieved hardware speedup is implied.
The numerical/export manifest is
`results/bert_aie_v1/export/w8a8_bf16_attention/manifest.json`.

`model.hpp` connects embeddings (word, position, token type), embedding LayerNorm,
four complete attention/residual/LayerNorm/feed-forward/GELU blocks, FP32 pooler
and classifier. Sequence length is256, hidden256, four64-wide heads, FFN1024.
Inputs are three contiguous int32[256] arrays: IDs, binary key mask, type IDs.
Outputs are two logits and a status float. Padding positions remain present;
masking excludes padded keys and an empty mask is rejected. Host document score
is the maximum window raw logit margin (injection minus benign) and the final
decision is strictly margin>frozen_margin_threshold.

The24 encoder linears use frozen signed W8/A8, symmetric[-127,127], ties-to-even,
exact INT32 dot products, then `(acc*input_scale)*weight_scale+bias` in FP32.
Attention rounds Q/K and probability/V to BF16, with FP32 products/reductions.
QK has K64; AV has four K64 reductions followed by FP32 chunk addition.
Softmax, normalization (epsilon1e-12), exact-erf GELU, residuals, embeddings and
head remain FP32. Floating reduction order differs between the portable scalar,
Torch and AIE-native paths; the actual differences must be measured. Integer
equality does not imply whole-model bit identity.

## Graph cut and control

`graph.cpp` contains one reusable INT8 matrix engine and one BF16 matrix engine,
with six actual128-bit PLIO ports. `pl.cpp` implements corresponding AXI4-Stream
interfaces and the entire surrounding model as one HLS kernel. `system.cfg`
connects these ports in Vitis. Model parameters reside in DDR and the wrapper
uses AXI master interfaces for all host-visible buffers. No model operation is
silently replaced by a completion stub or a host inference call.

The primary residency comparison compiles the same source twice:

- `BERT_RESIDENT=0`: full intermediate tensors use the external scratch allocation;
  the same per-operator operand tiles and full quantized activation reuse are local.
- `BERT_RESIDENT=1`: those same intermediate tensors use static local PL storage.

The local full-tensor capacity is720,896 FP32 values (2,883,584 bytes), plus
262,144 INT8 quantized activations and temporary tile/row storage. Resident
mapping feasibility and actual BRAM/URAM inference require the synthesis report.
The control preserves the same computation, matrix-engine resources and numerical
order; DDR traffic is the isolated placement difference. Neither mapping claims
two AIE tiles are optimal or makes a comparison against an optimized PL-only
implementation. Its usefulness is falsified if resource/timing infeasibility or
other bottlenecks eliminate the complete latency improvement.

## Analytically pruned service shape

The first diagnostic graph used4x64x8 INT8 tiles and exposed393,216 calls per
window, excessive interface overhead. Before final implementation, compare:

| W8 tile MxKxN | Double-buffer input/output bytes | Calls/window | Payload/window A+B+C |
|---|---:|---:|---:|
|32x64x32|16,384|12,288|100,663,296|
|32x256x32 (selected)|40,960|3,072|62,914,560|

Both reuse weights across32 rows and activations across32 columns. K256 fits
all projection/FFN widths without padding and reduces accumulator crossings4x.
Each native microtile is4x16x8. The selected40KiB ping-pong data footprint is
below64KiB/tile before compiler bank placement, stack and neighbouring memory
allocation; the compiled map is authoritative. INT8 stream order is documented
directly in `pl.cpp` and `tiles.cpp`; output native tiles are reordered in PL.

BF16 uses32x64x32 tiles with native4x8x8 microtiles,2,048 calls/window with
16,777,216 input bytes and8,388,608 output bytes. Its24KiB double-buffer data
footprint preserves K64 numerical reductions while removing repeated transfers
between independent output tiles. This replaced the initial4x64x8 diagnostic
shape, which needed65,536 calls and109,051,904 total bytes per window.
The current combined matrix service payload is88,080,384 bytes/window.
At128-bit/250MHz, a single direction has a4GB/s
ideal line rate; aggregate bytes cannot be divided by summed port bandwidth to
claim end-to-end latency because transfers and operations depend on one another.
These are source-derived counts, not measured bandwidth or timings. Attention
service overhead and serial PL nonlinear work can dominate despite W8 reuse.
The W8 input-A port alone carries25,165,824 bytes/window, implying an ideal
6.291456ms lower bound at that specified interface rate. This is a narrow
limitation of this two-engine mapping, not a universal BERT or FPGA bound.

## Offline and physical boundaries

After sourcing `/home/jp19126/Xilinx/2025.2/Vitis/settings64.sh`, run
`python3 scripts/bert_aie_v1/build_hardware.py STAGE` with finite timeouts.
Stages: `cpu`, `cpu-hls`, `wire`, `aie`, `aiesim`, `check-sim`,
`hls --mapping resident|external`, `link`.
The coordinator schedules jobs; the build driver obtains the shared vendor
lock per stage and also GPU lock for AIE compilation to exclude memory contention.
It requires4GiB Windows/8GiB WSL available before vendor launch and stops at
2GiB/4GiB runtime floors. `--Xelfgen=-j1` is the compiler's documented
low-concurrency correction after an observed memory-floor abort.
Attempt logs preserve exact commands; compiled archives/XO/XSA are not assumed
to exist merely because sources or a command are supplied.

Portable full-model differential command:

```
build/bert_aie_v1/common/bert_functional results/bert_aie_v1/export/w8a8_bf16_attention results/bert_aie_v1/numerics/fixtures/fit_0.bin resident
```

`host.cpp` is an XRT board continuation harness and **programs a device if run**.
It is never invoked by offline scripts. After separately authorized deployment,
build/package the linked design with the platform's matching Linux/XRT sysroot,
compile this host for that target, and run `host XCLBIN EXPORT_DIR THRESHOLD
DOC_WINDOWS.bin...`. Every document file must contain all complete window
records from the frozen tokenizer/window manifest. Host output labels measured
time as excluding tokenization; complete raw-document timing must add measured
tokenization/packing and retain the document until all windows return.
Packaging/host compilation and physical measurements remain pending until
their dependencies and actual outcomes are established.

The installed toolchain lacks XRT host headers/libraries and a verified matching
board Linux/XRT sysroot. The host source is therefore a prepared continuation,
not a successfully compiled or executed board application. `--stdin` keeps the
model loaded, flushes its cold-ready record, and processes one complete-document
binary path per input line; the raw-document wrapper can time tokenization
through return contiguously, without summing unrelated component timings.
