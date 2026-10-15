# Complete KD-CNN integer path

This is a supporting hardware/numerical implementation, not a new algorithm or
a hardware advantage claim. It realizes the fixed StudyCNN `dilated` graph:
L256, E64/C64, projection K1, six residual K3 blocks at dilation 1/2/4/8/16/32,
no normalization/gating, ReLU on each convolution branch, signed skip addition,
masked max + mean pooling, and a two-logit classifier. Local receptive field is
127 positions with centered lookahead 63; global pooling still covers all valid
positions. Inference requires no teacher, training projection, or transformer.

## Exact format

The binary is 2,036,944 bytes. Its eight little-endian uint32 header words are
magic `0x4b444332`, version 1, vocabulary 30522, channels 64, length 256,
depth 6, logit fractional bits 12, and total bytes. Fields follow the export
manifest order, with no struct padding: embedding W8, projection W8/bias64/
shift32, six convolution W8/bias64/shift32/skip-shift32 blocks, head W8/bias64/
shift32. Shapes/offsets/hashes are explicit in each export's manifest. Python
loads only hash-verified packed contents; the host must verify the same frozen
identity before handing bytes to hardware.

Embedding codes use one power-of-two scale. Conv/head weights use per-output
power-of-two scales. The embedding, projection and each residual output have
explicit activation exponents. Codes are signed symmetric [-127,127]. All
rounding is nearest, ties to even; division for masked mean uses that same rule.
Convolution products accumulate in INT32 (largest bound 192*127^2=3,096,768),
then add INT64 bias bounded by 2^55. Requantization uses INT64 with saturation.
Projection clips to127. A residual branch is shifted into the new output unit,
bounded to2^30, and rectified; the independently aligned skip is added before
final symmetric127 saturation. Max excludes masked sites; mean divides by the
valid count; all-masked windows have zero summary. Two Q12 logits are bounded
to Ã‚Â±(2^30-1), so their margin fits INT32 without overflow. Strict margin>
threshold is the decision; equality is allowed. Probability is an optional
host presentation of sigmoid(margin/4096), not a hidden FPGA operation.

Every intermediate masked position is zero. Arbitrary masks are supported;
masked garbage token IDs are never dereferenced, while live IDs>=30522 and
nonbinary masks are errors. Inputs include the established CLS/SEP tokens and
padded L256 tails. This is not the content-only span-guard interface.

## Memory and document interface

`kd_cnn_top` exposes model/input/output AXI memory masters and AXI-Lite control.
Model loading retains parameters in the PL model object. Two full-window
activation banks each hold16,384 code bytes; channel partitioning supplies
4 or16 MAC lanes. Embedding storage is explicitly bound to URAM and reshaped;
actual resource allocation, replication and schedule require synthesis reports.
`CNN_LANES=4` is the conventional schedule control for `CNN_LANES=16`; graph,
parameters, integer rules and decisions are identical. This implementation does
not claim streaming line-buffer storage savings.

A call handles1..16 full windows. `first=1` resets the document maximum and
requires `previous_windows=0`. Continuations supply the exact preceding maximum
and count. All windows/tails must be processed; any document length is supported
by additional chunks. `final=1` makes `decision_valid=1` only if every window in
the call succeeds. Intermediate `blocked=0` is not permission to release a
document. Reloading a model during continuation is rejected. Hosts must retain
the raw document until final completion, keep document/model associations, and
reject errors/timeouts. Tokenization, transfers, retention, cache maintenance,
submission, and response costs remain part of complete checking latency.

## Reproduction and current evidence

Use explicit canonical result/build paths, even when source lives in a Windows
worktree. `scripts/kd_cnn_v2/numerical.py --checkpoint CHECKPOINT --calibration
FIT --out EXPORT` calibrates and exports; `--exponents FROZEN_JSON` preserves a
QAT recipe's original exponents. FIT manifests must identify allowed roles,
L256 token inputs and matching array hashes. `IntegerCNN(EXPORT)` executes the
export; `fake_quant_model(student, exponents)` is training only, and checkpoints
must save `student.state_dict()`, not the wrapper's `base.`-prefixed state.

`hardware.py fixture --out RESULTS` makes an untrained, deterministic synthetic
fixture. `hardware.py cases --export EXPORT --prepared FIT --out CASES` makes
real FIT complete-document cases selected by lengths only. `hardware.py prepare
--out BUILD --export EXPORT --cases CASES/cases.bin --lanes 16` freezes sources
and Tcl without invoking a vendor; `hardware.py cpu --out BUILD` compiles and
runs native C++. On WSL, `hardware.py synthesize --out BUILD --seconds 1800`
runs C simulation + HLS synthesis; `cosim` and `export` are separate actions
against that same project. Vendor actions require the shared vendor lock and the coordinator-assigned dispatch/time budget.

`hardware_system.py --hls BUILD --config CANONICAL/configs/project.json --out
SYSTEM --implement` prepares the actual exported HLS IP with CIPS, SmartConnect
and DDR NoC. `--execute --seconds 7200` runs bounded Vivado synthesis and route;
it never invokes programming or a hardware server. Preparation requires real
HLS component metadata and extracts its actual control register map. Physical
board access, measured throughput/energy and downstream protection remain
separately authorized, pending evidence.

Initial native C evidence in `build/kd_cnn_v2/shape64_lanes16_v2` passes nine
full-chain cases, 6,273 arithmetic boundary cases and nine interface/reset
checks. `results/kd_cnn_v2/hardware_fixture_v1` records the five synthetic window
margins, ten exact QAT/reference logits, a nonzero STE gradient, and6,242 Python
rounding/division checks. V2 preserves base training dropout call order and
verifies identical RNG state advancement; evaluation remains exactly matched.

Actual HLS synthesis in `shape64_lanes16_v2` succeeded: estimated period4.475ns,
58,856 LUT,25,359 FF,98 BRAM18,54 URAM and17 DSP. The full valid-window bound is
2,541,118cycles and the model-loading bound2,038,364cycles, without physical DDR
timing. Six residual blocks consume2,265,624cycles; their inner loops achieve
II1. Inspected allocation is16 convolution banks using64 BRAM18, two activation
maps each using16 BRAM18 banks,16 projection LUTRAM banks, and an embedding
memory using54 URAM. The actual summary and generated register-map header are
in `results/kd_cnn_v2/hardware_lanes16_report_v2`. Lane4 synthesis also passed:54,552 LUT,25,477 FF,22 BRAM18,108 URAM,5 DSP and
6,278,462cycles per full window. This is an ordinary unroll/bank tradeoff.
IP export passed. Two UVM co-simulation attempts stopped at the unchanged
Windows2GiB memory floor before any transaction completed; they establish no
RTL arithmetic result. Their logs and costs are preserved. The compact exact-RTL replay passed four real FIT calls using the completed
supervised seed42 runtime, including cold load and a complete two-window
document:4,348,335;2,312,715;2,312,715;2,310,909 fixture cycles. The connected
system passed BD/address validation and generated a pre-implementation XSA;
core synthesis was resource-stopped and route remains pending.

The portable `host_driver` performs bounded polling, cache/barrier sequencing,
model cold loading, warm reuse,16-window chunks, exact maximum continuation,
strict decision checking, and raw-document retention on every error/timeout.
`host_a72` supplies actual BSP MMIO/cache/memory callbacks without executing
them during compilation. Native lifecycle tests use actual generated register
offsets with a fake register/DDR backend invoking the complete C CNN, including
a17-window document. They are not a DDR/RTL/platform validation. Cross-compiled
A72 objects use the existing recorded BSP; a matching new XSA/BSP/ELF, boot
package, model storage, deployed tokenizer/transport and physical permission
remain prerequisites. Model SHA identity validation is the caller's explicit
precondition; a device reset must be completed before `after_device_reset`.

`hardware_rtl.py` snapshots the same generated RTL, trained runtime model and
real FIT cases into a compact one-outstanding AXI memory fixture. Its assertions
check complete document decisions, continuation, masks and all six output
fields. This replaces memory-intensive UVM transaction objects; simulated
cycles describe this fixture rather than physical DDR. The passed representative replay is recorded in
`results/kd_cnn_v2/hardware_rtl_replay_summary.json`. A final nominated model
can use `--qualify-runtime` to append equality/one-unit-below threshold cases,
then `--runtime` to replay those exact parameters without resynthesizing.
These boundary thresholds are verification inputs, not quality calibration. All execution wrappers preserve failed attempts and
honor `BUILD/STOP_REQUESTED` at their next resource check.
