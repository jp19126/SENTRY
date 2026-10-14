# Span-guard pilot supporting implementation

These sources implement the fixed centered-context contract in
`reports/method_study.md`. They are supporting implementation; the closed
pilot provides the bounded evidence summarized below. The coordinator dispatched
jobs serially within the approved limits. No board or target API action was part
of this work. The command examples preserve reproduction details and do not
authorize any retry or further job. Both hardware correction slots are consumed.

| Final design outcome | Actual retained evidence | Unresolved evidence |
|---|---|---|
| Fused coupled, final correction v4 | HLS top: 104,176 LUT, 50,671 FF, 167 DSP, 99 BRAM18 (49.5 BRAM36 capacity equivalent), 65 URAM. LUT exceeds the 100,000 limit. | RTL stopped at the Windows memory floor with zero of five transactions completed; no verified warm/cold cycles. |
| Conventional buffered coupled v1 | First synthesis failed `SYNCHK 200-41/43` on the old snapshot's pointer selection. C replay and saved encoder/projection/paired-state/head checks passed. | No complete top report or RTL pass. No rebuild allowance remains. |
| Conventional span MLP v1 | HLS top: 95,120 LUT, 37,954 FF, 292 DSP, 119 BRAM18 (59.5 BRAM36 capacity equivalent), 65 URAM. DSP exceeds the 256 limit by 36. | RTL stopped at the Windows memory floor with zero of five transactions completed; no verified warm/cold cycles. |
| SoPa initial design | Seed-42 fixed training was safely stopped after 318.0 seconds on projected budget infeasibility. | No completed final-epoch export, so seed 1337 and the initial hardware design were not run. This is no SoPa quality or hardware failure result. |

Both completed top reports estimate 5.992 ns against the unchanged 5 ns target;
the declared 20% frequency derating gives 7.490 ns, approximately 133.51 MHz.
These are synthesis estimates, not achieved or routed timing. The fused and MLP
RTL terminations are recorded in their respective `vendor_full-flow.json` and
`hardware_terminal_report_v1.json` under `hls_coupled_stream_v4` and
`hls_span_mlp_v1`, within `results/span_coupled_pilot_v1`. Fused elapsed vendor
time was 294.338694 seconds; MLP was 422.374894 seconds. Both returned -15,
and cleanup confirmed no remaining owned process group. MLP's failed observation
was Windows available memory 1,902,804,992 bytes below the fixed 2 GiB floor;
WSL available memory was 12,141,441,024 bytes above its 4 GiB floor. The precise
dominant memory allocation remains unresolved; the no-waveform MLP run also
terminated, so tracing removal is not a demonstrated capacity solution.

Streamed coupled, buffered coupled and span MLP each passed 143 independent C
cases and 46 nonempty saved-window trace comparisons. These are bounded software
integer checks, not RTL or physical-device evidence. The MLP trace audit is
`results/span_coupled_pilot_v1/span_mlp_seed42_trace_v1.json`. No trained SoPa
WFSA-state trace equivalence was established; its existing compile/unit check
does not exercise the full WFSA path.

The SoPa early-stop decision uses two observed 50-batch intervals:
90.735 and 93.593 seconds. Their mean projects 19,833.6928 seconds (about
5.51 hours) for the fixed 10,760-batch recipe, versus its 2,700-second ceiling.
This was a coordinator early termination through the existing pause mechanism,
not an actual dispatch timeout or a partial-checkpoint quality result.
`sopa_seed42_v1/early_stop_decision_v1.json`, `resources.json` and `failure.json`
preserve the decision and exit. No epochs, scoring protocol or comparator were
retuned to produce a substitute result.

The coupled quality/mechanism and resource failures prevent conditional
integration. This implementation is not board-ready: no warm full-check cycle
measurement, routed frequency, physical throughput, energy or complete host
transport cost is available. Unfinished comparative evidence remains unfinished.

`span_guard.cpp` provides a literal completed-context C reference and a streamed
candidate with separate encoder/projection/state actors. The conventional
candidate buffers all local features and projections in on-chip arrays, then
evaluates the same contexts. It does not introduce a DDR spill or per-token host
launch. Both include the embedding table, local convolution, conversion,
recurrence, anchor/head/LUT, complete masking, document maximum and strict
decision. `SPAN_ALGORITHM=0/2/3` selects coupled/MLP/SoPa at compile time;
`SPAN_MAPPING=0/1` selects streamed/buffered coupled computation. Produced HLS
estimates are listed above; complete RTL cycles and routed/physical evidence
remain unavailable.

Only real content IDs enter the hardware window interface: remove CLS/SEP and
padding; `length<=254`. Each descriptor includes input offset, global content
start, document content length and true document-edge flags. Internal anchors
are valid only with their complete `j-31..j+32` footprint. State updates cover
`j-28..j+29`; virtual projection values are zero including bias. Window banks
reset, and duplicate complete footprints retain identical inputs. A nonempty
window without eligible anchors contributes zero risk. Empty documents use the
head-bias lookup through a distinct explicit flag.

At most64 windows are submitted per call; longer documents continue through
additional calls carrying the exact prior maximum. Oversized chunks are
rejected; no window truncation exists. The current fixed pilot's longest
training document has25 windows, as independently checked by the coordinator.
Caller-owned token memory consists of254 uint16 slots per descriptor. The top
validates offsets against that submitted capacity.

The binary model header is eight uint32 words: magic `0x53504731`, version1,
algorithm0/2/3, H58, R32, Z192, vocabulary30522, payload byte count. Payload arrays
are fixed-order little endian, described by `scripts/span_hardware.py` and its
prepared manifest. The packer verifies the original model.bin SHA256 and every
NPZ field against its canonical byte interval. It preserves the original export.
Candidate-only unused comparator arrays are zero; no struct padding is serialized.
Risk lookup entries are positive binary32 bit patterns, so unsigned bit maximum
and strict comparison have the same order as finite nonnegative float risks.

The tested environment uses native Windows Python for preparation because that
environment has NumPy. Preparation reads the saved NPZ export; it performs no
model inference. From PowerShell in the repository, choose a fresh directory
(the following `reproduction_v1` path is an example, not an executed attempt):

```powershell
& 'C:/research/GATE/.venv/Scripts/python.exe' scripts/span_hardware.py prepare `
  --design coupled_stream `
  --export results/span_coupled_pilot_v1/coupled_seed42_v1/export `
  --cases results/span_coupled_pilot_v1/coupled_seed42_fixtures_v1/cases.json `
  --output results/span_coupled_pilot_v1/hls_coupled_stream_reproduction_v1
```

C compilation, replay, Tcl preparation and vendor dispatch use Ubuntu through
WSL, not native Windows Python. Run the following from a WSL shell after
`cd /home/jp19126/Projects/SENTRY`. These driver actions use Python's standard
library; C compilation requires the installed `g++`:

```sh
python3 scripts/span_hardware.py build-c --design coupled_stream \
  --output results/span_coupled_pilot_v1/hls_coupled_stream_reproduction_v1
python3 scripts/span_hardware.py run-c --design coupled_stream \
  --output results/span_coupled_pilot_v1/hls_coupled_stream_reproduction_v1 --trace
python3 scripts/span_hardware.py prepare-hls --design coupled_stream \
  --output results/span_coupled_pilot_v1/hls_coupled_stream_reproduction_v1 \
  --vendor-action full-flow
```

The last command writes Tcl and returns a vendor command; it does not dispatch
it. Only the coordinator dispatches the approved attempt, after resource preflight:

```sh
python3 scripts/span_hardware.py run-vendor --design coupled_stream \
  --output results/span_coupled_pilot_v1/hls_coupled_stream_reproduction_v1 \
  --vendor-action full-flow --max-seconds 7200
```

The wrapper uses `/home/jp19126/Xilinx/2025.2/Vitis/settings64.sh` and the
corresponding `bin/vitis-run`. It requires passed C replay and unchanged source/
model/case hashes, keeps four CPU affinity and thread environment limits, checks
Windows/WSL memory floors every5seconds and terminates only its process group
on time, memory, telemetry or pause failure. The total7200second allowance is
shared across csim/synthesis/cosim in one design directory; a correction uses a
fresh directory and consumes the separately approved correction allowance.
Available design names are `coupled_stream`, `coupled_buffered`, `span_mlp`,
`sopa`. Sources are snapshotted under the fresh output directory before compilation;
subsequent simulation reuses those exact sources. Model load has a separate cold
top invocation. Test cases then use resident weights; RTL transaction timing must
distinguish that cold call from warm checks. Functional C replay alone does not
validate FIFO occupancy, RTL backpressure, physical memory or routed clock.

Run the standard-library report parser in WSL as well. Native Windows execution
can treat existing WSL report paths as missing through UNC path handling; that
is not evidence that synthesis produced no report. The output must be fresh:

```sh
python3 scripts/span_hardware_report.py \
  --run results/span_coupled_pilot_v1/hls_coupled_stream_v3 \
  --output results/span_coupled_pilot_v1/hls_coupled_stream_v3/hardware_report_reproduction_v1.json
```

Independent intermediate comparison uses native Python with NumPy and PyTorch,
reads only saved fixtures/traces, and also requires a fresh output:

```powershell
& 'C:/research/GATE/.venv/Scripts/python.exe' scripts/span_hardware_trace.py `
  --fixtures results/span_coupled_pilot_v1/coupled_seed42_fixtures_v1 `
  --replay results/span_coupled_pilot_v1/hls_coupled_stream_v3 `
  --output results/span_coupled_pilot_v1/coupled_seed42_trace_reproduction_v1.json
```

For the conventional mapping, replace the design with `coupled_buffered` and
use a separate fresh directory. The existing conventional snapshot is
`results/span_coupled_pilot_v1/hls_coupled_buffered_v1`. Reproduction commands
do not authorize further vendor attempts or overwrite existing evidence.

As of the retained v3 attempt on 2026-10-13, the streamed v3 and conventional
buffered v1 snapshots each pass all 143 independent C cases and all 46 nonempty
saved-window intermediate comparisons. Their trace audits are respectively
`results/span_coupled_pilot_v1/coupled_seed42_trace_v3.json` and
`results/span_coupled_pilot_v1/coupled_seed42_buffered_trace_v1.json`; the distinct
empty-document rule is checked by C replay. These are software integer
correctness results, not RTL or device measurements.

The streamed v2 vendor attempt failed before completed synthesis with
`HLS214-211` on the scalar model loader (`Model` field 14, `z_exp`). The corrected
v3 loader preserves serialized bytes and numerical behavior. Its vendor attempt
reached RTL generation but was terminated by the memory guard: available
Windows memory was 2,041,344,000 bytes, below the 2 GiB runtime floor, at
2026-10-13T03:16:16.751720Z. The runner recorded return code -15 and no remaining
owned process group; unrelated processes were not stopped. Both attempts retain
`vendor_full-flow.json` and `vendor_full-flow.stdout.log` in their own directories.
v3 has some completed module estimates, but no complete top synthesis report or
RTL cosimulation report. Full-region cycles, top resources and the common
resource-envelope gate remain unresolved. This is not a board-ready result.

The final streamed v4 correction then completed top HLS synthesis. Its actual
top estimates are 104,176 LUT, 50,671 FF, 167 DSP, 99 BRAM18 (49.5 BRAM36
capacity equivalent), and 65 URAM. The LUT limit is 100,000, so the common
resource gate fails. Estimated period is 5.992 ns at the unchanged 5 ns target;
the declared frequency derating gives 7.490 ns, approximately 133.51 MHz.
Neither this estimate nor the top's latency bounds are measured warm latency.

v4 entered RTL simulation but completed zero of five transactions before the
Windows memory floor stopped it at 2026-10-13T04:25:51.719376Z. Available
Windows/WSL memory was 1,738,067,968/12,254,113,792 bytes; runtime floors remain
2/4 GiB. The runner recorded 294.338694 seconds, return code -15 and no
remaining owned process group. No completed RTL pass or transaction cycle
table exists. Logs, Tcl, input identities, top report and simulation artifacts
are preserved in `results/span_coupled_pilot_v1/hls_coupled_stream_v4/`.
The waveform file was only 10,180,744 bytes; the precise dominant allocation
source is unresolved. Removing waveform logging is not a proven memory fix.
Both hardware correction slots are consumed, and the coordinator decided
not to restart the fused simulation. No board-ready or full-check speed claim
follows from this synthesis result.

The historical instrumentation recommendation for the initial buffered designs
used identical functional RTL cases with `-trace_level none`, omitting
`-enable_dataflow_profiling`.
These optional instruments are unnecessary for complete transaction-cycle and
output checks. Keep deadlock/dependency checks enabled, all numerical and
part/clock settings unchanged, and the same CPU, memory and time limits.
This is an instrumentation reduction, not a guaranteed capacity remedy.
The Tcl generator adds profiling and port tracing unconditionally; the
coordinator preserved its original output and changed only that option sequence
before first dispatch. The buffered WSL sequence is preserved for reproduction:

```sh
python3 scripts/span_hardware.py prepare-hls --design coupled_buffered \
  --output results/span_coupled_pilot_v1/hls_coupled_buffered_v1 \
  --vendor-action full-flow
sed -i.profiled 's/-trace_level port -enable_dataflow_profiling/-trace_level none/' \
  results/span_coupled_pilot_v1/hls_coupled_buffered_v1/full-flow.tcl
python3 scripts/span_hardware.py run-vendor --design coupled_buffered \
  --output results/span_coupled_pilot_v1/hls_coupled_buffered_v1 \
  --vendor-action full-flow --max-seconds 7200
```

The backup is `full-flow.tcl.profiled`; the runner records the actual revised
Tcl hash. This instrumentation reduction was used for the buffered and MLP
initial attempts; neither completed RTL simulation. The SoPa attempt was not
run. These historical command examples do not authorize any retry or job.

The standard-library report-parser regression completed all 12 checks in
`results/span_coupled_pilot_v1/hardware_parser_regression_v2.json`. Synthetic
generated-format fixtures verify separate cold/warm attribution only after
completed Verilog Pass, and rejection of missing/failed reports, missing
tables, duplicate/missing transaction IDs, scalar mismatches and reordered
cases with identical scalar controls. The latter exposed a real attribution
gap: scalar checks alone could attach cycles to the wrong content. The parser
now also verifies the prepared binary SHA and exact JSON repacking against
that binary before promoting cycles. Missing reports return explicit null
cycle fields. The failed v1 regression evidence remains preserved.
Both fresh inspection and the saved terminal v4 report confirm envelope
failure and no promoted RTL cycles. Synthetic cycle values are parser tests,
not FPGA measurements. A bounded reproduction uses WSL and a fresh output:

```sh
python3 results/span_coupled_pilot_v1/hardware_parser_regression_v1.py \
  --output results/span_coupled_pilot_v1/hardware_parser_regression_reproduction_v1.json
```

The regression reads retained reports and temporary small fixtures only; it
does not launch a vendor tool, train or execute a model, or access a board.

The buffered candidate's first vendor attempt subsequently failed synthesis
after 34.217416 seconds with `SYNCHK 200-41/43` at snapshot line 171: HLS
rejected a pointer-valued selection between `proj[p]` and a separate zero
array. The vendor log records only approximately 513 MB peak allocated memory;
the resource guard did not stop this attempt. No top synthesis or RTL pass
resulted. Evidence remains in `hls_coupled_buffered_v1`, including its original
source snapshot, exact Tcl, `vendor_full-flow.json` and stdout log. Cleanup
recorded return code 1 and no remaining owned process group. The correction
reserve is exhausted, so this buffered candidate is not retried.

Before the initial span-MLP build, the shared source removed that unsupported
pointer selection without copying a 64-element vector.
`reference_impl` selects a row index within the single `proj` array and passes
an explicit validity flag. The unchanged eight-lane update selects scalar zero
inputs when the context site is virtual; streamed calls default to valid input.
Recurrence ordering, RNE, clipping, horizon, masks, mapping and constraints are
unchanged, and no extra temporary array or serial copy loop is introduced.
SoPa's selected top path calls its own direct token loop rather than this span
recurrence; its arithmetic source is untouched. Existing snapshots and reports
are not modified by this first-build source correction.

A 5-second-bounded GNU C++ check compiled algorithms 0, 2 and 3 with buffered
mapping and ran only an isolated old-pointer versus new-validity recurrence
comparison. It passed 44,544 exact state-coordinate comparisons per variant
(133,632 total) across lengths 1/30/58/254, both padding sides, lambda endpoints
and clipped state endpoints; the successful command took 2.807373 seconds.
Two earlier shell-invocation errors did not execute those checks. This is
focused C correctness evidence, not full model inference or a SoPa WFSA replay.
The executed driver, captured result, command, input hashes and source snapshots
are preserved under `results/span_coupled_pilot_v1/hardware_pointer_selection_check_v1/`.
The subsequent MLP initial build passed C replay and top synthesis as summarized
above; SoPa remained unavailable because fixed training did not complete.
No further build or retry is authorized by this source correction.

Full-flow first checks all functional C cases, synthesizes once, then runs the
deterministically selected compact RTL subset: short, full, multiwindow and
continued document where available. `cases_rtl.json` and prepared source indices
preserve exact identities. Cold preload is transaction0; warm case transactions
follow in file order. RTL profiling/port traces and actual cosimulation reports
must separate those transactions before reporting warm cycles. `LOOP_TRIPCOUNT`
annotations bound valid lengths,58 active contexts,311 context sites and64
windows for estimation; they do not change functionality or measure cycles.
Conditional/dataflow reports may still show unknown latency. Failed/uncompleted
cosimulation leaves actual full-region cycles unresolved. [AMD UG1399 2025.2](https://docs.amd.com/r/en-US/ug1399-vitis-hls/cosim_design)
documents dataflow profiling; it does not guarantee the runtime of this design.

Case JSON is a list with `windows` entries containing content-only `ids`,
`content_start`, `document_content_length`; each case supplies `threshold_bits`,
`expected_risk_bits`, `expected_anchors`, optional `empty_document` and
`previous_risk_bits`. Expected values come from the independently implemented
integer model, not this C reference. Case packing refuses malformed IDs/intervals.

Trace files have four uint32 header words (magic `0x53505231`, version1, length,
algorithm), then z int8[N,192], projection int32[N,64], a/b int32[N,32], g
int64[N] and eligible uint8[N]. Coupled/MLP a,b,g are meaningful only at eligible
anchors; unused rows are zero/sentinel. SoPa traces store its two WFSA states in
the first16 a/b coordinates at each real token and leave g sentinel; projection
is clipped to the lookup domain. No head clipping is silently changed.

Pilot HLS uses the exact VE2802 part and5ns target with0.5ns uncertainty.
Pilot timing is synthesis-estimated and requires20% frequency derating; only
routed reports can establish achieved timing. DDR/NoC and raw-document host
transport remain explicit unresolved terms. The existing115200-baud UART
cannot be a competitive ingress; a functional kernel does not remove that cost.
