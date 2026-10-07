# Fixed internal AXI data mover

`hardware/gate_data_mover.sv` implements the command plan's four movement operations plus the sequencer's control-word reads. The focused XSIM2025.2 simulation passed all13 cases. This is functional RTL/cycle evidence under an explicit stall model; synthesis, area, routed clock, NoC/DDR integration, complete detector execution and board measurements have not been performed.

## Sequencer interface

Module `gate_data_mover` uses `clk/resetn`, `cmd_valid/ready`, `cmd_op[2:0]`, `cmd_src/dst[63:0]`, `cmd_rows/cols[15:0]`, `cmd_src_stride/dst_stride/limit[31:0]`, and `rsp_valid/ready/status[3:0]`. Command fields latch on acceptance. The sequencer's cached lookup contents must remain stable through completion; command arguments themselves need not remain driven after acceptance. Completion/status remain stable under response backpressure. Reset during a transaction requires coordinated reset of the attached AXI fabric.

| Op | Meaning | Required fields and bounds |
|---:|---|---|
|0 RECT|Row copy, head gather or tile scatter|Rows/cols1..256; strides in bytes, aligned4 and at least one row's width.|
|1 GATHER|Indexed embedding rows of256 raw FP32 words|Rows1..256,cols256,table limit1..30522; `index_addr[7:0]` selects full32-bit `index_value` from the sequencer cache. Source row stride is fixed1024bytes; destination stride is explicit. All indices are checked before any bus traffic.|
|2 TRANSPOSE16|Transpose a strided matrix using a16x16word tile|Rows/cols positive multiples16 through256; destination shape is cols x rows. Both strides are explicit. Each tile is fully loaded, then written in transposed order.|
|3 MASK|Expand one cached binary256-byte key mask across rows|Rows1..128,cols256. `mask_addr[7:0]` selects cached `mask_value[7:0]`. All256 values must be0/1 before writes. Valid1 emits literal0x80000000 (-0), masked0 emits0xff7fffff (-FLT_MAX), matching the saved reference. No DDR mask reread.|
|4 READ_WORDS|Fetch the sequencer's cached inputs or final risk|Rows1,cols1..256,aligned source. `rd_valid/ready/data[31:0]/last` supplies words; last marks the complete command, not each burst. No writes.|

Source/destination addresses are64-bit and four-byte aligned; byte-offset arithmetic remains64-bit. Dimension/count fields retain256 without truncating it to eight bits. Memory transforms reject overlapping bounding spans; in-place transforms are outside this fixed schedule. The logical local data buffer is256x32bits (one tile); copy uses its first16words. There is no additional DDR ID/mask read and no synthesized resource claim.

## AXI and error behavior

The plain `m_axi_*` interface carries32-bit data,64-bit address, eight-bit burst length, three-bit size and two-bit INCR burst fields, with independent VALID/READY handshakes and response codes. There are no transaction-ID ports: this is a single in-order path. The enclosing interconnect's sideband/default mapping is not a NoC implementation in this work.

Every burst has1..16 full-word beats and is split before a4KiB boundary. Rectangle copies choose a chunk satisfying both source and destination boundaries. Tile rows split independently during load/store. AR/AW addresses and W data/strobes/last remain stable while stalled; AW and W acceptance are separate. The implementation is conservative and sequential: at most one transaction total is outstanding, which satisfies the limit of one per direction. Writes use all four strobes because only full words are emitted. READ_WORDS propagates consumer backpressure, but error draining does not depend on consumer readiness.

Status0 is success;1 invalid command/address extent/alignment/overlap;2 invalid cached index/mask;3 read response error;4 write response error;5 RLAST protocol error. Failed outputs are unusable and may contain a prefix of completed work. A read response error drains the accepted burst before completion. Early RLAST terminates with protocol error; missing expected RLAST enters drain until an actual last beat arrives. Write errors complete only after B acceptance. No success/error completion leaves a transaction outstanding in the tested cases. There is no internal deadline for a bus that never responds or never provides RLAST: system watchdog/reset policy remains external and unimplemented here.

## Focused simulation evidence

`hardware/gate_mover_axi_memory.sv` is the separate simulation memory, shared with the sequencer's integration test. Parameters are DEPTH bytes, BASE64 and STALL. Its public byte array `mem[]` permits fixed input initialization and explicit completion-stub risk writes. It bounds every access, checks16-beat/aligned/4KiB/INCR bursts, stable stalled payloads, strobes and WLAST, and injects read/write response or early/late-last errors only when requested. Deterministic AR/AW/W stalls and delayed read/write responses exercise backpressure; stream and completion consumers also stall. Its simulation base0x100000000 tests address bits above32. These delays are not a measured DDR/NoC model.

A later opt-in simulation-only parameter `PIPELINED_READ=1` with `STALL=0` provides the original arithmetic harness's nominal response policy: AR acceptance registers the first R beat, accepted nonfinal R beats refill immediately, and the final W acceptance registers B without extra delay. Default `PIPELINED_READ=0` retains the focused test model. This mode was source-reviewed against `linear_axi_tb.sv`; its actual validation belongs to the integrated sequencer run. The focused stall-model cycle table below must not be added directly to arithmetic observations from the nominal model. No additional mover characterization was launched for this option.

The successful trace and exact source snapshots are `results/goal4/data_mover/2026-09-26T21_18_45_143345_00_00/`; `run.json` contains actual commands, cases and traffic. `scripts/run_data_mover.py` defaults to a preview; `--execute` uses existing xvlog/xelab/xsim only and preserves each attempt. The clock is5ns. Cycles run from accepted command edge to first response-valid edge, excluding the deliberately stalled completion-consumer interval.

| Case | Cycles | Status | Read bursts / words | Write bursts / words | Read / write bytes |
|---|---:|---:|---:|---:|---:|
|rectangle_4k_strides|481|0|11 / 111|11 / 111|444 / 444|
|embedding_gather|3,074|0|48 / 768|48 / 768|3,072 / 3,072|
|transpose_32x32_4k_strides|4,110|0|66 / 1,024|65 / 1,024|4,096 / 4,096|
|mask_expand|1,154|0|0 / 0|32 / 512|0 / 2,048|
|read_words_backpressure_4k|227|0|6 / 70|0 / 0|280 / 0|
|invalid_shape|1|1|0 / 0|0 / 0|0 / 0|
|invalid_index|4|2|0 / 0|0 / 0|0 / 0|
|invalid_mask|257|2|0 / 0|0 / 0|0 / 0|
|read_slverr_drain|36|3|1 / 16|0 / 0|64 / 0|
|write_slverr|63|4|1 / 16|1 / 16|64 / 64|
|early_rlast|5|5|1 / 1|0 / 0|4 / 0|
|late_rlast_drain|38|5|1 / 17|0 / 0|68 / 0|
|recovery_after_errors|64|0|1 / 16|1 / 16|64 / 64|

All copied/transposed/gathered payloads are compared bit-for-bit, with full32-bit cached indices and unequal row strides. The transpose covers four16x16tiles and boundary splits. READ_WORDS checks every word and its final marker under consumer stalls. Invalid shape/index/mask cases issue no memory transactions. The last case proves recovery after each error class. Error-case bytes describe deliberately malformed/failing simulation traffic, not useful detector workload.

Initial attempts remain saved: missing module timescales caused elaboration failure; a direct two-function-call comparison in the TB reported a mismatch while its diagnostic printed identical words. Evaluating actual/expected into separate explicit temporaries resolved that comparison without modifying mover RTL. A subsequent correction removed a one-cycle inclusive offset in the testbench latency counter; the final table uses the corrected accepted-edge boundary. No broader arithmetic tests or experiment campaign was added.

## Handoff and remaining boundary

The sequencer owner can instantiate the unchanged mover and shared simulation memory. Primary command-plan traffic/cycles must come from the integrated schedule; the small cases above are not extrapolated to all227 movement operations. Gathering cached inputs, service programming/completion, final risk handling, internal layout sequencing and controller cost remain the integration owner's work. Physical arena allocation, memory arbitration/bandwidth, transport, system resource/timing feasibility and board execution remain unmeasured. No HLS/Vivado synthesis allowance was consumed, and STATUS/config/the root cost ledger were not edited.
