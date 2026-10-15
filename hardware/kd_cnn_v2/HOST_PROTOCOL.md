# Offline A72 UART guard

`host_main.cpp` is a concrete standalone A72 application using the exported control registers and the complete `host_driver`. `hardware_host.py` prepares a source snapshot, runs the native service fixture, cross-links an ELF using an explicitly named existing BSP/linker, or produces an offline raw-document frame. It never opens serial ports or invokes Vitis, Vivado, Bootgen or hardware-server commands.

The application starts UART at115200baud, receives exactly2,036,944 packed model bytes, verifies software SHA256 against compiled `deployment_model_sha`, then returns a64-byte ready response. Each request binds the compiled model identity and threshold. The sender is a trusted preprocessing host: SHA256 detects byte mismatches; it is not authentication and does not prove that supplied token IDs represent the supplied raw text. The offline frame builder uses the hash-verified common tokenizer through `profile_runner.load_tokenizer` and `encode_document`: L256,254 content tokens,64 content overlap, stride190, CLS/SEP, complete tail and PAD0. No alternate tokenizer is introduced.

All multibyte fields use little-endian representation. The84-byte request header is: bytes0..3 `KDC2`;4..7 sequence starting at1;8..11 raw byte count;12..15 window count;16..47 packed model SHA256;48..51 signed Q12 integer threshold;52..83 SHA256 of bytes0..51 followed by the complete payload. Payload order is raw UTF-8 bytes, all windows of256 uint16 IDs, then all windows of256 uint8 masks. The raw bytes are retained unchanged. Sequence must increase exactly. Out-of-order, oversized, corrupt, truncated or identity-mismatched requests cause terminal error with no raw forwarding.

Every response is64bytes:0..3 `KDR2`;4..7 sequence;8..11 status (`0` allow, `1` block, `2` terminal error, `3` model ready);12..15 error code. For decisions,16..19 signed margin,20..23 window count,24..27 chunk count,28..31 cold flag;32..39 model write bytes,40..47 input write bytes,48..55 result read bytes. Ready instead puts the32-byte model SHA at16..47 and threshold at48..51. Only an allow response is followed by the exact original raw payload. Block/error never emits raw bytes. A downstream host must wait for the complete matching response and allowed payload; partial serial delivery is not a complete accepted document. No automatic restart occurs after a request failure.

The hardware driver's per-call cap is16windows, with exact previous-maximum/count continuation and final-only validation. The application buffers a complete request with explicit bounds: at most1MiB raw and4096windows. Over-bound input is rejected without forwarding; upstream ownership must retain it. These transport bounds are not a detector truncation or a scientific task change. Driver support for additional chunks remains independent of this application's capacity.

Memory allocation checked against the connected design's shared64MiB arena: hardware model0x40000000, IDs0x40300000, masks0x40302000, result0x40304000; retained host model0x41000000, raw0x41400000, IDs0x41500000, masks0x41700000. The model extends under0x41200000; raw ends0x41500000; IDs end0x41700000; masks end0x41800000. These regions do not overlap. Control is0xA4000000. The ELF resides in recorded BSP OCM, separate from those buffers. Physical memory ownership, cache attributes and reset still need matching-platform validation.

## Executed qualification

`build/kd_cnn_v2/host_application_v4/test.json` passed three SHA256 vectors and nine focused service scenarios using mock MMIO invoking the full C CNN:17windows/2chunks, equality allow, one-unit-below block, wrong model, corrupted request, truncated input, oversized input, accelerator timeout, request model/threshold mismatch. These are lifecycle fixtures; repeated token windows are not claimed to be the tokenizer output for the fixture label `retained`.

`host_application_v4/link.json` records the successful existing-BSP A72 ELF link. Its entry is0xfffc0000; load extent230,464bytes fits the recorded256KiB OCM. `hardware_host_frame_v2/request.json` records the separately executed real tokenizer frame for synthetic raw text and its model/threshold/digest association. Summary: `results/kd_cnn_v2/hardware_host_application_summary.json`. Earlier preparation/link attempts remain preserved. No new vendor dispatch or board execution was used.

## Commands

Run preparation from WSL with explicit paths:

```
python3 SOURCE/scripts/kd_cnn_v2/hardware_host.py prepare --out BUILD --export EXPORT --cases CASES/cases.bin --registers REGISTER_HEADER --bsp BSP --linker EXISTING_LINKER --threshold INTEGER_THRESHOLD
python3 SOURCE/scripts/kd_cnn_v2/hardware_host.py test --out BUILD
python3 SOURCE/scripts/kd_cnn_v2/hardware_host.py link --out BUILD
python3 SOURCE/scripts/kd_cnn_v2/hardware_host.py frame --out FRAME --raw RAW_UTF8 --canonical CANONICAL --contract CANONICAL/results/dual_track_v1/shared/contract.json --export EXPORT --threshold INTEGER_THRESHOLD
```

These placeholders must name immutable actual artifacts. Final nomination changes deployment constants and requires a fresh link. The current qualification ELF uses supervised candidate weights and threshold0; it is not the final selected detector. A matching new BSP can be generated from the connected XSA, but was not dispatched in this budget window. A bootable package additionally needs actual completed routed PDI and the matching rebuilt ELF; neither old BSP compatibility nor a preimplementation XSA substitutes for this. Physical serial, DDR, cache/reset behavior, latency, energy and downstream protection remain unmeasured.
