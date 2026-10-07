# Fixed L256 internal detector command plan

Current implementation update: this plan was instantiated by the fixed sequencer/mover and passed13 RTL control/layout cases. See `reports/goal4_internal_integration.md`. The plan JSON and its executed snapshot retain their original planning metadata; neither is a physical allocation or whole-detector measurement.

This is a concrete buffer and ordered-command plan for the existing BERT-Mini detector. The associated sequencer and mover have separate control-only simulation evidence in reports/goal4_internal_integration.md. This plan is not an allocated DDR region, a NoC design, or a measured complete-checking cost.

The proposed 64 MiB arena uses 42,225,672 bytes through its last buffer, with 24,883,192 bytes spare. Every buffer begins on a 4 KiB boundary. Physical base and allocation remain null. Input IDs, type IDs and mask are supplied by the host; all embedding lookups, detector arithmetic, layout work and final risk/decision control are planned on FPGA.

The generator reads actual successful integer and FP32 control RTL. All three selected integer variants have identical register maps. The source associations and every byte offset, argument, write and completion action are in `results/goal4/internal_integration/command_plan.json`.

## Memory and precision

| Buffer | Relative offset | Bytes | Purpose |
|---|---:|---:|---|
| dummy_zero | 0x00000000 | 4096 | pre-zeroed unused-pointer and zero-mask target |
| token_ids | 0x00001000 | 1024 | host window input, including padded positions |
| type_ids | 0x00002000 | 1024 | host window input |
| attention_mask | 0x00003000 | 256 | host binary key mask |
| word_table | 0x00004000 | 31254528 | frozen word embedding |
| type_table | 0x01dd3000 | 2048 | frozen type embedding |
| position_table | 0x01dd4000 | 524288 | frozen position embedding; positions0..255 used |
| input_scales | 0x01e54000 | 96 | frozen per-matrix A8 input scales, copied once per window |
| layer0_query_weights | 0x01e55000 | 65536 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer0_query_weight_scale4 | 0x01e65000 | 1024 | frozen per-output-channel parameter |
| layer0_query_weight_scale8 | 0x01e66000 | 1024 | frozen per-output-channel parameter |
| layer0_query_bias | 0x01e67000 | 1024 | frozen per-output-channel parameter |
| layer0_key_weights | 0x01e68000 | 65536 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer0_key_weight_scale4 | 0x01e78000 | 1024 | frozen per-output-channel parameter |
| layer0_key_weight_scale8 | 0x01e79000 | 1024 | frozen per-output-channel parameter |
| layer0_key_bias | 0x01e7a000 | 1024 | frozen per-output-channel parameter |
| layer0_value_weights | 0x01e7b000 | 65536 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer0_value_weight_scale4 | 0x01e8b000 | 1024 | frozen per-output-channel parameter |
| layer0_value_weight_scale8 | 0x01e8c000 | 1024 | frozen per-output-channel parameter |
| layer0_value_bias | 0x01e8d000 | 1024 | frozen per-output-channel parameter |
| layer0_attention_output_weights | 0x01e8e000 | 65536 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer0_attention_output_weight_scale4 | 0x01e9e000 | 1024 | frozen per-output-channel parameter |
| layer0_attention_output_weight_scale8 | 0x01e9f000 | 1024 | frozen per-output-channel parameter |
| layer0_attention_output_bias | 0x01ea0000 | 1024 | frozen per-output-channel parameter |
| layer0_ffn_input_weights | 0x01ea1000 | 262144 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer0_ffn_input_weight_scale4 | 0x01ee1000 | 4096 | frozen per-output-channel parameter |
| layer0_ffn_input_weight_scale8 | 0x01ee2000 | 4096 | frozen per-output-channel parameter |
| layer0_ffn_input_bias | 0x01ee3000 | 4096 | frozen per-output-channel parameter |
| layer0_ffn_output_weights | 0x01ee4000 | 262144 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer0_ffn_output_weight_scale4 | 0x01f24000 | 1024 | frozen per-output-channel parameter |
| layer0_ffn_output_weight_scale8 | 0x01f25000 | 1024 | frozen per-output-channel parameter |
| layer0_ffn_output_bias | 0x01f26000 | 1024 | frozen per-output-channel parameter |
| layer1_query_weights | 0x01f27000 | 65536 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer1_query_weight_scale4 | 0x01f37000 | 1024 | frozen per-output-channel parameter |
| layer1_query_weight_scale8 | 0x01f38000 | 1024 | frozen per-output-channel parameter |
| layer1_query_bias | 0x01f39000 | 1024 | frozen per-output-channel parameter |
| layer1_key_weights | 0x01f3a000 | 65536 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer1_key_weight_scale4 | 0x01f4a000 | 1024 | frozen per-output-channel parameter |
| layer1_key_weight_scale8 | 0x01f4b000 | 1024 | frozen per-output-channel parameter |
| layer1_key_bias | 0x01f4c000 | 1024 | frozen per-output-channel parameter |
| layer1_value_weights | 0x01f4d000 | 65536 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer1_value_weight_scale4 | 0x01f5d000 | 1024 | frozen per-output-channel parameter |
| layer1_value_weight_scale8 | 0x01f5e000 | 1024 | frozen per-output-channel parameter |
| layer1_value_bias | 0x01f5f000 | 1024 | frozen per-output-channel parameter |
| layer1_attention_output_weights | 0x01f60000 | 65536 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer1_attention_output_weight_scale4 | 0x01f70000 | 1024 | frozen per-output-channel parameter |
| layer1_attention_output_weight_scale8 | 0x01f71000 | 1024 | frozen per-output-channel parameter |
| layer1_attention_output_bias | 0x01f72000 | 1024 | frozen per-output-channel parameter |
| layer1_ffn_input_weights | 0x01f73000 | 262144 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer1_ffn_input_weight_scale4 | 0x01fb3000 | 4096 | frozen per-output-channel parameter |
| layer1_ffn_input_weight_scale8 | 0x01fb4000 | 4096 | frozen per-output-channel parameter |
| layer1_ffn_input_bias | 0x01fb5000 | 4096 | frozen per-output-channel parameter |
| layer1_ffn_output_weights | 0x01fb6000 | 262144 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer1_ffn_output_weight_scale4 | 0x01ff6000 | 1024 | frozen per-output-channel parameter |
| layer1_ffn_output_weight_scale8 | 0x01ff7000 | 1024 | frozen per-output-channel parameter |
| layer1_ffn_output_bias | 0x01ff8000 | 1024 | frozen per-output-channel parameter |
| layer2_query_weights | 0x01ff9000 | 65536 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer2_query_weight_scale4 | 0x02009000 | 1024 | frozen per-output-channel parameter |
| layer2_query_weight_scale8 | 0x0200a000 | 1024 | frozen per-output-channel parameter |
| layer2_query_bias | 0x0200b000 | 1024 | frozen per-output-channel parameter |
| layer2_key_weights | 0x0200c000 | 65536 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer2_key_weight_scale4 | 0x0201c000 | 1024 | frozen per-output-channel parameter |
| layer2_key_weight_scale8 | 0x0201d000 | 1024 | frozen per-output-channel parameter |
| layer2_key_bias | 0x0201e000 | 1024 | frozen per-output-channel parameter |
| layer2_value_weights | 0x0201f000 | 65536 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer2_value_weight_scale4 | 0x0202f000 | 1024 | frozen per-output-channel parameter |
| layer2_value_weight_scale8 | 0x02030000 | 1024 | frozen per-output-channel parameter |
| layer2_value_bias | 0x02031000 | 1024 | frozen per-output-channel parameter |
| layer2_attention_output_weights | 0x02032000 | 65536 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer2_attention_output_weight_scale4 | 0x02042000 | 1024 | frozen per-output-channel parameter |
| layer2_attention_output_weight_scale8 | 0x02043000 | 1024 | frozen per-output-channel parameter |
| layer2_attention_output_bias | 0x02044000 | 1024 | frozen per-output-channel parameter |
| layer2_ffn_input_weights | 0x02045000 | 262144 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer2_ffn_input_weight_scale4 | 0x02085000 | 4096 | frozen per-output-channel parameter |
| layer2_ffn_input_weight_scale8 | 0x02086000 | 4096 | frozen per-output-channel parameter |
| layer2_ffn_input_bias | 0x02087000 | 4096 | frozen per-output-channel parameter |
| layer2_ffn_output_weights | 0x02088000 | 262144 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer2_ffn_output_weight_scale4 | 0x020c8000 | 1024 | frozen per-output-channel parameter |
| layer2_ffn_output_weight_scale8 | 0x020c9000 | 1024 | frozen per-output-channel parameter |
| layer2_ffn_output_bias | 0x020ca000 | 1024 | frozen per-output-channel parameter |
| layer3_query_weights | 0x020cb000 | 65536 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer3_query_weight_scale4 | 0x020db000 | 1024 | frozen per-output-channel parameter |
| layer3_query_weight_scale8 | 0x020dc000 | 1024 | frozen per-output-channel parameter |
| layer3_query_bias | 0x020dd000 | 1024 | frozen per-output-channel parameter |
| layer3_key_weights | 0x020de000 | 65536 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer3_key_weight_scale4 | 0x020ee000 | 1024 | frozen per-output-channel parameter |
| layer3_key_weight_scale8 | 0x020ef000 | 1024 | frozen per-output-channel parameter |
| layer3_key_bias | 0x020f0000 | 1024 | frozen per-output-channel parameter |
| layer3_value_weights | 0x020f1000 | 65536 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer3_value_weight_scale4 | 0x02101000 | 1024 | frozen per-output-channel parameter |
| layer3_value_weight_scale8 | 0x02102000 | 1024 | frozen per-output-channel parameter |
| layer3_value_bias | 0x02103000 | 1024 | frozen per-output-channel parameter |
| layer3_attention_output_weights | 0x02104000 | 65536 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer3_attention_output_weight_scale4 | 0x02114000 | 1024 | frozen per-output-channel parameter |
| layer3_attention_output_weight_scale8 | 0x02115000 | 1024 | frozen per-output-channel parameter |
| layer3_attention_output_bias | 0x02116000 | 1024 | frozen per-output-channel parameter |
| layer3_ffn_input_weights | 0x02117000 | 262144 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer3_ffn_input_weight_scale4 | 0x02157000 | 4096 | frozen per-output-channel parameter |
| layer3_ffn_input_weight_scale8 | 0x02158000 | 4096 | frozen per-output-channel parameter |
| layer3_ffn_input_bias | 0x02159000 | 4096 | frozen per-output-channel parameter |
| layer3_ffn_output_weights | 0x0215a000 | 262144 | fixed maximum-W8 slot; W4 occupies first NK/2 bytes |
| layer3_ffn_output_weight_scale4 | 0x0219a000 | 1024 | frozen per-output-channel parameter |
| layer3_ffn_output_weight_scale8 | 0x0219b000 | 1024 | frozen per-output-channel parameter |
| layer3_ffn_output_bias | 0x0219c000 | 1024 | frozen per-output-channel parameter |
| embedding_ln_gamma | 0x0219d000 | 1024 | fixed LayerNorm gamma |
| embedding_ln_beta | 0x0219e000 | 1024 | fixed LayerNorm beta |
| layer0_attention_ln_gamma | 0x0219f000 | 1024 | fixed LayerNorm gamma |
| layer0_attention_ln_beta | 0x021a0000 | 1024 | fixed LayerNorm beta |
| layer0_output_ln_gamma | 0x021a1000 | 1024 | fixed LayerNorm gamma |
| layer0_output_ln_beta | 0x021a2000 | 1024 | fixed LayerNorm beta |
| layer1_attention_ln_gamma | 0x021a3000 | 1024 | fixed LayerNorm gamma |
| layer1_attention_ln_beta | 0x021a4000 | 1024 | fixed LayerNorm beta |
| layer1_output_ln_gamma | 0x021a5000 | 1024 | fixed LayerNorm gamma |
| layer1_output_ln_beta | 0x021a6000 | 1024 | fixed LayerNorm beta |
| layer2_attention_ln_gamma | 0x021a7000 | 1024 | fixed LayerNorm gamma |
| layer2_attention_ln_beta | 0x021a8000 | 1024 | fixed LayerNorm beta |
| layer2_output_ln_gamma | 0x021a9000 | 1024 | fixed LayerNorm gamma |
| layer2_output_ln_beta | 0x021aa000 | 1024 | fixed LayerNorm beta |
| layer3_attention_ln_gamma | 0x021ab000 | 1024 | fixed LayerNorm gamma |
| layer3_attention_ln_beta | 0x021ac000 | 1024 | fixed LayerNorm beta |
| layer3_output_ln_gamma | 0x021ad000 | 1024 | fixed LayerNorm gamma |
| layer3_output_ln_beta | 0x021ae000 | 1024 | fixed LayerNorm beta |
| pooler_weight | 0x021af000 | 262144 | fixed FP32 output-major weight |
| pooler_bias | 0x021ef000 | 1024 | fixed FP32 bias |
| classifier_weight | 0x021f0000 | 2048 | fixed FP32 output-major weight |
| classifier_bias | 0x021f1000 | 8 | fixed FP32 bias |
| embedding_word | 0x021f2000 | 262144 | working buffer; overwritten only after dependencies complete |
| embedding_type | 0x02232000 | 262144 | working buffer; overwritten only after dependencies complete |
| hidden | 0x02272000 | 262144 | working buffer; overwritten only after dependencies complete |
| query | 0x022b2000 | 262144 | working buffer; overwritten only after dependencies complete |
| key | 0x022f2000 | 262144 | working buffer; overwritten only after dependencies complete |
| value | 0x02332000 | 262144 | working buffer; overwritten only after dependencies complete |
| context | 0x02372000 | 262144 | working buffer; overwritten only after dependencies complete |
| projected | 0x023b2000 | 262144 | working buffer; overwritten only after dependencies complete |
| residual | 0x023f2000 | 262144 | working buffer; overwritten only after dependencies complete |
| normalized | 0x02432000 | 262144 | working buffer; overwritten only after dependencies complete |
| ffn_pre_gelu | 0x02472000 | 1048576 | working FFN buffer |
| ffn_activated | 0x02572000 | 1048576 | working FFN buffer |
| a8 | 0x02672000 | 262144 | one reusable full integer-input code buffer |
| accumulator | 0x026b2000 | 1048576 | one reusable full integer-output buffer |
| key_head | 0x027b2000 | 65536 | working or final output |
| value_head_transposed | 0x027c2000 | 65536 | working or final output |
| query_slab | 0x027d2000 | 32768 | working or final output |
| attention_tile | 0x027da000 | 32768 | working or final output |
| scores | 0x027e2000 | 131072 | working or final output |
| probabilities | 0x02802000 | 131072 | working or final output |
| mask_slab | 0x02822000 | 131072 | working or final output |
| pooler_output | 0x02842000 | 1024 | working or final output |
| pooled | 0x02843000 | 1024 | working or final output |
| logits | 0x02844000 | 8 | working or final output |
| class_probabilities | 0x02845000 | 8 | working or final output |

Each matrix owns a fixed maximum-W8 slot. W4 occupies its first NK/2 bytes with signed low/high nibbles; W8 occupies NK bytes. Both per-channel weight-scale tables are stored at fixed addresses. A frozen 16-group map selects the weight_bits register and scale pointer. It does not resize or rewire hardware, and is not changed per request. Selected weights/scales are loaded during cold setup, outside warm checking; cold loading and packing remain separately chargeable.

## Exact execution order

1. Validate and cache 256 token IDs, 256 type IDs, 256 mask bytes and 24 input scales. Gather 256 word/type embedding rows on PL. Positions 0..255 read the contiguous position table directly. Expand the binary mask into a 128x256 FP32 slab, reused across every head/layer.
2. Add embeddings in (word+type)+position order and apply width 256 LayerNorm. Each encoder matrix uses all 256 rows: quantize input in capacity-bounded chunks, execute one integer call, then rescale/bias in output-width chunks.
3. Per layer/head, gather the 256x64 key panel and transpose the corresponding value head into 64x256. For each 128-query slab, gather queries, run four 64-key QK panels and explicitly scatter their outputs into 128x256 scores. Softmax divides by 8 and applies the repeated mask. One AV dot produces 128x64 output, explicitly scattered into the full context tensor.
4. Attention projection, residual/LN, FFN input, exact-erf GELU, FFN output and residual/LN follow the saved service order. Separate buffers make every dependency explicit; operations remain sequential.
5. Pool the first token with four 64-output DOT panels, tanh, two-output classifier and ordered two-class softmax. Read class 1 probability on PL. Validate finite [0,1] risk and exact calibrated observed-FP32 threshold bits, initialize max to +0 for window 0 and require subsequent contiguous indices with the same document/base, update document max, and compare strictly greater after the last window. Positive IEEE-FP32 bit order is exact after validation and canonicalizing zero. No early reject skips remaining windows; return source IDs/status with the risk.

The plan has 664 commands: 24 integer calls, 411 FP32 calls,227 movement operations and 2 input/final-control operations. FP32 op counts are {'DOT': 165, 'QUANTIZE': 72, 'RESCALE_BIAS': 72, 'EMBED_ADD': 2, 'RESIDUAL_ADD': 16, 'LAYER_NORM': 18, 'SOFTMAX': 33, 'GELU': 32, 'TANH': 1}.

## Generated AXI-Lite contract

| Argument | Integer offset | FP32 offset |
|---|---|---|
| AP_CTRL | 0x00 | 0x00 |
| GIE | 0x04 | 0x04 |
| IER | 0x08 | 0x08 |
| ISR | 0x0c | 0x0c |
| STATUS_DATA_0 | 0x54 | 0x88 |
| STATUS_CTRL | 0x58 | 0x8c |

Integer pointer pairs are activations 0x10/14, weights 0x1c/20 and output 0x28/2c; rows 0x34, inner 0x3c, outputs 0x44 and bits 0x4c. FP32 op 0x10; pointer pairs x 0x18/1c, y 0x24/28, z 0x30/34, accumulators 0x3c/40, output 0x48/4c, codes 0x54/58; rows 0x60, width 0x68, outputs 0x70, scale 0x78 and dot_bias 0x80. These offsets are parsed from actual RTL, not invented absolute device addresses.

The fixed sequencer performs full register programming: 11 writes per integer call and 19 per FP32 call, including start. It waits on the service's done interrupt rather than repeated polling. It reads AP_CTRL to verify/clear done, reads status and requires zero, then toggles ISR bit 0 exactly once. AW and W channels handshake independently and every write waits for B; read waits for R. Bus errors or invalid status stop the command stream with unusable/error status.

This gives 8073 argument/start writes, 870 completion/status reads and 435 interrupt-clear writes per window; 4 interrupt-enable writes once per reset. The prior provisional ledger's one-read-per-call assumption is replaced by two explicit reads here. No cycles are assigned to these transactions yet.

Explicit layout/gather work reads 8,912,896 and writes 9,043,968 useful bytes per window, plus 2,404 control/input bytes read. These bytes exclude service-port traffic already charged by service execution. They are not a DDR-bandwidth estimate.

## Authorized internal RTL scope

One fixed-study command sequencer, one 32-bit AXI4-Lite control master selecting the two existing control interfaces, and one bounded 32-bit AXI4 memory mover with 64-bit addresses. The mover supports only this plan's row-copy/strided-rectangle, indexed 256-float embedding-row gather, 16x16 FP32 transpose and binary-mask expansion. Use maximum 16-beat bursts, split at4 KiB boundaries, honor strobes/last/independent backpressure, and allow at most one outstanding transaction per direction. A 16x16 word tile and small index/mask/scale storage are bounded implementation choices, not synthesized resource claims.

Host preparation must reject a threshold whose float32 round trip differs from the calibrated Python value. Calibration selects an observed FP32 score, so no interpolated or silently rounded threshold is admitted. The supplied bits remain fixed until reset.

The sequencer latches a deployment model ID, arena base, precision bits and exact threshold until reset; cold setup must load matching weight payloads and scale tables. Per-window sample/document IDs and index are latched on acceptance. The first window initializes max to +0; subsequent indices are contiguous and tied to the same document/base. Final output or any error clears document state, and errors require reset. The sequencer holds these fields and document max; service register values resolve from the arena base and fixed plan. A host-visible start/status mailbox is planned, but its external address and host/Linux transport remain unspecified. The bounded integration simulation connects the sequencer to the actual generated AXI-Lite register modules with bounded completion stubs plus the bounded mover memory model, checking command order/arguments, a mover read error, invalid inputs and final strict-threshold/max behavior. That validates integration control without resynthesizing arithmetic or pretending the stub executes the detector. Reusing the real service RTL for an end-to-end input remains a distinct later integration check.

No new HLS synthesis, vendor project, NoC configuration, driver, boot image, physical allocation or programming is performed by this generator. A conditional control/layout cycle term is measured separately with explicit arithmetic stubs. Physical DDR/host latency, system resources/timing, actual integrated arithmetic execution and broader numerical acceptance remain unresolved.
