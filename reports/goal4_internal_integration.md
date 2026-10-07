# Fixed L256 internal control and movement integration

This simulation connects the new fixed sequencer and data mover to the actual generated integer and FP32 AXI-Lite register blocks. Explicit completion stubs replace arithmetic datapaths. It is control/layout evidence, not a full detector numerical run or physical DDR measurement.

Execution state: **returned**. Passed: **True**. Evidence: `results/goal4/internal_integration/2026-09-26T21_29_59_195754_00_00/run.json`.

The frozen deployment configuration contains a model ID, W4/W8 group mask, exact FP32 threshold bits and arena base. Changes require reset and a matching cold load of weights/scales. The simulation address is deliberately synthetic and crosses a 32-bit pointer boundary; no address has been allocated on the board.

The nominal memory mode is AXI32/address64, one outstanding transaction per direction, maximum 16 beats split at 4 KiB. First read data and write response are registered; read responses can sustain one beat per cycle. There is no extra DDR delay, arbitration or periodic throttling. This opt-in mode matches the previously used arithmetic harness edge convention; the mover-focused backpressure tests retain their distinct model.

`inclusive_cycles` is the retained JSON field name for the elapsed sampled-edge difference: first result-valid cycle minus input valid/ready acceptance cycle, independent of result backpressure. No +1 is added to that elapsed time. It equals the DUT diagnostic `total_cycles` plus 1 because that state counter excludes the final observation boundary edge. Each stub interval is separately sampled from accepted ap_start while idle to ap_done, matching the existing arithmetic harness convention. The conditional control/layout term is this sampled elapsed difference minus the sum of those exact intervals. No assumed stub constant, polling gap or complete-detector latency is substituted.

The DUT also exposes categories for AXI-Lite request/response states, interrupt-wait states and mover command/response states; these are diagnostic FSM occupancies, not independently additive replacements for the subtraction boundary.

| Case | Status | Sampled elapsed cycles | Stub intervals | Control/layout cycles | Read beats | Write beats |
|---|---:|---:|---:|---:|---:|---:|
| document_first_above | 0 | 5402825 | 3480 | 5399345 | 2228825 | 2260992 |
| document_last_preserves_max | 0 | 5402825 | 3480 | 5399345 | 2228825 | 2260992 |
| new_document_equal_threshold | 0 | 5402825 | 3480 | 5399345 | 2228825 | 2260992 |
| noncontiguous_index | 2 | 1 | 0 | 1 | 0 | 0 |
| arena_overflow | 2 | 1 | 0 | 1 | 0 | 0 |
| arena_alignment | 2 | 1 | 0 | 1 | 0 | 0 |
| invalid_valid_length | 2 | 1 | 0 | 1 | 0 | 0 |
| full_width_token_id | 3 | 732 | 0 | 732 | 600 | 0 |
| full_width_type_id | 3 | 732 | 0 | 732 | 600 | 0 |
| binary_mask_validation | 3 | 732 | 0 | 732 | 600 | 0 |
| finite_scale_validation | 3 | 732 | 0 | 732 | 600 | 0 |
| mover_read_error | 4 | 23 | 0 | 23 | 16 | 0 |
| nonfinite_final_risk | 7 | 5402825 | 3480 | 5399345 | 2228825 | 2260992 |

Each complete schedule must issue 24 integer and 411 FP32 calls, 8,508 argument/start/IRQ-clear writes and 870 completion/status reads. Four IRQ-enable writes occur once during configuration and are excluded from warm-window counts. The TB checks every actual generated-register argument against the plan, both high and low pointer words, mixed group precision, local scales, source IDs, strict threshold equality, document maximum/reset, invalid IDs/masks/scales, address alignment/overflow and nonfinite final risk. Any error returns unusable status and requires reset.

Traffic reconciliation: movement useful reads are 8,912,896 bytes and writes 9,043,968 bytes, totaling 17,956,864. Separately the four cached input reads consume 2,400 bytes and final risk consumes4, giving 17,959,268 mover-port bytes per window. Token/type IDs are 32-bit and validated before use; each is read once, mask bytes are packed four per 32-bit word, and scales are 24 words. Embedding and mask commands use those caches, so there is no repeated DDR ID read. Every layout element is a 32-bit word; the finite valid mask is literal negative zero and invalid mask is 0xff7fffff. The earlier 18,092,032-byte provisional ledger assumed 131,072 bytes of repeated FP32 base-mask reads and 4,096 bytes of wider ID reads. Removing those 135,168 bytes yields 17,956,864 movement bytes; adding the actual 2,404 cached-input/final-read bytes gives17,959,268, a 132,764-byte reduction from the old layout figure. Service-port traffic, command ROM fetches, host transfers and cold loading are outside these mover counters.

No controller/mover synthesis, integrated timing closure, NoC/DDR contention, host latency, board measurement or broad detector numerical acceptance is established. Conditional addition of the control/layout term to the saved sequential service costs remains an analytical model, not observed complete execution.

Offline preflight also compared the fixed plan to `results/goal4/numerical_bridge/ordered_torch_v1/summary.json`: all 411 ordered (operation, rows, width, outputs) tuples and all 24 integer matrix names matched. Movement destination extents passed their row/transpose bounds. The existing calibrated threshold round-tripped exactly to `0x3b955a95`; an arbitrary double `0.1` was rejected. These checks read existing artifacts without inference.
