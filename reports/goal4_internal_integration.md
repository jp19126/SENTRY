# Fixed L256 internal control and movement integration

This simulation connects the new fixed sequencer and data mover to the actual generated integer and FP32 AXI-Lite register blocks. Explicit completion stubs replace arithmetic datapaths. It is control/layout evidence, not a full detector numerical run or physical DDR measurement.

Execution state: **returned**. Passed: **True**. Evidence: `results/goal4/internal_integration/2026-10-08T02_09_52_942713_00_00/run.json`.

The frozen deployment configuration contains a model ID, W4/W8 group mask, exact FP32 threshold bits and arena base. Changes require reset and a matching cold load of weights/scales. The simulation address is deliberately synthetic and crosses a 32-bit pointer boundary; no address has been allocated on the board.

The nominal memory mode is AXI32/address64, one outstanding transaction per direction, maximum 16 beats split at 4 KiB. First read data and write response are registered; read responses can sustain one beat per cycle. There is no extra DDR delay, arbitration or periodic throttling. This opt-in mode matches the previously used arithmetic harness edge convention; the mover-focused backpressure tests retain their distinct model.

`inclusive_cycles` is the retained JSON field name for the elapsed sampled-edge difference: first result-valid cycle minus input valid/ready acceptance cycle, independent of result backpressure. No +1 is added to that elapsed time. It equals the DUT diagnostic `total_cycles` plus 1 because that state counter excludes the final observation boundary edge. Each stub interval is separately sampled from accepted ap_start while idle to ap_done, matching the existing arithmetic harness convention. The conditional control/layout term is this sampled elapsed difference minus the sum of those exact intervals. No assumed stub constant, polling gap or complete-detector latency is substituted.

The DUT also exposes categories for AXI-Lite request/response states, interrupt-wait states and mover command/response states; these are diagnostic FSM occupancies, not independently additive replacements for the subtraction boundary.

| Case | Status | Sampled elapsed cycles | Stub intervals | Control/layout cycles | Read beats | Write beats |
|---|---:|---:|---:|---:|---:|---:|
| document_first_above | 0 | 5403985 | 3480 | 5400505 | 2228825 | 2260992 |
| document_last_preserves_max | 0 | 5403985 | 3480 | 5400505 | 2228825 | 2260992 |
| new_document_equal_threshold | 0 | 5403985 | 3480 | 5400505 | 2228825 | 2260992 |
| noncontiguous_index | 2 | 1 | 0 | 1 | 0 | 0 |
| arena_overflow | 2 | 1 | 0 | 1 | 0 | 0 |
| arena_alignment | 2 | 1 | 0 | 1 | 0 | 0 |
| invalid_valid_length | 2 | 1 | 0 | 1 | 0 | 0 |
| full_width_token_id | 3 | 752 | 0 | 752 | 600 | 0 |
| full_width_type_id | 3 | 752 | 0 | 752 | 600 | 0 |
| binary_mask_validation | 3 | 752 | 0 | 752 | 600 | 0 |
| finite_scale_validation | 3 | 752 | 0 | 752 | 600 | 0 |
| mover_read_error | 4 | 28 | 0 | 28 | 16 | 0 |
| nonfinite_final_risk | 7 | 5403985 | 3480 | 5400505 | 2228825 | 2260992 |

Each complete schedule must issue 24 integer and 411 FP32 calls, 8,508 argument/start/IRQ-clear writes and 870 completion/status reads. Four IRQ-enable writes occur once during configuration and are excluded from warm-window counts. The TB checks every actual generated-register argument against the plan, both high and low pointer words, mixed group precision, local scales, source IDs, strict threshold equality, document maximum/reset, invalid IDs/masks/scales, address alignment/overflow and nonfinite final risk. Any error returns unusable status and requires reset.

Traffic reconciliation: movement useful reads are 8,912,896 bytes and writes 9,043,968 bytes, totaling 17,956,864. Separately the four cached input reads consume 2,400 bytes and final risk consumes4, giving 17,959,268 mover-port bytes per window. Token/type IDs are 32-bit and validated before use; each is read once, mask bytes are packed four per 32-bit word, and scales are 24 words. Embedding and mask commands use those caches, so there is no repeated DDR ID read. Every layout element is a 32-bit word; the finite valid mask is literal negative zero and invalid mask is 0xff7fffff. The earlier 18,092,032-byte provisional ledger assumed 131,072 bytes of repeated FP32 base-mask reads and 4,096 bytes of wider ID reads. Removing those 135,168 bytes yields 17,956,864 movement bytes; adding the actual 2,404 cached-input/final-read bytes gives17,959,268, a 132,764-byte reduction from the old layout figure. Service-port traffic, command ROM fetches, host transfers and cold loading are outside these mover counters.

No controller/mover synthesis, integrated timing closure, NoC/DDR contention, host latency, board measurement or broad detector numerical acceptance is established. Conditional addition of the control/layout term to the saved sequential service costs remains an analytical model, not observed complete execution.


## Required length implementation closure — 2026-10-08T10:33:09.513933+00:00

Actual static64 L128 and L512 implementations now meet the5ns routed objective with clean timing checks and DRC. L128 WNS/WHS+0.029/+0.014ns, LUT66305/FF60398/BRAM103/DSP111/URAM0. L512 WNS/WHS+0.017/+0.017ns, LUT69267/FF63417/BRAM119/DSP111/URAM0. Sourceplans have302/3076commands respectively and full-length attention; both have realPDI/XSA and embedded NoCOPTIMAL seven achieved positive read/write paths, analytical only. Exact root acceptances are `build/vek280_system/l128/lanes64/20261008T065700_369382Z_implement/root_routed_completion_acceptance_v1.json` and `build/vek280_system/l512/lanes64/20261008T093118_039498Z_implement/root_routed_completion_acceptance_v1.json`. The L128 stockresume returned0 while its outer guard stopped only on XMLoutputdeclarations; that failure remains unchanged with a factualcompanion. L512 stockbuilder and ownedguard returned0,551originalfloorsamples/noOOM, all42PGs/340PIDs absent after3219.408s under7200. Requiredlength initialbuildlist2/2used; no renewed characterization/profile sweep. Genuine trueL nativeB1/B8/B32exports/hostELFs/arenas are reused; Ccompile/replay/numerical shape acceptance still pending. No physical allocation, detectorperformance or modelapproval is inferred.
