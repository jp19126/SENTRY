# One-input numerical bridge: measured drift and first cause

The bridge completes its exact local arithmetic checks, but it does **not** establish numerical acceptance of the original Torch detector. A small attention difference crosses one A8 rounding boundary in layer 1 and is followed by substantial downstream drift. The unchanged decision on this single benign document is insufficient evidence for the low-FPR operating point. A subsequent ordered Torch diagnostic reproduces the C-service logits and risk bit-for-bit on this input, while retaining small intermediate differences and unresolved nonlinear-library equivalence.

## Scope and preserved evidence

The sole input is aeslc:train:panus-s_inbox_1.subject:clean from data/prepared/search_candidate_scoring.jsonl: 175 content tokens, 177 including special tokens, padded to the fixed length 256. The source ID contains "train" because it names the original AESLC source; its prepared experimental partition is search candidate scoring. No final-calibration/test data, new sample export, training or target API was used.

Checkpoint: checkpoints/quantized/qat_w8_a8, loaded with its saved frozen input_scale and weight_scale_8. Reference execution is CPU Torch 2.9.1, four threads, SDPA attention, FP32, evaluation mode, without autocast/TF32. The saved original GPU scores are context, not substituted for the current CPU reference.

The bridge calls the actual panel_cache_batched_attention_v1 C service 411 times. Encoder products use a native int64 CPU reference of the validated W8/A8 contract, checked against signed-high/unsigned-low decomposition and int32 bounds. CPU orchestration, selected embedding gathers and layout copies are testbench operations, not implemented FPGA control, offload or timing evidence.

Primary artifacts are results/goal4/numerical_bridge/manifest.json, bridge_run.json, bridge_summary.json, bridge_vendor.log and the saved binary tensors. Original run: 2026-09-26T20:03:32.331780+00:00; associated FP32 synthesis: 2026-09-26T19:39:34.372157+00:00. C simulation returned zero and reported completion. All 2,359,296 same-input activation codes and 2,359,296 integer accumulators passed exact checks.

A single optional-instrumentation repeat started at 2026-09-26T20:16:01.680154+00:00. Its outputs are separate in diagnostic_first_divergence_v1/. It also returned zero. The summary and all seven original bridge output binaries reproduce exactly, as recorded in repeat_consistency.json. Arithmetic, checkpoint, scales, input and threshold were unchanged. No synthesis was run for this diagnostic.

## Measured drift

| Quantity | Original CPU Torch | Common-service bridge |
|---|---:|---:|
| Benign logit | 2.8854172229766846 | 2.9227449893951416 |
| Injection logit | -2.6302225589752197 | -2.6699333190917969 |
| Class-1 risk/document maximum | 0.004007230047136545 | 0.003711214056238532 |
| Fixed W8-QAT search threshold | 0.004557917360216379 | 0.004557917360216379 |
| Strict risk > threshold | false | false |

Maximum absolute logit difference is 0.03971076011657715. Risk decreases by 0.0002960159908980131, or 7.387% relative to the reference risk. The reference and bridge margins below threshold are 0.000550687313079834 and 0.000846703303977847. The error is approximately 53.8% of this input's original decision margin, although its direction here moves away from rejection.

Saved GPU context was logits [2.8854169845581055, -2.630223274230957] and risk 0.004007228650152683. The CPU/GPU baseline differences are much smaller than the common-service drift.

| Saved boundary | Maximum absolute error | RMS error |
|---|---:|---:|
| Embedding LayerNorm | 1.90735e-6 | 1.42104e-7 |
| Encoder layer 0 | 3.81470e-6 | 1.87276e-7 |
| Encoder layer 1 | 0.1098231 | 0.00214354 |
| Encoder layer 2 | 0.4714493 | 0.0329928 |
| Encoder layer 3 | 0.3244246 | 0.0333467 |

All layer-0 A8 codes match the original software. Layer-1 Q/K/V input codes also match. The first mismatch is one of 65,536 layer-1 attention-output input codes, followed by 4 FFN-input and 38 FFN-output code differences.

The amplified layer-1 error is confined to unmasked token 131: its largest difference is channel 138, 0.1098231; all other rows remain at most 2.86103e-6. At layers 2 and 3, respectively 90 and 223 rows have an absolute error above 1e-4. This cutoff is only a descriptive localization statistic, not a numerical acceptance tolerance.

## First changed code: established boundary crossing

The diagnostic saves layer-1 service Q/K/V outputs and the attention context. A separate saved-tensor calculation reconstructs Q/K/V solely from exported original input codes, W8 codes, frozen scales and biases; it then evaluates only the original CPU SDPA tensor operation on those tensors and the saved mask. It loads no model, checkpoint or new example.

Evidence: diagnostic_first_divergence_v1/saved_tensor_reconstruction.json and first_divergence_analysis.json.

1. Reconstructed Torch Q, K and V outputs are **bit-identical** to all saved service Q/K/V outputs. This rules out an exported-weight, scale or rescale-order discrepancy at this boundary.
2. The reconstructed Torch attention context quantizes to **all 65,536 original exported A8 context codes**, with zero differences. It reproduces the relevant original software boundary.
3. The service context differs from that context by at most 2.86102294921875e-6, with RMS 2.421462695897432e-7. Exactly one context code differs.

The changed coordinate is token 131, channel 114, equivalently head 1/channel 50, using zero-based indices:

| Value | Torch | Service |
|---|---:|---:|
| FP32 attention context | 1.629104495048523 | 1.6291048526763916 |
| Frozen activation scale | 0.03833186998963356 | 0.03833186998963356 |
| Rounded FP32 quotient | 42.5 | 42.5000114440918 |
| Distance above 42.5 boundary | 0 ULP | 3 ULP |
| Nearest-even A8 code | 42 | 43 |

The context difference is only 3.5762786865234375e-7. One quotient ULP here is 3.814697265625e-6. Torch lands exactly on the half-integer boundary, whose even neighbor is 42; the service lands above it and correctly rounds to 43. Both quantizers satisfy the same nearest-even rule on their own FP32 inputs.

Installed Transformers modeling_bert.py uses the expected [batch, heads, sequence, head-width] layout, CPU scaled_dot_product_attention, no inference dropout, additive key mask, and the inverse-square-root head scale. Service QK/8 plus mask is algebraically equivalent for head width 64, and the saved QKV comparison further constrains the discrepancy. No head-layout, mask or scale-placement bug was found.

The evidence locates the first discrete change in attention FP32 output and demonstrates quantization-boundary amplification. It does **not** isolate whether the small attention difference originates in QK/AV reduction ordering, fused operations, softmax reduction/division, or the exp implementation. The installed Torch SDPA Python wrapper does not expose its CPU kernel's exact reduction sequence. Earlier LayerNorm drift is real, but it was quantized away for these layer-1 projections.

This initial diagnostic evaluated service attention versus reconstructed CPU SDPA. The subsequent ordered Torch experiment below additionally executes the declared sequential reductions using the same saved input.

Do not force the one code back to 42, widen a score tolerance, alter a scale or refit a threshold from this example. Those actions would conceal the observed arithmetic difference.

## Executed ordered Torch diagnostic

The independently reviewed scripts/diagnose_ordered_reference.py was executed once on CPU Torch 2.9.1+cu128 with four threads, eager FP32 operations and separate tensor multiply/add boundaries. It reads only the existing 97 exported binaries and saved bridge outputs. It loads no model, checkpoint or dataset and invokes no vendor tools. Results and the executed script snapshot are in results/goal4/numerical_bridge/ordered_torch_v1/; summary.json records completion at 2026-09-26T20:35:02.594532+00:00.

The script mirrors all 411 service calls, including their row/panel segmentation. Calls by op remain [165,72,72,2,16,18,33,32,1]. Integer code-only FP32 matrix products satisfy an absolute-sum bound of at most 2^24 and all 2,359,296 resulting accumulators equal an independent int64 matrix product. The 2,359,296 same-input quantizer checks compare two Torch nearest-even implementations on the same Torch FP32 quotient; they are not new observed HLS-code checks. Checkpoint weights, scales, threshold and prior artifacts remain unchanged.

| Boundary | Ordered Torch versus saved C service: max absolute error |
|---|---:|
| Embedding LayerNorm | 9.5367431640625e-7 |
| Layer 0 | 1.9073486328125e-6 |
| Layer 1 | 1.9073486328125e-6 |
| Layer 2 | 1.9073486328125e-6 |
| Layer 3 | 1.430511474609375e-6 |
| Layer-1 attention context | 4.76837158203125e-7 |
| Final logits | 0, bit-identical |
| Final risk | 0, bit-identical |

Ordered layer-1 Q/K/V outputs are bit-identical to the service. At the identified context coordinate (131,114), ordered Torch produces exactly the service value 1.6291048526763916 and quotient 42.5000114440918, hence code43. It therefore reproduces the boundary crossing and downstream amplification relative to the original SDPA reference. Terminal logits are [2.9227449893951416,-2.6699333190917969] and risk is 0.0037112140562385321, exactly matching the saved C service. Strict rejection remains false at the unchanged threshold.

For all 24 linear inputs, the number of A8 differences from the original software matches the corresponding saved C-service report. Equal counts do not prove that the differing code tensors are identical. Thirteen service pre-quantization arrays are available: twelve Q/K/V inputs from saved normalized layer boundaries and the saved layer-1 attention context. Quantizing those saved C-service arrays with Torch gives zero code differences from the ordered path. These are explicitly labeled **Torch reconstructions**, not actual observed HLS A8 outputs; division identity with HLS remains unverified. The other eleven service input arrays were not saved, so their direct comparisons are null.

This result supports operation-order mismatch as a sufficient explanation for the observed original-SDPA divergence on this input. It is not a unique attribution to one reduction, nor general hardware equivalence. The context still has 12,615 bit differences and the normalized intermediates are not bit-identical. The residual differences already begin at embedding LayerNorm, before attention. Torch versus HLS exp, erf, tanh, sqrt and division remain unverified. The script records numerical_acceptance=null and end_to_end_tolerance=null; neither an exact terminal result on one input nor matching mismatch counts silently establishes detector acceptance.

## Smallest practical reference alignment before Goal 5

The executed fixed-input script establishes the operation definitions for one explicitly selected, inference-only ordered-FP32 reference path around the existing loaded model. The optional adapter and its one-input bring-up are now implemented, as recorded below; no dataset scorer has been switched to it. Keep checkpoint parameters, saved scales, current gate_quant encoder linears, data splits, windowing, max aggregation and strict threshold rule unchanged. Preserve all historical scores in their current directories. Name new outputs separately and state that nonlinear-library equivalence is initially unresolved.

A small fixed-BERT adapter is sufficient; a general exporter or a second evaluation framework is unnecessary:

- **Attention:** replace SDPA in the new path with explicit FP32 multiply then add, accumulating QK in ascending k=0..63 and AV in ascending sequence index. Vectorize batch, heads, query rows and output columns, but not the reduction dimension. Keep each multiply/add as a separate operation; disable autocast/TF32 and graph/compiler fusion. Apply divide-by-8, then the same finite additive mask.
- **Softmax:** use the declared maximum, subtraction, exp, ascending sum and per-element division sequence. Use it for both attention and final two-class risk. The evaluator may accept a narrowly scoped probability function while retaining its existing default; all existing window, aggregation and threshold logic remains shared.
- **LayerNorm:** replace the nine normalization sites in this path with ascending FP32 mean, centered-square sum, division by256, epsilon1e-12, square root, centered division, gamma multiply and beta add. Do not use the fused native LayerNorm kernel as an arithmetic reference.
- **FP32 dense work:** pooler and classifier must use ascending, separately rounded products/adds, with bias only after the reduction. Preserve pooler tanh and the installed embedding addition order.
- **GELU:** express the actual service grouping x/float32(sqrt2), x*0.5, erf, add1, multiply. Native fused GELU is not an order-matched substitute.
- **Integer projection/rescale:** retain current GateQuantLinear semantics and frozen scales. Its factor=(input_scale*weight_scale), integer-to-FP32 conversion, product and bias addition already have the declared grouping; the diagnostic establishes exact QKV outputs at the first divergence. Keep the existing exact integer-reference check.

This is practical to develop with Torch tensor operations: non-reduction dimensions remain batched, and the implementation is restricted to the actual four-layer BERT. It will be slower than fused SDPA/LayerNorm; no speed measurement or dataset-scoring cost is claimed. No retraining or hardware resynthesis is needed merely to implement and inspect this software path.

The explicit risk hook is necessary because gate_eval.py:237 currently applies torch.softmax after model.forward. Replacing only the model's forward path would leave this final numerical operation unmatched. Independent source review agrees with this scope; it does not add an execution result.

**What ordered Torch does not yet solve:** Torch exp, sqrt, erf, tanh and floating division are not proven bit-equivalent to the installed HLS library or synthesized arithmetic IP. CUDA and CPU kernels can also differ. Replacing only SDPA covers the observed attention entry point incompletely and does not address LayerNorm, nonlinear grouping or pooler/classifier reductions. Therefore the new path must initially be labeled an ordered reference with pending nonlinear validation, not bit-exact hardware emulation.

The single-input ordered alignment has now been executed as recorded above. The optional adapter retains those operations and reproduces the saved ordered results exactly on this input; broader use still requires the unresolved numerical boundaries to be reviewed. Report and resolve remaining operator-level differences without silently allowing them through an invented detector-score tolerance. Reuse the direct FP32 RTL operator checks to distinguish C-library and synthesized-IP behavior. Only after the reference and remaining numerical boundaries are reviewed should any full development/search scoring or Goal-5 protection decisions use it. A single unchanged final decision does not remove that prerequisite.


## Optional loaded-model reference and bring-up

gate_fpga_reference.py now provides OrderedBertMini(model) and reusable ordered dot, LayerNorm, softmax, GELU, embedding/residual and attention operations. It shares the existing loaded model's parameters and saved scales and calls its 24 GateQuantLinear modules unchanged. It is inference-only, requires evaluation mode and explicit FP32 operation boundaries, and rejects compilation/autocast. Independent leading batch dimensions are retained without reductions between examples. That bring-up validated only CPU B1/L256; the later bounded CPU batching and ordinary floating qualifications are recorded below and do not grant general device/batch equivalence.

The arithmetic identity is bert_mini_ordered_fp32_frozen_quant_v1. OrderedBertOutput exposes logits and risk; wrapper.predict_risk(logits) supplies the explicit ordered two-class probability calculation. Historical gate_quant.py and gate_eval.py were not edited. A future scorer must deliberately consume output.risk or predict_risk; passing the wrapper into the unchanged historical score_documents function would still recompute its ordinary torch.softmax and would not fully select the new arithmetic identity.

After independent read-only source review, scripts/bringup_ordered_reference.py --execute loaded only checkpoints/quantized/qat_w8_a8 locally and the saved manifest token IDs/masks, then ran one forward. No tokenizer, dataset, training, threshold refit, API or vendor invocation was involved. Evidence and executed source snapshots are in results/goal4/numerical_bridge/ordered_model_v1/; summary.json records completion at 2026-09-26T20:46:07.963982+00:00.

- All 65 parameter/input export comparisons were bit-exact before inference, including W8 codes, frozen scales, biases and selected embedding rows.
- All 97 original exports were consumed for the fixed-case comparisons. Frozen quantization parameters and existing quantized execution modes remained unchanged.
- All 11 captured floating tensors and all 24 A8 tensors reproduce ordered_torch_v1 with **zero differing bits**.
- Both final logits and risk again match the saved C service bit-for-bit. The small intermediate C-service differences and all library-equivalence caveats from the prior diagnostic remain.

The initial process wait yielded at 10 seconds; total process time including imports/checkpoint loading was not instrumented. Saved source-snapshot and summary timestamps span 1.2315 seconds for the post-load forward, comparisons and artifact writes. This is execution bookkeeping, not a detector timing measurement. No repeat was performed solely to collect timing.

The implementation integration check succeeded. Numerical detector acceptance remains null, with no invented tolerance and no development/search scoring started.

## Optional CUDA comparison on the same input

After independent review of the device-selection and timing changes, scripts/bringup_ordered_reference.py --execute --device cuda performed exactly one forward of the same saved B1/L256 input. The existing CPU output directory was preserved. CUDA outputs and executed source snapshots are separate in results/goal4/numerical_bridge/ordered_model_cuda_v1/; summary.json records completion at 2026-09-26T20:55:31.961767+00:00.

The actual device was NVIDIA GeForce RTX 4070, compute capability 8.9, with Torch 2.9.1+cu128 and CUDA 12.8. Execution used eager FP32, matmul precision highest, TF32 off and no autocast/compiler. The CUDA execution identity is bert_mini_ordered_fp32_frozen_quant_v1__cuda_eager_fp32; the separate ordered_cpu primitive identity names the saved comparison reference rather than the execution device. No setup, checkpoint, scales, precision map, default evaluator behavior or acceptance rule changed.

The first forward took **2.833694100030698 seconds**, with CUDA synchronization immediately before and after. It included Python/kernel launch overhead and all diagnostic intermediate/code captures, and excluded checkpoint loading and post-forward CPU comparison transfers. There was no warmup, repeated forward, new example or batch duplication. This records successful executable bring-up; it is not a throughput benchmark or evidence of a GPU advantage for development scoring.

| Comparison with saved ordered CPU | Observed result |
|---|---:|
| All 24 A8 tensors (2,359,296 values) | Exact |
| All 24 W8-code tensors recomputed on CUDA | Exact |
| Final logits | Bit-identical |
| Final risk absolute difference | 2.3283064365386963e-10 (one FP32 ULP) |
| Maximum difference among captured floating boundaries | 1.9073486328125e-6 |
| Captured scalar values with different bit patterns | 146,235 |

The JSON field named bit_differences counts scalar values whose bit patterns differ, not individual changed bits. CUDA risk was 0.0037112138234078884 versus CPU/C-service 0.0037112140562385321. The unchanged strict threshold decision is false. All 65 CPU parameter/input preflight checks and all 97 original export comparisons completed; frozen quantization parameters and execution modes remained unchanged.

CUDA agrees bit-for-bit with the saved C-service embedding LayerNorm and all four layer-end tensors, as well as layer-1 Q/K/V outputs and final logits. However, its layer-1 context still differs from the C service in 22,101 scalar values, with maximum absolute difference 4.76837158203125e-7. The known (131,114) context coordinate remains exactly 1.6291048526763916, and all captured A8 codes agree with ordered CPU. These observations do not establish unique operator-level causality or general CUDA/HLS equivalence.

The run saved completed numerical evidence and then returned exit 1 because exact CPU reproduction failed. That comparison rule was not relaxed to force a pass. Numerical acceptance and general backend eligibility remain null. No GPU backend has been selected for future scoring, and no dataset scores were generated.

## 2026-10-08 board-free numerical preparation and actual arithmetic

The 32 development inputs were selected prospectively before new replay discrepancies: 32 unique source groups, 16 benign and 16 attacked, from search temporary-threshold/candidate-scoring partitions only. Case1 is the existing known attention/A8 tie input. The record freezes66 primary-length windows and includes historical operating-point neighbors, historical score quantiles, padding and length strata. Selection uses historical scores for coverage, not approval or threshold refitting. Final calibration/test and NotInject remain closed. Evidence: `results/goal4/numerical_bridge/development_cases_v1/selection.json`; selection code `scripts/prepare_numerical_cases.py`. Root independently checked IDs/groups, quotas, masks/extents and exact case1 token correspondence.

The new explicit `risk_function` hook in `gate_eval.score_documents` retains the historical native-softmax default. `scripts/score_qualified_development.py` deliberately selects `OrderedBertMini.predict_risk`; full scoring remains blocked before model/data work until both approved identities resolve. Resumed/completed scores must bind exact prepared IDs, labels and source groups, as well as existing checkpoint/calibration/backend identities. The one saved-input shared-evaluator bring-up passed with exact logits/risk/document max and token IDs, one forward, no threshold fit. Evidence: `ordered_document_scorer_v1/summary.json`. This closes evaluator integration only.

New actual integer arithmetic used completed-prefix A8 and unchanged static64 query weights for rows0..15, twice. All4,096 int32 outputs per call were exact; each measured251,993cycles. New actual integration separately executed controller/mover input loads, gathers and mask expansion, then the unchanged synthesized FP embedding addition and interrupt completion. All32,768 output floats and the reconstructed complete64MiB arena were exact. Evidence: `rtl_query_integer_rows16_v1/run.json`, `results/goal4/real_arithmetic_integration/20261007T235814_007874Z/run.json`, and `connected_service1_check.json`. The partial connected interval452,336cycles covers setup/movement/one service, not a full detector or physical DDR.

The bounded numerical policy retains exact same-input signed integer/packing/A8 semantics, original local FP operator bounds and softmax row-sum checks, explicit downstream A8 and final-logit/risk propagation reporting, and stopping at the first unexplained discrepancy. No universal terminal-bit requirement or detector-wide tolerance is adopted. Ordinary floating L256 qualification has now been accepted under its separate bounded review below; quantized/mixed acceptance and their scoring approvals remain unresolved. Further checked replay must feed every operation the preceding actual output.

## Checked connected C qualification and independent review, 2026-10-08

The fresh checked build `build/development_qualification_bounds_v2/compiled.json` preserves the canonical testbench and successful synthesized FP32 service; its generated testbench adds independent long-double same-input formulas, exact FP32 add/rescale formulas and original operator/softmax-row-sum checks to every actual service call. First W8 case passes411 calls/all24 A8 tensors/logits/risk exactly. Ordinary floating case1 passes267 calls while retaining a2.3283064365386963e-8 risk difference and nonidentical logits; this is not terminal bit equality. W8case2/window0 passes411 calls/all24 A8 tensors and bit-identical risk, with equal oneULP shifts in both logits. Each actual tree is under `development_qualification_{w8,floating}_v1/bounds_v2/`. Earlier unbounded diagnostics remain preserved.

Independent capable review is saved in `connected_bounds_review_v2/independent_review.md` and `audit.json`, with runnable pure-stdlib saved-byte audits. It verifies exact consumed floating checkpoint parameter bytes, embeddings/mask/threshold/current compiled/reference source associations and actual per-call coverage. All256 floatingcase1 embedding-LayerNorm rows reproduce the unchanged FP32 formula using adjacent denominator encodings, including the first changed and maximum-error rows. This localizes an elementary boundary mechanism; exact internal sqrt-versus-variance attribution remains inference because internal denominator traces were not captured. It does not attribute every later layer error to a unique operator.

The coordinator authored and validated case-specific resume records for first floatingcase1 and W8case2, preserving actual errors and null global acceptance. A separate prospective floating-only policy, `connected_bounds_review_v2/recognized_floating_dispatch_policy_v1.json`, binds frozen identity/selection, exact classifier source and independent evidence. Future nonidentical floating terminals may be retained as known propagation pending final review only after complete original operator/completion/strict-decision/source/input checks and mechanical all-row embedding-LayerNorm recognition. Each classification keeps full row evidence and original stage/logit/risk errors. New A8 differences, failed classification, changed associations or operator/process/service/decision failures still stop. No detector epsilon or universal terminal-bit requirement is adopted. Each mode requires its own complete reviewed qualification and all 32 document maxima/strict decisions before its scoring identity can be approved; the ordinary floating decision below does not approve quantized paths.

Ordinary floating L256 qualification completed all 66 windows and 32 documents and was accepted by the coordinator in `results/goal4/numerical_bridge/connected_bounds_review_v2/floating_final_root_acceptance_v1.json`. Independent `floating_final_independent_v1/review.json` and `supplement.json` reproduce all 1,386 canonical stage-error metrics and validate 17,622 original operator checks. Nine terminals are bit-exact; 57 retain explicitly reviewed or recognized differences. Maximum window/document risk error is 2.5704503059387207e-7, reported as an outcome rather than a new tolerance. Document maximum winners did not change, and all 32 strict decisions agree at the identical frozen threshold. This is bounded development-reference acceptance, not universal library equality, calibrated protection or full-detector RTL/board evidence.

The approved floating development-scoring identity is arithmetic `bert_mini_ordered_fp32_floating_v1` with execution `torch_2.9.1+cu128__cpu_eager_fp32_threads4_tf32off_autocastoff_flushdenormalfalse__document_score_batch32_tail_actual`. Actual B8/B32 captures agree with all 66 B1 windows. New matching development protection scores remain to be generated; historical SDPA results cannot be relabelled. The first committed floating case ran CPU7 before guard deployment; the remaining 65 ran guarded CPU6. Retained failed resource attempts are not successful qualification evidence, and no retroactive guard claim is made.

W8 and W4 checked-C states currently retain 15 and 9 windows, respectively, with global quantized/mixed approvals null. Genuine single-forward captures and exact casewise resolutions are attached for W8 case 9/window 2 (`connected_bounds_review_v2/w8_case09_window002_review_v1/reviewed_resolution.json`) and W4 case 6/window 0 (`connected_bounds_review_v2/w4_case06_window000_review_v1/reviewed_resolution.json`). Their errors and valid-token A8 crossings remain preserved. The actual states are `development_qualification_w8_v1/bounds_v2/qualification.json` and `precision_qualification_v1/uniform_w4_qat/bounds_v1/qualification.json`. Continuation uses existing compatible executables and current locks/resource guards; no operator/process/service/decision failure is waived.

True-L128/L512 CPU exports and B8/B32 checks passed for W8 and ordinary floating on the same two frozen documents. L128 has 21 genuine windows per mode and L512 has 4; repeated batch rows are explicitly marked. Independent review checked 3,648 tensor comparisons and 537,921,792 bytes exactly, including the L512 valid key at 511. Evidence: `real_length_source_review_v1/actual_export_independent_v1/independent_review.md` under `results/goal4/numerical_bridge/`. These checks establish bounded CPU export/batching consistency only. No true-length C or quality approval exists, and some benign maxima exceed the transferred L256 threshold.

Mixed PTQ initial proposal 6, tuple `4488884884844448`, completed 66 genuine B1 exports and three B32 batches (32+32+2), with all 2,046 tensor checks exact in `precision_qualification_v1/campaign_ptq_p4488884884844448/`. Own-map batching does not grant C agreement, shared PTQ/QAT coverage or scoring approval; no campaign protection score was consulted and no logical map consultation occurred. Complete checked-C 66-window/32-document evidence, document maxima/strict decisions and an explicit coordinator scope decision remain required. The prospective 174-reference-capture preparation is pending and is not included in completed evidence. A/B/C comparison remains unexecuted and C remains null.

### Actual paired-input continuation, 2026-10-08T08:25:19.607500+00:00

W8 has22/66 saved connected checked-C windows and W4 has23/66; neither score identity is approved. Root attached exact-case foundational bounded resolutions for W8cases11,14,15 andW4case13, preserving stricter prospective-class failures. Each reviewed case retains411 original local checks,2359296 exact same-input quantizer/accumulator comparisons and21 independently recomputed error metrics. The accepted174 additional reference captures supply genuine paired non-QKV inputs; they are diagnostic forwards, not new training or numerical acceptance.

W8case14/window0 first crosses at layer0_attention_output token196/channel139: observed0xbf36c757/0xbf36c755, frozen scale.03173240274190903, FP32quotients-22.500003814697266/-22.5, correctcodes-23/-22. All337978changedpairs (274044valid/63934padding) have observed inputs and full-array exact requantization; layer2maxerror1.1211591958999634 andlayer3.7721477448940277 remain reported. Own terminal recipes reconstruct exactly, while crossbackendrisk differs4.707020707428455e-6. Both complete one-window strict decisions accept at unchangedthreshold; no detector epsilon or unique downstream causal chain is inferred. See `results/goal4/numerical_bridge/remaining_reference_preinputs_v1/future_case_audit_v1/w8_case14_window000_v1/`.

W8case15 has zeroA8changes; actual captured-tail logits reconstruct exactly but diagnostic math.exp hostrisk0x3af984cb differs from actualC0x3af984cc by oneFP32encodingstep. The reference ownrecipe is exact. Both own-saved-logit Decimal80softmax checks meet originallocalbounds; actual final two-output C check/row sum pass. Actual/ref risks.0019036768935620785/.0019036759622395039 and margins-.0026542404666543007/-.0026542413979768753 give a complete one-window strictacceptboth. Failedreadyflag/uncapturedexp-sumintermediates remain explicit in `.../w8_case15_window000_v1/`. W4case13 similarly retains a referencehostrecipe1ULP mismatch with both own-logit checks passing;1979padding/0validchanges, complete strictacceptboth. W4case14crossbackendlogits differ although risk is exact; case15both exact; case16both differ under recognized class and documentmax remains null at1/3windows. No protection-quality or fullRTL/physical result is asserted from these partial diagnostics.


### 2026-10-08T08:39:46.781098+00:00 â€” W8case16/window0 exact diagnostic resolution

All24 A8 arrays match; original411same-input calls,21stage metrics and exact2359296quant/int checks pass. Actual hostrecipe softmax risk differs1ULP0x3adcdc40â†’actualC0x3adcdc41; readyfalse retained. Original localtwo-outputsoftmax/rowsum and bothOWNsavedlogit Decimal80 checks pass (errors5.5943243e-11/4.6153783e-10, unchangedbound~3.0505509e-6), nouniqueelementarycause or fittedexpectation. Windowstrictacceptboth/margins-.00287288648542/-.00287288660184; only1/3 documentwindows, maxima null. Root exact resolution permitsdiagnosticcontinuation only at remaining_reference_preinputs_v1/future_case_audit_v1/w8_case16_window000_v1/reviewed_resolution.json. Numerical/scoreapprovalnull, prospectivepolicyunchanged. W4case16/window1 valid-tokenpropagationstop retained forindependentreview.


## Live bounded qualification checkpoint â€” 2026-10-08T10:44:01.675142+00:00

Floating66windows/32documents remain accepted; quantized global acceptance and score approvals are pending. Actual current W8/W4 saved counts are41/49 of66each. W8case22/window0 rootrawchecks establish only3valid L3Q/K/V rounding differences; its completeone-window ownmaxrisk.9998235106468201 is bitexact withstrictrejection. W8case23/window0 has462128validpropagatedcodes afteraoneULP L0QKV crossing -19.49999809265/-19.5 givescorrect-19/-20. Everychangedinput requantizesexactly and bothownterminalrecipes reproduce; CROSSriskerror5.245208740234375e-6 withunchangedwindowreject. The nexttwo case23windows passunchangedprospectivechecks; finaldocumentmax is rechecked in complete-cohort review. W4case24/window2 has6217validchanges atinitialL2QKV -9.5/-9.49999809265â†’-10/-9, ownterminalrecipesexact/CROSSriskerror1.5652913134545088e-6, unchangedwindowaccept. Existingrootcasewise records in `remaining_reference_preinputs_v1/future_case_audit_v1/` preserve original411per-window localchecks,21canonicalmetrics,2359296same-inputquant+accumulators/source/packing/process/guards. Prospective exactA8W8/zero-valid-changeW4classes remainunchanged. Compoundhiddenerrors/CLSdifferences remain reported, with no detector epsilon or universalterminalbits. TrueL128/512 actualnativeB1/B8/B32exports and routedbuilds/bootpackages exist but Ccompile/replay/shapeacceptance stillpending. Historicalthreshold agreement is bounded implementation evidence; newmatching development scores and any finalist threshold/checkpoint need their own later binding/checks. No physical detectornumerical or performance claim.


### 2026-10-08T12:14:40.465502+00:00 â€” exact uniform-W4-QAT bounded final acceptance

All66L256windows/32documents completed naturally. Independent final saved-byte/source/process/packing/guard review passed27,126originaloperatorrecords and1,386canonicalmetrics;155,713,536same-input quantized values/accumulators remain exact. PropagatedA8differences1,118,615 (878,581valid/240,034padding), largestfullhidden.96701580286026/finalCLS.5261246562004089 remain explicit. Twelve exactrootresolutions and43unchangedpaddingclass records remain bound to their actual cases. All66window/32documentstrict decisions and ownmaxwinnersagree; maximumrisk/docerror.00120466947555542 at strongly rejectedcase4, nearestsampledmargin.00299232080578804. ActualownB32proof3records/66rows/all31tensors exact; B8notclaimed. Root separately reread all132riskbinaries and recomputed32ownmax/winners/margins. Decision `results/goal4/numerical_bridge/connected_bounds_review_v2/uniform_w4_final_root_acceptance_v1.json` accepts only exactuniformW4QAT bounded numerical/scoringcomputation; genericquantized/mixedapproval remainsnull untilW8/common-scope decisions. This does not meet full-development protection eligibility, qualify unseenmaps, erasebenignfalsepositives or assertRTL/physicalagreement.


## 2026-10-08T13:13:36.959168+00:00 â€” exact W8-QAT L256 bounded qualification accepted

Root acceptance `results/goal4/numerical_bridge/connected_bounds_review_v2/uniform_w8_final_root_acceptance_v1.json` reuses the complete actual66/32Crun and one independent76.606182s saved-byte review. All27126originaloperatorchecks/1386metrics/155713536sameinputcodes andaccumulators pass. ActualB32forwards32/32/2 andB8 remain31-tensor byte-exact. All66window and32document strictdecisions/winners agree; root independently reread132riskbinaries. Maxriskerror .0005084958393126726, nearestobservedmargin .0008467033039778471. PropagatedA8differences2036679(valid1785485,pad251194,CLS7015), largesthidden1.133270025253296 andfinalCLS.2678844630718231 retained, with28casewise and20capturedtail records. Actualprocessscope65returnedrecords+1historicalcompilefirst;64guardrecordedwindows/33logs, no failed observations inreviewedlogs. These are bounded observations, not detector epsilon/universal equality. ExactcheckpointB32computationqualified; genericquantized/mixed approvals andprotectioneligibility remainnull, full matching scores andtrueLqualification stillpending.

## True-length diagnostic adapter and completed L512 floating replay

The coordinator accepted the exact four-scope source in `real_length_bringup_v1/shape_adapter_source_v1/root_source_acceptance_v1.json` and separately adopted `root_prospective_policy_v1.json`. The adapter uses the existing compiled L128/L512 floating/W8 routes and original inspector, bounds, source/input/process/guard and strict-decision checks. Its first actual L128 floating dispatch, c18_w004, retained a failed own-risk host recipe and the original numerical stop; exact read-only reconstruction and original-bound supplements were accepted as negative runtime evidence. The subsequent c18_w005 positive class reproduced both own logits/risk recipes exactly while retaining nonidentical host TANH outputs. Both cases passed all 105 original calls, 21 metrics, all 128 LN rows, own pre-TANH DOT and original TANH bounds. Their separate `first_actual_runtime_independent_v1/root_runtime_acceptance_v1.json` and `first_positive_runtime_independent_v1/root_runtime_acceptance_v1.json` accept callback/persistence behavior, not broad numerical or quality eligibility. No failed flag, detector epsilon or historical entry was silently changed.

Ordinary-floating L512 checked C now completes all four genuine windows and both own document outcomes in `real_length_bringup_v1/l512_floating/c.json`. Native saved-byte reads reproduce the raw maxima: benign case1 C/reference 0.006964461412280798/0.006964484695345163, both won by c01_w000; attacked case18 0.9993444085121155 for both, won by c18_w000 across its three windows. At the unchanged transferred L256 threshold 0.004557917360216379, both backends reject both documents. Benign margins +0.002406544052064419/+0.002406567335128784 therefore remain a matching false positive; attacked margins are +0.9947864911518991. These source-driven diagnostic outcomes do not recalibrate the threshold or establish protection quality. Original errors and explicit casewise/prospective records remain retained; numerical acceptance is null.

The complete-only four-route saved-byte reviewer has source acceptance at `real_length_bringup_v1/final_saved_review_source_v1/root_source_acceptance_v1.json`, supported by `independent_source_review_v1/independent_note.md`. Its actual complete review remains unexecuted: all 50 windows (21+21+4+4), eight route outcomes on the same two documents, original checks, raw document maxima and own B8/B32 bytes must be revalidated together after completion. Source and individual runtime acceptance do not grant global true-length numerical/scoring approval, full-detector RTL agreement or physical measurement.

The one selected L128 W8 c18_w003 reference-preinput capture remains unexecuted. Its exact cross-OS stat/original-Linux validation fix and prepared plan are retained under `real_length_bringup_v1/selected_reference_preinputs_v1/`; the observed 2026-10-08T14:01:39Z Windows availability of 5,733,195,776 bytes failed the unchanged 8 GiB native preflight despite no blocking C/vendor process. No forward was launched and no new preinputs were written. The source correction does not relax that resource gate or resolve the pending paired-input cause.

## Current true-length and mixed diagnostic handoff, 2026-10-08 around 14:44 UTC

This checkpoint supersedes the earlier live counts while preserving their attempt history. Actual shape-adapter positive and negative runtime behavior has been independently reviewed and accepted by the coordinator. That acceptance concerns original-inspector-first dispatch, retained false classifications, provenance and persistence; it does not approve the four true-length numerical identities, mixed computation, or protection quality.

The current states under `results/goal4/numerical_bridge/real_length_bringup_v1/` are:

| Route | Saved windows | Current result and remaining evidence |
|---|---:|---|
| L128 ordinary floating | 19/21 | Stopped at `c18_w016`. All 105 original calls and 21 canonical metrics pass. The actual own-logit host risk recipe is one FP32 encoding below its saved risk; both own-saved-logit Decimal80 checks satisfy the original softmax bounds. The failed recipe/class flag remains false. Document18 is partial at 17/19, so its document maxima remain null. |
| L512 ordinary floating | 4/4 | The four genuine windows and their two document outcomes are complete; there are no additional cohort windows. The complete four-route reviewer has not run, and numerical acceptance remains null. |
| L128 W8 | 6/21 | Stopped at `c18_w003`; the selected genuine reference-preinput capture is still pending. No missing paired input is inferred from a later case. |
| L512 W8 | 2/4 | Stopped at `c18_w000` with 848,597 propagated A8 differences. Its saved-byte review is ongoing; required reference preinputs are absent. The attacked document is incomplete, and no complete document maximum is extrapolated. |

Each route's actual state is its `c.json`. The L128 floating one-encoding review and unchanged false flags are retained in `true_length_case_review_v1/l128_floating_c18_w016/`, including `own_saved_logit_decimal80.json`. The current L512 W8 saved review is under `true_length_case_review_v1/l512_w8_c18_w000/corrected_prior_exact_association_v3/`. The earlier complete L512 floating document maxima and matching benign false positive remain as reported above; completion does not recalibrate the transferred L256 threshold.

Mixed PTQ tuple `4488884884844448` currently has 19/66 saved windows in `precision_qualification_v1/campaign_ptq_p4488884884844448/bounds_v1/qualification.json` and is stopped at case12/window0. Its independent saved-byte review is complete in `remaining_reference_preinputs_v1/future_case_audit_v1/mixed_ptq_case12_window000_v1/independent_review.md`; the coordinator's exact-case decision remains in progress. All 411 original checks, 21 canonical metrics and 2,359,296 same-input quantizer and accumulator values pass. All 62,080 propagated code changes have genuine paired preinputs: 11,172 valid and 50,908 padding. The first valid layer2 Q/K/V crossing, derived own-weight integer/rescale deltas and substantial final CLS error remain explicit. Both own terminal recipes reproduce exactly. The complete single-window benign document is accepted by both backends, but the actual prospective classification remains false and stopped. The generic audit's tail-reproducibility readiness flag is not prospective dispatch permission.

The selected native capture retains its unchanged 8 GiB Windows preflight floor. A fresh direct native observation around 14:44 UTC found only approximately 6.3298 GiB available; no capture was launched and no new reference preinputs were written. This resource observation does not resolve either pending true-length paired-input gap. It is separate from the lower original runtime memory floors.

The account observation at 14:39:15 UTC was 66% weekly usage consumed and 34% remaining. The user's stop threshold remains 90% consumed. These are scheduling facts, not numerical evidence or permission to expand the research budget.

Global true-length and mixed numerical/scoring approval remains null. Matching full-development scores, the A/B/C comparison and finalist selection remain unexecuted. Existing exact L256 floating/W4/W8 bounded acceptances do not transfer to another length, precision map, checkpoint or quality claim. No physical-board detector result follows from these diagnostics.

## Current offline numerical handoff â€” 2026-10-08 15:24 UTC

This checkpoint supersedes the earlier 14:44 UTC live counts. It records saved evidence only; no model, C, vendor, board, capture, or API work ran for this update.

All three L256 numerical scopes have root acceptance: floating, uniform W4-QAT, and uniform W8-QAT. Generic quantized acceptance and mixed-map approval remain null. The exact true-length and mixed artifacts below do not inherit those L256 decisions.

| True-length route | Saved windows | Current evidence |
|---|---:|---|
| L128 floating | 21/21 | The same two diagnostic documents have matching strict decisions and winners. Benign case1 covers2/2 windows and is a matching false positive; attacked case18 covers19/19 windows and has bit-exact maximum risk0.9994778037071228, winnerc18_w000. |
| L512 floating | 4/4 | Same two documents and matching strict decisions/winners. Benign 1/1 is a matching false positive; attacked 3/3 is bit-exact at max risk 0.9993444085121155, winner c18_w000. |
| L128 W8 | 6/21 | Stopped at c18_w003 pending genuine reference preinputs. |
| L512 W8 | 2/4 | Stopped at c18_w000 after 848,597 propagated A8 changes; its genuine reference preinputs are absent. |

The floating outcomes are still diagnostics, not numerical approval or protection-quality evidence. The L128 benign outcome remains a false positive at the unchanged transferred threshold 0.004557917360216379. The separate complete four-route review of all 50 windows remains pending, as does global true-length numerical/scoring acceptance.

The selected true-length W8 capture helper's post-adapter source is accepted (17 author and 21 independent checks); both current literal plans are accepted: L128 c18_w003 v3 and L512 c18_w000 v1. Earlier L128 v1/v2 plans remain preserved with stale helper text. The latest idle preflight at 15:10:11 UTC observed 5,854,224,384 Windows physical bytes, 22,315,327,488 WSL bytes, no C/vendor blockers, and the unchanged Windows 8 GiB floor as the sole blocker. No capture ran and no resource setting changed. The selected plan and preflight are under `real_length_bringup_v1/selected_reference_preinputs_v1/`; the actual route states are the four `c.json` files under `real_length_bringup_v1/`.

At this checkpoint, mixed PTQ tuple `4488884884844448` has26/66 saved windows; its case17/window0 numerical stop has received an exact root resolution and the bounded C batch has resumed. The selected capture and root source acceptances are in `real_length_bringup_v1/selected_reference_preinputs_v1/post_adapter_association_fix_v1/root_source_acceptance_v1.json`; its current observation is `root_idle_preflight_observation_v2.json`. Mixed case reviews are under `remaining_reference_preinputs_v1/future_case_audit_v1/`; pending true-length capture plans remain in the selected-capture directory above.

Case15/window0 and case16/window1 have root-reviewed bounded continuations under their original local bounds. Case16's later full three-window benign read is complete: both backends have bit-exact max risk 0.0013956876937299967, winner window 0, and both accept. For case17/window0, root raw review records 1,301 padding-only A8 changes, zero valid/CLS changes, and the first layer1 Q/K/V crossing at token 187/channel 247 (quotients -8.50000095367 and -8.49999904633 round to -9 and -8). Final logits, risk, and the complete one-window attacked document max are bit-exact at risk 0.9986974596977234; both reject. The case's original failed prospective classification remains false, and the casewise root resolution is attached to active parent 21749. These case records do not approve the mixed tuple or broaden policy.

The next evidence handoff is the complete four-route saved-byte review after the remaining route evidence is resolved. The selected capture remains gated by the 8 GiB Windows physical-memory floor; the latest preflight did not authorize or launch it. The complete mixed 66-window review, matching development scores, and actual A/B/C campaign remain pending. At 15:19:51 UTC weekly usage was 67% consumed / 33% remaining; the user's stop threshold remains 90%.

### Exact mixed qualification checkpoint — 2026-10-08T16:29:58.447290+00:00

Mixed PTQ4488884884844448 has41/66saved windows. Case23/window0 retains440valid A8 changes reconstructed directly; case23/window2 retains the one-encoding host-risk recipe failure, with both own-logit original bounds passing. Exact root records are `results/goal4/numerical_bridge/remaining_reference_preinputs_v1/future_case_audit_v1/mixed_ptq_case23_window002_v1/root_raw_review_v1.json` and the preceding window000 review. Complete document23 is rejected by both, with identical rawmaximum.9984835982322693 and solewinner0. These casewise resolutions authorize diagnostic continuation only. Parent68196/max6 is active; complete mixed saved review and generic quantized scoring approval remain pending. True-length progress remains floating21+4 complete and W8 6+2 saved; the same two diagnostic source documents and original memory guard apply. No new capture or physical measurement occurred.


### 2026-10-08T17:44:05.909753+00:00: Mixed document24 complete; distinct elementary and propagation diagnoses

The exact mixed PTQ4488884884844448 scope now completes document24 onfourwindows. RootreadbackfindsbothownrawMAX.998872697353363, solewinner0, strictreject atthreshold.004557917360216379. Nonwinningriskpairs remain [.0013265307061374187/.0013265300076454878, .0009423434967175126/.00093036302132532, .000944164174143225/.000944164174143225]; this is onecomplete diagnosticdocument, notdevelopmentprotection evidence.

Window1 required a nativepure-tail REPLAYfromgenuine savedCLS/ownparams. All256source-orderpoolerpre values reproducebitexact; twoTorch-vs-hostTANH one-encodingdifferences(channels6/142) explainhostreferenceclassifierlogit1. Nativepost→originalclassifier reproducesBOTH oldsavedlogits+risk exactly. NewDecimal80same-inputTANH/classifier/riskchecksmeetunchangedoriginalbounds; originalforwardintermediateswerenotcaptured. Window2 has402511validA8changes/0padding/1338CLS, beginningwitha genuineL0attentionoutput FP32divisiontie→-6/-7. Allownintegerdeltas/pairedquantizers check, but subsequent propagation islarge(layer2max1.0886800289154053/finalCLSmax.2245563268661499). Localservicebounds do not implysmallcross-inputerror. Theterminalhostrisk1encodingLOW remainsfalse; bothownsaved-logitDecimal80checks meetoriginalbounds. Exactcasewise diagnosticresumesretainallfalseflagsandgenericapprovalNULL. Rootraw/evidence: `results/goal4/numerical_bridge/remaining_reference_preinputs_v1/future_case_audit_v1/mixed_ptq_case24_window002_v1/root_raw_review_v1.json`, `root_complete_document_after_resume_v1.json`; window1nativeevidence is in sibling mixed_ptq_case24_window001_v1/native_tail_replay_source_v1/.


At2026-10-08T18:15:16.856197+00:00, actual saved-operand primitive replays for L128c18_w003 and L512c18_w000 each reproduced all31 original raw tensors before12 REPLAY preinputs were written. Independent checks verify original backend/source/98operands/guards/process handoff and re-quantize all3/12MiB outputs exactly. These are deterministic original-primitives replay evidence, not original-forward hook captures or synthesized RTL. The L128 complete paired audit resolves14 valid changes arising at the L3FFNinput rounding boundary while retaining all unchanged input/scales/operator bounds, false host-TANH flag and partial4/19 document extent. Its logits/risk are bit identical despite non-CLS propagated error; exact-case rootresolution permits continueddiagnostics, not universal equivalence. L512pairedcausalreview is still pending. No numerical contract or score identity is broadened by the replay allocation.


### Exact numerical handoff — 2026-10-08T18:56:51.690198+00:00

NewL128c18_w008 REPLAY provenance rootaccepted at `results/goal4/numerical_bridge/real_length_bringup_v1/selected_primitive_c18_w008_v1/root_runtime_acceptance_v1.json`:31oldoutputs exact,12newinputs/786432finitevalues independentlyrequantized; genuine98/source/request/native3ID+WSLhandoff pass. REPLAY is not original-forwardhookcapture or detectorapproval. L12811/21under7change pairedreview; L5123/4own3QKVpairresolution attached/finalwindowexecuting. Mixed52/66case25w6rootresolution attached; complete doc25bothownmax.999457061290741/winner0/strictreject. All1033codechangespadding/0CLS, all5hiddenCLSbitexact; separatecrosstail discrepancyretained. Earlierperwindowdifferences/falseclasses preserved. Globalmixed/trueL/scoreapprovals null.


### 2026-10-08T19:38:45.706299+00:00 - bounded true-L512 W8 completion

Actual genuine full512 joint-attention W8 route complete4/4 after root-only final case resolution; no extra C forward to finalize. Final c18_w002 masked primitive REPLAY reproduced31 original tensors before12 arrays and independent requantization, root actual cleanup/request/source acceptance. Complete paired raw arrays reconstruct361009changes=221747valid+139262padding/751CLS; first t296c32 quotients48.5000038147/48.5 produce49/48. All256 first signed-row projections rechecked; derived outputs remain distinguished from observations. Large intermediate max.9817926288/finalCLS.1782804728/crosslogits.0022230148 and unequal risk remain reported. Complete attacked3-window document chooses c18_w000/reject both; benign1-window risk.003711214056238532/accept both. This is bounded two-document true-length bringup, not quality approval. See `results/goal4/numerical_bridge/real_length_bringup_v1/true_length_case_review_v1/l512_w8_c18_w002_v1/replay_paired_inputs_v1/` and current c.json. L128W8 now18/21/new c18_w015 terminal stop; final50-window review remains pending.

### True-length final bounded acceptance (2026-10-08T20:21:24.789964+00:00)

Root acceptance: `results/goal4/numerical_bridge/real_length_bringup_v1/final_saved_review_source_v1/actual_complete_review_v1/root_bounded_acceptance_v1.json`. All50 windows/four routes complete,18458 original calls/1050 canonical metrics/43646976 identical-input codes and accumulators each; actual B8/B32 saved equality3648comparisons537921792bytes. All8 own-reference document-route strict decisions and winner sets agree on SAME TWO documents. Three benign false-positive outcomes remain, and W8 propagated1209630code changes/large hidden errors are retained. Scope is bounded bring-up, not broad32document true-length quality or generic score approval. The earlier unperformed independent floating512 full-checkpoint reread remains explicit; compiled/source/export association is preserved without fabricating it.


### Final mixed PTQ bounded acceptance — 2026-10-08T21:20:35.952516+00:00

Exact4488884884844448 L256 completes66windows/32documents. Main and independent reviews pass27126 original checks/1386canonicalmetrics/155713536same-input codes and accumulators each plus own B32sizes32/32/2. Root reread all raw maxima/winners/margins/strict decisions; all32 agree.1482297propagated A8changes (1339206valid,143091padding,4998CLS), maximum valid hidden1.0987344980239868, finalCLS.31253311038017273 and riskerror.0005818372592329979 remain; nearest documentmargin.0008822008967399597. Five benign diagnosticfalsepositives are not an FPR estimate. Evidence: results/goal4/numerical_bridge/mixed_campaign_review_v1/final_saved_review_source_v1/actual_complete_review_v1/root_bounded_acceptance_v1.json. A separate root decision permits PTQ development shared implementation reuse and matching scoring, with own-map proofs/fullquality mandatory; futuremixedQAT unapproved. Nophysical/deployment/protectionapproval.
