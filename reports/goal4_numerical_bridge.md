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

gate_fpga_reference.py now provides OrderedBertMini(model) and reusable ordered dot, LayerNorm, softmax, GELU, embedding/residual and attention operations. It shares the existing loaded model's parameters and saved scales and calls its 24 GateQuantLinear modules unchanged. It is inference-only, requires evaluation mode and explicit FP32 operation boundaries, and rejects compilation/autocast. Independent leading batch dimensions are retained without reductions between examples. Only the actual CPU B1/L256 bring-up is validated; other devices/batches remain unvalidated.

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
