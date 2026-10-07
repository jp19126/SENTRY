# Goal 5 adapter correction audit

The September 26 offline correction addresses Goal 5 D8ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“9 before any A/B campaign. Previously, strict pre-QAT B filtering removed every protection-failing candidate from parent pools; the proposals then fell back to random maps. Failed maps remained in the archive, but no QAT shortlist or trained-finalist selector existed. The two historical PTQ anchors fail FPR while the later W8 QAT anchor passes, making this omission material.

Changes in `scripts/search_precision.py`:

- B retains only resource/cap-fitting provisional parents when the corresponding strict pool is empty. Rank is `(max(FPR excess, recall deficit), sum of those violations, latency, -recall, precision tuple)`. An existing strict pool takes precedence.
- Each arm freezes at most two unique QAT maps across caps. A retains accuracy-first; B ranks strict candidates first and may fill the remaining allowance with provisional candidates. The one-epoch common allowance and original floating checkpoint remain mandatory. No training runs are dispatched.
- Post-QAT selection waits for all frozen shortlisted scores. Final B eligibility remains strict; a failed trained candidate cannot become a finalist through a provisional rank. A retains its conventional criterion and reports protection separately.
- Approved arithmetic and execution identities are required for candidates and the floating recall reference. Root added the two nullable configuration fields and left both null. Checkpoint, data/window/evaluation, scheme, calibration and frozen-scale associations remain mandatory. QAT reuse additionally checks the saved one-epoch source/map/calibration metadata. Historical SDPA scores are not relabelled as ordered scores.

The initial 16-map population, seed42, four layerwise crossover children, four mutations per generation, 1/16 group mutation probability, per-arm logical64-map limit and shared hardware profiles are unchanged. This is a completeness correction to the shared A/B adaptation, not C novelty or a protection relaxation.

## Bounded verification actually performed

Native Python command: `C:\research\GATE\.venv\Scripts\python.exe -B scripts/search_precision.py --adapter-check`.

The command exited0 (tool-observed elapsed0.54s). It used the existing `results/goal3/{ptq_w8_a8,ptq_w4_a8,qat_w8_a8,qat_w4_a8}/candidate.json` metric summaries and `floating_reference.json`; it did not load score rows, model weights or a hardware cost table. Explicit in-memory latency sentinels were used only to exercise rank/cap logic and were never saved as costs.

| Check | Observed result |
|---|---|
| Unresolved approval | Rejected |
| Historical record with no arithmetic/execution identity | Rejected |
| Explicit SDPA arithmetic mismatch | Rejected |
| Execution/backend mismatch | Rejected |
| A fixture order | W8, W4 (equal accuracy; latency tie-break) |
| B provisional fixture order | W4, W8 (smaller FPR violation first) |
| Cap restriction | Out-of-cap fixture excluded |
| Existing strict B parent | Provisional parents excluded |
| Frozen shortlist | Two unique maps; repeat request unchanged |
| Strict-plus-provisional shortlist order | Strict first, provisional second |
| Post-QAT rule on historical metric fixtures | W8 passes; W4 fails |

Source syntax compiled without execution. Independent read-only review by the data-design agent found no concrete blocker in the identity, rank, shortlist and post-QAT paths. No broad tests were added. The check prints to stdout only; historical artifacts and shared config were not modified by this task.

## Remaining boundary

Zero comparison maps, zero model scoring calls, zero training calls and zero new hardware lookups were performed. The fixture outcomes are control-logic checks, not campaign results or permission to reuse historical scores. Complete integrated costs/caps, approved arithmetic/backend and a later bounded execution/resume runner are still needed. Floating-reference and candidate scores must come from that same approved computation. Frozen-scale metadata checks associate the existing source/calibration policy; they are not a new tensor-level arithmetic validation. General hardware/reference numerical acceptance remains unresolved.


## Addendum: numerical-reference dependency before Goal 5

**The current ordered CPU reference cannot be approved as hardware-matched from the one saved input.** It establishes a useful implementation and causal diagnostic, not end-to-end agreement with synthesized arithmetic. This addendum is a read-only source/evidence assessment; no new inference, data inspection, score computation or vendor command was performed.

The three evidence levels must remain separate:

- The real-input C-simulation bridge executes the service source using native C++ FP32 operations and the installed HLS C math functions. Separate multiply/add, ascending reductions, `-ffp-contract=off` and `unsafe_math_optimizations=false` constrain ordering. They do not prove that CPU math, the HLS C model and generated floating-point IP return identical bits for division, sqrt, exp, erf, tanh, subnormal/zero handling or every model activation.
- The existing synthesized-service RTL checks pass all nine modes and the 15 primary invocation shapes using signed formula fixtures. They are not executions of checkpoint activation traces through the connected 411-call detector. Their unchanged rule is absolute error <= atol + rtol*abs(reference): DOT3e-5/3e-5, LayerNorm2e-4/2e-4, softmax/GELU/tanh3e-6/3e-5; quantization and selected exact-operation fixtures use zero tolerance. Softmax row sums must additionally be within2e-5 of1. These local operator checks cannot be promoted into a detector-score tolerance or a guarantee of equal downstream A8 codes.
- Ordered CPU reproduces the saved C-service terminal logits/risk on one W8-QAT input, but intermediate differences remain. Thirteen service-input code comparisons are Torch reconstructions, not observed RTL A8 tensors; equal mismatch counts at all24 sites do not establish equal tensors. Ordered CUDA matches all24 CPU A8 tensors/logits on that input but differs by one FP32 ULP in risk and retains an exact-comparison failure. Neither backend has general numerical acceptance.

The known failure mechanism is already decisive: an attention difference of3.5762786865234375e-7 moves a quotient from the exact42.5 tie to42.5000114440918, changing code42 to43 and later logits by about0.0397. Passing a local float tolerance or retaining the same decision on a benign example does not address that amplification.

### Smallest bounded next comparison

Use one prospectively specified numerical bring-up comparison, beginning with the **existing saved real input and frozen W8-QAT checkpoint/scales**. Reuse its97 exports, C bridge, ordered CPU captures, exact integer contract and unchanged synthesized service revision. The missing connection is an actual synthesized-arithmetic replay: feed the real traces through the common service/connected detector, saving the observed outputs at all24 A8 boundaries, integer accumulators, declared FP stage boundaries, logits and final ordered class-1 risk. A controller simulation using arithmetic stubs cannot supply this evidence. An isolated RTL-service replay may diagnose the first failing boundary before the integrated path is ready; testbench CPU orchestration remains numerical validation only, never FPGA offload or timing evidence.

Treat this saved input as case1 of the source document's roughly32 representative-input bring-up (Goal6 item3; config `measurement.bringup_agreement_examples=32`), undertaken when the real arithmetic path and run scope are ready. Predeclare the remaining development-only cases using existing source IDs and saved context: benign/attacked, padding/window coverage, and cases near the existing operating point. Include the known tie case; do not choose or replace cases after observing discrepancies. No final-calibration/test input, retraining, threshold optimization or candidate-search expansion belongs here. Reuse completed operator evidence and stop at the first unexplained mismatch instead of running an entire failing batch. W4 packing already has exact local evidence, but that does not validate W4/mixed full-network activations; use the same bounded input set for a selected W4 or mixed finalist when that precision path is introduced, rather than re-cosimulating every search map.

### Proposed qualification rule, not adopted in configuration

The audit recommends the following stricter qualification; it is not a new project requirement or an approved protocol. Require zero observed A8-code mismatches at every site, zero integer-accumulator mismatches, and exact terminal FP32 logits/risk plus document-max/strict-threshold results. Retain the existing local FP operator bounds as additional necessary checks, not substitutes. Record every FP boundary's actual error/ULP count; for a changed code, record its input, frozen scale, FP32 quotient and distance to the half-integer boundary. Any discrete mismatch, nonfinite/status failure or terminal bit mismatch blocks qualification even when every local tolerance or final decision happens to pass. Keep the threshold's exact FP32 bits and max/strict `>` semantics; no threshold refit can rescue the comparison.

A failed case should first identify the operation/order/library mismatch and update the optional software reference to model the **unchanged** synthesized arithmetic. No new synthesis allowance, score tolerance, scale change or forced code correction is implied. If exact terminal agreement is unattainable with the current software primitives, leave approval unresolved and report the difference; defining an approximate detector acceptance protocol would be a separate substantive decision. Passing32 cases would be bounded empirical qualification of the named checkpoint/arithmetic/device, not proof of equality for all inputs/maps or permission to call the reference universally bit-exact. Goal6 integration/finalist checks still apply.

A separate small software prerequisite remains: `OrderedBertMini` currently requires `GateQuantLinear` at every encoder linear and rejects the ordinary floating model. The new reuse gate therefore also needs a narrowly defined unquantized-linear path under the approved fixed-FP policy, or an explicitly reviewed reference-identity relationship, to obtain the floating recall floor. Historical SDPA recall1.0 must not silently receive an ordered identity. Only after these numerical dependencies are resolved should root set the approved identities and authorize matching development scores; complete costs/caps remain independent requirements. **Zero campaign maps remain consulted.**


## Ordinary floating reference path: implemented bring-up

The previously identified missing ordinary-linear software path is now implemented in `gate_fpga_reference.py`. Explicit construction is `OrderedBertMini(model, encoder_mode='floating')`; the default remains `'quantized'`. Floating mode requires all24 selected encoder modules to be ordinary `nn.Linear`. It uses the existing `ordered_dot`: separate eager FP32 multiply and add, ascending full reduction index0..K-1, and bias after the final addition. Only independent rows/outputs/batch dimensions are vectorized. FFN-output K=1024 is supported as a software reference calculation; this neither changes the physical DOT service's K<=512 interface nor claims a new411-call hardware mapping.

The distinct arithmetic identity is `bert_mini_ordered_fp32_floating_v1`; this bring-up's execution identity is `bert_mini_ordered_fp32_floating_v1__cpu_eager_fp32`. Embedding ordering, attention, LayerNorm, GELU, pooling/classifier and explicit final ordered risk remain shared with the existing wrapper. Architecture checks retain four layers, hidden256, four heads, FFN1024, epsilon1e-12, erf GELU, two classes, absolute positions, eval mode and FP32 parameters. The constructor additionally confirms24 selected linears and requires explicit mode selection; an ordinary floating checkpoint cannot silently enter the default quantized path.

The targeted unchanged-path check compared the edited source AST with the saved validated `ordered_model_v1/gate_fpga_reference.py`. All module-level primitive functions, the quantized dispatch/capture statements and every remaining forward statement were identical; the default mode was explicitly checked. No completed quantized CPU or GPU inference was rerun. Independent read-only review found no concrete branch/order/identity defect before the floating execution.

One CPU forward was executed at2026-09-26T21:37:42.012108+00:00 using only the existing manifest token IDs for `aeslc:train:panus-s_inbox_1.subject:clean` (B1/L256) and the selected `checkpoints/floating/epoch_3`, resolved from the saved Goal3 run. Loading was local-only with offline model access. Runtime was Torch2.9.1+cu128 on CPU with four threads, eager FP32, TF32/autocast disabled. No tokenizer or dataset was loaded.

| Bring-up observation | Result |
|---|---|
| Process exit | 0 |
| Ordinary encoder linears | 24 |
| Native encoder `nn.Linear.forward` calls | 0 (ordered weight operations used) |
| Parameter versions before/after | Unchanged |
| Finite captured floating boundaries | 11/11 |
| A8 captures | 0, as required for floating mode |
| Logits | [2.5572128295898438, -2.4027299880981445] |
| Ordered class-1 risk | 0.006964484695345163 |
| Captured-forward elapsed | 2.172508299991023seconds |

Elapsed time describes this one functional check including diagnostic captures and excluding checkpoint load. It is not a benchmark. No comparison with the W8-QAT score or threshold is implied: this is a different checkpoint/computation, and no threshold was fitted here. The default constructor also correctly rejected this ordinary floating model unless the floating mode was selected.

This closes the software implementation gap only. It does not accept the numerical reference, adopt the preceding proposed acceptance rule, generate a floating recall baseline, or run the32-input hardware comparison. The approved config identities remain null. The floating and quantized arithmetic names are intentionally distinct, so the later scoring protocol must explicitly approve their reference relationship/separate identities; it must not relabel historical SDPA scores or bypass the current fail-closed gate. Prior CPU/GPU quantized outputs and the historical evaluator/quantizer are unchanged. Totals for this task: one already-saved-input forward, zero new examples, zero dataset scores, zero training and zero campaign maps.


## Separate floating score gate: completed plumbing

`scripts/search_precision.py` now accepts the optional `floating` flag in the identity helpers and passes it through score validation. Quantized candidates require `methods.approved_score_identity`; the ordinary floating recall reference requires `methods.approved_floating_score_identity`. Both approval objects must resolve before reuse, and their execution identity strings must match. Their arithmetic identity strings can differ explicitly. Readiness lists both missing approvals; an approved cross-backend pair raises an error rather than silently mixing CPU/CUDA results. Root added the new nullable config object; all four fields remain null.

The execution identity for future scores must identify the common backend/version/settings independently of the quantized versus floating arithmetic name. Existing one-input bring-up labels and historical scores are preserved; this plumbing does not retroactively approve or relabel them. Checkpoint/data/window/calibration gates, strict QAT acceptance and provisional ranking are unchanged.

The one affected `--adapter-check` rerun passed exit0 (tool-observed0.542s), with all prior assertions retained. Added checks reject unresolved floating approval, quantized arithmetic used as the floating identity, missing historical floating identity and a cross-backend approval pair. A fixture with distinct arithmetic identities and the same execution identity is accepted. Fixtures remain in memory; no real identity was approved. No model forward, score rows, dataset scoring, training, hardware lookup or campaign map was added. This closes the code prerequisite only; numerical qualification, matching new score evidence and complete costs/caps remain required.
