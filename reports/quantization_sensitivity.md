# Quantization sensitivity — Goal 3 development evidence

The bounded experiment found **no additional missed attacks** among 2,738 attack documents after the shared threshold-refitting procedure. Quantization and one-epoch QAT changed benign false positives and score calibration. The W8A8 QAT anchor met the primary empirical search-scoring protection constraint (recall 1.0; FPR 13/1,369 = 0.9496%). The W4A8 QAT anchor did not meet the 1% scoring-FPR requirement (20/1,369 = 1.4609%). This is a development result from one seed, not evidence of general protection, a new allocation method or an FPGA speedup.

## Scope and numerical implementation

All variants start from `checkpoints/floating/epoch_3`, the unchanged Goal 2 selection. The 24 encoder linear modules form the fixed 16 groups: Q/K/V together, attention output, FFN input and FFN output for each of four layers. W4A8/W8A8 refer only to these modules. Embeddings, biases, attention matrix products, residual additions, LayerNorm, GELU, softmax, pooler and classifier remain FP32. Future FPGA accounting must include these operations; this experiment does not establish an integer-only detector.

`gate_quant.py` uses the installed Brevitas 0.13.4 `IntQuant` and its straight-through estimator, with a small fixed-BERT linear wrapper. The software route is **integer-valued FP32 emulation plus STE QAT**, not accelerated INT4/INT8 inference. The implementation follows the [Brevitas integer-quantizer interface](https://github.com/Xilinx/brevitas/blob/master/src/brevitas/core/quant/int.py); the installed version and executed settings are recorded in `results/goal3/run.json` and `reports/requirements-goal3.txt`.

| Quantity | Executed rule |
|---|---|
| Weight code | Signed W4 [-8,7] or W8 [-128,127]; zero point 0 |
| Weight scale | Per-output-channel FP-checkpoint max-absolute weight divided by positive maximum (7 or 127); separate scales for W4 and W8; frozen throughout QAT |
| Activation code | Signed A8 [-128,127]; zero point 0 |
| Activation scale | One static scale per selected module input; valid-token training max-absolute activation / 127 |
| Calibration | First 256 windows from seed-42 shuffled training documents; masked padding excluded; one calibration shared by every map and both QAT runs |
| Rounding and clipping | Nearest, ties to even; saturating full signed range; all-zero tensor scale 1 |
| Accumulation | Signed integer dot product with int32 range; software emulates it exactly in FP32 for K ≤ 1,024 |
| Output | FP32 accumulator × (input scale × output-row weight scale), then FP32 bias |
| Execution controls | TF32 matmul and cuDNN TF32 disabled; float32 matmul precision `highest`; AMP disabled |

Every signed A8×W8 product has absolute value at most 16,384; with K≤1,024, the sum of absolute products is at most 2²⁴. Thus the FP32 integer-valued products and partial sums remain exactly representable. This bound is enforced, not assumed for arbitrary matrices. The independent CPU reference computes int64 then checks the int32 range. The W8 atomic decomposition is signed high nibble `floor(w/16)` and unsigned low nibble `w−16*high`; using signed low nibbles would be incorrect. The [Quasar-ViT decomposition](https://arxiv.org/html/2407.18175) motivates this reference check, but its original A6 packing/resource results do not transfer to this A8 implementation.

`reports/quant_arithmetic_check.json` records five passed checks: nearest-even/signed limits, hand-calculated W4/W8 products, all 256 W8 values with representative A8 values for nibble decomposition, the K=1,024 exactness boundary and K=1,025 rejection, and Brevitas/reference agreement with finite nonzero STE gradients and fixed scales after an optimizer step. One actual 47-token training email also passed both full CPU paths with exactly identical logits (`results/goal3/real_input_agreement.json`). No FPGA agreement is claimed.

## Shared threshold protocol and uniform results

Each threshold is fitted only on 1,355 benign documents in `search_temporary_threshold`; the primary allowed empirical FPR is at most 1%, using `score > threshold`. All five uniform variants attained 13/1,355 = 0.9594% on that threshold-fitting subset. They were then scored on a separate 4,107-document set containing 2,738 attacks and 1,369 benign documents. Equal calibration FPR **does not mean equal realized scoring FPR**. Source groups are disjoint between these subsets; variants from one source remain grouped.

The floating threshold is 0.005549266934394836. Original floating predictions are reused from Goal 2. All rows below have 2,738 true positives and zero false negatives at both the floating and refitted thresholds.

| Reference | Refitted threshold | FP at floating threshold | FP after refit | Scoring FPR after refit | Document BCE | Accuracy at 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 0.00554927 | 18 | 18 | 1.3148% | 0.0051465 | 99.9270% |
| W8A8 PTQ | 0.00341436 | 8 | 18 | 1.3148% | 0.0031466 | 99.9270% |
| W4A8 PTQ | 0.01181794 | 31 | 17 | 1.2418% | 0.0040187 | 99.9270% |
| W8A8 QAT | 0.00455792 | 13 | 13 | 0.9496% | 0.0040976 | 99.9026% |
| W4A8 QAT | 0.08712488 | 34 | 20 | 1.4609% | 0.0187557 | 99.5861% |

QAT used exactly one epoch per uniform anchor, independently from the same selected FP checkpoint: 32,370 training documents, 1,012 effective batches of up to 32 documents, 32-window microbatches, AdamW learning rate 2e-5 and weight decay 0.01, seed 42. Window cross-entropies are averaged within each document and then across the actual document batch. Known injection spans supply window labels. Static W4/W8 and A8 scales remain frozen. Training took 163.42 s for W8 and 164.10 s for W4; these are research training costs, not inference benchmarks. No extra QAT epoch or favorable-epoch selection was added. QAT has one more training epoch than the original FP32 reference; no matched extra FP32 epoch was run. Consequently, the lower W8 QAT FPR relative to FP32 cannot be attributed to quantization alone.

Nearby 0.5% and 2% target operating points were derived from saved scores only. Every uniform row still has recall 1.0. The observed scoring FPRs for target 0.5% / 1% / 2% are: FP32 0.4383 / 1.3148 / 2.4836%; W8 PTQ 0.5113 / 1.3148 / 2.7757%; W4 PTQ 0.9496 / 1.2418 / 2.7757%; W8 QAT 0.2922 / 0.9496 / 3.1410%; W4 QAT 1.0957 / 1.4609 / 2.4836%. These supporting operating points do not replace the preregistered 1% design point.

All refitted QAT false positives were AESLC documents. For W8 QAT, the long-document FPR was 5/187 = 2.6738%, compared with 8/1,182 = 0.6768% for single-window benign documents. W4 QAT produced 8/187 = 4.2781% and 12/1,182 = 1.0152%, respectively. The 20 benign EmailQA scoring documents yielded no false positives; this small stratum cannot establish broad application robustness.

## Single groups, ranking and bounded interactions

Starting from W8 PTQ, each of 16 groups was independently lowered to W4 and restored before the next intervention. All 16 had zero additional misses at the W8 threshold and after refitting. Refitted false positives ranged from 14 to 25 (1.0226–1.8262%). No single-group map met the ≤1% scoring-FPR target.

The selected Quasar-ViT adaptation uses ordinary document accuracy at threshold 0.5 as A's conventional quality objective; it does not prescribe a Hessian ranking. The following are descriptive comparisons of the measured interventions, not invented source-method ranking rules. Spearman correlations with refitted FP count, using average tied ranks, were 0.264 for ordinary error, 0.396 for BCE change and −0.177 for window-logit MSE. Recall ranks are undefined because every recall loss is zero. These 16 dependent interventions are not independent trials; no significance or generalization interval is asserted.

A concrete difference is that lowering `layer0.ffn_output` improves ordinary accuracy to 99.9513% while increasing low-FPR false positives from 18 to 22. Lowering `layer1.ffn_input` attains the same ordinary accuracy with 14 false positives. Lowering `layer2.attention_output` gives the worst refitted count, 25, despite a relatively small logit MSE. Ordinary accuracy, loss and numerical error therefore do not uniquely determine the desired operating-point behavior in this pool. This motivates B's declared task criterion; it does not establish a defect in B or a need for C.

Only two unique informative pairs were evaluated after the deterministic maximum-three selection rule deduplicated its proposals. Recall was tied for all singles, so the top-recall pair came from the declared group-name tie break; it should not be described as a discovered recall-vulnerable pair.

| W4 pair, other groups W8 | FP after refit | Recall loss | BCE interaction beyond summed singles |
|---|---:|---:|---:|
| Layer0 attention output + layer0 FFN input | 18 | 0 | +0.00004582 |
| Layer0 FFN input + layer3 QKV | 13 | 0 | −0.00006751 |

The second pair has a larger BCE increase than its singles but lower refitted FPR, meeting the empirical development protection constraint. At the fixed W8 threshold it produces 34 FP; refitting reduces this to 13. This is a threshold-dependent, full-network result, not evidence that arbitrary group effects add, that the pair generalizes or that an interaction-aware method is novel. There is no quantization-induced attack miss for an ordinary importance ranking to fix; preserving all groups at W8 PTQ itself leaves 18 FP after refitting.

## Storage and execution boundary

The fixed model has 11,171,074 parameters, including about 7.95 million embedding parameters left in FP32. Selected linear weights occupy 3,145,728 bytes at logical W8 or 1,572,864 bytes at logical W4. Adding 32,101,384 bytes of fixed FP32 parameters and 36,960 bytes of active FP32 scales gives **analytical packed payloads** of 35,284,072 bytes (W8) and 33,711,208 bytes (W4), excluding container, alignment and transport overhead.

The actual saved QAT `quantized_state.pt` is 44,829,381 bytes for each precision, because it stores FP32 trainable parameters and quantization buffers, including both frozen weight-scale sets. The FP checkpoint `model.safetensors` is 44,692,608 bytes. No packed runtime checkpoint has been produced. Smaller logical payload is not evidence of lower shared-engine DSP use, latency or energy. The matched 32-check CPU/GPU pilot selected FP32 at eight CPU threads (8.4775 ms total) over actual oneDNN dynamic qint8 at four threads (9.7014 ms); reused CUDA FP32 averaged 5.8364 ms. CPU scoring FPRs were 1.3148% and 1.5340%, so neither was protection-feasible. The dynamic activation policy differs from static W8A8. Complete measurements are in `reports/software_profile.md`; no emulation timing is substituted for accelerated INT4. Hardware M2 evidence still requires Goal 4 target-specific synthesis and complete detector accounting.

## H1 answer, limitations and handoff

H1's predicted attack-recall degradation was not observed in this bounded development experiment. W4 changes score calibration and benign false-positive decisions, and threshold refitting removes much of the fixed-threshold difference. It does not guarantee scoring FPR≤1%; both the floating reference and W4 QAT exceed that target. W8 QAT provides a feasible uniform development anchor under the common allowance. The single checkpoint, one QAT seed, paired attack variants and finite benign tail limit the claim. Final calibration, final test, hard-benign and application generation are separate stages; these development scores do not stand in for them.

The most specific justified next step is the already declared A-to-B criterion integration, while retaining the actual evolutionary proposal and hardware-allocation rules. There is no evidence yet for a new B-to-C allocation mechanism. `reports/goal5_shortcoming.md` records this boundary. Hardware-dependent claims and C remain unresolved.

## Evidence and reproduction

- Implementation: `gate_quant.py`, `scripts/check_quant_arithmetic.py`, `scripts/quantization_sensitivity.py`, `scripts/analyze_quantization.py`.
- Calibration and numerical evidence: `results/goal3/calibration.json`, `results/goal3/run.json`, `results/goal3/real_input_agreement.json`, `reports/quant_arithmetic_check.json`.
- Every candidate folder in `results/goal3/` preserves `candidate.json`, raw document/window scores for both assigned search subsets and `scoring_progress.json`.
- Compact source data: `uniform_summary.csv`, `nearby_operating_points.csv`, `single_group_rank_comparison.csv`, `sensitivity_analysis.json`; original group/pair effects are in `single_group_sensitivity.json` and `pair_sensitivity.json`.
- Reusable one-epoch anchors: `checkpoints/quantized/qat_w8_a8` and `checkpoints/quantized/qat_w4_a8`; load with `gate_quant.load_quantized_checkpoint`.
- Run: `C:\research\GATE\.venv\Scripts\python.exe scripts\quantization_sensitivity.py --stage all --device cuda --batch-size 32`; report-only analysis: the same interpreter with `scripts\analyze_quantization.py`. Paths resolve relative to the script's repository, so the scripts can be invoked by full canonical UNC paths.
- Completed maps/splits and checkpoints are reused. A `results/goal3/pause.request` sentinel stops at an inference split/map boundary or saves QAT optimizer/RNG at a document-batch boundary. Periodic QAT state is also saved every 100 batches. After the authorized pause ends, remove the sentinel and repeat the same command; do not restart completed training.
