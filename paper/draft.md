# Quantization shifts false-positive behavior without observed recall loss in a BERT-Mini injection detector

**Interim research draft, 26 September 2026.** Goal 8 remains partial. This draft reports development quantization experiments, a short software timing pilot and the completed institutional DeepSeek application pilot. Final-test performance, hardware-grounded A/B/C comparisons and FPGA execution remain unmeasured. Earlier Gemini observations are preserved separately as historical preparation evidence.

## Abstract

Low-bit deployment of a prompt-injection detector requires preserving detection at a specified benign false-positive rate, rather than ordinary classification accuracy alone. We examined this distinction using a fixed BERT-Mini detector and grouped email data with constructed instruction insertions. Twenty fixed post-training precision maps and two one-epoch quantization-aware training anchors used common W4/W8 encoder groups and static A8 inputs. All retained the 2,738 observed scoring attacks after independently refitting thresholds on separate benign development groups. Their false-positive behavior differed: uniform W8A8 and W4A8 after quantization-aware training produced 13 and 20 false positives among 1,369 benign scoring documents, respectively. The W8A8 anchor met the empirical 1% target; the W4A8 anchor did not. Ordinary loss and numerical error did not uniquely rank this behavior. These results support integrating the protection criterion into the existing selection method, but establish neither a new allocation mechanism nor hardware acceleration. Final evaluation and a target-specific FPGA cost model remain necessary to test those claims.

## Introduction

An online detector for indirect prompt injection must distinguish malicious instructions in external documents from legitimate task content. Its usefulness depends on both missed attacks and unnecessary rejection of benign documents. This study treats instruction presence as the detector label; whether a target language model follows the instruction is a separate application outcome. The scoped application is English, plain-text, single-turn document question answering, using newly received external content.

Mixed-precision optimization already combines numerical sensitivity with execution cost. [HAO](https://arxiv.org/pdf/2104.12766) jointly optimizes hardware allocation and quantization using a latency-constrained formulation. [Quasar-ViT](https://arxiv.org/html/2407.18175) combines evolutionary proposals with reusable low-bit FPGA execution for vision Transformers. These methods motivate a fixed-model adaptation, but their conventional quality objectives do not directly specify recall at a low benign false-positive rate (FPR). The present question is whether quantization produces detection errors that remain after threshold refitting, before attributing value to a more elaborate allocation method.

We therefore separated three questions. H1 asks which quantization effects persist after the same threshold-calibration rule. H2 asks whether one additional mechanism improves complete checking cost beyond an existing method already adapted to the protection constraint. H3 asks whether that improvement yields a measured benefit on the same FPGA. This interim study answers H1 within a fixed development pool and prepares the controls needed for H2 and H3; it does not yet execute those comparisons.

## Results

### Threshold refitting changed false positives without revealing attack-recall loss

The selected floating checkpoint detected all 2,738 constructed attack documents, but its scoring FPR exceeded the target. Its threshold was fitted on 1,355 distinct benign development bodies, giving 13 exceedances, then applied to separate scoring groups containing 1,369 benign bodies. The resulting 18 false positives corresponded to 1.3148%. Checkpoint selection followed the previously fixed recall-then-loss rule; neither the checkpoint nor threshold was changed after observing this result.

Uniform post-training quantization (PTQ) altered scores without adding attack misses. W4A8 gave 31 false positives at the original floating threshold and 17 after its own refit. W8A8 gave 8 and 18, respectively. Thus fixed-threshold comparisons would describe different effects from the common calibration procedure. Every uniform variant had the same empirical calibration FPR of 13/1,355 = 0.9594%, but the realized scoring FPRs differed (Fig. 1). Matched calibration targets do not guarantee matched FPRs on a separate sample.

Exactly one epoch of quantization-aware training (QAT) was then applied to each uniform anchor. W8A8 QAT met the empirical scoring constraint with full observed recall; W4A8 QAT remained above the FPR limit (Table 1). The original FP32 reference did not receive a matched extra epoch. Consequently, the W8A8 QAT improvement relative to that reference cannot be attributed to quantization alone. The two QAT anchors did receive the same additional training allowance and shared frozen scales.

**Table 1. Uniform development results.** All rows detected 2,738/2,738 attacks at both displayed threshold policies. Refitted thresholds used only the separate threshold-fitting subset. FPR denominators are 1,369 benign scoring documents. QAT indicates one additional training epoch, not an accelerated execution format.

| Model | FP at floating threshold | FP after refit | FPR after refit | Ordinary accuracy at 0.5 |
|---|---:|---:|---:|---:|
| FP32 | 18 | 18 | 1.3148% | 99.9270% |
| W8A8 PTQ | 8 | 18 | 1.3148% | 99.9270% |
| W4A8 PTQ | 31 | 17 | 1.2418% | 99.9270% |
| W8A8 QAT | 13 | 13 | 0.9496% | 99.9026% |
| W4A8 QAT | 34 | 20 | 1.4609% | 99.5861% |

### Ordinary quality metrics did not determine the low-FPR ordering

Lowering one group at a time from W8 to W4 yielded no additional misses at either the common W8 threshold or the independently refitted threshold. Across the 16 interventions, refitted false positives ranged from 14 to 25 (Fig. 2). Lowering layer0 FFN output and lowering layer1 FFN input both improved ordinary accuracy to 99.9513%, yet produced 22 and 14 false positives, respectively. Tied-rank correlations with refitted FP count were 0.264 for ordinary error, 0.396 for document binary cross-entropy change and −0.177 for window-logit mean squared error. These are descriptive comparisons among dependent interventions, not independent statistical trials. A recall-loss ranking was undefined because every value was zero.

The bounded pair experiment also depended on the threshold policy. Two unique pairs remained after deduplicating at most three informative proposals. The pair lowering layer0 FFN input and layer3 QKV produced 34 false positives at the fixed W8 threshold and 13 after refitting, with no missed attacks. Its loss increase did not identify its favorable refitted FPR. This supports full-network evaluation under the chosen protection criterion, but does not show that the safety-matched baseline fails or that a new interaction mechanism is needed.

### Available integer execution did not provide a software speed advantage

A short matched-input pilot tested CPU FP32 and actual PyTorch dynamic qint8 execution at one, four and eight threads. The best tested CPU setting was FP32 at eight threads; dynamic qint8 at four threads was slower (Table 2; Fig. 3). The latter used oneDNN packed kernels with runtime activation quantization, rather than the experiment's static signed W8A8 policy. Its lower mean document loss did not imply feasible protection: it produced 21 false positives, compared with 18 for CPU FP32, while retaining full observed recall. Neither CPU route met the 1% scoring-FPR requirement.

**Table 2. Short software timing pilot.** Values are mean raw-text-to-decision times over 32 checks following eight warm-ups, using the same real single-window inputs, padded length 256 and batch 1. These measurements are not final sustained-load benchmarks or protection-matched winners. CUDA phases were synchronized; the saved matched FP32 GPU run was reused. Sources: `reports/software_profile.md` and `results/goal3/software_profile/profile.json`.

| Execution | Best tested CPU threads | Inference/aggregation (ms) | Total (ms) |
|---|---:|---:|---:|
| CPU FP32 | 8 | 7.8639 | 8.4775 |
| CPU dynamic qint8 | 4 | 9.1727 | 9.7014 |
| RTX 4070 FP32 | — | 5.1001 | 5.8364 |

Logical weight reduction also differs from executable checkpoint size. The fixed FP32 parameters outside the search weigh 32,101,384 bytes. Including selected packed weights and active scales gives analytical payloads of 35,284,072 bytes for W8 and 33,711,208 bytes for W4. Both saved QAT states actually occupy 44,829,381 bytes because they preserve FP32 parameters and quantization buffers. Neither logical byte estimates nor emulated quantization timings establish FPGA acceleration.

### Offline FPGA modeling did not establish an acceleration benefit

For the static-bank 64-lane profile, the modeled internal path required 1.145 s at uniform W4 and 1.216 s at uniform W8 per 256-token window, batch 1. These estimates combined frozen integer formulas, the weighted 411-call FP32 service subtotal of 474.3 ms, and 27.0 ms of conditional control and data movement. They exclude additional physical memory stalls and host overhead. The FP32 service alone met 200 MHz in a representative routed implementation; the integrated detector was not implemented or measured on the board. Component validation and the conditional model therefore do not establish a software-speed advantage or a B-to-C improvement (`reports/hardware_cost_model.md`).

Numerical bring-up also limited transfer of the earlier software protection results to hardware. On one saved input, same-input integer checks passed, but a small FP32 attention difference crossed an A8 rounding boundary and propagated through later layers. An ordered CPU reference reproduced the C-simulation's final logits and risk on that input. Agreement through the actual synthesized arithmetic and across representative inputs remains unverified (`reports/goal4_numerical_bridge.md`).

### Whole-chunk rejection prevented observed attacks while removing task evidence

The development application pilot completed 120 requests to institutional `deepseek-ai/DeepSeek-V4-Flash-0731`, representing 100 attack pairs from 20 source tasks. Under fixed rubric review, unguarded attacks succeeded in 11/100 cases. The floating guard rejected every attacked chunk, reducing observed attack success to 0/100 while also reducing reference-based task completion from 24/100 to 0/100. No clean source was rejected; clean-task completion was unchanged at 9/20 distinct tasks. These outcomes describe the trade-off of removing the sole evidence chunk, rather than preserving useful content from an attacked document.

The output allowance materially bounded this result. Seventy-seven of 120 responses reached the fixed 256-token limit, including 64 with no visible answer. All truncations counted as task non-completion. Two partial injected safety sentences did not meet the full attack target and remain documented as partial compliance. At least three source tasks contained upstream attribution or payer/charger ambiguities; their original references were retained. Thus the reported rates apply to this constrained configuration and fixed-reference scoring. They do not estimate behavior with a larger output allowance or establish robustness to unseen attacks. Raw responses and response-specific Codex judgments are retained in `results/goal2/application_deepseek/`; `reports/deepseek_pilot_findings.md` gives the four-condition counts and limitations.

## Discussion

The observed constraint was benign rejection rather than lost attack recall. Threshold movement accounted for substantial fixed-threshold differences, while score refitting still left different false-positive counts on held-out development groups. This combination favors direct integration of the task criterion into selection. It does not supply evidence that an additional allocation mechanism improves upon a baseline already using that criterion.

This distinction matters because the surrounding ideas have prior art. [Q-BERT](https://arxiv.org/pdf/1909.05840) assigns precision using Hessian sensitivity; [InfoQ](https://arxiv.org/html/2508.04753v1) measures downstream information changes; [Critical Weight Protection](https://aclanthology.org/2026.findings-acl.993.pdf) preserves high-precision weights associated with safety and fairness; and the [MixQuant preprint](https://arxiv.org/html/2607.23047v1) measures distortion in quantized upstream contexts. [FILM-QNN](https://www.sfu.ca/~zhenman/files/C23-FPGA2022-FILM-QNN.pdf) connects mixed precision with packing and layout. Our observed pair interaction neither contradicts these mechanisms nor independently establishes novelty. The selected Quasar-ViT adaptation remains the common A/B baseline, and C remains unspecified.

The absence of recall loss is limited by the data construction. Search source groups were disjoint from weight training, but their insertions came from the same prepared development payload pool. Each source supplied related attacked variants, and only 20 benign EmailQA sources occurred in scoring. These conditions do not establish robustness to unseen attack mechanisms or a precise EmailQA-specific 1% FPR. Long benign documents also remained harder: W8A8 QAT rejected 5/187 long documents versus 8/1,182 single-window documents. One training seed, fixed-scale calibration from 256 windows and one QAT epoch further limit claims about other quantization procedures.

The application results expose a separate deployment boundary: removing an entire attacked document can prevent the injected response while eliminating the evidence needed for the legitimate task. Heavy truncation and ambiguous benchmark references further constrain the interpretation of task completion. Before final application evaluation, the output allowance and task references need a frozen, evidence-supported policy; the present development run remains unchanged. Final-calibration, test and NotInject groups remain closed pending frozen methods and finalists. The FPGA model identifies substantial serial arithmetic cost, while the numerical trace shows why quantizer boundaries require an aligned reference. A matched numerical implementation and complete host-to-decision measurements are needed before selecting or attributing a hardware improvement over B.

## Methods

### Data, detector and thresholding

We constructed instruction-presence labels from [BIPIA's pinned official builders and insertion rules](https://github.com/microsoft/BIPIA/tree/a004b69ec0dd446e0afd461d98cb5e96e120a5d0) and [AESLC email bodies](https://github.com/ryanzhumich/AESLC/tree/7afc087a1cc234d07121f0d4a2d87102ddceabc2). Each source group contained one clean body and two exact start/end insertions sampled reproducibly from harmless task-diversion or answer-transformation payloads. Identical normalized bodies and their variants stayed together across splits. The training set contained 10,790 source groups and 32,370 documents; threshold fitting and candidate scoring contained 1,355 and 1,369 groups. Separate final-calibration and test pools were reserved, with 1,810 distinct benign test bodies, below the 3,000 aim. Detailed counts, source revisions and licensing are recorded in `reports/data.md`.

The detector was `google/bert_uncased_L-4_H-256_A-4`, pinned at revision `387825ce42dbb39b87911cdf8e383ee3b25184f8`, with four layers, hidden size 256 and four attention heads. Token windows had total length 256, including special tokens, and overlapped by 64 content tokens. Document risk was the maximum window injection probability. Exact inserted spans determined window training labels. Window cross-entropy was averaged within documents, then across up to 32 documents per effective batch. AdamW used learning rate 2e-5, weight decay 0.01 and seed 42 for three floating epochs; search recall, then document loss, selected epoch 3.

For n benign threshold-fitting scores and target α, scores were sorted ascending and the threshold was the value at one-based index n−floor(αn). A document was flagged only when its score was strictly greater. The primary α was 0.01; α=0.005 and 0.02 were supporting operating points derived from saved scores. B's required scoring recall is at least the floating reference minus one percentage point, with scoring FPR≤0.01. The calibration and scoring sets remained separate throughout.

### Quantized reference and sensitivity design

The search covered 24 encoder linears in 16 groups: Q/K/V together, attention output, FFN input and FFN output in each layer. Weight codes used signed W4 [-8,7] or W8 [-128,127], with static per-output-channel scales derived separately from the selected floating checkpoint. Inputs used signed A8 [-128,127], with static per-module scales fitted once on 256 training windows, excluding masked padding. All used zero point zero, nearest-even rounding and saturation. Embeddings, attention matrix products, nonlinear functions, residuals, biases, pooler and classifier remained FP32.

Brevitas 0.13.4 supplied integer-code STE quantization. Selected dot products were emulated with integer-valued FP32 under a checked K≤1,024 bound, with TF32 and AMP disabled. An independent int64 CPU reference, signed endpoint examples, exhaustive W8 nibble decomposition over all weight codes, and a real short document established the prescribed software arithmetic agreement. These checks did not execute HLS or validate an FPGA implementation.

Two uniform PTQ maps, 16 single-group interventions and two informative pairs were scored with common calibration. Both uniform maps then received exactly one QAT epoch from the same floating checkpoint; weight and activation scales stayed frozen. No extra search or epoch was added to obtain a preferred outcome. Raw window logits, document scores, fixed/refitted decisions and paired errors are retained in `results/goal3/`. Reported rates describe this fixed development pool; no seed-level interval or independence across attack variants is assumed.

### Development application protocol

The four conditions were clean/no guard, clean/guard, attacked/no guard and attacked/guard. Twenty source tasks each supplied five fixed development payloads. Identical target requests were reused only for quality, yielding 20 clean and 100 attacked calls; cached timing was not treated as execution. The original BIPIA template placed the email in the system message and the question in a separate user message. The requested settings were temperature 0, low reasoning effort and 256 completion tokens. The hosted model returned the requested dated label on every call, without an immutable weights revision.

Primary outcomes used the previously fixed task and attack rubrics, with response-ID-bound Codex review of every response. Visible text determined attack success; reasoning was excluded. Non-ok responses were task non-completions. Fully received empty, length-ended answers produced no visible attack target. All attacked cases, including guard rejections, remained in the denominator. Original source references were preserved despite documented ambiguities. Model-selection comparison was not performed; the earlier partial Gemini run remains separate in `results/goal2/application/`. API usage, timing boundaries and complete reproduction details are in `reports/deepseek_pilot_findings.md`.

### Planned method comparison and hardware boundary

The baseline identity, `quasar_vit_fixed_bert_evolution`, was fixed before comparative outcomes. It preserves elite-parent crossover, group mutation and resource-constrained shared-engine allocation from Quasar-ViT, while replacing architecture and rowwise-ratio search with the common fixed BERT groups. A optimizes ordinary accuracy at threshold 0.5 under preregistered latency caps. B changes candidate and elite eligibility to the same low-FPR/recall constraint as any later C, then selects minimum complete latency among feasible designs. The bounded population and evaluation defaults are project choices, not claimed source-paper hyperparameters (`reports/baseline_selection.md`).

The offline design used one shared W4/W8 × A8 integer engine and one FP32 service, targeting the VEK280 at 200 MHz with Vitis/Vivado 2025.2. Twelve HLS syntheses established three active integer profiles and the common FP32 implementation. Targeted and reserved RTL cases for the active integer profiles each passed twice against frozen predictions. Fifteen representative FP32 call shapes also passed twice. A separate controller/mover simulation checked the complete command schedule using explicit arithmetic completion stubs; its measured stub intervals were subtracted before adding the saved arithmetic costs. The resulting internal-path estimate assumes serialized execution and registered memory responses without additional DDR stalls. Full checking costs, supported latency caps and physical feasibility remain unresolved. Build identities, source snapshots, accounting and limitations are in `reports/hardware_cost_model.md`. H2 and H3 remain untested.

## Conclusion

This development study observed quantization-dependent score and false-positive changes without attack-recall loss under the tested threshold policies. It supplies a feasible uniform W8A8 QAT development anchor and evidence for the already planned criterion adaptation. Whether a further mechanism reduces complete checking cost at the same protection requirement remains open.

## Figures and evidence record

- **Figure 1:** [Uniform threshold effects](../results/goal3/figures/uniform_threshold_effects.pdf). Paired scoring FPRs at the original floating threshold and each model's refitted threshold. All variants use the same benign calibration target; observed scoring FPRs are displayed rather than assumed equal. PNG/SVG/TIFF and source CSV accompany the PDF.
- **Figure 2:** [Single-group sensitivity](../results/goal3/figures/group_sensitivity.pdf). All 16 one-group-at-a-time W8-to-W4 interventions in architectural order; common-W8 and refitted threshold policies. Attack misses remain zero. Source counts and saved paired predictions support the display.
- **Figure 3:** [Software cost breakdown](../results/goal3/figures/software_breakdown.pdf). Measured component means for the best tested CPU FP32 and dynamic qint8 settings and the reused matched CUDA FP32 pilot. The 32 measured checks are descriptive; no uncertainty across independent benchmark runs is implied.
- Complete numerical detail, nearby operating points, training records and storage definitions: `reports/quantization_sensitivity.md`, `reports/software_profile.md`, and `results/goal3/`.
- Verified literature scope and unavailable source details: `reports/prior_art.md`. No publication novelty claim follows from this bounded review.

## Draft status and missing evidence

This is a source-grounded interim draft prepared with Codex assistance, pending author review and final evidence. Authorship, target venue and any submission-specific disclosure are not assigned here. No submission package is implied. Hardware-grounded A/B/C selection, full-system implementation, board accuracy/latency/energy and final grouped evaluation remain unexecuted. Component synthesis and bounded RTL validation are complete within the recorded initial allowance. The DeepSeek development application pilot is complete under its stated limits; the separate historical Gemini run remains partial. The missing hardware and final results are not reported as null performance results. The final paper's main B-to-C claim cannot yet be written.
