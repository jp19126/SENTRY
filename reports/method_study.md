# SENTRY span-coupled pilot: bounded negative result and unresolved comparisons

**13 October 2026. The approved pilot closes with six completed fits and a
negative candidate result: quality, coupling and FPGA resource gates fail.
No algorithm qualifies for timing or conditional integration.** SoPa quality,
RTL behavior and end-to-end comparative cost remain unresolved. The preserved
approval and contract below do not establish novelty. The user instructed
execution of
`SENTRY_codex_execution_prompt.md`, whose explicit checkpoint is: “Pause here for
the user's selection and budget approval.” The later user update, “no usage limit
for this task,” removed the earlier 10%-remaining usage stop. The latest user
instruction is: **after the next global usage reset, pause at 50% weekly usage**.
Historically, at 2026-10-13 01:37:33 UTC the shared weekly meter was 5% used,
window 10,080 minutes, reset timestamp 1792323848; no reset had been observed.
Until that reset the no-limit instruction applied. The approved rule was to
detect a reset from an observed usage decrease or changed reset boundary,
then use the shared weekly used-percent meter; do not
consume a reset credit or create a periodic automation. This policy does not open
protected data or authorize target-model calls.

### Approval and execution history

The user's explicit decision was: “continue. I will go to sleep in the next hours
so keep going. I approve this”. This approves `span_coupled_pilot_v1`, its pilot,
and conditional integration under the unchanged gates below. Historical
“proposed/approval requested” wording later in the Phase A record is superseded
by this decision, not a requirement to ask again. New study settings are in
`configs/span_coupled_pilot_v1.json`; historical `configs/project.json` is unchanged.
Final data, paid services, target-model calls, physical hardware and GitHub
synchronization remain excluded. The historical baseline usage reading on
resumption was 6% used with reset timestamp 1792323848, before the observed reset.

## Final measured findings — six completed fits

**The approved pilot has not established a coupling benefit or a viable FPGA
implementation. Phase C integration fails its gates.** The table reports final
integer decisions at each model's separately calibrated 1% operating point;
counts are TP/FN/FP/TN, and BER is stress balanced error.

| Completed arm / seed | Natural counts | Stress counts | Stress BER |
|---|---:|---:|---:|
| Coupled / 42 | 2738 / 0 / 17 / 1352 | 258 / 254 / 258 / 254 | 50.0000% |
| Coupled / 1337 | 2738 / 0 / 6 / 1363 | 254 / 258 / 256 / 256 | 50.1953% |
| Independent pooling / 42 | 2738 / 0 / 20 / 1349 | 386 / 126 / 281 / 231 | 39.7461% |
| Independent pooling / 1337 | 2738 / 0 / 15 / 1354 | 249 / 263 / 263 / 249 | 51.3672% |
| Span MLP / 42 | 2738 / 0 / 21 / 1348 | 336 / 176 / 259 / 253 | 42.4805% |
| Span MLP / 1337 | 2738 / 0 / 9 / 1360 | 136 / 376 / 162 / 350 | 52.5391% |

All six fail the stress quality gate. Coupling's required benefit over
independent pooling fails in both seeds: integer BER gains are -10.2539 and
+1.1719 points, below the required +5 points each. Both FP32 and integer pooled
paired intervals include zero. This is a negative result for the frozen recipe
and authored probes, not a general rejection of span coupling. Both span-MLP
seeds are complete. SoPa seed 42 was stopped after 318 guarded seconds because
its observed progress projected beyond the fixed recipe budget; no quality
checkpoint was produced. Its second seed and hardware design were not run.
The conventional-arm comparison therefore remains incomplete.

Supporting C checks pass, but hardware evidence also fails to qualify: fused
synthesis uses **104,176 LUT > 100,000**, span MLP uses **292 DSP > 256**, and
the buffered candidate failed SYNCHK before a complete cost report. All RTL
behavior and transaction-cycle comparisons remain unverified. HLS estimates
and exact C agreement are not achieved timing or physical-device measurements.
No timing stage or Phase C integration ran. A new method, revised implementation
campaign or additional comparison requires a newly approved study; this result
does not authorize an automatic retry or a shortened training recipe.

The **post-reset 50%-used weekly cap is active**, superseding the earlier
pre-reset readings above. The reset was observed at 05:06 UTC on 13 October;
`results/span_coupled_pilot_v1/usage_reset_observed_v1.json` records the change
to reset boundary `1792472301`. The latest coordinator-reported official
reading is **7% used** at that boundary; pause at 50%, without extending any
experimental allowance. Protected data, target calls and board actions remain
excluded. Detailed results and evidence paths follow the preserved contract
and historical Phase A record below.

## Historical Phase A decision in brief

Recommend one bounded **span-coupled evidence guard** pilot. Its question is
whether binding suspicious evidence and benign contextual evidence to the same
explicit text span, before reducing to a document decision, improves the
security–hardware trade-off under a fixed FPGA state budget. The candidate is a
research hypothesis, not a cleared novelty claim. Its strongest competing
explanations are ordinary neural span classification, SoPa/rational recurrences,
better augmentation, and conventional classifier combinations.

The proposed contribution is the coupled decision representation and its bounded
span-bank realization. The saved-score audit and hardware arithmetic below are
supporting work. If ordinary span/WFSA computation matches the candidate, or if a
competent conventional hardware mapping removes its hardware advantage, the
study must report that negative result. A compact detector on FPGA is not enough.

Only one candidate is proposed. A generic cascade, linear-attention replacement,
BERT-to-AIE port, or new precision allocation campaign is not recommended as a
second supposed contribution.

## What was actually executed

The checkout began clean on `main`, HEAD
`d50abdca98b889a0de0ac09488da5154e7102389`. Read the governing documents, current
protocol/configuration, R1 report, application report and relevant saved/source
artifacts. Two native subagents were used: `gpt-6-astra` for method/security/prior
art, and `gpt-6.1-sol` for FPGA architecture, with a subsequent focused independent
review of the saved-score analysis. No nested team or heavy parallel job ran.

Implemented and ran `scripts/method_study_saved_audit.py`. It opens only the eight
existing R1 document-score files for four arms and the two development splits;
it validates document IDs, finite scores, full window arrays/maxima, matched
metadata and calibration/scoring source separation. It recomputes each threshold
and confusion counts, pairs decisions, counts conventional early-positive work,
and executes 2,000 scoring-source-group bootstrap resamples at a fixed threshold.
It never loads a model or opens a final partition. Input hashes and sizes are in
`results/method_study_phase_a/saved_score_audit_v2.json`.

The initial Windows Python invocation failed at a nonexistent UNC output-path
stat with `WinError 65`; no model or result was produced. The same stdlib analysis
completed with Ubuntu WSL Python. A second output explicitly clarifies the
fixed-threshold uncertainty boundary; the original is retained. An independent
PowerShell reread confirmed the BERT threshold, 447 skipped windows, group sizes,
and CNN/BERT joint confusion cells. This is actual saved-data analysis, not a
fresh detector or downstream experiment. Lightweight operation/storage formulas
were also evaluated in Python; no vendor simulation or build ran.

Reproduce from the repository using a fresh output filename:

```sh
python3 -B scripts/method_study_saved_audit.py \
  --output results/method_study_phase_a/reproduction.json
```

The script refuses to overwrite evidence. The local raw results are ignored by
Git; this report and the script alone do not reproduce missing score inputs in a
fresh clone. Worker inspection notes are in the same local result directory.

### Recomputed development findings

Thresholds use only 1,355 benign calibration documents, with strict `score > t`.
Scoring has 1,369 source groups, each with one benign and two injected documents.

| Existing arm | Threshold | TP / FN / FP / TN | Recall | FPR |
|---|---:|---|---:|---:|
| Native W8 BERT CUDA Graph | 0.004557917360216379 | 2738 / 0 / 13 / 1356 | 100% | 0.9496% |
| BF16-operand BERT reference | 0.005079554859548807 | 2738 / 0 / 13 / 1356 | 100% | 0.9496% |
| CNN | 0.14351105690002441 | 2738 / 0 / 24 / 1345 | 100% | 1.7531% |
| Hashed feature classifier | 0.6349835395812988 | 2639 / 99 / 20 / 1349 | 96.3842% | 1.4609% |

BERT's 95% source-group bootstrap FPR interval is **0.5113–1.4609%**, conditional
on the fitted threshold. It does not include calibration-threshold uncertainty.
Zero misses yield a degenerate empirical recall bootstrap, not a population
guarantee. All 55 development attack templates appeared in training; the 100%
recall is not unseen-template evidence. Existing compact-model failures concern
two recipes, not a proof that transformers are necessary.

Conventional sequential early rejection after the first above-threshold window
would evaluate 4,861 rather than 5,308 BERT windows: **447 fewer (8.421%)**. Only
9/1,754 benign windows would be avoided. This preserves the binary decision,
not the complete numeric maximum; no evaluator was changed, no ordering was
optimized, and no latency saving was measured. Attack prevalence and long-input
composition control its deployment value.

CNN and BERT share only four benign false positives; nine are BERT-only and
twenty CNN-only. Their conjunction would have 2,738 TP, zero FN and four FP on
these already inspected scores. That is an exploratory observation about a
standard cascade, not a prospectively validated result or a new method. On this
attack-heavy set, CNN-first/BERT-confirm would still send 2,762/4,107 documents
to BERT, plus pay for CNN on every document. Its separate timing and robustness
remain unresolved. Include it as a conventional explanation, not a success claim.

The saved BF16 result changes 5,160/5,308 scoring-window risks while retaining
the same scoring decisions. Its own threshold differs and two calibration
benign decisions exchange membership. This is a numerical sensitivity result,
not AIE numerical equivalence or a new detector. R1's 1.707225-ms warm CUDA-Graph
mean belongs to native W8/FP32-attention, 32 single-window inputs; it is neither
BF16 timing nor a universal hardware feasibility threshold.

The old application pilot blocked all 100 attacked cases, yielding zero attack
success but also zero attacked-task completion. Clean-task completion was
45/100 logical cases over only 20 backgrounds. Those saved outcomes cannot
establish utility for a new guard, sanitization, or changed target inputs.

## Verified nearest work and how it constrains the proposal

Search date: **13 October 2026**. Academic-search MCP tools were absent; public
primary publisher, author and arXiv pages were inspected via web fallback. This
is a targeted review, not an exhaustive novelty clearance. No cited code was
executed or benchmark reproduced. Method-level descriptions below come from
accessible full text; inaccessible or metadata-only evidence is labelled.
Journal hardware work is included alongside directly relevant conference work.

| Primary work and publication identity | Inspected evidence and relevant mechanism | Consequence for SENTRY |
|---|---|---|
| Li, Liu, Zhang & Xiao, **PIGuard: Prompt Injection Guardrail via Mitigating Overdefense for Free**, ACL July 2025, 30420–30437, [DOI 10.18653/v1/2025.acl-long.1468](https://aclanthology.org/2025.acl-long.1468/) | Published PDF §§3–5, App. B. Token-bias diagnosis, generated benign examples and retraining; DeBERTa-v3-base. Published successor of InjecGuard arXiv 2410.22770 (30 Oct 2024), not a second independent baseline. | Overdefense-oriented augmentation is established. BIPIA_train occurs in its source table; do not claim independent template exposure. NotInject examples remain closed. |
| Jacob et al., **PromptShield: Deployable Detection for Prompt Injection Attacks**, CODASPY 2025; [extended author version, 12 Apr 2025](https://arxiv.org/html/2501.15145v2) | §§3.2, 4, 5.1–5.5. Adapted detectors and application-oriented low-FPR operating points. Author record confirms conference status; publisher DOI access was unsuccessful. | Compare at calibrated low FPR, not overall accuracy/AUC. An earlier PromptGuard comparator is not automatically Prompt Guard 2. |
| Meta, **Llama Prompt Guard 2**, 2025 model release; [official 22M model card](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M) | Official documentation/card; 22M and 86M variants. Model release, not peer-reviewed paper. Training exposure incomplete. | A compact pretrained guard is a quality reference if accessible; vendor latency is not SENTRY latency. Do not access gated weights without entitlement. |
| Liu et al., **DataSentinel: A Game-Theoretic Detection of Prompt Injection Attacks**, IEEE S&P 2025, [DOI 10.1109/SP61157.2025.00250](https://doi.org/10.1109/SP61157.2025.00250); [author v4, 12 Nov 2025](https://arxiv.org/html/2504.11358v4) | §§III–IV/algorithms. Detection LLM with random known-answer instruction and alternating attacker/detector optimization. | Adaptive evasion and same-task limitations matter. Its LLM work cannot be removed from a cost comparison. No reproduction or target calls proposed. |
| Guo, Zhao & Wang, **BERM: Low-Overhead Prompt-Injection Detection via In-Situ Benign Representation Modeling**, IJCAI 2026, 599–607, [DOI 10.24963/ijcai.2026/68](https://www.ijcai.org/proceedings/2026/68) | Published PDF §§3.1–3.5. Reuses host final-layer prefill representations, pooling and a small classifier. | Separate guard computation may be avoidable when host internals are available. SENTRY's opaque hosted-target/pre-API boundary does not provide those representations; do not generalize that restriction to local hosts. |
| Jia et al., **PromptLocate: Localizing Prompt Injection Attacks**, [author v2, 17 Oct 2025](https://arxiv.org/html/2510.12252v2) | §IV/ablations; segmentation, adapted detector/group search and contextual inconsistency. Author record says accepted/to appear S&P 2026; publisher status not independently verified. | Localization and checking context are established; split instructions must be a falsifier. |
| Wang et al., **WebSentinel: Detecting and Localizing Prompt Injection Attacks for Web Agents**, [3 Feb 2026 preprint](https://arxiv.org/html/2602.03792v1) | §§4.2–4.3. Segments of interest plus analyzer LLM, pruned webpage context and alignment. No venue verified. | Region-level contextual verification is not itself novel; do not silently broaden this pilot to web agents. |
| Schwartz, Thomson & Smith, **Bridging CNNs, RNNs, and Weighted Finite-State Machines (SoPa)**, ACL 2018, 295–305, [DOI 10.18653/v1/P18-1028](https://aclanthology.org/P18-1028/) | Method PDF. Learned soft WFSA patterns and pooled matches for text classification. | The most direct conventional algorithm baseline; fixed state and interpretable matches are established. |
| Peng et al., **Rational Recurrences**, EMNLP 2018, 1203–1214, [DOI 10.18653/v1/D18-1152](https://aclanthology.org/D18-1152/) | PDF §§4–5.2. Recurrent calculations expressed as weighted-automata forward algorithms. | Recurrence or max-plus arithmetic alone is not a contribution. |
| Katharopoulos et al., **Transformers are RNNs**, ICML 2020, PMLR 119:5156–5165, [proceedings/full text](https://proceedings.mlr.press/v119/katharopoulos20a.html) | Associative recurrent implementation of linear attention. | A fixed-size prefix accumulator alone is already known. |
| **EQ-ViT**, IEEE TCAD 43(11):3949–3960, Nov 2024, [DOI 10.1109/TCAD.2024.3443692](https://doi.org/10.1109/TCAD.2024.3443692), [author full text](https://peipeizhou-eecs.github.io/publication/2024_esweek_eqvit/2024_esweek_eqvit.pdf) | Full method inspected: nonlinear quantization and Versal spatial pipeline. | Transformer nonlinear co-design/Versal pipelining are established, not a SENTRY contribution. |
| **GAMA**, FPL 2025 accepted author record; [v3 record](https://arxiv.org/abs/2504.09688), [v2 method](https://arxiv.org/html/2504.09688v2) | VE2802 AIE-ML GEMM with buffer, cascade and placement design; tensors assumed already in PL. | Competent AIE mapping matters, but GEMM timing excludes complete guard input/nonlinear costs. |
| **Quasar-ViT**, ICS 2024, [DOI 10.1145/3650200.3656622](https://doi.org/10.1145/3650200.3656622), [full method](https://arxiv.org/html/2407.18175) | Hardware-constrained mixed-precision/architecture search and shared low-bit engines; retained prior project review. | Quantization/search plus a guard loss is not sufficient novelty; the historical restricted A/B adaptation remains unchanged. |

Additional contemporary lead: Li & Gao, **BitFair**, arXiv v2 10 Aug 2026,
[author record](https://arxiv.org/abs/2607.05445), reports acceptance to IEEE
JETCAS with related DOI 10.1109/JETCAS.2026.3722396. Only abstract/metadata were
inspected here; it supports the limited observation that learned bit-level early
termination is an existing co-design topic, not detailed comparisons. The
Southampton repository page for *Realization of Early-Exit Dynamic Neural
Networks on Reconfigurable Hardware* returned HTTP 403 on direct retrieval;
search metadata alone is not used for method claims.

## One implementable candidate, with explicit novelty risks

**Hypothesis:** local coupling of attack evidence and its explanation prevents an
unrelated benign region from cancelling a suspicious region, while a fixed
all-start span bank avoids materializing all span representations. At matched
data, encoder, calibration and resource allowance, this may preserve low-FPR
quality more economically than independent pooling or generic span/WFSA guards.

**Scope:** English external-document-only EmailQA injection classification.
Output is a risk and block/allow decision. No question/answer/label enters the
guard; no sanitization, provenance authentication, or safe execution of accepted
instructions is claimed. An adversary may place operative instructions beside
quotations, disclaimers or unrelated benign text. Local context is evidence,
not a trusted signal, and must not automatically suppress an attack.

### Defined computation for the proposed pilot

Keep the existing tokenizer, total window length 256, 64-content-token overlap,
complete tail, maximum-window document risk and strict threshold. Preserve all
window computation for official scoring. All compact arms use the same single
parallel width-3/5/7 convolutional encoder: learned 64-dimensional embeddings,
64 output channels per width, ReLU, concatenated per-token width 192. This reuses
the R1 architecture as a component, not its frozen checkpoint or classification
head. Do not stack convolutions. A width-7 branch has radius three.

Fix **H=58 content tokens and R=32** before outcomes. Every content token j is
an anchor with prescribed context [j-28,j+29]. Including the encoder, its full
receptive footprint is [j-31,j+32], 64 tokens. Emit an anchor score **only after
that complete context is available**. At an internal window boundary, mask an
anchor unless its complete footprint is contained in that window; only actual
document edges receive zero padding. The existing 64-token overlap ensures
every global anchor can be evaluated in at least one complete-tail window.
Duplicate evaluations must have identical valid-context inputs and own-reference
scores. This is not coverage of relationships beyond the 64-token footprint.
Exclude special/padding tokens as anchors; reset banks at each window. Maintain
every eligible anchor, not a learned shortlist or stride-selected subset.

For each token, project the shared feature `z_t` to clipped vectors `u_t,v_t`.
For every context start i=j-28, update two vectors over its fixed 58 positions,
initialized to zero at that start. Mask virtual-padding projections to zero
(including bias), store the anchor feature z_j, and allow no intermediate score:

```text
a[i] = clip(lambda_a * a[i] + u[t], -8, 8)
b[i] = clip(lambda_b * b[i] + v[t], -8, 8)
if t == j + 29 and complete_footprint_available_or_true_document_edge:
    g[j] = bias + dot(w_anchor,z[j]) + dot(w_a,a[i]) + dot(w_b,b[i])
                + sum_r c[r] * a[i,r] * b[i,r]
    window_logit = max(window_logit, g[j])
window_risk = sigmoid(window_logit)
document_risk = max(window_risks)
```

Products are elementwise; lambda vectors are learned static per-channel values
constrained to [0,1]. All states in a pair retain exactly the same start/end
identity. There is no independently maximized context-free summary head.
Embeddings, encoder, projections, lambda, interaction and anchor weights are learned.
Tokenizer, H/R, window rules, masks, saturation and inference reductions are
fixed. The two latent streams are intended to learn suspicious/contextual
relations, but document labels alone do not prove those semantic roles. Report
anchor locations and perturbation effects; do not call their values explanations.
Use the inherited span-window training labels and document-averaged
cross-entropy; ties in max follow deterministic first-index subgradients.

The bilinear term prevents reducing the whole proposed score to additive
transition reweighting in general. Nevertheless, bilinear heads, span encoders,
recurrences and rings are all established. A neural span classifier could
express the same function. This is the strongest competing explanation, not a
technicality to hide. The study must test whether this particular constrained
decision representation is useful; it must not claim a new class of automata.
The reviewed draft's all-length span maximum was rejected before experiments:
it retained short-span alarms before benign context arrived. Fixed completed
contexts remove that particular monotonicity flaw, not general adversarial or
distant-context risk. Identical full footprints necessarily yield identical
anchor scores. Matching-substring probes within and beyond the footprint test
this remaining limitation; the anchor readout can still learn trigger shortcuts.
For an empty document, define risk as the exported sigmoid of the head bias and
report an empty-input flag; do not reduce an empty anchor set or emit NaN. This
interface case must have matching software/hardware behavior. A nonempty
document's window with no eligible anchors contributes risk zero to the document
maximum; the separate empty-document rule must not fire on such a window.

### Prospective arithmetic and software reference

Train FP32 for eight epochs followed by two fixed QAT epochs per compact fit.
No epoch/seed selection from scoring. Export W8 embeddings/convolution/projection
weights; use fixed training-calibrated per-output scales, RNE and explicit clips.
Lambda uses Q0.8, with 1 represented explicitly as 256. Store recurrence states
as INT32 Q12, clamped to [-32768,32768]; retain INT64 intermediates and RNE after
each rescale. The bilinear product is rescaled back to Q12 before its learned
coefficient. Derive and check static accumulator bounds from actual exported
weights; overflow is a failure, never wraparound. Define an exact sigmoid lookup
table over clipped logits [-16,16] with 4,097 entries and nearest-index/RNE
selection. These are proposed numerical choices, not historical BERT identities.

The integer Python/C reference, exported scales/tables and HLS must match
bit-for-bit at each state/head stage, window maxima and strict decisions. The
learned FP32 and exported fixed-point models may differ; retain both and report
the quality effect. Before any scoring, freeze their format description and
tests. Compared at their separately fitted thresholds, fixed-point recall may
drop at most **0.5 percentage points** and FPR increase at most **0.25 points**
relative to its FP32 counterpart, while still satisfying the absolute limits
below. These proposed bounds cannot be relaxed after discrepancies.

### Hardware partition, costs and value conditions

Prefer **PL** for the first complete region. Stream token IDs through embedding,
the parallel local convolution, projections, span-bank update/interaction,
anchor readout, maxima and threshold. Reuse each projected token across bank lanes;
fuse interaction/reduction without storing all time-by-span vectors. Include
bank ports, delayed convolution edges, fixed-point conversions, lookup, masks,
FIFOs and AXI backpressure. AIE-ML is not selected: its GEMM strengths do not by
themselves justify an extra PL/AIE transfer for these small recurrent states.

Analytical dense upper workload with L=256 sites (actual valid content is at
most 254), H=58, R=32:

| Quantity | Formula / analytical value |
|---|---|
| Context-state updates | LH = 14,848, including prescribed document-edge padding |
| Local encoder | L*64*64*(3+5+7) = 15,728,640 MACs |
| Two R-dimensional projections | L*192*2R = 3,145,728 MACs |
| Recurrence products | 2LHR = 950,272 |
| Completed context + anchor head | L*(4R+192) = 81,920 products |
| Total counted products | 19,906,560; excludes nonlinear/rounding/control/port costs |
| Paired ring + conservative anchor-feature storage | H*2R*4 + H*192 = 25,984 bytes with A8 anchor features, plus metadata/FIFOs |
| W8 embedding | 30,522*64 = 1,953,408 bytes; FP32 training version 7,813,632 bytes |
| Conv + projection weights | 61,440 + 12,288 W8 bytes, before biases/scales/head |
| Ideal 128 / 256 product lanes at 200 MHz | 0.7776 / 0.3888 ms, optimistic arithmetic-only bounds |

Products have different widths/dependencies; a “lane” is not automatically one
DSP, and these arithmetic bounds are not a schedule, synthesis or speedup.
The completed-context bank differs from both preliminary stride-16 and partial-
span-scoring examples; neither earlier calculation is used for this contract.
The 57 virtual context positions are already counted in LH; account separately
for the convolution's edge padding/flush clocks. Virtual z/u/v are zero after
masking projection bias; real z uses convolution padding only at true document
edges. No window-relative positional signal is introduced. Exact
storage, banking and achievable initiation interval must come from the build.
Set a provisional common envelope of 100,000 LUT, 200,000 FF, 256 DSP, 64 BRAM36
and 96 URAM blocks; validate these against the installed part before dispatch.
If the complete design cannot fit, stop within the correction allowance rather
than silently widening the hardware budget or reducing H/R.

The primary hardware workload is **warm, batch-one, complete raw-document to
strict decision**, including all windows. Offline evidence supplies complete
token-ID-to-decision region cycles, storage/resources,
transferred bytes and observed simulated stalls. CPU tokenization/packing and
software timing are measured separately; unmeasured physical transport remains
an explicit term. Pilot HLS timing is a synthesis estimate; evaluate its feasible
clock with a further 20% frequency derating. Only actual implementation can supply
achieved/routed timing, which must replace this assumption in Phase C. A
dependency-aware estimate is

`preparation + host packing/transport + scheduled region/required external-memory
service + result/decision`; do not sum stages that overlap in the same measured
schedule or count nominal memory waits twice. Startup includes tokenizer/model
load, weight placement and configuration and is reported separately.

An immediate existing-route limitation is decisive: the configured UART is
115,200 baud, 8N1, with a 64-byte request header and 64-byte response. Even an
empty request takes **11.111 ms** of serialization. A proposed uint16-token
window (512 bytes), 32-byte mask and that framing takes **58.333 ms**, before
compute. This unchanged route cannot beat the 1.707-ms CUDA measurement.
UART is therefore only a bring-up option, not the proposed performance path.
Conditional integration may implement an AXI/shared-memory ingress and host
packing interface. Report DMA launch/copy/synchronization and platform support
as unresolved until verified; neither a generic VEK280 capability nor a null
measurement supplies a zero-cost transport. No board operation is included.

Current tool/artifact inspection found Vitis/Vivado 2025.2 executables, AIE
compiler/simulator and AIE-ML headers, existing floating/QAT checkpoints and
tokenizer. The actual platform file is
`/home/jp19126/Xilinx/2025.2/Vitis/base_platforms/xilinx_vek280_base_202520_1/xilinx_vek280_base_202520_1.xpfm`;
the older sibling path in extraction notes is absent. File presence does not
prove license access, a successful compile or usable DMA host connectivity.

Official [DS950 v2.11, 3 Aug 2026](https://docs.amd.com/api/khub/documents/BY0FMkLH05y85Nwu4GmWsw/content),
Tables 5/21, identifies VE2802's 304 first-generation AIE-ML compute tiles and
76 memory tiles: 19 MiB local and 38 MiB shared distributed memory respectively.
[AM020 functional overview](https://docs.amd.com/r/en-US/am020-versal-aie-ml/Functional-Overview)
specifies INT8/INT32 and BF16/FP32 native paths; INT8-by-INT4 has INT8 throughput,
not a guaranteed 2x gain. The capacity is distributed, not a flat buffer; actual
usable placement/residency remains unverified. These facts inform alternatives,
not permission to resurrect the legacy BERT accelerator.

## Proposed experiment contract — approval requested, not granted

**Identifier: span_coupled_pilot_v1.** Approval would cover this method, the pilot
allowance, and conditional integration under the gates below. A negative pilot
ends this route without automatic model/horizon/metric search. Keep historical
project configuration and A/B/C labels unchanged; use one study configuration
and fresh result directories when execution is approved.

### Comparisons and information parity

Four newly fitted compact arms, two fixed seeds (42 and 1337), eight planned
fits; only diagnosed correctness failures may consume the separate correction
reserve below:

1. **Coupled span guard**, exactly the operator above.
2. **Independent-pooling ablation**, same encoder/recurrences/weights allowance,
   but pool completed attack/context/anchor vectors independently across anchors
   before interaction. It receives the same complete contexts, not early scores.
   Also evaluate that head swap on the coupled checkpoint without retraining to
   separate immediate operator effect from optimization effects.
3. **Conventional span classifier**, same encoder, all complete anchored contexts and input
   evidence, with a two-layer ReLU head on concatenated paired states. Match
   non-embedding parameter and state allowance within 5%; report actual operations.
   It is allowed to fuse/stream competently and is not forced to materialize spans.
4. **SoPa/rational-recurrence adaptation**, same embedding/encoder, train data,
   fixed epochs and calibration. Match the compact learned-parameter allowance
   within 5%, report its naturally smaller state, and preserve the published
   recurrence rather than inventing a weak approximation. Allow its ordinary
   full-window pattern pooling rather than weakening it to match the proposed
   head; report the different effective context. Record adaptation details.

All four get identical scratch embedding initialization recipe, training-only
augmentation, optimizer (AdamW 0.001, weight decay 0.01), document batch 32,
gradient clip 5, ten fixed epochs (eight FP32 + two QAT), fixed final checkpoint,
and no teacher/distillation. Microbatching may change to satisfy memory limits
without changing effective batch/loss. The inherited pretrained BERT has different
pretraining exposure; comparisons to it are useful quality/cost references, not
attribution of the coupled operator.

Reuse saved floating/W8 BERT, R1 CNN and feature results for identical natural
inputs. Evaluate the frozen CNN/BERT conjunction on the newly fixed probes as
a conventional combination, with complete cost of both stages. Permit one
compact published guard reference: official Prompt Guard 2 22M if entitled;
otherwise the public PIGuard checkpoint with verified license and declared
exposure. Maximum one checkpoint retrieval, 2 GiB, no access circumvention,
remote-code execution or paid service. If neither is accessible, retain the
blocker, complete independent work and limit the baseline claim. No large 8B
guard, host LLM training, or reproduction of every prior paper is included.

For each quality-qualified algorithm, implement a vectorized/fused or compiled
CPU/GPU reference with the same inference arithmetic where available. Do not
compare a streaming FPGA to a Python-per-token loop or assign BERT's measured
timing to another model. For hardware, compare the proposed fused span-bank
mapping with an equally constrained, competently tiled/streamed conventional
mapping of the **same** algorithm; also cost the strongest conventional compact
guard. Share ingress, embedding storage, precision and clock/resource limits.

### Data, exposure and prospective probes

Preserve the existing 10,790/1,355/1,369 source-group training/calibration/scoring
partitions. Fit the 1% operating point only on the natural benign temporary
threshold partition; no challenge-score threshold tuning. Report the saved-score
0.5%/2% secondary points without choosing a winner from them.

Before candidate training/scoring, author and save a small deterministic probe
specification and labels from the threat model, with no target-LLM calls. Use up
to 512 training source groups for at most 2,048 augmented documents shared by
all four compact arms. Use 128 scoring source groups, chosen by a fixed seeded
group sampler independent of scores, for eight predefined variants per group
(at most 1,024 scoring probes). Training recipes and scoring recipes must be
separate and frozen before results. Record source IDs, recipe author/date,
payload text identity, original templates and every model/checkpoint's known
exposure. Any ambiguity in whether an instruction is operative is marked
unresolved before scoring and excluded with a count, not relabelled afterward.

The eight scoring variants form four benign/attack pairs: quoted security
discussion versus operative direction using matched cue words; unrelated benign
decoy padding; directive/target cues separated across short and beyond-H gaps;
and placement across window boundaries. A quoted wrapper is not automatically
benign: the operative authority claim determines the constructed label. Measure
recall/FPR by pair/length/position and whether decoys change decisions. All arms
see the same variants. Agent-authored templates may overlap concepts already in
pretraining; label these as **newly authored development stress probes**, not an
independent unseen-distribution benchmark or adaptive-attack certification.

No final calibration/test/NotInject content or metadata is opened. Do not relabel
old 55-template subsets as unseen. Without independent available data, that
generalization question remains open. An additional corpus would be a separate
scope/data decision, not an automatic fallback.

For uncertainty use paired bootstrap over source groups, with calibration groups
also resampled and thresholds refitted for the prospective study; challenge
recipe and source-cluster sensitivity are reported separately. Primary point
estimates keep the fixed actual fitted thresholds. With only eight authored
recipes, no confidence interval establishes general attack-distribution coverage.

### Quality, utility and continuation gates fixed before results

Natural development requirements remain **recall >=99% and empirical FPR <=1%**.
With the current denominators that allows at most 27 missed attacks and 13 benign
false positives. The one-point recall loss from a 100% reference is an absolute
percentage-point allowance; relative growth from zero observed misses is
undefined, not a 1% relative miss allowance. Report all actual miss counts.

Proposed extra stress-probe requirements are **recall >=95%, FPR <=5%**, both
overall and in the decoy/long-gap attack strata where denominators permit. These
are new, weaker challenge-screen limits, requiring approval now; they do not
replace the natural 99%/1% requirements. Show discrete denominators and all
stratum counts, including failure at beyond-H inputs.

Continue toward integration only when all of the following hold:

- Both fixed seeds meet natural and stress quality bounds, including exported
  arithmetic limits. Calibration and probe specification remain unchanged.
- On the paired benign/operative probe task, the coupled method reduces the
  balanced error rate by at least **5 absolute percentage points** versus the
  independent-pooling ablation in each seed, with pooled paired 95% interval
  excluding zero, while natural recall loss stays within one point. This is the
  predeclared mechanism discriminator, not a post-hoc chosen stratum.
- Against the stronger of conventional span and SoPa arms, show either at least
  **2 points lower probe balanced error** with no natural-bound failure at
  <=1.10x complete region cost, or at least **20% lower complete region cost**
  with probe balanced error no more than one point worse. Both seeds must have
  the same direction. Otherwise the ordinary-computation explanation survives.
- Actual full-region simulation/synthesis establishes exact own-reference
  decisions, complete buffer/transport accounting and a synthesis-estimated resource/clock
  schedule. A matrix/recurrence-only kernel is insufficient. The proposed mapping
  must improve complete region latency by at least **20%** over the competent
  conventional mapping at the common resource envelope; otherwise do not attribute
  benefit to the co-design.
- A complete-check latency scenario, using simulated cycles at the conservatively
  derated synthesis-estimated clock and measured
  host preparation, predicts at least **20%** improvement versus the fastest
  quality-qualified software path on the same workload, and at least **10%**
  versus the fastest software implementation of the same candidate. Transport,
  memory service and launch terms must have documented defensible bounds. If
  those bounds cannot be established offline, report this as an access/evidence
  blocker rather than use zero or start an unbounded integration build.

The four pilot hardware designs below include both conventional algorithm
comparators, so the corresponding region-cost gates require their actual cycle
reports rather than an operation-count substitute. Primary hardware comparison
uses seed 42 checkpoints; seed 1337 is the quality/operation-structure repeat,
not another physical build. Phase C must recheck all cost gates with actual
routed frequency; a derated HLS estimate is still not achieved timing.

These are prospective pilot continuation rules, not proof thresholds for final
publication. Failures in novelty, quality, numerical correctness, hardware cost
or access are reported separately. Gains due only to augmentation, more model
capacity, smaller models versus BERT, or a poor conventional implementation do
not pass the contribution gate. If the task proves too data-limited, report
inconclusive rather than expand it silently.

Legitimate-use constraints are distinct. Natural benign retention must be >=99%
and stress benign retention >=95%. On unchanged original application-pilot
inputs, reuse saved target responses only to compare block/allow consequences:
zero additional rejection among the 20 distinct clean tasks is required. Keep
their low/clustered utility counts visible. Newly accepted/blocked original
attacks may reuse identical saved target requests with explicit blocking
non-completion; no downstream outcome is invented for transformed probes.
New end-to-end target protection and QA usefulness remain unmeasured and cannot
be declared passed by detector labels or these retention proxies.

### Finite resource allowance

The following is a **new proposed allowance**, not expenditure against R1 or
old Goal 4/5 budgets. User removal of the Codex usage limit does not make these
experiment budgets unlimited. Heavy jobs are serial, coordinator-dispatched;
at most two light native subagents, no nested teams. Preserve preflight available
memory floors of Windows 4 GiB / WSL 8 GiB and runtime floors of 2 / 4 GiB, with
at most four CPU threads for heavy jobs. Stop owned jobs on failed floors.

| Stage | Maximum new work | Stopping/resource limits |
|---|---|---|
| Pilot data and implementation | One probe/augmentation specification; four arms and one optional pretrained reference; four complete-region designs as enumerated below | No architecture or recipe sweep; saved baselines reused; small bring-up before fits |
| Pilot learning | 4 arms x 2 seeds = **8 fits**, ten epochs each, including two QAT epochs | **45 min per fit**, **6 GPU-hours total**; fixed final epoch; no hyperparameter rescue. Failure counts consume dispatches |
| Pilot evaluation/numerical work | At most **12 checkpoint bundles** (8 learned, <=3 frozen references, 1 diagnosed rerun); each learned bundle includes FP32 and fixed-point calibration/scoring/probes and the applicable coupled head-swap; frozen conjunction derives from constituent scores; fixed 32 real development numerical documents plus corner cases | **20 min per bundle**, **4 GPU-hours / 8 CPU-hours total**; zero final-set entry; no target generation |
| Pilot timing | All eligible seed-42 algorithms and frozen references/combinations, **<=8 algorithms**, including both CPU and CUDA paths where supported; existing 32 matched single-window documents plus fixed 32 long development documents; 8 warmups, 5 repeats | **30 min per algorithm**, **4 GPU-hours / 4 CPU-hours total**; report per-document distributions, cold separately; a missing competent path limits the fastest-qualified claim |
| Pilot hardware | **4 full-region designs** at the common 128-product-lane target: fused candidate, conventional mapping of that candidate, conventional span classifier, SoPa adaptation; corresponding C and RTL checks | **2 h per design**, **8 vendor wall-hours total**, four CPUs; successful tool access checked once, no device implementation yet; 256-lane calculations remain analytical only |
| Pilot correction reserve | <=**2** diagnosed implementation reruns across learning/evaluation, and <=**2** diagnosed hardware rebuilds | **1 additional GPU-hour**, **2 CPU-hours**, **4 vendor wall-hours**; no changed model, quality limit, horizon, data split or search |
| Conditional integration, only after all gates | One selected format/checkpoint and mapping; export, AXI ingress/host packing, full multi-window simulation, board handoff; **2 synthesis/implementation configurations** plus <=**1** corrective rebuild | **12 h per vendor attempt**, **36 vendor wall-hours total**, four CPUs; <=**6** complete RTL integration cases, <=**4 GPU / 8 CPU-hours** additional software/numerical work; no retraining |

Caps are independent ceilings, not promises to spend them. Total GPU ceiling is
19 hours including correction/integration, pilot vendor ceiling 12 wall-hours,
integration vendor ceiling 36 wall-hours. Install no speculative dependencies;
use current environments. Missing licenses/access block the dependent build,
not saved analysis or software correctness. A correctness repair may use its
stated reserve; an exhausted reserve or scientific method change returns to
the user with retained failed evidence.

Integration delivers an actual board-free result or an explicit incomplete
handoff. It includes existing implemented commands, exact checkpoint/export and
tool/platform identities, achieved report paths, host interface, replay inputs,
startup/steady-state boundaries and remaining physical measurements. It does
not program hardware, open final sets, measure physical energy, invent DMA
latency or claim final deployment validation. If a gate fails, preserve the
negative study and stop this route without pursuing another model until one wins.

## Historical Phase A checkpoint and requested decision

**Scientific conclusion:** no new method or FPGA advantage is established.
The saved evidence is narrow, with complementary detector errors, seen attack
templates and a fragile benign operating point. The prior-art review makes
ordinary compact/cascade/recurrence explanations serious competitors.

**Supporting engineering result:** the saved-score audit is implemented,
executed and independently checked; a complete-chain analytical workload and
specific transport blocker are recorded. Actual tool paths and device facts
were inspected. No training, inference campaign, vendor build/simulation,
protected-data access, target-model call or board action occurred in Phase A.

**Open question:** can the explicit same-span mechanism clear the matched
ordinary-span/WFSA and hardware tests above, and can a credible fast host path
be established without physical access? Neither answer is implied by this plan.

**Approval requested:** select `span_coupled_pilot_v1` and approve the listed
pilot plus conditional integration allowance, data/probe additions and
prospective quality/cost gates. After that one decision, Phase B/C can proceed
autonomously under the gates. Until then, substantial implementation and
experiments remain stopped at the checkpoint required by the invoked prompt.

## Phase B evidence — first completed fit (2026-10-13)

The subsequent user approval at the top of this report supersedes the historical
checkpoint above. The first planned coupled seed-42 fit completed all eight
FP32 and two QAT epochs in 1,013.062 seconds. It used the fixed augmentation,
document batch, optimizer and checkpoint rule. No architecture, threshold,
recipe or quality limit was changed after observing the following results.

At the separately fitted 1% natural-calibration operating points, confusion
counts are TP/FN/FP/TN. Natural scoring has 2,738 attacks and 1,369 benign
documents; stress scoring has 512 of each.

| Frozen variant | Natural counts | Stress counts | Interpretation |
|---|---:|---:|---|
| Epoch-8 FP32 | 2738 / 0 / 19 / 1350 | 227 / 285 / 16 / 496 | Natural FPR and stress recall fail |
| Post-QAT weights evaluated FP32 | 2738 / 0 / 16 / 1353 | 258 / 254 / 258 / 254 | Natural FPR and both stress limits fail |
| Exported integer reference | 2738 / 0 / 17 / 1352 | 258 / 254 / 258 / 254 | Same quality failure; arithmetic degradation within limits |
| Coupled checkpoint with independent head pooling, FP32 | 2655 / 83 / 26 / 1343 | 255 / 257 / 243 / 269 | Diagnostic head swap, not independently trained ablation |
| Same head swap, integer | 1229 / 1509 / 1 / 1368 | 128 / 384 / 85 / 427 | Severe clipping/tie effect in the swapped head; do not infer the learned ablation's result |

The exported candidate threshold is 0.00628990214318037; the same-weight FP32
threshold is 0.00641479529440403. Export adds one natural benign rejection
(0.073046 percentage points), no missed attacks, and no stress count change.
Thus this first model's poor stress result is not explained by its integer
conversion. Its 2,000-resample source/calibration-refit bootstrap natural FPR
interval is 0.4383–2.4836%; zero observed natural misses again produce a
degenerate empirical recall interval. This uncertainty does not override the
predeclared empirical continuation limits.

Stress behavior is sharply recipe-dependent: both boundary and decoy attack
recipes are detected in 128/128 cases, but their benign counterparts are also
rejected in 128/128. The operative-direction and beyond-context-gap attack
recipes are each detected in only 1/128, matching their benign counterparts'
1/128 rejection counts. These are failures on the authored development probes;
they neither establish a general attack rate nor justify changing the probes.
The trained independent, span-MLP and SoPa comparisons and second seed remain
necessary to explain the mechanism and are still pending at this checkpoint.

Supporting implementation evidence: exact full-chain C replay passed all 143
cases derived from 32 fixed training documents plus boundary, empty, strict-tie,
multiwindow and continued-maximum cases. The model weights are trained and the
expected values come from the independent integer path. This is C execution,
not RTL cycle evidence, synthesis success or physical measurement.

The unchanged application-pilot integer guard rejects none of the 20 distinct
clean tasks and all 100 attacks. Offline replay preserves nine completed clean
tasks among those 20; blocking all attacks also means no legitimate completion
on attacked inputs. No new target call occurred, and no stress-probe downstream
outcome is inferred. This small reuse result does not repair the failed stress
gate or establish deployment safety.

Evidence: `results/span_coupled_pilot_v1/coupled_seed42_v1/`,
`coupled_seed42_score_v1/`, `analysis_coupled_seed42_v1.json`,
`coupled_seed42_fixtures_v1/`, `hls_coupled_stream_v1/c_run.json`, and
`coupled_seed42_pilot_integer_replay_v1.json` under the same study root. The
first quality gate has failed; conditional Phase C integration is not justified.
The remaining bounded Phase B comparisons and hardware feasibility work continue.
All changes stay local; GitHub synchronization requires a separate user decision.

### Frozen-reference and unchanged-input application results

All references below use the same source-separated development calibration
rule and document maximum. BERT/CNN natural scores were reused only after
identity and exact short/full-window rechecks. PIGuard uses its own tokenizer
in the adapted length-256/window-maximum protocol; this is not a reproduction
of its native pipeline or reported paper result. Its external training overlap
with this population is unresolved. Prompt Guard 2 access was unavailable;
PIGuard is the approved pinned public fallback.

| Frozen reference | Natural TP/FN/FP/TN | Stress TP/FN/FP/TN |
|---|---:|---:|
| R1 CNN | 2738 / 0 / 24 / 1345 | 221 / 291 / 281 / 231 |
| R1 BERT W8 / CUDA Graph | 2738 / 0 / 13 / 1356 | 96 / 416 / 198 / 314 |
| Separately calibrated BERT AND CNN | 2738 / 0 / 4 / 1365 | 94 / 418 / 146 / 366 |
| PIGuard, pinned public checkpoint | 2258 / 480 / 22 / 1347 | 511 / 1 / 512 / 0 |

None passes the stress screen. Natural conjunction FPR is 0.2922%, with a
2,000-draw paired source/calibration-refit interval of 0–0.6574%; its apparent
natural benefit does not extend to the authored probe set. Both component
thresholds are refitted within each resample. Conjunction decisions are the
AND of strict component decisions, never the inherited BERT risk field in the
saved conjunction rows. No cascade latency advantage was measured.

On the unchanged 20 distinct clean tasks, BERT, conjunction and PIGuard reject
none and preserve nine saved successful completions. CNN rejects one additional
clean task and preserves eight completions, failing the clean-utility gate.
BERT, CNN and conjunction reject all 100 attacked inputs; PIGuard allows one,
whose saved response completes the legitimate task without attack success.
These response-specific offline consequences require no target calls and do
not supply downstream outcomes for the stress probes. Clean-task replication
across 100 logical pairs does not increase the number of distinct clean tasks.

Evidence: `analysis_first_seed_references_v1.json`,
`conjunction_analysis_v2/analysis.json` and the `*_pilot_replay_v1.json` files
under `results/span_coupled_pilot_v1/`. The conjunction adapter's first real
analysis exposed a NumPy-count JSON serialization error; v1 partial output
is retained, native-integer conversion is fixed and the v2 analysis completed.

### Bounded qualitative inspection

The fixed first source group `aeslc:train:beck-s_inbox_192.subject` is absent
from study training and calibration despite the upstream dataset's `train`
name. All eight constructed documents match their saved token IDs, tails and
complete-anchor coverage; labels follow the explicit reported-data versus
operative-direction text. This inspection found no reason to relabel data.
At the candidate's integer threshold, quoted/operative risks are
0.00005225/0.00061663 (both allowed), decoy risks 0.029535/0.276691 (both
rejected), gap risks 0.00004909/0.00008682 (both allowed), and boundary risks
0.994135/0.999531 (both rejected). Equal decisions do not mean equal scores or
representations.

The boundary paragraphs begin at global token 230, as specified. However, the
benign maximum comes from the later window beginning at token 380, containing
only punctuation and repeated background/record padding. That same tail risk
also occurs in the attacked document. This false alarm cannot isolate quotation
handling or context loss at a boundary. The 80-token gap exceeds the 64-token
footprint between early direction and target, but the operative preamble already
signals an attack. Moreover, the benign/attack documents differ in wording and
length (258 versus 247 content tokens in this pair, hence two versus one window).
These probes do not isolate long-range target binding or a pure role contrast.
The frozen screen remains unchanged; its failure is valid for these inputs,
while these mechanistic interpretations remain unsupported.

The only natural same-weight FP32-to-integer decision change is the benign
parking-availability notice `aeslc:train:rapp-b_inbox_50.subject:clean`.
Its score moves from 0.00637466367 to 0.00679753162, while its separately
calibrated threshold moves from 0.00641479529 to 0.00628990214. Either the score
increase or threshold decrease alone crosses the decision boundary. The added
false positive therefore cannot be attributed solely to sigmoid-table rounding.
Evidence is in the respective saved scoring rows and prepared stress documents;
the inspection made no data, label, model or threshold changes.

### Hardware progress and temporary shared-memory hold

The first synthesis attempt failed at the loader's scalar-to-array interface.
After a serialization-preserving correction, fresh v3 replay again matched
all 143 cases and 46 nonempty-window intermediate traces. Vitis passed the
previous error and reached RTL generation, but the resource guard terminated
it at 142.511 seconds when Windows available memory reached 2,041,344,000 bytes,
below the fixed 2 GiB runtime floor. Cleanup confirmed no owned process group
remained. Top-level synthesis reports and RTL co-simulation were not produced;
partial submodule estimates cannot establish full-region latency or resources.

A separate native Windows Vivado build was consuming substantial host memory.
It was inspected only and left running. The coupled seed-1337 launch was
refused at the 4 GiB preflight floor before any fitting. The original failed
directories are preserved; remaining software work awaits capacity. One of
two corrective vendor attempts remains available. This is a resource hold,
not an algorithm result or permission to reduce the memory floor.

### Second coupled seed completed after resource release

The competing chat completed its vendor and CPU measurements before SENTRY
resumed. The unchanged seed-1337 v3 fit completed in 1,141.735 seconds, with
zero no-anchor training windows; its frozen source hashes match seed 42.
The full evaluation bundle took 73.078 seconds. Conservative accounting
consumes the two software correction slots for the refused/interrupted sequence;
no further learning/evaluation campaign retry remains.

| Seed-1337 variant | Natural TP/FN/FP/TN | Stress TP/FN/FP/TN |
|---|---:|---:|
| Epoch-8 FP32 | 2738 / 0 / 16 / 1353 | 57 / 455 / 138 / 374 |
| Post-QAT weights evaluated FP32 | 2738 / 0 / 6 / 1363 | 254 / 258 / 256 / 256 |
| Exported integer reference | 2738 / 0 / 6 / 1363 | 254 / 258 / 256 / 256 |

The final FP32/integer thresholds are 0.030326848849654198 and
0.030908958986401558. Both pass the natural screen and fail the stress screen
(49.6094% recall, 50% FPR). Integer conversion changes neither confusion count
and passes its relative numerical limits; this is not a claim of identical
scores. Both coupled seeds now fail the required authored-stress gate.
Phase C therefore remains unjustified regardless of later hardware results.
Trained independent, span-MLP and SoPa comparisons still determine attribution.

Both application-pilot arithmetic variants preserve nine saved completions
among 20 distinct clean tasks, add no clean rejection and reject all 100
attacks. No new target response is generated. Evidence is under
`coupled_seed1337_v3`, `coupled_seed1337_score_v1`,
`coupled_seed1337_pilot_*_v1` and `analysis_coupled_both_seeds_v1.json`.
The final hardware correction is staged in `hls_coupled_stream_v4`; its fresh
C replay passes 143 cases. The vendor attempt begins only after the software
bundle has finished and consumes the last hardware correction slot.

### Final fused correction: synthesis resource failure, incomplete RTL

The v4 correction completed top-level HLS synthesis for the frozen fused
candidate on `xcve2802-vsvh1760-2MP-e-S`. The full top report estimates
104,176 LUT, 50,671 FF, 167 DSP, 99 BRAM18 (49.5 BRAM36 capacity equivalent)
and 65 URAM. LUT use exceeds the prospective 100,000 limit by 4,176 (4.176%);
the common resource-envelope gate therefore fails. The other reported resources
fit their limits. BRAM capacity equivalence is not a routed packing result.
Estimated period is 5.992 ns against the unchanged 5 ns target; applying the
declared 20% frequency derating gives 7.490 ns (approximately 133.51 MHz).
These are HLS estimates, not achieved frequency or physical-device measurements.
The top's average/worst latency bounds are not measured cold or warm costs.

The same full-flow attempt entered RTL simulation but completed zero of five
transactions before the memory guard stopped it at
2026-10-13T04:25:51.719376Z. Failed telemetry records Windows available memory
1,738,067,968 bytes below the 2 GiB runtime floor; WSL remained at
12,254,113,792 bytes above its 4 GiB floor. Elapsed vendor time was
294.338694 seconds, return code -15, and cleanup confirmed no owned process
group remained. There is no completed RTL pass or transaction latency table.
The separate cold preload and four warm cases therefore have unresolved
full-region cycles. C replay of all 143 cases remains a software result.

Evidence is preserved under
`results/span_coupled_pilot_v1/hls_coupled_stream_v4/`: `c_run.json`,
`vendor_full-flow.json`, `vendor_full-flow.resources.jsonl`,
`vendor_full-flow.stdout.log`, the exact `full-flow.tcl`, and
`hls_project/solution/syn/report/span_guard_top_csynth.xml`.
Generated simulation artifacts remain under `hls_project/solution/sim/`.
The generated testbench records latency as finish minus start clock count and
declares five transactions, but its transaction output was not produced.
The full-flow Tcl enables top-port waveform tracing and dataflow profiling;
the stopped waveform file was only 10,180,744 bytes. These facts do not
identify the dominant memory allocation or establish that disabling waveform
tracing would solve the capacity problem. Global memory pressure is observed;
its precise allocation source remains unresolved.

Both diagnosed hardware correction slots are now consumed. The coordinator
decided not to restart the fused simulation or broaden the envelope. The
earlier statement that one correction remained is superseded by this attempt.
This is a negative result for the implemented fused mapping's resource gate,
not proof that every mapping or every span method is infeasible. The two-seed
quality failure already prevents Phase C integration.

For the three remaining initial designs (buffered same-candidate, span MLP,
SoPa), the bounded operational recommendation is to use
`cosim_design -rtl verilog -tool xsim -trace_level none -argv {...}`, omitting
optional dataflow profiling. [AMD UG1399 2025.2](https://docs.amd.com/r/en-US/ug1399-vitis-hls/cosim_design)
documents that `none` saves no waveform data and that dataflow profiling tracks
channel sizes. All three mappings are buffered; complete transaction cycles
and independent output comparisons remain the primary evidence. Preserve
the original generated Tcl, record the actual revised Tcl hash, retain the
same cold preload and four compact warm cases, and keep deadlock/dependency
checks, numerical identities, part/clock constraints, four CPUs, memory floors
and the 7,200-second per-design budget unchanged. This removes unnecessary
instrumentation; it is not a demonstrated memory remedy or authorization for
another fused correction. Any further memory termination remains an unresolved
RTL result rather than a passed cycle gate.

### First trained independent-pooling comparison

The fixed seed-42 independent arm completed ten epochs in 1,174.938 seconds,
with zero training windows lacking eligible anchors. Its separately calibrated final FP32 and
integer thresholds are 0.0016837591538205743 and 0.0018969200318679214.

| Independent seed-42 variant | Natural TP/FN/FP/TN | Stress TP/FN/FP/TN |
|---|---:|---:|
| Post-QAT weights evaluated FP32 | 2738 / 0 / 23 / 1346 | 387 / 125 / 282 / 230 |
| Exported integer reference | 2738 / 0 / 20 / 1349 | 386 / 126 / 281 / 231 |

Both variants have stress balanced error **39.7461%**, versus **50%** for
the corresponding coupled seed-42 variants. The coupled method therefore has
a **10.2539-point disadvantage** in this comparison, opposite to its required
five-point benefit in each seed. The independent arm still fails natural FPR,
stress recall and stress benign-retention bounds; it is not a qualified solution.
Its integer natural false-positive count is also higher than the coupled arm's:
20 versus 17. Integer conversion loses one stress detection (0.1953 recall
points) and removes one stress false positive, within the relative arithmetic
limits; this does not repair the absolute quality failures.

The integer recipe counts locate the difference. Each cell below is the number
of rejected attack documents / rejected benign documents, each out of 128.

| Matched recipe family | Coupled | Independently trained pooling |
|---|---:|---:|
| Operative direction / quoted discussion | 1 / 1 | 128 / 24 |
| Decoy | 128 / 128 | 128 / 128 |
| Gap | 1 / 1 | 2 / 1 |
| Boundary | 128 / 128 | 128 / 128 |

The operative/quoted family accounts for 104 of the 105 fewer classification
errors; the gap family accounts for the remaining one. In final FP32, the
independent gap counts are 3 / 2, leaving the same net error improvement.
For the already inspected first source group
`aeslc:train:beck-s_inbox_192.subject`, the independent integer guard allows
quoted discussion at risk 0.00014768 and rejects operative direction at
0.61878043. The coupled guard allowed both. Both guards retain the same
incorrect benign rejections for the decoy and boundary examples and miss the
gap attack in this group.

This is evidence against the claimed coupling benefit for this frozen seed,
training recipe and authored probe task. It does not establish that pooling
generally recognizes instruction roles better: the separately trained weights
and calibrated operating points differ, the recipes are not minimal role-only
counterfactuals, and the documented background-padding and gap confounds remain.
No new bootstrap or model inference was performed for this inspection. The
second independent seed and ordinary span/SoPa comparisons remain pending.
Both independent application-pilot arithmetic variants add zero clean rejection,
retain nine saved completions among 20 distinct clean tasks and reject all
100 attacks; those unchanged-input replays add no new target-model evidence.

Evidence under `results/span_coupled_pilot_v1/`: `independent_seed42_v1/`,
`independent_seed42_score_v1/{fp32_postqat,integer}/metrics.json`, their saved
`stress_probes_documents.jsonl` rows, the corresponding coupled seed-42 rows,
and `independent_seed42_pilot_{fp32,integer}_replay_v1.json`. All labels,
threshold rules, data and frozen model sources remain unchanged.

### Conventional mapping: synthesis stopped before a cost result

The first buffered-candidate attempt (`hls_coupled_buffered_v1`) passed its
143-case C simulation but failed HLS synthesizability after34.217 vendor seconds.
SYNCHK200-41/43 rejects a pointer selection between a projection row and a
separate zero array at the saved source's line171. The last resource observation
had14.49GB Windows and23.44GB WSL available; this was a compiler error, not the
prior memory failure. The owned process group was cleaned up. No top resources,
completed RTL transactions or conventional-versus-fused latency ratio exist.
See its `vendor_full-flow.json`, stdout log and `hardware_terminal_report_v1.json`.

Both corrective hardware rebuilds were already consumed, so the buffered and
fused attempts are not being repeated. The current source has a narrow correction
for the two remaining initial MLP/SoPa builds: a validity flag selects zero scalar
inputs inside the existing eight-lane recurrence, using a row from a single
projection array. This avoids both the unsupported pointer selection and an
extra64-element copy loop. The recurrence, rounding, clipping, masking, horizon,
part/clock and resource limits are unchanged. Existing build snapshots retain
the old source. SoPa's selected WFSA path does not call the changed recurrence.
GNU C++ compiled the three buffered variants and133,632 isolated coordinate
comparisons passed in2.807 seconds. This establishes focused C equivalence,
not HLS acceptance or a replacement full trained-model replay. New MLP/SoPa
artifacts will need their own complete checks before their first vendor dispatch.

### Completed two-seed coupling discriminator

Independent seed1337 completed the same8+2 recipe in1173.813 seconds with frozen
source hashes and no uncovered training windows. Natural final FP32 and integer
counts are2738/0/15/1354, both failing the1% FPR bound. Stress counts are
251/261/263/249 (FP32) and249/263/263/249 (integer), both failing the stress gate.
The integer recall loss is0.390625 percentage points; FPR is unchanged, so its
separate relative arithmetic limits pass. Both application replays again add
zero rejection among20 distinct clean tasks, retain9 saved clean completions and
block100 attacks. These are saved-response consequences, not new target calls.

The predeclared paired source/calibration analysis is complete for both coupled
and independently trained ablation seeds. Coupling's stress balanced-error gain
is-10.25390625 points in seed42; seed1337 gives+0.9765625 points for FP32 and
+1.171875 for integer. Neither seed meets the required>=5-point gain. The pooled
paired95% intervals are[-6.005859375,+0.5859375] and[-5.712890625,+0.537109375]
points, respectively; both include zero. Thus this study does not establish a
coupling benefit. The average estimate is unfavorable, but these intervals do
not establish a uniformly harmful mechanism. They are conditional on two fixed
trained seeds and the authored development recipes, not a population of training
runs or general attack distributions. No metric, threshold or probe was changed.

Evidence: `independent_seed1337_v1`, `independent_seed1337_score_v1`, its two
`independent_seed1337_pilot_*_replay_v1.json` files, and
`analysis_coupled_independent_both_seeds_v1.json` under the study result root.
The latter took8.359 seconds and includes independent recounts, relative
numerical gates and2000 paired source/calibration resamples. Conventional
span-MLP/SoPa attribution remains pending; Phase C already fails its gates.

### First span-MLP result and its arithmetic-sensitive probe margin

The fixed span-MLP seed-42 fit completed in approximately 1,091.8 seconds;
its evaluation bundle took 41.344 seconds. Final same-weight FP32 and integer
thresholds are 0.0026411053258925676 and 0.0026316740550100803.

| Span-MLP seed-42 variant | Natural TP/FN/FP/TN | Stress TP/FN/FP/TN | Stress balanced error |
|---|---:|---:|---:|
| Post-QAT weights evaluated FP32 | 2738 / 0 / 21 / 1348 | 299 / 213 / 259 / 253 | 46.0938% |
| Exported integer reference | 2738 / 0 / 21 / 1348 | 336 / 176 / 259 / 253 | 42.4805% |

Both variants fail natural FPR and both stress quality limits. Their stress
error is below coupled seed 42's 50%, but the second span-MLP seed, SoPa and
actual complete-region costs are still required for the conventional-arm
comparison. These point results cannot pass a combined quality/cost gate.
Both application-pilot variants add zero rejection among 20 distinct clean
tasks, retain nine saved completions and reject all 100 attacks; no target
requests were made.

A bounded saved-score recount found **37 changed document decisions**, all
attack misses becoming detections in `operative_direction`: 42/128 detected
in FP32 versus 79/128 in integer arithmetic. Every other stress decision and
all 4,107 natural decisions are unchanged. In both arithmetic variants,
quoted-discussion false positives are 2/128; decoy and boundary attack/benign
rejections are each 128/128; gap attack/benign rejections are each 1/128.
The 37 detections raise stress recall by 7.2266 points without changing FPR.
This is within the prespecified one-sided arithmetic-degradation limits, but
is clearly not numerical invariance.

The threshold falls by only 0.00000943127. Applying the integer threshold to
all FP32 scores changes no natural or stress decisions; applying the FP32
threshold to all integer scores also changes none. Thus the 37 improvements
come from arithmetic-dependent score shifts across either threshold, not
from the small recalibration shift. Among changed rows, FP32 risks range from
0.00231825 to 0.00259384 and integer risks from 0.00265226 to 0.00298103;
the increases range from 0.00029331 to 0.00040825. There are 37 distinct FP32
risks and 14 distinct integer risks in this set.

The lexicographically first changed document is
`span:probe:aeslc:train:beck-s_sent_230.subject:operative_direction`.
Its two window scores are [0.00251526013, 0.00003925949] in FP32 and
[0.00284490967, 0.00004006369] in integer arithmetic. The first window remains
the maximum; its document margin moves from -0.00012584520 to +0.00021323562
against the respective thresholds. This is a systematic recipe-local margin
effect, not evidence of improved semantic generalization.

Final sigmoid-table rounding alone cannot explain the observed magnitude:
at these saved FP32 probabilities, a half-grid logit displacement of 1/256
changes probability by at most about 0.00001013, far below the smallest
observed increase. Upstream quantized weights, encoder/projection values,
states or head arithmetic must also affect the scores. Saved probabilities
do not localize those effects or establish a numerical implementation defect;
the independent C validation is a separate correctness check. No inference,
new bootstrap, threshold tuning or model/data edit was performed for this
inspection.

Evidence under `results/span_coupled_pilot_v1/`:
`span_mlp_seed42_v1/`, `span_mlp_seed42_score_v1/{fp32_postqat,integer}/metrics.json`,
their natural and stress document-score JSONL files, and
`span_mlp_seed42_pilot_{fp32,integer}_replay_v1.json`.

### First span-MLP hardware result

The corrected span-MLP design passed all 143 complete C cases and 46 nonempty
intermediate trace checks on the same 32 selected training documents used for
the coupled design. Its first HLS attempt accepted the pointer-selection fix
and completed synthesis: 95,120 LUT, 37,954 FF, 292 DSP, 119 BRAM18 (59.5 BRAM36)
and 65 URAM. DSP use exceeds the fixed 256 limit; other resource counts fit.
Estimated period is 5.992 ns against the requested 5 ns, giving the same
7.490 ns conservatively derated estimate as the fused candidate. No observed
warm-cycle comparison follows from the top's mixed latency estimates.

RTL simulation with optional waveform/profiling disabled stopped at the host
memory floor at 2026-10-13T05:46:37.227046Z. Windows available memory was
1,902,804,992 bytes, below 2 GiB; WSL had 12,141,441,024 bytes. No transaction
completed. Vendor time was 422.374894 seconds; return code -15 and
assigned_group_present=false confirm owned-process cleanup. Removing optional
instrumentation did not establish a viable simulation route on this host.
The precise dominant allocation remains unmeasured. No corrective restart ran.
Evidence is in `hls_span_mlp_v1/hardware_terminal_report_v1.json`, its vendor
JSON/log/resource records, and `span_mlp_seed42_trace_v1.json`. These results
separate exact C agreement, failed resource fit and unresolved RTL behavior.

### Second span-MLP seed: natural pass, stress failure

The fixed seed-1337 fit completed in 1,071.125 seconds. Its model, numerical
and training source hashes match the first coupled fit. Guarded scoring took
42.359 seconds; application-pilot FP32 and integer scoring took 9.375 and
16.594 seconds, respectively. These three subprocesses total 68.328 seconds.
The final same-weight FP32 and integer thresholds are 0.03790095075964928 and
0.037892427295446396.

| Span-MLP seed-1337 variant | Natural TP/FN/FP/TN | Stress TP/FN/FP/TN | Stress balanced error |
|---|---:|---:|---:|
| Post-QAT weights evaluated FP32 | 2738 / 0 / 9 / 1360 | 137 / 375 / 169 / 343 | 53.1250% |
| Exported integer reference | 2738 / 0 / 9 / 1360 | 136 / 376 / 162 / 350 | 52.5391% |

Both variants pass natural quality (100% recall, 0.6574% FPR) and fail both
stress limits. Integer stress recall falls by 0.1953 percentage points while
FPR falls by 1.3672 points; this passes the one-sided numerical-degradation
limits, without establishing decision invariance or absolute quality.

A saved-score comparison finds eight changed stress decisions, all from
rejection to acceptance: seven `decoy_benign` false positives are removed and
one `decoy_attack` becomes a miss. This reduces total errors by six. Swapping
the two calibrated thresholds changes no stress decisions in either score
set; arithmetic-dependent score shifts drive these crossings. For example,
`span:probe:aeslc:train:mann-k_sent_2033.subject:decoy_attack` falls from
0.0380309522151947 to 0.03488088771700859. These saved scores do not identify
the responsible operator or establish an implementation defect.

Both saved-response replays add zero rejection among 20 distinct clean tasks,
retain nine saved completions and reject all 100 attacks. No target calls ran.
The opposite seed-specific span-MLP versus coupled point ordering reinforces
the need for the prespecified paired comparison and actual complete-region
costs; it does not establish a conventional-arm or combined cost-gate pass.
At this checkpoint SoPa seed 42 was running under the unchanged 2,700-second
limit; seed 1337 and the initial SoPa hardware design were pending. All six completed fits
fail stress quality, so conditional integration remains disqualified.

Evidence under `results/span_coupled_pilot_v1/`: `span_mlp_seed1337_v1/`,
`span_mlp_seed1337_score_v1/{fp32_postqat,integer}/metrics.json` and
`stress_probes_documents.jsonl`, the three scoring `resources.json` records,
and `span_mlp_seed1337_pilot_{fp32,integer}_replay_v1.json`.

### Final stopping decision and evidence boundary

The coordinator stopped the first SoPa fit through the existing pause marker
after logged epoch-1 batches 0/50/100/150 at 5.593/96.328/189.921/282.109
seconds. The saved decision used the first two 50-batch intervals: 1.84328
seconds per batch projected 19,833.6928 seconds for the fixed 10,760 batches,
against the 2,700-second dispatch ceiling. This is an early projection for the
current implementation, not a measured complete runtime, actual timeout or
inherent SoPa cost. The process exited with `Study pause requested` after
318.000 guarded seconds; this was not a weekly-usage pause. The marker was
removed after exit. Evidence is in `sopa_seed42_v1/early_stop_decision_v1.json`,
`training_log.jsonl`, `resources.json` and `failure.json`.

No partial checkpoint was scored, no shorter recipe substituted, and no retry
ran. SoPa seed 1337 and its initial hardware design remain unexecuted; there is
no trained SoPa export for that hardware comparison. Caps were ceilings, not
promises to spend them. The candidate had already failed independent quality,
mechanism and resource gates, so this bounded negative study stops without
finishing the comparative protocol or claiming it complete.

The completed saved-only `analysis_final_available_v1.json` consolidates the
six fits and frozen references. The coupling mechanism gate is complete and
fails. All six final integer variants pass the relative numerical limits on
both natural and stress splits; none passes every absolute quality limit.
The conventional-arm discriminator has `quality_complete=false` because both
SoPa seeds are missing, `cost_evidence_complete=false`, and `passes=null`.
Unknown comparative evidence is not converted into a failed comparison or a
claim that the candidate is dominated on every metric.

### Final pilot budget ledger

Times below use saved resource guards or vendor-run manifests. They are process
wall seconds, not measured active GPU hours or CPU-core hours, and exclude
outer command-launch overhead. Evaluation totals include the separate FP32
and integer application-pilot scoring subprocesses. Small preparation,
bring-up, retrieval, C checking and saved-only analyses are separate supporting
work, not additional trained fits or inferred hardware measurements.

| Work recorded | Count and disposition | Guarded wall seconds |
|---|---|---:|
| Completed training | Six fixed 8+2-epoch fits | 6,666.439 |
| Interrupted training | Coupled 1337 v2: memory floor; SoPa 42: coordinator stop | 334.390 + 318.000 |
| All executed training above | Eight executions; plus one earlier coupled-1337 preflight refusal with no fitting | 7,318.829 |
| Learned evaluation | Six checkpoint bundles, 18 score/pilot subprocesses | 465.687 |
| Frozen references | BERT, CNN and PIGuard: three bundles | 402.578 |
| Completed checkpoint evaluation | Nine bundles total; no SoPa bundle | 868.265 |
| Conjunction | Derived from constituent saved scores, not a fourth reference inference | 1.110 |
| Initial vendor designs | Fused v2, buffered v1, span-MLP v1 | 43.458 + 34.217 + 422.375 |
| Corrective vendor attempts | Fused v3 and v4 | 142.511 + 294.339 |
| All vendor attempts | Three initial designs plus two corrections; SoPa unexecuted | 936.900 |

Conservative dispatch accounting retains the coupled-1337 preflight refusal as
its planned attempt and v2/v3 as the two software correction slots. Their
combined 1,476.125 seconds fit the one-hour additional GPU-dispatch allowance;
both slots are consumed. Both vendor correction slots are also consumed
(436.850 seconds). Unspent elapsed-time ceilings do not create new retry slots.
The evaluation table's diagnosed-rerun entry does not independently reopen the
shared exhausted software correction reserve. The initial `hls_coupled_stream_v1`
directory contains C preparation/replay, not another vendor invocation.

Evidence: each completed/failed fit and scoring directory's `resources.json`,
completed fits' `run.json`, `hls_*/vendor_full-*.json`, and the dispatch history
in `reports/execution_log.md`. No timing campaign or conditional integration
ran. No protected held-out partition or metadata was opened, no target-model API call
or physical-board action ran, and no GitHub synchronization occurred. The final
official usage reading is 7% at boundary 1792472301, below the active 50% cap.
The next proposed action is a newly approved study addressing the unresolved
comparison or a revised hypothesis; it is not automatic continuation of this
failed candidate or permission to change its frozen limits.
