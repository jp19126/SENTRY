# CNN prior art and preliminary hardware cost model

## Question and evidence boundary

The work package asks whether different convolutional context paths or evidence summaries improve indirect prompt-injection detection on the same external-document inputs, and what those choices would cost in a board-free FPGA design. It supports candidate selection; it is not itself a new guard method or an FPGA result. The approved application is untrusted external documents in document question answering. The broad CNN prompt authorizes local training/search, while final calibration/test, NotInject, target-LLM calls, paid services, cloud compute, board actions, and GitHub synchronization remain outside this evidence boundary.

The discriminator is a source-group-disjoint inner comparison across 16 registered model families (96 configurations, with two pairs sharing the same evaluation-time computation), followed by the frozen common development evaluation. Comparisons must retain complete-document tails, the same labels/windows, and maximum window risk unless an aggregation change is separately identified. Any ranking here is analytical or literature-derived; no synthesis number, FPGA latency, energy, or security benefit is inferred.

## What the previous CNN establishes

The R1 reference is a scratch-trained WordPiece model with 2,015,426 parameters: a 30,522 by 64 embedding table, three parallel Conv1d branches of widths 3, 5, and 7 with 64 output channels each, ReLU, valid-center masked global maximum pooling, dropout 0.2, and a two-class linear head. The fixed recipe was seed 42, ten AdamW epochs at 0.001, FP32 on an RTX 4070. This is a small TextCNN adaptation, not a reproduction of Kim's embedding/channel/training recipe.

At the saved R1 threshold, it produced TP/FN/FP/TN = 2,738/0/24/1,345 on 4,107 development documents: recall 100%, benign FPR 1.753%, so it failed the 1% empirical screen. The 24 clean false alarms were 12 of 187 long documents (6.42%), 12 of 1,182 single-window documents (1.02%), 23 AESLC examples, and one of 20 EmailQA examples. False positives included a long energy/newsletter digest, procedural work emails, quoted or enumerated text, and ordinary business correspondence; these descriptions are observations from the saved clean source rows, not a causal analysis. No attack was missed in this reused split. Its attack templates were all represented in training, and the benign EmailQA denominator is small. These results motivate length and hard-benign diagnostics; they do not reject CNNs or establish generalization.

Evidence paths: `reports/goal_r1_decision.md`; `scripts/r1_models.py`; `results/goal_r1/cnn_initial_v1/run.json`, `history.json`; and `results/goal_r1/score_cnn_cuda_v1/metrics.json` plus saved document scores. The source excerpts were inspected only for the 24 false-positive clean scoring examples. No model was rerun for this audit.

## Architecture literature map

All rows below map to a branch in `configs/cnn_search_v1.json`; the repository implementation defines the tested topology. A cited model is a structural precedent, not evidence that the cited system detected prompt injection. The architecture question is whether the branch changes receptive field, information retention, or mapping cost under SENTRY's fixed document task.

| Tested branch | Nearest structural precedent and accessible code | Concrete distinction and comparison |
|---|---|---|
| `textcnn` | Kim's parallel-filter sentence CNN; [paper](https://aclanthology.org/D14-1181/) and [author code](https://github.com/yoonkim/CNN_sentence). | Parallel short kernels followed by global evidence pooling. Recreate a competent shallow reference; vary kernels and pool only as named config factors. |
| `stacked` | Shallow word-level CNN comparison by [Johnson & Zhang](https://arxiv.org/abs/1609.00718), alongside the deeper [DPCNN paper](https://aclanthology.org/P17-1052/). | Successive token-preserving convolutions grow context without early sequence reduction; compare against matched receptive-field and MAC alternatives. |
| `residual` | Identity residual paths; [He et al., CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). | Residual learning is a vision-derived training/optimization precedent. The token-preserving text topology is a project adaptation; the cited vision result says nothing about injection detection. |
| `dpcnn` | [Deep Pyramid CNN for Text Categorization](https://aclanthology.org/P17-1052/); an [accessible PyTorch reference](https://github.com/Cheneng/DPCNN) exists but is not identified as the paper's official code. | The registry uses one temporal convolution per residual block and, after selected blocks, centered max-3 followed by nonoverlapping pair max. This is a DPCNN-style downsampling branch, not the paper's two-convolution block reproduction. Its pooling/mask alignment is part of the implementation comparison. |
| `dilated` | [Iterated/dilated context for sequence labeling](https://aclanthology.org/D17-1283/) and the general TCN study [Bai et al.](https://arxiv.org/abs/1803.01271). | Dilation expands receptive field without increasing tap count; compare to contiguous large kernels at matched coverage. Dilation holes still create delay storage and can hurt memory access. |
| `idcnn` | [Fast and Accurate Entity Recognition with Iterated Dilated Convolutions](https://aclanthology.org/D17-1283/). | Reuses an independent dilated block stack over repeated iterations. Weight sharing reduces stored weights, not the repeated MAC count; distinguish this from simply increasing depth. |
| `tcn` | [Temporal Convolutional Networks](https://arxiv.org/abs/1803.01271) and [author reference implementation](https://github.com/locuslab/TCN). | The repository includes causal, centered, and bidirectional contexts. The original TCN study covers sequence-modeling benchmarks (including time-series and language); it is not a prompt-injection result. Causal and bidirectional variants change lookahead and buffering. |
| `depthwise` | Text-specific [Squeezed VDCNN](https://arxiv.org/abs/1901.09821); no author implementation was confirmed in this focused check. | Factor temporal filtering by channel from pointwise channel mixing. It reduces arithmetic under the formula below but may leave channel parallelism, weight layout, and memory access as the limiting factors. |
| `grouped` | Grouped transformations in [ResNeXt](https://openaccess.thecvf.com/content_cvpr_2017/html/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.html); [authors' code](https://github.com/facebookresearch/ResNeXt). | Groups restrict cross-channel mixing and lower dense-convolution MACs by the group count. This is a vision architecture analogy; text-specific accuracy and FPGA efficiency are unestablished. |
| `large_kernel` | Text CNN kernel-width analysis in [Kim](https://aclanthology.org/D14-1181/) and the practitioner's [sensitivity study](https://aclanthology.org/I17-1026/). | Contiguous larger context increases both MACs and input delay span. Compare to dilation at similar receptive field, rather than using a kernel-size sweep as a separate family claim. |
| `gated` | [Gated Convolutional Networks for Language Modeling](https://arxiv.org/abs/1612.08083); implementation is included in [Meta/Facebook fairseq](https://github.com/facebookresearch/fairseq). | A learned gate modulates temporal features and doubles pre-gate channel output in the registry's GLU-like blocks. Language-model results motivate the operation, not security benefit. Count sigmoid/multiply and gate activations separately from MACs. |
| `inception` | Split-transform-merge is established in vision; its directly relevant text antecedent is Kim's [parallel-kernel TextCNN](https://aclanthology.org/D14-1181/). | This registry concatenates several kernel branches and fuses them before pooling. The meaningful contrast is fusion depth and branch width under a common capacity budget; there is no prompt-injection-specific Inception result. |
| `local_context` | A design combination of local text filters and contextual blocks; related long-context motivation in [DPCNN](https://aclanthology.org/P17-1052/). | Contextualizes local evidence before final pooling. Treat as a project hypothesis; compare to a capacity-matched plain stack and remove/replace the local/context path before claiming a mechanism. |
| `dense` | Dense feature reuse; [Huang et al., CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html) and [authors' code](https://github.com/liuzhuang13/DenseNet). | Each layer consumes accumulated prior channels. Dense concatenation may improve reuse while raising activation traffic and pointwise compression work. The cited benchmarks are vision classification. |
| `pooling` | [Dynamic k-max pooling for sentence models](https://aclanthology.org/P14-1062/). | Compare max, max-plus-mean, top-k, and learned attention summaries. Top-k needs ranking/selection and possibly position retention; attention adds score, softmax, and weighted reduction work. |
| `byte_hierarchy` | [Character-level CNN](https://arxiv.org/abs/1509.01626) and [VDCNN](https://aclanthology.org/E17-1104/); original character-CNN [author code](https://github.com/zhangxiangxiao/Crepe). | The registry uses byte inputs and hierarchical downsampling, whereas these precedents use character sequences. UTF-8 bytes are a separate representation; report exact byte lengths and complete tails, not token-normalized costs. |

The literature check found no direct prompt-injection result showing that one of these CNN topologies is superior. The attack-specific evidence is separate: [BIPIA](https://xyq7.github.io/papers/BIPIA.pdf) benchmarks indirect injections in external content and evaluates attack success; [PIGuard](https://aclanthology.org/2025.acl-long.1468/) proposes an injection guard and studies over-defense; Meta's [Prompt Guard model card](https://huggingface.co/meta-llama/Prompt-Guard-86M) describes an injection/jailbreak text classifier. They motivate detector and benign-content evaluation, but do not validate these 16 CNNs or establish that a detector's label blocks an attack or preserves a legitimate task. The SENTRY R1 development errors and this architecture search remain development evidence, with final partitions closed.

The source check covered primary papers and accessible official/reference code where available. In particular, Kim, TCN, gated CNN, lightweight/dynamic convolution and DenseNet/ResNeXt have author/project-hosted implementations. For DPCNN, a community implementation was found; for VDCNN the focused search found accessible implementations but did not verify one as author-official. The closest direct architectural overlap is established text classification/sequence labeling, not indirect injection. No performance numbers from external datasets are transferred to SENTRY.

## Configuration and analytical profile audit

All 96 configs were profiled as actual batch-one FP32 forwards on CPU with Torch 2.9.1 and one intra-op/inter-op thread. Token models used vocabulary 30,522 and 256 positions; byte hierarchy used vocabulary 257 and 1,024 positions. Every position was marked valid. `model_cost` and `cnn_cost.profile_cnn` agreed exactly on unique parameters and Conv1d/Linear MACs; the independent pure-Python structural counter also matched all 96. The detailed per-config values, embedding bytes/gather payload, largest module output, layer shapes, receptive fields, delay-buffer assumptions and complete-document calculations are in `results/cnn_study_v1/cost_registry.json`.

| Family | Parameters, M | Counted MACs/window, M | Local receptive field |
|---|---:|---:|---:|
| `textcnn` | 1.99–4.33 | 9.44–124.78 | 4–15 tokens |
| `stacked` | 1.98–4.71 | 7.34–205.52 | 5–39 tokens |
| `residual` | 1.98–4.75 | 7.34–213.91 | 5–41 tokens |
| `dpcnn` | 2.01–4.62 | 10.49–54.13 | 16–698 tokens |
| `dilated` | 1.98–4.42 | 7.34–130.02 | 7–125 tokens |
| `idcnn` | 1.99–4.12 | 19.92–205.52 | 17–121 tokens |
| `tcn` | 1.98–4.42 | 7.34–204.47 | 7–127 tokens |
| `depthwise` | 1.97–4.10 | 3.24–48.43 | 5–113 tokens |
| `grouped` | 1.96–4.09 | 2.62–46.14 | 5–37 tokens |
| `large_kernel` | 2.03–14.25 | 19.92–2,646.61 | 17–621 tokens |
| `gated` | 2.01–6.22 | 13.63–591.40 | 5–133 tokens |
| `inception` | 2.01–6.88 | 13.63–759.17 | 5–101 tokens |
| `local_context` | 2.01–5.37 | 13.63–373.29 | 9–121 tokens |
| `dense` | 1.97–4.34 | 4.39–109.31 | 5–33 tokens |
| `pooling` | 2.01–2.01 | 13.63–13.65 | 9 tokens |
| `byte_hierarchy` | 0.07–1.22 | 41.94–408.94 | 14–662 bytes |

Parameters include the embedding table and learned normalization/head weights. MACs count Conv1d/Linear products only; the table uses 256 token positions or 1,024 byte positions per window. RF for byte hierarchy is measured in bytes. The broad ranges reflect config size/depth/context differences, not measured throughput or FPGA resource costs. Delay spans are per-layer analytical bounds and are not simultaneous live activation storage. For complete padded-window chains, the prepared inner-selection token partition has 4,860 documents and 6,055 windows (maximum 12 windows/document); byte has 4,860 documents and 6,828 windows (maximum 47). Inner-calibration manifests record 4,854 documents, 6,090 token windows (maximum 25), and 6,912 byte windows (maximum 47). The JSON multiplies each config's measured per-window MAC and logical embedding-gather payload by these saved window totals, and records mean and maximum per-document costs. It leaves static parameter storage counted once. These counts come from the prepared manifests; final/test and NotInject partitions were not opened.

The source inspection found a few important search-label distinctions. In `textcnn`, the configured `normalization`, `depth`, and `dilations` do not affect the model forward path; `topk` is read only for `topk` and `topk_mean` pooling. Other family-specific fields (`groups`, `growth`, `repeats`, `direction`, and local-context controls) are only used by their corresponding branches. Dropout and optimizer settings can change training but do not change evaluation-time graph topology. The registry contains two exact evaluation-forward duplicates across labels: `dilated_02`/`tcn_02` and `dilated_06`/`tcn_06`. Several more share parameter layouts while differing in skip connections, padding, or pooling: the full groups are recorded in the JSON. These are config/topology observations only; they say nothing about equality of trained weights or task performance.

## Analytical cost equations

Let a convolution receive `L` positions with `Cin` channels and emit `Cout`, use kernel `K`, dilation `d`, stride `s`, and groups `g`. Its effective span is `E = 1 + (K - 1)d`; its parameter count is `K Cin Cout / g` (plus `Cout` bias parameters where used); and its MAC count is `Lout Cout (Cin/g) K`, where `Lout` follows the actual padding/stride rule. For same padding at stride one this is `L Cout (Cin/g) K`. MAC totals omit bias additions, activations, normalization, pooling, masking, indexing, nonlinearities, and transport unless listed separately.

For an ordered temporal path, the receptive field and token jump are `R0=1, J0=1`; each convolution/pool updates `Rnext = R + (E - 1)J` and `Jnext = J s`. Centered odd kernels add `(E - 1)J/2` lookahead positions at that layer. A causal path has zero future lookahead and must retain its left-context state. For a branch, calculate each path and report its maximum receptive field/lookahead; for DPCNN and byte hierarchy include pooling windows and strides. Global pooling summarizes all valid positions, so report local receptive field and global document aggregation as distinct facts.

A straightforward streamed convolution needs an input delay span of approximately `(E - 1) Cin q` bytes per independent stream, where `q` is activation bytes. Dilation changes this span even when it leaves MAC count unchanged. Centered operations also add lookahead latency. This is a logical ring/delay estimate, not a BRAM count: line-buffer banking, parallel lanes, tile reuse, padding, and output buffering alter the physical design. A fully materialized path instead stores feature tensors of `B L C q` bytes. Residual joins need aligned skip values; dense concatenation and multi-branch fusion can require additional live feature maps. A time-multiplexed engine trades replicated arithmetic for weight reloads, intermediate storage, and extra scheduling cycles.

For a vocabulary embedding of size `V` by `D`, full table storage is `V D q_w` bytes; each token requests one `D`-element row. The logical gather payload per window is `L D q_w`, but this is not necessarily external memory traffic because caching, row reuse, quantization scales, and placement are unspecified. For byte models use actual UTF-8 byte count `Bdoc` and its own vocabulary/embedding size. Do not compare MAC/token without converting both models back to the same raw documents.

For fixed content windows of width `W` and overlap `O`, the number of windows for `M` content units is `n = max(1, 1 + ceil(max(0, M-W)/(W-O)))`. The token protocol uses `L=256` total positions, including CLS/SEP, so `W=254`, `O=64`, and stride is 190. The byte protocol uses `L=W=1024`, `O=256`, and stride 768, with no special tokens. Per-document convolution and gather work is summed over all prepared full windows, including the final padded tail; static parameter storage is counted once. Max-window risk aggregation needs `n-1` scalar comparisons after scoring. Use actual prepared tensor shapes for model profiles and prepared content units only for the outer document window formula.

`scripts/cnn_cost.py` exposes `estimate_conv1d`, `conv_path_summary`, `registry_receptive_field`, `document_window_count`, `profile_cnn`, and `scale_window_profile`. The profile hooks count registered Conv1d/Linear modules on one actual-shaped forward, capture embedding table and logical gather payload, parameter storage, and module outputs. `registry_receptive_field` maps the current 16-family config to local RF/lookahead; it must be revised if the topology changes. It does not replace the registry's `model_cost` helper or claim peak activation memory. Functional masking, layer norm, GLU, pooling, attention, tokenization, and aggregation need separate operation accounting. No FPGA resources or latency are estimated.

## Worked reference cost: the saved R1 TextCNN

This calculation uses only the saved architecture above and padded sequence sizes, with FP32 embeddings/activations/weights. All three convolution branches retain every padded position and 64 output channels. The head contributes 384 MACs and 386 parameters. A 30,522 by 64 embedding table contributes 1,953,408 parameters.

| Input positions `L` | Conv + head MACs per window | Embedding logical row bytes/window (FP32) | Embedding activation bytes | Three branch outputs, total FP32 |
|---:|---:|---:|---:|---:|
| 128 | 7,864,704 | 32,768 | 32,768 | 98,304 |
| 256 | 15,729,024 | 65,536 | 65,536 | 196,608 |
| 512 | 31,457,664 | 131,072 | 131,072 | 393,216 |

The per-window conv total is `L * 64 * 64 * (3+5+7) + 384`. The total model has 2,015,426 parameters, 8,061,704 bytes at FP32 or 2,015,426 ideal payload bytes at INT8; the embedding table alone is 7,813,632 FP32 bytes (96.9% of parameters). Ideal INT8 bytes omit scales, alignment, calibration, arithmetic, and accuracy effects. For any input length, the maximum local receptive field is 7 tokens and centered lookahead is 3; individual branches look ahead by 1, 2, and 3 tokens. A shared embedded-input ring spanning the widest width needs six 64-channel vectors (1,536 bytes FP32 or 384 bytes at one byte/activation), followed by 192 pooled max accumulators (768 bytes FP32). This assumes one embedded stream fans out to all branches and serial channel accumulation; it is not a synthesized buffer allocation. Embedding reads, input packing, document-window repetition, branch scheduling, pooling comparisons, and host decision remain part of full-document cost.

At the recorded 256-total-position window (254 content-token capacity) and 64-token overlap, complete-document MACs are `15,729,024 * n`. Thus a one-window document costs 15.729M counted MACs, a two-window document 31.458M, and a 14-window document 220.206M, before excluded operations. Longer documents incur repeated embeddings and convolution even if each window has the same fixed input shape.

## Newer prior work and detector-specific competing explanations

[RepLKNet (CVPR 2022)](https://arxiv.org/abs/2203.06717) studies large depthwise
kernels and re-parameterization in vision. The registry's contiguous temporal
kernel sweep has neither its re-parameterization nor a demonstrated text-task
advantage. [MobileNetV2](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html)
uses expansion, depthwise filtering and a linear bottleneck;
[Blueprint Separable Convolutions (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Haase_Rethinking_Depthwise_Separable_Convolutions_How_Intra-Kernel_Correlations_Lead_to_Improved_CVPR_2020_paper.html)
factorizes intra-kernel/channel correlations. The tested standard temporal
depthwise/pointwise blocks implement neither complete design. These are
established vision structures, not detector-quality or VEK280 evidence. Official
CVF source records were verified; direct CVF fetches returned 403 in this review.

[PIGuard (ACL 2025)](https://aclanthology.org/2025.acl-long.1468/), with
[author implementation](https://github.com/leolee99/PIGuard), directly studies
prompt-guard false alarms associated with attack-related trigger words and a
training strategy to reduce that bias. Its contribution concerns detector
training/data, rather than a comparison of these CNN families. This is a closer
competing explanation for hard-benign failures than simply insufficient CNN
context. Its published NotInject benchmark was discussed as prior work only;
the protected local set remains closed. No published quality result is imported
as a SENTRY result. The focused primary-source search did not identify a direct
prompt-injection comparison of the tested CNN topologies; that limited search
is not a claim that none exists.

## Preliminary FPGA mapping by representative style

The target here is the repository's recorded offline VEK280 design: installed board definition `xilinx.com:vek280:part0:1.2` selects `xcve2802-vsvh1760-2MP-e-S`; actual Vivado 2025.2 catalog inspection found CIPS, AXI NoC and SmartConnect. The selected implementation uses a standalone Cortex-A72 supervisor in OCM and the existing PL detector. PL masters and PS mappings reach DDR_LOW0 through the NoC; the A72 submits work through a separate AXI-Lite mailbox. The generated host protocol currently sends 32-bit token IDs, 32-bit token types and byte prefix masks over UART at 115200 baud. Its exact L256 packet serialization is 211.11 ms before the separately modeled copy/cache/mailbox allowance. This is a real interface cost in the recorded design, not a CNN latency prediction. Byte-input CNNs do not fit that packet contract without a host/descriptor change.

The target's catalog lists 1,312 DSPs, 520,704 LUTs, 1,041,408 flip-flops, 600 BRAM36 blocks and 264 URAMs; the engineering model sets a 70% design budget. These are catalog capacities and a declared reserve, not free CNN resources. Existing routed BERT-system closures demonstrate that a particular implementation can meet timing and budgets at recorded profiles: for example, L256 lanes64 uses 67,785 LUT / 61,450 FF / 103 BRAM36 / 111 DSP and has +0.057 ns setup slack at 5 ns; cached lanes16 and lanes32 also have inspected closures. L128 lanes64 has a separate closure. These figures belong to those BERT implementations and cannot be allocated to or extrapolated as CNN utilization. No physical-board bandwidth, latency, power, or CNN result is established. The reports' NoC bandwidth records are compiler feasibility, not measured throughput. See `reports/goal4_platform_architecture.md`, `reports/fpga_toolchain.md`, and `hardware/vek280_full_check_model.json`.

| CNN style | VEK280 route worth comparing if a finalist emerges | Costs and interface details that must stay visible |
|---|---|---|
| Shallow token-preserving TextCNN / parallel kernels | PL temporal MAC engine after token embedding gather; either replicate kernel branches or time-share one engine. The existing platform has connected embedding-sum service, but its BERT table/layout is not evidence for CNN table placement or access rate. | Carry embedding table bytes, gather traffic/cache assumption, kernel fan-out, centered-kernel lookahead, activation/ReLU, masking and global max/top-k comparisons. Host still pays the serialized token packet boundary and one response/decision per queued item. |
| Stacked, residual, gated or local-context CNN | PL block pipeline or serialized engine with DDR/BRAM/URAM activation tiles and explicit skip/context paths. | Carry each layer's delay span and cumulative RF, residual retention/adds, LayerNorm reductions and scale/bias, gate nonlinear/multiply, mask work, intermediate writes/reads and branch joins. These operations are absent from Conv/Linear MAC totals. Centered layers require lookahead; a causal option instead retains left history. |
| DPCNN / byte hierarchy | PL pyramid stages can reduce later-position work after pool/downsample. A byte model needs a distinct ingress/embedding contract; the selected host currently emits token tensors. | Track pool3/pair-pool comparisons, mask/tail alignment, changed temporal jump and RF, activation retention between scales, and complete window count. Prepared byte selection has 6,828 windows versus 6,055 token windows for the same 4,860 documents (byte width 1,024/stride 768); this raises total pipeline work even when downsampling helps later stages. Do not reuse token packet cost or token MAC/window as a byte estimate. |
| Dilated / IDCNN / causal or bidirectional TCN | PL programmable/fixed tap engine; optionally retain recurrent history between chunks only if the outer document/window protocol is changed and validated. | Buffer extent follows effective kernel span, not number of taps. Repeated IDCNN layers still consume cycles even with shared weights. Causal uses history state and no future lookahead; centered requires future samples; bidirectional requires a reverse pass or stored sequence and separate directional weights. Preserve complete-document max across the actual overlapping windows. |
| Depthwise / grouped | PL depthwise channel lanes followed by pointwise mixing, with group-aware channel banking. A heterogeneous A72/PL split is also a candidate for gather or postprocessing only after transfer overhead is charged. | Compare channel utilization and data rearrangement against the saved MAC reduction. Include pointwise MACs, norm/gate/pool operations, kernel-span buffers, activation staging and NoC/DDR accesses; do not infer speedup from MACs alone. |

The existing A72/PL split is a recorded platform architecture, not evidence that CNN layers should move to the host: the current supervisor performs packet validation/copy/cache maintenance and mailbox submission, while detector arithmetic remains in PL. VEK280 AIE mapping is not an exercised route in these records. `reports/fpga_toolchain.md` says AIE compilation/XRT is unnecessary for its scoped integer-kernel build; it does not establish AIE availability, suitability, or absence. An AIE option therefore remains conditional on target/tool-flow verification, kernel support for gather/normalization/gating/pooling, numerical contract, and PL/A72 handoff cost. Do not assign it resource or performance figures now.

An optional CNN synthesis/build is not decision-useful yet: the live search has not selected a finalist, and no CNN embedding placement, common fixed-point/quantization chain, supported normalization/gate/pooling realization, or byte packet format has been qualified. A build now would measure an arbitrary mapping rather than compare finalists under a common contract. Once software selects a candidate and freezes these interfaces, one representative PL mapping can answer a bounded resource/timing question; any AIE or split route should be a separate, explicitly charged comparison. This is a scope/qualification stopping point, not a claim that the toolchain or target is unavailable. Existing streaming CNN line-buffer literature, such as [FINN](https://arxiv.org/abs/1612.07119) and the [AoCStream study](https://arxiv.org/abs/2212.11438), establishes implementation patterns only; its reported resources and performance do not transfer to this target or detector.

## Evidence boundary after the completed architecture study

This literature/cost note does not itself establish detector quality or hardware
performance. The completed [principal study](cnn_study.md) now records adequate
inner training, frozen three-seed confirmation, hard-benign/probe errors,
batch-one complete-document CUDA costs and emulated 8-bit sensitivity. No CNN
meets the primary empirical screen in all three seeds, and independent natural/
withheld-family evidence remains unresolved. The 16 branches are conventional
candidates. A defensible FPGA-aware method still requires a later method-level
difference, closest-prior comparison, controlled ablation and evidence that the
mechanism improves a prospective security/cost trade-off with hardware realization.
