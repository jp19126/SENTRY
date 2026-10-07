# Prior art and claim boundaries

Reviewed 2026-09-26 for Goal 1. Selected baseline:
**quasar_vit_fixed_bert_evolution**. A/B are its two criterion variants;
C remains null until development evidence supports one concrete change.

| Verified primary source | Established mechanism | Project boundary |
|---|---|---|
| [HAO (2021), Sections III-A/B](https://arxiv.org/pdf/2104.12766) | Hardware integer programming minimizes CNN-subgraph latency under DSP/LUT/BRAM limits. Quantization minimizes additive Hessian-trace-weighted squared weight error under a latency cap. MCTS explores architectures; SVR predicts their accuracy. | Joint precision/hardware optimization is established. Fixed BERT removes architecture MCTS/SVR and requires replacing the CNN template. Read, not selected. |
| [Quasar-ViT (ICS 2024), Sections 3.4-3.5 and 4](https://arxiv.org/html/2407.18175) | Elite-parent crossover/mutation under hardware constraints; reusable low-bit units, signed/unsigned W8 decomposition, DSP/LUT allocation and tiled execution. Also trains vision supernets and searches architectures/rowwise precision ratios. | Selected restricted adaptation. A6 packing and ZCU102 resource values cannot be transferred to A8 or an unresolved FPGA. Original CPU nonlinear execution must be adapted to the full-detector FPGA requirement. DOI 10.1145/3650200.3656622. |
| [Q-BERT, Section 3](https://arxiv.org/pdf/1909.05840) | Sensitivity uses absolute mean plus standard deviation of Hessian top eigenvalues; groupwise quantization and QAT follow. | BERT sensitivity allocation exists; this is not a complete FPGA allocation method. |
| [InfoQ (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/39039), [method](https://arxiv.org/html/2508.04753v1) | Downstream changes in sliced mutual information after individual-layer quantization supply ILP sensitivity costs under size/BitOps budgets. | Global/downstream sensitivity is established; full-network rescoring alone is not C's novelty. Published March 14, 2026; DOI 10.1609/aaai.v40i24.39039. |
| [Critical Weight Protection (Findings ACL 2026), Section 3](https://aclanthology.org/2026.findings-acl.993.pdf) | Squared-gradient sensitivity differences separate fairness/safety-critical weights from general capability; high-scoring weights retain original precision. | Safety-sensitive precision protection exists. Generative-model trustworthiness differs from detector recall at 1% benign FPR. DOI 10.18653/v1/2026.findings-acl.993. |
| [MixQuant (July 2026 preprint), Section 4](https://arxiv.org/html/2607.23047v1) | Averages distortion across random quantized upstream contexts, calibrates quantizer parameters on allocator-generated plans, then greedily buys distortion-reducing upgrades with a minimum-bit escape preference. | Context-dependent sensitivity/inter-layer effects exist. Its memory objective differs from complete FPGA checking latency. Verified publication status: arXiv preprint. |
| [FILM-QNN (FPGA 2022), Section 3](https://www.sfu.ca/~zhenman/files/C23-FPGA2022-FILM-QNN.pdf) | High precision goes to filters with largest low-bit output-feature error at a fixed intra-layer ratio during QAT. Execution includes DSP packing, weight reordering and data packing. | Packing/layout optimization is established. Storage reduction alone is not a latency result. |
| [He et al. (Scientific Reports, May 2026), Method/Experiments](https://www.nature.com/articles/s41598-026-53062-w) | Hessian sensitivity, binary 4/8-bit masks and ILP under size, BitOps or profiled latency limits. | Recent competing evidence that sensitivity plus hardware-aware allocation is established. Reported latency profiling uses RTX 2080 Ti, not this project's FPGA. |

Additional relevant lead: Byun, Woo and Mukhopadhyay, *FPGA Acceleration With
Hessian-Based Comprehensive Intra-Layer Mixed-Precision Quantization for
Transformer Models*, IEEE Access 13, 70282-70297 (2025),
[DOI](https://doi.org/10.1109/ACCESS.2025.3563196).
Search exposed metadata and author-uploaded excerpts; publisher/institutional
full-text retrieval failed. Detailed method verification remains incomplete.

All eight table entries had accessible primary method text. Academic-search MCP
tools were unavailable, so primary arXiv, publisher and author-hosted sources
were read through web fallback. HAO HTML failed; its PDF succeeded. This bounded
check establishes overlap, not exhaustive novelty or reproduced performance.
A-to-B measures criterion adaptation; B-to-C is the later main method test.
