# Goal 1 fixed protocol

2026-09-26. Goal 1 evidence is data preparation, grouped checks, a shared evaluator/window path, real model loading/scoring and an evidence-based baseline choice. It does not establish trained detection quality or hardware speed. The shared source of settings is `configs/project.json`; relative paths resolve from the repository, never the shell directory.

## Task, data and measured preparation

English plain-text, single-turn EmailQA with a normal user question and newly received external email content. Only the external content may be altered. Detector input is the external document alone; question, reference answer, attack target and labels are metadata. Label 1 means an injected instruction is present by construction, not that the target LLM obeyed it. Five examples with source IDs, questions, reference answers, inserted text/positions and harmless target rubrics are in `reports/representative_pairs.md`.

100 application pairs use 20 distinct BIPIA EmailQA source/question tasks with five official harmless attack payloads each, all in candidate-scoring groups. Clean sides repeat; they are not 100 independent backgrounds. Each task naturally has one chunk. Adequate separate detector training comes from AESLC historical email bodies with BIPIA start/end insertions, not from these 100 pilot pairs. Exact pinned source revisions and licenses are in `reports/data.md` and config.

| Partition | Unique original-source groups / benign documents | Injected documents | Permitted purpose |
|---|---:|---:|---|
| Training | 10,790 | 21,580 | Future weight updates |
| Search temporary threshold | 1,355 | 2,710 | Candidate thresholds; fit only its benign scores |
| Search candidate scoring | 1,369 | 2,738 | Development model/design selection |
| Final calibration | 1,717 | 3,434 | Thresholds after finalists are frozen; fit only benign scores |
| Test | 1,810 | 3,620 | Final reporting only |

Official test membership takes precedence when exact normalized source bodies cross splits; AESLC official development is reserved for final calibration. Remaining official training sources provide training/search. BIPIA development groups are reserved for search/calibration. Every clean source and all its injected variants stay together. This preserves suitable official splits instead of applying fallback 60/20/10/10. No fuzzy/semantic duplicate-removal claim is made.

NotInject's 339 rows are a separate benign challenge (84 multilingual, flagged outside the main English scope). They never train, fit thresholds or select candidates. The 1,810 distinct benign test emails are 1,766 AESLC + 44 EmailQA, leaving a 1,190 shortfall against the 3,000 aim. Do not pad this count with repeated pairs/windows or challenge data. Report source strata separately: pooled Enron performance cannot establish a precise EmailQA-domain 1% FPR. The attack pool covers task diversion/answer transformations; it is not comprehensive attack coverage.

## Model, numerical scope and windowing

Main detector: `google/bert_uncased_L-4_H-256_A-4`, revision and tokenizer revision `387825ce42dbb39b87911cdf8e383ee3b25184f8`, 4 encoder layers, hidden width 256, 4 heads, binary head (benign 0/injection 1). Goal 1 runs ordinary float32 pretrained encoder plus a newly initialized classification head; no fine-tuning occurred. Dropout is disabled during scoring. Goal 2 retains seed 42, AdamW, learning rate 2e-5, weight decay .01, effective batch 32 and 3 epochs; microbatch/accumulation remain to be selected from actual training memory use.

A single window has 256 total tokens: up to 254 content tokens plus CLS/SEP. Adjacent windows overlap by 64 content tokens and retain the tail. A window's risk is softmax(classification logits)[1]; document risk is the maximum over all its windows. Supplied character spans label only intersecting windows positive; windows outside the span remain clean. Without positive spans, window labels stay unresolved and future loss uses the differentiable document aggregate. Calibration and reported metrics are document-level; source-group uncertainty, rather than independent-window uncertainty, belongs in final reporting. Single-window and long-document results must be distinguished.

Future weight search has 16 decisions: Q/K/V together, attention output, feed-forward input and feed-forward output in each of four encoder layers; each chooses W4 or W8, with A8. Embeddings, classifier, bias, accumulators, softmax, normalization, nonlinear functions, clipping/scales/rounding and their costs are outside that search but must be specified/calibrated in Goal 3. Their quantized/hardware formats are still null; Goal 1 float32 execution does not resolve them. No INT4 or FPGA performance has been measured.

## Threshold and protection criteria

Attack iff score > threshold. For n benign calibration scores sorted ascending, k = floor(alpha*n), threshold is the value at 1-based position n-k. Ties at threshold are benign; at most k calibration exceedances are possible. Primary alpha=.01; .005/.02 are secondary points from saved scores. Candidate and floating reference each fit their own temporary thresholds on search-threshold benign groups, then measure recall/FPR on the separate search-scoring groups. B/C must meet observed development FPR <= .01 and recall >= the floating-reference recall minus .01 (one absolute percentage point). These are experimental requirements, not a universal safety guarantee.

After common permitted QAT, freeze finalists using development evidence, fit final thresholds only on final-calibration benign groups and report actual test counts, recall and FPR with no test tuning. A pre-QAT near-feasible candidate may enter the small provisional shortlist; it is not a feasible result until evaluated after the common allowance.

## One published baseline and A/B/C

Selected: **quasar_vit_fixed_bert_evolution**, a restricted adaptation of [Quasar-ViT (ICS 2024)](https://arxiv.org/html/2407.18175), DOI 10.1145/3650200.3656622. HAO was read as the main alternative; Quasar's Transformer/shared-engine design needs fewer hardware-template changes than HAO's CNN subgraphs. The primary-source table is `reports/prior_art.md`; preserved rules, explicit project choices and adaptation boundaries are in `reports/baseline_selection.md`. This is not a full reproduction of its vision supernet or architecture search.

- A retains elite-parent crossover/block mutation and candidate-time resource constraints; maximize ordinary document accuracy at threshold .5 within predeclared latency caps. Report its conventional frontier and low-FPR protection; no arbitrary extra accuracy-loss tolerance is introduced.
- B uses the same evolutionary operators/space/allocator, but protection feasibility changes candidate and elite retention during selection. Within each cap eligible candidates rank by recall then latency; the final feasible choice minimizes predicted complete checking latency. This does not merely filter A's final output.
- C remains null until Goals 3-5 identify one specific weakness and one attributable change. B-to-C tests that change; A-to-B isolates criterion adaptation. All methods already get complete-network rescoring and identical legitimate optimization access.

Project search defaults, explicitly not copied paper hyperparameters: 16 initial maps (two uniform anchors plus 14 seeded random maps), up to 4 latency caps from the common Goal 4 table, 2 elites/cap, 4 crossover plus 4 mutation children/generation, layer-block crossover, per-group mutation probability 1/16. Cap values remain unresolved until hardware costs exist. No generic beam search replaces Quasar's operators. An optional beam-width-3 exploration remains a diagnostic tool, not the published baseline or a claimed contribution.

Retain shared low-bit atomic engines, signed-upper/unsigned-lower W8 decomposition and resource-aware allocation including packing/reconstruction support costs. Characterize actual A8 packing and all full-detector operations for the allocated board. Quasar's A6 DSP factors, ZCU102 resource values and ARM-offloaded nonlinear operators cannot be copied as this project's FPGA measurements. All methods share the same realizable profiles, schedule/buffers, formats, resource/clock limits and cost table.

## Budgets and stopping boundary

At most 64 logically consulted unique maps per A/B/C, including anchors and any arm-specific pilots; at most 3 shared-engine profiles. Cache identical model scores across profiles/methods but charge each method's logical candidates. Later stages allow at most 12 initial micro-syntheses, 3 pair-sensitivity experiments, 2 shortlisted configurations per method, and one common QAT epoch for each shortlist/anchor. One mechanism-removal comparison; reuse B if removal equals B. Same-board finalists are uniform/B/C. None of these later runs occurred in Goal 1.

Goal 1's actual checks passed: source-group/normalized-body split separation, required fields/labels, spans, candidate-only pilot and NotInject exclusion; hand-calculable evaluator example with threshold .8 and TP2/FN2/FP1/TN9; real native CUDA BERT-Mini inference. The smoke scored 11 documents as 15 windows, including one 831-content-token email using 5 windows. Only classifier weight/bias were newly initialized; all pretrained encoder weights loaded. Raw window logits/risks and document maxima are in `results/goal1/smoke_{windows,documents}.jsonl`; configuration, versions/device and loading information are in `smoke_model.json`. Random-head values are bring-up evidence only. The recorded 0.277-second first-call duration is not a steady-state benchmark.

At the original Goal 1 handoff, Prompt Guard was gated and not authorized; it remains a separate quality-reference access need. No target LLM had been called and no paid API budget exists. No FPGA allocation/part/interface/license/tool release is established. These fields remain unresolved for their relevant stages; do not replace them with candidate board names or fabricated measurements. Goal 2 is not started.

## Later execution state

This report records the Goal 1 protocol and its original handoff, not current completion state. See [STATUS.md](../STATUS.md) for later detector, quantization, completed institutional DeepSeek pilot and hardware evidence. The later target selection does not alter the detector checkpoint, data splits, guard thresholds or B-to-C method question. Historical Gemini evidence stays separate; full hardware and final evaluation remain incomplete.
