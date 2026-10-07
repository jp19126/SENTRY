# Baseline selection and bounded adapter

Goal 1 decision, 2026-09-26: **quasar_vit_fixed_bert_evolution**, a restricted
adaptation of [Quasar-ViT, ICS 2024](https://arxiv.org/html/2407.18175)
(DOI 10.1145/3650200.3656622), not a full paper reproduction.

## Rationale and preserved boundaries

Quasar supplies Transformer-oriented evolutionary selection and reusable FPGA
mixed-precision execution. HAO was read as the main alternative, but its CNN
subgraphs, architecture MCTS and SVR would require greater changes for fixed
BERT. Q-BERT and InfoQ establish relevant sensitivity methods without the same
complete FPGA allocation principles. This choice precedes comparative outcomes.

| Preserve from Quasar | Adapt for this project | Omit |
|---|---|---|
| Elite-parent crossover, block mutation and candidate-time constraints | Sixteen fixed BERT-Mini W4/W8 groups, A8, document scores and bounded evaluations | Vision architecture/depth/width/head search |
| Precision and hardware evaluated together; explicit DSP/LUT packing costs | At most three common realizable shared-engine profiles | Vision supernet training, weight entanglement, vision scaling and teacher distillation |
| Shared low-bit engine, higher-bit decomposition and tiled execution | Actual allocated FPGA/toolchain and complete detector cost | Copying A6 packing factors/ZCU102 numbers, or silently retaining CPU nonlinear execution |

The published rowwise mixed-ratio space is replaced by whole-group W4/W8
decisions including uniform anchors. This is a declared granularity/search-space
adaptation. All candidates use the same floating checkpoint and quantization
rules; only the common limited shortlist receives QAT.

## Exact bounded project defaults

These population settings are ours, not claimed paper hyperparameters. Implement
in Goal 5 after Goals 3-4 establish formats and the common hardware cost table.

1. Encode Q/K/V together, attention output, feed-forward input and feed-forward
   output per encoder layer: sixteen W4/W8 decisions, with A8 and all other
   declared formats common.
2. Seed 42; initial population sixteen unique maps: all-W8, all-W4 and fourteen
   uniform random maps. Consult all resource-feasible common engine profiles
   for each map through cheap cost-table lookups.
3. Freeze four latency caps spanning predicted all-W4-to-all-W8 complete-checking
   latency from the common Goal 4 table before comparative runs; merge coincident
   caps. Record values and units. Unresolved costs are not numerical caps.
4. Retain the two best eligible candidates per cap and union/deduplicate their
   elite pools. Empty eligible slots remain empty. A separate provisional
   archive may supply the permitted near-feasible QAT shortlist; it does not
   certify feasibility.
5. Generate four crossover and four mutation offspring per generation.
   Crossover picks two elite parents uniformly and independently inherits each
   encoder layer's four-decision block from either parent with probability 1/2.
   Mutation picks one elite parent uniformly and flips each group independently
   with probability 1/16; resample an unchanged mutant. If every eligible elite
   pool is empty, use unique uniform proposals within the same allowance and
   report the empty feasible population.
6. Reject physically unrealizable designs using the resource table, deduplicate
   proposals, and score the complete network for each newly consulted map.
   Stop at 64 logically consulted unique maps per arm, including anchors and
   arm-specific pilots. Six batches of eight after the initial sixteen use that
   allowance when all are unique; there is no independent search-depth cap.
7. Reuse identical scores/checkpoints when computation/data/windowing agree, but
   charge each method for maps it consults. Report logical maps, actual
   scoring/training cost and cheap hardware lookups separately. Ties use lower
   predicted latency, then lexicographic precision tuple.

These are evolutionary proposals, not beam search. Complete-network rescoring
is common to A/B/C and cannot later become C's claimed novelty.

## A/B/C and the candidate-selection hook

Let L(q,h) denote predicted complete checking latency for map q and realizable
profile h. Document risk is maximum window risk under the shared policy.

**A - existing method:** within each preregistered latency cap, maximize ordinary
document accuracy at fixed threshold 0.5 on search-scoring groups. Report its
accuracy-latency frontier and low-FPR protection results. Do not introduce an
arbitrary accuracy-loss allowance or tune a tolerance to make A lose. Cap values
come from the common Goal 4 table before comparative experiments.

**B - safety-matched existing method:** retain A's evolutionary operators,
hardware allocator, maps/profiles, formats, data, QAT allowance and budget.
For each candidate, fit a temporary threshold on search-threshold groups by
the shared 1% benign-FPR rule; assess FPR/recall on separate search-scoring groups.
Before elite retention require FPR <= 0.01 and recall >= floating-reference
recall - 0.01, using the same threshold-fitting/scoring separation for the
floating reference. Rank eligible candidates within a cap by recall, then
latency. Final B selection minimizes L(q,h) among protection-feasible candidates
after allowed QAT/development evaluation. Thus B changes the breeding population
during search, rather than filtering A's final output.

If no feasible candidate exists, report that result. Near-feasible QAT leads
remain provisional; pre-QAT rejection alone does not decide eventual
feasibility. Fit final thresholds only after finalists are frozen, using the
reserved final-calibration partition. Test data never select candidates.

**C - null/unresolved:** choose one evidence-supported allocation/selection or
execution change in Goals 3-5, preserving B's protection rule and common
resources. A-to-B isolates criterion adaptation; B-to-C is the main method test.
A null improvement is valid.

## Hardware allocation and invalid transfers

Retain the principle of maximizing parallel atomic compute units across DSP and
LUT implementations within real resource limits, including LUT support costs
for DSP packing/reconstruction. Include actual BRAM, buffer, bandwidth and
timing constraints. Generate at most three realizable shared-engine profiles,
equally available/optimized for A/B/C; consult eligible profiles for each map.
A profile describes one physical engine, packing mode, schedule and buffer
arrangement, not fictitious independent per-layer datapaths.

For signed W8 arithmetic preserve w = 16h + l, with h in [-8,7] and l in [0,15].
The upper nibble is signed and the lower nibble unsigned:
x*w = 16*(x*h) + x*l. Characterize actual A8 packing, accumulation, reconstruction
and conversion on the allocated device. Quasar's A6 packing factors and ZCU102
resource counts are not A8 measurements.

The original design executes LayerNorm/Softmax/GELU on ARM. Here the main
detector belongs on FPGA. Embeddings, attention, nonlinear/normalization work,
pooling, classifier, conversions, transfers and host-to-decision overhead must
enter the common implementation/cost model. CPU offload cannot remain hidden.

## Goal boundary

This report fixes baseline identity and adaptation rules, not engine dimensions,
scales, latency-cap values, trained quality or C. Later authorized goals and
actual hardware establish those. Goal 5 implements/runs the adapter; Goal 6
supplies same-board measurements. This literature task performed no method
search, QAT, synthesis, board measurements or Goal 2 training.
See [prior_art.md](prior_art.md) for overlap and source-access limitations.
