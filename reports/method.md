# Goal 5: bounded A/B adapter handoff

`scripts/search_precision.py` implements the bounded adapter, including provisional B parents, a frozen QAT shortlist and post-QAT selection. The September 26 correction closes a pre-campaign completeness gap against Goal 5 D8â€“9. It does not establish campaign readiness: complete checking costs/caps and the approved score arithmetic/execution identity remain unresolved. No comparison campaign, new model scoring or QAT ran. C remains null.

## Baseline and criterion hook

This is the fixed-BERT adaptation selected in `reports/baseline_selection.md`, not a reproduction of the full vision architecture search. Quasar-ViT section 3.5 uses elite-parent crossover with a block selected from either parent for each layer, mutation and candidate-time constraints. Section 4 provides shared low-bit arithmetic and resource-aware allocation principles. The selected source is [Quasar-ViT, ICS 2024](https://arxiv.org/html/2407.18175), DOI 10.1145/3650200.3656622. No A6 packing factor or published ZCU102 resource value is transferred into these costs.

The sixteen decisions are Q/K/V together, attention output, feed-forward input and feed-forward output in each of four encoder layers. Each group uses W4 or W8 with A8; other operations follow the common FP32 contract. Architecture, depth, width and head count are fixed. A and B share proposal operators, seed 42, score computation, profiles, caps and the allowance of at most 64 logically consulted unique maps per arm.

A ranks cap/resource-fitting candidates by document accuracy at threshold 0.5, then lower complete checking latency, then lexicographic precision tuple. B's strict candidates satisfy separately calibrated development FPR <= 0.01 and recall >= floating-reference recall - 0.01, then rank by lower latency, higher recall and precision tuple. Each map chooses the cheapest feasible common profile; equal costs use profile ID as a stable tie-break. This criterion affects parent retention inside `elite_pools()`, not just a final filter on A.

When a cap has no strict B candidate, B retains explicitly provisional parents from its physically feasible, cap-fitting archive. For FPR f, recall r and floating-reference recall r0, define e=max(0,f-0.01) and d=max(0,r0-0.01-r), both in probability points. The predeclared provisional rank is `(max(e,d), e+d, latency, -r, precision_tuple)`. This orders the least violating candidates without granting feasibility or introducing a numerical acceptance tolerance. Any existing strict pool suppresses provisional parents for that cap; provisional parents never displace strict ones. Missing/undefined rates cannot enter this rank.

The bounded proposal sequence remains:

```text
Require approved score arithmetic/execution and real common Goal 4 costs/caps.
Start each arm with seed 42, all-W8, all-W4 and 14 distinct random maps.
For each new map, consult every common physical profile and charge one logical map.
Reject profiles that fail resources, timing, bandwidth or map feasibility.
Require the matching complete-network PTQ score for each physically realizable map.
For each cap:
    A keeps at most two accuracy-first parents.
    B keeps at most two strict latency-first parents when any exist;
      otherwise it keeps at most two explicitly provisional least-violation parents.
Union and deduplicate parent maps across caps.
Draw four layer-block crossover children and four mutation children.
Mutation flips each group with probability 1/16 and resamples an unchanged child.
Deduplicate proposals; duplicate-only batches retry the same operators.
Use uniform random proposals only when no cap-fitting parent remains.
Stop within 64 logically consulted unique maps per arm.
```

The cap/resource requirements apply equally to strict and provisional candidates. Provisional fallback changes only B's necessary task-criterion integration. Neither it nor the QAT shortlist is C novelty or a relaxation of protection. All maps consulted by an arm count even if their scores were already cached by another arm. Duplicate map consults within the same arm do not count twice; physical-profile lookups are recorded separately. The same proposal operators and seed are retained, although the different parent criterion naturally changes later maps.

## Common QAT allowance and final selection

`qat_shortlist()` freezes at most `methods.max_shortlist_per_method` (no more than two) unique precision maps across all caps, using the largest frozen cap and each map's cheapest feasible profile. A uses its accuracy rank. B orders strict maps first by latency/recall, then fills any remaining slot with the provisional rank above. Shortlisting closes further proposals and new consults. A repeated request returns the same maps. Entries state `provisional`, pre-QAT protection eligibility, the common one-epoch allowance and whether the map is a uniform anchor.

A later runner may execute at most one QAT epoch from the same selected floating checkpoint for each shortlisted map, reusing an identical completed run. Uniform W8/W4 anchors have the common separate allowance; if an anchor is shortlisted, its existing matching run is reused rather than trained twice. The selector handles the arm's at-most-two shortlist; common anchor reporting outside that list remains separate. No training dispatcher or campaign/resume system is implemented here.

`post_qat_selection()` requires unique matching QAT scores from the frozen shortlist. It reports an incomplete state and selects nothing until every shortlisted map has evidence. A selects by its declared accuracy criterion and explicitly reports protection. B accepts only post-QAT candidates meeting the unchanged FPR and recall conditions, then minimizes complete checking latency. A provisional PTQ lead can become feasible only through its measured post-QAT score. If none passes, B has no feasible finalist; no provisional status is carried into final acceptance. Final calibration/test data are never loaded by the adapter.

## Score reuse and arithmetic identity

`methods.approved_score_identity` governs quantized candidates; `methods.approved_floating_score_identity` separately governs the ordinary floating recall reference. Each contains `arithmetic_identity` and `execution_identity`; all four fields remain null because numerical/backend acceptance is unresolved. This is a scientific readiness gate, not a request for additional user permission. `readiness()` requires both objects. Identity validation allows their explicitly distinct arithmetic names but requires the same approved execution identity, naming the intended common backend/version/settings. A missing approval or cross-backend approval pair fails before any reuse. Saved floating records must match the floating arithmetic identity, not the quantized one. A scheme name alone cannot distinguish historical SDPA from the ordered reference, or CPU from CUDA when they differ numerically.

New score records must record the matching two identity strings alongside the exact common floating `checkpoint`, `scheme_id`, `precision_map`, explicit `qat` stage, `windowing`, `evaluation`, `data`, the existing `calibration_file`, and `frozen_scale_policy` containing the original weight/activation-scale rules. The common calibration file must match the completed Goal 3 run. A QAT score additionally names `scored_checkpoint`; its saved `quantization.json` must agree on map, scheme, format, train-only calibration, one epoch, seed, original floating source and frozen-scale policy. The frozen weight policy remains associated with the selected floating checkpoint; this metadata validation does not independently inspect every weight tensor. Ordinary source paths and version fields are used; no hashes or manifests are introduced.

`load_score(..., qat=False)` loads PTQ search scores; `qat=True` explicitly requests the later QAT stage. Saved document maxima are used to refit the threshold only on benign search-threshold documents and evaluate the separate search-scoring documents. Recomputed threshold/protection/ordinary metrics must exactly equal the saved candidate record. The floating recall reference requires its separate approved floating identity and matching checkpoint/data/window/evaluation association before reuse. Both approvals must resolve and use the same execution backend; this does not claim the two arithmetic computations are identical. A caller constructing `EvolutionAdapter` supplies that reference identity explicitly.

Historical score files and the earlier `results/goal5/score_adapter_bringup.json` remain unchanged. The old bring-up predated the ordered reference and the new identity gate; it is not evidence that those scores are eligible for a future ordered campaign. Do not append a new identity label to old SDPA scores. Existing QAT weights may be reusable, but scores from a different arithmetic/backend computation require a separately authorized matching evaluation.

## Existing evidence and bounded adapter check

Historical software development summaries are:

| Existing PTQ anchor | Accuracy at 0.5 | TP / attacks | FP / benign | Strict B protection |
|---|---:|---:|---:|---|
| W8A8 | 0.99926954 | 2,738 / 2,738 | 18 / 1,369 | Fails |
| W4A8 | 0.99926954 | 2,738 / 2,738 | 17 / 1,369 | Fails |

The later historical W8 QAT summary has 13/1,369 false positives and passes the rule; W4 QAT has 20/1,369 and fails. All retain attack recall 1.0. This demonstrates why PTQ failure alone cannot determine final QAT feasibility; it is not an A/B comparison or an ordered-backend result. The historical floating recall is 1.0, yielding a 0.99 recall floor in the diagnostic fixtures.

The single bounded check is:

```text
python -B scripts/search_precision.py --adapter-check
```

It passed with exit 0, using the four existing candidate metric summaries and floating recall only. Explicit in-memory latency sentinels test ordering/cap behavior; they are not measured or estimated hardware costs. The check verifies A's accuracy/tie-break order, B's provisional order, strict-parent precedence, two unique frozen shortlist maps, unchanged post-QAT protection decisions, and rejection of unresolved/missing/SDPA/backend-mismatched identities. The one affected rerun after separate floating-identity plumbing also rejects unresolved/wrong/missing floating identities and cross-backend approval pairs, while accepting distinct arithmetic names under one fixture execution identity. It reads no raw score rows, loads no model, constructs no CostTable, writes no results or config and consults zero campaign maps. Independent read-only review found no blocker in these changes. Details are in `reports/goal5_adapter_audit.md`.

`python scripts/search_precision.py` reports readiness only. The historical `--score-bringup` command now fails clearly while the approved identity is unresolved; it cannot silently overwrite the historical evidence with a relabelled reuse claim.

## Required hardware inputs and cost-table interface

Actual search still requires `methods.cost_table_path`, frozen `latency_cap_values` and `latency_cap_unit: "ms"`, actual allocation/tool fields, and the common active physical profiles. The confirmed board/tools and component evidence do not by themselves complete the integrated cost model. Unresolved values remain null; no illustrative cap or per-bit cost is used in the adapter.

The table must record `latency_unit: "ms"`, `latency_scope: "full_detector_host_to_decision"`, `detector_execution: "fpga"`, matching part/interface/clock/resource limits and primary length/batch, actual `source_reports`, an existing `validation_report` and matching `frozen_latency_caps_ms`. Each profile specifies its exact physical configuration, resources, timing and bandwidth feasibility. Each candidate row has the exact precision map and one entry for every profile, including infeasible ones. A feasible entry provides `estimated_complete_checking_ms`. Missing costs raise rather than being interpolated from an invented per-bit factor.

Estimated complete checking latency remains an estimate, not board timing. Embeddings, attention, normalization/nonlinear work, pooling/classifier, scales/conversions, buffers, transfers and host tokenization/transport/dispatch belong in the boundary. Kernel cycles or packed parameter bytes cannot substitute for that cost. The adapter never offloads fixed detector arithmetic to the CPU to claim FPGA feasibility.

## Proposed method boundary

No B-to-C mechanism is selected. Goal 3 error/sensitivity findings, actual Goal 4 costs and the bounded A/B campaign must support one concrete change and its nearest prior mechanism before C or its single removal comparison is defined. Protection-aware selection, complete-network rescoring, provisional leads and correcting the shared adapter are not claimed as new research contributions.
