# Interim research summary and claim boundary

26 September 2026. **Goal 8 is partial.** The interim manuscript is `paper/draft.md`. Numerical Goal 3 development experiments and the short supported software profile are complete. Hardware-grounded A/B/C comparison, final grouped evaluation and board measurement remain unexecuted. The selected institutional DeepSeek development pilot is complete under the fixed 256-token allowance. This does not complete hardware or final evaluation; the earlier Gemini observations remain a separate historical pilot.

## H1: quantization effects after refitting

**Answer within the observed development pool:** no quantization-induced attack-recall loss was found. Uniform PTQ/QAT anchors, 16 one-group interventions and two informative pairs retained 2,738/2,738 scoring attacks at the primary refitted threshold. All 16 single-group interventions also had zero additional misses at the common W8 threshold. The tested nearby uniform operating points retained recall 1.0.

False positives and score calibration changed. The original floating reference had 18/1,369 benign scoring errors (1.3148%). W4 PTQ had 31 at the floating threshold but 17 after refitting. After the common one-epoch QAT allowance, W8 had 13/1,369 = 0.9496% and W4 had 20/1,369 = 1.4609%. Only W8 QAT among these uniform primary-point anchors met the empirical ≤1% scoring-FPR condition. Equal calibration counts (13/1,355 for all five rows) are not equal scoring FPRs.

The experiment does not establish general attack robustness or the impossibility of W4 protection. Search source groups are disjoint from training, but inserted payloads share the development pool; only one training seed and one QAT epoch were used. The original FP32 model did not receive the additional QAT training epoch, so its comparison with W8 QAT does not isolate a quantization-only effect. Final calibration, test and NotInject remain unexamined for finalist evaluation.

## H2: improvement beyond the safety-matched existing method

**Untested.** The selected published-method adaptation is fixed as `quasar_vit_fixed_bert_evolution`. A retains ordinary accuracy at threshold 0.5. B keeps the same evolutionary proposals and hardware-allocation principles while applying FPR≤1% and recall≥FP−1 percentage point during candidate/elite eligibility. Both later require the common actual hardware cost table. A-to-B is task adaptation; B-to-C is the main proposed-method test.

Development results justify evaluating the task criterion directly. For example, two single-group maps share 99.9513% ordinary accuracy but produce 22 and 14 false positives after refitting. Across the 16 dependent interventions, descriptive Spearman correlations with FP count were 0.264 (ordinary error), 0.396 (BCE change) and −0.177 (logit MSE); recall ranks are undefined because every loss is zero. Actual CPU dynamic qint8 also has lower document BCE than FP32 but more false positives. None of these findings demonstrates a defect in B. Pair interaction, safety-sensitive precision and hardware-aware selection already have prior art. **C remains null.**

## H3: measured same-FPGA benefit

**Untested as a same-FPGA comparison.** All 12 initial HLS syntheses and all dispatched component RTL cases are complete. Active cached16/32 and static-bank64 profiles passed their targets and reserved cases twice; the final static64 reserved latency was 11,342,993 cycles in each repetition, matching the prior freeze. All 15 primary FP32 shapes passed twice, giving a weighted 411-call subtotal of 94,864,971 cycles. A representative FP32-only implementation met the 200 MHz objective with 4.932 ns routed period. These results do not establish integrated timing closure.

The fixed controller/mover passed 13 cases using actual control registers and explicit arithmetic stubs. Subtracting measured stub intervals gives a conditional 5,399,345-cycle control/layout term. Combined with the arithmetic evidence, static64's modeled internal path is 1,145.16874–1,215.94762 ms for uniform W4/W8 at 200 MHz. This is an analytical assembly under a declared nominal memory model, not measured full-detector execution. Physical DDR/NoC, host costs, numerical qualification and board measurements remain unresolved (`reports/hardware_cost_model.md`).

The real-input numerical bridge found an A8 rounding-boundary crossing from FP32 attention differences, despite exact same-input integer arithmetic. Ordered CPU bring-up reproduced C-simulation logits/risk on one input; it has not been approved as a synthesized-hardware reference. A separate ordered floating mode also ran once, without dataset rescoring. Historical software metrics remain historical. These boundaries prevent transferring the earlier protection counts to a claimed hardware result.

The quantized software checks establish only the declared arithmetic reference. A6 packing figures from Quasar-ViT do not establish A8 packing efficiency here. The main detector's fixed FP32 operations, embeddings, attention, memory movement, conversions and host-to-decision overhead require explicit accounting. The actual QAT state files contain FP32 parameters and are 44,829,381 bytes each; analytical W8/W4 packed payload estimates are 35,284,072 and 33,711,208 bytes, respectively. Neither quantity proves a hardware speedup.

## Available execution evidence

The matched length-256, batch-1 pilot measured 32 checks after eight warm-ups. CPU FP32 at eight threads averaged 8.4775 ms total; actual oneDNN dynamic qint8 at four threads averaged 9.7014 ms. The reused RTX 4070 FP32 pilot averaged 5.8364 ms. CPU FP32 and dynamic qint8 achieved scoring FPRs of 1.3148% and 1.5340%, so neither is a protection-feasible winner at the primary point. Dynamic qint8 uses a different activation policy from static W8A8. No fake-quantization INT4 latency is claimed. These are short optimization pilots, not the final repeated 10,000-check workload or energy campaign.

The completed DeepSeek application pilot used 120 requests for 100 pairs from 20 source tasks. Unguarded attack success was 11/100; the guard rejected all 100 attacked chunks, giving 0/100 attack success and 0/100 task completion, versus 24/100 task completion without the guard. Clean completion was 9/20 distinct tasks (45/100 repeated logical cases) with or without the guard. All 120 responses were reviewed under fixed rubrics. The trade-off is protection through evidence removal, not demonstrated utility-preserving sanitization.

The fixed 256-token cap produced 77/120 truncations, 64 with no visible answer. Partial injected fragments and three upstream reference ambiguities are disclosed; rates apply only to the recorded configuration and fixed-reference policy. All 120 model labels matched deepseek-ai/DeepSeek-V4-Flash-0731; immutable revision remained unavailable. There were no API errors or retries, and actual reported usage was 29,168 input / 28,006 completion tokens at USD 0 configured included cost. Full counts, timing boundaries and source-quality findings are in `reports/deepseek_pilot_findings.md` and `reports/application_reference_audit.md`. The historical Gemini pilot remains preserved separately and is not a model comparison.

## Evidence and writing audit

| Claim | Evidence | Status |
|---|---|---|
| No observed recall loss in the bounded quantization experiment | `results/goal3/uniform_summary.csv`, `single_group_sensitivity.json`, `pair_sensitivity.json` | Supported within the fixed development pool |
| Threshold movement and criterion choice change benign rejection | `nearby_operating_points.csv`, `single_group_rank_comparison.csv`, saved paired predictions | Supported descriptively |
| Uniform W8 QAT meets empirical development protection | `qat_w8_a8/candidate.json` | Supported; final/generalization claim unavailable |
| A-to-B improves the eventual hardware frontier | Fixed protocol only | Needs actual A/B comparison |
| C improves B at matched protection | No C selected | Unsupported/untested |
| FPGA checking cost is reduced | Component RTL plus conditional internal-path model; no B-to-C or board comparison | Unsupported/untested |
| Whole-chunk checking prevents the observed injected outputs while removing task evidence | DeepSeek pilot: 11/100 unguarded versus 0/100 guarded attack successes; attacked task completion 24/100 versus 0/100 | Supported within the fixed development run; constrained by 77 truncations and reference ambiguity; no general-robustness claim |

The manuscript follows a general-to-specific introduction and an evidence-first result chain: threshold effects, criterion mismatch, then executable software costs. Methods retain the exact numerical and data rules. Discussion positions those observations against verified prior art and names the unresolved inference. The conclusion does not promote the study to a new allocation or FPGA contribution.

| Canonical term | Definition and use |
|---|---|
| FP32 | Floating-point reference using the selected epoch-3 checkpoint |
| PTQ | Post-training quantization; unchanged floating weights with declared quantizers |
| QAT | Quantization-aware training; exactly one additional epoch with STE and frozen scales |
| W4A8 / W8A8 | Selected encoder linear weight/input formats; other operations remain explicitly FP32 |
| FPR | False positives / benign documents after max-window aggregation |
| A / B / C | Conventional existing-method adaptation / safety-matched adaptation / unresolved added mechanism |
| Dynamic qint8 | Actually accelerated PyTorch oneDNN reference; distinct from static W8A8 |
| DeepSeek-V4-Flash-0731 | Selected hosted target label; not an immutable served-weight revision |
| Attack success rate (ASR) | Strict fixed-rubric successes divided by all attacked cases, including guard rejections |
| Reference-based task completion | Fixed task-rubric agreement; non-ok responses count as non-completion; upstream ambiguities remain disclosed |

Main-text allocation: uniform threshold counts are the core result; matched QAT and the extra-epoch confound are necessary support/qualification. The single-group contrast advances the criterion argument; complete ranks, nearby operating points and pair details remain in the quantization report and CSVs. The short software timing comparison is a capability boundary, and its lack of protection feasibility remains visible. Detailed software arithmetic, parameter inventories and source revisions are in Methods or the linked evidence reports. No inferential p-values or seed-level intervals were invented from dependent maps.

Figures 1–3: `results/goal3/figures/uniform_threshold_effects`, `group_sensitivity` and `software_breakdown`, each with PDF/SVG/PNG/TIFF and source CSV. The coordinating task completed their export checks and visual review. Author, venue, final claim wording and submission disclosures remain review decisions; no submission readiness is claimed. To revise the draft, identify a specific claim or paragraph so that its verified evidence and unaffected sections can be preserved.

Application update allocation: the observed protection/task-evidence trade-off is supporting application evidence. Output truncation and reference ambiguity are main-text qualifications. Request IDs, full tables, provider accounting and timing are provenance details retained in the linked reports and raw data. The old partial-Gemini Discussion inventory was replaced by a concise interpretation of the completed DeepSeek study; historical Gemini files were preserved. No new H2/H3 claim or submission-readiness claim was added.


Hardware update allocation: conditional internal cost and the numerical rounding boundary are main-text qualifications because they constrain any acceleration/protection claim. Per-shape cycles, register/traffic counts, HLS resources and failure logs remain in the Goal 4 reports. The prior one-kernel status paragraph was replaced; the new Results subsection contains 174 words. The hardware Methods paragraph changed from 92 to 128 words. Existing quantization, application and literature claims were preserved. No new paper or external policy claim was added.

| Additional canonical term | Meaning |
|---|---|
| Static-bank64 | The selected common 64-lane integer source with explicit bank indexing; not a proposed C method |
| Modeled internal path | Sequential arithmetic evidence plus measured conditional control/layout cost under the declared nominal memory model |
| Full checking latency | Host preparation through returned decision, including deployed transfers and physical memory behavior; still unresolved |

Claim/evidence update: component validation is supported by saved RTL records; modeled internal cost is conditional analytical inference; software acceleration, hardware protection and B-to-C benefit remain unestablished. This local revision preserves the existing manuscript argument and terminology rather than promoting incomplete hardware work to a new method claim.
