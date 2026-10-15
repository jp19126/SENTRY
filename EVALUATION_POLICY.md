# SENTRY evaluation policy

**Standing policy adopted 15 October 2026.** This document governs how future
SENTRY studies choose evaluation criteria, select research candidates, and
interpret evidence. It complements [RESEARCH_SCOPE.md](RESEARCH_SCOPE.md)
(scientific purpose), [AGENTS.md](AGENTS.md) (execution rules), and
[STATUS.md](STATUS.md) (current evidence). It prescribes neither an algorithm
nor an experiment and **does not authorize new work, spending, data access,
or hardware operation**.

## 1. Match criteria to the scientific decision

A study must state which decision it supports **before** its decisive
evaluation:

| Decision stage | What must be established | What is *not* required at this stage |
| --- | --- | --- |
| **Explore / advance to FPGA co-design** | Credible, nontrivial detection behavior on relevant inputs; a specific FPGA-aware hypothesis or plausible implementation trade-off worth testing; comparison to appropriate references | Production-grade security, a universally fixed recall/FPR, proven publication novelty, a fully built board implementation |
| **Claim an FPGA-centered scientific contribution** | A method distinct from closest prior work; a justified security–hardware advantage over competent baselines, isolated by controls/ablations; evidence at the level claimed | Superiority on every metric, or always beating a highly optimized GPU |
| **Qualify for real deployment** | Application-specific missed-attack, false-alarm, legitimate-task and end-to-end protection requirements; adequate independent validation; relevant actual-system and FPGA measurements | No generic numerical requirement can replace a threat model and accepted risk budget |

Advancement is a decision to **investigate**, not certification of a guard.
A weak or non-discriminating detector does not become promising merely because
it is small, and an apparently favorable Pareto point is not by itself novelty.
Important catastrophic or mechanism-specific failures must still be investigated,
even when aggregate scores are high.

## 2. Give each numerical criterion an explicit status

In each new study's concise protocol, classify proposed thresholds as:

- **Claim-defining or application-specific requirement:** a genuine pass/fail
  bound justified by the intended claim, threat model, error costs, operating
  constraints and available evidence. Explain why failing it disqualifies
  *that stage* of work.
- **Literature-precedented evaluation point:** a useful, comparable operating
  point or reporting convention, **not** automatically an accepted safety
  requirement. Cite the source and its relevant workload.
- **Exploratory target:** a direction for optimization, *not* an independent
  candidate veto (for example, a hoped-for compression, latency or recall goal).

For every hard bound, record its source, rationale, target population, measured
quantity, statistical limitations and the decision it controls. Do not present
a value selected by the project as universal academic consensus, nor convert
a published model's measured performance into a mandatory standard.

Examples of **study-specific, not globally mandatory**, choices in earlier
SENTRY protocols include: recall >=99%; observed benign FPR <=1%; nominal
0.5% calibration; withheld-mechanism recall >=95%; hard-benign rejection <=5%;
a one-percentage-point BERT recall margin; all three seeds individually passing;
and >=50% parameter/scale storage reduction. Some reflect worthwhile concerns
or published operating-point precedent; **the complete conjunction is not an
established universal academic adoption rule**. Their status in a frozen study
is addressed in section 6.

## 3. Evaluate security behavior without arbitrary vetoes

- Describe the actual attack surface, legitimate task, input/decision unit,
  deployment class balance where known, and what the labels demonstrate.
  Detector labels, attack success, blocking/sanitization, and preservation of
  legitimate task outcomes are separate claims.
- For a detector, report **recall against observed benign FPR**, including
  low-FPR operating points relevant to the application and usable score curves
  when appropriate. A **1% FPR evaluation point**, with **0.5% and/or stricter
  points where sample size allows**, is literature-precedented
  (PromptShield [2]), not a universal admission threshold. An application
  may need stricter or different limits.
- Select thresholds on separate calibration/development data; freeze the
  selection procedure before held-out scoring. A nominal calibration FPR is
  not a guarantee of held-out FPR. State denominators and uncertainty; for
  rare events, the number of **independent** benign examples matters
  (Neyman–Pearson framework [1]).
- Report hard-benign over-defense, withheld attack mechanisms, source/genre,
  length and other relevant strata separately where the evidence supports it.
  Do not hide a severe failure with pooling, but also do not impose an
  unsupported fixed recall bound on every exploratory stratum. Sparse or
  correlated strata are uncertain, not verified passes (PIGuard [3]).
- Use source/payload-ancestry-aware partitions and guard against duplicates,
  paraphrase/template leakage, model-selection exposure and other dependencies
  that invalidate an independence claim. Distinguish constructed diagnostics
  from naturalistic generalization and actual downstream protection.

## 4. Account for variability and near-boundary outcomes

Report available individual training seeds, calibration variation, effective
independent sample counts and appropriate source/group-aware confidence
intervals. Identify what each interval does *not* cover. Fixed seeds are not
independent samples of the deployment population; three seed runs do not
establish a universal reliability guarantee (NeurIPS reporting guidance [6]).

Do not automatically reject a research architecture because one seed or a
small number of documents narrowly misses an exploratory threshold. Explain
the magnitude and uncertainty of the difference. Conversely, diagnose large,
repeatable failures and unstable learning rather than averaging them away.
Predeclare any deployment-style consistency requirement instead of imposing
an across-all-seeds veto after results are visible.

## 5. Evaluate the **security–FPGA** trade-off

For research selection, compare candidates on a common task and sensible
software/FPGA reference implementations. Report model or decision quality
alongside relevant hardware quantities: supported arithmetic and numerical
differences, resource use, on-chip/external storage, activation liveness,
transport, latency, throughput and energy when measured or credibly estimated.
Use actual target constraints and a complete-document, end-to-end boundary,
including CPU/host work and transfers when those matter to the claim.

Do not use parameter bytes, MAC counts, nominal bitwidth, or an assumed
CNN-friendly structure as proof of FPGA benefit. Hardware-aware compression
and mapping research supports exploring **different accuracy/resource/latency
trade-offs**, rather than a universal 50% reduction requirement ([4], [5]).
Label analytical models, synthesis, RTL simulation and board measurements
separately; speedup and power claims require the corresponding evidence.

An ordinary existing model may be a **research backbone or baseline**. Its port
or conventional optimization alone is not the FPGA-centered scientific
contribution required by [RESEARCH_SCOPE.md](RESEARCH_SCOPE.md).

## 6. Preserve pre-registered results and permissions

- **Never rewrite historical pass/fail outcomes.** Earlier frozen protocols,
  including the active CNN adoption contract and protected evaluation splits,
  retain their original primary metrics and conclusions.
- A new standing policy does **not** silently override an already authorized
  study's registered gates. If the user authorizes a changed prospective
  decision rule, document the amendment, its timing and affected evaluations.
  Keep the original result. A reanalysis of existing scores under this policy
  must be labelled **secondary/retrospective** and cannot be reported as a pass
  of the original protocol.
- Establish the new decision stage, metrics and rule **before** inspecting its
  decisive holdout. Do not exploit test outcomes to choose a more favorable
  operating point, comparison, threshold or candidate.
- Do not turn honest negative evidence into positive claims; equally, do not
  generalize a failed candidate or an arbitrarily narrow boundary miss into
  impossibility for a whole model family or for FPGA-centered research.
- Execution prompts must say which metrics are **hard requirements**, which are
  **comparison/reporting metrics**, and which are **aspirational targets**,
  together with their grounds and scope. Keep these experiment-specific
  numbers out of AGENTS.md and this standing policy.

## References and scope of support

These sources **support evaluation methods and trade-off thinking**, not any
single universal SENTRY acceptance threshold.

1. Tong, X., Feng, Y., & Li, J. J., "Neyman-Pearson classification algorithms
   and NP receiver operating characteristics," *Science Advances* 4,
   eaao1659 (2018). https://doi.org/10.1126/sciadv.aao1659
   — application-chosen false-positive constraints and statistical control.
2. Jacob, D. et al., "PromptShield: Deployable Detection for Prompt Injection
   Attacks" (2025). https://arxiv.org/abs/2501.15145
   — low-FPR detector comparisons, not a universal recall floor.
3. Li, H. et al., "PIGuard: Prompt Injection Guardrail via Mitigating
   Overdefense for Free," *ACL* (2025).
   https://doi.org/10.18653/v1/2025.acl-long.1468
   — benign over-defense evaluation and attacks-versus-benign trade-offs.
4. Aarrestad, T. et al., "Fast convolutional neural networks on FPGAs with
   hls4ml," *Machine Learning: Science and Technology* 2, 045015 (2021).
   https://doi.org/10.1088/2632-2153/ac0ea1
   — measured FPGA compression, resources and accuracy trade-offs.
5. Wang, K. et al., "HAQ: Hardware-Aware Automated Quantization With
   Mixed Precision," *CVPR* (2019).
   https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.html
   — hardware-feedback-guided optimization beyond parameter/MAC proxies.
6. NeurIPS, "Paper Checklist," section on experiment statistical significance.
   https://neurips.cc/public/guides/PaperChecklist
   — state and justify uncertainty and variability; no mandated seed gate.
