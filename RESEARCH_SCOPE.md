# SENTRY research scope

**Scope recorded:** 13 October 2026. This document describes the approved
scientific direction, **not an instruction to start experiments**. No concrete
new algorithm, campaign, budget or hardware run is approved merely by this file.
Current decisions and execution state are tracked in [STATUS.md](STATUS.md).

## Approved scientific purpose

SENTRY investigates **LLM security through a genuinely new FPGA-aware guard
algorithm or algorithm–hardware co-design method**. The research must make the
FPGA essential to the method, its scientific motivation, implementation and
evaluation, rather than treating it as an optional device on which to run a
pre-existing software detector.

The contribution may change the guard's computation, representation, decision
procedure, or the way the algorithm and FPGA architecture are jointly designed.
It does not have to invent a new language foundation model or a wholly new
theory of LLM attacks. Established guards, pretrained encoders and published
hardware designs are legitimate baselines, teachers or components, but an
unchanged detector port, conventional model compression or a standalone
accelerator is not the intended scientific contribution.

Novelty is a **hypothesis to establish**, not an assumption: a candidate must
be distinguished from the closest algorithmic and FPGA/co-design prior work,
with an observable difference and a falsifiable benefit. If no defensible new
method emerges, that finding should be reported rather than substituting a
routine accelerator project.

## Evaluation obligations

The intended result is a justified **security–hardware trade-off**, not an
unconditional claim that FPGA outperforms a CPU or GPU. Some degradation in
detection performance or legitimate-task usefulness may be acceptable, but
its limits must be defined before choosing the winning result, and justified
by measured hardware advantages. Missing attacks, benign false positives and
legitimate task outcomes need separate accounting.

Evidence should attribute outcomes to the method, implementation and their
interaction. Relevant comparisons include competent existing-method software
and FPGA baselines, the proposed method in software where practical, and an
ablation or conventional implementation isolating the new co-design choice.
Account for complete checking cost, including transfers and any offloaded
work. Offline synthesis and simulated RTL can support intermediate findings;
actual FPGA performance and deployment claims require physical measurements.

An **FPGA realization of the proposed method** is part of the intended completed
research. Versal PL/AIE-ML is a possible route where justified, not a mandated
algorithm or a substitute for methodological novelty. Software-only protection
or an accelerated established model alone does not complete this purpose.

## Application and evidence boundary

The **existing pilot** uses indirect prompt injection in attacker-controlled
external documents for a document-question-answering workflow. Its BERT-Mini,
CNN and other studies are reusable baseline evidence. The pilot does not
establish defenses against training-data poisoning, model-weight poisoning,
all contaminated retrieval sources or general misinformation. Detection,
blocking, sanitization and preservation of a legitimate task are distinct
claims; detector labels alone do not demonstrate end-to-end protection.

The exact future threat model and guard mechanism remain subject to research
selection and user approval. The current pilot must not be silently described
as the original broader notion of LLM poisoning, nor must the project quietly
expand to an unrelated attack type to claim success.

## What is and is not decided

| Decision | State |
|---|---|
| New, falsifiable FPGA-aware algorithm/co-design contribution | **Approved scientific purpose** |
| FPGA centrality and eventual hardware realization | **Required for completion** |
| Existing guard ports and BERT/CNN/CUDA observations | **Supporting baselines; not the contribution** |
| Specific new method, hardware mapping, hypothesis or experiment | **Not selected or approved** |
| Numerical/security tolerances for a new method | **To be declared before comparative selection** |
| Further training, API calls, spending, FPGA builds or board programming | **Not authorized by this document** |

For the latest findings and the unresolved research decision, use
[STATUS.md](STATUS.md). The completed R1 result is
[reports/goal_r1_decision.md](reports/goal_r1_decision.md).
Earlier protocols and prompts are retained as historical records in
[docs/archive/](docs/archive/); they cannot override this scope without a new
explicit user decision.
