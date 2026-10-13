# SENTRY — FPGA-aware methods for LLM security

SENTRY is a research project on **new FPGA-aware algorithms and
algorithm–hardware co-design for LLM security**. The intention is to create
and test a scientific method whose FPGA properties matter to the design, not
simply to accelerate an established detector. Neither a particular method
nor an FPGA advantage has yet been demonstrated.

The existing research pilot studies **indirect prompt injection in external
documents** for a document-question-answering application. It includes a
BERT-Mini detector, quantization experiments, software baselines and offline
FPGA components. These are **supporting evidence**, not a completed new method
or general proof against LLM/model poisoning.

## Where to start

| Document | Purpose |
|---|---|
| [RESEARCH_SCOPE.md](RESEARCH_SCOPE.md) | Approved scientific purpose, application boundary and criteria for a contribution |
| [AGENTS.md](AGENTS.md) | Standing execution and evidence rules; deliberately **no specific research goals** |
| [STATUS.md](STATUS.md) | Current findings, unresolved claims, permissions and next decision |
| [Experimental protocol](reports/protocol.md) | Saved pilot data, split, threshold, numerical and evaluation definitions |
| [Execution log](reports/execution_log.md) | Detailed historical actions and evidence paths |
| [Documentation archive](docs/archive/README.md) | Old goal documents, execution prompts and setup handoffs |

The user's explicit research decisions govern. Historical instructions in an
archived prompt or proposal do not authorize new execution or replace the
current scientific scope.

## Repository map

| Location | Contents |
|---|---|
| [configs/project.json](configs/project.json) | Recorded experimental settings, paths and resource fields; not independent authorization |
| [gate_eval.py](gate_eval.py) | Window aggregation, threshold fitting and detector evaluation |
| [gate_quant.py](gate_quant.py), [gate_fpga_reference.py](gate_fpga_reference.py) | Numerical/quantization references |
| [scripts/](scripts/) | Data, training, scoring, comparison and hardware workflows |
| [hardware/](hardware/) | HLS/RTL components, testbenches, host integration and build inputs |
| [reports/](reports/) | Protocol, prior-art review, numerical/hardware reports and execution record |
| [paper/draft.md](paper/draft.md) | Interim scientific manuscript; not a finished contribution claim |
| [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) | Recorded Windows/WSL/toolchain setup and reproduction cautions |
| [docs/FPGA_LLM_Security_Research_Plan_CN_v2_1.pdf](docs/FPGA_LLM_Security_Research_Plan_CN_v2_1.pdf) | Historical proposal, not the current research authorization |

`data/`, `checkpoints/`, `results/`, `build/` and local environments are
Git-ignored. **A fresh clone alone does not contain the saved models, measurements
or generated vendor builds.** Keep these separate artifacts and their
identities when attempting reproduction.

## Current evidence boundary

As of **13 October 2026**, the bounded
[Goal R1 decision study](reports/goal_r1_decision.md) is complete. It produced
software detector and comparison evidence, including a measured W8-QAT
BERT/CUDA-Graph development baseline. It did **not** identify or validate the
new FPGA-aware method required by the current research scope. The older
FPGA component synthesis, RTL probes and analytical cost estimates likewise
do not establish end-to-end hardware protection or a physical FPGA speedup.

The original precision-allocation A/B/C study has not produced a demonstrated
new C mechanism or completed method comparison. Final held-out quality,
end-to-end legitimate-task protection and physical-board measurement remain
unresolved. Consult [STATUS.md](STATUS.md) for the latest authorization and
[reports/research_summary.md](reports/research_summary.md) for historical
experiment details and claim limitations.

## Environment and paths

The recorded research checkout is
`/home/jp19126/Projects/SENTRY` in Ubuntu WSL; historical reports and the
existing native Windows Python environment may still use the former `GATE`
name. Preserve those source/artifact associations rather than assuming the
renamed repository makes compiled outputs portable. Vendor tools and physical
hardware have separate requirements. See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)
and [reports/environment.md](reports/environment.md).

## Commands and checks

Do not treat examples from historical prompts as work orders. Consult the
relevant script's `--help`, the current protocol and the approved task before
running a model, vendor build, paid service or board operation. The repository
has no single general test-suite command; use focused checks for affected code.

A read-only documentation check in the existing checkout is:

```sh
python3 -B scripts/update_handoff.py --check
```

The detailed commands formerly included in this README are available in the
[pre-cleanup snapshot](https://github.com/jp19126/SENTRY/blob/3ae7b5b41efe02dde2f11f8a82ef7de850c4877f/README.md)
and the relevant reports. They document earlier experimental stages and
**are not permission to rerun them**.

## Updating the Markdown handoffs

Use `scripts/update_handoff.py --write STATUS.md --from-file CANDIDATE`
(or `reports/board_handoff.md` as the target) for a reviewed UTF-8 candidate,
then run `--check`. The helper is designed to protect the handoff from encoding
corruption or concurrent replacement. Current findings belong in `STATUS.md`;
detailed experiment records belong in `reports/`, not in this README.

## Scientific protocol and evidence boundaries

The existing pilot's grouped partitions, scoring/threshold rules, numerical
identities and closed final sets are documented in
[reports/protocol.md](reports/protocol.md). Pilot-specific thresholds are not
universal requirements for a future method. Distinguish classifier quality,
real protection of a target LLM, software timing, synthesis and measured
end-to-end FPGA behavior. Report each only within its actual evidence boundary.

## Git synchronization after the rename

The canonical remote is [jp19126/SENTRY](https://github.com/jp19126/SENTRY)
and the primary branch is `main`. Preserve existing local artifacts and
uncommitted changes. Review the actual diff and remote head before
committing, pushing or merging; do not interpret document edits as approval
to run experiments. Older `GATE` checkout instructions are archived in
[docs/archive/](docs/archive/).
