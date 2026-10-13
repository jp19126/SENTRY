# Windows handoff

> Historical initial-PC setup instructions. Read [STATUS.md](STATUS.md) for the
> current research pause, completed evidence, budgets and next actions. Read
> [README.md](README.md#environment-and-paths) for current checkout paths and
> commands. The original one-goal text and `GATE` clone/setup examples below are
> historical; they do not restart Goal 1 or authorize research resumption.

## Current FPGA handoff

Use WSL Python with the existing configured Vitis/Vivado 2025.2 installation.
The VEK280 catalog part and tool execution have recorded evidence; current board
connection, physical memory/host integration and programming authorization remain
unresolved or unavailable. [STATUS.md](STATUS.md#current-offline-fpga-evidence)
links the completed synthesis, RTL, controller and real-input-prefix results.
Reuse those artifacts; all 12 initial HLS syntheses have been used.

The saved RTL measurements use a declared simulated-memory model. Neither those
measurements nor the controller checks with arithmetic stubs establish a working
full FPGA detector. The [numerical bridge](reports/goal4_numerical_bridge.md) and
[remaining measurements](reports/unfinished_measurements.md) describe the open
work. No additional pilot API call is needed.

## Historical one-goal startup text

```text
This is the prepared Mixed-Precision FPGA Co-Design for LLM Guards project.
We are now on the Windows PC where execution should take place.
Read AGENTS.md, STATUS.md, WINDOWS_HANDOFF.md, configs/project.json and
CODEX_RESEARCH_GOALS_v2_1.md. Refer to the PDF in docs/ for research requirements.
Apply the Shared Instructions and execute Goal 1 only, including implementation,
actual data preparation, the small evaluator/split checks, model-loading smoke
run and evidence-based baseline selection. Do not stop at scaffolding.
First inspect this PC once and reuse any suitable existing environment. If a new
Python environment is needed, use scripts/bootstrap.ps1. Install only the
packages needed now, choosing PyTorch for the actual GPU and driver. Record
working versions. Use configs/project.json as the shared source of settings.
Do not treat the macOS preparation as Windows, GPU or FPGA validation. Keep
unresolved board details separate from software progress. Do not start Goal 2.
At the handoff, update STATUS.md with actual results, output paths, remaining
access needs and the reproduction command. Ask only for essential access,
physical work, a budget extension or a substantive research change.
```

## Current Git synchronization

The renamed remote is [jp19126/SENTRY](https://github.com/jp19126/SENTRY).
For a new native Windows checkout:

```powershell
git clone https://github.com/jp19126/SENTRY.git C:\research\SENTRY
cd C:\research\SENTRY
```

The active research checkout is in WSL at `/home/jp19126/Projects/SENTRY`;
use [README's current Git commands](README.md#git-synchronization-after-the-rename)
for that checkout. The verified native Python environment remains at
`C:\research\GATE\.venv`; preserve it and the ignored research artifacts.
The examples below record the original setup and are not current clone commands.

## Original sync procedure (historical)

For a new checkout:

```powershell
git clone https://github.com/jp19126/GATE.git C:\research\GATE
cd C:\research\GATE
```

For an existing checkout, commit or preserve local changes, then run
`git pull --ff-only` from that checkout. Python environments and generated
research artifacts are machine-local and are not transferred by Git.

## Original setup procedure (historical; preserve the working environments)

Use a short writable local path, e.g. `C:\research\GATE`. Preserve an existing
working environment if compatible. Otherwise use 64-bit Python 3.11 (a
conservative starting choice, not a claim that later versions are unsupported):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
.\.venv\Scripts\python.exe .\scripts\inspect_environment.py
```

The execution-policy override applies only to this process. If organizational
policy prevents scripts, use `py -3.11 -m venv .venv` directly. The bootstrap also
accepts `-PythonExe C:\path\to\python.exe` for an existing Python 3.11 runtime.
No activation or administrator access is required. The bootstrap downloads
nothing. The inspection writes `reports/environment.json` once and preserves it
unless `--refresh` is explicitly supplied after a relevant environment change.
A missing command on PATH does not prove a vendor tool is uninstalled; inspect
known installation locations once when needed. Do not scan the entire disk.

After inspection, choose a PyTorch installation from its official selector for
the actual Windows/GPU combination. Install Transformers and dataset dependencies
only as the Goal 1 implementation needs them. Defer Brevitas, ONNX Runtime,
FINN/QONNX and vendor-tool installation to their relevant stage. Save successful
versions with `.\.venv\Scripts\python.exe -m pip freeze` in a report, plus the
chosen PyTorch installation command. Record immutable source/model revisions in
config before using them for comparative experiments; do not invent revisions.

## Windows and FPGA boundary

Windows is the execution/control PC. Native Windows Python is the initial
software route. Hardware execution must follow the installed tool release's OS
and board-interface support. Do not assume Windows support for Vivado implies
support for every Vitis/XRT/Alveo flow. WSL2 availability does not establish PCIe,
DMA, USB/JTAG, driver or vendor support. Use an existing supported native tool
flow where possible; a Linux hardware host may be needed for the allocated
board. Establish that at the hardware stage without blocking Goal 1 software.
Do not install WSL, change drivers or choose a board merely to fill null fields.

Primary platform references checked during preparation (2026-09-26):
- [PyTorch installation selector](https://docs.pytorch.org/get-started/locally/)
- [AMD tool-specific installer and OS support](https://www.amd.com/en/support/adaptive-socs-and-fpgas/installer-info-general.html)
- [Vivado 2025.2 supported OS](https://docs.amd.com/r/2025.2-English/ug973-vivado-release-notes-install-license/Supported-Operating-Systems)

The installed 2025.2 toolchain, VEK280 catalog part and initial profiles are now
configured. Goal 4 C simulation and first HLS synthesis passed without a connected
board. Physical connection is required later for programming and board measurements.
Keep FPGA computation and target-LLM execution separate as the proposal requires.

## Original preparation checklist (historical)

- Environment note for this PC, actual data-source/version and label definition.
- 100 EmailQA development pairs, adequate separate training data, grouped
  train/search/final-calibration/test partitions and five representative pairs.
- Search-threshold/search-scoring group separation and separate NotInject data.
- Shared windowing/document aggregation, evaluator and the prescribed tiny checks.
- Real BERT-Mini loading/scoring result; random-head output is only bring-up.
- Read actual HAO and Quasar-ViT methods and needed related sources; select one
  faithful baseline adaptation with explicit A/B/C roles and protocol report.

At the original preparation stage, source algorithms and publication claims had not yet been verified. Current completed evidence is recorded in `STATUS.md`.
No credentials are included. Prompt Guard access and a target endpoint, if needed,
are separate access questions; no paid usage is authorized by this package.
