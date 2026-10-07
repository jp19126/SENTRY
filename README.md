# SENTRY: Mixed-Precision FPGA Co-Design for LLM Guards

SENTRY studies whether one improvement to precision allocation or FPGA execution
can reduce the complete cost of checking external documents for indirect prompt
injection at the same protection requirement. The detector is BERT-Mini; the
application is BIPIA EmailQA. The target LLM runs separately on an authorized
endpoint, not on the FPGA.

The selected baseline is a restricted Quasar-ViT adaptation. A uses its ordinary
quality criterion; B incorporates the low-FPR protection requirement during
selection; C will add one evidence-supported change. **B-to-C is the main method
test.** C and the actual A/B/C campaign remain unfinished.

Start with [STATUS.md](STATUS.md) for the current objective, evidence and next
steps. Research is paused; this documentation review does not resume it. Goals 1
and 2 and the available Goal 3 software work have saved results. Goal 4 has real
component synthesis/RTL evidence, but full-detector numerical acceptance,
complete checking costs and board measurements remain unresolved.

## Repository map

| Location | Role |
|---|---|
| [configs/project.json](configs/project.json) | Shared model revisions, data splits, formats, budgets, hardware and execution controls |
| [gate_eval.py](gate_eval.py) | Document windowing, maximum-risk aggregation, threshold fitting and metrics |
| [gate_quant.py](gate_quant.py), [gate_fpga_reference.py](gate_fpga_reference.py) | Fixed-scale W4/W8-A8 emulation and the optional ordered FP32 reference under numerical qualification |
| [scripts/](scripts/) | Data preparation, training, quantization, application pilot, profiling, search adapter and hardware drivers |
| [hardware/](hardware/) | Integer and FP32 HLS services, RTL mover/sequencer, testbenches and Tcl |
| `data/raw/`, `data/prepared/` | Downloaded sources; grouped JSONL partitions, development pairs, separate NotInject and `summary.json` |
| `checkpoints/`, `results/goal*/`, `build/` | Models; scores, API ledgers, measurements and figures; generated vendor projects/snapshots |
| [reports/](reports/), [paper/draft.md](paper/draft.md) | Protocol, evidence and detailed execution log; interim manuscript |
| [research goals](CODEX_RESEARCH_GOALS_v2_1.md), [proposal PDF](docs/FPGA_LLM_Security_Research_Plan_CN_v2_1.pdf) | Operational defaults and scientific requirements |

The implemented path is prepared documents -> shared window/evaluation code ->
floating detector -> quantized development candidates. Hardware drivers exercise
separate integer/FP32 services and control integration; the complete deployed
detector and host link are not yet validated. `search_precision.py` has an A/B
adapter and offline checks, but no CLI campaign runner. The application pilot
combines saved guard scores with target responses in four clean/attacked ×
checked/unchecked conditions.

`data/`, `checkpoints/`, `results/`, `build/` and environments are Git-ignored;
a clone alone cannot reproduce saved results. Preserve or obtain the existing
artifacts. The selected floating checkpoint is `checkpoints/floating/epoch_3`;
quantized anchors are under `checkpoints/quantized/`. Only floating epoch 3 is
present in this checkout. Raw responses and judgments for the completed DeepSeek
pilot are in `results/goal2/application_deepseek/`; historical Gemini results in
`results/goal2/application/` stay separate.

## Environment and paths

This checkout is `\\wsl.localhost\Ubuntu\home\jp19126\Projects\SENTRY`
(Linux `/home/jp19126/Projects/SENTRY`). The historical WSL `GATE` checkout still
exists and is a **different directory**. Use this checkout's scripts for current
work. Saved records retain their original `GATE` paths; compiled projects and
cached identities may need path review before reuse. A successful preview does
not validate relocated vendor builds.

Software experiments used native Windows 64-bit Python 3.11.16, torch 2.9.1+cu128,
Transformers 4.57.3 and an RTX 4070. Quantization used Brevitas 0.13.4; figures used
matplotlib 3.11.2. The existing interpreter is
`C:\research\GATE\.venv\Scripts\python.exe`. See the recorded installation commands
in [environment.md](reports/environment.md) and package freezes
[Goal 1](reports/requirements-goal1.txt) / [Goal 3](reports/requirements-goal3.txt).
For a new native environment, [bootstrap.ps1](scripts/bootstrap.ps1) requires
64-bit Python 3.11 (`py -3.11`, or `-PythonExe`); it creates/reuses `.venv` without
downloading research packages. Follow the [historical setup procedure](WINDOWS_HANDOFF.md#original-setup-procedure-historical-preserve-the-working-environments)
only when needed; install dependencies for the relevant stage.

Hardware commands use `/usr/bin/python3` in Ubuntu WSL and the configured
Vitis/Vivado 2025.2 tools under `/home/jp19126/Xilinx/2025.2`. Native Windows
Python cannot launch these Linux tools. Saved software checkpoint paths also
contain Windows separators, so do not assume all software scripts are portable
to Linux. [fpga_toolchain.md](reports/fpga_toolchain.md) records the tool/part
checks; [STATUS.md](STATUS.md) records later validation and remaining access needs.

## Git synchronization after the rename

The current remote is [jp19126/SENTRY](https://github.com/jp19126/SENTRY), with
`origin` set to `git@github.com:jp19126/SENTRY.git` and `main` tracking
`origin/main`. Commit or preserve local changes before pulling. For this WSL
checkout, use its owner's Linux Git:

```powershell
wsl.exe -d Ubuntu --exec git -C /home/jp19126/Projects/SENTRY remote -v
wsl.exe -d Ubuntu --exec git -C /home/jp19126/Projects/SENTRY pull --ff-only
wsl.exe -d Ubuntu --exec git -C /home/jp19126/Projects/SENTRY push origin main
```

Windows Git also works after the exact checkout is trusted. The rename audit
added only this directory to Windows Git's `safe.directory` list:

```powershell
git config --global --add safe.directory '%(prefix)///wsl.localhost/Ubuntu/home/jp19126/Projects/SENTRY'
```

Keep the existing native interpreter, external credential variable names and
`gate_*`/`GATE_*` software/RTL interfaces. They are still valid dependencies;
changing their names would break archived checkpoint and synthesized-service
associations. Historical commands and saved records retain their original paths.
Run current Python drivers from SENTRY; saved vendor Tcl/shell projects still
contain GATE paths and must not be launched directly as relocated builds.
Read-only source-association checks do not qualify a compiled snapshot for reuse.

Git carries source and reports. The ignored data, checkpoints, results, vendor
builds and environments still require separate preservation. See the
[rename audit](reports/execution_log.md#2026-10-08--sentry-rename-audit-and-git-synchronization)
for the checked paths and validation boundary.

## Commands and checks

PowerShell variables for this checkout:

```powershell
$Repo = '\\wsl.localhost\Ubuntu\home\jp19126\Projects\SENTRY'
$Python = 'C:\research\GATE\.venv\Scripts\python.exe'
```

The following short commands were run successfully during the 2026-10-07
documentation review. They print help/a plan without inference, network calls,
vendor execution or result-file writes:

```powershell
& $Python -B "$Repo\scripts\application_pilot.py" --help
wsl.exe -d Ubuntu --exec /usr/bin/python3 -B /home/jp19126/Projects/SENTRY/scripts/run_real_prefix_rtl.py
```

The commands below were checked against source and prior reports, **not rerun**
for this review. They write artifacts and may perform expensive work. Use them
only for the corresponding authorized stage; reuse completed evidence and honor
pause markers (`results/goal2/pause.request` for training/application/hardware;
`results/goal3/pause.request` for quantization/profiling). The application command
without `--execute` makes no API calls, but regenerates reports and derived
JSON/JSONL files.

| Purpose | PowerShell command | Evidence / prerequisite |
|---|---|---|
| Prepare data and check fields/splits | `& $Python "$Repo\scripts\prepare_data.py"` | [Data sources and checks](reports/data.md); may download pinned sources |
| Tiny evaluator check | `& $Python "$Repo\scripts\check_evaluator.py"` | [Saved check](reports/evaluator_check.json); after evaluator changes |
| Real model bring-up | `& $Python "$Repo\scripts\smoke_model.py" --device cuda --batch-size 8` | [Protocol](reports/protocol.md); model access/cache and CUDA |
| Floating training/resume | `& $Python "$Repo\scripts\train_detector.py" --microbatch-windows 32 --resume` | [Floating run](reports/floating_baseline.md); completed run already saved |
| Quantization arithmetic check | `& $Python "$Repo\scripts\check_quant_arithmetic.py"` | [Saved check](reports/quant_arithmetic_check.json); after format/logic changes |
| Bounded quantization/QAT | `& $Python "$Repo\scripts\quantization_sensitivity.py" --stage all --device cuda --batch-size 32` | [Quantization results](reports/quantization_sensitivity.md); checkpoint and prepared data |
| Short CPU profiling | `& $Python "$Repo\scripts\profile_software.py"` | [Software profile](reports/software_profile.md); training/quantization idle |
| Figures from saved results | `& $Python "$Repo\scripts\plot_development.py" --stage all` | [Figure contract](reports/figure_contract.md); all source measurements present |
| Offline application summary | `& $Python "$Repo\scripts\application_pilot.py"` | [DeepSeek findings](reports/deepseek_pilot_findings.md); preserve the exhausted request ledger |

There is no general test-suite command. The research goals prescribe only data,
evaluator, arithmetic and small real-path checks after relevant changes.
Hardware resume steps are in [STATUS.md](STATUS.md#next-step-and-resumption) and
[unfinished measurements](reports/unfinished_measurements.md); the initial
12-synthesis allowance is already exhausted.

## Scientific protocol and evidence boundaries

[protocol.md](reports/protocol.md) defines the source-group splits, document-only
inputs, L256 windows with 64-token overlap, maximum window risk and strict
`score > threshold` rule. Temporary threshold fitting and candidate scoring use
separate groups. B/C require FPR <= 1% and at most one percentage point of recall
loss from the matching floating reference. Final calibration/test and NotInject
remain closed until methods are frozen. The benign test pool has 1,810 distinct
emails, below the 3,000 aim; 100 pilot pairs represent only 20 original tasks.

W4A8/W8A8 describe selected encoder linears, with the remaining operations
explicitly FP32. Software quantization is integer-arithmetic emulation, not
accelerated INT4 performance. Ordered-reference scores cannot inherit historical
SDPA scores; both approved score identities remain null. The completed DeepSeek
pilot's 256-token cap caused 77/120 truncations, and rejecting all attacked chunks
removed legitimate task completion as well as attack success. It is development
evidence, not held-out end-to-end protection.

Use [baseline selection](reports/baseline_selection.md), [method](reports/method.md),
[numerical bridge](reports/goal4_numerical_bridge.md) and
[research summary](reports/research_summary.md) for assumptions and claim limits.
Keep current progress in [STATUS.md](STATUS.md), detailed historical attempts in
[execution_log.md](reports/execution_log.md), and repository working instructions
in [AGENTS.md](AGENTS.md).
