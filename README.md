# SENTRY: Mixed-Precision FPGA Co-Design for LLM Guards

**Scientific objective:** Develop and evaluate a **substantive new FPGA-aware
guard algorithm or algorithm–hardware co-design method** for LLM security.
The method must change the guard's computation, representation, decision
procedure or its joint algorithm/FPGA design. FPGA implementation is central;
merely porting a successful existing BERT or other detector to FPGA/AIE-ML is
not the intended research contribution. Established detectors remain useful
baselines, teachers or components, not automatic answers.

The goal is a defensible security–hardware trade-off, potentially permitting
prospectively bounded degradation in detection quality or legitimate-task
utility when a significant measured FPGA benefit warrants it. Neither the
novelty nor superiority of any proposed method is presumed. A completely new
pretrained language model is not required.

The current pilot uses BERT-Mini to detect indirect prompt injection in external
documents (BIPIA EmailQA). The target LLM runs separately on an authorized
endpoint. The pilot does not establish general training/model poisoning defense.

The prior precision-search protocol adapts Quasar-ViT: A uses ordinary quality,
B adds a low-FPR constraint and C was intended to test an added method. This
historical B-to-C protocol is unfinished and **not a command to continue the
existing-model implementation**. No particular new algorithm/co-design method
or experiment is yet approved.

**Current instructions:** [AGENTS.md](AGENTS.md) defines the required
method-level novelty and FPGA role; [STATUS.md](STATUS.md) records evidence
and authority; [SENTRY_next_goal_prompt.md](SENTRY_next_goal_prompt.md)
governs selection of a concrete method and bounded test. [Goal R1](reports/goal_r1_decision.md)
is a completed detector/architecture decision study, not a scientific
completion. The [old board-free prompt](SENTRY_BOARD_FREE_COMPLETION_PROMPT.md)
is historical. Goals 1 and 2 and available Goal 3 software work retain useful
evidence; Goal 4 has component synthesis/RTL evidence but complete numerical
acceptance, full checking costs and board measurements remain unresolved.

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
| [research goals](CODEX_RESEARCH_GOALS_v2_1.md), [proposal PDF](docs/FPGA_LLM_Security_Research_Plan_CN_v2_1.pdf) | Historical research protocol and defaults; current purpose governed by AGENTS.md |

The implemented path is prepared documents -> shared window/evaluation code ->
floating detector -> quantized development candidates. Hardware drivers exercise
separate integer/FP32 services and control integration; the complete deployed
detector and host link are not yet validated. `search_precision.py` has an A/B
adapter and a gated resumable campaign runner; its numerical/cost prerequisites
remain unresolved and zero comparison maps have run. The application pilot
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

Goal R1 entry points are `scripts/r1_challengers.py` (prepare, bringup,
train-feature, train-cnn, score, timing), `scripts/r1_bert_int8.py` (bringup,
score, timing), `scripts/r1_bert_cuda_graph.py` (bringup, score, timing), and
`scripts/r1_bert_bf16_attention.py` (bringup, score). Only `--help` is read-only:
their stages execute work and write fresh outputs under `results/goal_r1/`.
`scripts/r1_cpu_int8_pilot.py` is explicitly diagnostic because complete CPU
quality was not established. `scripts/r1_anchor_audit.py` recomputes saved anchor
metrics without model execution. Reuse the completed evidence; the decision
report records exact commands, limits and arithmetic boundaries. R1 uses its
own `results/goal_r1/pause.request`; the older pause markers stay intact.

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
| Offline whole-check cost figure | `& $Python -B "$Repo\scripts\plot_offline_cost.py" --cost-table "$Repo\results\goal4\cost_table_nominal_caps_7624_7642_7660_7679_v1.json"` | Reexports SVG/PDF/PNG/TIFF and source CSV in `results/goal4/figures/offline_cost_v2/`; no inference/vendor work. Uses the installed Python figure/alignment helpers; rerun rendered QA after changing the figure. |
| Offline application summary | `& $Python "$Repo\scripts\application_pilot.py"` | [DeepSeek findings](reports/deepseek_pilot_findings.md); preserve the exhausted request ledger |

There is no general test-suite command. The research goals prescribe only data,
evaluator, arithmetic and small real-path checks after relevant changes.
Current board-free entry points (default previews do not run a campaign):

```powershell
& $Python -B scripts/export_qualification_cases.py
& $Python -B scripts/qualify_ordered_batching.py
& $Python -B scripts/score_qualified_development.py
& $Python -B scripts/run_precision_campaign.py
& $Python -B scripts/freeze_goal4_cost_table.py
& $Python -B "$Repo\scripts\bringup_detector_shapes.py" --length 128
& $Python -B "$Repo\scripts\bringup_detector_shapes.py" --length 512
wsl.exe -d Ubuntu --exec /usr/bin/python3 -B /home/jp19126/Projects/SENTRY/scripts/run_precision_qualification.py --name uniform_w4_qat
wsl.exe -d Ubuntu --exec /usr/bin/python3 -B /home/jp19126/Projects/SENTRY/scripts/run_qualification_cases.py
wsl.exe -d Ubuntu --exec /usr/bin/python3 -B /home/jp19126/Projects/SENTRY/scripts/run_real_detector_rtl.py
wsl.exe -d Ubuntu --exec /usr/bin/python3 -B /home/jp19126/Projects/SENTRY/scripts/plan_detector_commands.py --length 512
```

`prepare_numerical_cases.py` freezes the 32 development input records using historical
scores and the local tokenizer; it performs no inference. The new shared scorer
explicitly selects the ordered final risk and resumes full development partitions
only after the separate numerical approvals. `run_precision_campaign.py` preserves
pending proposals/RNG state, per-arm logical charging and the common QAT allowance.
Campaign B8/B32 scoring validates an exact map/checkpoint batching proof. The runner
reuses compatible saved exports or dispatches only the already charged pending map's
bounded 66 B1 windows and three B32 (or nine B8) batches. Mixed-map scores also require
an explicit complete checked-C coordinator review at
`methods.mixed_numerical_review_path`; a batching proof grants no numerical approval.
Before the first campaign snapshot, declare a separate
`methods.mixed_qat_numerical_review_path` when PTQ is approved first. Its absent
record leaves mixed QAT unresolved. Separated QAT scoring requires its own actual
complete matching-source 66-window/32-document exemplar and root decision; each
consumed review remains immutable. The default without this optional field retains
the existing shared-path contract. The native one-job guard at
`results/goal5/campaign_next_execution_review_v2/guarded_campaign_dispatch_v1/guard_native_campaign.py`
previews with `--arm A` (or `B`) without importing Torch. Actual execution requires
a fresh prepared plan, exact scientific approvals, an exclusive resource handoff
and a root-authored release. Plans bind both review records or their absence;
new records require a fresh plan. See `reports/board_handoff.md` for the current
prerequisites and commands.
The fixed `methods.c_decision_path` reserves the later post-A/B scientific
decision record; its absent file is unresolved. C execution and final-freeze
consumer hooks remain to be implemented after actual A/B findings without
rewriting the frozen methods configuration or common cost sources.
The default preview and `results/goal5/campaign_batch_preparation_source_v1/` describe
the gates and independent work accounting.

`freeze_goal4_cost_table.py` previews the actual integrated closure and full checking
cost associations. Actual freeze requires explicit common latency caps and inspected
current implementations for all active profiles, with at least one eligible profile;
it writes the requested cost table and its validation report only. Configured table
selection and score approvals are separate coordinator decisions. Current source
checks are in `results/goal4/cost_table_freeze_source_v1/`. The actual selected table is
`results/goal4/cost_table_nominal_caps_7624_7642_7660_7679_v1.json`, with
its adjacent validation report and common caps7624/7642/7660/7679ms. All three
L256 profiles have actual routed closure. These are conditional analytical
warm-window costs; cost selection does not approve the numerical score identities.

`inspect_vek280_system.py --system-dir PATH --output FRESH.json` reparses existing
implemented reports, artifacts, source associations and embedded NoC records; it
launches no vendor process. Its CLI writes three fresh JSON reports and refuses
to replace an inspection. `--implementation-record PATH` selects a successful
same-project resume when the original attempt stopped. The imported
`inspect_system(...)` API returns the same facts without writing files.

`resume_vek280_system.py --system-dir PATH --limit-seconds 7200` prepares a fresh
resume directory, copies interrupted run trees and saves driver/Tcl/records even
without `--execute`. Execution serially resumes the frozen implementation and
reuses completed DCPs; it is limited to the existing system build tree. It needs
8 GiB WSL/4 GiB Windows headroom before dispatch and retains the 4 GiB/2 GiB
runtime floors, CPU0..5, jobs1, general threads2 and synthesis workers1. Only a
cause-identified interrupted build should be resumed; completed valid artifacts
are reused. Neither entry point accesses a physical board.

`bringup_detector_shapes.py` prepares and qualifies genuine L128/L512 executions
from the two frozen development documents. Its default preview imports no model
and writes no artifacts. `--stage prepare --execute` imports the tokenizer/model stack and saves regenerated
whole-document windows without a model forward; B1 export and B8/B32 checks run
genuine model forwards and preserve their own capture bytes. Repeated rows that fill a batch are recorded explicitly.
The C stage creates a separate shape-specific build and uses the original memory,
operator, strict-decision and source-association checks. It caps each window at
1,800 seconds and compilation plus the first window at 3,600 seconds; those caps
are estimates from primary-length work, not measured shape runtimes. A discrepancy
stops execution for an exact, case-specific coordinator record supplied with
`--reviewed-resolution PATH`. `--stage c-validate --max-forwards 0` reads saved
results without dispatching work or changing qualification state. Plans, exports
and partial C results alone grant no numerical or deployment approval. Source
review is saved in `results/goal4/numerical_bridge/real_length_source_review_v1/`.

`run_detector_arithmetic.py` prepares or executes actual-service integration; its
`--execute --run-only --stop-after N` branch uses a compiled snapshot and writes a
fresh numerical capture. These simulations never access physical hardware.
Alternate workload plans include full-length attention; their generation alone
does not qualify controller, host or numerical execution at those shapes.

Hardware continuation steps are in [STATUS.md](STATUS.md#next-step-and-resumption) and
[unfinished measurements](reports/unfinished_measurements.md); the initial
12-synthesis allowance is already exhausted.

### Updating the Markdown handoffs

Use [update_handoff.py](scripts/update_handoff.py) for agent updates to
`STATUS.md` and `reports/board_handoff.md`. Its default check is read-only and
uses no research dependencies. Reads are bounded before strict UTF-8 decoding;
the limit is 128 KiB per file and 8 KiB per line. Common mojibake and invalid text
are rejected before a same-directory atomic replacement. UTF-8 candidates are
written with LF line endings, preserving normal Unicode punctuation and Chinese
text. The helper also refuses a detected concurrent destination change and
verifies the resulting bytes. It never guesses or repeatedly converts encodings.

```powershell
wsl.exe -d Ubuntu --cd /home/jp19126/Projects/SENTRY --exec /usr/bin/python3 -B scripts/update_handoff.py --check
```

Prepare a complete reviewed candidate inside the repository using an editor set
to UTF-8, or Python's explicit `encoding="utf-8"`. For example, after saving
`results/maintenance/status_next.md`:

```powershell
wsl.exe -d Ubuntu --cd /home/jp19126/Projects/SENTRY --exec /usr/bin/python3 -B scripts/update_handoff.py --write STATUS.md --from-file results/maintenance/status_next.md
wsl.exe -d Ubuntu --cd /home/jp19126/Projects/SENTRY --exec /usr/bin/python3 -B scripts/update_handoff.py --check
```

For the board handoff, use `--write reports/board_handoff.md` and its reviewed
candidate. Avoid default-encoding PowerShell text round trips. An already-corrupt
file must be preserved and explicitly repaired before using the updater; there
is no force option to bypass the checks. Direct edits in another editor are
outside this helper, so use `--check` afterwards. These checks do not resume
research or rerun experiments.

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

The frozen numerical set is exported with `export_qualification_cases.py --execute --max-new-windows N` (add `--floating` for the separate ordinary floating reference). `run_qualification_cases.py --execute --compile --max-new-windows 1` performs the one initial actual C-service compilation/run in WSL; later bounded runs omit `--compile`. `validate_qualification_exports.py --compiled` checks actual associations without inference. Every unexplained A8/terminal discrepancy stops continuation; this diagnostic stop does not adopt universal terminal-bit equality. The first C window and selected actual RTL probes pass; broader qualification/score approval remains unfinished.

`qualify_ordered_batching.py --execute --max-new-batches 12` executed the frozen W8 B8/B32 checks. Its optional floating mode is separately gated by complete floating exports. `build_vek280_system.py --phase prepare --length 256 --execute` creates a preserved actual platform-validation attempt; implementation remains a separate authorized offline phase. Both prepare and implementation have no physical-board branch.


Current checked-C continuation uses `run_qualification_cases.py --bounds` and the separate `bounds_v2` evidence trees. Read-only complete validation is `validate_qualification_exports.py --bounds` (add `--floating`); `--allow-partial` audits committed windows without approval. Bounded dispatch is assigned one of CPUs6/7 and uses the configuration memory guard. Recognized propagation requires the separately authored, explicit mode-specific policy; older stopped discrepancies require their exact coordinator resolution. Neither a policy nor a complete validator grants numerical approval. See the explicit decisions in [protocol.md](reports/protocol.md).

`scripts/export_precision_qualification.py` accepts an explicit `--name`, `--precision-map`, and `--source ptq|qat` (with matching `--qat-checkpoint` for QAT). It uses the same frozen32/66 inputs, genuine selected checkpoint/scales, decoded signed codes and separately validated packed W4/W8 bytes. `scripts/run_precision_qualification.py --name NAME` previews its separate compatible checked-C build; no model, C or synthesis runs by default. Actual introduced precision paths require their affected numerical and batching checks before scores are used.

`scripts/measure_frozen_software.py` defaults to a gated preview. After the actual method/numerical freeze, `--stage qualify-backend --finalist-role floating|U|B|C --backend cpu|cuda --format float32|dynamic_qint8|float16|bfloat16` generates permitted development scores and observed runtime metadata. QAT finalists restore every original trained FP32 parameter into ordinary Hugging Face layers, with exact state checks before native format conversion; static fake quantization is omitted, and this separate native computation makes no INT4 claim. Use an actual same-backend `--floating-reference DIRECTORY` for U/B/C or optimized formats and `--resume-attempt DIRECTORY` for an interrupted qualifier. `--stage timing --backend-descriptor PATH` measures only a protection-qualified native candidate. The adopted final grid covers all nine lengths128/256/512 by batches1/8/32 at both boundaries, at least10000 completed checks per repeat for3repeats; B32 rounds up to10016. Smaller explicit `--secondary-checks` coverage is diagnostic. `--stage load` separately requires actual software/FPGA capacities and common arrival files. Quality/backend prerequisites and actual host packet associations are checked before measurements. Source checks pass; actual new QAT loading/backend qualification/timing remain unverified.

`scripts/prepare_frozen_deployment.py` packs genuine checkpoint parameters, both frozen scale tables, precision-selected weight bytes and one true-length input into the existing64MiB arena/job/header layout. Default mode is a stdlib preview; explicit `--execute` reads the checkpoint and packs without a model forward. Bring-up supports128/256/512; `--mode finalist --role U|B|C` requires the actual development freeze. Required explicit bindings include `--window-key`, `--template-job`, `--model-id`, `--threshold-record`, `--purpose` and `--output`. Threshold transfer between lengths is explicitly bring-up-only; final evaluation requires the actual freeze-bound length/window/threshold policy. `--validate-only` reads existing saved bytes without inference. Matching actual bring-up arenas and A72 links are listed in [board handoff](reports/board_handoff.md).

`scripts/evaluate_frozen_finalists.py` previews final-quality readiness without loading a model or reading held-out bodies. The single extension is pinned public BIPIA TableQA; `--phase extension --check-prerequisites` checks its gates. After the actual finalist freeze, `--prepare-tableqa` freezes exactly20 source-disjoint groups with two existing attack variants each; after final calibration, `--execute --phase extension --max-new-documents 32` resumes bounded scoring with unchanged methods and thresholds. Public source and attribution/license evidence are retained in `results/goal7/tableqa_extension_source_v1/`; actual selection/evaluation remain unrun. Historical NewsQA helper code is preserved and its CLI retired.

Offline platform/host entry points are `scripts/build_vek280_system.py`, `scripts/link_vek280_supervisor.py` and `scripts/pack_deployment_model.py`. Generated PDI/XSA files alone do not establish timing/resource eligibility: actual routed inspection must pass. `scripts/run_board_evaluation.py` provides a non-hardware preview; its physical branch requires a later explicit request and approved runtime/settings. See [board handoff](reports/board_handoff.md) for the selected boot/runtime inputs and exact later sequence. No programming or physical measurements are authorized in the active continuation.


`scripts/run_mixed_exemplar_qualification.py` adds prospective diagnostic continuation only for frozen L256 PTQ tuple4488884884844448. It wraps the unchanged canonical precision executor under its original guard, with source-bound root policy, exact24rawA8 and original local/own-terminal checks. Default is a read-only preview. Actual run requires `--stage run --execute --policy results/goal4/numerical_bridge/mixed_campaign_review_v1/exact_exemplar_adapter_source_v1/root_prospective_policy_v1.json --max-new-windows 1` on a free assigned CPU6/7; an existing stop additionally requires its exact `--reviewed-resolution`. Read-only saved revalidation is `--stage validate --execute --policy PATH --max-new-windows 0`. This entry point grants no numerical/scoring approval; actual first callback must be reviewed.

`scripts/run_shape_exemplar_qualification.py` continues only the four existing compiled L128/512 floating/W8 diagnostic paths under the original shape inspector and memory guard. The adopted policy is `results/goal4/numerical_bridge/real_length_bringup_v1/shape_adapter_source_v1/root_prospective_policy_v1.json`. On one free assigned CPU6/7, use `--length 128 --mode floating --stage c-run --execute --policy PATH --max-forwards 1`, adding only the exact saved stop's `--reviewed-resolution` when required. Read-only revalidation uses `--stage c-validate --max-forwards 0 --policy PATH`. No compile/export/native action is exposed. New unknown A8, LN or own-terminal propagation still stops; actual positive and negative callbacks have independent/root runtime verification under `results/goal4/numerical_bridge/real_length_bringup_v1/shape_adapter_source_v1/`; complete true-length numerical acceptance remains pending.

Native guarded campaign commands use extended Windows UNC paths (`\\?\UNC\wsl.localhost\Ubuntu\home\jp19126\Projects\SENTRY\...`) for both the guard and attempt directory, as in `results/goal5/campaign_next_execution_review_v2/first_A_root_handoff_v1.md`. Ordinary UNC hides some saved RTL files beyond MAX_PATH during mandatory frozen-cost validation. This filesystem syntax preserves the same source and cost checks.
