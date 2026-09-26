# Mixed-Precision FPGA Co-Design for LLM Guards

Portable project preparation for a separate Windows PC session. The research
has **not started**: there are no downloaded datasets, trained models, selected
published baseline, synthesis results or board measurements.

1. Clone `https://github.com/jp19126/GATE.git` to a short local path such as
   `C:\research\GATE` on the Windows PC, or pull updates into an existing clone.
2. Open that folder in the next session and paste the startup request from
   `WINDOWS_HANDOFF.md`. That session should perform setup and Goal 1.
3. Continue one goal at a time, using `STATUS.md` for actual evidence.

Files:
- `CODEX_RESEARCH_GOALS_v2_1.md`: supplied prompt, preserved verbatim.
- `docs/FPGA_LLM_Security_Research_Plan_CN_v2_1.pdf`: original research plan.
- `configs/project.json`: shared starting settings; unresolved choices are null.
- `WINDOWS_HANDOFF.md`: startup request, setup and platform boundaries.
- `scripts/bootstrap.ps1`: create a Python 3.11 environment without downloads.
- `scripts/inspect_environment.py`: one-time target environment record.

The setup scripts use no research packages. The next session chooses the actual
PyTorch build after inspecting the target GPU/driver and installs only Goal 1
dependencies. Package and model revisions must be recorded after a working
installation; this preparation does not claim a validated ML environment.

The experiment asks whether one concrete improvement reduces complete checking
cost beyond a published method adapted to the same low-FPR requirement. A is the
existing method, B adds the task requirement during selection, and C adds one
evidence-supported mechanism. Main board designs are uniform/B/C.
