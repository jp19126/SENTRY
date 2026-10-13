# Environment and reproduction notes

**Recorded state, not a fresh validation.** These details describe the
previously used environment and known path constraints. They do not authorize
new experiments, purchases, driver changes or FPGA programming. For the current
research decision, read [../STATUS.md](../STATUS.md).

## Existing checkouts and toolchains

| Item | Recorded location or version | Boundary |
|---|---|---|
| Research checkout | `/home/jp19126/Projects/SENTRY` in Ubuntu WSL | Current source tree; not the older `GATE` directory |
| Native Windows interpreter | `C:\research\GATE\.venv\Scripts\python.exe` | Retained environment and some historical checkpoint associations |
| Python and software stack | Python 3.11.16; PyTorch 2.9.1+cu128; Transformers 4.57.3; Brevitas 0.13.4 | Versions from completed experiments, not a fresh compatibility test |
| Software GPU | RTX 4070 | Previous measurements, not a substitute for FPGA results |
| FPGA toolchain | Vitis/Vivado 2025.2 in `/home/jp19126/Xilinx/2025.2` | Offline tool/part checks recorded; board deployment not established |
| Candidate development platform | VEK280 part catalog/Versal | No current physical access or board timing claim established |

See [reports/environment.md](../reports/environment.md),
[reports/fpga_toolchain.md](../reports/fpga_toolchain.md) and
[reports/board_handoff.md](../reports/board_handoff.md) for saved details.

## Reproducing work safely

The repository does not contain ignored `data/`, `checkpoints/`, `results/` or
`build/` artifacts. Keep downloaded dataset revisions, generated model files,
exact checkpoint identities and vendor outputs associated with their recorded
experiments. Old `GATE` strings in source, paths and generated vendor projects
can be legitimate identifiers; a repository rename does not validate relocated
builds. Do not rewrite them indiscriminately.

For a genuinely new setup, use [scripts/bootstrap.ps1](../scripts/bootstrap.ps1)
only as needed and check the actually installed versions. The historical
[Windows handoff](archive/WINDOWS_HANDOFF_legacy.md) records further setup and
OS caveats but includes **obsolete one-goal commands**. Consult it as history,
not as an execution plan. Avoid installing late-stage dependencies or scanning
an entire disk simply to reproduce a document check.

A safe, dependency-light documentation validation is:

```sh
python3 -B scripts/update_handoff.py --check
```

Run it from the repository root using the available Linux Python. Use a
reviewed UTF-8 candidate and `--write` for `STATUS.md` or
`reports/board_handoff.md`. Other tests should be selected for the code actually
changed and the permissions currently granted.

## FPGA and external access

A toolchain installation, simulation, an implemented design or a generated
boot image does not show that a connected board works. Live JTAG/UART, host
I/O, memory transfer, device programming, boot steps and energy/latency
measurements must be separately established and authorized. Likewise,
preparation of an LLM application evaluation does not authorize paid API calls
or opening held-out data.

## Git and archival paths

The canonical repository is <https://github.com/jp19126/SENTRY>; `main` is the
primary branch. Check the working tree, remote head and generated-artifact
boundaries before synchronizing. The original clone/pull instructions, old
Windows paths and old execution prompt are preserved in the
[documentation archive](archive/README.md).
