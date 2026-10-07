# WSL FPGA toolchain verification

2026-09-26. Existing user-supplied installations are verified; tool absence no longer blocks Goal 4. The earlier Windows-only PATH/usual-folder inspection remains historical and unchanged.

- Host: Ubuntu 24.04.1 LTS under WSL2, Linux 6.6.114.1-microsoft-standard-WSL2.
- Executables: `/home/jp19126/Xilinx/2025.2/Vitis/bin/{vitis,vitis-run,v++}` and `/home/jp19126/Xilinx/2025.2/Vivado/bin/vivado`.
- Actual `vitis-run --version` and `--help` work: 2025.2, SW Build 6295257. Actual Vivado catalog query exited 0: 2025.2, SW Build 6299465, IP Build 6300035.
- Installed board `xilinx.com:vek280:part0:1.2` selects `xcve2802-vsvh1760-2MP-e-S`. Evidence: `results/goal4/toolchain/vivado_probe.log` and `board_definition.json`. Physical unit revision remains unchecked.

| Catalog resource | Total | Declared 70% design budget |
|---|---:|---:|
| DSP | 1,312 | 918 |
| LUT | 520,704 | 364,492 |
| Flip-flop | 1,041,408 | 728,985 |
| BRAM, 36 Kb blocks | 600 | 420 |
| URAM | 264 | 184 |

The 30% reserve is an engineering allowance for integration, not measured utilization. Shared configuration selects Vivado PL IP, three AXI4 memory master bundles and AXI4-Lite control; DDR/NoC and host integration remain pending. The 200 MHz objective, 0.5 ns uncertainty and lanes4/8/16 profiles (tiles 4×16×64, maximum 256 rows) are characterization choices. No AI Engine compilation or XRT platform is required for this scoped integer-kernel build.

The first lanes4 C-simulation attempt failed because Vitis handled quoted `-I` paths incorrectly. Corrected flags produced a passing retry: six W4/W8 signed matrix cases at five rows agreed with the independent int64 reference, and inner dimension 128 was rejected. Both attempts are preserved in `results/goal4/toolchain/lanes4_csim_attempt1_*` and `lanes4_csim_pass_*`.

First lanes4 HLS synthesis also exited 0. The XML reports a 4.422 ns estimated period (approximately 226.14 MHz), against a 5 ns target and 0.5 ns uncertainty; estimated resources are 2 BRAM18K, 3 DSP, 6,291 FF, 6,809 LUT and 0 URAM. The HLS report uses 18 Kb BRAM units, whereas the catalog table above uses 36 Kb units. Top-level latency and interval are `undef` because dimensions/loops are runtime controlled. Final saved evidence is `results/goal4/lanes4/{summary.json,csynth.xml,csynth.rpt,vendor.log,run.json}`; no vendor process remains running. This is synthesis evidence, not post-route timing closure or measured performance.

Goal 4 remains partial: characterize the remaining profiles and runtime RTL points, then build and validate full-detector costs. No C/RTL co-simulation, place-route or board measurement has run. Follow `reports/goal4_characterization_plan.md` and reuse unchanged successful builds within the original bounded allowance.

Use Linux Python; native Windows Python cannot directly launch the configured Linux executable. The completed simulation command is:

```powershell
wsl.exe -d Ubuntu --exec /usr/bin/python3 /home/jp19126/Projects/GATE/scripts/build_linear_hls.py --csim --profile lanes4
```

AMD's [Vitis 2025.2 installation requirements](https://docs.amd.com/r/2025.2-English/ug1742-vitis-release-notes/Installation-Requirements) and [Vivado 2025.2 OS list](https://docs.amd.com/r/2025.2-English/ug973-vivado-release-notes-install-license/Supported-Operating-Systems) include Ubuntu 24.04.1, without explicitly certifying WSL2. Passing C simulation/HLS synthesis establish those paths here; later-stage compatibility and exact license coverage remain unproven. [2025.2 target flows](https://docs.amd.com/r/2025.2-English/ug1399-vitis-hls/Target-Flow-Overview) distinguish PL-IP builds from accelerated application deployment.

The board is temporarily unavailable for connection. No programming or hardware-server operation was attempted. Offline implementation can proceed when its requirements are met; [post-route estimates](https://docs.amd.com/r/2025.2-English/ug1399-vitis-hls/Running-Implementation) still do not establish Goal 6 on-board latency, transfer behavior or energy.