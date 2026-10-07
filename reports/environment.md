# Goal 1 environment

Native execution checkout: `C:\research\GATE`. Source/handoff repository: `\\wsl.localhost\Ubuntu\home\jp19126\Projects\GATE`. Both locations are explicitly distinguished; the environment is Windows-native, not a WSL CUDA validation.

Observed on 2026-09-26: Windows 11 Pro 10.0.26200; AMD Ryzen 7 5700X3D, 8 cores/16 logical; 33,478,696 KiB visible RAM; NVIDIA GeForce RTX 4070, 12,282 MiB VRAM, driver 591.86. `nvidia-smi` reports driver CUDA compatibility 13.1. The installed PyTorch runtime is CUDA 12.8. `torch.cuda.is_available()` returned true and the device name matched. Real model inference evidence is recorded separately in `results/goal1/smoke_model.json` when complete; this inventory alone is not a performance measurement.

The one-time machine-readable record is `reports/environment.json`; it is preserved by default. `reports/requirements-goal1.txt` records every installed package. Core working versions: CPython 3.11.16 (64-bit), torch 2.9.1+cu128, transformers 4.57.3, numpy 2.4.6, tokenizers 0.22.2, huggingface-hub 0.36.2. Dataset preparation uses Python standard-library JSON/archive/URL routines, so `datasets` was unnecessary. No later-stage quantization, ONNX, FINN or vendor packages were installed.

Setup commands actually used (PowerShell):

```powershell
& C:\research\bootstrap-tools\bin\uv.exe python install 3.11 --install-dir C:\research\python --no-bin
powershell -NoProfile -ExecutionPolicy Bypass -File C:\research\GATE\scripts\bootstrap.ps1 -PythonExe C:\research\python\cpython-3.11.16-windows-x86_64-none\python.exe
& C:\research\GATE\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
& C:\research\GATE\.venv\Scripts\python.exe -m pip install transformers==4.57.3
& C:\research\GATE\.venv\Scripts\python.exe C:\research\GATE\scripts\inspect_environment.py
& C:\research\GATE\.venv\Scripts\python.exe -m pip freeze
```

uv 0.12.19 was installed into `C:\research\bootstrap-tools` using the bundled Python, after the Python.org 3.11.9 per-user MSI failed before installation with `0x80070003`. The portable runtime avoids the failed MSI route; the repository bootstrap then created the actual research venv. No PATH, drivers, WSL setup or board settings were changed. Native command execution required escalation because the Codex sandbox helper could not launch; this is an execution-environment issue, not a research finding.

The [official PyTorch Windows guidance](https://docs.pytorch.org/get-started/locally/) and [versioned CUDA wheel commands](https://pytorch.org/get-started/previous-versions/#v291) were consulted. The build is pinned for reproducibility rather than described as the latest release. Install output: `reports/install_torch.log`, `reports/install_transformers.log`.

FPGA boundary: no Vivado/Vitis/Vitis HLS/v++/hw_server on PATH. Usual C:/D: Xilinx/AMD installation locations were checked once; only AMD chipset software was found under `C:\AMD`. This is not proof that no installation exists elsewhere. Board allocation, exact part, connection, licenses, tool release and supported execution host remain null. No synthesis, programming, board measurement or WSL device access has been validated. These are later hardware-stage requirements and do not block Goal 1 software.

Repository base revision: df904fcefe6566650490f7476e13914524db6907. The Goal 1 experiment used the uncommitted working-tree implementation delivered in this repository; no source hashes or custom integrity manifests were generated.
