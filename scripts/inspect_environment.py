"""Record the execution host once. No installs, downloads or board access."""
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/environment.json")
    parser.add_argument("--refresh", action="store_true", help="Replace the record after a relevant host/environment change")
    args = parser.parse_args()
    if args.output.exists() and not args.refresh:
        print(f"Existing record preserved: {args.output}. Use --refresh only after a relevant change.")
        return
    packages = {}
    for name in ("torch", "transformers", "datasets", "numpy", "brevitas", "onnxruntime"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    commands = {name: shutil.which(name) for name in
                ("git", "nvidia-smi", "vivado", "vitis", "vitis_hls", "v++", "hw_server", "wsl")}
    gpu = {"status": "nvidia-smi not on PATH", "output": None}
    if commands["nvidia-smi"]:
        try:
            result = subprocess.run([commands["nvidia-smi"],
                "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15)
            gpu = {"returncode": result.returncode, "output": result.stdout.strip(),
                   "error": result.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as exc:
            gpu = {"status": "query failed", "error": str(exc)}
    record = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "machine": platform.machine(),
              "python": sys.version, "python_executable": sys.executable,
              "packages": packages, "commands_on_path": commands, "gpu": gpu,
              "fpga_part": None, "fpga_interface": None, "vendor_tool_versions": None,
              "note": "PATH discovery only; no CUDA execution, tool license or board validation."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Environment record written: {args.output}")


if __name__ == "__main__":
    main()
