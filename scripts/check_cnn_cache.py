"""Serial CUDA cache-limit diagnostic; parent uses only Python standard library.

Root runs this between GPU jobs. Two fresh children compare cuDNN's unset
default cache limit with TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=64. No optimizer,
training fits, scoring partitions, model changes or registry edits are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results/cnn_study_v1"
CONFIG_IDS = ("textcnn_01", "residual_01", "dpcnn_01", "byte_hierarchy_01")
BATCH_SIZES = (1, 2, 3, 5, 8, 13, 21)


def save_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def memory():
    if os.name != "nt":
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        available = None
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
        return {"peak_process_working_set_bytes": peak, "available_physical_bytes": available}
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
    class Status(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
            "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
    status = Status(); status.dwLength = ctypes.sizeof(status)
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"process_working_set_bytes": counters.WorkingSetSize, "peak_process_working_set_bytes": counters.PeakWorkingSetSize,
            "process_private_bytes": counters.PrivateUsage, "available_physical_bytes": status.ullAvailPhys}


def sample_inputs(rep, np):
    folder = DEST / "prepared" / rep / "fit"
    docs = [json.loads(line) for line in (folder / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
    arrays = {name: np.load(folder / (name + ".npy"), mmap_mode="r", allow_pickle=False)
              for name in ("ids", "masks", "labels", "starts")}
    selected, indices = [], []
    for label, longest in ((0, False), (1, True), (0, True), (1, False)):
        choices = sorted((i for i, d in enumerate(docs) if d["label"] == label),
                         key=lambda i: (docs[i]["content_length"], docs[i]["document_id"]))
        index = choices[-1] if longest else choices[0]
        lo, hi = map(int, arrays["starts"][index:index + 2])
        windows = sorted({lo, hi - 1})
        indices.extend(windows)
        selected.append({"document_id": docs[index]["document_id"], "text_sha256": docs[index]["text_sha256"],
                         "document_windows": hi - lo, "selected_window_offsets": [w - lo for w in windows]})
    values = {name: np.asarray(arrays[name][indices], dtype=np.int64).copy() for name in ("ids", "masks", "labels")}
    return values, {"fold": "fit", "representation": rep, "documents": selected,
                    "selected_window_indices": indices, "array_sha256_int64": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in values.items()}}


def worker(args):
    # Environment is set by the standard-library parent before this import.
    import numpy as np
    import torch
    from torch.nn import functional as F
    from cnn_models import build_model, load_search_configs
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; this diagnostic must run in root's GPU queue")
    start = time.monotonic()
    configs = {c["id"]: c for c in load_search_configs()}
    samples = {rep: sample_inputs(rep, np) for rep in ("token", "byte")}
    results = []
    for config_id in CONFIG_IDS:
        config = configs[config_id]
        torch.manual_seed(814); torch.cuda.manual_seed_all(814)
        model = build_model(config, 257 if config["representation"] == "byte" else 30522, 0).cuda().eval()
        arrays, input_identity = samples[config["representation"]]
        tensors = {k: torch.from_numpy(v).cuda() for k, v in arrays.items()}
        parameter_names = [name for name, _ in model.named_parameters()]
        for batch in BATCH_SIZES:
            if time.monotonic() - start > args.worker_seconds:
                raise RuntimeError("Diagnostic child runtime ceiling reached")
            observed = memory()
            if observed["available_physical_bytes"] < 3 * 1024**3:
                raise RuntimeError("Diagnostic physical memory floor reached")
            torch.cuda.reset_peak_memory_stats()
            index = torch.arange(batch, device="cuda") % len(tensors["ids"])
            model.zero_grad(set_to_none=True)
            logits = model(tensors["ids"][index], tensors["masks"][index]).logits
            loss = F.cross_entropy(logits, tensors["labels"][index])
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError("Nonfinite diagnostic output")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("Nonfinite or missing full gradient")
            if torch.count_nonzero(model.embedding.weight.grad[0]):
                raise RuntimeError("Padding embedding received gradient")
            torch.cuda.synchronize()
            recovered = {"logits": logits.detach().cpu().numpy(), "loss": loss.detach().cpu().numpy()}
            recovered.update({f"gradient_{i:03d}": p.grad.detach().cpu().numpy()
                              for i, p in enumerate(model.parameters())})
            output = args.folder / f"{config_id}_b{batch}.npz"
            np.savez_compressed(output, **recovered)
            results.append({"config_id": config_id, "batch_size": batch, "input_identity": input_identity,
                            "parameter_names": parameter_names, "tensor_path": str(output.relative_to(ROOT)),
                            "file_sha256": digest(output),
                            "tensor_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in recovered.items()},
                            "memory": {**memory(), "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                       "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()}})
            # Preserve identity and parameter-name mapping even if a later case
            # reaches the floor or the parent stops this bounded child.
            save_json(args.folder / "manifest.json", {"complete": False, "results": results,
                       "cache_environment": os.environ.get("TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT"),
                       "wall_seconds": time.monotonic() - start})
        del model, tensors
        torch.cuda.empty_cache()
    result = {"complete": True, "torch": torch.__version__, "cudnn_version": torch.backends.cudnn.version(), "device": torch.cuda.get_device_name(),
              "cache_environment": os.environ.get("TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT"),
              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "seed": 814,
              "dropout_disabled": True, "tf32_disabled": True, "deterministic_algorithms": True,
              "cudnn_benchmark": False, "batch_sizes": list(BATCH_SIZES), "results": results,
              "wall_seconds": time.monotonic() - start, "final_memory": memory()}
    save_json(args.folder / "manifest.json", result)
    print(json.dumps({"cache": result["cache_environment"], "cases": len(results), "wall_seconds": result["wall_seconds"]}))


def compare(args):
    import numpy as np
    manifests = [json.loads((args.artifacts / mode / "manifest.json").read_text()) for mode in ("default", "limit64")]
    checks = []
    for left, right in zip(manifests[0]["results"], manifests[1]["results"], strict=True):
        assert (left["config_id"], left["batch_size"], left["input_identity"], left["parameter_names"]) == (
            right["config_id"], right["batch_size"], right["input_identity"], right["parameter_names"])
        all_equal = left["tensor_sha256"] == right["tensor_sha256"]
        deltas = {}
        with np.load(ROOT / left["tensor_path"], allow_pickle=False) as a, np.load(ROOT / right["tensor_path"], allow_pickle=False) as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                assert a[key].shape == b[key].shape and a[key].dtype == b[key].dtype
                if left["tensor_sha256"][key] != right["tensor_sha256"][key]:
                    deltas[key] = float(np.max(np.abs(a[key].astype(np.float64) - b[key].astype(np.float64))))
        checks.append({"config_id": left["config_id"], "batch_size": left["batch_size"], "all_tensors_bitwise_equal": all_equal,
                       "unequal_tensor_max_absolute_differences": deltas,
                       "default_memory": left["memory"], "limit64_memory": right["memory"]})
    report = {"complete": True, "all_tensors_bitwise_equal": all(c["all_tensors_bitwise_equal"] for c in checks),
              "checks": checks, "children": manifests,
              "evidence_boundary": "Same initial models and permitted real-fit inputs; forward, CE and full backward, no optimizer/training/scoring. Numerical cache comparison only on recorded shapes.",
              "memory_boundary": "Process working-set/physical availability and CUDA allocator peaks; Windows process peak cumulative per fresh child. No attribution of all RSS differences to cuDNN cache.",
              "artifact_encoding": "Compressed NumPy archives retain logits, scalar loss and every named parameter gradient; per-tensor and file SHA256 in child manifests.",
              "source_sha256": digest(Path(__file__)), "model_source_sha256": digest(ROOT / "scripts/cnn_models.py")}
    save_json(DEST / "cudnn_cache_check.json", report)
    print(json.dumps({"complete": True, "all_tensors_bitwise_equal": report["all_tensors_bitwise_equal"], "cases": len(checks)}))
    if not report["all_tensors_bitwise_equal"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=("worker", "compare"))
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--artifacts", type=Path, default=DEST / "cudnn_cache_artifacts")
    parser.add_argument("--max-seconds", type=int, default=120)
    parser.add_argument("--worker-seconds", type=int, default=50)
    args = parser.parse_args()
    if args.child == "worker":
        worker(args); return
    if args.child == "compare":
        compare(args); return
    if not 1 <= args.max_seconds <= 120 or not 1 <= args.worker_seconds <= 50:
        raise ValueError("Diagnostic is bounded by 120 total seconds, 50 per GPU child")
    started = time.monotonic()
    logs = []
    try:
        for mode, cache in (("default", None), ("limit64", "64")):
            folder = args.artifacts / mode
            folder.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy(); env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            if cache is None:
                env.pop("TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT", None)
            else:
                env["TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT"] = cache
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--child", "worker", "--folder", str(folder),
                       "--worker-seconds", str(args.worker_seconds)]
            result = subprocess.run(command, env=env, capture_output=True, text=True,
                                    timeout=max(1, args.max_seconds - (time.monotonic() - started)),
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            logs.append({"mode": mode, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
            if result.returncode:
                raise RuntimeError(f"{mode} child failed; see dispatch log")
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--child", "compare", "--artifacts", str(args.artifacts)]
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=max(1, args.max_seconds - (time.monotonic() - started)),
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        logs.append({"mode": "cpu_comparison", "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            raise RuntimeError("Cache arithmetic comparison failed; inspect saved tensors and report")
    finally:
        save_json(DEST / "cudnn_cache_dispatch.json", {"command": sys.argv, "wall_seconds": time.monotonic() - started, "children": logs})
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
