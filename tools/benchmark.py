"""Inference speed benchmark — measures latency on your hardware.

Supports both ONNX (via onnxruntime) and PyTorch .pt (via ultralytics).

Usage:
    # ONNX model:
    python tools/benchmark.py --model models/neurabreak_v1.onnx --runs 200

    # PyTorch .pt model:
    python tools/benchmark.py --model yolo26n.pt --runs 100

    # Quick sysinfo only:
    python tools/benchmark.py --sysinfo-only
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not installed. Run: uv sync --extra ai")
    raise



def _best_onnx_providers() -> list[str]:
    """Same priority list as engine._best_onnx_providers — allows standalone use."""
    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
        ordered = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        providers = [p for p in ordered if p in available]
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        return providers
    except ImportError:
        return ["CPUExecutionProvider"]


def _select_torch_device() -> str:
    """Pick best PyTorch device: CUDA > MPS > CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def benchmark_onnx(model_path: str, runs: int = 200, warmup: int = 10, imgsz: int = 320) -> None:
    """Benchmark an ONNX model with onnxruntime (CUDA / DirectML / CoreML / CPU)."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed. Run: uv sync --extra ai")
        return

    providers = _best_onnx_providers()
    session = ort.InferenceSession(model_path, providers=providers)
    active_provider = session.get_providers()[0]
    print(f"  Provider: {active_provider}")
    print(f"  All available: {providers}")

    inp = session.get_inputs()[0]
    dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

    print(f"  Input shape: {inp.shape}  dtype: {inp.type}")
    print(f"  Warming up ({warmup} runs)...", end=" ", flush=True)
    for _ in range(warmup):
        session.run(None, {inp.name: dummy})
    print("done")

    print(f"  Timing {runs} runs...", end=" ", flush=True)
    latencies: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {inp.name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)
    print("done")

    avg = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p95 = sorted(latencies)[int(0.95 * len(latencies))]
    fps = 1000.0 / avg if avg > 0 else 0.0

    print()
    print(f"  Model   : {model_path}  ({Path(model_path).stat().st_size / 1e6:.1f} MB)")
    print(f"  Device  : {active_provider.replace('ExecutionProvider', '')}")
    print(f"  Avg     : {avg:.1f} ms  →  {fps:.1f} FPS")
    print(f"  P50     : {p50:.1f} ms")
    print(f"  P95     : {p95:.1f} ms")
    print(f"  Max     : {max(latencies):.1f} ms")


def benchmark_pt(model_path: str, runs: int = 100, warmup: int = 10, imgsz: int = 320) -> None:
    """Benchmark a PyTorch .pt model — uses best available device (CUDA > MPS > CPU)."""
    try:
        import torch
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics/torch not installed. Run: uv sync --extra ai")
        return

    device = _select_torch_device()
    print(f"  Device  : {device.upper()}")
    if device == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
    elif device == "mps":
        print(f"  GPU     : Apple Silicon MPS")

    model = YOLO(model_path)
    if device in ("cuda", "mps"):
        model.to(device)
    dummy = np.random.randint(0, 256, (imgsz, imgsz, 3), dtype=np.uint8)

    print(f"  Warming up ({warmup} runs)...", end=" ", flush=True)
    for _ in range(warmup):
        model(dummy, verbose=False, conf=0.25, imgsz=imgsz, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    print("done")

    print(f"  Timing {runs} runs...", end=" ", flush=True)
    latencies: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(dummy, verbose=False, conf=0.25, imgsz=imgsz, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)
    print("done")

    avg = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p95 = sorted(latencies)[int(0.95 * len(latencies))]
    fps = 1000.0 / avg if avg > 0 else 0.0

    size_mb = Path(model_path).stat().st_size / 1e6
    print()
    print(f"  Model   : {model_path}  ({size_mb:.1f} MB)")
    print(f"  Avg     : {avg:.1f} ms  →  {fps:.1f} FPS")
    print(f"  P50     : {p50:.1f} ms")
    print(f"  P95     : {p95:.1f} ms")
    print(f"  Max     : {max(latencies):.1f} ms")


def sysinfo() -> None:
    """Print system hardware summary including all GPU types."""
    import platform
    import subprocess
    try:
        import psutil
        mem = psutil.virtual_memory()
        cpu_freq = psutil.cpu_freq()
        print(f"  CPU     : {platform.processor()}")
        print(f"  Cores   : {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count()} logical", end="")
        if cpu_freq:
            print(f"  @ {cpu_freq.max/1000:.2f} GHz max", end="")
        print()
        print(f"  RAM     : {mem.total/1e9:.1f} GB total, {mem.available/1e9:.1f} GB free")
    except ImportError:
        print("  psutil not installed")

    # NVIDIA
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            name, mem_total, mem_used, util, temp = r.stdout.strip().split(",")
            print(f"  GPU     : NVIDIA {name.strip()}")
            print(f"  VRAM    : {float(mem_total)/1024:.1f} GB total, {float(mem_used):.0f} MB used")
            print(f"  GPU util: {util.strip()}%   Temp: {temp.strip()} °C")
    except Exception:
        pass

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        print(f"  PyTorch : {torch.__version__}  CUDA={cuda_ok}  MPS={mps_ok}")
        if not cuda_ok and not mps_ok:
            print("  WARN    : torch has no GPU backend — run: python tools/install_gpu_torch.py --run")
    except ImportError:
        print("  PyTorch : not installed")

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"  ORT     : {ort.__version__}  providers={providers}")
    except ImportError:
        print("  ORT     : not installed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX or PyTorch YOLO model inference speed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="", help="Path to .onnx or .pt model file")
    parser.add_argument("--runs", type=int, default=100, help="Number of timed inference runs")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup runs (not counted)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default 640)")
    parser.add_argument("--sysinfo-only", action="store_true", help="Print hardware info and exit")
    args = parser.parse_args()

    print("\nSystem Info")
    sysinfo()

    if args.sysinfo_only:
        return

    model_path = args.model
    if not model_path:
        # Auto-detect available model
        for candidate in ["yolo26n.pt", "yolo26s.pt", "models/neurabreak_v1.onnx", "yolo11n.pt"]:
            if Path(candidate).exists():
                model_path = candidate
                break
        if not model_path:
            print("\nNo model found. Specify with --model <path>")
            print("Train and export first: make train && make export")
            return

    if not Path(model_path).exists():
        print(f"\nModel not found: {model_path}")
        return

    print(f"\nBenchmark: {model_path}")
    suffix = Path(model_path).suffix.lower()
    if suffix == ".onnx":
        benchmark_onnx(model_path, runs=args.runs, warmup=args.warmup, imgsz=args.imgsz)
    else:
        benchmark_pt(model_path, runs=args.runs, warmup=args.warmup, imgsz=args.imgsz)
    print()


if __name__ == "__main__":
    main()
