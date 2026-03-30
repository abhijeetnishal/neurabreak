"""Compute profiler for NeuraBreak — measures CPU, RAM, GPU during inference.

Benchmarks the active model variant and compares nano vs small, helping you
pick the right YOLO26 size for your hardware.

Usage:
    # Basic profile with current model (yolo26s.pt)
    python tools/profile_compute.py

    # Compare nano vs small
    python tools/profile_compute.py --compare

    # Profile a specific model file
    python tools/profile_compute.py --model yolo26s.pt --runs 100
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Ensure src/ is on the path so neurabreak imports work when run directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

try:
    import psutil
except ImportError:
    print("ERROR: psutil not installed. Run: uv sync --extra system")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not installed. Run: uv sync --extra ai")
    sys.exit(1)


# Hardware helpers

def _gpu_query() -> dict | None:
    """Query NVIDIA GPU stats via nvidia-smi. Returns None if unavailable."""
    fields = [
        "name",
        "memory.total",
        "memory.used",
        "memory.free",
        "utilization.gpu",
        "utilization.memory",
        "temperature.gpu",
        "power.draw",
    ]
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < len(fields):
            return None
        return {
            "name": parts[0],
            "vram_total_mb": float(parts[1]),
            "vram_used_mb": float(parts[2]),
            "vram_free_mb": float(parts[3]),
            "gpu_util_pct": float(parts[4]),
            "mem_util_pct": float(parts[5]),
            "temp_c": float(parts[6]),
            "power_w": float(parts[7]) if parts[7] not in ("N/A", "[N/A]") else None,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def _torch_cuda_status() -> tuple[bool, str]:
    """Returns (cuda_available, info_string)."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            return True, f"{name} ({vram:.1f} GB VRAM)"
        elif torch.version.cuda is None:
            return False, "CPU-only build of PyTorch installed (CUDA support missing)"
        else:
            return False, f"PyTorch built for CUDA {torch.version.cuda} but driver/device unavailable"
    except ImportError:
        return False, "PyTorch not installed"


def _system_snapshot() -> dict:
    """Collect a full system snapshot."""
    mem = psutil.virtual_memory()
    cpu_freq = psutil.cpu_freq()
    return {
        "cpu_name": platform.processor(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "cpu_freq_max_ghz": (cpu_freq.max / 1000) if cpu_freq else None,
        "ram_total_gb": mem.total / 1e9,
        "ram_available_gb": mem.available / 1e9,
        "ram_used_pct": mem.percent,
        "os": f"{platform.system()} {platform.version()[:40]}",
        "python": platform.python_version(),
    }


# Resource monitor (runs in a background thread during inference)

@dataclass
class ResourceSample:
    ts: float
    cpu_pct: float
    ram_used_mb: float
    gpu_util_pct: float | None = None
    vram_used_mb: float | None = None
    gpu_temp_c: float | None = None


class ResourceMonitor:
    """Samples CPU/RAM/GPU at a fixed interval in a daemon thread."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval = interval_s
        self.samples: list[ResourceSample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        proc = psutil.Process(os.getpid())
        while not self._stop.is_set():
            try:
                cpu = proc.cpu_percent(interval=None)
                ram = proc.memory_info().rss / 1e6
                gpu = _gpu_query()
                self.samples.append(ResourceSample(
                    ts=time.perf_counter(),
                    cpu_pct=cpu,
                    ram_used_mb=ram,
                    gpu_util_pct=gpu["gpu_util_pct"] if gpu else None,
                    vram_used_mb=gpu["vram_used_mb"] if gpu else None,
                    gpu_temp_c=gpu["temp_c"] if gpu else None,
                ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            self._stop.wait(self.interval)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        cpu = [s.cpu_pct for s in self.samples]
        ram = [s.ram_used_mb for s in self.samples]
        gpu = [s.gpu_util_pct for s in self.samples if s.gpu_util_pct is not None]
        vram = [s.vram_used_mb for s in self.samples if s.vram_used_mb is not None]
        temp = [s.gpu_temp_c for s in self.samples if s.gpu_temp_c is not None]

        def _stats(vals: list[float]) -> dict:
            if not vals:
                return {}
            return {
                "avg": sum(vals) / len(vals),
                "max": max(vals),
                "min": min(vals),
            }

        return {
            "cpu": _stats(cpu),
            "ram_mb": _stats(ram),
            "gpu_util": _stats(gpu) if gpu else None,
            "vram_mb": _stats(vram) if vram else None,
            "gpu_temp_c": _stats(temp) if temp else None,
        }


# Inference benchmark

@dataclass
class BenchmarkResult:
    model_path: str
    model_size_mb: float
    runs: int
    warmup: int
    latencies_ms: list[float] = field(default_factory=list)
    load_time_ms: float = 0.0
    ram_before_mb: float = 0.0
    ram_after_mb: float = 0.0
    resources: dict = field(default_factory=dict)
    device: str = "cpu"
    error: str = ""

    @property
    def avg_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p50_ms(self) -> float:
        s = sorted(self.latencies_ms)
        return s[len(s) // 2] if s else 0.0

    @property
    def p95_ms(self) -> float:
        s = sorted(self.latencies_ms)
        return s[int(0.95 * len(s))] if s else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def fps(self) -> float:
        return (1000.0 / self.avg_ms) if self.avg_ms > 0 else 0.0


def _file_size_mb(path: str) -> float:
    try:
        return Path(path).stat().st_size / 1e6
    except FileNotFoundError:
        return 0.0


def run_benchmark(
    model_path: str,
    runs: int = 100,
    warmup: int = 15,
    img_size: int = 640,
) -> BenchmarkResult:
    """Benchmark a YOLO model. Returns a BenchmarkResult."""
    result = BenchmarkResult(
        model_path=model_path,
        model_size_mb=_file_size_mb(model_path),
        runs=runs,
        warmup=warmup,
    )

    if not Path(model_path).exists():
        result.error = f"Model file not found: {model_path}"
        return result

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        result.error = f"Import error: {e}. Run: uv sync --extra ai"
        return result

    # Pick best available device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result.device = device

    proc = psutil.Process(os.getpid())
    result.ram_before_mb = proc.memory_info().rss / 1e6

    # Load model
    t_load = time.perf_counter()
    try:
        model = YOLO(model_path)
        model.to(device)
    except Exception as e:
        result.error = f"Model load failed: {e}"
        return result
    result.load_time_ms = (time.perf_counter() - t_load) * 1000

    result.ram_after_mb = proc.memory_info().rss / 1e6

    # Dummy frame — BGR uint8 like a real webcam frame
    dummy_frame = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)

    # Start resource monitor
    monitor = ResourceMonitor(interval_s=0.25)
    monitor.start()

    # Warmup
    for _ in range(warmup):
        model(dummy_frame, verbose=False, conf=0.25)

    if device == "cuda":
        import torch
        torch.cuda.synchronize()

    # Timed runs
    latencies: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(dummy_frame, verbose=False, conf=0.25)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    monitor.stop()
    result.latencies_ms = latencies
    result.resources = monitor.summary()

    return result


# Reporting

SEP = "─" * 72


def _pct_bar(pct: float, width: int = 30) -> str:
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def print_system_info(snap: dict, cuda_ok: bool, cuda_info: str, gpu: dict | None) -> None:
    print(f"\n{'═' * 72}")
    print("  SYSTEM HARDWARE REPORT — NeuraBreak Compute Profiler")
    print(f"{'═' * 72}")
    print(f"  OS          : {snap['os']}")
    print(f"  Python      : {snap['python']}")
    print(f"  CPU         : {snap['cpu_name']}")
    print(f"  CPU Cores   : {snap['cpu_cores_physical']} physical / {snap['cpu_cores_logical']} logical", end="")
    if snap["cpu_freq_max_ghz"]:
        print(f"  @ {snap['cpu_freq_max_ghz']:.2f} GHz max", end="")
    print()
    print(f"  RAM Total   : {snap['ram_total_gb']:.1f} GB")
    print(f"  RAM Free    : {snap['ram_available_gb']:.1f} GB  ({100 - snap['ram_used_pct']:.0f}% available)")
    print()
    if gpu:
        print(f"  GPU         : {gpu['name']}")
        print(f"  VRAM Total  : {gpu['vram_total_mb'] / 1024:.1f} GB")
        print(f"  VRAM Used   : {gpu['vram_used_mb']:.0f} MB  ({gpu['vram_used_mb']/gpu['vram_total_mb']*100:.1f}%)")
        print(f"  GPU Temp    : {gpu['temp_c']:.0f} °C")
        if gpu["power_w"]:
            print(f"  GPU Power   : {gpu['power_w']:.1f} W")
    else:
        print("  GPU         : No NVIDIA GPU detected (or nvidia-smi unavailable)")

    print()
    status_icon = "✓" if cuda_ok else "✗"
    print(f"  PyTorch CUDA: [{status_icon}] {cuda_info}")
    print(SEP)


def print_benchmark(result: BenchmarkResult, label: str = "") -> None:
    name = label or Path(result.model_path).name
    print(f"\n  MODEL: {name}")
    print(f"  File : {result.model_path}  ({result.model_size_mb:.1f} MB)")
    print(f"  Device: {result.device.upper()}")

    if result.error:
        print(f"  ERROR: {result.error}")
        return

    print()
    print(f"  Load time  : {result.load_time_ms:.0f} ms")
    print(f"  RAM delta  : +{result.ram_after_mb - result.ram_before_mb:.0f} MB  "
          f"(process now {result.ram_after_mb:.0f} MB)")
    print()
    print(f"  ── Latency ({result.runs} runs, img {640}×{640})")
    print(f"  Average    : {result.avg_ms:7.1f} ms   →  {result.fps:.1f} FPS")
    print(f"  Median P50 : {result.p50_ms:7.1f} ms")
    print(f"  P95        : {result.p95_ms:7.1f} ms")
    print(f"  Max        : {result.max_ms:7.1f} ms")

    res = result.resources
    if res:
        print()
        print(" Resource Usage During Inference")
        cpu = res.get("cpu", {})
        if cpu:
            bar = _pct_bar(cpu.get("avg", 0))
            print(f"  CPU avg    : {cpu.get('avg', 0):5.1f}%  [{bar}]  max {cpu.get('max', 0):.1f}%")
        ram = res.get("ram_mb", {})
        if ram:
            print(f"  RAM (proc) : {ram.get('avg', 0):.0f} MB avg  /  {ram.get('max', 0):.0f} MB peak")
        gpu_u = res.get("gpu_util", {})
        if gpu_u:
            bar = _pct_bar(gpu_u.get("avg", 0))
            print(f"  GPU util   : {gpu_u.get('avg', 0):5.1f}%  [{bar}]  max {gpu_u.get('max', 0):.1f}%")
        vram = res.get("vram_mb", {})
        if vram:
            print(f"  VRAM used  : {vram.get('avg', 0):.0f} MB avg  /  {vram.get('max', 0):.0f} MB peak")
        temp = res.get("gpu_temp_c", {})
        if temp:
            print(f"  GPU temp   : {temp.get('avg', 0):.0f} °C avg  /  {temp.get('max', 0):.0f} °C peak")


def print_comparison(results: list[BenchmarkResult], labels: list[str]) -> None:
    if len(results) < 2:
        return
    valid = [(r, l) for r, l in zip(results, labels) if not r.error and r.avg_ms > 0]
    if len(valid) < 2:
        return

    print(f"\n{'═' * 72}")
    print("  SIDE-BY-SIDE COMPARISON")
    print(SEP)
    print(f"  {'Metric':<28} {'':>20}  {'':>20}")
    header = f"  {'Metric':<28} {valid[0][1]:>20}  {valid[1][1]:>20}"
    print(header)
    print(SEP)

    def row(label: str, v1: str, v2: str) -> None:
        print(f"  {label:<28} {v1:>20}  {v2:>20}")

    r0, _ = valid[0]
    r1, _ = valid[1]

    row("Model size (MB)", f"{r0.model_size_mb:.1f}", f"{r1.model_size_mb:.1f}")
    row("Device", r0.device.upper(), r1.device.upper())
    row("RAM delta (MB)", f"+{r0.ram_after_mb - r0.ram_before_mb:.0f}", f"+{r1.ram_after_mb - r1.ram_before_mb:.0f}")
    row("Load time (ms)", f"{r0.load_time_ms:.0f}", f"{r1.load_time_ms:.0f}")
    row("Avg latency (ms)", f"{r0.avg_ms:.1f}", f"{r1.avg_ms:.1f}")
    row("Throughput (FPS)", f"{r0.fps:.1f}", f"{r1.fps:.1f}")
    row("P95 latency (ms)", f"{r0.p95_ms:.1f}", f"{r1.p95_ms:.1f}")

    cpu0 = r0.resources.get("cpu", {})
    cpu1 = r1.resources.get("cpu", {})
    if cpu0 and cpu1:
        row("CPU avg (%)", f"{cpu0.get('avg', 0):.1f}", f"{cpu1.get('avg', 0):.1f}")

    vram0 = r0.resources.get("vram_mb", {})
    vram1 = r1.resources.get("vram_mb", {})
    if vram0 and vram1:
        row("VRAM used (MB)", f"{vram0.get('avg', 0):.0f}", f"{vram1.get('avg', 0):.0f}")

    print(SEP)

    # Speedup
    speedup = r1.avg_ms / r0.avg_ms if r0.avg_ms > 0 else 1.0
    if speedup > 1:
        winner = valid[0][1]
        factor = speedup
    else:
        winner = valid[1][1]
        factor = 1.0 / speedup

    print(f"  WINNER (speed): {winner}  is  {factor:.1f}×  faster")


def print_recommendations(results: list[BenchmarkResult], cuda_ok: bool, gpu: dict | None) -> None:
    print(f"\n{'═' * 72}")
    print("  OPTIMIZATION RECOMMENDATIONS")
    print(f"{'═' * 72}")

    recs: list[tuple[str, str, str]] = []  # (priority, title, detail)

    # 1. CUDA / GPU
    if gpu and not cuda_ok:
        recs.append((
            "CRITICAL",
            "Install CUDA-enabled PyTorch",
            textwrap.dedent(f"""\
                Your RTX 4060 Laptop GPU ({gpu['vram_total_mb']/1024:.0f} GB VRAM) is being WASTED.
                Torch is CPU-only.  Install the CUDA build:

                  # Uninstall CPU torch first:
                  .venv\\Scripts\\pip.exe uninstall torch torchvision -y

                  # Install CUDA 12.x build (matches driver 595.71):
                  .venv\\Scripts\\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu124

                Expected improvement: 10-40× faster inference (ms → sub-ms on GPU).
                GPU YOLO26n can do 200+ FPS; YOLO26s 100+ FPS — far beyond what
                NeuraBreak needs at 5 FPS target."""
            ),
        ))

    # 2. Model selection
    nano_res = next((r for r in results if "nano" in Path(r.model_path).stem.lower() or "n.pt" in r.model_path), None)
    small_res = next((r for r in results if "small" in Path(r.model_path).stem.lower() or "s.pt" in r.model_path or "26s" in r.model_path), None)

    if cuda_ok or (gpu and not cuda_ok):
        recs.append((
            "HIGH",
            "Use YOLO26n on GPU (recommended),  YOLO26s acceptable",
            textwrap.dedent("""\
                With CUDA enabled on your RTX 4060:
                  • YOLO26n: ~2-4 ms/frame  →  GPU barely stressed  →  best choice
                  • YOLO26s: ~5-10 ms/frame  →  still 100+ FPS  →  slightly more accurate
                  • YOLO26m: overkill for a 5 FPS app; wastes VRAM for no benefit

                For NeuraBreak's 5 FPS polling rate, YOLO26n on GPU is ideal:
                  config.toml → [detection]
                    model_variant = "nano"
                    fps = 5   # 5 FPS is plenty for posture monitoring"""),
        ))
    else:
        recs.append((
            "HIGH",
            "Use YOLO26n on CPU (until CUDA is fixed)",
            textwrap.dedent("""\
                Without GPU, YOLO26n uses ~2× less RAM and runs ~2× faster than YOLO26s.
                Typical CPU results:
                  • YOLO26n: 60-120 ms/frame  →  8-17 FPS headroom
                  • YOLO26s: 100-200 ms/frame  →  5-10 FPS headroom (tight)
                At 5 FPS polling, YOLO26s works but hogs 1-2 CPU cores.
                  config.toml → [detection]
                    model_variant = "nano"
                    fps = 3   # lower to 3 FPS on CPU to ease the load"""),
        ))

    # 3. ONNX export
    recs.append((
        "HIGH",
        "Export model to ONNX / TensorRT for faster CPU+GPU runtime",
        textwrap.dedent("""\
            ONNX Runtime is already in your deps (onnxruntime>=1.17.0).
            Export once, run forever — no ultralytics overhead at runtime:

              # Export from Python after training:
              from ultralytics import YOLO
              model = YOLO("yolo26n.pt")
              model.export(format="onnx", imgsz=320, half=False, simplify=True)
              # → saves yolo26n.onnx

              # For TensorRT (NVIDIA GPU, maximum speed):
              model.export(format="engine", imgsz=320, half=True)
              # → saves yolo26n.engine  (FP16, runs at ~1 ms/frame on RTX 4060)

            Update config.toml:
              model_path = "~/.neurabreak/models/yolo26n.onnx"
              model_variant = "nano"

            Expected gain: 1.5–3× faster than PyTorch .pt on CPU;
                           3–8× faster than .pt on GPU with TensorRT."""),
    ))

    # 4. Input resolution
    recs.append((
        "MEDIUM",
        "Reduce inference resolution from 640 → 320 px",
        textwrap.dedent("""\
            YOLO's default input is 640×640.  For posture detection (large objects,
            person fills most of the frame) 320×320 is sufficient and is ~4× cheaper:

              model.export(format="onnx", imgsz=320, simplify=True)

            In engine.py, pass imgsz to model() call:
              results = self._model(frame, verbose=False, conf=yolo_conf, imgsz=320)

            Expected: 2-4× faster, ~25% less RAM on CPU; imperceptible accuracy drop
            for near-field posture classification."""),
    ))

    # 5. FPS / frame skipping
    recs.append((
        "MEDIUM",
        "Cap capture FPS and skip redundant frames",
        textwrap.dedent("""\
            NeuraBreak polls at 5 FPS (config default).  Posture doesn't change
            in milliseconds — consider:
              • 3 FPS on CPU (saves ~40% CPU vs 5 FPS)
              • 5 FPS on GPU (GPU is idle between frames anyway)

            Additionally, implement frame-skip when the previous and current frame
            are visually identical (delta < threshold) using:
              diff = cv2.absdiff(prev_gray, gray)
              if diff.mean() < SKIP_THRESHOLD: continue  # skip inference

            This can halve inference calls when the user is stationary."""),
    ))

    # 6. Half-precision
    recs.append((
        "MEDIUM",
        "Enable FP16 (half-precision) inference on GPU",
        textwrap.dedent("""\
            RTX 4060 has dedicated Tensor Cores for FP16:

              model.export(format="engine", half=True, imgsz=320)

            Or with PyTorch + CUDA:
              model.half()  # convert weights to fp16
              # inference with fp16 frame:
              frame_t = torch.from_numpy(frame).half().cuda() / 255.0

            Expected: ~2× faster GPU inference, ~50% VRAM reduction.
            Accuracy drop: negligible for posture classification."""),
    ))

    # 7. Thread + memory
    recs.append((
        "LOW",
        "Limit PyTorch CPU thread count",
        textwrap.dedent("""\
            By default PyTorch spawns threads equal to CPU core count.
            With 16 physical cores this causes thread contention with Qt.

            Add to app.py startup (before YOLO loads):
              import torch
              torch.set_num_threads(4)        # limit inference threads
              torch.set_num_interop_threads(2)

            This reduces CPU context switching and lowers overall CPU usage by
            10-20% while keeping throughput sufficient for 5 FPS inference."""),
    ))

    recs.append((
        "LOW",
        "Disable YOLO telemetry and verbose output",
        textwrap.dedent("""\
            Ultralytics logs and checks updates on every run.  Silence them:

              from ultralytics.utils import SETTINGS
              SETTINGS.update({"sync": False})   # no telemetry

              # Already set in engine.py: verbose=False ✓
              # Also set in environment:
              os.environ["YOLO_VERBOSE"] = "False"

            Saves ~5-10 ms per startup, eliminates network calls."""),
    ))

    # Print
    icons = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
    for i, (priority, title, detail) in enumerate(recs, 1):
        icon = icons.get(priority, "  ")
        print(f"\n  {icon} [{priority}] {i}. {title}")
        for line in detail.splitlines():
            print(f"       {line}")

    print(f"\n{'═' * 72}")
    print("  QUICK-WIN SUMMARY (apply in this order for max effect)")
    print(SEP)
    print("  1. Install CUDA torch  → 20× speedup, frees all CPU cores")
    print("  2. Use YOLO26n model   → 2× smaller, faster, less RAM")
    print("  3. Export to ONNX/TRT  → 2-8× additional speedup")
    print("  4. Set imgsz=320       → 4× cheaper compute per frame")
    print("  5. Set fps=3 on CPU    → -40% CPU when on battery")
    print("  6. torch.set_num_threads(4)  → less Qt contention")
    print(f"{'═' * 72}\n")


# CLI entry point

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NeuraBreak compute profiler — CPU/RAM/GPU benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="Model file to benchmark (default: yolo26n.pt)",
    )
    parser.add_argument(
        "--runs", type=int, default=80,
        help="Number of timed inference runs (default: 80)",
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Number of warmup runs not counted in timing (default: 10)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare nano vs small models side by side",
    )
    parser.add_argument(
        "--sysinfo-only", action="store_true",
        help="Only print hardware info, skip benchmarking",
    )
    args = parser.parse_args()

    # system info
    snap = _system_snapshot()
    cuda_ok, cuda_info = _torch_cuda_status()
    gpu = _gpu_query()

    print_system_info(snap, cuda_ok, cuda_info, gpu)

    if args.sysinfo_only:
        return

    results: list[BenchmarkResult] = []
    labels: list[str] = []

    if args.compare:
        # Try to find nano and small models
        nano_candidates = ["yolo26n.pt", "models/yolo26n.pt"]
        small_candidates = [args.model, "yolo26s.pt", "models/yolo26s.pt"]

        nano_path = next((p for p in nano_candidates if Path(p).exists()), nano_candidates[0])
        small_path = next((p for p in small_candidates if Path(p).exists()), small_candidates[0])

        print(f"\n  Benchmarking YOLO26n: {nano_path}  ({args.runs} runs) ...")
        r_nano = run_benchmark(nano_path, runs=args.runs, warmup=args.warmup)
        print_benchmark(r_nano, "YOLO26n (nano)")
        results.append(r_nano)
        labels.append("YOLO26n")

        print(f"\n{SEP}")
        print(f"\n  Benchmarking YOLO26s: {small_path}  ({args.runs} runs) ...")
        r_small = run_benchmark(small_path, runs=args.runs, warmup=args.warmup)
        print_benchmark(r_small, "YOLO26s (small)")
        results.append(r_small)
        labels.append("YOLO26s")

        print_comparison(results, labels)
    else:
        model_path = args.model
        if not Path(model_path).exists():
            # Try common locations
            for candidate in ["yolo26n.pt", "models/yolo26n.pt", "yolo11n.pt"]:
                if Path(candidate).exists():
                    model_path = candidate
                    break

        print(f"\n  Benchmarking: {model_path}  ({args.runs} runs) ...")
        r = run_benchmark(model_path, runs=args.runs, warmup=args.warmup)
        print_benchmark(r)
        results.append(r)
        labels.append(Path(model_path).stem)

    print_recommendations(results, cuda_ok, gpu)


if __name__ == "__main__":
    main()
