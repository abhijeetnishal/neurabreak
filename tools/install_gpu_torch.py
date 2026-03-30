"""GPU-aware PyTorch installer for NeuraBreak.

Detects your GPU vendor/driver and prints (or runs) the exact pip command
to install a torch build that uses your GPU:

  NVIDIA (CUDA) → torch + torchvision via PyTorch CUDA index
  AMD / Intel   → onnxruntime-directml  (Windows DirectML, no ROCm needed)
  Apple Silicon → torch is already MPS-capable; just needs a regular install
  No GPU        → CPU-only (already installed by uv sync)

Usage:
    python tools/install_gpu_torch.py          # print command, don't run
    python tools/install_gpu_torch.py --run    # detect + install automatically
    python tools/install_gpu_torch.py --check  # just show current GPU status
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path


# GPU detection helpers

def _nvidia_driver_version() -> str | None:
    """Return NVIDIA driver version string or None."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip().split("\n")[0].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _nvidia_gpu_name() -> str | None:
    """Return first NVIDIA GPU name or None."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip().split("\n")[0].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _cuda_compute_from_driver(driver_ver: str) -> str:
    """Map NVIDIA driver version to the highest supported CUDA toolkit version.

    Based on: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
    Returns e.g. "cu121", "cu124", "cu118".
    """
    try:
        major = int(driver_ver.split(".")[0])
    except (ValueError, IndexError):
        return "cu124"  # safe default — torch cu124 runs on driver >= 525.60

    if major >= 550:
        return "cu124"
    if major >= 525:
        return "cu121"
    if major >= 520:
        return "cu118"
    # Too old for modern torch — can't reliably enable CUDA
    return ""


def _torch_cuda_ok() -> bool:
    """Return True if the installed torch already has CUDA support."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _torch_mps_ok() -> bool:
    """Return True if the installed torch has MPS support (Apple Silicon)."""
    try:
        import torch
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        return False


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _has_directml_ort() -> bool:
    """Return True if onnxruntime-directml is installed and DML provider is available."""
    try:
        import onnxruntime as ort
        return "DmlExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


def _pip() -> str:
    """Return the path to pip inside the current venv."""
    venv = Path(sys.prefix)
    for candidate in [
        venv / "Scripts" / "pip.exe",   # Windows
        venv / "bin" / "pip",            # Unix
    ]:
        if candidate.exists():
            return str(candidate)
    return "pip"


# Recommendation logic

def detect_and_recommend() -> dict:
    """Gather system info and return a recommendation dict."""
    os_name = platform.system()
    rec: dict = {
        "os": os_name,
        "machine": platform.machine(),
        "nvidia_gpu": None,
        "nvidia_driver": None,
        "cuda_tag": None,
        "apple_silicon": _is_apple_silicon(),
        "torch_cuda_ok": _torch_cuda_ok(),
        "torch_mps_ok": _torch_mps_ok(),
        "directml_ok": _has_directml_ort(),
        "action": "none",
        "install_cmd": [],
        "ort_cmd": [],
        "notes": [],
    }

    # NVIDIA
    driver = _nvidia_driver_version()
    if driver:
        rec["nvidia_gpu"] = _nvidia_gpu_name()
        rec["nvidia_driver"] = driver
        cuda_tag = _cuda_compute_from_driver(driver)
        rec["cuda_tag"] = cuda_tag

        if rec["torch_cuda_ok"]:
            rec["action"] = "cuda_ready"
            rec["notes"].append(
                f"CUDA already active. torch.cuda.is_available() = True."
            )
        elif not cuda_tag:
            rec["action"] = "driver_too_old"
            rec["notes"].append(
                f"NVIDIA driver {driver} is too old for modern CUDA torch. "
                "Update your GPU driver to >= 525.60."
            )
        else:
            rec["action"] = "install_cuda_torch"
            index_url = f"https://download.pytorch.org/whl/{cuda_tag}"
            rec["install_cmd"] = [
                _pip(), "install",
                "torch", "torchvision",
                "--index-url", index_url,
            ]
            rec["notes"].append(
                f"Detected NVIDIA {rec['nvidia_gpu']} (driver {driver}). "
                f"Installing PyTorch CUDA {cuda_tag.upper()} build."
            )
            # Also install onnxruntime-gpu for the ONNX path
            rec["ort_cmd"] = [
                _pip(), "install",
                "onnxruntime-gpu",
            ]

    # Apple Silicon
    elif rec["apple_silicon"]:
        if rec["torch_mps_ok"]:
            rec["action"] = "mps_ready"
            rec["notes"].append("Apple MPS already active. torch.backends.mps.is_available() = True.")
        else:
            rec["action"] = "install_mps_torch"
            # Standard pip torch for macOS arm64 includes MPS from torch >= 1.12
            rec["install_cmd"] = [
                _pip(), "install",
                "torch", "torchvision",
            ]
            rec["notes"].append(
                "Apple Silicon detected. Standard torch includes MPS support "
                "(no special index URL needed)."
            )

    # AMD / Intel / Other (Windows DirectML via onnxruntime-directml)
    elif os_name == "Windows":
        if rec["directml_ok"]:
            rec["action"] = "directml_ready"
            rec["notes"].append(
                "onnxruntime-directml already installed. "
                "DirectML provider active for ONNX inference on AMD/Intel GPU."
            )
        else:
            rec["action"] = "install_directml"
            rec["ort_cmd"] = [
                _pip(), "install",
                "onnxruntime-directml",
            ]
            rec["notes"].append(
                "No NVIDIA GPU detected on Windows. "
                "Installing onnxruntime-directml for AMD/Intel iGPU acceleration "
                "via Windows DirectML. Works on any DirectX 12-capable GPU."
            )
    else:
        rec["action"] = "cpu_only"
        rec["notes"].append(
            "No supported GPU detected. Running on CPU. "
            "Consider installing a GPU or enabling WSL2 with CUDA passthrough."
        )

    return rec


# Reporting

SEP = "─" * 60


def print_report(rec: dict) -> None:
    print(f"\n{'═' * 60}")
    print("  NeuraBreak — GPU Torch Installer")
    print(f"{'═' * 60}")
    print(f"  OS       : {rec['os']}  ({rec['machine']})")

    if rec["nvidia_gpu"]:
        print(f"  GPU      : {rec['nvidia_gpu']}")
        print(f"  Driver   : {rec['nvidia_driver']}")
        print(f"  CUDA tag : {rec['cuda_tag'] or 'N/A (driver too old)'}")
    elif rec["apple_silicon"]:
        print(f"  GPU      : Apple Silicon (MPS)")
    else:
        print(f"  GPU      : No NVIDIA/Apple GPU detected")

    print(f"  CUDA ok  : {rec['torch_cuda_ok']}")
    print(f"  MPS ok   : {rec['torch_mps_ok']}")
    print(f"  DirectML : {rec['directml_ok']}")
    print(SEP)

    for note in rec["notes"]:
        print(f"  NOTE: {note}")

    action = rec["action"]

    if action in ("cuda_ready", "mps_ready", "directml_ready"):
        print(f"\n  GPU acceleration is already active. Nothing to install.")

    elif action == "driver_too_old":
        print(f"\n  ACTION REQUIRED: Update your NVIDIA GPU driver.")
        print(f"  https://www.nvidia.com/Download/index.aspx")

    elif action == "cpu_only":
        print(f"\n  Running on CPU. No GPU action available.")

    else:
        # Commands to run
        if rec["install_cmd"]:
            print(f"\n  Run this to enable GPU PyTorch:")
            print(f"    {' '.join(rec['install_cmd'])}")
        if rec["ort_cmd"]:
            print(f"\n  Run this to enable GPU ONNX Runtime:")
            print(f"    {' '.join(rec['ort_cmd'])}")

        print(f"\n  Or use --run to execute automatically.")

    print(f"{'═' * 60}\n")


def run_install(rec: dict) -> None:
    """Execute the install commands."""
    action = rec["action"]

    if action in ("cuda_ready", "mps_ready", "directml_ready"):
        print("  GPU acceleration already active. Nothing to install.")
        return

    if action == "driver_too_old":
        print("  ERROR: Driver too old — update your NVIDIA GPU driver first.")
        sys.exit(1)

    if action == "cpu_only":
        print("  No GPU available. Staying on CPU.")
        return

    cmds = []
    if rec["install_cmd"]:
        cmds.append(rec["install_cmd"])
    if rec["ort_cmd"]:
        cmds.append(rec["ort_cmd"])

    for cmd in cmds:
        print(f"\n  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  ERROR: command failed (exit {result.returncode})")
            sys.exit(result.returncode)

    print("\n  Installation complete. Restart NeuraBreak to use the GPU.")
    print("  Verify with:  python tools/install_gpu_torch.py --check")


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect GPU and install the right torch/onnxruntime build",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Actually run the install commands (default: print only)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Just print GPU status and exit",
    )
    args = parser.parse_args()

    rec = detect_and_recommend()
    print_report(rec)

    if args.check:
        return

    if args.run:
        run_install(rec)
    else:
        print("  (Dry run — pass --run to actually install.)\n")


if __name__ == "__main__":
    main()
