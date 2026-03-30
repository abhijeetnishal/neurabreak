"""Export the best trained checkpoint to deployment formats.

Creates:
  - ONNX FP32 (universal, works on CPU/CUDA/DirectML/CoreML)
  - TensorRT FP16 (NVIDIA CUDA only — fastest on RTX GPUs)
  - CoreML (macOS only)

Usage:
    python training/export.py
    python training/export.py --weights runs/train/neurabreak_v1/weights/best.pt
    python training/export.py --format onnx tensorrt
    python training/export.py --format onnx --imgsz 320
"""

import argparse
import shutil
import sys
from pathlib import Path


def export_model(
    weights: str,
    formats: list[str],
    output_dir: str = "models",
    imgsz: int = 320,
) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: Run `pip install ultralytics` first.")
        sys.exit(1)

    model = YOLO(weights)
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    for fmt in formats:
        print(f"\nExporting to {fmt.upper()}  (imgsz={imgsz}) ...")
        try:
            if fmt == "onnx":
                # FP32 ONNX — works on every runtime:
                #   onnxruntime-cpu    → CPUExecutionProvider
                #   onnxruntime-gpu    → CUDAExecutionProvider
                #   onnxruntime-directml → DmlExecutionProvider (AMD/Intel/NVIDIA, Windows)
                #   onnxruntime (macOS) → CoreMLExecutionProvider
                path = model.export(
                    format="onnx",
                    imgsz=imgsz,
                    optimize=True,    # fuse BN for CPU
                    simplify=True,    # ONNX graph simplification (onnxsim)
                    opset=17,
                    dynamic=False,    # fixed batch=1 for desktop app
                    half=False,       # stay FP32 for CPU; DML also prefers FP32
                )
            elif fmt == "tensorrt":
                # TensorRT FP16 — NVIDIA only, ~5-10× faster than ONNX on RTX GPUs
                path = model.export(
                    format="engine",
                    imgsz=imgsz,
                    half=True,    # FP16 tensor cores
                    device=0,
                )
            elif fmt == "coreml":
                path = model.export(
                    format="coreml",
                    imgsz=imgsz,
                    nms=True,
                )
            else:
                print(f"  Unknown format: {fmt} — skipping")
                continue

            dest = out / Path(path).name
            shutil.copy2(path, dest)
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  Saved: {dest} ({size_mb:.1f} MB)")

        except Exception as e:
            print(f"  ERROR exporting {fmt}: {e}")

    print(f"\nExport complete. Files in: {out}/")
    print("Update models/MODEL_CARD.md with the new benchmarks.")
    print()
    print("Next step — update ~/.neurabreak/config.toml:")
    print(f"  [detection]")
    print(f"  model_path = \"~/.neurabreak/models/<exported_file>\"")
    print(f"  imgsz = {imgsz}")


def main() -> None:
    p = argparse.ArgumentParser(description="Export trained model to deployment formats")
    p.add_argument("--weights", default="runs/train/neurabreak_v1/weights/best.pt")
    p.add_argument(
        "--format", nargs="+", default=["onnx"],
        choices=["onnx", "tensorrt", "coreml"],
    )
    p.add_argument("--output", default="models")
    p.add_argument(
        "--imgsz", type=int, default=320,
        help="Export input resolution (default 320 — 4× cheaper than 640).",
    )
    args = p.parse_args()
    export_model(args.weights, args.format, args.output, args.imgsz)


if __name__ == "__main__":
    main()
