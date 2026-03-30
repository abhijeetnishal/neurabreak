"""Train the NeuraBreak YOLO26 posture detection model.

Prerequisites:
    pip install ultralytics wandb
    # Annotate data first — see PLAN.md §5.3

Usage:
    python training/train.py                     # train with defaults
    python training/train.py --model yolo26n.pt  # use 'nano' base model
    python training/train.py --epochs 50 --no-wandb  # quick local run
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NeuraBreak YOLO model")
    p.add_argument("--model", default="yolo26n.pt",
                   help="Base model: yolo26n.pt / yolo26s.pt / yolo26m.pt")
    p.add_argument("--data", default="training/dataset.yaml")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="",
                   help="'' = auto, 'cpu' = force CPU, '0' = first GPU")
    p.add_argument("--name", default="neurabreak_v1")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Make sure ultralytics is installed
    try:
        from ultralytics import YOLO
        import torch
    except ImportError:
        print("ERROR: Run `pip install ultralytics` before training.")
        sys.exit(1)

    # Optional W&B experiment tracking
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(project="neurabreak", name=args.name)
        except ImportError:
            print("WARNING: wandb not installed — training without experiment tracking")
            print("         Install with: pip install wandb")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on device: {device}")
    print(f"Base model:         {args.model}")
    print(f"Dataset:            {args.data}")
    print(f"Epochs:             {args.epochs}\n")

    # Load hyperparams from yaml so they're version-controlled
    import yaml
    with open("training/hyperparams.yaml") as f:
        hyp = yaml.safe_load(f)

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        # Augmentation from hyperparams.yaml
        hsv_h=hyp["hsv_h"],
        hsv_s=hyp["hsv_s"],
        hsv_v=hyp["hsv_v"],
        degrees=hyp["degrees"],
        translate=hyp["translate"],
        scale=hyp["scale"],
        fliplr=hyp["fliplr"],
        mosaic=hyp["mosaic"],
        mixup=hyp["mixup"],
        # Regularisation
        dropout=hyp["dropout"],
        weight_decay=hyp["weight_decay"],
        # Transfer learning
        freeze=hyp["freeze"],
        pretrained=hyp["pretrained"],
        # Logging / saving
        project="runs/train",
        name=args.name,
        save=True,
        save_period=hyp["save_period"],
        patience=hyp["patience"],
        exist_ok=True,
        verbose=True,
    )

    best_map = results.results_dict.get("metrics/mAP50(B)", 0)
    print(f"\nTraining complete.")
    print(f"Best mAP50: {best_map:.4f}")
    print(f"Best weights: runs/train/{args.name}/weights/best.pt")

    if best_map < 0.80:
        print("\nWARNING: mAP50 < 0.80 — check your annotations and class distribution.")
        print("         Run: python tools/dataset_stats.py --data data/splits")


if __name__ == "__main__":
    main()
