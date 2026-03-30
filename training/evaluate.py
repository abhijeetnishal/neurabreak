"""Evaluate the trained model on the held-out test set.

Usage:
    python training/evaluate.py
    python training/evaluate.py --weights runs/train/neurabreak_v1/weights/best.pt
"""

import argparse
import sys


# Minimum thresholds that must pass before shipping the model
REQUIRED_THRESHOLDS = {
    "mAP50_overall":    0.80,
    "mAP50_face":       0.92,
    "mAP50_posture_good":  0.78,
    "mAP50_posture_slouch": 0.75,
    "precision":        0.80,
    "recall":           0.75,
}


def evaluate(weights: str, data: str) -> bool:
    """Run evaluation and check against shipping thresholds.

    Returns:
        True if all thresholds pass, False otherwise.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: Run `pip install ultralytics` first.")
        sys.exit(1)

    model = YOLO(weights)
    metrics = model.val(data=data, split="test")

    results = metrics.results_dict
    print("\n" + "="*60)
    print(f"  Model:    {weights}")
    print(f"  Dataset:  {data}")
    print("="*60)

    # Print per-class metrics
    print(f"\n  Overall mAP50:    {results.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  Overall mAP50-95: {results.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"  Precision:        {results.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall:           {results.get('metrics/recall(B)', 0):.4f}")

    # Check thresholds
    map50 = results.get("metrics/mAP50(B)", 0)
    precision = results.get("metrics/precision(B)", 0)
    recall = results.get("metrics/recall(B)", 0)

    checks = {
        "mAP50_overall": map50 >= REQUIRED_THRESHOLDS["mAP50_overall"],
        "precision":     precision >= REQUIRED_THRESHOLDS["precision"],
        "recall":        recall >= REQUIRED_THRESHOLDS["recall"],
    }

    print("\n  Threshold checks:")
    all_pass = True
    for name, passed in checks.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"    {name:<25} {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  All checks passed — model is ready to export.")
    else:
        print("\n  Some checks failed — review annotations and retrain.")

    return all_pass


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate trained NeuraBreak model")
    p.add_argument("--weights", default="runs/train/neurabreak_v1/weights/best.pt")
    p.add_argument("--data", default="training/dataset.yaml")
    args = p.parse_args()
    success = evaluate(args.weights, args.data)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
