"""Dataset statistics — quick summary of your annotated dataset.

Useful before training to spot class imbalance or missing data.

Usage:
    python tools/dataset_stats.py --data data/splits
"""

import argparse
from collections import Counter
from pathlib import Path


CLASS_NAMES = [
    "face_present", "posture_good", "posture_slouch",
    "posture_head_forward", "posture_head_tilt", "posture_too_close",
    "eye_rubbing", "phone_at_desk", "person_absent",
]


def analyze_split(split_dir: Path) -> dict:
    """Count class occurrences in one dataset split."""
    label_dir = split_dir / "labels"
    if not label_dir.exists():
        return {}

    counts: Counter = Counter()
    total_images = 0

    for label_file in label_dir.glob("*.txt"):
        total_images += 1
        for line in label_file.read_text().strip().splitlines():
            if line.strip():
                class_id = int(line.split()[0])
                counts[class_id] += 1

    return {"total_images": total_images, "class_counts": dict(counts)}


def print_stats(data_root: str) -> None:
    root = Path(data_root)

    for split_name in ("train", "val", "test"):
        split_dir = root / split_name
        if not split_dir.exists():
            continue

        stats = analyze_split(split_dir)
        if not stats:
            print(f"\n{split_name}: no labels found at {split_dir}")
            continue

        print(f"\n{'='*50}")
        print(f"  Split: {split_name}  ({stats['total_images']} images)")
        print(f"{'='*50}")
        total_boxes = sum(stats["class_counts"].values())
        for class_id, count in sorted(stats["class_counts"].items()):
            name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
            pct = 100 * count / total_boxes if total_boxes else 0
            bar = "█" * int(pct / 2)
            print(f"  {name:<25} {count:5d}  ({pct:5.1f}%)  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print dataset class distribution")
    parser.add_argument("--data", default="data/splits", help="Path to data/splits directory")
    args = parser.parse_args()
    print_stats(args.data)


if __name__ == "__main__":
    main()
