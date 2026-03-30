"""Annotation utilities — helpers for working with YOLO label files.

Useful for:
  - Checking annotation quality (blurry frames, empty labels)
  - Converting from other formats to YOLO format
  - Visualising annotations on frames

Usage:
    python tools/annotate_helper.py --check data/annotated/
    python tools/annotate_helper.py --visualise data/splits/train --sample 20
"""

import argparse
from pathlib import Path

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


CLASS_NAMES = [
    "face_present", "posture_good", "posture_slouch",
    "posture_head_forward", "posture_head_tilt", "posture_too_close",
    "eye_rubbing", "phone_at_desk", "person_absent",
]

# One colour per class (BGR)
CLASS_COLOURS = [
    (0, 255, 0),    # face_present       — green
    (0, 200, 255),  # posture_good       — yellow
    (0, 0, 255),    # posture_slouch     — red
    (255, 0, 0),    # posture_head_fwd   — blue
    (255, 0, 255),  # posture_head_tilt  — magenta
    (0, 165, 255),  # posture_too_close  — orange
    (128, 0, 128),  # eye_rubbing        — purple
    (255, 255, 0),  # phone_at_desk      — cyan
    (128, 128, 128),# person_absent      — grey
]

LAPLACIAN_BLUR_THRESHOLD = 100  # frames below this variance are considered blurry


def check_quality(annotated_dir: str) -> None:
    """Scan annotated images for quality issues."""
    root = Path(annotated_dir)
    images_dir = root / "images"
    labels_dir = root / "labels"

    if not images_dir.exists():
        print(f"No images/ directory in {annotated_dir}")
        return

    blurry, empty, total = 0, 0, 0
    for img_path in sorted(images_dir.glob("*.jpg")):
        total += 1
        label_path = labels_dir / (img_path.stem + ".txt")

        if not label_path.exists() or label_path.read_text().strip() == "":
            empty += 1
            print(f"  EMPTY labels: {img_path.name}")

        if HAS_CV2:
            img = cv2.imread(str(img_path))
            if img is not None:
                grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                variance = cv2.Laplacian(grey, cv2.CV_64F).var()
                if variance < LAPLACIAN_BLUR_THRESHOLD:
                    blurry += 1
                    print(f"  BLURRY (var={variance:.0f}): {img_path.name}")
        elif total == 1:
            print("  Note: cv2 not installed — skipping blur check")

    print(f"\nSummary: {total} images, {empty} empty labels, {blurry} blurry")


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotation quality helpers")
    parser.add_argument("--check", metavar="DIR", help="Check annotation quality in DIR")
    parser.add_argument("--visualise", metavar="SPLIT_DIR", help="Visualise annotations")
    parser.add_argument("--sample", type=int, default=10, help="Number of images to show")
    args = parser.parse_args()

    if args.check:
        check_quality(args.check)
    elif args.visualise:
        print("Visualisation not yet implemented — coming when the dataset is ready.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
