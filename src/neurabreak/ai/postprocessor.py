"""Post-processing constants — class index to label mapping for the YOLO model."""

from __future__ import annotations

# Maps class index → label name (must match dataset.yaml order)
CLASS_NAMES: dict[int, str] = {
    0: "face_present",
    1: "posture_good",
    2: "posture_bad",
    3: "person_absent",
}

# Classes that indicate the person is present at the desk
PRESENCE_CLASSES = {0, 1, 2}


