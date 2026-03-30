"""Data collection helper — saves webcam frames for annotation.

Run this script while adopting a specific posture class for ~5 minutes.
The frames are saved to data/raw/<class_name>/ and are ready to upload
to Roboflow for annotation.

Usage:
    python tools/collect_data.py --class posture_good --duration 300
    python tools/collect_data.py --class posture_slouch --duration 300 --fps 2
"""

import argparse
import os
import time

# opencv-python must be installed: pip install opencv-python
try:
    import cv2
except ImportError:
    print("ERROR: OpenCV not installed. Run: pip install opencv-python")
    raise


VALID_CLASSES = [
    "face_present",
    "posture_good",
    "posture_slouch",
    "posture_head_forward",
    "posture_head_tilt",
    "posture_too_close",
    "eye_rubbing",
    "phone_at_desk",
    "person_absent",
]


def collect_frames(class_name: str, duration_secs: int = 300, fps: int = 2, camera: int = 0) -> int:
    """Capture frames for a specific posture class.

    Args:
        class_name:    e.g. "posture_good"
        duration_secs: how long to record
        fps:           frames per second to save (2 is enough, avoids near-duplicate frames)
        camera:        camera device index

    Returns:
        Number of frames saved.
    """
    out_dir = os.path.join("data", "raw", class_name)
    os.makedirs(out_dir, exist_ok=True)

    # Figure out where to start numbering so we don't overwrite existing frames
    existing = [f for f in os.listdir(out_dir) if f.endswith(".jpg")]
    start_index = len(existing)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera}")
        return 0

    interval = 1.0 / fps
    count = 0
    start = time.time()

    print(f"\n  Class:    {class_name}")
    print(f"  Duration: {duration_secs}s")
    print(f"  FPS:      {fps}")
    print(f"  Output:   {out_dir}/")
    print("\n  Adopt the posture NOW. Press Q to stop early.\n")

    # 3-second countdown
    for i in (3, 2, 1):
        print(f"  Starting in {i}...")
        time.sleep(1)
    print("  Recording!")

    while time.time() - start < duration_secs:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to read frame from camera")
            break

        filename = os.path.join(out_dir, f"{class_name}_{start_index + count:05d}.jpg")
        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        count += 1

        elapsed = time.time() - start
        remaining = duration_secs - elapsed
        cv2.putText(frame, f"{remaining:.0f}s remaining | {count} frames | Q to stop",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(f"Collecting: {class_name}", frame)

        if cv2.waitKey(int(interval * 1000)) & 0xFF == ord("q"):
            print("\n  Stopped early by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n  Done. Saved {count} frames to {out_dir}/")
    print(f"  Total frames for class: {start_index + count}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture webcam frames for annotation")
    parser.add_argument(
        "--class", dest="class_name", required=True, choices=VALID_CLASSES,
        help="Posture class to record",
    )
    parser.add_argument(
        "--duration", type=int, default=300,
        help="Recording duration in seconds (default: 300)",
    )
    parser.add_argument(
        "--fps", type=int, default=2,
        help="Frames per second to save (default: 2)",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device index (default: 0)",
    )
    args = parser.parse_args()
    collect_frames(args.class_name, args.duration, args.fps, args.camera)


if __name__ == "__main__":
    main()
