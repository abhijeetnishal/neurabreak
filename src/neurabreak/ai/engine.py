"""YOLO inference engine wrapper.

The engine runs in its own thread. Never call infer() from the UI thread.

Device selection order (automatic):
  1. CUDA   — NVIDIA GPU via PyTorch CUDA
  2. MPS    — Apple Silicon GPU (M1/M2/M3)
  3. CPU    — universal fallback

For ONNX models, onnxruntime automatically uses the best available provider:
  TensorRT → CUDA → DirectML (AMD/Intel on Windows) → CoreML → CPU
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from neurabreak.ai.postprocessor import CLASS_NAMES, PRESENCE_CLASSES

if TYPE_CHECKING:
    import numpy as np

log = structlog.get_logger()

# COCO class ID for "person" — used for presence detection
_PRESENCE_CONF = 0.40
_PRESENCE_CLASSES: frozenset[int] = frozenset(PRESENCE_CLASSES)
_YOLO_FLOOR = 0.25

def _resolve_model_path(filename: str) -> str:
    """Return the full path to *filename*, checking the PyInstaller bundle first.

    When running as a frozen PyInstaller bundle (sys.frozen == True), models
    are unpacked into ``sys._MEIPASS/models/`` (or occasionally the root of
    ``sys._MEIPASS``). Falls back to *filename* unchanged so the normal
    file-system lookup still works in development.
    """
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass is not None:
            for subdir in ("models", "."):
                candidate = Path(meipass) / subdir / filename
                if candidate.exists():
                    return str(candidate)
    return filename


def _resolve_default_model_path(model_variant: str) -> str:
    """Prefer a bundled ONNX model, then fall back to the variant .pt model."""
    onnx_name = "neurabreak.onnx"
    if getattr(sys, "frozen", False):
        resolved = _resolve_model_path(onnx_name)
        if resolved != onnx_name:
            return resolved

    local_onnx = Path("models") / onnx_name
    if local_onnx.exists():
        return str(local_onnx)

    return _resolve_model_path(_VARIANT_MAP.get(model_variant, "yolo26n.pt"))


# Map model_variant → default model filename (used when model_path is empty)
_VARIANT_MAP: dict[str, str] = {
    "nano":   "yolo26n.pt",
    "small":  "yolo26s.pt",
    "medium": "yolo26m.pt",
}


def select_best_device(preferred: str = "auto") -> str:
    """Return the best available compute device string.

    Args:
        preferred: "auto" to detect, or force "cuda" / "mps" / "cpu".

    Returns:
        A device string understood by PyTorch and Ultralytics:
        "cuda", "mps", or "cpu".
    """
    if preferred not in ("auto", ""):
        return preferred  # user-forced device; trust them

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"

        # Apple Silicon MPS — requires torch >= 1.12 on macOS 12.3+
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


def _best_onnx_providers() -> list[str]:
    """Return onnxruntime execution providers in priority order.

    Covers:
      - TensorrtExecutionProvider  (NVIDIA — fastest)
      - CUDAExecutionProvider      (NVIDIA)
      - DmlExecutionProvider       (Windows DirectML — AMD, Intel, NVIDIA)
      - CoreMLExecutionProvider    (macOS / Apple Silicon)
      - CPUExecutionProvider       (always available)
    """
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

# Priority order for posture class selection when multiple boxes are detected.
# Higher number = selected first.  Classes absent from this map have priority 0.
_POSTURE_PRIORITY: dict[str, int] = {
    "posture_bad":    10,
    "posture_good":    4,
    "face_present":    1,
    "person_absent":   0,
}


@dataclass
class DetectionResult:
    """One inference pass worth of results."""

    presence: bool
    posture_class: str | None  # e.g. "posture_good", "posture_bad", None
    confidence: float
    phone_detected: bool = False
    eye_rubbing: bool = False
    latency_ms: float = 0.0
    raw_boxes: list[dict] = field(default_factory=list)  # raw YOLO output for debugging

    @property
    def is_bad_posture(self) -> bool:
        """True whenever the model is flagging something we should act on."""
        return self.posture_class == "posture_bad"


class InferenceEngine:
    """Wraps the YOLO model and owns its lifecyle.

    One engine instance per app. Load once at startup (in the inference
    thread), call infer() in a loop, unload on shutdown.

    Device priority (when device="auto"):
        CUDA (NVIDIA) > MPS (Apple) > CPU
    For .onnx models onnxruntime auto-selects:
        TensorRT > CUDA > DirectML (AMD/Intel) > CoreML > CPU
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.40,
        model_variant: str = "nano",
        device: str = "auto",
        use_half: bool = True,
        imgsz: int = 320,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        self.model_variant = model_variant
        self._device_pref = device   # raw preference ("auto", "cuda", "mps", "cpu")
        self.use_half = use_half
        self.imgsz = imgsz
        self._model = None
        self._ort_session = None     # set when mode == "onnx"
        self._ort_input_name: str = ""
        self._mode = "stub"          # "stub" | "ultralytics" | "onnx"
        self._loaded = False
        self._device = "cpu"         # resolved device string after load()
        self._half_active = False    # whether FP16 is actually in use

    def load(self) -> None:
        """Load the model. Call this from the inference thread, not the UI thread."""
        self._device = select_best_device(self._device_pref)
        path = self.model_path or _resolve_default_model_path(self.model_variant)
        is_onnx = str(path).lower().endswith(".onnx")

        # ONNX Runtime path
        if is_onnx:
            try:
                import onnxruntime as ort  # type: ignore

                providers = _best_onnx_providers()
                self._ort_session = ort.InferenceSession(path, providers=providers)
                self._ort_input_name = self._ort_session.get_inputs()[0].name
                self._mode = "onnx"
                active_provider = self._ort_session.get_providers()[0]
                log.info(
                    "onnx_model_loaded",
                    path=path,
                    provider=active_provider,
                    providers_tried=providers,
                )
            except Exception as e:
                log.warning("onnx_load_failed_using_stub", error=str(e))
                self._mode = "stub"
            self._loaded = True
            return

        # PyTorch / Ultralytics path
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(path)

            # Move weights to the selected device
            if self._device in ("cuda", "mps"):
                self._model.to(self._device)

                # FP16 (half precision) — GPU only; CPU half precision is not
                # supported by all ops and is generally slower on x86.
                if self.use_half and self._device == "cuda":
                    self._model.model.half()
                    self._half_active = True

            self._mode = "ultralytics"
            log.info(
                "model_loaded",
                path=path,
                device=self._device,
                half=self._half_active,
                imgsz=self.imgsz,
            )
        except Exception as e:
            log.warning("model_load_failed_using_stub", error=str(e))
            self._model = None
            self._mode = "stub"

        self._loaded = True

    def infer(self, frame: "np.ndarray") -> DetectionResult:
        """Run one forward pass. Returns a DetectionResult.

        Args:
            frame: BGR frame from OpenCV, any resolution.

        Raises:
            RuntimeError: if load() hasn't been called yet.
        """
        if not self._loaded:
            raise RuntimeError("Call InferenceEngine.load() before infer()")

        if self._mode == "stub" or (self._model is None and self._ort_session is None):
            return DetectionResult(presence=False, posture_class=None, confidence=0.0)

        t0 = time.perf_counter()

        if self._mode == "onnx":
            return self._infer_onnx(frame, t0)

        return self._infer_ultralytics(frame, t0)

    # Private inference helpers

    def _infer_ultralytics(self, frame: "np.ndarray", t0: float) -> DetectionResult:
        try:
            yolo_conf = min(_YOLO_FLOOR, self.confidence_threshold)
            results = self._model(
                frame,
                verbose=False,
                conf=yolo_conf,
                imgsz=self.imgsz,
                device=self._device,
                half=self._half_active,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            return self._parse_ultralytics_results(results, latency_ms)
        except Exception as e:
            log.error("inference_error", error=str(e))
            return DetectionResult(presence=False, posture_class=None, confidence=0.0)

    def _infer_onnx(self, frame: "np.ndarray", t0: float) -> DetectionResult:
        """ONNX Runtime inference: CUDA / DirectML / CoreML / CPU depending on provider."""
        try:
            import cv2  # type: ignore
            import numpy as np

            resized = cv2.resize(frame, (self.imgsz, self.imgsz))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

            raw = self._ort_session.run(None, {self._ort_input_name: tensor})
            latency_ms = (time.perf_counter() - t0) * 1000

            raw_boxes = self._parse_onnx_output(raw[0], frame.shape[:2])

            return self._parse_raw_boxes(raw_boxes, latency_ms)
        except Exception as e:
            log.error("onnx_inference_error", error=str(e))
            return DetectionResult(presence=False, posture_class=None, confidence=0.0)

    def _parse_onnx_output(self, preds: "np.ndarray", frame_shape: tuple[int, int]) -> list[dict]:
        """Parse YOLO ONNX output with or without export-time NMS.

        Exported Ultralytics models without NMS usually return
        ``[1, 4 + num_classes, anchors]``. With ``nms=True`` they return
        ``[1, max_det, 6]`` as ``xyxy, confidence, class``. The runtime accepts
        both so older exported models do not silently become all-neutral.
        """
        if preds.ndim == 3:
            preds = preds[0]
        if preds.ndim != 2:
            return []

        # Raw YOLO tensors are commonly [channels, anchors]; NMS tensors are
        # already [detections, 6].
        if preds.shape[0] in (len(CLASS_NAMES) + 4, len(CLASS_NAMES) + 5, 6):
            if preds.shape[0] < preds.shape[1]:
                preds = preds.transpose(1, 0)

        if preds.shape[1] == 6:
            boxes = self._parse_onnx_nms_rows(preds, frame_shape)
        else:
            boxes = self._decode_onnx_yolo_rows(preds, frame_shape)

        return self._nms_boxes(boxes, iou_threshold=0.45)

    def _parse_onnx_nms_rows(self, rows: "np.ndarray", frame_shape: tuple[int, int]) -> list[dict]:
        """Parse rows shaped as x1, y1, x2, y2, confidence, class_id."""
        boxes: list[dict] = []
        frame_h, frame_w = frame_shape
        scale_x = frame_w / float(self.imgsz)
        scale_y = frame_h / float(self.imgsz)

        for det in rows:
            x1, y1, x2, y2 = (float(det[0]), float(det[1]), float(det[2]), float(det[3]))
            conf = float(det[4])
            cls_id = int(det[5])
            if conf < _YOLO_FLOOR:
                continue
            boxes.append(
                self._box_dict(
                    cls_id=cls_id,
                    conf=conf,
                    x1=x1 * scale_x,
                    y1=y1 * scale_y,
                    x2=x2 * scale_x,
                    y2=y2 * scale_y,
                )
            )
        return boxes

    def _decode_onnx_yolo_rows(self, rows: "np.ndarray", frame_shape: tuple[int, int]) -> list[dict]:
        """Decode raw YOLO rows shaped as cx, cy, w, h, [obj], class_scores."""
        import numpy as np

        boxes: list[dict] = []
        frame_h, frame_w = frame_shape
        scale_x = frame_w / float(self.imgsz)
        scale_y = frame_h / float(self.imgsz)
        num_classes = len(CLASS_NAMES)

        for det in rows:
            if len(det) < 4 + num_classes:
                continue

            if len(det) >= 5 + num_classes:
                objectness = float(det[4])
                scores = det[5:5 + num_classes]
            else:
                objectness = 1.0
                scores = det[4:4 + num_classes]

            cls_id = int(np.argmax(scores))
            conf = objectness * float(scores[cls_id])
            if conf < _YOLO_FLOOR:
                continue

            cx, cy, width, height = (
                float(det[0]),
                float(det[1]),
                float(det[2]),
                float(det[3]),
            )
            x1 = (cx - width / 2.0) * scale_x
            y1 = (cy - height / 2.0) * scale_y
            x2 = (cx + width / 2.0) * scale_x
            y2 = (cy + height / 2.0) * scale_y
            boxes.append(self._box_dict(cls_id=cls_id, conf=conf, x1=x1, y1=y1, x2=x2, y2=y2))

        return boxes

    def _box_dict(
        self,
        *,
        cls_id: int,
        conf: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> dict:
        cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        return {
            "cls": cls_id,
            "cls_name": cls_name,
            "confidence": conf,
            "conf": conf,
            "x1": max(0, int(x1)),
            "y1": max(0, int(y1)),
            "x2": max(0, int(x2)),
            "y2": max(0, int(y2)),
        }

    def _nms_boxes(self, boxes: list[dict], iou_threshold: float) -> list[dict]:
        """Apply class-aware non-maximum suppression to decoded boxes."""
        kept: list[dict] = []
        by_class: dict[int, list[dict]] = {}
        for box in boxes:
            by_class.setdefault(int(box["cls"]), []).append(box)

        for class_boxes in by_class.values():
            pending = sorted(class_boxes, key=lambda item: float(item["conf"]), reverse=True)
            while pending:
                best = pending.pop(0)
                kept.append(best)
                pending = [
                    box for box in pending
                    if self._box_iou(best, box) < iou_threshold
                ]

        return kept

    @staticmethod
    def _box_iou(a: dict, b: dict) -> float:
        x_left = max(a["x1"], b["x1"])
        y_top = max(a["y1"], b["y1"])
        x_right = min(a["x2"], b["x2"])
        y_bottom = min(a["y2"], b["y2"])
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = float((x_right - x_left) * (y_bottom - y_top))
        area_a = float(max(0, a["x2"] - a["x1"]) * max(0, a["y2"] - a["y1"]))
        area_b = float(max(0, b["x2"] - b["x1"]) * max(0, b["y2"] - b["y1"]))
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def _parse_ultralytics_results(self, results: list, latency_ms: float) -> DetectionResult:
        presence = False
        best_conf = 0.0
        raw_boxes: list[dict] = []

        for r in results:
            if r.boxes is None:
                continue
            names = r.names if hasattr(r, "names") and r.names else {}
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                raw_boxes.append({
                    "cls": cls_id,
                    "cls_name": names.get(cls_id, f"class_{cls_id}"),
                    "confidence": conf,
                    "conf": conf,
                    "x1": int(xyxy[0]),
                    "y1": int(xyxy[1]),
                    "x2": int(xyxy[2]),
                    "y2": int(xyxy[3]),
                })
                if cls_id in _PRESENCE_CLASSES and conf > _PRESENCE_CONF:
                    presence = True
                    best_conf = max(best_conf, conf)

        result = self._parse_raw_boxes(raw_boxes, latency_ms)
        if presence:
            result.presence = presence
            result.confidence = best_conf
        return result

    def _parse_raw_boxes(self, raw_boxes: list[dict], latency_ms: float) -> DetectionResult:
        """Shared posture/presence logic for both ultralytics and ONNX paths."""
        presence = False
        best_conf = 0.0
        best_posture: str | None = None
        for box_info in raw_boxes:
            cls_name = box_info["cls_name"]
            conf = box_info["conf"]
            cls_id = box_info["cls"]

            if cls_id in _PRESENCE_CLASSES and conf > _PRESENCE_CONF:
                presence = True
                best_conf = max(best_conf, conf)

            if conf < self.confidence_threshold:
                continue
            if cls_name not in _POSTURE_PRIORITY:
                continue
            priority = _POSTURE_PRIORITY[cls_name]
            current_priority = _POSTURE_PRIORITY.get(best_posture, -1) if best_posture else -1
            if priority > current_priority:
                best_posture = cls_name

        posture_class = best_posture if best_posture else ("face_present" if presence else None)
        return DetectionResult(
            presence=presence,
            posture_class=posture_class,
            confidence=best_conf,
            latency_ms=latency_ms,
            raw_boxes=raw_boxes,
        )

    def unload(self) -> None:
        self._model = None
        self._ort_session = None
        self._mode = "stub"
        self._loaded = False
        self._half_active = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded
