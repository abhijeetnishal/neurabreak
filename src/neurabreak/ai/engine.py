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

if TYPE_CHECKING:
    import numpy as np

log = structlog.get_logger()

# COCO class ID for "person" — used for presence detection
_PRESENCE_CONF = 0.40
# Custom 4-class model: 0=face_present, 1=posture_good, 2=posture_bad all indicate presence
_PRESENCE_CLASSES: frozenset[int] = frozenset({0, 1, 2})
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
        _default = _VARIANT_MAP.get(self.model_variant, "yolo26n.pt")
        path = self.model_path or _resolve_model_path(_default)
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

            # shape [1, num_boxes, 6] or [1, 6, num_boxes]
            preds = raw[0]
            if preds.ndim == 3 and preds.shape[1] < preds.shape[2]:
                preds = preds.transpose(0, 2, 1)

            raw_boxes: list[dict] = []
            if preds.ndim >= 2 and preds.shape[0] > 0:
                for det in preds[0]:
                    if len(det) < 6:
                        continue
                    x1, y1, x2, y2, conf, cls_id = (
                        float(det[0]), float(det[1]), float(det[2]),
                        float(det[3]), float(det[4]), int(det[5]),
                    )
                    if conf < _YOLO_FLOOR:
                        continue
                    raw_boxes.append({
                        "cls": cls_id,
                        "cls_name": f"class_{cls_id}",
                        "confidence": conf,
                        "conf": conf,
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2),
                    })

            return self._parse_raw_boxes(raw_boxes, latency_ms)
        except Exception as e:
            log.error("onnx_inference_error", error=str(e))
            return DetectionResult(presence=False, posture_class=None, confidence=0.0)

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
