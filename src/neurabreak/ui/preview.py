"""Live webcam preview window — optional, shows what the model sees.

Displays the webcam feed with bounding box overlays so the user can
verify the model is detecting the right thing.  Off by default — the
user enables it from the tray menu.

Frame updates come via update_frame() which is called from the inference
thread.  The actual Qt widget update is bounced to the main thread via
QMetaObject.invokeMethod to keep things thread-safe.
"""

from __future__ import annotations

import structlog
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from neurabreak.ui.branding import apply_window_icon

log = structlog.get_logger()

# How many milliseconds to wait before assuming the camera is gone
_TIMEOUT_MS = 3_000


class PreviewWindow(QWidget):
    """Small floating window showing the live camera feed + bounding boxes.

    Useful for checking that the model is detecting correctly.
    Toggle from the tray menu or settings.
    """

    # Carries raw RGB bytes + (width, height) from any thread → main thread.
    # QueuedConnection is chosen automatically for cross-thread connections.
    _frame_ready: Signal = Signal(bytes, int, int)

    # Emitted when the window is closed via the title-bar X button
    closed: Signal = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("NeuraBreak — Camera Preview")
        apply_window_icon(self)
        self.setMinimumSize(320, 240)
        self.resize(480, 360)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setStyleSheet("background-color: #111;")
        # Prevent this window closing from triggering an app-quit
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._label.setStyleSheet("color: #555; font-size: 14px;")
        self._label.setText("Waiting for camera…")
        layout.addWidget(self._label)

        # Status bar at bottom
        self._status = QLabel()
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setMaximumHeight(20)
        self._status.setStyleSheet("color: #666; font-size: 10px; background: #1a1a1a;")
        layout.addWidget(self._status)

        # Stale-frame watchdog — if no frame arrives for 3 s, show a message
        self._watchdog = QTimer(self)
        self._watchdog.setInterval(_TIMEOUT_MS)
        self._watchdog.setSingleShot(True)
        self._watchdog.timeout.connect(self._on_camera_stalled)

        # Cross-thread frame delivery (QueuedConnection → always runs on main thread)
        self._frame_ready.connect(self._on_frame_ready)

    #  Public API

    def show(self) -> None:  # type: ignore[override]
        super().show()
        self.activateWindow()
        self.raise_()
        self._watchdog.start()

    def update_frame(self, frame, boxes: list | None = None) -> None:
        """Push a new BGR frame (numpy array) with optional detection boxes.

        Fully thread-safe: does only numpy/cv2 work on the calling thread,
        then emits _frame_ready which is delivered to the main thread via
        Qt's QueuedConnection mechanism.
        """
        try:
            import cv2  # type: ignore[import-untyped]

            # All numpy/cv2 work is fine on the inference thread
            display = frame.copy()
            if boxes:
                self._draw_boxes(display, boxes)

            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            # Emit raw bytes — Qt delivers this to _on_frame_ready on the main thread
            self._frame_ready.emit(rgb.tobytes(), w, h)

        except ImportError:
            self._frame_ready.emit(b"", -1, -1)  # sentinel → show error on main thread
        except Exception as exc:
            log.error("preview_frame_error", error=str(exc))

    @property
    def is_visible(self) -> bool:
        return self.isVisible()

    #  Internal helpers

    @Slot(bytes, int, int)
    def _on_frame_ready(self, raw_bytes: bytes, w: int, h: int) -> None:
        """Runs on the main/GUI thread — all Qt operations are safe here."""
        if not self.isVisible():
            return

        if w < 0:
            self._label.setText("opencv-python not installed")
            return

        bytes_per_line = w * 3
        qimg = QImage(raw_bytes, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
        self._status.setText(f"{w}\xd7{h}")
        self._watchdog.start()  # safe — we are on the main thread

    def _draw_boxes(self, frame, boxes: list) -> None:
        """Draw YOLO bounding boxes on the frame in-place.

        Each box: dict with keys x1, y1, x2, y2, cls_name, confidence
        """
        import cv2  # type: ignore[import-untyped]

        colour_map = {
            "face_present":  (52, 152, 219),   # blue
            "posture_good":  (46, 204, 113),   # green
            "posture_bad":   (231, 76, 60),    # red
            "person_absent": (149, 165, 166),  # grey
        }

        for box in boxes:
            x1 = int(box.get("x1", 0))
            y1 = int(box.get("y1", 0))
            x2 = int(box.get("x2", 0))
            y2 = int(box.get("y2", 0))
            cls = box.get("cls_name", "unknown")
            conf = box.get("confidence", 0.0)

            colour = colour_map.get(cls, (149, 165, 166))  # grey fallback

            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            label_text = f"{cls.replace('_', ' ')} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(frame, label_text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def _on_camera_stalled(self) -> None:
        self._label.setText("No camera feed…\nIs the camera in use by another app?")
        self._status.setText("stalled")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Stop the watchdog and notify listeners (e.g. the tray menu)."""
        self._watchdog.stop()
        self.closed.emit()
        super().closeEvent(event)

