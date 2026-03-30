"""Webcam frame capture service.

Runs in a background thread and pushes frames onto a bounded queue.
The inference engine reads from this queue at its own pace.

Key design choices:
- Drops frames when the queue is full rather than blocking the camera.
  This keeps latency low — we'd rather skip a frame than pile up a backlog.
- Targets 5 FPS (configurable). Higher rates waste CPU for no real benefit
  when detecting posture, which changes slowly.
"""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    import numpy as np

log = structlog.get_logger()

# Max frames we'll hold in the queue before dropping new ones
QUEUE_MAX_DEPTH = 3


class FrameCaptureService:
    """Captures webcam frames in a background thread.

    Usage:
        service = FrameCaptureService(fps=5)
        service.start()
        frame = service.get_frame(timeout=1.0)   # returns np.ndarray or None
        service.stop()
    """

    def __init__(self, camera_index: int = 0, fps: int = 5) -> None:
        self.camera_index = camera_index
        self.fps = fps
        self._queue: queue.Queue["np.ndarray"] = queue.Queue(maxsize=QUEUE_MAX_DEPTH)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        """Start the capture thread."""
        if self._running:
            log.warning("camera_already_running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="FrameCapture",
            daemon=True,
        )
        self._thread.start()
        self._running = True

    def stop(self) -> None:
        """Signal the capture thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._running = False
        log.info("camera_stopped")

    def get_frame(self, timeout: float = 0.5) -> "np.ndarray | None":
        """Get the next available frame. Returns None if no frame arrives in time."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_running(self) -> bool:
        return self._running

    def _capture_loop(self) -> None:
        """Main capture loop — runs in the background thread."""
        try:
            import cv2
        except ImportError:
            log.error("opencv_not_installed", hint="Install the ai extras: uv sync --extra ai")
            self._running = False
            return

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            log.error("camera_open_failed", index=self.camera_index)
            self._running = False
            return

        interval = 1.0 / self.fps
        consecutive_failures = 0
        _MAX_FAILURES = 10  # retry up to 10 consecutive read failures before giving up

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    log.warning("camera_read_failed", attempt=consecutive_failures)
                    if consecutive_failures >= _MAX_FAILURES:
                        log.error("camera_too_many_failures", count=consecutive_failures)
                        self._running = False
                        break
                    self._stop_event.wait(timeout=interval)
                    continue
                consecutive_failures = 0
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    pass  # inference is still busy — drop this frame
                self._stop_event.wait(timeout=interval)
        finally:
            cap.release()
        log.info("camera_capture_loop_exited")
