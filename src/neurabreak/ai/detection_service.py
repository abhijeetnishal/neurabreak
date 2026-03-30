"""Detection service — owns the inference thread.

Pulls frames from the camera service, runs them through the engine, and
feeds results into the state machine. Everything here runs in a background
thread so the Qt UI stays completely free.

The only side effects that reach the UI happen via the event bus.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import structlog

from neurabreak.core.events import Event, EventType, bus

if TYPE_CHECKING:
    from neurabreak.ai.camera import FrameCaptureService
    from neurabreak.ai.engine import InferenceEngine
    from neurabreak.core.config import AppConfig
    from neurabreak.core.state_machine import PostureStateMachine
    from neurabreak.data.journal import HealthJournalService

log = structlog.get_logger()


class DetectionService:
    """Manages the full pipeline: camera → engine → state machine.

    Lifecycle:
        service = DetectionService(camera, engine, state_machine, config)
        service.start()   # begins camera + inference threads
        ...
        service.stop()    # clean shutdown, waits for threads to join
    """

    def __init__(
        self,
        camera: FrameCaptureService,
        engine: InferenceEngine,
        state_machine: PostureStateMachine,
        config: AppConfig,
        journal: HealthJournalService | None = None,
    ) -> None:
        self.camera = camera
        self.engine = engine
        self.state_machine = state_machine
        self.config = config
        self._journal = journal

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False

        # Rolling stats for diagnostics
        self._frames_processed = 0
        self._frames_skipped = 0
        self._last_latency_ms = 0.0

        # Optional callback(frame, boxes) for the live preview window
        self._frame_sink = None

        # Frame-skip state — grayscale of the previous frame for diff comparison
        self._prev_gray = None
        self._consecutive_static_skips = 0

        # Cached stable result used on skipped frames
        self._cached_presence: bool = False
        self._cached_posture: str | None = None
        self._cached_confidence: float = 0.0
        self._cached_boxes: list[dict] = []
        self._has_cached_result = False

        # Posture-class debounce — only forward a bad-posture class to the state
        # machine after it has appeared for consecutive_frames frames in a row.
        self._posture_candidate: str | None = None
        self._posture_candidate_count: int = 0

    def start(self) -> None:
        if self._running:
            log.warning("detection_service_already_running")
            return

        self.camera.start()

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._inference_loop,
            name="InferenceThread",
            daemon=True,
        )
        self._thread.start()
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        self._stop_event.set()
        self.camera.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.engine.unload()
        self._running = False
        log.info("detection_service_stopped", frames_processed=self._frames_processed)

    def set_frame_sink(self, callback) -> None:
        """Register a callable(frame, boxes) that receives every inference frame.

        Called from the inference thread — the callback must be thread-safe.
        Pass None to remove the sink.
        """
        self._frame_sink = callback

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    def _inference_loop(self) -> None:
        """The main body of the inference thread. Loads the model then loops."""
        bus.publish(Event(EventType.MODEL_LOADING))
        self.engine.load()
        bus.publish(Event(EventType.MODEL_LOADED))

        skip_threshold = getattr(self.config.detection, "frame_skip_threshold", 8.0)
        fps = max(1, int(getattr(self.config.detection, "fps", 5)))
        # Allow a few static-frame skips, then force one inference
        max_static_skips = max(1, int(round(fps * 0.6)))

        while not self._stop_event.is_set():
            frame = self.camera.get_frame(timeout=0.5)
            if frame is None:
                continue  # timed out waiting for a frame, loop again

            # Frame-skip optimisation
            # When the scene is nearly static (user not moving), skip inference to save CPU/GPU
            if skip_threshold > 0.0:
                try:
                    import cv2  # type: ignore
                    import numpy as np

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if self._prev_gray is not None and gray.shape == self._prev_gray.shape:
                        diff = cv2.absdiff(gray, self._prev_gray)
                        is_static_scene = float(np.mean(diff)) < skip_threshold
                        if is_static_scene and self._consecutive_static_skips < max_static_skips:
                            self._frames_skipped += 1
                            self._consecutive_static_skips += 1

                            # Keep preview responsive with the latest known boxes.
                            sink = self._frame_sink
                            if sink is not None:
                                try:
                                    sink(frame, self._cached_boxes if self._has_cached_result else [])
                                except Exception:  # noqa: BLE001
                                    pass

                            # Keep state-machine timing/presence progression alive
                            # while we temporarily skip expensive inference.
                            if self._has_cached_result:
                                self.state_machine.process(
                                    present=self._cached_presence,
                                    posture_class=self._cached_posture,
                                    confidence=self._cached_confidence,
                                )

                            self._prev_gray = gray
                            continue

                    self._consecutive_static_skips = 0
                    self._prev_gray = gray
                except Exception:  # noqa: BLE001
                    self._consecutive_static_skips = 0
                    pass  # cv2 unavailable or shape mismatch — just run inference

            result = self.engine.infer(frame)
            self._last_latency_ms = result.latency_ms
            self._frames_processed += 1

            # Forward frame to the live preview window if one is registered
            sink = self._frame_sink
            if sink is not None:
                try:
                    sink(frame, result.raw_boxes)
                except Exception as exc:  # noqa: BLE001
                    log.warning("frame_sink_error", error=str(exc))

            # Publish raw result for anything that wants to observe it (e.g. live preview)
            bus.publish(
                Event(
                    EventType.DETECTION_COMPLETE,
                    {
                        "presence": result.presence,
                        "posture_class": result.posture_class,
                        "confidence": result.confidence,
                        "latency_ms": result.latency_ms,
                        "phone_detected": result.phone_detected,
                    },
                )
            )

            # Posture-class debounce: nano/small models can flicker between classes
            # on back-to-back frames.  Require the same bad-posture label for
            # consecutive_frames frames before it reaches the state machine.
            _GOOD_CLASSES: frozenset[str | None] = frozenset(
                {"posture_good", "face_present", "person_absent", None}
            )
            consecutive_frames = self.config.detection.consecutive_frames
            if result.posture_class not in _GOOD_CLASSES:
                if result.posture_class == self._posture_candidate:
                    self._posture_candidate_count += 1
                else:
                    # New candidate — start fresh count
                    self._posture_candidate = result.posture_class
                    self._posture_candidate_count = 1
                # Only graduate to "stable" once we have enough consecutive frames
                stable_posture = (
                    result.posture_class
                    if self._posture_candidate_count >= consecutive_frames
                    else None  # treat as neutral until confirmed
                )
            else:
                # Good/neutral or absent — forward immediately and reset debounce
                self._posture_candidate = None
                self._posture_candidate_count = 0
                stable_posture = result.posture_class

            # Cache the latest stable result so skipped frames can still drive
            self._cached_presence = result.presence
            self._cached_posture = stable_posture
            self._cached_confidence = result.confidence
            self._cached_boxes = result.raw_boxes
            self._has_cached_result = True

            # Feed the state machine — this is what drives session timing and alerts
            self.state_machine.process(
                present=result.presence,
                posture_class=stable_posture,
                confidence=result.confidence,
            )

            # Persist the result to the health journal if available.
            if (
                self._journal is not None
                and self.config.privacy.store_detections
                and result.posture_class is not None
            ):
                try:
                    self._journal.record_detection(
                        posture_class=result.posture_class,
                        confidence=result.confidence,
                        is_face_present=result.presence,
                        phone_detected=result.phone_detected,
                        inference_ms=result.latency_ms,
                    )
                except Exception as exc:
                    log.warning("journal_record_failed", error=str(exc))

            # Log phone detection as its own event so the notification manager can react
            if result.phone_detected:
                bus.publish(Event(EventType.PHONE_DETECTED))

        log.info(
            "inference_thread_stopped",
            frames_processed=self._frames_processed,
            frames_skipped=self._frames_skipped,
        )
