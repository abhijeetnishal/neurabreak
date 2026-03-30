"""Unit tests for DetectionService.

Uses mocked camera and engine so no real webcam or model is required.
Verifies that the service feeds results into the state machine correctly
and publishes the expected bus events.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from neurabreak.ai.engine import DetectionResult
from neurabreak.core.events import EventType, bus
from neurabreak.core.state_machine import AppState, PostureStateMachine


def make_mock_camera(frames: list):
    """Return a fake camera that yields the given frames in sequence."""
    camera = MagicMock()
    camera.start = MagicMock()
    camera.stop = MagicMock()
    camera.is_running = True

    frame_iter = iter(frames + [None] * 100)  # pad with None so it drains gracefully
    camera.get_frame = MagicMock(side_effect=lambda timeout=0.5: next(frame_iter, None))
    return camera


def make_mock_engine(results: list[DetectionResult]):
    """Return a fake engine that returns the given results in sequence."""
    engine = MagicMock()
    engine.is_loaded = False
    engine.model_path = "fake.onnx"

    result_iter = iter(results)
    engine.infer = MagicMock(
        side_effect=lambda frame: next(result_iter, DetectionResult(presence=False, posture_class=None, confidence=0.0))
    )
    engine.load = MagicMock()
    engine.unload = MagicMock()
    return engine


class _FakeGrayFrame:
    """Tiny grayscale-like frame object used to exercise frame-skip logic."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.shape = (16, 16)


class _FakeDiff:
    def __init__(self, value: float) -> None:
        self.value = value


class _FakeCV2:
    COLOR_BGR2GRAY = 0

    @staticmethod
    def cvtColor(frame, _code):
        return frame

    @staticmethod
    def absdiff(a, b):
        return _FakeDiff(abs(float(a.value) - float(b.value)))


class _FakeNumpy:
    @staticmethod
    def mean(diff):
        return diff.value


class TestDetectionServiceLifecycle:
    def test_starts_camera_and_loads_engine(self):
        from neurabreak.core.config import AppConfig
        from neurabreak.ai.detection_service import DetectionService

        camera = make_mock_camera([])
        engine = make_mock_engine([])
        sm = PostureStateMachine(fps=5)
        cfg = AppConfig()

        service = DetectionService(camera, engine, sm, cfg)
        service.start()
        time.sleep(0.1)
        service.stop()

        camera.start.assert_called_once()
        engine.load.assert_called_once()
        engine.unload.assert_called_once()

    def test_stop_is_idempotent(self):
        from neurabreak.core.config import AppConfig
        from neurabreak.ai.detection_service import DetectionService

        service = DetectionService(
            make_mock_camera([]),
            make_mock_engine([]),
            PostureStateMachine(),
            AppConfig(),
        )
        service.start()
        service.stop()
        service.stop()  # should not raise


class TestDetectionServicePipeline:
    def test_presence_results_feed_state_machine(self):
        from neurabreak.core.config import AppConfig
        from neurabreak.ai.detection_service import DetectionService

        # Enough present frames to trigger IDLE → MONITORING (15 at 5 fps)
        frames = [object()] * 20
        results = [
            DetectionResult(presence=True, posture_class="face_present", confidence=0.9)
        ] * 20

        camera = make_mock_camera(frames)
        engine = make_mock_engine(results)
        sm = PostureStateMachine(fps=5)
        cfg = AppConfig()

        service = DetectionService(camera, engine, sm, cfg)
        service.start()
        time.sleep(0.3)  # give the thread time to process
        service.stop()

        # State machine should have moved to MONITORING
        assert sm.state == AppState.MONITORING

    def test_detection_complete_event_published(self):
        from neurabreak.core.config import AppConfig
        from neurabreak.ai.detection_service import DetectionService

        received: list = []
        bus.subscribe(EventType.DETECTION_COMPLETE, received.append)

        frames = [object()] * 3
        results = [
            DetectionResult(presence=True, posture_class="face_present", confidence=0.85)
        ] * 3

        service = DetectionService(
            make_mock_camera(frames),
            make_mock_engine(results),
            PostureStateMachine(),
            AppConfig(),
        )
        service.start()
        time.sleep(0.2)
        service.stop()

        bus.unsubscribe(EventType.DETECTION_COMPLETE, received.append)
        assert len(received) > 0
        assert received[0].data["presence"] is True

    def test_phone_event_published_when_detected(self):
        from neurabreak.core.config import AppConfig
        from neurabreak.ai.detection_service import DetectionService

        received: list = []
        bus.subscribe(EventType.PHONE_DETECTED, received.append)

        results = [
            DetectionResult(
                presence=True,
                posture_class="face_present",
                confidence=0.9,
                phone_detected=True,
            )
        ] * 2

        service = DetectionService(
            make_mock_camera([object()] * 2),
            make_mock_engine(results),
            PostureStateMachine(),
            AppConfig(),
        )
        service.start()
        time.sleep(0.2)
        service.stop()

        bus.unsubscribe(EventType.PHONE_DETECTED, received.append)
        assert len(received) >= 1

    def test_static_scene_still_reaches_monitoring(self):
        from neurabreak.core.config import AppConfig
        from neurabreak.ai.detection_service import DetectionService

        # Use constant grayscale values so frame-diff sees a static scene.
        frames = [_FakeGrayFrame(10.0) for _ in range(30)]
        results = [
            DetectionResult(presence=True, posture_class="face_present", confidence=0.9)
        ] * 30

        camera = make_mock_camera(frames)
        engine = make_mock_engine(results)
        sm = PostureStateMachine(fps=5)
        cfg = AppConfig()

        with patch.dict("sys.modules", {"cv2": _FakeCV2, "numpy": _FakeNumpy}):
            service = DetectionService(camera, engine, sm, cfg)
            service.start()
            time.sleep(0.25)
            service.stop()

        # Regression guard: static scenes should still promote IDLE -> MONITORING.
        assert sm.state == AppState.MONITORING
        assert service._frames_skipped > 0

    def test_preview_sink_reuses_cached_boxes_on_skipped_frames(self):
        from neurabreak.core.config import AppConfig
        from neurabreak.ai.detection_service import DetectionService

        frames = [_FakeGrayFrame(20.0) for _ in range(20)]
        box = {
            "cls": 0,
            "cls_name": "face_present",
            "confidence": 0.92,
            "conf": 0.92,
            "x1": 1,
            "y1": 2,
            "x2": 3,
            "y2": 4,
        }
        results = [
            DetectionResult(
                presence=True,
                posture_class="face_present",
                confidence=0.92,
                raw_boxes=[box],
            )
        ] * 20

        sink = MagicMock()
        service = DetectionService(
            make_mock_camera(frames),
            make_mock_engine(results),
            PostureStateMachine(fps=5),
            AppConfig(),
        )
        service.set_frame_sink(sink)

        with patch.dict("sys.modules", {"cv2": _FakeCV2, "numpy": _FakeNumpy}):
            service.start()
            time.sleep(0.25)
            service.stop()

        infer_calls = service.engine.infer.call_count
        sink_calls = len(sink.call_args_list)
        calls_with_boxes = sum(1 for call in sink.call_args_list if call.args[1] == [box])

        assert sink_calls > 0
        # If cached boxes are reused during skipped frames, calls-with-boxes should
        # outnumber actual inference calls.
        assert calls_with_boxes > infer_calls


class TestDetectionServiceStats:
    def test_last_latency_updated_after_inference(self):
        from neurabreak.core.config import AppConfig
        from neurabreak.ai.detection_service import DetectionService

        result = DetectionResult(presence=False, posture_class=None, confidence=0.0, latency_ms=42.5)
        service = DetectionService(
            make_mock_camera([object()]),
            make_mock_engine([result]),
            PostureStateMachine(),
            AppConfig(),
        )
        service.start()
        time.sleep(0.2)
        service.stop()

        assert service.last_latency_ms == pytest.approx(42.5, abs=0.1)
