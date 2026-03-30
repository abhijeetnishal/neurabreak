"""Integration smoke tests — detection pipeline end-to-end."""

from __future__ import annotations


class TestDetectionPipelineImports:
    def test_imports_do_not_raise(self):
        from neurabreak.ai.postprocessor import CLASS_NAMES

        assert CLASS_NAMES[0] == "face_present"

    def test_camera_service_can_be_instantiated(self):
        from neurabreak.ai.camera import FrameCaptureService

        service = FrameCaptureService(camera_index=0, fps=5)
        assert service.fps == 5
        assert not service.is_running

    def test_inference_engine_can_be_instantiated(self):
        from neurabreak.ai.engine import InferenceEngine

        engine = InferenceEngine(model_path="/fake/path/model.onnx")
        assert not engine.is_loaded

    def test_state_machine_starts_idle(self):
        from neurabreak.core.state_machine import AppState, PostureStateMachine

        sm = PostureStateMachine()
        assert sm.state == AppState.IDLE


class TestEventBusPipeline:
    def test_events_flow_from_state_machine_to_subscriber(self):
        from neurabreak.core.events import Event, EventType, bus
        from neurabreak.core.state_machine import PostureStateMachine

        received = []
        bus.subscribe(EventType.SESSION_STARTED, received.append)

        sm = PostureStateMachine(fps=5)
        for _ in range(15):
            sm.process(present=True, posture_class="posture_good", confidence=0.9)

        bus.unsubscribe(EventType.SESSION_STARTED, received.append)

        assert len(received) == 1
        assert received[0].type == EventType.SESSION_STARTED
