"""Tests for the inference engine interface."""

from __future__ import annotations

import importlib

import pytest

from neurabreak.ai.engine import DetectionResult, InferenceEngine

numpy_available = importlib.util.find_spec("numpy") is not None
requires_numpy = pytest.mark.skipif(not numpy_available, reason="numpy not installed (install ai extras)")


class TestDetectionResult:
    def test_good_posture_not_flagged_as_bad(self):
        result = DetectionResult(presence=True, posture_class="posture_good", confidence=0.95)
        assert result.is_bad_posture is False

    def test_none_posture_not_flagged_as_bad(self):
        result = DetectionResult(presence=False, posture_class=None, confidence=0.0)
        assert result.is_bad_posture is False

    def test_slouch_flagged_as_bad(self):
        result = DetectionResult(presence=True, posture_class="posture_bad", confidence=0.88)
        assert result.is_bad_posture is True

    def test_head_forward_flagged_as_bad(self):
        result = DetectionResult(presence=True, posture_class="posture_bad", confidence=0.9)
        assert result.is_bad_posture is True

    def test_phone_detected_default_false(self):
        result = DetectionResult(presence=True, posture_class="posture_good", confidence=0.9)
        assert result.phone_detected is False


class TestInferenceEngineInterface:
    @requires_numpy
    def test_infer_raises_before_load(self):
        engine = InferenceEngine(model_path="fake_model.onnx")
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            import numpy as np
            engine.infer(np.zeros((480, 640, 3), dtype="uint8"))

    def test_load_sets_loaded_flag(self):
        engine = InferenceEngine(model_path="fake_model.onnx")
        assert engine.is_loaded is False
        engine.load()  # stub — doesn't load a real file
        assert engine.is_loaded is True

    @requires_numpy
    def test_stub_infer_returns_no_presence(self):
        import numpy as np

        engine = InferenceEngine(model_path="fake_model.onnx")
        engine.load()
        result = engine.infer(np.zeros((480, 640, 3), dtype="uint8"))
        assert isinstance(result, DetectionResult)
        assert result.presence is False

    def test_unload_clears_loaded_flag(self):
        engine = InferenceEngine(model_path="fake_model.onnx")
        engine.load()
        engine.unload()
        assert engine.is_loaded is False
