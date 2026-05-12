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

    @requires_numpy
    def test_onnx_nms_output_maps_class_names(self):
        import numpy as np

        engine = InferenceEngine(model_path="fake_model.onnx", confidence_threshold=0.4)
        boxes = engine._parse_onnx_output(
            np.array([[[10, 20, 100, 120, 0.9, 2]]], dtype=np.float32),
            frame_shape=(320, 320),
        )
        result = engine._parse_raw_boxes(boxes, latency_ms=1.0)

        assert boxes[0]["cls_name"] == "posture_bad"
        assert result.presence is True
        assert result.posture_class == "posture_bad"

    @requires_numpy
    def test_onnx_raw_yolo_output_decodes_class_scores(self):
        import numpy as np

        engine = InferenceEngine(model_path="fake_model.onnx", confidence_threshold=0.4)
        # Shape is [1, 4 + num_classes, anchors].
        preds = np.array(
            [[[160], [160], [40], [60], [0.1], [0.2], [0.95], [0.0]]],
            dtype=np.float32,
        )

        boxes = engine._parse_onnx_output(preds, frame_shape=(320, 320))
        result = engine._parse_raw_boxes(boxes, latency_ms=1.0)

        assert boxes[0]["cls_name"] == "posture_bad"
        assert result.posture_class == "posture_bad"
