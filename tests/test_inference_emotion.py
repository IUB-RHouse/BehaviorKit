"""Tests that MultiModelRunner wires the `emotion` modality correctly:
enabled/disabled gating, sharing the YOLO face crop with gaze, and surfacing
results in Result/describe_modalities(). Every dependency (YOLO, gaze,
emonet) is faked so these run without loading any real models.
"""
import numpy as np
import pytest

import module.inference as inf


class _FakeYolo:
    def predict(self, **kwargs):
        return [None]  # content is irrelevant; _best_yolo_box is patched below


class _FakeGazeModel:
    method_name = "fake_gaze"

    def predict(self, crop):
        return 0.1, 0.2


class _FakeEmonetModel:
    method_name = "emonet"

    def __init__(self):
        self.calls = []

    def predict(self, crop):
        self.calls.append(crop)
        return {
            "expression": "happy",
            "expression_scores": {"happy": 1.0},
            "valence": 0.5,
            "arousal": -0.25,
        }


def _make_runner(monkeypatch, enabled: dict, emonet_model=None, face_box=(0, 0, 10, 10)):
    reg = inf.reg

    monkeypatch.setattr(reg, "modality_enabled", lambda name: enabled.get(name, False))
    monkeypatch.setattr(reg, "modality_cfg", lambda name: {"chunk_seconds": 3.0, "word_timestamps": True})
    monkeypatch.setattr(reg, "get_config", lambda: {})
    monkeypatch.setattr(reg, "landmarks", lambda: {})
    monkeypatch.setattr(reg, "hypers", lambda: {})
    monkeypatch.setattr(reg, "videos_dir", lambda: "demo_data")
    monkeypatch.setattr(reg, "get_face_detector", lambda: _FakeYolo())
    monkeypatch.setattr(reg, "get_gaze_model", lambda: (_FakeGazeModel(), "cpu"))
    monkeypatch.setattr(reg, "get_emonet_model", lambda: (emonet_model or _FakeEmonetModel(), "cpu"))

    monkeypatch.setattr(inf, "_best_yolo_box", lambda result: list(face_box) if face_box else None)

    return inf.MultiModelRunner(use_gpu=False)


def _sample() -> inf.Sample:
    return inf.Sample(
        frame_bgr=np.zeros((20, 20, 3), dtype=np.uint8),
        audio_1s_16k_mono=np.zeros(16000, dtype=np.float32),
    )


class TestEmotionGating:
    def test_runs_and_populates_result_when_enabled_with_face(self, monkeypatch):
        fake_emonet = _FakeEmonetModel()
        runner = _make_runner(monkeypatch, {"face_detection": True, "emotion": True}, emonet_model=fake_emonet)

        result = runner.process(_sample())

        assert result.emotion == {
            "expression": "happy",
            "expression_scores": {"happy": 1.0},
            "valence": 0.5,
            "arousal": -0.25,
        }
        assert len(fake_emonet.calls) == 1

    def test_skipped_when_emotion_disabled(self, monkeypatch):
        fake_emonet = _FakeEmonetModel()
        runner = _make_runner(monkeypatch, {"face_detection": True, "emotion": False}, emonet_model=fake_emonet)

        result = runner.process(_sample())

        assert result.emotion is None
        assert fake_emonet.calls == []

    def test_skipped_when_face_detection_disabled(self, monkeypatch):
        fake_emonet = _FakeEmonetModel()
        runner = _make_runner(monkeypatch, {"face_detection": False, "emotion": True}, emonet_model=fake_emonet)

        result = runner.process(_sample())

        assert result.emotion is None
        assert fake_emonet.calls == []

    def test_skipped_when_no_face_found_in_frame(self, monkeypatch):
        fake_emonet = _FakeEmonetModel()
        runner = _make_runner(
            monkeypatch, {"face_detection": True, "emotion": True}, emonet_model=fake_emonet, face_box=None
        )

        result = runner.process(_sample())

        assert result.emotion is None
        assert fake_emonet.calls == []

    def test_gaze_and_emotion_share_the_same_face_crop(self, monkeypatch):
        fake_emonet = _FakeEmonetModel()
        runner = _make_runner(
            monkeypatch,
            {"face_detection": True, "gaze": True, "emotion": True},
            emonet_model=fake_emonet,
        )

        result = runner.process(_sample())

        assert result.gaze is not None
        assert result.emotion is not None
        assert len(fake_emonet.calls) == 1


class TestDescribeModalities:
    def test_reports_emotion_enabled_state(self, monkeypatch):
        runner = _make_runner(monkeypatch, {"emotion": True})
        assert runner.describe_modalities()["emotion"] == {"enabled": True}

    def test_reports_emotion_disabled_state(self, monkeypatch):
        runner = _make_runner(monkeypatch, {"emotion": False})
        assert runner.describe_modalities()["emotion"] == {"enabled": False}


def test_emotion_in_modalities_tuple():
    assert "emotion" in inf.MODALITIES
