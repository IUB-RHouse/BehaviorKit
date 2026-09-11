"""Broader tests for module/inference.py: the pure helper functions
(_best_yolo_box, _mp_image_from_bgr, device detection), and MultiModelRunner
coverage for face/pose landmarks, speech, sentiment, cross-modality
warnings, and the get_runner() singleton -- all with fake model backends so
nothing heavy actually loads.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import module.inference as inf


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class _FakeBoxes:
    def __init__(self, confs, xyxys):
        self.conf = torch.tensor(confs, dtype=torch.float32)
        self.xyxy = torch.tensor(xyxys, dtype=torch.float32)

    def __len__(self):
        return len(self.conf)


class _FakeYoloResult:
    def __init__(self, boxes):
        self.boxes = boxes


class TestBestYoloBox:
    def test_picks_highest_confidence_box(self):
        boxes = _FakeBoxes(confs=[0.3, 0.9, 0.5], xyxys=[[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]])
        box = inf._best_yolo_box(_FakeYoloResult(boxes))
        assert box == [2.0, 2.0, 3.0, 3.0]

    def test_returns_none_when_boxes_is_none(self):
        assert inf._best_yolo_box(_FakeYoloResult(None)) is None

    def test_returns_none_when_boxes_empty(self):
        assert inf._best_yolo_box(_FakeYoloResult(_FakeBoxes([], []))) is None


def test_mp_image_from_bgr_builds_real_mediapipe_image():
    import mediapipe as mp

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    img = inf._mp_image_from_bgr(frame)
    assert isinstance(img, mp.Image)


def test_device_str_is_cpu_or_cuda():
    assert inf._device_str() in ("cpu", "cuda")


# ---------------------------------------------------------------------------
# MultiModelRunner, with every model backend faked
# ---------------------------------------------------------------------------

class _FakeYolo:
    def predict(self, **kwargs):
        return [None]


class _Point:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _FakeLandmarkResult:
    def __init__(self, attr, landmarks):
        setattr(self, attr, landmarks)


class _FakeLandmarker:
    def __init__(self, attr, landmarks):
        self.attr = attr
        self.landmarks = landmarks

    def detect(self, mp_img):
        return _FakeLandmarkResult(self.attr, self.landmarks)


class _FakeGazeModel:
    method_name = "fake_gaze"

    def predict(self, crop):
        return 0.1, 0.2


class _FakeEmonetModel:
    method_name = "emonet"

    def predict(self, crop):
        return {"expression": "neutral", "expression_scores": {}, "valence": 0.0, "arousal": 0.0}


class _FakeWhisper:
    def __init__(self, text="hello world"):
        self.device = "cpu"
        self.text = text
        self.calls = []

    def transcribe(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "text": self.text,
            "segments": [{"words": [{"word": "hello", "start": 0.0, "end": 0.4},
                                     {"word": "world", "start": 0.4, "end": 0.8}]}],
        }


class _FakeSentiment:
    def __init__(self, label="positive", score=0.87):
        self.label, self.score = label, score
        self.calls = []

    def __call__(self, text, truncation=True):
        self.calls.append(text)
        return [{"label": self.label, "score": self.score}]


def _make_runner(monkeypatch, enabled: dict, *, cfg: dict | None = None, **fakes):
    reg = inf.reg

    monkeypatch.setattr(reg, "modality_enabled", lambda name: enabled.get(name, False))
    monkeypatch.setattr(reg, "modality_cfg", lambda name: (cfg or {}).get(name, {}))
    monkeypatch.setattr(reg, "get_config", lambda: {})
    monkeypatch.setattr(reg, "landmarks", lambda: {})
    monkeypatch.setattr(reg, "hypers", lambda: {"blink_threshold": 0.3})
    monkeypatch.setattr(reg, "videos_dir", lambda: "demo_data")
    monkeypatch.setattr(reg, "get_face_detector", lambda: fakes.get("yolo", _FakeYolo()))
    monkeypatch.setattr(
        reg, "get_face_landmarker",
        lambda cpu=True: fakes.get("face_lm", _FakeLandmarker("face_landmarks", [])),
    )
    monkeypatch.setattr(
        reg, "get_pose_landmarker",
        lambda cpu=True: fakes.get("pose_lm", _FakeLandmarker("pose_landmarks", [])),
    )
    monkeypatch.setattr(reg, "get_gaze_model", lambda: (fakes.get("gaze", _FakeGazeModel()), "cpu"))
    monkeypatch.setattr(reg, "get_emonet_model", lambda: (fakes.get("emonet", _FakeEmonetModel()), "cpu"))
    monkeypatch.setattr(reg, "get_whisper_model", lambda device=None: fakes.get("whisper", _FakeWhisper()))
    monkeypatch.setattr(reg, "get_sentiment_pipeline", lambda device=-1: fakes.get("sentiment", _FakeSentiment()))
    monkeypatch.setattr(reg, "resolve_pose_variant", lambda: "full")
    monkeypatch.setattr(reg, "resolve_whisper_size", lambda: "base")

    monkeypatch.setattr(inf, "_best_yolo_box", lambda result: fakes.get("face_box", [0, 0, 10, 10]))

    return inf.MultiModelRunner(use_gpu=False)


def _sample(t_sec: int = 0, audio: np.ndarray | None = None) -> inf.Sample:
    return inf.Sample(
        frame_bgr=np.zeros((20, 20, 3), dtype=np.uint8),
        audio_1s_16k_mono=audio if audio is not None else np.zeros(16000, dtype=np.float32),
        t_sec=t_sec,
    )


class TestFaceAndPoseLandmarks:
    def test_face_landmarks_populate_result_when_enabled(self, monkeypatch):
        fake_lm = _FakeLandmarker("face_landmarks", [[_Point(0.1, 0.2, 0.3)]])
        runner = _make_runner(monkeypatch, {"face_landmarks": True}, face_lm=fake_lm)

        result = runner.process(_sample())

        assert result.face_landmarks == [[{"x": 0.1, "y": 0.2, "z": 0.3}]]

    def test_face_landmarks_empty_when_disabled(self, monkeypatch):
        runner = _make_runner(monkeypatch, {"face_landmarks": False})
        result = runner.process(_sample())
        assert result.face_landmarks == []

    def test_pose_landmarks_populate_result_when_enabled(self, monkeypatch):
        fake_lm = _FakeLandmarker("pose_landmarks", [[_Point(0.4, 0.5, 0.6)]])
        runner = _make_runner(monkeypatch, {"pose_landmarks": True}, pose_lm=fake_lm)

        result = runner.process(_sample())

        assert result.pose_landmarks == [[{"x": 0.4, "y": 0.5, "z": 0.6}]]


class TestSpeechAndSentiment:
    def test_speech_does_not_run_before_buffer_full(self, monkeypatch):
        fake_whisper = _FakeWhisper()
        runner = _make_runner(
            monkeypatch, {"speech": True}, cfg={"speech": {"chunk_seconds": 3.0}}, whisper=fake_whisper
        )
        # one 1s chunk while chunk_seconds=3.0 -> shouldn't trigger whisper yet
        result = runner.process(_sample(audio=np.zeros(16000, dtype=np.float32)))

        assert result.whisper_text == ""
        assert fake_whisper.calls == []

    def test_speech_runs_once_buffer_full_and_resets(self, monkeypatch):
        fake_whisper = _FakeWhisper(text="hello world")
        runner = _make_runner(
            monkeypatch, {"speech": True}, cfg={"speech": {"chunk_seconds": 1.0}}, whisper=fake_whisper
        )

        result = runner.process(_sample(audio=np.zeros(16000, dtype=np.float32)))

        assert result.whisper_text == "hello world"
        assert result.whisper_words == [
            {"word": "hello", "start": 0.0, "end": 0.4},
            {"word": "world", "start": 0.4, "end": 0.8},
        ]
        assert len(fake_whisper.calls) == 1
        assert result.debug["audio_buffer_size"] == 0  # buffer cleared after a run

    def test_sentiment_runs_after_speech_when_both_enabled(self, monkeypatch):
        fake_whisper = _FakeWhisper(text="I am happy")
        fake_sentiment = _FakeSentiment(label="positive", score=0.9)
        runner = _make_runner(
            monkeypatch,
            {"speech": True, "sentiment": True},
            cfg={"speech": {"chunk_seconds": 1.0}},
            whisper=fake_whisper,
            sentiment=fake_sentiment,
        )

        result = runner.process(_sample(audio=np.zeros(16000, dtype=np.float32)))

        assert result.sentiment == {"label": "positive", "score": 0.9}
        assert fake_sentiment.calls == ["I am happy"]

    def test_sentiment_skipped_when_disabled_even_if_speech_ran(self, monkeypatch):
        fake_whisper = _FakeWhisper(text="hello")
        runner = _make_runner(
            monkeypatch, {"speech": True, "sentiment": False},
            cfg={"speech": {"chunk_seconds": 1.0}}, whisper=fake_whisper,
        )
        result = runner.process(_sample(audio=np.zeros(16000, dtype=np.float32)))
        assert result.sentiment is None

    def test_run_sentiment_returns_none_for_empty_text(self, monkeypatch):
        fake_sentiment = _FakeSentiment()
        runner = _make_runner(monkeypatch, {"sentiment": True}, sentiment=fake_sentiment)
        assert runner._run_sentiment("") is None
        assert fake_sentiment.calls == []


class TestCrossModalityWarnings:
    def test_warns_when_gaze_enabled_without_face_detection(self, monkeypatch, capsys):
        _make_runner(monkeypatch, {"gaze": True, "face_detection": False})
        assert "gaze is enabled but face_detection is disabled" in capsys.readouterr().out

    def test_warns_when_emotion_enabled_without_face_detection(self, monkeypatch, capsys):
        _make_runner(monkeypatch, {"emotion": True, "face_detection": False})
        assert "emotion is enabled but face_detection is disabled" in capsys.readouterr().out

    def test_warns_when_sentiment_enabled_without_speech(self, monkeypatch, capsys):
        _make_runner(monkeypatch, {"sentiment": True, "speech": False})
        assert "sentiment is enabled but speech is disabled" in capsys.readouterr().out

    def test_no_warnings_when_dependencies_satisfied(self, monkeypatch, capsys):
        _make_runner(monkeypatch, {"gaze": True, "emotion": True, "face_detection": True,
                                    "sentiment": True, "speech": True},
                      cfg={"speech": {"chunk_seconds": 3.0}})
        out = capsys.readouterr().out
        assert "WARNING" not in out


class TestDescribeModalitiesFull:
    def test_all_modalities_reported_when_enabled(self, monkeypatch):
        runner = _make_runner(
            monkeypatch,
            {m: True for m in inf.MODALITIES},
            cfg={"sentiment": {"model": "some-model"}},
        )
        info = runner.describe_modalities()

        assert info["face_detection"] == {"enabled": True}
        assert info["pose_landmarks"] == {"enabled": True, "variant": "full"}
        assert info["gaze"] == {"enabled": True, "method": "fake_gaze"}
        assert info["emotion"] == {"enabled": True}
        assert info["speech"]["model"] == "base"
        assert info["sentiment"] == {"enabled": True, "model": "some-model"}

    def test_disabled_modalities_report_null_fields(self, monkeypatch):
        runner = _make_runner(monkeypatch, {})
        info = runner.describe_modalities()

        assert info["gaze"] == {"enabled": False, "method": None}
        assert info["pose_landmarks"] == {"enabled": False, "variant": None}
        assert info["speech"]["model"] is None
        assert info["sentiment"] == {"enabled": False, "model": None}


class TestDebugFields:
    def test_frame_counter_increments_and_metadata_present(self, monkeypatch):
        runner = _make_runner(monkeypatch, {})
        r1 = runner.process(_sample())
        r2 = runner.process(_sample())

        assert r1.debug["frame_counter"] == 1
        assert r2.debug["frame_counter"] == 2
        assert r1.debug["videos_dir"] == "demo_data"
        assert r1.debug["blink_threshold"] == 0.3
        assert r1.debug["device"] == "cpu"


class TestGetRunnerSingleton:
    def test_get_runner_builds_once_and_reuses(self, monkeypatch):
        monkeypatch.setattr(inf, "_global_runner", None)

        build_calls = []
        real_init = inf.MultiModelRunner.__init__

        def _counting_init(self, use_gpu=True):
            build_calls.append(use_gpu)
            real_init(self, use_gpu=use_gpu)

        _make_runner_reg_patches(monkeypatch)
        monkeypatch.setattr(inf.MultiModelRunner, "__init__", _counting_init)

        r1 = inf.get_runner(use_gpu=False)
        r2 = inf.get_runner(use_gpu=False)

        assert r1 is r2
        assert build_calls == [False]

        monkeypatch.setattr(inf, "_global_runner", None)

    def test_run_models_on_sample_delegates_to_runner(self, monkeypatch):
        monkeypatch.setattr(inf, "_global_runner", None)
        _make_runner_reg_patches(monkeypatch)

        result = inf.run_models_on_sample(_sample(t_sec=7), use_gpu=False)

        assert result.t_sec == 7
        monkeypatch.setattr(inf, "_global_runner", None)


def _make_runner_reg_patches(monkeypatch):
    """Same backend fakes as _make_runner, without constructing a runner
    (used by get_runner()/run_models_on_sample() tests, which construct
    their own runner internally)."""
    reg = inf.reg
    monkeypatch.setattr(reg, "modality_enabled", lambda name: False)
    monkeypatch.setattr(reg, "modality_cfg", lambda name: {})
    monkeypatch.setattr(reg, "get_config", lambda: {})
    monkeypatch.setattr(reg, "landmarks", lambda: {})
    monkeypatch.setattr(reg, "hypers", lambda: {})
    monkeypatch.setattr(reg, "videos_dir", lambda: "demo_data")
    monkeypatch.setattr(reg, "get_face_detector", lambda: _FakeYolo())
    monkeypatch.setattr(reg, "get_face_landmarker", lambda cpu=True: _FakeLandmarker("face_landmarks", []))
    monkeypatch.setattr(reg, "get_pose_landmarker", lambda cpu=True: _FakeLandmarker("pose_landmarks", []))
    monkeypatch.setattr(reg, "get_gaze_model", lambda: (_FakeGazeModel(), "cpu"))
    monkeypatch.setattr(reg, "get_emonet_model", lambda: (_FakeEmonetModel(), "cpu"))
    monkeypatch.setattr(reg, "get_whisper_model", lambda device=None: _FakeWhisper())
    monkeypatch.setattr(reg, "get_sentiment_pipeline", lambda device=-1: _FakeSentiment())
