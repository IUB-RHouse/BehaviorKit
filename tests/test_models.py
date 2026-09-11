"""Tests for module/models.py: config loading/validation, quality-tier
derivation logic, the bundled-weight loaders (face detector, mediapipe
landmarkers, gaze), the ImportError-wrapping for each optional dependency,
and whisper/sentiment with their network-calling factory functions stubbed
out (they auto-download real weights on first use otherwise).
"""
import textwrap
from pathlib import Path

import pytest

import module.models as models

YOLO_WEIGHTS = Path("data/models/yolov8-face/yolov8m-face.pt")
FACE_LANDMARKER = Path("data/models/mediapipe/face_landmarker.task")
POSE_LANDMARKER_LITE = Path("data/models/mediapipe/pose_landmarker_lite.task")


_CACHED_FNS = (
    models.get_config,
    models.get_face_detector,
    models.get_face_landmarker,
    models.get_pose_landmarker,
    models.get_emonet_model,
    models.get_gaze_model,
    models.get_whisper_model,
    models.get_sentiment_pipeline,
)


@pytest.fixture(autouse=True)
def _clear_caches():
    for fn in _CACHED_FNS:
        fn.cache_clear()
    yield
    for fn in _CACHED_FNS:
        fn.cache_clear()


def _write_config(tmp_path: Path, body: str) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent(body))
    return cfg


class TestGetConfig:
    def test_missing_file_raises(self, monkeypatch):
        monkeypatch.setenv("CONFIG_PATH", "definitely/does/not/exist.yaml")
        with pytest.raises(FileNotFoundError):
            models.get_config()

    def test_missing_modalities_key_raises(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, """\
            dataset:
              videos_dir: "demo_data"
            """)
        monkeypatch.setenv("CONFIG_PATH", str(cfg))
        with pytest.raises(KeyError):
            models.get_config()

    def test_missing_dataset_key_raises(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, """\
            modalities: {}
            """)
        monkeypatch.setenv("CONFIG_PATH", str(cfg))
        with pytest.raises(KeyError):
            models.get_config()

    def test_empty_file_is_treated_as_empty_dict_then_fails_guardrail(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, "")
        monkeypatch.setenv("CONFIG_PATH", str(cfg))
        with pytest.raises(KeyError):
            models.get_config()

    def test_valid_config_loads(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, """\
            quality: "high"
            dataset:
              videos_dir: "somewhere"
            modalities: {}
            """)
        monkeypatch.setenv("CONFIG_PATH", str(cfg))
        data = models.get_config()
        assert data["quality"] == "high"
        assert data["dataset"]["videos_dir"] == "somewhere"


class TestSimpleAccessors:
    def test_modality_cfg_missing_modality_returns_empty_dict(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, """\
            dataset:
              videos_dir: "demo_data"
            modalities: {}
            """)
        monkeypatch.setenv("CONFIG_PATH", str(cfg))
        assert models.modality_cfg("nonexistent") == {}

    def test_modality_enabled_defaults_true_when_modality_absent(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, """\
            dataset:
              videos_dir: "demo_data"
            modalities: {}
            """)
        monkeypatch.setenv("CONFIG_PATH", str(cfg))
        assert models.modality_enabled("speech") is True

    def test_modality_enabled_respects_explicit_false(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, """\
            dataset:
              videos_dir: "demo_data"
            modalities:
              speech:
                enabled: false
            """)
        monkeypatch.setenv("CONFIG_PATH", str(cfg))
        assert models.modality_enabled("speech") is False

    def test_quality_defaults_to_medium(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, """\
            dataset:
              videos_dir: "demo_data"
            modalities: {}
            """)
        monkeypatch.setenv("CONFIG_PATH", str(cfg))
        assert models.quality() == "medium"

    def test_landmarks_hypers_videos_dir(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, """\
            dataset:
              videos_dir: "my_videos"
            modalities: {}
            landmarks:
              nose_landmarks: [1, 2, 3]
            hyperparameters:
              blink_threshold: 0.5
            """)
        monkeypatch.setenv("CONFIG_PATH", str(cfg))
        assert models.landmarks() == {"nose_landmarks": [1, 2, 3]}
        assert models.hypers() == {"blink_threshold": 0.5}
        assert models.videos_dir() == "my_videos"


class TestResolvePoseVariant:
    @pytest.mark.parametrize(
        "quality,expected",
        [("minimum", "lite"), ("low", "lite"), ("medium", "full"), ("high", "full"), ("maximum", "heavy")],
    )
    def test_derives_from_quality_when_variant_unset(self, tmp_path, monkeypatch, quality, expected):
        cfg = _write_config(tmp_path, f"""\
            quality: "{quality}"
            dataset:
              videos_dir: "demo_data"
            modalities: {{}}
            """)
        monkeypatch.setenv("CONFIG_PATH", str(cfg))
        assert models.resolve_pose_variant() == expected

    def test_explicit_variant_wins_over_quality(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, """\
            quality: "minimum"
            dataset:
              videos_dir: "demo_data"
            modalities:
              pose_landmarks:
                variant: "heavy"
            """)
        monkeypatch.setenv("CONFIG_PATH", str(cfg))
        assert models.resolve_pose_variant() == "heavy"

    def test_invalid_explicit_variant_falls_back_to_quality(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, """\
            quality: "medium"
            dataset:
              videos_dir: "demo_data"
            modalities:
              pose_landmarks:
                variant: "not-a-real-variant"
            """)
        monkeypatch.setenv("CONFIG_PATH", str(cfg))
        assert models.resolve_pose_variant() == "full"


class TestResolveWhisperSize:
    @pytest.mark.parametrize(
        "quality,expected",
        [("minimum", "tiny"), ("low", "tiny"), ("medium", "base"), ("high", "large-v2"), ("maximum", "large-v3")],
    )
    def test_derives_from_quality_when_model_unset(self, tmp_path, monkeypatch, quality, expected):
        cfg = _write_config(tmp_path, f"""\
            quality: "{quality}"
            dataset:
              videos_dir: "demo_data"
            modalities: {{}}
            """)
        monkeypatch.setenv("CONFIG_PATH", str(cfg))
        assert models.resolve_whisper_size() == expected

    def test_explicit_model_wins_over_quality(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, """\
            quality: "minimum"
            dataset:
              videos_dir: "demo_data"
            modalities:
              speech:
                model: "small"
            """)
        monkeypatch.setenv("CONFIG_PATH", str(cfg))
        assert models.resolve_whisper_size() == "small"


@pytest.mark.skipif(not YOLO_WEIGHTS.exists(), reason="yolov8-face weights not present")
def test_get_face_detector_loads_bundled_weights(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, f"""\
        dataset:
          videos_dir: "demo_data"
        modalities:
          face_detection:
            model_path: "{YOLO_WEIGHTS.as_posix()}"
        """)
    monkeypatch.setenv("CONFIG_PATH", str(cfg))
    detector = models.get_face_detector()
    assert detector is not None


@pytest.mark.skipif(not FACE_LANDMARKER.exists(), reason="mediapipe face_landmarker not present")
def test_get_face_landmarker_loads_bundled_weights(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, f"""\
        dataset:
          videos_dir: "demo_data"
        modalities:
          face_landmarks:
            model_path: "{FACE_LANDMARKER.as_posix()}"
        """)
    monkeypatch.setenv("CONFIG_PATH", str(cfg))
    landmarker = models.get_face_landmarker(cpu=True)
    assert landmarker is not None


@pytest.mark.skipif(not POSE_LANDMARKER_LITE.exists(), reason="mediapipe pose_landmarker_lite not present")
def test_get_pose_landmarker_loads_bundled_weights(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, f"""\
        quality: "minimum"
        dataset:
          videos_dir: "demo_data"
        modalities:
          pose_landmarks:
            model_path: "{POSE_LANDMARKER_LITE.as_posix()}"
        """)
    monkeypatch.setenv("CONFIG_PATH", str(cfg))
    landmarker = models.get_pose_landmarker(cpu=True)
    assert landmarker is not None


def test_get_face_landmarker_cpu_false_falls_back_on_gpu_failure(tmp_path, monkeypatch):
    """cpu=False tries the GPU delegate first, falling back to CPU on
    failure. Real mediapipe GPU delegate init doesn't reliably raise on
    machines without a working GPU context (it can hang instead, since
    that path is Linux-only per the code's own comment) -- so this mocks
    _make_face_landmarker to simulate a GPU failure rather than exercising
    real GPU init, to test the fallback logic without the hang risk."""
    cfg = _write_config(tmp_path, """\
        dataset:
          videos_dir: "demo_data"
        modalities:
          face_landmarks:
            model_path: "unused/placeholder.task"
        """)
    monkeypatch.setenv("CONFIG_PATH", str(cfg))

    calls = []

    def _fake_make(mp_python, mp_vision, model_path, delegate):
        calls.append(delegate)
        if len(calls) == 1:
            raise RuntimeError("simulated GPU delegate failure")
        return "cpu-landmarker"

    monkeypatch.setattr(models, "_make_face_landmarker", _fake_make)
    result = models.get_face_landmarker(cpu=False)

    assert result == "cpu-landmarker"
    assert len(calls) == 2


def test_get_pose_landmarker_cpu_false_falls_back_on_gpu_failure(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, """\
        quality: "minimum"
        dataset:
          videos_dir: "demo_data"
        modalities:
          pose_landmarks:
            model_path: "unused/placeholder_lite.task"
        """)
    monkeypatch.setenv("CONFIG_PATH", str(cfg))

    calls = []

    def _fake_make(mp_python, mp_vision, model_path, delegate):
        calls.append(delegate)
        if len(calls) == 1:
            raise RuntimeError("simulated GPU delegate failure")
        return "cpu-landmarker"

    monkeypatch.setattr(models, "_make_pose_landmarker", _fake_make)
    result = models.get_pose_landmarker(cpu=False)

    assert result == "cpu-landmarker"
    assert len(calls) == 2


def test_get_gaze_model_builds_default_mobilegaze_backend(tmp_path, monkeypatch):
    weights = Path("data/models/mobilegaze/mobilenetv2.pt")
    if not weights.exists():
        pytest.skip("mobilegaze weights not present")
    cfg = _write_config(tmp_path, f"""\
        dataset:
          videos_dir: "demo_data"
        modalities:
          gaze:
            method: "mobilegaze"
            mobilegaze_arch: "mobilenetv2"
            mobilegaze_model: "{weights.as_posix()}"
        """)
    monkeypatch.setenv("CONFIG_PATH", str(cfg))
    backend, device = models.get_gaze_model()
    assert backend.method_name == "mobilegaze"
    assert device in ("cpu", "cuda")


def test_has_cuda_returns_a_bool():
    assert models._has_cuda() in (True, False)


class TestImportErrorWrapping:
    """Each loader wraps a missing optional dependency in a friendlier
    ImportError. These simulate the dependency being absent via
    builtins.__import__ rather than actually uninstalling anything.
    """

    @staticmethod
    def _blocking_import(blocked_name):
        import builtins
        real_import = builtins.__import__

        def _fake(name, *args, **kwargs):
            if name == blocked_name or name.startswith(blocked_name + "."):
                raise ImportError(f"simulated missing {blocked_name}")
            return real_import(name, *args, **kwargs)

        return _fake

    def test_get_config_wraps_missing_yaml(self, tmp_path, monkeypatch):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("modalities: {}\ndataset:\n  videos_dir: demo_data\n")
        monkeypatch.setenv("CONFIG_PATH", str(cfg))
        monkeypatch.setattr("builtins.__import__", self._blocking_import("yaml"))

        with pytest.raises(ImportError, match="PyYAML is required"):
            models.get_config()

    def test_get_face_landmarker_wraps_missing_mediapipe(self, monkeypatch):
        monkeypatch.setattr("builtins.__import__", self._blocking_import("mediapipe"))
        with pytest.raises(ImportError, match="mediapipe is required"):
            models.get_face_landmarker(cpu=True)

    def test_get_pose_landmarker_wraps_missing_mediapipe(self, monkeypatch):
        monkeypatch.setattr("builtins.__import__", self._blocking_import("mediapipe"))
        with pytest.raises(ImportError, match="mediapipe is required"):
            models.get_pose_landmarker(cpu=True)

    def test_get_face_detector_wraps_missing_ultralytics(self, monkeypatch):
        monkeypatch.setattr("builtins.__import__", self._blocking_import("ultralytics"))
        with pytest.raises(ImportError, match="ultralytics is required"):
            models.get_face_detector()

    def test_get_gaze_model_wraps_missing_torch(self, monkeypatch):
        monkeypatch.setattr("builtins.__import__", self._blocking_import("torch"))
        with pytest.raises(ImportError, match="PyTorch is required"):
            models.get_gaze_model()

    def test_get_emonet_model_wraps_missing_torch(self, monkeypatch):
        monkeypatch.setattr("builtins.__import__", self._blocking_import("torch"))
        with pytest.raises(ImportError, match="PyTorch is required"):
            models.get_emonet_model()

    def test_get_whisper_model_wraps_missing_whisper(self, monkeypatch):
        monkeypatch.setattr("builtins.__import__", self._blocking_import("whisper"))
        with pytest.raises(ImportError, match="openai-whisper is required"):
            models.get_whisper_model()

    def test_get_sentiment_pipeline_wraps_missing_transformers(self, monkeypatch):
        monkeypatch.setattr("builtins.__import__", self._blocking_import("transformers"))
        with pytest.raises(ImportError, match="transformers is required"):
            models.get_sentiment_pipeline()


class TestWhisperAndSentimentWithStubbedFactories:
    """get_whisper_model/get_sentiment_pipeline auto-download real weights
    from the network on first use; these stub the underlying factory
    function (whisper.load_model / transformers.pipeline) so the rest of
    each function's logic (device resolution, config lookup) is still
    exercised for real, without touching the network.
    """

    def test_get_whisper_model_resolves_device_and_size(self, tmp_path, monkeypatch):
        import whisper as whisper_mod

        calls = {}

        def _fake_load_model(size, device):
            calls["size"], calls["device"] = size, device
            return "fake-whisper-model"

        monkeypatch.setattr(whisper_mod, "load_model", _fake_load_model)

        cfg = _write_config(tmp_path, """\
            quality: "minimum"
            dataset:
              videos_dir: "demo_data"
            modalities: {}
            """)
        monkeypatch.setenv("CONFIG_PATH", str(cfg))

        result = models.get_whisper_model(device="cpu")
        assert result == "fake-whisper-model"
        assert calls == {"size": "tiny", "device": "cpu"}

    def test_get_sentiment_pipeline_uses_configured_model_name(self, tmp_path, monkeypatch):
        # transformers' lazy-loading `_LazyModule` (transformers 5.x)
        # self-caches resolved attributes on first access in a way that
        # makes monkeypatching `transformers.pipeline` directly unreliable
        # (a fresh `import transformers` elsewhere can resolve to a
        # different module identity than the one just patched). Intercept
        # at the import statement instead, which sidesteps that entirely.
        import builtins
        real_import = builtins.__import__

        calls = {}

        def _fake_pipeline(task, model, device):
            calls.update(task=task, model=model, device=device)
            return "fake-sentiment-pipeline"

        class _FakeTransformersModule:
            pipeline = staticmethod(_fake_pipeline)

        def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "transformers":
                return _FakeTransformersModule()
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        cfg = _write_config(tmp_path, """\
            dataset:
              videos_dir: "demo_data"
            modalities:
              sentiment:
                model: "some/custom-model"
            """)
        monkeypatch.setenv("CONFIG_PATH", str(cfg))

        result = models.get_sentiment_pipeline(device=-1)
        assert result == "fake-sentiment-pipeline"
        assert calls == {"task": "sentiment-analysis", "model": "some/custom-model", "device": -1}
