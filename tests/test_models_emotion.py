"""Tests for the `emotion` modality's wiring through module/models.py:
config parsing, the enabled-by-default-false stance, and get_emonet_model()
actually building a backend from a config-supplied path.
"""
import textwrap
from pathlib import Path

import pytest

import module.models as models

WEIGHTS_PATH = Path("data/models/emonet/emonet_8.pth")


def _write_config(tmp_path: Path, emotion_block: str) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent(f"""\
        quality: "medium"
        dataset:
          videos_dir: "demo_data"
        modalities:
          emotion:
{textwrap.indent(textwrap.dedent(emotion_block), "            ")}
    """))
    return cfg


@pytest.fixture(autouse=True)
def _clear_caches(monkeypatch):
    # get_config / get_emonet_model are lru_cache'd; each test needs its own config.
    models.get_config.cache_clear()
    models.get_emonet_model.cache_clear()
    yield
    models.get_config.cache_clear()
    models.get_emonet_model.cache_clear()


def test_emotion_reads_enabled_and_model_path(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, """\
        enabled: true
        model_path: "some/custom/path.pth"
        """)
    monkeypatch.setenv("CONFIG_PATH", str(cfg))

    assert models.modality_enabled("emotion") is True
    assert models.modality_cfg("emotion")["model_path"] == "some/custom/path.pth"


def test_emotion_defaults_to_repo_config_disabled():
    """The repo's checked-in config.yaml should ship emotion disabled by
    default, since EmoNet's CC BY-NC-ND license makes it opt-in only."""
    models.get_config.cache_clear()
    assert models.modality_enabled("emotion") is False
    assert models.modality_cfg("emotion")["model_path"] == "data/models/emonet/emonet_8.pth"


@pytest.mark.skipif(not WEIGHTS_PATH.exists(), reason="emonet weights not present")
def test_get_emonet_model_builds_backend_from_config(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, f"""\
        enabled: true
        model_path: "{WEIGHTS_PATH.as_posix()}"
        """)
    monkeypatch.setenv("CONFIG_PATH", str(cfg))

    backend, device = models.get_emonet_model()

    assert backend.method_name == "emonet"
    assert device in ("cpu", "cuda")


def test_get_emonet_model_raises_clear_error_on_missing_path(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, """\
        enabled: true
        model_path: "data/models/emonet/does_not_exist.pth"
        """)
    monkeypatch.setenv("CONFIG_PATH", str(cfg))

    with pytest.raises(FileNotFoundError):
        models.get_emonet_model()
