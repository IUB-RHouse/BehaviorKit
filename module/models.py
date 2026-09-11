from __future__ import annotations
import os
from functools import lru_cache
from typing import Any, Dict, Tuple, Optional

@lru_cache(maxsize=1)
def get_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = path or os.getenv("CONFIG_PATH", "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"Config file not found at '{cfg_path}'. "
            "Set CONFIG_PATH env var or put your YAML next to this file."
        )
    try:
        import yaml  # PyYAML
    except Exception as e:
        raise ImportError("PyYAML is required: pip install pyyaml") from e

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Basic structure guardrails
    for k in ("modalities", "dataset"):
        if k not in data:
            raise KeyError(f"Missing top-level key '{k}' in {cfg_path}")
    return data

def modality_cfg(name: str) -> Dict[str, Any]:
    """Settings for one modality (modalities.<name> in config.yaml). Missing
    modalities are treated as enabled with no overrides, so a config that
    doesn't mention a modality keeps the old always-on behavior."""
    return get_config().get("modalities", {}).get(name) or {}

def modality_enabled(name: str) -> bool:
    return bool(modality_cfg(name).get("enabled", True))

def quality() -> str:
    return get_config().get("quality", "medium")

def landmarks() -> Dict[str, list]:
    return get_config().get("landmarks", {})

def hypers() -> Dict[str, Any]:
    return get_config().get("hyperparameters", {})

def videos_dir() -> str:
    return get_config()["dataset"]["videos_dir"]

os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), ".hf_cache"))

def _make_face_landmarker(mp_python, mp_vision, model_path, delegate):
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=model_path,
            delegate=delegate
        ),
        num_faces=1,
        running_mode=mp_vision.RunningMode.IMAGE,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)

@lru_cache(maxsize=1)
def get_face_landmarker(cpu: bool = True):
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except Exception as e:
        raise ImportError("mediapipe is required: pip install mediapipe") from e

    model_path = modality_cfg("face_landmarks")["model_path"]

    if cpu:
        return _make_face_landmarker(mp_python, mp_vision, model_path, mp_python.BaseOptions.Delegate.CPU)

    # The mediapipe GPU delegate isn't built into the official Windows/macOS
    # wheels (only Linux); fall back to CPU rather than failing to start.
    try:
        return _make_face_landmarker(mp_python, mp_vision, model_path, mp_python.BaseOptions.Delegate.GPU)
    except Exception:
        return _make_face_landmarker(mp_python, mp_vision, model_path, mp_python.BaseOptions.Delegate.CPU)

def _make_pose_landmarker(mp_python, mp_vision, model_path, delegate):
    base = mp_python.BaseOptions(
        model_asset_path=model_path,
        delegate=delegate,
    )
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)

def resolve_pose_variant() -> str:
    """Explicit modalities.pose_landmarks.variant wins; otherwise derive
    lite/full/heavy from the global `quality` tier."""
    variant = modality_cfg("pose_landmarks").get("variant")
    if variant in ("lite", "full", "heavy"):
        return variant
    q = quality()
    if q in ("minimum", "low"):
        return "lite"
    if q in ("medium", "high"):
        return "full"
    return "heavy"  # "maximum"

@lru_cache(maxsize=1)
def get_pose_landmarker(cpu: bool = True):
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except Exception as e:
        raise ImportError("mediapipe is required: pip install mediapipe") from e

    model_path = modality_cfg("pose_landmarks")["model_path"]
    variant = resolve_pose_variant()
    for tag in ("lite", "full", "heavy"):
        if tag in model_path:
            model_path = model_path.replace(tag, variant)
            break

    if cpu:
        return _make_pose_landmarker(mp_python, mp_vision, model_path, mp_python.BaseOptions.Delegate.CPU)

    # The mediapipe GPU delegate isn't built into the official Windows/macOS
    # wheels (only Linux); fall back to CPU rather than failing to start.
    try:
        return _make_pose_landmarker(mp_python, mp_vision, model_path, mp_python.BaseOptions.Delegate.GPU)
    except Exception:
        return _make_pose_landmarker(mp_python, mp_vision, model_path, mp_python.BaseOptions.Delegate.CPU)

@lru_cache(maxsize=1)
def get_face_detector():
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise ImportError("ultralytics is required: pip install ultralytics") from e

    return YOLO(modality_cfg("face_detection")["model_path"])

@lru_cache(maxsize=1)
def get_gaze_model() -> Tuple[Any, str]:
    """Builds the configured gaze backend (modalities.gaze.method:
    gaze360 | l2cs | mobilegaze).

    Returns (backend, device) where backend implements
    module.gaze_backends.base.GazeBackend.predict(face_bgr) -> (yaw, pitch) in radians.
    """
    try:
        import torch
    except Exception as e:
        raise ImportError("PyTorch is required: pip install torch") from e

    from module.gaze_backends import build_gaze_backend

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = build_gaze_backend(modality_cfg("gaze"), device)
    return backend, device

@lru_cache(maxsize=1)
def get_emonet_model() -> Tuple[Any, str]:
    """Builds the EmoNet emotion backend (valence/arousal + 8-class
    expression). CC BY-NC-ND 4.0, non-commercial use only — see
    data/models/emonet/LICENSE and NOTICE.

    Returns (backend, device) where backend.predict(face_bgr) -> dict with
    'expression', 'expression_scores', 'valence', 'arousal'.
    """
    try:
        import torch
    except Exception as e:
        raise ImportError("PyTorch is required: pip install torch") from e

    from module.emonet_backend import EmoNetBackend

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_path = modality_cfg("emotion")["model_path"]
    backend = EmoNetBackend(weights_path, device)
    return backend, device

def resolve_whisper_size() -> str:
    """Explicit modalities.speech.model wins; otherwise derive a whisper
    size from the global `quality` tier."""
    model = modality_cfg("speech").get("model")
    if model:
        return model
    q = quality()
    if q in ("minimum", "low"):
        return "tiny"
    if q == "medium":
        return "base"
    if q == "high":
        return "large-v2"
    return "large-v3"  # "maximum"

@lru_cache(maxsize=1)
def get_whisper_model(device: Optional[str] = None):
    """Returns a Whisper model ready for transcribe(). device: 'cuda' | 'cpu' | None (auto)."""
    try:
        import whisper
    except Exception as e:
        raise ImportError("openai-whisper is required: pip install -U openai-whisper") from e

    if device is None:
        device = "cuda" if _has_cuda() else "cpu"
    return whisper.load_model(resolve_whisper_size(), device=device)

def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

@lru_cache(maxsize=1)
def get_sentiment_pipeline(device: int = -1):
    try:
        from transformers import pipeline
    except Exception as e:
        raise ImportError("transformers is required: pip install -U transformers") from e

    model_name = modality_cfg("sentiment").get("model", "cardiffnlp/twitter-xlm-roberta-base-sentiment")
    return pipeline("sentiment-analysis", model=model_name, device=device)

__all__ = [
    "get_config",
    "modality_cfg",
    "modality_enabled",
    "quality",
    "resolve_pose_variant",
    "resolve_whisper_size",
    "get_face_landmarker",
    "get_pose_landmarker",
    "get_face_detector",
    "get_gaze_model",
    "get_emonet_model",
    "get_whisper_model",
    "get_sentiment_pipeline",
    "landmarks",
    "hypers",
    "videos_dir",
]
