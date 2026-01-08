from __future__ import annotations
import os
import sys
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
    for k in ("models", "dataset"):
        if k not in data:
            raise KeyError(f"Missing top-level key '{k}' in {cfg_path}")
    return data

def _model_path(key: str) -> str:
    cfg = get_config()
    try:
        return cfg["models"][key]
    except KeyError as e:
        raise KeyError(f"Missing models.{key} in config") from e

def landmarks() -> Dict[str, list]:
    return get_config().get("landmarks", {})

def hypers() -> Dict[str, Any]:
    return get_config().get("hyperparameters", {})

def videos_dir() -> str:
    return get_config()["dataset"]["videos_dir"]

os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), ".hf_cache"))

@lru_cache(maxsize=1)
def get_face_landmarker(cpu: bool = True):
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except Exception as e:
        raise ImportError("mediapipe is required: pip install mediapipe") from e

    delegate = mp_python.BaseOptions.Delegate.CPU if cpu else mp_python.BaseOptions.Delegate.GPU
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=_model_path("face_landmark_model"),
            delegate=delegate
        ),
        num_faces=1,
        running_mode=mp_vision.RunningMode.IMAGE,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)

@lru_cache(maxsize=1)
def get_pose_landmarker(cpu: bool = True):
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except Exception as e:
        raise ImportError("mediapipe is required: pip install mediapipe") from e

    model_path = _model_path("pose_landmark_model")

    if get_config().get("quality", "medium") == "minimum":
        model_path = model_path.replace("heavy", "lite")  # use lite model
    if get_config().get("quality", "medium") == "low":
        model_path = model_path.replace("heavy", "lite")  # use lite model
    if get_config().get("quality", "medium") == "medium":
        model_path = model_path.replace("heavy", "full")  # use full model
    if get_config().get("quality", "medium") == "high":
        model_path = model_path.replace("heavy", "full")  # use full model
    if get_config().get("quality", "medium") == "maximum":
        model_path = model_path  # keep heavy model

    delegate = mp_python.BaseOptions.Delegate.CPU if cpu else mp_python.BaseOptions.Delegate.GPU
    base = mp_python.BaseOptions(
        model_asset_path=_model_path("pose_landmark_model"),
        delegate=delegate,
    )
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)

@lru_cache(maxsize=1)
def get_face_detector():
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise ImportError("ultralytics is required: pip install ultralytics") from e

    return YOLO(_model_path("face_detection_model"))

@lru_cache(maxsize=1)
def get_gaze_model() -> Tuple["torch.nn.Module", str]:
    try:
        import torch
    except Exception as e:
        raise ImportError("PyTorch is required: pip install torch") from e

    try:
        from module.gaze_model import GazeLSTM
    except Exception as e:
        hint = (
            "Could not import 'GazeLSTM'. "
            "Please ensure that you have followed the readme.md instructions on how to install Gaze360's model."
        )
        raise ImportError(hint) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GazeLSTM().to(device)

    ckpt_path = _model_path("gaze_model")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, device

@lru_cache(maxsize=1)
def get_whisper_model(name: str = "large-v3", device: Optional[str] = None):
    """
    Returns a Whisper model ready for transcribe().
    device: 'cuda' | 'cpu' | None (auto)
    """

    if get_config().get("quality", "medium") == "minimum":
        name = "tiny"  # override for lowest resource usage
    if get_config().get("quality", "medium") == "low":
        name = "tiny"  # override for low resource usage
    if get_config().get("quality", "medium") == "medium":
        name = "base"  # override for medium resource usage
    if get_config().get("quality", "medium") == "high":
        name = "large-v2"  # override for high resource usage
    if get_config().get("quality", "medium") == "maximum":
        name = "large-v3"  # override for maximum quality

    try:
        import whisper
    except Exception as e:
        raise ImportError("openai-whisper is required: pip install -U openai-whisper") from e

    if device is None:
        device = "cuda" if _has_cuda() else "cpu"
    return whisper.load_model(name, device=device)

def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

@lru_cache(maxsize=1)
def get_sentiment_pipeline(model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
                           device: int = -1):
    try:
        from transformers import pipeline
    except Exception as e:
        raise ImportError("transformers is required: pip install -U transformers") from e

    return pipeline("sentiment-analysis", model=model_name, device=device)

__all__ = [
    "get_config",
    "get_face_landmarker",
    "get_pose_landmarker",
    "get_face_detector",
    "get_gaze_model",
    "get_whisper_model",
    "get_sentiment_pipeline",
    "landmarks",
    "hypers",
    "videos_dir",
]