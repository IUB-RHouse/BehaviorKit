from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from ..base import GazeBackend

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Gaze360 dataset config (matches upstream config.py: data_config["gaze360"])
_NUM_BINS = 90
_BINWIDTH = 4
_ANGLE_OFFSET = 180

_ARCH_BUILDERS = {}


def _lazy_builders():
    if not _ARCH_BUILDERS:
        from .resnet import resnet18, resnet34, resnet50
        from .mobilenet import mobilenet_v2
        _ARCH_BUILDERS.update({
            "resnet18": resnet18,
            "resnet34": resnet34,
            "resnet50": resnet50,
            "mobilenetv2": mobilenet_v2,
        })
    return _ARCH_BUILDERS


def _resize_short_side(rgb: np.ndarray, target: int = 448) -> np.ndarray:
    """Matches torchvision transforms.Resize(448) on a non-square image."""
    h, w = rgb.shape[:2]
    if h <= w:
        new_h, new_w = target, max(1, round(w * target / h))
    else:
        new_w, new_h = target, max(1, round(h * target / w))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _prep(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = _resize_short_side(rgb, 448).astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    return np.transpose(rgb, (2, 0, 1))


class MobileGazeBackend(GazeBackend):
    """MobileGaze (yakhyo/gaze-estimation), built on L2CS-Net, trained on Gaze360.
    https://github.com/yakhyo/gaze-estimation

    Supports resnet18/34/50 and mobilenetv2 backbones with a binned yaw/pitch
    classification head (same bin scheme as L2CS-Net: 90 bins, 4deg width).
    """

    method_name = "mobilegaze"

    def __init__(self, weights_path: str, device: str, arch: str = "mobilenetv2") -> None:
        import torch
        import torch.nn as nn

        builders = _lazy_builders()
        if arch not in builders:
            raise ValueError(f"Unknown mobilegaze arch '{arch}'. Available: {sorted(builders)}")

        self.device = device
        self.arch = arch
        self.torch = torch
        self.softmax = nn.Softmax(dim=1)
        self.idx_tensor = torch.arange(_NUM_BINS, dtype=torch.float32, device=device)

        model = builders[arch](pretrained=False, num_classes=_NUM_BINS).to(device)
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        self.model = model

    def predict(self, face_bgr: np.ndarray) -> Tuple[float, float]:
        torch = self.torch
        chw = _prep(face_bgr)
        tensor = torch.from_numpy(chw).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            yaw_logits, pitch_logits = self.model(tensor)
            yaw_deg = torch.sum(self.softmax(yaw_logits) * self.idx_tensor, dim=1) * _BINWIDTH - _ANGLE_OFFSET
            pitch_deg = torch.sum(self.softmax(pitch_logits) * self.idx_tensor, dim=1) * _BINWIDTH - _ANGLE_OFFSET

        yaw = float(np.radians(yaw_deg.item()))
        pitch = float(np.radians(pitch_deg.item()))
        return yaw, pitch
