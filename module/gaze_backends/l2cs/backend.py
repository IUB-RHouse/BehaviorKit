from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from ..base import GazeBackend

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_NUM_BINS = 90
_BINWIDTH = 4
_ANGLE_OFFSET = 180


def _prep_448(bgr: np.ndarray) -> np.ndarray:
    """Matches the official L2CS-Net Pipeline: crop -> resize(224) -> resize(448) -> normalize."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.resize(rgb, (448, 448), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    return np.transpose(rgb, (2, 0, 1))


class L2CSBackend(GazeBackend):
    """L2CS-Net (ResNet50, trained on Gaze360). https://github.com/Ahmednull/L2CS-Net

    Single-frame model with a binned yaw/pitch classification head
    (num_bins=90, binwidth=4deg, range [-180, 180)deg), decoded via the
    expected value of the softmax distribution over bins.
    """

    method_name = "l2cs"

    def __init__(self, weights_path: str, device: str) -> None:
        import torch
        import torch.nn as nn
        import torchvision

        from .model import L2CS

        self.device = device
        self.torch = torch
        self.softmax = nn.Softmax(dim=1)
        self.idx_tensor = torch.arange(_NUM_BINS, dtype=torch.float32, device=device)

        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], _NUM_BINS).to(device)
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        self.model = model

    def predict(self, face_bgr: np.ndarray) -> Tuple[float, float]:
        torch = self.torch
        chw = _prep_448(face_bgr)
        tensor = torch.from_numpy(chw).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            yaw_logits, pitch_logits = self.model(tensor)
            yaw_deg = torch.sum(self.softmax(yaw_logits) * self.idx_tensor, dim=1) * _BINWIDTH - _ANGLE_OFFSET
            pitch_deg = torch.sum(self.softmax(pitch_logits) * self.idx_tensor, dim=1) * _BINWIDTH - _ANGLE_OFFSET

        yaw = float(np.radians(yaw_deg.item()))
        pitch = float(np.radians(pitch_deg.item()))
        return yaw, pitch
