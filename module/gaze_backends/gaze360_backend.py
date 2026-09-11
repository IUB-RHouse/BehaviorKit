from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .base import GazeBackend

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _crop_norm_224(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    return np.transpose(rgb, (2, 0, 1))


class Gaze360Backend(GazeBackend):
    """Original Gaze360 GazeLSTM: ResNet18 + BiLSTM over a 7-frame window.

    We don't keep a real temporal buffer upstream, so (as in the original
    BehaviorKit code) a single frame is repeated 7 times to satisfy the
    model's expected input shape.
    """

    method_name = "gaze360"

    def __init__(self, weights_path: str, device: str) -> None:
        import torch

        from module.gaze_model import GazeLSTM

        self.device = device
        self.torch = torch

        model = GazeLSTM().to(device)
        ckpt = torch.load(weights_path, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model.eval()
        self.model = model

    def predict(self, face_bgr: np.ndarray) -> Tuple[float, float]:
        torch = self.torch
        chw = _crop_norm_224(face_bgr)
        tensor = torch.from_numpy(chw).unsqueeze(0).to(self.device)
        seq = tensor.unsqueeze(0).repeat(1, 7, 1, 1, 1)
        with torch.inference_mode():
            gaze_angles, _ = self.model(seq)
            yaw, pitch = float(gaze_angles[0, 0]), float(gaze_angles[0, 1])
        return yaw, pitch
