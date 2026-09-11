"""Wrapper around the vendored EmoNet model (./emonet.py, byte-for-byte from
https://github.com/face-analysis/emonet, CC BY-NC-ND 4.0 — non-commercial,
no derivatives; see data/models/emonet/LICENSE and NOTICE). This file is
original code that only calls the vendored model; emonet.py itself is left
untouched.
"""
from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

_IMAGE_SIZE = 256
_EXPRESSION_CLASSES = {
    0: "neutral", 1: "happy", 2: "sad", 3: "surprise",
    4: "fear", 5: "disgust", 6: "anger", 7: "contempt",
}


def _prep_256(bgr: np.ndarray) -> np.ndarray:
    """Matches the official EmoNet demo: resize(256), RGB, [0,1] range, CHW."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (_IMAGE_SIZE, _IMAGE_SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))


class EmoNetBackend:
    """8-class EmoNet: discrete expression plus continuous valence/arousal
    from a face crop. https://github.com/face-analysis/emonet
    """

    method_name = "emonet"

    def __init__(self, weights_path: str, device: str) -> None:
        import torch
        import torch.nn as nn

        from .emonet import EmoNet

        self.device = device
        self.torch = torch
        self.softmax = nn.Softmax(dim=1)

        model = EmoNet(n_expression=8).to(device)
        state = torch.load(weights_path, map_location=device)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model.eval()
        self.model = model

    def predict(self, face_bgr: np.ndarray) -> Dict[str, Any]:
        torch = self.torch
        chw = _prep_256(face_bgr)
        tensor = torch.from_numpy(chw).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            out = self.model(tensor)
            probs = self.softmax(out["expression"])
            idx = int(torch.argmax(probs, dim=1).item())
            valence = float(out["valence"].clamp(-1.0, 1.0).item())
            arousal = float(out["arousal"].clamp(-1.0, 1.0).item())

        return {
            "expression": _EXPRESSION_CLASSES[idx],
            "expression_scores": {
                _EXPRESSION_CLASSES[i]: float(probs[0, i]) for i in range(len(_EXPRESSION_CLASSES))
            },
            "valence": valence,
            "arousal": arousal,
        }
