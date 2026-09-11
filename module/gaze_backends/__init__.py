from __future__ import annotations

from typing import Any, Dict

from .base import GazeBackend

_METHODS = ("gaze360", "l2cs", "mobilegaze")


def build_gaze_backend(gaze_cfg: Dict[str, Any], device: str) -> GazeBackend:
    """Construct the configured gaze backend.

    gaze_cfg is the `gaze:` section of config.yaml, e.g.:
        gaze:
          method: "mobilegaze"
          gaze360_model: "data/models/gaze360/gaze360_model.pth"
          l2cs_model: "data/models/l2cs/L2CSNet_gaze360.pkl"
          mobilegaze_arch: "mobilenetv2"
          mobilegaze_model: "data/models/mobilegaze/mobilenetv2.pt"
    """
    method = gaze_cfg.get("method", "gaze360")
    if method not in _METHODS:
        raise ValueError(f"Unknown gaze method '{method}'. Available: {_METHODS}")

    if method == "gaze360":
        from .gaze360_backend import Gaze360Backend
        return Gaze360Backend(gaze_cfg["gaze360_model"], device)

    if method == "l2cs":
        from .l2cs import L2CSBackend
        return L2CSBackend(gaze_cfg["l2cs_model"], device)

    from .mobilegaze import MobileGazeBackend
    return MobileGazeBackend(
        gaze_cfg["mobilegaze_model"],
        device,
        arch=gaze_cfg.get("mobilegaze_arch", "mobilenetv2"),
    )


__all__ = ["GazeBackend", "build_gaze_backend"]
