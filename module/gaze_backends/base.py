from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class GazeBackend(ABC):
    """Common interface every gaze-estimation backend implements.

    A backend takes a BGR face crop (as produced by the YOLO face detector)
    and returns (yaw, pitch) in radians, using the same sign/axis convention
    as the original Gaze360 GazeLSTM model (yaw: left/right, pitch: up/down;
    x = -cos(pitch)*sin(yaw), y = -sin(pitch), z = -cos(pitch)*cos(yaw)).
    """

    method_name: str
    device: str

    @abstractmethod
    def predict(self, face_bgr: np.ndarray) -> Tuple[float, float]:
        """Return (yaw, pitch) in radians for a single BGR face crop."""
        raise NotImplementedError
