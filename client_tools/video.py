import platform
from typing import Optional, Tuple

import cv2
import numpy as np

def _pick_backend() -> Optional[int]:
    sys = platform.system() # Prefer OS-native backends to avoid needing system ffmpeg.
    if sys == "Windows":
        return cv2.CAP_MSMF
    if sys == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return None  # OpenCV decides (V4L2/GStreamer/etc.)

class VideoReader:
    def __init__(self, path: str, target_size: Optional[Tuple[int, int]] = None):
        backend = _pick_backend()
        self.cap = cv2.VideoCapture(path, backend) if backend is not None else cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.target_size = target_size
        self.index = 0

    def read(self) -> Optional[np.ndarray]:
        ok, frame_bgr = self.cap.read()
        if not ok:
            return None
        if self.target_size:
            tw, th = self.target_size
            if self.w != tw or self.h != th:
                frame_bgr = cv2.resize(frame_bgr, (tw, th), interpolation=cv2.INTER_AREA)
        self.index += 1
        return frame_bgr

    def release(self):
        self.cap.release()