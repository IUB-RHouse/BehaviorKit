import base64
import numpy as np

def encode_frame_ndarray_bgr(frame: np.ndarray):
    b64 = base64.b64encode(frame.tobytes()).decode("ascii")
    return b64, list(frame.shape)  # [H, W, 3]

def encode_audio_f32(audio: np.ndarray) -> str:
    return base64.b64encode(audio.astype(np.float32).tobytes()).decode("ascii")