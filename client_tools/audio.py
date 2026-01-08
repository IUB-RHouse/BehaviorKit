import contextlib
import wave
from typing import Tuple

import numpy as np

def wav_info(path: str):
    with contextlib.closing(wave.open(path, "rb")) as wf:
        return wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()

def read_wav_chunk(path: str, start_frame: int, num_frames: int) -> Tuple[np.ndarray, int, int]:
    with contextlib.closing(wave.open(path, "rb")) as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        total_frames = wf.getnframes()

        if start_frame >= total_frames:
            zeros = np.zeros(num_frames, dtype=np.float32)
            return zeros, sr, 0

        wf.setpos(start_frame)
        frames_to_read = min(num_frames, total_frames - start_frame)
        raw = wf.readframes(frames_to_read)

        if sampwidth == 1:
            data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            data = (data - 128.0) / 128.0
        elif sampwidth == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 3:
            a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            b = (a[:, 0].astype(np.int32) |
                 (a[:, 1].astype(np.int32) << 8) |
                 (a[:, 2].astype(np.int32) << 16))
            mask = b & 0x800000
            b = b - (mask << 1)
            data = (b.astype(np.float32) / 8388608.0)
        elif sampwidth == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

        # Convert to mono if needed
        if n_channels > 1:
            data = data.reshape(-1, n_channels).mean(axis=1)

        # Pad with zeros if needed
        if frames_to_read < num_frames:
            pad = np.zeros(num_frames - frames_to_read, dtype=np.float32)
            data = np.concatenate([data, pad], axis=0)

        return data.astype(np.float32), sr, frames_to_read