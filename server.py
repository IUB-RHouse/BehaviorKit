from __future__ import annotations

from module.check_gpus import select_and_mask_gpu
select_and_mask_gpu(min_free_gb=2, set_visible=True)

import asyncio
import json
import base64
import numpy as np
import cv2
import io
import soundfile as sf
import librosa
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
from dataclasses import dataclass
from typing import Optional

from module.inference import Sample, run_models_on_sample, get_runner

app = FastAPI(title="Robot Vision+Audio Inference API (WebSocket)", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TARGET_SR = 16000
ROLLING_SECONDS = 30
ROLLING_MAX_SAMPLES = TARGET_SR * ROLLING_SECONDS

total_requests = 0
total_frames_processed = 0
total_frames_skipped = 0
total_processing_time = 0

import torch
import os


@dataclass
class FrameData:
    """Container for frame data"""
    frame: np.ndarray
    audio_1s: np.ndarray
    audio_rolling: np.ndarray
    t: int
    message: dict


class AudioRollingBuffer:
    def __init__(self):
        self.buf = np.zeros((0,), dtype=np.float32)
        self.current_video_key = None
        self.last_t = None

    def _make_key(self, msg: dict) -> str:
        vid_name = msg.get("video_name")
        stream_idx = msg.get("stream_index")
        if vid_name is not None:
            return f"name:{vid_name}"
        if stream_idx is not None:
            return f"idx:{stream_idx}"
        return "unknown"

    def maybe_reset(self, msg: dict):
        key = self._make_key(msg)
        t = msg.get("t")

        should_reset = False
        if self.current_video_key is None:
            should_reset = True
        elif key != self.current_video_key:
            should_reset = True
        elif self.last_t is not None and t is not None and t < self.last_t:
            should_reset = True

        if should_reset:
            self.buf = np.zeros((0,), dtype=np.float32)
            self.current_video_key = key

        self.last_t = t

    def push(self, audio_16k_mono: np.ndarray):
        """Append new samples and trim to the last 30s."""
        if audio_16k_mono.ndim != 1:
            audio_16k_mono = np.asarray(audio_16k_mono).reshape(-1)
        if audio_16k_mono.dtype != np.float32:
            audio_16k_mono = audio_16k_mono.astype(np.float32)

        self.buf = np.concatenate([self.buf, audio_16k_mono], axis=0)
        if self.buf.size > ROLLING_MAX_SAMPLES:
            self.buf = self.buf[-ROLLING_MAX_SAMPLES:]

    def get(self) -> np.ndarray:
        """Return the full rolling buffer (<=30s)."""
        return self.buf.copy()


@app.on_event("startup")
async def startup():
    print("=" * 60)
    print("Robot Vision+Audio Inference API (WebSocket)")
    print("=" * 60)
    print("\nInitializing models...")

    global _RUNNER
    _RUNNER = None

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from module.inference import get_runner
    _RUNNER = get_runner(use_gpu=True)

    print(f"Models loaded on device: {_RUNNER.device}")
    print("\nEndpoints:")
    print("  - GET  /health       - Health check")
    print("  - WS   /ws/analyze   - WebSocket inference")
    print("=" * 60)


def _decode_frame(frame_b64: str, shape: list, compress: str = "raw-bgr") -> np.ndarray:
    frame_bytes = base64.b64decode(frame_b64)
    if compress == "jpeg":
        frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode JPEG frame")
        return frame
    return np.frombuffer(frame_bytes, dtype=np.uint8).reshape(shape)


def _decode_audio(audio_b64: str, audio_rate: int | None = None) -> np.ndarray:
    audio_bytes = base64.b64decode(audio_b64)
    raw = np.frombuffer(audio_bytes, dtype=np.float32)

    if audio_rate is None or audio_rate == TARGET_SR:
        mono = raw
    else:
        mono = librosa.resample(y=raw.astype(np.float32), orig_sr=audio_rate, target_sr=TARGET_SR)

    mono = np.asarray(mono, dtype=np.float32).reshape(-1)
    return mono


def _ensure_1s(audio_16k: np.ndarray) -> np.ndarray:
    """Return exactly 1 second at 16kHz by trimming or zero-padding."""
    if audio_16k.size > TARGET_SR:
        return audio_16k[:TARGET_SR]
    if audio_16k.size < TARGET_SR:
        return np.pad(audio_16k, (0, TARGET_SR - audio_16k.size))
    return audio_16k


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "Robot Vision+Audio Inference API (WebSocket)",
        "version": "1.0",
        "endpoints": ["/health", "/ws/analyze"],
        "requests_received": total_requests,
        "frames_processed": total_frames_processed,
        "frames_skipped": total_frames_skipped,
    }


@app.get("/health")
def health():
    """Health check endpoint"""
    try:
        runner = get_runner()
        return {
            "status": "healthy",
            "device": runner.device,
            "requests_received": total_requests,
            "frames_processed": total_frames_processed,
            "frames_skipped": total_frames_skipped,
            "avg_processing_time_ms": (
                total_processing_time / total_frames_processed if total_frames_processed > 0 else 0
            )
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    global total_requests, total_frames_processed, total_frames_skipped, total_processing_time

    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    print(f"\n🔌 Client connected: {client_id}")

    # Tell the client up front which modalities are active and which model
    # variant each is running, so it knows what to expect in every response
    # without having to guess from a single frame's (possibly-null) fields.
    await websocket.send_json({"type": "session_info", "modalities": _RUNNER.describe_modalities()})

    rolling = AudioRollingBuffer()
    latest_frame: Optional[FrameData] = None
    processing_lock = asyncio.Lock()
    should_stop = False
    
    async def receiver_task():
        """Continuously receives messages and updates latest frame + audio buffer"""
        nonlocal latest_frame, should_stop
        global total_requests, total_frames_skipped
        
        frame_count = 0
        
        try:
            while not should_stop:
                data = await websocket.receive_text()
                
                try:
                    message = json.loads(data)
                    frame_count += 1
                    total_requests += 1
                    t = message.get('t', frame_count)

                    frame_b64 = message.get('frame')
                    frame_shape = message.get('frame_shape')

                    if not frame_b64 or not frame_shape:
                        await websocket.send_json({"error": "Missing frame data"})
                        continue

                    frame = _decode_frame(frame_b64, frame_shape, compress=message.get('compress', 'raw-bgr'))

                    audio_b64 = message.get('audio')
                    if not audio_b64:
                        await websocket.send_json({"error": "Missing audio data"})
                        continue

                    audio_rate = message.get('audio_rate', None)
                    audio_16k = _decode_audio(audio_b64, audio_rate=audio_rate)

                    rolling.maybe_reset(message)
                    rolling.push(audio_16k)
                    audio_rolling = rolling.get()
                    audio_1s = _ensure_1s(audio_16k)

                    if processing_lock.locked():
                        # Processing is busy, we'll skip this frame but audio is cached
                        if latest_frame is not None:
                            total_frames_skipped += 1
                            print(f"Skipping frame t={latest_frame.t} (still processing)")
                    
                    latest_frame = FrameData(
                        frame=frame,
                        audio_1s=audio_1s,
                        audio_rolling=audio_rolling,
                        t=t,
                        message=message
                    )
                    
                except json.JSONDecodeError as e:
                    await websocket.send_json({"error": f"Invalid JSON: {str(e)}"})
                except Exception as e:
                    error_msg = f"Receiver error: {str(e)}"
                    print(f"{error_msg}")
                    import traceback
                    traceback.print_exc()
                    await websocket.send_json({"error": error_msg})
                    
        except WebSocketDisconnect:
            should_stop = True
            print(f"\n🔌 Client disconnected: {client_id}")
        except Exception as e:
            should_stop = True
            print(f"Receiver error: {e}")
            import traceback
            traceback.print_exc()

    async def processor_task():
        nonlocal latest_frame
        global total_frames_processed, total_processing_time
        
        while not should_stop:
            # Wait a bit for frames to arrive
            await asyncio.sleep(0.01)
            
            if latest_frame is None:
                continue
                
            # Acquire lock to process
            async with processing_lock:
                if latest_frame is None:
                    continue
                    
                # Get the frame to process and clear it
                frame_data = latest_frame
                latest_frame = None
                
                try:
                    print(f"\nProcessing frame | t={frame_data.t}")
                    print(f"Frame shape: {frame_data.frame.shape}, Audio 1s: {frame_data.audio_1s.shape}, Rolling: {frame_data.audio_rolling.shape}")
                    
                    start_time = time.perf_counter()

                    sample = Sample(
                        frame_bgr=frame_data.frame,
                        audio_1s_16k_mono=frame_data.audio_1s,
                        audio_rolling_16k_mono=frame_data.audio_rolling,
                        t_sec=frame_data.t
                    )

                    result = run_models_on_sample(sample)

                    processing_time = (time.perf_counter() - start_time) * 1000
                    total_processing_time += processing_time
                    total_frames_processed += 1

                    print(f"Processed in {processing_time:.1f}ms")

                    if result.whisper_text:
                        print(f"Speech: '{result.whisper_text[:50]}...'")
                    if getattr(result, "gaze", None):
                        g = result.gaze
                        print(f"Gaze: yaw={g['yaw']:.2f}, pitch={g['pitch']:.2f}")
                    if getattr(result, "emotion", None):
                        e = result.emotion
                        print(f"Emotion: {e['expression']} (valence={e['valence']:.2f}, arousal={e['arousal']:.2f})")
                    if getattr(result, "sentiment", None):
                        print(f"Sentiment: {result.sentiment['label']} ({result.sentiment['score']:.2f})")

                    response = {
                        "t_sec": result.t_sec,
                        "processing_time_ms": round(processing_time, 2),
                        "yolo_face": getattr(result, "yolo_face_xyxy", None),
                        "face_landmarks": getattr(result, "face_landmarks", None),
                        "pose_landmarks": getattr(result, "pose_landmarks", None),
                        "gaze": getattr(result, "gaze", None),
                        "emotion": getattr(result, "emotion", None),
                        "whisper_text": getattr(result, "whisper_text", None),
                        "whisper_words": getattr(result, "whisper_words", None),
                        "sentiment": getattr(result, "sentiment", None),
                        "debug": getattr(result, "debug", None),
                        "audio_buffer_seconds": round(frame_data.audio_rolling.size / TARGET_SR, 2),
                    }

                    await websocket.send_json(response)
                    
                except Exception as e:
                    error_msg = f"Processing error: {str(e)}"
                    print(f"{error_msg}")
                    import traceback
                    traceback.print_exc()
                    await websocket.send_json({"error": error_msg})

    receiver = asyncio.create_task(receiver_task())
    processor = asyncio.create_task(processor_task())
    
    try:
        await asyncio.gather(receiver, processor)
    except Exception as e:
        print(f"WebSocket error: {e}")
        should_stop = True
    finally:
        # Clean up
        should_stop = True
        if not receiver.done():
            receiver.cancel()
        if not processor.done():
            processor.cancel()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Starting WebSocket Inference Server")
    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8010,
        log_level="info"
    )