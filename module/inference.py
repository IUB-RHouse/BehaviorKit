from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np
import cv2

import module.models as reg

def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def _device_str() -> str:
    return "cuda" if _has_cuda() else "cpu"

def _mp_image_from_bgr(bgr: np.ndarray):
    import mediapipe as mp
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

def _best_yolo_box(result) -> Optional[List[float]]:
    if getattr(result, "boxes", None) is None or len(result.boxes) == 0:
        return None
    confs = result.boxes.conf.detach().cpu().numpy()
    i = int(confs.argmax())
    return result.boxes.xyxy[i].detach().cpu().numpy().tolist()

@dataclass
class Sample:
    frame_bgr: np.ndarray
    audio_1s_16k_mono: np.ndarray
    t_sec: int = 0
    audio_rolling_16k_mono: Optional[np.ndarray] = None

@dataclass
class Result:
    t_sec: int
    yolo_face_xyxy: Optional[List[float]]
    face_landmarks: List[List[Dict[str, float]]]
    pose_landmarks: List[List[Dict[str, float]]]
    gaze: Optional[Dict[str, float]]
    emotion: Optional[Dict[str, Any]]
    whisper_text: str
    whisper_words: Optional[List[Dict[str, float]]]
    sentiment: Optional[Dict[str, Any]]
    debug: Dict[str, Any]

MODALITIES = ("face_detection", "face_landmarks", "pose_landmarks", "gaze", "emotion", "speech", "sentiment")

class MultiModelRunner:
    def __init__(self, use_gpu: bool = True) -> None:
        self.device = _device_str() if use_gpu else "cpu"
        print(f"[MultiModelRunner] Using device: {self.device}")

        self.mediapipe_cpu = not use_gpu
        self.enabled: Dict[str, bool] = {m: reg.modality_enabled(m) for m in MODALITIES}

        if self.enabled["gaze"] and not self.enabled["face_detection"]:
            print("[MultiModelRunner] WARNING: gaze is enabled but face_detection is disabled; "
                  "gaze has no face crop to run on and will never produce output.")
        if self.enabled["emotion"] and not self.enabled["face_detection"]:
            print("[MultiModelRunner] WARNING: emotion is enabled but face_detection is disabled; "
                  "emotion has no face crop to run on and will never produce output.")
        if self.enabled["sentiment"] and not self.enabled["speech"]:
            print("[MultiModelRunner] WARNING: sentiment is enabled but speech is disabled; "
                  "sentiment has no text to analyze and will always be null.")

        speech_cfg = reg.modality_cfg("speech")
        self.audio_chunk_seconds = float(speech_cfg.get("chunk_seconds", 3.0))
        self.return_word_ts = bool(speech_cfg.get("word_timestamps", True))
        self.audio_sample_rate = 16000
        self.samples_per_chunk = int(self.audio_chunk_seconds * self.audio_sample_rate)
        self.audio_buffer: List[np.ndarray] = []
        self.frame_counter = 0
        self.last_whisper_text = ""
        self.last_whisper_words: Optional[List[Dict[str, float]]] = None
        self.last_sentiment: Optional[Dict[str, Any]] = None
        self.cfg = reg.get_config()
        self.landmarks_cfg = reg.landmarks()
        self.hypers_cfg = reg.hypers()

        print("[MultiModelRunner] Loading models...")
        self.yolo = reg.get_face_detector() if self.enabled["face_detection"] else None
        self.face_lm = reg.get_face_landmarker(cpu=self.mediapipe_cpu) if self.enabled["face_landmarks"] else None
        self.pose_lm = reg.get_pose_landmarker(cpu=self.mediapipe_cpu) if self.enabled["pose_landmarks"] else None

        if self.enabled["gaze"]:
            self.gaze_model, self.gaze_device = reg.get_gaze_model()
        else:
            self.gaze_model, self.gaze_device = None, None

        if self.enabled["emotion"]:
            self.emonet_model, self.emonet_device = reg.get_emonet_model()
        else:
            self.emonet_model, self.emonet_device = None, None

        self.whisper = reg.get_whisper_model(device=self.device) if self.enabled["speech"] else None

        if self.enabled["sentiment"]:
            # 0 = GPU:0, -1 = CPU
            sentiment_device = 0 if self.device == "cuda" else -1
            self.sentiment = reg.get_sentiment_pipeline(device=sentiment_device)
        else:
            self.sentiment = None

        print("[MultiModelRunner] All models loaded!")
        print(f"[MultiModelRunner] Active modalities: {self.describe_modalities()}")
        if self.enabled["speech"]:
            print(f"[MultiModelRunner] Whisper will run every {self.audio_chunk_seconds}s ({self.samples_per_chunk} samples)")

    def describe_modalities(self) -> Dict[str, Dict[str, Any]]:
        """What's actually running, for the session_info handshake sent to clients."""
        info: Dict[str, Dict[str, Any]] = {}

        info["face_detection"] = {"enabled": self.enabled["face_detection"]}
        info["face_landmarks"] = {"enabled": self.enabled["face_landmarks"]}
        info["pose_landmarks"] = {
            "enabled": self.enabled["pose_landmarks"],
            "variant": reg.resolve_pose_variant() if self.enabled["pose_landmarks"] else None,
        }
        info["gaze"] = {
            "enabled": self.enabled["gaze"],
            "method": self.gaze_model.method_name if self.gaze_model is not None else None,
        }
        info["emotion"] = {"enabled": self.enabled["emotion"]}
        info["speech"] = {
            "enabled": self.enabled["speech"],
            "model": reg.resolve_whisper_size() if self.enabled["speech"] else None,
            "word_timestamps": self.return_word_ts,
            "chunk_seconds": self.audio_chunk_seconds,
        }
        info["sentiment"] = {
            "enabled": self.enabled["sentiment"],
            "model": reg.modality_cfg("sentiment").get("model") if self.enabled["sentiment"] else None,
        }
        return info

    def _should_run_whisper(self) -> bool:
        total_samples = sum(len(chunk) for chunk in self.audio_buffer)
        return total_samples >= self.samples_per_chunk

    def _run_whisper_on_buffer(self) -> Tuple[str, Optional[List[Dict[str, float]]]]:
        if not self.audio_buffer:
            return "", None

        combined_audio = np.concatenate(self.audio_buffer)
        combined_audio = combined_audio[:self.samples_per_chunk]

        if len(combined_audio) < self.samples_per_chunk:
            combined_audio = np.pad(
                combined_audio,
                (0, self.samples_per_chunk - len(combined_audio)),
                mode='constant'
            )

        import whisper as _wh

        if self.return_word_ts:
            trans = self.whisper.transcribe(
                audio=combined_audio,
                word_timestamps=True,
                fp16=(self.device == "cuda"),
                language="en",
                verbose=False
            )
            text_out = (trans.get("text") or "").strip()
            words_out = []
            for seg in trans.get("segments", []) or []:
                for w in seg.get("words", []) or []:
                    words_out.append({
                        "word": (w.get("word") or "").strip(),
                        "start": float(w.get("start", 0.0)),
                        "end": float(w.get("end", 0.0)),
                    })
            return text_out, words_out
        else:
            mel = _wh.log_mel_spectrogram(_wh.pad_or_trim(combined_audio)).to(self.whisper.device)
            dec = _wh.DecodingOptions(language="en", without_timestamps=True, fp16=(self.device == "cuda"))
            dec_res = _wh.decode(self.whisper, mel, dec)
            text_out = (dec_res.text or "").strip()
            return text_out, None

    def _run_sentiment(self, text: str) -> Optional[Dict[str, Any]]:
        """Run sentiment analysis on text."""
        if not text:
            return None
        pred = self.sentiment(text, truncation=True)[0]
        return {"label": pred["label"], "score": float(pred["score"])}

    def process(self, sample: Sample) -> Result:
        frame = sample.frame_bgr
        audio = sample.audio_1s_16k_mono

        self.audio_buffer.append(audio)
        self.frame_counter += 1

        if self.enabled["speech"] and self._should_run_whisper():
            print(f"[Frame {self.frame_counter}] Running Whisper on {self.audio_chunk_seconds}s chunk...")
            self.last_whisper_text, self.last_whisper_words = self._run_whisper_on_buffer()
            if self.enabled["sentiment"]:
                self.last_sentiment = self._run_sentiment(self.last_whisper_text)

            self.audio_buffer = []

            print(f"[Frame {self.frame_counter}] Whisper result: '{self.last_whisper_text[:50]}...'")

        yolo_box = None
        if self.enabled["face_detection"]:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yolo_preds = self.yolo.predict(source=[rgb], device=self.device, verbose=False)
            yolo_box = _best_yolo_box(yolo_preds[0])

        faces_out: List[List[Dict[str, float]]] = []
        if self.enabled["face_landmarks"]:
            mp_img = _mp_image_from_bgr(frame)
            face_res = self.face_lm.detect(mp_img)
            if face_res.face_landmarks:
                for face in face_res.face_landmarks:
                    faces_out.append([{"x": float(p.x), "y": float(p.y), "z": float(p.z)} for p in face])

        poses_out: List[List[Dict[str, float]]] = []
        if self.enabled["pose_landmarks"]:
            mp_img = _mp_image_from_bgr(frame)
            pose_res = self.pose_lm.detect(mp_img)
            if pose_res.pose_landmarks:
                for pose in pose_res.pose_landmarks:
                    poses_out.append([{"x": float(p.x), "y": float(p.y), "z": float(p.z)} for p in pose])

        face_crop = None
        if yolo_box is not None:
            x1, y1, x2, y2 = map(int, yolo_box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    face_crop = crop

        gaze_out = None
        if self.enabled["gaze"] and face_crop is not None:
            yaw, pitch = self.gaze_model.predict(face_crop)
            x = -math.cos(pitch) * math.sin(yaw)
            y = -math.sin(pitch)
            z = -math.cos(pitch) * math.cos(yaw)
            gaze_out = {
                "yaw": yaw, "pitch": pitch, "x": x, "y": y, "z": z,
                "method": self.gaze_model.method_name,
            }

        emotion_out = None
        if self.enabled["emotion"] and face_crop is not None:
            emotion_out = self.emonet_model.predict(face_crop)

        return Result(
            t_sec=sample.t_sec,
            yolo_face_xyxy=yolo_box,
            face_landmarks=faces_out,
            pose_landmarks=poses_out,
            gaze=gaze_out,
            emotion=emotion_out,
            whisper_text=self.last_whisper_text,
            whisper_words=self.last_whisper_words,
            sentiment=self.last_sentiment,
            debug={
                "device": self.device,
                "frame_counter": self.frame_counter,
                "audio_buffer_size": sum(len(c) for c in self.audio_buffer),
                "videos_dir": reg.videos_dir(),
                "blink_threshold": self.hypers_cfg.get("blink_threshold"),
            },
        )

_global_runner: Optional[MultiModelRunner] = None

def get_runner(use_gpu: bool = True) -> MultiModelRunner:
    global _global_runner
    if _global_runner is None:
        print("Initializing global MultiModelRunner...")
        _global_runner = MultiModelRunner(use_gpu=use_gpu)
    return _global_runner

def run_models_on_sample(sample: Sample, **kwargs) -> Result:
    runner = get_runner(**kwargs)
    return runner.process(sample)
