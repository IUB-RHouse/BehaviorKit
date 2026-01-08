from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
WAV_EXTS = {".wav"}
TARGET_SR = 16000
DEFAULT_SERVER = "ws://localhost:8010/ws/analyze"
DEFAULT_VIDEO_DIR = Path("demo_data")
DEFAULT_AUDIO_DIR = Path("demo_data/waves")
DEFAULT_PAIR_STRATEGY = "basename"  # "basename" | "index"
DEFAULT_AUDIO_CHUNK_SEC = 0.0667
DEFAULT_JPEG_QUALITY = 85
DEFAULT_OUT_WIDTH = 854
DEFAULT_OUT_HEIGHT = 480