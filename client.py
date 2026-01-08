#!/usr/bin/env python3
import argparse
import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import websockets

from client_tools.audio import wav_info, read_wav_chunk
from client_tools.codecs import encode_audio_f32, encode_frame_ndarray_bgr
from client_tools.config import TARGET_SR
from client_tools.video import VideoReader

VIDEO_EXTS: Set[str] = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
WAV_EXTS: Set[str] = {".wav"}

def _natural_key(s: str):
    """Natural sort key (e.g., file2 < file10)."""
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def list_media(dirpath: Optional[Path], allowed_exts: Set[str]) -> List[Path]:
    if dirpath is None:
        return []
    if not dirpath.exists() or not dirpath.is_dir():
        raise SystemExit(f"Directory not found or not a directory: {dirpath}")
    files = [p for p in dirpath.iterdir() if p.is_file() and p.suffix.lower() in allowed_exts]
    files.sort(key=lambda p: _natural_key(p.name))
    if not files:
        exts = ", ".join(sorted(allowed_exts))
        raise SystemExit(f"No matching files found in {dirpath} (extensions: {exts})")
    return files

def _stem_map(paths: List[Path]) -> Dict[str, Path]:
    return {p.stem: p for p in paths}

def pair_media(
    videos: List[Path],
    wavs: List[Path],
    strategy: str = "basename",  # "basename" or "index"
    allow_missing_audio: bool = True,
) -> List[Tuple[Path, Optional[Path]]]:
    if not wavs:
        return [(v, None) for v in videos]

    pairs: List[Tuple[Path, Optional[Path]]] = []
    if strategy == "basename":
        m = _stem_map(wavs)
        for v in videos:
            wp = m.get(v.stem)
            if wp is None and not allow_missing_audio:
                raise SystemExit(f"No WAV found matching video basename: {v.name}")
            pairs.append((v, wp))
    elif strategy == "index":
        for i, v in enumerate(videos):
            wp = wavs[i] if i < len(wavs) else None
            if wp is None and not allow_missing_audio:
                raise SystemExit(f"Missing WAV for index {i} (video {v.name}); have only {len(wavs)} wavs.")
            pairs.append((v, wp))
    else:
        raise SystemExit(f"Unknown pairing strategy: {strategy} (use 'basename' or 'index')")
    return pairs

class SequentialClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.request_count = 0
        self.total_time_ms = 0.0
        self.processing_times: List[float] = []

    async def _stream_one(
        self,
        ws,
        video_path: Path,
        wav_path: Optional[Path],
        max_ticks: Optional[int],
        fps_limit: Optional[float],
        target_size: Optional[Tuple[int, int]],
        audio_chunk_sec: float,
        compress_jpeg: bool,
        jpeg_quality: int,
        global_tick_start: int,
        stream_idx: int,
    ) -> int:
        vid = VideoReader(str(video_path), target_size)
        tick = global_tick_start

        # Prepare audio if available
        if wav_path:
            _, _, sr, _ = wav_info(str(wav_path))
            samples_per_chunk = int(sr * audio_chunk_sec)
            wav_pos = 0
        else:
            sr = None
            samples_per_chunk = 0
            wav_pos = 0

        frame_delay = 1.0 / fps_limit if fps_limit else 0.0

        print(f"\n--- Streaming: {video_path.name}" +
              (f" + {wav_path.name}" if wav_path else " (no audio)") + " ---")

        try:
            while True:
                loop_start = time.time()

                # Read one frame
                frame = vid.read()
                if frame is None:
                    print("Reached EOF.")
                    break

                if wav_path:
                    chunk, sr_local, _ = read_wav_chunk(str(wav_path), wav_pos, samples_per_chunk)
                    audio = chunk.astype(np.float32)
                    audio_rate = int(sr_local)
                    wav_pos += samples_per_chunk
                else:
                    silence_len = int(TARGET_SR * audio_chunk_sec)
                    audio = np.zeros(silence_len, dtype=np.float32)
                    audio_rate = TARGET_SR

                audio_b64 = encode_audio_f32(audio)

                if compress_jpeg:
                    ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                    if not ok:
                        raise RuntimeError("cv2.imencode failed")
                    frame_b64 = base64.b64encode(enc.tobytes()).decode("ascii")
                    h, w = frame.shape[:2]
                    frame_shape = [h, w, 3]
                    compress_mode = "jpeg"
                else:
                    frame_b64, frame_shape = encode_frame_ndarray_bgr(frame)
                    compress_mode = "raw-bgr"

                # Singular payload expected by the server
                message = {
                    "frame": frame_b64,
                    "frame_shape": frame_shape,
                    "audio": audio_b64,
                    "audio_rate": audio_rate,
                    "t": tick,
                    "stream_index": stream_idx,
                    "video_name": video_path.name,
                    "audio_name": wav_path.name if wav_path else None,
                    "audio_chunk_sec": audio_chunk_sec,
                    "compress": compress_mode,
                }

                send_start = time.time()
                await ws.send(json.dumps(message))
                response = await ws.recv()
                rtt_ms = (time.time() - send_start) * 1000.0

                result = json.loads(response)
                self.request_count += 1
                self.total_time_ms += rtt_ms

                if "error" not in result:
                    proc = float(result.get("processing_time_ms", 0.0))
                    self.processing_times.append(proc)
                    if (tick - global_tick_start) % 5 == 0:
                        print(f"Tick {tick:4d} | Server: {proc:6.1f} ms | RTT: {rtt_ms:6.1f} ms")
                        if result.get("whisper_text"):
                            print(f"  Speech: '{result['whisper_text'][:60]}...'")
                        if result.get("gaze"):
                            g = result["gaze"]
                            print(f"  Gaze: yaw={g.get('yaw', 0):.2f}, pitch={g.get('pitch', 0):.2f}")
                        if result.get("face_landmarks"):
                            n_faces = len(result["face_landmarks"])
                            print(f"  Faces: {n_faces} detected", "number of landmarks per face:",
                                  [len(f) for f in result["face_landmarks"]])
                        if result.get("pose_landmarks"):
                            n_poses = len(result["pose_landmarks"])
                            print(f"  Poses: {n_poses} detected", "number of landmarks per pose:",
                                  [len(p) for p in result["pose_landmarks"]])
                        if result.get("sentiment"):
                            print(f"  Sentiment: {result['sentiment']}")
                else:
                    print(f"Tick {tick:4d} | Error: {result['error']}")

                tick += 1
                if max_ticks is not None and (tick - global_tick_start) >= max_ticks:
                    print("Reached --max-ticks for this stream.")
                    break

                # Rate limit
                if frame_delay > 0:
                    elapsed_loop = time.time() - loop_start
                    sleep_for = frame_delay - elapsed_loop
                    if sleep_for > 0:
                        await asyncio.sleep(sleep_for)
        finally:
            vid.release()

        return tick

    async def run(
        self,
        pairs: List[Tuple[Path, Optional[Path]]],
        max_ticks: Optional[int],
        fps_limit: Optional[float],
        out_width: int,
        out_height: int,
        audio_chunk_sec: float,
        compress_jpeg: bool,
        jpeg_quality: int,
        server_url: str,
    ):
        target_size = (out_width, out_height) if (out_width and out_height) else None

        print(f"Connecting to {server_url}")
        print("Plan:")
        for i, (v, w) in enumerate(pairs, 1):
            print(f"  {i:02d}. {v.name}" + (f"  |  {w.name}" if w else "  |  (no audio)"))
        print(f"Resize: {out_width}x{out_height}" if target_size else "Resize: native")
        if fps_limit:
            print(f"FPS limit: {fps_limit}")
        print(f"Audio chunk: {audio_chunk_sec:.2f}s\n")

        start_wall = time.time()
        async with websockets.connect(server_url) as ws:
            tick = 0
            for idx, (vp, wp) in enumerate(pairs):
                print(f"== Stream {idx+1}/{len(pairs)} ==")
                tick = await self._stream_one(
                    ws=ws,
                    video_path=vp,
                    wav_path=wp,
                    max_ticks=max_ticks,
                    fps_limit=fps_limit,
                    target_size=target_size,
                    audio_chunk_sec=audio_chunk_sec,
                    compress_jpeg=compress_jpeg,
                    jpeg_quality=jpeg_quality,
                    global_tick_start=tick,
                    stream_idx=idx,
                )

        self.print_stats(time.time() - start_wall)

    def print_stats(self, total_time_s: float):
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Messages sent:         {self.request_count}")
        print(f"Total runtime:         {total_time_s:.2f}s")
        if total_time_s > 0:
            print(f"Actual send rate:      {self.request_count / total_time_s:.2f} msg/s")
        if self.request_count:
            avg_total = self.total_time_ms / self.request_count
            print(f"\nAverage RTT:           {avg_total:.1f} ms")
        if self.processing_times:
            arr = np.array(self.processing_times, dtype=np.float32)
            print("\nServer Processing Time (ms):")
            print(f"  mean={arr.mean():.1f}  median={np.median(arr):.1f}  min={arr.min():.1f}  max={arr.max():.1f}")
            print(f"  p95={np.percentile(arr,95):.1f}  p99={np.percentile(arr,99):.1f}")
        print("=" * 60 + "\n")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WebSocket client: sequential videos from folders (+ optional WAVs)")
    p.add_argument("--server", default="ws://localhost:8010/ws/analyze", help="WebSocket server URL")
    p.add_argument("--video-dir", default="demo_data", help="Directory containing video files")
    p.add_argument("--audio-dir", default="demo_data/waves", help="Directory containing WAV files (or omit)")
    p.add_argument("--pair", choices=["basename", "index"], default="basename",
                   help="Pair videos to WAVs by basename (default) or by sorted index")
    p.add_argument("--strict-audio", action="store_true",
                   help="Error if a video has no matching WAV under the chosen pairing strategy")
    p.add_argument("--max-ticks", type=int, default=None,
                   help="Max messages per video (default: full video)")
    p.add_argument("--fps", type=float, default=None, help="Limit sending rate (messages per second)")
    p.add_argument("--width", type=int, default=854, help="Resize width (set both or leave both native)")
    p.add_argument("--height", type=int, default=480, help="Resize height")
    p.add_argument("--audio-chunk-sec", type=float, default=0.0667, help="Seconds per audio chunk")
    p.add_argument("--jpeg", action="store_true", help="Send frames as JPEG (smaller payload)")
    p.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality if --jpeg")
    return p

async def _amain(args: argparse.Namespace):
    video_dir = Path(args.video_dir) if args.video_dir else None
    audio_dir = Path(args.audio_dir) if args.audio_dir else None

    videos = list_media(video_dir, VIDEO_EXTS)
    wavs = list_media(audio_dir, WAV_EXTS) if audio_dir else []

    pairs = pair_media(
        videos=videos,
        wavs=wavs,
        strategy=args.pair,
        allow_missing_audio=not args.strict_audio,
    )

    client = SequentialClient(args.server)
    await client.run(
        pairs=pairs,
        max_ticks=args.max_ticks,
        fps_limit=args.fps,
        out_width=args.width,
        out_height=args.height,
        audio_chunk_sec=args.audio_chunk_sec,
        compress_jpeg=args.jpeg,
        jpeg_quality=args.jpeg_quality,
        server_url=args.server,
    )

def main():
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(_amain(args))

if __name__ == "__main__":
    main()