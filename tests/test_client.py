"""Tests for client.py: natural sort, media discovery/pairing, the
modality-gating helper, the session_info handshake, and the video/audio
streaming loop (_stream_one/run) -- driven against a fake websocket (an
object with async send/recv) and real tiny generated video/WAV files, so
no live server or network connection is needed.
"""
import asyncio
import json
import wave
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import client


class TestNaturalKey:
    def test_orders_numeric_suffixes_naturally(self):
        names = ["file10", "file2", "file1"]
        assert sorted(names, key=client._natural_key) == ["file1", "file2", "file10"]

    def test_case_insensitive(self):
        assert client._natural_key("File") == client._natural_key("file")


class TestListMedia:
    def test_lists_and_naturally_sorts_matching_files(self, tmp_path):
        for name in ["b2.mp4", "b10.mp4", "b1.mp4", "ignore.txt"]:
            (tmp_path / name).write_bytes(b"")
        files = client.list_media(tmp_path, client.VIDEO_EXTS)
        assert [f.name for f in files] == ["b1.mp4", "b2.mp4", "b10.mp4"]

    def test_none_dir_returns_empty_list(self):
        assert client.list_media(None, client.VIDEO_EXTS) == []

    def test_missing_dir_raises_system_exit(self, tmp_path):
        with pytest.raises(SystemExit):
            client.list_media(tmp_path / "nope", client.VIDEO_EXTS)

    def test_empty_dir_raises_system_exit(self, tmp_path):
        with pytest.raises(SystemExit):
            client.list_media(tmp_path, client.VIDEO_EXTS)


class TestPairMedia:
    def _paths(self, *names) -> list[Path]:
        return [Path(n) for n in names]

    def test_no_wavs_pairs_none(self):
        videos = self._paths("a.mp4", "b.mp4")
        assert client.pair_media(videos, []) == [(videos[0], None), (videos[1], None)]

    def test_basename_matches_by_stem(self):
        videos = self._paths("a.mp4", "b.mp4")
        wavs = self._paths("b.wav", "a.wav")
        pairs = client.pair_media(videos, wavs, strategy="basename")
        assert pairs == [(videos[0], wavs[1]), (videos[1], wavs[0])]

    def test_basename_allows_missing_audio_by_default(self):
        videos = self._paths("a.mp4", "b.mp4")
        wavs = self._paths("a.wav")
        pairs = client.pair_media(videos, wavs, strategy="basename")
        assert pairs == [(videos[0], wavs[0]), (videos[1], None)]

    def test_basename_strict_raises_on_missing_audio(self):
        videos = self._paths("a.mp4", "b.mp4")
        wavs = self._paths("a.wav")
        with pytest.raises(SystemExit):
            client.pair_media(videos, wavs, strategy="basename", allow_missing_audio=False)

    def test_index_pairs_by_position(self):
        videos = self._paths("a.mp4", "b.mp4")
        wavs = self._paths("x.wav", "y.wav")
        pairs = client.pair_media(videos, wavs, strategy="index")
        assert pairs == [(videos[0], wavs[0]), (videos[1], wavs[1])]

    def test_index_strict_raises_when_fewer_wavs(self):
        videos = self._paths("a.mp4", "b.mp4")
        wavs = self._paths("x.wav")
        with pytest.raises(SystemExit):
            client.pair_media(videos, wavs, strategy="index", allow_missing_audio=False)

    def test_unknown_strategy_raises(self):
        videos = self._paths("a.mp4")
        wavs = self._paths("a.wav")
        with pytest.raises(SystemExit):
            client.pair_media(videos, wavs, strategy="bogus")


class TestSequentialClientModalityGating:
    def test_defaults_true_when_handshake_never_happened(self):
        c = client.SequentialClient("ws://x")
        assert c._modality_enabled("gaze") is True

    def test_reflects_enabled_state_after_handshake(self):
        c = client.SequentialClient("ws://x")
        c.modalities = {"gaze": {"enabled": False}, "speech": {"enabled": True}}
        assert c._modality_enabled("gaze") is False
        assert c._modality_enabled("speech") is True

    def test_unknown_modality_defaults_true(self):
        c = client.SequentialClient("ws://x")
        c.modalities = {"gaze": {"enabled": False}}
        assert c._modality_enabled("nonexistent") is True


class _FakeWS:
    def __init__(self, messages):
        self._messages = list(messages)

    async def recv(self):
        return self._messages.pop(0)


class TestHandshake:
    def test_stores_modalities_from_session_info(self):
        c = client.SequentialClient("ws://x")
        payload = json.dumps({
            "type": "session_info",
            "modalities": {"gaze": {"enabled": True, "method": "mobilegaze"}, "speech": {"enabled": False}},
        })
        asyncio.run(c._handshake(_FakeWS([payload])))
        assert c.modalities["gaze"]["method"] == "mobilegaze"
        assert c._modality_enabled("speech") is False

    def test_warns_and_continues_on_unexpected_first_message(self, capsys):
        c = client.SequentialClient("ws://x")
        asyncio.run(c._handshake(_FakeWS([json.dumps({"type": "something_else"})])))
        assert c.modalities == {}
        assert "expected a session_info handshake" in capsys.readouterr().out


class TestBuildParser:
    def test_defaults(self):
        parser = client.build_parser()
        args = parser.parse_args([])
        assert args.server == "ws://localhost:8010/ws/analyze"
        assert args.video_dir == "demo_data"
        assert args.pair == "basename"
        assert args.width == 854
        assert args.height == 480
        assert args.jpeg is False

    def test_overrides(self):
        parser = client.build_parser()
        args = parser.parse_args(["--pair", "index", "--jpeg", "--max-ticks", "10"])
        assert args.pair == "index"
        assert args.jpeg is True
        assert args.max_ticks == 10


def _write_tiny_video(path, n_frames=3, size=(16, 12)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = size
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (w, h))
    for i in range(n_frames):
        writer.write(np.full((h, w, 3), i * 10, dtype=np.uint8))
    writer.release()


def _write_tiny_wav(path, n_samples=1600, sr=16000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        pcm = np.zeros(n_samples, dtype=np.int16)
        wf.writeframes(pcm.tobytes())


class _FakeStreamWS:
    """Fake websocket: async send/recv, serving canned JSON responses in
    order (one per recv() call, including the initial handshake)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.sent = []

    async def send(self, msg):
        self.sent.append(json.loads(msg))

    async def recv(self):
        if not self._responses:
            return json.dumps({"processing_time_ms": 1.0})
        return json.dumps(self._responses.pop(0))


class _FakeConnectCM:
    def __init__(self, ws):
        self._ws = ws

    async def __aenter__(self):
        return self._ws

    async def __aexit__(self, *exc_info):
        return False


class TestStreamOne:
    def _stream(self, tmp_path, monkeypatch, *, responses, wav=False, n_frames=3, **kwargs):
        video_path = tmp_path / "clip.mp4"
        _write_tiny_video(video_path, n_frames=n_frames)
        wav_path = None
        if wav:
            wav_path = tmp_path / "clip.wav"
            _write_tiny_wav(wav_path)

        ws = _FakeStreamWS(responses)
        c = client.SequentialClient("ws://fake")
        defaults = dict(
            ws=ws, video_path=video_path, wav_path=wav_path,
            max_ticks=None, fps_limit=1_000_000.0, target_size=None,
            audio_chunk_sec=0.1, compress_jpeg=False, jpeg_quality=85,
            global_tick_start=0, stream_idx=0,
        )
        defaults.update(kwargs)
        final_tick = asyncio.run(c._stream_one(**defaults))
        return c, ws, final_tick

    def test_streams_every_frame_until_eof(self, tmp_path, monkeypatch):
        c, ws, final_tick = self._stream(
            tmp_path, monkeypatch, responses=[{"processing_time_ms": 5.0}] * 3, n_frames=3,
        )
        assert final_tick == 3
        assert c.request_count == 3
        assert len(ws.sent) == 3

    def test_stops_early_at_max_ticks(self, tmp_path, monkeypatch):
        c, ws, final_tick = self._stream(
            tmp_path, monkeypatch, responses=[{"processing_time_ms": 5.0}] * 5,
            n_frames=5, max_ticks=2,
        )
        assert final_tick == 2
        assert c.request_count == 2

    def test_sends_raw_bgr_by_default(self, tmp_path, monkeypatch):
        c, ws, _ = self._stream(
            tmp_path, monkeypatch, responses=[{"processing_time_ms": 5.0}] * 3, n_frames=3,
        )
        assert ws.sent[0]["compress"] == "raw-bgr"
        assert ws.sent[0]["frame_shape"] == [12, 16, 3]

    def test_sends_jpeg_when_requested(self, tmp_path, monkeypatch):
        c, ws, _ = self._stream(
            tmp_path, monkeypatch, responses=[{"processing_time_ms": 5.0}] * 3,
            n_frames=3, compress_jpeg=True, jpeg_quality=80,
        )
        assert ws.sent[0]["compress"] == "jpeg"

    def test_includes_real_audio_chunks_when_wav_given(self, tmp_path, monkeypatch):
        c, ws, _ = self._stream(
            tmp_path, monkeypatch, responses=[{"processing_time_ms": 5.0}] * 3,
            n_frames=3, wav=True, audio_chunk_sec=0.05,
        )
        assert ws.sent[0]["audio_name"] == "clip.wav"
        assert ws.sent[0]["audio"] is not None

    def test_sends_silence_when_no_wav(self, tmp_path, monkeypatch):
        c, ws, _ = self._stream(
            tmp_path, monkeypatch, responses=[{"processing_time_ms": 5.0}] * 3, n_frames=3,
        )
        assert ws.sent[0]["audio_name"] is None

    def test_error_response_is_counted_but_not_processing_times(self, tmp_path, monkeypatch, capsys):
        c, ws, final_tick = self._stream(
            tmp_path, monkeypatch,
            responses=[{"error": "server exploded"}, {"processing_time_ms": 5.0}, {"processing_time_ms": 5.0}],
            n_frames=3,
        )
        assert c.request_count == 3
        assert c.processing_times == [5.0, 5.0]
        assert "Error: server exploded" in capsys.readouterr().out

    def test_tick_numbering_continues_from_global_tick_start(self, tmp_path, monkeypatch):
        c, ws, final_tick = self._stream(
            tmp_path, monkeypatch, responses=[{"processing_time_ms": 5.0}] * 2,
            n_frames=2, global_tick_start=10,
        )
        assert final_tick == 12
        assert ws.sent[0]["t"] == 10
        assert ws.sent[1]["t"] == 11

    def test_raises_when_jpeg_encode_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(client.cv2, "imencode", lambda *a, **k: (False, None))
        with pytest.raises(RuntimeError, match="cv2.imencode failed"):
            self._stream(
                tmp_path, monkeypatch, responses=[{"processing_time_ms": 5.0}],
                n_frames=1, compress_jpeg=True,
            )

    def test_prints_all_active_modalities_on_periodic_tick(self, tmp_path, monkeypatch, capsys):
        rich_result = {
            "processing_time_ms": 5.0,
            "whisper_text": "hello there",
            "gaze": {"yaw": 0.1, "pitch": 0.2},
            "emotion": {"expression": "happy", "valence": 0.5, "arousal": 0.1},
            "face_landmarks": [[{"x": 0, "y": 0, "z": 0}]],
            "pose_landmarks": [[{"x": 0, "y": 0, "z": 0}]],
            "sentiment": {"label": "positive", "score": 0.9},
        }
        # tick 0 is a "periodic" tick (0 % 5 == 0), so this prints on the first frame.
        self._stream(tmp_path, monkeypatch, responses=[rich_result], n_frames=1)

        out = capsys.readouterr().out
        assert "Speech:" in out
        assert "Gaze:" in out
        assert "Emotion:" in out
        assert "Faces:" in out
        assert "Poses:" in out
        assert "Sentiment:" in out

    def test_rate_limits_by_sleeping_between_frames(self, tmp_path, monkeypatch):
        sleeps = []

        async def _fake_sleep(seconds):
            sleeps.append(seconds)

        monkeypatch.setattr(client.asyncio, "sleep", _fake_sleep)
        # A tiny fps_limit makes frame_delay huge, guaranteeing sleep_for > 0
        # regardless of how long the (mocked, near-instant) send/recv took.
        self._stream(
            tmp_path, monkeypatch, responses=[{"processing_time_ms": 1.0}] * 2,
            n_frames=2, fps_limit=0.01,
        )
        assert len(sleeps) == 2
        assert all(s > 0 for s in sleeps)


class TestPrintStats:
    def test_runs_without_error_with_data(self, capsys):
        c = client.SequentialClient("ws://fake")
        c.request_count = 3
        c.total_time_ms = 30.0
        c.processing_times = [5.0, 10.0, 15.0]
        c.print_stats(1.5)
        out = capsys.readouterr().out
        assert "BENCHMARK RESULTS" in out
        assert "Messages sent:         3" in out

    def test_runs_without_error_with_no_data(self, capsys):
        c = client.SequentialClient("ws://fake")
        c.print_stats(0.0)
        assert "BENCHMARK RESULTS" in capsys.readouterr().out


class TestAmain:
    def test_lists_pairs_and_runs_client(self, tmp_path, monkeypatch):
        video_dir = tmp_path / "vids"
        video_dir.mkdir()
        (video_dir / "a.mp4").write_bytes(b"")

        captured = {}

        class _FakeClient:
            def __init__(self, server):
                captured["server"] = server

            async def run(self, **kwargs):
                captured["run_kwargs"] = kwargs

        monkeypatch.setattr(client, "SequentialClient", _FakeClient)

        args = client.build_parser().parse_args([
            "--video-dir", str(video_dir),
            "--audio-dir", "",
            "--server", "ws://fake:1234/ws",
        ])
        asyncio.run(client._amain(args))

        assert captured["server"] == "ws://fake:1234/ws"
        pairs = captured["run_kwargs"]["pairs"]
        assert len(pairs) == 1
        assert pairs[0][0].name == "a.mp4"
        assert pairs[0][1] is None  # no audio dir -> no wav pairing


class TestMain:
    def test_parses_argv_and_dispatches_to_amain(self, monkeypatch):
        captured = {}

        def _fake_run(coro):
            captured["coro"] = coro
            coro.close()  # avoid "coroutine was never awaited" warning

        monkeypatch.setattr(client.asyncio, "run", _fake_run)
        monkeypatch.setattr(
            "sys.argv", ["client.py", "--server", "ws://fake:1/ws", "--video-dir", "somewhere"]
        )

        client.main()

        assert "coro" in captured


class TestRun:
    def test_run_handshakes_streams_and_prints_stats(self, tmp_path, monkeypatch, capsys):
        video_path = tmp_path / "clip.mp4"
        _write_tiny_video(video_path, n_frames=2)

        handshake = {"type": "session_info", "modalities": {"gaze": {"enabled": True}}}
        responses = [handshake, {"processing_time_ms": 1.0}, {"processing_time_ms": 1.0}]
        ws = _FakeStreamWS(responses)
        monkeypatch.setattr(client.websockets, "connect", lambda url: _FakeConnectCM(ws))

        c = client.SequentialClient("ws://fake")
        asyncio.run(c.run(
            pairs=[(video_path, None)],
            max_ticks=None,
            fps_limit=1_000_000.0,
            out_width=0,
            out_height=0,
            audio_chunk_sec=0.1,
            compress_jpeg=False,
            jpeg_quality=85,
            server_url="ws://fake",
        ))

        assert c.modalities["gaze"]["enabled"] is True
        assert c.request_count == 2
        assert "BENCHMARK RESULTS" in capsys.readouterr().out

    def test_run_streams_multiple_pairs_with_continuous_ticks(self, tmp_path, monkeypatch):
        video_a = tmp_path / "a.mp4"
        video_b = tmp_path / "b.mp4"
        _write_tiny_video(video_a, n_frames=2)
        _write_tiny_video(video_b, n_frames=2)

        handshake = {"type": "session_info", "modalities": {}}
        responses = [handshake] + [{"processing_time_ms": 1.0}] * 4
        ws = _FakeStreamWS(responses)
        monkeypatch.setattr(client.websockets, "connect", lambda url: _FakeConnectCM(ws))

        c = client.SequentialClient("ws://fake")
        asyncio.run(c.run(
            pairs=[(video_a, None), (video_b, None)],
            max_ticks=None,
            fps_limit=1_000_000.0,
            out_width=0,
            out_height=0,
            audio_chunk_sec=0.1,
            compress_jpeg=False,
            jpeg_quality=85,
            server_url="ws://fake",
        ))

        assert c.request_count == 4
        assert ws.sent[0]["t"] == 0
        assert ws.sent[1]["t"] == 1
        assert ws.sent[2]["t"] == 2  # second video continues ticking, doesn't reset to 0
        assert ws.sent[3]["t"] == 3
