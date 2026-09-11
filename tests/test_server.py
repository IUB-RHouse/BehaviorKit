"""Tests for server.py's pure logic: the rolling audio buffer, frame/audio
decoding, 1s padding/trimming, and the two plain HTTP endpoints. The
websocket handler itself (two concurrent asyncio tasks sharing state over a
live connection) is integration-shaped and deliberately not covered here.
"""
import base64

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")

import server


class TestAudioRollingBuffer:
    def test_starts_empty(self):
        buf = server.AudioRollingBuffer()
        assert buf.get().size == 0

    def test_push_concatenates(self):
        buf = server.AudioRollingBuffer()
        buf.push(np.ones(100, dtype=np.float32))
        buf.push(np.ones(50, dtype=np.float32) * 2)
        assert buf.get().size == 150

    def test_push_trims_to_rolling_max(self):
        buf = server.AudioRollingBuffer()
        buf.push(np.ones(server.ROLLING_MAX_SAMPLES + 1000, dtype=np.float32))
        assert buf.get().size == server.ROLLING_MAX_SAMPLES

    def test_push_coerces_dtype_and_shape(self):
        buf = server.AudioRollingBuffer()
        buf.push(np.ones((10, 1), dtype=np.int16))
        out = buf.get()
        assert out.dtype == np.float32
        assert out.shape == (10,)

    def test_get_returns_a_copy(self):
        buf = server.AudioRollingBuffer()
        buf.push(np.ones(10, dtype=np.float32))
        a = buf.get()
        a[:] = 0
        assert buf.get().sum() == 10  # underlying buffer untouched

    def test_make_key_prefers_video_name(self):
        buf = server.AudioRollingBuffer()
        assert buf._make_key({"video_name": "clip.mp4", "stream_index": 3}) == "name:clip.mp4"

    def test_make_key_falls_back_to_stream_index(self):
        buf = server.AudioRollingBuffer()
        assert buf._make_key({"stream_index": 3}) == "idx:3"

    def test_make_key_unknown_when_neither_present(self):
        buf = server.AudioRollingBuffer()
        assert buf._make_key({}) == "unknown"

    def test_maybe_reset_on_first_message(self):
        buf = server.AudioRollingBuffer()
        buf.push(np.ones(10, dtype=np.float32))
        buf.maybe_reset({"video_name": "a.mp4", "t": 0})
        assert buf.get().size == 0  # first message always resets
        assert buf.current_video_key == "name:a.mp4"

    def test_maybe_reset_on_key_change(self):
        buf = server.AudioRollingBuffer()
        buf.maybe_reset({"video_name": "a.mp4", "t": 0})
        buf.push(np.ones(10, dtype=np.float32))
        buf.maybe_reset({"video_name": "b.mp4", "t": 1})
        assert buf.get().size == 0
        assert buf.current_video_key == "name:b.mp4"

    def test_maybe_reset_on_time_going_backwards(self):
        buf = server.AudioRollingBuffer()
        buf.maybe_reset({"video_name": "a.mp4", "t": 5})
        buf.push(np.ones(10, dtype=np.float32))
        buf.maybe_reset({"video_name": "a.mp4", "t": 2})  # t went backwards -> new loop of same video
        assert buf.get().size == 0

    def test_maybe_reset_keeps_buffer_for_increasing_t_same_key(self):
        buf = server.AudioRollingBuffer()
        buf.maybe_reset({"video_name": "a.mp4", "t": 0})
        buf.push(np.ones(10, dtype=np.float32))
        buf.maybe_reset({"video_name": "a.mp4", "t": 1})
        assert buf.get().size == 10


class TestDecodeFrame:
    def test_round_trips_a_frame(self):
        frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        b64 = base64.b64encode(frame.tobytes()).decode("ascii")
        decoded = server._decode_frame(b64, [2, 3, 3])
        assert np.array_equal(decoded, frame)


class TestDecodeAudio:
    def test_no_resample_when_rate_matches_target(self):
        audio = np.linspace(-1, 1, 1000, dtype=np.float32)
        b64 = base64.b64encode(audio.tobytes()).decode("ascii")
        out = server._decode_audio(b64, audio_rate=server.TARGET_SR)
        assert np.allclose(out, audio)

    def test_no_resample_when_rate_is_none(self):
        audio = np.linspace(-1, 1, 1000, dtype=np.float32)
        b64 = base64.b64encode(audio.tobytes()).decode("ascii")
        out = server._decode_audio(b64, audio_rate=None)
        assert np.allclose(out, audio)

    def test_resamples_when_rate_differs(self):
        audio = np.linspace(-1, 1, 8000, dtype=np.float32)  # 0.25s @ 32kHz
        b64 = base64.b64encode(audio.tobytes()).decode("ascii")
        out = server._decode_audio(b64, audio_rate=32000)
        # 0.25s @ 16kHz target -> ~4000 samples
        assert abs(out.size - 4000) < 10
        assert out.dtype == np.float32


class TestEnsure1s:
    def test_pads_short_audio(self):
        out = server._ensure_1s(np.ones(100, dtype=np.float32))
        assert out.size == server.TARGET_SR
        assert out[:100].sum() == 100
        assert out[100:].sum() == 0

    def test_trims_long_audio(self):
        out = server._ensure_1s(np.ones(server.TARGET_SR + 500, dtype=np.float32))
        assert out.size == server.TARGET_SR

    def test_leaves_exact_length_untouched(self):
        audio = np.arange(server.TARGET_SR, dtype=np.float32)
        out = server._ensure_1s(audio)
        assert np.array_equal(out, audio)


class TestEndpoints:
    def test_root_reports_counters(self):
        response = server.root()
        assert response["service"].startswith("Robot Vision+Audio")
        assert "requests_received" in response

    def test_health_reports_healthy_when_runner_available(self, monkeypatch):
        class _FakeRunner:
            device = "cpu"

        monkeypatch.setattr(server, "get_runner", lambda: _FakeRunner())
        response = server.health()
        assert response["status"] == "healthy"
        assert response["device"] == "cpu"

    def test_health_reports_unhealthy_on_error(self, monkeypatch):
        def _boom():
            raise RuntimeError("models not loaded")

        monkeypatch.setattr(server, "get_runner", _boom)
        response = server.health()
        assert response["status"] == "unhealthy"
        assert "models not loaded" in response["error"]
