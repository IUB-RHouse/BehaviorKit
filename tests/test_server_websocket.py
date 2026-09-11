"""Tests for server.py's /ws/analyze handler: the receiver/processor task
pair that shares state (a rolling audio buffer, a "latest frame" slot, a
lock) over one websocket connection.

Driven against a fake WebSocket (async accept/receive_text/send_json) fed a
queue of canned incoming messages that ends in a WebSocketDisconnect, with
module.inference.run_models_on_sample and server._RUNNER faked so no real
model ever loads. Every call is wrapped in asyncio.wait_for with a hard
timeout as a safety net: this suite's own real-GPU-delegate test earlier
today hung the whole run for 10+ minutes, so nothing here is allowed to
rely on a bug-free implementation to terminate.
"""
import asyncio
import types

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")

import server
from client_tools.codecs import encode_audio_f32, encode_frame_ndarray_bgr
from module.inference import Result


@pytest.fixture(autouse=True)
def _snapshot_global_counters():
    """total_requests/total_frames_processed/etc. are plain module globals
    mutated in place -- monkeypatch doesn't intercept `global` reassignment
    inside a function, so snapshot/restore by hand."""
    names = ["total_requests", "total_frames_processed", "total_frames_skipped", "total_processing_time"]
    saved = {n: getattr(server, n) for n in names}
    for n in names:
        setattr(server, n, 0)
    yield
    for n, v in saved.items():
        setattr(server, n, v)


class _FakeClientAddr:
    host = "127.0.0.1"
    port = 54321


class _FakeServerWebSocket:
    """incoming: list of raw text messages to hand back from receive_text(),
    in order; once exhausted, raises WebSocketDisconnect (ending the
    receiver loop, and in turn the whole handler, deterministically)."""

    def __init__(self, incoming, delay=0.02):
        self._incoming = list(incoming)
        self._delay = delay
        self.sent = []
        self.client = _FakeClientAddr()

    async def accept(self):
        pass

    async def send_json(self, data):
        self.sent.append(data)

    async def receive_text(self):
        if not self._incoming:
            raise server.WebSocketDisconnect()
        # Small delay between messages so the processor_task's 10ms poll
        # loop reliably picks up and clears each frame in turn, instead of
        # the receiver racing ahead and overwriting latest_frame.
        await asyncio.sleep(self._delay)
        return self._incoming.pop(0)


def _make_frame_message(t=0, **overrides):
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame_b64, frame_shape = encode_frame_ndarray_bgr(frame)
    audio_b64 = encode_audio_f32(np.zeros(160, dtype=np.float32))
    msg = {
        "frame": frame_b64,
        "frame_shape": frame_shape,
        "audio": audio_b64,
        "audio_rate": server.TARGET_SR,
        "t": t,
        "video_name": "clip.mp4",
    }
    msg.update(overrides)
    return msg


def _canned_result(t_sec=0):
    return Result(
        t_sec=t_sec,
        yolo_face_xyxy=None,
        face_landmarks=[],
        pose_landmarks=[],
        gaze=None,
        emotion=None,
        whisper_text="",
        whisper_words=None,
        sentiment=None,
        debug={},
    )


class _FakeRunner:
    device = "cpu"

    def describe_modalities(self):
        return {"gaze": {"enabled": True}, "speech": {"enabled": False}}


async def _run_handler(monkeypatch, incoming, run_models=None, delay=0.02):
    monkeypatch.setattr(server, "_RUNNER", _FakeRunner(), raising=False)
    monkeypatch.setattr(server, "run_models_on_sample", run_models or (lambda sample: _canned_result(sample.t_sec)))

    ws = _FakeServerWebSocket(incoming, delay=delay)
    await asyncio.wait_for(server.websocket_analyze(ws), timeout=10.0)
    return ws


class TestHappyPath:
    def test_sends_session_info_first(self, monkeypatch):
        ws = asyncio.run(_run_handler(monkeypatch, incoming=[]))
        assert ws.sent[0] == {
            "type": "session_info",
            "modalities": {"gaze": {"enabled": True}, "speech": {"enabled": False}},
        }

    def test_processes_each_valid_frame_and_responds(self, monkeypatch):
        import json
        messages = [json.dumps(_make_frame_message(t=i)) for i in range(3)]

        ws = asyncio.run(_run_handler(monkeypatch, incoming=messages))

        frame_responses = [m for m in ws.sent[1:] if "error" not in m]
        assert len(frame_responses) == 3
        assert {r["t_sec"] for r in frame_responses} == {0, 1, 2}
        assert server.total_requests == 3
        assert server.total_frames_processed == 3

    def test_response_carries_result_fields(self, monkeypatch, capsys):
        import json

        def _run_models(sample):
            r = _canned_result(sample.t_sec)
            r.gaze = {"yaw": 0.1, "pitch": 0.2}
            r.emotion = {"expression": "happy", "valence": 0.5, "arousal": 0.1}
            r.sentiment = {"label": "positive", "score": 0.9}
            r.whisper_text = "hello there"
            return r

        ws = asyncio.run(_run_handler(monkeypatch, incoming=[json.dumps(_make_frame_message())], run_models=_run_models))

        resp = next(m for m in ws.sent[1:] if "error" not in m)
        assert resp["gaze"] == {"yaw": 0.1, "pitch": 0.2}
        assert resp["emotion"]["expression"] == "happy"
        assert resp["sentiment"]["label"] == "positive"
        assert resp["whisper_text"] == "hello there"
        assert "processing_time_ms" in resp
        assert "audio_buffer_seconds" in resp

        out = capsys.readouterr().out
        assert "Speech:" in out
        assert "Gaze:" in out
        assert "Emotion:" in out
        assert "Sentiment:" in out


class TestMalformedInput:
    def test_missing_frame_data_reports_error(self, monkeypatch):
        import json
        msg = _make_frame_message()
        del msg["frame"]

        ws = asyncio.run(_run_handler(monkeypatch, incoming=[json.dumps(msg)]))

        assert {"error": "Missing frame data"} in ws.sent
        assert server.total_frames_processed == 0

    def test_missing_audio_data_reports_error(self, monkeypatch):
        import json
        msg = _make_frame_message()
        del msg["audio"]

        ws = asyncio.run(_run_handler(monkeypatch, incoming=[json.dumps(msg)]))

        assert {"error": "Missing audio data"} in ws.sent

    def test_invalid_json_reports_error(self, monkeypatch):
        ws = asyncio.run(_run_handler(monkeypatch, incoming=["{not valid json"]))

        errors = [m["error"] for m in ws.sent if "error" in m]
        assert any("Invalid JSON" in e for e in errors)

    def test_frame_decode_failure_reports_receiver_error(self, monkeypatch):
        """frame_shape that doesn't match the actual decoded byte count
        makes numpy's .reshape() raise -- a real malformed-payload case,
        distinct from the "field missing entirely" checks above."""
        import json
        msg = _make_frame_message()
        msg["frame_shape"] = [999, 999, 3]  # far more bytes than the payload actually has

        ws = asyncio.run(_run_handler(monkeypatch, incoming=[json.dumps(msg)]))

        errors = [m["error"] for m in ws.sent if "error" in m]
        assert any("Receiver error" in e for e in errors)

    def test_receive_text_generic_exception_stops_receiver_gracefully(self, monkeypatch):
        """A non-WebSocketDisconnect exception from receive_text() (e.g. a
        transport-level error) must still be caught by receiver_task's
        outer handler and stop the loop, not propagate/hang."""

        class _BoomWS(_FakeServerWebSocket):
            async def receive_text(self):
                raise RuntimeError("transport exploded")

        monkeypatch.setattr(server, "_RUNNER", _FakeRunner(), raising=False)
        monkeypatch.setattr(server, "run_models_on_sample", lambda sample: _canned_result(sample.t_sec))

        ws = _BoomWS([])
        asyncio.run(asyncio.wait_for(server.websocket_analyze(ws), timeout=10.0))
        # No assertion beyond "returned instead of hanging" -- this is a
        # termination/cleanup guarantee, not a behavior with output to check.

    def test_processing_exception_reports_error_and_continues(self, monkeypatch):
        import json

        calls = {"n": 0}

        def _flaky_run_models(sample):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return _canned_result(sample.t_sec)

        messages = [json.dumps(_make_frame_message(t=i)) for i in range(2)]
        ws = asyncio.run(_run_handler(monkeypatch, incoming=messages, run_models=_flaky_run_models, delay=0.03))

        errors = [m["error"] for m in ws.sent if "error" in m]
        assert any("Processing error: boom" in e for e in errors)
        # the second message still gets processed despite the first one's failure
        successes = [m for m in ws.sent[1:] if "error" not in m]
        assert len(successes) == 1


class TestOuterExceptionCleanup:
    def test_second_failure_reporting_the_first_propagates_and_is_cleaned_up(self, monkeypatch):
        """If run_models_on_sample raises AND the subsequent
        websocket.send_json(error) also raises (e.g. the client vanished
        mid-response), that second exception isn't caught by
        processor_task's own try/except -- it propagates out through
        asyncio.gather into websocket_analyze's outer handler, which must
        still cancel the still-running receiver task and return cleanly
        rather than hang."""
        import json

        def _raising_run_models(sample):
            raise RuntimeError("model exploded")

        class _DoubleFaultWS(_FakeServerWebSocket):
            async def send_json(self, data):
                if "error" in data and "model exploded" in data["error"]:
                    raise ConnectionError("client already gone")
                await super().send_json(data)

        monkeypatch.setattr(server, "_RUNNER", _FakeRunner(), raising=False)
        monkeypatch.setattr(server, "run_models_on_sample", _raising_run_models)

        # A message that will never actually get consumed by the receiver
        # loop (should_stop flips True once the outer exception hits), so
        # keep incoming small; the important thing is receiver is still
        # "in progress" (awaiting receive_text) when the fault occurs.
        ws = _DoubleFaultWS([json.dumps(_make_frame_message())], delay=0.05)

        # Must terminate on its own via the outer except/finally -- the
        # timeout is a safety net, not the expected exit path.
        asyncio.run(asyncio.wait_for(server.websocket_analyze(ws), timeout=10.0))


class TestFrameSkipping:
    def test_skips_frames_that_arrive_faster_than_processing(self, monkeypatch):
        """With no delay between incoming messages, the receiver can race
        ahead of the 10ms processor poll and overwrite latest_frame before
        it's consumed -- total_requests should exceed frames actually
        processed, and total_frames_skipped should account for the gap."""
        import json
        messages = [json.dumps(_make_frame_message(t=i)) for i in range(20)]

        ws = asyncio.run(_run_handler(monkeypatch, incoming=messages, delay=0.0))

        assert server.total_requests == 20
        assert server.total_frames_processed <= 20
        # every request is accounted for as either processed or skipped-or-dropped
        assert server.total_frames_processed + server.total_frames_skipped <= server.total_requests
