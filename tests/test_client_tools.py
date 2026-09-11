"""Tests for client_tools/: WAV reading, base64 frame/audio encoding, and
VideoReader. Uses real (tiny, generated) WAV/video files rather than mocks,
since these modules are thin wrappers around the `wave` and `cv2` file
formats and the point is to catch real format-handling bugs.
"""
import base64
import struct
import wave

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from client_tools.audio import read_wav_chunk, wav_info
from client_tools.codecs import encode_audio_f32, encode_frame_ndarray_bgr
from client_tools.video import VideoReader


def _write_wav(path, samples: np.ndarray, sr: int = 16000, sampwidth: int = 2, n_channels: int = 1):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


class TestWavInfo:
    def test_reports_basic_properties(self, tmp_path):
        path = tmp_path / "a.wav"
        pcm = (np.sin(np.linspace(0, 2 * np.pi, 1600)) * 20000).astype(np.int16)
        _write_wav(path, pcm, sr=16000)

        n_channels, sampwidth, sr, n_frames = wav_info(str(path))
        assert n_channels == 1
        assert sampwidth == 2
        assert sr == 16000
        assert n_frames == 1600


class TestReadWavChunk:
    def test_reads_int16_and_normalizes_to_minus1_1(self, tmp_path):
        path = tmp_path / "a.wav"
        pcm = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        _write_wav(path, pcm, sr=8000)

        data, sr, n_read = read_wav_chunk(str(path), 0, 5)
        assert sr == 8000
        assert n_read == 5
        assert data.dtype == np.float32
        assert data[0] == pytest.approx(0.0)
        assert data[1] == pytest.approx(0.5, abs=1e-3)
        assert data[3] == pytest.approx(32767 / 32768, abs=1e-3)

    def test_pads_when_fewer_frames_than_requested(self, tmp_path):
        path = tmp_path / "a.wav"
        pcm = np.array([100, 200, 300], dtype=np.int16)
        _write_wav(path, pcm, sr=8000)

        data, sr, n_read = read_wav_chunk(str(path), 0, 10)
        assert n_read == 3
        assert data.size == 10
        assert np.all(data[3:] == 0)

    def test_all_zeros_when_start_past_end(self, tmp_path):
        path = tmp_path / "a.wav"
        pcm = np.array([1, 2, 3], dtype=np.int16)
        _write_wav(path, pcm, sr=8000)

        data, sr, n_read = read_wav_chunk(str(path), 100, 5)
        assert n_read == 0
        assert data.size == 5
        assert np.all(data == 0)

    def test_downmixes_stereo_to_mono(self, tmp_path):
        path = tmp_path / "a.wav"
        # interleaved L/R: L=full scale, R=silence -> mono average = half scale
        left = np.full(4, 32767, dtype=np.int16)
        right = np.zeros(4, dtype=np.int16)
        interleaved = np.empty(8, dtype=np.int16)
        interleaved[0::2] = left
        interleaved[1::2] = right
        _write_wav(path, interleaved, sr=8000, n_channels=2)

        data, sr, n_read = read_wav_chunk(str(path), 0, 4)
        assert n_read == 4
        assert data[0] == pytest.approx(0.5, abs=1e-3)

    def test_reads_24bit_with_sign_extension(self, tmp_path):
        path = tmp_path / "a.wav"
        # 24-bit isn't writable via wave.Wave_write.writeframes' usual
        # helpers, but setsampwidth(3) + raw little-endian bytes works.
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(3)
            wf.setframerate(8000)
            vals = [0, 8388607, -8388608]  # zero, max positive, min negative
            raw = b"".join(int(v).to_bytes(3, "little", signed=True) for v in vals)
            wf.writeframes(raw)

        data, sr, n_read = read_wav_chunk(str(path), 0, 3)
        assert n_read == 3
        assert data[0] == pytest.approx(0.0)
        assert data[1] == pytest.approx(1.0, abs=1e-6)
        assert data[2] == pytest.approx(-1.0, abs=1e-6)

    def test_unsupported_sample_width_raises(self, tmp_path):
        path = tmp_path / "a.wav"
        # `wave` refuses to write non-standard widths, so hand-build a
        # minimal WAV header (bits_per_sample=40 -> sampwidth=5 bytes),
        # which `wave` happily reads back, to exercise the real
        # unsupported-width branch in read_wav_chunk.
        bits_per_sample, n_channels, sample_rate = 40, 1, 8000
        data = b"\x00" * (5 * 3)
        byte_rate = sample_rate * n_channels * (bits_per_sample // 8)
        block_align = n_channels * (bits_per_sample // 8)
        fmt_chunk = struct.pack("<HHIIHH", 1, n_channels, sample_rate, byte_rate, block_align, bits_per_sample)
        riff_size = 4 + (8 + len(fmt_chunk)) + (8 + len(data))
        wav_bytes = (
            b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
            + b"fmt " + struct.pack("<I", len(fmt_chunk)) + fmt_chunk
            + b"data" + struct.pack("<I", len(data)) + data
        )
        path.write_bytes(wav_bytes)

        with pytest.raises(ValueError, match="Unsupported WAV sample width"):
            read_wav_chunk(str(path), 0, 3)


class TestCodecs:
    def test_encode_frame_round_trips(self):
        frame = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        b64, shape = encode_frame_ndarray_bgr(frame)
        assert shape == [4, 5, 3]
        decoded = np.frombuffer(base64.b64decode(b64), dtype=np.uint8).reshape(shape)
        assert np.array_equal(decoded, frame)

    def test_encode_audio_round_trips(self):
        audio = np.array([0.1, -0.5, 1.0, -1.0], dtype=np.float32)
        b64 = encode_audio_f32(audio)
        decoded = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
        assert np.allclose(decoded, audio)

    def test_encode_audio_coerces_dtype(self):
        audio = np.array([1, 2, 3], dtype=np.float64)
        b64 = encode_audio_f32(audio)
        decoded = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
        assert np.allclose(decoded, [1.0, 2.0, 3.0])


def _write_tiny_video(path, n_frames=5, size=(16, 12)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = size
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestVideoReader:
    def test_reads_expected_frame_count_then_none(self, tmp_path):
        path = tmp_path / "clip.mp4"
        _write_tiny_video(path, n_frames=5)

        reader = VideoReader(str(path))
        try:
            count = 0
            while reader.read() is not None:
                count += 1
            assert count == 5
        finally:
            reader.release()

    def test_resizes_to_target_size(self, tmp_path):
        path = tmp_path / "clip.mp4"
        _write_tiny_video(path, n_frames=2, size=(16, 12))

        reader = VideoReader(str(path), target_size=(8, 6))
        try:
            frame = reader.read()
            assert frame is not None
            assert frame.shape[:2] == (6, 8)  # (h, w)
        finally:
            reader.release()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(RuntimeError):
            VideoReader(str(tmp_path / "nope.mp4"))
