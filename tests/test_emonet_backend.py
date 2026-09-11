"""Tests for the EmoNet emotion backend (module/emonet_backend/).

Uses the real bundled weights (data/models/emonet/emonet_8.pth) rather than
mocking torch, since the whole point of these tests is to catch a mismatch
between the vendored architecture and the checkpoint it has to load.
"""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from module.emonet_backend import EmoNetBackend
from module.emonet_backend.backend import _EXPRESSION_CLASSES, _prep_256
from module.emonet_backend.emonet import EmoNet

WEIGHTS_PATH = Path("data/models/emonet/emonet_8.pth")

pytestmark = pytest.mark.skipif(
    not WEIGHTS_PATH.exists(),
    reason="data/models/emonet/emonet_8.pth not present",
)


@pytest.fixture(scope="module")
def backend() -> EmoNetBackend:
    return EmoNetBackend(str(WEIGHTS_PATH), "cpu")


def _random_face_crop(h: int = 300, w: int = 220) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


class TestCheckpointCompatibility:
    def test_vendored_architecture_matches_checkpoint_keys(self):
        """Regression guard: if module/emonet_backend/emonet.py is ever
        edited, this fails loudly instead of silently loading a
        partially-initialized (effectively random) model via strict=False.
        """
        model = EmoNet(n_expression=8)
        state = torch.load(str(WEIGHTS_PATH), map_location="cpu")
        state = {k.replace("module.", ""): v for k, v in state.items()}

        model_keys = set(dict(model.named_parameters())) | set(dict(model.named_buffers()))
        state_keys = set(state)

        assert model_keys - state_keys == set(), "checkpoint is missing keys the model expects"
        assert state_keys - model_keys == set(), "checkpoint has keys the model doesn't define"


class TestPreprocessing:
    def test_prep_256_shape_dtype_range(self):
        crop = _random_face_crop(300, 220)
        chw = _prep_256(crop)

        assert chw.shape == (3, 256, 256)
        assert chw.dtype == np.float32
        assert chw.min() >= 0.0
        assert chw.max() <= 1.0

    def test_prep_256_handles_non_square_crops(self):
        # face crops from the YOLO detector are rarely square
        for h, w in [(50, 300), (300, 50), (1, 1), (256, 256)]:
            chw = _prep_256(_random_face_crop(h, w))
            assert chw.shape == (3, 256, 256)


class TestEmoNetBackend:
    def test_method_name(self, backend: EmoNetBackend):
        assert backend.method_name == "emonet"

    def test_predict_output_contract(self, backend: EmoNetBackend):
        out = backend.predict(_random_face_crop())

        assert set(out) == {"expression", "expression_scores", "valence", "arousal"}
        assert out["expression"] in _EXPRESSION_CLASSES.values()
        assert set(out["expression_scores"]) == set(_EXPRESSION_CLASSES.values())

        scores = out["expression_scores"]
        assert all(0.0 <= s <= 1.0 for s in scores.values())
        assert scores[out["expression"]] == max(scores.values())
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-4)

        assert -1.0 <= out["valence"] <= 1.0
        assert -1.0 <= out["arousal"] <= 1.0

    def test_predict_is_deterministic(self, backend: EmoNetBackend):
        crop = _random_face_crop()
        out1 = backend.predict(crop)
        out2 = backend.predict(crop)
        assert out1 == out2

    def test_predict_accepts_arbitrary_crop_sizes(self, backend: EmoNetBackend):
        for h, w in [(64, 64), (300, 220), (480, 640)]:
            out = backend.predict(_random_face_crop(h, w))
            assert out["expression"] in _EXPRESSION_CLASSES.values()
