"""Tests for module/gaze_backends/: the pluggable gaze.method dispatcher,
plus real-weight load/predict tests for the two backends whose weights are
bundled (l2cs, mobilegaze). gaze360 is skipped since its weights are
license-gated and not present in the repo.
"""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from module.gaze_backends import GazeBackend, build_gaze_backend

L2CS_WEIGHTS = Path("data/models/l2cs/L2CSNet_gaze360.pkl")
MOBILEGAZE_WEIGHTS = Path("data/models/mobilegaze/mobilenetv2.pt")


def _random_face_crop(h: int = 220, w: int = 180) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


class TestBuildGazeBackendDispatch:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown gaze method"):
            build_gaze_backend({"method": "not-a-real-method"}, "cpu")

    def test_defaults_to_gaze360_when_method_missing(self, monkeypatch):
        # Dispatch logic only - avoid actually loading gaze360 weights
        # (license-gated, not bundled) by faking the backend class.
        import module.gaze_backends as gb

        built = {}

        class _FakeGaze360:
            def __init__(self, weights_path, device):
                built["weights_path"] = weights_path
                built["device"] = device

        monkeypatch.setattr("module.gaze_backends.gaze360_backend.Gaze360Backend", _FakeGaze360)
        backend = build_gaze_backend({"gaze360_model": "some/path.pth"}, "cpu")
        assert isinstance(backend, _FakeGaze360)
        assert built == {"weights_path": "some/path.pth", "device": "cpu"}

    def test_dispatches_to_l2cs(self, monkeypatch):
        import module.gaze_backends.l2cs as l2cs_pkg

        sentinel = object()
        monkeypatch.setattr(l2cs_pkg, "L2CSBackend", lambda weights_path, device: sentinel)
        backend = build_gaze_backend({"method": "l2cs", "l2cs_model": "x.pkl"}, "cpu")
        assert backend is sentinel

    def test_dispatches_to_mobilegaze_with_arch(self, monkeypatch):
        import module.gaze_backends.mobilegaze as mg_pkg

        captured = {}

        def _fake(weights_path, device, arch="mobilenetv2"):
            captured.update(weights_path=weights_path, device=device, arch=arch)
            return "backend"

        monkeypatch.setattr(mg_pkg, "MobileGazeBackend", _fake)
        result = build_gaze_backend(
            {"method": "mobilegaze", "mobilegaze_model": "y.pt", "mobilegaze_arch": "resnet34"}, "cpu"
        )
        assert result == "backend"
        assert captured == {"weights_path": "y.pt", "device": "cpu", "arch": "resnet34"}


@pytest.mark.skipif(not L2CS_WEIGHTS.exists(), reason="l2cs weights not present")
class TestL2CSBackend:
    @staticmethod
    @pytest.fixture(scope="class")
    def backend():
        from module.gaze_backends.l2cs import L2CSBackend
        return L2CSBackend(str(L2CS_WEIGHTS), "cpu")

    def test_method_name(self, backend):
        assert backend.method_name == "l2cs"
        assert isinstance(backend, GazeBackend)

    def test_predict_returns_yaw_pitch_in_radians(self, backend):
        yaw, pitch = backend.predict(_random_face_crop())
        assert isinstance(yaw, float)
        assert isinstance(pitch, float)
        # Gaze360-convention angles should stay within a physically plausible range.
        assert -np.pi <= yaw <= np.pi
        assert -np.pi <= pitch <= np.pi

    def test_predict_is_deterministic(self, backend):
        crop = _random_face_crop()
        assert backend.predict(crop) == backend.predict(crop)


@pytest.mark.skipif(not MOBILEGAZE_WEIGHTS.exists(), reason="mobilegaze weights not present")
class TestMobileGazeBackend:
    @staticmethod
    @pytest.fixture(scope="class")
    def backend():
        from module.gaze_backends.mobilegaze import MobileGazeBackend
        return MobileGazeBackend(str(MOBILEGAZE_WEIGHTS), "cpu", arch="mobilenetv2")

    def test_method_name(self, backend):
        assert backend.method_name == "mobilegaze"
        assert isinstance(backend, GazeBackend)

    def test_predict_returns_yaw_pitch_in_radians(self, backend):
        yaw, pitch = backend.predict(_random_face_crop())
        assert isinstance(yaw, float)
        assert isinstance(pitch, float)
        assert -np.pi <= yaw <= np.pi
        assert -np.pi <= pitch <= np.pi

    def test_predict_is_deterministic(self, backend):
        crop = _random_face_crop()
        assert backend.predict(crop) == backend.predict(crop)

    def test_unknown_arch_raises(self):
        from module.gaze_backends.mobilegaze import MobileGazeBackend
        with pytest.raises(ValueError, match="Unknown mobilegaze arch"):
            MobileGazeBackend(str(MOBILEGAZE_WEIGHTS), "cpu", arch="not-a-real-arch")


class TestMobileGazeResNetArchitectures:
    """resnet18/34/50 are alternative mobilegaze backbones with no bundled
    checkpoint (only mobilenetv2.pt ships in the repo), so these exercise
    the architectures directly with pretrained=False (no network access)
    rather than through MobileGazeBackend.
    """

    @pytest.mark.parametrize("builder_name", ["resnet18", "resnet34", "resnet50"])
    def test_forward_returns_yaw_pitch_logits(self, builder_name):
        from module.gaze_backends.mobilegaze import resnet as resnet_mod

        builder = getattr(resnet_mod, builder_name)
        model = builder(pretrained=False, num_classes=90)
        model.eval()
        x = torch.zeros(2, 3, 448, 448)
        with torch.no_grad():
            yaw, pitch = model(x)
        assert yaw.shape == (2, 90)
        assert pitch.shape == (2, 90)


class TestGaze360Backend:
    """gaze360's official weights are license-gated and not bundled, so
    these build a real GazeLSTM with random init (mocking the network
    call its constructor otherwise makes for ImageNet-pretrained
    resnet18), save it as a checkpoint, and load it back through the real
    Gaze360Backend.__init__/predict path -- covering the actual
    state-dict-stripping and inference logic, not just the dispatcher.
    """

    @pytest.fixture
    def synthetic_checkpoint(self, tmp_path, monkeypatch):
        import module.gaze_model as gm

        monkeypatch.setattr(gm.model_zoo, "load_url", lambda url: {})
        model = gm.GazeLSTM()
        # Mimic the real checkpoint's "module." (DataParallel) prefix and
        # its {"state_dict": ...} wrapping, both of which the backend strips.
        state = {"module." + k: v for k, v in model.state_dict().items()}
        ckpt_path = tmp_path / "gaze360_model.pth"
        torch.save({"state_dict": state}, ckpt_path)
        return ckpt_path

    def test_predict_returns_yaw_pitch_in_radians(self, monkeypatch, synthetic_checkpoint):
        import module.gaze_model as gm
        monkeypatch.setattr(gm.model_zoo, "load_url", lambda url: {})

        from module.gaze_backends.gaze360_backend import Gaze360Backend

        backend = Gaze360Backend(str(synthetic_checkpoint), "cpu")
        assert backend.method_name == "gaze360"

        rng = np.random.default_rng(2)
        crop = rng.integers(0, 256, size=(200, 180, 3), dtype=np.uint8)
        yaw, pitch = backend.predict(crop)

        assert isinstance(yaw, float)
        assert isinstance(pitch, float)
        assert -np.pi - 1e-4 <= yaw <= np.pi + 1e-4
        assert -np.pi / 2 - 1e-4 <= pitch <= np.pi / 2 + 1e-4
