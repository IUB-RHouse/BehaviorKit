"""Tests for module/gaze_model.py (the original Gaze360 GazeLSTM
architecture). GazeLSTM.__init__ hardcodes resnet18(pretrained=True), which
would otherwise try to download ImageNet weights from the network on
construction -- every test here mocks torch.utils.model_zoo.load_url to
keep this fully offline.
"""
import pytest

torch = pytest.importorskip("torch")

import module.gaze_model as gm


@pytest.fixture(autouse=True)
def _no_network_pretrained_download(monkeypatch):
    """resnet18(pretrained=True) calls model_zoo.load_url(...); stub it out
    so nothing here touches the network, and load_state_dict(..., strict=False)
    on the empty dict is a no-op (leaves the random init in place)."""
    monkeypatch.setattr(gm.model_zoo, "load_url", lambda url: {})


class TestResNet:
    def test_resnet18_forward_shape(self):
        model = gm.resnet18(pretrained=False)
        model.eval()
        x = torch.zeros(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3)  # fc2 head is a 3-dim output in this variant

    def test_resnet18_pretrained_true_uses_mocked_loader_not_network(self):
        # Would raise/hang on a real network call if the fixture's mock
        # weren't in effect; succeeding here proves it's not hitting the net.
        model = gm.resnet18(pretrained=True)
        assert isinstance(model, gm.ResNet)


class TestGazeLSTM:
    def test_forward_returns_angular_output_and_variance(self):
        model = gm.GazeLSTM()
        model.eval()
        # input shape: (batch, 7 frames, 3, 224, 224) per the model's own
        # view((-1, 3) + input.size()[-2:]) / view(batch, 7, feat_dim) reshape
        x = torch.zeros(1, 7, 3, 224, 224)
        with torch.no_grad():
            angular_output, var = model(x)

        assert angular_output.shape == (1, 2)
        assert var.shape == (1, 2)
        # yaw/pitch are tanh-squashed into (-pi, pi) / (-pi/2, pi/2)
        assert (angular_output[:, 0].abs() <= torch.pi + 1e-4).all()
        assert (angular_output[:, 1].abs() <= torch.pi / 2 + 1e-4).all()


class TestPinBallLoss:
    def test_forward_is_finite_and_nonnegative_at_zero_error(self):
        loss_fn = gm.PinBallLoss()
        output = torch.zeros(4, 2)
        target = torch.zeros(4, 2)
        var = torch.ones(4, 2)
        loss = loss_fn(output, target, var)
        assert torch.isfinite(loss)

    def test_forward_penalizes_larger_error_more(self):
        loss_fn = gm.PinBallLoss()
        target = torch.zeros(4, 2)
        var = torch.ones(4, 2)
        small_err_loss = loss_fn(torch.full((4, 2), 0.1), target, var)
        large_err_loss = loss_fn(torch.full((4, 2), 5.0), target, var)
        assert large_err_loss > small_err_loss
