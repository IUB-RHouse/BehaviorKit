"""Tests for module/check_gpus.py: GPU selection logic and the
CUDA_VISIBLE_DEVICES masking side effect. Real pynvml/nvidia-smi hardware
access is faked (a fake pynvml module injected into sys.modules; canned
nvidia-smi CSV output) since this machine has no GPU to query for real --
but the parsing/plumbing logic downstream of those calls is exercised for
real, not mocked away.
"""
import subprocess
import sys

import pytest

import module.check_gpus as cg


def _gpu(index, free_gb, total_gb=24.0, util_pct=None):
    return cg.GPUInfo(index=index, free_gb=free_gb, total_gb=total_gb, util_pct=util_pct)


class TestPickFreestGpu:
    def test_picks_most_free_among_candidates(self):
        gpus = [_gpu(0, 2.0), _gpu(1, 8.0), _gpu(2, 5.0)]
        chosen = cg.pick_freest_gpu(gpus, min_free_gb=1.0)
        assert chosen.index == 1

    def test_excludes_below_min_free_gb(self):
        gpus = [_gpu(0, 0.5), _gpu(1, 8.0)]
        chosen = cg.pick_freest_gpu(gpus, min_free_gb=1.0)
        assert chosen.index == 1

    def test_excludes_indices_in_exclude_list(self):
        gpus = [_gpu(0, 8.0), _gpu(1, 4.0)]
        chosen = cg.pick_freest_gpu(gpus, min_free_gb=1.0, exclude=[0])
        assert chosen.index == 1

    def test_excludes_above_max_util_pct(self):
        gpus = [_gpu(0, 8.0, util_pct=95.0), _gpu(1, 4.0, util_pct=10.0)]
        chosen = cg.pick_freest_gpu(gpus, min_free_gb=1.0, max_util_pct=50.0)
        assert chosen.index == 1

    def test_falls_back_to_emptiest_when_nothing_meets_threshold(self):
        gpus = [_gpu(0, 0.2), _gpu(1, 0.5)]
        chosen = cg.pick_freest_gpu(gpus, min_free_gb=10.0)
        assert chosen.index == 1  # emptiest overall, even though below threshold

    def test_returns_none_when_no_gpus(self):
        assert cg.pick_freest_gpu([], min_free_gb=1.0) is None


class TestMaskGpu:
    def test_sets_cuda_visible_devices(self, monkeypatch):
        # mask_gpu writes os.environ directly, so use monkeypatch.setenv
        # (not the real call) to make sure the mutation is undone even
        # though we're exercising the same os.environ assignment it does.
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        try:
            cg.mask_gpu(3)
            import os
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
        finally:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


class TestSelectAndMaskGpu:
    def test_returns_none_when_no_gpus_detected(self, monkeypatch):
        monkeypatch.setattr(cg, "list_gpus", lambda: [])
        assert cg.select_and_mask_gpu() is None

    def test_selects_and_masks_best_gpu(self, monkeypatch):
        monkeypatch.setattr(cg, "list_gpus", lambda: [_gpu(0, 2.0), _gpu(1, 10.0)])
        masked = {}
        monkeypatch.setattr(cg, "mask_gpu", lambda idx: masked.setdefault("index", idx))

        chosen = cg.select_and_mask_gpu(min_free_gb=1.0, set_visible=True)

        assert chosen == 1
        assert masked["index"] == 1

    def test_does_not_mask_when_set_visible_false(self, monkeypatch):
        monkeypatch.setattr(cg, "list_gpus", lambda: [_gpu(0, 10.0)])
        called = []
        monkeypatch.setattr(cg, "mask_gpu", lambda idx: called.append(idx))

        chosen = cg.select_and_mask_gpu(min_free_gb=1.0, set_visible=False)

        assert chosen == 0
        assert called == []

    def test_returns_none_when_nothing_meets_threshold_and_strict(self, monkeypatch):
        # pick_freest_gpu never actually returns None when gpus is non-empty
        # (it falls back to the emptiest one), so this exercises the
        # no-gpus-at-all early-return path under strict=True.
        monkeypatch.setattr(cg, "list_gpus", lambda: [])
        assert cg.select_and_mask_gpu(strict=True) is None


class _FakeMemInfo:
    def __init__(self, free_bytes, total_bytes):
        self.free = free_bytes
        self.total = total_bytes


class _FakeUtilRates:
    def __init__(self, gpu_pct):
        self.gpu = gpu_pct


def _make_fake_pynvml(devices, util_error=False, count_error=False, shutdown_error=False):
    """devices: list of (free_bytes, total_bytes, util_pct) tuples."""

    class _FakePynvml:
        def nvmlInit(self):
            pass

        def nvmlDeviceGetCount(self):
            if count_error:
                raise RuntimeError("simulated NVML failure")
            return len(devices)

        def nvmlDeviceGetHandleByIndex(self, i):
            return i

        def nvmlDeviceGetMemoryInfo(self, handle):
            free_b, total_b, _ = devices[handle]
            return _FakeMemInfo(free_b, total_b)

        def nvmlDeviceGetUtilizationRates(self, handle):
            if util_error:
                raise RuntimeError("simulated utilization read failure")
            _, _, util = devices[handle]
            return _FakeUtilRates(util)

        def nvmlShutdown(self):
            if shutdown_error:
                raise RuntimeError("simulated shutdown failure")

    return _FakePynvml()


class TestQueryWithPynvmlRealPath:
    def test_reports_gpus_from_fake_pynvml(self, monkeypatch):
        one_gb = 1024**3
        fake = _make_fake_pynvml([(2 * one_gb, 8 * one_gb, 50.0), (4 * one_gb, 8 * one_gb, 10.0)])
        monkeypatch.setitem(sys.modules, "pynvml", fake)

        gpus = cg._query_with_pynvml()

        assert len(gpus) == 2
        assert gpus[0].index == 0
        assert gpus[0].free_gb == pytest.approx(2.0, abs=1e-6)
        assert gpus[0].total_gb == pytest.approx(8.0, abs=1e-6)
        assert gpus[0].util_pct == 50.0

    def test_util_read_failure_leaves_util_pct_none(self, monkeypatch):
        one_gb = 1024**3
        fake = _make_fake_pynvml([(1 * one_gb, 4 * one_gb, 0.0)], util_error=True)
        monkeypatch.setitem(sys.modules, "pynvml", fake)

        gpus = cg._query_with_pynvml()

        assert len(gpus) == 1
        assert gpus[0].util_pct is None

    def test_device_count_failure_returns_empty_list(self, monkeypatch):
        fake = _make_fake_pynvml([], count_error=True)
        monkeypatch.setitem(sys.modules, "pynvml", fake)

        assert cg._query_with_pynvml() == []

    def test_shutdown_failure_does_not_propagate(self, monkeypatch):
        one_gb = 1024**3
        fake = _make_fake_pynvml([(one_gb, one_gb, 0.0)], shutdown_error=True)
        monkeypatch.setitem(sys.modules, "pynvml", fake)

        gpus = cg._query_with_pynvml()  # must not raise despite nvmlShutdown failing
        assert len(gpus) == 1


class TestQueryWithNvidiaSmiRealParsing:
    def test_parses_real_csv_output(self, monkeypatch):
        monkeypatch.setattr(cg.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
        csv_out = "0, 2048, 8192, 50\n1, 4096, 8192, N/A\n"
        monkeypatch.setattr(cg.subprocess, "check_output", lambda *a, **k: csv_out)

        gpus = cg._query_with_nvidia_smi()

        assert len(gpus) == 2
        assert gpus[0].index == 0
        assert gpus[0].free_gb == pytest.approx(2.0, abs=1e-6)  # MiB -> GiB
        assert gpus[0].total_gb == pytest.approx(8.0, abs=1e-6)
        assert gpus[0].util_pct == 50.0
        assert gpus[1].util_pct is None  # "N/A" -> None

    def test_skips_malformed_lines(self, monkeypatch):
        monkeypatch.setattr(cg.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
        csv_out = "not,enough\n0, 2048, 8192, 10\nnot_an_int, 2048, 8192, 10\n"
        monkeypatch.setattr(cg.subprocess, "check_output", lambda *a, **k: csv_out)

        gpus = cg._query_with_nvidia_smi()

        assert len(gpus) == 1
        assert gpus[0].index == 0

    def test_subprocess_failure_returns_empty_list(self, monkeypatch):
        monkeypatch.setattr(cg.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

        def _raise(*a, **k):
            raise subprocess.CalledProcessError(1, "nvidia-smi")

        monkeypatch.setattr(cg.subprocess, "check_output", _raise)

        assert cg._query_with_nvidia_smi() == []

    def test_small_gib_values_not_misread_as_mib(self, monkeypatch):
        """Values <=256 are assumed already-GiB (a GPU with <=256GB total
        would never report in MiB in practice); confirms the heuristic
        leaves small values untouched."""
        monkeypatch.setattr(cg.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
        csv_out = "0, 12, 24, 5\n"
        monkeypatch.setattr(cg.subprocess, "check_output", lambda *a, **k: csv_out)

        gpus = cg._query_with_nvidia_smi()

        assert gpus[0].free_gb == pytest.approx(12.0, abs=1e-6)
        assert gpus[0].total_gb == pytest.approx(24.0, abs=1e-6)


class TestQueryGracefulFailure:
    def test_pynvml_query_returns_empty_list_when_unavailable(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "pynvml":
                raise ImportError("no pynvml")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        assert cg._query_with_pynvml() == []

    def test_nvidia_smi_query_returns_empty_list_when_binary_missing(self, monkeypatch):
        monkeypatch.setattr(cg.shutil, "which", lambda name: None)
        assert cg._query_with_nvidia_smi() == []

    def test_list_gpus_falls_back_to_nvidia_smi_when_pynvml_empty(self, monkeypatch):
        monkeypatch.setattr(cg, "_query_with_pynvml", lambda: [])
        monkeypatch.setattr(cg, "_query_with_nvidia_smi", lambda: [_gpu(0, 5.0)])
        gpus = cg.list_gpus()
        assert len(gpus) == 1
        assert gpus[0].index == 0


class TestFmtGpu:
    def test_formats_known_utilization(self):
        s = cg._fmt_gpu(_gpu(0, 5.0, total_gb=10.0, util_pct=42.0))
        assert s == "cuda:0  free=5.0GB / 10.0GB  util=42%"

    def test_formats_unknown_utilization(self):
        s = cg._fmt_gpu(_gpu(1, 5.0, total_gb=10.0, util_pct=None))
        assert "util=N/A" in s


class TestMainCli:
    def test_prints_no_gpus_message_and_returns_0(self, monkeypatch, capsys):
        monkeypatch.setattr(cg, "list_gpus", lambda: [])
        code = cg.main([])
        assert code == 0
        assert "No GPUs detected" in capsys.readouterr().out

    def test_strict_returns_1_when_no_gpus(self, monkeypatch):
        monkeypatch.setattr(cg, "list_gpus", lambda: [])
        assert cg.main(["--strict"]) == 1

    def test_prints_selection_and_sets_env_when_requested(self, monkeypatch, capsys):
        monkeypatch.setattr(cg, "list_gpus", lambda: [_gpu(0, 5.0)])
        masked = []
        monkeypatch.setattr(cg, "mask_gpu", lambda idx: masked.append(idx))

        code = cg.main(["--set-env"])

        assert code == 0
        assert masked == [0]
        out = capsys.readouterr().out
        assert "Selected:" in out
        assert "CUDA_VISIBLE_DEVICES set to 0" in out

    def test_prints_no_suitable_gpu_when_pick_freest_returns_none(self, monkeypatch, capsys):
        # pick_freest_gpu never actually returns None given a non-empty
        # gpus list (it falls back to the emptiest one), so this defensive
        # branch is unreachable through real inputs -- exercise it directly
        # by faking pick_freest_gpu's return value.
        monkeypatch.setattr(cg, "list_gpus", lambda: [_gpu(0, 5.0)])
        monkeypatch.setattr(cg, "pick_freest_gpu", lambda *a, **k: None)

        code = cg.main([])

        assert code == 0
        assert "No suitable GPU found." in capsys.readouterr().out

    def test_strict_returns_1_when_pick_freest_returns_none(self, monkeypatch):
        monkeypatch.setattr(cg, "list_gpus", lambda: [_gpu(0, 5.0)])
        monkeypatch.setattr(cg, "pick_freest_gpu", lambda *a, **k: None)

        assert cg.main(["--strict"]) == 1


class TestSelectAndMaskGpuChosenNoneDefensiveBranch:
    """Same unreachable-in-practice defensive branch as
    TestMainCli::test_*_pick_freest_returns_none, but inside
    select_and_mask_gpu instead of the main() CLI."""

    def test_returns_none_when_pick_freest_returns_none(self, monkeypatch):
        monkeypatch.setattr(cg, "list_gpus", lambda: [_gpu(0, 5.0)])
        monkeypatch.setattr(cg, "pick_freest_gpu", lambda *a, **k: None)

        assert cg.select_and_mask_gpu() is None

    def test_strict_also_returns_none_when_pick_freest_returns_none(self, monkeypatch):
        monkeypatch.setattr(cg, "list_gpus", lambda: [_gpu(0, 5.0)])
        monkeypatch.setattr(cg, "pick_freest_gpu", lambda *a, **k: None)

        assert cg.select_and_mask_gpu(strict=True) is None


class TestMainGuard:
    def test_running_as_a_script_invokes_main_and_exits_cleanly(self):
        # Deliberately doesn't assert on GPU-count-dependent output (this
        # machine may or may not have a real NVIDIA GPU) -- just confirms
        # the `if __name__ == "__main__":` entry point actually runs main()
        # and exits with a valid code rather than hanging or crashing.
        result = subprocess.run(
            [sys.executable, "module/check_gpus.py"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode in (0, 1)
        assert result.stdout.strip() != ""
