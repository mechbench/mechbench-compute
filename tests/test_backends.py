from __future__ import annotations

import importlib.util
import sys

import pytest

from mechbench_compute import backends


def test_this_machine_reports_its_backend():
    active = backends.active()
    assert active is not None, "the test machine should have MLX"
    assert active.name == "mlx"
    assert "darwin" in backends.describe_platform()


def test_a_machine_with_no_backend_gets_an_explanation(monkeypatch):
    real_find_spec = importlib.util.find_spec

    def blind(name, *args, **kwargs):
        if name.split(".")[0] in {"mlx", "torch"}:
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(backends.importlib.util, "find_spec", blind)

    assert backends.available() == []
    assert backends.active() is None
    with pytest.raises(ImportError) as exc:
        backends.require()

    message = str(exc.value)
    assert "no compute backend" in message
    assert backends.describe_platform() in message
    assert "macOS on Apple Silicon" in message
    assert "doctor" in message


def test_backend_detection_does_not_import_the_substrate(monkeypatch):
    loaded = []
    real_import = __import__

    def watched(name, *args, **kwargs):
        if name.split(".")[0] == "mlx":
            loaded.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", watched)
    for m in [k for k in sys.modules if k.startswith("mlx")]:
        pass
    before = len(loaded)
    backends.available()
    assert len(loaded) == before


class TestImportableWithoutABackend:
    @staticmethod
    def _without_mlx(monkeypatch):
        import importlib.util
        import sys

        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda n, *a, **k: None if n.split(".")[0] == "mlx" else real(n, *a, **k),
        )
        for name in [m for m in list(sys.modules) if m.startswith("mechbench_compute")]:
            monkeypatch.delitem(sys.modules, name, raising=False)

    def test_the_package_imports(self, monkeypatch):
        self._without_mlx(monkeypatch)
        import mechbench_compute

        assert mechbench_compute.active_backend() is None

    def test_the_reporting_submodules_are_reachable(self, monkeypatch):
        self._without_mlx(monkeypatch)
        from mechbench_compute import backends, inventory

        assert backends.active() is None
        assert callable(inventory.scan)

    def test_touching_the_model_api_explains_itself(self, monkeypatch):
        self._without_mlx(monkeypatch)
        import mechbench_compute

        with pytest.raises(ImportError, match="no compute backend"):
            _ = mechbench_compute.Model
