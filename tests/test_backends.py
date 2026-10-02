from __future__ import annotations

import importlib.util
import sys

import pytest

from mechbench_compute import backends


@pytest.mark.skipif(backends.detect_accelerator() != "metal",
                    reason="the MLX test machine is Apple silicon; this one is not")
def test_this_machine_reports_its_backend():
    active = backends.active()
    assert active is not None, "the test machine should have MLX"
    assert active.name == "mlx"
    assert "darwin" in backends.describe_platform()
    assert backends.detect_accelerator() == "metal"
    assert backends.advertise() == {"accelerator": "metal", "backends": ["mlx"]}


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
    assert "mlx: mlx.core is not installed" in message
    assert "macOS on Apple Silicon" in message
    assert "doctor" in message


def test_a_backend_whose_package_is_missing_is_absent_not_an_error():
    assert backends.is_importable("numpy")
    assert not backends.is_importable("no_such_package_anywhere.core")
    gone = backends.Backend(name="gone", module="no_such_package_anywhere.core", label="gone",
                            platform_label="nowhere", accelerators=("cpu",), extra="gone")
    assert backends.read_absence(gone, "cpu") == (
        "no_such_package_anywhere.core is not installed: pip install 'mechbench-compute[gone]'")
    three = backends.Backend(name="three", module="no_such_a", label="three", platform_label="nowhere",
                             accelerators=("cpu",), requires=("no_such_b", "no_such_c"))
    assert backends.read_absence(three, "cpu") == "no_such_a, no_such_b and no_such_c are not installed"
    assert backends.available("cpu", declared=(gone,)) == []


def test_backend_detection_does_not_import_the_backend(monkeypatch):
    loaded = []
    real_import = __import__

    def watched(name, *args, **kwargs):
        if name.split(".")[0] in {"mlx", "torch", "nnsight"}:
            loaded.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", watched)
    before = len(loaded)
    backends.available()
    backends.describe()
    backends.advertise()
    backends.select(backends.advertise())
    assert len(loaded) == before


@pytest.mark.parametrize("platform_name,machine,tools,accelerator", [
    ("darwin", "arm64", set(), "metal"),
    ("linux", "x86_64", {"nvidia-smi"}, "cuda"),
    ("linux", "x86_64", {"rocm-smi"}, "rocm"),
    ("linux", "x86_64", set(), "cpu"),
    ("darwin", "x86_64", set(), "cpu"),
])
def test_the_accelerator_is_the_hardware(monkeypatch, platform_name, machine, tools, accelerator):
    monkeypatch.setattr(backends.sys, "platform", platform_name)
    monkeypatch.setattr(backends.platform, "machine", lambda: machine)
    monkeypatch.setattr(backends.shutil, "which", lambda tool: f"/usr/bin/{tool}" if tool in tools else None)
    assert backends.detect_accelerator() == accelerator
    assert accelerator in backends.ACCELERATORS


def test_a_backend_installed_for_another_accelerator_is_absent_but_named():
    assert "mlx" not in [b.name for b in backends.available("cuda")]
    described = backends.describe("cuda")
    assert [d["name"] for d in described] == [b.name for b in backends.BACKENDS] == ["mlx", "torch"]
    mlx = next(d for d in described if d["name"] == "mlx")
    assert mlx["present"] is False
    assert mlx["absent"] == "it runs on metal, and this machine's accelerator is cuda"
    on_metal = backends.describe("metal")
    assert on_metal[0] == {
        "name": "mlx", "label": backends.BACKENDS[0].label, "accelerators": ["metal"],
        "present": True}
    assert on_metal[1]["present"] is False
    absent = on_metal[1]["absent"]
    assert (absent == "it runs on cuda, and this machine's accelerator is metal"
            or absent.endswith(" not installed: pip install 'mechbench-compute[torch]'")), absent


def test_the_torch_backend_is_offered_on_cuda_and_installed_with_its_extra():
    torch = next(b for b in backends.BACKENDS if b.name == "torch")
    assert torch.accelerators == ("cuda",)
    assert torch.modules == ("torch", "nnsight", "transformers")
    assert torch.extra == "torch"
    installed = backends.is_installed(torch)
    assert backends.is_available("torch", "cuda") is installed
    assert backends.is_available("torch", "metal") is False
    assert backends.advertise("cuda")["backends"] == (["torch"] if installed else [])


class TestSelectionByCapability:
    def test_the_fake_backend_is_selected_by_what_a_runner_advertises(self):
        from tests.fake_backend import DECLARED, FAKE, KIT

        assert backends.select(KIT.capabilities, backend="fake", declared=DECLARED) is FAKE
        assert backends.select(KIT.capabilities, declared=DECLARED) is FAKE
        assert KIT.select() is FAKE

    def test_a_job_naming_a_backend_this_laptop_lacks_is_refused_by_name(self):
        from tests.fake_backend import DECLARED

        laptop = {"accelerator": "metal", "backends": ["mlx"]}
        with pytest.raises(backends.BackendRefused,
                           match=r"^it needs the fake backend, and this runner has mlx$"):
            backends.select(laptop, backend="fake", declared=DECLARED)
        assert backends.select(laptop, backend="mlx", declared=DECLARED).name == "mlx"

    def test_a_job_may_need_an_accelerator_or_a_backend_or_both(self):
        laptop = {"accelerator": "metal", "backends": ["mlx"]}
        assert backends.select(laptop, backend="mlx", accelerator="metal").name == "mlx"
        with pytest.raises(backends.BackendRefused,
                           match=r"^it needs a cuda accelerator, and this runner has metal$"):
            backends.select(laptop, accelerator="cuda")
        with pytest.raises(backends.BackendRefused,
                           match=r"^'jax' is not a backend compute declares; it declares mlx, torch$"):
            backends.select(laptop, backend="jax")
        with pytest.raises(backends.BackendRefused, match=r"advertises none on metal"):
            backends.select({"accelerator": "metal", "backends": []})

    def test_a_runner_with_two_backends_runs_a_job_that_names_neither_on_the_first_declared(self):
        from tests.fake_backend import DECLARED

        both = {"accelerator": "cpu", "backends": ["fake", "mlx"]}
        assert backends.select(both, declared=DECLARED).name == "fake"
        assert backends.select({**both, "accelerator": "metal"}, declared=DECLARED).name == "mlx"


class TestImportableWithoutABackend:
    @staticmethod
    def _without_mlx(monkeypatch):
        import importlib.util

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
        assert backends.describe()[0]["absent"] == "mlx.core is not installed"
        assert callable(inventory.scan)

    def test_touching_the_model_api_explains_itself(self, monkeypatch):
        self._without_mlx(monkeypatch)
        import mechbench_compute

        with pytest.raises(ImportError, match="no compute backend"):
            _ = mechbench_compute.Model
