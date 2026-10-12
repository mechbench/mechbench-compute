from __future__ import annotations

import types

import pytest

from mechbench_compute.torch_backend import memory

torch = pytest.importorskip("torch")

CUDA = types.SimpleNamespace(type="cuda", index=0)


def fake_gpu(monkeypatch, integrated, free_gb, reserved_gb=0):
    capped = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda index: types.SimpleNamespace(is_integrated=integrated,
                                                            total_memory=120 << 30))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (free_gb << 30, 120 << 30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda index: reserved_gb << 30)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction, index: capped.append((fraction, index)))
    return capped


def test_a_cpu_device_has_no_free_figure():
    assert memory.read_free_bytes(torch.device("cpu")) is None


@pytest.mark.parametrize("integrated", [True, False])
def test_free_memory_is_what_the_driver_reports(monkeypatch, integrated):
    fake_gpu(monkeypatch, integrated, 85)
    assert memory.read_free_bytes(CUDA) == 85 << 30


def test_an_integrated_gpu_is_capped_below_what_the_machine_has_free(monkeypatch):
    capped = fake_gpu(monkeypatch, True, 100, reserved_gb=10)
    got = memory.cap_integrated(CUDA)
    assert got == pytest.approx((100 + 10 - 8) / 120)
    assert capped == [(got, 0)]


def test_a_discrete_gpu_and_the_cpu_are_left_alone(monkeypatch):
    capped = fake_gpu(monkeypatch, False, 100)
    assert memory.cap_integrated(CUDA) is None
    assert memory.cap_integrated(torch.device("cpu")) is None
    assert capped == []


def test_the_cap_never_falls_to_nothing(monkeypatch):
    fake_gpu(monkeypatch, True, 2)
    assert memory.cap_integrated(CUDA) == 0.05
