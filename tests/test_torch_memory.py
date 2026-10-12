from __future__ import annotations

import types

import pytest

from mechbench_compute.torch_backend import memory

torch = pytest.importorskip("torch")


def test_host_available_is_read_from_meminfo(tmp_path):
    info = tmp_path / "meminfo"
    info.write_text("MemTotal:       121700000 kB\nMemFree:        1000 kB\nMemAvailable:   120000000 kB\n")
    assert memory.read_host_available(info) == 120000000 * 1024


def test_host_available_without_the_line_is_none(tmp_path):
    info = tmp_path / "meminfo"
    info.write_text("MemTotal: 1 kB\n")
    assert memory.read_host_available(info) is None
    assert memory.read_host_available(tmp_path / "absent") is None


def test_a_cpu_device_has_no_free_figure():
    assert memory.read_free_bytes(torch.device("cpu")) is None


@pytest.mark.parametrize("integrated, expected", [(True, 64 << 30), (False, 2 << 30)])
def test_an_integrated_gpu_reads_the_host(monkeypatch, integrated, expected):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda index: types.SimpleNamespace(is_integrated=integrated))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (2 << 30, 121 << 30))
    monkeypatch.setattr(memory, "read_host_available", lambda: 64 << 30)
    assert memory.read_free_bytes(types.SimpleNamespace(type="cuda", index=0)) == expected


def fake_gpu(monkeypatch, integrated, reserved=0):
    capped = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda index: types.SimpleNamespace(is_integrated=integrated,
                                                            total_memory=120 << 30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda index: reserved)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction, index: capped.append((fraction, index)))
    return capped


def meminfo(tmp_path, available_gb):
    info = tmp_path / "meminfo"
    info.write_text(f"MemAvailable: {available_gb * 1024 * 1024} kB\n")
    return info


def test_an_integrated_gpu_is_capped_below_what_the_host_can_spare(monkeypatch, tmp_path):
    capped = fake_gpu(monkeypatch, True, reserved=10 << 30)
    got = memory.cap_integrated(types.SimpleNamespace(type="cuda", index=0), meminfo(tmp_path, 100))
    assert got == pytest.approx((100 + 10 - 8) / 120)
    assert capped == [(got, 0)]


def test_a_discrete_gpu_and_the_cpu_are_left_alone(monkeypatch, tmp_path):
    capped = fake_gpu(monkeypatch, False)
    assert memory.cap_integrated(types.SimpleNamespace(type="cuda", index=0), meminfo(tmp_path, 100)) is None
    assert memory.cap_integrated(torch.device("cpu")) is None
    assert capped == []


def test_the_cap_never_falls_to_nothing(monkeypatch, tmp_path):
    fake_gpu(monkeypatch, True)
    assert memory.cap_integrated(types.SimpleNamespace(type="cuda", index=0), meminfo(tmp_path, 2)) == 0.05
