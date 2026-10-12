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
