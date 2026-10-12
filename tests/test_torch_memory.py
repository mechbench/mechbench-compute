from __future__ import annotations

import subprocess
import types

import pytest

from mechbench_compute.torch_backend import memory

torch = pytest.importorskip("torch")

CUDA = types.SimpleNamespace(type="cuda", index=0)
GB = 2**30


def meminfo(tmp_path, total_gb, available_gb):
    info = tmp_path / "meminfo"
    info.write_text(f"MemTotal: {total_gb * 1024 * 1024} kB\nMemFree: 1 kB\n"
                    f"MemAvailable: {available_gb * 1024 * 1024} kB\n")
    return info


def fake_gpu(monkeypatch, tmp_path, integrated, *, driver_free_gb=60, reserved_gb=0,
             host_used_gb=4, others_gb=0):
    capped = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda index: types.SimpleNamespace(is_integrated=integrated,
                                                            total_memory=120 * GB))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (driver_free_gb * GB, 120 * GB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda index: reserved_gb * GB)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction, index: capped.append((fraction, index)))
    monkeypatch.setattr(memory, "MEMINFO", meminfo(tmp_path, 116, 116 - host_used_gb))
    monkeypatch.setattr(memory.read_integrated_budget, "__defaults__",
                        (meminfo(tmp_path, 116, 116 - host_used_gb),))
    monkeypatch.setattr(memory, "read_others_on_gpu", lambda own=None: others_gb * GB)
    return capped


def test_host_use_is_read_from_meminfo(tmp_path):
    assert memory.read_host_used(meminfo(tmp_path, 116, 100)) == 16 * GB


def test_host_use_without_the_lines_is_none(tmp_path):
    info = tmp_path / "meminfo"
    info.write_text("MemTotal: 1 kB\n")
    assert memory.read_host_used(info) is None
    assert memory.read_host_used(tmp_path / "absent") is None


def test_a_cpu_device_has_no_free_figure():
    assert memory.read_free_bytes(torch.device("cpu")) is None


def test_a_discrete_gpu_reads_the_driver(monkeypatch, tmp_path):
    fake_gpu(monkeypatch, tmp_path, False, driver_free_gb=60)
    assert memory.read_free_bytes(CUDA) == 60 * GB


def test_an_integrated_gpu_counts_the_hosts_cache_as_free(monkeypatch, tmp_path):
    fake_gpu(monkeypatch, tmp_path, True, driver_free_gb=59, reserved_gb=10, host_used_gb=4,
             others_gb=6)
    assert memory.read_free_bytes(CUDA) == (120 - 4 - 6 - 10) * GB


def test_an_integrated_gpu_is_capped_below_the_machines_budget(monkeypatch, tmp_path):
    capped = fake_gpu(monkeypatch, tmp_path, True, driver_free_gb=59, host_used_gb=4, others_gb=6)
    got = memory.cap_integrated(CUDA)
    assert got == pytest.approx((120 - 4 - 6 - 12) / 120)
    assert capped == [(got, 0)]


def test_without_meminfo_the_cap_falls_back_to_the_driver(monkeypatch, tmp_path):
    fake_gpu(monkeypatch, tmp_path, True, driver_free_gb=100, reserved_gb=10)
    monkeypatch.setattr(memory, "read_integrated_budget", lambda index: None)
    assert memory.cap_integrated(CUDA) == pytest.approx((100 + 10 - 12) / 120)


def test_a_discrete_gpu_and_the_cpu_are_left_alone(monkeypatch, tmp_path):
    capped = fake_gpu(monkeypatch, tmp_path, False)
    assert memory.cap_integrated(CUDA) is None
    assert memory.cap_integrated(torch.device("cpu")) is None
    assert capped == []


def test_the_cap_never_falls_to_nothing(monkeypatch, tmp_path):
    fake_gpu(monkeypatch, tmp_path, True, host_used_gb=110)
    assert memory.cap_integrated(CUDA) == 0.05


def test_other_processes_are_summed_without_this_one(monkeypatch):
    out = "100, 997\n200, 5314\n300, [N/A]\n"
    monkeypatch.setattr(memory.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout=out))
    assert memory.read_others_on_gpu(own=100) == 5314 * 2**20


def test_without_nvidia_smi_no_other_process_is_counted(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(memory.subprocess, "run", missing)
    assert memory.read_others_on_gpu() == 0
    monkeypatch.setattr(memory.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("x", 10)))
    assert memory.read_others_on_gpu() == 0
