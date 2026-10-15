from __future__ import annotations

import subprocess
import sys
import types

import pytest

from mechbench_compute.torch_backend import memory

torch = pytest.importorskip("torch")

ORIGINAL_READ_OTHERS = memory.read_others
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
    monkeypatch.setattr(memory, "read_others",
                        lambda own=None, index=0: {"bytes": others_gb * GB, "source": "nvidia-smi",
                                                   "reason": None})
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
    monkeypatch.setattr(memory, "read_integrated_breakdown", lambda index: None)
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


NA_ROWS = "100, [N/A]\n200, [N/A]\n"


def smi_prints(monkeypatch, out):
    monkeypatch.setattr(memory.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout=out))


def fake_nvml(monkeypatch, compute, graphics=(), *, fails=None):
    calls = []

    def init():
        calls.append("init")
        if fails:
            raise RuntimeError(fails)

    nvml = types.SimpleNamespace(
        nvmlInit=init,
        nvmlDeviceGetHandleByIndex=lambda index: calls.append(("handle", index)) or "h",
        nvmlDeviceGetComputeRunningProcesses=lambda h: [
            types.SimpleNamespace(pid=p, usedGpuMemory=u) for p, u in compute],
        nvmlDeviceGetGraphicsRunningProcesses=lambda h: [
            types.SimpleNamespace(pid=p, usedGpuMemory=u) for p, u in graphics])
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    return calls


def test_nvidia_smi_answering_names_itself(monkeypatch):
    smi_prints(monkeypatch, "100, 997\n200, 5314\n")
    fake_nvml(monkeypatch, [(200, 1)])
    assert memory.read_others(own=100) == {"bytes": 5314 * 2**20, "source": "nvidia-smi",
                                          "reason": None}


def test_no_process_on_the_gpu_is_an_answer_of_nothing(monkeypatch):
    smi_prints(monkeypatch, "")
    fake_nvml(monkeypatch, [(200, 7 * GB)])
    assert memory.read_others(own=100)["source"] == "nvidia-smi"
    assert memory.read_others(own=100)["bytes"] == 0


def test_where_nvidia_smi_says_na_nvml_answers(monkeypatch):
    smi_prints(monkeypatch, NA_ROWS)
    calls = fake_nvml(monkeypatch, [(100, 3 * GB), (200, 5 * GB)], [(200, 5 * GB), (300, 1 * GB)])
    got = memory.read_others(own=100, index=0)
    assert got == {"bytes": 6 * GB, "source": "nvml", "reason": None}
    assert ("handle", 0) in calls
    assert memory.read_others_on_gpu(own=100) == 6 * GB


def test_where_both_say_na_nothing_is_counted_and_the_reason_is_kept(monkeypatch):
    smi_prints(monkeypatch, NA_ROWS)
    fake_nvml(monkeypatch, [(200, None), (300, 2**64 - 1)])
    got = memory.read_others(own=100)
    assert got["bytes"] is None and got["source"] is None
    assert "N/A" in got["reason"] and "not available" in got["reason"]
    assert memory.read_others_on_gpu(own=100) == 0


def test_without_pynvml_the_reason_says_so(monkeypatch):
    smi_prints(monkeypatch, NA_ROWS)
    monkeypatch.setitem(sys.modules, "pynvml", None)
    got = memory.read_others(own=100)
    assert got["source"] is None and "pynvml is not installed" in got["reason"]


def test_an_nvml_that_refuses_is_a_reason_not_a_crash(monkeypatch):
    smi_prints(monkeypatch, NA_ROWS)
    fake_nvml(monkeypatch, [], fails="NVML_ERROR_DRIVER_NOT_LOADED")
    got = memory.read_others(own=100)
    assert got["bytes"] is None and "NVML_ERROR_DRIVER_NOT_LOADED" in got["reason"]


def test_the_budget_breakdown_names_the_source_of_others(monkeypatch, tmp_path):
    fake_gpu(monkeypatch, tmp_path, True, host_used_gb=4)
    monkeypatch.setattr(memory, "read_others", ORIGINAL_READ_OTHERS)
    smi_prints(monkeypatch, NA_ROWS)
    fake_nvml(monkeypatch, [(200, 6 * GB)])
    got = memory.read_integrated_breakdown(0)
    assert got == {"total": 120 * GB, "host_used": 4 * GB, "others": 6 * GB,
                   "others_source": "nvml", "others_reason": None,
                   "budget": (120 - 4 - 6) * GB}
