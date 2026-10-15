from __future__ import annotations

import ctypes
import os
import sys

SYSTEM_REPORTS_ENV = "MECHBENCH_SYSTEM_CRASH_REPORTS"

EXC_MASK_CRASH = 1 << 10
EXC_MASK_CORPSE_NOTIFY = 1 << 13
EXCEPTION_DEFAULT = 1


def keep_from_crash_reporter() -> bool:
    if sys.platform != "darwin" or os.environ.get(SYSTEM_REPORTS_ENV, "") not in ("", "0"):
        return False
    try:
        libc = ctypes.CDLL(None)
        task = ctypes.c_uint.in_dll(libc, "mach_task_self_").value
        setter = libc.task_set_exception_ports
        setter.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_int, ctypes.c_int]
        setter.restype = ctypes.c_int
        return setter(task, EXC_MASK_CRASH | EXC_MASK_CORPSE_NOTIFY, 0, EXCEPTION_DEFAULT, 0) == 0
    except (OSError, AttributeError, ValueError):
        return False
