from __future__ import annotations

import pathlib
import signal
import subprocess
import sys
import time

import pytest

from tests.crash_reporter import keep_from_crash_reporter

REPORTS = pathlib.Path.home() / "Library" / "Logs" / "DiagnosticReports"
ROOT = pathlib.Path(__file__).resolve().parent.parent
KEPT_THEN_ABORT = ("import os\n"
                   "from tests.crash_reporter import keep_from_crash_reporter\n"
                   "assert keep_from_crash_reporter()\n"
                   "os.abort()\n")


def reports_naming(pid: int) -> list[pathlib.Path]:
    found = []
    for path in REPORTS.glob("*.ips"):
        try:
            if f'"pid" : {pid},' in path.read_text(errors="replace"):
                found.append(path)
        except OSError:
            continue
    return found


@pytest.mark.skipif(sys.platform != "darwin", reason="ReportCrash is macOS's")
def test_a_child_kept_from_the_crash_reporter_aborts_without_a_report_or_a_dialog():
    proc = subprocess.Popen([sys.executable, "-c", KEPT_THEN_ABORT], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert proc.wait(timeout=60) == -signal.SIGABRT
    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline:
        assert not reports_naming(proc.pid)
        time.sleep(0.5)


def test_the_switch_is_off_where_there_is_no_crash_reporter_or_the_developer_asks(monkeypatch):
    monkeypatch.setenv("MECHBENCH_SYSTEM_CRASH_REPORTS", "1")
    assert keep_from_crash_reporter() is False
    monkeypatch.delenv("MECHBENCH_SYSTEM_CRASH_REPORTS")
    monkeypatch.setattr(sys, "platform", "linux")
    assert keep_from_crash_reporter() is False
