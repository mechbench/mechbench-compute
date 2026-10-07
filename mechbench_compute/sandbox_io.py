from __future__ import annotations

import os
import stat
import threading
from typing import Self

QUOTA_POLL_S = 0.25


class CappedOutput:
    def __init__(self, limit: int) -> None:
        self.room = limit + 1
        self.parts: list[bytes] = []

    def __call__(self, data: bytes) -> None:
        if self.room > 0 and data:
            kept = data[:self.room]
            self.parts.append(kept)
            self.room -= len(kept)

    def read(self) -> bytes:
        return b"".join(self.parts)


class DiskQuota:
    def __init__(self, root: str | os.PathLike[str], *, max_bytes: int,
                 max_entries: int) -> None:
        self.root = os.fspath(root)
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.tripped = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True,
                                        name="mechbench-sandbox-quota")

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join()
        if not self.tripped:
            self.check()

    def _watch(self) -> None:
        while not self._stop.wait(QUOTA_POLL_S):
            self.check()

    def check(self) -> bool:
        used, entries = measure_tree(self.root, stop_at=self.max_entries)
        if used > self.max_bytes or entries > self.max_entries:
            self.tripped = True
            truncate_tree(self.root)
        return self.tripped


def measure_tree(root: str, *, stop_at: int) -> tuple[int, int]:
    used = entries = 0
    try:
        for _top, dirs, files, dir_fd in os.fwalk(root):
            for name in (*dirs, *files):
                entries += 1
                try:
                    st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISREG(st.st_mode):
                    used += st.st_blocks * 512
            if entries > stop_at:
                break
    except OSError:
        pass
    return used, entries


def truncate_tree(root: str) -> None:
    try:
        for _top, _dirs, files, dir_fd in os.fwalk(root):
            for name in files:
                try:
                    fd = os.open(name, os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                 dir_fd=dir_fd)
                except OSError:
                    continue
                try:
                    if stat.S_ISREG(os.fstat(fd).st_mode):
                        os.ftruncate(fd, 0)
                finally:
                    os.close(fd)
    except OSError:
        pass
