from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

LIMIT_WORDS = {
    "fuel": "ran out of fuel (the computation was too long)",
    "memory_mb": "ran out of memory",
    "wall_seconds": "ran past its time",
    "output_bytes": "wrote more output than the cap",
}


def run_guest_json(op: str, guest: str, argv: Sequence[str], request: Any, *,
                   memory_mb: int, seconds: float, output_mb: int) -> list[Any]:
    from mechbench_compute import sandbox
    from mechbench_compute import snapshots as fs

    limits = sandbox.Limits(memory_mb=int(memory_mb), wall_seconds=float(seconds),
                            output_bytes=int(output_mb) * 1024 * 1024)
    try:
        stdin = json.dumps(request, allow_nan=False, ensure_ascii=False)
    except ValueError as e:
        raise ValueError(f"{op}: the input is not JSON ({e})") from None
    try:
        got = sandbox.run(fs.EMPTY, list(argv), guest=guest, strict=True, stdin=stdin, limits=limits)
    except sandbox.SandboxError as e:
        raise ValueError(f"{op}: the sandbox could not run: {e}") from None
    if got.limit:
        raise ValueError(f"{op}: the code {LIMIT_WORDS.get(got.limit, got.limit)} ({got.limit})")
    if got.truncated:
        raise ValueError(f"{op}: the code wrote more output than the cap ({', '.join(got.truncated)})")
    if got.exit_code != 0:
        raise ValueError(f"{op}: the code failed (exit {got.exit_code}): {got.stderr.strip()[-2000:]}")
    values = []
    for line in got.stdout.splitlines():
        if line.strip():
            try:
                values.append(json.loads(line))
            except ValueError:
                raise ValueError(f"{op}: the code wrote something that is not JSON: {line[:200]!r}") from None
    return values
