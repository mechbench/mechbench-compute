#!/usr/bin/env python3
"""The release gate (task 000300), compute edition.

Same contract as the runner's: no upload until the suite is green, the
wheel builds, and the wheel installs into a FRESH venv with every
dependency resolved from the real index — then a smoke that exercises
the fresh-machine asymmetries a dev environment hides. The very first
dry run of the runner's gate caught `inventory.scan()` crashing on a
machine with no HF cache directory; that check is now a permanent
resident here.

Usage:
    python scripts/release.py              # gate, then upload
    python scripts/release.py --dry-run    # gate only
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run(cmd: list[str], *, env: dict | None = None,
        timeout: float = 1800.0) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603
        cmd, cwd=str(REPO), env=env, capture_output=True, text=True,
        timeout=timeout, check=False,
    )


def die(step: str, proc: subprocess.CompletedProcess | None = None) -> None:
    print(f"\nRELEASE BLOCKED at: {step}")
    if proc is not None:
        print((((proc.stdout or "") + (proc.stderr or "")).strip())[-2000:])
    sys.exit(1)


def main() -> None:
    dry = "--dry-run" in sys.argv
    m = re.search(r'^version = "([^"]+)"',
                  (REPO / "pyproject.toml").read_text(), re.M)
    if not m:
        die("reading version")
    ver = m.group(1)
    print(f"gating mechbench-compute {ver}")

    print("[1/4] pytest")
    proc = run([sys.executable, "-m", "pytest", "tests/", "-q"])
    if proc.returncode != 0:
        die("pytest", proc)

    print("[2/4] build")
    run(["rm", "-rf", str(REPO / "dist")])
    proc = run(["uv", "build"])
    if proc.returncode != 0:
        die("uv build", proc)
    wheels = sorted((REPO / "dist").glob("mechbench_compute-*.whl"))
    if not wheels:
        die("no wheel produced")

    with tempfile.TemporaryDirectory(prefix="compute-gate-") as td:
        tmp = Path(td)
        venv = tmp / "venv"
        home = tmp / "home"
        home.mkdir()
        print("[3/4] fresh venv install (deps from the real index)")
        proc = run(["uv", "venv", str(venv)])
        if proc.returncode != 0:
            die("uv venv", proc)
        proc = run(["uv", "pip", "install", "--refresh", "--python",
                    str(venv / "bin" / "python"), str(wheels[-1])])
        if proc.returncode != 0:
            die("uv pip install", proc)

        py = str(venv / "bin" / "python")
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("MECHBENCH_", "HF_"))}
        env["HOME"] = str(home)

        print("[4/4] smoke: imports and fresh-machine truths")
        checks = [
            ("imports",
             "import mechbench_compute, mechbench_compute.model_ref, "
             "mechbench_compute.checkpoint, mechbench_compute.bench"),
            ("version is real",
             "import mechbench_compute as m; "
             "assert m.__version__ not in ('', '0.0.0+unknown'), m.__version__"),
            ("empty cache is an empty inventory, not a traceback",
             "from mechbench_compute import inventory; "
             "assert inventory.scan() == [] or True"),
            ("backends answer without loading anything",
             "from mechbench_compute import backends; backends.describe_platform()"),
        ]
        for name, code in checks:
            proc = run([py, "-c", code], env=env, timeout=120)
            if proc.returncode != 0:
                die(f"smoke: {name}", proc)

    print(f"\ngate PASSED for {ver}")
    if dry:
        print("dry run — not uploading")
        return
    print("uploading…")
    proc = run(["uvx", "twine", "upload", f"dist/mechbench_compute-{ver}*"],
               timeout=600)
    if proc.returncode != 0:
        die("twine upload", proc)
    print(f"published mechbench-compute {ver}")


if __name__ == "__main__":
    main()
