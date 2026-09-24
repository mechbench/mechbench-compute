from __future__ import annotations

import ast
import inspect
import io
import pathlib
import re
import subprocess
import tokenize
import warnings

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

SCANNED = ("mechbench_compute/", "tests/", "scripts/")

DIRECTIVE = re.compile(
    r"# (?:noqa(?:: [A-Z]+[0-9]+(?:, ?[A-Z]+[0-9]+)*)?"
    r"|type: ignore(?:\[[\w\-, ]+\])?"
    r"|pragma: no cover"
    r"|fmt: (?:off|on|skip))")

EXTERNAL = re.compile(r"# external: \S[^—]* — \S.*")

READ_AT_RUN_TIME = {
    "mechbench_compute/bench.py":
        "the docs site's client reference reads inspect.getdoc of each public function",
    "mechbench_compute/bench_items.py":
        "bench re-exports these functions, so the client reference reads them too",
}

TASK_ID = re.compile(r"(?<![0-9])000[0-9]{3}(?![0-9])")

NOT_SCANNED_FOR_IDS = {"CHANGELOG.md"}


def _list_tracked() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return out.split()


def _list_python() -> list[str]:
    return [p for p in _list_tracked()
            if p.endswith(".py") and p.startswith(SCANNED)]


def _parse(src: str) -> ast.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ast.parse(src)


def _is_read_at_run_time(rel: str, tree: ast.Module, owner: ast.AST) -> bool:
    return (rel in READ_AT_RUN_TIME and isinstance(owner, ast.FunctionDef)
            and owner in tree.body and not owner.name.startswith("_"))


def test_no_comment_but_a_directive_or_an_external_fact() -> None:
    offences = []
    for rel in _list_python():
        src = (ROOT / rel).read_text(encoding="utf-8")
        for token in tokenize.generate_tokens(io.StringIO(src).readline):
            if token.type != tokenize.COMMENT:
                continue
            text = token.string
            if token.start[0] == 1 and text.startswith("#!"):
                continue
            if DIRECTIVE.fullmatch(text) or EXTERNAL.fullmatch(text):
                continue
            offences.append(f"{rel}:{token.start[0]}: {text[:80]}")
    assert not offences, (
        f"{len(offences)} comment(s) that are neither a directive nor "
        f"`# external: <the outside thing> — <the fact>`. Say it in the code, "
        f"hold it with a test, or delete it (docs/COMMENTS.md):\n"
        + "\n".join(offences))


def test_no_docstring_but_those_read_at_run_time() -> None:
    offences = []
    for rel in _list_python():
        tree = _parse((ROOT / rel).read_text(encoding="utf-8"))
        for owner in ast.walk(tree):
            if not isinstance(owner, (ast.Module, ast.ClassDef,
                                      ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if ast.get_docstring(owner, clean=False) is None:
                continue
            if _is_read_at_run_time(rel, tree, owner):
                continue
            offences.append(f"{rel}:{owner.body[0].lineno}: "
                            f"{getattr(owner, 'name', '<module>')}")
    assert not offences, (
        f"{len(offences)} docstring(s) nothing reads at run time. Delete them, "
        f"or register the reader in READ_AT_RUN_TIME (docs/COMMENTS.md):\n"
        + "\n".join(offences))


def test_every_docstring_read_at_run_time_is_there() -> None:
    from mechbench_compute import bench

    missing = [name for name, fn in inspect.getmembers(bench, inspect.isfunction)
               if not name.startswith("_")
               and fn.__module__.startswith("mechbench_compute.bench")
               and not inspect.getdoc(fn)]
    assert not missing, (
        f"the client reference renders these with no summary: {missing}")


def test_no_task_id_anywhere() -> None:
    offences = []
    for rel in _list_tracked():
        if rel in NOT_SCANNED_FOR_IDS:
            continue
        try:
            text = (ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            hit = TASK_ID.search(line)
            if hit:
                offences.append(f"{rel}:{number}: {hit.group(0)} — {line.strip()[:70]}")
    assert not offences, (
        f"{len(offences)} line(s) name a task in a tracker this repository's "
        f"readers cannot open. Say what is true instead:\n" + "\n".join(offences))
