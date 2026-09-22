"""No comment or docstring narrates the code's history (docs/COMMENTS.md).

This repository is read by agents. A comment that states a constraint —
an ordering the code depends on, why a threshold has its value, what a
caller may assume — is the reason to read it. A comment that narrates
how the code came to be costs the same attention and answers a question
nobody asked, and its references leave the repository: a task id points
into a private tracker, a date and a version number point at a tree that
is no longer here.

So three spellings are refused wherever a comment or a docstring can
carry them: a six-digit task id, an ISO date, and a three-part version
number.

Only comments and docstrings are in scope. Every other string is DATA —
an op's declared prose, a kind's name, an error message, a fixture, a
version this code compares against — and is left alone. `scripts/` is
excluded: it is not read the way the package and the suite are.

The property behind the spelling is the one that matters: when a comment
records a fixed bug, keep the invariant it enforces and drop the
incident. "`Y` reads what `X` writes, so `X` runs first" is signal; the
number of the task that found it crashing is not.
"""

from __future__ import annotations

import ast
import io
import pathlib
import re
import tokenize
import warnings

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: The trees whose comments are read as documentation.
SCANNED = ("mechbench_compute", "tests")

#: What a comment may never carry, and what to write instead.
FORBIDDEN = (
    (re.compile(r"\b000\d{3}\b"), "a task id — say what the code must do"),
    (re.compile(r"20\d\d-\d\d-\d\d"), "a date — say what is true now"),
    (re.compile(r"\b\d+\.\d+\.\d+\b"), "a version — say what the code does"),
)


def _python_files() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for tree in SCANNED:
        for path in sorted((ROOT / tree).rglob("*.py")):
            if "__pycache__" not in path.parts:
                out.append(path)
    return out


def _prose(path: pathlib.Path) -> list[tuple[int, str]]:
    """(line, text) for every comment token and every docstring."""
    src = path.read_text(encoding="utf-8")
    found: list[tuple[int, str]] = []
    for token in tokenize.generate_tokens(io.StringIO(src).readline):
        if token.type == tokenize.COMMENT:
            found.append((token.start[0], token.string))
    with warnings.catch_warnings():
        # A file's own string literals may carry escapes the compiler
        # warns about; this gate reads prose, not literals.
        warnings.simplefilter("ignore", SyntaxWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            doc = body[0].value
            # One entry per line, so the report names the line to edit.
            for offset, line in enumerate(doc.value.splitlines()):
                found.append((doc.lineno + offset, line))
    return found


def test_no_comment_carries_a_task_id_a_date_or_a_version() -> None:
    offences: list[str] = []
    for path in _python_files():
        rel = path.relative_to(ROOT)
        for line, text in _prose(path):
            for pattern, why in FORBIDDEN:
                hit = pattern.search(text)
                if hit:
                    offences.append(
                        f"{rel}:{line}: {hit.group(0)!r} is {why} — "
                        f"{text.strip()[:70]}")
    assert not offences, (
        f"{len(offences)} comment(s) narrate history rather than state a "
        f"constraint (docs/COMMENTS.md):\n" + "\n".join(sorted(offences)))
