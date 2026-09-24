from __future__ import annotations

import contextlib
import io
import pathlib
import runpy

import pytest

HERE = pathlib.Path(__file__).resolve().parent.parent
SIBLINGS = HERE.parent

COPIES = {
    "mechbench-models/src/lexicon.generated.ts": "dump_lexicon_ts.py",
    "mechbench-models/src/kinds.generated.ts": "dump_kinds_ts.py",
}


def _body(text: str) -> str:
    return text.split("\n", 1)[1]


@pytest.mark.parametrize(("copy", "script"), COPIES.items())
def test_the_ui_copy_is_current(copy: str, script: str) -> None:
    target = SIBLINGS / copy
    if not target.exists():
        pytest.skip(f"no {copy.split('/')[0]} checkout beside this one")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_path(str(HERE / "scripts" / script))["main"]()
    assert _body(target.read_text()) == _body(buf.getvalue()), (
        f"{copy} is stale: run\n"
        f"  python scripts/{script} > ../{copy}")
