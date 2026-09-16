"""The UI's copies of the lexicon are the lexicon.

`mechbench-ui` commits two generated modules: `lexicon.generated.ts`
(every operation's ports and parameter declarations, which the composer
builds each node's editor from) and `kindAliases.generated.ts` (every
kind and retired spelling, which it reads stored objects with). A
declaration changed here and not regenerated there is an editor offering
a choice the executor refuses, or a stored object the UI reads under a
different kind than the executor does — and the kind table had drifted
two releases before this test existed. So when the UI checkout sits
beside this one, each copy must equal what its script writes now. Only
the header line, which carries the build's version string, may differ.
"""

from __future__ import annotations

import contextlib
import io
import pathlib
import runpy

import pytest

HERE = pathlib.Path(__file__).resolve().parent.parent
UI_LIB = HERE.parent / "mechbench-ui" / "src" / "lib"

COPIES = {
    "lexicon.generated.ts": "dump_lexicon_ts.py",
    "kindAliases.generated.ts": "dump_kinds_ts.py",
}


def _body(text: str) -> str:
    return text.split("\n", 1)[1]


@pytest.mark.parametrize(("copy", "script"), COPIES.items())
def test_the_ui_copy_is_current(copy: str, script: str) -> None:
    target = UI_LIB / copy
    if not target.exists():
        pytest.skip("no mechbench-ui checkout beside this one")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_path(str(HERE / "scripts" / script))["main"]()
    assert _body(target.read_text()) == _body(buf.getvalue()), (
        f"mechbench-ui's {copy} is stale: run\n"
        f"  python scripts/{script} > ../mechbench-ui/src/lib/{copy}")
