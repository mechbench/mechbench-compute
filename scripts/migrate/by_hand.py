"""The edits the mover cannot make, applied after it (docs/OPS_LAYOUT.md).

    python scripts/migrate/by_hand.py records

Each family's edits are a function here, so they survive a reset and
are on the record: what needed a person, and why.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from plan import PKG, ROOT  # noqa: E402


def edit(rel: str, old: str, new: str, count: int = 1) -> None:
    path = (ROOT / rel) if rel.startswith("tests/") else (PKG / rel)
    s = path.read_text()
    assert s.count(old) == count, f"{rel}: expected {count} of {old[:60]!r}, found {s.count(old)}"
    path.write_text(s.replace(old, new))


def repoint(op: str, rel: str) -> None:
    """The params gate reads an operation's code where SITES says it is."""
    path = ROOT / "tests" / "test_block_params.py"
    s = path.read_text()
    fns = [n.name for n in ast.parse((PKG / rel).read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    fns.sort(key=lambda n: n != "run")
    sites = ",\n        ".join(f'("{rel}", "{fn}")' for fn in fns)
    s, n = re.subn(rf'^    "{re.escape(op)}": \[.*?\],\n', f'    "{op}": [\n        {sites}],\n', s, count=1, flags=re.M | re.S)
    assert n == 1, op
    path.write_text(s)


def records() -> None:
    # records/plot had no registry entry: the dispatcher called it with
    # the label its input arrived from. The context carries that.
    edit("ops/records/plot.py", "\n\ndef _layer_axis_from(", '''

def run(ctx, inputs, params):
    # A viz references its upstream by LABEL when the executor knows it
    # (lineage-true, renders live).
    return viz_spec(
        inputs.get("records"), params,
        source_label=ctx.input_paths.get("records") or None)


def _layer_axis_from(''')
    repoint("records/plot", "ops/records/plot.py")

    # records/total, rank and bin were registered through a factory,
    # `_block_of(name)`, whose only purpose was to look the monoid up by
    # name at call time. With MONOID in the file, run says what it does.
    for name in ("total", "rank", "bin"):
        edit(f"ops/records/{name}.py", f'''def run(ctx, inputs, params):
    return _block_of("records/{name}")(inputs, params)
''', '''def run(ctx, inputs, params):
    from mechbench_compute.lexicon import kinds as K

    m = MONOID()
    if hasattr(m, "bind"):
        m.bind(params or {})
    raw = inputs.get("records")
    recs = K.items_of(raw if raw is not None else [])
    return m.finalize(m.partial(recs, params), params)
''')
        edit(f"ops/records/{name}.py", "from mechbench_compute.reduce.block_of import _block_of\n", "")

    # A test called the executor's map method by name.
    edit("tests/test_blocks.py", """        ex = P.ProtocolExecutor()
        monkeypatch.setattr(P, "ProtocolExecutor", Child)
        out = ex._block_map(
            {"records": K.collection("records/record", [{"id": "l0", "layer": 0}])},""",
         """        from mechbench_compute import ops
        from mechbench_compute.ops.records import map as map_op

        ex = P.ProtocolExecutor()
        monkeypatch.setattr(P, "ProtocolExecutor", Child)
        out = map_op.run(
            ops.Context(executor=ex),
            {"records": K.collection("records/record", [{"id": "l0", "layer": 0}])},""")


if __name__ == "__main__":
    globals()[sys.argv[1]]()
    print(f"applied: {sys.argv[1]}")
