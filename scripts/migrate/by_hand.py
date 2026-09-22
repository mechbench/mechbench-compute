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


def edit(rel: str, old: str, new: str, count: int | None = 1) -> None:
    """Replace `old` with `new` in `rel`; `count` occurrences are expected
    (None: at least one, all replaced)."""
    path = (ROOT / rel) if rel.startswith("tests/") else (PKG / rel)
    s = path.read_text()
    found = s.count(old)
    if count is None:
        assert found >= 1, f"{rel}: none of {old[:60]!r}"
    else:
        assert found == count, f"{rel}: expected {count} of {old[:60]!r}, found {found}"
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


def records_tests() -> None:
    """Tests that reached a records operation through a table the
    executor no longer consults for it: the pure-block registry, the
    monoid table, the executor class by module attribute. Each now
    patches the operation's own file, which is where it runs from."""
    # The join tests made records/fill flaky through PURE_BLOCKS.
    edit("tests/test_join_semantics.py", """    from mechbench_compute import blocks as blocks_mod

    real = blocks_mod.PURE_BLOCKS["records/fill"]
    seen = {"ran": []}

    def flaky(inputs, params):
        if params.get("templates", {}).get("user", "").startswith("{x}"):
            raise RuntimeError("this branch died")
        seen["ran"].append(params["templates"]["user"])
        return real(inputs, params)

    monkeypatch.setitem(blocks_mod.PURE_BLOCKS, "records/fill", flaky)""",
         """    from mechbench_compute.ops.records import fill

    real = fill.run
    seen = {"ran": []}

    def flaky(ctx, inputs, params):
        if params.get("templates", {}).get("user", "").startswith("{x}"):
            raise RuntimeError("this branch died")
        seen["ran"].append(params["templates"]["user"])
        return real(ctx, inputs, params)

    monkeypatch.setattr(fill, "run", flaky)""")
    if True:   # two tests carry their own copy of the same patch
        edit("tests/test_join_semantics.py", """        from mechbench_compute import blocks as blocks_mod

        real = blocks_mod.PURE_BLOCKS["records/fill"]
        ran = []

        def flaky(inputs, params):
            user = params.get("templates", {}).get("user", "")
            if user.startswith("{x}"):
                raise RuntimeError("this branch died")
            ran.append(user)
            return real(inputs, params)

        monkeypatch.setitem(blocks_mod.PURE_BLOCKS, "records/fill", flaky)""",
             """        from mechbench_compute.ops.records import fill

        real = fill.run
        ran = []

        def flaky(ctx, inputs, params):
            user = params.get("templates", {}).get("user", "")
            if user.startswith("{x}"):
                raise RuntimeError("this branch died")
            ran.append(user)
            return real(ctx, inputs, params)

        monkeypatch.setattr(fill, "run", flaky)""", count=None)
    edit("tests/test_join_semantics.py", """        from mechbench_compute import blocks as blocks_mod

        real = blocks_mod.PURE_BLOCKS["records/fill"]

        def flaky(inputs, params):
            if params.get("templates", {}).get("user", "").startswith("{x}"):
                raise RuntimeError("this branch died")
            return real(inputs, params)

        monkeypatch.setitem(blocks_mod.PURE_BLOCKS, "records/fill", flaky)""",
         """        from mechbench_compute.ops.records import fill

        real = fill.run

        def flaky(ctx, inputs, params):
            if params.get("templates", {}).get("user", "").startswith("{x}"):
                raise RuntimeError("this branch died")
            return real(ctx, inputs, params)

        monkeypatch.setattr(fill, "run", flaky)""")
    # A broken monoid was planted in the table; the operation's file holds it now.
    edit("tests/test_dataflow_law.py",
         """        monkeypatch.setitem(rd.MONOIDS, "records/total", Bad)""",
         """        from mechbench_compute.ops.records import total

        monkeypatch.setattr(total, "MONOID", Bad)""")
    # The map spawns `type(ctx.executor)`, so the stand-in has to BE the executor.
    edit("tests/test_blocks.py", """        class Child:
            _model = None
            _model_id = None

            def __init__(self, *a, **k):
                pass

            def run(self, spec, **kw):
                return SimpleNamespace(payload={"outputs": {"separation": body_out}})

        from mechbench_compute import ops
        from mechbench_compute.ops.records import map as map_op

        ex = P.ProtocolExecutor()
        monkeypatch.setattr(P, "ProtocolExecutor", Child)
        out = map_op.run(
            ops.Context(executor=ex),""",
         """        class Child:
            _model = None
            _model_id = None
            _on_download = _on_download_bytes = _limiter = _budget = None

            def __init__(self, *a, **k):
                pass

            def run(self, spec, **kw):
                return SimpleNamespace(payload={"outputs": {"separation": body_out}})

        from mechbench_compute import ops
        from mechbench_compute.ops.records import map as map_op

        out = map_op.run(
            ops.Context(executor=Child()),""")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        globals()[name]()
        print(f"applied: {name}")
