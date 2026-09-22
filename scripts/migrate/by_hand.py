"""The edits the mover cannot make, applied after it (docs/OPS_LAYOUT.md).

    python scripts/migrate/by_hand.py records

Each family's edits are a function here, so they survive a reset and
are on the record: what needed a person, and why.
"""
from __future__ import annotations

import ast
import re
import subprocess
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


def inline_into_run(rel: str, wrapper: str) -> None:
    """`run` calls `wrapper(inputs, params)` and does nothing else, and
    `wrapper` is not called from anywhere else: give `run` the body and
    take the wrapper out. A name that only forwards is not a name worth
    fixing (000629)."""
    path = PKG / rel
    lines = path.read_text().split("\n")
    tree = ast.parse("\n".join(lines))
    fn = next(n for n in tree.body if getattr(n, "name", None) == wrapper)
    run = next(n for n in tree.body if getattr(n, "name", None) == "run")
    assert len(run.body) == 1 and ast.unparse(run.body[0]) == \
        f"return {wrapper}(inputs, params)", f"{rel}: run is not a pass-through"
    assert [a.arg for a in fn.args.args] == ["inputs", "params"], rel
    # By line range, so comments and formatting travel with the body;
    # both functions are module-level, so the indent already matches.
    first = fn.body[0].lineno
    while first - 2 >= 0 and lines[first - 2].lstrip().startswith("#"):
        first -= 1
    out = (lines[:run.body[0].lineno - 1]
           + lines[first - 1:fn.end_lineno]
           + lines[run.body[0].end_lineno:fn.lineno - 2]
           + lines[fn.end_lineno:])
    path.write_text("\n".join(out))


def delete(rel: str) -> None:
    subprocess.run(["git", "rm", "-q", "-f", f"mechbench_compute/{rel}"], cwd=ROOT, check=True)


def names() -> None:
    """000629, the rows of renames.tsv marked `(inline)` or `(delete)`.

    Dead indirection is not a naming problem, so it is not renamed. The
    nine `direction/*` wrappers each stood between `run` and the
    mechanism and read one line; `geometry/compare` had two such layers,
    the inner one still carrying the stray `pass` the ops-layout move
    left; and two helpers had no caller left at all once the package
    `__init__.py` that imports them back is discounted."""
    for op, wrapper in (
            ("direction/add", "block_add"),
            ("direction/average", "block_average"),
            ("direction/classify", "block_classify"),
            ("direction/decompose", "block_from_pca"),
            ("direction/fit", "block_from_vectors"),
            ("direction/normalize", "block_normalize"),
            ("direction/orthogonalize", "block_orthogonalize"),
            ("direction/project", "block_project"),
            ("direction/regress", "block_from_regression")):
        rel = f"ops/{op}.py"
        inline_into_run(rel, wrapper)
        repoint(op, rel)

    # geometry/compare: run -> _geometry_similarity -> compare_geometry,
    # and the middle one's whole body was a `pass` and a forward.
    edit("ops/geometry/compare.py",
         "def run(ctx, inputs, params):\n"
         "    return _geometry_similarity(inputs, params)\n"
         "\n"
         "\n"
         "def _geometry_similarity(inputs, params):\n"
         "    pass  # its imports now live in this file\n"
         "\n"
         "    return compare_geometry(inputs, params)\n",
         "def run(ctx, inputs, params):\n"
         "    return compare_geometry(inputs, params)\n")
    repoint("geometry/compare", "ops/geometry/compare.py")

    # Two helpers nothing calls: the only mention of each was the
    # package __init__.py importing it back.
    edit("blocks/__init__.py",
         "from mechbench_compute.blocks.transcript_mod import _transcript_mod  # noqa: F401\n", "")
    delete("blocks/transcript_mod.py")
    edit("reduce/__init__.py",
         "from mechbench_compute.reduce.block_of import _block_of  # noqa: F401\n", "")
    delete("reduce/block_of.py")
    edit("tests/test_block_params.py",
         "    # `_block_of` is the closure that reads the records port for all three.\n", "")

    # direction/add's wrapper carried the only annotations in the file.
    edit("ops/direction/add.py",
         "from collections.abc import Mapping\nfrom typing import Any\n\n", "")

    # Prose naming a renamed thing: a docstring the renamer cannot read.
    edit("blocks/expand_cells.py",
         "a grid's cells expanded (`grid_rows`)", "a grid's cells expanded (`expand_grid`)")
    edit("directions/__init__.py",
         "`vocab_projection`, which needs a model's unembedding.",
         "`unembed_direction`, which needs a model's unembedding.")
    edit("ops/direction/regress.py",
         '`from_vectors` answers "which way does THIS group lie from THAT',
         '`fit_mean_difference` answers "which way does THIS group lie from THAT')
    edit("ops/eval/benchmark.py", "paired_delta for base-vs-adapter deltas.",
         "subtract_baseline for base-vs-adapter deltas.")
    edit("ops/records/contrast.py", "A field read the way `_group_key` reads one",
         "A field read the way `read_group_key` reads one")

    # tokenizer_stats.py is what is left of a module the ops-layout move
    # emptied: a docstring and an `__all__` naming two functions that
    # live in ops/text/tokenize.py now. Say their new names.
    edit("tokenizer_stats.py",
         '__all__: Sequence[str] = ("block", "tokenizer_stats")',
         '__all__: Sequence[str] = ("measure_model_tokenizer", "measure_tokenizer")')


if __name__ == "__main__":
    for name in sys.argv[1:]:
        globals()[name]()
        print(f"applied: {name}")
