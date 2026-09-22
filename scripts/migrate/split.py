"""Split the protocol executor into one file per topic (task 000632).

    python scripts/migrate/split.py --dry     # what it would write
    python scripts/migrate/split.py           # write it, then verify

`mechbench_compute/protocol/__init__.py` held `ProtocolExecutor` and
every module-level helper it used: 1,729 lines, and the only way to find
out how a node reaches an operation's `run()` was to read all of them.

Each topic becomes a file. An executor method moves into a MIXIN named
for its topic, and `__init__.py` composes `class
ProtocolExecutor(<topics...>)` from them, so every call site
(`self._run_op(...)`) and every method body is unchanged — the move is
text, not a rewrite. A module-level helper becomes a file named for
itself, the arrangement docs/OPS_LAYOUT.md already describes for the
definitions operations share, and which `protocol/` already used for
`serialize_params` and `read_tokenizer_id`.

Text-preserving, like scripts/migrate/move.py: a definition is lifted by
line range, so its comments and formatting travel with it byte for byte,
and nothing inside a definition is rewritten. `verify()` proves it: for
every definition moved, the `ast.dump` where it landed equals the
`ast.dump` where it came from, once package-import plumbing is stripped.
"""
from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "mechbench_compute"
#: The module as it stood before the split, read out of git rather than
#: off disk, so this stays re-runnable after it has written the files.
BASE = "adc88f8"
SOURCE = "mechbench_compute/protocol/__init__.py"

HEADER = "from __future__ import annotations\n"


def before_text() -> str:
    return subprocess.run(["git", "show", f"{BASE}:{SOURCE}"], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout


class Piece:
    """One definition, lifted from the source by line range."""

    def __init__(self, name: str, start: int, end: int) -> None:
        self.name, self.start, self.end = name, start, end

    def text(self, lines: list[str]) -> str:
        return "".join(lines[self.start - 1:self.end]).rstrip("\n") + "\n"


#: Every definition the executor's module held, by the line range that
#: carries its own comments with it. The ranges are read once, from the
#: file as it stood at the head of this task.
PIECES = {
    "__init__": (47, 81),
    "_model_loaded": (83, 130),
    "run": (132, 158),
    "model_ref": (161, 166),
    "_run_layer_ablation": (168, 212),
    "_legacy_decision_distribution": (215, 249),
    "_run_pipeline": (251, 1131),
    "_tool_block_runner": (1134, 1152),
    "_Memo": (1155, 1157),
    "_open_memo": (1159, 1204),
    "_close_memo": (1206, 1230),
    "_block_chat_local": (1232, 1241),
    "_run_remote_wave": (1244, 1314),
    "_dispatch_remote": (1316, 1323),
    "_run_op": (1326, 1339),
    "_run_model_block": (1341, 1368),
    "_adapter_fused": (1370, 1411),
    "_materialize_checkpoint": (1414, 1460),
    "node_summary": (1463, 1487),
    "_spend_total": (1490, 1508),
    "MISSING_POLICIES": (1511, 1512),
    "REMOTE_BLOCKS": (1514, 1521),
    "_carry_arch": (1524, 1537),
    "MAX_PARALLEL_NODES": (1539, 1543),
    "_is_remote": (1546, 1554),
    "_missing_policy": (1557, 1569),
    "_MissingUpstream": (1572, 1588),
    "_ordered_edges": (1591, 1600),
    "_preflight": (1603, 1710),
    "_last_logp": (1713, 1717),
    "canonical_json": (1720, 1729),
}

#: The topics, each a file under `protocol/`. `mixin` names the class the
#: executor composes from; a file with no mixin holds module-level
#: definitions only. `holds` is in the order they are written out.
TOPICS: list[dict] = [
    {
        "file": "pipeline.py",
        "mixin": "Pipeline",
        "doc": '''"""Walking the graph: the one method that runs a protocol.

`_run_pipeline` takes a spec's graph, puts its nodes in topological
order, and for each one resolves its params, gathers its inputs off its
in-edges, decides whether a previous attempt's work can be reused, sends
it to an operation, hashes what came back, emits it, and accounts for
what it cost. The manifest at the end is the record of all of that.

The dispatch itself — which of an operation, a remote wave or a pure
block a node goes to — is the `try` near the end of the node loop, and
what it calls is `_run_op` in protocol/dispatch.py.
"""''',
        "imports": [
            "from collections.abc import Mapping",
            "from datetime import UTC",
            "from typing import Any",
            "",
            "from mechbench_compute import lexicon, ops",
            "from mechbench_compute.protocol.carry_arch import _carry_arch",
            "from mechbench_compute.protocol.is_remote import _is_remote",
            "from mechbench_compute.protocol.missing_policy import _missing_policy",
            "from mechbench_compute.protocol.missing_upstream import _MissingUpstream",
            "from mechbench_compute.protocol.node_summary import node_summary",
            "from mechbench_compute.protocol.ordered_edges import _ordered_edges",
            "from mechbench_compute.protocol.preflight import _preflight",
            "from mechbench_compute.protocol.protocol_spec import ProtocolSpec",
            "from mechbench_compute.protocol.serialize_params import serialize_params",
            "from mechbench_compute.protocol.spend_total import _spend_total",
        ],
        "holds": ["_run_pipeline"],
    },
    {
        "file": "dispatch.py",
        "mixin": "Dispatch",
        "doc": '''"""How a node reaches an operation\'s `run()`.

One hop: the graph walk in protocol/pipeline.py has an operation\'s name,
its inputs and its params, and hands them here. `_run_op` finds the
operation\'s module (docs/OPS_LAYOUT.md: one operation, one file), lends
it a `Context` built from whatever the caller passed, and calls its
`run(ctx, inputs, params)`. An operation that declares weights to fuse an
adapter onto goes through the model-block wrapper on its way.
"""''',
        "imports": ["from mechbench_compute import ops"],
        "holds": ["_run_op"],
    },
    {
        "file": "model.py",
        "mixin": "ModelLoading",
        "doc": '''"""Weights: loading them, and fusing adapters around a block.

One model is resident per executor and swapping ids reloads. A model
reference names a base — a hub repo, or a checkpoint on the bench, which
is materialized into a local directory here — and a stack of adapters,
which fuse around the block that runs and are restored afterwards.
"""''',
        "imports": ["from mechbench_compute import Model"],
        "holds": ["_model_loaded", "model_ref", "_run_model_block",
                  "_adapter_fused", "_materialize_checkpoint"],
    },
    {
        "file": "remote.py",
        "mixin": "Remote",
        "doc": '''"""Nodes whose work happens on somebody else\'s machine.

Two remote nodes that do not depend on each other have no reason to wait
for each other: the time is latency. `_run_remote_wave` runs the node the
graph walk asked for and every remote node already ready beside it, and
hands every result back for the caller to account for in topological
order.
"""''',
        "imports": [
            "from typing import Any",
            "",
            "from mechbench_compute import lexicon",
            "from mechbench_compute.protocol.is_remote import _is_remote",
            "from mechbench_compute.protocol.ordered_edges import _ordered_edges",
        ],
        "before": ["MAX_PARALLEL_NODES"],
        "holds": ["_run_remote_wave", "_dispatch_remote"],
    },
    {
        "file": "tools.py",
        "mixin": "Tools",
        "doc": '''"""What a model may call mid-turn.

The toolbox runs pure blocks itself. The handlers that need an executor —
a model block, a recording fetch off the bench — are this, which is what
makes `logits/read` available AS A TOOL: a model that can consult another
model, or the bench, in the middle of answering.
"""''',
        "imports": [],
        "holds": ["_tool_block_runner"],
    },
    {
        "file": "chat.py",
        "mixin": "Chat",
        "doc": '''"""The local half of `text/chat`.

`text/chat` serves two tiers from one operation: a provider\'s endpoint,
which the remote path dispatches, and local weights, which are this. The
operation asks its executor for it rather than loading a model itself.
"""''',
        "imports": [],
        "holds": ["_block_chat_local"],
    },
    {
        "file": "memo.py",
        "mixin": "Memo",
        "doc": '''"""A node\'s memo of the remote calls it made.

`cache: "<bench label>"` on a node opens a cassette, so a re-run pays a
provider only for what it has not asked before. The label is explicit, or
derived from the protocol\'s identity and the node\'s id — never from the
node fingerprint, which every compute release changes, where the memo is
worth keeping across them.
"""''',
        "imports": ["from collections import namedtuple"],
        "holds": ["_Memo", "_open_memo", "_close_memo"],
    },
    {
        "file": "legacy_kinds.py",
        "mixin": "LegacyKinds",
        "doc": '''"""The two spec kinds that came before the graph.

`layer_ablation` is the v0 protocol, still run as itself;
`decision_distribution` is a thin shim over the decision-read operation,
kept so a spec written for it still answers. Everything since is a
`pipeline` spec, in protocol/pipeline.py.
"""''',
        "imports": [
            "from datetime import UTC",
            "from typing import Any",
            "",
            "import mlx.core as mx",
            "import numpy as np",
            "from mechbench_schema import (",
            "    AblationPrompt,",
            "    LayerAblationPayload,",
            "    LayerAggregates,",
            ")",
            "",
            "from mechbench_compute import GLOBAL_LAYERS, N_LAYERS, Ablate, lexicon",
            "from mechbench_compute.protocol.protocol_spec import ProtocolSpec",
        ],
        "holds": ["_run_layer_ablation", "_legacy_decision_distribution"],
        "after": ["_last_logp"],
    },
    {
        "file": "preflight.py",
        "mixin": None,
        "doc": '''"""Everything about a graph that is decidable before it runs."""''',
        "imports": ["from mechbench_compute import lexicon"],
        "holds": ["MISSING_POLICIES", "_preflight"],
    },
    {
        "file": "ordered_edges.py",
        "mixin": None,
        "doc": '''"""The edges into a node, in the one order the platform reads them."""''',
        "imports": ["from typing import Any"],
        "holds": ["_ordered_edges"],
    },
    {
        "file": "is_remote.py",
        "mixin": None,
        "doc": '''"""Whether a node\'s work happens on somebody else\'s machine."""''',
        "imports": ["from collections.abc import Mapping", "from typing import Any"],
        "holds": ["REMOTE_BLOCKS", "_is_remote"],
    },
    {
        "file": "missing_policy.py",
        "mixin": None,
        "doc": '''"""What to do about an input its upstream never produced."""''',
        "imports": [],
        "holds": ["_missing_policy"],
    },
    {
        "file": "missing_upstream.py",
        "mixin": None,
        "doc": '''"""The refusal a `fail` port raises when its upstream produced nothing."""''',
        "imports": ["from collections.abc import Mapping", "from typing import Any"],
        "holds": ["_MissingUpstream"],
    },
    {
        "file": "node_summary.py",
        "mixin": None,
        "doc": '''"""What a node produced, small enough to read beside the node."""''',
        "imports": [
            "from collections.abc import Mapping",
            "from typing import Any",
            "",
            "from mechbench_compute import lexicon",
        ],
        "holds": ["node_summary"],
    },
    {
        "file": "spend_total.py",
        "mixin": None,
        "doc": '''"""The run\'s bill: total, per provider, per node."""''',
        "imports": ["from typing import Any"],
        "holds": ["_spend_total"],
    },
    {
        "file": "carry_arch.py",
        "mixin": None,
        "doc": '''"""A model\'s depth landmarks, carried onto a result that has none."""''',
        "imports": ["from collections.abc import Mapping", "from typing import Any"],
        "holds": ["_carry_arch"],
    },
]

#: What stays in `protocol/__init__.py`: the composed class with the
#: executor's construction and its one entry point, and the package's own
#: public utility.
KEPT = ["__init__", "run", "canonical_json"]

INIT_DOC = '''"""Protocol execution: turn an experiment spec into results.

Lives in the compute layer, beside the primitives it drives, because it
is pure computation — it takes a spec, runs operations against a loaded
model, and returns typed payloads. It reaches no network and knows
nothing about jobs, queues or credentials; `mechbench-runner` owns all
of that and calls in here once it has claimed something to do.

The model is loaded once per process and reused. Load cost is
significant (minutes on first call, seconds on cached weights), so
callers should hold the executor for the process's lifetime rather
than instantiating per request.

`ProtocolExecutor` is composed from one mixin per topic, each a file of
its own (task 000632), because at 1,729 lines this module was not
something an agent could read to answer one question about it:

    pipeline.py       walking the graph: order, resume, emission
    dispatch.py       how a node reaches an operation's run()
    model.py          loading weights, fusing adapters
    remote.py         the nodes a provider answers, run in a wave
    tools.py          what a model may call mid-turn
    chat.py           the local half of text/chat
    memo.py           a node's memo of the remote calls it made
    legacy_kinds.py   the two spec kinds that came before the graph

The methods are unchanged by that move, and so are their call sites: a
mixin is how a method keeps its `self`.
"""'''


def load() -> tuple[list[str], dict[str, str]]:
    lines = before_text().splitlines(keepends=True)
    text = {name: Piece(name, *span).text(lines) for name, span in PIECES.items()}
    return lines, text


def gap(chunk: str) -> list[str]:
    """Two blank lines before a definition, one before the comment a
    constant carries — which is what the import block wants above it."""
    return [""] if chunk.lstrip().startswith("#") else ["", ""]


def render(topic: dict, text: dict[str, str]) -> str:
    out = [topic["doc"], "", HEADER.rstrip("\n")]
    if topic["imports"]:
        out += ["", *topic["imports"]]
    for name in topic.get("before", []):
        out += gap(text[name]) + [text[name].rstrip("\n")]
    if topic["mixin"]:
        mixin = topic["mixin"]
        out += ["", "", f"class {mixin}:",
                f'    """{mixin}: see this module\'s docstring."""']
        for name in topic["holds"]:
            out += ["", text[name].rstrip("\n")]
    else:
        for name in topic["holds"]:
            out += gap(text[name]) + [text[name].rstrip("\n")]
    for name in topic.get("after", []):
        out += gap(text[name]) + [text[name].rstrip("\n")]
    return "\n".join(out) + "\n"


#: What `protocol/__init__.py` imports: the mixins it composes the
#: executor from, and the names the package has always offered, imported
#: back so that nothing which said `from mechbench_compute.protocol
#: import X` breaks. In the order ruff's isort wants them.
INIT_IMPORTS = """import json
from typing import Any

from mechbench_compute import Model
from mechbench_compute.protocol.chat import Chat
from mechbench_compute.protocol.dispatch import Dispatch
from mechbench_compute.protocol.is_remote import (  # noqa: F401
    REMOTE_BLOCKS,
    _is_remote,
)
from mechbench_compute.protocol.legacy_kinds import LegacyKinds
from mechbench_compute.protocol.memo import Memo
from mechbench_compute.protocol.model import ModelLoading
from mechbench_compute.protocol.node_summary import node_summary  # noqa: F401
from mechbench_compute.protocol.ordered_edges import _ordered_edges  # noqa: F401
from mechbench_compute.protocol.pipeline import Pipeline
from mechbench_compute.protocol.protocol_spec import ProtocolSpec
from mechbench_compute.protocol.read_tokenizer_id import read_tokenizer_id  # noqa: F401
from mechbench_compute.protocol.remote import (  # noqa: F401
    MAX_PARALLEL_NODES,
    Remote,
)
from mechbench_compute.protocol.serialize_model import serialize_model  # noqa: F401
from mechbench_compute.protocol.serialize_params import serialize_params  # noqa: F401
from mechbench_compute.protocol.tools import Tools"""


def render_init(text: dict[str, str]) -> str:
    mixins = sorted(t["mixin"] for t in TOPICS if t["mixin"])
    missing = [m for m in mixins if f" import {m}\n" not in INIT_IMPORTS + "\n"
               and f"    {m},\n" not in INIT_IMPORTS + "\n"]
    assert not missing, f"INIT_IMPORTS does not import {missing}"
    lead = "class ProtocolExecutor("
    wrapped = textwrap.wrap(", ".join(mixins), width=88 - len(lead),
                            break_long_words=False, break_on_hyphens=False)
    bases = [lead + wrapped[0]] + [" " * len(lead) + w for w in wrapped[1:]]
    bases[-1] += "):"
    out = [INIT_DOC, "", HEADER.rstrip("\n"), "", INIT_IMPORTS]
    out += ["", "", *bases,
            '    """The executor itself: what it is built with, and the one',
            '    call that runs a spec. Everything it does is a topic beside',
            '    this file — see the module docstring."""']
    out += ["", text["__init__"].rstrip("\n")]
    out += ["", text["run"].rstrip("\n")]
    out += ["", "", text["canonical_json"].rstrip("\n")]
    return "\n".join(out) + "\n"


def strip_plumbing(node: ast.AST) -> str:
    class NoPlumbing(ast.NodeTransformer):
        """Imports from this package are where code lives, not what it does."""

        def visit_ImportFrom(self, n):
            return None if (n.module or "").startswith("mechbench_compute") else n

        def visit_Pass(self, n):
            return None

    return ast.dump(NoPlumbing().visit(ast.parse(ast.unparse(node))))


def find(tree: ast.AST, name: str):
    """A definition at the top of a module, or in the one class it holds."""
    bodies = [tree.body]
    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            bodies.append(n.body)
    for body in bodies:
        for n in body:
            if getattr(n, "name", None) == name:
                return n
            targets = getattr(n, "targets", None) or (
                [n.target] if isinstance(n, ast.AnnAssign) else [])
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                return n
    return None


def verify(before: str) -> None:
    """Every definition is the same code where it landed."""
    old = ast.parse(before)
    trees = {t["file"]: ast.parse((PKG / "protocol" / t["file"]).read_text())
             for t in TOPICS}
    trees["__init__.py"] = ast.parse((PKG / "protocol" / "__init__.py").read_text())
    where = {}
    for t in TOPICS:
        for name in t.get("before", []) + t["holds"] + t.get("after", []):
            where[name] = t["file"]
    for name in KEPT:
        where[name] = "__init__.py"
    bad = 0
    for name in PIECES:
        landed = where.get(name)
        if landed is None:
            print(f"   ✗ {name} was not placed anywhere")
            bad += 1
            continue
        a, b = find(old, name), find(trees[landed], name)
        if a is None or b is None:
            print(f"   ✗ {name} did not land in {landed}")
            bad += 1
            continue
        if strip_plumbing(a) != strip_plumbing(b):
            print(f"   ✗ {name} differs after the move")
            bad += 1
    print(f"   verified {len(PIECES) - bad} of {len(PIECES)} definitions "
          f"identical by AST")
    if bad:
        sys.exit(1)


def main(dry: bool) -> None:
    before = before_text()
    _lines, text = load()
    written = []
    for topic in TOPICS:
        path = PKG / "protocol" / topic["file"]
        body = render(topic, text)
        written.append((path, body))
    written.append((PKG / "protocol" / "__init__.py", render_init(text)))
    for path, body in written:
        n = len(body.splitlines())
        print(f"   {'would write' if dry else 'wrote'} "
              f"{path.relative_to(ROOT)}  {n} lines")
        if not dry:
            path.write_text(body)
    if dry:
        return
    subprocess.run(["git", "add", "--"] + [str(p) for p, _ in written],
                   cwd=ROOT, check=True)
    verify(before)


if __name__ == "__main__":
    main(dry="--dry" in sys.argv)
