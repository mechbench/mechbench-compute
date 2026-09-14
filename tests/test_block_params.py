"""The param declarations must cover every op, and must not drift (000438, 000478).

Declaring params per block buys a loud failure when a protocol asks for
something a block cannot do. Two bugs sit either side of that:

* **Under-declaring** refuses a param the block does read — a false
  refusal, for no reason the author can see.
* **Over-declaring** accepts a param the block never reads, which is
  precisely the failure 000438 exists to prevent: six variety jobs
  declared `center: true`, all six succeeded, all six were uncentered.
  Declaring a port name as a param does exactly this, and `vectors/mst`
  was doing it.

So this reads the source and asserts the table equals what the code
reads, in both directions, for EVERY registered op. 000438 made
declaration opt-in on purpose ("listing all forty at once would be a
refactor with no failing test behind it"); 000478 is that refactor, and
this is the failing test behind it.

Finding what a block reads is not a one-function grep. A block is often a
thin wrapper in `protocol.py` that hands `params` to a module elsewhere
(`chat` → `chat.py`, the interp readouts → `interp.py`), and those in turn
hand it to helpers, sometimes in a package (`chat` → `providers/budget.py`
for `budget_usd`). The scanner follows `params` wherever it is passed,
across modules and into packages, which is what makes the equality
assertion safe to make.
"""
from __future__ import annotations

import ast
import pathlib
import re

import pytest

from mechbench_compute.block_params import ACCEPTED, COMMON, check_params

ROOT = pathlib.Path(__file__).resolve().parent.parent / "mechbench_compute"
P = "protocol.py"

#: op -> the places its params are read: (module, symbol). `symbol` is a
#: function or a class; None means the whole module. Several sites are
#: unioned — a wrapper plus what it delegates to.
SITES: dict[str, list[tuple[str, str | None]]] = {
    # --- model blocks implemented inline in the executor ---
    "logits/decision": [(P, "_block_decision_read")],
    "text/generate": [(P, "_block_generate")],
    "adapter/train": [(P, "_block_finetune_lora")],
    "eval/suite": [(P, "_block_eval_suite")],
    "eval/metric": [(P, "_block_eval_hf_metric")],
    "logits/funnel": [(P, "_block_lens")],
    "text/score": [(P, "_block_score")],
    "adapter/merge": [(P, "_block_merge")],
    "adapter/publish": [(P, "_block_hf_push_adapter")],
    "records/chart": [("blocks.py", "viz_spec")],
    # --- model blocks that delegate to a module ---
    "text/chat": [(P, "_block_chat"), (P, "_block_chat_local"),
                              ("chat.py", "run_remote"), ("chat.py", "run_local")],
    "eval/judge": [(P, "_block_judge"), ("judge.py", "run")],
    "text/conversation": [(P, "_block_conversation"),
                                      ("conversation.py", "run")],
    "intervene/apply": [(P, "_block_intervene"), ("intervene.py", "run")],
    "direction/vocab": [(P, "_block_direction_vocab"),
                                         ("directions.py", "vocab_projection")],
    "trajectory/capture": [(P, "_block_trajectory_capture"),
                                            ("trajectory.py", "capture")],
    "text/tokenize": [(P, "_block_tokenize_stats"),
                                        ("tokenizer_stats.py", "block")],
    "intervene/layers": [(P, "_block_ablate_layers"),
                                       ("interp.py", "ablate_layers")],
    "intervene/heads": [(P, "_block_ablate_heads"),
                                      ("interp.py", "ablate_heads")],
    "intervene/steer": [(P, "_block_steer_inject"),
                                      ("interp.py", "steer_inject")],
    "logits/attribution": [(P, "_block_logit_attribution"),
                                            ("interp.py", "logit_attribution")],
    "intervene/trace": [(P, "_block_patch_trace"),
                                     ("interp.py", "patch_trace")],
    "activations/attention": [(P, "_block_attention_patterns"),
                                            ("interp.py", "attention_patterns")],
    "logits/lens": [(P, "_block_lens_positions"),
                                        ("interp.py", "lens_positions")],
    "activations/vectors": [(P, "_block_residual_vectors"),
                                           ("interp.py", "residual_vectors")],
    "activations/divergence": [(P, "_block_residual_divergence"),
                                              ("interp.py", "residual_divergence")],
    # --- pure blocks ---
    "records/cross": [("blocks.py", "factor_cross")],
    # An alias for factor-cross: protocols pinned before the rename.
    "records/cross": [("blocks.py", "factor_cross")],
    "records/template": [("blocks.py", "template")],
    "records/select": [("blocks.py", "select")],
    "records/delta": [("blocks.py", "paired_delta")],
    "records/stats": [("blocks.py", "group_stats")],
    "records/table": [("blocks.py", "table_from_records")],
    "records/union": [("blocks.py", "union")],
    "text/stats": [("blocks.py", "text_stats")],
    "eval/expectation": [("blocks.py", "eval_expectation")],
    "geometry/similarity": [("blocks.py", "_vector_similarity")],
    "geometry/mst": [("trees.py", "mst")],
    "direction/add": [("directions.py", "block_add")],
    "direction/average": [("directions.py", "block_average")],
    "direction/from-pca": [("directions.py", "block_from_pca")],
    "direction/from-vectors": [("directions.py", "block_from_vectors")],
    "direction/normalize": [("directions.py", "block_normalize")],
    "direction/orthogonalize": [("directions.py", "block_orthogonalize")],
    "direction/project": [("directions.py", "block_project")],
    "direction/similarity": [("directions.py", "block_similarity")],
    "trajectory/project": [("trajectory.py", "project")],
    "trajectory/compare": [("trajectory.py", "compare")],
    "trajectory/aggregate": [("trajectory.py", "aggregate")],
    # The reduce ops share one closure; the MONOID is what differs, and
    # each one's params are its own.
    "records/sum": [("reduce.py", "FloatSum")],
    "records/top-k": [("reduce.py", "TopK")],
    "records/histogram": [("reduce.py", "Histogram")],
    "tools/calc": [("tools.py", "calc")],
    "tools/lookup": [("tools.py", "bench_lookup")],
}

#: Params a block genuinely reads somewhere the scanner cannot follow —
#: each with the reason, in the 000438 spirit that an exemption is a
#: statement, not a shrug. Empty is the goal.
EXEMPT: dict[str, dict[str, str]] = {}

READS = re.compile(r"""params(?:\.get\(\s*|\[\s*)["']([^"']+)["']""")
_MODULES: dict[str, tuple[str, dict[str, ast.AST]]] = {}


def _resolve(mod: str) -> list[str]:
    """A module name to the file(s) that can define it: a `.py`, or every
    `.py` in a package directory."""
    if mod.endswith(".py"):
        return [mod] if (ROOT / mod).is_file() else []
    if (ROOT / f"{mod}.py").is_file():
        return [f"{mod}.py"]
    d = ROOT / mod
    return [f"{mod}/{f.name}" for f in sorted(d.glob("*.py"))] if d.is_dir() else []


def _load(rel: str) -> tuple[str, dict[str, ast.AST]]:
    if rel not in _MODULES:
        src = (ROOT / rel).read_text()
        syms: dict[str, ast.AST] = {}
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                syms.setdefault(n.name, n)
        _MODULES[rel] = (src, syms)
    return _MODULES[rel]


def _imports(src: str) -> dict[str, str]:
    """Local name -> the mechbench_compute module it came from."""
    out: dict[str, str] = {}
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("mechbench_compute"):
            tail = (n.module or "").split(".")
            for a in n.names:
                out[a.asname or a.name] = tail[-1] if len(tail) > 1 else a.name
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("mechbench_compute."):
                    out[a.asname or a.name.split(".")[-1]] = a.name.split(".")[-1]
    return out


def _hands_params(call: ast.Call) -> bool:
    if any(isinstance(a, ast.Name) and a.id == "params" for a in call.args):
        return True
    if any(isinstance(a, ast.Starred) and isinstance(a.value, ast.Name)
           and a.value.id == "params" for a in call.args):
        return True
    return any(k.value is not None and isinstance(k.value, ast.Name)
               and k.value.id == "params" for k in call.keywords)


def _called_name(call: ast.Call) -> tuple[str | None, str | None]:
    """(local name to resolve a module by, symbol called)."""
    if isinstance(call.func, ast.Name):
        return call.func.id, call.func.id
    if isinstance(call.func, ast.Attribute):
        base = call.func.value
        return (base.id if isinstance(base, ast.Name) else None), call.func.attr
    return None, None


def params_read(sites: list[tuple[str, str | None]]) -> set[str]:
    """Every param name read at these sites, following `params` wherever
    it is handed on — module-local calls, cross-module imports, and into
    packages. Depth-bounded and cycle-safe."""
    found: set[str] = set()
    seen: set[tuple[str, str | None]] = set()
    stack: list[tuple[str, str | None, int]] = [(m, s, 0) for m, s in sites]
    while stack:
        mod, sym, depth = stack.pop()
        for rel in _resolve(mod):
            if (rel, sym) in seen or depth > 4:
                continue
            seen.add((rel, sym))
            src, syms = _load(rel)
            if sym is None:
                found |= set(READS.findall(src))
                continue
            node = syms.get(sym)
            if node is None:
                continue
            body = ast.get_source_segment(src, node) or ""
            found |= set(READS.findall(body))
            imports = _imports(src)
            for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
                if not _hands_params(call):
                    continue
                base, target = _called_name(call)
                if target is None:
                    continue
                if target in syms:
                    stack.append((rel, target, depth + 1))
                elif base and base in imports:
                    stack.append((imports[base], target, depth + 1))
                elif target in imports:
                    stack.append((imports[target], target, depth + 1))
    return found


def registered_ops() -> set[str]:
    """Every op the executor can run: the pure registry plus the blocks
    the dispatcher names."""
    from mechbench_compute.blocks import PURE_BLOCKS

    # The dispatcher compares the resolved bare name (docs/LEXICON.md §1).
    dispatched = set(re.findall(r'block == "([a-z0-9-]+/[a-z0-9-]+)"',
                                (ROOT / P).read_text()))
    return set(PURE_BLOCKS) | dispatched


# --- the gate ----------------------------------------------------------------

def test_every_registered_op_is_declared():
    """000478: opt-in declaration left 46 of 54 ops unchecked, which is
    the 000438 hole standing open everywhere it was not found."""
    missing = sorted(registered_ops() - set(ACCEPTED))
    assert not missing, (
        f"{len(missing)} registered ops declare no params: {missing}. "
        f"An undeclared block accepts anything and silently ignores what "
        f"it does not read.")


def test_every_declared_op_is_registered():
    stale = sorted(set(ACCEPTED) - registered_ops())
    assert not stale, f"declared but not registered: {stale}"


def test_every_declared_op_has_sites():
    assert sorted(SITES) == sorted(ACCEPTED), (
        "every declaration needs a source to check it against: "
        f"no sites for {sorted(set(ACCEPTED) - set(SITES))}, "
        f"no declaration for {sorted(set(SITES) - set(ACCEPTED))}")


@pytest.mark.parametrize("ref", sorted(SITES))
def test_the_declaration_covers_what_the_block_reads(ref):
    """Under-declaring is a false refusal of a param that works."""
    read = {p for p in params_read(SITES[ref]) if not p.startswith("_")}
    missing = sorted(read - ACCEPTED[ref] - COMMON)
    assert not missing, (
        f"{ref} reads {missing} but does not declare them — "
        f"a protocol using one would be refused for no reason")


@pytest.mark.parametrize("ref", sorted(SITES))
def test_the_declaration_claims_nothing_the_block_ignores(ref):
    """Over-declaring is the 000438 bug itself: accepted, and silently
    ignored. `vectors/mst` declared `similarity`, which is an input PORT
    it reads from `inputs`, never a param."""
    read = params_read(SITES[ref])
    exempt = set(EXEMPT.get(ref, {}))
    claimed = sorted(ACCEPTED[ref] - read - COMMON - exempt)
    assert not claimed, (
        f"{ref} declares {claimed} but never reads them from params. A "
        f"protocol setting one would be accepted and ignored — the exact "
        f"failure 000438 exists to prevent. Remove it, or list it in "
        f"EXEMPT with the reason it cannot be seen here.")


def test_exemptions_carry_a_reason():
    for ref, entries in EXEMPT.items():
        assert ref in ACCEPTED, f"exemption for an undeclared block: {ref}"
        for param, why in entries.items():
            assert why.strip(), f"{ref}.{param} is exempt with no reason given"


# --- check_params behaviour ---------------------------------------------------

def test_an_executor_injection_is_not_refused():
    # `_block_runner` and friends are added by the executor, never
    # declared by a protocol.
    check_params("geometry/mst", {"_block_runner": object()})


def test_an_unknown_param_is_refused_by_name():
    with pytest.raises(ValueError) as caught:
        check_params("geometry/mst", {"centre": True})
    msg = str(caught.value)
    assert "centre" in msg and "geometry/mst" in msg
    assert "compute version" in msg


def test_an_unregistered_block_is_not_second_guessed():
    # A block this runner does not know is the api's problem, not ours.
    check_params("~someone/ops/custom/1", {"anything": 1})
