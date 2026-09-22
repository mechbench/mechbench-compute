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

from mechbench_compute.block_params import ACCEPTED, COMMON, check_inputs, check_params
from mechbench_compute import ops
from mechbench_compute.lexicon import BY_NAME

ROOT = pathlib.Path(__file__).resolve().parent.parent / "mechbench_compute"
P = "protocol/__init__.py"

#: op -> the places its params are read: (module, symbol). `symbol` is a
#: function or a class; None means the whole module. Several sites are
#: unioned — a wrapper plus what it delegates to.
SITES: dict[str, list[tuple[str, str | None]]] = {
    # --- model blocks implemented inline in the executor ---
    "logits/read": [(P, "_block_decision_read")],
    "text/generate": [
        ("ops/text/generate.py", "run")],
    "adapter/train": [(P, "_block_finetune_lora")],
    "eval/benchmark": [(P, "_block_eval_suite")],
    "eval/score": [(P, "_block_eval_hf_metric")],
    "logits/read-layers": [(P, "_block_lens")],
    "text/score": [(P, "_block_score")],
    "adapter/merge": [(P, "_block_merge")],
    "adapter/publish": [(P, "_block_hf_push_adapter")],
    "records/plot": [
        ("ops/records/plot.py", "run"),
        ("ops/records/plot.py", "_layer_axis_from"),
        ("ops/records/plot.py", "_check_layer_axis"),
        ("ops/records/plot.py", "_check_annotations"),
        ("ops/records/plot.py", "_check_references"),
        ("ops/records/plot.py", "viz_spec")],
    # --- model blocks that delegate to a module ---
    "text/chat": [(P, "_block_chat"), (P, "_block_chat_local"),
                              ("chat.py", "run_remote"), ("chat.py", "run_local")],
    "eval/judge": [(P, "_block_judge"), ("judge.py", "run")],
    "intervene/apply": [(P, "_block_intervene"), ("intervene/__init__.py", "run")],
    "direction/unembed": [(P, "_block_direction_vocab"),
                                         ("directions/__init__.py", "vocab_projection")],
    "trajectory/capture": [(P, "_block_trajectory_capture"),
                                            ("trajectory.py", "capture")],
    "text/tokenize": [(P, "_block_tokenize_stats"),
                                        ("tokenizer_stats.py", "block")],
    "intervene/ablate-layers": [(P, "_block_ablate_layers"),
                                       ("interp/__init__.py", "ablate_layers")],
    "intervene/ablate-heads": [(P, "_block_ablate_heads"),
                                      ("interp/__init__.py", "ablate_heads")],
    "intervene/steer": [(P, "_block_steer_inject"),
                                      ("interp/__init__.py", "steer_inject")],
    "logits/attribute": [(P, "_block_logit_attribution"),
                                            ("interp/__init__.py", "logit_attribution")],
    "intervene/patch": [
        ("ops/intervene/patch.py", "run"),
        ("ops/intervene/patch.py", "patch_trace"),
        ("ops/intervene/patch.py", "_last_logits"),
        ("ops/intervene/patch.py", "_attribution_grid")],
    "activations/capture-attention": [(P, "_block_attention_patterns"),
                                            ("interp/__init__.py", "attention_patterns")],
    "logits/scan": [(P, "_block_lens_positions"),
                                        ("interp/__init__.py", "lens_positions")],
    "activations/capture": [(P, "_block_residual_vectors"),
                                           ("interp/__init__.py", "residual_vectors")],
    "activations/capture-tokens": [(P, "_block_capture_tokens"),
                                   ("interp/__init__.py", "capture_tokens")],
    "activations/contrast": [(P, "_block_residual_divergence"),
                                              ("interp/__init__.py", "residual_divergence")],
    # --- pure blocks ---
    "records/cross": [
        ("ops/records/cross.py", "run"),
        ("ops/records/cross.py", "_sample_value"),
        ("ops/records/cross.py", "_factor_levels"),
        ("ops/records/cross.py", "factor_cross")],
    "records/fill": [
        ("ops/records/fill.py", "run"),
        ("ops/records/fill.py", "template")],
    "records/rename": [
        ("ops/records/rename.py", "run"),
        ("ops/records/rename.py", "_pop_path"),
        ("ops/records/rename.py", "_set_path"),
        ("ops/records/rename.py", "rename")],
    "records/select": [
        ("ops/records/select.py", "run"),
        ("ops/records/select.py", "select"),
        ("ops/records/select.py", "_selected")],
    "records/subtract": [
        ("ops/records/subtract.py", "run"),
        ("ops/records/subtract.py", "paired_delta")],
    "records/summarize": [
        ("ops/records/summarize.py", "run"),
        ("ops/records/summarize.py", "_bootstrap_mean"),
        ("ops/records/summarize.py", "summary_rows"),
        ("ops/records/summarize.py", "group_stats"),
        ("ops/records/summarize.py", "GroupStats")],
    "records/contrast": [
        ("ops/records/contrast.py", "run"),
        ("ops/records/contrast.py", "_field_of"),
        ("ops/records/contrast.py", "contrast")],
    "text/render": [("transcript.py", "render_records")],
    "text/extend": [("transcript.py", "extend")],
    "records/tabulate": [
        ("ops/records/tabulate.py", "run"),
        ("ops/records/tabulate.py", "table_from_records")],
    "records/union": [
        ("ops/records/union.py", "run"),
        ("ops/records/union.py", "union"),
        ("ops/records/union.py", "_shared_item_kind")],
    "records/zip": [
        ("ops/records/zip.py", "run"),
        ("ops/records/zip.py", "zip_branches")],
    "records/map": [
        ("ops/records/map.py", "run")],
    "records/fold": [
        ("ops/records/fold.py", "run")],
    "weights/circuit": [(P, "_block_weights_circuit"), ("weights/__init__.py", "head_circuits")],
    "activations/examples": [(P, "_block_examples"), ("interp/__init__.py", "examples")],
    "intervene/path": [(P, "_block_path_patch"), ("paths.py", "run")],
    "text/measure": [("blocks/__init__.py", "text_stats")],
    "eval/expect": [("blocks/__init__.py", "eval_expectation")],
    "geometry/compare": [("similarity.py", "geometry_similarity")],
    "geometry/span": [("trees.py", "mst")],
    "direction/add": [("directions/__init__.py", "block_add")],
    "direction/average": [("directions/__init__.py", "block_average")],
    "direction/decompose": [("directions/__init__.py", "block_from_pca")],
    "direction/fit": [("directions/__init__.py", "block_from_vectors")],
    "direction/regress": [("directions/__init__.py", "block_from_regression")],
    "direction/classify": [("directions/__init__.py", "block_classify")],
    "direction/normalize": [("directions/__init__.py", "block_normalize")],
    "direction/orthogonalize": [("directions/__init__.py", "block_orthogonalize")],
    "direction/project": [("directions/__init__.py", "block_project")],
    "trajectory/project": [("trajectory.py", "project")],
    "trajectory/compare": [("trajectory.py", "compare")],
    "trajectory/aggregate": [("trajectory.py", "aggregate")],
    # The reduce ops share one closure; the MONOID is what differs, and
    # each one's params are its own.
    # `_block_of` is the closure that reads the records port for all three.
    "records/total": [
        ("ops/records/total.py", "run"),
        ("ops/records/total.py", "FloatSum")],
    "records/rank": [
        ("ops/records/rank.py", "run"),
        ("ops/records/rank.py", "TopK")],
    "records/bin": [
        ("ops/records/bin.py", "run"),
        ("ops/records/bin.py", "Histogram")],
    "adapter/measure": [("weights/__init__.py", "measure_adapter")],
    "weights/capture": [(P, "_block_capture_weights"),
                        ("weights/__init__.py", "capture_weights")],
    "weights/decompose": [(P, "_block_decompose_weights"),
                          ("weights/__init__.py", "decompose_weights")],
    "tools/calc": [("tools.py", "calc")],
    "tools/lookup": [("tools.py", "bench_lookup")],
}

#: Params a block genuinely reads somewhere the scanner cannot follow —
#: each with the reason, in the 000438 spirit that an exemption is a
#: statement, not a shrug. Empty is the goal.
EXEMPT: dict[str, dict[str, str]] = {}

#: Names a block reads off `inputs` that are not ports — each with the
#: reason.
PORT_EXEMPT: dict[str, dict[str, str]] = {
    "tools/calc": {"arguments": "a tool's arguments come from the model's call, not an edge"},
    "tools/lookup": {"arguments": "a tool's arguments come from the model's call, not an edge"},
}

def _reads_of(var: str) -> re.Pattern[str]:
    # `params["x"]`, `params.get("x")`, and the guarded `(inputs or {}).get("x")`.
    return re.compile(rf"""(?:\({var} or \{{\}}\)|{var})(?:\.get\(\s*|\[\s*)["']([^"']+)["']""")


READS = _reads_of("params")
READS_INPUTS = _reads_of("inputs")
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
                # The whole path under the package, so a module inside a
                # subpackage resolves to its file rather than to a
                # top-level module that happens to share its last name.
                out[a.asname or a.name] = "/".join(tail[1:]) if len(tail) > 1 else a.name
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("mechbench_compute."):
                    out[a.asname or a.name.split(".")[-1]] = "/".join(a.name.split(".")[1:])
    return out


def _hands(call: ast.Call, var: str) -> bool:
    if any(isinstance(a, ast.Name) and a.id == var for a in call.args):
        return True
    if any(isinstance(a, ast.Starred) and isinstance(a.value, ast.Name)
           and a.value.id == var for a in call.args):
        return True
    return any(k.value is not None and isinstance(k.value, ast.Name)
               and k.value.id == var for k in call.keywords)


def _hands_params(call: ast.Call) -> bool:
    return _hands(call, "params")


def _called_name(call: ast.Call) -> tuple[str | None, str | None]:
    """(local name to resolve a module by, symbol called)."""
    if isinstance(call.func, ast.Name):
        return call.func.id, call.func.id
    if isinstance(call.func, ast.Attribute):
        base = call.func.value
        return (base.id if isinstance(base, ast.Name) else None), call.func.attr
    return None, None


def names_read(sites: list[tuple[str, str | None]], var: str = "params",
               *, bodies: list[str] | None = None) -> set[str]:
    """Every name read from `var` at these sites — `params["x"]`,
    `inputs.get("x")` — following `var` wherever it is handed on:
    module-local calls, cross-module imports, and into packages.
    Depth-bounded and cycle-safe. `bodies`, when given, collects the
    source of every site visited."""
    reads = _reads_of(var)
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
                found |= set(reads.findall(src))
                continue
            node = syms.get(sym)
            if node is None:
                # Not defined here, but perhaps imported here: follow a
                # re-export to where the definition is. A module that
                # hands on a name it took from elsewhere is a hop, not a
                # dead end.
                origin = _imports(src).get(sym)
                if origin is not None:
                    stack.append((origin, sym, depth + 1))
                continue
            body = ast.get_source_segment(src, node) or ""
            if bodies is not None:
                bodies.append(body)
            found |= set(reads.findall(body))
            imports = _imports(src)
            for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
                if not _hands(call, var):
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


def params_read(sites: list[tuple[str, str | None]]) -> set[str]:
    return names_read(sites, "params")


def _dispatch_branch(ref: str) -> str:
    """The executor's dispatch branch for `ref`: the lines between
    `block == "ref":` and the next `elif`/`else`, where a port is
    sometimes read straight off `inputs` (a chart's source, a score's
    collection path)."""
    src = (ROOT / P).read_text()
    m = re.search(
        rf'block == "{re.escape(ref)}":\s*\n(.*?)(?=\n\s+elif block|\n\s+else:)',
        src, re.S)
    return m.group(1) if m else ""


def _registry_entry(ref: str) -> str:
    """The pure-block registry's adapter for `ref` — the lambda that
    hands `inputs["records"]` to the function the site names."""
    src = (ROOT / "blocks/__init__.py").read_text()
    m = re.search(rf'"{re.escape(ref)}":\s*\n?\s*lambda inputs, params:(.*?)(?=\n\s+"[a-z]|\n\}})',
                  src, re.S)
    return m.group(1) if m else ""


def ports_read(ref: str) -> set[str]:
    """Every port name the op reads off `inputs`: at its sites, in the
    registry adapter, and in the executor's dispatch branch. An op run
    through `_run_model_block` reads `adapter` there, for every op alike."""
    bodies: list[str] = []
    read = names_read(SITES[ref], "inputs", bodies=bodies)
    branch = _dispatch_branch(ref)
    read |= set(READS_INPUTS.findall(branch))
    read |= set(READS_INPUTS.findall(_registry_entry(ref)))
    if "_run_model_block" in branch or any("_run_model_block" in b for b in bodies):
        read.add("adapter")
    # An operation in its own file (docs/OPS_LAYOUT.md) has no dispatch
    # branch: the executor fuses an adapter around it when its
    # declaration says there are local weights to fuse onto.
    if ops.find(ref) is not None and ops.fuses_adapter(ref):
        read.add("adapter")
    return read


def registered_ops() -> set[str]:
    """Every op the executor can run: the pure registry plus the blocks
    the dispatcher names."""
    from mechbench_compute.blocks import PURE_BLOCKS

    # The dispatcher compares the resolved bare name (docs/LEXICON.md §1).
    dispatched = set(re.findall(r'block == "([a-z0-9-]+/[a-z0-9-]+)"',
                                (ROOT / P).read_text()))
    return set(PURE_BLOCKS) | dispatched | set(ops.modules())


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


@pytest.mark.parametrize("ref", sorted(SITES))
def test_the_declaration_covers_every_port_the_block_reads(ref):
    """A port the block reads but does not declare would be refused by
    `check_inputs` when wired — a false refusal."""
    op = BY_NAME[ref]
    if op.wildcard is not None:
        return  # any port name lands on the wildcard
    missing = sorted(ports_read(ref) - op.port_names - set(PORT_EXEMPT.get(ref, {})))
    assert not missing, (
        f"{ref} reads {missing} from its inputs but declares no such port — "
        f"an edge onto one would be refused for no reason")


@pytest.mark.parametrize("ref", sorted(SITES))
def test_the_declaration_claims_no_port_the_block_ignores(ref):
    """A declared port nothing reads is the 000438 bug on the input side:
    wired, accepted, and silently unused."""
    op = BY_NAME[ref]
    declared = {p.name for p in op.inputs if not p.wildcard}
    claimed = sorted(declared - ports_read(ref))
    assert not claimed, (
        f"{ref} declares ports {claimed} but never reads them from inputs")


def test_exemptions_carry_a_reason():
    for table in (EXEMPT, PORT_EXEMPT):
        for ref, entries in table.items():
            assert ref in ACCEPTED, f"exemption for an undeclared block: {ref}"
            for param, why in entries.items():
                assert why.strip(), f"{ref}.{param} is exempt with no reason given"


# --- check_inputs behaviour ---------------------------------------------------

def test_an_unknown_port_is_refused_by_name():
    with pytest.raises(ValueError) as caught:
        check_inputs("geometry/span", {"matrix": {"kind": "collection", "item_kind": "geometry/similarity", "key": [], "items": []}})
    msg = str(caught.value)
    assert "'matrix'" in msg and "geometry/span" in msg and "similarity" in msg


def test_an_unwired_required_port_is_refused_before_anything_runs():
    with pytest.raises(ValueError) as caught:
        check_inputs("direction/project", {"vectors": [{"id": "a"}]})
    msg = str(caught.value)
    assert "direction/project" in msg and "'direction'" in msg and "direction/vector" in msg


def test_a_kind_that_does_not_satisfy_the_port_is_refused_with_both_names():
    verdicts = {"kind": "collection", "item_kind": "eval/verdict", "key": ["id"], "items": []}
    with pytest.raises(ValueError) as caught:
        check_inputs("geometry/compare", {"items": verdicts})
    msg = str(caught.value)
    assert "geometry/compare" in msg and "activations/vector" in msg and "eval/verdict" in msg


def test_a_kind_that_extends_the_port_kind_satisfies_it():
    reads = {"kind": "collection", "item_kind": "logits/decision", "key": ["id"], "items": []}
    out = check_inputs("eval/expect", {"results": reads, "expectations": [{"id": "x", "expect": {}}]})
    assert out["results"] is reads
    # A bare list on a collection port is wrapped as the collection it stands for.
    assert out["expectations"]["kind"] == "collection"
    assert out["expectations"]["item_kind"] == "records/record"
    # A retired spelling resolves before it is compared.
    old = {"kind": "decision_read", "conditions": []}
    check_inputs("eval/expect", {"results": old, "expectations": []})


def test_a_document_collection_is_a_record_collection():
    docs = {"kind": "collection", "item_kind": "text/document", "key": ["id"], "items": []}
    check_inputs("text/measure", {"records": docs})
    check_inputs("eval/judge", {"records": docs})


def test_a_wildcard_op_takes_any_port_name_but_needs_one():
    check_inputs("records/union", {"base": [{"id": "a"}], "adapted": [{"id": "b"}]})
    with pytest.raises(ValueError, match="at least one"):
        check_inputs("records/union", {})


def test_a_value_with_no_kind_is_not_second_guessed():
    # An older stored object, or a literal, carries no name to refuse by.
    check_inputs("direction/normalize", {"direction": {"vector": [1.0, 0.0]}})
    check_inputs("text/tokenize", {"vocabulary": ["red", "blue"]})
    # Nor does a kind the registry does not know — an extension's, or a
    # string an author wrote before kinds were named.
    check_inputs("activations/capture", {"records": {"kind": "owner/x", "items": [{"id": "a"}]}})


def test_the_prompt_objects_the_experiments_stored_are_record_collections():
    # `{"kind": "records", "records": [...]}` is what every experiment
    # author emitted for a prompt set; it is a collection of records.
    stored = {"kind": "records", "records": [{"id": "p", "user": "u"}]}
    out = check_inputs("activations/capture", {"records": stored})
    assert out["records"] is stored
    from mechbench_compute.lexicon import kinds as K
    assert K.item_kind_of(stored) == "records/record"
    assert K.items_of(stored) == [{"id": "p", "user": "u"}]


def test_a_port_given_as_a_param_is_refused_by_name():
    """A protocol stored before inputs left ports (0.78.0) put a port's
    value under `params`. That was lifted onto the port with a warning
    until 0.82.0, and is refused now — by name, saying it is a port."""
    from mechbench_compute import protocol

    assert not hasattr(protocol, "_lift_port_params"), \
        "the lift was removed in 0.82.0"
    with pytest.raises(ValueError, match="input port"):
        check_params("logits/read", {"conditions": []})
    with pytest.raises(ValueError, match="input port"):
        check_params("logits/read",
                     {"model": "m", "conditions": [{"id": "c"}], "top_k": 3})


# --- check_params behaviour ---------------------------------------------------

def test_an_executor_injection_is_not_refused():
    # `_block_runner` and friends are added by the executor, never
    # declared by a protocol.
    check_params("geometry/span", {"_block_runner": object()})


def test_an_unknown_param_is_refused_by_name():
    with pytest.raises(ValueError) as caught:
        check_params("geometry/span", {"centre": True})
    msg = str(caught.value)
    assert "centre" in msg and "geometry/span" in msg
    assert "compute version" in msg


def test_an_unregistered_block_is_not_second_guessed():
    # A block this runner does not know is the api's problem, not ours.
    check_params("~someone/ops/custom/1", {"anything": 1})
