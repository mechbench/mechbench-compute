from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from mechbench_compute.live.read_notable import K_FLOORS

GRAPH_NODES_MAX = 16
SPEAK_ITEMS = 5
WALL_SECONDS = 120.0
TRAINING = frozenset({"adapter/train", "adapter/merge"})
RESULT = "result"
TOO_LONG = "too long for a try: run it as a job"


class TryRefused(ValueError):
    pass


class TryTooLong(TryRefused):
    def __init__(self) -> None:
        super().__init__(TOO_LONG)


def run_try(executor: Any, *, op: str | None = None, graph: Mapping[str, Any] | None = None,
            inputs: Mapping[str, Any] | None = None, params: Mapping[str, Any] | None = None,
            model: str, on_token: Callable[..., None] | None = None,
            live_run_id: str = "", seq: int = 0, wall_seconds: float = WALL_SECONDS,
            clock: Callable[[], float] = time.monotonic, baseline: Mapping[str, Any] | None = None,
            tries: Sequence[Mapping[str, Any]] = (), noise: Any = None, k: float = K_FLOORS,
            machine: str | None = None) -> dict[str, Any]:
    from mechbench_compute import __version__ as compute_version
    from mechbench_compute.protocol.summarize_node import summarize_node
    from mechbench_compute.resume import content_hash
    from mechbench_compute.seeds import derive

    if (op is None) == (graph is None):
        raise TryRefused("a try names one operation or gives one graph")
    if not k >= 1:
        raise TryRefused(f"k is how many floors a change must pass, at least 1, not {k:g}")
    given = {"op": op, "inputs": dict(inputs or {}), "params": dict(params or {})} if op is not None else {}
    seed = derive(live_run_id, seq)
    if op is not None:
        nodes = [{"id": "try", "block": op, "params": dict(params or {}), "inputs": dict(inputs or {})}]
        target = {"nodes": nodes, "edges": []}
        run_inputs: dict[str, Any] = {}
        bound: dict[str, Any] = {}
    else:
        nodes = [dict(n) for n in (graph or {}).get("nodes") or []]
        target = {**dict(graph or {}), "nodes": nodes}
        run_inputs = dict(inputs or {})
        bound = dict(params or {})
    if not nodes:
        raise TryRefused("a try's graph has no nodes")
    if len(nodes) > GRAPH_NODES_MAX:
        raise TryRefused(f"a try runs at most {GRAPH_NODES_MAX} nodes, and this graph has {len(nodes)}: run it as a job")
    end = _end_of(target)
    blocks = []
    for node in nodes:
        resolved = _admit(node, model, bound)
        node["block"] = resolved.name
        node["params"] = {**dict(node.get("params") or {})}
        node["params"].setdefault("seed", seed)
        blocks.append(resolved)

    deadline = clock() + wall_seconds

    def in_time(*_: Any) -> None:
        if clock() > deadline:
            raise TryTooLong

    def token(*args: Any, **kwargs: Any) -> None:
        in_time()
        if on_token is not None:
            on_token(*args, **kwargs)

    started = clock()
    got = executor.run_sub(target, run_inputs, bound, on_token=token, on_node_start=in_time,
                           outputs=[{"name": RESULT, "from": {"node": end}}])
    seconds = clock() - started
    result = got[RESULT]
    summary = summarize_node(result)
    end_block = next(b for n, b in zip(nodes, blocks, strict=True) if n["id"] == end)
    return {
        "result": result,
        "hash": content_hash(result),
        "kind": summary.get("kind"),
        "summary": summary,
        "lines": speak_lines(result, summary),
        "notable": read_try_notable(result, summary, end_block, executor, given, baseline=baseline,
                                    tries=tries, noise=noise, k=k, machine=machine),
        "seconds": seconds,
        "provenance": {
            "op": end_block.name if op is not None else [b.name for b in blocks],
            "pin": end_block.pin,
            "model": _model_ref(executor, model),
            "compute": compute_version,
            "seed": seed,
        },
    }


def _end_of(graph: Mapping[str, Any]) -> str:
    ids = [str(n.get("id")) for n in graph.get("nodes") or []]
    feeding = {str((e.get("from") or {}).get("node")) for e in graph.get("edges") or []}
    ends = [i for i in ids if i not in feeding]
    if len(ends) != 1:
        raise TryRefused(f"a try's graph ends in one node, and this one ends in {', '.join(ends) or 'none'}")
    return ends[0]


def _admit(node: dict[str, Any], model: str, bound: Mapping[str, Any]) -> Any:
    from mechbench_compute import model_ref
    from mechbench_compute.protocol.is_remote import is_remote
    from mechbench_compute.registry import REGISTRY

    name = str(node.get("block") or "")
    resolved = REGISTRY.find(name)
    if resolved is None:
        raise TryRefused(f"{name} is not an operation this runner has: run it as a job")
    if resolved.name in TRAINING:
        raise TryRefused(f"{resolved.name} is a job, not a try")
    params = dict(node.get("params") or {})
    given = params.get("model")
    if isinstance(given, Mapping) and "$param" in given:
        given = bound.get(str(given["$param"]))
    if is_remote(resolved, params) or (resolved.op.requires == "remote"
                                       and any(n.startswith("provider.") for n in resolved.op.needs)):
        raise TryRefused(f"{resolved.name} calls a hosted provider, and a try spends nothing: run it as a job")
    if resolved.op.requires == "remote":
        raise TryRefused(f"{resolved.name} reaches the network, and a try runs on this machine alone: run it as a job")
    if given is None:
        if resolved.op.requires != "pure":
            params["model"] = model
            node["params"] = params
        return resolved
    ref = model_ref.parse(given)
    if ref.is_endpoint:
        raise TryRefused(f"{resolved.name} names a hosted model, and a try spends nothing: run it as a job")
    if ref.base != model_ref.parse(model).base:
        raise TryRefused(f"this live run holds {model}, not {ref.base}")
    return resolved


def _model_ref(executor: Any, model: str) -> str:
    loaded = getattr(executor, "_model", None)
    ref = executor.model_ref(loaded) if loaded is not None and hasattr(executor, "model_ref") else None
    return ref or model


def read_try_notable(result: Any, summary: Mapping[str, Any], block: Any, executor: Any,
                     given: Mapping[str, Any], *, baseline: Mapping[str, Any] | None,
                     tries: Sequence[Mapping[str, Any]], noise: Any, k: float,
                     machine: str | None) -> dict[str, Any] | None:
    from mechbench_compute.lexicon import kinds as K
    from mechbench_compute.live.choose_baseline import choose_baseline
    from mechbench_compute.live.read_notable import read_items_and_header, read_notable

    kind = K.BY_KIND.get(str(summary.get("kind")))
    if kind is None or kind.notable is None:
        return None
    items, _ = read_items_and_header(result)
    chosen = choose_baseline(kind.notable, items, declared=baseline, tries=tries, op=given.get("op"),
                             inputs=given.get("inputs"), params=given.get("params"))
    architecture = getattr(getattr(getattr(executor, "_model", None), "architecture", None), "model_type", None)
    return read_notable(result, kind.notable, kind=kind.name, operation=block.name,
                        architecture=architecture, baseline=chosen, noise=noise, k=k, machine=machine,
                        pure=block.op.requires == "pure")


def speak_lines(result: Any, summary: Mapping[str, Any]) -> list[str]:
    from mechbench_compute.lexicon import kinds as K

    kind = K.BY_KIND.get(str(summary.get("kind")))
    if kind is not None and kind.speak:
        try:
            from mechbench_compute.expr.engine import load_engine

            if isinstance(result, Mapping) and K.item_kind_of(result) is not None:
                items, header = K.items_of(result)[:SPEAK_ITEMS], dict(result.get("header") or {})
            else:
                items, header = [result], {}
            said = load_engine().render(kind.speak, [dict(i) for i in items], {}, header).values
            return [str(s) for s in said]
        except Exception:  # noqa: BLE001
            pass
    return [summary_line(summary)]


def summary_line(summary: Mapping[str, Any]) -> str:
    kind = summary.get("kind") or "value"
    if summary.get("collection"):
        n = int(summary.get("items") or 0)
        return f"{n} {kind} item{'' if n == 1 else 's'}"
    rows = summary.get("rows")
    return f"one {kind}" + (f" of {rows} row{'' if rows == 1 else 's'}" if rows is not None else "")
