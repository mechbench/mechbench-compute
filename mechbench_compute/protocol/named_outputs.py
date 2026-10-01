from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.lexicon._base import DEFAULT_OUTPUT


def find_named_outputs(block: str) -> dict[str, Any]:
    from mechbench_compute.registry import REGISTRY

    resolved = REGISTRY.find(block)
    op = getattr(resolved, "op", None)
    return dict(getattr(op, "outputs", None) or {})


def split_outputs(block: str, result: Any) -> tuple[Any, dict[str, Any] | None]:
    named = find_named_outputs(block)
    if not named:
        return result, None
    from mechbench_compute.registry import REGISTRY

    if not isinstance(result, Mapping):
        raise TypeError(
            f"{block} declares the outputs {sorted(named)}; its run returns a "
            f"mapping of output name to value, not {type(result).__name__}")
    has_default = getattr(REGISTRY.find(block).op, "output", None) is not None
    if has_default and DEFAULT_OUTPUT not in result:
        raise ValueError(
            f"{block} returned no {DEFAULT_OUTPUT!r}: the output an edge that "
            f"names none reads")
    default = result[DEFAULT_OUTPUT] if has_default else result
    return default, {name: result[name] for name in named
                     if result.get(name) is not None}


def read_edge_output(edge: Mapping[str, Any]) -> str:
    return str(edge["from"].get("port") or DEFAULT_OUTPUT)


def is_named_edge(state, edge: Mapping[str, Any]) -> bool:
    return (read_edge_output(edge) != DEFAULT_OUTPUT
            and edge["from"]["node"] in state.named_results)


def read_edge_value(state, edge: Mapping[str, Any]) -> Any:
    node, name = edge["from"]["node"], read_edge_output(edge)
    if not is_named_edge(state, edge):
        return state.results[node]
    named = state.named_results[node]
    if name not in named:
        gave = ", ".join(sorted(named)) or "none but its default"
        raise ValueError(
            f"{node!r} gave no output {name!r} in this run (its named outputs "
            f"here: {gave}); an op emits a named output only when its params "
            f"ask for it")
    return named[name]


def read_edge_hash(state, edge: Mapping[str, Any]) -> str:
    node = edge["from"]["node"]
    if not is_named_edge(state, edge):
        return state.node_hashes.get(node, "")
    return state.named_hashes.get(node, {}).get(read_edge_output(edge), "")


def read_edge_path(state, edge: Mapping[str, Any]) -> str:
    node = edge["from"]["node"]
    if not is_named_edge(state, edge):
        return state.node_paths.get(node, "")
    return state.named_paths.get(node, {}).get(read_edge_output(edge), "")


def name_output_targets(state, nid: str, name: str) -> list[str]:
    from mechbench_compute import dataflow

    declared = state.named_outputs_of.get((nid, name), [])
    if state.declared_outputs is None:
        return [f"{state.result_base}/{nid}/{name}"]
    if declared:
        return [f"{state.result_base}/{n}" for n in declared]
    return [f"{state.result_base}/{dataflow.INTERMEDIATES}/{nid}/{name}"]
