from __future__ import annotations

import importlib
import pkgutil
import warnings
from collections.abc import Mapping
from typing import Any

from mechbench_compute.lexicon._base import COLLECTION, KIND_ROOT, Kind
from mechbench_compute.lexicon.order import canonical_collection


def resolve_kind_module(kind: str) -> str:
    if "/" not in kind:
        return f"{__name__}.{kind}"
    family, name = kind.split("/", 1)
    return f"{__name__}.{family}.{name.replace('-', '_')}"


def find_kinds() -> tuple[Kind, ...]:
    found: list[Kind] = []
    for info in pkgutil.walk_packages(__path__, prefix=f"{__name__}."):
        if info.ispkg or info.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        kind = getattr(importlib.import_module(info.name), "KIND", None)
        if kind is None:
            raise ImportError(f"{info.name} is under kinds/ and declares no KIND")
        if resolve_kind_module(kind.name) != info.name:
            raise ImportError(
                f"{info.name} declares {kind.name!r}, which belongs at {resolve_kind_module(kind.name)}: "
                "a kind's path is a function of its name")
        found.append(kind)
    return tuple(sorted(found, key=lambda k: (k.name == COLLECTION, k.name)))


KINDS: tuple[Kind, ...] = find_kinds()

BY_KIND: dict[str, Kind] = {k.name: k for k in KINDS}

COLLECTION_KIND: Kind = BY_KIND[COLLECTION]

PLATFORM: tuple[Kind, ...] = tuple(k for k in KINDS if k.platform)

KIND_ALIASES: dict[str, tuple[str, bool]] = {
    "document_collection": ("text/document", True),
    "record_set": ("records/record", True),
    "records": ("records/record", True),
    "~canonical/kinds/text": ("text/document", False),
    "~canonical/kinds/transcript": ("text/transcript", False),
    "transcript": ("text/transcript", False),
    "~canonical/kinds/conversation": ("text/transcript", False),
    "~canonical/kinds/lens-trajectory": ("logits/funnel", False),
    "~canonical/kinds/lens-trajectory/2": ("logits/funnel", False),
    "~canonical/kinds/condition-set": ("records/condition", True),
    "~canonical/kinds/trajectory": ("trajectory/point", True),
    "~canonical/kinds/tokenizer-stats": ("text/tokenization", False),
    "~canonical/kinds/annotated-tokens": ("text/annotation", True),
    "~canonical/kinds/embeddings": ("activations/vector", True),
    "embeddings": ("activations/vector", True),
    "~canonical/kinds/call-provenance": ("provider/call", False),
    "~canonical/kinds/fs-snapshot": ("sandbox/snapshot", False),
    "fs_snapshot": ("sandbox/snapshot", False),
    "~canonical/kinds/sandbox-image": ("sandbox/image", False),
    "~canonical/kinds/sandbox-tool-call": ("sandbox/call", False),
    "~canonical/kinds/provider-cassette": ("provider/cassette", False),
    "provider_cassette": ("provider/cassette", False),
    "completion": ("provider/completion", False),
    "metric_table": ("records/table", False),
    "viz_spec": ("records/chart", False),
    "chart_spec": ("records/chart", False),
    "word_list": ("text/word-list", False),
    "decision_read": ("logits/decision", True),
    "decision_distribution": ("logits/decision", True),
    "intervene_readout": ("intervene/readout", True),
    "ablation_sweep": ("intervene/ablation", True),
    "head_ablation": ("intervene/heads", False),
    "steer_sweep": ("intervene/readout", True),
    "patch_trace": ("intervene/trace", True),
    "attention_patterns": ("activations/attention", True),
    "logit_attribution": ("logits/attribution", True),
    "lens_map": ("logits/lens", True),
    "divergence_map": ("activations/divergence", True),
    "residual_vectors": ("activations/vector", True),
    "similarity_matrix": ("geometry/similarity", True),
    "mst_summary": ("geometry/mst", True),
    "direction": ("direction/vector", False),
    "direction_similarity": ("geometry/similarity", False),
    "direction_similarity_matrix": ("geometry/similarity", False),
    "projections": ("activations/coordinate", True),
    "direction_vocab": ("direction/vocab", False),
    "trajectory": ("trajectory/point", True),
    "trajectory_projection": ("trajectory/point", True),
    "trajectory_comparison": ("trajectory/comparison", True),
    "trajectory_summary": ("trajectory/summary", True),
    "tokenizer_stats": ("text/tokenization", False),
    "annotation_layer": ("text/annotation", True),
    "adapter": ("adapter/lora", False),
    "checkpoint_manifest": ("adapter/checkpoint", False),
    "model_pointer": ("model/pointer", False),
    "hf_push": ("adapter/push", False),
    "ladder": ("run/ladder", False),
    "pipeline_result": ("run/result", False),
}

_LEGACY_ITEMS_FIELD: dict[str, str] = {
    "document_collection": "items", "record_set": "records", "records": "records",
    "decision_read": "conditions", "decision_distribution": "conditions",
    "intervene_readout": "rows", "ablation_sweep": "rows", "steer_sweep": "rows",
    "patch_trace": "pairs", "attention_patterns": "rows", "logit_attribution": "rows",
    "lens_map": "rows", "divergence_map": "pairs", "residual_vectors": "rows",
    "similarity_matrix": "layers", "mst_summary": "layers", "projections": "rows",
    "trajectory": "rows", "trajectory_projection": "rows", "trajectory_comparison": "rows",
    "trajectory_summary": "rows", "annotation_layer": "values",
}


def all_fields(kind: Kind) -> dict[str, dict[str, Any]]:
    chain: list[Kind] = []
    cur: Kind | None = kind
    while cur is not None:
        chain.append(cur)
        cur = BY_KIND[cur.extends] if cur.extends else None
    out: dict[str, dict[str, Any]] = {}
    for k in reversed(chain):
        out.update(k.fields)
    return out


def ancestry(name: str) -> tuple[str, ...]:
    out: list[str] = []
    cur: str | None = name
    while cur is not None and cur in BY_KIND and cur not in out:
        out.append(cur)
        cur = BY_KIND[cur].extends
    return tuple(out)


def satisfies(actual: str, declared: str) -> bool:
    if declared == COLLECTION:
        return True
    return declared in ancestry(actual)


def canonical_kind_path(name: str) -> str:
    if name == COLLECTION or name.startswith("~"):
        return name
    return f"{KIND_ROOT}{name}"


class RetiredKindName(DeprecationWarning):
    pass


_warned: set[str] = set()


def resolve_kind(kind: str, *, warn: bool = True) -> tuple[str, bool]:
    s = kind.strip()
    if s.startswith(KIND_ROOT):
        s = s[len(KIND_ROOT):]
    if s in BY_KIND:
        return s, s == COLLECTION
    hit = KIND_ALIASES.get(kind) or KIND_ALIASES.get(s)
    if hit is None:
        raise KeyError(kind)
    if warn and kind not in _warned:
        _warned.add(kind)
        warnings.warn(
            f"kind {kind!r} is a retired spelling of {hit[0]!r}"
            + (" (a collection of it)" if hit[1] else "")
            + "; new objects carry the current name.",
            RetiredKindName, stacklevel=2)
    return hit


def item_kind_of(obj: Mapping[str, Any]) -> str | None:
    k = obj.get("kind")
    if k == COLLECTION:
        ik = obj.get("item_kind")
        try:
            return resolve_kind(str(ik))[0] if ik else None
        except KeyError:
            return str(ik)
    if isinstance(k, str):
        try:
            name, plural = resolve_kind(k)
        except KeyError:
            return None
        return name if plural else None
    return None


def items_of(obj: Any) -> list[Any]:
    if isinstance(obj, list):
        return obj
    if not isinstance(obj, Mapping):
        raise ValueError("not a collection: a list or a mapping was expected")
    k = obj.get("kind")
    if k == COLLECTION and obj.get("storage") == "tensor":
        from mechbench_compute import tensors

        return tensors.items_of(obj)  # type: ignore[return-value]
    if k == COLLECTION and isinstance(obj.get("items"), list):
        return list(obj["items"])
    if isinstance(k, str) and k in _LEGACY_ITEMS_FIELD:
        return list(obj.get(_LEGACY_ITEMS_FIELD[k]) or [])
    for f in ("records", "conditions", "rows", "items"):
        if isinstance(obj.get(f), list):
            return list(obj[f])
    raise ValueError(f"not a collection: kind {k!r} carries no items")


def collection(item_kind: str, items: list[Any], **header: Any) -> dict[str, Any]:
    kind = BY_KIND[item_kind]
    out: dict[str, Any] = {"kind": COLLECTION, "item_kind": item_kind, "key": list(kind.key), "items": list(items)}
    out.update({k: v for k, v in header.items() if v is not None})
    return out


__all__ = [
    "BY_KIND", "COLLECTION", "COLLECTION_KIND", "KINDS", "KIND_ALIASES", "PLATFORM",
    "all_fields", "ancestry", "canonical_collection", "canonical_kind_path", "collection",
    "item_kind_of", "items_of", "resolve_kind", "satisfies",
]
