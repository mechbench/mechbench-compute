from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.chat.constants import ITEM_KIND
from mechbench_compute.chat.write_turn import write_turn
from mechbench_compute.providers import messages as pm


def build_item(rec: Mapping[str, Any], k: int, text: str, *,
               model_wire: Any, params: Mapping[str, Any],
               parts: Sequence[Any] = (), call: Mapping[str, Any] | None = None,
               sampling: Mapping[str, Any] | None = None,
               tool_runs: Sequence[Any] = (),
               rounds: Sequence[Any] = (),
               tool_errors: Sequence[Any] = (),
               sandbox_calls: Sequence[Any] = (),
               sandbox_snapshot: Any = None,
               cell: Any = None) -> dict[str, Any]:
    coords = {**(rec.get("coords") or {}), "sample": k}
    if cell is not None:
        coords.update(cell.axes)
    meta: dict[str, Any] = {
        "coords": dict(coords),
        "model": model_wire,
    }
    if sampling:
        meta["sampling"] = dict(sampling)
    if call is not None:
        meta["call"] = dict(call)
    if tool_runs:
        # What the model actually did with the capabilities it was
        # given — the point of offering them.
        meta["tool_runs"] = [dict(r) for r in tool_runs]
    if tool_errors:
        # On the item, not only in the node's aggregate, so a failure
        # can be sliced by the condition that produced it.
        meta["tool_errors"] = [dict(e) for e in tool_errors]
    if sandbox_calls:
        # The snapshot chain this item drove — what the filesystem did,
        # call by call. The final snapshot's digest is the last one.
        meta["sandbox"] = [c.to_wire() for c in sandbox_calls]
    if sandbox_snapshot is not None:
        # The final workspace as a browsable fs-snapshot.
        meta["sandbox_final"] = sandbox_snapshot
    tool_parts = [p.to_wire() for p in parts
                  if not isinstance(p, (pm.TextPart, pm.ReasoningPart))]
    if tool_parts:
        meta["parts"] = tool_parts
    if any(isinstance(p, pm.ReasoningPart) for m in rounds for p in m.content):
        # The assistant turns of the tool loop before the reply, with
        # the reasoning each carried, as they went back to the provider.
        meta["rounds"] = [m.to_wire() for m in rounds]
    turn = write_turn(parts)
    if turn is not None:
        meta["turn"] = turn
    # A document is a record: its coordinates sit on the item as every
    # other record's do (and under `metadata` as well, where the readers
    # of older collections look).
    item = {"id": f"{rec.get('id')}-s{k}" + (f"-{cell.slug}" if cell is not None else ""),
            "kind": ITEM_KIND, "text": text,
            "coords": dict(meta["coords"]), "metadata": meta}
    # Reasoning is never prose: it sits beside `text`, with whatever the
    # provider needs to accept it back on a later turn.
    reasoning = pm.read_reasoning(parts)
    if reasoning:
        item["reasoning"] = reasoning
    if params.get("keep_fields"):
        for f in params["keep_fields"]:
            if f in rec:
                item[f] = rec[f]
    return item
