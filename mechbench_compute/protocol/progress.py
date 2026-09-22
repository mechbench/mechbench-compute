"""What a watcher is told while the graph runs.

One unit per node, except that a node which announces an item count
expands the denominator to it and ticks per item — the board's bar moves
per condition or per story rather than per node.

Alongside the flat scalar there is STRUCTURE: which node the run is in
and how far through it. A denominator that grows mid-run reads as a bug
to anyone watching; "node 3/5, 12/40" only ever counts up. The node view
is passed as a third argument only when the callback accepts one, so a
two-argument callback keeps working.
"""

from __future__ import annotations

import inspect
from typing import Any


class Progress:
    """Progress: see this module's docstring."""

    def __init__(self, on_progress, node_count: int, on_spool_item=None) -> None:
        self.on_progress = on_progress
        self.on_spool_item = on_spool_item
        self.total_units = node_count
        self.done_units = 0
        self.node_view: dict[str, Any] = {"index": 0, "count": node_count,
                                          "id": "", "done": 0, "total": 0}
        self.expanded = False
        try:
            self.wants_node = (
                on_progress is not None
                and len(inspect.signature(on_progress).parameters) >= 3)
        except (TypeError, ValueError):   # builtins, odd callables
            self.wants_node = False

    def report(self) -> None:
        if self.on_progress is None:
            return
        if self.wants_node:
            self.on_progress(self.done_units, self.total_units,
                             dict(self.node_view))
        else:
            self.on_progress(self.done_units, self.total_units)

    def bump(self, n: int = 1) -> None:
        self.done_units += n
        self.report()

    def expand(self, n_items) -> None:
        self.expanded = True
        if n_items > 1:
            self.total_units += n_items - 1
        self.node_view["total"] = max(int(n_items), 0)
        self.report()

    def start_node(self, pos: int, nid: str) -> None:
        self.node_view.update(index=pos + 1, id=nid, done=0, total=0)
        self.expanded = False
        self.report()

    def open_items(self, nid: str):
        """One node's item callback. Bound to the node rather than
        reading a shared "current node", because two nodes can be in
        flight at once and an item spooled under the wrong node's id is
        a resumed job reusing another node's work."""

        def on_item(key=None, item=None, reused=False):
            # Blocks that know nothing of resume call this bare; an
            # item-resumable block names the item so the runner can
            # spool it. A reused item counts as progress and is not
            # spooled again.
            self.node_view["done"] += 1
            self.bump(1)
            if (self.on_spool_item is not None and key is not None
                    and not reused):
                self.on_spool_item(nid, key, item)

        return on_item
