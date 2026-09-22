from __future__ import annotations

#: What a chat node emits per item, for both paths.
ITEM_KIND = "text/document"

#: What a remote node does with a reply that carried no prose and no
#: tool call: emit it marked, leave it out, or fail the node.
ON_EMPTY = ("keep", "skip", "error")
