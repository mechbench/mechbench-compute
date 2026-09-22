from __future__ import annotations

#: What a chat node emits per item, for both paths.
ITEM_KIND = "text/document"

#: What a remote node does with a reply that carried no prose and no
#: tool call: emit it marked, leave it out, or fail the node.
ON_EMPTY = ("keep", "skip", "error")

#: The provider name a local model's reasoning is stamped with: it goes
#: back only through a local chat template, and only to the same model.
LOCAL = "local"
