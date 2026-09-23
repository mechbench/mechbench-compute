from __future__ import annotations

#: What a chat node emits per item, for both paths.
ITEM_KIND = "text/document"

#: What a remote node does with a reply that carried no prose and no
#: tool call: emit it marked, leave it out, or fail the node.
ON_EMPTY = ("keep", "skip", "error")

#: The provider name a local model's reasoning is stamped with: it goes
#: back only through a local chat template, and only to the same model.
LOCAL = "local"

#: How a generated item ended, as `metadata.sampling.ended` says it on
#: every item of `text/generate` and `text/chat`, local and remote, and
#: the keys of the header's `ended` count, in this order:
#: `end` the model ended its turn; `stop` at one of the node's `stop`
#: strings; `max_tokens` cut off at the token limit; `tool_call` the
#: reply was a tool call left unanswered (the tool rounds ran out, or no
#: handler); `filtered` the provider's filter or a refusal ended a reply
#: that still has prose; `empty` no prose and no tool call, the item's
#: `metadata.empty` saying why; `other` a provider's reason none of these
#: name, kept verbatim in `metadata.call.stop_reason`.
ENDINGS = ("end", "stop", "max_tokens", "tool_call", "filtered", "empty", "other")
