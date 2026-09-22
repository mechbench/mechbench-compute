from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Params the remote path implements by asking the provider for them,
#: and the local path has no equivalent of. The remote path refuses
#: these by name when the provider cannot honour them
#: (`providers.base.check_supported`); the local path refuses them
#: outright, because a request field nobody reads is a wrong answer
#: with no error.
_REMOTE_ONLY = {
    "json_mode": "no local decoder constrains output to JSON; "
                 "ask for JSON in the prompt, or run this node on a provider",
    "logprobs": "the local path returns text, not scores — "
                "`logits/read` is the block that reads a distribution here",
    "tool_choice": "the local tool protocol is the model's chat template, "
                   "which offers tools and does not constrain the choice",
}


def refuse_remote_only(params: Mapping[str, Any]) -> None:
    asked = [p for p in sorted(_REMOTE_ONLY)
             if params.get(p) not in (None, False)]
    if not asked:
        return
    lines = "; ".join(f"{p} — {_REMOTE_ONLY[p]}" for p in asked)
    raise ValueError(
        f"text/chat on local weights cannot honour {', '.join(asked)}: {lines}. "
        "Drop the parameter, or give the node a provider model reference.")
