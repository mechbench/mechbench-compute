from __future__ import annotations


def describe_reasoning_only(model_name: str, max_tokens: int) -> dict[str, str]:
    return {"cause": "reasoning", "message": (
        f"{model_name}: the reply is reasoning only — its "
        f"thinking ran to max_tokens ({max_tokens}) or closed "
        "with nothing after it. The reasoning is kept in the "
        "item's `reasoning`; raise max_tokens so the reply has "
        "room after it.")}
