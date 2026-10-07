from __future__ import annotations

import copy
from typing import Any


def make_prompt_cache(model: Any) -> Any:
    from transformers import DynamicCache

    return DynamicCache(config=model.config)


def copy_cache(cache: Any) -> Any:
    return copy.deepcopy(cache)


def read_cache_offset(cache: Any) -> int:
    return 0 if cache is None else int(cache.get_seq_length())


def read_cached_kwargs(cache: Any) -> dict[str, Any]:
    return {} if cache is None else {"past_key_values": cache, "use_cache": True}
