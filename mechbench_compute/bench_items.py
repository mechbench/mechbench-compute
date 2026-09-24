"""A collection's items, read on the server.

`bench.items` is `mechbench object items` and the MCP `object(verb="items")`
as a Python call: the fields asked for, of the items that pass every
`where`, a page at a time, so the rest of each item never leaves the
store. `bench.fetch_items` is the older whole-item page, kept for its
callers.

    bench.items("benji/lab/results/j_…/gen", fields="id,text", lines=1)
    bench.items(path, where={"coords.prompt": "flash"}, count=True)

`where` is `PATH OP VALUE` strings (OP one of `= != < <= > >= ~`), or a
mapping read as equalities. The grammar is the API's
(`mechbench-api/src/lib/item_query.ts`).
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Iterable, Mapping
from typing import Any


def _where(where: str | Iterable[str] | Mapping[str, Any] | None) -> list[str]:
    import json

    if where is None:
        return []
    if isinstance(where, str):
        return [where]
    if isinstance(where, Mapping):
        return [f"{k}={v if isinstance(v, str) else json.dumps(v)}"
                for k, v in where.items()]
    return list(where)


def items(path: str, *, fields: str | Iterable[str] | None = None,
          where: str | Iterable[str] | Mapping[str, Any] | None = None,
          sort: str | None = None, order: str | None = None,
          offset: int | None = None, limit: int | None = None,
          lines: int | None = None, chars: int | None = None,
          count: bool = False, header: bool = False,
          api_url: str | None = None, api_key: str | None = None) -> dict:
    """A page of a collection's items, projected and filtered on the server:
    `{path, total, matched, offset, limit, items}`, each item flat and keyed
    by the `fields` asked for (whole when none are). With `count`,
    `{path, total, matched}`; with `header`, `{path, total, header}`, the
    object without its items. `lines` and `chars` cut every string in the
    answer to its first N non-empty lines, then N characters."""
    from . import bench

    if fields is not None and not isinstance(fields, str):
        fields = ",".join(fields)
    query: list[tuple[str, Any]] = [("path", path)]
    for k, v in (("fields", fields), ("sort", sort), ("order", order),
                 ("offset", offset), ("limit", limit), ("lines", lines),
                 ("chars", chars)):
        if v is not None:
            query.append((k, v))
    query += [("where", w) for w in _where(where)]
    if count:
        query.append(("count", 1))
    if header:
        query.append(("header", 1))
    url, key = bench._config(api_url, api_key)
    return bench._request(
        "GET", f"{url}/objects/~items?{urllib.parse.urlencode(query)}", key)


def fetch_items(target: str, offset: int = 0, limit: int = 20, *,
                api_url: str | None = None,
                api_key: str | None = None) -> dict:
    """Fetch one page of a collection object's items, whole (server-side
    slicing; never transfers the whole collection). `items` reads chosen
    fields of chosen items."""
    return items(target, offset=offset, limit=limit, api_url=api_url,
                 api_key=api_key)
