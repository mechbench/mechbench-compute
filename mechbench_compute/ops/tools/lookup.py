from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.lexicon._base import Op, P

OP = Op(
    name="tools/lookup",
    summary=(
        "Fetch a stored bench object by path — a tool that lets a model "
        "consult what the platform already knows."
    ),
    description="""\
Returns the object's payload, or one field of it when the call names a
`field`. Offered to a `chat` node as `"bench.lookup"`;
the model's call supplies `arguments: {path, field?}`. Every fetch is
recorded on the item that made it. A tool has no input ports — its
arguments come from the call.
""",
    inputs=(),
    output=None,
    params=(
        P("path", "string",
          "The object path, when the block is run directly rather than as "
          "a tool call.",
          None),
        P("fetch", "callable",
          "For tests: the function used to fetch, in place of the bench "
          "client. Not for protocols.",
          None),
    ),
    example={"path": "benjismith/stories/word-lists/opening-phrases"},
)


def run(ctx, inputs, params):
    return fetch_bench_object(inputs, params)


def fetch_bench_object(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    """`tools/lookup` — fetch a bench object by
    path, so a model can consult what the platform already knows.

    `fetch` is injectable (the executor passes the recording fetch, and
    tests pass their own) — nothing here touches the network directly.
    """
    args = dict(inputs.get("arguments") or {})
    path = str(args.get("path") or params.get("path") or "")
    if not path:
        raise ValueError("bench.lookup needs a `path`")
    fetch = params.get("fetch")
    if fetch is None:
        from mechbench_compute import bench

        fetch = bench.fetch
    fetched = fetch(path)
    payload = (fetched.get("payload", fetched)
               if isinstance(fetched, Mapping) else fetched)
    field_name = args.get("field")
    if field_name and isinstance(payload, Mapping):
        return {str(field_name): payload.get(str(field_name))}
    return payload
