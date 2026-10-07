from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.lexicon._base import Op, P, Resume

OP = Op(
    name="tools/lookup",
    resume=Resume("restart"),
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

A model's call reads only under the run's own project
(`owner/project/…`) and under the `prefixes` the tool's definition
grants; any other path is refused, so text a model reads cannot steer
it to another object the runner's key can see. A path is plain segments of letters, digits and
`_ . @ + = : , -`, none of them `.`, `..` or starting with `~`.
""",
    inputs=(),
    output=None,
    params=(
        P("path", "string",
          "The object path, when the block is run directly rather than as "
          "a tool call.",
          None),
        P("prefixes", "list[string]",
          "Further path prefixes a model may read under, beyond the run's own "
          "project: each `owner/project` or deeper.",
          None),
        P("fetch", "callable",
          "For tests: the function used to fetch, in place of the bench "
          "client. Not for protocols.",
          None),
    ),
    example={"path": "benjismith/stories/word-lists/opening-phrases"},
)


SEGMENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.@+=:,-]*")


class LookupRefused(ValueError):
    pass


def run(ctx, inputs, params):
    return fetch_bench_object(inputs, params)


def read_run_project(result_base: str | None) -> str | None:
    parts = str(result_base or "").split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 and all(parts[:2]) else None


def check_lookup_path(path: str, allowed: Sequence[str]) -> str:
    segments = path.split("/")
    if not all(SEGMENT.fullmatch(s) for s in segments):
        raise LookupRefused(
            f"bench.lookup refuses {path!r}: a path is plain segments of letters, digits "
            "and _ . @ + = : , -, none of them '.', '..' or starting with '~'")
    for prefix in allowed:
        head = [s for s in str(prefix).split("/") if s]
        if head and segments[:len(head)] == head:
            return path
    where = ", ".join(sorted({str(p).strip("/") for p in allowed if str(p).strip("/")}))
    raise LookupRefused(
        f"bench.lookup refuses {path!r}: it reads only under "
        + (where if where else "the run's project, and this run has none")
        + "; grant another prefix in the tool's `prefixes`")


def fetch_bench_object(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    args = dict(inputs.get("arguments") or {})
    path = str(args.get("path") or params.get("path") or "").strip("/")
    if not path:
        raise ValueError("bench.lookup needs a `path`")
    if args.get("path"):
        granted = params.get("prefixes") or ()
        if isinstance(granted, str):
            granted = (granted,)
        check_lookup_path(path, (*granted, *([params["_project"]] if params.get("_project") else ())))
    else:
        check_lookup_path(path, (path,))
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
