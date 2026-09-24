"""Bench client — emit results from local experiments to the workspace
(docs/THE_BENCH.md §3).

The researcher's front door to the bench is a local call, not the job
queue:

    from mechbench_compute import bench

    bench.emit(
        "benji/my-project/results/ladder-4",
        {"kind": "run/ladder", "rows": [...]},
        inputs=["benji/my-project/adapters/joint4"],
        params={"steps": 1000, "seed": 7},
        fidelity="text",
    )

`emit` wraps the payload in the mechbench-schema `Emitted` envelope
(provenance stamped with this package's version, `params_fingerprint`
computed over canonical CBOR), serializes canonically, and PUTs to the
API with the content hash header. Typed records that already carry a
top-level `provenance` field are sent as-is (no double wrapping).

Configuration comes from `MECHBENCH_API_URL` and `MECHBENCH_API_KEY`
(or explicit arguments). Everything here is import-lazy with respect to
mechbench-schema so offline interp work never pays for the dependency;
the schema package is required only when a bench call is actually made.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import ssl


class BenchError(RuntimeError):
    """A bench API call failed; the message carries the server detail.

    `status` and `body` carry the same answer in a form a caller can
    branch on: the HTTP status, and the decoded JSON body when the
    server sent one (the raw text otherwise). Reading a refusal's
    `code` beats matching on the message, which is prose and will be
    rewritten. Both are None for a failure that never reached a server.
    """

    def __init__(self, message: str, *, status: int | None = None,
                 body: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body

    def code(self) -> str | None:
        """The refusal's machine-readable code, when it carries one."""
        return (self.body or {}).get("code") if isinstance(self.body, dict) else None


class BenchTransportError(BenchError):
    """No answer the server stands behind: a dead socket, a timeout, or a
    5xx that survived every retry. A 4xx says the payload is wrong and
    will be wrong next time; this says nothing about the payload, so a host
    holding expensive compute keeps the bytes and tries again later."""


#: Bounded retry for failures that carry no verdict on the request.
#: Five attempts with exponential backoff and jitter spans ~2 minutes,
#: which covers a prod deploy's restart window. Small enough that a
#: genuinely dead API still surfaces in a couple of minutes, not hours.
_RETRY_ATTEMPTS = 5
_RETRY_BASE_DELAY = 2.0
_RETRY_STATUS = frozenset({502, 503, 504, 429})

#: The API's request-body ceiling for one object, mirrored so an emit
#: can refuse locally. The server is authoritative — it returns
#: 413 with `limitBytes` — and this constant must move with it
#: (`mechbench-api/src/lib/body_limit.ts`). 64 MiB.
MAX_OBJECT_BYTES = 64 * 1024 * 1024


#: Set by a host that already holds credentials — see `configure()`.
_DEFAULTS: dict[str, str] = {}


def configure(*, api_url: str | None = None, api_key: str | None = None) -> None:
    """Tell this module where the API is without the environment, for a
    host (mechbench-runner) whose credentials live in
    ~/.mechbench/config.toml: without it such a host's pipeline runs and
    then fails at its first emit with "no API url". Explicit arguments to
    a call still win over anything set here."""
    if api_url:
        _DEFAULTS["url"] = api_url
    if api_key:
        _DEFAULTS["key"] = api_key


def _config_file() -> Path:
    """Where `mechbench login` writes the credential — the same file the
    CLI reads (mechbench_runner/paths.py). A function, not a constant, so
    a test can point it at a fixture."""
    return Path.home() / ".mechbench" / "config.toml"


def _stored() -> tuple[str, str] | None:
    """The `(api_url, api_key)` pair `mechbench login` stored, or None.

    Read straight from `~/.mechbench/config.toml` with the standard-library
    TOML reader — deliberately NOT by importing mechbench-runner, which
    embeds this package and must not be imported back. The file format
    (`[runner]` table, `api_url`/`api_key`) is the contract; a malformed or
    absent file reads as "not logged in", never as an error.
    """
    import tomllib

    try:
        data = tomllib.loads(_config_file().read_text("utf-8"))
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError):
        return None
    table = data.get("runner")
    if not isinstance(table, dict):
        return None
    url, key = table.get("api_url"), table.get("api_key")
    if isinstance(url, str) and isinstance(key, str) and url and key:
        return url, key
    return None


def _config(api_url: str | None, api_key: str | None) -> tuple[str, str]:
    """Resolve (url, key) the way the CLI does, so an experiment script
    imports neither. In order: explicit argument, then
    `configure()`, then the environment, then the credential
    `mechbench login` stored.

    `MECHBENCH_API_KEY` owns the credential entirely when set — its URL is
    the env URL, and the stored file is not consulted — mirroring
    mechbench_runner/credentials.py, so a production key can never be sent
    to a localhost URL taken from a leftover config file.
    """
    url = api_url or _DEFAULTS.get("url")
    key = api_key or _DEFAULTS.get("key")

    env_key = os.environ.get("MECHBENCH_API_KEY")
    if not key and env_key:
        key = env_key
        url = url or os.environ.get("MECHBENCH_API_URL")
    else:
        url = url or os.environ.get("MECHBENCH_API_URL")
        if not key:
            stored = _stored()
            if stored:
                # Take the file's URL with its key, unless a URL was named
                # explicitly — never half a pairing from each source.
                key = stored[1]
                if not (api_url or _DEFAULTS.get("url")):
                    url = stored[0]

    if not url:
        raise BenchError(
            "no API url: set MECHBENCH_API_URL, run `mechbench login`, call "
            "mechbench_compute.bench.configure(api_url=...), or pass api_url=")
    if not key:
        raise BenchError(
            "no API key: set MECHBENCH_API_KEY, run `mechbench login`, call "
            "mechbench_compute.bench.configure(api_key=...), or pass api_key=")
    return url.rstrip("/"), key


def _tls(url: str) -> ssl.SSLContext | None:
    """A TLS context with CA roots that exist: `urllib` uses OpenSSL's
    trust store, which a Python from uv or python.org on macOS leaves
    empty, so every https:// call fails CERTIFICATE_VERIFY_FAILED while
    httpx, which bundles certifi, works. The runner's websocket client
    says the same."""
    if not url.startswith("https://"):
        return None
    import ssl

    import certifi

    return ssl.create_default_context(cafile=certifi.where())


def _retry_delay(attempt: int) -> float:
    """Exponential backoff with full jitter, for attempt 1, 2, 3…

    Jitter matters here even with one client: a runner emitting several
    nodes, or several runners riding out the same deploy, otherwise retry
    in lockstep and hit the API at the same instants.
    """
    import random

    return random.uniform(0.0, _RETRY_BASE_DELAY * (2 ** (attempt - 1)))


def _request(method: str, url: str, key: str, body: bytes | None = None,
             headers: dict[str, str] | None = None,
             return_headers: bool = False, timeout: float = 60,
             attempts: int = _RETRY_ATTEMPTS) -> Any:
    """One API call, retrying only what carries no verdict.

    Retried: a dead or timing-out socket, and 502/503/504/429. Each is
    silent about whether the request was acceptable, and every write in
    this module is safe to repeat — object PUTs are content-addressed, so
    the second attempt either writes identical bytes or is a no-op.

    Never retried: any other 4xx or 5xx. A rejected payload does not
    become acceptable by being sent again, and retrying it turns a clear
    error into a slow one.
    """
    import time

    last: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        req = urllib.request.Request(url, data=body, method=method)
        req.add_header("Authorization", f"Bearer {key}")
        for h, v in (headers or {}).items():
            req.add_header(h, v)
        try:
            with urllib.request.urlopen(
                    req, timeout=timeout, context=_tls(url)) as resp:
                raw = resp.read()
                ctype = resp.headers.get("content-type", "")
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
            break
        except urllib.error.HTTPError as e:
            # Parsed whole, quoted short: a refusal that names what is in
            # its way (a deletion's citing articles) runs past any
            # length worth printing, and a truncated body is not JSON.
            full = e.read().decode("utf-8", "replace")
            detail = full[:500]
            try:
                parsed: Any = json.loads(full)
            except ValueError:
                parsed = detail
            if e.code not in _RETRY_STATUS:
                raise BenchError(
                    f"{method} {url} -> {e.code}: {detail}",
                    status=e.code, body=parsed) from None
            last = BenchTransportError(
                f"{method} {url} -> {e.code}: {detail}",
                status=e.code, body=parsed)
        except urllib.error.URLError as e:
            # Includes socket.timeout on read/write ("The write
            # operation timed out"), which carries no verdict either.
            last = BenchTransportError(f"{method} {url} unreachable: {e.reason}")
        except TimeoutError as e:
            last = BenchTransportError(f"{method} {url} unreachable: {e}")
        if attempt >= max(1, attempts):
            raise last from None
        delay = _retry_delay(attempt)
        print(f"[bench] {method} {url} failed ({last}); "
              f"retry {attempt + 1}/{attempts} in {delay:.1f}s")
        time.sleep(delay)
    if ctype.startswith("application/json"):
        parsed = json.loads(raw)
        return (parsed, resp_headers) if return_headers else parsed
    return (raw, resp_headers) if return_headers else raw


def path(owner: str, project: str, *segments: str) -> str:
    """Build and validate a user MechbenchPath:
    ``path('benji', 'proj', 'corpora', 'stories')`` →
    ``'benji/proj/corpora/stories'``."""
    from mechbench_schema import parse_path

    p = "/".join([owner, project, *segments])
    parse_path(p)  # raises InvalidPathError on bad segments
    return p


def emit(target: str, payload: Any, *, inputs: tuple[str, ...] | list[str] = (),
         params: Any = None, fidelity: str | None = None,
         operation: str | None = None, params_ref: str | None = None,
         api_url: str | None = None, api_key: str | None = None) -> dict:
    """Emit one object to the bench; returns the server's write receipt
    (path, content hash, size, lineage parent count).

    If `payload` is a mapping that already carries a top-level
    `provenance` key (a typed record family), it is sent unchanged and
    `inputs`/`params`/`fidelity` must not also be passed. Otherwise the
    payload is wrapped in the `Emitted` envelope with provenance built
    from the arguments.
    """
    import mechbench_schema as ms

    from mechbench_compute import __version__ as core_version

    url, key = _config(api_url, api_key)

    if isinstance(payload, dict) and "provenance" in payload:
        if inputs or params is not None or fidelity is not None or operation:
            raise BenchError(
                "payload already carries provenance; pass inputs/params/"
                "fidelity through the typed record, not emit()")
        body_obj = payload
    else:
        for p in inputs:
            ms.parse_path(p)
        prov: dict[str, Any] = {
            "created_at": datetime.now(UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            "produced_by": {"tool": "mechbench-compute",
                            "version": core_version},
            "inputs": list(inputs),
            "params_fingerprint": (ms.fingerprint_params(params)
                                   if params is not None else None),
            "schema_version": ms.__version__,
            "fidelity": fidelity,
            "operation": operation,
            "params_ref": params_ref,
        }
        envelope = ms.Emitted(provenance=ms.Provenance(**prov),
                              payload=payload)
        # mode="python", not "json": binary payloads (adapter
        # safetensors bytes) must survive to CBOR, which encodes bytes
        # natively. Canonical bytes are identical for JSON-safe
        # payloads, so the two modes agree on every object that has
        # one.
        body_obj = envelope.model_dump(mode="python")

    body = ms.dump_canonical(body_obj)
    if len(body) > MAX_OBJECT_BYTES:
        # Refused HERE, in one line, rather than after a minute per
        # attempt against a server that will not take it. The
        # API enforces the same ceiling with a 413; this is the version
        # that names the size before any bytes leave the machine.
        raise BenchError(
            f"emit {target!r}: the canonical body is {len(body):,} bytes, "
            f"over the API's {MAX_OBJECT_BYTES:,}-byte object limit. Split "
            f"the result, lower its fidelity, or store the large part by "
            f"reference. (A result this size usually means a record is "
            f"carrying something it should not — bytes, a live object.)")
    import hashlib

    digest = hashlib.sha256(body).hexdigest()
    receipt = _request(
        "PUT", f"{url}/objects/{target}", key, body=body,
        headers={"Content-Type": "application/cbor",
                 "X-Content-Hash": f"sha256:{digest}"})
    return receipt


def register_kind(manifest: Any, *, api_url: str | None = None,
                  api_key: str | None = None) -> dict:
    """Register an item-kind manifest (a mechbench_schema.KindManifest
    or an equivalent dict). Registration is idempotent for identical
    content; changed content for an existing path is refused by the
    server (manifest versions are immutable)."""
    import hashlib

    import mechbench_schema as ms

    url, key = _config(api_url, api_key)
    obj = (manifest.model_dump(mode="json")
           if hasattr(manifest, "model_dump") else manifest)
    body = ms.dump_canonical(obj)
    digest = hashlib.sha256(body).hexdigest()
    return _request(
        "PUT", f"{url}/kinds/{obj['path']}", key, body=body,
        headers={"Content-Type": "application/cbor",
                 "X-Content-Hash": f"sha256:{digest}"})


def get_kind(kind_path: str, *, api_url: str | None = None,
             api_key: str | None = None) -> dict:
    """Fetch a registered kind manifest (decoded)."""
    url, key = _config(api_url, api_key)
    return _request("GET", f"{url}/kinds/{kind_path}", key)


def _unwrap_envelope(obj: Any) -> Any:
    """The payload inside an Emitted envelope, or the object unchanged.

    An envelope is a mapping carrying BOTH `payload` and `provenance`;
    a typed record (its own top-level `provenance`, no `payload`) and a
    bare payload are left alone. The same rule the CLI and the API's
    binding matcher use — one definition of "the envelope"."""
    if isinstance(obj, dict) and "payload" in obj and "provenance" in obj:
        return obj["payload"]
    return obj


def _fetch_decoded(target: str, api_url: str | None, api_key: str | None,
                   with_meta: bool) -> Any:
    import mechbench_schema as ms

    url, key = _config(api_url, api_key)
    raw = _request("GET", f"{url}/objects/{target}", key,
                   return_headers=with_meta)
    meta = None
    if with_meta:
        raw, headers = raw
        meta = {"content_hash": headers.get("x-content-hash")}
    if isinstance(raw, (bytes, bytearray)):
        try:
            decoded = ms.load_raw(bytes(raw))
            return (decoded, meta) if with_meta else decoded
        except Exception:  # noqa: BLE001 — not CBOR/JSON: hand back the bytes
            return (bytes(raw), meta) if with_meta else bytes(raw)
    return (raw, meta) if with_meta else raw


def fetch(target: str, *, api_url: str | None = None,
          api_key: str | None = None, with_meta: bool = False) -> Any:
    """Fetch an object and return its PAYLOAD.

    CBOR objects are decoded, JSON parsed, other mime types returned as
    bytes. An Emitted envelope is unwrapped — the payload is what a reader
    wants. Use `fetch_envelope()` for the rare read that needs the
    provenance. ``with_meta=True`` returns ``(payload, meta)`` where meta
    carries the server's ``content_hash``."""
    got = _fetch_decoded(target, api_url, api_key, with_meta)
    if with_meta:
        obj, meta = got
        return _unwrap_envelope(obj), meta
    return _unwrap_envelope(got)


def fetch_envelope(target: str, *, api_url: str | None = None,
                   api_key: str | None = None, with_meta: bool = False) -> Any:
    """The object exactly as stored — the Emitted envelope with its
    provenance, not just the payload: for the caller that wants lineage,
    the params fingerprint, or the producing tool version."""
    return _fetch_decoded(target, api_url, api_key, with_meta)


def fetch_items(target: str, offset: int = 0, limit: int = 20, *,
                api_url: str | None = None,
                api_key: str | None = None) -> dict:
    """Fetch one page of a collection object's items (server-side
    slicing; never transfers the whole collection)."""
    url, key = _config(api_url, api_key)
    return _request(
        "GET",
        f"{url}/objects/~items?path={target}&offset={offset}&limit={limit}",
        key)


def listing(prefix: str, *, api_url: str | None = None,
            api_key: str | None = None) -> dict:
    """List objects under a path prefix (owner-scoped)."""
    url, key = _config(api_url, api_key)
    return _request("GET", f"{url}/objects?prefix={prefix}", key)


def lineage(target: str, direction: str = "up", depth: int = 3, *,
            api_url: str | None = None, api_key: str | None = None) -> dict:
    """Walk the lineage graph from an object (`up` = inputs it was
    computed from; `down` = objects computed from it)."""
    url, key = _config(api_url, api_key)
    return _request(
        "GET",
        f"{url}/objects/~lineage?path={target}&direction={direction}"
        f"&depth={depth}", key)


def put_file(label: str, filepath, *, kind: str = "checkpoint_file",
             api_url: str | None = None, api_key: str | None = None,
             timeout: float = 3600.0) -> dict:
    """Upload one raw file as a binary object.

    The hash is computed here and the server verifies it after
    streaming — a torn upload can never be fetched. Bytes stream from
    disk; nothing holds the file in memory.
    """
    import hashlib

    import httpx

    url, key = _config(api_url, api_key)
    h = hashlib.sha256()
    size = 0
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)

    def _chunks():
        with open(filepath, "rb") as f2:
            while True:
                chunk = f2.read(1 << 20)
                if not chunk:
                    return
                yield chunk

    res = httpx.put(
        f"{url}/objects/{label}",
        content=_chunks(),
        headers={
            "authorization": f"Bearer {key}",
            "content-type": "application/octet-stream",
            "x-content-hash": f"sha256:{h.hexdigest()}",
            "x-object-kind": kind,
            "content-length": str(size),
        },
        timeout=timeout,
        verify=_tls(url),
    )
    if res.status_code not in (200, 201):
        raise BenchError(f"put_file {label}: {res.status_code} {res.text[:200]}")
    return res.json()


def get_file_chunks(label: str, *, api_url: str | None = None,
                    api_key: str | None = None, timeout: float = 3600.0):
    """Stream a binary object's bytes, chunk by chunk."""
    import httpx

    url, key = _config(api_url, api_key)
    with httpx.stream(
        "GET",
        f"{url}/objects/{label}",
        headers={"authorization": f"Bearer {key}"},
        timeout=timeout,
        verify=_tls(url),
    ) as res:
        if res.status_code != 200:
            raise BenchError(f"get_file {label}: {res.status_code}")
        yield from res.iter_bytes(1 << 20)


def list_prefix_hashes(prefix: str, *, api_url: str | None = None,
                       api_key: str | None = None) -> dict[str, str]:
    """{filename: sha256hex} for objects already stored under a label
    prefix — what retry-as-resume consults before uploading.
    Best-effort: an empty dict just means upload everything."""
    import httpx

    try:
        url, key = _config(api_url, api_key)
        res = httpx.get(
            f"{url}/objects/~inventory",
            params={"limit": 200},
            headers={"authorization": f"Bearer {key}"},
            timeout=30.0,
            verify=_tls(url),
        )
        if res.status_code != 200:
            return {}
        out: dict[str, str] = {}
        for row in res.json().get("objects", []):
            path = str(row.get("path") or "")
            if path.startswith(prefix + "/"):
                ch = str(row.get("contentHash") or "")
                if ch.startswith("sha256:"):
                    out[path[len(prefix) + 1:]] = ch[len("sha256:"):]
        return out
    except Exception:  # noqa: BLE001 — resume is an optimization, never a gate
        return {}


# --- runs, jobs, results -----------------------------------------------------
#
# Launch, watch, find and read: the one implementation under the
# `mechbench run/watch/result/runs/label` verbs (mechbench-runner).

#: A job is finished — successfully or not — in exactly these states.
#: `done_with_missing` is finished: the run completed and part
#: of the graph did not, which is a result to read rather than a job to
#: keep waiting on.
TERMINAL = ("done", "done_with_missing", "failed", "cancelled", "interrupted")


def launch(protocol: str, *,
           params: dict[str, Any] | None = None,
           inputs: dict[str, Any] | None = None,
           keep: str | None = None,
           budget: float | None = None, label: str | None = None,
           api_url: str | None = None,
           api_key: str | None = None) -> dict:
    """Bind a protocol and queue its job. `POST /protocols/:ref/runs`.

    A protocol declares `params` (typed values: the model, an `n`) and
    `inputs` (stored objects, by path or as a `{"$ref"}`), and a run
    binds each by name. `keep="outputs"` asks for the
    intermediates to be held on the runner rather than stored.
    `label`, one line, says what the run is for: see `runs`, `label_run`.

    Returns the bare run, with `id` and `jobId` on it —
    record the job id at once; a job id in a scrollback is a job id lost.
    """
    url, key = _config(api_url, api_key)
    body: dict[str, Any] = {"params": dict(params or {})}
    if inputs is not None:
        body["inputs"] = {
            name: ({"$ref": {"bench": v}} if isinstance(v, str) else v)
            for name, v in inputs.items()}
    if keep is not None:
        if keep not in ("all", "outputs"):
            raise ValueError(f"keep is 'all' or 'outputs', not {keep!r}")
        body["keep"] = keep
    if budget is not None:
        body["budgetUsd"] = budget
    if label is not None:
        body["label"] = label
    return _request(
        "POST", f"{url}/protocols/{protocol}/runs", key,
        body=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, timeout=90)


def create_protocol(owner: str, project: str, name: str, *, graph: dict,
                    description: str = "",
                    params: list[dict] | None = None,
                    inputs: list[dict] | None = None,
                    outputs: list[dict] | None = None,
                    owner_kind: str = "user", exists: str = "version",
                    publish: bool = False,
                    api_url: str | None = None,
                    api_key: str | None = None) -> dict:
    """Register a protocol: `POST /protocols` with its graph and
    `params=[{"name", "type", "default"?, "doc"?}]`, `inputs=[{"name",
    "kind", "many"?, "default"?}]`, `outputs=[{"name", "from": {"node",
    "output"?}}]`. Returns the bare protocol. `dataflow: 2` is added when
    the graph leaves it out; the legacy form raises before anything is sent.

    A project holds one protocol per name, so a taken name is PATCHed,
    which makes the next VERSION and keeps the old one replayable
    (`exists="error"` raises instead). A protocol kept as a file is better
    pushed (`push_protocol`), which makes no version when nothing changed.
    `publish=True` publishes the version left at the head, under
    `published`.
    """
    if exists not in ("version", "error"):
        raise ValueError(f"exists is 'version' or 'error', not {exists!r}")
    from mechbench_compute import dataflow

    if isinstance(graph, dict) and dataflow.find_undeclared(graph) == dataflow.NO_MARKER:
        graph = {"dataflow": dataflow.DATAFLOW, **graph}
    dataflow.check_form({"graph": graph})
    signature = {"params": list(params or []), "inputs": list(inputs or []),
                 "outputs": list(outputs or [])}
    url, key = _config(api_url, api_key)
    body: dict[str, Any] = {
        "ownerKind": owner_kind, "ownerHandle": owner, "projectSlug": project,
        "name": name, "description": description, "graph": graph,
        "signature": signature,
    }
    try:
        out = _request(
            "POST", f"{url}/protocols", key,
            body=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, timeout=90)
    except BenchError as e:
        taken = _name_taken(e)
        if taken is None or exists == "error":
            raise
        # The name is this protocol's; give it the new graph. The server
        # bumps the version, snapshots the last one, and the runs that
        # pinned it still replay.
        patch: dict[str, Any] = {"graph": graph, "description": description,
                                 "signature": signature}
        out = _request(
            "PATCH", f"{url}/protocols/{taken}", key,
            body=json.dumps(patch).encode("utf-8"),
            headers={"Content-Type": "application/json"}, timeout=90)
    # The protocols routes answer `{protocol: …}`. Unwrapped here,
    # once, so no caller has to.
    protocol = out.get("protocol", out) if isinstance(out, dict) else out
    if publish:
        protocol = {**protocol, "published": publish_protocol_version(
            protocol["id"], protocol["version"], api_url=api_url, api_key=api_key)}
    return protocol


def push_protocol(file: str | Path | dict, into: str, *, owner_kind: str = "user",
                  api_url: str | None = None, api_key: str | None = None) -> dict:
    """Push a protocol file (a path, or its parsed JSON: `name`,
    `description`, `params`, `inputs`, `outputs`, `graph`) into `into`,
    `owner/project`: `POST /protocols/push`. By its name the server
    answers `{action, protocol, findings}`, the action `created`,
    `versioned` (graph or signature changed), `described` (only the
    description) or `unchanged`. The legacy form or failed wiring raises
    `BenchError` with its code and findings, and nothing is stored."""
    owner, _, project = into.partition("/")
    if not owner or not project or "/" in project:
        raise ValueError(f"a project is named owner/project, not {into!r}")
    try:
        content = file if isinstance(file, dict) else json.loads(Path(file).read_text())
    except ValueError as e:
        raise BenchError(f"{file}: not a JSON protocol file ({e})") from None
    url, key = _config(api_url, api_key)
    body = {"ownerKind": owner_kind, "ownerHandle": owner, "projectSlug": project,
            "protocol": content}
    return _request("POST", f"{url}/protocols/push", key, body=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"}, timeout=90)


def export_protocol(protocol: str, *, version: int | None = None,
                    path: str | Path | None = None, api_url: str | None = None,
                    api_key: str | None = None) -> dict:
    """A version (the head by default) as its canonical file text, which a
    push reads back as `unchanged`: `{protocolId, name, version, text}`,
    and the text written to `path` exactly when one is given."""
    url, key = _config(api_url, api_key)
    query = "" if version is None else f"?version={int(version)}"
    out = _request("GET", f"{url}/protocols/{protocol}/export{query}", key)
    if path is not None:
        Path(path).write_text(out["text"])
    return out


def _name_taken(e: BenchError) -> str | None:
    """The id in a `NAME_TAKEN` refusal, or None for any other error.
    Read by CODE rather than by matching the message, which is prose;
    the server names the protocol holding the name so a caller need not
    go looking for it."""
    if e.code() != "NAME_TAKEN":
        return None
    got = e.body.get("protocolId") if isinstance(e.body, dict) else None
    return str(got) if got else None


# --- publishing, copying, deleting ------------------------------------------
#
# A published protocol version is readable by anyone, and is what an
# article embeds; the version is the published unit, so a later edit
# never reaches it. A copy brings its sub-protocols along. Every
# deletion answers a dry run, is refused while something outside it
# depends on it, and names the articles citing it until told
# otherwise.


def _public_path(version: dict) -> str | None:
    """The site path of a published version's page, when the answer names
    enough to build it: `/<owner>/<project>/protocols/<id>/v/<n>`."""
    owner, project = version.get("ownerHandle"), version.get("projectSlug")
    pid, n = version.get("protocolId"), version.get("version")
    if not (owner and project and pid and n):
        return None
    return f"/{owner}/{project}/protocols/{pid}/v/{n}"


def get_protocol(protocol: str, *, api_url: str | None = None,
                 api_key: str | None = None) -> dict:
    """`GET /protocols/:id` — the bare protocol, its head `version` among
    the rest."""
    url, key = _config(api_url, api_key)
    out = _request("GET", f"{url}/protocols/{protocol}", key)
    return out.get("protocol", out) if isinstance(out, dict) else out


def publish_protocol_version(protocol: str, version: int, *,
                             api_url: str | None = None,
                             api_key: str | None = None) -> dict:
    """Make one sealed version readable by anyone.

    `POST /protocols/:id/versions/:n/publish`, which takes someone who can
    administer the protocol. Returns `{version, unpublishedIncludes,
    publicPath}`: the version as the public reads it, the sub-protocols
    it includes that are not published (a reader sees only their names),
    and its page's path on the site. Publishing twice is the same answer.
    """
    url, key = _config(api_url, api_key)
    out = _request("POST", f"{url}/protocols/{protocol}/versions/{int(version)}/publish", key)
    return {**out, "publicPath": _public_path(out.get("version") or {})}


def unpublish_protocol_version(protocol: str, version: int, *,
                               api_url: str | None = None,
                               api_key: str | None = None) -> dict:
    """Withdraw a published version. Returns `{version, published,
    citedBy, unreadable}` — the articles that embed it, or link to it from
    a result, now show a placeholder there."""
    url, key = _config(api_url, api_key)
    return _request("POST", f"{url}/protocols/{protocol}/versions/{int(version)}/unpublish", key)


def copy_protocol_version(protocol: str, version: int, owner: str, project: str, *,
                          name: str | None = None, owner_kind: str = "user",
                          dry_run: bool = False, api_url: str | None = None,
                          api_key: str | None = None) -> dict:
    """Copy a version into a project of yours.

    A protocol includes only protocols with its own owner, so the
    copy brings every sub-protocol along, or reuses an earlier copy of the
    same version in that project. Returns `{protocol, copied, reused}`;
    with `dry_run`, `{name, copied, reused}` and nothing made. A
    sub-protocol you cannot read refuses the whole copy
    (`UNREADABLE_INCLUDES`).
    """
    url, key = _config(api_url, api_key)
    body: dict[str, Any] = {"ownerKind": owner_kind, "ownerHandle": owner, "projectSlug": project}
    if name is not None:
        body["name"] = name
    return _request(
        "POST", f"{url}/protocols/{protocol}/versions/{int(version)}/copy"
                f"{'?dryRun=1' if dry_run else ''}", key,
        body=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, timeout=90)


#: What an id names, by its prefix. Anything else is an object path.
_DELETABLE = {"prt": "protocols", "j": "jobs", "art": "articles", "ds": "datasets", "proj": "projects"}


def _deletion_url(url: str, target: str, prefix: bool) -> str:
    head, sep, _ = target.partition("_")
    if sep and "/" not in target and head in _DELETABLE:
        return f"{url}/{_DELETABLE[head]}/{target}"
    if "/" not in target:
        raise ValueError(f"{target!r} is neither an object path nor a protocol, job, "
                         f"article, dataset or project id")
    return f"{url}/objects/{target}{'?prefix=1' if prefix else ''}"


def delete(target: str, *, prefix: bool = False, dry_run: bool = False,
           acknowledge_citations: bool = False, api_url: str | None = None,
           api_key: str | None = None) -> dict:
    """Delete an object (a path; everything under it with `prefix`), or a
    protocol, job, article, dataset or project (an id).

    With `dry_run`, answers what it would do — `{deletes, keeps, refusal,
    citedBy, unreadable}` — and deletes nothing. Otherwise a refusal
    raises `BenchError` with its code (`LINEAGE_CHILDREN`,
    `DATASET_REFERENT`, `INCLUDED`, `JOBS_RUNNING`, `DEPENDED_ON`, …), and
    articles citing the target raise `CITED` until
    `acknowledge_citations=True`. What is deleted keeps its history
    (`history`), and its address is free for something new at once.
    """
    url, key = _config(api_url, api_key)
    target_url = _deletion_url(url, target, prefix)
    query = [q for q, on in (("dryRun=1", dry_run), ("acknowledge=citations", acknowledge_citations)) if on]
    if query:
        target_url += ("&" if "?" in target_url else "?") + "&".join(query)
    return _request("DELETE", target_url, key)


def history(kind: str, entity_id: str, *, api_url: str | None = None,
            api_key: str | None = None) -> dict:
    """A lifetime's audit log, readable after the thing is gone:
    `{lifetime, events, others}`, where `others` are the other
    lifetimes that have held its address. `kind` is object, protocol,
    article, project, dataset or job."""
    url, key = _config(api_url, api_key)
    return _request("GET", f"{url}/history/{kind}/{entity_id}", key)


def cancel(job_id: str, *, reason: str = "", api_url: str | None = None,
           api_key: str | None = None) -> dict:
    """Withdraw a job nobody is running, the counterpart of `launch`:
    `POST /jobs/:id/cancel`. Works while no compute is being spent
    (`queued`, `preparing`, `interrupted`); refused for a running job,
    which a server cannot stop. Idempotent, with `alreadyCancelled` set,
    so two people draining a queue do not race. Returns `{ok, status,
    from}`."""
    url, key = _config(api_url, api_key)
    body = {"reason": reason} if reason else {}
    return _request(
        "POST", f"{url}/jobs/{job_id}/cancel", key,
        body=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})


def get_job(job_id: str, *, api_url: str | None = None,
            api_key: str | None = None) -> dict:
    """`GET /jobs/:id` — the bare job row (status, progress, spend,
    resultPath)."""
    url, key = _config(api_url, api_key)
    return _request("GET", f"{url}/jobs/{job_id}", key)


def _progress_sig(j: dict) -> tuple:
    """The job's progress-bearing fields, so `watch` yields on a real
    change and not on an identical poll."""
    node = (j.get("progressNode") or {}).get("id")
    return (j.get("status"), j.get("progressNum"), j.get("progressDen"),
            j.get("progressUnit"), node, j.get("spentUsd"))


def watch(jobs: list[str], *, interval: float = 4.0, api_url: str | None = None,
          api_key: str | None = None):
    """Poll jobs to a terminal state, yielding `(job_id, job)` each time a
    job's progress CHANGES, never the same state twice. A transient fetch
    error yields `(job_id, {"status": None, "error": <str>})` and the poll
    continues. It prints nothing: `last = dict(bench.watch(jobs))` keeps
    each job's terminal state, since that is its last yield."""
    import time

    url, key = _config(api_url, api_key)
    last: dict[str, tuple] = {}
    pending = list(dict.fromkeys(jobs))  # de-dup, preserve order
    while pending:
        for jid in list(pending):
            try:
                j = _request("GET", f"{url}/jobs/{jid}", key)
            except BenchError as e:
                yield jid, {"status": None, "error": str(e)}
                continue
            sig = _progress_sig(j)
            if sig != last.get(jid):
                last[jid] = sig
                yield jid, j
            if j.get("status") in TERMINAL:
                pending.remove(jid)
        if pending:
            time.sleep(interval)


def results_for(protocol: str, **bindings: Any) -> list[dict]:
    """The protocol's runs, newest first, filtered by binding value
    (`GET /protocols/:ref/runs?binding.k=v`). Each carries its `jobId`,
    `jobStatus` and `resultPath`.

    A string binding matches raw; a structured one (a model ref) is sent as
    canonical JSON, which the server parses and deep-equals — so
    `results_for("024", ref={"provider": "x", "model": "y"})` filters too.
    Credentials come from the environment or `mechbench login`; pass a
    custom endpoint through `configure()`.
    """
    import urllib.parse

    url, key = _config(None, None)
    params = [
        (f"binding.{k}",
         v if isinstance(v, str)
         else json.dumps(v, sort_keys=True, separators=(",", ":")))
        for k, v in bindings.items()
    ]
    path = f"{url}/protocols/{protocol}/runs"
    if params:
        path += "?" + urllib.parse.urlencode(params)
    return _request("GET", path, key)


def runs(*, label: str | None = None, label_contains: str | None = None,
         protocol: str | None = None, project: str | None = None,
         owner: str | None = None, limit: int | None = None,
         api_url: str | None = None, api_key: str | None = None) -> list[dict]:
    """Runs newest first (`GET /runs`), by exact `label` or
    `label_contains`, `protocol` id, `project` (`owner/project` or id),
    or an `owner`'s (yours by default). Each row carries its job, status,
    result path, protocol and compute versions, spend and label."""
    import urllib.parse

    url, key = _config(api_url, api_key)
    query = {"label": label, "labelContains": label_contains, "protocol": protocol,
             "project": project, "owner": owner, "limit": limit}
    qs = urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
    return _request("GET", f"{url}/runs{'?' + qs if qs else ''}", key)


def label_run(run: str, label: str | None, *, api_url: str | None = None,
              api_key: str | None = None) -> dict:
    """Relabel a run (its id or its job's), or clear it with None:
    `PATCH /runs/:id`. The change is in the job's history (`run.label`)."""
    url, key = _config(api_url, api_key)
    return _request("PATCH", f"{url}/runs/{run}", key,
                    body=json.dumps({"label": label}).encode(),
                    headers={"Content-Type": "application/json"})


def result(job: str | dict, node: str, *, api_url: str | None = None,
           api_key: str | None = None) -> Any:
    """One node's output, unwrapped.

    `job` is a job id, or any object carrying `resultPath` — a run row from
    `results_for`, or a job row from `get_job` — which is read directly,
    saving a round trip. The Emitted envelope is stripped (via `fetch`);
    what returns is the payload.
    """
    from collections.abc import Mapping

    if isinstance(job, Mapping):
        base = job.get("resultPath")
        if not base:
            raise BenchError("that run/job has no result path yet")
    else:
        row = get_job(job, api_url=api_url, api_key=api_key)
        base = row.get("resultPath")
        if not base:
            raise BenchError(
                f"job {job} has no result yet (status {row.get('status')})")
    return fetch(f"{base}/{node}", api_url=api_url, api_key=api_key)

