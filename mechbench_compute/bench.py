"""Bench client — emit results from local experiments to the workspace
(task 000238; docs/THE_BENCH.md §3).

The researcher's front door to the bench is a local call, not the job
queue:

    from mechbench_compute import bench

    bench.emit(
        "benji/my-project/results/ladder-2026-08-17",
        {"kind": "ladder", "rows": [...]},
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
    """A bench API call failed; the message carries the server detail."""


#: Set by a host that already holds credentials — see `configure()`.
_DEFAULTS: dict[str, str] = {}


def configure(*, api_url: str | None = None, api_key: str | None = None) -> None:
    """Tell this module where the API is, without using the environment.

    Research scripts set MECHBENCH_API_URL/MECHBENCH_API_KEY and that
    stays the default. A *host* embedding this layer — mechbench-runner —
    resolves credentials its own way: since `login` they live in
    ~/.mechbench/config.toml, not the environment, and before this hook
    existed a pipeline could execute perfectly and then fail on its first
    emit with "no API url".

    Explicit arguments to a call still win over anything set here.
    """
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
    imports neither (task 000450). In order: explicit argument, then
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
    """A TLS context with CA roots that exist.

    `urllib` uses OpenSSL's default trust store, and a Python installed
    by uv or from python.org on macOS has none — the default context
    holds *zero* certificates. Every https:// call then fails with
    CERTIFICATE_VERIFY_FAILED while anything using httpx keeps working,
    because httpx bundles certifi.

    The same asymmetry hit mechbench-runner's websocket client. Two
    modules in this stack talk TLS without httpx; both now say so
    explicitly.
    """
    if not url.startswith("https://"):
        return None
    import ssl

    import certifi

    return ssl.create_default_context(cafile=certifi.where())


def _request(method: str, url: str, key: str, body: bytes | None = None,
             headers: dict[str, str] | None = None,
             return_headers: bool = False, timeout: float = 60) -> Any:
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {key}")
    for h, v in (headers or {}).items():
        req.add_header(h, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_tls(url)) as resp:
            raw = resp.read()
            ctype = resp.headers.get("content-type", "")
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        raise BenchError(f"{method} {url} -> {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise BenchError(f"{method} {url} unreachable: {e.reason}") from None
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
        # Validate against the schema model before sending.
        envelope = ms.Emitted(provenance=ms.Provenance(**prov),
                              payload=payload)
        # mode="python", not "json": binary payloads (adapter
        # safetensors bytes, 000259) must survive to CBOR, which
        # encodes bytes natively. Canonical bytes are identical for
        # JSON-safe payloads (verified), so no drift for existing
        # objects.
        body_obj = envelope.model_dump(mode="python")

    body = ms.dump_canonical(body_obj)
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
    """Fetch an object and return its PAYLOAD (task 000450).

    CBOR objects are decoded, JSON parsed, other mime types returned as
    bytes. An Emitted envelope is unwrapped — the payload is what a reader
    wants, and stripping it by hand
    (`(lambda o: o.get("payload", o))(...)`) was the idiom in every
    experiment. Use `fetch_envelope()` for the rare read that needs the
    provenance. ``with_meta=True`` returns ``(payload, meta)`` where meta
    carries the server's ``content_hash`` (task 000260)."""
    got = _fetch_decoded(target, api_url, api_key, with_meta)
    if with_meta:
        obj, meta = got
        return _unwrap_envelope(obj), meta
    return _unwrap_envelope(got)


def fetch_envelope(target: str, *, api_url: str | None = None,
                   api_key: str | None = None, with_meta: bool = False) -> Any:
    """The object exactly as stored — the Emitted envelope with its
    provenance, not just the payload. What `fetch` returned before task
    000450; for the caller that wants lineage, params fingerprint, or the
    producing tool version."""
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
    """Upload one raw file as a binary object (000312 Arc C).

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
    prefix — what retry-as-resume consults before uploading (000312
    Arc C). Best-effort: an empty dict just means upload everything."""
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
# The library under the three CLI verbs (task 000450): launch a protocol,
# watch its jobs, find a run by what it ran, read a node's result. Before
# this, every experiment wrote its own `api()` over httpx — config,
# credential and transport all borrowed from a module whose job is protocol
# graphs. The `mechbench run/watch/result` verbs (mechbench-runner, task
# 000448) are thin wrappers over these; there is one implementation.

#: A job is finished — successfully or not — in exactly these states.
TERMINAL = ("done", "failed", "cancelled", "interrupted")


def launch(protocol: str, bindings: dict[str, Any] | None = None, *,
           budget: float | None = None, api_url: str | None = None,
           api_key: str | None = None) -> dict:
    """Bind a protocol and queue its job. `POST /protocols/:ref/runs`.

    Returns the bare run, with `id` and `jobId` on it (task 000451) —
    record the job id at once; a job id in a scrollback is a job id lost.
    """
    url, key = _config(api_url, api_key)
    body: dict[str, Any] = {"bindings": dict(bindings or {})}
    if budget is not None:
        body["budgetUsd"] = budget
    return _request(
        "POST", f"{url}/protocols/{protocol}/runs", key,
        body=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, timeout=90)


def create_protocol(owner: str, project: str, name: str, *, graph: dict,
                    description: str = "", signature: dict | None = None,
                    owner_kind: str = "user", api_url: str | None = None,
                    api_key: str | None = None) -> dict:
    """Register a protocol: `POST /protocols` with its graph and
    signature. Returns the bare protocol (id, version, name, ...).

    The authoring half of an experiment used to carry its own `api()`
    for exactly this call; it belongs beside `launch`, which runs what
    this registers.
    """
    url, key = _config(api_url, api_key)
    body: dict[str, Any] = {
        "ownerKind": owner_kind, "ownerHandle": owner, "projectSlug": project,
        "name": name, "description": description, "graph": graph,
    }
    if signature is not None:
        body["signature"] = signature
    out = _request(
        "POST", f"{url}/protocols", key,
        body=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, timeout=90)
    # The protocols routes still answer `{protocol: …}` — the one wrapper
    # 000451 left behind (task 000456). Unwrapped here, once, so no
    # caller has to; drop this line when the route goes bare.
    return out.get("protocol", out) if isinstance(out, dict) else out


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
    job's progress CHANGES — never the same state twice, so a long run does
    not bury its interesting moment under identical lines. A transient fetch
    error yields `(job_id, {"status": None, "error": <str>})` and the poll
    continues; the job is retried next round. The generator is exhausted
    once every job is terminal.

    It prints nothing: the caller renders (the CLI) or collects the final
    states (`last = dict(bench.watch(jobs))` keeps the terminal one per job,
    since each job's last yield is its terminal state).
    """
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
    (`GET /protocols/:ref/runs?binding.k=v`, task 000449). Each carries
    its `jobId`, `jobStatus` and `resultPath` — the end of the job-id
    sidecars an experiment used to maintain by hand.

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


def result(job: str | dict, node: str, *, api_url: str | None = None,
           api_key: str | None = None) -> Any:
    """One node's output, unwrapped (task 000450).

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

