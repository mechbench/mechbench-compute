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


def _config(api_url: str | None, api_key: str | None) -> tuple[str, str]:
    url = api_url or _DEFAULTS.get("url") or os.environ.get("MECHBENCH_API_URL", "")
    key = api_key or _DEFAULTS.get("key") or os.environ.get("MECHBENCH_API_KEY", "")
    if not url:
        raise BenchError(
            "no API url: set MECHBENCH_API_URL, call "
            "mechbench_compute.bench.configure(api_url=...), or pass api_url=")
    if not key:
        raise BenchError(
            "no API key: set MECHBENCH_API_KEY, call "
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
             return_headers: bool = False) -> Any:
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {key}")
    for h, v in (headers or {}).items():
        req.add_header(h, v)
    try:
        with urllib.request.urlopen(req, timeout=60, context=_tls(url)) as resp:
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


def fetch(target: str, *, api_url: str | None = None,
          api_key: str | None = None, with_meta: bool = False) -> Any:
    """Fetch an object; CBOR objects are decoded, JSON parsed, other
    mime types returned as bytes. ``with_meta=True`` returns
    ``(payload, meta)`` where meta carries the server's
    ``content_hash`` (task 000260 — record what resolved)."""
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
        except Exception:
            return (bytes(raw), meta) if with_meta else bytes(raw)
    return (raw, meta) if with_meta else raw


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

