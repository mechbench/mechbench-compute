"""A node's memo of the remote calls it made.

`cache: "<bench label>"` on a node opens a cassette, so a re-run pays a
provider only for what it has not asked before. The label is explicit, or
derived from the protocol's identity and the node's id — never from the
node fingerprint, which every compute release changes, where the memo is
worth keeping across them.
"""

from __future__ import annotations

from collections import namedtuple


class Memo:
    """Memo: see this module's docstring."""

    #: An opened memo: the label it came from, and the cassette that
    #: will be written back to it.
    _Memo = namedtuple("_Memo", "label tape existing")

    def _open_memo(self, params):
        """Load this node's memo of remote calls, if it asked for one.

        `cache: "<bench label>"` — an explicit label, not a derivation
        from the node's identity. A memo keyed on node identity would
        be thrown away by every compute release, which is exactly
        backwards: the compute version is not part of what a provider
        was asked, and the REQUEST hash inside the memo is what
        decides a hit. A named memo survives a version bump, which is
        the point of having one.
        """
        label = params.get("cache")
        if not label:
            return None
        if label is True:
            # Derived from the PROTOCOL's identity and the node's id —
            # both stable across compute releases, which a node
            # fingerprint is not. The request hash inside the memo is
            # what decides a hit; this only decides where the memo
            # lives. Reconsidered 2026-09-11: refusing `cache: true` and
            # demanding a name was friction for no gain, since the job
            # spec already carries the identity needed.
            pid, _ = getattr(self, "_protocol_ref", (None, None))
            nid = (getattr(self, "_current", None) or {}).get("nid") \
                if hasattr(self, "_current") else None
            nid = nid or params.get("_nid")
            if not pid or not nid:
                raise ValueError(
                    "`cache: true` needs the run's protocol id and the node "
                    "id to derive a label, and this execution has neither "
                    "(a bare ProtocolSpec with no protocolId). Name it: "
                    '`cache: "<owner>/<project>/memos/<name>"`.')
            owner = params.get("_owner") or "memos"
            label = f"{owner}/memos/{pid}/{nid}"
        from mechbench_compute import bench
        from mechbench_compute.providers.cassette import Cassette

        label = str(label)
        try:
            obj = bench.fetch(label)
            payload = obj.get("payload", obj)
            tape = Cassette.from_wire(payload)
            existing = tape.n_responses
        except Exception:  # noqa: BLE001 — no memo yet is the ordinary case
            tape, existing = Cassette(provider="", label=label), 0
        return self._Memo(label, tape, existing)

    def _close_memo(self, memo, out):
        """Write the memo back, and say what it saved.

        Emitted even when nothing new was recorded: a run that was a
        complete hit is exactly the run worth being able to point at.
        """
        from mechbench_compute import bench

        added = memo.tape.n_responses - memo.existing
        summary = out.get("summary")
        if isinstance(summary, dict):
            calls = summary.get("calls") or 0
            summary["cache"] = {
                "label": memo.label,
                "hits": max(0, calls - added),
                "recorded": added,
                "entries": memo.tape.n_responses,
            }
        try:
            bench.emit(memo.label, memo.tape.to_wire(),
                       operation="chat/memo")
        except Exception as e:  # noqa: BLE001 — a run must not fail on its memo
            if isinstance(summary, dict):
                summary.setdefault("cache", {})["store_error"] = str(e)[:200]
        return out
