from __future__ import annotations

from collections import namedtuple


class Memo:
    _Memo = namedtuple("_Memo", "label tape existing")

    def _open_memo(self, params):
        label = params.get("cache")
        if not label:
            return None
        if label is True:
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
        except Exception:  # noqa: BLE001
            tape, existing = Cassette(provider="", label=label), 0
        return self._Memo(label, tape, existing)

    def _close_memo(self, memo, out):
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
        except Exception as e:  # noqa: BLE001
            if isinstance(summary, dict):
                summary.setdefault("cache", {})["store_error"] = str(e)[:200]
        return out
