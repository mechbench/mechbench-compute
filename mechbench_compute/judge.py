"""`~canonical/ops/judge/1` — model-graded scoring (task 000356, epic
000334).

The first real consumer of hosted models on the bench: a judge reads
records (or transcripts) against a rubric and returns a score, a label,
or a preference — with its reasoning, its provenance, and its cost.

Three commitments make a judge's numbers usable rather than merely
available:

**Votes, not a verdict.** `n_votes` repeats the call and aggregates —
mean for numeric scales, majority for categorical and pairwise — and
the object records the SPREAD. A judge that disagrees with itself is
telling you something about the rubric, and averaging that away would
throw out the finding.

**Position is randomized and recorded.** In pairwise mode the A/B order
flips per vote from a key-derived seed, and each vote records the order
it saw. Position bias is real and large; a judge that always picks A is
a result you can only see if you looked.

**Parsing is honest.** The judge is asked for JSON and read leniently
(a bare number, a quoted label), but a vote that could not be parsed is
recorded as unparsed rather than silently scored — an unreadable answer
is data about the rubric, not a zero.

Everything rides the chat block, so a judge inherits the budget cap,
the concurrency bound, item spooling and per-call provenance for free,
and a LOCAL judge (Gemma through the same path) is the cheap first
test rather than a special case.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute import chat as chat_mod

SCALES = ("numeric", "categorical", "pairwise")

_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)
_FIRST_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


class Scale:
    """What the judge is being asked for, and how to read the answer."""

    def __init__(self, spec: Mapping[str, Any] | None) -> None:
        spec = dict(spec or {})
        self.kind = str(spec.get("kind", "numeric"))
        if self.kind not in SCALES:
            raise ValueError(f"unknown scale {self.kind!r} — one of {SCALES}")
        rng = spec.get("range")
        self.low = float(spec.get("min", rng[0] if rng else 1))
        self.high = float(spec.get("max", rng[1] if rng else 5))
        if self.kind == "numeric" and self.high <= self.low:
            raise ValueError(f"a numeric scale needs max > min, got {self.low}–{self.high}")
        self.labels = [str(x) for x in (spec.get("labels") or [])]
        if self.kind == "categorical" and len(self.labels) < 2:
            raise ValueError("a categorical scale needs at least two labels")

    def instruction(self) -> str:
        if self.kind == "numeric":
            return (f"Answer with JSON: {{\"score\": <number from {self.low:g} to "
                    f"{self.high:g}>, \"rationale\": \"<one sentence>\"}}.")
        if self.kind == "categorical":
            options = ", ".join(f'"{x}"' for x in self.labels)
            return (f"Answer with JSON: {{\"label\": <one of {options}>, "
                    f"\"rationale\": \"<one sentence>\"}}.")
        return ('Answer with JSON: {"winner": "A" or "B", '
                '"rationale": "<one sentence>"}.')

    def read(self, text: str) -> dict[str, Any]:
        """Parse one vote. Returns `{}` when nothing could be read —
        the caller records that as unparsed rather than as a score."""
        payload = _json_object(text)
        if self.kind == "numeric":
            value = payload.get("score") if payload else None
            if value is None:
                m = _FIRST_NUMBER.search(text)
                value = m.group(0) if m else None
            try:
                score = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return {}
            # Out-of-range is clamped and FLAGGED: a judge that answers
            # 9 on a 1–5 scale did not mean 5, and pretending otherwise
            # would hide a broken rubric.
            clamped = min(max(score, self.low), self.high)
            out: dict[str, Any] = {"score": clamped,
                                   "rationale": _rationale(payload, text)}
            if clamped != score:
                out["out_of_range"] = score
            return out
        if self.kind == "categorical":
            label = str(payload.get("label", "")) if payload else ""
            if label not in self.labels:
                label = next((x for x in self.labels
                              if re.search(rf"\b{re.escape(x)}\b", text, re.IGNORECASE)), "")
            if not label:
                return {}
            return {"label": label, "rationale": _rationale(payload, text)}
        winner = str(payload.get("winner", "")).strip().upper()[:1] if payload else ""
        if winner not in ("A", "B"):
            m = re.search(r"\b([AB])\b", text.upper())
            winner = m.group(1) if m else ""
        if not winner:
            return {}
        return {"winner": winner, "rationale": _rationale(payload, text)}


def _json_object(text: str) -> dict[str, Any]:
    m = _JSON_OBJECT.search(text)
    if not m:
        return {}
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _rationale(payload: Mapping[str, Any], text: str) -> str:
    value = payload.get("rationale") or payload.get("reason") or ""
    return str(value) if value else text.strip()[:400]


def render_subject(rec: Mapping[str, Any], fields: Sequence[str]) -> str:
    """What the judge is shown. Named fields only — a judge that can
    see the condition labels is grading the labels."""
    parts = []
    for f in fields:
        value = rec.get(f)
        if value is None and isinstance(rec.get("metadata"), Mapping):
            value = rec["metadata"].get(f)
        if value is None:
            continue
        parts.append(f"{f}:\n{value}" if len(fields) > 1 else str(value))
    return "\n\n".join(parts)


def build_prompts(records: Sequence[Mapping[str, Any]], *, scale: Scale,
                  rubric: str, fields: Sequence[str], n_votes: int,
                  seed: Any, pairwise_fields: Sequence[str] = ()) -> list[dict[str, Any]]:
    """One chat record per (subject, vote). The vote index rides in the
    id so the chat block's own item keys stay unique and stable, which
    is what makes a judged corpus resumable."""
    from mechbench_compute.seeds import item_seed

    out: list[dict[str, Any]] = []
    for rec in records:
        rid = str(rec.get("id", ""))
        for k in range(n_votes):
            body: dict[str, Any] = {
                "id": f"{rid}:v{k}",
                "coords": {**(rec.get("coords") or {}), "subject": rid, "vote": k},
                "system": rubric,
            }
            if scale.kind == "pairwise":
                a_field, b_field = pairwise_fields
                # The order flips per vote from a key-derived seed, and
                # the vote records what it saw: position bias is real,
                # and only visible if it was randomized on purpose.
                flipped = bool(item_seed(seed, rid, k) % 2)
                first, second = ((b_field, a_field) if flipped
                                 else (a_field, b_field))
                body["order"] = "BA" if flipped else "AB"
                body["user"] = (
                    f"A:\n{rec.get(first, '')}\n\nB:\n{rec.get(second, '')}\n\n"
                    f"{scale.instruction()}")
            else:
                body["user"] = (f"{render_subject(rec, fields)}\n\n"
                                f"{scale.instruction()}")
            out.append(body)
    return out


def aggregate(subject: Mapping[str, Any], votes: Sequence[Mapping[str, Any]], *,
              scale: Scale) -> dict[str, Any]:
    """One subject's verdict from its votes, with the spread kept."""
    parsed = [v for v in votes if v.get("parsed")]
    row: dict[str, Any] = {
        "id": subject.get("id"),
        "coords": dict(subject.get("coords") or {}),
        "n_votes": len(votes),
        "n_parsed": len(parsed),
        "votes": [dict(v) for v in votes],
    }
    if not parsed:
        row["unparsed"] = True
        return row
    if scale.kind == "numeric":
        scores = [float(v["score"]) for v in parsed]
        row["score"] = round(statistics.fmean(scores), 4)
        row["spread"] = (round(statistics.stdev(scores), 4) if len(scores) > 1
                         else 0.0)
        row["min"] = min(scores)
        row["max"] = max(scores)
    else:
        key = "label" if scale.kind == "categorical" else "winner"
        counts: dict[str, int] = {}
        for v in parsed:
            counts[str(v[key])] = counts.get(str(v[key]), 0) + 1
        winner, top = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        row[key] = winner
        row["counts"] = counts
        # How much the judge agreed with itself. A rubric that produces
        # 0.5 here is the finding, not a number to average away.
        row["agreement"] = round(top / len(parsed), 4)
    row["rationale"] = str(parsed[0].get("rationale", ""))
    return row


def summarize(rows: Sequence[Mapping[str, Any]], *, scale: Scale,
              votes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The aggregate a publication cites, plus the diagnostics that say
    whether to trust it."""
    scored = [r for r in rows if not r.get("unparsed")]
    out: dict[str, Any] = {
        "scale": scale.kind,
        "n_subjects": len(rows),
        "n_unparsed": sum(1 for r in rows if r.get("unparsed")),
    }
    if scale.kind == "numeric" and scored:
        values = [float(r["score"]) for r in scored]
        out["mean"] = round(statistics.fmean(values), 4)
        out["median"] = round(statistics.median(values), 4)
        if len(values) > 1:
            out["stdev"] = round(statistics.stdev(values), 4)
    elif scored:
        key = "label" if scale.kind == "categorical" else "winner"
        counts: dict[str, int] = {}
        for r in scored:
            counts[str(r[key])] = counts.get(str(r[key]), 0) + 1
        out["counts"] = counts
        out["mean_agreement"] = round(
            statistics.fmean([float(r.get("agreement", 0.0)) for r in scored]), 4)
    if scale.kind == "pairwise":
        # The position-bias diagnostic: how often the FIRST-SHOWN
        # option won, across every vote. 0.5 is the honest number.
        shown_first = [v for v in votes if v.get("parsed")]
        if shown_first:
            first_wins = sum(
                1 for v in shown_first
                if (v.get("winner") == "A") == (v.get("order", "AB") == "AB")
            )
            out["first_shown_win_rate"] = round(first_wins / len(shown_first), 4)
    return out


def run(params: Mapping[str, Any], *, inputs: Mapping[str, Any] | None = None,
        secrets=None, limiter=None, job_budget=None, model=None,
        on_item=None, on_start=None, resume_items=None) -> dict[str, Any]:
    """Judge a record set. `model` is a loaded local model when the
    judge runs on local weights; endpoints need nothing but credentials."""
    from mechbench_compute import model_ref as mr

    inputs = inputs or {}
    judge = dict(params.get("judge") or {})
    if "model" not in judge:
        raise ValueError(
            "a judge node needs `judge: {model, system}` — who is grading is "
            "the first thing a reader will ask")
    scale = Scale(params.get("scale"))
    fields = [str(f) for f in (params.get("fields") or ["text"])]
    pairwise_fields = [str(f) for f in (params.get("pairwise_fields")
                                        or ["text_a", "text_b"])]
    if scale.kind == "pairwise" and len(pairwise_fields) != 2:
        raise ValueError("pairwise judging needs exactly two `pairwise_fields`")
    n_votes = max(1, int(params.get("n_votes", 1)))
    seed = params.get("seed", 0)
    rubric = "\n\n".join(x for x in (str(judge.get("system", "")),
                                     str(params.get("rubric", ""))) if x)
    if not rubric:
        raise ValueError(
            "a judge node needs a rubric — an unstated standard is not a "
            "measurement")

    subjects = chat_mod._records(inputs.get("records") or params.get("records") or [])
    prompts = build_prompts(subjects, scale=scale, rubric=rubric, fields=fields,
                            n_votes=n_votes, seed=seed,
                            pairwise_fields=pairwise_fields)

    ref = mr.parse(judge["model"])
    chat_params = {
        "model": judge["model"],
        "max_tokens": int(judge.get("max_tokens", 512)),
        "temperature": judge.get("temperature", 0.0),
        "seed": seed,
        "budget_usd": params.get("budget_usd") or judge.get("budget_usd"),
        "provider_options": dict(judge.get("provider_options") or {}),
        "concurrency": int(params.get("concurrency", 4)),
        "dry_run": bool(params.get("dry_run", False)),
        "name": params.get("name", "judgements"),
    }
    if ref.is_endpoint:
        graded = chat_mod.run_remote(ref, prompts, chat_params, secrets=secrets,
                                     limiter=limiter, job_budget=job_budget,
                                     on_item=on_item, on_start=on_start,
                                     resume_items=resume_items)
    else:
        if model is None:
            raise ValueError(
                "a local judge needs the executor's loaded model — run this "
                "as a protocol node")
        graded = chat_mod.run_local(model, ref, prompts, chat_params,
                                    on_item=on_item, on_start=on_start,
                                    resume_items=resume_items)

    by_subject: dict[str, list[dict[str, Any]]] = {}
    order_by_id = {p["id"]: p.get("order", "AB") for p in prompts}
    all_votes: list[dict[str, Any]] = []
    for item in graded["items"]:
        coords = (item.get("metadata") or {}).get("coords") or {}
        subject_id = str(coords.get("subject", ""))
        read = scale.read(str(item.get("text", "")))
        vote: dict[str, Any] = {
            "vote": int(coords.get("vote", 0)),
            "parsed": bool(read),
            "order": order_by_id.get(item["id"].rsplit("-s", 1)[0], "AB"),
            **read,
        }
        call = (item.get("metadata") or {}).get("call")
        if call:
            vote["call"] = call
        by_subject.setdefault(subject_id, []).append(vote)
        all_votes.append(vote)

    rows = [aggregate(s, by_subject.get(str(s.get("id", "")), []), scale=scale)
            for s in subjects]
    out: dict[str, Any] = {
        "kind": "record_set",
        "name": params.get("name", "judgements"),
        "description": params.get("description", ""),
        "records": rows,
        "judge": {"model": ref.to_wire() if ref.is_endpoint else judge["model"],
                  "scale": scale.kind, "n_votes": n_votes,
                  "rubric": rubric[:2000]},
        "summary": summarize(rows, scale=scale, votes=all_votes),
    }
    if graded.get("spend"):
        out["spend"] = graded["spend"]
    return out
