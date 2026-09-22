from __future__ import annotations

import re
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute import chat as chat_mod
from mechbench_compute.judge.constants import FIRST_NUMBER, SCALES
from mechbench_compute.judge.parse_json_object import parse_json_object
from mechbench_compute.judge.read_rationale import read_rationale
from mechbench_compute.lexicon._base import In, Op, Output, P

_PROVIDER_OPTIONS_DOC = (
    "Provider-native request fields this block does not model, **keyed by "
    "provider** — `{\"anthropic\": {\"thinking\": {…}}}` — passed through "
    "as given. A key that names no provider is refused.")


OP = Op(
    name="eval/judge",
    requires="by-model",
    summary=(
        "Have a model grade each record against a rubric — a score, a label "
        "or an A/B preference — with repeated votes, the spread between "
        "them, and the position order randomised and recorded."
    ),
    description="""\
Each record's `text` (for a pairwise scale, its `text_a` and `text_b`) is
shown to the judge — that field and nothing else, so it cannot see the
condition labels — together with the rubric and an instruction to answer in
JSON. A record that carries the text under another name goes through
`records/rename` first. Three commitments make the numbers usable:

* **Votes, not a verdict.** `n_votes` repeats the call. Numeric scores
  are averaged and their spread kept; labels and preferences take the
  majority, and `agreement` records how often the judge agreed with itself.
  A rubric that produces 0.5 there is a finding.
* **Position is randomised, recorded, and undone.** In pairwise mode the
  A/B order flips per vote from a seeded coin; each vote records the
  order it saw and the letter it answered, and the answer is mapped back
  to the record's own `text_a`/`text_b` before anything counts it. The
  summary reports how often the option shown FIRST won across every vote
  — 0.5 is the honest number, and 1.0 is a judge with no opinion about
  the writing.
* **Parsing is honest.** A vote that could not be read is recorded as
  unparsed rather than scored; a numeric answer outside the scale is
  clamped and flagged.
* **An empty subject is not judged.** A record whose judged field is
  missing or blank is refused by name, because a winner over an empty
  string looks exactly like every other winner in the column. This is
  reachable: a `records/zip` with `on_missing: "placeholder"` keeps the
  key of a branch that failed. `on_missing: "skip"` keeps those records
  as unjudged rows and grades the rest.

The judge runs through `chat`, so it inherits the budget cap, concurrency,
resumability and per-call provenance; a local model is the cheap first test.
""",
    inputs=(
        In("records", "records/record",
           "The subjects to grade, each with `text` — or `text_a` and "
           "`text_b` for a pairwise scale. A document collection is read the "
           "same way.", many=True),
    ),
    output=Output('eval/verdict', collection=True, doc='One item per subject: `id`, `coords`, the verdict (`score`/`spread`/`min`/`max`, or `label`/`counts`/`agreement`, or `winner`/`counts`/`agreement`), `rationale`, `n_votes`, `n_parsed`, every `vote`, `unparsed: true` when no vote could be read, and `unjudged: true` with the `missing` field names when there was nothing to judge. The header carries `judge` (who graded and how), `summary` (mean/median/stdev or counts, `n_unparsed`, `n_unjudged` and which, `first_shown_win_rate` for pairwise) and `spend`.'),
    params=(
        P("judge", "object",
          "Who grades: `{\"model\": …, \"system\": rubric, \"max_tokens\": "
          "512, \"temperature\": …, \"budget_usd\": …, "
          "\"provider_options\": …}`. `model` is required; `rubric`, when "
          "given, is appended to `system`. `temperature` is sent only if "
          "you name one — a judge's steadiness comes from `n_votes` and "
          "is reported as `agreement`, and some models refuse the "
          "parameter outright.", fields=(
              P("model", "model", "The judge's model."),
              P("system", "string", "The judge's system prompt; `rubric` is appended to it.", ""),
              P("max_tokens", "int", "The longest verdict, in tokens.", 512),
              P("temperature", "float", "Sampling temperature, sent only when named.", None),
              P("budget_usd", "float", "The spend cap, when the node sets none.", None),
              P("provider_options", "map[string, map[string, json]]", _PROVIDER_OPTIONS_DOC, None),
          )),
        P("rubric", "string",
          "The standard the judge applies, appended to `judge.system`. One "
          "of the two must be present — an unstated standard is not a "
          "measurement.",
          ""),
        P("scale", "object",
          "What the judge answers with: `{\"type\": \"numeric\", \"min\": 1, "
          "\"max\": 5}` (or `\"range\": [1, 5]`); `{\"type\": "
          "\"categorical\", \"labels\": [...]}`; or `{\"type\": "
          "\"pairwise\"}`.",
          {"type": "numeric", "min": 1, "max": 5}, fields=(
              P("type", "string", "What kind of answer.", "numeric",
                choices=("numeric", "categorical", "pairwise")),
              P("kind", "string", "The older spelling of `type`; read when `type` is absent.", None,
                choices=("numeric", "categorical", "pairwise")),
              P("min", "float", "For `numeric`: the lowest score. Defaults to `range[0]`, else 1.", None),
              P("max", "float", "For `numeric`: the highest score. Defaults to `range[1]`, else 5.", None),
              P("range", "list[float]", "For `numeric`: `[min, max]` in one field.", None),
              P("labels", "list[string]", "For `categorical`: the labels, at least two.", None),
          )),
        P("n_votes", "int", "How many times each subject is judged.", 1),
        P("budget_usd", "float",
          "The most this node may spend on provider calls, in US dollars. "
          "Required when the model is a hosted endpoint; the node stops "
          "with what it has when the cap is reached. A job-level cap, if "
          "one is set, bounds it further.",
          None),
        P("concurrency", "int",
          "How many judge requests are in flight at once (remote judges).",
          4),
        P("on_missing", "string",
          "A record whose judged field is missing or blank: `\"error\"` "
          "refuses it by name; `\"skip\"` keeps it as an unjudged row, "
          "naming what was absent, and grades the rest.",
          "error", choices=("error", "skip")),
    ),
    example={
        "judge": {"model": {"provider": "anthropic", "model": "claude-sonnet-5"},
                  "system": "You grade short stories for originality."},
        "rubric": "1 = a stock plot told plainly; 5 = a premise you have not seen before.",
        "scale": {"type": "numeric", "min": 1, "max": 5},
        "n_votes": 3,
        "budget_usd": 3.0,
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/stories"}}},
)


def run(ctx, inputs, params):
    """eval/judge (task 000356): model-graded scoring.
    A local judge is the cheap first test, so it loads here through
    the same path as any other model block; an endpoint judge needs
    only its credentials and its cap."""
    from mechbench_compute import model_ref as model_ref_mod

    spec = dict(params.get("judge") or {})
    ref = model_ref_mod.parse(spec.get("model")) if spec.get("model") else None
    model = None
    if ref is not None and not ref.is_endpoint:
        model = ctx.model(ref)
    return run_judge(params, inputs=inputs, secrets=ctx.secrets,
                         limiter=ctx.executor._limiter, job_budget=ctx.executor._budget,
                         model=model, on_item=ctx.on_item, on_start=ctx.on_start,
                         resume_items=ctx.resume_items)



class Scale:
    """What the judge is being asked for, and how to read the answer."""

    def __init__(self, spec: Mapping[str, Any] | None) -> None:
        spec = dict(spec or {})
        # `type` names the scale; `kind` is the retired spelling.
        self.kind = str(spec.get("type") or spec.get("kind") or "numeric")
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
        payload = parse_json_object(text)
        if self.kind == "numeric":
            value = payload.get("score") if payload else None
            if value is None:
                m = FIRST_NUMBER.search(text)
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
                                   "rationale": read_rationale(payload, text)}
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
            return {"label": label, "rationale": read_rationale(payload, text)}
        winner = str(payload.get("winner", "")).strip().upper()[:1] if payload else ""
        if winner not in ("A", "B"):
            m = re.search(r"\b([AB])\b", text.upper())
            winner = m.group(1) if m else ""
        if not winner:
            return {}
        return {"winner": winner, "rationale": read_rationale(payload, text)}


def read_subject_coords(rec: Mapping[str, Any]) -> dict[str, Any]:
    """A subject's coordinates, wherever they live. A generate node's
    items carry them under `metadata`, a record set at the top level;
    judging must not lose them either way, because slicing a judged
    corpus by the condition that produced it is the whole point."""
    top = rec.get("coords")
    if isinstance(top, Mapping):
        return dict(top)
    meta = rec.get("metadata")
    if isinstance(meta, Mapping) and isinstance(meta.get("coords"), Mapping):
        return dict(meta["coords"])
    return {}


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
                "coords": {**read_subject_coords(rec), "subject": rid, "vote": k},
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
        "coords": read_subject_coords(subject),
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
    # A pairwise rationale talks about "A" and "B" as the judge saw
    # them, so prefer one written under the unswapped order — otherwise
    # its letters mean the opposite of the row's, and say so.
    spoke = next((v for v in parsed if v.get("order", "AB") == "AB"), parsed[0])
    row["rationale"] = str(spoke.get("rationale", ""))
    if scale.kind == "pairwise" and spoke.get("order") == "BA":
        row["rationale_order"] = "BA"
    return row


def summarize(rows: Sequence[Mapping[str, Any]], *, scale: Scale,
              votes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The aggregate a publication cites, plus the diagnostics that say
    whether to trust it."""
    scored = [r for r in rows
              if not r.get("unparsed") and not r.get("unjudged")]
    out: dict[str, Any] = {
        "scale": scale.kind,
        "n_subjects": len(rows),
        "n_unparsed": sum(1 for r in rows if r.get("unparsed")),
    }
    unjudged = [r for r in rows if r.get("unjudged")]
    if unjudged:
        # Named, not just counted: which subjects went ungraded is the
        # question a reader of the number will ask next.
        out["n_unjudged"] = len(unjudged)
        out["unjudged"] = [str(r.get("id")) for r in unjudged][:50]
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


def run_judge(params: Mapping[str, Any], *, inputs: Mapping[str, Any] | None = None,
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
    # The judge reads `text` — `text_a`/`text_b` for a pairwise scale —
    # and nothing else. A record that carries the text under another
    # name goes through records/rename first: the graph shows the move.
    fields = ["text"]
    pairwise_fields = ["text_a", "text_b"]
    n_votes = max(1, int(params.get("n_votes", 1)))
    seed = params.get("seed", 0)
    rubric = "\n\n".join(x for x in (str(judge.get("system", "")),
                                     str(params.get("rubric", ""))) if x)
    if not rubric:
        raise ValueError(
            "a judge node needs a rubric — an unstated standard is not a "
            "measurement")

    from mechbench_compute.lexicon import kinds as K

    subjects = K.items_of(inputs.get("records") or [])
    # A subject with nothing to read is not a hard subject; it is not a
    # subject. Judging it would produce a winner over an empty string —
    # a number that looks like every other number in the column. This is
    # reachable: a `records/zip` with `on_missing: "placeholder"` keeps
    # the key of a branch that failed, and the missing side arrives here
    # as an absent field.
    want = list(pairwise_fields if scale.kind == "pairwise" else fields)
    on_missing = str(params.get("on_missing", "error"))
    if on_missing not in ("error", "skip"):
        raise ValueError(
            f"on_missing is 'error' or 'skip', not {on_missing!r}")
    def absent(rec):
        return [f for f in want if not str(rec.get(f, "")).strip()]

    empty = {id(s): absent(s) for s in subjects if absent(s)}
    if empty and on_missing == "error":
        names = ", ".join(repr(str(s.get("id"))) for s in subjects
                          if id(s) in empty)
        raise ValueError(
            f"{len(empty)} record(s) have no {' and '.join(want)} to judge "
            f"({names[:120]}). An empty side would be scored against a real "
            f"one. `on_missing: \"skip\"` records them as unjudged and "
            f"grades the rest.")
    judged = [s for s in subjects if id(s) not in empty]
    prompts = build_prompts(judged, scale=scale, rubric=rubric, fields=fields,
                            n_votes=n_votes, seed=seed,
                            pairwise_fields=pairwise_fields)

    ref = mr.parse(judge["model"])
    chat_params = {
        "model": judge["model"],
        "max_tokens": int(judge.get("max_tokens", 512)),
        # Sent only when the author asks for one. A judge's steadiness
        # is bought with `n_votes` and reported as `agreement`, not
        # assumed from a sampling parameter — and a parameter some
        # models now REFUSE outright (claude-sonnet-5 answers HTTP 400,
        # "`temperature` is deprecated for this model") cannot be a
        # silent default: it made those models unusable as judges.
        "temperature": judge.get("temperature"),
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
    from mechbench_compute.lexicon import kinds as K

    for item in K.items_of(graded):
        coords = (item.get("metadata") or {}).get("coords") or {}
        subject_id = str(coords.get("subject", ""))
        read = dict(scale.read(str(item.get("text", ""))))
        order = order_by_id.get(item["id"].rsplit("-s", 1)[0], "AB")
        # The judge answers about what it SAW, and half the time it saw
        # the sides swapped. Map the answer back to the record's own
        # `text_a`/`text_b` before anything counts it, keeping the seen
        # label beside it. Without this the position randomisation
        # scrambled the result it was supposed to make trustworthy: a
        # judge that picked the same passage every time was reported as
        # disagreeing with itself, and `first_shown_win_rate` — the
        # diagnostic for exactly this — was computed from labels that
        # had never been mapped.
        if scale.kind == "pairwise" and read.get("winner"):
            read["shown_winner"] = read["winner"]
            if order == "BA":
                read["winner"] = "B" if read["winner"] == "A" else "A"
        vote: dict[str, Any] = {
            "vote": int(coords.get("vote", 0)),
            "parsed": bool(read),
            "order": order,
            **read,
        }
        call = (item.get("metadata") or {}).get("call")
        if call:
            vote["call"] = call
        by_subject.setdefault(subject_id, []).append(vote)
        all_votes.append(vote)

    # Skipped subjects keep their row — unjudged and saying what was
    # missing. A table with a gap in it is the finding; a table that
    # quietly lost the row is a smaller corpus with no note of why.
    rows = [(aggregate(s, by_subject.get(str(s.get("id", "")), []), scale=scale)
             if id(s) not in empty else
             {"id": s.get("id"), "coords": read_subject_coords(s), "unjudged": True,
              "missing": empty[id(s)]})
            for s in subjects]
    return K.collection(
        "eval/verdict", rows,
        name=params.get("name", "judgements"),
        description=params.get("description", ""),
        judge={"model": ref.to_wire() if ref.is_endpoint else judge["model"],
               "scale": scale.kind, "n_votes": n_votes,
               "rubric": rubric[:2000]},
        summary=summarize(rows, scale=scale, votes=all_votes),
        spend=graded.get("spend") or None,
    )
