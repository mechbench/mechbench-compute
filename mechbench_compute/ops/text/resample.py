from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute.api import (
    DEFAULT_OUTPUT,
    In,
    Op,
    Output,
    P,
    Resume,
    collection,
    derive_seed,
    encode,
    items_of,
    prefill_decision,
    read_last_logp,
    read_metric,
    render,
    resolve_outcomes,
    serialize_model,
)

WHERE = ("position", "sentence", "turn")

BOUNDARIES = (".", "!", "?", "\n")

OP = Op(
    name="text/resample",
    needs=frozenset({"model.forward", "model.sample"}),
    resume=Resume("restart"),
    summary=(
        "Branch k continuations at each point of a generation and report how far the outcome they reach "
        "moves from one point to the next: which text the answer hinged on."
    ),
    description="""\
Each record is generated once, seeded, as `text/generate` would: that is
the full generation, and its outcome is read at its end. Then at each
branch point the op keeps what the model wrote before the point and
samples the rest again, `k` times, each branch with its own seeded
stream, and reads each branch's outcome at its end the same way.

`where` places the branch points. `position` puts one at every step:
step 0 samples the whole generation again, step *s* keeps the first *s*
tokens, and the last point keeps them all. `sentence` puts one at step
0, after every token whose text ends a sentence (one of `boundaries`),
and at the end. `turn` reads the record's `messages` as the user's turns
and plays them, the model writing a reply to each; a branch point at
turn *t* keeps the replies before it and writes reply *t* and every one
after it again, and the last point keeps them all.

The outcome is read from the next-token distribution at the end of a
generation, after `cue` when one is given (`" So the answer is"`), the
cue tokenized on its own. With the record's `outcomes`, a branch's
outcome is that distribution renormalised over them, as the
`entropy_outcomes` metric reads it, and a branch point's distribution is
the mean over its branches; the outcome named is the most likely of
them. Without `outcomes`, a branch's outcome is the token it would write
next, and a branch point's distribution is how many branches reached
each. `shift` is the Jensen-Shannon divergence in bits between a branch
point's distribution and the previous one's: what keeping the text
between them did to the outcome.

**Cost.** `k` generations per branch point, and a branch point at every
step under `position`: a 200-token generation at `k: 8` is 1,608
generations, each followed by one forward pass to read its outcome.
`max_positions` refuses a record whose branch points outnumber it,
before any branch is sampled; `sentence` is the cheap way to cover a
long generation. Under `turn` each branch writes every reply from its
turn on.
""",
    inputs=(
        In("records", "records/record",
           "Chat-shaped records, as `text/generate` reads them: `user`, `system`, `prefill`, an `id`, "
           "optionally `coords` and `outcomes` (the answers the outcome is read over, each one token). "
           "Under `where: \"turn\"`, `messages` holds the user's turns (`{role: \"user\", content}`) and "
           "`user` alone is one turn."
           " A model reads each record as [its kind](/kinds/records/record/) says: `user` in the chat "
           "template, `prompt` and `text` raw.", many=True),
        In("adapter", "adapter/lora",
           "A LoRA adapter to fuse on top of the model for this node only — "
           "from an `adapter/train` node, a `{\"$ref\": {\"hf_adapter\": {\"repo\": …}}}` "
           "reference, or a stored adapter. Fuses last, on top of any "
           "adapters the model reference itself carries; `adapter_scale` "
           "scales this one.",
           required=False),
    ),
    output=Output("text/branch-point", collection=True,
                  doc="One item per branch point per record, id `<record id>-<step>`: `record_id`, `step`, "
                      "`position`, `turn` under `turn`, `kept`, `k`, `outcome`, `share`, `entropy`, `shift`, "
                      "`shares`, `mass` with `outcomes`, and `largest`. The header carries `model`, `where`, `k`, "
                      "`seed`, `temperature`, `top_p`, `max_tokens`, `cue`, `boundaries` under `sentence`, and "
                      "`generations`: per record the full generation's `record_id`, `text`, `outcome`, `shares`, "
                      "`steps` and `ended`."),
    outputs={"branches": Output("records/record", collection=True,
                                doc="One record per branch, id `<record id>-<step>-b<branch>`, with `coords` "
                                    "`{record, step, position, branch}` (and `turn` under `turn`) beside the "
                                    "record's own: `text` (what the branch wrote; replies joined by a blank "
                                    "line under `turn`), `outcome`, `shares`, `ended` (`end` or `max_tokens`), "
                                    "and with `outcomes` the branch's `entropy_outcomes` and `mass_outcomes`.")},
    params=(
        P("where", "string",
          "Where the branch points go: every `position` (step), every `sentence` boundary, or every `turn` "
          "of a conversation.",
          "sentence", choices=WHERE),
        P("k", "int", "How many continuations to sample at each branch point.", 8),
        P("max_positions", "int",
          "The most branch points one record may have; a generation with more is refused before any branch "
          "is sampled.",
          64),
        P("cue", "string",
          "Text appended to every generation before its outcome is read, such as `\" So the answer is\"`. "
          "Without one the outcome is read where the generation ended.",
          None),
        P("boundaries", "list[string]",
          "Under `sentence`: a token whose text, trailing spaces aside, ends with one of these ends a "
          "sentence.",
          list(BOUNDARIES)),
        P("temperature", "float", "Sampling temperature, for the full generation and every branch.", 0.9),
        P("top_p", "float", "Nucleus sampling, for the full generation and every branch.", 0.95),
        P("max_tokens", "int",
          "The longest generation, in tokens, a branch's included: a branch at step s may write max_tokens − s "
          "more. Under `turn`, the longest reply.",
          256),
    ),
    example={"model": {"$param": "model"}, "where": "sentence", "k": 8,
             "cue": " So the answer is", "max_tokens": 200},
    example_inputs={"records": {"$ref": {"bench": "you/lab/problems"}}},
)


def run(ctx, inputs, params):
    model = ctx.model(params.get("model"))
    records = items_of(inputs.get("records") or [])
    if not records:
        raise ValueError("text/resample: no records to run over — wire records to the `records` port")
    return resample(model, records, params)


def resample(model, records: Sequence[Mapping[str, Any]], params: Mapping[str, Any]) -> dict[str, Any]:
    where = str(params.get("where", "sentence"))
    if where not in WHERE:
        raise ValueError(f"text/resample: unknown where {where!r}; one of {', '.join(WHERE)}")
    k = int(params.get("k", 8))
    if k < 1:
        raise ValueError(f"text/resample: k is how many branches per point, at least 1, not {k}")
    limit = int(params.get("max_positions", 64))
    max_tokens = int(params.get("max_tokens", 256))
    cue = params.get("cue")
    boundaries = tuple(str(b) for b in (params.get("boundaries") or BOUNDARIES) if str(b))
    run = _Sampler(model, seed=params.get("seed", 0),
                   temperature=float(params.get("temperature", 0.9)),
                   top_p=float(params.get("top_p", 0.95)),
                   cue_ids=encode(model.tokenizer, str(cue)) if cue else [])
    points_out: list[dict[str, Any]] = []
    branches_out: list[dict[str, Any]] = []
    generations: list[dict[str, Any]] = []
    for rec in records:
        answers, names = _read_outcomes(model, rec)
        play = (_Conversation(run, rec, max_tokens) if where == "turn"
                else _Reply(run, rec, max_tokens, sentences=boundaries if where == "sentence" else None))
        full = run.read(play.full_ids(), answers, names)
        points = play.points()
        if len(points) > limit:
            raise ValueError(
                f"text/resample: record {rec.get('id')!r} has {len(points)} branch points under where "
                f"{where!r}, past max_positions {limit}; at k {k} that is {len(points) * k} generations. "
                "Raise `max_positions`, lower `max_tokens`, or branch at sentences")
        generations.append({"record_id": rec.get("id"), "text": play.text(), "outcome": full["outcome"],
                            "shares": full["shares"], "steps": play.steps(), "ended": play.ended})
        previous: dict[str, float] | None = None
        made: list[dict[str, Any]] = []
        for i, point in enumerate(points):
            reads = []
            for b in range(k):
                written, ids, ended = play.branch(point, b)
                read = run.read(ids, answers, names)
                reads.append(read)
                coords = {**rec.get("coords", {}), "record": rec.get("id"), **play.coords(point), "branch": b}
                branches_out.append({
                    "id": f"{rec.get('id')}-{play.step(point)}-b{b}", "coords": coords,
                    "text": written, "ended": ended,
                    **{key: read[key] for key in ("outcome", "shares", "entropy_outcomes", "mass_outcomes")
                       if key in read}})
            shares = _average_shares([r["shares"] for r in reads])
            item = {
                "id": f"{rec.get('id')}-{play.step(point)}",
                "coords": {**rec.get("coords", {}), "record": rec.get("id"), **play.coords(point)},
                "record_id": rec.get("id"), **play.coords(point),
                "kept": play.kept(points[i - 1], point) if i else "",
                "k": k, "outcome": full["outcome"],
                "share": shares.get(full["outcome"], 0.0),
                "entropy": measure_entropy(shares),
                "shift": None if previous is None else measure_divergence(previous, shares),
                "shares": shares, "largest": False}
            if answers:
                item["mass"] = float(np.mean([r["mass_outcomes"] for r in reads]))
            made.append(item)
            previous = shares
        moved = [m for m in made if m["shift"] is not None]
        if moved:
            max(moved, key=lambda m: m["shift"])["largest"] = True
        points_out.extend(made)
    header = {"model": serialize_model(params.get("model"), model), "where": where, "k": k,
              "seed": params.get("seed", 0), "temperature": run.temperature, "top_p": run.top_p,
              "max_tokens": max_tokens, "cue": cue, "generations": generations,
              **({"boundaries": list(boundaries)} if where == "sentence" else {})}
    return {DEFAULT_OUTPUT: collection("text/branch-point", points_out,
                                       name=params.get("name", "resampled"),
                                       description=params.get("description", ""), **header),
            "branches": collection("records/record", branches_out, name="branches",
                                   where=where, k=k, seed=header["seed"])}


def measure_entropy(shares: Mapping[str, float]) -> float:
    p = np.array([v for v in shares.values() if v > 0], dtype=np.float64)
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0


def measure_divergence(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    keys = sorted(set(a) | set(b))
    p = np.array([a.get(x, 0.0) for x in keys], dtype=np.float64)
    q = np.array([b.get(x, 0.0) for x in keys], dtype=np.float64)
    m = (p + q) / 2

    def kl(x: np.ndarray) -> float:
        nz = x > 0
        return float((x[nz] * np.log2(x[nz] / m[nz])).sum())

    return max(0.0, 0.5 * kl(p) + 0.5 * kl(q))


def _average_shares(each: Sequence[Mapping[str, float]]) -> dict[str, float]:
    total: dict[str, float] = {}
    for shares in each:
        for name, v in shares.items():
            total[name] = total.get(name, 0.0) + float(v)
    return {name: v / len(each) for name, v in sorted(total.items(), key=lambda kv: (-kv[1], kv[0]))}


def _read_outcomes(model, rec: Mapping[str, Any]) -> tuple[list[Any], list[str]]:
    if not rec.get("outcomes"):
        return [], []
    answers = resolve_outcomes(model, rec)
    return answers, list(dict.fromkeys(str(o) for o in rec["outcomes"]))


class _Sampler:
    def __init__(self, model, *, seed: Any, temperature: float, top_p: float, cue_ids: list[int]) -> None:
        self.model, self.seed = model, seed
        self.temperature, self.top_p, self.cue_ids = temperature, top_p, list(cue_ids)

    def rng(self, *path: Any) -> np.random.Generator:
        return np.random.default_rng(derive_seed(self.seed, *path))

    def write(self, ids: Sequence[int], budget: int, rng: np.random.Generator,
              prefill: Any = None) -> tuple[list[int], str]:
        if budget <= 0:
            return [], "max_tokens"
        from mechbench_compute.api import sample_completion_cached

        _, out = sample_completion_cached(
            self.model, list(ids), max_tokens=budget, temperature=self.temperature, top_p=self.top_p,
            rng=rng, return_ids=True, prefill=prefill)
        return [int(t) for t in out], "max_tokens" if len(out) >= budget else "end"

    def read(self, ids: Sequence[int], answers: Sequence[Any], names: Sequence[str]) -> dict[str, Any]:
        logits = self.model.run(self.model.make_ids(list(ids) + self.cue_ids)).logits
        lp = read_last_logp(logits)
        if not answers:
            top = int(np.argmax(lp))
            word = self.model.tokenizer.decode([top])
            return {"outcome": word.strip() or word, "shares": {word.strip() or word: 1.0}}
        p = np.array([a.p(lp) for a in answers], dtype=np.float64)
        q = p / p.sum()
        return {"outcome": names[int(np.argmax(q))],
                "shares": {n: float(v) for n, v in zip(names, q, strict=True)},
                "entropy_outcomes": read_metric(None, "entropy_outcomes", logits, answers),
                "mass_outcomes": read_metric(None, "mass_outcomes", logits, answers)}


class _Reply:
    def __init__(self, run: _Sampler, rec: Mapping[str, Any], max_tokens: int,
                 sentences: Sequence[str] | None) -> None:
        self.run, self.rec, self.max_tokens, self.sentences = run, rec, int(max_tokens), sentences
        tok = run.model.tokenizer
        self.prompt = (_render_turns(run.model, rec, list(rec["messages"])) if rec.get("messages")
                       else list(render(run.model, rec).ids))
        self.out, self.ended = run.write(self.prompt, self.max_tokens, run.rng(rec.get("id"), "full"))
        self.pieces = [tok.decode(self.out[:s]) for s in range(len(self.out) + 1)]
        self._prefill: tuple[int, Any] = (-1, None)

    def full_ids(self) -> list[int]:
        return self.prompt + self.out

    def text(self) -> str:
        return self.pieces[-1]

    def steps(self) -> int:
        return len(self.out)

    def points(self) -> list[int]:
        n = len(self.out)
        if self.sentences is None:
            return list(range(n + 1))
        ends = [s for s in range(1, n + 1)
                if any(self.pieces[s].rstrip(" ").endswith(b) for b in self.sentences)
                and len(self.pieces[s]) > len(self.pieces[s - 1])]
        return sorted({0, *ends, n})

    def step(self, point: int) -> int:
        return point

    def coords(self, point: int) -> dict[str, Any]:
        return {"step": point, "position": len(self.prompt) + point - 1}

    def kept(self, before: int, point: int) -> str:
        return self.pieces[point][len(self.pieces[before]):]

    def branch(self, point: int, b: int) -> tuple[str, list[int], str]:
        prefix = self.prompt + self.out[:point]
        budget = self.max_tokens - point
        if self._prefill[0] != point:
            self._prefill = (point, prefill_decision(self.run.model, prefix) if budget > 0 else None)
        wrote, ended = self.run.write(prefix, budget, self.run.rng(self.rec.get("id"), point, b),
                                      prefill=self._prefill[1])
        return self.run.model.tokenizer.decode(wrote), prefix + wrote, ended


class _Conversation:
    def __init__(self, run: _Sampler, rec: Mapping[str, Any], max_tokens: int) -> None:
        self.run, self.rec, self.max_tokens = run, rec, int(max_tokens)
        turns = rec.get("messages") or ([{"role": "user", "content": rec["user"]}] if rec.get("user") else [])
        if not turns:
            raise ValueError(f"text/resample: record {rec.get('id')!r} has no turns: where 'turn' reads "
                             "`messages` (the user's turns) or `user`")
        spoken = [m for m in turns if not (isinstance(m, Mapping) and m.get("role") == "user")]
        if spoken:
            raise ValueError(f"text/resample: record {rec.get('id')!r}: where 'turn' writes the replies "
                             f"itself and takes only the user's turns; it has {len(spoken)} other")
        self.users = [str(m.get("content", "")) for m in turns]
        self.replies, self.ends, self.prompts = self._play([], 0, run.rng(rec.get("id"), "full"))
        self.ended = self.ends[-1]

    def _play(self, kept: list[list[int]], start: int,
              rng: np.random.Generator) -> tuple[list[list[int]], list[str], list[list[int]]]:
        replies, ends, prompts = list(kept), ["end"] * len(kept), []
        for t in range(len(self.users)):
            prompts.append(self._prompt(t, replies, prompts[-1] if prompts else []))
            if t >= start:
                wrote, ended = self.run.write(prompts[t], self.max_tokens, rng)
                replies.append(wrote)
                ends.append(ended)
        return replies, ends, prompts

    def _prompt(self, t: int, replies: list[list[int]], before: list[int]) -> list[int]:
        tok = self.run.model.tokenizer
        said = [tok.decode(r) for r in replies[:t]]
        whole = _render_text(self.run.model, self.rec, self.users[:t + 1], said)
        if t == 0:
            return list(encode(tok, whole))
        head = _render_text(self.run.model, self.rec, self.users[:t], said[:t - 1]) + said[t - 1]
        if whole.startswith(head):
            return before + replies[t - 1] + list(encode(tok, whole[len(head):]))
        return list(encode(tok, whole))

    def full_ids(self) -> list[int]:
        return self.prompts[-1] + self.replies[-1]

    def text(self) -> str:
        return "\n\n".join(self.run.model.tokenizer.decode(r) for r in self.replies)

    def steps(self) -> int:
        return sum(len(r) for r in self.replies)

    def points(self) -> list[int]:
        return list(range(len(self.users) + 1))

    def step(self, point: int) -> int:
        return sum(len(r) for r in self.replies[:point])

    def coords(self, point: int) -> dict[str, Any]:
        position = (len(self.prompts[point]) if point < len(self.users)
                    else len(self.prompts[-1]) + len(self.replies[-1])) - 1
        return {"step": self.step(point), "position": position, "turn": point}

    def kept(self, before: int, point: int) -> str:
        tok = self.run.model.tokenizer
        return "\n\n".join(tok.decode(r) for r in self.replies[before:point])

    def branch(self, point: int, b: int) -> tuple[str, list[int], str]:
        tok = self.run.model.tokenizer
        replies, ends, prompts = self._play(self.replies[:point], point, self.run.rng(self.rec.get("id"), point, b))
        written = "\n\n".join(tok.decode(r) for r in replies[point:])
        return written, prompts[-1] + replies[-1], ends[-1] if point < len(self.users) else self.ended


def _render_turns(model, rec: Mapping[str, Any], turns: list[Mapping[str, Any]]) -> list[int]:
    return list(encode(model.tokenizer, _apply_template(model, rec, [dict(t) for t in turns])))


def _render_text(model, rec: Mapping[str, Any], users: Sequence[str], said: Sequence[str]) -> str:
    turns: list[dict[str, Any]] = []
    for i, user in enumerate(users):
        turns.append({"role": "user", "content": user})
        if i < len(said):
            turns.append({"role": "assistant", "content": said[i]})
    return _apply_template(model, rec, turns)


def _apply_template(model, rec: Mapping[str, Any], turns: list[dict[str, Any]]) -> str:
    system = str(rec.get("system") or "")
    if system and turns and turns[0].get("role") == "user":
        turns[0]["content"] = f"{system}\n\n{turns[0].get('content', '')}"
    return model.tokenizer.apply_chat_template(turns, tokenize=False, add_generation_prompt=True)
