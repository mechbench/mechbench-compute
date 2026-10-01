from __future__ import annotations

import math
from collections import defaultdict

import pytest

from mechbench_compute import generate as generate_mod
from mechbench_compute.distill import render
from mechbench_compute.expr.engine import load_engine
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.ops import Context
from mechbench_compute.ops.text import resample as resample_op
from tests.tiny_models import WORDS, build_tiny_model

OUTCOMES = ["cat", "dog", "mat"]

RECORD = {"id": "r", "user": "the cat sat on a mat", "outcomes": OUTCOMES, "coords": {"cond": "x"}}

TURNS = {"id": "t", "outcomes": OUTCOMES,
         "messages": [{"role": "user", "content": "the cat"}, {"role": "user", "content": "a dog ran"}]}

BASE = {"model": "tiny", "k": 3, "max_tokens": 4, "seed": 5, "cue": " the"}


@pytest.fixture(autouse=True)
def no_stop(monkeypatch):
    monkeypatch.setattr(generate_mod, "_stop_ids", lambda tokenizer: set())


@pytest.fixture(scope="module")
def tiny():
    return build_tiny_model("gemma3")


def _run(model, records, **params):
    return resample_op.run(Context(loaded=model), {"records": records}, {**BASE, **params})


def _js(p, q):
    keys = set(p) | set(q)
    m = {x: (p.get(x, 0.0) + q.get(x, 0.0)) / 2 for x in keys}
    half = 0.0
    for d in (p, q):
        half += 0.5 * sum(d[x] * math.log2(d[x] / m[x]) for x in d if d[x] > 0)
    return half


def _by_hand(branches):
    at = defaultdict(list)
    for b in branches:
        at[b["coords"]["step"]].append(b["shares"])
    out = {}
    for step, each in at.items():
        names = {n for s in each for n in s}
        out[step] = {n: sum(s.get(n, 0.0) for s in each) / len(each) for n in names}
    return out


class TestAtEveryPosition:
    def test_a_branch_point_at_every_step_and_k_branches_at_each(self, tiny):
        out = _run(tiny, [RECORD], where="position")
        points, branches = out["out"]["items"], out["branches"]["items"]
        assert [p["step"] for p in points] == [0, 1, 2, 3, 4]
        assert len(branches) == 5 * 3
        assert out["out"]["generations"][0]["steps"] == 4

    def test_the_coords_name_the_record_step_position_and_branch(self, tiny):
        out = _run(tiny, [RECORD], where="position")
        prompt = len(render(tiny, RECORD).ids)
        for b in out["branches"]["items"]:
            c = b["coords"]
            assert c["record"] == "r" and c["cond"] == "x"
            assert c["position"] == prompt + c["step"] - 1
            assert b["id"] == f"r-{c['step']}-b{c['branch']}"
        assert sorted({b["coords"]["branch"] for b in out["branches"]["items"]}) == [0, 1, 2]
        for p in out["out"]["items"]:
            assert p["coords"] == {"cond": "x", "record": "r", "step": p["step"], "position": prompt + p["step"] - 1}

    def test_share_entropy_and_shift_are_the_branches_arithmetic(self, tiny):
        out = _run(tiny, [RECORD], where="position")
        hand = _by_hand(out["branches"]["items"])
        full = out["out"]["generations"][0]["outcome"]
        previous = None
        for p in out["out"]["items"]:
            shares = hand[p["step"]]
            assert p["shares"] == pytest.approx(shares)
            assert sum(p["shares"].values()) == pytest.approx(1.0)
            assert p["outcome"] == full
            assert p["share"] == pytest.approx(shares[full])
            assert p["entropy"] == pytest.approx(-sum(v * math.log2(v) for v in shares.values() if v > 0))
            if previous is None:
                assert p["shift"] is None
            else:
                assert p["shift"] == pytest.approx(_js(previous, shares), abs=1e-9)
            previous = shares
        moved = [p for p in out["out"]["items"] if p["shift"] is not None]
        largest = [p for p in out["out"]["items"] if p["largest"]]
        assert largest == [max(moved, key=lambda p: p["shift"])]

    def test_a_branch_carries_the_outcome_metrics_and_the_point_their_mass(self, tiny):
        out = _run(tiny, [RECORD], where="position")
        branches = out["branches"]["items"]
        for b in branches:
            q = list(b["shares"].values())
            assert b["entropy_outcomes"] == pytest.approx(-sum(v * math.log2(v) for v in q if v > 0), abs=1e-5)
            assert 0.0 < b["mass_outcomes"] <= 1.0
            assert b["outcome"] == max(b["shares"], key=b["shares"].get)
        for p in out["out"]["items"]:
            mine = [b["mass_outcomes"] for b in branches if b["coords"]["step"] == p["step"]]
            assert p["mass"] == pytest.approx(sum(mine) / len(mine))

    def test_the_last_point_keeps_everything_and_reads_the_full_generation(self, tiny):
        out = _run(tiny, [RECORD], where="position")
        last = out["out"]["items"][-1]
        assert last["shares"] == pytest.approx(out["out"]["generations"][0]["shares"], abs=1e-5)
        assert {b["text"] for b in out["branches"]["items"] if b["coords"]["step"] == 4} == {""}

    def test_the_same_seed_writes_the_same_branches(self, tiny):
        a = _run(tiny, [RECORD], where="position")["branches"]["items"]
        b = _run(tiny, [RECORD], where="position")["branches"]["items"]
        assert [x["text"] for x in a] == [x["text"] for x in b]
        c = _run(tiny, [RECORD], where="position", seed=6)["branches"]["items"]
        assert [x["shares"] for x in a] != [x["shares"] for x in c]

    def test_without_outcomes_the_shares_count_the_branches(self, tiny):
        plain = {k: v for k, v in RECORD.items() if k != "outcomes"}
        out = _run(tiny, [plain], where="position")
        for b in out["branches"]["items"]:
            assert list(b["shares"].values()) == [1.0] and "mass_outcomes" not in b
        for p in out["out"]["items"]:
            assert all(round(v * 3) == pytest.approx(v * 3) for v in p["shares"].values())
            assert "mass" not in p

    def test_too_many_branch_points_is_refused_before_branching(self, tiny):
        with pytest.raises(ValueError, match="has 5 branch points under where 'position', past max_positions 4"):
            _run(tiny, [RECORD], where="position", max_positions=4)


class TestAtSentences:
    def test_the_points_are_the_start_each_sentence_end_and_the_end(self, tiny):
        ends = list(WORDS[4:])
        sentences = _run(tiny, [RECORD], where="sentence", boundaries=ends)["out"]
        positions = _run(tiny, [RECORD], where="position")["out"]
        steps = [p["step"] for p in sentences["items"]]
        assert steps[0] == 0 and steps[-1] == 4
        assert set(steps) <= {p["step"] for p in positions["items"]}
        assert sentences["generations"][0]["text"] == positions["generations"][0]["text"]
        for p in sentences["items"][1:-1]:
            assert any(p["kept"].rstrip(" ").endswith(b) for b in ends)
        assert sentences["boundaries"] == ends

    def test_with_no_sentence_ended_the_points_are_the_start_and_the_end(self, tiny):
        out = _run(tiny, [RECORD], where="sentence", boundaries=["zzz"])["out"]
        assert [p["step"] for p in out["items"]] == [0, 4]
        assert out["items"][1]["kept"] == out["generations"][0]["text"]


class TestAtTurns:
    def test_one_point_per_reply_and_one_after_the_last(self, tiny):
        out = _run(tiny, [TURNS], where="turn")
        points = out["out"]["items"]
        assert [p["turn"] for p in points] == [0, 1, 2]
        assert [p["step"] for p in points] == [0, 4, 8]
        assert out["out"]["generations"][0]["steps"] == 8
        assert len(out["branches"]["items"]) == 9
        assert all(b["coords"]["turn"] == b["coords"]["step"] // 4 for b in out["branches"]["items"])
        assert points[-1]["shares"] == pytest.approx(out["out"]["generations"][0]["shares"], abs=1e-5)

    def test_a_reply_in_the_record_is_refused(self, tiny):
        spoken = {**TURNS, "messages": [*TURNS["messages"], {"role": "assistant", "content": "a cat"}]}
        with pytest.raises(ValueError, match="takes only the user's turns; it has 1 other"):
            _run(tiny, [spoken], where="turn")


class TestRefusals:
    def test_an_unknown_where(self, tiny):
        with pytest.raises(ValueError, match="unknown where 'word'"):
            _run(tiny, [RECORD], where="word")

    def test_no_branches(self, tiny):
        with pytest.raises(ValueError, match="at least 1, not 0"):
            _run(tiny, [RECORD], k=0)


class TestTheKind:
    def test_it_speaks_the_largest_shift_and_draws_a_line_over_step(self, tiny):
        kind = K.BY_KIND["text/branch-point"]
        assert kind.extends == "records/record" and kind.key == ("id",)
        assert kind.draw.mark == "line" and kind.draw.encoding == {"x": "step", "y": "shift", "series": "record_id"}
        out = _run(tiny, [RECORD], where="position")["out"]
        header = {k: v for k, v in out.items() if k != "items"}
        said = load_engine().render(kind.speak, out["items"], header=header).values
        loud = [s for s, p in zip(said, out["items"], strict=True) if p["largest"]]
        assert len(loud) == 1 and "the largest shift in this generation" in loud[0]
        assert said[0].startswith("r at step 0: ") and "bits from the point before" not in said[0]

    def test_the_header(self, tiny):
        out = _run(tiny, [RECORD], where="position")["out"]
        assert {k: out[k] for k in ("where", "k", "seed", "max_tokens", "cue")} == {
            "where": "position", "k": 3, "seed": 5, "max_tokens": 4, "cue": " the"}
        assert out["generations"][0]["record_id"] == "r" and out["generations"][0]["ended"] == "max_tokens"
