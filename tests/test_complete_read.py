"""Complete outcomes read exactly: multi-token outcomes
scored whole, closed by their closer, into `tracked` by name — with the
scorer replaced, so the arithmetic is the test's own, no model."""

import math

import pytest

from mechbench_compute import distill
from mechbench_compute.ops.eval.expect import check_expectations


class WordTok:
    """One token per character, so a sequence's length is its text's."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


PROMPT = 'Pick: "'


@pytest.fixture
def scored(monkeypatch):
    """A scorer that gives each outcome the log-prob its test says, and
    records the token sequences it was asked about."""
    asked: dict[str, list[int]] = {}
    logps: dict[str, float] = {}

    def fake(model, prompt_ids, sequences, chunk=16):
        asked.update(sequences)
        return {k: logps[k] for k in sequences}

    monkeypatch.setattr(distill, "score_items_fast", fake)
    return asked, logps


def test_outcomes_are_scored_with_their_closer(scored):
    asked, logps = scored
    tok = WordTok()
    ids = tok.encode(PROMPT)
    logps.update({"Mystery": math.log(0.3), "Mystery Thriller": math.log(0.1)})
    entries, mass, entropy = distill.score_complete(
        None, tok, PROMPT, ids, {"items": ["Mystery", "Mystery Thriller"], "closer": '"'})
    assert asked["Mystery"] == tok.encode('Mystery"')
    assert entries["Mystery"] == {"text": "Mystery", "tokens": 8, "p": 0.3, "logp": round(math.log(0.3), 4)}
    assert mass == pytest.approx(0.4)
    # 0.3 : 0.1 renormalized is 0.75 : 0.25.
    assert entropy == pytest.approx(-(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25)))


def test_an_opener_is_scored_but_not_named(scored):
    asked, logps = scored
    tok = WordTok()
    logps.update({"Humor": math.log(0.2)})
    entries, _, _ = distill.score_complete(
        None, tok, "List: Fiction,", tok.encode("List: Fiction,"),
        {"items": ["Humor"], "opener": " ", "closer": ","})
    assert asked["Humor"] == tok.encode(" Humor,")
    assert entries["Humor"]["text"] == "Humor" and entries["Humor"]["tokens"] == 7


def test_items_can_be_a_target_spec_narrowed_by_its_transforms(scored):
    _, logps = scored
    tok = WordTok()
    freqs = {"weights": {"a": 5.0, "b": 3.0, "c": 1.0}, "transform": [{"op": "top_k", "k": 2}]}
    logps.update({"a": math.log(0.5), "b": math.log(0.25)})
    entries, _, _ = distill.score_complete(None, tok, PROMPT, tok.encode(PROMPT), {"items": freqs})
    assert list(entries) == ["a", "b"]
    # A fetched target_map object arrives as its payload, {kind, weights}.
    fetched = {"weights": {"kind": "target_map", "weights": freqs["weights"]},
               "transform": freqs["transform"]}
    entries, _, _ = distill.score_complete(None, tok, PROMPT, tok.encode(PROMPT), {"items": fetched})
    assert list(entries) == ["a", "b"]


def test_items_must_name_outcomes():
    with pytest.raises(ValueError, match="names no outcomes"):
        distill.complete_items([])
    with pytest.raises(TypeError, match="list of outcomes or a target spec"):
        distill.complete_items("Mystery")


def test_a_complete_read_is_judged_against_its_target_like_a_token_read(scored):
    """The point of putting complete outcomes in `tracked`: `eval/expect`
    needs nothing new to judge a many-token vocabulary."""
    _, logps = scored
    tok = WordTok()
    target = {"Mystery": 2.0, "Humor": 1.0, "Science Fiction": 1.0}
    logps.update({k: math.log(0.9 * v / 4) for k, v in target.items()})
    entries, mass, _ = distill.score_complete(None, tok, PROMPT, tok.encode(PROMPT), {"items": list(target)})
    out = check_expectations({
        "results": [{"id": "r", "entropy_bits": 3.0, "tracked": entries}],
        "expectations": [{"id": "r", "expect": {"type": "weights", "weights": target, "max_kl_bits": 0.01}}],
    }, {})
    (row,) = out["items"]
    assert row["pass"] is True and row["kl_bits"] == pytest.approx(0.0, abs=1e-4)
    assert row["mass"] == pytest.approx(0.9, abs=1e-4) and mass == pytest.approx(0.9)
