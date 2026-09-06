"""The declarative target layer and sequence machinery (experiment 023
platform work): transform chains, entropy-matched tempering, position
runs (the 018 position-0 cap), and the per-step sequence factory —
all exercised against a char-level fake tokenizer, no model."""

import math

import numpy as np
import pytest

from mechbench_compute.finetune import (
    build_marginal_items,
    build_sequence_factory,
    entropy_bits,
    naturalism_gate,
    position_runs,
    resolve_slot_targets,
    target_map_from_spec,
    temper_to_entropy,
)


class CharTok:
    """One token per character — the naturalism gate's ideal world."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


TOK = CharTok()


# --- target specs ----------------------------------------------------


def test_uniform_and_weights_specs():
    t = target_map_from_spec({"uniform": ["a", "b", "c", "d"]})
    assert abs(entropy_bits(t) - 2.0) < 1e-9
    t2 = target_map_from_spec({"weights": {"a": 3, "b": 1}})
    d = t2.to_dict()
    assert abs(d["a"] - 0.75) < 1e-9 and abs(d["b"] - 0.25) < 1e-9


def test_transform_chain_sqrt_and_inverse():
    freqs = {"a": 100.0, "b": 25.0, "c": 4.0, "d": 1.0}
    sq = target_map_from_spec(
        {"weights": freqs, "transform": [{"op": "sqrt"}]})
    d = sq.to_dict()
    # sqrt weights: 10, 5, 2, 1 → normalized
    assert abs(d["a"] - 10 / 18) < 1e-9 and abs(d["d"] - 1 / 18) < 1e-9
    inv = target_map_from_spec(
        {"weights": freqs, "transform": [{"op": "pow", "exponent": -1}]})
    di = inv.to_dict()
    # rank order inverted: rarest is now most probable
    assert di["d"] > di["c"] > di["b"] > di["a"]


def test_temper_to_entropy_hits_target():
    freqs = {c: float(f) for c, f in zip("abcdefgh", [64, 32, 16, 8, 4, 2, 1, 1])}
    base = target_map_from_spec({"weights": freqs})
    for bits in (1.5, 2.0, 2.5):
        t = temper_to_entropy(base, bits)
        assert abs(entropy_bits(t) - bits) < 1e-3
    # declarative form, on the INVERTED map (the 023 R3 rung shape)
    t = target_map_from_spec(
        {"weights": freqs,
         "transform": [{"op": "pow", "exponent": -1},
                       {"op": "temper_to_entropy", "bits": 2.2}]})
    assert abs(entropy_bits(t) - 2.2) < 1e-3
    d = t.to_dict()
    assert d["h"] > d["a"]  # inversion survived the tempering


def test_temper_to_entropy_rejects_out_of_range():
    base = target_map_from_spec({"uniform": ["a", "b", "c", "d"]})
    with pytest.raises(ValueError):
        temper_to_entropy(base, 3.5)  # > log2(4)


def test_unknown_transform_raises():
    with pytest.raises(ValueError, match="unknown target transform"):
        target_map_from_spec(
            {"uniform": ["a"], "transform": [{"op": "banana"}]})


# --- positions (the cap) ---------------------------------------------


def test_position_runs_all_and_skip_first():
    assert position_runs(4, "all") == [(0, 4)]
    assert position_runs(4, "skip_first") == [(1, 4)]
    assert position_runs(6, "skip_first") == [(1, 6)]


def test_position_runs_explicit_and_invalid():
    assert position_runs(5, [0, 1, 3]) == [(0, 2), (3, 4)]
    with pytest.raises(ValueError):
        position_runs(4, [4])
    with pytest.raises(ValueError):
        position_runs(4, [])


# --- per-slot targets ------------------------------------------------


def test_resolve_slot_targets_shared_and_per_slot():
    shared = resolve_slot_targets({"uniform": ["a", "b"], "depth": 3}, 3)
    assert len(shared) == 3 and shared[0] is shared[1]
    per = resolve_slot_targets(
        {"depth": 2, "per_slot": [{"uniform": ["a", "b"]},
                                  {"weights": {"c": 1.0}}]}, 2)
    assert list(per[1].keys()) == ["c"]
    with pytest.raises(ValueError):
        resolve_slot_targets({"per_slot": [{"uniform": ["a"]}]}, 2)


# --- the sequence factory --------------------------------------------


def _targets(depth):
    return resolve_slot_targets({"uniform": list("wxyz")}, depth)


def test_factory_all_positions_single_example():
    f = build_sequence_factory(TOK, _targets(3), ["P>"], join="",
                               closer='"', positions="all")
    rng = np.random.default_rng(0)
    exs = f(rng)
    assert len(exs) == 1
    ex = exs[0]
    assert ex.prompt_ids == [ord("P"), ord(">")]
    assert len(ex.tokens) == 4  # 3 slots + closer
    assert ex.tokens[-1] == ord('"')


def test_factory_skip_first_conditions_on_slot_one():
    f = build_sequence_factory(TOK, _targets(3), ["P>"], join="",
                               closer='"', positions="skip_first")
    exs = f(np.random.default_rng(0))
    assert len(exs) == 1
    ex = exs[0]
    # prompt absorbed the sampled first slot; loss covers slots 2..3 + closer
    assert len(ex.prompt_ids) == 3
    assert len(ex.tokens) == 3
    assert chr(ex.prompt_ids[-1]) in "wxyz"


def test_factory_explicit_positions_split_runs():
    f = build_sequence_factory(TOK, _targets(4), ["P>"], join="",
                               closer='"', positions=[0, 2, 3])
    exs = f(np.random.default_rng(0))
    assert len(exs) == 2
    first, second = exs
    assert len(first.prompt_ids) == 2 and len(first.tokens) == 1
    assert len(second.prompt_ids) == 4  # prompt + slots 0,1 sampled
    assert len(second.tokens) == 3  # slots 2,3 + closer


def test_factory_join_spaces():
    f = build_sequence_factory(TOK, _targets(2), ["P>"], join=" ",
                               closer="", positions="all")
    ex = f(np.random.default_rng(0))[0]
    text = "".join(chr(t) for t in ex.tokens)
    assert len(text) == 3 and text[1] == " "


# --- the naturalism gate ---------------------------------------------


def test_naturalism_gate_passes_single_token_items():
    targets = _targets(3)
    naturalism_gate(TOK, targets, ["P>"], [[ord("P"), ord(">")]],
                    join="", closer='"', samples=10)


def test_naturalism_gate_rejects_multi_token_items():
    targets = resolve_slot_targets({"uniform": ["a", "bb"]}, 2)
    with pytest.raises(ValueError, match="NATURALISM"):
        naturalism_gate(TOK, targets, ["P>"], [[ord("P"), ord(">")]],
                        join="", closer="", samples=25)


# --- marginal rows ---------------------------------------------------


def test_marginal_items_soft_row_over_first_tokens():
    t = target_map_from_spec({"weights": {"a": 3.0, "b": 1.0}})
    items = build_marginal_items(TOK, t, ["P>"])
    assert len(items) == 1
    ex = items[0]
    assert ex.tokens == []
    assert abs(ex.soft[ord("a")] - 0.75) < 1e-9
    assert abs(ex.soft[ord("b")] - 0.25) < 1e-9
