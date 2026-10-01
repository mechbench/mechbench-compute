from __future__ import annotations

import pytest

from mechbench_compute import generate as generate_mod
from mechbench_compute import intervene as iv
from mechbench_compute import positions as POS
from mechbench_compute.distill import render
from mechbench_compute.ops import Context
from mechbench_compute.ops.text import chat as chat_op
from mechbench_compute.ops.text import generate as generate_op
from tests.tiny_models import build_tiny_model

RECORD = {"id": "r", "user": "the cat sat on a mat"}

FLIP = {"point": "logits", "op": "scale", "strength": -1.0}


@pytest.fixture(autouse=True)
def no_stop(monkeypatch):
    monkeypatch.setattr(generate_mod, "_stop_ids", lambda tokenizer: set())


@pytest.fixture(scope="module")
def tiny():
    return build_tiny_model("gemma3")


def _generate(model, positions=None, **params):
    spec = {"spec": [{**FLIP, "positions": positions}], "control": False} if positions is not None else {}
    out = generate_op.run(Context(loaded=model), {"records": [RECORD]},
                          {"model": "tiny", "temperature": 0.0, "max_tokens": 6,
                           "fidelity": "trace", **spec, **params})
    item = out["items"][0]
    start = item["trace"]["generation_spans"][0]["token_start"]
    return item, item["trace"]["token_ids"][start:]


class TestASelectorOverSteps:
    def test_step_k_is_read_at_the_position_before_the_kth_token(self):
        assert POS.resolve({"step": 0}, 10, prompt_len=5) == [4]
        assert POS.resolve({"step": 2}, 10, prompt_len=5) == [6]
        assert POS.resolve({"step": [0, 3]}, 10, prompt_len=5) == [4, 7]
        assert POS.resolve({"step": -1}, 10, prompt_len=5) == [9]

    def test_the_position_forms_count_over_steps(self):
        assert POS.resolve({"step": "all"}, 8, prompt_len=5) == [4, 5, 6, 7]
        assert POS.resolve({"step": "last"}, 8, prompt_len=5) == [7]
        assert POS.resolve({"step": {"range": [1, 3]}}, 8, prompt_len=5) == [5, 6]
        assert POS.resolve({"step": {"after": 2}}, 8, prompt_len=5) == [6, 7]
        toks = list("abcde") + [".", "x", "."]
        assert POS.resolve({"step": {"tokens": ["."]}}, 8, tokens=toks, prompt_len=5) == [5, 7]

    def test_a_generation_span_wins_over_the_prompt_length(self):
        assert POS.resolve({"step": 1}, 10, prompt_len=3, gen_start=6) == [6]

    def test_a_step_past_a_written_sequence_is_refused(self):
        with pytest.raises(ValueError, match="step 4 is past the end: this sequence holds 4 steps"):
            POS.resolve({"step": 4}, 8, prompt_len=5)

    def test_a_step_not_yet_reached_selects_nothing_while_generating(self):
        assert POS.resolve({"step": 4}, 8, prompt_len=5, absent="none") == []

    def test_forms_that_name_positions_are_refused_inside_a_step(self):
        for bad in ("generated", "subject", {"segment": "thinking"}, {"step": 1}, True, 1.5):
            with pytest.raises(ValueError, match="unknown step"):
                POS.resolve({"step": bad}, 8, prompt_len=5)

    def test_a_step_needs_to_know_where_generation_began(self):
        with pytest.raises(ValueError, match="counts from where generation began"):
            POS.resolve({"step": 0}, 8)

    def test_the_furthest_step_a_selector_names(self):
        assert POS.furthest_step({"step": 3}) == 3
        assert POS.furthest_step({"step": [1, 7]}) == 7
        assert POS.furthest_step({"step": {"range": [2, 9]}}) == 2
        assert POS.furthest_step({"step": {"after": 4}}) == 4
        assert POS.furthest_step({"step": "all"}) is None
        assert POS.furthest_step({"step": -1}) is None
        assert POS.furthest_step("last") is None


class TestAnInterventionAtAStep:
    def test_step_two_changes_the_third_token_and_nothing_before_it(self, tiny):
        _, plain = _generate(tiny)
        item, flipped = _generate(tiny, {"step": 2})
        assert len(plain) == len(flipped) == 6
        assert flipped[:2] == plain[:2]
        assert flipped[2] != plain[2]
        assert item["metadata"]["intervention"] == {"steps": [2]}

    def test_it_is_the_absolute_position_counted_by_hand(self, tiny):
        p = len(render(tiny, RECORD).ids)
        _, by_step = _generate(tiny, {"step": 3})
        item, by_hand = _generate(tiny, [p + 2])
        assert by_step == by_hand
        assert item["metadata"]["intervention"] == {"steps": [3]}

    def test_step_zero_is_the_prompts_last_pass(self, tiny):
        _, plain = _generate(tiny)
        item, flipped = _generate(tiny, {"step": 0})
        assert flipped[0] != plain[0]
        assert item["metadata"]["intervention"] == {"steps": [0]}

    def test_a_range_of_steps_is_recorded(self, tiny):
        item, _ = _generate(tiny, {"step": {"range": [1, 3]}})
        assert item["metadata"]["intervention"] == {"steps": [1, 2]}

    def test_every_step_is_every_pass_that_wrote_a_token(self, tiny):
        item, ids = _generate(tiny, {"step": "all"})
        assert item["metadata"]["intervention"]["steps"] == list(range(6)) == list(range(len(ids)))

    def test_the_header_keeps_the_selector_as_given(self, tiny):
        out = generate_op.run(Context(loaded=tiny), {"records": [RECORD]},
                              {"model": "tiny", "temperature": 0.0, "max_tokens": 4,
                               "spec": [{**FLIP, "positions": {"step": 2}}]})
        assert out["spec"][0]["positions"] == {"step": 2}
        control, flipped = out["items"]
        assert control["coords"]["factor"] == 0.0 and "intervention" not in control["metadata"]
        assert flipped["metadata"]["intervention"] == {"steps": [2]}

    def test_a_step_the_node_can_never_reach_is_refused(self, tiny):
        with pytest.raises(iv.SpecError, match="step 6 is past the end: this node writes at most 6 tokens"):
            _generate(tiny, {"step": 6})
        with pytest.raises(iv.SpecError, match="step 6 is past the end"):
            _generate(tiny, {"step": {"after": 6}})

    def test_a_swept_step_is_checked_too(self, tiny):
        with pytest.raises(iv.SpecError, match="step 9 is past the end"):
            generate_op.run(Context(loaded=tiny), {"records": [RECORD]},
                            {"model": "tiny", "max_tokens": 6, "spec": [FLIP],
                             "sweep": {"positions": [{"step": 1}, {"step": 9}]}})

    def test_chat_honours_a_step(self, tiny):
        params = {"model": "tiny", "temperature": 0.0, "max_tokens": 6, "control": False}
        flipped = chat_op.run(Context(loaded=tiny), {"records": [RECORD]},
                              {**params, "spec": [{**FLIP, "positions": {"step": 2}}]})["items"][0]
        assert flipped["metadata"]["intervention"] == {"steps": [2]}
        with pytest.raises(iv.SpecError, match="step 6 is past the end"):
            chat_op.run(Context(loaded=tiny), {"records": [RECORD]},
                        {**params, "spec": [{**FLIP, "positions": {"step": 6}}]})
