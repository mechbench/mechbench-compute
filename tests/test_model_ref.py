"""The model algebra's reference form (task 000312, Arc A).

What is worth asserting: the grammar's explicitness (no guessing
between HF repos and bench labels), back-compat (bare strings mean
what they always meant), and that the not-yet arcs fail at load time
with a sentence naming the arc — not at parse time, and never with a
shrug.
"""

import pytest

from mechbench_compute import model_ref


def _no_fetch(label):  # a fetch that must not be reached
    raise AssertionError(f"unexpected fetch of {label!r}")


class TestParse:
    def test_a_bare_string_is_an_hf_base_with_no_stack(self):
        ref = model_ref.parse("mlx-community/gemma-4-e2b-it-bf16@abc")
        assert ref.base_kind == "hf"
        assert ref.base == "mlx-community/gemma-4-e2b-it-bf16@abc"
        assert ref.adapter_labels == ()

    def test_structured_hf_base(self):
        ref = model_ref.parse({"base": {"hf": "org/model@dead"}, "adapters": []})
        assert (ref.base_kind, ref.base) == ("hf", "org/model@dead")

    def test_base_source_is_never_guessed(self):
        # An HF repo and a bench label are both slash-paths; a bare
        # dict base without a source key must refuse, not guess.
        with pytest.raises(ValueError, match="base"):
            model_ref.parse({"base": {"repo": "org/model"}})

    def test_bench_base_parses(self):
        ref = model_ref.parse({"base": {"bench": "me/proj/checkpoints/v1"}})
        assert (ref.base_kind, ref.base) == ("bench", "me/proj/checkpoints/v1")

    def test_adapters_parse_in_order(self):
        ref = model_ref.parse(
            {"base": "org/m", "adapters": [{"bench": "a/b/one"}, {"bench": "a/b/two"}]}
        )
        assert ref.adapter_labels == ("a/b/one", "a/b/two")

    def test_empty_string_refuses(self):
        with pytest.raises(ValueError, match="empty"):
            model_ref.parse("")


class TestResolveLimits:
    """Arc boundaries fail loudly and name their arc."""

    def test_checkpoint_base_names_arc_c(self):
        with pytest.raises(NotImplementedError, match="Arc C"):
            model_ref.resolve({"base": {"bench": "me/p/ckpt"}}, fetch=_no_fetch)

    def test_a_stack_resolves_in_order(self):
        # Order IS the semantics (000312 Arc B): round two fused onto
        # round one is not round one fused onto round two.
        fetched = []

        def fetch(label):
            fetched.append(label)
            return {"data": b"\x00", "lora": {"rank": 8, "alpha": 16}}

        ref = model_ref.resolve(
            {"base": "org/m", "adapters": [{"bench": "x/a"}, {"bench": "x/b"}]},
            fetch=fetch,
        )
        assert fetched == ["x/a", "x/b"]
        assert ref.adapter_labels == ("x/a", "x/b")

    def test_depth_nine_names_the_merge_valve(self):
        with pytest.raises(ValueError, match="merge"):
            model_ref.resolve(
                {"base": "org/m", "adapters": [{"bench": f"x/a{i}"} for i in range(9)]},
                fetch=_no_fetch,
            )

    def test_one_adapter_resolves_through_the_injected_fetch(self):
        fetched = []

        def fetch(label):
            fetched.append(label)
            return {"data": b"\x00", "lora": {"rank": 8, "alpha": 16}}

        ref = model_ref.resolve(
            {"base": "org/m@rev", "adapters": [{"bench": "me/p/adapters/a1"}]},
            fetch=fetch,
        )
        assert fetched == ["me/p/adapters/a1"]
        assert ref.adapter_payloads[0]["data"] == b"\x00"

    def test_a_non_adapter_object_is_refused_by_name(self):
        with pytest.raises(ValueError, match="me/p/notes/x"):
            model_ref.resolve(
                {"base": "org/m", "adapters": [{"bench": "me/p/notes/x"}]},
                fetch=lambda _l: {"kind": "note"},
            )


class TestDescribe:
    def test_reads_like_a_sentence_fragment(self):
        assert model_ref.parse("org/m@r").describe() == "hf:org/m@r"
        ref = model_ref.parse({"base": "org/m", "adapters": [{"bench": "a/b/c"}]})
        assert ref.describe() == "hf:org/m (+1 adapter)"


class TestWireForm:
    def test_to_wire_round_trips_through_parse(self):
        ref = model_ref.parse(
            {"base": {"hf": "org/m@rev"}, "adapters": [{"bench": "a/b/c"}]}
        )
        assert model_ref.parse(ref.to_wire()) == ref

    def test_provenance_can_fingerprint_a_normalized_ref(self):
        # Round two of spinner-fairness failed at emit time: the
        # normalized ModelRef object reached the CBOR fingerprint
        # ("cannot encode type ModelRef"). The emit path wire-safes
        # params; this asserts the wire form actually encodes.
        from mechbench_schema.provenance import fingerprint_params

        from mechbench_compute.protocol import _wire_params

        ref = model_ref.parse({"base": "org/m", "adapters": [{"bench": "x/a"}]})
        digest = fingerprint_params(_wire_params({"model": ref, "steps": 40}))
        assert digest == fingerprint_params(
            {"model": {"base": {"hf": "org/m"}, "adapters": [{"bench": "x/a"}]},
             "steps": 40}
        )

