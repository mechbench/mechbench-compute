from __future__ import annotations

import copy

import pytest

from mechbench_compute import directions as dirs
from mechbench_compute import intervene as iv
from mechbench_compute import shapes as S
from mechbench_compute.architectures import BY_MODEL_TYPE
from mechbench_compute.ops.activations.capture import capture_residual_vectors
from mechbench_compute.ops.intervene.apply import run_intervene
from tests.tiny_models import MODEL_TYPES, build_tiny_model

POINT = "mlp.down_in"

NEURON = 3

OTHER = 5

A = {"id": "a", "user": "the cat sat on a mat"}

B = {"id": "b", "user": "a dog ran across the park"}

C = {"id": "c", "user": "rain fell on the old roof"}


@pytest.fixture(scope="module", params=MODEL_TYPES)
def tiny(request):
    return build_tiny_model(request.param, BY_MODEL_TYPE[request.param])


def capture(model, records, layers) -> dict:
    return capture_residual_vectors(model, records, {
        "storage": "json", "point": POINT, "layers": layers, "position": "last"})


def mark(source: dict, values: dict) -> dict:
    out = copy.deepcopy(source)
    for row in out["items"]:
        vector = list(row["vector"])
        vector[NEURON] = values[(row["id"], S.layer_of(row))]
        row["vector"] = vector
    return out


def patch(model, records, source, *, layers, row, control=False):
    item = {"point": POINT, "layers": layers, "positions": "last", "op": "patch",
            "neurons": [NEURON], "row": row}
    out = run_intervene(model, records, {
        "spec": [item], "control": control,
        "readout": {"type": "capture", "position": "last",
                    "points": [f"blocks.{layer}.{POINT}" for layer in layers]}},
        inputs={"source": source})
    return out, {(r["id"], S.layer_of(r), r["factor"]): r["vector"] for r in out["items"]}


class TestPatchingRecordForRecord:
    def test_each_record_receives_its_own_row(self, tiny):
        plain = capture(tiny, [B, A], [1])
        source = mark(plain, {("a", 1): 7.0, ("b", 1): -3.0})
        out, read = patch(tiny, [A, B], source, layers=[1], row={"match": "id"})
        assert read[("a", 1, 1.0)][NEURON] == 7.0
        assert read[("b", 1, 1.0)][NEURON] == -3.0
        own = {r["id"]: r["vector"] for r in plain["items"]}
        assert read[("a", 1, 1.0)][OTHER] == own["a"][OTHER]
        assert read[("b", 1, 1.0)][OTHER] == own["b"][OTHER]
        assert "unmatched" not in out

    def test_the_row_is_matched_to_the_layer_it_is_written_at(self, tiny):
        values = {("a", 0): 7.0, ("a", 1): 5.0, ("b", 0): -3.0, ("b", 1): -9.0}
        source = mark(capture(tiny, [B, A], [0, 1]), values)
        _, read = patch(tiny, [A, B], source, layers=[0, 1], row={"match": "id"})
        assert {k[:2]: v[NEURON] for k, v in read.items()} == values

    def test_one_item_per_layer_reads_one_source_over_many_layers(self, tiny):
        values = {("a", 0): 7.0, ("a", 1): 5.0, ("b", 0): -3.0, ("b", 1): -9.0}
        source = mark(capture(tiny, [B, A], [0, 1]), values)
        _, read = patch(tiny, [A, B], source, layers=[1], row={"match": "id"})
        assert {k[:2]: v[NEURON] for k, v in read.items()} == {("a", 1): 5.0, ("b", 1): -9.0}

    def test_a_record_with_no_row_is_skipped_and_reported(self, tiny):
        source = mark(capture(tiny, [A, B], [1]), {("a", 1): 7.0, ("b", 1): -3.0})
        out, read = patch(tiny, [A, B, C], source, layers=[1], row={"match": "id"}, control=True)
        assert out["unmatched"] == ["c"]
        assert sorted({k[0] for k in read}) == ["a", "b"]
        assert read[("a", 1, 1.0)][NEURON] == 7.0

    def test_a_record_missing_one_of_the_items_layers_is_unmatched(self, tiny):
        source = mark(capture(tiny, [A, B], [0, 1]),
                      {("a", 0): 7.0, ("a", 1): 5.0, ("b", 0): -3.0, ("b", 1): -9.0})
        source["items"] = [r for r in source["items"] if (r["id"], S.layer_of(r)) != ("b", 1)]
        out, read = patch(tiny, [A, B], source, layers=[0, 1], row={"match": "id"})
        assert out["unmatched"] == ["b"]
        assert {k[0] for k in read} == {"a"}
        out, _ = patch(tiny, [A, B], source, layers=[0], row={"match": "id"})
        assert "unmatched" not in out

    def test_no_record_matched_is_refused(self, tiny):
        source = capture(tiny, [A], [1])
        with pytest.raises(iv.SpecError, match="no `source` row for any record: b, c"):
            patch(tiny, [B, C], source, layers=[1], row={"match": "id"})

    def test_two_rows_for_one_record_are_refused(self, tiny):
        source = capture(tiny, [A, B], [1])
        source["items"] = source["items"] + [copy.deepcopy(source["items"][0])]
        with pytest.raises(iv.SpecError, match="finds 2 `source` rows"):
            patch(tiny, [A, B], source, layers=[1], row={"match": "id"})

    def test_an_index_still_writes_one_row_into_every_record(self, tiny):
        source = mark(capture(tiny, [B, A], [1]), {("a", 1): 7.0, ("b", 1): -3.0})
        out, read = patch(tiny, [A, B], source, layers=[1], row={"index": 0})
        assert read[("a", 1, 1.0)][NEURON] == -3.0
        assert read[("b", 1, 1.0)][NEURON] == -3.0
        assert "unmatched" not in out


class TestTheMatchIsRead:
    def _spec(self, **kw):
        item = {"point": POINT, "layers": [1], "op": "patch", "source": {"items": []}}
        item.update(kw)
        return iv.Spec(item, n_layers=4, seed=0)

    def test_index_and_match_together_are_refused(self):
        with pytest.raises(iv.SpecError, match="by `index` or by `match`, not both"):
            self._spec(row={"index": 0, "match": "id"})

    def test_a_match_belongs_to_a_patch(self):
        with pytest.raises(iv.SpecError, match="this item's op is 'mean'"):
            self._spec(op="mean", row={"match": "id"})

    def test_a_match_needs_a_source(self):
        with pytest.raises(iv.SpecError, match="the item has no source"):
            direction = dirs.make([1.0, 0.0], S.space(model=None, layer=1, point=POINT, d=2), method="t")
            iv.Spec({"point": POINT, "op": "patch", "direction": direction,
                     "row": {"match": "id"}}, n_layers=4, seed=0)

    def test_a_match_names_a_field(self):
        with pytest.raises(iv.SpecError, match="names a field"):
            self._spec(row={"match": 3})
