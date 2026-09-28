from __future__ import annotations

import pytest

from mechbench_compute import guests
from mechbench_compute.blocks.read_guest_records import read_guest_records
from mechbench_compute.lexicon import kinds as K


def _cached(name: str) -> bool:
    try:
        guests.resolve(name, fetch=False)
    except Exception:  # noqa: BLE001
        return False
    return True


needs_cpython = pytest.mark.skipif(not _cached("cpython"), reason="the pinned cpython guest is not cached here")
needs_mbshell = pytest.mark.skipif(not _cached("mbshell"), reason="the pinned mbshell guest is not cached here")

RECORDS = K.collection("records/record", [
    {"id": "a", "coords": {"prompt": "x"}, "p": 0.2, "votes": [{"vote": 0, "winner": "A"}, {"vote": 1, "winner": "B"}]},
    {"id": "b", "coords": {"prompt": "y"}, "p": 0.9, "votes": []},
], note="h")


class TestRecordsFromTheGuest:
    def test_a_list_or_a_stream_of_objects(self):
        assert read_guest_records("op", [[{"id": "a"}, {"id": "b"}]]) == [{"id": "a"}, {"id": "b"}]
        assert read_guest_records("op", [{"id": "a"}, {"id": "b"}]) == [{"id": "a"}, {"id": "b"}]

    def test_records_without_ids_are_numbered(self):
        assert read_guest_records("op", [[{"x": 1}, {"x": 2}]]) == [{"id": "0", "x": 1}, {"id": "1", "x": 2}]

    def test_some_ids_or_a_non_record_is_refused(self):
        with pytest.raises(ValueError, match="1 of 2 records have no id"):
            read_guest_records("op", [[{"id": "a"}, {"x": 1}]])
        with pytest.raises(ValueError, match="must be a record"):
            read_guest_records("op", [[1, 2]])


@needs_cpython
class TestPython:
    def test_over_the_collection_with_header_and_params(self):
        from mechbench_compute.ops.records.python import transform_records

        src = ("def transform(records, header, params):\n"
               "    return [dict(r, big=r['p'] > params['cut'], note=header['note']) for r in records]\n")
        out = transform_records(RECORDS, {"source": src}, {"cut": 0.5})
        assert [(r["id"], r["big"], r["note"]) for r in out["items"]] == [("a", False, "h"), ("b", True, "h")]

    def test_per_record_may_drop_or_split(self):
        from mechbench_compute.ops.records.python import transform_records

        src = "def transform(r, header, params):\n    return None if r['p'] < 0.5 else [{'id': r['id'] + '/1'}, {'id': r['id'] + '/2'}]\n"
        out = transform_records(RECORDS, {"source": src, "over": "record"}, {})
        assert [r["id"] for r in out["items"]] == ["b/1", "b/2"]

    def test_the_inputs_order_is_kept_while_its_fields_are(self):
        from mechbench_compute.ops.records.python import transform_records
        from mechbench_compute.ops.records.sort import sort_records

        ranked = sort_records(RECORDS, {"by": ["-p"]}, {})
        kept = transform_records(ranked, {"source": "def transform(rs, h, p):\n    return [dict(r, q=1) for r in rs]\n"}, {})
        assert kept["order_by"] == ["rank"]
        dropped = transform_records(ranked, {"source": "def transform(rs, h, p):\n    return [{'id': r['id']} for r in rs]\n"}, {})
        assert "order_by" not in dropped

    def test_random_is_seeded_from_the_input(self):
        from mechbench_compute.ops.records.python import transform_records

        src = "import random\ndef transform(records, header, params):\n    return [dict(r, u=random.random()) for r in records]\n"
        a = transform_records(RECORDS, {"source": src}, {})
        b = transform_records(RECORDS, {"source": src}, {})
        assert a == b

    @pytest.mark.parametrize(("src", "match"), [
        ("def nope():\n    pass\n", "defines no function `transform`"),
        ("def transform(r, h, p):\n    return 1 / 0\n", "ZeroDivisionError"),
        ("def transform(r, h, p):\n    return {'id': 'a'}\n", "must return a list"),
        ("def transform(r, h, p):\n    while True:\n        pass\n", r"ran past its time \(wall_seconds\)"),
    ])
    def test_a_failure_is_refused_by_name(self, src, match):
        from mechbench_compute.ops.records.python import transform_records

        with pytest.raises(ValueError, match=match):
            transform_records(RECORDS, {"source": src, "seconds": 3}, {})


@needs_mbshell
class TestJq:
    def test_a_program_over_the_records_with_header_and_params(self):
        from mechbench_compute.ops.records.jq import reshape_records

        out = reshape_records(RECORDS, {"program": "map(select(.p > $params.cut) | .note = $header.note)"}, {"cut": 0.5})
        assert [(r["id"], r["note"]) for r in out["items"]] == [("b", "h")]

    def test_a_stream_unnests(self):
        from mechbench_compute.ops.records.jq import reshape_records

        program = '.[] | .votes[] as $v | {id: "\\(.id)/\\($v.vote)", coords, winner: $v.winner}'
        out = reshape_records(RECORDS, {"program": program}, {})
        assert [(r["id"], r["winner"]) for r in out["items"]] == [("a/0", "A"), ("a/1", "B")]

    def test_rank_keeps_the_order_the_program_wrote(self):
        from mechbench_compute.ops.records.jq import reshape_records

        out = reshape_records(RECORDS, {"program": "sort_by(-.p)", "rank": "rank"}, {})
        stored = K.canonical_collection(out)
        assert [(r["id"], r["rank"]) for r in stored["items"]] == [("b", 1), ("a", 2)]
        assert out["order_by"] == ["rank"]

    def test_a_bad_program_is_refused_with_jqs_message(self):
        from mechbench_compute.ops.records.jq import reshape_records

        with pytest.raises(ValueError, match=r"invalid query: map\(\.x \+\)"):
            reshape_records(RECORDS, {"program": "map(.x +)"}, {})
