"""Sequential adapter fusing (task 000312 Arc B).

What is worth asserting is the SEQUENCING, with the fuse math faked:
stacks fuse left to right, restore runs strictly in reverse, and the
scale override touches only the last (node-level) position. The real
`W += scale*(B@A)` arithmetic is lora.fuse's business and exercised
where adapters are actually trained.
"""

import pytest

from mechbench_compute import lora

PAYLOAD_A = {"data": b"a", "lora": {"rank": 8, "alpha": 16}}
PAYLOAD_B = {"data": b"b", "lora": {"rank": 4, "alpha": 4}}


@pytest.fixture()
def recorded(monkeypatch):
    calls = {"fused": [], "restored": [], "loaded": []}
    def _read(path):
        with open(path, "rb") as f:
            return f.read()

    monkeypatch.setattr(lora, "load_adapter", _read)
    monkeypatch.setattr(
        lora,
        "fuse",
        lambda lm, weights, scale: calls["fused"].append((weights, scale)) or f"h{len(calls['fused'])}",
    )
    monkeypatch.setattr(
        lora, "restore", lambda lm, handle: calls["restored"].append(handle)
    )
    return calls


class TestFuseStack:
    def test_left_to_right_with_each_payloads_own_scale(self, recorded):
        handles = lora.fuse_adapter_stack("lm", [PAYLOAD_A, PAYLOAD_B])
        assert [w for w, _ in recorded["fused"]] == [b"a", b"b"]
        assert [s for _, s in recorded["fused"]] == [2.0, 1.0]  # 16/8, 4/4
        assert handles == ["h1", "h2"]

    def test_override_scale_touches_only_the_last(self, recorded):
        lora.fuse_adapter_stack("lm", [PAYLOAD_A, PAYLOAD_B], override_scale=0.5)
        assert [s for _, s in recorded["fused"]] == [2.0, 0.5]

    def test_a_payload_without_bytes_refuses(self, recorded):
        with pytest.raises(ValueError, match="data"):
            lora.fuse_adapter_stack("lm", [{"lora": {}}])
        assert recorded["fused"] == []


class TestRestoreStack:
    def test_strictly_reverse_order(self, recorded):
        handles = lora.fuse_adapter_stack("lm", [PAYLOAD_A, PAYLOAD_B])
        lora.restore_adapter_stack("lm", handles)
        # h2 captured the post-a1 weights; h1 the originals. Any other
        # order reinstalls the wrong past.
        assert recorded["restored"] == ["h2", "h1"]
