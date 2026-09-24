from __future__ import annotations

import random
import subprocess
import sys

import mechbench_schema as ms
import numpy as np
import pytest

from mechbench_compute import bench, shapes, tensors
from mechbench_compute.interp.constants import MAX_VECTOR_FLOATS
from mechbench_compute.ops.activations.capture_attention import MAX_ATTN_FLOATS
from mechbench_compute.ops.activations.capture_tokens import MAX_TOKEN_VECTOR_FLOATS
from mechbench_compute.ops.weights.capture import MAX_VALUES
from mechbench_compute.providers import limiter
from mechbench_compute.providers import messages as msg
from mechbench_compute.providers import openai_responses


def test_an_operation_file_imported_first_still_registers_its_operation():
    code = ("import mechbench_compute.ops.records.select\n"
            "from mechbench_compute import lexicon\n"
            "assert 'records/select' in lexicon.BY_NAME\n")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


def _read_canonical_bytes_per_float() -> float:
    rng = random.Random(0)
    values = [rng.uniform(-1.0, 1.0) for _ in range(10_000)]
    return len(ms.dump_canonical({"vector": values})) / len(values)


@pytest.mark.parametrize("cap", [MAX_TOKEN_VECTOR_FLOATS, MAX_VECTOR_FLOATS,
                                 MAX_ATTN_FLOATS, MAX_VALUES])
def test_a_capture_at_its_float_cap_fits_one_stored_object(cap):
    assert cap * _read_canonical_bytes_per_float() < bench.MAX_OBJECT_BYTES


def test_a_tensor_shard_fits_one_stored_object():
    assert tensors.MAX_SHARD_BYTES < bench.MAX_OBJECT_BYTES


class _Words:
    def decode(self, ids):
        return f"w{ids[0]}"


def test_tied_log_probabilities_rank_by_token_id():
    logp = np.full(4096, -np.log(4096.0))
    logp[[7, 3000, 11]] = -1.0
    top = shapes.distribution(logp, _Words(), top_k=10)["top"]
    assert [e["token"]["id"] for e in top] == [7, 11, 3000, 0, 1, 2, 3, 4, 5, 6]


def test_two_keys_for_one_provider_are_two_limiter_scopes_and_neither_names_its_key():
    a = limiter.scope_for("openai", {"token": "sk-first"})
    b = limiter.scope_for("openai", {"token": "sk-second"})
    assert a != b
    assert "sk-" not in a + b


def test_a_function_call_replayed_without_its_reasoning_names_no_item_id():
    call = msg.ToolCallPart(id="call_1", name="calc", arguments={"x": 1})
    req = msg.ChatRequest(model="gpt-6-astra-2026-08-01", messages=(
        msg.Message(role="user", content=(msg.TextPart(text="hi"),)),
        msg.Message(role="assistant", content=(call,)),
    ))
    items = [i for i in openai_responses.input_items(req, "openai")
             if i.get("type") == "function_call"]
    assert items == [{"type": "function_call", "call_id": "call_1", "name": "calc",
                      "arguments": '{"x": 1}'}]
