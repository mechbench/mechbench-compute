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


def test_the_tls_contexts_trust_certifi_not_the_interpreter_store(monkeypatch):
    import ssl

    import certifi

    from mechbench_compute import guests
    from mechbench_compute.providers import http

    seen = []
    monkeypatch.setattr(ssl, "create_default_context",
                        lambda *a, **kw: seen.append(kw.get("cafile")) or object())
    monkeypatch.setattr(http, "_ctx", None)
    bench._tls("https://api.test")
    guests._tls_context()
    http._ssl_context()
    assert seen == [certifi.where()] * 3


def test_the_runner_reads_the_credential_file_compute_reads(monkeypatch, tmp_path):
    import pathlib

    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    assert bench._config_file() == tmp_path / ".mechbench" / "config.toml"


def test_a_checkpoint_directory_carries_the_names_the_runner_eviction_reads(
        monkeypatch, tmp_path):
    import pathlib

    from mechbench_compute import checkpoint
    from mechbench_compute.protocol import ProtocolExecutor

    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(bench, "fetch", lambda label, with_meta=False: (
        {"kind": "checkpoint_manifest", "files": []}, {}))
    target = tmp_path / "materialized"
    target.mkdir()
    roots = []
    monkeypatch.setattr(checkpoint, "materialize",
                        lambda payload, fetch_file, root, **kw: roots.append(root) or target)
    ProtocolExecutor()._materialize_checkpoint("o/p/checkpoints/c")
    assert roots == [tmp_path / ".mechbench" / "checkpoints"]
    assert (target / ".label").read_text() == "o/p/checkpoints/c"
    assert checkpoint._COMPLETE_MARK == ".complete"


def test_an_emitted_payload_keeps_its_bytes_binary(monkeypatch):
    sent = []
    monkeypatch.setattr(bench, "_request", lambda method, url, key, body=None, **kw:
                        sent.append(body) or {})
    bench.configure(api_url="https://api.test", api_key="k")
    raw = b"\xff\x00\x80safetensors"
    bench.emit("o/p/results/adapter", {"kind": "adapter/lora", "data": raw})
    assert ms.load_raw(sent[0])["payload"]["data"] == raw


def test_hub_progress_counts_the_byte_bars_and_not_the_file_count_bar():
    import io

    from mechbench_compute import hub

    seen = []
    bar = hub._progress_tqdm(lambda done, total: seen.append((done, total)))
    files = bar(total=3, unit="it", file=io.StringIO())
    shard = bar(total=100, unit="B", file=io.StringIO())
    files.update(1)
    shard.update(40)
    assert seen == [(40, 100)]


class _AddedToken:
    def __init__(self, content):
        self.content = content


class _UnkTokenizer:
    eos_token_id = 1
    unk_token_id = 3

    def __init__(self):
        self.added_tokens_decoder = {106: _AddedToken("<turn|>"), 2: _AddedToken("<bos>")}

    def convert_tokens_to_ids(self, token):
        return self.unk_token_id


def test_turn_end_ids_come_from_the_added_tokens_and_never_the_unk_id():
    from mechbench_compute import generate

    assert generate._stop_ids(_UnkTokenizer()) == {1, 106}


def test_a_head_ablation_keeps_the_float32_mask_promotion():
    import mlx.core as mx

    from mechbench_compute import Ablate

    hook = Ablate.head(0, 1).as_hooks()["blocks.0.attn.per_head_out"]
    out = hook(mx.ones((1, 4, 2, 3), dtype=mx.bfloat16), None)
    assert out.dtype == mx.float32
    assert float(mx.abs(out[:, 1]).sum()) == 0.0 and float(out[:, 0].sum()) == 6.0


def test_a_sampled_noise_level_is_the_value_every_stored_corpus_drew():
    from mechbench_compute.ops.records import cross

    value = cross._sample_value({"kind": "noise", "size": 12, "seed": 7}, 3)
    assert value == 'Trf?x"ihw"I!'
