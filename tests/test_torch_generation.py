from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

if importlib.util.find_spec("nnsight") is None:
    pytest.skip("nnsight is not installed: pip install 'mechbench-compute[torch]'",
                allow_module_level=True)
torch = pytest.importorskip("torch")

from mechbench_compute import backends
from mechbench_compute.generate import sample_completion_cached
from mechbench_compute.ops import Context
from mechbench_compute.ops.logits import read as read_op
from mechbench_compute.ops.text import generate as generate_op
from mechbench_compute.ops.text import resample as resample_op
from mechbench_compute.ops.text import score as score_op
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec
from mechbench_compute.torch_backend.batched import forward_rows, left_pad
from mechbench_compute.torch_backend.decoding import make_prompt_cache
from mechbench_compute.torch_backend.throughput import measure_throughput
from tests.kit_backends import WINDOW
from tests.tiny_torch_models import KIT_MODELS, build_tiny_model

ROOT = pathlib.Path(__file__).resolve().parent.parent

MODELS = tuple(name for name, _ in KIT_MODELS)

PROMPTS = [[2, 4, 7, 8, 9, 10, 11], [2, 7, 8], [2, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 7, 8, 9], [5]]

END = 3

PADDED_ROWS_ATOL = 1e-4

RECORDS = [{"id": "a", "user": "the cat sat on a"}, {"id": "b", "user": "the dog ran"},
           {"id": "c", "prompt": "the cat sat on the mat and the dog ran on the", "template": "raw"}]


@pytest.fixture(scope="module", params=MODELS)
def tiny(request):
    return build_tiny_model(request.param)


def read_row(model, ids):
    return model.run(model.make_ids(ids)).logits[0, -1].float()


def greedy_by_whole_forwards(model, prompt, steps, hooks=None):
    ids, out = list(prompt), []
    for _ in range(steps):
        nxt = int(model.run(model.make_ids(ids), hooks=hooks).logits[0, -1].float().argmax())
        if nxt == END:
            break
        out.append(nxt)
        ids.append(nxt)
    return out


def sample(model, prompt, *, seed=0, temperature=0.9, **kw):
    return sample_completion_cached(model, prompt, max_tokens=12, temperature=temperature, top_p=0.95,
                                    rng=np.random.default_rng(seed), return_ids=True, **kw)[1]


def draw(model, prompts, *, temperature, batch_size, seeds=None, **kw):
    seeds = seeds if seeds is not None else range(len(prompts))
    out, used = model.generate_batch(prompts, [np.random.default_rng(s) for s in seeds],
                                     max_tokens=12, temperature=temperature, top_p=0.95,
                                     batch_size=batch_size, **kw)
    return [ids for _, ids in out], used


def test_decoding_from_the_cache_reads_the_rows_a_whole_forward_reads_past_the_window(tiny):
    ids = PROMPTS[2]
    assert len(ids) > 3 * WINDOW
    cache, row = tiny.prefill_decision(ids[:5])
    rows = [row]
    for t in ids[5:]:
        rows.append(tiny.run(tiny.make_ids([t]), kv_cache=cache).logits[0, -1].float())
    for i, r in enumerate(rows):
        assert torch.allclose(r, read_row(tiny, ids[:5 + i]), atol=1e-4)
    assert cache.get_seq_length() == len(ids)


def test_greedy_generation_with_the_cache_is_the_argmax_of_whole_forwards(tiny):
    for prompt in PROMPTS:
        assert sample(tiny, prompt, temperature=0) == greedy_by_whole_forwards(tiny, prompt, 12)


def test_a_seeded_generation_twice_on_one_device_is_identical(tiny):
    assert [sample(tiny, p, seed=7) for p in PROMPTS] == [sample(tiny, p, seed=7) for p in PROMPTS]
    first, _ = draw(tiny, PROMPTS, temperature=0.9, batch_size=4)
    again, _ = draw(tiny, PROMPTS, temperature=0.9, batch_size=4)
    assert first == again


def test_a_greedy_sequence_is_the_same_tokens_whatever_shares_its_batch(tiny):
    alone = [sample(tiny, p, temperature=0) for p in PROMPTS]
    for size in (1, 2, 4):
        assert draw(tiny, PROMPTS, temperature=0, batch_size=size)[0] == alone
    shuffled = [PROMPTS[i] for i in (3, 1, 0, 2)]
    assert draw(tiny, shuffled, temperature=0, batch_size=4)[0] == [alone[i] for i in (3, 1, 0, 2)]


def test_left_padding_moves_a_row_by_float_rounding_only(tiny):
    ids, mask, positions = left_pad(PROMPTS, 0, tiny.device)
    rows = forward_rows(tiny, ids, mask, positions, make_prompt_cache(tiny._model))
    for i, p in enumerate(PROMPTS):
        assert torch.allclose(rows[i], read_row(tiny, p), atol=PADDED_ROWS_ATOL)


def test_a_batched_seeded_sample_draws_from_its_own_stream(tiny):
    alone = [sample(tiny, p, seed=s) for s, p in enumerate(PROMPTS)]
    assert draw(tiny, PROMPTS, temperature=0.9, batch_size=4)[0] == alone


def test_a_batch_stops_each_sequence_at_its_own_stop_string_and_streams_each():
    tiny = build_tiny_model("llama")
    words = [tiny.tokenizer.decode([t]) for t in range(16)]
    stop = next(w for w in words[7:] if w)
    seen: dict[int, list] = {0: [], 1: []}
    out, _ = tiny.generate_batch(PROMPTS[:2], [np.random.default_rng(s) for s in (1, 2)],
                                 max_tokens=10, temperature=5.0, top_p=1.0, stop_strings=[stop],
                                 on_tokens=[seen[0].append, seen[1].append], batch_size=2)
    for i, (text, ids) in enumerate(out):
        assert stop not in text
        assert len(ids) <= 10
        assert [e["id"] for e in seen[i]] == ids[:len(seen[i])]
        alone = sample_completion_cached(tiny, PROMPTS[i], max_tokens=10, temperature=5.0, top_p=1.0,
                                         rng=np.random.default_rng(i + 1), stop_strings=[stop],
                                         return_ids=True)
        assert (text, ids) == alone


def run_generate(model, params, records=RECORDS):
    return generate_op.run(Context(loaded=model), {"records": records},
                           {"model": "tiny", "n": 2, "max_tokens": 6, "seed": 3, **params})


def test_text_generate_on_torch_batches_and_matches_one_at_a_time(tiny):
    batched = run_generate(tiny, {"batch": 4, "fidelity": "trace"})
    single = run_generate(tiny, {"batch": 1, "fidelity": "trace"})
    assert batched["batch"] == 4 and single["batch"] == 1
    assert [i["text"] for i in batched["items"]] == [i["text"] for i in single["items"]]
    assert [i["trace"]["token_ids"] for i in batched["items"]] == [i["trace"]["token_ids"] for i in single["items"]]
    again = run_generate(tiny, {"batch": 4, "fidelity": "trace"})
    assert json.dumps(again["items"], sort_keys=True) == json.dumps(batched["items"], sort_keys=True)


def test_an_intervention_applies_at_every_generated_step_on_torch(tiny):
    direction = np.random.default_rng(0).normal(size=tiny.arch.d_model).astype(np.float32) * 40
    spec = [{"point": "resid_post", "layers": [1], "op": "add", "positions": "all",
             "direction": {"kind": "direction/vector", "vector": direction.tolist(),
                           "space": {"model": None, "layer": 1, "point": "resid_post", "head": None,
                                     "d": tiny.arch.d_model}}}]
    out = run_generate(tiny, {"spec": spec, "temperature": 0, "control": False, "fidelity": "trace"})
    plain = run_generate(tiny, {"temperature": 0, "fidelity": "trace"})
    added = torch.as_tensor(direction)

    def add(act, info):
        return act + added.to(act.dtype)

    for item, rec in zip(out["items"][::2], RECORDS, strict=True):
        start = item["segmentations"][0]["segments"][1]["token_start"]
        ids = item["trace"]["token_ids"]
        expected = greedy_by_whole_forwards(tiny, ids[:start], 6, hooks={"blocks.1.resid_post": add})
        assert ids[start:] == expected, rec["id"]
    ids = [i["trace"]["token_ids"] for i in out["items"]]
    assert ids != [i["trace"]["token_ids"] for i in plain["items"]]


def test_a_weight_edit_generates_on_torch_and_puts_the_weights_back():
    tiny = build_tiny_model("llama")
    before = {k: v.detach().clone() for k, v in tiny._model.state_dict().items()}
    edited = run_generate(tiny, {"spec": [{"parameter": "layers.1.mlp.down_proj.weight", "op": "zero"}],
                                 "temperature": 0, "control": True})
    after = tiny._model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    by_factor = {}
    for item in edited["items"]:
        by_factor.setdefault(item["coords"]["factor"], []).append(item["text"])
    assert by_factor[0.0] == [i["text"] for i in run_generate(tiny, {"temperature": 0})["items"]]


def test_a_projection_reads_the_residual_at_each_generated_token_on_torch():
    tiny = build_tiny_model("gemma3")
    vector = np.random.default_rng(1).normal(size=tiny.arch.d_model).astype(np.float32)
    project = {"kind": "direction/vector", "vector": vector.tolist(),
               "space": {"model": None, "layer": 2, "point": "resid_post", "head": None,
                         "d": tiny.arch.d_model}}
    out = generate_op.run(Context(loaded=tiny), {"records": RECORDS[:1], "project": project},
                          {"model": "tiny", "max_tokens": 5, "temperature": 0, "fidelity": "trace"})
    item = out["items"][0]
    ids, start = item["trace"]["token_ids"], item["segmentations"][0]["segments"][1]["token_start"]
    coords = item["projection"]["coords"]
    assert len(coords) == len(ids) - start
    for j, coord in enumerate(coords[:-1]):
        res = tiny.run(tiny.make_ids(ids[:start + j + 1]), capture=["blocks.2.resid_post"])
        row = res.cache["blocks.2.resid_post"][0, -1].float().numpy()
        assert coord == pytest.approx(float(row @ vector), abs=1e-3)


def whole_logprob(model, prompt, continuation):
    lp, ids = 0.0, list(prompt)
    for t in continuation:
        row = read_row(model, ids)
        lp += float(row[t] - torch.logsumexp(row, -1))
        ids.append(t)
    return lp


def test_logits_read_completes_and_rolls_out_on_torch(tiny):
    from mechbench_compute.distill import render, suffix_tokens

    params = {"model": "tiny", "top_k": 3, "complete": {"items": ["mat", "dog", "cat sat"], "closer": ""},
              "rollout": {"top_k": 3, "max_tokens": 2, "terminators": ["<end>"]}}
    out = read_op.run(Context(loaded=tiny), {"conditions": RECORDS[:1]}, params)
    entry = out["items"][0]
    r = render(tiny, RECORDS[0])
    for name in ("mat", "dog", "cat sat"):
        seq = suffix_tokens(tiny.tokenizer, r.text, r.ids, name)
        assert entry["tracked"][name]["logp"] == pytest.approx(whole_logprob(tiny, r.ids, seq), abs=1e-3)
    assert {"top_outcomes", "completed_mass", "forwards_used"} <= set(entry["rollout"])
    assert entry["rollout"]["forwards_used"] > 1


def test_text_score_on_torch_batches_items_of_different_lengths(tiny):
    items = [{"id": f"d{i}", "trace": {"token_ids": p}} for i, p in enumerate(PROMPTS)]
    out = score_op.run(Context(loaded=tiny), {"collection": items}, {"model": "tiny"})
    by_item: dict[str, list[float]] = {}
    for v in out["items"]:
        by_item.setdefault(v["anchor"]["item_id"], []).append(v["value"])
    for i, p in enumerate(PROMPTS):
        expected = [-whole_logprob(tiny, p[:j], [p[j]]) / np.log(2.0) for j in range(1, len(p))]
        assert by_item.get(f"d{i}", []) == pytest.approx(expected, abs=2e-3)


def test_text_resample_runs_on_torch_and_repeats():
    tiny = build_tiny_model("llama")
    params = {"model": "tiny", "where": "position", "k": 2, "max_tokens": 4, "seed": 1}
    first = resample_op.resample(tiny, RECORDS[:1], params)
    again = resample_op.resample(tiny, RECORDS[:1], params)
    assert json.dumps(first, sort_keys=True, default=str) == json.dumps(again, sort_keys=True, default=str)
    assert first["out"]["items"]


@pytest.fixture
def torch_job(monkeypatch):
    from mechbench_compute.torch_backend.model import TorchModel

    tiny = build_tiny_model("gemma3")
    monkeypatch.setattr(TorchModel, "load", classmethod(lambda cls, model_id, **_: tiny))
    return tiny


def run_job(nodes, edges=()):
    out = ProtocolExecutor().run(ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
        "graph": {"dataflow": 2, "nodes": nodes, "edges": list(edges)},
        "requirements": {"class": "local", "backend": "torch"}}))
    return out.payload if hasattr(out, "payload") else out


def test_a_torch_job_generates_chats_and_scores_and_says_what_ran(torch_job):
    payload = run_job([
        {"id": "gen", "block": "text/generate", "inputs": {"records": RECORDS},
         "params": {"model": "tiny/gemma3@rev", "n": 2, "max_tokens": 4, "fidelity": "trace"}},
        {"id": "chat", "block": "text/chat", "inputs": {"records": RECORDS[:2]},
         "params": {"model": "tiny/gemma3@rev", "max_tokens": 4}},
        {"id": "score", "block": "text/score", "params": {"model": "tiny/gemma3@rev"}},
    ], [{"from": {"node": "gen"}, "to": {"node": "score", "port": "collection"}}])
    for name in ("chat", "score"):
        assert payload["outputs"][name]["backend"] == "torch", name
    assert payload["node_summaries"]["gen"]["items"] == 6
    assert len(payload["outputs"]["chat"]["items"]) == 2
    assert {v["anchor"]["item_id"] for v in payload["outputs"]["score"]["items"]} == {
        f"{r['id']}-s{k}" for r in RECORDS for k in range(2)}
    numerics = payload["resources"]["hardware"]["numerics"]
    assert numerics["threads"] == torch.get_num_threads()
    assert numerics["deterministic_algorithms"] == (torch_job.device.type == "cuda")


def test_a_torch_job_intervenes_with_an_operator_and_patches(torch_job):
    payload = run_job([
        {"id": "apply", "block": "intervene/apply", "inputs": {"records": RECORDS},
         "params": {"model": "tiny/gemma3@rev", "spec": [
             {"point": "resid_post", "layers": [1], "f": "x * 2 if x > 0 else x", "positions": "all"}]}},
        {"id": "patch", "block": "intervene/patch",
         "inputs": {"records": [{"id": "p", "a": "the cat sat", "b": "the dog sat", "template": "raw"}]},
         "params": {"model": "tiny/gemma3@rev", "layers": [0, 1]}},
    ])
    for name in ("apply", "patch"):
        assert payload["outputs"][name]["backend"] == "torch", name
    assert len(payload["outputs"]["apply"]["items"]) == len(RECORDS) * 2
    assert len(payload["outputs"]["patch"]["items"][0]["measures"]["recovery"]) == 2


def test_an_operation_the_torch_backend_does_not_run_is_refused_by_name(torch_job):
    with pytest.raises(backends.BackendRefused, match=r"weights/circuit does not run on the torch backend yet"):
        run_job([{"id": "c", "block": "weights/circuit",
                  "params": {"model": "tiny/gemma3@rev", "head": {"layer": 1, "index": 0}}}])


def build_mlx_twin(name, tiny):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.models import llama

    from mechbench_compute.model import Model
    from tests.tiny_models import build_gemma3
    from tests.tiny_tokenizer import build_tokenizer

    if name == "llama":
        wrapped = lm = llama.Model(llama.ModelArgs(
            model_type="llama", hidden_size=32, num_hidden_layers=4, intermediate_size=64,
            num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-6, vocab_size=64,
            tie_word_embeddings=False))
    else:
        wrapped, lm = build_gemma3()
    names = set(dict(tree_flatten(lm.parameters())))
    weights = {k: mx.array(v.detach().cpu().float().numpy()) for k, v in tiny._model.state_dict().items()}
    lm.load_weights([(k, v) for k, v in weights.items() if k in names], strict=True)
    return Model(wrapped, build_tokenizer())


@pytest.mark.skipif(importlib.util.find_spec("mlx") is None, reason="MLX is not installed here")
@pytest.mark.parametrize("name", ["llama", "gemma3"])
def test_mlx_and_torch_with_the_same_weights_generate_the_same_greedy_tokens(name):
    tiny = build_tiny_model(name)
    twin = build_mlx_twin(name, tiny)
    for prompt in PROMPTS:
        assert sample(twin, prompt, temperature=0) == sample(tiny, prompt, temperature=0)
    spec = [{"point": "resid_post", "layers": [1], "op": "scale", "strength": -1.0, "positions": "all"}]
    on = [run_generate(m, {"spec": spec, "temperature": 0, "control": False})["items"] for m in (twin, tiny)]
    assert [i["text"] for i in on[0]] == [i["text"] for i in on[1]]


GENERATE_WITHOUT_MLX = """
import sys
for name in ("mlx", "mlx.core", "mlx_lm", "mlx_vlm"):
    sys.modules[name] = None
import numpy as np
from tests.tiny_torch_models import KIT_MODELS, build_tiny_model
from mechbench_compute.ops import Context
from mechbench_compute.ops.text import generate
tiny = build_tiny_model("llama")
out = generate.run(Context(loaded=tiny), {"records": [{"id": "a", "user": "the cat"}]},
                   {"model": "tiny", "n": 2, "max_tokens": 3})
print(len(out["items"]), sys.modules["mlx"] is not None)
"""


def test_torch_generation_runs_where_mlx_cannot_be_imported():
    out = subprocess.run([sys.executable, "-c", GENERATE_WITHOUT_MLX], cwd=ROOT, capture_output=True,
                         text=True, timeout=300, check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split()[-2:] == ["2", "False"]


def test_the_throughput_measure_reports_prefill_and_decode_rates():
    tiny = build_tiny_model("llama")
    rows = measure_throughput(tiny, batch_sizes=(1, 2), prompt_tokens=8, new_tokens=4)
    assert [r["batch"] for r in rows] == [1, 2]
    for r in rows:
        assert r["prefill_tokens_per_s"] > 0 and r["decode_tokens_per_s"] > 0
        assert r["model_type"] == "llama" and r["accelerator"] == tiny.accelerator


def test_scoring_continuations_batches_them_and_reads_what_whole_forwards_read(tiny):
    prompt = PROMPTS[0]
    sequences = {"a": [8, 9], "b": [10, 11], "c": [12], "d": [7, 8, 9]}
    for size in (1, 3):
        scored = tiny.score_items(prompt, sequences, batch_size=size)
        for name, seq in sequences.items():
            assert scored[name] == pytest.approx(whole_logprob(tiny, prompt, seq), abs=1e-3)



def test_a_local_judge_grades_on_torch_and_repeats():
    from mechbench_compute.ops.eval.judge import run_judge

    tiny = build_tiny_model("llama")
    stories = [{"id": "s1", "text": "the cat sat"}, {"id": "s2", "text": "the dog ran"}]
    params = {"judge": {"model": "tiny/llama", "system": "Grade the story.", "max_tokens": 4},
              "scale": {"kind": "numeric", "min": 1, "max": 5}, "seed": 2}
    first = run_judge(dict(params), inputs={"records": stories}, model=tiny)
    again = run_judge(dict(params), inputs={"records": stories}, model=tiny)
    assert [r["id"] for r in first["items"]] == ["s1", "s2"]
    assert json.dumps(first["items"], sort_keys=True) == json.dumps(again["items"], sort_keys=True)


def test_a_checkpoint_loads_straight_onto_its_device_with_no_copy_of_the_whole_model(tmp_path, monkeypatch):
    from mechbench_compute.torch_backend.loading import load_transformers

    tiny = build_tiny_model("llama")
    tiny._model.save_pretrained(tmp_path)
    tiny.tokenizer.save_pretrained(tmp_path)
    moved: list[type] = []
    real_to = torch.nn.Module.to

    def watched(module, *args, **kwargs):
        moved.append(type(module))
        return real_to(module, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "to", watched)
    model, _ = load_transformers(str(tmp_path), classes={}, device=str(tiny.device), dtype=torch.float32)
    assert type(tiny._model) not in moved
    assert {p.device for p in model.parameters()} == {tiny.device}
    loaded, saved = model.state_dict(), tiny._model.state_dict()
    assert loaded.keys() == saved.keys()
    assert all(torch.equal(loaded[k], saved[k]) for k in saved)
