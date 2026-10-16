from __future__ import annotations

import dataclasses
import json

import mlx.core as mx
import numpy as np
import pytest

from mechbench_compute import architectures, lexicon
from mechbench_compute import model as model_mod
from mechbench_compute.dictionaries import train_dictionary as train_mod
from mechbench_compute.dictionaries import train_hub
from mechbench_compute.dictionaries.check_sae import check_sae
from mechbench_compute.dictionaries.encode_features import encode_features
from mechbench_compute.dictionaries.read_dictionary_weights import (
    read_dictionary_weights,
)
from mechbench_compute.dictionaries.train_corpus import TokenStream, open_stream
from mechbench_compute.dictionaries.train_dictionary import train_dictionary
from mechbench_compute.dictionaries.train_sae import Trainer, init_params
from mechbench_compute.dictionaries.train_settings import read_settings
from mechbench_compute.ops import Context
from mechbench_compute.ops.dictionary import encode as encode_op
from mechbench_compute.ops.dictionary import train as train_op
from tests.checkpoint_offers import due_only
from tests.tiny_models import build_tiny_model
from tests.tiny_tokenizer import build_tokenizer

WORDS = ["the", "cat", "sat", "on", "a", "mat", "and", "dog", "ran"]

LAYER = 2

SMALL = {"layer": LAYER, "width": 64, "k": 4, "tokens": 4096, "context": 16, "batch_tokens": 128,
         "buffer_tokens": 512, "heldout_tokens": 60, "checkpoint_tokens": 1024, "seed": 3}


def make_records(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return [{"id": str(i), "text": " ".join(rng.choice(WORDS, size=int(rng.integers(5, 30))))}
            for i in range(n)]


@pytest.fixture(scope="module")
def tiny():
    return build_tiny_model("gemma3")


@pytest.fixture(scope="module")
def trained(tiny):
    return train_dictionary(tiny, make_records(), SMALL)


def draw_planted(rng, atoms, n, active):
    c = np.zeros((n, len(atoms)))
    for i in range(n):
        j = rng.choice(len(atoms), active, replace=False)
        c[i, j] = rng.uniform(0.5, 1.5, active)
    return (c @ atoms).astype(np.float32)


class TestPlantedDictionary:
    @pytest.mark.parametrize("activation", ["batchtopk", "jumprelu"])
    def test_the_trainer_recovers_known_directions(self, activation):
        d, n_atoms, active, batch, steps = 32, 48, 3, 256, 1500
        rng = np.random.default_rng(0)
        atoms = rng.normal(size=(n_atoms, d))
        atoms /= np.linalg.norm(atoms, axis=1, keepdims=True)
        s = read_settings({"layer": 0, "width": 64, "k": active, "l0": active, "tokens": batch * steps,
                           "batch_tokens": batch, "buffer_tokens": batch, "activation": activation,
                           "lr": 3e-3})
        first = mx.array(draw_planted(rng, atoms, batch, active))
        scale = (d / float(mx.mean(mx.sum(first * first, axis=-1)).item())) ** 0.5
        trainer = Trainer(s, init_params(s, first * scale, 0), d)
        for _ in range(steps):
            metrics = trainer.train(mx.array(draw_planted(rng, atoms, batch, active)) * scale)
        w = np.array(trainer.params["w_dec"], dtype=np.float64)
        assert np.allclose(np.linalg.norm(w, axis=1), 1.0, atol=1e-5)
        best = (atoms @ w.T).max(axis=1)
        assert (best > 0.9).mean() >= 0.85, np.sort(best)[:8]
        assert metrics["variance_explained"] > 0.9
        assert metrics["l0"] == pytest.approx(active, rel=0.5)


class TestTheEmittedDictionary:
    def test_it_passes_the_readers_checks(self, trained):
        space, activation = check_sae(trained)
        assert space == {"model": None, "layer": LAYER, "point": "resid_post", "head": None, "d": 32}
        assert activation == {"fn": "jumprelu", "training": "batchtopk", "k": 4}
        weights = read_dictionary_weights(trained)
        assert weights["w_enc"].shape == (32, 64) and weights["w_dec"].shape == (64, 32)
        items = lexicon.items_of(trained)
        assert [it["index"] for it in items] == list(range(64))
        assert {"id", "index", "vector", "encoder", "norm", "b_enc", "threshold"} <= set(items[0])
        assert trained["derivation"] == "sae" and "published" not in trained

    def test_the_header_says_how_it_was_trained_and_how_well_it_reads(self, trained):
        t = trained["trained"]
        assert t["tokens"] == 4096 and t["steps"] == 32 and t["seed"] == 3
        assert t["scale"] > 0 and len(t["first_chunk_sha256"]) == 64 and len(t["heldout_sha256"]) == 64
        assert t["corpus"][0]["records"]["count"] == 40 and t["corpus"][0]["documents"] > 40
        assert t["hyperparameters"]["width"] == 64 and t["hyperparameters"]["steps_per_chunk"] == 4
        assert [p["step"] for p in t["curve"]] == list(range(1, 33))
        f = trained["fidelity"]
        assert 40 < f["tokens"] <= 4 * 15 and 0 < f["variance_explained"] < 1
        assert 0 <= f["dead"] <= 1 and 0 <= f["dense"] <= 1 and f["l0"] > 0
        loss = f["loss"]
        assert loss["delta"] == pytest.approx(loss["spliced"] - loss["clean"], abs=2e-6)
        assert loss["zero_delta"] == pytest.approx(loss["zero"] - loss["clean"], abs=2e-6)
        assert loss["recovered"] == pytest.approx(1 - loss["delta"] / loss["zero_delta"], abs=1e-4)

    def test_two_runs_are_identical(self, tiny, trained):
        again = train_dictionary(tiny, make_records(), SMALL)
        a, b = read_dictionary_weights(trained), read_dictionary_weights(again)
        for k in ("w_enc", "w_dec", "b_enc", "b_dec", "threshold"):
            assert np.array_equal(a[k], b[k]), k
        assert again["fidelity"] == trained["fidelity"]


class TestScaleFolding:
    def test_encode_on_raw_activations_gives_the_features_the_trainer_saw(self, tiny, monkeypatch):
        seen = {}
        fold = train_mod.fold

        def spy(params, thresholds, scale):
            seen.update(params={k: np.array(v) for k, v in params.items()}, threshold=thresholds, scale=scale)
            return fold(params, thresholds, scale)

        monkeypatch.setattr(train_mod, "fold", spy)
        out = train_dictionary(tiny, make_records(), {**SMALL, "tokens": 1024})
        name = f"blocks.{LAYER}.resid_post"
        ids = mx.array([[2, 7, 8, 9, 10, 11, 12, 13, 14, 15]], dtype=mx.int32)
        x = np.array(tiny.run(ids, capture=[name]).cache[name].astype(mx.float32))[0, 1:]
        p = seen["params"]
        pre = (x.astype(np.float64) * seen["scale"] - p["b_dec"]) @ p["w_enc"] + p["b_enc"]
        trainer_saw = np.where(pre > seen["threshold"], pre, 0.0)
        assert seen["scale"] != pytest.approx(1.0)
        got = encode_features(x, read_dictionary_weights(out))
        assert np.array_equal(got != 0, trainer_saw != 0)
        np.testing.assert_allclose(got, trainer_saw, rtol=1e-4, atol=1e-4)

    def test_dictionary_encode_reads_the_trained_dictionary(self, tiny, trained):
        records = [{"id": "a", "user": "the cat sat on a mat"}, {"id": "b", "text": "a dog ran", "template": "raw"}]
        out = encode_op.run(Context(loaded=tiny), {"records": records, "dictionary": trained}, {})
        assert out["item_kind"] == "activations/feature" and lexicon.items_of(out)
        assert out["dictionary"]["activation"]["fn"] == "jumprelu"
        assert out["fidelity"]["published_l0"] is None and out["fidelity"]["variance_explained"] is not None


def spool_roundtrip(state, tmp_path):
    from mechbench_schema import dump_canonical, load_raw

    np.savez(tmp_path / "weights.npz", **{k: np.asarray(v) for k, v in state["weights"].items()})
    arrays = {k: np.asarray(v) for k, v in state["opt_state"].items() if hasattr(v, "shape")}
    scalars = {k: v for k, v in state["opt_state"].items() if not hasattr(v, "shape")}
    np.savez(tmp_path / "opt.npz", **arrays)
    meta = load_raw(dump_canonical({"step": int(state["step"]), "opt_scalars": scalars,
                                    "np_rng_json": json.dumps(state["np_rng"]), "mx_key": None}))
    with np.load(tmp_path / "weights.npz", allow_pickle=False) as f:
        weights = {k: f[k] for k in f.files}
    with np.load(tmp_path / "opt.npz", allow_pickle=False) as f:
        opt = {k: f[k] for k in f.files}
    opt.update(meta["opt_scalars"])
    return {"step": meta["step"], "weights": weights, "opt_state": opt,
            "np_rng": json.loads(meta["np_rng_json"]), "mx_key": None}


class Interrupted(Exception):
    pass


class TestResume:
    def test_an_interrupted_run_resumes_to_the_uninterrupted_result(self, tiny, trained, tmp_path):
        saved = []

        def interrupt(state):
            saved.append(spool_roundtrip(state, tmp_path))
            raise Interrupted

        with pytest.raises(Interrupted):
            train_dictionary(tiny, make_records(), SMALL, on_checkpoint=due_only(interrupt))
        assert saved[0]["step"] == 8
        restored = []
        resumed = train_dictionary(tiny, make_records(), SMALL, resume_state=saved[0],
                                   on_item=lambda *a: restored.append(a))
        assert restored[:16] == [(None, None, True)] * 16 and restored[16] == ()
        a, b = read_dictionary_weights(trained), read_dictionary_weights(resumed)
        for k in ("w_enc", "w_dec", "b_enc", "b_dec", "threshold"):
            assert np.array_equal(a[k], b[k]), k
        assert resumed["fidelity"] == trained["fidelity"]
        assert resumed["trained"] == trained["trained"]

    def test_checkpoints_fall_at_the_end_of_a_buffer(self, tiny):
        steps = []
        train_dictionary(tiny, make_records(), SMALL, on_checkpoint=due_only(lambda st: steps.append(st["step"])))
        assert steps == [8, 16, 24]

    def test_a_state_from_another_trainer_is_refused(self, tiny, tmp_path):
        saved = []
        train_dictionary(tiny, make_records(), {**SMALL, "tokens": 2048}, on_checkpoint=due_only(saved.append))
        with pytest.raises(ValueError, match="its `activation` differs"):
            train_dictionary(tiny, make_records(), {**SMALL, "activation": "jumprelu"}, resume_state=saved[0])


class TestProgress:
    def test_every_harvested_batch_and_every_step_is_reported(self, tiny):
        started, items = [], []
        train_dictionary(tiny, make_records(), {**SMALL, "tokens": 1024}, on_start=started.append,
                         on_item=lambda *a: items.append(a))
        windows = -(-60 // 15)
        assert started == [2 * 8 + 3 * windows] and len(items) == started[0]


class FakeTable:
    def __init__(self, values):
        self.values = values

    def column(self, name):
        return type("Column", (), {"to_pylist": lambda _self: list(self.values)})()


class FakeParquet:
    def __init__(self, groups, field):
        self.groups, self.field = groups, field
        self.num_row_groups = len(groups)

    def read_row_group(self, g, columns):
        assert columns == [self.field]
        return FakeTable(self.groups[g])


@pytest.fixture
def fake_hub(monkeypatch):
    files = {"data/a.parquet": [["the cat sat", "a dog ran"], ["on a mat"]],
             "data/b.parquet": [["the dog sat on the cat", ""], ["a mat ran"]],
             "chat/c.parquet": [[[{"role": "user", "content": "the cat"},
                                  {"role": "assistant", "content": "sat on a mat"}]]]}
    configs = [{"config_name": "default", "data_files": [{"split": "train", "path": "data/*"}]},
               {"config_name": "chat", "data_files": "chat/*"}]
    calls = []

    def read_dataset(repo, revision):
        calls.append((repo, revision))
        return "f" * 40, [*files, "README.md"], configs

    def open_parquet(repo, commit, path):
        field = "messages" if path.startswith("chat/") else "text"
        return FakeParquet(files[path], field)

    monkeypatch.setattr(train_hub, "read_dataset", read_dataset)
    monkeypatch.setattr(train_hub, "open_parquet", open_parquet)
    return calls


class TestHubCorpus:
    def test_a_config_names_its_files_and_the_order_is_the_seeds(self, fake_hub):
        spec = {"hub": {"repo": "you/web", "revision": "v1", "field": "text"}}
        source = train_hub.HubSource(spec["hub"], 5, 0)
        assert source.files == ["data/a.parquet", "data/b.parquet"] and fake_hub == [("you/web", "v1")]
        docs = [source.next_doc()["text"] for _ in range(10)]
        assert sorted(docs[:5]) == sorted(["the cat sat", "a dog ran", "on a mat", "the dog sat on the cat",
                                           "a mat ran"])
        again = train_hub.HubSource(spec["hub"], 5, 0)
        assert [again.next_doc()["text"] for _ in range(10)] == docs
        other = train_hub.HubSource(spec["hub"], 6, 0)
        assert [other.next_doc()["text"] for _ in range(10)] != docs
        assert source.describe()["hub"]["commit"] == "f" * 40 and source.describe()["epochs"] in (1, 2)

    def test_a_cursor_resumes_mid_group(self, fake_hub):
        spec = {"repo": "you/web", "field": "text"}
        first = train_hub.HubSource(spec, 1, 0)
        for _ in range(3):
            first.next_doc()
        cursor = first.cursor()
        rest = [first.next_doc() for _ in range(6)]
        second = train_hub.HubSource(spec, 1, 0)
        second.seek(cursor)
        assert [second.next_doc() for _ in range(6)] == rest

    def test_a_chat_column_is_rendered_through_the_template(self, fake_hub):
        tok = build_tokenizer()
        stream = open_stream({"hub": {"repo": "you/chat", "config": "chat", "chat": "messages"}}, [], tok,
                             seed=0, context=8)
        window = stream.next_window()
        assert tok.decode(window).split()[:4] == ["<start>", "user", "the", "cat"]

    def test_sources_interleave_by_weight(self, fake_hub):
        tok = build_tokenizer()
        corpus = [{"hub": {"repo": "you/web"}, "weight": 2}, {"hub": {"repo": "you/chat", "config": "chat",
                                                                       "chat": "messages"}, "weight": 1}]
        stream = open_stream(corpus, [], tok, seed=0, context=4)
        for _ in range(12):
            stream.next_window()
        a, b = stream.counts
        assert abs(a - 2 * b) <= 2

    def test_unknown_configs_and_splits_are_refused_by_name(self, fake_hub):
        with pytest.raises(ValueError, match="no config 'sample'; its configs are default, chat"):
            train_hub.HubSource({"repo": "you/web", "config": "sample"}, 0, 0)
        with pytest.raises(ValueError, match="no split 'test'; its splits are train"):
            train_hub.HubSource({"repo": "you/web", "split": "test"}, 0, 0)
        with pytest.raises(ValueError, match="takes repo"):
            train_hub.HubSource({"repo": "you/web", "column": "x"}, 0, 0)

    def test_files_name_the_parquet_directly(self, fake_hub):
        assert train_hub.HubSource({"repo": "you/web", "files": "data/b*"}, 0, 0).files == ["data/b.parquet"]

    def test_without_pyarrow_the_hub_is_refused_by_name(self, monkeypatch):
        import builtins

        real = builtins.__import__

        def no_pyarrow(name, *args, **kwargs):
            if name.startswith("pyarrow"):
                raise ImportError(name)
            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_pyarrow)
        with pytest.raises(RuntimeError, match=r"mechbench-compute\[dictionaries\]"):
            train_hub.open_parquet("you/web", "f" * 40, "data/a.parquet")

    def test_pyarrow_reads_a_parquet_file(self, tmp_path, monkeypatch):
        pa = pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        pq.write_table(pa.table({"text": ["the cat", "a dog", "on a mat"]}), tmp_path / "a.parquet",
                       row_group_size=2)
        import huggingface_hub

        class LocalFs:
            def open(self, path, mode):
                assert path == f"datasets/you/web@{'f' * 40}/data/a.parquet"
                return open(tmp_path / "a.parquet", mode)

        monkeypatch.setattr(huggingface_hub, "HfFileSystem", LocalFs)
        f = train_hub.open_parquet("you/web", "f" * 40, "data/a.parquet")
        assert f.num_row_groups == 2
        assert f.read_row_group(1, columns=["text"]).column("text").to_pylist() == ["on a mat"]


class TestRefusals:
    def test_a_later_derivation_is_refused_by_name(self):
        with pytest.raises(ValueError, match="the derivation 'transcoder' is not trained yet"):
            train_op.run(Context(), {}, {**SMALL, "derivation": "transcoder"})

    def test_records_and_a_corpus_together(self, tiny):
        with pytest.raises(ValueError, match="this node gives both"):
            train_dictionary(tiny, make_records(), {**SMALL, "corpus": {"hub": {"repo": "x"}}})

    def test_neither_records_nor_a_corpus(self, tiny):
        with pytest.raises(ValueError, match="give a `corpus`"):
            train_dictionary(tiny, [], SMALL)

    def test_a_point_with_heads(self, tiny):
        with pytest.raises(ValueError, match="'attn.q' is not one"):
            train_dictionary(tiny, make_records(), {**SMALL, "point": "attn.q"})

    def test_a_layer_past_the_model(self, tiny):
        with pytest.raises(ValueError, match="`layer` 9 is not a layer of the model; it has 4"):
            train_dictionary(tiny, make_records(), {**SMALL, "layer": 9})

    @pytest.mark.parametrize("params,match", [
        ({"k": 65}, "more features than the dictionary has"),
        ({"buffer_tokens": 64}, "fewer activations than one batch"),
        ({"context": 1}, "`context` is a whole number, at least 2"),
        ({"layer": None}, "`layer` names the layer"),
    ])
    def test_settings(self, params, match):
        with pytest.raises(ValueError, match=match):
            read_settings({**SMALL, **params})


MODEL = "tiny/gemma3"


@pytest.fixture
def tiny_hub(monkeypatch, tmp_path):
    def load_tiny(model_id, **_):
        t = build_tiny_model("gemma3")
        return t._model, t._processor

    architecture = dataclasses.replace(architectures.BY_MODEL_TYPE["gemma3"], load=load_tiny)
    monkeypatch.setattr("mechbench_compute.hub.ensure_model", lambda model_id, **_: (model_id, "0" * 40, tmp_path))
    monkeypatch.setattr(model_mod, "_peek_config", lambda _path: {"model_type": "gemma3"})
    monkeypatch.setattr(architectures, "for_type", lambda _t: architecture)


class TestInAGraph:
    def test_train_then_encode(self, tiny_hub, monkeypatch):
        from mechbench_compute import bench
        from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

        monkeypatch.setattr(bench, "emit", lambda path, payload, **kw: {"path": path})
        records = make_records(12)
        nodes = [{"id": "train", "block": "dictionary/train",
                  "params": {**SMALL, "tokens": 1024, "model": MODEL}, "inputs": {"records": records}},
                 {"id": "encode", "block": "dictionary/encode", "params": {"model": MODEL},
                  "inputs": {"records": records[:2]}}]
        edges = [{"from": {"node": "train"}, "to": {"node": "encode", "port": "dictionary"}}]
        spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                            extra={"graph": {"dataflow": 2, "nodes": nodes, "edges": edges}})
        out = ProtocolExecutor().run(spec).payload["outputs"]["encode"]
        assert out["item_kind"] == "activations/feature"
        assert out["dictionary"]["reads"][0]["model"] == MODEL
        assert out["fidelity"]["positions"] > 0


def test_the_stream_packs_documents_into_windows():
    tok = build_tokenizer()
    stream = open_stream(None, [{"id": "a", "text": "the cat"}, {"id": "b", "text": "a dog ran"}], tok,
                         seed=0, context=4)
    assert isinstance(stream, TokenStream)
    windows = [stream.next_window() for _ in range(3)]
    flat = [t for w in windows for t in w]
    bos = tok.bos_token_id
    assert all(len(w) == 4 for w in windows) and flat.count(bos) >= 3
    state = stream.state()
    after = [stream.next_window() for _ in range(3)]
    other = open_stream(None, [{"id": "a", "text": "the cat"}, {"id": "b", "text": "a dog ran"}], tok,
                        seed=0, context=4)
    other.restore(state)
    assert [other.next_window() for _ in range(3)] == after
