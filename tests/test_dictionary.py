from __future__ import annotations

import dataclasses
import hashlib
import json

import mlx.core as mx
import numpy as np
import pytest
from safetensors.numpy import save_file

from mechbench_compute import architectures, lexicon
from mechbench_compute import model as model_mod
from mechbench_compute import resume as rm
from mechbench_compute.ops import Context
from mechbench_compute.ops.dictionary import encode as encode_op
from mechbench_compute.ops.dictionary import load as load_op
from mechbench_compute.ops.dictionary.encode import encode_records
from mechbench_compute.ops.dictionary.load import load_dictionary
from tests.tiny_models import build_tiny_model

D = 32

WIDTH = 48

LAYER = 2

REPO = "google/gemma-scope-2-tiny-it"

RECORDS = [{"id": "a", "user": "the cat sat on a mat", "coords": {"animal": "cat"}},
           {"id": "b", "user": "a dog ran", "coords": {"animal": "dog"}}]


def write_dictionary(root, path, *, kind="sae", hook=f"model.layers.{LAYER}.output", hook_out=None,
                     affine=False, weights=None, architecture="jump_relu"):
    folder = root / path
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({
        "hf_hook_point_in": hook, "hf_hook_point_out": hook_out or hook, "width": WIDTH,
        "model_name": "tiny/gemma3", "architecture": architecture, "l0": 7,
        "affine_connection": affine, "type": kind}))
    if weights is None:
        rng = np.random.default_rng(5)
        weights = {"w_enc": rng.normal(size=(D, WIDTH)).astype(np.float32) * 0.3,
                   "b_enc": rng.normal(size=WIDTH).astype(np.float32) * 0.1,
                   "threshold": np.abs(rng.normal(size=WIDTH)).astype(np.float32),
                   "w_dec": rng.normal(size=(WIDTH, D)).astype(np.float32) * 0.3,
                   "b_dec": rng.normal(size=D).astype(np.float32) * 0.1}
    save_file(weights, str(folder / "params.safetensors"))
    return weights


def identity_weights(d=D):
    return {"w_enc": np.eye(d, dtype=np.float32), "b_enc": np.zeros(d, dtype=np.float32),
            "threshold": np.full(d, -1e30, dtype=np.float32),
            "w_dec": np.eye(d, dtype=np.float32), "b_dec": np.zeros(d, dtype=np.float32)}


def load(root, path="resid_post/layer_2_width_48_l0_small", **params):
    return load_dictionary({"repo": REPO, "path": path, **params},
                           fetch=lambda repo, p, rev: (root / p, "c" * 40),
                           read_model_type=lambda model_id: "gemma3")


@pytest.fixture(scope="module")
def tiny():
    return build_tiny_model("gemma3")


@pytest.fixture
def sae(tmp_path):
    weights = write_dictionary(tmp_path, "resid_post/layer_2_width_48_l0_small")
    return load(tmp_path), weights


def read_residuals(model, record, point="resid_post"):
    from mechbench_compute.distill import render

    r = render(model, record)
    name = f"blocks.{LAYER}.{point}"
    acts = np.array(model.run(r.array, capture=[name]).cache[name].astype(mx.float32))[0]
    return r.ids, acts


class TestLoad:
    def test_one_item_per_feature_with_its_weights(self, sae, tmp_path):
        dictionary, w = sae
        assert dictionary["item_kind"] == "direction/dictionary" and dictionary["storage"] == "tensor"
        items = list(lexicon.items_of(dictionary))
        assert [it["index"] for it in items] == list(range(WIDTH))
        for it in items[:5]:
            i = it["index"]
            np.testing.assert_array_equal(it["vector"], w["w_dec"][i])
            np.testing.assert_array_equal(it["encoder"], w["w_enc"][:, i])
            assert it["norm"] == pytest.approx(float(np.linalg.norm(w["w_dec"][i].astype(np.float64))))
            assert it["b_enc"] == float(w["b_enc"][i]) and it["threshold"] == float(w["threshold"][i])

    def test_the_header_says_what_it_reads_and_where_it_came_from(self, sae, tmp_path):
        dictionary, w = sae
        params = tmp_path / "resid_post/layer_2_width_48_l0_small/params.safetensors"
        assert dictionary["derivation"] == "sae"
        space = {"model": "tiny/gemma3", "layer": LAYER, "point": "resid_post", "head": None, "d": D}
        assert dictionary["reads"] == [space] and dictionary["writes"] == [space]
        assert dictionary["model"] == {"id": "tiny/gemma3", "architecture": "gemma3"}
        assert (dictionary["width"], dictionary["d_in"], dictionary["d_out"]) == (WIDTH, D, D)
        assert dictionary["activation"] == {"fn": "jumprelu"}
        assert dictionary["published"] == {"l0": 7}
        hub = dictionary["source"]["hub"]
        assert (hub["repo"], hub["revision"], hub["commit"]) == (REPO, "main", "c" * 40)
        assert hub["files"]["params.safetensors"] == hashlib.sha256(params.read_bytes()).hexdigest()
        assert dictionary["b_dec"] == [float(x) for x in w["b_dec"]]

    def test_the_hooks_map_to_points(self, tmp_path):
        for hook, point in (("model.layers.3.post_feedforward_layernorm.output", "mlp_out"),
                            ("model.layers.3.self_attn.o_proj.input", "attn.o_in")):
            path = f"x/{point}"
            write_dictionary(tmp_path, path, hook=hook)
            assert load(tmp_path, path)["reads"][0]["point"] == point

    def test_loading_twice_writes_the_same_shards(self, tmp_path):
        write_dictionary(tmp_path, "p")
        first, second = load(tmp_path, "p"), load(tmp_path, "p")
        assert first["shards"] == second["shards"]
        assert rm.content_hash(first) == rm.content_hash(second)


class TestLoadRefuses:
    @pytest.mark.parametrize("kind", ["transcoder", "crosscoder"])
    def test_the_derivations_it_does_not_read_yet(self, tmp_path, kind):
        write_dictionary(tmp_path, "p", kind=kind)
        with pytest.raises(ValueError, match=f"is a {kind}"):
            load(tmp_path, "p")

    @pytest.mark.parametrize(("param", "value", "found"), [
        ("derivation", "transcoder", "sae"), ("point", "mlp_out", "resid_post"), ("layer", 5, "2"),
        ("activation", "topk", "jumprelu"), ("sha256", "0" * 64, "[0-9a-f]{64}")])
    def test_a_source_that_says_otherwise(self, sae, tmp_path, param, value, found):
        with pytest.raises(ValueError, match=f"{param} .* was asked for, and the source is .*{found}"):
            load(tmp_path, **{param: value})

    def test_a_derivation_outside_the_three(self, sae, tmp_path):
        with pytest.raises(ValueError, match="derivation 'pca' is not one of sae, transcoder, crosscoder"):
            load(tmp_path, derivation="pca")

    def test_an_affine_skip_and_an_unknown_hook(self, tmp_path):
        write_dictionary(tmp_path, "affine", affine=True)
        with pytest.raises(ValueError, match="affine skip connection"):
            load(tmp_path, "affine")
        write_dictionary(tmp_path, "hook", hook="model.layers.2.mlp.up_proj.output")
        with pytest.raises(ValueError, match="is not one this operation maps to a point"):
            load(tmp_path, "hook")

    def test_a_folder_without_the_layout(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValueError, match="Gemma Scope 2's layout"):
            load(tmp_path, "empty")


class TestEncode:
    def test_an_identity_dictionary_reconstructs_everything_with_every_feature(self, tiny, tmp_path):
        write_dictionary(tmp_path, "id", weights=identity_weights())
        out = encode_records(tiny, RECORDS, load(tmp_path, "id"), {})
        fidelity = out["fidelity"]
        assert fidelity["variance_explained"] == 1.0 and fidelity["mse"] == 0.0
        assert fidelity["l0"] == D and fidelity["inactive"] == 0

    def test_values_are_the_jumprelu_of_the_activations(self, tiny, sae):
        dictionary, w = sae
        out = encode_records(tiny, RECORDS[:1], dictionary, {})
        ids, acts = read_residuals(tiny, RECORDS[0])
        assert ids[0] == tiny.tokenizer.bos_token_id
        x = acts[1:]
        pre = x @ w["w_enc"] + w["b_enc"]
        f = np.where(pre > w["threshold"], pre, 0).astype(np.float32)
        want = {(p + 1, j): float(f[p, j]) for p, j in zip(*np.nonzero(f), strict=True)}
        got = {(it["position"], it["feature"]): it["value"] for it in lexicon.items_of(out)}
        assert got.keys() == want.keys()
        for key, value in want.items():
            assert got[key] == pytest.approx(value, rel=1e-5, abs=1e-6)
        recon = f @ w["w_dec"] + w["b_dec"]
        x64 = x.astype(np.float64)
        ve = 1 - np.sum((x64 - recon) ** 2) / np.sum((x64 - x64.mean(axis=0)) ** 2)
        assert out["fidelity"]["variance_explained"] == pytest.approx(ve, abs=1e-5)
        assert out["fidelity"]["l0"] == pytest.approx(len(want) / len(x))
        assert out["fidelity"]["positions"] == len(x) and out["fidelity"]["published_l0"] == 7
        assert out["fidelity"]["inactive"] == WIDTH - len({j for _, j in want})

    def test_items_carry_the_record_and_the_reading(self, tiny, sae):
        out = encode_records(tiny, RECORDS, sae[0], {})
        it = lexicon.items_of(out)[0]
        assert it["coords"] == {"animal": "cat", "model": "base"}
        assert set(it["token"]) == {"id", "text"} and it["id"] == "a"
        assert out["point"] == {"point": "resid_post", "layer": LAYER}
        assert out["dictionary"]["source"]["hub"]["repo"] == REPO

    def test_last_position_and_kept_features(self, tiny, sae):
        last = encode_records(tiny, RECORDS, sae[0], {"position": "last"})
        lengths = {r["id"]: len(read_residuals(tiny, r)[0]) for r in RECORDS}
        assert {(it["id"], it["position"]) for it in lexicon.items_of(last)} <= {
            (k, n - 1) for k, n in lengths.items()}
        assert last["fidelity"]["positions"] == len(RECORDS)
        every = encode_records(tiny, RECORDS, sae[0], {})
        kept = encode_records(tiny, RECORDS, sae[0], {"features": [3, 1]})
        assert kept["features"] == [1, 3]
        assert all(it["feature"] in (1, 3) for it in lexicon.items_of(kept))
        assert kept["fidelity"] == every["fidelity"]

    def test_the_bos_position_is_read_when_asked(self, tiny, sae):
        skipped = encode_records(tiny, RECORDS[:1], sae[0], {})
        kept = encode_records(tiny, RECORDS[:1], sae[0], {"skip_bos": False})
        assert kept["fidelity"]["positions"] == skipped["fidelity"]["positions"] + 1

    @pytest.mark.parametrize("name", ["gemma3", "llama"])
    def test_attention_dictionaries_read_the_heads_concatenated(self, name, tmp_path):
        model = build_tiny_model(name)
        write_dictionary(tmp_path, "o", hook=f"model.layers.{LAYER}.self_attn.o_proj.input",
                         weights=identity_weights(32))
        out = encode_records(model, RECORDS[:1], load(tmp_path, "o"), {"skip_bos": False})
        ids, heads = read_residuals(model, RECORDS[0], "attn.per_head_out")
        assert heads.ndim == 3
        z = heads.transpose(1, 0, 2).reshape(heads.shape[1], -1)
        got = {(it["position"], it["feature"]): it["value"] for it in lexicon.items_of(out)}
        assert all(got.get((p, j), 0.0) == float(z[p, j]) for p in range(len(ids)) for j in range(32))

    def test_a_dictionary_at_the_mlp_input_reads_mlp_in_norm(self, tiny, tmp_path):
        write_dictionary(tmp_path, "in", hook=f"model.layers.{LAYER}.pre_feedforward_layernorm.output",
                         weights=identity_weights())
        out = encode_records(tiny, RECORDS[:1], load(tmp_path, "in"), {"skip_bos": False})
        ids, acts = read_residuals(tiny, RECORDS[0], "mlp.in_norm")
        assert out["point"] == {"point": "mlp.in_norm", "layer": LAYER}
        got = {(it["position"], it["feature"]): it["value"] for it in lexicon.items_of(out)}
        assert all(got.get((p, j), 0.0) == float(acts[p, j]) for p in range(len(ids)) for j in range(D))

    def test_two_runs_are_identical(self, tiny, sae):
        first = encode_records(tiny, RECORDS, sae[0], {})
        second = encode_records(tiny, RECORDS, sae[0], {})
        assert rm.content_hash(lexicon.canonical_collection(first)) == rm.content_hash(
            lexicon.canonical_collection(second))


class TestEncodeRefuses:
    def test_a_width_that_is_not_the_points(self, tiny, tmp_path):
        rng = np.random.default_rng(0)
        write_dictionary(tmp_path, "wide", weights={
            "w_enc": rng.normal(size=(16, 8)).astype(np.float32), "b_enc": np.zeros(8, np.float32),
            "threshold": np.zeros(8, np.float32), "w_dec": rng.normal(size=(8, 16)).astype(np.float32),
            "b_dec": np.zeros(16, np.float32)})
        with pytest.raises(ValueError, match="reads 16-wide activations"):
            encode_records(tiny, RECORDS, load(tmp_path, "wide"), {})

    def test_a_point_the_architecture_does_not_offer(self, tmp_path):
        write_dictionary(tmp_path, "tc", hook=f"model.layers.{LAYER}.pre_feedforward_layernorm.output")
        dictionary = load(tmp_path, "tc")
        with pytest.raises(ValueError, match="mlp.in_norm, which Llama does not offer"):
            encode_records(build_tiny_model("llama"), RECORDS, dictionary, {})

    def test_a_derivation_it_does_not_read(self, tiny, sae):
        with pytest.raises(ValueError, match="the dictionary is a 'transcoder'"):
            encode_records(tiny, RECORDS, {**sae[0], "derivation": "transcoder"}, {})

    def test_a_layer_past_the_model(self, tiny, tmp_path):
        write_dictionary(tmp_path, "deep", hook="model.layers.9.output")
        with pytest.raises(ValueError, match="reads layer 9; the model has 4"):
            encode_records(tiny, RECORDS, load(tmp_path, "deep"), {})


class TestTheOperations:
    def test_run_reads_its_ports(self, tiny, sae):
        out = encode_op.run(Context(loaded=tiny), {"records": RECORDS, "dictionary": sae[0]}, {})
        assert out["item_kind"] == "activations/feature" and "adapted_fidelity" not in out

    def test_load_run_fetches_through_the_hub(self, tmp_path, monkeypatch):
        write_dictionary(tmp_path, "p")
        monkeypatch.setattr(load_op, "fetch_folder", lambda repo, path, rev: (tmp_path / path, "d" * 40))
        monkeypatch.setattr(load_op, "read_architecture", lambda model_id: "gemma3")
        out = load_op.run(Context(), {}, {"repo": REPO, "path": "p"})
        assert out["source"]["hub"]["commit"] == "d" * 40

    def test_the_kinds_speak(self, tiny, sae):
        from mechbench_compute.expr.engine import load_engine
        from mechbench_compute.lexicon import kinds as K

        dictionary = sae[0]
        said = load_engine().render(K.BY_KIND["direction/dictionary"].speak,
                                    [dict(lexicon.items_of(dictionary)[0])], header=dictionary).values
        assert said[0].startswith("feature 0 of a 48-wide sae at resid_post layer 2: decoder norm ")
        out = encode_records(tiny, RECORDS[:1], dictionary, {})
        speak = K.BY_KIND["activations/feature"].speak
        item = lexicon.items_of(out)[0]
        plain = load_engine().render(speak, [item], header=out).values[0]
        assert "The dictionary explains" in plain and "under the adapter" not in plain
        both = {**out, "adapted_fidelity": {"variance_explained": 0.5, "l0": 3.0}}
        assert "; under the adapter, 0.5 at L0 3.0." in load_engine().render(speak, [item], header=both).values[0]


MODEL = "tiny/gemma3"


@pytest.fixture
def tiny_hub(monkeypatch, tmp_path):
    def load_tiny(model_id, **_):
        t = build_tiny_model("gemma3")
        return t._model, t._processor

    architecture = dataclasses.replace(architectures.BY_MODEL_TYPE["gemma3"], load=load_tiny)
    monkeypatch.setattr("mechbench_compute.hub.ensure_model",
                        lambda model_id, **_: (model_id, "0" * 40, tmp_path))
    monkeypatch.setattr(model_mod, "_peek_config", lambda _path: {"model_type": "gemma3"})
    monkeypatch.setattr(architectures, "for_type", lambda _t: architecture)


class TestAdaptedReading:
    def test_base_and_adapted_side_by_side(self, tiny_hub, tmp_path, monkeypatch):
        from mechbench_compute import bench
        from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

        monkeypatch.setattr(bench, "emit", lambda path, payload, **kw: {"path": path})
        write_dictionary(tmp_path, "p")
        monkeypatch.setattr(load_op, "fetch_folder", lambda repo, path, rev: (tmp_path / path, "c" * 40))
        monkeypatch.setattr(load_op, "read_architecture", lambda model_id: "gemma3")
        nodes = [
            {"id": "train", "block": "adapter/train",
             "params": {"model": MODEL, "target": {"uniform": ["cat", "dog"]}, "steps": 4, "lr": 5e-2,
                        "seed": 3, "closer": " mat", "lora": {"rank": 2, "alpha": 4}},
             "inputs": {"records": RECORDS}},
            {"id": "load", "block": "dictionary/load", "params": {"repo": REPO, "path": "p"}},
            {"id": "encode", "block": "dictionary/encode", "params": {"model": MODEL},
             "inputs": {"records": RECORDS}},
        ]
        edges = [{"from": {"node": "load"}, "to": {"node": "encode", "port": "dictionary"}},
                 {"from": {"node": "train"}, "to": {"node": "encode", "port": "adapted"}}]

        def run_once():
            spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                                extra={"graph": {"dataflow": 2, "nodes": nodes, "edges": edges}})
            return ProtocolExecutor().run(spec).payload["outputs"]["encode"]

        out = run_once()
        readings = {it["coords"]["model"] for it in lexicon.items_of(out)}
        assert readings == {"base", "adapted"}
        base, adapted = out["fidelity"], out["adapted_fidelity"]
        assert base["positions"] == adapted["positions"]
        assert base != adapted
        assert out["fidelity_drop"] == pytest.approx(
            base["variance_explained"] - adapted["variance_explained"], abs=1e-6)
        again = run_once()
        assert rm.content_hash(lexicon.canonical_collection(out)) == rm.content_hash(
            lexicon.canonical_collection(again))
