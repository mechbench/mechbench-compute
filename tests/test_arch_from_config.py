from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from mechbench_compute._arch import read_arch_from_config
from mechbench_compute.architectures import ARCHITECTURES

FIXTURES = sorted((Path(__file__).parent / "fixtures" / "configs").glob("*.json"))


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_a_config_reads_as_the_loader_reads_it(path):
    fixture = json.loads(path.read_text())
    arch = dataclasses.asdict(read_arch_from_config(fixture["config"], fixture["repo"]))
    arch["global_layers"] = list(arch["global_layers"])
    assert arch == fixture["arch"]


def test_every_architecture_has_a_config_fixture():
    types = {json.loads(p.read_text())["config"]["model_type"] for p in FIXTURES}
    assert types == {a.model_type for a in ARCHITECTURES}


def test_an_architecture_compute_does_not_load_is_refused_by_name():
    with pytest.raises(NotImplementedError, match="gemma4_unified"):
        read_arch_from_config({"model_type": "gemma4_unified"})


@pytest.mark.parametrize("arch", [a for a in ARCHITECTURES if a.loader == "mlx-vlm"],
                         ids=lambda a: a.model_type)
def test_the_stated_defaults_are_the_loader_s(arch):
    module = pytest.importorskip(f"mlx_vlm.models.{arch.model_type}")
    fields = {f.name: f.default for f in dataclasses.fields(module.TextConfig)}
    for key, value in arch.config_defaults.items():
        assert fields[key] == value, f"{arch.model_type}.{key}"
    shape_keys = {"num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
                  "sliding_window_pattern", "num_kv_shared_layers",
                  "hidden_size_per_layer_input", "hidden_size", "num_hidden_layers"}
    unstated = {k for k in shape_keys & set(fields)
                if fields[k] is not dataclasses.MISSING and k not in arch.config_defaults}
    assert not unstated, f"{arch.model_type} defaults not stated: {sorted(unstated)}"


@pytest.mark.parametrize("stem,absent", [("gemma-4-31b-it-bf16", {"gate_out"}),
                                         ("gemma-4-e2b-it-bf16", set()),
                                         ("gemma-4-e4b-it-bf16", set()),
                                         ("gemma-3-4b-it-bf16", {"gate_out"}),
                                         ("gemma-3-12b-it-bf16", {"gate_out"})])
def test_a_gemma_config_names_the_points_its_checkpoint_lacks(stem, absent):
    from mechbench_compute.architectures import BY_MODEL_TYPE
    from mechbench_compute.support import refusal

    fixture = json.loads((Path(__file__).parent / "fixtures" / "configs" / f"{stem}.json").read_text())
    arch = read_arch_from_config(fixture["config"], fixture["repo"])
    assert refusal(fixture["config"]) is None
    assert set(BY_MODEL_TYPE[arch.model_type].absent_points(arch)) == absent
