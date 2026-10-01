from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from mechbench_compute.api import Op, Output, P, Resume, ShardWriter, tensor_collection

DERIVATIONS = ("sae", "transcoder", "crosscoder")

ACTIVATIONS = ("jumprelu", "relu", "topk")

HOOKS: tuple[tuple[str, str], ...] = (
    (r"model\.layers\.(\d+)\.output", "resid_post"),
    (r"model\.layers\.(\d+)\.post_feedforward_layernorm\.output", "mlp_out"),
    (r"model\.layers\.(\d+)\.self_attn\.o_proj\.input", "attn.o_in"),
    (r"model\.layers\.(\d+)\.pre_feedforward_layernorm\.output", "mlp.in_norm"),
)

SOURCE_ACTIVATIONS = {"jump_relu": "jumprelu", "jumprelu": "jumprelu", "relu": "relu", "topk": "topk"}

CONFIG = "config.json"

PARAMS = "params.safetensors"

OP = Op(
    name="dictionary/load",
    needs=frozenset({"network:huggingface.co"}),
    resume=Resume("restart"),
    summary=(
        "Load a published sparse dictionary from the Hugging Face hub, such as a Gemma Scope sparse "
        "autoencoder, pinned to its commit and the hash of its weights."
    ),
    description="""\
Fetches one dictionary from a Hugging Face repository — the folder at
`path`, holding a `config.json` and a `params.safetensors` in Gemma
Scope 2's layout — and writes it as a `direction/dictionary`: one item
per feature with its encoder column and decoder row, and on the header
what it reads and writes, the checkpoint it was trained on, its width and
nonlinearity, and where it came from, with the commit the revision
resolved to and the sha256 of every file. The weights go to the tensor
store as shards, so a dictionary of any width is one object.

The point and layer are read from the source's hook names
(`model.layers.17.output` is `resid_post` at layer 17; the
post-feedforward norm's output is `mlp_out`; the attention output
projection's input is `attn.o_in`). `point`, `layer`, `derivation` and
`activation`, when given, are checks: a source that says otherwise is
refused rather than relabelled. `sha256` pins the weights file.

Sparse autoencoders load today. A transcoder or a crosscoder is the same
kind with another derivation, and is refused by name until this
operation reads them.
""",
    params=(
        P("repo", "string", "The Hugging Face repository, `\"google/gemma-scope-2-4b-it\"`."),
        P("path", "string",
          "The dictionary's folder in the repository, `\"resid_post/layer_17_width_16k_l0_medium\"`."),
        P("revision", "string",
          "A branch, tag or commit; the commit it resolves to is recorded either way. Pin a commit "
          "for a result that cannot move.",
          "main"),
        P("sha256", "string",
          "The sha256 of the weights file. A download that hashes otherwise is refused.",
          None),
        P("point", "string",
          "The point the dictionary must read, checked against the source.",
          None),
        P("layer", "int",
          "The layer the dictionary must read, checked against the source.",
          None),
        P("derivation", "string",
          "What the dictionary must be, checked against the source: a sparse autoencoder, a "
          "transcoder or a crosscoder.",
          None, choices=DERIVATIONS),
        P("activation", "string",
          "The nonlinearity the dictionary must use, checked against the source.",
          None, choices=ACTIVATIONS),
    ),
    inputs=(),
    output=Output("direction/dictionary", collection=True,
                  doc="One item per feature `{id, index, vector, encoder, norm, b_enc, threshold}`, stored as "
                      "shards; the header carries `derivation`, `reads`, `writes`, `model`, `source`, `width`, "
                      "`d_in`, `d_out`, `activation`, `b_dec` and `published`."),
    example={"repo": "google/gemma-scope-2-4b-it", "path": "resid_post/layer_17_width_16k_l0_medium",
             "derivation": "sae"},
)


def run(ctx, inputs, params):
    return load_dictionary(params, fetch=fetch_folder, read_model_type=read_architecture)


def fetch_folder(repo: str, path: str, revision: str) -> tuple[Path, str]:
    from huggingface_hub import snapshot_download

    local = snapshot_download(repo, revision=revision,
                              allow_patterns=[f"{path}/{CONFIG}", f"{path}/{PARAMS}"])
    # external: huggingface_hub — snapshot_download lands in a directory named for the resolved commit
    return Path(local) / path, Path(local).name


def read_architecture(model_id: str) -> str | None:
    from huggingface_hub import HfApi

    config = HfApi().model_info(model_id).config or {}
    model_type = config.get("model_type")
    return str(model_type) if model_type else None


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_hook(hook: str) -> tuple[str, int]:
    for pattern, point in HOOKS:
        m = re.fullmatch(pattern, hook)
        if m:
            return point, int(m.group(1))
    raise ValueError(f"dictionary/load: the hook {hook!r} is not one this operation maps to a point; "
                     f"it reads {', '.join(p for _, p in HOOKS)}")


def check_declared(name: str, want: Any, found: Any) -> None:
    if want is not None and want != found:
        raise ValueError(f"dictionary/load: {name} {want!r} was asked for, and the source is {found!r}")


def read_derivation(config: Mapping[str, Any]) -> str:
    kind = str(config.get("type") or "sae")
    if kind not in DERIVATIONS:
        raise ValueError(f"dictionary/load: the source's type {kind!r} is not one of {', '.join(DERIVATIONS)}")
    return kind


def read_activation(config: Mapping[str, Any]) -> str:
    found = str(config.get("architecture") or "")
    if found not in SOURCE_ACTIVATIONS:
        raise ValueError(f"dictionary/load: the source's architecture {found!r} is not a nonlinearity this "
                         f"operation knows ({', '.join(ACTIVATIONS)})")
    return SOURCE_ACTIVATIONS[found]


def read_params(path: Path) -> dict[str, np.ndarray]:
    from safetensors import safe_open

    with safe_open(str(path), framework="np") as f:
        return {k.lower(): f.get_tensor(k) for k in f.keys()}  # noqa: SIM118


def load_dictionary(params: Mapping[str, Any], *,
                    fetch: Callable[[str, str, str], tuple[Path, str]],
                    read_model_type: Callable[[str], str | None]) -> dict[str, Any]:
    repo, path = str(params["repo"]), str(params["path"]).strip("/")
    revision = str(params.get("revision") or "main")
    for name, choices in (("derivation", DERIVATIONS), ("activation", ACTIVATIONS)):
        given = params.get(name)
        if given is not None and given not in choices:
            raise ValueError(f"dictionary/load: {name} {given!r} is not one of {', '.join(choices)}")
    folder, commit = fetch(repo, path, revision)
    if not (folder / CONFIG).is_file() or not (folder / PARAMS).is_file():
        raise ValueError(f"dictionary/load: {repo}/{path} holds no {CONFIG} and {PARAMS}; this operation "
                         "reads Gemma Scope 2's layout, one folder per dictionary")
    config = json.loads((folder / CONFIG).read_text())
    derivation = read_derivation(config)
    check_declared("derivation", params.get("derivation"), derivation)
    if derivation != "sae":
        raise ValueError(
            f"dictionary/load: {repo}/{path} is a {derivation}. A {derivation} is a direction/dictionary "
            "with its own derivation, and this operation reads sparse autoencoders only so far: "
            + ("a transcoder reads `mlp.in_norm`, which needs its architecture's full hook points"
               if derivation == "transcoder" else
               "a crosscoder reads several layers at once, which encode does not read yet"))
    activation = read_activation(config)
    check_declared("activation", params.get("activation"), activation)
    point, layer = read_hook(str(config.get("hf_hook_point_in") or ""))
    out_point, out_layer = read_hook(str(config.get("hf_hook_point_out") or config.get("hf_hook_point_in")))
    if (out_point, out_layer) != (point, layer):
        raise ValueError(f"dictionary/load: {repo}/{path} reads {point} at layer {layer} and writes "
                         f"{out_point} at layer {out_layer}; a sparse autoencoder writes what it reads")
    check_declared("point", params.get("point"), point)
    check_declared("layer", params.get("layer"), layer)
    if config.get("affine_connection"):
        raise ValueError(f"dictionary/load: {repo}/{path} has an affine skip connection, which this "
                         "operation does not read")
    digest = hash_file(folder / PARAMS)
    check_declared("sha256", params.get("sha256"), digest)
    weights = read_params(folder / PARAMS)
    missing = sorted({"w_enc", "b_enc", "w_dec", "b_dec"} - set(weights)
                     | ({"threshold"} - set(weights) if activation == "jumprelu" else set()))
    if missing:
        raise ValueError(f"dictionary/load: {repo}/{path}/{PARAMS} lacks {', '.join(missing)}")
    w_enc, w_dec = weights["w_enc"].astype(np.float32), weights["w_dec"].astype(np.float32)
    d_in, width = w_enc.shape
    if w_dec.shape[0] != width or weights["b_enc"].shape != (width,):
        raise ValueError(f"dictionary/load: the encoder is {d_in}×{width} and the decoder "
                         f"{'×'.join(map(str, w_dec.shape))}; they disagree on the width")
    d_out = int(w_dec.shape[1])
    model_id = str(config.get("model_name") or "") or None
    space = {"model": model_id, "layer": layer, "point": point, "head": None, "d": int(d_in)}
    writer = ShardWriter(tempfile.mkdtemp(prefix="mechbench-tensor-"))
    encoders = np.ascontiguousarray(w_enc.T)
    norms = np.linalg.norm(w_dec.astype(np.float64), axis=1)
    b_enc = weights["b_enc"].astype(np.float64)
    threshold = weights["threshold"].astype(np.float64) if activation == "jumprelu" else None
    for i in range(width):
        item = {"id": str(i), "index": i, "vector": w_dec[i], "encoder": encoders[i],
                "norm": float(norms[i]), "b_enc": float(b_enc[i])}
        if threshold is not None:
            item["threshold"] = float(threshold[i])
        writer.add(item)
    published = {k: config[k] for k in ("l0",) if k in config}
    return tensor_collection(
        "direction/dictionary", writer.close(),
        derivation=derivation,
        reads=[space], writes=[{**space, "d": d_out}],
        model={"id": model_id, "architecture": read_model_type(model_id) if model_id else None},
        source={"hub": {"repo": repo, "path": path, "revision": revision, "commit": commit,
                        "files": {CONFIG: hash_file(folder / CONFIG), PARAMS: digest}}},
        width=int(width), d_in=int(d_in), d_out=d_out,
        activation={"fn": activation},
        b_dec=[float(x) for x in weights["b_dec"].astype(np.float64)],
        published=published,
    )
