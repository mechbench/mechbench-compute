from __future__ import annotations

import json
import re
from pathlib import Path

import mlx.core as mx

_PEFT_KEY_RE = re.compile(
    r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_([AB])\.weight$")

CONFIG_NAME = "adapter_config.json"
WEIGHTS_NAME = "adapter_model.safetensors"


def peft_export(adapter: dict, out_dir: str) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tmp = out / "_ours.safetensors"
    tmp.write_bytes(adapter["data"])
    ours = dict(mx.load(str(tmp)))
    tmp.unlink()

    peft_weights = {}
    targets = set()
    for key, w in ours.items():
        m = re.match(
            r"^model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_([ab])$",
            key)
        if m is None:
            raise ValueError(f"unrecognized adapter key {key!r}")
        i, container, proj, ab = m.groups()
        targets.add(proj)
        peft_weights[
            f"base_model.model.model.layers.{i}.{container}.{proj}"
            f".lora_{ab.upper()}.weight"] = w
    mx.save_safetensors(str(out / WEIGHTS_NAME), peft_weights)

    lora = adapter.get("lora") or {}
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": int(lora.get("rank", 8)),
        "lora_alpha": float(lora.get("alpha", 16)),
        "lora_dropout": 0.0,
        "target_modules": sorted(lora.get("target_modules") or targets),
        "base_model_name_or_path": adapter.get("base_model"),
        "bias": "none",
    }
    (out / CONFIG_NAME).write_text(json.dumps(config, indent=2))
    return str(out)


def peft_import(repo_dir: str) -> dict:
    d = Path(repo_dir)
    config = json.loads((d / CONFIG_NAME).read_text())
    if config.get("peft_type", "LORA").upper() != "LORA":
        raise ValueError(
            f"only LORA adapters supported; got {config.get('peft_type')!r}")
    peft_weights = dict(mx.load(str(d / WEIGHTS_NAME)))
    ours = {}
    extra = []
    for key, w in peft_weights.items():
        m = _PEFT_KEY_RE.search(key)
        if m is None:
            extra.append(key)
            continue
        i, container, proj, AB = m.groups()
        ours[f"model.layers.{i}.{container}.{proj}.lora_{AB.lower()}"] = w
    if extra:
        raise ValueError(
            f"adapter carries {len(extra)} non-LoRA weight(s) "
            f"(e.g. {extra[0]!r}) — modules_to_save/full-module "
            f"adapters are not supported by the LoRA fuse path")
    if not ours:
        raise ValueError("no LoRA weights found in adapter")
    import os
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    try:
        mx.save_safetensors(tmp, ours)
        data = Path(tmp).read_bytes()
    finally:
        os.unlink(tmp)

    r = int(config.get("r", 8))
    alpha = float(config.get("lora_alpha", 16))
    return {
        "kind": "adapter/lora",
        "format": "safetensors",
        "base_model": config.get("base_model_name_or_path"),
        "lora": {"rank": r, "alpha": alpha, "scale": alpha / r,
                  "target_modules": config.get("target_modules") or []},
        "train": {"source": "peft_import"},
        "data": data,
    }
