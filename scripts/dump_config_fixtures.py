#!/usr/bin/env python3

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mechbench_compute._arch import Arch
from mechbench_compute.architectures import BY_MODEL_TYPE

HUB = Path.home() / ".cache" / "huggingface" / "hub"
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "configs"

REPOS = (
    "mlx-community/gemma-4-e2b-it-bf16",
    "mlx-community/gemma-4-E4B-it-bf16",
    "mlx-community/gemma-3-4b-it-bf16",
    "mlx-community/gemma-3-12b-it-bf16",
    "mlx-community/Qwen2.5-3B-Instruct-bf16",
    "mlx-community/Qwen2.5-7B-bf16",
    "mlx-community/Llama-3.2-3B-Instruct-bf16",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16",
)


def read_cached_config(repo: str) -> dict[str, Any]:
    snapshots = HUB / f"models--{repo.replace('/', '--')}" / "snapshots"
    return json.loads(next(snapshots.glob("*/config.json")).read_text())


def read_arch_by_loader(repo: str, config: dict[str, Any]) -> Arch:
    import importlib

    top = config["model_type"].lower()
    if BY_MODEL_TYPE[top].loader == "mlx-lm":
        args = importlib.import_module(f"mlx_lm.models.{top}").ModelArgs.from_dict(config)
        return Arch.from_mlx_model(SimpleNamespace(args=args), repo)
    text = importlib.import_module(f"mlx_vlm.models.{top}").TextConfig.from_dict(config["text_config"])
    return Arch.from_mlx_model(SimpleNamespace(config=SimpleNamespace(text_config=text)), repo)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for repo in REPOS:
        config = read_cached_config(repo)
        arch = dataclasses.asdict(read_arch_by_loader(repo, config))
        arch["global_layers"] = list(arch["global_layers"])
        name = repo.split("/")[1].lower()
        (OUT / f"{name}.json").write_text(
            json.dumps({"repo": repo, "config": config, "arch": arch}, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(REPOS)} config fixtures to {OUT}")


if __name__ == "__main__":
    main()
