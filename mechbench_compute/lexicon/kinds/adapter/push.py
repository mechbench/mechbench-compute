from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "adapter/push",
    "The record of an adapter published to the Hugging Face hub: the repository, the files, and the commit that can fetch it back.",
    fields={"repo": F("string", "`<namespace>/<name>`."), "private": F("boolean", "Whether the repository is private."),
            "dry_run": F("boolean", "Whether nothing was uploaded."), "files": F("array", "`{name, bytes}` per file.", items={"type": "object"}),
            "lora": F("object", "`{rank, alpha, target_modules}`."), "base_model": F("string", "The base named in the model card."),
            "commit": F("string", "The commit created."), "url": F("string", "The repository at that commit."),
            "hf_adapter_ref": F("object", "`{repo, revision}` to fetch it back.")},
    required=("repo", "files"),
)
