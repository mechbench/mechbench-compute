from __future__ import annotations

from mechbench_compute.lexicon._base import In, Op, Output, P
from mechbench_compute.protocol.read_tokenizer_id import read_tokenizer_id

OP = Op(
    name="adapter/publish",
    requires="remote",
    summary=(
        "Publish an adapter object to the Hugging Face hub as a PEFT LoRA "
        "repository, with a model card carrying its bench provenance."
    ),
    description="""\
The adapter's weights and config are written in PEFT's layout with a README
naming the base model, rank, alpha, target modules and training steps, and
uploaded as one commit. The result names that commit, so
`{"$ref": {"hf_adapter": {"repo": …, "revision": …}}}` fetches it straight back into
a protocol — the round trip.

Needs a Hugging Face write token in the job owner's vault. `dry_run` stages
the repository locally and reports the files and sizes without touching the
hub or needing a token.
""",
    inputs=(
        In("adapter", "adapter/lora",
           "The adapter to publish, usually from `adapter/train`."),
    ),
    output=(
        Output('adapter/push', collection=False, doc='`repo`, `private`, `files` (name and size), `lora`, `base_model`, `commit`, `url`, and `hf_adapter_ref` to fetch it back.')
    ),
    params=(
        P("repo", "string", "The destination, `\"<namespace>/<name>\"`."),
        P("private", "bool", "Create the repository private.", True),
        P("commit_message", "string", "The commit message.", "mechbench: adapter push"),
    ),
    example={"repo": "benjismith/spinner-fair-v1", "private": True},
    example_inputs={"adapter": {"$ref": {"bench": "you/lab/adapter"}}},
)


def run(ctx, inputs, params):
    import os
    import tempfile

    from mechbench_compute.peft import peft_export

    payload = inputs.get("adapter")
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError("hf/push-adapter needs an adapter object "
                         "(input port `adapter`, by edge or "
                         "`{\"$ref\": …}`)")
    repo = str(params.get("repo") or "").strip()
    if "/" not in repo:
        raise ValueError("hf/push-adapter needs params.repo as "
                         "'<namespace>/<name>'")
    private = bool(params.get("private", True))
    dry_run = bool(params.get("dry_run", False))
    lora = payload.get("lora") or {}
    base_model = payload.get("base_model") or read_tokenizer_id(params.get("model"))

    with tempfile.TemporaryDirectory() as d:
        out = peft_export(payload, d)
        card = [
            "---",
            "library_name: peft",
            f"base_model: {base_model}" if base_model else "base_model: unknown",
            "tags: [lora, mechbench]",
            "---",
            "",
            f"# {repo.split('/')[-1]}",
            "",
            "LoRA adapter exported from the mechbench bench.",
            "",
            f"- base model: `{base_model}`",
            f"- rank: {lora.get('rank')}  alpha: {lora.get('alpha')}",
            f"- target modules: {', '.join(lora.get('target_modules') or [])}",
            f"- trained for {payload.get('steps', '?')} steps"
            if payload.get("steps") else "",
            "",
            ("Load with PEFT (`PeftModel.from_pretrained`) or fetch "
             "back into a mechbench protocol via `{\"$ref\": {\"hf_adapter\": …}}`."),
        ]
        with open(os.path.join(out, "README.md"), "w") as f:
            f.write("\n".join(line for line in card if line is not None))
        files = sorted(
            ({"name": name,
              "bytes": os.path.getsize(os.path.join(out, name))}
             for name in os.listdir(out)),
            key=lambda f: f["name"],
        )
        record = {"kind": "adapter/push", "repo": repo, "private": private,
                  "dry_run": dry_run, "files": files,
                  "lora": {k: lora.get(k) for k in ("rank", "alpha",
                                                    "target_modules")},
                  "base_model": base_model}
        if dry_run:
            return {**record, "commit": None, "url": None}

        token = (ctx.secrets or {}).get("hf", {}).get("token")
        if not token:
            raise ValueError(
                "hf/push-adapter needs an HF WRITE token in the job "
                "owner's vault (Settings -> Integrations); none was "
                "delivered at claim")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo, private=private, exist_ok=True,
                        repo_type="model")
        info = api.upload_folder(
            repo_id=repo, folder_path=out, repo_type="model",
            commit_message=str(params.get(
                "commit_message", "mechbench: adapter push")))
        commit = getattr(info, "oid", None)
        return {**record, "commit": commit,
                "url": f"https://huggingface.co/{repo}"
                       + (f"/tree/{commit}" if commit else ""),
                "hf_adapter_ref": {"repo": repo, "revision": commit}}
