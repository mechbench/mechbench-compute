from __future__ import annotations

import re

from mechbench_compute import lexicon
from mechbench_compute.lexicon._base import Op, Output, P
from mechbench_compute.protocol.serialize_params import serialize_params

OP = Op(
    name="adapter/merge",
    summary=(
        "Collapse a model's adapter stack into one standalone checkpoint and "
        "publish it — to the bench or to the Hugging Face hub — so \"base "
        "plus these three adapters\" becomes a single thing anyone can load."
    ),
    description="""\
The `model` must be a structured reference carrying at least one adapter.
The merge never loads the model into memory: it rewrites the base's weight
shards one at a time with the adapters' deltas applied, so peak memory is
one shard. A manifest records every file's hash and the full stack that was
merged.

Destinations mirror the base grammar: `{"bench": {"name": "spinner-fair-v1"}}`
stores the checkpoint under the run's project, usable afterwards as
`{"base": {"bench": …}}`; `{"hf": {"repo": "user/repo", "private": true}}`
commits it to the hub, usable as `{"base": {"hf": "repo@sha"}}`. A failed
upload to the bench resumes: files already there with matching hashes are
skipped.

Nothing arrives by edge: the stack to merge is the `model` reference's.
""",
    inputs=(),
    output=Output('adapter/checkpoint', collection=False, doc='Where the checkpoint landed, its files with their hashes, and the stack that was merged.'),
    params=(
        P("to", "object",
          "Where to publish: `{\"bench\": {\"name\": …}}` (lowercase, "
          "digits, `-`, `_`) or `{\"hf\": {\"repo\": …, \"private\": …}}`. "
          "Exactly one.", fields=(
              P("bench", "object", "Publish to this platform as a stored checkpoint.", None,
                fields=(P("name", "string",
                          "The checkpoint's name: lowercase letters, digits, `-` and `_`, "
                          "starting with a letter or digit, at most 61 characters."),)),
              P("hf", "object", "Publish to a Hugging Face repository.", None,
                fields=(P("repo", "string", "The destination, `\"<namespace>/<name>\"`."),
                        P("private", "bool", "Create the repository private.", True))),
          )),
    ),
    example={"model": {"$param": "model"}, "to": {"bench": {"name": "spinner-fair-v1"}}},
)


def run(ctx, inputs, params):
    """Collapse a model's adapter stack into one standalone
    checkpoint, published to the bench or to Hugging Face.

    What turns "base plus these three adapters" into a single thing
    someone else can load.

    The merge never loads the model: it is a delta-shard rewrite
    (see checkpoint.py), so peak memory is one shard. Destination
    is explicit and mirrors the base grammar:

        {"to": {"bench": {"name": "spinner-fair-v1"}}}
            -> objects under <owner>/<project>/checkpoints/<name>/
               plus a manifest; usable as {"base": {"bench": ...}}.
        {"to": {"hf": {"repo": "user/repo", "private": true}}}
            -> a Hub commit; usable as {"base": {"hf": "repo@sha"}}.
    """
    import shutil
    import tempfile
    from pathlib import Path

    from mechbench_compute import bench, checkpoint
    from mechbench_compute.hub import ensure_model

    mref = params.get("model")
    if not hasattr(mref, "adapter_payloads"):
        raise ValueError(
            "merge needs a structured $model with adapters — merging "
            "a bare base would republish it unchanged")
    if not mref.adapter_payloads:
        raise ValueError("merge needs at least one adapter in the stack")
    to = params.get("to")
    if not isinstance(to, dict) or len(to) != 1 or not {"bench", "hf"} & set(to):
        raise ValueError(
            'merge needs to: {"bench": {"name": ...}} or '
            '{"hf": {"repo": ..., "private": ...}}')

    if mref.base_kind == "hf":
        repo_id, sha, snapshot = ensure_model(mref.base)
        base_snapshot = f"{repo_id}@{sha}"
    else:
        snapshot = ctx.executor._materialize_checkpoint(mref.base)
        base_snapshot = f"bench:{mref.base}"

    workdir = tempfile.mkdtemp(prefix="mechbench-merge-")
    try:
        out = Path(workdir) / "checkpoint"
        files = checkpoint.export_merged(
            snapshot, list(mref.adapter_payloads), out)
        manifest = checkpoint.build_manifest(
            out, files, mref.to_wire(), base_snapshot)
        # One unit per file: an hour of shard upload must read as
        # motion on the jobs page, not as a wedged run.
        if ctx.on_start:
            ctx.on_start(len(files))

        if "bench" in to:
            name = str((to["bench"] or {}).get("name") or "").strip()
            if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,60}", name):
                raise ValueError(
                    "bench destination needs a name: lowercase, "
                    "digits, - and _")
            if not ctx.result_base:
                raise ValueError("no result path — cannot derive the project")
            owner, project = ctx.result_base.split("/")[:2]
            prefix = f"{owner}/{project}/checkpoints/{name}"
            # Retry-as-resume: a failed upload leaves its intact
            # files on the server (torn ones were deleted by the
            # hash check), so ask what is already there and skip
            # byte-identical matches. On a residential uplink this
            # is the difference between resuming a 9.6GB upload and
            # restarting it.
            have = bench.list_prefix_hashes(prefix)
            by_name = {f["name"]: f["sha256"] for f in manifest["files"]}
            total = 0
            skipped = 0
            for fname in files:
                if have.get(fname) == by_name.get(fname):
                    skipped += 1
                    if ctx.on_item:
                        ctx.on_item()
                    continue
                receipt = bench.put_file(f"{prefix}/{fname}", out / fname)
                total += int(receipt.get("sizeBytes") or 0)
                if ctx.on_item:
                    ctx.on_item()
            bench.emit(
                f"{prefix}/{checkpoint.MANIFEST_NAME}",
                manifest,
                inputs=list(mref.adapter_labels),
                operation=lexicon.canonical_path("adapter/merge"),
                params=serialize_params(params),
            )
            return {
                "kind": "model/pointer",
                "base": {"bench": prefix},
                "manifest": f"{prefix}/{checkpoint.MANIFEST_NAME}",
                "files": len(files),
                "bytes": total,
                "skipped_already_stored": skipped,
            }

        hf_cfg = to["hf"] or {}
        repo = str(hf_cfg.get("repo") or "").strip()
        if "/" not in repo:
            raise ValueError('hf destination needs repo: "user/name"')
        token = (ctx.secrets or {}).get("hf", {}).get("token")
        if not token:
            raise ValueError(
                "hf destination needs an hf token — connect one under "
                "Settings -> Integrations")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(
            repo, private=bool(hf_cfg.get("private", True)), exist_ok=True)
        info = api.upload_folder(
            repo_id=repo, folder_path=str(out),
            commit_message=f"mechbench merge: {mref.describe()}")
        sha_out = getattr(info, "oid", None) or ""
        return {
            "kind": "model/pointer",
            "base": {"hf": f"{repo}@{sha_out}" if sha_out else repo},
            "files": len(files),
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
