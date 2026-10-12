from mechbench_compute.hub import parse_model_ref


def test_bare_repo_has_no_revision():
    assert parse_model_ref("mlx-community/gemma-4-e2b-it-bf16") == (
        "mlx-community/gemma-4-e2b-it-bf16", None)


def test_pinned_repo_splits_at_the_at_sign():
    assert parse_model_ref("org/model@22a2753") == ("org/model", "22a2753")


def test_trailing_at_means_no_revision():
    assert parse_model_ref("org/model@") == ("org/model", None)


def snapshot(tmp_path, files, index=None):
    import json

    root = tmp_path / "snap"
    root.mkdir()
    for name in files:
        (root / name).write_text("x")
    if index is not None:
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    return root


def test_a_snapshot_with_every_indexed_shard_is_complete(tmp_path):
    from mechbench_compute.hub import is_complete

    root = snapshot(tmp_path, ["config.json", "a.safetensors", "b.safetensors"],
                    {"w1": "a.safetensors", "w2": "b.safetensors"})
    assert is_complete(root)


def test_a_snapshot_missing_a_shard_is_not_complete(tmp_path):
    from mechbench_compute.hub import is_complete

    root = snapshot(tmp_path, ["config.json", "b.safetensors"],
                    {"w1": "a.safetensors", "w2": "b.safetensors"})
    assert not is_complete(root)


def test_a_dangling_link_is_not_a_shard(tmp_path):
    from mechbench_compute.hub import is_complete

    root = snapshot(tmp_path, ["config.json"], {"w1": "a.safetensors"})
    (root / "a.safetensors").symlink_to(tmp_path / "blob-never-written")
    assert not is_complete(root)


def test_a_snapshot_without_its_config_is_not_complete(tmp_path):
    from mechbench_compute.hub import is_complete

    assert not is_complete(snapshot(tmp_path, ["model.safetensors"]))


def test_a_single_file_snapshot_with_its_config_is_complete(tmp_path):
    from mechbench_compute.hub import is_complete

    assert is_complete(snapshot(tmp_path, ["config.json", "model.safetensors"]))


def test_an_incomplete_cached_snapshot_is_downloaded_again(tmp_path, monkeypatch):
    from mechbench_compute import hub

    repo = tmp_path / "models--org--m"
    snap = repo / "snapshots" / ("a" * 40)
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors.index.json").write_text('{"weight_map": {"w": "a.safetensors"}}')
    monkeypatch.setattr(hub, "_repo_dir", lambda repo_id: repo)
    monkeypatch.setattr(hub, "resolve_cached_revision", lambda repo_id, revision: "a" * 40)
    monkeypatch.setattr(hub, "is_offline", lambda: False)
    fetched = []

    def fake_download(repo_id, revision=None, allow_patterns=None, **kwargs):
        fetched.append(repo_id)
        (snap / "a.safetensors").write_text("x")
        return str(snap)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    assert hub.ensure_model("org/m") == ("org/m", "a" * 40, snap)
    assert fetched == ["org/m"]
    assert hub.ensure_model("org/m") == ("org/m", "a" * 40, snap)
    assert fetched == ["org/m"]
