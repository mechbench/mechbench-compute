from __future__ import annotations

import pytest

from mechbench_compute import hub
from mechbench_compute.ops.eval.benchmark import check_task_names
from mechbench_compute.protocol import ProtocolExecutor


class TestAProtocolNamesAModelByItsHubId:
    @pytest.mark.parametrize("ref", [
        "mlx-community/gemma-4-E4B-it-bf16", "gpt2", "org/name@abc1234", "org/name@main",
    ])
    def test_a_hub_id_is_accepted(self, ref):
        assert hub.check_hub_id(ref) == ref

    @pytest.mark.parametrize("ref", [
        "/Users/someone/model", "../model", "org/..", "./org/name", "org/name@../x",
        "org/name@refs/pr/1", ".hidden/x", "org/x/y", "org/name@", "C:\\model",
    ])
    def test_anything_else_is_refused(self, ref):
        with pytest.raises(ValueError, match="not a hub id"):
            hub.check_hub_id(ref)

    def test_the_executor_refuses_a_local_path_before_loading(self, tmp_path):
        with pytest.raises(ValueError, match="not a hub id"):
            ProtocolExecutor()._model_loaded(str(tmp_path))

    def test_a_hub_id_that_names_a_local_directory_still_comes_from_the_hub(
            self, tmp_path, monkeypatch):
        from mechbench_compute.model import Model

        monkeypatch.chdir(tmp_path)
        (tmp_path / "org" / "name").mkdir(parents=True)
        asked = []

        def ensure(model_id, **_):
            asked.append(model_id)
            raise ValueError("stop here")

        monkeypatch.setattr(hub, "ensure_model", ensure)
        with pytest.raises(ValueError, match="stop here"):
            Model.load("org/name", hub_only=True)
        assert asked == ["org/name"]


class TestADownloadTakesOnlyWhatALoaderReads:
    def test_snapshot_download_is_given_the_allowed_patterns(self, tmp_path, monkeypatch):
        import huggingface_hub

        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "cache"))
        got = {}

        def download(repo_id, **kw):
            got.update(kw)
            where = tmp_path / ("a" * 40)
            where.mkdir()
            return str(where)

        monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
        hub.ensure_model("org/name", offline=False)
        assert got["allow_patterns"] == list(hub.ALLOW_PATTERNS)
        assert not any(p.endswith((".py", ".bin", ".pt", ".pkl")) for p in got["allow_patterns"])


class TestABenchmarkTaskIsAName:
    def test_names_pass(self):
        assert check_task_names(["hellaswag", "arc_easy", "mmlu-pro"]) == [
            "hellaswag", "arc_easy", "mmlu-pro"]

    @pytest.mark.parametrize("task", ["../tasks/x.yaml", "/abs/task.yaml", "x.yaml", "", 3])
    def test_a_path_is_refused(self, task):
        with pytest.raises(ValueError, match="never a path"):
            check_task_names([task])
