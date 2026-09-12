"""The sandbox session and its tools (task 000360).

Snapshot-only tools (read_file / write_file / list) need no guest and
run everywhere. The shell-backed tools (bash / find / grep) run the
real mbshell guest and skip when it is not built on this machine.
"""
from __future__ import annotations

import os
import pathlib

import pytest

from mechbench_compute import guests
from mechbench_compute import sandbox
from mechbench_compute import snapshots as fs
from mechbench_compute import tools as T
from mechbench_compute.providers import messages as pm
from mechbench_compute.sandbox_session import (
    SandboxImage, SandboxRefused, SandboxSession)

REPO = pathlib.Path(__file__).resolve().parent.parent
BUILT = pathlib.Path(os.environ.get(
    "MECHBENCH_MBSHELL_WASM", REPO / "guests" / "mbshell" / "build" / "mbshell.wasm"))
needs_guest = pytest.mark.skipif(
    not BUILT.is_file(), reason="mbshell.wasm not built — guests/mbshell/build.sh")


@pytest.fixture(scope="module")
def guest_installed(tmp_path_factory):
    if not BUILT.is_file():
        pytest.skip("mbshell.wasm not built on this machine")
    os.environ["MECHBENCH_GUEST_CACHE"] = str(tmp_path_factory.mktemp("guests"))
    guests.install_local("mbshell", BUILT)


SEED = {"a.txt": "one two three\n", "sub/b.txt": "four\nfive\n", "notes.md": "hi\n"}


def _image(**kw):
    kw.setdefault("snapshot", fs.seeded(SEED))
    return SandboxImage.parse(kw)


class TestTheSnapshotTools:
    """No guest: read/write/list are pure snapshot operations."""

    def test_read_a_seeded_file(self):
        s = SandboxSession(_image())
        assert s.read_file("a.txt") == "one two three\n"

    def test_read_a_missing_file_is_a_message_not_a_raise(self):
        s = SandboxSession(_image())
        out = s.read_file("nope.txt")
        assert "no such file" in out and s.calls[-1].exit_code == 1

    def test_write_advances_the_snapshot(self):
        s = SandboxSession(_image())
        before = s.snapshot.digest()
        s.write_file("report.txt", "counted\n")
        assert s.snapshot.digest() != before
        assert s.read_file("report.txt") == "counted\n"
        assert s.calls[-2].tool == "write_file"
        assert s.calls[-2].changed["added"] == ["report.txt"]

    def test_write_a_large_file_rides_the_blob_sidecar(self):
        s = SandboxSession(_image())
        big = "x" * (fs.INLINE_MAX + 10)
        s.write_file("big.txt", big)
        assert s.read_file("big.txt") == big     # from the sidecar

    def test_list_is_scoped_by_prefix(self):
        s = SandboxSession(_image())
        assert s.list() == "a.txt\nnotes.md\nsub/b.txt"
        assert s.list("sub") == "sub/b.txt"

    def test_the_session_never_mutates_the_image(self):
        img = _image()
        before = img.snapshot.digest()
        SandboxSession(img).write_file("x.txt", "y")
        assert img.snapshot.digest() == before

    def test_a_read_records_provenance_without_changing_the_tree(self):
        s = SandboxSession(_image())
        d = s.snapshot.digest()
        s.read_file("a.txt")
        c = s.calls[-1]
        assert c.snapshot_in == c.snapshot_out == d and c.tool == "read_file"


class TestTheImage:
    def test_an_unknown_tool_is_refused(self):
        with pytest.raises(SandboxRefused, match="unknown tool"):
            SandboxImage(tools=("bash", "telepathy"))

    def test_the_default_image_offers_the_workhorse_and_conveniences(self):
        assert set(SandboxImage().tools) == {"bash", "read_file", "write_file", "list"}

    def test_inline_snapshot_map(self):
        img = SandboxImage.parse({"snapshot": {"x.txt": "hi\n"}})
        assert SandboxSession(img).read_file("x.txt") == "hi\n"

    def test_limits_come_through(self):
        img = SandboxImage.parse({"limits": {"memory_mb": 32, "wall_seconds": 5}})
        assert img.limits.memory_mb == 32 and img.limits.wall_seconds == 5

    def test_tool_defs_carry_a_sandbox_handler(self):
        defs = SandboxSession(_image()).tool_defs()
        assert {d["name"] for d in defs} == set(SandboxImage().tools)
        assert all(d["handler"].get("sandbox") for d in defs)


class TestThroughTheToolbox:
    """The path a model takes: a ToolCallPart into a Toolbox bound to
    the session."""

    def _box(self):
        session = SandboxSession(_image())
        box = T.toolbox_from(session.tool_defs(), session=session)
        return box, session

    def test_a_write_then_list_through_tool_calls(self):
        box, session = self._box()
        box.call(pm.ToolCallPart(id="c1", name="write_file",
                                 arguments={"path": "r.txt", "content": "ok"}))
        out = box.call(pm.ToolCallPart(id="c2", name="list", arguments={}))
        assert "r.txt" in out.content
        assert [c.tool for c in session.calls] == ["write_file", "list"]

    def test_a_sandbox_tool_without_a_session_says_so(self):
        box = T.Toolbox(SandboxSession(_image()).tool_defs())   # no session
        out = box.call(pm.ToolCallPart(id="c1", name="list", arguments={}))
        assert out.is_error and "without a session" in out.content

    def test_the_provenance_record_is_serializable(self):
        import json
        _, session = self._box()
        session.write_file("r.txt", "x")
        json.dumps([c.to_wire() for c in session.calls])   # must not raise


@needs_guest
class TestTheShellTools:
    def test_bash_runs_a_pipeline(self, guest_installed):
        s = SandboxSession(_image())
        out = s.bash('find . -name "*.txt" | sort | wc -l')
        assert out.strip() == "2"
        assert s.calls[-1].tool == "bash" and s.calls[-1].exit_code == 0

    def test_grep_finds_matches(self, guest_installed):
        s = SandboxSession(_image())
        out = s.grep("four")
        assert "sub/b.txt" in out

    def test_a_failing_command_shows_its_exit_in_band(self, guest_installed):
        s = SandboxSession(_image())
        out = s.bash("nosuchcommand")
        assert "[exit 127]" in out

    def test_the_acceptance_scenario(self, guest_installed):
        # "Write a script that counts the words in every file under the
        # tree and writes a report" — the task's own acceptance test,
        # driven as a model would drive it: write the script, run it,
        # read the report back.
        s = SandboxSession(_image())
        s.write_file("count.sh",
                     'for f in $(find . -type f -name "*.txt" | sort); do\n'
                     '  echo "$f $(wc -w < "$f")"\n'
                     'done > report.out\n')
        run = s.bash("sh count.sh")
        assert s.calls[-1].exit_code == 0, run
        report = s.read_file("report.out")
        assert report == "a.txt 3\nsub/b.txt 2\n"
        # The report is now part of the workspace: a real object.
        assert s.snapshot.get("report.out") is not None

    def test_a_run_is_a_function_of_the_snapshot_it_saw(self, guest_installed):
        # Replay: the same starting image and the same argv give the
        # same snapshot chain, digest for digest.
        def drive():
            s = SandboxSession(_image())
            s.bash("echo hello > out.txt")
            s.bash("wc -c out.txt")
            return [(c.snapshot_in, c.snapshot_out, c.stdout) for c in s.calls]
        assert drive() == drive()

    def test_a_limit_is_named_in_band(self, guest_installed):
        s = SandboxSession(_image(limits={"wall_seconds": 1}))
        out = s.bash("sleep 30; echo woke")   # waits are virtual — no hang
        assert "woke" in out                   # completed at once


class TestThroughTheChatNode:
    """The sandbox reaches a model as a node param; the loop records the
    snapshot chain onto the item. `list` needs no guest, so this runs
    everywhere."""

    def _params(self, **kw):
        base = {
            "model": {"provider": "mock", "model": "mock-large"},
            "budget_usd": 1.0,
            "records": [{"id": "r0", "user": "what is in the workspace?"}],
            "sandbox": {"tools": ["list"], "snapshot": {"a.txt": "hi\n"}},
            "max_tool_rounds": 1,
            # The mock calls `list` when asked.
            "provider_options": {"mock": {"tool_call": "list"}},
        }
        base.update(kw)
        return base

    def test_a_chat_node_offers_the_sandbox_and_records_the_chain(self):
        from mechbench_compute import chat as chat_mod
        from mechbench_compute import model_ref as mr
        params = self._params()
        out = chat_mod.run_remote(mr.parse(params["model"]), params["records"], params)
        meta = out["items"][0]["metadata"]
        assert "sandbox" in meta, "the snapshot chain was not recorded"
        call = meta["sandbox"][0]
        assert call["tool"] == "list" and call["snapshot_in"] == call["snapshot_out"]
        # Two model calls: the one that asked for the tool, the one after.
        assert out["spend"]["calls"] == 2

    def test_no_sandbox_means_no_sandbox_key(self):
        from mechbench_compute import chat as chat_mod
        from mechbench_compute import model_ref as mr
        params = self._params(sandbox=None, provider_options={})
        out = chat_mod.run_remote(mr.parse(params["model"]), params["records"], params)
        assert "sandbox" not in out["items"][0]["metadata"]
