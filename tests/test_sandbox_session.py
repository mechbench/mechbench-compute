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
CPY_WASM = pathlib.Path(os.environ.get(
    "MECHBENCH_CPYTHON_WASM", REPO / "guests" / "cpython" / "build" / "python.wasm"))
CPY_STDLIB = pathlib.Path(os.environ.get(
    "MECHBENCH_CPYTHON_STDLIB", REPO / "guests" / "cpython" / "build" / "stdlib"))
needs_python = pytest.mark.skipif(
    not (CPY_WASM.is_file() and CPY_STDLIB.is_dir()),
    reason="cpython guest not built — guests/cpython/build.sh")
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


class TestObjectMounts:
    """Read-only trees mounted at absolute paths — the user-extensible
    stdlib, and any data dir. Guest-agnostic parts use mbshell."""

    def test_a_mount_is_parsed_as_a_resolved_tree(self):
        img = SandboxImage.parse({"mounts": [
            {"path": "/opt/data", "snapshot": {"x.txt": "hi\n"}}]})
        assert len(img.object_mounts) == 1
        at, tree = img.object_mounts[0]
        assert at == "/opt/data" and tree.get("x.txt") is not None

    def test_a_bench_ref_mount_stays_unresolved_for_the_executor(self):
        img = SandboxImage.parse({"mounts": [{"path": "/opt/corpus", "object": "benji/c"}]})
        assert img.object_mounts == () and img.mounts[0].object == "benji/c"

    def test_a_mount_without_a_tree_or_ref_is_refused(self):
        with pytest.raises(SandboxRefused, match="needs a .snapshot.*or an .object"):
            SandboxImage.parse({"mounts": [{"path": "/opt/x"}]})

    @needs_guest
    def test_a_mount_is_materialized_once_and_reused(self, guest_installed):
        # A read-only mount is content-addressed: the same tree is
        # materialized to the cache once and every later run — this
        # session or another — preopens the same directory.
        import glob
        cache = os.environ["MECHBENCH_GUEST_CACHE"]
        before = set(glob.glob(cache + "/mount-*"))
        img = SandboxImage.parse({
            "mounts": [{"path": "/opt/data", "snapshot": {"n.txt": "shared\n"}}]})
        s = SandboxSession(img)
        for _ in range(3):
            assert s.bash("cat /opt/data/n.txt").strip() == "shared"
        new = set(glob.glob(cache + "/mount-*")) - before
        assert len(new) == 1, "the mount was materialized more than once"
        # a second session with the same tree adds no new dir
        SandboxSession(img).bash("cat /opt/data/n.txt")
        assert set(glob.glob(cache + "/mount-*")) - before == new

    @needs_guest
    def test_a_mounted_file_is_readable_and_not_captured(self, guest_installed):
        img = SandboxImage.parse({
            "snapshot": {"work.txt": "start\n"},
            "mounts": [{"path": "/opt/data", "snapshot": {"note.txt": "mounted\n"}}]})
        s = SandboxSession(img)
        assert s.bash("cat /opt/data/note.txt").strip() == "mounted"
        s.bash("echo x > new.txt")
        # the mount is a separate preopen: only the working tree is captured
        assert not any("/opt/data" in p or "note.txt" in p for p in s.snapshot.paths())
        assert s.snapshot.get("new.txt") is not None


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


@pytest.fixture(scope="module")
def both_guests(tmp_path_factory):
    os.environ["MECHBENCH_GUEST_CACHE"] = str(tmp_path_factory.mktemp("guests"))
    guests.install_local("cpython", CPY_WASM, replace=True,
                         mounts=[(str(CPY_STDLIB), "/usr/local/lib/python3.13")])
    if BUILT.is_file():
        guests.install_local("mbshell", BUILT)


@needs_python
class TestThePythonGuest:
    """CPython over the same snapshot as the shell (task 000453). The
    guest is installed with its standard library as a read-only mount;
    the session picks it for `python` and mbshell for `bash`, sharing
    one workspace."""

    def _sess(self):
        return SandboxSession(SandboxImage.parse({
            "tools": ["bash", "python", "read_file", "write_file", "list"],
            "snapshot": {"data/a.txt": "one two three\n", "data/b.txt": "four\n"},
            "limits": {"memory_mb": 512, "wall_seconds": 30}}))

    def test_a_snippet_runs(self, both_guests):
        assert self._sess().python(code="print(sum(range(10)))") == "45\n"

    def test_the_common_stdlib_is_there(self, both_guests):
        out = self._sess().python(
            code="import json,re,math,collections,hashlib,csv,statistics;"
                 "print(hashlib.sha256(b'x').hexdigest()[:8])")
        assert out.strip() == "2d711642"

    def test_python_reads_and_writes_the_snapshot(self, both_guests):
        s = self._sess()
        s.python(script="")  # no-op guard
        s.write_file("count.py",
                     "import glob\nt=0\n"
                     "for p in sorted(glob.glob('data/*.txt')):\n"
                     "  t += len(open(p).read().split())\n"
                     "open('total.txt','w').write(str(t))\n")
        s.python(script="count.py")
        assert s.read_file("total.txt") == "4"
        assert s.snapshot.get("total.txt") is not None

    def test_python_and_bash_share_the_workspace(self, both_guests):
        if not BUILT.is_file():
            pytest.skip("mbshell not built")
        s = self._sess()
        s.python(code="open('x.json','w').write('{\"n\": 7}')")
        assert s.bash("cat x.json").strip() == '{"n": 7}'
        assert [c.tool for c in s.calls] == ["python", "bash"]

    def test_a_user_package_extends_the_stdlib(self, both_guests):
        # A pure-Python package mounted at site-packages imports — the
        # user-extensible stdlib (the bench-object mount).
        img = SandboxImage.parse({
            "tools": ["python"],
            "mounts": [{"path": "/usr/local/lib/python3.13/site-packages",
                        "snapshot": {"mymath/__init__.py": "def triple(n): return n*3\n"}}]})
        out = SandboxSession(img).python(code="import mymath; print(mymath.triple(14))")
        assert out.strip() == "42"

    def test_no_network(self, both_guests):
        r = self._sess().python(
            code="import socket\ns=socket.socket()\n"
                 "s.connect(('127.0.0.1', 9))")
        assert "[exit 1]" in r and "rror" in r

    def test_subprocess_is_refused(self, both_guests):
        r = self._sess().python(code="import subprocess; subprocess.run(['ls'])")
        assert "wasi does not support processes" in r or "[exit 1]" in r

    def test_the_guest_cannot_write_its_own_stdlib(self, both_guests):
        # The stdlib mount is read-only: a guest scribbling on it would
        # poison the shared cache for every other run.
        r = self._sess().python(
            code="open('/usr/local/lib/python3.13/os.py','a').write('x')")
        assert "[exit 1]" in r
