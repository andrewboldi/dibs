"""dibs submit --snapshot: a queued job runs its code as it was at submit, so its agent can keep
editing while it waits (#172)."""

import os
import subprocess

import pytest
from conftest import run_until
from test_daemon import PY, finished, job, req

from dibs import snapshot
from dibs.model import State
from dibs.units import GiB


def _repo(tmp_path):
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    git = lambda *a: subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    (root / ".gitignore").write_text(".venv/\nresults/\n")
    (root / "src" / "model.py").write_text("SIZE = 1\n")
    (root / "src" / "old.py").write_text("x\n")
    (root / "train.py").write_text("import sys; sys.path.insert(0, 'src'); import model; print('SIZE', model.SIZE)\n")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    (root / "src" / "model.py").write_text("SIZE = 2\n")  # uncommitted change
    (root / "notes.py").write_text("new, not yet added\n")  # new file git doesn't ignore
    (root / "src" / "old.py").unlink()  # deleted, still tracked
    (root / ".venv").mkdir()
    (root / ".venv" / "lib.py").write_text("venv\n")
    (root / "results").mkdir()
    return root


def test_a_snapshot_copies_the_code_as_it_is_and_links_what_git_ignores(tmp_path):
    root = _repo(tmp_path)
    taken = snapshot.take(str(root / "src"), tmp_path / "snap" / "tree")
    copy = tmp_path / "snap" / "tree"
    assert taken["root"] == str(root) and taken["copy"] == str(copy)
    assert (copy / "src" / "model.py").read_text() == "SIZE = 2\n"  # with its uncommitted change
    assert (copy / "notes.py").exists() and not (copy / "src" / "old.py").exists()
    assert os.path.islink(copy / ".venv") and os.readlink(copy / ".venv") == str(root / ".venv")
    assert os.path.islink(copy / "results") and os.path.islink(copy / ".git")
    assert set(taken["files"]) == {".gitignore", "notes.py", "src/model.py", "train.py"}


def test_a_snapshot_is_refused_outside_a_repository_too_large_or_short_of_disk(tmp_path, monkeypatch):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(snapshot.SnapshotError, match="git repository"):
        snapshot.take(str(plain), tmp_path / "a" / "tree")
    root = _repo(tmp_path)
    monkeypatch.setattr(snapshot, "SNAPSHOT_MAX", 10)
    with pytest.raises(snapshot.SnapshotError, match="more than 10"):
        snapshot.take(str(root), tmp_path / "b" / "tree")
    monkeypatch.setattr(snapshot, "SNAPSHOT_MAX", GiB)
    monkeypatch.setattr(snapshot.shutil, "disk_usage", lambda p: type("u", (), {"free": GiB})())
    with pytest.raises(snapshot.SnapshotError, match="less than 2G free"):
        snapshot.take(str(root), tmp_path / "c" / "tree")


def test_a_queued_job_runs_its_code_as_at_submit_and_its_copy_goes_when_it_ends(make_daemon, tmp_path):
    root = _repo(tmp_path)
    d = make_daemon()
    go = tmp_path / "go"
    blocker = d.submit(req(tmp_path, [PY, "-c", f"import os, time\nwhile not os.path.exists({str(go)!r}): time.sleep(0.02)"],
                           cpus="all"))["job"]["id"]  # fmt: skip
    staged = snapshot.stage(d.paths, str(root))
    resp = d.submit(req(root, [PY, str(root / "train.py")], snapshot=staged))
    assert resp["ok"], resp
    waiting = resp["job"]["id"]
    assert job(d, waiting).state == State.QUEUED and job(d, waiting).cwd == str(root)  # its record keeps the real folder
    (root / "src" / "model.py").write_text("SIZE = 3\n")  # the agent keeps editing while it waits
    go.touch()
    run_until(d, lambda: finished(d, blocker) and finished(d, waiting))
    assert open(job(d, waiting).output).read().strip() == "SIZE 2"
    assert not (d.paths.job_dir(waiting) / "snapshot").exists()


def test_a_copy_the_job_wrote_into_is_kept_and_said_where(make_daemon, tmp_path):
    from dibs.render import render_show

    root = _repo(tmp_path)
    d = make_daemon()
    staged = snapshot.stage(d.paths, str(root))
    resp = d.submit(req(root, [PY, "-c", "open('out.txt', 'w').write('result')"], snapshot=staged))
    run_until(d, lambda: finished(d, resp["job"]["id"]))
    ended = job(d, resp["job"]["id"])
    kept = d.paths.job_dir(ended.id) / "snapshot" / "tree"
    assert (kept / "out.txt").read_text() == "result"
    assert f"it wrote files into its copy of the code, kept at {kept}" in render_show(d.show(ended.id)["job"], d.clock())


def test_submit_with_snapshot_outside_a_repository_says_why(dibs_env, tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    result = subprocess.run([PY, "-m", "dibs", "submit", "--cpus", "1", "--mem", "1G", "--time", "1m", "--snapshot", "--", "true"],
                            env=dibs_env, cwd=plain, capture_output=True, text=True, timeout=60)  # fmt: skip
    assert result.returncode == 2 and "--snapshot needs the job's folder to be in a git repository" in result.stderr
    subprocess.run([PY, "-m", "dibs", "daemon", "stop"], env=dibs_env, capture_output=True, timeout=60)


def test_paths_into_the_repository_inside_a_shell_string_point_into_the_copy():
    # Agents' pipelines submit `bash -c "cd /repo/...; python /repo/src/..."`: those ran the live files.
    cwd, argv = snapshot.run_paths("/w/proj", "/s/tree", "/w/proj/sub", [
        "bash", "-c", "set -e; cd /w/proj/data/pin; python /w/proj/src/gen.py --out '/w/proj/out' /w/project2/x /w/proj",
        "/w/proj/run.py"])  # fmt: skip
    assert cwd == "/s/tree/sub" and argv[3] == "/s/tree/run.py"
    assert argv[2] == "set -e; cd /s/tree/data/pin; python /s/tree/src/gen.py --out '/s/tree/out' /w/project2/x /s/tree"


def test_the_snapshot_help_says_what_still_points_at_the_real_files():
    from dibs.instructions import AGENT_HELP

    assert "~/ paths in a shell string and paths in environment variables still point at the real files" in " ".join(
        AGENT_HELP.split())


def _live_copy(paths, root, job_id):
    """A copy staged and moved into a job's folder, as the daemon does at submit."""
    staged = snapshot.stage(paths, str(root))
    paths.job_dir(job_id).mkdir(parents=True, exist_ok=True)
    os.rename(staged, paths.job_dir(job_id) / "snapshot")
    return paths.job_dir(job_id) / "snapshot" / "tree"


def _paths(tmp_path):
    from dibs.paths import ensure_dirs, get_paths

    paths = get_paths({"DIBS_HOME": str(tmp_path / "home"), "HOME": str(tmp_path)})
    ensure_dirs(paths)
    return paths


def test_a_copy_shares_unchanged_files_with_the_newest_copy_and_copies_the_rest(tmp_path):
    # Issue #233: seven waiting jobs' copies of one repository took 755M each, 5G in 12 minutes.
    import json

    root, paths = _repo(tmp_path), _paths(tmp_path)
    first = _live_copy(paths, root, 1)
    (root / "train.py").write_text("print('changed')\n")  # edited between the two submits
    second = _live_copy(paths, root, 2)
    same = lambda rel: os.stat(first / rel).st_ino == os.stat(second / rel).st_ino  # noqa: E731
    assert same("src/model.py") and same("notes.py") and not same("train.py")
    assert (second / "train.py").read_text() == "print('changed')\n"
    assert not os.access(second / "src" / "model.py", os.W_OK)  # shared, so nobody writes it in place
    assert os.stat(root / "src" / "model.py").st_ino not in (os.stat(first / "src" / "model.py").st_ino,)  # never the repository's
    manifest = json.loads((paths.job_dir(2) / "snapshot" / snapshot.MANIFEST).read_text())
    assert manifest["copied"] == len("print('changed')\n")  # only what it copied
    assert snapshot.clear(paths.job_dir(2) / "snapshot") is None  # shared files aren't taken for written ones
    assert (first / "src" / "model.py").read_text() == "SIZE = 2\n"  # the other copy keeps its files


def test_a_file_changed_in_the_old_copy_is_copied_not_shared(tmp_path):
    root, paths = _repo(tmp_path), _paths(tmp_path)
    first = _live_copy(paths, root, 1)
    (first / "notes.py").write_text("a job wrote here\n")
    second = _live_copy(paths, root, 2)
    assert os.stat(first / "notes.py").st_ino != os.stat(second / "notes.py").st_ino
    assert (second / "notes.py").read_text() == "new, not yet added\n"


def test_a_file_that_cant_be_linked_is_copied(tmp_path, monkeypatch):
    root, paths = _repo(tmp_path), _paths(tmp_path)
    _live_copy(paths, root, 1)

    def refuse(*a, **k):
        raise OSError("the old copy went meanwhile")

    monkeypatch.setattr(snapshot.os, "link", refuse)
    second = _live_copy(paths, root, 2)
    assert (second / "src" / "model.py").read_text() == "SIZE = 2\n"


def test_shared_files_dont_count_against_the_free_disk_a_copy_needs(tmp_path, monkeypatch):
    root, paths = _repo(tmp_path), _paths(tmp_path)
    _live_copy(paths, root, 1)
    size = sum(os.lstat(root / f).st_size for f in (".gitignore", "notes.py", "src/model.py", "train.py"))
    monkeypatch.setattr(snapshot.shutil, "disk_usage", lambda p: type("u", (), {"free": 2 * GiB + size - 1})())
    _live_copy(paths, root, 2)  # nothing changed: it copies nothing, so it fits
