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
