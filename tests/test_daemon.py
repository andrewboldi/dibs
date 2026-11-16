import os
import shutil
import subprocess
import sys
import time

from conftest import FakeBackend, run_until

from dibs import launcher, procs
from dibs.model import State
from dibs.units import GiB

PY = sys.executable


def req(cwd, argv, *, cpus=1, mem=1 * GiB, vram=None, time_=60.0, session="s1", **extra):
    return {
        "argv": list(argv),
        "cwd": str(cwd),
        "env": dict(os.environ),
        "session": session,
        "session_label": f"proj·{session[:4]}",
        "request": {"cpus": cpus, "mem": mem, "vram": vram, "gpus": None, "time": time_},
        **extra,
    }


def submit(daemon, *args, **kw):
    resp = daemon.submit(req(*args, **kw))
    assert resp["ok"], resp
    return resp["job"]["id"]


def job(daemon, job_id):
    return daemon.store.get_job(job_id)


def finished(daemon, job_id):
    return job(daemon, job_id).state in (State.DONE, State.FAILED, State.CANCELLED)


def command_started(daemon, job_id, name):
    j = job(daemon, job_id)
    return any((procs.cmdline(p) or [""])[0] == name for p in launcher.job_pids(j))


def test_submit_runs_and_finishes(daemon, tmp_path):
    job_id = submit(daemon, tmp_path, [PY, "-c", "print('hello')"])
    assert job_id == 1
    run_until(daemon, lambda: finished(daemon, job_id))
    j = job(daemon, job_id)
    assert (j.state, j.exit_code, j.reason) == (State.DONE, 0, None)
    assert "hello" in open(j.output).read()


def test_short_job_overtakes_long_end_to_end(daemon, tmp_path):
    x = submit(daemon, tmp_path, ["sleep", "1"], vram="all", time_=5, session="x")
    assert job(daemon, x).state == State.RUNNING
    b = submit(daemon, tmp_path, ["sleep", "0.2"], vram="all", time_=600, session="b")
    a = submit(daemon, tmp_path, ["sleep", "0.2"], vram="all", time_=60, session="a")
    run_until(daemon, lambda: finished(daemon, a) and finished(daemon, b))
    assert job(daemon, a).started < job(daemon, b).started


def test_cancel_queued(daemon, tmp_path):
    blocker = submit(daemon, tmp_path, ["sleep", "30"], vram="all")
    waiting = submit(daemon, tmp_path, ["sleep", "1"], vram="all")
    resp = daemon.cancel([waiting])
    assert resp == {"ok": True, "cancelled": [waiting], "errors": []}
    j = job(daemon, waiting)
    assert (j.state, j.reason) == (State.CANCELLED, "cancelled")
    daemon.cancel([blocker])


def test_cancel_running(daemon, tmp_path):
    job_id = submit(daemon, tmp_path, ["sleep", "30"])
    run_until(daemon, lambda: command_started(daemon, job_id, "sleep"))
    daemon.cancel([job_id])
    run_until(daemon, lambda: finished(daemon, job_id))
    j = job(daemon, job_id)
    assert (j.state, j.exit_code, j.reason) == (State.CANCELLED, 143, "cancelled")


def test_cancel_reports_unknown_and_finished(daemon, tmp_path):
    job_id = submit(daemon, tmp_path, ["true"])
    run_until(daemon, lambda: finished(daemon, job_id))
    resp = daemon.cancel([job_id, 99])
    assert resp["cancelled"] == []
    assert resp["errors"] == [f"job {job_id} already done", "no job 99"]


def test_failed_job(daemon, tmp_path):
    job_id = submit(daemon, tmp_path, [PY, "-c", "raise SystemExit(1)"])
    run_until(daemon, lambda: finished(daemon, job_id))
    j = job(daemon, job_id)
    assert (j.state, j.exit_code, j.reason) == (State.FAILED, 1, None)


def test_rejects_unfit(daemon, tmp_path):
    resp = daemon.submit(req(tmp_path, ["true"], vram=12 * GiB))
    assert resp == {
        "ok": False,
        "code": "rejected",
        "error": "asks for 12G VRAM, but the largest GPU dibs manages has 8G",
    }


def test_rejects_unknown_command_missing_cwd_and_empty_argv(daemon, tmp_path):
    assert daemon.submit(req(tmp_path, ["pyhton", "x.py"]))["error"] == "command not found: pyhton"
    assert daemon.submit(req(tmp_path, ["./nope.sh"]))["error"] == "command not found: ./nope.sh"
    gone = tmp_path / "gone"
    assert daemon.submit(req(gone, ["true"]))["error"] == f"working directory {gone} doesn't exist"
    assert daemon.submit(req(tmp_path, []))["error"] == "nothing to run: give the command after --"
    assert daemon.store.jobs() == []


def test_relative_script_found_in_cwd(daemon, tmp_path):
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/sh\necho ran\n")
    script.chmod(0o755)
    job_id = submit(daemon, tmp_path, ["./run.sh"])
    run_until(daemon, lambda: finished(daemon, job_id))
    assert job(daemon, job_id).state == State.DONE


def test_missing_cwd_fails_cleanly(daemon, tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    blocker = submit(daemon, tmp_path, ["sleep", "0.5"], vram="all")
    victim = submit(daemon, project, ["true"], vram="all")
    assert job(daemon, victim).state == State.QUEUED
    shutil.rmtree(project)
    run_until(daemon, lambda: finished(daemon, victim))
    j = job(daemon, victim)
    assert j.state == State.FAILED and "no longer exists" in j.reason
    after = submit(daemon, tmp_path, ["true"])
    run_until(daemon, lambda: finished(daemon, after))
    assert job(daemon, after).state == State.DONE
    assert job(daemon, blocker).state == State.DONE


def test_recover_adopts_running_job(make_daemon, tmp_path):
    first = make_daemon()
    job_id = submit(first, tmp_path, ["sleep", "1.5"])
    assert job(first, job_id).state == State.RUNNING
    first.close()  # the daemon goes away; the job doesn't
    second = make_daemon()
    second.recover()
    assert job(second, job_id).state == State.RUNNING
    run_until(second, lambda: finished(second, job_id))
    assert job(second, job_id).state == State.DONE


def test_recover_marks_lost(make_daemon, tmp_path):
    first = make_daemon()
    submit(first, tmp_path, ["sleep", "30"], vram="all")
    job_id = submit(first, tmp_path, ["true"], vram="all")  # queued behind the blocker
    dead = subprocess.Popen(["true"])
    dead.wait()
    j = job(first, job_id)
    j.state, j.started, j.shim_pid, j.shim_ticks = State.RUNNING, time.time(), dead.pid, None
    first.store.update_job(j)
    first.close()
    second = make_daemon()
    second.recover()
    j = job(second, job_id)
    assert (j.state, j.reason) == (State.FAILED, "lost: its process ended while dibs was not running")


def test_retention_deletes_old_jobs(make_daemon, tmp_path):
    now = [time.time()]
    d = make_daemon(clock=lambda: now[0])
    job_id = submit(d, tmp_path, ["true"])
    run_until(d, lambda: finished(d, job_id))
    job_dir = d.paths.job_dir(job_id)
    assert job_dir.exists()
    now[0] += 15 * 86400
    d.tick()
    assert d.store.get_job(job_id) is None and not job_dir.exists()


def test_snapshot_has_predictions(daemon, tmp_path):
    blocker = submit(daemon, tmp_path, ["sleep", "30"], vram="all", time_=100)
    waiting = submit(daemon, tmp_path, ["true"], vram=2 * GiB)
    snap = daemon.snapshot()
    views = {v["id"]: v for v in snap["jobs"]}
    assert views[blocker]["state"] == "running"
    assert views[blocker]["expected_end"] > snap["now"] and views[blocker]["over_estimate"] == 0
    q = views[waiting]
    assert q["position"] == 1 and q["predicted_start"] >= snap["now"] + 99
    assert q["waiting_for"] == "needs 2G VRAM (0 free)"
    assert "env" not in q
    assert snap["capacity"]["gpus"][0]["vram"] == 8 * GiB
    assert snap["guard"]["mode"] == "all"
    daemon.cancel([blocker, waiting])


def test_snapshot_session_filter_and_finished(daemon, tmp_path):
    mine = submit(daemon, tmp_path, ["true"], session="mine")
    submit(daemon, tmp_path, ["sleep", "30"], session="other")
    run_until(daemon, lambda: finished(daemon, mine))
    assert [v["id"] for v in daemon.snapshot(session="mine")["jobs"]] == []
    assert [v["id"] for v in daemon.snapshot(include_finished=True, session="mine")["jobs"]] == [mine]


def test_finished_callbacks(daemon, tmp_path):
    seen = []
    daemon.on_finished(lambda j: seen.append((j.id, j.state)))
    job_id = submit(daemon, tmp_path, ["true"])
    run_until(daemon, lambda: seen)
    assert seen == [(job_id, State.DONE)]


def test_info(daemon):
    info = daemon.info()
    assert info["ok"] and info["pid"] == os.getpid()
    assert info["gpu_backend"] == "fake"
    assert info["enforcement"] == {
        "cpu_pinning": True,
        "ram_cap": False,
        "vram_watch": True,
        "guard": "all",
    }


def test_vram_watch_off_when_driver_is_blind(make_daemon):
    backend = FakeBackend()
    backend.blind = True
    assert make_daemon(backend=backend).info()["enforcement"]["vram_watch"] is False


def test_lost_job_says_what_the_launcher_said(daemon, tmp_path, monkeypatch):
    def broken_launch(job, paths, *, use_systemd):
        job_dir = paths.job_dir(job.id)
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "shim.log").write_text("Failed to start transient scope unit: Unknown assignment\n")
        gone = subprocess.Popen(["true"])
        gone.wait()
        return gone.pid, None, None

    monkeypatch.setattr(launcher, "launch", broken_launch)
    job_id = submit(daemon, tmp_path, ["true"])
    run_until(daemon, lambda: finished(daemon, job_id))
    assert job(daemon, job_id).reason == (
        "lost: its launcher exited without reporting (Failed to start transient scope unit: Unknown assignment)"
    )


class _DaemonDies(BaseException):
    """Stands in for the daemon dying (not an error schedule() would catch)."""


def test_crash_while_launching_is_adopted_not_lost(make_daemon, tmp_path, monkeypatch):
    # Issue #4: the job is saved as running before launch() returns its launcher's pid; a daemon
    # that dies in between used to leave a live job that recover() then declared lost.
    first = make_daemon()
    real_launch = launcher.launch

    def launch_then_die(job, paths, *, use_systemd):
        real_launch(job, paths, use_systemd=use_systemd)
        raise _DaemonDies

    monkeypatch.setattr(launcher, "launch", launch_then_die)
    try:
        first.submit(req(tmp_path, ["sleep", "1.5"]))
    except _DaemonDies:
        pass
    monkeypatch.setattr(launcher, "launch", real_launch)
    job_id = 1
    assert job(first, job_id).state == State.RUNNING and job(first, job_id).shim_pid is None
    first.close()
    second = make_daemon()
    second.recover()
    assert job(second, job_id).state == State.RUNNING and job(second, job_id).shim_pid
    run_until(second, lambda: finished(second, job_id))
    assert job(second, job_id).state == State.DONE


def test_relative_path_entries_are_looked_up_from_the_jobs_folder(daemon, tmp_path):
    # Issue #7: the check that a command exists resolved PATH entries like "bin" from the
    # daemon's folder, while the job runs (and finds its command) from its own.
    project = tmp_path / "proj"
    (project / "bin").mkdir(parents=True)
    tool = project / "bin" / "mytool"
    tool.write_text("#!/bin/sh\necho found-it\n")
    tool.chmod(0o755)
    request = req(project, ["mytool"])
    request["env"] = {**request["env"], "PATH": "bin:/usr/bin:/bin"}
    resp = daemon.submit(request)
    assert resp["ok"], resp
    job_id = resp["job"]["id"]
    run_until(daemon, lambda: finished(daemon, job_id))
    assert job(daemon, job_id).state == State.DONE
    assert "found-it" in open(job(daemon, job_id).output).read()
