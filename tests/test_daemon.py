import json
import os
import re
import shutil
import subprocess
import sys
import time

import pytest

from conftest import FakeBackend, capacity, run_until

from dibs import launcher, procs
from dibs.daemon import OUTSIDE_WINDOW
from dibs.model import State
from dibs.units import GiB, MiB

PY = sys.executable


def req(cwd, argv, *, cpus=1, mem=1 * GiB, vram=None, time_=60.0, session="s1", exclusive=False, **extra):
    return {
        "argv": list(argv),
        "cwd": str(cwd),
        "env": dict(os.environ),
        "session": session,
        "session_label": f"proj·{session[:4]}",
        "request": {"cpus": cpus, "mem": mem, "vram": vram, "gpus": None, "time": time_, "exclusive": exclusive},
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


@pytest.mark.parametrize("retry", [False, True])
def test_a_job_a_reboot_killed_keeps_the_sleep_counted_before_it(make_daemon, tmp_path, monkeypatch, retry):
    # Issue #78: a reboot resets the machine's count of time asleep. A job that had run 13 min
    # across 4h38m of sleep before a reboot was recorded as lost "after 4h51m".
    start, now, slept = 1_000_000.0, [1_000_000.0], [100.0]
    monkeypatch.setattr(procs, "boot_id", lambda: "boot-a")
    first = make_daemon(clock=lambda: now[0], slept=lambda: slept[0])
    submit(first, tmp_path, ["sleep", "30"], vram="all")
    job_id = submit(first, tmp_path, ["true"], vram="all")  # queued behind the blocker
    now[0] += 10 * 60 + 4 * 3600  # 10 minutes running, then 4 hours asleep
    slept[0] += 4 * 3600
    first.tick()  # awake again: the daemon notes it
    noted = now[0]
    dead = subprocess.Popen(["true"])
    dead.wait()
    j = job(first, job_id)  # as if it had been running since `start`
    j.state, j.started, j.sleep_at_start, j.shim_pid, j.shim_ticks = State.RUNNING, start, 100.0, dead.pid, None
    j.retry = retry
    first.store.update_job(j)
    first.close()
    monkeypatch.setattr(procs, "boot_id", lambda: "boot-b")  # the machine restarted
    now[0] += 120
    slept[0] = 0.0
    second = make_daemon(clock=lambda: now[0], slept=lambda: slept[0])
    second.recover()
    j = job(second, job_id)
    if not retry:
        assert (j.state, j.reason) == (State.FAILED, "lost: the machine restarted while it ran")
        assert (j.ended, j.asleep) == (noted, 4 * 3600)  # so it ran 10 minutes
    else:  # Issue #84: queued again, same request, with the lost run on record
        assert j.state == State.QUEUED and j.request.vram == 8 * GiB
        assert [(a["reason"], a["ended"], a["asleep"]) for a in j.attempts] == [
            ("lost: the machine restarted while it ran; retrying", noted, 4 * 3600)
        ]


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
    def broken_launch(job, paths, **kwargs):
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

    def launch_then_die(job, paths, **kwargs):
        real_launch(job, paths, **kwargs)
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


def test_peak_ram_is_memory_in_use_not_file_cache(daemon, tmp_path, monkeypatch):
    # Issue #32: memory.peak counts page cache, so a job that wrote big scratch files looked like it
    # had filled its reservation ("RAM 13G of 13G"). dibs keeps the peak of anon + shmem instead.
    job_id = submit(daemon, tmp_path, [PY, "-c", "import time; time.sleep(3)"])
    run_until(daemon, lambda: job(daemon, job_id).shim_pid)
    daemon.tick()
    assert job(daemon, job_id).peak_mem is None  # no scope of its own: nothing to read
    j = job(daemon, job_id)
    j.unit = "dibs-test-job-1"
    daemon.store.update_job(j)
    scope = tmp_path / "cgroup" / "app.slice" / "dibs-test-job-1.scope"
    scope.mkdir(parents=True)
    monkeypatch.setattr(procs, "CGROUP_ROOT", str(tmp_path / "cgroup"))
    real_cgroup = procs.cgroup
    monkeypatch.setattr(procs, "cgroup", lambda pid: "/app.slice/dibs-test-job-1.scope" if pid == j.shim_pid else real_cgroup(pid))

    def peak_after(anon, shmem, file):
        (scope / "memory.stat").write_text(f"anon {anon}\nfile {file}\nkernel 4096\nshmem {shmem}\n")
        daemon.tick()
        return job(daemon, job_id).peak_mem

    assert peak_after(2 * GiB, 1 * GiB, 9 * GiB) == 3 * GiB
    assert peak_after(1 * GiB, 0, 12 * GiB) == 3 * GiB  # less in use, more cache: the peak stays
    assert peak_after(4 * GiB, 0, 0) == 4 * GiB
    # The launcher reports memory.peak, cache and all; the sampled peak wins.
    real_read_exit = launcher.read_exit
    monkeypatch.setattr(launcher, "read_exit", lambda paths, i: (r := real_read_exit(paths, i)) and {**r, "peak_mem": 13 * GiB})
    run_until(daemon, lambda: finished(daemon, job_id))
    assert job(daemon, job_id).peak_mem == 4 * GiB


def test_peak_ram_falls_back_to_the_launchers_figure_and_says_what_it_counts(daemon, tmp_path, monkeypatch):
    from dibs.model import job_to_dict
    from dibs.render import render_show

    real_read_exit = launcher.read_exit
    monkeypatch.setattr(launcher, "read_exit", lambda paths, i: (r := real_read_exit(paths, i)) and {**r, "peak_mem": 2 * GiB})
    job_id = submit(daemon, tmp_path, [PY, "-c", "pass"])
    run_until(daemon, lambda: finished(daemon, job_id))
    assert job(daemon, job_id).peak_mem == 2 * GiB  # never sampled: the launcher's figure stands
    text = render_show(job_to_dict(job(daemon, job_id)), time.time())
    assert "RAM 2G of 1G (not counting file cache)" in text


def test_memory_stat_parsing():
    stat = procs.keyed_numbers("anon 2048\nfile 4096\nshmem 1024\nbogus x\n")
    assert stat == {"anon": 2048, "file": 4096, "shmem": 1024}
    assert procs.keyed_numbers(None) == {}


def test_memory_events_parsing():
    # How many of a cgroup's processes the kernel killed for RAM, and whether it reached its own limit.
    assert procs.oom_kills("low 0\nhigh 0\nmax 3178\noom 170\noom_kill 11\noom_group_kill 0\n") == (11, True)
    assert procs.oom_kills("oom 0\noom_kill 2\n") == (2, False)  # the machine ran out, not its own limit
    assert procs.oom_kills("oom 4\noom_kill 0\n") == (0, True)  # at its limit, but the kernel reclaimed enough
    assert procs.oom_kills("oom_kill x\n") == (0, False)
    assert procs.oom_kills(None) == (0, False)


def test_ram_used_outside_dibs_is_held_back_for_a_few_minutes(make_daemon, tmp_path):
    # Issue #30: the daemon measures the RAM in use outside its jobs and plans with its peak over the
    # last few minutes, so a job doesn't start into RAM a busy program let go of for a moment.
    now, used = [time.time()], [9 * GiB]
    d = make_daemon(clock=lambda: now[0], meminfo=lambda: (16 * GiB, used[0]))
    job_id = submit(d, tmp_path, [PY, "-c", "pass"], mem=8 * GiB)
    d.tick()
    assert job(d, job_id).state == State.QUEUED
    assert "9G in use outside dibs" in json.dumps(d.show(job_id)["job"])
    used[0] = 2 * GiB
    d.tick()
    assert job(d, job_id).state == State.QUEUED  # still the peak of the last few minutes
    now[0] += OUTSIDE_WINDOW + 1
    run_until(d, lambda: finished(d, job_id))


def test_ram_in_use_by_jobs_is_not_counted_as_outside(make_daemon, tmp_path, monkeypatch):
    used = [2 * GiB]  # in use outside dibs
    d = make_daemon(meminfo=lambda: (16 * GiB, used[0]))
    first = submit(d, tmp_path, [PY, "-c", "import time; time.sleep(5)"], mem=6 * GiB)
    run_until(d, lambda: job(d, first).shim_pid)
    j = job(d, first)
    j.unit = "dibs-test-job-1"
    d.store.update_job(j)
    scope = tmp_path / "cgroup" / "app.slice" / "dibs-test-job-1.scope"
    scope.mkdir(parents=True)
    (scope / "memory.stat").write_text(f"anon {5 * GiB}\nfile {3 * GiB}\nshmem 0\n")
    monkeypatch.setattr(procs, "CGROUP_ROOT", str(tmp_path / "cgroup"))
    real_cgroup = procs.cgroup
    monkeypatch.setattr(procs, "cgroup", lambda pid: "/app.slice/dibs-test-job-1.scope" if pid == j.shim_pid else real_cgroup(pid))
    used[0] = 7 * GiB  # the job now holds 5G
    d.tick()
    # 7G in use, 5G of it the running job's: at most 2G outside. 16G - 6G reserved - 2G leaves room for 7G.
    second = submit(d, tmp_path, [PY, "-c", "pass"], mem=7 * GiB)
    d.tick()
    assert job(d, second).state != State.QUEUED, json.dumps(d.show(second)["job"])


def _job_holding(d, tmp_path, monkeypatch, mem, **extra):
    """A running job whose scope dibs reads its memory from, and a file that ends the job once it exists."""
    go = tmp_path / "go"
    job_id = submit(d, tmp_path, [PY, "-c", f"import os, time\nwhile not os.path.exists({str(go)!r}): time.sleep(0.02)"],
                    mem=mem, **extra)
    run_until(d, lambda: job(d, job_id).shim_pid)
    j = job(d, job_id)
    j.unit = "dibs-test-job-1"
    d.store.update_job(j)
    scope = tmp_path / "cgroup" / "app.slice" / "dibs-test-job-1.scope"
    scope.mkdir(parents=True)
    monkeypatch.setattr(procs, "CGROUP_ROOT", str(tmp_path / "cgroup"))
    real_cgroup = procs.cgroup
    monkeypatch.setattr(procs, "cgroup", lambda pid: "/app.slice/dibs-test-job-1.scope" if pid == j.shim_pid else real_cgroup(pid))
    return job_id, scope / "memory.stat", go


def test_ram_a_running_job_is_letting_go_of_is_not_counted_as_outside(make_daemon, tmp_path, monkeypatch):
    # Issue #72: a job holding about 20G was killed at its RAM limit, and for the next two minutes the
    # waiting jobs' reasons said 19.6G was in use outside dibs while the machine had 9G in use. One
    # sample had caught the machine still holding memory the job no longer did, and the peak kept it.
    now, used = [time.time()], [2 * GiB]
    d = make_daemon(clock=lambda: now[0], meminfo=lambda: (16 * GiB, used[0]))
    first, stat, go = _job_holding(d, tmp_path, monkeypatch, 8 * GiB)
    stat.write_text(f"anon {6 * GiB}\nshmem 0\n")
    used[0] = 8 * GiB  # 6G of it the job's, 2G outside dibs
    now[0] += 1
    d.tick()
    stat.write_text(f"anon {1 * GiB}\nshmem 0\n")  # the job lets go of 5G
    now[0] += 1
    d.tick()  # while the machine still counts it as in use
    used[0] = 3 * GiB
    now[0] += 1
    d.tick()
    # 2G outside dibs at most. 16G - 8G reserved - 2G leaves room for 6G.
    second = submit(d, tmp_path, [PY, "-c", "pass"], mem=6 * GiB)
    d.tick()
    assert job(d, second).state != State.QUEUED, json.dumps(d.show(second)["job"])
    go.touch()


def test_ram_a_finished_job_is_letting_go_of_is_not_counted_as_outside(make_daemon, tmp_path, monkeypatch):
    # Issue #72, the other order: dibs has seen the job end while the machine still holds its memory.
    now, used = [time.time()], [2 * GiB]
    d = make_daemon(clock=lambda: now[0], meminfo=lambda: (16 * GiB, used[0]))
    first, stat, go = _job_holding(d, tmp_path, monkeypatch, 8 * GiB)
    stat.write_text(f"anon {6 * GiB}\nshmem 0\n")
    used[0] = 8 * GiB  # 6G of it the job's, 2G outside dibs
    now[0] += 1
    d.tick()
    go.touch()
    run_until(d, lambda: finished(d, first))
    now[0] += 1
    d.tick()  # the job has ended, and the machine still counts its 6G as in use
    used[0] = 2 * GiB
    now[0] += 1
    d.tick()
    # 2G outside dibs. 16G - 2G leaves room for 10G.
    second = submit(d, tmp_path, [PY, "-c", "pass"], mem=10 * GiB)
    d.tick()
    assert job(d, second).state != State.QUEUED, json.dumps(d.show(second)["job"])


def _killed_for_ram_holding(d, tmp_path, monkeypatch, mem, anon, shmem, file=4 * GiB, mapped=None):
    """A job that held this much (anon, shmem) at its last sample, then was killed at its RAM limit.
    mapped is memory.stat's file_mapped: the file cache its processes map, shared memory included."""
    job_id, stat, go = _job_holding(d, tmp_path, monkeypatch, mem, retry=False)
    stat.write_text(f"anon {anon}\nfile {file}\nshmem {shmem}\n" + ("" if mapped is None else f"file_mapped {mapped}\n"))
    d.tick()
    real_read_exit = launcher.read_exit
    monkeypatch.setattr(
        launcher, "read_exit",
        lambda paths, i: (r := real_read_exit(paths, i)) and {**r, "exit_code": 137, "oom_killed": True, "hit_limit": True},
    )  # fmt: skip
    go.touch()
    run_until(d, lambda: finished(d, job_id))
    return job(d, job_id)


def test_a_ram_kill_says_when_most_of_the_ram_was_shared_memory(daemon, tmp_path, monkeypatch):
    # Issue #74: six of seven RAM kills were mostly scratch files in a /tmp kept in RAM, which count
    # against --mem, and the reason said only "went over its 10G RAM reservation".
    gib = 1 << 30
    killed = _killed_for_ram_holding(daemon, tmp_path, monkeypatch, 10 * gib, int(4.1 * gib), int(5.4 * gib))
    assert killed.peak_shmem == int(5.4 * gib)
    assert killed.reason == (
        "killed: went over its 10G RAM reservation "
        "(5.4G of it was files in /dev/shm or a RAM-backed /tmp); resubmit with --mem 13G"
    )


def test_a_ram_kill_says_when_its_shared_memory_was_mapped_by_its_processes(daemon, tmp_path, monkeypatch):
    # Issue #132, the kernel's figures at a real kill: all 6.5G of shared memory was mapped by the job's
    # processes, as buffers pinned for GPU transfers are, with no files in RAM. The reason blamed files
    # in a RAM-backed /tmp, and pointing TMPDIR at a disk wouldn't have helped.
    killed = _killed_for_ram_holding(
        daemon, tmp_path, monkeypatch, 12 * GiB, 5868503040, 6968758272, file=6968762368, mapped=6970818560
    )
    assert killed.reason == (
        "killed: went over its 12G RAM reservation (6.5G of it was shared memory its processes mapped, "
        "such as buffers for GPU transfers or between worker processes); resubmit with --mem 15G"
    )


def test_a_ram_kill_names_both_kinds_of_shared_memory_when_the_counters_cant_tell(daemon, tmp_path, monkeypatch):
    # 4G shared and 2G of other file cache, with 3G of the two mapped: 1-3G of the shared memory was.
    killed = _killed_for_ram_holding(daemon, tmp_path, monkeypatch, 10 * GiB, 5 * GiB, 4 * GiB, file=6 * GiB, mapped=3 * GiB)
    assert killed.reason == (
        "killed: went over its 10G RAM reservation (4G of it was files in /dev/shm or a RAM-backed /tmp, "
        "or shared memory its processes mapped, such as buffers for GPU transfers or between worker processes); "
        "resubmit with --mem 13G"
    )


def test_a_ram_kill_with_little_shared_memory_says_only_that_it_went_over(daemon, tmp_path, monkeypatch):
    gib = 1 << 30
    killed = _killed_for_ram_holding(daemon, tmp_path, monkeypatch, 8 * gib, int(7.8 * gib), int(0.1 * gib))
    assert killed.reason == "killed: went over its 8G RAM reservation; resubmit with --mem 10G"


def test_a_job_killed_at_its_ram_limit_shows_it_reached_that_limit(daemon, tmp_path, monkeypatch):
    # Issue #127: a job can grow past dibs's last sample before the kernel kills it. One killed at its
    # 200M limit showed "peak RAM 14M of 200M", and agents size their next run from the peak.
    killed = _killed_for_ram_holding(daemon, tmp_path, monkeypatch, 8 * GiB, 2 * GiB, 0)
    assert killed.peak_mem == 8 * GiB


def test_a_retry_job_out_of_retries_says_what_to_resubmit_with(daemon, tmp_path, monkeypatch):
    job_id = submit(daemon, tmp_path, [PY, "-c", "pass"], mem=2 * GiB, retry=True)
    j = job(daemon, job_id)
    j.attempts = [{"reason": "an earlier try"}, {"reason": "another"}]
    daemon.store.update_job(j)
    _oom_on_first_exit(monkeypatch, job_id)
    run_until(daemon, lambda: finished(daemon, job_id))
    j = job(daemon, job_id)
    assert (j.state, j.reason) == (State.FAILED, "killed: went over its 2G RAM reservation; resubmit with --mem 3G")


def _launcher_finds(monkeypatch, job_id, **found):
    """Add to a job's first report what its launcher reads in its scope's memory.events as it ends. The
    tests' scopes are plain folders, which a launcher never reads."""
    real_read_exit = launcher.read_exit
    seen = []

    def read_exit(paths, i):
        r = real_read_exit(paths, i)
        if r and i == job_id and not seen:
            seen.append(True)
            return {**r, **found}
        return r

    monkeypatch.setattr(launcher, "read_exit", read_exit)


def test_a_job_is_stopped_whole_once_one_of_its_processes_is_killed_at_its_ram_limit(daemon, tmp_path, monkeypatch):
    # Issue #125: the kernel kills only the process that went over, and the job's scope lives on so its
    # launcher can report. A job whose pool waited on its killed workers sat at 0 CPU for three hours.
    job_id, stat, go = _job_holding(daemon, tmp_path, monkeypatch, 2 * GiB, retry=False)
    events = stat.parent / "memory.events"
    events.write_text("low 0\nhigh 0\nmax 40\noom 3\noom_kill 0\n")
    daemon.tick()
    assert job(daemon, job_id).stop_requested is None  # it reached its limit, but nothing was killed
    events.write_text("low 0\nhigh 0\nmax 41\noom 4\noom_kill 1\n")
    daemon.tick()
    assert job(daemon, job_id).stop_reason == "killed: went over its 2G RAM reservation"
    _launcher_finds(monkeypatch, job_id, oom_killed=True, hit_limit=True)
    run_until(daemon, lambda: finished(daemon, job_id))  # its command would never end by itself
    j = job(daemon, job_id)
    assert (j.state, j.reason) == (State.FAILED, "killed: went over its 2G RAM reservation; resubmit with --mem 3G")


def test_a_retry_job_stopped_that_way_runs_again_with_more_ram(daemon, tmp_path, monkeypatch):
    # The same retry, for the same reason, as when the process killed was the job's own command.
    job_id, stat, go = _job_holding(daemon, tmp_path, monkeypatch, 2 * GiB, retry=True)
    (stat.parent / "memory.events").write_text("oom 1\noom_kill 1\n")
    _launcher_finds(monkeypatch, job_id, oom_killed=True, hit_limit=True)
    run_until(daemon, lambda: job(daemon, job_id).attempts)
    j = job(daemon, job_id)
    assert j.request.mem == 3 * GiB
    assert [a["reason"] for a in j.attempts] == ["killed: went over its 2G RAM reservation; retrying with --mem 3G"]
    go.touch()


def test_a_job_is_stopped_whole_when_the_machine_ran_out_and_killed_one_of_its_processes(daemon, tmp_path, monkeypatch):
    job_id, stat, go = _job_holding(daemon, tmp_path, monkeypatch, 2 * GiB, retry=False)
    stat.write_text(f"anon {GiB}\nshmem 0\n")
    (stat.parent / "memory.events").write_text("oom 0\noom_kill 2\n")  # not at its own limit
    daemon.tick()
    assert job(daemon, job_id).stop_reason == "killed when the machine ran out of RAM"
    _launcher_finds(monkeypatch, job_id, oom_killed=True, hit_limit=False)
    run_until(daemon, lambda: finished(daemon, job_id))
    j = job(daemon, job_id)
    assert (j.state, j.reason) == (State.FAILED, "killed when the machine ran out of RAM")
    assert j.peak_mem == GiB  # it never reached its own limit, and needs no more


def test_a_job_that_exits_0_after_one_of_its_processes_was_killed_for_ram_has_failed(daemon, tmp_path, monkeypatch):
    # Issue #125: such a job was recorded as done, with the reason "killed: went over its RAM
    # reservation". Part of its work was killed, and dibs stops the rest when it sees the kill first.
    job_id = submit(daemon, tmp_path, [PY, "-c", "pass"], mem=2 * GiB)
    _launcher_finds(monkeypatch, job_id, exit_code=0, oom_killed=True, hit_limit=True)
    run_until(daemon, lambda: finished(daemon, job_id))
    j = job(daemon, job_id)
    assert (j.state, j.reason) == (State.FAILED, "killed: went over its 2G RAM reservation; resubmit with --mem 3G")


def test_a_job_being_cancelled_stays_cancelled_when_one_of_its_processes_is_killed_for_ram(daemon, tmp_path, monkeypatch):
    job_id, stat, go = _job_holding(daemon, tmp_path, monkeypatch, 2 * GiB, retry=True)
    daemon.cancel([job_id])
    (stat.parent / "memory.events").write_text("oom 1\noom_kill 1\n")
    _launcher_finds(monkeypatch, job_id, oom_killed=True, hit_limit=True)
    run_until(daemon, lambda: finished(daemon, job_id))
    j = job(daemon, job_id)
    assert (j.state, j.reason, j.attempts) == (State.CANCELLED, "cancelled", [])


def test_memory_events_dibs_cant_read_stop_nothing(daemon, tmp_path, monkeypatch):
    job_id, stat, go = _job_holding(daemon, tmp_path, monkeypatch, 2 * GiB)
    events = stat.parent / "memory.events"
    for garbled in (b"oom_kill many\noom ?\n", b"oom_kill\n", b"oom 1\noom_kill \xff\xfe1\n"):
        events.write_bytes(garbled)
        daemon.tick()
    events.unlink()
    events.mkdir()  # can't be read as a file
    daemon.tick()
    assert job(daemon, job_id).stop_requested is None
    other = submit(daemon, tmp_path, [PY, "-c", "pass"])
    run_until(daemon, lambda: finished(daemon, other))  # and scheduling went on
    go.touch()
    run_until(daemon, lambda: finished(daemon, job_id))
    assert job(daemon, job_id).state == State.DONE


def test_time_the_machine_slept_is_not_run_time(make_daemon, tmp_path, monkeypatch):
    # Issue #76: a 2.5-hour job that had run 65 min when a laptop slept for 3 hours showed "over
    # estimate by 1h35m" on waking, and the planner took it to be about to end.
    start, now, slept = 1_000_000.0, [1_000_000.0], [500.0]
    d = make_daemon(clock=lambda: now[0], slept=lambda: slept[0])
    go = tmp_path / "go"
    holding = [PY, "-c", f"import os, time\nwhile not os.path.exists({str(go)!r}): time.sleep(0.02)"]
    running = submit(d, tmp_path, holding, cpus="all", time_=9000.0)
    run_until(d, lambda: job(d, running).shim_pid)
    now[0] += 65 * 60  # 65 minutes running
    now[0] += 3 * 3600  # then 3 hours asleep
    slept[0] += 3 * 3600
    waiting = submit(d, tmp_path, [PY, "-c", "pass"], cpus="all")
    assert d.show(waiting)["job"]["predicted_start"] == start + 3 * 3600 + 9000.0
    view = d.show(running)["job"]
    assert view["asleep"] == 3 * 3600 and view["over_estimate"] == 0
    assert view["expected_end"] == start + 3 * 3600 + 9000.0
    # The launcher reads the counter when the command exits; the job keeps what it slept.
    real_read_exit = launcher.read_exit
    monkeypatch.setattr(launcher, "read_exit", lambda paths, i: (r := real_read_exit(paths, i)) and {**r, "slept": 500.0 + 3 * 3600})
    go.touch()
    run_until(d, lambda: finished(d, running))
    assert job(d, running).asleep == 3 * 3600


def test_run_times_shown_leave_out_the_time_the_machine_slept():
    from dibs.model import Job, Request, State, job_to_dict
    from dibs.render import render_earlier, render_show

    gib, t = 1 << 30, 1_000_000.0
    request = Request(cpus=6, mem=8 * gib, gpus=1, vram=4 * gib, time=9000.0)
    ended = Job(id=263, name="train.py", argv=["python", "train.py"], cwd="/w", env={}, request=request,
                session="s", session_label="w·s", submitted=t, tag=t + 9000, state=State.DONE, output="/w/out",
                started=t, ended=t + 4 * 3600 + 300, exit_code=0, cpu_seconds=2.0 * 3900, asleep=3 * 3600.0)  # fmt: skip
    assert "#263 done after 1h5m (estimated 2h30m): 2.0 of 6 CPUs" in render_earlier([job_to_dict(ended)])
    assert "after 1h5m (plus 3h asleep), exit 0" in render_show(job_to_dict(ended), t + 5 * 3600)
    view = {**job_to_dict(ended), "state": "running", "ended": None, "exit_code": None, "cpu_ids": [0], "gpu_ids": [0],
            "expected_end": t + 3 * 3600 + 9000, "over_estimate": 0.0}  # fmt: skip
    assert "(1h5m, plus 3h asleep) on CPUs 0" in render_show(view, t + 4 * 3600 + 300)


def _held_behind(d, tmp_path):
    """A running 3.4G job, session a's queued 7.2G job, and session b's 3.4G hour-long job, which
    would fit now but is held back so as not to delay a's job."""
    gib = 1 << 30
    runner = submit(d, tmp_path, ["sleep", "30"], vram=int(3.4 * gib), time_=900.0, session="r")
    run_until(d, lambda: job(d, runner).state == State.RUNNING)
    first = submit(d, tmp_path, ["sleep", "1"], vram=int(7.2 * gib), time_=480.0, session="a")
    held = submit(d, tmp_path, ["sleep", "1"], vram=int(3.4 * gib), time_=3600.0, session="b")
    d.tick()
    assert job(d, held).state == State.QUEUED
    return first, held


def test_replacing_a_queued_job_leaves_no_gap_for_another_job(daemon, tmp_path):
    # Issue #93: an agent cancelled its queued job and resubmitted it; in between, the other session's
    # held-back job started and kept the GPU for its hour.
    first, held = _held_behind(daemon, tmp_path)
    gib = 1 << 30
    resp = daemon.submit(req(tmp_path, ["sleep", "1"], vram=int(7.2 * gib), time_=1080.0, session="a", replace=first))
    assert resp["ok"], resp
    assert resp["replaced"] == first and job(daemon, first).state == State.CANCELLED
    assert job(daemon, first).reason == f"replaced by job {resp['job']['id']}"
    assert job(daemon, held).state == State.QUEUED  # still held: the replacement is ahead of it
    assert job(daemon, resp["job"]["id"]).tag < job(daemon, held).tag


def test_a_replacement_that_is_refused_leaves_the_old_job_alone(daemon, tmp_path):
    first, _ = _held_behind(daemon, tmp_path)
    resp = daemon.submit(req(tmp_path, ["no-such-command-xyz"], session="a", replace=first))
    assert not resp["ok"] and job(daemon, first).state == State.QUEUED
    resp = daemon.submit(req(tmp_path, ["sleep", "1"], session="a", replace=99999))
    assert not resp["ok"] and resp["error"] == "no job 99999 to replace"


def test_replacing_a_running_job_stops_it(daemon, tmp_path):
    running = submit(daemon, tmp_path, ["sleep", "30"])
    run_until(daemon, lambda: job(daemon, running).state == State.RUNNING)
    resp = daemon.submit(req(tmp_path, ["sleep", "1"], replace=running))
    assert resp["ok"], resp
    run_until(daemon, lambda: finished(daemon, running))
    assert job(daemon, running).state == State.CANCELLED


def _oom_on_first_exit(monkeypatch, job_id):
    real_read_exit = launcher.read_exit
    seen = []

    def read_exit(paths, i):
        r = real_read_exit(paths, i)
        if r and i == job_id and not seen:
            seen.append(True)
            return {**r, "exit_code": 137, "oom_killed": True, "hit_limit": True}  # at its own limit
        return r

    monkeypatch.setattr(launcher, "read_exit", read_exit)


def test_a_job_killed_at_its_ram_limit_is_retried_with_more_ram(daemon, tmp_path, monkeypatch):
    # Issue #84: 25% more, rounded up to a whole gigabyte, with the same id and place in line.
    job_id = submit(daemon, tmp_path, [PY, "-c", "pass"], mem=2 * GiB, retry=True)
    _oom_on_first_exit(monkeypatch, job_id)
    run_until(daemon, lambda: job(daemon, job_id).state == State.DONE)
    j = job(daemon, job_id)
    assert j.request.mem == 3 * GiB
    assert [a["reason"] for a in j.attempts] == ["killed: went over its 2G RAM reservation; retrying with --mem 3G"]


def test_a_ram_kill_is_not_retried_when_more_ram_wont_fit(daemon, tmp_path, monkeypatch):
    job_id = submit(daemon, tmp_path, [PY, "-c", "pass"], mem=14 * GiB, retry=True)  # 17.5G won't fit in 16G
    _oom_on_first_exit(monkeypatch, job_id)
    run_until(daemon, lambda: finished(daemon, job_id))
    j = job(daemon, job_id)
    assert (j.state, j.reason, j.attempts) == (State.FAILED, "killed: went over its 14G RAM reservation", [])


def test_a_job_submitted_without_retry_is_not_run_again(daemon, tmp_path, monkeypatch):
    # Issue #116: a command that isn't safe to repeat must not run twice; the agent opts in.
    job_id = submit(daemon, tmp_path, [PY, "-c", "pass"], mem=2 * GiB)
    _oom_on_first_exit(monkeypatch, job_id)
    run_until(daemon, lambda: finished(daemon, job_id))
    assert (job(daemon, job_id).state, job(daemon, job_id).attempts) == (State.FAILED, [])


def test_dibs_show_lists_the_tries_that_failed_for_a_resource_reason(daemon, tmp_path, monkeypatch):
    from dibs.render import render_show

    job_id = submit(daemon, tmp_path, [PY, "-c", "pass"], mem=2 * GiB, retry=True)
    _oom_on_first_exit(monkeypatch, job_id)
    run_until(daemon, lambda: job(daemon, job_id).state == State.DONE)
    text = render_show(daemon.show(job_id)["job"], time.time())
    assert re.search(r"try 1 +killed: went over its 2G RAM reservation; retrying with --mem 3G", text), text


def test_a_job_gets_its_own_temp_folder_and_dibs_deletes_it_when_the_job_ends(daemon, tmp_path):
    # Issue #89: TMPDIR is a fresh folder of the job's own, and nothing in it outlives the job.
    script = "import os, tempfile; d = tempfile.gettempdir(); open(os.path.join(d, 'scratch'), 'w').write('x'); print(d)"
    job_id = submit(daemon, tmp_path, [PY, "-c", script])
    run_until(daemon, lambda: finished(daemon, job_id))
    folder = open(job(daemon, job_id).output).read().strip()
    assert os.path.basename(folder) == f"dibs-{daemon.paths.instance}-job-{job_id}"
    assert not os.path.exists(folder)


def test_recovery_sweeps_temp_folders_of_jobs_that_ended(make_daemon, tmp_path):
    from dibs import launcher as dl

    d = make_daemon()
    orphan = dl.job_tmp(d.paths, 999)
    orphan.mkdir(parents=True)
    (orphan / "scratch").write_text("x")
    d.recover()
    assert not orphan.exists()


def test_submit_tells_the_session_how_its_estimates_compared(daemon, tmp_path):
    # Issue #95: deterministic feedback, never a rescaled estimate.
    for _ in range(5):
        j = submit(daemon, tmp_path, [PY, "-c", "pass"], time_=600.0, session="s1")
        run_until(daemon, lambda: finished(daemon, j))
    resp = daemon.submit(req(tmp_path, [PY, "-c", "pass"], session="s1"))
    assert resp["estimates"]["jobs"] == 5 and resp["estimates"]["median"] < 0.05
    assert "estimates" not in daemon.submit(req(tmp_path, [PY, "-c", "pass"], session="s2"))  # too few to say


def _borrowing(make_daemon, tmp_path, monkeypatch, *, ran, pressure=0.0, config=None, now=None, retry=True,
               meminfo=None):  # fmt: skip
    """Job A reserves 12G of 16G and has used 2G after `ran` seconds; job B then asks for 8G."""
    from dibs.machine import Config

    now = now or [time.time()]
    d = make_daemon(config=config or Config(), clock=lambda: now[0], meminfo=meminfo or (lambda: (16 * GiB, 2 * GiB)),
                    pressure=pressure if callable(pressure) else (lambda: pressure))  # fmt: skip
    a, stat, go = _job_holding(d, tmp_path, monkeypatch, 12 * GiB)
    stat.write_text(f"anon {2 * GiB}\nshmem 0\n")
    now[0] += 1
    d.tick()
    now[0] += ran
    d.tick()
    b = submit(d, tmp_path, [PY, "-c", "import time; time.sleep(3)"], mem=8 * GiB, retry=retry)
    d.tick()
    return d, a, b, go


def test_ram_a_running_job_does_not_use_can_go_to_another_job(make_daemon, tmp_path, monkeypatch):
    # Issue #86: agents reserve 2-4x the RAM their jobs use. After ten minutes, a job counts at its
    # peak plus headroom (2G plus 1G here), so a job that fits in the rest starts, marked as borrowing.
    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=700)
    assert job(d, b).state == State.RUNNING and job(d, b).borrowed_ram
    assert not job(d, a).borrowed_ram
    go.touch()


def test_a_job_without_retry_starts_on_borrowed_ram_too(make_daemon, tmp_path, monkeypatch):
    # Issue #148: only --retry jobs could borrow, 2 of 96 jobs in an hour used it, and idle
    # reservations held back hours of waiting. Any job may borrow now; --retry decides what follows a stop.
    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=700, retry=False)
    assert job(d, b).state == State.RUNNING and job(d, b).borrowed_ram
    go.touch()


NOT_RETRIED = "; resubmit it, or submit it with --retry so dibs runs it again by itself"


def test_a_borrower_without_retry_stopped_to_give_ram_back_fails_saying_how_to_avoid_it(make_daemon, tmp_path,
                                                                                      monkeypatch):  # fmt: skip
    level, used, now = [0.0], [2 * GiB], [time.time()]
    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=700, pressure=lambda: level[0], now=now,
                             meminfo=lambda: (16 * GiB, used[0]), retry=False)  # fmt: skip
    level[0], used[0] = 60.0, 15 * GiB
    for _ in range(3):
        d.tick()
        now[0] += 6.0
    run_until(d, lambda: finished(d, b))
    j = job(d, b)
    assert (j.state, j.attempts) == (State.FAILED, [])
    assert j.reason == "stopped to give back borrowed RAM: the machine was short of memory" + NOT_RETRIED
    go.touch()


def test_a_borrower_without_retry_killed_when_the_machine_ran_out_fails_saying_how_to_avoid_it(make_daemon, tmp_path,
                                                                                            monkeypatch):  # fmt: skip
    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=700, retry=False)
    real_read_exit = launcher.read_exit

    def read_exit(paths, i):
        r = real_read_exit(paths, i)
        return {**r, "exit_code": 137, "oom_killed": True, "hit_limit": False} if r and i == b else r

    monkeypatch.setattr(launcher, "read_exit", read_exit)
    run_until(d, lambda: finished(d, b))
    j = job(d, b)
    assert j.state == State.FAILED
    assert j.reason == "killed when the machine ran out of RAM, while it used RAM other jobs had reserved" + NOT_RETRIED
    go.touch()


def test_a_young_job_without_history_counts_at_its_whole_reservation(make_daemon, tmp_path, monkeypatch):
    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=300)  # it may still be growing
    assert job(d, b).state == State.QUEUED
    go.touch()


def test_no_borrowing_under_memory_pressure_or_when_switched_off(make_daemon, tmp_path, monkeypatch):
    from dibs.machine import Config

    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=700, pressure=20.0)
    assert job(d, b).state == State.QUEUED
    go.touch()


def test_borrowing_can_be_switched_off(make_daemon, tmp_path, monkeypatch):
    from dibs.machine import Config

    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=700, config=Config(borrow_ram=False))
    assert job(d, b).state == State.QUEUED
    go.touch()


def test_a_borrower_killed_when_the_machine_ran_out_is_retried_without_borrowing(make_daemon, tmp_path, monkeypatch):
    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=700)
    real_read_exit = launcher.read_exit

    def read_exit(paths, i):
        r = real_read_exit(paths, i)
        return {**r, "exit_code": 137, "oom_killed": True, "hit_limit": False} if r and i == b else r

    monkeypatch.setattr(launcher, "read_exit", read_exit)
    run_until(d, lambda: job(d, b).attempts)
    j = job(d, b)
    assert (j.state, j.no_borrow, j.request.mem) == (State.QUEUED, True, 8 * GiB)
    assert j.attempts[0]["reason"] == (
        "killed when the machine ran out of RAM, while it used RAM other jobs had reserved; retrying without borrowing"
    )
    d.tick()
    assert job(d, b).state == State.QUEUED  # 8G doesn't fit beside A's 12G without borrowing
    go.touch()


def test_a_borrower_killed_for_borrowing_is_retried_even_out_of_retries(make_daemon, tmp_path, monkeypatch):
    # Issue #114: dibs chose to start it on borrowed RAM, so the kill isn't the job's to pay for.
    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=700)
    j = job(d, b)
    j.attempts = [{"reason": "an earlier try"}, {"reason": "another"}]
    d.store.update_job(j)
    real_read_exit = launcher.read_exit

    def read_exit(paths, i):
        r = real_read_exit(paths, i)
        return {**r, "exit_code": 137, "oom_killed": True, "hit_limit": False} if r and i == b else r

    monkeypatch.setattr(launcher, "read_exit", read_exit)
    run_until(d, lambda: len(job(d, b).attempts) == 3)
    assert (job(d, b).state, job(d, b).no_borrow) == (State.QUEUED, True)
    go.touch()


def test_under_memory_pressure_dibs_gives_borrowed_ram_back(make_daemon, tmp_path, monkeypatch):
    # Issue #114: the machine thrashed for five minutes before the kernel's OOM killer acted. Under
    # sustained pressure dibs stops the newest borrower itself and runs it again without borrowing.
    level, used, now = [0.0], [2 * GiB], [time.time()]
    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=700, pressure=lambda: level[0], now=now,
                             meminfo=lambda: (16 * GiB, used[0]))  # fmt: skip
    assert job(d, b).borrowed_ram
    level[0], used[0] = 60.0, 15 * GiB  # 1G of 16G left
    d.tick()
    now[0] += 5.0
    d.tick()
    assert job(d, b).stop_requested is None  # not on a moment's pressure
    now[0] += 6.0
    d.tick()
    assert job(d, b).stop_reason == "stopped to give back borrowed RAM: the machine was short of memory"
    run_until(d, lambda: job(d, b).attempts)
    j = job(d, b)
    assert (j.state, j.no_borrow) == (State.QUEUED, True)
    assert j.attempts[0]["reason"] == (
        "stopped to give back borrowed RAM: the machine was short of memory; retrying without borrowing"
    )
    assert not job(d, a).stop_requested  # the job that reserved the RAM keeps running
    go.touch()


def test_memory_pressure_with_most_of_the_ram_free_stops_no_borrower(make_daemon, tmp_path, monkeypatch):
    # Issue #129: pressure stayed at 50-60% for minutes with half the RAM free, from one job stalling
    # inside itself (most likely the kernel compacting memory for huge pages). Stopping a borrower
    # then would free RAM nobody needed and throw its work away.
    level, now = [0.0], [time.time()]
    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=700, pressure=lambda: level[0], now=now)
    assert job(d, b).borrowed_ram
    level[0] = 60.0  # with 14G of 16G available
    for _ in range(5):
        d.tick()
        now[0] += 6.0
    assert job(d, b).stop_requested is None
    go.touch()


def test_submit_reports_earlier_runs_of_the_same_command(daemon, tmp_path):
    # Issue #31: agents guess their requests; dibs tells them what the same command used before.
    first = submit(daemon, tmp_path, [PY, "-c", "pass"])
    run_until(daemon, lambda: finished(daemon, first))
    again = daemon.submit(req(tmp_path, [PY, "-c", "pass"]))
    assert [e["id"] for e in again["earlier"]] == [first]
    assert daemon.submit(req(tmp_path, [PY, "-c", "print(1)"]))["earlier"] == []


def test_earlier_runs_render_what_they_used():
    from dibs.render import render_earlier

    gib = 1 << 30
    earlier = [
        {"id": 43, "state": "failed", "started": 0.0, "ended": 10.0, "cpu_seconds": None,
         "peak_mem": int(2.7 * gib), "peak_vram": int(4.4 * gib),
         "request": {"cpus": 6, "mem": 6 * gib, "gpus": 1, "vram": 4 * gib, "time": 1200.0},
         "reason": "stopped: used 4.4G VRAM on GPU 0, reserved 4G"},
        {"id": 41, "state": "done", "started": 0.0, "ended": 1140.0, "cpu_seconds": 3306.0,
         "peak_mem": int(2.8 * gib), "peak_vram": int(4.1 * gib),
         "request": {"cpus": 6, "mem": 6 * gib, "gpus": 1, "vram": 5 * gib, "time": 2400.0}, "reason": None},
    ]  # fmt: skip
    assert render_earlier(earlier) == (
        "Earlier runs of this command here:\n"
        "  #43 failed after 10s (estimated 20m): RAM 2.7G of 6G, VRAM 4.4G of 4G; stopped: used 4.4G VRAM on GPU 0, reserved 4G\n"
        "  #41 done after 19m (estimated 40m): 2.9 of 6 CPUs, RAM 2.8G of 6G, VRAM 4.1G of 5G"
    )
    assert render_earlier([]) == ""


def test_job_on_an_amd_machine_sees_only_its_gpus_through_rocr(make_daemon, tmp_path):
    class AmdBackend(FakeBackend):
        vendor = "amd"

    amd_daemon = make_daemon(backend=AmdBackend())
    script = "import os; print('ROCR=' + os.environ['ROCR_VISIBLE_DEVICES'], 'CUDA' in str(os.environ.keys()))"
    job_id = submit(amd_daemon, tmp_path, [PY, "-c", script], vram=1 * GiB)
    run_until(amd_daemon, lambda: finished(amd_daemon, job_id))
    assert "ROCR=0 False" in open(job(amd_daemon, job_id).output).read()


def test_priority_goes_ahead_and_the_sessions_next_job_pays_it_back(make_daemon, tmp_path):
    # Issue #47: a free priority would have every agent pick 10. So a job's priority moves it in
    # line, and its session's next job pays back the time it saved.
    now = [1000.0]
    d = make_daemon(clock=lambda: now[0], cap=capacity(cpus=[0]))
    submit(d, tmp_path, [PY, "-c", "import time; time.sleep(30)"], session="blocker")  # holds the CPU
    urgent = submit(d, tmp_path, [PY, "-c", "pass"], time_=1200.0, session="s1", priority=10)
    other = submit(d, tmp_path, [PY, "-c", "pass"], time_=300.0, session="s2")
    after = submit(d, tmp_path, [PY, "-c", "pass"], time_=300.0, session="s1")
    assert job(d, urgent).tag == 1000.0 + 120.0  # a tenth of its 20 minutes
    assert job(d, other).tag == 1000.0 + 300.0  # so the urgent job goes ahead of a shorter one
    assert job(d, after).tag == 1000.0 + 1200.0 + 300.0 + 1080.0  # its place without priority, plus the time saved
    assert job(d, urgent).priority == 10 and job(d, after).priority == 5


def test_priority_never_passes_a_job_that_waited_its_estimate(make_daemon, tmp_path):
    now = [1000.0]
    d = make_daemon(clock=lambda: now[0], cap=capacity(cpus=[0]))
    submit(d, tmp_path, [PY, "-c", "import time; time.sleep(30)"], session="blocker")
    waited = submit(d, tmp_path, [PY, "-c", "pass"], time_=60.0, session="s1")
    now[0] += 120.0  # it has now waited twice its estimate
    late = submit(d, tmp_path, [PY, "-c", "pass"], time_=600.0, session="s2", priority=10)
    assert job(d, late).tag >= job(d, waited).tag


def test_priority_must_be_one_to_ten(daemon, tmp_path):
    for bad in (0, 11, "high", 2.5, True):
        resp = daemon.submit(req(tmp_path, [PY, "-c", "pass"], priority=bad))
        assert not resp["ok"] and "1 to 10" in resp["error"]


def test_what_a_session_owes_lapses_after_a_day(make_daemon, tmp_path):
    now = [1000.0]
    d = make_daemon(clock=lambda: now[0], cap=capacity(cpus=[0]))
    submit(d, tmp_path, [PY, "-c", "import time; time.sleep(30)"], session="blocker")
    submit(d, tmp_path, [PY, "-c", "pass"], time_=1200.0, session="s1", priority=10)
    now[0] += 86400 + 1
    later = submit(d, tmp_path, [PY, "-c", "pass"], time_=300.0, session="s1")
    assert job(d, later).tag == now[0] + 300.0


def test_cleanup_forgets_what_sessions_owed_more_than_a_day_ago(daemon):
    now = time.time()
    daemon.store.set_meta("owed:old", json.dumps({"owed": 600.0, "time": now - 86400 - 60}))
    daemon.store.set_meta("owed:new", json.dumps({"owed": 600.0, "time": now - 60}))
    daemon._retention(now)
    assert daemon.store.get_meta("owed:old") is None and daemon.store.get_meta("owed:new") is not None


def test_a_batch_of_high_priority_jobs_pays_back_after_the_first(make_daemon, tmp_path):
    # Issue #56: the session's jobs lined up behind the previous job's boosted tag, so a batch of
    # priority-10 jobs kept every boost. They must line up behind each job's priority-5 place.
    now = [1000.0]
    d = make_daemon(clock=lambda: now[0], cap=capacity(cpus=[0]))
    submit(d, tmp_path, [PY, "-c", "import time; time.sleep(30)"], session="blocker")
    batch = [submit(d, tmp_path, [PY, "-c", "pass"], time_=300.0, session="s1", priority=10) for _ in range(3)]
    assert [job(d, j).tag for j in batch] == [1030.0, 1600.0, 1900.0]  # priority 5: 1300, 1600, 1900


def test_the_job_after_a_boosted_one_lands_later_by_the_time_saved(make_daemon, tmp_path):
    now = [1000.0]
    d = make_daemon(clock=lambda: now[0], cap=capacity(cpus=[0]))
    submit(d, tmp_path, [PY, "-c", "import time; time.sleep(30)"], session="blocker")
    submit(d, tmp_path, [PY, "-c", "pass"], time_=300.0, session="s1", priority=10)
    second = submit(d, tmp_path, [PY, "-c", "pass"], time_=300.0, session="s1")
    assert job(d, second).tag == 1600.0 + 270.0  # its priority-5 place, plus the 270 s the first saved


def test_a_higher_priority_passes_its_own_sessions_lower_priority_jobs(make_daemon, tmp_path):
    # Issue #91: a job resubmitted at priority 7 still lined up behind its own session's queued
    # 2.5-hour job. Reordering one's own jobs is zero-sum: the jobs it passes move back by its estimate.
    now = [1000.0]
    d = make_daemon(clock=lambda: now[0], cap=capacity(cpus=[0]))
    submit(d, tmp_path, ["sleep", "30"], session="blocker")
    long = submit(d, tmp_path, [PY, "-c", "pass"], time_=9000.0, session="s1")
    urgent = submit(d, tmp_path, [PY, "-c", "pass"], time_=600.0, session="s1", priority=7)
    assert job(d, urgent).plain_tag == 1000.0 + 600.0  # no longer behind its own long job
    assert job(d, urgent).tag < job(d, long).tag
    assert (job(d, long).plain_tag, job(d, long).tag) == (10600.0, 10600.0)  # moved back by 600 s
    same = submit(d, tmp_path, [PY, "-c", "pass"], time_=60.0, session="s1", priority=5)
    assert job(d, same).plain_tag == 10600.0 + 60.0  # equal priority still lines up behind


def test_a_boost_that_passed_nobody_costs_nothing(make_daemon, tmp_path):
    # Issue #91: with nobody else waiting, the boost moved nothing, so the next job owes nothing.
    now = [1000.0]
    d = make_daemon(clock=lambda: now[0])
    urgent = submit(d, tmp_path, [PY, "-c", "pass"], time_=600.0, session="s1", priority=10)
    run_until(d, lambda: finished(d, urgent))
    after = submit(d, tmp_path, [PY, "-c", "pass"], time_=300.0, session="s1")
    assert job(d, after).tag == 1000.0 + 300.0


def test_a_throwaway_low_priority_job_cancelled_at_once_earns_no_credit(make_daemon, tmp_path):
    # Issue #90: a priority-1 job cancelled at once gave the session's next job an hour's head start.
    now = [1000.0]
    d = make_daemon(clock=lambda: now[0], cap=capacity(cpus=[0]))
    submit(d, tmp_path, ["sleep", "30"], session="other")
    dummy = submit(d, tmp_path, [PY, "-c", "pass"], time_=1200.0, session="gamer", priority=1)
    d.cancel([dummy])
    real = submit(d, tmp_path, [PY, "-c", "pass"], time_=1200.0, session="gamer")
    assert job(d, real).tag == 1000.0 + 1200.0


def test_a_throwaway_job_that_starts_at_once_does_not_absorb_a_debt(make_daemon, tmp_path):
    # Issue #90: after a priority-10 job, a 1-second job took its debt and started at once on a free
    # CPU, so the debt was never paid. Debt is paid by waiting; what isn't paid rolls on.
    now = [1000.0]
    d = make_daemon(clock=lambda: now[0], cap=capacity(cpus=[0, 1]))
    submit(d, tmp_path, ["sleep", "30"], session="other")  # holds one CPU
    submit(d, tmp_path, [PY, "-c", "pass"], cpus=2, time_=1200.0, session="other")  # waits for both
    submit(d, tmp_path, [PY, "-c", "pass"], cpus=2, time_=7200.0, session="g2", priority=10)
    absorber = submit(d, tmp_path, [PY, "-c", "pass"], cpus=1, time_=1.0, session="g2")
    run_until(d, lambda: finished(d, absorber))
    nxt = submit(d, tmp_path, [PY, "-c", "pass"], cpus=2, time_=1200.0, session="g2")
    assert job(d, nxt).tag - job(d, nxt).plain_tag == 6480.0  # the 108 min the boost saved


def test_a_lowered_priority_earns_credit_only_for_time_it_waited(make_daemon, tmp_path):
    # Issue #90: a priority-3 job that started at once gave up nothing, so it earns nothing.
    now = [1000.0]
    d = make_daemon(clock=lambda: now[0])
    low = submit(d, tmp_path, [PY, "-c", "pass"], time_=1800.0, session="s1", priority=3)
    run_until(d, lambda: finished(d, low))
    nxt = submit(d, tmp_path, [PY, "-c", "pass"], time_=300.0, session="s1")
    assert job(d, nxt).tag == 1000.0 + 300.0


def test_submit_falls_back_to_earlier_runs_of_the_same_script(daemon, tmp_path):
    (tmp_path / "train.py").write_text("import sys\n")
    first = submit(daemon, tmp_path, [PY, "train.py", "--lr", "1"])
    run_until(daemon, lambda: finished(daemon, first))
    again = daemon.submit(req(tmp_path, [PY, "train.py", "--lr", "2"]))
    assert [e["id"] for e in again["earlier"]] == [first] and again["earlier_match"] == "script"
    same = daemon.submit(req(tmp_path, [PY, "train.py", "--lr", "1"]))
    assert same["earlier_match"] == "command"


def test_earlier_runs_of_the_script_say_so():
    from dibs.render import render_earlier

    run = {"id": 7, "state": "done", "started": 0.0, "ended": 60.0, "cpu_seconds": None, "peak_mem": None,
           "peak_vram": None, "reason": None, "argv": ["python", "train.py", "--lr", "1"],
           "request": {"cpus": 1, "mem": 1 << 30, "gpus": 0, "vram": 0, "time": 120.0}}  # fmt: skip
    assert render_earlier([run], "script").startswith("Earlier runs of train.py here, with other arguments:")
    assert render_earlier([run]).startswith("Earlier runs of this command here:")


def test_render_estimate_feedback():
    from dibs.render import render_estimates

    assert render_estimates({"jobs": 20, "median": 0.11}) == (
        "Your session's last 20 jobs ran a median 11% of their estimates: a closer --time gets jobs started sooner."
    )
    assert render_estimates({"jobs": 8, "median": 1.6}) == (
        "Your session's last 8 jobs ran a median 160% of their estimates: a longer --time keeps predictions right."
    )
    assert render_estimates({"jobs": 20, "median": 0.9}) == "" and render_estimates(None) == ""


def test_momentum_moves_a_held_up_sessions_jobs_ahead_after_the_users_yes(make_daemon, tmp_path):
    # Issue #94: for a project doing better than expected and held up by another project's jobs; the
    # agent asks the user in plain words and records the grant only after the user's yes.
    now = [1000.0]
    d = make_daemon(clock=lambda: now[0], cap=capacity(cpus=[0]))
    submit(d, tmp_path, ["sleep", "30"], session="other")  # another project holds the CPU
    theirs = submit(d, tmp_path, [PY, "-c", "pass"], time_=600.0, session="other")
    mine = submit(d, tmp_path, [PY, "-c", "pass"], time_=1200.0, session="s1")
    assert job(d, mine).tag > job(d, theirs).tag
    grant = {"session": "s1", "session_label": "proj·s1", "action": "grant", "seconds": 7200.0,
             "why": "results beat the baseline and the user said yes"}  # fmt: skip
    resp = d.momentum(grant)
    assert resp["ok"], resp
    assert job(d, mine).tag == 1000.0 + 300.0 < job(d, theirs).tag  # a quarter of its estimate now
    later = submit(d, tmp_path, [PY, "-c", "pass"], time_=400.0, session="s1")
    assert job(d, later).tag == 1000.0 + 300.0 + 100.0  # behind it, at a quarter, with no payback
    assert d.snapshot()["momentum"]["label"] == "proj·s1"


def test_momentum_is_only_for_a_session_another_project_holds_up(make_daemon, tmp_path):
    now = [1000.0]
    d = make_daemon(clock=lambda: now[0], cap=capacity(cpus=[0]))
    ask = {"session": "s1", "action": "grant", "seconds": 3600.0, "why": "going well"}
    resp = d.momentum(ask)
    assert not resp["ok"] and "nothing of yours is waiting" in resp["error"]
    submit(d, tmp_path, ["sleep", "30"], session="s1")  # its own job holds the CPU
    submit(d, tmp_path, [PY, "-c", "pass"], time_=60.0, session="s1")
    resp = d.momentum(ask)
    assert not resp["ok"] and "only your own jobs" in resp["error"]
    assert not d.momentum({**ask, "why": " "})["ok"]


def test_one_project_has_momentum_at_a_time_and_it_runs_out(make_daemon, tmp_path):
    now = [1000.0]
    d = make_daemon(clock=lambda: now[0], cap=capacity(cpus=[0]))
    submit(d, tmp_path, ["sleep", "30"], session="other")
    submit(d, tmp_path, [PY, "-c", "pass"], time_=60.0, session="s1")
    submit(d, tmp_path, [PY, "-c", "pass"], time_=60.0, session="s2")
    assert d.momentum({"session": "s1", "action": "grant", "seconds": 3600.0, "why": "x"})["ok"]
    resp = d.momentum({"session": "s2", "action": "grant", "seconds": 3600.0, "why": "y"})
    assert not resp["ok"] and "already has momentum" in resp["error"]
    now[0] += 3601
    assert d.snapshot()["momentum"] is None
    after = submit(d, tmp_path, [PY, "-c", "pass"], time_=400.0, session="s1")
    assert job(d, after).tag - now[0] >= 400.0  # counted in full again
    assert d.momentum({"session": "s2", "action": "grant", "seconds": 3600.0, "why": "y"})["ok"]
    assert d.momentum({"action": "end"})["momentum"] is None and d.snapshot()["momentum"] is None


def test_dibs_show_says_how_busy_the_machine_was_beside_a_timing_run(make_daemon, tmp_path, monkeypatch):
    # Issue #112: nothing else from dibs runs beside an exclusive job, but other programs can still
    # disturb its timing. dibs samples the CPU in use outside it and dibs show reports it.
    from dibs.render import render_show

    now, busy = [time.time()], [0.0]
    d = make_daemon(clock=lambda: now[0], cpu_busy=lambda: busy[0])
    job_id, stat, go = _job_holding(d, tmp_path, monkeypatch, GiB, exclusive=True)
    cpu = stat.parent / "cpu.stat"
    used = 0.0
    for others in (0.5, 1.5, 0.5, 0.5):  # cores other programs keep busy over each 5 s
        cpu.write_text(f"usage_usec {int(used * 1e6)}\n")
        d.tick()
        now[0] += 5.0
        used += 2 * 5.0  # the job itself keeps 2 cores busy
        busy[0] += (2 + others) * 5.0
    cpu.write_text(f"usage_usec {int(used * 1e6)}\n")
    d.tick()
    j = job(d, job_id)
    assert (round(j.background["mean"], 2), round(j.background["max"], 2)) == (0.75, 1.5)
    text = render_show(d.show(job_id)["job"], now[0])
    assert re.search(r"beside it +other programs kept 0.8 cores busy on average, at most 1.5", text), text
    go.touch()


def _busy_outside(d, now, busy, cpu, *, job_cores, outside_cores, steps):
    """Advance 5 s at a time: the job keeps job_cores busy, programs outside dibs outside_cores."""
    for _ in range(steps):
        now[0] += 5.0
        busy[0] += (job_cores + outside_cores) * 5.0
        if cpu is not None:
            used = float(cpu.read_text().split()[1]) / 1e6 if cpu.exists() else 0.0
            cpu.write_text(f"usage_usec {int((used + job_cores * 5.0) * 1e6)}\n")
        d.tick()


def test_snapshot_says_how_many_cores_are_busy_outside_dibs(make_daemon, tmp_path, monkeypatch):
    # Issue #120: a program outside dibs kept most cores busy while dibs top showed 7 of 24 reserved.
    now, busy = [time.time()], [0.0]
    d = make_daemon(clock=lambda: now[0], cpu_busy=lambda: busy[0])
    job_id, stat, go = _job_holding(d, tmp_path, monkeypatch, GiB)
    cpu = stat.parent / "cpu.stat"
    cpu.write_text("usage_usec 0\n")
    assert d.snapshot()["outside"]["cpus"] is None  # not measured yet
    _busy_outside(d, now, busy, cpu, job_cores=2, outside_cores=12, steps=4)
    assert d.snapshot()["outside"]["cpus"] == pytest.approx(12.0)
    go.touch()


def test_cores_busy_outside_dibs_are_unknown_when_a_jobs_own_use_cant_be_read(make_daemon, tmp_path, monkeypatch):
    # Never count a job's own work as outside dibs.
    now, busy = [time.time()], [0.0]
    d = make_daemon(clock=lambda: now[0], cpu_busy=lambda: busy[0])
    job_id, stat, go = _job_holding(d, tmp_path, monkeypatch, GiB)  # no cpu.stat to read
    _busy_outside(d, now, busy, None, job_cores=2, outside_cores=1, steps=4)
    assert d.snapshot()["outside"]["cpus"] is None
    go.touch()


def test_cores_busy_outside_dibs_restart_when_a_job_ends_or_the_machine_sleeps(make_daemon, tmp_path, monkeypatch):
    now, busy, slept = [time.time()], [0.0], [0.0]
    d = make_daemon(clock=lambda: now[0], cpu_busy=lambda: busy[0], slept=lambda: slept[0])
    job_id, stat, go = _job_holding(d, tmp_path, monkeypatch, GiB)
    cpu = stat.parent / "cpu.stat"
    cpu.write_text("usage_usec 0\n")
    _busy_outside(d, now, busy, cpu, job_cores=2, outside_cores=4, steps=3)
    assert d.snapshot()["outside"]["cpus"] == pytest.approx(4.0)
    go.touch()
    run_until(d, lambda: job(d, job_id).state == State.DONE)
    _busy_outside(d, now, busy, None, job_cores=0, outside_cores=3, steps=1)
    assert d.snapshot()["outside"]["cpus"] is None  # the job's last seconds aren't known
    _busy_outside(d, now, busy, None, job_cores=0, outside_cores=3, steps=1)
    assert d.snapshot()["outside"]["cpus"] == pytest.approx(3.0)
    slept[0] += 600.0  # the machine slept
    _busy_outside(d, now, busy, None, job_cores=0, outside_cores=3, steps=1)
    assert d.snapshot()["outside"]["cpus"] is None


def test_snapshot_shows_the_ram_in_use_outside_dibs_that_waiting_jobs_are_told(make_daemon, tmp_path):
    # The RAM bar draws the same figure the planner subtracts and the reasons quote.
    d = make_daemon(meminfo=lambda: (16 * GiB, 10 * GiB))
    big = submit(d, tmp_path, [PY, "-c", "pass"], mem=8 * GiB)
    d.tick()
    snap = d.snapshot()
    assert snap["outside"]["mem"] == 10 * GiB
    assert "10G in use outside dibs" in next(j for j in snap["jobs"] if j["id"] == big)["waiting_for"]


def test_snapshot_keeps_ten_minutes_of_the_machines_busy_cores(make_daemon):
    # Issue #122: dibs top's load row. Its own samples, so they survive jobs starting and ending.
    now, busy, slept = [time.time()], [0.0], [0.0]
    d = make_daemon(clock=lambda: now[0], cpu_busy=lambda: busy[0], slept=lambda: slept[0])
    assert d.snapshot()["load"] == {"every": 4.0, "cores": []}
    for cores in (2, 6, 12):
        d.tick()
        now[0] += 4.0
        busy[0] += cores * 4.0
    d.tick()
    assert d.snapshot()["load"]["cores"] == pytest.approx([2, 6, 12])
    slept[0] += 300.0  # the machine slept: that interval is a gap, not an idle machine
    now[0] += 304.0
    busy[0] += 4.0
    d.tick()
    assert d.snapshot()["load"]["cores"][-1] is None
    for _ in range(200):  # only the last 10 minutes are kept
        now[0] += 4.0
        busy[0] += 4.0
        d.tick()
    assert len(d.snapshot()["load"]["cores"]) == 150


def test_the_load_row_never_gets_in_the_way_of_scheduling(make_daemon, tmp_path):
    def broken():
        raise OSError("no /proc/stat")

    d = make_daemon(cpu_busy=broken)
    job_id = submit(d, tmp_path, [PY, "-c", "pass"])
    run_until(d, lambda: job(d, job_id).state == State.DONE)
    assert d.snapshot()["load"]["cores"] == []


LONG = [PY, "-c", "import time; time.sleep(600)"]


def _machine_held(daemon, tmp_path):
    """A job of another session that holds every CPU, and its id: the jobs after it wait."""
    return submit(daemon, tmp_path, LONG, cpus=len(daemon.capacity.cpus), time_=3600.0, session="other")


def test_a_job_lines_up_only_behind_its_sessions_jobs_still_waiting(daemon, tmp_path):
    # Issue #150: places never moved when the jobs ahead of them started, so a busy session's line ran
    # hours ahead of the clock: a job submitted at 10:43 had its place at 21:22, and every newer job of
    # another session went ahead of it.
    all_cpus = len(daemon.capacity.cpus)
    hold = _machine_held(daemon, tmp_path)
    a = submit(daemon, tmp_path, LONG, cpus=all_cpus, time_=3600.0)
    b = submit(daemon, tmp_path, LONG, cpus=all_cpus, time_=3600.0)
    assert job(daemon, b).tag == pytest.approx(job(daemon, a).tag + 3600, abs=1)  # B after A
    daemon.cancel([hold])
    run_until(daemon, lambda: job(daemon, a).state == State.RUNNING)
    b_now = job(daemon, b)
    assert b_now.tag == pytest.approx(b_now.submitted + 3600, abs=1)  # A has started: B is due an hour after it came
    daemon.cancel([a, b])


def test_cancelling_a_waiting_job_moves_the_ones_behind_it_forward(daemon, tmp_path):
    hold = _machine_held(daemon, tmp_path)
    a = submit(daemon, tmp_path, LONG, time_=7200.0)
    b = submit(daemon, tmp_path, LONG, time_=600.0)
    before = job(daemon, b).tag
    daemon.cancel([a])
    after = job(daemon, b)
    assert after.tag == pytest.approx(after.submitted + 600, abs=1) and after.tag < before
    daemon.cancel([hold, b])


def test_lining_up_again_keeps_a_higher_priority_job_ahead_of_its_sessions_others(daemon, tmp_path):
    hold = _machine_held(daemon, tmp_path)
    first = submit(daemon, tmp_path, LONG, time_=600.0)
    low = submit(daemon, tmp_path, LONG, time_=600.0)
    high = submit(daemon, tmp_path, LONG, time_=600.0, priority=8)  # goes ahead of low, which moves back
    assert job(daemon, high).tag < job(daemon, low).tag
    daemon.cancel([first])
    assert job(daemon, high).tag < job(daemon, low).tag
    assert all(job(daemon, j).tag <= job(daemon, j).submitted + 1200 + 1 for j in (high, low))
    daemon.cancel([hold, low, high])


def test_a_restart_lines_every_session_up_again(daemon, tmp_path):
    # Issue #150 relines a session when one of its waiting jobs starts or is cancelled; places set by
    # an older dibs stay inflated until then, so recovery relines every session once.
    hold = _machine_held(daemon, tmp_path)
    b = submit(daemon, tmp_path, LONG, time_=600.0)
    j = job(daemon, b)
    j.tag = j.plain_tag = j.submitted + 24 * 3600.0  # a place an older dibs gave it, a day out
    daemon.store.update_job(j)
    daemon.recover()
    j = job(daemon, b)
    assert j.tag == pytest.approx(j.submitted + 600, abs=1)
    daemon.cancel([hold, b])


def _growing(make_daemon, tmp_path, monkeypatch, *, mem=8 * GiB, ok=True, setter=True, retry=False):
    """A running job reserving `mem` of a 16G machine, 1G of it in use outside dibs, and the limits dibs
    gives it (recorded, never set for real: the jobs here have no scopes)."""
    state, limits = {"held": 1 * GiB}, []

    def set_limit(j, limit):
        limits.append((j.id, limit))
        return ok

    d = make_daemon(meminfo=lambda: (16 * GiB, state["held"] + 1 * GiB), set_mem_limit=set_limit if setter else None)
    a, stat, go = _job_holding(d, tmp_path, monkeypatch, mem, retry=retry)
    run_until(d, lambda: launcher.job_pids(job(d, a)))  # its command runs: a job near its limit has processes

    def hold(n):
        state["held"] = n
        stat.write_text(f"anon {n}\nfile {n}\nshmem 0\nfile_mapped 0\n")
        d.tick()

    return d, a, hold, limits, go


def test_a_job_near_its_ram_limit_grows_into_ram_nobody_reserved(make_daemon, tmp_path, monkeypatch):
    # Issue #155: at all 64 kills at a job's own limit, 2.7-17.7G was unreserved and free; one job lost 2
    # hours at its very end. A job may grow past its reservation, a step at a time, tracked.
    d, a, hold, limits, go = _growing(make_daemon, tmp_path, monkeypatch)
    hold(6 * GiB)
    assert limits == []  # not near its limit yet
    hold(int(7.2 * GiB))  # within half a step (a quarter of 8G: 2G) of its 8G limit
    assert limits == [(a, 10 * GiB)] and job(d, a).ram_limit == 10 * GiB
    # Tracked: the planner counts it at 10G, so a job that fits beside 8G but not 10G waits.
    b = submit(d, tmp_path, [PY, "-c", "pass"], mem=6 * GiB)
    d.tick()
    assert job(d, b).state == State.QUEUED
    assert open(f"/proc/{launcher.job_pids(job(d, a))[0]}/oom_score_adj").read().strip() == str(launcher.BORROWER_OOM_SCORE)
    from dibs.render import render_show

    assert "grown to a 10G limit" in render_show(d.show(a)["job"], d.clock())
    go.touch()


def test_a_job_never_grows_into_ram_another_job_reserved(make_daemon, tmp_path, monkeypatch):
    # Measured against reservations, not use: another job could grow back into RAM it reserved.
    d, a, hold, limits, go = _growing(make_daemon, tmp_path, monkeypatch)
    # 16G, less 2G outside dibs (its peak over the last 2 minutes), 8G and 6G reserved: nothing left.
    c = submit(d, tmp_path, [PY, "-c", "import time; time.sleep(30)"], mem=6 * GiB)
    run_until(d, lambda: job(d, c).state == State.RUNNING)
    hold(int(7.2 * GiB))
    assert limits == [] and job(d, a).ram_limit is None
    go.touch()
    d.cancel([c])


@pytest.mark.parametrize("ok,setter", [(False, True), (True, False)])
def test_a_job_grows_only_when_its_limit_can_really_be_raised(make_daemon, tmp_path, monkeypatch, ok, setter):
    d, a, hold, limits, go = _growing(make_daemon, tmp_path, monkeypatch, ok=ok, setter=setter)
    hold(int(7.2 * GiB))
    assert job(d, a).ram_limit is None
    go.touch()


def test_a_job_killed_at_a_raised_limit_says_so_and_what_to_ask_for(make_daemon, tmp_path, monkeypatch):
    d, a, hold, limits, go = _growing(make_daemon, tmp_path, monkeypatch)
    hold(int(7.2 * GiB))
    real_read_exit = launcher.read_exit
    monkeypatch.setattr(
        launcher, "read_exit",
        lambda paths, i: (r := real_read_exit(paths, i)) and {**r, "exit_code": 137, "oom_killed": True, "hit_limit": True},
    )  # fmt: skip
    go.touch()
    run_until(d, lambda: finished(d, a))
    j = job(d, a)
    assert j.peak_mem == 10 * GiB
    assert j.reason == ("killed: went over its RAM limit of 10G (8G reserved, the rest grown into RAM nobody reserved); "
                        "resubmit with --mem 13G")  # fmt: skip


def test_a_retried_job_starts_again_from_its_reservation(make_daemon, tmp_path, monkeypatch):
    d, a, hold, limits, go = _growing(make_daemon, tmp_path, monkeypatch, retry=True)
    hold(int(7.2 * GiB))
    _oom_on_first_exit(monkeypatch, a)
    go.touch()
    run_until(d, lambda: job(d, a).attempts)
    j = job(d, a)
    assert (j.request.mem, j.ram_limit) == (13 * GiB, None)


@pytest.mark.parametrize(
    "mem,held,grows",
    [
        (int(1.5 * GiB), int(1.2 * GiB), False),  # a small job's usual peak is no reason to grow
        (700 * MiB, 300 * MiB, False),
        (1 * GiB, int(0.95 * GiB), True),  # within an eighth of its limit
        (8 * GiB, int(7.2 * GiB), True),  # within half a step (1G) of its limit
    ],
)
def test_a_job_grows_only_once_it_is_close_to_its_own_limit(make_daemon, tmp_path, monkeypatch, mem, held, grows):
    d, a, hold, limits, go = _growing(make_daemon, tmp_path, monkeypatch, mem=mem)
    hold(held)
    assert bool(limits) == grows, limits
    go.touch()


def test_a_grown_job_killed_when_the_machine_ran_out_says_so_and_what_to_ask_for(make_daemon, tmp_path, monkeypatch):
    # A grown job has the borrowers' OOM score, so it's the kernel's first pick when the machine runs out.
    d, a, hold, limits, go = _growing(make_daemon, tmp_path, monkeypatch)
    hold(int(7.2 * GiB))
    real_read_exit = launcher.read_exit
    monkeypatch.setattr(
        launcher, "read_exit",
        lambda paths, i: (r := real_read_exit(paths, i)) and {**r, "exit_code": 137, "oom_killed": True, "hit_limit": False},
    )  # fmt: skip
    go.touch()
    run_until(d, lambda: finished(d, a))
    assert job(d, a).reason == ("killed when the machine ran out of RAM, after its limit grew past its 8G reservation; "
                                "resubmit with --mem 10G")  # fmt: skip


def test_a_grown_retry_job_killed_when_the_machine_ran_out_runs_again_with_more(make_daemon, tmp_path, monkeypatch):
    d, a, hold, limits, go = _growing(make_daemon, tmp_path, monkeypatch, retry=True)
    hold(int(7.2 * GiB))
    real_read_exit = launcher.read_exit
    seen = []

    def read_exit(paths, i):
        r = real_read_exit(paths, i)
        if r and i == a and not seen:
            seen.append(True)
            return {**r, "exit_code": 137, "oom_killed": True, "hit_limit": False}
        return r

    monkeypatch.setattr(launcher, "read_exit", read_exit)
    go.touch()
    run_until(d, lambda: job(d, a).attempts)
    assert (job(d, a).request.mem, job(d, a).ram_limit) == (10 * GiB, None)


def test_after_a_limit_it_couldnt_raise_dibs_waits_a_minute_before_trying_again(make_daemon, tmp_path, monkeypatch):
    d, a, hold, limits, go = _growing(make_daemon, tmp_path, monkeypatch, ok=False)
    hold(int(7.2 * GiB))
    hold(int(7.3 * GiB))
    assert len(limits) == 1
    go.touch()



def test_earlier_runs_say_when_the_gpu_they_reserved_went_unused():
    # Issue #139: an agent sizing its next run saw nothing about VRAM for runs that never used the GPU.
    from dibs.model import Job, Request, State, job_to_dict
    from dibs.render import render_earlier

    t = 1_000_000.0

    def run(i, peak_vram, ran=120.0):
        r = Request(cpus=1, mem=GiB, gpus=1, vram=GiB, time=600.0)
        return job_to_dict(Job(id=i, name="x", argv=["python", "x.py"], cwd="/w", env={}, request=r, session="s",
                               session_label="w·s", submitted=t, tag=t, state=State.DONE, output="/w/o", started=t,
                               ended=t + ran, exit_code=0, peak_vram=peak_vram))  # fmt: skip

    hint = "None of them used the GPU they reserved: leave --vram out, so the job holds no GPU memory another job could use."
    text = render_earlier([run(1, 0), run(2, 0)])
    assert "VRAM 0 of 1G" in text and text.endswith(hint)
    assert hint not in render_earlier([run(1, 0), run(2, 300 * MiB)])  # one did use it
    assert hint not in render_earlier([run(1, 0, ran=3.0)])  # too short to tell
    assert hint not in render_earlier([run(1, None)])  # dibs couldn't see the GPU's processes


def test_a_waiting_jobs_view_names_the_job_past_its_estimate_it_waits_for(make_daemon, tmp_path):
    # #181: its predicted start is that job's assumed end, which moves with every pass.
    now = [1_000_000.0]
    d = make_daemon(clock=lambda: now[0], slept=lambda: 0.0)
    go = tmp_path / "go"
    holding = [PY, "-c", f"import os, time\nwhile not os.path.exists({str(go)!r}): time.sleep(0.02)"]
    running = submit(d, tmp_path, holding, cpus="all", time_=600.0)
    run_until(d, lambda: job(d, running).shim_pid)
    now[0] += 900  # 5 minutes past its 10-minute estimate
    waiting = submit(d, tmp_path, [PY, "-c", "pass"], cpus="all")
    assert d.show(waiting)["job"]["after"] == {"id": running, "over": 300.0, "estimate": 600.0}
    go.touch()
    run_until(d, lambda: finished(d, running))


def test_a_start_waiting_on_a_job_past_its_estimate_is_shown_as_that_job_ending():
    from dibs.model import Job, Request, State, job_to_dict
    from dibs.render import _when_cell, render_show, render_submit, top_screen

    t = 1_000_000.0
    request = Request(cpus=2, mem=8 * GiB, gpus=1, vram=1 * GiB, time=3600.0)
    waiting = Job(id=2644, name="game", argv=["python", "game.py"], cwd="/w", env={}, request=request, session="s",
                  session_label="w·s", submitted=t, tag=t + 3600, state=State.QUEUED, output="/w/out")  # fmt: skip
    view = {**job_to_dict(waiting), "position": 1, "predicted_start": t + 300, "predicted_end": t + 3900,
            "waiting_for": "needs 1G VRAM (0 free)", "lacks": ["VRAM"], "behind": None,
            "after": {"id": 2270, "over": 3 * 3600 + 840.0, "estimate": 3000.0}}  # fmt: skip
    ending = "when #2270 ends (3h14m past its 50m estimate)"
    assert _when_cell(view, t) == f"starts {ending}: needs 1G VRAM (0 free)"
    assert f"expected to start {ending}." in render_submit(view, t)
    assert f"start {ending}" in render_show(view, t) and "in 5m" not in render_show(view, t)


def _ended_job(**kw):
    from dibs.model import Job, Request, State, job_to_dict

    t = 1_000_000.0
    fields = dict(id=2218, name="export", argv=["python", "export.py"], cwd="/w", env={}, session="s",
                  request=Request(cpus=4, mem=9 * GiB, gpus=0, vram=0, time=3600.0), session_label="w·s",
                  submitted=t, tag=t + 3600, state=State.FAILED, output="/w/out", started=t, ended=t + 960,
                  exit_code=143, reason="killed: went over its 9G RAM reservation; resubmit with --mem 12G",
                  peak_mem=9 * GiB, cpu_seconds=3.3 * 960)  # fmt: skip
    fields.update(kw)
    return job_to_dict(Job(**fields)), t


def test_dibs_show_gives_a_failed_jobs_reason_on_its_ended_line():
    # #175: agents keep the ended line and drop the reason line, which says what to resubmit with.
    from dibs.render import render_show

    view, t = _ended_job()
    text = render_show(view, t + 1000)
    ended = next(line for line in text.splitlines() if line.lstrip().startswith("ended"))
    assert ended.endswith("after 16m, exit 143: killed: went over its 9G RAM reservation; resubmit with --mem 12G")
    assert re.search(r"reason +killed: went over its 9G RAM reservation", text)  # kept, for filters that look for it


def test_dibs_show_says_on_the_ended_line_which_try_it_was():
    # #175: a retried job's ended line gave only its last try, and the tries are on lines of their own.
    from dibs.render import render_show

    t = 1_000_000.0
    first = {"started": t - 400, "ended": t - 40, "exit_code": 137, "asleep": 0.0, "peak_mem": 8 * GiB, "peak_vram": None,
             "reason": "killed: went over its 8G RAM reservation; retrying with --mem 10G", "request": {}}  # fmt: skip
    view, _ = _ended_job(started=t, ended=t + 6, exit_code=1, reason=None, attempts=[first], cpu_seconds=None)
    ended = next(line for line in render_show(view, t + 60).splitlines() if line.lstrip().startswith("ended"))
    assert ended.endswith("after 6s, exit 1, on its 2nd try (the 1st ended after 6m: killed: went over its 8G RAM reservation)")


def test_dibs_show_gives_cpu_use_on_its_peak_line():
    # #175: while right-sizing, an agent wrote that dibs show reports peak RAM but not CPU.
    from dibs.render import render_show

    view, t = _ended_job()
    assert "peak      RAM 9G of 9G (not counting file cache), CPU 3.3 of 4" in render_show(view, t + 1000)
    short, _ = _ended_job(ended=t + 5, cpu_seconds=4.0)  # too short a run to tell
    assert "CPU" not in render_show(short, t + 1000).split("peak", 1)[1]
    running = {**view, "state": "running", "ended": None, "exit_code": None, "reason": None, "cpu_ids": [0, 1, 2, 3],
               "gpu_ids": [], "expected_end": t + 3600, "over_estimate": 0.0, "cpu_seconds": 1.5 * 600}  # fmt: skip
    assert "CPU 1.5 of 4" in render_show(running, t + 600)


def test_a_running_jobs_view_carries_the_cpu_time_its_scope_has_used(daemon, tmp_path, monkeypatch):
    # #175: so dibs show's peak line can give CPU use while the job runs.
    go = tmp_path / "go"
    holding = [PY, "-c", f"import os, time\nwhile not os.path.exists({str(go)!r}): time.sleep(0.02)"]
    running = submit(daemon, tmp_path, holding)
    run_until(daemon, lambda: job(daemon, running).shim_pid)
    monkeypatch.setattr(daemon, "_job_cpu_seconds", lambda j: 42.0)
    assert daemon.show(running)["job"]["cpu_seconds"] == 42.0
    go.touch()
    run_until(daemon, lambda: finished(daemon, running))


def test_a_restart_in_a_shortage_doesnt_start_the_relief_wait_over(make_daemon, tmp_path):
    # #190: two restarts in a shortage's first minute each started the 10-second wait over, and dibs
    # stopped nothing; then systemd-oomd killed a whole terminal, an agent and its subagents with it.
    now = [1_000_000.0]
    short = dict(clock=lambda: now[0], pressure=lambda: 60.0, meminfo=lambda: (16 * GiB, 15 * GiB))
    first = make_daemon(**short)
    first._relieve_shortage(now[0])  # the shortage begins
    first.close()
    now[0] += 6
    second = make_daemon(**short)
    stopped = []
    second._stop_ram_hog = lambda t: stopped.append(t) or True
    second._relieve_shortage(now[0])
    assert stopped == []  # 6 s in
    now[0] += 5
    second._relieve_shortage(now[0])
    assert stopped == [now[0]]  # 11 s since it began, across the restart


def test_a_shortage_long_over_isnt_resumed_after_a_restart(make_daemon, tmp_path):
    now = [1_000_000.0]
    short = dict(clock=lambda: now[0], pressure=lambda: 60.0, meminfo=lambda: (16 * GiB, 15 * GiB))
    first = make_daemon(**short)
    first._relieve_shortage(now[0])
    first.close()
    now[0] += 600  # dibs was down for 10 minutes
    second = make_daemon(**short)
    stopped = []
    second._stop_ram_hog = lambda t: stopped.append(t) or True
    second._relieve_shortage(now[0])
    assert stopped == []
