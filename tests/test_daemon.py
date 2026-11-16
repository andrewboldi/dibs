import json
import os
import shutil
import subprocess
import sys
import time

from conftest import FakeBackend, capacity, run_until

from dibs import launcher, procs
from dibs.daemon import OUTSIDE_WINDOW
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


def _job_holding(d, tmp_path, monkeypatch, mem):
    """A running job whose scope dibs reads its memory from, and a file that ends the job once it exists."""
    go = tmp_path / "go"
    job_id = submit(d, tmp_path, [PY, "-c", f"import os, time\nwhile not os.path.exists({str(go)!r}): time.sleep(0.02)"],
                    mem=mem)
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
