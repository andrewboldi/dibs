"""Jobs that hold GPU memory with the GPU idle: recorded, shown, and their session told early (#180, #186)."""

import json
import os

import pytest
from conftest import FakeBackend, run_until
from test_daemon import PY, finished, job, submit

from dibs.gpu import GpuProcess
from dibs.units import GiB, MiB


def _holding(tmp_path):
    go = tmp_path / "go"
    return go, [PY, "-c", f"import os, time\nwhile not os.path.exists({str(go)!r}): time.sleep(0.02)"]


def _idle_holder(make_daemon, tmp_path, held=int(6.3 * GiB), util=0.0):
    """A job holding `held` of the 8G GPU, at `util` percent, and another session's job waiting for GPU memory."""
    now = [1_000_000.0]
    backend = FakeBackend()
    d = make_daemon(backend=backend, clock=lambda: now[0], slept=lambda: 0.0)
    go, holding = _holding(tmp_path)
    holder = submit(d, tmp_path, holding, vram=7 * GiB, session="s1")
    run_until(d, lambda: job(d, holder).shim_pid)
    backend.procs = [GpuProcess(job(d, holder).shim_pid, 0, held)]
    backend.util = {0: util}
    waiting = submit(d, tmp_path, holding, vram=2 * GiB, session="s2")
    now[0] += 30  # past the last check, made while the job started
    return d, now, go, holder, waiting


def _idle_notices(d, session, now):
    return [n for n in d.notices(session, now)["notices"] if n["kind"] == "idle_gpu"]


def _end(d, go, *ids):
    go.touch()
    run_until(d, lambda: all(finished(d, i) for i in ids))


def test_a_job_holding_an_idle_gpus_memory_is_caught_after_ten_minutes_and_its_session_told(make_daemon, tmp_path):
    # #180, #186: a job held 6.3G of the GPU's 8G with the GPU idle for 4.5 hours while others waited.
    from dibs.render import _when_cell

    d, now, go, holder, waiting = _idle_holder(make_daemon, tmp_path)
    d._watch_gpus(now[0])  # idle from here
    now[0] += 9 * 60
    d._watch_gpus(now[0])
    assert _idle_notices(d, "s1", now[0]) == []  # 9 minutes: not yet
    now[0] += 61
    d._watch_gpus(now[0])
    (told,) = _idle_notices(d, "s1", now[0])
    assert f"Job {holder} has held 6.3G of GPU memory with the GPU idle for 10m" in told["message"]
    assert "while 1 job waits for GPU memory" in told["message"]
    assert "until its processes free it or exit" in told["message"]
    assert "GPU idle 10m, holding 6.3G" in _when_cell(d.show(holder)["job"], now[0])
    now[0] += 30 * 60
    d._watch_gpus(now[0])
    assert _idle_notices(d, "s1", now[0]) == []  # at most once an hour
    assert _idle_notices(d, "s2", now[0]) == []  # only the job's own session
    _end(d, go, holder, waiting)


@pytest.mark.parametrize("held, util", [(int(6.3 * GiB), 60.0), (500 * MiB, 0.0)])  # computing; a rendering context
def test_a_busy_gpu_or_a_small_hold_isnt_an_idle_hold(make_daemon, tmp_path, held, util):
    d, now, go, holder, waiting = _idle_holder(make_daemon, tmp_path, held=held, util=util)
    for _ in range(25):
        d._watch_gpus(now[0])
        now[0] += 30
    assert job(d, holder).gpu_idle_since is None and _idle_notices(d, "s1", now[0]) == []
    _end(d, go, holder, waiting)


def test_no_notice_while_nobody_waits_for_gpu_memory(make_daemon, tmp_path):
    d, now, go, holder, waiting = _idle_holder(make_daemon, tmp_path)
    d.cancel([waiting])
    for _ in range(25):
        d._watch_gpus(now[0])
        now[0] += 30
    assert job(d, holder).gpu_idle_since is not None  # recorded and shown all the same
    assert _idle_notices(d, "s1", now[0]) == []
    _end(d, go, holder)


def test_per_process_use_tells_which_of_two_jobs_on_a_gpu_is_idle(make_daemon, tmp_path):
    d, now, go, holder, waiting = _idle_holder(make_daemon, tmp_path, held=3 * GiB)
    d.cancel([waiting])
    _, busy_cmd = _holding(tmp_path)
    busy = submit(d, tmp_path, [*busy_cmd, "busy"], vram=int(0.9 * GiB), session="s3")
    run_until(d, lambda: job(d, busy).shim_pid)
    d.backend.procs.append(GpuProcess(job(d, busy).shim_pid, 0, int(0.9 * GiB)))
    d.backend.util = {0: 70.0}  # the GPU is busy, with the other job's work
    d.backend.proc_util = {job(d, holder).shim_pid: 0.0, job(d, busy).shim_pid: 70.0}
    for _ in range(25):
        d._watch_gpus(now[0])
        now[0] += 30
    assert job(d, holder).gpu_idle_since is not None and job(d, busy).gpu_idle_since is None
    _end(d, go, holder, busy)


def test_a_finished_jobs_idle_hold_counts_in_its_record_and_shows_in_its_earlier_runs(make_daemon, tmp_path):
    from dibs.model import job_to_dict
    from dibs.render import render_earlier, render_show

    d, now, go, holder, waiting = _idle_holder(make_daemon, tmp_path)
    d.cancel([waiting])
    for _ in range(31):  # 15 minutes
        d._watch_gpus(now[0])
        now[0] += 30
    _end(d, go, holder)
    ended = job(d, holder)
    assert 15 * 60 <= ended.gpu_idle <= 16 * 60 and ended.gpu_idle_since is None
    assert "GPU idle 15m (6.3G)" in render_show(job_to_dict(ended), now[0])
    assert "GPU idle 15m" in render_earlier([job_to_dict(ended)])


def test_a_job_saved_before_idle_holds_were_recorded_still_loads():
    from dibs.model import Job, Request, State, job_from_dict, job_to_dict

    old = job_to_dict(Job(id=1, name="x", argv=["x"], cwd="/w", env={}, request=Request(cpus=1, mem=GiB, gpus=0, vram=0, time=60.0),
                          session="s", session_label="w·s", submitted=0.0, tag=60.0, state=State.DONE, output="/w/o"))  # fmt: skip
    for key in ("gpu_idle", "gpu_idle_since", "gpu_idle_held", "gpu_idle_told"):
        old.pop(key, None)
    assert job_from_dict(old).gpu_idle == 0.0


def test_nvidia_smi_reports_each_gpus_use(tmp_path):
    from dibs.gpu import SmiBackend

    gpu_file = tmp_path / "gpus.json"
    gpu_file.write_text(json.dumps({"gpus": [{"index": 0, "uuid": "GPU-fake-0", "name": "Fake GPU", "memory_mib": 8192,
                                              "utilization": 37}]}))  # fmt: skip
    fakes = os.path.join(os.path.dirname(__file__), "fakes")
    smi = SmiBackend({"PATH": f"{fakes}:{os.environ['PATH']}", "DIBS_FAKE_GPU_FILE": str(gpu_file)})
    assert smi.utilization() == {0: 37.0} and smi.process_utilization() is None


def test_top_and_the_agent_help_show_an_idle_gpu_hold():
    from test_top import NOW, view

    from dibs.instructions import AGENT_HELP
    from dibs.model import State
    from dibs.render import render_top

    idle = view(1, "train", State.RUNNING, 4, 8 * GiB, 6 * GiB, 1500, started=NOW - 9000, cpu_ids=[0, 1, 2, 3],
                gpu_ids=[0], expected_end=NOW + 900, over_estimate=7500, gpu_idle_since=NOW - 6000, gpu_idle_held=6 * GiB)  # fmt: skip
    snap = {"now": NOW, "jobs": [idle], "guard": {"mode": "all", "stopped": [], "outside": []},
            "capacity": {"cpus": list(range(24)), "mem": 32 * GiB,
                         "gpus": [{"index": 0, "uuid": "GPU-0", "name": "Fake GPU", "vram": 8 * GiB}]}}  # fmt: skip
    assert "GPU idle 1h40m" in render_top(snap, width=160, height=40, color=False)
    assert "until its processes free it or exit" in " ".join(AGENT_HELP.split())


@pytest.mark.parametrize("retry", [False, True])
def test_a_job_the_machine_restarted_under_ends_its_idle_hold_with_its_run(make_daemon, tmp_path, monkeypatch, retry):
    # #290: a training job looked idle for a minute after a wake, then the machine shut down. Run again after
    # the reboot, it kept that open idle stretch, and its first check counted the 37-minute shutdown as idle:
    # "GPU idle 36m" for a run that kept the GPU 99% busy. Lost without a retry, its idle time grew for good.
    import subprocess

    from dibs import procs
    from dibs.model import State
    from dibs.render import _idle_total

    now = [1_000_000.0]
    monkeypatch.setattr(procs, "boot_id", lambda: "boot-a")
    first = make_daemon(clock=lambda: now[0], slept=lambda: 0.0)
    submit(first, tmp_path, ["sleep", "30"], vram="all")
    job_id = submit(first, tmp_path, ["true"], vram="all")  # queued behind the blocker
    now[0] += 20 * 60
    first.tick()  # the last time dibs saw the machine up
    noted = now[0]
    dead = subprocess.Popen(["true"])
    dead.wait()
    j = job(first, job_id)  # as if it had run 20 minutes, the last 15 holding GPU memory with the GPU idle
    j.state, j.started, j.sleep_at_start, j.shim_pid, j.shim_ticks = State.RUNNING, noted - 20 * 60, 0.0, dead.pid, None
    j.gpu_idle_since, j.gpu_idle_held, j.retry = noted - 15 * 60, 6 * GiB, retry
    first.store.update_job(j)
    first.close()
    monkeypatch.setattr(procs, "boot_id", lambda: "boot-b")  # the machine restarted
    now[0] += 37 * 60
    second = make_daemon(clock=lambda: now[0], slept=lambda: 0.0)
    second.recover()
    j = job(second, job_id)
    if not retry:  # its record holds the 15 minutes, and stops there
        assert (j.state, j.gpu_idle, j.gpu_idle_since) == (State.FAILED, 15 * 60, None)
        assert _idle_total(second.show(job_id)["job"], now[0] + 3600) == 15 * 60
    else:  # the try's record holds them, and the new run starts with none
        assert j.state == State.QUEUED and [a.get("gpu_idle") for a in j.attempts] == [15 * 60]
        assert (j.gpu_idle, j.gpu_idle_since, j.gpu_idle_held, j.gpu_idle_told) == (0.0, None, None, None)


# #292: a job computing on the GPU when the machine suspends doesn't come back on this machine (the NVIDIA
# driver's save of GPU memory fails): its process spins with its GPU memory held and the GPU idle. On
# 2026-11-24 two jobs did so for 52 minutes after a wake, 8 jobs waiting for GPU memory; the idle notice
# came 10 minutes in, only because others waited, and said the GPU was idle, not that the work was gone.


def _sleeper(make_daemon, tmp_path, util_before):
    """A job holding 6.3G of the GPU at `util_before` percent, looked at; then the machine sleeps two hours.
    Nothing else waits. Returns the daemon, its clock, how long the machine slept, the backend, the job."""
    now, slept = [1_000_000.0], [0.0]
    backend = FakeBackend()
    d = make_daemon(backend=backend, clock=lambda: now[0], slept=lambda: slept[0])
    go, holding = _holding(tmp_path)
    holder = submit(d, tmp_path, holding, vram=7 * GiB, session="s1")
    run_until(d, lambda: job(d, holder).shim_pid)
    backend.procs = [GpuProcess(job(d, holder).shim_pid, 0, int(6.3 * GiB))]
    backend.util = {0: util_before}
    now[0] += 30
    d._watch_gpus(now[0])  # the last look before the sleep
    now[0] += 7200
    slept[0] += 7200
    return d, now, go, backend, holder


def test_a_job_computing_when_the_machine_slept_and_idle_since_the_wake_is_told_at_once(make_daemon, tmp_path):
    d, now, go, backend, holder = _sleeper(make_daemon, tmp_path, util_before=100.0)
    backend.util = {0: 0.0}  # its work didn't survive
    d._watch_gpus(now[0])  # the first look after the wake
    assert _idle_notices(d, "s1", now[0]) == []  # not on one look: the GPU may still be waking
    now[0] += 30
    d._watch_gpus(now[0])
    now[0] += 30
    d._watch_gpus(now[0])
    (told,) = _idle_notices(d, "s1", now[0])  # a minute after the wake, with nothing else waiting
    assert f"Job {holder}'s GPU work most likely didn't survive the suspend" in told["message"]
    assert "6.3G of GPU memory" in told["message"] and "cancel it and resubmit it" in told["message"]
    now[0] += 30 * 60
    d._watch_gpus(now[0])
    assert len([e for e in d.store.events(0) if e["kind"] == "idle_gpu"]) == 1  # once
    _end(d, go, holder)


@pytest.mark.parametrize("util_before, util_after", [(0.0, 0.0), (100.0, 100.0)])  # idle before; busy after
def test_a_job_idle_before_the_sleep_or_busy_after_the_wake_isnt_told_it_was_lost(make_daemon, tmp_path, util_before, util_after):
    d, now, go, backend, holder = _sleeper(make_daemon, tmp_path, util_before=util_before)
    backend.util = {0: util_after}
    for _ in range(5):
        d._watch_gpus(now[0])
        now[0] += 30
    assert _idle_notices(d, "s1", now[0]) == []
    _end(d, go, holder)
