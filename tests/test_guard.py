import os
import subprocess
import sys
import time

import pytest
from conftest import FAKE_GPU, FakeBackend, capacity, run_until

from dibs import guard, launcher, procs
from dibs.gpu import GpuProcess
from dibs.machine import Config
from dibs.model import Gpu, State
from dibs.render import render_queue
from dibs.units import GiB, MiB

PY = sys.executable
SLEEPER = [PY, "-c", "import time; time.sleep(60)"]


def wait_for(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("timed out")


def exec_done(pid, argv0):
    """Popen returns before /proc shows the new program; the guard reads /proc."""
    wait_for(lambda: (procs.cmdline(pid) or [""])[0] == argv0)


@pytest.fixture
def spawned():
    started = []

    def spawn(argv, **kw):
        p = subprocess.Popen(argv, **kw)
        started.append(p)
        exec_done(p.pid, argv[0])
        return p

    yield spawn
    for p in started:
        if p.poll() is None:
            p.kill()
        p.wait()


@pytest.fixture
def world(make_daemon):
    """A daemon with a fake GPU backend; the guard is switched on before anything is spawned."""

    def _world(config=Config(), gpus=(FAKE_GPU,)):
        backend = FakeBackend(gpus=gpus)
        d = make_daemon(config=config, cap=capacity(gpus=gpus), backend=backend)
        time.sleep(0.05)  # processes spawned from here on start after the guard did
        return d, backend

    return _world


def outside(d):
    return {o["pid"]: o["why"] for o in d.snapshot()["guard"]["outside"]}


def test_unmanaged_process_stopped_with_message(world, spawned):
    d, backend = world()
    p = spawned(SLEEPER, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    backend.procs = [GpuProcess(p.pid, 0, int(3.1 * GiB))]
    d.tick()
    assert p.wait(timeout=5) == -9
    out = p.stdout.read().decode()
    assert out.count("dibs: stopped") == 1
    assert "--vram 4G" in out and "--cpus N" in out and "--time ESTIMATE" in out
    events = d.store.events(0)
    assert [(e["kind"], e["pid"], e["gpu"]) for e in events] == [("stop", p.pid, 0)]
    assert events[0]["command"] == SLEEPER


def test_parent_shell_survives(world, spawned):
    d, backend = world()
    script = f"'{PY}' -c 'import time; time.sleep(60)'; echo parent-survived"
    shell = spawned(["bash", "-c", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    child = wait_for(lambda: next(iter(procs.descendants(shell.pid)), None))
    exec_done(child, PY)
    backend.procs = [GpuProcess(child, 0, 1 * GiB)]
    d.tick()
    out, _ = shell.communicate(timeout=10)
    assert "parent-survived" in out and "dibs: stopped" in out


def test_job_process_untouched(world, tmp_path):
    d, backend = world()
    resp = d.submit(
        {
            "argv": SLEEPER,
            "cwd": str(tmp_path),
            "env": dict(os.environ),
            "session": "s",
            "request": {"cpus": 1, "mem": GiB, "vram": 6 * GiB, "gpus": None, "time": 60},
        }
    )
    job = d.store.get_job(resp["job"]["id"])
    pid = wait_for(lambda: next((p for p in launcher.job_pids(job) if (procs.cmdline(p) or [""])[0] == PY), None))
    backend.procs = [GpuProcess(pid, 0, 5 * GiB)]
    d.tick()
    assert procs.alive(pid)
    assert d.store.get_job(job.id).peak_vram == 5 * GiB
    assert d.store.events(0) == [] and outside(d) == {}


def test_allowlisted_left_alone(world, spawned):
    d, backend = world(Config(guard_allow=("sleep",)))
    p = spawned(["sleep", "60"])
    backend.procs = [GpuProcess(p.pid, 0, GiB)]
    d.tick()
    assert procs.alive(p.pid) and outside(d) == {p.pid: "allowlisted"}


APP_SLICE = "/user.slice/user-1000.slice/user@1000.service/app.slice/"


@pytest.mark.parametrize(
    "scope,config,left_alone",
    [
        # Issue #92: the guard stopped a screen recorder the user launched from the desktop
        ("app-gnome-com.obsproject.Studio-3016784.scope", Config(), True),
        ("app-org.kde.kdenlive@3f2a1b.service", Config(), True),
        # where agents run: terminals, tmux, and editors with a built-in terminal
        ("tmux-spawn-f3ccf531-0060-48f4-920d-4dd6eefb12fe.scope", Config(), False),
        ("app-org.gnome.Terminal.slice/vte-spawn-1b2c.scope", Config(), False),
        ("app-gnome-org.gnome.Terminal-4410.scope", Config(), False),
        ("app-org.kde.konsole@a1b2.service", Config(), False),
        ("app-gnome-code-insiders-5120.scope", Config(), False),
        ("app-gnome-com.obsproject.Studio-3016784.scope", Config(guard_desktop=False), False),
    ],
)
def test_programs_launched_from_the_desktop_are_left_alone(world, spawned, monkeypatch, scope, config, left_alone):
    d, backend = world(config)
    p = spawned(["sleep", "60"])
    real_cgroup = procs.cgroup
    monkeypatch.setattr(procs, "cgroup", lambda pid: APP_SLICE + scope if pid == p.pid else real_cgroup(pid))
    backend.procs = [GpuProcess(p.pid, 0, 10 * 1024 * 1024)]
    d.tick()
    if left_alone:
        assert procs.alive(p.pid) and outside(d) == {p.pid: "launched from the desktop"}
    else:
        p.wait(timeout=10)
        assert not procs.alive(p.pid)


def test_preexisting_left_alone(make_daemon, spawned):
    p = spawned(["sleep", "60"])  # already running when dibs starts
    time.sleep(0.05)
    backend = FakeBackend()
    d = make_daemon(backend=backend)
    backend.procs = [GpuProcess(p.pid, 0, GiB)]
    d.tick()
    assert procs.alive(p.pid)
    assert outside(d) == {p.pid: "already running when the guard started"}


def test_other_user_reported(world):
    d, backend = world()
    backend.procs = [GpuProcess(1, 0, GiB)]  # init belongs to root
    d.tick()
    assert outside(d) == {1: "another user's"} and d.store.events(0) == []


def test_warn_mode_reports_only(world, spawned):
    d, backend = world(Config(guard="warn"))
    p = spawned(["sleep", "60"])
    backend.procs = [GpuProcess(p.pid, 0, GiB)]
    d.tick()
    assert procs.alive(p.pid) and outside(d) == {p.pid: "warn mode"}
    assert d.store.events(0) == []


def test_off_mode_ignores(world, spawned):
    d, backend = world(Config(guard="off"))
    p = spawned(["sleep", "60"])
    backend.procs = [GpuProcess(p.pid, 0, GiB)]
    d.tick()
    assert procs.alive(p.pid) and outside(d) == {}


def test_unmanaged_gpu_is_not_policed(world, spawned):
    d, backend = world()  # dibs manages GPU 0 only
    p = spawned(["sleep", "60"])
    backend.procs = [GpuProcess(p.pid, 1, GiB)]
    d.tick()
    assert procs.alive(p.pid)


def test_process_on_two_gpus_stopped_once(world, spawned):
    gpus = (FAKE_GPU, Gpu(1, "GPU-fake-1", "Fake GPU", 8 * GiB))
    d, backend = world(gpus=gpus)
    p = spawned(SLEEPER, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    backend.procs = [GpuProcess(p.pid, 0, GiB), GpuProcess(p.pid, 1, GiB)]
    d.tick()
    p.wait(timeout=5)
    assert p.stdout.read().decode().count("dibs: stopped") == 1
    assert len(d.store.events(0)) == 1


def test_blind_driver_is_harmless(world, spawned):
    d, backend = world()
    backend.blind = True
    d.tick()
    assert outside(d) == {}


def test_guard_since_survives_restart(make_daemon):
    first = make_daemon()
    since = first.guard_since
    first.close()
    time.sleep(0.05)
    second = make_daemon()
    assert second.guard_since == since
    second.close()
    make_daemon(config=Config(guard="off")).close()
    assert make_daemon().guard_since > since  # switched off and on again: a new start


EARLY = ", in its first minute, so it was likely still growing"


@pytest.mark.parametrize(
    "used,ran,reason",
    [
        (7 * GiB, 0, f"stopped: used 7G VRAM on GPU 0, reserved 6G{EARLY}; resubmit with --vram all"),
        # 1 MiB over: rounded to 6G, the figure would hide the overage
        (6 * GiB + MiB, 0, f"stopped: used 6145M VRAM on GPU 0, reserved 6G{EARLY}; resubmit with --vram all"),
        # Issue #80: after its first minute, the peak is a fair guide and 10% on top will do
        (6 * GiB + MiB, 120, "stopped: used 6145M VRAM on GPU 0, reserved 6G; resubmit with --vram 7G"),
    ],
)
def test_vram_overuse_stops_job(world, tmp_path, used, ran, reason):
    d, backend = world()
    resp = d.submit(
        {
            "argv": SLEEPER,
            "cwd": str(tmp_path),
            "env": dict(os.environ),
            "session": "s",
            "request": {"cpus": 1, "mem": GiB, "vram": 6 * GiB, "gpus": None, "time": 60},
        }
    )
    job_id = resp["job"]["id"]
    job = d.store.get_job(job_id)
    pid = wait_for(lambda: next((p for p in launcher.job_pids(job) if (procs.cmdline(p) or [""])[0] == PY), None))
    job = d.store.get_job(job_id)
    job.started -= ran  # as if it had been running this long
    d.store.update_job(job)
    backend.procs = [GpuProcess(pid, 0, used)]
    run_until(d, lambda: d.store.get_job(job_id).state == State.FAILED)
    j = d.store.get_job(job_id)
    assert j.reason == reason
    assert j.peak_vram == used


def test_stop_message_text():
    text = guard.stop_message(4121, ["python", "train.py", "--epochs", "3"], 0, int(3.1 * GiB), int(5.2 * GiB))
    assert text == (
        "dibs: stopped `python train.py --epochs 3` (pid 4121): it used 3.1G on GPU 0 outside dibs.\n"
        "dibs: GPUs on this machine are shared through dibs. Run it as a job instead:\n"
        "dibs:   dibs run --cpus N --mem 6G --vram 4G --time ESTIMATE -- python train.py --epochs 3\n"
        "dibs: `dibs agent-help` explains how to choose N and ESTIMATE."
    )


def test_stop_message_without_usage_numbers():
    text = guard.stop_message(7, ["./a.out"], 1, None, None)
    assert "it used GPU 1 outside dibs" in text
    assert "--mem SIZE --vram SIZE" in text


def test_queue_shows_guard_lines(world, spawned):
    d, backend = world()
    p = spawned(SLEEPER)
    backend.procs = [GpuProcess(p.pid, 0, int(3.1 * GiB))]
    d.tick()
    p.wait(timeout=5)
    text = render_queue(d.snapshot(), color=False)
    assert "Stopped outside dibs: pid" in text and "(3.1G on GPU 0)" in text


def _find_by_marker(marker):
    for name in os.listdir("/proc"):
        if name.isdigit() and marker in (procs.cmdline(int(name)) or []) and procs.alive(int(name)):
            return int(name)
    return None


def test_new_gpu_processes_of_work_already_running_are_left_alone(make_daemon, spawned):
    # Like torchrun restarting a worker, or a new kernel under a Jupyter server that was running.
    sleeper = f"'{PY}' -c 'import time; time.sleep(60)'"
    launcher_proc = spawned(["bash", "-c", f"{sleeper} & sleep 1.5; {sleeper} & wait"])
    first = wait_for(lambda: next(iter(procs.descendants(launcher_proc.pid)), None))
    exec_done(first, PY)
    backend = FakeBackend()
    d = make_daemon(backend=backend)  # the guard switches on while `first` runs
    backend.procs = [GpuProcess(first, 0, GiB)]
    d.tick()
    second = wait_for(
        lambda: next(
            (c for c in procs.descendants(launcher_proc.pid) if c != first and (procs.cmdline(c) or [""])[0] == PY),
            None,
        )
    )
    backend.procs = [GpuProcess(first, 0, GiB), GpuProcess(second, 0, GiB)]
    d.tick()
    assert procs.alive(first) and procs.alive(second)
    assert outside(d)[second] == "started by work that was already running when the guard started"
    d.close()
    restarted = make_daemon(backend=backend)  # remembered across a restart
    restarted.tick()
    assert procs.alive(second)


def test_orphans_never_make_their_adopter_exempt(make_daemon):
    # An orphan is adopted by init or the session manager; remembering that parent would exempt
    # everything the user runs.
    def orphan(marker):
        subprocess.run(["bash", "-c", f"('{PY}' -c 'import time, sys; time.sleep(60)' {marker} &)"], check=True)
        pid = wait_for(lambda: _find_by_marker(marker))
        exec_done(pid, PY)
        return pid

    old = orphan("dibs-test-old-orphan")
    backend = FakeBackend()
    d = make_daemon(backend=backend)
    try:
        backend.procs = [GpuProcess(old, 0, GiB)]
        d.tick()
        time.sleep(0.05)  # start times are in 10 ms ticks: keep `new` clearly after the guard started
        new = orphan("dibs-test-new-orphan")
        backend.procs = [GpuProcess(old, 0, GiB), GpuProcess(new, 0, GiB)]
        d.tick()
        assert procs.alive(old)
        wait_for(lambda: not procs.alive(new))
    finally:
        for marker in ("dibs-test-old-orphan", "dibs-test-new-orphan"):
            pid = _find_by_marker(marker)
            if pid:
                os.kill(pid, 9)


def test_another_dibs_instances_job_is_left_alone(world, spawned):
    # Two dibs of one user (issue #3): each saw the other's jobs as GPU processes outside dibs.
    d, backend = world()
    p = spawned(SLEEPER, env={**os.environ, "DIBS_INSTANCE": "0ther1st", "DIBS_JOB_ID": "3"})
    backend.procs = [GpuProcess(p.pid, 0, GiB)]
    d.tick()
    assert procs.alive(p.pid)
    assert outside(d) == {p.pid: "another dibs's job"}


def test_a_leftover_of_our_own_finished_job_is_still_stopped(world, spawned):
    d, backend = world()
    p = spawned(SLEEPER, env={**os.environ, "DIBS_INSTANCE": d.paths.instance, "DIBS_JOB_ID": "99"})
    backend.procs = [GpuProcess(p.pid, 0, GiB)]
    d.tick()
    wait_for(lambda: not procs.alive(p.pid))


def test_a_process_that_already_exited_is_not_recorded_as_stopped():
    # Issue #60: the guard recorded a stop, with no command and no session, for a GPU process that
    # had exited between being seen on the GPU and being stopped.
    gone = subprocess.Popen([PY, "-c", "pass"])
    gone.wait()
    assert guard.enforce([guard.Verdict(gone.pid, 0, 1 << 30, "stop", None)]) == []


def test_a_live_process_without_a_readable_command_is_recorded_by_its_name(monkeypatch):
    p = subprocess.Popen(SLEEPER)
    try:
        monkeypatch.setattr(procs, "cmdline", lambda pid: None)
        events = guard.enforce([guard.Verdict(p.pid, 0, 1 << 30, "stop", None)])
        assert len(events) == 1 and events[0]["command"][0].startswith("python")
        p.wait(timeout=15)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
