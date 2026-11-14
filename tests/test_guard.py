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
from dibs.units import GiB

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


def test_vram_overuse_stops_job(world, tmp_path):
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
    backend.procs = [GpuProcess(pid, 0, 7 * GiB)]
    run_until(d, lambda: d.store.get_job(job_id).state == State.FAILED)
    j = d.store.get_job(job_id)
    assert j.reason == "stopped: used 7G VRAM on GPU 0, reserved 6G"
    assert j.peak_vram == 7 * GiB


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
