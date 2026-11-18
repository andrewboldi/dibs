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


def test_snapshot_shows_vram_in_use_outside_dibs(make_daemon, spawned):
    # Issue #120: dibs top draws what programs outside dibs use on each GPU.
    p = spawned(["sleep", "60"])
    time.sleep(0.05)
    backend = FakeBackend()
    d = make_daemon(backend=backend)
    assert d.snapshot()["outside"]["vram"] == {}
    backend.procs = [GpuProcess(p.pid, 0, GiB)]
    d.tick()
    assert d.snapshot()["outside"]["vram"] == {"0": GiB}


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
    _neighbour(d, tmp_path, 2 * GiB)  # holds the rest of the GPU, so going over collides with it (#88)
    resp = d.submit(
        {
            "argv": SLEEPER,
            "cwd": str(tmp_path),
            "env": dict(os.environ),
            "session": "s",
            "request": {"cpus": 1, "mem": GiB, "vram": 6 * GiB, "gpus": None, "time": 60},
            "retry": False,
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


def _gpu_job(d, tmp_path, vram, argv, **extra):
    resp = d.submit(
        {
            "argv": argv,
            "cwd": str(tmp_path),
            "env": dict(os.environ),
            "session": "s",
            "request": {"cpus": 1, "mem": GiB, "vram": vram, "gpus": None, "time": 60},
            **extra,
        }
    )
    assert resp["ok"], resp
    return resp["job"]["id"]


def _neighbour(d, tmp_path, vram, argv=None):
    """Another running job holding vram of the GPU: its own use isn't reported, so it holds its reservation."""
    job_id = _gpu_job(d, tmp_path, vram, argv or [PY, "-c", "import time; time.sleep(60)"])
    run_until(d, lambda: d.store.get_job(job_id).state == State.RUNNING)
    return job_id


def _job_python_pid(d, job_id):
    job = d.store.get_job(job_id)
    return wait_for(lambda: next((p for p in launcher.job_pids(d.store.get_job(job_id)) if (procs.cmdline(p) or [""])[0] == PY), None))


def test_a_job_stopped_for_vram_is_retried_with_the_advised_size(world, tmp_path):
    # Issue #84: dibs runs a job stopped for going over its VRAM reservation again, at the size the
    # stop advises, with the same id and place in line, so its agent needn't work out what happened.
    d, backend = world()
    _neighbour(d, tmp_path, 2 * GiB, [PY, "-c", "import time; time.sleep(2)"])  # leaves soon after the stop
    job_id = _gpu_job(d, tmp_path, 6 * GiB, [PY, "-c", "import time; time.sleep(1.5)"], retry=True)
    tag = d.store.get_job(job_id).tag
    backend.procs = [GpuProcess(_job_python_pid(d, job_id), 0, 6 * GiB + MiB)]
    run_until(d, lambda: d.store.get_job(job_id).attempts)
    backend.procs = []
    run_until(d, lambda: d.store.get_job(job_id).state == State.DONE)
    j = d.store.get_job(job_id)
    assert (j.request.vram, j.tag) == (8 * GiB, tag)
    assert [a["reason"] for a in j.attempts] == [
        "stopped: used 6145M VRAM on GPU 0, reserved 6G, in its first minute, so it was likely still growing; "
        "retrying with --vram all"
    ]
    assert "dibs: stopped: used 6145M VRAM on GPU 0" in open(j.output).read()


def test_a_job_is_retried_at_most_twice(world, tmp_path):
    d, backend = world()
    _neighbour(d, tmp_path, 2 * GiB)
    job_id = _gpu_job(d, tmp_path, 6 * GiB, [PY, "-c", "import time; time.sleep(30)"], retry=True)
    pid = _job_python_pid(d, job_id)
    j = d.store.get_job(job_id)
    j.attempts = [{"reason": "an earlier try"}, {"reason": "another"}]  # retried twice already
    d.store.update_job(j)
    backend.procs = [GpuProcess(pid, 0, 6 * GiB + MiB)]
    run_until(d, lambda: d.store.get_job(job_id).state == State.FAILED)
    j = d.store.get_job(job_id)
    assert len(j.attempts) == 2 and j.reason.endswith("; resubmit with --vram all")


def test_a_job_may_borrow_vram_nobody_reserved(world, tmp_path):
    # Issue #88: a job that went 0.1G over its VRAM reservation lost 50 minutes of work while 5G sat
    # unreserved. It may use memory nobody reserved, and the planner counts it at what it uses.
    d, backend = world()
    job_id = _gpu_job(d, tmp_path, 3 * GiB, [PY, "-c", "import time; time.sleep(60)"])
    backend.procs = [GpuProcess(_job_python_pid(d, job_id), 0, int(3.6 * GiB))]
    for _ in range(5):
        d.tick()
    assert d.store.get_job(job_id).state == State.RUNNING
    waiting = _gpu_job(d, tmp_path, int(4.6 * GiB), [PY, "-c", "pass"])  # fits beside 3G, not beside 3.6G
    d.tick()
    assert d.store.get_job(waiting).state == State.QUEUED


def test_a_borrower_is_stopped_when_it_reaches_memory_another_job_holds(world, tmp_path):
    d, backend = world()
    _neighbour(d, tmp_path, 2 * GiB)
    job_id = _gpu_job(d, tmp_path, 3 * GiB, [PY, "-c", "import time; time.sleep(60)"], retry=False)
    pid = _job_python_pid(d, job_id)
    backend.procs = [GpuProcess(pid, 0, int(5.9 * GiB))]  # within the 3G nobody holds: fine
    for _ in range(3):
        d.tick()
    assert d.store.get_job(job_id).state == State.RUNNING
    backend.procs = [GpuProcess(pid, 0, int(6.1 * GiB))]  # into the neighbour's 2G
    run_until(d, lambda: d.store.get_job(job_id).state == State.FAILED)
    assert d.store.get_job(job_id).reason.startswith("stopped: used 6.1G VRAM on GPU 0, reserved 3G")


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


# Programs outside dibs that run the machine out of RAM (#135) ----------------------------------

CLEAN_ENV = {k: v for k, v in os.environ.items() if not k.startswith("DIBS_")}  # not a job of any dibs


PANE = "app-org.gnome.Terminal.slice/tmux-spawn-f3ccf531-0060-48f4-920d-4dd6eefb12fe.scope"


@pytest.fixture
def short_world(make_daemon, monkeypatch):
    """A daemon on a 16G machine with a fake clock, fake memory pressure and RAM in use, and a fake
    view of the machine's big programs: the guard must never see this machine's real ones. place()
    puts a process in a scope, by default an agent's terminal pane."""
    scopes = {}
    real_cgroup = procs.cgroup
    monkeypatch.setattr(procs, "cgroup", lambda pid: APP_SLICE + scopes[pid] if pid in scopes else real_cgroup(pid))
    # The test runner itself may run under an agent CLI (an agent running the tests): its ancestors
    # are not part of the world under test.
    runner, pid = {os.getpid()}, os.getpid()
    while (pid := procs.parent(pid) or 0) > 1:
        runner.add(pid)
    real_is_agent = guard._is_agent
    monkeypatch.setattr(guard, "_is_agent", lambda pid: pid not in runner and real_is_agent(pid))

    def place(pid, scope=PANE):
        scopes[pid] = scope

    def _short_world(config=Config()):
        state = {"now": time.time(), "pressure": 0.0, "used": 4 * GiB, "programs": []}
        d = make_daemon(
            config=config,
            clock=lambda: state["now"],
            meminfo=lambda: (16 * GiB, state["used"]),
            pressure=lambda: state["pressure"],
            programs=lambda minimum: list(state["programs"]),
        )
        time.sleep(0.05)  # processes spawned from here on start after the guard did

        def run(seconds, pressure=60.0, used=15 * GiB):
            state.update(pressure=pressure, used=used)
            for _ in range(int(seconds // 6)):
                state["now"] += 6
                d.tick()

        return d, state, run, place

    return _short_world


def test_when_memory_runs_short_the_guard_stops_the_largest_program_outside_dibs(short_world, spawned):
    # Issue #135: a program an agent ran outside dibs grew to 14.1G and ran the machine out of RAM and
    # swap. The kernel killed it after a minute and a half of thrashing, and the agent's pane with it.
    d, state, run, place = short_world()
    small = spawned(SLEEPER, env=CLEAN_ENV)
    big = spawned(SLEEPER, env=CLEAN_ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    place(small.pid), place(big.pid)
    state["programs"] = [(small.pid, 2 * GiB), (big.pid, 9 * GiB)]
    run(6)
    assert procs.alive(big.pid)  # not before the shortage has lasted 10 s
    run(12)
    assert big.wait(timeout=5) == -9 and procs.alive(small.pid)  # one per stretch of shortage
    out = big.stdout.read().decode()
    assert "it held 9G of RAM outside dibs while the machine was running out of memory (1G of 16G left)" in out
    assert f"dibs run --cpus N --mem 10G --time ESTIMATE -- {PY} -c" in out
    assert [(e["kind"], e["pid"], e["ram"]) for e in d.store.events(0)] == [("stop", big.pid, 9 * GiB)]
    assert "(9G of RAM while memory ran short)" in render_queue(d.snapshot(), color=False)


@pytest.mark.parametrize(
    "held,pressure,used",
    [
        (GiB, 60.0, 15 * GiB),  # less than a tenth of the RAM: not the cause
        (9 * GiB, 60.0, 8 * GiB),  # half the RAM available: one program stalling inside itself (#129)
        (9 * GiB, 5.0, 15 * GiB),  # little available, but nothing waits for memory
    ],
)
def test_a_big_program_is_left_alone_unless_the_machine_is_short(short_world, spawned, held, pressure, used):
    d, state, run, place = short_world()
    p = spawned(SLEEPER, env=CLEAN_ENV)
    place(p.pid)
    state["programs"] = [(p.pid, held)]
    run(30, pressure=pressure, used=used)
    assert procs.alive(p.pid) and d.store.events(0) == []


def test_the_agent_itself_and_desktop_programs_are_never_stopped_for_ram(short_world, spawned, tmp_path):
    d, state, run, place = short_world()
    (tmp_path / "claude").write_text("import time\ntime.sleep(60)\n")  # an agent CLI run by its interpreter
    agent = spawned([PY, str(tmp_path / "claude")], env=CLEAN_ENV)
    browser = spawned(["sleep", "61"], env=CLEAN_ENV)
    script = spawned(SLEEPER, env=CLEAN_ENV)
    place(agent.pid), place(script.pid), place(browser.pid, "app-gnome-org.mozilla.firefox-7731.scope")
    state["programs"] = [(agent.pid, 12 * GiB), (browser.pid, 10 * GiB), (script.pid, 3 * GiB)]
    run(18)
    assert script.wait(timeout=5) == -9
    assert procs.alive(agent.pid) and procs.alive(browser.pid)


def test_a_job_is_never_stopped_for_ram_by_the_guard(short_world, tmp_path):
    # A job holds the RAM it reserved, and its own limit stops it if it goes over.
    d, state, run, place = short_world()
    resp = d.submit(
        {
            "argv": SLEEPER,
            "cwd": str(tmp_path),
            "env": CLEAN_ENV,
            "session": "s",
            "request": {"cpus": 1, "mem": GiB, "vram": 0, "gpus": None, "time": 60},
        }
    )
    job = d.store.get_job(resp["job"]["id"])
    pid = wait_for(lambda: next((p for p in launcher.job_pids(job) if (procs.cmdline(p) or [""])[0] == PY), None))
    state["programs"] = [(pid, 9 * GiB)]
    run(18)
    assert procs.alive(pid) and d.store.events(0) == []


def test_in_warn_mode_a_big_program_is_only_logged(short_world, spawned):
    d, state, run, place = short_world(Config(guard="warn"))
    p = spawned(SLEEPER, env=CLEAN_ENV)
    place(p.pid)
    state["programs"] = [(p.pid, 9 * GiB)]
    run(18)
    assert procs.alive(p.pid) and d.store.events(0) == []


@pytest.mark.parametrize(
    "scope,stopped",
    [
        # where agents work: tmux and VTE terminal panes
        (PANE, True),
        ("app-org.gnome.Terminal.slice/vte-spawn-1b2c3d4e.scope", True),
        # the rest of the desktop, which runs in scopes and services of its own, however new or big
        ("snap.chromium.chromium-776a4be7-f856-402c-827c-556237ea8bf6.scope", False),
        ("org.gnome.Shell@wayland.service", False),
        ("tracker-miner-fs-3.service", False),
        ("app-gnome-code-insiders-5120.scope", False),  # an editor itself, not a program an agent started
    ],
)
def test_only_programs_where_agents_run_are_stopped_for_ram(short_world, spawned, scope, stopped):
    d, state, run, place = short_world()
    p = spawned(SLEEPER, env=CLEAN_ENV)
    place(p.pid, scope)
    state["programs"] = [(p.pid, 9 * GiB)]
    run(18)
    if stopped:
        assert p.wait(timeout=5) == -9
    else:
        assert procs.alive(p.pid) and d.store.events(0) == []


def test_a_program_an_agent_started_in_an_editors_terminal_is_stopped_for_ram(short_world, spawned, tmp_path):
    d, state, run, place = short_world()
    (tmp_path / "claude").write_text(
        f"import subprocess, time\nsubprocess.Popen([{PY!r}, '-c', 'import time; time.sleep(60)'])\ntime.sleep(60)\n"
    )
    agent = spawned([PY, str(tmp_path / "claude")], env=CLEAN_ENV)
    child = wait_for(lambda: next(iter(procs.descendants(agent.pid)), None))
    exec_done(child, PY)
    editor = "app-gnome-code-insiders-5120.scope"
    place(agent.pid, editor), place(child, editor)
    state["programs"] = [(child, 9 * GiB)]
    run(18)
    wait_for(lambda: not procs.alive(child))
    assert procs.alive(agent.pid)


def test_the_agent_is_told_of_a_ram_stop_at_its_next_dibs_command(short_world, spawned):
    # The note in a stopped program's output can be lost: that night's loop piped the program's output
    # through grep. The agent's session gets it again, once, from its next dibs command.
    d, state, run, place = short_world()
    p = spawned(SLEEPER, env={**CLEAN_ENV, "CLAUDE_CODE_SESSION_ID": "agent-1"})
    place(p.pid)
    state["programs"] = [(p.pid, 9 * GiB)]
    run(18)
    p.wait(timeout=5)
    told = d.notices("agent-1", state["now"])["notices"]
    assert len(told) == 1 and "it held 9G of RAM outside dibs" in told[0]["message"]
    assert d.notices("agent-1", state["now"])["notices"] == []  # once
    assert d.notices("someone-else", state["now"])["notices"] == []


def test_the_agent_is_told_of_a_gpu_stop_at_its_next_dibs_command(world, spawned):
    d, backend = world()
    p = spawned(SLEEPER, env={**CLEAN_ENV, "CLAUDE_CODE_SESSION_ID": "agent-2"})
    backend.procs = [GpuProcess(p.pid, 0, int(3.1 * GiB))]
    d.tick()
    p.wait(timeout=5)
    told = d.notices("agent-2", time.time())["notices"]
    assert len(told) == 1 and "it used 3.1G on GPU 0 outside dibs" in told[0]["message"]


def test_a_program_an_agent_runs_under_is_never_stopped_for_ram(short_world, spawned, tmp_path):
    # Issue #141: stopping a program kills everything it started. The tmux server sits in a terminal
    # tab's scope and starts every pane, so stopping it would kill every agent on the machine.
    d, state, run, place = short_world()
    (tmp_path / "claude").write_text("import time\ntime.sleep(60)\n")
    server = spawned(
        [PY, "-c", f"import subprocess, time; subprocess.Popen([{PY!r}, {str(tmp_path / 'claude')!r}]); time.sleep(60)"],
        env=CLEAN_ENV,
    )
    agent = wait_for(lambda: next(iter(procs.descendants(server.pid)), None))
    wait_for(lambda: (procs.cmdline(agent) or ["", ""])[1:2] == [str(tmp_path / "claude")])
    script = spawned(SLEEPER, env=CLEAN_ENV)
    place(server.pid), place(agent), place(script.pid)
    state["programs"] = [(server.pid, 9 * GiB), (script.pid, 3 * GiB)]
    run(18)
    assert script.wait(timeout=5) == -9
    assert procs.alive(server.pid) and procs.alive(agent)
    procs.stop_with_message([agent], None)


def test_ram_stop_message_says_why_and_how_to_keep_it_from_happening_again():
    text = guard.ram_stop_message(4121, ["python", "prepare.py", "--shards", "8"], int(14.1 * GiB),
                                  int(0.8 * GiB), int(30.6 * GiB))  # fmt: skip
    assert text == (
        "dibs: stopped `python prepare.py --shards 8` (pid 4121): it held 14.1G of RAM outside dibs while "
        "the machine was running out of memory (819M of 30.6G left).\n"
        "dibs: Programs outside dibs get no RAM set aside, so left running it would have stalled every program "
        "here until the kernel killed it, or something else.\n"
        "dibs: So it doesn't happen again, run it as a job: a job gets the RAM it reserves.\n"
        "dibs:   dibs run --cpus N --mem 16G --time ESTIMATE -- python prepare.py --shards 8\n"
        "dibs: Run every step that may need more than a few GB of RAM through dibs, not only GPU work. "
        "`dibs agent-help` explains how to choose N, --mem and ESTIMATE."
    )



# Telling an agent about CPU-heavy work in its terminal (#158) ---------------------------------


def test_an_agent_is_told_when_a_program_in_its_terminal_keeps_many_cores_busy(make_daemon, spawned, tmp_path, monkeypatch):
    # A program run in an agent's terminal kept 13-15 cores busy outside dibs, and nothing told the agent.
    state = {"now": time.time(), "usec": 0}
    pane = "/app.slice/app-org.gnome.Terminal.slice/tmux-spawn-f3cc.scope"
    d = make_daemon(clock=lambda: state["now"], panes=lambda: {pane: state["usec"]})
    busy = spawned(SLEEPER, env={**CLEAN_ENV, "CLAUDE_CODE_SESSION_ID": "agent-9"})
    scope = tmp_path / "cgroup" / pane.lstrip("/")
    scope.mkdir(parents=True)
    (scope / "cgroup.procs").write_text(f"{busy.pid}\n")
    monkeypatch.setattr(procs, "CGROUP_ROOT", str(tmp_path / "cgroup"))
    cores = len(d.capacity.cpus)

    def busy_for(seconds):
        for _ in range(seconds // 15):  # half the machine's cores busy
            state["now"] += 15
            state["usec"] += int(15 * 1e6 * cores / 2)
            d.tick()

    busy_for(60)
    assert d.notices("agent-9", state["now"])["notices"] == []  # not on a minute's burst
    busy_for(90)
    told = d.notices("agent-9", state["now"])["notices"]
    assert len(told) == 1, told
    assert f"has kept about {cores // 2} cores busy" in told[0]["message"]
    assert f"dibs run --cpus {cores // 2} --mem SIZE --time ESTIMATE -- {PY}" in told[0]["message"]
    assert procs.alive(busy.pid)  # nothing is stopped
    busy_for(150)
    assert d.notices("agent-9", state["now"])["notices"] == []  # once an hour at most


def test_a_terminal_busy_on_a_core_or_two_is_left_alone(make_daemon):
    state = {"now": time.time(), "usec": 0}
    d = make_daemon(clock=lambda: state["now"], panes=lambda: {"/app.slice/tmux-spawn-a.scope": state["usec"]})
    for _ in range(12):
        state["now"] += 15
        state["usec"] += int(15 * 1e6 * 2)  # two cores
        d.tick()
    assert [e for e in d.store.events(0) if e["kind"] == "heavy"] == []
