"""Carrying GPU jobs across a suspend (#293). On this machine GPU work running when the machine sleeps doesn't
come back (#292): dibs pauses the jobs' CUDA state before the sleep, through the driver's checkpoint API, and
restores it after the wake. Nothing here loads the real driver or touches logind."""

import json

import pytest

from dibs import cudapause


class FakeDriver:
    """The CUDA driver's checkpoint calls, on made-up processes: pid -> state (0 running, 1 locked,
    2 checkpointed); `fail` makes one call return an error for one pid."""

    def __init__(self, states, fail=None):
        self.states, self.fail, self.calls = dict(states), fail or {}, []

    def _call(self, name, pid, to=None, needs=None):
        self.calls.append((name, pid))
        if self.fail.get(pid) == name:
            return 999
        if pid not in self.states or (needs is not None and self.states[pid] not in needs):
            return 1
        if to is not None:
            self.states[pid] = to
        return 0

    def cuCheckpointProcessGetState(self, pid, out):
        if pid not in self.states:
            return 1
        out._obj.value = self.states[pid]
        return 0

    def cuCheckpointProcessLock(self, pid, args):
        return self._call("lock", pid, to=1, needs=(0,))

    def cuCheckpointProcessCheckpoint(self, pid, args):
        return self._call("checkpoint", pid, to=2, needs=(1,))

    def cuCheckpointProcessRestore(self, pid, args):
        return self._call("restore", pid, to=1, needs=(2,))

    def cuCheckpointProcessUnlock(self, pid, args):
        return self._call("unlock", pid, to=0, needs=(1,))


def test_pausing_locks_a_process_then_moves_its_gpu_memory_out_and_resuming_undoes_both():
    drv = FakeDriver({10: 0, 11: 0})
    assert cudapause.run("pause", 5, [10, 11], lib=drv) == {10: None, 11: None}
    assert drv.states == {10: 2, 11: 2}
    assert cudapause.run("resume", 5, [10, 11], lib=drv) == {10: None, 11: None}
    assert drv.states == {10: 0, 11: 0}


def test_a_pause_that_fails_after_the_lock_unlocks_again():
    drv = FakeDriver({10: 0}, fail={10: "checkpoint"})
    assert cudapause.run("pause", 5, [10], lib=drv) == {10: "checkpoint: CUDA error 999"}
    assert drv.states == {10: 0} and drv.calls[-1] == ("unlock", 10)


@pytest.mark.parametrize("state", [0, 1, 2])  # never paused, locked only, fully paused
def test_resuming_finishes_from_whatever_step_a_pause_got_to(state):
    drv = FakeDriver({10: state})
    assert cudapause.run("resume", 5, [10], lib=drv) == {10: None}
    assert drv.states == {10: 0}


def test_a_process_that_is_gone_or_not_cuda_is_reported_not_raised(capsys, monkeypatch):
    drv = FakeDriver({})
    assert cudapause.run("resume", 5, [42], lib=drv) == {42: "not a CUDA process, or gone"}
    monkeypatch.setattr(cudapause, "driver", lambda: (_ for _ in ()).throw(OSError("no CUDA driver")))
    assert cudapause.main(["pause", "5", "7"]) == 0
    assert json.loads(capsys.readouterr().out) == {"7": "no CUDA driver"}


from dibs import suspend  # noqa: E402

SLEEP = "/org/freedesktop/login1: org.freedesktop.login1.Manager.PrepareForSleep (true,)"
WAKE = "/org/freedesktop/login1: org.freedesktop.login1.Manager.PrepareForSleep (false,)"


def test_logind_saying_the_machine_sleeps_or_woke_is_read_from_gdbus_monitor():
    assert suspend.sleep_signal(SLEEP) is True and suspend.sleep_signal(WAKE) is False
    assert suspend.sleep_signal("/org/freedesktop/login1: org.freedesktop.login1.Manager.PrepareForShutdown (true,)") is None
    assert suspend.sleep_signal("Monitoring signals on object /org/freedesktop/login1 owned by org.freedesktop.login1") is None


class FakeProc:
    def __init__(self, argv):
        self.argv, self.ended, self.stdin = argv, None, None

    def poll(self):
        return self.ended

    def terminate(self):
        self.ended = -15

    def kill(self):
        self.ended = -9

    def wait(self, timeout=None):
        return self.ended


def _carrier(tmp_path, results=None):
    """A Suspend with fakes for the lock holder and the pauser; `calls` records the pauser's."""
    spawned, calls = [], []

    def spawn(argv, **kw):
        spawned.append(FakeProc(argv))
        return spawned[-1]

    def pauser(op, seconds, pids):
        calls.append((op, sorted(pids)))
        return {pid: (results or {}).get((op, pid)) for pid in pids}

    s = suspend.Suspend(tmp_path / "paused.json", pauser, spawn=spawn)
    return s, spawned, calls


def _lock(spawned):
    held = [p for p in spawned if p.argv[0] == "systemd-inhibit" and p.ended is None]
    return held[0] if held else None


def test_dibs_holds_a_sleep_delay_lock_only_while_jobs_hold_gpu_memory(tmp_path):
    s, spawned, _ = _carrier(tmp_path)
    s.tick({})
    assert _lock(spawned) is None  # nothing to carry: no suspend waits on dibs
    s.tick({100: 7})
    lock = _lock(spawned)
    assert lock.argv[:4] == ["systemd-inhibit", "--what=sleep", "--mode=delay", "--who=dibs"]
    s.tick({100: 7})
    assert _lock(spawned) is lock  # one lock, kept
    s.tick({})
    assert _lock(spawned) is None and lock.ended is not None


def test_jobs_are_paused_before_the_sleep_and_restored_after_the_wake(tmp_path):
    s, spawned, calls = _carrier(tmp_path)
    s.tick({100: 7, 101: 8})
    s.on_line(SLEEP)
    assert calls == [("pause", [100, 101])]
    assert s.paused_jobs() == {7, 8} and _lock(spawned) is None  # paused, then the sleep goes on
    s.tick({100: 7, 101: 8})
    assert _lock(spawned) is None and calls == [("pause", [100, 101])]  # nothing more until the wake
    s.on_line(WAKE)
    assert calls[-1] == ("resume", [100, 101]) and s.paused_jobs() == set()
    assert not (tmp_path / "paused.json").exists()
    s.tick({100: 7, 101: 8})
    assert _lock(spawned) is not None  # held again for the next sleep


def test_a_pause_that_failed_is_still_undone_at_the_wake(tmp_path):
    s, spawned, calls = _carrier(tmp_path, results={("pause", 101): "lock: CUDA error 2"})
    s.tick({100: 7, 101: 8})
    s.on_line(SLEEP)
    s.on_line(WAKE)
    assert calls[-1] == ("resume", [100, 101])  # resuming finishes whatever a pause left


def test_a_sleep_with_no_gpu_jobs_pauses_nothing(tmp_path):
    s, spawned, calls = _carrier(tmp_path)
    s.tick({})
    s.on_line(SLEEP)
    s.on_line(WAKE)
    assert calls == [] and s.paused_jobs() == set()


def test_jobs_left_paused_by_a_missed_wake_or_a_restart_are_restored(tmp_path):
    # A dibs that stopped while jobs were paused, or a wake that never came through: whatever is on record
    # as paused is restored at the next look while the machine is awake.
    s, spawned, calls = _carrier(tmp_path)
    s.tick({100: 7})
    s.on_line(SLEEP)
    again, _, calls_again = _carrier(tmp_path)
    assert again.paused_jobs() == {7}
    again.tick({100: 7})
    assert calls_again == [("resume", [100])] and again.paused_jobs() == set()


class Recorder:
    """Stands in for the daemon's Suspend: records what each look reports, and says which jobs are paused."""

    def __init__(self):
        self.ticks, self.paused = [], set()

    def tick(self, gpu_pids):
        self.ticks.append(dict(gpu_pids))

    def paused_jobs(self):
        return set(self.paused)


def test_each_look_at_the_gpus_tells_which_jobs_cuda_processes_hold_gpu_memory(make_daemon, tmp_path):
    from conftest import FakeBackend, run_until
    from test_daemon import job, submit
    from test_idle_gpu import _end, _holding

    from dibs.gpu import GpuProcess
    from dibs.units import GiB

    backend, rec = FakeBackend(), Recorder()
    d = make_daemon(backend=backend, suspend=rec)
    go, holding = _holding(tmp_path)
    holder = submit(d, tmp_path, holding, vram=2 * GiB, session="s1")
    run_until(d, lambda: job(d, holder).shim_pid)
    pid = job(d, holder).shim_pid
    backend.procs = [GpuProcess(pid, 0, GiB), GpuProcess(999_999, 0, GiB)]  # the second is nobody's job
    d._watch_gpus(d.clock())
    assert rec.ticks[-1] == {pid: holder}
    _end(d, go, holder)


def test_no_job_starts_while_jobs_are_paused_for_a_sleep(make_daemon, tmp_path):
    from test_daemon import job, submit

    from dibs.model import State

    rec = Recorder()
    rec.paused = {1}
    d = make_daemon(suspend=rec)
    waiting = submit(d, tmp_path, ["true"], session="s1")
    d.schedule()
    assert job(d, waiting).state == State.QUEUED
    rec.paused = set()
    d.schedule()
    assert job(d, waiting).state != State.QUEUED
