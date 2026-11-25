"""Carrying GPU jobs across a suspend (#293).

On this machine GPU work that is running when the machine sleeps doesn't come back: the NVIDIA driver's save
of GPU memory fails, and the job's process spins on with its GPU memory held and the GPU idle (#292). A sleep
taken with the GPU holding nothing wakes cleanly. So while jobs hold GPU memory, dibs holds a "delay" sleep
lock from logind; when logind says the machine is about to sleep, dibs pauses each of those processes' CUDA
state (cudapause: their GPU memory moves into their RAM and the GPU is released), records them, and lets the
sleep go on. After the wake it restores them. Whatever is on record as paused is restored at the next look
while the machine is awake, so a missed wake, a sleep called off or a dibs restart still ends in a restore.

Only the real daemon makes one of these (it needs `gdbus` and `systemd-inhibit`); tests use fakes."""

import json
import logging
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

from dibs.cudapause import GONE as cudapause_gone

log = logging.getLogger("dibs")

PAUSE_FOR = 20.0  # seconds dibs gives itself to pause GPU jobs before letting the sleep go on (logind waits 30 here)
RESUME_FOR = 60.0  # seconds a restore may take
RESUME_EVERY = 5.0  # seconds between tries at a restore that failed: the NVIDIA driver wakes 0-9 s after the machine
RESUME_GIVE_UP = 300.0  # seconds of failed restores after which the job's session is told and the jobs start again
WAKE_MISSED = 90.0  # seconds awake after the sleep signal with no wake signal: the sleep didn't happen, or the wake was missed
MONITOR = ["gdbus", "monitor", "--system", "--dest", "org.freedesktop.login1", "--object-path", "/org/freedesktop/login1"]
LOCK = ["systemd-inhibit", "--what=sleep", "--mode=delay", "--who=dibs",
        "--why=dibs pauses its GPU jobs so that they carry on after the sleep", "cat"]  # fmt: skip


def sleep_signal(line: str) -> bool | None:
    """What a line of `gdbus monitor` says: True, the machine is about to sleep; False, it woke (or the sleep
    was called off); None, something else."""
    if ".PrepareForSleep (" not in line:
        return None
    return True if "(true" in line else False if "(false" in line else None


def read_limit(pid: int) -> tuple[str, str] | None:
    """The systemd unit of a process's dibs job scope and its RAM limit (memory.max: bytes, or "max"); None
    for a process outside one, such as a program that left its job's scope (#265)."""
    try:
        cgroup = Path(f"/proc/{pid}/cgroup").read_text().strip().split("::", 1)[1]
        name = cgroup.rsplit("/", 1)[-1]
        if not (name.startswith("dibs-") and name.endswith(".scope")):
            return None
        return cgroup.rsplit("/", 1)[-1], Path(f"/sys/fs/cgroup{cgroup}/memory.max").read_text().strip()
    except (OSError, IndexError):
        return None


def set_limit(unit: str, value: str) -> bool:
    """Set a scope's RAM limit, as dibs does when a job grows (launcher.set_memory_max)."""
    try:
        return subprocess.run(["systemctl", "--user", "set-property", "--runtime", unit, f"MemoryMax={value}"],
                              capture_output=True, timeout=10).returncode == 0  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return False


class Suspend:
    """Holds the sleep lock while jobs hold GPU memory, and pauses and restores their CUDA processes around a
    sleep. `pauser(op, seconds, pids)` is cudapause's "pause" or "resume", run apart from the daemon."""

    def __init__(self, paused_file: Path, pauser: Callable[[str, float, list[int]], dict[int, str | None]],
                 spawn: Callable = subprocess.Popen, read_limit: Callable[[int], tuple[str, str] | None] = lambda pid: None,
                 set_limit: Callable[[str, str], bool] = lambda unit, value: False) -> None:  # fmt: skip
        self.paused_file, self.pauser, self.spawn = paused_file, pauser, spawn
        self.read_limit, self.set_limit = read_limit, set_limit
        self.gave_up: Callable[[dict[int, str]], None] = lambda failed: None  # job id -> why it couldn't be restored
        self._resuming: float | None = None  # monotonic time of the first try at restoring what's on record
        self._tried: float = -float("inf")
        self._mutex = threading.Lock()
        self._gpu_pids: dict[int, int] = {}  # pid -> job id, as of the daemon's last look
        self._lock: subprocess.Popen | None = None
        self._monitor: subprocess.Popen | None = None
        self._sleeping: float | None = None  # monotonic time of the sleep signal, until the wake
        self._paused, self._limits = self._read()  # pid -> job; scope unit -> its RAM limit before the pause

    def start(self) -> None:
        """Listen for logind's sleep signal (the real daemon only)."""
        self._keep_monitor()

    def tick(self, gpu_pids: dict[int, int]) -> None:
        """At each look: hold the lock while jobs hold GPU memory, restore what a missed wake or a restart left
        paused, and keep listening. Never waits on a pause under way."""
        if not self._mutex.acquire(blocking=False):
            return
        try:
            self._gpu_pids = dict(gpu_pids)
            if self._sleeping is not None and time.monotonic() - self._sleeping < WAKE_MISSED:
                return  # between the sleep signal and the wake: leave things as they are
            self._sleeping = None
            if self._paused and time.monotonic() - self._tried >= RESUME_EVERY:
                failed = self._resume("a missed wake, a restart or a failed restore")
                if failed and time.monotonic() - (self._resuming or time.monotonic()) >= RESUME_GIVE_UP:
                    self._give_up(failed)
            if self._gpu_pids and not self._held():
                # It holds the lock while `cat` reads dibs's end of a pipe: when dibs dies, the lock goes too.
                self._lock = self.spawn(LOCK, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif not self._gpu_pids:
                self._release()
            if self._monitor is not None:
                self._keep_monitor()
        finally:
            self._mutex.release()

    def on_line(self, line: str) -> None:
        said = sleep_signal(line)
        if said is not None:
            log.info("logind: the machine %s", "is about to sleep" if said else "woke")
        if said is True:
            self._before_sleep()
        elif said is False:
            self._after_wake()

    def close(self) -> None:
        """Let go of the lock and stop listening, when dibs stops."""
        with self._mutex:
            self._release()
            if self._monitor is not None and self._monitor.poll() is None:
                self._monitor.terminate()

    def paused_jobs(self) -> set[int]:
        """The jobs whose CUDA processes are paused now."""
        return set(self._paused.values())

    def _before_sleep(self) -> None:
        with self._mutex:
            if self._sleeping is not None:
                return
            self._sleeping = time.monotonic()
            pids = dict(self._gpu_pids)
            try:
                self._pause(pids)
            finally:
                self._release()  # whatever happened, the sleep goes on

    def _pause(self, pids: dict[int, int]) -> None:
        if not pids:
            return
        # A paused process's GPU memory moves into its RAM, which counts against its limit: lifted for the pause,
        # so nothing spills into swap just before the sleep, and put back once the job is restored.
        limits = {found[0]: (found[1], pids[pid]) for pid in pids if (found := self.read_limit(pid))}
        for unit in limits:
            self.set_limit(unit, "infinity")
        self._paused, self._limits = dict(pids), limits  # on record first: resuming finishes whatever a pause left
        self._write({})
        results = self.pauser("pause", PAUSE_FOR, sorted(pids))
        self._write(results)
        failed = {pid: error for pid, error in results.items() if error}
        log.info("paused the GPU work of jobs %s before the sleep in %.1f s%s", sorted(set(pids.values())),
                 time.monotonic() - (self._sleeping or time.monotonic()), f"; failed: {failed}" if failed else "")  # fmt: skip

    def _after_wake(self) -> None:
        with self._mutex:
            self._sleeping = None
            if self._paused:
                failed = self._resume("the wake")
                if failed:
                    log.info("will try again every %.0f s: the GPU driver may still be waking", RESUME_EVERY)

    def _resume(self, why: str) -> dict[int, str]:
        """Restore what's on record; keep on record, with its RAM limit still lifted, what didn't come back.
        What failed, by pid."""
        began = self._tried = time.monotonic()
        if self._resuming is None:
            self._resuming = began
        results = self.pauser("resume", RESUME_FOR, sorted(self._paused))
        failed = {pid: error for pid, error in results.items() if error and error != cudapause_gone}
        log.info("restored the GPU work of jobs %s after %s in %.1f s%s", sorted(set(self._paused.values())), why,
                 time.monotonic() - began, f"; failed: {failed}" if failed else "")  # fmt: skip
        stuck = {self._paused[pid] for pid in failed}
        for unit, (value, job) in list(self._limits.items()):
            if job not in stuck:
                self.set_limit(unit, "infinity" if value == "max" else value)
                del self._limits[unit]
        self._paused = {pid: job for pid, job in self._paused.items() if pid in failed}
        if self._paused:
            self._write(failed)
        else:
            self._resuming = None
            self.paused_file.unlink(missing_ok=True)
        return failed

    def _give_up(self, failed: dict[int, str]) -> None:
        """Stop trying: tell the jobs' sessions, put their limits back, and let jobs start again."""
        by_job: dict[int, str] = {}
        for pid, error in failed.items():
            by_job.setdefault(self._paused[pid], error)
        log.info("gave up restoring the GPU work of jobs %s after %.0f s: %s", sorted(by_job), RESUME_GIVE_UP, failed)
        for unit, (value, _) in self._limits.items():
            self.set_limit(unit, "infinity" if value == "max" else value)
        self._paused, self._limits, self._resuming = {}, {}, None
        self.paused_file.unlink(missing_ok=True)
        self.gave_up(by_job)

    def _held(self) -> bool:
        return self._lock is not None and self._lock.poll() is None

    def _release(self) -> None:
        if self._lock is not None:
            if self._lock.stdin is not None:
                self._lock.stdin.close()
            self._lock.terminate()
            try:
                self._lock.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._lock.kill()
            self._lock = None

    def _keep_monitor(self) -> None:
        if self._monitor is not None and self._monitor.poll() is None:
            return
        try:
            self._monitor = self.spawn(MONITOR, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL, text=True)  # fmt: skip
        except OSError as e:
            log.warning("can't listen for the machine's sleep: %s", e)
            return
        threading.Thread(target=self._listen, args=(self._monitor,), daemon=True).start()

    def _listen(self, monitor: subprocess.Popen) -> None:
        for line in monitor.stdout:
            try:
                self.on_line(line)
            except Exception:  # noqa: BLE001 - the listener must outlive any one signal's trouble
                log.exception("handling the machine's sleep signal failed")

    def _read(self) -> tuple[dict[int, int], dict[str, str]]:
        try:
            record = json.loads(self.paused_file.read_text())
            limits = {unit: (value, job) for unit, (value, job) in record.get("limits", {}).items()}
            return {int(pid): job for pid, job in record["pids"].items()}, limits
        except (OSError, ValueError, KeyError):
            return {}, {}

    def _write(self, results: dict[int, str | None]) -> None:
        record = {"pids": {str(pid): job for pid, job in self._paused.items()},
                  "limits": {unit: [value, job] for unit, (value, job) in self._limits.items()},
                  "time": time.time(), "results": {str(pid): error for pid, error in results.items()}}  # fmt: skip
        tmp = self.paused_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(record))
        tmp.replace(self.paused_file)
