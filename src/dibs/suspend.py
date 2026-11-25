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

log = logging.getLogger("dibs")

PAUSE_FOR = 20.0  # seconds dibs gives itself to pause GPU jobs before letting the sleep go on (logind waits 30 here)
RESUME_FOR = 60.0  # seconds a restore may take
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
            if self._paused:
                self._resume("a missed wake or a restart")
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
        if said is True:
            self._before_sleep()
        elif said is False:
            self._after_wake()

    def paused_jobs(self) -> set[int]:
        """The jobs whose CUDA processes are paused now."""
        return set(self._paused.values())

    def _before_sleep(self) -> None:
        with self._mutex:
            if self._sleeping is not None:
                return
            self._sleeping = time.monotonic()
            pids = dict(self._gpu_pids)
            if pids:
                # A paused process's GPU memory moves into its RAM, which counts against its limit: lifted for the
                # pause, so nothing spills into swap just before the sleep, and put back after the restore.
                limits = {found[0]: found[1] for pid in pids if (found := self.read_limit(pid))}
                for unit in limits:
                    self.set_limit(unit, "infinity")
                results = self.pauser("pause", PAUSE_FOR, sorted(pids))
                self._paused, self._limits = pids, limits  # the failed too: resuming finishes whatever a pause left
                self._write(results)
                failed = {pid: error for pid, error in results.items() if error}
                log.info("paused the GPU work of jobs %s before the sleep in %.1f s%s", sorted(set(pids.values())),
                         time.monotonic() - self._sleeping, f"; failed: {failed}" if failed else "")  # fmt: skip
            self._release()

    def _after_wake(self) -> None:
        with self._mutex:
            self._sleeping = None
            if self._paused:
                self._resume("the wake")

    def _resume(self, why: str) -> None:
        began = time.monotonic()
        results = self.pauser("resume", RESUME_FOR, sorted(self._paused))
        failed = {pid: error for pid, error in results.items() if error}
        log.info("restored the GPU work of jobs %s after %s in %.1f s%s", sorted(set(self._paused.values())), why,
                 time.monotonic() - began, f"; failed: {failed}" if failed else "")  # fmt: skip
        for unit, value in self._limits.items():
            self.set_limit(unit, "infinity" if value == "max" else value)
        self._paused, self._limits = {}, {}
        self.paused_file.unlink(missing_ok=True)

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
            self.on_line(line)

    def _read(self) -> tuple[dict[int, int], dict[str, str]]:
        try:
            record = json.loads(self.paused_file.read_text())
            return {int(pid): job for pid, job in record["pids"].items()}, dict(record.get("limits", {}))
        except (OSError, ValueError, KeyError):
            return {}, {}

    def _write(self, results: dict[int, str | None]) -> None:
        record = {"pids": {str(pid): job for pid, job in self._paused.items()}, "limits": self._limits,
                  "time": time.time(), "results": {str(pid): error for pid, error in results.items()}}  # fmt: skip
        tmp = self.paused_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(record))
        tmp.replace(self.paused_file)
