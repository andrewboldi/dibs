"""The launcher that runs one job: python -m dibs.shim JOB_DIR.

It pins itself to the job's CPUs, starts the command with the job's environment and working
directory, waits for it, stops anything the command left behind, and writes the result to
JOB_DIR/exit.json. It must outlive the command to report, so after starting it, it ignores the
signals used to stop jobs.
"""

import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from dibs import procs

_PR_SET_CHILD_SUBREAPER = 36


def _become_subreaper() -> None:
    """Orphaned descendants of the job get re-parented to us, so we can find and stop them."""
    try:
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    except (OSError, AttributeError):
        pass


def _cgroup_file(name: str) -> str | None:
    cg = procs.cgroup(os.getpid())
    if not cg:
        return None
    try:
        with open(f"/sys/fs/cgroup{cg}/{name}") as f:
            return f.read()
    except OSError:
        return None


def _stop_leftovers(in_scope: bool) -> None:
    me = os.getpid()
    if in_scope:
        listing = _cgroup_file("cgroup.procs") or ""
        others = [int(p) for p in listing.split() if int(p) != me]
    else:
        others = procs.descendants(me)
    for pid in others:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid == 0:
            if not any(procs.alive(p) for p in others):
                break
            time.sleep(0.02)


def _wait_reaping(main_pid: int) -> int:
    """Wait for the command, reaping every orphan handed to us meanwhile (we're their subreaper);
    otherwise they'd pile up as zombies for as long as the job runs."""
    while True:
        pid, status = os.waitpid(-1, 0)
        if pid == main_pid:
            return os.waitstatus_to_exitcode(status)


def _write_exit(job_dir: Path, result: dict) -> None:
    tmp = job_dir / "exit.json.tmp"
    with open(tmp, "w") as f:
        json.dump(result, f)
    os.replace(tmp, job_dir / "exit.json")


def main(argv: list[str]) -> int:
    job_dir = Path(argv[0])
    with open(job_dir / "spec.json") as f:
        spec = json.load(f)
    _become_subreaper()
    if spec["cpu_ids"] and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, spec["cpu_ids"])
        except OSError:
            pass

    exit_code, reason = 127, None
    try:
        out = os.open(spec["output"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    except OSError as e:
        out, reason = None, f"couldn't start: can't open output file: {e}"
    if out is not None and not os.path.isdir(spec["cwd"]):
        reason = f"couldn't start: working directory {spec['cwd']} no longer exists"
    elif out is not None:
        try:
            command = subprocess.Popen(
                spec["argv"],
                cwd=spec["cwd"],
                env=spec["env"],
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                process_group=0,
            )
        except OSError as e:
            reason = f"couldn't start: {e}"
        else:
            # Only now: ignored signals would otherwise be inherited by the command.
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                signal.signal(sig, signal.SIG_IGN)
            code = _wait_reaping(command.pid)
            command.returncode = code
            exit_code = code if code >= 0 else 128 - code
    if out is not None:
        os.close(out)

    in_scope = bool(spec.get("unit"))
    oom_killed, peak_mem, cpu_seconds = False, None, None
    if in_scope:
        usage = procs.keyed_numbers(_cgroup_file("cpu.stat")).get("usage_usec")
        cpu_seconds = usage / 1e6 if usage is not None else None
        for line in (_cgroup_file("memory.events") or "").splitlines():
            key, _, count = line.partition(" ")
            if key == "oom_kill" and count.strip() not in ("", "0"):
                oom_killed = True
        peak = _cgroup_file("memory.peak")
        if peak and peak.strip().isdigit():
            # memory.peak counts file cache; take off what's still cached, like the daemon's samples
            peak_mem = max(0, int(peak) - procs.keyed_numbers(_cgroup_file("memory.stat")).get("file", 0))
    _stop_leftovers(in_scope)
    _write_exit(
        job_dir,
        {
            "exit_code": exit_code,
            "ended": time.time(),
            "oom_killed": oom_killed,
            "peak_mem": peak_mem,
            "cpu_seconds": cpu_seconds,
            "reason": reason,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
