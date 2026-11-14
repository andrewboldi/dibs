"""The guard: GPU processes that aren't dibs jobs are stopped, with a note saying how to use dibs.

It watches the GPUs themselves, not the agents, so it works whichever agent CLI started the
process, including ones dibs has never heard of.
"""

import os
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dibs import procs
from dibs.gpu import GpuProcess
from dibs.units import format_size, suggest_size

WHY = {
    "allowed": "allowlisted",
    "other-user": "another user's",
    "preexisting": "already running when the guard started",
    "warn": "warn mode",
}


@dataclass(frozen=True)
class Verdict:
    pid: int
    gpu_index: int
    used: int | None
    kind: str  # job, stop, warn, allowed, other-user, preexisting
    job_id: int | None


def _allowed(pid: int, allow: Sequence[str]) -> bool:
    if not allow:
        return False
    names = {procs.exe_name(pid) or ""}
    argv = procs.cmdline(pid)
    if argv:
        names.add(os.path.basename(argv[0]))
    return bool(names & set(allow))


def classify(
    gpu_procs: Sequence[GpuProcess],
    *,
    job_of_pid: Mapping[int, int],
    mode: str,
    allow: Sequence[str],
    since: float,
    my_uid: int,
) -> list[Verdict]:
    verdicts = []
    for p in gpu_procs:
        if p.pid in job_of_pid:
            verdicts.append(Verdict(p.pid, p.gpu_index, p.used, "job", job_of_pid[p.pid]))
            continue
        if mode == "off":
            continue
        owner = procs.uid(p.pid)
        started = procs.start_time(p.pid)
        if owner is None or started is None:
            continue  # already gone
        if owner != my_uid:
            kind = "other-user"
        elif _allowed(p.pid, allow):
            kind = "allowed"
        elif started < since:
            kind = "preexisting"
        else:
            kind = "stop" if mode == "all" else "warn"
        verdicts.append(Verdict(p.pid, p.gpu_index, p.used, kind, None))
    return verdicts


def stop_message(pid: int, argv: Sequence[str], gpu_index: int, used: int | None, rss: int | None) -> str:
    command = shlex.join(argv) if argv else "this process"
    usage = f"{format_size(used)} on GPU {gpu_index}" if used else f"GPU {gpu_index}"
    mem = format_size(suggest_size(rss)) if rss else "SIZE"
    vram = format_size(suggest_size(used)) if used else "SIZE"
    return (
        f"dibs: stopped `{command}` (pid {pid}): it used {usage} outside dibs.\n"
        "dibs: GPUs on this machine are shared through dibs. Run it as a job instead:\n"
        f"dibs:   dibs run --cpus N --mem {mem} --vram {vram} --time ESTIMATE -- {command}\n"
        "dibs: `dibs agent-help` explains how to choose N and ESTIMATE."
    )


def enforce(verdicts: Sequence[Verdict]) -> list[dict]:
    """Stop every process judged "stop" (and what it started), each once; returns one event per stop."""
    events, done = [], set()
    for v in verdicts:
        if v.kind != "stop" or v.pid in done:
            continue
        done.add(v.pid)
        argv = procs.cmdline(v.pid) or []
        env = procs.environ(v.pid) or {}
        message = stop_message(v.pid, argv, v.gpu_index, v.used, procs.rss(v.pid))
        procs.freeze_write_kill([v.pid, *procs.descendants(v.pid)], message)
        events.append(
            {
                "pid": v.pid,
                "command": argv,
                "gpu": v.gpu_index,
                "used": v.used,
                "session": env.get("DIBS_SESSION") or env.get("CLAUDE_CODE_SESSION_ID"),
            }
        )
    return events
