"""The guard: GPU processes that aren't dibs jobs are stopped, with a note saying how to use dibs.
So is the largest program outside dibs when the machine runs short of memory (#135).

It watches the GPUs and the machine's memory, not the agents, so it works whichever agent CLI
started the process, including ones dibs has never heard of.
"""

import os
import re
import shlex
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass

from dibs import procs
from dibs.agents import AGENTS
from dibs.gpu import GpuProcess
from dibs.units import format_size, suggest_size

WHY = {
    "allowed": "allowlisted",
    "desktop": "launched from the desktop",
    "other-user": "another user's",
    "other-dibs": "another dibs's job",
    "preexisting": "already running when the guard started",
    "inherited": "started by work that was already running when the guard started",
    "warn": "warn mode",
}


@dataclass(frozen=True)
class Verdict:
    pid: int
    gpu_index: int
    used: int | None
    kind: str  # job, stop, warn, allowed, desktop, other-user, other-dibs, preexisting, inherited
    job_id: int | None


def _allowed(pid: int, allow: Sequence[str]) -> bool:
    if not allow:
        return False
    names = {procs.exe_name(pid) or ""}
    argv = procs.cmdline(pid)
    if argv:
        names.add(os.path.basename(argv[0]))
    return bool(names & set(allow))


# Apps that host a shell, where agents run (terminals, editors with a built-in terminal, agent apps):
# their scopes are policed even when they were launched from the desktop.
_SHELL_HOSTS = re.compile(
    r"term|konsole|kitty|alacritty|foot|tilix|ghostty|ptyxis|kgx|console|guake|tilda|yakuake|rxvt|"
    r"wezterm|warp|contour|blackbox|sakura|code|codium|cursor|windsurf|zed|jetbrains|idea|pycharm|"
    r"clion|goland|webstorm|rider|studio\.android|emacs|vim|claude",
    re.IGNORECASE,
)


def _launched_from_desktop(pid: int) -> bool:
    """Whether the process runs in the scope a desktop launcher gives an app (app-gnome-<app>-<n>.scope,
    app-<app>@<n>.service), for an app that doesn't host shells. The cgroup, unlike the parent
    processes, survives nohup and setsid, so an agent's detached process is still its terminal's (#92)."""
    leaf = (procs.cgroup(pid) or "").rstrip("/").rsplit("/", 1)[-1]
    if not leaf.startswith("app-") or not leaf.endswith((".scope", ".service")):
        return False
    return not _SHELL_HOSTS.search(leaf)


def classify(
    gpu_procs: Sequence[GpuProcess],
    *,
    job_of_pid: Mapping[int, int],
    mode: str,
    allow: Sequence[str],
    since: float,
    my_uid: int,
    parents: Collection[tuple[int, int]] = (),
    instance: str = "",
    desktop: bool = True,
) -> list[Verdict]:
    """What to do about each GPU process. `parents` are (pid, start ticks) of programs whose GPU
    processes were already running when the guard started: what they start later is theirs too."""
    verdicts = []
    for p in gpu_procs:
        if p.pid in job_of_pid:
            verdicts.append(Verdict(p.pid, p.gpu_index, p.used, "job", job_of_pid[p.pid]))
            continue
        if mode == "off":
            continue
        kind = _judge(p.pid, mode, allow, since, my_uid, parents, instance, desktop)
        if kind is not None:
            verdicts.append(Verdict(p.pid, p.gpu_index, p.used, kind, None))
    return verdicts


def _judge(
    pid: int, mode: str, allow: Sequence[str], since: float, my_uid: int,
    parents: Collection[tuple[int, int]], instance: str, desktop: bool,
) -> str | None:  # fmt: skip
    """What to do about a process outside dibs: "stop", "warn", or why it's left alone (a WHY key);
    None if it's gone."""
    owner = procs.uid(pid)
    started = procs.start_time(pid)
    if owner is None or started is None:
        return None
    if owner != my_uid:
        return "other-user"
    if _other_dibs_job(pid, instance):
        return "other-dibs"
    if _allowed(pid, allow):
        return "allowed"
    if desktop and _launched_from_desktop(pid):
        return "desktop"
    if started < since:
        return "preexisting"
    if parents and _descends_from(pid, parents):
        return "inherited"
    return "stop" if mode == "all" else "warn"


AGENT_COMMANDS = frozenset(command for *_, command in AGENTS)


def _is_agent(pid: int) -> bool:
    """Whether the process is an agent CLI itself (claude, codex, ...), run directly or by an
    interpreter: stopping one is what the RAM guard is there to prevent."""
    argv = procs.cmdline(pid) or []
    names = {procs.exe_name(pid) or ""} | {os.path.basename(arg) for arg in argv[:2]}
    return bool(names & AGENT_COMMANDS)


def ram_hog(
    programs: Sequence[tuple[int, int]],
    *,
    minimum: int,
    in_dibs: Callable[[int], bool],
    mode: str,
    allow: Sequence[str],
    since: float,
    my_uid: int,
    parents: Collection[tuple[int, int]] = (),
    instance: str = "",
    desktop: bool = True,
) -> tuple[int, int, str] | None:
    """The program to stop while the machine is short of memory, as (pid, RAM it holds, "stop" or
    "warn"): the largest one outside dibs that holds at least `minimum` bytes, leaving alone what the
    GPU guard leaves alone, and the agent CLIs themselves."""
    for pid, held in sorted(programs, key=lambda p: -p[1]):
        if held < minimum:
            break
        if mode == "off" or in_dibs(pid) or _is_agent(pid):
            continue
        kind = _judge(pid, mode, allow, since, my_uid, parents, instance, desktop)
        if kind in ("stop", "warn"):
            return pid, held, kind
    return None


def _other_dibs_job(pid: int, instance: str) -> bool:
    """Whether the process is a job of another dibs of this user (another DIBS_HOME): every job
    carries its dibs's instance id and its job id in its environment."""
    env = procs.environ(pid) or {}
    theirs = env.get("DIBS_INSTANCE")
    return bool(env.get("DIBS_JOB_ID") and theirs and theirs != instance)


def _descends_from(pid: int, parents: Collection[tuple[int, int]]) -> bool:
    for _ in range(64):
        pid = procs.parent(pid) or 0
        if pid <= 1:
            return False
        if (pid, procs.start_ticks(pid)) in parents:
            return True
    return False


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


def ram_stop_message(pid: int, argv: Sequence[str], held: int, available: int, total: int) -> str:
    """What a program stopped for RAM leaves in its output: why, and how to keep it from happening again."""
    command = shlex.join(argv) if argv else "this process"
    return (
        f"dibs: stopped `{command}` (pid {pid}): it held {format_size(held)} of RAM outside dibs while the "
        f"machine was running out of memory ({format_size(available)} of {format_size(total)} left).\n"
        "dibs: Programs outside dibs get no RAM set aside, so left running it would have stalled every program "
        "here until the kernel killed it, or something else.\n"
        "dibs: So it doesn't happen again, run it as a job: a job gets the RAM it reserves.\n"
        f"dibs:   dibs run --cpus N --mem {format_size(suggest_size(held))} --time ESTIMATE -- {command}\n"
        "dibs: Run every step that may need more than a few GB of RAM through dibs, not only GPU work. "
        "`dibs agent-help` explains how to choose N, --mem and ESTIMATE."
    )


def stop_for_ram(pid: int, held: int, available: int, total: int) -> dict | None:
    """Stop a program (and what it started) for holding too much RAM outside dibs, leaving the note in
    its output; the event to record, or None if it was already gone."""
    if not procs.alive(pid):
        return None
    argv = procs.cmdline(pid) or [name for name in [procs.exe_name(pid)] if name]
    env = procs.environ(pid) or {}
    procs.stop_with_message([pid], ram_stop_message(pid, argv, held, available, total))
    return {"pid": pid, "command": argv, "ram": held, "session": env.get("DIBS_SESSION") or env.get("CLAUDE_CODE_SESSION_ID")}


def enforce(verdicts: Sequence[Verdict]) -> list[dict]:
    """Stop every process judged "stop" (and what it started), each once; returns one event per stop."""
    events, done = [], set()
    for v in verdicts:
        if v.kind != "stop" or v.pid in done:
            continue
        done.add(v.pid)
        if not procs.alive(v.pid):
            continue  # it exited before the guard got to it: nothing was stopped
        argv = procs.cmdline(v.pid) or [name for name in [procs.exe_name(v.pid)] if name]
        env = procs.environ(v.pid) or {}
        message = stop_message(v.pid, argv, v.gpu_index, v.used, procs.rss(v.pid))
        procs.stop_with_message([v.pid], message)
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
