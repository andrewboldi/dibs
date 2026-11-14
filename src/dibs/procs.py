"""Reading /proc, and stopping processes with a message in their output (Linux).

Every reader returns None when the process is gone or can't be read, rather than raising.
"""

import os
import signal
import time
from collections.abc import Sequence

_CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def _read(path: str, binary: bool = False):
    try:
        with open(path, "rb" if binary else "r") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


def _stat_fields(pid: int) -> list[str] | None:
    """Fields of /proc/<pid>/stat after the command name: [state, ppid, ...]."""
    text = _read(f"/proc/{pid}/stat")
    if not text or ")" not in text:
        return None
    return text.rsplit(")", 1)[1].split()


def boot_time() -> float:
    """When the machine booted, in epoch seconds, to the precision of process start times.

    /proc/stat's btime is whole seconds; process start times count from boot in clock ticks
    (CLOCK_BOOTTIME), so derive boot from that same clock.
    """
    return time.time() - time.clock_gettime(time.CLOCK_BOOTTIME)


def state(pid: int) -> str | None:
    fields = _stat_fields(pid)
    return fields[0] if fields else None


def parent(pid: int) -> int | None:
    fields = _stat_fields(pid)
    return int(fields[1]) if fields else None


def start_ticks(pid: int) -> int | None:
    """When the process started, in clock ticks since boot: with the pid, it names one process."""
    fields = _stat_fields(pid)
    return int(fields[19]) if fields else None


def start_time(pid: int) -> float | None:
    ticks = start_ticks(pid)
    return None if ticks is None else boot_time() + ticks / _CLK_TCK


def alive(pid: int, ticks: int | None = None) -> bool:
    """Whether the process runs (zombies don't count). With ticks, also that the pid wasn't reused."""
    fields = _stat_fields(pid)
    if not fields or fields[0] in ("Z", "X"):
        return False
    return ticks is None or int(fields[19]) == ticks


def children_map() -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            ppid = parent(int(name))
            if ppid is not None:
                children.setdefault(ppid, []).append(int(name))
    return children


def descendants(pid: int, cmap: dict[int, list[int]] | None = None) -> list[int]:
    cmap = children_map() if cmap is None else cmap
    found, todo = [], list(cmap.get(pid, []))
    while todo:
        child = todo.pop()
        found.append(child)
        todo.extend(cmap.get(child, []))
    return sorted(found)


def environ(pid: int) -> dict[str, str] | None:
    raw = _read(f"/proc/{pid}/environ", binary=True)
    if raw is None:
        return None
    env = {}
    for item in raw.split(b"\0"):
        key, sep, value = item.partition(b"=")
        if sep:
            env[key.decode(errors="surrogateescape")] = value.decode(errors="surrogateescape")
    return env


def cmdline(pid: int) -> list[str] | None:
    raw = _read(f"/proc/{pid}/cmdline", binary=True)
    if not raw:
        return None
    return [a.decode(errors="surrogateescape") for a in raw.rstrip(b"\0").split(b"\0")]


def exe_name(pid: int) -> str | None:
    comm = _read(f"/proc/{pid}/comm")
    return comm.strip() if comm else None


def uid(pid: int) -> int | None:
    try:
        return os.stat(f"/proc/{pid}").st_uid
    except OSError:
        return None


def rss(pid: int) -> int | None:
    statm = _read(f"/proc/{pid}/statm")
    return int(statm.split()[1]) * os.sysconf("SC_PAGE_SIZE") if statm else None


def cgroup(pid: int) -> str | None:
    for line in (_read(f"/proc/{pid}/cgroup") or "").splitlines():
        if line.startswith("0::"):
            return line[3:]
    return None


def _open_outputs(pids: Sequence[int]) -> list[int]:
    """Open each distinct stdout/stderr destination of these processes for appending.

    Destinations that can't be opened (sockets, gone processes) are skipped, and so is /dev/null.
    O_APPEND puts what we write at the end of a log file even if the process kept its own offset.
    """
    seen, fds = set(), []
    for pid in pids:
        for n in (1, 2):
            path = f"/proc/{pid}/fd/{n}"
            try:
                st = os.stat(path)
                if os.path.realpath(path) == "/dev/null" or (st.st_dev, st.st_ino) in seen:
                    continue
                fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_NONBLOCK | os.O_NOCTTY)
            except OSError:
                continue
            seen.add((st.st_dev, st.st_ino))
            fds.append(fd)
    return fds


def _write(fd: int, data: bytes) -> None:
    try:
        os.write(fd, data)
    except OSError:
        pass


def _signal(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError, OverflowError):
        pass


def stop_with_message(pids: Sequence[int], text: str | None) -> None:
    """Kill these processes and everything they started, leaving text in their output.

    The text goes to a terminal before the kill, so it reads before the shell's next prompt, and
    to log files and pipes after it, so a dying process can't write over it. Nothing is paused
    first: SIGSTOP would make an interactive shell report the job as "Stopped".
    """
    targets = set(pids)
    cmap = children_map()
    targets |= {d for pid in pids for d in descendants(pid, cmap)}
    fds = _open_outputs(sorted(targets)) if text else []
    message = ("\n" + (text or "").rstrip("\n") + "\n").encode()
    ttys = {fd for fd in fds if os.isatty(fd)}
    for fd in ttys:
        _write(fd, message)
    killed: set[int] = set()
    for _ in range(5):  # children forked just before the kill turn up on the next look
        for pid in targets - killed:
            _signal(pid, signal.SIGKILL)
            killed.add(pid)
        cmap = children_map()
        more = {d for pid in killed for d in descendants(pid, cmap)} - killed
        if not more:
            break
        targets |= more
    for fd in fds:
        if fd not in ttys:
            _write(fd, message)
        os.close(fd)
