"""Reading /proc, and stopping processes with a message in their output (Linux).

Every reader returns None when the process is gone or can't be read, rather than raising.
"""

import os
import signal
from collections.abc import Sequence

_CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
_boot_time: float | None = None


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
    global _boot_time
    if _boot_time is None:
        for line in (_read("/proc/stat") or "").splitlines():
            if line.startswith("btime "):
                _boot_time = float(line.split()[1])
                break
        else:
            _boot_time = 0.0
    return _boot_time


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


def write_once(pids: Sequence[int], text: str) -> int:
    """Append text to the stdout and stderr of these processes, once per distinct destination.

    Opening with O_APPEND puts it at the end of a log file even if the process wrote with its own
    offset. Destinations that can't be opened (sockets, gone processes) are skipped, and so is
    /dev/null. Returns how many destinations got the text.
    """
    message = ("\n" + text.rstrip("\n") + "\n").encode()
    seen, written = set(), 0
    for pid in pids:
        for fd in (1, 2):
            path = f"/proc/{pid}/fd/{fd}"
            try:
                st = os.stat(path)
                if os.path.realpath(path) == "/dev/null" or (st.st_dev, st.st_ino) in seen:
                    continue
                seen.add((st.st_dev, st.st_ino))
                out = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_NONBLOCK)
            except OSError:
                continue
            try:
                os.write(out, message)
                written += 1
            except OSError:
                pass
            finally:
                os.close(out)
    return written


def _signal(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError, OverflowError):
        pass


def freeze_write_kill(pids: Sequence[int], text: str | None) -> None:
    """Stop these processes and everything they started, leaving text in their output first.

    Freezing first means none of them can write over the message or start anything new.
    """
    frozen: set[int] = set()
    todo = set(pids)
    while todo - frozen:
        for pid in todo - frozen:
            _signal(pid, signal.SIGSTOP)
            frozen.add(pid)
        cmap = children_map()
        todo = frozen | {d for pid in frozen for d in descendants(pid, cmap)}
    if text:
        write_once(sorted(frozen), text)
    for pid in frozen:
        _signal(pid, signal.SIGKILL)
