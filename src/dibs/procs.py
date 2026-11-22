"""Reading /proc, and stopping processes with a message in their output (Linux).

Every reader returns None when the process is gone or can't be read, rather than raising.
"""

import os
import signal
import stat
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


def tty(pid: int) -> int | None:
    """The process's controlling terminal (0 for none)."""
    fields = _stat_fields(pid)
    return int(fields[4]) if fields else None


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


def big_programs(minimum: int) -> list[tuple[int, int]]:
    """(pid, RAM it holds) of every process holding at least `minimum` bytes of RAM, largest first."""
    found = []
    for name in os.listdir("/proc"):
        if name.isdigit():
            held = rss(int(name))
            if held is not None and held >= minimum:
                found.append((int(name), held))
    return sorted(found, key=lambda p: -p[1])


CGROUP_ROOT = "/sys/fs/cgroup"  # where cgroup v2 is mounted


def cpu_rate(pid: int) -> float | None:
    """The cores a process has kept busy on average since it started (CPU seconds per second), over at
    least a second: CPU time counts in clock ticks, and one tick read milliseconds after a process
    started came to several cores (#219)."""
    fields, started = _stat_fields(pid), start_time(pid)
    if not fields or started is None:
        return None
    return (int(fields[11]) + int(fields[12])) / _CLK_TCK / max(1.0, time.time() - started)


def cgroup_procs(cgroup_path: str) -> list[int]:
    text = _read(f"{CGROUP_ROOT}{cgroup_path}/cgroup.procs") or ""
    return [int(line) for line in text.split() if line.isdigit()]


def pane_cpu(prefixes: tuple[str, ...]) -> dict[str, int]:
    """The CPU time (µs) of each terminal pane's cgroup in this user's session, by cgroup path: the
    scopes named with one of these prefixes, up to two levels below app.slice (#158)."""
    uid = os.getuid()
    base = f"/user.slice/user-{uid}.slice/user@{uid}.service/app.slice"

    def dirs(path: str) -> list[str]:
        try:
            return [e.name for e in os.scandir(CGROUP_ROOT + path) if e.is_dir()]
        except OSError:
            return []

    found = {}
    for first in dirs(base):
        for path in (f"{base}/{first}", *(f"{base}/{first}/{second}" for second in dirs(f"{base}/{first}"))):
            leaf = path.rsplit("/", 1)[-1]
            if leaf.startswith(prefixes) and leaf.endswith(".scope"):
                usec = keyed_numbers(_read(f"{CGROUP_ROOT}{path}/cpu.stat")).get("usage_usec")
                if usec is not None:
                    found[path] = usec
    return found


def meminfo() -> tuple[int, int] | None:
    """The machine's RAM in bytes: (total, in use), in use being MemTotal - MemAvailable (file cache
    the kernel can drop counts as available). DIBS_MEMINFO points tests at a stand-in file."""
    fields = {}
    for line in (_read(os.environ.get("DIBS_MEMINFO") or "/proc/meminfo") or "").splitlines():
        key, _, rest = line.partition(":")
        value = rest.split()
        if value and value[0].isdigit():
            fields[key] = int(value[0]) * 1024
    if "MemTotal" not in fields or "MemAvailable" not in fields:
        return None
    return fields["MemTotal"], fields["MemTotal"] - fields["MemAvailable"]


def keyed_numbers(text: str | None) -> dict[str, int]:
    """A cgroup "key value" file (memory.stat, cpu.stat), as numbers."""
    stat = {}
    for line in (text or "").splitlines():
        key, _, value = line.partition(" ")
        if value.strip().isdigit():
            stat[key] = int(value)
    return stat


def memory_in_use(cgroup_path: str) -> tuple[int, int, tuple[int, int]] | None:
    """What a cgroup's processes hold (anon + shmem), leaving out the file cache that memory.current
    and memory.peak also count: the kernel drops that cache when memory runs short. With it, the
    shmem part (files in /dev/shm or a RAM-backed /tmp, and shared memory its processes map, such as
    buffers pinned for GPU transfers), and how much of that part they had mapped, at least and at
    most: memory.stat counts it in file_mapped together with the other files they map (#132)."""
    stat = keyed_numbers(_read(f"{CGROUP_ROOT}{cgroup_path}/memory.stat"))
    if "anon" not in stat:
        return None
    shmem = stat.get("shmem", 0)
    mapped, other_files = stat.get("file_mapped", 0), max(0, stat.get("file", 0) - shmem)
    most = min(shmem, mapped)
    return stat["anon"] + shmem, shmem, (min(most, max(0, mapped - other_files)), most)


def oom_kills(events: str | None) -> tuple[int, bool]:
    """From a cgroup's memory.events: how many of its processes the kernel killed for lack of RAM, and
    whether the cgroup reached its own limit. A kill without that was the machine running out."""
    stat = keyed_numbers(events)
    return stat.get("oom_kill", 0), stat.get("oom", 0) > 0


def cpu_busy_seconds() -> float | None:
    """CPU time the whole machine has spent busy since boot, in core-seconds (from /proc/stat)."""
    for line in (_read("/proc/stat") or "").splitlines():
        if line.startswith("cpu "):
            fields = [int(x) for x in line.split()[1:]]
            user, nice, system, _idle, _iowait, irq, softirq, steal = (fields + [0] * 8)[:8]
            return (user + nice + system + irq + softirq + steal) / os.sysconf("SC_CLK_TCK")
    return None


SHORT_OF_RAM = 0.2  # share of the RAM: with less available, high pressure means the machine is short (#129)
SHORT_PRESSURE = 40.0  # percent: memory pressure this high, with little RAM available, is a shortage (#114)


def short_of_memory(meminfo=None, pressure=None) -> bool:
    """Whether the machine is short of memory right now: little RAM available, and memory pressure high.
    Pressure alone doesn't say so: it stays high a while after a shortage, as swapped-out pages come back
    in with RAM available again (#131)."""
    info = (meminfo or globals()["meminfo"])()
    if info is None:
        return False
    total, used = info
    return total - used < total * SHORT_OF_RAM and (pressure or memory_pressure)() >= SHORT_PRESSURE


CPU_ROOT = "/sys/devices/system/cpu"
CAPPED_BELOW = 0.9  # a CPU whose speed cap is under this share of its top speed is capped (#179)


def cpu_cap(root: str = CPU_ROOT) -> dict | None:
    """When most CPUs are capped below their top speed (cpufreq's scaling_max_freq under 90% of
    cpuinfo_max_freq), the lowest cap among them and the highest top speed, in MHz; None otherwise, or
    without cpufreq. A cap set while the GPU draws a lot slowed every CPU job up to about 3 times on one
    laptop, and nothing showed it (#179). dibs only reads the cap; it never sets it."""
    try:
        names = [n for n in os.listdir(root) if n.startswith("cpu") and n[3:].isdigit()]
    except OSError:
        return None
    speeds = []
    for name in names:
        top = (_read(f"{root}/{name}/cpufreq/cpuinfo_max_freq") or "").strip()
        cap = (_read(f"{root}/{name}/cpufreq/scaling_max_freq") or "").strip()
        if top.isdigit() and cap.isdigit():
            speeds.append((int(top) // 1000, int(cap) // 1000))
    capped = [(top, cap) for top, cap in speeds if cap < top * CAPPED_BELOW]
    if not speeds or len(capped) * 2 < len(speeds):
        return None
    return {"capped_mhz": min(cap for _, cap in capped), "max_mhz": max(top for top, _ in speeds)}


def outside_holders(minimum: int) -> list[dict]:
    """What holds RAM outside dibs, per app or terminal pane (#164): each scope or service under the
    user's systemd instance, but dibs's own jobs and service, by what its processes hold (anon + shmem:
    exact, shared pages counted once) and its swap, largest first. An app's scopes are added up; each
    terminal pane is its own. Only those holding `minimum` or more in RAM and swap together, four at most."""
    from dibs.agents import PANE_SCOPES

    base = f"{CGROUP_ROOT}/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service"
    found: dict[str, dict] = {}
    for path, dirs, files in os.walk(base):
        leaf = os.path.basename(path)
        if "memory.stat" not in files or not leaf.endswith((".scope", ".service")) or path == base:
            continue
        if any(os.path.exists(f"{path}/{d}/memory.stat") for d in dirs):
            continue  # it holds others, each counted itself
        if leaf == "dibs.service" or (leaf.startswith("dibs-") and "-job-" in leaf):
            continue
        stat = keyed_numbers(_read(f"{path}/memory.stat"))
        swap = (_read(f"{path}/memory.swap.current") or "0").strip()
        pane = leaf.startswith(PANE_SCOPES)
        name = _holder_name(path, leaf, pane)
        held = found.setdefault(leaf if pane else name, {"name": name, "ram": 0, "swap": 0})
        held["ram"] += stat.get("anon", 0) + stat.get("shmem", 0)
        held["swap"] += int(swap) if swap.isdigit() else 0
    holders = [h for h in found.values() if h["ram"] + h["swap"] >= minimum]
    return sorted(holders, key=lambda h: -(h["ram"] + h["swap"]))[:4]


def _holder_name(path: str, leaf: str, pane: bool) -> str:
    """A readable name for a scope or service: the app, or a terminal pane and its largest program."""
    if pane:
        largest = max(((_rss(pid), pid) for pid in _pids(path)), default=None)
        program = (_read(f"/proc/{largest[1]}/comm") or "").strip() if largest else ""
        return f"terminal pane ({program})" if program else "terminal pane"
    if leaf.startswith("snap."):  # snap.chromium.chromium-<id>.scope
        return leaf.split(".")[1]
    name = leaf.rsplit(".", 1)[0].replace("\\x2d", "-")
    if name.startswith("app-"):  # app-gnome-org.gnome.Evolution-12618.scope
        name = name[4:].removeprefix("gnome-").rstrip("0123456789").rstrip("-")
    return name


def _pids(path: str) -> list[int]:
    return [int(p) for p in (_read(f"{path}/cgroup.procs") or "").split() if p.isdigit()]


def _rss(pid: int) -> int:
    fields = (_read(f"/proc/{pid}/statm") or "").split()
    return int(fields[1]) if len(fields) > 1 and fields[1].isdigit() else 0


def tmp_files(path: str = "/tmp") -> int | None:
    """What files in a RAM-backed /tmp (a tmpfs) take; None when /tmp is on disk (#164). They're charged
    to whichever cgroup wrote them, so a per-app list misnames them."""
    mounts = [line.split() for line in (_read("/proc/mounts") or "").splitlines()]
    if not any(len(m) > 2 and m[1] == path and m[2] == "tmpfs" for m in mounts):
        return None
    try:
        st = os.statvfs(path)
    except OSError:
        return None
    return (st.f_blocks - st.f_bfree) * st.f_frsize


def memory_pressure() -> float:
    """The share of the last 10 s in which some task waited for memory, in percent (PSI, "some
    avg10"); 0 where the kernel doesn't report it."""
    for line in (_read("/proc/pressure/memory") or "").splitlines():
        if line.startswith("some "):
            for field in line.split()[1:]:
                key, _, value = field.partition("=")
                if key == "avg10":
                    try:
                        return float(value)
                    except ValueError:
                        return 0.0
    return 0.0


def slept() -> float:
    """Seconds this machine has spent asleep (suspended) since it booted: CLOCK_BOOTTIME counts that
    time and CLOCK_MONOTONIC doesn't. 0 where there's no CLOCK_BOOTTIME."""
    boottime = getattr(time, "CLOCK_BOOTTIME", None)
    if boottime is None:
        return 0.0
    return max(0.0, time.clock_gettime(boottime) - time.monotonic())


def boot_id() -> str | None:
    """This boot's id: it changes when the machine restarts."""
    return (_read("/proc/sys/kernel/random/boot_id") or "").strip() or None


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
    seen, fds, stopping = set(), [], set(pids)
    for pid in pids:
        for n in (1, 2):
            path = _read_output(pid, n, stopping)
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


def _read_output(pid: int, n: int, stopping: set[int]) -> str:
    """Where what the process writes to fd n gets read: the /proc path of that destination.

    A file without a name is a program's own capture (pytest's, while a test runs): nobody reads it
    once every process holding it is killed. Then it's where the parent's fd n goes, and so on up,
    to the first process that isn't being stopped (#222).
    """
    path = f"/proc/{pid}/fd/{n}"
    while pid in stopping and _unnamed(path):
        pid = parent(pid) or 0
        path = f"/proc/{pid}/fd/{n}"
    return path


def _unnamed(path: str) -> bool:
    try:
        st = os.stat(path)
    except OSError:
        return False
    return stat.S_ISREG(st.st_mode) and st.st_nlink == 0


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
    to log files and pipes after it, so a dying process can't write over it.

    Processes without a terminal (an agent's commands, background jobs) are frozen first: frozen
    processes can't fork, so the tree that gets killed is complete. Processes on a terminal aren't,
    since an interactive shell would report a frozen job as "Stopped" before the message.
    """
    targets = set(pids)
    cmap = children_map()
    targets |= {d for pid in pids for d in descendants(pid, cmap)}
    if not any(tty(pid) for pid in targets):
        frozen: set[int] = set()
        while targets - frozen:
            for pid in targets - frozen:
                _signal(pid, signal.SIGSTOP)
                frozen.add(pid)
            cmap = children_map()
            targets |= {d for pid in frozen for d in descendants(pid, cmap)}
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
