"""The scheduler: the one process that owns the queue, starts jobs and watches them.

Everything it knows is in the store, written before it acts, so a crash or restart loses nothing
and never touches running jobs: they're adopted again on start (recover).
"""

import asyncio
import contextlib
import fcntl
import json
import logging
import logging.handlers
import math
import os
import shutil
import signal
import statistics
import sys
import time
from collections import deque
from dataclasses import asdict, replace
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from dibs import __version__, build_id, guard, launcher, procs, snapshot
from dibs.agents import PANE_SCOPES
from dibs.gpu import GpuBackend, open_backend
from dibs.machine import Config, ConfigError, build_capacity, detect_cpus, detect_mem, load_config
from dibs.model import (
    CREDIT_MAX,
    GPU_IDLE_AFTER,
    TERMINAL,
    Capacity,
    Job,
    Request,
    RequestError,
    State,
    DEFAULT_PRIORITY,
    compute_tag,
    default_name,
    next_owed,
    job_to_dict,
    resolve_request,
    script_name,
    script_of,
    vram_to_ask,
)
from dibs.paths import Paths, ensure_dirs, get_paths
from dibs.planner import Plan, Queued, Running, plan
from dibs.store import Store
from dibs.units import GiB, format_duration, format_size, format_size_against, parse_size, suggest_size

KILL_AFTER = 10.0  # seconds between asking a job to stop (SIGTERM) and forcing it (SIGKILL)
OWED_FOR = 86400.0  # seconds: what a session owes (or is owed) for its priorities lapses after a day
OUTSIDE_WINDOW = 120.0  # seconds: RAM in use outside dibs counts as its peak over this long
JOBS_RAM_WINDOW = 10.0  # seconds: RAM dibs jobs hold counts as its peak over this long, when taken off the in-use total
RETENTION_EVERY = 3600.0
MAX_RETRIES = 2  # runs again after a resource failure, at most (#84)
YOUNG = 600.0  # seconds: a job with no earlier runs counts at its whole RAM reservation this long (#86)
SESSION_RUNS_AGE = 7 * 86400.0  # seconds: how far back a session's runs in other folders count (#229)
PRESSURE_LIMIT = 10.0  # percent (PSI some avg10): above it, no job starts on borrowed RAM
EVICT_PRESSURE = procs.SHORT_PRESSURE  # percent: memory pressure this high for EVICT_AFTER stops a borrower (#114)
EVICT_AFTER = 10.0  # seconds
SHORT_OF_RAM = procs.SHORT_OF_RAM  # share of the RAM: with less available, high pressure means the machine is short (#129)
SHORT_KEPT = 120.0  # seconds: when a shortage began is kept across a restart this long (#190)
GROW_SHARE = 0.25  # a job grows past its RAM reservation in steps of a quarter of it, at least 1G (#155)
GROW_RETRY = 60.0  # seconds before dibs tries again to raise a limit it couldn't
HOG_SHARE = 0.1  # share of the RAM: a program outside dibs holding this much is stopped when the machine is short (#135)
OUTSIDE_CPU_EVERY = 4.0  # seconds between samples of the CPU busy outside dibs jobs (#120)
OUTSIDE_CPU_WINDOW = 15.0  # seconds the figure dibs top shows is averaged over
LOAD_EVERY = 4.0  # seconds between samples of the machine's busy cores, for dibs top's load row (#122)
LOAD_WINDOW = 600.0  # seconds of them kept
MOMENTUM_SHARE = 0.25  # under momentum, a session's jobs count as this share of their estimates (#94)
MOMENTUM_MAX = 12 * 3600.0  # seconds a grant may last
EARLY_STOP = 60.0  # seconds: a job stopped for VRAM sooner than this was likely still growing (#80)
ALIVE_EVERY = 60.0  # seconds: how often the daemon notes it's alive, so a reboot can be dated (#78)
RECENT = 86400.0  # how far back `queue --all` looks
STOPS_SHOWN = 3600.0  # how far back guard stops are shown
PANE_EVERY = 15.0  # seconds between samples of terminals' CPU time (#158)
HEAVY_FOR = 120.0  # a terminal that keeps HEAVY_SHARE of the cores busy this long gets its agent told
HEAVY_SHARE = 1 / 6  # of the machine's cores, at least 2
TELL_EVERY = 3600.0  # seconds: at most this often per terminal
TOLD_WINDOW = 86400.0  # how far back a session is told, at its next dibs command, what the guard stopped (#135)
FAILED_FOR = 3600.0  # seconds: a failed job is told at each of its session's dibs commands this long at most (#176)
# A job holding GPU memory with the GPU idle (#180): a hold of at least a tenth of the GPU (1G or more), at
# under IDLE_UTIL percent use, counts once it has gone on IDLE_AFTER; checked every IDLE_EVERY; its session
# is told then, while other jobs wait for GPU memory, and at most once every IDLE_TELL_EVERY (#186).
IDLE_UTIL, IDLE_EVERY, IDLE_TELL_EVERY, IDLE_SHARE = 5.0, 30.0, 3600.0, 0.1
IDLE_AFTER = GPU_IDLE_AFTER
ADVISE_EVERY = 3600.0  # seconds: each kind of advice, at most this often per session and script (#169)
SHORTAGE_LOG_EVERY = 60.0  # seconds: while short of memory, what the relief did, logged this often (#191)
SLOW_PASS = 5.0  # seconds: a pass of the daemon's loop this long is logged (#191)
HOLDERS_EVERY = 10.0  # seconds: what holds RAM outside dibs is read this often (#164)
ADVISE_RUNS = 5  # runs of a script it takes to say how it uses what it asks for (#169)
WAIT_RECHECK = 5.0  # a long-polling wait re-checks its jobs at least this often
LOG_MAX = 5 * 1024 * 1024

log = logging.getLogger("dibs")


def _rejected(message: str) -> dict:
    return {"ok": False, "code": "rejected", "error": message}


def _ram_kill_reason(job: Job, result: dict) -> str:
    """Why a job was killed for RAM: at its own limit, or because the whole machine ran out (#86)."""
    if result.get("hit_limit") is False:
        borrowed = ", while it used RAM other jobs had reserved" if job.borrowed_ram else ""
        if job.ram_limit and job.ram_limit > job.request.mem:  # a grown job is the kernel's first pick (#155)
            borrowed += f", after its limit grew past its {format_size(job.request.mem)} reservation"
        return "killed when the machine ran out of RAM" + borrowed
    if job.ram_limit and job.ram_limit > job.request.mem:  # it had grown into RAM nobody reserved (#155)
        return (f"killed: went over its RAM limit of {format_size(job.ram_limit)} ({format_size(job.request.mem)} "
                "reserved, the rest grown into RAM nobody reserved)" + _shared_part(job))  # fmt: skip
    return f"killed: went over its {format_size(job.request.mem)} RAM reservation" + _shared_part(job)


def _limit(job: Job) -> int:
    """The RAM a running job may hold: its reservation, or the limit it grew to past it (#155)."""
    return max(job.request.mem, job.ram_limit or 0)


def _holding(job: Job, mem: int | None = None, vram: int | None = None) -> Request:
    """A running job's request as the planner should count it: VRAM it borrowed counts as held (#88),
    and so does `vram`, what its command's earlier runs held, while it's young (#183); its RAM counts
    as `mem` when RAM is planned by use (#86)."""
    request = job.request
    if job.peak_vram and request.gpus and job.peak_vram > request.vram:
        request = replace(request, vram=job.peak_vram)
    if vram and request.gpus and vram > request.vram:
        request = replace(request, vram=vram)
    if mem is not None and mem != request.mem:
        request = replace(request, mem=mem)
    return request


def _plain(job: Job) -> float:
    """A job's priority-5 place in line (#91)."""
    return job.plain_tag if job.plain_tag is not None else job.tag


def _keeps_place(old: Job, session: str, argv: list[str], cwd: str, new: Request, priority: int) -> bool:
    """Whether a replacement keeps the place in line of the job it replaces (#184): that job was waiting,
    in the same session, and the replacement runs the same command in the same folder and asks for no
    more of anything. Keeping it for more would let a 1-minute placeholder hold a place for a 10-hour job."""
    was = old.request
    return (
        old.state == State.QUEUED and old.session == session and list(old.argv) == argv and old.cwd == cwd
        and new.cpus <= was.cpus and new.mem <= was.mem and new.gpus <= was.gpus and new.vram <= was.vram
        and new.time <= was.time and (was.exclusive or not new.exclusive) and priority <= old.priority
    )  # fmt: skip


def _idle_message(job: Job, now: float, waiting: int) -> str:
    """What a session is told of its job holding GPU memory with the GPU idle (#186): by number, as the
    failure notice is (#176)."""
    jobs = "1 job waits" if waiting == 1 else f"{waiting} jobs wait"
    return (f"Job {job.id} has held {format_size(job.gpu_idle_held or 0)} of GPU memory with the GPU idle for "
            f"{format_duration(now - (job.gpu_idle_since or now))}, while {jobs} for GPU memory. A job keeps its "
            "GPU memory until its processes free it or exit.")  # fmt: skip


def _shared_part(job: Job) -> str:
    """For a job killed at its RAM limit: how much of its peak was shared memory, when that was at
    least a quarter of its reservation, and of which kind, since what helps differs. Files a job writes
    to a folder kept in RAM count as its RAM, which agents don't expect (#74), and a TMPDIR on a disk
    moves them out; memory its processes map, such as buffers pinned for GPU transfers, needs more
    --mem (#132). Each kind is named alone only when it was at least three quarters of it."""
    shared = job.peak_shmem
    if shared is None or shared < job.request.mem / 4:
        return ""
    low, high = job.peak_shmem_mapped or (0, shared)
    files = "files in /dev/shm or a RAM-backed /tmp"
    mapped = "shared memory its processes mapped, such as buffers for GPU transfers or between worker processes"
    kind = mapped if low >= shared * 3 / 4 else files if high <= shared / 4 else f"{files}, or {mapped}"
    return f" ({format_size(shared)} of it was {kind})"


def _command_exists(argv0: str, env: dict, cwd: str) -> bool:
    """Whether the job will find its command, looking where it will: relative paths and relative
    PATH entries (bin, .) count from the job's folder, not the daemon's."""
    if "/" in argv0:
        path = argv0 if os.path.isabs(argv0) else os.path.join(cwd, argv0)
        return os.path.isfile(path) and os.access(path, os.X_OK)
    entries = env.get("PATH", os.defpath).split(os.pathsep)
    search = os.pathsep.join(e if os.path.isabs(e) else os.path.join(cwd, e or ".") for e in entries)
    return shutil.which(argv0, path=search) is not None


class Daemon:
    def __init__(
        self,
        paths: Paths,
        config: Config,
        capacity: Capacity,
        backend: GpuBackend,
        *,
        use_systemd: bool,
        clock: Callable[[], float] = time.time,
        meminfo: Callable[[], tuple[int, int] | None] = procs.meminfo,
        slept: Callable[[], float] = procs.slept,
        pressure: Callable[[], float] = procs.memory_pressure,
        cpu_busy: Callable[[], float | None] = procs.cpu_busy_seconds,
        programs: Callable[[int], list[tuple[int, int]]] = lambda minimum: [],
        set_mem_limit: Callable[[Job, int], bool] | None = None,
        panes: Callable[[], dict[str, int]] = lambda: {},
    ):
        self.paths = paths
        self.config = config
        self.capacity = capacity
        self._meminfo = meminfo
        self._outside: deque[tuple[float, int, int]] = deque()  # (time, RAM in use outside dibs, total)
        self._jobs_held: deque[tuple[float, int]] = deque()  # (time, RAM dibs jobs held)
        self.backend = backend
        self.use_systemd = use_systemd
        self.clock = clock
        self.slept = slept
        self.pressure = pressure
        self._pressured_since: float | None = None  # when memory pressure rose above EVICT_PRESSURE
        self._idle_at = -math.inf  # when idle GPU holds were last checked (#180)
        self._relief_note = ""  # what the memory relief did last, for the log (#191)
        self._holders: tuple[float, list[dict], int | None] = (-math.inf, [], None)  # (when, holders, /tmp files)
        self._shortage_logged = -math.inf
        self._passes: deque[tuple[float, float]] = deque()  # (time, seconds the pass took), the last minute
        self._held: dict[int, int] = {}  # job id -> what it held (anon + shmem) at this tick's sample
        self._grow_failed: dict[int, float] = {}  # job id -> when raising its RAM limit last failed
        self.cpu_busy = cpu_busy
        # (pid, RAM held) of the machine's programs holding at least `minimum` bytes, for the guard
        # (#135). Empty unless given: only the real daemon (run) looks at the real ones.
        self.programs = programs
        # Raises a running job's RAM limit (#155): None unless given, so only the real daemon (run), whose
        # jobs have systemd scopes, ever does. Tests pass a stand-in.
        self.set_mem_limit = set_mem_limit
        # The CPU time (µs) of each terminal pane's cgroup, by path (#158): empty unless given, so only
        # the real daemon (run) watches the real terminals.
        self.panes = panes
        self._pane_samples: dict[str, deque[tuple[float, int]]] = {}  # pane cgroup -> (time, CPU µs)
        self._panes_at = -math.inf
        self._told_programs: dict[tuple[str, str], float] = {}  # (pane cgroup, program) -> when last told
        self._beside: dict[int, tuple[float, float, float]] = {}  # job -> (time, machine busy, its own CPU)
        # For dibs top (#120): (time, machine busy, jobs' own CPU, the jobs, time asleep) samples, and
        # what programs outside dibs use: cores (None while unknown), RAM, and VRAM per GPU index.
        self._cpu_samples: deque[tuple[float, float, float, frozenset[int], float]] = deque()
        self._outside_cpu: float | None = None
        self._shown_outside_mem = 0
        self._outside_vram: dict[str, int] = {}
        self._load: deque[tuple[float, float, float]] = deque()  # (time, machine busy core-seconds, time asleep)
        self._display_failed: set[str] = set()  # display-only watchers that raised, logged once each
        ensure_dirs(paths)
        self.store = Store(paths.db)
        # A restart in a shortage resumes its wait, rather than starting it over (#190).
        began = self.store.get_meta("short_since")
        if began is not None and 0 <= clock() - float(began) <= SHORT_KEPT:
            self._pressured_since = float(began)
        self.closed = False
        self.outside: list[dict] = []  # GPU processes outside dibs that the guard left alone
        self._callbacks: list[Callable[[Job], None]] = []
        self._terminated: set[int] = set()  # jobs that have been sent SIGTERM
        self._last_retention = -math.inf
        self._last_alive = -math.inf
        self._sees_processes = backend.processes() is not None
        self._wake: set[asyncio.Event] = set()  # waits to wake when any job finishes
        self._stop: asyncio.Event | None = None
        # The guard leaves alone whatever was already running when it was first switched on.
        if config.guard != "off" and self.store.get_meta("guard_mode") in (None, "off"):
            self.store.set_meta("guard_since", repr(clock()))
            self.store.set_meta("guard_parents", "[]")
        self.store.set_meta("guard_mode", config.guard)
        self.guard_since = float(self.store.get_meta("guard_since") or clock())
        # Programs whose GPU processes were already running when the guard started (see _remember).
        self.guard_parents = {tuple(p) for p in json.loads(self.store.get_meta("guard_parents") or "[]")}
        self._ancestors = set()  # never remembered: adopting an orphan would exempt everything
        pid = os.getpid()
        while (pid := procs.parent(pid) or 0) > 1:
            self._ancestors.add(pid)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.store.close()

    def on_finished(self, callback: Callable[[Job], None]) -> None:
        """Call callback(job) whenever a job reaches a final state."""
        self._callbacks.append(callback)

    # Requests -----------------------------------------------------------------------------------

    def submit(self, req: dict) -> dict:
        argv, cwd, env = req.get("argv"), req.get("cwd") or "", req.get("env") or {}
        if not argv or not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
            return _rejected("nothing to run: give the command after --")
        if not os.path.isdir(cwd):
            return _rejected(f"working directory {cwd} doesn't exist")
        if not _command_exists(argv[0], env, cwd):
            return _rejected(f"command not found: {argv[0]}")
        output = req.get("output")
        if output:
            output = os.path.join(cwd, output)
            if not os.path.isdir(os.path.dirname(output)):
                return _rejected(f"can't write output to {output}: its folder doesn't exist")
        try:
            request = resolve_request(req.get("request") or {}, self.capacity)
        except RequestError as e:
            return _rejected(str(e))

        priority = req.get("priority")
        priority = DEFAULT_PRIORITY if priority is None else priority
        if isinstance(priority, bool) or not isinstance(priority, int) or not 1 <= priority <= 10:
            return _rejected("priority must be a whole number from 1 to 10")

        staged = None
        if req.get("snapshot") is not None:  # a copy of its code the CLI made (#172)
            staged = Path(str(req["snapshot"])).resolve()
            if staged.parent != (self.paths.state_dir / "snapshots").resolve() or not (staged / snapshot.MANIFEST).exists():
                return _rejected("no such snapshot of the job's code")

        replaced = None
        if req.get("replace") is not None:
            replaced = self.store.get_job(int(req["replace"]))
            if replaced is None:
                return _rejected(f"no job {req['replace']} to replace")

        now = self.clock()
        session = req.get("session") or "unknown"
        keeps = replaced is not None and _keeps_place(replaced, session, list(argv), cwd, request, priority)
        if replaced is not None:
            # In the same step as the new job, so nothing can start in the gap (#93), and before the
            # backlog is read, so the replacement doesn't line up behind the job it replaces.
            self._cancel_job(replaced, now, in_place=keeps)
        # The session's queued jobs' priority-5 places: lining up behind boosted tags would pass each
        # boost down the line, and the time a job saved would never be paid back. Only those of at
        # least its priority: it goes ahead of the session's own lower-priority jobs (#91).
        own = [j for j in self.store.jobs([State.QUEUED]) if j.session == session]
        backlog = [j.plain_tag if j.plain_tag is not None else j.tag for j in own if j.priority >= priority]
        start = max([now, *backlog])
        passed = [j for j in own if j.priority < priority and (j.plain_tag if j.plain_tag is not None else j.tag) > start]
        momentum = self._momentum(now)
        counted = request.time * (MOMENTUM_SHARE if momentum and momentum["session"] == session else 1.0)
        carried, left = self._owed(session, now), next_owed(counted, priority)
        tag, plain_tag = compute_tag(now, counted, backlog, priority, carried), compute_tag(now, counted, backlog)
        if keeps:
            # The earlier of its old place and a fresh one (#184), and its debts as they were: the job it
            # replaces paid for its place, and passed the session's lower-priority jobs, when it was queued.
            tag, plain_tag = min(tag, replaced.tag), min(plain_tag, _plain(replaced))
            carried, left, passed = replaced.owed_carried, replaced.owed_left, []
        job = Job(
            id=0,
            name=req.get("name") or default_name(argv),
            argv=list(argv),
            cwd=cwd,
            env=dict(env),
            request=request,
            session=session,
            session_label=req.get("session_label") or session[:4],
            submitted=now,
            tag=tag,
            priority=priority,
            plain_tag=plain_tag,
            own_time=counted,
            retry=req.get("retry") is True,  # the agent's choice: only it knows the job is safe to repeat (#116)
            owed_carried=carried,
            owed_left=left,
            history_mem=self._history_mem(cwd, list(argv), session),
            history_vram=self._history_vram(cwd, list(argv), request, session),
            state=State.QUEUED,
            output=output or "",
        )
        self.store.add_job(job)
        if not keeps:
            self.store.set_meta(f"owed:{session}", json.dumps({"owed": left, "time": now}))
        for j in passed:  # zero-sum within the session: the jobs it went ahead of move back by its estimate
            j.tag += counted
            if j.plain_tag is not None:
                j.plain_tag += counted
            self.store.update_job(j)
        self.paths.job_dir(job.id).mkdir(mode=0o700, parents=True, exist_ok=True)
        if staged is not None:  # it runs this copy of its code, moved into its folder (#172)
            os.rename(staged, self.paths.job_dir(job.id) / "snapshot")
            job.snapshot = json.loads((self.paths.job_dir(job.id) / "snapshot" / snapshot.MANIFEST).read_text())["root"]
            self.store.update_job(job)
        if not job.output:
            job.output = str(self.paths.job_dir(job.id) / "output.log")
            self.store.update_job(job)
        if replaced is not None and replaced.state == State.CANCELLED:
            replaced.reason = f"replaced by job {job.id}"
            self.store.update_job(replaced)
        self.schedule()
        response = self.show(job.id)
        self._advise(job, response["job"], now)
        if replaced is not None:
            response["replaced"] = replaced.id
            response["kept_place"] = keeps
        # What the same command used before, so the agent can size its next request from it.
        earlier, match = self._earlier(job.cwd, job.argv, session, lambda j: j.started is not None)
        response["earlier"] = [job_to_dict(j) for j in earlier[:3]]
        response["earlier_match"] = match
        # How the session's recent estimates compared with real run times (#95): feedback only, the
        # estimate is never rescaled.
        ratios = [
            (j.ended - j.started - j.asleep) / j.request.time
            for j in self.store.finished_since(now - 86400)
            if j.session == session and j.state == State.DONE and j.started and j.ended and j.request.time > 0
        ]
        if len(ratios) >= 5:
            response["estimates"] = {"jobs": len(ratios), "median": statistics.median(ratios)}
        return response

    def _add_owed(self, session: str, delta: float, now: float) -> None:
        if delta:
            owed = max(-CREDIT_MAX, self._owed(session, now) + delta)
            self.store.set_meta(f"owed:{session}", json.dumps({"owed": owed, "time": now}))

    def _settle_priority(self, job: Job, now: float) -> None:
        """When a job starts, settle its priority against how long it really waited (#90, #91): debt it
        carried is paid only by waiting, and the rest rolls on to the session's next job; credit for a
        lowered priority counts only as far as it waited; and a boost that passed no other session's
        job costs nothing."""
        if job.owed_settled:
            return
        job.owed_settled = True
        waited = max(0.0, now - job.submitted)
        delta = 0.0
        if job.owed_carried > 0:
            delta += job.owed_carried - min(job.owed_carried, waited)
        if job.owed_left < 0:
            delta += -job.owed_left - min(-job.owed_left, waited)
        if job.owed_left > 0 and not self._passed_another_session(job):
            delta -= job.owed_left
        self._add_owed(job.session, delta, now)

    def _passed_another_session(self, job: Job) -> bool:
        """Whether another session's job is still waiting that, without priorities, would have gone first."""
        mine = job.plain_tag if job.plain_tag is not None else job.tag
        return any(
            (j.plain_tag if j.plain_tag is not None else j.tag) < mine
            for j in self.store.jobs([State.QUEUED])
            if j.session != job.session
        )

    def _momentum(self, now: float) -> dict | None:
        """The current momentum grant, if one hasn't run out (#94)."""
        grant = json.loads(self.store.get_meta("momentum") or "null")
        return grant if grant and now < grant["until"] else None

    def momentum(self, req: dict) -> dict:
        """Momentum (#94): for a project working better than expected and held up by another project's
        jobs, after the user's yes. While it lasts, the session's jobs count as a quarter of their
        estimates, with no payback; one project at a time. Actions: show, grant, end."""
        now = self.clock()
        current = self._momentum(now)
        action = req.get("action") or "show"
        if action == "show":
            return {"ok": True, "momentum": current}
        if action == "end":
            self.store.set_meta("momentum", "null")
            if current:
                self.store.add_event("momentum", {**current, "ended": now}, now)
            return {"ok": True, "momentum": None}
        session = req.get("session") or "unknown"
        why, seconds = (req.get("why") or "").strip(), req.get("seconds")
        if not why:
            return _rejected("say why in a sentence: what is going better than expected, and that the user agreed")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 60 <= seconds <= MOMENTUM_MAX:
            return _rejected("momentum lasts from a minute to 12 hours")
        if current and current["session"] != session:
            until = time.strftime("%H:%M", time.localtime(current["until"]))
            return _rejected(f"another project already has momentum ({current['label']}, until {until})")
        _, running, queued = self._plan(now)
        mine = [j for j in queued if j.session == session]
        if not mine:
            return _rejected("momentum is for a project held up by others: nothing of yours is waiting")
        if not any(j.session != session for j in running):
            return _rejected("momentum is for a project held up by others: only your own jobs are running")
        grant = {"session": session, "label": req.get("session_label") or session[:4], "granted": now,
                 "until": now + float(seconds), "why": why}  # fmt: skip
        self.store.set_meta("momentum", json.dumps(grant))
        self.store.add_event("momentum", grant, now)
        for j in mine:  # its waiting jobs count as a quarter of their estimates from now on
            pull = (1 - MOMENTUM_SHARE) * j.request.time
            j.tag = max(j.submitted, j.tag - pull)
            if j.plain_tag is not None:
                j.plain_tag = max(j.submitted, j.plain_tag - pull)
            j.own_time = max(0.0, (j.own_time if j.own_time is not None else j.request.time) - pull)
            self.store.update_job(j)
        log.info("momentum for %s until %s: %s", grant["label"], time.strftime("%H:%M", time.localtime(grant["until"])), why)
        self.schedule()
        return {"ok": True, "momentum": grant}

    def _owed(self, session: str, now: float) -> float:
        """What the session's previous job left it (see model.next_owed), if it was within OWED_FOR."""
        entry = json.loads(self.store.get_meta(f"owed:{session}") or "{}")
        return entry.get("owed", 0.0) if now - entry.get("time", -math.inf) <= OWED_FOR else 0.0

    def cancel(self, ids: list[int]) -> dict:
        now = self.clock()
        cancelled, errors = [], []
        for job_id in ids:
            job = self.store.get_job(job_id)
            if job is None:
                errors.append(f"no job {job_id}")
            elif self._cancel_job(job, now):
                cancelled.append(job_id)
            else:
                errors.append(f"job {job_id} already {job.state}")
        self.schedule()
        return {"ok": True, "cancelled": cancelled, "errors": errors}

    def _reline(self, session: str) -> None:
        """Line a session's waiting jobs up again by submit()'s rules, in the order they came, behind its
        jobs still waiting only: one that started or was cancelled no longer holds back those after it,
        so a busy session's line can't run hours ahead of the clock (#150). Places only move earlier."""
        own = sorted((j for j in self.store.jobs([State.QUEUED]) if j.session == session), key=lambda j: (j.submitted, j.id))
        lined: list[Job] = []
        new: dict[int, list[float]] = {}  # job id -> [tag, plain tag]
        for j in own:
            counted = j.own_time if j.own_time is not None else j.request.time
            backlog = [new[r.id][1] for r in lined if r.priority >= j.priority]
            start = max([j.submitted, *backlog])
            for r in lined:  # it goes ahead of the session's lower-priority jobs, which move back by its estimate
                if r.priority < j.priority and new[r.id][1] > start:
                    new[r.id][0] += counted
                    new[r.id][1] += counted
            new[j.id] = [compute_tag(j.submitted, counted, backlog, j.priority, j.owed_carried),
                         compute_tag(j.submitted, counted, backlog)]  # fmt: skip
            lined.append(j)
        for j in own:
            tag, plain = new[j.id]
            tag, plain = min(tag, j.tag), min(plain, j.plain_tag if j.plain_tag is not None else plain)
            if (tag, plain) != (j.tag, j.plain_tag):
                j.tag, j.plain_tag = tag, plain
                self.store.update_job(j)

    def _cancel_job(self, job: Job, now: float, in_place: bool = False) -> bool:
        """Cancel one job without scheduling: a queued one ends now, a running one is asked to stop.
        in_place: a replacement takes its place in line and its debts (#184), so nothing else moves.
        False if it had already ended."""
        if job.state == State.QUEUED:
            job.state, job.ended, job.reason = State.CANCELLED, now, "cancelled"
            self.store.update_job(job)
            if not in_place:
                # It never ran: it hands back what it carried and takes back what it left (#90).
                self._add_owed(job.session, job.owed_carried - job.owed_left, now)
            self._notify(job)
            if not in_place:
                self._reline(job.session)
            return True
        if job.state == State.RUNNING:
            if job.cancel_requested is None:
                job.cancel_requested = now
                self.store.update_job(job)
            self._push_stop(job, now)
            return True
        return False

    def show(self, job_id: int) -> dict:
        job = self.store.get_job(job_id)
        if job is None:
            return {"ok": False, "code": "not_found", "error": f"no job {job_id}"}
        now = self.clock()
        current, _, _ = self._plan(now)
        return {"ok": True, "now": now, "job": self._view(job, now, current)}

    def snapshot(self, *, include_finished: bool = False, session: str | None = None) -> dict:
        now = self.clock()
        current, running, queued = self._plan(now)
        position = {p.job_id: p.position for p in current.predictions.values()}
        jobs = sorted(running, key=lambda j: (j.started or 0, j.id))
        jobs += sorted(queued, key=lambda j: position.get(j.id, math.inf))
        if include_finished:
            recent = self.store.finished_since(now - RECENT)
            jobs += sorted(recent, key=lambda j: (j.ended or 0, j.id))
        if session is not None:
            jobs = [j for j in jobs if j.session == session]
        stopped = [e for e in self.store.events(now - STOPS_SHOWN) if e["kind"] == "stop"]
        return {
            "ok": True,
            "now": now,
            "jobs": [self._view(j, now, current) for j in jobs],
            "capacity": self._capacity(),
            "guard": {"mode": self.config.guard, "stopped": stopped, "outside": self.outside},
            "momentum": self._momentum(now),
            # What programs outside dibs use, for dibs top (#120); the RAM is what _plan just subtracted.
            "outside": {"cpus": self._outside_cpu, "mem": self._shown_outside_mem, "vram": dict(self._outside_vram)},
            "load": {"every": LOAD_EVERY, "cores": self._load_history()},
            "cpu_cap": procs.cpu_cap(),  # only read, for dibs top (#179)
            # What holds the RAM in use outside dibs, and files in a RAM-backed /tmp (#164).
            "outside_holders": self._ram_holders(now)[0],
            "tmp_files": self._ram_holders(now)[1],
        }

    def info(self) -> dict:
        return {
            "ok": True,
            "version": __version__,
            "build": build_id(),
            "pid": os.getpid(),
            "capacity": self._capacity(),
            "enforcement": {
                "cpu_pinning": hasattr(os, "sched_setaffinity"),
                "ram_cap": self.use_systemd,
                "vram_watch": self._sees_processes and bool(self.capacity.gpus),
                "guard": self.config.guard,
            },
            "gpu_backend": self.backend.name,
            "paths": {
                "config": str(self.paths.config_file),
                "state": str(self.paths.state_dir),
                "socket": str(self.paths.socket),
            },
        }

    # Serving ------------------------------------------------------------------------------------

    async def serve(self) -> None:
        """Answer requests on the socket and tick every second, until shutdown or SIGTERM/SIGINT."""
        loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._stop.set)
            except (NotImplementedError, RuntimeError, ValueError):
                pass  # not the main thread (tests)
        self.on_finished(lambda job: [event.set() for event in list(self._wake)])
        ensure_dirs(self.paths)
        if os.path.lexists(self.paths.socket):
            os.unlink(self.paths.socket)  # stale: we hold the lock, so nobody else serves it
        clients: set[asyncio.Task] = set()

        async def client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            clients.add(asyncio.current_task())
            try:
                while line := await reader.readline():
                    try:
                        request = json.loads(line)
                    except ValueError:
                        response = {"ok": False, "code": "bad_request", "error": "requests are JSON lines"}
                    else:
                        response = await self.handle(request)
                    writer.write(json.dumps(response).encode() + b"\n")
                    await writer.drain()
            except (ConnectionError, asyncio.CancelledError):
                pass
            finally:
                clients.discard(asyncio.current_task())
                writer.close()

        server = await asyncio.start_unix_server(client, path=str(self.paths.socket), limit=16 * 1024 * 1024)
        os.chmod(self.paths.socket, 0o600)
        ticker = asyncio.create_task(self._tick_forever())
        try:
            await self._stop.wait()
        finally:
            ticker.cancel()
            server.close()
            for task in list(clients):
                task.cancel()
            try:
                await asyncio.wait_for(server.wait_closed(), 2)
            except asyncio.TimeoutError:
                pass
            if os.path.lexists(self.paths.socket):
                os.unlink(self.paths.socket)
            log.info("stopped")

    async def _tick_forever(self) -> None:
        while True:
            try:
                self.tick()
            except Exception:
                log.exception("tick failed")
            await asyncio.sleep(1.0)

    async def handle(self, request: dict) -> dict:
        op = request.get("op")
        if op not in ("ping", "shutdown") and request.get("build") != build_id():
            return {"ok": False, "code": "build_mismatch", "build": build_id()}
        try:
            if op == "ping":
                return {"ok": True, "build": build_id(), "pid": os.getpid()}
            if op == "submit":
                return self.submit(request)
            if op == "queue":
                return self.snapshot(include_finished=bool(request.get("all")), session=request.get("session"))
            if op == "show":
                return self.show(int(request["id"]))
            if op == "cancel":
                return self.cancel([int(i) for i in request["ids"]])
            if op == "wait":
                return await self._wait([int(i) for i in request["ids"]], request.get("timeout"))
            if op == "info":
                return self.info()
            if op == "notices":
                return self.notices(str(request["session"]), self.clock(), request.get("seen") or (),
                                    at_submit=bool(request.get("at_submit")))  # fmt: skip
            if op == "seen":
                self._mark_seen(str(request["session"]), request.get("ids") or (), self.clock())
                return {"ok": True}
            if op == "momentum":
                return self.momentum(request)
            if op == "shutdown":
                if self._stop is not None:
                    self._stop.set()
                return {"ok": True}
            return {"ok": False, "code": "bad_request", "error": f"unknown request {op!r}"}
        except (KeyError, TypeError, ValueError) as e:
            return {"ok": False, "code": "bad_request", "error": f"bad {op} request: {e}"}
        except Exception as e:
            log.exception("request %s failed", op)
            return {"ok": False, "code": "error", "error": str(e)}

    async def _wait(self, ids: list[int], timeout: float | None) -> dict:
        for job_id in ids:
            if self.store.get_job(job_id) is None:
                return {"ok": False, "code": "not_found", "error": f"no job {job_id}"}
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            jobs = [self.store.get_job(i) for i in ids]
            done = all(j.state in TERMINAL for j in jobs)
            left = None if deadline is None else deadline - time.monotonic()
            if done or (left is not None and left <= 0):
                views = [self.show(i)["job"] for i in ids]
                return {"ok": True, "jobs": views, "timed_out": not done}
            event = asyncio.Event()
            self._wake.add(event)
            try:
                await asyncio.wait_for(event.wait(), WAIT_RECHECK if left is None else min(left, WAIT_RECHECK))
            except asyncio.TimeoutError:
                pass
            finally:
                self._wake.discard(event)

    # Scheduling ---------------------------------------------------------------------------------

    def _plan(self, now: float) -> tuple[Plan, list[Job], list[Job]]:
        running = self.store.jobs([State.RUNNING])
        queued = self.store.jobs([State.QUEUED])
        borrowing = self._borrowing()
        outside_mem = self._shown_outside_mem = self._outside_mem(now, running)  # dibs top draws it (#120)
        current = plan(
            self.capacity,
            # A job's estimate is of its running time, so the time the machine slept moves its start on.
            # A job that borrows VRAM (#88) counts at what it has used, so nothing starts into it.
            # A job that grew past its RAM reservation counts at its raised limit, so nothing starts into
            # RAM it may yet use (#155).
            [Running(j.id, _holding(j, _limit(j) if j.ram_limit else self._planned_mem(j, now) if borrowing else None,
                                    self._young_vram(j, now)),
                     tuple(j.cpu_ids), tuple(j.gpu_ids), (j.started or now) + self._asleep(j))
             for j in running],
            # A job that asks for less GPU memory than its command's earlier runs held is planned at what
            # they held: nothing holds a job that renders to its reservation (#173, #183).
            [Queued(j.id, replace(j.request, vram=j.history_vram), j.tag, j.submitted, asked_vram=j.request.vram)
             if j.history_vram and j.request.gpus and j.history_vram > j.request.vram
             else Queued(j.id, j.request, j.tag, j.submitted) for j in queued],
            now,
            outside_mem=outside_mem,
            outside_top=(lambda h: (h[0]["name"], h[0]["ram"]) if h else None)(self._ram_holders(now)[0]),
        )
        return current, running, queued

    def _ram_holders(self, now: float) -> tuple[list[dict], int | None]:
        """What holds RAM outside dibs, per app or terminal pane, and what files in a RAM-backed /tmp take,
        read every HOLDERS_EVERY seconds (#164)."""
        if now - self._holders[0] >= HOLDERS_EVERY:
            try:
                holders = procs.outside_holders(minimum=max(GiB, self.capacity.mem // 30))
            except OSError:
                holders = []
            self._holders = (now, holders, procs.tmp_files())
        return self._holders[1], self._holders[2]

    def _history_vram(self, cwd: str, argv: list[str], request: Request, session: str) -> int | None:
        """The most GPU memory earlier runs of the same command here (or the same script) held, at most a
        GPU's worth, for a job that asks for a GPU (#183); None for one that doesn't."""
        if not request.gpus:
            return None
        runs, _ = self._earlier(cwd, argv, session, lambda j: j.peak_vram)
        most = max((j.peak_vram for j in runs), default=0)
        return min(most, max((g.vram for g in self.capacity.gpus), default=0)) or None

    def _young_vram(self, job: Job, now: float) -> int | None:
        """What a running job's command's earlier runs held on the GPU, for its first ten minutes, while its
        own use may not show yet (#183)."""
        ran = now - (job.started or now) - self._asleep(job)
        return job.history_vram if job.history_vram and ran < YOUNG else None

    def _history_mem(self, cwd: str, argv: list[str], session: str) -> int | None:
        """The peak RAM of earlier runs of the same command here (or the same script), if any."""
        runs, _ = self._earlier(cwd, argv, session, lambda j: j.peak_mem)
        return max((j.peak_mem for j in runs), default=None)

    def _earlier(self, cwd: str, argv: list[str], session: str, keep, limit: int = 6) -> tuple[list[Job], str]:
        """Earlier runs to size a job from, newest first, and what they match: runs of the same command
        here ("command"), else of the same script here with other arguments ("script"), else the
        session's recent runs of a script of the same file name in any folder ("session"), since agents
        submit from fresh copies of their code (#229). Only runs `keep` accepts count."""
        runs = [j for j in self.store.earlier_runs(cwd, argv, limit=limit) if keep(j)]
        if runs:
            return runs, "command"
        script = script_of(argv)
        if script:
            runs = [j for j in self.store.earlier_runs_of_script(cwd, script, limit=limit) if keep(j)]
            if runs:
                return runs, "script"
            since = self.clock() - SESSION_RUNS_AGE
            runs = [j for j in self.store.earlier_runs_in_session(session, script_name(argv), since, limit=limit) if keep(j)]
            if runs:
                return runs, "session"
        return [], "command"

    def _borrowing(self) -> bool:
        """Whether jobs may start on RAM other jobs reserved but aren't using (#86): unless switched off,
        and not while the machine is short of memory."""
        return self.config.borrow_ram and self.pressure() < PRESSURE_LIMIT

    def _planned_mem(self, job: Job, now: float) -> int:
        """What a running job counts as holding when RAM is planned by use (#86): the larger of its peak
        so far (which only grows) and earlier runs' peak, plus 25% or 1G, whichever is more, never more
        than its reservation. A job with no earlier runs counts at its whole reservation for its first
        ten minutes, while it may still be growing, and an exclusive job always does."""
        if job.request.exclusive:
            return job.request.mem
        known = max(job.peak_mem or 0, job.history_mem or 0)
        ran = now - (job.started or now) - self._asleep(job)
        if not known or (not job.history_mem and ran < YOUNG):
            return job.request.mem
        return min(job.request.mem, max(int(known * 1.25), known + GiB))

    def _outside_mem(self, now: float, running: list[Job]) -> int:
        """RAM in use outside dibs jobs that eats into what dibs hands out, as its peak over the last
        OUTSIDE_WINDOW seconds, so a job doesn't start into RAM a busy program let go of for a moment.
        A job whose memory dibs can't see counts as holding its whole reservation. What the jobs hold
        is taken off as its peak over the last JOBS_RAM_WINDOW seconds: the machine can still count
        memory a job let go of, or a job that just ended, as in use for a moment, and one sample
        taken then would otherwise count it as outside dibs for the whole OUTSIDE_WINDOW (#72)."""
        if not self._outside or now - self._outside[-1][0] >= 1.0:
            info = self._meminfo()
            if info is not None:
                total, used = info
                jobs = 0
                for job in running:
                    held = self._memory_in_use(job)
                    jobs += held[0] if held is not None else job.request.mem
                self._jobs_held.append((now, jobs))
                while self._jobs_held[0][0] < now - JOBS_RAM_WINDOW:
                    self._jobs_held.popleft()
                jobs = max(held for _, held in self._jobs_held)
                self._outside.append((now, max(0, used - jobs), total))
        while self._outside and self._outside[0][0] < now - OUTSIDE_WINDOW:
            self._outside.popleft()
        if not self._outside:
            return 0
        total = self._outside[-1][2]
        return max(0, max(outside for _, outside, _ in self._outside) - (total - self.capacity.mem))

    def schedule(self) -> None:
        """Start every queued job the plan says can start now."""
        now = self.clock()
        current, running, queued = self._plan(now)
        by_id = {j.id: j for j in queued}
        started: set[str] = set()
        # RAM nobody has reserved: a job that needs more starts on RAM other jobs reserved but aren't
        # using, as a borrower (#86), which a job retried after the machine ran out may not do. Any
        # job may borrow (#148): --retry decides whether one dibs has to stop runs again.
        unreserved = self.capacity.mem - self._outside_mem(now, running) - sum(_limit(j) for j in running)
        for start in current.starts:
            job = by_id[start.job_id]
            borrowed = job.request.mem > unreserved
            if borrowed and job.no_borrow:
                continue
            unreserved -= job.request.mem
            job.borrowed_ram = borrowed
            job.cpu_ids, job.gpu_ids = list(start.cpu_ids), list(start.gpu_ids)
            job.started, job.state = now, State.RUNNING
            started.add(job.session)  # it left the queue, launched or not
            job.sleep_at_start = self.slept()
            self._settle_priority(job, now)
            self.store.update_job(job)
            try:
                job.shim_pid, job.shim_ticks, job.unit = launcher.launch(
                    job, self.paths, use_systemd=self.use_systemd, vendor=getattr(self.backend, "vendor", "nvidia")
                )
            except Exception as e:  # anything that stops the launch fails this job, not the daemon
                job.state, job.ended, job.exit_code = State.FAILED, now, 127
                job.reason = f"couldn't start: {e}"
                self.store.update_job(job)
                self._notify(job)
                continue
            self.store.update_job(job)
        for session in started:  # the jobs that lined up behind the ones that started move up (#150)
            self._reline(session)

    def tick(self) -> None:
        """Called about once a second: notice finished jobs, push stops along, start what can start. A slow
        pass is logged: the relief of a memory shortage needs it to keep up (#191)."""
        began = time.monotonic()
        try:
            self._tick()
        finally:
            took = time.monotonic() - began
            now = self.clock()
            self._passes.append((now, took))
            while self._passes and self._passes[0][0] < now - SHORTAGE_LOG_EVERY:
                self._passes.popleft()
            if took >= SLOW_PASS:
                log.info("a pass took %.1f s", took)

    def _tick(self) -> None:
        now = self.clock()
        launcher.reap()
        for job in self.store.jobs([State.RUNNING]):
            result = launcher.read_exit(self.paths, job.id)
            if result is None and not (job.shim_pid and procs.alive(job.shim_pid, job.shim_ticks)):
                result = launcher.read_exit(self.paths, job.id)  # it may have just been written
                if result is None:
                    self._lose(job, now, "lost: its launcher exited without reporting" + self._shim_said(job))
                    continue
            if result is not None:
                self._finish(job, result, now)
            else:
                self._push_stop(job, now)
        self._watch_memory()
        self._stop_ram_killed(now)
        self._relieve_shortage(now)
        self._watch_beside(now)
        self._watch_gpus(now)
        self.schedule()
        self._grow(now)  # after starting what can start: growth takes only RAM left unreserved
        for watch in (self._watch_outside_cpu, self._watch_load, self._watch_panes):  # only shown by dibs top, after scheduling
            try:
                watch(now)
            except Exception:
                if watch.__name__ not in self._display_failed:
                    self._display_failed.add(watch.__name__)
                    log.exception("%s failed; dibs top won't show it", watch.__name__)
        if now - self._last_retention >= RETENTION_EVERY:
            self._retention(now)
        if now - self._last_alive >= ALIVE_EVERY:
            self._last_alive = now
            self.store.set_meta("alive", json.dumps({"time": now, "slept": self.slept(), "boot": procs.boot_id()}))

    def recover(self) -> None:
        """After a (re)start: finish jobs that ended meanwhile, adopt the ones still running."""
        now = self.clock()
        for session in {j.session for j in self.store.jobs([State.QUEUED])}:
            self._reline(session)  # places an older dibs set, before #150, stay until a job starts
        # The daemon's last note before it stopped; on another boot, the machine restarted since.
        alive = json.loads(self.store.get_meta("alive") or "{}")
        rebooted = bool(alive) and alive.get("boot") != procs.boot_id()
        for job in self.store.jobs([State.RUNNING]):
            result = launcher.read_exit(self.paths, job.id)
            if result is not None:
                self._finish(job, result, now)
            elif job.shim_pid is None and (found := self._find_launcher(job)):
                # dibs died between saving the job as running and recording its launcher.
                job.shim_pid, job.shim_ticks, job.unit = found
                self.store.update_job(job)
            elif not (job.shim_pid and procs.alive(job.shim_pid, job.shim_ticks)) and rebooted:
                # It died with the old boot, after dibs last saw it; the count of time asleep restarted
                # at boot, so what it slept is taken from that note.
                slept = max(0.0, alive["slept"] - job.sleep_at_start) if job.sleep_at_start is not None else 0.0
                ended = max(alive["time"], job.started or alive["time"])
                if job.retry and len(job.attempts) < MAX_RETRIES:
                    job.ended, job.asleep = ended, slept
                    self._retry(job, "lost: the machine restarted while it ran; retrying", job.request)
                else:
                    self._lose(job, ended, "lost: the machine restarted while it ran", asleep=slept)
            elif not (job.shim_pid and procs.alive(job.shim_pid, job.shim_ticks)):
                self._lose(job, now, "lost: its process ended while dibs was not running")
        launcher.sweep_job_tmps(self.paths, keep={j.id for j in self.store.jobs([State.RUNNING])})

    def _find_launcher(self, job: Job) -> tuple[int, int | None, str | None] | None:
        """The live launcher of a job whose pid was never recorded, found by its command line
        (`python -m dibs.shim JOB_DIR`), with its systemd unit if it runs in one."""
        job_dir = str(self.paths.job_dir(job.id))
        unit = launcher.unit_name(self.paths.instance, job.id)
        for name in os.listdir("/proc"):
            if name.isdigit() and (procs.cmdline(int(name)) or [])[-3:] == ["-m", "dibs.shim", job_dir]:
                pid = int(name)
                if procs.alive(pid):
                    in_unit = (procs.cgroup(pid) or "").endswith(f"/{unit}.scope")
                    return pid, procs.start_ticks(pid), unit if in_unit else None
        return None

    # Watching memory and the GPUs ----------------------------------------------------------------

    def _memory_in_use(self, job: Job) -> tuple[int, int, tuple[int, int]] | None:
        """What a running job holds (anon + shmem), the shmem part of it and how much of that part its
        processes had mapped (procs.memory_in_use), read from its own scope; None if dibs can't see it."""
        if not (job.unit and job.shim_pid):
            return None
        cg = procs.cgroup(job.shim_pid)
        if not cg or not cg.endswith(f"/{job.unit}.scope"):
            return None
        return procs.memory_in_use(cg)

    def _watch_memory(self) -> None:
        """Keep each running job's peak of memory in use (anon + shmem), read from its own scope.
        memory.peak also counts file cache, so a job that wrote big files looked as if it had filled
        its reservation."""
        self._held = {}  # only this tick's samples: a retried job never grows on its last try's reading
        for job in self.store.jobs([State.RUNNING]):
            held = self._memory_in_use(job)
            if held is not None:
                self._held[job.id] = held[0]
            if held is not None and (job.peak_mem is None or held[0] > job.peak_mem):
                job.peak_mem, job.peak_shmem, mapped = held
                job.peak_shmem_mapped = list(mapped)
                self.store.update_job(job)

    def _ram_kills(self, job: Job) -> tuple[int, bool]:
        """How many of a running job's processes the kernel killed for lack of RAM, read from its own
        scope, and whether at the job's own limit; none when dibs can't see them."""
        cg = procs.cgroup(job.shim_pid) if job.unit and job.shim_pid else None
        if not cg or not cg.endswith(f"/{job.unit}.scope"):
            return 0, False
        return procs.oom_kills(procs._read(f"{procs.CGROUP_ROOT}{cg}/memory.events"))

    def _stop_ram_killed(self, now: float) -> None:
        """Stop the rest of a job once the kernel has killed one of its processes for lack of RAM (#125).
        The kernel kills only the process that went over, and the job's scope lives on so its launcher
        can report: a parent waiting on a killed worker, like a Python process pool, would wait forever
        and keep the job's reservation."""
        for job in self.store.jobs([State.RUNNING]):
            if job.stop_reason is not None or job.cancel_requested is not None:
                continue  # already on its way out, for its own reason
            kills, at_limit = self._ram_kills(job)
            if not kills:
                continue
            job.stop_reason = _ram_kill_reason(job, {"hit_limit": at_limit})
            job.stop_requested = now
            self.store.update_job(job)
            log.info("job %d %s; stopping the rest of it", job.id, job.stop_reason)
            self._push_stop(job, now)

    def _grow(self, now: float) -> None:
        """Let a running job grow past its RAM reservation, a step at a time, when what it holds comes
        within half a step of its limit (#155). Only into RAM no job has reserved and nothing outside dibs
        uses: every running job counts at its reservation or raised limit, whichever is larger, never at
        what it uses, since a job may grow back into RAM it reserved. The planner counts a grown job at
        its new limit at once, so that RAM is never handed out twice. Limits only rise during a run."""
        if self.set_mem_limit is None:
            return
        running = self.store.jobs([State.RUNNING])
        outside = self._outside_mem(now, running)
        room = self.capacity.mem - outside - sum(_limit(j) for j in running)
        # Failing that, RAM other jobs reserved but aren't using, by the plan by use that lets a waiting
        # job start in it (#149, #182): a job at its limit has work to lose, so it goes ahead of them.
        lend = self.capacity.mem - outside - sum(_limit(j) if j.ram_limit else self._planned_mem(j, now) for j in running)
        lend = lend if self._borrowing() else 0
        for job in sorted(running, key=lambda j: (j.started or 0, j.id)):
            held = self._held.get(job.id)
            if held is None or job.request.exclusive or job.stop_requested is not None or job.cancel_requested is not None:
                continue
            limit, step = _limit(job), max(GiB, int(job.request.mem * GROW_SHARE))
            # Near its limit: within half a step, or an eighth of its limit for a small job, whose 1G step
            # would otherwise make its usual peak look like growth.
            borrowing = room < step
            if held < limit - min(step // 2, limit // 8) or (borrowing and (job.no_borrow or lend < step)):
                continue
            if now - self._grow_failed.get(job.id, -math.inf) < GROW_RETRY:
                continue
            if not self.set_mem_limit(job, limit + step):
                self._grow_failed[job.id] = now  # a slow or failing systemd shouldn't stall every tick
                continue
            if job.ram_limit is None:  # past its reservation now: if the machine runs out, it goes first
                launcher.raise_oom_score(job)
            job.ram_limit = limit + step
            job.grew_borrowed = job.grew_borrowed or borrowing  # given back first in a shortage (#182)
            room, lend = room - step, lend - step
            self.store.update_job(job)
            log.info("job %d grew past its %s RAM reservation into RAM %s: its limit is now %s", job.id,
                     format_size(job.request.mem), "other jobs reserved but weren't using" if borrowing else "nobody reserved",
                     format_size(job.ram_limit))  # fmt: skip

    def notices(self, session: str, now: float, seen: Sequence[int] = (), at_submit: bool = False) -> dict:
        """What the guard stopped of this session's since it was last told, each told once: the note in a
        stopped program's own output can be lost, in a pipe through grep or a log nobody reads (#135).
        And its jobs that failed in the last hour, at each of its dibs commands until one of them looks
        at the job (seen: the jobs this command shows), or it runs the same command again (#176): a failed
        job drops out of dibs queue as a finished one does, and a loop's command may be the one told."""
        self._mark_seen(session, seen, now)
        key = f"told:{session}"
        told = float(self.store.get_meta(key) or 0.0)
        events = [e for e in self.store.events(min(max(told, now - TOLD_WINDOW), now - FAILED_FOR)) if e.get("session") == session]
        stops = [e for e in events if e["kind"] in ("stop", "heavy", "idle_gpu") and e["time"] > told and e.get("message")]
        if stops:
            self.store.set_meta(key, repr(stops[-1]["time"]))
        told_now = [{"time": e["time"], "message": e["message"], "kind": e["kind"]} for e in stops]
        failed = [e for e in events if e["kind"] == "failed" and now - FAILED_FOR <= e["time"] <= now]
        told_now += [n for n in (self._failure_notice(e, session) for e in failed) if n is not None]
        if not at_submit:  # advice on a request goes out at the session's other commands (#169)
            key = f"told_advice:{session}"
            advised = float(self.store.get_meta(key) or 0.0)
            advice = [e for e in self.store.events(max(advised, now - TOLD_WINDOW))
                      if e["kind"] == "advice" and e.get("session") == session and e["time"] > advised]  # fmt: skip
            if advice:
                self.store.set_meta(key, repr(advice[-1]["time"]))
            told_now += [{"time": e["time"], "message": e["message"], "kind": "advice"} for e in advice]
        return {"ok": True, "notices": told_now}

    def _advise(self, job: Job, view: dict, now: float) -> None:
        """Tell the session, at its next dibs command other than a submit, when this job's request doesn't
        match how the same command or script ran before (#169): little of submit's own output reaches
        agents, and a loop's submit may be the one told. Each kind at most once an hour per script."""
        script = script_of(job.argv)
        name = os.path.basename(script) if script and not script.startswith("-m ") else (script or os.path.basename(job.argv[0]))
        by_command = self.store.earlier_runs(job.cwd, job.argv, limit=6)
        by_script = self.store.earlier_runs_of_script(job.cwd, script, limit=6) if script else []
        by_session = []
        if script and not (by_command or by_script):  # a fresh copy of the code: the session's other folders (#229)
            by_session = self.store.earlier_runs_in_session(job.session, script_name(job.argv), now - SESSION_RUNS_AGE, limit=6)
        runs, req, said = by_script or by_command or by_session, job.request, []
        last = (by_command or by_script)[:1]
        if last and last[0].state == State.FAILED and (last[0].reason or "").startswith("killed: went over its"):
            limit = max(last[0].request.mem, last[0].ram_limit or 0)
            if req.mem <= limit:
                more = self._more_ram(last[0])
                said.append(("ram_kill", f"Job {job.id} asks for {format_size(req.mem)} of RAM, and the last run of this "
                             f"command (#{last[0].id}) was killed at its {format_size(limit)} RAM limit"
                             + (f"; dibs suggested --mem {format_size(more)}" if more else "")
                             + f". dibs submit --replace {job.id} with more RAM fixes it before it runs."))  # fmt: skip
        peaks = [r.peak_mem for r in runs if r.peak_mem]
        if "RAM" in (view.get("lacks") or []) and len(peaks) >= ADVISE_RUNS and max(peaks) < req.mem / 2:
            said.append(("ram_high", f"Job {job.id} waits for {format_size(req.mem)} of RAM; the last {len(peaks)} runs of "
                         f"{name} peaked at {format_size(max(peaks))} at most: --mem {format_size(suggest_size(max(peaks)))} "
                         f"would do (dibs submit --replace {job.id})."))  # fmt: skip
        if req.gpus:
            vram = [r.peak_vram for r in runs if r.peak_vram is not None and r.request.gpus]
            if len(vram) >= ADVISE_RUNS and max(vram) < req.vram / 4:
                if not max(vram):
                    said.append(("vram_high", f"The last {len(vram)} runs of {name} used no GPU memory, and job {job.id} "
                                 f"asks for {format_size(req.vram)}: leave --vram out, so the GPU's memory stays free "
                                 "for jobs that use it."))  # fmt: skip
                else:
                    said.append(("vram_high", f"The last {len(vram)} runs of {name} held at most {format_size(max(vram))} "
                                 f"of GPU memory, and job {job.id} asks for {format_size(req.vram)}: --vram "
                                 f"{format_size(suggest_size(max(vram)))} would do."))  # fmt: skip
            held = [v for v in vram if v]
            if held and max(held) > req.vram:
                said.append(("vram_low", f"Job {job.id} asks for {format_size(req.vram)} of GPU memory, and the last "
                             f"{len(held)} runs of {name} held up to {format_size(max(held))}: dibs plans it at that "
                             f"until it shows its own use, and --vram {format_size(suggest_size(max(held)))} would ask "
                             "for it."))  # fmt: skip
        done = [r for r in runs if r.state == State.DONE and r.started is not None and r.ended is not None]
        if len(done) >= ADVISE_RUNS:
            took = [r.ended - r.started - (r.asleep or 0.0) for r in done[:ADVISE_RUNS]]
            ratios = sorted(t / r.request.time for t, r in zip(took, done) if r.request.time)
            median = ratios[len(ratios) // 2] if ratios else 1.0
            estimates = sorted({r.request.time for r in done[:ADVISE_RUNS]})
            est = format_duration(estimates[0]) + (f"-{format_duration(estimates[-1])}" if len(estimates) > 1 else "")
            times = ", ".join(format_duration(t) for t in took)
            if median < 0.25:
                said.append(("estimate", f"The last {len(took)} runs of {name} took {times}, against {est} estimates: "
                             "a closer --time gets its jobs started sooner."))  # fmt: skip
            elif median > 2:
                said.append(("estimate", f"The last {len(took)} runs of {name} took {times}, over twice their {est} "
                             "estimates: a longer --time keeps predictions right."))  # fmt: skip
        for kind, message in said:
            key = f"advised:{job.session}:{kind}:{script or ' '.join(job.argv)}"
            if now - float(self.store.get_meta(key) or -math.inf) < ADVISE_EVERY:
                continue
            self.store.set_meta(key, repr(now))
            self.store.add_event("advice", {"session": job.session, "job": job.id, "message": message}, now)

    def _mark_seen(self, session: str, ids: Sequence[int], now: float) -> None:
        """The session looked at these jobs (dibs show, wait or logs): the ended ones are no longer told (#176)."""
        for job_id in ids:
            looked = self.store.get_job(int(job_id))
            if looked is not None and looked.session == session and looked.ended is not None:
                self.store.set_meta(f"seen:{looked.id}", repr(now))

    def _failure_notice(self, event: dict, session: str) -> dict | None:
        """A failed job's notice (#176), unless the session has looked at it or run the same command again."""
        job = self.store.get_job(int(event["job"]))
        if job is None or job.state != State.FAILED or self.store.get_meta(f"seen:{job.id}") is not None:
            return None
        ended = job.ended or event["time"]
        later = self.store.jobs([State.QUEUED, State.RUNNING]) + self.store.finished_since(ended)
        if any(j.session == session and j.id != job.id and j.submitted >= ended and (j.argv, j.cwd) == (job.argv, job.cwd)
               for j in later):  # fmt: skip
            return None
        ran = ended - job.started - (job.asleep or 0.0) if job.started is not None else None
        return {"time": event["time"], "kind": "failed", "job": job.id, "exit_code": job.exit_code, "ran": ran,
                "reason": job.reason or "", "message": ""}  # fmt: skip

    def _watch_panes(self, now: float) -> None:
        """Tell an agent when a program in its terminal keeps a sixth of the machine's cores busy outside
        dibs for two minutes (#158), from each terminal pane's cgroup: at most once an hour per program
        in a pane, so one program's notice doesn't hold back another's (#170)."""
        if now - self._panes_at < PANE_EVERY:
            return
        self._panes_at = now
        seen = self.panes()
        for path in set(self._pane_samples) - set(seen):
            del self._pane_samples[path]
        heavy = max(2.0, len(self.capacity.cpus) * HEAVY_SHARE)
        for path, usec in seen.items():
            samples = self._pane_samples.setdefault(path, deque())
            samples.append((now, usec))
            while samples[0][0] < now - HEAVY_FOR - PANE_EVERY:
                samples.popleft()
            (t0, u0), (t1, u1) = samples[0], samples[-1]
            if t1 - t0 < HEAVY_FOR:
                continue
            tp, up = samples[-2]  # there's one: the samples span HEAVY_FOR
            cores = (u1 - u0) / 1e6 / (t1 - t0)
            # Busy over the last interval too: once a heavy program ends, the two-minute average stays
            # high for a while, and the busiest program left would be named with it (#213).
            recent = (u1 - up) / 1e6 / (t1 - tp)
            if cores < heavy or recent < heavy:
                continue
            event = guard.heavy_notice(path, cores, t1 - t0, least=heavy, recent=recent)
            if event is None:
                continue
            # The same program is the same script, whatever its arguments: a sweep gets one notice an hour.
            program = (path, script_of(event["command"]) or next(iter(event["command"]), ""))
            if now - self._told_programs.get(program, -math.inf) < TELL_EVERY:
                continue
            self._told_programs[program] = now
            self.store.add_event("heavy", event, now)
            log.info("told the agent in %s: %s kept %.0f cores busy outside dibs", path.rsplit("/", 1)[-1],
                     event["command"], cores)  # fmt: skip

    def _relieve_shortage(self, now: float) -> None:
        """While the machine stays short of memory, stop one program per stretch of shortage: the largest
        program outside dibs that holds a tenth of the RAM or more (#135), otherwise the most recently
        started job on borrowed RAM (#114). The kernel's OOM killer acts only after minutes of thrashing,
        and then kills whatever is largest: a job, or the program an agent runs in its own terminal."""
        if self.pressure() < EVICT_PRESSURE or not self._short_of_ram():
            if self._pressured_since is not None:
                self.store.delete_meta(["short_since"])
            self._pressured_since = None
            return
        if self._pressured_since is None:
            self._pressured_since = now
            self.store.set_meta("short_since", repr(now))  # kept across a restart (#190)
            return
        if now - self._pressured_since < EVICT_AFTER:
            return
        self._pressured_since = now
        self.store.set_meta("short_since", repr(now))
        # RAM dibs lent a job to grow into comes back first (#182), then a program outside dibs (#135),
        # then RAM a job started on (#114).
        self._relief_note = "nothing to stop"
        if not self._give_back_growth(now) and not self._stop_ram_hog(now):
            self._give_back_borrowed(now)
        self._log_shortage(now)

    def _log_shortage(self, now: float) -> None:
        """Once a minute while short of memory: how short, what the relief did, and the longest pass of
        the daemon's loop (#191), so a shortage it didn't relieve can be told from one it couldn't see."""
        if now - self._shortage_logged < SHORTAGE_LOG_EVERY:
            return
        self._shortage_logged = now
        info = self._meminfo()
        total, used = info if info is not None else (self.capacity.mem, self.capacity.mem)
        passes = f"; longest pass in the last minute {max(t for _, t in self._passes):.1f} s" if self._passes else ""
        log.info("short of memory: %s of %s available, memory pressure %.0f%%: %s%s", format_size(max(0, total - used)),
                 format_size(total), self.pressure(), self._relief_note, passes)  # fmt: skip

    def _give_back_growth(self, now: float) -> bool:
        """Stop the most recently started job that grew into RAM other jobs reserved (#182): dibs lent it
        that RAM, so in a shortage it's the first to go. Whether one was stopped."""
        growers = [j for j in self.store.jobs([State.RUNNING])
                   if j.grew_borrowed and j.stop_requested is None and j.cancel_requested is None]  # fmt: skip
        if not growers:
            return False
        job = max(growers, key=lambda j: (j.started or 0, j.id))
        self._relief_note = f"stopped job {job.id}, which had grown into RAM other jobs reserved"
        job.evicted = True
        job.stop_reason = ("stopped to give back RAM other jobs reserved, which it had grown into: the machine was "
                           "short of memory")  # fmt: skip
        job.stop_requested = now
        self.store.update_job(job)
        log.info("job %d %s", job.id, job.stop_reason)
        self._push_stop(job, now)
        return True

    def _stop_ram_hog(self, now: float) -> bool:
        """Stop the largest program outside dibs that holds a tenth of the RAM or more, with a note in its
        output on why and on running it as a job instead (#135). Whether one was stopped."""
        if self.config.guard == "off":
            return False
        info = self._meminfo()
        total, used = info if info is not None else (self.capacity.mem, self.capacity.mem)
        minimum = int(total * HOG_SHARE)
        running = self.store.jobs([State.RUNNING])
        shims = {j.shim_pid: j.id for j in running if j.shim_pid}
        units = {j.unit: j.id for j in running if j.unit}
        mine = {os.getpid(), *self._ancestors}
        left: list[tuple[int, int, str]] = []
        hog = guard.ram_hog(
            self.programs(minimum),
            minimum=minimum,
            in_dibs=lambda pid: pid in mine or self._job_of(pid, shims, units) is not None,
            mode=self.config.guard,
            allow=self.config.guard_allow,
            since=self.guard_since,
            my_uid=os.getuid(),
            parents=self.guard_parents,
            instance=self.paths.instance,
            desktop=self.config.guard_desktop,
            left=left,
        )
        if hog is None:
            if left:
                pid, held, why = left[0]
                self._relief_note = f"nothing to stop: the largest program outside dibs, pid {pid} holding {format_size(held)}, is left alone ({why})"
            else:
                self._relief_note = f"nothing to stop: no program outside dibs holds {format_size(minimum)} or more"
            return False
        pid, held, kind = hog
        if kind == "warn":
            log.info("guard would stop pid %d, holding %s of RAM while the machine is short of memory (warn mode): %s",
                     pid, format_size(held), procs.cmdline(pid))  # fmt: skip
            return False
        event = guard.stop_for_ram(pid, held, max(0, total - used), total)
        if event is None:
            return False
        self.store.add_event("stop", event, now)
        log.info("guard stopped pid %d, holding %s of RAM while the machine was short of memory: %s",
                 pid, format_size(held), event["command"])  # fmt: skip
        self._relief_note = f"stopped pid {pid}, holding {format_size(held)}"
        return True

    def _give_back_borrowed(self, now: float) -> None:
        """Stop the most recently started job on borrowed RAM, to run again within RAM nobody reserved
        (#114)."""
        borrowers = [
            j for j in self.store.jobs([State.RUNNING])
            if j.borrowed_ram and j.stop_requested is None and j.cancel_requested is None
        ]  # fmt: skip
        if not borrowers:
            return
        job = max(borrowers, key=lambda j: (j.started or 0, j.id))
        self._relief_note = f"stopped job {job.id}, which started on RAM other jobs reserved"
        job.evicted = True
        job.stop_reason = "stopped to give back borrowed RAM: the machine was short of memory"
        job.stop_requested = now
        self.store.update_job(job)
        log.info("job %d %s", job.id, job.stop_reason)
        self._push_stop(job, now)

    def _short_of_ram(self) -> bool:
        """Whether little of the machine's RAM is available. Pressure alone doesn't say so: one job
        stalling inside itself, at its own limit or while the kernel compacts memory for huge pages,
        raises it with most of the RAM free (#129), and so does the minute after a real shortage, while
        pages pushed out to swap come back in (#131). Unreadable, it leaves the pressure to decide."""
        info = self._meminfo()
        return info is None or info[0] - info[1] < info[0] * SHORT_OF_RAM

    def _watch_beside(self, now: float) -> None:
        """Beside an exclusive job (a timing run), sample the CPU other programs keep busy, so dibs show
        can say whether its timing was disturbed (#112): the machine's busy cores minus its own."""
        for job in self.store.jobs([State.RUNNING]):
            if not job.request.exclusive or not (job.unit and job.shim_pid):
                continue
            busy = self.cpu_busy()
            own = self._job_cpu_seconds(job)
            if busy is None or own is None:
                continue
            last = self._beside.get(job.id)
            if last is not None and now - last[0] < 4.0:
                continue
            self._beside[job.id] = (now, busy, own)
            if last is None:
                continue
            others = max(0.0, (busy - last[1]) - (own - last[2])) / (now - last[0])
            stats = dict(job.background or {"seconds": 0.0, "core_seconds": 0.0, "max": 0.0})
            stats["seconds"] += now - last[0]
            stats["core_seconds"] += others * (now - last[0])
            stats["max"] = max(stats["max"], others)
            stats["mean"] = stats["core_seconds"] / stats["seconds"]
            job.background = stats
            self.store.update_job(job)

    def _job_cpu_seconds(self, job: Job) -> float | None:
        """The CPU time a running job's own scope has used, in core-seconds; None if it can't be read."""
        cg = procs.cgroup(job.shim_pid) if job.unit and job.shim_pid else None
        if not cg or not cg.endswith(f"/{job.unit}.scope"):
            return None
        usage = procs.keyed_numbers(procs._read(f"{procs.CGROUP_ROOT}{cg}/cpu.stat")).get("usage_usec")
        return None if usage is None else usage / 1e6

    def _watch_outside_cpu(self, now: float) -> None:
        """Sample the cores programs outside dibs keep busy, for dibs top (#120): the machine's busy
        core-seconds less the running jobs' own, over the last OUTSIDE_CPU_WINDOW seconds. Unknown
        (None) when a job's own use can't be read, a job started or ended between samples, or the
        machine slept: each would count a job's work, or nothing, as outside dibs. Only shown: the
        planner doesn't use it."""
        if self._cpu_samples and now - self._cpu_samples[-1][0] < OUTSIDE_CPU_EVERY:
            return
        running = self.store.jobs([State.RUNNING])
        busy, own = self.cpu_busy(), [self._job_cpu_seconds(j) for j in running]
        if busy is None or None in own:
            self._cpu_samples.clear()
            self._outside_cpu = None
            return
        sample = (now, busy, sum(o for o in own if o is not None), frozenset(j.id for j in running), self.slept())
        if self._cpu_samples and (
            self._cpu_samples[-1][3] != sample[3] or sample[4] - self._cpu_samples[-1][4] > 1.0
        ):
            self._cpu_samples.clear()
        self._cpu_samples.append(sample)
        while now - self._cpu_samples[0][0] > OUTSIDE_CPU_WINDOW:
            self._cpu_samples.popleft()
        first = self._cpu_samples[0]
        if now - first[0] <= 0:
            self._outside_cpu = None
            return
        self._outside_cpu = max(0.0, (busy - first[1]) - (sample[2] - first[2])) / (now - first[0])

    def _watch_load(self, now: float) -> None:
        """Keep the machine's busy core-seconds every LOAD_EVERY seconds over LOAD_WINDOW, for dibs top's
        load row (#122). Its own samples: they don't reset when jobs start or end."""
        if self._load and now - self._load[-1][0] < LOAD_EVERY:
            return
        busy = self.cpu_busy()
        if busy is None:
            return
        self._load.append((now, busy, self.slept()))
        while len(self._load) > LOAD_WINDOW / LOAD_EVERY + 1:
            self._load.popleft()

    def _load_history(self) -> list[float | None]:
        """Busy cores between consecutive samples, oldest first; None for an interval the machine slept."""
        samples = list(self._load)
        return [
            None if s1 - s0 > 1.0 or t1 <= t0 else max(0.0, (b1 - b0) / (t1 - t0))
            for (t0, b0, s0), (t1, b1, s1) in zip(samples, samples[1:])
        ]

    def _watch_idle(self, now: float, running: dict[int, Job], held: dict[int, dict[int, dict[int, int]]]) -> None:
        """Record which jobs hold GPU memory with the GPU idle (#180), and tell a job's session once its
        hold has gone on IDLE_AFTER while other jobs wait for GPU memory (#186): a job keeps its memory
        until its processes free it or exit, and one did so, idle, for hours. Nothing is stopped."""
        if now - self._idle_at < IDLE_EVERY:
            return
        self._idle_at = now
        util = self.backend.utilization() if held else {}
        per_process = self.backend.process_utilization() if held else None
        size = {g.index: g.vram for g in self.capacity.gpus}
        waiting: int | None = None  # jobs waiting for GPU memory, counted only when someone is to be told
        for job in running.values():
            holds = {g: sum(pids.values()) for g, pids in held.get(job.id, {}).items()}
            big = {g: amount for g, amount in holds.items() if amount >= max(GiB, size.get(g, 0) * IDLE_SHARE)}
            idle = bool(big) and util is not None and all(
                self._gpu_idle(job.id, g, held, util, per_process) for g in big
            )
            if not idle:
                if job.gpu_idle_since is not None:
                    self._end_idle(job, now)
                    self.store.update_job(job)
                continue
            if job.gpu_idle_since is None:
                job.gpu_idle_since, job.gpu_idle_held = now, 0
            job.gpu_idle_held = max(job.gpu_idle_held or 0, sum(big.values()))
            if now - job.gpu_idle_since >= IDLE_AFTER and (job.gpu_idle_told is None or now - job.gpu_idle_told >= IDLE_TELL_EVERY):
                if waiting is None:
                    current, _, _ = self._plan(now)
                    waiting = sum(1 for p in current.predictions.values() if {"VRAM", "GPUs"} & set(p.lacks))
                if waiting:
                    job.gpu_idle_told = now
                    self.store.add_event("idle_gpu", {"session": job.session, "job": job.id,
                                                      "message": _idle_message(job, now, waiting)}, now)  # fmt: skip
                    log.info("told the session of job %d: it holds %s of GPU memory with the GPU idle", job.id,
                             format_size(job.gpu_idle_held))  # fmt: skip
            self.store.update_job(job)

    def _gpu_idle(self, job_id: int, gpu: int, held: dict, util: dict[int, float], per_process: dict[int, float] | None) -> bool:
        """Whether a job's processes on a GPU are using it: by each process's use where the driver says,
        else by the whole GPU's, when no other job holds memory there."""
        if per_process is not None:
            return all(per_process.get(pid, 0.0) < IDLE_UTIL for pid in held[job_id][gpu])
        alone = all(gpu not in gpus for other, gpus in held.items() if other != job_id)
        return alone and gpu in util and util[gpu] < IDLE_UTIL

    def _end_idle(self, job: Job, now: float) -> None:
        """An idle stretch ends (the job used the GPU again, freed its memory, or ended): it counts in the
        job's record if it went on IDLE_AFTER or more (#180)."""
        if job.gpu_idle_since is not None and now - job.gpu_idle_since >= IDLE_AFTER:
            job.gpu_idle += now - job.gpu_idle_since
        job.gpu_idle_since = None

    def _job_of(self, pid: int, shims: dict[int, int], units: dict[str, int]) -> int | None:
        """The running job a process belongs to: by its systemd scope, else by its shim ancestor."""
        cg = procs.cgroup(pid) or ""
        scope = cg.rsplit("/", 1)[-1].removesuffix(".scope")
        if scope in units:
            return units[scope]
        for _ in range(64):  # the shim is a subreaper, so job processes always descend from it
            if pid in shims:
                return shims[pid]
            parent = procs.parent(pid)
            if not parent or parent == pid:
                return None
            pid = parent
        return None

    def _watch_gpus(self, now: float) -> None:
        """Hold jobs to their VRAM reservations, and run the guard on everything else."""
        seen = self.backend.processes() if self.capacity.gpus else []
        managed = {g.index for g in self.capacity.gpus}
        graphics = []  # processes rendering through EGL or OpenGL: they count only toward job peaks (#173)
        if seen is not None:
            graphics = [p for p in self.backend.graphics_processes() or [] if p.gpu_index in managed]
        if not seen and not graphics:
            self.outside = []
            self._outside_vram = {}
            self._watch_idle(now, {j.id: j for j in self.store.jobs([State.RUNNING])}, {})
            return
        seen = [p for p in seen or [] if p.gpu_index in managed]
        running = {j.id: j for j in self.store.jobs([State.RUNNING])}
        shims = {j.shim_pid: j.id for j in running.values() if j.shim_pid}
        units = {j.unit: j.id for j in running.values() if j.unit}
        job_of_pid = {}
        for p in seen:
            job_id = self._job_of(p.pid, shims, units)
            if job_id is not None:
                job_of_pid[p.pid] = job_id
        verdicts = guard.classify(
            seen,
            job_of_pid=job_of_pid,
            mode=self.config.guard,
            allow=self.config.guard_allow,
            desktop=self.config.guard_desktop,
            since=self.guard_since,
            my_uid=os.getuid(),
            parents=self.guard_parents,
            instance=self.paths.instance,
        )
        self._remember(verdicts)

        usage: dict[int, dict[int, int]] = {}
        held: dict[int, dict[int, dict[int, int]]] = {}  # job -> GPU -> pid -> bytes, for its peak
        for v in verdicts:
            if v.kind == "job" and v.used is not None:
                per_gpu = usage.setdefault(v.job_id, {})
                per_gpu[v.gpu_index] = per_gpu.get(v.gpu_index, 0) + v.used
                pids = held.setdefault(v.job_id, {}).setdefault(v.gpu_index, {})
                pids[v.pid] = pids.get(v.pid, 0) + v.used
        # A job that renders through EGL or OpenGL holds VRAM only the graphics-process list shows
        # (#173). It counts toward the job's peak, and a process holding both kinds of context counts
        # once. The watch below still holds jobs to their reservations by compute use alone.
        for p in graphics:
            job_id = self._job_of(p.pid, shims, units) if p.used is not None else None
            if job_id is not None:
                pids = held.setdefault(job_id, {}).setdefault(p.gpu_index, {})
                pids[p.pid] = max(pids.get(p.pid, 0), p.used)
        self._watch_idle(now, running, held)
        # A job may borrow VRAM beyond its reservation that no other job holds and no program outside
        # dibs uses (#88); it's stopped only when it reaches into theirs. A job holds its reservation,
        # or more while it borrows.
        outside_use: dict[int, int] = {}
        for v in verdicts:
            if v.kind != "job" and v.used is not None:
                outside_use[v.gpu_index] = outside_use.get(v.gpu_index, 0) + v.used
        self._outside_vram = {str(g): used for g, used in outside_use.items() if used}
        capacity_of = {g.index: g.vram for g in self.capacity.gpus}

        def holds(j: Job, g: int) -> int:
            return max(j.request.vram, usage.get(j.id, {}).get(g, 0)) if g in j.gpu_ids else 0

        def allowed(j: Job, g: int) -> int:
            if g not in j.gpu_ids:
                return 0
            others = sum(holds(o, g) for o in running.values() if o.id != j.id)
            return max(j.request.vram, capacity_of.get(g, 0) - others - outside_use.get(g, 0))

        for job_id, per_gpu_pids in held.items():
            job = running[job_id]
            per_gpu = usage.get(job_id, {})
            peak = max(sum(pids.values()) for pids in per_gpu_pids.values())
            changed = job.peak_vram is None or peak > job.peak_vram
            if changed:
                job.peak_vram = peak
            over = [(g, used) for g, used in sorted(per_gpu.items()) if used > allowed(job, g)]
            if over and job.stop_reason is None:
                gpu, used = over[0]
                reserved = job.request.vram if gpu in job.gpu_ids else 0
                total = next((g.vram for g in self.capacity.gpus if g.index == gpu), None)
                early = now - (job.started or now) - self._asleep(job) < EARLY_STOP
                ask = vram_to_ask(max(used, job.peak_vram or 0), total, early=early)
                job.retry_vram = total if ask == "all" else parse_size(ask)
                job.stop_reason = (
                    f"stopped: used {format_size_against(used, reserved)} VRAM on GPU {gpu}, "
                    f"reserved {format_size(reserved)}"
                    + (", in its first minute, so it was likely still growing" if early else "")
                )
                job.stop_requested = now
                changed = True
                log.info("job %d %s", job.id, job.stop_reason)
            if changed:
                self.store.update_job(job)
            if job.stop_requested is not None:
                self._push_stop(job, now)

        for event in guard.enforce(verdicts):
            self.store.add_event("stop", event, now)
            log.info("guard stopped pid %d on GPU %d: %s", event["pid"], event["gpu"], event["command"])
        self.outside = [
            {
                "pid": v.pid,
                "gpu": v.gpu_index,
                "used": v.used,
                "command": procs.cmdline(v.pid) or [],
                "why": guard.WHY[v.kind],
            }
            for v in verdicts
            if v.kind in guard.WHY
        ]

    def _remember(self, verdicts: list) -> None:
        """Remember the programs that started GPU processes which were running when the guard
        started (torchrun, a Jupyter server, a pipeline script), so what they start next is spared.
        Never init, the session manager or dibs's own ancestors: they adopt orphans, and sparing
        them would spare everything."""
        remembered = {p for p in self.guard_parents if procs.alive(p[0], p[1])}
        for v in verdicts:
            if v.kind != "preexisting":
                continue
            parent = procs.parent(v.pid) or 0
            if parent <= 1 or parent in self._ancestors or procs.exe_name(parent) == "systemd":
                continue
            ticks = procs.start_ticks(parent)
            if ticks is not None:
                remembered.add((parent, ticks))
        if remembered != self.guard_parents:
            self.guard_parents = remembered
            self.store.set_meta("guard_parents", json.dumps(sorted(remembered)))

    # Internals ----------------------------------------------------------------------------------

    def _push_stop(self, job: Job, now: float) -> None:
        """Move a requested cancel or stop along: SIGTERM until it lands, then SIGKILL after a grace."""
        asked = job.cancel_requested if job.cancel_requested is not None else job.stop_requested
        if asked is None:
            return
        if now - asked >= KILL_AFTER:
            launcher.signal_job(job, signal.SIGKILL)
        elif job.id not in self._terminated and launcher.signal_job(job, signal.SIGTERM):
            self._terminated.add(job.id)

    def _finish(self, job: Job, result: dict, now: float) -> None:
        launcher.remove_job_tmp(self.paths, job.id)
        if job.sleep_at_start is not None:  # counted to the command's exit, so a later sleep isn't added
            slept = result.get("slept")
            job.asleep = max(0.0, (slept if slept is not None else self.slept()) - job.sleep_at_start)
        job.exit_code = result.get("exit_code")
        job.ended = result.get("ended") or now
        self._end_idle(job, now)  # an idle hold ends with the job (#180)
        job.cpu_seconds = result.get("cpu_seconds")
        if job.peak_mem is None:  # never sampled while it ran: the launcher's figure is all there is
            job.peak_mem = result.get("peak_mem")
        if job.request.vram and job.peak_vram is None and self._sees_processes:
            job.peak_vram = 0  # it reserved VRAM and never showed on a GPU dibs could see (#139)
        at_limit = bool(result.get("oom_killed")) and result.get("hit_limit") is not False
        if at_limit:  # it reached its limit, maybe after dibs's last sample: that's what killed it (#127)
            job.peak_mem = max(job.peak_mem or 0, _limit(job))
        retry = self._retry_with(job, result) if job.cancel_requested is None else None
        if retry is not None:
            request, how, reason = retry
            self._retry(job, f"{reason}; retrying" + (f" {how}" if how else ""), request)
            return
        if job.cancel_requested is not None:
            job.state = State.CANCELLED
        elif job.exit_code == 0 and job.stop_reason is None and not result.get("oom_killed"):
            job.state = State.DONE  # a process killed for RAM failed it even if the rest exited 0 (#125)
        else:
            job.state = State.FAILED
        if job.stop_reason:
            job.reason = job.stop_reason + (f"; resubmit with --vram {self._vram_text(job.retry_vram)}" if job.retry_vram else "")
        elif job.cancel_requested is not None:
            job.reason = "cancelled"
        elif result.get("oom_killed"):
            job.reason = _ram_kill_reason(job, result)
        else:
            job.reason = result.get("reason")
        if at_limit and job.state == State.FAILED and (mem := self._more_ram(job)):
            job.reason += f"; resubmit with --mem {format_size(mem)}"  # as a VRAM stop says its size
        grown = bool(job.ram_limit and job.ram_limit > job.request.mem)
        if job.state == State.FAILED and grown and not at_limit and result.get("oom_killed") and (mem := self._more_ram(job)):
            job.reason += f"; resubmit with --mem {format_size(mem)}"  # it needed more than it reserved (#155)
        lost_borrowing = job.evicted or (result.get("oom_killed") and result.get("hit_limit") is False and job.borrowed_ram)
        if job.state == State.FAILED and lost_borrowing and not job.retry:  # dibs chose to start it there (#148)
            more = f" with --mem {format_size(mem)}" if job.grew_borrowed and (mem := self._more_ram(job)) else ""  # #182
            job.reason += f"; resubmit it{more}, or submit it with --retry so dibs runs it again by itself"
        self.store.update_job(job)
        self._terminated.discard(job.id)
        self._notify(job)

    def _retry_with(self, job: Job, result: dict) -> tuple[Request, str, str] | None:
        """For a job that failed for a resource reason and may run again: the request to run it with,
        how (after "retrying") and why. None when it isn't such a failure, or it can't fit."""
        if not job.retry:
            return None
        if job.evicted or (result.get("oom_killed") and result.get("hit_limit") is False and job.borrowed_ram):
            # It ran on borrowed RAM, dibs's choice rather than the job's, so it's always queued again,
            # outside the retry limit: it then can't borrow, so this happens once (#114).
            job.no_borrow = True
            reason = job.stop_reason if job.evicted else _ram_kill_reason(job, result)
            if job.grew_borrowed and (mem := self._more_ram(job)):  # it showed it needs more (#182)
                return replace(job.request, mem=mem), f"with --mem {format_size(mem)}, without borrowing", reason
            return job.request, "without borrowing", reason
        if len(job.attempts) >= MAX_RETRIES:
            return None
        if job.stop_reason and job.retry_vram and job.retry_vram > job.request.vram:
            return replace(job.request, vram=job.retry_vram), f"with --vram {self._vram_text(job.retry_vram)}", job.stop_reason
        if result.get("oom_killed") and result.get("hit_limit") is False:
            # The machine ran out, not the job's own limit: run it as it was, within RAM nobody reserved
            # if it had borrowed (#86).
            job.no_borrow = job.no_borrow or job.borrowed_ram
            if job.ram_limit and job.ram_limit > job.request.mem and (mem := self._more_ram(job)):
                return replace(job.request, mem=mem), f"with --mem {format_size(mem)}", _ram_kill_reason(job, result)
            return job.request, "without borrowing" if job.borrowed_ram else "", _ram_kill_reason(job, result)
        if result.get("oom_killed") and (mem := self._more_ram(job)):
            return replace(job.request, mem=mem), f"with --mem {format_size(mem)}", _ram_kill_reason(job, result)
        return None

    def _more_ram(self, job: Job) -> int | None:
        """The RAM to run a job killed at its RAM limit with: 25% more than it reached, rounded up to a
        whole gigabyte (#84); None when the machine hasn't that much."""
        mem = math.ceil(max(job.peak_mem or 0, job.request.mem) * 1.25 / GiB) * GiB
        return mem if mem <= self.capacity.mem else None

    def _vram_text(self, vram: int) -> str:
        return "all" if self.capacity.gpus and vram >= max(g.vram for g in self.capacity.gpus) else format_size(vram)

    def _retry(self, job: Job, reason: str, request: Request) -> None:
        """Queue a job that failed for a resource reason again (#84): same id, same place in line, with
        the failed run on record and a note in its output. It never looks finished, so `dibs wait` and
        `dibs run` keep waiting: the agent needn't work out what happened and resubmit."""
        job.attempts.append(
            {
                "started": job.started,
                "ended": job.ended,
                "exit_code": job.exit_code,
                "reason": reason,
                "request": asdict(job.request),
                "peak_mem": job.peak_mem,
                "peak_vram": job.peak_vram,
                "asleep": job.asleep,
            }
        )
        job.request, job.state, job.reason = request, State.QUEUED, None
        job.started = job.ended = job.exit_code = job.cpu_seconds = None
        job.cpu_ids, job.gpu_ids = [], []
        job.shim_pid = job.shim_ticks = job.unit = None
        job.stop_reason = job.stop_requested = job.retry_vram = None
        job.borrowed_ram = job.evicted = job.grew_borrowed = False
        job.peak_mem = job.peak_shmem = job.peak_shmem_mapped = job.peak_vram = job.ram_limit = None
        job.sleep_at_start, job.asleep = None, 0.0
        with contextlib.suppress(FileNotFoundError):
            (self.paths.job_dir(job.id) / "exit.json").unlink()
        launcher.remove_job_tmp(self.paths, job.id)  # the next try starts with an empty one
        with contextlib.suppress(OSError), open(job.output, "a") as f:
            f.write(f"\ndibs: {reason} (try {len(job.attempts) + 1} of {MAX_RETRIES + 1})\n")
        self.store.update_job(job)
        self._terminated.discard(job.id)
        log.info("job %d %s", job.id, reason)

    def _shim_said(self, job: Job) -> str:
        """The last thing the job's launcher logged, as " (...)", for a job that was lost."""
        try:
            lines = (self.paths.job_dir(job.id) / "shim.log").read_text(errors="replace").splitlines()
        except OSError:
            return ""
        last = next((line.strip() for line in reversed(lines) if line.strip()), "")
        return f" ({last[:200]})" if last else ""

    def _lose(self, job: Job, now: float, reason: str, asleep: float | None = None) -> None:
        launcher.remove_job_tmp(self.paths, job.id)
        job.asleep = self._asleep(job) if asleep is None else asleep
        job.state, job.ended, job.reason = State.FAILED, now, reason
        self.store.update_job(job)
        self._notify(job)

    def _notify(self, job: Job) -> None:
        if job.snapshot and job.snapshot_kept is None:  # its copy goes, unless it wrote into it (#172)
            job.snapshot_kept = snapshot.clear(self.paths.job_dir(job.id) / "snapshot")
            if job.snapshot_kept:
                self.store.update_job(job)
        if job.state == State.FAILED:  # its session is told at its next dibs commands (#176)
            self.store.add_event("failed", {"session": job.session, "job": job.id}, job.ended or self.clock())
        for callback in self._callbacks:
            callback(job)

    def _retention(self, now: float) -> None:
        cutoff = now - self.config.keep_days * 86400
        old = self.store.finished_before(cutoff)
        for job in old:
            shutil.rmtree(self.paths.job_dir(job.id), ignore_errors=True)
        self.store.delete_jobs(j.id for j in old)
        self.store.delete_events_before(cutoff)
        lapsed = [k for k, v in self.store.meta_items("owed:") if now - json.loads(v).get("time", 0) > OWED_FOR]
        lapsed += [k for k, v in self.store.meta_items("seen:") if now - float(v) > TOLD_WINDOW]  # #176
        lapsed += [k for k, v in self.store.meta_items("advised:") if now - float(v) > TOLD_WINDOW]  # #169
        staging = self.paths.state_dir / "snapshots"  # copies a submit never claimed (#172)
        for folder in staging.iterdir() if staging.is_dir() else ():
            with contextlib.suppress(OSError):
                if now - folder.stat().st_mtime > TOLD_WINDOW:
                    shutil.rmtree(folder, ignore_errors=True)
        self.store.delete_meta(lapsed)
        self._last_retention = now

    def _capacity(self) -> dict:
        return {
            "cpus": list(self.capacity.cpus),
            "mem": self.capacity.mem,
            "gpus": [
                {"index": g.index, "uuid": g.uuid, "name": g.name, "vram": g.vram} for g in self.capacity.gpus
            ],
        }

    def _view(self, job: Job, now: float, current: Plan) -> dict:
        view = job_to_dict(job)
        if job.state == State.QUEUED:
            p = current.predictions.get(job.id)
            view.update(
                position=p.position if p else None,
                predicted_start=p.start if p else None,
                predicted_end=p.end if p else None,
                waiting_for=p.reason if p else "",
                lacks=list(p.lacks) if p else [],
                behind=p.behind if p else None,
                after=dict(zip(("id", "over", "estimate"), p.after)) if p and p.after else None,
            )
        elif job.state == State.RUNNING:
            started = (job.started or now) + self._asleep(job)
            view["asleep"] = self._asleep(job)
            view["expected_end"] = current.running_ends.get(job.id, started + job.request.time)
            view["over_estimate"] = max(0.0, now - (started + job.request.time))
            own = self._job_cpu_seconds(job)  # for CPU use on dibs show's peak line (#175)
            if own is not None:
                view["cpu_seconds"] = own
        return view

    def _asleep(self, job: Job) -> float:
        """How long the machine has slept since a running job started (#76); 0 for jobs started
        before dibs kept count."""
        if job.state != State.RUNNING:
            return job.asleep
        return max(0.0, self.slept() - job.sleep_at_start) if job.sleep_at_start is not None else 0.0


_lock_fd: int | None = None  # held for the daemon's whole life


def _setup_logging(paths: Paths) -> None:
    # Size-capped while running, so even an error repeating every second can't fill the disk.
    handler = logging.handlers.RotatingFileHandler(paths.daemon_log, maxBytes=LOG_MAX, backupCount=1)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.handlers[:] = [handler]
    log.setLevel(logging.INFO)


def run_daemon(env: Mapping[str, str] = os.environ) -> int:
    """Run the daemon in this process until it's told to stop. Returns an exit code."""
    global _lock_fd
    # The CLI sets SIGPIPE to its default for `dibs queue | head`. The daemon must outlive a client
    # that hangs up before its answer: then writing it raises BrokenPipeError instead (#82).
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    paths = get_paths(env)
    ensure_dirs(paths)
    fd = os.open(paths.lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return 0  # another daemon is already running here
    _lock_fd = fd
    _setup_logging(paths)
    try:
        config = load_config(paths.config_file)
        backend = open_backend(env)
        capacity = build_capacity(config, detect_cpus(), detect_mem(), backend.list_gpus())
    except (ConfigError, ValueError, OSError) as e:
        log.error("can't start: %s", e)
        print(f"dibs: can't start: {e}", file=sys.stderr)
        return 1
    use_systemd = launcher.systemd_available(env)
    daemon = Daemon(paths, config, capacity, backend, use_systemd=use_systemd, programs=procs.big_programs,
                    set_mem_limit=launcher.set_memory_max if use_systemd else None,
                    panes=lambda: procs.pane_cpu(PANE_SCOPES))  # fmt: skip
    daemon.recover()
    gpus = ", ".join(f"GPU {g.index} {format_size(g.vram)}" for g in capacity.gpus) or "no GPUs"
    log.info(
        "dibs %s started (pid %d): %d CPUs, %s RAM, %s; guard %s",
        build_id(), os.getpid(), len(capacity.cpus), format_size(capacity.mem), gpus, config.guard,
    )
    try:
        asyncio.run(daemon.serve())
    finally:
        daemon.close()
    return 0
