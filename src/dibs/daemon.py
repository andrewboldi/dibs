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
from collections.abc import Callable, Mapping

from dibs import __version__, build_id, guard, launcher, procs
from dibs.gpu import GpuBackend, open_backend
from dibs.machine import Config, ConfigError, build_capacity, detect_cpus, detect_mem, load_config
from dibs.model import (
    CREDIT_MAX,
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
    script_of,
    vram_to_ask,
)
from dibs.paths import Paths, ensure_dirs, get_paths
from dibs.planner import Plan, Queued, Running, plan
from dibs.store import Store
from dibs.units import GiB, format_size, format_size_against, parse_size

KILL_AFTER = 10.0  # seconds between asking a job to stop (SIGTERM) and forcing it (SIGKILL)
OWED_FOR = 86400.0  # seconds: what a session owes (or is owed) for its priorities lapses after a day
OUTSIDE_WINDOW = 120.0  # seconds: RAM in use outside dibs counts as its peak over this long
JOBS_RAM_WINDOW = 10.0  # seconds: RAM dibs jobs hold counts as its peak over this long, when taken off the in-use total
RETENTION_EVERY = 3600.0
MAX_RETRIES = 2  # runs again after a resource failure, at most (#84)
YOUNG = 600.0  # seconds: a job with no earlier runs counts at its whole RAM reservation this long (#86)
PRESSURE_LIMIT = 10.0  # percent (PSI some avg10): above it, no job starts on borrowed RAM
EVICT_PRESSURE = 40.0  # percent: memory pressure this high for EVICT_AFTER stops a borrower (#114)
EVICT_AFTER = 10.0  # seconds
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
WAIT_RECHECK = 5.0  # a long-polling wait re-checks its jobs at least this often
LOG_MAX = 5 * 1024 * 1024

log = logging.getLogger("dibs")


def _rejected(message: str) -> dict:
    return {"ok": False, "code": "rejected", "error": message}


def _ram_kill_reason(job: Job, result: dict) -> str:
    """Why a job was killed for RAM: at its own limit, or because the whole machine ran out (#86)."""
    if result.get("hit_limit") is False:
        borrowed = ", while it used RAM other jobs had reserved" if job.borrowed_ram else ""
        return "killed when the machine ran out of RAM" + borrowed
    return f"killed: went over its {format_size(job.request.mem)} RAM reservation" + _shared_part(job)


def _holding(job: Job, mem: int | None = None) -> Request:
    """A running job's request as the planner should count it: VRAM it borrowed counts as held (#88),
    and its RAM as `mem` when RAM is planned by use (#86)."""
    request = job.request
    if job.peak_vram and request.gpus and job.peak_vram > request.vram:
        request = replace(request, vram=job.peak_vram)
    if mem is not None and mem != request.mem:
        request = replace(request, mem=mem)
    return request


def _shared_part(job: Job) -> str:
    """For a job killed at its RAM limit: how much of its peak was shared memory, when that was at
    least a quarter of its reservation. Files a job writes to a folder kept in RAM count as its RAM,
    which agents don't expect (#74)."""
    if job.peak_shmem is None or job.peak_shmem < job.request.mem / 4:
        return ""
    return f" ({format_size(job.peak_shmem)} of it was files in /dev/shm or a RAM-backed /tmp, or other shared memory)"


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
        self.cpu_busy = cpu_busy
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

        replaced = None
        if req.get("replace") is not None:
            replaced = self.store.get_job(int(req["replace"]))
            if replaced is None:
                return _rejected(f"no job {req['replace']} to replace")

        now = self.clock()
        if replaced is not None:
            # In the same step as the new job, so nothing can start in the gap (#93), and before the
            # backlog is read, so the replacement doesn't line up behind the job it replaces.
            self._cancel_job(replaced, now)
        session = req.get("session") or "unknown"
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
            tag=compute_tag(now, counted, backlog, priority, carried),
            priority=priority,
            plain_tag=compute_tag(now, counted, backlog),
            retry=req.get("retry") is True,  # the agent's choice: only it knows the job is safe to repeat (#116)
            owed_carried=carried,
            owed_left=left,
            history_mem=self._history_mem(cwd, list(argv)),
            state=State.QUEUED,
            output=output or "",
        )
        self.store.add_job(job)
        self.store.set_meta(f"owed:{session}", json.dumps({"owed": left, "time": now}))
        for j in passed:  # zero-sum within the session: the jobs it went ahead of move back by its estimate
            j.tag += counted
            if j.plain_tag is not None:
                j.plain_tag += counted
            self.store.update_job(j)
        self.paths.job_dir(job.id).mkdir(mode=0o700, parents=True, exist_ok=True)
        if not job.output:
            job.output = str(self.paths.job_dir(job.id) / "output.log")
            self.store.update_job(job)
        if replaced is not None and replaced.state == State.CANCELLED:
            replaced.reason = f"replaced by job {job.id}"
            self.store.update_job(replaced)
        self.schedule()
        response = self.show(job.id)
        if replaced is not None:
            response["replaced"] = replaced.id
        # What the same command used before, so the agent can size its next request from it.
        earlier = [j for j in self.store.earlier_runs(job.cwd, job.argv, limit=6) if j.started is not None]
        match = "command"
        script = script_of(job.argv)
        if not earlier and script:  # none of this exact command: the same script with other arguments
            earlier = [j for j in self.store.earlier_runs_of_script(job.cwd, script, limit=6) if j.started is not None]
            match = "script"
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

    def _cancel_job(self, job: Job, now: float) -> bool:
        """Cancel one job without scheduling: a queued one ends now, a running one is asked to stop.
        False if it had already ended."""
        if job.state == State.QUEUED:
            job.state, job.ended, job.reason = State.CANCELLED, now, "cancelled"
            self.store.update_job(job)
            # It never ran: it hands back what it carried and takes back what it left (#90).
            self._add_owed(job.session, job.owed_carried - job.owed_left, now)
            self._notify(job)
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
            [Running(j.id, _holding(j, self._planned_mem(j, now) if borrowing else None), tuple(j.cpu_ids),
                     tuple(j.gpu_ids), (j.started or now) + self._asleep(j))
             for j in running],
            [Queued(j.id, j.request, j.tag, j.submitted) for j in queued],
            now,
            outside_mem=outside_mem,
        )
        return current, running, queued

    def _history_mem(self, cwd: str, argv: list[str]) -> int | None:
        """The peak RAM of earlier runs of the same command here (or the same script), if any."""
        runs = [j for j in self.store.earlier_runs(cwd, argv, limit=6) if j.peak_mem]
        script = script_of(argv)
        if not runs and script:
            runs = [j for j in self.store.earlier_runs_of_script(cwd, script, limit=6) if j.peak_mem]
        return max((j.peak_mem for j in runs), default=None)

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
        # RAM nobody has reserved: a job that needs more starts on RAM other jobs reserved but aren't
        # using, as a borrower (#86), which a job retried after the machine ran out may not do.
        unreserved = self.capacity.mem - self._outside_mem(now, running) - sum(j.request.mem for j in running)
        for start in current.starts:
            job = by_id[start.job_id]
            borrowed = job.request.mem > unreserved
            if borrowed and (job.no_borrow or not job.retry):
                continue  # dibs may have to stop a borrower, so only a job safe to run again may borrow (#116)
            unreserved -= job.request.mem
            job.borrowed_ram = borrowed
            job.cpu_ids, job.gpu_ids = list(start.cpu_ids), list(start.gpu_ids)
            job.started, job.state = now, State.RUNNING
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

    def tick(self) -> None:
        """Called about once a second: notice finished jobs, push stops along, start what can start."""
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
        self._give_back_borrowed(now)
        self._watch_beside(now)
        self._watch_gpus(now)
        self.schedule()
        for watch in (self._watch_outside_cpu, self._watch_load):  # only shown by dibs top, after scheduling
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

    def _memory_in_use(self, job: Job) -> tuple[int, int] | None:
        """What a running job holds (anon + shmem) and the shmem part of it, read from its own scope;
        None if dibs can't see it."""
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
        for job in self.store.jobs([State.RUNNING]):
            held = self._memory_in_use(job)
            if held is not None and (job.peak_mem is None or held[0] > job.peak_mem):
                job.peak_mem, job.peak_shmem = held
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

    def _give_back_borrowed(self, now: float) -> None:
        """While jobs run on borrowed RAM and memory pressure stays high, stop the most recently started
        borrower and run it again within RAM nobody reserved (#114). The kernel's OOM killer acts only
        after minutes of thrashing; one more after each further stretch of pressure."""
        borrowers = [
            j for j in self.store.jobs([State.RUNNING])
            if j.borrowed_ram and j.stop_requested is None and j.cancel_requested is None
        ]  # fmt: skip
        if not borrowers or self.pressure() < EVICT_PRESSURE:
            self._pressured_since = None
            return
        if self._pressured_since is None:
            self._pressured_since = now
            return
        if now - self._pressured_since < EVICT_AFTER:
            return
        job = max(borrowers, key=lambda j: (j.started or 0, j.id))
        job.evicted = True
        job.stop_reason = "stopped to give back borrowed RAM: the machine was short of memory"
        job.stop_requested = now
        self.store.update_job(job)
        log.info("job %d %s", job.id, job.stop_reason)
        self._push_stop(job, now)
        self._pressured_since = now

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
        if not seen:
            self.outside = []
            self._outside_vram = {}
            return
        managed = {g.index for g in self.capacity.gpus}
        seen = [p for p in seen if p.gpu_index in managed]
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
        for v in verdicts:
            if v.kind == "job" and v.used is not None:
                per_gpu = usage.setdefault(v.job_id, {})
                per_gpu[v.gpu_index] = per_gpu.get(v.gpu_index, 0) + v.used
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

        for job_id, per_gpu in usage.items():
            job = running[job_id]
            peak = max(per_gpu.values())
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
        job.cpu_seconds = result.get("cpu_seconds")
        if job.peak_mem is None:  # never sampled while it ran: the launcher's figure is all there is
            job.peak_mem = result.get("peak_mem")
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
            return job.request, "without borrowing", reason
        if len(job.attempts) >= MAX_RETRIES:
            return None
        if job.stop_reason and job.retry_vram and job.retry_vram > job.request.vram:
            return replace(job.request, vram=job.retry_vram), f"with --vram {self._vram_text(job.retry_vram)}", job.stop_reason
        if result.get("oom_killed") and result.get("hit_limit") is False:
            # The machine ran out, not the job's own limit: run it as it was, within RAM nobody reserved
            # if it had borrowed (#86).
            job.no_borrow = job.no_borrow or job.borrowed_ram
            return job.request, "without borrowing" if job.borrowed_ram else "", _ram_kill_reason(job, result)
        if result.get("oom_killed"):
            mem = math.ceil(max(job.peak_mem or 0, job.request.mem) * 1.25 / GiB) * GiB
            if mem <= self.capacity.mem:
                return replace(job.request, mem=mem), f"with --mem {format_size(mem)}", _ram_kill_reason(job, result)
        return None

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
        job.borrowed_ram = job.evicted = False
        job.peak_mem = job.peak_shmem = job.peak_vram = None
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
            )
        elif job.state == State.RUNNING:
            started = (job.started or now) + self._asleep(job)
            view["asleep"] = self._asleep(job)
            view["expected_end"] = current.running_ends.get(job.id, started + job.request.time)
            view["over_estimate"] = max(0.0, now - (started + job.request.time))
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
    daemon = Daemon(paths, config, capacity, backend, use_systemd=launcher.systemd_available(env))
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
