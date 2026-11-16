"""The scheduler: the one process that owns the queue, starts jobs and watches them.

Everything it knows is in the store, written before it acts, so a crash or restart loses nothing
and never touches running jobs: they're adopted again on start (recover).
"""

import asyncio
import fcntl
import json
import logging
import logging.handlers
import math
import os
import shutil
import signal
import sys
import time
from collections import deque
from collections.abc import Callable, Mapping

from dibs import __version__, build_id, guard, launcher, procs
from dibs.gpu import GpuBackend, open_backend
from dibs.machine import Config, ConfigError, build_capacity, detect_cpus, detect_mem, load_config
from dibs.model import (
    TERMINAL,
    Capacity,
    Job,
    RequestError,
    State,
    compute_tag,
    default_name,
    job_to_dict,
    resolve_request,
)
from dibs.paths import Paths, ensure_dirs, get_paths
from dibs.planner import Plan, Queued, Running, plan
from dibs.store import Store
from dibs.units import format_size

KILL_AFTER = 10.0  # seconds between asking a job to stop (SIGTERM) and forcing it (SIGKILL)
OUTSIDE_WINDOW = 120.0  # seconds: RAM in use outside dibs counts as its peak over this long
RETENTION_EVERY = 3600.0
RECENT = 86400.0  # how far back `queue --all` looks
STOPS_SHOWN = 3600.0  # how far back guard stops are shown
WAIT_RECHECK = 5.0  # a long-polling wait re-checks its jobs at least this often
LOG_MAX = 5 * 1024 * 1024

log = logging.getLogger("dibs")


def _rejected(message: str) -> dict:
    return {"ok": False, "code": "rejected", "error": message}


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
    ):
        self.paths = paths
        self.config = config
        self.capacity = capacity
        self._meminfo = meminfo
        self._outside: deque[tuple[float, int, int]] = deque()  # (time, RAM in use outside dibs, total)
        self.backend = backend
        self.use_systemd = use_systemd
        self.clock = clock
        ensure_dirs(paths)
        self.store = Store(paths.db)
        self.closed = False
        self.outside: list[dict] = []  # GPU processes outside dibs that the guard left alone
        self._callbacks: list[Callable[[Job], None]] = []
        self._terminated: set[int] = set()  # jobs that have been sent SIGTERM
        self._last_retention = -math.inf
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

        now = self.clock()
        session = req.get("session") or "unknown"
        backlog = [j.tag for j in self.store.jobs([State.QUEUED]) if j.session == session]
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
            tag=compute_tag(now, request.time, backlog),
            state=State.QUEUED,
            output=output or "",
        )
        self.store.add_job(job)
        self.paths.job_dir(job.id).mkdir(mode=0o700, parents=True, exist_ok=True)
        if not job.output:
            job.output = str(self.paths.job_dir(job.id) / "output.log")
            self.store.update_job(job)
        self.schedule()
        response = self.show(job.id)
        # What the same command used before, so the agent can size its next request from it.
        earlier = [j for j in self.store.earlier_runs(job.cwd, job.argv, limit=6) if j.started is not None]
        response["earlier"] = [job_to_dict(j) for j in earlier[:3]]
        return response

    def cancel(self, ids: list[int]) -> dict:
        now = self.clock()
        cancelled, errors = [], []
        for job_id in ids:
            job = self.store.get_job(job_id)
            if job is None:
                errors.append(f"no job {job_id}")
            elif job.state == State.QUEUED:
                job.state, job.ended, job.reason = State.CANCELLED, now, "cancelled"
                self.store.update_job(job)
                self._notify(job)
                cancelled.append(job_id)
            elif job.state == State.RUNNING:
                if job.cancel_requested is None:
                    job.cancel_requested = now
                    self.store.update_job(job)
                self._push_stop(job, now)
                cancelled.append(job_id)
            else:
                errors.append(f"job {job_id} already {job.state}")
        self.schedule()
        return {"ok": True, "cancelled": cancelled, "errors": errors}

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
        current = plan(
            self.capacity,
            [Running(j.id, j.request, tuple(j.cpu_ids), tuple(j.gpu_ids), j.started or now) for j in running],
            [Queued(j.id, j.request, j.tag, j.submitted) for j in queued],
            now,
            outside_mem=self._outside_mem(now, running),
        )
        return current, running, queued

    def _outside_mem(self, now: float, running: list[Job]) -> int:
        """RAM in use outside dibs jobs that eats into what dibs hands out, as its peak over the last
        OUTSIDE_WINDOW seconds, so a job doesn't start into RAM a busy program let go of for a moment.
        A job whose memory dibs can't see counts as holding its whole reservation."""
        if not self._outside or now - self._outside[-1][0] >= 1.0:
            info = self._meminfo()
            if info is not None:
                total, used = info
                jobs = 0
                for job in running:
                    held = self._memory_in_use(job)
                    jobs += held if held is not None else job.request.mem
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
        current, _, queued = self._plan(now)
        by_id = {j.id: j for j in queued}
        for start in current.starts:
            job = by_id[start.job_id]
            job.cpu_ids, job.gpu_ids = list(start.cpu_ids), list(start.gpu_ids)
            job.started, job.state = now, State.RUNNING
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
        self._watch_gpus(now)
        self.schedule()
        if now - self._last_retention >= RETENTION_EVERY:
            self._retention(now)

    def recover(self) -> None:
        """After a (re)start: finish jobs that ended meanwhile, adopt the ones still running."""
        now = self.clock()
        for job in self.store.jobs([State.RUNNING]):
            result = launcher.read_exit(self.paths, job.id)
            if result is not None:
                self._finish(job, result, now)
            elif job.shim_pid is None and (found := self._find_launcher(job)):
                # dibs died between saving the job as running and recording its launcher.
                job.shim_pid, job.shim_ticks, job.unit = found
                self.store.update_job(job)
            elif not (job.shim_pid and procs.alive(job.shim_pid, job.shim_ticks)):
                self._lose(job, now, "lost: its process ended while dibs was not running")

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

    def _memory_in_use(self, job: Job) -> int | None:
        """What a running job holds (anon + shmem), read from its own scope; None if dibs can't see it."""
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
            used = self._memory_in_use(job)
            if used is not None and (job.peak_mem is None or used > job.peak_mem):
                job.peak_mem = used
                self.store.update_job(job)

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
        for job_id, per_gpu in usage.items():
            job = running[job_id]
            peak = max(per_gpu.values())
            changed = job.peak_vram is None or peak > job.peak_vram
            if changed:
                job.peak_vram = peak
            over = [(g, used) for g, used in sorted(per_gpu.items())
                    if used > (job.request.vram if g in job.gpu_ids else 0)]  # fmt: skip
            if over and job.stop_reason is None:
                gpu, used = over[0]
                reserved = job.request.vram if gpu in job.gpu_ids else 0
                job.stop_reason = (
                    f"stopped: used {format_size(used)} VRAM on GPU {gpu}, reserved {format_size(reserved)}"
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
        job.exit_code = result.get("exit_code")
        job.ended = result.get("ended") or now
        job.cpu_seconds = result.get("cpu_seconds")
        if job.peak_mem is None:  # never sampled while it ran: the launcher's figure is all there is
            job.peak_mem = result.get("peak_mem")
        if job.cancel_requested is not None:
            job.state = State.CANCELLED
        elif job.exit_code == 0 and job.stop_reason is None:
            job.state = State.DONE
        else:
            job.state = State.FAILED
        if job.stop_reason:
            job.reason = job.stop_reason
        elif job.cancel_requested is not None:
            job.reason = "cancelled"
        elif result.get("oom_killed"):
            job.reason = f"killed: went over its {format_size(job.request.mem)} RAM reservation"
        else:
            job.reason = result.get("reason")
        self.store.update_job(job)
        self._terminated.discard(job.id)
        self._notify(job)

    def _shim_said(self, job: Job) -> str:
        """The last thing the job's launcher logged, as " (...)", for a job that was lost."""
        try:
            lines = (self.paths.job_dir(job.id) / "shim.log").read_text(errors="replace").splitlines()
        except OSError:
            return ""
        last = next((line.strip() for line in reversed(lines) if line.strip()), "")
        return f" ({last[:200]})" if last else ""

    def _lose(self, job: Job, now: float, reason: str) -> None:
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
            started = job.started or now
            view["expected_end"] = current.running_ends.get(job.id, started + job.request.time)
            view["over_estimate"] = max(0.0, now - (started + job.request.time))
        return view


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
