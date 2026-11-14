"""The scheduler: the one process that owns the queue, starts jobs and watches them.

Everything it knows is in the store, written before it acts, so a crash or restart loses nothing
and never touches running jobs: they're adopted again on start (recover).
"""

import math
import os
import shutil
import signal
import time
from collections.abc import Callable

from dibs import __version__, launcher, procs
from dibs.gpu import GpuBackend
from dibs.machine import Config
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
from dibs.paths import Paths, ensure_dirs
from dibs.planner import Plan, Queued, Running, plan
from dibs.store import Store
from dibs.units import format_size

KILL_AFTER = 10.0  # seconds between asking a job to stop (SIGTERM) and forcing it (SIGKILL)
RETENTION_EVERY = 3600.0
RECENT = 86400.0  # how far back `queue --all` looks
STOPS_SHOWN = 3600.0  # how far back guard stops are shown


def _rejected(message: str) -> dict:
    return {"ok": False, "code": "rejected", "error": message}


def _command_exists(argv0: str, env: dict, cwd: str) -> bool:
    if "/" in argv0:
        path = argv0 if os.path.isabs(argv0) else os.path.join(cwd, argv0)
        return os.path.isfile(path) and os.access(path, os.X_OK)
    return shutil.which(argv0, path=env.get("PATH", os.defpath)) is not None


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
    ):
        self.paths = paths
        self.config = config
        self.capacity = capacity
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
        return self.show(job.id)

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
            recent = [j for j in self.store.jobs(TERMINAL) if (j.ended or 0) >= now - RECENT]
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

    # Scheduling ---------------------------------------------------------------------------------

    def _plan(self, now: float) -> tuple[Plan, list[Job], list[Job]]:
        running = self.store.jobs([State.RUNNING])
        queued = self.store.jobs([State.QUEUED])
        current = plan(
            self.capacity,
            [Running(j.id, j.request, tuple(j.cpu_ids), tuple(j.gpu_ids), j.started or now) for j in running],
            [Queued(j.id, j.request, j.tag, j.submitted) for j in queued],
            now,
        )
        return current, running, queued

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
                    job, self.paths, use_systemd=self.use_systemd
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
                    self._lose(job, now, "lost: its launcher exited without reporting")
                    continue
            if result is not None:
                self._finish(job, result, now)
            else:
                self._push_stop(job, now)
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
            elif not (job.shim_pid and procs.alive(job.shim_pid, job.shim_ticks)):
                self._lose(job, now, "lost: its process ended while dibs was not running")

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
