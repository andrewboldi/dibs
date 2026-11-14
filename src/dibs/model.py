"""What a machine has, what a job asks for, and where a job stands in line."""

import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from dibs.units import format_size


@dataclass(frozen=True)
class Gpu:
    index: int
    uuid: str
    name: str
    vram: int  # bytes dibs may hand out on this GPU


@dataclass(frozen=True)
class Capacity:
    cpus: tuple[int, ...]  # CPU ids dibs may hand out
    mem: int  # bytes
    gpus: tuple[Gpu, ...]

    @property
    def max_vram(self) -> int:
        return max((g.vram for g in self.gpus), default=0)


@dataclass(frozen=True)
class Request:
    cpus: int
    mem: int  # bytes
    gpus: int  # how many GPUs
    vram: int  # bytes on each GPU it uses
    time: float  # estimated run time, seconds


class State(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL = frozenset({State.DONE, State.FAILED, State.CANCELLED})


@dataclass
class Job:
    id: int
    name: str
    argv: list[str]
    cwd: str
    env: dict[str, str]
    request: Request
    session: str
    session_label: str
    submitted: float
    tag: float
    state: State
    output: str
    started: float | None = None
    ended: float | None = None
    exit_code: int | None = None
    reason: str | None = None
    cpu_ids: list[int] = field(default_factory=list)
    gpu_ids: list[int] = field(default_factory=list)
    shim_pid: int | None = None
    shim_ticks: int | None = None
    unit: str | None = None
    cancel_requested: float | None = None
    stop_reason: str | None = None
    stop_requested: float | None = None
    peak_vram: int | None = None
    peak_mem: int | None = None


class RequestError(ValueError):
    """A request dibs refuses, with a message that says why."""


def compute_tag(submitted: float, estimate: float, session_queued_tags: Iterable[float]) -> float:
    """A job's place in line (lower goes first): it lines up behind its own session's queued jobs."""
    return max(submitted, max(session_queued_tags, default=-math.inf)) + estimate


def fits_machine(req: Request, cap: Capacity) -> str | None:
    """Why the request can never run on this machine, or None if it can."""
    if req.cpus > len(cap.cpus):
        return f"asks for {req.cpus} CPUs, but dibs manages {len(cap.cpus)}"
    if req.mem > cap.mem:
        return f"asks for {format_size(req.mem)} RAM, but dibs manages {format_size(cap.mem)}"
    if req.gpus == 0:
        return None
    if not cap.gpus:
        return "this machine has no GPUs dibs can hand out"
    if req.gpus > len(cap.gpus):
        return f"asks for {req.gpus} GPUs, but dibs manages {len(cap.gpus)}"
    big_enough = sum(1 for g in cap.gpus if g.vram >= req.vram)
    if big_enough >= req.gpus:
        return None
    if req.gpus == 1:
        return (
            f"asks for {format_size(req.vram)} VRAM, but the largest GPU dibs manages has "
            f"{format_size(cap.max_vram)}"
        )
    return (
        f"asks for {req.gpus} GPUs with {format_size(req.vram)} VRAM each, "
        f"but only {big_enough} {'has' if big_enough == 1 else 'have'} that much"
    )


def _whole(value: Any, flag: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestError(f"{flag} must be a whole number")
    return value


def resolve_request(raw: Mapping[str, Any], cap: Capacity) -> Request:
    """Turn what was typed (numbers or "all") into a concrete request, or explain the refusal."""
    time = raw.get("time")
    if not isinstance(time, (int, float)) or isinstance(time, bool) or time <= 0:
        raise RequestError("--time must be more than 0")

    cpus = len(cap.cpus) if raw.get("cpus") == "all" else _whole(raw.get("cpus"), "--cpus")
    if cpus < 1:
        raise RequestError("--cpus must be at least 1")

    mem = cap.mem if raw.get("mem") == "all" else _whole(raw.get("mem"), "--mem")
    if mem <= 0:
        raise RequestError("--mem must be more than 0")

    vram_raw, gpus_raw = raw.get("vram"), raw.get("gpus")
    if vram_raw == "all" and not cap.gpus:
        raise RequestError("this machine has no GPUs dibs can hand out")
    vram = cap.max_vram if vram_raw == "all" else _whole(vram_raw or 0, "--vram")
    gpus = None if gpus_raw is None else _whole(gpus_raw, "--gpus")
    if vram < 0:
        raise RequestError("--vram can't be negative")
    if gpus is not None and gpus < 0:
        raise RequestError("--gpus can't be negative")
    if gpus and not vram:
        raise RequestError("--gpus needs --vram")
    if vram and gpus == 0:
        raise RequestError("--vram needs at least one GPU")
    if not vram:
        gpus = 0
    elif gpus is None:
        gpus = 1

    req = Request(cpus=cpus, mem=mem, gpus=gpus, vram=vram, time=float(time))
    problem = fits_machine(req, cap)
    if problem:
        raise RequestError(problem)
    return req


def job_to_dict(job: Job, *, include_env: bool = False) -> dict:
    d = asdict(job)
    d["state"] = str(job.state)
    if not include_env:
        del d["env"]
    return d


def job_from_dict(d: Mapping[str, Any]) -> Job:
    fields = dict(d)
    fields["request"] = Request(**fields["request"])
    fields["state"] = State(fields["state"])
    fields.setdefault("env", {})
    return Job(**fields)


_SCRIPT_SUFFIXES = (".py", ".sh", ".js", ".ts", ".rb", ".pl")


def default_name(argv: Sequence[str]) -> str:
    """A short label for a job: its script's name if it runs one, else its program's name."""
    if not argv:
        return "job"
    name = next(
        (os.path.basename(a) for a in argv if a.endswith(_SCRIPT_SUFFIXES)),
        os.path.basename(argv[0]),
    )
    return name[:24]
