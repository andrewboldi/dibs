"""What a machine has, what a job asks for, and where a job stands in line."""

import math
import shlex
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from dibs.units import GiB, format_size, format_size_against


GPU_IDLE_AFTER = 600.0  # seconds: a job's hold of GPU memory with the GPU idle counts once it has gone on this long (#180)


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
    exclusive: bool = False  # nothing else runs beside it, e.g. for timing (#104)


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
    peak_shmem: int | None = None  # the shared memory in peak_mem: files in /dev/shm or a RAM-backed /tmp, and the like
    peak_shmem_mapped: list[int] | None = None  # how much of peak_shmem its processes had mapped: at least, at most
    cpu_seconds: float | None = None  # CPU time the job used, from its scope
    sleep_at_start: float | None = None  # the machine's time asleep since boot (procs.slept) when the job started
    asleep: float = 0.0  # how long the machine slept while the job ran (set when it ends); run times leave it out
    priority: int = 5  # 1-10; see compute_tag
    plain_tag: float | None = None  # its place at priority 5 with nothing owed; the session's next jobs line up behind it
    own_time: float | None = None  # what it adds to its session's line: its estimate, at any momentum share (#150)
    ram_limit: int | None = None  # the RAM limit it grew to past its reservation, into RAM nobody reserved (#155)
    retry: bool = False  # its agent marked it safe to run again after a resource failure (#84, #116)
    attempts: list[dict] = field(default_factory=list)  # earlier runs that failed for a resource reason
    retry_vram: int | None = None  # the VRAM a stop for going over the reservation advises
    owed_carried: float = 0.0  # what its session owed (or was owed) when it was submitted, in its place
    owed_left: float = 0.0  # what it left its session's next job: payback for a boost, credit for less
    owed_settled: bool = False  # whether the two were settled against its real wait, when it started
    history_mem: int | None = None  # the peak RAM of earlier runs of the same command, at submit (#86)
    # GPU memory held with the GPU idle (#180): stretches of GPU_IDLE_AFTER or more, in seconds; the one going
    # on now, since when; the most it held then; and when its session was last told of it (#186).
    gpu_idle: float = 0.0
    gpu_idle_since: float | None = None
    gpu_idle_held: int | None = None
    gpu_idle_told: float | None = None
    history_vram: int | None = None  # the most GPU memory earlier runs of the same command held, at submit (#183)
    history_longest: float | None = None  # the longest earlier run of its script with its estimate, at submit (#227)
    kept: list[float] | None = None  # [tag, plain tag] a replacement kept from the job it replaced (#184, #226)
    gpu_unasked_told: bool = False  # it asked for no GPU, held a little on one, and its session was told (#246)
    hard_limit: int | None = None  # its scope's RAM limit at launch, a step above its request while RAM was free (#249)
    grew_told: bool = False  # it held over twice its RAM request, and its session was told (#251)
    peak_swap: int | None = None  # the most it held in swap, spilled past its RAM limit or moved there (#257)
    swap_stopped: bool = False  # dibs stopped it for thrashing at its limit, or for the swap it held (#257)
    title: str | None = None  # what it does, in plain words, for the person watching dibs top (#259)
    peak_escaped: int | None = None  # the most programs it started held after leaving its scope, as snap apps do (#265)
    escaped_told: bool = False  # its session was told its --mem doesn't hold those programs (#265)
    peak_igpu: float | None = None  # the most of the integrated GPU it kept busy over 10 s, in percent (#274)
    borrowed_ram: bool = False  # this run started on RAM other jobs reserved but weren't using
    no_borrow: bool = False  # must start within RAM nobody reserved (after the machine ran out once)
    evicted: bool = False  # stopped by dibs to give borrowed RAM back under memory pressure (#114)
    grew_borrowed: bool = False  # this run grew into RAM other jobs reserved but weren't using (#182)
    snapshot: str | None = None  # the repository this job runs a copy of, made at submit (#172)
    snapshot_kept: str | None = None  # where that copy is kept, when the job wrote files into it
    background: dict | None = None  # beside an exclusive job: other programs' CPU use, mean and max cores (#112)


class RequestError(ValueError):
    """A request dibs refuses, with a message that says why."""


DEFAULT_PRIORITY = 5
# How much of its estimate a job counts as in line, by priority, as an exact fraction.
PRIORITY_SHARE = {1: (4, 1), 2: (3, 1), 3: (2, 1), 4: (3, 2), 5: (1, 1),
                  6: (7, 10), 7: (1, 2), 8: (7, 20), 9: (1, 5), 10: (1, 10)}  # fmt: skip
CREDIT_MAX = 3600.0  # seconds: a session's head start from lower-priority jobs is capped at an hour


def compute_tag(
    submitted: float,
    estimate: float,
    session_queued_tags: Iterable[float],
    priority: int = DEFAULT_PRIORITY,
    owed: float = 0.0,
) -> float:
    """A job's place in line (lower goes first). It lines up behind its own session's queued jobs (pass
    their priority-5 places, so a boost isn't passed down the line), plus what its session owes
    (minus its credit), plus its estimate scaled by its priority: at 10 a tenth of it, at 1 four
    times it. Never earlier than its submit time, so no new job goes ahead of one whose place has
    come due."""
    num, den = PRIORITY_SHARE[priority]
    start = max(submitted, max(session_queued_tags, default=-math.inf))
    return max(submitted, start + owed + estimate * num / den)


def folder_places(jobs: Iterable["Job"], place_of: Callable[[str], str], offsets: Mapping[str, float]) -> dict[int, tuple[float, float]]:
    """Each waiting job's place among its session's folders, with and without its priority (#226): a coding
    agent's subagents share their session's id, so its folders take turns at the session's places.
    Each folder's jobs line up as a session's do (compute_tag): in the order they came, behind the
    folder's jobs of at least their priority, and ahead of its lower-priority ones, which move back by
    their counted time. A folder's offset, what its boosts that passed a sibling saved, moves the start
    of every one of its jobs back once. A folder that had nothing waiting counts from its own submit."""
    by_place: dict[str, list[Job]] = {}
    for j in jobs:
        by_place.setdefault(place_of(j.cwd), []).append(j)
    places: dict[int, list[float]] = {}
    for place, own in by_place.items():
        offset = offsets.get(place, 0.0)
        lined: list[Job] = []
        for j in sorted(own, key=lambda j: (j.submitted, j.id)):
            counted = j.own_time if j.own_time is not None else j.request.time
            backlog = [places[r.id][1] for r in lined if r.priority >= j.priority]
            start = max([j.submitted + offset, *backlog])
            for r in lined:
                if r.priority < j.priority and places[r.id][1] > start:
                    places[r.id][0] += counted
                    places[r.id][1] += counted
            places[j.id] = [compute_tag(j.submitted + offset, counted, backlog, j.priority, j.owed_carried),
                            compute_tag(j.submitted + offset, counted, backlog)]  # fmt: skip
            lined.append(j)
    return {i: (tag, plain) for i, (tag, plain) in places.items()}


def vram_to_ask(peak: int, gpu_vram: int | None, early: bool = False) -> str:
    """What --vram to ask for after a job was stopped at this peak, rounded up to half a gigabyte, or
    "all" when that's more than the GPU has: 10% headroom, or, when it was stopped in its first
    minute and likely still growing, the larger of 25% and 1G (#80)."""
    half = GiB // 2
    want = max(peak * 1.25, peak + GiB) if early else peak * 1.1
    ask = math.ceil(want / half) * half
    return "all" if gpu_vram is not None and ask >= gpu_vram else format_size(ask)


def next_owed(estimate: float, priority: int) -> float:
    """What a job leaves its session's next job: the time its priority saved it, to be paid back, or
    the time it gave up, credited (at most CREDIT_MAX). A free priority would have every agent pick
    10; this way a high priority moves your own work around and can't take others' share."""
    num, den = PRIORITY_SHARE[priority]
    return max(-CREDIT_MAX, estimate * (den - num) / den)


def fits_machine(req: Request, cap: Capacity) -> str | None:
    """Why the request can never run on this machine, or None if it can."""
    if req.cpus > len(cap.cpus):
        return f"asks for {req.cpus} CPUs, but dibs manages {len(cap.cpus)}"
    if req.mem > cap.mem:
        return f"asks for {format_size(req.mem)} RAM, but dibs manages {format_size_against(cap.mem, req.mem)}"
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
            f"{format_size_against(cap.max_vram, req.vram)}"
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
    if not isinstance(time, (int, float)) or isinstance(time, bool) or math.isnan(time) or time <= 0:
        raise RequestError("--time must be more than 0")
    if not math.isfinite(time):
        raise RequestError("--time must be a finite duration")  # one infinite job would stall the plan

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

    req = Request(cpus=cpus, mem=mem, gpus=gpus, vram=vram, time=float(time), exclusive=raw.get("exclusive") is True)
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


_SHELLS = ("sh", "bash", "zsh", "dash")
_TEST_RUNNERS = ("pytest", "unittest")  # modules whose arguments pick what runs: their test files are the script (#209)


def script_of(argv: Sequence[str]) -> str | None:
    """The script a command runs: the first script path in it (also inside a shell's -c string), or
    "-m module" for Python. Earlier runs of the same script with other arguments are a guide to
    sizing the next one. None if there's no script, or a -c string doesn't parse."""
    words = list(argv)
    for i, word in enumerate(argv):
        if word == "-c" and i + 1 < len(argv) and any(os.path.basename(w) in _SHELLS for w in argv[:i]):
            try:
                words = [*argv[:i], *shlex.split(argv[i + 1]), *argv[i + 2 :]]
            except ValueError:
                return None
            break
    for i, word in enumerate(words):
        if word == "-m" and i + 1 < len(words):
            module = words[i + 1]
            if module in _TEST_RUNNERS:  # it runs the tests its arguments name, as `pytest file.py` does (#209)
                named = (w.split("::")[0] for w in words[i + 2 :])
                return next((w for w in named if w.endswith(_SCRIPT_SUFFIXES)), f"-m {module}")
            return f"-m {module}"
        if word.endswith(_SCRIPT_SUFFIXES):
            return word
    return None


def script_name(argv: Sequence[str]) -> str | None:
    """What makes two runs in a session's different folders runs of the same script (#229): the script's
    file name (a copy of the code puts it under another path), or "-m module"."""
    script = script_of(argv)
    return script if not script or script.startswith("-m ") else os.path.basename(script)


def default_name(argv: Sequence[str]) -> str:
    """A short label for a job: its script's name if it runs one, else its program's name."""
    if not argv:
        return "job"
    name = next(
        (os.path.basename(a) for a in argv if a.endswith(_SCRIPT_SUFFIXES)),
        os.path.basename(argv[0]),
    )
    return name[:24]
