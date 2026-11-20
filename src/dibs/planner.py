"""Who goes next.

Queued jobs are taken in order of their tags (see model.compute_tag). Each one is placed at the
earliest time it fits for its whole estimated run, around the running jobs and the places already
given to jobs ahead of it ("conservative backfill"). Jobs placed at "now" start now. So a job can
only start early in a gap if that delays no one ahead of it, and every queued job gets a predicted
start time.

plan() is a pure function: same inputs, same plan; no clock, no I/O.
"""

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass

from dibs.model import Capacity, Request, fits_machine
from dibs.units import format_duration, format_size, format_size_against

OVERRUN_MIN = 60.0
OVERRUN_FRACTION = 0.10
OVERRUN_LONG = 3.0  # past this many times its estimate, a job's overrun so far is the better guess (#185)


@dataclass(frozen=True)
class Running:
    job_id: int
    request: Request
    cpu_ids: tuple[int, ...]
    gpu_ids: tuple[int, ...]
    started: float


@dataclass(frozen=True)
class Queued:
    job_id: int
    request: Request
    tag: float
    submitted: float
    asked_vram: int | None = None  # what it asks for, when it's planned at more: its earlier runs held more (#183)


@dataclass(frozen=True)
class Start:
    job_id: int
    cpu_ids: tuple[int, ...]
    gpu_ids: tuple[int, ...]


@dataclass(frozen=True)
class Prediction:
    job_id: int
    position: int  # 1 = next in line
    start: float | None  # None: it can't fit this machine
    end: float | None
    gpu_ids: tuple[int, ...]
    reason: str  # why it isn't starting now; "" if it is
    lacks: tuple[str, ...] = ()  # what it waits for, in a word each: CPUs, RAM, VRAM, GPUs (#143)
    behind: int | None = None  # the job ahead in line it would delay by starting now (#143)
    # (job, how far past its estimate, its estimate): a running job past its estimate whose assumed end
    # this start rests on, which moves with every pass (#181)
    after: tuple[int, float, float] | None = None


@dataclass(frozen=True)
class Plan:
    starts: list[Start]
    predictions: dict[int, Prediction]
    running_ends: dict[int, float]


def expected_end(job: Running, now: float) -> float:
    """When a running job should end. Past its estimate, it's assumed to need a little longer; once it
    has run three times its estimate, as long again as it has run past it (#185): "a little longer"
    is wrong pass after pass then, and holds back every job that would still be running at that end."""
    end = job.started + job.request.time
    if end > now:
        return end
    longer = max(OVERRUN_MIN, OVERRUN_FRACTION * job.request.time)
    if now - job.started >= OVERRUN_LONG * job.request.time:
        longer = max(longer, now - end)
    return now + longer


class _Profile:
    """Free resources over time, as a step function: segment k covers [times[k], times[k+1])."""

    def __init__(self, now: float, cpus: int, mem: int, vram: list[int], jobs: list[int] | None = None):
        self.times = [now]
        self.cpus = [cpus]
        self.mem = [mem]
        self.vram = [list(vram)]
        self.jobs = [list(jobs) if jobs is not None else [0] * len(vram)]  # jobs on each GPU (#146)

    def _boundary(self, t: float) -> int:
        """Make t the start of a segment (t >= times[0]) and return that segment's index."""
        k = bisect.bisect_right(self.times, t) - 1
        if self.times[k] == t:
            return k
        self.times.insert(k + 1, t)
        self.cpus.insert(k + 1, self.cpus[k])
        self.mem.insert(k + 1, self.mem[k])
        self.vram.insert(k + 1, list(self.vram[k]))
        self.jobs.insert(k + 1, list(self.jobs[k]))
        return k + 1

    def block(self, start: float, end: float) -> None:
        """Nothing else fits over [start, end): an exclusive job has the machine (#104)."""
        self.add(start, end, -_ALL, 0, {pos: -_ALL for pos in range(len(self.vram[0]))})

    def add(self, start: float, end: float, cpus: int, mem: int, vram: dict[int, int], jobs: int = 0) -> None:
        """Add (or, with negative amounts, take) resources over [start, end); jobs: how many jobs that
        puts on (or takes off) each GPU in vram."""
        first = self._boundary(start)
        last = len(self.times) if end == math.inf else self._boundary(end)
        for k in range(first, last):
            self.cpus[k] += cpus
            self.mem[k] += mem
            for pos, amount in vram.items():
                self.vram[k][pos] += amount
                self.jobs[k][pos] += jobs

    def earliest(self, req: Request, whole: tuple[int, list[int]] | None = None) -> tuple[float, tuple[int, ...]]:
        """The earliest start at which req fits for its whole run, and the GPU positions to use. With
        whole (every CPU and each GPU's VRAM), only when all of them are free: no other job runs."""
        n, k = len(self.times), 0
        while k < n:
            start = self.times[k]
            end = start + req.time
            low_vram = list(self.vram[k])
            most_jobs = list(self.jobs[k])
            j, short = k, False
            while j < n and self.times[j] < end:
                cpus_needed = whole[0] if whole else req.cpus
                if self.cpus[j] < cpus_needed or self.mem[j] < req.mem or (
                    whole and any(free < total for free, total in zip(self.vram[j], whole[1]))
                ):
                    short = True
                    break
                low_vram = [min(a, b) for a, b in zip(low_vram, self.vram[j])]
                most_jobs = [max(a, b) for a, b in zip(most_jobs, self.jobs[j])]
                j += 1
            if short:
                # Every start up to segment j overlaps it, so none of them fit either.
                k = j + 1
                continue
            if req.gpus == 0:
                return start, ()
            # The GPUs with the fewest jobs over its run, since jobs on one GPU share its compute (#146);
            # then the least room that fits, keeping a big GPU for big jobs.
            eligible = sorted((most_jobs[pos], free, pos) for pos, free in enumerate(low_vram) if free >= req.vram)
            if len(eligible) >= req.gpus:
                return start, tuple(sorted(pos for _, _, pos in eligible[: req.gpus]))
            k += 1
        raise AssertionError("a request that fits the machine must fit once everything ends")


_ALL = 1 << 62  # more than any machine has: what an exclusive job takes from everyone else


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def plan(
    capacity: Capacity, running: Sequence[Running], queued: Sequence[Queued], now: float, outside_mem: int = 0
) -> Plan:
    """outside_mem is RAM in use by programs outside dibs: it isn't handed out, and a job that can't fit
    beside it waits, without a predicted start, until those programs free some."""
    gpu_pos = {g.index: pos for pos, g in enumerate(capacity.gpus)}
    cpu_set = set(capacity.cpus)

    busy_cpus = {c for r in running for c in r.cpu_ids if c in cpu_set}
    free_vram = [g.vram for g in capacity.gpus]
    on_gpu = [0] * len(capacity.gpus)
    usable_mem = capacity.mem - outside_mem
    free_mem = usable_mem
    for r in running:
        free_mem -= r.request.mem
        for g in r.gpu_ids:
            if g in gpu_pos:
                free_vram[gpu_pos[g]] -= r.request.vram
                on_gpu[gpu_pos[g]] += 1
    profile = _Profile(now, len(cpu_set) - len(busy_cpus), free_mem, free_vram, on_gpu)

    running_ends = {}
    blocker = None  # a running job that has the machine to itself
    for r in running:
        end = expected_end(r, now)
        running_ends[r.job_id] = end
        released = {gpu_pos[g]: r.request.vram for g in r.gpu_ids if g in gpu_pos}
        profile.add(end, math.inf, sum(1 for c in r.cpu_ids if c in cpu_set), r.request.mem, released, jobs=-1)
        if r.request.exclusive:
            profile.block(now, end)
            blocker = r.job_id
    whole = (len(cpu_set), [g.vram for g in capacity.gpus])

    order = sorted(queued, key=lambda q: (q.tag, q.submitted, q.job_id))
    placed: dict[int, tuple[float, float, tuple[int, ...]]] = {}
    starts: list[Start] = []
    for q in order:
        if fits_machine(q.request, capacity) or q.request.mem > usable_mem:
            continue
        start, positions = profile.earliest(q.request, whole if q.request.exclusive else None)
        end = start + q.request.time
        if q.request.exclusive:
            profile.block(start, end)
        else:
            profile.add(start, end, -q.request.cpus, -q.request.mem, {p: -q.request.vram for p in positions}, jobs=1)
        gpu_ids = tuple(capacity.gpus[p].index for p in positions)
        placed[q.job_id] = (start, end, gpu_ids)
        if start == now:
            cpu_ids = tuple(c for c in capacity.cpus if c not in busy_cpus)[: q.request.cpus]
            busy_cpus.update(cpu_ids)
            starts.append(Start(q.job_id, cpu_ids, gpu_ids))

    # What is free right now, once the jobs starting now have taken their share.
    now_cpus, now_mem, now_vram = profile.cpus[0], profile.mem[0], profile.vram[0]
    # Starts that rest on a running job's assumed end past its estimate (#181).
    overrunning = {running_ends[r.job_id]: r for r in running if r.started + r.request.time <= now}
    rests = {job_id: overrunning[start] for job_id, (start, _, _) in placed.items() if start > now and start in overrunning}
    predictions: dict[int, Prediction] = {}
    for position, q in enumerate(order, 1):
        problem = fits_machine(q.request, capacity)
        if problem:
            predictions[q.job_id] = Prediction(
                q.job_id, position, None, None, (), f"can't fit this machine: {problem}"
            )
            continue
        if q.job_id not in placed:  # waits for programs outside dibs to free RAM
            reason, lacks, behind = _reason(q, order, placed, now, now_cpus, now_mem, now_vram, outside_mem, rests)
            predictions[q.job_id] = Prediction(q.job_id, position, None, None, (), reason, lacks, behind)
            continue
        start, end, gpu_ids = placed[q.job_id]
        reason, lacks, behind = "", (), None
        if start > now and q.request.exclusive:
            reason = f"needs the machine to itself ({_plural(len(running), 'job')} running)"
        elif start > now and blocker is not None:
            reason = f"waits for #{blocker}, which has the machine to itself"
        elif start > now:
            reason, lacks, behind = _reason(q, order, placed, now, now_cpus, now_mem, now_vram, outside_mem, rests)
        r = rests.get(q.job_id)
        after = (r.job_id, now - r.started - r.request.time, r.request.time) if r else None
        predictions[q.job_id] = Prediction(q.job_id, position, start, end, gpu_ids, reason, lacks, behind, after)

    return Plan(starts, predictions, running_ends)


def _reason(
    q, order, placed, now, free_cpus, free_mem, free_vram, outside_mem=0, rests=None
) -> tuple[str, tuple[str, ...], int | None]:
    """Why a queued job isn't starting now, and the same in a word or two for dibs top: what it lacks,
    and the job ahead in line it would delay."""
    req = q.request
    short, lacks = [], []
    if req.cpus > free_cpus:
        short.append(f"{_plural(req.cpus, 'CPU')} ({max(free_cpus, 0)} free)")
        lacks.append("CPUs")
    if req.mem > free_mem:
        outside = f", {format_size(outside_mem)} in use outside dibs" if outside_mem else ""
        left = format_size_against(max(free_mem, 0), req.mem)
        short.append(f"{format_size(req.mem)} RAM ({left} free{outside})")
        lacks.append("RAM")
    if req.gpus:
        roomy = sum(1 for free in free_vram if free >= req.vram)
        if roomy < req.gpus:
            vram = format_size(req.vram)
            lacks.append("GPUs" if req.gpus > 1 else "VRAM")
            if req.gpus > 1:
                short.append(f"{req.gpus} GPUs with {vram} VRAM free ({roomy} {'has' if roomy == 1 else 'have'} it)")
            else:
                most = format_size_against(max(max(free_vram, default=0), 0), req.vram)
                where = f"{most} free" if len(free_vram) == 1 else f"at most {most} free on one GPU"
                short.append(f"{vram} VRAM ({where})")
                if q.asked_vram is not None and q.asked_vram < req.vram:  # #183
                    short[-1] += f": its earlier runs held that much, though it asks for {format_size(q.asked_vram)}"
    if short:
        return "needs " + ", ".join(short), tuple(lacks), None
    for ahead in order:
        if ahead is q:
            break
        if ahead.job_id in placed:
            start, _, _ = placed[ahead.job_id]
            if now < start < now + req.time:
                r = (rests or {}).get(ahead.job_id)
                if r is not None:  # #181: and that start moves with every pass
                    over, estimate = format_duration(now - r.started - r.request.time), format_duration(r.request.time)
                    why = f"would delay #{ahead.job_id}, which waits for #{r.job_id} to end ({over} past its {estimate} estimate)"
                    return why, (), ahead.job_id
                return f"would delay #{ahead.job_id}, which is ahead in line", (), ahead.job_id
    return "waiting for a job ahead in line", (), None
