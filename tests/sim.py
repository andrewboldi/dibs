"""A discrete-event simulation of the dibs scheduler, driving the real planner.

Jobs arrive, get tagged the way the daemon tags them, and start when plan() says so; each runs for
its *actual* duration, which may differ from its estimate. After every scheduling pass the
simulator checks that no resource is ever over-committed.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from dibs.model import Capacity, Request, compute_tag
from dibs.planner import Queued, Running, plan


@dataclass
class SimJob:
    job_id: int
    session: str
    submit: float
    request: Request
    actual: float


def _check(capacity: Capacity, running: dict[int, tuple[Running, float]]) -> None:
    runs = [r for r, _ in running.values()]
    cpu_ids = [c for r in runs for c in r.cpu_ids]
    assert len(cpu_ids) == len(set(cpu_ids)), "two running jobs share a CPU"
    assert set(cpu_ids) <= set(capacity.cpus), "a job got a CPU dibs doesn't manage"
    assert sum(r.request.mem for r in runs) <= capacity.mem, "RAM over-committed"
    for gpu in capacity.gpus:
        used = sum(r.request.vram for r in runs if gpu.index in r.gpu_ids)
        assert used <= gpu.vram, f"VRAM over-committed on GPU {gpu.index}"
    for r in runs:
        assert len(r.cpu_ids) == r.request.cpus and len(r.gpu_ids) == r.request.gpus


def simulate(capacity: Capacity, jobs: Sequence[SimJob]) -> dict[int, tuple[float, float]]:
    """Run the workload; return each job's (start, end)."""
    by_id = {j.job_id: j for j in jobs}
    arrivals = sorted(jobs, key=lambda j: (j.submit, j.job_id))
    queued: dict[int, Queued] = {}
    running: dict[int, tuple[Running, float]] = {}
    result: dict[int, tuple[float, float]] = {}
    next_arrival, wake = 0, math.inf
    while next_arrival < len(arrivals) or queued or running:
        now = min(
            arrivals[next_arrival].submit if next_arrival < len(arrivals) else math.inf,
            min((end for _, end in running.values()), default=math.inf),
            wake,
        )
        assert now < math.inf, "jobs are queued but nothing can ever start"
        for job_id, (_, end) in list(running.items()):
            if end <= now:
                del running[job_id]
        while next_arrival < len(arrivals) and arrivals[next_arrival].submit <= now:
            j = arrivals[next_arrival]
            next_arrival += 1
            backlog = [q.tag for q in queued.values() if by_id[q.job_id].session == j.session]
            queued[j.job_id] = Queued(
                j.job_id, j.request, compute_tag(j.submit, j.request.time, backlog), j.submit
            )
        p = plan(capacity, [r for r, _ in running.values()], list(queued.values()), now)
        for s in p.starts:
            q = queued.pop(s.job_id)
            running[s.job_id] = (Running(s.job_id, q.request, s.cpu_ids, s.gpu_ids, now), now + by_id[s.job_id].actual)
            result[s.job_id] = (now, now + by_id[s.job_id].actual)
        _check(capacity, running)
        # Like the daemon's timer: look again when the plan next expects something to start.
        wake = min((pr.start for pr in p.predictions.values() if pr.start and pr.start > now), default=math.inf)
    return result
