"""Properties of the whole scheduling policy, checked by simulation over the real planner."""

import random

import pytest
from sim import SimJob, simulate

from dibs.model import Capacity, Gpu, Request, compute_tag
from dibs.planner import Queued, plan
from dibs.units import GiB

ONE_GPU = Capacity(cpus=tuple(range(8)), mem=32 * GiB, gpus=(Gpu(0, "GPU-0", "Fake", 8 * GiB),))
TWO_GPUS = Capacity(
    cpus=tuple(range(16)),
    mem=64 * GiB,
    gpus=(Gpu(0, "GPU-0", "Fake", 8 * GiB), Gpu(1, "GPU-1", "Fake", 24 * GiB)),
)


def whole_gpu(time):
    return Request(cpus=1, mem=1 * GiB, gpus=1, vram=8 * GiB, time=time)


def everything(time):
    return Request(cpus=8, mem=32 * GiB, gpus=1, vram=8 * GiB, time=time)


def random_workload(seed, n=40):
    rng = random.Random(seed)
    jobs = []
    for job_id in range(1, n + 1):
        vram = rng.choice([0, 0, 2, 6, 12, 20]) * GiB
        estimate = rng.uniform(30, 1800)
        jobs.append(
            SimJob(
                job_id=job_id,
                session=f"s{rng.randint(1, 4)}",
                submit=rng.uniform(0, 3600),
                request=Request(
                    cpus=rng.randint(1, 16),
                    mem=rng.randint(1, 64) * GiB,
                    gpus=1 if vram else 0,
                    vram=vram,
                    time=estimate,
                ),
                actual=estimate * rng.uniform(0.2, 3.0),
            )
        )
    return jobs


def test_sim_user_example():
    a = SimJob(1, "agent-a", 0.0, everything(60), 60)
    b = SimJob(2, "agent-b", 0.0, everything(600), 600)
    result = simulate(ONE_GPU, [b, a])
    assert result[1][1] <= result[2][0]


@pytest.mark.parametrize("seed", range(200))
def test_never_overcommits(seed):
    simulate(TWO_GPUS, random_workload(seed))  # simulate() asserts after every pass


@pytest.mark.parametrize("seed", range(20))
def test_every_job_runs(seed):
    jobs = random_workload(seed)
    result = simulate(TWO_GPUS, jobs)
    assert set(result) == {j.job_id for j in jobs}
    for j in jobs:
        start, end = result[j.job_id]
        assert j.submit <= start < end


def test_long_job_not_starved():
    long_job = SimJob(1, "L", 0.0, whole_gpu(1200), 1200)
    stream = []
    for i in range(40):  # three sessions, one 60 s whole-GPU job each per minute: 3x overload
        for s in ("s1", "s2", "s3"):
            stream.append(SimJob(len(stream) + 2, s, 60.0 * i, whole_gpu(60), 60))
    result = simulate(ONE_GPU, [long_job] + stream)
    started = result[1][0]
    tag = 1200.0
    # Fair share: with four sessions competing, L waits at most about four times its tag.
    assert started <= 4 * tag
    # And nothing submitted after L's tag runs before it.
    assert all(result[j.job_id][0] > started for j in stream if j.submit > tag)


def test_flood_is_fair():
    flood = [SimJob(i, "s1", 0.0, whole_gpu(60), 60) for i in range(1, 51)]
    other = SimJob(51, "s2", 0.0, whole_gpu(600), 600)
    result = simulate(ONE_GPU, flood + [other])
    assert result[51][0] == 600.0


@pytest.mark.parametrize("seed", range(50))
def test_lower_priority_jobs_never_move_higher_ones(seed):
    rng = random.Random(seed)
    jobs = random_workload(seed, n=25)
    queued, backlog = [], {}
    for j in jobs:
        tag = compute_tag(j.submit, j.request.time, backlog.get(j.session, []))
        backlog.setdefault(j.session, []).append(tag)
        queued.append(Queued(j.job_id, j.request, tag, j.submit))
    full = plan(TWO_GPUS, [], queued, 0.0)
    order = sorted(queued, key=lambda q: (q.tag, q.submitted, q.job_id))
    victim = rng.choice(order)
    without = plan(TWO_GPUS, [], [q for q in queued if q is not victim], 0.0)
    for q in order[: order.index(victim)]:
        a, b = full.predictions[q.job_id], without.predictions[q.job_id]
        assert (a.start, a.gpu_ids) == (b.start, b.gpu_ids)
