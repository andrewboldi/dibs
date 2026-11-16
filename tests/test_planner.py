import random
import time

import pytest

from dibs.model import Capacity, Gpu, Request
from dibs.planner import Queued, Running, Start, expected_end, plan
from dibs.units import GiB

NOW = 10_000.0


def cap(cpus=8, mem=32 * GiB, gpus=(8 * GiB,)):
    return Capacity(
        cpus=tuple(range(cpus)),
        mem=mem,
        gpus=tuple(Gpu(i, f"GPU-{i}", "Fake", v) for i, v in enumerate(gpus)),
    )


def req(cpus=1, mem=1 * GiB, gpus=0, vram=0, time=60.0):
    return Request(cpus=cpus, mem=mem, gpus=gpus, vram=vram, time=time)


ALL = dict(cpus=8, mem=32 * GiB, gpus=1, vram=8 * GiB)


def running(job_id, request, cpu_ids, gpu_ids=(), started=NOW):
    return Running(job_id, request, tuple(cpu_ids), tuple(gpu_ids), started)


def queued(job_id, request, submitted=NOW, tag=None):
    return Queued(job_id, request, submitted + request.time if tag is None else tag, submitted)


def test_empty_queue_starts_nothing():
    p = plan(cap(), [], [], NOW)
    assert p.starts == [] and p.predictions == {}


def test_job_starts_now_when_free():
    p = plan(cap(), [], [queued(1, req(cpus=4, gpus=1, vram=6 * GiB))], NOW)
    assert p.starts == [Start(1, (0, 1, 2, 3), (0,))]
    assert p.predictions[1].start == NOW
    assert p.predictions[1].reason == ""


def test_start_avoids_busy_cpus():
    r = running(1, req(cpus=4, time=600), cpu_ids=(0, 1, 2, 3))
    p = plan(cap(), [r], [queued(2, req(cpus=2))], NOW)
    assert p.starts == [Start(2, (4, 5), ())]


def test_user_example_short_first():
    t0 = NOW - 5
    x = running(1, req(**ALL, time=200), cpu_ids=range(8), gpu_ids=(0,), started=NOW - 100)
    b = queued(2, req(**ALL, time=1000), submitted=t0)
    a = queued(3, req(**ALL, time=60), submitted=t0 + 5)
    p = plan(cap(), [x], [b, a], NOW)
    assert p.starts == []
    assert p.predictions[3].position == 1 and p.predictions[3].start == NOW + 100
    assert p.predictions[2].position == 2 and p.predictions[2].start == p.predictions[3].end


def test_long_job_is_not_overtaken_after_waiting_its_estimate():
    t0 = NOW - 601
    long_job = queued(1, req(**ALL, time=600), submitted=t0)
    newcomer = queued(2, req(**ALL, time=1), submitted=t0 + 601)
    p = plan(cap(), [], [newcomer, long_job], NOW)
    assert p.predictions[1].position == 1
    assert [s.job_id for s in p.starts] == [1]


def test_backfill_fills_gap():
    r = running(1, req(cpus=2, gpus=1, vram=8 * GiB, time=200), range(2), (0,), started=NOW - 100)
    head = queued(2, req(cpus=2, gpus=1, vram=8 * GiB, time=300), tag=0)
    small = queued(3, req(cpus=2, time=30), tag=1)
    p = plan(cap(), [r], [head, small], NOW)
    assert [s.job_id for s in p.starts] == [3]
    assert p.predictions[2].start == NOW + 100


def _six_cpus_busy_until_100():
    return running(1, req(cpus=6, time=200), range(6), started=NOW - 100)


def test_backfill_never_delays_head():
    head = queued(2, req(cpus=8, time=100), tag=0)
    small = queued(3, req(cpus=2, time=300), tag=1)
    p = plan(cap(), [_six_cpus_busy_until_100()], [head, small], NOW)
    assert p.starts == []
    assert p.predictions[3].start >= p.predictions[2].end
    assert p.predictions[3].reason == "would delay #2, which is ahead in line"


def test_short_backfill_allowed_when_it_ends_first():
    head = queued(2, req(cpus=8, time=100), tag=0)
    small = queued(3, req(cpus=2, time=50), tag=1)
    p = plan(cap(), [_six_cpus_busy_until_100()], [head, small], NOW)
    assert [s.job_id for s in p.starts] == [3]
    assert p.predictions[2].start == NOW + 100


def test_vram_packing():
    jobs = [queued(i, req(gpus=1, vram=3 * GiB), tag=i) for i in (1, 2, 3)]
    p = plan(cap(), [], jobs, NOW)
    assert [s.job_id for s in p.starts] == [1, 2]
    assert p.predictions[3].reason == "needs 3G VRAM (2G free)"


def test_best_fit_gpu():
    small = queued(1, req(gpus=1, vram=6 * GiB), tag=1)
    big = queued(2, req(gpus=1, vram=16 * GiB), tag=2)
    p = plan(cap(gpus=(24 * GiB, 8 * GiB)), [], [small, big], NOW)
    assert {s.job_id: s.gpu_ids for s in p.starts} == {1: (1,), 2: (0,)}


def test_multi_gpu_request():
    p = plan(cap(gpus=(8 * GiB, 8 * GiB)), [], [queued(1, req(gpus=2, vram=6 * GiB))], NOW)
    assert p.starts[0].gpu_ids == (0, 1)


def test_zero_gpu_machine():
    cpu_job = queued(1, req(cpus=2), tag=1)
    gpu_job = queued(2, req(gpus=1, vram=1 * GiB), tag=2)
    p = plan(cap(gpus=()), [], [cpu_job, gpu_job], NOW)
    assert [s.job_id for s in p.starts] == [1]
    assert p.predictions[2].start is None
    assert p.predictions[2].reason.startswith("can't fit this machine:")


def test_expected_end_overrun():
    assert expected_end(running(1, req(time=600), (0,), started=NOW - 1000), NOW) == NOW + 60
    assert expected_end(running(1, req(time=6000), (0,), started=NOW - 7000), NOW) == NOW + 600
    assert expected_end(running(1, req(time=600), (0,), started=NOW - 100), NOW) == NOW + 500


@pytest.mark.parametrize(
    "held,wanted,reason",
    [
        (dict(cpus=4, mem=1 * GiB), dict(cpus=8, mem=1 * GiB), "needs 8 CPUs (4 free)"),
        (dict(cpus=1, mem=24 * GiB), dict(cpus=1, mem=16 * GiB), "needs 16G RAM (8G free)"),
        (
            dict(cpus=4, mem=24 * GiB),
            dict(cpus=8, mem=16 * GiB),
            "needs 8 CPUs (4 free), 16G RAM (8G free)",
        ),
    ],
)
def test_reason_names_short_resources(held, wanted, reason):
    r = running(1, req(**held, time=600), range(held["cpus"]))
    p = plan(cap(), [r], [queued(2, req(**wanted))], NOW)
    assert p.predictions[2].reason == reason


def test_multi_gpu_reason():
    r = running(1, req(gpus=1, vram=4 * GiB, time=600), (0,), (0,))
    p = plan(cap(gpus=(8 * GiB, 8 * GiB)), [r], [queued(2, req(gpus=2, vram=6 * GiB))], NOW)
    assert p.predictions[2].reason == "needs 2 GPUs with 6G VRAM free (1 has it)"


def test_vram_reason_on_a_multi_gpu_machine():
    r = running(1, req(gpus=1, vram=6 * GiB, time=600), (0,), (0,))
    s = running(2, req(gpus=1, vram=5 * GiB, time=600), (1,), (1,))
    p = plan(cap(gpus=(8 * GiB, 8 * GiB)), [r, s], [queued(3, req(gpus=1, vram=4 * GiB))], NOW)
    assert p.predictions[3].reason == "needs 4G VRAM (at most 3G free on one GPU)"


def test_positions_follow_tag_order():
    blocker = running(1, req(**ALL, time=600), range(8), (0,))
    jobs = [queued(2, req(), tag=300), queued(3, req(), tag=100), queued(4, req(), tag=200)]
    p = plan(cap(), [blocker], jobs, NOW)
    assert {j: p.predictions[j].position for j in (2, 3, 4)} == {3: 1, 4: 2, 2: 3}


def _random_world(seed):
    rng = random.Random(seed)
    capacity = cap(cpus=24, mem=64 * GiB, gpus=(8 * GiB, 24 * GiB))
    run, free_cpus, mem_left, vram_left = [], list(range(24)), 64 * GiB, [8 * GiB, 24 * GiB]
    for job_id in range(1, 21):
        c, m, g = rng.randint(1, 2), rng.randint(1, 3) * GiB, rng.randint(0, 2) * GiB
        gpu = next((i for i, v in enumerate(vram_left) if v >= g), None) if g else None
        if c > len(free_cpus) or m > mem_left or (g and gpu is None):
            continue
        ids, free_cpus = free_cpus[:c], free_cpus[c:]
        mem_left -= m
        if g:
            vram_left[gpu] -= g
        r = req(cpus=c, mem=m, gpus=1 if g else 0, vram=g, time=rng.uniform(60, 3600))
        run.append(running(job_id, r, ids, (gpu,) if g else (), started=NOW - rng.uniform(0, 3000)))
    queue = []
    for job_id in range(100, 200):
        g = rng.choice([0, 0, 2, 6, 12]) * GiB
        r = req(
            cpus=rng.randint(1, 16),
            mem=rng.randint(1, 32) * GiB,
            gpus=1 if g else 0,
            vram=g,
            time=rng.uniform(30, 7200),
        )
        queue.append(queued(job_id, r, submitted=NOW - rng.uniform(0, 600)))
    return capacity, run, queue


def test_plan_is_deterministic():
    capacity, run, queue = _random_world(1)
    assert plan(capacity, run, queue, NOW) == plan(capacity, run, queue, NOW)


def test_plan_is_fast():
    capacity, run, queue = _random_world(2)
    started = time.perf_counter()
    plan(capacity, run, queue, NOW)
    assert time.perf_counter() - started < 0.5


def test_ram_used_outside_dibs_is_not_handed_out():
    # Issue #30: dibs treated all of the machine's RAM as its own, so reservations could add up to
    # more RAM than exists while other programs were using it.
    jobs = [queued(1, req(mem=12 * GiB))]
    assert [s.job_id for s in plan(cap(mem=16 * GiB), [], jobs, NOW).starts] == [1]
    p = plan(cap(mem=16 * GiB), [], jobs, NOW, outside_mem=6 * GiB)
    assert p.starts == [] and p.predictions[1].start is None
    assert p.predictions[1].reason == "needs 12G RAM (10G free, 6G in use outside dibs)"


def test_a_job_waiting_for_ram_used_outside_dibs_doesnt_hold_up_the_line():
    jobs = [queued(1, req(mem=12 * GiB), tag=1.0), queued(2, req(mem=4 * GiB), tag=2.0)]
    p = plan(cap(mem=16 * GiB), [], jobs, NOW, outside_mem=6 * GiB)
    assert [s.job_id for s in p.starts] == [2]
    assert p.predictions[1].start is None and "6G in use outside dibs" in p.predictions[1].reason


def test_ram_used_outside_dibs_never_makes_a_job_unfit_for_the_machine():
    p = plan(cap(mem=16 * GiB), [running(1, req(mem=8 * GiB), [0])], [queued(2, req(mem=14 * GiB))], NOW,
             outside_mem=12 * GiB)
    reason = p.predictions[2].reason
    assert "can't fit this machine" not in reason and "12G in use outside dibs" in reason
