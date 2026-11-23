import random
import time

import pytest

from dibs.model import Capacity, Gpu, Request
from dibs.planner import Queued, Running, Start, expected_end, plan
from dibs.units import GiB, MiB, parse_size

NOW = 10_000.0


def cap(cpus=8, mem=32 * GiB, gpus=(8 * GiB,)):
    return Capacity(
        cpus=tuple(range(cpus)),
        mem=mem,
        gpus=tuple(Gpu(i, f"GPU-{i}", "Fake", v) for i, v in enumerate(gpus)),
    )


def req(cpus=1, mem=1 * GiB, gpus=0, vram=0, time=60.0, exclusive=False):
    return Request(cpus=cpus, mem=mem, gpus=gpus, vram=vram, time=time, exclusive=exclusive)


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


def test_reason_shows_a_shortfall_that_rounding_would_hide():
    # An "8G" GPU reports 8188 MiB: next to a 6G job, a 2G job is 4 MiB short.
    held = running(1, req(cpus=1, gpus=1, vram=6 * GiB, time=600), (0,), gpu_ids=(0,))
    p = plan(cap(gpus=(8188 * MiB,)), [held], [queued(2, req(gpus=1, vram=2 * GiB))], NOW)
    assert p.starts == []
    assert p.predictions[2].reason == "needs 2G VRAM (2044M free)"


def test_reason_shows_a_ram_shortfall_that_rounding_would_hide():
    typed = parse_size("1.1G")
    held = running(1, req(cpus=1, mem=32 * GiB - int(1.08 * GiB), time=600), (0,))
    p = plan(cap(), [held], [queued(2, req(cpus=1, mem=typed))], NOW)
    assert p.predictions[2].reason == "needs 1.1G RAM (1105M free)"


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


def test_expected_end_far_past_its_estimate_is_as_long_again_as_it_has_overrun():
    # #185: once a job has run 3 times its estimate, "a little longer" has been wrong pass after pass.
    assert expected_end(running(1, req(time=600), (0,), started=NOW - 1799), NOW) == NOW + 60
    assert expected_end(running(1, req(time=600), (0,), started=NOW - 1800), NOW) == NOW + 1200
    # A 50-minute job 4h20m in: 3h30m past its estimate, so 3h30m more.
    assert expected_end(running(1, req(time=3000), (0,), started=NOW - 15600), NOW) == NOW + 12600


def test_an_overrunning_job_is_expected_to_run_as_long_as_its_longest_similar_run():
    # #227: a 40-minute job 45 minutes in, whose batch's other runs of the same script and estimate ran
    # 14-89 minutes, is expected to end at 89 minutes, not 4 minutes from now pass after pass.
    j = Running(1, req(time=2400), (0,), (), NOW - 2700, longest=89 * 60)
    assert expected_end(j, NOW) == NOW - 2700 + 89 * 60
    # Never sooner than the old rule: past the longest run, a little longer again.
    late = Running(1, req(time=2400), (0,), (), NOW - 95 * 60, longest=89 * 60)
    assert expected_end(late, NOW) == NOW + 240


def test_short_jobs_start_beside_an_overrunning_job_whose_similar_runs_ran_long():
    # #227, the 11-21 21:31 freeze: a big job waited for the RAM of a 40-minute job 45 minutes in, and
    # every shorter job "would delay" it, since dibs expected the 40-minute job to end within 4 minutes.
    old = running(1, req(mem=10 * GiB, time=2400), (0,), started=NOW - 2700)
    big = queued(2, req(mem=14 * GiB, time=2400), submitted=NOW - 3600, tag=NOW - 3600)
    short = queued(3, req(mem=4 * GiB, time=1200), submitted=NOW - 600)
    assert plan(cap(mem=16 * GiB), [old], [big, short], NOW).starts == []
    known = Running(1, old.request, old.cpu_ids, old.gpu_ids, old.started, longest=89 * 60)
    p = plan(cap(mem=16 * GiB), [known], [big, short], NOW)
    assert [s.job_id for s in p.starts] == [3]  # it ends in 20 minutes, before the 89 minutes are up
    assert p.predictions[2].start == NOW - 2700 + 89 * 60


def test_short_job_starts_beside_a_job_far_past_its_estimate():
    # #185, as seen: a GPU job 3h30m past its 50-minute estimate holds most of the GPU. A GPU job waits
    # for it, and a 15-minute CPU job that fits now would take RAM the GPU job needs when it starts.
    # Assumed to end in 5 minutes, the overrunning job held the CPU job back ("would delay #2").
    long = running(1, req(cpus=8, mem=10 * GiB, gpus=1, vram=7 * GiB, time=3000), range(8), (0,), started=NOW - 15600)
    gpu_job = queued(2, req(cpus=2, mem=20 * GiB, gpus=1, vram=7 * GiB, time=3600), submitted=NOW - 7200, tag=NOW - 3600)
    cpu_job = queued(3, req(cpus=1, mem=14 * GiB, time=900), submitted=NOW - 300)
    p = plan(cap(cpus=24, mem=32 * GiB), [long], [gpu_job, cpu_job], NOW)
    assert [s.job_id for s in p.starts] == [3]
    assert p.predictions[2].start == NOW + 12600


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


def test_an_exclusive_job_waits_until_no_other_job_runs():
    # Issue #104: a timing run needs the machine to itself. It may wait; a job that can finish
    # before it starts may still go first, and one that can't, waits.
    c = cap(cpus=8, gpus=(8 * GiB,))
    other = running(1, req(cpus=1, time=100.0), (0,))
    timing = queued(2, req(cpus=2, time=50.0, exclusive=True), tag=NOW + 1)
    short = queued(3, req(cpus=1, time=50.0), tag=NOW + 2)
    long = queued(4, req(cpus=1, time=500.0), tag=NOW + 3)
    p = plan(c, [other], [timing, short, long], NOW)
    assert [s.job_id for s in p.starts] == [3]
    assert p.predictions[2].start == NOW + 100 and p.predictions[2].reason == "needs the machine to itself (1 job running)"
    assert p.predictions[4].start == NOW + 150


def test_nothing_starts_beside_a_running_exclusive_job():
    c = cap(cpus=8)
    timing = running(1, req(cpus=2, time=100.0, exclusive=True), (0, 1))
    p = plan(c, [timing], [queued(2, req(cpus=1, time=10.0))], NOW)
    assert p.starts == []
    assert (p.predictions[2].start, p.predictions[2].reason) == (NOW + 100, "waits for #1, which has the machine to itself")


def test_an_exclusive_job_gets_only_the_cpus_and_gpus_it_asked_for():
    c = cap(cpus=8, gpus=(8 * GiB, 8 * GiB))
    p = plan(c, [], [queued(1, req(cpus=2, gpus=1, vram=2 * GiB, exclusive=True))], NOW)
    assert p.starts == [Start(1, (0, 1), (0,))]


def test_a_waiting_job_says_in_a_word_what_it_lacks_or_whom_it_would_delay():
    # dibs top shows "needs RAM, VRAM" or "behind #2" instead of the whole reason (#143).
    head = queued(2, req(cpus=8, time=100), tag=0)
    small = queued(3, req(cpus=2, time=300), tag=1)
    p = plan(cap(), [_six_cpus_busy_until_100()], [head, small], NOW)
    assert (p.predictions[3].lacks, p.predictions[3].behind) == ((), 2)
    assert (p.predictions[2].lacks, p.predictions[2].behind) == (("CPUs",), None)
    big = queued(4, req(cpus=1, mem=40 * GiB, gpus=1, vram=9 * GiB), tag=0)
    assert plan(cap(), [], [big], NOW).predictions[4].lacks == ()  # can't fit this machine at all
    hungry = queued(5, req(cpus=1, mem=30 * GiB, gpus=1, vram=6 * GiB), tag=0)
    holder = running(6, req(cpus=1, mem=8 * GiB, gpus=1, vram=4 * GiB, time=200), [0], [0])
    assert plan(cap(), [holder], [hungry], NOW).predictions[5].lacks == ("RAM", "VRAM")


def test_one_gpu_jobs_spread_over_identical_gpus():
    # Issue #146: placed by VRAM alone, four 20G jobs all went to GPU 0 of two 179G GPUs while GPU 1
    # stayed empty, and jobs on one GPU share its compute.
    big = cap(cpus=64, mem=512 * GiB, gpus=(179 * GiB, 179 * GiB))
    jobs = [queued(i, req(gpus=1, vram=20 * GiB), tag=i) for i in range(1, 5)]
    p = plan(big, [], jobs, NOW)
    assert sorted(s.gpu_ids for s in p.starts) == [(0,), (0,), (1,), (1,)]


def test_a_small_job_takes_the_idle_gpu_not_the_busy_one():
    big = cap(cpus=64, mem=512 * GiB, gpus=(179 * GiB, 179 * GiB))
    busy = running(1, req(gpus=1, vram=100 * GiB, time=7200), (0,), (0,))
    p = plan(big, [busy], [queued(2, req(gpus=1, vram=20 * GiB))], NOW)
    assert p.starts[0].gpu_ids == (1,)


def test_a_job_counts_on_a_gpu_only_while_it_runs():
    # A job on GPU 1 that ends before the new one starts doesn't make GPU 1 busy for it.
    big = cap(cpus=64, mem=512 * GiB, gpus=(179 * GiB, 179 * GiB))
    soon = running(1, req(gpus=1, vram=20 * GiB, time=100), (0,), (1,), started=NOW - 90)
    later = running(2, req(gpus=1, vram=20 * GiB, time=7200), (1,), (0,))
    waits = queued(3, req(cpus=63, gpus=1, vram=20 * GiB, time=600))  # waits for job 1's core
    p = plan(big, [soon, later], [waits], NOW)
    assert p.predictions[3].gpu_ids == (1,)


def test_a_start_that_rests_on_a_job_past_its_estimate_names_it():
    # #181: a GPU job waits for a job 10 minutes past its 50-minute estimate. Assumed to end 5 minutes
    # from now, pass after pass, it made the wait read "starts in 5m" for as long as it kept running.
    long = running(1, req(cpus=8, mem=10 * GiB, gpus=1, vram=7 * GiB, time=3000), range(8), (0,), started=NOW - 3600)
    gpu_job = queued(2, req(cpus=2, mem=20 * GiB, gpus=1, vram=7 * GiB, time=3600), submitted=NOW - 7200, tag=NOW - 3600)
    cpu_job = queued(3, req(cpus=1, mem=14 * GiB, time=900), submitted=NOW - 300)
    p = plan(cap(cpus=24, mem=32 * GiB), [long], [gpu_job, cpu_job], NOW)
    assert p.predictions[2].start == NOW + 300 and p.predictions[2].after == (1, 600.0, 3000)
    assert p.predictions[3].reason == "would delay #2, which waits for #1 to end (10m past its 50m estimate)"
    assert p.predictions[3].after is None  # its own start rests on #2's end, not on #1's


def test_a_start_that_rests_on_a_job_within_its_estimate_names_no_job():
    # The same wait, but the running job is due to end in 5 minutes within its estimate.
    long = running(1, req(cpus=8, mem=10 * GiB, gpus=1, vram=7 * GiB, time=3000), range(8), (0,), started=NOW - 2700)
    gpu_job = queued(2, req(cpus=2, mem=20 * GiB, gpus=1, vram=7 * GiB, time=3600), submitted=NOW - 7200, tag=NOW - 3600)
    cpu_job = queued(3, req(cpus=1, mem=14 * GiB, time=900), submitted=NOW - 300)
    p = plan(cap(cpus=24, mem=32 * GiB), [long], [gpu_job, cpu_job], NOW)
    assert p.predictions[2].after is None
    assert p.predictions[3].reason == "would delay #2, which is ahead in line"
