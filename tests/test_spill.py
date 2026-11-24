"""A job may spill past its --mem into swap, up to half as much again, instead of being killed (#257).
dibs counts what it holds in RAM and swap together, stops a job that thrashes at its own limit (sized
from both, so it runs again with enough), and stops the job that spilled the most when swap runs short.
A job "presses its own limit" when its memory.events max count rose since dibs's last look: when the
machine runs short instead, the kernel takes pages from every job and that count stays put."""

import time

import pytest
from conftest import run_until
from test_daemon import PY, _borrowing, _job_holding, finished, job, submit

from dibs import launcher, procs
from dibs.daemon import RECENT_PEAK, THRASH_AFTER
from dibs.model import State
from dibs.units import GiB, MiB


def test_a_scope_lets_its_job_hold_the_given_swap():
    props = launcher.scope_properties(8 * GiB, 4 * GiB)
    assert f"MemoryMax={8 * GiB}" in props and f"MemorySwapMax={4 * GiB}" in props
    assert "MemorySwapMax=0" in launcher.scope_properties(64 * MiB)  # the probe's, and a timing run's


@pytest.mark.parametrize("exclusive, swap", [(False, 4 * GiB), (True, 0)])
def test_a_job_may_spill_half_its_request_into_swap_but_a_timing_run_none(daemon, tmp_path, monkeypatch, exclusive, swap):
    seen = []
    real = launcher.launch
    monkeypatch.setattr(launcher, "launch", lambda j, paths, **kw: seen.append(kw.get("swap_limit")) or real(j, paths, **kw))
    j = submit(daemon, tmp_path, [PY, "-c", "pass"], mem=8 * GiB, exclusive=exclusive)
    run_until(daemon, lambda: finished(daemon, j))
    assert seen == [swap]


def test_a_scopes_stall_and_limit_hits_are_read_from_its_files(tmp_path, monkeypatch):
    scope = tmp_path / "x.scope"
    scope.mkdir()
    monkeypatch.setattr(procs, "CGROUP_ROOT", str(tmp_path))
    (scope / "memory.pressure").write_text("some avg10=31.50 avg60=1.00 avg300=0.00 total=9\nfull avg10=22.25 avg60=0.50 avg300=0.00 total=5\n")
    (scope / "memory.events").write_text("low 0\nhigh 0\nmax 879\noom 0\noom_kill 0\noom_group_kill 0\n")
    (scope / "memory.swap.current").write_text("43180032\n")
    assert procs.memory_stall("/x.scope") == (31.5, 22.25)
    assert procs.limit_hits("/x.scope") == 879 and procs.swap_in_use("/x.scope") == 43180032
    assert procs.memory_stall("/missing.scope") is None and procs.limit_hits("/missing.scope") is None


def _reading(stat, ram, swap=0, stall=0.0, hits=0):
    """What the job's scope says: RAM it holds, what it holds in swap, how much of the last 10 s its
    processes waited on memory, and how many times its memory reached its own limit."""
    stat.write_text(f"anon {ram}\nshmem 0\n")
    (stat.parent / "memory.swap.current").write_text(f"{swap}\n")
    (stat.parent / "memory.pressure").write_text(
        f"some avg10={stall:.2f} avg60=0.00 avg300=0.00 total=0\nfull avg10={stall:.2f} avg60=0.00 avg300=0.00 total=0\n"
    )
    (stat.parent / "memory.events").write_text(f"low 0\nhigh 0\nmax {hits}\noom 0\noom_kill 0\n")


def _stat_of(tmp_path):
    return tmp_path / "cgroup" / "app.slice" / "dibs-test-job-1.scope" / "memory.stat"


def test_a_job_whose_pages_went_to_swap_still_counts_them(make_daemon, tmp_path, monkeypatch):
    # The kernel moved 1.5G of A's 2G to swap: that RAM isn't free for another job, since A may touch it.
    now = [time.time()]
    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=700, now=now)
    _reading(_stat_of(tmp_path), ram=GiB // 2, swap=int(1.5 * GiB))
    now[0] += RECENT_PEAK + 60  # the 2G samples are out of the window
    d.tick()
    assert d._planned_mem(job(d, a), now[0]) == 2 * GiB
    go.touch()


def _thrashing(make_daemon, tmp_path, monkeypatch, *, retry, stall=60.0):
    """A 4G job at its limit, 1.5G more in swap, waiting on memory `stall`% of the time. 12G of 16G is in
    use, so it was launched with no headroom: its limit is its request. `look(seconds)` moves the clock
    on and ticks, the job having reached its own limit again since the last look unless `pressing` is
    false (the machine took its pages)."""
    now = [time.time()]
    d = make_daemon(clock=lambda: now[0], meminfo=lambda: (16 * GiB, 12 * GiB))
    a, stat, go = _job_holding(d, tmp_path, monkeypatch, 4 * GiB, retry=retry)
    assert job(d, a).hard_limit is None
    hits = [0]

    def look(seconds, *, ram=4 * GiB, pressing=True):
        now[0] += seconds
        hits[0] += 100 if pressing else 0
        _reading(stat, ram=ram, swap=int(1.5 * GiB), stall=stall, hits=hits[0])
        d.tick()

    look(0)
    return d, a, look, go


def test_a_job_thrashing_at_its_own_limit_is_stopped_and_runs_again_with_enough(make_daemon, tmp_path, monkeypatch):
    d, a, look, go = _thrashing(make_daemon, tmp_path, monkeypatch, retry=True)
    look(1)
    look(THRASH_AFTER / 2)
    assert job(d, a).stop_requested is None  # not on a moment's stall
    look(THRASH_AFTER)
    assert "waited on memory 60% of the time at its RAM limit, with 1.5G spilled into swap" in job(d, a).stop_reason
    run_until(d, lambda: job(d, a).attempts)
    j = job(d, a)
    assert j.state == State.QUEUED and j.request.mem == 7 * GiB  # (4G + 1.5G) and a quarter, in whole G
    assert j.attempts[0]["reason"].endswith("retrying with --mem 7G")
    go.touch()


def test_without_retry_a_job_thrashing_at_its_limit_fails_saying_what_to_ask(make_daemon, tmp_path, monkeypatch):
    d, a, look, go = _thrashing(make_daemon, tmp_path, monkeypatch, retry=False)
    for _ in range(4):
        look(THRASH_AFTER / 2)
    run_until(d, lambda: finished(d, a))
    assert job(d, a).state == State.FAILED and job(d, a).reason.endswith("; resubmit with --mem 7G")
    go.touch()


def test_a_job_spilling_without_thrashing_runs_on(make_daemon, tmp_path, monkeypatch):
    d, a, look, go = _thrashing(make_daemon, tmp_path, monkeypatch, retry=True, stall=5.0)
    for _ in range(4):
        look(THRASH_AFTER)
    assert job(d, a).stop_requested is None and job(d, a).peak_swap == int(1.5 * GiB)
    go.touch()


@pytest.mark.parametrize("ram", [4 * GiB, 2 * GiB])
def test_a_job_waiting_on_memory_the_machine_took_isnt_stopped_for_it(make_daemon, tmp_path, monkeypatch, ram):
    # At its limit or below it, its count of reaching its own limit stays put: the machine ran short and
    # took its pages. That's the shortage relief's to answer, not this job's sizing.
    d, a, look, go = _thrashing(make_daemon, tmp_path, monkeypatch, retry=True)
    for _ in range(4):
        look(THRASH_AFTER, ram=ram, pressing=False)
    assert job(d, a).stop_requested is None
    go.touch()


def test_a_job_below_the_headroom_it_was_launched_with_isnt_at_its_limit(make_daemon, tmp_path, monkeypatch):
    # Launched with a step of headroom while the machine had RAM free, its limit is 5G: holding 4G with
    # 1G in swap, it didn't spill past its limit, so it isn't the one to stop when swap runs short.
    now = [time.time()]
    d = make_daemon(clock=lambda: now[0], meminfo=lambda: (16 * GiB, 6 * GiB))
    a, stat, go = _job_holding(d, tmp_path, monkeypatch, 4 * GiB, retry=True)
    assert job(d, a).hard_limit == 5 * GiB
    _reading(stat, ram=4 * GiB, swap=GiB)
    d.tick()
    assert a not in d._at_limit
    _reading(stat, ram=5 * GiB, swap=GiB)
    d.tick()
    assert a in d._at_limit
    go.touch()


def test_a_job_pressing_its_own_limit_isnt_the_machine_running_short(make_daemon, tmp_path, monkeypatch):
    # Its processes waiting on its own swap raise the machine's pressure too: that mustn't stop jobs
    # starting on unused RAM, or growing into RAM the machine has free, or count as a shortage.
    d, a, look, go = _thrashing(make_daemon, tmp_path, monkeypatch, retry=True, stall=15.0)
    d.pressure = lambda: 15.0  # all of it A's own
    d._meminfo = lambda: (16 * GiB, int(15.5 * GiB))  # and the machine under its reserve
    d.swap_free = lambda: 10 * GiB
    look(1)
    assert d._borrowing() and not d._shortage()
    d._meminfo = lambda: (16 * GiB, 12 * GiB)
    assert d._spare(d.store.jobs([State.RUNNING]))[0] > 0
    d.pressure = lambda: 30.0  # more than A explains: the machine's own
    assert not d._borrowing() and d._spare(d.store.jobs([State.RUNNING])) == (0, 0)
    go.touch()


def _packed(make_daemon, tmp_path, monkeypatch, *, a_ram, a_swap, a_stall, a_pressing, swap_free, pressure):
    """A (12G) and borrower B (8G) on 16G with 1G available, under the 1.5G reserve."""
    level, used, now = [0.0], [2 * GiB], [time.time()]
    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=700, pressure=lambda: level[0], now=now,
                             meminfo=lambda: (16 * GiB, used[0]))  # fmt: skip
    assert job(d, b).borrowed_ram
    hogs = []
    d._stop_ram_hog = lambda t: hogs.append(t) or True
    d.swap_free = lambda: swap_free
    level[0], used[0] = pressure, 15 * GiB
    for i in range(4):
        _reading(_stat_of(tmp_path), ram=a_ram, swap=a_swap, stall=a_stall, hits=100 * i if a_pressing else 0)
        d.tick()
        now[0] += 6.0
    return d, a, b, go, hogs


def test_one_job_stalling_at_its_own_limit_on_a_packed_machine_stops_nothing_else(make_daemon, tmp_path, monkeypatch):
    # The pressure is A's own, at its limit, and too little to stop A for: stopping B wouldn't help A.
    d, a, b, go, hogs = _packed(make_daemon, tmp_path, monkeypatch, a_ram=12 * GiB, a_swap=GiB, a_stall=15.0,
                                a_pressing=True, swap_free=10 * GiB, pressure=15.0)  # fmt: skip
    assert job(d, b).stop_requested is None and job(d, a).stop_requested is None and hogs == []
    go.touch()


def test_a_machine_short_of_memory_stops_the_newest_borrower_not_a_job_at_its_limit(make_daemon, tmp_path, monkeypatch):
    # A sits at its limit and waits on memory, but its count of reaching its limit stays put: the
    # machine is taking its pages. The relief stops B, which started on A's unused RAM; A runs on.
    d, a, b, go, hogs = _packed(make_daemon, tmp_path, monkeypatch, a_ram=12 * GiB, a_swap=GiB, a_stall=60.0,
                                a_pressing=False, swap_free=10 * GiB, pressure=60.0)  # fmt: skip
    assert job(d, b).stop_reason == "stopped to give back borrowed RAM: the machine was short of memory"
    assert job(d, a).stop_requested is None and hogs == []
    go.touch()


def test_when_swap_runs_short_the_job_that_spilled_the_most_goes_first(make_daemon, tmp_path, monkeypatch):
    d, a, b, go, hogs = _packed(make_daemon, tmp_path, monkeypatch, a_ram=12 * GiB, a_swap=3 * GiB, a_stall=0.0,
                                a_pressing=True, swap_free=GiB, pressure=0.0)  # fmt: skip
    assert "ran short of swap" in job(d, a).stop_reason
    assert job(d, b).stop_requested is None and hogs == []
    go.touch()


def test_dibs_show_says_what_a_job_spilled_into_swap(make_daemon, tmp_path, monkeypatch):
    from dibs.render import render_show

    d, a, look, go = _thrashing(make_daemon, tmp_path, monkeypatch, retry=True, stall=5.0)
    shown = render_show(d.show(a)["job"], d.clock())
    assert "RAM 4G of 4G (not counting file cache), 1.5G more spilled into swap" in shown
    go.touch()
