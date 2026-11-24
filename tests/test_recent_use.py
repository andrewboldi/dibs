"""dibs counts a running job at the most it used over the last ten minutes, with no margin, so the
machine fills with what jobs really use (#255). A job with no earlier runs counts at its request while
it may still be growing; RAM a job grew into counts by its use too, never past its limit."""

import time
from collections import deque

from test_daemon import _borrowing, _job_holding, job, submit, PY

from dibs.daemon import RECENT_PEAK
from dibs.model import State
from dibs.units import GiB


def test_a_job_counts_at_its_recent_use_with_no_margin(make_daemon, tmp_path, monkeypatch):
    # A reserves 12G of 16G and uses 2G after 700 s: it counted as 3G (its peak and 1G); now 2G.
    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=700)
    now = d.clock()
    assert d._planned_mem(job(d, a), now) == 2 * GiB
    go.touch()


def test_an_old_peak_drops_out_after_ten_minutes(make_daemon, tmp_path, monkeypatch):
    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=700)
    now = d.clock()
    d._recent_held[a] = deque([(now - RECENT_PEAK - 60, 6 * GiB), (now - RECENT_PEAK + 30, 2 * GiB), (now, 2 * GiB)])
    assert d._planned_mem(job(d, a), now) == 2 * GiB  # the 6G peak is older than the window
    d._recent_held[a] = deque([(now - RECENT_PEAK + 30, 6 * GiB), (now, 2 * GiB)])
    assert d._planned_mem(job(d, a), now) == 6 * GiB
    go.touch()


def test_after_a_restart_a_job_counts_at_its_peak_until_dibs_has_watched_it_ten_minutes(make_daemon, tmp_path, monkeypatch):
    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=700)
    now = d.clock()
    j = job(d, a)
    j.peak_mem = 6 * GiB  # from before the restart
    d.store.update_job(j)
    d._recent_held[a] = deque([(now - 30, 2 * GiB), (now, 2 * GiB)])
    d._watching[a] = now - 30  # dibs restarted 30 s ago, ten minutes into the job
    assert d._planned_mem(job(d, a), now) == 6 * GiB
    d._watching[a] = now - RECENT_PEAK  # ten minutes after the restart: the old peak no longer counts
    assert d._planned_mem(job(d, a), now) == 2 * GiB
    go.touch()


def test_a_grown_job_counts_at_its_use_never_past_its_limit(make_daemon, tmp_path, monkeypatch):
    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=700)
    now = d.clock()
    j = job(d, a)
    j.ram_limit = 14 * GiB  # grown past its 12G
    d.store.update_job(j)
    d._recent_held[a] = deque([(now - RECENT_PEAK, 13 * GiB), (now, 13 * GiB)])
    assert d._planned_mem(job(d, a), now) == 13 * GiB  # not its whole 14G limit
    d._recent_held[a] = deque([(now - RECENT_PEAK, 15 * GiB)])
    assert d._planned_mem(job(d, a), now) == 14 * GiB
    go.touch()


def test_a_young_job_counts_at_half_its_request_or_what_it_holds_if_more(make_daemon, tmp_path, monkeypatch):
    # #266: in its first ten minutes it may still be growing, but most young jobs never reach their
    # request; the user picked half, measured to overflow the reserve 6.4% of the time against 3.7%.
    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=100)  # A asked 12G and holds 2G
    now = d.clock()
    assert d._planned_mem(job(d, a), now) == 6 * GiB
    d._recent_held[a] = deque([(now, 8 * GiB)])
    assert d._planned_mem(job(d, a), now) == 8 * GiB
    go.touch()


def test_a_young_job_with_earlier_runs_counts_at_half_their_peak_or_what_it_holds(make_daemon, tmp_path, monkeypatch):
    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=100)
    j = job(d, a)
    j.history_mem = 8 * GiB
    d.store.update_job(j)
    now = d.clock()
    assert d._planned_mem(job(d, a), now) == 4 * GiB  # it holds 2G
    d._recent_held[a] = deque([(now, 5 * GiB)])
    assert d._planned_mem(job(d, a), now) == 5 * GiB
    go.touch()


def test_a_young_job_dibs_cant_see_counts_whole(make_daemon, tmp_path, monkeypatch):
    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=100)
    now = d.clock()
    d._recent_held.pop(a)
    assert d._planned_mem(job(d, a), now) == 12 * GiB
    j = job(d, a)
    j.history_mem = 8 * GiB
    d.store.update_job(j)
    assert d._planned_mem(job(d, a), now) == 8 * GiB
    go.touch()


def test_a_job_starts_beside_a_young_job_on_the_half_it_isnt_holding(make_daemon, tmp_path, monkeypatch):
    # A asked 12G of 16G 100 s ago and holds 2G: it counts at 6G, so B (8G) starts on borrowed RAM,
    # leaving the 1.5G reserve; it used to wait for A's first ten minutes to pass.
    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=100)
    assert job(d, b).state == State.RUNNING and job(d, b).borrowed_ram
    go.touch()


def test_a_young_job_that_grew_counts_at_its_raised_limit(make_daemon, tmp_path, monkeypatch):
    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=100)
    j = job(d, a)
    j.ram_limit = 14 * GiB  # it grew past its 12G within its first ten minutes
    d.store.update_job(j)
    assert d._planned_mem(job(d, a), d.clock()) == 14 * GiB
    go.touch()


def test_ram_outside_dibs_counts_at_its_peak_over_half_a_minute(make_daemon, tmp_path):
    from dibs.daemon import OUTSIDE_WINDOW

    assert OUTSIDE_WINDOW == 30.0
