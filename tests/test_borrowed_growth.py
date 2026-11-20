"""A job at its RAM limit grows into RAM other jobs reserved but aren't using, and gives it back first
when the machine runs short (#182)."""

from conftest import run_until
from test_daemon import PY, finished, job, submit

from dibs.machine import Config
from dibs.units import GiB


def _two_jobs(make_daemon, tmp_path, *, borrow=True, retry=False):
    """On a 16G machine with 1G in use outside dibs: job a (8G) near its limit, and job c, which reserved
    6G and is planned by its use at 1.5G. Nothing is unreserved; 4.5G is reserved but unused."""
    limits = []
    d = make_daemon(config=Config(borrow_ram=borrow), meminfo=lambda: (16 * GiB, 1 * GiB),
                    set_mem_limit=lambda j, limit: limits.append((j.id, limit)) or True)  # fmt: skip
    go = tmp_path / "go"
    holding = [PY, "-c", f"import os, time\nwhile not os.path.exists({str(go)!r}): time.sleep(0.02)"]
    a = submit(d, tmp_path, holding, mem=8 * GiB, retry=retry)
    c = submit(d, tmp_path, [*holding, "c"], mem=6 * GiB)
    run_until(d, lambda: job(d, a).shim_pid and job(d, c).shim_pid)
    d._outside_mem = lambda now, running: 1 * GiB
    d._planned_mem = lambda j, now: int(1.5 * GiB) if j.id == c else j.request.mem
    d._held = {a: int(7.2 * GiB)}
    return d, a, c, limits, go


def _end(d, go, *ids):
    go.touch()
    run_until(d, lambda: all(finished(d, i) for i in ids))


def test_a_job_at_its_limit_grows_into_ram_another_job_reserved_but_isnt_using(make_daemon, tmp_path):
    # An export killed at its 8G limit after 48 of its 50 minutes, while other jobs held 5.7G reserved and
    # unused, and dibs's own plan by use had 2.2G free: one growth step (#182).
    d, a, c, limits, go = _two_jobs(make_daemon, tmp_path)
    d._grow(d.clock())
    assert limits == [(a, 10 * GiB)] and job(d, a).grew_borrowed
    _end(d, go, a, c)


def test_no_growth_into_reserved_ram_when_borrowing_is_off(make_daemon, tmp_path):
    d, a, c, limits, go = _two_jobs(make_daemon, tmp_path, borrow=False)
    d._grow(d.clock())
    assert limits == []
    _end(d, go, a, c)


def test_a_job_that_may_not_borrow_doesnt_grow_into_reserved_ram(make_daemon, tmp_path):
    d, a, c, limits, go = _two_jobs(make_daemon, tmp_path)
    j = job(d, a)
    j.no_borrow = True  # it was stopped for borrowing before
    d.store.update_job(j)
    d._grow(d.clock())
    assert limits == []
    _end(d, go, a, c)


def test_growth_into_reserved_ram_is_given_back_first_when_the_machine_runs_short(make_daemon, tmp_path):
    # dibs lent that RAM: it takes it back before stopping a program of an agent's outside dibs.
    d, a, c, limits, go = _two_jobs(make_daemon, tmp_path)
    d._grow(d.clock())
    hogs = []
    d._stop_ram_hog = lambda now: hogs.append(now) or True
    assert d._give_back_growth(d.clock()) is True and hogs == []
    stopped = job(d, a)
    assert stopped.evicted and "grown into" in stopped.stop_reason
    _end(d, go, a, c)


def test_a_retry_job_stopped_to_give_back_growth_runs_again_with_the_ram_it_needed(make_daemon, tmp_path):
    d, a, c, limits, go = _two_jobs(make_daemon, tmp_path, retry=True)
    d._grow(d.clock())
    j = job(d, a)
    j.evicted, j.peak_mem = True, 10 * GiB
    request, how, _ = d._retry_with(j, {"exit_code": 143})
    assert request.mem == 13 * GiB and "without borrowing" in how and j.no_borrow
    _end(d, go, a, c)


def test_the_agent_help_says_a_job_may_grow_into_ram_others_reserved_and_is_stopped_first():
    from dibs.instructions import AGENT_HELP

    text = " ".join(AGENT_HELP.split())
    assert "into RAM other jobs reserved but aren't using" in text and "the first one stopped" in text
