"""A program a job starts can leave the job's scope: snapd moves a snap app (Chromium, chromedriver)
into a scope of its own. dibs counts such a scope as the job's, since its processes descend from the
job's shim, rather than as RAM in use outside dibs, and tells the session once that the job's --mem
doesn't limit it (#265)."""

import time

from conftest import run_until
from test_daemon import PY, _job_holding, job, submit

from dibs import procs
from dibs.model import State
from dibs.units import GiB

DRIVER, USERS = 4100001, 4100003  # a chromedriver the job started, and the user's own browser


def _scope(root, name, *, ram, pids):
    scope = root / "cgroup" / "app.slice" / name
    scope.mkdir(parents=True)
    (scope / "memory.stat").write_text(f"anon {ram}\nshmem 0\n")
    (scope / "memory.swap.current").write_text("0\n")
    (scope / "cgroup.procs").write_text("".join(f"{p}\n" for p in pids))
    return scope


def _with_a_snap_browser(make_daemon, tmp_path, monkeypatch, *, own=GiB, used=7 * GiB, mem=4 * GiB):
    """Job A (`mem`) holds `own` in its scope; the chromedriver it started holds 3G in a snap scope;
    the user's own browser 2G in another. Clock at 700 s, past A's first ten minutes."""
    now = [time.time()]
    d = make_daemon(clock=lambda: now[0], meminfo=lambda: (16 * GiB, used))
    a, stat, go = _job_holding(d, tmp_path, monkeypatch, mem)
    stat.write_text(f"anon {own}\nshmem 0\n")
    shim = job(d, a).shim_pid
    _scope(tmp_path, "snap.chromium.chromedriver-9f3c.scope", ram=3 * GiB, pids=[DRIVER, DRIVER + 1])
    _scope(tmp_path, "snap.chromium.chromium-77aa.scope", ram=2 * GiB, pids=[USERS])
    where = {DRIVER: "/app.slice/snap.chromium.chromedriver-9f3c.scope", DRIVER + 1: "/app.slice/snap.chromium.chromedriver-9f3c.scope",
             USERS: "/app.slice/snap.chromium.chromium-77aa.scope"}  # fmt: skip
    parents = {DRIVER: shim, DRIVER + 1: DRIVER, USERS: 1}
    real_cgroup, real_parent = procs.cgroup, procs.parent
    monkeypatch.setattr(procs, "cgroup", lambda pid: where.get(pid) or real_cgroup(pid))
    monkeypatch.setattr(procs, "parent", lambda pid: parents.get(pid) or real_parent(pid))
    now[0] += 700
    d.tick()
    return d, a, now, go


def test_a_snap_app_a_job_started_counts_as_the_jobs_ram_not_outside(make_daemon, tmp_path, monkeypatch):
    d, a, now, go = _with_a_snap_browser(make_daemon, tmp_path, monkeypatch)
    assert d._escaped == {a: 3 * GiB}  # the user's own browser isn't the job's
    # 7G in use: the job's 1G and its browser's 3G are dibs's; the user's 2G browser and 1G else aren't.
    assert d._outside_mem(now[0] + 2, d.store.jobs([State.RUNNING])) == 3 * GiB
    go.touch()


def test_a_jobs_programs_outside_its_scope_count_on_top_of_its_limit(make_daemon, tmp_path, monkeypatch):
    # A (4G) holds all 4G in its scope, and its browser 3G more that its limit doesn't hold: with 1G of
    # other programs, 8G of 16G is free. A 9G job waits; it would start if the 3G counted nowhere.
    d, a, now, go = _with_a_snap_browser(make_daemon, tmp_path, monkeypatch, own=4 * GiB, used=10 * GiB)
    d._outside_mem = lambda now, running: 1 * GiB  # the user's browser and the rest, taken as settled
    big = submit(d, tmp_path, [PY, "-c", "pass"], mem=9 * GiB)
    d.tick()
    assert job(d, big).state == State.QUEUED
    fits = submit(d, tmp_path, [PY, "-c", "pass"], mem=7 * GiB)
    d.tick()
    assert job(d, fits).state != State.QUEUED
    go.touch()
    d.cancel([big])


def test_the_session_is_told_once_its_job_runs_programs_its_limit_doesnt_hold(make_daemon, tmp_path, monkeypatch):
    d, a, now, go = _with_a_snap_browser(make_daemon, tmp_path, monkeypatch)
    for _ in range(3):
        now[0] += 15
        d.tick()
    told = [e["message"] for e in d.store.events(0.0) if e["kind"] == "advice" and e.get("job") == a]
    assert len(told) == 1
    assert "snap.chromium.chromedriver" in told[0] and "3G" in told[0] and "isn't a snap" in told[0]
    assert job(d, a).peak_escaped == 3 * GiB
    go.touch()


def test_dibs_show_says_what_ran_outside_the_jobs_scope(make_daemon, tmp_path, monkeypatch):
    from dibs.render import render_show

    d, a, now, go = _with_a_snap_browser(make_daemon, tmp_path, monkeypatch)
    shown = render_show(d.show(a)["job"], now[0])
    assert "plus 3G in programs it started that left its scope (its --mem doesn't hold them)" in shown
    go.touch()


def test_a_scope_whose_processes_have_ended_counts_for_nothing(make_daemon, tmp_path, monkeypatch):
    d, a, now, go = _with_a_snap_browser(make_daemon, tmp_path, monkeypatch)
    (tmp_path / "cgroup" / "app.slice" / "snap.chromium.chromedriver-9f3c.scope" / "cgroup.procs").write_text("")
    now[0] += 2  # before dibs looks for such scopes again: the one it knows has emptied
    d.tick()
    assert d._escaped == {}
    go.touch()
    run_until(d, lambda: job(d, a).state == State.DONE)
