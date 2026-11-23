"""Every dibs command says when a disk the jobs use runs low, and what used it lately (#211)."""

import json

from conftest import run_until
from test_daemon import PY, finished, job, submit

from dibs import snapshot
from dibs.render import top_screen
from dibs.units import GiB

SLEEP = [PY, "-c", "import time; time.sleep(30)"]


def _feed(d, samples):
    """Run the disk watch at each (time, free bytes, {job id: bytes written}) sample, in order."""
    for t, free, written in samples:
        d.disk_free = lambda path, free=free: free
        d._job_written = lambda j, written=written: written.get(j.id)
        d._watch_disk(t)


def _disk_lines(d, session, now):
    return [n["message"] for n in d.notices(session, now)["notices"] if n["kind"] == "disk"]


def test_nothing_is_said_while_the_disk_has_room(make_daemon, tmp_path):
    d = make_daemon()
    j = submit(d, tmp_path, SLEEP)
    _feed(d, [(1000.0, 60 * GiB, {j: 0}), (1600.0, 50 * GiB, {j: 9 * GiB})])
    assert _disk_lines(d, "s1", 1600.0) == []
    d.cancel([j])
    run_until(d, lambda: finished(d, j))


def test_every_command_of_every_session_is_told_and_the_job_that_wrote_it_named(make_daemon, tmp_path):
    d = make_daemon()
    j = submit(d, tmp_path, SLEEP, name="bench")
    _feed(d, [(1000.0, 9 * GiB, {j: 0}), (1300.0, 7 * GiB, {j: 2 * GiB}), (1600.0, 4 * GiB, {j: int(4.5 * GiB)})])
    line = f"the disk has 4G free, 5G less than 10 minutes ago: job {j} (bench) wrote 4.5G of it."
    assert _disk_lines(d, "s1", 1600.0) == [line]
    assert _disk_lines(d, "s2", 1600.0) == [line]  # whoever runs a dibs command
    assert _disk_lines(d, "s2", 1601.0) == [line]  # at each command, not once
    d.cancel([j])
    run_until(d, lambda: finished(d, j))


def test_a_drop_the_jobs_didnt_write_is_said_to_come_from_outside_dibs(make_daemon, tmp_path):
    # Issue #211's update: copies made outside dibs were 70% of a drop; naming the busiest job would
    # have blamed the wrong writer.
    d = make_daemon()
    j = submit(d, tmp_path, SLEEP, name="bench")
    _feed(d, [(1000.0, 9 * GiB, {j: 0}), (1600.0, 4 * GiB, {j: GiB})])
    assert _disk_lines(d, "s1", 1600.0) == ["the disk has 4G free, 5G less than 10 minutes ago, mostly written outside dibs."]
    d.cancel([j])
    run_until(d, lambda: finished(d, j))


def test_waiting_jobs_code_copies_are_counted_as_dibs_own_use(make_daemon, tmp_path):
    # 11-22: seven waiting jobs' --snapshot copies of 755M each took 5G in 12 minutes.
    d = make_daemon()
    blocker = submit(d, tmp_path, SLEEP, cpus=len(d.capacity.cpus))  # the copies' jobs wait behind it
    waiting = []
    for _ in range(2):
        w = submit(d, tmp_path, SLEEP, time_=3600.0, session="s3")
        folder = d.paths.job_dir(w) / "snapshot"
        folder.mkdir(parents=True)
        (folder / snapshot.MANIFEST).write_text(json.dumps({"root": str(tmp_path), "copy": str(folder),
                                                            "files": {"a.pkl": [2 * GiB, 0], "b.py": [GiB // 2, 0]}}))  # fmt: skip
        stored = job(d, w)
        stored.snapshot = str(tmp_path)
        d.store.update_job(stored)
        waiting.append(w)
    t0 = job(d, waiting[0]).submitted - 60
    _feed(d, [(t0, 9 * GiB, {}), (t0 + 600, 4 * GiB, {})])
    assert _disk_lines(d, "s1", t0 + 600) == [
        "the disk has 4G free, 5G less than 10 minutes ago: 2 jobs' code copies (--snapshot) took 5G of it."
    ]
    for j in (blocker, *waiting):
        d.cancel([j])
    run_until(d, lambda: all(finished(d, j) for j in (blocker, *waiting)))


def test_the_writers_session_is_told_once_an_hour(make_daemon, tmp_path):
    d = make_daemon()
    j = submit(d, tmp_path, SLEEP, name="bench")
    _feed(d, [(1000.0, 9 * GiB, {j: 0}), (1600.0, 4 * GiB, {j: int(4.5 * GiB)})])
    told = [n["message"] for n in d.notices("s1", 1600.0)["notices"] if n["kind"] == "disk_writer"]
    assert told == [f"Job {j} (bench) wrote 4.5G in the last 10 minutes, and the disk has 4G free."]
    assert [n for n in d.notices("s2", 1600.0)["notices"] if n["kind"] == "disk_writer"] == []  # not its job
    _feed(d, [(1630.0, int(3.8 * GiB), {j: int(4.7 * GiB)})])
    assert [n for n in d.notices("s1", 1631.0)["notices"] if n["kind"] == "disk_writer"] == []  # once an hour
    d.cancel([j])
    run_until(d, lambda: finished(d, j))


def test_dibs_top_shows_the_disk_line(make_daemon, tmp_path):
    d = make_daemon()
    j = submit(d, tmp_path, SLEEP, name="bench")
    _feed(d, [(1000.0, 9 * GiB, {j: 0}), (1600.0, 4 * GiB, {j: int(4.5 * GiB)})])
    snap = d.snapshot()
    assert snap["disk"].startswith("the disk has 4G free")
    screen, _, _ = top_screen(snap, width=160, height=60, color=False)
    assert "disk: the disk has 4G free" in screen
    d.cancel([j])
    run_until(d, lambda: finished(d, j))
