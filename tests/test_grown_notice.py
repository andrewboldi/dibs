"""A job that grew well past its RAM request saves itself from a kill, but its agent isn't told it asked
for too little; the next run, on a fuller machine, dies. Its session is told once (#251)."""

import time

from conftest import run_until
from test_daemon import PY, finished, job, submit

from dibs.units import GiB

WAIT = "import os, time\nwhile not os.path.exists({!r}): time.sleep(0.02)"


def _advice(d, **kw):
    return [n["message"] for n in d.notices("s1", time.time(), **kw)["notices"] if n["kind"] == "advice"]


def _running(make_daemon, tmp_path, mem):
    d = make_daemon()
    go = tmp_path / "go"
    j = submit(d, tmp_path, [PY, "-c", WAIT.format(str(go))], mem=mem, session="s1", name="sweep")
    run_until(d, lambda: job(d, j).started)
    return d, j, go


def test_a_job_holding_over_twice_its_request_is_told_once_with_the_mem_to_ask(make_daemon, tmp_path):
    # 11-23: two jobs of a sweep asked 2.4G each and grew to hold 7.8G and 4.4G.
    d, j, go = _running(make_daemon, tmp_path, int(2.4 * GiB))
    d._held = {j: int(7.8 * GiB)}
    d._tell_grown(d.clock())
    (told,) = _advice(d, cwd=str(tmp_path))
    assert f"Job {j} (sweep) asked for 2.4G of RAM and holds 7.8G" in told
    assert "--mem 8G would size its next run" in told  # its peak, with no margin (#257)
    d._tell_grown(d.clock())
    assert _advice(d, cwd=str(tmp_path)) == []  # once
    go.touch()
    run_until(d, lambda: finished(d, j))


def test_a_job_holding_under_twice_its_request_isnt_told(make_daemon, tmp_path):
    d, j, go = _running(make_daemon, tmp_path, 2 * GiB)
    d._held = {j: int(3.9 * GiB)}
    d._tell_grown(d.clock())
    assert _advice(d, cwd=str(tmp_path)) == []
    go.touch()
    run_until(d, lambda: finished(d, j))
