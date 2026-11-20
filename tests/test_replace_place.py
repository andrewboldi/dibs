"""A replacement that runs the same command and asks for no more keeps its place in line (#184)."""

from conftest import run_until
from test_daemon import PY, finished, job, req, submit

from dibs.units import GiB


def _line(make_daemon, tmp_path):
    """A job holding every CPU; then session a's job, then session b's, each waiting, a's ahead."""
    go = tmp_path / "go"
    holding = [PY, "-c", f"import os, time\nwhile not os.path.exists({str(go)!r}): time.sleep(0.02)"]
    d = make_daemon()
    blocker = submit(d, tmp_path, holding, cpus="all", session="s0")
    run_until(d, lambda: job(d, blocker).shim_pid)
    a = submit(d, tmp_path, [*holding, "a"], mem=8 * GiB, time_=600.0, session="sa")
    b = submit(d, tmp_path, [*holding, "b"], mem=8 * GiB, time_=600.0, session="sb")
    return d, go, a, b


def _replace(d, tmp_path, old, **kw):
    resp = d.submit(req(tmp_path, kw.pop("argv", job(d, old).argv), session=kw.pop("session", "sa"), replace=old, **kw))
    assert resp["ok"], resp
    return resp


def _end(d, go):
    go.touch()
    run_until(d, lambda: all(finished(d, j.id) for j in d.store.jobs()))


def test_a_replacement_asking_for_less_keeps_its_place_in_line(make_daemon, tmp_path):
    # #184: an agent kept two waiting jobs' oversized requests, since resubmitting would lose their place.
    d, go, a, b = _line(make_daemon, tmp_path)
    old_tag = job(d, a).tag
    resp = _replace(d, tmp_path, a, mem=4 * GiB, time_=600.0)
    new = resp["job"]["id"]
    assert job(d, new).tag == old_tag and resp.get("kept_place") is True
    assert d.show(new)["job"]["position"] < d.show(b)["job"]["position"]
    _end(d, go)


def test_a_replacement_asking_for_more_of_anything_gets_a_fresh_place(make_daemon, tmp_path):
    # Keeping the place for more would let a 1-minute placeholder hold one for a 10-hour job.
    d, go, a, b = _line(make_daemon, tmp_path)
    resp = _replace(d, tmp_path, a, mem=4 * GiB, time_=1200.0)
    new = resp["job"]["id"]
    assert not resp.get("kept_place")
    assert d.show(new)["job"]["position"] > d.show(b)["job"]["position"]
    _end(d, go)


def test_a_replacement_with_another_command_gets_a_fresh_place(make_daemon, tmp_path):
    d, go, a, b = _line(make_daemon, tmp_path)
    resp = _replace(d, tmp_path, a, argv=[*job(d, a).argv, "--fixed"], mem=4 * GiB, time_=600.0)
    assert not resp.get("kept_place")
    assert d.show(resp["job"]["id"])["job"]["position"] > d.show(b)["job"]["position"]
    _end(d, go)


def test_a_replacement_keeping_its_place_moves_no_job_back_again(make_daemon, tmp_path):
    # A job going ahead of its session's lower-priority jobs moves them back by its estimate (#91), once.
    d, go, a, b = _line(make_daemon, tmp_path)
    low = submit(d, tmp_path, [*job(d, a).argv, "low"], time_=600.0, session="sa", priority=3)
    tags = (job(d, low).tag, job(d, low).plain_tag)
    owed = d.store.get_meta("owed:sa")
    _replace(d, tmp_path, a, mem=4 * GiB, time_=600.0)
    assert (job(d, low).tag, job(d, low).plain_tag) == tags
    assert d.store.get_meta("owed:sa") == owed
    _end(d, go)


def test_the_help_and_agent_help_say_a_no_larger_replacement_keeps_its_place():
    import subprocess
    import sys

    from dibs.instructions import AGENT_HELP

    help_text = subprocess.run([sys.executable, "-m", "dibs", "submit", "--help"], capture_output=True, text=True).stdout
    assert "asks for no more keeps its place in line" in " ".join(help_text.split())
    assert "asks for no more of anything keeps the job's place in line" in " ".join(AGENT_HELP.split())
