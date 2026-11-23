"""While the machine has RAM free, a job's hard RAM limit sits one growth step above its request, with
the plan still counting the request: a fast jump past the request is caught by dibs's next look, not
killed by the kernel (#249)."""

from conftest import capacity, run_until
from test_daemon import PY, finished, job, submit

from dibs import launcher
from dibs.daemon import _ram_kill_reason
from dibs.machine import Config
from dibs.units import GiB

WAIT = "import os, time\nwhile not os.path.exists({!r}): time.sleep(0.02)"


def _daemon(make_daemon, monkeypatch, *, used, borrow=True):
    """A 16G machine with `used` of it in use, and the hard limits dibs launches jobs with."""
    limits = {}
    real = launcher.launch

    def launch(j, paths, **kw):
        limits[j.id] = kw.get("mem_limit")
        return real(j, paths, **kw)

    monkeypatch.setattr(launcher, "launch", launch)
    d = make_daemon(config=Config(borrow_ram=borrow), cap=capacity(mem=16 * GiB), meminfo=lambda: (16 * GiB, used))
    return d, limits


def test_a_job_gets_a_step_of_headroom_while_the_machine_has_room(make_daemon, tmp_path, monkeypatch):
    d, limits = _daemon(make_daemon, monkeypatch, used=4 * GiB)
    go = tmp_path / "go"
    a = submit(d, tmp_path, [PY, "-c", WAIT.format(str(go))], mem=2 * GiB)
    assert limits[a] == 3 * GiB and job(d, a).hard_limit == 3 * GiB  # 2G and a 1G step
    assert job(d, a).ram_limit is None  # the plan still counts 2G
    go.touch()
    run_until(d, lambda: finished(d, a))


def test_no_headroom_without_room_or_with_borrowing_off(make_daemon, tmp_path, monkeypatch):
    for used, borrow in ((14 * GiB, True), (4 * GiB, False)):  # 2G available; or borrowing switched off
        d, limits = _daemon(make_daemon, monkeypatch, used=used, borrow=borrow)
        a = submit(d, tmp_path, [PY, "-c", "pass"], mem=2 * GiB)
        assert limits[a] is None and job(d, a).hard_limit is None
        run_until(d, lambda: finished(d, a))
        d.close()


def test_headroom_granted_in_one_pass_doesnt_compound(make_daemon, tmp_path, monkeypatch):
    # 6G available: the first 2G job and its 1G step leave 3G; the second's would leave 0.
    d, limits = _daemon(make_daemon, monkeypatch, used=10 * GiB)
    go = tmp_path / "go"
    hold = [PY, "-c", "import time; time.sleep(30)"]
    blocker = submit(d, tmp_path, [PY, "-c", WAIT.format(str(go))], cpus=len(d.capacity.cpus), mem=GiB // 2)
    first = submit(d, tmp_path, hold, mem=2 * GiB)
    second = submit(d, tmp_path, [*hold, "x"], mem=2 * GiB)
    assert limits[blocker] == int(1.5 * GiB)
    go.touch()
    run_until(d, lambda: job(d, first).started and job(d, second).started)
    given = sorted(limits[j] or 0 for j in (first, second))
    assert given == [0, 3 * GiB]
    d.cancel([first, second])
    run_until(d, lambda: finished(d, first) and finished(d, second))


def test_a_job_using_its_headroom_is_recorded_as_grown_even_without_room(make_daemon, tmp_path, monkeypatch):
    raised = []
    monkeypatch.setattr(launcher, "raise_oom_score", lambda j: raised.append(j.id))
    d, limits = _daemon(make_daemon, monkeypatch, used=4 * GiB)
    go = tmp_path / "go"
    a = submit(d, tmp_path, [PY, "-c", WAIT.format(str(go))], mem=2 * GiB)
    d.set_mem_limit = lambda j, limit: False  # and raising the limit fails: the kernel already let it
    d._meminfo = lambda: (16 * GiB, int(15.5 * GiB))  # no room for growth now: not even unreserved RAM
    d._outside_mem = lambda now, running: int(13.5 * GiB)
    d._held = {a: int(2.5 * GiB)}
    d._grow(d.clock())
    grown = job(d, a)
    assert grown.ram_limit == 3 * GiB and grown.grew_borrowed and raised == [a]
    go.touch()
    run_until(d, lambda: finished(d, a))


def test_a_kill_at_the_headroom_limit_names_it():
    from dibs.model import Job, Request, State

    j = Job(id=1, name="x", argv=["x"], cwd="/w", env={}, session="s", session_label="w·s",
            request=Request(cpus=1, mem=2 * GiB, gpus=0, vram=0, time=60.0), submitted=0.0, tag=0.0,
            state=State.RUNNING, output="/dev/null", hard_limit=3 * GiB)  # fmt: skip
    assert _ram_kill_reason(j, {"hit_limit": True}).startswith(
        "killed: went over its RAM limit of 3G (2G reserved, the rest room dibs gave it while the machine had RAM free)"
    )


def test_the_help_says_a_job_has_a_step_of_room_past_its_mem_while_ram_is_free():
    from dibs.instructions import AGENT_HELP

    text = " ".join(AGENT_HELP.split())
    assert "While the machine has RAM free, a job's limit sits a step (a quarter of --mem, at least 1G) above it" in text
