"""A run that failed or was cancelled within seconds of starting says nothing of what its command needs:
its peak is about what the interpreter takes to start. It no longer sets a job's RAM history, which a
young job counts by (#266), or the sizing fact (#280)."""

import time

from conftest import run_until
from test_daemon import PY, finished, job
from test_daemon import req as req_for

from dibs.model import Job, Request, State
from dibs.units import GiB, MiB


def _ran(d, cwd, argv, *, state=State.DONE, ran=480.0, peak_mem=1 * GiB, ago=7200.0, **fields):
    """A finished run of `argv` in `cwd`, `ran` seconds long, started `ago` seconds ago."""
    t = time.time() - ago
    d.store.add_job(Job(id=0, name="run", argv=argv, cwd=str(cwd), env={}, session="s1", session_label="w·s1",
                        request=Request(cpus=1, mem=2 * GiB, gpus=0, vram=0, time=600.0), submitted=t, tag=t,
                        state=state, output="/dev/null", started=t, ended=t + ran,
                        exit_code=0 if state == State.DONE else 1, peak_mem=peak_mem, **fields))  # fmt: skip


def _history(d, cwd, argv):
    new = d.submit(req_for(cwd, argv, mem=2 * GiB))["job"]["id"]
    run_until(d, lambda: finished(d, new))
    return job(d, new).history_mem


def test_runs_that_ended_within_seconds_leave_the_history_to_the_scripts_real_runs(make_daemon, tmp_path):
    # A failure at startup and a cancel of one command, at 0.1G, set its history while runs of the same
    # script with other arguments had peaked at 983M: the young job counted at what it held.
    d = make_daemon()
    (tmp_path / "train.py").write_text("pass\n")
    _ran(d, tmp_path, [PY, "train.py", "--lr", "1"], peak_mem=983 * MiB)
    _ran(d, tmp_path, [PY, "train.py", "--lr", "2"], state=State.FAILED, ran=2.0, peak_mem=100 * MiB, ago=3600.0)
    _ran(d, tmp_path, [PY, "train.py", "--lr", "2"], state=State.CANCELLED, ran=9.0, peak_mem=110 * MiB, ago=1800.0)
    assert _history(d, tmp_path, [PY, "train.py", "--lr", "2"]) == 983 * MiB


def test_with_only_runs_that_ended_within_seconds_a_job_has_no_ram_history(make_daemon, tmp_path):
    d = make_daemon()
    _ran(d, tmp_path, [PY, "-c", "pass"], state=State.FAILED, ran=1.0, peak_mem=70 * MiB)
    assert _history(d, tmp_path, [PY, "-c", "pass"]) is None  # young, it counts at half its request


def test_runs_that_show_what_their_command_needs_still_count(make_daemon, tmp_path):
    d = make_daemon()
    _ran(d, tmp_path, [PY, "-c", "pass"], ran=3.0, peak_mem=80 * MiB)  # finished: its whole run
    _ran(d, tmp_path, [PY, "-c", "x = 1"], state=State.FAILED, ran=1625.0, peak_mem=1400 * MiB)  # failed late
    _ran(d, tmp_path, [PY, "-c", "x = 2"], state=State.FAILED, ran=20.0, peak_mem=2 * GiB,
         reason="killed: went over its 2G RAM reservation")  # stopped at its limit: it needed more  # fmt: skip
    assert _history(d, tmp_path, [PY, "-c", "pass"]) == 80 * MiB
    assert _history(d, tmp_path, [PY, "-c", "x = 1"]) == 1400 * MiB
    assert _history(d, tmp_path, [PY, "-c", "x = 2"]) == 2 * GiB


def test_runs_that_ended_within_seconds_dont_make_the_sizing_fact(make_daemon, tmp_path):
    # Five runs that died at startup would have told the agent --mem 256M would do.
    d = make_daemon()
    (tmp_path / "train.py").write_text("pass\n")
    for i in range(5):
        _ran(d, tmp_path, [PY, "train.py", str(i)], state=State.FAILED, ran=2.0, peak_mem=100 * MiB, ago=7200.0 - i * 600)
    new = d.submit(req_for(tmp_path, [PY, "train.py", "9"], mem=2 * GiB))
    assert new["job"].get("sizing") is None
    run_until(d, lambda: finished(d, new["job"]["id"]))
