"""Telling an agent, at its other dibs commands, when a request doesn't match how the same script ran
before (#169): told once, at most once an hour per script for each kind."""

import time

from conftest import capacity, run_until
from test_daemon import PY, finished, job, submit
from test_daemon import req as req_for

from dibs.model import Job, Request, State
from dibs.units import GiB, MiB


def _runs(d, tmp_path, script, n, *, request=None, ran=480.0, state=State.DONE, start=1_000_000.0, **fields):
    """n finished runs of `script` here, newest last."""
    req = {"cpus": 1, "mem": 5 * GiB, "gpus": 0, "vram": 0, "time": 600.0, **(request or {})}
    for i in range(n):
        t = start + i * 1000
        d.store.add_job(Job(id=0, name="run", argv=[PY, str(script), str(i)], cwd=str(tmp_path), env={},
                            request=Request(**req), session="s1", session_label="w·s1", submitted=t, tag=t + 600,
                            state=state, output="/dev/null", started=t, ended=t + ran, exit_code=0, **fields))  # fmt: skip


def _advice(d, now=None, **kw):
    import time

    return [n["message"] for n in d.notices("s1", now or time.time(), **kw)["notices"] if n["kind"] == "advice"]


def _script(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("pass\n")
    return script


def test_a_job_asking_no_more_ram_than_its_last_run_was_killed_at_is_told(make_daemon, tmp_path):
    # An export killed at 8G was resubmitted at 8G, then killed again (#169).
    d = make_daemon()
    script = _script(tmp_path)
    d.store.add_job(Job(id=0, name="run", argv=[PY, str(script)], cwd=str(tmp_path), env={}, session="s1",
                        request=Request(cpus=1, mem=8 * GiB, gpus=0, vram=0, time=600.0), session_label="w·s1",
                        submitted=1e6, tag=1e6, state=State.FAILED, output="/dev/null", started=1e6, ended=1e6 + 360,
                        exit_code=137, peak_mem=8 * GiB,
                        reason="killed: went over its 8G RAM reservation; resubmit with --mem 10G"))  # fmt: skip
    new = submit(d, tmp_path, [PY, str(script)], mem=8 * GiB, session="s1")
    assert _advice(d, at_submit=True) == []  # a loop's submit may be the one told: not there
    (told,) = _advice(d)
    assert f"Job {new} asks for 8G of RAM" in told and "killed at its 8G RAM limit" in told
    assert "--mem 10G" in told and f"dibs submit --replace {new}" in told
    assert _advice(d) == []  # told once
    run_until(d, lambda: finished(d, new))


def test_a_job_waiting_for_ram_far_above_what_its_scripts_runs_used_is_told(make_daemon, tmp_path):
    # A game script asked for 5G per run while its runs peaked at about 1G, and its jobs waited for RAM.
    d = make_daemon(cap=capacity(mem=8 * GiB))
    script = _script(tmp_path)
    _runs(d, tmp_path, script, 5, peak_mem=1 * GiB)
    go = tmp_path / "go"
    holder = submit(d, tmp_path, [PY, "-c", f"import os, time\nwhile not os.path.exists({str(go)!r}): time.sleep(0.02)"],
                    mem=5 * GiB, session="s0")  # fmt: skip
    waiting = submit(d, tmp_path, [PY, str(script), "9"], mem=5 * GiB, session="s1")
    assert job(d, waiting).state == State.QUEUED
    (told,) = _advice(d)
    assert f"Job {waiting} waits for 5G of RAM" in told and "the last 5 runs of train.py peaked at 1G at most" in told
    assert "--mem 1.5G would do" in told  # the peak plus a quarter (#212)
    go.touch()
    run_until(d, lambda: finished(d, holder) and finished(d, waiting))


def test_submit_carries_the_sizing_fact_when_earlier_runs_used_far_less(make_daemon, tmp_path):
    # Issue #212: the fact goes on submit's first line, the one line agents keep: 5+ runs, and the peak
    # plus a quarter at most three quarters of the request.
    d = make_daemon()
    script = _script(tmp_path)
    _runs(d, tmp_path, script, 5, peak_mem=1 * GiB)
    padded = d.submit(req_for(tmp_path, [PY, str(script), "9"], mem=2 * GiB))
    assert padded["job"]["sizing"] == {"runs": 5, "peak": 1 * GiB, "suggest": int(1.5 * GiB)}
    close = d.submit(req_for(tmp_path, [PY, str(script), "9"], mem=int(1.75 * GiB)))  # 1.5G isn't a quarter less
    assert close["job"].get("sizing") is None
    run_until(d, lambda: finished(d, padded["job"]["id"]) and finished(d, close["job"]["id"]))


def test_four_runs_are_too_few_for_the_sizing_fact(make_daemon, tmp_path):
    d = make_daemon()
    script = _script(tmp_path)
    _runs(d, tmp_path, script, 4, peak_mem=1 * GiB)
    new = d.submit(req_for(tmp_path, [PY, str(script), "9"], mem=4 * GiB))
    assert new["job"].get("sizing") is None
    run_until(d, lambda: finished(d, new["job"]["id"]))


def test_runs_of_the_same_script_in_the_sessions_other_folders_count(make_daemon, tmp_path):
    # Issue #229: a batch submitted from a fresh copy of the code is told what the copies before it used.
    d = make_daemon(cap=capacity(mem=8 * GiB))
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()
    _runs(d, tmp_path / "old", _script(tmp_path / "old"), 5, peak_mem=1 * GiB, start=time.time() - 86400)
    go = tmp_path / "go"
    holder = submit(d, tmp_path, [PY, "-c", f"import os, time\nwhile not os.path.exists({str(go)!r}): time.sleep(0.02)"],
                    mem=5 * GiB, session="s0")  # fmt: skip
    waiting = submit(d, tmp_path / "new", [PY, str(_script(tmp_path / "new")), "9"], mem=5 * GiB, session="s1")
    (told,) = _advice(d)
    assert f"Job {waiting} waits for 5G of RAM" in told and "the last 5 runs of train.py peaked at 1G at most" in told
    go.touch()
    run_until(d, lambda: finished(d, holder) and finished(d, waiting))


def test_gpu_memory_far_above_what_the_scripts_runs_held_is_told(make_daemon, tmp_path):
    d = make_daemon()
    script = _script(tmp_path)
    _runs(d, tmp_path, script, 5, request={"gpus": 1, "vram": 5 * GiB}, peak_vram=200 * MiB)
    new = submit(d, tmp_path, [PY, str(script), "9"], vram=5 * GiB, session="s1")
    (told,) = _advice(d)
    assert "The last 5 runs of train.py held at most 200M of GPU memory" in told and "--vram 1G would do" in told
    run_until(d, lambda: finished(d, new))


def test_gpu_memory_unused_by_the_scripts_runs_is_told_to_leave_out(make_daemon, tmp_path):
    d = make_daemon()
    script = _script(tmp_path)
    _runs(d, tmp_path, script, 5, request={"gpus": 1, "vram": 5 * GiB}, peak_vram=0)
    new = submit(d, tmp_path, [PY, str(script), "9"], vram=5 * GiB, session="s1")
    (told,) = _advice(d)
    assert "used no GPU memory" in told and "leave --vram out" in told
    run_until(d, lambda: finished(d, new))


def test_gpu_memory_below_what_the_scripts_runs_held_is_told(make_daemon, tmp_path):
    # A rendering script's request was cut to 256M; its runs held up to 759M.
    d = make_daemon()
    script = _script(tmp_path)
    _runs(d, tmp_path, script, 2, request={"gpus": 1, "vram": 1 * GiB}, peak_vram=759 * MiB)
    new = submit(d, tmp_path, [PY, str(script), "9"], vram=256 * MiB, session="s1")
    (told,) = _advice(d)
    assert f"Job {new} asks for 256M of GPU memory" in told and "the last 2 runs of train.py held up to 759M" in told
    run_until(d, lambda: finished(d, new))


def test_estimates_far_off_either_way_are_told_per_script(make_daemon, tmp_path):
    d = make_daemon()
    short, long = tmp_path / "short.py", tmp_path / "long.py"
    for s in (short, long):
        s.write_text("pass\n")
    _runs(d, tmp_path, short, 5, ran=30.0)  # 5% of their 10-minute estimates
    _runs(d, tmp_path, long, 5, ran=1500.0)  # 2.5 times
    a = submit(d, tmp_path, [PY, str(short), "9"], session="s1", time_=600.0)
    b = submit(d, tmp_path, [PY, str(long), "9"], session="s1", time_=600.0)
    told = " | ".join(_advice(d))
    assert "The last 5 runs of short.py took 30s, 30s, 30s, 30s, 30s, against 10m estimates" in told
    assert "The last 5 runs of long.py took 25m, 25m, 25m, 25m, 25m, over twice their 10m estimates" in told
    run_until(d, lambda: finished(d, a) and finished(d, b))


def test_a_loops_identical_submits_give_one_notice_an_hour(make_daemon, tmp_path):
    d = make_daemon()
    script = _script(tmp_path)
    _runs(d, tmp_path, script, 5, ran=30.0)
    ids = [submit(d, tmp_path, [PY, str(script), "9"], session="s1", time_=600.0) for _ in range(3)]
    assert len(_advice(d)) == 1
    run_until(d, lambda: all(finished(d, i) for i in ids))


def test_submit_asks_for_notices_as_a_submit(monkeypatch, tmp_path, capsys):
    import time

    from dibs import cli
    from dibs import client as dc

    monkeypatch.setenv("DIBS_HOME", str(tmp_path / "home"))
    asked = []
    job = {"id": 7, "name": "train", "state": "running", "request": {"cpus": 1, "mem": 1 << 30, "gpus": 0, "vram": 0, "time": 60.0}}

    def call(self, op, **fields):
        asked.append((op, fields.get("at_submit")))
        return {"ok": True, "now": time.time(), "job": job} if op == "submit" else {"ok": True, "notices": []}

    monkeypatch.setattr(dc.Client, "call", call)
    cli.main(["submit", "--cpus", "1", "--mem", "1G", "--time", "1m", "--", "true"])
    assert ("notices", True) in asked


def test_a_job_knows_the_longest_earlier_run_of_its_script_with_its_estimate(make_daemon, tmp_path):
    # Issue #227: what an overrunning job is expected to need, from runs like it: 3 or more of the same
    # script with the same estimate. Runs with another estimate are another kind of job.
    d = make_daemon()
    script = _script(tmp_path)
    recent = time.time() - 86400
    _runs(d, tmp_path, script, 2, ran=900.0, start=recent)
    _runs(d, tmp_path, script, 1, ran=1500.0, start=recent + 3000)
    _runs(d, tmp_path, script, 2, ran=3000.0, request={"time": 1200.0}, start=recent + 6000)
    new = submit(d, tmp_path, [PY, str(script), "9"], time_=600.0)
    assert job(d, new).history_longest == 1500.0
    fewer = submit(d, tmp_path, [PY, str(script), "9"], time_=1200.0)  # only 2 runs with this estimate
    assert job(d, fewer).history_longest is None
    run_until(d, lambda: finished(d, new) and finished(d, fewer))
