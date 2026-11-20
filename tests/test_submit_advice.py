"""Submit's advice goes to stderr, after the job's lines, where agents' filters don't drop it (#169)."""

import time

import pytest


def _submit(monkeypatch, tmp_path, capsys, state):
    from dibs import cli
    from dibs import client as dc

    monkeypatch.setenv("DIBS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "agent-1")
    req = {"cpus": 1, "mem": 1 << 30, "gpus": 0, "vram": 0, "time": 600.0}
    job = {"id": 7, "name": "train", "state": state, "request": req, "position": 2, "predicted_start": time.time() + 600,
           "waiting_for": "needs 4G RAM (2G free)"}  # fmt: skip
    earlier = [{"id": 5, "state": "done", "argv": ["python", "train.py"], "request": req, "started": 0.0, "ended": 30.0,
                "cpu_seconds": 30.0, "peak_mem": 1 << 29}]  # fmt: skip

    def call(self, op, **fields):
        if op == "submit":
            return {"ok": True, "now": time.time(), "job": job, "earlier": earlier, "estimates": {"jobs": 9, "median": 0.1}}
        return {"ok": True, "notices": []}

    monkeypatch.setattr(dc.Client, "call", call)
    assert cli.main(["submit", "--cpus", "1", "--mem", "1G", "--time", "10m", "--", "python", "train.py"]) == 0
    return capsys.readouterr()


def test_submits_advice_goes_to_stderr_after_the_jobs_lines(monkeypatch, tmp_path, capsys):
    # 68% of submits kept only their first one to three lines: the advice at the end of stdout was lost.
    out, err = _submit(monkeypatch, tmp_path, capsys, "queued")
    assert out.startswith("Job 7 (train) queued") and "Earlier runs" not in out
    assert "Earlier runs of this command here:" in err and "#5 done after 30s" in err


@pytest.mark.parametrize("state, told", [("queued", True), ("running", False)])
def test_the_estimates_line_comes_only_when_the_job_waits(monkeypatch, tmp_path, capsys, state, told):
    # "A closer --time gets jobs started sooner" is false for a job that started at once.
    out, err = _submit(monkeypatch, tmp_path, capsys, state)
    assert ("a closer --time gets jobs started sooner" in err) is told and "closer --time" not in out
