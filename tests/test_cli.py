import json
import os
import re
import signal
import subprocess
import sys
import time

import pytest

PY = sys.executable
RES = ["--cpus", "1", "--mem", "1G", "--time", "1m"]


@pytest.fixture
def dibs(dibs_env, tmp_path):
    def run(*args, env=None, timeout=60, check=None):
        result = subprocess.run(
            [PY, "-m", "dibs", *args],
            env={**dibs_env, **(env or {})},
            capture_output=True,
            text=True,
            cwd=tmp_path,
            timeout=timeout,
        )
        if check is not None:
            assert result.returncode == check, (result.stdout, result.stderr)
        return result

    run.env = dibs_env
    yield run
    run("daemon", "stop")


def js(result):
    return json.loads(result.stdout)


def wait_state(dibs, job_id, states, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = dibs("show", str(job_id), "--json")
        if result.returncode == 0 and js(result)["state"] in states:
            return js(result)
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} never reached {states}")


def test_submit_prints_id_and_prediction(dibs):
    result = dibs("submit", *RES, "--", "sleep", "2", check=0)
    assert re.match(r"Job 1 \(sleep\) (started|queued)", result.stdout)
    assert "dibs wait 1" in result.stdout


def test_submit_json(dibs):
    dibs("submit", *RES, "--vram", "all", "--", "sleep", "30", check=0)
    job = js(dibs("submit", *RES, "--vram", "all", "--json", "--", "true", check=0))
    assert job["id"] == 2 and job["state"] == "queued"
    assert job["predicted_start"] > time.time() and job["position"] == 1
    assert "env" not in job
    dibs("cancel", "1", "2")


def test_submit_requires_cpus_mem_time(dibs):
    result = dibs("submit", "--cpus", "1", "--mem", "1G", "--", "true")
    assert result.returncode == 2 and "--time" in result.stderr


def test_submit_needs_double_dash(dibs):
    result = dibs("submit", *RES, "sleep", "1")
    assert result.returncode == 2 and "--" in result.stderr


def test_submit_rejects_unknown_command(dibs):
    result = dibs("submit", *RES, "--", "pyhton", "x.py")
    assert result.returncode == 2
    assert "command not found: pyhton" in result.stderr


def test_submit_rejects_unfit(dibs):
    result = dibs("submit", *RES, "--vram", "12G", "--", "true")
    assert result.returncode == 2
    assert "asks for 12G VRAM, but the largest GPU dibs manages has 8G" in result.stderr


def test_submit_rejects_bad_sizes(dibs):
    result = dibs("submit", "--cpus", "1", "--mem", "16", "--time", "1m", "--", "true")
    assert result.returncode == 2 and "unit" in result.stderr


def test_run_streams_output_and_exit_code(dibs):
    result = dibs("run", *RES, "--", PY, "-c", "print('hello'); raise SystemExit(5)")
    assert result.returncode == 5
    assert "hello" in result.stdout
    assert "Job 1" in result.stderr


def test_run_preserves_argv(dibs):
    args = ["--flag", "two words", 'quo"te', "", "üñí", "-h", "--help"]
    code = "import json, sys; print(json.dumps(sys.argv[1:]))"
    result = dibs("run", *RES, "--", PY, "-c", code, *args, check=0)
    assert json.loads(result.stdout.strip().splitlines()[-1]) == args


def test_queue_table_and_json(dibs):
    dibs("submit", *RES, "--vram", "all", "--name", "blocker", "--", "sleep", "30", check=0)
    dibs("submit", *RES, "--vram", "all", "--name", "waiter", "--", "true", check=0)
    table = dibs("queue", check=0).stdout
    header = table.splitlines()[0].split()
    assert header == ["JOB", "STATE", "SESSION", "NAME", "CPU", "RAM", "VRAM", "EST", "WHEN"]
    assert "blocker" in table and "waiter" in table and "running" in table and "queued" in table
    snap = js(dibs("queue", "--json", check=0))
    assert [j["name"] for j in snap["jobs"]] == ["blocker", "waiter"]
    dibs("cancel", "1", "2")


def test_show(dibs):
    dibs("submit", *RES, "--", "sleep", "30", check=0)
    text = dibs("show", "1", check=0).stdout
    assert text.startswith("Job 1") and "asks for" in text and "1 CPU" in text
    assert js(dibs("status", "1", "--json", check=0))["id"] == 1
    assert dibs("show", "99").returncode == 1
    dibs("cancel", "1")


def test_logs(dibs):
    dibs("run", *RES, "--", PY, "-c", "print('one'); print('two')", check=0)
    assert dibs("logs", "1", check=0).stdout == "one\ntwo\n"
    assert dibs("logs", "1", "-n", "1", check=0).stdout == "two\n"


def test_logs_follow(dibs):
    code = "import time; print('first', flush=True); time.sleep(1.5); print('second')"
    dibs("submit", *RES, "--", PY, "-c", code, check=0)
    result = dibs("logs", "1", "-f", timeout=30, check=0)
    assert result.stdout == "first\nsecond\n"


def test_wait_exit_code(dibs):
    dibs("submit", *RES, "--", PY, "-c", "raise SystemExit(3)", check=0)
    result = dibs("wait", "1")
    assert result.returncode == 3 and "exit 3" in result.stdout


def test_wait_timeout_124(dibs):
    dibs("submit", *RES, "--", "sleep", "30", check=0)
    result = dibs("wait", "1", "--timeout", "1s")
    assert result.returncode == 124 and "still running" in result.stderr
    dibs("cancel", "1")


def test_wait_cancelled_130(dibs):
    dibs("submit", *RES, "--", "sleep", "30", check=0)
    wait_state(dibs, 1, {"running"})
    dibs("cancel", "1", check=0)
    assert dibs("wait", "1").returncode == 130


def test_cancel(dibs):
    dibs("submit", *RES, "--", "sleep", "30", check=0)
    assert "Cancelled job 1" in dibs("cancel", "1", check=0).stdout
    result = dibs("cancel", "99")
    assert result.returncode == 1 and "no job 99" in result.stderr


def test_info_json(dibs):
    info = js(dibs("info", "--json", check=0))
    assert info["capacity"]["gpus"][0]["vram"] == 8 << 30
    assert set(info["enforcement"]) == {"cpu_pinning", "ram_cap", "vram_watch", "guard"}
    assert "Fake GPU" in dibs("info", check=0).stdout


def test_autostart_and_stop(dibs):
    from dibs.paths import get_paths

    paths = get_paths(dibs.env)
    assert not paths.socket.exists()
    dibs("queue", check=0)
    assert paths.socket.exists()
    dibs("submit", *RES, "--", "sleep", "3", check=0)
    assert "stopped" in dibs("daemon", "stop", check=0).stdout
    assert not paths.socket.exists()
    assert dibs("daemon", "status").returncode == 3
    dibs("daemon", "start", check=0)
    assert dibs("wait", "1", "--timeout", "30s").returncode == 0


def test_job_survives_submitter_death(dibs):
    script = f"'{PY}' -m dibs submit --cpus 1 --mem 1G --time 1m -- sleep 3; sleep 60"
    shell = subprocess.Popen(["bash", "-c", script], env=dibs.env, start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait_state(dibs, 1, {"running"})
    os.killpg(shell.pid, signal.SIGKILL)
    shell.wait()
    assert dibs("wait", "1", "--timeout", "30s").returncode == 0
    assert dibs("daemon", "status").returncode == 0


def test_build_mismatch_restarts_daemon(dibs):
    dibs("daemon", "start", env={"DIBS_BUILD_OVERRIDE": "old"}, check=0)
    old = dibs("daemon", "status", env={"DIBS_BUILD_OVERRIDE": "old"}, check=0).stdout
    assert "build old" in old
    dibs("queue", check=0)
    new = dibs("daemon", "status", check=0).stdout
    assert "build old" not in new
    assert re.search(r"pid (\d+)", old).group(1) != re.search(r"pid (\d+)", new).group(1)


def test_interrupted_run_keeps_job(dibs, tmp_path):
    run = subprocess.Popen([PY, "-m", "dibs", "run", *RES, "--", "sleep", "4"], env=dibs.env,
                           cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    wait_state(dibs, 1, {"running"})
    run.send_signal(signal.SIGINT)
    _, err = run.communicate(timeout=20)
    assert run.returncode == 130 and "keeps running" in err
    assert dibs("wait", "1", "--timeout", "30s").returncode == 0


def test_queue_into_closed_pipe(dibs):
    dibs("queue", check=0)
    result = subprocess.run(f"'{PY}' -m dibs queue --all | head -c1", shell=True, env=dibs.env,
                            capture_output=True, text=True)
    assert "Traceback" not in result.stderr


def test_a_client_that_hangs_up_before_its_answer_leaves_the_daemon_running(dibs):
    # Issue #82: the CLI resets SIGPIPE to its default for `dibs queue | head`, and `dibs daemon run`
    # kept it, so the daemon died without a trace when it answered a client that had already gone,
    # such as a `dibs wait` its agent's command timeout killed.
    import socket

    from dibs import build_id

    job_id = js(dibs("submit", *RES, "--json", "--", "sleep", "30", check=0))["id"]
    before = dibs("daemon", "status", check=0).stdout
    client = socket.socket(socket.AF_UNIX)
    client.connect(os.path.join(dibs.env["DIBS_HOME"], "dibs.sock"))
    request = {"op": "wait", "ids": [job_id], "timeout": 0.5, "build": build_id()}
    client.sendall(json.dumps(request).encode() + b"\n")
    client.close()  # gone before the answer comes
    time.sleep(1.5)
    assert dibs("daemon", "status").stdout == before  # still running, the same pid
    dibs("cancel", str(job_id), check=0)


def test_submit_replace(dibs):
    # Issue #93: cancel a waiting job and queue its replacement in one step.
    first = js(dibs("submit", *RES, "--vram", "all", "--json", "--", "sleep", "30", check=0))["id"]
    waiting = js(dibs("submit", *RES, "--vram", "all", "--json", "--", "sleep", "1", check=0))["id"]
    out = dibs("submit", *RES, "--vram", "all", "--replace", str(waiting), "--", "sleep", "1", check=0).stdout
    assert f"It replaces job {waiting}, which is cancelled." in out
    assert js(dibs("show", str(waiting), "--json"))["state"] == "cancelled"
    assert dibs("submit", *RES, "--replace", "99999", "--", "sleep", "1").returncode != 0
    dibs("cancel", str(first), check=0)


def test_submit_no_retry(dibs):
    # Issue #84: retries are on unless a job opts out.
    assert js(dibs("submit", *RES, "--json", "--", "true", check=0))["retry"] is True
    assert js(dibs("submit", *RES, "--no-retry", "--json", "--", "true", check=0))["retry"] is False


def test_session_from_claude_env(dibs):
    env = {"CLAUDE_CODE_SESSION_ID": "abcd1234-5678-90ab-cdef-1234567890ab"}
    job = js(dibs("submit", *RES, "--json", "--", "true", env=env, check=0))
    assert job["session"] == env["CLAUDE_CODE_SESSION_ID"]
    assert job["session_label"].endswith("·abcd")


def test_mine_filters_session(dibs):
    dibs("submit", *RES, "--", "sleep", "30", env={"DIBS_SESSION": "aaaa"}, check=0)
    dibs("submit", *RES, "--", "sleep", "30", env={"DIBS_SESSION": "bbbb"}, check=0)
    mine = js(dibs("queue", "--mine", "--json", env={"DIBS_SESSION": "aaaa"}, check=0))
    assert [j["session"] for j in mine["jobs"]] == ["aaaa"]
    dibs("cancel", "1", "2")


def test_submit_shows_earlier_runs(dibs):
    dibs("run", *RES, "--", "true", check=0)
    out = dibs("submit", *RES, "--", "true", check=0).stdout
    assert "Earlier runs of this command here:" in out and "#1 done" in out


def test_version_flag_prints_the_version(dibs):
    from dibs import __version__

    result = dibs("--version", check=0)
    assert result.stdout.strip() == f"dibs {__version__}"


def test_priority_flag(dibs):
    assert dibs("submit", *RES, "--priority", "11", "--", "true").returncode == 2
    dibs("submit", *RES, "--priority", "8", "--", "sleep", "30", check=0)
    shown = dibs("show", "1", check=0).stdout
    assert "priority" in shown and "8" in shown
    assert "(p8)" in dibs("queue", check=0).stdout
    dibs("cancel", "1")


UPGRADED = ("dibs was upgraded while this command ran: the daemon runs 0.1.0+new, this command still runs "
            "the code it started with (0.1.0+old). Run the command again")


def _upgraded_under(monkeypatch, tmp_path):
    """Every request finds dibs upgraded under this command; execve is recorded, not done."""
    from dibs import cli
    from dibs import client as dc

    monkeypatch.setenv("DIBS_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("DIBS_RERUN", raising=False)

    def call(self, op, **fields):
        raise dc.DibsError(UPGRADED, "upgraded")

    monkeypatch.setattr(dc.Client, "call", call)
    execs = []

    def execve(path, args, env):
        execs.append((path, args, env.get("DIBS_RERUN")))
        raise SystemExit(0)

    monkeypatch.setattr(cli.os, "execve", execve)
    return cli, execs


@pytest.mark.parametrize("argv", [["wait", "5"], ["queue"], ["show", "5"], ["info"], ["top"]])
def test_a_repeatable_command_runs_again_on_the_upgraded_code(argv, monkeypatch, tmp_path):
    # Issue #68: a dibs wait that found dibs upgraded under it exited at once, before its job ended,
    # so an agent's `dibs wait N; next step` ran the next step early.
    cli, execs = _upgraded_under(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        cli.main(argv)
    assert execs == [(PY, [PY, sys.argv[0], *argv], "1")]  # the same dibs, started the same way


@pytest.mark.parametrize(
    "argv",
    [["cancel", "5"], ["submit", *RES, "--", "true"], ["run", *RES, "--", "true"], ["logs", "5", "-f"]],
)
def test_a_command_that_would_do_something_twice_isnt_run_again(argv, monkeypatch, tmp_path, capsys):
    cli, execs = _upgraded_under(monkeypatch, tmp_path)
    assert cli.main(argv) == 1
    assert execs == []
    assert "dibs was upgraded while this command ran" in capsys.readouterr().err


def test_a_run_the_upgrade_reaches_after_its_submit_names_the_running_job(monkeypatch, tmp_path, capsys):
    # Issue #70: `dibs run` had already submitted its job when the upgrade reached its follow, and
    # "Run the command again" would have an agent submit the job a second time.
    from dibs import client as dc

    cli, execs = _upgraded_under(monkeypatch, tmp_path)
    job = {"id": 7, "name": "train", "state": "running", "output": str(tmp_path / "7.out")}

    def call(self, op, **fields):
        if op == "submit":
            return {"ok": True, "now": 0.0, "job": job}
        raise dc.DibsError(UPGRADED, "upgraded")

    monkeypatch.setattr(dc.Client, "call", call)
    assert cli.main(["run", *RES, "--", "true"]) == 1
    err = capsys.readouterr().err
    assert execs == [] and "Run the command again" not in err
    assert err.rstrip().splitlines()[-1] == (
        "dibs: dibs was upgraded while this command ran, and job 7 keeps running. "
        "Wait for it: dibs wait 7   Follow it: dibs logs 7 -f"
    )


def test_a_command_runs_again_only_once(monkeypatch, tmp_path, capsys):
    cli, execs = _upgraded_under(monkeypatch, tmp_path)
    monkeypatch.setenv("DIBS_RERUN", "1")
    assert cli.main(["wait", "5"]) == 1
    assert execs == []


def test_a_wait_dibs_was_upgraded_under_runs_again_until_its_job_ends(dibs, tmp_path):
    # Issue #68 end to end: a dibs wait running code older than the files on disk runs itself again
    # on the new code, leaves the daemon alone, and returns when the job ends, with its exit code.
    dibs("submit", *RES, "--", "sh", "-c", "sleep 2; exit 3", check=0)
    before = dibs("daemon", "status", check=0).stdout
    started_before_upgrade = tmp_path / "old_dibs.py"
    started_before_upgrade.write_text(
        "import os, sys\n"
        "import dibs\n"
        "if not os.environ.get('DIBS_RERUN'):\n"
        "    dibs._build = '0.1.0+startedwith'  # the code this command started with\n"
        "from dibs.cli import main\n"
        "sys.exit(main())\n"
    )
    result = subprocess.run([PY, str(started_before_upgrade), "wait", "1"], env=dibs.env, cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)  # fmt: skip
    assert result.returncode == 3, (result.stdout, result.stderr)
    after = dibs("daemon", "status", check=0).stdout
    assert re.search(r"pid (\d+)", before).group(1) == re.search(r"pid (\d+)", after).group(1)
