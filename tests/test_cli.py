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
