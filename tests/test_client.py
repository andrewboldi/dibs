import os

import pytest

from dibs.client import Client, DibsError, current_session, daemon_setenv, stop_daemon
from dibs.paths import get_paths


def test_claude_session_used_when_present(tmp_path):
    env = {"CLAUDE_CODE_SESSION_ID": "abcd1234-5678"}
    assert current_session(env, str(tmp_path)) == ("abcd1234-5678", f"{tmp_path.name}·abcd")


def test_explicit_dibs_session_beats_detected_one(tmp_path):
    env = {"CLAUDE_CODE_SESSION_ID": "abcd1234-5678", "DIBS_SESSION": "zzzz9999"}
    assert current_session(env, str(tmp_path))[0] == "zzzz9999"


def test_terminal_session_fallback(tmp_path):
    assert current_session({}, str(tmp_path))[0] == f"t{os.getsid(0)}"


def test_label_uses_git_root(tmp_path):
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    (tmp_path / "repo" / "src" / "deep").mkdir(parents=True)
    _, label = current_session({"DIBS_SESSION": "s123"}, str(tmp_path / "repo" / "src" / "deep"))
    assert label == "repo·s123"



def test_no_autostart_from_inside_a_job_without_systemd(tmp_path):
    # It would be pinned to the job's CPUs and stopped along with the job's leftovers.
    env = {"DIBS_HOME": str(tmp_path / "home"), "DIBS_NO_SYSTEMD": "1", "DIBS_JOB_ID": "7",
           "DIBS_GPU_BACKEND": "none", "PATH": os.environ["PATH"], "HOME": str(tmp_path)}  # fmt: skip
    paths = get_paths(env)
    try:
        with pytest.raises(DibsError) as err:
            Client(paths, env=env).call("queue")
        assert "inside a dibs job" in str(err.value)
        assert not paths.socket.exists()
    finally:
        stop_daemon(paths, env=env)


def test_daemon_started_by_systemd_gets_the_folders(tmp_path):
    env = {"PATH": "/bin", "DIBS_HOME": "/d", "XDG_CONFIG_HOME": "/c", "XDG_STATE_HOME": "/s",
           "XDG_RUNTIME_DIR": "/r", "HOME": "/h", "OTHER": "x"}  # fmt: skip
    assert sorted(daemon_setenv(env)) == sorted(
        f"--setenv={k}={v}" for k, v in env.items() if k != "OTHER"
    )


def test_stop_goes_through_systemctl_when_the_service_runs_dibs(tmp_path):
    (tmp_path / ".config" / "systemd" / "user").mkdir(parents=True)
    (tmp_path / ".config" / "systemd" / "user" / "dibs.service").write_text("[Service]\n")
    bin_dir, log = tmp_path / "bin", tmp_path / "systemctl.log"
    bin_dir.mkdir()
    (bin_dir / "systemctl").write_text(f'#!/bin/sh\necho "$@" >> {log}\nexit 0\n')
    (bin_dir / "systemctl").chmod(0o755)
    env = {"HOME": str(tmp_path), "XDG_RUNTIME_DIR": str(tmp_path / "run"), "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    assert stop_daemon(get_paths(env), env=env) is True
    assert "--user stop dibs.service" in log.read_text()
