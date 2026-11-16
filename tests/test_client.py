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


def test_a_version_clash_a_restart_cant_fix_names_both_builds(tmp_path):
    # Issue #5: the service runs another install of dibs, so restarting it brings back the same
    # other build, and the second try used to fail with a bare "the dibs daemon refused".
    from dibs import build_id

    env = {**os.environ, "DIBS_HOME": str(tmp_path / "home"), "DIBS_NO_SYSTEMD": "1",
           "DIBS_GPU_BACKEND": "none", "DIBS_BUILD_OVERRIDE": "0.0.9+old"}  # fmt: skip
    env.pop("DIBS_JOB_ID", None)
    paths = get_paths(env)
    try:
        with pytest.raises(DibsError) as err:
            Client(paths, env=env).call("queue")
        assert "0.0.9+old" in str(err.value) and build_id() in str(err.value)
    finally:
        stop_daemon(paths, env=env)


def test_no_restart_when_the_service_runs_another_install(tmp_path):
    # Issue #39: a dibs from a development checkout restarted the installed service's daemon, which
    # came back as the same installed build: a pointless restart for every command.
    from dibs import client

    home = tmp_path / "home"
    unit = home / ".config" / "systemd" / "user" / "dibs.service"
    unit.parent.mkdir(parents=True)
    unit.write_text("[Service]\nExecStart=/opt/installed/bin/dibs daemon run\nRestart=always\n")
    env = {"HOME": str(home), "XDG_RUNTIME_DIR": str(tmp_path / "run")}
    paths = get_paths(env)
    assert not client._restart_brings_this_build(paths, env, own_exe="/checkout/.venv/bin/dibs")
    assert client._restart_brings_this_build(paths, env, own_exe="/opt/installed/bin/dibs")


def test_a_command_started_before_an_upgrade_doesnt_restart_the_upgraded_daemon(tmp_path, monkeypatch):
    # Issue #66: a `dibs wait` still running the old code when dibs was upgraded in place saw the new
    # daemon as another build and restarted it. It came back as the same new build, and the command
    # failed saying the service runs another install.
    import dibs
    from dibs import client

    monkeypatch.delenv("DIBS_BUILD_OVERRIDE", raising=False)
    upgraded = dibs.build_id()
    monkeypatch.setattr(dibs, "_build", "0.1.0+startedwith")  # the code the command started with
    paths = get_paths({"DIBS_HOME": str(tmp_path / "h"), "HOME": str(tmp_path)})
    reply = {"ok": False, "code": "build_mismatch", "build": upgraded}
    monkeypatch.setattr(client.Client, "_send", lambda self, payload, io_timeout: reply)
    stops = []
    monkeypatch.setattr(client, "stop_daemon", lambda *a, **k: stops.append(1) or True)
    with pytest.raises(DibsError) as err:
        client.Client(paths).call("queue")
    assert stops == []  # a restart would bring up the upgraded code again, not this command's
    assert str(err.value) == (
        f"dibs was upgraded while this command ran: the daemon runs {upgraded}, this command still "
        "runs the code it started with (0.1.0+startedwith). Run the command again"
    )


def test_a_daemon_another_command_already_restarted_isnt_restarted_again(tmp_path, monkeypatch):
    # After an upgrade, several agents' commands saw the old daemon at once and each restarted it,
    # the later ones restarting the already-upgraded daemon: two or three restarts per upgrade.
    from dibs import client

    paths = get_paths({"DIBS_HOME": str(tmp_path / "h"), "HOME": str(tmp_path)})
    replies = [{"ok": False, "code": "build_mismatch", "build": "old"}, {"ok": True}]
    monkeypatch.setattr(client.Client, "_send", lambda self, payload, io_timeout: replies.pop(0))
    stops = []
    monkeypatch.setattr(client, "stop_daemon", lambda *a, **k: stops.append(1) or True)
    assert client.Client(paths).call("queue")["ok"]
    assert stops == []  # the second look, under the restart lock, found the new build running
