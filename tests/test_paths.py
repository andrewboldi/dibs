import os
import socket
import stat
from pathlib import Path

from dibs.paths import ensure_dirs, get_paths


def mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def test_default_paths_follow_xdg():
    p = get_paths(
        {
            "HOME": "/home/u",
            "XDG_CONFIG_HOME": "/cfg",
            "XDG_STATE_HOME": "/st",
            "XDG_RUNTIME_DIR": "/run/user/1000",
        }
    )
    assert not p.home_mode
    assert p.config_file == Path("/cfg/dibs/config.toml")
    assert p.state_dir == Path("/st/dibs")
    assert p.db == Path("/st/dibs/dibs.db")
    assert p.socket == Path("/run/user/1000/dibs/dibs.sock")


def test_default_paths_without_xdg():
    p = get_paths({"HOME": "/home/u"})
    assert p.config_file == Path("/home/u/.config/dibs/config.toml")
    assert p.state_dir == Path("/home/u/.local/state/dibs")
    assert p.socket == Path("/home/u/.local/state/dibs/dibs.sock")


def test_dibs_home_puts_everything_under_it(tmp_path):
    home = tmp_path / "h"
    p = get_paths({"DIBS_HOME": str(home), "HOME": "/home/u", "XDG_RUNTIME_DIR": "/run/user/1000"})
    assert p.home_mode
    assert p.config_file == home / "config.toml"
    assert p.state_dir == home / "state"
    assert p.db == home / "state" / "dibs.db"
    assert p.jobs_dir == home / "state" / "jobs"
    assert p.daemon_log == home / "state" / "daemon.log"
    assert p.lock == home / "state" / "daemon.lock"
    assert p.socket == home / "dibs.sock"
    assert p.job_dir(5) == home / "state" / "jobs" / "5"


def test_long_home_uses_short_socket(tmp_path):
    p = get_paths({"DIBS_HOME": str(tmp_path / ("x" * 120)), "HOME": "/home/u"})
    assert len(str(p.socket)) <= 100
    ensure_dirs(p)
    assert mode(p.socket.parent) == 0o700
    s = socket.socket(socket.AF_UNIX)
    try:
        s.bind(str(p.socket))
    finally:
        s.close()
        p.socket.unlink()


def test_instance_differs_per_home(tmp_path):
    a = get_paths({"DIBS_HOME": str(tmp_path / "a"), "HOME": "/home/u"})
    b = get_paths({"DIBS_HOME": str(tmp_path / "b"), "HOME": "/home/u"})
    a2 = get_paths({"DIBS_HOME": str(tmp_path / "a"), "HOME": "/home/u"})
    assert a.instance != b.instance
    assert a.instance == a2.instance
    assert len(a.instance) == 8 and int(a.instance, 16) >= 0


def test_ensure_dirs_private(tmp_path):
    p = get_paths({"DIBS_HOME": str(tmp_path / "h"), "HOME": "/home/u"})
    ensure_dirs(p)
    for d in (p.state_dir, p.jobs_dir, p.socket.parent):
        assert mode(d) == 0o700
