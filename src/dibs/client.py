"""Talking to the daemon, and starting it when nobody has yet."""

import contextlib
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

from dibs import build_id, installed_build, launcher
from dibs.paths import Paths, ensure_dirs

START_TIMEOUT = 15.0
WAIT_CHUNK = 60.0  # a long wait is made of requests this long, so a vanished daemon is noticed


class DibsError(Exception):
    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.code = code


def _request(sock: Path, payload: dict, io_timeout: float | None) -> dict:
    with socket.socket(socket.AF_UNIX) as s:
        s.settimeout(io_timeout)
        s.connect(str(sock))
        s.sendall(json.dumps(payload).encode() + b"\n")
        buf = bytearray()
        while not buf.endswith(b"\n"):
            chunk = s.recv(1 << 16)
            if not chunk:
                raise ConnectionResetError("the dibs daemon closed the connection")
            buf += chunk
    return json.loads(buf)


def ping(paths: Paths) -> dict | None:
    """The running daemon's {build, pid}, or None if none answers."""
    try:
        return _request(paths.socket, {"op": "ping"}, 5)
    except (OSError, ValueError):
        return None


def _daemon_gone(paths: Paths) -> bool:
    try:
        fd = os.open(paths.lock, os.O_RDWR)
    except FileNotFoundError:
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    finally:
        os.close(fd)  # closing releases the lock if we got it
    return True


_PASSED_ON = ("PATH", "HOME", "XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_RUNTIME_DIR")


def daemon_setenv(env: Mapping[str, str]) -> list[str]:
    """systemd-run options that give a daemon started by systemd the caller's dibs settings and
    folders (a systemd unit starts with the user manager's environment, not ours)."""
    return [f"--setenv={k}={v}" for k, v in env.items() if k.startswith("DIBS_") or k in _PASSED_ON]


def _service_installed(paths: Paths, env: Mapping[str, str]) -> bool:
    """Whether `dibs setup` installed the systemd user service that runs this dibs."""
    unit = Path(env.get("HOME") or Path.home()) / ".config/systemd/user/dibs.service"
    return not paths.home_mode and unit.exists()


def _restart_brings_this_build(paths: Paths, env: Mapping[str, str], own_exe: str | None = None) -> bool:
    """Whether restarting the daemon can bring up this command's own code. Not when the systemd
    service runs another install of dibs (a development checkout's dibs against the installed
    service, say): the daemon would come back as the same other build. Nor when dibs was upgraded
    while this command ran: the daemon would come back as the new code, not this command's."""
    if installed_build() != build_id():
        return False
    if not _service_installed(paths, env):
        return True  # start_daemon starts this very interpreter
    unit = Path(env.get("HOME") or Path.home()) / ".config/systemd/user/dibs.service"
    for line in unit.read_text().splitlines():
        if line.startswith("ExecStart="):
            service_exe = line.split("=", 1)[1].split()[0]
            return os.path.realpath(service_exe) == os.path.realpath(own_exe or sys.argv[0])
    return True


@contextlib.contextmanager
def _restart_lock(paths: Paths):
    """One upgrade restart at a time: the commands that saw the old daemon queue up here."""
    ensure_dirs(paths)
    fd = os.open(paths.state_dir / "restart.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def start_daemon(paths: Paths, env: Mapping[str, str] = os.environ) -> None:
    """Start the daemon, fully detached from this process (it must outlive the caller)."""
    ensure_dirs(paths)
    command = [sys.executable, "-m", "dibs", "daemon", "run"]
    if launcher.systemd_available(env):
        if _service_installed(paths, env):
            subprocess.run(["systemctl", "--user", "start", "dibs.service"], capture_output=True, timeout=30)
            return
        unit = f"--unit=dibs-{paths.instance}-daemon"
        subprocess.run(
            ["systemd-run", "--user", "--quiet", "--collect", unit, *daemon_setenv(env), "--", *command],
            capture_output=True,
            timeout=30,
        )
        return
    if env.get("DIBS_JOB_ID"):
        # Started from here, it would run on the job's CPUs and be stopped with the job's leftovers.
        raise DibsError(
            "dibs isn't running, and it can't be started from inside a dibs job (it would stop when "
            "the job ends). Start it outside the job: dibs daemon start",
            "not_running",
        )
    subprocess.Popen(
        command,
        env=dict(env),
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_daemon(paths: Paths, timeout: float = 10.0, env: Mapping[str, str] = os.environ) -> bool:
    """Ask the daemon to stop and wait until it has. Running jobs keep running. False if none ran."""
    if _service_installed(paths, env):
        active = subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", "dibs.service"], capture_output=True, timeout=30, env=dict(env)
        )
        if active.returncode == 0:  # the service restarts a daemon that just exits, so stop the service
            subprocess.run(["systemctl", "--user", "stop", "dibs.service"], capture_output=True, timeout=60, env=dict(env))
            return True
    try:
        _request(paths.socket, {"op": "shutdown"}, 5)
    except (OSError, ValueError):
        return False
    deadline = time.monotonic() + timeout
    while not _daemon_gone(paths):
        if time.monotonic() > deadline:
            raise DibsError("the dibs daemon didn't stop in time")
        time.sleep(0.05)
    return True


class Client:
    def __init__(self, paths: Paths, *, autostart: bool = True, env: Mapping[str, str] = os.environ):
        self.paths = paths
        self.autostart = autostart
        self.env = env

    def _send(self, payload: dict, io_timeout: float | None) -> dict:
        try:
            return _request(self.paths.socket, payload, io_timeout)
        except (FileNotFoundError, ConnectionRefusedError):
            if not self.autostart:
                raise DibsError("dibs isn't running", "not_running") from None
        start_daemon(self.paths, self.env)
        deadline = time.monotonic() + START_TIMEOUT
        while True:
            try:
                return _request(self.paths.socket, payload, io_timeout)
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() > deadline:
                    raise DibsError(
                        f"the dibs daemon didn't start; see {self.paths.daemon_log}", "not_running"
                    ) from None
                time.sleep(0.05)

    def call(self, op: str, *, io_timeout: float | None = 60.0, **fields) -> dict:
        payload = {"op": op, "build": build_id(), **fields}
        response = self._send(payload, io_timeout)
        if response.get("code") == "build_mismatch" and _restart_brings_this_build(self.paths, self.env):
            # The daemon runs other code (dibs was upgraded): restarting it is safe, jobs keep running.
            # Other commands may have seen the old daemon too; restart once, under a lock, and only if
            # it still runs other code by then.
            with _restart_lock(self.paths):
                response = self._send(payload, io_timeout)
                if response.get("code") == "build_mismatch":
                    stop_daemon(self.paths)
                    response = self._send(payload, io_timeout)
        if response.get("code") == "build_mismatch" and installed_build() != build_id():
            raise DibsError(
                f"dibs was upgraded while this command ran: the daemon runs {response.get('build')}, "
                f"this command still runs the code it started with ({build_id()}). Run the command again",
                "build_mismatch",
            )
        if response.get("code") == "build_mismatch":
            # Still: what starts the daemon (the systemd service) runs another install of dibs.
            raise DibsError(
                f"this dibs ({build_id()}) and the dibs daemon ({response.get('build')}) are different "
                "versions, and the service that runs the daemon runs another install of dibs, so "
                "restarting it can't fix that. Use that install's dibs, or reinstall so they match, "
                "then run dibs daemon restart",
                "build_mismatch",
            )
        if not response.get("ok"):
            raise DibsError(response.get("error") or "the dibs daemon refused", response.get("code"))
        return response

    def wait(self, ids: list[int], timeout: float | None) -> dict:
        """Wait for the jobs to finish, across daemon restarts; {"jobs", "timed_out"}."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            left = None if deadline is None else max(0.0, deadline - time.monotonic())
            chunk = WAIT_CHUNK if left is None else min(left, WAIT_CHUNK)
            try:
                response = self.call("wait", ids=ids, timeout=chunk, io_timeout=chunk + 30)
            except (ConnectionResetError, BrokenPipeError, socket.timeout):
                time.sleep(0.2)  # the daemon went away mid-wait; the next call brings it back
                continue
            if not response["timed_out"] or (left is not None and left <= chunk):
                return response


def _git_root(path: str) -> str | None:
    current = Path(path).resolve()
    for folder in (current, *current.parents):
        if (folder / ".git").exists():
            return str(folder)
    return None


def current_session(env: Mapping[str, str] = os.environ, cwd: str | None = None) -> tuple[str, str]:
    """(session id, label): DIBS_SESSION if set, else the Claude Code session, else the terminal's."""
    session = env.get("DIBS_SESSION") or env.get("CLAUDE_CODE_SESSION_ID") or f"t{os.getsid(0)}"
    cwd = cwd or os.getcwd()
    project = os.path.basename((_git_root(cwd) or cwd).rstrip("/")) or "/"
    return session, f"{project}·{session[:4]}"
