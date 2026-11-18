import asyncio
import fcntl
import json
import os
import socket
import subprocess
import sys
import threading
import time

import pytest

from dibs import build_id, procs
from dibs.daemon import run_daemon
from dibs.paths import ensure_dirs, get_paths
from dibs.units import GiB

PY = sys.executable


def call(sock_path, op, build=None, **fields):
    with socket.socket(socket.AF_UNIX) as s:
        s.settimeout(30)
        s.connect(str(sock_path))
        s.sendall((json.dumps({"op": op, "build": build or build_id(), **fields}) + "\n").encode())
        data = b""
        while not data.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
    return json.loads(data)


def wait_for(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    raise AssertionError("timed out")


def answers(sock_path) -> bool:
    """Whether a daemon is listening there: its socket file appears a moment before it listens."""
    try:
        return bool(call(sock_path, "ping").get("ok"))
    except OSError:
        return False


def serve_in_thread(d):
    thread = threading.Thread(target=lambda: asyncio.run(d.serve()), daemon=True)
    thread.start()
    wait_for(lambda: answers(d.paths.socket))
    return thread


@pytest.fixture
def server(make_daemon):
    d = make_daemon()
    thread = serve_in_thread(d)
    yield d, lambda op, **kw: call(d.paths.socket, op, **kw)
    try:
        call(d.paths.socket, "shutdown")
    except OSError:
        pass  # the test already shut it down
    thread.join(5)


def submit_req(tmp_path, argv, **request):
    return {
        "argv": argv,
        "cwd": str(tmp_path),
        "env": dict(os.environ),
        "session": "s1",
        "session_label": "proj·s1",
        "request": {"cpus": 1, "mem": 1 * GiB, "vram": None, "gpus": None, "time": 60.0, **request},
    }


def test_build_id_is_stable_and_overridable(monkeypatch):
    assert build_id() == build_id() and build_id().startswith("0.1.0+")
    monkeypatch.setenv("DIBS_BUILD_OVERRIDE", "old")
    assert build_id() == "old"


def test_ping(server):
    _, rpc = server
    assert rpc("ping") == {"ok": True, "build": build_id(), "pid": os.getpid()}


def test_submit_and_wait_over_socket(server, tmp_path):
    _, rpc = server
    job = rpc("submit", **submit_req(tmp_path, [PY, "-c", "print('hi')"]))["job"]
    resp = rpc("wait", ids=[job["id"]], timeout=10)
    assert resp["ok"] and not resp["timed_out"]
    assert resp["jobs"][0]["state"] == "done" and resp["jobs"][0]["exit_code"] == 0


def test_wait_timeout(server, tmp_path):
    _, rpc = server
    job = rpc("submit", **submit_req(tmp_path, ["sleep", "30"]))["job"]
    started = time.time()
    resp = rpc("wait", ids=[job["id"]], timeout=1)
    assert resp["timed_out"] and resp["jobs"][0]["state"] == "running"
    assert 0.9 <= time.time() - started < 4
    rpc("cancel", ids=[job["id"]])


def test_wait_unknown_job(server):
    _, rpc = server
    assert rpc("wait", ids=[42], timeout=1) == {"ok": False, "code": "not_found", "error": "no job 42"}


def test_queue_show_cancel_info(server, tmp_path):
    _, rpc = server
    job = rpc("submit", **submit_req(tmp_path, ["sleep", "30"]))["job"]
    assert [j["id"] for j in rpc("queue")["jobs"]] == [job["id"]]
    assert rpc("show", id=job["id"])["job"]["state"] == "running"
    assert rpc("cancel", ids=[job["id"]])["cancelled"] == [job["id"]]
    info = rpc("info")
    assert info["build"] == build_id() and info["capacity"]["gpus"][0]["vram"] == 8 * GiB


def test_build_mismatch_reported(server):
    _, rpc = server
    assert rpc("queue", build="old") == {"ok": False, "code": "build_mismatch", "build": build_id()}
    assert rpc("ping", build="old")["ok"]


def test_unknown_op(server):
    _, rpc = server
    resp = rpc("dance")
    assert resp["ok"] is False and resp["code"] == "bad_request"


def test_shutdown_leaves_jobs_running(server, tmp_path):
    d, rpc = server
    job = rpc("submit", **submit_req(tmp_path, ["sleep", "3"]))["job"]
    shim = d.store.get_job(job["id"]).shim_pid
    assert rpc("shutdown") == {"ok": True}
    wait_for(lambda: not d.paths.socket.exists())
    assert procs.alive(shim)


def test_stale_socket_file_replaced(make_daemon):
    d = make_daemon()
    ensure_dirs(d.paths)
    d.paths.socket.write_text("stale")
    thread = threading.Thread(target=lambda: asyncio.run(d.serve()), daemon=True)
    thread.start()
    wait_for(lambda: answers(d.paths.socket))  # the file appears a moment before the server listens
    assert call(d.paths.socket, "ping")["ok"]
    call(d.paths.socket, "shutdown")
    thread.join(5)


def test_second_daemon_exits_quietly(dibs_env):
    paths = get_paths(dibs_env)
    ensure_dirs(paths)
    with open(paths.lock, "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        started = time.time()
        assert run_daemon(dibs_env) == 0
        assert time.time() - started < 1


def _start_daemon_process(env):
    return subprocess.Popen(
        [PY, "-c", "from dibs.daemon import run_daemon; raise SystemExit(run_daemon())"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_run_daemon_end_to_end(dibs_env):
    paths = get_paths(dibs_env)
    p = _start_daemon_process(dibs_env)
    try:
        wait_for(lambda: answers(paths.socket))
        info = call(paths.socket, "info")
        assert info["gpu_backend"] == "nvidia-smi"
        assert info["capacity"]["gpus"] == [
            {"index": 0, "uuid": "GPU-fake-0", "name": "Fake GPU", "vram": 8 * GiB}
        ]
        assert info["enforcement"]["ram_cap"] is False  # DIBS_NO_SYSTEMD
        call(paths.socket, "shutdown")
        assert p.wait(timeout=10) == 0
        assert "started" in paths.daemon_log.read_text()
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()


def test_config_error_exits_1(dibs_env):
    paths = get_paths(dibs_env)
    paths.config_file.parent.mkdir(parents=True, exist_ok=True)
    paths.config_file.write_text('guard = "maybe"\n')
    p = _start_daemon_process(dibs_env)
    _, err = p.communicate(timeout=20)
    assert p.returncode == 1
    assert "guard" in err.decode() and "guard" in paths.daemon_log.read_text()


def test_daemon_log_rotates_while_running(tmp_path, monkeypatch):
    # A failure repeating every second must not fill the disk.
    from dibs import daemon as dmod

    paths = get_paths({"DIBS_HOME": str(tmp_path / "h"), "HOME": str(tmp_path)})
    ensure_dirs(paths)
    monkeypatch.setattr(dmod, "LOG_MAX", 10_000)
    try:
        dmod._setup_logging(paths)
        for _ in range(2000):
            dmod.log.info("x" * 50)
        assert paths.daemon_log.stat().st_size <= 10_000
        assert paths.daemon_log.with_name("daemon.log.1").exists()
    finally:
        for handler in dmod.log.handlers:
            handler.close()
        dmod.log.handlers.clear()


def test_waiting_for_the_daemon_outlasts_the_gap_before_it_listens(make_daemon, monkeypatch):
    # Issue #45: asyncio creates the socket file at bind(), a moment before listen(), and a request
    # in between is refused. Hold that gap open: waiting for the daemon must outlast it.
    real = asyncio.start_unix_server

    async def slow_to_listen(client, path, **kw):
        early = socket.socket(socket.AF_UNIX)
        early.bind(path)  # the file exists; nothing listens yet
        await asyncio.sleep(0.5)
        early.close()
        os.unlink(path)
        return await real(client, path=path, **kw)

    monkeypatch.setattr(asyncio, "start_unix_server", slow_to_listen)
    d = make_daemon()
    thread = serve_in_thread(d)
    try:
        assert call(d.paths.socket, "ping")["ok"]
    finally:
        call(d.paths.socket, "shutdown")
        thread.join(5)
