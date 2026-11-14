import os
import subprocess
import sys
import time

import pytest

from dibs import procs

PY = sys.executable
SLEEP = [PY, "-c", "import time; time.sleep(30)"]


def wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("timed out")


def two_children(pid):
    kids = procs.descendants(pid)
    return kids if len(kids) == 2 else None


@pytest.fixture
def spawn():
    started = []

    def _spawn(*args, **kw):
        p = subprocess.Popen(*args, **kw)
        started.append(p)
        return p

    yield _spawn
    for p in started:
        if p.poll() is None:
            p.kill()
        p.wait()


def test_start_time_close_to_now(spawn):
    p = spawn(["sleep", "5"])
    assert abs(procs.start_time(p.pid) - time.time()) < 2


def test_alive_detects_pid_reuse(spawn):
    p = spawn(["sleep", "5"])
    ticks = procs.start_ticks(p.pid)
    assert procs.alive(p.pid) and procs.alive(p.pid, ticks)
    assert not procs.alive(p.pid, ticks + 1)
    p.kill()
    p.wait()
    assert not procs.alive(p.pid, ticks)


def test_alive_false_for_zombie(spawn):
    p = spawn(["true"])
    wait_for(lambda: procs.state(p.pid) == "Z")
    assert not procs.alive(p.pid)


def test_parent_and_descendants(spawn):
    p = spawn(["bash", "-c", "sleep 30 & sleep 30 & wait"])
    assert procs.parent(p.pid) == os.getpid()
    kids = wait_for(lambda: two_children(p.pid))
    assert all(procs.cmdline(k)[0] == "sleep" for k in kids)
    for k in kids:
        os.kill(k, 9)


def test_environ_and_cmdline(spawn):
    p = spawn([PY, "-c", "import time; time.sleep(30)", "arg two"], env={**os.environ, "FOO": "bar baz"})
    assert procs.environ(p.pid)["FOO"] == "bar baz"
    assert procs.cmdline(p.pid) == [PY, "-c", "import time; time.sleep(30)", "arg two"]


def test_exe_name_uid_rss_cgroup(spawn):
    p = spawn(["sleep", "5"])
    assert procs.exe_name(p.pid) == "sleep"
    assert procs.uid(p.pid) == os.getuid()
    assert procs.rss(p.pid) > 0
    assert procs.cgroup(p.pid).startswith("/")


def test_gone_process_reads_none():
    gone = 2**22 + 7
    assert procs.environ(gone) is None and procs.cmdline(gone) is None and procs.start_ticks(gone) is None
    assert not procs.alive(gone)


def test_write_once_pipe(spawn):
    p = spawn(SLEEP, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert procs.write_once([p.pid], "dibs: hello") == 1
    p.kill()
    out = p.stdout.read().decode()
    assert out.count("dibs: hello") == 1


def test_freeze_write_kill_log_file(spawn, tmp_path):
    log = tmp_path / "out.log"
    with open(log, "w") as f:  # truncating, not appending: the process keeps its own offset
        code = "print('line 1', flush=True); import time; time.sleep(30); print('never')"
        p = spawn([PY, "-c", code], stdout=f, stderr=subprocess.STDOUT)
    wait_for(lambda: "line 1" in log.read_text())
    procs.freeze_write_kill([p.pid], "dibs: stopped")
    p.wait()
    text = log.read_text()
    assert text.startswith("line 1\n")
    assert text.rstrip().endswith("dibs: stopped")
    assert text.count("dibs: stopped") == 1 and "never" not in text


def test_freeze_write_kill_kills_tree(spawn):
    p = spawn(["bash", "-c", "sleep 30 & sleep 30 & wait"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    kids = wait_for(lambda: two_children(p.pid))
    procs.freeze_write_kill([p.pid, *kids], "dibs: stopped")
    assert p.wait(timeout=5) == -9
    wait_for(lambda: not any(procs.alive(k) for k in kids))
    assert p.stdout.read().decode().count("dibs: stopped") == 1


def test_freeze_write_kill_catches_children_spawned_late(spawn):
    p = spawn(["bash", "-c", "sleep 30 & wait"])
    kid = wait_for(lambda: (d := procs.descendants(p.pid)) and d[0])
    procs.freeze_write_kill([p.pid], None)  # only the parent is named
    p.wait(timeout=5)
    wait_for(lambda: not procs.alive(kid))


def test_freeze_write_kill_ignores_gone_pids(spawn):
    p = spawn(["true"])
    p.wait()
    procs.freeze_write_kill([p.pid, 2**22 + 7], "x")
