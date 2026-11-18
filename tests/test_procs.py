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
    assert wait_for(lambda: all((procs.cmdline(k) or [""])[0] == "sleep" for k in kids))
    for k in kids:
        os.kill(k, 9)


def test_environ_and_cmdline(spawn):
    p = spawn([PY, "-c", "import time; time.sleep(30)", "arg two"], env={**os.environ, "FOO": "bar baz"})
    # Popen returns as soon as exec begins; /proc shows the new environment a moment later.
    assert wait_for(lambda: (procs.environ(p.pid) or {}).get("FOO")) == "bar baz"
    expected = [PY, "-c", "import time; time.sleep(30)", "arg two"]
    assert wait_for(lambda: procs.cmdline(p.pid) == expected)


def test_exe_name_uid_rss_cgroup(spawn):
    p = spawn(["sleep", "5"])
    assert wait_for(lambda: procs.exe_name(p.pid) == "sleep")
    assert procs.uid(p.pid) == os.getuid()
    assert wait_for(lambda: procs.rss(p.pid))
    assert procs.cgroup(p.pid).startswith("/")


def test_gone_process_reads_none():
    gone = 2**22 + 7
    assert procs.environ(gone) is None and procs.cmdline(gone) is None and procs.start_ticks(gone) is None
    assert not procs.alive(gone)


def test_stop_with_message_pipe(spawn):
    p = spawn(SLEEP, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    procs.stop_with_message([p.pid], "dibs: hello")
    assert p.wait(timeout=5) == -9
    out = p.stdout.read().decode()
    assert out.count("dibs: hello") == 1


def test_stop_with_message_log_file(spawn, tmp_path):
    log = tmp_path / "out.log"
    with open(log, "w") as f:  # truncating, not appending: the process keeps its own offset
        code = "print('line 1', flush=True); import time; time.sleep(30); print('never')"
        p = spawn([PY, "-c", code], stdout=f, stderr=subprocess.STDOUT)
    wait_for(lambda: "line 1" in log.read_text())
    procs.stop_with_message([p.pid], "dibs: stopped")
    p.wait()
    text = log.read_text()
    assert text.startswith("line 1\n")
    assert text.rstrip().endswith("dibs: stopped")
    assert text.count("dibs: stopped") == 1 and "never" not in text


def test_stop_with_message_kills_tree(spawn):
    p = spawn(["bash", "-c", "sleep 30 & sleep 30 & wait"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    kids = wait_for(lambda: two_children(p.pid))
    procs.stop_with_message([p.pid, *kids], "dibs: stopped")
    assert p.wait(timeout=5) == -9
    wait_for(lambda: not any(procs.alive(k) for k in kids))
    assert p.stdout.read().decode().count("dibs: stopped") == 1


def test_stop_with_message_finds_children_not_named(spawn):
    p = spawn(["bash", "-c", "sleep 30 & wait"])
    kid = wait_for(lambda: (d := procs.descendants(p.pid)) and d[0])
    procs.stop_with_message([p.pid], None)  # only the parent is named
    p.wait(timeout=5)
    wait_for(lambda: not procs.alive(kid))


def test_stop_with_message_ignores_gone_pids(spawn):
    p = spawn(["true"])
    p.wait()
    procs.stop_with_message([p.pid, 2**22 + 7], "x")


def test_start_time_is_precise(spawn):
    # The guard compares start times with the moment it was switched on, so whole seconds won't do.
    errors = []
    for _ in range(5):
        before = time.time()
        p = spawn(["sleep", "5"])
        errors.append(abs(procs.start_time(p.pid) - before))
    assert max(errors) < 0.1


def test_stopping_a_job_in_an_interactive_shell_reads_cleanly():
    """An interactive bash must not report the job as "Stopped" (SIGSTOP would make it), and the
    message must come before the shell's next prompt. `script` gives bash a real terminal, so job
    control is on."""
    import select

    shell = subprocess.Popen(
        ["script", "-qfc", "bash --noprofile --norc -i", "/dev/null"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )  # fmt: skip
    out = b""

    def type_line(text):
        shell.stdin.write(text.encode() + b"\n")
        shell.stdin.flush()

    def read_until(predicate, seconds):
        nonlocal out
        deadline = time.time() + seconds
        while time.time() < deadline and not predicate(out.decode(errors="replace")):
            if select.select([shell.stdout], [], [], 0.05)[0]:
                chunk = os.read(shell.stdout.fileno(), 65536)
                if not chunk:
                    break
                out += chunk
        return out.decode(errors="replace")

    try:
        type_line("PS1='PROMPT> '")
        read_until(lambda text: "PROMPT> " in text, 10)
        type_line(f"'{PY}' -c 'import time; time.sleep(30)'")
        job = wait_for(lambda: next((c for c in procs.descendants(shell.pid) if (procs.cmdline(c) or [""])[0] == PY), None))
        read_until(lambda text: False, 0.3)
        out = b""
        procs.stop_with_message([job], "dibs: stopped it")
        text = read_until(lambda text: "dibs: stopped it" in text and "PROMPT> " in text.split("dibs: stopped it")[-1], 10)
        assert "Stopped" not in text, text
        assert "dibs: stopped it" in text, text
        assert "PROMPT> " in text.split("dibs: stopped it")[-1], text
    finally:
        shell.kill()
        shell.wait()
        shell.stdin.close()
        shell.stdout.close()


def _survivors(marker):
    found = []
    for name in os.listdir("/proc"):
        if name.isdigit() and procs.cmdline(int(name)) == ["sleep", marker] and procs.alive(int(name)):
            found.append(int(name))
    return found


def test_children_forked_at_the_moment_of_the_stop_are_stopped_too(spawn, monkeypatch):
    # Issue #8: once a parent is killed its children are re-parented, so a child forked between
    # the look for children and the kill used to escape. The fork is placed exactly in that window.
    import signal

    marker = "31.4159"
    code = (
        "import signal, subprocess, time; "
        f"signal.signal(signal.SIGUSR1, lambda *a: subprocess.Popen(['sleep', '{marker}'])); "
        "time.sleep(60)"
    )
    parent = spawn([PY, "-c", code], start_new_session=True)  # no terminal, like an agent's command
    wait_for(lambda: procs.cmdline(parent.pid) and procs.cmdline(parent.pid)[0] == PY)
    time.sleep(0.2)  # let it install the handler
    real_children_map = procs.children_map
    looked = []

    def fork_right_after_the_first_look():
        snapshot = real_children_map()
        if not looked:
            looked.append(True)
            os.kill(parent.pid, signal.SIGUSR1)
            wait_for(lambda: _survivors(marker))
        return snapshot

    monkeypatch.setattr(procs, "children_map", fork_right_after_the_first_look)
    try:
        procs.stop_with_message([parent.pid], None)
        parent.wait(timeout=5)
        time.sleep(0.3)
        assert _survivors(marker) == []
    finally:
        for pid in _survivors(marker):
            os.kill(pid, 9)


def test_memory_pressure_reads_psi(monkeypatch):
    text = "some avg10=12.50 avg60=3.00 avg300=1.00 total=123\nfull avg10=4.00 avg60=1.00 avg300=0.50 total=45\n"
    monkeypatch.setattr(procs, "_read", lambda path, binary=False: text if path == "/proc/pressure/memory" else None)
    assert procs.memory_pressure() == 12.5
    monkeypatch.setattr(procs, "_read", lambda path, binary=False: None)
    assert procs.memory_pressure() == 0.0
