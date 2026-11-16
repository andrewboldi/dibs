import json
import os
import signal
import time
from pathlib import Path

import pytest

from dibs import launcher, paths
from dibs.daemon import Daemon
from dibs.gpu import GpuProcess
from dibs.machine import Config, detect_cpus
from dibs.model import Capacity, Gpu, State
from dibs.paths import get_paths
from dibs.units import GiB

FAKES = Path(__file__).parent / "fakes"
FAKE_GPU = Gpu(0, "GPU-fake-0", "Fake GPU", 8 * GiB)


@pytest.fixture(autouse=True)
def _never_the_real_runtime_folder(monkeypatch, tmp_path_factory):
    """Without XDG_RUNTIME_DIR, dibs falls back to /run/user/UID, where a real dibs may listen.
    Tests that unset it to stay isolated must land in a folder of their own instead."""
    monkeypatch.setattr(paths, "SYSTEM_RUNTIME_ROOT", tmp_path_factory.getbasetemp() / "no-run-user")


class FakeBackend:
    """GPUs and GPU processes that tests control."""

    name = "fake"

    def __init__(self, gpus=(FAKE_GPU,), procs=()):
        self.gpus = list(gpus)
        self.procs: list[GpuProcess] = list(procs)
        self.blind = False  # True: the driver doesn't report processes

    def list_gpus(self):
        return list(self.gpus)

    def processes(self):
        return None if self.blind else list(self.procs)


@pytest.fixture
def dibs_env(tmp_path):
    """Environment for running the real dibs CLI/daemon in isolation, with one fake 8G GPU."""
    gpu_file = tmp_path / "gpus.json"
    gpu_file.write_text(
        json.dumps({"gpus": [{"index": 0, "uuid": "GPU-fake-0", "name": "Fake GPU", "memory_mib": 8192}]})
    )
    env = dict(os.environ)
    env.update(
        DIBS_HOME=str(tmp_path / "home"),
        DIBS_NO_SYSTEMD="1",
        DIBS_GPU_BACKEND="smi",
        DIBS_FAKE_GPU_FILE=str(gpu_file),
        DIBS_MEMINFO=f"{FAKES}/meminfo",  # nothing in use outside dibs, whatever this machine is doing
        PATH=f"{FAKES}:{os.environ['PATH']}",
    )
    env.pop("CLAUDE_CODE_SESSION_ID", None)
    env.pop("DIBS_SESSION", None)
    return env


def capacity(cpus=None, mem=16 * GiB, gpus=(FAKE_GPU,)):
    return Capacity(cpus=tuple(cpus if cpus is not None else detect_cpus()), mem=mem, gpus=tuple(gpus))


def stop_everything(daemon):
    """Kill whatever jobs a test left running and collect their shims."""
    running = daemon.store.jobs([State.RUNNING])
    deadline = time.time() + 10
    while running and time.time() < deadline:
        for job in running:
            launcher.signal_job(job, signal.SIGKILL)
        running = [j for j in running if not launcher.read_exit(daemon.paths, j.id)]
        time.sleep(0.05)
    launcher.reap()


@pytest.fixture
def make_daemon(tmp_path):
    made = []

    def _make(config=Config(), cap=None, backend=None, clock=time.time, meminfo=lambda: None):
        paths = get_paths({"DIBS_HOME": str(tmp_path / "home"), "HOME": str(tmp_path)})
        d = Daemon(
            paths, config, cap or capacity(), backend or FakeBackend(), use_systemd=False, clock=clock, meminfo=meminfo
        )
        made.append(d)
        return d

    yield _make
    for d in made:
        if not d.closed:
            stop_everything(d)
            d.close()


@pytest.fixture
def daemon(make_daemon):
    return make_daemon()


def run_until(daemon, predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        daemon.tick()
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("timed out waiting")
