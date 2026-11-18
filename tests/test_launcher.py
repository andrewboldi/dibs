import os
import signal
import subprocess
import sys
import time

import pytest

from dibs import launcher, procs
from dibs.model import Job, Request, State
from dibs.paths import ensure_dirs, get_paths
from dibs.units import GiB, MiB

PY = sys.executable


@pytest.fixture
def paths(tmp_path):
    p = get_paths({"DIBS_HOME": str(tmp_path / "home"), "HOME": str(tmp_path)})
    ensure_dirs(p)
    yield p
    launcher.sweep_job_tmps(p, keep=())  # launched without a daemon to delete their temp folders


_ids = iter(range(1, 10_000))


def make_job(paths, argv, *, cpu_ids=(0,), gpu_ids=(), mem=1 * GiB, vram=0, cwd=None, env=None):
    job_id = next(_ids)
    return Job(
        id=job_id,
        name="test",
        argv=list(argv),
        cwd=str(cwd or paths.state_dir),
        env=dict(os.environ if env is None else env),
        request=Request(cpus=len(cpu_ids), mem=mem, gpus=len(gpu_ids), vram=vram, time=60.0),
        session="s",
        session_label="test·s",
        submitted=time.time(),
        tag=0.0,
        state=State.RUNNING,
        output=str(paths.job_dir(job_id) / "output.log"),
        cpu_ids=list(cpu_ids),
        gpu_ids=list(gpu_ids),
    )


def run(paths, job, *, use_systemd=False):
    job.shim_pid, job.shim_ticks, job.unit = launcher.launch(job, paths, use_systemd=use_systemd)
    return job


def wait_exit(paths, job, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = launcher.read_exit(paths, job.id)
        if result:
            launcher.reap()
            return result
        time.sleep(0.05)
    raise AssertionError("job never reported an exit")


def wait_running(job, name, timeout=5.0):
    """The job's processes running `name` (after the fork, a child still looks like the shim)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        pids = [p for p in launcher.job_pids(job) if (procs.cmdline(p) or [""])[0] == name]
        if pids:
            return pids
        time.sleep(0.02)
    raise AssertionError(f"{name} never started")


def output(job):
    with open(job.output) as f:
        return f.read()


def test_job_env_sets_gpu_threads_and_ids(paths):
    job = make_job(paths, ["true"], cpu_ids=(0, 1, 2, 3), gpu_ids=(0,), vram=6 * GiB, env={"PATH": "/usr/bin"})
    env = launcher.job_env(job, "abcd1234")
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "4"
    assert env["NUMEXPR_NUM_THREADS"] == env["VECLIB_MAXIMUM_THREADS"] == "4"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["DIBS_JOB_ID"] == str(job.id) and env["DIBS_INSTANCE"] == "abcd1234"
    assert env["DIBS_VRAM"] == str(6 * GiB) and env["DIBS_MEM"] == str(1 * GiB)
    assert env["DIBS_CPUS"] == "4" and env["DIBS_GPUS"] == "0"
    assert env["PATH"] == "/usr/bin"


def test_job_env_points_temp_files_at_the_jobs_own_folder(paths, tmp_path):
    # Issue #89: a killed job's scratch files in a RAM-backed /tmp kept holding RAM after it ended.
    job = make_job(paths, ["true"], env={"PATH": "/usr/bin", "USER": "ann"})
    env = launcher.job_env(job, "abcd1234", tmp=tmp_path / "dibs-abcd1234-job-7")
    assert env["TMPDIR"] == env["TMP"] == env["TEMP"] == str(tmp_path / "dibs-abcd1234-job-7")
    assert env["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "torchinductor_ann")  # its usual place, kept across jobs
    mine = make_job(paths, ["true"], env={"TMPDIR": "/big/disk", "TORCHINDUCTOR_CACHE_DIR": "/cache"})
    env = launcher.job_env(mine, "abcd1234", tmp=tmp_path / "dibs-abcd1234-job-8")
    assert env["TMPDIR"] == "/big/disk" and "TMP" not in env and env["TORCHINDUCTOR_CACHE_DIR"] == "/cache"


def test_job_env_hides_gpus_and_keeps_submitter_threads(paths):
    job = make_job(paths, ["true"], env={"OMP_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": "0"})
    env = launcher.job_env(job, "abcd1234")
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["OMP_NUM_THREADS"] == "2"


def test_launch_runs_and_reports_exit(paths):
    job = run(paths, make_job(paths, [PY, "-c", "print('hi'); raise SystemExit(3)"]))
    result = wait_exit(paths, job)
    assert result["exit_code"] == 3 and result["reason"] is None
    assert "hi" in output(job)


def test_the_exit_record_has_how_long_the_machine_had_slept(paths):
    # Issue #76: run times leave out the time the machine slept, counted up to the command's exit.
    job = run(paths, make_job(paths, ["true"]))
    result = wait_exit(paths, job)
    assert isinstance(result["slept"], float) and 0 <= result["slept"] <= time.clock_gettime(time.CLOCK_BOOTTIME)


def test_launch_pins_cpus(paths):
    job = run(paths, make_job(paths, [PY, "-c", "import os; print(sorted(os.sched_getaffinity(0)))"], cpu_ids=(0, 1)))
    wait_exit(paths, job)
    assert output(job).strip() == "[0, 1]"


def test_launch_uses_cwd_and_env(paths, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    code = "import os; print(os.getcwd()); print(os.environ['FOO'])"
    job = run(paths, make_job(paths, [PY, "-c", code], cwd=work, env={**os.environ, "FOO": "bar baz"}))
    wait_exit(paths, job)
    assert output(job).split("\n")[:2] == [str(work), "bar baz"]


def test_launch_is_detached(paths):
    job = run(paths, make_job(paths, ["sleep", "30"]))
    try:
        wait_running(job, "sleep")
        assert os.getsid(job.shim_pid) != os.getsid(0)
    finally:
        launcher.signal_job(job, signal.SIGKILL)
        wait_exit(paths, job)


def test_bad_command_reports_127(paths):
    job = run(paths, make_job(paths, ["definitely-not-a-command-xyz"]))
    result = wait_exit(paths, job)
    assert result["exit_code"] == 127
    assert result["reason"].startswith("couldn't start:")


def test_vanished_cwd_reports_127(paths, tmp_path):
    gone = tmp_path / "gone"
    job = run(paths, make_job(paths, ["true"], cwd=gone))
    result = wait_exit(paths, job)
    assert result["exit_code"] == 127
    assert result["reason"] == f"couldn't start: working directory {gone} no longer exists"


def _job_processes(paths, job):
    found = []
    for name in os.listdir("/proc"):
        if name.isdigit():
            env = procs.environ(int(name)) or {}
            if env.get("DIBS_JOB_ID") == str(job.id) and env.get("DIBS_INSTANCE") == paths.instance:
                if procs.alive(int(name)):
                    found.append(int(name))
    return found


def test_leftovers_are_stopped(paths):
    job = run(paths, make_job(paths, ["bash", "-c", "sleep 300 & echo started"]))
    wait_exit(paths, job)
    assert "started" in output(job)
    deadline = time.time() + 3
    while _job_processes(paths, job) and time.time() < deadline:
        time.sleep(0.05)
    assert _job_processes(paths, job) == []


def test_signal_job_terminates(paths):
    job = run(paths, make_job(paths, ["sleep", "60"]))
    wait_running(job, "sleep")
    launcher.signal_job(job, signal.SIGTERM)
    assert wait_exit(paths, job)["exit_code"] == 143


def test_job_pids_empty_once_shim_is_gone(paths):
    job = run(paths, make_job(paths, ["true"]))
    wait_exit(paths, job)
    deadline = time.time() + 5
    while procs.alive(job.shim_pid, job.shim_ticks) and time.time() < deadline:
        time.sleep(0.02)
    assert launcher.job_pids(job) == []


def test_unit_name_includes_instance():
    assert launcher.unit_name("ab12cd34", 7) == "dibs-ab12cd34-job-7"


def test_systemd_can_be_turned_off():
    assert launcher.systemd_available({"DIBS_NO_SYSTEMD": "1"}) is False


needs_systemd = pytest.mark.skipif(
    not launcher.systemd_available({k: v for k, v in os.environ.items() if k != "DIBS_NO_SYSTEMD"}),
    reason="needs systemd-run --user with memory limits",
)


@needs_systemd
def test_ram_cap_kills_with_oom_flag(paths):
    code = "x = bytearray(300 << 20); print('allocated')"
    job = run(paths, make_job(paths, [PY, "-c", code], mem=100 * MiB), use_systemd=True)
    result = wait_exit(paths, job)
    assert result["exit_code"] == 137
    assert result["oom_killed"] is True
    assert result["peak_mem"] >= 99 * MiB
    assert "allocated" not in output(job)


@needs_systemd
def test_job_pids_from_scope(paths):
    job = run(paths, make_job(paths, ["sleep", "30"]), use_systemd=True)
    assert job.unit == launcher.unit_name(paths.instance, job.id)
    wait_running(job, "sleep")
    pids = launcher.job_pids(job)
    assert pids and job.shim_pid not in pids
    assert all(procs.cmdline(p)[0] == "sleep" for p in pids)
    launcher.signal_job(job, signal.SIGKILL)
    assert wait_exit(paths, job)["exit_code"] == 137


def test_orphans_are_reaped_while_the_job_runs(paths):
    # (sleep &) orphans each sleep; as a subreaper the shim inherits them and must reap them.
    script = "for i in $(seq 20); do (sleep 0.1 &); done; sleep 3"
    job = run(paths, make_job(paths, ["bash", "-c", script]))
    time.sleep(1.5)
    zombies = [c for c in procs.children_map().get(job.shim_pid, []) if procs.state(c) == "Z"]
    try:
        assert zombies == []
    finally:
        wait_exit(paths, job)


def test_probe_uses_the_properties_jobs_need(tmp_path, monkeypatch):
    # An older systemd that doesn't know OOMPolicy= for scopes must mean "no systemd", not lost jobs.
    fake = tmp_path / "systemd-run"
    fake.write_text(
        '#!/bin/sh\nfor a in "$@"; do case "$a" in OOMPolicy=*) echo "Unknown assignment: $a" >&2; exit 1;; esac; done\n'
        "echo 67108864\n"
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setattr(launcher, "_systemd", None)
    try:
        assert launcher.systemd_available({}) is False
    finally:
        launcher._systemd = None


def test_job_pids_never_reach_outside_the_jobs_own_scope(paths):
    # Issue #33: a job whose recorded unit wasn't the scope its launcher ran in made job_pids list
    # every process in the launcher's cgroup, and stopping the job then killed the terminal and the
    # agent that ran the test. Only the job's own scope may be read whole. This test only lists.
    stand_in = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        job = make_job(paths, ["true"])
        job.shim_pid, job.shim_ticks = stand_in.pid, procs.start_ticks(stand_in.pid)
        job.unit = "dibs-test-job-1"  # not the scope the stand-in runs in
        pids = launcher.job_pids(job)
        assert os.getpid() not in pids
        assert pids == []  # the stand-in launcher has no children
    finally:
        stand_in.kill()
        stand_in.wait()


@needs_systemd
def test_cpu_time_is_reported(paths):
    # Issue #31: what a job used is shown to the next agent that submits it; CPU time is part of it.
    code = "import time\nend = time.process_time() + 0.5\nwhile time.process_time() < end: pass"
    job = run(paths, make_job(paths, [PY, "-c", code]), use_systemd=True)
    result = wait_exit(paths, job)
    assert result["exit_code"] == 0
    assert 0.4 <= result["cpu_seconds"] < 5


def test_job_env_for_amd_uses_rocr_visible_devices(paths):
    job = make_job(paths, ["true"], gpu_ids=(1, 3), env={"HIP_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "0"})
    env = launcher.job_env(job, "abcd1234", "amd")
    assert env["ROCR_VISIBLE_DEVICES"] == "1,3"
    assert "HIP_VISIBLE_DEVICES" not in env and "CUDA_VISIBLE_DEVICES" not in env
    assert env["DIBS_GPUS"] == "1,3"


def test_job_env_for_amd_without_gpus_hides_them_all(paths):
    job = make_job(paths, ["true"], gpu_ids=())
    assert launcher.job_env(job, "abcd1234", "amd")["ROCR_VISIBLE_DEVICES"] == "-1"


def test_job_env_for_nvidia_leaves_rocm_variables_alone(paths):
    job = make_job(paths, ["true"], gpu_ids=(0,))
    env = launcher.job_env(job, "abcd1234")
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert "ROCR_VISIBLE_DEVICES" not in env
