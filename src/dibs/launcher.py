"""Starting jobs, signalling them, and finding their processes (the daemon's side).

Each job runs under a small launcher process, the shim (dibs/shim.py), detached from the daemon so
that restarting dibs never touches running jobs. Where systemd allows it without root, the shim and
the job run in their own scope with MemoryMax set to the job's RAM reservation.
"""

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from dibs import procs
from dibs.model import Job
from dibs.paths import Paths
from dibs.units import MiB

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_systemd: bool | None = None
_children: dict[int, subprocess.Popen] = {}


def systemd_available(env: Mapping[str, str] = os.environ) -> bool:
    """Whether jobs can run in systemd scopes whose memory limits are really enforced."""
    global _systemd
    if env.get("DIBS_NO_SYSTEMD") == "1" or not sys.platform.startswith("linux"):
        return False
    if _systemd is None:
        _systemd = _probe_systemd()
    return _systemd


def scope_properties(mem: int) -> list[str]:
    """What a job's scope gets. The probe asks for exactly these, so an older systemd that doesn't
    know one of them means running without systemd, not every job failing to launch."""
    return ["-p", f"MemoryMax={mem}", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue"]


def _probe_systemd() -> bool:
    if not shutil.which("systemd-run"):
        return False
    script = "cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.max"
    try:
        out = subprocess.run(
            ["systemd-run", "--user", "--scope", "--quiet", *scope_properties(64 * MiB), "--", "sh", "-c", script],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0 and out.stdout.strip() == str(64 * MiB)


def unit_name(instance: str, job_id: int) -> str:
    return f"dibs-{instance}-job-{job_id}"


def job_env(job: Job, instance: str) -> dict[str, str]:
    """The submitter's environment, plus what the job was granted."""
    env = dict(job.env)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in job.gpu_ids)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    for var in THREAD_VARS:
        env.setdefault(var, str(job.request.cpus))
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["DIBS_JOB_ID"] = str(job.id)
    env["DIBS_INSTANCE"] = instance
    env["DIBS_CPUS"] = str(job.request.cpus)
    env["DIBS_MEM"] = str(job.request.mem)
    env["DIBS_VRAM"] = str(job.request.vram)
    env["DIBS_GPUS"] = ",".join(str(g) for g in job.gpu_ids)
    return env


def _write_private(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def launch(job: Job, paths: Paths, *, use_systemd: bool) -> tuple[int, int | None, str | None]:
    """Start the job's shim, detached. Returns (shim pid, its start ticks, systemd unit or None)."""
    job_dir = paths.job_dir(job.id)
    job_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    unit = unit_name(paths.instance, job.id) if use_systemd else None
    spec = {
        "job_id": job.id,
        "argv": job.argv,
        "cwd": job.cwd,
        "env": job_env(job, paths.instance),
        "cpu_ids": job.cpu_ids,
        "output": job.output,
        "mem": job.request.mem,
        "unit": unit,
    }
    _write_private(job_dir / "spec.json", json.dumps(spec))
    command = [sys.executable, "-m", "dibs.shim", str(job_dir)]
    if unit:
        command = [
            "systemd-run", "--user", "--scope", "--quiet", "--collect", f"--unit={unit}",
            *scope_properties(job.request.mem), "--", *command,
        ]  # fmt: skip
    with open(job_dir / "shim.log", "ab") as log:
        child = subprocess.Popen(
            command, start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=log
        )
    _children[child.pid] = child
    return child.pid, procs.start_ticks(child.pid), unit


def reap() -> None:
    """Collect shims that have exited, so they don't linger as zombies."""
    for pid, child in list(_children.items()):
        if child.poll() is not None:
            del _children[pid]


def job_pids(job: Job) -> list[int]:
    """The job's processes, not counting its shim."""
    if job.shim_pid is None or not procs.alive(job.shim_pid, job.shim_ticks):
        return []
    pids: list[int] = []
    cg = procs.cgroup(job.shim_pid) if job.unit else None
    if cg and cg.endswith(f"/{job.unit}.scope"):  # only the job's own scope, never a shared cgroup
        try:
            with open(f"/sys/fs/cgroup{cg}/cgroup.procs") as f:
                pids = [int(line) for line in f if line.strip()]
        except (OSError, ValueError):
            pids = []
    if not pids:
        pids = procs.descendants(job.shim_pid)
    return [p for p in pids if p != job.shim_pid]


def signal_job(job: Job, sig: int) -> int:
    """Signal every process of the job; returns how many got it."""
    sent = 0
    for pid in job_pids(job):
        try:
            os.kill(pid, sig)
            sent += 1
        except (ProcessLookupError, PermissionError):
            pass
    return sent


def read_exit(paths: Paths, job_id: int) -> dict | None:
    try:
        with open(paths.job_dir(job_id) / "exit.json") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
