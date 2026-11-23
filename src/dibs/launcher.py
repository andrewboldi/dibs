"""Starting jobs, signalling them, and finding their processes (the daemon's side).

Each job runs under a small launcher process, the shim (dibs/shim.py), detached from the daemon so
that restarting dibs never touches running jobs. Where systemd allows it without root, the shim and
the job run in their own scope with MemoryMax set to the job's RAM reservation.
"""

import getpass
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Collection, Mapping
from pathlib import Path

from dibs import procs, snapshot
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


JOB_CPU_WEIGHT = 1000  # systemd's default is 100: on contended cores a job's threads come first (#152)


def scope_properties(mem: int) -> list[str]:
    """What a job's scope gets. The probe asks for exactly these, so an older systemd that doesn't
    know one of them means running without systemd, not every job failing to launch. A job is pinned
    to its cores, but pinning doesn't keep other programs off them; the weight gives it first claim."""
    return ["-p", f"MemoryMax={mem}", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue",
            "-p", f"CPUWeight={JOB_CPU_WEIGHT}"]  # fmt: skip


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


def set_memory_max(job: Job, limit: int) -> bool:
    """Raise a running job's RAM limit (#155). --runtime keeps it through a systemd reload, which would
    otherwise put the scope back at its reservation and kill a job holding more."""
    if not job.unit:
        return False
    try:
        done = subprocess.run(["systemctl", "--user", "set-property", "--runtime", f"{job.unit}.scope", f"MemoryMax={limit}"],
                              capture_output=True, timeout=10)  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0


def raise_oom_score(job: Job) -> None:
    """Give a job's processes the borrowers' OOM score: if the machine runs out, the kernel kills it
    first. Raising it needs no privilege, and what the job starts later inherits it."""
    for pid in job_pids(job):
        try:
            with open(f"/proc/{pid}/oom_score_adj", "w") as f:
                f.write(str(BORROWER_OOM_SCORE))
        except OSError:
            pass


def unit_name(instance: str, job_id: int) -> str:
    return f"dibs-{instance}-job-{job_id}"


BORROWER_OOM_SCORE = 800


def job_tmp(paths: Paths, job_id: int) -> Path:
    """The job's own temp folder, under the system temp directory, deleted when the job ends (#89)."""
    return Path(tempfile.gettempdir()) / f"dibs-{paths.instance}-job-{job_id}"


def remove_job_tmp(paths: Paths, job_id: int) -> None:
    shutil.rmtree(job_tmp(paths, job_id), ignore_errors=True)


def sweep_job_tmps(paths: Paths, keep: Collection[int]) -> None:
    """Delete this dibs's job temp folders whose jobs aren't running (left by a crash or a reboot)."""
    prefix = f"dibs-{paths.instance}-job-"
    for entry in Path(tempfile.gettempdir()).glob(prefix + "*"):
        job_id = entry.name[len(prefix):]
        if job_id.isdigit() and int(job_id) not in keep:
            shutil.rmtree(entry, ignore_errors=True)


def job_env(job: Job, instance: str, vendor: str = "nvidia", tmp: Path | None = None) -> dict[str, str]:
    """The submitter's environment, plus what the job was granted. `vendor` is the machine's GPU
    vendor: NVIDIA jobs get CUDA_VISIBLE_DEVICES, AMD jobs ROCR_VISIBLE_DEVICES (ROCm's recommended
    variable on Linux; it hides GPUs from the HSA runtime, under HIP, OpenCL and everything else)."""
    env = dict(job.env)
    if vendor == "amd":
        # HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES count within what ROCR leaves visible, so a
        # value the submitter had set would hide the granted GPU (or reveal none); drop them.
        for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
            env.pop(var, None)
        # "-1" is an invalid index, which ROCm reads as "no GPUs"; an empty value isn't documented.
        env["ROCR_VISIBLE_DEVICES"] = ",".join(str(g) for g in job.gpu_ids) or "-1"
    else:
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
    if tmp is not None and "TMPDIR" not in env:
        # Its own temp folder, deleted when it ends, so scratch files a killed job leaves behind don't
        # hold RAM where /tmp is a tmpfs (#89). Tools that cache under the temp directory keep their
        # usual place, so compiled kernels aren't thrown away with each job.
        user = env.get("USER") or getpass.getuser()
        env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(tmp.parent / f"torchinductor_{user}"))
        for var in ("TMPDIR", "TMP", "TEMP"):
            env[var] = str(tmp)
    return env


def _write_private(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def launch(job: Job, paths: Paths, *, use_systemd: bool, vendor: str = "nvidia",
           huge_pages: bool = False, mem_limit: int | None = None) -> tuple[int, int | None, str | None]:  # fmt: skip
    """Start the job's shim, detached. Returns (shim pid, its start ticks, systemd unit or None)."""
    job_dir = paths.job_dir(job.id)
    job_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = job_tmp(paths, job.id)
    tmp.mkdir(mode=0o700, parents=True, exist_ok=True)
    unit = unit_name(paths.instance, job.id) if use_systemd else None
    cwd, argv, env = job.cwd, job.argv, job_env(job, paths.instance, vendor, tmp=tmp)
    if job.snapshot:  # it runs its code as it was at submit (#172)
        cwd, argv = snapshot.run_paths(job.snapshot, str(job_dir / "snapshot" / "tree"), job.cwd, job.argv)
        env["PWD"] = cwd
    spec = {
        "job_id": job.id,
        "argv": argv,
        "cwd": cwd,
        "env": env,
        "cpu_ids": job.cpu_ids,
        "output": job.output,
        "mem": job.request.mem,
        "unit": unit,
        # Started on borrowed RAM: if the machine runs out, the kernel kills this job first (#86).
        "oom_score_adj": BORROWER_OOM_SCORE if job.borrowed_ram else None,
        "huge_pages": huge_pages,  # off unless the config keeps them (#225)
    }
    _write_private(job_dir / "spec.json", json.dumps(spec))
    command = [sys.executable, "-m", "dibs.shim", str(job_dir)]
    if unit:
        command = [
            "systemd-run", "--user", "--scope", "--quiet", "--collect", f"--unit={unit}",
            *scope_properties(mem_limit or job.request.mem), "--", *command,
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
