"""A job that asked for no GPU may hold a little GPU memory, as a browser's own GPU process takes,
without being stopped (#246): CUDA_VISIBLE_DEVICES doesn't hide the GPU from such a process."""

import time

from conftest import run_until
from test_guard import PY, _gpu_job, _job_python_pid, world  # noqa: F401  (world is a fixture)

from dibs.gpu import GpuProcess
from dibs.model import State
from dibs.units import GiB, MiB

SLEEP = [PY, "-c", "import time; time.sleep(30)"]


def _cpu_job(d, tmp_path, argv=SLEEP):
    job_id = _gpu_job(d, tmp_path, None, argv)
    run_until(d, lambda: d.store.get_job(job_id).state == State.RUNNING)
    return job_id


def _told(d):
    return [n["message"] for n in d.notices("s", time.time())["notices"] if n["kind"] == "gpu_unasked"]


def test_a_job_that_asked_for_no_gpu_may_hold_a_little_on_one_and_is_told_once(world, tmp_path):
    # 11-22: a job rendering pages with two headless browsers was stopped after 48 minutes for 4M.
    d, backend = world()
    job_id = _cpu_job(d, tmp_path)
    backend.procs = [GpuProcess(_job_python_pid(d, job_id), 0, 4 * MiB)]
    for _ in range(3):
        d.tick()
    j = d.store.get_job(job_id)
    assert j.state == State.RUNNING and j.stop_reason is None and j.peak_vram == 4 * MiB
    (told,) = _told(d)
    assert f"Job {job_id}" in told and "asked for no GPU" in told and "4M" in told and "--disable-gpu" in told
    assert _told(d) == []  # once
    d.cancel([job_id])
    run_until(d, lambda: d.store.get_job(job_id).state == State.CANCELLED)


def test_past_the_allowance_it_is_stopped_with_both_ways_out(world, tmp_path):
    d, backend = world()
    job_id = _cpu_job(d, tmp_path)
    backend.procs = [GpuProcess(_job_python_pid(d, job_id), 0, 100 * MiB)]
    run_until(d, lambda: d.store.get_job(job_id).state == State.FAILED)
    reason = d.store.get_job(job_id).reason
    assert reason.startswith("stopped: used 100M VRAM on GPU 0, reserved 0")
    assert "--disable-gpu" in reason and "--vram" in reason


def test_a_gpu_job_on_a_gpu_it_wasnt_given_is_still_stopped(world, tmp_path):
    from conftest import FAKE_GPU
    from dibs.model import Gpu

    second = Gpu(1, "GPU-fake-1", "Fake GPU", 8 * GiB)
    d, backend = world(gpus=(FAKE_GPU, second))
    job_id = _gpu_job(d, tmp_path, GiB, SLEEP)
    run_until(d, lambda: d.store.get_job(job_id).state == State.RUNNING)
    other = 1 if d.store.get_job(job_id).gpu_ids == [0] else 0
    backend.procs = [GpuProcess(_job_python_pid(d, job_id), other, 4 * MiB)]
    run_until(d, lambda: d.store.get_job(job_id).state == State.FAILED)


def test_the_help_no_longer_says_a_job_without_vram_cant_see_a_gpu():
    from dibs.instructions import AGENT_HELP

    text = " ".join(AGENT_HELP.split())
    assert "can't see any GPU" not in text
    assert "may hold up to 64M on a GPU" in text and "--disable-gpu" in text
