"""Planning a job's GPU memory by what its command's earlier runs held (#183)."""

from conftest import capacity, run_until
from test_daemon import PY, finished, job, submit

from dibs.model import State
from dibs.units import GiB, MiB


def _render_script(tmp_path):
    """A script whose runs with --done end at once, and the others when `go` exists."""
    go = tmp_path / "go"
    script = tmp_path / "render.py"
    script.write_text(f"import os, sys, time\nwhile '--done' not in sys.argv and not os.path.exists({str(go)!r}):\n"
                      "    time.sleep(0.02)\n")  # fmt: skip
    return script, go


def test_a_job_asking_less_gpu_memory_than_its_runs_held_is_planned_at_what_they_held(make_daemon, tmp_path):
    # #183: a rendering script's request was cut from 1G to 256M, though its last runs held 544-759M,
    # and dibs doesn't hold rendering jobs to their reservations: four could have started where one fit.
    from dibs.model import Gpu

    d = make_daemon(cap=capacity(gpus=(Gpu(0, "GPU-0", "Fake GPU", 2 * GiB),)))
    script, go = _render_script(tmp_path)
    for n, peak in enumerate((544 * MiB, 759 * MiB)):  # earlier runs, and what they held
        earlier = submit(d, tmp_path, [PY, str(script), "--done", str(n)], vram=1 * GiB)
        run_until(d, lambda e=earlier: finished(d, e))
        j = job(d, earlier)
        j.peak_vram = peak
        d.store.update_job(j)
    holder = submit(d, tmp_path, [PY, str(script), "--hold"], vram=1 * GiB)
    games = [submit(d, tmp_path, [PY, str(script), "--game", str(n)], vram=256 * MiB) for n in range(4)]
    assert [job(d, g).state for g in games].count(State.RUNNING) == 1  # 759M each, in the 1G left
    waiting = d.show(games[-1])["job"]["waiting_for"]
    assert "its earlier runs held that much, though it asks for 256M" in waiting, waiting
    go.touch()
    run_until(d, lambda: all(finished(d, g) for g in [holder, *games]))


def test_a_job_that_asks_for_no_gpu_isnt_planned_for_its_runs_gpu_memory(make_daemon, tmp_path):
    # #183: the same script moved to rendering on the CPU, with no --vram.
    from dibs.model import Gpu

    d = make_daemon(cap=capacity(gpus=(Gpu(0, "GPU-0", "Fake GPU", 1 * GiB),)))
    script, go = _render_script(tmp_path)
    earlier = submit(d, tmp_path, [PY, str(script), "--done"], vram=1 * GiB)
    run_until(d, lambda: finished(d, earlier))
    j = job(d, earlier)
    j.peak_vram = 900 * MiB
    d.store.update_job(j)
    holder = submit(d, tmp_path, [PY, str(script), "--hold"], vram=1 * GiB)  # the whole GPU
    on_cpu = submit(d, tmp_path, [PY, str(script), "--cpu"])
    assert job(d, on_cpu).state == State.RUNNING
    go.touch()
    run_until(d, lambda: finished(d, holder) and finished(d, on_cpu))
