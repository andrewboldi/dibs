"""dibs sees the machine's integrated GPU, which drives the built-in screen and renders through Mesa's
OpenGL and Vulkan, and how busy each job keeps it (#274). An agent had put a render on it unseen,
asking dibs for no GPU at all."""

import os
import time

from test_daemon import _job_holding, job

from dibs import procs
from dibs.units import GiB

INTEL, NVIDIA = "0000:00:02.0", "0000:01:00.0"


def _drm(root, entries):
    """A stand-in /sys/class/drm: each (node, vendor, pci address)."""
    drm = root / "drm"
    for node, vendor, pci in entries:
        device = root / "devices" / pci
        device.mkdir(parents=True, exist_ok=True)
        (device / "vendor").write_text(f"{vendor}\n")
        (drm / node).mkdir(parents=True)
        os.symlink(device, drm / node / "device")
    return str(drm)


def test_the_integrated_gpu_is_found_by_its_device_not_its_number(tmp_path, monkeypatch):
    monkeypatch.setattr(procs, "DRM_ROOT", _drm(tmp_path, [
        ("card1", "0x8086", INTEL), ("card2", "0x10de", NVIDIA), ("renderD128", "0x8086", INTEL),
        ("renderD129", "0x10de", NVIDIA), ("card1-eDP-1", "0x8086", INTEL)]))  # fmt: skip
    assert procs.find_igpu() == (INTEL, ("/dev/dri/card1", "/dev/dri/renderD128"))


def test_a_machine_without_an_integrated_gpu_has_none(tmp_path, monkeypatch):
    monkeypatch.setattr(procs, "DRM_ROOT", _drm(tmp_path, [("card0", "0x10de", NVIDIA), ("renderD128", "0x10de", NVIDIA)]))
    assert procs.find_igpu() is None
    monkeypatch.setattr(procs, "DRM_ROOT", str(tmp_path / "missing"))
    assert procs.find_igpu() is None


def _process(proc, pid, fds):
    """A stand-in /proc/PID: each fd (number, target, client id, {engine: ns})."""
    (proc / str(pid) / "fd").mkdir(parents=True, exist_ok=True)
    (proc / str(pid) / "fdinfo").mkdir(parents=True, exist_ok=True)
    for fd, target, client, engines in fds:
        link = proc / str(pid) / "fd" / str(fd)
        if link.is_symlink():
            link.unlink()
        os.symlink(target, link)
        lines = [f"drm-driver:\ti915", f"drm-client-id:\t{client}"] + [f"drm-engine-{e}:\t{ns} ns" for e, ns in engines.items()]
        (proc / str(pid) / "fdinfo" / str(fd)).write_text("\n".join(lines) + "\n")


def test_a_processs_clients_on_the_integrated_gpu_are_read_once_each(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    nodes = ("/dev/dri/card1", "/dev/dri/renderD128")
    _process(proc, 10, [(5, "/dev/dri/renderD128", 7, {"render": 900, "copy": 50, "video": 0}),
                        (6, "/dev/dri/renderD129", 8, {"render": 99999}),  # the NVIDIA GPU: not this one
                        (7, "/dev/null", 0, {})])  # fmt: skip
    _process(proc, 11, [(3, "/dev/dri/renderD128", 7, {"render": 900, "copy": 50, "video": 0})])  # a fork: client 7 again
    monkeypatch.setattr(procs, "PROC_ROOT", str(proc))
    assert procs.drm_clients([10, 11, 12], nodes) == {"7": {"render": 900, "copy": 50, "video": 0}}


def _rendering(make_daemon, tmp_path, monkeypatch, **daemon):
    """A job (pid 4242 in its scope) rendering on the integrated GPU; `look(seconds, render_ns)` moves the
    clock on, sets its render engine's busy time so far, and ticks."""
    now = [time.time()]
    proc = tmp_path / "proc"
    monkeypatch.setattr(procs, "PROC_ROOT", str(proc))
    d = make_daemon(clock=lambda: now[0], igpu=(INTEL, ("/dev/dri/card1", "/dev/dri/renderD128")), **daemon)
    a, stat, go = _job_holding(d, tmp_path, monkeypatch, 4 * GiB)
    (stat.parent / "cgroup.procs").write_text("4242\n")

    def look(seconds, render_ns):
        now[0] += seconds
        _process(proc, 4242, [(6, "/dev/dri/renderD128", 1234, {"render": render_ns, "copy": 0})])
        d.tick()

    return d, a, look, go


def test_how_busy_a_job_keeps_the_integrated_gpu_is_measured(make_daemon, tmp_path, monkeypatch):
    d, a, look, go = _rendering(make_daemon, tmp_path, monkeypatch)
    look(10, 0)  # dibs looks every 10 s: first sight
    look(10, 3_300_000_000)  # 3.3 s of its render engine in 10 s
    assert round(d._igpu_busy[a]) == 33 and round(job(d, a).peak_igpu) == 33
    look(10, 3_800_000_000)
    assert round(d._igpu_busy[a]) == 5 and round(job(d, a).peak_igpu) == 33
    go.touch()


def test_dibs_top_and_dibs_show_say_how_busy_a_job_keeps_the_integrated_gpu(make_daemon, tmp_path, monkeypatch):
    import re

    from dibs.render import render_show, render_top

    d, a, look, go = _rendering(make_daemon, tmp_path, monkeypatch)
    look(10, 0)
    look(10, 3_300_000_000)
    snap = d.snapshot()
    top = re.sub(r"\x1b\[[0-9;]*m", "", render_top(snap, width=120, height=40, color=False))
    assert re.search(r"igpu +▕.*▏ +33%/100%", top), top
    assert "Intel GPU up to 33% busy" in render_show(d.show(a)["job"], d.clock())
    go.touch()


def test_a_machine_without_one_shows_no_integrated_gpu_row(make_daemon):
    from dibs.render import render_top

    d = make_daemon()
    assert "igpu" not in render_top(d.snapshot(), width=120, height=40, color=False)
