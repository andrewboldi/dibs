"""dibs top says when the CPUs are capped below their top speed (#179). dibs only reads the cap."""

from dibs import procs
from dibs.units import GiB


def _sysfs(tmp_path, cores):
    """cores: (top speed, cap) in MHz per CPU."""
    for n, (top, cap) in enumerate(cores):
        d = tmp_path / f"cpu{n}" / "cpufreq"
        d.mkdir(parents=True)
        (d / "cpuinfo_max_freq").write_text(f"{top * 1000}\n")
        (d / "scaling_max_freq").write_text(f"{cap * 1000}\n")
    return str(tmp_path)


def test_cpus_capped_below_their_top_speed_are_read(tmp_path):
    # On one laptop the cap fell from 3.7-5.0 GHz to 1.5 GHz whenever the GPU drew about 50 W or more.
    root = _sysfs(tmp_path, [(5000, 1500)] * 4 + [(4800, 1500)] * 4 + [(3700, 1500)] * 4)
    assert procs.cpu_cap(root) == {"capped_mhz": 1500, "max_mhz": 5000}


def test_cpus_at_or_near_their_top_speed_arent_capped(tmp_path):
    assert procs.cpu_cap(_sysfs(tmp_path / "a", [(5000, 5000)] * 4 + [(3700, 3700)] * 4)) is None
    assert procs.cpu_cap(_sysfs(tmp_path / "b", [(5000, 4700)] * 8)) is None  # within a tenth
    assert procs.cpu_cap(_sysfs(tmp_path / "c", [(5000, 1500)] * 2 + [(5000, 5000)] * 6)) is None  # most aren't
    assert procs.cpu_cap(str(tmp_path / "none")) is None


def test_top_shows_the_cap_and_the_snapshot_carries_it(make_daemon, monkeypatch):
    from test_top import NOW

    from dibs.render import render_top

    monkeypatch.setattr(procs, "cpu_cap", lambda root=procs.CPU_ROOT: {"capped_mhz": 1500, "max_mhz": 4800})
    assert make_daemon().snapshot()["cpu_cap"] == {"capped_mhz": 1500, "max_mhz": 4800}
    snap = {"now": NOW, "jobs": [], "guard": {"mode": "all", "stopped": [], "outside": []},
            "cpu_cap": {"capped_mhz": 1500, "max_mhz": 4800},
            "capacity": {"cpus": list(range(24)), "mem": 32 * GiB, "gpus": []}}  # fmt: skip
    assert "capped at 1.5 GHz of 4.8" in render_top(snap, width=140, height=30, color=False)
