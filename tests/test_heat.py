"""dibs top shows the CPU's and GPUs' temperatures, and how much of the last 10 minutes the CPU slowed
itself for heat (#278). On 11-23 the CPU sat at a median 95-96°C, against a 100°C limit, and slowed itself
28% of the time since boot, whatever the GPU drew; nothing in dibs top said so."""

import re
import time

from dibs import procs
from dibs.gpu import SmiBackend
from dibs.render import render_top


def _hwmon(root, name, sensors):
    """A stand-in /sys/class/hwmon entry: each (label, °C, critical °C)."""
    h = root / f"hwmon{len(list(root.glob('hwmon*'))) if root.exists() else 0}"
    h.mkdir(parents=True)
    (h / "name").write_text(f"{name}\n")
    for i, (label, temp, crit) in enumerate(sensors, 1):
        (h / f"temp{i}_label").write_text(f"{label}\n")
        (h / f"temp{i}_input").write_text(f"{int(temp * 1000)}\n")
        if crit is not None:
            (h / f"temp{i}_crit").write_text(f"{int(crit * 1000)}\n")


def test_the_cpu_package_temperature_and_its_limit_are_read(tmp_path):
    root = tmp_path / "hwmon"
    _hwmon(root, "nvme", [("Composite", 58, 86)])
    _hwmon(root, "coretemp", [("Core 0", 75, 100), ("Package id 0", 96, 100), ("Core 4", 88, 100)])
    assert procs.cpu_temp(str(root)) == (96.0, 100.0)
    assert procs.cpu_temp(str(tmp_path / "missing")) is None


def test_the_time_the_cpu_slowed_itself_for_heat_is_read(tmp_path):
    for cpu, ms in (("cpu0", 9_180_000), ("cpu1", 9_180_000), ("cpu2", 9_179_000)):
        t = tmp_path / cpu / "thermal_throttle"
        t.mkdir(parents=True)
        (t / "package_throttle_total_time_ms").write_text(f"{ms}\n")
    assert procs.heat_throttled_ms(str(tmp_path)) == 9_180_000
    assert procs.heat_throttled_ms(str(tmp_path / "missing")) is None


def test_nvidia_smi_says_each_gpus_temperature(dibs_env, monkeypatch):
    for key, value in dibs_env.items():
        monkeypatch.setenv(key, value)
    assert SmiBackend().temperatures() == {0: 50.0}


def _heated(make_daemon, *, temp=96.0, gpu=61.0):
    now = [time.time()]
    throttled = [0]
    d = make_daemon(clock=lambda: now[0], cpu_heat=lambda: (temp, 100.0), throttled=lambda: throttled[0])
    d.backend.temps = {0: gpu}

    def later(seconds, slowed_ms):
        now[0] += seconds
        throttled[0] += slowed_ms
        d.tick()

    later(0, 0)
    return d, later


def test_how_much_of_the_last_ten_minutes_the_cpu_slowed_itself_is_measured(make_daemon):
    d, later = _heated(make_daemon)
    for _ in range(150):  # 10 minutes, a sample every 4 s, slowed 1.24 s of each 4
        later(4, 1240)
    heat = d.snapshot()["heat"]
    assert heat["cpu"] == 96.0 and heat["cpu_limit"] == 100.0 and heat["gpus"] == {"0": 61.0}
    assert round(heat["slowed"], 2) == 0.31 and heat["over"] == 600.0


def test_dibs_top_draws_the_temperatures(make_daemon):
    d, later = _heated(make_daemon)
    for _ in range(150):
        later(4, 1240)
    top = re.sub(r"\x1b\[[0-9;]*m", "", render_top(d.snapshot(), width=140, height=40, color=False))
    assert re.search(r"cpu°C +▕.*▏ +96°C/100°C +slowed itself for heat 31% of the last 10m", top), top
    assert re.search(r"gpu°C +▕.*▏ +61°C/100°C", top), top


def test_no_temperatures_no_rows(make_daemon):
    d = make_daemon()
    top = render_top(d.snapshot(), width=140, height=40, color=False)
    assert "°C" not in top
