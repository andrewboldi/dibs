"""The machine keeps a small RAM reserve, and when it runs short dibs stops a job it squeezed in before
anything outside dibs: the newest job started on RAM other jobs reserved goes first (#253)."""

import time

from test_daemon import _borrowing, job

from dibs.machine import Config, ConfigError, load_config
from dibs.units import GiB


def _short(make_daemon, tmp_path, monkeypatch, *, swap, pressure):
    """Job A reserves 12G of 16G and uses 2G; borrower B started beside it. Then 15G of 16G is in use
    (1G available, under the 1.5G reserve), with `swap` free and memory pressure `pressure`."""
    level, used, now = [0.0], [2 * GiB], [time.time()]
    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=700, pressure=lambda: level[0], now=now,
                             meminfo=lambda: (16 * GiB, used[0]))  # fmt: skip
    assert job(d, b).borrowed_ram
    hogs = []
    d._stop_ram_hog = lambda t: hogs.append(t) or True
    d.swap_free = lambda: swap
    level[0], used[0] = pressure, 15 * GiB
    for _ in range(3):
        d.tick()
        now[0] += 6.0
    return d, a, b, go, hogs


def test_under_the_reserve_with_room_in_swap_and_no_pressure_nothing_is_stopped(make_daemon, tmp_path, monkeypatch):
    # The kernel can move idle programs' memory to swap: a returning peak needn't cost any work yet.
    d, a, b, go, hogs = _short(make_daemon, tmp_path, monkeypatch, swap=10 * GiB, pressure=0.0)
    assert job(d, b).stop_requested is None and hogs == []
    go.touch()


def test_under_the_reserve_with_swap_nearly_full_the_newest_borrower_goes_first(make_daemon, tmp_path, monkeypatch):
    d, a, b, go, hogs = _short(make_daemon, tmp_path, monkeypatch, swap=GiB, pressure=0.0)
    assert job(d, b).stop_reason == "stopped to give back borrowed RAM: the machine was short of memory"
    assert hogs == [] and job(d, a).stop_requested is None  # nothing outside dibs, nor the job that reserved
    go.touch()


def test_under_the_reserve_with_pressure_the_newest_borrower_goes_first(make_daemon, tmp_path, monkeypatch):
    d, a, b, go, hogs = _short(make_daemon, tmp_path, monkeypatch, swap=10 * GiB, pressure=12.0)
    assert job(d, b).stop_requested is not None and hogs == []
    go.touch()


def test_with_no_borrower_left_a_program_outside_dibs_is_stopped_as_before(make_daemon, tmp_path, monkeypatch):
    d, a, b, go, hogs = _short(make_daemon, tmp_path, monkeypatch, swap=GiB, pressure=0.0)
    stopped = job(d, b)
    stopped.borrowed_ram = False  # as if it had gone
    d.store.update_job(stopped)
    d._pressured_since = None
    d._give_back_borrowed = lambda now: False
    for _ in range(3):
        d.tick()
        d.clock = (lambda t: (lambda: t))(d.clock() + 6.0)
    assert hogs
    go.touch()


def test_the_reserve_is_a_config_setting():
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.toml"
        assert load_config(path).ram_reserve == int(1.5 * GiB)  # the default
        path.write_text('ram_reserve = "3G"\n')
        assert load_config(path) == Config(ram_reserve=3 * GiB)
        path.write_text("ram_reserve = 3\n")
        try:
            load_config(path)
        except ConfigError as e:
            assert "ram_reserve" in str(e)
        else:
            raise AssertionError("a size without a unit should be refused")
