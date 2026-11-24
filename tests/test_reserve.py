"""The machine keeps a small RAM reserve, and when it runs short dibs stops a job it squeezed in before
anything outside dibs: the newest job started on RAM other jobs reserved goes first (#253)."""

import time
from collections import deque

import pytest
from test_daemon import _borrowing, job

from dibs.model import State

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


@pytest.mark.parametrize("asks, starts", [(13 * GiB, False), (int(12.5 * GiB), True)])
def test_a_job_starts_on_unused_ram_only_leaving_the_reserve(make_daemon, tmp_path, monkeypatch, asks, starts):
    # A reserves 12G of 16G and counts at the 2G it uses (#255): 14G free by the count. Asking 13G
    # would leave 1G, under the 1.5G reserve; 12.5G leaves it.
    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=700, mem=asks)
    assert (job(d, b).state == State.RUNNING) == starts
    go.touch()


@pytest.mark.parametrize("held, limit, grows", [(8 * GiB, None, True), (int(11.5 * GiB), int(11.5 * GiB), False)])
def test_a_job_grows_into_unused_ram_only_leaving_the_reserve(make_daemon, tmp_path, monkeypatch, held, limit, grows):
    # A counts at the 2G it uses. B (8G, started on A's unused RAM, with earlier runs) nears its limit:
    # a 2G step from 8G leaves 16 - 2 - 10 = 4G; from 11.5G it would leave 0.5G, under the 1.5G reserve.
    used = [2 * GiB]
    d, a, b, go = _borrowing(make_daemon, tmp_path, monkeypatch, ran=700, meminfo=lambda: (16 * GiB, used[0]))
    assert job(d, b).borrowed_ram
    j = job(d, b)
    j.history_mem, j.ram_limit = 8 * GiB, limit
    d.store.update_job(j)
    used[0] = 2 * GiB + held
    real = d._memory_in_use
    d._memory_in_use = lambda job: (held, 0, (0, 0)) if job.id == b else real(job)
    d._held[b] = held
    d._recent_held[b] = deque([(d.clock(), held)])
    raised = []
    d.set_mem_limit = lambda job, new: raised.append(new) or True
    d._grow(d.clock())
    assert bool(raised) == grows
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
