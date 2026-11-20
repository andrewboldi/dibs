"""What dibs's memory relief decides, and how long its passes take, in its log (#191)."""

import logging
import re

from dibs import guard


def test_the_ram_guard_says_why_it_left_each_large_program_alone(monkeypatch):
    monkeypatch.setattr(guard, "_is_agent", lambda pid: pid == 1)
    monkeypatch.setattr(guard, "_where_agents_run", lambda pid: pid != 2)
    monkeypatch.setattr(guard.procs, "children_map", lambda: {})
    monkeypatch.setattr(guard.procs, "descendants", lambda pid, cmap: [])
    monkeypatch.setattr(guard, "_judge", lambda pid, *a: "preexisting")
    left = []
    hog = guard.ram_hog([(1, 9 << 30), (2, 8 << 30), (3, 7 << 30), (4, 1 << 30)], minimum=3 << 30, in_dibs=lambda pid: False,
                        mode="all", allow=(), since=0.0, my_uid=1000, left=left)  # fmt: skip
    assert hog is None
    assert left == [(1, 9 << 30, "an agent CLI"), (2, 8 << 30, "not in an agent's terminal"),
                    (3, 7 << 30, "it was running before the guard started")]  # fmt: skip


def test_a_shortage_is_logged_once_a_minute_with_what_the_relief_did(make_daemon, caplog):
    from dibs.units import GiB

    now = [1_000_000.0]
    d = make_daemon(clock=lambda: now[0], pressure=lambda: 60.0, meminfo=lambda: (16 * GiB, 15 * GiB))
    d._stop_ram_hog = lambda t: False
    caplog.set_level(logging.INFO, logger="dibs")
    for _ in range(9):  # 80 seconds of shortage, a check every 10
        d._relieve_shortage(now[0])
        now[0] += 10
    lines = [r.getMessage() for r in caplog.records if "short of memory" in r.getMessage()]
    assert len(lines) == 2  # at its first action and a minute later
    assert "1G of 16G available, memory pressure 60%" in lines[0] and "nothing to stop" in lines[0]


def test_a_slow_pass_is_logged(make_daemon, caplog, monkeypatch):
    import itertools

    d = make_daemon()
    ticks = itertools.count(step=7.0)  # each reading of the clock 7 s later than the last
    monkeypatch.setattr("dibs.daemon.time.monotonic", lambda: next(ticks))
    caplog.set_level(logging.INFO, logger="dibs")
    d.tick()
    took = [float(m.group(1)) for r in caplog.records if (m := re.search(r"a pass took (\d+\.\d) s", r.getMessage()))]
    assert took and took[0] >= 7.0
