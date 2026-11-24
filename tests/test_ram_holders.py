"""dibs top says what holds the RAM in use outside dibs, and a job waiting for RAM names the largest
holder (#164)."""

import os

from dibs import procs
from dibs.units import GiB


def _scope(root, path, anon, shmem=0, swap=0):
    d = root / path
    d.mkdir(parents=True)
    (d / "memory.stat").write_text(f"anon {anon}\nfile 0\nshmem {shmem}\n")
    (d / "memory.swap.current").write_text(f"{swap}\n")


def _tree(tmp_path):
    uid = os.getuid()
    service = tmp_path / "user.slice" / f"user-{uid}.slice" / f"user@{uid}.service"
    _scope(service, "app.slice/snap.chromium.chromium-776a4be7-f856-402c-827c-556237ea8bf6.scope", 6 * GiB, GiB, 3 * GiB)
    _scope(service, "app.slice/snap.chromium.chromium-1f54c94b-36bd-412d-8bc8-46c5fba1a975.scope", GiB // 2, 0, GiB)
    _scope(service, "app.slice/app-org.gnome.Terminal.slice/tmux-spawn-c8b7962d-4ecc-4032-b228-c2d24f2dd37d.scope", 2 * GiB)
    _scope(service, "app.slice/dibs-fdc580e1-job-2926.scope", 9 * GiB)  # a dibs job: not outside dibs
    _scope(service, "app.slice/ollama.service", GiB // 10)  # too small to name
    (service / "memory.stat").write_text(f"anon {20 * GiB}\nshmem 0\n")  # the whole instance: not a holder
    (service / "app.slice" / "app-org.gnome.Terminal.slice" / "memory.stat").write_text(f"anon {3 * GiB}\nshmem 0\n")
    return str(tmp_path)


def test_the_largest_holders_of_ram_outside_dibs_are_named_with_their_swap(tmp_path, monkeypatch):
    # #164: most of the RAM in use outside dibs was one automated browser, over 44 processes, and its swap.
    monkeypatch.setattr(procs, "CGROUP_ROOT", _tree(tmp_path))
    holders = procs.outside_holders(minimum=GiB)
    assert holders == [
        {"name": "chromium", "ram": int(7.5 * GiB), "swap": 4 * GiB},
        {"name": "terminal pane", "ram": 2 * GiB, "swap": 0},
    ]


def test_a_scope_dibs_counts_as_a_jobs_isnt_named_as_outside_dibs(tmp_path, monkeypatch):
    # #269: a job's snap browser left its scope, and dibs counts it as the job's (#265); a waiting job's
    # reason named it as the user's browser all the same.
    monkeypatch.setattr(procs, "CGROUP_ROOT", _tree(tmp_path))
    holders = procs.outside_holders(minimum=GiB, skip={"snap.chromium.chromium-776a4be7-f856-402c-827c-556237ea8bf6.scope"})
    assert holders == [
        {"name": "terminal pane", "ram": 2 * GiB, "swap": 0},
        {"name": "chromium", "ram": GiB // 2, "swap": GiB},
    ]


def test_the_daemon_leaves_its_jobs_escaped_scopes_out_of_whats_outside(make_daemon, monkeypatch):
    d = make_daemon()
    asked = []
    monkeypatch.setattr(procs, "outside_holders", lambda minimum, skip=frozenset(): asked.append(set(skip)) or [])
    d._escaped_scopes = {"/user.slice/app.slice/snap.chromium.chromedriver-9f3c.scope": 7}
    d._ram_holders(d.clock())
    assert asked == [{"snap.chromium.chromedriver-9f3c.scope"}]


def test_a_job_waiting_for_ram_names_the_largest_holder_outside_dibs():
    from dibs.model import Capacity, Request
    from dibs.planner import Queued, plan

    cap = Capacity(cpus=tuple(range(8)), mem=32 * GiB, gpus=())
    p = plan(cap, [], [Queued(1, Request(cpus=1, mem=8 * GiB, gpus=0, vram=0, time=60.0), 0.0, 0.0)], 0.0,
             outside_mem=28 * GiB, outside_top=("chromium", 21 * GiB))  # fmt: skip
    assert "28G in use outside dibs, 21G of it by chromium" in p.predictions[1].reason


def test_top_lists_the_holders_and_the_files_in_a_ram_backed_tmp():
    from test_top import NOW

    from dibs.render import render_top

    snap = {"now": NOW, "jobs": [], "guard": {"mode": "all", "stopped": [], "outside": []},
            "outside": {"cpus": 0, "mem": 14 * GiB, "vram": {}},
            "outside_holders": [{"name": "chromium", "ram": int(7.1 * GiB), "swap": int(3.8 * GiB)},
                                {"name": "terminal pane (claude)", "ram": int(1.9 * GiB), "swap": 0}],
            "tmp_files": int(2.3 * GiB),
            "capacity": {"cpus": list(range(24)), "mem": 32 * GiB, "gpus": []}}  # fmt: skip
    text = render_top(snap, width=160, height=30, color=False)
    assert "chromium 7.1G (+3.8G swap)" in text and "terminal pane (claude) 1.9G" in text and "files in /tmp 2.3G" in text


def test_the_snapshot_carries_the_holders(make_daemon, monkeypatch):
    monkeypatch.setattr(procs, "outside_holders", lambda minimum, skip=frozenset(): [{"name": "chromium", "ram": 7 * GiB, "swap": 0}])
    monkeypatch.setattr(procs, "tmp_files", lambda: 2 * GiB)
    snap = make_daemon().snapshot()
    assert snap["outside_holders"][0]["name"] == "chromium" and snap["tmp_files"] == 2 * GiB
