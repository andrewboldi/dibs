import json
import os
import stat

import pytest

from dibs.model import Job, Request, State
from dibs.store import Store
from dibs.units import GiB


def make_job(**kw):
    fields = dict(
        id=0,
        name="train.py",
        argv=["python", "train.py"],
        cwd="/tmp",
        env={"API_KEY": "secret"},
        request=Request(cpus=2, mem=4 * GiB, gpus=0, vram=0, time=60.0),
        session="s1",
        session_label="proj·s1",
        submitted=100.0,
        tag=160.0,
        state=State.QUEUED,
        output="/tmp/out.log",
    )
    fields.update(kw)
    return Job(**fields)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "dibs.db")
    yield s
    s.close()


def test_store_roundtrip(store):
    job = make_job()
    job_id = store.add_job(job)
    assert job_id == 1 and job.id == 1
    assert store.get_job(1) == job
    job.state, job.started, job.cpu_ids = State.RUNNING, 101.0, [0, 1]
    store.update_job(job)
    assert store.get_job(1) == job
    assert store.get_job(99) is None


def test_store_filters_by_state(store):
    for state in (State.QUEUED, State.RUNNING, State.DONE, State.QUEUED):
        store.add_job(make_job(state=state))
    assert [j.id for j in store.jobs([State.QUEUED])] == [1, 4]
    assert [j.id for j in store.jobs([State.RUNNING, State.DONE])] == [2, 3]
    assert [j.id for j in store.jobs()] == [1, 2, 3, 4]


def test_finished_before(store):
    store.add_job(make_job(state=State.DONE, ended=50.0))
    store.add_job(make_job(state=State.FAILED, ended=500.0))
    store.add_job(make_job(state=State.RUNNING))
    assert [j.id for j in store.finished_before(100.0)] == [1]


def test_ids_not_reused_after_delete(store):
    store.add_job(make_job())
    store.add_job(make_job())
    store.delete_jobs([1, 2])
    assert store.add_job(make_job()) == 3
    assert store.jobs() and store.get_job(1) is None


def test_meta(store):
    assert store.get_meta("guard_since") is None
    store.set_meta("guard_since", "123.5")
    store.set_meta("guard_since", "124.5")
    assert store.get_meta("guard_since") == "124.5"


def test_events_since(store):
    store.add_event("stop", {"pid": 1, "command": "python a.py"}, 10.0)
    store.add_event("stop", {"pid": 2, "command": "python b.py"}, 20.0)
    assert store.events(15.0) == [{"kind": "stop", "time": 20.0, "pid": 2, "command": "python b.py", "event_id": 2}]
    store.delete_events_before(25.0)
    assert store.events(0.0) == []


def test_db_file_private(tmp_path):
    s = Store(tmp_path / "dibs.db")
    s.add_job(make_job())
    s.close()
    assert stat.S_IMODE(os.stat(tmp_path / "dibs.db").st_mode) == 0o600


def test_store_survives_reopen(tmp_path):
    s = Store(tmp_path / "dibs.db")
    job = make_job()
    s.add_job(job)
    s.set_meta("k", "v")
    s.close()
    again = Store(tmp_path / "dibs.db")
    assert again.get_job(1) == job and again.get_meta("k") == "v"
    again.close()


def test_finished_jobs_are_found_in_sql_not_by_decoding_them_all(tmp_path, monkeypatch):
    # Issue #6: queue --all and the hourly cleanup decoded every finished job (environment and
    # all) just to look at when it ended.
    from dibs import store as store_module

    s = Store(tmp_path / "dibs.db")
    for i in range(50):
        s.add_job(make_job(state=State.DONE, ended=100.0 + i))
    s.add_job(make_job(state=State.DONE, ended=10_000.0))
    s.add_job(make_job(state=State.RUNNING))
    decoded = []
    real = store_module.job_from_dict
    monkeypatch.setattr(store_module, "job_from_dict", lambda d: decoded.append(d["id"]) or real(d))
    assert [j.id for j in s.finished_since(5_000.0)] == [51]
    assert decoded == [51]
    decoded.clear()
    assert [j.id for j in s.finished_before(102.0)] == [1, 2]
    assert decoded == [1, 2]
    s.close()


def test_a_database_from_before_the_ended_column_keeps_working(tmp_path):
    import json
    import sqlite3

    from dibs.model import job_to_dict

    path = tmp_path / "dibs.db"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, state TEXT NOT NULL, data TEXT NOT NULL);"
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, time REAL NOT NULL, data TEXT NOT NULL);"
    )
    job = make_job(id=1, state=State.DONE, ended=50.0)
    old.execute("INSERT INTO jobs (id, state, data) VALUES (1, 'done', ?)", (json.dumps(job_to_dict(job, include_env=True)),))
    old.commit()
    old.close()
    s = Store(path)
    assert [j.id for j in s.finished_before(60.0)] == [1]
    assert s.get_job(1) == job
    s.close()


def test_earlier_runs_of_a_command_are_found_in_sql(tmp_path, monkeypatch):
    # Issue #31: submitting a command shows what its earlier runs in the same folder used.
    from dibs import store as store_module

    s = Store(tmp_path / "dibs.db")
    same = dict(cwd="/w", argv=["python", "train.py"])
    s.add_job(make_job(state=State.DONE, ended=100.0, **same))  # 1
    s.add_job(make_job(state=State.FAILED, ended=200.0, **same))  # 2
    s.add_job(make_job(state=State.RUNNING, **same))  # 3: not finished
    s.add_job(make_job(state=State.DONE, ended=300.0, cwd="/w", argv=["python", "eval.py"]))  # 4
    s.add_job(make_job(state=State.DONE, ended=400.0, cwd="/other", argv=["python", "train.py"]))  # 5
    decoded = []
    real = store_module.job_from_dict
    monkeypatch.setattr(store_module, "job_from_dict", lambda d: decoded.append(d["id"]) or real(d))
    assert [j.id for j in s.earlier_runs("/w", ["python", "train.py"])] == [2, 1]
    assert sorted(decoded) == [1, 2]
    assert [j.id for j in s.earlier_runs("/w", ["python", "train.py"], limit=1)] == [2]
    s.close()


def test_a_database_from_before_the_command_column_finds_earlier_runs(tmp_path):
    import sqlite3

    from dibs.model import job_to_dict

    path = tmp_path / "dibs.db"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, state TEXT NOT NULL, data TEXT NOT NULL, ended REAL);"
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, time REAL NOT NULL, data TEXT NOT NULL);"
    )
    job = make_job(id=1, state=State.DONE, ended=50.0)
    old.execute("INSERT INTO jobs (id, state, data, ended) VALUES (1, 'done', ?, 50.0)",
                (json.dumps(job_to_dict(job, include_env=True)),))  # fmt: skip
    old.commit()
    old.close()
    s = Store(path)
    assert [j.id for j in s.earlier_runs(job.cwd, job.argv)] == [1]
    assert [j.id for j in s.earlier_runs_of_script(job.cwd, "train.py")] == [1]  # make_job runs python train.py
    assert [j.id for j in s.earlier_runs_in_session(job.session, "train.py", since=0.0)] == [1]
    s.close()


def test_earlier_runs_of_the_same_script_are_found_in_sql(tmp_path):
    s = Store(tmp_path / "dibs.db")
    s.add_job(make_job(state=State.DONE, ended=100.0, cwd="/w", argv=["python", "train.py", "--lr", "1"]))
    s.add_job(make_job(state=State.DONE, ended=200.0, cwd="/w", argv=["python", "train.py", "--lr", "2"]))
    s.add_job(make_job(state=State.DONE, ended=300.0, cwd="/w", argv=["python", "eval.py"]))
    s.add_job(make_job(state=State.DONE, ended=400.0, cwd="/x", argv=["python", "train.py"]))
    assert [j.id for j in s.earlier_runs_of_script("/w", "train.py")] == [2, 1]
    s.close()


def test_a_sessions_runs_of_a_script_of_the_same_name_are_found_in_any_folder(tmp_path):
    # Issue #229: agents that submit each batch from a fresh copy of their code had no earlier runs.
    s = Store(tmp_path / "dibs.db")
    s.add_job(make_job(state=State.DONE, ended=100.0, cwd="/copy1", argv=["python", "/copy1/tools/train.py"]))  # 1
    s.add_job(make_job(state=State.DONE, ended=200.0, cwd="/copy2", argv=["python", "tools/train.py", "--lr", "2"]))  # 2
    s.add_job(make_job(state=State.DONE, ended=300.0, cwd="/copy3", argv=["python", "train.py"], session="s2"))  # 3
    s.add_job(make_job(state=State.DONE, ended=50.0, cwd="/copy4", argv=["python", "train.py"]))  # 4: too old
    s.add_job(make_job(state=State.RUNNING, cwd="/copy5", argv=["python", "train.py"]))  # 5: not finished
    s.add_job(make_job(state=State.DONE, ended=400.0, cwd="/copy1", argv=["python", "eval.py"]))  # 6
    s.add_job(make_job(state=State.DONE, ended=500.0, cwd="/w", argv=["python", "-m", "pkg.run", "x"]))  # 7
    assert [j.id for j in s.earlier_runs_in_session("s1", "train.py", since=60.0)] == [2, 1]
    assert [j.id for j in s.earlier_runs_in_session("s1", "train.py", since=60.0, limit=1)] == [2]
    assert [j.id for j in s.earlier_runs_in_session("s1", "-m pkg.run", since=0.0)] == [7]
    s.close()
