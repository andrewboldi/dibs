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
    assert store.events(15.0) == [{"kind": "stop", "time": 20.0, "pid": 2, "command": "python b.py"}]
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
