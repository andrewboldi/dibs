import pytest

from dibs.model import (
    Capacity,
    Gpu,
    Job,
    Request,
    RequestError,
    State,
    compute_tag,
    default_name,
    job_from_dict,
    job_to_dict,
    resolve_request,
)
from dibs.units import GiB

CAP = Capacity(cpus=tuple(range(24)), mem=30 * GiB, gpus=(Gpu(0, "GPU-a", "A", 8 * GiB),))
NO_GPU = Capacity(cpus=tuple(range(24)), mem=30 * GiB, gpus=())


def raw(**kw):
    base = {"cpus": 4, "mem": 16 * GiB, "vram": None, "gpus": None, "time": 1200.0}
    base.update(kw)
    return base


def test_tag_without_backlog():
    assert compute_tag(100.0, 60.0, []) == 160.0


def test_tag_chains_behind_session_backlog():
    assert compute_tag(100.0, 60.0, [500.0, 300.0]) == 560.0


def test_tag_ignores_backlog_in_the_past():
    assert compute_tag(1000.0, 60.0, [500.0]) == 1060.0


def test_user_example_tags():
    tag_a = compute_tag(0.0, 60.0, [])  # session A
    tag_b = compute_tag(0.0, 600.0, [])  # session B
    assert tag_a < tag_b


def test_flood_interleaves():
    flood = []  # (tag, submitted, id) for session s1
    for i in range(50):
        flood.append((compute_tag(0.0, 60.0, [t for t, _, _ in flood]), 0.0, i + 1))
    assert [t for t, _, _ in flood] == [60.0 * (i + 1) for i in range(50)]
    other = (compute_tag(0.0, 600.0, []), 0.0, 51)  # session s2
    assert other[0] == 600.0
    order = sorted(flood + [other])
    assert order.index(other) == 10


def test_resolve_basic():
    req = resolve_request(raw(vram=6 * GiB), CAP)
    assert req == Request(cpus=4, mem=16 * GiB, gpus=1, vram=6 * GiB, time=1200.0)


def test_resolve_all():
    req = resolve_request(raw(cpus="all", mem="all", vram="all"), CAP)
    assert req == Request(cpus=24, mem=30 * GiB, gpus=1, vram=8 * GiB, time=1200.0)


def test_resolve_no_gpu():
    req = resolve_request(raw(), CAP)
    assert (req.gpus, req.vram) == (0, 0)


@pytest.mark.parametrize(
    "request_fields,cap,message",
    [
        ({"vram": 12 * GiB}, CAP, "asks for 12G VRAM, but the largest GPU dibs manages has 8G"),
        ({"cpus": 32}, CAP, "asks for 32 CPUs, but dibs manages 24"),
        ({"mem": 64 * GiB}, CAP, "asks for 64G RAM, but dibs manages 30G"),
        ({"gpus": 2}, CAP, "--gpus needs --vram"),
        ({"vram": 1 * GiB}, NO_GPU, "this machine has no GPUs dibs can hand out"),
        ({"gpus": 2, "vram": 6 * GiB}, CAP, "asks for 2 GPUs, but dibs manages 1"),
        ({"time": 0}, CAP, "--time must be more than 0"),
        ({"cpus": 0}, CAP, "--cpus must be at least 1"),
        ({"mem": 0}, CAP, "--mem must be more than 0"),
    ],
)
def test_resolve_rejects(request_fields, cap, message):
    with pytest.raises(RequestError) as err:
        resolve_request(raw(**request_fields), cap)
    assert str(err.value) == message


def test_job_roundtrip():
    job = Job(
        id=7,
        name="train.py",
        argv=["python", "train.py", "--epochs", "3"],
        cwd="/home/u/proj",
        env={"PATH": "/usr/bin", "SECRET": "x"},
        request=Request(cpus=4, mem=16 * GiB, gpus=1, vram=6 * GiB, time=1200.0),
        session="abcd1234",
        session_label="proj·abcd",
        submitted=1000.0,
        tag=2200.0,
        state=State.RUNNING,
        output="/tmp/out.log",
        started=1001.0,
        cpu_ids=[0, 1, 2, 3],
        gpu_ids=[0],
        shim_pid=4242,
        shim_ticks=99,
        unit="dibs-abc-job-7",
        peak_vram=5 * GiB,
    )
    assert job_from_dict(job_to_dict(job, include_env=True)) == job
    assert "env" not in job_to_dict(job)
    assert job_to_dict(job)["state"] == "running"


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["python", "train.py", "--x"], "train.py"),
        (["bash", "-c", "echo hi"], "bash"),
        (["./run.sh"], "run.sh"),
        (["uv", "run", "python", "eval.py"], "eval.py"),
        (["/usr/bin/" + "x" * 40], "x" * 24),
    ],
)
def test_default_name(argv, expected):
    assert default_name(argv) == expected


@pytest.mark.parametrize("time", [float("inf"), float("nan"), -float("inf")])
def test_resolve_rejects_non_finite_time(time):
    with pytest.raises(RequestError) as err:
        resolve_request(raw(time=time), CAP)
    assert "--time" in str(err.value)


def test_priority_changes_how_long_a_job_counts_in_line():
    # Issue #47: a higher priority counts as a shorter job, a lower one as a longer job.
    assert compute_tag(100.0, 600.0, []) == 700.0  # 5, the default: as before
    assert compute_tag(100.0, 600.0, [], priority=10) == 160.0  # a tenth of its estimate
    assert compute_tag(100.0, 600.0, [], priority=1) == 2500.0  # four times it


def test_owed_time_is_added_and_credit_never_puts_a_job_before_its_submit_time():
    assert compute_tag(100.0, 600.0, [], owed=540.0) == 1240.0
    assert compute_tag(100.0, 60.0, [], owed=-3600.0) == 100.0  # no earlier than it was submitted


def test_what_a_job_leaves_its_sessions_next_job():
    from dibs.model import next_owed

    assert next_owed(600.0, 10) == 540.0  # the time it saved, paid back by the next job
    assert next_owed(600.0, 5) == 0.0
    assert next_owed(600.0, 1) == -1800.0  # the time it gave up, credited
    assert next_owed(7200.0, 1) == -3600.0  # at most an hour of credit
