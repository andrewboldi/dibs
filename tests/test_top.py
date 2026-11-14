import re
import subprocess
import sys

import pytest

from dibs.model import Job, Request, State, job_to_dict
from dibs.render import render_top
from dibs.units import GiB

NOW = 1_800_000_000.0
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def view(job_id, name, state, cpus, mem, vram, est, **extra):
    job = Job(
        id=job_id,
        name=name,
        argv=["python", f"{name}.py"],
        cwd="/home/u/proj",
        env={},
        request=Request(cpus=cpus, mem=mem, gpus=1 if vram else 0, vram=vram, time=est),
        session=f"session-{job_id}",
        session_label=f"proj·s{job_id}",
        submitted=NOW - 300,
        tag=0.0,
        state=state,
        output="/tmp/out.log",
        started=extra.pop("started", None),
        cpu_ids=extra.pop("cpu_ids", []),
        gpu_ids=extra.pop("gpu_ids", []),
    )
    return {**job_to_dict(job), **extra}


@pytest.fixture
def snapshot():
    return {
        "now": NOW,
        "capacity": {
            "cpus": list(range(24)),
            "mem": 32 * GiB,
            "gpus": [{"index": 0, "uuid": "GPU-0", "name": "Fake GPU", "vram": 8 * GiB}],
        },
        "jobs": [
            view(1, "train", State.RUNNING, 4, 8 * GiB, 6 * GiB, 1500, started=NOW - 600,
                 cpu_ids=[0, 1, 2, 3], gpu_ids=[0], expected_end=NOW + 900, over_estimate=0),
            view(4, "tokenize", State.RUNNING, 4, 4 * GiB, 0, 300, started=NOW - 60,
                 cpu_ids=[4, 5, 6, 7], expected_end=NOW + 240, over_estimate=0),
            view(3, "eval", State.QUEUED, 2, 4 * GiB, 8 * GiB, 180, position=1,
                 predicted_start=NOW + 900, predicted_end=NOW + 1080,
                 waiting_for="needs 8G VRAM (2G free)"),
            view(2, "finetune", State.QUEUED, 8, 16 * GiB, 8 * GiB, 7200, position=2,
                 predicted_start=NOW + 1080, predicted_end=NOW + 8280, waiting_for="needs 8G VRAM (2G free)"),
        ],
        "guard": {
            "mode": "all",
            "stopped": [{"kind": "stop", "time": NOW - 120, "pid": 4121, "command": ["python", "x.py"],
                         "gpu": 0, "used": int(3.1 * GiB), "session": None}],
            "outside": [],
        },
    }  # fmt: skip


def test_render_top_shows_everything(snapshot):
    text = ANSI.sub("", render_top(snapshot, width=120, height=40, color=True))
    for word in ("CPU", "RAM", "GPU 0", "train", "tokenize", "eval", "finetune", "Stopped outside dibs"):
        assert word in text, word
    assert "8/24" in text  # CPUs reserved of total
    assert "6G/8G" in text  # VRAM reserved on GPU 0
    assert "starts in 15m" in text  # eval's predicted start
    assert "+" in text and "now" in text  # the timeline's axis


@pytest.mark.parametrize("width", [60, 100, 160])
def test_render_top_fits_width(snapshot, width):
    text = render_top(snapshot, width=width, height=40, color=False)
    assert max(len(line) for line in text.splitlines()) <= width
    colored = render_top(snapshot, width=width, height=40, color=True)
    assert max(len(ANSI.sub("", line)) for line in colored.splitlines()) <= width


def test_render_top_fits_height(snapshot):
    many = dict(snapshot)
    many["jobs"] = snapshot["jobs"] + [
        view(10 + i, f"job{i}", State.QUEUED, 1, GiB, 0, 60, position=3 + i,
             predicted_start=NOW + 100 * i, predicted_end=NOW + 100 * i + 60, waiting_for="")
        for i in range(40)
    ]  # fmt: skip
    text = render_top(many, width=100, height=30, color=False)
    assert len(text.splitlines()) <= 30
    assert "more" in text


def test_render_top_empty_queue(snapshot):
    empty = {**snapshot, "jobs": [], "guard": {"mode": "all", "stopped": [], "outside": []}}
    text = render_top(empty, width=80, height=24, color=False)
    assert "Nothing running or queued" in text


def test_top_without_tty_prints_once(dibs_env, tmp_path):
    result = subprocess.run([sys.executable, "-m", "dibs", "top"], env=dibs_env, cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    subprocess.run([sys.executable, "-m", "dibs", "daemon", "stop"], env=dibs_env, timeout=60)
    assert result.returncode == 0
    assert "GPU 0" in result.stdout and "\x1b[?1049h" not in result.stdout


def test_short_plans_get_second_ticks(snapshot):
    short = {**snapshot, "jobs": [
        view(1, "train", State.RUNNING, 4, 8 * GiB, 6 * GiB, 25, started=NOW - 5,
             cpu_ids=[0, 1, 2, 3], gpu_ids=[0], expected_end=NOW + 20, over_estimate=0),
        view(2, "finetune", State.QUEUED, 8, 8 * GiB, 8 * GiB, 45, position=1,
             predicted_start=NOW + 20, predicted_end=NOW + 65, waiting_for=""),
    ]}  # fmt: skip
    text = render_top(short, width=110, height=30, color=False)
    assert "+10s" in text and "+30s" in text


def test_tick_labels_are_distinct(snapshot):
    longer = {**snapshot, "jobs": [
        view(1, "train", State.RUNNING, 4, 8 * GiB, 6 * GiB, 25, started=NOW - 5,
             cpu_ids=[0, 1, 2, 3], gpu_ids=[0], expected_end=NOW + 20, over_estimate=0),
        view(2, "finetune", State.QUEUED, 8, 8 * GiB, 8 * GiB, 60, position=1,
             predicted_start=NOW + 20, predicted_end=NOW + 80, waiting_for=""),
    ]}  # fmt: skip
    axis = next(l for l in render_top(longer, width=160, height=30, color=False).splitlines() if "PLAN" in l)
    labels = axis.split()[2:]
    assert len(labels) == len(set(labels)) and "+1m10s" in labels
