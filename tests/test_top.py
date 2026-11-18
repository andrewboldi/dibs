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


def _crowded(snapshot):
    """Issue #118: ids, a VRAM cell and a start time at their longest, and a reason too long to fit."""
    return {**snapshot, "jobs": [
        view(599, "cls-F-wd2", State.RUNNING, 4, int(2.5 * GiB), int(4.7 * GiB), 3600, started=NOW - 60,
             cpu_ids=[0, 1, 2, 3], gpu_ids=[0], expected_end=NOW + 4140, over_estimate=0),
        view(1234, "a-rather-long-job-name", State.QUEUED, 24, 12 * GiB, int(4.5 * GiB), 600, position=1,
             predicted_start=NOW + 4560, predicted_end=NOW + 5160,
             waiting_for="needs 24 CPUs (17 free), 12G RAM (172M free, 21.7G in use outside dibs)"),
    ]}  # fmt: skip


def test_columns_never_run_together(snapshot):
    text = render_top(_crowded(snapshot), width=200, height=40, color=False)
    assert re.search(r"#599 +cls-F-wd2", text)
    assert re.search(r"#1234 +a-rather-lon", text)
    assert re.search(r"4\.7G +ends in 1h9m", text)
    assert re.search(r"starts in 1h16m +needs 24 CPUs", text)


@pytest.mark.parametrize("color", [False, True])
def test_a_line_cut_at_the_edge_ends_with_an_ellipsis(snapshot, color):
    text = ANSI.sub("", render_top(_crowded(snapshot), width=100, height=40, color=color))
    row = next(line for line in text.splitlines() if "#1234" in line)
    assert len(row) <= 100 and row.endswith("…"), row
    assert not row.endswith(" …") and not row.endswith("……")
    for width in range(95, 140):  # wherever it's cut, never just after a space, comma or bracket
        text = render_top(_crowded(snapshot), width=width, height=40, color=False)
        cut = next(line for line in text.splitlines() if "#1234" in line)
        assert not re.search(r"[\s,(]…$", cut), cut


def test_sections_say_how_many_jobs_they_hold(snapshot):
    text = render_top(snapshot, width=120, height=40, color=False)
    assert re.search(r"^ RUNNING +2\b", text, re.M) and re.search(r"^ QUEUED +2\b", text, re.M)


def test_resource_numbers_line_up_under_their_labels(snapshot):
    lines = render_top(snapshot, width=120, height=40, color=False).splitlines()
    head = next(line for line in lines if line.startswith(" RUNNING"))
    ends = {label: re.search(rf"\b{label}\b", head).end() for label in ("CPUs", "RAM", "VRAM")}
    cells = {"train": ("4", "8G", "6G"), "tokenize": ("4", "4G", "-"), "eval": ("2", "4G", "8G"),
             "finetune": ("8", "16G", "8G")}  # fmt: skip
    for name, values in cells.items():
        row = next(line for line in lines if f" {name} " in line)
        for label, value in zip(("CPUs", "RAM", "VRAM"), values):
            assert row[ends[label] - len(value) : ends[label]] == value, (name, label, row)


def test_a_cut_session_label_never_ends_in_a_piece_of_its_session(snapshot):
    # One session often works in several project folders, so the project part is what tells rows
    # apart: cut from the end as before, but not to "project·…" or "project·42…".
    def shown(label):
        jobs = [{**snapshot["jobs"][0], "session_label": label}]
        return render_top({**snapshot, "jobs": jobs}, width=120, height=40, color=False)

    assert "project-ab…" in shown("project-ab·4249") and "·4" not in shown("project-ab·4249")
    assert "a-long-projec…" in shown("a-long-project-name·4249")
    assert "proj·s1" in shown("proj·s1")


def test_the_gpu_index_shows_only_when_there_are_several(snapshot):
    assert "6G@0" not in render_top(snapshot, width=120, height=40, color=False)
    gpus = snapshot["capacity"]["gpus"] + [{"index": 1, "uuid": "GPU-1", "name": "Fake GPU", "vram": 8 * GiB}]
    two = {**snapshot, "capacity": {**snapshot["capacity"], "gpus": gpus}}
    assert "6G@0" in render_top(two, width=120, height=40, color=False)


def test_plan_bars_start_under_now(snapshot):
    lines = render_top(snapshot, width=120, height=40, color=False).splitlines()
    axis = next(line for line in lines if line.startswith(" PLAN"))
    bar = next(line for line in lines[lines.index(axis) :] if " train " in line)
    assert bar.index("█") == axis.index("now")


def test_a_reason_shows_only_when_enough_of_it_fits(snapshot):
    def row(width):
        lines = render_top(snapshot, width=width, height=40, color=False).splitlines()
        return next(line for line in lines if " eval " in line)

    assert row(80).endswith("starts in 15m"), row(80)  # not a stub like "ne…"
    assert row(100).endswith("needs 8G VRAM (2G free)"), row(100)


def test_job_colours_are_fixed_hues_that_no_terminal_theme_turns_grey():
    # Issue #118: the 16 basic colours follow the terminal's theme, and some themes (Solarized)
    # make the bright ones greys, so several jobs looked alike. 256-colour hues don't change.
    from dibs.render import _PALETTE, _job_color

    assert all(re.fullmatch(r"38;5;\d+", c) for c in _PALETTE) and len(set(_PALETTE)) == len(_PALETTE)
    assert all(_job_color(i) != _job_color(i + 1) for i in range(100))


def test_a_coloured_dot_marks_each_job_and_its_text_stays_plain(snapshot):
    from dibs.render import _job_color

    lines = render_top(snapshot, width=120, height=40, color=True).splitlines()
    row = next(line for line in lines if " train " in ANSI.sub("", line))
    assert f"\x1b[{_job_color(1)}m●\x1b[0m #1 " in row
    assert f"\x1b[{_job_color(1)}m#1" not in row and f"\x1b[{_job_color(1)}mtrain" not in "".join(lines)


def test_neighbouring_jobs_in_a_bar_are_set_apart(snapshot):
    cpu = next(line for line in render_top(snapshot, width=120, height=40, color=False).splitlines()
               if line.startswith(" CPU"))  # fmt: skip
    assert re.search(r"█+▌█+░", cpu), cpu  # two jobs' segments, a half-cell gap between them


def _with_outside(snapshot, cpus=None, mem=0, vram=None):
    return {**snapshot, "outside": {"cpus": cpus, "mem": mem, "vram": vram or {}}}


def _bars(text):
    return {line.split()[0] + (" " + line.split()[1] if line.startswith(" GPU") else ""): line
            for line in text.splitlines() if line.startswith((" CPU", " RAM", " GPU"))}  # fmt: skip


def test_bars_show_what_programs_outside_dibs_use(snapshot):
    # Issue #120: top showed 7 of 24 CPUs reserved while a program outside dibs kept most cores busy.
    bars = _bars(render_top(_with_outside(snapshot, cpus=12.4, mem=20 * GiB, vram={"0": GiB}), width=120,
                            height=40, color=False))  # fmt: skip
    assert "▒" in bars["CPU"] and "8/24 · 12 busy outside dibs" in bars["CPU"], bars["CPU"]
    assert "▒" in bars["RAM"] and "12G/32G · 20G in use outside dibs" in bars["RAM"], bars["RAM"]
    assert "▒" in bars["GPU 0"] and "6G/8G · 1G in use outside dibs" in bars["GPU 0"], bars["GPU 0"]


def test_outside_use_fills_only_what_jobs_left_and_its_figure_stays_whole(snapshot):
    for width in (80, 100, 160):
        bars = _bars(render_top(_with_outside(snapshot, cpus=23.0, mem=30 * GiB), width=width, height=40, color=False))
        assert "░" not in bars["CPU"] and "23 busy outside dibs" in bars["CPU"], (width, bars["CPU"])
        assert "30G in use outside dibs" in bars["RAM"], (width, bars["RAM"])
        cells = [sum(bars[k].count(c) for c in "█▌▒░") for k in ("CPU", "RAM", "GPU 0")]
        assert len(set(cells)) == 1, (width, cells)  # the bars stay one width


def test_little_or_unknown_outside_use_draws_nothing(snapshot):
    plain = render_top(snapshot, width=120, height=40, color=False)
    for quiet in (_with_outside(snapshot), _with_outside(snapshot, cpus=0.3)):
        bars = _bars(render_top(quiet, width=120, height=40, color=False))
        assert "▒" not in bars["CPU"] and "outside" not in bars["CPU"] and "outside" not in bars["GPU 0"]
    assert "▒" not in plain and "outside dibs" not in "".join(_bars(plain).values())  # an older daemon
