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


def lines_of(snapshot, width=120, height=40, color=False):
    return ANSI.sub("", render_top(snapshot, width=width, height=height, color=color)).splitlines()


def inside(lines, text):
    """The inside of the first panel row that holds text, between "│ " and " │"."""
    return next(line for line in lines if line.startswith("│") and text in line)[2:-2]


def border(lines, title):
    return next(line for line in lines if line.startswith("╭─ " + title))


def meters(lines):
    return {line[2:7].strip(): line for line in lines if line.startswith(("│ cpu", "│ mem", "│ gpu"))}


def test_render_top_shows_everything(snapshot):
    text = "\n".join(lines_of(snapshot))
    for word in ("cpu", "mem", "gpu", "train", "tokenize", "eval", "finetune", "Stopped outside dibs", "Fake GPU",
                 "guard all"):  # fmt: skip
        assert word in text, word
    assert "8/24" in text  # CPUs reserved of total
    assert "6G/8G" in text  # VRAM reserved on the GPU
    assert "starts in 15m" in text  # eval's predicted start
    assert "now" in border(lines_of(snapshot), "waiting")  # the plan's time axis


@pytest.mark.parametrize("width", [60, 100, 160])
def test_render_top_fits_width(snapshot, width):
    text = render_top(snapshot, width=width, height=40, color=False)
    assert max(len(line) for line in text.splitlines()) <= width
    colored = render_top(snapshot, width=width, height=40, color=True)
    assert max(len(ANSI.sub("", line)) for line in colored.splitlines()) <= width


def test_every_panel_line_spans_the_width_between_its_borders(snapshot):
    # Issue #122: a border that drifts by a column is the first thing anyone sees.
    crowded = {**_crowded(snapshot), "guard": snapshot["guard"],
               "outside": {"cpus": 17.6, "mem": 21 * GiB, "vram": {"0": GiB}},
               "load": {"every": 4.0, "cores": [3.0, 12.0, 24.0, None, 8.0]}}  # fmt: skip
    for width in range(40, 161, 3):
        for color in (False, True):
            for line in lines_of(crowded, width=width, color=color):
                assert len(line) == width, (width, color, line)
                assert (line[0], line[-1]) in {("╭", "╮"), ("│", "│"), ("╰", "╯")}, (width, color, line)


def test_render_top_fits_height(snapshot):
    many = dict(snapshot)
    many["jobs"] = snapshot["jobs"] + [
        view(10 + i, f"job{i}", State.QUEUED, 1, GiB, 0, 60, position=3 + i,
             predicted_start=NOW + 100 * i, predicted_end=NOW + 100 * i + 60, waiting_for="")
        for i in range(40)
    ]  # fmt: skip
    for height in (24, 30):
        lines = lines_of(many, width=100, height=height)
        assert len(lines) <= height
        assert "more" in "\n".join(lines)
        assert sum(line.startswith("╭") for line in lines) == sum(line.startswith("╰") for line in lines)


def test_render_top_empty_queue(snapshot):
    empty = {**snapshot, "jobs": [], "guard": {"mode": "all", "stopped": [], "outside": []}}
    assert "Nothing running or waiting" in "\n".join(lines_of(empty, width=80, height=24))


def test_top_without_tty_prints_once(dibs_env, tmp_path):
    result = subprocess.run([sys.executable, "-m", "dibs", "top"], env=dibs_env, cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    subprocess.run([sys.executable, "-m", "dibs", "daemon", "stop"], env=dibs_env, timeout=60)
    assert result.returncode == 0
    assert "╭─ dibs" in result.stdout and "│ gpu" in result.stdout and "\x1b[?1049h" not in result.stdout


def test_short_plans_get_second_ticks(snapshot):
    short = {**snapshot, "jobs": [
        view(1, "train", State.RUNNING, 4, 8 * GiB, 6 * GiB, 25, started=NOW - 5,
             cpu_ids=[0, 1, 2, 3], gpu_ids=[0], expected_end=NOW + 20, over_estimate=0),
        view(2, "finetune", State.QUEUED, 8, 8 * GiB, 8 * GiB, 45, position=1,
             predicted_start=NOW + 20, predicted_end=NOW + 65, waiting_for=""),
    ]}  # fmt: skip
    axis = border(lines_of(short, width=160, height=30), "waiting")
    assert "+10s" in axis and "+30s" in axis, axis


def test_tick_labels_are_distinct(snapshot):
    longer = {**snapshot, "jobs": [
        view(1, "train", State.RUNNING, 4, 8 * GiB, 6 * GiB, 25, started=NOW - 5,
             cpu_ids=[0, 1, 2, 3], gpu_ids=[0], expected_end=NOW + 20, over_estimate=0),
        view(2, "finetune", State.QUEUED, 8, 8 * GiB, 8 * GiB, 60, position=1,
             predicted_start=NOW + 20, predicted_end=NOW + 80, waiting_for=""),
    ]}  # fmt: skip
    axis = border(lines_of(longer, width=160, height=30), "waiting")
    assert re.search(r"\+\S+ ─", axis), axis  # the border goes on right after the last label
    labels = re.findall(r"\+\S+", axis)
    assert len(labels) == len(set(labels)) and any(re.fullmatch(r"\+1m\d+s", label) for label in labels), labels


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
    text = "\n".join(lines_of(_crowded(snapshot), width=200))
    assert re.search(r"\b599 +cls-F-wd2", text)
    assert re.search(r"\b1234 +a-rather-lon", text)
    assert re.search(r"4\.7G +ends in 1h9m", text)
    assert re.search(r"starts in 1h16m +━+ +needs 24 CPUs", text)


@pytest.mark.parametrize("color", [False, True])
def test_a_line_cut_at_the_edge_ends_with_an_ellipsis(snapshot, color):
    cut = inside(lines_of(_crowded(snapshot), width=100, color=color), "1234").rstrip()
    assert cut.endswith("…") and not cut.endswith(" …") and not cut.endswith("……"), cut
    for width in range(95, 140):  # wherever it's cut, never just after a space, comma or bracket
        cut = inside(lines_of(_crowded(snapshot), width=width), "1234").rstrip()
        assert not re.search(r"[\s,(]…$", cut), cut


def test_sections_say_how_many_jobs_they_hold(snapshot):
    lines = lines_of(snapshot)
    assert border(lines, "running 2 ") and border(lines, "waiting 2 ")


def test_resource_numbers_line_up_under_their_labels(snapshot):
    lines = lines_of(snapshot)
    head = border(lines, "running")
    ends = {label: re.search(rf"\b{label}\b", head).end() for label in ("cpus", "ram", "vram")}
    cells = {"train": ("4", "8G", "6G"), "tokenize": ("4", "4G", "-"), "eval": ("2", "4G", "8G"),
             "finetune": ("8", "16G", "8G")}  # fmt: skip
    for name, values in cells.items():
        row = next(line for line in lines if f" {name} " in line)
        for label, value in zip(("cpus", "ram", "vram"), values):
            assert row[ends[label] - len(value) : ends[label]] == value, (name, label, row)


def test_a_cut_session_label_never_ends_in_a_piece_of_its_session(snapshot):
    # One session often works in several project folders, so the project part is what tells rows
    # apart: cut from the end as before, but not to "project·…" or "project·42…".
    def shown(label):
        jobs = [{**snapshot["jobs"][0], "session_label": label}]
        return "\n".join(lines_of({**snapshot, "jobs": jobs}))

    assert "project-ab…" in shown("project-ab·4249") and "·4" not in shown("project-ab·4249")
    assert "a-long-projec…" in shown("a-long-project-name·4249")
    assert "proj·s1" in shown("proj·s1")


def test_sessions_show_on_wide_terminals_only(snapshot):
    assert "proj·s1" in "\n".join(lines_of(snapshot, width=120))
    assert "proj·s1" not in "\n".join(lines_of(snapshot, width=90))


def test_the_gpu_index_shows_only_when_there_are_several(snapshot):
    assert "6G@0" not in "\n".join(lines_of(snapshot))
    gpus = snapshot["capacity"]["gpus"] + [{"index": 1, "uuid": "GPU-1", "name": "Fake GPU", "vram": 8 * GiB}]
    two = {**snapshot, "capacity": {**snapshot["capacity"], "gpus": gpus}}
    assert "6G@0" in "\n".join(lines_of(two)) and "gpu 1" in "\n".join(lines_of(two))


def test_waiting_bars_start_under_now(snapshot):
    jobs = snapshot["jobs"][:2] + [view(5, "next-up", State.QUEUED, 1, GiB, 0, 600, position=1,
                                        predicted_start=NOW, predicted_end=NOW + 600, waiting_for="")]  # fmt: skip
    lines = lines_of({**snapshot, "jobs": jobs})
    bar = next(line for line in lines if " next-up " in line)
    assert bar.index("━") == border(lines, "waiting").index("now")


def test_a_reason_shows_with_enough_of_it_or_not_at_all(snapshot):
    for width in range(60, 101):
        row = inside(lines_of(snapshot, width=width), " eval ").rstrip()
        assert "needs" not in row or len(row[row.index("needs") :]) >= 10, (width, row)
    assert inside(lines_of(snapshot, width=140), " eval ").rstrip().endswith("needs 8G VRAM (2G free)")


def test_job_colours_are_fixed_hues_that_no_terminal_theme_turns_grey():
    # Issue #118: the 16 basic colours follow the terminal's theme, and some themes (Solarized)
    # make the bright ones greys, so several jobs looked alike. 256-colour hues don't change.
    from dibs.render import _PALETTE, _job_color

    assert all(re.fullmatch(r"38;5;\d+", c) for c in _PALETTE) and len(set(_PALETTE)) == len(_PALETTE)
    assert all(_job_color(i) != _job_color(i + 1) for i in range(100))


def test_a_coloured_dot_marks_each_job_and_its_text_stays_plain(snapshot):
    from dibs.render import _job_color

    raw = render_top(snapshot, width=120, height=40, color=True).splitlines()
    row = next(line for line in raw if " train " in ANSI.sub("", line))
    assert f"\x1b[{_job_color(1)}m●\x1b[0m 1 " in row
    assert f"\x1b[{_job_color(1)}mtrain" not in "".join(raw)


def test_neighbouring_jobs_in_a_bar_are_set_apart(snapshot):
    assert re.search(r"█+▌█+[▒░]", meters(lines_of(snapshot))["cpu"])  # a half-cell gap between two jobs


def _with_outside(snapshot, cpus=None, mem=0, vram=None):
    return {**snapshot, "outside": {"cpus": cpus, "mem": mem, "vram": vram or {}}}


def test_bars_show_what_programs_outside_dibs_use(snapshot):
    # Issue #120: top showed 7 of 24 CPUs reserved while a program outside dibs kept most cores busy.
    bars = meters(lines_of(_with_outside(snapshot, cpus=12.4, mem=20 * GiB, vram={"0": GiB})))
    assert "▒" in bars["cpu"] and "8/24" in bars["cpu"] and "12 busy outside" in bars["cpu"], bars["cpu"]
    assert "▒" in bars["mem"] and "12G/32G" in bars["mem"] and "20G used outside" in bars["mem"], bars["mem"]
    assert "▒" in bars["gpu"] and "6G/8G" in bars["gpu"] and "1G used outside" in bars["gpu"], bars["gpu"]


def test_outside_use_fills_only_what_jobs_left_and_its_figure_stays_whole(snapshot):
    for width in (80, 100, 160):
        bars = meters(lines_of(_with_outside(snapshot, cpus=23.0, mem=30 * GiB), width=width))
        cpu = bars["cpu"]
        assert "░" not in cpu[cpu.index("▕") : cpu.index("▏")] and "23 busy outside" in cpu, (width, cpu)
        assert "30G used outside" in bars["mem"], (width, bars["mem"])
        cells = [line.index("▏") - line.index("▕") for line in bars.values()]
        assert len(set(cells)) == 1, (width, cells)  # the meters stay one width


def test_little_or_unknown_outside_use_draws_nothing(snapshot):
    for quiet in (snapshot, _with_outside(snapshot), _with_outside(snapshot, cpus=0.3)):  # first: an older daemon
        bars = meters(lines_of(quiet))
        assert "▒" not in bars["cpu"] and "outside" not in bars["cpu"] and "outside" not in bars["gpu"]
        assert "outside" not in bars["mem"] or quiet is not snapshot


def test_running_jobs_show_progress_against_their_estimate(snapshot):
    lines = lines_of(snapshot)
    assert "▰▰▰▰▱▱▱▱▱▱" in inside(lines, " train ")  # 600 s of a 1500 s estimate
    assert "▰▰▱▱▱▱▱▱▱▱" in inside(lines, " tokenize ")  # 60 s of 300 s


def test_a_job_past_its_estimate_fills_its_bar_and_says_by_how_much(snapshot):
    jobs = [{**snapshot["jobs"][0], "over_estimate": 120.0}] + snapshot["jobs"][1:]
    row = inside(lines_of({**snapshot, "jobs": jobs}), " train ")
    assert "over by 2m" in row and "▰" * 10 in row, row


def test_the_load_row_shows_the_last_minutes_newest_on_the_right(snapshot):
    load = {**snapshot, "load": {"every": 4.0, "cores": [1.0] * 10 + [24.0] * 5}}
    row = next(line for line in lines_of(load) if line.startswith("│ load"))
    assert re.search(r"▁+█{5} +busy cores, last 1m", row), row
    for none in (snapshot, {**snapshot, "load": {"every": 4.0, "cores": [3.0]}}):  # an older daemon, one point
        assert not any(line.startswith("│ load") for line in lines_of(none))


def test_the_guard_panel_shows_only_when_something_happened(snapshot):
    assert "Stopped outside dibs" in "\n".join(lines_of(snapshot)) and border(lines_of(snapshot), "guard")
    quiet = {**snapshot, "guard": {"mode": "all", "stopped": [], "outside": []}}
    assert not any(line.startswith("╭─ guard") for line in lines_of(quiet))


def test_the_machine_panel_names_the_gpu_and_the_guard_mode(snapshot):
    lines = lines_of(snapshot)
    bottom = next(line for line in lines if line.startswith("╰"))
    assert "Fake GPU" in bottom and "guard all" in bottom, bottom
    assert lines[0].startswith("╭─ dibs ")


def test_reasons_leave_out_what_the_mem_meter_already_says(snapshot):
    # Every waiting row repeated "17.9G in use outside dibs"; the machine panel shows it once.
    why = "needs 6G RAM (268M free, 17.9G in use outside dibs), 4G VRAM (2.3G free)"
    jobs = snapshot["jobs"][:2] + [{**snapshot["jobs"][2], "waiting_for": why}] + snapshot["jobs"][3:]
    row = inside(lines_of({**snapshot, "jobs": jobs}, width=160), " eval ")
    assert row.rstrip().endswith("needs 6G RAM (268M free), 4G VRAM (2.3G free)"), row
