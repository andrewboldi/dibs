import re
import subprocess
import sys

import pytest

from dibs.model import Job, Request, State, job_to_dict
from dibs.render import render_top, top_screen
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


def lines_of(snapshot, width=120, height=40, color=False, offset=0, names=None):
    text = render_top(snapshot, width=width, height=height, color=color, offset=offset, names=names)
    return ANSI.sub("", text).splitlines()


def is_header(line):
    """The columns' names (cpus, ram, vram): the one row under the meters that isn't a panel's."""
    return not line.startswith(("╭", "│", "╰")) and re.search(r"\bcpus\b", line) is not None


def axis_rows(lines):
    """The rows that carry the waiting jobs' time axis: "now", then "+5m" and the like."""
    return [line for line in lines if line.startswith("│") and " now " in line and not re.search("[●○]", line)]


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
    assert axis_rows(lines_of(snapshot))  # the plan's time axis


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
                if is_header(line):  # the columns' names, a plain row between the meters and the panels
                    continue
                assert len(line) == width, (width, color, line)
                assert (line[0], line[-1]) in {("╭", "╮"), ("│", "│"), ("╰", "╯")}, (width, color, line)


def _many(snapshot, n=40):
    return {**snapshot, "jobs": snapshot["jobs"] + [
        view(10 + i, f"job{i}", State.QUEUED, 1, GiB, 0, 60, position=3 + i,
             predicted_start=NOW + 100 * i, predicted_end=NOW + 100 * i + 60, waiting_for="")
        for i in range(n)
    ]}  # fmt: skip


def test_jobs_that_dont_fit_scroll_under_meters_that_stay_put(snapshot):
    # Issue #167: lists were cut to "N more"; now every job can be scrolled to, as in htop.
    many = _many(snapshot)
    for height in (24, 30):
        top_lines = lines_of(many, width=100, height=height)
        assert len(top_lines) == height and top_lines[0].startswith("╭─ dibs ")
        assert "↓" in top_lines[-1] and "below" in top_lines[-1] and "above" not in top_lines[-1], top_lines[-1]
        text, offset, page = top_screen(many, width=100, height=height, color=False, offset=10**6)
        bottom = ANSI.sub("", text).splitlines()
        assert offset > 0 and page > 0 and len(bottom) == height
        assert bottom[:5] == top_lines[:5]  # the meters stay at the top
        assert " job39 " in "\n".join(bottom) and "↑" in bottom[-1] and "above" in bottom[-1], bottom[-1]
        assert "below" not in bottom[-1]
        again, clamped, _ = top_screen(many, width=100, height=height, color=False, offset=offset + 50)
        assert clamped == offset and again == text  # scrolling past the end stays at the end


def test_scrolling_by_a_line_moves_the_panels_by_a_line(snapshot):
    many = _many(snapshot)
    first, _, _ = top_screen(many, width=100, height=24, color=False, offset=0)
    second, _, _ = top_screen(many, width=100, height=24, color=False, offset=1)
    first, second = first.splitlines(), second.splitlines()
    body = next(i for i, line in enumerate(first) if is_header(line)) + 1  # the panels start under the header
    assert second[body:-2] == first[body + 1 : -1]


def test_a_screen_that_fits_shows_no_scroll_line(snapshot):
    lines = lines_of(snapshot, height=60)
    assert not any("above" in line or "below" in line for line in lines)
    assert lines[-1].startswith("╰")


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
    (axis,) = axis_rows(lines_of(short, width=160, height=30))
    assert "+10s" in axis and "+30s" in axis, axis


def test_tick_labels_are_distinct(snapshot):
    longer = {**snapshot, "jobs": [
        view(1, "train", State.RUNNING, 4, 8 * GiB, 6 * GiB, 25, started=NOW - 5,
             cpu_ids=[0, 1, 2, 3], gpu_ids=[0], expected_end=NOW + 20, over_estimate=0),
        view(2, "finetune", State.QUEUED, 8, 8 * GiB, 8 * GiB, 60, position=1,
             predicted_start=NOW + 20, predicted_end=NOW + 80, waiting_for=""),
    ]}  # fmt: skip
    (axis,) = axis_rows(lines_of(longer, width=160, height=30))
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


def test_each_panel_says_how_many_of_its_agents_jobs_run_and_wait(snapshot):
    lines = lines_of(_two_agents(snapshot), names={"sess-a": "VLA Chess"})
    assert border(lines, "VLA Chess  2 running · 1 waiting ") and border(lines, "other·b  1 waiting ")


def test_resource_numbers_line_up_under_their_labels(snapshot):
    # Every agent has its own panel (each job in the fixture is its own session), and an agent's name
    # can fill its title, so the columns are named once, in a header row under the meters that
    # stays put while the panels scroll, as in htop; the columns line up across every panel.
    lines = lines_of(snapshot)
    head = next(line for line in lines if re.search(r"\bcpus\b", line) and not line.startswith(("╭", "│", "╰")))
    assert lines.index(head) == next(i for i, line in enumerate(lines) if line.startswith("╰")) + 1
    for offset in (0, 3):
        assert head in lines_of(_many(snapshot), width=120, height=24, offset=offset)
    ends = {label: re.search(rf"\b{label}\b", head).end() for label in ("cpus", "ram", "vram")}
    cells = {"train": ("4", "8G", "6G"), "tokenize": ("4", "4G", "-"), "eval": ("2", "4G", "8G"),
             "finetune": ("8", "16G", "8G")}  # fmt: skip
    for name, values in cells.items():
        row = next(line for line in lines if f" {name} " in line)
        for label, value in zip(("cpus", "ram", "vram"), values):
            assert row[ends[label] - len(value) : ends[label]] == value, (name, label, row)


def test_a_cut_session_label_never_ends_in_a_piece_of_its_session(snapshot):
    # A panel without its agent's own name is titled by dibs's label (project·session); cut to fit
    # a narrow terminal, it never ends in "project·…" or "project·42…", and its counts stay.
    def title(label, width):
        jobs = [{**snapshot["jobs"][0], "session_label": label}]
        return next(line for line in lines_of({**snapshot, "jobs": jobs}, width=width)[5:] if line.startswith("╭"))

    cut = title("a-long-project-name-here·4249", 36)
    assert "a-long-proj" in cut and "…" in cut and "·4" not in cut and "1 running" in cut, cut
    assert "proj·s1  1 running" in title("proj·s1", 120)


def test_the_agent_shows_once_in_its_panels_title_not_in_each_row(snapshot):
    for width in (90, 120, 160):
        lines = lines_of(_two_agents(snapshot), width=width)
        assert not any("other·b" in line or "proj·" in line for line in lines if line.startswith("│")), width
        assert sum(line.startswith("╭─ other·b") for line in lines) == 1, width


def _two_agents(snapshot):
    """train, tokenize and eval from one agent session; finetune from another."""
    one = {"session": "sess-a", "session_label": "proj·a"}
    jobs = [{**j, **one} for j in snapshot["jobs"][:3]] + [{**snapshot["jobs"][3], "session": "sess-b",
                                                            "session_label": "other·b"}]  # fmt: skip
    return {**snapshot, "jobs": jobs}


def test_each_agent_gets_a_panel_with_its_running_jobs_then_its_waiting_ones(snapshot):
    lines = lines_of(_two_agents(snapshot), names={"sess-a": "VLA Chess"})
    panels = [i for i, line in enumerate(lines) if line.startswith("╭─ ")][1:]  # after the machine panel
    titles = [lines[i] for i in panels]
    assert titles[0].startswith("╭─ VLA Chess ") and titles[1].startswith("╭─ other·b "), titles
    first = lines[panels[0] : panels[1]]
    order = [name for line in first for name in ("train", "tokenize", "now", "eval") if f" {name} " in line]
    assert order == ["train", "tokenize", "now", "eval"], order
    assert " finetune " in "\n".join(lines[panels[1] :])


def test_panels_keep_their_order_when_jobs_start(snapshot):
    # Panels go by each agent's oldest job, so they don't jump about as jobs start and end while
    # the view is read: b's waiting job is the oldest, and b stays first once it starts.
    a = {"session": "sess-a", "session_label": "proj·a"}
    running = [{**snapshot["jobs"][0], **a, "id": 5}, {**snapshot["jobs"][1], **a, "id": 6}]
    waiting = {**snapshot["jobs"][2], "id": 3, "session": "sess-b", "session_label": "other·b"}
    started = {**waiting, "state": "running", "started": NOW, "expected_end": NOW + 600, "over_estimate": 0,
               "cpu_ids": [8, 9]}  # fmt: skip
    for jobs in (running + [waiting], running + [started], [started] + running):
        titles = [line for line in lines_of({**snapshot, "jobs": jobs})[5:] if line.startswith("╭─ ")]
        assert titles[0].startswith("╭─ other·b ") and titles[1].startswith("╭─ proj·a "), titles


def test_bars_and_columns_line_up_across_agents_panels(snapshot):
    # The Gantt bars share one time scale, and the columns one grid, whichever panel a job is in.
    twin = {**snapshot["jobs"][3], "id": 7, "name": "finetune2", "session": "sess-c", "session_label": "third·c",
            "position": 3}  # fmt: skip
    lines = lines_of({**_two_agents(snapshot), "jobs": _two_agents(snapshot)["jobs"] + [twin]}, width=140)
    rows = [next(line for line in lines if f" {name} " in line) for name in ("finetune", "finetune2")]
    assert rows[0].index("━") == rows[1].index("━") and rows[0].rindex("━") == rows[1].rindex("━")
    assert rows[0].index("starts in") == rows[1].index("starts in")
    assert len({line.index(" now ") for line in axis_rows(lines)}) == 1 and len(axis_rows(lines)) == 3


@pytest.mark.parametrize("name", ["\x1b[31mred\x1b[0m", "日本語のセッション", "🚀 launch it", "x" * 200, "tab\there"])
def test_a_long_or_odd_session_name_keeps_the_borders_and_the_counts(snapshot, name):
    # Session names are whatever the user typed: escape codes are stripped, wide characters would
    # throw the borders out of line, and a name too long for the title is cut, never dropped.
    for width in (60, 100, 140):
        lines = lines_of(_two_agents(snapshot), width=width, names={"sess-a": name})
        assert all(len(line) == width for line in lines if not is_header(line)), (width, name)
        title = next(line for line in lines[5:] if line.startswith("╭─ "))
        assert "2 running · 1 waiting" in title and "\x1b" not in title and "\t" not in title, (width, title)


def test_the_gpu_index_shows_only_when_there_are_several(snapshot):
    assert "6G@0" not in "\n".join(lines_of(snapshot))
    gpus = snapshot["capacity"]["gpus"] + [{"index": 1, "uuid": "GPU-1", "name": "Fake GPU", "vram": 8 * GiB}]
    two = {**snapshot, "capacity": {**snapshot["capacity"], "gpus": gpus}}
    assert "6G@0" in "\n".join(lines_of(two)) and "gpu 1" in "\n".join(lines_of(two))


def test_waiting_bars_start_under_now(snapshot):
    jobs = snapshot["jobs"][:2] + [view(5, "next-up", State.QUEUED, 1, GiB, 0, 600, position=1,
                                        predicted_start=NOW, predicted_end=NOW + 600, waiting_for="")]  # fmt: skip
    lines = lines_of({**snapshot, "jobs": jobs})
    at = next(i for i, line in enumerate(lines) if " next-up " in line)
    axis = next(lines[i] for i in range(at - 1, -1, -1) if lines[i] in axis_rows(lines))  # its panel's axis
    assert lines[at].index("━") == axis.index(" now ") + 1


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


def test_job_names_get_the_room_the_long_reasons_had(snapshot):
    # Issue #143: names were cut to 13 characters, so eval-model_small and eval-model_base both read
    # "eval-model_…", while a waiting job's reason filled the rest of its row.
    jobs = [{**snapshot["jobs"][2], "name": "eval-model_small", "lacks": [], "behind": 2,
             "waiting_for": "would delay #2, which is ahead in line"},
            {**snapshot["jobs"][3], "name": "eval-model_base", "lacks": ["RAM", "VRAM"], "behind": None,
             "waiting_for": "needs 16G RAM (220M free), 8G VRAM (2G free)"}]  # fmt: skip
    lines = lines_of({**snapshot, "jobs": snapshot["jobs"][:2] + jobs}, width=120)
    small, base = inside(lines, "eval-model_small").rstrip(), inside(lines, "eval-model_base").rstrip()
    assert small.endswith("behind #2") and base.endswith("needs RAM, VRAM")
    assert "which is ahead in line" not in small and "free" not in base


def test_a_name_too_long_for_the_room_is_cut_and_rows_still_fit(snapshot):
    jobs = [{**snapshot["jobs"][2], "name": "x" * 60, "lacks": ["VRAM"], "behind": None}]
    for width in (60, 80, 100, 120, 160):
        lines = lines_of({**snapshot, "jobs": snapshot["jobs"][:2] + jobs}, width=width)
        assert all(len(line) <= width for line in lines), width
        if width >= 160:  # room for all of it: a title of up to 80 characters gets the room (#263)
            assert "x" * 60 + " " in inside(lines, "xxxxx"), width
        else:
            assert "x" * (20 if width >= 120 else 12) + "…" in inside(lines, "xxxxx"), width


def test_top_reads_keys_as_htop_does():
    from dibs.cli import _top_keys

    assert _top_keys(b"q") == (["quit"], b"") and _top_keys(b"\x03") == (["quit"], b"")
    assert _top_keys(b"\x1b[A\x1b[B\x1bOAjk") == (["up", "down", "up", "down", "up"], b"")
    assert _top_keys(b"\x1b[5~\x1b[6~ ") == (["page_up", "page_down", "page_down"], b"")
    assert _top_keys(b"\x1b[H\x1b[F\x1b[1~\x1b[4~\x1bOH\x1bOFgG")[0] == ["home", "end"] * 4
    assert _top_keys(b"\x1b[<64;10;5M\x1b[<65;10;5M") == (["wheel_up", "wheel_down"], b"")
    assert _top_keys(b"\x1b[M`!!\x1b[Ma!!") == (["wheel_up", "wheel_down"], b"")  # the older mouse encoding
    assert _top_keys(b"\x1b[<0;10;5M\x1b[<0;10;5mx\x1b[C") == ([], b"")  # clicks and other keys do nothing
    assert _top_keys(b"\x1b[5") == ([], b"\x1b[5")  # half a sequence waits for the rest of it
    assert _top_keys(b"\x1b[5" + b"~q") == (["page_up", "quit"], b"")
    assert _top_keys(b"\x1bq") == (["quit"], b"")  # a lone Escape is dropped, not held forever


def test_top_scrolls_and_quits_in_a_terminal(dibs_env, tmp_path):
    # The real loop in a pseudo-terminal: it turns the mouse wheel on, takes PgDn, quits on q, and
    # puts the terminal back as it found it, mouse reporting off included.
    import fcntl
    import os
    import pty
    import select
    import struct
    import termios
    import time

    fd, terminal = pty.openpty()
    fcntl.ioctl(terminal, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 100, 0, 0))
    top = subprocess.Popen([sys.executable, "-m", "dibs", "top"], stdin=terminal, stdout=terminal, stderr=terminal,
                           env=dibs_env, cwd=tmp_path, start_new_session=True)  # fmt: skip
    os.close(terminal)
    out = b""

    def read_until(pattern, seconds):
        nonlocal out
        end = time.monotonic() + seconds
        while pattern not in out and time.monotonic() < end:
            if select.select([fd], [], [], 0.1)[0]:
                try:
                    out += os.read(fd, 65536)
                except OSError:
                    return

    try:
        read_until(b"guard", 30)  # the first frame's machine panel
        os.write(fd, b"\x1b[6~")
        read_until(b"\x00never", 1)
        os.write(fd, b"q")
        read_until(b"\x1b[?1049l", 15)
        code = top.wait(timeout=15)
    finally:
        top.kill()
        os.close(fd)
        subprocess.run([sys.executable, "-m", "dibs", "daemon", "stop"], env=dibs_env, timeout=60)
    assert code == 0, out[-500:]
    for sequence in (b"\x1b[?1049h", b"\x1b[?1000h", b"\x1b[?1006h", b"\x1b[?1000l", b"\x1b[?1006l",
                     b"\x1b[?1049l", b"\x1b[?25h"):  # fmt: skip
        assert sequence in out, sequence
