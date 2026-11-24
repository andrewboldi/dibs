"""Each job carries a title: what it does, in plain words, for the person watching dibs top, where the
job's name (a script, or a label like a CI run's commit hash) says nothing (#259). Agents are asked to
give one with every job; dibs top and dibs queue show it, falling back to the name."""

import json

from test_cli import RES, dibs  # noqa: F401 (dibs is a fixture)
from test_daemon import PY, job, submit
from test_top import lines_of, snapshot, view  # noqa: F401 (snapshot is a fixture)

from dibs.instructions import AGENT_HELP, BLOCK, SKILL
from dibs.model import State
from dibs.render import render_queue, render_show, render_submit
from dibs.units import GiB


def test_a_job_keeps_the_title_it_was_given(daemon, tmp_path):
    j = submit(daemon, tmp_path, [PY, "-c", "pass"], title="  Train the chess\tmodel,\n3 epochs ")
    assert job(daemon, j).title == "Train the chess model, 3 epochs"
    assert job(daemon, submit(daemon, tmp_path, [PY, "-c", "pass"])).title is None


def test_a_title_past_80_characters_is_refused(daemon, tmp_path):
    from test_daemon import req

    resp = daemon.submit(req(tmp_path, [PY, "-c", "pass"], title="x" * 81))
    assert not resp["ok"] and "--title" in resp["error"]


def test_a_replacement_keeps_the_title_unless_it_gives_its_own(daemon, tmp_path):
    blocker = submit(daemon, tmp_path, [PY, "-c", "import time; time.sleep(30)"], cpus=len(daemon.capacity.cpus))
    waiting = submit(daemon, tmp_path, [PY, "-c", "pass"], title="Render the demo video")
    kept = submit(daemon, tmp_path, [PY, "-c", "pass"], replace=waiting)
    assert job(daemon, kept).title == "Render the demo video"
    renamed = submit(daemon, tmp_path, [PY, "-c", "pass"], replace=kept, title="Render the demo video, 4K")
    assert job(daemon, renamed).title == "Render the demo video, 4K"
    daemon.cancel([blocker, renamed])


def test_dibs_top_shows_a_jobs_title_and_a_name_only_without_one(snapshot):
    snapshot["jobs"][0]["name"] = "localci-c3139972-s1"
    snapshot["jobs"][0]["title"] = "Run the robot simulator tests for one commit"
    text = "\n".join(lines_of(snapshot, width=160))
    assert "Run the robot simulator tests for one commit" in text and "localci-c3139972-s1" not in text
    assert "tokenize" in text  # no title: its name


def test_dibs_queue_shows_the_title():
    jobs = [view(7, "v2-chain", State.QUEUED, 2, 10 * GiB, 0, 3600, position=1, predicted_start=None,
                 waiting_for="needs 10G RAM", title="Check the planner on 200 hard tasks"),
            view(8, "h141-probe", State.QUEUED, 1, GiB, 0, 600, position=2, predicted_start=None, waiting_for="")]  # fmt: skip
    text = render_queue({"now": 0.0, "jobs": jobs}, color=False)
    assert "Check the planner on 200 hard tasks" in text and "v2-chain" not in text and "h141-probe" in text


def test_dibs_show_lists_the_title():
    shown = render_show(view(7, "v2-chain", State.QUEUED, 2, 10 * GiB, 0, 3600, position=1, predicted_start=None,
                             waiting_for="needs 10G RAM", title="Check the planner on 200 hard tasks"), 0.0)  # fmt: skip
    assert shown.startswith("Job 7  v2-chain  queued")
    (row,) = [line for line in shown.splitlines() if line.split()[0] == "title"]
    assert row.endswith("  Check the planner on 200 hard tasks")


def test_a_submit_without_a_title_asks_for_one_next_time():
    base = view(7, "v2-chain", State.QUEUED, 2, 10 * GiB, 0, 3600, position=1, predicted_start=None,
                waiting_for="needs 10G RAM")  # fmt: skip
    assert "--title" in render_submit(base, 0.0)
    assert "--title" not in render_submit({**base, "title": "Check the planner on 200 hard tasks"}, 0.0)


def test_the_instructions_ask_for_a_title_on_every_job():
    for text in (BLOCK, AGENT_HELP, SKILL):
        flat = " ".join(text.split())
        assert 'dibs submit --title "' in flat, text[:40]
    assert "Give every job a --title" in " ".join(BLOCK.split())


def test_the_cli_sends_the_title(dibs):
    dibs("submit", *RES, "--title", "Sleep for half a minute", "--", "sleep", "30", check=0)
    snap = json.loads(dibs("queue", "--json", check=0).stdout)
    assert [j["title"] for j in snap["jobs"]] == ["Sleep for half a minute"]
    dibs("cancel", str(snap["jobs"][0]["id"]))
