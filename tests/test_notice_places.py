"""A job's notices go to commands from its own folder or git working tree: a coding agent's subagents
share its session id, and each was told the others' failures and advice (#221)."""

import time

from conftest import run_until
from test_advice import _runs, _script
from test_daemon import PY, _failed, finished, submit

FAIL = [PY, "-c", "import sys; sys.exit(3)"]


def _tree(tmp_path, name):
    """A git working tree of its own, as a subagent's worktree is (a .git file), with a subfolder."""
    root = tmp_path / name
    (root / "sub").mkdir(parents=True)
    (root / ".git").write_text("gitdir: /elsewhere\n")
    return root


def test_a_failed_jobs_notice_goes_to_commands_from_its_own_working_tree(daemon, tmp_path):
    mine, sibling = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    failing = submit(daemon, mine / "sub", FAIL, session="s1")
    run_until(daemon, lambda: finished(daemon, failing))
    now = time.time()
    assert _failed(daemon.notices("s1", now, cwd=str(sibling))) == []  # a sibling subagent's worktree
    assert [n["job"] for n in _failed(daemon.notices("s1", now, cwd=str(mine)))] == [failing]
    assert [n["job"] for n in _failed(daemon.notices("s1", now, cwd=str(mine / "sub")))] == [failing]
    assert [n["job"] for n in _failed(daemon.notices("s1", now))] == [failing]  # a CLI that sends no folder


def test_a_failure_nobody_in_its_folder_picks_up_goes_to_the_session_after_half_an_hour(make_daemon, tmp_path):
    # The subagent that submitted the job may have ended: its failure isn't lost.
    now = [time.time()]
    d = make_daemon(clock=lambda: now[0])
    mine, sibling = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    failing = submit(d, mine, FAIL, session="s1")
    run_until(d, lambda: finished(d, failing))
    ended = d.store.get_job(failing).ended
    assert _failed(d.notices("s1", ended + 1700, cwd=str(sibling))) == []
    assert [n["job"] for n in _failed(d.notices("s1", ended + 1900, cwd=str(sibling)))] == [failing]


def test_a_failure_its_own_folder_is_still_at_work_on_doesnt_go_to_siblings(make_daemon, tmp_path):
    now = [time.time()]
    d = make_daemon(clock=lambda: now[0])
    mine, sibling = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    failing = submit(d, mine, FAIL, session="s1")
    run_until(d, lambda: finished(d, failing))
    ended = d.store.get_job(failing).ended
    assert len(_failed(d.notices("s1", ended + 600, cwd=str(mine)))) == 1
    assert _failed(d.notices("s1", ended + 1900, cwd=str(sibling))) == []


def test_request_advice_goes_once_to_its_own_working_tree(make_daemon, tmp_path):
    # Told once, advice went to whichever subagent ran the session's next dibs command (#169, #221).
    d = make_daemon()
    mine, sibling = _tree(tmp_path, "a"), _tree(tmp_path, "b")
    script = _script(mine)
    _runs(d, mine, script, 5, ran=60.0)  # 5 runs that took a tenth of their estimates
    new = submit(d, mine, [PY, str(script), "9"], session="s1", time_=600.0)
    now = time.time()

    def advice(cwd):
        return [n for n in d.notices("s1", now, cwd=cwd)["notices"] if n["kind"] == "advice"]

    assert advice(str(sibling)) == []
    assert len(advice(str(mine / "sub"))) == 1
    assert advice(str(mine)) == []  # told once
    run_until(d, lambda: finished(d, new))
