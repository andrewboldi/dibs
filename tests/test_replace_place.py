"""A replacement that runs the same command and asks for no more keeps its place in line (#184)."""

from conftest import run_until
from test_daemon import PY, finished, job, req, submit

from dibs.units import GiB


def _line(make_daemon, tmp_path):
    """A job holding every CPU; then session a's job, then session b's, each waiting, a's ahead."""
    go = tmp_path / "go"
    holding = [PY, "-c", f"import os, time\nwhile not os.path.exists({str(go)!r}): time.sleep(0.02)"]
    d = make_daemon()
    blocker = submit(d, tmp_path, holding, cpus="all", session="s0")
    run_until(d, lambda: job(d, blocker).shim_pid)
    a = submit(d, tmp_path, [*holding, "a"], mem=8 * GiB, time_=600.0, session="sa")
    b = submit(d, tmp_path, [*holding, "b"], mem=8 * GiB, time_=600.0, session="sb")
    return d, go, a, b


def _replace(d, tmp_path, old, **kw):
    resp = d.submit(req(tmp_path, kw.pop("argv", job(d, old).argv), session=kw.pop("session", "sa"), replace=old, **kw))
    assert resp["ok"], resp
    return resp


def _end(d, go):
    go.touch()
    run_until(d, lambda: all(finished(d, j.id) for j in d.store.jobs()))


def test_a_replacement_asking_for_less_keeps_its_place_in_line(make_daemon, tmp_path):
    # #184: an agent kept two waiting jobs' oversized requests, since resubmitting would lose their place.
    d, go, a, b = _line(make_daemon, tmp_path)
    old_tag = job(d, a).tag
    resp = _replace(d, tmp_path, a, mem=4 * GiB, time_=600.0)
    new = resp["job"]["id"]
    assert job(d, new).tag == old_tag and resp.get("kept_place") is True
    assert d.show(new)["job"]["position"] < d.show(b)["job"]["position"]
    _end(d, go)


def test_a_replacement_asking_for_more_of_anything_gets_a_fresh_place(make_daemon, tmp_path):
    # Keeping the place for more would let a 1-minute placeholder hold one for a 10-hour job.
    d, go, a, b = _line(make_daemon, tmp_path)
    resp = _replace(d, tmp_path, a, mem=4 * GiB, time_=1200.0)
    new = resp["job"]["id"]
    assert not resp.get("kept_place")
    assert d.show(new)["job"]["position"] > d.show(b)["job"]["position"]
    _end(d, go)


def test_a_replacement_with_another_command_gets_a_fresh_place(make_daemon, tmp_path):
    d, go, a, b = _line(make_daemon, tmp_path)
    resp = _replace(d, tmp_path, a, argv=[*job(d, a).argv, "--fixed"], mem=4 * GiB, time_=600.0)
    assert not resp.get("kept_place")
    assert d.show(resp["job"]["id"])["job"]["position"] > d.show(b)["job"]["position"]
    _end(d, go)


def test_a_replacement_keeping_its_place_moves_no_job_back_again(make_daemon, tmp_path):
    # A job going ahead of its session's lower-priority jobs moves them back by its estimate (#91), once.
    d, go, a, b = _line(make_daemon, tmp_path)
    low = submit(d, tmp_path, [*job(d, a).argv, "low"], time_=600.0, session="sa", priority=3)
    tags = (job(d, low).tag, job(d, low).plain_tag)
    owed = d.store.get_meta("owed:sa")
    _replace(d, tmp_path, a, mem=4 * GiB, time_=600.0)
    assert (job(d, low).tag, job(d, low).plain_tag) == tags
    assert d.store.get_meta("owed:sa") == owed
    _end(d, go)


def test_the_help_and_agent_help_say_a_no_larger_replacement_keeps_its_place():
    import subprocess
    import sys

    from dibs.instructions import AGENT_HELP

    help_text = subprocess.run([sys.executable, "-m", "dibs", "submit", "--help"], capture_output=True, text=True).stdout
    assert "asks for no more keeps its place in line" in " ".join(help_text.split())
    assert "asks for no more of anything keeps the job's place in line" in " ".join(AGENT_HELP.split())


# #282: lining a session up again took a replacement as newly submitted, at the back of the session's line,
# and also as holding the place it kept: the replaced job's estimate left the line while its place stayed.
# An agent halved 20 waiting jobs' RAM with --replace; with no job starting, its session's other jobs moved
# a median 6.5 hours ahead, and 18 came to share a place.


def _session(make_daemon, tmp_path, folders=2):
    """A daemon whose only CPU a blocker holds, a clock to move, and session s1's folders."""
    from conftest import capacity
    from test_folder_turns import _trees

    now = [1000.0]
    d = make_daemon(clock=lambda: now[0], cap=capacity(cpus=[0]))
    submit(d, tmp_path, ["sleep", "30"], session="blocker")
    trees = _trees(tmp_path)
    return d, now, trees[:folders]


def _wait(d, now, folder, n):
    """Session s1 submits a 10-minute job in a folder, a second later than the last."""
    now[0] += 1
    return submit(d, folder, [PY, "-c", "pass", str(n)], time_=600.0, mem=2 * GiB, session="s1")


def _places(d):
    from dibs.model import State

    return {j.id: (j.tag, j.plain_tag) for j in d.store.jobs([State.QUEUED]) if j.session == "s1"}


def _again(d, old):
    """The same command, asking for less RAM: it keeps its place (#184)."""
    j = job(d, old)
    resp = d.submit(req(j.cwd, j.argv, mem=1 * GiB, time_=600.0, session="s1", replace=old))
    assert resp["ok"] and resp["kept_place"], resp
    return resp["job"]["id"]


def test_a_replacement_keeping_its_place_moves_no_other_job_of_a_multi_folder_session(make_daemon, tmp_path):
    d, now, (a, b) = _session(make_daemon, tmp_path)
    a1, a2, b1 = _wait(d, now, a, 1), _wait(d, now, a, 2), _wait(d, now, b, 3)
    before = _places(d)
    assert sorted(t for t, _ in before.values()) == [1601.0, 2201.0, 2801.0]
    assert before[b1][0] < before[a2][0]  # the folders take turns: b1 has an earlier place than its submit's
    b2 = _again(d, b1)
    after = _places(d)
    assert after[b2] == before[b1] and {i: after[i] for i in (a1, a2)} == {i: before[i] for i in (a1, a2)}
    b3 = _again(d, b2)  # replacing the replacement keeps that place too
    assert _places(d) == {a1: before[a1], a2: before[a2], b3: before[b1]}


def test_a_replacement_keeping_its_place_moves_nothing_in_a_one_folder_sessions_next_lining_up(make_daemon, tmp_path):
    d, now, (a,) = _session(make_daemon, tmp_path, folders=1)
    j1, j2, j3, j4 = (_wait(d, now, a, n) for n in range(4))
    before = _places(d)
    j2b = _again(d, j2)
    d.cancel([j4])  # the session's last waiting job held nothing back: its line is lined up again
    assert _places(d) == {j1: before[j1], j2b: before[j2], j3: before[j3]}


def test_a_replacement_of_a_job_whose_place_has_come_due_keeps_that_place(make_daemon, tmp_path):
    d, now, (a, b) = _session(make_daemon, tmp_path)
    a1, b1 = _wait(d, now, a, 1), _wait(d, now, b, 2)
    before = _places(d)
    now[0] = 2500.0  # both places have come due; the blocker still holds the CPU
    a2 = _again(d, a1)
    assert _places(d) == {a2: before[a1], b1: before[b1]}


def test_replacing_a_folders_waiting_jobs_one_by_one_leaves_the_session_the_places_it_had(make_daemon, tmp_path):
    d, now, (a, b) = _session(make_daemon, tmp_path)
    sweep = [_wait(d, now, a, n) for n in range(6)]
    [_wait(d, now, b, 10 + n) for n in range(3)]
    before = sorted(_places(d).values())
    for old in sweep:
        _again(d, old)
    assert sorted(_places(d).values()) == before  # none gained, none shared


def _before_282(make_daemon, tmp_path):
    """Session s1 as dibs before #282 left it after a replacement: the replacement without the time it
    lines up at, and the places that lining it up at its own submit, while keeping 2201 for it too, gave
    the session."""
    d, now, (a, b) = _session(make_daemon, tmp_path)
    a1, b1 = _wait(d, now, a, 1), _wait(d, now, b, 2)
    now[0] = 1500.0
    a3 = _wait(d, now, a, 3)
    now[0] = 1600.0
    b2 = _again(d, b1)
    old = {a1: 1601.0, b2: 2201.0, a3: 2201.0}
    for i, place in old.items():
        j = job(d, i)
        j.tag = j.plain_tag = place
        if i == b2:
            j.lined = None
        d.store.update_job(j)
    return d, old, b2


def _tags(d):
    return {i: tag for i, (tag, _) in _places(d).items()}


def test_a_replacement_queued_before_282_keeps_its_place_and_nothing_moves_across_the_restart(make_daemon, tmp_path):
    # Dropping the place such a replacement kept would move its session's jobs at the restart, and could put
    # it behind jobs submitted after the one it replaced, after dibs told its agent it kept its place.
    d, old, _ = _before_282(make_daemon, tmp_path)
    d.recover()  # the restart that installs #282 lines every session up again
    assert _tags(d) == old


def test_replacing_a_replacement_made_before_282_again_moves_nothing(make_daemon, tmp_path):
    # 20 such replacements were waiting when #282 was installed, and their agent had resized them once.
    d, old, b2 = _before_282(make_daemon, tmp_path)
    b3 = _again(d, b2)
    assert _tags(d) == {**{i: t for i, t in old.items() if i != b2}, b3: old[b2]}


def test_a_replacement_lowering_its_priority_keeps_the_place_its_priority_took_and_moves_nothing(make_daemon, tmp_path):
    # #184 keeps the earlier of a replacement's old and fresh place: for one that asks for a lower priority,
    # the place the old one took, whose payback it carries. Lined up at the lower priority, it lost that
    # place in a session waiting in several folders, while still carrying the payback.
    d, now, (a, b) = _session(make_daemon, tmp_path)
    a1, a2 = _wait(d, now, a, 1), _wait(d, now, a, 2)
    now[0] += 1
    b1 = submit(d, b, [PY, "-c", "pass", "3"], time_=600.0, mem=2 * GiB, session="s1", priority=9)
    before = _places(d)
    j = job(d, b1)
    resp = d.submit(req(j.cwd, j.argv, mem=1 * GiB, time_=600.0, session="s1", replace=b1, priority=5))
    assert resp["ok"] and resp["kept_place"], resp
    b2 = resp["job"]["id"]
    assert _places(d) == {a1: before[a1], a2: before[a2], b2: before[b1]}
    assert (job(d, b2).owed_carried, job(d, b2).owed_left) == (job(d, b1).owed_carried, job(d, b1).owed_left)
