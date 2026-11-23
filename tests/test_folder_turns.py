"""A session's folders take turns at its places in line, and a boost that passes a sibling folder's job
is paid back by its own folder (#226): a coding agent's subagents share their session's id."""

from dibs.model import Job, Request, State, folder_places
from dibs.units import GiB


def _job(i, folder, submitted, minutes, priority=5):
    return Job(id=i, name=f"j{i}", argv=["python", "x.py"], cwd=folder, env={}, session="s",
               request=Request(cpus=1, mem=GiB, gpus=0, vram=0, time=minutes * 60.0), session_label="w·s",
               submitted=submitted, tag=0.0, state=State.QUEUED, output="/dev/null", priority=priority)  # fmt: skip


def _order(jobs, offsets=None):
    places = folder_places(jobs, lambda cwd: cwd, offsets or {})
    return [j.id for j in sorted(jobs, key=lambda j: (places[j.id][0], j.submitted, j.id))]


def test_a_siblings_job_submitted_after_a_sweep_gets_the_next_turn():
    # 11-19: one subagent's 33-job sweep, submitted first, took all 22 starts of an hour while 8 jobs
    # from 5 sibling folders waited 32 minutes to 2.6 hours.
    sweep = [_job(i, "/a", 1000.0 + i, 10) for i in range(33)]
    sibling = _job(100, "/b", 1100.0, 10)
    order = _order([*sweep, sibling])
    assert order.index(100) == 1  # after the sweep's first job, not its 33rd


def test_one_folder_alone_keeps_todays_order():
    jobs = [_job(1, "/a", 1000.0, 30), _job(2, "/a", 1001.0, 5), _job(3, "/a", 1002.0, 5, priority=8)]
    assert _order(jobs) == [3, 1, 2]  # a higher priority ahead of its own folder's lower ones, as now


def test_a_folders_debt_moves_its_whole_line_back_once():
    # (a) and (b): the folder that boosted pays, with all its later turns, by the time it saved, once.
    mine = [_job(1, "/a", 1000.0, 10), _job(2, "/a", 1001.0, 10)]
    theirs = [_job(3, "/b", 1000.5, 10), _job(4, "/b", 1001.5, 10)]
    plain = folder_places([*mine, *theirs], lambda c: c, {})
    owing = folder_places([*mine, *theirs], lambda c: c, {"/a": 900.0})
    assert [owing[i][0] - plain[i][0] for i in (1, 2)] == [900.0, 900.0]  # not 900 per job, piled up
    assert [owing[i][0] for i in (3, 4)] == [plain[i][0] for i in (3, 4)]  # the sibling doesn't pay
    assert _order([*mine, *theirs], {"/a": 900.0}) == [3, 4, 1, 2]


def test_a_folder_that_comes_back_starts_from_when_it_submits():
    # A folder that had nothing waiting doesn't leapfrog: its place counts from its own submit.
    busy = [_job(i, "/a", 1000.0 + i, 10) for i in range(5)]
    back = _job(9, "/b", 2700.0, 10)  # came back 28 minutes later: its place is 2700 + 600
    assert _order([*busy, back]) == [0, 1, 2, 9, 3, 4]  # the busy folder's places: 1600, 2200, 2800, 3400


def test_a_folders_debt_reaches_its_boosted_jobs_too():
    # A later priority-9 job doesn't line up behind its folder's priority-5 jobs, so the debt must move
    # its own start back as well, or boosting again would dodge it.
    jobs = [_job(1, "/a", 1000.0, 10), _job(2, "/a", 1001.0, 10, priority=9)]
    plain = folder_places(jobs, lambda c: c, {})
    owing = folder_places(jobs, lambda c: c, {"/a": 900.0})
    assert owing[2][0] - plain[2][0] == 900.0


def _trees(tmp_path):
    """Two git working trees, as two subagents' worktrees are."""
    trees = []
    for name in ("a", "b"):
        tree = tmp_path / name
        tree.mkdir()
        (tree / ".git").write_text("gitdir: /elsewhere\n")
        trees.append(tree)
    return trees


def test_a_sessions_folders_take_turns_and_other_sessions_dont_move(make_daemon, tmp_path):
    from conftest import capacity
    from test_daemon import PY, job, submit

    now = [1000.0]
    d = make_daemon(clock=lambda: now[0], cap=capacity(cpus=[0]))
    submit(d, tmp_path, ["sleep", "30"], session="blocker")  # holds the only CPU
    a, b = _trees(tmp_path)
    other = submit(d, tmp_path, [PY, "-c", "pass"], time_=600.0, session="s2")
    sweep = [submit(d, a, [PY, "-c", "pass"], time_=600.0, session="s1") for _ in range(5)]
    before = job(d, other).tag
    now[0] += 60
    sibling = submit(d, b, [PY, "-c", "pass"], time_=600.0, session="s1")
    order = [j for _, j in sorted((job(d, j).tag, j) for j in [*sweep, sibling])]
    assert order.index(sibling) == 1  # the next turn, not the sixth place
    assert job(d, other).tag == before  # another session's place doesn't move


def test_a_boost_that_passed_a_siblings_job_is_paid_back_by_its_own_folder(make_daemon, tmp_path):
    from conftest import capacity, run_until
    from test_daemon import PY, finished, job, submit

    now = [1000.0]
    d = make_daemon(clock=lambda: now[0], cap=capacity(cpus=[0]))
    go = tmp_path / "go"
    blocker = submit(d, tmp_path, [PY, "-c", f"import os, time\nwhile not os.path.exists({str(go)!r}): time.sleep(0.02)"],
                     session="blocker")  # fmt: skip
    a, b = _trees(tmp_path)
    theirs = submit(d, b, ["sleep", "30"], time_=600.0, session="s1")
    now[0] += 10
    mine = submit(d, a, [PY, "-c", "pass"], time_=600.0, session="s1", priority=10)
    assert job(d, mine).tag < job(d, theirs).tag  # the boost passes the sibling's job
    go.touch()
    run_until(d, lambda: finished(d, blocker) and job(d, mine).started is not None)
    owed = d._turn_owed("s1", now[0])
    assert owed == {str(a): 540.0}  # the 9 minutes its priority saved, on its own folder only
    run_until(d, lambda: finished(d, mine) and job(d, theirs).started is not None)
    d.cancel([theirs])
    run_until(d, lambda: finished(d, theirs))


def test_a_boost_that_passed_only_its_own_folders_jobs_costs_its_folder_nothing(make_daemon, tmp_path):
    from conftest import capacity, run_until
    from test_daemon import PY, finished, job, submit

    now = [1000.0]
    d = make_daemon(clock=lambda: now[0], cap=capacity(cpus=[0]))
    go = tmp_path / "go"
    blocker = submit(d, tmp_path, [PY, "-c", f"import os, time\nwhile not os.path.exists({str(go)!r}): time.sleep(0.02)"],
                     session="blocker")  # fmt: skip
    a, _ = _trees(tmp_path)
    own = submit(d, a, ["sleep", "30"], time_=600.0, session="s1")
    now[0] += 10
    mine = submit(d, a, [PY, "-c", "pass"], time_=600.0, session="s1", priority=10)
    go.touch()
    run_until(d, lambda: finished(d, blocker) and job(d, mine).started is not None)
    assert d._turn_owed("s1", now[0]) == {}
    run_until(d, lambda: finished(d, mine) and job(d, own).started is not None)
    d.cancel([own])
    run_until(d, lambda: finished(d, own))
