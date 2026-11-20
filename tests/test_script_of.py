"""Which earlier runs count as the same script (#209)."""


from dibs.model import script_of


def test_a_test_runner_module_is_known_by_the_test_file_its_arguments_name():
    # #209: every `python -m pytest ...` here counted as one script, whatever tests it ran.
    assert script_of(["python", "-m", "pytest", "tests/test_heavy.py", "-x"]) == "tests/test_heavy.py"
    assert script_of(["python", "-m", "pytest", "-q", "tests/test_heavy.py::test_big"]) == "tests/test_heavy.py"
    assert script_of(["python", "-m", "unittest", "tests/test_heavy.py"]) == "tests/test_heavy.py"
    assert script_of(["python", "-m", "pytest", "-q"]) == "-m pytest"  # the whole suite
    assert script_of(["python", "-m", "myproj.train", "--config", "a.yaml"]) == "-m myproj.train"
    assert script_of(["pytest", "tests/test_heavy.py"]) == "tests/test_heavy.py"  # as it was


def test_light_test_runs_dont_advise_a_heavy_one(make_daemon, tmp_path):
    from conftest import capacity, run_until
    from test_advice import _advice
    from test_daemon import PY, finished, job, submit

    from dibs.model import Job, Request, State
    from dibs.units import GiB

    d = make_daemon(cap=capacity(mem=8 * GiB))
    go = tmp_path / "go"
    holder = submit(d, tmp_path, [PY, "-c", f"import os, time\nwhile not os.path.exists({str(go)!r}): time.sleep(0.02)"],
                    mem=5 * GiB, session="s0")  # fmt: skip
    for i in range(6):  # light tests, each 1.7G at most
        t = 1_000_000.0 + i * 1000
        d.store.add_job(Job(id=0, name="t", argv=[PY, "-m", "pytest", "tests/test_light.py"], cwd=str(tmp_path), env={},
                            request=Request(cpus=1, mem=5 * GiB, gpus=0, vram=0, time=600.0), session="s1",
                            session_label="w·s1", submitted=t, tag=t + 600, state=State.DONE, output="/dev/null",
                            started=t, ended=t + 480, exit_code=0, peak_mem=int(1.7 * GiB)))  # fmt: skip
    heavy = submit(d, tmp_path, [PY, "-m", "pytest", "tests/test_heavy.py"], mem=int(4.9 * GiB), session="s1")
    assert job(d, heavy).state == State.QUEUED  # it waits for RAM, when the advice would come
    assert _advice(d) == []
    go.touch()
    run_until(d, lambda: finished(d, holder) and finished(d, heavy))
