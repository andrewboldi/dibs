"""A job asks for the integrated GPU with --igpu, to render through OpenGL and Vulkan: dibs gives it to
igpu_jobs jobs at a time (one by default), points the job's OpenGL and Vulkan at it, and counts a job
seen keeping it busy as holding it whether or not it asked, telling that job's session once (#276).
An agent had put a render there unseen, asking for no GPU at all, rather than wait for the NVIDIA one."""

from dataclasses import replace

import pytest
from conftest import capacity
from test_cli import RES, dibs  # noqa: F401 (dibs is a fixture)
from test_daemon import PY, job, req, submit
from test_igpu import _rendering

from dibs import launcher
from dibs.daemon import _keeps_place
from dibs.model import Capacity, Job, Request, RequestError, State, job_from_dict, job_to_dict, resolve_request
from dibs.planner import Queued, Running, plan
from dibs.units import GiB

NOW = 1_000_000.0
CAP = Capacity(cpus=tuple(range(8)), mem=32 * GiB, gpus=(), igpu=1)


def _r(igpu=False, **kw):
    return Request(cpus=1, mem=GiB, gpus=0, vram=0, time=600.0, igpu=igpu, **kw)


def test_one_job_at_a_time_gets_the_integrated_gpu():
    running = [Running(1, _r(igpu=True), (0,), (), NOW - 60, None)]
    queued = [Queued(2, _r(igpu=True), NOW, NOW), Queued(3, _r(), NOW + 1, NOW + 1)]
    p = plan(CAP, running, queued, NOW)
    assert [s.job_id for s in p.starts] == [3]  # a job that didn't ask isn't held up
    assert p.predictions[2].reason == "needs the integrated GPU (#1 has it)"
    assert p.predictions[2].start == NOW + 540  # when #1 is expected to end
    assert p.predictions[2].lacks == ("iGPU",)


def test_a_timing_run_takes_the_integrated_gpu_too():
    # It holds every CPU, and every job needs one: nothing starts beside it, on this GPU or any.
    running = [Running(1, _r(exclusive=True), (0,), (), NOW - 60, None)]
    p = plan(CAP, running, [Queued(2, _r(igpu=True), NOW, NOW)], NOW)
    assert p.starts == [] and p.predictions[2].start == NOW + 540


def test_more_seats_when_the_config_gives_them():
    running = [Running(1, _r(igpu=True), (0,), (), NOW - 60, None)]
    p = plan(replace(CAP, igpu=2), running, [Queued(2, _r(igpu=True), NOW, NOW)], NOW)
    assert [s.job_id for s in p.starts] == [2]


def test_asking_for_it_where_there_is_none_is_refused():
    raw = {"cpus": 1, "mem": GiB, "time": 60.0, "igpu": True}
    with pytest.raises(RequestError, match="no integrated GPU"):
        resolve_request(raw, replace(CAP, igpu=0))
    assert resolve_request(raw, CAP).igpu
    assert not resolve_request({**raw, "igpu": None}, CAP).igpu


def test_a_stored_job_from_before_still_loads():
    old = job_to_dict(_job(_r()))
    del old["request"]["igpu"]
    assert job_from_dict(old).request.igpu is False


def test_adding_igpu_to_a_waiting_job_is_asking_for_more():
    waiting = _job(_r())
    assert _keeps_place(waiting, "s1", waiting.argv, waiting.cwd, _r(), 5)
    assert not _keeps_place(waiting, "s1", waiting.argv, waiting.cwd, _r(igpu=True), 5)


def _job(request, env=None, gpu_ids=()):
    return Job(id=7, name="render", argv=["blender", "-b"], cwd="/w", env=dict(env or {}), request=request,
               session="s1", session_label="w·s1", submitted=NOW, tag=NOW, state=State.QUEUED, output="/w/out.log",
               gpu_ids=list(gpu_ids))  # fmt: skip


def test_an_igpu_jobs_opengl_and_vulkan_are_pointed_at_the_integrated_gpu(tmp_path, monkeypatch):
    egl, vk = tmp_path / "egl", tmp_path / "vk"
    for folder, names in ((egl, ("10_nvidia.json", "50_mesa.json")), (vk, ("nvidia_icd.json", "intel_icd.json"))):
        folder.mkdir()
        for name in names:
            (folder / name).write_text("{}")
    monkeypatch.setattr(launcher, "EGL_VENDOR_DIRS", (str(egl),))
    monkeypatch.setattr(launcher, "VULKAN_ICD_DIRS", (str(vk),))
    inherited = {"__NV_PRIME_RENDER_OFFLOAD": "1", "__GLX_VENDOR_LIBRARY_NAME": "nvidia", "__VK_LAYER_NV_optimus": "NVIDIA_only"}
    env = launcher.job_env(_job(_r(igpu=True), inherited), "inst")
    assert env["__EGL_VENDOR_LIBRARY_FILENAMES"] == str(egl / "50_mesa.json")
    assert env["VK_DRIVER_FILES"] == env["VK_ICD_FILENAMES"] == str(vk / "intel_icd.json")
    assert env["__GLX_VENDOR_LIBRARY_NAME"] == "mesa" and env["DIBS_IGPU"] == "1"
    assert "__NV_PRIME_RENDER_OFFLOAD" not in env and "__VK_LAYER_NV_optimus" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == ""  # no NVIDIA GPU asked for: CUDA sees none
    both = launcher.job_env(_job(replace(_r(igpu=True), gpus=1, vram=2 * GiB), inherited, gpu_ids=[0]), "inst")
    assert both["CUDA_VISIBLE_DEVICES"] == "0" and both["__GLX_VENDOR_LIBRARY_NAME"] == "mesa"  # CUDA there, drawing here
    plain = launcher.job_env(_job(_r(), inherited), "inst")
    assert "__EGL_VENDOR_LIBRARY_FILENAMES" not in plain and plain["__GLX_VENDOR_LIBRARY_NAME"] == "nvidia"


def _ask(d, tmp_path, **kw):
    r = req(tmp_path, [PY, "-c", "import time; time.sleep(30)"], **kw)
    r["request"]["igpu"] = True
    resp = d.submit(r)
    assert resp["ok"], resp
    return resp["job"]["id"]


@pytest.mark.parametrize("render_ns, holds", [(3_000_000_000, True), (200_000_000, False)])
def test_a_job_seen_keeping_it_busy_holds_it_whether_or_not_it_asked(make_daemon, tmp_path, monkeypatch, render_ns, holds):
    # 30% busy over 10 s holds the seat; 2% (a headless browser's page, say) doesn't.
    d, a, look, go = _rendering(make_daemon, tmp_path, monkeypatch, cap=replace(capacity(), igpu=1))
    look(10, 0)
    look(10, render_ns)
    b = _ask(d, tmp_path)
    d.tick()
    if holds:
        assert job(d, b).state == State.QUEUED
        assert d.show(b)["job"]["waiting_for"] == f"needs the integrated GPU (#{a} has it)"
    else:
        assert job(d, b).state == State.RUNNING
    told = [e["message"] for e in d.store.events(0.0) if e["kind"] == "advice" and e.get("job") == a]
    assert len(told) == (1 if holds else 0)
    if holds:
        assert "keeps the integrated GPU 30% busy without asking for it" in told[0] and "--igpu" in told[0]
    look(10, render_ns * 2)
    assert len([e for e in d.store.events(0.0) if e["kind"] == "advice" and e.get("job") == a]) == len(told)  # once
    d.cancel([b])
    go.touch()


def test_dibs_queue_and_dibs_show_say_a_job_asked_for_the_integrated_gpu():
    from test_top import view

    from dibs.render import asks_for, render_queue

    v = view(7, "render", State.QUEUED, 1, GiB, 0, 600, position=1, predicted_start=None, waiting_for="")
    v["request"]["igpu"] = True
    row = [line for line in render_queue({"now": 0.0, "jobs": [v]}, color=False).splitlines() if line.strip().startswith("7")][0]
    assert " igpu " in row
    assert asks_for(v["request"]) == "1 CPU, 1G RAM, the integrated GPU, ~10m"


def test_the_help_says_what_the_integrated_gpu_is_for():
    from dibs.instructions import AGENT_HELP, SKILL

    text = " ".join(AGENT_HELP.split())
    for phrase in ("`--igpu`", "OpenGL", "Vulkan", "not CUDA or PyTorch", "draws the screen"):
        assert phrase in text, phrase
    assert "--igpu" in SKILL


def test_the_config_says_how_many_jobs_share_it(tmp_path):
    from dibs.machine import ConfigError, load_config

    path = tmp_path / "config.toml"
    assert load_config(path).igpu_jobs == 1
    path.write_text("igpu_jobs = 0\n")
    assert load_config(path).igpu_jobs == 0
    path.write_text("igpu_jobs = -1\n")
    with pytest.raises(ConfigError, match="igpu_jobs"):
        load_config(path)


def test_the_cli_says_where_there_is_none(dibs):
    out = dibs("submit", *RES, "--igpu", "--", "true", check=2)
    assert "this machine has no integrated GPU dibs hands out" in out.stderr
