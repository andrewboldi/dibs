import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from dibs.gpu import GpuProcess, NoGpuBackend, NvmlBackend, SmiBackend, open_backend
from dibs.model import Gpu
from dibs.units import GiB, MiB

FAKES = Path(__file__).parent / "fakes"


@pytest.fixture
def fake_env(tmp_path):
    gpu_file = tmp_path / "gpus.json"
    gpu_file.write_text(
        json.dumps(
            {
                "gpus": [
                    {"index": 0, "uuid": "GPU-fake-0", "name": "Fake GPU", "memory_mib": 8192},
                    {"index": 1, "uuid": "GPU-fake-1", "name": "Fake GPU", "memory_mib": 24576},
                ],
                "processes": [
                    {"pid": 123, "gpu_uuid": "GPU-fake-1", "used_mib": 3100},
                    {"pid": 124, "gpu_uuid": "GPU-fake-0", "used_mib": None},
                ],
            }
        )
    )
    return {"PATH": f"{FAKES}:{os.environ['PATH']}", "DIBS_FAKE_GPU_FILE": str(gpu_file)}


def test_smi_backend_lists_gpus(fake_env):
    assert SmiBackend(env=fake_env).list_gpus() == [
        Gpu(0, "GPU-fake-0", "Fake GPU", 8 * GiB),
        Gpu(1, "GPU-fake-1", "Fake GPU", 24 * GiB),
    ]


def test_smi_backend_processes(fake_env):
    assert SmiBackend(env=fake_env).processes() == [
        GpuProcess(123, 1, 3100 * MiB),
        GpuProcess(124, 0, None),
    ]


def test_smi_backend_without_binary(tmp_path):
    backend = SmiBackend(env={"PATH": str(tmp_path)})
    assert backend.list_gpus() == []
    assert backend.processes() is None


def test_open_backend_honours_env(fake_env):
    assert isinstance(open_backend({**fake_env, "DIBS_GPU_BACKEND": "smi"}), SmiBackend)
    assert isinstance(open_backend({**fake_env, "DIBS_GPU_BACKEND": "none"}), NoGpuBackend)
    with pytest.raises(ValueError):
        open_backend({**fake_env, "DIBS_GPU_BACKEND": "cuda"})


def test_no_gpu_backend():
    backend = NoGpuBackend()
    assert backend.list_gpus() == [] and backend.processes() == [] and backend.graphics_processes() == []


def test_nvml_backend_reads_real_gpu():
    try:
        backend = NvmlBackend()
    except OSError:
        pytest.skip("no NVIDIA driver here")
    gpus = backend.list_gpus()
    assert gpus and all(g.vram > 0 and g.uuid.startswith("GPU-") for g in gpus)
    assert isinstance(backend.processes(), list)
    assert isinstance(backend.graphics_processes(), list)


# --- AMD: amd-smi ---------------------------------------------------------------------------------

from dibs import gpu, launcher  # noqa: E402
from dibs.gpu import (  # noqa: E402
    AmdSmiBackend,
    parse_amd_list,
    parse_amd_memory_total,
    parse_amd_processes,
    parse_amd_static,
)


def amd_env(tmp_path, flavor="units", processes=None):
    gpu_file = tmp_path / "amd-gpus.json"
    gpu_file.write_text(
        json.dumps(
            {
                "flavor": flavor,
                "gpus": [
                    {"index": 0, "uuid": "amd-uuid-0", "name": "AMD Instinct MI300X", "memory_mib": 196592},
                    {"index": 1, "uuid": "amd-uuid-1", "name": "AMD Radeon RX 7900 XTX", "memory_mib": 24560},
                ],
                "processes": processes
                if processes is not None
                else [
                    {"pid": 123, "gpu_uuid": "amd-uuid-1", "used_mib": 3100},
                    {"pid": 124, "gpu_uuid": "amd-uuid-0", "used_mib": None},
                ],
            }
        )
    )
    # a PATH with amd-smi but no nvidia-smi, so auto-detection can't find the NVIDIA fake
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    if not (bin_dir / "amd-smi").exists():
        (bin_dir / "amd-smi").symlink_to(FAKES / "amd-smi")
        (bin_dir / "python3").symlink_to(sys.executable)  # for the fake's `env python3`
    return {"PATH": str(bin_dir), "DIBS_FAKE_GPU_FILE": str(gpu_file)}


@pytest.mark.parametrize("flavor", ["units", "plain", "gpu_data", "no_static_size"])
def test_amd_backend_lists_gpus_in_every_output_shape(tmp_path, flavor):
    assert AmdSmiBackend(env=amd_env(tmp_path, flavor)).list_gpus() == [
        Gpu(0, "amd-uuid-0", "AMD Instinct MI300X", 196592 * MiB),
        Gpu(1, "amd-uuid-1", "AMD Radeon RX 7900 XTX", 24560 * MiB),
    ]


@pytest.fixture
def kfd(tmp_path, monkeypatch):
    """A file standing in for /dev/kfd, and a way to start children that hold it open (or not)."""
    device = tmp_path / "kfd"
    device.write_text("")
    monkeypatch.setattr("dibs.gpu.KFD_DEVICE", str(device))
    children = []

    def spawn(holds):
        code = "import sys, time\n" + (f"f = open({str(device)!r})\n" if holds else "") + "print('up', flush=True); time.sleep(60)"
        child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        child.stdout.readline()  # the fd is open once it says so
        children.append(child)
        return child.pid

    yield spawn
    for child in children:
        child.kill()
        child.wait()


@pytest.mark.parametrize("flavor", ["units", "plain", "gpu_data"])
def test_amd_backend_processes_in_every_output_shape(tmp_path, flavor, kfd):
    pid_a, pid_b = kfd(True), kfd(True)
    env = amd_env(
        tmp_path,
        flavor,
        [{"pid": pid_a, "gpu_uuid": "amd-uuid-1", "used_mib": 3100}, {"pid": pid_b, "gpu_uuid": "amd-uuid-0", "used_mib": None}],
    )
    backend = AmdSmiBackend(env=env)
    backend.list_gpus()
    assert backend.processes() == [GpuProcess(pid_b, 0, None), GpuProcess(pid_a, 1, 3100 * MiB)]


def test_amd_backend_drops_graphics_clients_that_amd_smi_lists(tmp_path, kfd):
    compute, graphics = kfd(True), kfd(False)
    env = amd_env(
        tmp_path,
        processes=[
            {"pid": compute, "gpu_uuid": "amd-uuid-0", "used_mib": 100},
            {"pid": graphics, "gpu_uuid": "amd-uuid-0", "used_mib": 200},
        ],
    )
    backend = AmdSmiBackend(env=env)
    backend.list_gpus()
    assert backend.processes() == [GpuProcess(compute, 0, 100 * MiB)]


def test_amd_backend_drops_processes_it_cannot_inspect(tmp_path, kfd):
    env = amd_env(tmp_path, processes=[{"pid": pid, "gpu_uuid": "amd-uuid-0", "used_mib": 100} for pid in (1, 2**22 + 12345)])
    backend = AmdSmiBackend(env=env)
    backend.list_gpus()
    assert backend.processes() == []  # pid 1 (fds unreadable to us) and a pid that is gone


def test_vram_watch_sees_a_jobs_kfd_process(tmp_path, kfd, make_daemon):
    from conftest import capacity

    env = amd_env(tmp_path)
    backend = AmdSmiBackend(env=env)
    gpus = backend.list_gpus()
    d = make_daemon(cap=capacity(gpus=gpus), backend=backend)
    code = f"import time; f = open({str(tmp_path / 'kfd')!r}); time.sleep(60)"
    resp = d.submit(
        {
            "argv": [sys.executable, "-c", code],
            "cwd": str(tmp_path),
            "env": dict(os.environ),
            "session": "s",
            "request": {"cpus": 1, "mem": GiB, "vram": 6 * GiB, "gpus": None, "time": 60},
        }
    )
    job = d.store.get_job(resp["job"]["id"])
    deadline = time.time() + 10
    pid = None
    while time.time() < deadline and pid is None:
        pid = next((p for p in launcher.job_pids(job) if gpu._holds_kfd(p)), None)
        time.sleep(0.02)
    assert pid, "job never opened the stand-in kfd"
    data = json.loads(open(env["DIBS_FAKE_GPU_FILE"]).read())
    data["processes"] = [{"pid": pid, "gpu_uuid": gpus[0].uuid, "used_mib": 5000}]
    open(env["DIBS_FAKE_GPU_FILE"], "w").write(json.dumps(data))
    d.tick()
    assert d.store.get_job(job.id).peak_vram == 5000 * MiB


def test_amd_backend_with_no_processes(tmp_path):
    backend = AmdSmiBackend(env=amd_env(tmp_path, processes=[]))
    backend.list_gpus()
    assert backend.processes() == []


def test_amd_backend_without_binary(tmp_path):
    backend = AmdSmiBackend(env={"PATH": str(tmp_path)})
    assert backend.list_gpus() == []
    assert backend.processes() is None


def test_amd_backend_when_amd_smi_fails(tmp_path):
    env = amd_env(tmp_path)
    env["DIBS_FAKE_GPU_FILE"] = str(tmp_path / "missing.json")
    backend = AmdSmiBackend(env=env)
    assert backend.list_gpus() == []
    assert backend.processes() is None


def test_amd_backend_is_amd():
    assert AmdSmiBackend(env={"PATH": ""}).vendor == "amd"
    assert SmiBackend(env={"PATH": ""}).vendor == NoGpuBackend().vendor == "nvidia"


def test_open_backend_amd_by_name(tmp_path):
    env = amd_env(tmp_path)
    assert isinstance(open_backend({**env, "DIBS_GPU_BACKEND": "amd"}), AmdSmiBackend)
    assert isinstance(open_backend({**env, "DIBS_GPU_BACKEND": "amd-smi"}), AmdSmiBackend)


@pytest.fixture
def no_nvml(monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("no NVIDIA driver")

    monkeypatch.setattr("dibs.gpu.NvmlBackend", refuse)


def test_open_backend_finds_amd_smi_when_nothing_else_is_there(tmp_path, no_nvml):
    assert isinstance(open_backend(amd_env(tmp_path)), AmdSmiBackend)


def test_open_backend_prefers_nvidia_smi_over_amd_smi(tmp_path, no_nvml):
    env = amd_env(tmp_path)
    env["PATH"] = f"{FAKES}:{env['PATH']}"  # both fakes on PATH
    assert isinstance(open_backend(env), SmiBackend)


def test_open_backend_without_any_smi(tmp_path, no_nvml):
    assert isinstance(open_backend({"PATH": str(tmp_path)}), NoGpuBackend)


# The parsers, on the JSON shapes amd-smi has printed (sources in the comment in gpu.py).

LIST_6_2 = [{"gpu": 0, "bdf": "0000:0c:00.0", "uuid": "c4ff74a1-0000-1000-80ee-aaaaaaaaaaaa"}]
LIST_6_4 = [
    {"gpu": 0, "bdf": "0000:0c:00.0", "uuid": "c4ff74a1", "kfd_id": 1, "node_id": 2, "partition_id": 0},
    {"gpu": 1, "bdf": "0000:0d:00.0", "uuid": "N/A", "kfd_id": "N/A", "node_id": "N/A", "partition_id": 0},
]


def test_parse_list_across_versions():
    assert parse_amd_list(LIST_6_2) == [(0, "c4ff74a1-0000-1000-80ee-aaaaaaaaaaaa")]
    assert parse_amd_list(LIST_6_4) == [(0, "c4ff74a1"), (1, "0000:0d:00.0")]  # N/A uuid -> bdf
    assert parse_amd_list({"gpu_data": LIST_6_2}) == parse_amd_list(LIST_6_2)
    assert parse_amd_list(LIST_6_4[0]) == [(0, "c4ff74a1")]  # a lone object
    assert parse_amd_list("Please run with sudo") == parse_amd_list(None) == parse_amd_list([{"x": 1}]) == []


def test_parse_static_vram_size_shapes():
    def size(v):
        return parse_amd_static([{"gpu": 0, "vram": {"type": "HBM3", "size": v}}])[0][1]

    assert size({"value": 196592, "unit": "MB"}) == 196592 * MiB
    assert size(196592) == 196592 * MiB  # bare numbers are MB
    assert size("196592 MB") == 196592 * MiB
    assert size({"value": 16, "unit": "GB"}) == 16 * GiB
    assert size("N/A") is None
    assert size({"value": "N/A", "unit": "MB"}) is None
    assert size(0) is None


def test_parse_static_name():
    assert parse_amd_static([{"gpu": 0, "asic": {"market_name": "AMD Instinct MI300X"}}]) == {
        0: ("AMD Instinct MI300X", None)
    }
    assert parse_amd_static([{"gpu": 0, "asic": {"market_name": "N/A"}}]) == {0: (None, None)}


def test_parse_memory_total_shapes():
    usage = {"total_vram": {"value": 16368, "unit": "MB"}, "used_vram": {"value": 280, "unit": "MB"}}
    assert parse_amd_memory_total([{"gpu": 0, "mem_usage": usage}]) == {0: 16368 * MiB}
    assert parse_amd_memory_total([{"gpu": 0, "values": {"mem_usage": usage}}]) == {0: 16368 * MiB}
    assert parse_amd_memory_total([{"gpu": 0, "mem_usage": {"total_vram": 16368}}]) == {0: 16368 * MiB}
    assert parse_amd_memory_total([{"gpu": 0, "mem_usage": {"total_vram": "N/A"}}]) == {}


def _proc(pid, vram=None, mem=None):
    info = {"name": "python3", "pid": pid, "usage": {}, "cu_occupancy": {"value": 0, "unit": "%"}}
    if vram is not None:
        info["memory_usage"] = {"gtt_mem": 0, "cpu_mem": 0, "vram_mem": vram}
    if mem is not None:
        info["mem_usage"] = mem
    return {"process_info": info}


def test_parse_processes_shapes():
    gib = {"value": GiB, "unit": "B"}
    payload = [
        {"gpu": 0, "process_list": [{"process_info": "No running processes detected"}]},
        {"gpu": 1, "process_list": [_proc(7, vram=gib, mem={"value": 2 * GiB, "unit": "B"})]},
        {"gpu": 2, "process_list": [_proc(8, vram=GiB), _proc("9", mem=GiB), _proc(10), _proc(11, vram="N/A")]},
    ]
    assert parse_amd_processes(payload) == [
        GpuProcess(7, 1, GiB),  # vram_mem, not the process's whole mem_usage
        GpuProcess(8, 2, GiB),  # bare bytes
        GpuProcess(9, 2, GiB),  # string pid, only mem_usage given
        GpuProcess(10, 2, None),
        GpuProcess(11, 2, None),
    ]
    assert parse_amd_processes({"gpu_data": payload}) == parse_amd_processes(payload)
    assert parse_amd_processes([{"gpu": 0, "process_list": "No running processes detected"}]) == []
    assert parse_amd_processes(None) == []


def test_amd_smi_json_with_warnings_around_it(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    exe = fake / "amd-smi"
    exe.write_text(
        "#!/bin/sh\n"
        "echo 'WARNING: AMD GPU device(s) is/are in a non-default partition mode'\n"
        'echo \'[{"gpu": 0, "bdf": "0000:0c:00.0", "uuid": "u0", "vram": {"size": 1024}, '
        '"asic": {"market_name": "AMD Test"}}]\'\n'
    )
    exe.chmod(0o755)
    assert AmdSmiBackend(env={"PATH": f"{fake}:/usr/bin:/bin"}).list_gpus() == [Gpu(0, "u0", "AMD Test", GiB)]
