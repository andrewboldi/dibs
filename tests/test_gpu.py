import json
import os
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
    assert backend.list_gpus() == [] and backend.processes() == []


def test_nvml_backend_reads_real_gpu():
    try:
        backend = NvmlBackend()
    except OSError:
        pytest.skip("no NVIDIA driver here")
    gpus = backend.list_gpus()
    assert gpus and all(g.vram > 0 and g.uuid.startswith("GPU-") for g in gpus)
    assert isinstance(backend.processes(), list)
