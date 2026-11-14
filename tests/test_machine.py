import os

import pytest

from dibs.machine import Config, ConfigError, build_capacity, detect_cpus, detect_mem, load_config
from dibs.model import Capacity, Gpu
from dibs.units import GiB

CPUS = tuple(range(24))
GPUS = [Gpu(0, "GPU-0", "Small", 8 * GiB), Gpu(1, "GPU-1", "Big", 24 * GiB)]


def write(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def test_load_config_missing_file(tmp_path):
    assert load_config(tmp_path / "missing.toml") == Config()


def test_load_config_values(tmp_path):
    path = write(
        tmp_path,
        'cpus = 20\nmem = "24G"\ngpus = [0]\nvram = "7.5G"\nkeep_days = 3\n'
        'guard = "warn"\nguard_allow = ["ollama"]\n',
    )
    assert load_config(path) == Config(
        cpus=20,
        mem=24 * GiB,
        gpus=(0,),
        vram=int(7.5 * GiB),
        keep_days=3,
        guard="warn",
        guard_allow=("ollama",),
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ('guard = "maybe"', "guard"),
        ('mem = "lots"', "mem"),
        ("cpu = 4", "unknown setting 'cpu'"),
        ("cpus = 0", "cpus"),
        ('gpus = "0"', "gpus"),
        ("keep_days = -1", "keep_days"),
        ("this is not toml", "config"),
    ],
)
def test_load_config_rejects(tmp_path, text, expected):
    with pytest.raises(ConfigError) as err:
        load_config(write(tmp_path, text))
    assert expected in str(err.value)


def test_build_capacity_defaults_to_everything():
    assert build_capacity(Config(), CPUS, 30 * GiB, GPUS) == Capacity(CPUS, 30 * GiB, tuple(GPUS))


def test_build_capacity_narrows():
    cap = build_capacity(Config(cpus=4, mem=8 * GiB, gpus=(1,), vram=6 * GiB), CPUS, 30 * GiB, GPUS)
    assert cap == Capacity((0, 1, 2, 3), 8 * GiB, (Gpu(1, "GPU-1", "Big", 6 * GiB),))


def test_vram_cap_applies_to_every_gpu():
    cap = build_capacity(Config(vram=12 * GiB), CPUS, 30 * GiB, GPUS)
    assert [g.vram for g in cap.gpus] == [8 * GiB, 12 * GiB]


@pytest.mark.parametrize(
    "config,message",
    [
        (Config(cpus=64), "config asks for 64 CPUs, but this machine has 24"),
        (Config(mem=64 * GiB), "config asks for 64G RAM, but this machine has 30G"),
        (Config(gpus=(2,)), "config lists GPU 2, but this machine has GPUs 0, 1"),
        (Config(vram=32 * GiB), "config asks for 32G VRAM per GPU, but the largest GPU has 24G"),
    ],
)
def test_build_capacity_rejects_more_than_detected(config, message):
    with pytest.raises(ConfigError) as err:
        build_capacity(config, CPUS, 30 * GiB, GPUS)
    assert str(err.value) == message


def test_detect_cpus_matches_affinity():
    assert detect_cpus() == tuple(sorted(os.sched_getaffinity(0)))


def test_detect_mem_matches_meminfo():
    with open("/proc/meminfo") as f:
        kib = int(next(line for line in f if line.startswith("MemTotal:")).split()[1])
    assert detect_mem() == kib * 1024
