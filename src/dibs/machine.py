"""What this machine has, and how much of it dibs may hand out (config.toml)."""

import os
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dibs.model import Capacity, Gpu
from dibs.units import GiB, UnitError, format_size, parse_size

GUARD_MODES = ("all", "warn", "off")


@dataclass(frozen=True)
class Config:
    cpus: int | None = None  # hand out this many CPUs (the lowest-numbered ones)
    mem: int | None = None  # bytes of RAM to hand out
    gpus: tuple[int, ...] | None = None  # GPU indices to manage
    vram: int | None = None  # at most this many bytes of VRAM per GPU
    keep_days: int = 14
    guard: str = "all"
    guard_allow: tuple[str, ...] = ()
    guard_desktop: bool = True  # leave alone GPU programs launched from the desktop (#92)
    borrow_ram: bool = True  # plan RAM by what running jobs use, not only what they reserved (#86)
    huge_pages: bool = False  # let jobs have transparent huge pages (#225)
    agent_files: tuple[str, ...] = ()  # more instructions files for dibs setup (agent CLIs it doesn't know)


class ConfigError(ValueError):
    """config.toml asks for something dibs can't do."""


def _size(key: str, value) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    if isinstance(value, str):
        try:
            size = parse_size(value)
        except UnitError as e:
            raise ConfigError(f"config setting '{key}': {e}") from None
        if size > 0:
            return size
    raise ConfigError(f"config setting '{key}' must be a size like \"24G\"")


def load_config(path: Path) -> Config:
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        return Config()
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"can't read config {path}: {e}") from None

    fields = {}
    for key, value in data.items():
        if key == "cpus":
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError("config setting 'cpus' must be a whole number, at least 1")
            fields[key] = value
        elif key in ("mem", "vram"):
            fields[key] = _size(key, value)
        elif key == "gpus":
            if not isinstance(value, list) or not all(
                isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in value
            ):
                raise ConfigError("config setting 'gpus' must be a list of GPU indices, like [0, 1]")
            fields[key] = tuple(value)
        elif key == "keep_days":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigError("config setting 'keep_days' must be a whole number of days")
            fields[key] = value
        elif key == "guard":
            if value not in GUARD_MODES:
                raise ConfigError('config setting \'guard\' must be "all", "warn" or "off"')
            fields[key] = value
        elif key == "guard_allow":
            if not isinstance(value, list) or not all(isinstance(s, str) and s for s in value):
                raise ConfigError(
                    "config setting 'guard_allow' must be a list of program names, like [\"ollama\"]"
                )
            fields[key] = tuple(value)
        elif key == "guard_desktop":
            if not isinstance(value, bool):
                raise ConfigError("config setting 'guard_desktop' must be true or false")
            fields[key] = value
        elif key in ("borrow_ram", "huge_pages"):
            if not isinstance(value, bool):
                raise ConfigError(f"config setting '{key}' must be true or false")
            fields[key] = value
        elif key == "agent_files":
            if not isinstance(value, list) or not all(isinstance(s, str) and s for s in value):
                raise ConfigError(
                    "config setting 'agent_files' must be a list of files, like [\"~/.cursor/rules/dibs.mdc\"]"
                )
            fields[key] = tuple(value)
        else:
            raise ConfigError(f"unknown setting '{key}' in {path}")
    return Config(**fields)


def detect_cpus() -> tuple[int, ...]:
    """The CPU ids this process may run on."""
    if hasattr(os, "sched_getaffinity"):
        return tuple(sorted(os.sched_getaffinity(0)))
    return tuple(range(os.cpu_count() or 1))


def detect_mem() -> int:
    """Total RAM in bytes."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        if out.returncode == 0:
            return int(out.stdout.strip())
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def nominal_vram(vram: int) -> int:
    """A GPU's memory as sold: a reported size within 0.5% under a whole GiB counts as that GiB (an
    "8 GB" card reports 8188 MiB), so a 6G and a 2G job fit together and --vram 8G is accepted (#85).
    The driver keeps a few hundred MiB for itself anyway, so this changes nothing a job can allocate."""
    whole = -(-vram // GiB) * GiB
    return whole if whole - vram <= whole // 200 else vram


def build_capacity(config: Config, cpus: tuple[int, ...], mem: int, gpus: Sequence[Gpu]) -> Capacity:
    """What dibs hands out: everything detected, narrowed by the config."""
    gpus = [Gpu(g.index, g.uuid, g.name, nominal_vram(g.vram)) for g in gpus]
    if config.cpus is not None:
        if config.cpus > len(cpus):
            raise ConfigError(f"config asks for {config.cpus} CPUs, but this machine has {len(cpus)}")
        cpus = tuple(sorted(cpus))[: config.cpus]
    if config.mem is not None:
        if config.mem > mem:
            raise ConfigError(
                f"config asks for {format_size(config.mem)} RAM, but this machine has {format_size(mem)}"
            )
        mem = config.mem
    managed = list(gpus)
    if config.gpus is not None:
        known = {g.index for g in gpus}
        for index in config.gpus:
            if index not in known:
                have = "GPUs " + ", ".join(str(i) for i in sorted(known)) if known else "no GPUs"
                raise ConfigError(f"config lists GPU {index}, but this machine has {have}")
        managed = [g for g in gpus if g.index in config.gpus]
    if config.vram is not None and managed:
        largest = max(g.vram for g in managed)
        if config.vram > largest:
            raise ConfigError(
                f"config asks for {format_size(config.vram)} VRAM per GPU, "
                f"but the largest GPU has {format_size(largest)}"
            )
        managed = [Gpu(g.index, g.uuid, g.name, min(g.vram, config.vram)) for g in managed]
    return Capacity(cpus=tuple(cpus), mem=mem, gpus=tuple(managed))
