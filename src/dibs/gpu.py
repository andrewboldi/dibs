"""Seeing the GPUs: what they are, and which processes are using them.

NVML through ctypes is the default: dibs polls the GPUs every second, and calling NVML costs about
0.2 ms of CPU where running nvidia-smi costs about 20 ms. nvidia-smi is the fallback, and the
backend tests and the demo drive through a fake nvidia-smi.
"""

import ctypes
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from dibs.model import Gpu
from dibs.units import MiB


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    gpu_index: int
    used: int | None  # bytes of VRAM, None if the driver doesn't say


class GpuBackend(Protocol):
    name: str

    def list_gpus(self) -> list[Gpu]: ...

    def processes(self) -> list[GpuProcess] | None:
        """Compute processes on every GPU, or None if they can't be seen."""
        ...


class NoGpuBackend:
    name = "none"

    def list_gpus(self) -> list[Gpu]:
        return []

    def processes(self) -> list[GpuProcess] | None:
        return []


class _Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _ProcessInfo(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
        ("gpuInstanceId", ctypes.c_uint),
        ("computeInstanceId", ctypes.c_uint),
    ]


_NVML_SUCCESS = 0
_NVML_ERROR_INSUFFICIENT_SIZE = 7
_NVML_VALUE_NOT_AVAILABLE = 0xFFFFFFFFFFFFFFFF


class NvmlBackend:
    name = "nvml"

    def __init__(self, library: str = "libnvidia-ml.so.1"):
        self._lib = ctypes.CDLL(library)  # OSError if the driver isn't installed
        rc = self._lib.nvmlInit_v2()
        if rc != _NVML_SUCCESS:
            raise OSError(f"NVML failed to start (error {rc})")
        self._handles: list[ctypes.c_void_p] | None = None

    def _devices(self) -> list[ctypes.c_void_p]:
        if self._handles is None:
            count = ctypes.c_uint()
            if self._lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != _NVML_SUCCESS:
                return []
            handles = []
            for i in range(count.value):
                handle = ctypes.c_void_p()
                if self._lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(handle)) == _NVML_SUCCESS:
                    handles.append(handle)
            self._handles = handles
        return self._handles

    def list_gpus(self) -> list[Gpu]:
        gpus = []
        for index, handle in enumerate(self._devices()):
            name = ctypes.create_string_buffer(96)
            uuid = ctypes.create_string_buffer(96)
            memory = _Memory()
            self._lib.nvmlDeviceGetName(handle, name, 96)
            self._lib.nvmlDeviceGetUUID(handle, uuid, 96)
            if self._lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != _NVML_SUCCESS:
                continue
            gpus.append(Gpu(index, uuid.value.decode(), name.value.decode(), memory.total))
        return gpus

    def processes(self) -> list[GpuProcess] | None:
        found = []
        for index, handle in enumerate(self._devices()):
            size = 64
            while True:
                count = ctypes.c_uint(size)
                infos = (_ProcessInfo * size)()
                rc = self._lib.nvmlDeviceGetComputeRunningProcesses_v3(
                    handle, ctypes.byref(count), infos
                )
                if rc == _NVML_ERROR_INSUFFICIENT_SIZE:
                    size = count.value + 16
                    continue
                if rc != _NVML_SUCCESS:
                    return None
                break
            for info in infos[: count.value]:
                used = None if info.usedGpuMemory == _NVML_VALUE_NOT_AVAILABLE else info.usedGpuMemory
                found.append(GpuProcess(info.pid, index, used))
        return found


class SmiBackend:
    name = "nvidia-smi"

    def __init__(self, env: Mapping[str, str] | None = None):
        self._env = dict(os.environ if env is None else env)
        self._exe = shutil.which("nvidia-smi", path=self._env.get("PATH"))
        self._uuid_to_index: dict[str, int] | None = None

    def _query(self, field: str) -> list[list[str]] | None:
        if not self._exe:
            return None
        try:
            out = subprocess.run(
                [self._exe, field, "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
                env=self._env,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if out.returncode != 0:
            return None
        return [[part.strip() for part in line.split(",")] for line in out.stdout.splitlines() if line.strip()]

    def list_gpus(self) -> list[Gpu]:
        gpus = []
        for row in self._query("--query-gpu=index,uuid,name,memory.total") or []:
            index, uuid, *name, total = row
            gpus.append(Gpu(int(index), uuid, ", ".join(name), int(float(total)) * MiB))
        self._uuid_to_index = {g.uuid: g.index for g in gpus}
        return gpus

    def processes(self) -> list[GpuProcess] | None:
        if self._uuid_to_index is None:
            self.list_gpus()
        rows = self._query("--query-compute-apps=pid,used_memory,gpu_uuid")
        if rows is None:
            return None
        found = []
        for pid, used, uuid in rows:
            if uuid not in self._uuid_to_index:
                continue
            amount = None if not used.isdigit() else int(used) * MiB
            found.append(GpuProcess(int(pid), self._uuid_to_index[uuid], amount))
        return found


def open_backend(env: Mapping[str, str] = os.environ) -> GpuBackend:
    """The backend named by DIBS_GPU_BACKEND, else NVML, else nvidia-smi, else none."""
    choice = env.get("DIBS_GPU_BACKEND")
    if choice == "nvml":
        return NvmlBackend()
    if choice == "smi":
        return SmiBackend(env)
    if choice == "none":
        return NoGpuBackend()
    if choice:
        raise ValueError(f"DIBS_GPU_BACKEND={choice!r}: use nvml, smi or none")
    try:
        return NvmlBackend()
    except OSError:
        pass
    if shutil.which("nvidia-smi", path=env.get("PATH")):
        return SmiBackend(env)
    return NoGpuBackend()
