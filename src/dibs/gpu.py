"""Seeing the GPUs: what they are, and which processes are using them.

NVML through ctypes is the default: dibs polls the GPUs every second, and calling NVML costs about
0.2 ms of CPU where running nvidia-smi costs about 20 ms. nvidia-smi is the fallback, and the
backend tests and the demo drive through a fake nvidia-smi.

AMD GPUs go through the amd-smi command line (never its Python bindings, which aren't in the
standard library): `AmdSmiBackend`. A machine has one vendor's GPUs, so the backend decides which
visibility variables jobs get (see `vendor` and launcher.job_env).
"""

import ctypes
import json
import os
import re
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
    vendor: str  # "nvidia" or "amd": which environment variables hide GPUs from a job

    def list_gpus(self) -> list[Gpu]: ...

    def processes(self) -> list[GpuProcess] | None:
        """Compute processes on every GPU, or None if they can't be seen."""
        ...

    def graphics_processes(self) -> list[GpuProcess] | None:
        """Processes holding a graphics context (EGL, OpenGL) on every GPU, or None if they can't be
        seen. A job that renders on the GPU holds VRAM only this list shows (#173). Only NVML reports
        them; the other backends return none."""
        ...


class NoGpuBackend:
    name = "none"
    vendor = "nvidia"

    def list_gpus(self) -> list[Gpu]:
        return []

    def processes(self) -> list[GpuProcess] | None:
        return []

    def graphics_processes(self) -> list[GpuProcess] | None:
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
    vendor = "nvidia"

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
        return self._running(self._lib.nvmlDeviceGetComputeRunningProcesses_v3)

    def graphics_processes(self) -> list[GpuProcess] | None:
        return self._running(self._lib.nvmlDeviceGetGraphicsRunningProcesses_v3)

    def _running(self, query) -> list[GpuProcess] | None:
        found = []
        for index, handle in enumerate(self._devices()):
            size = 64
            while True:
                count = ctypes.c_uint(size)
                infos = (_ProcessInfo * size)()
                rc = query(handle, ctypes.byref(count), infos)
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
    vendor = "nvidia"

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

    def graphics_processes(self) -> list[GpuProcess] | None:
        return []  # nvidia-smi has a query for compute processes only


# --- AMD: the amd-smi command line -------------------------------------------------------------
#
# What the JSON parsers below were written against (ROCm/amdsmi, amdsmi_cli/amdsmi_commands.py and
# amdsmi_logger.py, read at tags rocm-6.2.0 e3427443, rocm-6.4.0 f51e0583, rocm-7.0.0 24f979e4,
# rocm-7.2.0 49f47150, and at rocm-systems develop 4f6eeafa; nothing was run on real hardware):
#
# * Every command prints a JSON list with one object per GPU, each carrying "gpu": N, the id that
#   `amd-smi -g N` takes. rocm-7.x can also wrap lists as {"gpu_data": [...]}; both are accepted.
# * `amd-smi list --json`: {"gpu", "bdf", "uuid"} (+ "kfd_id", "node_id", "partition_id" from 6.4).
# * `amd-smi static --asic --vram --json`: "vram": {"type", "vendor", "size"}, with size as
#   {"value": 196592, "unit": "MB"} (6.2 and later; MB means MiB, amd-smi divides bytes by 1024*1024),
#   and "asic": {"market_name", ...}. A plain number (MB) and "N/A" are tolerated.
# * `amd-smi metric --mem-usage --json`: "mem_usage": {"total_vram": {"value", "unit": "MB"}, ...},
#   used only when static can't say the size.
# * `amd-smi process --json`: "process_list": [{"process_info": {"name", "pid", "mem_usage":
#   {"value", "unit": "B"}, "memory_usage": {"gtt_mem", "cpu_mem", "vram_mem"} (each the same
#   value/unit object), "usage": {...}, "cu_occupancy"}}], and where a GPU has no process,
#   "process_info" is the string "No running processes detected". In 7.0 the dict is rebuilt but
#   keeps these keys. vram_mem is the VRAM; mem_usage is the process's total, the fallback.
#
# Units: {"value", "unit"} objects, bare numbers (in the field's native unit) and "123 MB" strings
# all parse. Anything unreadable becomes None, never a guess.
#
# amd-smi's process list is not limited to compute: a compositor or browser using the GPU for
# graphics is listed too. `processes()` therefore keeps only processes holding /dev/kfd open.

_UNITS = {"": 1, "B": 1, "KB": 1 << 10, "KIB": 1 << 10, "MB": MiB, "MIB": MiB, "GB": 1 << 30, "GIB": 1 << 30,
          "TB": 1 << 40, "TIB": 1 << 40}
_AMOUNT = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]*)\s*$")


def _bytes(value, default_unit: str) -> int | None:
    """Bytes from an amd-smi quantity: {"value", "unit"}, a bare number, or "12 MB"."""
    unit = default_unit
    if isinstance(value, dict):
        unit = str(value.get("unit") or default_unit)
        value = value.get("value")
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        match = _AMOUNT.match(value)
        if not match:
            return None
        number = float(match.group(1))
        unit = match.group(2) or unit
    else:
        return None
    factor = _UNITS.get(unit.upper())
    if factor is None or number < 0:
        return None
    return int(number * factor)


def _per_gpu(payload) -> list[dict]:
    """The per-GPU objects from amd-smi's JSON, whichever wrapper it came in."""
    if isinstance(payload, dict):
        inner = payload.get("gpu_data")
        payload = inner if isinstance(inner, list) else [payload]
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict) and _gpu_id(item) is not None]


def _gpu_id(item: dict) -> int | None:
    value = item.get("gpu")
    if isinstance(value, bool) or not isinstance(value, (int, str)) or not str(value).strip().isdigit():
        return None
    return int(value)


def _values(item: dict) -> dict:
    """Metrics sit at the top of an object, or (older releases) under "values"."""
    inner = item.get("values")
    return inner if isinstance(inner, dict) else item


def parse_amd_list(payload) -> list[tuple[int, str]]:
    """(gpu id, uuid or bdf) from `amd-smi list --json`."""
    found = []
    for item in _per_gpu(payload):
        ident = next((str(item[k]) for k in ("uuid", "bdf") if item.get(k) not in (None, "", "N/A")), None)
        found.append((_gpu_id(item), ident))
    return found


def parse_amd_static(payload) -> dict[int, tuple[str | None, int | None]]:
    """{gpu id: (marketing name, VRAM bytes)} from `amd-smi static --asic --vram --json`."""
    found = {}
    for item in _per_gpu(payload):
        asic, vram = item.get("asic"), item.get("vram")
        name = asic.get("market_name") if isinstance(asic, dict) else None
        size = _bytes(vram.get("size"), "MB") if isinstance(vram, dict) else None
        found[_gpu_id(item)] = (name if name and name != "N/A" else None, size or None)
    return found


def parse_amd_memory_total(payload) -> dict[int, int]:
    """{gpu id: VRAM bytes} from `amd-smi metric --mem-usage --json`."""
    found = {}
    for item in _per_gpu(payload):
        usage = _values(item).get("mem_usage")
        total = _bytes(usage.get("total_vram"), "MB") if isinstance(usage, dict) else None
        if total:
            found[_gpu_id(item)] = total
    return found


def parse_amd_processes(payload) -> list[GpuProcess]:
    """Processes with VRAM in use from `amd-smi process --json`."""
    found = []
    for item in _per_gpu(payload):
        entries = item.get("process_list")
        for entry in entries if isinstance(entries, list) else []:
            info = entry.get("process_info") if isinstance(entry, dict) else None
            if not isinstance(info, dict):
                continue  # "No running processes detected"
            pid = info.get("pid")
            if isinstance(pid, bool) or not str(pid).isdigit():
                continue
            memory = info.get("memory_usage")
            used = _bytes(memory.get("vram_mem"), "B") if isinstance(memory, dict) else None
            if used is None:
                used = _bytes(info.get("mem_usage"), "B")
            found.append(GpuProcess(int(pid), _gpu_id(item), used))
    return found


def _json_in(text: str):
    """amd-smi may print warnings around its JSON; take the JSON."""
    try:
        return json.loads(text)
    except ValueError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except ValueError:
                continue
    return None


# Every ROCm compute process opens the KFD device; graphics-only clients (the compositor, browsers)
# don't. amd-smi lists both, and the guard stops what it sees, so only KFD holders are compute.
KFD_DEVICE = "/dev/kfd"


def _holds_kfd(pid: int) -> bool:
    """Whether the process has KFD_DEVICE open. A process whose fds can't be read (another user's)
    doesn't count: the guard leaves other users' processes alone anyway."""
    try:
        fds = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        return False
    for fd in fds:
        try:
            if os.readlink(f"/proc/{pid}/fd/{fd}") == KFD_DEVICE:
                return True
        except OSError:
            continue
    return False


class AmdSmiBackend:
    """AMD GPUs through the amd-smi command line. A GPU's index is amd-smi's own GPU id, which is
    also the index ROCR_VISIBLE_DEVICES takes (both count GPUs in the order the driver found them)."""

    name = "amd-smi"
    vendor = "amd"

    def __init__(self, env: Mapping[str, str] | None = None):
        self._env = dict(os.environ if env is None else env)
        self._exe = shutil.which("amd-smi", path=self._env.get("PATH"))
        self._known: set[int] = set()

    def _query(self, *args: str):
        if not self._exe:
            return None
        try:
            out = subprocess.run(
                [self._exe, *args, "--json"], capture_output=True, text=True, timeout=30, env=self._env
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if out.returncode != 0:
            return None
        return _json_in(out.stdout)

    def list_gpus(self) -> list[Gpu]:
        listed = parse_amd_list(self._query("list"))
        static = parse_amd_static(self._query("static", "--asic", "--vram"))
        totals = None
        gpus = []
        for index, ident in listed:
            name, size = static.get(index, (None, None))
            if size is None:
                if totals is None:
                    totals = parse_amd_memory_total(self._query("metric", "--mem-usage"))
                size = totals.get(index)
            if not size:
                continue  # dibs can't share a GPU whose memory it can't size
            gpus.append(Gpu(index, ident or f"amd-{index}", name or "AMD GPU", size))
        self._known = {g.index for g in gpus}
        return gpus

    def processes(self) -> list[GpuProcess] | None:
        payload = self._query("process")
        if payload is None:
            return None
        found = parse_amd_processes(payload)
        if self._known:
            found = [p for p in found if p.gpu_index in self._known]
        return [p for p in found if _holds_kfd(p.pid)]

    def graphics_processes(self) -> list[GpuProcess] | None:
        return []  # amd-smi lists graphics clients, but processes() keeps compute ones only (/dev/kfd)


def open_backend(env: Mapping[str, str] = os.environ) -> GpuBackend:
    """The backend named by DIBS_GPU_BACKEND, else NVML, else nvidia-smi, else amd-smi, else none."""
    choice = env.get("DIBS_GPU_BACKEND")
    if choice == "nvml":
        return NvmlBackend()
    if choice == "smi":
        return SmiBackend(env)
    if choice in ("amd", "amd-smi"):
        return AmdSmiBackend(env)
    if choice == "none":
        return NoGpuBackend()
    if choice:
        raise ValueError(f"DIBS_GPU_BACKEND={choice!r}: use nvml, smi, amd or none")
    try:
        return NvmlBackend()
    except OSError:
        pass
    if shutil.which("nvidia-smi", path=env.get("PATH")):
        return SmiBackend(env)
    if shutil.which("amd-smi", path=env.get("PATH")):
        return AmdSmiBackend(env)
    return NoGpuBackend()
