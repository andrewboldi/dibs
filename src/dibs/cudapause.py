"""Pausing other processes' CUDA state, and restoring it, through the CUDA driver's checkpoint API (driver 570
and later), so GPU jobs can be carried across a suspend (#293).

A paused process's GPU memory is copied into its own RAM and the GPU is released; its CPU threads carry on,
and a CUDA call it makes waits until it is restored. dibs runs this as a process of its own,
`python -m dibs.cudapause pause|resume SECONDS PID...`, which prints one JSON object of each pid's error (null
when it worked): so the daemon never loads the driver, and a call that hangs can't hold the daemon up."""

import ctypes
import json
import sys
import threading
import time

RUNNING, LOCKED, CHECKPOINTED, FAILED = 0, 1, 2, 3  # CUprocessState
GONE = "not a CUDA process, or gone"


class _LockArgs(ctypes.Structure):
    _fields_ = [("timeoutMs", ctypes.c_uint), ("reserved0", ctypes.c_uint), ("reserved1", ctypes.c_uint64 * 7)]


def driver():
    """The CUDA driver, initialized; OSError if there is none."""
    lib = ctypes.CDLL("libcuda.so.1")
    if r := lib.cuInit(0):
        raise OSError(f"the CUDA driver didn't start: error {r}")
    return lib


def _state(lib, pid: int) -> int | None:
    state = ctypes.c_int(-1)
    return state.value if lib.cuCheckpointProcessGetState(pid, ctypes.byref(state)) == 0 else None


def pause(lib, pid: int, timeout_ms: int) -> str | None:
    """Lock the process's CUDA calls, waiting up to timeout_ms for its submitted GPU work to finish, then move
    its GPU memory into its RAM; unlocked again if that fails. None once paused, else what went wrong."""
    if r := lib.cuCheckpointProcessLock(pid, ctypes.byref(_LockArgs(timeout_ms, 0))):
        return f"lock: CUDA error {r}"
    if r := lib.cuCheckpointProcessCheckpoint(pid, None):
        lib.cuCheckpointProcessUnlock(pid, None)
        return f"checkpoint: CUDA error {r}"
    return None


def resume(lib, pid: int) -> str | None:
    """Put a paused process's GPU memory back and unlock it, from whatever step its pause got to. None once it
    runs again, else what went wrong."""
    state = _state(lib, pid)
    if state is None:
        return GONE
    if state == FAILED:
        return "its CUDA state failed"
    if state == CHECKPOINTED and (r := lib.cuCheckpointProcessRestore(pid, None)):
        return f"restore: CUDA error {r}"
    if state in (CHECKPOINTED, LOCKED) and (r := lib.cuCheckpointProcessUnlock(pid, None)):
        return f"unlock: CUDA error {r}"
    return None


def run(op: str, seconds: float, pids: list[int], lib=None) -> dict[int, str | None]:
    """Each pid's result, done side by side so one slow process doesn't hold up the others, all within one
    deadline `seconds` from now: "timed out" for one still going then (resuming it later finishes from
    whatever step it got to)."""
    lib = lib or driver()
    deadline = time.monotonic() + seconds
    results: dict[int, str | None] = {}

    def one(pid: int) -> None:  # the lock waits at most half the time, leaving the rest for the copy
        results[pid] = pause(lib, pid, int(seconds * 500)) if op == "pause" else resume(lib, pid)

    threads = [threading.Thread(target=one, args=(pid,), daemon=True) for pid in pids]
    for t in threads:
        t.start()
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    return {pid: results.get(pid, "timed out") for pid in pids}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    op, seconds, pids = argv[0], float(argv[1]), [int(p) for p in argv[2:]]
    try:
        results = run(op, seconds, pids)
    except OSError as e:
        results = {pid: str(e) for pid in pids}
    print(json.dumps({str(pid): result for pid, result in results.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
