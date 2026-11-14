"""What people and agents read in their terminal."""

import os
import shlex
import time

from dibs.units import format_duration, format_size

_COLORS = {"running": "32", "queued": "33", "failed": "31", "done": "2", "cancelled": "2"}


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def clock(t: float, now: float) -> str:
    """Local time of day, with the date when it's more than a day away."""
    fmt = "%b %d %H:%M" if abs(t - now) > 86400 else "%H:%M"
    return time.strftime(fmt, time.localtime(t))


def when(t: float, now: float) -> str:
    """A future moment: ~14:32 (in 8m)."""
    if t <= now + 0.5:
        return "now"
    return f"~{clock(t, now)} (in {format_duration(t - now)})"


def ago(t: float, now: float) -> str:
    return f"{format_duration(max(0.0, now - t))} ago"


def cpu_list(ids: list[int]) -> str:
    """0,1,2,3,8 → 0-3,8"""
    parts, run = [], []
    for i in sorted(ids):
        if run and i == run[-1] + 1:
            run.append(i)
            continue
        if run:
            parts.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
        run = [i]
    if run:
        parts.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ",".join(parts)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _home(path: str) -> str:
    home = os.path.expanduser("~")
    return "~" + path[len(home) :] if home != "/" and path.startswith(home + "/") else path


def asks_for(request: dict) -> str:
    parts = [_plural(request["cpus"], "CPU"), f"{format_size(request['mem'])} RAM"]
    if request["gpus"]:
        parts.append(f"{format_size(request['vram'])} VRAM on {_plural(request['gpus'], 'GPU')}")
    return ", ".join(parts) + f", ~{format_duration(request['time'])}"


def exit_code(job: dict) -> int:
    """What `dibs wait` and `dibs run` exit with for a finished job."""
    if job["state"] == "done":
        return 0
    if job["state"] == "cancelled":
        return 130
    return job["exit_code"] if job.get("exit_code") not in (None, 0) else 1


def finished_line(job: dict) -> str:
    line = f"Job {job['id']} ({job['name']}) {job['state']}"
    if job["state"] != "cancelled" and job.get("exit_code") is not None:
        line += f", exit {job['exit_code']}"
    if job.get("reason") and job["reason"] != "cancelled":
        line += f": {job['reason']}"
    return line


def render_submit(job: dict, now: float) -> str:
    name, job_id = f"Job {job['id']} ({job['name']})", job["id"]
    if job["state"] == "running":
        head = f"{name} started."
    elif job["state"] == "queued" and job.get("predicted_start") is None:
        head = f"{name} queued, but it can't run on this machine as it is: {job['waiting_for']}"
    elif job["state"] == "queued":
        head = (
            f"{name} queued: {ordinal(job['position'])} in line, "
            f"expected to start {when(job['predicted_start'], now)}."
        )
        if job.get("waiting_for"):
            head += f"\n  Waiting: {job['waiting_for']}"
    else:
        return finished_line(job)
    return f"{head}\n  dibs wait {job_id}      dibs logs {job_id} -f"


def _when_cell(job: dict, now: float) -> str:
    state = job["state"]
    if state == "running":
        if job.get("over_estimate"):
            return f"over estimate by {format_duration(job['over_estimate'])}"
        return f"ends {when(job['expected_end'], now)}"
    if state == "queued":
        if job.get("predicted_start") is None:
            return f"can't run here: {job['waiting_for']}"
        if job["predicted_start"] <= now + 0.5:
            return "starting"
        cell = f"starts {when(job['predicted_start'], now)}"
        return cell + (f": {job['waiting_for']}" if job.get("waiting_for") else "")
    cell = f"{state} {clock(job['ended'], now)}" if job.get("ended") else state
    if state != "cancelled" and job.get("exit_code") is not None:
        cell += f", exit {job['exit_code']}"
    if job.get("reason") and job["reason"] != "cancelled":
        cell += f": {job['reason']}"
    return cell


def _vram_cell(job: dict) -> str:
    req = job["request"]
    if not req["vram"]:
        return "-"
    if job["state"] == "running" and job.get("gpu_ids"):
        return f"{format_size(req['vram'])}@{','.join(str(g) for g in job['gpu_ids'])}"
    return format_size(req["vram"]) + (f"×{req['gpus']}" if req["gpus"] > 1 else "")


def _command(argv: list[str], width: int = 60) -> str:
    text = shlex.join(argv)
    return text if len(text) <= width else text[: width - 1] + "…"


def guard_lines(guard: dict, now: float) -> list[str]:
    lines = []
    for e in guard.get("stopped", []):
        used = f"{format_size(e['used'])} on GPU {e['gpu']}" if e.get("used") else f"GPU {e['gpu']}"
        lines.append(f"Stopped outside dibs: pid {e['pid']} {_command(e['command'])} ({used}), {ago(e['time'], now)}")
    for o in guard.get("outside", []):
        used = f"{format_size(o['used'])} on GPU {o['gpu']}" if o.get("used") else f"GPU {o['gpu']}"
        lines.append(f"Outside dibs, left alone: pid {o['pid']} {_command(o['command'])} ({used}): {o['why']}")
    return lines


def render_queue(snapshot: dict, *, color: bool) -> str:
    now = snapshot["now"]
    header = ["JOB", "STATE", "SESSION", "NAME", "CPU", "RAM", "VRAM", "EST", "WHEN"]
    rows = []
    for job in snapshot["jobs"]:
        req = job["request"]
        rows.append(
            [
                str(job["id"]),
                job["state"],
                job["session_label"],
                job["name"],
                str(req["cpus"]),
                format_size(req["mem"]),
                _vram_cell(job),
                format_duration(req["time"]),
                _when_cell(job, now),
            ]
        )
    lines = []
    if rows:
        widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header) - 1)]
        right = {0, 4, 5, 7}  # numbers line up on the right
        for r in [header, *rows]:
            cells = []
            for i, text in enumerate(r[:-1]):
                padded = text.rjust(widths[i]) if i in right else text.ljust(widths[i])
                if color and i == 1 and r is not header:
                    padded = f"\033[{_COLORS.get(text, '0')}m{padded}\033[0m"
                cells.append(padded)
            lines.append("  ".join([*cells, r[-1]]).rstrip())
    else:
        lines.append("Nothing running or queued.")
    extra = guard_lines(snapshot.get("guard", {}), now)
    if extra:
        lines += ["", *extra]
    return "\n".join(lines)


def render_show(job: dict, now: float) -> str:
    state = job["state"]
    title = f"queued ({ordinal(job['position'])} in line)" if state == "queued" and job.get("position") else state
    rows = [
        ("command", shlex.join(job["argv"])),
        ("folder", _home(job["cwd"])),
        ("session", job["session_label"]),
        ("asks for", asks_for(job["request"])),
    ]
    if state == "queued":
        if job.get("predicted_start") is None:
            rows.append(("can't run", job["waiting_for"]))
        else:
            rows.append(
                ("expected", f"start {when(job['predicted_start'], now)}, end ~{clock(job['predicted_end'], now)}")
            )
            if job.get("waiting_for"):
                rows.append(("waiting", job["waiting_for"]))
    elif state == "running":
        got = f"CPUs {cpu_list(job['cpu_ids'])}"
        if job.get("gpu_ids"):
            got += f", GPU {','.join(str(g) for g in job['gpu_ids'])}"
        rows.append(("running", f"since {clock(job['started'], now)} ({format_duration(now - job['started'])}) on {got}"))
        if job.get("over_estimate"):
            rows.append(("expected", f"over estimate by {format_duration(job['over_estimate'])}"))
        else:
            rows.append(("expected", f"end {when(job['expected_end'], now)}"))
    else:
        if job.get("ended"):
            ran = f" after {format_duration(job['ended'] - job['started'])}" if job.get("started") else ""
            code = f", exit {job['exit_code']}" if job.get("exit_code") is not None else ""
            rows.append(("ended", f"{clock(job['ended'], now)}{ran}{code}"))
        if job.get("reason"):
            rows.append(("reason", job["reason"]))
    peaks = []
    if job.get("peak_vram") is not None and job["request"]["vram"]:
        peaks.append(f"VRAM {format_size(job['peak_vram'])} of {format_size(job['request']['vram'])}")
    if job.get("peak_mem") is not None:
        peaks.append(f"RAM {format_size(job['peak_mem'])} of {format_size(job['request']['mem'])}")
    if peaks:
        rows.append(("peak", ", ".join(peaks)))
    rows.append(("output", _home(job["output"])))
    width = max(len(k) for k, _ in rows)
    lines = [f"Job {job['id']}  {job['name']}  {title}"]
    lines += [f"  {k.ljust(width)}  {v}" for k, v in rows]
    return "\n".join(lines)


def render_info(info: dict) -> str:
    cap = info["capacity"]
    gpus = [f"GPU {g['index']} {g['name']} ({format_size(g['vram'])})" for g in cap["gpus"]] or ["no GPUs"]
    enforce = info["enforcement"]

    def yes(flag: bool) -> str:
        return "yes" if flag else "no"

    lines = [
        f"dibs {info['version']} (daemon pid {info['pid']}, build {info.get('build', '?')})",
        f"Hands out  {_plural(len(cap['cpus']), 'CPU')}, {format_size(cap['mem'])} RAM, " + ", ".join(gpus),
        "Enforced   "
        + f"CPU pinning {yes(enforce['cpu_pinning'])} · RAM caps {yes(enforce['ram_cap'])} · "
        + f"VRAM watch {yes(enforce['vram_watch'])} · guard {enforce['guard']}",
        f"GPUs via   {info['gpu_backend']}",
        f"Config     {_home(info['paths']['config'])}",
        f"State      {_home(info['paths']['state'])}",
    ]
    return "\n".join(lines)
