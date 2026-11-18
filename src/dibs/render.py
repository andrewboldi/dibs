"""What people and agents read in their terminal."""

import os
import shlex
import time

from dibs.model import script_of
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


def render_earlier(earlier: list[dict], match: str = "command") -> str:
    """What earlier runs of the same command (or, with match="script", the same script with other
    arguments) used: what to size the next request from."""
    if not earlier:
        return ""
    if match == "script":
        lines = [f"Earlier runs of {script_of(earlier[0].get('argv') or []) or 'this script'} here, with other arguments:"]
    else:
        lines = ["Earlier runs of this command here:"]
    for e in earlier:
        r = e["request"]
        ran = e["ended"] - e["started"] - (e.get("asleep") or 0.0) if e.get("ended") and e.get("started") is not None else None
        line = f"  #{e['id']} {e['state']}" + (f" after {format_duration(ran)}" if ran is not None else "")
        line += f" (estimated {format_duration(r['time'])})"
        used = []
        if e.get("cpu_seconds") is not None and ran:
            used.append(f"{e['cpu_seconds'] / ran:.1f} of {r['cpus']} CPUs")
        if e.get("peak_mem") is not None:
            used.append(f"RAM {format_size(e['peak_mem'])} of {format_size(r['mem'])}")
        if e.get("peak_vram") is not None and r.get("vram"):
            used.append(f"VRAM {format_size(e['peak_vram'])} of {format_size(r['vram'])}")
        if used:
            line += ": " + ", ".join(used)
        if e.get("reason") and e["state"] != "done":
            line += f"; {e['reason']}"
        lines.append(line)
    return "\n".join(lines)


def render_estimates(estimates: dict | None) -> str:
    """A line on how the session's recent jobs ran against their estimates, when they're well off (#95)."""
    if not estimates:
        return ""
    median, n = estimates["median"], estimates["jobs"]
    head = f"Your session's last {n} jobs ran a median {median:.0%} of their estimates"
    if median < 0.5:
        return f"{head}: a closer --time gets jobs started sooner."
    if median > 1.25:
        return f"{head}: a longer --time keeps predictions right."
    return ""


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


def render_momentum(grant: dict | None, now: float) -> str:
    if not grant:
        return "No project has momentum."
    return f"Momentum: {grant['label']} until {when(grant['until'], now)}: {grant['why']}"


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
                job["name"]
                + (f" (p{job['priority']})" if job.get("priority", 5) != 5 else "")
                + (" (alone)" if job["request"].get("exclusive") else ""),
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
    if momentum := snapshot.get("momentum"):  # #94
        lines += ["", render_momentum(momentum, now)]
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
        *([("priority", str(job["priority"]))] if job.get("priority", 5) != 5 else []),
        ("asks for", asks_for(job["request"]) + (", with the machine to itself" if job["request"].get("exclusive") else "")),
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
        asleep = job.get("asleep") or 0.0
        ran = format_duration(max(0.0, now - job["started"] - asleep))
        if asleep >= 60:
            ran += f", plus {format_duration(asleep)} asleep"
        rows.append(("running", f"since {clock(job['started'], now)} ({ran}) on {got}"))
        if job.get("over_estimate"):
            rows.append(("expected", f"over estimate by {format_duration(job['over_estimate'])}"))
        else:
            rows.append(("expected", f"end {when(job['expected_end'], now)}"))
    else:
        if job.get("ended"):
            asleep = job.get("asleep") or 0.0
            ran = f" after {format_duration(max(0.0, job['ended'] - job['started'] - asleep))}" if job.get("started") else ""
            if ran and asleep >= 60:
                ran += f" (plus {format_duration(asleep)} asleep)"
            code = f", exit {job['exit_code']}" if job.get("exit_code") is not None else ""
            rows.append(("ended", f"{clock(job['ended'], now)}{ran}{code}"))
        if job.get("reason"):
            rows.append(("reason", job["reason"]))
    if job.get("background"):  # beside a timing run (#112)
        b = job["background"]
        rows.append(("beside it", f"other programs kept {b['mean']:.1f} cores busy on average, at most {b['max']:.1f}"))
    if state == "running" and job.get("borrowed_ram"):  # #86
        rows.append(("RAM", "started on RAM other jobs reserved but weren't using"))
    for i, attempt in enumerate(job.get("attempts") or [], 1):  # runs that failed for a resource reason (#84)
        rows.append((f"try {i}", attempt.get("reason") or ""))
    peaks = []
    if job.get("peak_vram") is not None and job["request"]["vram"]:
        peaks.append(f"VRAM {format_size(job['peak_vram'])} of {format_size(job['request']['vram'])}")
    if job.get("peak_mem") is not None:
        peaks.append(f"RAM {format_size(job['peak_mem'])} of {format_size(job['request']['mem'])} (not counting file cache)")
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


# dibs top ----------------------------------------------------------------------------------------

# Job colours (#118): fixed 256-colour hues, since terminal themes remap the 16 basic colours (Solarized
# makes the bright ones greys). Mid-lightness, so each reads at 3:1 or more on light and dark
# backgrounds, and in an order where consecutive job ids, the last and first included, stay apart
# for normal and red-green colour-blind vision (OKLab ΔE ≥ 19 and ≥ 10).
_PALETTE = [f"38;5;{n}" for n in (167, 68, 64, 168, 136, 31, 166, 133, 29, 98)]
_TICKS = [10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400, 172800, 604800]
# dibs top's columns, each followed by a space that's always there (#118)
_ID_W, _NAME_W, _SESSION_W, _WHEN_W = 5, 13, 14, 16
_PLAN_LEFT = 4 + _ID_W + 1 + _NAME_W + 1  # where the plan's bars start, under "now"
_NUMBERS = _PLAN_LEFT + _SESSION_W + 1  # where the CPU, RAM and VRAM numbers start
_REASON_MIN = 10  # a reason shows only when this much of it fits
_OUTSIDE_SGR = "38;5;245"  # a grey that reads on light and dark backgrounds: what programs outside dibs use (#120)


def _job_color(job_id: int) -> str:
    return _PALETTE[job_id % len(_PALETTE)]


class _Line:
    """A line of styled pieces that can be cut to a width without counting escape codes."""

    def __init__(self, *parts: tuple[str, str | None]):
        self.parts = list(parts)

    def add(self, text: str, sgr: str | None = None) -> "_Line":
        self.parts.append((text, sgr))
        return self

    def __len__(self) -> int:
        return sum(len(t) for t, _ in self.parts)

    def render(self, width: int, color: bool) -> str:
        """The line cut to width; where visible text is lost, it ends with an ellipsis (#118)."""
        cut = len("".join(t for t, _ in self.parts).rstrip()) > width
        pieces, used = [], 0
        for text, sgr in self.parts:
            text = text[: max(0, (width - 1 if cut else width) - used)]
            used += len(text)
            pieces.append([text, sgr])
        if cut:
            while pieces and not pieces[-1][0].rstrip(" ,;:("):
                pieces.pop()
            if pieces:
                pieces[-1][0] = pieces[-1][0].rstrip(" ,;:(")
                if not pieces[-1][0].endswith("…"):
                    pieces[-1][0] += "…"
        return "".join(f"\033[{s}m{t}\033[0m" if color and s and t else t for t, s in pieces).rstrip()


def _bar(width: int, total: float, shares: list[tuple[float, str]], outside: float = 0) -> list[tuple[str, str | None]]:
    """Jobs' reservations in their colours, then what programs outside dibs use in grey, in what the
    jobs left (#120), then what's free."""
    pieces, filled = [], 0
    for amount, sgr in shares:
        cells = round(amount / total * width) if total else 0
        cells = min(max(cells, 1 if amount > 0 else 0), width - filled)
        if cells:
            pieces.append(("█" * cells, sgr))
            filled += cells
    for i in range(len(pieces) - 1):  # a half-cell gap between neighbouring jobs' segments (#118)
        pieces[i] = (pieces[i][0][:-1] + "▌", pieces[i][1])
    grey = min(width - filled, max(1, round(outside / total * width))) if total and outside > 0 else 0
    if grey:
        pieces.append(("▒" * grey, _OUTSIDE_SGR))
        filled += grey
    pieces.append(("░" * (width - filled), "2"))
    return pieces


def _short(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _short_label(label: str, n: int) -> str:
    """A session label (project·session) cut to n characters from the end like any text, since one
    session often works in several project folders, but never to a piece of its session part, such
    as "project·…" (#118)."""
    if len(label) <= n:
        return label
    kept = label[: n - 1]
    return (kept[: kept.rindex("·")] if "·" in kept else kept) + "…"


def _top_vram(job: dict, several_gpus: bool) -> str:
    """The VRAM cell in dibs top: which GPU a job runs on only when there are several (#118)."""
    if several_gpus or not job["request"]["vram"]:
        return _vram_cell(job)
    return format_size(job["request"]["vram"])


def _job_cells(job: dict, now: float, several_gpus: bool) -> tuple[str, str, str]:
    """(resources, when, why) for a job row: CPUs, RAM and VRAM right-aligned under their labels."""
    req = job["request"]
    width = 8 if several_gpus else 5
    res = f"{req['cpus']:>4} {format_size(req['mem']):>5} {_top_vram(job, several_gpus):>{width}}"
    if job["state"] == "running":
        if job.get("over_estimate"):
            return res, f"over by {format_duration(job['over_estimate'])}", ""
        return res, f"ends in {format_duration(max(0.0, job['expected_end'] - now))}", ""
    if job.get("predicted_start") is None:
        return res, "can't run", job.get("waiting_for", "")
    if job["predicted_start"] <= now + 0.5:
        return res, "starting", ""
    return res, f"starts in {format_duration(job['predicted_start'] - now)}", job.get("waiting_for", "")


def _span(job: dict, now: float) -> tuple[float, float] | None:
    if job["state"] == "running":
        return now, job["expected_end"]
    if job.get("predicted_start") is None:
        return None
    return job["predicted_start"], job["predicted_end"]


def _tick(seconds: float) -> str:
    """A tick label that keeps the units ticks need: 1m10s, 1h30m."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60}s" if s % 60 else f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h{s % 3600 // 60}m" if s % 3600 // 60 else f"{s // 3600}h"
    return format_duration(s)


def _axis(width: int, left: int, span: float) -> tuple[_Line, float]:
    """The timeline's time axis, and the tick interval chosen."""
    area = width - left
    interval = next((t for t in _TICKS if span / t <= max(1, area // 9)), _TICKS[-1])
    axis = [" "] * width
    marks = [(left, "now")]
    t = interval
    while t <= span:
        marks.append((left + int(t / span * (area - 1)), "+" + _tick(t)))
        t += interval
    for x, label in marks:
        if x + len(label) <= width and all(c == " " for c in axis[max(0, x - 1) : x + len(label) + 1]):
            axis[x : x + len(label)] = label
    return _Line((" PLAN".ljust(left), "1"), ("".join(axis[left:]).rstrip(), "2")), interval


def _frame(snapshot: dict, width: int, rows_per_list: int) -> list[_Line]:
    now, cap, guard = snapshot["now"], snapshot["capacity"], snapshot.get("guard", {})
    running = [j for j in snapshot["jobs"] if j["state"] == "running"]
    queued = [j for j in snapshot["jobs"] if j["state"] == "queued"]
    lines: list[_Line] = []

    summary = (
        f"guard {guard.get('mode', '?')} · {_plural(len(cap['cpus']), 'CPU')} · "
        f"{format_size(cap['mem'])} RAM · {_plural(len(cap['gpus']), 'GPU')} "
    )
    head = f" dibs  {time.strftime('%H:%M:%S', time.localtime(now))}"
    lines.append(_Line((" dibs ", "1;7"), (head[6:], None), (summary.rjust(max(0, width - len(head))), "2")))
    lines.append(_Line())

    label = 7
    used_cpu = [(j["request"]["cpus"], _job_color(j["id"])) for j in running]
    used_mem = [(j["request"]["mem"], _job_color(j["id"])) for j in running]
    # What programs outside dibs use (#120), drawn when it's more than a trace; an older daemon sends none.
    outside = snapshot.get("outside")
    away = outside or {}
    cpus_away = away.get("cpus") if (away.get("cpus") or 0) >= 0.5 else 0
    mem_away = away.get("mem") or 0
    rows = [
        ("CPU", len(cap["cpus"]), used_cpu, lambda n: str(round(n)), cpus_away, "busy", None),
        ("RAM", cap["mem"], used_mem, format_size, mem_away if mem_away >= cap["mem"] / 100 else 0, "in use", None),
    ]
    for g in cap["gpus"]:
        shares = [(j["request"]["vram"], _job_color(j["id"])) for j in running if g["index"] in j.get("gpu_ids", [])]
        vram_away = (away.get("vram") or {}).get(str(g["index"]), 0)
        rows.append((f"GPU {g['index']}", g["vram"], shares, format_size, vram_away, "in use", g["name"]))
    # Room for the figures beside the bars, from the totals alone, so the bars keep one width as top redraws.
    room = 14
    if outside is not None:
        room = max(room, *(len(f" {fmt(t)}/{fmt(t)} · {fmt(t)} {verb} outside dibs") + 4 for _, t, _, fmt, _, verb, _ in rows))
    bar_width = max(10, min(48, width - label - room))
    for name, total, shares, fmt, used_away, verb, gpu_name in rows:
        used = sum(a for a, _ in shares)
        line = _Line((f" {name}".ljust(label), "1"), *_bar(bar_width, total, shares, used_away),
                     (f" {fmt(used)}/{fmt(total)}", None))  # fmt: skip
        if used_away:
            line.add(f" · {fmt(used_away)} {verb} outside dibs", "2")
        if gpu_name:
            line.add(f"  {gpu_name}", "2")
        lines.append(line)
    lines.append(_Line())

    if not running and not queued:
        lines.append(_Line(("  Nothing running or queued.", "2")))
    several = len(cap["gpus"]) > 1
    sections = [(title, jobs) for title, jobs in (("RUNNING", running), ("QUEUED", queued)) if jobs]
    for n, (title, jobs) in enumerate(sections):
        head = _Line((f" {title}", "1"), (f"  {len(jobs)}", "2"))
        if n == 0:  # the numbers' labels, over their columns
            head.add(" " * max(1, _NUMBERS - len(head)))
            head.add(f"{'CPUs':>4} {'RAM':>5} {'VRAM':>{8 if several else 5}}", "2")
        else:
            lines.append(_Line())
        lines.append(head)
        for job in jobs[:rows_per_list]:
            res, when_text, why = _job_cells(job, now, several)
            line = _Line(
                ("  ", None),
                ("●", _job_color(job["id"])),  # the job's colour on a mark, its text in plain ink
                (" " + f"#{job['id']}".ljust(_ID_W) + " " + _short(job["name"], _NAME_W).ljust(_NAME_W) + " ", None),
                (_short_label(job["session_label"], _SESSION_W).ljust(_SESSION_W) + " ", "2"),
                (res + "  ", None),
                (when_text.ljust(_WHEN_W) + "  ", None),
            )
            if why and width - len(line) >= _REASON_MIN:
                line.add(why, "2")
            lines.append(line)
        if len(jobs) > rows_per_list:
            lines.append(_Line((f"  … {len(jobs) - rows_per_list} more {title.lower()}", "2")))

    planned = [(j, s) for j in running + queued if (s := _span(j, now))]
    if planned:
        left = _PLAN_LEFT
        horizon = max(end for _, (_, end) in planned)
        span = max(60.0, horizon - now)
        lines.append(_Line())
        axis, _ = _axis(width, left, span)
        lines.append(axis)
        area = max(1, width - left)
        for job, (start, end) in planned[: rows_per_list * 2]:
            x0 = int((max(start, now) - now) / span * (area - 1))
            x1 = max(x0 + 1, int((end - now) / span * (area - 1)))
            lines.append(
                _Line(
                    ("    " + f"#{job['id']}".ljust(_ID_W) + " " + _short(job["name"], _NAME_W).ljust(_NAME_W) + " ",
                     None),  # fmt: skip
                    (" " * x0, None),
                    ("█" * (min(x1, area) - x0), _job_color(job["id"])),
                )
            )
        if len(planned) > rows_per_list * 2:
            lines.append(_Line((f"  … {len(planned) - rows_per_list * 2} more", "2")))

    extra = guard_lines(guard, now)
    if extra:
        lines.append(_Line())
        lines += [_Line((f" {text}", "33" if text.startswith("Stopped") else "2")) for text in extra[:4]]
    return lines


def render_top(snapshot: dict, *, width: int, height: int, color: bool) -> str:
    """One screen of `dibs top`: what's reserved, what runs, what's next, and the plan over time."""
    width = max(width, 20)
    rows = max(1, len(snapshot["jobs"]))
    lines = _frame(snapshot, width, rows)
    while len(lines) > height and rows > 1:
        rows -= 1
        lines = _frame(snapshot, width, rows)
    return "\n".join(line.render(width, color) for line in lines[:height])
