"""What people and agents read in their terminal."""

import os
import re
import shlex
import time
import unicodedata

from dibs.model import GPU_IDLE_AFTER, script_name, script_of
from dibs.units import format_duration, format_size

_COLORS = {"running": "32", "queued": "33", "failed": "31", "done": "2", "cancelled": "2"}
CPU_USE_AFTER = 10.0  # seconds: a job's mean CPU use is shown for runs at least this long (#175)


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
    elif match == "session":  # none here: the session's runs of a script of the same name elsewhere (#229)
        lines = [f"Earlier runs of {script_name(earlier[0].get('argv') or []) or 'this script'} in this session's other folders:"]
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
        if e.get("gpu_idle"):  # it held GPU memory with the GPU idle (#180)
            used.append(f"GPU idle {format_duration(e['gpu_idle'])}")
        if used:
            line += ": " + ", ".join(used)
        if e.get("reason") and e["state"] != "done":
            line += f"; {e['reason']}"
        lines.append(line)
    # Runs long enough to tell (10 s) that reserved VRAM and never showed on the GPU (#139).
    told = [e for e in earlier if e["request"].get("vram") and e.get("ended") and e.get("started") is not None
            and e["ended"] - e["started"] >= 10]  # fmt: skip
    if told and all(e.get("peak_vram") == 0 for e in told):
        lines.append("None of them used the GPU they reserved: leave --vram out, so the job holds no GPU memory "
                     "another job could use.")  # fmt: skip
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


def _sizing(job: dict) -> str:
    """What the job's earlier runs say its RAM request could be, for submit's first line (#212)."""
    sizing = job.get("sizing")
    if not sizing:
        return ""
    fact = f" Its last {sizing['runs']} runs peaked at {format_size(sizing['peak'])}: --mem {format_size(sizing['suggest'])}"
    if job["state"] == "queued":
        return f"{fact} would start it sooner (--replace {job['id']} keeps its place)."
    return f"{fact} would do."


def render_submit(job: dict, now: float) -> str:
    name, job_id = f"Job {job['id']} ({job['name']})", job["id"]
    if job["state"] == "running":
        head = f"{name} started.{_sizing(job)}"
    elif job["state"] == "queued" and job.get("predicted_start") is None:
        head = f"{name} queued, but it can't run on this machine as it is: {job['waiting_for']}"
    elif job["state"] == "queued":
        head = (
            f"{name} queued: {ordinal(job['position'])} in line, "
            f"expected to start {_after(job['after']) if job.get('after') else when(job['predicted_start'], now)}."
            f"{_sizing(job)}"
        )
        if job.get("waiting_for"):
            head += f"\n  Waiting: {job['waiting_for']}"
    else:
        return finished_line(job)
    ask = "" if job.get("title") else (  # #259
        '\n  Next time add --title "what it does, in plain words": the user watches dibs top to see what\'s running.'
    )
    return f"{head}\n  dibs wait {job_id}      dibs logs {job_id} -f{ask}"


def _idle_now(job: dict, now: float) -> float:
    """How long a running job's current hold of GPU memory with the GPU idle has gone on, once it counts
    (#180); 0 if it doesn't."""
    since = job.get("gpu_idle_since")
    return now - since if since is not None and now - since >= GPU_IDLE_AFTER else 0.0


def _idle_total(job: dict, now: float) -> float:
    """All the time a job has held GPU memory with the GPU idle, in stretches that count (#180)."""
    return (job.get("gpu_idle") or 0.0) + _idle_now(job, now)


def _after(after: dict) -> str:
    """A start that rests on a running job past its estimate, whose assumed end moves with every pass (#181)."""
    over, estimate = format_duration(after["over"]), format_duration(after["estimate"])
    return f"when #{after['id']} ends ({over} past its {estimate} estimate)"


def _when_cell(job: dict, now: float) -> str:
    state = job["state"]
    if state == "running":
        if job.get("over_estimate"):
            cell = f"over estimate by {format_duration(job['over_estimate'])}"
        else:
            cell = f"ends {when(job['expected_end'], now)}"
        if idle := _idle_now(job, now):  # #180
            cell += f"; GPU idle {format_duration(idle)}, holding {format_size(job.get('gpu_idle_held') or 0)}"
        return cell
    if state == "queued":
        if job.get("predicted_start") is None:
            return f"can't run here: {job['waiting_for']}"
        if job["predicted_start"] <= now + 0.5:
            return "starting"
        cell = f"starts {_after(job['after']) if job.get('after') else when(job['predicted_start'], now)}"
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
        if "ram" in e:  # the machine was short of memory (#135)
            used = f"{format_size(e['ram'])} of RAM while memory ran short"
        else:
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
                _what(job)
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
        *([("title", job["title"])] if job.get("title") else []),  # #259
        ("command", shlex.join(job["argv"])),
        ("folder", _home(job["cwd"]) + (", run from a copy of its code made at submit" if job.get("snapshot") else "")),
        ("session", job["session_label"]),
        *([("priority", str(job["priority"]))] if job.get("priority", 5) != 5 else []),
        ("asks for", asks_for(job["request"]) + (", with the machine to itself" if job["request"].get("exclusive") else "")),
    ]
    if state == "queued":
        if job.get("predicted_start") is None:
            rows.append(("can't run", job["waiting_for"]))
        else:
            if job.get("after"):
                rows.append(("expected", f"start {_after(job['after'])}"))
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
            # Agents keep this line and drop the others (#175): which try it was, and why it failed.
            tries = [a for a in job.get("attempts") or [] if a.get("started") is not None]
            if tries:
                earlier = "; ".join(
                    f"the {ordinal(i)} ended after {format_duration(max(0.0, a['ended'] - a['started'] - (a.get('asleep') or 0.0)))}"
                    + (f": {a['reason'].split(';')[0]}" if a.get("reason") else "")
                    for i, a in enumerate(tries, 1) if a.get("ended") is not None
                )  # fmt: skip
                code += f", on its {ordinal(len(tries) + 1)} try" + (f" ({earlier})" if earlier else "")
            why = f": {job['reason']}" if job.get("reason") else ""
            rows.append(("ended", f"{clock(job['ended'], now)}{ran}{code}{why}"))
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
        grown = f", grown to a {format_size(job['ram_limit'])} limit" if (job.get("ram_limit") or 0) > job["request"]["mem"] else ""
        spilled = f", {format_size(job['peak_swap'])} more spilled into swap" if job.get("peak_swap") else ""  # #257
        peaks.append(f"RAM {format_size(job['peak_mem'])} of {format_size(job['request']['mem'])}{grown} (not counting file cache){spilled}")
    cpu = _cpu_use(job, now)
    if cpu:
        peaks.append(cpu)
    if idle := _idle_total(job, now):  # #180
        peaks.append(f"GPU idle {format_duration(idle)} ({format_size(job.get('gpu_idle_held') or 0)})")
    if peaks:
        rows.append(("peak", ", ".join(peaks)))
    if job.get("snapshot_kept"):  # #172
        rows.append(("copy", f"it wrote files into its copy of the code, kept at {_home(job['snapshot_kept'])}"))
    rows.append(("output", _home(job["output"])))
    width = max(len(k) for k, _ in rows)
    lines = [f"Job {job['id']}  {job['name']}  {title}"]
    lines += [f"  {k.ljust(width)}  {v}" for k, v in rows]
    return "\n".join(lines)


def _cpu_use(job: dict, now: float) -> str:
    """A job's mean CPU use over its run, for runs long enough to tell (#175): "CPU 3.3 of 4"."""
    if job.get("cpu_seconds") is None or job.get("started") is None:
        return ""
    ran = (job.get("ended") or now) - job["started"] - (job.get("asleep") or 0.0)
    if ran < CPU_USE_AFTER:
        return ""
    return f"CPU {job['cpu_seconds'] / ran:.1f} of {job['request']['cpus']}"


def failed_notice(notice: dict) -> str:
    """A job of the session's that failed, told at each of its dibs commands until it looks (#176). By
    number only: loops count dibs queue's lines that match a job's name, a session or a state."""
    when = time.strftime("%H:%M", time.localtime(notice["time"]))
    line = f"job {notice['job']} failed at {when}"
    if notice.get("ran") is not None:
        line += f" after {format_duration(max(0.0, notice['ran']))}"
    if notice.get("exit_code") is not None:
        line += f", exit {notice['exit_code']}"
    if notice.get("reason"):
        line += f": {notice['reason']}"
    return line + ". (Told at each dibs command for an hour, until dibs show, wait or logs looks at it.)"


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
_OUTSIDE_SGR = "38;5;245"  # a grey that reads on light and dark backgrounds: what programs outside dibs use (#120)
# dibs top is an instrument panel (#122): rounded panels, quiet borders with their titles in them.
_BORDER = "38;5;244"  # a mid grey: thin lines need more than dim to show on light and dark (3.5:1 or more)
_WARN_SGR = "38;5;136"  # a job past its estimate, and what the guard stopped
_LOAD_SGR = ("38;5;29", "38;5;136", "38;5;167")  # the load row with under half the cores busy, under 85%, more
_SPARK = "▁▂▃▄▅▆▇█"
# The job rows' columns, each followed by a space that's always there (#118)
_ID_W, _NAME_W, _WHEN_W, _PROGRESS_W = 4, 13, 16, 10  # "starts in 12h30m" is 16
_NAME_MAX = 80  # titles (at most 80 characters), or names, get up to this much room, as the width allows (#143, #259, #263)
_PLAN_MIN = 30  # what names leave a row for its bar or timeline and its reason
_REASON_MIN = 10  # a reason shows only when this much of it fits
_METER_MAX = 60  # the widest a meter gets


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

    def fit(self, width: int) -> "_Line":
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
        return _Line(*((t, s) for t, s in pieces if t))

    def render(self, width: int, color: bool) -> str:
        return "".join(f"\033[{s}m{t}\033[0m" if color and s else t for t, s in self.fit(width).parts).rstrip()


def _spaced(pieces) -> list[tuple[str, str | None]]:
    out: list[tuple[str, str | None]] = [(" ", None)]
    for i, piece in enumerate(pieces):
        out += [piece, (" ", None)] if i < len(pieces) - 1 else [piece]
    return out + [(" ", None)]


def _border(corners: str, width: int, *, title=(), note=()) -> _Line:
    """A panel's top or bottom border, the terminal's width: a title on the left, a note on the right."""
    n = max(0, width - 2)
    cells: list[tuple[str, str | None]] = [("─", _BORDER)] * n
    taken = [False] * n

    def place(x: int, pieces) -> None:
        size = sum(len(t) for t, _ in pieces)
        if x < 1 or x + size > n - 1 or any(taken[max(0, x - 1) : x + size + 1]):
            return
        for text, sgr in pieces:
            for ch in text:
                cells[x], taken[x] = (ch, sgr), True
                x += 1

    if title:
        place(1, _spaced(title))
    if note:
        piece = _spaced(note)
        place(n - 1 - sum(len(t) for t, _ in piece), piece)
    line = _Line((corners[0], _BORDER))
    for ch, sgr in cells:
        if line.parts[-1][1] == sgr:
            line.parts[-1] = (line.parts[-1][0] + ch, sgr)
        else:
            line.parts.append((ch, sgr))
    return line.add(corners[1], _BORDER)


def _panel(width: int, rows: list[_Line], *, title, note=(), foot=(), foot_note=()) -> list[_Line]:
    """A rounded panel the terminal's width, each row fitted inside its borders so they never drift."""
    inner = max(1, width - 4)
    out = [_border("╭╮", width, title=title, note=note)]
    for row in rows:
        fitted = row.fit(inner)
        out.append(_Line(("│ ", _BORDER), *fitted.parts, (" " * (inner - len(fitted)), None), (" │", _BORDER)))
    out.append(_border("╰╯", width, title=foot, note=foot_note))
    return out


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


def _spark(cores: list[float | None], total: int, width: int) -> list[tuple[str, str | None]]:
    """The load row: the machine's busy cores over time, the newest on the right."""
    n = len(cores)
    if n <= width:
        cells = [None] * (width - n) + list(cores)
    else:
        cells = []
        for i in range(width):
            known = [v for v in cores[i * n // width : (i + 1) * n // width] if v is not None]
            cells.append(sum(known) / len(known) if known else None)
    pieces: list[tuple[str, str | None]] = []
    for v in cells:
        share = (v or 0) / total if total else 0
        ch, sgr = (" ", None) if v is None else (
            _SPARK[min(7, max(0, round(share * 8) - 1))], _LOAD_SGR[0 if share < 0.5 else 1 if share < 0.85 else 2])
        if pieces and pieces[-1][1] == sgr:
            pieces[-1] = (pieces[-1][0] + ch, sgr)
        else:
            pieces.append((ch, sgr))
    return pieces


def _what(job: dict) -> str:
    """What a person sees a job as: its title, what it does in plain words, else its name (#259)."""
    return job.get("title") or job["name"]


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


def _grid(several: bool, name_w: int = _NAME_W) -> tuple[int, int, int]:
    """Inside a job panel: where the CPU, RAM and VRAM numbers start, the VRAM column's width, and
    where the progress bar or the plan's timeline starts. Every agent's panel shares it (#167)."""
    numbers = 2 + _ID_W + 1 + name_w + 1
    vram = 8 if several else 5
    return numbers, vram, numbers + 3 + 1 + 5 + 1 + vram + 2 + _WHEN_W + 2


def _name_width(jobs: list[dict], inner: int, several: bool) -> int:
    """The name column: as wide as the longest name shown, from _NAME_W up to _NAME_MAX, leaving each
    row _PLAN_MIN for its bar or timeline and its reason (#143)."""
    longest = max((len(_what(j)) for j in jobs), default=0)
    room = inner - _grid(several, 0)[2] - _PLAN_MIN
    return max(_NAME_W, min(longest, _NAME_MAX, room))


def _short_reason(job: dict) -> str:
    """Why a job waits, in a few words: what it lacks (its sizes are in its row, and the meters say
    what's in use), or the job ahead in line it would delay; else the planner's words."""
    if job.get("behind") is not None:
        return f"behind #{job['behind']}"
    if job.get("lacks"):
        return "needs " + ", ".join(job["lacks"])
    # The mem meter already says how much is in use outside dibs; every row repeated it.
    return re.sub(r", [\d.]+[KMGT]? in use outside dibs", "", job.get("waiting_for", ""))


def _job_columns(job: dict, dot: str, several: bool, name_w: int = _NAME_W) -> _Line:
    req = job["request"]
    line = _Line((dot, _job_color(job["id"])),
                 (f" {str(job['id']).ljust(_ID_W)} {_short(_what(job), name_w).ljust(name_w)} ", None))  # fmt: skip
    vram = 8 if several else 5
    return line.add(f"{req['cpus']:>3} {format_size(req['mem']):>5} {_top_vram(job, several):>{vram}}  ", None)


def _running_row(job: dict, now: float, several: bool, name_w: int = _NAME_W) -> _Line:
    """A running job: its resources, when it ends, and a bar of its time so far against its estimate."""
    line = _job_columns(job, "●", several, name_w)
    over, estimate = job.get("over_estimate") or 0, job["request"]["time"]
    if over:
        line.add(f"over by {format_duration(over)}".ljust(_WHEN_W) + "  ", _WARN_SGR)
        done = _PROGRESS_W
    else:
        left = max(0.0, job["expected_end"] - now)
        line.add(f"ends in {format_duration(left)}".ljust(_WHEN_W) + "  ", None)
        done = round(min(1.0, max(0.0, 1 - left / estimate)) * _PROGRESS_W) if estimate else 0
    line.add("▰" * done, _WARN_SGR if over else _job_color(job["id"])).add("▱" * (_PROGRESS_W - done), "2")
    if idle := _idle_now(job, now):  # holding GPU memory with the GPU idle (#180)
        line.add(f"  GPU idle {format_duration(idle)}", _WARN_SGR)
    return line


def _machine(snapshot: dict, width: int) -> list[_Line]:
    """The machine panel: a meter for CPU, RAM and each GPU, and the load over the last minutes."""
    now, cap, guard = snapshot["now"], snapshot["capacity"], snapshot.get("guard", {})
    running = [j for j in snapshot["jobs"] if j["state"] == "running"]
    outside = snapshot.get("outside")  # what programs outside dibs use (#120); an older daemon sends none
    away = outside or {}
    cpus_away = away.get("cpus") if (away.get("cpus") or 0) >= 0.5 else 0
    mem_away = away.get("mem") or 0
    rows = [
        ("cpu", len(cap["cpus"]), [(j["request"]["cpus"], _job_color(j["id"])) for j in running],
         lambda n: str(round(n)), cpus_away, "busy outside"),
        ("mem", cap["mem"], [(max(j["request"]["mem"], j.get("ram_limit") or 0), _job_color(j["id"])) for j in running], format_size,
         mem_away if mem_away >= cap["mem"] / 100 else 0, "used outside"),
    ]  # fmt: skip
    for g in cap["gpus"]:
        shares = [(j["request"]["vram"], _job_color(j["id"])) for j in running if g["index"] in j.get("gpu_ids", [])]
        rows.append((f"gpu {g['index']}" if len(cap["gpus"]) > 1 else "gpu", g["vram"], shares, format_size,
                     (away.get("vram") or {}).get(str(g["index"]), 0), "used outside"))  # fmt: skip
    # The figures' room depends on the totals alone, so the meters keep one width as top redraws.
    room = max(len(f"{fmt(t)}/{fmt(t)}") + (len(f"   {fmt(t)} {verb}") if outside is not None else 0)
               for _, t, _, fmt, _, verb in rows) + 4  # fmt: skip
    meter = max(10, min(_METER_MAX, width - 4 - 6 - 2 - 2 - room))
    lines = []
    for name, total, shares, fmt, used_away, verb in rows:
        used = sum(a for a, _ in shares)
        line = _Line((f"{name:<6}", None), ("▕", _BORDER), *_bar(meter, total, shares, used_away), ("▏", _BORDER),
                     (f"  {fmt(used)}/{fmt(total)}", None))  # fmt: skip
        if used_away:
            line.add(f"   {fmt(used_away)} {verb}", "2")
        if name == "cpu" and (speed := snapshot.get("cpu_cap")):  # every CPU job runs slower (#179)
            line.add(f"   capped at {speed['capped_mhz'] / 1000:.1f} GHz of {speed['max_mhz'] / 1000:.1f}", _WARN_SGR)
        lines.append(line)
    holders = [f"{h['name']} {format_size(h['ram'])}" + (f" (+{format_size(h['swap'])} swap)" if h["swap"] >= 2**27 else "")
               for h in snapshot.get("outside_holders") or []]  # fmt: skip
    if (snapshot.get("tmp_files") or 0) >= 2**30:  # charged to whichever program wrote them (#164)
        holders.append(f"files in /tmp {format_size(snapshot['tmp_files'])}")
    if holders:
        lines.append(_Line(("       ", None), ("outside dibs: " + ", ".join(holders), "2")))
    if snapshot.get("disk"):  # a disk the jobs use is low (#211)
        lines.append(_Line(("       ", None), ("disk: " + snapshot["disk"], _WARN_SGR)))
    load = snapshot.get("load") or {}
    cores = load.get("cores") or []
    if sum(v is not None for v in cores) >= 2:
        span = format_duration(len(cores) * load.get("every", 4.0))
        lines.append(_Line(("load   ", None), *_spark(cores, len(cap["cpus"]), meter), (f"   busy cores, last {span}", "2")))
    names = sorted({g["name"] for g in cap["gpus"]})
    gpus = ", ".join(names) if len(names) > 1 else (f"{len(cap['gpus'])}× " if len(cap["gpus"]) > 1 else "") + "".join(names)
    return _panel(width, lines, title=[("dibs", "1")], note=[(time.strftime("%H:%M:%S", time.localtime(now)), None)],
                  foot=[(gpus, "2")] if gpus else (), foot_note=[(f"guard {guard.get('mode', '?')}", "2")])  # fmt: skip


_ESCAPES = re.compile(r"\x1b(?:\[[0-9;?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)?|.)")


def _plain(text: str) -> str:
    """Text the user typed, such as a session's name, made safe to draw: no escape codes or other
    control characters (they would restyle or move the screen), no zero-width marks, and wide
    characters (CJK, most emoji) as "?", since each takes two cells and would push the border out."""
    out = []
    for ch in _ESCAPES.sub("", text):
        kind = unicodedata.category(ch)
        if kind[0] == "C" or kind in ("Mn", "Me"):
            out.append(" " if ch in "\t\n\r" else "")
        else:
            out.append("?" if unicodedata.east_asian_width(ch) in ("W", "F") else ch)
    return " ".join("".join(out).split())


def _agent_title(name: str, own_name: bool, running: int, waiting: int, width: int) -> list[tuple[str, str]]:
    """A panel's title: the agent's name, cut to leave room for how many of its jobs run and wait,
    so the title always fits its border and never disappears."""
    counts = " · ".join(part for part in (f"{running} running" if running else "",
                                          f"{waiting} waiting" if waiting else "") if part)  # fmt: skip
    room = max(1, width - 8 - len(counts))  # the border's corners, its margins, and the gaps around each piece
    cut = _short(name, room) if own_name else _short_label(name, room)
    return [(cut, "1"), (" " + counts, "2")]


def _axis_row(inner: int, marks: list[tuple[int, str]]) -> _Line:
    """The waiting jobs' time axis, a row above their Gantt bars: "now" where a job starting now
    starts, then a tick label at each step, whole labels only, never touching."""
    cells, end = [" "] * inner, -2
    for col, label in marks:
        if col < end + 2 or col + len(label) > inner:
            continue
        cells[col : col + len(label)] = label
        end = col + len(label) - 1
    return _Line(("".join(cells).rstrip(), "2"))


def _sections(snapshot: dict, width: int, names: dict) -> tuple[list[_Line], list[tuple[_Line, bool]]]:
    """The machine panel, which stays put, and the lines below it, which scroll: a panel for each
    agent with its running jobs then its waiting ones (#167), and the guard's. Each of the lines
    below says whether it's a job's row, so the scroll line can count the jobs out of sight."""
    now, cap, guard = snapshot["now"], snapshot["capacity"], snapshot.get("guard", {})
    running = [j for j in snapshot["jobs"] if j["state"] == "running"]
    queued = [j for j in snapshot["jobs"] if j["state"] == "queued"]
    inner = max(1, width - 4)
    several = len(cap["gpus"]) > 1
    name_w = _name_width(running + queued, inner, several)
    numbers, vram_w, graphic = _grid(several, name_w)
    body: list[tuple[_Line, bool]] = []
    if not running and not queued:
        body += [(line, False) for line in _panel(width, [
            _Line(("Nothing running or waiting.", None)),
            _Line(("Queue a job: dibs submit --cpus 4 --mem 8G --time 20m -- python train.py", "2"))],
            title=[("queue", "1")])]  # fmt: skip
    # One time scale for every agent's Gantt bars, so bars in different panels compare.
    planned = [j for j in queued if j.get("predicted_start") is not None]
    span = max(60.0, max((j["predicted_end"] for j in planned), default=now + 60) - now)
    timeline = max(8, (inner - graphic) * 3 // 5)
    x = lambda t: int((max(t, now) - now) / span * (timeline - 1))  # noqa: E731
    interval = next((t for t in _TICKS if span / t <= max(1, timeline // 7)), _TICKS[-1])
    axis = _axis_row(inner, [(graphic, "now")] + [(graphic + x(now + t), "+" + _tick(t))
                                                  for t in range(int(interval), int(span) + 1, int(interval))])  # fmt: skip

    def waiting_row(job: dict) -> _Line:
        line = _job_columns(job, "○", several, name_w)
        start = job.get("predicted_start")
        if start is None:
            line.add("can't run".ljust(_WHEN_W) + "  ", None)
        else:
            line.add(("starting" if start <= now + 0.5 else f"after #{job['after']['id']}" if job.get("after")
                      else f"starts in {format_duration(start - now)}").ljust(_WHEN_W) + "  ", None)  # fmt: skip
            x0, x1 = x(start), min(timeline, max(x(start) + 1, x(job["predicted_end"])))
            line.add(" " * x0).add("━" * (x1 - x0), _job_color(job["id"]))
        why = _short_reason(job)
        shown = _Line((why, None)).fit(max(0, inner - len(line) - 2))
        if why and len(shown) >= min(len(why), _REASON_MIN):  # enough of it, or none
            line.add("  " + why, "2")
        return line

    # Panels go by each agent's oldest job, so they keep their places as jobs start and end.
    agents: dict[str, dict] = {}
    for job in running + queued:
        agent = agents.setdefault(job["session"], {"running": [], "queued": [], "oldest": job})
        agent["running" if job["state"] == "running" else "queued"].append(job)
        if job["id"] < agent["oldest"]["id"]:
            agent["oldest"] = job
    for session, agent in sorted(agents.items(), key=lambda item: item[1]["oldest"]["id"]):
        own = _plain(names.get(session) or "")
        title = _agent_title(own or _plain(agent["oldest"]["session_label"]), bool(own), len(agent["running"]),
                             len(agent["queued"]), width)  # fmt: skip
        rows = [(_running_row(j, now, several, name_w), True) for j in agent["running"]]
        if agent["queued"]:
            rows += [(axis, False)] + [(waiting_row(j), True) for j in agent["queued"]]
        panel = _panel(width, [row for row, _ in rows], title=title)
        body += list(zip(panel, [False] + [is_job for _, is_job in rows] + [False]))
    extra = guard_lines(guard, now)
    if extra:
        rows = [_Line((text, _WARN_SGR if text.startswith("Stopped") else "2")) for text in extra[:4]]
        body += [(line, False) for line in _panel(width, rows, title=[("guard", "1")])]
    fixed = _machine(snapshot, width)
    if agents:  # the columns' names, once, over every panel's rows, as htop's header row is
        head = [" "] * width
        for cell, label in ((numbers, "cpus"), (numbers + 7, "ram"), (numbers + 7 + vram_w, "vram")):
            head[cell + 1 : cell + 1 + len(label)] = label  # ending where its column's numbers end
        fixed.append(_Line(("".join(head).rstrip(), "2")))
    return fixed, body


def _scroll_line(above: int, hidden_above: bool, below: int, hidden_below: bool) -> _Line:
    """The bottom line while the panels scroll: how many jobs are out of sight each way, and the keys."""
    def side(arrow: str, jobs: int, hidden: bool, where: str) -> str:
        if jobs:
            return f"{arrow} {jobs} job{'s' if jobs != 1 else ''} {where}"
        return f"{arrow} more {where}" if hidden else ""

    parts = [p for p in (side("↑", above, hidden_above, "above"), side("↓", below, hidden_below, "below")) if p]
    return _Line((" " + " · ".join(parts + ["↑↓ PgUp PgDn Home End to scroll"]), "2"))


def top_screen(snapshot: dict, *, width: int, height: int, color: bool, offset: int = 0,
               names: dict | None = None) -> tuple[str, int, int]:  # fmt: skip
    """One screen of `dibs top` (#122), scrolled offset lines down when the panels don't fit (#167):
    the screen, the offset kept within what there is to scroll, and how many lines a page shows."""
    width = max(width, 30)
    fixed, body = _sections(snapshot, width, names or {})
    if len(fixed) + len(body) <= height:
        lines, offset, page = fixed + [line for line, _ in body], 0, len(body)
    else:
        page = max(1, height - len(fixed) - 1)
        offset = max(0, min(offset, len(body) - page))
        out_above, out_below = body[:offset], body[offset + page :]
        lines = fixed + [line for line, _ in body[offset : offset + page]] + [_scroll_line(
            sum(is_job for _, is_job in out_above), bool(out_above),
            sum(is_job for _, is_job in out_below), bool(out_below))]  # fmt: skip
    return "\n".join(line.render(width, color) for line in lines[:height]), offset, page


def render_top(snapshot: dict, *, width: int, height: int, color: bool, offset: int = 0,
               names: dict | None = None) -> str:  # fmt: skip
    """One screen of `dibs top`: an instrument panel of what's reserved, what runs, what's next (#122)."""
    return top_screen(snapshot, width=width, height=height, color=color, offset=offset, names=names)[0]
