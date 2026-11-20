"""The dibs command."""

import argparse
import contextlib
import json
import os
import re
import select
import shutil
import signal
import stat
import sys
import termios
import time
import tty
from pathlib import Path

from dibs import __version__, agents
from dibs import client as dc
from dibs import render
from dibs.paths import Paths, get_paths
from dibs.units import UnitError, format_duration, parse_duration, parse_size

COMMAND_HINT = "put the command after --, like: dibs submit --cpus 4 --mem 16G --time 20m -- python train.py"


class _Interrupted(Exception):
    def __init__(self, signum: int):
        self.signum = signum


def _raise_interrupted(signum, frame):
    raise _Interrupted(signum)


def _cpus(text: str):
    if text == "all":
        return "all"
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} isn't a number of CPUs (or all)") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _priority(text: str) -> int:
    if not text.isdigit() or not 1 <= int(text) <= 10:
        raise argparse.ArgumentTypeError(f"{text!r}: give a whole number from 1 to 10")
    return int(text)


def _size(text: str):
    if text == "all":
        return "all"
    try:
        return parse_size(text)
    except UnitError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _duration(text: str) -> float:
    try:
        return parse_duration(text)
    except UnitError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dibs",
        description="A job queue for one machine: reserve CPUs, RAM and GPU memory, and run when it's your turn.",
    )
    parser.add_argument("--version", action="version", version=f"dibs {__version__}")
    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")
    for name, about in (
        ("submit", "queue a job and print its id and when it's expected to start"),
        ("run", "queue a job, show its output as it runs, and exit with its exit code"),
    ):
        s = sub.add_parser(name, help=about, description=f"{about}. {COMMAND_HINT}.")
        s.add_argument("-c", "--cpus", type=_cpus, required=True, help="CPUs to reserve: a number, or all")
        s.add_argument("-m", "--mem", type=_size, required=True, help="RAM to reserve, like 16G (or all)")
        s.add_argument(
            "-g", "--vram", type=_size, help="GPU memory to reserve on each GPU, like 6G (or all); omit for no GPU"
        )
        s.add_argument("--gpus", type=int, help="how many GPUs (default: 1 when --vram is given)")
        s.add_argument(
            "-t", "--time", type=_duration, required=True,
            help="how long you expect it to take, like 20m or 1h30m (only orders the queue; never enforced)",
        )  # fmt: skip
        s.add_argument(
            "-p", "--priority", type=_priority, default=5,
            help="1-10 (default 5): higher goes ahead sooner, and your session's next job pays the time back",
        )  # fmt: skip
        s.add_argument("-n", "--name", help="a label for the job (default: the script or program name)")
        s.add_argument(
            "--exclusive", action="store_true",
            help="nothing else runs beside it (for timing runs); it waits until no other job runs",
        )  # fmt: skip
        s.add_argument(
            "--retry", action="store_true",
            help="it's safe to run again from the start, over whatever a stopped try left (it resumes, or "
            "repeating does no harm): if it runs out of VRAM or RAM or a restart kills it, dibs runs it again",
        )  # fmt: skip
        s.add_argument("--no-retry", action="store_true", help=argparse.SUPPRESS)  # the default since #116
        s.add_argument(
            "--replace", type=int, metavar="ID",
            help="cancel job ID and queue this one in its place, in one step, so no other job starts in between",
        )  # fmt: skip
        s.add_argument("-o", "--output", help="write the job's output to this file instead of dibs's log")
        if name == "submit":
            s.add_argument("--json", action="store_true", help="print the job as JSON")

    m = sub.add_parser(
        "momentum", help="show, record or end momentum: a rare boost for a project, after the user's yes",
        description="Momentum is for a project working better than expected and held up by another project's jobs, "
        "with more experiments to run. Ask the user first; record it only after they say yes.",
    )  # fmt: skip
    m.add_argument("--for", dest="duration", type=_duration, metavar="DURATION", help="record it for this long, like 6h")
    m.add_argument("--why", help="what is going better than expected, and that the user agreed")
    m.add_argument("--end", action="store_true", help="end it now")

    q = sub.add_parser("queue", aliases=["q"], help="running and queued jobs, with predicted start times")
    q.add_argument("--all", action="store_true", help="also show jobs that finished in the last day")
    q.add_argument("--mine", action="store_true", help="only this session's jobs")
    q.add_argument("--json", action="store_true")

    s = sub.add_parser("show", aliases=["status"], help="everything about one job")
    s.add_argument("id", type=int)
    s.add_argument("--json", action="store_true")

    w = sub.add_parser("wait", help="wait for jobs to finish; exits with the job's exit code")
    w.add_argument("ids", type=int, nargs="+", metavar="ID")
    w.add_argument("--timeout", type=_duration, help="give up after this long, like 9m (exit 124)")
    w.add_argument("--json", action="store_true")

    lg = sub.add_parser("logs", help="print a job's output")
    lg.add_argument("id", type=int)
    lg.add_argument("-f", "--follow", action="store_true", help="keep printing until the job ends")
    lg.add_argument("-n", "--lines", type=int, help="only the last LINES lines")

    c = sub.add_parser("cancel", help="cancel queued or running jobs")
    c.add_argument("ids", type=int, nargs="+", metavar="ID")
    c.add_argument("--json", action="store_true")

    sub.add_parser("top", help="live view: what's reserved, what runs, what's next")

    i = sub.add_parser("info", help="what dibs hands out and what it enforces")
    i.add_argument("--json", action="store_true")

    st = sub.add_parser("setup", help="tell your agent CLIs about dibs and keep it running (turns the guard on)")
    st.add_argument("--dry-run", action="store_true", help="only show what would change")
    st.add_argument("--undo", action="store_true", help="remove everything setup added")
    st.add_argument("--no-service", action="store_true", help="don't install the systemd user service")

    sub.add_parser("agent-help", help="the full instructions for agents")

    d = sub.add_parser("daemon", help="start, stop or check the background scheduler")
    d.add_argument("action", choices=["start", "stop", "restart", "status", "run"])
    return parser


def _print_json(data) -> None:
    print(json.dumps(data, indent=2))


def _submit_request(args, command: list[str]) -> dict:
    cwd = os.getcwd()
    session, label = dc.current_session(os.environ, cwd)
    return {
        "argv": command,
        "cwd": cwd,
        "env": dict(os.environ),
        "session": session,
        "session_label": label,
        "name": args.name,
        "priority": args.priority,
        "replace": args.replace,
        "retry": args.retry and not args.no_retry,
        "output": os.path.abspath(args.output) if args.output else None,
        "request": {
            "cpus": args.cpus, "mem": args.mem, "vram": args.vram, "gpus": args.gpus, "time": args.time,
            "exclusive": args.exclusive,
        },  # fmt: skip
    }


def _copy_new(path: str, position: int, sink) -> int:
    try:
        with open(path, "rb") as f:
            f.seek(position)
            data = f.read()
    except FileNotFoundError:
        return position
    if data:
        sink.write(data)
        sink.flush()
    return position + len(data)


def _follow(client: dc.Client, job_id: int, path: str, position: int, sink) -> dict:
    """Copy the job's output to sink as it's written, until the job ends. Returns the finished job."""
    while True:
        position = _copy_new(path, position, sink)
        try:
            response = client.call("wait", ids=[job_id], timeout=0.5)
        except (ConnectionResetError, BrokenPipeError):
            time.sleep(0.5)
            continue
        if not response["timed_out"]:
            _copy_new(path, position, sink)
            return response["jobs"][0]


def _keeps_running(job_id: int, signum: int) -> int:
    print(
        f"\ndibs: job {job_id} keeps running. Wait for it: dibs wait {job_id}   Stop it: dibs cancel {job_id}",
        file=sys.stderr,
    )
    return 128 + signum


def cmd_submit(args, paths: Paths, command: list[str]) -> int:
    response = dc.Client(paths).call("submit", **_submit_request(args, command))
    earlier, match = response.get("earlier", []), response.get("earlier_match", "command")
    if args.json:
        _print_json({**response["job"], "earlier": earlier, "earlier_match": match, "replaced": response.get("replaced")})
    else:
        print(render.render_submit(response["job"], response["now"]))
        if response.get("replaced") is not None:
            print(f"It replaces job {response['replaced']}, which is cancelled.")
        if earlier:
            print(render.render_earlier(earlier, match))
        if line := render.render_estimates(response.get("estimates")):
            print(line)
    sys.stdout.flush()  # the job's lines first, in a merged stream too (#169)
    _tell(paths, failures=not args.json)
    return 0


def cmd_run(args, paths: Paths, command: list[str]) -> int:
    client = dc.Client(paths)
    response = client.call("submit", **_submit_request(args, command))
    job = response["job"]
    print(render.render_submit(job, response["now"]), file=sys.stderr)
    if response.get("replaced") is not None:
        print(f"It replaces job {response['replaced']}, which is cancelled.", file=sys.stderr)
    if response.get("earlier"):
        print(render.render_earlier(response["earlier"], response.get("earlier_match", "command")), file=sys.stderr)
    if line := render.render_estimates(response.get("estimates")):
        print(line, file=sys.stderr)
    _tell(paths)
    try:
        finished = _follow(client, job["id"], job["output"], 0, sys.stdout.buffer)
    except KeyboardInterrupt:
        return _keeps_running(job["id"], signal.SIGINT)
    except _Interrupted as e:
        return _keeps_running(job["id"], e.signum)
    except dc.DibsError as e:
        if e.code != "upgraded":
            raise
        # Not "run the command again": the job is submitted already, and would be submitted twice.
        print(
            f"\ndibs: dibs was upgraded while this command ran, and job {job['id']} keeps running. "
            f"Wait for it: dibs wait {job['id']}   Follow it: dibs logs {job['id']} -f",
            file=sys.stderr,
        )
        return 1
    if finished["state"] != "done":
        print(f"dibs: {render.finished_line(finished)}", file=sys.stderr)
    return render.exit_code(finished)


def cmd_momentum(args, paths: Paths, _) -> int:
    client = dc.Client(paths)
    if args.end:
        client.call("momentum", action="end")
        print("Momentum ended.")
        return 0
    if args.duration is None:
        print(render.render_momentum(client.call("momentum", action="show")["momentum"], time.time()))
        return 0
    session, label = dc.current_session(os.environ, os.getcwd())
    grant = client.call("momentum", action="grant", session=session, session_label=label,
                        seconds=args.duration, why=args.why)["momentum"]  # fmt: skip
    print(f"{grant['label']} has momentum until {render.when(grant['until'], time.time())}.")
    return 0


def cmd_queue(args, paths: Paths, _) -> int:
    session = dc.current_session()[0] if args.mine else None
    snapshot = dc.Client(paths).call("queue", all=args.all, session=session)
    if args.json:
        _print_json({k: v for k, v in snapshot.items() if k != "ok"})
    else:
        print(render.render_queue(snapshot, color=sys.stdout.isatty()))
    return 0


def cmd_show(args, paths: Paths, _) -> int:
    response = dc.Client(paths).call("show", id=args.id)
    if args.json:
        _print_json(response["job"])
    else:
        print(render.render_show(response["job"], response["now"]))
    return 0


def cmd_wait(args, paths: Paths, _) -> int:
    try:
        response = dc.Client(paths).wait(args.ids, args.timeout)
    except KeyboardInterrupt:
        return _keeps_running(args.ids[0], signal.SIGINT)
    except _Interrupted as e:
        return _keeps_running(args.ids[0], e.signum)
    if args.json:
        _print_json({"jobs": response["jobs"], "timed_out": response["timed_out"]})
    if response["timed_out"]:
        for job in response["jobs"]:
            if job["state"] in ("queued", "running"):
                print(
                    f"dibs: job {job['id']} still {job['state']} after {format_duration(args.timeout)}. "
                    f"Wait again: dibs wait {job['id']}",
                    file=sys.stderr,
                )
        return 124
    if not args.json:
        for job in response["jobs"]:
            print(render.finished_line(job))
    if not _discarded(sys.stdout):  # it read how they ended: no need to tell it again (#176)
        with contextlib.suppress(dc.DibsError, OSError):
            dc.Client(paths).call("seen", session=dc.current_session()[0], ids=args.ids)
    return next((render.exit_code(j) for j in response["jobs"] if render.exit_code(j)), 0)


def cmd_logs(args, paths: Paths, _) -> int:
    client = dc.Client(paths)
    job = client.call("show", id=args.id)["job"]
    path, sink = job["output"], sys.stdout.buffer
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        data = b""
        if not args.follow:
            print(f"dibs: job {args.id} hasn't written any output yet", file=sys.stderr)
    shown = data
    if args.lines is not None:
        shown = b"".join(data.splitlines(keepends=True)[-args.lines :]) if args.lines > 0 else b""
    sink.write(shown)
    sink.flush()
    if args.follow and job["state"] in ("queued", "running"):
        try:
            _follow(client, args.id, path, len(data), sink)
        except (KeyboardInterrupt, _Interrupted):
            return 130
    return 0


def cmd_cancel(args, paths: Paths, _) -> int:
    response = dc.Client(paths).call("cancel", ids=args.ids)
    if args.json:
        _print_json({k: v for k, v in response.items() if k != "ok"})
    else:
        for job_id in response["cancelled"]:
            print(f"Cancelled job {job_id}.")
    for error in response["errors"]:
        print(f"dibs: {error}", file=sys.stderr)
    return 1 if response["errors"] else 0


def cmd_info(args, paths: Paths, _) -> int:
    info = dc.Client(paths).call("info")
    if args.json:
        _print_json({k: v for k, v in info.items() if k != "ok"})
    else:
        print(render.render_info(info))
    return 0


# dibs top's keys (#167), as htop takes them: the arrows and j/k scroll a line, PgUp/PgDn and space a
# page, Home/End and g/G to either end, the mouse wheel three lines; q or Ctrl-C quits.
_TOP_SEQUENCES = {b"\x1b[A": "up", b"\x1bOA": "up", b"\x1b[B": "down", b"\x1bOB": "down",
                  b"\x1b[5~": "page_up", b"\x1b[6~": "page_down", b"\x1b[H": "home", b"\x1bOH": "home",
                  b"\x1b[1~": "home", b"\x1b[7~": "home", b"\x1b[F": "end", b"\x1bOF": "end", b"\x1b[4~": "end",
                  b"\x1b[8~": "end"}  # fmt: skip
_TOP_CHARS = {b"q": "quit", b"Q": "quit", b"\x03": "quit", b"k": "up", b"j": "down", b" ": "page_down",
              b"g": "home", b"G": "end"}  # fmt: skip
_SGR_MOUSE = re.compile(rb"\x1b\[<(\d+);\d+;\d+([Mm])")
_SEQUENCE = re.compile(rb"\x1b(?:\[[0-9;<?]*[ -/]*[@-~]|O[@-~])")
_PARTIAL = re.compile(rb"\x1b(?:\[[0-9;<?]*|O)?\Z")
_TOP_STEPS = {"up": -1, "down": 1, "wheel_up": -3, "wheel_down": 3}


def _top_keys(data: bytes) -> tuple[list[str], bytes]:
    """What the bytes a terminal sent ask dibs top to do, and an escape sequence cut off at their end,
    which waits for the next read to finish it. Other keys, and mouse clicks, do nothing."""
    actions: list[str] = []
    i = 0
    while i < len(data):
        rest = data[i:]
        if rest[:1] != b"\x1b":
            if action := _TOP_CHARS.get(rest[:1]):
                actions.append(action)
            i += 1
        elif rest.startswith(b"\x1b[M"):  # the older mouse encoding: a button byte and two more after it
            if len(rest) < 6:
                return actions, rest
            if rest[3] - 32 in (64, 65):
                actions.append("wheel_up" if rest[3] - 32 == 64 else "wheel_down")
            i += 6
        elif mouse := _SGR_MOUSE.match(rest):
            if int(mouse.group(1)) in (64, 65) and mouse.group(2) == b"M":
                actions.append("wheel_up" if mouse.group(1) == b"64" else "wheel_down")
            i += mouse.end()
        elif sequence := _SEQUENCE.match(rest):
            if action := _TOP_SEQUENCES.get(sequence.group(0)):
                actions.append(action)
            i += sequence.end()
        elif _PARTIAL.match(rest) and len(rest) < 32:
            return actions, rest
        else:  # a lone Escape
            i += 1
    return actions, b""


def _scrolled(offset: int, page: int, action: str) -> int:
    """Where the panels start after a key; top_screen keeps it within what there is to scroll."""
    if action == "home":
        return 0
    if action == "end":
        return 1 << 30
    step = {"page_up": -max(1, page - 1), "page_down": max(1, page - 1)}.get(action, _TOP_STEPS.get(action, 0))
    return max(0, offset + step)


def cmd_top(args, paths: Paths, _) -> int:
    client = dc.Client(paths)
    home = Path.home()
    if not sys.stdout.isatty():  # everything at once, for a pipe or a file
        size = shutil.get_terminal_size((100, 40))
        print(render.render_top(client.call("queue"), width=size.columns, height=1 << 30, color=False,
                                names=agents.session_names(home)))  # fmt: skip
        return 0
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd) if sys.stdin.isatty() else None
    if saved:
        tty.setcbreak(fd)
    # Alternate screen, no cursor, and mouse reporting so the wheel scrolls, in tmux too (#167).
    sys.stdout.write("\033[?1049h\033[?25l\033[?1000h\033[?1006h")
    offset, page, pending = 0, 1, b""
    try:
        while True:
            size = shutil.get_terminal_size()
            frame, offset, page = render.top_screen(client.call("queue"), width=size.columns, height=size.lines,
                                                    color=True, offset=offset, names=agents.session_names(home))  # fmt: skip
            sys.stdout.write("\033[H" + frame.replace("\n", "\033[K\n") + "\033[K\033[J")
            sys.stdout.flush()
            ready, _, _ = select.select([fd] if saved else [], [], [], 1.0)
            if not ready:
                continue
            data = os.read(fd, 1024)
            if not data:  # the terminal hung up
                return 0
            actions, pending = _top_keys(pending + data)
            for action in actions:
                if action == "quit":
                    return 0
                offset = _scrolled(offset, page, action)
    except (KeyboardInterrupt, _Interrupted):
        return 0
    finally:
        if saved:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        sys.stdout.write("\033[?1006l\033[?1000l\033[?25h\033[?1049l")
        sys.stdout.flush()


def cmd_setup(args, paths: Paths, _) -> int:
    from dibs import agents

    home = Path.home()
    if args.undo:
        print("\n".join(agents.undo(home, service=not args.no_service)))
        return 0
    from dibs.machine import ConfigError, load_config

    try:
        agent_files = load_config(paths.config_file).agent_files
    except ConfigError as e:
        print(f"dibs: {e}", file=sys.stderr)
        return 1
    exe = shutil.which("dibs") or f"{sys.executable} -m dibs"
    lines = agents.setup(
        home,
        path_env=os.environ.get("PATH", ""),
        dry_run=args.dry_run,
        service=not args.no_service,
        dibs_exe=exe,
        agent_files=agent_files,
    )
    print("\n".join(lines))
    if not args.dry_run:
        if (home / ".config" / "systemd" / "user" / "dibs.service").exists():
            running = (
                "The guard is on: GPU processes started outside dibs get stopped, and so does the largest "
                "program outside dibs when memory runs short"
            )
        else:
            running = (
                "No service runs dibs here, so it starts on the first dibs command, and its guard runs "
                "while it does: GPU processes started outside dibs then get stopped, and so does the "
                "largest program outside dibs when memory runs short"
            )
        print(
            f"\n{running}, with a note on how to run them as jobs.\nTo only warn instead, put "
            f'guard = "warn" in {paths.config_file} and run dibs daemon restart. Undo all of this '
            "with dibs setup --undo."
        )
    return 0


def cmd_agent_help(args, paths: Paths, _) -> int:
    from dibs.instructions import AGENT_HELP

    print(AGENT_HELP.rstrip())
    return 0


def _start(paths: Paths) -> int:
    running = dc.ping(paths)
    if running:
        print(f"dibs is already running (pid {running['pid']}).")
        return 0
    dc.start_daemon(paths)
    deadline = time.monotonic() + dc.START_TIMEOUT
    while time.monotonic() < deadline:
        running = dc.ping(paths)
        if running:
            print(f"dibs started (pid {running['pid']}).")
            return 0
        time.sleep(0.05)
    print(f"dibs: the daemon didn't start; see {paths.daemon_log}", file=sys.stderr)
    return 1


def cmd_daemon(args, paths: Paths, _) -> int:
    if args.action == "run":
        from dibs.daemon import run_daemon

        return run_daemon()
    if args.action == "status":
        running = dc.ping(paths)
        if not running:
            print("dibs isn't running.")
            return 3
        print(f"dibs is running (pid {running['pid']}, build {running['build']}).")
        return 0
    if args.action in ("stop", "restart"):
        if dc.ping(paths):
            dc.wait_for_memory(what="stops" if args.action == "stop" else "restarts")
        if dc.stop_daemon(paths):
            print("dibs stopped. Jobs that were running keep running.")
        elif args.action == "stop":
            print("dibs wasn't running.")
    if args.action in ("start", "restart"):
        return _start(paths)
    return 0


# Commands that do nothing twice when run again, so they run again on the new code after an upgrade.
REPEATABLE = {"wait", "queue", "q", "show", "status", "info", "top"}


def _run_again(argv: list[str]) -> None:
    """dibs was upgraded while this command ran: run it again, started the same way, on the new code."""
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, [sys.executable, sys.argv[0], *argv], {**os.environ, "DIBS_RERUN": "1"})


HANDLERS = {
    "submit": cmd_submit,
    "run": cmd_run,
    "queue": cmd_queue,
    "q": cmd_queue,
    "momentum": cmd_momentum,
    "show": cmd_show,
    "status": cmd_show,
    "wait": cmd_wait,
    "logs": cmd_logs,
    "cancel": cmd_cancel,
    "info": cmd_info,
    "top": cmd_top,
    "setup": cmd_setup,
    "agent-help": cmd_agent_help,
    "daemon": cmd_daemon,
}


TELLS = {"submit", "run", "queue", "q", "momentum", "show", "status", "wait", "logs", "cancel", "info"}  # not top
AFTER = {"submit", "run"}  # these tell after the job's own lines: loops read its number from the first (#169)


def _discarded(stream) -> bool:
    """Whether what's written to this stream goes nowhere (/dev/null), so nobody reads it."""
    try:
        here, null = os.fstat(stream.fileno()), os.stat(os.devnull)
    except (OSError, ValueError, AttributeError):
        return False
    return stat.S_ISCHR(here.st_mode) and here.st_rdev == null.st_rdev


def _tell(paths, seen: list[int] | None = None, failures: bool = True) -> None:
    """Tell this session what the guard stopped of its since it was last told: the note in a stopped
    program's own output can be lost, in a pipe through grep or a log nobody reads (#135). And its jobs
    that failed in the last hour, at each command until one looks at them (seen) (#176). failures=False
    for JSON output, which a program reads."""
    try:
        notices = dc.Client(paths).call("notices", session=dc.current_session()[0], seen=seen or []).get("notices", [])
    except (dc.DibsError, OSError):
        return  # the command itself will say what's wrong
    for notice in notices:
        if notice.get("kind") == "failed":
            if failures:
                print(f"dibs: {render.failed_notice(notice)}", file=sys.stderr)
            continue
        when = time.strftime("%H:%M", time.localtime(notice["time"]))
        what = ("a program in your terminal kept many cores busy outside dibs" if notice.get("kind") == "heavy"
                else "the guard stopped a program of yours")  # fmt: skip
        print(f"dibs: at {when}, {what}:\n{notice['message']}\n", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # `dibs queue | head` ends quietly
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _raise_interrupted)
    argv = list(sys.argv[1:] if argv is None else argv)
    given = list(argv)
    parser = build_parser()
    command: list[str] = []
    split = argv.index("--") if "--" in argv else len(argv)
    # Only dibs's own options count: a -h in the job's command (df -h) belongs to the job.
    if argv and argv[0] in ("submit", "run") and not {"-h", "--help"} & set(argv[:split]):
        if split == len(argv):
            print(f"dibs: {COMMAND_HINT}", file=sys.stderr)
            return 2
        argv, command = argv[:split], argv[split + 1 :]
        if not command:
            print(f"dibs: nothing to run: {COMMAND_HINT}", file=sys.stderr)
            return 2
    args = parser.parse_args(argv)
    if not args.cmd:
        parser.print_help(sys.stderr)
        return 2
    try:
        if args.cmd in TELLS and args.cmd not in AFTER:
            looks = [args.id] if args.cmd in ("show", "logs") and not _discarded(sys.stdout) else []
            _tell(get_paths(), seen=looks, failures=not getattr(args, "json", False))
        return HANDLERS[args.cmd](args, get_paths(), command)
    except dc.DibsError as e:
        if e.code == "upgraded" and args.cmd in REPEATABLE and not os.environ.get("DIBS_RERUN"):
            _run_again(given)  # once: a second upgrade under the same command shows the error
        print(f"dibs: {e}", file=sys.stderr)
        return 2 if e.code == "rejected" else 1
    except KeyboardInterrupt:
        return 130
    except _Interrupted as e:
        return 128 + e.signum
