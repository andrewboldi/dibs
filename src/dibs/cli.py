"""The dibs command."""

import argparse
import json
import os
import select
import shutil
import signal
import sys
import termios
import time
import tty

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
        s.add_argument("-n", "--name", help="a label for the job (default: the script or program name)")
        s.add_argument("-o", "--output", help="write the job's output to this file instead of dibs's log")
        if name == "submit":
            s.add_argument("--json", action="store_true", help="print the job as JSON")

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
        "output": os.path.abspath(args.output) if args.output else None,
        "request": {"cpus": args.cpus, "mem": args.mem, "vram": args.vram, "gpus": args.gpus, "time": args.time},
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
    if args.json:
        _print_json(response["job"])
    else:
        print(render.render_submit(response["job"], response["now"]))
    return 0


def cmd_run(args, paths: Paths, command: list[str]) -> int:
    client = dc.Client(paths)
    response = client.call("submit", **_submit_request(args, command))
    job = response["job"]
    print(render.render_submit(job, response["now"]), file=sys.stderr)
    try:
        finished = _follow(client, job["id"], job["output"], 0, sys.stdout.buffer)
    except KeyboardInterrupt:
        return _keeps_running(job["id"], signal.SIGINT)
    except _Interrupted as e:
        return _keeps_running(job["id"], e.signum)
    if finished["state"] != "done":
        print(f"dibs: {render.finished_line(finished)}", file=sys.stderr)
    return render.exit_code(finished)


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


def cmd_top(args, paths: Paths, _) -> int:
    client = dc.Client(paths)
    if not sys.stdout.isatty():
        size = shutil.get_terminal_size((100, 40))
        print(render.render_top(client.call("queue"), width=size.columns, height=size.lines, color=False))
        return 0
    saved = termios.tcgetattr(sys.stdin) if sys.stdin.isatty() else None
    if saved:
        tty.setcbreak(sys.stdin)
    sys.stdout.write("\033[?1049h\033[?25l")  # alternate screen, hide the cursor
    try:
        while True:
            size = shutil.get_terminal_size()
            frame = render.render_top(client.call("queue"), width=size.columns, height=size.lines, color=True)
            sys.stdout.write("\033[H" + frame.replace("\n", "\033[K\n") + "\033[K\033[J")
            sys.stdout.flush()
            ready, _, _ = select.select([sys.stdin] if saved else [], [], [], 1.0)
            if ready and sys.stdin.read(1) in ("q", "Q", "\x03"):
                return 0
    except (KeyboardInterrupt, _Interrupted):
        return 0
    finally:
        if saved:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, saved)
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()


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
        if dc.stop_daemon(paths):
            print("dibs stopped. Jobs that were running keep running.")
        elif args.action == "stop":
            print("dibs wasn't running.")
    if args.action in ("start", "restart"):
        return _start(paths)
    return 0


HANDLERS = {
    "submit": cmd_submit,
    "run": cmd_run,
    "queue": cmd_queue,
    "q": cmd_queue,
    "show": cmd_show,
    "status": cmd_show,
    "wait": cmd_wait,
    "logs": cmd_logs,
    "cancel": cmd_cancel,
    "info": cmd_info,
    "top": cmd_top,
    "daemon": cmd_daemon,
}


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # `dibs queue | head` ends quietly
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _raise_interrupted)
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    command: list[str] = []
    if argv and argv[0] in ("submit", "run") and not {"-h", "--help"} & set(argv):
        if "--" not in argv:
            print(f"dibs: {COMMAND_HINT}", file=sys.stderr)
            return 2
        split = argv.index("--")
        argv, command = argv[:split], argv[split + 1 :]
        if not command:
            print(f"dibs: nothing to run: {COMMAND_HINT}", file=sys.stderr)
            return 2
    args = parser.parse_args(argv)
    if not args.cmd:
        parser.print_help(sys.stderr)
        return 2
    try:
        return HANDLERS[args.cmd](args, get_paths(), command)
    except dc.DibsError as e:
        print(f"dibs: {e}", file=sys.stderr)
        return 2 if e.code == "rejected" else 1
    except KeyboardInterrupt:
        return 130
    except _Interrupted as e:
        return 128 + e.signum
