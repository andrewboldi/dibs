"""dibs setup: tell every installed agent CLI about dibs, and keep the daemon (and guard) running.

Each agent's global instructions file gets a short marked block. What setup changed is recorded,
so `dibs setup --undo` restores untouched files byte for byte and only removes the block from
files edited since.
"""

import json
import os
import pwd
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dibs import launcher
from dibs.instructions import BEGIN, BLOCK, END, SKILL

# (name, config folder under $HOME, instructions file in it, binary). Locations checked against
# the installed CLIs on 2026-11-14.
AGENTS = [
    ("Claude Code", ".claude", "CLAUDE.md", "claude"),
    ("Codex", ".codex", "AGENTS.md", "codex"),
    ("opencode", ".config/opencode", "AGENTS.md", "opencode"),
    ("pi", ".pi/agent", "AGENTS.md", "pi"),
    ("Gemini CLI", ".gemini", "GEMINI.md", "gemini"),
]


@dataclass(frozen=True)
class Target:
    agent: str
    path: Path
    detected: bool


def targets(home: Path, path_env: str) -> list[Target]:
    found = []
    for name, folder, filename, binary in AGENTS:
        detected = (home / folder).exists() or shutil.which(binary, path=path_env or os.defpath) is not None
        found.append(Target(name, home / folder / filename, detected))
    return found


def put_block(text: str, block: str) -> str:
    """Put the dibs block in text: replacing an earlier one in place, else appending it."""
    wrapped = f"{BEGIN}\n{block.rstrip()}\n{END}\n"
    start = text.find(BEGIN)
    if start != -1 and text.find(END, start) != -1:
        end = text.find(END, start) + len(END)
        if text[end : end + 1] == "\n":
            end += 1
        return text[:start] + wrapped + text[end:]
    if not text:
        return wrapped
    return text + ("\n" if text.endswith("\n") else "\n\n") + wrapped


def remove_block(text: str) -> str:
    """Take the dibs block out again, with the blank line put_block added before it."""
    start = text.find(BEGIN)
    if start == -1 or text.find(END, start) == -1:
        return text
    end = text.find(END, start) + len(END)
    if text[end : end + 1] == "\n":
        end += 1
    before, after = text[:start], text[end:]
    if not after and before.endswith("\n\n"):
        before = before[:-1]
    return before + after


def service_unit(dibs_exe: str) -> str:
    return (
        "[Unit]\n"
        "Description=dibs job scheduler\n"
        "\n"
        "[Service]\n"
        f"ExecStart={dibs_exe} daemon run\n"
        "Restart=on-failure\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def _record_path(home: Path) -> Path:
    return home / ".config" / "dibs" / "setup.json"


def _load_record(home: Path) -> dict:
    try:
        return json.loads(_record_path(home).read_text())
    except (OSError, ValueError):
        return {"created": [], "dirs": [], "modified": {}, "written": {}, "service": False, "linger_enabled": False}


def _save_record(home: Path, record: dict) -> None:
    path = _record_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))


def _write(path: Path, text: str, record: dict) -> None:
    missing = [p for p in [path.parent, *path.parent.parents] if not p.exists()]
    for folder in reversed(missing):
        folder.mkdir()
        record["dirs"].append(str(folder))
    key = str(path)
    if path.exists():
        if key not in record["modified"] and key not in record["created"]:
            record["modified"][key] = path.read_text()
    elif key not in record["created"]:
        record["created"].append(key)
    path.write_text(text)
    record["written"][key] = text


def _run(*command: str) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, timeout=60)


def _install_service(home: Path, dibs_exe: str, record: dict, lines: list[str]) -> None:
    if not launcher.systemd_available():
        lines.append(
            "No systemd user session here, so dibs can't start itself at login: "
            "run `dibs daemon start` from your shell's startup file instead."
        )
        return
    from dibs.client import stop_daemon
    from dibs.paths import get_paths

    stop_daemon(get_paths())  # a daemon started on demand hands over to the service; jobs keep running
    _write(home / ".config" / "systemd" / "user" / "dibs.service", service_unit(dibs_exe), record)
    _run("systemctl", "--user", "daemon-reload")
    enabled = _run("systemctl", "--user", "enable", "--now", "dibs.service")
    if enabled.returncode != 0:
        lines.append(f"Couldn't start the dibs service: {enabled.stderr.strip()}")
        return
    record["service"] = True
    lines.append("Service: dibs now starts at login and restarts if it fails (systemd user unit dibs.service).")
    user = pwd.getpwuid(os.getuid()).pw_name
    linger = _run("loginctl", "show-user", user, "--property=Linger").stdout.strip()
    if linger == "Linger=no" and _run("loginctl", "enable-linger", user).returncode == 0:
        record["linger_enabled"] = True
        lines.append("Lingering: on, so dibs keeps running for agents in tmux after you log out.")


def setup(home: Path, *, path_env: str, dry_run: bool, service: bool, dibs_exe: str) -> list[str]:
    record = _load_record(home)
    lines = []
    found = [t for t in targets(home, path_env) if t.detected]
    if not found:
        lines.append("No agent CLIs found (Claude Code, Codex, opencode, pi, Gemini CLI).")
    for target in found:
        old = target.path.read_text() if target.path.exists() else ""
        new = put_block(old, BLOCK)
        if new == old:
            lines.append(f"{target.agent}: {target.path} already has the dibs instructions.")
            continue
        verb = "update" if BEGIN in old else "add"
        lines.append(f"{target.agent}: {'would ' + verb if dry_run else verb + 'ed'} the dibs instructions in {target.path}")
        if not dry_run:
            _write(target.path, new, record)
    if any(t.agent == "Claude Code" for t in found):
        skill = home / ".claude" / "skills" / "dibs" / "SKILL.md"
        if not skill.exists() or skill.read_text() != SKILL:
            lines.append(f"Claude Code: {'would write' if dry_run else 'wrote'} the dibs skill to {skill}")
            if not dry_run:
                _write(skill, SKILL, record)
    if service:
        if dry_run:
            lines.append("Would install and start a systemd user service so dibs (and its guard) always runs.")
        else:
            _install_service(home, dibs_exe, record, lines)
    if not dry_run:
        _save_record(home, record)
    return lines


def undo(home: Path, *, service: bool) -> list[str]:
    record = _load_record(home)
    lines = []
    if service and record.get("service"):
        _run("systemctl", "--user", "disable", "--now", "dibs.service")
        lines.append("Service: stopped and disabled dibs.service.")
    if record.get("linger_enabled"):
        user = pwd.getpwuid(os.getuid()).pw_name
        _run("loginctl", "disable-linger", user)
        lines.append("Lingering: turned back off.")
    for key, original in record.get("modified", {}).items():
        path = Path(key)
        if not path.exists():
            continue
        current = path.read_text()
        path.write_text(original if current == record["written"].get(key) else remove_block(current))
        lines.append(f"Restored {path}")
    for key in record.get("created", []):
        path = Path(key)
        if not path.exists():
            continue
        current = path.read_text()
        if current == record["written"].get(key):
            path.unlink()
            lines.append(f"Removed {path}")
        else:
            path.write_text(remove_block(current))
            lines.append(f"Removed the dibs instructions from {path}, keeping your later edits")
    for key in reversed(record.get("dirs", [])):
        try:
            Path(key).rmdir()
        except OSError:
            pass  # not empty: something else lives there now
    if service:
        _run("systemctl", "--user", "daemon-reload")
    _record_path(home).unlink(missing_ok=True)
    if not lines:
        lines.append("Nothing to undo.")
    return lines
