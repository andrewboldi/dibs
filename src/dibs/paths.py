"""Where dibs keeps its config, its state and its socket.

By default it follows the XDG base directories. With DIBS_HOME set, everything lives under that
folder instead, which is how tests and the demo run isolated from a real dibs.
"""

import hashlib
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# AF_UNIX socket paths are limited to 108 bytes on Linux (104 on macOS); stay well under.
_SOCKET_PATH_MAX = 100


@dataclass(frozen=True)
class Paths:
    home_mode: bool  # True when DIBS_HOME is set
    config_file: Path
    state_dir: Path
    db: Path
    jobs_dir: Path
    daemon_log: Path
    lock: Path
    socket: Path
    instance: str  # short id telling this dibs apart from others on the machine

    def job_dir(self, job_id: int) -> Path:
        return self.jobs_dir / str(job_id)


def get_paths(env: Mapping[str, str] = os.environ) -> Paths:
    if env.get("DIBS_HOME"):
        home = Path(env["DIBS_HOME"])
        config_file, state_dir, socket = home / "config.toml", home / "state", home / "dibs.sock"
    else:
        user_home = Path(env.get("HOME") or Path.home())
        config_home = Path(env.get("XDG_CONFIG_HOME") or user_home / ".config")
        state_home = Path(env.get("XDG_STATE_HOME") or user_home / ".local" / "state")
        config_file, state_dir = config_home / "dibs" / "config.toml", state_home / "dibs"
        runtime = env.get("XDG_RUNTIME_DIR")
        socket = Path(runtime) / "dibs" / "dibs.sock" if runtime else state_dir / "dibs.sock"
    instance = hashlib.sha1(str(state_dir.resolve()).encode()).hexdigest()[:8]
    if len(str(socket)) > _SOCKET_PATH_MAX:
        socket = Path("/tmp") / f"dibs-{os.getuid()}" / f"{instance}.sock"
    return Paths(
        home_mode=bool(env.get("DIBS_HOME")),
        config_file=config_file,
        state_dir=state_dir,
        db=state_dir / "dibs.db",
        jobs_dir=state_dir / "jobs",
        daemon_log=state_dir / "daemon.log",
        lock=state_dir / "daemon.lock",
        socket=socket,
        instance=instance,
    )


def _private_dir(path: Path) -> None:
    """Create path (and parents) if needed; make sure it's ours and only we can enter it."""
    path.mkdir(parents=True, exist_ok=True)
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
        raise RuntimeError(f"{path} must be a directory owned by you")
    if stat.S_IMODE(st.st_mode) != 0o700:
        os.chmod(path, 0o700)


def ensure_dirs(paths: Paths) -> None:
    for d in (paths.state_dir, paths.jobs_dir, paths.socket.parent):
        _private_dir(d)
