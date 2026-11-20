"""A copy of a job's code as it is at submit, so its agent can keep editing while the job waits (#172).

`dibs submit --snapshot` copies the git repository around the job's folder: the files git tracks, with
their uncommitted changes, and new files git doesn't ignore. What git ignores (environments such as
.venv, data, outputs, caches) isn't copied: the copy links to the real paths, so the job reads its
data and writes its results where they always go. The job runs in the matching folder of the copy.
When it ends, the copy goes, unless the job wrote files into it: then it's kept, and said where.
"""

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from dibs.paths import Paths
from dibs.units import GiB, format_size

SNAPSHOT_MAX = GiB  # bytes: the most code a copy may hold
DISK_KEEP = 2 * GiB  # bytes: a copy that would leave the disk with less free than this is refused
MANIFEST = "snapshot.json"


class SnapshotError(Exception):
    pass


def _git(cwd: str, *args: str) -> bytes:
    return subprocess.run(["git", "-C", cwd, *args], capture_output=True, check=True, timeout=60).stdout


def take(cwd: str, into: Path) -> dict:
    """Copy the repository around `cwd` into `into` (which mustn't exist): {"root": the repository,
    "copy": into, "files": {path: [size, mtime_ns]}} for each file copied."""
    try:
        root = _git(cwd, "rev-parse", "--show-toplevel").decode().strip()
    except (OSError, subprocess.SubprocessError):
        raise SnapshotError("--snapshot needs the job's folder to be in a git repository") from None
    listed = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard").split(b"\0")
    files = sorted({p.decode() for p in listed if p})
    files = [f for f in files if os.path.lexists(os.path.join(root, f))]  # tracked, but deleted now
    size = sum(os.lstat(os.path.join(root, f)).st_size for f in files)
    if size > SNAPSHOT_MAX:
        raise SnapshotError(
            f"--snapshot would copy {format_size(size)} of code from {root}, more than {format_size(SNAPSHOT_MAX)}: "
            "have git ignore what isn't code (data, outputs), or submit from a git worktree instead"
        )
    into.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(into.parent).free
    if free - size < DISK_KEEP:
        raise SnapshotError(f"--snapshot would leave the disk with less than {format_size(DISK_KEEP)} free")
    ignored = _git(root, "ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--directory").split(b"\0")
    into.mkdir()
    copied = {}
    for rel in files:
        source, target = os.path.join(root, rel), into / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if os.path.islink(source) or os.path.isdir(source):  # a link as it is; a submodule linked
            os.symlink(os.readlink(source) if os.path.islink(source) else source, target)
            continue
        shutil.copy2(source, target)
        stat = os.lstat(target)
        copied[rel] = [stat.st_size, stat.st_mtime_ns]
    for rel in [p.decode().rstrip("/") for p in ignored if p] + [".git"]:
        target = into / rel
        if not os.path.lexists(target) and os.path.lexists(os.path.join(root, rel)):
            target.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(os.path.join(root, rel), target)  # the real one: environments, data, outputs, history
    return {"root": root, "copy": str(into), "files": copied}


def stage(paths: Paths, cwd: str) -> str:
    """Take a snapshot for a job not yet submitted, in a folder of its own that the daemon moves into
    the job's folder at submit. Returns that folder."""
    folder = paths.state_dir / "snapshots" / uuid.uuid4().hex
    try:
        taken = take(cwd, folder / "tree")
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    (folder / MANIFEST).write_text(json.dumps(taken))
    return str(folder)


def run_paths(root: str, copy: str, cwd: str, argv: list[str]) -> tuple[str, list[str]]:
    """The job's folder and command with paths into the repository pointing into its copy."""

    def moved(path: str) -> str:
        return copy + path[len(root) :] if path == root or path.startswith(root + os.sep) else path

    return moved(cwd), [moved(a) for a in argv]


def clear(folder: Path) -> str | None:
    """When the job ends: delete its copy, unless the job wrote files into it (outside the linked
    paths); then keep it and return where it is."""
    try:
        copied = json.loads((folder / MANIFEST).read_text())["files"]
    except (OSError, ValueError, KeyError):
        shutil.rmtree(folder, ignore_errors=True)
        return None
    copy = folder / "tree"
    written = []
    for path, dirs, files in os.walk(copy):  # os.walk doesn't follow the links into the real paths
        dirs[:] = [d for d in dirs if d != "__pycache__"]  # Python's caches of what it imported
        for name in files:
            target = Path(path) / name
            if target.is_symlink() or name.endswith((".pyc", ".pyo")):
                continue
            rel = str(target.relative_to(copy))
            stat = target.lstat()
            if copied.get(rel) != [stat.st_size, stat.st_mtime_ns]:
                written.append(rel)
    if written:  # what git would ignore isn't kept for: caches, build output
        try:
            ignored = _git_input(str(copy), "\0".join(written), "check-ignore", "-z", "--stdin").split(b"\0")
        except (OSError, subprocess.SubprocessError):
            ignored = []
        written = [w for w in written if w.encode() not in ignored]
    if written:
        return str(copy)
    shutil.rmtree(folder, ignore_errors=True)
    return None


def _git_input(cwd: str, text: str, *args: str) -> bytes:
    """git with `text` on its input; exit 1 (nothing matched) is an answer, not a failure."""
    done = subprocess.run(["git", "-C", cwd, *args], input=text.encode(), capture_output=True, timeout=60)
    if done.returncode not in (0, 1):
        raise subprocess.CalledProcessError(done.returncode, args)
    return done.stdout
