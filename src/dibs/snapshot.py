"""A copy of a job's code as it is at submit, so its agent can keep editing while the job waits (#172).

`dibs submit --snapshot` copies the git repository around the job's folder: the files git tracks, with
their uncommitted changes, and new files git doesn't ignore. What git ignores (environments such as
.venv, data, outputs, caches) isn't copied: the copy links to the real paths, so the job reads its
data and writes its results where they always go. The job runs in the matching folder of the copy.
When it ends, the copy goes, unless the job wrote files into it: then it's kept, and said where.
"""

import json
import os
import re
import shutil
import stat
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


def _newest_copy(jobs_dir: Path | None, root: str) -> tuple[Path | None, dict]:
    """The newest live copy of this repository among the jobs' folders, and the files it copied (#233)."""
    best: tuple[float, Path, dict] | None = None
    for manifest in jobs_dir.glob(f"*/snapshot/{MANIFEST}") if jobs_dir else []:
        try:
            taken = json.loads(manifest.read_text())
            when = manifest.stat().st_mtime
        except (OSError, ValueError):
            continue
        if taken.get("root") == root and isinstance(taken.get("files"), dict) and (best is None or when > best[0]):
            best = (when, manifest.parent / "tree", taken["files"])
    return (best[1], best[2]) if best else (None, {})


def _shared(source_stat: os.stat_result, rel: str, tree: Path | None, files: dict) -> os.stat_result | None:
    """The stat of the old copy's file to share for `rel`, when the repository's file, the old copy's
    record of it and the old copy's file as it stands all agree on size and modification time: a
    copy its job wrote into, or a file edited since, is copied afresh (#233)."""
    if tree is None or files.get(rel) != [source_stat.st_size, source_stat.st_mtime_ns]:
        return None
    try:
        old = os.lstat(tree / rel)
    except OSError:
        return None
    same = stat.S_ISREG(old.st_mode) and (old.st_size, old.st_mtime_ns) == (source_stat.st_size, source_stat.st_mtime_ns)
    return old if same else None


def take(cwd: str, into: Path, jobs_dir: Path | None = None) -> dict:
    """Copy the repository around `cwd` into `into` (which mustn't exist): {"root": the repository,
    "copy": into, "files": {path: [size, mtime_ns]}} for each file copied."""
    try:
        root = _git(cwd, "rev-parse", "--show-toplevel").decode().strip()
    except (OSError, subprocess.SubprocessError):
        raise SnapshotError("--snapshot needs the job's folder to be in a git repository") from None
    listed = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard").split(b"\0")
    files = sorted({p.decode() for p in listed if p})
    files = [f for f in files if os.path.lexists(os.path.join(root, f))]  # tracked, but deleted now
    stats = {f: os.lstat(os.path.join(root, f)) for f in files}
    size = sum(st.st_size for st in stats.values())
    # Files unchanged since the newest copy of this repository are shared with it, not copied (#233).
    tree, old_files = _newest_copy(jobs_dir, root)
    shared = {f: old for f in files if stat.S_ISREG(stats[f].st_mode) and (old := _shared(stats[f], f, tree, old_files))}
    to_copy = sum(st.st_size for f, st in stats.items() if f not in shared)
    if size > SNAPSHOT_MAX:
        raise SnapshotError(
            f"--snapshot would copy {format_size(size)} of code from {root}, more than {format_size(SNAPSHOT_MAX)}: "
            "have git ignore what isn't code (data, outputs), or submit from a git worktree instead"
        )
    into.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(into.parent).free
    if free - to_copy < DISK_KEEP:
        raise SnapshotError(f"--snapshot would leave the disk with less than {format_size(DISK_KEEP)} free")
    ignored = _git(root, "ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--directory").split(b"\0")
    into.mkdir()
    copied, copied_bytes = {}, 0
    for rel in files:
        source, target = os.path.join(root, rel), into / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if os.path.islink(source) or os.path.isdir(source):  # a link as it is; a submodule linked
            os.symlink(os.readlink(source) if os.path.islink(source) else source, target)
            continue
        if rel in shared:  # the same file as the old copy's: one inode, read-only so neither job's changes
            try:
                os.link(tree / rel, target)
                os.chmod(target, stat.S_IMODE(shared[rel].st_mode) & ~0o222)
            except OSError:  # the old copy may be going as its job ends: copy it instead
                shared.pop(rel)
                if os.path.lexists(target):
                    os.unlink(target)
        if rel not in shared:
            shutil.copy2(source, target)
            copied_bytes += os.lstat(target).st_size
        st = os.lstat(target)
        copied[rel] = [st.st_size, st.st_mtime_ns]
    for rel in [p.decode().rstrip("/") for p in ignored if p] + [".git"]:
        target = into / rel
        if not os.path.lexists(target) and os.path.lexists(os.path.join(root, rel)):
            target.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(os.path.join(root, rel), target)  # the real one: environments, data, outputs, history
    return {"root": root, "copy": str(into), "files": copied, "copied": copied_bytes}


def stage(paths: Paths, cwd: str) -> str:
    """Take a snapshot for a job not yet submitted, in a folder of its own that the daemon moves into
    the job's folder at submit. Returns that folder."""
    folder = paths.state_dir / "snapshots" / uuid.uuid4().hex
    try:
        taken = take(cwd, folder / "tree", paths.jobs_dir)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    (folder / MANIFEST).write_text(json.dumps(taken))
    return str(folder)


SHELLS = ("sh", "bash", "zsh", "dash")  # as model.script_of


def run_paths(root: str, copy: str, cwd: str, argv: list[str]) -> tuple[str, list[str]]:
    """The job's folder and command with paths into the repository pointing into its copy: arguments
    that are such paths, and such paths inside a shell's -c string (`bash -c "cd /repo/...; python
    /repo/src/x.py"`), where the repository's path stands as a path of its own."""

    def moved(path: str) -> str:
        return copy + path[len(root) :] if path == root or path.startswith(root + os.sep) else path

    inside = re.compile(re.escape(root) + r"(?=/|$|[\s'\";)&|<>])")
    out = []
    for i, arg in enumerate(argv):
        shell = i >= 2 and argv[i - 1] == "-c" and any(os.path.basename(w) in SHELLS for w in argv[: i - 1])
        out.append(inside.sub(lambda m: copy, arg) if shell else moved(arg))
    return moved(cwd), out


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
