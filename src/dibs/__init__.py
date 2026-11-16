"""dibs: a job queue for one machine, so coding agents can share its GPUs, CPUs and RAM."""

import hashlib
import os
from pathlib import Path

__version__ = "0.1.0"

_build: str | None = None


def _code_on_disk() -> str:
    digest = hashlib.sha1()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    return f"{__version__}+{digest.hexdigest()[:10]}"


def build_id() -> str:
    """The version plus a hash of this package's code: a CLI and daemon from different code differ.
    Taken once, so a command still running after an upgrade names the code it started with."""
    override = os.environ.get("DIBS_BUILD_OVERRIDE")  # tests only
    if override:
        return override
    global _build
    if _build is None:
        _build = _code_on_disk()
    return _build


def installed_build() -> str:
    """The build of the code on disk now, where this package was loaded from: after dibs is upgraded
    in place, no longer the build_id() of a command that was already running."""
    return os.environ.get("DIBS_BUILD_OVERRIDE") or _code_on_disk()
