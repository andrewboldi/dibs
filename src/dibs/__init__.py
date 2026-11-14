"""dibs: a job queue for one machine, so coding agents can share its GPUs, CPUs and RAM."""

import hashlib
import os
from pathlib import Path

__version__ = "0.1.0"

_build: str | None = None


def build_id() -> str:
    """The version plus a hash of this package's code: a CLI and daemon from different code differ."""
    override = os.environ.get("DIBS_BUILD_OVERRIDE")  # tests only
    if override:
        return override
    global _build
    if _build is None:
        digest = hashlib.sha1()
        for source in sorted(Path(__file__).parent.glob("*.py")):
            digest.update(source.name.encode())
            digest.update(source.read_bytes())
        _build = f"{__version__}+{digest.hexdigest()[:10]}"
    return _build
