"""install.sh's choice of what to install, run through bash with a stand-in curl."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRANCH = "dibs @ https://github.com/andrewboldi/dibs/archive/refs/heads/main.tar.gz"
SHA = "0123456789abcdef0123456789abcdef01234567"


def source_spec(tmp_path, body="", code=0, **env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    curl = bin_dir / "curl"
    curl.write_text(f"#!/bin/sh\nprintf '%s' '{body}'\nexit {code}\n")
    curl.chmod(0o755)
    run_env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), "DIBS_INSTALL_LIB": "1", **env}
    out = subprocess.run(
        ["bash", "-c", f'source "{ROOT}/install.sh" && source_spec'], env=run_env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_installs_the_commit_main_points_to(tmp_path):
    # Issue #50: GitHub serves branch archives from a cache for a few minutes, so right after a push
    # the branch archive was still the previous version. A commit's archive never goes stale.
    assert source_spec(tmp_path, SHA) == f"dibs @ https://github.com/andrewboldi/dibs/archive/{SHA}.tar.gz"


def test_falls_back_to_the_branch_archive_when_the_lookup_fails(tmp_path):
    assert source_spec(tmp_path, '{"message": "API rate limit exceeded"}') == BRANCH
    assert source_spec(tmp_path, "", code=22) == BRANCH


def test_dibs_source_still_wins(tmp_path):
    assert source_spec(tmp_path, SHA, DIBS_SOURCE="https://example.invalid/x.tar.gz") == "dibs @ https://example.invalid/x.tar.gz"
