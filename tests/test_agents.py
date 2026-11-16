import os
import subprocess
import sys

import pytest

from dibs import agents
from dibs.instructions import AGENT_HELP, BEGIN, BLOCK, END, SKILL

EXE = "/opt/dibs/bin/dibs"


def snapshot(root):
    return {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def setup(home, **kw):
    return agents.setup(home, path_env=kw.pop("path_env", ""), dry_run=kw.pop("dry_run", False),
                        service=False, dibs_exe=EXE)  # fmt: skip


@pytest.fixture
def home(tmp_path):
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".codex").mkdir()
    return tmp_path


def test_put_block_appends_with_markers():
    assert agents.put_block("hello\n", "B") == f"hello\n\n{BEGIN}\nB\n{END}\n"
    assert agents.put_block("", "B") == f"{BEGIN}\nB\n{END}\n"


def test_put_block_replaces_in_place():
    text = f"top\n\n{BEGIN}\nold\n{END}\nbottom\n"
    assert agents.put_block(text, "new") == f"top\n\n{BEGIN}\nnew\n{END}\nbottom\n"


@pytest.mark.parametrize("original", ["hello\n", "a\n\nb\n", "", "x\n\n"])
def test_remove_block_restores_original(original):
    assert agents.remove_block(agents.put_block(original, BLOCK)) == original


def test_remove_block_without_trailing_newline_gains_one():
    # Byte-exact undo of this case comes from the setup record instead (test_undo_restores_exactly).
    assert agents.remove_block(agents.put_block("hello", BLOCK)) == "hello\n"


def test_setup_only_detected_agents(home):
    lines = setup(home)
    assert BEGIN in (home / ".claude" / "CLAUDE.md").read_text()
    assert BEGIN in (home / ".codex" / "AGENTS.md").read_text()
    assert not (home / ".gemini").exists()
    assert not (home / ".config" / "opencode").exists()
    assert not (home / ".pi").exists()
    assert any("Claude Code" in line for line in lines) and any("Codex" in line for line in lines)


def test_setup_detects_agents_by_binary(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gemini = bin_dir / "gemini"
    gemini.write_text("#!/bin/sh\n")
    gemini.chmod(0o755)
    setup(tmp_path, path_env=str(bin_dir))
    assert BEGIN in (tmp_path / ".gemini" / "GEMINI.md").read_text()


def test_setup_twice_no_duplicates(home):
    setup(home)
    first = snapshot(home)
    lines = setup(home)
    assert (home / ".claude" / "CLAUDE.md").read_text().count(BEGIN) == 1
    assert {p: b for p, b in snapshot(home).items() if p.name != "setup.json"} == {
        p: b for p, b in first.items() if p.name != "setup.json"
    }
    assert any("already" in line for line in lines)


def test_setup_keeps_existing_content(home):
    (home / ".claude" / "CLAUDE.md").write_text("# my rules\nbe nice\n")
    setup(home)
    text = (home / ".claude" / "CLAUDE.md").read_text()
    assert text.startswith("# my rules\nbe nice\n") and text.count(BEGIN) == 1


def test_setup_writes_skill(home):
    setup(home)
    skill = (home / ".claude" / "skills" / "dibs" / "SKILL.md").read_text()
    assert skill.startswith("---\nname: dibs\n") and skill == SKILL


def test_undo_restores_exactly(home):
    (home / ".claude" / "CLAUDE.md").write_text("# my rules")  # no trailing newline
    before = snapshot(home)
    setup(home)
    agents.undo(home, service=False)
    assert snapshot(home) == before
    assert not (home / ".claude" / "skills").exists()


def test_undo_keeps_later_edits(home):
    (home / ".claude" / "CLAUDE.md").write_text("# my rules\n")
    setup(home)
    with open(home / ".claude" / "CLAUDE.md", "a") as f:
        f.write("added later\n")
    agents.undo(home, service=False)
    text = (home / ".claude" / "CLAUDE.md").read_text()
    assert "# my rules" in text and "added later" in text and BEGIN not in text


def test_dry_run_changes_nothing(home):
    before = snapshot(home)
    lines = setup(home, dry_run=True)
    assert snapshot(home) == before
    assert any("would" in line for line in lines)


def test_service_unit_text():
    unit = agents.service_unit(EXE)
    assert "Description=dibs job scheduler" in unit
    assert f"ExecStart={EXE} daemon run" in unit
    assert "Restart=always" in unit and "WantedBy=default.target" in unit


def test_block_warns_about_guard():
    assert "stopped" in BLOCK and "dibs submit" in BLOCK and "dibs agent-help" in BLOCK


def test_agent_help_covers_the_essentials():
    for phrase in ("dibs submit", "--vram", "--time", "dibs wait", "dibs queue", "dibs cancel",
                   "run_in_background", "CUDA context", "stopped"):  # fmt: skip
        assert phrase in AGENT_HELP, phrase


def test_agent_help_cli(tmp_path):
    result = subprocess.run([sys.executable, "-m", "dibs", "agent-help"], capture_output=True, text=True,
                            env={**os.environ, "DIBS_HOME": str(tmp_path)}, timeout=60)  # fmt: skip
    assert result.returncode == 0 and result.stdout.strip() == AGENT_HELP.strip()


@pytest.mark.parametrize("existed", [True, False])
def test_rerun_after_edits_then_undo_keeps_the_edits(home, monkeypatch, existed):
    target = home / ".codex" / "AGENTS.md"
    if existed:
        target.write_text("original\n")
    setup(home)
    with open(target, "a") as f:
        f.write("MY NOTES\n")
    monkeypatch.setattr(agents, "BLOCK", BLOCK + "\nA newer dibs says more.")  # an upgrade
    setup(home)
    agents.undo(home, service=False)
    text = target.read_text()
    assert "MY NOTES" in text and BEGIN not in text
    if existed:
        assert text.startswith("original\n")


def test_failed_service_install_still_leaves_a_record(home, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("systemctl hung")

    monkeypatch.setattr(agents, "_install_service", broken)
    with pytest.raises(RuntimeError):
        agents.setup(home, path_env="", dry_run=False, service=True, dibs_exe=EXE)
    assert BEGIN in (home / ".claude" / "CLAUDE.md").read_text()
    agents.undo(home, service=False)
    assert not (home / ".claude" / "CLAUDE.md").exists()


def test_service_install_and_undo(tmp_path, monkeypatch):
    import json
    import pwd

    from dibs import launcher

    (tmp_path / ".claude").mkdir()
    bin_dir, log = tmp_path / "bin", tmp_path / "calls.log"
    bin_dir.mkdir()
    for tool in ("systemctl", "loginctl"):
        script = f'#!/bin/sh\necho "{tool} $*" >> {log}\n'
        if tool == "loginctl":
            script += 'if [ "$1" = show-user ]; then echo Linger=no; fi\n'
        (bin_dir / tool).write_text(script)
        (bin_dir / tool).chmod(0o755)
    # Every path under tmp, so stopping "the running daemon" can never reach a real one.
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.delenv("DIBS_HOME", raising=False)
    monkeypatch.setattr(launcher, "systemd_available", lambda env=None: True)
    user = pwd.getpwuid(os.getuid()).pw_name

    agents.setup(tmp_path, path_env="", dry_run=False, service=True, dibs_exe=EXE)
    unit = tmp_path / ".config" / "systemd" / "user" / "dibs.service"
    assert unit.read_text() == agents.service_unit(EXE)
    assert "Restart=always" in unit.read_text() and f"ExecStart={EXE} daemon run" in unit.read_text()
    calls = log.read_text().splitlines()
    assert calls.index("systemctl --user daemon-reload") < calls.index("systemctl --user enable --now dibs.service")
    assert f"loginctl enable-linger {user}" in calls
    record = json.loads((tmp_path / ".config" / "dibs" / "setup.json").read_text())
    assert record["service"] is True and record["linger_enabled"] is True

    agents.undo(tmp_path, service=True)
    calls = log.read_text().splitlines()
    assert "systemctl --user disable --now dibs.service" in calls
    assert f"loginctl disable-linger {user}" in calls
    assert not unit.exists() and not (tmp_path / ".claude" / "CLAUDE.md").exists()
