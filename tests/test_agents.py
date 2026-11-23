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


def test_agent_help_explains_momentum():
    # Issue #94: rare, asked in plain words, recorded only after the user's yes.
    for phrase in ("all\nthree hold", "Never record it without the user's yes", "dibs momentum --for"):
        assert phrase in AGENT_HELP, phrase


def test_agent_help_explains_timing_runs():
    assert "--exclusive" in AGENT_HELP and "timing run" in AGENT_HELP


def test_agent_help_explains_priority_reordering():
    # Issues #90, #91
    assert "ahead of your own waiting jobs of lower priority" in AGENT_HELP


def test_agent_help_says_dibs_retries_resource_failures():
    # Issues #84, #116: retries are the agent's choice, for jobs safe to run again from the start.
    for phrase in ("--retry", "safe to run again from the start", "runs it again by itself"):
        assert phrase in AGENT_HELP, phrase


def test_agent_help_says_a_retry_runs_over_what_the_stopped_try_left():
    # Issue #177: an export that refuses to start when its output exists failed both of its retries.
    for phrase in ("over whatever a stopped try left", "refuses to start when its output already exists"):
        assert phrase in AGENT_HELP, phrase


def test_agent_help_says_files_in_a_folder_kept_in_ram_count_as_ram():
    # Issue #74: most RAM kills were scratch files in a /tmp kept in RAM.
    for phrase in ("/dev/shm", "findmnt /tmp", "TMPDIR", "until deleted", "dibs deletes when the job ends"):
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


@pytest.fixture
def fake_systemd(tmp_path, monkeypatch):
    """Fake systemctl and loginctl that record their calls; every dibs path under tmp."""
    import pwd

    from dibs import launcher

    (tmp_path / ".claude").mkdir(exist_ok=True)
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
    log.write_text("")
    return log, pwd.getpwuid(os.getuid()).pw_name


def test_service_install_and_undo(tmp_path, fake_systemd):
    import json

    log, user = fake_systemd
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


PANES = ("tmux-spawn-", "vte-spawn-")


def test_setup_keeps_a_terminal_pane_running_when_the_kernel_kills_a_program_in_it(tmp_path, fake_systemd):
    # Issue #134: the kernel's OOM killer killed a 14G program an agent ran in its tmux pane, and
    # systemd's default OOMPolicy=stop then stopped the whole pane, the agent with it.
    log, _ = fake_systemd
    agents.setup(tmp_path, path_env="", dry_run=False, service=True, dibs_exe=EXE)
    user_units = tmp_path / ".config" / "systemd" / "user"
    for prefix in PANES:
        assert "[Scope]\nOOMPolicy=continue\n" in (user_units / f"{prefix}.scope.d" / "50-dibs.conf").read_text()
    assert "systemctl --user daemon-reload" in log.read_text().splitlines()

    agents.undo(tmp_path, service=False)  # --no-service keeps them, with the service
    assert all((user_units / f"{prefix}.scope.d" / "50-dibs.conf").exists() for prefix in PANES)
    agents.undo(tmp_path, service=True)
    assert not any((user_units / f"{prefix}.scope.d").exists() for prefix in PANES)


def test_undo_without_service_leaves_the_service_alone(tmp_path, fake_systemd):
    # dibs setup --undo --no-service (issue #2): instructions go, the service and lingering stay,
    # and a later full undo still knows about them.
    log, user = fake_systemd
    agents.setup(tmp_path, path_env="", dry_run=False, service=True, dibs_exe=EXE)
    unit = tmp_path / ".config" / "systemd" / "user" / "dibs.service"
    log.write_text("")
    agents.undo(tmp_path, service=False)
    calls = log.read_text().splitlines()
    assert not any("disable" in call for call in calls)
    assert unit.exists() and not (tmp_path / ".claude" / "CLAUDE.md").exists()

    agents.undo(tmp_path, service=True)
    calls = log.read_text().splitlines()
    assert "systemctl --user disable --now dibs.service" in calls
    assert f"loginctl disable-linger {user}" in calls
    assert not unit.exists()


def test_setup_writes_into_configured_agent_files_and_undo_takes_it_back(home):
    # Issue #26: an agent CLI dibs doesn't know gets the instructions through config, without a PR.
    (home / "mine.md").write_text("# my agent's rules\n")
    before = snapshot(home)
    lines = agents.setup(home, path_env="", dry_run=False, service=False, dibs_exe=EXE,
                         agent_files=["~/mine.md", "~/.cursor/rules/dibs.mdc"])  # fmt: skip
    assert (home / "mine.md").read_text().startswith("# my agent's rules\n") and BEGIN in (home / "mine.md").read_text()
    assert BEGIN in (home / ".cursor" / "rules" / "dibs.mdc").read_text()
    assert any("mine.md" in line for line in lines)
    agents.undo(home, service=False)
    assert snapshot(home) == before
    assert not (home / ".cursor").exists()


def test_setup_reads_agent_files_from_config(tmp_path):
    home, dibs_home = tmp_path / "home", tmp_path / "dibs"
    home.mkdir()
    dibs_home.mkdir()
    (dibs_home / "config.toml").write_text('agent_files = ["~/notes/agent.md"]\n')
    env = {**os.environ, "HOME": str(home), "DIBS_HOME": str(dibs_home), "PATH": "/usr/bin:/bin"}
    run = lambda *a: subprocess.run([sys.executable, "-m", "dibs", "setup", *a], env=env, capture_output=True, text=True)  # noqa: E731
    out = run("--no-service")
    assert out.returncode == 0, out.stderr
    assert BEGIN in (home / "notes" / "agent.md").read_text()
    assert run("--undo", "--no-service").returncode == 0
    assert not (home / "notes").exists()


def test_undo_without_systemctl_on_the_machine(home, monkeypatch):
    # A container has no systemctl: undo used to die with FileNotFoundError after restoring files.
    setup(home)
    monkeypatch.setenv("PATH", str(home / "empty-bin"))
    lines = agents.undo(home, service=True)
    assert lines
    assert not (home / ".config" / "dibs" / "setup.json").exists()


def test_undo_leaves_no_blank_line_when_text_follows_the_block(home):
    # Issue #41: the blank line setup put before its block stayed behind when text followed it.
    path = home / ".claude" / "CLAUDE.md"
    path.write_text("# mine\n")
    setup(home)
    path.write_text(path.read_text() + "Added later.\n")
    agents.undo(home, service=False)
    assert path.read_text() == "# mine\nAdded later.\n"


def test_setup_says_the_guard_runs_only_when_something_runs_it(tmp_path):
    # Issue #40: setup said "The guard is on" even where no service was installed (a container).
    home, dibs_home = tmp_path / "home", tmp_path / "dibs"
    home.mkdir()
    dibs_home.mkdir()
    env = {**os.environ, "HOME": str(home), "DIBS_HOME": str(dibs_home), "PATH": "/usr/bin:/bin"}
    run = lambda *a: subprocess.run([sys.executable, "-m", "dibs", "setup", *a], env=env, capture_output=True, text=True)  # noqa: E731
    out = run("--no-service").stdout
    assert "The guard is on" not in out and "first dibs command" in out
    unit = home / ".config" / "systemd" / "user" / "dibs.service"
    unit.parent.mkdir(parents=True)
    unit.write_text("[Service]\nExecStart=/opt/dibs/bin/dibs daemon run\n")
    assert "The guard is on" in run("--no-service").stdout


def test_setup_says_what_it_did_in_good_english(home):
    # Issue #140: a rerun over an older dibs block said "updateed the dibs instructions".
    first = setup(home)
    assert any("Claude Code: added the dibs instructions in" in line for line in first), first
    claude_md = home / ".claude" / "CLAUDE.md"
    claude_md.write_text(claude_md.read_text().replace("Run `dibs agent-help`", "Run dibs agent-help"))
    again = setup(home)
    assert any("Claude Code: updated the dibs instructions in" in line for line in again), again
    assert not any("updateed" in line for line in again)


def test_tmp_in_ram_reads_the_mount_table(tmp_path):
    # Issue #165: whether /tmp is a tmpfs decides one plain line in the agents' instructions.
    mounts = tmp_path / "mounts"
    mounts.write_text("/dev/nvme0n1p5 / ext4 rw,relatime 0 0\ntmpfs /tmp tmpfs rw,nosuid,nodev,size=16777216k 0 0\n")
    assert agents.tmp_in_ram(mounts)
    mounts.write_text("/dev/nvme0n1p5 / ext4 rw,relatime 0 0\n")  # /tmp is a folder on the root disk
    assert not agents.tmp_in_ram(mounts)
    mounts.write_text("tmpfs /tmp tmpfs rw 0 0\n/dev/sdb1 /tmp ext4 rw 0 0\n")  # a later mount hides the tmpfs
    assert not agents.tmp_in_ram(mounts)
    assert not agents.tmp_in_ram(tmp_path / "missing")  # no mount table to read, as on macOS


@pytest.mark.parametrize("in_ram", [True, False])
def test_agents_are_told_plainly_whether_tmp_is_stored_in_ram(home, monkeypatch, in_ram):
    # Issue #165: agents get the fact, and no advice for or against keeping files in /tmp.
    monkeypatch.setattr(agents, "tmp_in_ram", lambda: in_ram)
    setup(home)
    for target in (home / ".claude" / "CLAUDE.md", home / ".codex" / "AGENTS.md"):
        block = target.read_text()
        assert ("/tmp on this machine is stored in RAM (a tmpfs), not on disk." in block) is in_ram, target
        if not in_ram:
            assert "/tmp" not in block, target
        for advice in ("keep big ones on disk", "take RAM from jobs", "your own scratch"):
            assert advice not in block, (target, advice)
    again = [line for line in setup(home) if "dibs instructions" in line]
    assert len(again) == 2 and all("already has" in line for line in again), again  # a rerun changes nothing


def test_agent_help_gives_no_advice_on_where_to_keep_files_in_tmp():
    # Issue #165, replacing #138's advice: nothing steers agents toward /tmp or away from it. Sizing
    # still says a job's files in a tmpfs count toward its own RAM, since that is how dibs counts it.
    neighbour = AGENT_HELP.split("## Being a good neighbour", 1)[1].split("\n## ", 1)[0]
    assert "/tmp" not in neighbour and "/tmp" not in BLOCK
    assert "/tmp where it is a tmpfs" in AGENT_HELP.split("## Sizing a request", 1)[1].split("\n## ", 1)[0]


def test_sizing_says_what_a_padded_ram_request_costs_and_how_to_fix_one():
    # Issue #230: one session's jobs peaked at a median 33% of the RAM they asked for, while jobs waited
    # for RAM. Agents are told the size to ask, what padding costs, and how to shrink a waiting job.
    sizing = " ".join(AGENT_HELP.split("## Sizing a request", 1)[1].split("\n## ", 1)[0].split())
    for phrase in ("RAM peak plus about a quarter", "run one job first and size the rest from its peak",
                   "holds back every job behind it", "--replace ID"):  # fmt: skip
        assert phrase in sizing, phrase
    assert sizing in " ".join(SKILL.split())  # the installed skill says the same


def test_sizing_doesnt_promise_a_tight_request_is_safe(tmp_path):
    # Issue #244: two tightly sized jobs were killed at their limits with RAM free: one jumped past its
    # limit within seconds, faster than dibs grows a job; the other's run needed more than earlier ones.
    sizing = " ".join(AGENT_HELP.split("## Sizing a request", 1)[1].split("\n## ", 1)[0].split())
    assert "costs little" not in sizing
    for phrase in ("jumps past it faster than dibs can act is killed at it",
                   "add `--retry` to a tightly sized job that is safe to run again",
                   "from the run with its largest input"):  # fmt: skip
        assert phrase in sizing, phrase


def test_session_names_come_from_claude_codes_session_files(tmp_path):
    # Issue #167: dibs top heads each agent's panel with the name the user knows the session by.
    # Claude Code keeps a file per running session, with the session id dibs stores with each job.
    sessions = tmp_path / ".claude" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "101.json").write_text('{"pid": 101, "sessionId": "aaaa-1", "name": "VLA Chess", "cwd": "/w"}')
    (sessions / "102.json").write_text('{"pid": 102, "sessionId": "bbbb-2", "name": "  ccsdt-5c\\n"}')
    (sessions / "103.json").write_text("not json")
    (sessions / "104.json").write_text('["a", "list"]')
    (sessions / "105.json").write_text('{"sessionId": "cccc-3"}')  # no name
    (sessions / "106.json").write_text('{"sessionId": "dddd-4", "name": "  "}')
    (sessions / "107.key").write_text('{"sessionId": "eeee-5", "name": "a key file"}')
    assert agents.session_names(tmp_path) == {"aaaa-1": "VLA Chess", "bbbb-2": "ccsdt-5c"}
    assert agents.session_names(tmp_path / "nowhere") == {}


def test_a_session_named_twice_takes_its_newest_name(tmp_path):
    sessions = tmp_path / ".claude" / "sessions"
    sessions.mkdir(parents=True)
    old, new = sessions / "201.json", sessions / "202.json"
    old.write_text('{"sessionId": "aaaa-1", "name": "old name"}')
    new.write_text('{"sessionId": "aaaa-1", "name": "new name"}')
    os.utime(old, (1_000_000, 1_000_000))
    assert agents.session_names(tmp_path) == {"aaaa-1": "new name"}
