import os

from dibs.client import current_session


def test_claude_session_used_when_present(tmp_path):
    env = {"CLAUDE_CODE_SESSION_ID": "abcd1234-5678"}
    assert current_session(env, str(tmp_path)) == ("abcd1234-5678", f"{tmp_path.name}·abcd")


def test_explicit_dibs_session_beats_detected_one(tmp_path):
    env = {"CLAUDE_CODE_SESSION_ID": "abcd1234-5678", "DIBS_SESSION": "zzzz9999"}
    assert current_session(env, str(tmp_path))[0] == "zzzz9999"


def test_terminal_session_fallback(tmp_path):
    assert current_session({}, str(tmp_path))[0] == f"t{os.getsid(0)}"


def test_label_uses_git_root(tmp_path):
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    (tmp_path / "repo" / "src" / "deep").mkdir(parents=True)
    _, label = current_session({"DIBS_SESSION": "s123"}, str(tmp_path / "repo" / "src" / "deep"))
    assert label == "repo·s123"
