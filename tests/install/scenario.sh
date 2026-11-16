#!/usr/bin/env bash
# Runs inside a container as the user under test: four starting states, each ending in an uninstall
# that must leave $HOME exactly as it was.
set -uo pipefail
BIN="$HOME/.local/bin"
fails=0
ok() { echo "  ok: $*"; }
bad() { echo "  FAIL: $*"; fails=$((fails + 1)); }
check() { local what="$1"; shift; if "$@" >/dev/null 2>&1; then ok "$what"; else bad "$what"; fi; }

# What a stranger types: the script piped into bash. PATH has no ~/.local/bin, as on a fresh box.
install() { cat /tmp/install.sh | bash -s -- "$@"; }
d() { env PATH="$BIN:$PATH" dibs "$@"; }
snapshot() { # file list plus a checksum of every file, for $HOME and the places dibs may use
  { find "$HOME" -mindepth 1 | sort; find "$HOME" -type f | sort | xargs -r sha256sum
    ls -d /tmp/dibs-* "/run/user/$(id -u)/dibs" 2>/dev/null || true; } 2>/dev/null
}
daemon_procs() { pgrep -f 'dibs.*daemon' || true; }

works() { # dibs is installed and answers
  check "dibs --version" d --version
  check "dibs queue answers" d queue
  check "a new interactive shell finds dibs" env -i HOME="$HOME" PATH=/usr/bin:/bin SHELL=/bin/bash bash -ic 'dibs --version'
}
uninstalled() { # $1 = snapshot taken before anything was installed
  local after; after="$(snapshot)"
  if [ "$after" = "$1" ]; then ok "uninstall left HOME as it was"
  else bad "uninstall left changes:"; diff <(echo "$1") <(echo "$after") | sed 's/^/      /'; fi
  check "no uv left" bash -c '! command -v uv && [ ! -e "$HOME/.local/bin/uv" ]'
  check "no dibs left" bash -c '! command -v dibs && [ ! -e "$HOME/.local/bin/dibs" ]'
  [ -z "$(daemon_procs)" ] && ok "no dibs daemon running" || bad "dibs daemon still running"
}

seed_user_files() {
  printf '# my shell\nalias ll="ls -l"\n' > "$HOME/.bashrc"
  mkdir -p "$HOME/.claude" "$HOME/.codex"
  printf '# my claude notes\nBe terse.\n' > "$HOME/.claude/CLAUDE.md"
  printf '# my codex notes\n' > "$HOME/.codex/AGENTS.md"
}

scenario() { echo "--- $1"; }

scenario "fresh"
before="$(snapshot)"
install || bad "install"
works
install --uninstall || bad "uninstall"
uninstalled "$before"

scenario "already installed: a second run updates and changes nothing"
before="$(snapshot)"
install || bad "first install"
mid="$(snapshot | grep -v -e /.cache/ -e /.local/share/uv/)"
install || bad "second install"
if [ "$(snapshot | grep -v -e /.cache/ -e /.local/share/uv/)" = "$mid" ]; then ok "second run changed nothing"; else bad "second run changed files"; diff <(echo "$mid") <(snapshot | grep -v -e /.cache/ -e /.local/share/uv/) | sed "s/^/      /"; fi
grep -c '>>> dibs >>>' "$HOME/.bashrc" | grep -qx 1 && ok "one PATH block" || bad "PATH block count"
works
install --uninstall || bad "uninstall"
uninstalled "$before"

scenario "edited: user files with their own content, edited again after install"
seed_user_files
before="$(snapshot)"
install || bad "install"
for f in .claude/CLAUDE.md .codex/AGENTS.md; do grep -q 'dibs' "$HOME/$f" && ok "$f has the dibs block" || bad "$f has no dibs block"; done
echo '# added later by me' >> "$HOME/.bashrc"
echo 'Added later by me.' >> "$HOME/.claude/CLAUDE.md"
install || bad "second install"
works
install --uninstall || bad "uninstall"
[ "$(cat "$HOME/.bashrc")" = "$(printf '# my shell\nalias ll="ls -l"\n# added later by me')" ] && ok ".bashrc is theirs again, with their later edit" || { bad ".bashrc differs"; cat "$HOME/.bashrc"; }
# (dibs leaves the blank line it put before its block when text follows the block: compare without blanks.)
[ "$(grep -v '^$' "$HOME/.claude/CLAUDE.md")" = "$(printf '# my claude notes\nBe terse.\nAdded later by me.')" ] && ok "CLAUDE.md keeps both edits" || { bad "CLAUDE.md differs"; cat "$HOME/.claude/CLAUDE.md"; }
[ "$(cat "$HOME/.codex/AGENTS.md")" = "# my codex notes" ] && ok "AGENTS.md restored" || bad "AGENTS.md differs"
ls "$HOME"/.bashrc.before-dibs >/dev/null 2>&1 && bad "backup left behind"
rm -f "$HOME/.bashrc" "$HOME/.claude/CLAUDE.md" "$HOME/.codex/AGENTS.md"; rmdir "$HOME/.claude" "$HOME/.codex" 2>/dev/null
rest="$(snapshot)"; [ "$rest" = "$(echo "$before" | grep -v -e 'bashrc' -e '\.claude' -e '\.codex')" ] && ok "nothing else left" || { bad "other files left"; diff <(echo "$before") <(echo "$rest") | sed 's/^/      /'; }

scenario "running: the daemon is up during an update and during uninstall"
before="$(snapshot)"
install || bad "install"
d queue >/dev/null 2>&1; d daemon status >/dev/null 2>&1 && ok "daemon is running" || bad "daemon not running"
install || bad "update while running"
works
d daemon status >/dev/null 2>&1 && ok "daemon answers after the update" || bad "daemon down after update"
install --uninstall || bad "uninstall while running"
uninstalled "$before"

[ "$fails" -eq 0 ] && { echo "ALL PASSED"; exit 0; }
echo "$fails FAILED"; exit 1
