#!/usr/bin/env bash
# Install dibs, the job queue that lets coding agents share one machine's GPU, CPU and RAM:
#
#   curl -fsSL https://raw.githubusercontent.com/andrewboldi/dibs/main/install.sh | bash
#
# What it does:
#   1. Checks this is Linux, and installs tar and gzip with the system package manager if missing
#      (through sudo when you aren't root).
#   2. Installs uv (https://docs.astral.sh/uv/) into ~/.local/bin if it isn't there. uv fetches
#      its own Python 3.11+, so you don't need one.
#   3. Installs dibs as a uv tool, or updates it if it's already installed.
#   4. Puts ~/.local/bin on your PATH, between "# >>> dibs >>>" markers in ~/.bashrc (or
#      ~/.zshrc), only if it isn't on PATH already. The original is kept as FILE.before-dibs.
#   5. Runs `dibs setup`: instructions for your agent CLIs, and a systemd user service if you have
#      a systemd user session.
#
# Run it again to update. To undo exactly what it did:
#
#   curl -fsSL https://raw.githubusercontent.com/andrewboldi/dibs/main/install.sh | bash -s -- --uninstall
#
# What it added is recorded in ~/.local/share/dibs-install/record, and uninstall removes only that.
# Options: --uninstall, --keep-data (with --uninstall: keep dibs's job history and config), --help.
# Environment: DIBS_SOURCE=<folder or URL to install from; default is the GitHub source archive>.

set -euo pipefail

DEFAULT_SOURCE="https://github.com/andrewboldi/dibs/archive/refs/heads/main.tar.gz"
UNINSTALL_CMD="curl -fsSL https://raw.githubusercontent.com/andrewboldi/dibs/main/install.sh | bash -s -- --uninstall"
BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"
RECORD_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/dibs-install"
RECORD="$RECORD_DIR/record"
MARK_BEGIN="# >>> dibs >>>"
MARK_END="# <<< dibs <<<"

say() { printf '%s\n' "$*"; }
die() { printf 'dibs install: %s\n' "$*" >&2; exit 1; }

# ---- the record of what we added -------------------------------------------------------------

record_has() { [ -f "$RECORD" ] && grep -qxF "$1" "$RECORD"; }
record_add() {
  mkdir -p "$RECORD_DIR"
  record_has "$1" || printf '%s\n' "$1" >> "$RECORD"
}
record_value() { [ -f "$RECORD" ] && sed -n "s/^$1=//p" "$RECORD" | head -n 1 || true; }

# ---- checks and system packages --------------------------------------------------------------

check_platform() {
  [ "$(uname -s)" = "Linux" ] || die "dibs runs on Linux only (this is $(uname -s))."
  case "$(uname -m)" in
    x86_64|aarch64|arm64) ;;
    *) die "unsupported CPU $(uname -m): dibs needs x86_64 or aarch64." ;;
  esac
  [ -n "${HOME:-}" ] && [ -d "$HOME" ] || die "HOME isn't set to a folder; set it and run this again."
  command -v curl >/dev/null 2>&1 || die "curl is missing. Install curl with your package manager, then run this again."
}

as_root() {
  if [ "$(id -u)" -eq 0 ]; then "$@"
  elif command -v sudo >/dev/null 2>&1; then sudo "$@"
  else die "need root to install $NEEDED_PKGS and there's no sudo. Run this as root, or install them yourself: $NEEDED_PKGS."
  fi
}

# Installs what the uv installer unpacks with. Never `pacman -Sy <pkg>`: a partial upgrade breaks Arch.
install_system_packages() {
  NEEDED_PKGS=""
  command -v tar >/dev/null 2>&1 || NEEDED_PKGS="$NEEDED_PKGS tar"
  command -v gzip >/dev/null 2>&1 || NEEDED_PKGS="$NEEDED_PKGS gzip"
  NEEDED_PKGS="${NEEDED_PKGS# }"
  [ -n "$NEEDED_PKGS" ] || return 0
  say "Installing $NEEDED_PKGS with your package manager..."
  # shellcheck disable=SC2086
  if command -v apt-get >/dev/null 2>&1; then
    as_root apt-get update -qq && as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $NEEDED_PKGS \
      || die "apt-get couldn't install $NEEDED_PKGS. Run 'sudo apt-get update', then run this again."
  elif command -v dnf >/dev/null 2>&1; then
    as_root dnf install -y -q $NEEDED_PKGS || die "dnf couldn't install $NEEDED_PKGS. Run 'sudo dnf upgrade --refresh', then run this again."
  elif command -v pacman >/dev/null 2>&1; then
    as_root pacman -S --needed --noconfirm $NEEDED_PKGS \
      || die "pacman couldn't install $NEEDED_PKGS. Update your system first (sudo pacman -Syu), then run this again."
  else
    die "no apt, dnf or pacman found. Install $NEEDED_PKGS yourself, then run this again."
  fi
  record_add "packages=$NEEDED_PKGS"
}

# ---- uv --------------------------------------------------------------------------------------

find_uv() {
  if command -v uv >/dev/null 2>&1; then command -v uv
  elif [ -x "$BIN_DIR/uv" ]; then printf '%s\n' "$BIN_DIR/uv"
  fi
}

ensure_uv() {
  UV="$(find_uv || true)"
  if [ -n "$UV" ]; then return 0; fi
  say "Installing uv into $BIN_DIR (it fetches its own Python)..."
  # UV_UNMANAGED_INSTALL: no receipt file and no shell edits from uv; we handle PATH ourselves.
  curl -fsSL https://astral.sh/uv/install.sh | UV_UNMANAGED_INSTALL="$BIN_DIR" sh >/dev/null 2>&1 \
    || die "couldn't install uv. Check your network, or install it from https://docs.astral.sh/uv/ and run this again."
  UV="$BIN_DIR/uv"
  [ -x "$UV" ] || die "uv's installer ran but $UV isn't there. Install uv from https://docs.astral.sh/uv/ and run this again."
  record_add "uv_dir=$BIN_DIR"
}

# ---- dibs ------------------------------------------------------------------------------------

source_spec() {
  local src="${DIBS_SOURCE:-$DEFAULT_SOURCE}"
  if [ -d "$src" ]; then (cd "$src" && pwd)
  elif [ -f "$src" ]; then printf 'dibs @ file://%s\n' "$(cd "$(dirname "$src")" && pwd)/$(basename "$src")"
  else printf 'dibs @ %s\n' "$src"
  fi
}

install_dibs() {
  local was=""
  "$UV" tool list </dev/null 2>/dev/null | grep -q '^dibs ' && was=1
  if [ -n "$was" ]; then say "Updating dibs..."; else say "Installing dibs..."; fi
  "$UV" tool install --force --reinstall --refresh --python ">=3.11" "$(source_spec)" </dev/null >/dev/null 2>"${TMPDIR:-/tmp}/dibs-install.$$.log" \
    || { cat "${TMPDIR:-/tmp}/dibs-install.$$.log" >&2; rm -f "${TMPDIR:-/tmp}/dibs-install.$$.log"; die "uv couldn't install dibs (output above)."; }
  rm -f "${TMPDIR:-/tmp}/dibs-install.$$.log"
  [ -x "$BIN_DIR/dibs" ] || die "dibs installed but $BIN_DIR/dibs isn't there; check 'uv tool dir --bin'."
}

# ---- PATH ------------------------------------------------------------------------------------

rc_file() {
  case "${SHELL:-}" in
    */zsh) printf '%s\n' "${ZDOTDIR:-$HOME}/.zshrc" ;;
    *) printf '%s\n' "$HOME/.bashrc" ;;
  esac
}

on_path() { case ":$PATH:" in *":$BIN_DIR:"*) return 0 ;; *) return 1 ;; esac; }

ensure_path() {
  PATH_ADDED=""
  on_path && return 0
  export PATH="$BIN_DIR:$PATH"  # for the rest of this script
  local rc; rc="$(rc_file)"
  if [ -f "$rc" ] && grep -qxF "$MARK_BEGIN" "$rc"; then PATH_ADDED="$rc"; return 0; fi
  if [ -f "$rc" ]; then
    [ -e "$rc.before-dibs" ] || cp -p "$rc" "$rc.before-dibs"
    record_add "rc_backup=$rc.before-dibs"
    [ -z "$(tail -c1 "$rc")" ] || printf '\n' >> "$rc"
  else
    record_add "rc_created=$rc"
  fi
  # shellcheck disable=SC2016
  printf '\n%s\nexport PATH="%s:$PATH"\n%s\n' "$MARK_BEGIN" "$BIN_DIR" "$MARK_END" >> "$rc"
  record_add "rc_file=$rc"
  PATH_ADDED="$rc"
}

# ---- install / uninstall ---------------------------------------------------------------------

do_install() {
  check_platform
  install_system_packages
  ensure_uv
  install_dibs
  ensure_path
  local version; version="$("$BIN_DIR/dibs" --version </dev/null)"
  say ""
  "$BIN_DIR/dibs" setup </dev/null || die "'dibs setup' failed (output above). dibs itself is installed; fix the problem and run 'dibs setup'."
  say ""
  say "Done: $version is installed."
  say "  - dibs: $BIN_DIR/dibs (a uv tool)"
  [ -z "$(record_value uv_dir)" ] || say "  - uv: installed in $BIN_DIR"
  [ -z "$PATH_ADDED" ] || say "  - PATH: added $BIN_DIR to $PATH_ADDED (open a new shell to pick it up)"
  say "  - agent instructions: written by 'dibs setup' (listed above)"
  say "Try: dibs queue"
  say "Undo everything: $UNINSTALL_CMD"
}

strip_rc_block() {
  local rc="$1" tmp
  [ -f "$rc" ] && grep -qxF "$MARK_BEGIN" "$rc" || return 0
  tmp="$(mktemp)"
  # Drop the marked block and the one blank line the installer put before it.
  awk -v b="$MARK_BEGIN" -v e="$MARK_END" '
    { line[NR] = $0 }
    END {
      for (i = 1; i <= NR; i++) {
        if (line[i] == "" && line[i + 1] == b) continue
        if (line[i] == b) { while (i <= NR && line[i] != e) i++; continue }
        print line[i]
      }
    }
  ' "$rc" > "$tmp"
  cat "$tmp" > "$rc"; rm -f "$tmp"
}

# Remove DIR and its parents up to (not including) HOME, as far as they're empty.
prune_empty() {
  local d="$1"
  while [ -n "$d" ] && [ "$d" != "$HOME" ] && [ "$d" != "/" ] && rmdir "$d" 2>/dev/null; do d="$(dirname "$d")"; done
}

do_uninstall() {
  [ "$(uname -s)" = "Linux" ] || die "dibs runs on Linux only (this is $(uname -s))."
  local keep_data="$1" UV dibs_bin="" removed=""
  UV="$(find_uv || true)"
  [ -x "$BIN_DIR/dibs" ] && dibs_bin="$BIN_DIR/dibs"
  [ -n "$dibs_bin" ] || dibs_bin="$(command -v dibs || true)"

  if [ -n "$dibs_bin" ]; then
    "$dibs_bin" setup --undo </dev/null || true
    "$dibs_bin" daemon stop </dev/null >/dev/null 2>&1 || true
  fi
  if [ -n "$UV" ]; then
    "$UV" tool uninstall dibs </dev/null >/dev/null 2>&1 && removed="$removed dibs"
  fi

  local rc backup
  rc="$(record_value rc_file)"
  if [ -n "$rc" ]; then
    strip_rc_block "$rc"
    backup="$(record_value rc_backup)"
    if [ -n "$backup" ] && [ -f "$backup" ]; then
      rm -f "$backup"  # the stripped file is the original again; the copy has done its job
    fi
    if [ -n "$(record_value rc_created)" ] && [ -f "$rc" ] && [ -z "$(tr -d '[:space:]' < "$rc")" ]; then
      rm -f "$rc"
    fi
    removed="$removed PATH-edit"
  fi

  local uv_dir; uv_dir="$(record_value uv_dir)"
  if [ -n "$uv_dir" ] && [ -n "$UV" ]; then
    local d
    for d in "$("$UV" python dir 2>/dev/null || true)" "$("$UV" tool dir 2>/dev/null || true)" "$("$UV" cache dir 2>/dev/null || true)"; do
      if [ -n "$d" ] && [ -d "$d" ]; then rm -rf "$d"; prune_empty "$(dirname "$d")"; fi
    done
    rm -f "$uv_dir/uv" "$uv_dir/uvx" "$uv_dir/uvw"
    removed="$removed uv"
  fi

  if [ -z "$keep_data" ]; then
    rm -rf "${XDG_STATE_HOME:-$HOME/.local/state}/dibs" "${XDG_CONFIG_HOME:-$HOME/.config}/dibs" \
      "/tmp/dibs-$(id -u)" "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/dibs" 2>/dev/null || true
    prune_empty "${XDG_STATE_HOME:-$HOME/.local/state}"; prune_empty "${XDG_CONFIG_HOME:-$HOME/.config}"
  fi

  local pkgs; pkgs="$(record_value packages)"
  rm -rf "$RECORD_DIR"
  prune_empty "$BIN_DIR"; prune_empty "$(dirname "$RECORD_DIR")"

  say "Removed dibs${removed:+ and what the installer added:$removed}."
  [ -n "$keep_data" ] || say "Its job history and config are gone too."
  [ -z "$pkgs" ] || say "The installer also installed $pkgs with your package manager; those are left in place."
}

main() {
  local mode=install keep_data=""
  for arg in "$@"; do
    case "$arg" in
      --uninstall) mode=uninstall ;;
      --keep-data) keep_data=1 ;;
      -h|--help) sed -n '2,24p' "$0" 2>/dev/null | sed 's/^# \{0,1\}//' ; return 0 ;;
      *) die "unknown option $arg (try --help)." ;;
    esac
  done
  if [ "$mode" = uninstall ]; then do_uninstall "$keep_data"; else do_install; fi
}

if [ -z "${DIBS_INSTALL_LIB:-}" ]; then main "$@"; fi
