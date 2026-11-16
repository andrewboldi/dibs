# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:7510c1e2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->


## Build & Test

```bash
uv sync                    # dev environment (Python 3.11+, pytest)
uv run pytest -q           # the whole suite, about 70 s; everything must pass
uv run python -m dibs ...  # run the CLI from this checkout
```

Tests and experiments must never touch this machine's real dibs: set `DIBS_HOME` to a temp folder,
and use `DIBS_GPU_BACKEND=smi` with `tests/fakes` first on `PATH` (or `guard = "off"`/`"warn"` in
that folder's `config.toml`). A daemon that can see the real GPU with the guard on stops other
agents' GPU processes. Never run `dibs setup` against the real `$HOME` in a test.

## Architecture Overview

- `src/dibs/planner.py`: the pure planner (priority tags, conservative backfill, predictions).
- `src/dibs/daemon.py`: the scheduler process: socket server, ticks, VRAM watch, guard wiring.
- `src/dibs/launcher.py`, `src/dibs/shim.py`: starting jobs (systemd scope, CPU pinning, environment).
- `src/dibs/guard.py`, `src/dibs/procs.py`: judging GPU processes; stopping them with a message.
- `src/dibs/client.py`, `src/dibs/cli.py`, `src/dibs/render.py`: the `dibs` command.
- `src/dibs/agents.py`, `src/dibs/instructions.py`: `dibs setup` and what agents are told.
- The design: `docs/superpowers/specs/2026-11-13-dibs-design.md`.

## Conventions & Patterns

- Python 3.11+, standard library only at runtime.
- Test first: write the test that shows the problem, watch it fail, then fix.
- Every change starts as a bd issue mirrored to a GitHub issue (`bd github sync`), and lands
  through a branch and a pull request that closes it.
- Commit messages: a plain sentence as the subject and a body that says why; no prefixes.
