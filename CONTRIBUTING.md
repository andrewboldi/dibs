# Contributing to dibs

dibs is small, standard-library Python. Every change is tracked the same way, whether a person or
an agent makes it.

## Track the work with beads (required)

Every change starts as an issue in [beads](https://github.com/gastownhall/beads) (`bd`), mirrored to
this repo's GitHub issues, and lands through a pull request that closes it. Agents working here get
the workflow from `AGENTS.md` (or `CLAUDE.md`) and `bd prime`.

Install bd once, any of these ways:

```bash
brew install beads
npm install -g @beads/bd
curl -fsSL https://raw.githubusercontent.com/gastownhall/beads/main/scripts/install.sh | bash
```

Then, in your clone:

```bash
bd init --non-interactive --skip-agents --role contributor
bd config set github.owner andrewboldi
bd config set github.repo dibs
GITHUB_TOKEN=$(gh auth token) bd github sync --pull-only   # bring in the GitHub issues
bd ready                                                   # what nobody has claimed
```

The beads database stays on your machine; issues are shared through GitHub. `--skip-agents` keeps
`bd init` from rewriting the repo's `AGENTS.md` and Claude Code settings.

For each change:

1. Find the issue, or file it: `bd create "What's wrong or missing" -d "Why it matters"
   --acceptance "Done when..."`. Then give it a GitHub number:
   `GITHUB_TOKEN=$(gh auth token) bd github sync --push-only --issues <id>`.
2. Claim it (`bd update <id> --claim`) and work on a branch.
3. Open a pull request that says `Fixes #N` in its description. A check fails pull requests that
   don't name their issue.

## Run the tests

```bash
uv sync
uv run pytest -q    # about a minute; everything must pass
```

- Test first: write the test that shows the problem, watch it fail, then fix it.
- Tests and experiments must never touch a real dibs on your machine. Set `DIBS_HOME` to a temp
  folder, and use `DIBS_GPU_BACKEND=smi` with `tests/fakes` first on `PATH` (or `guard = "off"` in
  that folder's `config.toml`). A daemon that can see the real GPU with the guard on stops other
  programs' GPU processes. Never run `dibs setup` against your real `$HOME` in a test.
- When an agent runs the suite, run it in a throwaway scope, so a broken test can't reach the
  agent's own processes: `systemd-run --user --scope --quiet --collect -- uv run pytest -q`.

## Write the code

- Python 3.11+, standard library only at runtime.
- Commit messages: a plain sentence as the subject, and a body that says why. No prefixes.
