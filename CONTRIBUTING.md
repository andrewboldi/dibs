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

## Add an agent CLI

`dibs setup` writes the dibs instructions into each agent CLI's own instructions file. To support
another CLI for everyone, add a line to `AGENTS` in `src/dibs/agents.py`: the CLI's name, its config
folder under the home folder, the instructions file in that folder, and its binary (setup detects an
agent by either one). Add a test in `tests/test_agents.py`.

To add one just for yourself, without a pull request, list its instructions file in
`~/.config/dibs/config.toml`, then run `dibs setup` again. `dibs setup --undo` takes it back out.

```toml
agent_files = ["~/.cursor/rules/dibs.mdc"]
```

## Add a GPU backend

A backend is one class in `src/dibs/gpu.py`. `list_gpus()` says what's there; `processes()` says
which processes use which GPU and how much memory, or `None` if the driver can't tell (that turns
the VRAM watch and the guard off). `vendor` decides which variable hides other GPUs from a job.

```python
class ExampleBackend:
    name = "example-smi"  # what dibs info shows
    vendor = "nvidia"  # "nvidia": CUDA_VISIBLE_DEVICES; "amd": ROCR_VISIBLE_DEVICES

    def list_gpus(self) -> list[Gpu]:
        # One Gpu(index, uuid, name, vram in bytes) per device, numbered as jobs will see them.
        return [Gpu(0, "GPU-0", "Example GPU", 8 << 30)]

    def processes(self) -> list[GpuProcess] | None:
        # Every compute process: GpuProcess(pid, gpu index, VRAM in bytes or None).
        return [GpuProcess(1234, 0, 2 << 30)]
```

Then choose it in `open_backend`, both by `DIBS_GPU_BACKEND` and in the auto-detection order, and
test it against a stand-in for the vendor's tool in `tests/fakes/`, as `tests/fakes/amd-smi` does.
List only compute processes: the guard stops what `processes()` returns, so a desktop's graphics
clients must never be in it.

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
