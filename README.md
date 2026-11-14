<p align="center">
  <img src="assets/logo.svg" alt="" width="132">
</p>

<h1 align="center">dibs</h1>

<p align="center"><b>One GPU, many coding agents. They take turns.</b></p>

<p align="center">
  <a href="https://github.com/andrewboldi/dibs/actions/workflows/test.yml"><img alt="tests" src="https://img.shields.io/github/actions/workflow/status/andrewboldi/dibs/test.yml?branch=main&label=tests&style=flat-square"></a>
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/github/license/andrewboldi/dibs?style=flat-square"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776ab?style=flat-square&logo=python&logoColor=white">
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#for-agents">For agents</a> ·
  <a href="#who-goes-next">Who goes next</a> ·
  <a href="#what-dibs-enforces">What it enforces</a> ·
  <a href="#commands">Commands</a> ·
  <a href="#configuration">Configuration</a>
</p>

Run a few Claude Code, Codex or opencode sessions on one machine and sooner or later two of them
load a model onto the GPU at the same time, and one dies out of memory. dibs is a Slurm-style
queue for a single machine: each agent says what a job needs (CPU cores, RAM, GPU memory) and
roughly how long it will take, and dibs runs it when it's its turn. Short jobs go first, long jobs
still get theirs, and anything that grabs the GPU behind the queue's back is stopped.

<p align="center">
  <img src="assets/demo.gif" alt="dibs top above, agents below. A training job holds the GPU; a long fine-tune queues behind it; a short eval is queued next and goes ahead of the fine-tune; a CPU-only job starts at once in the gap; then a script started without dibs is stopped within a second, with a note in its output saying how to queue it." width="820">
</p>

## What it does

- **Reservations, not guesses.** A job declares its cores, RAM and VRAM up front and gets exactly
  that: pinned to its cores, its RAM capped, its VRAM watched, and no GPU at all if it didn't ask
  for one. Jobs that fit run side by side; jobs that don't, wait.
- **Short jobs first, and nobody starves.** The time estimate decides the order and every queued
  job's predicted start time. It never stops anything: a job that runs long just runs long.
- **Gaps get filled.** A small job starts early if it fits around the jobs ahead of it without
  delaying any of them.
- **Every agent can see the queue.** `dibs queue` says where each job is in line, when it should
  start, and why it's waiting: `needs 8G VRAM (2G free)`.
- **Any agent CLI, no exceptions.** `dibs setup` tells Claude Code, Codex, opencode, pi and Gemini
  CLI how to use dibs. The guard watches the GPU itself, so a GPU process started any other way is
  stopped, whichever agent (or person) started it.
- **Jobs outlive everything else.** They run detached: a closed session, a command that hit its
  timeout or a dibs restart doesn't touch them.

## Install

```sh
uv tool install git+https://github.com/andrewboldi/dibs
dibs setup
```

(Or `pipx install git+https://github.com/andrewboldi/dibs`.) Python 3.11 or newer, on Linux. No root, no dependencies. dibs works as soon as it's installed: the
first `dibs` command starts the scheduler. `dibs setup` is what makes it the rule:

- adds a short marked block about dibs to the global instructions of every agent CLI you have
  (`~/.claude/CLAUDE.md`, `~/.codex/AGENTS.md`, `~/.config/opencode/AGENTS.md`,
  `~/.pi/agent/AGENTS.md`, `~/.gemini/GEMINI.md`), plus a Claude Code skill with the details;
- installs a systemd user service, with lingering on, so dibs and its guard always run, even
  before any agent has called it and after you log out.

`dibs setup --dry-run` shows what it would change; `dibs setup --undo` puts every file back as it
was. Using another agent CLI? Paste the output of `dibs agent-help` into its instructions.

## For agents

```sh
dibs submit --cpus 4 --mem 16G --vram 6G --time 20m -- python train.py --epochs 3
dibs wait 12        # until it ends; exits with the job's exit code (124: --timeout ran out)
dibs queue          # where every job is in line, when it should start, why it's waiting
dibs logs 12 -f     # its output
dibs cancel 12      # as soon as it's no longer needed
```

- `--cpus`, `--mem` and `--time` are required; add `--vram` for a GPU (`--gpus 2` for two). `all`
  works for a whole machine or a whole GPU.
- The command runs in the current folder with the current environment, so virtualenvs work.
- `dibs run …` is submit plus the output as it comes plus the job's exit code, for short jobs.
- An agent whose commands time out runs `dibs wait` in the background (Claude Code:
  `run_in_background`). Interrupting `dibs wait` or `dibs run` never cancels the job.
- `dibs show 12` has everything about one job, including the peak VRAM and RAM it used, for sizing
  the next one.

## Who goes next

If two jobs both need the whole machine and one takes a minute while the other takes ten, the
one-minute job goes first: its agent is unblocked nine minutes sooner, and the long job starts
only a minute later.

Each job is due at its submit time plus its estimate, and the job due first goes first. That gives
three guarantees:

1. **Shortest first.** Jobs submitted around the same time run shortest first.
2. **No starvation.** Once a job has waited as long as its own estimate, no newer job can pass it.
3. **No crowding out.** A session's queued jobs line up behind each other, so an agent that queues
   fifty jobs interleaves with everyone else instead of going ahead of them all.

Every queued job gets a place on a timeline: the earliest moment it fits for its whole estimated
run, around the running jobs and the places already given to jobs ahead of it. A small job whose
place is "now" starts now, so gaps get filled without delaying anyone. `dibs top` draws that
timeline live. Predictions are only as good as the estimates, and they shift when a job runs
longer than expected.

## What dibs enforces

| | How | Where |
|---|---|---|
| CPU cores | each job is pinned to its cores; `OMP_NUM_THREADS` and friends match | Linux |
| RAM | the job runs in its own systemd scope with `MemoryMax`; going over kills it, not its neighbours | Linux with a systemd user session |
| VRAM | watched every second through NVML; a job over its reservation is stopped, with the reason | NVIDIA |
| Which GPUs | `CUDA_VISIBLE_DEVICES` shows a job only its GPUs, and none to jobs without `--vram` | NVIDIA |
| GPU use outside dibs | the guard stops it within about a second, after writing into its output how to queue it | needs the daemon running (`dibs setup`) |
| Time | never: the estimate only orders the queue | |

The guard leaves alone processes of other users, programs in `guard_allow`, and work that was
already running when it was switched on (including what that work starts later, like torchrun
restarting a worker), and lists them in `dibs queue`. Its limits, plainly: a job
that grabs memory very fast can still collide once before it's stopped; CPU and RAM use outside
dibs isn't stopped (nothing reliably tells a heavy job from a build or a test run); and an agent
determined to get around it could stop the daemon, which its instructions tell it not to do.
`dibs info` shows what's enforced on your machine.

## Commands

| | |
|---|---|
| `dibs submit [resources] -- CMD …` | queue a job; prints its id and when it's expected to start |
| `dibs run [resources] -- CMD …` | queue it, show its output, exit with its exit code |
| `dibs queue [--all] [--mine]` | running and queued jobs (`--all`: and the last day's finished ones) |
| `dibs top` | the live view from the demo |
| `dibs show ID` | everything about one job |
| `dibs wait ID… [--timeout 9m]` | wait for jobs to end |
| `dibs logs ID [-f] [-n N]` | a job's output |
| `dibs cancel ID…` | cancel queued or running jobs |
| `dibs info` | what dibs hands out and enforces |
| `dibs setup [--dry-run] [--undo]` | tell your agent CLIs, keep dibs running |
| `dibs agent-help` | the full instructions agents follow |
| `dibs daemon start\|stop\|restart\|status` | the scheduler itself; stopping it never stops jobs |

Commands that show something take `--json`. Sizes are binary (`16G` is 16 GiB); times look like
`90s`, `20m`, `1h30m` or `2d`.

## Configuration

dibs hands out every CPU, all the RAM and every NVIDIA GPU it finds. `~/.config/dibs/config.toml`
can hold some back:

```toml
cpus = 20            # hand out only this many CPUs
mem = "24G"          # and this much RAM
gpus = [0]           # manage only these GPUs
vram = "7.5G"        # at most this much VRAM per GPU
guard = "all"        # "all" stops GPU processes outside dibs, "warn" only lists them, "off"
guard_allow = ["ollama"]
keep_days = 14       # how long finished jobs' logs are kept
```

Run `dibs daemon restart` after changing it; running jobs keep running.

## How it works

One scheduler per user runs in the background and owns the queue, which it keeps in SQLite, so a
crash or restart loses nothing. The `dibs` command talks to it over a Unix socket and starts it if
it isn't running. Each job runs under a small launcher in its own session (and, with systemd, its
own memory-capped scope), so it never depends on the agent that submitted it. Every second the
scheduler starts whatever can start, notices what has finished, and looks at the GPUs through NVML
(about 0.2 ms of CPU). Everything is in the [design](docs/superpowers/specs/2026-11-13-dibs-design.md).

## Caveats

- To run jobs with the submitter's virtualenv and settings, dibs stores each job's environment,
  which can include API keys, in `~/.local/state/dibs`, readable only by you.
- One user per machine; NVIDIA GPUs only. macOS runs, but without core pinning, RAM caps or the
  guard.

## License

MIT.
