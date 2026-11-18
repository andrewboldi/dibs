# How dibs works

One scheduler per user runs in the background and owns the queue. The `dibs` command talks to it
over a Unix socket, and starts it if it isn't running. With `dibs setup`, a systemd user service
keeps it running from login on, guard included. The queue lives in SQLite, so a crash or restart
loses nothing, and jobs keep running while the scheduler is down. Every second the scheduler
starts what can start, notices what has finished, measures memory, and looks at the GPUs.

## Who goes next

If two jobs both need the whole machine and one takes a minute while the other takes ten, the
one-minute job goes first: its agent is unblocked nine minutes sooner, and the long job starts only
a minute later.

Each job is due at its submit time plus its estimate, and the job due first goes first. That gives
three guarantees:

1. **Shortest first.** Jobs submitted around the same time run shortest first.
2. **No starvation.** Once a job's place in line comes due (for a job at the default priority,
   after it has waited its estimate), no newer job can pass it.
3. **No crowding out.** A session's queued jobs line up behind each other, so an agent that queues
   fifty jobs interleaves with everyone else instead of going ahead of them all.

Every queued job gets a place on a timeline: the earliest moment it fits for its whole estimated
run, around the running jobs and the places already given to the jobs ahead of it. A small job
whose place is "now" starts now, so gaps get filled without delaying anyone. A job past its
estimate is assumed to need another 10% of it (at least a minute), and nothing is ever stopped for
running long: the estimate only orders the queue. `dibs top` draws the timeline live.

**Priority.** `--priority N` (1 to 10, default 5) lets an agent say a job is urgent. A higher
priority makes the job count as shorter in line (at 10, a tenth of its estimate), so it goes ahead
sooner, and its session's next job pays that time back: it lands that much later than it would
have. A lower priority does the reverse and earns the next job a head start, up to an
hour. Priority moves an agent's own work around but can't take other agents' share, so an agent
that always picks 10 gains nothing after its first job. What a session owes lapses after a day.

`dibs queue` says why each waiting job waits, e.g. `needs 6 CPUs (2 free), 4G VRAM (3G free)`.

## What a job gets, and what's enforced

| | How | Where |
|---|---|---|
| CPU cores | pinned to its cores; `OMP_NUM_THREADS` and the like set to match | Linux |
| RAM | its own systemd scope with `MemoryMax` and no swap; going over kills the job, not its neighbours, and dibs runs it again with more | systemd user session |
| VRAM | watched every second; a job may borrow VRAM nobody reserved, and one that reaches another job's is stopped, with the reason, and run again with more | NVIDIA (NVML or nvidia-smi), AMD (amd-smi) |
| Which GPUs | `CUDA_VISIBLE_DEVICES` (NVIDIA) or `ROCR_VISIBLE_DEVICES` (AMD) shows a job only its GPUs, and none to a job without `--vram` | |
| GPU use outside dibs | the guard stops it within about a second, after writing into its output how to queue it | daemon running |
| Time | never enforced: the estimate only orders the queue | |

`dibs info` shows what's enforced on your machine.

**RAM other programs use.** Browsers, editors and the agents themselves use RAM too, and dibs
doesn't hand that out. It measures the RAM in use outside its jobs (in use, minus what each job
holds) and plans with its highest value over the last two minutes, so a job doesn't start into RAM
a busy program let go of for a moment. A job that doesn't fit beside it waits with a reason like
`needs 12G RAM (10G free, 6G in use outside dibs)`.

**Timing runs.** `--exclusive` gives a job the machine to itself, for benchmarks: it waits until no
other job is running, and no job starts beside it. It is still pinned to its own `--cpus` and sees
only its own GPUs; in the plan it holds everything.

**The guard.** It stops GPU processes that aren't dibs jobs. It leaves alone other users'
processes, programs listed in `guard_allow`, programs launched from the desktop, and work that was
already running when it was switched on (including what that work starts later, like torchrun
restarting a worker), and lists all of them in `dibs queue`. "Launched from the desktop" means the
process runs in the scope a desktop launcher gives an app (`app-gnome-…`, `app-…@….service`), for an
app that doesn't host a shell: terminals and editors with a built-in terminal, where agents run,
are still policed, and so is a process an agent detached with `nohup` or `setsid`, which keeps its
terminal's scope. `guard_desktop = false` polices desktop programs too. On AMD machines it only looks at processes with the ROCm compute device
(`/dev/kfd`) open, so a desktop's graphics clients are never touched. `guard = "warn"` only lists
offenders; `guard = "off"` turns it off.

**Sizing.** `dibs show ID` lists a job's peak RAM (memory in use, not file cache) and peak VRAM.
`dibs submit` shows what earlier runs of the same command in the same folder used:

```
Earlier runs of this command here:
  #41 done after 19m (estimated 40m): 2.9 of 6 CPUs, RAM 2.8G of 6G, VRAM 4.1G of 5G
```

## Commands

| | |
|---|---|
| `dibs submit [resources] -- CMD …` | queue a job; prints its id, when it's expected to start, and earlier runs |
| `dibs run [resources] -- CMD …` | queue it, show its output, exit with its exit code |
| `dibs queue [--all] [--mine]` | running and queued jobs (`--all`: and the last day's finished ones) |
| `dibs top` | the live view from the demo |
| `dibs show ID` | everything about one job |
| `dibs wait ID… [--timeout 9m]` | wait for jobs to end (exit 124 when the timeout runs out) |
| `dibs logs ID [-f] [-n N]` | a job's output |
| `dibs cancel ID…` | cancel queued or running jobs |
| `dibs info` | what dibs hands out and enforces |
| `dibs setup [--dry-run] [--undo]` | tell your agent CLIs about dibs, keep dibs running |
| `dibs agent-help` | the full instructions agents follow |
| `dibs daemon start\|stop\|restart\|status` | the scheduler itself; stopping it never stops jobs |

Resources: `--cpus N --mem SIZE --time DURATION` are required, plus `--vram SIZE` for a GPU
(`--gpus 2` for two); `all` asks for a whole machine or a whole GPU. `--priority N` is optional. Sizes are binary (`16G` is 16
GiB); times look like `90s`, `20m`, `1h30m` or `2d`. Commands that show something take `--json`.
The command runs in the current folder with the current environment, so virtualenvs work.

## Configuration

dibs hands out every CPU, the RAM not in use by other programs, and every GPU it finds.
`~/.config/dibs/config.toml` can hold some back:

```toml
cpus = 20            # hand out only this many CPUs (the lowest-numbered ones)
mem = "24G"          # and at most this much RAM
gpus = [0]           # manage only these GPUs
vram = "7.5G"        # at most this much VRAM per GPU
guard = "all"        # "all" stops GPU processes outside dibs, "warn" only lists them, "off"
guard_allow = ["ollama"]
guard_desktop = true # leave alone GPU programs launched from the desktop (not terminals or editors)
keep_days = 14       # how long finished jobs' logs are kept
agent_files = ["~/.cursor/rules/dibs.mdc"]   # more instructions files for dibs setup
```

Run `dibs daemon restart` after changing it; running jobs keep running.

## Files

| | |
|---|---|
| `~/.config/dibs/config.toml` | the configuration above |
| `~/.local/state/dibs/dibs.db` | the queue and job history, with each job's environment; only you can read it |
| `~/.local/state/dibs/jobs/ID/output.log` | a job's output, unless it was given `-o FILE` |
| `~/.local/state/dibs/daemon.log` | what the scheduler did, size-capped |
| `$XDG_RUNTIME_DIR/dibs/dibs.sock` | the socket the `dibs` command talks to |
| `~/.config/systemd/user/dibs.service` | the service `dibs setup` installs |
| `~/.config/dibs/setup.json` | what `dibs setup` changed, so `--undo` can put it back |
| `~/.local/share/dibs-install/record` | what `install.sh` added, so `--uninstall` removes only that |

`dibs setup` also adds a marked block to each agent CLI's instructions file it finds
(`~/.claude/CLAUDE.md`, `~/.codex/AGENTS.md`, `~/.config/opencode/AGENTS.md`,
`~/.pi/agent/AGENTS.md`, `~/.gemini/GEMINI.md`, and the files in `agent_files`), plus a Claude Code
skill. `dibs setup --undo` puts every file back as it was, keeping edits you made since.

The full design, with the reasoning behind each choice, is in
[docs/superpowers/specs/2026-11-13-dibs-design.md](superpowers/specs/2026-11-13-dibs-design.md).
