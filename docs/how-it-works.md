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
   fifty jobs interleaves with everyone else instead of going ahead of them all. A job lines up
   only behind its session's jobs that are still waiting: when one starts or is cancelled, the ones
   after it move up, so a busy session's line never runs ahead of the clock (#150).

Every queued job gets a place on a timeline: the earliest moment it fits for its whole estimated
run, around the running jobs and the places already given to the jobs ahead of it. A small job
whose place is "now" starts now, so gaps get filled without delaying anyone. A job past its
estimate is assumed to need another 10% of it (at least a minute), and nothing is ever stopped for
running long: the estimate only orders the queue. `dibs top` draws the timeline live. Its meters
show, for CPU, RAM and each GPU, what jobs reserved (in their colours), what programs outside dibs
use (grey) and what's free, and its load row the machine's busy cores over the last 10 minutes. RAM in use outside dibs is never handed out; CPU busy outside dibs is
only shown, since dibs still starts a job on cores another program keeps busy: the job's threads
then come first on them (#152).

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
| CPU cores | pinned to its cores, with first claim on them (its scope's `CPUWeight` is 1000, other programs' 100); `OMP_NUM_THREADS` and the like set to match | Linux (first claim: systemd user session) |
| RAM | its own systemd scope with `MemoryMax` and no swap; going over kills the job, not its neighbours, and when the kernel kills only one of its processes, dibs stops the rest within a second or two (with `--retry`, dibs runs it again with more) | systemd user session |
| VRAM | watched every second; a job may borrow VRAM nobody reserved, and one that reaches another job's is stopped, with the reason (with `--retry`, run again with more) | NVIDIA (NVML or nvidia-smi), AMD (amd-smi) |
| Which GPUs | `CUDA_VISIBLE_DEVICES` (NVIDIA) or `ROCR_VISIBLE_DEVICES` (AMD) shows a job only its GPUs, and none to a job without `--vram` | |
| GPU use outside dibs | the guard stops it within about a second, after writing into its output how to queue it | daemon running |
| RAM use outside dibs | when the machine runs short of memory, the guard stops the largest program outside dibs that holds a tenth of the RAM or more, after writing into its output why and how to run it as a job | daemon running |
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

**RAM by use.** Agents reserve a median 2-4 times the RAM their jobs use, and each job keeps its own
reservation as a hard cap. To let others use what sits idle, dibs plans with what a running job
uses: the larger of its peak so far and earlier runs' peak, plus 25% or 1G, never more than it
reserved. Any job may start in the RAM this frees (#148); if dibs has to stop it, a job submitted with
`--retry` runs again by itself and any other fails with a note to resubmit. A job with no earlier runs counts at its whole reservation for its first ten minutes,
and an exclusive job always does. A job that starts only thanks to this is marked as borrowing and
runs with a high OOM score. If memory pressure (PSI some avg10) stays above 40% for 10 seconds
while less than a fifth of the RAM is available, dibs stops a program itself rather than
wait for the kernel's OOM killer, which acts only after minutes of thrashing; if the kernel gets
there first, it kills a borrower rather than anything else. The program dibs stops is the largest
one outside dibs, if one holds a tenth of the RAM or more (see the guard below), and otherwise
the newest borrower; one per stretch of shortage. Either way dibs runs a borrower again
within RAM nobody reserved, outside its retry limit, since starting it there was dibs's choice.
Pressure with RAM to spare stops nothing: it comes from one job stalling inside itself, at its own
limit or while the kernel compacts memory for huge pages. No job starts this way while pressure is
above 10%. `borrow_ram = false` turns it off.

**Momentum.** For a project working better than expected and held up by another project's jobs,
the user can agree to a boost: the agent asks them in plain words and, after their yes, records it
with `dibs momentum --for 6h --why "…"`. dibs accepts it only while another project's jobs are in
the way, for one project at a time; the project's jobs then count as a quarter of their estimates,
with no payback, until it runs out. `dibs queue` shows it, and every grant is logged.

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

The guard also watches RAM, but only while the machine is short of it (the test above). Then it
stops the largest program outside dibs that holds a tenth of the RAM or more, writing into its
output what it held, why it was stopped and the `dibs run --mem` command that keeps it from
happening again. Left running, such a program stalls everything for minutes until the kernel's
OOM killer kills whatever is largest, a job or the program itself, and systemd may then stop the
terminal it ran in, the agent with it. It leaves alone what it leaves alone on the GPU, the agent
CLIs themselves, and any program an agent runs under (the tmux server, a pane's shell), since
stopping a program stops what it started; a job is never stopped this way, since it holds the RAM
it reserved. Only
programs where agents work qualify: in a tmux or VTE terminal pane (`tmux-spawn-*`, `vte-spawn-*`
scopes), or started by an agent CLI, as in an editor's terminal; snap apps, the desktop shell and
user services never do. A note in a stopped program's output can be lost (a pipe through `grep`, a
log nobody reads), so the agent's session also gets it, once, from its next `dibs` command; this
holds for GPU stops too.

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
guard = "all"        # "all" stops GPU processes outside dibs, and the largest program outside dibs
                     # when memory runs short; "warn" only lists them, "off"
guard_allow = ["ollama"]
guard_desktop = true # leave alone GPU programs launched from the desktop (not terminals or editors)
borrow_ram = true    # let jobs start in RAM other jobs reserved but aren't using
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
| `~/.config/systemd/user/{tmux,vte}-spawn-.scope.d/50-dibs.conf` | keeps a terminal pane, and the agent in it, running when the kernel kills a program there for lack of memory (`OOMPolicy=continue`) |
| `~/.config/dibs/setup.json` | what `dibs setup` changed, so `--undo` can put it back |
| `~/.local/share/dibs-install/record` | what `install.sh` added, so `--uninstall` removes only that |

`dibs setup` also adds a marked block to each agent CLI's instructions file it finds
(`~/.claude/CLAUDE.md`, `~/.codex/AGENTS.md`, `~/.config/opencode/AGENTS.md`,
`~/.pi/agent/AGENTS.md`, `~/.gemini/GEMINI.md`, and the files in `agent_files`), plus a Claude Code
skill. `dibs setup --undo` puts every file back as it was, keeping edits you made since.

The full design, with the reasoning behind each choice, is in
[docs/superpowers/specs/2026-11-13-dibs-design.md](superpowers/specs/2026-11-13-dibs-design.md).
