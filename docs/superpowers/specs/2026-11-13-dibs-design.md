# dibs: design

A Slurm-like job scheduler for one machine, built so that several Claude Code sessions (and the
human) can share one machine's GPUs, CPU cores and RAM without stepping on each other.

Status: approved approach (daemon + CLI). Date: 2026-11-13.

## 1. Why

The user runs many Claude Code sessions at once on one machine with one GPU. Each session sometimes
needs the GPU (training, inference, evals) or lots of CPU/RAM. Today they collide: two agents load
models at the same time and one or both die with CUDA out-of-memory, or they slow each other down.
The sessions can't coordinate because they don't know about each other.

dibs gives them one place to ask. An agent says up front what a job needs (cores, RAM, VRAM) and
how long it expects the job to take, and submits it. dibs decides the order, starts each job when
its resources are free, and keeps a queue every agent can read, including when each waiting job is
expected to start.

The user's example sets the bar: if job A and job B both need the whole machine and B takes 10x
longer, A should run first, so A's agent is unblocked quickly and B still runs right after.

Success looks like:

- Agents that declare honestly never crash each other with out-of-memory errors.
- Short jobs don't sit behind long ones, and long jobs still always get their turn.
- An agent can submit a job, let its shell command return, and later check on it, wait for it or
  read its output, even across the agent's 10-minute command timeout.
- It works on any Linux machine: any number of cores, any amount of RAM, zero or more NVIDIA GPUs.
- It's a public GitHub repo with a polished README and an animated demo, in the style of
  github.com/andrewboldi/tmux-restore.

## 2. Goals, non-goals, anti-goals

**Goals**

- Jobs reserve fixed amounts of CPU cores, RAM and VRAM (per GPU, on one or more GPUs). The
  scheduler hands out exactly that and holds jobs to it.
- Every job carries a time estimate. The scheduler uses it only to order the queue and predict
  start and end times.
- A background scheduler (daemon) that starts itself on first use, keeps the queue on disk, and
  survives its own restarts without losing or killing jobs.
- A CLI for agents and humans: submit, run, queue, show, wait, logs, cancel, top, info, plus
  daemon management. Every command that prints information also offers `--json`.
- Agents can't work around it, whichever agent CLI they run in (Claude Code, Codex, opencode, pi,
  Gemini CLI, and ones nobody has heard of yet): a guard stops any GPU process that isn't a dibs job
  and tells it how to use dibs, and `dibs setup` puts short instructions into every installed
  agent's global instructions file (section 13).
- Tests that pin down the scheduling behaviour, CI on GitHub Actions, and a README with a demo GIF.

**Non-goals (v1)**

- More than one machine.
- Preemption, suspending or checkpointing jobs.
- Killing or stopping a job for running past its time estimate. Never.
- Looking at live free VRAM/RAM from processes dibs didn't start. Capacity is the machine's
  detected totals (optionally narrowed by config), and dibs counts only its own reservations.
- Job dependencies, job arrays, user-set priorities.
- GPUs other than NVIDIA. Windows. Multiple Unix users sharing one dibs.
- Publishing to PyPI (the name `dibs` is taken there by an unrelated project).
- Guarding CPU and RAM use outside dibs. Nothing reliably tells a heavy job from an agent's normal
  builds and test suites, so agents are told to queue heavy CPU/RAM work but aren't stopped.
- An "agents only" guard mode that tries to recognise agent processes (backgrounded jobs lose
  their parent chain, and not every agent CLI marks its commands).
- Per-agent command hooks (they only see command text, not what a script does) and locking the
  GPU device files (airtight, but needs root and can break laptop graphics). Possible later.

**Anti-goals** (would make it a failure even if it technically works)

- An agent loses its job because its shell command timed out, it was interrupted, or its session
  closed. Only an explicit `dibs cancel` stops a job.
- A long job waits forever because shorter ones keep arriving.
- A job silently gets fewer resources than it reserved, or silently uses more and crashes a
  neighbour.
- Setup before first use: no root, no config file, no service install required. (`dibs setup`
  is what turns on enforcement; plain use works without it.)
- The scheduler itself noticeably loads the machine.
- The guard kills work that was already running when it was switched on, or kills an agent
  itself rather than the GPU process the agent started.

## 3. Constraints

- Python 3.11+ (3.10 reached end of life in October 2026; 3.11 has `tomllib`). Standard library
  only at runtime. Dev/test tooling (pytest, a build backend) is fine.
- Linux is the supported platform. macOS is best-effort: it should run, but there is no core
  pinning or RAM capping there.
- No root. Everything runs as the user.
- Zero GPUs must be a normal, fully working configuration.
- Claude Code's Bash tool times out commands after 2 minutes by default and 10 at most. Jobs must
  therefore run detached from the command that submitted them.

## 4. Approaches considered

1. **Daemon + CLI (picked by the user).** One background process owns the queue, decides who runs,
   launches jobs and enforces reservations. The CLI talks to it over a Unix socket and starts it
   if it isn't running. One decision-maker means no races. Cost: a long-running process to manage.
2. **No daemon.** Every CLI call locks a shared SQLite file and runs a scheduling pass; each job has
   its own detached waiter process. Nothing to keep running, but the decision is spread across many
   polling processes, which is much harder to get right and to observe.
3. **Wrap Slurm, pueue or task-spooler.** Slurm needs root and munge, can't hand out VRAM by
   amount, and treats `--time` as a kill limit (which the user explicitly doesn't want). pueue and
   task-spooler queue commands but know nothing about resources or durations.

## 5. Decisions

**Decided by the user**

- Jobs must disclose CPU cores, RAM and VRAM up front. Those amounts are fixed: the scheduler
  allocates exactly that and holds jobs to it. They are not estimates.
- The time estimate is the only estimate. It is for the scheduler only. Jobs are never killed for
  exceeding it.
- It must work on any machine with arbitrary resources, not just the user's. Don't design around
  what happens to be running on the user's machine.
- Approach 1: daemon + CLI.
- Agents must not be able to work around dibs, and this has to work for every agent CLI, not just
  Claude Code (Claude Code, opencode, pi, Codex, "etc").
- The guard stops every GPU process that isn't part of a dibs job, the user's own included, with a
  config allowlist for exceptions. CPU/RAM aren't guarded in v1.
- After reading the design and the guard proposal: "Defaults all sound great. GO". So the calls
  below, which started as mine, are now approved defaults.

**My calls, approved by the user as defaults**

- Name: `dibs`. Command `dibs`, repo `andrewboldi/dibs`.
- Ordering: shortest first, with a guarantee that long jobs get their turn (section 7).
- Session fairness: one session's queued jobs line up behind each other (section 7).
- Enforcement: cores pinned; RAM capped through systemd where available; VRAM watched and a job
  that exceeds its reservation is stopped (section 9).
- Python, no runtime dependencies, Linux first.
- Agents learn the tool from instructions `dibs setup` writes into every installed agent CLI's
  global instructions file (plus a Claude Code skill with the details); jobs are tagged with the
  submitting session.

Further calls of mine are marked "(my call)" where they appear; the user approved the defaults as a
whole.

## 6. Resources

### 6.1 What a machine has

At daemon start, dibs detects:

- **CPUs**: the set of logical CPUs this process may run on (`os.sched_getaffinity(0)`; falls back
  to `range(os.cpu_count())`). dibs hands out specific CPU ids from this set.
- **RAM**: `MemTotal` from `/proc/meminfo` (macOS: `sysctl hw.memsize`).
- **GPUs**: `nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader,nounits`.
  Each GPU is its own pool of VRAM (MiB). If `nvidia-smi` is missing or fails, the machine has
  zero GPUs.

Capacity is stored as named pools, never as "the GPU": `cpus` (a set of ids), `mem` (bytes), and
one VRAM pool per GPU (keyed by index, carrying uuid and name). Multi-GPU placement falls out of
this.

### 6.2 Config (optional)

`config.toml` (path in section 12) can narrow what dibs hands out. Every key is optional; without
the file dibs hands out everything it detected (my call: predictable, and the user said not to
design around other software on the machine; the README tells desktop users how to hold some back).

```toml
cpus = 20          # hand out this many CPUs (the lowest-numbered ones), default: all
mem  = "24G"       # hand out this much RAM, default: all
gpus = [0]         # manage only these GPU indices, default: all
vram = "7.5G"      # hand out at most this much VRAM per GPU, default: each GPU's total
keep_days = 14     # delete finished jobs' logs after this many days
guard = "all"      # "all": stop GPU processes outside dibs; "warn": only report them; "off"
guard_allow = []   # executable names the guard leaves alone, e.g. ["ollama", "obs"]
```

Values larger than what was detected are an error at daemon start (with a clear message). Config is
read at daemon start; `dibs daemon restart` applies changes (safe: jobs keep running, see 10.3).

If capacity shrinks below what a queued job needs, the job stays queued and `queue`/`show` say it
can't fit the current machine. Nothing is deleted.

### 6.3 What a job asks for

| Flag | Meaning | Required |
|---|---|---|
| `--cpus N` | logical CPUs, integer ≥ 1, or `all` | yes |
| `--mem SIZE` | RAM, e.g. `512M`, `16G`, or `all` | yes |
| `--vram SIZE` | VRAM on each GPU it uses, or `all` | no; omitted means no GPU |
| `--gpus N` | number of GPUs | no; 1 if `--vram` given, else 0. `--gpus` without `--vram` is an error |
| `--time DURATION` | estimated run time | yes |
| `--name NAME` | label | no; derived from the command |

- Sizes are binary: `G` = GiB, `M` = MiB. Accept `16G`, `16GB`, `16GiB`, `1.5G`, `512M`.
- Durations: `90s`, `20m`, `1h`, `1h30m`, `1.5h`, `2d`, `HH:MM:SS`, `MM:SS`; a bare number is
  minutes (Slurm's convention). Must be > 0.
- `all` resolves at submit time to the full capacity: all of dibs's CPUs, all of its RAM, or the
  VRAM of the largest GPU (so a `--vram all` job occupies a whole GPU of the largest size).
- A request that can never fit is rejected at submit with a message naming the resource, e.g.
  `asks for 12G VRAM, but the largest GPU dibs manages has 8G`. Also reject `--gpus` greater
  than the number of GPUs.
- Requiring `--cpus`, `--mem` and `--time` follows the user's "must disclose up front". `--vram`
  may be omitted because "no GPU" is itself a disclosure (my call).

## 7. Who goes next

### 7.1 Priority tag

Each job gets a tag when it is submitted, and the tag never changes:

```
tag(j) = max(submitted(j), L) + estimate(j)
L      = the largest tag among the same session's jobs that are queued (not running, not
         finished) at the moment j is submitted; if there are none, L = -infinity
```

Queue order is ascending tag, then submit time, then job id. Cancelling a job doesn't re-tag the
others (my call: simple and predictable).

What this gives:

- **Shortest first across sessions.** Jobs submitted at about the same time are ordered by
  estimate. The user's A/B example: tags are `s+1` and `s+10`, so A goes first.
- **Long jobs can't starve.** A job submitted after time `tag(J)` gets a tag greater than `tag(J)`,
  so once a job has waited as long as its own estimate (plus its session's earlier queued jobs),
  no newer job can be ordered ahead of it.
- **One session can't crowd out the others.** If a session queues 50 one-minute jobs at once,
  their tags are `s+1, s+2, ..., s+50`; another session's 10-minute job (tag `s+10`) goes after
  the first ten, not after all fifty. Because only *queued* jobs count toward `L`, a session isn't
  penalised for jobs that are already running or finished.
- **Within one session, jobs keep the order they were submitted in** when they compete for the
  same resources. (Trade-off, my call: if an agent queues a long job and then a quick one that
  needs the same GPU, the quick one waits behind it; the agent can cancel and resubmit to reorder.
  The alternatives that let a session's short jobs jump its own long ones either let that long job
  starve or let a session's tags reset, which breaks the fairness above.)

Considered and rejected: highest-response-ratio-next, `(wait + est) / est`. In the user's example,
if B has waited even a few seconds when A arrives, B's ratio exceeds A's 1.0 and B goes first. Plain
shortest-first starves long jobs.

### 7.2 Planning and filling gaps (conservative backfill)

On every scheduling pass, dibs builds a plan for every queued job, in priority order:

1. Start from the free resources right now (capacity minus running jobs' reservations), plus a
   timeline of when each running job is expected to release its resources.
2. Take the next queued job by priority. Find the earliest time `t ≥ now` at which it fits for its
   whole estimated duration, given the running jobs and the reservations already placed for
   higher-priority jobs. Candidate times are `now` and every time something is expected to end.
   "Fits" means: enough free CPUs, enough free RAM, and `gpus` distinct GPUs each with at least
   `vram` free, all for the entire interval `[t, t + estimate)`.
3. Reserve those resources for that interval (choose GPUs best-fit: the GPUs with the least
   sufficient free VRAM over the interval; ties go to the lowest index). Move to the next job.
4. Every job whose planned start is `now` is started now, in priority order.

So a lower-priority job only starts early if it fits in a gap without delaying any reservation
ahead of it. That is the promise made to the user: "a small job can start early in a gap only if
that doesn't delay anyone ahead of it".

The plan is recomputed from scratch on every pass, so it always reflects the current queue. A pass
runs on submit, cancel, job end, daemon start, and on a timer (every few seconds). The planner is a
pure, deterministic function:

```
plan(capacity, running_jobs, queued_jobs, now) -> (jobs_to_start_with_assignments,
                                                   predicted start/end per queued job,
                                                   wait reason per queued job)
```

This function is the heart of the project and must be testable with a fake clock and no I/O.

### 7.3 Jobs that run past their estimate

They are never stopped. For planning, a running job's expected end is `start + estimate` while
that's in the future. After that, it is assumed to need a little longer: `now + max(60 s, 10% of
its estimate)`, re-evaluated every pass (my call). Everyone's predicted start times shift as it
keeps running. `queue` and `show` mark it as "over estimate by …".

### 7.4 Predictions and reasons

- Each queued job shows its predicted start and end (from the plan) and its place in line.
- Each queued job also shows why it is waiting, in plain words, computed from the plan:
  - a resource it lacks right now, e.g. `needs 6G VRAM; most free on any GPU: 2G (held by #9
    until ~14:31)` or `needs 8 CPUs; 4 free`; or
  - that it fits now but would delay a job ahead of it, e.g. `would delay #7, which is ahead in
    line`; or
  - that it can't fit the current machine at all (6.2).
- Predictions are only as good as the estimates. The README and skill say so.

## 8. Running a job

### 8.1 What gets captured at submit

- The command as an argv list (`dibs submit ... -- cmd arg ...`). No shell is involved; for shell
  features the agent writes `-- bash -c '...'`.
- The working directory and the full environment of the submitting process (like `sbatch`'s
  default). This keeps virtualenvs, `PATH` and API keys working. The environment is stored on disk
  with private permissions (section 12).
- The session (section 11.2), name, resources, estimate, tag, submit time.

### 8.2 How it is started

When a job starts, dibs assigns concrete resources:

- **CPUs**: specific CPU ids among those not held by running jobs (lowest-numbered free ones;
  smarter placement is left to the builder).
- **GPUs**: the GPU indices chosen by the plan.

It then launches the job fully detached from both the daemon and the submitter (own session/process
group; under systemd, its own scope unit), so that neither a daemon restart nor the agent's shell
ending affects it. stdin is `/dev/null`; stdout and stderr go together into the job's
`output.log` (or the path given with `--output PATH`).

The job's environment is the captured one plus:

| Variable | Value |
|---|---|
| `CUDA_VISIBLE_DEVICES` | assigned GPU indices, comma-separated; empty string for jobs without GPUs, so they can't touch a GPU |
| `CUDA_DEVICE_ORDER` | `PCI_BUS_ID`, so indices match `nvidia-smi` |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` | the CPU count, unless the captured environment already sets them (prevents PyTorch/NumPy from oversubscribing the pinned cores) |
| `PYTHONUNBUFFERED` | `1` unless already set, so output reaches the log as it happens (verified: without it, a Python job's prints are lost if it's killed) |
| `DIBS_JOB_ID`, `DIBS_CPUS`, `DIBS_MEM`, `DIBS_VRAM`, `DIBS_GPUS` | the job's id and what it was granted (bytes for sizes), so job code can size itself, e.g. `torch.cuda.set_per_process_memory_fraction` |

A small launcher process ("shim") sits between dibs and the command: it applies affinity,
environment and working directory, starts the command, waits for it, and writes the exit status
(and any limit that was hit) atomically to the job's folder before exiting. The daemon learns that
a job finished by watching that file and the shim's pid, which works the same for jobs it started
and jobs it adopted after a restart (10.3).

When a job's command exits, any processes it left behind (daemonised children, stray workers) are
stopped, as Slurm does: they would otherwise keep holding resources the job no longer reserves.

### 8.3 States and exit codes

States: `queued` → `running` → `done` (exit code 0) or `failed` (non-zero exit, killed by a
signal, stopped for exceeding a reservation, or couldn't be started); `queued`/`running` →
`cancelled`. Terminal jobs keep their exit code and a one-line reason when dibs has one, e.g.
`stopped: used 7.1G VRAM on GPU 0, reserved 6G` or `killed: went over its 16G RAM reservation`.

Cancel and over-limit stops send SIGTERM to every process of the job, then SIGKILL after 10
seconds.

## 9. Holding jobs to their reservations

This is what "fixed by the scheduler" means in practice, and the README must state its limits
plainly.

- **CPU cores (enforced on Linux).** The job's processes are pinned to its CPU ids with
  `sched_setaffinity`; children inherit it. The thread-count variables above keep libraries in
  line. (A process could deliberately re-pin itself; agents aren't adversaries.) macOS: not
  enforced.
- **RAM (enforced where systemd allows it, without root).** When `systemd-run --user` works, the
  job runs in its own scope with `MemoryMax=<reserved>` and `MemorySwapMax=0`; going over gets the
  job OOM-killed by the kernel and nobody else is affected. Verified on the user's machine: a 200M
  scope allocating 400M exits with 137. dibs must report this as a RAM-limit kill. Verified
  mechanism: run the shim inside the scope and pass `-p OOMPolicy=continue` (without it, systemd
  stops the whole scope after the OOM kill and the shim dies with SIGTERM before it can report);
  after the command exits, the shim reads its own cgroup's `memory.events` (`oom_kill 1`) and
  `memory.peak`. Never use `RLIMIT_AS`: CUDA reserves huge virtual address ranges and breaks under
  it. Without systemd (containers, macOS, some CI): RAM is not enforced, and `dibs info` says so.
- **VRAM (watched, not capped).** Consumer NVIDIA GPUs have no per-process VRAM limit. dibs polls
  the GPUs' compute processes about once a second (section 10.4 on how), maps pids to jobs (via the
  job's scope cgroup, or `DIBS_JOB_ID` in `/proc/<pid>/environ`), and sums usage per job per GPU.
  The same poll feeds the guard (section 13.2). A job using
  more than its reservation on any GPU is stopped with a reason (8.3). This can't always prevent a
  neighbour's OOM between polls, but it stops the over-user and tells its agent why. If the driver
  doesn't report per-process memory, VRAM isn't watched and `dibs info` says so. Agents are told to
  include CUDA context overhead (~0.3–0.5G) in `--vram`.
- **GPU visibility (enforced by convention).** `CUDA_VISIBLE_DEVICES` limits which GPUs a job sees.
- **Time (never enforced).**

dibs also records each job's peak VRAM (from the polls) and peak RAM (from the cgroup's
`memory.peak` when available) and shows them in `show` and in finished jobs' listings, so agents can
size the next request from real numbers.

## 10. The daemon

### 10.1 Lifecycle

- Any CLI command that needs the daemon connects to the socket. If nothing answers, it starts the
  daemon and retries for up to ~5 seconds.
- The daemon must be fully detached from whoever started it (new session; when systemd is
  available, start it as a transient user unit with `systemd-run --user`), so it survives the
  agent's command, its tmux pane, and its Claude session ending.
- Exactly one daemon per user (per `DIBS_HOME`): it holds an exclusive `flock` on a lock file for
  its whole life. A second one exits quietly. A stale socket is removed under that lock.
- `dibs daemon start|stop|restart|status` and `dibs daemon run` (foreground, for systemd).
- `dibs setup` installs a systemd user unit (`Restart=always`) and enables it, and turns on
  lingering (`loginctl enable-linger`) so the daemon, and with it the guard, runs from boot even
  when nobody is logged in and agents keep running in tmux. This matters for enforcement: an agent
  that bypasses dibs never starts the daemon itself. `dibs setup --undo` turns lingering back off
  only if setup turned it on (as tmux-restore's installer does). Without systemd, setup says how
  to start the daemon at login instead.
- Stopping or restarting the daemon never stops jobs.

### 10.2 Talking to it

- Unix socket, newline-delimited JSON requests and responses. Each request carries the client's
  version. If the daemon's version differs (the user upgraded dibs), the CLI restarts the daemon
  (safe, 10.3) and retries once.
- `wait` is a long-poll: the daemon answers when the job reaches a terminal state or the client's
  timeout passes. If the daemon restarts mid-wait, the client reconnects and keeps waiting.
- `logs` reads the log file directly (same user, same machine).
- The daemon is the only writer of state; the CLI never opens the database.

### 10.3 State and recovery

- SQLite (WAL mode) holds jobs and per-session bookkeeping. Every state change is written before
  it's acted on.
- On start, for each job marked running: if its exit file exists, finish it; else if its shim pid
  is alive (and the process start time matches the stored one, to guard against pid reuse), adopt
  it and keep watching; else mark it `failed` with `lost: its process ended while dibs was not
  running`. Queued jobs stay queued.

### 10.4 Load

The daemon wakes about once a second. Because of the guard, it polls the GPUs' compute processes
on every wake, even when no dibs job is running, so the poll must be cheap. Measured on the user's
machine: running `nvidia-smi --query-compute-apps=...` costs about 20 ms of CPU per call (2% of a
core at 1 Hz), while calling NVML directly through `ctypes` (`libnvidia-ml.so.1`,
`nvmlDeviceGetComputeRunningProcesses_v3`) costs 0.18 ms. So: NVML via `ctypes` (still no
dependency), falling back to `nvidia-smi` if the library can't be loaded. Tests force the
`nvidia-smi` backend with a fake binary on `PATH`. Detection at start (6.1) may use either. The
daemon should be invisible in `top`.

## 11. The CLI

### 11.1 Commands

```
dibs submit [resources] [--name N] [--output PATH] -- cmd ...   queue a job, print its id and prediction
dibs run    [same as submit] -- cmd ...       submit, stream its output, exit with its exit code
dibs queue  [--all] [--mine] [--json]         running and queued jobs (--all adds recently finished)
dibs show   ID [--json]                       everything about one job (alias: status)
dibs wait   ID... [--timeout DURATION]        block until the jobs finish
dibs logs   ID [-f] [-n LINES]                print or follow a job's output
dibs cancel ID...                             cancel queued or running jobs
dibs top                                      live full-screen view (for humans and the demo)
dibs info   [--json]                          capacity, what's enforced, guard state, paths, daemon status
dibs daemon start|stop|restart|status|run
dibs setup  [--dry-run] [--undo]              turn on enforcement: service + agent instructions (section 13)
dibs agent-help                               print the full agent instructions
```

`queue` and `top` also show the guard's recent stops and any GPU processes running outside dibs
that it left alone (allowlisted, another user's, or already running when the guard was switched
on), with their VRAM.

- `run` and `wait` exit with the job's exit code; a cancelled job gives 130; a `wait --timeout`
  that runs out gives 124 and says the job is still going and how to wait again.
- Interrupting `run` or `wait` (Ctrl-C, a killed shell, a tool timeout) never cancels the job. It
  prints the job id and the commands to re-attach or cancel (my call: one rule for humans and
  agents).
- Human output is compact, coloured only on a TTY. `--json` gives stable, documented fields.
- Exit code 2 for usage errors and rejected requests.

Example of the intended feel (exact layout left to the builder):

```
$ dibs submit --cpus 4 --mem 16G --vram 6G --time 20m --name train -- python train.py
Job 12 (train) queued: 2nd in line, expected to start ~14:32 (in 8m).
  dibs wait 12      dibs logs 12 -f

$ dibs queue
 JOB STATE    SESSION         NAME   CPU RAM  VRAM   EST  WHEN
   9 running  chess-bot·6098  train    8 16G  6G@0   30m  ends ~14:31 (in 7m)
  11 running  api·a1b2        lint     2  2G  -       2m  over estimate by 1m
  12 queued   web·77f0        eval     4  8G  6G      5m  starts ~14:31 (in 7m): needs 6G VRAM, 2G free
  10 queued   chess-bot·6098  sweep    8 16G  6G      2h  starts ~14:36 (in 12m)
```

`dibs top` shows capacity bars (CPUs, RAM, each GPU's VRAM), running jobs, and the queue as a
timeline of the plan, refreshing every second.

### 11.2 Sessions

- A job's session is `CLAUDE_CODE_SESSION_ID` when set (Claude Code exports it to every command),
  else `DIBS_SESSION` if set, else the submitting terminal's session id (`os.getsid`).
- Display label: the project folder's name (git root of the submit directory, or the directory)
  plus the first four characters of the session id, e.g. `chess-bot·6098`.
- `--mine` filters to the current session.
- Jobs are never cancelled because their session ended. (The user's tmux-restore brings sessions
  back after reboots with the same session id and a new pid, and a waiting job may still be wanted.)

## 12. Files

Default locations (XDG):

- Config: `~/.config/dibs/config.toml`
- State: `~/.local/state/dibs/`: `dibs.db`, `daemon.log`, `daemon.lock`, `jobs/<id>/output.log`,
  `jobs/<id>/exit`
- Socket: `$XDG_RUNTIME_DIR/dibs/dibs.sock`, falling back to the state folder

If `DIBS_HOME` is set, everything (config, state, socket) lives under it instead. Tests and the demo
rely on this to run isolated from the user's real dibs.

Folders are `0700` and the database `0600`, because it stores submitters' environments, which can
hold API keys. The README mentions this.

Finished jobs' folders are deleted after `keep_days` (default 14, my call). Agents that need output
for longer use `--output` to put the log in their project.

## 13. Making agents use dibs

Two layers. Instructions tell agents what to do; the guard makes going around dibs not work. The
guard is what makes it work for any agent CLI, including ones that never read the instructions.

### 13.1 Instructions (`dibs setup`)

`dibs setup` writes a short block, between `<!-- dibs:begin -->` and `<!-- dibs:end -->` markers,
into the global instructions file of every agent CLI that is installed (its config folder exists
or its binary is on `PATH`), creating the file if needed. Locations, verified against the installed
packages on the user's machine on 2026-11-14:

| Agent | Global instructions file |
|---|---|
| Claude Code | `~/.claude/CLAUDE.md` (always loaded, unlike a skill) |
| Codex | `~/.codex/AGENTS.md` (`$CODEX_HOME/AGENTS.md`) |
| opencode | `~/.config/opencode/AGENTS.md` |
| pi | `~/.pi/agent/AGENTS.md` |
| Gemini CLI | `~/.gemini/GEMINI.md` |

- Also writes a Claude Code skill, `~/.claude/skills/dibs/SKILL.md`, carrying the full
  instructions, so the always-loaded block can stay short.
- Running setup again updates the blocks in place (it never duplicates them). `--dry-run` prints
  what it would change. `--undo` removes every block and file it added and reverses the service
  and lingering changes it made (10.1). Setup records what it changed so undo is exact.
- `dibs agent-help` prints the full instructions; the README tells users of other agent CLIs to
  paste the block into theirs.
- Tests and the demo never touch the real files: every path is resolved under `$HOME`, which tests
  point at a temporary folder.

The block (wording left to the builder, keep it about this short):

```
## Heavy jobs and the GPU: use dibs
This machine's GPUs, and its CPU cores and RAM for heavy work, are shared with other agents
through `dibs`, a job queue. Anything that uses a GPU must run through dibs: GPU processes
started any other way are stopped within seconds. Also queue CPU- or RAM-heavy work that runs
for more than a minute or two.
  dibs submit --cpus 4 --mem 16G --vram 6G --time 20m -- python train.py
Then `dibs wait ID` (in the background if your commands time out), `dibs logs ID`, `dibs queue`.
`dibs agent-help` explains how to size a request.
```

The full instructions (skill and `agent-help`) cover:
- run GPU work only through dibs (anything else is stopped), and queue heavy CPU/RAM work too;
- how to size a request: VRAM = weights + activations/optimizer state + ~0.5G CUDA context, rounded
  up; RAM = peak including data-loader workers; CPUs = threads/workers actually used; and to look
  at the peaks of earlier runs in `dibs show` / `dibs queue --all`;
- give an honest `--time`: it only affects ordering, nothing is killed for running long;
- submit, then `dibs wait ID` in the background (Claude Code: `run_in_background`), or check
  `dibs queue` / `dibs show ID`; read output with `dibs logs ID`;
- cancel jobs that are no longer needed;
- a job stopped for exceeding a reservation should be resubmitted with more;
- don't stop the daemon or change its config to get around it.

### 13.2 The guard

On every poll (about once a second, 10.4) the daemon looks at every compute process on every GPU
it manages:

- **Part of a running dibs job** (in the job's scope cgroup, or `DIBS_JOB_ID` in its environment
  naming a running job): fine; its usage counts toward that job's VRAM (section 9).
- **Not part of one**, and owned by this user, not allowlisted, and started after the guard was
  switched on: **stopped**.
- **Not part of one, but left alone** and reported: another user's process (dibs can't stop it);
  allowlisted (`guard_allow` matches its executable name); or already running when the guard was
  first switched on. "Switched on" is a timestamp stored in the database when the daemon first runs
  with `guard = "all"`, kept across daemon restarts, so turning on enforcement never kills work
  that was already running, while processes started during a daemon restart are still caught.

Graphics-only processes (the desktop, browsers drawing on the GPU) don't appear in the compute
process list (seen on the user's machine: `gnome-shell` holds the GPU but isn't listed), so they're
never touched.

**Stopping** (verified on the user's machine with pipes and with `nohup ... > log 2>&1 &`):

1. Target every unmanaged GPU process being stopped plus all of its descendants. Never its
   ancestors: the agent itself, and the shell it ran the command in, are left alone (a launcher
   like `torchrun` that doesn't hold the GPU itself sees its workers die and exits on its own).
2. Open each distinct destination among their stdout and stderr (`/proc/<pid>/fd/1` and `/fd/2`
   with `O_WRONLY|O_APPEND|O_NONBLOCK|O_NOCTTY`; one per device and inode; skip ones that can't be
   opened, such as sockets, and `/dev/null`).
3. Write the message to the destinations that are terminals, so it reads before the shell's next
   prompt.
4. SIGKILL them all, looking again for children forked at the last moment.
5. Write the message to the other destinations (log files, pipes). The processes are dead, so
   none can write over it; with `O_APPEND` it lands at the end of a log file even though the
   process had its own offset; a reader of a pipe gets it before end-of-file.
6. Record the stop (time, pid, command, GPU, VRAM, the session marker if the environment has one)
   so `queue` and `top` can show it.

(An earlier version froze the processes with SIGSTOP before writing. The demo showed why not: an
interactive shell reports a stopped foreground job as "Stopped" and returns to its prompt before
the message. Covered by `test_stopping_a_job_in_an_interactive_shell_reads_cleanly`.)

The message, filled from what the guard saw (`/proc/<pid>/cmdline`, VRAM rounded up, RAM rounded
up from the process's resident memory). CPUs and time can't be observed, so they stay for the
agent to fill in rather than being invented (my call):

```
dibs: stopped `python train.py --epochs 3` (pid 4121): it used 3.1G on GPU 0 outside dibs.
dibs: GPUs on this machine are shared through dibs. Run it as a job instead:
dibs:   dibs run --cpus N --mem 6G --vram 4G --time ESTIMATE -- python train.py --epochs 3
dibs: `dibs agent-help` explains how to choose N and ESTIMATE.
```

`guard = "warn"` records and shows these processes without stopping them; `guard = "off"` ignores
them. Limits, stated plainly in the README:
- A process is caught within about a second of creating its GPU context, so a job that allocates
  quickly can still collide once before it's stopped.
- If the driver doesn't report compute processes (some containers), the guard can't see them, and
  `dibs info` says so.
- It only covers GPUs. CPU and RAM use outside dibs isn't stopped (section 2).
- A determined agent could stop the daemon or edit the config. The instructions forbid it; agents
  aren't adversaries.

## 14. Testing

- **Planner unit tests** with a fake clock, no I/O. Must include:
  - the user's A/B example, including B submitted a few seconds before A;
  - a long job under a steady stream of short jobs from other sessions still starts, no later
    than the bound in 7.1;
  - a session flooding the queue interleaves with other sessions as in 7.1;
  - backfill never delays a higher-priority job's planned start;
  - multi-GPU requests, mixed VRAM sizes, best-fit placement, zero-GPU machines;
  - overrunning jobs: not stopped, predictions shift;
  - rejection of requests that can never fit; `all` values.
- **Simulation tests**: a discrete-event simulator drives the real planner with random job streams
  (actual durations differing from estimates) and checks invariants: no resource is ever
  over-committed, every job eventually starts, and no job starts in a way that delays a
  higher-priority job's planned start.
- **Integration tests** run a real daemon under a temporary `DIBS_HOME`, with a fake `nvidia-smi`
  on `PATH` that emulates any GPU layout and reports per-process VRAM from a file the test
  controls. Jobs are short shell/Python commands. Cover: ordering, environment variables, CPU
  affinity, logs, `wait` exit codes and timeout, cancel, VRAM over-use stop, daemon restart with a
  running job (adopted, finishes normally), auto-start, two CLIs auto-starting at once.
- **RAM cap test** only when `systemd-run --user` works; skipped otherwise.
- **Guard tests**, using a fake GPU backend that reports chosen pids as compute processes: an
  unmanaged process (and its children) is stopped and the message reaches both a pipe and a log
  file exactly once; its parent shell survives; a dibs job's processes are untouched; allowlisted,
  other-user and pre-existing processes are reported, not stopped; `warn` and `off` modes; leftovers
  of a finished job are stopped.
- **Setup tests** with `HOME` pointed at a temporary folder: blocks are written only for installed
  agents, re-running doesn't duplicate, existing file content is preserved, `--undo` restores the
  files exactly; the service part is skipped where systemd isn't available.
- **CI**: GitHub Actions, ubuntu-latest, Python 3.11 through 3.14.

## 15. Repo, README and demo

- Public repo `andrewboldi/dibs`. The user asked for regular commits and pushes: commit and push
  after each completed piece of the plan. Commit messages follow tmux-restore's style (plain
  sentences, no prefixes).
- Install: `uv tool install git+https://github.com/andrewboldi/dibs` (or `pipx install git+...`),
  then `dibs setup` to turn on enforcement.
- README modelled on tmux-restore's: centred logo (an SVG drawn for dibs), title, one-line pitch,
  badges (tests, license, Python), short nav, a two-sentence intro, the demo GIF, then: what it
  does, install, quick start (for agents and for humans), how it decides who goes next (with the
  A/B example), what's enforced and what isn't, commands, config, Claude Code skill, caveats,
  license.
- Demo GIF recorded with asciinema and converted with agg (same tools as tmux-restore), from a
  scripted driver in `demo/`, isolated with `DIBS_HOME` and a fake `nvidia-smi` (one 8G GPU) so
  anyone can re-record it. Seconds-scale jobs. The story:
  1. A training job holds the GPU.
  2. Agent B queues a long job that needs the whole GPU; it's predicted to start when training
     ends.
  3. Agent A queues a short job that also needs the whole GPU; it goes ahead of B, and B's
     predicted start moves back.
  4. Agent C queues a CPU-only job; it starts immediately in the gap.
  5. Training ends, A runs and finishes, then B runs.
  6. An agent runs `python train.py` directly; within a second it's stopped and the dibs message
     appears in its output.
  `dibs top` (or `queue`) is visible throughout so the reordering and predictions can be seen.

- During development, nothing is installed into the user's real setup: no `dibs setup` against
  their real `$HOME`, and no daemon that can see the real GPUs with the guard on (a temporary
  `DIBS_HOME` doesn't isolate the GPU: the guard would stop GPU processes the user's other sessions
  are running). Tests and the demo use the fake GPU backend, or `guard = "off"`/`"warn"` when they
  must touch the real GPU. Installing it for real is offered to the user at the end.

## 16. Left to the builder

- Module layout, internal APIs, the JSON schema's exact field names (documented once chosen).
- Table layouts, colours, and the design of `dibs top`.
- Polling intervals (within the spirit of 10.4) and the shim's exact mechanics.
- CPU placement beyond "lowest free ids" (e.g. keeping hyperthread siblings together).
- Packaging details (build backend, `pyproject.toml`).
