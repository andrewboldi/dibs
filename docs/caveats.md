# Caveats

What dibs doesn't do, or does only partly. Each item says what happens and what you can do about it.

- **Linux only.** Core pinning, RAM caps and the guard rely on Linux (sched_setaffinity, systemd
  scopes, /proc); the installer stops on other systems. Without a systemd user session (most
  containers), jobs run without a RAM cap and `dibs info` says so.
- **One user, one machine.** Each user gets their own scheduler, and the schedulers don't know about
  each other. Two people sharing a GPU through dibs would each hand out all of it.
- **AMD support is untested on real hardware.** It was written against amd-smi's documented output
  for ROCm 6.2 to 7.2 and tested with a stand-in. NVIDIA is what dibs runs on day to day. If you have
  an AMD GPU, please report what `dibs info` and a GPU job do.
- **The guard has limits.**
  - A job that grabs GPU memory very fast can collide with another once before it's stopped.
  - CPU-heavy work outside dibs isn't stopped, since nothing reliably tells a heavy job from a
    build or a test run; when a terminal keeps a sixth of the cores busy for two minutes, its agent
    is told instead (#158). RAM-heavy work isn't stopped either while the machine has RAM to spare:
    dibs only stops handing out the RAM it uses. When memory runs short, the guard stops the largest
    program outside dibs, which can be a person's own program in a terminal; `guard_allow`
    exempts it by name. A swarm of small processes, none holding a tenth of the RAM, is left to
    the kernel.
  - An agent determined to get around it could stop the daemon. Agents' instructions tell them
    not to.
- **Going over the VRAM reservation into another job's memory stops the job, at once and without a
  grace margin.** A job may borrow VRAM that no job reserved and no other program uses; it is
  stopped only when it reaches memory another job holds. With `--retry`, dibs then runs it again with
  the advised size, as it does a job killed at its RAM limit or by a restart, up to twice and keeping
  its place in line, but from the start: the work the stopped run did is lost unless the job saves
  and resumes it. Without `--retry`, the agent resubmits. Reserve the peak plus some
  headroom. `dibs submit` shows what earlier runs of the same command used.
- **Estimates only order the queue.** Nothing stops a job for running past its estimate, so
  predictions are only as good as the estimates. An agent that underestimates on purpose would get
  ahead of honest ones. In practice agents overestimate: over the first two hours of real use, jobs
  took a median 0.45 of their estimates.
- **A session's first job can use priority for free.** The time a high-priority job saves is paid
  back by its session's next job, so a session that submits only one job never pays.
- **A busy desktop leaves jobs less RAM.** dibs doesn't hand out the RAM other programs use, so with
  a browser holding 12G, jobs share what's left. Close programs to give it back; jobs already
  running keep their reservations.
- **One process over the RAM limit ends the whole job.** When the kernel kills one of a job's
  processes for lack of RAM, dibs stops the rest, so a parent waiting on a killed worker can't hang
  for hours holding the job's reservation. A job built to carry on without a killed process (a sweep
  that skips a failed trial) fails too. Reserve for the job's peak across all its processes, or run
  such pieces as jobs of their own.
- **Only `--mem` is promised.** A job nearing its limit may grow past it into RAM no other job
  reserved, when there is some; with none, it spills into swap, up to half its `--mem`, and is killed
  past that. A job that spills runs slower whenever it touches what went to swap, and one that waits
  on swap most of the time is stopped. Size `--mem` for its real peak. A job that grew is the first
  the kernel kills if the machine runs out.
- **Files in a folder kept in RAM count as RAM.** What a job writes to /dev/shm, or to /tmp where
  /tmp is a tmpfs, counts against its RAM reservation, and so does shared memory its processes map,
  such as buffers pinned for GPU transfers. A job killed at its limit says when either was a big part
  of it, and which: only files move out with a TMPDIR on a disk. Each job's TMPDIR is a folder of its
  own that dibs deletes when the job ends; files a job writes elsewhere in such a folder keep holding
  RAM after it ends, and dibs counts them as in use outside dibs until someone deletes them. A job
  whose scratch is bigger than its RAM can point TMPDIR at a disk with room for it.
- **A laptop's sleep pauses the jobs, not the queue's clock.** Run times, overruns and predictions
  leave out the time the machine slept, but places in line are set by the wall clock: after a long
  sleep every queued job has waited its estimate, so newer jobs can't pass any of them. A job dibs
  was stopping when the machine went to sleep is killed as soon as it wakes.
- **RAM is planned by use, so the machine can run short.** dibs lets jobs start in RAM other jobs
  reserved but aren't using. If those jobs (or programs outside dibs) then grow, the machine runs
  short: dibs stops the newest job that started this way once memory pressure stays high while
  little RAM is available, or the kernel kills it first (it runs with a high OOM score), and its work
  so far is lost: with `--retry` dibs runs it again within RAM nobody reserved, and without it the
  job fails with a note to resubmit it.
  `borrow_ram = false` plans by reservations only.
- **Jobs' environments are stored.** To run a job with the submitter's virtualenv and settings,
  dibs keeps its environment, which can include API keys, in `~/.local/state/dibs/dibs.db`. Only
  you can read the file. Finished jobs are deleted after `keep_days` (14 days by default).
- **A `dibs` command from another install can't use the running scheduler.** When the scheduler
  runs another version of dibs and restarting it can't change that (the service runs the other
  install), the command says so instead of working. Use the installed `dibs`.
