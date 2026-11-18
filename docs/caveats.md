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
  - It only polices the GPU. CPU- and RAM-heavy work outside dibs isn't stopped, since nothing
    reliably tells a heavy job from a build or a test run. dibs only stops handing out the RAM that
    work uses.
  - An agent determined to get around it could stop the daemon. Agents' instructions tell them
    not to.
- **Going over the VRAM reservation stops the job, at once and without a grace margin**, even when
  no other job needed that memory. dibs then runs it again with the advised size, as it does a job
  killed at its RAM limit or by a restart, up to twice and keeping its place in line, but the work
  the stopped run did is lost unless the job saves and resumes it. Reserve the peak plus some
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
- **Files in a folder kept in RAM count as RAM.** What a job writes to /dev/shm, or to /tmp where
  /tmp is a tmpfs, counts against its RAM reservation; a job killed at its limit says so when such
  files were a big part of it. Files a job leaves there keep holding RAM after it ends, and dibs then
  counts them as in use outside dibs, until someone deletes them. Point big scratch files at disk
  (most tools follow TMPDIR).
- **A laptop's sleep pauses the jobs, not the queue's clock.** Run times, overruns and predictions
  leave out the time the machine slept, but places in line are set by the wall clock: after a long
  sleep every queued job has waited its estimate, so newer jobs can't pass any of them. A job dibs
  was stopping when the machine went to sleep is killed as soon as it wakes.
- **Jobs' environments are stored.** To run a job with the submitter's virtualenv and settings,
  dibs keeps its environment, which can include API keys, in `~/.local/state/dibs/dibs.db`. Only
  you can read the file. Finished jobs are deleted after `keep_days` (14 days by default).
- **A `dibs` command from another install can't use the running scheduler.** When the scheduler
  runs another version of dibs and restarting it can't change that (the service runs the other
  install), the command says so instead of working. Use the installed `dibs`.
