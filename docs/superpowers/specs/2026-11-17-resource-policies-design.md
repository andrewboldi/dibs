# Resource policies, round 2: design notes

The maintainer picked these from the hourly usage reviews (2026-11-17). Each lands as its own
issue and pull request, test first, and is checked on a live queue after it ships. They are listed
in build order: small and certain first, and prerequisites before what depends on them.

## 1. Count a GPU at its nominal size

An "8 GB" card reports 8188 MiB, so a 6G and a 2G job can't run together and `--vram 8G` is
refused. When dibs builds the capacity it plans with, a size just under a whole GiB (within 0.5%)
counts as that GiB. Sizes further off stay as reported (7000M stays 7000M), and the GPU backend
that the VRAM watch and the guard read is unchanged. The driver already keeps a few hundred MiB
for itself, so the extra 4 MiB changes nothing a job can allocate.

## 2. Replace a job in one step

`dibs submit --replace ID` (and `dibs run --replace ID`) cancels job ID and queues the new job
in one daemon request, so no other job can start in the gap between a cancel and a resubmit. The
new job is validated before anything is cancelled. Its place in line is computed as for any new
submission (now plus its estimate), after the replaced job has left the session's backlog, so a
replacement gains nothing a cancel and a resubmit wouldn't. ID may be queued or running; an
unknown ID is refused.

## 3. Leave desktop programs alone

The guard stops every GPU process of the user that isn't a job. It was meant for agents, and it
also stopped a program the user launched from the desktop. The guard now leaves alone a process
whose cgroup is a desktop launcher's app scope (`app-<launcher>-<app>-<n>.scope`, as GNOME and
other desktops name them), unless that app is a terminal emulator. It matches the cgroup, not the
parent processes, because a cgroup survives `nohup` and `setsid`: an agent's detached process
stays in its terminal's scope and is still stopped. A config setting switches this off.

## 4. Retry jobs that failed for resource reasons

Agents shouldn't spend effort working out why a job didn't run; dibs should handle it. When a job
is stopped for going over its VRAM reservation, killed at its RAM limit, or lost to a reboot, dibs
queues the same job again, with the same id and its place in line, instead of marking it failed:

- VRAM: at the size the stop message advises.
- RAM: with more RAM (the peak plus 25%, rounded up to a gigabyte), when that fits the machine.
- Reboot: the same request.

The retry happens in the same step that would have marked the job failed, so `dibs wait` never
sees a failure between attempts and keeps waiting. A job is retried at most twice. It is never
retried after a cancel or a normal exit, and never when the bigger request can't fit. The job's
output gets a line saying what happened and what dibs changed. `--no-retry` opts a job out (for
commands that must not run twice). `dibs show` and the earlier-runs history list the attempts.

## 5. Priority: an agent's own queue, and the loopholes

Two changes, one design, kept deterministic:

- **Reordering one's own jobs is free.** A new job with a higher priority than some of its
  session's queued jobs goes ahead of them; they move back by its estimate, so the session's total
  is unchanged and nobody else is affected. That part carries no payback.
- **Payback and credit settle against real waiting.** A job cancelled before it starts hands back
  what it carried and voids what it left. Debt a job carries is paid only by time it actually
  waited; whatever its wait didn't cover passes to the session's next job. Credit for a lowered
  priority counts only as far as the job really waited. This closes both loopholes the reviews
  found: a throwaway job that absorbs a debt and starts at once, and a throwaway low-priority job
  cancelled for free credit.

## 6. A temp folder per job

Each job gets its own folder under the system temp directory, created before it starts and
deleted when it ends, and TMPDIR, TMP and TEMP point at it unless the agent set TMPDIR itself. A
killed job's leftover scratch files no longer hold RAM on machines whose /tmp is a tmpfs. Leftover
folders are swept when the daemon recovers. When TORCHINDUCTOR_CACHE_DIR isn't set, it is pointed
at the default it would have had, so compiled kernels aren't thrown away with each job.

## 7. Borrow unreserved GPU memory

A job that goes over its VRAM reservation into memory no other job has reserved keeps running. It
is stopped only when its use exceeds the GPU's capacity minus the other jobs' reservations. While
it borrows, the planner counts it at its actual use, so no job starts into the borrowed memory.
Nothing borrows beside an exclusive job (section 12).

## 8. Why agents over-reserve: investigation

Measure requests against use per session and per command over the whole history: how often
requests repeat round numbers or agent-help's example, whether agents shrink requests after
seeing earlier runs, and how first runs differ from repeats. The findings shape agent-help's
sizing advice and item 9. One cause is structural: a RAM kill costs the agent its whole run, so
padding is rational; with item 4 the cost of a tight request is a retry, and agent-help can say so.
Deliberate isolation for timing (section 12) is counted apart from padding.

## 9. Plan RAM by measured use

A running job counts in the plan at the larger of its peak so far (which only grows) and the peak
of earlier runs of the same command, plus headroom (25% or 1G, whichever is more), and never more
than its reservation. A job with no history counts at its full reservation for its first ten
minutes. Each job's hard cap stays its reservation. Jobs started on RAM borrowed this way get a
higher OOM score, so if several jobs grow at once and the machine runs out, the kernel kills a
borrower first; dibs then retries it (item 4) at its full reservation, without borrowing.
Borrowing stops while memory pressure (PSI) is high. A machine-wide kill is told apart from the
job hitting its own limit by the scope's memory.events counters. No job borrows RAM beside an
exclusive job (section 12), whose reservation is idle on purpose.

## 10. Momentum

For exceptional cases: a project working better than expected, blocked because another project
holds the resources it needs, with ideas for more experiments. The agent asks the user in its
own conversation, in plain language about the project. Only after the user's yes does it
record a grant with the reason and a duration. dibs accepts a grant only while that session is
waiting on resources another session holds, allows one grant at a time, shows it in
`dibs queue`, and expires it. While it lasts, the session's jobs count as a quarter of their
estimates, with no payback; a job that has already waited its estimate still can't be passed.
How the user's yes is recorded is the maintainer's call (agent records it after the yes, or the
user runs the command).

## 11. Estimate feedback

Overestimating costs only the agent that does it (a later place in line, missed backfills), so
estimates are not rescaled. At most, `dibs submit` shows the session how its recent jobs' run
times compared with their estimates.

## 12. A job with the machine to itself

Added at the maintainer's request: some performance work times a job and reserves the whole
machine so that nothing else disturbs the measurement. `--exclusive` says so. The job waits until
no other job runs, and while it runs no job starts. In the plan it holds every CPU and all the
VRAM, but it is pinned to and sees only what it asked for (its own --cpus and --vram), and its RAM
is its own request: all the RAM can never be free while other programs use some. A job that can
finish before it starts may still go first, as with any job. Borrowing (sections 7 and 9) never
puts a job beside it, and the over-reservation investigation (section 8) counts it apart from
padding. A follow-up has `dibs show` say how busy the machine was outside the job while it ran,
so an agent can tell whether its timing was disturbed.
