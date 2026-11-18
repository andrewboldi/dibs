"""What agents are told about dibs: a short block for their always-loaded instructions file, and
the full instructions (`dibs agent-help`, and the Claude Code skill)."""

BEGIN = "<!-- dibs:begin -->"
END = "<!-- dibs:end -->"

BLOCK = """\
## Heavy jobs and the GPU: use dibs

This machine's GPUs, and its CPU cores and RAM for heavy work, are shared with other agents
through `dibs`, a job queue. Anything that uses a GPU must run through dibs: GPU processes
started any other way are stopped within seconds. Also queue CPU- or RAM-heavy work that runs
for more than a minute or two.

    dibs submit --cpus 4 --mem 16G --vram 6G --time 20m -- python train.py

Then `dibs wait ID` (in the background if your commands time out), `dibs logs ID`, `dibs queue`.
Run `dibs agent-help` before your first job: it explains how to size a request."""

AGENT_HELP = """\
# dibs: share this machine's GPUs, CPUs and RAM with the other agents

This machine runs several coding agents at once. Its GPUs, CPU cores and RAM are shared through
dibs, a job queue: you say what a job needs and how long you expect it to take, and dibs runs it
when those resources are free, so jobs never crash each other by running out of memory.

## When to use it

- Anything that uses a GPU: training, fine-tuning, inference, evals, benchmarks, tests that touch
  CUDA, a model server. A GPU process started outside dibs is stopped within seconds, with a
  message saying how to resubmit it.
- Work that keeps several CPU cores or a lot of RAM busy for more than a minute or two: big builds,
  data processing, CPU training, long test suites.
- Not needed for quick commands: git, editing, small scripts, light unit tests.

## Submitting

    dibs submit --cpus 4 --mem 16G --vram 6G --time 20m -- python train.py --epochs 3

- `--cpus N`: CPU cores. The job is pinned to exactly these, and OMP_NUM_THREADS and friends are
  set to N.
- `--mem SIZE`: RAM, like 16G or 512M. Where the system allows it, a job that goes over is killed.
- `--vram SIZE`: GPU memory on each GPU it uses. Leave it out for a CPU-only job, which then
  can't see any GPU. A job that uses more VRAM than it reserved is stopped.
- `--gpus N`: more than one GPU (each with --vram).
- `--time DURATION`: your honest estimate, like 90s, 20m or 1h30m. It only decides the order:
  shorter jobs go first, and once a job's place in line comes due (at the default priority, after
  waiting its estimate) newer jobs can't pass it. Nothing is ever stopped for running longer than its estimate.
- `--priority N`: 1 to 10, default 5. A higher priority makes the job count as shorter in line, so
  it goes ahead sooner; your session's next job pays that time back. Keep 5 unless a person is
  waiting on this job right now (then 8-10), or nobody is waiting at all (then lower, which earns
  your next job a head start).
- `all` works for --cpus, --mem and --vram when a job needs the whole machine or a whole GPU.
- `--name LABEL` to label it, `-o FILE` to write its output somewhere you choose.
- `--replace ID` to change a job that's still waiting (a smaller request, a fixed command): dibs
  cancels ID and queues the new job in one step. Cancelling first and then submitting can let
  another job start in the gap and keep the resources.
- The command goes after `--`, exactly as you would run it. It runs in your current folder with
  your current environment, so virtualenvs work. For shell syntax, use `-- bash -c '...'`.

## Sizing a request

- VRAM: model weights + activations + optimizer state + about 0.5G of CUDA context per process,
  rounded up. If unsure, start with a small run and look at its peak.
- RAM: the job's peak, including every data-loader worker, and the files it writes to a folder
  kept in RAM: /dev/shm, and /tmp where it is a tmpfs (`findmnt /tmp` says). Those files stay in
  RAM until deleted, even after the job ends: put big scratch files on disk (most tools follow
  TMPDIR), and delete what a job leaves behind.
- CPUs: the threads or worker processes it actually uses.
- `dibs show ID` lists a job's peak VRAM and RAM, and `dibs submit` shows what earlier runs of the
  same command in the same folder used. Size the next run from those: the peaks plus a little
  headroom (going over VRAM stops the job), and the CPUs it kept busy.
- If a job is stopped for going over its VRAM reservation, killed at its RAM limit, or lost when
  the machine restarts, dibs runs it again by itself, up to twice, with more VRAM or RAM when that
  was the cause, keeping its id and its place in line. `dibs wait` keeps waiting through that, and
  `dibs show ID` lists the tries. So ask for what the job really needs: a request that turns out too
  small costs a retry, not your work. Add `--no-retry` for a command that must not run twice.
- When it has run out of retries, the reason ends with the --vram to resubmit with: use that, not
  the same request again.

## Waiting and checking

- `dibs wait ID` blocks until the job ends and exits with the job's exit code.
- If your shell commands time out (Claude Code stops them after 2 to 10 minutes), run
  `dibs wait ID` in the background (in Claude Code, with run_in_background) and you'll hear when
  it's done. Or use `dibs wait ID --timeout 9m` and call it again: exit code 124 means it is still
  going.
- `dibs run ...` (same options as submit) submits, prints the output as it comes, and exits with
  the job's exit code: handy for short jobs.
- `dibs queue` shows every running and queued job, where yours is in line, when it's expected to
  start, and why it's waiting. `dibs queue --mine` shows only yours.
- `dibs show ID` has everything about one job; `dibs logs ID` prints its output (`-f` follows it).
- Interrupting `dibs wait` or `dibs run` never cancels the job.

## Being a good neighbour

- `dibs cancel ID` as soon as you no longer need a job, for example after finding a bug.
- A long-running server is a job too: give it an estimate of how long you'll need it, and cancel
  it when you're done.
- Never run GPU work outside dibs, and never stop the dibs daemon or edit its config to get
  around it: the other agents rely on it.
- Inside a job, DIBS_CPUS, DIBS_MEM and DIBS_VRAM (bytes) say what it was given, and
  CUDA_VISIBLE_DEVICES shows only its GPUs. On a machine with AMD GPUs it is
  ROCR_VISIBLE_DEVICES instead (the GPU ids are amd-smi's), and HIP_VISIBLE_DEVICES and
  CUDA_VISIBLE_DEVICES are removed from the job's environment: don't set them yourself.
"""

SKILL = (
    "---\n"
    "name: dibs\n"
    "description: Use before running anything that uses a GPU (training, fine-tuning, inference, "
    "evals, CUDA tests, model servers) or keeps several CPU cores or lots of RAM busy for more than "
    "a minute or two. This machine's GPUs, CPUs and RAM are shared with other agents through the "
    "dibs job queue, and GPU processes started outside it are stopped.\n"
    "---\n\n" + AGENT_HELP
)
