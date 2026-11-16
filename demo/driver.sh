#!/usr/bin/env bash
# Plays the demo inside tmux: `dibs top` above, agents typing below. Run by record.sh.
work=$1
unset TMUX TMUX_PANE CLAUDE_CODE_SESSION_ID DIBS_SESSION
export DIBS_HOME=$work/home DIBS_GPU_BACKEND=smi DIBS_FAKE_GPU_FILE=$work/gpus.json DIBS_NO_SYSTEMD=1
export DIBS_MEMINFO=$work/meminfo
export PATH="$work/bin:$(cd "$(dirname "$0")/.." && pwd)/tests/fakes:$PATH"
T() { tmux -L dibs-demo -f "$work/tmux.conf" "$@"; }

cat > "$work/tmux.conf" <<'CONF'
set -g status off
set -g default-terminal tmux-256color
set -g pane-border-style fg=colour239
set -g pane-active-border-style fg=colour239
CONF
# Each project is its own agent session, as separate Claude Code sessions would be.
cat > "$work/bashrc" <<'RC'
PS1='\[\e[34m\]~/${PWD##*/}\[\e[0m\] \[\e[32m\]❯\[\e[0m\] '
PROMPT_COMMAND='export DIBS_SESSION=$(printf %s "$PWD" | md5sum | cut -c1-8)'
RC

dibs daemon start >/dev/null
T new-session -d -s demo -x 110 -y 32 -c "$work/vision" "bash --noprofile --rcfile $work/bashrc -i"
T split-window -v -b -l 20 -t demo "dibs top"
T select-pane -t demo:0.1

say() {  # type into the agents' pane like a person, then press Enter
  local s=$1 i
  for ((i = 0; i < ${#s}; i++)); do T send-keys -t demo:0.1 -l -- "${s:i:1}"; sleep 0.012; done
  sleep 0.5
  T send-keys -t demo:0.1 Enter
}

director() {
  sleep 1.5
  # 1. A training job takes the GPU; a long fine-tune that needs all of it queues behind.
  say "dibs submit --cpus 4 --mem 8G --vram 6G --time 30s -- python train.py"
  sleep 1.5
  say "cd ../finetune"  # its own line: the session follows the folder when the prompt comes back
  say "dibs submit --cpus 4 --mem 8G --vram all --time 1m -- python finetune.py"
  sleep 2
  # 2. A short eval that also needs the whole GPU goes ahead of the fine-tune.
  say "cd ../evals"
  say "dibs submit --cpus 2 --mem 4G --vram all --time 10s -- python eval.py"
  sleep 2.5
  # 3. A script started without dibs is stopped within a second, with a note on how to queue it.
  say "clear"  # separate commands: tmux's send-keys reads a lone ";" as its own command separator
  say "cd ../detector"
  say "python detect.py"
  sleep 1.5
  shell=$(T display-message -p -t demo:0.1 '#{pane_pid}')
  pid=$(pgrep -P "$shell" -f detect.py | head -1)
  # The stray process "allocates" 3.1G on the GPU; the guard sees it on its next look.
  printf '{"gpus": [{"index": 0, "uuid": "GPU-demo-0", "name": "NVIDIA GeForce RTX 3070", "memory_mib": 8192}], "processes": [{"pid": %s, "gpu_uuid": "GPU-demo-0", "used_mib": 3174}]}\n' "$pid" > "$work/gpus.json"
  sleep 6
  T detach-client
}

director &
T attach -t demo
