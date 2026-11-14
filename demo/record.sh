#!/usr/bin/env bash
# Records assets/demo.gif. Needs tmux, asciinema and agg, and the repo's venv (uv sync).
# Everything runs in a throwaway world: its own DIBS_HOME, a fake 8G GPU, no systemd, so it
# never touches a real dibs or a real GPU.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(dirname "$here")
work=$(mktemp -d /tmp/dibs-demo.XXXXXX)
cleanup() {
  set +e
  tmux -L dibs-demo kill-server 2>/dev/null
  tmux -L dibs-rec kill-server 2>/dev/null
  for id in $(seq 1 20); do DIBS_HOME=$work/home "$work/bin/dibs" cancel "$id"; done >/dev/null 2>&1
  DIBS_HOME=$work/home "$work/bin/dibs" daemon stop >/dev/null 2>&1
  rm -rf "$work"
}
trap cleanup EXIT

mkdir -p "$work/bin" "$work/home"
cp "$here/config.toml" "$work/home/config.toml"
cp "$here/gpus.json" "$work/gpus.json"
printf '#!/bin/sh\nexec "%s/.venv/bin/python" -m dibs "$@"\n' "$repo" > "$work/bin/dibs"
chmod +x "$work/bin/dibs"
# A plain `python`, so commands show up as `python train.py` rather than a long interpreter path.
ln -s "$(python3 -c 'import os, sys; print(os.path.realpath(sys.executable))')" "$work/bin/python"

# Each "agent" works in its own project; the scripts just print and sleep.
for project in chess-bot finetune llm-eval data-prep vision; do mkdir -p "$work/$project"; done
printf 'import time\nfor e in range(7):\n    print(f"epoch {e}", flush=True); time.sleep(5)\n' > "$work/chess-bot/train.py"
printf 'import time\nprint("finetuning", flush=True); time.sleep(15)\n' > "$work/finetune/finetune.py"
printf 'import time\nprint("evaluating", flush=True); time.sleep(6)\n' > "$work/llm-eval/eval.py"
printf 'import time\nprint("tokenizing", flush=True); time.sleep(5)\n' > "$work/data-prep/tokenize.py"
printf 'import time\nprint("loading weights onto the GPU...", flush=True); time.sleep(60)\n' > "$work/vision/detect.py"

tmux -L dibs-rec -f /dev/null new-session -d -x 110 -y 32 \
  "asciinema rec --overwrite --cols 110 --rows 32 -c '$here/driver.sh $work' $work/demo.cast; tmux -L dibs-rec wait-for -S recorded"
tmux -L dibs-rec wait-for recorded
agg --theme monokai --font-size 15 --idle-time-limit 2 --last-frame-duration 4 "$work/demo.cast" "$repo/assets/demo.gif"
cp "$work/demo.cast" "${DEMO_CAST:-/dev/null}" 2>/dev/null || true
echo "wrote $repo/assets/demo.gif"
