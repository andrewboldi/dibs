# Install dibs by hand

These are the steps [install.sh](../install.sh) takes, for people who'd rather run them one at a
time. dibs runs on Linux.

1. Install uv, if you don't have it. It fetches its own Python 3.11 or newer, so you don't need one.

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Open a new shell afterwards, so `~/.local/bin` is on your PATH.

2. Install dibs:

   ```sh
   uv tool install --python ">=3.11" "dibs @ https://github.com/andrewboldi/dibs/archive/refs/heads/main.tar.gz"
   ```

3. See what `dibs setup` would change, then run it:

   ```sh
   dibs setup --dry-run
   dibs setup
   ```

   It adds a short marked block about dibs to each agent CLI's instructions file it finds, writes
   a Claude Code skill, and installs a systemd user service so dibs and its guard always run. Add
   `--no-service` to skip the service; dibs then starts on the first `dibs` command.

4. Check what dibs hands out and enforces:

   ```sh
   dibs info
   ```

**Update:** run step 2 again with `--force --reinstall`. The next `dibs` command restarts the
scheduler on the new version, and running jobs keep running. A `dibs wait`, `queue`, `show`, `info` or
`top` that is running from before the update starts again on the new version.

**Uninstall:**

```sh
dibs setup --undo
dibs daemon stop
uv tool uninstall dibs
```

Job history and config stay in `~/.local/state/dibs` and `~/.config/dibs`; delete those folders to
remove them too.
