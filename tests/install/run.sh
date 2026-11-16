#!/usr/bin/env bash
# Test install.sh in fresh containers: ubuntu:24.04, debian:12, fedora:41, archlinux:latest, each as
# root and as a sudo user, through four starting states (see scenario.sh).
#
#   bash tests/install/run.sh                 # the whole matrix
#   bash tests/install/run.sh ubuntu:24.04    # one image
#
# Needs docker. The checkout is mounted read-only and installed with DIBS_SOURCE, so nothing here
# touches the host or the network copy of dibs. Logs go to $LOGS (default: a temp folder).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
images=("$@")
[ ${#images[@]} -gt 0 ] || images=(ubuntu:24.04 debian:12 fedora:41 archlinux:latest)
LOGS="${LOGS:-$(mktemp -d)}"
mkdir -p "$LOGS"

run_one() { # image user
  local image="$1" user="$2" log="$LOGS/${1//[:\/]/_}-$2.log"
  if docker run --rm -v "$repo:/checkout:ro" -v "$here:/tests:ro" -e TEST_USER="$user" "$image" \
      bash /tests/container.sh >"$log" 2>&1; then
    echo "PASS $image $user"
  else
    echo "FAIL $image $user (log: $log)"
    return 1
  fi
}

pids=()
for image in "${images[@]}"; do
  ( rc=0; for user in root sudo; do run_one "$image" "$user" || rc=1; done; exit $rc ) &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "logs: $LOGS"
exit $status
