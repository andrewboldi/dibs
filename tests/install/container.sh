#!/usr/bin/env bash
# Runs inside a container as root: prepare the distro, then run scenario.sh as root or as a sudo user.
set -euo pipefail
. /etc/os-release
case "$ID" in
  ubuntu|debian) apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl ca-certificates sudo >/dev/null ;;
  fedora) dnf install -y -q curl sudo util-linux >/dev/null ;;
  arch) pacman -Syu --noconfirm >/dev/null && pacman -S --needed --noconfirm curl sudo >/dev/null ;;
  *) echo "unsupported test distro $ID"; exit 1 ;;
esac
# Install from a copy of the checkout, so a build can't write into the mount.
mkdir /opt/dibs-src && tar -C /checkout --exclude=.git --exclude=.venv --exclude=.beads -cf - . | tar -C /opt/dibs-src -xf -
chmod -R a+rX /opt/dibs-src
cp /tests/scenario.sh /tmp/scenario.sh && chmod a+rx /tmp/scenario.sh
cp /opt/dibs-src/install.sh /tmp/install.sh && chmod a+r /tmp/install.sh
echo "== $PRETTY_NAME as $TEST_USER"
if [ "$TEST_USER" = root ]; then
  HOME=/root DIBS_SOURCE=/opt/dibs-src bash /tmp/scenario.sh
else
  useradd -m -s /bin/bash tester
  echo 'tester ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/tester
  runuser -u tester -- env HOME=/home/tester USER=tester SHELL=/bin/bash DIBS_SOURCE=/opt/dibs-src bash /tmp/scenario.sh
fi
