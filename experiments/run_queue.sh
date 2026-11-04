#!/usr/bin/env bash
# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only
#
# Run one runner on many configurations, one after another:
#
#   experiments/run_queue.sh <runner> [<runner's options>...] -- <configuration>...
#
# It stops at the first configuration that fails, and before starting one if the
# repository's commit or its tracked files have changed since the queue started: every
# configuration starts the runner afresh, and its run.toml records the commit.
#
# For a long queue, such as a search's grid, run it as a systemd user service, so that
# every process it starts can be paused and resumed at once by the cgroup freezer:
#
#   systemd-run --user --unit=arm-grid --same-dir --collect --setenv=PATH="$PATH" \
#       experiments/run_queue.sh experiments/sweep_robot_esn.py --workers 30 -- <configuration>...
#   systemctl --user freeze arm-grid   # pause: its processes stop, keeping their memory
#   systemctl --user thaw arm-grid     # resume where they stopped
#   journalctl --user -u arm-grid -f   # follow its output
#   systemctl --user stop arm-grid     # abandon it: the configuration running is lost
#
# A paused queue does not survive logging out or rebooting. Pausing changes no
# computation, but it moves which worker of sweep_robot_esn.py runs which ESN, as any
# load on the machine does, and an ESN run after others in the same worker agrees with
# one run in a fresh worker to rounding (about 1e-10 relative), not bit for bit: a sweep
# whose workers run several ESNs each repeats to rounding, paused or not.

set -euo pipefail

usage="usage: $0 <runner> [<runner's options>...] -- <configuration>..."
if (($# < 1)); then
    echo "$usage" >&2
    exit 2
fi
runner=$1
shift
options=()
while (($#)) && [[ $1 != -- ]]; do
    options+=("$1")
    shift
done
if (($# < 2)); then
    echo "$usage" >&2
    exit 2
fi
shift # the --
for config in "$@"; do
    if [[ ! -f $config ]]; then
        echo "no configuration $config" >&2
        exit 2
    fi
done

commit=$(git rev-parse HEAD)
total=$#
k=0
for config in "$@"; do
    k=$((k + 1))
    if [[ $(git rev-parse HEAD) != "$commit" ]] || ! git diff --quiet HEAD --; then
        echo "=== stopped before [$k/$total] $config: the repository changed since $commit" >&2
        exit 1
    fi
    echo "=== [$k/$total] $(date '+%F %T') $runner $config"
    uv run python "$runner" "$config" "${options[@]}"
done
echo "=== done $(date '+%F %T'): $total configurations"
