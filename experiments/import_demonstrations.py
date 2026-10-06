# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Import a reaching demonstration taught by hand into a demonstration run.

    uv run python experiments/import_demonstrations.py experiments/demonstrations/reach_manual_single.toml

The take recorded with skelarm's ``tools/trajectory_recorder.py`` (``take`` in
``[recording]``, relative to the storage root) is checked first, with
:func:`arm_esn_ctrl.demonstrations.check_take`: it must have been recorded with the
configuration's arm, begin at the start posture, arrive at the target and hold
there, and never jump. A take that fails a check is not imported: the problems are
printed, no run is created, and the take must be recorded again.

A take that passes is imported unchanged, its pause before moving included, into a
new run directory, which a configuration's ``[demonstrations]`` then names as any
other demonstration run:

- ``demo_00.sklog.npz``: a copy of the take; its samples keep their uneven times,
  and the ESN resamples them at its own period;
- ``metrics.csv``: the take's reach metrics (see
  :func:`arm_esn_ctrl.metrics.reach_metrics`), when the hand starts to move (5 mm
  from where it started) and when it arrives, how it was sampled (the number of
  samples, the length, the requested and achieved rates, and the longest
  interval), and the SHA-256 of the file, which ties the run to the take;
- ``demonstrations.png``: the hand path, the joint angles and the hand speed over
  time, and the sampling intervals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from skelarm import Skeleton, StateLog, Task

from arm_esn_ctrl.demonstrations import check_take, endpoint_positions
from arm_esn_ctrl.metrics import hand_speed, hold_metrics, onset_index, reach_metrics
from arm_esn_ctrl.storage import REPO_ROOT, resolve_run_path, start_run

TAKE_COLOR = "#2a78d6"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="demonstration configuration file (TOML)")
    args = parser.parse_args()

    # Check the take before creating a run, so that a take that fails leaves nothing behind.
    with args.config.open("rb") as f:
        config = tomllib.load(f)
    recording = config["recording"]
    take = resolve_run_path(recording["take"])
    log = StateLog.load(take)
    skeleton = Skeleton.from_toml(args.config)
    task = Task.from_dict(config["task"])
    if task.tolerance is None:
        msg = "the [task] target needs a tolerance, which is the goal radius"
        raise ValueError(msg)
    (start_deg,) = config["demonstrations"]["start_q"]  # one take, from one start posture
    problems = check_take(
        log,
        skeleton,
        start_q=np.radians(start_deg),
        target=task.require_target(),
        radius=task.tolerance,
        hold=recording["hold"],
        start_tolerance=float(np.radians(recording["start_tolerance_deg"])),
        max_tip_speed=recording["max_tip_speed"],
    )
    if problems:
        print(f"Not imported: {take}")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    _, run_dir = start_run(args.config)
    shutil.copy2(take, run_dir / "demo_00.sklog.npz")
    q = log.channel("q").reshape(len(log.times), -1)
    hand = endpoint_positions(skeleton, q)
    row = take_metrics(log, hand, start_deg, task, recording["hold"]) | {"sha256": sha256_of(take)}
    with (run_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print_metrics(row)
    plot_take(log, hand, task, row, title=args.config.stem).savefig(run_dir / "demonstrations.png", dpi=150)

    print(f"\nImported {take} to {run_dir}")
    player = (REPO_ROOT / "third_party/skelarm/tools/player.py").relative_to(REPO_ROOT)
    print(f"Replay it with:\n  uv run python {player} {run_dir / 'demo_00.sklog.npz'}")


def take_metrics(
    log: StateLog, hand: NDArray[np.float64], start_deg: list[float], task: Task, hold: float
) -> dict[str, Any]:
    """The row of ``metrics.csv`` for the take: reach metrics, onset and arrival, and sampling."""
    times = log.times
    target = task.require_target()
    onset = onset_index(hand)
    acquisition = log.extra.get("acquisition", {})
    return (
        {"demo": 0, "start_q1_deg": start_deg[0], "start_q2_deg": start_deg[1]}
        | reach_metrics(times, hand, target)
        | {
            "onset_time_s": float("nan") if onset is None else float(times[onset]),
            "arrival_time_s": hold_metrics(times, hand, target, task.tolerance or 0.0, hold)["arrival_time_s"],
            "samples": len(times),
            "length_s": float(times[-1] - times[0]),
            "requested_rate_hz": acquisition.get("requested_rate_hz", float("nan")),
            "achieved_rate_hz": acquisition.get("achieved_rate_hz", float("nan")),
            "longest_interval_s": float(np.diff(times).max()),
        }
    )


def sha256_of(path: Path) -> str:
    """The SHA-256 of a file, in hexadecimal."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def print_metrics(row: dict[str, Any]) -> None:
    """Print the take's timing, reach metrics, and sampling."""
    print(
        f"Moves at {row['onset_time_s']:.2f} s and arrives at {row['arrival_time_s']:.2f} s of {row['length_s']:.2f} s;"
        f" movement {row['movement_time']:.2f} s, peak speed {row['peak_speed']:.2f} m/s,"
        f" {int(row['speed_peaks'])} speed peaks, path deviation {row['path_deviation']:.1%},"
        f" final error {1000 * row['final_error']:.1f} mm"
    )
    print(
        f"{row['samples']} samples at {row['achieved_rate_hz']:.1f} Hz of {row['requested_rate_hz']:g} Hz requested;"
        f" longest interval {1000 * row['longest_interval_s']:.1f} ms"
    )


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def plot_take(log: StateLog, hand: NDArray[np.float64], task: Task, row: dict[str, Any], title: str) -> Figure:
    """The hand path, the joint angles and hand speed over time, and the sampling intervals of the take.

    The dotted lines mark when the hand starts to move and when it arrives.
    """
    times = log.times
    q = np.degrees(log.channel("q").reshape(len(times), -1))
    target = task.require_target()
    fig = Figure(figsize=(12, 8.5), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(f"Demonstration taught by hand: {title}", color="#0b0b0b")
    ax_hand, ax_joint, ax_speed, ax_interval = fig.subplots(2, 2).flat
    for ax in (ax_hand, ax_joint, ax_speed, ax_interval):
        style(ax)

    ax_hand.plot(hand[:, 0], hand[:, 1], color=TAKE_COLOR, linewidth=1.5)
    ax_hand.plot(*hand[0], marker="o", markersize=6, color=TAKE_COLOR)
    ax_hand.plot(*target, marker="+", markersize=12, color="#0b0b0b", markeredgewidth=1.5)
    if task.tolerance is not None:
        ring = np.linspace(0.0, 2 * np.pi, 100)
        ax_hand.plot(
            target[0] + task.tolerance * np.cos(ring),
            target[1] + task.tolerance * np.sin(ring),
            color="#0b0b0b",
            linewidth=0.8,
        )
    ax_hand.set(title="Hand path (circle: start; cross: target)", xlabel="x (m)", ylabel="y (m)", aspect="equal")

    for j, style_name in ((0, "-"), (1, "--")):
        ax_joint.plot(times, q[:, j], color=TAKE_COLOR, linestyle=style_name, linewidth=1.5, label=f"joint {j + 1}")
    ax_joint.set(title="Joint angles", xlabel="time (s)", ylabel="angle (deg)")
    ax_joint.legend(frameon=False, labelcolor=TEXT_COLOR)

    ax_speed.plot(times, hand_speed(times, hand), color=TAKE_COLOR, linewidth=1.5)
    ax_speed.set(title="Hand speed", xlabel="time (s)", ylabel="speed (m/s)")
    for ax in (ax_joint, ax_speed):
        for key in ("onset_time_s", "arrival_time_s"):
            if np.isfinite(row[key]):
                ax.axvline(row[key], color=TEXT_COLOR, linewidth=0.8, linestyle=":")

    ax_interval.plot(times[1:], 1000 * np.diff(times), color=TAKE_COLOR, linewidth=1.0)
    requested = row["requested_rate_hz"]
    if np.isfinite(requested):
        ax_interval.axhline(1000 / requested, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
    ax_interval.set(title="Sampling interval (dotted: requested)", xlabel="time (s)", ylabel="interval (ms)")
    return fig


if __name__ == "__main__":
    main()
