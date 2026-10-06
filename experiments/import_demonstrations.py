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

A take that passes is imported, its pause before moving included, into a new run
directory, as two demonstrations that a configuration's ``[demonstrations]`` names
as any others:

- ``demo_00.sklog.npz``: a copy of the take as recorded, with its uneven sample
  times; the ESN resamples it at its own period;
- ``demo_00_filtered.sklog.npz``: the take smoothed by the filter of
  ``[recording.filter]`` (see :func:`arm_esn_ctrl.demonstrations.smooth_take`),
  which removes the steps of the cursor's whole screen pixels; the same checks
  apply to it;

and their measurements:

- ``metrics.csv``: one row per demonstration: its reach metrics (see
  :func:`arm_esn_ctrl.metrics.reach_metrics`), when the hand starts to move (5 mm
  from where it started) and when it arrives, its joint jitter, how the take was
  sampled (the number of samples, the length, the requested and achieved rates,
  and the longest interval), and the SHA-256 of the take, which ties the run to it;
- ``demonstrations.png``: the hand path, the joint angles and the hand speed over
  time of both, and the sampling intervals of the take.
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

from arm_esn_ctrl.demonstrations import check_take, endpoint_positions, smooth_take
from arm_esn_ctrl.metrics import hand_speed, hold_metrics, jitter, onset_index, reach_metrics
from arm_esn_ctrl.storage import REPO_ROOT, resolve_run_path, start_run

TAKE_COLOR = "#2a78d6"
RECORDED_COLOR = "#a3a29d"
# Color, line width, and legend label of each demonstration: the take as recorded, under the filtered one.
DEMO_STYLES = {"demo_00": (RECORDED_COLOR, 2.2, "as recorded"), "demo_00_filtered": (TAKE_COLOR, 1.2, "filtered")}
JITTER_WINDOW = 5  # samples of the moving average the jitter is measured from, as in the autonomous runs
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
    skeleton = Skeleton.from_toml(args.config)
    task = Task.from_dict(config["task"])
    if task.tolerance is None:
        msg = "the [task] target needs a tolerance, which is the goal radius"
        raise ValueError(msg)
    (start_deg,) = config["demonstrations"]["start_q"]  # one take, from one start posture
    recorded = StateLog.load(take)
    demos = {
        "demo_00": recorded,
        "demo_00_filtered": smooth_take(recorded, skeleton, config["task"], recording["filter"]),
    }
    problems = {
        name: check_take(
            log,
            skeleton,
            start_q=np.radians(start_deg),
            target=task.require_target(),
            radius=task.tolerance,
            hold=recording["hold"],
            start_tolerance=float(np.radians(recording["start_tolerance_deg"])),
            max_tip_speed=recording["max_tip_speed"],
        )
        for name, log in demos.items()
    }
    if any(problems.values()):
        print(f"Not imported: {take}")
        for name, found in problems.items():
            for problem in found:
                print(f"  - {name}: {problem}")
        raise SystemExit(1)

    _, run_dir = start_run(args.config)
    shutil.copy2(take, run_dir / "demo_00.sklog.npz")
    demos["demo_00_filtered"].save(run_dir / "demo_00_filtered.sklog.npz")
    sampling = sampling_metrics(recorded) | {"take_sha256": sha256_of(take)}
    hands = {name: endpoint_positions(skeleton, joint_angles(log)) for name, log in demos.items()}
    rows = [
        {"demo": name} | demo_metrics(log, hands[name], start_deg, task, recording["hold"]) | sampling
        for name, log in demos.items()
    ]
    with (run_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print_metrics(rows)
    plot_take(demos, hands, task, rows, title=args.config.stem).savefig(run_dir / "demonstrations.png", dpi=150)

    print(f"\nImported {take} to {run_dir}")
    player = (REPO_ROOT / "third_party/skelarm/tools/player.py").relative_to(REPO_ROOT)
    print(f"Replay them with:\n  uv run python {player} {run_dir / 'demo_00.sklog.npz'}")
    print(f"  uv run python {player} {run_dir / 'demo_00_filtered.sklog.npz'}")


def joint_angles(log: StateLog) -> NDArray[np.float64]:
    """The joint angles of a log, shape ``(n, joints)`` (rad)."""
    return log.channel("q").reshape(len(log.times), -1)


def demo_metrics(
    log: StateLog, hand: NDArray[np.float64], start_deg: list[float], task: Task, hold: float
) -> dict[str, Any]:
    """A demonstration's start, reach metrics, onset and arrival, and joint jitter (5 samples)."""
    times = log.times
    target = task.require_target()
    onset = onset_index(hand)
    return (
        {"start_q1_deg": start_deg[0], "start_q2_deg": start_deg[1]}
        | reach_metrics(times, hand, target)
        | {
            "onset_time_s": float("nan") if onset is None else float(times[onset]),
            "arrival_time_s": hold_metrics(times, hand, target, task.tolerance or 0.0, hold)["arrival_time_s"],
            "jitter_deg": float(np.degrees(jitter(joint_angles(log), JITTER_WINDOW))),
        }
    )


def sampling_metrics(log: StateLog) -> dict[str, Any]:
    """How the take was sampled: the number of samples, the length, the rates, and the longest interval."""
    times = log.times
    acquisition = log.extra.get("acquisition", {})
    return {
        "samples": len(times),
        "length_s": float(times[-1] - times[0]),
        "requested_rate_hz": acquisition.get("requested_rate_hz", float("nan")),
        "achieved_rate_hz": acquisition.get("achieved_rate_hz", float("nan")),
        "longest_interval_s": float(np.diff(times).max()),
    }


def sha256_of(path: Path) -> str:
    """The SHA-256 of a file, in hexadecimal."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def print_metrics(rows: list[dict[str, Any]]) -> None:
    """Print each demonstration's timing and reach metrics, then how the take was sampled."""
    for row in rows:
        print(
            f"{row['demo']}: moves at {row['onset_time_s']:.2f} s and arrives at {row['arrival_time_s']:.2f} s"
            f" of {row['length_s']:.2f} s; movement {row['movement_time']:.2f} s, peak speed"
            f" {row['peak_speed']:.2f} m/s, {int(row['speed_peaks'])} speed peaks, joint jitter"
            f" {row['jitter_deg']:.3f} deg, path deviation {row['path_deviation']:.1%},"
            f" final error {1000 * row['final_error']:.1f} mm"
        )
    row = rows[0]
    print(
        f"The take: {row['samples']} samples at {row['achieved_rate_hz']:.1f} Hz of {row['requested_rate_hz']:g} Hz"
        f" requested; longest interval {1000 * row['longest_interval_s']:.1f} ms"
    )


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def plot_take(
    demos: dict[str, StateLog],
    hands: dict[str, NDArray[np.float64]],
    task: Task,
    rows: list[dict[str, Any]],
    title: str,
) -> Figure:
    """The hand path, the joint angles and hand speed over time of each demonstration, and the take's sampling.

    The dotted lines mark when the recorded take's hand starts to move and when it arrives.
    """
    target = task.require_target()
    fig = Figure(figsize=(12, 8.5), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(f"Demonstration taught by hand: {title}", color="#0b0b0b")
    ax_hand, ax_joint, ax_speed, ax_interval = fig.subplots(2, 2).flat
    for ax in (ax_hand, ax_joint, ax_speed, ax_interval):
        style(ax)

    for name, log in demos.items():
        times, hand = log.times, hands[name]
        color, width, label = DEMO_STYLES[name]
        q = np.degrees(joint_angles(log))
        ax_hand.plot(hand[:, 0], hand[:, 1], color=color, linewidth=width, label=label)
        for j, linestyle in ((0, "-"), (1, "--")):
            ax_joint.plot(times, q[:, j], color=color, linestyle=linestyle, linewidth=width)
        ax_speed.plot(times, hand_speed(times, hand), color=color, linewidth=width)
    start = hands["demo_00"][0]
    ax_hand.plot(*start, marker="o", markersize=6, color="#0b0b0b")
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
    ax_hand.legend(frameon=False, labelcolor=TEXT_COLOR)
    ax_joint.set(title="Joint angles (solid: joint 1; dashed: joint 2)", xlabel="time (s)", ylabel="angle (deg)")
    ax_speed.set(title="Hand speed", xlabel="time (s)", ylabel="speed (m/s)")
    for ax in (ax_joint, ax_speed):
        for key in ("onset_time_s", "arrival_time_s"):
            if np.isfinite(rows[0][key]):
                ax.axvline(rows[0][key], color=TEXT_COLOR, linewidth=0.8, linestyle=":")

    times = demos["demo_00"].times
    ax_interval.plot(times[1:], 1000 * np.diff(times), color=TAKE_COLOR, linewidth=1.0)
    requested = rows[0]["requested_rate_hz"]
    if np.isfinite(requested):
        ax_interval.axhline(1000 / requested, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
    ax_interval.set(
        title="Sampling interval of the take (dotted: requested)", xlabel="time (s)", ylabel="interval (ms)"
    )
    return fig


if __name__ == "__main__":
    main()
