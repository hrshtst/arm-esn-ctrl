# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Import demonstrations taught by hand into a demonstration run.

    uv run python experiments/import_demonstrations.py experiments/demonstrations/reach_manual_single.toml

The take recorded with skelarm's ``tools/trajectory_recorder.py`` (``take`` in
``[recording]``, relative to the storage root) is checked first, with
:func:`arm_esn_ctrl.demonstrations.check_take`: it must have been recorded with the
configuration's arm, begin at the start posture, arrive at the target and hold
there, and never jump. A take that fails a check is not imported: the problems are
printed, no run is created, and the take must be recorded again.

Several takes of one task, each from its own start posture, are listed as ``takes``
instead, in the order of the start postures of ``[demonstrations]`` ``start_q``.
They are imported together, as ``demo_00``, ``demo_01``, ..., each with its filtered
copy, only if every one passes; the figure then draws them together.

A periodic motion (a ``[task]`` of type ``periodic_curve``) has no target to arrive
at and hold: its hand must go around the curve's center instead, at least
``[recording]`` ``min_loops`` whole times, counterclockwise, the way the curve's angle
grows.

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
  of a periodic motion, its loops instead of the reach metrics and the arrival
  (see :func:`loop_metrics`);
- ``demonstrations.png``: the hand path, over the curve of a periodic task, the
  joint angles and the hand speed over time of both, and the sampling intervals of
  the take.
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

from arm_esn_ctrl.demonstrations import check_take, endpoint_positions, smooth_take, task_path
from arm_esn_ctrl.metrics import (
    distances_to_path,
    hand_speed,
    hold_metrics,
    jitter,
    loop_periods,
    mean_loop,
    onset_index,
    reach_metrics,
    taught_loop,
)
from arm_esn_ctrl.postures import joint_columns
from arm_esn_ctrl.storage import REPO_ROOT, resolve_run_path, start_run

TAKE_COLOR = "#2a78d6"
RECORDED_COLOR = "#a3a29d"
# Color, line width, and legend label of each demonstration: the take as recorded, under the filtered one.
DEMO_STYLES = {"demo_00": (RECORDED_COLOR, 2.2, "as recorded"), "demo_00_filtered": (TAKE_COLOR, 1.2, "filtered")}
# The colors of several takes, one each, in turn.
TAKE_COLORS = ["#2a78d6", "#eb6834", "#2e8b57", "#8b5fbf", "#c2185b", "#00838f", "#9e7b00", "#5d6d7e", "#6d4c41"]
JITTER_WINDOW = 5  # samples of the moving average the jitter is measured from, as in the autonomous runs
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
# The line style of each joint's angle, and its name in the panel's title.
JOINT_STYLES = [("-", "solid"), ("--", "dashed"), (":", "dotted"), ("-.", "dash-dot")]
# A periodic task's curve is measured and drawn as a polygon of this many sides, within 0.01 mm of it.
CURVE_SIDES = 360


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="demonstration configuration file (TOML)")
    args = parser.parse_args()

    # Check the take before creating a run, so that a take that fails leaves nothing behind.
    with args.config.open("rb") as f:
        config = tomllib.load(f)
    recording = config["recording"]
    if ("take" in recording) == ("takes" in recording):
        msg = "[recording] names one take, or lists several takes, not both"
        raise ValueError(msg)
    takes = [resolve_run_path(path) for path in recording.get("takes", [recording.get("take")])]
    starts_deg = config["demonstrations"]["start_q"]
    if len(starts_deg) != len(takes):
        msg = f"[demonstrations] start_q lists {len(starts_deg)} start postures for {len(takes)} takes"
        raise ValueError(msg)
    skeleton = Skeleton.from_toml(args.config)
    task = Task.from_dict(config["task"])
    if task.target is not None and task.tolerance is None:
        msg = "the [task] target needs a tolerance, which is the goal radius"
        raise ValueError(msg)
    periodic = task.type == "periodic_curve"
    demos: dict[str, StateLog] = {}
    start_of: dict[str, list[float]] = {}
    take_of: dict[str, Path] = {}
    for i, (take, start_deg) in enumerate(zip(takes, starts_deg, strict=True)):
        recorded = StateLog.load(take)
        for name, log in (
            (f"demo_{i:02d}", recorded),
            (f"demo_{i:02d}_filtered", smooth_take(recorded, skeleton, config["task"], recording["filter"])),
        ):
            demos[name], start_of[name], take_of[name] = log, start_deg, take
    problems = {
        name: check_take(
            log,
            skeleton,
            start_q=np.radians(start_of[name]),
            target=task.target,
            radius=task.tolerance or 0.0,
            hold=0.0 if periodic else recording["hold"],  # a periodic motion has no target to hold at
            start_tolerance=float(np.radians(recording["start_tolerance_deg"])),
            max_tip_speed=recording["max_tip_speed"],
            center=task.params["center"] if periodic else None,
            min_loops=recording["min_loops"] if periodic else 0,
        )
        for name, log in demos.items()
    }
    if any(problems.values()):
        print("Not imported: " + ", ".join(str(take) for take in takes))
        for name, found in problems.items():
            for problem in found:
                print(f"  - {name} ({take_of[name].name}): {problem}")
        raise SystemExit(1)

    _, run_dir = start_run(args.config)
    for name, log in demos.items():
        if name.endswith("_filtered"):
            log.save(run_dir / f"{name}.sklog.npz")
        else:
            shutil.copy2(take_of[name], run_dir / f"{name}.sklog.npz")
    hands = {name: endpoint_positions(skeleton, joint_angles(log)) for name, log in demos.items()}
    rows = [
        {"demo": name}
        | (
            loop_metrics(log, hands[name], start_of[name], task)
            if periodic
            else demo_metrics(log, hands[name], start_of[name], task, recording["hold"])
        )
        | sampling_metrics(demos[name.removesuffix("_filtered")])
        | {"take_sha256": sha256_of(take_of[name])}
        for name, log in demos.items()
    ]
    with (run_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print_metrics(rows)
    plot_take(demos, hands, task, rows, title=args.config.stem).savefig(run_dir / "demonstrations.png", dpi=150)

    print(f"\nImported {len(takes)} take{'s' if len(takes) > 1 else ''} to {run_dir}")
    player = (REPO_ROOT / "third_party/skelarm/tools/player.py").relative_to(REPO_ROOT)
    print(f"Replay them with:\n  uv run python {player} {run_dir / 'demo_00.sklog.npz'}")
    print(f"  uv run python {player} {run_dir / 'demo_00_filtered.sklog.npz'}")


def joint_angles(log: StateLog) -> NDArray[np.float64]:
    """The joint angles of a log, shape ``(n, joints)`` (rad)."""
    return log.channel("q").reshape(len(log.times), -1)


def demo_metrics(
    log: StateLog, hand: NDArray[np.float64], start_deg: list[float], task: Task, hold: float
) -> dict[str, Any]:
    """A demonstration's start, reach metrics, onset and arrival, and joint jitter (5 samples).

    A task without a target has no reach to measure: only the start, the onset, and the jitter.
    """
    times = log.times
    onset = onset_index(hand)
    reach: dict[str, Any] = {}
    arrival = float("nan")
    if task.target is not None:
        reach = reach_metrics(times, hand, task.target)
        arrival = hold_metrics(times, hand, task.target, task.tolerance or 0.0, hold)["arrival_time_s"]
    return (
        joint_columns("start_q{j}_deg", start_deg)
        | reach
        | {
            "onset_time_s": float("nan") if onset is None else float(times[onset]),
            "arrival_time_s": arrival,
            "jitter_deg": float(np.degrees(jitter(joint_angles(log), JITTER_WINDOW))),
        }
    )


def loop_metrics(log: StateLog, hand: NDArray[np.float64], start_deg: list[float], task: Task) -> dict[str, Any]:
    """A periodic demonstration's start, its loops, its taught loop, its onset, and its joint jitter (5 samples).

    The loops are the hand's around the center of the task's curve:

    - ``loops``: how many whole times it goes around, and ``median_period_s``,
      ``shortest_period_s``, and ``longest_period_s``: how long a loop takes (see
      :func:`arm_esn_ctrl.metrics.loop_periods`);
    - ``loop_spread_m``: the hand's RMS distance from its mean loop, how much its
      loops differ from each other (see :func:`arm_esn_ctrl.metrics.mean_loop`);
    - ``curve_rms_m`` and ``curve_max_m``: the hand's RMS and largest distance from
      the task's curve;
    - ``taught_loop_start_s`` and ``taught_loop_end_s``: the times of the first
      sample and of the one past the last of the loop a replay goes through again
      and again, at the take's own samples (see
      :func:`arm_esn_ctrl.metrics.taught_loop`), and ``taught_loop_mismatch_deg``:
      how far apart the postures at the two are, the step the replay misses where
      it goes back to the first.
    """
    times, q = log.times, joint_angles(log)
    center = task.params["center"]
    periods = loop_periods(times, hand, center)
    curve = task_path(task.to_dict(), CURVE_SIDES)
    assert curve is not None
    from_curve = distances_to_path(hand, curve)
    first, end = taught_loop(q, hand, center)
    onset = onset_index(hand)
    return joint_columns("start_q{j}_deg", start_deg) | {
        "loops": len(periods),
        "median_period_s": float(np.median(periods)),
        "shortest_period_s": float(periods.min()),
        "longest_period_s": float(periods.max()),
        "loop_spread_m": mean_loop(hand, center)[1],
        "curve_rms_m": float(np.sqrt(np.mean(from_curve**2))),
        "curve_max_m": float(from_curve.max()),
        "taught_loop_start_s": float(times[first]),
        "taught_loop_end_s": float(times[end]),
        "taught_loop_mismatch_deg": float(np.degrees(np.linalg.norm(q[end] - q[first]))),
        "onset_time_s": float("nan") if onset is None else float(times[onset]),
        "jitter_deg": float(np.degrees(jitter(q, JITTER_WINDOW))),
    }


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
    """Print each demonstration's timing and reach or loop metrics, then how each take was sampled."""
    for row in rows:
        if "loops" in row:  # a periodic motion
            print(
                f"{row['demo']}: moves at {row['onset_time_s']:.2f} s of {row['length_s']:.2f} s;"
                f" {int(row['loops'])} whole loops, of {row['median_period_s']:.2f} s"
                f" ({row['shortest_period_s']:.2f} to {row['longest_period_s']:.2f} s), spread"
                f" {1000 * row['loop_spread_m']:.1f} mm from their mean; from the curve"
                f" {1000 * row['curve_rms_m']:.1f} mm RMS and {1000 * row['curve_max_m']:.1f} mm at most;"
                f" taught loop {row['taught_loop_start_s']:.2f} to {row['taught_loop_end_s']:.2f} s, closing"
                f" {row['taught_loop_mismatch_deg']:.3f} deg apart; joint jitter {row['jitter_deg']:.3f} deg"
            )
            continue
        if "movement_time" not in row:  # a task without a target
            print(
                f"{row['demo']}: moves at {row['onset_time_s']:.2f} s of {row['length_s']:.2f} s;"
                f" joint jitter {row['jitter_deg']:.3f} deg"
            )
            continue
        print(
            f"{row['demo']}: moves at {row['onset_time_s']:.2f} s and arrives at {row['arrival_time_s']:.2f} s"
            f" of {row['length_s']:.2f} s; movement {row['movement_time']:.2f} s, peak speed"
            f" {row['peak_speed']:.2f} m/s, {int(row['speed_peaks'])} speed peaks, joint jitter"
            f" {row['jitter_deg']:.3f} deg, path deviation {row['path_deviation']:.1%},"
            f" final error {1000 * row['final_error']:.1f} mm"
        )
    recorded = [row for row in rows if not row["demo"].endswith("_filtered")]
    for row in recorded:
        name = "The take" if len(recorded) == 1 else f"The take of {row['demo']}"
        print(
            f"{name}: {row['samples']} samples at {row['achieved_rate_hz']:.1f} Hz of {row['requested_rate_hz']:g} Hz"
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
    """The hand path, the joint angles and hand speed over time of each demonstration, and the takes' sampling.

    With one take, the dotted lines mark when its hand starts to move and when it
    arrives, or, of a periodic motion, where its taught loop begins and ends. With
    several, each take has its own color, its recorded hand path gray under the
    filtered one, and only the filtered joint angles and speeds are drawn. The curve
    of a periodic task is drawn under the hand paths.
    """
    several = len(demos) > 2
    fig = Figure(figsize=(12, 8.5), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(f"Demonstration{'s' if several else ''} taught by hand: {title}", color="#0b0b0b")
    ax_hand, ax_joint, ax_speed, ax_interval = fig.subplots(2, 2).flat
    for ax in (ax_hand, ax_joint, ax_speed, ax_interval):
        style(ax)

    path = task_path(task.to_dict(), CURVE_SIDES)
    if path is not None:
        ax_hand.plot(path[:, 0], path[:, 1], color="#0b0b0b", linewidth=0.8, label="the task's curve", zorder=1)
    for name, log in demos.items():
        times, hand = log.times, hands[name]
        filtered = name.endswith("_filtered")
        take = int(name.split("_")[1])
        if several:
            color = TAKE_COLORS[take % len(TAKE_COLORS)] if filtered else RECORDED_COLOR
            width = 1.2 if filtered else 2.2
            label = f"take {take:02d}" if filtered else None
        else:
            color, width, label = DEMO_STYLES[name]
        q = np.degrees(joint_angles(log))
        ax_hand.plot(hand[:, 0], hand[:, 1], color=color, linewidth=width, label=label)
        if several and not filtered:
            ax_hand.plot(*hand[0], marker="o", markersize=5, color="#0b0b0b")
            ax_interval.plot(times[1:], 1000 * np.diff(times), color=TAKE_COLORS[take % len(TAKE_COLORS)], linewidth=1)
            continue
        for j, (linestyle, _) in zip(range(q.shape[1]), JOINT_STYLES, strict=False):
            ax_joint.plot(times, q[:, j], color=color, linestyle=linestyle, linewidth=width)
        ax_speed.plot(times, hand_speed(times, hand), color=color, linewidth=width)
    if not several:
        ax_hand.plot(*hands["demo_00"][0], marker="o", markersize=6, color="#0b0b0b")
    if task.target is not None:
        ax_hand.plot(*task.target, marker="+", markersize=12, color="#0b0b0b", markeredgewidth=1.5)
    if task.target is not None and task.tolerance is not None:
        ring = np.linspace(0.0, 2 * np.pi, 100)
        ax_hand.plot(
            task.target[0] + task.tolerance * np.cos(ring),
            task.target[1] + task.tolerance * np.sin(ring),
            color="#0b0b0b",
            linewidth=0.8,
        )
    marks = "circle: start; cross: target" if task.target is not None else "circle: start"
    ax_hand.set(title=f"Hand path ({marks})", xlabel="x (m)", ylabel="y (m)", aspect="equal")
    ax_hand.legend(frameon=False, labelcolor=TEXT_COLOR, fontsize=7 if several else None)
    n_joints = joint_angles(demos["demo_00"]).shape[1]
    styles = ", ".join(name for _, name in JOINT_STYLES[:n_joints])
    ax_joint.set(title=f"Joint angles (joints 1 to {n_joints}: {styles})", xlabel="time (s)", ylabel="angle (deg)")
    ax_speed.set(title="Hand speed", xlabel="time (s)", ylabel="speed (m/s)")
    if not several:
        for ax in (ax_joint, ax_speed):
            for key in ("onset_time_s", "arrival_time_s", "taught_loop_start_s", "taught_loop_end_s"):
                if key in rows[0] and np.isfinite(rows[0][key]):
                    ax.axvline(rows[0][key], color=TEXT_COLOR, linewidth=0.8, linestyle=":")
        times = demos["demo_00"].times
        ax_interval.plot(times[1:], 1000 * np.diff(times), color=TAKE_COLOR, linewidth=1.0)
    requested = rows[0]["requested_rate_hz"]
    if np.isfinite(requested):
        ax_interval.axhline(1000 / requested, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
    ax_interval.set(
        title=f"Sampling interval of the take{'s' if several else ''} (dotted: requested)",
        xlabel="time (s)",
        ylabel="interval (ms)",
    )
    return fig


if __name__ == "__main__":
    main()
