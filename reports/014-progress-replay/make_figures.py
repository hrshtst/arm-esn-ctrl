# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 014.

    uv run python reports/014-progress-replay/make_figures.py
    uv run python reports/014-progress-replay/make_figures.py --logs

From the copies of the runs under ``results/`` (configurations, run records, and
metrics), it prints the tables of the report: the completion, the measures after each
disturbance, over the 13 pushes, and from the 49 offset starts, and the references'
peak speeds. With ``--logs``, it reads the runs' logs, which are not kept in Git, from
the runs under the storage root, checks that the progress replay is the replay
undisturbed, and draws into ``results/summary``:

- ``progress.png``: the progress along the taught path of the arms and of their
  references over time, under the block and under log 001's pushes by hand
  (Section 3.1);
- ``offsets.png``: from the 49 starts, each arm's peak torque against how far its start
  is from the demonstrated one (Section 3.2);
- ``<scenario>_joints.png``, ``<scenario>_torques.png``, ``<scenario>_paths.png``: the
  joint angles and the joint torques over time, and the tip's path in the plane, of
  ESN #1, ESN #2, the replay, and the progress replay, a column per tracker and
  reference velocity (Sections 3.2 to 3.4). Over time, the ESNs' lines are drawn over
  the replays', and the progress replay's over the replay's.
"""

from __future__ import annotations

import argparse
import csv
import functools
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrow
from numpy.typing import NDArray
from skelarm import Skeleton, StateLog, Task, compute_forward_kinematics

from arm_esn_ctrl.demonstrations import endpoint_positions, resample_joint_angles
from arm_esn_ctrl.disturbances import disturbance_spans
from arm_esn_ctrl.metrics import path_progress
from arm_esn_ctrl.storage import resolve_run_path

REPORT = Path(__file__).resolve().parent
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"
# The take, as report 005 copied it: what the ESNs were trained on, and what the replays replay.
TAKE_RUN = REPORT.parent / "005-tuned-on-the-robot" / "data" / "20261008-170034-reach_manual_v2"
DT = 0.01  # the ESNs' period (s)
STEP = 0.002  # the simulation's step (s)
DURATION = 24.0  # s: the runs' task, after the warm-up
DPI = 90

# The trackers: name, its runs' setting directory, its configurations' infix, and its label.
TRACKERS = {
    "scaled": ("pd_w20_z0.4", "w20_z0.4", "ω 20, ζ 0.4"),
    "stiff": ("pd_kp80_kd1.2", "kp80_kd1.2", "kp 80, kd 1.2"),
}
VELOCITIES = {"reference": "reference velocity", "zero": "zero reference velocity"}
COLUMNS = [(tracker, velocity) for tracker in TRACKERS for velocity in VELOCITIES]
# The scenarios and their words.
SCENARIOS = {
    "nominal": "undisturbed",
    "block": "under the block",
    "pushes": "under the three pushes",
    "offsets": "from the start offset by -15° in both joints",
    "recorded000": "under log 000's pushes by hand",
    "recorded001": "under log 001's pushes by hand",
    "recorded002": "under log 002's pushes by hand",
}
OFFSET_ORIGIN = "demo_00_filtered -15,-15 deg"  # the offset start drawn over time: joint 2 at 15°, the lowest
LOGS = ("000", "001", "002")
# The arms compared: name, the tag of the runs, the arm in them, color, and marker. The replays are the same in
# every ESN's runs; they are read from ESN #1's.
ARMS = [
    ("ESN #1", "lr0.03_sr1.2_is1.5_seed8", "esn", "#2a78d6", "o"),
    ("ESN #2", "lr0.03_sr1.2_is1_seed2", "esn", "#5b2a86", "s"),
    ("replay", "lr0.03_sr1.2_is1.5_seed8", "replay", "#eb6834", "D"),
    ("progress replay", "lr0.03_sr1.2_is1.5_seed8", "progress", "#2e8b57", "^"),
]
# The measures after each disturbance: the column in metrics.csv, its label, its scale, and its digits.
MEASURES = [
    ("following_gain", "following gain", 1.0, 2),
    ("ringing_m", "ringing over the next 2 s (mm)", 1000.0, 1),
    ("return_time_s", "back within 5 mm of the path (s)", 1.0, 2),
    ("off_path_m", "farthest from the path (mm)", 1000.0, 0),
]

TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
DISTURBANCE_COLOR = "#f0efec"
TAUGHT_COLOR = "#b9b7b0"
INITIAL_COLOR = "#a3a29d"
PUSH_COLOR = "#dc0000"  # the force arrows in the plane, red as the robot app's drag arrow
FORCE_SCALE = 0.025  # m/N: the length of a force arrow in the plane, 25 mm per newton
# The layers of the lines over time: the ESNs' arms on top, their references (dashed) under them, then the
# progress replay's arm and reference, the replay's at the bottom.
ARM_ZORDERS = {"esn": 3.0, "progress": 2.2, "replay": 2.0}
REFERENCE_ZORDERS = {"esn": 2.5, "progress": 2.1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", action="store_true", help="also analyze the run logs in the storage root")
    args = parser.parse_args()

    figures = {}
    if args.logs:
        SUMMARY.mkdir(parents=True, exist_ok=True)
        figures = {"progress.png": plot_progress(), "offsets.png": plot_offsets()}
        for scenario in SCENARIOS:
            name = "offset" if scenario == "offsets" else scenario
            for channel in ("joints", "torques"):
                figures[f"{name}_{channel}.png"] = plot_series(scenario, channel)
            figures[f"{name}_paths.png"] = plot_paths(scenario)
        for name, fig in figures.items():
            fig.savefig(SUMMARY / name, dpi=DPI)
    print_tables(logs=args.logs)
    if figures:
        print(f"\nWrote {', '.join(figures)} to {SUMMARY}")


# ---------------------------------------------------------------------------- the arm and the take


@functools.cache
def skeleton() -> Skeleton:
    return Skeleton.from_toml(TAKE_RUN / "config.toml")


@functools.cache
def taught() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The take filtered at 2 Hz, what the ESNs were trained on, every 10 ms: its joint angles and its hand path."""
    q = resample_joint_angles(StateLog.load(TAKE_RUN / "demo_00_filtered.sklog.npz"), DT)[1]
    return q, endpoint_positions(skeleton(), q)


# ---------------------------------------------------------------------------- reading the runs


def value(text: str) -> Any:
    """A CSV field as a bool, a float, or the text itself."""
    if text in ("True", "False"):
        return text == "True"
    try:
        return float(text)
    except ValueError:
        return text


@functools.cache
def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return [{key: value(text) for key, text in row.items()} for row in csv.DictReader(f)]


def run_name(scenario: str, tag: str, velocity: str, tracker: str) -> str:
    """The name of the one run of a configuration."""
    stem = f"{scenario}_{TRACKERS[tracker][1]}_robot_{tag}_{velocity}_velocity_filtered"
    matches = sorted(p.name for p in RESULTS.glob(f"*-{stem}") if p.is_dir())
    if len(matches) != 1:
        msg = f"expected one run of {stem}, found {matches}"
        raise FileNotFoundError(msg)
    return matches[0]


def rows_of(scenario: str, tag: str, velocity: str, tracker: str) -> list[dict[str, Any]]:
    return read_csv(RESULTS / run_name(scenario, tag, velocity, tracker) / "metrics.csv")


def run_of(scenario: str, tag: str, arm: str, velocity: str, tracker: str, start: int = 0) -> dict[str, Any]:
    """One arm's run from one start, with one tracker."""
    return next(r for r in rows_of(scenario, tag, velocity, tracker) if r["arm"] == arm and r["start"] == start)


def offset_start() -> int:
    """The index of the offset start drawn over time."""
    rows = rows_of("offsets", ARMS[0][1], "reference", "scaled")
    return next(int(r["start"]) for r in rows if r["origin"] == OFFSET_ORIGIN)


def start_of(scenario: str) -> int:
    return offset_start() if scenario == "offsets" else 0


def log_of(scenario: str, tag: str, arm: str, velocity: str, tracker: str, start: int = 0) -> StateLog:
    """One arm's log from a run under the storage root."""
    run = resolve_run_path(f"results/{run_name(scenario, tag, velocity, tracker)}")
    return StateLog.load(run / TRACKERS[tracker][0] / f"{arm}_{start:02d}.sklog.npz")


@functools.cache
def config_of(scenario: str) -> dict[str, Any]:
    with (RESULTS / run_name(scenario, ARMS[0][1], "reference", "scaled") / "config.toml").open("rb") as f:
        return tomllib.load(f)


def spans_of(scenario: str) -> list[tuple[float, float]]:
    """When each disturbance of a scenario acts."""
    return disturbance_spans(config_of(scenario).get("disturbance"))


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def column_title(tracker: str, velocity: str) -> str:
    return f"{TRACKERS[tracker][2]}\n{VELOCITIES[velocity]}"


def arm_handles(*, references: bool) -> list[Line2D]:
    """The legend's entries: each arm, and the references (dashed) and the taught motion when drawn."""
    handles = [Line2D([], [], color=color, linewidth=1.4, label=name) for name, _, _, color, _ in ARMS]
    if references:
        handles += [
            Line2D([], [], color=TEXT_COLOR, linewidth=1.0, linestyle="--", label="its reference"),
            Line2D([], [], color=TAUGHT_COLOR, linewidth=4.0, label="the taught motion"),
        ]
    return handles


# ---------------------------------------------------------------------------- figures


def plot_progress() -> Figure:
    """The progress along the taught path of each arm and of its reference over time, under the block and log 001."""
    scenarios = ["block", "recorded001"]
    fig = Figure(figsize=(17, 8.2), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "Progress along the taught joint path, from 0 at its start to 1 at its end: each arm (solid) and the"
        " references of the ESNs and the progress replay (dashed);\nthe replay's reference is the taught motion"
        " (thick gray)",
        color="#0b0b0b",
    )
    axes = fig.subplots(len(scenarios), len(COLUMNS), sharex=True, sharey=True)
    q_taught = taught()[0]
    t_taught = DT * np.arange(len(q_taught))
    every = round(DT / STEP)
    for row, scenario in enumerate(scenarios):
        for col, (tracker, velocity) in enumerate(COLUMNS):
            ax = axes[row, col]
            style(ax)
            for onset, end in spans_of(scenario):
                ax.axvspan(onset, end, color=DISTURBANCE_COLOR, zorder=0)
            ax.plot(t_taught, path_progress(q_taught, q_taught), color=TAUGHT_COLOR, linewidth=4.0, zorder=1)
            for _, tag, arm, color, _ in ARMS:
                log = log_of(scenario, tag, arm, velocity, tracker)
                times = np.asarray(log.times)[::every]
                ax.plot(
                    times,
                    path_progress(log.channel("q")[::every], q_taught),
                    color=color,
                    linewidth=1.2,
                    zorder=ARM_ZORDERS[arm],
                )
                if arm in REFERENCE_ZORDERS:
                    ax.plot(
                        times,
                        path_progress(log.channel("q_ref")[::every], q_taught),
                        color=color,
                        linewidth=1.0,
                        linestyle="--",
                        zorder=REFERENCE_ZORDERS[arm],
                    )
            ax.set_xlim(0, 16)
            if row == 0:
                ax.set_title(column_title(tracker, velocity), color="#0b0b0b", fontsize=10)
            if row == len(scenarios) - 1:
                ax.set_xlabel("time (s)", color=TEXT_COLOR)
            if col == 0:
                ax.set_ylabel(f"{SCENARIOS[scenario]}\nprogress along the path", color=TEXT_COLOR)
    fig.legend(
        handles=arm_handles(references=True),
        loc="outside lower center",
        ncol=len(ARMS) + 2,
        frameon=False,
        labelcolor=TEXT_COLOR,
    )
    return fig


def plot_offsets() -> Figure:
    """From the 49 starts: each arm's peak torque against how far its start is from the demonstrated one."""
    fig = Figure(figsize=(17, 4.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "From the 49 starts: the peak joint torque against the start's distance from the demonstrated start,"
        " in joint space (the two replays' marks coincide)",
        color="#0b0b0b",
    )
    axes = fig.subplots(1, len(COLUMNS), sharex=True, sharey=True)
    demonstrated = taught()[0][0]
    for ax, (tracker, velocity) in zip(axes, COLUMNS, strict=True):
        style(ax)
        for name, tag, arm, color, marker in ARMS:
            rows = [r for r in rows_of("offsets", tag, velocity, tracker) if r["arm"] == arm]
            x = []
            for r in rows:
                start = log_of("offsets", tag, arm, velocity, tracker, int(r["start"])).channel("q")[0]
                x.append(float(np.degrees(np.linalg.norm(start - demonstrated))))
            ax.plot(
                x,
                [r["peak_torque_nm"] for r in rows],
                linestyle="none",
                marker=marker,
                markersize=5,
                color=color,
                markerfacecolor="none",
                label=name,
                zorder=ARM_ZORDERS[arm],
            )
        ax.set_title(column_title(tracker, velocity), color="#0b0b0b", fontsize=10)
        ax.set_xlabel("start's distance from the demonstrated start (deg)", color=TEXT_COLOR)
        ax.set_yscale("log")
    axes[0].set_ylabel("peak joint torque (N m)", color=TEXT_COLOR)
    handles = [
        Line2D([], [], linestyle="none", marker=marker, markersize=6, markerfacecolor="none", color=color, label=name)
        for name, _, _, color, marker in ARMS
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(ARMS), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_series(scenario: str, channel: str) -> Figure:
    """The joint angles or torques of the four arms over time: a column per tracker and reference velocity.

    Under a disturbance, a first row draws each arm's tip force.
    """
    words = SCENARIOS[scenario]
    spans = spans_of(scenario)
    rows = (1 if spans else 0) + 2
    fig = Figure(figsize=(17, 2.4 + 2.6 * rows), facecolor=SURFACE_COLOR, layout="constrained")
    what = "Joint angles" if channel == "joints" else "Joint torques"
    fig.suptitle(
        f"{what} {words}, joint PD, the torque every 10 ms: ESN #1 (blue), ESN #2 (purple), the replay (orange),"
        " and the progress replay (green)"
        + (
            "\ndashed: the references of the ESNs and the progress replay; thick gray: the taught motion"
            if channel == "joints"
            else ""
        ),
        color="#0b0b0b",
    )
    height_ratios = ([0.6] if spans else []) + [1.0, 1.0]
    axes = np.asarray(fig.subplots(rows, len(COLUMNS), sharex=True, height_ratios=height_ratios)).reshape(
        rows, len(COLUMNS)
    )
    start = start_of(scenario)
    q_taught = np.degrees(taught()[0])
    t_taught = DT * np.arange(len(q_taught))
    logs = {
        column: {name: log_of(scenario, tag, arm, column[1], column[0], start) for name, tag, arm, _, _ in ARMS}
        for column in COLUMNS
    }
    # One scale per joint across the columns.
    limits = []
    if channel == "torques":
        for joint in range(2):
            values = [log.channel("tau")[:, joint] for c in COLUMNS for log in logs[c].values()]
            low, high = float(min(v.min() for v in values)), float(max(v.max() for v in values))
            limits.append((low - 0.1 * (high - low), high + 0.1 * (high - low)))
    for col, column in enumerate(COLUMNS):
        for row in range(rows):
            style(axes[row, col])
            for onset, end in spans:
                axes[row, col].axvspan(onset, end, color=DISTURBANCE_COLOR, zorder=0)
        axes[0, col].set_title(column_title(*column), color="#0b0b0b", fontsize=10)
        if spans:
            for name, _, arm, color, _ in ARMS:
                force = np.linalg.norm(logs[column][name].channel("ext_force"), axis=1)
                axes[0, col].plot(logs[column][name].times, force, color=color, linewidth=1.0, zorder=ARM_ZORDERS[arm])
            top = max(
                np.linalg.norm(log.channel("ext_force"), axis=1).max() for c in COLUMNS for log in logs[c].values()
            )
            axes[0, col].set_ylim(-0.05 * top, 1.1 * top)
        for joint in range(2):
            ax = axes[rows - 2 + joint, col]
            if channel == "joints":
                ax.plot(t_taught, q_taught[:, joint], color=TAUGHT_COLOR, linewidth=4.0, zorder=1)
            for name, _, arm, color, _ in ARMS:
                log = logs[column][name]
                if channel == "joints":
                    ax.plot(
                        log.times,
                        np.degrees(log.channel("q")[:, joint]),
                        color=color,
                        linewidth=1.2,
                        zorder=ARM_ZORDERS[arm],
                    )
                    if arm in REFERENCE_ZORDERS:
                        ax.plot(
                            log.times,
                            np.degrees(log.channel("q_ref")[:, joint]),
                            color=color,
                            linewidth=1.0,
                            linestyle="--",
                            zorder=REFERENCE_ZORDERS[arm],
                        )
                else:
                    ax.plot(
                        log.times, log.channel("tau")[:, joint], color=color, linewidth=1.0, zorder=ARM_ZORDERS[arm]
                    )
            if channel == "torques":
                ax.set_ylim(*limits[joint])
            ax.set_xlim(0, 20)
        axes[-1, col].set_xlabel("time (s)", color=TEXT_COLOR)
    if spans:
        axes[0, 0].set_ylabel("tip force (N)", color=TEXT_COLOR)
    unit = "deg" if channel == "joints" else "N m"
    for joint in range(2):
        axes[rows - 2 + joint, 0].set_ylabel(f"joint {joint + 1} ({unit})", color=TEXT_COLOR)
    handles = arm_handles(references=channel == "joints")
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def arm_points(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """The base, every joint, and the tip of the arm at ``q``."""
    arm = skeleton().clone()
    for link, angle in zip(arm.links[1:], q, strict=True):
        link.q = float(angle)
    compute_forward_kinematics(arm)
    return np.array([[0.0, 0.0]] + [[link.xe, link.ye] for link in arm.links[1:]])


def plot_paths(scenario: str) -> Figure:
    """Each arm's tip path in the plane: a row per arm, a column per tracker and reference velocity.

    Each panel draws the arm at its initial posture, the taught path, the goal, the
    tip's path, and, for each disturbance, an arrow from where the tip is at its
    onset along the mean force over it.
    """
    words = SCENARIOS[scenario]
    spans = spans_of(scenario)
    start = start_of(scenario)
    fig = Figure(
        figsize=(4.0 * len(COLUMNS) + 0.6, 2.5 * len(ARMS) + 1.4), facecolor=SURFACE_COLOR, layout="constrained"
    )
    fig.suptitle(
        f"The tip's path {words}, joint PD, the torque every 10 ms"
        + (", and the force at the start of each disturbance" if spans else ""),
        color="#0b0b0b",
    )
    axes = np.asarray(fig.subplots(len(ARMS), len(COLUMNS), sharex=True, sharey=True)).reshape(len(ARMS), len(COLUMNS))
    taught_hand = taught()[1]
    first = log_of(scenario, ARMS[0][1], ARMS[0][2], "reference", "scaled", start)
    task = Task.from_dict(first.extra["playback"]["task"])
    target, radius = task.require_target(), task.tolerance or 0.0
    for row, (name, tag, arm, color, _) in enumerate(ARMS):
        for col, (tracker, velocity) in enumerate(COLUMNS):
            ax = axes[row, col]
            style(ax)
            log = log_of(scenario, tag, arm, velocity, tracker, start)
            q = log.channel("q")
            joints = arm_points(q[0])
            ax.plot(joints[:, 0], joints[:, 1], color=INITIAL_COLOR, linewidth=5.0, alpha=0.6, solid_capstyle="round")
            ax.plot(joints[:, 0], joints[:, 1], "o", color=INITIAL_COLOR, markersize=4)
            ax.plot(taught_hand[:, 0], taught_hand[:, 1], color=TAUGHT_COLOR, linewidth=4.0, zorder=1)
            ax.add_patch(Circle(tuple(target), radius, fill=False, color=TEXT_COLOR, linewidth=1.0, zorder=2))
            hand = endpoint_positions(skeleton(), q)
            ax.plot(hand[:, 0], hand[:, 1], color=color, linewidth=1.0, zorder=3)
            times = np.asarray(log.times)
            force = log.channel("ext_force") if spans else np.zeros((len(times), 2))
            for onset, end in spans:
                acting = (times > onset - 1e-9) & (times < end - 1e-9)
                first_step = int(np.flatnonzero(acting)[0])
                ax.add_patch(
                    FancyArrow(
                        *hand[first_step],
                        *(FORCE_SCALE * force[acting].mean(axis=0)),
                        width=0.006,
                        length_includes_head=True,
                        head_width=0.03,
                        head_length=0.03,
                        color=PUSH_COLOR,
                        zorder=4,
                    )
                )
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(column_title(tracker, velocity), color="#0b0b0b", fontsize=10)
            if row == len(ARMS) - 1:
                ax.set_xlabel("x (m)", color=TEXT_COLOR)
            if col == 0:
                ax.set_ylabel(f"{name}\ny (m)", color=TEXT_COLOR)
    handles = [
        *arm_handles(references=False),
        Line2D([], [], color=TAUGHT_COLOR, linewidth=4.0, label="the taught path"),
        Line2D([], [], color=INITIAL_COLOR, linewidth=5.0, alpha=0.6, label="the initial posture"),
    ]
    if spans:
        handles.append(
            Line2D([], [], color=PUSH_COLOR, linewidth=2.0, label=f"mean force, {1000 * FORCE_SCALE:g} mm per N")
        )
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, labelcolor=TEXT_COLOR)
    return fig


# ---------------------------------------------------------------------------- tables


def print_tables(*, logs: bool) -> None:
    if logs:
        print_identity()
    print_speeds()
    print_completion()
    print_disturbances()
    print_pushes()
    print_offsets()


def label(tracker: str, velocity: str) -> str:
    return f"{TRACKERS[tracker][2]}, {VELOCITIES[velocity]}"


def cells(runs: list[dict[str, Any]], key: str, scale: float = 1.0, digits: int = 1) -> str:
    return " / ".join("\N{EN DASH}" if not np.isfinite(r[key]) else f"{r[key] * scale:,.{digits}f}" for r in runs)


def print_identity() -> None:
    """Undisturbed, from the demonstrated start, how far the progress replay's arm is from the replay's."""
    print("\n== Undisturbed: the largest difference between the progress replay's and the replay's joint angles (deg)")
    for tracker, velocity in COLUMNS:
        tag = ARMS[0][1]
        replay = log_of("nominal", tag, "replay", velocity, tracker).channel("q")
        progress = log_of("nominal", tag, "progress", velocity, tracker).channel("q")
        print(f"{label(tracker, velocity)}: {np.degrees(np.abs(progress - replay).max()):.2e}")


def print_speeds() -> None:
    print("\n== The references' peak joint speed (deg/s), ESN #1 / ESN #2 / replay / progress replay")
    print(f"| Scenario | {' | '.join(label(*c) for c in COLUMNS)} |")
    print(f"| --- |{' --- |' * len(COLUMNS)}")
    for scenario in SCENARIOS:
        if scenario == "offsets":
            continue
        row = [
            cells(
                [run_of(scenario, tag, arm, velocity, tracker) for _, tag, arm, _, _ in ARMS],
                "peak_reference_speed_dps",
                digits=0,
            )
            for tracker, velocity in COLUMNS
        ]
        print(f"| {SCENARIOS[scenario]} | {' | '.join(row)} |")


def print_completion() -> None:
    print("\n== Completion: arrival (s), * if the hand leaves the goal in its hold; final distance (mm); peak torque")
    print(
        "   (N m); peak hand speed (m/s); RMS distance from the taught path (mm); ESN #1 / ESN #2 / replay / progress"
    )
    for scenario in SCENARIOS:
        if scenario == "offsets":
            continue
        print(f"-- {scenario}")
        print("| Tracker | Arrival (s) | Final distance (mm) | Peak torque (N m) | Peak hand speed (m/s) | Path (mm) |")
        print("| --- | --- | --- | --- | --- | --- |")
        for tracker, velocity in COLUMNS:
            runs = [run_of(scenario, tag, arm, velocity, tracker) for _, tag, arm, _, _ in ARMS]
            arrival = " / ".join(
                ("\N{EN DASH}" if not np.isfinite(r["arrival_time_s"]) else f"{r['arrival_time_s']:.2f}")
                + ("*" if r["left_goal"] else "")
                for r in runs
            )
            print(
                f"| {label(tracker, velocity)} | {arrival} | {cells(runs, 'final_distance_m', 1000.0)} |"
                f" {cells(runs, 'peak_torque_nm')} | {cells(runs, 'peak_hand_speed_mps', digits=2)} |"
                f" {cells(runs, 'taught_path_rmse_m', 1000.0)} |"
            )
    print("\n== Under the block, from 4.5 s to 6 s: the peak joint torque (N m) and the block's peak force (N)")
    print("| Tracker | Peak torque (N m) | Block's force (N) |")
    print("| --- | --- | --- |")
    for tracker, velocity in COLUMNS:
        runs = [run_of("block", tag, arm, velocity, tracker) for _, tag, arm, _, _ in ARMS]
        print(
            f"| {label(tracker, velocity)} | {cells(runs, 'peak_torque_nm')} | {cells(runs, 'peak_external_force_n')} |"
        )


def disturbances() -> list[tuple[str, str, str]]:
    """Every disturbance: its scenario, the suffix of its columns in metrics.csv, and its name."""
    out = [("block", "", "block")] + [("pushes", f"_{k}", f"push {k}") for k in (1, 2, 3)]
    for log in LOGS:
        n = len(spans_of(f"recorded{log}"))
        out += [(f"recorded{log}", f"_{k}", f"log {log}, push {k}") for k in range(1, n + 1)]
    return out


def print_disturbances() -> None:
    for key, name, scale, digits in MEASURES:
        arms = [a for a in ARMS if key != "following_gain" or a[2] != "replay"]
        print(f"\n== {name}, {' / '.join(a[0] for a in arms)}")
        for scenario in ("block", "pushes", *(f"recorded{log}" for log in LOGS)):
            items = [(s, suffix, n) for s, suffix, n in disturbances() if s == scenario]
            print(f"-- {scenario}")
            print(f"| Tracker | {' | '.join(n for _, _, n in items)} |")
            print(f"| --- |{' --- |' * len(items)}")
            for tracker, velocity in COLUMNS:
                row = []
                for _, suffix, _ in items:
                    runs = [run_of(scenario, tag, arm, velocity, tracker) for _, tag, arm, _, _ in arms]
                    row.append(cells(runs, f"{key}{suffix}", scale, digits))
                print(f"| {label(tracker, velocity)} | {' | '.join(row)} |")


def return_cap(scenario: str, suffix: str) -> float:
    """How long after a disturbance its return is watched (s): until the next one, or the run's end."""
    spans = spans_of(scenario)
    k = int(suffix[1:]) if suffix else 1
    nxt = spans[k][0] if k < len(spans) else DURATION
    return nxt - spans[k - 1][1]


def print_pushes() -> None:
    print("\n== Over the 13 pushes (report 011's three and the ten by hand): the median (least to most); not back")
    print("| Tracker | Following | Ringing (mm) | Back within 5 mm (s); not back | Farthest (mm) |")
    print("| --- | --- | --- | --- | --- |")
    pushes = [d for d in disturbances() if d[0] != "block"]
    for tracker, velocity in COLUMNS:
        row = []
        for key, _, scale, digits in MEASURES:
            parts = []
            for _, tag, arm, _, _ in ARMS:
                if key == "following_gain" and arm == "replay":
                    continue
                values, missing = [], 0
                for scenario, suffix, _ in pushes:
                    v = run_of(scenario, tag, arm, velocity, tracker)[f"{key}{suffix}"]
                    if not np.isfinite(v):
                        missing += 1
                        v = return_cap(scenario, suffix)
                    values.append(v * scale)
                part = f"{np.median(values):.{digits}f} ({min(values):.{digits}f} to {max(values):.{digits}f})"
                parts.append(part + (f"; {missing}" if key == "return_time_s" else ""))
            row.append(" / ".join(parts))
        print(f"| {label(tracker, velocity)} | {' | '.join(row)} |")


def print_offsets() -> None:
    print(
        "\n== From the 49 starts: successes; peak torque, median (largest); RMS distance from the taught path, median"
    )
    print("| Tracker | ESN #1 | ESN #2 | replay | progress replay |")
    print("| --- | --- | --- | --- | --- |")
    for tracker, velocity in COLUMNS:
        row = []
        for _, tag, arm, _, _ in ARMS:
            runs = [r for r in rows_of("offsets", tag, velocity, tracker) if r["arm"] == arm]
            torque = [r["peak_torque_nm"] for r in runs]
            path = [1000 * r["taught_path_rmse_m"] for r in runs]
            row.append(
                f"{sum(r['success'] for r in runs)}/{len(runs)}; {np.median(torque):.1f} ({max(torque):.1f}) N m;"
                f" {np.median(path):.1f} mm"
            )
        print(f"| {label(tracker, velocity)} | {' | '.join(row)} |")


if __name__ == "__main__":
    for scenario in SCENARIOS:
        if config_of(scenario)["evaluation"]["duration"] != DURATION:
            sys.exit(f"{scenario}: the runs' duration is not DURATION")
    main()
