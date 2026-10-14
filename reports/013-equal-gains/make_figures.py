# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 013.

    uv run python reports/013-equal-gains/make_figures.py
    uv run python reports/013-equal-gains/make_figures.py --logs

From the copies of the runs under ``results/`` (configurations, run records, and
metrics), and of reports 011's and 012's runs, whose joint PD at the damping ratio 0.4
is compared here, it draws into ``results/summary``:

- ``stability.png``: how much the tracking error of a still arm grows or shrinks over
  one 10 ms period of the held torque, against joint 2's angle, with the equal gains
  and with report 011's at 0.4 (Section 3.1);

and prints the tables of the report: each joint's natural frequency and damping
ratio, the modes of the two joints together, the completion, and the measures after
each disturbance. With ``--logs``, it reads the runs' logs, which are not kept in Git,
from the runs under the storage root, measures the chatter of joint 2's torque, and
also draws:

- ``chatter.png``: the first 4 s undisturbed, joint 2's angle and torque, and the
  torque over 0.2 s, period by period (Section 3.2);
- ``offsets.png``: from the 49 starts, when the chatter ends and the peak torque,
  against joint 2's start angle (Section 3.3);
- ``<scenario>_joints.png``, ``<scenario>_torques.png``, ``<scenario>_paths.png``:
  the joint angles and the joint torques over time, and the tip's path in the plane,
  of ESN #1, ESN #2, and the replay, with the equal gains and with report 011's at
  0.4, each given the reference velocity and a zero reference velocity (Sections 3.3
  to 3.5). Over time, the ESNs' lines are drawn over the replay's.
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
from skelarm import Skeleton, StateLog, Task, compute_forward_kinematics, compute_mass_matrix

from arm_esn_ctrl.demonstrations import endpoint_positions, resample_joint_angles
from arm_esn_ctrl.disturbances import disturbance_spans
from arm_esn_ctrl.metrics import distances_to_path
from arm_esn_ctrl.storage import resolve_run_path
from arm_esn_ctrl.tracking import TrackerConfig, error_dynamics, gain_posture, tracking_gains

REPORT = Path(__file__).resolve().parent
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"
# The earlier runs of the same scenarios, with joint PD at damping ratios 0.2, 0.4, and 0.8: their copies.
EARLIER = {
    "011": REPORT.parent / "011-following-the-arm" / "results",
    "012": REPORT.parent / "012-pushes-by-hand" / "results",
}
# The take, as report 005 copied it: what the ESNs were trained on, and what the replay replays.
TAKE_RUN = REPORT.parent / "005-tuned-on-the-robot" / "data" / "20261008-170034-reach_manual_v2"
DT = 0.01  # the ESNs' period (s)
PERIOD = 0.01  # the tracker computes its torque this often and holds it between (s)
STEP = 0.002  # the simulation's step (s)
DURATION = 24.0  # s: the runs' task, after the warm-up

# The trackers compared: the equal gains, and report 011's joint PD at the damping ratio 0.4, its gains scaled by
# each joint's inertia: name, tracker setting (the runs' directory), and label.
KP, KD = 80.0, 3.0  # N m/rad, N m s/rad: both joints
OMEGA, DAMPING = 20.0, 0.4
TRACKERS = {
    "equal": ("pd_kp80_kd3", "kp 80, kd 3 for both joints"),
    "scaled": ("pd_w20_z0.4", "ω 20, ζ 0.4 (reports 011, 012)"),
}
VELOCITIES = {"reference": "reference velocity", "zero": "zero reference velocity"}
COLUMNS = [(velocity, tracker) for velocity in VELOCITIES for tracker in TRACKERS]
# The scenarios: the earlier report that ran them, and their words.
SCENARIOS = {
    "nominal": ("011", "undisturbed"),
    "block": ("011", "under the block"),
    "pushes": ("011", "under the three pushes"),
    "offsets": ("011", "from the start offset by -15° in both joints"),
    "recorded000": ("012", "under log 000's pushes by hand"),
    "recorded001": ("012", "under log 001's pushes by hand"),
    "recorded002": ("012", "under log 002's pushes by hand"),
}
OFFSET_ORIGIN = "demo_00_filtered -15,-15 deg"  # the offset start drawn over time: the farthest into the chatter
LOGS = ("000", "001", "002")
# The arms compared: name, the tag of the runs, the arm in them, color, and marker. The replay is the same in every
# ESN's runs; it is read from ESN #1's.
ARMS = [
    ("ESN #1", "lr0.03_sr1.2_is1.5_seed8", "esn", "#2a78d6", "o"),
    ("ESN #2", "lr0.03_sr1.2_is1_seed2", "esn", "#5b2a86", "s"),
    ("replay", "lr0.03_sr1.2_is1.5_seed8", "replay", "#eb6834", "D"),
]
# The measures after each disturbance: the column in metrics.csv, its label, its scale, and its digits.
MEASURES = [
    ("following_gain", "following gain", 1.0, 2),
    ("ringing_m", "ringing over the next 2 s (mm)", 1000.0, 1),
    ("return_time_s", "back within 5 mm of the path (s)", 1.0, 2),
    ("off_path_m", "farthest from the path (mm)", 1000.0, 0),
]
# The chatter: joint 2's held torque turning back and forth by more than this from one period to the next (N m).
CHATTER_STEP = 10.0
AFTER_CHATTER = 3.1  # s: every run's chatter has ended by then

TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
DISTURBANCE_COLOR = "#f0efec"
TAUGHT_COLOR = "#b9b7b0"
INITIAL_COLOR = "#a3a29d"
PUSH_COLOR = "#dc0000"  # the force arrows in the plane, red as the robot app's drag arrow
FORCE_SCALE = 0.025  # m/N: the length of a force arrow in the plane, 25 mm per newton
# The layers of the lines over time: the ESNs' arms on top, their references (dashed) under them, the replay below.
ARM_ZORDERS = {"esn": 3.0, "replay": 2.0}
REFERENCE_ZORDER = 2.5
# The stability figure's lines: the equal gains, held and computed every step, and report 011's gains, held.
STABILITY_LINES = [
    ("kp 80, kd 3 for both joints, the torque held 10 ms", "#0b0b0b", "-"),
    ("kp 80, kd 3 for both joints, the torque every 2 ms", "#0b0b0b", ":"),
    ("ω 20, ζ 0.4 (reports 011, 012), the torque held 10 ms", "#a3a29d", "-"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", action="store_true", help="also analyze the run logs in the storage root")
    args = parser.parse_args()

    SUMMARY.mkdir(parents=True, exist_ok=True)
    figures = {"stability.png": plot_stability()}
    if args.logs:
        figures |= {"chatter.png": plot_chatter(), "offsets.png": plot_offsets()}
        for scenario in SCENARIOS:
            name = "offset" if scenario == "offsets" else scenario
            for channel in ("joints", "torques"):
                figures[f"{name}_{channel}.png"] = plot_series(scenario, channel)
            figures[f"{name}_paths.png"] = plot_paths(scenario)
    for name, fig in figures.items():
        fig.savefig(SUMMARY / name, dpi=150)
    print_tables(logs=args.logs)
    print(f"\nWrote {', '.join(figures)} to {SUMMARY}")


# ---------------------------------------------------------------------------- the arm and its gains


@functools.cache
def skeleton() -> Skeleton:
    return Skeleton.from_toml(TAKE_RUN / "config.toml")


@functools.cache
def taught() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The take filtered at 2 Hz, what the ESNs were trained on, every 10 ms: its joint angles and its hand path."""
    q = resample_joint_angles(StateLog.load(TAKE_RUN / "demo_00_filtered.sklog.npz"), DT)[1]
    return q, endpoint_positions(skeleton(), q)


def end_posture() -> NDArray[np.float64]:
    """Where report 011's gains were set, as the runs set them: the take's final posture."""
    return gain_posture([taught()[0]])


def gains(tracker: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A tracker's per-joint gains ``(kp, kd)``."""
    if tracker == "equal":
        return np.array([KP, KP]), np.array([KD, KD])
    config = TrackerConfig("pd", OMEGA, 0.02, damping=DAMPING)
    return tracking_gains(config, skeleton(), end_posture())


def mass(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """The arm's mass matrix at the joint angles ``q``."""
    model = skeleton().clone()
    for link, angle in zip(model.links[1:], q, strict=True):
        link.q = float(angle)
    compute_forward_kinematics(model)
    return compute_mass_matrix(model)


def continuous_modes(m: NDArray[np.float64], kp: NDArray[np.float64], kd: NDArray[np.float64]) -> NDArray[Any]:
    """The eigenvalues s (1/s) of the error ``M ë + Kd ė + Kp e = 0`` of the arm at rest, the torque continuous."""
    n = len(kp)
    inverse = np.linalg.inv(m)
    system = np.block([[np.zeros((n, n)), np.eye(n)], [-inverse @ np.diag(kp), -inverse @ np.diag(kd)]])
    return np.linalg.eigvals(system)


def held_modes(m: NDArray[np.float64], kp: NDArray[np.float64], kd: NDArray[np.float64], period: float) -> NDArray[Any]:
    """The eigenvalues z of the tracking error's map over one period, the torque computed at its start and held.

    The arm is a pure inertia ``M`` at rest, so that over a period the held torque
    ``u = -Kp e - Kd ė`` changes the error by ``T ė + T²/2 M⁻¹u`` and its rate by
    ``T M⁻¹u``. The error shrinks period by period if every ``|z|`` is below 1, and
    grows if one is above.
    """
    n = len(kp)
    inverse = np.linalg.inv(m)
    torque = -np.hstack([np.diag(kp), np.diag(kd)])
    gain = np.vstack([period**2 / 2 * inverse, period * inverse])
    free = np.block([[np.eye(n), period * np.eye(n)], [np.zeros((n, n)), np.eye(n)]])
    return np.linalg.eigvals(free + gain @ torque)


def largest_per_period(q2: float, tracker: str, period: float) -> float:
    """The largest |z| at joint 2's angle ``q2`` (rad), as a factor over one 10 ms period."""
    z = held_modes(mass(np.array([0.0, q2])), *gains(tracker), period)
    return float(np.abs(z).max()) ** (PERIOD / period)


@functools.cache
def stability_curve(tracker: str, period: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Joint 2's angles (deg) from 0 to 180, and the largest |z| over 10 ms at each."""
    angles = np.arange(0.0, 180.0 + 1e-9, 0.05)
    return angles, np.array([largest_per_period(np.radians(a), tracker, period) for a in angles])


def unstable_below(tracker: str, period: float) -> float | None:
    """The joint 2 angle (deg) below which the error grows, or None if it shrinks at every angle."""
    angles, largest = stability_curve(tracker, period)
    if (largest < 1.0).all():
        return None
    return float(angles[largest >= 1.0].max())


def as_mode(s: complex) -> tuple[float, float]:
    """A mode's natural frequency (rad/s) and damping ratio from its eigenvalue s."""
    return abs(s), -s.real / abs(s)


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


def results_of(scenario: str, tracker: str) -> Path:
    """Where a tracker's runs of a scenario are copied: here, or in the earlier report."""
    return RESULTS if tracker == "equal" else EARLIER[SCENARIOS[scenario][0]]


def run_name(scenario: str, tag: str, velocity: str, tracker: str) -> str:
    """The name of the one run of a configuration."""
    stem = f"{scenario}_robot_{tag}_{velocity}_velocity_filtered"
    matches = sorted(p.name for p in results_of(scenario, tracker).glob(f"*-{stem}") if p.is_dir())
    if len(matches) != 1:
        msg = f"expected one run of {stem}, found {matches}"
        raise FileNotFoundError(msg)
    return matches[0]


def rows_of(scenario: str, tag: str, velocity: str, tracker: str) -> list[dict[str, Any]]:
    """The metrics of a run with one tracker: every row with the equal gains, the damping ratio 0.4's otherwise."""
    rows = read_csv(results_of(scenario, tracker) / run_name(scenario, tag, velocity, tracker) / "metrics.csv")
    return [r for r in rows if tracker == "equal" or r["damping"] == DAMPING]


def run_of(scenario: str, tag: str, arm: str, velocity: str, tracker: str, start: int = 0) -> dict[str, Any]:
    """One arm's run from one start, with one tracker."""
    return next(r for r in rows_of(scenario, tag, velocity, tracker) if r["arm"] == arm and r["start"] == start)


def offset_start() -> int:
    """The index of the offset start drawn over time."""
    rows = rows_of("offsets", ARMS[0][1], "reference", "equal")
    return next(int(r["start"]) for r in rows if r["origin"] == OFFSET_ORIGIN)


def start_of(scenario: str) -> int:
    return offset_start() if scenario == "offsets" else 0


def log_of(scenario: str, tag: str, arm: str, velocity: str, tracker: str, start: int = 0) -> StateLog:
    """One arm's log from a run under the storage root."""
    run = resolve_run_path(f"results/{run_name(scenario, tag, velocity, tracker)}")
    return StateLog.load(run / TRACKERS[tracker][0] / f"{arm}_{start:02d}.sklog.npz")


@functools.cache
def config_of(scenario: str) -> dict[str, Any]:
    with (RESULTS / run_name(scenario, ARMS[0][1], "reference", "equal") / "config.toml").open("rb") as f:
        return tomllib.load(f)


def spans_of(scenario: str) -> list[tuple[float, float]]:
    """When each disturbance of a scenario acts."""
    return disturbance_spans(config_of(scenario).get("disturbance"))


def chatter(log: StateLog) -> tuple[float, float, float]:
    """When joint 2's torque chatters: its first and last chattering period's start (s), and the time it chatters (s).

    A period chatters when joint 2's held torque turns back and forth by more than
    CHATTER_STEP from the period before to it and from it to the period after. NaN
    when none does.
    """
    every = round(PERIOD / STEP)
    times, torque = np.asarray(log.times)[::every], log.channel("tau")[::every, 1]
    change = np.diff(torque)
    turns = (change[1:] * change[:-1] < 0) & (np.abs(change[1:]) > CHATTER_STEP) & (np.abs(change[:-1]) > CHATTER_STEP)
    at = times[1:-1][turns]
    if not len(at):
        return float("nan"), float("nan"), 0.0
    return float(at.min()), float(at.max()), PERIOD * len(at)


@functools.cache
def chatter_of(scenario: str, tag: str, arm: str, velocity: str, tracker: str, start: int = 0) -> tuple[float, ...]:
    """A run's chatter (see :func:`chatter`), and its peak joint torque until the chatter ends (N m)."""
    log = log_of(scenario, tag, arm, velocity, tracker, start)
    first, last, total = chatter(log)
    if not np.isfinite(last):
        return first, last, total, float("nan")
    until = np.asarray(log.times) <= last + PERIOD
    return first, last, total, float(np.abs(log.channel("tau")[until]).max())


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def column_title(velocity: str, tracker: str) -> str:
    return f"{TRACKERS[tracker][1]}\n{VELOCITIES[velocity]}"


# ---------------------------------------------------------------------------- figures


def plot_stability() -> Figure:
    """The largest |z| over one 10 ms period against joint 2's angle, with the reach's angles and the starts marked."""
    fig = Figure(figsize=(11, 5.2), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "Joint PD on the still arm: how much the tracking error grows (above 1) or shrinks (below 1) over 10 ms,"
        "\nits largest mode, against joint 2's angle (the arm's inertia depends on it alone)",
        color="#0b0b0b",
    )
    ax = fig.subplots()
    style(ax)
    q2 = np.degrees(taught()[0][:, 1])
    ax.axvspan(q2.min(), q2.max(), color=DISTURBANCE_COLOR, zorder=0, label="joint 2 over the take")
    for (label, color, linestyle), (tracker, period) in zip(
        STABILITY_LINES, [("equal", PERIOD), ("equal", STEP), ("scaled", PERIOD)], strict=True
    ):
        angles, largest = stability_curve(tracker, period)
        ax.plot(angles, largest, color=color, linestyle=linestyle, linewidth=1.6, label=label, zorder=3)
    ax.axhline(1.0, color=PUSH_COLOR, linewidth=1.0, zorder=2)
    below = unstable_below("equal", PERIOD)
    # Each mark: its angle, its words, their height, and their side of the line.
    marks = [(q2[0], "the start", 4.1, "left"), (q2[-1], "the target", 3.6, "left")]
    if below is not None:
        marks.append((below, "the bound", 3.6, "right"))
    for angle, text, height, side in marks:
        ax.axvline(angle, color=TEXT_COLOR, linewidth=0.8, linestyle="--", zorder=1)
        ax.annotate(
            f"{text}, {angle:.1f}°",
            xy=(angle, height),
            xytext=(4 if side == "left" else -4, 0),
            textcoords="offset points",
            ha=side,
            color=TEXT_COLOR,
            fontsize=9,
        )
    ax.annotate("grows", xy=(178, 1.0), xytext=(0, 4), textcoords="offset points", ha="right", color=PUSH_COLOR)
    ax.annotate("shrinks", xy=(178, 1.0), xytext=(0, -30), textcoords="offset points", ha="right", color=PUSH_COLOR)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, None)
    ax.set_xticks(np.arange(0, 181, 15))
    ax.set_xlabel("joint 2's angle (deg)", color=TEXT_COLOR)
    ax.set_ylabel("largest |z| over 10 ms", color=TEXT_COLOR)
    ax.legend(frameon=False, labelcolor=TEXT_COLOR, loc="upper right")
    return fig


def plot_chatter() -> Figure:
    """The first 4 s undisturbed with the equal gains: joint 2's angle and torque, and the torque over 0.2 s."""
    zoom = (0.5, 0.7)
    fig = Figure(figsize=(14, 9.5), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "Undisturbed, joint PD at kp 80, kd 3 for both joints, the torque every 10 ms: the first 4 s;"
        "\nESN #1 (blue), ESN #2 (purple), and the replay (orange); dashed: each ESN's reference; thick gray: the"
        " taught motion",
        color="#0b0b0b",
    )
    axes = fig.subplots(4, len(VELOCITIES), height_ratios=[1.0, 1.0, 1.0, 1.0])
    q_taught = np.degrees(taught()[0])
    t_taught = DT * np.arange(len(q_taught))
    for col, velocity in enumerate(VELOCITIES):
        logs = {name: log_of("nominal", tag, arm, velocity, "equal") for name, tag, arm, _, _ in ARMS}
        for row in range(4):
            ax = axes[row, col]
            style(ax)
            joint = 0 if row == 0 else 1
            if row < 2:
                ax.plot(t_taught, q_taught[:, joint], color=TAUGHT_COLOR, linewidth=4.0, zorder=1)
            for name, _, arm, color, _ in ARMS:
                log = logs[name]
                times = np.asarray(log.times)
                if row < 2:
                    ax.plot(
                        times,
                        np.degrees(log.channel("q")[:, joint]),
                        color=color,
                        linewidth=1.0,
                        zorder=ARM_ZORDERS[arm],
                    )
                    if arm == "esn":
                        ax.plot(
                            times,
                            np.degrees(log.channel("q_ref")[:, joint]),
                            color=color,
                            linewidth=1.0,
                            linestyle="--",
                            zorder=REFERENCE_ZORDER,
                        )
                else:
                    ax.plot(
                        times,
                        log.channel("tau")[:, 1],
                        color=color,
                        linewidth=0.8 if row == 2 else 1.2,
                        drawstyle="steps-post",
                        zorder=ARM_ZORDERS[arm],
                    )
            if row == 3:
                ax.set_xlim(*zoom)
                ax.set_xticks(np.arange(zoom[0], zoom[1] + 1e-9, PERIOD), minor=True)
                shown = [
                    logs[name].channel("tau")[(times >= zoom[0]) & (times <= zoom[1]), 1]
                    for name, times in ((name, np.asarray(logs[name].times)) for name, *_ in ARMS)
                ]
                low, high = min(v.min() for v in shown), max(v.max() for v in shown)
                ax.set_ylim(low - 0.1 * (high - low), high + 0.1 * (high - low))
                ax.grid(which="minor", color=GRID_COLOR, linewidth=0.5)
                ax.set_xlabel("time (s), the minor grid every 10 ms", color=TEXT_COLOR)
            else:
                ax.set_xlim(0, 4)
                ax.axvspan(*zoom, color=DISTURBANCE_COLOR, zorder=0)
            if row == 0:
                ax.set_title(VELOCITIES[velocity], color="#0b0b0b", fontsize=10)
            if col == 0:
                ax.set_ylabel(
                    [
                        "joint 1 (deg)",
                        "joint 2 (deg)",
                        "joint 2's torque (N m)",
                        "joint 2's torque (N m),\nshaded span",
                    ][row],
                    color=TEXT_COLOR,
                )
    handles = [Line2D([], [], color=color, linewidth=1.4, label=name) for name, _, _, color, _ in ARMS]
    handles += [
        Line2D([], [], color=TEXT_COLOR, linewidth=1.0, linestyle="--", label="an ESN's reference"),
        Line2D([], [], color=TAUGHT_COLOR, linewidth=4.0, label="the taught motion"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_offsets() -> Figure:
    """From the 49 starts: when the chatter ends, and the peak torque, against joint 2's start angle."""
    fig = Figure(figsize=(14, 7.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "From the 49 starts, joint PD, the torque every 10 ms: ESN #1 (blue), ESN #2 (purple), and the replay (orange);"
        "\nfilled: kp 80, kd 3 for both joints; hollow: ω 20, ζ 0.4 (report 011)",
        color="#0b0b0b",
    )
    axes = fig.subplots(2, len(VELOCITIES), sharex=True, sharey="row")
    shifts = {name: shift for (name, *_), shift in zip(ARMS, (-0.8, 0.0, 0.8), strict=True)}
    for col, velocity in enumerate(VELOCITIES):
        for row in range(2):
            ax = axes[row, col]
            style(ax)
            for name, tag, arm, color, marker in ARMS:
                for tracker in TRACKERS:
                    if row == 0 and tracker == "scaled":
                        continue  # no chatter
                    runs = rows_of("offsets", tag, velocity, tracker)
                    starts = [r for r in runs if r["arm"] == arm]
                    log = {r["start"]: log_of("offsets", tag, arm, velocity, tracker, int(r["start"])) for r in starts}
                    x = [np.degrees(log[r["start"]].channel("q")[0, 1]) + shifts[name] for r in starts]
                    if row == 0:
                        y = [chatter_of("offsets", tag, arm, velocity, tracker, int(r["start"]))[1] for r in starts]
                    else:
                        y = [r["peak_torque_nm"] for r in starts]
                    filled = tracker == "equal"
                    ax.plot(
                        x,
                        y,
                        linestyle="none",
                        marker=marker,
                        markersize=4,
                        color=color,
                        markerfacecolor=color if filled else "none",
                        zorder=ARM_ZORDERS[arm],
                    )
            if row == 0:
                ax.set_title(VELOCITIES[velocity], color="#0b0b0b", fontsize=10)
                ax.set_ylim(0, None)
            else:
                ax.set_yscale("log")
                ax.set_xlabel("joint 2's start angle (deg)", color=TEXT_COLOR)
            if col == 0:
                ax.set_ylabel(["the chatter's last period (s)", "peak joint torque (N m)"][row], color=TEXT_COLOR)
    handles = [
        Line2D([], [], linestyle="none", marker=marker, markersize=5, color=color, label=name)
        for name, _, _, color, marker in ARMS
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_series(scenario: str, channel: str) -> Figure:
    """The joint angles or torques of the three arms over time: a column per tracker and reference velocity.

    Under a disturbance, a first row draws each arm's tip force. With the equal
    gains, the torques' scale leaves out the chatter at the start, whose peak the
    panel states.
    """
    words = SCENARIOS[scenario][1]
    spans = spans_of(scenario)
    rows = (1 if spans else 0) + 2
    fig = Figure(figsize=(17, 2.2 + 2.6 * rows), facecolor=SURFACE_COLOR, layout="constrained")
    what = "Joint angles" if channel == "joints" else "Joint torques"
    fig.suptitle(
        f"{what} {words}, joint PD, the torque every 10 ms: ESN #1 (blue), ESN #2 (purple), and the replay (orange)"
        + ("\ndashed: each ESN's reference; thick gray: the taught motion" if channel == "joints" else ""),
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
        column: {name: log_of(scenario, tag, arm, column[0], column[1], start) for name, tag, arm, _, _ in ARMS}
        for column in COLUMNS
    }
    # With the equal gains, the chatter is left out of the torques' scale: each arm's samples after its last
    # chattering period count.
    ends = {
        column: {name: chatter_of(scenario, tag, arm, column[0], column[1], start)[1] for name, tag, arm, _, _ in ARMS}
        for column in COLUMNS
    }
    limits = []
    if channel == "torques":
        for joint in range(2):
            values = []
            for column in COLUMNS:
                for name, log in logs[column].items():
                    end = ends[column][name]
                    after = np.asarray(log.times) > (end + PERIOD if np.isfinite(end) else -np.inf)
                    values.append(log.channel("tau")[after, joint])
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
                log = logs[column][name]
                force = np.linalg.norm(log.channel("ext_force"), axis=1)
                axes[0, col].plot(log.times, force, color=color, linewidth=1.0, zorder=ARM_ZORDERS[arm])
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
                    if arm == "esn":
                        ax.plot(
                            log.times,
                            np.degrees(log.channel("q_ref")[:, joint]),
                            color=color,
                            linewidth=1.0,
                            linestyle="--",
                            zorder=REFERENCE_ZORDER,
                        )
                else:
                    ax.plot(
                        log.times, log.channel("tau")[:, joint], color=color, linewidth=1.0, zorder=ARM_ZORDERS[arm]
                    )
            if channel == "torques":
                ax.set_ylim(*limits[joint])
                chattered = [
                    (ends[column][name], np.abs(logs[column][name].channel("tau")[:, joint]).max())
                    for name in logs[column]
                    if np.isfinite(ends[column][name])
                ]
                if chattered:
                    ax.annotate(
                        f"off the scale until {max(e for e, _ in chattered) + PERIOD:.1f} s:"
                        f"\nchatter up to {max(p for _, p in chattered):,.0f} N m",
                        xy=(0.98, 0.95),
                        xycoords="axes fraction",
                        ha="right",
                        va="top",
                        color=TEXT_COLOR,
                        fontsize=8,
                        bbox={"facecolor": SURFACE_COLOR, "edgecolor": GRID_COLOR, "boxstyle": "round"},
                        zorder=5,
                    )
            ax.set_xlim(0, 20)
        axes[-1, col].set_xlabel("time (s)", color=TEXT_COLOR)
    if spans:
        axes[0, 0].set_ylabel("tip force (N)", color=TEXT_COLOR)
    unit = "deg" if channel == "joints" else "N m"
    for joint in range(2):
        axes[rows - 2 + joint, 0].set_ylabel(f"joint {joint + 1} ({unit})", color=TEXT_COLOR)
    handles = [Line2D([], [], color=color, linewidth=1.4, label=name) for name, _, _, color, _ in ARMS]
    if channel == "joints":
        handles += [
            Line2D([], [], color=TEXT_COLOR, linewidth=1.0, linestyle="--", label="an ESN's reference"),
            Line2D([], [], color=TAUGHT_COLOR, linewidth=4.0, label="the taught motion"),
        ]
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
    words = SCENARIOS[scenario][1]
    spans = spans_of(scenario)
    start = start_of(scenario)
    fig = Figure(
        figsize=(4.4 * len(COLUMNS) + 0.6, 2.7 * len(ARMS) + 1.4), facecolor=SURFACE_COLOR, layout="constrained"
    )
    fig.suptitle(
        f"The tip's path {words}, joint PD, the torque every 10 ms"
        + (", and the force at the start of each disturbance" if spans else ""),
        color="#0b0b0b",
    )
    axes = np.asarray(fig.subplots(len(ARMS), len(COLUMNS), sharex=True, sharey=True)).reshape(len(ARMS), len(COLUMNS))
    taught_hand = taught()[1]
    first = log_of(scenario, ARMS[0][1], ARMS[0][2], "reference", "equal", start)
    task = Task.from_dict(first.extra["playback"]["task"])
    target, radius = task.require_target(), task.tolerance or 0.0
    for row, (name, tag, arm, color, _) in enumerate(ARMS):
        for col, (velocity, tracker) in enumerate(COLUMNS):
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
                ax.set_title(column_title(velocity, tracker), color="#0b0b0b", fontsize=10)
            if row == len(ARMS) - 1:
                ax.set_xlabel("x (m)", color=TEXT_COLOR)
            if col == 0:
                ax.set_ylabel(f"{name}\ny (m)", color=TEXT_COLOR)
    handles = [Line2D([], [], color=color, linewidth=1.4, label=name) for name, _, _, color, _ in ARMS]
    handles += [
        Line2D([], [], color=TAUGHT_COLOR, linewidth=4.0, label="the taught path"),
        Line2D([], [], color=INITIAL_COLOR, linewidth=5.0, alpha=0.6, label="the initial posture"),
    ]
    if spans:
        handles.append(
            Line2D([], [], color=PUSH_COLOR, linewidth=2.0, label=f"mean force, {1000 * FORCE_SCALE:g} mm per N")
        )
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


# ---------------------------------------------------------------------------- tables


def print_tables(*, logs: bool) -> None:
    print_gains()
    print_modes()
    print_completion()
    print_disturbances()
    print_offsets()
    if logs:
        print_chatter()
        print_after_chatter()


def print_gains() -> None:
    posture = end_posture()
    q = taught()[0]
    print(f"\n== Each joint alone (at the take's final posture, {np.degrees(posture).round(2)} deg)")
    print("| Tracker | Joint | kp (N m/rad) | kd (N m s/rad) | Inertia (kg m²) | ω (rad/s) | ζ | Over the take: ω, ζ |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    inertia = np.array([np.diag(mass(p)) for p in q])
    for tracker in TRACKERS:
        kp, kd = gains(tracker)
        omega, zeta = error_dynamics("pd", (kp, kd), skeleton(), posture)
        omegas = np.sqrt(kp / inertia)
        zetas = kd / (2 * np.sqrt(kp * inertia))
        for j in range(2):
            print(
                f"| {TRACKERS[tracker][1]} | {j + 1} | {kp[j]:.2f} | {kd[j]:.3f} | {mass(posture)[j, j]:.4f} |"
                f" {omega[j]:.1f} | {zeta[j]:.2f} | {omegas[:, j].min():.1f} to {omegas[:, j].max():.1f},"
                f" {zetas[:, j].min():.2f} to {zetas[:, j].max():.2f} |"
            )
    print(f"inertia over the take: joint 1 {inertia[:, 0].min():.4f} to {inertia[:, 0].max():.4f}")


def print_modes() -> None:
    posture = end_posture()
    start = taught()[0][0]
    for where, q in (("the take's final posture", posture), ("the start", start)):
        m = mass(q)
        print(f"\n== The two joints together, at {where} ({np.degrees(q).round(2)} deg); M =\n{m.round(5)}")
        print("| Tracker | Torque | Mode | |z| over 10 ms | ω (rad/s) | ζ |")
        print("| --- | --- | --- | ---: | ---: | ---: |")
        for tracker in TRACKERS:
            kp, kd = gains(tracker)
            for s in sorted(continuous_modes(m, kp, kd), key=lambda s: abs(s)):
                if s.imag < -1e-9:
                    continue
                omega, zeta = as_mode(s)
                kind = "oscillates" if s.imag > 1e-9 else "decays"
                decay = np.exp(s.real * PERIOD)
                print(f"| {TRACKERS[tracker][1]} | continuous | {kind} | {decay:.3f} | {omega:.1f} | {zeta:.2f} |")
            for z in sorted(held_modes(m, kp, kd, PERIOD), key=lambda z: -abs(z)):
                if z.imag < -1e-9:
                    continue
                s = np.log(complex(z)) / PERIOD
                omega, zeta = as_mode(complex(s.real, abs(s.imag)))
                if abs(z.imag) > 1e-9:
                    kind = "oscillates"
                elif z.real < 0:
                    kind = "flips sign every period"
                else:
                    kind = "decays" if abs(z) < 1 else "grows"
                print(f"| {TRACKERS[tracker][1]} | held 10 ms | {kind} | {abs(z):.3f} | {omega:.1f} | {zeta:.2f} |")
    for tracker in TRACKERS:
        for period in (PERIOD, STEP):
            below = unstable_below(tracker, period)
            _, largest = stability_curve(tracker, period)
            print(
                f"{TRACKERS[tracker][1]}, the torque every {1000 * period:g} ms: "
                + ("stable at every joint 2 angle" if below is None else f"grows below joint 2 at {below:.2f} deg")
                + f"; largest |z| over 10 ms from {largest.min():.3f} to {largest.max():.3f}"
            )


def print_completion() -> None:
    print("\n== Completion: arrival (s), * if the hand leaves the goal in its hold; final distance (mm); peak torque")
    print("   (N m); peak hand speed (m/s); RMS distance from the taught path (mm); ESN #1 / ESN #2 / replay")
    for scenario in SCENARIOS:
        if scenario == "offsets":
            continue
        print(f"-- {scenario}")
        print("| Tracker | Arrival (s) | Final distance (mm) | Peak torque (N m) | Peak hand speed (m/s) | Path (mm) |")
        print("| --- | --- | --- | --- | --- | --- |")
        for velocity, tracker in COLUMNS:
            runs = [run_of(scenario, tag, arm, velocity, tracker) for _, tag, arm, _, _ in ARMS]
            cells = [
                " / ".join(f"{r['arrival_time_s']:.2f}" + ("*" if r["left_goal"] else "") for r in runs),
                " / ".join(f"{1000 * r['final_distance_m']:.1f}" for r in runs),
                " / ".join(f"{r['peak_torque_nm']:,.1f}" for r in runs),
                " / ".join(f"{r['peak_hand_speed_mps']:.2f}" for r in runs),
                " / ".join(f"{1000 * r['taught_path_rmse_m']:.1f}" for r in runs),
            ]
            print(f"| {TRACKERS[tracker][1]}, {VELOCITIES[velocity]} | {' | '.join(cells)} |")


def disturbances() -> list[tuple[str, str, str]]:
    """Every disturbance: its scenario, the suffix of its columns in metrics.csv, and its name."""
    out = [("block", "", "block")] + [("pushes", f"_{k}", f"push {k}") for k in (1, 2, 3)]
    for log in LOGS:
        n = len(spans_of(f"recorded{log}"))
        out += [(f"recorded{log}", f"_{k}", f"log {log}, push {k}") for k in range(1, n + 1)]
    return out


def print_disturbances() -> None:
    for key, label, scale, digits in MEASURES:
        arms = ARMS[:2] if key == "following_gain" else ARMS
        print(f"\n== {label}, {' / '.join(a[0] for a in arms)}")
        for scenario in ("block", "pushes", *(f"recorded{log}" for log in LOGS)):
            items = [(s, suffix, name) for s, suffix, name in disturbances() if s == scenario]
            print(f"-- {scenario}")
            print(f"| Tracker | {' | '.join(name for _, _, name in items)} |")
            print(f"| --- |{' --- |' * len(items)}")
            for velocity, tracker in COLUMNS:
                cells = []
                for _, suffix, _ in items:
                    values = [
                        run_of(scenario, tag, arm, velocity, tracker)[f"{key}{suffix}"] for _, tag, arm, _, _ in arms
                    ]
                    cells.append(
                        " / ".join("\N{EN DASH}" if not np.isfinite(v) else f"{v * scale:.{digits}f}" for v in values)
                    )
                print(f"| {TRACKERS[tracker][1]}, {VELOCITIES[velocity]} | {' | '.join(cells)} |")
    print("\n== Over the 13 pushes (report 011's three and the ten by hand): the median (least to most); not back")
    print("| Tracker | Following, ESN #1 / ESN #2 | Ringing (mm) | Back within 5 mm (s); not back | Farthest (mm) |")
    print("| --- | --- | --- | --- | --- |")
    pushes = [d for d in disturbances() if d[0] != "block"]
    for velocity, tracker in COLUMNS:
        cells = []
        for key, _, scale, digits in MEASURES:
            arms = ARMS[:2] if key == "following_gain" else ARMS
            parts = []
            for _, tag, arm, _, _ in arms:
                values, missing = [], 0
                for scenario, suffix, _ in pushes:
                    v = run_of(scenario, tag, arm, velocity, tracker)[f"{key}{suffix}"]
                    if not np.isfinite(v):
                        missing += 1
                        v = return_cap(scenario, suffix)
                    values.append(v * scale)
                part = f"{np.median(values):.{digits}f} ({min(values):.{digits}f} to {max(values):.{digits}f})"
                parts.append(part + (f"; {missing}" if key == "return_time_s" else ""))
            cells.append(" / ".join(parts))
        print(f"| {TRACKERS[tracker][1]}, {VELOCITIES[velocity]} | {' | '.join(cells)} |")


def return_cap(scenario: str, suffix: str) -> float:
    """How long after a disturbance its return is watched (s): until the next one, or the run's end."""
    spans = spans_of(scenario)
    k = int(suffix[1:]) if suffix else 1
    nxt = spans[k][0] if k < len(spans) else DURATION
    return nxt - spans[k - 1][1]


def print_offsets() -> None:
    print("\n== From the 49 starts: successes; peak torque, median (largest)")
    print("| Tracker | ESN #1 | ESN #2 | replay |")
    print("| --- | --- | --- | --- |")
    for velocity, tracker in COLUMNS:
        cells = []
        for _, tag, arm, _, _ in ARMS:
            runs = [r for r in rows_of("offsets", tag, velocity, tracker) if r["arm"] == arm]
            torque = [r["peak_torque_nm"] for r in runs]
            cells.append(
                f"{sum(r['success'] for r in runs)}/{len(runs)}; {np.median(torque):,.0f} ({max(torque):,.0f}) N m"
            )
        print(f"| {TRACKERS[tracker][1]}, {VELOCITIES[velocity]} | {' | '.join(cells)} |")


def print_chatter() -> None:
    print(
        "\n== The chatter with the equal gains: from, to (s); peak torque until it ends (N m); ESN #1 / ESN #2 / replay"
    )
    print("| Scenario | Velocity | From (s) | Last period (s) | Peak torque (N m) |")
    print("| --- | --- | --- | --- | --- |")
    for scenario in SCENARIOS:
        for velocity in VELOCITIES:
            found = [
                chatter_of(scenario, tag, arm, velocity, "equal", start_of(scenario)) for _, tag, arm, _, _ in ARMS
            ]
            print(
                f"| {scenario} | {velocity} | {' / '.join(f'{c[0]:.2f}' for c in found)} |"
                f" {' / '.join(f'{c[1]:.2f}' for c in found)} | {' / '.join(f'{c[3]:,.0f}' for c in found)} |"
            )
    for velocity in VELOCITIES:
        for _, tag, arm, _, _ in ARMS:
            ends, others = [], []
            for r in rows_of("offsets", tag, velocity, "equal"):
                if r["arm"] != arm:
                    continue
                ends.append(chatter_of("offsets", tag, arm, velocity, "equal", int(r["start"]))[1])
                others.append(chatter_of("offsets", tag, arm, velocity, "scaled", int(r["start"]))[2])
            print(
                f"offsets, {velocity}, {tag} {arm}: the chatter's last period {min(ends):.2f} to {max(ends):.2f} s;"
                f" with ζ 0.4, chattering periods: {sum(others):g} s in all"
            )
    for scenario in SCENARIOS:
        for velocity in VELOCITIES:
            for _, tag, arm, _, _ in ARMS:
                found = chatter_of(scenario, tag, arm, velocity, "scaled", start_of(scenario))
                if found[2] > 0:
                    print(f"with ζ 0.4, {scenario} {velocity} {tag} {arm} chatters: {found}")
    # After the chatter at the start: the joint 2 angle stays above the bound, and how much the torque still flips.
    bound = unstable_below("equal", PERIOD)
    for scenario in SCENARIOS:
        for velocity in VELOCITIES:
            for name, tag, arm, _, _ in ARMS:
                log = log_of(scenario, tag, arm, velocity, "equal", start_of(scenario))
                end = chatter_of(scenario, tag, arm, velocity, "equal", start_of(scenario))[1]
                after = np.asarray(log.times) > end + PERIOD
                q2 = np.degrees(log.channel("q")[after, 1])
                print(
                    f"after the chatter, {scenario} {velocity} {name}: joint 2 from {q2.min():.1f} deg"
                    f" ({'below' if bound is not None and q2.min() < bound else 'above'} the bound)"
                )


def print_after_chatter() -> None:
    """Undisturbed, from AFTER_CHATTER to 20 s: how far the hand is from the taught path, RMS and largest."""
    print(f"\n== Undisturbed from {AFTER_CHATTER:g} s to 20 s: from the taught path, RMS (largest), mm")
    print("| Tracker | ESN #1 | ESN #2 | replay |")
    print("| --- | --- | --- | --- |")
    for velocity, tracker in COLUMNS:
        cells = []
        for _, tag, arm, _, _ in ARMS:
            log = log_of("nominal", tag, arm, velocity, tracker)
            times = np.asarray(log.times)
            window = (times >= AFTER_CHATTER) & (times <= 20.0)
            distance = 1000 * distances_to_path(endpoint_positions(skeleton(), log.channel("q")[window]), taught()[1])
            cells.append(f"{np.sqrt(np.mean(distance**2)):.1f} ({distance.max():.1f})")
        print(f"| {TRACKERS[tracker][1]}, {VELOCITIES[velocity]} | {' | '.join(cells)} |")


if __name__ == "__main__":
    for scenario in SCENARIOS:
        if config_of(scenario)["evaluation"]["duration"] != DURATION:
            sys.exit(f"{scenario}: the runs' duration is not DURATION")
    main()
