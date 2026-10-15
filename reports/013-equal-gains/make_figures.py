# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 013.

    uv run python reports/013-equal-gains/make_figures.py
    uv run python reports/013-equal-gains/make_figures.py --logs

From the copies of the runs under ``results/`` (configurations, run records, and
metrics), and of reports 011's and 012's runs, whose joint PD at the damping ratio 0.4
is compared here, it draws into ``results/summary``:

- ``stability.png``: how much the tracking error of a still arm grows or shrinks over
  one 10 ms period of the held torque, against joint 2's angle, for each tracker
  (Section 3.1);

and prints the tables of the report: the bound on kd, each joint's natural frequency
and damping ratio, the modes of the two joints together, the completion, and the
measures after each disturbance. With ``--logs``, it reads the runs' logs, which are
not kept in Git, from the runs under the storage root, measures how joint 2's torque
flips, and also draws:

- ``bound.png``: undisturbed, past the bound, the part of joint 2's torque that flips
  sign every period, and the reference's joint 2 angle (Section 3.2);
- ``offsets.png``: from the 49 starts, how long joint 2's torque flips, and the peak
  torque, against joint 2's start angle (Section 3.3);
- ``<scenario>_<velocity>_joints.png``, ``<scenario>_<velocity>_torques.png``,
  ``<scenario>_<velocity>_paths.png``: the joint angles and the joint torques over
  time, and the tip's path in the plane, of ESN #1, ESN #2, and the replay, with the
  two equal gains and with report 011's at 0.4, given the reference velocity or a zero
  one (Sections 3.3 to 3.5). Over time, the ESNs' lines are drawn over the replay's.
"""

from __future__ import annotations

import argparse
import csv
import functools
import sys
import tomllib
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrow
from numpy.typing import NDArray
from skelarm import Skeleton, StateLog, Task, compute_forward_kinematics, compute_mass_matrix

from arm_esn_ctrl.demonstrations import endpoint_positions, resample_joint_angles
from arm_esn_ctrl.disturbances import disturbance_spans
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
DPI = 100


class Tracker(NamedTuple):
    setting: str  # the runs' directory of the tracker
    label: str
    infix: str  # in the configurations' names, after the scenario
    gains: tuple[float, float] | None  # kp and kd for both joints; None: report 011's, scaled by each joint's inertia
    color: str  # in the figures that compare the trackers


OMEGA, DAMPING = 20.0, 0.4
TRACKERS = {
    "stiff": Tracker("pd_kp80_kd1.2", "kp 80, kd 1.2 for both joints", "kp80_kd1.2_", (80.0, 1.2), "#0f8b8d"),
    "soft": Tracker("pd_kp16_kd1.4", "kp 16, kd 1.4 for both joints", "kp16_kd1.4_", (16.0, 1.4), "#b5179e"),
    "scaled": Tracker("pd_w20_z0.4", "ω 20, ζ 0.4 (reports 011, 012)", "", None, "#a3a29d"),
    "past": Tracker("pd_kp80_kd1.7", "kp 80, kd 1.7 for both joints", "kp80_kd1.7_", (80.0, 1.7), "#0b0b0b"),
    "far": Tracker("pd_kp80_kd3", "kp 80, kd 3 for both joints", "", (80.0, 3.0), "#0b0b0b"),
}
COMPARED = ("stiff", "soft", "scaled")  # run in every scenario
CHECKED = ("past", "far")  # just and far past the bound: run undisturbed only
VELOCITIES = {"reference": "reference velocity", "zero": "zero reference velocity"}
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
OFFSET_ORIGIN = "demo_00_filtered -15,-15 deg"  # the offset start drawn over time: joint 2 at 15°, the lowest
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
# Joint 2's torque flips while the part of it that flips sign every period is above this (N m).
FLIP_LEVEL = 1.0
GROWTH_FIT = 10.0  # N m: the flipping part's growth at the start is fitted until it first reaches this

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", action="store_true", help="also analyze the run logs in the storage root")
    args = parser.parse_args()

    SUMMARY.mkdir(parents=True, exist_ok=True)
    figures = {"stability.png": plot_stability()}
    if args.logs:
        figures |= {"bound.png": plot_bound(), "offsets.png": plot_offsets()}
        for scenario in SCENARIOS:
            name = "offset" if scenario == "offsets" else scenario
            for velocity in VELOCITIES:
                for channel in ("joints", "torques"):
                    figures[f"{name}_{velocity}_{channel}.png"] = plot_series(scenario, velocity, channel)
                figures[f"{name}_{velocity}_paths.png"] = plot_paths(scenario, velocity)
    for name, fig in figures.items():
        fig.savefig(SUMMARY / name, dpi=DPI)
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
    given = TRACKERS[tracker].gains
    if given is not None:
        return np.array([given[0], given[0]]), np.array([given[1], given[1]])
    config = TrackerConfig("pd", OMEGA, 0.02, damping=DAMPING)
    return tracking_gains(config, skeleton(), end_posture())


def mass(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """The arm's mass matrix at the joint angles ``q``."""
    model = skeleton().clone()
    for link, angle in zip(model.links[1:], q, strict=True):
        link.q = float(angle)
    compute_forward_kinematics(model)
    return compute_mass_matrix(model)


def smaller_inertia(q2: float) -> float:
    """The smaller eigenvalue of the mass matrix at joint 2's angle ``q2`` (rad) (kg m²)."""
    return float(np.linalg.eigvalsh(mass(np.array([0.0, q2])))[0])


def kd_bound(kp: float, inertia: float, period: float = PERIOD, step: float = STEP) -> float:
    """The largest kd (N m s/rad) under which a mode of ``inertia`` (kg m²) does not grow, the torque held.

    With the same gains for both joints, the error splits along the mass matrix's
    eigenvectors, each a single inertia μ under PD. Over a period T of n simulation
    steps, the error moves by T ė + c T² u / μ, c = (n + 1) / (2n), and its rate by
    T u / μ; the map's characteristic polynomial is z² - (2 - c a - b) z + 1 - b +
    (1 - c) a, with a = kp T² / μ and b = kd T / μ. Its root at z < -1 appears when
    kd > 2 μ / T - kp T / (2n).
    """
    return 2 * inertia / period - kp * period / (2 * round(period / step))


def continuous_modes(m: NDArray[np.float64], kp: NDArray[np.float64], kd: NDArray[np.float64]) -> NDArray[Any]:
    """The eigenvalues s (1/s) of the error ``M ë + Kd ė + Kp e = 0`` of the arm at rest, the torque continuous."""
    n = len(kp)
    inverse = np.linalg.inv(m)
    system = np.block([[np.zeros((n, n)), np.eye(n)], [-inverse @ np.diag(kp), -inverse @ np.diag(kd)]])
    return np.linalg.eigvals(system)


def held_modes(
    m: NDArray[np.float64], kp: NDArray[np.float64], kd: NDArray[np.float64], step: float | None = STEP
) -> NDArray[Any]:
    """The eigenvalues z of the tracking error's map over one 10 ms period, the torque computed at its start and held.

    The arm is a pure inertia ``M`` at rest, and the torque ``u = -Kp e - Kd ė``. As
    the simulation does, each step of ``step`` changes the rate by ``h M⁻¹u`` first,
    then the error by ``h ė`` (semi-implicit Euler): over the five 2 ms steps of a
    period, the error moves by ``T ė + 0.6 T² M⁻¹u``. With ``step`` None, the hold is
    exact: ``T ė + T²/2 M⁻¹u``. The error shrinks period by period if every ``|z|``
    is below 1, and grows if one is above.
    """
    n = len(kp)
    inverse = np.linalg.inv(m)
    torque = -np.hstack([np.diag(kp), np.diag(kd)])
    if step is None:
        drive = np.vstack([PERIOD**2 / 2 * inverse, PERIOD * inverse])
        free = np.block([[np.eye(n), PERIOD * np.eye(n)], [np.zeros((n, n)), np.eye(n)]])
        return np.linalg.eigvals(free + drive @ torque)
    move = np.block([[np.eye(n), step * np.eye(n)], [np.zeros((n, n)), np.eye(n)]])
    drive = np.vstack([step**2 * inverse, step * inverse]) @ torque
    period_map = np.eye(2 * n)
    for _ in range(round(PERIOD / step)):
        period_map = move @ period_map + drive
    return np.linalg.eigvals(period_map)


def largest_per_period(q2: float, tracker: str, step: float | None = STEP) -> float:
    """The largest |z| over one 10 ms period at joint 2's angle ``q2`` (rad)."""
    return float(np.abs(held_modes(mass(np.array([0.0, q2])), *gains(tracker), step)).max())


@functools.cache
def stability_curve(tracker: str, step: float | None = STEP) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Joint 2's angles (deg) from 0 to 180, and the largest |z| over 10 ms at each."""
    angles = np.arange(0.0, 180.0 + 1e-9, 0.05)
    return angles, np.array([largest_per_period(np.radians(a), tracker, step) for a in angles])


def unstable_below(tracker: str, step: float | None = STEP) -> float | None:
    """The joint 2 angle (deg) below which the error grows, or None if it shrinks at every angle."""
    angles, largest = stability_curve(tracker, step)
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
    return RESULTS if TRACKERS[tracker].gains is not None else EARLIER[SCENARIOS[scenario][0]]


def run_name(scenario: str, tag: str, velocity: str, tracker: str) -> str:
    """The name of the one run of a configuration."""
    stem = f"{scenario}_{TRACKERS[tracker].infix}robot_{tag}_{velocity}_velocity_filtered"
    matches = sorted(p.name for p in results_of(scenario, tracker).glob(f"*-{stem}") if p.is_dir())
    if len(matches) != 1:
        msg = f"expected one run of {stem}, found {matches}"
        raise FileNotFoundError(msg)
    return matches[0]


def rows_of(scenario: str, tag: str, velocity: str, tracker: str) -> list[dict[str, Any]]:
    """The metrics of a run with one tracker: every row with given gains, the damping ratio 0.4's otherwise."""
    rows = read_csv(results_of(scenario, tracker) / run_name(scenario, tag, velocity, tracker) / "metrics.csv")
    return [r for r in rows if TRACKERS[tracker].gains is not None or r["damping"] == DAMPING]


def run_of(scenario: str, tag: str, arm: str, velocity: str, tracker: str, start: int = 0) -> dict[str, Any]:
    """One arm's run from one start, with one tracker."""
    return next(r for r in rows_of(scenario, tag, velocity, tracker) if r["arm"] == arm and r["start"] == start)


def offset_start() -> int:
    """The index of the offset start drawn over time."""
    rows = rows_of("offsets", ARMS[0][1], "reference", "stiff")
    return next(int(r["start"]) for r in rows if r["origin"] == OFFSET_ORIGIN)


def start_of(scenario: str) -> int:
    return offset_start() if scenario == "offsets" else 0


def log_of(scenario: str, tag: str, arm: str, velocity: str, tracker: str, start: int = 0) -> StateLog:
    """One arm's log from a run under the storage root."""
    run = resolve_run_path(f"results/{run_name(scenario, tag, velocity, tracker)}")
    return StateLog.load(run / TRACKERS[tracker].setting / f"{arm}_{start:02d}.sklog.npz")


@functools.cache
def config_of(scenario: str) -> dict[str, Any]:
    with (RESULTS / run_name(scenario, ARMS[0][1], "reference", "stiff") / "config.toml").open("rb") as f:
        return tomllib.load(f)


def spans_of(scenario: str) -> list[tuple[float, float]]:
    """When each disturbance of a scenario acts."""
    return disturbance_spans(config_of(scenario).get("disturbance"))


def held_torque(log: StateLog) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The times of the periods' starts (s), and joint 2's torque held over each (N m)."""
    every = round(PERIOD / STEP)
    return np.asarray(log.times)[::every], log.channel("tau")[::every, 1]


def flipping(log: StateLog) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The part of joint 2's held torque that flips sign every period, |u(k-1) - 2 u(k) + u(k+1)| / 4 (N m)."""
    times, torque = held_torque(log)
    return times[1:-1], np.abs(torque[:-2] - 2 * torque[1:-1] + torque[2:]) / 4


def flips(log: StateLog) -> tuple[float, float, float]:
    """When joint 2's torque flips: its first and last flipping period's start (s), and the time it flips (s).

    A period flips when the part of joint 2's held torque that flips sign every period
    (see :func:`flipping`) is above FLIP_LEVEL. NaN when none does.
    """
    times, part = flipping(log)
    at = times[part > FLIP_LEVEL]
    if not len(at):
        return float("nan"), float("nan"), 0.0
    return float(at.min()), float(at.max()), PERIOD * len(at)


@functools.cache
def flips_of(scenario: str, tag: str, arm: str, velocity: str, tracker: str, start: int = 0) -> tuple[float, ...]:
    """A run's flips (see :func:`flips`), and its peak joint torque until the last (N m)."""
    log = log_of(scenario, tag, arm, velocity, tracker, start)
    first, last, total = flips(log)
    if not np.isfinite(last):
        return first, last, total, float("nan")
    until = np.asarray(log.times) <= last + PERIOD
    return first, last, total, float(np.abs(log.channel("tau")[until]).max())


def first_flips(log: StateLog) -> tuple[float, float]:
    """The first and last period (s) of the flips from the start: the first span with no gap over 0.1 s."""
    times, part = flipping(log)
    at = times[part > FLIP_LEVEL]
    gaps = np.flatnonzero(np.diff(at) > 0.1 + 1e-9)
    return float(at[0]), float(at[gaps[0]] if len(gaps) else at[-1])


def growth(log: StateLog) -> float:
    """How much the flipping part of joint 2's torque grows per period at the start, fitted until GROWTH_FIT."""
    times, part = flipping(log)
    task = times >= 0.0
    times, part = times[task], part[task]
    first = int(np.argmax(part > 0.0))
    last = int(np.argmax(part > GROWTH_FIT))
    slope = np.polyfit(times[first : last + 1], np.log(part[first : last + 1]), 1)[0]
    return float(np.exp(slope * PERIOD))


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


# ---------------------------------------------------------------------------- figures


def plot_stability() -> Figure:
    """The largest |z| over one 10 ms period against joint 2's angle, with the reach's angles and the starts marked."""
    fig = Figure(figsize=(11, 5.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "Joint PD on the still arm, the torque held 10 ms: how much the tracking error grows (above 1) or shrinks"
        "\n(below 1) over a period, its largest mode, against joint 2's angle (the arm's inertia depends on it alone)",
        color="#0b0b0b",
    )
    ax = fig.subplots()
    style(ax)
    q2 = np.degrees(taught()[0][:, 1])
    ax.axvspan(q2.min(), q2.max(), color=DISTURBANCE_COLOR, zorder=0, label="joint 2 over the take")
    for tracker in ("far", "past", "stiff", "soft", "scaled"):
        angles, largest = stability_curve(tracker)
        ax.plot(
            angles,
            largest,
            color=TRACKERS[tracker].color,
            linestyle="--" if tracker == "past" else "-",
            linewidth=1.6,
            label=TRACKERS[tracker].label,
            zorder=3,
        )
    ax.axhline(1.0, color=PUSH_COLOR, linewidth=1.0, zorder=2)
    # Each mark: its angle, its words, their height, and their side of the line.
    marks = [
        (q2[0], "the start", 2.9, "left"),
        (q2[-1], "the target", 2.9, "left"),
        (15.0, "the lowest start", 2.9, "right"),
    ]
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
    ax.annotate(
        "grows above, shrinks below",
        xy=(178, 1.0),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        color=PUSH_COLOR,
    )
    ax.set_xlim(0, 180)
    ax.set_ylim(0.8, 3.2)
    ax.set_xticks(np.arange(0, 181, 15))
    ax.set_xlabel("joint 2's angle (deg)", color=TEXT_COLOR)
    ax.set_ylabel("largest |z| over 10 ms", color=TEXT_COLOR)
    ax.legend(frameon=False, labelcolor=TEXT_COLOR, loc="upper right")
    return fig


def plot_bound() -> Figure:
    """Undisturbed past the bound: joint 2's flipping torque over time, and the reference's joint 2."""
    columns = [(tracker, velocity) for tracker in CHECKED for velocity in VELOCITIES]
    fig = Figure(figsize=(16, 7.4), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "Undisturbed past the bound, the torque every 10 ms: ESN #1 (blue), ESN #2 (purple), and the replay (orange);"
        "\ntop, the part of joint 2's torque that flips sign every period, and the model's growth from the start"
        " (dotted); bottom, the reference's joint 2 (the replay's, the take's)",
        color="#0b0b0b",
    )
    axes = fig.subplots(2, len(columns), sharex=True, sharey="row")
    start_posture = taught()[0][0]
    for col, (tracker, velocity) in enumerate(columns):
        bound = unstable_below(tracker)
        model = float(np.abs(held_modes(mass(start_posture), *gains(tracker))).max())
        for row in range(2):
            style(axes[row, col])
        for _, tag, arm, color, _ in ARMS:
            log = log_of("nominal", tag, arm, velocity, tracker)
            times, part = flipping(log)
            axes[0, col].semilogy(times, np.maximum(part, 1e-9), color=color, linewidth=1.0, zorder=ARM_ZORDERS[arm])
            axes[1, col].plot(
                log.times, np.degrees(log.channel("q_ref")[:, 1]), color=color, linewidth=1.2, zorder=ARM_ZORDERS[arm]
            )
            if arm == "replay":
                # The model's growth from the replay's first flipping torque, its arm still at the start.
                first = int(np.argmax((times >= 0.0) & (part > 0.0)))
                ahead = times[first] + PERIOD * np.arange(80)
                axes[0, col].semilogy(
                    ahead, part[first] * model ** np.arange(80), color=TEXT_COLOR, linestyle=":", linewidth=1.4
                )
        axes[0, col].axhline(FLIP_LEVEL, color=PUSH_COLOR, linewidth=0.8)
        axes[0, col].set_ylim(1e-6, 1e4)
        axes[0, col].set_title(
            f"{TRACKERS[tracker].label}, {VELOCITIES[velocity]}\nthe model: {model:.2f} times per period at the start",
            color="#0b0b0b",
            fontsize=10,
        )
        if bound is not None:
            axes[1, col].axhline(bound, color=PUSH_COLOR, linewidth=1.0)
            axes[1, col].annotate(
                f"the bound, {bound:.1f}°",
                xy=(0.98, bound),
                xycoords=("axes fraction", "data"),
                xytext=(0, -12),
                textcoords="offset points",
                ha="right",
                color=PUSH_COLOR,
                fontsize=9,
            )
        axes[1, col].set_xlim(0, 3.5)
        axes[1, col].set_xlabel("time (s)", color=TEXT_COLOR)
    axes[0, 0].set_ylabel("flipping part of joint 2's torque (N m)", color=TEXT_COLOR)
    axes[1, 0].set_ylabel("reference's joint 2 (deg)", color=TEXT_COLOR)
    handles = [Line2D([], [], color=color, linewidth=1.4, label=name) for name, _, _, color, _ in ARMS]
    handles += [
        Line2D([], [], color=TEXT_COLOR, linestyle=":", linewidth=1.4, label="the model's growth at the start"),
        Line2D([], [], color=PUSH_COLOR, linewidth=0.8, label=f"{FLIP_LEVEL:g} N m; the bound"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_offsets() -> Figure:
    """From the 49 starts: how long joint 2's torque flips, and the peak torque, against joint 2's start angle."""
    fig = Figure(figsize=(14, 7.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "From the 49 starts, joint PD, the torque every 10 ms: ESN #1 (circles), ESN #2 (squares), and the replay"
        " (diamonds),\ncolored by the tracker",
        color="#0b0b0b",
    )
    axes = fig.subplots(2, len(VELOCITIES), sharex=True, sharey="row")
    shifts = {"stiff": -1.0, "soft": 0.0, "scaled": 1.0}
    nudges = {name: nudge for (name, *_), nudge in zip(ARMS, (-0.3, 0.0, 0.3), strict=True)}
    for col, velocity in enumerate(VELOCITIES):
        for row in range(2):
            ax = axes[row, col]
            style(ax)
            for tracker in COMPARED:
                for name, tag, arm, _, marker in ARMS:
                    starts = [r for r in rows_of("offsets", tag, velocity, tracker) if r["arm"] == arm]
                    x = [
                        np.degrees(log_of("offsets", tag, arm, velocity, tracker, int(r["start"])).channel("q")[0, 1])
                        + shifts[tracker]
                        + nudges[name]
                        for r in starts
                    ]
                    if row == 0:
                        y = [flips_of("offsets", tag, arm, velocity, tracker, int(r["start"]))[2] for r in starts]
                    else:
                        y = [r["peak_torque_nm"] for r in starts]
                    ax.plot(
                        x,
                        y,
                        linestyle="none",
                        marker=marker,
                        markersize=4,
                        color=TRACKERS[tracker].color,
                        markerfacecolor="none",
                        zorder=ARM_ZORDERS[arm],
                    )
            if row == 0:
                ax.set_title(VELOCITIES[velocity], color="#0b0b0b", fontsize=10)
            else:
                ax.set_yscale("log")
                ax.set_xlabel("joint 2's start angle (deg), each tracker's marks side by side", color=TEXT_COLOR)
            if col == 0:
                ax.set_ylabel(
                    [f"time joint 2's torque flips\nby over {FLIP_LEVEL:g} N m (s)", "peak joint torque (N m)"][row],
                    color=TEXT_COLOR,
                )
    handles = [Line2D([], [], color=TRACKERS[t].color, linewidth=2.0, label=TRACKERS[t].label) for t in COMPARED]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_series(scenario: str, velocity: str, channel: str) -> Figure:
    """The joint angles or torques of the three arms over time, a column per tracker, one reference velocity.

    Under a disturbance, a first row draws each arm's tip force.
    """
    words = SCENARIOS[scenario][1]
    spans = spans_of(scenario)
    rows = (1 if spans else 0) + 2
    fig = Figure(figsize=(13, 2.2 + 2.6 * rows), facecolor=SURFACE_COLOR, layout="constrained")
    what = "Joint angles" if channel == "joints" else "Joint torques"
    fig.suptitle(
        f"{what} {words}, joint PD given the {VELOCITIES[velocity]}, the torque every 10 ms:"
        "\nESN #1 (blue), ESN #2 (purple), and the replay (orange)"
        + ("; dashed: each ESN's reference; thick gray: the taught motion" if channel == "joints" else ""),
        color="#0b0b0b",
    )
    height_ratios = ([0.6] if spans else []) + [1.0, 1.0]
    axes = np.asarray(fig.subplots(rows, len(COMPARED), sharex=True, height_ratios=height_ratios)).reshape(
        rows, len(COMPARED)
    )
    start = start_of(scenario)
    q_taught = np.degrees(taught()[0])
    t_taught = DT * np.arange(len(q_taught))
    logs = {
        tracker: {name: log_of(scenario, tag, arm, velocity, tracker, start) for name, tag, arm, _, _ in ARMS}
        for tracker in COMPARED
    }
    # One scale per joint across the trackers.
    limits = []
    if channel == "torques":
        for joint in range(2):
            values = [log.channel("tau")[:, joint] for t in COMPARED for log in logs[t].values()]
            low, high = float(min(v.min() for v in values)), float(max(v.max() for v in values))
            limits.append((low - 0.1 * (high - low), high + 0.1 * (high - low)))
    for col, tracker in enumerate(COMPARED):
        for row in range(rows):
            style(axes[row, col])
            for onset, end in spans:
                axes[row, col].axvspan(onset, end, color=DISTURBANCE_COLOR, zorder=0)
        axes[0, col].set_title(TRACKERS[tracker].label, color="#0b0b0b", fontsize=10)
        if spans:
            for name, _, arm, color, _ in ARMS:
                log = logs[tracker][name]
                force = np.linalg.norm(log.channel("ext_force"), axis=1)
                axes[0, col].plot(log.times, force, color=color, linewidth=1.0, zorder=ARM_ZORDERS[arm])
            top = max(
                np.linalg.norm(log.channel("ext_force"), axis=1).max() for t in COMPARED for log in logs[t].values()
            )
            axes[0, col].set_ylim(-0.05 * top, 1.1 * top)
        for joint in range(2):
            ax = axes[rows - 2 + joint, col]
            if channel == "joints":
                ax.plot(t_taught, q_taught[:, joint], color=TAUGHT_COLOR, linewidth=4.0, zorder=1)
            for name, _, arm, color, _ in ARMS:
                log = logs[tracker][name]
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


def plot_paths(scenario: str, velocity: str) -> Figure:
    """Each arm's tip path in the plane: a row per arm, a column per tracker, one reference velocity.

    Each panel draws the arm at its initial posture, the taught path, the goal, the
    tip's path, and, for each disturbance, an arrow from where the tip is at its
    onset along the mean force over it.
    """
    words = SCENARIOS[scenario][1]
    spans = spans_of(scenario)
    start = start_of(scenario)
    fig = Figure(
        figsize=(4.4 * len(COMPARED) + 0.6, 2.7 * len(ARMS) + 1.6), facecolor=SURFACE_COLOR, layout="constrained"
    )
    fig.suptitle(
        f"The tip's path {words}, joint PD given the {VELOCITIES[velocity]}, the torque every 10 ms"
        + ("\nand the force at the start of each disturbance" if spans else ""),
        color="#0b0b0b",
    )
    axes = np.asarray(fig.subplots(len(ARMS), len(COMPARED), sharex=True, sharey=True)).reshape(
        len(ARMS), len(COMPARED)
    )
    taught_hand = taught()[1]
    first = log_of(scenario, ARMS[0][1], ARMS[0][2], velocity, "stiff", start)
    task = Task.from_dict(first.extra["playback"]["task"])
    target, radius = task.require_target(), task.tolerance or 0.0
    for row, (name, tag, arm, color, _) in enumerate(ARMS):
        for col, tracker in enumerate(COMPARED):
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
                ax.set_title(TRACKERS[tracker].label, color="#0b0b0b", fontsize=10)
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
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    return fig


# ---------------------------------------------------------------------------- tables


def print_tables(*, logs: bool) -> None:
    print_bound()
    print_gains()
    print_modes()
    print_completion()
    print_disturbances()
    print_offsets()
    if logs:
        print_check()
        print_flips()


def print_bound() -> None:
    q = taught()[0]
    smaller = np.array([np.linalg.eigvalsh(mass(p))[0] for p in q])
    print("\n== The bound kd < 2 mu / T - kp T / (2 n), n = 5 steps per period")
    print(f"the smaller inertia over the take: {smaller.min():.5f} (at the start) to {smaller.max():.5f} kg m²")
    for q2 in (0.0, 15.0, 30.0, 74.59):
        mu = smaller_inertia(np.radians(q2))
        print(
            f"joint 2 at {q2:g} deg: mu {mu:.5f}; kd bound at kp 80 {kd_bound(80.0, mu):.3f}, at kp 16"
            f" {kd_bound(16.0, mu):.3f}; with an exact hold 2 mu / T = {2 * mu / PERIOD:.3f}"
        )
    print("| Tracker | grows below joint 2 at (5 steps) | (exact hold) | largest |z| over the take |")
    print("| --- | --- | --- | --- |")
    for tracker in TRACKERS:
        largest = [largest_per_period(p[1], tracker) for p in q[:: max(1, len(q) // 400)]]
        cells = []
        for step in (STEP, None):
            below = unstable_below(tracker, step)
            cells.append("stable at every angle" if below is None else f"{below:.2f} deg")
        print(f"| {TRACKERS[tracker].label} | {' | '.join(cells)} | {min(largest):.3f} to {max(largest):.3f} |")


def print_gains() -> None:
    posture = end_posture()
    q = taught()[0]
    print(f"\n== Each joint alone (at the take's final posture, {np.degrees(posture).round(2)} deg)")
    print("| Tracker | Joint | kp (N m/rad) | kd (N m s/rad) | Inertia (kg m²) | ω (rad/s) | ζ | Over the take: ω, ζ |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    inertia = np.array([np.diag(mass(p)) for p in q])
    for tracker in COMPARED:
        kp, kd = gains(tracker)
        omega, zeta = error_dynamics("pd", (kp, kd), skeleton(), posture)
        omegas = np.sqrt(kp / inertia)
        zetas = kd / (2 * np.sqrt(kp * inertia))
        for j in range(2):
            print(
                f"| {TRACKERS[tracker].label} | {j + 1} | {kp[j]:.2f} | {kd[j]:.3f} | {mass(posture)[j, j]:.4f} |"
                f" {omega[j]:.1f} | {zeta[j]:.2f} | {omegas[:, j].min():.1f} to {omegas[:, j].max():.1f},"
                f" {zetas[:, j].min():.2f} to {zetas[:, j].max():.2f} |"
            )


def print_modes() -> None:
    posture = end_posture()
    start = taught()[0][0]
    for where, q in (("the take's final posture", posture), ("the start", start)):
        m = mass(q)
        print(f"\n== The two joints together, at {where} ({np.degrees(q).round(2)} deg); M =\n{m.round(5)}")
        print("| Tracker | Torque | Mode | Decay over 10 ms | ω (rad/s) | ζ |")
        print("| --- | --- | --- | ---: | ---: | ---: |")
        for tracker in COMPARED:
            kp, kd = gains(tracker)
            for s in sorted(continuous_modes(m, kp, kd), key=lambda s: abs(s)):
                if s.imag < -1e-9:
                    continue
                omega, zeta = as_mode(s)
                kind = "oscillates" if s.imag > 1e-9 else "decays"
                decay = np.exp(s.real * PERIOD)
                print(f"| {TRACKERS[tracker].label} | continuous | {kind} | {decay:.3f} | {omega:.1f} | {zeta:.2f} |")
            for z in sorted(held_modes(m, kp, kd), key=lambda z: -abs(z)):
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
                print(f"| {TRACKERS[tracker].label} | held 10 ms | {kind} | {abs(z):.3f} | {omega:.1f} | {zeta:.2f} |")


def print_completion() -> None:
    print("\n== Completion: arrival (s), * if the hand leaves the goal in its hold; final distance (mm); peak torque")
    print("   (N m); peak hand speed (m/s); RMS distance from the taught path (mm); ESN #1 / ESN #2 / replay")
    for scenario in SCENARIOS:
        if scenario == "offsets":
            continue
        print(f"-- {scenario}")
        print("| Tracker | Arrival (s) | Final distance (mm) | Peak torque (N m) | Peak hand speed (m/s) | Path (mm) |")
        print("| --- | --- | --- | --- | --- | --- |")
        for velocity in VELOCITIES:
            for tracker in COMPARED + (CHECKED if scenario == "nominal" else ()):
                runs = [run_of(scenario, tag, arm, velocity, tracker) for _, tag, arm, _, _ in ARMS]
                cells = [
                    " / ".join(f"{r['arrival_time_s']:.2f}" + ("*" if r["left_goal"] else "") for r in runs),
                    " / ".join(f"{1000 * r['final_distance_m']:.1f}" for r in runs),
                    " / ".join(f"{r['peak_torque_nm']:,.1f}" for r in runs),
                    " / ".join(f"{r['peak_hand_speed_mps']:.2f}" for r in runs),
                    " / ".join(f"{1000 * r['taught_path_rmse_m']:.1f}" for r in runs),
                ]
                print(f"| {TRACKERS[tracker].label}, {VELOCITIES[velocity]} | {' | '.join(cells)} |")


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
            for velocity in VELOCITIES:
                for tracker in COMPARED:
                    cells = []
                    for _, suffix, _ in items:
                        values = [
                            run_of(scenario, tag, arm, velocity, tracker)[f"{key}{suffix}"]
                            for _, tag, arm, _, _ in arms
                        ]
                        cells.append(
                            " / ".join(
                                "\N{EN DASH}" if not np.isfinite(v) else f"{v * scale:.{digits}f}" for v in values
                            )
                        )
                    print(f"| {TRACKERS[tracker].label}, {VELOCITIES[velocity]} | {' | '.join(cells)} |")
    print("\n== Over the 13 pushes (report 011's three and the ten by hand): the median (least to most); not back")
    print("| Tracker | Following, ESN #1 / ESN #2 | Ringing (mm) | Back within 5 mm (s); not back | Farthest (mm) |")
    print("| --- | --- | --- | --- | --- |")
    pushes = [d for d in disturbances() if d[0] != "block"]
    for velocity in VELOCITIES:
        for tracker in COMPARED:
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
            print(f"| {TRACKERS[tracker].label}, {VELOCITIES[velocity]} | {' | '.join(cells)} |")


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
    for velocity in VELOCITIES:
        for tracker in COMPARED:
            cells = []
            for _, tag, arm, _, _ in ARMS:
                runs = [r for r in rows_of("offsets", tag, velocity, tracker) if r["arm"] == arm]
                torque = [r["peak_torque_nm"] for r in runs]
                cells.append(
                    f"{sum(r['success'] for r in runs)}/{len(runs)}; {np.median(torque):,.1f} ({max(torque):,.1f}) N m"
                )
            print(f"| {TRACKERS[tracker].label}, {VELOCITIES[velocity]} | {' | '.join(cells)} |")


def print_check() -> None:
    """Past the bound, undisturbed: the growth per period against the model, and when the flips from the start end."""
    start = taught()[0][0]
    print("\n== Past the bound, undisturbed: ESN #1 / ESN #2 / replay")
    print(
        "| Tracker | Velocity | Model, per period at the start | Measured, per period | Flips (s) |"
        " Peak torque (N m) |"
        " Reference passes the bound (s) | Reference's joint 2 at the last flip (deg) |"
    )
    print("| --- | --- | ---: | --- | --- | --- | --- | --- |")
    for tracker in CHECKED:
        bound = unstable_below(tracker)
        assert bound is not None
        model = float(np.abs(held_modes(mass(start), *gains(tracker))).max())
        for velocity in VELOCITIES:
            cells: list[list[str]] = [[], [], [], [], []]
            for _, tag, arm, _, _ in ARMS:
                log = log_of("nominal", tag, arm, velocity, tracker)
                first, last = first_flips(log)
                peak = float(np.abs(log.channel("tau")[np.asarray(log.times) <= last + PERIOD]).max())
                times = np.asarray(log.times)
                q2 = np.degrees(log.channel("q_ref")[:, 1])
                passes = times[np.argmax((times >= 0.0) & (q2 > bound))]
                cells[0].append(f"{growth(log):.2f}")
                cells[1].append(f"{first:.2f} to {last:.2f}")
                cells[2].append(f"{peak:,.0f}")
                cells[3].append(f"{passes:.2f}")
                cells[4].append(f"{q2[np.argmin(np.abs(times - last))]:.1f}")
            print(
                f"| {TRACKERS[tracker].label} | {velocity} | {model:.3f} | {' | '.join(' / '.join(c) for c in cells)} |"
            )


def print_flips() -> None:
    """How long joint 2's torque flips with the compared trackers, from each start, and after each force ends."""
    print(f"\n== The compared trackers: the time joint 2's torque flips by over {FLIP_LEVEL:g} N m (s)")
    print("| Scenario | Tracker | Velocity | ESN #1 | ESN #2 | replay |")
    print("| --- | --- | --- | --- | --- | --- |")
    for scenario in SCENARIOS:
        for tracker in COMPARED:
            for velocity in VELOCITIES:
                cells = []
                for _, tag, arm, _, _ in ARMS:
                    starts = [int(r["start"]) for r in rows_of(scenario, tag, velocity, tracker) if r["arm"] == arm]
                    found = {start: flips_of(scenario, tag, arm, velocity, tracker, start) for start in starts}
                    if scenario != "offsets":
                        first, last, total, peak = found[0]
                        cells.append(
                            "none" if total == 0 else f"{total:.2f}, {first:.2f} to {last:.2f}, {peak:.1f} N m"
                        )
                        continue
                    longest = max(found, key=lambda k: found[k][2])
                    q0 = np.degrees(log_of(scenario, tag, arm, velocity, tracker, longest).channel("q")[0])
                    over = sorted(
                        {
                            round(
                                float(np.degrees(log_of(scenario, tag, arm, velocity, tracker, k).channel("q")[0, 1]))
                            )
                            for k, v in found.items()
                            if v[2] > 0.2
                        }
                    )
                    count = sum(v[2] > 0 for v in found.values())
                    if not count:
                        cells.append("none")
                        continue
                    cells.append(
                        f"{count} starts, up to {found[longest][2]:.2f} s (from {q0[0]:.0f}°, {q0[1]:.0f}°)"
                        + (f"; over 0.2 s from joint 2 at {', '.join(f'{a}°' for a in over)}" if over else "")
                    )
                print(f"| {scenario} | {TRACKERS[tracker].label} | {velocity} | {' | '.join(cells)} |")


if __name__ == "__main__":
    for scenario in SCENARIOS:
        if config_of(scenario)["evaluation"]["duration"] != DURATION:
            sys.exit(f"{scenario}: the runs' duration is not DURATION")
    main()
