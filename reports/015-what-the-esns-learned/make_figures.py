# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 015.

    uv run python reports/015-what-the-esns-learned/make_figures.py

It reads the study's three runs under the storage root, whose arrays are not kept in
Git, copies their configurations, run records, and CSV files into ``results/``,
prints the tables of the report, and draws into ``results/summary``:

- step 1, the ESNs on their own from rest anywhere (Section 3.1):
  ``arrival.png`` (the arrival time over the start postures), ``portrait.png`` (the
  runs in joint space), ``first_step.png`` (how far the first step jumps, and where
  it lands), ``hands.png`` (the hand paths),
  ``funnel.png`` (the distance from the take's joint path against the progress
  along it), ``joining.png`` (the progress at which each run joins the path),
  ``timing.png`` (the arrival time against the start), ``states.png`` (the
  reservoir states against the take's), ``settling.png`` (the state held at the
  start posture), and ``series.png`` (the joint angles over time from six starts);
- step 2, the robot pushed along the reach (Section 3.2): ``pulse_joints.png`` and
  ``pulse_torques.png`` (the joint angles and torques over time, undisturbed and
  under four pulses), ``pulse_paths.png`` (the tip's paths under the pulses of 6 N),
  ``pulse_maps.png`` and ``pulse_maps_3n.png`` (the measures over the onsets and
  directions), and ``pulse_progress.png`` (the progress along the path of the arms
  and their references);
- step 3, the fixed points and the rates (Section 3.3): ``spectra.png`` (the
  eigenvalues at the target), ``modes.png`` (how the slowest modes move the
  posture), and ``rates.png`` (how fast changes across the ESN's run decay).

The runs default to the report's; ``--global-starts``, ``--pulses``, and
``--fixed-points`` name others of the study.
"""

from __future__ import annotations

import argparse
import csv
import functools
import shutil
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrow, Rectangle
from numpy.typing import NDArray
from skelarm import Skeleton, StateLog, compute_forward_kinematics

from arm_esn_ctrl.demonstrations import endpoint_positions, resample_joint_angles
from arm_esn_ctrl.esn import ReachingEsn
from arm_esn_ctrl.metrics import onset_index, path_progress
from arm_esn_ctrl.states import teacher_forced_states
from arm_esn_ctrl.storage import resolve_run_path

REPORT = Path(__file__).resolve().parent
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"
STUDY = "results/manual_demonstration_v2_esn_dynamics"
RUNS = {
    "global_starts": "20261015-223608-global_starts",
    "warmup_esn1": "20261015-225101-warmup_lr0.03_sr1.2_is1.5_seed8",
    "warmup_esn2": "20261015-225101-warmup_lr0.03_sr1.2_is1_seed2",
    "pulses": "20261015-223608-push_pulses",
    "fixed_points": "20261015-225557-fixed_points",
}
# The take, as report 005 copied it: what the ESNs were trained on, and what the replays replay.
TAKE_RUN = REPORT.parent / "005-tuned-on-the-robot" / "data" / "20261008-170034-reach_manual_v2"
DT = 0.01  # the ESNs' period (s)
DPI = 90

# The ESNs, and the arms on the robot: key, label, color, and marker.
ESNS = [("esn1", "ESN #1", "#2a78d6", "o"), ("esn2", "ESN #2", "#5b2a86", "s")]
ARMS = [*ESNS, ("replay", "replay", "#eb6834", "D"), ("progress", "progress replay", "#2e8b57", "^")]
# The layers of the lines over time: the ESNs' arms on top, their references (dashed) under them, then the
# progress replay's arm and reference, the replay's at the bottom.
ARM_ZORDERS = {"esn1": 3.0, "esn2": 3.0, "progress": 2.2, "replay": 2.0}
REFERENCE_ZORDERS = {"esn1": 2.5, "esn2": 2.5, "progress": 2.1}
OUTCOMES = ["target", "rest elsewhere", "moving", "out of range"]

TEXT_COLOR = "#52514e"
TITLE_COLOR = "#0b0b0b"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
DISTURBANCE_COLOR = "#f0efec"
TAUGHT_COLOR = "#b9b7b0"
MISSING_COLOR = "#d9d8d3"
PUSH_COLOR = "#dc0000"
FORCE_SCALE = 0.025  # m/N: the length of a force arrow in the plane, 25 mm per newton
# The six starts drawn over time, and their colors: the take's start, three far from it, two near the target.
SHOWN_STARTS = [(30.0, 30.0), (-150.0, -120.0), (150.0, -60.0), (-60.0, 150.0), (120.0, 120.0), (90.0, 60.0)]
SHOWN_COLORS = ["#0b0b0b", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9"]
# The pulses drawn over time: none, and (onset, angle, force).
SHOWN_PULSES: list[tuple[float, float, float] | None] = [
    None,
    (4.0, 0.0, 6.0),
    (4.0, 180.0, 6.0),
    (7.0, 90.0, 6.0),
    (12.0, 270.0, 6.0),
]
COLOR_TIME = 14.0  # s: the runs colored by time are colored up to this time, past every arrival
PATH_ONSETS = [2.0, 4.0, 6.0, 8.0, 12.0]  # the onsets whose tip paths are drawn
PATH_WINDOW = 0.28  # m: each panel of the tip's paths spans this far from where the take's hand is at the onset
# The measures of the pulses: the column of metrics.csv, its label, its scale, and whether it has a sign.
PULSE_MEASURES = [
    ("off_path_m", "farthest from the path (mm)", 1000.0, False),
    ("return_time_s", "back within 5 mm of the path (s)", 1.0, False),
    ("arrival_shift_s", "arrival, later than undisturbed (s)", 1.0, True),
    ("setback", "back along the path (fraction)", 1.0, False),
    ("following_gain", "following gain", 1.0, False),
    ("peak_torque_rise_nm", "peak torque over 2 s, rise (N m)", 1.0, False),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, run in RUNS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", default=run, help=f"the {name.replace('_', ' ')} run")
    args = parser.parse_args()
    for name in RUNS:
        RUNS[name] = getattr(args, name)
    copy_results()
    SUMMARY.mkdir(parents=True, exist_ok=True)
    figures = {
        "arrival.png": plot_arrival(),
        "portrait.png": plot_portrait(),
        "first_step.png": plot_first_step(),
        "hands.png": plot_hands(),
        "funnel.png": plot_funnel(),
        "joining.png": plot_joining(),
        "timing.png": plot_timing(),
        "states.png": plot_states(),
        "settling.png": plot_settling(),
        "series.png": plot_series(),
        "warmup.png": plot_warmup(),
        "pulse_joints.png": plot_pulse_series("joints"),
        "pulse_torques.png": plot_pulse_series("torques"),
        "pulse_paths.png": plot_pulse_paths(),
        "pulse_maps.png": plot_pulse_maps(6.0),
        "pulse_maps_3n.png": plot_pulse_maps(3.0),
        "pulse_progress.png": plot_pulse_progress(),
        "fixed_points.png": plot_fixed_points(),
        "spectra.png": plot_spectra(),
        "held.png": plot_held(),
        "rates.png": plot_rates(),
    }
    for name, fig in figures.items():
        fig.savefig(SUMMARY / name, dpi=DPI)
    print_global_starts()
    print_warmup()
    print_pulses()
    print_fixed_points()
    print(f"\nWrote {', '.join(figures)} to {SUMMARY}")


# ---------------------------------------------------------------------------- the runs


def run_dir(name: str) -> Path:
    """One of the study's runs, under the storage root."""
    return resolve_run_path(f"{STUDY}/{RUNS[name]}")


def copy_results() -> None:
    """Copy each run's configuration, run record, and CSV files into ``results/<run>``."""
    for name in RUNS:
        if not RUNS[name]:
            continue
        source = run_dir(name)
        target = RESULTS / RUNS[name]
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(source.glob("*.toml")) + sorted(source.glob("*.csv")):
            shutil.copy2(path, target / path.name)


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


@functools.cache
def config_of(name: str) -> dict[str, Any]:
    with (run_dir(name) / "config.toml").open("rb") as f:
        return tomllib.load(f)


@functools.cache
def starts(key: str) -> list[dict[str, Any]]:
    """Step 1's rows of one ESN, one per start posture."""
    return read_csv(run_dir("global_starts") / f"starts_{key}.csv")


@functools.cache
def runs(key: str) -> dict[str, NDArray[Any]]:
    """Step 1's arrays of one ESN."""
    with np.load(run_dir("global_starts") / f"runs_{key}.npz") as data:
        return {name: data[name] for name in data.files}


def grid_step() -> float:
    return float(config_of("global_starts")["grid"]["step_deg"])


def grid_values() -> NDArray[np.float64]:
    """The start postures' values of each joint (deg)."""
    step = grid_step()
    return np.arange(-180.0, 180.0 - step / 2, step)


def grid_of(values: list[float] | NDArray[np.float64]) -> NDArray[np.float64]:
    """Values, one per start posture, as an image: a row per joint 2's value, a column per joint 1's."""
    n = len(grid_values())
    return np.asarray(values, dtype=np.float64).reshape(n, n)


def grid_extent() -> tuple[float, float, float, float]:
    step = grid_step()
    return (-180.0 - step / 2, 180.0 - step / 2, -180.0 - step / 2, 180.0 - step / 2)


def start_index(q_deg: tuple[float, float]) -> int:
    """The index of the start posture ``q_deg``."""
    q = runs(ESNS[0][0])["start_q_deg"]
    return int(np.argmin(np.linalg.norm(q - np.asarray(q_deg), axis=1)))


def every(step_deg: float) -> list[int]:
    """The start postures on a coarser grid, every ``step_deg`` in both joints."""
    q = runs(ESNS[0][0])["start_q_deg"]
    on = np.all(
        np.isclose(np.mod(q + 180.0, step_deg), 0.0) | np.isclose(np.mod(q + 180.0, step_deg), step_deg), axis=1
    )
    return [int(i) for i in np.flatnonzero(on)]


@functools.cache
def pulse_rows() -> list[dict[str, Any]]:
    return read_csv(run_dir("pulses") / "metrics.csv")


@functools.cache
def pulse_runs() -> dict[str, NDArray[Any]]:
    with np.load(run_dir("pulses") / "runs.npz") as data:
        return {name: data[name] for name in data.files}


def pulse_index(arm: str, pulse: tuple[float, float, float] | None) -> int:
    """The index of an arm's run under ``pulse``, (onset, angle, force), or undisturbed (None)."""
    data = pulse_runs()
    if pulse is None:
        match = (data["arm"] == arm) & ~np.isfinite(data["onset"])
    else:
        onset, angle, force = pulse
        match = (
            (data["arm"] == arm) & (data["onset"] == onset) & (data["angle_deg"] == angle) & (data["force_n"] == force)
        )
    (index,) = np.flatnonzero(match)
    return int(index)


def pulse_row(arm: str, pulse: tuple[float, float, float] | None) -> dict[str, Any]:
    return pulse_rows()[pulse_index(arm, pulse)]


# ---------------------------------------------------------------------------- the arm and the take


@functools.cache
def skeleton() -> Skeleton:
    return Skeleton.from_toml(TAKE_RUN / "config.toml")


@functools.cache
def taught() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The take filtered at 2 Hz, what the ESNs were trained on, every 10 ms: its joint angles and its hand path."""
    q = resample_joint_angles(StateLog.load(TAKE_RUN / "demo_00_filtered.sklog.npz"), DT)[1]
    return q, endpoint_positions(skeleton(), q)


@functools.cache
def target() -> tuple[NDArray[np.float64], float]:
    with np.load(run_dir("global_starts") / "take.npz") as data:
        return np.asarray(data["target"]), float(data["radius"])


def taught_arrival() -> float:
    """When the take's hand arrives within the goal radius (s)."""
    hand = taught()[1]
    goal, radius = target()
    return float(DT * np.argmax(np.linalg.norm(hand - goal, axis=1) <= radius))


def arm_points(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """The base, every joint, and the tip of the arm at ``q``."""
    arm = skeleton().clone()
    for link, angle in zip(arm.links[1:], q, strict=True):
        link.q = float(angle)
    compute_forward_kinematics(arm)
    return np.array([[0.0, 0.0]] + [[link.xe, link.ye] for link in arm.links[1:]])


# ---------------------------------------------------------------------------- drawing


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def new_figure(width: float, height: float, title: str) -> Figure:
    fig = Figure(figsize=(width, height), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(title, color=TITLE_COLOR)
    return fig


def draw_take_in_joint_space(ax: Axes, *, start: bool = True) -> None:
    """The take's joint path (thick gray), its start (circle), and its end (star), in degrees."""
    q = np.degrees(taught()[0])
    ax.plot(q[:, 0], q[:, 1], color=TAUGHT_COLOR, linewidth=4.0, zorder=4, solid_capstyle="round")
    if start:
        ax.plot(*q[0], "o", color=TITLE_COLOR, markersize=6, markerfacecolor="white", zorder=6)
    ax.plot(*q[-1], "*", color=TITLE_COLOR, markersize=12, markerfacecolor="white", zorder=6)


def joint_axes(ax: Axes) -> None:
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-180, 181, 60))
    ax.set_aspect("equal")
    ax.set_xlabel("joint 1 (deg)", color=TEXT_COLOR)
    ax.set_ylabel("joint 2 (deg)", color=TEXT_COLOR)


def take_handles() -> list[Line2D]:
    return [
        Line2D([], [], color=TAUGHT_COLOR, linewidth=4.0, label="the take's path"),
        Line2D([], [], linestyle="none", marker="o", color=TITLE_COLOR, markerfacecolor="white", label="its start"),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="*",
            markersize=12,
            color=TITLE_COLOR,
            markerfacecolor="white",
            label="its end, at the target",
        ),
    ]


def time_lines(
    ax: Axes, points: NDArray[np.float64], times: NDArray[np.float64], norm: Normalize, width: float
) -> None:
    """A line through ``points`` colored by time."""
    segments = np.stack([points[:-1], points[1:]], axis=1)
    lines = LineCollection(list(segments), cmap="viridis", norm=norm, linewidth=width, zorder=3)
    lines.set_array(times[:-1])
    ax.add_collection(lines)


# ---------------------------------------------------------------------------- step 1


def plot_arrival() -> Figure:
    """The arrival time over the start postures, for each ESN."""
    fig = new_figure(
        13.5,
        6.8,
        "Run on its own from rest at each start posture: when the hand arrives within the goal radius"
        " (gray: never, or not at rest at the target at the end)",
    )
    axes = fig.subplots(1, len(ESNS))
    times = [[row["arrival_time_s"] for row in starts(key)] for key, *_ in ESNS]
    finite = np.concatenate([np.asarray(t)[np.isfinite(t)] for t in times])
    norm = Normalize(float(finite.min()), float(finite.max()))
    cmap = colormaps["Blues"].with_extremes(bad=MISSING_COLOR)
    for ax, (key, label, *_), arrival in zip(axes, ESNS, times, strict=True):
        style(ax)
        at_target = np.array([row["outcome"] == "target" for row in starts(key)])
        data = np.where(at_target, arrival, np.nan)
        ax.imshow(grid_of(data), origin="lower", extent=grid_extent(), cmap=cmap, norm=norm, zorder=1)
        draw_take_in_joint_space(ax)
        ax.add_patch(Rectangle((15.0, 15.0), 30.0, 30.0, fill=False, linestyle="--", color=TITLE_COLOR, zorder=5))
        joint_axes(ax)
        counts = {name: sum(row["outcome"] == name for row in starts(key)) for name in OUTCOMES}
        ax.set_title(
            f"{label}: " + ", ".join(f"{name} {count}" for name, count in counts.items() if count),
            color=TITLE_COLOR,
            fontsize=10,
        )
    fig.colorbar(ScalarMappable(norm=norm, cmap="Blues"), ax=list(axes), label="arrival time (s)", shrink=0.8)
    handles = [
        *take_handles(),
        Line2D([], [], color=TITLE_COLOR, linestyle="--", label="report 011's 49 offset starts"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_portrait() -> Figure:
    """The runs from every 30 deg in joint space: the first step (dashed), then the run colored by time."""
    fig = new_figure(
        13.5,
        7.2,
        "Run on its own from rest: the runs from every 30° (dots) in joint space, their first step (dashed),"
        " and the rest of each run, colored by time",
    )
    axes = fig.subplots(1, len(ESNS))
    times = runs(ESNS[0][0])["times"]
    norm = Normalize(0.0, COLOR_TIME)
    for ax, (key, label, *_) in zip(axes, ESNS, strict=True):
        style(ax)
        data = runs(key)
        for i in every(30.0):
            q = np.degrees(data["q"][i].astype(np.float64))
            ax.plot(q[:2, 0], q[:2, 1], color=TEXT_COLOR, linestyle="--", linewidth=0.6, zorder=2)
            ax.plot(*q[0], "o", color=TEXT_COLOR, markersize=2.5, zorder=2)
            time_lines(ax, q[1:], times[1:], norm, 1.0)
        draw_take_in_joint_space(ax)
        joint_axes(ax)
        ax.set_title(label, color=TITLE_COLOR, fontsize=10)
    fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=list(axes), label="time (s)", shrink=0.8, extend="max")
    fig.legend(handles=take_handles(), loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_first_step() -> Figure:
    """How far the first step jumps from each start posture, and where it lands."""
    fig = new_figure(
        13.5,
        13.0,
        "Run on its own from rest: how far the ESN's first output is from the start posture (top),\nand where it"
        " lands (bottom), colored by the start's distance from the take's start",
    )
    axes = fig.subplots(2, len(ESNS))
    jumps = [[row["first_step_deg"] for row in starts(key)] for key, *_ in ESNS]
    norm = Normalize(0.0, float(max(max(j) for j in jumps)))
    origin = np.degrees(taught()[0][0])
    far = Normalize(0.0, 255.0)
    for col, (key, label, *_) in enumerate(ESNS):
        ax = axes[0, col]
        style(ax)
        ax.imshow(grid_of(jumps[col]), origin="lower", extent=grid_extent(), cmap="Blues", norm=norm, zorder=1)
        draw_take_in_joint_space(ax)
        joint_axes(ax)
        ax.set_title(label, color=TITLE_COLOR, fontsize=10)
        ax = axes[1, col]
        style(ax)
        data = runs(key)
        first = np.degrees(data["q"][:, 1].astype(np.float64))
        distance = np.linalg.norm(data["start_q_deg"] - origin, axis=1)
        ax.scatter(first[:, 0], first[:, 1], c=distance, cmap="viridis", norm=far, s=6, zorder=3)
        draw_take_in_joint_space(ax)
        joint_axes(ax)
    fig.colorbar(ScalarMappable(norm=norm, cmap="Blues"), ax=list(axes[0]), label="first step (deg)", shrink=0.8)
    fig.colorbar(ScalarMappable(norm=far, cmap="viridis"), ax=list(axes[1]),
                 label="start's distance from the take's start (deg)", shrink=0.8)  # fmt: skip
    fig.legend(handles=take_handles(), loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_hands() -> Figure:
    """The hand paths of the runs from every 30 deg, colored by time."""
    fig = new_figure(
        13.5,
        7.0,
        "Run on its own from rest: the hand's paths from every 30°, the first step dashed, then colored by time",
    )
    axes = fig.subplots(1, len(ESNS), sharex=True, sharey=True)
    times = runs(ESNS[0][0])["times"]
    norm = Normalize(0.0, COLOR_TIME)
    goal, radius = target()
    reach = float(np.linalg.norm(arm_points(np.zeros(2))[-1]))
    for ax, (key, label, *_) in zip(axes, ESNS, strict=True):
        style(ax)
        data = runs(key)
        for i in every(30.0):
            hand = endpoint_positions(skeleton(), data["q"][i].astype(np.float64))
            ax.plot(hand[:2, 0], hand[:2, 1], color=TEXT_COLOR, linestyle="--", linewidth=0.5, zorder=2)
            time_lines(ax, hand[1:], times[1:], norm, 0.8)
        ax.plot(*taught()[1].T, color=TAUGHT_COLOR, linewidth=4.0, zorder=4)
        ax.add_patch(Circle(tuple(goal), radius, fill=False, color=TITLE_COLOR, linewidth=1.2, zorder=5))
        ax.add_patch(Circle((0.0, 0.0), reach, fill=False, color=GRID_COLOR, linewidth=1.0, zorder=1))
        ax.plot(0.0, 0.0, "o", color=TITLE_COLOR, markersize=5)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)", color=TEXT_COLOR)
        ax.set_title(label, color=TITLE_COLOR, fontsize=10)
    axes[0].set_ylabel("y (m)", color=TEXT_COLOR)
    fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=list(axes), label="time (s)", shrink=0.8, extend="max")
    handles = [
        Line2D([], [], color=TAUGHT_COLOR, linewidth=4.0, label="the take's hand path"),
        Line2D([], [], color=TITLE_COLOR, linewidth=1.2, label="the goal radius"),
        Line2D([], [], color=GRID_COLOR, linewidth=1.0, label="the arm's reach"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_funnel() -> Figure:
    """Where the runs spend their time: the distance from the take's joint path against the progress along it."""
    threshold = config_of("global_starts")["analysis"]["path_threshold_deg"]
    fig = new_figure(
        13.5,
        5.8,
        "Run on its own from rest: the time the runs spend at each distance from the take's joint path and progress"
        f" along it, after the first step;\ndashed: {threshold:g}°, within which a run has joined the path",
    )
    axes = fig.subplots(1, len(ESNS), sharex=True, sharey=True)
    progress_edges = np.linspace(0.0, 1.0, 101)
    distance_edges = np.logspace(-2, np.log10(400.0), 81)
    counts = []
    for key, *_ in ESNS:
        data = runs(key)
        distance = np.maximum(data["path_distance_deg"][:, 1:].ravel(), 0.0101)
        counts.append(np.histogram2d(data["progress"][:, 1:].ravel(), distance, [progress_edges, distance_edges])[0])
    norm = LogNorm(DT, float(max(c.max() for c in counts)) * DT)
    for ax, (_, label, *_), count in zip(axes, ESNS, counts, strict=True):
        style(ax)
        ax.pcolormesh(progress_edges, distance_edges, np.where(count > 0, count * DT, np.nan).T, cmap="Blues",
                      norm=norm, zorder=1)  # fmt: skip
        ax.axhline(threshold, color=TITLE_COLOR, linestyle="--", linewidth=1.0, zorder=2)
        ax.set_yscale("log")
        ax.set_xlabel("progress along the take's joint path", color=TEXT_COLOR)
        ax.set_title(label, color=TITLE_COLOR, fontsize=10)
    axes[0].set_ylabel("distance from the path (deg)", color=TEXT_COLOR)
    fig.colorbar(ScalarMappable(norm=norm, cmap="Blues"), ax=list(axes), label="time spent, all runs (s)", shrink=0.9)
    return fig


def plot_joining() -> Figure:
    """The progress along the take's path at which each run joins it, over the start postures."""
    threshold = config_of("global_starts")["analysis"]["path_threshold_deg"]
    fig = new_figure(
        13.5,
        6.8,
        f"Run on its own from rest: the progress along the take's joint path when the run comes within {threshold:g}°"
        " of it for good (gray: never)",
    )
    axes = fig.subplots(1, len(ESNS))
    norm = Normalize(0.0, 1.0)
    cmap = colormaps["Blues"].with_extremes(bad=MISSING_COLOR)
    for ax, (key, label, *_) in zip(axes, ESNS, strict=True):
        style(ax)
        ax.imshow(grid_of([row["join_progress"] for row in starts(key)]), origin="lower", extent=grid_extent(),
                  cmap=cmap, norm=norm, zorder=1)  # fmt: skip
        draw_take_in_joint_space(ax)
        joint_axes(ax)
        ax.set_title(label, color=TITLE_COLOR, fontsize=10)
    fig.colorbar(ScalarMappable(norm=norm, cmap="Blues"), ax=list(axes), label="progress when it joins", shrink=0.8)
    fig.legend(handles=take_handles(), loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def taught_time_at(progress: NDArray[np.float64]) -> NDArray[np.float64]:
    """When the take first reaches each ``progress`` along its joint path (s)."""
    q = taught()[0]
    reached = np.maximum.accumulate(path_progress(q, q))
    return np.interp(progress, reached, DT * np.arange(len(q)))


def plot_timing() -> Figure:
    """The arrival time against the start: its distance from the target's posture, and its progress on the path."""
    threshold = config_of("global_starts")["analysis"]["path_threshold_deg"]
    arrival = taught_arrival()
    fig = new_figure(
        13.5,
        8.6,
        "Run on its own from rest: when the hand arrives, against how far the start is from the take's end posture"
        f" (top),\nand, for the starts within {threshold:g}° of the take's path, against their progress along it"
        " (bottom)",
    )
    axes = fig.subplots(2, len(ESNS), sharey=True)
    end = np.degrees(taught()[0][-1])
    progress = np.linspace(0.0, 1.0, 101)
    for col, (key, label, color, marker) in enumerate(ESNS):
        rows = starts(key)
        q = np.array([[row["q1_deg"], row["q2_deg"]] for row in rows])
        t = np.array([row["arrival_time_s"] for row in rows])
        ax = axes[0, col]
        style(ax)
        ax.plot(np.linalg.norm(q - end, axis=1), t, linestyle="none", marker=marker, markersize=3, color=color,
                markerfacecolor="none", alpha=0.6)  # fmt: skip
        ax.axhline(arrival, color=TEXT_COLOR, linestyle=":", linewidth=1.2)
        ax.set_xlabel("start's distance from the take's end posture (deg)", color=TEXT_COLOR)
        ax.set_title(label, color=TITLE_COLOR, fontsize=10)
        ax = axes[1, col]
        style(ax)
        near = np.array([row["start_path_distance_deg"] <= threshold for row in rows])
        start_progress = np.array([row["start_progress"] for row in rows])
        ax.plot(start_progress[near], t[near], linestyle="none", marker=marker, markersize=5, color=color,
                markerfacecolor="none")  # fmt: skip
        ax.plot(progress, np.full_like(progress, arrival), color=TEXT_COLOR, linestyle=":", linewidth=1.2)
        ax.plot(progress, np.maximum(arrival - taught_time_at(progress), 0.0), color=TEXT_COLOR, linestyle="--",
                linewidth=1.2)  # fmt: skip
        ax.set_xlim(0, 1)
        ax.set_xlabel("start's progress along the take's joint path", color=TEXT_COLOR)
    for row in range(2):
        axes[row, 0].set_ylabel("arrival time (s)", color=TEXT_COLOR)
        axes[row, 0].set_ylim(-0.5, None)
    handles = [
        Line2D(
            [],
            [],
            color=TEXT_COLOR,
            linestyle=":",
            label=f"the take's arrival, {arrival:.2f} s: a clock from the reset",
        ),
        Line2D([], [], color=TEXT_COLOR, linestyle="--", label="the rest of the take from that progress"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_states() -> Figure:
    """The reservoir states against the take's: the first two components, the phase, and the distance."""
    fig = new_figure(
        13.5,
        13.0,
        "Run on its own from rest: the reservoir states against the take's (fed in, as in training)",
    )
    axes = fig.subplots(3, len(ESNS))
    times = runs(ESNS[0][0])["state_times"]
    norm = Normalize(0.0, COLOR_TIME)
    for col, (key, label, *_) in enumerate(ESNS):
        data = runs(key)
        explained = data["explained"]
        ax = axes[0, col]
        style(ax)
        for i in every(30.0):
            time_lines(ax, data["projections"][i, :, :2].astype(np.float64), times, norm, 0.8)
        take = data["take_projections"]
        ax.plot(take[:, 0], take[:, 1], color=TAUGHT_COLOR, linewidth=4.0, zorder=4)
        ax.plot(*take[0, :2], "o", color=TITLE_COLOR, markerfacecolor="white", zorder=6)
        ax.plot(*take[-1, :2], "*", color=TITLE_COLOR, markersize=12, markerfacecolor="white", zorder=6)
        ax.set_xlabel(f"component 1 ({100 * explained[0]:.0f}%)", color=TEXT_COLOR)
        ax.set_ylabel(f"component 2 ({100 * explained[1]:.0f}%)", color=TEXT_COLOR)
        ax.set_title(f"{label}: the runs from every 30°, colored by time; the take (thick gray)", color=TITLE_COLOR,
                     fontsize=10)  # fmt: skip
        ax = axes[1, col]
        style(ax)
        phase = data["phase_s"]
        for i in range(len(phase)):
            ax.plot(times, phase[i], color=ESNS[col][2], linewidth=0.4, alpha=0.08)
        ax.plot(times, np.median(phase, axis=0), color=TITLE_COLOR, linewidth=1.5)
        ax.plot(times, np.minimum(times, DT * (len(taught()[0]) - 1)), color=TEXT_COLOR, linestyle="--", linewidth=1.0)
        ax.set_xlabel("time (s)", color=TEXT_COLOR)
        ax.set_ylabel("phase: the time of the take's nearest state (s)", color=TEXT_COLOR)
        ax.set_title("every run's phase (median black; dashed: the take's own timing)", color=TITLE_COLOR, fontsize=10)
        ax = axes[2, col]
        style(ax)
        for name, line in (("state_distance", "-"), ("matched_distance", "--")):
            values = data[name]
            low, mid, high = np.percentile(values, [10, 50, 90], axis=0)
            ax.fill_between(times, low, high, color=ESNS[col][2], alpha=0.15 if line == "-" else 0.08, linewidth=0)
            ax.plot(times, mid, color=ESNS[col][2], linestyle=line, linewidth=1.5)
        ax.set_yscale("log")
        ax.set_xlabel("time (s)", color=TEXT_COLOR)
        ax.set_ylabel("distance, relative to the take's spread", color=TEXT_COLOR)
        ax.set_title("distance from the take's nearest state (solid) and from its state at the same time\n(dashed):"
                     " median, 10% to 90%", color=TITLE_COLOR, fontsize=10)  # fmt: skip
    fig.colorbar(
        ScalarMappable(norm=norm, cmap="viridis"), ax=list(axes[0]), label="time (s)", shrink=0.8, extend="max"
    )
    return fig


def plot_settling() -> Figure:
    """How far the state still moves after the warm-up with the start posture held, and how each step shrinks."""
    hold = config_of("global_starts")["settling"]["hold"]
    fig = new_figure(
        13.5,
        11.0,
        f"Each start posture held for {hold:g} s: how far the state moves from the end of the 1 s warm-up to"
        f" {hold:g} s (top),\nand the size of each step while it is held (bottom), relative to the spread of the"
        " take's states",
    )
    axes = fig.subplots(2, len(ESNS))
    change = [[row["settling_change"] for row in starts(key)] for key, *_ in ESNS]
    norm = Normalize(0.0, float(max(max(c) for c in change)))
    for col, (key, label, color, _) in enumerate(ESNS):
        ax = axes[0, col]
        style(ax)
        ax.imshow(grid_of(change[col]), origin="lower", extent=grid_extent(), cmap="Blues", norm=norm, zorder=1)
        draw_take_in_joint_space(ax)
        joint_axes(ax)
        ax.set_title(label, color=TITLE_COLOR, fontsize=10)
        ax = axes[1, col]
        style(ax)
        data = runs(key)
        held_times, steps = data["held_times"], data["held_step"]
        low, mid, high = np.percentile(steps, [10, 50, 90], axis=0)
        ax.fill_between(held_times, low, high, color=color, alpha=0.2, linewidth=0)
        ax.plot(held_times, mid, color=color, linewidth=1.5)
        i = start_index((30.0, 30.0))
        ax.plot(held_times, steps[i], color=TITLE_COLOR, linewidth=1.0)
        ax.axvline(0.0, color=TEXT_COLOR, linestyle="--", linewidth=1.0)
        ax.axhline(taught_state_step(key), color=TEXT_COLOR, linestyle=":", linewidth=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("time on the task clock, the warm-up ending at 0 (s)", color=TEXT_COLOR)
        ax.set_ylabel("size of a step (relative)", color=TEXT_COLOR)
        ax.set_title("median and 10% to 90% over the starts; black: the take's start;\ndotted: the take's states'"
                     " median step during its reach", color=TITLE_COLOR, fontsize=10)  # fmt: skip
    fig.colorbar(ScalarMappable(norm=norm, cmap="Blues"), ax=list(axes[0]), label="state's move after the warm-up",
                 shrink=0.8)  # fmt: skip
    return fig


@functools.cache
def esn(key: str) -> ReachingEsn:
    """One of the ESNs, loaded from its model."""
    model = next(table["model"] for table in config_of("global_starts")["esn"] if table["key"] == key)
    return ReachingEsn.load(resolve_run_path(model))


def taught_state_step(key: str) -> float:
    """The median size of a step of the take's states during its reach, fed in, relative to their spread."""
    trained = esn(key)
    states = teacher_forced_states(trained, taught()[0])[trained.config.warmup_steps :]
    steps = np.linalg.norm(np.diff(states, axis=0), axis=1) / float(runs(key)["spread"])
    reach = slice(onset_index(taught()[1]) or 0, round(taught_arrival() / DT))  # from when the take's hand moves
    return float(np.median(steps[reach]))


def plot_series() -> Figure:
    """The joint angles over time from six starts, for each ESN."""
    fig = new_figure(
        13.5, 7.6, "Run on its own from rest: the joint angles over time from six starts; thick gray: the take"
    )
    axes = fig.subplots(2, len(ESNS), sharex=True, sharey="row")
    q_taught = np.degrees(taught()[0])
    t_taught = DT * np.arange(len(q_taught))
    for col, (key, label, *_) in enumerate(ESNS):
        data = runs(key)
        for joint in range(2):
            ax = axes[joint, col]
            style(ax)
            ax.plot(t_taught, q_taught[:, joint], color=TAUGHT_COLOR, linewidth=4.0, zorder=1)
            for start, color in zip(SHOWN_STARTS, SHOWN_COLORS, strict=True):
                i = start_index(start)
                ax.plot(data["times"], np.degrees(data["q"][i, :, joint]), color=color, linewidth=1.2, zorder=3)
            ax.set_ylabel(f"joint {joint + 1} (deg)", color=TEXT_COLOR)
        axes[0, col].set_title(label, color=TITLE_COLOR, fontsize=10)
        axes[1, col].set_xlabel("time (s)", color=TEXT_COLOR)
    handles = [
        Line2D([], [], color=color, linewidth=1.2, label=f"from ({start[0]:g}°, {start[1]:g}°)")
        for start, color in zip(SHOWN_STARTS, SHOWN_COLORS, strict=True)
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


# ---------------------------------------------------------------------------- step 2


def pulse_words(pulse: tuple[float, float, float] | None) -> str:
    if pulse is None:
        return "undisturbed"
    onset, angle, force = pulse
    direction = {0.0: "forward", 90.0: "across, counterclockwise", 180.0: "backward", 270.0: "across, clockwise"}
    return f"{force:g} N at {onset:g} s, {direction.get(angle, f'{angle:g}°')}"


def arm_handles(*, references: bool) -> list[Line2D]:
    handles = [Line2D([], [], color=color, linewidth=1.4, label=label) for _, label, color, _ in ARMS]
    if references:
        handles += [
            Line2D([], [], color=TEXT_COLOR, linewidth=1.0, linestyle="--", label="its reference"),
            Line2D([], [], color=TAUGHT_COLOR, linewidth=4.0, label="the taught motion"),
        ]
    return handles


def plot_pulse_series(channel: str) -> Figure:
    """The joint angles or torques of the four arms over time, undisturbed and under four pulses: a column each."""
    data = pulse_runs()
    times = data["times"]
    what = "Joint angles" if channel == "joints" else "Joint torques"
    fig = new_figure(
        3.6 * len(SHOWN_PULSES) + 1.0,
        9.6,
        f"{what} on the robot, joint PD at ω 20, ζ 0.4, given the reference velocity: ESN #1 (blue), ESN #2 (purple),"
        " the replay (orange), and the progress replay (green)"
        + ("\ndashed: the references of the ESNs and the progress replay" if channel == "joints" else ""),
    )
    axes = fig.subplots(3, len(SHOWN_PULSES), sharex=True, sharey="row", height_ratios=[0.6, 1.0, 1.0])
    q_taught = np.degrees(taught()[0])
    t_taught = DT * np.arange(len(q_taught))
    duration = config_of("pulses")["pulses"]["duration"]
    for col, pulse in enumerate(SHOWN_PULSES):
        for row in range(3):
            style(axes[row, col])
            if pulse is not None:
                axes[row, col].axvspan(pulse[0], pulse[0] + duration, color=DISTURBANCE_COLOR, zorder=0)
        axes[0, col].set_title(pulse_words(pulse), color=TITLE_COLOR, fontsize=10)
        for key, _, color, _ in ARMS:
            i = pulse_index(key, pulse)
            axes[0, col].plot(times, np.linalg.norm(data["ext_force"][i], axis=1), color=color, linewidth=1.0,
                              zorder=ARM_ZORDERS[key])  # fmt: skip
            for joint in range(2):
                ax = axes[1 + joint, col]
                if channel == "joints":
                    ax.plot(
                        times, np.degrees(data["q"][i, :, joint]), color=color, linewidth=1.2, zorder=ARM_ZORDERS[key]
                    )
                    if key in REFERENCE_ZORDERS:
                        ax.plot(times, np.degrees(data["q_ref"][i, :, joint]), color=color, linewidth=1.0,
                                linestyle="--", zorder=REFERENCE_ZORDERS[key])  # fmt: skip
                else:
                    ax.plot(times, data["tau"][i, :, joint], color=color, linewidth=1.0, zorder=ARM_ZORDERS[key])
        if channel == "joints":
            for joint in range(2):
                axes[1 + joint, col].plot(t_taught, q_taught[:, joint], color=TAUGHT_COLOR, linewidth=4.0, zorder=1)
        axes[-1, col].set_xlabel("time (s)", color=TEXT_COLOR)
        axes[-1, col].set_xlim(0, float(times[-1]))
    axes[0, 0].set_ylabel("tip force (N)", color=TEXT_COLOR)
    unit = "deg" if channel == "joints" else "N m"
    for joint in range(2):
        axes[1 + joint, 0].set_ylabel(f"joint {joint + 1} ({unit})", color=TEXT_COLOR)
    handles = arm_handles(references=channel == "joints")
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_pulse_paths() -> Figure:
    """The tip's paths under the pulses of 6 N: a row per arm, a column per onset, the 8 directions overlaid."""
    data = pulse_runs()
    angles = sorted(config_of("pulses")["pulses"]["angles_deg"])
    cmap = colormaps["twilight"]
    fig = new_figure(
        3.2 * len(PATH_ONSETS) + 0.6,
        3.2 * len(ARMS) + 2.0,
        "The tip's paths on the robot under a pulse of 6 N for 0.2 s, in 8 directions (colored), with an arrow"
        f" along the force where the tip is at its onset ({1000 * FORCE_SCALE:g} mm per N)",
    )
    axes = fig.subplots(len(ARMS), len(PATH_ONSETS), sharex="col", sharey="col")
    goal, radius = target()
    for row, (key, label, _, _) in enumerate(ARMS):
        for col, onset in enumerate(PATH_ONSETS):
            ax = axes[row, col]
            style(ax)
            center = taught()[1][min(round(onset / DT), len(taught()[1]) - 1)]
            ax.set_xlim(center[0] - PATH_WINDOW, center[0] + PATH_WINDOW)
            ax.set_ylim(center[1] - PATH_WINDOW, center[1] + PATH_WINDOW)
            ax.plot(*taught()[1].T, color=TAUGHT_COLOR, linewidth=4.0, zorder=1)
            ax.add_patch(Circle(tuple(goal), radius, fill=False, color=TITLE_COLOR, linewidth=1.0, zorder=2))
            ax.plot(*center, "o", color=TITLE_COLOR, markersize=4, markerfacecolor="white", zorder=5)
            for angle in angles:
                i = pulse_index(key, (onset, angle, 6.0))
                color = cmap(angle / 360.0)
                hand = endpoint_positions(skeleton(), data["q"][i].astype(np.float64))
                ax.plot(*hand.T, color=color, linewidth=0.9, zorder=3)
                k = int(np.argmax(data["times"] > onset - 1e-9))
                force = data["ext_force"][i, k + 1].astype(np.float64)
                ax.add_patch(FancyArrow(*hand[k], *(FORCE_SCALE * force), width=0.003, length_includes_head=True,
                                        head_width=0.018, head_length=0.018, color=color, zorder=4))  # fmt: skip
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(f"onset {onset:g} s", color=TITLE_COLOR, fontsize=10)
            if row == len(ARMS) - 1:
                ax.set_xlabel("x (m)", color=TEXT_COLOR)
            if col == 0:
                ax.set_ylabel(f"{label}\ny (m)", color=TEXT_COLOR)
    norm = Normalize(0.0, 360.0)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=list(axes[-1]), label="direction from the target's (deg)",
                 location="bottom", shrink=0.5, ticks=angles)  # fmt: skip
    return fig


def plot_pulse_maps(force: float) -> Figure:
    """The measures of each pulse of ``force``: a row per measure, a column per arm, over the onsets and directions."""
    table = config_of("pulses")["pulses"]
    onsets, angles = sorted(table["onsets"]), sorted(table["angles_deg"])
    fig = new_figure(
        3.3 * len(ARMS) + 1.6,
        2.9 * len(PULSE_MEASURES) + 0.8,
        f"The pulses of {force:g} N for 0.2 s on the robot: each measure over the onset and the direction"
        " (0° forward, toward the target; 90° counterclockwise across; 180° backward)",
    )
    axes = fig.subplots(len(PULSE_MEASURES), len(ARMS), sharex=True, sharey=True)
    for row, (column, label, scale, signed) in enumerate(PULSE_MEASURES):
        grids = []
        for key, *_ in ARMS:
            grid = np.array([[scale * float(pulse_row(key, (onset, angle, force))[column]) for angle in angles]
                             for onset in onsets])  # fmt: skip
            grids.append(grid)
        values = np.concatenate([g[np.isfinite(g)] for g in grids])
        if signed:
            bound = float(np.abs(values).max()) or 1.0
            norm: Normalize = TwoSlopeNorm(0.0, -bound, bound)
            cmap = colormaps["PuOr_r"].with_extremes(bad=MISSING_COLOR)
        else:
            norm = Normalize(0.0, float(values.max()) or 1.0)
            cmap = colormaps["Blues"].with_extremes(bad=MISSING_COLOR)
        for col, ((_, arm_label, _, _), grid) in enumerate(zip(ARMS, grids, strict=True)):
            ax = axes[row, col]
            ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=norm,
                      extent=(-22.5, 337.5, -0.5, len(onsets) - 0.5))  # fmt: skip
            ax.set_xticks(angles)
            ax.set_yticks(range(len(onsets)), [f"{o:g}" for o in onsets])
            ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR, labelsize=8)
            if row == 0:
                ax.set_title(arm_label, color=TITLE_COLOR, fontsize=10)
            if row == len(PULSE_MEASURES) - 1:
                ax.set_xlabel("direction (deg)", color=TEXT_COLOR)
            if col == 0:
                ax.set_ylabel("onset (s)", color=TEXT_COLOR)
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=list(axes[row]), label=label, shrink=0.9)
    return fig


def plot_pulse_progress() -> Figure:
    """The progress along the take's path of each arm and its reference, under pushes forward and backward."""
    data = pulse_runs()
    times = data["times"]
    q_taught = taught()[0]
    shown = [(onset, angle) for angle in (0.0, 180.0) for onset in (2.0, 4.0, 6.0)]
    fig = new_figure(
        14.0,
        8.4,
        "Progress along the take's joint path on the robot under a pulse of 6 N, forward (top) and backward (bottom):"
        "\neach arm (solid), and the references of the ESNs and the progress replay (dashed); the replay's reference"
        " is the taught motion (thick gray)",
    )
    axes = fig.subplots(2, 3, sharex=True, sharey=True)
    duration = config_of("pulses")["pulses"]["duration"]
    for ax, (onset, angle) in zip(axes.ravel(), shown, strict=True):
        style(ax)
        ax.axvspan(onset, onset + duration, color=DISTURBANCE_COLOR, zorder=0)
        ax.plot(DT * np.arange(len(q_taught)), path_progress(q_taught, q_taught), color=TAUGHT_COLOR, linewidth=4.0,
                zorder=1)  # fmt: skip
        for key, _, color, _ in ARMS:
            i = pulse_index(key, (onset, angle, 6.0))
            ax.plot(times, data["progress"][i], color=color, linewidth=1.2, zorder=ARM_ZORDERS[key])
            if key in REFERENCE_ZORDERS:
                ax.plot(times, path_progress(data["q_ref"][i].astype(np.float64), q_taught), color=color,
                        linewidth=1.0, linestyle="--", zorder=REFERENCE_ZORDERS[key])  # fmt: skip
        ax.set_xlim(0, 14)
        ax.set_title(pulse_words((onset, angle, 6.0)), color=TITLE_COLOR, fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", color=TEXT_COLOR)
    for ax in axes[:, 0]:
        ax.set_ylabel("progress along the path", color=TEXT_COLOR)
    handles = arm_handles(references=True)
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


# ---------------------------------------------------------------------------- step 1b


@functools.cache
def warmup_rows(key: str) -> list[dict[str, Any]]:
    return read_csv(run_dir(f"warmup_{key}") / "warmup.csv")


def warmup_shift(key: str, start: int) -> tuple[list[float], list[float]]:
    """The warm-ups of one start, and its arrival time with each, minus that with the trained 1 s warm-up."""
    rows = sorted((r for r in warmup_rows(key) if r["start"] == start), key=lambda r: r["warmup_s"])
    trained = next(r["arrival_time_s"] for r in rows if r["warmup_s"] == 1.0)
    return [r["warmup_s"] for r in rows], [r["arrival_time_s"] - trained for r in rows]


def plot_warmup() -> Figure:
    """The arrival against the warm-up, from the offset starts and from the starts on the take's path."""
    fig = new_figure(
        13.5,
        5.6,
        "Run on its own with other warm-ups than the trained 1 s: when the hand arrives, against the arrival with the"
        " 1 s warm-up;\ndotted: a clock from the reset; the 49 offset starts (median, and their range) and three"
        " starts on the take's path, at rest",
    )
    axes = fig.subplots(1, len(ESNS), sharey=True)
    path_colors = ["#e69f00", "#009e73", "#cc79a7"]
    for ax, (key, label, color, _) in zip(axes, ESNS, strict=True):
        style(ax)
        offsets = sorted({int(r["start"]) for r in warmup_rows(key) if r["group"] in ("offset", "demonstrated")})
        shifts = np.array([warmup_shift(key, start)[1] for start in offsets])
        warmups = warmup_shift(key, offsets[0])[0]
        ax.fill_between(warmups, shifts.min(axis=0), shifts.max(axis=0), color=color, alpha=0.15, linewidth=0)
        ax.plot(warmups, np.median(shifts, axis=0), color=color, linewidth=1.8, marker="o")
        paths = sorted({int(r["start"]) for r in warmup_rows(key) if r["group"] == "path"})
        for n, (start, path_color) in enumerate(zip(paths, path_colors, strict=True)):
            ax.plot(*warmup_shift(key, start), color=path_color, linewidth=1.2, marker="s", label=f"path {n}")
        line = np.array([min(warmups), max(warmups)])
        ax.plot(line, -(line - 1.0), color=TEXT_COLOR, linestyle=":", linewidth=1.2)
        ax.axhline(0.0, color=GRID_COLOR, linewidth=1.0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(warmups, [f"{w:g}" for w in warmups])
        ax.set_xlabel("warm-up (s)", color=TEXT_COLOR)
        ax.set_title(label, color=TITLE_COLOR, fontsize=10)
    axes[0].set_ylabel("arrival, later than with 1 s (s)", color=TEXT_COLOR)
    progress = ["a quarter", "half", "three quarters"]
    handles = [
        Line2D([], [], color=TEXT_COLOR, linewidth=1.8, marker="o", label="the 49 offset starts, median"),
        *[
            Line2D([], [], color=c, marker="s", label=f"from rest {p} of the way along the take's path")
            for c, p in zip(path_colors, progress, strict=True)
        ],
        Line2D([], [], color=TEXT_COLOR, linestyle=":", label="a clock from the reset"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    return fig


# ---------------------------------------------------------------------------- step 3


@functools.cache
def fixed_points(key: str) -> list[dict[str, Any]]:
    return read_csv(run_dir("fixed_points") / f"fixed_points_{key}.csv")


@functools.cache
def spectra(key: str) -> dict[str, NDArray[Any]]:
    with np.load(run_dir("fixed_points") / f"spectra_{key}.npz") as data:
        return {name: data[name] for name in data.files}


@functools.cache
def rates(key: str) -> dict[str, NDArray[Any]]:
    with np.load(run_dir("fixed_points") / f"rates_{key}.npz") as data:
        return {name: data[name] for name in data.files}


@functools.cache
def held_rows(key: str) -> list[dict[str, Any]]:
    return read_csv(run_dir("fixed_points") / f"starts_{key}.csv")


def target_point(key: str) -> int:
    """The index of the fixed point at the target."""
    return int(min(fixed_points(key), key=lambda r: r["target_distance_m"])["fixed_point"])


def plot_fixed_points() -> Figure:
    """Each fixed point's posture, and how its slowest modes move the posture around it."""
    fig = new_figure(
        13.5,
        6.8,
        "Fed its own output, where the ESN can rest: its fixed points (star: attracting; cross: unstable), and how"
        " the slowest modes move the posture around each,\nas ellipses scaled to 15° (labels: time constant or"
        " growth, and period, of the six slowest modes)",
    )
    axes = fig.subplots(1, len(ESNS))
    mode_colors = ["#0b0b0b", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9"]
    angle = np.linspace(0.0, 2 * np.pi, 121)
    for ax, (key, label, *_) in zip(axes, ESNS, strict=True):
        style(ax)
        draw_take_in_joint_space(ax, start=True)
        for row in fixed_points(key):
            j = int(row["fixed_point"])
            center = np.array([row["q1_deg"], row["q2_deg"]])
            stable = row["n_unstable"] == 0
            ax.plot(*center, "*" if stable else "X", color=TITLE_COLOR, markersize=14 if stable else 10,
                    markerfacecolor="white", zorder=7)  # fmt: skip
            eigenvalues = spectra(key)[f"eigenvalues_{j}"]
            modes = spectra(key)[f"modes_{j}"]
            for n, (mode, color) in enumerate(zip(modes, mode_colors, strict=False)):
                motion = np.degrees(np.real(np.outer(np.exp(1j * angle), mode)))
                size = np.abs(motion).max()
                if size == 0:
                    continue
                motion *= 15.0 / size
                ax.plot(center[0] + motion[:, 0], center[1] + motion[:, 1], color=color, linewidth=1.0, zorder=6,
                        label=mode_label(complex(eigenvalues[n])) if j == target_point(key) else None)  # fmt: skip
        joint_axes(ax)
        ax.set_xlim(-30, 180)
        ax.set_ylim(-30, 180)
        ax.set_title(label, color=TITLE_COLOR, fontsize=10)
        ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=TEXT_COLOR, title="at the target",
                  title_fontsize=8)  # fmt: skip
    return fig


def mode_label(eigenvalue: complex) -> str:
    """A mode's time constant (or growth time) and period."""
    modulus, angle = abs(eigenvalue), abs(np.angle(eigenvalue))
    rate = np.log(modulus) / DT
    timing = f"decays in {-1 / rate:.2f} s" if rate < 0 else f"grows in {1 / rate:.2f} s"
    return timing + (f", period {2 * np.pi * DT / angle:.1f} s" if angle > 1e-9 else "")


def plot_spectra() -> Figure:
    """The eigenvalues of the Jacobian at each fixed point, near 1: a panel per fixed point."""
    # The attracting fixed points first, those at the target, then the unstable ones.
    points = sorted(
        ((key, label, color, row) for key, label, color, _ in ESNS for row in fixed_points(key)),
        key=lambda point: point[3]["n_unstable"] > 0,
    )
    columns = (len(points) + 1) // 2
    fig = new_figure(
        4.6 * columns + 1.0,
        10.0,
        "The eigenvalues of the map's Jacobian at each fixed point, all within 0.04 of 0.97;\ngray: the unit circle,"
        " outside which a mode grows; dashed: 1 - the leak rate, where a neuron alone decays",
    )
    axes = np.asarray(fig.subplots(2, columns, sharex=True, sharey=True)).reshape(2, columns)
    circle = np.exp(1j * np.linspace(-0.1, 0.1, 201))
    for ax, (key, label, color, row) in zip(axes.ravel(), points, strict=False):
        style(ax)
        leak = esn(key).config.leak_rate
        ax.plot(np.real(circle), np.imag(circle), color=TAUGHT_COLOR, linewidth=1.5, zorder=1)
        inner = (1 - leak) * circle
        ax.plot(np.real(inner), np.imag(inner), color=TEXT_COLOR, linestyle="--", linewidth=0.8, zorder=1)
        eigenvalues = spectra(key)[f"eigenvalues_{int(row['fixed_point'])}"]
        outside = np.abs(eigenvalues) > 1.0
        ax.plot(np.real(eigenvalues)[~outside], np.imag(eigenvalues)[~outside], "o", color=color, markersize=3,
                markerfacecolor="none", zorder=3)  # fmt: skip
        ax.plot(
            np.real(eigenvalues)[outside], np.imag(eigenvalues)[outside], "x", color=PUSH_COLOR, markersize=8, zorder=4
        )
        stable = "attracting" if row["n_unstable"] == 0 else f"{int(row['n_unstable'])} unstable modes (red)"
        ax.set_title(f"{label}: at ({row['q1_deg']:.0f}°, {row['q2_deg']:.0f}°), {stable}", color=TITLE_COLOR,
                     fontsize=10)  # fmt: skip
        ax.set_aspect("equal")
        ax.set_xlim(0.93, 1.015)
        ax.set_ylim(-0.035, 0.035)
    for ax in axes[-1]:
        ax.set_xlabel("real part", color=TEXT_COLOR)
    for ax in axes[:, 0]:
        ax.set_ylabel("imaginary part", color=TEXT_COLOR)
    for ax in axes.ravel()[len(points) :]:
        ax.set_visible(False)
    return fig


def plot_held() -> Figure:
    """With each start posture held, how slowly the reservoir settles, and how far the warm-up leaves it."""
    fig = new_figure(
        13.5,
        11.6,
        "Each start posture held: the slowest time constant at the reservoir's fixed point (top), and how far the"
        " state after the 1 s warm-up is from it,\nrelative to the spread of the take's states (bottom)",
    )
    axes = fig.subplots(2, len(ESNS))
    tau = [[r["held_time_constant_s"] for r in held_rows(key)] for key, *_ in ESNS]
    gap = [[r["warmup_gap"] for r in held_rows(key)] for key, *_ in ESNS]
    tau_norm = Normalize(0.0, float(np.nanmax(tau)))
    gap_norm = Normalize(0.0, float(np.nanmax(gap)))
    cmap = colormaps["Blues"].with_extremes(bad=MISSING_COLOR)
    for col, (_, label, *_) in enumerate(ESNS):
        for row, (values, norm) in enumerate(((tau[col], tau_norm), (gap[col], gap_norm))):
            ax = axes[row, col]
            style(ax)
            ax.imshow(grid_of(values), origin="lower", extent=grid_extent(), cmap=cmap, norm=norm, zorder=1)
            draw_take_in_joint_space(ax)
            joint_axes(ax)
        axes[0, col].set_title(label, color=TITLE_COLOR, fontsize=10)
    fig.colorbar(ScalarMappable(norm=tau_norm, cmap="Blues"), ax=list(axes[0]), label="slowest time constant (s)",
                 shrink=0.8)  # fmt: skip
    fig.colorbar(ScalarMappable(norm=gap_norm, cmap="Blues"), ax=list(axes[1]), label="left from settled (relative)",
                 shrink=0.8)  # fmt: skip
    return fig


def plot_rates() -> Figure:
    """Along the ESN's run from the take's start: how fast changes across it grow or decay, and posture deviations."""
    fig = new_figure(
        13.5,
        9.2,
        "Along the ESN's own run from the take's start, over windows of 0.5 to 4 s starting at each time: the largest"
        " rate at which a change of the state across the run grows (top; negative: it decays),\nand the most a"
        " deviation of the output posture is scaled by the window's end (bottom); dotted: when the take arrives",
    )
    axes = fig.subplots(2, len(ESNS), sharex=True, sharey="row")
    windows = [float(w) for w in rates(ESNS[0][0])["windows_s"]]
    ramp = colormaps["viridis"]
    for col, (key, label, *_) in enumerate(ESNS):
        data = rates(key)
        for n, window in enumerate(windows):
            color = ramp(n / max(1, len(windows) - 1) * 0.85)
            starts_s = data[f"starts_s_{window:g}"]
            axes[0, col].plot(starts_s, data[f"across_{window:g}"][:, 0], color=color, linewidth=1.4,
                              label=f"{window:g} s")  # fmt: skip
            axes[1, col].plot(starts_s, data[f"posture_{window:g}"][:, 0], color=color, linewidth=1.4)
        for row in range(2):
            ax = axes[row, col]
            style(ax)
            ax.axvline(taught_arrival(), color=TEXT_COLOR, linestyle=":", linewidth=1.2)
        axes[0, col].axhline(0.0, color=TEXT_COLOR, linewidth=0.8)
        axes[1, col].axhline(1.0, color=TEXT_COLOR, linewidth=0.8)
        axes[0, col].set_title(label, color=TITLE_COLOR, fontsize=10)
        axes[1, col].set_xlabel("start of the window (s)", color=TEXT_COLOR)
        axes[1, col].set_yscale("log")
    axes[0, 0].set_ylabel("largest rate across the run (1/s)", color=TEXT_COLOR)
    axes[1, 0].set_ylabel("posture deviation, scaled by", color=TEXT_COLOR)
    axes[0, 0].legend(title="window", fontsize=8, title_fontsize=8, frameon=False, labelcolor=TEXT_COLOR)
    return fig


# ---------------------------------------------------------------------------- tables


def quantiles(values: list[float] | NDArray[np.float64], digits: int = 2) -> str:
    """The median and range of the finite ``values``."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return "n/a"
    return f"{np.median(finite):.{digits}f} ({finite.min():.{digits}f} to {finite.max():.{digits}f})"


def print_global_starts() -> None:
    print("\nStep 1, from rest anywhere: median (range) over the start postures")
    for key, label, *_ in ESNS:
        rows = starts(key)
        outcomes = {name: sum(row["outcome"] == name for row in rows) for name in OUTCOMES}
        print(f"\n{label}: {len(rows)} starts; " + ", ".join(f"{k} {v}" for k, v in outcomes.items()))
        for column, words, digits in [
            ("arrival_time_s", "arrival time (s)", 2),
            ("final_distance_m", "final distance from the target (m)", 4),
            ("max_path_distance_deg", "largest distance from the take's path (deg)", 1),
            ("join_progress", "progress when it joins the path", 2),
            ("join_time_s", "time when it joins the path (s)", 2),
            ("start_state_distance", "state at t = 0 from the take's nearest (relative)", 3),
            ("start_phase_s", "phase at t = 0 (s)", 2),
            ("settling_change", "state's move after the warm-up, held to 4 s (relative)", 3),
        ]:
            print(f"  {words}: {quantiles([row[column] for row in rows], digits)}")
        joined = sum(np.isfinite(row["join_progress"]) for row in rows)
        print(f"  {joined} of {len(rows)} join the take's path")


def print_warmup() -> None:
    print(
        "\nStep 1b, the warm-up: arrival, later than with 1 s (s), the median of the 49 offset starts; the path starts"
    )
    for key, label, *_ in ESNS:
        offsets = sorted({int(r["start"]) for r in warmup_rows(key) if r["group"] in ("offset", "demonstrated")})
        warmups = warmup_shift(key, offsets[0])[0]
        median = np.median([warmup_shift(key, start)[1] for start in offsets], axis=0)
        print(f"  {label}: " + ", ".join(f"{w:g} s {m:+.2f}" for w, m in zip(warmups, median, strict=True)))
        for start in sorted({int(r["start"]) for r in warmup_rows(key) if r["group"] == "path"}):
            rows = sorted((r for r in warmup_rows(key) if r["start"] == start), key=lambda r: r["warmup_s"])
            print(
                f"    path start {start}: arrival "
                + ", ".join(f"{r['warmup_s']:g} s {r['arrival_time_s']:.2f}{'' if r['success'] else '*'}" for r in rows)
            )
        failed = [r for r in warmup_rows(key) if not r["success"]]
        print(f"    {len(failed)} runs fail to arrive and hold (*), of {len(warmup_rows(key))}")


def print_fixed_points() -> None:
    print("\nStep 3, the fixed points")
    for key, label, *_ in ESNS:
        for row in fixed_points(key):
            print(
                f"  {label} {int(row['fixed_point'])}: ({row['q1_deg']:.2f}, {row['q2_deg']:.2f}) deg,"
                f" {1000 * row['target_distance_m']:.1f} mm from the target, {int(row['n_unstable'])} unstable,"
                f" max |λ| {row['max_modulus']:.5f}, time constant {row['time_constant_s']:.2f} s, period"
                f" {row['period_s']:.2f} s; from {int(row['from_starts'])} starts and {int(row['from_ends'])} ends"
            )
        rows = held_rows(key)
        print(
            f"  held: found {sum(r['held_converged'] for r in rows)} of {len(rows)}; time constant "
            + quantiles([r["held_time_constant_s"] for r in rows])
            + "; gap after the warm-up "
            + quantiles([r["warmup_gap"] for r in rows], 3)
            + "; max |λ| "
            + quantiles([r["held_max_modulus"] for r in rows], 4)
        )
        data = rates(key)
        for window in data["windows_s"]:
            across = data[f"across_{float(window):g}"][:, 0]
            posture = data[f"posture_{float(window):g}"][:, 0]
            print(
                f"  {float(window):g} s windows: largest rate across {quantiles(across)} /s,"
                f" posture scaled by {quantiles(posture)}"
            )


def print_pulses() -> None:
    print("\nStep 2, the pulses: median (range) over the 80 pulses of each force")
    for force in (3.0, 6.0):
        print(f"\n{force:g} N | " + " | ".join(label for _, label, _, _ in ARMS))
        for column, label, scale, _ in PULSE_MEASURES:
            cells = []
            for key, *_ in ARMS:
                values = [scale * float(r[column]) for r in pulse_rows() if r["arm"] == key and r["force_n"] == force]
                cells.append(quantiles(values, 2))
            print(f"  {label} | " + " | ".join(cells))
        for key, label, *_ in ARMS:
            failed = [r for r in pulse_rows() if r["arm"] == key and r["force_n"] == force and not r["success"]]
            print(f"  {label}: {len(failed)} pulsed runs fail to arrive and hold")
    print("\nUndisturbed:")
    for key, label, *_ in ARMS:
        row = pulse_row(key, None)
        print(f"  {label}: arrives at {row['arrival_time_s']:.2f} s, peak torque {row['peak_torque_nm']:.2f} N m")


if __name__ == "__main__":
    main()
