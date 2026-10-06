# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures, tables, and animations of report 003.

    uv run python reports/003-single-demonstration/make_figures.py
    uv run python reports/003-single-demonstration/make_figures.py --logs --animations

From the copies of the runs under ``results/`` (configurations, run records, and
per-run metrics), it draws into ``results/summary``:

- ``grids.png``: the autonomous runs of the three ESNs over the grid of start
  offsets (Section 3.1);
- ``warmup.png``: the arrival against the warm-up, for the two ESNs of Phase 1
  (Section 3.3);
- ``sweeps.png``: the two hyperparameter sweeps, condensed (Section 3.4);
- ``robot.png``: the arms in every robot scenario at the two featured tracker
  settings (Section 3.5);
- ``offsets.png``: the robot runs over the grid of start offsets (Section 3.5);
- ``gains.png``: the robot runs against the gain of joint PD (Sections 3.5 and 3.6);

and prints the tables of the report. Two options read the run logs, which are not
kept in Git, from the runs under the storage root, where
``experiments/robot_esn.py`` wrote them:

- ``--logs`` draws ``block.png`` (the block over time, Section 3.6) and the joint
  angles of the examples, scenario by scenario (``example_<name>.png``, Section
  3.8), and writes ``departures.csv``, ``offset_references.csv``, and
  ``push_progress.csv`` (Sections 3.6 and 3.7), from which their tables are
  printed (also without the option);
- ``--animations`` animates each example (``example_<name>.gif``): the four arms
  side by side, each rendered by skelarm's player.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from numpy.typing import NDArray
from skelarm import StateLog

REPORT = Path(__file__).resolve().parent
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"
DEMONSTRATION = REPORT / "data" / "20261005-185525-reach_tvs_single" / "demo_00.sklog.npz"

# The three ESNs trained on the demonstration: key, its name, and its run over the grid of start offsets.
ESNS = {
    "first": ("first settings", "20261005-190403-grid_single_demo_settings"),
    "eight": ("eight-demonstration settings", "20261005-190405-grid_multi_demo_settings"),
    "tuned": ("tuned settings", "20261005-212407-grid_tuned_settings"),
}
STATES_RUNS = {
    "first": "20261005-191929-states_single_demo_settings",
    "eight": "20261005-191932-states_multi_demo_settings",
}
WARMUP_RUNS = {
    "first": "20261005-204749-warmup_single_demo_settings",
    "eight": "20261005-205009-warmup_multi_demo_settings",
}
SWEEP_RUNS = {
    "sweep 1: ridge, leak rate, input scaling": "20261005-210423-sweep_single_demo",
    "sweep 2: reservoir size, spectral radius, warm-up": "20261005-210421-sweep_single_demo_reservoir",
}
# The robot: the ESNs on the robot (key: configuration suffix), the scenarios (label: configuration prefix),
# the two featured tracker settings (law, omega, suffix of the runs that have it), and the gains of joint PD.
ROBOT_ESNS = {"tuned": "tuned", "eight": "multi_demo_settings"}
SCENARIOS = {
    "nominal": "nominal",
    "offsets": "offsets",
    "push across": "push_across",
    "push forward": "push_forward",
    "push backward": "push_backward",
    "block": "block",
}
FEATURED = [("computed_torque", 10.0, ""), ("pd", 20.0, "_pd_gains")]
PD_OMEGAS = [10.0, 20.0, 40.0]
LAW_NAMES = {"computed_torque": "computed torque", "pd": "joint PD"}
ARMS = ("tuned", "eight", "replay", "demonstrator")
ARM_LABELS = {
    "tuned": "ESN, tuned settings",
    "eight": "ESN, eight-demonstration settings",
    "replay": "replay",
    "demonstrator": "demonstrator",
}
ARM_COLORS = {"tuned": "#2a78d6", "eight": "#4a3aa7", "replay": "#eb6834", "demonstrator": "#52514e"}
ARM_MARKERS = {"tuned": "o", "eight": "^", "replay": "s", "demonstrator": "D"}
OUTCOME_COLORS = ["#2a78d6", "#eb6834", "#52514e"]  # arrive and hold, leave the goal, never arrive
OUTCOMES = ["arrive and hold", "leave the goal", "never arrive"]
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
DISTURBANCE_COLOR = "#f0efec"
MAX_RATIO_SHOWN = 1.5  # the training path ratio's color scale ends here
# The analyses of the logs: the windows of the departure from the demonstration (s), as in report 002.
DEPARTURE_WINDOWS = {
    "nominal": (0.0, 1.5),
    "push across": (0.4, 1.4),
    "push forward": (0.4, 1.4),
    "push backward": (0.4, 1.4),
    "block": (0.3, 1.3),
}
OFFSET_WINDOW = (0.0, 1.5)  # the reach from an offset start (s)
PROGRESS_TIMES = (0.5, 0.6)  # the end of the pushes, and 0.1 s later (s)
DEPARTURES = SUMMARY / "departures.csv"
OFFSET_REFERENCES = SUMMARY / "offset_references.csv"
PUSH_PROGRESS = SUMMARY / "push_progress.csv"
# When each disturbance acts (task time, s).
SPANS = {"push across": (0.4, 0.5), "push forward": (0.4, 0.5), "push backward": (0.4, 0.5), "block": (0.3, 0.8)}
# The examples, scenario by scenario: name, scenario, and start offset (deg) from the demonstrated start.
EXAMPLES = [
    ("nominal", "nominal", (0.0, 0.0)),
    ("offset ahead", "offsets", (10.0, -10.0)),  # roughly along the demonstrated motion
    ("offset across", "offsets", (-10.0, -10.0)),  # mostly across it
    ("push across", "push across", (0.0, 0.0)),
    ("push forward", "push forward", (0.0, 0.0)),
    ("push backward", "push backward", (0.0, 0.0)),
    ("block", "block", (0.0, 0.0)),
]
EXAMPLE_END = 2.5  # the joint angles are drawn until this task time (s)
# The underdamped trackers (Section 3.9): the damping ratios of the runs, the one of the examples' figures,
# how long those figures run (s), and the window after every reach in which the ringing is measured (s).
DAMPINGS = [1.0, 0.5, 0.3, 0.1]
UNDERDAMPED = 0.1
UNDERDAMPED_END = 5.0
RINGING_WINDOW = (2.5, 5.0)
RINGING = SUMMARY / "ringing.csv"
OVERSHOOT = SUMMARY / "overshoot.csv"
FLOOR_MM = 0.1  # final distances below this are drawn at it
LAW_STYLES = {"computed_torque": "-", "pd": "--"}
# The animations of the examples: one tracker setting, the task times they cover (s), and their frame rate.
ANIMATION_SETTING = ("pd", 20.0)
ANIMATION_SPAN = (-0.2, 3.0)
ANIMATION_FPS = 20.0  # a whole number of milliseconds per frame, so GIFs play in real time
ANIMATION_HOLD_MS = 1500  # the last frame stays this long before the GIF loops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", action="store_true", help="also analyze the run logs in the storage root")
    parser.add_argument("--animations", action="store_true", help="also export GIFs from the logs in the storage root")
    args = parser.parse_args()

    SUMMARY.mkdir(parents=True, exist_ok=True)
    figures = {
        "grids.png": plot_grids(),
        "warmup.png": plot_warmup(),
        "sweeps.png": plot_sweeps(),
        "robot.png": plot_robot(),
        "offsets.png": plot_offsets(),
        "gains.png": plot_gains(),
        "damping.png": plot_damping(),
    }
    if args.logs:
        figures["block.png"] = plot_block()
        for name, scenario, offset in EXAMPLES:
            figures[f"example_{slug(name)}.png"] = plot_example(name, scenario, offset)
            figures[f"underdamped_{slug(name)}.png"] = plot_example(name, scenario, offset, UNDERDAMPED)
        write_ringing()
        write_overshoot()
        write_departures()
        write_offset_references()
        write_push_progress()
    for name, fig in figures.items():
        fig.savefig(SUMMARY / name, dpi=150)
    print_tables()
    if args.animations:
        for name, scenario, offset in EXAMPLES:
            export_example_animation(name, scenario, offset)
    print(f"\nWrote {', '.join(figures)} to {SUMMARY}")


# ---------------------------------------------------------------------------- reading the runs


def value(text: str) -> Any:
    """A CSV field as a bool, a float, or the text itself."""
    if text in ("True", "False"):
        return text == "True"
    try:
        return float(text)
    except ValueError:
        return text


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return [{key: value(text) for key, text in row.items()} for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def trained_warmup(key: str) -> float:
    """The warm-up (s) an ESN was trained with, from its esn.toml."""
    with (RESULTS / ESNS[key][1] / "esn.toml").open("rb") as f:
        return float(tomllib.load(f)["esn"]["warmup"])


def offset_of(origin: str) -> tuple[float, float]:
    """The start offset (deg) in a start's origin, such as "demo_00 +2.5,-5 deg"; (0, 0) for "demo_00"."""
    match = re.search(r"([+-][\d.]+),([+-][\d.]+) deg", origin)
    return (0.0, 0.0) if match is None else (float(match[1]), float(match[2]))


def robot_run(scenario: str, esn: str, suffix: str) -> str:
    """The name of the robot run of a scenario with one of the ESNs, and the suffix of its tracker settings."""
    matches = sorted(p.name for p in RESULTS.glob(f"*-{SCENARIOS[scenario]}_{ROBOT_ESNS[esn]}{suffix}"))
    if len(matches) != 1:
        msg = f"expected one run of {scenario} with {esn}{suffix}, found {matches}"
        raise FileNotFoundError(msg)
    return matches[0]


def robot_rows(scenario: str, arm: str, law: str, omega: float) -> list[dict[str, Any]]:
    """The metrics rows of one arm in a robot scenario at one tracker setting (any setting for the demonstrator).

    The replay and the demonstrator are the same in the runs of both ESNs; they are
    read from the tuned ESN's runs.
    """
    suffix = "" if law == "computed_torque" and omega == 10.0 else "_pd_gains"
    esn = arm if arm in ROBOT_ESNS else "tuned"
    rows = read_csv(RESULTS / robot_run(scenario, esn, suffix) / "metrics.csv")
    if arm == "demonstrator":
        return [r for r in rows if r["arm"] == "demonstrator"]
    name = "esn" if arm in ROBOT_ESNS else arm
    return [r for r in rows if r["arm"] == name and r["law"] == law and r["omega"] == omega]


def damping_rows(scenario: str, arm: str, law: str, damping: float) -> list[dict[str, Any]]:
    """The metrics rows of one arm in the underdamped runs of a scenario, at one law and damping ratio.

    Each law runs at its featured natural frequency. The replay and the demonstrator
    are read from the tuned ESN's runs.
    """
    rows = read_csv(RESULTS / robot_run(scenario, arm if arm in ROBOT_ESNS else "tuned", "_damping") / "metrics.csv")
    if arm == "demonstrator":
        return [r for r in rows if r["arm"] == "demonstrator"]
    name = "esn" if arm in ROBOT_ESNS else arm
    return [r for r in rows if r["arm"] == name and r["law"] == law and r["damping"] == damping]


def mean(rows: list[dict[str, Any]], key: str) -> float:
    """The mean of a metric over rows, ignoring NaN (NaN if all are); a share in percent for "success"."""
    values = np.array([float(r[key]) for r in rows])
    if key == "success":
        return 100.0 * float(values.mean())
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def draw_map(
    ax: Axes, offsets: list[tuple[float, float]], values: list[float], norm: Normalize, cmap: Any = "Blues"
) -> None:
    """Draw values over the grid of start offsets (deg), the demonstrated start marked with a cross."""
    xs, ys = sorted({o[0] for o in offsets}), sorted({o[1] for o in offsets})
    cell = dict(zip(offsets, values, strict=True))
    data = np.array([[cell.get((x, y), np.nan) for x in xs] for y in ys])
    step_x, step_y = xs[1] - xs[0], ys[1] - ys[0]
    extent = (xs[0] - step_x / 2, xs[-1] + step_x / 2, ys[0] - step_y / 2, ys[-1] + step_y / 2)
    ax.imshow(data, origin="lower", extent=extent, cmap=cmap, norm=norm)
    ax.plot(0.0, 0.0, marker="+", markersize=10, color="#0b0b0b", markeredgewidth=1.5)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR, labelsize=8)


def log_norm(values: list[list[float]]) -> LogNorm:
    """A logarithmic color scale over all the positive finite values."""
    flat = np.array([v for group in values for v in group])
    flat = flat[np.isfinite(flat) & (flat > 0)]
    return LogNorm(vmin=float(flat.min()), vmax=float(flat.max()) * 1.0001)


# ---------------------------------------------------------------------------- autonomous runs


def plot_grids() -> Figure:
    """The autonomous runs of the three ESNs over the grid of start offsets: one row per ESN."""
    columns = [
        ("reach_path_distance_m", "Path distance from the demonstrator's\nreach from each start (mm)", 1000.0, "log"),
        (
            "training_path_ratio",
            "Training path ratio: 0 returns onto the\ndemonstration, 1 as the demonstrator",
            1.0,
            "ratio",
        ),
        ("first_step_m", "First step of the hand (mm)", 1000.0, "log"),
    ]
    runs = {key: read_csv(RESULTS / run / "metrics.csv") for key, (_, run) in ESNS.items()}
    fig = Figure(figsize=(12.5, 11.5), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle("The ESN trained on one demonstration, run on its own from the grid of start offsets", color="#0b0b0b")
    axes = fig.subplots(len(ESNS), len(columns), squeeze=False)
    for column, (key, title, factor, scale) in enumerate(columns):
        values = {esn: [float(r[key]) * factor for r in rows] for esn, rows in runs.items()}
        norm = Normalize(0.0, MAX_RATIO_SHOWN) if scale == "ratio" else log_norm(list(values.values()))
        for row, (esn, rows) in enumerate(runs.items()):
            ax = axes[row, column]
            draw_map(ax, [offset_of(r["origin"]) for r in rows], values[esn], norm)
            ax.set_title(title, color=TEXT_COLOR, fontsize=9)
            if column == 0:
                held = sum(r["success"] for r in rows)
                ax.set_ylabel(
                    f"{ESNS[esn][0]}\n({held} of {len(rows)} arrive and hold)\njoint 2 offset (deg)",
                    color=TEXT_COLOR,
                    fontsize=9,
                )
            if row == len(ESNS) - 1:
                ax.set_xlabel("joint 1 offset (deg)", color=TEXT_COLOR, fontsize=9)
        extend = "max" if scale == "ratio" else "neither"
        fig.colorbar(
            ScalarMappable(norm=norm, cmap="Blues"), ax=axes[:, column].tolist(), location="bottom", extend=extend
        )
    return fig


def plot_warmup() -> Figure:
    """The arrival delay and the failed runs against the warm-up, for the two ESNs of Phase 1."""
    fig = Figure(figsize=(12, 7.5), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle("Runs with other warm-ups than the trained one (dotted)", color="#0b0b0b")
    axes = fig.subplots(2, len(WARMUP_RUNS), squeeze=False, height_ratios=[2, 1])
    for column, (key, run) in enumerate(WARMUP_RUNS.items()):
        rows = read_csv(RESULTS / run / "warmup.csv")
        trained = trained_warmup(key)
        durations = sorted({r["warmup_s"] for r in rows})
        x = np.array(durations)
        offset = [
            [r["arrival_delay_s"] for r in rows if r["warmup_s"] == w and r["group"] != "demonstrated"]
            for w in durations
        ]
        offset = [[d for d in group if np.isfinite(d)] for group in offset]
        median = np.array([np.median(d) for d in offset])
        demonstrated = [
            next(r["arrival_delay_s"] for r in rows if r["warmup_s"] == w and r["group"] == "demonstrated")
            for w in durations
        ]
        ax = axes[0, column]
        style(ax)
        ax.fill_between(
            x,
            [np.percentile(d, 25) for d in offset],
            [np.percentile(d, 75) for d in offset],
            color=ARM_COLORS["tuned"],
            alpha=0.2,
            linewidth=0,
            label="offset starts, middle half",
        )
        ax.plot(x, median, color=ARM_COLORS["tuned"], marker="o", label="offset starts, median")
        ax.plot(x, demonstrated, color="#0b0b0b", marker="s", linestyle="none", label="demonstrated start")
        at_trained = float(median[durations.index(trained)])
        shown = np.concatenate([median, demonstrated, [np.percentile(d, q) for d in offset for q in (25, 75)]])
        margin = 0.1 * max(float(np.ptp(shown)), 0.1)
        ax.plot(x, at_trained - (x - trained), color=TEXT_COLOR, linestyle="--", label="a clock from the reset")
        ax.set_ylim(float(shown.min()) - margin, float(shown.max()) + margin)
        ax.axvline(trained, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
        ax.set_title(f"ESN, {ESNS[key][0]}: arrival delay", color=TEXT_COLOR, fontsize=10)
        ax.set_ylabel("after the demonstrator's arrival (s)", color=TEXT_COLOR)
        ax.legend(fontsize=8)
        ax = axes[1, column]
        style(ax)
        n_runs = sum(r["warmup_s"] == trained for r in rows)
        failed = [sum(not r["success"] for r in rows if r["warmup_s"] == w) for w in durations]
        ax.plot(x, failed, color=ARM_COLORS["tuned"], marker="o")
        ax.axvline(trained, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
        ax.set(ylim=(-0.05 * n_runs, 1.05 * n_runs), xlabel="warm-up (s)")
        ax.set_ylabel(f"failed runs (of {n_runs})", color=TEXT_COLOR)
    return fig


def plot_sweeps() -> Figure:
    """The two sweeps, condensed: failures by input scaling and by ridge, and the course of the runs that hold."""
    sweep1, sweep2 = (read_csv(RESULTS / run / "sweep.csv") for run in SWEEP_RUNS.values())
    fig = Figure(figsize=(16, 5.2), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "Hyperparameter sweeps of the ESN trained on one demonstration, over the grid of start offsets", color="#0b0b0b"
    )
    axes = fig.subplots(1, 3, width_ratios=[1, 1, 1.3])
    rng = np.random.default_rng(0)  # only spreads the points of each column sideways
    for ax, key in zip(axes[:2], ("input_scaling", "ridge"), strict=True):
        style(ax)
        values = sorted({r[key] for r in sweep1})
        for i, v in enumerate(values):
            failures = [r["failures"] for r in sweep1 if r[key] == v]
            ax.scatter(
                i + rng.uniform(-0.25, 0.25, len(failures)), failures, s=14, color=ARM_COLORS["tuned"], alpha=0.6
            )
            ax.plot([i - 0.3, i + 0.3], [np.median(failures)] * 2, color="#0b0b0b", linewidth=2)
        ax.set_xticks(range(len(values)), [f"{v:g}" for v in values])
        ax.set(xlabel=key.replace("_", " "), ylim=(-8, 177))
        ax.set_ylabel(f"failed runs (of {int(sweep1[0]['runs'])})", color=TEXT_COLOR)
        title = "input scaling\n(the scale of the normalization)" if key == "input_scaling" else "ridge"
        ax.set_title(f"Sweep 1: failed runs, by {title}", color=TEXT_COLOR, fontsize=10)
    ax = axes[2]
    style(ax)
    markers = {"sweep 1": ("o", ARM_COLORS["tuned"]), "sweep 2": ("^", "#1baf7a")}
    for (label, rows), (marker, color) in zip(
        zip(SWEEP_RUNS, (sweep1, sweep2), strict=True), markers.values(), strict=True
    ):
        held = [r for r in rows if r["failures"] == 0]
        ax.scatter(
            [r["other_median_training_path_ratio"] for r in held],
            [1000 * r["other_reach_path_distance_m"] for r in held],
            marker=marker,
            s=22,
            color=color,
            alpha=0.6,
            label=f"{label} ({len(held)} of {len(rows)} hold everywhere)",
        )
    for name, run in ESNS.values():
        rows = [r for r in read_csv(RESULTS / run / "metrics.csv") if r["origin"] != "demo_00"]
        ratio = float(np.median([r["training_path_ratio"] for r in rows if np.isfinite(r["training_path_ratio"])]))
        distance = 1000 * mean(rows, "reach_path_distance_m")
        ax.scatter([ratio], [distance], marker="*", s=180, color="#0b0b0b", zorder=3)
        ax.annotate(name, (ratio, distance), textcoords="offset points", xytext=(8, 4), fontsize=8, color="#0b0b0b")
    ax.axvline(1.0, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
    ax.set(xscale="log", yscale="log", xlabel="median training path ratio of the offset starts")
    ax.set_ylabel("mean path distance from the demonstrator (mm)", color=TEXT_COLOR)
    ax.set_title(
        "Combinations that arrive and hold from every start\n"
        "(stars: the three ESNs of the report; dotted: as the demonstrator)",
        color=TEXT_COLOR,
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper left")
    return fig


# ---------------------------------------------------------------------------- the robot


def plot_robot() -> Figure:
    """Means over the start postures of every arm in every scenario, at the two featured tracker settings.

    Every arm arrives and holds from every start in every scenario at both settings,
    so the figure shows when they arrive instead.
    """
    metrics = [
        ("reach_path_distance_m", "Path distance from the demonstrator's\nundisturbed reach (mm)", 1000.0, True),
        ("arrival_time_s", "Arrival time (s)", 1.0, False),
        ("peak_torque_nm", "Peak joint torque (N m)", 1.0, True),
        ("effort_n2m2s", "Integral of squared torque (N² m² s)", 1.0, True),
    ]
    fig = Figure(figsize=(17, 8.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle("The ESNs trained on one demonstration as the robot's reference generator", color="#0b0b0b")
    axes = fig.subplots(len(FEATURED), len(metrics), squeeze=False)
    labels = list(SCENARIOS)
    for row, (law, omega, _) in enumerate(FEATURED):
        for column, (key, title, factor, log) in enumerate(metrics):
            ax = axes[row, column]
            style(ax)
            for i, arm in enumerate(ARMS):
                x = np.arange(len(labels)) + (i - 1.5) * 0.15
                y = [mean(robot_rows(s, arm, law, omega), key) * factor for s in labels]
                if key == "reach_path_distance_m" and arm == "demonstrator":
                    y = [np.nan if v == 0.0 else v for v in y]  # its own reference: off the log scale
                ax.scatter(x, y, marker=ARM_MARKERS[arm], s=40, color=ARM_COLORS[arm], label=ARM_LABELS[arm], zorder=3)
            if log:
                ax.set_yscale("log")
            ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
            ax.set_title(f"{LAW_NAMES[law]}, ω = {omega:g} rad/s\n{title}", color=TEXT_COLOR, fontsize=9)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=4, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_offsets() -> Figure:
    """The robot runs over the grid of start offsets: one row per tracked arm."""
    columns = [
        (
            "reach_path_distance_m",
            "computed_torque",
            10.0,
            "Path distance from the demonstrator's\nreach (mm), computed torque, ω = 10",
            1000.0,
        ),
        ("peak_torque_nm", "computed_torque", 10.0, "Peak joint torque (N m),\ncomputed torque, ω = 10", 1.0),
        ("outcome", "pd", 10.0, "Outcome, joint PD, ω = 10\n(all arrive and hold at ω = 20 and 40)", 1.0),
    ]
    arms = ("tuned", "eight", "replay")
    fig = Figure(figsize=(12.5, 11.5), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle("The robot from the grid of start offsets", color="#0b0b0b")
    axes = fig.subplots(len(arms), len(columns), squeeze=False)
    for column, (key, law, omega, title, factor) in enumerate(columns):
        rows = {arm: robot_rows("offsets", arm, law, omega) for arm in arms}
        if key == "outcome":
            values = {
                arm: [0.0 if r["success"] else 1.0 if r["arrived"] else 2.0 for r in group]
                for arm, group in rows.items()
            }
            norm: Normalize = Normalize(-0.5, 2.5)
            cmap: Any = ListedColormap(OUTCOME_COLORS)
        else:
            values = {arm: [float(r[key]) * factor for r in group] for arm, group in rows.items()}
            norm, cmap = log_norm(list(values.values())), "Blues"
        for row, arm in enumerate(arms):
            ax = axes[row, column]
            draw_map(ax, [offset_of(r["origin"]) for r in rows[arm]], values[arm], norm, cmap)
            ax.set_title(title, color=TEXT_COLOR, fontsize=9)
            if column == 0:
                ax.set_ylabel(f"{ARM_LABELS[arm]}\njoint 2 offset (deg)", color=TEXT_COLOR, fontsize=9)
            if row == len(arms) - 1:
                ax.set_xlabel("joint 1 offset (deg)", color=TEXT_COLOR, fontsize=9)
        if key == "outcome":
            handles = [Patch(color=color, label=text) for color, text in zip(OUTCOME_COLORS, OUTCOMES, strict=True)]
            axes[-1, column].legend(
                handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8, ncol=1, frameon=False
            )
        else:
            fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes[:, column].tolist(), location="bottom")
    return fig


def plot_gains() -> Figure:
    """Against the gain of joint PD: holds from the offsets, the block's force and effort, and the pushes' shifts."""
    fig = Figure(figsize=(17, 4.8), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle("The robot against the gain of joint PD", color="#0b0b0b")
    axes = fig.subplots(1, 4)
    for ax in axes:
        style(ax)
        ax.set_xscale("log")
        ax.set_xticks(PD_OMEGAS, [f"{w:g}" for w in PD_OMEGAS])
        ax.minorticks_off()
        ax.set_xlabel("natural frequency ω of joint PD (rad/s)", color=TEXT_COLOR)
    tracked = ("tuned", "eight", "replay")
    for arm in tracked:
        kwargs = {"color": ARM_COLORS[arm], "marker": ARM_MARKERS[arm], "label": ARM_LABELS[arm]}
        axes[0].plot(PD_OMEGAS, [mean(robot_rows("offsets", arm, "pd", w), "success") for w in PD_OMEGAS], **kwargs)
        axes[1].plot(
            PD_OMEGAS, [mean(robot_rows("block", arm, "pd", w), "peak_external_force_n") for w in PD_OMEGAS], **kwargs
        )
        axes[2].plot(PD_OMEGAS, [mean(robot_rows("block", arm, "pd", w), "effort_n2m2s") for w in PD_OMEGAS], **kwargs)
        for direction, line in (("forward", "-"), ("backward", "--")):
            shift = [
                mean(robot_rows(f"push {direction}", arm, "pd", w), "arrival_time_s")
                - mean(robot_rows("nominal", arm, "pd", w), "arrival_time_s")
                for w in PD_OMEGAS
            ]
            axes[3].plot(PD_OMEGAS, shift, color=ARM_COLORS[arm], marker=ARM_MARKERS[arm], linestyle=line)
    for ax, key in ((axes[1], "peak_external_force_n"), (axes[2], "effort_n2m2s")):
        level = mean(robot_rows("block", "demonstrator", "", 0.0), key)
        ax.axhline(
            level, color=ARM_COLORS["demonstrator"], linestyle="--", linewidth=1.2, label=ARM_LABELS["demonstrator"]
        )
        ax.set_yscale("log")
    axes[0].set(ylim=(-5, 105))
    axes[0].set_title("Offsets: arrive and hold (% of 169 starts)", color=TEXT_COLOR, fontsize=10)
    axes[1].set_title("Block: peak holding force (N)", color=TEXT_COLOR, fontsize=10)
    axes[2].set_title("Block: integral of squared torque\n(N² m² s, 0.3 to 1.3 s)", color=TEXT_COLOR, fontsize=10)
    axes[3].axhline(0.0, color=TEXT_COLOR, linewidth=0.8)
    axes[3].set_title(
        "Pushes along the reach: arrival shift (s)\n(solid: forward; dashed: backward)", color=TEXT_COLOR, fontsize=10
    )
    handles, names = axes[1].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=4, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_damping() -> Figure:
    """The underdamped runs against the damping ratio (falling to the right): one column per scenario.

    Rows: the share of the runs that arrive and hold, their settling time (of those
    that settle), and the final distance to the target.
    """
    measures = [
        ("success", "Arrive and hold (% of runs)", 1.0, False),
        ("settling_time_s", "Settling time (s)", 1.0, False),
        (
            "final_distance_m",
            f"Final distance to the target (mm;\n{FLOOR_MM:g} mm and below drawn at {FLOOR_MM:g})",
            1000.0,
            True,
        ),
    ]
    fig = Figure(figsize=(18, 9.5), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "Underdamped trackers: computed torque at ω = 10 rad/s (solid), joint PD at ω = 20 rad/s (dashed)",
        color="#0b0b0b",
    )
    axes = fig.subplots(len(measures), len(SCENARIOS), squeeze=False)
    for row, (key, title, factor, log) in enumerate(measures):
        for column, scenario in enumerate(SCENARIOS):
            ax = axes[row, column]
            style(ax)
            for arm in ("tuned", "eight", "replay"):
                for law, _, _ in FEATURED:
                    values = [mean(damping_rows(scenario, arm, law, z), key) * factor for z in DAMPINGS]
                    if log:
                        values = [max(v, FLOOR_MM) for v in values]
                    label = f"{ARM_LABELS[arm]}, {LAW_NAMES[law]}"
                    kwargs = {"color": ARM_COLORS[arm], "marker": ARM_MARKERS[arm], "markersize": 6}
                    ax.plot(DAMPINGS, values, linestyle=LAW_STYLES[law], linewidth=1.6, label=label, **kwargs)
            ax.set_xscale("log")
            ax.set_xticks(DAMPINGS, [f"{z:g}" for z in DAMPINGS])
            ax.minorticks_off()
            ax.set_xlim(1.25, 0.08)  # less damping to the right
            if log:
                ax.set_yscale("log")
                ax.set_ylim(FLOOR_MM * 0.7, None)
            if key == "success":
                ax.set_ylim(-5, 105)
            if row == 0:
                n = len(damping_rows(scenario, "replay", "pd", 1.0))
                ax.set_title(f"{scenario} ({n} start{'s' if n > 1 else ''})", color="#0b0b0b", fontsize=10)
            if column == 0:
                ax.set_ylabel(title, color=TEXT_COLOR)
            if row == len(measures) - 1:
                ax.set_xlabel("damping ratio ζ", color=TEXT_COLOR, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    return fig


# ---------------------------------------------------------------------------- the run logs


def demonstrated_path() -> NDArray[np.float64]:
    """The demonstration's joint angles at the ESN's period, as the robot runs measured progress on them."""
    from arm_esn_ctrl.demonstrations import load_joint_angles

    return load_joint_angles(DEMONSTRATION, 0.01)[1]


def robot_log_path(scenario: str, esn: str, suffix: str, setting: str | None, arm: str, start: int = 0) -> Path:
    """The path of a run log under the storage root: ``setting`` is None for the demonstrator."""
    from arm_esn_ctrl.storage import resolve_run_path

    run = resolve_run_path(f"results/{robot_run(scenario, esn, suffix)}")
    name = f"{arm}_{start:02d}.sklog.npz"
    return run / name if setting is None else run / setting / name


def robot_log(scenario: str, esn: str, suffix: str, setting: str | None, arm: str, start: int = 0) -> StateLog:
    """A run log from the storage root: ``setting`` is None for the demonstrator."""
    return StateLog.load(robot_log_path(scenario, esn, suffix, setting, arm, start))


def setting_name(law: str, omega: float) -> tuple[str, str]:
    """The directory of a tracker setting in a run, and the suffix of the runs that have it."""
    suffix = "" if law == "computed_torque" and omega == 10.0 else "_pd_gains"
    return f"{law}_w{omega:g}", suffix


def tracked_logs(scenario: str, law: str, omega: float) -> dict[str, StateLog]:
    """The logs of the three tracked arms and the demonstrator, from the demonstrated start."""
    setting, suffix = setting_name(law, omega)
    return {
        "tuned": robot_log(scenario, "tuned", suffix, setting, "esn"),
        "eight": robot_log(scenario, "eight", suffix, setting, "esn"),
        "replay": robot_log(scenario, "tuned", suffix, setting, "replay"),
        "demonstrator": robot_log(scenario, "tuned", suffix, None, "demonstrator"),
    }


def joint_angles_at(log: StateLog, times: NDArray[np.float64], channel: str = "q") -> NDArray[np.float64]:
    """A log's joint angles (or reference ``q_ref``) interpolated at ``times``, in degrees."""
    values = log.channel(channel).reshape(len(log.times), -1)
    return np.degrees(np.column_stack([np.interp(times, log.times, values[:, j]) for j in range(values.shape[1])]))


def plot_block() -> Figure:
    """The block over time at the featured settings: the hand's approach, the progress, and the holding force."""
    from arm_esn_ctrl.demonstrations import endpoint_positions
    from arm_esn_ctrl.metrics import path_progress

    path = demonstrated_path()
    target = np.array([0.0, 1.2])
    fig = Figure(figsize=(13, 10), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle("Block from 0.3 s to 0.8 s, from the demonstrated start", color="#0b0b0b")
    axes = fig.subplots(3, len(FEATURED), sharex=True, squeeze=False)
    for column, (law, omega, _) in enumerate(FEATURED):
        logs = tracked_logs("block", law, omega)
        undisturbed = robot_log("nominal", "tuned", "", None, "demonstrator")
        skeleton = undisturbed.build_skeleton()
        ax_distance, ax_progress, ax_force = axes[:, column]
        for ax in axes[:, column]:
            style(ax)
            ax.axvspan(*SPANS["block"], color=DISTURBANCE_COLOR, zorder=0)
        q = undisturbed.channel("q").reshape(len(undisturbed.times), -1)
        hand = endpoint_positions(skeleton, q)
        ax_distance.plot(
            undisturbed.times,
            1000 * np.linalg.norm(hand - target, axis=1),
            color=GRID_COLOR,
            linewidth=5,
            label="demonstrator, undisturbed",
        )
        ax_progress.plot(undisturbed.times, path_progress(q, path), color=GRID_COLOR, linewidth=5)
        for arm, log in logs.items():
            q = log.channel("q").reshape(len(log.times), -1)
            color = ARM_COLORS[arm]
            ax_distance.plot(
                log.times,
                1000 * np.linalg.norm(endpoint_positions(skeleton, q) - target, axis=1),
                color=color,
                linewidth=1.5,
                label=ARM_LABELS[arm],
            )
            ax_progress.plot(log.times, path_progress(q, path), color=color, linewidth=1.5)
            if "q_ref" in log.channel_names:
                ax_progress.plot(
                    log.times, path_progress(log.channel("q_ref"), path), color=color, linewidth=1.5, linestyle="--"
                )
            ax_force.plot(log.times, np.linalg.norm(log.channel("ext_force"), axis=1), color=color, linewidth=1.5)
        ax_distance.axhline(20.0, color="#0b0b0b", linewidth=0.8, linestyle=":")
        ax_distance.set_title(f"{LAW_NAMES[law]}, ω = {omega:g} rad/s", color=TEXT_COLOR, fontsize=11)
        ax_distance.set(xlim=(-0.2, 3.0), ylim=(0, None))
        ax_progress.set_ylim(-0.05, 1.05)
        ax_force.set_yscale(
            "symlog", linthresh=1.0
        )  # linear below 1 N, so that 0 shows: the force is 0 outside the block
        ax_force.set_ylim(0.0, None)
        ax_force.set_xlabel("time (s)", color=TEXT_COLOR)
        if column == 0:
            ax_distance.set_ylabel("hand to target (mm)", color=TEXT_COLOR)
            ax_progress.set_ylabel(
                "progress along the demonstrated path\n(solid: arm; dashed: its reference)", color=TEXT_COLOR
            )
            ax_force.set_ylabel("holding force at the tip (N)", color=TEXT_COLOR)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=5, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def write_departures() -> None:
    """How far the ESNs' output departs from the demonstration, against how far the arm does, as in report 002.

    The largest distance in joint space of the output and of the arm from the
    demonstrator's undisturbed reach, over each scenario's window; their ratio is 0
    if the output replays the demonstration, 1 if it departs as far as the arm.
    """
    undisturbed = robot_log("nominal", "tuned", "", None, "demonstrator")
    rows = []
    for scenario, (begin, end) in DEPARTURE_WINDOWS.items():
        for law, omega, _ in FEATURED:
            logs = tracked_logs(scenario, law, omega)
            for esn in ROBOT_ESNS:
                log = logs[esn]
                times = log.times[(log.times >= begin - 1e-9) & (log.times <= end + 1e-9)]
                demonstration = joint_angles_at(undisturbed, times)
                output = float(np.linalg.norm(joint_angles_at(log, times, "q_ref") - demonstration, axis=1).max())
                arm = float(np.linalg.norm(joint_angles_at(log, times) - demonstration, axis=1).max())
                rows.append(
                    {"scenario": scenario, "law": law, "omega": omega, "esn": esn}
                    | {"output_departure_deg": output, "arm_departure_deg": arm, "ratio": output / arm}
                )
    write_csv(DEPARTURES, rows)


def write_offset_references() -> None:
    """From each offset start, how far the ESNs' output is from the demonstrator's reach and from the replay.

    The RMS distance in joint space over the reach of the output from the
    demonstrator's reach from that start, and from the replayed demonstration's
    reference, as in report 002.
    """
    from arm_esn_ctrl.storage import resolve_run_path

    rows = []
    for law, omega, _ in FEATURED:
        setting, suffix = setting_name(law, omega)
        for esn in ROBOT_ESNS:
            run = resolve_run_path(f"results/{robot_run('offsets', esn, suffix)}")
            metrics = [r for r in read_csv(run / "metrics.csv") if r["arm"] == "demonstrator"]
            for r in metrics:
                start = int(r["start"])
                log = StateLog.load(run / setting / f"esn_{start:02d}.sklog.npz")
                replay = StateLog.load(run / setting / f"replay_{start:02d}.sklog.npz")
                demonstrator = StateLog.load(run / f"demonstrator_{start:02d}.sklog.npz")
                times = log.times[(log.times >= OFFSET_WINDOW[0] - 1e-9) & (log.times <= OFFSET_WINDOW[1] + 1e-9)]
                output = joint_angles_at(log, times, "q_ref")
                from_start = output - joint_angles_at(demonstrator, times)
                from_replayed = output - joint_angles_at(replay, times, "q_ref")
                q1, q2 = offset_of(r["origin"])
                rows.append(
                    {"law": law, "omega": omega, "esn": esn, "start": start, "offset_q1_deg": q1, "offset_q2_deg": q2}
                    | {"from_this_start_deg": float(np.sqrt(np.mean(np.sum(from_start**2, axis=1))))}
                    | {"from_replayed_deg": float(np.sqrt(np.mean(np.sum(from_replayed**2, axis=1))))}
                )
    write_csv(OFFSET_REFERENCES, rows)


def write_push_progress() -> None:
    """How far each push along the reach moves the arm along the demonstrated path, against the undisturbed run.

    The difference in progress (a fraction of the path) between the pushed and the
    undisturbed run of the same arm and tracker setting, at the end of the push and
    0.1 s later.
    """
    from arm_esn_ctrl.metrics import path_progress

    path = demonstrated_path()
    rows = []
    settings = [("computed_torque", 10.0)] + [("pd", w) for w in PD_OMEGAS]
    for law, omega in settings:
        nominal = tracked_logs("nominal", law, omega)
        for direction in ("forward", "backward"):
            pushed = tracked_logs(f"push {direction}", law, omega)
            for arm in ("tuned", "eight", "replay"):
                row = {"direction": direction, "law": law, "omega": omega, "arm": arm}
                for t in PROGRESS_TIMES:
                    progress = []
                    for log in (pushed[arm], nominal[arm]):
                        k = int(np.argmin(np.abs(log.times - t)))
                        progress.append(float(path_progress(log.channel("q")[k : k + 1], path)[0]))
                    row[f"moved_at_{t:g}s"] = progress[0] - progress[1]
                rows.append(row)
    write_csv(PUSH_PROGRESS, rows)


def write_ringing() -> None:
    """How each ESN's output rings with its arm after the reach, with the trackers at the damping ratio 0.1.

    For every example and featured law, over a window after every reach: the
    standard deviation of the output against the arm's, joint by joint (1: the
    output rings as much as the arm; 0: it holds still), and the lag of the output
    behind the arm, from their cross-correlation. Joints that ring less than 0.2°
    (standard deviation) are left out, as report 002 did.
    """
    rows = []
    begin, end = RINGING_WINDOW
    for name, scenario, offset in EXAMPLES:
        for law, omega, _ in FEATURED:
            paths = example_paths(scenario, offset, law, omega, UNDERDAMPED)
            for esn in ROBOT_ESNS:
                log = StateLog.load(paths[esn])
                times = log.times[(log.times >= begin - 1e-9) & (log.times <= end + 1e-9)]
                arm, output = joint_angles_at(log, times), joint_angles_at(log, times, "q_ref")
                for joint in range(arm.shape[1]):
                    a, b = arm[:, joint] - arm[:, joint].mean(), output[:, joint] - output[:, joint].mean()
                    if a.std() < 0.2:  # this joint has stopped ringing (deg)
                        continue
                    lag = (np.argmax(np.correlate(b, a, "full")) - (len(a) - 1)) * float(np.diff(times).mean())
                    rows.append(
                        {"example": name, "law": law, "omega": omega, "damping": UNDERDAMPED, "esn": esn}
                        | {"joint": joint + 1, "arm_std_deg": float(a.std()), "lag_s": lag}
                        | {"amplitude_ratio": float(b.std() / a.std())}
                    )
    write_csv(RINGING, rows)


def write_overshoot() -> None:
    """How far each tracked arm passes the end posture, critically damped and with the damping ratio 0.1.

    For each joint, the largest excursion beyond its end posture in the direction it
    moves in the demonstration (0 if it never passes it), for every example and
    featured law.
    """
    path = demonstrated_path()
    end, direction = np.degrees(path[-1]), np.sign(path[-1] - path[0])
    rows = []
    for name, scenario, offset in EXAMPLES:
        for law, omega, _ in FEATURED:
            for damping in (1.0, UNDERDAMPED):
                paths = example_paths(scenario, offset, law, omega, damping)
                for arm in ("tuned", "eight", "replay"):
                    log = StateLog.load(paths[arm])
                    q = np.degrees(log.channel("q").reshape(len(log.times), -1))[log.times > -1e-9]
                    past = np.maximum((q - end) * direction, 0.0).max(axis=0)
                    rows.append(
                        {"example": name, "law": law, "omega": omega, "damping": damping, "arm": arm}
                        | {"joint_1_deg": float(past[0]), "joint_2_deg": float(past[1])}
                    )
    write_csv(OVERSHOOT, rows)


def slug(name: str) -> str:
    """A file name part for an example's name."""
    return name.replace(" ", "_")


def example_start(scenario: str, esn: str, suffix: str, offset: tuple[float, float]) -> int:
    """The index of the start posture at ``offset`` (deg) in a robot run."""
    rows = read_csv(RESULTS / robot_run(scenario, esn, suffix) / "metrics.csv")
    return next(int(r["start"]) for r in rows if r["arm"] == "demonstrator" and offset_of(r["origin"]) == offset)


def example_paths(
    scenario: str, offset: tuple[float, float], law: str, omega: float, damping: float | None = None
) -> dict[str, Path]:
    """The logs of an example: the three tracked arms, the demonstrator, and its undisturbed reach from that start.

    With a ``damping`` ratio, the logs are those of the underdamped runs.
    """
    setting, suffix = setting_name(law, omega)
    if damping is not None:
        setting, suffix = f"{law}_w{omega:g}" + ("" if damping == 1.0 else f"_z{damping:g}"), "_damping"
    paths = {}
    for arm, esn, name in (("tuned", "tuned", "esn"), ("eight", "eight", "esn"), ("replay", "tuned", "replay")):
        paths[arm] = robot_log_path(scenario, esn, suffix, setting, name, example_start(scenario, esn, suffix, offset))
    start = example_start(scenario, "tuned", suffix, offset)
    paths["demonstrator"] = robot_log_path(scenario, "tuned", suffix, None, "demonstrator", start)
    undisturbed = scenario if scenario not in SPANS else "nominal"
    start = example_start(undisturbed, "tuned", suffix, offset)
    paths["undisturbed"] = robot_log_path(undisturbed, "tuned", suffix, None, "demonstrator", start)
    return paths


def example_title(name: str, offset: tuple[float, float]) -> str:
    """Where an example starts, for titles."""
    if offset == (0.0, 0.0):
        return f"{name}, from the demonstrated start"
    return f"{name}: start offset by {offset[0]:+g}° in joint 1 and {offset[1]:+g}° in joint 2"


def plot_example(name: str, scenario: str, offset: tuple[float, float], damping: float | None = None) -> Figure:
    """The joint angles of one example over time: each ESN's arm and output against the replay and the demonstrator.

    One column per ESN and featured tracker setting, one row per joint. With a
    ``damping`` ratio, the trackers are underdamped, and the figure covers the whole run.
    """
    columns = [(esn, law, omega) for esn in ROBOT_ESNS for law, omega, _ in FEATURED]
    logs = {
        (law, omega): {
            arm: StateLog.load(path) for arm, path in example_paths(scenario, offset, law, omega, damping).items()
        }
        for law, omega, _ in FEATURED
    }
    end = EXAMPLE_END if damping is None else UNDERDAMPED_END
    times = np.arange(0.0, end + 1e-9, 0.005)
    from_start = offset != (0.0, 0.0)
    fig = Figure(figsize=(18, 7.2), facecolor=SURFACE_COLOR, layout="constrained")
    damped = "" if damping is None else f", trackers underdamped (ζ = {damping:g})"
    fig.suptitle(f"Joint angles over time: {example_title(name, offset)}{damped}", color="#0b0b0b")
    axes = fig.subplots(2, len(columns), sharex=True, squeeze=False)
    for column, (esn, law, omega) in enumerate(columns):
        run = logs[(law, omega)]
        undisturbed_label = (
            "demonstrator's reach from this start" if from_start else "demonstration (the replay's reference)"
        )
        lines = [(joint_angles_at(run["undisturbed"], times), undisturbed_label, "#c9c8c2", "-", 5.0)]
        if scenario in SPANS:
            lines.append((joint_angles_at(run["demonstrator"], times), "demonstrator, disturbed", "#0b0b0b", "-.", 1.0))
        if from_start:
            lines.append(
                (joint_angles_at(run["replay"], times, "q_ref"), "replay: reference", ARM_COLORS["replay"], "--", 1.3)
            )
        lines += [
            (joint_angles_at(run["replay"], times), "replay: arm", ARM_COLORS["replay"], "-", 1.3),
            (
                joint_angles_at(run[esn], times, "q_ref"),
                f"{ARM_LABELS[esn]}: output (reference)",
                ARM_COLORS[esn],
                "--",
                1.8,
            ),
            (joint_angles_at(run[esn], times), f"{ARM_LABELS[esn]}: arm", ARM_COLORS[esn], "-", 1.8),
        ]
        for joint in range(2):
            ax = axes[joint, column]
            style(ax)
            if scenario in SPANS:
                ax.axvspan(*SPANS[scenario], color=DISTURBANCE_COLOR, zorder=0)
            for values, label, color, line, width in lines:
                ax.plot(times, values[:, joint], color=color, linestyle=line, linewidth=width, label=label)
            ax.set_title(
                f"{ARM_LABELS[esn]}\n{LAW_NAMES[law]}, ω = {omega:g} rad/s: joint {joint + 1}",
                color=TEXT_COLOR,
                fontsize=9,
            )
            if column == 0:
                ax.set_ylabel(f"joint {joint + 1} (deg)", color=TEXT_COLOR)
        axes[1, column].set_xlabel("time (s)", color=TEXT_COLOR)
    handles, labels = {}, []
    for ax in (axes[0, 0], axes[0, len(columns) // 2]):
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            if label not in handles:
                handles[label] = handle
                labels.append(label)
    fig.legend(
        [handles[label] for label in labels],
        labels,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        labelcolor=TEXT_COLOR,
    )
    return fig


def export_example_animation(name: str, scenario: str, offset: tuple[float, float]) -> None:
    """Animate an example: the four arms side by side, each rendered by skelarm's player, on one task clock.

    The player exports each arm's run as a GIF (``--export``); its frames, which the
    GIF merges where the arm rests, are spread back over time, cropped to where the
    arms move, labeled, and tiled into one GIF with the task time and the
    disturbance.
    """
    import tempfile

    from PIL import Image, ImageDraw, ImageFont, ImageSequence

    from arm_esn_ctrl.storage import REPO_ROOT

    player = REPO_ROOT / "third_party" / "skelarm" / "tools" / "player.py"
    law, omega = ANIMATION_SETTING
    paths = example_paths(scenario, offset, law, omega)
    arms = ("tuned", "eight", "replay", "demonstrator")
    frames, starts = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for arm in arms:
            exported = Path(tmp) / f"{arm}.gif"
            subprocess.run(
                [
                    sys.executable,
                    str(player),
                    str(paths[arm]),
                    "--export",
                    str(exported),
                    "--fps",
                    f"{ANIMATION_FPS:g}",
                ],
                check=True,
                capture_output=True,
                env=os.environ | {"QT_QPA_PLATFORM": "offscreen"},  # render without a window
            )
            with Image.open(exported) as gif:
                spread = []
                for frame in ImageSequence.Iterator(gif):
                    repeats = max(round(frame.info.get("duration", 1000.0 / ANIMATION_FPS) * ANIMATION_FPS / 1000.0), 1)
                    spread += [np.asarray(frame.convert("RGB"))] * repeats
            frames[arm] = np.array(spread)
            starts[arm] = float(StateLog.load(paths[arm]).times[0])
    ink = np.any([(f < 235).any(axis=3).any(axis=0) for f in frames.values()], axis=0)
    rows, cols = np.nonzero(ink)
    margin = 24
    top, bottom = max(rows.min() - margin, 0), rows.max() + margin
    left, right = max(cols.min() - margin, 0), cols.max() + margin
    width, height = right - left, bottom - top
    header, label_height, gap = 44, 30, 8
    font = ImageFont.load_default(size=20)
    small = ImageFont.load_default(size=17)
    labels = {
        "tuned": "ESN, tuned settings",
        "eight": "ESN, eight-demo. settings",
        "replay": "replay",
        "demonstrator": "demonstrator",
    }
    images = []
    times = np.arange(ANIMATION_SPAN[0], ANIMATION_SPAN[1] + 1e-9, 1.0 / ANIMATION_FPS)
    for t in times:
        canvas = Image.new("RGB", (len(arms) * width + (len(arms) - 1) * gap, header + label_height + height), "white")
        draw = ImageDraw.Draw(canvas)
        status = ""
        if scenario in SPANS and SPANS[scenario][0] <= t < SPANS[scenario][1]:
            status = "   pushed" if scenario.startswith("push") else "   blocked"
        draw.text((10, 10), f"{example_title(name, offset)}   t = {t:+.2f} s{status}", fill="#0b0b0b", font=font)
        for i, arm in enumerate(arms):
            k = int(np.clip(round((t - starts[arm]) * ANIMATION_FPS), 0, len(frames[arm]) - 1))
            x = i * (width + gap)
            canvas.paste(Image.fromarray(frames[arm][k][top:bottom, left:right]), (x, header + label_height))
            draw.text((x + 8, header + 4), labels[arm], fill=ARM_COLORS[arm], font=small)
        images.append(canvas)
    # One palette for every frame, so that the GIF stores only what changes from frame to frame.
    # It is learned from all the frames, so that brief colors, such as a push's arrow, keep theirs.
    sample = Image.new("RGB", (images[0].width, len(images) * images[0].height))
    for i, image in enumerate(images):
        sample.paste(image, (0, i * images[0].height))
    palette = sample.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    images = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    durations = [round(1000.0 / ANIMATION_FPS)] * (len(images) - 1) + [ANIMATION_HOLD_MS]
    gif = SUMMARY / f"example_{slug(name)}.gif"
    images[0].save(gif, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    print(f"wrote {gif.name}: {len(images)} frames")


# ---------------------------------------------------------------------------- the tables


def print_tables() -> None:
    """Print the tables of the report."""
    print("== 3.1 Autonomous runs over the grid of start offsets (168 offset starts)")
    print("settings: arrive and hold; path distance mean (mm); training path ratio median; first step mean (mm)")
    for name, run in ESNS.values():
        rows = read_csv(RESULTS / run / "metrics.csv")
        offset = [r for r in rows if r["origin"] != "demo_00"]
        demonstrated = next(r for r in rows if r["origin"] == "demo_00")
        ratios = [r["training_path_ratio"] for r in offset if np.isfinite(r["training_path_ratio"])]
        print(
            f"  {name}: {sum(r['success'] for r in rows)}/{len(rows)};"
            f" {1000 * mean(offset, 'reach_path_distance_m'):.1f};"
            f" {np.median(ratios):.2f}; {1000 * mean(offset, 'first_step_m'):.1f}"
            f" | demonstrated start: joint error {demonstrated['reach_joint_error_deg']:.2f} deg,"
            f" path {1000 * demonstrated['reach_path_distance_m']:.1f} mm,"
            f" first step {1000 * demonstrated['first_step_m']:.1f} mm"
        )

    print("\n== 3.2 Reservoir states of the 168 offset starts: time-matched distance relative to the spread of the")
    print("demonstration's states, median (range) at t = 0, 0.25, 0.5, 1 s; join time median (max);")
    print("phase lead median (range), and its correlation with the offset along the demonstrated motion")
    from arm_esn_ctrl.demonstrations import load_joint_angles

    q = np.degrees(load_joint_angles(DEMONSTRATION, 0.01)[1])
    motion = (q[-1] - q[0]) / np.linalg.norm(q[-1] - q[0])
    for key, run in STATES_RUNS.items():
        rows = [r for r in read_csv(RESULTS / run / "states.csv") if r["offset_deg"] > 1e-9]
        cells = []
        for column in ("warmup_distance", "distance_0.25s", "distance_0.5s", "distance_1s"):
            v = [r[column] for r in rows]
            cells.append(f"{np.median(v):.2f} ({min(v):.2f}-{max(v):.2f})")
        joins = [r["join_time_s"] for r in rows]
        leads = np.array([r["phase_lead_s"] for r in rows])
        along = np.array([[r["offset_q1_deg"], r["offset_q2_deg"]] for r in rows]) @ motion
        print(
            f"  {ESNS[key][0]}: {'; '.join(cells)}; join {np.median(joins):.2f} ({max(joins):.2f}) s;"
            f" lead {np.median(leads):+.3f} ({leads.min():+.2f} to {leads.max():+.2f}) s,"
            f" correlation {np.corrcoef(along, leads)[0, 1]:+.2f}"
        )

    print("\n== 3.3 Warm-up: median arrival delay of the offset starts (s), demonstrated start (s), failed runs,")
    print("median training path ratio of the offset starts")
    for key, run in WARMUP_RUNS.items():
        rows = read_csv(RESULTS / run / "warmup.csv")
        print(f"  {ESNS[key][0]} (trained with {trained_warmup(key):g} s):")
        for w in sorted({r["warmup_s"] for r in rows}):
            group = [r for r in rows if r["warmup_s"] == w]
            offset = [
                r["arrival_delay_s"]
                for r in group
                if r["group"] != "demonstrated" and np.isfinite(r["arrival_delay_s"])
            ]
            demonstrated = next(r["arrival_delay_s"] for r in group if r["group"] == "demonstrated")
            print(
                f"    {w:5.2f} s: {np.median(offset):+.2f}, {demonstrated:+.2f},"
                f" {sum(not r['success'] for r in group)}/{len(group)},"
                f" {np.median([r['training_path_ratio'] for r in group if np.isfinite(r['training_path_ratio'])]):.2f}"
            )

    print("\n== 3.4 Sweeps")
    for label, run in SWEEP_RUNS.items():
        rows = read_csv(RESULTS / run / "sweep.csv")
        held = [r for r in rows if r["failures"] == 0]
        ratios = [r["other_median_training_path_ratio"] for r in held]
        distances = [1000 * r["other_reach_path_distance_m"] for r in held]
        print(
            f"  {label}: {len(held)} of {len(rows)} hold everywhere; ratio {min(ratios):.2f}-{max(ratios):.2f}"
            f" (median {np.median(ratios):.2f}); path distance {min(distances):.1f}-{max(distances):.1f} mm"
        )
        for key in ("ridge", "leak_rate"):
            if key in rows[0]:
                counts = {
                    v: sum(r[key] == v and r["failures"] == 0 for r in rows) for v in sorted({r[key] for r in rows})
                }
                print(f"    hold everywhere, by {key}: " + ", ".join(f"{v:g}: {n}" for v, n in counts.items()))
        if "warmup" in rows[0]:
            for v in sorted({r["warmup"] for r in rows}):
                group = [r for r in rows if r["warmup"] == v]
                print(
                    f"    warm-up {v:g} s: mean first step from the offset starts"
                    f" {1000 * np.mean([r['other_first_step_m'] for r in group]):.0f} mm,"
                    f" path distance {1000 * min(r['other_reach_path_distance_m'] for r in group):.1f}"
                    f"-{1000 * max(r['other_reach_path_distance_m'] for r in group):.1f} mm"
                )
        if "input_scaling" in rows[0]:
            for v in sorted({r["input_scaling"] for r in rows}):
                group = [r for r in rows if r["input_scaling"] == v]
                print(
                    f"    input scaling {v:g}: {sum(r['failures'] == 0 for r in group)} of {len(group)}"
                    " hold everywhere,"
                    f" median failed runs {np.median([r['failures'] for r in group]):.0f}"
                )

    for law, omega, _ in FEATURED:
        print(f"\n== 3.5 Robot, {LAW_NAMES[law]}, ω = {omega:g} rad/s (tuned / eight / replay / demonstrator)")
        print("scenario: path distance (mm); arrive and hold (%); arrival (s); peak torque (N m); ∫τ² (N² m² s)")
        for scenario in SCENARIOS:
            cells = []
            for key, factor, digits in (
                ("reach_path_distance_m", 1000.0, 1),
                ("success", 1.0, 0),
                ("arrival_time_s", 1.0, 2),
                ("peak_torque_nm", 1.0, 1),
                ("effort_n2m2s", 1.0, 1),
            ):
                cells.append(
                    " / ".join(
                        f"{mean(robot_rows(scenario, arm, law, omega), key) * factor:.{digits}f}" for arm in ARMS
                    )
                )
            print(f"  {scenario}: " + "; ".join(cells))

    print(
        "\n== 3.5 From the four 10° diagonal offsets, as report 002's offset 10°: peak torque (N m); path distance (mm)"
    )
    diagonals = {(a, b) for a in (-10.0, 10.0) for b in (-10.0, 10.0)}
    for law, omega, _ in FEATURED:
        cells = []
        for arm in ("tuned", "eight", "replay"):
            rows = [r for r in robot_rows("offsets", arm, law, omega) if offset_of(r["origin"]) in diagonals]
            cells.append(f"{mean(rows, 'peak_torque_nm'):.0f}; {1000 * mean(rows, 'reach_path_distance_m'):.0f}")
        print(f"  {LAW_NAMES[law]} {omega:g} (tuned / eight / replay): " + " / ".join(cells))

    print("\n== 3.5 Joint PD at ω = 10 / 20 / 40: arrive and hold (tuned, eight, replay)")
    for scenario in SCENARIOS:
        cells = [
            " / ".join(f"{sum(r['success'] for r in robot_rows(scenario, arm, 'pd', w))}" for w in PD_OMEGAS)
            for arm in ("tuned", "eight", "replay")
        ]
        n = len(robot_rows(scenario, "replay", "pd", 10.0))
        print(f"  {scenario} (of {n}): " + "; ".join(cells))

    print("\n== 3.6 Block: reference lead at release; arrival (s); peak holding force (N); ∫τ²; path distance (mm)")
    settings = [("computed_torque", 10.0)] + [("pd", w) for w in PD_OMEGAS]
    for arm in ARMS:
        for law, omega in settings:
            rows = robot_rows("block", arm, law, omega)
            print(
                f"  {ARM_LABELS[arm]}, {LAW_NAMES[law]} {omega:g}: {mean(rows, 'reference_lead'):+.2f};"
                f" {mean(rows, 'arrival_time_s'):.2f}; {mean(rows, 'peak_external_force_n'):.0f};"
                f" {mean(rows, 'effort_n2m2s'):.1f}; {1000 * mean(rows, 'reach_path_distance_m'):.1f}"
            )
            if arm == "demonstrator":
                break

    print("\n== 3.6 Arrival (s) undisturbed, and its shift pushed forward / backward / blocked")
    for arm in ARMS:
        for law, omega in settings:
            arrival = {
                s: mean(robot_rows(s, arm, law, omega), "arrival_time_s")
                for s in ("nominal", "push forward", "push backward", "block")
            }
            shifts = " / ".join(
                f"{arrival[s] - arrival['nominal']:+.2f}" for s in ("push forward", "push backward", "block")
            )
            print(f"  {ARM_LABELS[arm]}, {LAW_NAMES[law]} {omega:g}: {arrival['nominal']:.2f}; {shifts}")
            if arm == "demonstrator":
                break

    print("\n== 3.9 Underdamped: of the 7 example runs, those that arrive and hold; those that settle;")
    print("the worst final distance (mm)")
    for law, omega, _ in FEATURED:
        for damping in DAMPINGS:
            cells = []
            for arm in ("tuned", "eight", "replay"):
                rows = [r for scenario in SCENARIOS for r in damping_rows(scenario, arm, law, damping)]
                settled = sum(np.isfinite(r["settling_time_s"]) for r in rows)
                worst = 1000 * max(r["final_distance_m"] for r in rows)
                cells.append(f"{sum(r['success'] for r in rows)}, {settled}, {worst:.1f}")
            print(f"  {LAW_NAMES[law]} {omega:g}, ζ = {damping:g} (tuned / eight / replay): " + " / ".join(cells))
    print("\n== 3.9 Underdamped: per scenario, arrive and hold (tuned / eight / replay), ζ = 1 / 0.5 / 0.3 / 0.1")
    for law, omega, _ in FEATURED:
        for scenario in SCENARIOS:
            cells = []
            for arm in ("tuned", "eight", "replay"):
                cells.append(
                    " ".join(f"{sum(r['success'] for r in damping_rows(scenario, arm, law, z))}" for z in DAMPINGS)
                )
            n = len(damping_rows(scenario, "replay", law, 1.0))
            print(f"  {LAW_NAMES[law]} {omega:g}, {scenario} (of {n}): " + " | ".join(cells))
    if OVERSHOOT.exists():
        print("\n== 3.9 Overshoot past the end posture, the larger of the two joints (deg), ζ = 1 → 0.1")
        print("(tuned / eight / replay)")
        rows = read_csv(OVERSHOOT)
        for law, omega, _ in FEATURED:
            for name, _, _ in EXAMPLES:
                cells = []
                for arm in ("tuned", "eight", "replay"):
                    values = []
                    for damping in (1.0, UNDERDAMPED):
                        r = next(
                            r
                            for r in rows
                            if (r["example"], r["law"], r["damping"], r["arm"]) == (name, law, damping, arm)
                        )
                        values.append(max(r["joint_1_deg"], r["joint_2_deg"]))
                    cells.append(f"{values[0]:.1f} → {values[1]:.1f}")
                print(f"  {LAW_NAMES[law]} {omega:g}, {name}: " + " / ".join(cells))
    if RINGING.exists():
        print("\n== 3.9 Ringing after the reach (ζ = 0.1): the output's amplitude against the arm's, and its lag (s),")
        print("median (range) over the ringing joints of the 7 examples")
        rows = read_csv(RINGING)
        for law, omega, _ in FEATURED:
            for esn in ROBOT_ESNS:
                group = [r for r in rows if r["law"] == law and r["esn"] == esn]
                if not group:
                    print(f"  {LAW_NAMES[law]} {omega:g}, {ARM_LABELS[esn]}: no joint rings")
                    continue
                ratios, lags = [r["amplitude_ratio"] for r in group], [r["lag_s"] for r in group]
                print(
                    f"  {LAW_NAMES[law]} {omega:g}, {ARM_LABELS[esn]}: {len(group)} joints; amplitude"
                    f" {np.median(ratios):.2f} ({min(ratios):.2f}-{max(ratios):.2f}); lag {np.median(lags):+.3f}"
                    f" ({min(lags):+.3f} to {max(lags):+.3f})"
                )

    if PUSH_PROGRESS.exists():
        print("\n== 3.6 Progress moved by the pushes along the reach (fraction of the path), at 0.5 s and 0.6 s")
        for r in read_csv(PUSH_PROGRESS):
            print(
                f"  {r['direction']}, {LAW_NAMES[r['law']]} {r['omega']:g}, {ARM_LABELS[r['arm']]}:"
                f" {r['moved_at_0.5s']:+.3f}, {r['moved_at_0.6s']:+.3f}"
            )
    if DEPARTURES.exists():
        print("\n== 3.7 Departure from the demonstration: output / arm (deg), ratio")
        for r in read_csv(DEPARTURES):
            print(
                f"  {r['scenario']}, {LAW_NAMES[r['law']]} {r['omega']:g}, {ARM_LABELS[r['esn']]}:"
                f" {r['output_departure_deg']:.1f} / {r['arm_departure_deg']:.1f}, {r['ratio']:.2f}"
            )
    if OFFSET_REFERENCES.exists():
        print(
            "\n== 3.7 Offsets: RMS distance of the ESN's output from the demonstrator's reach from that start,"
            " and from the replayed demonstration (deg): median (range)"
        )
        rows = read_csv(OFFSET_REFERENCES)
        for law, omega, _ in FEATURED:
            for esn in ROBOT_ESNS:
                group = [
                    r
                    for r in rows
                    if r["law"] == law
                    and r["omega"] == omega
                    and r["esn"] == esn
                    and (r["offset_q1_deg"], r["offset_q2_deg"]) != (0.0, 0.0)
                ]
                cells = []
                for key in ("from_this_start_deg", "from_replayed_deg"):
                    v = [r[key] for r in group]
                    cells.append(f"{np.median(v):.1f} ({min(v):.1f}-{max(v):.1f})")
                print(f"  {LAW_NAMES[law]} {omega:g}, {ARM_LABELS[esn]}: " + "; ".join(cells))


if __name__ == "__main__":
    main()
