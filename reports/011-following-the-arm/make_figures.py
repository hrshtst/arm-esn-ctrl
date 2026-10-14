# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 011.

    uv run python reports/011-following-the-arm/make_figures.py
    uv run python reports/011-following-the-arm/make_figures.py --logs --animations

From the copies of the runs under ``results/`` (configurations, run records, metrics,
and sweep tables), it draws into ``results/summary``:

- ``grid.png``: the grid, each setting's median ringing against its median following
  over its ten seeds, with the replay's (Section 3.1);
- ``undisturbed.png``: undisturbed, the path RMSE and the arrival of the two chosen
  ESNs and the replay, and from the offset starts, their median and largest peak
  torque, against the damping ratio (Sections 3.4 and 3.6);
- ``disturbed.png``: after the block's release and each push, their following,
  ringing, and return against the damping ratio (Section 3.5);

and prints the tables of the report. With ``--logs``, it reads the runs' logs, which
are not kept in Git, from the runs under the storage root, and also draws each arm's
distance from the taught path under the block and the pushes, one panel per tracker
(``block_distances.png``, ``pushes_distances.png``), and the joint angles and the
joint torques over time under the block and the pushes, and from the start offset by
+15 deg in both joints, one panel per joint and damping ratio, for each reference
velocity (``<scenario>_joints_<velocity>.png``, ``<scenario>_torques_<velocity>.png``,
Sections 3.5 and 3.6). With ``--animations``, it animates the block and the pushes
with the arms of PANELS side by side, each rendered by skelarm's player
(``block.gif``, ``pushes.gif``).
"""

from __future__ import annotations

import argparse
import csv
import functools
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter
from numpy.typing import NDArray
from skelarm import Skeleton, StateLog

from arm_esn_ctrl.demonstrations import endpoint_positions, resample_joint_angles
from arm_esn_ctrl.disturbances import disturbance_spans
from arm_esn_ctrl.metrics import distances_to_path
from arm_esn_ctrl.storage import REPO_ROOT, resolve_run_path

REPORT = Path(__file__).resolve().parent
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"
# The take, as report 005 copied it: the runs here read it from the storage root, as there.
TAKE_RUN = REPORT.parent / "005-tuned-on-the-robot" / "data" / "20261008-170034-reach_manual_v2"
DT = 0.01  # the ESNs' period (s)
GOAL = 0.02  # the goal's radius (m): a run ending farther from the target ends away from it

# The grid: its leak rates (one sweep each per scenario) and scenarios.
LEAKS = (0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1)
SCENARIOS = ("nominal", "block", "pushes")
# Report 007's chosen setting (leak rate, spectral radius, input scaling); its ESN is seed 7.
REFERENCE = (0.02, 1.35, 2.0)
# After each disturbance (the block's release, then the three pushes), a hand not back on the path before the next
# one or the end of the 24 s run counts as back then (s after its end).
RETURN_CAPS = (24.0 - 5.5, 7.0 - 3.5, 15.0 - 7.5, 24.0 - 15.5)
# The shortlist: its name, its configurations' tag, its setting, and its seed. The first two were chosen in the app;
# report 007's chosen ESN joined them by the rule.
SHORTLIST = [
    ("ESN #1", "lr0.03_sr1.2_is1.5_seed8", (0.03, 1.2, 1.5), 8),
    ("ESN #2", "lr0.03_sr1.2_is1_seed2", (0.03, 1.2, 1.0), 2),
    ("ESN #3", "lr0.03_sr1.35_is1.5_seed6", (0.03, 1.35, 1.5), 6),
    ("report 007's chosen ESN", "lr0.02_sr1.35_is2_seed7", REFERENCE, 7),
]

# The validation: the trackers' damping ratios, and their reference velocity (the configurations' tag, and its name).
DAMPINGS = (0.2, 0.4, 0.8)
VELOCITIES = {"zero": "zero reference velocity", "reference": "reference velocity"}
SEARCH_TRACKER = ("reference", 0.4)  # the search's tracker: given the reference velocity, damping ratio 0.4
# The arms compared: name, the tag of the runs, the arm in them, color, and marker. The replay is the same in every
# ESN's runs; it is read from ESN #1's.
ARMS = [
    ("ESN #1", "lr0.03_sr1.2_is1.5_seed8", "esn", "#2a78d6", "o"),
    ("ESN #2", "lr0.03_sr1.2_is1_seed2", "esn", "#5b2a86", "s"),
    ("replay", "lr0.03_sr1.2_is1.5_seed8", "replay", "#eb6834", "D"),
]
# The disturbances, each with its run, the suffix of its columns in metrics.csv, its name, and when a hand not back
# counts as back (RETURN_CAPS).
DISTURBANCES = [
    ("block", "", "the block's release", RETURN_CAPS[0]),
    ("pushes", "_1", "the push forward at 3 s", RETURN_CAPS[1]),
    ("pushes", "_2", "the push across at 7 s", RETURN_CAPS[2]),
    ("pushes", "_3", "the push across at 15 s", RETURN_CAPS[3]),
]
BACK_WITHIN = 5.0  # mm: an arm is back on the taught path once it stays within this of it
# The time series: the scenario, the start (its index among the run's starts), and the time shown (s).
SERIES = {
    "block": ("block", 0, (0.0, 20.0)),
    "pushes": ("pushes", 0, (0.0, 20.0)),
    "offset": ("offsets", 48, (0.0, 4.0)),
}

TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
DISTURBANCE_COLOR = "#f0efec"
TAUGHT_COLOR = "#b9b7b0"
LINESTYLES = {"zero": "-", "reference": "--"}

# The animations, with the search's tracker: per panel, the label (the arm's reference, then its tracker), the arm's
# runs' tag, the arm, and the label's color.
PANELS = [
    ("ESN #1 (input scaling 1.5)\ntracker: joint PD, damping ratio 0.4", ARMS[0][1], "esn", ARMS[0][3]),
    ("ESN #2 (input scaling 1)\ntracker: joint PD, damping ratio 0.4", ARMS[1][1], "esn", ARMS[1][3]),
    ("replay of the take by time\ntracker: joint PD, damping ratio 0.4", ARMS[2][1], "replay", ARMS[2][3]),
]
ANIMATION_SPAN = (-0.2, 20.0)
ANIMATION_FPS = 10.0
ANIMATION_HOLD_MS = 1500  # the last frame stays this long before the GIF loops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", action="store_true", help="also analyze the run logs in the storage root")
    parser.add_argument("--animations", action="store_true", help="also export GIFs from the logs in the storage root")
    args = parser.parse_args()

    SUMMARY.mkdir(parents=True, exist_ok=True)
    figures = {"grid.png": plot_grid(), "undisturbed.png": plot_undisturbed(), "disturbed.png": plot_disturbed()}
    if args.logs:
        figures |= {f"{scenario}_distances.png": plot_distances(scenario) for scenario in ("block", "pushes")}
        for name in SERIES:
            for velocity in VELOCITIES:
                for channel in ("joints", "torques"):
                    figures[f"{name}_{channel}_{velocity}.png"] = plot_series(name, velocity, channel)
    for name, fig in figures.items():
        fig.savefig(SUMMARY / name, dpi=150)
    print_tables()
    if args.animations:
        tracker = "joint PD at 0.4, reference velocity, the torque every 10 ms"
        export_animation("block", "block", f"the block, from 4.5 s to 5.5 s; {tracker}")
        export_animation("pushes", "pushes", f"three pushes, at 3 s, 7 s, and 15 s; {tracker}")
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


@functools.cache
def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return [{key: value(text) for key, text in row.items()} for row in csv.DictReader(f)]


def run_name(stem: str) -> str:
    """The name of the one run of a configuration, by its stem."""
    matches = sorted(p.name for p in RESULTS.glob(f"*-{stem}") if p.is_dir())
    if len(matches) != 1:
        msg = f"expected one run of {stem}, found {matches}"
        raise FileNotFoundError(msg)
    return matches[0]


def stem_of(scenario: str, tag: str, velocity: str) -> str:
    """A tracker run's configuration stem, by its scenario, its ESN's tag, and its reference velocity."""
    return f"{scenario}_robot_{tag}_{velocity}_velocity_filtered"


def rows_of(scenario: str, tag: str, velocity: str) -> list[dict[str, Any]]:
    return read_csv(RESULTS / run_name(stem_of(scenario, tag, velocity)) / "metrics.csv")


def run_of(scenario: str, tag: str, arm: str, velocity: str, damping: float, start: int = 0) -> dict[str, Any]:
    """One arm's run from one start, with one tracker."""
    return next(
        r
        for r in rows_of(scenario, tag, velocity)
        if r["damping"] == damping and r["arm"] == arm and r["start"] == start
    )


def offset_runs(tag: str, arm: str, velocity: str, damping: float) -> list[dict[str, Any]]:
    """One arm's runs from the 49 starts, with one tracker."""
    return [r for r in rows_of("offsets", tag, velocity) if r["damping"] == damping and r["arm"] == arm]


def returned(run: dict[str, Any], suffix: str, cap: float) -> float:
    """When the hand is back on the path after a disturbance (s), or ``cap`` if it is not back by the window's end."""
    back = run[f"return_time_s{suffix}"]
    return float(back) if np.isfinite(back) else cap


@functools.cache
def taught() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The take filtered at 2 Hz, what the ESNs were trained on, every 10 ms: its joint angles and its hand path."""
    skeleton = Skeleton.from_toml(TAKE_RUN / "config.toml")
    q = resample_joint_angles(StateLog.load(TAKE_RUN / "demo_00_filtered.sklog.npz"), DT)[1]
    return q, endpoint_positions(skeleton, q)


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


# ---------------------------------------------------------------------------- the search


def failure_of(run: dict[str, Any], scenario: str) -> str:
    """How a run of a sweep fails by the rule, or "" if it does not: under the pushes, leaving the goal is excused."""
    if not np.isfinite(run["arrival_time_s"]):
        return "never arrives"
    if run["final_distance_m"] > GOAL:
        return "ends away"
    if scenario == "pushes":
        return ""
    if run["left_goal"]:
        return "leaves the goal during the hold"
    # A sweep's runs record whether the run failed; robot_esn.py's, whether it succeeded.
    failed = run["failed"] if "failed" in run else not run["success"]
    return "hold cut short" if failed else ""


def seed_measures(n: dict[str, Any], b: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    """One seed's measures from its runs undisturbed (``n``), under the block (``b``), and under the pushes (``p``).

    ``failures``, how each scenario's run fails by the rule, if it does; ``rmse``, the
    undisturbed path RMSE (mm); after each disturbance (the block's release, then the
    three pushes), ``gains``, the following gain, ``ringings`` (mm), ``returns`` (s, a
    hand not back counting as back at RETURN_CAPS), and ``offs``, the farthest from
    the path (mm); ``following``, ``ringing``, and ``ret``, their means over the four
    disturbances; and under the block, the reference ``lead`` (% of the path) and the
    peak torque (N m).
    """
    runs = dict(zip(SCENARIOS, (n, b, p), strict=True))
    gains = [b["following_gain"]] + [p[f"following_gain_{k}"] for k in (1, 2, 3)]
    ringings = [1000 * b["ringing_m"]] + [1000 * p[f"ringing_m_{k}"] for k in (1, 2, 3)]
    returns = [b["return_time_s"]] + [p[f"return_time_s_{k}"] for k in (1, 2, 3)]
    returns = [r if np.isfinite(r) else cap for r, cap in zip(returns, RETURN_CAPS, strict=True)]
    return {
        "seed": int(n["seed"]),
        "failures": {s: failure_of(r, s) for s, r in runs.items() if failure_of(r, s)},
        "rmse": 1000 * n["taught_path_rmse_m"],
        "gains": gains,
        "following": float(np.mean(gains)),
        "ringings": ringings,
        "ringing": float(np.mean(ringings)),
        "returns": returns,
        "ret": float(np.mean(returns)),
        "offs": [1000 * b["off_path_m"]] + [1000 * p[f"off_path_m_{k}"] for k in (1, 2, 3)],
        "lead": 100 * b["reference_lead"],
        "block_torque": b["peak_torque_nm"],
    }


Setting = tuple[float, float, float]


@functools.cache
def grid_seeds() -> dict[Setting, list[dict[str, Any]]]:
    """Each seed of every setting of the grid, by (leak rate, spectral radius, input scaling): :func:`seed_measures`."""
    out: dict[Setting, list[dict[str, Any]]] = {}
    for leak in LEAKS:
        runs = {
            scenario: {
                (r["spectral_radius"], r["input_scaling"], r["seed"]): r
                for r in read_csv(
                    RESULTS / run_name(f"sweep_robot_following_lr{leak:g}_{scenario}_filtered") / "runs.csv"
                )
            }
            for scenario in SCENARIOS
        }
        for radius, scaling, seed in sorted(runs["nominal"]):
            seeds = out.setdefault((leak, radius, scaling), [])
            seeds.append(seed_measures(*(runs[scenario][radius, scaling, seed] for scenario in SCENARIOS)))
    return out


def passes(seeds: list[dict[str, Any]]) -> bool:
    """A setting whose seeds all pass the undisturbed, block, and pushes runs (the rule's first points)."""
    return not any(s["failures"] for s in seeds)


@functools.cache
def offset_seeds() -> dict[Setting, dict[int, list[str]]]:
    """For each setting that passes, its seeds failing from an offset start, with the starts and how they fail."""
    out: dict[Setting, dict[int, list[str]]] = {}
    for k, seeds in grid_seeds().items():
        if not passes(seeds):
            continue
        stem = f"sweep_robot_following_offsets_lr{k[0]:g}_sr{k[1]:g}_is{k[2]:g}_filtered"
        failing: dict[int, list[str]] = {}
        for r in read_csv(RESULTS / run_name(stem) / "runs.csv"):
            how = failure_of(r, "offsets")
            if how:
                failing.setdefault(int(r["seed"]), []).append(f"{str(r['origin']).split()[1]} deg: {how}")
        out[k] = failing
    return out


def qualifies(k: Setting) -> bool:
    """A setting whose seeds all pass the three scenarios and the eight offset starts."""
    return passes(grid_seeds()[k]) and not offset_seeds()[k]


def median(seeds: list[dict[str, Any]], key: str, i: int | None = None) -> float:
    """The median over seeds of a measure, or of its ``i``-th entry."""
    return float(np.median([s[key] if i is None else s[key][i] for s in seeds]))


def typical_seed(seeds: list[dict[str, Any]]) -> int:
    """The seed nearest the medians of the RMSE, the following, and the ringing, each by its interquartile range."""
    x = np.array([[s["rmse"], s["following"], s["ringing"]] for s in seeds])
    iqr = np.subtract(*np.percentile(x, [75, 25], axis=0))
    iqr[iqr == 0] = 1.0
    return int(seeds[int(np.argmin(np.sum(np.abs(x - np.median(x, axis=0)) / iqr, axis=1)))]["seed"])


def replay_ringing() -> float:
    """The replay's ringing (mm), its mean over the four disturbances, with the search's tracker: the rule's guard."""
    velocity, damping = SEARCH_TRACKER
    tag = ARMS[2][1]
    runs = [run_of(scenario, tag, "replay", velocity, damping) for scenario, *_ in DISTURBANCES]
    return float(
        np.mean([1000 * r[f"ringing_m{suffix}"] for r, (_, suffix, _, _) in zip(runs, DISTURBANCES, strict=True)])
    )


def plot_grid() -> Figure:
    """Every setting of the grid: the median over its seeds of the ringing against that of the following.

    Filled: the settings that qualify; hollow: those with a failing seed. The black
    star is report 007's chosen setting; the orange diamond the replay with the same
    tracker, which never follows.
    """
    settings = grid_seeds()
    fig = Figure(figsize=(9.0, 6.0), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The grid, joint PD at damping ratio 0.4, reference velocity, the torque every 10 ms:"
        "\nringing against following after the block and the pushes, each setting over its ten seeds",
        color="#0b0b0b",
    )
    ax = fig.subplots()
    style(ax)
    ramp = colormaps["viridis"]
    colors = {leak: ramp(0.92 * i / (len(LEAKS) - 1)) for i, leak in enumerate(LEAKS)}
    for k, seeds in settings.items():
        color = colors[k[0]]
        filled = passes(seeds) and qualifies(k)
        ax.scatter(
            median(seeds, "following"),
            median(seeds, "ringing"),
            s=36,
            zorder=3,
            linewidth=1.3,
            color=color if filled else SURFACE_COLOR,
            edgecolor=color,
        )
    for (name, _, k, _), offset in zip(SHORTLIST[:2], ((30, -36), (-60, -40)), strict=True):
        seeds = settings[k]
        ax.annotate(
            name,
            (median(seeds, "following"), median(seeds, "ringing")),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            color=TEXT_COLOR,
            arrowprops={"arrowstyle": "-", "color": TEXT_COLOR, "linewidth": 0.6},
        )
    reference = settings[REFERENCE]
    ax.scatter(
        [median(reference, "following")], [median(reference, "ringing")], marker="*", s=220, color="#0b0b0b", zorder=5
    )
    ax.scatter([0.0], [replay_ringing()], marker="D", s=60, color=ARMS[2][3], zorder=5)
    ax.set_xlim(-0.03, 1.0)
    ax.set_ylim(0, None)
    ax.set_xlabel("following: median over the seeds of the mean gain under the block and the pushes", color=TEXT_COLOR)
    ax.set_ylabel("ringing (mm): median over the seeds of the mean after the four", color=TEXT_COLOR)
    handles = [
        Line2D([], [], color=c, marker="o", linestyle="none", label=f"leak rate {k:g}") for k, c in colors.items()
    ]
    handles += [
        Line2D(
            [],
            [],
            color=TEXT_COLOR,
            marker="o",
            markerfacecolor=SURFACE_COLOR,
            linestyle="none",
            label="a seed fails (hollow)",
        ),
        Line2D([], [], color="#0b0b0b", marker="*", markersize=12, linestyle="none", label="report 007's setting"),
        Line2D([], [], color=ARMS[2][3], marker="D", linestyle="none", label="the replay"),
    ]
    ax.legend(handles=handles, frameon=False, labelcolor=TEXT_COLOR, fontsize=8, ncol=2, loc="upper left")
    return fig


# ---------------------------------------------------------------------------- the validation


def plot_undisturbed() -> Figure:
    """Undisturbed, the path RMSE and the arrival; from the offset starts, the peak torques; against the damping."""
    fig = Figure(figsize=(14, 4.2), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "Joint PD, the torque every 10 ms: ESN #1 (blue), ESN #2 (purple), and the replay (orange),"
        " against the damping ratio;\nsolid: zero reference velocity, dashed: reference velocity",
        color="#0b0b0b",
    )
    axes = fig.subplots(1, 3)
    titles = (
        "undisturbed: path RMSE (mm)",
        "undisturbed: arrival (s)",
        "offset starts: peak torque (N m), median and largest (▲)",
    )
    for ax, title in zip(axes, titles, strict=True):
        style(ax)
        ax.set_title(title, color="#0b0b0b", fontsize=10)
        ax.set_xticks(DAMPINGS)
        ax.set_xlabel("damping ratio", color=TEXT_COLOR)
    for velocity, linestyle in LINESTYLES.items():
        for _, tag, arm, color, marker in ARMS:
            nominal = [run_of("nominal", tag, arm, velocity, damping) for damping in DAMPINGS]
            kwargs = {"color": color, "linestyle": linestyle, "marker": marker, "markersize": 4, "linewidth": 1.4}
            axes[0].plot(DAMPINGS, [1000 * r["taught_path_rmse_m"] for r in nominal], **kwargs)
            axes[1].plot(DAMPINGS, [r["arrival_time_s"] for r in nominal], **kwargs)
            peaks = [[r["peak_torque_nm"] for r in offset_runs(tag, arm, velocity, d)] for d in DAMPINGS]
            axes[2].plot(DAMPINGS, [np.median(p) for p in peaks], **kwargs)
            axes[2].scatter(DAMPINGS, [max(p) for p in peaks], color=color, marker="^", s=24)
    for ax, ticks in ((axes[0], (0.2, 0.5, 1, 2, 5)), (axes[2], (5, 10, 20, 50, 100, 200))):
        ax.set_yscale("log")
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    return fig


def plot_disturbed() -> Figure:
    """After each disturbance, the following, the ringing, and the return of the three arms against the damping ratio.

    Solid lines: the trackers given a zero reference velocity; dashed: given the
    reference's velocity. A hand not back on the path before the next disturbance
    is drawn at the window's end, marked with a cross.
    """
    fig = Figure(figsize=(14, 9.0), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "After each disturbance, joint PD, the torque every 10 ms: ESN #1 (blue), ESN #2 (purple), and the replay"
        " (orange),\nagainst the damping ratio; solid: zero reference velocity, dashed: reference velocity",
        color="#0b0b0b",
    )
    measures = ("following", "ringing", "return")
    labels = {
        "following": "following gain",
        "ringing": "ringing over the next 2 s (mm)",
        "return": "back within 5 mm\nof the path (s)",
    }
    axes = fig.subplots(len(measures), len(DISTURBANCES), sharex=True)
    for col, (scenario, suffix, name, cap) in enumerate(DISTURBANCES):
        for row, measure in enumerate(measures):
            ax = axes[row, col]
            style(ax)
            for velocity, linestyle in LINESTYLES.items():
                for _, tag, arm, color, marker in ARMS:
                    runs = [run_of(scenario, tag, arm, velocity, damping) for damping in DAMPINGS]
                    if measure == "following":
                        y = [r[f"following_gain{suffix}"] for r in runs]
                    elif measure == "ringing":
                        y = [1000 * r[f"ringing_m{suffix}"] for r in runs]
                    else:
                        y = [returned(r, suffix, cap) for r in runs]
                    ax.plot(DAMPINGS, y, color=color, linestyle=linestyle, marker=marker, markersize=4, linewidth=1.4)
                    if measure == "return":
                        missing = [
                            d
                            for d, r in zip(DAMPINGS, runs, strict=True)
                            if not np.isfinite(r[f"return_time_s{suffix}"])
                        ]
                        ax.scatter(missing, [cap] * len(missing), marker="x", s=60, color=color, zorder=3)
            ax.set_xticks(DAMPINGS)
            if measure == "following":
                ax.set_ylim(-0.05, 1.0)
            else:
                ax.set_ylim(0, None)
            if row == 0:
                ax.set_title(name, color="#0b0b0b", fontsize=10)
            if row == len(measures) - 1:
                ax.set_xlabel("damping ratio", color=TEXT_COLOR)
            if col == 0:
                ax.set_ylabel(labels[measure], color=TEXT_COLOR)
    return fig


# ---------------------------------------------------------------------------- figures from the logs


def log_of(scenario: str, tag: str, arm: str, velocity: str, damping: float, start: int = 0) -> StateLog:
    """One arm's log from a run under the storage root."""
    run = resolve_run_path(f"results/{run_name(stem_of(scenario, tag, velocity))}")
    return StateLog.load(run / f"pd_w20_z{damping:g}" / f"{arm}_{start:02d}.sklog.npz")


def path_distances(scenario: str, tag: str, arm: str, velocity: str, damping: float) -> tuple[NDArray[np.float64], ...]:
    """The times (every 10 ms) and the hand's distance (mm) from the taught path in a run's first start."""
    log = log_of(scenario, tag, arm, velocity, damping)
    task = log.times >= 0
    skeleton = Skeleton.from_toml(TAKE_RUN / "config.toml")
    hand = endpoint_positions(skeleton, log.channel("q").reshape(len(log.times), -1)[task])
    every = round(DT / float(np.median(np.diff(log.times))))
    return log.times[task][::every], 1000 * distances_to_path(hand[::every], taught()[1])


def spans_of(scenario: str) -> list[tuple[float, float]]:
    """When each disturbance of a scenario acts, from its configuration."""
    with (RESULTS / run_name(stem_of(scenario, ARMS[0][1], "reference")) / "config.toml").open("rb") as f:
        return disturbance_spans(tomllib.load(f).get("disturbance"))


def plot_distances(scenario: str) -> Figure:
    """Each arm's distance from the taught path over time in a scenario, one panel per tracker."""
    fig = Figure(figsize=(14, 6.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        f"{scenario.capitalize()}, joint PD, the torque every 10 ms: how far each arm goes from the taught path,"
        " and how it comes back",
        color="#0b0b0b",
    )
    axes = fig.subplots(len(VELOCITIES), len(DAMPINGS), sharex=True, sharey=True)
    for row, (velocity, velocity_name) in enumerate(VELOCITIES.items()):
        for col, damping in enumerate(DAMPINGS):
            ax = axes[row, col]
            style(ax)
            for onset, end in spans_of(scenario):
                ax.axvspan(onset, end, color=DISTURBANCE_COLOR, zorder=0)
            for name, tag, arm, color, _ in ARMS:
                times, distance = path_distances(scenario, tag, arm, velocity, damping)
                ax.plot(times, distance, color=color, linewidth=1.2, label=name)
            ax.axhline(BACK_WITHIN, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
            ax.set_title(f"damping ratio {damping:g}, {velocity_name}", color="#0b0b0b", fontsize=10)
            ax.set_xlim(0, 20)
            ax.set_ylim(0, None)
            if col == 0:
                ax.set_ylabel("from the taught path (mm)", color=TEXT_COLOR)
            if row == len(VELOCITIES) - 1:
                ax.set_xlabel("time (s)", color=TEXT_COLOR)
    axes[0, 0].legend(frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_series(name: str, velocity: str, channel: str) -> Figure:
    """The joint angles or the joint torques of the three arms over time, one panel per joint and damping ratio.

    The joint angles are drawn over the taught motion (thick gray), with each ESN's
    reference, its output, dashed in its color; the replay's reference is the taught
    motion itself.
    """
    scenario, start, span = SERIES[name]
    where = {
        "block": "under the block",
        "pushes": "under the pushes",
        "offset": "from the start offset by +15° in both joints",
    }
    fig = Figure(figsize=(14, 6.4), facecolor=SURFACE_COLOR, layout="constrained")
    given = "a zero reference velocity" if velocity == "zero" else "the reference velocity"
    fig.suptitle(
        f"{'Joint angles' if channel == 'joints' else 'Joint torques'} {where[name]}, joint PD given {given},"
        " the torque every 10 ms:\nESN #1 (blue), ESN #2 (purple), and the replay (orange)"
        + ("; dashed: each ESN's reference; thick gray: the taught motion" if channel == "joints" else ""),
        color="#0b0b0b",
    )
    axes = fig.subplots(2, len(DAMPINGS), sharex=True)
    spans = [] if scenario == "offsets" else spans_of(scenario)
    q_taught = np.degrees(taught()[0])
    t_taught = DT * np.arange(len(q_taught))
    for col, damping in enumerate(DAMPINGS):
        logs = {arm_name: log_of(scenario, tag, arm, velocity, damping, start) for arm_name, tag, arm, _, _ in ARMS}
        for joint in range(2):
            ax = axes[joint, col]
            style(ax)
            for onset, end in spans:
                ax.axvspan(onset, end, color=DISTURBANCE_COLOR, zorder=0)
            if channel == "joints":
                ax.plot(t_taught, q_taught[:, joint], color=TAUGHT_COLOR, linewidth=4.0, zorder=1)
            for arm_name, _, arm, color, _ in ARMS:
                log = logs[arm_name]
                n = len(log.times)
                if channel == "joints":
                    ax.plot(
                        log.times,
                        np.degrees(log.channel("q").reshape(n, -1)[:, joint]),
                        color=color,
                        linewidth=1.2,
                        zorder=3,
                    )
                    if arm == "esn":
                        ax.plot(
                            log.times,
                            np.degrees(log.channel("q_ref").reshape(n, -1)[:, joint]),
                            color=color,
                            linewidth=1.0,
                            linestyle="--",
                            zorder=2,
                        )
                else:
                    ax.plot(
                        log.times, log.channel("tau").reshape(n, -1)[:, joint], color=color, linewidth=1.0, zorder=3
                    )
            ax.set_xlim(*span)
            if joint == 0:
                ax.set_title(f"damping ratio {damping:g}", color="#0b0b0b", fontsize=10)
            else:
                ax.set_xlabel("time (s)", color=TEXT_COLOR)
            if col == 0:
                unit = "deg" if channel == "joints" else "N m"
                ax.set_ylabel(f"joint {joint + 1} ({unit})", color=TEXT_COLOR)
    handles = [Line2D([], [], color=color, linewidth=1.4, label=arm_name) for arm_name, _, _, color, _ in ARMS]
    if channel == "joints":
        handles += [
            Line2D([], [], color=TEXT_COLOR, linewidth=1.0, linestyle="--", label="an ESN's reference"),
            Line2D([], [], color=TAUGHT_COLOR, linewidth=4.0, label="the taught motion"),
        ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


# ---------------------------------------------------------------------------- the animation


def player_export(log: Path, out: Path, fps: float) -> None:
    """Render a run's log into ``out`` (a GIF) with skelarm's player, without a window."""
    player = REPO_ROOT / "third_party" / "skelarm" / "tools" / "player.py"
    subprocess.run(
        [sys.executable, str(player), str(log), "--export", str(out), "--fps", f"{fps:g}"],
        check=True,
        capture_output=True,
        env=os.environ | {"QT_QPA_PLATFORM": "offscreen"},  # render without a window
    )


def export_animation(name: str, scenario: str, title: str) -> None:
    """Animate a scenario's demonstrated start into ``<name>.gif``: the arms of PANELS side by side, by skelarm.

    The player exports each arm's run as a GIF; its frames, which the GIF merges
    where the arm rests, are spread back over time, cropped to where the arms move,
    labeled, and tiled into one GIF with the task time and the disturbance acting.
    """
    import tempfile

    from PIL import Image, ImageDraw, ImageFont, ImageSequence

    velocity, damping = SEARCH_TRACKER
    with (RESULTS / run_name(stem_of(scenario, ARMS[0][1], velocity)) / "config.toml").open("rb") as f:
        disturbance = tomllib.load(f).get("disturbance")
    tables = [] if disturbance is None else [disturbance] if isinstance(disturbance, dict) else disturbance
    spans = disturbance_spans(disturbance)
    frames, starts = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (_, tag, arm, _) in enumerate(PANELS):
            path = resolve_run_path(f"results/{run_name(stem_of(scenario, tag, velocity))}")
            path = path / f"pd_w20_z{damping:g}" / f"{arm}_00.sklog.npz"
            exported = Path(tmp) / f"{i}.gif"
            player_export(path, exported, ANIMATION_FPS)
            with Image.open(exported) as gif:
                spread = []
                for frame in ImageSequence.Iterator(gif):
                    repeats = max(round(frame.info.get("duration", 1000.0 / ANIMATION_FPS) * ANIMATION_FPS / 1000.0), 1)
                    spread += [np.asarray(frame.convert("RGB"))] * repeats
            frames.append(np.array(spread))
            starts.append(float(StateLog.load(path).times[0]))
    ink = np.any([(f < 235).any(axis=3).any(axis=0) for f in frames], axis=0)
    rows, cols = np.nonzero(ink)
    margin = 24
    top, bottom = max(int(rows.min()) - margin, 0), int(rows.max()) + margin
    left, right = max(int(cols.min()) - margin, 0), int(cols.max()) + margin
    width, height = right - left, bottom - top
    header, label_height, gap = 44, 48, 8
    canvas_width = len(frames) * width + (len(frames) - 1) * gap

    def header_line(t: float) -> str:
        status = ""
        for table, (begin, end) in zip(tables, spans, strict=True):
            if begin <= t < end:
                status = f"   pushed ({table['force']:g} N)" if table["type"] == "push" else "   blocked"
        return f"{title}   t = {t:+.1f} s{status}"

    # One font for every frame: the largest, up to 20, in which the header fits, with a status or the last time.
    lines = [header_line(t) for t in (*[begin for begin, _ in spans], ANIMATION_SPAN[1])]
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    size = 20
    while size > 12 and max(measure.textlength(line, font=ImageFont.load_default(size=size)) for line in lines) > (
        canvas_width - 20
    ):
        size -= 1
    font, small = ImageFont.load_default(size=size), ImageFont.load_default(size=15)
    images = []
    for t in np.arange(ANIMATION_SPAN[0], ANIMATION_SPAN[1] + 1e-9, 1.0 / ANIMATION_FPS):
        canvas = Image.new("RGB", (canvas_width, header + label_height + height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), header_line(t), fill="#0b0b0b", font=font)
        for i, (label, _, _, color) in enumerate(PANELS):
            k = int(np.clip(round((t - starts[i]) * ANIMATION_FPS), 0, len(frames[i]) - 1))
            x = i * (width + gap)
            canvas.paste(Image.fromarray(frames[i][k][top:bottom, left:right]), (x, header + label_height))
            draw.text((x + 8, header + 4), label, fill=color, font=small)
        images.append(canvas)
    # One palette for every frame, learned from all of them, so that the GIF stores only what changes and brief
    # colors, such as a push's arrow, keep theirs.
    sample = Image.new("RGB", (images[0].width, len(images) * images[0].height))
    for i, image in enumerate(images):
        sample.paste(image, (0, i * images[0].height))
    palette = sample.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    images = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    durations = [round(1000.0 / ANIMATION_FPS)] * (len(images) - 1) + [ANIMATION_HOLD_MS]
    out = SUMMARY / f"{name}.gif"
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    print(f"wrote {out.name}: {len(images)} frames")


# ---------------------------------------------------------------------------- tables


def describe(seeds: list[dict[str, Any]]) -> str:
    """A setting's medians over its seeds, as the tables print them."""
    each = [
        " / ".join(f"{median(seeds, key, i):.{digits}f}" for i in range(4))
        for key, digits in (("gains", 2), ("ringings", 1), ("offs", 0), ("returns", 2))
    ]
    return (
        f"{median(seeds, 'following'):.2f} | {median(seeds, 'ringing'):.2f} | {median(seeds, 'ret'):.2f} |"
        f" {median(seeds, 'rmse'):.2f} | {' | '.join(each)} | {median(seeds, 'lead'):.0f}% |"
        f" {median(seeds, 'block_torque'):.1f}"
    )


HEADER = (
    "following | ringing (mm) | return (s) | undisturbed path RMSE (mm) | after the block's release and pushes 1, 2,"
    " 3: following | ringing (mm) | farthest from the path (mm) | back on it (s) | the block's reference lead |"
    " peak torque under the block (N m)"
)


def print_tables() -> None:
    print_grid()
    print_shortlist()
    print_validation()


def print_grid() -> None:
    settings = grid_seeds()
    print(
        "== 3.1 The grid: per leak rate, the settings that pass the three scenarios (of 15), how the failing runs fail,"
    )
    print("and the range of the settings' median following")
    for leak in LEAKS:
        group = {k: v for k, v in settings.items() if k[0] == leak}
        kinds: dict[str, int] = {}
        for seeds in group.values():
            for s in seeds:
                for scenario, how in s["failures"].items():
                    kinds[f"{scenario}: {how}"] = kinds.get(f"{scenario}: {how}", 0) + 1
        following = [median(v, "following") for v in group.values()]
        print(
            f"  leak rate {leak:g}: {sum(passes(v) for v in group.values())};"
            f" {dict(sorted(kinds.items(), key=lambda kv: -kv[1]))}; following {min(following):.2f} to"
            f" {max(following):.2f}"
        )
    print("\n== 3.2 The offset starts of the settings that pass: the failing seeds, from which starts")
    for k, failing in offset_seeds().items():
        print(f"  {k}: {failing or 'none'}")
    ranked = sorted((k for k in offset_seeds() if qualifies(k)), key=lambda k: -median(settings[k], "following"))
    guard = replay_ringing()
    print(
        f"\n== 3.2 The {len(ranked)} qualifying settings, ranked by the following (guard: ringing <= {guard:.2f} mm):"
    )
    print(f"medians over the seeds of the {HEADER}; the typical seed")
    for rank, k in enumerate(ranked, start=1):
        mark = "  <- report 007's" if k == REFERENCE else ""
        flag = "" if median(settings[k], "ringing") <= guard else "  [guard]"
        name = f"leak rate {k[0]:g}, spectral radius {k[1]:g}, input scaling {k[2]:g}"
        print(f"  {rank}. {name}: {describe(settings[k])} | seed {typical_seed(settings[k])}{flag}{mark}")
    for name, _, k, _ in SHORTLIST[:3]:
        a, b = settings[k], settings[REFERENCE]
        better = sum(x["following"] > y["following"] for x, y in zip(a, b, strict=True))
        print(f"  {name}'s setting follows more than report 007's in {better} of 10 seeds (the same reservoirs)")


def print_shortlist() -> None:
    print("\n== 3.3 The shortlist: the ESN's own following (block; pushes), ringing (mm), return (s), path RMSE (mm);")
    print("on its own, the starts that arrive and hold, and its first step from the 48 offset starts, mean (mm);")
    print("with the six trackers: runs that fail, by the rule (undisturbed, block, pushes, 49 offset starts)")
    for name, tag, k, seed in SHORTLIST:
        s = next(x for x in grid_seeds()[k] if x["seed"] == seed)
        if tag == "lr0.02_sr1.35_is2_seed7":
            rows = read_csv(
                REPORT.parent
                / "007-ringing-with-joint-pd"
                / "results"
                / "20261013-153020-grid_lr0.02_sr1.35_is2_seed7_filtered"
                / "metrics.csv"
            )
        else:
            rows = read_csv(RESULTS / run_name(f"grid_{tag}_filtered") / "metrics.csv")
        steps = [1000 * r["first_step_m"] for r in rows if " deg" in str(r["origin"])]
        failing = 0
        for velocity in VELOCITIES:
            for scenario in ("nominal", "block", "pushes", "offsets"):
                for r in rows_of(scenario, tag, velocity):
                    if r["arm"] == "esn":
                        failing += bool(failure_of(r, scenario))
        print(
            f"  {name} ({tag}): {s['following']:.2f} ({' / '.join(f'{g:.2f}' for g in s['gains'])}),"
            f" {s['ringing']:.2f}, {s['ret']:.2f}, {s['rmse']:.2f} | {sum(r['success'] for r in rows)}/{len(rows)},"
            f" {np.mean(steps):.0f} | {failing} failing runs"
        )
    replay_failing = sum(
        bool(failure_of(r, scenario))
        for velocity in VELOCITIES
        for scenario in ("nominal", "block", "pushes", "offsets")
        for r in rows_of(scenario, ARMS[0][1], velocity)
        if r["arm"] == "replay"
    )
    print(f"  the replay: {replay_failing} failing runs")
    same = all(
        np.isclose(
            run_of(s, ARMS[0][1], "replay", v, d)["ringing_m" if s == "block" else "ringing_m_2"],
            run_of(s, ARMS[1][1], "replay", v, d)["ringing_m" if s == "block" else "ringing_m_2"],
        )
        for s in ("block", "pushes")
        for v in VELOCITIES
        for d in DAMPINGS
    )
    print(f"  the replay of ESN #2's runs repeats ESN #1's: {same}")


def print_validation() -> None:
    print("\n== 3.4 Undisturbed: arm, reference velocity, damping ratio: arrival (s), holds, path RMSE (mm), tracking")
    print("error (deg RMS), peak torque (N m)")
    for velocity in VELOCITIES:
        for damping in DAMPINGS:
            for name, tag, arm, _, _ in ARMS:
                r = run_of("nominal", tag, arm, velocity, damping)
                print(
                    f"  {name}, {velocity}, {damping:g}: {r['arrival_time_s']:.2f}, {r['success']},"
                    f" {1000 * r['taught_path_rmse_m']:.2f}, {r['tracking_error_deg']:.2f}, {r['peak_torque_nm']:.2f}"
                )
    print("\n== 3.5 After each disturbance (the block's release; the pushes at 3 s, 7 s, 15 s): arm, reference")
    print("velocity, damping ratio: following | ringing (mm) | back within 5 mm (s; - if not back) | farthest from the")
    print("path (mm) | gap at its end (deg)")
    for velocity in VELOCITIES:
        for damping in DAMPINGS:
            for name, tag, arm, _, _ in ARMS:
                runs = [run_of(scenario, tag, arm, velocity, damping) for scenario, *_ in DISTURBANCES]
                cells = [
                    " / ".join(
                        fmt(r[f"{key}{suffix}"], scale, digits)
                        for r, (_, suffix, _, _) in zip(runs, DISTURBANCES, strict=True)
                    )
                    for key, scale, digits in (
                        ("following_gain", 1, 2),
                        ("ringing_m", 1000, 1),
                        ("return_time_s", 1, 2),
                        ("off_path_m", 1000, 0),
                        ("end_gap_deg", 1, 1),
                    )
                ]
                print(f"  {name}, {velocity}, {damping:g}: {' | '.join(cells)}")
    print(
        "\n== 3.5 Under the block and the pushes: arm, reference velocity, damping ratio: arrival under the block (s),"
    )
    print("holds; peak torque under the block (N m), reference lead at the release; arrival under the pushes (s),")
    print("leaves the goal during the hold, final distance (mm)")
    for velocity in VELOCITIES:
        for damping in DAMPINGS:
            for name, tag, arm, _, _ in ARMS:
                b, p = run_of("block", tag, arm, velocity, damping), run_of("pushes", tag, arm, velocity, damping)
                print(
                    f"  {name}, {velocity}, {damping:g}: {b['arrival_time_s']:.2f}, {b['success']};"
                    f" {b['peak_torque_nm']:.1f}, {100 * b['reference_lead']:.0f}%; {p['arrival_time_s']:.2f},"
                    f" {p['left_goal']}, {1000 * p['final_distance_m']:.1f}"
                )
    print(
        "\n== 3.6 From the offset starts: arm, reference velocity, damping ratio: starts that arrive and hold (of 49),"
    )
    print("median path RMSE (mm), peak torque median / largest (N m), peak reference speed median (deg/s)")
    for velocity in VELOCITIES:
        for damping in DAMPINGS:
            for name, tag, arm, _, _ in ARMS:
                runs = offset_runs(tag, arm, velocity, damping)
                torque = [r["peak_torque_nm"] for r in runs]
                print(
                    f"  {name}, {velocity}, {damping:g}: {sum(r['success'] for r in runs)},"
                    f" {np.median([1000 * r['taught_path_rmse_m'] for r in runs]):.1f},"
                    f" {np.median(torque):.0f} / {max(torque):.0f},"
                    f" {np.median([r['peak_reference_speed_dps'] for r in runs]):.0f}"
                )
    print("\n== In passing, report 007's chosen ESN with the same runs: reference velocity, damping ratio: following")
    print("(block; pushes) | mean ringing (mm) | offset starts' peak torque median / largest (N m)")
    tag = SHORTLIST[3][1]
    for velocity in VELOCITIES:
        for damping in DAMPINGS:
            runs = [run_of(scenario, tag, "esn", velocity, damping) for scenario, *_ in DISTURBANCES]
            gains = [r[f"following_gain{suffix}"] for r, (_, suffix, _, _) in zip(runs, DISTURBANCES, strict=True)]
            ringing = [1000 * r[f"ringing_m{suffix}"] for r, (_, suffix, _, _) in zip(runs, DISTURBANCES, strict=True)]
            torque = [r["peak_torque_nm"] for r in offset_runs(tag, "esn", velocity, damping)]
            print(
                f"  {velocity}, {damping:g}: {' / '.join(f'{g:.2f}' for g in gains)} | {np.mean(ringing):.1f} |"
                f" {np.median(torque):.0f} / {max(torque):.0f}"
            )


def fmt(x: float, scale: float, digits: int) -> str:
    return f"{scale * x:.{digits}f}" if np.isfinite(x) else "-"


if __name__ == "__main__":
    main()
