# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 009.

    uv run python reports/009-noisy-training-for-a-faster-return/make_figures.py
    uv run python reports/009-noisy-training-for-a-faster-return/make_figures.py --logs --animations

From the copies of the runs under ``results/`` (configurations, run records, metrics,
and sweep tables), it draws into ``results/summary``:

- ``grid.png``: the grid, each setting's median ringing against its median return
  over its ten seeds, with the replay's (Section 3.1);
- ``noise.png``: the noise sweep likewise, each noise's settings joined in the order
  of the noise level, at each ridge (Section 3.2);
- ``undisturbed.png``: undisturbed, the path RMSE and the arrival of the chosen ESN
  and the replay, and from the offset starts, their median and largest peak torque,
  against the damping ratio (Sections 3.4 and 3.6);
- ``disturbed.png``: after the block's release and each push, their ringing and
  return against the damping ratio (Section 3.5);

and prints the tables of the report. With ``--logs``, it reads the runs' logs, which
are not kept in Git, from the runs under the storage root, and also draws each arm's
distance from the taught path under the block and the pushes, one panel per tracker
(``block_distances.png``, ``pushes_distances.png``, Section 3.5), prints how far each
arm's reference goes from the taught path under the pushes, with report 007's ESN
from report 008's run in passing, and copies three of the runs' own figures
(FIGURES). With ``--animations``, it animates the block and the pushes with the arms
of PANELS side by side, each rendered by skelarm's player (``block.gif``,
``pushes.gif``).
"""

from __future__ import annotations

import argparse
import csv
import functools
import os
import shutil
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
# Report 007's copies, where its chosen ESN's trained run is.
REPORT_007_RESULTS = REPORT.parent / "007-ringing-with-joint-pd" / "results"
# Report 008's run of report 007's chosen ESN under the pushes with this study's trackers, for a comparison in passing.
REPORT_008_PUSHES = "20261013-174228-pushes_robot_pd_reference_velocity_control10ms_filtered"
DT = 0.01  # the ESNs' period (s)
GOAL = 0.02  # the goal's radius (m): a run ending farther from the target ends away from it

# The grid: its leak rates (one sweep each per scenario) and scenarios.
LEAKS = (0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1)
SCENARIOS = ("nominal", "block", "pushes")
# Report 007's chosen setting (leak rate, spectral radius, input scaling), the score's reference; its ESN is seed 7.
REFERENCE = (0.02, 1.35, 2.0)
# The noise sweep, at report 007's chosen setting: its ridges (one sweep each per scenario), its noise levels (deg),
# and its noises, by the exponent of their power spectrum.
RIDGES = ("1e-2", "4e-2")
LEVELS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
NOISES = {0.0: "white", 1.0: "pink", 2.0: "brown"}
# After each disturbance (the block's release, then the three pushes), a hand not back on the path before the next
# one or the end of the 24 s run counts as back then (s after its end).
RETURN_CAPS = (24.0 - 5.5, 7.0 - 3.5, 15.0 - 7.5, 24.0 - 15.5)
# The shortlist tried in the robot app: its name, its trained run's directory, and its sweep's setting and seed; the
# ESN trained with white noise was chosen.
SHORTLIST = [
    ("report 007's chosen ESN", REPORT_007_RESULTS, "grid_lr0.02_sr1.35_is2_seed7_filtered", ("grid", REFERENCE), 7),
    ("grid, rank 2", RESULTS, "grid_lr0.02_sr1.35_is1.5_seed0_filtered", ("grid", (0.02, 1.35, 1.5)), 0),
    ("grid, rank 3", RESULTS, "grid_lr0.02_sr0.99_is1.5_seed6_filtered", ("grid", (0.02, 0.99, 1.5)), 6),
    ("white noise of 1 deg", RESULTS, "noise_white1deg_ridge4e-2_seed3_filtered", ("noise", ("4e-2", 1.0, 0.0)), 3),
    ("pink noise of 0.5 deg", RESULTS, "noise_pink0.5deg_ridge4e-2_seed8_filtered", ("noise", ("4e-2", 0.5, 1.0)), 8),
]

# The validation: the chosen ESN's configuration tag, the trackers' damping ratios, and their reference velocity (the
# configurations' tag, and its name).
CHOSEN = "white1deg_seed3"
DAMPINGS = (0.1, 0.4, 0.8)
VELOCITIES = {"zero": "zero reference velocity", "reference": "reference velocity"}
SEARCH_TRACKER = ("reference", 0.4)  # the search's tracker: given the reference velocity, damping ratio 0.4
ARMS = {"esn": "chosen ESN", "replay": "replay"}
# The disturbances, each with its run, the suffix of its columns in metrics.csv, its name, and when a hand not back
# counts as back (RETURN_CAPS).
DISTURBANCES = [
    ("block", "", "the block's release", RETURN_CAPS[0]),
    ("pushes", "_1", "the push forward at 3 s", RETURN_CAPS[1]),
    ("pushes", "_2", "the push across at 7 s", RETURN_CAPS[2]),
    ("pushes", "_3", "the push across at 15 s", RETURN_CAPS[3]),
]
BACK_WITHIN = 5.0  # mm: an arm is back on the taught path once it stays within this of it

ESN_COLOR = "#2a78d6"
REPLAY_COLOR = "#eb6834"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
DISTURBANCE_COLOR = "#f0efec"
COLORS = {"esn": ESN_COLOR, "replay": REPLAY_COLOR}
LINESTYLES = {"zero": "-", "reference": "--"}
NOISE_COLORS = {0.0: "#6b6b6b", 1.0: "#d6457a", 2.0: "#8b5a2b"}  # white noise drawn in gray
LABEL_OFFSETS = {0.0: (-8, -13), 1.0: (-6, 6), 2.0: (5, -11)}  # where each noise's levels are written (points)

# The animations, with the search's tracker: per panel, the label (the arm's reference, then its tracker), the
# tracker setting, the arm, and the label's color.
PANELS = [
    ("chosen ESN (white noise)\ntracker: joint PD, damping ratio 0.4", "pd_w20_z0.4", "esn", ESN_COLOR),
    ("replay of the take by time\ntracker: joint PD, damping ratio 0.4", "pd_w20_z0.4", "replay", REPLAY_COLOR),
]
# The runs' own figures copied into the summary, by the name there: the scenario, the reference velocity, and the
# figure in the run.
FIGURES = {
    "block_joints.png": ("block", "reference", "joints.png"),
    "pushes_joints.png": ("pushes", "reference", "joints.png"),
    "offset_torques.png": ("offsets", "reference", "torques_offset_+15_+15.png"),
}
ANIMATION_SPAN = (-0.2, 19.0)
ANIMATION_FPS = 10.0
ANIMATION_HOLD_MS = 1500  # the last frame stays this long before the GIF loops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", action="store_true", help="also analyze the run logs in the storage root")
    parser.add_argument("--animations", action="store_true", help="also export a GIF from the logs in the storage root")
    args = parser.parse_args()

    SUMMARY.mkdir(parents=True, exist_ok=True)
    figures = {
        "grid.png": plot_grid(),
        "noise.png": plot_noise(),
        "undisturbed.png": plot_undisturbed(),
        "disturbed.png": plot_disturbed(),
    }
    if args.logs:
        figures |= {f"{scenario}_distances.png": plot_distances(scenario) for scenario in ("block", "pushes")}
        for name, (scenario, velocity, figure) in FIGURES.items():
            shutil.copyfile(
                resolve_run_path(f"results/{run_name(stem_of(scenario, velocity))}") / figure, SUMMARY / name
            )
    for name, fig in figures.items():
        fig.savefig(SUMMARY / name, dpi=150)
    print_tables()
    if args.logs:
        print_reference_offsets()
    if args.animations:
        tracker = "joint PD at 0.4, reference velocity, the torque every 10 ms"
        export_animation("block", stem_of("block", "reference"), f"the block, from 4.5 s to 5.5 s; {tracker}")
        export_animation("pushes", stem_of("pushes", "reference"), f"three pushes, at 3 s, 7 s, and 15 s; {tracker}")
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


def run_name(stem: str, results: Path = RESULTS) -> str:
    """The name of the one run of a configuration, by its stem."""
    matches = sorted(p.name for p in results.glob(f"*-{stem}") if p.is_dir())
    if len(matches) != 1:
        msg = f"expected one run of {stem}, found {matches}"
        raise FileNotFoundError(msg)
    return matches[0]


def stem_of(scenario: str, velocity: str) -> str:
    """A validation configuration's stem, by its scenario and its reference velocity."""
    return f"{scenario}_robot_{CHOSEN}_{velocity}_velocity_filtered"


def rows_of(scenario: str, velocity: str) -> list[dict[str, Any]]:
    """The metrics of a validation run with the trackers given ``velocity``."""
    return read_csv(RESULTS / run_name(stem_of(scenario, velocity)) / "metrics.csv")


def run_of(scenario: str, velocity: str, damping: float, arm: str, start: int = 0) -> dict[str, Any]:
    """One arm's run from one start, with one tracker."""
    return next(
        r for r in rows_of(scenario, velocity) if r["damping"] == damping and r["arm"] == arm and r["start"] == start
    )


def offset_runs(velocity: str, damping: float, arm: str) -> list[dict[str, Any]]:
    """One arm's runs from the 49 starts, with one tracker."""
    return [r for r in rows_of("offsets", velocity) if r["damping"] == damping and r["arm"] == arm]


def returned(run: dict[str, Any], suffix: str, cap: float) -> float:
    """When the hand is back on the path after a disturbance (s), or ``cap`` if it is not back by the window's end."""
    back = run[f"return_time_s{suffix}"]
    return float(back) if np.isfinite(back) else cap


@functools.cache
def taught_hand() -> NDArray[np.float64]:
    """The hand path of the take filtered at 2 Hz, what the ESN was trained on, every 10 ms."""
    skeleton = Skeleton.from_toml(TAKE_RUN / "config.toml")
    q = resample_joint_angles(StateLog.load(TAKE_RUN / "demo_00_filtered.sklog.npz"), DT)[1]
    return endpoint_positions(skeleton, q)


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


# ---------------------------------------------------------------------------- the search


def failure_of(run: dict[str, Any]) -> str:
    """How a run of a sweep fails, or "" if it does not."""
    if not np.isfinite(run["arrival_time_s"]):
        return "never arrives"
    if run["final_distance_m"] > GOAL:
        return "ends away"
    if run["left_goal"]:
        return "leaves the goal during the hold"
    return "hold cut short" if run["failed"] else ""


def seed_measures(n: dict[str, Any], b: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    """One seed's measures from its runs undisturbed (``n``), under the block (``b``), and under the pushes (``p``).

    ``failing``, the scenarios where its run fails, so that it does not qualify;
    ``failures``, how each fails (see :func:`failure_of`); ``rmse``, the undisturbed
    path RMSE (mm); ``ringings``, ``offs``, and ``returns``, after each disturbance
    (the block's release, then the three pushes), the ringing (mm), the farthest from
    the path (mm), and the return onto it (s, a hand not back counting as back at
    RETURN_CAPS); ``ringing`` and ``ret``, their means over the four disturbances,
    as the rule takes them; ``lead``, the block's reference lead at the release (% of
    the path); and the peak torque under the block and over the run under the pushes
    (N m).
    """
    runs = dict(zip(SCENARIOS, (n, b, p), strict=True))
    ringings = [1000 * b["ringing_m"]] + [1000 * p[f"ringing_m_{k}"] for k in (1, 2, 3)]
    returns = [b["return_time_s"]] + [p[f"return_time_s_{k}"] for k in (1, 2, 3)]
    returns = [r if np.isfinite(r) else cap for r, cap in zip(returns, RETURN_CAPS, strict=True)]
    return {
        "seed": int(n["seed"]),
        "failing": [scenario for scenario, r in runs.items() if failure_of(r)],
        "failures": {scenario: failure_of(r) for scenario, r in runs.items() if failure_of(r)},
        "rmse": 1000 * n["taught_path_rmse_m"],
        "ringings": ringings,
        "ringing": float(np.mean(ringings)),
        "offs": [1000 * b["off_path_m"]] + [1000 * p[f"off_path_m_{k}"] for k in (1, 2, 3)],
        "returns": returns,
        "ret": float(np.mean(returns)),
        "lead": 100 * b["reference_lead"],
        "block_torque": b["peak_torque_nm"],
        "pushes_torque": p["peak_torque_nm"],
    }


@functools.cache
def grid_seeds() -> dict[tuple[float, float, float], list[dict[str, Any]]]:
    """Each seed of every setting of the grid, by (leak rate, spectral radius, input scaling): :func:`seed_measures`."""
    out: dict[tuple[float, float, float], list[dict[str, Any]]] = {}
    for leak in LEAKS:
        runs = {
            scenario: {
                (r["spectral_radius"], r["input_scaling"], r["seed"]): r
                for r in read_csv(RESULTS / run_name(f"sweep_robot_damped_lr{leak:g}_{scenario}_filtered") / "runs.csv")
            }
            for scenario in SCENARIOS
        }
        for radius, scaling, seed in sorted(runs["nominal"]):
            seeds = out.setdefault((leak, radius, scaling), [])
            seeds.append(seed_measures(*(runs[scenario][radius, scaling, seed] for scenario in SCENARIOS)))
    return out


@functools.cache
def noise_seeds() -> dict[tuple[str, float, float], list[dict[str, Any]]]:
    """Each seed of every setting of the noise sweep, by (ridge, noise level, exponent): :func:`seed_measures`."""
    out: dict[tuple[str, float, float], list[dict[str, Any]]] = {}
    for ridge in RIDGES:
        runs = {
            scenario: {
                (r["teacher_noise_deg"], r["teacher_noise_exponent"], r["seed"]): r
                for r in read_csv(
                    RESULTS / run_name(f"sweep_robot_damped_noise_ridge{ridge}_{scenario}_filtered") / "runs.csv"
                )
            }
            for scenario in SCENARIOS
        }
        for level, exponent, seed in sorted(runs["nominal"]):
            seeds = out.setdefault((ridge, level, exponent), [])
            seeds.append(seed_measures(*(runs[scenario][level, exponent, seed] for scenario in SCENARIOS)))
    return out


def qualifies(seeds: list[dict[str, Any]]) -> bool:
    """A setting whose seeds all arrive, hold, and end in the goal in the three scenarios (the rule)."""
    return not any(s["failing"] for s in seeds)


def median(seeds: list[dict[str, Any]], key: str, i: int | None = None) -> float:
    """The median over seeds of a measure, or of its ``i``-th entry."""
    return float(np.median([s[key] if i is None else s[key][i] for s in seeds]))


def score(ringing: float, ret: float) -> float:
    """The rule's score of a ringing (mm) and a return (s): the mean of their ratios to report 007's setting's."""
    reference = grid_seeds()[REFERENCE]
    return 0.5 * (ringing / median(reference, "ringing") + ret / median(reference, "ret"))


def setting_score(seeds: list[dict[str, Any]]) -> float:
    """A setting's score: that of the medians over its seeds of the ringing and the return."""
    return score(median(seeds, "ringing"), median(seeds, "ret"))


def typical_seed(seeds: list[dict[str, Any]]) -> int:
    """The seed nearest the medians of the RMSE, the ringing, and the return, each by its interquartile range."""
    x = np.array([[s["rmse"], s["ringing"], s["ret"]] for s in seeds])
    iqr = np.subtract(*np.percentile(x, [75, 25], axis=0))
    iqr[iqr == 0] = 1.0
    return int(seeds[int(np.argmin(np.sum(np.abs(x - np.median(x, axis=0)) / iqr, axis=1)))]["seed"])


def replay_measures() -> tuple[float, float]:
    """The replay's ringing (mm) and return (s), each its mean over the four disturbances, with the search's tracker."""
    velocity, damping = SEARCH_TRACKER
    runs = [run_of(scenario, velocity, damping, "replay") for scenario, *_ in DISTURBANCES]
    ringing = [1000 * r[f"ringing_m{suffix}"] for r, (_, suffix, _, _) in zip(runs, DISTURBANCES, strict=True)]
    back = [returned(r, suffix, cap) for r, (_, suffix, _, cap) in zip(runs, DISTURBANCES, strict=True)]
    return float(np.mean(ringing)), float(np.mean(back))


def tradeoff_axes(ax: Axes, xmax: float, ylim: tuple[float, float]) -> None:
    """Report 007's setting, the replay, and the lines of equal score on a plot of the ringing against the return."""
    reference = grid_seeds()[REFERENCE]
    ringing, ret = median(reference, "ringing"), median(reference, "ret")
    for level in (0.6, 0.8, 1.0, 1.2):
        x = np.linspace(0.0, 2 * level * ret, 50)
        ax.plot(x, (2 * level - x / ret) * ringing, color=TEXT_COLOR, linewidth=0.6, linestyle=":", zorder=1)
        # Labeled where the line enters the axes, at the left or at the top.
        start = (
            (0.0, 2 * level * ringing)
            if 2 * level * ringing <= ylim[1]
            else ((2 * level - ylim[1] / ringing) * ret, ylim[1])
        )
        if ylim[0] <= start[1] and start[0] <= xmax:
            ax.annotate(
                f"score {level:g}",
                start,
                xytext=(4, -2),
                textcoords="offset points",
                fontsize=7,
                color=TEXT_COLOR,
                va="top",
            )
    ax.scatter([ret], [ringing], marker="*", s=220, color="#0b0b0b", zorder=5)
    replay_ringing, replay_return = replay_measures()
    ax.scatter([replay_return], [replay_ringing], marker="D", s=60, color=REPLAY_COLOR, zorder=5)
    ax.set_xlim(0, xmax)
    ax.set_ylim(*ylim)
    ax.set_xlabel("return onto the path (s): median over the seeds of the mean after the four", color=TEXT_COLOR)


def tradeoff_limits(settings: list[list[dict[str, Any]]], ymin: float) -> tuple[float, tuple[float, float]]:
    """The axes' limits for settings and the replay: the largest return (s), and the ringing (mm) from ``ymin``."""
    top = max(max(median(s, "ringing") for s in settings), replay_measures()[0])
    return 1.06 * max(median(s, "ret") for s in settings), (ymin, 1.08 * top)


def plot_grid() -> Figure:
    """Every setting of the grid: the median over its seeds of the ringing against that of the return.

    Filled: the settings that qualify; hollow: those with a failing seed. The black
    star is report 007's chosen setting, the score's reference; the orange diamond
    the replay with the same tracker.
    """
    settings = grid_seeds()
    fig = Figure(figsize=(9.0, 6.0), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The grid, joint PD at damping ratio 0.4, reference velocity, the torque every 10 ms:"
        "\nringing against return after the block and the pushes, each setting over its ten seeds",
        color="#0b0b0b",
    )
    ax = fig.subplots()
    style(ax)
    ramp = colormaps["Blues"]
    colors = {leak: ramp(0.4 + 0.6 * i / (len(LEAKS) - 1)) for i, leak in enumerate(LEAKS)}
    for k, seeds in settings.items():
        color = colors[k[0]]
        ax.scatter(
            median(seeds, "ret"),
            median(seeds, "ringing"),
            s=36,
            color=color if qualifies(seeds) else SURFACE_COLOR,
            edgecolor=color,
            linewidth=1.3,
            zorder=3,
        )
    tradeoff_axes(ax, *tradeoff_limits(list(settings.values()), 0.0))
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
        Line2D([], [], color=REPLAY_COLOR, marker="D", linestyle="none", label="the replay"),
    ]
    ax.legend(handles=handles, frameon=False, labelcolor=TEXT_COLOR, fontsize=8, ncol=2, loc="upper right")
    return fig


def plot_noise() -> Figure:
    """Every setting of the noise sweep, as in :func:`plot_grid`, one panel per ridge.

    Each noise's settings are joined in the order of the noise level, from the
    noiseless setting, which they share; filled: the settings that qualify.
    """
    settings = noise_seeds()
    fig = Figure(figsize=(13.0, 5.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The noise sweep, at report 007's setting, the take fed in four times with noise on the input:"
        " ringing against return, each setting over its ten seeds",
        color="#0b0b0b",
    )
    axes = fig.subplots(1, len(RIDGES), sharey=True)
    limits = tradeoff_limits(list(settings.values()), 4.0)
    for ax, ridge in zip(axes, RIDGES, strict=True):
        style(ax)
        plain = settings[ridge, 0.0, 0.0]
        for exponent, color in NOISE_COLORS.items():
            group = [settings[ridge, level, exponent] for level in LEVELS]
            x, y = [median(s, "ret") for s in group], [median(s, "ringing") for s in group]
            ax.plot(x, y, color=color, linewidth=1.3, zorder=2)
            for level, seeds, xi, yi in zip(LEVELS[1:], group[1:], x[1:], y[1:], strict=True):
                ax.scatter(
                    xi,
                    yi,
                    s=40,
                    color=color if qualifies(seeds) else SURFACE_COLOR,
                    edgecolor=color,
                    linewidth=1.4,
                    zorder=3,
                )
                ax.annotate(
                    f"{level:g}°",
                    (xi, yi),
                    xytext=LABEL_OFFSETS[exponent],
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                )
        ax.scatter(
            median(plain, "ret"),
            median(plain, "ringing"),
            marker="s",
            s=44,
            zorder=4,
            color=TEXT_COLOR if qualifies(plain) else SURFACE_COLOR,
            edgecolor=TEXT_COLOR,
            linewidth=1.4,
        )
        tradeoff_axes(ax, *limits)
        copies = "as report 007's 1e-2 with one copy" if ridge == "4e-2" else "a quarter of that"
        ax.set_title(f"ridge {ridge} with four copies, {copies}", color="#0b0b0b", fontsize=10)
    axes[0].set_ylabel("ringing (mm): median over the seeds of the mean after the four", color=TEXT_COLOR)
    handles = [Line2D([], [], color=c, marker="o", label=f"{NOISES[e]} noise") for e, c in NOISE_COLORS.items()]
    handles += [
        Line2D([], [], color=TEXT_COLOR, marker="s", linestyle="none", label="no noise"),
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
        Line2D([], [], color=REPLAY_COLOR, marker="D", linestyle="none", label="the replay"),
    ]
    axes[1].legend(handles=handles, frameon=False, labelcolor=TEXT_COLOR, fontsize=8, loc="upper right")
    return fig


# ---------------------------------------------------------------------------- the validation


def plot_disturbed() -> Figure:
    """After each disturbance, the ringing and the return of both arms against the damping ratio.

    Solid lines: the trackers given a zero reference velocity; dashed: given the
    reference's velocity. A hand not back on the path before the next disturbance
    is drawn at the window's end, marked with a cross.
    """
    fig = Figure(figsize=(14, 6.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "After each disturbance, joint PD, the torque every 10 ms: the chosen ESN (blue) and the replay (orange),"
        "\nagainst the damping ratio; solid: zero reference velocity, dashed: reference velocity",
        color="#0b0b0b",
    )
    axes = fig.subplots(2, len(DISTURBANCES), sharex=True)
    for col, (scenario, suffix, name, cap) in enumerate(DISTURBANCES):
        for row, measure in enumerate(("ringing", "return")):
            ax = axes[row, col]
            style(ax)
            for velocity, linestyle in LINESTYLES.items():
                for arm, color in COLORS.items():
                    runs = [run_of(scenario, velocity, damping, arm) for damping in DAMPINGS]
                    if measure == "ringing":
                        y = [1000 * r[f"ringing_m{suffix}"] for r in runs]
                    else:
                        y = [returned(r, suffix, cap) for r in runs]
                    ax.plot(DAMPINGS, y, color=color, linestyle=linestyle, marker="o", markersize=4, linewidth=1.4)
                    if measure == "return":
                        missing = [
                            d
                            for d, r in zip(DAMPINGS, runs, strict=True)
                            if not np.isfinite(r[f"return_time_s{suffix}"])
                        ]
                        ax.scatter(missing, [cap] * len(missing), marker="x", s=60, color=color, zorder=3)
            ax.set_xticks(DAMPINGS)
            ax.set_ylim(0, None)
            if row == 0:
                ax.set_title(name, color="#0b0b0b", fontsize=10)
            else:
                ax.set_xlabel("damping ratio", color=TEXT_COLOR)
            if col == 0:
                ax.set_ylabel(
                    "ringing over the next 2 s (mm)" if measure == "ringing" else "back within 5 mm\nof the path (s)",
                    color=TEXT_COLOR,
                )
    return fig


def plot_undisturbed() -> Figure:
    """Undisturbed, the path RMSE and the arrival; from the offset starts, the peak torques; against the damping."""
    fig = Figure(figsize=(14, 4.0), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "Joint PD, the torque every 10 ms: the chosen ESN (blue) and the replay (orange), against the damping ratio;"
        "\nsolid: zero reference velocity, dashed: reference velocity",
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
        for arm, color in COLORS.items():
            nominal = [run_of("nominal", velocity, damping, arm) for damping in DAMPINGS]
            style_kwargs = {"color": color, "linestyle": linestyle, "marker": "o", "markersize": 4, "linewidth": 1.4}
            axes[0].plot(DAMPINGS, [1000 * r["taught_path_rmse_m"] for r in nominal], **style_kwargs)
            axes[1].plot(DAMPINGS, [r["arrival_time_s"] for r in nominal], **style_kwargs)
            peaks = [[r["peak_torque_nm"] for r in offset_runs(velocity, d, arm)] for d in DAMPINGS]
            axes[2].plot(DAMPINGS, [np.median(p) for p in peaks], **style_kwargs)
            axes[2].scatter(DAMPINGS, [max(p) for p in peaks], color=color, marker="^", s=24)
    for ax, ticks in ((axes[0], (0.2, 0.5, 1, 2, 5)), (axes[2], (10, 20, 50, 100, 200))):
        ax.set_yscale("log")
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    return fig


# ---------------------------------------------------------------------------- figures from the logs


def path_distances(stem: str, setting: str, arm: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The times (every 10 ms) and the hand's distance (mm) from the taught path in a run's first start."""
    log = StateLog.load(resolve_run_path(f"results/{run_name(stem)}") / setting / f"{arm}_00.sklog.npz")
    task = log.times >= 0
    skeleton = Skeleton.from_toml(TAKE_RUN / "config.toml")
    hand = endpoint_positions(skeleton, log.channel("q").reshape(len(log.times), -1)[task])
    every = round(DT / float(np.median(np.diff(log.times))))
    return log.times[task][::every], 1000 * distances_to_path(hand[::every], taught_hand())


def print_reference_offsets() -> None:
    """How far each arm's reference goes from the taught path under the pushes, with the search's tracker.

    For each push, the farthest the reference's hand goes from the taught hand path
    while the push acts and over the next 1 s, and the farthest the arm's hand goes
    while it acts. Report 007's ESN, from report 008's run with the same tracker, is
    given in passing.
    """
    skeleton = Skeleton.from_toml(TAKE_RUN / "config.toml")
    velocity, damping = SEARCH_TRACKER
    setting = f"pd_w20_z{damping:g}"
    arms = [
        (ARMS["esn"], resolve_run_path(f"results/{run_name(stem_of('pushes', velocity))}"), "esn"),
        (ARMS["replay"], resolve_run_path(f"results/{run_name(stem_of('pushes', velocity))}"), "replay"),
        ("report 007's ESN", resolve_run_path(f"results/{REPORT_008_PUSHES}"), "esn"),
    ]
    with (RESULTS / run_name(stem_of("pushes", velocity)) / "config.toml").open("rb") as f:
        spans = disturbance_spans(tomllib.load(f)["disturbance"])
    print(
        "\n== 3.5 Under the pushes, with the search's tracker: for each push, the reference's farthest from the taught"
    )
    print("path while it acts and over the next 1 s, and the arm's while it acts (mm)")
    for name, run, arm in arms:
        log = StateLog.load(run / setting / f"{arm}_00.sklog.npz")
        q = log.channel("q").reshape(len(log.times), -1)
        reference = log.channel("q_ref").reshape(len(log.times), -1)
        cells = []
        for onset, end in spans:
            acting = (log.times >= onset) & (log.times < end)
            after = (log.times >= end) & (log.times < end + 1.0)
            off = [
                1000 * distances_to_path(endpoint_positions(skeleton, x[span]), taught_hand()).max()
                for x, span in ((reference, acting), (reference, after), (q, acting))
            ]
            cells.append(" / ".join(f"{v:.1f}" for v in off))
        print(f"  {name}: {' | '.join(cells)}")


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
        stem = stem_of(scenario, velocity)
        with (RESULTS / run_name(stem) / "config.toml").open("rb") as f:
            spans = disturbance_spans(tomllib.load(f)["disturbance"])
        for col, damping in enumerate(DAMPINGS):
            ax = axes[row, col]
            style(ax)
            for onset, end in spans:
                ax.axvspan(onset, end, color=DISTURBANCE_COLOR, zorder=0)
            for arm, color in COLORS.items():
                times, distance = path_distances(stem, f"pd_w20_z{damping:g}", arm)
                ax.plot(times, distance, color=color, linewidth=1.2, label=ARMS[arm])
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


def export_animation(name: str, stem: str, title: str) -> None:
    """Animate a run's demonstrated start into ``<name>.gif``: the arms of PANELS side by side, rendered by skelarm.

    The player exports each arm's run as a GIF; its frames, which the GIF merges
    where the arm rests, are spread back over time, cropped to where the arms move,
    labeled, and tiled into one GIF with the task time and the disturbance acting.
    """
    import tempfile

    from PIL import Image, ImageDraw, ImageFont, ImageSequence

    with (RESULTS / run_name(stem) / "config.toml").open("rb") as f:
        disturbance = tomllib.load(f).get("disturbance")
    tables = [] if disturbance is None else [disturbance] if isinstance(disturbance, dict) else disturbance
    spans = disturbance_spans(disturbance)
    frames, starts = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (_, setting, arm, _) in enumerate(PANELS):
            path = resolve_run_path(f"results/{run_name(stem)}") / setting / f"{arm}_00.sklog.npz"
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


def print_tables() -> None:
    print_grid()
    print_noise()
    print_shortlist()
    print_validation()


def describe(seeds: list[dict[str, Any]]) -> str:
    """A setting's score and medians over its seeds, as the tables print them."""
    each = [
        " / ".join(f"{median(seeds, key, i):.{digits}f}" for i in range(4))
        for key, digits in (("ringings", 1), ("offs", 0), ("returns", 2))
    ]
    return (
        f"{setting_score(seeds):.3f} | {median(seeds, 'ringing'):.2f} | {median(seeds, 'ret'):.2f} |"
        f" {median(seeds, 'rmse'):.2f} | {' | '.join(each)} | {median(seeds, 'lead'):.0f}% |"
        f" {median(seeds, 'block_torque'):.1f} | {median(seeds, 'pushes_torque'):.1f}"
    )


HEADER = (
    "score | ringing (mm) | return (s) | undisturbed path RMSE (mm) | after the block's release and pushes 1, 2, 3:"
    " ringing (mm) | farthest from the path (mm) | back on it (s) | the block's reference lead | peak torque under"
    " the block, and over the run under the pushes (N m)"
)


def print_grid() -> None:
    settings = grid_seeds()
    print("== 3.1 The grid: per leak rate, the settings that qualify (of 15), the failing runs of 150 per scenario,")
    print("and how the runs under the pushes fail")
    for leak in LEAKS:
        group = {k: v for k, v in settings.items() if k[0] == leak}
        fails = {scenario: sum(scenario in s["failing"] for v in group.values() for s in v) for scenario in SCENARIOS}
        kinds: dict[str, int] = {}
        for seeds in group.values():
            for s in seeds:
                if "pushes" in s["failures"]:
                    kinds[s["failures"]["pushes"]] = kinds.get(s["failures"]["pushes"], 0) + 1
        print(f"  leak rate {leak:g}: {sum(qualifies(v) for v in group.values())}; {fails}; {kinds}")

    print(f"\n== 3.1 The qualifying settings, ranked by the score: medians over the seeds of the {HEADER}")
    ranked = sorted((k for k, v in settings.items() if qualifies(v)), key=lambda k: setting_score(settings[k]))
    for rank, k in enumerate(ranked, start=1):
        mark = "  <- report 007's" if k == REFERENCE else ""
        name = f"leak rate {k[0]:g}, spectral radius {k[1]:g}, input scaling {k[2]:g}"
        print(f"  {rank}. {name}: {describe(settings[k])}{mark}")

    fastest = sorted(settings, key=lambda k: median(settings[k], "ret"))
    print("\n== 3.1 The fastest median returns of all settings, qualifying or not: return (s), ringing (mm),")
    print("failing seeds")
    for k in fastest[:5]:
        seeds = settings[k]
        print(
            f"  {k}: {median(seeds, 'ret'):.2f}, {median(seeds, 'ringing'):.2f},"
            f" {sum(bool(s['failing']) for s in seeds)}"
        )
    returns = [median(v, "ret") for v in settings.values()]
    print(f"  over all settings: {min(returns):.2f} to {max(returns):.2f} s")
    reference = settings[REFERENCE]
    print(f"  report 007's setting, its seeds' returns: {', '.join(f'{s["ret"]:.2f}' for s in reference)}")
    replay_ringing, replay_return = replay_measures()
    print(
        f"  the replay with the same tracker: ringing {replay_ringing:.2f} mm, return {replay_return:.2f} s,"
        f" score {score(replay_ringing, replay_return):.3f}"
    )


def print_noise() -> None:
    settings = noise_seeds()
    print(f"\n== 3.2 The noise sweep, every setting: Q if it qualifies, then medians over the seeds of the {HEADER};")
    print("and the failing seeds, how they fail")
    for ridge in RIDGES:
        for exponent, noise in NOISES.items():
            for level in LEVELS:
                if level == 0 and exponent != 0:
                    continue  # the noiseless setting, the same with every exponent
                seeds = settings[ridge, level, exponent]
                name = "no noise" if level == 0 else f"{level:g} deg {noise}"
                failing = ", ".join(f"{s['seed']}: {s['failures']}" for s in seeds if s["failing"])
                print(
                    f"  {'Q' if qualifies(seeds) else ' '} ridge {ridge}, {name}: {describe(seeds)}"
                    f"{f'  [{failing}]' if failing else ''}"
                )
    same = all(
        np.isclose(a["ringing"], b["ringing"]) and np.isclose(a["ret"], b["ret"]) and np.isclose(a["rmse"], b["rmse"])
        for a, b in zip(settings["4e-2", 0.0, 0.0], grid_seeds()[REFERENCE], strict=True)
    )
    print(f"  the noiseless setting at ridge 4e-2 repeats report 007's in the grid: {same}")
    print("\n== 3.2 For each noise, the qualifying setting with noise that scores the lowest, and its typical seed")
    for exponent, noise in NOISES.items():
        candidates = [k for k, v in settings.items() if k[2] == exponent and k[1] > 0 and qualifies(v)]
        if not candidates:
            print(f"  {noise}: none qualifies")
            continue
        best = min(candidates, key=lambda k: setting_score(settings[k]))
        print(
            f"  {noise}: ridge {best[0]}, {best[1]:g} deg, score {setting_score(settings[best]):.3f},"
            f" typical seed {typical_seed(settings[best])}"
        )
    plain, white = settings["4e-2", 0.0, 0.0], settings["4e-2", 1.0, 0.0]
    print("\n== 3.2 Seed by seed, 1 deg of white noise at ridge 4e-2 against none: return (s), ringing (mm)")
    for a, b in zip(plain, white, strict=True):
        print(f"  seed {a['seed']}: {a['ret']:.2f} -> {b['ret']:.2f}, {a['ringing']:.1f} -> {b['ringing']:.1f}")


def print_shortlist() -> None:
    print("\n== 3.3 The shortlist: its setting's score; the ESN's ringing and return (means after the four),")
    print("undisturbed path RMSE, and peak torque under the block; on its own, the starts that arrive and hold,")
    print("and its first step from the 48 offset starts, median and largest (mm)")
    for name, results, stem, (sweep, key), seed in SHORTLIST:
        settings: dict[Any, list[dict[str, Any]]] = grid_seeds() if sweep == "grid" else noise_seeds()
        seeds = settings[key]
        s = next(x for x in seeds if x["seed"] == seed)
        rows = read_csv(results / run_name(stem, results) / "metrics.csv")
        steps = [1000 * r["first_step_m"] for r in rows if " deg" in r["origin"]]
        print(
            f"  {name} ({stem}): {setting_score(seeds):.3f} | {s['ringing']:.2f} mm, {s['ret']:.2f} s,"
            f" {s['rmse']:.2f} mm, {s['block_torque']:.1f} N m | {sum(r['success'] for r in rows)}/{len(rows)},"
            f" {np.median(steps):.0f}, {max(steps):.0f}"
        )


def print_validation() -> None:
    print("\n== 3.4 Undisturbed: arm, reference velocity, damping ratio: arrival (s), holds, path RMSE (mm), tracking")
    print("error (deg RMS), peak torque (N m)")
    for velocity in VELOCITIES:
        for damping in DAMPINGS:
            for arm in ARMS:
                r = run_of("nominal", velocity, damping, arm)
                print(
                    f"  {ARMS[arm]}, {velocity}, {damping:g}: {r['arrival_time_s']:.2f}, {r['success']},"
                    f" {1000 * r['taught_path_rmse_m']:.2f}, {r['tracking_error_deg']:.2f}, {r['peak_torque_nm']:.2f}"
                )

    print("\n== 3.5 After each disturbance (the block's release; the pushes at 3 s, 7 s, 15 s): arm, reference")
    print("velocity, damping ratio: gap (deg) | ringing (mm) | farthest from the path (mm) | back within 5 mm (s);")
    print("then the arrival (s) and whether it holds, under the block and under the pushes, the block's peak torque")
    print("(N m) and reference lead (%), and the score of the means after the four")
    for velocity in VELOCITIES:
        for damping in DAMPINGS:
            for arm in ARMS:
                cells, ringing, back = [], [], []
                for scenario, suffix, _, cap in DISTURBANCES:
                    r = run_of(scenario, velocity, damping, arm)
                    t = r[f"return_time_s{suffix}"]
                    cells.append(
                        f"{r[f'end_gap_deg{suffix}']:.1f} | {1000 * r[f'ringing_m{suffix}']:.1f} |"
                        f" {1000 * r[f'off_path_m{suffix}']:.0f} | {f'{t:.2f}' if np.isfinite(t) else 'not back'}"
                    )
                    ringing.append(1000 * r[f"ringing_m{suffix}"])
                    back.append(returned(r, suffix, cap))
                block, pushes = run_of("block", velocity, damping, arm), run_of("pushes", velocity, damping, arm)
                print(
                    f"  {ARMS[arm]}, {velocity}, {damping:g}: {'  ||  '.join(cells)}"
                    f"  ||  block {block['arrival_time_s']:.2f} {block['success']},"
                    f" pushes {pushes['arrival_time_s']:.2f} {pushes['success']}"
                    f" (left the goal: {pushes['left_goal']}), block {block['peak_torque_nm']:.1f} N m,"
                    f" lead {100 * block['reference_lead']:.0f}%, score {score(np.mean(ringing), np.mean(back)):.3f}"
                )

    print("\n== 3.6 Offset starts: arm, reference velocity, damping ratio: starts that hold of 49, median path RMSE")
    print("(mm), median and largest peak torque (N m), the start of the largest, median first step (mm)")
    for velocity in VELOCITIES:
        for damping in DAMPINGS:
            for arm in ARMS:
                group = offset_runs(velocity, damping, arm)
                largest = max(group, key=lambda r: r["peak_torque_nm"])
                print(
                    f"  {ARMS[arm]}, {velocity}, {damping:g}: {sum(r['success'] for r in group)},"
                    f" {1000 * np.median([r['taught_path_rmse_m'] for r in group]):.1f},"
                    f" {np.median([r['peak_torque_nm'] for r in group]):.1f}, {largest['peak_torque_nm']:.0f}"
                    f" ({largest['origin'].split(' ', 1)[-1]}),"
                    f" {1000 * np.median([r['first_step_m'] for r in group]):.1f}"
                )


if __name__ == "__main__":
    main()
