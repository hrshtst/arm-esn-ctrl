# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 024.

    uv run python reports/024-many-starts-with-pushes-in-the-score/make_figures.py

It reads the copies of the runs under ``results/`` (their configurations, run records,
and CSV files), report 023's copies of its runs and of the takes, and the robot runs'
logs under the storage root, which are not kept in Git. It prints the tables of the
report and draws into ``results/summary``:

- ``search_coarse.png``: the coarse search's settings by their runs that pass, undisturbed
  and pushed (Section 3.1);
- ``search_following.png``: the settings' pushed runs that pass against how much their
  references follow the pushed arm, by the leak rate (Section 3.1);
- ``rescoring.png``: the fine search's 20 best settings scored again with the untaught
  starts (Section 3.1);
- ``paths.png``: the hand's paths, undisturbed, from the takes' starts and from the
  unseen ones (Sections 3.2 and 3.3);
- ``unseen.png``: the detour ratio, the reference's fastest joint speed, and the largest
  torque from each unseen start (Section 3.3);
- ``joints_esn1.png``, ``torques_esn1.png``, ``joints_esn2.png``, ``torques_esn2.png``:
  each ESN's joint angles or torques over time against the replay's, undisturbed and
  under the pushes of each size (Sections 3.2 to 3.4);
- ``pushes_paths.png``: the hand's paths under the pushes (Section 3.4);
- ``pushes_distance.png``: the hand's distance to the target over time under the pushes
  (Section 3.4);
- ``pushes_long.png``: the same over 48 s, at the pushes' original forces (Section 3.4);
- ``one_take.png``: the hand's paths of the nine-take and the one-take ESNs from the 44
  starts take 00 does not start from (Section 3.5);
- ``takes.png``: the runs that hold, the detour ratio, the reference's speed, and the
  torque against the number of takes (Section 3.6).
"""

from __future__ import annotations

import csv
import functools
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from numpy.typing import NDArray
from skelarm import Skeleton, StateLog

from arm_esn_ctrl.demonstrations import endpoint_positions, resample_joint_angles
from arm_esn_ctrl.storage import resolve_run_path

REPORT = Path(__file__).resolve().parent
REPO_ROOT = REPORT.parents[1]
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"
REPORT_023 = REPORT.parent / "023-one-target-from-many-starts"  # its copies of the takes and of its ESNs' runs
TAKES_RUN = REPORT_023 / "data" / "20261117-164658-reach_manual_v2_no_limits"
ARM_CONFIG = REPO_ROOT / "experiments" / "demonstrations" / "reach_manual_v2_no_limits.toml"
EXPERIMENT = "manual_demonstration_v2_many_starts_push_search"
CONFIGS = REPO_ROOT / "experiments" / EXPERIMENT
TRACKER = "pd_w20_z0.4"  # the one tracker's directory in a robot run
DT = 0.01  # the ESNs' period (s)
HOLD = 2.0  # the hold window after arrival (s)
DPI = 90

with ARM_CONFIG.open("rb") as _f:
    _ARM = tomllib.load(_f)
SKELETON = Skeleton.from_toml(ARM_CONFIG)
TARGET = np.array(_ARM["task"]["target"]["pos"])  # m
TOLERANCE = float(_ARM["task"]["target"]["tolerance"])  # the goal's radius (m)
TAKE_STARTS = np.array(_ARM["demonstrations"]["start_q"])  # deg
N_TAKES = len(TAKE_STARTS)

# The chosen ESNs and the one-take baseline's: their names and their undisturbed robot runs.
CHOSEN = {"ESN #1": "robot_esn1", "ESN #2": "robot_esn2"}
ONE_TAKE = {"One-take ESN #1": "robot_one_take_esn1", "One-take ESN #2": "robot_one_take_esn2"}
SIZES = {1: "report 007's forces", 2: "twice", 4: "four times"}  # the pushes' sizes
THIRD_PUSH = (15.0, 15.5)  # the third push, on the task clock (s)
UNSEEN_SHOWN = "unseen 1"  # the unseen start drawn over time: the replay's peak torque from it is the median
SETTINGS = {  # the take counts' settings: leak rate, spectral radius, input scaling, ridge
    "ESN #1's setting": (0.03, 0.9, 2.0, 1.0),
    "ESN #2's setting": (0.017, 1.05, 4.0, 10.0),
}
TAKE_COUNTS = [1, 3, 6, 9]
COARSE = {  # the coarse grid
    "leak_rate": [0.01, 0.03, 0.1, 0.3],
    "spectral_radius": [0.9, 1.2, 1.5],
    "input_scaling": [0.1, 0.5, 2.0],
    "ridge": [1e-4, 1e-2, 1.0],
}
NAMES = ("leak_rate", "spectral_radius", "input_scaling", "ridge")

TEXT_COLOR = "#52514e"
TITLE_COLOR = "#0b0b0b"
GRID_COLOR = "#e1e0d9"
SURFACE_COLOR = "#fcfcfb"
DISTURBANCE_COLOR = "#f0efec"
TAUGHT_COLOR = "#b9b7b0"
NONE_COLOR = "#f0efec"  # a heatmap cell of zero
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
ESN_COLOR, REPLAY_COLOR, GOAL_COLOR = "#2a78d6", "#eb6834", "#7b3fa0"
ESN_COLORS = {"ESN #1": "#1c5cab", "ESN #2": "#6da7ec"}  # where both are drawn
ONE_TAKE_COLORS = {"One-take ESN #1": "#1baf7a", "One-take ESN #2": "#eda100"}
# The layers of the lines over time: the ESN's arm on top, its reference (dashed) under it, the replay's below.
ARM_ZORDERS = {"esn": 3.0, "replay": 2.0}
REFERENCE_ZORDER = 2.5


def main() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Callable[[], Figure]] = {
        "search_coarse.png": plot_coarse,
        "search_following.png": plot_following,
        "rescoring.png": plot_rescoring,
        "paths.png": plot_paths,
        "unseen.png": plot_unseen,
        "pushes_paths.png": plot_push_paths,
        "pushes_distance.png": plot_push_distance,
        "pushes_long.png": plot_long,
        "one_take.png": plot_one_take,
        "takes.png": plot_takes,
    }
    for name in CHOSEN:
        for channel in ("joints", "torques"):
            figures[f"{channel}_esn{name[-1]}.png"] = functools.partial(plot_series, name, channel)
    for name, plot in figures.items():
        plot().savefig(SUMMARY / name, dpi=DPI)
    print_tables()
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
    """The name of the one run of a configuration."""
    matches = sorted(p.name for p in RESULTS.glob(f"*-{stem}") if p.is_dir())
    if len(matches) != 1:
        msg = f"expected one run of {stem}, found {matches}"
        raise FileNotFoundError(msg)
    return matches[0]


def rows(stem: str, arm: str) -> list[dict[str, Any]]:
    """One arm's runs of a robot run, in the order of its starts."""
    return [row for row in read_csv(RESULTS / run_name(stem) / "metrics.csv") if row["arm"] == arm]


def group(stem: str, arm: str, which: str) -> list[dict[str, Any]]:
    """One arm's runs from a set of starts: ``demo`` (the takes'), ``other_takes``, ``unseen``, or ``not_00``."""
    if which == "not_00":
        return [row for row in rows(stem, arm) if row["origin"] != "demo_00_filtered"]
    return [row for row in rows(stem, arm) if row["origin"].startswith(which)]


@functools.lru_cache(maxsize=256)
def log(stem: str, arm: str, start: int) -> StateLog:
    """One arm's log from a robot run under the storage root."""
    run = resolve_run_path(f"results/{EXPERIMENT}/{run_name(stem)}")
    return StateLog.load(run / TRACKER / f"{arm}_{start:02d}.sklog.npz")


def start_of(stem: str, origin: str) -> int:
    return int(next(row["start"] for row in rows(stem, "esn") if row["origin"] == origin))


def pushes(n: str, times: int) -> str:
    """The pushed run of ESN #``n`` at ``times`` the pushes' forces."""
    return f"pushes{times}x_robot_esn{n}"


def scores(stem: str) -> list[dict[str, Any]]:
    return read_csv(RESULTS / run_name(stem) / "scores.csv")


def setting_of(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(row[name] for name in NAMES)


@functools.cache
def take(index: int) -> NDArray[np.float64]:
    """A take filtered at 2 Hz, what the ESNs were trained on, every 10 ms: its joint angles (rad)."""
    return resample_joint_angles(StateLog.load(TAKES_RUN / f"demo_{index:02d}_filtered.sklog.npz"), DT)[1]


def extra_starts(config: str, name: str) -> NDArray[np.float64]:
    """A set of start postures no take starts from, as a configuration lists it (deg)."""
    with (CONFIGS / config).open("rb") as f:
        return np.array(tomllib.load(f)["evaluation"]["extra_starts"][name])


# ---------------------------------------------------------------------------- measures


def hand(q: NDArray[np.float64]) -> NDArray[np.float64]:
    return endpoint_positions(SKELETON, q)


def holds(row: dict[str, Any]) -> bool:
    """The hand arrives within the goal and stays for the 2 s after."""
    return bool(row["success"]) and row["hold_observed_s"] >= HOLD - 1e-9


def ends_in_goal(row: dict[str, Any]) -> bool:
    """Under the pushes: the hand arrives and ends within the goal; leaving it during the hold is excused."""
    return bool(row["arrived"]) and row["final_distance_m"] <= TOLERANCE


def from_target(state: StateLog) -> NDArray[np.float64]:
    """The hand's distance to the target at every step of a log (m)."""
    return np.linalg.norm(hand(state.channel("q")) - TARGET, axis=1)


def pushed(state: StateLog) -> NDArray[np.bool_]:
    """The steps of a log at which a force acts on the tip."""
    return np.linalg.norm(state.channel("ext_force"), axis=1) > 0.0


def in_goal_for_good(state: StateLog) -> float:
    """When the hand enters the goal for the last time (s); NaN if it ends outside."""
    outside = np.flatnonzero(from_target(state) > TOLERANCE)
    if len(outside) == 0:
        return float(state.times[0])
    return float("nan") if outside[-1] == len(state.times) - 1 else float(state.times[outside[-1] + 1])


def third_push(state: StateLog) -> dict[str, float]:
    """The third push's effect on the hand, from the log: distances to the target (m) and times (s)."""
    times, distance = state.times, from_target(state)
    before = (times >= THIRD_PUSH[0] - 1.0) & (times < THIRD_PUSH[0])
    after = times >= THIRD_PUSH[0]
    gap = np.degrees(np.abs(state.channel("q_ref") - state.channel("q"))[after]).max()
    return {
        "before": float(distance[before].max()),
        "at_end": float(distance[np.searchsorted(times, THIRD_PUSH[1])]),
        "farthest": float(distance[after].max()),
        "farthest_at": float(times[after][distance[after].argmax()]),
        "end": float(distance[-1]),
        "end_speed_dps": float(np.degrees(np.abs(state.channel("dq")[-1]).max())),
        "reference_gap_deg": float(gap),
    }


def median(values: list[float]) -> float:
    return float(np.nanmedian(np.array(values, dtype=float)))


def column(runs: list[dict[str, Any]], key: str) -> list[float]:
    return [float(run[key]) for run in runs]


def joint_distance(row: dict[str, Any], starts: NDArray[np.float64]) -> float:
    """How far a start is from the start of the take the replay plays (deg, over the two joints)."""
    played = TAKE_STARTS[int(row["replayed"].split("_")[1])]
    start = starts[int(row["origin"].split()[1])]
    return float(np.linalg.norm(start - played))


# ---------------------------------------------------------------------------- figures


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def workspace(ax: Axes, title: str = "") -> None:
    """The workspace's axes: the ring the hand reaches, the goal, and equal scales."""
    style(ax)
    upper, forearm = (link.prop.length for link in SKELETON.links[1:])
    for radius in (abs(upper - forearm), upper + forearm):
        ax.add_patch(Circle((0.0, 0.0), radius, fill=False, edgecolor=GRID_COLOR, linewidth=1.2, zorder=1))
    ax.add_patch(Circle(tuple(TARGET), TOLERANCE, fill=False, edgecolor=GOAL_COLOR, linewidth=1.6, zorder=5))
    ax.plot(*TARGET, marker="+", color=GOAL_COLOR, markersize=9, zorder=5)
    ax.plot(0.0, 0.0, marker="s", color=TEXT_COLOR, markersize=4, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(-0.95, 0.95)
    ax.set_ylim(-0.95, 0.95)
    if title:
        ax.set_title(title, color=TITLE_COLOR, fontsize=10)


def draw_path(ax: Axes, state: StateLog, color: str, alpha: float = 0.9, mark_end: bool = False) -> None:
    """A hand path over the task, its start as a dot, thick where a force acts, and with ``mark_end`` its end."""
    task = state.times >= 0.0
    points = hand(state.channel("q")[task])
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1.0, alpha=alpha, zorder=3)
    ax.plot(*points[0], marker="o", color=color, markersize=3.5, zorder=4)
    if "ext_force" in state.channel_names:
        force = np.where(pushed(state)[task][:, None], points, np.nan)
        ax.plot(force[:, 0], force[:, 1], color=TITLE_COLOR, linewidth=2.2, zorder=3.5)
    if mark_end:
        ax.plot(*points[-1], marker="x", color=TITLE_COLOR, markersize=5, zorder=4.5)


def count_colormap(top: int) -> tuple[ListedColormap, BoundaryNorm]:
    """A discrete sequential map of the counts 0 to ``top``: zero near the surface, then the blue ramp."""
    ramp = LinearSegmentedColormap.from_list("blues", BLUES)
    colors = [NONE_COLOR] + [ramp(k / max(top - 1, 1)) for k in range(top)]
    return ListedColormap(colors), BoundaryNorm(np.arange(-0.5, top + 1.5), top + 1)


def ink_on(color: tuple[float, ...]) -> str:
    """The text color on a fill: dark on a light fill, white on a dark one."""
    r, g, b = color[:3]
    return TITLE_COLOR if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.45 else "#ffffff"


def plot_coarse() -> Figure:
    """The coarse search: each setting's runs that pass, undisturbed and pushed, of the nine from the takes' starts."""
    table = scores("coarse_score")
    by_setting = {setting_of(row): row for row in table}
    best = setting_of(table[0])
    leaks, radii = COARSE["leak_rate"], COARSE["spectral_radius"]
    scalings, ridges = COARSE["input_scaling"], COARSE["ridge"]
    fig = Figure(figsize=(11, 7.4), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The coarse search (one reservoir a setting): the runs that hold undisturbed, then the runs that pass the"
        " pushes,\nof the nine each from the takes' starts; colored by their sum; outlined: all 18 pass; C: the best"
        " setting, the fine search's center",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(len(ridges), len(scalings), sharex=True, sharey=True)
    cmap, norm = count_colormap(2 * N_TAKES)
    for i, ridge in enumerate(reversed(ridges)):
        for j, scaling in enumerate(scalings):
            ax = axes[i, j]
            cells = [[by_setting[(leak, radius, scaling, ridge)] for leak in leaks] for radius in radii]
            counts = np.array([[cell["passing_runs"] for cell in line] for line in cells])
            ax.imshow(counts, cmap=cmap, norm=norm, origin="lower", aspect="auto")
            for y, radius in enumerate(radii):
                for x, leak in enumerate(leaks):
                    cell = cells[y][x]
                    text = f"{int(cell['holding_runs'])}, {int(cell['passing_pushed_runs'])}"
                    if (leak, radius, scaling, ridge) == best:
                        text += "\nC"
                    ax.text(x, y, text, ha="center", va="center", fontsize=9, color=ink_on(cmap(norm(counts[y, x]))))
                    if counts[y, x] == 2 * N_TAKES:
                        ax.add_patch(
                            Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor=TITLE_COLOR, linewidth=2)
                        )
            ax.set_xticks(range(len(leaks)), [f"{leak:g}" for leak in leaks])
            ax.set_yticks(range(len(radii)), [f"{radius:g}" for radius in radii])
            ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR, length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f"input scaling {scaling:g}, ridge {ridge:g}", color=TITLE_COLOR, fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("leak rate", color=TEXT_COLOR)
    for ax in axes[:, 0]:
        ax.set_ylabel("spectral radius", color=TEXT_COLOR)
    return fig


def pushed_settings(stem: str) -> dict[tuple[float, ...], list[dict[str, Any]]]:
    """The pushed runs of a search's step, by setting (over its seeds)."""
    settings: dict[tuple[float, ...], list[dict[str, Any]]] = {}
    for run in read_csv(RESULTS / run_name(stem) / "runs.csv"):
        settings.setdefault(setting_of(run), []).append(run)
    return settings


def following(runs: list[dict[str, Any]]) -> float:
    """How much the reference follows the pushed arm: the median over the runs of the mean over the three pushes."""
    return median([float(np.mean([run[f"following_gain_{push}"] for push in (1, 2, 3)])) for run in runs])


def plot_following() -> Figure:
    """Each setting's pushed runs that pass against its following gain, by its leak rate."""
    fig = Figure(figsize=(13, 5.2), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "Each setting under the pushes: the share of its runs that arrive and end within the goal, against how much"
        " its reference follows the pushed arm\n(the following gain: the median over its runs of the mean over the"
        " three pushes; 0 keeps to the take, 1 goes with the arm)",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(1, 2, sharey=True)
    panels = [
        ("coarse_pushes", "the coarse search: 108 settings, one reservoir, nine runs each"),
        ("fine_pushes", "the fine search: 81 settings, five reservoirs, 45 runs each"),
    ]
    leaks = sorted({setting[0] for stem, _ in panels for setting in pushed_settings(stem)})
    ramp = LinearSegmentedColormap.from_list("blues", BLUES)
    colors = {leak: ramp(k / (len(leaks) - 1)) for k, leak in enumerate(leaks)}
    for ax, (stem, title) in zip(axes, panels, strict=True):
        style(ax)
        for setting, runs in pushed_settings(stem).items():
            ax.plot(
                following(runs),
                sum(ends_in_goal(run) for run in runs) / len(runs),
                linestyle="",
                marker="o",
                markersize=6,
                color=colors[setting[0]],
                markeredgecolor=TEXT_COLOR,
                markeredgewidth=0.4,
                alpha=0.9,
            )
        ax.set_xlim(0.0, 1.02)
        ax.set_title(title, color=TITLE_COLOR, fontsize=10)
        ax.set_xlabel("following gain", color=TEXT_COLOR)
    axes[0].set_ylabel("share of the pushed runs that pass", color=TEXT_COLOR)
    handles = [
        Line2D([], [], linestyle="", marker="o", color=colors[leak], markeredgecolor=TEXT_COLOR, label=f"{leak:g}")
        for leak in leaks
    ]
    fig.legend(
        handles=handles, title="leak rate", loc="outside right center", frameon=False, labelcolor=TEXT_COLOR
    ).get_title().set_color(TEXT_COLOR)
    return fig


def plot_rescoring() -> Figure:
    """The fine search's 20 best settings, scored again with the 36 untaught starts."""
    rescored = scores("untaught_score")
    fig = Figure(figsize=(8.6, 5.2), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The fine search's 20 best settings, scored again over 270 runs: five reservoirs,"
        "\neach undisturbed from the nine takes' starts and the 36 untaught ones,"
        "\nand pushed from the nine takes' starts",
        color=TITLE_COLOR,
    )
    ax = fig.subplots()
    style(ax)
    labels = {setting_of(rescored[0]): "ESN #1's setting", setting_of(rescored[1]): "ESN #2's setting"}
    for row in rescored:
        label = labels.get(setting_of(row))
        ax.plot(
            row["passing_runs"],
            row["median_detour_ratio"],
            marker="o",
            markersize=8 if label else 6,
            color=ESN_COLOR if label else TAUGHT_COLOR,
            zorder=3 if label else 2,
        )
        if label:
            ax.annotate(
                label,
                (row["passing_runs"], row["median_detour_ratio"]),
                textcoords="offset points",
                xytext=(-8, 6),
                ha="right",
                color=TITLE_COLOR,
                fontsize=9,
            )
    ax.set_xlabel("runs that pass, of 270", color=TEXT_COLOR)
    ax.set_ylabel("median detour ratio of the undisturbed runs that arrive", color=TEXT_COLOR)
    return fig


def plot_paths() -> Figure:
    """The hand's paths, undisturbed: from the takes' starts above, from the unseen starts below."""
    fig = Figure(figsize=(15, 10.4), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The hand's paths on the robot, undisturbed: from the nine takes' starts (above; gray: the takes) and from"
        " the 36 unseen starts (below)",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(2, 3)
    columns = [("ESN #1", "robot_esn1", "esn"), ("ESN #2", "robot_esn2", "esn"), ("the replay", "robot_esn1", "replay")]
    for row, (which, heading) in enumerate((("demo", "from the takes' starts"), ("unseen", "from the unseen starts"))):
        for col, (name, stem, arm) in enumerate(columns):
            ax = axes[row, col]
            suffix = " of the nearest take" if (arm == "replay" and which == "unseen") else ""
            workspace(ax, f"{name}{suffix}, {heading}")
            if which == "demo":
                for index in range(N_TAKES):
                    points = hand(take(index))
                    ax.plot(points[:, 0], points[:, 1], color=TAUGHT_COLOR, linewidth=3.5, zorder=2)
            for run in group(stem, arm, which):
                draw_path(ax, log(stem, arm, int(run["start"])), ESN_COLOR if arm == "esn" else REPLAY_COLOR, 0.8)
    for ax in axes[-1]:
        ax.set_xlabel("x (m)", color=TEXT_COLOR)
    for ax in axes[:, 0]:
        ax.set_ylabel("y (m)", color=TEXT_COLOR)
    return fig


def plot_unseen() -> Figure:
    """From each unseen start: the detour ratio, the reference's fastest joint speed, and the largest torque."""
    starts = extra_starts("robot_esn1.toml", "unseen")
    fig = Figure(figsize=(15, 4.8), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "From each of the 36 unseen starts, against how far it is from the start of the take the replay plays",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(1, 3)
    panels = [
        ("detour_ratio", "detour ratio"),
        ("peak_reference_speed_dps", "the reference's fastest joint speed (°/s)"),
        ("peak_torque_nm", "largest joint torque (N m)"),
    ]
    arms = [
        ("the replay", "robot_esn1", "replay", REPLAY_COLOR, "^"),
        ("ESN #2", "robot_esn2", "esn", ESN_COLORS["ESN #2"], "s"),
        ("ESN #1", "robot_esn1", "esn", ESN_COLORS["ESN #1"], "o"),
    ]
    for ax, (key, label) in zip(axes, panels, strict=True):
        style(ax)
        for _, stem, arm, color, marker in arms:
            runs = group(stem, arm, "unseen")
            replayed = group(stem, "replay", "unseen")
            ax.plot(
                [joint_distance(run, starts) for run in replayed],
                column(runs, key),
                linestyle="",
                marker=marker,
                color=color,
                markersize=5,
                alpha=0.85,
            )
        ax.set_yscale("log")
        ax.set_xlabel("distance from the played take's start (°, over both joints)", color=TEXT_COLOR)
        ax.set_ylabel(label, color=TEXT_COLOR)
    handles = [
        Line2D([], [], linestyle="", marker=marker, color=color, label=name)
        for name, _, _, color, marker in reversed(arms)
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def series_columns(name: str) -> list[tuple[str, int, int, str]]:
    """The columns of an ESN's series over time: the run, its start, the take the replay plays, and the title."""
    n = name[-1]
    undisturbed = CHOSEN[name]
    unseen = next(row for row in rows(undisturbed, "replay") if row["origin"] == UNSEEN_SHOWN)
    posture = extra_starts("robot_esn1.toml", "unseen")[int(UNSEEN_SHOWN.split()[1])]
    columns = [
        (undisturbed, 0, 0, "undisturbed, from take 00's start"),
        (
            undisturbed,
            int(unseen["start"]),
            int(unseen["replayed"].split("_")[1]),
            f"undisturbed, from the unseen start ({posture[0]:g}°, {posture[1]:g}°)",
        ),
    ]
    columns += [(pushes(n, times), 0, 0, f"pushes at {label}, from take 00's start") for times, label in SIZES.items()]
    return columns


def plot_series(name: str, channel: str) -> Figure:
    """One ESN's joint angles or torques over time against the replay's: a column per scenario."""
    columns = series_columns(name)
    fig = Figure(figsize=(18, 9.4), facecolor=SURFACE_COLOR, layout="constrained")
    what = "Joint angles" if channel == "joints" else "Joint torques"
    fig.suptitle(
        f"{what} of {name}'s arm (blue) and the replay's (orange)"
        + (
            "; dashed: the ESN's reference; thick gray: the take the replay plays"
            if channel == "joints"
            else "; each column's axis is cut at the largest torque of the ESN's arm after the first 50 ms"
        ),
        color=TITLE_COLOR,
    )
    axes = fig.subplots(3, len(columns), sharex="col", height_ratios=[0.55, 1.0, 1.0])
    arms = [("esn", ESN_COLOR), ("replay", REPLAY_COLOR)]
    for col, (stem, start, played, title) in enumerate(columns):
        logs = {arm: log(stem, arm, start) for arm, _ in arms}
        taught = np.degrees(take(played))
        for row in range(3):
            style(axes[row, col])
        axes[0, col].set_title(title, color=TITLE_COLOR, fontsize=10)
        for arm, color in arms:
            state = logs[arm]
            force = (
                np.linalg.norm(state.channel("ext_force"), axis=1)
                if "ext_force" in state.channel_names
                else np.zeros(len(state.times))
            )
            axes[0, col].plot(state.times, force, color=color, linewidth=1.0, zorder=ARM_ZORDERS[arm])
            for joint in range(2):
                ax = axes[1 + joint, col]
                if channel == "joints":
                    ax.plot(
                        state.times,
                        np.degrees(state.channel("q")[:, joint]),
                        color=color,
                        linewidth=1.2,
                        zorder=ARM_ZORDERS[arm],
                    )
                    if arm == "esn":
                        ax.plot(
                            state.times,
                            np.degrees(state.channel("q_ref")[:, joint]),
                            color=color,
                            linewidth=1.0,
                            linestyle="--",
                            zorder=REFERENCE_ZORDER,
                        )
                else:
                    ax.plot(
                        state.times, state.channel("tau")[:, joint], color=color, linewidth=1.0, zorder=ARM_ZORDERS[arm]
                    )
        for joint in range(2):
            ax = axes[1 + joint, col]
            if channel == "joints":
                ax.plot(DT * np.arange(len(taught)), taught[:, joint], color=TAUGHT_COLOR, linewidth=4.0, zorder=1)
            else:
                esn = logs["esn"]
                top = max(1.15 * float(np.abs(esn.channel("tau")[esn.times >= 0.05]).max()), 0.5)
                ax.set_ylim(-top, top)
            ax.set_xlim(-1.0, 24.0)
        axes[0, col].set_ylim(-0.5, 22.0)
        axes[-1, col].set_xlabel("time (s)", color=TEXT_COLOR)
    axes[0, 0].set_ylabel("tip force (N)", color=TEXT_COLOR)
    unit = "deg" if channel == "joints" else "N m"
    for joint in range(2):
        axes[1 + joint, 0].set_ylabel(f"joint {joint + 1} ({unit})", color=TEXT_COLOR)
    handles = [
        Line2D([], [], color=ESN_COLOR, linewidth=1.4, label=f"{name}'s arm"),
        Line2D([], [], color=REPLAY_COLOR, linewidth=1.4, label="the replay's arm"),
    ]
    if channel == "joints":
        handles += [
            Line2D([], [], color=ESN_COLOR, linewidth=1.0, linestyle="--", label=f"{name}'s reference"),
            Line2D([], [], color=TAUGHT_COLOR, linewidth=4.0, label="the take"),
        ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def push_panels() -> list[tuple[str, str, str, str]]:
    """The arms under the pushes: name, ESN number of the run, the arm in it, and color."""
    return [
        ("ESN #1", "1", "esn", ESN_COLOR),
        ("ESN #2", "2", "esn", ESN_COLOR),
        ("the replay", "1", "replay", REPLAY_COLOR),
    ]


def plot_push_paths() -> Figure:
    fig = Figure(figsize=(15, 15), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The hand's paths under the three pushes, from the nine takes' starts (dots); thick black: while a force"
        " acts; crosses: where the hand is at 24 s",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(3, 3)
    for row, (name, n, arm, color) in enumerate(push_panels()):
        for col, (times, label) in enumerate(SIZES.items()):
            ax = axes[row, col]
            workspace(ax, f"{name}, pushes at {label}")
            for start in range(N_TAKES):
                draw_path(ax, log(pushes(n, times), arm, start), color, 0.75, mark_end=True)
    for ax in axes[-1]:
        ax.set_xlabel("x (m)", color=TEXT_COLOR)
    for ax in axes[:, 0]:
        ax.set_ylabel("y (m)", color=TEXT_COLOR)
    return fig


def draw_distance(ax: Axes, state: StateLog, color: str) -> None:
    """A hand's distance to the target over the task (mm), thick while a force acts."""
    task = state.times >= 0.0
    times, distance = state.times[task], 1e3 * from_target(state)[task]
    ax.plot(times, distance, color=color, linewidth=1.0, alpha=0.8, zorder=3)
    ax.plot(times, np.where(pushed(state)[task], distance, np.nan), color=TITLE_COLOR, linewidth=2.4, zorder=3.5)


def distance_axes(ax: Axes, title: str, end: float) -> None:
    style(ax)
    ax.axhline(1e3 * TOLERANCE, color=GOAL_COLOR, linewidth=1.2, zorder=2)
    ax.axvspan(*THIRD_PUSH, color=DISTURBANCE_COLOR, zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(1.0, 2500.0)
    ax.set_xlim(0.0, end)
    ax.set_title(title, color=TITLE_COLOR, fontsize=10)


def plot_push_distance() -> Figure:
    fig = Figure(figsize=(15, 10.5), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The hand's distance to the target under the three pushes, from the nine takes' starts; thick black: while a"
        " force acts; purple: the goal's radius, 20 mm",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(3, 3, sharex=True, sharey=True)
    for row, (times, label) in enumerate(SIZES.items()):
        for col, (name, n, arm, color) in enumerate(push_panels()):
            ax = axes[row, col]
            distance_axes(ax, f"{name}, pushes at {label}", 24.0)
            for start in range(N_TAKES):
                draw_distance(ax, log(pushes(n, times), arm, start), color)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", color=TEXT_COLOR)
    for ax in axes[:, 0]:
        ax.set_ylabel("distance to the target (mm)", color=TEXT_COLOR)
    return fig


def plot_long() -> Figure:
    fig = Figure(figsize=(15, 4.4), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The same over 48 s, at report 007's forces: the hand's distance to the target from the nine takes' starts",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(1, 3, sharey=True)
    for ax, (name, n, arm, color) in zip(axes, push_panels(), strict=True):
        distance_axes(ax, name, 48.0)
        for start in range(N_TAKES):
            draw_distance(ax, log(f"pushes1x_long_robot_esn{n}", arm, start), color)
        ax.set_xlabel("time (s)", color=TEXT_COLOR)
    axes[0].set_ylabel("distance to the target (mm)", color=TEXT_COLOR)
    return fig


def plot_one_take() -> Figure:
    """The hand's paths from the 44 starts take 00 does not start from: nine takes against one."""
    fig = Figure(figsize=(18, 4.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The hand's paths, undisturbed, from the 44 starts take 00 does not start from: the other eight takes'"
        " starts and the 36 unseen ones; thick gray: take 00",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(1, 5)
    panels = [(name, stem, "esn", ESN_COLOR) for name, stem in {**CHOSEN, **ONE_TAKE}.items()]
    panels.append(("The replay of take 00", "robot_one_take_esn1", "replay", REPLAY_COLOR))
    for ax, (name, stem, arm, color) in zip(axes, panels, strict=True):
        workspace(ax, name)
        points = hand(take(0))
        ax.plot(points[:, 0], points[:, 1], color=TAUGHT_COLOR, linewidth=3.5, zorder=2)
        for run in group(stem, arm, "not_00"):
            draw_path(ax, log(stem, arm, int(run["start"])), color, 0.7)
        ax.set_xlabel("x (m)", color=TEXT_COLOR)
    axes[0].set_ylabel("y (m)", color=TEXT_COLOR)
    return fig


def take_runs(n_takes: int, setting: tuple[float, ...], which: str) -> list[dict[str, Any]]:
    """The runs of a setting's five reservoirs trained on ``n_takes`` takes, from a set of starts."""
    runs = read_csv(RESULTS / run_name(f"takes{n_takes}") / "runs.csv")
    return [run for run in runs if setting_of(run) == setting and run["origin"].startswith(which)]


def plot_takes() -> Figure:
    fig = Figure(figsize=(16, 4.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "Against the number of takes, from the 36 unseen starts: each dot one reservoir (seeds 1 to 5), the line"
        " over all five together",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(1, 4)
    panels: list[tuple[str, str, bool]] = [
        ("holds", "runs that arrive and hold, of 36", False),
        ("detour_ratio", "median detour ratio", False),
        ("peak_reference_speed_dps", "median of the reference's fastest joint speed (°/s)", True),
        ("peak_torque_nm", "median of the largest joint torque (N m)", True),
    ]
    colors = dict(zip(SETTINGS, ESN_COLORS.values(), strict=True))
    for ax, (key, label, logarithmic) in zip(axes, panels, strict=True):
        style(ax)
        for offset, (name, setting) in zip((-0.08, 0.08), SETTINGS.items(), strict=True):
            pooled = []
            for x, n_takes in enumerate(TAKE_COUNTS):
                runs = take_runs(n_takes, setting, "unseen")
                seeds = sorted({run["seed"] for run in runs})
                each = [[run for run in runs if run["seed"] == seed] for seed in seeds]
                if key == "holds":
                    values = [float(sum(not run["failed"] for run in group_)) for group_ in each]
                    pooled.append(float(np.mean(values)))
                else:
                    values = [median(column(group_, key)) for group_ in each]
                    pooled.append(median(column(runs, key)))
                ax.plot([x + offset] * len(values), values, linestyle="", marker="o", color=colors[name], markersize=5)
            ax.plot(np.arange(len(TAKE_COUNTS)) + offset, pooled, color=colors[name], linewidth=1.4)
        if logarithmic:
            ax.set_yscale("log")
        ax.set_xticks(range(len(TAKE_COUNTS)), [str(n) for n in TAKE_COUNTS])
        ax.set_xlabel("takes", color=TEXT_COLOR)
        ax.set_ylabel(label, color=TEXT_COLOR)
    handles = [Line2D([], [], color=color, marker="o", label=name) for name, color in colors.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, labelcolor=TEXT_COLOR)
    return fig


# ---------------------------------------------------------------------------- tables


def print_tables() -> None:
    print_search()
    print_undisturbed()
    print_pushes()
    print_one_take()
    print_takes()
    print_against_023()
    print_cost()
    print_checks()


def setting_text(row: dict[str, Any]) -> str:
    return "lr {:g}, sr {:g}, is {:g}, ridge {:g}".format(*setting_of(row))


def print_search() -> None:
    print("\n== The search")
    for prefix, total in (("", 9), ("one_take_", 1)):
        table = scores(f"{prefix}coarse_score")
        hold = sum(row["holding_runs"] == total for row in table)
        push = sum(row["passing_pushed_runs"] == total for row in table)
        both = [row for row in table if row["passing_runs"] == 2 * total]
        print(
            f"{prefix}coarse_score: of {len(table)} settings, {hold} hold in all {total} undisturbed, {push} pass all"
            f" {total} pushed, {len(both)} both; best {setting_text(table[0])}, detour"
            f" {table[0]['median_detour_ratio']:.4f}"
        )
        for row in both[:6] if not prefix else []:
            print(f"  all pass: {setting_text(row)}, detour {row['median_detour_ratio']:.4f}")
        print_map(f"{prefix}coarse_pushes", table)
    for prefix in ("", "one_take_"):
        table = scores(f"{prefix}fine_score")
        runs, pushed_runs = int(table[0]["runs"]), int(table[0]["pushed_runs"])
        hold = sum(row["holding_runs"] == row["undisturbed_runs"] for row in table)
        push = sum(row["passing_pushed_runs"] == pushed_runs for row in table)
        both = sum(row["passing_runs"] == runs for row in table)
        print(
            f"{prefix}fine_score: of {len(table)} settings, {hold} hold in all undisturbed runs, {push} pass all"
            f" {pushed_runs} pushed, {both} pass all {runs}; the 20th best passes {int(table[19]['passing_runs'])}"
        )
        print_map(f"{prefix}fine_pushes", table)
    for prefix in ("", "one_take_"):
        table = scores(f"{prefix}untaught_score")
        print(f"{prefix}untaught_score: {len(table)} settings, of {int(table[0]['runs'])} runs:")
        for rank, row in enumerate(table, start=1):
            print(
                f"  {rank:2d}. {setting_text(row)}: {int(row['passing_runs'])} pass"
                f" ({int(row['holding_runs'])}/{int(row['undisturbed_runs'])} hold,"
                f" {int(row['passing_pushed_runs'])}/{int(row['pushed_runs'])} pushed),"
                f" detour {row['median_detour_ratio']:.3f}, typical seed {int(row['typical_seed'])}"
            )
    fine = pushed_settings("fine_pushes")
    for name, setting in SETTINGS.items():
        print(f"{name} in the fine search: following {following(fine[setting]):.2f} over its five reservoirs")


def print_map(stem: str, table: list[dict[str, Any]]) -> None:
    """A step's pushed runs that pass, its undisturbed runs that hold, and its following gain, by each parameter."""
    settings = pushed_settings(stem)
    for name in NAMES:
        cells = []
        for v in sorted({row[name] for row in table}):
            of_value = [row for row in table if row[name] == v]
            runs = [run for setting, each in settings.items() for run in each if setting[NAMES.index(name)] == v]
            cells.append(
                f"{v:g}: {int(sum(row['passing_pushed_runs'] for row in of_value))}/{len(runs)} pushed,"
                f" {int(sum(row['holding_runs'] for row in of_value))}/"
                f"{int(sum(row['undisturbed_runs'] for row in of_value))} hold,"
                f" following {following(runs):.2f},"
                f" detour {median([row['median_detour_ratio'] for row in of_value]):.3f}"
            )
        print(f"  by {name}: {' | '.join(cells)}")


def summary(runs: list[dict[str, Any]]) -> str:
    """A set of runs in one line: those that hold, and the medians (and largest) of their measures."""
    detour, speed = column(runs, "detour_ratio"), column(runs, "peak_reference_speed_dps")
    torque, arrival = column(runs, "peak_torque_nm"), column(runs, "arrival_time_s")
    return (
        f"hold {sum(holds(run) for run in runs)}/{len(runs)} | detour {median(detour):.3f} ({np.nanmax(detour):.2f})"
        f" | speed {median(speed):.0f} ({max(speed):.0f}) dps | torque {median(torque):.1f} ({max(torque):.1f}) N m"
        f" | arrival {median(arrival):.2f} ({np.nanmax(arrival):.2f}) s"
        f" | path RMSE {1e3 * median(column(runs, 'taught_path_rmse_m')):.1f} mm"
        f" | farthest end {1e3 * max(column(runs, 'final_distance_m')):.1f} mm"
    )


def print_undisturbed() -> None:
    print("\n== Undisturbed (medians, the largest in parentheses)")
    for name, stem in CHOSEN.items():
        for arm in ("esn", "replay"):
            for which in ("demo", "unseen"):
                print(f"{name} {arm:6s} {which:6s}: {summary(group(stem, arm, which))}")
    shown = next(row for row in rows("robot_esn1", "replay") if row["origin"] == UNSEEN_SHOWN)
    print(f"the unseen start drawn over time: {UNSEEN_SHOWN}, the replay plays {shown['replayed']},")
    print(f"  its peak torque {shown['peak_torque_nm']:.1f} N m")
    for name, stem in CHOSEN.items():
        for origin in ("demo_00_filtered", UNSEEN_SHOWN):
            for arm in ("esn", "replay"):
                state = log(stem, arm, start_of(stem, origin))
                first = (state.times >= 0.0) & (state.times < 1.0)
                gap = np.degrees(np.abs(state.channel("q_ref") - state.channel("q"))[first]).max()
                later = np.abs(state.channel("tau")[state.times >= 0.05]).max()
                print(
                    f"  {name} {arm:6s} from {origin}: reference to arm in the first second at most"
                    f" {gap:.1f} deg, torque after 50 ms at most {later:.2f} N m"
                )


def print_pushes() -> None:
    print("\n== Under the pushes: runs that arrive and end within the goal, of nine")
    for times in SIZES:
        cells = [f"ESN #{n} {sum(ends_in_goal(run) for run in rows(pushes(n, times), 'esn'))}" for n in "12"]
        cells += [
            f"{arm} {sum(ends_in_goal(run) for run in rows(pushes('1', times), arm))}" for arm in ("replay", "progress")
        ]
        ends = [
            1e3 * max(column(rows(pushes(n, times), arm), "final_distance_m"))
            for n, arm in (("1", "esn"), ("2", "esn"), ("1", "replay"), ("1", "progress"))
        ]
        print(f"x{times}: {', '.join(cells)}; farthest ends (mm): {', '.join(f'{end:.0f}' for end in ends)}")
    print("Each push, over the nine runs: median (smallest to largest)")
    for times in SIZES:
        for name, n, arm, _ in push_panels():
            runs = rows(pushes(n, times), arm)
            parts = []
            for push in (1, 2, 3):
                off = column(runs, f"off_path_m_{push}")
                gain = column(runs, f"following_gain_{push}")
                parts.append(
                    f"push {push}: off the path {median(off):.2f} ({min(off):.2f}-{max(off):.2f}) m,"
                    f" following {median(gain):.2f} ({min(gain):.2f}-{max(gain):.2f})"
                )
            torque = column(runs, "peak_torque_nm")
            print(f"x{times} {name}: {'; '.join(parts)}; peak torque {median(torque):.1f} ({max(torque):.1f}) N m")
    print("The third push, per run (distances to the target)")
    for times in SIZES:
        for name, n, arm, _ in push_panels():
            details = [third_push(log(pushes(n, times), arm, start)) for start in range(N_TAKES)]
            inside = sum(d["before"] <= TOLERANCE for d in details)
            print(
                f"x{times} {name}: in the goal through the second before it {inside}/9;"
                f" at its end {1e3 * median([d['at_end'] for d in details]):.0f}"
                f" ({1e3 * min(d['at_end'] for d in details):.0f}-{1e3 * max(d['at_end'] for d in details):.0f}) mm;"
                f" farthest {median([d['farthest'] for d in details]):.2f}"
                f" ({min(d['farthest'] for d in details):.2f}-{max(d['farthest'] for d in details):.2f}) m"
                f" at {median([d['farthest_at'] for d in details]):.1f} s;"
                f" at 24 s {1e3 * median([d['end'] for d in details]):.0f} mm,"
                f" moving over 1 dps in {sum(d['end_speed_dps'] > 1.0 for d in details)}/9;"
                f" reference to arm at most {max(d['reference_gap_deg'] for d in details):.1f} deg"
            )
            if arm == "esn":
                print(
                    "    per start: "
                    + ", ".join(
                        f"{start}: {d['before'] * 1e3:.0f}>{d['farthest']:.2f} m>{d['end'] * 1e3:.0f} mm"
                        for start, d in enumerate(details)
                    )
                )
    print("Over 48 s, at report 007's forces: in the goal for good from (s), per start")
    for name, n, arm, _ in push_panels():
        stem = f"pushes1x_long_robot_esn{n}"
        times_ = [in_goal_for_good(log(stem, arm, start)) for start in range(N_TAKES)]
        ends = [1e3 * from_target(log(stem, arm, start))[-1] for start in range(N_TAKES)]
        farthest = [third_push(log(stem, arm, start))["farthest"] for start in range(N_TAKES)]
        print(
            f"{name}: {', '.join(f'{t:.1f}' for t in times_)}; ends {', '.join(f'{e:.0f}' for e in ends)} mm;"
            f" farthest after the third push {', '.join(f'{f:.2f}' for f in farthest)} m"
        )
        if arm == "esn":
            same = all(
                np.allclose(
                    log(stem, arm, s).channel("q")[: len(log(pushes(n, 1), arm, s).times)],
                    log(pushes(n, 1), arm, s).channel("q"),
                    atol=1e-9,
                )
                for s in range(N_TAKES)
            )
            print(f"  its first 24 s equal the 24 s run's: {same}")


def print_one_take() -> None:
    print("\n== Nine takes against one (medians, the largest in parentheses)")
    for name, stem in {**CHOSEN, **ONE_TAKE}.items():
        for which in ("not_00", "demo", "other_takes", "unseen"):
            runs = group(stem, "esn", which)
            if which == "demo":
                runs = [run for run in runs if run["origin"] == "demo_00_filtered"]
            if runs:
                print(f"{name:16s} {which:11s}: {summary(runs)}")
    for which in ("not_00", "other_takes", "unseen"):
        print(f"{'replay of take 00':16s} {which:11s}: {summary(group('robot_one_take_esn1', 'replay', which))}")
    for name, stem in ONE_TAKE.items():
        state = log(stem, "esn", start_of(stem, UNSEEN_SHOWN))
        first = (state.times >= 0.0) & (state.times < 1.0)
        gap = np.degrees(np.abs(state.channel("q_ref") - state.channel("q"))[first]).max()
        print(f"{name} from {UNSEEN_SHOWN}: reference to arm in the first second at most {gap:.1f} deg")


def print_takes() -> None:
    print("\n== The take counts (five reservoirs a setting)")
    for name, setting in SETTINGS.items():
        for n_takes in TAKE_COUNTS:
            parts = []
            for which in ("demo", "other_takes", "unseen"):
                runs = take_runs(n_takes, setting, which)
                if not runs:
                    continue
                by_seed = [sum(not run["failed"] for run in runs if run["seed"] == seed) for seed in range(1, 6)]
                parts.append(
                    f"{which} {sum(not run['failed'] for run in runs)}/{len(runs)} {by_seed},"
                    f" detour {median(column(runs, 'detour_ratio')):.3f},"
                    f" speed {median(column(runs, 'peak_reference_speed_dps')):.0f} dps,"
                    f" torque {median(column(runs, 'peak_torque_nm')):.1f} N m"
                )
            print(f"{name}, {n_takes} takes: {' | '.join(parts)}")


def rows_023(stem: str, arm: str) -> list[dict[str, Any]]:
    """One arm's runs of a robot run of report 023, from its copy there."""
    (run,) = (p for p in (REPORT_023 / "results").glob(f"*-{stem}") if p.is_dir())
    return [row for row in read_csv(run / "metrics.csv") if row["arm"] == arm]


def print_against_023() -> None:
    print("\n== Against report 023's ESNs (medians, the largest in parentheses)")
    sources = {"": rows, "report 023's ": rows_023}
    for label, source in sources.items():
        for n in "12":
            unseen = [row for row in source(f"robot_esn{n}", "esn") if row["origin"].startswith("unseen")]
            taught = [row for row in source(f"robot_esn{n}", "esn") if row["origin"].startswith("demo")]
            print(f"{label}ESN #{n} unseen: {summary(unseen)}")
            print(f"{label}ESN #{n} takes : {summary(taught)}")
            for times in SIZES:
                runs = source(pushes(n, times), "esn")
                gains = [median(column(runs, f"following_gain_{push}")) for push in (1, 2, 3)]
                off = [median(column(runs, f"off_path_m_{push}")) for push in (1, 2, 3)]
                print(
                    f"  x{times}: pass {sum(ends_in_goal(run) for run in runs)}/9; following"
                    f" {', '.join(f'{g:.2f}' for g in gains)}; off the path {', '.join(f'{o:.2f}' for o in off)} m;"
                    f" ends {1e3 * median(column(runs, 'final_distance_m')):.0f}"
                    f" ({1e3 * max(column(runs, 'final_distance_m')):.0f}) mm"
                )


def print_cost() -> None:
    """The 20 settings scored again: the following gain under the pushes and the start from the untaught starts."""
    print("\n== The 20 settings scored again, by leak rate and input scaling (ranges over the settings)")
    fine = pushed_settings("fine_pushes")
    untaught: dict[tuple[float, ...], list[dict[str, Any]]] = {}
    for run in read_csv(RESULTS / run_name("untaught_search") / "runs.csv"):
        if run["origin"].startswith("untaught"):
            untaught.setdefault(setting_of(run), []).append(run)
    groups: dict[tuple[float, float], list[tuple[float, ...]]] = {}
    for setting in untaught:
        groups.setdefault((setting[0], setting[2]), []).append(setting)
    for (leak, scaling), settings in sorted(groups.items()):
        cells = []
        for label, measure in (
            ("following", lambda s: following(fine[s])),
            ("speed (dps)", lambda s: median(column(untaught[s], "peak_reference_speed_dps"))),
            ("torque (N m)", lambda s: median(column(untaught[s], "peak_torque_nm"))),
            ("detour", lambda s: median(column(untaught[s], "detour_ratio"))),
            ("hold of 180", lambda s: float(sum(not run["failed"] for run in untaught[s]))),
        ):
            values = [measure(s) for s in settings]
            cells.append(f"{label} {min(values):.3g} to {max(values):.3g}")
        print(f"lr {leak:g}, is {scaling:g} ({len(settings)} settings): {'; '.join(cells)}")


def print_checks() -> None:
    print("\n== Checks")
    records = {}
    for run in sorted(p for p in RESULTS.iterdir() if p.is_dir() and p.name != "summary"):
        with (run / "run.toml").open("rb") as f:
            record = tomllib.load(f)
        records[run.name] = (record["commit"][:7], record["uncommitted_changes"])
    for name, (commit, dirty) in records.items():
        print(f"{name}: {commit}{' WITH UNCOMMITTED CHANGES' if dirty else ''}")
    replays = [
        np.allclose(column(rows("robot_esn1", "replay"), key), column(rows("robot_esn2", "replay"), key))
        for key in ("detour_ratio", "peak_torque_nm", "final_distance_m")
    ]
    print(f"the replay beside ESN #1 equals the one beside ESN #2: {all(replays)}")


if __name__ == "__main__":
    main()
