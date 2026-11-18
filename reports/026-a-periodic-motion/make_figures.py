# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 026.

    uv run python reports/026-a-periodic-motion/make_figures.py

It reads the copies of the runs under ``results/`` (their configurations, run records, and CSV files), the takes
under ``data/``, and the robot runs' logs and the takes the ESNs were trained on under the storage root, which are
not kept in Git. Each take in ``TAKES`` has its own search, ESNs, and runs, and its own figures and tables. It
prints the tables of the report and draws into ``results/summary``:

- ``starts.png``: the curve, the takes' hand paths, and the two sets of start postures;
- ``<take>_search.png``: the coarse search's settings by their runs that cycle, and the fine search's 20 best
  settings scored again with the untaught starts;
- ``<take>_paths.png``: the hand's paths over the first 8 s, undisturbed, from the starts on the curve and from the
  untaught ones;
- ``<take>_<scenario>_joints.png`` and ``<take>_<scenario>_torques.png``: the joint angles or torques of ESN #1's
  and ESN #2's arms over time against the replay's, in each scenario of ``SCENARIOS``: undisturbed from two starts
  on the curve and from an untaught start, under the pushes at each size, and with the block;
- ``<take>_off_curve.png``: the hand's distance from the curve over time, from the untaught starts, under the
  pushes, and with the block;
- ``<take>_long.png``: claim 1's runs of 120 s: the hand's distance from the curve over time, and its last five
  loops.
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
from matplotlib.patches import Circle, Patch, Rectangle
from numpy.typing import NDArray
from skelarm import Skeleton, StateLog

from arm_esn_ctrl.autonomous import Cycle, Setup, load_setup, task_cycle
from arm_esn_ctrl.demonstrations import endpoint_positions, resample_joint_angles
from arm_esn_ctrl.metrics import distances_to_path, loop_phase
from arm_esn_ctrl.storage import resolve_run_path

REPORT = Path(__file__).resolve().parent
REPO_ROOT = REPORT.parents[1]
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"
DATA = REPORT / "data"
ARM_CONFIG = REPO_ROOT / "experiments" / "demonstrations" / "reach_manual_v2_no_limits.toml"  # stage 1's robot
EXPERIMENT = "manual_demonstration_v2_periodic"
TRACKER = "pd_w20_z0.4"  # the one tracker's directory in a robot run
DT = 0.01  # the ESNs' period (s)
DPI = 90

SKELETON = Skeleton.from_toml(ARM_CONFIG)

# The takes, each with its own search, ESNs, and runs: the prefix of its configurations' names, its demonstration
# run (a copy under data/), the file of it the ESNs are trained on, and its name in the figures.
TAKES = {
    "scripted": {
        "prefix": "scripted_",
        "demonstrations": "20261118-171132-ellipse_scripted_v2_no_limits",
        "train": "demo_00.sklog.npz",
        "label": "The scripted take",
    },
}
ESNS = {"ESN #1": "1", "ESN #2": "2"}  # the chosen ESNs and their number in the configurations' names
SIZES = {1: "report 007's forces", 2: "twice their forces", 4: "four times their forces"}  # the pushes' sizes
ON_CURVE = 8  # the starts on the curve, an eighth of a loop apart; start 0 is the take's
# The scenarios drawn over time: the runs' configuration without the take's prefix and the ESN's number, the starts
# drawn (None: the untaught start of shown_untaught), the time drawn (s), and the columns' title.
SCENARIOS: dict[str, tuple[str, tuple[int, ...] | None, tuple[float, float], str]] = {
    "on_curve": ("robot_esn", (0, 4), (-1.0, 12.0), "undisturbed, from start {start} on the curve ({angle}°)"),
    "untaught": (
        "untaught_robot_esn",
        None,
        (-1.0, 12.0),
        "undisturbed, from untaught start {start} ({q1:g}°, {q2:g}°)",
    ),
    "pushes1x": ("pushes1x_robot_esn", (0,), (6.0, 40.0), "pushes at report 007's forces, from start 0"),
    "pushes2x": ("pushes2x_robot_esn", (0,), (6.0, 40.0), "pushes at twice their forces, from start 0"),
    "pushes4x": ("pushes4x_robot_esn", (0,), (6.0, 40.0), "pushes at four times their forces, from start 0"),
    "block": ("block_robot_esn", (0,), (8.0, 20.0), "the tip held from 10 s to 13 s, from start 0"),
}
NAMES = ("leak_rate", "spectral_radius", "input_scaling", "ridge")
# The stretches of the workspace drawn, x and y (m): all of it, the curve and the paths from the starts on it in
# their first 8 s, and the curve alone.
WHOLE_VIEW = ((-0.95, 0.95), (-0.95, 0.95))
STARTS_VIEW = ((-0.35, 0.45), (0.37, 0.73))
CURVE_VIEW = ((-0.34, 0.14), (0.40, 0.70))

TEXT_COLOR = "#52514e"
TITLE_COLOR = "#0b0b0b"
GRID_COLOR = "#e1e0d9"
SURFACE_COLOR = "#fcfcfb"
DISTURBANCE_COLOR = "#f0efec"
TAUGHT_COLOR = "#b9b7b0"
NONE_COLOR = "#f0efec"  # a heatmap cell of zero
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
REPLAY_COLOR, GOAL_COLOR = "#eb6834", "#7b3fa0"
ESN_COLORS = {"ESN #1": "#1c5cab", "ESN #2": "#6da7ec"}
# The layers of the lines over time: the ESN's arm on top, its reference (dashed) under it, the replay's below.
ARM_ZORDERS = {"esn": 3.0, "replay": 2.0}
REFERENCE_ZORDER = 2.5


def main() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Callable[[], Figure]] = {"starts.png": plot_starts}
    for name in TAKES:
        figures[f"{name}_search.png"] = functools.partial(plot_search, name)
        figures[f"{name}_paths.png"] = functools.partial(plot_paths, name)
        for scenario in SCENARIOS:
            for channel in ("joints", "torques"):
                figures[f"{name}_{scenario}_{channel}.png"] = functools.partial(plot_series, name, scenario, channel)
        figures[f"{name}_off_curve.png"] = functools.partial(plot_off_curve, name)
        figures[f"{name}_long.png"] = functools.partial(plot_long, name)
    for file, plot in figures.items():
        plot().savefig(SUMMARY / file, dpi=DPI)
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


def stem(name: str, base: str, n: str = "") -> str:
    """The configuration of a take's run: ``base`` with the take's prefix and the ESN's number ``n``."""
    return f"{TAKES[name]['prefix']}{base}{n}"


def table(run: str, file: str) -> list[dict[str, Any]]:
    """A CSV file of a run, by the run's configuration (see ``stem``)."""
    return read_csv(RESULTS / run_name(run) / file)


@functools.cache
def config(directory: str) -> dict[str, Any]:
    """The configuration of a run, by its directory's name under ``results/``."""
    with (RESULTS / directory / "config.toml").open("rb") as f:
        return tomllib.load(f)


def rows(run: str, arm: str) -> list[dict[str, Any]]:
    """One arm's runs of a robot run, in the order of its starts."""
    return [row for row in table(run, "metrics.csv") if row["arm"] == arm]


@functools.lru_cache(maxsize=8)
def log(run: str, arm: str, start: int) -> StateLog:
    """One arm's log from a robot run under the storage root; a few are kept, each some 50 MB in memory."""
    path = resolve_run_path(f"results/{EXPERIMENT}/{run_name(run)}")
    return StateLog.load(path / TRACKER / f"{arm}_{start:02d}.sklog.npz")


def esn_run(name: str, n: str) -> str:
    """The run that trained a take's ESN #``n``, as the configuration of its undisturbed robot run names it."""
    return Path(config(run_name(stem(name, "robot_esn", n)))["esn"]["model"]).parent.name


@functools.cache
def take(name: str) -> NDArray[np.float64]:
    """A take as the ESNs were trained on it, every 10 ms: its joint angles (rad), from its copy under ``data/``."""
    spec = TAKES[name]
    return resample_joint_angles(StateLog.load(DATA / spec["demonstrations"] / spec["train"]), DT)[1]


@functools.cache
def setup(name: str) -> Setup:
    """What a take's runs were compared with, from ESN #1's configuration, its take read from the storage root."""
    return load_setup(config(esn_run(name, "1")))


@functools.cache
def cycle(name: str) -> Cycle:
    """What a take's runs are measured against: the curve, its center, the taught period, and the replay's loop."""
    measured = task_cycle(setup(name), config(esn_run(name, "1"))["evaluation"])
    assert measured is not None
    return measured


def extra_starts(run: str, group: str) -> NDArray[np.float64]:
    """A group of start postures, as a robot run's configuration lists it (deg)."""
    return np.array(config(run_name(run))["evaluation"]["extra_starts"][group])


def starts_on_curve(name: str) -> NDArray[np.float64]:
    """The eight starts on the curve (deg): the take's own, then the other seven in the order of their angles."""
    others = extra_starts(stem(name, "robot_esn", "1"), "on_curve")
    return np.vstack([np.degrees(take(name)[0]), others])


def spans(run: str) -> list[tuple[float, float]]:
    """When each disturbance of a robot run acts, from its onset to its end (s)."""
    tables = config(run_name(run)).get("disturbance", [])
    tables = tables if isinstance(tables, list) else [tables]
    return [(d["onset"], d["release"] if d["type"] == "block" else d["onset"] + d["duration"]) for d in tables]


def scores(run: str) -> list[dict[str, Any]]:
    return table(run, "scores.csv")


def setting_of(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(row[name] for name in NAMES)


# ---------------------------------------------------------------------------- measures


def hand(q: NDArray[np.float64]) -> NDArray[np.float64]:
    return endpoint_positions(SKELETON, q)


def window(state: StateLog, begin: float, end: float) -> NDArray[np.bool_]:
    return (state.times >= begin - 1e-9) & (state.times <= end + 1e-9)


def off_curve(name: str, state: StateLog, begin: float, end: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The hand's distance from the curve (m) over a stretch of a log, and its times (s)."""
    shown = window(state, begin, end)
    return state.times[shown], distances_to_path(hand(state.channel("q")[shown]), cycle(name).curve)


@functools.cache
def phase(name: str, run: str, arm: str, start: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The times (s) of one arm's log and the hand's angle about the curve's center then, unwrapped (rad)."""
    state = log(run, arm, start)
    return state.times, loop_phase(hand(state.channel("q")), cycle(name).center)


def delay(name: str, base: str, n: str, arm: str, start: int) -> float:
    """How far behind its own undisturbed run the hand is when a disturbed run ends (s).

    The undisturbed run is claim 1's run of 120 s of the same arm from the same start. The delay is the disturbed
    run's end less the time at which the undisturbed hand had the angle about the curve's center (unwrapped, see
    ``loop_phase``) that the disturbed hand has at its end. It is taken within a loop, between a quarter of the
    taught period below zero and three quarters above it (-1 s to 3 s at 4 s a loop): a hand pushed around the far
    side of the center loses or gains a turn of the angle, which is no delay.
    """
    times, angle = phase(name, stem(name, base, n), arm, start)
    undisturbed_times, undisturbed = phase(name, stem(name, "robot_esn", n), arm, start)
    period = cycle(name).period
    late = float(times[-1] - np.interp(angle[-1], undisturbed, undisturbed_times))
    return (late + period / 4) % period - period / 4


def shown_untaught(name: str) -> int:
    """The untaught start drawn over time: the replay's largest torque from it is the median of the 36, the lower of
    the two middle ones."""
    replay = rows(stem(name, "untaught_robot_esn", "1"), "replay")
    order = np.argsort(column(replay, "peak_torque_nm"))
    return int(replay[order[(len(replay) - 1) // 2]]["start"])


def column(runs: list[dict[str, Any]], key: str) -> list[float]:
    return [float(run[key]) for run in runs]


def spread(values: list[float], fmt: str, scale: float = 1.0) -> str:
    """Values as their median and range, ``median (smallest to largest)``; NaN left out and counted."""
    finite = [scale * v for v in values if not np.isnan(v)]
    if not finite:
        return "none"
    text = f"{np.median(finite):{fmt}} ({min(finite):{fmt}} to {max(finite):{fmt}})"
    return text if len(finite) == len(values) else f"{text}, {len(finite)} of {len(values)}"


def smaller(esn: list[dict[str, Any]], replay: list[dict[str, Any]], key: str) -> str:
    """In how many starts a measure of the ESN's arm is smaller than the replay's."""
    return f"{sum(e < r for e, r in zip(column(esn, key), column(replay, key), strict=True))} of {len(esn)}"


# ---------------------------------------------------------------------------- figures


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def workspace(ax: Axes, name: str, view: tuple[tuple[float, float], ...], title: str = "") -> None:
    """The workspace's axes over ``view``: the ring the hand reaches, the curve, and equal scales."""
    style(ax)
    upper, forearm = (link.prop.length for link in SKELETON.links[1:])
    for radius in (abs(upper - forearm), upper + forearm):
        ax.add_patch(Circle((0.0, 0.0), radius, fill=False, edgecolor=GRID_COLOR, linewidth=1.2, zorder=1))
    curve = cycle(name).curve
    ax.plot(curve[:, 0], curve[:, 1], color=GOAL_COLOR, linewidth=1.4, zorder=2)
    ax.plot(0.0, 0.0, marker="s", color=TEXT_COLOR, markersize=4, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(*view[0])
    ax.set_ylim(*view[1])
    if title:
        ax.set_title(title, color=TITLE_COLOR, fontsize=10)


def draw_path(ax: Axes, state: StateLog, color: str, end: float) -> None:
    """A hand path from the start of the task to ``end`` (s), its start as a dot."""
    points = hand(state.channel("q")[window(state, 0.0, end)])
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1.0, alpha=0.8, zorder=3)
    ax.plot(*points[0], marker="o", color=color, markersize=3.5, zorder=4)


def plot_starts() -> Figure:
    first = next(iter(TAKES))
    fig = Figure(figsize=(7.6, 7.4), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The curve, the takes' hand paths, and the start postures by their hands;"
        "\nthe starts on the curve numbered by the curve's angle, 45° apart, 0 the take's start",
        color=TITLE_COLOR,
    )
    ax = fig.subplots()
    workspace(ax, first, WHOLE_VIEW)
    for name in TAKES:
        points = hand(take(name))
        ax.plot(points[:, 0], points[:, 1], color=TAUGHT_COLOR, linewidth=4.0, zorder=1.5)
    on_curve = hand(np.radians(starts_on_curve(first)))
    center = cycle(first).center
    for index, point in enumerate(on_curve):
        ax.plot(*point, marker="o", color=TITLE_COLOR, markersize=6, zorder=4)
        x, y = 14 * (center - point) / np.linalg.norm(center - point)  # inward, toward the center (points)
        ax.annotate(str(index), point, textcoords="offset points", xytext=(x, y), ha="center", va="center")
    untaught = hand(np.radians(extra_starts(stem(first, "untaught_robot_esn", "1"), "untaught")))
    ax.plot(untaught[:, 0], untaught[:, 1], linestyle="", marker="s", color=REPLAY_COLOR, markersize=5, zorder=3)
    ax.set_xlabel("x (m)", color=TEXT_COLOR)
    ax.set_ylabel("y (m)", color=TEXT_COLOR)
    handles = [
        Line2D([], [], color=GOAL_COLOR, linewidth=1.4, label="the curve"),
        Line2D([], [], color=TAUGHT_COLOR, linewidth=4.0, label="a take"),
        Line2D([], [], linestyle="", marker="o", color=TITLE_COLOR, label="start on the curve"),
        Line2D([], [], linestyle="", marker="s", color=REPLAY_COLOR, label="untaught start"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def count_colormap(top: int) -> tuple[ListedColormap, BoundaryNorm]:
    """A discrete sequential map of the counts 0 to ``top``: zero near the surface, then the blue ramp."""
    ramp = LinearSegmentedColormap.from_list("blues", BLUES)
    colors = [NONE_COLOR] + [ramp(k / max(top - 1, 1)) for k in range(top)]
    return ListedColormap(colors), BoundaryNorm(np.arange(-0.5, top + 1.5), top + 1)


def ink_on(color: tuple[float, ...]) -> str:
    """The text color on a fill: dark on a light fill, white on a dark one."""
    r, g, b = color[:3]
    return TITLE_COLOR if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.45 else "#ffffff"


def setting_text(row: dict[str, Any]) -> str:
    return "lr {:g}, sr {:g}, is {:g}, ridge {:g}".format(*setting_of(row))


def plot_search(name: str) -> Figure:
    """The coarse search's settings by their runs that cycle, and the 20 settings scored with the untaught starts."""
    fig = Figure(figsize=(11, 13.4), facecolor=SURFACE_COLOR, layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.0])
    top, bottom = fig.add_subfigure(grid[0]), fig.add_subfigure(grid[1])
    coarse = scores(stem(name, "coarse_search"))
    by_setting = {setting_of(row): row for row in coarse}
    best = setting_of(coarse[0])
    leaks, radii, scalings, ridges = (sorted({row[key] for row in coarse}) for key in NAMES)
    total = int(coarse[0]["runs"])
    top.suptitle(
        f"{TAKES[name]['label']}: the coarse search (one reservoir a setting), its runs that cycle of the {total} from"
        " the starts on the curve;\noutlined: all cycle; C: the best setting, the fine search's center",
        color=TITLE_COLOR,
    )
    axes = top.subplots(len(ridges), len(scalings), sharex=True, sharey=True)
    cmap, norm = count_colormap(total)
    for i, ridge in enumerate(reversed(ridges)):
        for j, scaling in enumerate(scalings):
            ax = axes[i, j]
            counts = np.array(
                [[by_setting[(leak, radius, scaling, ridge)]["cycling_runs"] for leak in leaks] for radius in radii]
            )
            ax.imshow(counts, cmap=cmap, norm=norm, origin="lower", aspect="auto")
            for y, radius in enumerate(radii):
                for x, leak in enumerate(leaks):
                    count = int(counts[y, x])
                    text = f"{count}\nC" if (leak, radius, scaling, ridge) == best else str(count)
                    ax.text(x, y, text, ha="center", va="center", fontsize=9, color=ink_on(cmap(norm(count))))
                    if count == total:
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

    rescored = scores(stem(name, "untaught_search"))
    seeds = table(stem(name, "untaught_search"), "sweep.csv")
    runs = int(rescored[0]["runs"])
    bottom.suptitle(
        f"The fine search's {len(rescored)} best settings scored again over {runs} runs:"
        "\nfive reservoirs, each from the starts on the curve and the untaught ones;"
        "\nlarge dot: the setting's median curve RMS; small dots: each reservoir's own; ringed: the typical seed",
        color=TITLE_COLOR,
    )
    ax = bottom.subplots()
    style(ax)
    chosen = dict(enumerate(ESN_COLORS.values(), start=1))  # the chosen ESNs' settings, by their rank
    for rank, row in enumerate(rescored, start=1):
        color = chosen.get(rank, TEXT_COLOR)
        own = [seed for seed in seeds if setting_of(seed) == setting_of(row)]
        ax.plot(
            [rank] * len(own),
            [1e3 * seed["median_curve_rms_m"] for seed in own],
            linestyle="",
            marker="o",
            color=color,
            markersize=3.5,
            alpha=0.6,
            zorder=2,
        )
        typical = next(seed for seed in own if seed["seed"] == row["typical_seed"])
        ax.plot(
            rank,
            1e3 * typical["median_curve_rms_m"],
            marker="o",
            markersize=9,
            markerfacecolor="none",
            color=color,
            zorder=2.5,
        )
        ax.plot(rank, 1e3 * row["median_curve_rms_m"], marker="o", markersize=7, color=color, zorder=3)
        if row["cycling_runs"] < row["runs"]:
            ax.annotate(
                f"{int(row['cycling_runs'])} of {runs} cycle",
                (rank, 1e3 * row["median_curve_rms_m"]),
                textcoords="offset points",
                xytext=(-60, 22),
                ha="right",
                color=TITLE_COLOR,
                fontsize=9,
                arrowprops={"arrowstyle": "-", "color": TEXT_COLOR, "linewidth": 0.8},
            )
    for rank, label in enumerate(ESNS, start=1):
        ax.annotate(
            f"{label}'s setting",
            (rank, 1e3 * rescored[rank - 1]["median_curve_rms_m"]),
            textcoords="offset points",
            xytext=(10, -16 if rank == 1 else 10),
            color=TITLE_COLOR,
            fontsize=9,
        )
    ax.set_xticks(
        range(1, len(rescored) + 1),
        ["{:g}, {:g}, {:g}, {:g}".format(*setting_of(row)) for row in rescored],
        rotation=60,
        ha="right",
        fontsize=8,
    )
    ax.set_xlabel("setting, by its rank: leak rate, spectral radius, input scaling, ridge", color=TEXT_COLOR)
    ax.set_ylabel("median curve RMS (mm)", color=TEXT_COLOR)
    return fig


def plot_paths(name: str) -> Figure:
    """The hand's paths over the first 8 s, undisturbed: from the starts on the curve above, the untaught ones below."""
    fig = Figure(figsize=(15, 9.4), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        f"{TAKES[name]['label']}: the hand's paths on the robot over the first 8 s, undisturbed;"
        "\nfrom the eight starts on the curve (above, near the curve) and from the 36 untaught starts (below);"
        " purple: the curve",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(2, 3, height_ratios=[0.5, 1.0])
    arms = [(label, n, "esn", ESN_COLORS[label]) for label, n in ESNS.items()] + [
        ("the replay", "1", "replay", REPLAY_COLOR)
    ]
    for row, (base, heading) in enumerate(
        (("robot_esn", "from the starts on the curve"), ("untaught_robot_esn", "from the untaught starts"))
    ):
        for col, (label, n, arm, color) in enumerate(arms):
            ax = axes[row, col]
            workspace(ax, name, STARTS_VIEW if row == 0 else WHOLE_VIEW, f"{label}, {heading}")
            run = stem(name, base, n)
            for start in column(rows(run, arm), "start"):
                draw_path(ax, log(run, arm, int(start)), color, 8.0)
    for ax in axes[-1]:
        ax.set_xlabel("x (m)", color=TEXT_COLOR)
    for ax in axes[:, 0]:
        ax.set_ylabel("y (m)", color=TEXT_COLOR)
    return fig


def series_columns(name: str, scenario: str) -> list[tuple[str, str, int, str]]:
    """The columns of a scenario over time: the ESN, the run, the start, and the title."""
    base, starts, _, title = SCENARIOS[scenario]
    if starts is None:
        starts = (shown_untaught(name),)
    columns = []
    for start in starts:
        for label, n in ESNS.items():
            run = stem(name, base, n)
            q1, q2 = extra_starts(run, "untaught")[start] if scenario == "untaught" else (np.nan, np.nan)
            text = title.format(start=start, angle=round(360 * start / ON_CURVE), q1=q1, q2=q2)
            columns.append((label, run, start, f"{label}, {text}"))
    return columns


def plot_series(name: str, scenario: str, channel: str) -> Figure:
    """The joint angles or torques of the ESNs' arms over time against the replay's: a column per ESN and start."""
    columns = series_columns(name, scenario)
    begin, end = SCENARIOS[scenario][2]
    fig = Figure(figsize=(1.0 + 5.4 * len(columns), 7.6), facecolor=SURFACE_COLOR, layout="constrained")
    disturbed = bool(spans(columns[0][1]))
    what = "joint angles" if channel == "joints" else "joint torques"
    notes = []
    if channel == "joints":
        notes.append("dashed: the ESN's reference")
    if disturbed:
        notes.append("shaded: while a disturbance acts")
    if channel == "torques":
        notes.append("each column's axis is cut at the largest torque of the ESN's arm after the first 50 ms")
    fig.suptitle(
        f"{TAKES[name]['label']}: {what} of the ESNs' arms (blue) and the replay's (orange);\n" + "; ".join(notes),
        color=TITLE_COLOR,
    )
    axes = fig.subplots(2, len(columns), sharex=True)
    for col, (label, run, start, title) in enumerate(columns):
        axes[0, col].set_title(title, color=TITLE_COLOR, fontsize=10)
        logs = {arm: log(run, arm, start) for arm in ("esn", "replay")}
        for joint in range(2):
            ax = axes[joint, col]
            style(ax)
            for onset, stop in spans(run):
                ax.axvspan(onset, stop, color=DISTURBANCE_COLOR, zorder=0)
            for arm, state in logs.items():
                color = ESN_COLORS[label] if arm == "esn" else REPLAY_COLOR
                shown = window(state, begin, end)
                times = state.times[shown]
                if channel == "joints":
                    q = np.degrees(state.channel("q")[shown, joint])
                    ax.plot(times, q, color=color, linewidth=1.2, zorder=ARM_ZORDERS[arm])
                    if arm == "esn":
                        reference = np.degrees(state.channel("q_ref")[shown, joint])
                        ax.plot(times, reference, color=color, linewidth=1.0, linestyle="--", zorder=REFERENCE_ZORDER)
                else:
                    ax.plot(
                        times, state.channel("tau")[shown, joint], color=color, linewidth=1.0, zorder=ARM_ZORDERS[arm]
                    )
            if channel == "torques":
                esn = logs["esn"]
                tau = esn.channel("tau")[window(esn, max(begin, 0.05), end)]
                top = max(1.15 * float(np.abs(tau).max()), 0.5)
                ax.set_ylim(-top, top)
            ax.set_xlim(begin, end)
        axes[-1, col].set_xlabel("time (s)", color=TEXT_COLOR)
    unit = "deg" if channel == "joints" else "N m"
    for joint in range(2):
        axes[joint, 0].set_ylabel(f"joint {joint + 1} ({unit})", color=TEXT_COLOR)
    handles: list[Line2D | Patch] = [
        Line2D([], [], color=color, linewidth=1.4, label=f"{label}'s arm") for label, color in ESN_COLORS.items()
    ]
    if channel == "joints":
        handles += [Line2D([], [], color=TEXT_COLOR, linewidth=1.0, linestyle="--", label="the ESN's reference")]
    handles += [Line2D([], [], color=REPLAY_COLOR, linewidth=1.4, label="the replay's arm")]
    if disturbed:
        handles += [Patch(color=DISTURBANCE_COLOR, label="a disturbance acts")]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def distance_axes(ax: Axes, name: str, title: str, begin: float, end: float, run: str) -> None:
    """The axes of the distance from the curve over time: logarithmic, with the 20 mm line and the disturbances."""
    style(ax)
    ax.axhline(1e3 * cycle(name).tolerance, color=GOAL_COLOR, linewidth=1.2, zorder=4)
    for onset, stop in spans(run):
        ax.axvspan(onset, stop, color=DISTURBANCE_COLOR, zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 2000.0)
    ax.set_xlim(begin, end)
    ax.set_title(title, color=TITLE_COLOR, fontsize=10)


def draw_off_curve(ax: Axes, name: str, run: str, color: str, begin: float, end: float) -> None:
    """Every start's distance from the curve over time (mm), the ESN's arm in ``color`` over the replay's."""
    for arm, line in (("esn", color), ("replay", REPLAY_COLOR)):
        for start in column(rows(run, arm), "start"):
            times, distance = off_curve(name, log(run, arm, int(start)), begin, end)
            ax.plot(times, 1e3 * distance, color=line, linewidth=0.7, alpha=0.7, zorder=ARM_ZORDERS[arm])


def plot_off_curve(name: str) -> Figure:
    fig = Figure(figsize=(15, 16), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        f"{TAKES[name]['label']}: the hand's distance from the curve over time, from every start, the ESN's arm (blue)"
        " over the replay's (orange);\npurple: 20 mm, the curve RMS a run that cycles stays within; shaded: while a"
        " disturbance acts",
        color=TITLE_COLOR,
    )
    panels = [("untaught_robot_esn", "undisturbed, from the 36 untaught starts", 0.0, 6.0)]
    panels += [
        (f"pushes{times}x_robot_esn", f"pushes at {label}, from the eight starts on the curve", 6.0, 40.0)
        for times, label in SIZES.items()
    ]
    panels += [("block_robot_esn", "the tip held from 10 s to 13 s, from the eight starts on the curve", 8.0, 20.0)]
    axes = fig.subplots(len(panels), len(ESNS), sharey=True)
    for row, (base, heading, begin, end) in enumerate(panels):
        for col, (label, n) in enumerate(ESNS.items()):
            run = stem(name, base, n)
            distance_axes(axes[row, col], name, f"{label}, {heading}", begin, end, run)
            draw_off_curve(axes[row, col], name, run, ESN_COLORS[label], begin, end)
        axes[row, 0].set_ylabel("from the curve (mm)", color=TEXT_COLOR)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", color=TEXT_COLOR)
    return fig


def plot_long(name: str) -> Figure:
    """Claim 1's runs of 120 s: the distance from the curve over time, and the hand's last five loops."""
    fig = Figure(figsize=(15, 9.6), facecolor=SURFACE_COLOR, layout="constrained")
    period, loops = cycle(name).period, cycle(name).periods
    fig.suptitle(
        f"{TAKES[name]['label']}: undisturbed for 120 s from the eight starts on the curve."
        "\nAbove: the hand's distance from the curve, the ESN's arm (blue) over the replay's (orange); purple: 20 mm."
        f"\nBelow: the hand's last {loops} taught periods ({loops * period:g} s), over the curve (purple)",
        color=TITLE_COLOR,
    )
    grid = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.0])
    top, bottom = fig.add_subfigure(grid[0]), fig.add_subfigure(grid[1])
    above = top.subplots(1, len(ESNS), sharey=True)
    for ax, (label, n) in zip(above, ESNS.items(), strict=True):
        run = stem(name, "robot_esn", n)
        distance_axes(ax, name, f"{label} and the replay", -1.0, 120.0, run)
        draw_off_curve(ax, name, run, ESN_COLORS[label], -1.0, 120.0)
        ax.set_xlabel("time (s)", color=TEXT_COLOR)
    above[0].set_ylabel("from the curve (mm)", color=TEXT_COLOR)
    below = bottom.subplots(1, len(ESNS) + 1)
    arms = [(label, n, "esn", ESN_COLORS[label]) for label, n in ESNS.items()] + [
        ("the replay", "1", "replay", REPLAY_COLOR)
    ]
    for ax, (label, n, arm, color) in zip(below, arms, strict=True):
        workspace(ax, name, CURVE_VIEW, label)
        run = stem(name, "robot_esn", n)
        for start in column(rows(run, arm), "start"):
            state = log(run, arm, int(start))
            last = hand(state.channel("q")[state.times > state.times[-1] - loops * period - 1e-9])
            ax.plot(last[:, 0], last[:, 1], color=color, linewidth=0.8, alpha=0.7, zorder=3)
        ax.set_xlabel("x (m)", color=TEXT_COLOR)
    below[0].set_ylabel("y (m)", color=TEXT_COLOR)
    return fig


# ---------------------------------------------------------------------------- tables


def print_tables() -> None:
    for name in TAKES:
        print(f"\n######## {TAKES[name]['label']}")
        print_search(name)
        print_esns(name)
        print_on_curve(name)
        print_start(name)
        print_joining(name)
        print_pushes(name)
        print_block(name)
    print_checks()


def print_search(name: str) -> None:
    print("\n== The search")
    coarse = scores(stem(name, "coarse_search"))
    full = [row for row in coarse if row["cycling_runs"] == row["runs"]]
    none = [row for row in coarse if row["cycling_runs"] == 0]
    print(
        f"coarse search: {len(full)} of {len(coarse)} settings cycle from all {int(coarse[0]['runs'])} starts,"
        f" {len(none)} from none; the best, the center: {setting_text(coarse[0])},"
        f" median curve RMS {1e3 * coarse[0]['median_curve_rms_m']:.3f} mm"
    )
    for key in NAMES:
        counts = {v: sum(row[key] == v for row in full) for v in sorted({row[key] for row in coarse})}
        print(f"  settings that cycle from all starts, by {key}: {counts}")
    fine = scores(stem(name, "fine_search"))
    full = [row for row in fine if row["cycling_runs"] == row["runs"]]
    others = [row for row in fine if row["cycling_runs"] < row["runs"]]
    print(f"fine search: {len(full)} of {len(fine)} settings cycle in all {int(fine[0]['runs'])} runs; its values:")
    for key in NAMES:
        print(f"  {key}: {sorted({row[key] for row in fine})}")
    print(f"  the {len(others)} others, by leak rate and input scaling (of the settings with these values):")
    for leak in sorted({row["leak_rate"] for row in fine}):
        for scaling in sorted({row["input_scaling"] for row in fine}):
            pair = [row for row in fine if (row["leak_rate"], row["input_scaling"]) == (leak, scaling)]
            short = [row for row in pair if row["cycling_runs"] < row["runs"]]
            if short:
                print(f"    lr {leak:g}, is {scaling:g}: {len(short)} of {len(pair)}")
    for row in others:
        print(f"    {setting_text(row)}: {int(row['cycling_runs'])} of {int(row['runs'])}")
    rescored = scores(stem(name, "untaught_search"))
    best = fine[: len(rescored)]
    rms = column(best, "median_curve_rms_m")
    print(
        f"  the {len(best)} best, median curve RMS {1e3 * min(rms):.3f} to {1e3 * max(rms):.3f} mm, by leak rate and"
        " input scaling:"
    )
    pairs = sorted({(row["leak_rate"], row["input_scaling"]) for row in best})
    for leak, scaling in pairs:
        chosen = [row for row in best if (row["leak_rate"], row["input_scaling"]) == (leak, scaling)]
        print(
            f"    lr {leak:g}, is {scaling:g}: {len(chosen)}, sr {sorted({row['spectral_radius'] for row in chosen})}"
        )
    print(
        f"scoring with the untaught starts: {len(rescored)} settings, {int(rescored[0]['runs'])} runs each; the same"
        f" settings as the fine search's best: {[setting_of(r) for r in best] == [setting_of(r) for r in rescored]}"
        " (in its order)"
    )
    for rank, row in enumerate(rescored, start=1):
        print(
            f"  {rank:2d}. {setting_text(row)}: {int(row['cycling_runs'])} of {int(row['runs'])} cycle, median curve"
            f" RMS {1e3 * row['median_curve_rms_m']:.3f} mm, typical seed {int(row['typical_seed'])}"
        )
    seeds = table(stem(name, "untaught_search"), "sweep.csv")
    for row in seeds:
        if row["cycling_runs"] < row["runs"]:
            print(
                f"  not every run cycles: {setting_text(row)}, seed {int(row['seed'])}:"
                f" {int(row['cycling_runs'])} of {int(row['runs'])}"
            )
    for label, row in zip(ESNS, rescored, strict=False):
        own = [seed for seed in seeds if setting_of(seed) == setting_of(row)]
        each = ", ".join(
            f"{int(seed['seed'])}: {1e3 * seed['median_curve_rms_m']:.4f} mm ({int(seed['cycling_runs'])})"
            for seed in own
        )
        print(
            f"  {label}: {setting_text(row)}, the setting's median {1e3 * row['median_curve_rms_m']:.4f} mm; each"
            f" seed's (runs that cycle): {each}; typical seed {int(row['typical_seed'])}"
        )


def print_esns(name: str) -> None:
    print("\n== The chosen ESNs")
    for label, n in ESNS.items():
        run = esn_run(name, n)
        esn = config(run)["esn"]
        with (RESULTS / run / "esn.toml").open("rb") as f:
            saved = tomllib.load(f)["esn"]
        swept = ", ".join(f"{key} {esn[key]:g}" for key in (*NAMES, "seed"))
        fixed = ", ".join(f"{key} {esn[key]}" for key in esn if key not in (*NAMES, "seed"))
        extra = ", ".join(f"{key} {saved[key]}" for key in saved if key not in esn)
        demonstrations = config(run)["demonstrations"]
        print(f"{label}, run {run}:")
        print(f"  swept: {swept}")
        print(f"  fixed: {fixed}; and in esn.toml: {extra}")
        print(f"  trained on {demonstrations['run']}: {', '.join(demonstrations['train'])}")
    measured = cycle(name)
    first, last = measured.loop
    q = setup(name).demos[TAKES[name]["train"].split(".")[0]]
    step = np.degrees(np.linalg.norm(q[last] - q[first]))
    print(
        f"the take: taught period {measured.period:.3f} s; the replay's loop from {DT * first:.2f} s to"
        f" {DT * last:.2f} s, the step at its seam {step:.2g} deg"
    )
    tracker = config(run_name(stem(name, "robot_esn", "1")))["tracker"]
    print(
        f"the tracker: {tracker['laws']}, omega {tracker['omegas']} rad/s, damping ratio {tracker['dampings']},"
        f" reference velocity {tracker['reference_velocity']}, control period {tracker['control_period']} s"
    )


def print_on_curve(name: str) -> None:
    print("\n== Claim 1: undisturbed for 120 s from the eight starts on the curve")
    for label, n in ESNS.items():
        run = stem(name, "robot_esn", n)
        for arm in ("esn", "replay"):
            runs = rows(run, arm)
            print(
                f"{label} {arm:6s}: cycle {sum(row['cycles'] for row in runs)} of {len(runs)}, curve RMS"
                f" {spread(column(runs, 'curve_rms_m'), '.3f', 1e3)} mm, period ratio"
                f" {spread(column(runs, 'period_ratio'), '.4f')}, largest off the curve"
                f" {1e3 * max(column(runs, 'off_curve_max_m')):.2f} mm"
            )


def print_start(name: str) -> None:
    print("\n== Claim 2: how a run starts, the reference's fastest joint speed and the largest joint torque")
    groups = [
        ("the take's own start", "robot_esn", lambda row: row["start"] == 0),
        ("the seven other starts on the curve", "robot_esn", lambda row: row["start"] != 0),
        ("the 36 untaught starts", "untaught_robot_esn", lambda row: True),
    ]
    for label, n in ESNS.items():
        for heading, base, pick in groups:
            run = stem(name, base, n)
            esn = [row for row in rows(run, "esn") if pick(row)]
            replay = [row for row in rows(run, "replay") if pick(row)]
            print(f"{label}, {heading} ({len(esn)}):")
            for arm, runs in (("esn", esn), ("replay", replay)):
                print(
                    f"  {arm:6s}: speed {spread(column(runs, 'peak_reference_speed_dps'), '.0f')} deg/s, torque"
                    f" {spread(column(runs, 'peak_torque_nm'), '.1f')} N m"
                )
            print(
                f"  the ESN's smaller than the replay's: speed in {smaller(esn, replay, 'peak_reference_speed_dps')},"
                f" torque in {smaller(esn, replay, 'peak_torque_nm')}"
            )
    shown = shown_untaught(name)
    torque = next(row for row in rows(stem(name, "untaught_robot_esn", "1"), "replay") if row["start"] == shown)
    print(
        f"the untaught start drawn over time: {shown}, the replay's largest torque {torque['peak_torque_nm']:.1f} N m"
    )


def print_joining(name: str) -> None:
    print("\n== Claim 3: undisturbed for 60 s from the 36 untaught starts")
    for label, n in ESNS.items():
        run = stem(name, "untaught_robot_esn", n)
        for arm in ("esn", "replay"):
            runs = rows(run, arm)
            print(
                f"{label} {arm:6s}: cycle {sum(row['cycles'] for row in runs)} of {len(runs)}, joining time"
                f" {spread(column(runs, 'joining_time_s'), '.2f')} s, curve RMS"
                f" {spread(column(runs, 'curve_rms_m'), '.3f', 1e3)} mm, period ratio"
                f" {spread(column(runs, 'period_ratio'), '.4f')}"
            )


def print_pushes(name: str) -> None:
    print("\n== Claim 4: under the three pushes for 80 s from the eight starts on the curve")
    back: dict[str, list[float]] = {}
    delays: dict[str, list[float]] = {}
    for times, size in SIZES.items():
        for label, n in ESNS.items():
            base = f"pushes{times}x_robot_esn"
            run = stem(name, base, n)
            for arm in ("esn", "replay"):
                runs = rows(run, arm)
                late = [delay(name, base, n, arm, int(row["start"])) for row in runs]
                who = label if arm == "esn" else "the replay"
                back.setdefault(who, []).extend(v for k in (1, 2, 3) for v in column(runs, f"return_time_s_{k}"))
                delays.setdefault(who, []).extend(late)
                print(
                    f"x{times} ({size}) {label} {arm:6s}: cycle {sum(row['cycles'] for row in runs)} of {len(runs)},"
                    f" curve RMS {spread(column(runs, 'curve_rms_m'), '.3f', 1e3)} mm, delay {spread(late, '+.2f')} s"
                )
                for k in (1, 2, 3):
                    print(
                        f"    push {k}: off the curve {spread(column(runs, f'off_path_m_{k}'), '.1f', 1e3)} mm, back"
                        f" within 20 mm in {spread(column(runs, f'return_time_s_{k}'), '.2f')} s, largest torque"
                        f" after {spread(column(runs, f'torque_after_nm_{k}'), '.1f')} N m"
                    )
    for who in back:
        print(
            f"{who}, over every push and size: back within 20 mm in {np.nanmax(back[who]):.2f} s at most; delay"
            f" {min(delays[who]):+.2f} to {max(delays[who]):+.2f} s"
        )


def print_block(name: str) -> None:
    print("\n== Claim 5: the tip held from 10 s to 13 s, for 60 s from the eight starts on the curve")
    for label, n in ESNS.items():
        base = "block_robot_esn"
        run = stem(name, base, n)
        esn, replay = rows(run, "esn"), rows(run, "replay")
        for arm, runs in (("esn", esn), ("replay", replay)):
            late = [delay(name, base, n, arm, int(row["start"])) for row in runs]
            print(
                f"{label} {arm:6s}: cycle {sum(row['cycles'] for row in runs)} of {len(runs)}; in the 2 s after the"
                f" release: largest torque {spread(column(runs, 'torque_after_nm'), '.1f')} N m, largest off the curve"
                f" {spread(column(runs, 'off_curve_after_m'), '.1f', 1e3)} mm; off the curve after the onset"
                f" {spread(column(runs, 'off_path_m'), '.1f', 1e3)} mm; back within 20 mm in"
                f" {spread(column(runs, 'return_time_s'), '.2f')} s; delay {spread(late, '+.2f')} s"
            )
        print(
            f"  the ESN's smaller than the replay's: largest torque after in {smaller(esn, replay, 'torque_after_nm')},"
            f" largest off the curve after in {smaller(esn, replay, 'off_curve_after_m')}"
        )


def print_checks() -> None:
    print("\n== Checks")
    commits: dict[tuple[str, bool], list[str]] = {}
    for path in sorted(p for p in RESULTS.iterdir() if p.is_dir() and p.name != "summary"):
        with (path / "run.toml").open("rb") as f:
            record = tomllib.load(f)
        commits.setdefault((record["commit"][:7], record["uncommitted_changes"]), []).append(path.name)
    for (commit, dirty), runs in commits.items():
        print(f"commit {commit}{' WITH UNCOMMITTED CHANGES' if dirty else ''}: {', '.join(runs)}")
    assert not any(dirty for _, dirty in commits)
    counts = {"robot_esn": ON_CURVE, "untaught_robot_esn": 36, "block_robot_esn": ON_CURVE}
    counts |= {f"pushes{times}x_robot_esn": ON_CURVE for times in SIZES}
    for name, spec in TAKES.items():
        for n in ESNS.values():
            training = config(esn_run(name, n))["demonstrations"]
            assert Path(training["run"]).name == spec["demonstrations"] and training["train"] == [spec["train"]]
            for base, count in counts.items():
                run = stem(name, base, n)
                assert config(run_name(run))["esn"]["model"].endswith(f"{esn_run(name, n)}/esn.toml")
                assert [len(rows(run, arm)) for arm in ("esn", "replay")] == [count, count]
        same = [
            np.allclose(
                column(rows(stem(name, base, "1"), "replay"), key), column(rows(stem(name, base, "2"), "replay"), key)
            )
            for base in counts
            for key in ("curve_rms_m", "peak_torque_nm", "off_curve_max_m")
        ]
        print(f"{name}: each ESN's runs and its training take as named; runs from each start: {counts}")
        print(f"{name}: the replay beside ESN #1 equals the one beside ESN #2 in every scenario: {all(same)}")
        trained = setup(name).demos[spec["train"].split(".")[0]]
        assert np.array_equal(trained, take(name))
        print(
            f"{name}: the take under data/ is the one the ESNs were trained on;"
            f" taught period {cycle(name).period:.3f} s"
        )
        block = stem(name, "block_robot_esn", "1")
        first = log(block, "replay", 0)
        later = first.times >= 10.0
        gap = max(
            float(np.abs(log(block, "replay", start).channel("q")[later] - first.channel("q")[later]).max())
            for start in range(1, ON_CURVE)
        )
        print(
            f"{name}: the block's replay runs differ from start 0's from 10 s on by at most {np.degrees(gap):.2g} deg"
        )


if __name__ == "__main__":
    main()
