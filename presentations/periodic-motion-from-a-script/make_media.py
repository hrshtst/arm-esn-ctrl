# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Videos and figures of report 026, a periodic motion from a script, for slides.

    uv run python presentations/periodic-motion-from-a-script/make_media.py [--workers 8] [--only FOLDER ...]
        [--starts N ...]

From the scripted take and the robot runs of report 026
(``reports/026-a-periodic-motion-from-a-script/README.md``, Sections 2 and 3.2 to 3.6),
under the storage root, it writes into ``presentations/periodic-motion-from-a-script/``
there a folder per section:

- ``00_take``: the curve and the start postures, and the take's video, joint angles, and
  hand path;
- ``3.2_on_curve``: undisturbed from the eight starts on the curve: the hand paths of
  ESN #1 and ESN #2 against the replay's, the hand's distance from the curve over the
  120 s and its last five loops, each run, and each arm's runs one after another;
- ``3.3_untaught_starts``: undisturbed from the 36 untaught starts: the hand paths, the
  distance from the curve over the first seconds, each run, and each arm's runs one after
  another;
- ``3.5_pushes``: under the pushes, a folder per size: the distance from the curve, each
  run, and each arm's runs one after another;
- ``3.6_block``: with the tip held, likewise.

A run's files are its video in skelarm's player (``--export --panel``), its joint angles
and torques over time, and its hand path. The runs last a minute or two, so each shows a
window of its run (the ``*_WINDOW`` constants): its video is the player's export of a copy
of the log cut to the window. The videos of an arm's runs one after another are made from
those videos, with the speed, the arm, and the start written in the player's side panel.
The folder's ``README.md`` explains them, and ``media.csv`` lists every file with the logs
it shows. The replays do not depend on the ESN: they are taken from ESN #1's runs, checked
to equal ESN #2's.

``--workers`` sets the number of processes that render. ``--only`` renders the named
folders alone, and ``--starts`` the runs from the named starts alone, by their index, both
in the run folders and in the videos of all starts; the README and ``media.csv`` are always
written in full. It overwrites its own files and deletes nothing.
"""

from __future__ import annotations

import argparse
import csv
import functools
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable, Collection
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, NamedTuple

import imageio.v2 as imageio
import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from skelarm import StateLog, compute_forward_kinematics

from arm_esn_ctrl.autonomous import Cycle, load_setup, task_cycle
from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.metrics import distances_to_path
from arm_esn_ctrl.storage import REPO_ROOT, resolve_run_path, storage_root

OUT = storage_root() / "presentations" / "periodic-motion-from-a-script"
EXPERIMENT = "results/manual_demonstration_v2_periodic"
TAKE_RUN = "results/demonstrations/20261118-171132-ellipse_scripted_v2_no_limits"
TAKE = "demo_00.sklog.npz"  # the take as made, which the ESNs were trained on
PREFIX = "scripted_"  # the scripted take's runs' configurations begin with it
PLAYER = REPO_ROOT / "third_party" / "skelarm" / "tools" / "player.py"
TRACKER = "pd_w20_z0.4"  # the one tracker's directory in a robot run
FPS = 30.0
SPEEDS = (1,)  # the speeds of the videos of an arm's runs one after another
CUT = 0.05  # s: the torque axes leave out the first 50 ms after t = 0, as report 023's figures do

# The windows of the runs that a run's video and figures show (s), on the task's clock: the warm-up from -1 s, the
# task from 0.
ON_CURVE_WINDOW = (-1.0, 12.0)  # undisturbed from the starts on the curve: the start and three loops
UNTAUGHT_WINDOW = (-1.0, 8.0)  # undisturbed from the untaught starts: joining the curve and two loops
PUSHES_WINDOW = (6.0, 40.0)  # the three pushes, at 8.0, 20.5, and 33.0 s, and 7 s after the last
BLOCK_WINDOW = (8.0, 20.0)  # the tip held from 10 s to 13 s, and 7 s after
BLOCK_PATHS = (10.0, 17.0)  # the block's hand paths: from the hold's onset to 4 s after its release
UNTAUGHT_DISTANCE = (0.0, 6.0)  # the distance from the curve from the untaught starts, as in report 026

# The stretches of the workspace drawn, x and y (m): all of it; the curve and the paths from the starts on it; the
# take's posture; the curve alone; the paths under the pushes of every size; and the paths with the block.
View = tuple[tuple[float, float], tuple[float, float]]
WHOLE_VIEW: View = ((-0.95, 0.95), (-0.95, 0.95))
STARTS_VIEW: View = ((-0.35, 0.45), (0.37, 0.73))
TAKE_VIEW: View = ((-0.4, 0.5), (-0.05, 0.75))
CURVE_VIEW: View = ((-0.34, 0.14), (0.40, 0.70))
PUSHES_VIEW: View = ((-0.45, 0.65), (0.0, 0.9))
BLOCK_VIEW: View = ((-0.36, 0.18), (0.34, 0.75))

FOLDERS = ("00_take", "3.2_on_curve", "3.3_untaught_starts", "3.5_pushes", "3.6_block")
# The arms: their label, the ESN whose robot runs hold them, and their name in those runs.
ARMS = {"esn1": ("ESN #1", "1", "esn"), "esn2": ("ESN #2", "2", "esn"), "replay": ("Replay", "1", "replay")}
SIZES = {1: "pushes at report 007's forces", 2: "pushes at twice the forces", 4: "pushes at four times the forces"}
# The scenarios of the robot runs: the configuration without the take's prefix and the ESN's number.
SCENARIOS = ("robot_esn", "untaught_robot_esn", *(f"pushes{times}x_robot_esn" for times in SIZES), "block_robot_esn")

# Report 026's colors.
TEXT_COLOR, TITLE_COLOR, GRID_COLOR, SURFACE_COLOR = "#52514e", "#0b0b0b", "#e1e0d9", "#fcfcfb"
DISTURBANCE_COLOR, TAUGHT_COLOR = "#e8e6df", "#b9b7b0"
ESN_COLOR, REPLAY_COLOR, GOAL_COLOR = "#2a78d6", "#eb6834", "#7b3fa0"
COLORS = {"esn1": ESN_COLOR, "esn2": ESN_COLOR, "replay": REPLAY_COLOR}

DPI = 150
TWO_PANELS = (12.0, 6.4)  # 1800 x 960 px
WIDE = (12.0, 6.75)  # a figure over time: 1800 x 1012 px, 16:9
RUN = (10.0, 5.625)  # a run's own figures: 1500 x 843 px, 16:9
STARTS_SIZE = (12.0, 7.6)  # the start postures: 1800 x 1140 px
CURVE_PATHS = (12.0, 4.6)  # two panels of the paths from the starts on the curve: 1800 x 690 px
LOOPS = (16.0, 5.1)  # three panels of the last loops: 2400 x 765 px
matplotlib.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 13,
        "figure.titlesize": 16,
    }
)
# Where a video of runs one after another writes the speed, the arm, and the start: in the player's side panel, right
# of its 800 px canvas and under its readouts (px).
MARK_AT, ARM_AT, LABEL_AT = (822, 250), (822, 410), (822, 510)


class Medium(NamedTuple):
    """A file of the output: its path in the output folder, what it shows, the logs it shows, how it is made, and the
    files of the output it is made from, which are made before it."""

    path: str
    shows: str
    logs: list[Path]
    make: Callable[..., None]  # called with ``out``, the file to write
    reads: tuple[str, ...] = ()


# ---------------------------------------------------------------------------- the runs


def take_path() -> Path:
    return resolve_run_path(f"{TAKE_RUN}/{TAKE}")


with resolve_run_path(f"{TAKE_RUN}/config.toml").open("rb") as _f:
    TAKE_CONFIG = tomllib.load(_f)  # the take's configuration, with its robot, its curve, and its simulation step
SKELETON = StateLog.load(take_path()).build_skeleton()


@functools.cache
def run_dir(stem: str) -> Path:
    """The one run of a configuration of the experiment, under the storage root."""
    pattern = re.compile(rf"\d{{8}}-\d{{6}}-{re.escape(stem)}")
    matches = sorted(p for p in resolve_run_path(EXPERIMENT).iterdir() if p.is_dir() and pattern.fullmatch(p.name))
    if len(matches) != 1:
        msg = f"expected one run of {stem}, found {[p.name for p in matches]}"
        raise FileNotFoundError(msg)
    return matches[0]


def robot_run(scenario: str, n: str) -> str:
    """A robot run of ESN #``n``, by its configuration: ``scenario`` with the take's prefix and the ESN's number."""
    return f"{PREFIX}{scenario}{n}"


@functools.cache
def config(stem: str) -> dict[str, Any]:
    """The configuration a run recorded."""
    with (run_dir(stem) / "config.toml").open("rb") as f:
        return tomllib.load(f)


@functools.cache
def rows(stem: str) -> list[dict[str, str]]:
    with (run_dir(stem) / "metrics.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def arm_rows(stem: str, arm: str) -> list[dict[str, str]]:
    """One arm's runs of a robot run, in the order of its starts."""
    return [row for row in rows(stem) if row["arm"] == arm]


def log_path(stem: str, arm: str, start: int) -> Path:
    return run_dir(stem) / TRACKER / f"{arm}_{start:02d}.sklog.npz"


def arm_log(scenario: str, name: str, start: int) -> Path:
    """The log of an arm of ``ARMS`` in a scenario, from a start."""
    _, n, arm = ARMS[name]
    return log_path(robot_run(scenario, n), arm, start)


def starts_of(scenario: str) -> list[int]:
    """A scenario's starts, by their index in its runs."""
    return [int(row["start"]) for row in arm_rows(robot_run(scenario, "1"), "replay")]


@functools.cache
def esn_config(n: str) -> dict[str, Any]:
    """The configuration of the run that trained ESN #``n``, as its undisturbed robot run names it."""
    with (resolve_run_path(config(robot_run("robot_esn", n))["esn"]["model"]).parent / "config.toml").open("rb") as f:
        return tomllib.load(f)


@functools.cache
def esn_of(n: str) -> dict[str, Any]:
    """The ``[esn]`` table of ESN #``n``, as saved with it."""
    with resolve_run_path(config(robot_run("robot_esn", n))["esn"]["model"]).open("rb") as f:
        return tomllib.load(f)["esn"]


@functools.cache
def cycle() -> Cycle:
    """What the runs are measured against, as report 026 measures them: the curve, its center, the taught period, the
    replay's loop, and the 20 mm."""
    esn = esn_config("1")
    measured = task_cycle(load_setup(esn), esn["evaluation"])
    assert measured is not None
    return measured


@functools.cache
def take() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The take's times (s) and joint angles (rad), as made."""
    state = StateLog.load(take_path())
    return state.times, state.channel("q")


def posture_of(stem: str, origin: str) -> NDArray[np.float64]:
    """The start posture of a run, by its origin in the run's ``metrics.csv`` (deg): the take's start, or a start of
    the configuration's ``[evaluation.extra_starts]``."""
    if origin == "demo_00":
        return np.degrees(take()[1][0])
    group, index = origin.split()
    return np.array(config(stem)["evaluation"]["extra_starts"][group][int(index)])


def curve_angle(posture: NDArray[np.float64]) -> int:
    """The curve's angle of the hand at a posture (deg), rounded: 0 at the curve's right end, counterclockwise."""
    task = TAKE_CONFIG["task"]
    (x, y), (cx, cy) = hand(np.radians(posture)[None])[0], task["center"]
    return round(float(np.degrees(np.arctan2((y - cy) / task["b"], (x - cx) / task["a"]))) % 360)


def start_names(scenario: str, start: int) -> tuple[str, str, str]:
    """A start as a folder name, a phrase, and a label in a video's side panel: ``start_3``, "from start 3 on the curve
    (135°)", "start 3 (135°)" on two lines; or ``untaught_14``, "from untaught start 14 (-53°, 60°)", "untaught
    start 14" on two lines."""
    stem = robot_run(scenario, "1")
    (origin,) = [row["origin"] for row in arm_rows(stem, "replay") if int(row["start"]) == start]
    posture = posture_of(stem, origin)
    if origin.startswith("untaught"):
        q1, q2 = (f"{q:g}".replace("-", "\N{MINUS SIGN}") for q in posture)
        return (
            f"untaught_{start:02d}",
            f"from untaught start {start:02d} ({q1}°, {q2}°)",
            f"untaught\nstart {start:02d}",
        )
    angle = curve_angle(posture)
    return f"start_{start}", f"from start {start} on the curve ({angle}°)", f"start {start}\n({angle}°)"


def spans(stem: str) -> list[tuple[float, float]]:
    """When each disturbance of a robot run acts, from its onset to its end (s), as its configuration sets them."""
    return [
        (d["onset"], d["release"] if d["type"] == "block" else d["onset"] + d["duration"]) for d in disturbances(stem)
    ]


def disturbances(stem: str) -> list[dict[str, Any]]:
    """The ``[disturbance]`` tables of a robot run's configuration: the pushes, the block, or none."""
    tables = config(stem).get("disturbance", [])
    return tables if isinstance(tables, list) else [tables]


def shade_name(stem: str) -> str:
    """What the shaded stretches of a robot run's figures are, in a legend: a push or the tip held."""
    kinds = {d["type"] for d in disturbances(stem)}
    return "" if not kinds else "the tip held" if kinds == {"block"} else "a push"


def check_runs() -> None:
    """Every run is there; the ESNs were trained on the take; the robot runs are of these ESNs and from the same starts;
    and the replays beside ESN #1 equal those beside ESN #2."""
    for n in "12":
        train = esn_config(n)["demonstrations"]
        if Path(train["run"]).name != Path(TAKE_RUN).name or train["train"] != [TAKE]:
            msg = f"ESN #{n} was trained on {train}, not on {TAKE_RUN}/{TAKE}"
            raise ValueError(msg)
    for scenario in SCENARIOS:
        first, second = (robot_run(scenario, n) for n in "12")
        for n, stem in zip("12", (first, second), strict=True):
            if config(stem)["esn"]["model"] != config(robot_run("robot_esn", n))["esn"]["model"]:
                msg = f"{stem} is not of ESN #{n}"
                raise ValueError(msg)
        if config(first)["evaluation"]["extra_starts"] != config(second)["evaluation"]["extra_starts"]:
            msg = f"the starts of {first} and {second} differ"
            raise ValueError(msg)
        starts = [(row["start"], row["origin"]) for row in arm_rows(first, "replay")]
        for stem in (first, second):
            for arm in ("esn", "replay"):
                if [(row["start"], row["origin"]) for row in arm_rows(stem, arm)] != starts:
                    msg = f"the {arm} runs of {stem} are not from the starts of {first}'s replay"
                    raise ValueError(msg)
        for start, _ in starts:
            a, b = (StateLog.load(log_path(stem, "replay", int(start))).channel("q") for stem in (first, second))
            if not np.array_equal(a, b):
                msg = f"the replay runs of {first} and {second} from start {start} differ"
                raise ValueError(msg)
    on_curve = [
        config(robot_run(scenario, "1"))["evaluation"]["extra_starts"].get("on_curve") for scenario in SCENARIOS
    ]
    if any(starts != on_curve[0] for starts in on_curve if starts is not None):
        msg = "the runs' starts on the curve differ"
        raise ValueError(msg)


# ---------------------------------------------------------------------------- measures


def hand(q: NDArray[np.float64]) -> NDArray[np.float64]:
    return endpoint_positions(SKELETON, q)


def posture(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """The arm's base, elbow, and hand at joint angles ``q`` (rad): (x, y) per point."""
    for link, angle in zip(SKELETON.links[1:], q, strict=True):
        link.q = float(angle)
    compute_forward_kinematics(SKELETON)
    return np.array([[0.0, 0.0], *[[link.xe, link.ye] for link in SKELETON.links[1:]]])


def within(state: StateLog, window: tuple[float, float]) -> NDArray[np.bool_]:
    """The samples of a log within a window of its times (s)."""
    return (state.times >= window[0] - 1e-9) & (state.times <= window[1] + 1e-9)


def off_curve(state: StateLog, window: tuple[float, float]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The times (s) of a log within a window and the hand's distance from the curve then (mm)."""
    shown = within(state, window)
    return state.times[shown], 1e3 * distances_to_path(hand(state.channel("q")[shown]), cycle().curve)


def pushed(state: StateLog) -> NDArray[np.bool_]:
    """The samples of a log at which a force acts on the tip."""
    if "ext_force" not in state.channel_names:
        return np.zeros(len(state.times), dtype=bool)
    return np.linalg.norm(state.channel("ext_force"), axis=1) > 0.0


def cut(state: StateLog, window: tuple[float, float]) -> StateLog:
    """A copy of a log over a window of its times (s): its robot, producer, channel descriptors, and extra metadata,
    the task the player draws among them, and each sample within the window, at its own time, with every channel."""
    copy = StateLog(state.build_skeleton(), producer=state.producer, channel_meta=state.channel_meta, extra=state.extra)
    times = state.times
    channels = {name: state.channel(name) for name in state.channel_names}
    for index in np.flatnonzero(within(state, window)):
        copy.record(float(times[index]), **{name: values[index] for name, values in channels.items()})
    return copy


# ---------------------------------------------------------------------------- the figures


def figure(size: tuple[float, float]) -> Figure:
    return Figure(figsize=size, dpi=DPI, facecolor=SURFACE_COLOR, layout="constrained")


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def workspace(ax: Axes, view: View, title: str = "") -> None:
    """The workspace's axes over ``view``: the rings that bound where the hand reaches, the curve, the base, and equal
    scales."""
    style(ax)
    upper, forearm = (link.prop.length for link in SKELETON.links[1:])
    for radius in (abs(upper - forearm), upper + forearm):
        ax.add_patch(Circle((0.0, 0.0), radius, fill=False, edgecolor=GRID_COLOR, linewidth=1.4, zorder=1))
    curve = cycle().curve
    ax.plot(curve[:, 0], curve[:, 1], color=GOAL_COLOR, linewidth=2.0, zorder=2)
    ax.plot(0.0, 0.0, marker="s", color=TEXT_COLOR, markersize=6, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(*view[0])
    ax.set_ylim(*view[1])
    if view == WHOLE_VIEW:
        ax.set_xticks([-0.5, 0.0, 0.5])  # every 0.25 m, the labels touch at a slide's font size
        ax.set_yticks([-0.5, 0.0, 0.5])
    ax.set_xlabel("x (m)", color=TEXT_COLOR)
    ax.set_ylabel("y (m)", color=TEXT_COLOR)
    if title:
        ax.set_title(title, color=TITLE_COLOR)


def draw_path(
    ax: Axes,
    state: StateLog,
    window: tuple[float, float],
    color: str,
    *,
    linewidth: float = 1.2,
    dot: float = 5.0,
    mark_end: bool = False,
) -> None:
    """A hand path over a window of a log, its first point as a dot, thick black where a force acts, and with
    ``mark_end`` a cross at its end."""
    shown = within(state, window)
    points = hand(state.channel("q")[shown])
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=linewidth, alpha=0.85, zorder=3)
    ax.plot(*points[0], marker="o", color=color, markersize=dot, zorder=4)
    forced = np.where(pushed(state)[shown][:, None], points, np.nan)
    ax.plot(forced[:, 0], forced[:, 1], color=TITLE_COLOR, linewidth=linewidth + 1.8, zorder=3.5)
    if mark_end:
        ax.plot(*points[-1], marker="x", color=TITLE_COLOR, markersize=dot + 3, markeredgewidth=2.0, zorder=4.5)


def path_handles(*, forced: bool) -> list[Line2D]:
    """The legend of hand paths: the start of the window, the force, its end, and the curve."""
    handles = [Line2D([], [], linestyle="", marker="o", color=TEXT_COLOR, label="start")]
    if forced:
        handles += [
            Line2D([], [], color=TITLE_COLOR, linewidth=3.0, label="while a force acts"),
            Line2D([], [], linestyle="", marker="x", markeredgewidth=2.0, color=TITLE_COLOR, label="end"),
        ]
    handles.append(Line2D([], [], color=GOAL_COLOR, linewidth=2.0, label="the curve"))
    return handles


def note(ax: Axes, text: str) -> None:
    """A short note in an axes' upper right corner."""
    ax.text(
        0.99,
        0.97,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=TEXT_COLOR,
        fontsize=12,
        bbox={"facecolor": SURFACE_COLOR, "edgecolor": "none", "alpha": 0.85},
        zorder=6,
    )


def starts_figure(*, out: Path) -> None:
    """The curve, the take's hand path, the eight starts on the curve numbered, and the 36 untaught starts."""
    stem = robot_run("robot_esn", "1")
    fig = figure(STARTS_SIZE)
    ax = fig.subplots()
    workspace(ax, WHOLE_VIEW)
    points = hand(take()[1])
    ax.plot(points[:, 0], points[:, 1], color=TAUGHT_COLOR, linewidth=6.0, zorder=1.5)
    center = cycle().center
    for row in arm_rows(stem, "replay"):
        point = hand(np.radians(posture_of(stem, row["origin"]))[None])[0]
        ax.plot(*point, marker="o", color=TITLE_COLOR, markersize=7, zorder=4)
        x, y = 15 * (center - point) / np.linalg.norm(center - point)  # inward, toward the center (points)
        ax.annotate(row["start"], point, textcoords="offset points", xytext=(x, y), ha="center", va="center")
    untaught_stem = robot_run("untaught_robot_esn", "1")
    untaught = hand(np.radians(config(untaught_stem)["evaluation"]["extra_starts"]["untaught"]))
    ax.plot(untaught[:, 0], untaught[:, 1], linestyle="", marker="s", color=REPLAY_COLOR, markersize=7, zorder=3)
    handles = [
        Line2D([], [], color=GOAL_COLOR, linewidth=2.0, label="the curve"),
        Line2D([], [], color=TAUGHT_COLOR, linewidth=6.0, label="the take"),
        Line2D([], [], linestyle="", marker="o", color=TITLE_COLOR, label="start on the curve"),
        Line2D([], [], linestyle="", marker="s", color=REPLAY_COLOR, label="untaught start"),
    ]
    fig.legend(handles=handles, loc="outside right center", frameon=False, labelcolor=TEXT_COLOR)
    fig.savefig(out)


def take_joints(*, out: Path) -> None:
    """The take's two joint angles over time."""
    times, q = take()
    fig = figure(RUN)
    axes = fig.subplots(2, 1, sharex=True)
    axes[0].set_title("The scripted take", color=TITLE_COLOR)
    for joint, ax in enumerate(axes):
        style(ax)
        ax.plot(times, np.degrees(q[:, joint]), color=TITLE_COLOR, linewidth=2.0)
        ax.set_xlim(times[0], times[-1])
        ax.set_ylabel(f"joint {joint + 1} (deg)", color=TEXT_COLOR)
    axes[-1].set_xlabel("time (s)", color=TEXT_COLOR)
    fig.savefig(out)


def take_tip(*, out: Path) -> None:
    """The take's hand path over the curve, with its start posture."""
    _, q = take()
    fig = figure(RUN)
    ax = fig.subplots()
    fig.suptitle("The scripted take", color=TITLE_COLOR)
    workspace(ax, TAKE_VIEW)
    ax.plot(*posture(q[0]).T, color="#9a9890", linewidth=4.0, marker="o", markersize=7, zorder=3)
    points = hand(q)
    ax.plot(points[:, 0], points[:, 1], color=TAUGHT_COLOR, linewidth=7.0, zorder=1.5)  # under the curve
    ax.plot(*points[0], marker="o", color=TITLE_COLOR, markersize=8, zorder=4)
    handles = [
        Line2D([], [], color=TAUGHT_COLOR, linewidth=7.0, label="the take's hand path"),
        Line2D([], [], linestyle="", marker="o", color=TITLE_COLOR, label="its start"),
        Line2D([], [], color="#9a9890", linewidth=4.0, marker="o", label="start posture"),
        Line2D([], [], color=GOAL_COLOR, linewidth=2.0, label="the curve"),
    ]
    fig.legend(handles=handles, loc="outside right center", frameon=False, labelcolor=TEXT_COLOR)
    fig.savefig(out)


def run_series(
    log: Path,
    window: tuple[float, float],
    shaded: list[tuple[float, float]],
    shade: str,
    title: str,
    color: str,
    channel: str,
    *,
    out: Path,
) -> None:
    """A run's two joint angles (``joints``), its reference dashed, or its two torques (``torques``) over a window, the
    disturbances shaded and named ``shade`` in the legend.

    A torque axis spans the largest torque over the window, but for the first ``CUT`` after t = 0, and at least
    0.5 N m; a larger torque in that first moment is written on the axes.
    """
    state = StateLog.load(log)
    shown = within(state, window)
    times = state.times[shown]
    joints = channel == "joints"
    values = np.degrees(state.channel("q")[shown]) if joints else state.channel("tau")[shown]
    kept = (times < 0.0) | (times >= CUT)
    fig = figure(RUN)
    axes = fig.subplots(2, 1, sharex=True)
    axes[0].set_title(title, color=TITLE_COLOR)
    for joint, ax in enumerate(axes):
        style(ax)
        for onset, end in shaded:
            ax.axvspan(onset, end, color=DISTURBANCE_COLOR, zorder=0)
        ax.plot(times, values[:, joint], color=color, linewidth=1.8, zorder=3)
        if joints:
            reference = np.degrees(state.channel("q_ref")[shown, joint])
            ax.plot(times, reference, color=color, linewidth=1.4, linestyle="--", zorder=2.5)
        else:
            top = max(1.15 * float(np.abs(values[kept, joint]).max()), 0.5)
            ax.set_ylim(-top, top)
            peak = float(np.abs(values[:, joint]).max())
            if peak > top:
                note(ax, f"off the axis in the first {1e3 * CUT:g} ms: up to {peak:.1f} N m")
        ax.set_xlim(*window)
        ax.set_ylabel(f"joint {joint + 1} ({'deg' if joints else 'N m'})", color=TEXT_COLOR)
    axes[-1].set_xlabel("time (s)", color=TEXT_COLOR)
    handles: list[Line2D | Patch] = [Line2D([], [], color=color, linewidth=1.8, label="the arm")]
    if joints:
        handles.append(Line2D([], [], color=color, linewidth=1.4, linestyle="--", label="its reference"))
    if shaded:
        handles.append(Patch(facecolor=DISTURBANCE_COLOR, label=shade))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    fig.savefig(out)


def run_tip(log: Path, window: tuple[float, float], view: View, title: str, color: str, *, out: Path) -> None:
    """A run's hand path over a window, over the curve."""
    state = StateLog.load(log)
    forced = bool(pushed(state)[within(state, window)].any())
    fig = figure(RUN)
    ax = fig.subplots()
    fig.suptitle(f"{title},\nthe hand's path {window_text(window)}", color=TITLE_COLOR)
    workspace(ax, view)
    draw_path(ax, state, window, color, linewidth=2.2, dot=8.0, mark_end=forced)
    handles = [Line2D([], [], color=color, linewidth=2.2, label="the hand's path"), *path_handles(forced=forced)]
    fig.legend(handles=handles, loc="outside right center", frameon=False, labelcolor=TEXT_COLOR)
    fig.savefig(out)


def block_tip(log: Path, title: str, color: str, reference: bool, *, out: Path) -> None:
    """A run's hand path with the block over ``BLOCK_PATHS``, a cross where the tip is held, and with ``reference``
    the reference's hand path over the hold dashed, as report 026 draws them from start 0."""
    state = StateLog.load(log)
    (hold,) = spans(robot_run("block_robot_esn", "1"))  # the same in every run
    fig = figure(RUN)
    ax = fig.subplots()
    fig.suptitle(f"{title},\nthe hand's path {window_text(BLOCK_PATHS)}", color=TITLE_COLOR)
    workspace(ax, BLOCK_VIEW)
    points = hand(state.channel("q")[within(state, BLOCK_PATHS)])
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=2.2, zorder=3)
    ax.plot(*points[0], marker="X", color=color, markeredgecolor=TITLE_COLOR, markersize=13, zorder=5)
    handles = [
        Line2D([], [], color=color, linewidth=2.2, label="the hand's path"),
        Line2D([], [], linestyle="", marker="X", color=TEXT_COLOR, markersize=11, label="where the tip is held"),
    ]
    if reference:
        held = hand(state.channel("q_ref")[within(state, hold)])
        ax.plot(*held.T, color=color, linewidth=2.0, linestyle="--", zorder=2.5)
        handles.append(Line2D([], [], color=color, linewidth=2.0, linestyle="--", label="its reference\nover the hold"))
    handles.append(Line2D([], [], color=GOAL_COLOR, linewidth=2.0, label="the curve"))
    fig.legend(handles=handles, loc="outside right center", frameon=False, labelcolor=TEXT_COLOR)
    fig.savefig(out)


def panel_row(heading: str, n: int, size: tuple[float, float]) -> tuple[Figure, list[Axes]]:
    """A figure of ``n`` workspace panels side by side under a heading. Its margins are fixed, in inches: the
    constrained layout puts the titles of equal-aspect panels off the figure or on the heading."""
    fig = Figure(figsize=size, dpi=DPI, facecolor=SURFACE_COLOR)
    fig.suptitle(heading, y=1.0 - 0.1 / size[1], va="top", color=TITLE_COLOR)
    width, height = size
    margins = {"left": 0.8 / width, "right": 1.0 - 0.16 / width, "bottom": 1.05 / height, "top": 1.0 - 1.0 / height}
    axes = fig.subplots(1, n, sharex=True, sharey=True, gridspec_kw={**margins, "wspace": 0.05})
    return fig, list(axes)


def paths_figure(
    heading: str,
    panels: list[tuple[str, str, list[Path]]],
    window: tuple[float, float],
    view: View,
    size: tuple[float, float],
    *,
    out: Path,
) -> None:
    """Hand paths over a window side by side, a panel per arm: its title, color, and logs."""
    fig, axes = panel_row(heading, len(panels), size)
    for ax, (title, color, logs) in zip(axes, panels, strict=True):
        workspace(ax, view, title)
        for path in logs:
            draw_path(ax, StateLog.load(path), window, color)
    for ax in axes[1:]:
        ax.set_ylabel("")
    handles = path_handles(forced=False)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    fig.savefig(out)


def distance_figure(
    heading: str,
    panels: list[tuple[str, list[Path], list[Path]]],
    window: tuple[float, float],
    shaded: list[tuple[float, float]],
    shade: str,
    *,
    out: Path,
) -> None:
    """The hand's distance from the curve over a window, a panel per ESN: its title, the ESN's logs, drawn blue over
    the replay's logs, orange; logarithmic, with 20 mm and the disturbances shaded, named ``shade`` in the legend."""
    fig = figure(TWO_PANELS if len(panels) == 2 else WIDE)
    fig.suptitle(heading, color=TITLE_COLOR)
    axes = fig.subplots(1, len(panels), sharex=True, sharey=True, squeeze=False)[0]
    for ax, (title, esn_logs, replay_logs) in zip(axes, panels, strict=True):
        style(ax)
        ax.axhline(1e3 * cycle().tolerance, color=GOAL_COLOR, linewidth=1.6, zorder=4)
        for onset, end in shaded:
            ax.axvspan(onset, end, color=DISTURBANCE_COLOR, zorder=0)
        for logs, color, zorder in ((replay_logs, REPLAY_COLOR, 2.0), (esn_logs, ESN_COLOR, 3.0)):
            for path in logs:
                times, distance = off_curve(StateLog.load(path), window)
                ax.plot(times, distance, color=color, linewidth=1.0, alpha=0.75, zorder=zorder)
        ax.set_yscale("log")
        ax.set_ylim(0.1, 2000.0)
        ax.set_xlim(*window)
        ax.set_title(title, color=TITLE_COLOR)
        ax.set_xlabel("time (s)", color=TEXT_COLOR)
    axes[0].set_ylabel("from the curve (mm)", color=TEXT_COLOR)
    handles: list[Line2D | Patch] = [
        Line2D([], [], color=ESN_COLOR, linewidth=1.6, label="the ESN's arm"),
        Line2D([], [], color=REPLAY_COLOR, linewidth=1.6, label="the replay's arm"),
        Line2D([], [], color=GOAL_COLOR, linewidth=1.6, label=f"{1e3 * cycle().tolerance:g} mm"),
    ]
    if shaded:
        handles.append(Patch(facecolor=DISTURBANCE_COLOR, label=shade))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    fig.savefig(out)


def last_loops(heading: str, panels: list[tuple[str, str, list[Path]]], *, out: Path) -> None:
    """The hand's path over the last taught periods of each run, a panel per arm: its title, color, and logs."""
    loops, period = cycle().periods, cycle().period
    fig, axes = panel_row(heading, len(panels), LOOPS)
    for ax, (title, color, logs) in zip(axes, panels, strict=True):
        workspace(ax, CURVE_VIEW, title)
        for path in logs:
            state = StateLog.load(path)
            last = hand(state.channel("q")[state.times > state.times[-1] - loops * period - 1e-9])
            ax.plot(last[:, 0], last[:, 1], color=color, linewidth=1.0, alpha=0.7, zorder=3)
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.savefig(out)


# ---------------------------------------------------------------------------- the videos


def player_export(log: Path, speed: float = 1.0, *, out: Path) -> None:
    """Render a log into an MP4 with skelarm's player and its side panel, without a window, at ``speed`` times."""
    command = [sys.executable, str(PLAYER), str(log), "--export", str(out), "--fps", f"{FPS:g}", "--panel"]
    subprocess.run(
        [*command, "--speed", f"{speed:g}"],
        check=True,
        capture_output=True,
        env=os.environ | {"QT_QPA_PLATFORM": "offscreen"},
    )


def clip(log: Path, window: tuple[float, float] | None, *, out: Path) -> None:
    """A log in skelarm's player over a window of its times (s), or the whole log with None. The player has no window
    of its own, so it plays a copy of the log cut to the window, made in a temporary folder."""
    if window is None:
        player_export(log, out=out)
        return
    with tempfile.TemporaryDirectory() as tmp:
        part = Path(tmp) / log.name
        cut(StateLog.load(log), window).save(part)
        player_export(part, out=out)


def sequence_video(parts: list[tuple[Path, str]], speed: int, arm: str, *, out: Path) -> None:
    """Videos of ``clip`` one after another at ``speed`` times, with the speed, the arm, and each video's label written
    in the player's side panel. Each is read as exported, at its own speed, and every ``speed``-th frame of it kept,
    which are the frames the player shows at that speed."""
    bold = findfont(FontProperties(family="DejaVu Sans", weight="bold"))
    big, small, smaller = (ImageFont.truetype(bold, size) for size in (120, 60, 44))
    # Quality 10 would be lossless H.264, a profile (High 4:4:4 Predictive) that PowerPoint shows as black.
    writer: Any = imageio.get_writer(out, fps=FPS, codec="libx264", quality=9.5, macro_block_size=16)
    with writer:
        for part, label in parts:
            reader: Any = imageio.get_reader(part)
            for index, frame in enumerate(reader):
                if index % speed:
                    continue
                image = Image.fromarray(frame)
                draw = ImageDraw.Draw(image)
                draw.text(MARK_AT, f"\N{MULTIPLICATION SIGN}{speed}", fill=TITLE_COLOR, font=big)
                draw.text(ARM_AT, arm, fill=TITLE_COLOR, font=small)
                draw.multiline_text(LABEL_AT, label, fill=TITLE_COLOR, font=smaller)
                writer.append_data(np.asarray(image))
            reader.close()


# ---------------------------------------------------------------------------- the media


def window_text(window: tuple[float, float]) -> str:
    return f"from {window[0]:g} s to {window[1]:g} s".replace("-", "\N{MINUS SIGN}")


def logs_of(scenario: str, name: str) -> list[Path]:
    """The logs of an arm of ``ARMS`` in a scenario, from every start."""
    return [arm_log(scenario, name, start) for start in starts_of(scenario)]


def run_media(
    scenario: str, folder: str, name: str, start: int, window: tuple[float, float], view: View, what: str
) -> list[Medium]:
    """An arm's run from a start in ``folder``, four files named after the arm: its video, joint angles, torques, and
    hand path, over a window of the run; with the block, the hand path over ``BLOCK_PATHS``."""
    label = ARMS[name][0]
    log, color, stem = arm_log(scenario, name, start), COLORS[name], robot_run(scenario, "1")
    title = f"{label} {start_names(scenario, start)[1]}, {what}"
    over = window_text(window)
    if scenario == "block_robot_esn":
        tip = functools.partial(block_tip, log, title, color, name != "replay")
        tip_over = window_text(BLOCK_PATHS)
    else:
        tip = functools.partial(run_tip, log, window, view, title, color)
        tip_over = over
    series = (log, window, spans(stem), shade_name(stem), title, color)
    return [
        Medium(
            f"{folder}/{name}.mp4", f"{title}, in skelarm's player, {over}", [log], functools.partial(clip, log, window)
        ),
        Medium(
            f"{folder}/{name}_joints.png",
            f"{title}: the joint angles over time, {over}",
            [log],
            functools.partial(run_series, *series, "joints"),
        ),
        Medium(
            f"{folder}/{name}_torques.png",
            f"{title}: the joint torques over time, {over}",
            [log],
            functools.partial(run_series, *series, "torques"),
        ),
        Medium(f"{folder}/{name}_tip.png", f"{title}: the hand's path, {tip_over}", [log], tip),
    ]


def runs_media(
    folder: str, scenario: str, window: tuple[float, float], view: View, what: str, starts: Collection[int] | None
) -> list[Medium]:
    """Each arm's runs in a scenario, a folder per start, and each arm's runs one after another at each of ``SPEEDS``;
    with ``starts``, from those starts alone, and none one after another if the scenario has none of them."""
    chosen = [start for start in starts_of(scenario) if starts is None or start in starts]
    items = []
    for start in chosen:
        for name in ARMS:
            items += run_media(scenario, f"{folder}/{start_names(scenario, start)[0]}", name, start, window, view, what)
    if not chosen:
        return items
    for name, (label, _, _) in ARMS.items():
        clips = [f"{folder}/{start_names(scenario, start)[0]}/{name}.mp4" for start in chosen]
        parts = [(OUT / path, start_names(scenario, start)[2]) for path, start in zip(clips, chosen, strict=True)]
        who = "the replay" if name == "replay" else label
        items += [
            Medium(
                f"{folder}/all_starts_{name}_{speed}x.mp4",
                f"the runs of {who}, {what}, one after another, each {window_text(window)}, "
                + ("at their own speed" if speed == 1 else f"{speed} times as fast"),
                [arm_log(scenario, name, start) for start in chosen],
                functools.partial(sequence_video, parts, speed, label),
                tuple(clips),
            )
            for speed in SPEEDS
        ]
    return items


def take_media() -> list[Medium]:
    folder = FOLDERS[0]
    path = take_path()
    configs = [run_dir(robot_run(scenario, "1")) / "config.toml" for scenario in ("robot_esn", "untaught_robot_esn")]
    return [
        Medium(
            f"{folder}/starts.png",
            "the curve, the take's hand path, the eight starts on the curve, numbered, and the 36 untaught starts",
            [path, *configs],
            starts_figure,
        ),
        Medium(
            f"{folder}/take.mp4", "the scripted take, in skelarm's player", [path], functools.partial(clip, path, None)
        ),
        Medium(f"{folder}/take_joints.png", "the scripted take: the joint angles over time", [path], take_joints),
        Medium(f"{folder}/take_tip.png", "the scripted take: the hand's path and the start posture", [path], take_tip),
    ]


def paths_media(
    folder: str, scenario: str, window: tuple[float, float], view: View, size: tuple[float, float], starts: str
) -> list[Medium]:
    """The hand paths of ESN #1 and of ESN #2, each against the replay's, undisturbed, from ``starts``, over a
    window."""
    replay = logs_of(scenario, "replay")
    items = []
    for name in ("esn1", "esn2"):
        label, esn = ARMS[name][0], logs_of(scenario, name)
        items.append(
            Medium(
                f"{folder}/paths_{name}_vs_replay.png",
                f"the hand's paths of {label} and of the replay, undisturbed, {starts}, {window_text(window)}",
                esn + replay,
                functools.partial(
                    paths_figure,
                    f"Undisturbed {starts}, {window_text(window)}",
                    [(label, ESN_COLOR, esn), ("Replay", REPLAY_COLOR, replay)],
                    window,
                    view,
                    size,
                ),
            )
        )
    return items


def on_curve_media(starts: Collection[int] | None) -> list[Medium]:
    folder, scenario = FOLDERS[1], "robot_esn"
    duration = config(robot_run(scenario, "1"))["evaluation"]["duration"]
    loops = cycle().periods
    esns, replay = [logs_of(scenario, name) for name in ("esn1", "esn2")], logs_of(scenario, "replay")
    arms = [(label, COLORS[name], logs_of(scenario, name)) for name, (label, _, _) in ARMS.items()]
    items = paths_media(
        folder, scenario, ON_CURVE_WINDOW, STARTS_VIEW, CURVE_PATHS, "from the eight starts on the curve"
    )
    items += [
        Medium(
            f"{folder}/off_curve_{duration:g}s.png",
            f"the hand's distance from the curve over the {duration:g} s, ESN #1 and ESN #2 against the replay,"
            " undisturbed, from the eight starts on the curve",
            [*esns[0], *esns[1], *replay],
            functools.partial(
                distance_figure,
                f"Undisturbed for {duration:g} s from the eight starts on the curve:"
                " the hand's distance from the curve",
                [(f"ESN #{n} and the replay", esn, replay) for n, esn in zip("12", esns, strict=True)],
                (-1.0, duration),
                [],
                "",
            ),
        ),
        Medium(
            f"{folder}/last_loops.png",
            f"the hand's last {loops} loops of the runs of {duration:g} s, ESN #1, ESN #2, and the replay, from the"
            " eight starts on the curve",
            [path for _, _, logs in arms for path in logs],
            functools.partial(
                last_loops,
                f"Undisturbed for {duration:g} s from the eight starts on the curve: the hand's last {loops} loops"
                f" ({loops * cycle().period:g} s)",
                arms,
            ),
        ),
    ]
    return items + runs_media(folder, scenario, ON_CURVE_WINDOW, STARTS_VIEW, "undisturbed", starts)


def untaught_media(starts: Collection[int] | None) -> list[Medium]:
    folder, scenario = FOLDERS[2], "untaught_robot_esn"
    esns, replay = [logs_of(scenario, name) for name in ("esn1", "esn2")], logs_of(scenario, "replay")
    items = paths_media(folder, scenario, UNTAUGHT_WINDOW, WHOLE_VIEW, TWO_PANELS, "from the 36 untaught starts")
    items.append(
        Medium(
            f"{folder}/off_curve.png",
            "the hand's distance from the curve, ESN #1 and ESN #2 against the replay, undisturbed, from the 36"
            f" untaught starts, {window_text(UNTAUGHT_DISTANCE)}",
            [*esns[0], *esns[1], *replay],
            functools.partial(
                distance_figure,
                "Undisturbed from the 36 untaught starts: the hand's distance from the curve",
                [(f"ESN #{n} and the replay", esn, replay) for n, esn in zip("12", esns, strict=True)],
                UNTAUGHT_DISTANCE,
                [],
                "",
            ),
        )
    )
    return items + runs_media(folder, scenario, UNTAUGHT_WINDOW, WHOLE_VIEW, "undisturbed", starts)


def disturbed_media(
    folder: str, scenario: str, window: tuple[float, float], view: View, what: str, starts: Collection[int] | None
) -> list[Medium]:
    """The hand's distance from the curve of each ESN against the replay's under a disturbance, and each run."""
    stem = robot_run(scenario, "1")
    replay = logs_of(scenario, "replay")
    items = []
    for name in ("esn1", "esn2"):
        label, esn = ARMS[name][0], logs_of(scenario, name)
        items.append(
            Medium(
                f"{folder}/off_curve_{name}.png",
                f"the hand's distance from the curve, {label} against the replay, {what}, from the eight starts on the"
                f" curve, {window_text(window)}",
                esn + replay,
                functools.partial(
                    distance_figure,
                    f"{label} and the replay, {what}, from the eight starts on the curve:"
                    "\nthe hand's distance from the curve",
                    [("", esn, replay)],
                    window,
                    spans(stem),
                    shade_name(stem),
                ),
            )
        )
    return items + runs_media(folder, scenario, window, view, what, starts)


def pushes_media(starts: Collection[int] | None) -> list[Medium]:
    items = []
    for times, size in SIZES.items():
        folder = f"{FOLDERS[3]}/{times}x"
        items += disturbed_media(folder, f"pushes{times}x_robot_esn", PUSHES_WINDOW, PUSHES_VIEW, size, starts)
    return items


def block_media(starts: Collection[int] | None) -> list[Medium]:
    scenario = "block_robot_esn"
    (hold,) = spans(robot_run(scenario, "1"))
    what = f"the tip held from {hold[0]:g} s to {hold[1]:g} s"
    return disturbed_media(FOLDERS[4], scenario, BLOCK_WINDOW, BLOCK_VIEW, what, starts)


def media(starts: Collection[int] | None) -> list[Medium]:
    """Every file of the output; with ``starts``, the runs from those starts alone."""
    return take_media() + on_curve_media(starts) + untaught_media(starts) + pushes_media(starts) + block_media(starts)


# ---------------------------------------------------------------------------- the README


README = """\
# A periodic motion from a script, for slides

Videos and figures of report 026 of the arm-esn-ctrl repository
(`reports/026-a-periodic-motion-from-a-script/README.md`), the first part of the stage
that adds a periodic motion: the hand of the two-link arm goes around an ellipse without
end, taught by one take made by a script. The folders follow the report's Sections 2 and
3.2 to 3.6. They are not kept in Git;
`presentations/periodic-motion-from-a-script/make_media.py` there makes them.

## What they show

- **The arm:** a two-link arm on a horizontal plane, its links {lengths} m long and of
  {masses} kg, with no joint limits, simulated every {sim_ms:g} ms. The gray rings bound
  where the hand can reach.
- **The curve:** an ellipse centered at ({cx}, {cy:g}) m, with semi-axes of {a:g} m along x
  and {b:g} m along y, drawn purple in the figures and gray in the player. The hand goes
  around it counterclockwise, {period:g} s a loop.
- **The take** is made by a script: the joint angles that put the hand on the ellipse at an
  even rate of the curve's angle, for {take_s:g} s, {loops:g} loops, sampled every
  {take_ms:g} ms. It starts at the posture ({q1:.1f}°, {q2:.1f}°), the hand on the curve's
  right end, and moves at its full pace from its first sample.
- **The start postures:** the eight starts on the curve put the hand at the curve's
  angles {angles}, an eighth of a loop apart; start 0 is the
  take's. The 36 untaught starts are report 023's: hands on three rings around the base,
  spread evenly over the workspace.
  The folders `start_0` to `start_7` and `untaught_00` to `untaught_35` are named after
  them, in the order of the configurations.
- **The arms**, each giving the reference of the same tracker:
{arms}
  - **Replay**, orange: the take played by time, without end: it plays the take to
    {loop_end:g} s, then goes back to {loop_start:g} s, again and again. It begins at the
    take's start wherever the arm is.

  Both ESNs have {neurons} tanh neurons, sparsity {sparsity:g}, a bias per neuron drawn
  uniformly from [−1, 1], a {warmup:g} s warm-up on the held start posture, and a
  {period_ms:g} ms period, and are trained by teacher forcing on the take, without noise.
  Their input is the arm's two joint angles every {period_ms:g} ms; they are given no
  curve, no phase, and no context. They differ in the spectral radius alone.
- **The tracker** of every arm: joint PD at ω = {omega:g} rad/s and a damping ratio of
  {damping:g}, given the reference's velocity, its torque computed every
  {control_ms:g} ms and held between. Its gains are set where the take of reports 005 to
  020 ends, as in report 023.
- **The pushes** (`3.5_pushes`) are report 007's three, at the tip, each for {push_s:g} s:
  {pushes}. A push's angle is turned
  counterclockwise from the direction from the run's start hand toward the curve's center. They are run at those forces (`1x`), at twice
  (`2x`: {forces_2x}), and at four times (`4x`: {forces_4x}).
- **The block** (`3.6_block`) is a stiff spring-damper ({stiffness:,g} N/m,
  {block_damping:g} N s/m). It holds the tip from t = {onset:g} s to {release:g} s where
  it was at {onset:g} s.
- **Time:** every run holds its start posture for the {warmup:g} s warm-up, from
  t = −{warmup:g} s; the task begins at t = 0. The runs last {on_curve_s:g} s from the
  starts on the curve, {untaught_s:g} s from the untaught starts, {pushes_s:g} s under the
  pushes, and {block_s:g} s with the block.

## The windows

The runs are long, so a run's video and its figures show a window of it. A video is the
player's export of a copy of the run's log cut to the window, with every sample in it at
its own time; the side panel's clock is the run's.

| Runs | Window |
| --- | --- |
| The take (`00_take`) | the whole take, 0 to {take_s:g} s |
| Undisturbed from the starts on the curve (`3.2_on_curve`) | {on_curve_window}: the start and three loops |
| Undisturbed from the untaught starts (`3.3_untaught_starts`) | {untaught_window}: joining the curve and two loops |
| Under the pushes (`3.5_pushes`) | {pushes_window}: the three pushes and 7 s after the last |
| With the block (`3.6_block`) | {block_window}: the hold and 7 s after; the hand paths {block_paths} |

## The folders

| Folder | What it holds |
| --- | --- |
| `00_take` | The take and the starts (Section 2): `starts.png`, `take.mp4`, `take_joints.png`, and `take_tip.png`. |
| `3.2_on_curve` | Undisturbed from the eight starts on the curve, claim 1 (Section 3.2): the hand paths, the distance from the curve over the {on_curve_s:g} s and the last loops, `all_starts_*.mp4`, each arm's runs one after another, and `start_0` to `start_7`, the runs from each start. |
| `3.3_untaught_starts` | Undisturbed from the 36 untaught starts, claims 2 and 3 (Sections 3.3 and 3.4): the hand paths, the distance from the curve over the first {untaught_distance_s:g} s, `all_starts_*.mp4`, and `untaught_00` to `untaught_35`. |
| `3.5_pushes` | Under the pushes, claim 4 (Section 3.5): `1x`, `2x`, and `4x`, each with the distance from the curve, `all_starts_*.mp4`, and `start_0` to `start_7`. |
| `3.6_block` | With the block, claim 5 (Section 3.6): the distance from the curve, `all_starts_*.mp4`, and `start_0` to `start_7`. |

## The figures and videos

| File | What it shows |
| --- | --- |
| `00_take/starts.png` | the curve (purple), the take's hand path (thick gray under it), the eight starts on the curve (black dots, numbered), and the 36 untaught starts (orange squares): the report's Section 2 figure |
| `00_take/take.mp4` | the take in skelarm's player |
| `00_take/take_joints.png` | its two joint angles over time |
| `00_take/take_tip.png` | its hand path over the curve, with its start posture |
| `3.2_on_curve/paths_esn1_vs_replay.png`, `paths_esn2_…` | the hand paths of ESN #1 (or #2) and of the replay over the window, from the eight starts on the curve (dots) |
| `3.2_on_curve/off_curve_{on_curve_s:g}s.png` | the hand's distance from the curve over the whole runs, logarithmic, the ESN's arm (blue) over the replay's (orange), the 20 mm line purple: the upper row of the report's `scripted_long.png` |
| `3.2_on_curve/last_loops.png` | the hand's last {last_loops} loops ({last_s:g} s) of each run over the curve: the lower row of the report's `scripted_long.png` |
| `3.3_untaught_starts/paths_esn1_vs_replay.png`, `paths_esn2_…` | the same as in `3.2` from the 36 untaught starts |
| `3.3_untaught_starts/off_curve.png` | the hand's distance from the curve over the first {untaught_distance_s:g} s from every untaught start: the first row of the report's `scripted_off_curve.png` |
| `3.5_pushes/1x/off_curve_esn1.png`, `off_curve_esn2.png`, and likewise in `2x` and `4x` | the hand's distance from the curve over the window from the eight starts, the ESN's arm over the replay's, the pushes shaded: a row of the report's `scripted_off_curve.png` |
| `3.6_block/off_curve_esn1.png`, `off_curve_esn2.png` | the same with the block, the hold shaded |
| `all_starts_esn1_1x.mp4`, `all_starts_esn2_1x.mp4`, `all_starts_replay_1x.mp4`, in `3.2`, `3.3`, each size of `3.5`, and `3.6` | each arm's runs one after another, start 0's to start 7's or untaught 00's to untaught 35's, each its own video over the window, at their own speed; the speed, the arm, and the start written in the side panel |

## The runs

A run folder (`start_N`, `untaught_NN`) has four files for each arm, named after it:
`esn1` and `esn2` (ESN #1 and ESN #2), and `replay`.

| File | What it shows |
| --- | --- |
| `<arm>.mp4` | the run over the window in skelarm's player with its side panel: the time, the joint angles, the hand's position and speed, and under the pushes and the block the force at the tip, also drawn as a red arrow |
| `<arm>_joints.png` | the two joint angles over the window: the arm, and its reference dashed. The replay's reference is the take. The pushes or the hold are shaded. |
| `<arm>_torques.png` | the two joint torques over the window, likewise |
| `<arm>_tip.png` | the hand's path over the window, over the curve: a dot where the window begins, and under the pushes thick black while a force acts and a cross where the window ends. With the block, the path {block_paths}, a cross where the tip is held, and for the ESNs their reference's hand path over the hold, dashed, as the report draws them from start 0. |

## Notes

- **The torque axes** span the largest torque over the window, but for the first
  {cut_ms:g} ms after t = 0, and at least 0.5 N m. Where a run's start sends the torque
  higher in those {cut_ms:g} ms, the figure writes how high. The report cuts its torque
  axes lower, at the largest torque of the ESN's arm; these do not.
- **The replays** do not depend on the ESN. They are taken from ESN #1's runs; ESN #2's
  runs give the same, joint angle for joint angle, and are not repeated.
- **The colors:** the ESNs blue, the replay orange, and the curve purple, as in the report.
- **The force arrow** in the player is as long as 40% of the arm's reach at the largest
  force of the log it plays, whatever the pushes' size: its length shows a force against
  the run's largest, and the side panel gives it in newtons. A window holds every push or
  the whole hold, so the arrow is as long as in the whole run. Where the hand is high in
  the workspace, the arrow can run past the top of the frame.
- **Sizes:** the videos are H.264 at {fps:g} fps, 1104 x 800 px. The figures are
  {two_px} px with two panels ({curve_paths_px} px for the paths from the starts on the
  curve) and {loops_px} px for `last_loops.png`, {wide_px} px for the distance from the
  curve of one ESN, {starts_px} px for `starts.png`, and {run_px} px for a run's or the
  take's own.
- [`media.csv`](media.csv) lists every file with the logs it shows, relative to the
  storage root.

## Reproducing them

From the repository, at commit `{commit}`{dirty}:

    uv run python presentations/periodic-motion-from-a-script/make_media.py --workers 8

It reads the take and the runs under the storage root and overwrites the files here, this
README included; `--only` renders the named folders alone, and `--starts` the runs from
the named starts alone. This README was written on {day}.
"""  # noqa: E501, RUF001


def commit_state() -> tuple[str, str]:
    """The repository's commit, and `` (with uncommitted changes)`` if it has any, else nothing."""
    git = [["git", "rev-parse", "--short", "HEAD"], ["git", "status", "--porcelain"]]
    commit, changes = (subprocess.run(c, cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout for c in git)
    return commit.strip(), " (with uncommitted changes)" if changes.strip() else ""


def setting_text(esn: dict[str, Any]) -> str:
    return (
        f"leak rate {esn['leak_rate']:g}, spectral radius {esn['spectral_radius']:g}, input scaling"
        f" {esn['input_scaling']:g}, ridge {esn['ridge']:g}, seed {esn['seed']}"
    )


def pixels(size: tuple[float, float]) -> str:
    """A figure's size in pixels, as matplotlib saves it."""
    return f"{int(size[0] * DPI)} x {int(size[1] * DPI)}"


def write_readme() -> None:
    """The output folder's README.md, its numbers read from the take's and the runs' configurations."""
    esns = {n: esn_of(n) for n in "12"}
    fixed = ("n_neurons", "sparsity", "bias", "bias_scaling", "warmup", "dt", "teacher_noise_deg")
    if any(esns["1"][key] != esns["2"][key] for key in fixed):
        msg = f"the ESNs differ in what the README gives as fixed: {fixed}"
        raise ValueError(msg)
    first = esns["1"]
    if not (first["bias"] and first["bias_scaling"] == 1.0 and first["teacher_noise_deg"] == 0.0):
        msg = "the README's bias or training does not match the ESNs'"
        raise ValueError(msg)
    if any(esns["1"][key] != esns["2"][key] for key in ("leak_rate", "input_scaling", "ridge", "seed")):
        msg = "the ESNs differ in more than the spectral radius, as the README says they do not"
        raise ValueError(msg)
    arms = "\n".join(f"  - **ESN #{n}**, blue: {setting_text(esn)}." for n, esn in esns.items())
    stems = [robot_run(scenario, n) for scenario in SCENARIOS for n in "12"]
    tracker = config(stems[0])["tracker"]
    if any(config(stem)["tracker"] != tracker for stem in stems):
        msg = "the runs' trackers differ"
        raise ValueError(msg)
    pushes = config(robot_run("pushes1x_robot_esn", "1"))["disturbance"]
    parts = [
        f"{t['force']:g} N at {t['angle_deg']:g}° at t = {t['onset']:g} s".replace("-", "\N{MINUS SIGN}")
        for t in pushes
    ]
    forces = {
        times: ", ".join(f"{t['force']:g}" for t in config(robot_run(f"pushes{times}x_robot_esn", "1"))["disturbance"])
        for times in SIZES
    }
    block = config(robot_run("block_robot_esn", "1"))["disturbance"]
    task = TAKE_CONFIG["task"]
    times, q = take()
    loop_start, loop_end = cycle().loop
    dt = first["dt"]
    angles = [
        curve_angle(posture_of(robot_run("robot_esn", "1"), row["origin"])) for row in arm_rows(stems[0], "replay")
    ]
    links = SKELETON.links[1:]
    duration = {scenario: config(robot_run(scenario, "1"))["evaluation"]["duration"] for scenario in SCENARIOS}
    commit, dirty = commit_state()
    (OUT / "README.md").write_text(
        README.format(
            lengths=" and ".join(f"{link.prop.length:g}" for link in links),
            masses=" and ".join(f"{link.prop.m:g}" for link in links),
            sim_ms=1e3 * TAKE_CONFIG["simulator"]["dt"],
            cx=f"{task['center'][0]:g}".replace("-", "\N{MINUS SIGN}"),
            cy=task["center"][1],
            a=task["a"],
            b=task["b"],
            period=task["period"],
            take_s=times[-1] - times[0],
            loops=(times[-1] - times[0]) / task["period"],
            take_ms=1e3 * float(np.median(np.diff(times))),
            q1=np.degrees(q[0, 0]),
            q2=np.degrees(q[0, 1]),
            angles=", ".join(f"{angle}°" for angle in angles[:-1]) + f", and {angles[-1]}°",
            arms=arms,
            loop_end=dt * loop_end,
            loop_start=dt * loop_start,
            neurons=first["n_neurons"],
            sparsity=first["sparsity"],
            warmup=first["warmup"],
            period_ms=1e3 * dt,
            omega=tracker["omegas"][0],
            damping=tracker["dampings"][0],
            control_ms=1e3 * tracker["control_period"],
            push_s=pushes[0]["duration"],
            pushes="; ".join(parts[:-1]) + f"; and {parts[-1]}",
            forces_2x=f"{forces[2]} N",
            forces_4x=f"{forces[4]} N",
            stiffness=block["stiffness"],
            block_damping=block["damping"],
            onset=block["onset"],
            release=block["release"],
            on_curve_s=duration["robot_esn"],
            untaught_s=duration["untaught_robot_esn"],
            pushes_s=duration["pushes1x_robot_esn"],
            block_s=duration["block_robot_esn"],
            on_curve_window=window_text(ON_CURVE_WINDOW),
            untaught_window=window_text(UNTAUGHT_WINDOW),
            pushes_window=window_text(PUSHES_WINDOW),
            block_window=window_text(BLOCK_WINDOW),
            block_paths=window_text(BLOCK_PATHS),
            untaught_distance_s=UNTAUGHT_DISTANCE[1],
            last_loops=cycle().periods,
            last_s=cycle().periods * cycle().period,
            cut_ms=1e3 * CUT,
            fps=FPS,
            two_px=pixels(TWO_PANELS),
            curve_paths_px=pixels(CURVE_PATHS),
            loops_px=pixels(LOOPS),
            wide_px=pixels(WIDE),
            starts_px=pixels(STARTS_SIZE),
            run_px=pixels(RUN),
            commit=commit,
            dirty=dirty,
            day=date.today().isoformat(),
        )
    )


def write_media_csv(items: list[Medium]) -> None:
    root = storage_root()
    with (OUT / "media.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["folder", "file", "shows", "logs"])
        for item in items:
            folder, _, name = item.path.rpartition("/")
            writer.writerow([folder, name, item.shows, "; ".join(str(p.relative_to(root)) for p in item.logs)])


# ---------------------------------------------------------------------------- the whole


def render(items: list[Medium], workers: int) -> None:
    """Make the files of ``items`` in ``workers`` processes: first those made from the logs, the videos first, as they
    take longest, the pushed runs' first among them; then those made from other files of the output, the videos of
    many runs one after another, the most runs first."""
    os.environ["OPENBLAS_NUM_THREADS"] = "1"  # for the processes started from here on
    for item in items:
        (OUT / item.path).parent.mkdir(parents=True, exist_ok=True)
    first = sorted(
        (item for item in items if not item.reads),
        key=lambda item: (not item.path.endswith(".mp4"), FOLDERS[3] not in item.path),
    )
    then = sorted((item for item in items if item.reads), key=lambda item: -len(item.reads))
    done, total = 0, len(items)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for batch in (first, then):
            futures = {pool.submit(item.make, out=OUT / item.path): item for item in batch}
            for future in as_completed(futures):
                if future.exception() is not None:
                    print(f"failed: {futures[future].path}", flush=True)
                    pool.shutdown(cancel_futures=True)
                    future.result()
                done += 1
                if done % 50 == 0 or done == total:
                    print(f"made {done} of {total} files", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=8, help="processes that render the files (default: 8)")
    parser.add_argument(
        "--only", nargs="+", choices=FOLDERS, metavar="FOLDER", help=f"render these folders alone, of {FOLDERS}"
    )
    parser.add_argument(
        "--starts",
        nargs="+",
        type=int,
        metavar="N",
        help="render the runs from these starts alone, by their index, and the videos of all starts of them alone",
    )
    args = parser.parse_args()
    check_runs()
    items = media(None)
    chosen = [item for item in media(args.starts) if args.only is None or item.path.split("/")[0] in args.only]
    print(f"{len(chosen)} of {len(items)} files to make into {OUT}", flush=True)
    render(chosen, args.workers)
    write_media_csv(items)
    write_readme()
    print(f"wrote media.csv and README.md to {OUT}")


if __name__ == "__main__":
    main()
