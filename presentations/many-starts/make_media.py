# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Videos and figures of report 023, one target from many starts, for slides.

    uv run python presentations/many-starts/make_media.py [--workers 8] [--only FOLDER ...]

From the takes and the robot runs of report 023
(``reports/023-one-target-from-many-starts/README.md``, Sections 2 and 3.2 to 3.5),
under the storage root, it writes into ``presentations/many-starts/`` there a folder per
section:

- ``00_takes``: the takes' hand paths and the start postures, the nine takes one after
  another at two and five times their speed, and each take's video, joint angles, and
  hand path;
- ``3.2_takes_starts``: the hand paths of ESN #1 and ESN #2 against the replay's,
  undisturbed, from the nine takes' starts and from the 36 unseen starts, and each run
  from a take's start;
- ``3.3_unseen_starts``: the report's measures from each unseen start and its joint
  angles and torques over time, a figure per panel or column, and each run from an
  unseen start;
- ``3.4_pushes``: the hand paths and the distance to the target under the pushes over
  60 s, and each pushed run, the progress replay's too;
- ``3.5_nine_takes_vs_one``: the nine-take ESNs against the one-take ESNs and the
  replay of take 00, and each run of the one-take ESNs and of that replay.

A run's files are its video in skelarm's player (``--export --panel``), its joint angles
and torques over time, and its hand path. The folder's ``README.md`` explains them, and
``media.csv`` lists every file with the logs it shows. The replays do not depend on the
ESN: they are taken from ESN #1's runs, checked to equal ESN #2's. ``--workers`` sets the
number of processes that render; ``--only`` renders the named folders alone, while the
README and ``media.csv`` are always written in full. It overwrites its own files and
deletes nothing.
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
from collections.abc import Callable
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
from skelarm import Skeleton, StateLog, compute_forward_kinematics

from arm_esn_ctrl.demonstrations import endpoint_positions, resample_joint_angles
from arm_esn_ctrl.storage import REPO_ROOT, resolve_run_path, storage_root

OUT = storage_root() / "presentations" / "many-starts"
EXPERIMENT = "results/manual_demonstration_v2_many_starts"
TAKES_RUN = "results/demonstrations/20261117-164658-reach_manual_v2_no_limits"
ARM_CONFIG = REPO_ROOT / "experiments" / "demonstrations" / "reach_manual_v2_no_limits.toml"
CONFIGS = REPO_ROOT / "experiments" / "manual_demonstration_v2_many_starts"
PLAYER = REPO_ROOT / "third_party" / "skelarm" / "tools" / "player.py"
TRACKER = "pd_w20_z0.4"  # the one tracker's directory in a robot run
DT = 0.01  # the ESNs' period, and the takes' resampling for the plots (s)
FPS = 30.0
SPEEDS = (2, 5)  # the speeds of the videos of all takes
CUT = 0.05  # s: the torque axes leave out the first 50 ms, as report 023's figures do
FORCE_TOP = 22.0  # N: the top of the tip force's axis, as in report 023
UNSEEN_SHOWN = "unseen 1"  # report 023's unseen start drawn over time

with ARM_CONFIG.open("rb") as _f:
    _ARM = tomllib.load(_f)
SKELETON = Skeleton.from_toml(ARM_CONFIG)
TARGET = np.array(_ARM["task"]["target"]["pos"])  # m
TOLERANCE = float(_ARM["task"]["target"]["tolerance"])  # the goal's radius (m)
TAKE_STARTS = np.array(_ARM["demonstrations"]["start_q"])  # deg
N_TAKES = len(TAKE_STARTS)

FOLDERS = ("00_takes", "3.2_takes_starts", "3.3_unseen_starts", "3.4_pushes", "3.5_nine_takes_vs_one")
# The ESNs: their label, their undisturbed robot run, and their color's name.
ESNS = {
    "esn1": ("ESN #1", "robot_esn1", "blue"),
    "esn2": ("ESN #2", "robot_esn2", "blue"),
    "one_take_esn1": ("One-take ESN #1", "robot_one_take_esn1", "green"),
    "one_take_esn2": ("One-take ESN #2", "robot_one_take_esn2", "amber"),
}
SIZES = {1: "pushes at report 007's forces", 2: "pushes at twice the forces", 4: "pushes at four times the forces"}

# Report 023's colors; the progress replay's is report 018's.
TEXT_COLOR, TITLE_COLOR, GRID_COLOR, SURFACE_COLOR = "#52514e", "#0b0b0b", "#e1e0d9", "#fcfcfb"
DISTURBANCE_COLOR, TAUGHT_COLOR = "#e8e6df", "#b9b7b0"
ESN_COLOR, REPLAY_COLOR, GOAL_COLOR = "#2a78d6", "#eb6834", "#7b3fa0"
ESN_COLORS = {"ESN #1": "#1c5cab", "ESN #2": "#6da7ec"}  # where both are drawn
COLORS = {
    "esn1": ESN_COLOR,
    "esn2": ESN_COLOR,
    "one_take_esn1": "#1baf7a",
    "one_take_esn2": "#eda100",
    "replay": REPLAY_COLOR,
    "progress": "#1baf7a",
}

DPI = 150
TWO_PANELS = (12.0, 6.4)  # 1800 x 960 px
THREE_PANELS = (16.0, 6.0)  # 2400 x 900 px
WIDE = (12.0, 6.75)  # a figure over time: 1800 x 1012 px, 16:9
RUN = (10.0, 5.625)  # a run's own figures: 1500 x 843 px, 16:9
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


class Medium(NamedTuple):
    """A file of the output: its path in the output folder, what it shows, the logs it shows, and how it is made."""

    path: str
    shows: str
    logs: list[Path]
    make: Callable[..., None]  # called with ``out``, the file to write


# ---------------------------------------------------------------------------- the runs


@functools.cache
def run_dir(stem: str) -> Path:
    """The one run of a configuration of the experiment, under the storage root."""
    pattern = re.compile(rf"\d{{8}}-\d{{6}}-{re.escape(stem)}")
    matches = sorted(p for p in resolve_run_path(EXPERIMENT).iterdir() if p.is_dir() and pattern.fullmatch(p.name))
    if len(matches) != 1:
        msg = f"expected one run of {stem}, found {[p.name for p in matches]}"
        raise FileNotFoundError(msg)
    return matches[0]


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


def group(stem: str, arm: str, which: str) -> list[dict[str, str]]:
    """One arm's runs from a set of starts: ``demo`` (the takes'), ``unseen``, or ``not_00`` (all but take 00's)."""
    if which == "not_00":
        return [row for row in arm_rows(stem, arm) if row["origin"] != "demo_00_filtered"]
    return [row for row in arm_rows(stem, arm) if row["origin"].startswith(which)]


def start_of(stem: str, arm: str, origin: str) -> int:
    (start,) = [int(row["start"]) for row in arm_rows(stem, arm) if row["origin"] == origin]
    return start


def log_path(stem: str, arm: str, start: int) -> Path:
    return run_dir(stem) / TRACKER / f"{arm}_{start:02d}.sklog.npz"


def played(row: dict[str, str]) -> int:
    """The take a replay's run plays."""
    return int(row["replayed"].split("_")[1])


def take_of(stem: str, origin: str) -> int | None:
    """The take whose start a run's start is, or None for a start no take starts from."""
    if origin.startswith("demo_"):
        return int(origin.split("_")[1])
    name, index = origin.split()
    posture = config(stem)["evaluation"]["extra_starts"][name][int(index)]
    matches = [k for k, start in enumerate(TAKE_STARTS) if np.allclose(start, posture)]
    return matches[0] if matches else None


def start_names(stem: str, origin: str) -> tuple[str, str]:
    """A run's start as a folder name and as a phrase: ``take_03_start``, "from take 03's start"."""
    k = take_of(stem, origin)
    if k is not None:
        return f"take_{k:02d}_start", f"from take {k:02d}'s start"
    name, index = origin.split()
    if name != "unseen":
        msg = f"the start {origin} of {stem} is neither a take's nor an unseen one"
        raise ValueError(msg)
    return f"unseen_{int(index):02d}", f"from unseen start {int(index):02d}"


def pushes_60s(n: str, times: int) -> str:
    """The pushed run of ESN #``n`` at ``times`` the pushes' forces, over 60 s."""
    return f"pushes{times}x_60s_robot_esn{n}"


def take_path(index: int) -> Path:
    """A take filtered at 2 Hz, what the ESNs learned, under the storage root."""
    return resolve_run_path(f"{TAKES_RUN}/demo_{index:02d}_filtered.sklog.npz")


@functools.cache
def take(index: int) -> NDArray[np.float64]:
    """A take's joint angles (rad), every 10 ms."""
    return resample_joint_angles(StateLog.load(take_path(index)), DT)[1]


def extra_starts(name: str, which: str) -> NDArray[np.float64]:
    """A set of start postures no take starts from, as a configuration of the experiment lists it (deg)."""
    with (CONFIGS / name).open("rb") as f:
        return np.array(tomllib.load(f)["evaluation"]["extra_starts"][which])


def esn_of(stem: str) -> tuple[dict[str, Any], list[str]]:
    """The ``[esn]`` table of a run's ESN, as saved with it, and the takes it was trained on."""
    model = resolve_run_path(config(stem)["esn"]["model"])
    with model.open("rb") as f:
        esn = tomllib.load(f)["esn"]
    with (model.parent / "config.toml").open("rb") as f:
        return esn, tomllib.load(f)["demonstrations"]["train"]


def check_runs() -> None:
    """The replays of ESN #1's runs equal those of ESN #2's, and every undisturbed run has the same unseen starts."""
    pairs: list[tuple[str, str, tuple[str, ...]]] = [
        ("robot_esn1", "robot_esn2", ("replay",)),
        ("robot_one_take_esn1", "robot_one_take_esn2", ("replay",)),
    ]
    pairs += [(pushes_60s("1", times), pushes_60s("2", times), ("replay", "progress")) for times in SIZES]
    for first, second, arms in pairs:
        for arm in arms:
            starts = [(row["start"], row["origin"]) for row in arm_rows(first, arm)]
            if starts != [(row["start"], row["origin"]) for row in arm_rows(second, arm)]:
                msg = f"the {arm} runs of {first} and {second} are from different starts"
                raise ValueError(msg)
            for start, _ in starts:
                a, b = (StateLog.load(log_path(stem, arm, int(start))).channel("q") for stem in (first, second))
                if not np.array_equal(a, b):
                    msg = f"the {arm} runs of {first} and {second} from start {start} differ"
                    raise ValueError(msg)
    unseen = [config(ESNS[esn][1])["evaluation"]["extra_starts"]["unseen"] for esn in ESNS]
    if any(starts != unseen[0] for starts in unseen):
        msg = "the undisturbed runs' unseen starts differ"
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


def from_target(state: StateLog) -> NDArray[np.float64]:
    """The hand's distance to the target at every step of a log (m)."""
    return np.linalg.norm(hand(state.channel("q")) - TARGET, axis=1)


def force(state: StateLog) -> NDArray[np.float64]:
    """The force on the tip at every step of a log (N), zero in a log without one."""
    if "ext_force" not in state.channel_names:
        return np.zeros(len(state.times))
    return np.linalg.norm(state.channel("ext_force"), axis=1)


def pushed(state: StateLog) -> NDArray[np.bool_]:
    """The steps of a log at which a force acts on the tip."""
    return force(state) > 0.0


def force_spans(state: StateLog) -> list[tuple[float, float]]:
    """When a force acts on the tip: (onset, end) of each push (s)."""
    on = np.concatenate(([False], pushed(state), [False]))
    edges = np.flatnonzero(on[1:] != on[:-1])
    last = len(state.times) - 1
    return [
        (float(state.times[a]), float(state.times[min(b, last)])) for a, b in zip(edges[::2], edges[1::2], strict=True)
    ]


def torque_top(state: StateLog) -> float:
    """The top of a torque axis, as report 023's: 1.15 times the largest torque after the first 50 ms, at least
    0.5 N m."""
    return max(1.15 * float(np.abs(state.channel("tau")[state.times >= CUT]).max()), 0.5)


def joint_distance(row: dict[str, str], starts: NDArray[np.float64]) -> float:
    """How far a start is from the start of the take the replay plays (deg, over the two joints)."""
    start = starts[int(row["origin"].split()[1])]
    return float(np.linalg.norm(start - TAKE_STARTS[played(row)]))


# ---------------------------------------------------------------------------- the figures


def figure(size: tuple[float, float]) -> Figure:
    return Figure(figsize=size, dpi=DPI, facecolor=SURFACE_COLOR, layout="constrained")


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
        ax.add_patch(Circle((0.0, 0.0), radius, fill=False, edgecolor=GRID_COLOR, linewidth=1.4, zorder=1))
    ax.add_patch(Circle(tuple(TARGET), TOLERANCE, fill=False, edgecolor=GOAL_COLOR, linewidth=2.0, zorder=5))
    ax.plot(*TARGET, marker="+", color=GOAL_COLOR, markersize=13, markeredgewidth=2.0, zorder=5)
    ax.plot(0.0, 0.0, marker="s", color=TEXT_COLOR, markersize=6, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(-0.95, 0.95)
    ax.set_ylim(-0.95, 0.95)
    ax.set_xticks([-0.5, 0.0, 0.5])  # every 0.25 m, the labels touch at a slide's font size
    ax.set_yticks([-0.5, 0.0, 0.5])
    ax.set_xlabel("x (m)", color=TEXT_COLOR)
    ax.set_ylabel("y (m)", color=TEXT_COLOR)
    if title:
        ax.set_title(title, color=TITLE_COLOR)


def draw_take(ax: Axes, index: int, linewidth: float = 5.0) -> None:
    points = hand(take(index))
    ax.plot(points[:, 0], points[:, 1], color=TAUGHT_COLOR, linewidth=linewidth, zorder=2)


def draw_path(
    ax: Axes, state: StateLog, color: str, *, linewidth: float = 1.2, dot: float = 5.0, mark_end: bool = False
) -> None:
    """A hand path over the task, its start as a dot, thick black where a force acts, and with ``mark_end`` a cross at
    its end."""
    task = state.times >= 0.0
    points = hand(state.channel("q")[task])
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=linewidth, alpha=0.85, zorder=3)
    ax.plot(*points[0], marker="o", color=color, markersize=dot, zorder=4)
    forced = np.where(pushed(state)[task][:, None], points, np.nan)
    ax.plot(forced[:, 0], forced[:, 1], color=TITLE_COLOR, linewidth=linewidth + 1.8, zorder=3.5)
    if mark_end:
        ax.plot(*points[-1], marker="x", color=TITLE_COLOR, markersize=dot + 3, markeredgewidth=2.0, zorder=4.5)


def path_handles(takes: str, *, forced: bool) -> list[Line2D]:
    """The legend of hand paths: the takes drawn (if ``takes`` names them), the start, the force, the end, the goal."""
    handles = [Line2D([], [], color=TAUGHT_COLOR, linewidth=5.0, label=takes)] if takes else []
    handles.append(Line2D([], [], linestyle="", marker="o", color=TEXT_COLOR, label="start"))
    if forced:
        handles += [
            Line2D([], [], color=TITLE_COLOR, linewidth=3.0, label="while a force acts"),
            Line2D([], [], linestyle="", marker="x", markeredgewidth=2.0, color=TITLE_COLOR, label="end of the run"),
        ]
    handles.append(
        Line2D([], [], linestyle="", marker="o", markerfacecolor="none", color=GOAL_COLOR, label="target (20 mm)")
    )
    return handles


def takes_name(takes: list[int]) -> str:
    return "" if not takes else f"take {takes[0]:02d}" if len(takes) == 1 else "the takes"


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


def paths_figure(heading: str, panels: list[tuple[str, str, list[Path], list[int]]], *, out: Path) -> None:
    """Hand paths side by side, a panel per arm: its title, color, logs, and the takes drawn thick gray behind."""
    # Fixed margins: the constrained layout puts the titles of equal-aspect panels off the figure or on the heading.
    two = len(panels) == 2
    fig = Figure(figsize=TWO_PANELS if two else THREE_PANELS, dpi=DPI, facecolor=SURFACE_COLOR)
    if heading:
        fig.suptitle(heading, y=0.985, color=TITLE_COLOR)
    margins = {"left": 0.07 if two else 0.05, "right": 0.99, "bottom": 0.16, "top": 0.87 if heading else 0.92}
    axes = fig.subplots(1, len(panels), sharex=True, sharey=True, gridspec_kw={**margins, "wspace": 0.05})
    forced = False
    for ax, (title, color, logs, takes) in zip(axes, panels, strict=True):
        workspace(ax, title)
        for index in takes:
            draw_take(ax, index, 5.0 if len(takes) == 1 else 3.5)
        for path in logs:
            state = StateLog.load(path)
            mark = "ext_force" in state.channel_names
            forced |= mark
            draw_path(ax, state, color, linewidth=2.2 if len(logs) == 1 else 1.2, mark_end=mark)
    for ax in axes[1:]:
        ax.set_ylabel("")
    handles = path_handles(takes_name(panels[0][3]), forced=forced)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    fig.savefig(out)


def distance_figure(
    heading: str, panels: list[tuple[str, str, list[Path]]], third: tuple[float, float], *, out: Path
) -> None:
    """The hand's distance to the target over the task, a panel per arm, thick while a force acts."""
    fig = figure(TWO_PANELS)
    fig.suptitle(heading, color=TITLE_COLOR)
    axes = fig.subplots(1, len(panels), sharex=True, sharey=True)
    end = 0.0
    for ax, (title, color, logs) in zip(axes, panels, strict=True):
        style(ax)
        ax.axhline(1e3 * TOLERANCE, color=GOAL_COLOR, linewidth=1.6, zorder=2)
        ax.axvspan(*third, color=DISTURBANCE_COLOR, zorder=0)
        for path in logs:
            state = StateLog.load(path)
            task = state.times >= 0.0
            times, distance = state.times[task], 1e3 * from_target(state)[task]
            linewidth = 2.0 if len(logs) == 1 else 1.2
            ax.plot(times, distance, color=color, linewidth=linewidth, alpha=0.85, zorder=3)
            ax.plot(
                times, np.where(pushed(state)[task], distance, np.nan), color=TITLE_COLOR, linewidth=3.0, zorder=3.5
            )
            end = max(end, float(times[-1]))
        ax.set_yscale("log")
        ax.set_ylim(1.0, 2500.0)
        ax.set_title(title, color=TITLE_COLOR)
        ax.set_xlabel("time (s)", color=TEXT_COLOR)
    for ax in axes:
        ax.set_xlim(0.0, end)
    axes[0].set_ylabel("distance to the target (mm)", color=TEXT_COLOR)
    handles = [
        Line2D([], [], color=TITLE_COLOR, linewidth=3.0, label="while a force acts"),
        Line2D([], [], color=GOAL_COLOR, linewidth=1.6, label="the goal's radius, 20 mm"),
        Patch(facecolor=DISTURBANCE_COLOR, label=f"the third push, at {third[0]:g} s"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    fig.savefig(out)


def hand_paths(with_others: bool, *, out: Path) -> None:
    """The nine takes' hand paths and their starts; with ``with_others``, the untaught and the unseen starts too."""
    fig = figure((12.0, 7.6))
    ax = fig.subplots()
    workspace(ax)
    for index in range(N_TAKES):
        points = hand(take(index))
        ax.plot(points[:, 0], points[:, 1], color=TAUGHT_COLOR, linewidth=3.0, zorder=2)
        ax.plot(*points[0], marker="o", color=TITLE_COLOR, markersize=9, zorder=4)
        ax.annotate(f"{index:02d}", points[0], textcoords="offset points", xytext=(8, 6), color=TITLE_COLOR)
    handles = [
        Line2D([], [], color=TAUGHT_COLOR, linewidth=3.0, marker="o", markerfacecolor=TITLE_COLOR, label="a take"),
    ]
    if with_others:
        untaught = hand(np.radians(extra_starts("hand_esn.toml", "untaught")))
        unseen = hand(np.radians(extra_starts("robot_esn1.toml", "unseen")))
        ax.plot(untaught[:, 0], untaught[:, 1], linestyle="", marker="s", color=REPLAY_COLOR, markersize=7, zorder=3)
        ax.plot(
            unseen[:, 0], unseen[:, 1], linestyle="", marker="^", color=ESN_COLOR, markerfacecolor="none", markersize=9
        )
        handles += [
            Line2D(
                [], [], linestyle="", marker="s", color=REPLAY_COLOR, label="untaught start\n(the search's scoring)"
            ),
            Line2D([], [], linestyle="", marker="^", color=ESN_COLOR, markerfacecolor="none", label="unseen start"),
        ]
    handles.append(Line2D([], [], linestyle="", marker="+", markeredgewidth=2.0, color=GOAL_COLOR, label="target"))
    fig.legend(handles=handles, loc="outside right center", frameon=False, labelcolor=TEXT_COLOR)
    fig.savefig(out)


def take_joints(index: int, *, out: Path) -> None:
    """A take's two joint angles over time."""
    taught = np.degrees(take(index))
    times = DT * np.arange(len(taught))
    fig = figure(RUN)
    axes = fig.subplots(2, 1, sharex=True)
    axes[0].set_title(f"Take {index:02d}, filtered at 2 Hz", color=TITLE_COLOR)
    for joint, ax in enumerate(axes):
        style(ax)
        ax.plot(times, taught[:, joint], color=TITLE_COLOR, linewidth=2.0)
        ax.set_xlim(0.0, times[-1])
        ax.set_ylabel(f"joint {joint + 1} (deg)", color=TEXT_COLOR)
    axes[-1].set_xlabel("time (s)", color=TEXT_COLOR)
    fig.savefig(out)


def take_tip(index: int, *, out: Path) -> None:
    """A take's hand path, with its start and end postures."""
    taught = take(index)
    fig = figure(RUN)
    ax = fig.subplots()
    workspace(ax, f"Take {index:02d}, filtered at 2 Hz")
    for q, shade in ((taught[0], "#9a9890"), (taught[-1], "#cfcdc6")):
        ax.plot(*posture(q).T, color=shade, linewidth=4.0, marker="o", markersize=7, zorder=2)
    points = hand(taught)
    ax.plot(points[:, 0], points[:, 1], color=TITLE_COLOR, linewidth=2.2, zorder=3)
    ax.plot(*points[0], marker="o", color=TITLE_COLOR, markersize=8, zorder=4)
    handles = [
        Line2D([], [], color=TITLE_COLOR, linewidth=2.2, marker="o", label="the hand's path"),
        Line2D([], [], color="#9a9890", linewidth=4.0, marker="o", label="start posture"),
        Line2D([], [], color="#cfcdc6", linewidth=4.0, marker="o", label="end posture"),
        Line2D([], [], linestyle="", marker="o", markerfacecolor="none", color=GOAL_COLOR, label="target (20 mm)"),
    ]
    fig.legend(handles=handles, loc="outside right center", frameon=False, labelcolor=TEXT_COLOR)
    fig.savefig(out)


def run_series(log: Path, title: str, color: str, take_index: int | None, channel: str, *, out: Path) -> None:
    """A run's two joint angles (``joints``) or torques (``torques``) over time; under pushes, the tip force above them
    and the pushes shaded."""
    state = StateLog.load(log)
    forced = "ext_force" in state.channel_names
    n = 3 if forced else 2
    fig = figure(RUN)
    axes = fig.subplots(n, 1, sharex=True, squeeze=False, height_ratios=[0.55, 1.0, 1.0][-n:])[:, 0]
    axes[0].set_title(title, color=TITLE_COLOR)
    for ax in axes:
        style(ax)
        for onset, end in force_spans(state):
            ax.axvspan(onset, end, color=DISTURBANCE_COLOR, zorder=0)
    if forced:
        axes[0].plot(state.times, force(state), color=color, linewidth=1.6)
        axes[0].set_ylim(-0.5, FORCE_TOP)
        axes[0].set_ylabel("tip force (N)", color=TEXT_COLOR)
    joints = channel == "joints"
    values = np.degrees(state.channel("q")) if joints else state.channel("tau")
    top = torque_top(state)
    for joint, ax in enumerate(axes[-2:]):
        if joints and take_index is not None:
            taught = np.degrees(take(take_index))
            ax.plot(DT * np.arange(len(taught)), taught[:, joint], color=TAUGHT_COLOR, linewidth=5.0, zorder=1)
        ax.plot(state.times, values[:, joint], color=color, linewidth=1.8, zorder=3)
        if joints:
            reference = np.degrees(state.channel("q_ref")[:, joint])
            ax.plot(state.times, reference, color=color, linewidth=1.4, linestyle="--", zorder=2.5)
        else:
            ax.set_ylim(-top, top)
            peak = float(np.abs(values[:, joint]).max())
            if peak > top:
                note(ax, f"off the axis in the first 50 ms: up to {peak:.1f} N m")
        ax.set_xlim(state.times[0], state.times[-1])
        ax.set_ylabel(f"joint {joint + 1} ({'deg' if joints else 'N m'})", color=TEXT_COLOR)
    axes[-1].set_xlabel("time (s)", color=TEXT_COLOR)
    handles: list[Line2D | Patch] = [Line2D([], [], color=color, linewidth=1.8, label="the arm")]
    if joints:
        handles.append(Line2D([], [], color=color, linewidth=1.4, linestyle="--", label="its reference"))
        if take_index is not None:
            handles.append(Line2D([], [], color=TAUGHT_COLOR, linewidth=5.0, label=f"take {take_index:02d}"))
    if forced:
        handles.append(Patch(facecolor=DISTURBANCE_COLOR, label="a push"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    fig.savefig(out)


def run_tip(log: Path, title: str, color: str, take_index: int | None, *, out: Path) -> None:
    """A run's hand path in the workspace, with the take that applies thick gray behind it."""
    state = StateLog.load(log)
    forced = "ext_force" in state.channel_names
    fig = figure(RUN)
    ax = fig.subplots()
    workspace(ax, title)
    if take_index is not None:
        draw_take(ax, take_index)
    draw_path(ax, state, color, linewidth=2.2, dot=8.0, mark_end=forced)
    handles = [
        Line2D([], [], color=color, linewidth=2.2, label="the hand's path"),
        *path_handles(takes_name([] if take_index is None else [take_index]), forced=forced),
    ]
    fig.legend(handles=handles, loc="outside right center", frameon=False, labelcolor=TEXT_COLOR)
    fig.savefig(out)


def series_column(
    esn_log: Path, replay_log: Path, take_index: int, label: str, title: str, channel: str, *, out: Path
) -> None:
    """One column of report 023's figures over time: an ESN's joint angles or torques over the replay's, with the ESN's
    reference dashed and the take the replay plays thick gray; the tip force above them under pushes."""
    logs = {"esn": StateLog.load(esn_log), "replay": StateLog.load(replay_log)}
    colors, zorders = {"esn": ESN_COLOR, "replay": REPLAY_COLOR}, {"esn": 3.0, "replay": 2.0}
    forced = "ext_force" in logs["esn"].channel_names
    n = 3 if forced else 2
    fig = figure(WIDE)
    axes = fig.subplots(n, 1, sharex=True, squeeze=False, height_ratios=[0.55, 1.0, 1.0][-n:])[:, 0]
    axes[0].set_title(title, color=TITLE_COLOR)
    for ax in axes:
        style(ax)
    joints = channel == "joints"
    for arm, state in logs.items():
        if forced:
            axes[0].plot(state.times, force(state), color=colors[arm], linewidth=1.4, zorder=zorders[arm])
        values = np.degrees(state.channel("q")) if joints else state.channel("tau")
        for joint, ax in enumerate(axes[-2:]):
            ax.plot(state.times, values[:, joint], color=colors[arm], linewidth=1.6, zorder=zorders[arm])
            if joints and arm == "esn":
                reference = np.degrees(state.channel("q_ref")[:, joint])
                ax.plot(state.times, reference, color=colors[arm], linewidth=1.3, linestyle="--", zorder=2.5)
    if forced:
        axes[0].set_ylim(-0.5, FORCE_TOP)
        axes[0].set_ylabel("tip force (N)", color=TEXT_COLOR)
    taught = np.degrees(take(take_index))
    top = torque_top(logs["esn"])
    for joint, ax in enumerate(axes[-2:]):
        if joints:
            ax.plot(DT * np.arange(len(taught)), taught[:, joint], color=TAUGHT_COLOR, linewidth=5.0, zorder=1)
        else:
            ax.set_ylim(-top, top)
            peaks = {arm: float(np.abs(state.channel("tau")[:, joint]).max()) for arm, state in logs.items()}
            beyond = [f"{label}'s arm up to {peaks['esn']:.1f} N m in the first 50 ms"] if peaks["esn"] > top else []
            beyond += [f"the replay's arm up to {peaks['replay']:.1f} N m"] if peaks["replay"] > top else []
            if beyond:
                note(ax, "off the axis: " + "; ".join(beyond))
        ax.set_xlim(-1.0, logs["esn"].times[-1])
        ax.set_ylabel(f"joint {joint + 1} ({'deg' if joints else 'N m'})", color=TEXT_COLOR)
    axes[-1].set_xlabel("time (s)", color=TEXT_COLOR)
    handles = [
        Line2D([], [], color=ESN_COLOR, linewidth=1.6, label=f"{label}'s arm"),
        Line2D([], [], color=REPLAY_COLOR, linewidth=1.6, label="the replay's arm"),
    ]
    if joints:
        handles += [
            Line2D([], [], color=ESN_COLOR, linewidth=1.3, linestyle="--", label=f"{label}'s reference"),
            Line2D([], [], color=TAUGHT_COLOR, linewidth=5.0, label=f"take {take_index:02d}"),
        ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    fig.savefig(out)


def unseen_panel(key: str, title: str, ylabel: str, *, out: Path) -> None:
    """One panel of report 023's figure of the unseen starts: a measure from each, against how far the start is from
    the start of the take the replay plays."""
    starts = extra_starts("robot_esn1.toml", "unseen")
    fig = figure((10.0, 6.4))
    ax = fig.subplots()
    style(ax)
    arms = [
        ("Replay of the nearest take", "robot_esn1", "replay", REPLAY_COLOR, "^"),
        ("ESN #2", "robot_esn2", "esn", ESN_COLORS["ESN #2"], "s"),
        ("ESN #1", "robot_esn1", "esn", ESN_COLORS["ESN #1"], "o"),
    ]
    for _, stem, arm, color, marker in arms:
        distances = [joint_distance(row, starts) for row in group(stem, "replay", "unseen")]
        values = [float(row[key]) for row in group(stem, arm, "unseen")]
        ax.plot(distances, values, linestyle="", marker=marker, color=color, markersize=8, alpha=0.85)
    ax.set_yscale("log")
    ax.set_title(title, color=TITLE_COLOR)
    ax.set_xlabel("distance from the start of the take the replay plays (°, over both joints)", color=TEXT_COLOR)
    ax.set_ylabel(ylabel, color=TEXT_COLOR)
    handles = [
        Line2D([], [], linestyle="", marker=marker, markersize=8, color=color, label=name)
        for name, _, _, color, marker in reversed(arms)
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
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


def all_takes_video(speed: int, *, out: Path) -> None:
    """The nine takes one after another at ``speed`` times, with the speed and the take written in the side panel."""
    big = ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans", weight="bold")), 120)
    small = ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans", weight="bold")), 60)
    # Quality 10 would be lossless H.264, a profile (High 4:4:4 Predictive) that PowerPoint shows as black.
    writer: Any = imageio.get_writer(out, fps=FPS, codec="libx264", quality=9.5, macro_block_size=16)
    with tempfile.TemporaryDirectory() as tmp, writer:
        for index in range(N_TAKES):
            part = Path(tmp) / f"take_{index:02d}.mp4"
            player_export(take_path(index), speed, out=part)
            reader: Any = imageio.get_reader(part)
            for frame in reader:
                image = Image.fromarray(frame)
                draw = ImageDraw.Draw(image)
                draw.text((822, 250), f"\N{MULTIPLICATION SIGN}{speed}", fill=TITLE_COLOR, font=big)
                draw.text((822, 410), f"take {index:02d}", fill=TITLE_COLOR, font=small)
                writer.append_data(np.asarray(image))
            reader.close()


# ---------------------------------------------------------------------------- the media


def run_media(
    folder: str, name: str, log: Path, title: str, color: str, take_index: int | None, what: str
) -> list[Medium]:
    """A run's four files in ``folder``, named after its arm: its video, joint angles, torques, and hand path."""
    logs = [log] if take_index is None else [log, take_path(take_index)]
    joints = functools.partial(run_series, log, title, color, take_index, "joints")
    torques = functools.partial(run_series, log, title, color, take_index, "torques")
    return [
        Medium(f"{folder}/{name}.mp4", f"{what}, in skelarm's player", [log], functools.partial(player_export, log)),
        Medium(f"{folder}/{name}_joints.png", f"{what}: the joint angles over time", logs, joints),
        Medium(f"{folder}/{name}_torques.png", f"{what}: the joint torques over time", [log], torques),
        Medium(
            f"{folder}/{name}_tip.png",
            f"{what}: the hand's path",
            logs,
            functools.partial(run_tip, log, title, color, take_index),
        ),
    ]


def takes_media() -> list[Medium]:
    folder = FOLDERS[0]
    takes = [take_path(index) for index in range(N_TAKES)]
    items = [
        Medium(
            f"{folder}/hand_paths.png",
            "the nine takes' hand paths, their starts, and the untaught and the unseen starts",
            takes,
            functools.partial(hand_paths, True),
        ),
        Medium(
            f"{folder}/hand_paths_takes_only.png",
            "the nine takes' hand paths and their starts",
            takes,
            functools.partial(hand_paths, False),
        ),
    ]
    items += [
        Medium(
            f"{folder}/all_takes_{speed}x.mp4",
            f"the nine takes one after another, {speed} times as fast",
            takes,
            functools.partial(all_takes_video, speed),
        )
        for speed in SPEEDS
    ]
    for index, path in enumerate(takes):
        sub = f"{folder}/take_{index:02d}"
        what = f"take {index:02d} filtered at 2 Hz"
        items += [
            Medium(f"{sub}/take.mp4", f"{what}, in skelarm's player", [path], functools.partial(player_export, path)),
            Medium(
                f"{sub}/take_joints.png",
                f"{what}: the joint angles over time",
                [path],
                functools.partial(take_joints, index),
            ),
            Medium(f"{sub}/take_tip.png", f"{what}: the hand's path", [path], functools.partial(take_tip, index)),
        ]
    return items


def takes_starts_media() -> list[Medium]:
    folder = FOLDERS[1]
    items = []
    every_take = list(range(N_TAKES))
    for esn in ("esn1", "esn2"):
        label, stem, _ = ESNS[esn]
        for which, name, heading, replay, takes in (
            ("demo", "takes_starts", "From the nine takes' starts", "Replay", every_take),
            ("unseen", "unseen_starts", "From the 36 unseen starts", "Replay of the nearest take", []),
        ):
            esn_logs = [log_path(stem, "esn", int(row["start"])) for row in group(stem, "esn", which)]
            replay_rows = group("robot_esn1", "replay", which)
            replay_logs = [log_path("robot_esn1", "replay", int(row["start"])) for row in replay_rows]
            panels = [(label, ESN_COLOR, esn_logs, takes), (replay, REPLAY_COLOR, replay_logs, takes)]
            items.append(
                Medium(
                    f"{folder}/paths_{esn}_vs_replay_{name}.png",
                    f"the hand's paths of {label} and of the {replay.lower()}, undisturbed, {heading.lower()}",
                    esn_logs + replay_logs + [take_path(index) for index in takes],
                    functools.partial(paths_figure, heading, panels),
                )
            )
    items += per_start_media(folder, "demo", "Replay")
    return items


def per_start_media(folder: str, which: str, replay: str) -> list[Medium]:
    """ESN #1's, ESN #2's, and the replay's undisturbed runs from a set of starts, a folder per start."""
    items = []
    for row in group("robot_esn1", "replay", which):
        sub, phrase = start_names("robot_esn1", row["origin"])
        k = take_of("robot_esn1", row["origin"])
        for esn in ("esn1", "esn2"):
            label, stem, _ = ESNS[esn]
            log = log_path(stem, "esn", start_of(stem, "esn", row["origin"]))
            what = f"{label} {phrase}, undisturbed"
            items += run_media(f"{folder}/{sub}", esn, log, f"{label}, {phrase}", ESN_COLOR, k, what)
        played_take = played(row)
        title = f"Replay {phrase}" if replay == "Replay" else f"Replay of take {played_take:02d}, {phrase}"
        log = log_path("robot_esn1", "replay", int(row["start"]))
        what = f"the replay of take {played_take:02d} {phrase}, undisturbed"
        items += run_media(f"{folder}/{sub}", "replay", log, title, REPLAY_COLOR, played_take, what)
    return items


def unseen_media() -> list[Medium]:
    folder = FOLDERS[2]
    tables = [run_dir(stem) / "metrics.csv" for stem in ("robot_esn1", "robot_esn2")]
    items = [
        Medium(
            f"{folder}/unseen_{name}.png",
            f"{title.lower()} from each unseen start, against how far it is from the start of the take the"
            " replay plays",
            tables,
            functools.partial(unseen_panel, key, title, ylabel),
        )
        for name, key, title, ylabel in (
            ("detour_ratio", "detour_ratio", "Detour ratio", "detour ratio"),
            (
                "reference_speed",
                "peak_reference_speed_dps",
                "The reference's fastest joint speed",
                "the reference's fastest joint speed (°/s)",
            ),
            ("torque", "peak_torque_nm", "Largest joint torque", "largest joint torque (N m)"),
        )
    ]
    # The columns of report 023's figures over time: the unseen start it draws, and the take its replay plays.
    index = int(UNSEEN_SHOWN.split()[1])
    shown = config("robot_esn1")["evaluation"]["extra_starts"]["unseen"][index]
    (unseen,) = [row for row in arm_rows("robot_esn1", "replay") if row["origin"] == UNSEEN_SHOWN]
    for esn in ("esn1", "esn2"):
        label, stem, _ = ESNS[esn]
        columns = [
            ("1_undisturbed_take00", stem, "robot_esn1", "demo_00_filtered", 0, "undisturbed, from take 00's start"),
            (
                f"2_undisturbed_unseen{index:02d}",
                stem,
                "robot_esn1",
                UNSEEN_SHOWN,
                played(unseen),
                f"undisturbed, from the unseen start ({shown[0]:g}°, {shown[1]:g}°)",
            ),
        ]
        columns += [
            (
                f"{3 + k}_pushes{times}x_take00",
                pushes_60s(esn[-1], times),
                pushes_60s("1", times),
                "demo_00_filtered",
                0,
                f"{size}, from take 00's start",
            )
            for k, (times, size) in enumerate(SIZES.items())
        ]
        for name, esn_stem, replay_stem, origin, take_index, title in columns:
            esn_log = log_path(esn_stem, "esn", start_of(esn_stem, "esn", origin))
            replay_log = log_path(replay_stem, "replay", start_of(replay_stem, "replay", origin))
            for channel, what in (("joints", "joint angles"), ("torques", "joint torques")):
                items.append(
                    Medium(
                        f"{folder}/series_columns/{channel}_{esn}_{name}.png",
                        f"the {what} of {label}'s arm and the replay's over time, {title}",
                        [esn_log, replay_log, take_path(take_index)],
                        functools.partial(
                            series_column, esn_log, replay_log, take_index, label, f"{label}, {title}", channel
                        ),
                    )
                )
    items += per_start_media(folder, "unseen", "Replay of the nearest take")
    return items


def third_push(stem: str) -> tuple[float, float]:
    """The push at a fixed time, on the task clock (s)."""
    (table,) = [table for table in config(stem)["disturbance"] if "onset" in table]
    return float(table["onset"]), float(table["onset"] + table["duration"])


def pushes_media() -> list[Medium]:
    folder = FOLDERS[3]
    items = []
    for times, size in SIZES.items():
        replay_stem = pushes_60s("1", times)
        third = third_push(replay_stem)
        for n in "12":
            esn, stem = f"esn{n}", pushes_60s(n, times)
            label = ESNS[esn][0]
            for name, origins, takes, heading in (
                ("", [row["origin"] for row in arm_rows(stem, "esn")], [], size.capitalize()),
                ("take00_", ["demo_00_filtered"], [0], f"{size.capitalize()}, from take 00's start"),
            ):
                esn_logs = [log_path(stem, "esn", start_of(stem, "esn", origin)) for origin in origins]
                replay_logs = [
                    log_path(replay_stem, "replay", start_of(replay_stem, "replay", origin)) for origin in origins
                ]
                logs = esn_logs + replay_logs
                starts = "from take 00's start" if takes else "from the nine takes' starts"
                items += [
                    Medium(
                        f"{folder}/paths_{name}{esn}_vs_replay_{times}x.png",
                        f"the hand's paths of {label} and of the replay, {size}, {starts}",
                        logs + [take_path(index) for index in takes],
                        functools.partial(
                            paths_figure,
                            heading,
                            [(label, ESN_COLOR, esn_logs, takes), ("Replay", REPLAY_COLOR, replay_logs, takes)],
                        ),
                    ),
                    Medium(
                        f"{folder}/distance_{name}{esn}_vs_replay_{times}x.png",
                        f"the hand's distance to the target over time, {label} and the replay, {size}, {starts}",
                        logs,
                        functools.partial(
                            distance_figure,
                            heading,
                            [(label, ESN_COLOR, esn_logs), ("Replay", REPLAY_COLOR, replay_logs)],
                            third,
                        ),
                    ),
                ]
        for row in arm_rows(replay_stem, "replay"):
            sub, phrase = start_names(replay_stem, row["origin"])
            sub = f"{folder}/{times}x/{sub}"
            k = take_of(replay_stem, row["origin"])
            for n in "12":
                stem = pushes_60s(n, times)
                label = ESNS[f"esn{n}"][0]
                log = log_path(stem, "esn", start_of(stem, "esn", row["origin"]))
                items += run_media(
                    sub, f"esn{n}", log, f"{label} {phrase}, {size}", ESN_COLOR, k, f"{label} {phrase}, {size}"
                )
            for arm, name, arm_label in (
                ("replay", "replay", "Replay"),
                ("progress", "progress_replay", "Progress replay"),
            ):
                log = log_path(replay_stem, arm, start_of(replay_stem, arm, row["origin"]))
                what = f"the {arm_label.lower()} of take {played(row):02d} {phrase}, {size}"
                items += run_media(sub, name, log, f"{arm_label} {phrase}, {size}", COLORS[arm], played(row), what)
    return items


def one_take_media() -> list[Medium]:
    folder = FOLDERS[4]
    items = []
    replay_stem = "robot_one_take_esn1"
    replay_logs = [log_path(replay_stem, "replay", int(row["start"])) for row in group(replay_stem, "replay", "not_00")]
    for n in "12":
        label, stem, _ = ESNS[f"esn{n}"]
        one_label, one_stem, _ = ESNS[f"one_take_esn{n}"]
        esn_logs = [log_path(stem, "esn", int(row["start"])) for row in group(stem, "esn", "not_00")]
        one_logs = [log_path(one_stem, "esn", int(row["start"])) for row in group(one_stem, "esn", "not_00")]
        panels = [
            (f"{label} (nine takes)", ESN_COLOR, esn_logs, [0]),
            (one_label, COLORS[f"one_take_esn{n}"], one_logs, [0]),
            ("Replay of take 00", REPLAY_COLOR, replay_logs, [0]),
        ]
        items.append(
            Medium(
                f"{folder}/paths_esn{n}_nine_takes_vs_one_take_vs_replay.png",
                f"the hand's paths of {label}, of {one_label}, and of the replay of take 00, undisturbed, from the 44"
                " starts take 00 does not start from",
                esn_logs + one_logs + replay_logs + [take_path(0)],
                functools.partial(paths_figure, "", panels),
            )
        )
    for row in arm_rows(replay_stem, "replay"):
        sub, phrase = start_names(replay_stem, row["origin"])
        for esn in ("one_take_esn1", "one_take_esn2"):
            label, stem, _ = ESNS[esn]
            log = log_path(stem, "esn", start_of(stem, "esn", row["origin"]))
            what = f"{label} {phrase}, undisturbed"
            items += run_media(f"{folder}/{sub}", esn, log, f"{label}, {phrase}", COLORS[esn], 0, what)
        log = log_path(replay_stem, "replay", int(row["start"]))
        what = f"the replay of take 00 {phrase}, undisturbed"
        items += run_media(
            f"{folder}/{sub}", "replay_of_take_00", log, f"Replay of take 00, {phrase}", REPLAY_COLOR, played(row), what
        )
    return items


# ---------------------------------------------------------------------------- the README


README = """\
# One target from many starts, for slides

Videos and figures of report 023 of the arm-esn-ctrl repository
(`reports/023-one-target-from-many-starts/README.md`), the first stage of the multi-task
study: one target, reached from many start postures. The folders follow the report's
Sections 2 and 3.2 to 3.5. They are not kept in Git;
`presentations/many-starts/make_media.py` there makes them.

## What they show

- **The arm:** a two-link arm on a horizontal plane, its links {lengths} m long and of
  {masses} kg, with no joint limits, simulated every {sim_ms:g} ms. The target is at ({tx:g}, {ty:g}) m,
  {tol_mm:g} mm in radius: the purple circle and cross. The gray rings bound where the hand
  can reach.
- **The takes:** nine reaches to the target taught by hand, one from each of nine start
  postures whose hands are spread over the workspace, six on an outer ring and three on an
  inner one. They last {shortest:.1f} to {longest:.1f} s. The ESNs learn them filtered at
  2 Hz, and the replays play them filtered.
- **The start postures:** the nine takes' starts; 36 untaught starts, from which the
  search scored the ESNs; and 36 unseen starts, the same rings turned 15° back, from which
  nothing was searched or scored. `unseen_00` to `unseen_35` are the unseen starts in the
  order of the configurations.
- **The arms**, each giving the reference of the same tracker:
{arms}
  - **Replay**, orange: the take whose start is nearest the run's start, played by time,
    whatever the arm does. In `3.5_nine_takes_vs_one` it plays take 00 from every start.
  - **Progress replay**, green, under the pushes only: the same take played by time, but
    never more than {lead:g}° ahead of the arm along the take's joint path (a window of
    {window:g} s), so that it waits for the arm.

  All four ESNs have {neurons} tanh neurons, sparsity {sparsity:g}, a bias per neuron drawn
  uniformly from [−1, 1], a {warmup:g} s warm-up on the held start posture, and a
  {period_ms:g} ms period, and are trained by teacher forcing without noise. Their input is
  the arm's two joint angles every {period_ms:g} ms; they are given no target and no
  context.
- **The tracker** of every arm: joint PD at ω = {omega:g} rad/s and a damping ratio of
  {damping:g}, given the reference's velocity, its torque computed every
  {control_ms:g} ms and held between. Its gains are set where the take of reports 005 to
  020 ends, whatever takes the ESN learned.
- **The pushes** (`3.4_pushes`) are report 007's three, at the hand, each for
  {push_s:g} s: {pushes}. The first two thus come at different times from different
  starts; the third comes on an arm at rest at the target in most runs. A push's direction is measured from the
  direction toward the target, and fixed for the run by its start posture. They are run
  at those forces (`1x`), at twice (`2x`: {forces_2x}), and at four times (`4x`:
  {forces_4x}).
- **Time:** every run holds its start posture for the {warmup:g} s warm-up, from
  t = −{warmup:g} s; the task begins at t = 0. The undisturbed runs end at t = {short:g} s,
  the pushed ones at {long:g} s.

## The folders

| Folder | What it holds |
| --- | --- |
| `00_takes` | The takes (Section 2): `hand_paths.png`, `hand_paths_takes_only.png`, `all_takes_2x.mp4`, `all_takes_5x.mp4`, and `take_00` to `take_08`, each with `take.mp4`, `take_joints.png`, and `take_tip.png`. |
| `3.2_takes_starts` | ESN #1 and ESN #2 against the replay, undisturbed (Sections 3.2 and 3.3): the hand paths from the nine takes' starts and from the 36 unseen starts, and `take_00_start` to `take_08_start`, the runs from each take's start. |
| `3.3_unseen_starts` | From the unseen starts (Section 3.3): the report's measures (`unseen_*.png`), its joint angles and torques over time (`series_columns`), and `unseen_00` to `unseen_35`, the runs from each unseen start. |
| `3.4_pushes` | Under the pushes, over {long:g} s (Section 3.4): the hand paths and the distance to the target, and `1x`, `2x`, and `4x`, each with `take_00_start` to `take_08_start`, the runs from each take's start, the progress replay's too. |
| `3.5_nine_takes_vs_one` | The nine-take ESNs against the one-take ESNs and the replay of take 00 (Section 3.5): the hand paths from the 44 starts take 00 does not start from, and `take_00_start` to `take_08_start` and `unseen_00` to `unseen_35`, the runs of the one-take ESNs and of the replay of take 00 from all 45 starts. ESN #1's and ESN #2's runs from these starts are in `3.2_takes_starts` and `3.3_unseen_starts`. |

## The figures and videos

| File | What it shows |
| --- | --- |
| `00_takes/hand_paths.png` | the nine takes' hand paths (gray) and their starts (black dots, numbered), the 36 untaught starts (orange squares), and the 36 unseen starts (blue triangles): the report's Section 2 figure |
| `00_takes/hand_paths_takes_only.png` | the same without the untaught and the unseen starts |
| `00_takes/all_takes_2x.mp4`, `all_takes_5x.mp4` | the nine takes one after another at twice and at five times their speed, the speed and the take written in the side panel |
| `00_takes/take_NN/take.mp4` | the take in skelarm's player |
| `00_takes/take_NN/take_joints.png` | its two joint angles over time |
| `00_takes/take_NN/take_tip.png` | its hand path, with its start and end postures |
| `3.2_takes_starts/paths_esn1_vs_replay_takes_starts.png`, `paths_esn2_…` | the hand paths of ESN #1 (or #2) and of the replay from the nine takes' starts, the takes thick gray behind |
| `3.2_takes_starts/paths_esn1_vs_replay_unseen_starts.png`, `paths_esn2_…` | the same from the 36 unseen starts, against the replay of the nearest take |
| `3.3_unseen_starts/unseen_detour_ratio.png`, `unseen_reference_speed.png`, `unseen_torque.png` | the report's three panels, one a file: the detour ratio, the reference's fastest joint speed, and the largest joint torque from each unseen start, against how far the start is from the start of the take the replay plays (logarithmic vertical axes) |
| `3.3_unseen_starts/series_columns/joints_esn1_1_undisturbed_take00.png`, … | the report's joint angles (`joints_…`) or torques (`torques_…`) of ESN #1 or #2 over time, one file a column: 1, undisturbed from take 00's start; 2, undisturbed from the unseen start of `unseen_01`; 3 to 5, under the pushes at 1x, 2x, and 4x from take 00's start. The ESN's arm (blue) is over the replay's (orange), the ESN's reference dashed, the take the replay plays thick gray, and in the pushed columns the tip force on top. |
| `3.4_pushes/paths_esn1_vs_replay_1x.png`, … | the hand paths of ESN #1 (or #2) and of the replay under the pushes at one size, from the nine takes' starts (dots): thick black while a force acts, a cross where the run ends |
| `3.4_pushes/paths_take00_esn1_vs_replay_1x.png`, … | the same from take 00's start alone, take 00 thick gray |
| `3.4_pushes/distance_esn1_vs_replay_1x.png`, … | the hand's distance to the target over time (logarithmic): thick black while a force acts, the goal's radius purple, the third push shaded |
| `3.4_pushes/distance_take00_esn1_vs_replay_1x.png`, … | the same from take 00's start alone |
| `3.5_nine_takes_vs_one/paths_esn1_nine_takes_vs_one_take_vs_replay.png`, `paths_esn2_…` | the hand paths of ESN #1 (or #2), of one-take ESN #1 (or #2), and of the replay of take 00, from the 44 starts take 00 does not start from, take 00 thick gray |

## The runs

A run folder (`take_NN_start`, `unseen_NN`) has four files for each arm, named after it:
`esn1` and `esn2` (ESN #1 and ESN #2), `replay`, and `progress_replay` in `3.2` to `3.4`;
`one_take_esn1`, `one_take_esn2`, and `replay_of_take_00` in `3.5`.

| File | What it shows |
| --- | --- |
| `<arm>.mp4` | the run in skelarm's player with its side panel: the time, the joint angles, the hand's position and speed, and the force at the hand, also drawn as an arrow |
| `<arm>_joints.png` | the two joint angles over time: the arm, its reference dashed, and thick gray the take that applies. That is the take the replay plays; for a nine-take ESN from a take's start, that take, and from an unseen start none; for a one-take ESN, take 00. Under the pushes, the tip force on top and the pushes shaded. |
| `<arm>_torques.png` | the two joint torques over time, likewise |
| `<arm>_tip.png` | the hand's path, with the take that applies thick gray, the start's dot, the target, thick black while a force acts, and a cross where a pushed run ends |

## Notes

- **The takes have no torques.** They were taught by moving the arm by hand in skelarm's
  recorder: a filtered take's log holds the joint angles and velocities, and the take as
  recorded the joint angles and the hand's position. No torque was recorded, so none is
  drawn.
- **The torque axes**, as in report 023's figures, end at 1.15 times the largest torque of
  the arm after the first {cut_ms:g} ms (of the ESN's arm in `series_columns`), and at
  least 0.5 N m. A run's first {cut_ms:g} ms hold a brief peak that would crush the rest;
  where a line goes beyond the axis, the figure writes how far.
- **The replays** do not depend on the ESN. They are taken from ESN #1's runs; ESN #2's
  runs give the same, joint angle for joint angle, and are not repeated.
- **The colors:** ESNs blue, one-take ESN #1 green and one-take ESN #2 amber as in the
  report, the replay orange, and the progress replay green as in report 018; the progress
  replay and one-take ESN #1 never share a figure.
- **Sizes:** the videos are H.264 at {fps:g} fps, 1104 x 800 px. The figures are
  1800 x 960 px with two panels and 2400 x 900 px with three, 1800 x 1140 px for
  `hand_paths*.png`, 1500 x 960 px for `unseen_*.png`, 1800 x 1012 px for
  `series_columns`, and 1500 x 843 px for a run's or a take's own.
- [`media.csv`](media.csv) lists every file with the logs (or, for `unseen_*.png`, the
  tables) it shows, relative to the storage root.

## Reproducing them

From the repository, at commit `{commit}`{dirty}:

    uv run python presentations/many-starts/make_media.py --workers 8

It reads the takes and the runs under the storage root and overwrites the files here, this
README included; `--only` renders the named folders alone. This README was written on
{day}.
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


def training_text(train: list[str]) -> str:
    names = [f"take {int(name.split('_')[1]):02d}" for name in train]
    return "the nine takes" if len(names) == N_TAKES else f"{' and '.join(names)} alone"


def write_readme() -> None:
    """The output folder's README.md, its numbers read from the runs' configurations."""
    esns = {esn: esn_of(stem) for esn, (_, stem, _) in ESNS.items()}
    fixed = ("n_neurons", "sparsity", "bias", "bias_scaling", "warmup", "dt", "teacher_noise_deg")
    first = next(iter(esns.values()))[0]
    if any(esn[key] != first[key] for esn, _ in esns.values() for key in fixed):
        msg = f"the ESNs differ in what the README gives as fixed: {fixed}"
        raise ValueError(msg)
    if not (first["bias"] and first["bias_scaling"] == 1.0 and first["teacher_noise_deg"] == 0.0):
        msg = "the README's bias or training does not match the ESNs'"
        raise ValueError(msg)
    arms = "\n".join(
        f"  - **{label}**, {color}: {setting_text(esns[esn][0])}; trained on {training_text(esns[esn][1])}."
        for esn, (label, _, color) in ESNS.items()
    )
    stems = [stem for _, stem, _ in ESNS.values()] + [pushes_60s(n, times) for n in "12" for times in SIZES]
    tracker = config("robot_esn1")["tracker"]
    if any(config(stem)["tracker"] != tracker for stem in stems):
        msg = "the runs' trackers differ"
        raise ValueError(msg)
    pushes = config(pushes_60s("1", 1))
    tables = pushes["disturbance"]
    parts = [
        f"{t['force']:g} N at {t['angle_deg']:g}°".replace("-", "\N{MINUS SIGN}")
        + (f" where the take's hand is {t['way']:.0%} of the way to the target" if "way" in t else "")
        + (f" at t = {t['onset']:g} s" if "onset" in t else "")
        for t in tables
    ]
    forces = {
        times: ", ".join(f"{t['force']:g}" for t in config(pushes_60s("1", times))["disturbance"]) for times in SIZES
    }
    take_times = [StateLog.load(take_path(index)).times for index in range(N_TAKES)]
    lengths = [float(times[-1] - times[0]) for times in take_times]
    links = SKELETON.links[1:]
    commit, dirty = commit_state()
    (OUT / "README.md").write_text(
        README.format(
            lengths=" and ".join(f"{link.prop.length:g}" for link in links),
            masses=" and ".join(f"{link.prop.m:g}" for link in links),
            sim_ms=1e3 * _ARM["simulator"]["dt"],
            tx=TARGET[0],
            ty=TARGET[1],
            tol_mm=1e3 * TOLERANCE,
            shortest=min(lengths),
            longest=max(lengths),
            arms=arms,
            lead=pushes["progress"]["lead_deg"],
            window=pushes["progress"]["window"],
            neurons=first["n_neurons"],
            sparsity=first["sparsity"],
            warmup=first["warmup"],
            period_ms=1e3 * first["dt"],
            omega=tracker["omegas"][0],
            damping=tracker["dampings"][0],
            control_ms=1e3 * tracker["control_period"],
            push_s=tables[0]["duration"],
            pushes="; ".join(parts[:-1]) + f"; and {parts[-1]}",
            forces_2x=f"{forces[2]} N",
            forces_4x=f"{forces[4]} N",
            short=config("robot_esn1")["evaluation"]["duration"],
            long=pushes["evaluation"]["duration"],
            cut_ms=1e3 * CUT,
            fps=FPS,
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
    """Make the files of ``items`` in ``workers`` processes, the videos first, as they take longest."""
    os.environ["OPENBLAS_NUM_THREADS"] = "1"  # for the processes started from here on
    for item in items:
        (OUT / item.path).parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(items, key=lambda item: (not item.path.endswith(".mp4"), "3.4_pushes" not in item.path))
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(item.make, out=OUT / item.path): item for item in ordered}
        for done, future in enumerate(as_completed(futures), start=1):
            if future.exception() is not None:
                print(f"failed: {futures[future].path}", flush=True)
                pool.shutdown(cancel_futures=True)
                future.result()
            if done % 50 == 0 or done == len(futures):
                print(f"made {done} of {len(futures)} files", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=8, help="processes that render the files (default: 8)")
    parser.add_argument(
        "--only", nargs="+", choices=FOLDERS, metavar="FOLDER", help=f"render these folders alone, of {FOLDERS}"
    )
    args = parser.parse_args()
    check_runs()
    items = takes_media() + takes_starts_media() + unseen_media() + pushes_media() + one_take_media()
    chosen = [item for item in items if args.only is None or item.path.split("/")[0] in args.only]
    print(f"{len(chosen)} of {len(items)} files to make into {OUT}", flush=True)
    render(chosen, args.workers)
    write_media_csv(items)
    write_readme()
    print(f"wrote media.csv and README.md to {OUT}")


if __name__ == "__main__":
    main()
