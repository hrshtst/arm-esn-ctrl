# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 025.

    uv run python reports/025-three-targets-by-context/make_figures.py

It reads the copies of the runs under ``results/`` (their configurations, run records,
and CSV files), the takes and the demonstrations' logs under ``data/``, and the sweeps'
runs and the robot runs' logs under the storage root, which are not kept in Git. It
prints the tables of the report and draws into ``results/summary``:

- ``starts.png``: the takes to the three targets and the two sets of start postures
  (Section 2);
- ``search_coarse.png``: the coarse search's settings by their runs that hold (Section 3.1);
- ``rescoring.png``: the fine search's 20 best settings, their runs that hold at each
  target there and when scored again with the untaught starts (Section 3.1);
- ``paths_esn1.png``, ``paths_esn2.png``: each ESN's hand paths, undisturbed, with each
  context, from the takes' starts and from the unseen ones, over the replay's (Sections
  3.2 and 3.3);
- ``unseen.png``: the detour ratio, the reference's fastest joint speed, and the largest
  torque from each unseen start, at each target (Section 3.3);
- ``joints_esn1.png``, ``torques_esn1.png``, ``joints_esn2.png``, ``torques_esn2.png``:
  each ESN's joint angles or torques over time against the replay's, undisturbed, from a
  take's start and from an unseen start at each target (Sections 3.2 and 3.3);
- ``switch_paths_esn1.png``, ``switch_paths_esn2.png``: the hand's paths when the context
  changes, at rest and on the way (Sections 3.4 and 3.5);
- ``switch_joints_esn1.png``, ``switch_torques_esn1.png``, ``switch_joints_esn2.png``,
  ``switch_torques_esn2.png``: each ESN's joint angles or torques over time when the
  context changes, one run of each change, against the replay's (Sections 3.4 and 3.5);
- ``switch_distance.png``: the hand's distance to the new target over time from the
  change, in every run that changes (Sections 3.4 and 3.5);
- ``demonstration_esn1.png``, ``demonstration_esn2.png``: each ESN's run pushed by hand
  in the robot app while the target changes, the pushes shaded over time (Section 3.6).
"""

from __future__ import annotations

import csv
import filecmp
import functools
import hashlib
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

from arm_esn_ctrl.demonstrations import endpoint_positions, resample_joint_angles
from arm_esn_ctrl.storage import resolve_run_path, storage_root

REPORT = Path(__file__).resolve().parent
REPO_ROOT = REPORT.parents[1]
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"
DATA = REPORT / "data"
TAKES = DATA / "reach_manual_v2_targets"  # the 27 takes as recorded
ARM_CONFIG = REPO_ROOT / "experiments" / "demonstrations" / "reach_manual_v2_targets.toml"
EXPERIMENT = "manual_demonstration_v2_targets"
CONFIGS = REPO_ROOT / "experiments" / EXPERIMENT
STAGE_1 = REPO_ROOT / "experiments" / "manual_demonstration_v2_many_starts"  # report 023's configurations
TRACKER = "pd_w20_z0.4"  # the one tracker's directory in a robot run
DT = 0.01  # the ESNs' period (s)
HOLD = 2.0  # the hold window after arrival (s)
DPI = 90
# The runs pushed by hand in the robot app while the target changes, one of each ESN, saved there by the maintainer.
RECORDED = DATA / "recorded_forces"
DEMONSTRATIONS = {
    "ESN #1": RECORDED / "20261119-094959-20261119-000048-esn1_lr0.17_sr1.05_is1_ridge10_seed1_pd_w20_z0.4.sklog.npz",
    "ESN #2": RECORDED / "20261119-095414-20261119-000238-esn2_lr0.17_sr1.2_is1_ridge10_seed5_pd_w20_z0.4.sklog.npz",
}

with ARM_CONFIG.open("rb") as _f:
    _ARM = tomllib.load(_f)
SKELETON = Skeleton.from_toml(ARM_CONFIG)
TARGETS = {target["label"]: np.array(target["pos"]) for target in _ARM["task"]["targets"]}  # m, the contexts' order
TOLERANCE = float(_ARM["task"]["targets"][0]["tolerance"])  # the goal's radius (m), the same at every target
LABELS = tuple(TARGETS)
STRIDE = round(DT / _ARM["simulator"]["dt"])  # the simulation steps in an ESN period
DEMO_RUNS = {label: next(DATA.glob(f"*-reach_manual_v2_targets_{label.lower()}")) for label in LABELS}
N_TAKES = 9  # each target's takes

ESNS = {"ESN #1": "1", "ESN #2": "2"}  # the chosen ESNs, by the number in their runs' names
# When the context changes in the runs of claims 4 and 5: the start of their configurations' names.
TIMINGS = {"at rest": "switch_to", "on the way": "switch_early_to"}
CHANGES = [(first, new) for first in LABELS for new in LABELS if first != new]  # the six changes of context
# The runs shown over time: each target's first take, demo_00, from its start (also in the runs that change), and
# the unseen start from which the replay's largest torque is the median of the 36, the lower of the two middle ones.
TAKE_SHOWN = "demo_00_filtered"
UNSEEN_SHOWN = {"A": "unseen 18", "B": "unseen 17", "C": "unseen 6"}
NAMES = ("leak_rate", "spectral_radius", "input_scaling", "ridge")
# A reference that jumps: its fastest joint speed above this (deg/s), and the run's largest torque above this (N m).
JUMP_SPEED = 500.0
JUMP_TORQUE = 20.0
NEAR = 45.0  # an unseen start near the start of the take the replay plays: within this, over both joints (deg)
LOW_HAND = -0.2  # no take's hand starts below this (y, m)
# The window after each change of context in a demonstration, whose peaks are tabulated apart from the rest (s).
AFTER_CHANGE = 2.0
# The time after each push's release left out, with the pushes, where a demonstration's arm is compared with its
# reference (s).
AFTER_RELEASE = 0.5
# The last part of a stretch of a demonstration, over which the hand's speed at the stretch's end is measured (s).
SPEED_WINDOW = 0.1

TEXT_COLOR = "#52514e"
TITLE_COLOR = "#0b0b0b"
GRID_COLOR = "#e1e0d9"
SURFACE_COLOR = "#fcfcfb"
TAUGHT_COLOR = "#b9b7b0"
NONE_COLOR = "#f0efec"  # a heatmap cell of zero
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
ESN_COLOR, REPLAY_COLOR = "#2a78d6", "#eb6834"
ESN_COLORS = {"ESN #1": "#1c5cab", "ESN #2": "#6da7ec"}  # where both are drawn
TARGET_COLORS = {"A": "#7b3fa0", "B": "#1baf7a", "C": "#eda100"}  # the recorder's purple, teal, and orange
PUSH_COLOR, PUSH_ALPHA = TITLE_COLOR, 0.2  # the band over a push in the demonstrations' panels over time
# Where each target's letter goes from it, off the paths into it (pt).
LABEL_OFFSETS = {"A": (-16, 6), "B": (6, 8), "C": (6, -16)}
# The layers of the lines over time: the ESN's arm on top, its reference (dashed) under it, the replay's below.
ARM_ZORDERS = {"esn": 3.0, "replay": 2.0}
REFERENCE_ZORDER = 2.5


def main() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Callable[[], Figure]] = {
        "starts.png": plot_starts,
        "search_coarse.png": plot_coarse,
        "rescoring.png": plot_rescoring,
        "unseen.png": plot_unseen,
        "switch_distance.png": plot_switch_distance,
    }
    for name, n in ESNS.items():
        figures[f"paths_esn{n}.png"] = functools.partial(plot_paths, name)
        figures[f"switch_paths_esn{n}.png"] = functools.partial(plot_switch_paths, name)
        for channel in ("joints", "torques"):
            figures[f"{channel}_esn{n}.png"] = functools.partial(plot_series, name, channel)
            figures[f"switch_{channel}_esn{n}.png"] = functools.partial(plot_switch_series, name, channel)
        figures[f"demonstration_esn{n}.png"] = functools.partial(plot_demonstration, name)
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


def stored(stem: str) -> Path:
    """A run's directory under the storage root."""
    return resolve_run_path(f"results/{EXPERIMENT}/{run_name(stem)}")


@functools.cache
def config_of(stem: str) -> dict[str, Any]:
    with (RESULTS / run_name(stem) / "config.toml").open("rb") as f:
        return tomllib.load(f)


def robot(n: str) -> str:
    """The undisturbed robot run of ESN #``n``, of claims 1 to 3."""
    return f"robot_esn{n}"


def switch(n: str, timing: str, new: str) -> str:
    """The robot run of ESN #``n`` whose context changes to target ``new``, at rest or on the way."""
    return f"{TIMINGS[timing]}_{new.lower()}_robot_esn{n}"


def change_time(stem: str) -> float:
    """When the context changes in a run of claims 4 and 5, on the task clock (s)."""
    return float(config_of(stem)["switch"][0]["time"])


def run_end(stem: str) -> float:
    return float(config_of(stem)["evaluation"]["duration"])


def rows(stem: str, arm: str, task: str) -> list[dict[str, Any]]:
    """One arm's runs with one target's context in a robot run, in the order of their starts."""
    table = read_csv(RESULTS / run_name(stem) / "metrics.csv")
    return [row for row in table if row["arm"] == arm and row["task"] == task]


def group(stem: str, arm: str, task: str, which: str) -> list[dict[str, Any]]:
    """One arm's runs with one target's context from a set of starts: ``demo`` (the takes') or ``unseen``."""
    return [row for row in rows(stem, arm, task) if row["origin"].startswith(which)]


def start_of(stem: str, task: str, origin: str) -> int:
    return int(next(row["start"] for row in rows(stem, "esn", task) if row["origin"] == origin))


@functools.lru_cache(maxsize=32)
def log(stem: str, arm: str, task: str, start: int) -> StateLog:
    """One arm's log with one target's context from a robot run under the storage root."""
    return StateLog.load(stored(stem) / task / TRACKER / f"{arm}_{start:02d}.sklog.npz")


@functools.cache
def path(stem: str, arm: str, task: str, start: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A run's hand over the task, every ESN period to its end: the times (s) and the hand's places (m)."""
    state = log(stem, arm, task, start)
    every = np.flatnonzero(state.times > -1e-9)[::STRIDE]
    return state.times[every], hand(state.channel("q")[every])


def scores(stem: str) -> list[dict[str, Any]]:
    return read_csv(RESULTS / run_name(stem) / "scores.csv")


def sweep_runs(stem: str) -> list[dict[str, Any]]:
    """Every run of a sweep, from its ``runs.csv`` under the storage root."""
    return read_csv(stored(stem) / "runs.csv")


def setting_of(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(row[name] for name in NAMES)


@functools.cache
def take(task: str, index: int) -> NDArray[np.float64]:
    """A take filtered at 2 Hz, what the ESNs were trained on, every 10 ms: its joint angles (rad)."""
    return resample_joint_angles(StateLog.load(DEMO_RUNS[task] / f"demo_{index:02d}_filtered.sklog.npz"), DT)[1]


def take_rows(task: str, filtered: bool) -> list[dict[str, Any]]:
    """The measures of a target's nine takes, as recorded or filtered, from its demonstration run."""
    table = read_csv(DEMO_RUNS[task] / "metrics.csv")
    return [row for row in table if row["demo"].endswith("_filtered") == filtered]


def take_start(task: str, index: int) -> NDArray[np.float64]:
    """Where a take starts (deg)."""
    row = take_rows(task, True)[index]
    return np.array([row["start_q1_deg"], row["start_q2_deg"]])


@functools.cache
def extra_starts(config: str, name: str) -> NDArray[np.float64]:
    """A set of start postures no take starts from, as a configuration lists it (deg)."""
    with (CONFIGS / config).open("rb") as f:
        return np.array(tomllib.load(f)["evaluation"]["extra_starts"][name])


def start_posture(task: str, row: dict[str, Any]) -> NDArray[np.float64]:
    """Where a robot run starts (deg): a take's start or an unseen one."""
    if row["origin"].startswith("demo"):
        return take_start(task, int(row["origin"].split("_")[1]))
    return extra_starts("robot_esn1.toml", "unseen")[int(row["origin"].split()[1])]


@functools.cache
def demonstration(name: str) -> StateLog:
    """The run of an ESN pushed by hand in the robot app while the target changes, from its saved log."""
    return StateLog.load(DEMONSTRATIONS[name])


# ---------------------------------------------------------------------------- measures


def hand(q: NDArray[np.float64]) -> NDArray[np.float64]:
    return endpoint_positions(SKELETON, q)


def holds(row: dict[str, Any]) -> bool:
    """The hand arrives within the goal and stays for the 2 s after."""
    return bool(row["success"]) and row["hold_observed_s"] >= HOLD - 1e-9


def failure(row: dict[str, Any]) -> str:
    """How a run that does not hold fails."""
    if not row["arrived"]:
        return "never arrives"
    return "arrives and leaves within the 2 s" if row["left_goal"] else "arrives too late for the 2 s"


def peaks(state: StateLog, part: NDArray[np.bool_]) -> tuple[float, float]:
    """The reference's fastest joint speed (°/s) and the largest joint torque (N m) over a part of a log, as
    ``experiments/robot_esn.py`` measures them over the task."""
    q_ref, times = state.channel("q_ref")[part], state.times[part]
    speed = np.degrees(np.abs(np.diff(q_ref, axis=0)).max() / np.diff(times).min())
    return float(speed), float(np.abs(state.channel("tau")[part]).max())


@functools.cache
def at_change(stem: str, task: str, start: int) -> dict[str, float]:
    """An ESN run whose context changes, from its log: the hand's distances to its first target and to the new one
    at the change (m), and the reference's fastest joint speed and the largest torque before the change and after."""
    state = log(stem, "esn", task, start)
    time, new = change_time(stem), config_of(stem)["switch"][0]["task"]
    times = state.times
    k = int(np.searchsorted(times, time - 1e-9))
    place = hand(state.channel("q")[k : k + 1])[0]
    before = peaks(state, (times > -1e-9) & (times < time - 1e-9))
    after = peaks(state, times > time - 1e-9)
    return {
        "to_first": float(np.linalg.norm(place - TARGETS[task])),
        "to_new": float(np.linalg.norm(place - TARGETS[new])),
        "speed_before": before[0],
        "torque_before": before[1],
        "speed_after": after[0],
        "torque_after": after[1],
    }


def median(values: list[float]) -> float:
    return float(np.nanmedian(np.array(values, dtype=float)))


def column(runs: list[dict[str, Any]], key: str) -> list[float]:
    return [float(run[key]) for run in runs]


def joint_distance(task: str, row: dict[str, Any]) -> float:
    """How far a run's start is from the start of the take the replay plays (deg, over the two joints)."""
    played = take_start(task, int(row["replayed"].split("_")[1]))
    return float(np.linalg.norm(start_posture(task, row) - played))


def played_distances(task: str) -> NDArray[np.float64]:
    """How far each unseen start is from the start of the take the replay plays with ``task``'s context (deg)."""
    return np.array([joint_distance(task, replay) for replay in group(robot("1"), "replay", task, "unseen")])


def rest_postures(task: str) -> NDArray[np.float64]:
    """Where each of a target's nine takes ends (deg)."""
    return np.degrees(np.array([take(task, index)[-1] for index in range(N_TAKES)]))


def named_targets(
    state: StateLog, points: NDArray[np.float64]
) -> tuple[list[str], NDArray[np.intp], NDArray[np.float64]]:
    """The context of a robot app's log: its targets in the order of its columns, the index of the one it names at
    each step, and the hand's distance to that one at each step (m), from the hand's ``points``."""
    labels = state.extra["robot_app"]["tasks"]
    named = np.argmax(state.channel("context"), axis=1)
    places = np.array([TARGETS[label] for label in labels])[named]
    return labels, named, np.linalg.norm(points - places, axis=1)


def stretches(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """The stretches of a mask where it holds, each as its first index and the index after its last."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(int), [0]))))
    return [(int(first), int(after)) for first, after in zip(edges[::2], edges[1::2], strict=True)]


def changes_and_pushes(state: StateLog) -> tuple[list[int], list[tuple[int, int]]]:
    """A robot app's log over the task: its changes of context, each as the first step with the new one, and its
    pushes, each as a stretch of steps where the tip force is not zero. The force is exactly zero while the tip is
    not dragged, so one drag is one push."""
    task = state.times > -1e-9
    first = int(np.flatnonzero(task)[0])
    named = np.argmax(state.channel("context"), axis=1)
    changes = [first + 1 + int(k) for k in np.flatnonzero(np.diff(named[first:]))]
    force = np.linalg.norm(state.channel("ext_force"), axis=1)
    return changes, stretches(task & (force > 0.0))


# ---------------------------------------------------------------------------- figures


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def workspace(ax: Axes, title: str = "", named: str = "") -> None:
    """The workspace's axes: the ring the hand reaches, the three goals (``named``'s bold), and equal scales."""
    style(ax)
    upper, forearm = (link.prop.length for link in SKELETON.links[1:])
    for radius in (abs(upper - forearm), upper + forearm):
        ax.add_patch(Circle((0.0, 0.0), radius, fill=False, edgecolor=GRID_COLOR, linewidth=1.2, zorder=1))
    for label, target in TARGETS.items():
        bold = label == named
        color = TARGET_COLORS[label]
        ax.add_patch(Circle(tuple(target), TOLERANCE, fill=False, edgecolor=color, linewidth=2.4 if bold else 1.2))
        ax.plot(*target, marker="+", color=color, markersize=10 if bold else 7, zorder=5)
        ax.annotate(
            label,
            tuple(target),
            textcoords="offset points",
            xytext=LABEL_OFFSETS[label],
            color=color,
            fontsize=11 if bold else 10,
            fontweight="bold" if bold else "normal",
            bbox={"boxstyle": "round,pad=0.1", "facecolor": SURFACE_COLOR, "edgecolor": "none", "alpha": 0.8},
            zorder=5,
        )
    ax.plot(0.0, 0.0, marker="s", color=TEXT_COLOR, markersize=4, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(-0.95, 0.95)
    ax.set_ylim(-0.95, 0.95)
    if title:
        ax.set_title(title, color=TITLE_COLOR, fontsize=10)


def draw_path(ax: Axes, points: NDArray[np.float64], color: str, alpha: float = 0.9, zorder: float = 3.0) -> None:
    """A hand path, its start as a dot."""
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1.0, alpha=alpha, zorder=zorder)
    ax.plot(*points[0], marker="o", color=color, markersize=3.5, zorder=zorder + 1)


def xy_labels(axes: Any) -> None:
    for ax in axes[-1]:
        ax.set_xlabel("x (m)", color=TEXT_COLOR)
    for ax in axes[:, 0]:
        ax.set_ylabel("y (m)", color=TEXT_COLOR)


def plot_starts() -> Figure:
    fig = Figure(figsize=(8.2, 7.8), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The 27 takes' hand paths, nine to each target, and the start postures, by their hands", color=TITLE_COLOR
    )
    ax = fig.subplots()
    workspace(ax)
    for label in LABELS:
        for index in range(N_TAKES):
            points = hand(take(label, index))
            ax.plot(points[:, 0], points[:, 1], color=TARGET_COLORS[label], linewidth=1.6, alpha=0.75, zorder=2)
            ax.plot(*points[0], marker="o", color=TARGET_COLORS[label], markersize=6, zorder=4)
    untaught = hand(np.radians(extra_starts("untaught_search.toml", "untaught")))
    unseen = hand(np.radians(extra_starts("robot_esn1.toml", "unseen")))
    ax.plot(untaught[:, 0], untaught[:, 1], linestyle="", marker="s", color=TEXT_COLOR, markersize=4.5, zorder=3)
    ax.plot(unseen[:, 0], unseen[:, 1], linestyle="", marker="^", color=ESN_COLOR, markerfacecolor="none", markersize=6)
    ax.set_xlabel("x (m)", color=TEXT_COLOR)
    ax.set_ylabel("y (m)", color=TEXT_COLOR)
    handles = [
        Line2D([], [], color=TARGET_COLORS[label], linewidth=1.6, marker="o", label=f"a take to {label}")
        for label in LABELS
    ]
    handles += [
        Line2D([], [], linestyle="", marker="s", color=TEXT_COLOR, label="untaught start (the scoring)"),
        Line2D([], [], linestyle="", marker="^", color=ESN_COLOR, markerfacecolor="none", label="unseen start"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
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


def plot_coarse() -> Figure:
    """The coarse search: each setting's runs that hold, of the 27 from the takes' starts, and at each target."""
    table = scores("coarse_search")
    by_setting = {setting_of(row): row for row in table}
    center = setting_of(table[0])
    grid = config_of("coarse_search")["sweep"]
    leaks, radii = grid["leak_rate"], grid["spectral_radius"]
    scalings, ridges = grid["input_scaling"], grid["ridge"]
    total = 3 * N_TAKES
    fig = Figure(figsize=(11, 8.2), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The coarse search (one reservoir a setting): runs that arrive and hold, of the"
        f" {total} from the takes' starts;\nbelow, at A, B, and C, of nine each; outlined: all {total} hold;"
        " C: the best setting, the fine search's center",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(len(ridges), len(scalings), sharex=True, sharey=True)
    cmap, norm = count_colormap(total)
    for i, ridge in enumerate(reversed(ridges)):
        for j, scaling in enumerate(scalings):
            ax = axes[i, j]
            cells = [[by_setting[(leak, radius, scaling, ridge)] for leak in leaks] for radius in radii]
            counts = np.array([[cell["holding_runs"] for cell in line] for line in cells])
            ax.imshow(counts, cmap=cmap, norm=norm, origin="lower", aspect="auto")
            for y, line in enumerate(cells):
                for x, cell in enumerate(line):
                    count = int(cell["holding_runs"])
                    each = " ".join(str(int(cell[f"holding_runs_{label}"])) for label in LABELS)
                    mark = " C" if setting_of(cell) == center else ""
                    ink = ink_on(cmap(norm(count)))
                    ax.text(x, y + 0.12, f"{count}{mark}", ha="center", va="center", fontsize=10, color=ink)
                    ax.text(x, y - 0.22, each, ha="center", va="center", fontsize=7.5, color=ink)
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
    return fig


def plot_rescoring() -> Figure:
    """The fine search's 20 best settings: their runs that hold at each target there and with the untaught starts."""
    rescored = scores("untaught_search")
    fine = {setting_of(row): row for row in scores("fine_search")}
    fig = Figure(figsize=(13, 7.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The fine search's 20 best settings, five reservoirs each, ranked by their scoring with the untaught starts:"
        "\nruns that arrive and hold at each target; in parentheses, of all three, in the fine search and in the"
        " scoring",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(1, 2, sharey=True)
    panels = [
        (fine, "in the fine search, from the takes' starts: of 45 a target"),
        ({setting_of(row): row for row in rescored}, "scored again with the 36 untaught starts too: of 225 a target"),
    ]
    count = len(rescored)
    for ax, (table, label) in zip(axes, panels, strict=True):
        style(ax)
        for rank, row in enumerate(rescored):
            each = table[setting_of(row)]
            for offset, target in zip((0.22, 0.0, -0.22), LABELS, strict=True):
                ax.plot(
                    each[f"holding_runs_{target}"],
                    count - 1 - rank + offset,
                    marker="o",
                    linestyle="",
                    color=TARGET_COLORS[target],
                    markersize=6,
                )
        ax.set_xlabel(label, color=TEXT_COLOR)
    names = {0: "ESN #1: ", 1: "ESN #2: "}
    ticks = [
        f"{names.get(rank, '')}{setting_text(row)} ({int(fine[setting_of(row)]['holding_runs'])};"
        f" {int(row['holding_runs'])})"
        for rank, row in enumerate(rescored)
    ]
    axes[0].set_yticks(range(count), list(reversed(ticks)))
    axes[0].set_ylim(-0.6, count - 0.4)
    for tick, rank in zip(axes[0].get_yticklabels(), reversed(range(count)), strict=True):
        if rank in names:
            tick.set_color(ESN_COLOR)
            tick.set_fontweight("bold")
    handles = [
        Line2D([], [], linestyle="", marker="o", color=TARGET_COLORS[label], label=f"at {label}") for label in LABELS
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_paths(name: str) -> Figure:
    """One ESN's hand paths, undisturbed, with each context: from the takes' starts above, the unseen ones below."""
    stem = robot(ESNS[name])
    fig = Figure(figsize=(15, 10.8), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        f"The hand's paths of {name} (blue) on the robot, undisturbed, over the replay's of the take to the context's"
        " target that starts nearest (orange)",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(2, 3)
    for row, (which, heading) in enumerate(
        (("demo", "from its nine takes' starts"), ("unseen", "from the 36 unseen starts"))
    ):
        for col, label in enumerate(LABELS):
            ax = axes[row, col]
            workspace(ax, f"context {label}, {heading}", named=label)
            for arm, color, alpha, zorder in (("replay", REPLAY_COLOR, 0.6, 2.0), ("esn", ESN_COLOR, 0.8, 3.0)):
                for run in group(stem, arm, label, which):
                    draw_path(ax, path(stem, arm, label, int(run["start"]))[1], color, alpha, zorder)
    xy_labels(axes)
    handles = [
        Line2D([], [], color=ESN_COLOR, marker="o", markersize=3.5, label=f"{name}'s arm"),
        Line2D([], [], color=REPLAY_COLOR, marker="o", markersize=3.5, label="the replay's arm"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_unseen() -> Figure:
    """From each unseen start, at each target: the detour ratio, the reference's fastest speed, the largest torque."""
    fig = Figure(figsize=(15, 12), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "From each of the 36 unseen starts, with each context, against how far it is from the start of the take the"
        " replay plays",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(3, 3, sharex=True)
    panels = [
        ("detour_ratio", "detour ratio"),
        ("peak_reference_speed_dps", "the reference's fastest joint speed (°/s)"),
        ("peak_torque_nm", "largest joint torque (N m)"),
    ]
    arms = [
        ("the replay", robot("1"), "replay", REPLAY_COLOR, "^"),
        ("ESN #2", robot("2"), "esn", ESN_COLORS["ESN #2"], "s"),
        ("ESN #1", robot("1"), "esn", ESN_COLORS["ESN #1"], "o"),
    ]
    for row, label in enumerate(LABELS):
        for col, (key, title) in enumerate(panels):
            ax = axes[row, col]
            style(ax)
            for _, stem, arm, color, marker in arms:
                runs = group(stem, arm, label, "unseen")
                replayed = group(stem, "replay", label, "unseen")
                ax.plot(
                    [joint_distance(label, run) for run in replayed],
                    column(runs, key),
                    linestyle="",
                    marker=marker,
                    color=color,
                    markersize=5,
                    alpha=0.85,
                )
            ax.set_yscale("log")
            ax.set_title(f"context {label}", color=TITLE_COLOR, fontsize=10)
            ax.set_ylabel(title, color=TEXT_COLOR)
    for ax in axes[-1]:
        ax.set_xlabel("distance from the played take's start (°, over both joints)", color=TEXT_COLOR)
    handles = [
        Line2D([], [], linestyle="", marker=marker, color=color, label=name)
        for name, _, _, color, marker in reversed(arms)
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def draw_series(
    axes: Any, logs: dict[str, StateLog], channel: str, end: float, played: NDArray[np.float64] | None = None
) -> None:
    """Two joints' angles or torques over time, one joint an axis: the ESN's arm, its reference dashed, and the
    replay's arm beneath; with ``played``, the take the replay plays in thick gray. A torque axis is cut at the
    largest torque of the ESN's arm after the first 50 ms."""
    for arm, state in logs.items():
        color = ESN_COLOR if arm == "esn" else REPLAY_COLOR
        for joint, ax in enumerate(axes):
            if channel == "joints":
                angle = np.degrees(state.channel("q")[:, joint])
                ax.plot(state.times, angle, color=color, linewidth=1.2, zorder=ARM_ZORDERS[arm])
                if arm == "esn":
                    reference = np.degrees(state.channel("q_ref")[:, joint])
                    ax.plot(state.times, reference, color=color, linewidth=1.0, linestyle="--", zorder=REFERENCE_ZORDER)
            else:
                ax.plot(
                    state.times, state.channel("tau")[:, joint], color=color, linewidth=1.0, zorder=ARM_ZORDERS[arm]
                )
    esn = logs["esn"]
    for joint, ax in enumerate(axes):
        if channel == "joints" and played is not None:
            taught = np.degrees(played[:, joint])
            ax.plot(DT * np.arange(len(taught)), taught, color=TAUGHT_COLOR, linewidth=4.0, zorder=1)
        if channel == "torques":
            top = max(1.15 * float(np.abs(esn.channel("tau")[esn.times >= 0.05, joint]).max()), 0.5)
            ax.set_ylim(-top, top)
        ax.set_xlim(-1.0, end)


def series_legend(fig: Figure, name: str, channel: str, taught: bool) -> None:
    handles = [
        Line2D([], [], color=ESN_COLOR, linewidth=1.4, label=f"{name}'s arm"),
        Line2D([], [], color=REPLAY_COLOR, linewidth=1.4, label="the replay's arm"),
    ]
    if channel == "joints":
        handles.append(Line2D([], [], color=ESN_COLOR, linewidth=1.0, linestyle="--", label=f"{name}'s reference"))
        if taught:
            handles.append(Line2D([], [], color=TAUGHT_COLOR, linewidth=4.0, label="the take the replay plays"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)


def plot_series(name: str, channel: str) -> Figure:
    """One ESN's joint angles or torques over time against the replay's, undisturbed: a column per run shown."""
    stem = robot(ESNS[name])
    columns = [(label, origin) for label in LABELS for origin in (TAKE_SHOWN, UNSEEN_SHOWN[label])]
    fig = Figure(figsize=(19, 7.0), facecolor=SURFACE_COLOR, layout="constrained")
    what = "Joint angles" if channel == "joints" else "Joint torques"
    fig.suptitle(
        f"{what} of {name}'s arm (blue) and the replay's (orange), undisturbed, from take demo_00's start and from an"
        " unseen start with each context"
        + ("" if channel == "joints" else ";\neach axis is cut at the largest torque of the ESN's arm after 50 ms"),
        color=TITLE_COLOR,
    )
    axes = fig.subplots(2, len(columns), sharex=True)
    for col, (label, origin) in enumerate(columns):
        start = start_of(stem, label, origin)
        replay = next(row for row in rows(stem, "replay", label) if row["origin"] == origin)
        logs = {arm: log(stem, arm, label, start) for arm in ("esn", "replay")}
        for ax in axes[:, col]:
            style(ax)
        played = take(label, int(replay["replayed"].split("_")[1]))
        draw_series(axes[:, col], logs, channel, run_end(stem), played)
        posture = start_posture(label, replay)
        where = "take demo_00's start" if origin == TAKE_SHOWN else "the unseen start"
        axes[0, col].set_title(
            f"context {label}, from {where}\n({posture[0]:.0f}°, {posture[1]:.0f}°)", color=TITLE_COLOR, fontsize=10
        )
        axes[-1, col].set_xlabel("time (s)", color=TEXT_COLOR)
    unit = "deg" if channel == "joints" else "N m"
    for joint in range(2):
        axes[joint, 0].set_ylabel(f"joint {joint + 1} ({unit})", color=TEXT_COLOR)
    series_legend(fig, name, channel, taught=True)
    return fig


def plot_switch_series(name: str, channel: str) -> Figure:
    """One ESN's joint angles or torques over time when the context changes: a column per change, from the first
    target's take demo_00's start, at rest above and on the way below."""
    n = ESNS[name]
    fig = Figure(figsize=(19, 12), facecolor=SURFACE_COLOR, layout="constrained")
    what = "Joint angles" if channel == "joints" else "Joint torques"
    fig.suptitle(
        f"{what} of {name}'s arm (blue) and the replay's (orange, which keeps to its take) when the context changes"
        " (gray line), from the first target's take demo_00's start"
        + (
            ";\ndotted: where the takes to the first target and to the new one rest, in the target's color"
            if channel == "joints"
            else ";\neach axis is cut at the largest torque of the ESN's arm after 50 ms"
        ),
        color=TITLE_COLOR,
    )
    axes = fig.subplots(4, len(CHANGES), sharex="row")
    for block, timing in enumerate(TIMINGS):
        for col, (first, new) in enumerate(CHANGES):
            stem = switch(n, timing, new)
            start = start_of(stem, first, TAKE_SHOWN)
            logs = {arm: log(stem, arm, first, start) for arm in ("esn", "replay")}
            pair = axes[2 * block : 2 * block + 2, col]
            for ax in pair:
                style(ax)
                ax.axvline(change_time(stem), color=TEXT_COLOR, linewidth=1.0, zorder=1.8)  # above the grid
            draw_series(pair, logs, channel, run_end(stem))
            if channel == "joints":
                for label in (first, new):
                    rest = np.median(rest_postures(label), axis=0)
                    for joint, ax in enumerate(pair):
                        ax.axhline(rest[joint], color=TARGET_COLORS[label], linewidth=1.4, linestyle=":", zorder=1)
            pair[0].set_title(f"{first} → {new}, {timing} (at {change_time(stem):g} s)", color=TITLE_COLOR, fontsize=10)
            pair[1].set_xlabel("time (s)", color=TEXT_COLOR)
    unit = "deg" if channel == "joints" else "N m"
    for row in range(4):
        axes[row, 0].set_ylabel(f"joint {row % 2 + 1} ({unit})", color=TEXT_COLOR)
    series_legend(fig, name, channel, taught=False)
    return fig


def plot_switch_paths(name: str) -> Figure:
    """One ESN's hand paths when the context changes: a column per new target, at rest above and on the way below."""
    n = ESNS[name]
    fig = Figure(figsize=(15, 8.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        f"The hand's paths of {name} when the context changes, from the takes' starts (dots), in the color of the"
        " first target;\ndiamonds: where the hand is when the context changes; the workspace above y = -0.35 m",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(2, 3)
    for row, timing in enumerate(TIMINGS):
        for col, new in enumerate(LABELS):
            ax = axes[row, col]
            stem = switch(n, timing, new)
            time = change_time(stem)
            workspace(ax, f"changed to {new} {timing}, at {time:g} s", named=new)
            ax.set_ylim(-0.35, 0.95)  # every run that changes stays above it (to -0.29 m)
            for first in LABELS:
                if first == new:
                    continue
                for start in range(N_TAKES):
                    times, points = path(stem, "esn", first, start)
                    draw_path(ax, points, TARGET_COLORS[first], 0.75)
                    k = int(np.searchsorted(times, time - 1e-9))
                    ax.plot(
                        *points[k],
                        marker="D",
                        color=TARGET_COLORS[first],
                        markeredgecolor=TITLE_COLOR,
                        markersize=5,
                        zorder=6,
                    )
    xy_labels(axes)
    handles = [Line2D([], [], color=TARGET_COLORS[label], label=f"first to {label}") for label in LABELS]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_switch_distance() -> Figure:
    """The hand's distance to the new target from the change, in every run that changes."""
    fig = Figure(figsize=(15, 9.4), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The hand's distance to the new target from the change of context, in each of the 54 runs that change,"
        " in the color of the new target;\ndashed: the goal's radius, 20 mm",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(len(TIMINGS), len(ESNS), sharex=True, sharey=True)
    for row, timing in enumerate(TIMINGS):
        for col, (name, n) in enumerate(ESNS.items()):
            ax = axes[row, col]
            style(ax)
            for first, new in CHANGES:
                stem = switch(n, timing, new)
                time = change_time(stem)
                for start in range(N_TAKES):
                    times, points = path(stem, "esn", first, start)
                    after = times > time - 1e-9
                    distance = 1e3 * np.linalg.norm(points[after] - TARGETS[new], axis=1)
                    ax.plot(times[after] - time, distance, color=TARGET_COLORS[new], linewidth=1.0, alpha=0.7)
                ax.set_xlim(0.0, run_end(stem) - time)
            ax.axhline(1e3 * TOLERANCE, color=TITLE_COLOR, linewidth=1.2, linestyle="--", zorder=3)
            ax.set_yscale("log")
            ax.set_ylim(1.0, 2500.0)
            ax.set_title(f"{name}, changed {timing}", color=TITLE_COLOR, fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("time from the change (s)", color=TEXT_COLOR)
    for ax in axes[:, 0]:
        ax.set_ylabel("distance to the new target (mm)", color=TEXT_COLOR)
    handles = [Line2D([], [], color=TARGET_COLORS[label], label=f"changed to {label}") for label in LABELS]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_demonstration(name: str) -> Figure:
    """The run of an ESN pushed by hand in the robot app while the target changes, from its saved log: the hand's
    path in the color of the target its context names, and over time the joint angles and their reference, the
    torques, the tip force, the hand's distance to that target, and the context, each push shaded."""
    state = demonstration(name)
    times = state.times
    task = times > -1e-9
    points = hand(state.channel("q"))
    labels, named, distance = named_targets(state, points)
    changes, pushes = changes_and_pushes(state)
    force = np.linalg.norm(state.channel("ext_force"), axis=1)

    fig = Figure(figsize=(17, 10), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        f"{name} in the robot app, pushed by hand while the target changes: the hand's path in the color of the target"
        " the context names;\nthick black, and shaded over time: while a force acts; diamonds: where the hand is when"
        " the context changes (gray lines)",
        color=TITLE_COLOR,
    )
    grid = fig.add_gridspec(6, 2, width_ratios=[1.0, 1.5], height_ratios=[1.0, 1.0, 1.0, 0.8, 1.0, 0.45])
    ax = fig.add_subplot(grid[:, 0])
    workspace(ax)
    for k, label in enumerate(labels):
        part = np.where((task & (named == k))[:, None], points, np.nan)
        ax.plot(part[:, 0], part[:, 1], color=TARGET_COLORS[label], linewidth=1.2, zorder=3)
    pushed = np.where((task & (force > 0.0))[:, None], points, np.nan)
    ax.plot(pushed[:, 0], pushed[:, 1], color=TITLE_COLOR, linewidth=2.4, zorder=3.5)
    ax.plot(*points[task][0], marker="o", color=TARGET_COLORS[labels[named[task][0]]], markersize=5, zorder=4)
    for k in changes:
        color = TARGET_COLORS[labels[named[k]]]
        ax.plot(*points[k], marker="D", color=color, markeredgecolor=TITLE_COLOR, markersize=6, zorder=6)
    ax.set_xlabel("x (m)", color=TEXT_COLOR)
    ax.set_ylabel("y (m)", color=TEXT_COLOR)

    series = [fig.add_subplot(grid[row, 1]) for row in range(6)]
    for other in series[1:]:
        other.sharex(series[0])
    for row, ax_ in enumerate(series):
        style(ax_)
        for k in changes:
            ax_.axvline(times[k], color=TEXT_COLOR, linewidth=1.0, zorder=1.8)  # above the grid
        for a, b in pushes:
            ax_.axvspan(times[a], times[b - 1], color=PUSH_COLOR, alpha=PUSH_ALPHA, linewidth=0.0, zorder=1.0)
        ax_.set_xlim(times[0], times[-1])
        if row < len(series) - 1:
            ax_.tick_params(labelbottom=False)
    joint_colors = list(ESN_COLORS.values())
    for joint in range(2):
        q, q_ref = np.degrees(state.channel("q")[:, joint]), np.degrees(state.channel("q_ref")[:, joint])
        series[joint].plot(times, q, color=ESN_COLOR, linewidth=1.2, zorder=3)
        series[joint].plot(times, q_ref, color=ESN_COLOR, linewidth=1.0, linestyle="--", zorder=REFERENCE_ZORDER)
        series[joint].set_ylabel(f"joint {joint + 1} (deg)", color=TEXT_COLOR)
        series[2].plot(times, state.channel("tau")[:, joint], color=joint_colors[joint], linewidth=1.0)
    series[2].set_ylabel("torque (N m)", color=TEXT_COLOR)
    series[3].plot(times, force, color=TITLE_COLOR, linewidth=1.0)
    series[3].set_ylabel("tip force (N)", color=TEXT_COLOR)
    for k, label in enumerate(labels):
        series[4].plot(times, np.where(named == k, 1e3 * distance, np.nan), color=TARGET_COLORS[label], linewidth=1.2)
        series[5].plot(
            times, np.where(named == k, k, np.nan), color=TARGET_COLORS[label], linewidth=6.0, solid_capstyle="butt"
        )
    series[4].axhline(1e3 * TOLERANCE, color=TITLE_COLOR, linewidth=1.0, linestyle="--")
    series[4].set_yscale("log")
    series[4].set_ylabel("to the named\ntarget (mm)", color=TEXT_COLOR)
    series[5].set_yticks(range(len(labels)), labels)
    series[5].set_ylim(-0.6, len(labels) - 0.4)
    series[5].set_ylabel("context", color=TEXT_COLOR)
    series[5].set_xlabel("time (s)", color=TEXT_COLOR)
    handles = [
        Line2D([], [], color=ESN_COLOR, linewidth=1.2, label="the arm"),
        Line2D([], [], color=ESN_COLOR, linewidth=1.0, linestyle="--", label="its reference"),
        Line2D([], [], color=joint_colors[0], linewidth=1.0, label="joint 1's torque"),
        Line2D([], [], color=joint_colors[1], linewidth=1.0, label="joint 2's torque"),
        Patch(facecolor=PUSH_COLOR, alpha=PUSH_ALPHA, label="a push"),
    ]
    handles += [Line2D([], [], color=TARGET_COLORS[label], linewidth=2.0, label=f"target {label}") for label in labels]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def print_demonstration(name: str) -> None:
    """The run of an ESN pushed by hand in the robot app, from its saved log: where it starts; its changes of
    context and its pushes; the stretches between events, and whether the hand arrives at the named target in each;
    the reference's fastest joint speed and the largest torque, over the run, after each change, and elsewhere; and
    how far the arm is from its reference where no hand acts.

    The events are the task's start, each change of context, and each push's release. A stretch runs from an event
    to the next push's onset, the next change, or the run's end."""
    print(f"\n== The demonstration of {name}: in the robot app, pushed by hand while the target changes")
    state = demonstration(name)
    times = state.times
    task = times > -1e-9
    points = hand(state.channel("q"))
    labels, named, distance = named_targets(state, points)
    first = int(np.flatnonzero(task)[0])
    start = np.degrees(state.channel("q")[first])
    print(
        f"{times[-1]:.1f} s after a warm-up of {-times[0]:.1f} s; from ({start[0]:.0f}, {start[1]:.0f}) deg, the hand"
        f" at ({points[first, 0]:.2f}, {points[first, 1]:.2f}) m, with the context {labels[named[first]]}"
    )
    changes, pushes = changes_and_pushes(state)
    ends = sorted([*changes, *(a for a, _ in pushes), len(times)])  # the steps where a stretch ends

    def end_of(k: int) -> int:
        """The end of the stretch from step ``k``: the first push's onset or change after it, or the run's end."""
        return next(end for end in ends if end > k)

    print("The changes of context: time (s) | from | to | the hand's distance to the new target then (mm)")
    for k in changes:
        print(f"  {times[k]:.2f} | {labels[named[k - 1]]} | {labels[named[k]]} | {1e3 * distance[k]:.0f}")
    force = np.linalg.norm(state.channel("ext_force"), axis=1)
    gaps = np.degrees(np.abs(state.channel("q") - state.channel("q_ref"))).max(axis=1)  # the larger joint's
    print(
        f"{len(pushes)} pushes, where the tip force is not zero: push | onset (s) | duration (s) | largest force (N) |"
        " the hand moved, from the onset to the release (mm) | the largest difference of a joint's angle from its"
        " reference (deg) | context | the hand's distance to the named target (mm): at the onset | at the release |"
        " farthest from the onset to the next push's onset, change, or the run's end"
    )
    for i, (a, b) in enumerate(pushes, start=1):
        print(
            f"  {i} | {times[a]:.2f} | {(b - a) * state.extra['robot_app']['dt']:.2f} | {force[a:b].max():.1f} |"
            f" {1e3 * np.linalg.norm(points[b] - points[a]):.0f} | {gaps[a:b].max():.1f} | {labels[named[a]]} |"
            f" {1e3 * distance[a]:.0f} | {1e3 * distance[b]:.0f} | {1e3 * distance[a : end_of(a)].max():.0f}"
        )
    begins = sorted(
        [(first, "the start"), *((k, f"the change to {labels[named[k]]}") for k in changes)]
        + [(b, f"push {i}'s release") for i, (_, b) in enumerate(pushes, start=1)]
    )
    print(
        "The stretches between events: from (s) | to (s) | begun by | context | the hand's distance to the named"
        " target at the start (mm) | the nearest (mm) | within the goal, from the start | the distance at the end"
        f" (mm) | the hand's speed at the end, over the last {SPEED_WINDOW:g} s (mm/s)"
    )
    for a, what in begins:
        b = end_of(a)
        inside = np.flatnonzero(distance[a:b] <= TOLERANCE)
        if distance[a] <= TOLERANCE:
            arrival = "already within it"
        elif len(inside):
            arrival = f"arrives after {times[a + inside[0]] - times[a]:.2f} s"
        else:
            arrival = "does not arrive"
        last = max(int(np.searchsorted(times, times[b - 1] - SPEED_WINDOW - 1e-9)), a)
        speed = np.linalg.norm(points[b - 1] - points[last]) / (times[b - 1] - times[last])
        print(
            f"  {times[a]:.2f} | {times[min(b, len(times) - 1)]:.2f} | {what} | {labels[named[a]]} |"
            f" {1e3 * distance[a]:.0f} | {1e3 * distance[a:b].min():.0f} | {arrival} | {1e3 * distance[b - 1]:.0f} |"
            f" {1e3 * speed:.0f}"
        )
    print("The reference's fastest joint speed (deg/s) | the largest torque (N m):")
    speed, torque = peaks(state, task)
    print(f"  over the run | {speed:.1f} | {torque:.2f}")
    windows = [(times >= times[k]) & (times < times[k] + AFTER_CHANGE) for k in changes]
    for k, window in zip(changes, windows, strict=True):
        speed, torque = peaks(state, window)
        print(
            f"  the {AFTER_CHANGE:g} s from the change at {times[k]:.2f} s ({labels[named[k - 1]]} →"
            f" {labels[named[k]]}) | {speed:.1f} | {torque:.2f}"
        )
    # peaks takes the differences of neighboring samples, so each part outside the windows is measured on its own.
    steps = np.arange(len(times))
    outside = [peaks(state, (steps >= a) & (steps < b)) for a, b in stretches(task & ~np.any(windows, axis=0))]
    print(f"  outside those | {max(s for s, _ in outside):.1f} | {max(t for _, t in outside):.2f}")
    pushed = np.any([(times >= times[a]) & (times < times[b] + AFTER_RELEASE) for a, b in pushes], axis=0)
    free = np.flatnonzero(task & ~pushed)
    k = free[np.argmax(gaps[free])]
    print(
        f"Outside the pushes and the {AFTER_RELEASE:g} s after each release, a joint's angle is at most"
        f" {gaps[k]:.2f} deg from its reference (at {times[k]:.2f} s)"
    )


# ---------------------------------------------------------------------------- tables


def print_tables() -> None:
    print_takes()
    print_search()
    print_claims_1_2()
    print_claim_3()
    print_switches()
    for name in ESNS:
        print_demonstration(name)
    print_settings()
    print_checks()


def setting_text(row: dict[str, Any]) -> str:
    return "lr {:g}, sr {:g}, is {:g}, ridge {:g}".format(*setting_of(row))


def per_target(row: dict[str, Any]) -> str:
    return ", ".join(f"{label} {int(row[f'holding_runs_{label}'])}" for label in LABELS)


def print_takes() -> None:
    print("\n== The takes: their start, and their arrival filtered, as trained on (as recorded in parentheses)")
    arrivals: list[float] = []
    early: list[str] = []
    lowest = (np.inf, "")  # the lowest hand at a take's start (y, m) and its take
    for label in LABELS:
        for row, filtered in zip(take_rows(label, False), take_rows(label, True), strict=True):
            start = np.array([row["start_q1_deg"], row["start_q2_deg"]])
            at = hand(np.radians(start)[np.newaxis])[0]
            name = Path(row["take"]).name.split(".")[0]
            print(
                f"{label} {row['demo']} {name}: start ({start[0]:.0f}, {start[1]:.0f}) deg,"
                f" hand ({at[0]:.2f}, {at[1]:.2f}) m, {np.linalg.norm(at - TARGETS[label]):.2f} m to {label},"
                f" arrives {filtered['arrival_time_s']:.1f} s ({row['arrival_time_s']:.1f} s),"
                f" lasts {row['length_s']:.1f} s"
            )
            arrivals.append(filtered["arrival_time_s"])
            if filtered["arrival_time_s"] <= 4.0:
                early.append(f"{label} {row['demo']}")
            lowest = min(lowest, (float(at[1]), f"{label} {row['demo']}"))
    print(
        f"filtered, they arrive {min(arrivals):.1f} to {max(arrivals):.1f} s, median {median(arrivals):.1f} s;"
        f" arrived by 4 s: {', '.join(early)}"
    )
    print(f"the lowest hand at a take's start: y = {lowest[0]:.2f} m ({lowest[1]})")
    for label in LABELS:
        rests = rest_postures(label)
        center = np.median(rests, axis=0)
        spread = np.abs(rests - center).max()
        print(
            f"{label}: the filtered takes rest at ({center[0]:.1f}, {center[1]:.1f}) deg, each within {spread:.1f} deg"
        )


def print_failures(runs: list[dict[str, Any]], heading: str) -> None:
    """A set of sweep runs' failing ones by kind, and where those that leave end."""
    failed = [run for run in runs if run["failed"]]
    kinds = {kind: [run for run in failed if failure(run) == kind] for kind in sorted({failure(run) for run in failed})}
    by_target = ", ".join(f"{label} {sum(run['task'] == label for run in failed)}" for label in LABELS)
    print(f"  {heading}: {len(failed)} of {len(runs)} fail ({by_target}):")
    for kind, those in kinds.items():
        ends = 1e3 * np.array(column(those, "final_distance_m"))
        print(
            f"    {kind}: {len(those)}; they end {ends.min():.0f} to {ends.max():.0f} mm from the target"
            f" (median {np.median(ends):.0f}), within 20 mm {int((ends <= 1e3 * TOLERANCE).sum())}"
        )


def print_search() -> None:
    print("\n== The search")
    coarse = scores("coarse_search")
    full = [row for row in coarse if row["holding_runs"] == row["runs"]]
    print(f"coarse_search: {len(full)} of {len(coarse)} settings hold in all {int(coarse[0]['runs'])}; the best five:")
    for row in coarse[:5]:
        print(
            f"  {setting_text(row)}: {int(row['holding_runs'])} ({per_target(row)}),"
            f" detour {row['median_detour_ratio']:.4f}"
        )
    for label in LABELS:
        print(
            f"  settings that hold in all nine at {label}: {sum(row[f'holding_runs_{label}'] == 9 for row in coarse)}"
        )
    for name in NAMES:
        counts = {v: sum(row[name] == v for row in full) for v in sorted({row[name] for row in coarse})}
        print(f"  settings that hold in all 27 by {name}: {counts}")

    fine = scores("fine_search")
    total = int(fine[0]["runs"])
    print(f"fine_search: {len(fine)} settings, five reservoirs each, {total} runs a setting")
    print(f"  holding in all {total}: {[setting_text(row) for row in fine if row['holding_runs'] == total]}")
    print(f"  holding in {total - 5} or more: {sum(row['holding_runs'] >= total - 5 for row in fine)}")
    for label in LABELS:
        print(f"  holding in all 45 at {label}: {sum(row[f'holding_runs_{label}'] == 45 for row in fine)}")
    print("  the ten best:")
    for rank, row in enumerate(fine[:10], start=1):
        print(
            f"    {rank}. {setting_text(row)}: {int(row['holding_runs'])} ({per_target(row)}),"
            f" detour {row['median_detour_ratio']:.4f}, typical seed {int(row['typical_seed'])}"
        )
    center = setting_of(scores("coarse_search")[0])
    rank, row = next((rank, row) for rank, row in enumerate(fine, start=1) if setting_of(row) == center)
    print(f"  the center ({setting_text(row)}): rank {rank}, {int(row['holding_runs'])} ({per_target(row)})")
    for name in NAMES:
        means = {v: np.mean([r["holding_runs"] for r in fine if r[name] == v]) for v in sorted({r[name] for r in fine})}
        print(f"  mean holding runs of {total} by {name}: " + ", ".join(f"{v:g}: {m:.1f}" for v, m in means.items()))
    runs = sweep_runs("fine_search")
    best = {setting_of(row) for row in fine[:20]}
    print_failures(runs, "all the fine search's runs")
    print_failures([run for run in runs if setting_of(run) in best], "the 20 best settings' runs")

    rescored = scores("untaught_search")
    print(f"untaught_search: the 20 settings scored again, {int(rescored[0]['runs'])} runs a setting")
    fine_rank = {setting_of(row): rank for rank, row in enumerate(fine, start=1)}
    fine_row = {setting_of(row): row for row in fine}
    for rank, row in enumerate(rescored, start=1):
        before = fine_row[setting_of(row)]
        print(
            f"  {rank:2d}. {setting_text(row)}: {int(row['holding_runs'])} of 675 ({per_target(row)} of 225),"
            f" detour {row['median_detour_ratio']:.3f}, typical seed {int(row['typical_seed'])};"
            f" fine search {int(before['holding_runs'])} of 135 ({per_target(before)} of 45),"
            f" rank {fine_rank[setting_of(row)]}"
        )
    runs = sweep_runs("untaught_search")
    for name, row in zip(ESNS, rescored[:2], strict=True):
        print(f"  {name}'s setting ({setting_text(row)}), each seed's runs that hold:")
        mine = [run for run in runs if setting_of(run) == setting_of(row)]
        for seed in sorted({run["seed"] for run in mine}):
            each = [run for run in mine if run["seed"] == seed]
            taught = [run for run in each if run["origin"].startswith("demo")]
            untaught = [run for run in each if run["origin"].startswith("untaught")]
            cells = ", ".join(
                f"{label} {sum(not run['failed'] for run in untaught if run['task'] == label)}" for label in LABELS
            )
            print(
                f"    seed {int(seed)}: from the takes' starts {sum(not run['failed'] for run in taught)} of"
                f" {len(taught)}; from the untaught starts {cells} of 36"
            )
        print_failures(mine, f"{name}'s setting's runs")
    print_failures([run for run in runs if run["origin"].startswith("demo")], "all 20 settings, from the takes' starts")
    print_failures([run for run in runs if run["origin"].startswith("untaught")], "all 20, from the untaught starts")


def print_claims_1_2() -> None:
    print("\n== Claims 1 and 2: runs that arrive and hold at the target the context names")
    arms = [("ESN #1", robot("1"), "esn"), ("ESN #2", robot("2"), "esn"), ("the replay", robot("1"), "replay")]
    for name, stem, arm in arms:
        cells = []
        for which, of in (("demo", N_TAKES), ("unseen", 36)):
            counts = ", ".join(
                f"{label} {sum(holds(run) for run in group(stem, arm, label, which))}" for label in LABELS
            )
            cells.append(f"from the {'takes' if which == 'demo' else 'unseen'} starts {counts} of {of}")
        print(f"{name}: {'; '.join(cells)}")
    print("The runs that do not hold, from the log: where the hand ends, and its distance then to each target")
    for name, stem, arm in arms:
        for label in LABELS:
            for run in rows(stem, arm, label):
                if holds(run):
                    continue
                posture = start_posture(label, run)
                end = path(stem, arm, label, int(run["start"]))[1][-1]
                to = ", ".join(f"{other} {1e3 * np.linalg.norm(end - TARGETS[other]):.0f}" for other in LABELS)
                print(
                    f"  {name}, context {label}, {run['origin']} ({posture[0]:g}, {posture[1]:g}) deg:"
                    f" arrived {run['arrived']}, {failure(run)}; ends at ({end[0]:.2f}, {end[1]:.2f}) m,"
                    f" to {to} mm"
                )


def print_claim_3() -> None:
    print("\n== Claim 3 and the takes' starts: medians over the starts (the largest in parentheses);")
    print("   the detour ratio and the arrival over the runs that arrive")
    arms = [("ESN #1", robot("1"), "esn"), ("ESN #2", robot("2"), "esn"), ("replay", robot("1"), "replay")]
    for label in LABELS:
        for which in ("demo", "unseen"):
            for name, stem, arm in arms:
                runs = group(stem, arm, label, which)
                speed, torque = column(runs, "peak_reference_speed_dps"), column(runs, "peak_torque_nm")
                print(
                    f"{label} {'takes' if which == 'demo' else 'unseen':6s} {name:6s}:"
                    f" detour {median(column(runs, 'detour_ratio')):.3f}"
                    f" | speed {median(speed):7.1f} ({max(speed):7.1f}) dps"
                    f" | torque {median(torque):6.2f} ({max(torque):6.2f}) N m"
                    f" | arrival {median(column(runs, 'arrival_time_s')):5.2f} s"
                )
    print("From how many unseen starts the ESN's measure is smaller than the replay's from the same start (of 36;")
    print("the detour ratio over the starts from which both arrive)")
    for label in LABELS:
        for name, n in ESNS.items():
            pairs = list(
                zip(group(robot(n), "esn", label, "unseen"), group(robot(n), "replay", label, "unseen"), strict=True)
            )
            arriving = [(run, replay) for run, replay in pairs if run["arrived"] and replay["arrived"]]
            cells = []
            for key, those in (
                ("detour_ratio", arriving),
                ("peak_reference_speed_dps", pairs),
                ("peak_torque_nm", pairs),
            ):
                cells.append(f"{key} {sum(run[key] < replay[key] for run, replay in those)} of {len(those)}")
            print(f"{label} {name}: {', '.join(cells)}")
    print(
        f"A reference that jumps: runs whose reference's fastest joint speed is above {JUMP_SPEED:g} deg/s, whose"
        f" largest torque is above {JUMP_TORQUE:g} N m, and both"
    )
    for label in LABELS:
        for which, of in (("demo", N_TAKES), ("unseen", 36)):
            cells = []
            for name, stem, arm in arms:
                runs = group(stem, arm, label, which)
                fast = [run["peak_reference_speed_dps"] > JUMP_SPEED for run in runs]
                strong = [run["peak_torque_nm"] > JUMP_TORQUE for run in runs]
                both = sum(f and s for f, s in zip(fast, strong, strict=True))
                cells.append(f"{name} {sum(fast)}, {sum(strong)}, {both}")
            print(f"{label} {'takes' if which == 'demo' else 'unseen':6s} (of {of}): {'; '.join(cells)}")
    print(
        f"Medians from the unseen starts within {NEAR:g} deg of the start of the take the replay plays (over both"
        " joints), and from those farther"
    )
    for label in LABELS:
        distances = played_distances(label)
        for name, stem, arm in arms:
            runs = group(stem, arm, label, "unseen")
            cells = []
            for heading, side in (("within", distances <= NEAR), ("farther", distances > NEAR)):
                chosen = [run for run, here in zip(runs, side, strict=True) if here]
                cells.append(
                    f"{heading} ({len(chosen)}): speed {median(column(chosen, 'peak_reference_speed_dps')):7.1f} dps,"
                    f" torque {median(column(chosen, 'peak_torque_nm')):6.2f} N m"
                )
            print(f"{label} {name:6s}: {' | '.join(cells)}")
    print(
        f"Unseen starts whose reference jumps by either measure (speed above {JUMP_SPEED:g} deg/s or torque above"
        f" {JUMP_TORQUE:g} N m), and how many of them are farther than {NEAR:g} deg from the played take's start"
    )
    for label in LABELS:
        far = played_distances(label) > NEAR
        cells = []
        for name, n in ESNS.items():
            runs = group(robot(n), "esn", label, "unseen")
            jumps = np.array(
                [run["peak_reference_speed_dps"] > JUMP_SPEED or run["peak_torque_nm"] > JUMP_TORQUE for run in runs]
            )
            cells.append(f"{name} {int((jumps & far).sum())} of {int(jumps.sum())}")
        print(f"{label}: {', '.join(cells)}")
    print(
        f"Unseen starts whose hand is below y = {LOW_HAND:g} m, among those far from the played take's start and near"
    )
    for label in LABELS:
        far = played_distances(label) > NEAR
        starts = np.array([start_posture(label, replay) for replay in group(robot("1"), "replay", label, "unseen")])
        low = hand(np.radians(starts))[:, 1] < LOW_HAND
        print(
            f"{label}: far {int((low & far).sum())} of {int(far.sum())},"
            f" near {int((low & ~far).sum())} of {int((~far).sum())}"
        )
    print("The unseen starts shown over time (the replay's largest torque is the lower middle of the 36):")
    for label, origin in UNSEEN_SHOWN.items():
        replay = next(row for row in rows(robot("1"), "replay", label) if row["origin"] == origin)
        posture = start_posture(label, replay)
        print(
            f"  {label}: {origin} ({posture[0]:g}, {posture[1]:g}) deg, the replay plays {replay['replayed']},"
            f" its largest torque {replay['peak_torque_nm']:.2f} N m"
        )
    print("The runs shown over time: the reference's largest distance from the arm in the first second, and the")
    print("largest torque, over the first 50 ms and after them (the torque axes are cut at the ESN's after them)")
    for name, n in ESNS.items():
        for label in LABELS:
            for origin in (TAKE_SHOWN, UNSEEN_SHOWN[label]):
                cells = []
                for arm in ("esn", "replay"):
                    state = log(robot(n), arm, label, start_of(robot(n), label, origin))
                    first = (state.times >= 0.0) & (state.times < 1.0)
                    gap = np.degrees(np.abs(state.channel("q_ref") - state.channel("q"))[first]).max()
                    tau = np.abs(state.channel("tau"))
                    early = tau[(state.times >= 0.0) & (state.times < 0.05)].max()
                    cells.append(f"{arm} {gap:.1f} deg, {early:.2f} then {tau[state.times >= 0.05].max():.2f} N m")
                print(f"  {name}, {label}, {origin}: {'; '.join(cells)}")


def print_switches() -> None:
    print("\n== Claims 4 and 5: the context changed, from each take's start (of nine a change)")
    for name, n in ESNS.items():
        for timing in TIMINGS:
            total, holding = 0, 0
            lines = []
            for first, new in CHANGES:
                stem = switch(n, timing, new)
                time = change_time(stem)
                runs = rows(stem, "esn", first)
                details = [at_change(stem, first, int(run["start"])) for run in runs]
                arrival = column(runs, "switch_arrival_time_s")
                arrived = sum(bool(run["arrived"]) and run["arrival_time_s"] <= time for run in runs)

                def spread(key: str, details: list[dict[str, float]] = details) -> str:
                    values = [d[key] for d in details]
                    return f"{median(values):.1f} ({max(values):.1f})"

                lines += [
                    f"  {first} → {new}: hold {sum(bool(run['switch_success']) for run in runs)}/{len(runs)};"
                    f" arrival from the change {median(arrival):.2f} ({np.nanmin(arrival):.2f} to"
                    f" {np.nanmax(arrival):.2f}) s"
                    + (
                        f", the three latest {', '.join(f'{a:.2f}' for a in sorted(arrival)[-3:])} s"
                        if timing == "on the way"
                        else ""
                    )
                    + f"; final distance at most {1e3 * max(column(runs, 'switch_final_distance_m')):.1f} mm",
                    f"      at the change: {arrived}/{len(runs)} have arrived at {first}; median distance to {first}"
                    f" {1e3 * median([d['to_first'] for d in details]):.0f} mm, to {new}"
                    f" {1e3 * median([d['to_new'] for d in details]):.0f} mm",
                    f"      after the change: speed {spread('speed_after')} dps, torque {spread('torque_after')} N m;"
                    f" before it: speed {spread('speed_before')} dps, torque {spread('torque_before')} N m",
                ]
                total += len(runs)
                holding += sum(bool(run["switch_success"]) for run in runs)
            print(f"{name}, changed {timing} (at {change_time(switch(n, timing, 'A')):g} s): {holding} of {total} hold")
            print("\n".join(lines))


def esn_table(config: Path) -> dict[str, Any]:
    with config.open("rb") as f:
        return tomllib.load(f)["esn"]


def print_settings() -> None:
    print("\n== The settings against stage 1's (report 023's)")
    esns = {
        f"{stage} {name}": next(directory.glob(f"esn{n}_*.toml"))
        for stage, directory in (("this stage's", CONFIGS), ("report 023's", STAGE_1))
        for name, n in ESNS.items()
    }
    print("  leak rate, spectral radius, input scaling, ridge, seed:")
    for name, config in esns.items():
        table = esn_table(config)
        print(f"  {name}: {', '.join(f'{table[key]:g}' for key in (*NAMES, 'seed'))} ({config.name})")
    for name, n in ESNS.items():
        table = esn_table(next(CONFIGS.glob(f"esn{n}_*.toml")))
        fixed = {key: item for key, item in table.items() if key not in (*NAMES, "seed")}
        print(f"  {name}, fixed: {fixed}")
        with next(CONFIGS.glob(f"esn{n}_*.toml")).open("rb") as f:
            for demonstrations in tomllib.load(f)["demonstrations"]:
                train = demonstrations["train"]
                print(
                    f"    trained with the context {demonstrations['context']} on {demonstrations['run']}:"
                    f" {train[0]} to {train[-1]} ({len(train)})"
                )


def print_checks() -> None:
    print("\n== Checks")
    copies = [p for p in RESULTS.iterdir() if p.is_dir() and p.name != "summary"] + list(DEMO_RUNS.values())
    for run in sorted(copies, key=lambda p: p.name):
        with (run / "run.toml").open("rb") as f:
            record = tomllib.load(f)
        dirty = " WITH UNCOMMITTED CHANGES" if record["uncommitted_changes"] else ""
        print(f"{run.name}: {record['commit'][:7]}{dirty}")
    differ = []
    for run in copies:
        for copy in run.iterdir():
            if not filecmp.cmp(copy, resolve_run_path(f"results/{run.name}") / copy.name, shallow=False):
                differ.append(str(copy.relative_to(REPORT)))
    for copy in TAKES.iterdir():
        if not filecmp.cmp(copy, storage_root() / "data" / TAKES.name / copy.name, shallow=False):
            differ.append(str(copy.relative_to(REPORT)))
    print(f"the copies equal the storage root's files: {not differ}" + (f", but {differ}" if differ else ""))
    hashes = {row["take"]: row["take_sha256"] for label in LABELS for row in take_rows(label, False)}
    same = all(
        hashlib.sha256((REPORT / take_path).read_bytes()).hexdigest() == digest for take_path, digest in hashes.items()
    )
    print(f"the 27 takes are those the demonstration runs imported (SHA-256): {same and len(hashes) == 27}")
    radii = {target["tolerance"] for target in _ARM["task"]["targets"]}
    print(f"every target's goal radius is {1e3 * TOLERANCE:g} mm: {radii == {TOLERANCE}}")
    replays = [
        np.allclose(column(rows(robot("1"), "replay", label), key), column(rows(robot("2"), "replay", label), key))
        for label in LABELS
        for key in ("detour_ratio", "peak_torque_nm", "final_distance_m")
    ]
    print(f"the replay beside ESN #1 equals the one beside ESN #2: {all(replays)}")
    print_switch_checks()
    picks = {}
    for label in LABELS:
        replays_ = group(robot("1"), "replay", label, "unseen")
        order = np.argsort(column(replays_, "peak_torque_nm"))
        picks[label] = replays_[order[len(order) // 2 - 1]]["origin"]
    print(f"the unseen starts shown are those of the rule: {picks == UNSEEN_SHOWN}")
    print_demonstration_checks()


def print_demonstration_checks() -> None:
    """The demonstrations' logs against the storage root's, and their ESNs and trackers against the robot runs'."""
    for name, n in ESNS.items():
        copy = DEMONSTRATIONS[name]
        same = filecmp.cmp(copy, storage_root() / "data" / RECORDED.name / copy.name, shallow=False)
        app, tracking = demonstration(name).extra["robot_app"], demonstration(name).extra["tracking"]
        config = config_of(robot(n))
        esn, tracker = Path(config["esn"]["model"]).parent.name, config["tracker"]
        agree = (
            [tracking["law"]] == tracker["laws"]
            and [tracking["omega"]] == tracker["omegas"]
            and [tracking["damping"]] == tracker["dampings"]
            and tracking["control_period"] == tracker["control_period"]
            and tracking["reference_velocity"] == tracker["reference_velocity"]
        )
        dirty = " WITH UNCOMMITTED CHANGES" if app["uncommitted_changes"] else ""
        print(f"the demonstration of {name}, recorded at {app['commit'][:7]}{dirty}:")
        print(f"  {copy.relative_to(REPORT)} equals the storage root's file: {same}")
        print(f"  it runs the ESN of {robot(n)}, {esn}: {app['name'] == esn}")
        print(
            f"  its tracker is that of {robot(n)}.toml ({tracking['law']}, omega {tracking['omega']:g}, damping"
            f" {tracking['damping']:g}, every {tracking['control_period']:g} s, with the reference's velocity"
            f" {tracking['reference_velocity']}): {agree}"
        )
        print_first_reach(name, n)


def print_first_reach(name: str, n: str) -> None:
    """A demonstration's first reach against the undisturbed run of its ESN from take demo_00's start with the
    context A, which it should equal until its first change of context: the arrival times, and the largest
    difference in joint angle between the two logs up to that change."""
    state = demonstration(name)
    stem = robot(n)
    run = log(stem, "esn", "A", start_of(stem, "A", TAKE_SHOWN))
    run_arrival = float(next(row for row in rows(stem, "esn", "A") if row["origin"] == TAKE_SHOWN)["arrival_time_s"])
    task = state.times > -1e-9
    first = int(np.flatnonzero(task)[0])
    labels, named, distance = named_targets(state, hand(state.channel("q")))
    end = min(changes_and_pushes(state)[0][0], len(run.times))  # the first change, within the undisturbed run
    inside = np.flatnonzero(distance[first:end] <= TOLERANCE)
    arrival = float(state.times[first + inside[0]]) if len(inside) else np.nan
    gaps = np.degrees(np.abs(state.channel("q")[:end] - run.channel("q")[:end])).max(axis=1)
    same = bool(gaps[0] <= 0.05 and labels[named[first]] == "A" and np.array_equal(state.times[:end], run.times[:end]))
    print(
        f"  its first reach, to {state.times[end]:.2f} s, against {stem}'s run from take demo_00's start with the"
        f" context A (the same start posture, context, and clock: {same}): arrives at {arrival:.2f} s and"
        f" {run_arrival:.2f} s, the joint angles differ by at most {gaps.max():.4f} deg; within 0.01 s and 0.05 deg:"
        f" {same and abs(arrival - run_arrival) <= 0.01 and gaps.max() <= 0.05}"
    )


def print_switch_checks() -> None:
    """The runs of claims 4 and 5 against the undisturbed ones, and their contexts and peaks against their metrics."""
    before, unchanged, replays, contexts, peaks_agree = [], [], [], [], []
    for n in ESNS.values():
        for timing in TIMINGS:
            for new in LABELS:
                stem = switch(n, timing, new)
                time = change_time(stem)
                for first in LABELS:
                    for run in rows(stem, "esn", first):
                        start = int(run["start"])
                        state, undisturbed = log(stem, "esn", first, start), log(robot(n), "esn", first, start)
                        q, q_undisturbed = state.channel("q"), undisturbed.channel("q")
                        common = min(len(q), len(q_undisturbed))
                        named = np.argmax(state.channel("context"), axis=1)
                        if first == new:
                            unchanged.append(np.array_equal(q[:common], q_undisturbed[:common]))
                            contexts.append(bool(np.all(named == LABELS.index(first))))
                            continue
                        part = state.times < time - 1e-9
                        before.append(np.array_equal(q[part], q_undisturbed[part[: len(q_undisturbed)]]))
                        first_part, new_part = named[part] == LABELS.index(first), named[~part] == LABELS.index(new)
                        contexts.append(bool(np.all(first_part) and np.all(new_part)))
                        details = at_change(stem, first, start)
                        peaks_agree.append(
                            np.isclose(
                                max(details["speed_before"], details["speed_after"]), run["peak_reference_speed_dps"]
                            )
                            and np.isclose(
                                max(details["torque_before"], details["torque_after"]), run["peak_torque_nm"]
                            )
                        )
                        replay, replay_undisturbed = (
                            log(stem, "replay", first, start),
                            log(robot(n), "replay", first, start),
                        )
                        common = len(replay_undisturbed.times)
                        replays.append(np.array_equal(replay.channel("q")[:common], replay_undisturbed.channel("q")))
    print(
        "before the change, every ESN run that changes equals the undisturbed run from the same start, joint angle"
        f" for joint angle: {all(before)} ({len(before)} runs)"
    )
    print(f"the runs whose context is already the new one equal the undisturbed run over its 24 s: {all(unchanged)}")
    print(
        "the context in the logs is the first target's before the change's time and the new one's from it, and"
        f" never changes in the runs that do not change: {all(contexts)}"
    )
    print(
        "the reference's fastest speed and the largest torque, before and after the change from the logs, give the"
        f" runs' metrics over the task: {all(peaks_agree)}"
    )
    print(f"the replays of these runs equal those beside the undisturbed runs over their 24 s: {all(replays)}")


if __name__ == "__main__":
    main()
