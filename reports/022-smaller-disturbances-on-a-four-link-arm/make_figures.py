# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 022.

    uv run python reports/022-smaller-disturbances-on-a-four-link-arm/make_figures.py

It reads the copies of step 6's runs under ``results/`` and of step 5's under report
021's ``results/`` (their configurations, run records, and metrics), and the runs'
logs under the storage root, which are not kept in Git. It prints the tables of the
report and draws into ``results/summary``:

- ``pushes_paths.png``: the hand's path under the pushes at each force (Section 3.1);
- ``pushes_displacement.png``: the hand's largest displacement against each push's force
  (Section 3.1);
- ``pushes_following.png``: the following in each joint against the force (Section 3.2);
- ``pushes_joints_esn1.png``, ``pushes_torques_esn1.png``, ``pushes_joints_esn2.png``,
  and ``pushes_torques_esn2.png``: each joint's angle less the taught motion, or its
  torque, over time, at each force and undisturbed (Section 3.3);
- ``offsets_paths.png``: the hand's paths from every start of each size (Section 3.4);
- ``offsets_minus_joints_esn1.png``, ``offsets_minus_torques_esn1.png``, and the same for
  ESN #2: from -s in joints 3 and 4, at each size s; ``offsets_plus_*``: from +s in every
  joint (Section 3.5);
- ``lower_joints.png``, ``lower_torques.png``: both ESNs from (0, -s, -s, -s), where joint
  2 starts lower too, at 5 and 2.5 deg (Section 3.6).

It loads report 021's ``make_figures.py`` as a module, to read the runs and for its
measures.
"""

from __future__ import annotations

import functools
import importlib.util
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from numpy.typing import NDArray
from skelarm import Skeleton, StateLog

from arm_esn_ctrl.disturbances import disturbance_spans
from arm_esn_ctrl.tracking import mass_matrix_at

REPORT = Path(__file__).resolve().parent
REPO_ROOT = REPORT.parents[1]
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"
EARLIER = "021-search-on-a-four-link-arm"
DPI = 90


def load_script(report: str) -> ModuleType:
    """An earlier report's make_figures.py, as a module."""
    path = REPO_ROOT / "reports" / report / "make_figures.py"
    spec = importlib.util.spec_from_file_location(f"report_{report[:3]}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


R021: Any = load_script(EARLIER)
STEP5 = R021.RESULTS  # step 5's runs: undisturbed, the block, the full pushes, and the 15 deg offset starts
STEP5_SCENARIOS = ("nominal", "block", "pushes", "offsets")
ESNS: dict[str, str] = {name: tag for name, (tag, _) in R021.CHOSEN.items()}  # ESN #1, ESN #2: their runs' tags
TAKE_START = R021.TAKE_START
DT = R021.DT  # the ESNs' period (s)
DURATION = 23.9  # the task, after the warm-up (s)
TIMES = DT * np.arange(round(DURATION / DT) + 1)  # the task, every 10 ms
FOLLOWING_WINDOW = 0.5  # the following gain's window: from a disturbance's onset to this long after its end (s)
# The pushes: the runs' scenario, the size's name, and its share of step 5's force.
PUSHES = [("pushes", "full", 1.0), ("pushes_half", "half", 0.5), ("pushes_quarter", "a quarter", 0.25)]
# The offset starts: the runs' scenario and the offset s (deg).
OFFSETS = [("offsets", 15.0), ("offsets_5deg", 5.0), ("offsets_2.5deg", 2.5)]
SIZES = {15.0: "±15", 5.0: "±5", 2.5: "±2.5"}
LOWER, UPPER = R021.LOWER, R021.UPPER  # the joint limits (rad)

with R021.ARM_CONFIG.open("rb") as f:
    _TASK = tomllib.load(f)["task"]["target"]
TARGET = np.array(_TASK["pos"])  # m
TOLERANCE = float(_TASK["tolerance"])  # the goal's radius (m)

TEXT_COLOR, TITLE_COLOR, GRID_COLOR = R021.TEXT_COLOR, R021.TITLE_COLOR, R021.GRID_COLOR
SURFACE_COLOR, DISTURBANCE_COLOR, TAUGHT_COLOR = R021.SURFACE_COLOR, R021.DISTURBANCE_COLOR, R021.TAUGHT_COLOR
ESN_COLOR, REPLAY_COLOR = R021.ARMS[0][2], R021.ARMS[1][2]
ESN_COLORS = R021.ESN_COLORS  # ESN #1 dark blue, ESN #2 light blue, where both are drawn
GOAL_COLOR = "#7b3fa0"
JOINT_MARKERS = ["o", "s", "^", "D"]
ARM_ZORDERS = {"esn": 3.0, "replay": 2.0}
REFERENCE_ZORDER = 2.5


def main() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Callable[[], Figure]] = {
        "pushes_paths.png": plot_push_paths,
        "pushes_displacement.png": plot_displacement,
        "pushes_following.png": plot_following,
        "offsets_paths.png": plot_offset_paths,
    }
    for name in ESNS:
        number = name[-1]
        push_columns = [
            (scenario, name, TAKE_START, f"{label} force ({force_text(scenario)})") for scenario, label, _ in PUSHES
        ]
        push_columns.append(("nominal", name, TAKE_START, "undisturbed"))
        for channel in ("joints", "torques"):
            figures[f"pushes_{channel}_esn{number}.png"] = functools.partial(
                plot_series, f"{name}, under the pushes", push_columns, channel
            )
        for tag, pattern, heading in (
            ("minus", "+0,+0,-{s},-{s}", "from -s in joints 3 and 4"),
            ("plus", "+{s},+{s},+{s},+{s}", "from +s in every joint"),
        ):
            columns = [
                (scenario, name, origin_of(pattern, s), f"s = {s:g}°: {origin_of(pattern, s).split()[1]}°")
                for scenario, s in OFFSETS
            ]
            for channel in ("joints", "torques"):
                figures[f"offsets_{tag}_{channel}_esn{number}.png"] = functools.partial(
                    plot_series, f"{name}, {heading}", columns, channel
                )
    lower = [
        (scenario, name, origin_of("+0,-{s},-{s},-{s}", s), f"{name}, {origin_of('+0,-{s},-{s},-{s}', s).split()[1]}°")
        for name in ESNS
        for scenario, s in OFFSETS[1:]
    ]
    for channel in ("joints", "torques"):
        figures[f"lower_{channel}.png"] = functools.partial(
            plot_series, "Both ESNs from (0, -s, -s, -s), where joint 2 starts lower too", lower, channel
        )
    for name, draw in figures.items():
        draw().savefig(SUMMARY / name, dpi=DPI)
    print_tables()
    print(f"\nWrote {', '.join(figures)} to {SUMMARY}")


# ---------------------------------------------------------------------------- reading the runs


def results_of(scenario: str) -> Path:
    """Where a scenario's copies are: step 5's in report 021, step 6's here."""
    return STEP5 if scenario in STEP5_SCENARIOS else RESULTS


def rows(scenario: str, esn: str) -> list[dict[str, Any]]:
    return R021.robot_rows(scenario, ESNS[esn], results_of(scenario))


def run(scenario: str, esn: str, arm: str, origin: str = TAKE_START) -> dict[str, Any]:
    return R021.robot_run(scenario, ESNS[esn], arm, origin, results_of(scenario))


@functools.lru_cache(maxsize=64)
def log(scenario: str, esn: str, arm: str, origin: str = TAKE_START) -> StateLog:
    """One arm's log under the storage root; the replays are the same beside either ESN."""
    return R021.robot_log(scenario, ESNS[esn], arm, origin, results_of(scenario))


def config(scenario: str) -> dict[str, Any]:
    return R021.robot_config(scenario, ESNS["ESN #1"], results_of(scenario))


def spans(scenario: str) -> list[tuple[float, float]]:
    return disturbance_spans(config(scenario).get("disturbance"))


def forces(scenario: str) -> list[float]:
    return [float(table["force"]) for table in config(scenario).get("disturbance", [])]


def force_text(scenario: str) -> str:
    return ", ".join(f"{force:g}" for force in forces(scenario)) + " N"


def origin_of(pattern: str, s: float) -> str:
    """A start's name in the runs: the take's start, offset by ``pattern`` with s in place of {s}."""
    return f"{TAKE_START} {pattern.format(s=f'{s:g}')} deg"


def offset_of(origin: str) -> tuple[float, ...]:
    """A start's offset in each joint (deg), from its name."""
    if not origin.endswith(" deg"):
        return (0.0, 0.0, 0.0, 0.0)
    return tuple(float(angle) for angle in origin.split()[1].split(","))


# ---------------------------------------------------------------------------- measures


@functools.cache
def taught_times() -> NDArray[np.float64]:
    return DT * np.arange(len(R021.taught()))


def taught_at(times: NDArray[np.float64]) -> NDArray[np.float64]:
    """The taught joint angles at ``times``: the take's start before it, its end after it."""
    q = R021.taught()
    return np.column_stack([np.interp(times, taught_times(), q[:, j]) for j in range(q.shape[1])])


@functools.cache
def taught_hand() -> NDArray[np.float64]:
    """The taught hand every 10 ms over the task."""
    return R021.hand(taught_at(TIMES))


def sampled(state: StateLog, channel: str) -> NDArray[np.float64]:
    """A channel every 10 ms over the task."""
    values = state.channel(channel)
    return np.column_stack([np.interp(TIMES, state.times, values[:, j]) for j in range(values.shape[1])])


def hand_of(state: StateLog, channel: str = "q") -> NDArray[np.float64]:
    """The hand every 10 ms over the task, of the arm or (``q_ref``) of its reference."""
    return R021.hand(sampled(state, channel))


def window(onset: float, end: float) -> NDArray[np.bool_]:
    """The following gain's window over the task's samples."""
    return (onset - 1e-9 < TIMES) & (end + FOLLOWING_WINDOW - 1e-9 > TIMES)


def displacement(state: StateLog, onset: float, end: float) -> float:
    """The hand's largest displacement from the taught hand at the same time, in a push's window, from its onset (m)."""
    moved = (hand_of(state) - taught_hand())[window(onset, end)]
    return float(np.linalg.norm(moved - moved[0], axis=1).max())


def gain(arm: NDArray[np.float64], reference: NDArray[np.float64]) -> float:
    arm, reference = arm - arm[0], reference - reference[0]
    return float((reference * arm).sum() / (arm * arm).sum())


def hand_following_less_undisturbed(state: StateLog, undisturbed: StateLog, onset: float, end: float) -> float:
    """The hand's following gain with each signal less its own in the same arm's undisturbed run."""
    inside = window(onset, end)
    arm = (hand_of(state) - hand_of(undisturbed))[inside]
    reference = (hand_of(state, "q_ref") - hand_of(undisturbed, "q_ref"))[inside]
    return gain(arm, reference)


def joint_following_less_undisturbed(
    state: StateLog, undisturbed: StateLog, onset: float, end: float
) -> NDArray[np.float64]:
    """The following in each joint with each signal less its own in the same arm's undisturbed run."""
    inside = window(onset, end)
    arm = (sampled(state, "q") - sampled(undisturbed, "q"))[inside]
    reference = (sampled(state, "q_ref") - sampled(undisturbed, "q_ref"))[inside]
    arm, reference = arm - arm[0], reference - reference[0]
    return (reference * arm).sum(axis=0) / (arm * arm).sum(axis=0)


def peak_after_start(state: StateLog) -> float:
    """The largest torque at any joint after the task's first control period, which holds the reference's handover."""
    return float(np.abs(state.channel("tau")[state.times >= DT - 1e-9]).max())


def path_length(state: StateLog) -> float:
    return float(np.linalg.norm(np.diff(hand_of(state), axis=0), axis=1).sum())


def from_target(state: StateLog) -> NDArray[np.float64]:
    return np.linalg.norm(hand_of(state) - TARGET, axis=1)


def out_of_goal(state: StateLog, after: float) -> list[tuple[float, float]]:
    """When the hand is outside the goal after ``after``: each span's first and last sample (s)."""
    outside = (from_target(state) > TOLERANCE) & (after - 1e-9 <= TIMES)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], outside.astype(int), [0]])))
    return [(float(TIMES[a]), float(TIMES[b - 1])) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def start_details(state: StateLog) -> dict[str, Any]:
    """From a start: how far the reference goes past a limit, each joint's time at a stop, and its first second."""
    q, reference, times = state.channel("q"), state.channel("q_ref"), state.times
    task = times >= 0
    past = max(np.max(LOWER - reference[task]), np.max(reference[task] - UPPER), 0.0)
    at_stop = ((q[task] <= LOWER + 1e-9) | (q[task] >= UPPER - 1e-9)).sum(axis=0) * np.diff(times).mean()
    first = (times >= 0) & (times < 1.0)
    return {
        "past_deg": float(np.degrees(past)),
        "at_stop_s": at_stop,
        "first_second_deg": float(np.degrees(np.linalg.norm((reference - q)[first], axis=1)).max()),
    }


@functools.cache
def offset_runs(scenario: str, esn: str, arm: str) -> list[dict[str, Any]]:
    """One arm's runs from every start of a scenario: their metrics, offsets, hand paths, and details from the logs."""
    out = []
    for row in rows(scenario, esn):
        if row["arm"] != arm:
            continue
        state = R021.robot_log(scenario, ESNS[esn], arm, row["origin"], results_of(scenario))
        out.append(
            {
                **row,
                **start_details(state),
                "offset": offset_of(row["origin"]),
                "moved": str(row["origin"]).endswith(" deg"),
                "hand": hand_of(state),
            }
        )
    return out


def group(scenario: str, esn: str, arm: str, which: str) -> list[dict[str, Any]]:
    """The moved starts: those whose joint 2 is not lowered ("shared", as at 15 deg) or is ("lower")."""
    moved = [r for r in offset_runs(scenario, esn, arm) if r["moved"]]
    return [r for r in moved if (r["offset"][1] >= 0) == (which == "shared")]


def holds(row: dict[str, Any]) -> bool:
    """Arrives and never leaves the goal: arrives, does not leave it in the 2 s hold, and ends in it."""
    return bool(row["arrived"] and row["success"] and not row["left_goal"])


# ---------------------------------------------------------------------------- figures


def style(ax: Axes) -> None:
    R021.style(ax)


def draw_goal(ax: Axes) -> None:
    ax.add_patch(Circle(tuple(TARGET), TOLERANCE, fill=False, edgecolor=GOAL_COLOR, linewidth=1.3, zorder=5))


def path_axes(ax: Axes) -> None:
    style(ax)
    ax.plot(*taught_hand().T, color=TAUGHT_COLOR, linewidth=4.0, zorder=1)
    draw_goal(ax)
    ax.set_aspect("equal")


def path_handles(*arms: tuple[str, str]) -> list[Line2D]:
    handles = [Line2D([], [], color=color, linewidth=1.6, label=label) for label, color in arms]
    return [
        *handles,
        Line2D([], [], color=TAUGHT_COLOR, linewidth=4.0, label="the taught path"),
        Line2D([], [], color=GOAL_COLOR, marker="o", markerfacecolor="none", linestyle="none", label="the goal"),
    ]


def plot_push_paths() -> Figure:
    """The hand's path under the pushes at each force: a row per ESN, against the replay's."""
    fig = Figure(figsize=(13, 7.2), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle("The hand's path under the pushes, at each force", color=TITLE_COLOR)
    axes = fig.subplots(2, len(PUSHES), sharex=True, sharey=True)
    for row, name in enumerate(ESNS):
        for col, (scenario, label, _) in enumerate(PUSHES):
            ax = axes[row, col]
            path_axes(ax)
            ax.plot(*hand_of(log(scenario, name, "replay")).T, color=REPLAY_COLOR, linewidth=1.3, zorder=2)
            ax.plot(*hand_of(log(scenario, name, "esn")).T, color=ESN_COLORS[name], linewidth=1.5, zorder=3)
            ax.set_title(f"{name}, {label} force ({force_text(scenario)})", color=TITLE_COLOR, fontsize=10)
        axes[row, 0].set_ylabel("y (m)", color=TEXT_COLOR)
    for ax in axes[-1]:
        ax.set_xlabel("x (m)", color=TEXT_COLOR)
    handles = path_handles(*((f"{name}'s arm", ESN_COLORS[name]) for name in ESNS), ("the replay's arm", REPLAY_COLOR))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


@functools.cache
def displacements(esn: str, arm: str) -> dict[str, list[float]]:
    """The hand's largest displacement in each push's window (m), at each force and undisturbed ("nominal")."""
    windows = spans("pushes")
    out = {}
    for scenario in [scenario for scenario, _, _ in PUSHES] + ["nominal"]:
        state = log(scenario, esn, arm)
        out[scenario] = [displacement(state, onset, end) for onset, end in windows]
    return out


def plot_displacement() -> Figure:
    """The hand's largest displacement against each push's force."""
    fig = Figure(figsize=(13, 4.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The hand's largest displacement from the taught hand, from each push's onset to 0.5 s after its end",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(1, 3, sharey=False)
    series = [
        ("ESN #1", "esn", ESN_COLORS["ESN #1"], "o", "-"),
        ("ESN #2", "esn", ESN_COLORS["ESN #2"], "s", "--"),
        ("the replay", "replay", REPLAY_COLOR, "^", "-"),
    ]
    scenarios = ["nominal"] + [scenario for scenario, _, _ in reversed(PUSHES)]
    for i, ax in enumerate(axes):
        style(ax)
        onset = spans("pushes")[i][0]
        x = [0.0] + [forces(scenario)[i] for scenario in scenarios[1:]]
        for label, arm, color, marker, line in series:
            esn = label if arm == "esn" else "ESN #1"
            y = [1000.0 * displacements(esn, arm)[scenario][i] for scenario in scenarios]
            ax.plot(x, y, color=color, marker=marker, markersize=8, linewidth=2, linestyle=line, label=label, zorder=3)
        ax.set_xlim(-0.05 * x[-1], 1.08 * x[-1])
        ax.set_ylim(bottom=0)
        ax.set_title(f"push {i + 1}, at {onset:g} s", color=TITLE_COLOR, fontsize=10, loc="left")
        ax.set_xlabel("force (N); 0: undisturbed", color=TEXT_COLOR)
    axes[0].set_ylabel("largest displacement (mm)", color=TEXT_COLOR)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    return fig


@functools.cache
def joint_following(esn: str, scenario: str, push: int) -> tuple[NDArray[np.float64], float]:
    """Report 021's following in joint space under one push: per joint, and pooled."""
    onset, end = spans(scenario)[push]
    return R021.joint_following(log(scenario, esn, "esn"), log(scenario, esn, "replay"), onset, end)


def plot_following() -> Figure:
    """The following in each joint against the force: a row per ESN, a column per push."""
    fig = Figure(figsize=(13, 7.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The following in joint space under each push, against its force; 1: the reference moves as far as the arm",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(2, 3, sharex=True, sharey=True)
    scales = [scale for _, _, scale in reversed(PUSHES)]
    scenarios = [scenario for scenario, _, _ in reversed(PUSHES)]
    for row, name in enumerate(ESNS):
        for i in range(3):
            ax = axes[row, i]
            style(ax)
            ax.axhline(1.0, color=TAUGHT_COLOR, linewidth=1.2, zorder=1)
            for joint in range(4):
                y = [joint_following(name, scenario, i)[0][joint] for scenario in scenarios]
                ax.plot(scales, y, color=TEXT_COLOR, marker=JOINT_MARKERS[joint], markersize=7, linewidth=1.2)
            pooled = [joint_following(name, scenario, i)[1] for scenario in scenarios]
            ax.plot(scales, pooled, color=ESN_COLOR, linewidth=2.4, linestyle="--", zorder=3)
            ax.set_title(f"{name}, push {i + 1}", color=TITLE_COLOR, fontsize=10, loc="left")
            ax.set_xticks(scales, ["¼", "½", "1"])
            ax.set_xlim(0.18, 1.07)
        axes[row, 0].set_ylabel("following", color=TEXT_COLOR)
    for ax in axes[-1]:
        ax.set_xlabel("share of step 5's force", color=TEXT_COLOR)
    handles = [
        Line2D([], [], color=TEXT_COLOR, marker=marker, linewidth=1.2, label=f"joint {joint + 1}")
        for joint, marker in enumerate(JOINT_MARKERS)
    ]
    handles.append(Line2D([], [], color=ESN_COLOR, linewidth=2.4, linestyle="--", label="all joints"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_offset_paths() -> Figure:
    """The hand's paths from every start of each size: a row per arm."""
    fig = Figure(figsize=(13, 11.4), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The hand's paths from every start, at each size of the offsets (dots: where they start)", color=TITLE_COLOR
    )
    axes = fig.subplots(3, len(OFFSETS), sharex=True, sharey=True)
    arms = [
        ("ESN #1", "esn", ESN_COLORS["ESN #1"]),
        ("ESN #2", "esn", ESN_COLORS["ESN #2"]),
        ("The replay", "replay", REPLAY_COLOR),
    ]
    for row, (label, arm, color) in enumerate(arms):
        esn = label if arm == "esn" else "ESN #1"
        for col, (scenario, s) in enumerate(OFFSETS):
            ax = axes[row, col]
            path_axes(ax)
            runs = offset_runs(scenario, esn, arm)
            for r in runs:
                ax.plot(*r["hand"].T, color=color, linewidth=0.9, alpha=0.75, zorder=2)
                ax.plot(*r["hand"][0], marker="o", markersize=3, color=TEXT_COLOR, zorder=3)
            ax.set_title(f"{label}, {SIZES[s]}°: {len(runs)} starts", color=TITLE_COLOR, fontsize=10)
        axes[row, 0].set_ylabel("y (m)", color=TEXT_COLOR)
    for ax in axes[-1]:
        ax.set_xlabel("x (m)", color=TEXT_COLOR)
    handles = path_handles(*((label, color) for label, _, color in arms))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_series(title: str, columns: list[tuple[str, str, str, str]], channel: str) -> Figure:
    """Each joint's angle less the taught motion, or its torque, over time: a column per run, each panel its own axis.

    ``columns`` holds each run's scenario, ESN, start, and heading. The ESN's arm, its
    reference (for the joint angles), and the replay's arm, the ESN's lines over the replay's.
    """
    pushed = any(spans(scenario) for scenario, *_ in columns)
    joints = range(4)
    head = [0.45] if pushed else []
    fig = Figure(figsize=(4.0 * len(columns) + 1.0, 10.6 + (1.4 if pushed else 0.0)), facecolor=SURFACE_COLOR)
    fig.set_layout_engine("constrained")
    what = (
        "each joint's angle less the taught motion: the arm, its reference (dashed), and the replay's arm"
        if channel == "joints"
        else "each joint's torque, the ESN's and the replay's; the axes leave out the first 50 ms of a run"
    )
    fig.suptitle(f"{title}\n{what}; each panel its own axis", color=TITLE_COLOR)
    axes = fig.subplots(
        len(head) + 4, len(columns), sharex=True, height_ratios=[*head, 1.0, 1.0, 1.0, 1.0], squeeze=False
    )
    for col, (scenario, esn, origin, heading) in enumerate(columns):
        arms = {arm: log(scenario, esn, arm, origin) for arm in ("esn", "replay")}
        for ax in axes[:, col]:
            style(ax)
            for onset, end in spans(scenario):
                ax.axvspan(onset, end, color=DISTURBANCE_COLOR, zorder=0)
        axes[0, col].set_title(heading, color=TITLE_COLOR, fontsize=10)
        if pushed:
            for arm, state in arms.items():
                axes[0, col].plot(
                    state.times,
                    R021.tip_force(state),
                    color=ESN_COLOR if arm == "esn" else REPLAY_COLOR,
                    linewidth=1.0,
                    zorder=ARM_ZORDERS[arm],
                )
            axes[0, col].set_ylim(-0.3, 1.1 * max(forces("pushes")))
        for joint in joints:
            ax = axes[len(head) + joint, col]
            shown = []
            if channel == "joints":
                ax.axhline(0.0, color=TAUGHT_COLOR, linewidth=3.0, zorder=1)
                for arm, state in arms.items():
                    taught = taught_at(state.times)[:, joint]
                    values = np.degrees(state.channel("q")[:, joint] - taught)
                    ax.plot(
                        state.times,
                        values,
                        color=ESN_COLOR if arm == "esn" else REPLAY_COLOR,
                        linewidth=1.3,
                        zorder=ARM_ZORDERS[arm],
                    )
                    shown.append(values)
                    if arm == "esn":
                        reference = np.degrees(state.channel("q_ref")[:, joint] - taught)
                        ax.plot(
                            state.times,
                            reference,
                            color=ESN_COLOR,
                            linewidth=1.0,
                            linestyle="--",
                            zorder=REFERENCE_ZORDER,
                        )
                        shown.append(reference)
            else:
                ax.axhline(0.0, color=TAUGHT_COLOR, linewidth=1.0, zorder=1)
                for arm, state in arms.items():
                    values = state.channel("tau")[:, joint]
                    ax.plot(
                        state.times,
                        values,
                        color=ESN_COLOR if arm == "esn" else REPLAY_COLOR,
                        linewidth=1.1,
                        zorder=ARM_ZORDERS[arm],
                    )
                    shown.append(values[state.times >= 0.05])
            low, high = float(min(v.min() for v in shown)), float(max(v.max() for v in shown))
            pad = 0.06 * max(high - low, 1e-6)
            ax.set_ylim(low - pad, high + pad)
            ax.set_xlim(-1.0, DURATION)
        axes[-1, col].set_xlabel("time (s)", color=TEXT_COLOR)
    if pushed:
        axes[0, 0].set_ylabel("tip force (N)", color=TEXT_COLOR)
    unit = "deg" if channel == "joints" else "N m"
    for joint in joints:
        axes[len(head) + joint, 0].set_ylabel(f"joint {joint + 1} ({unit})", color=TEXT_COLOR)
    handles = [
        Line2D([], [], color=ESN_COLOR, linewidth=1.4, label="the ESN's arm"),
        Line2D([], [], color=REPLAY_COLOR, linewidth=1.4, label="the replay's arm"),
    ]
    if channel == "joints":
        handles.insert(1, Line2D([], [], color=ESN_COLOR, linewidth=1.0, linestyle="--", label="its reference"))
        handles.append(Line2D([], [], color=TAUGHT_COLOR, linewidth=3.0, label="the taught motion (zero)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


# ---------------------------------------------------------------------------- tables


def print_tables() -> None:
    print_pushes()
    print_following()
    print_swing()
    print_offsets()
    print_lower()
    print_coupling()
    print_checks()


def cells(values: list[float], digits: int = 1) -> str:
    return " / ".join(R021.fmt(v, digits) for v in values)


def push_arms() -> list[tuple[str, str, str]]:
    """The arms of the pushes' tables: name, the ESN whose runs hold it, and the arm in the runs."""
    return [("ESN #1", "ESN #1", "esn"), ("ESN #2", "ESN #2", "esn"), ("Replay", "ESN #1", "replay")]


def farthest_before_third(esn: str, arm: str, scenario: str) -> str:
    """How far the hand gets from the target between its arrival and the third push (mm), if it leaves the goal."""
    arrival, third = run(scenario, esn, arm)["arrival_time_s"], spans(scenario)[2][0]
    state = log(scenario, esn, arm)  # every sample of the log, every 2 ms
    inside = (arrival - 1e-9 <= state.times) & (third > state.times)
    farthest = np.linalg.norm(R021.hand(state.channel("q")[inside]) - TARGET, axis=1).max()
    return R021.fmt(1000 * farthest, 0) if farthest > TOLERANCE else "in the goal"


def print_pushes() -> None:
    scenarios = [scenario for scenario, _, _ in PUSHES]
    print("\nUnder the pushes; each cell at the full, half, and quarter force:")
    print("| | ESN #1 | ESN #2 | Replay |\n| --- | --- | --- | --- |")
    measures: list[tuple[str, Callable[[str, str, str], str]]] = []
    for i in range(3):
        measures.append(
            (
                f"Hand's largest displacement, push {i + 1} (mm)",
                lambda esn, arm, scenario, i=i: R021.fmt(1000 * displacements(esn, arm)[scenario][i], 0),
            )
        )
    measures += [
        ("Path RMSE (mm)", lambda esn, arm, s: R021.fmt(1000 * run(s, esn, arm)["taught_path_rmse_m"], 1)),
        ("Hand path length (mm)", lambda esn, arm, s: R021.fmt(1000 * path_length(log(s, esn, arm)), 0)),
        ("Peak torque (N m)", lambda esn, arm, s: R021.fmt(run(s, esn, arm)["peak_torque_nm"], 2)),
        ("Accumulated torque (N m s)", lambda esn, arm, s: R021.fmt(run(s, esn, arm)["task_torque_nms"], 1)),
        ("Arrival (s)", lambda esn, arm, s: R021.fmt(run(s, esn, arm)["arrival_time_s"], 2)),
        (
            "Leaves the goal within 2 s of arriving",
            lambda esn, arm, s: "yes" if run(s, esn, arm)["left_goal"] else "no",
        ),
        ("Farthest from the target between arrival and push 3 (mm)", farthest_before_third),
        ("In the goal from then on, from (s)", lambda esn, arm, s: R021.fmt(run(s, esn, arm)["settling_time_s"], 1)),
        (
            "Ends in the goal (mm from the target)",
            lambda esn, arm, s: R021.fmt(1000 * run(s, esn, arm)["final_distance_m"], 1),
        ),
    ]
    for label, measure in measures:
        print(
            f"| {label} | "
            + " | ".join(" / ".join(measure(esn, arm, s) for s in scenarios) for _, esn, arm in push_arms())
            + " |"
        )
    print("  The undisturbed runs' largest displacement in the same windows (mm):")
    for label, esn, arm in push_arms():
        print(f"    {label}: {cells([1000 * d for d in displacements(esn, arm)['nominal']], 2)}")
    print("  The ESNs' peak and accumulated torque against the replay's (%), at the full, half, and quarter force:")
    for name in ESNS:
        peak = [
            100 * run(s, name, "esn")["peak_torque_nm"] / run(s, name, "replay")["peak_torque_nm"] for s in scenarios
        ]
        task = [
            100 * run(s, name, "esn")["task_torque_nms"] / run(s, name, "replay")["task_torque_nms"] for s in scenarios
        ]
        print(f"    {name}: peak {cells(peak, 0)}; accumulated {cells(task, 0)}")
    same = all(
        np.isclose(peak_after_start(log(s, esn, arm)), run(s, esn, arm)["peak_torque_nm"])
        for s in scenarios
        for _, esn, arm in push_arms()
    )
    print(f"  Under the pushes, every arm's peak torque comes after the first 10 ms: {same}")
    print("  Undisturbed: the peak torque, and after the first 10 ms (N m):")
    for label, esn, arm in push_arms():
        state = log("nominal", esn, arm)
        print(f"    {label}: {run('nominal', esn, arm)['peak_torque_nm']:.2f}, {peak_after_start(state):.2f}")
    print("  Each arm's displacement against its own at the full force, by push (%), at half and a quarter:")
    for label, esn, arm in push_arms():
        d = displacements(esn, arm)
        ratios = [f"{100 * d[s][i] / d['pushes'][i]:.0f}" for i in range(3) for s in scenarios[1:]]
        print(f"    {label}: {', '.join(ratios)}")
    print("  The ESNs' displacement against the replay's, over the pushes and sizes: ", end="")
    ratios = [
        displacements(name, "esn")[s][i] / displacements("ESN #1", "replay")[s][i]
        for name in ESNS
        for s in scenarios
        for i in range(3)
    ]
    print(f"{min(ratios):.2f} to {max(ratios):.2f}")
    print("  Back within 5 mm of the taught path after each push (s), at the full, half, and quarter force:")
    for label, esn, arm in push_arms():
        texts = [cells([run(s, esn, arm)[f"return_time_s_{i + 1}"] for s in scenarios], 2) for i in range(3)]
        print(f"    {label}: {'; '.join(texts)}")
    print("  Undisturbed, each joint's largest distance from the taught motion (deg):")
    for label, esn, arm in push_arms():
        state = log("nominal", esn, arm)
        gap = np.degrees(np.abs(state.channel("q") - taught_at(state.times)))[state.times >= 0].max(axis=0)
        print(f"    {label}: {np.round(gap, 2).tolist()}")
    print("  When the hand is out of the goal after arriving (s):")
    for label, esn, arm in push_arms():
        for s in scenarios:
            spans_out = out_of_goal(log(s, esn, arm), run(s, esn, arm)["arrival_time_s"])
            print(f"    {label}, {s}: {', '.join(f'{a:.2f}-{b:.2f}' for a, b in spans_out) or 'never'}")


def print_following() -> None:
    scenarios = [scenario for scenario, _, _ in PUSHES]
    print("\nThe following under each push; each cell at the full, half, and quarter force:")
    print(
        "| | Hand | Joint 1 | Joint 2 | Joint 3 | Joint 4 | All joints |\n| --- | --- | --- | --- | --- | --- | --- |"
    )
    for name in ESNS:
        for i in range(3):
            hand = [run(s, name, "esn")[f"following_gain_{i + 1}"] for s in scenarios]
            per = [[joint_following(name, s, i)[0][joint] for s in scenarios] for joint in range(4)]
            pooled = [joint_following(name, s, i)[1] for s in scenarios]
            print(
                f"| {name}, push {i + 1} | {cells(hand, 2)} | "
                + " | ".join(cells(p, 2) for p in per)
                + f" | {cells(pooled, 2)} |"
            )
    replay = [abs(run(s, "ESN #1", "replay")[f"following_gain_{i + 1}"]) for s in scenarios for i in range(3)]
    print(f"  The replay's hand following, largest: {max(replay):.2f}")
    hand_changes: list[tuple[float, str]] = []
    joint_changes: list[tuple[float, str]] = []
    for name in ESNS:
        undisturbed = log("nominal", name, "esn")
        for s in scenarios:
            for i, (onset, end) in enumerate(spans(s)):
                state = log(s, name, "esn")
                where = f"{name}, {s}, push {i + 1}"
                hand = hand_following_less_undisturbed(state, undisturbed, onset, end)
                hand_changes.append((abs(hand - run(s, name, "esn")[f"following_gain_{i + 1}"]), where))
                per = joint_following_less_undisturbed(state, undisturbed, onset, end) - joint_following(name, s, i)[0]
                joint = int(np.argmax(np.abs(per)))
                joint_changes.append((float(abs(per[joint])), f"{where}, joint {joint + 1}"))
    print("  Less the same arm's undisturbed run in place of the take, the largest changes of the gains:")
    print(f"    the hand's: {max(hand_changes)[0]:.3f} ({max(hand_changes)[1]})")
    print(f"    the joints': {max(joint_changes)[0]:.3f} ({max(joint_changes)[1]})")
    print("  Joint 1 under push 1: its time at the lower stop from the onset to 1 s after, and its reference's lowest:")
    onset = spans("pushes")[0][0]
    for name in ESNS:
        texts = []
        for s in scenarios:
            state = log(s, name, "esn")
            inside = (state.times >= onset) & (state.times < onset + 1.0)
            at_stop = float((state.channel("q")[inside, 0] <= LOWER[0] + 1e-9).sum() * np.diff(state.times).mean())
            lowest = float(np.degrees(state.channel("q_ref")[inside, 0].min()))
            texts.append(f"{at_stop:.2f} s, {lowest:.1f}")
        print(f"    {name}: {' / '.join(texts)}")


def extremes(values: NDArray[np.float64], times: NDArray[np.float64], share: float = 0.2) -> list[tuple[float, float]]:
    """A swing's turning points: the largest or smallest samples within 0.5 s, above ``share`` of its size."""
    size = np.abs(values).max()
    half = round(0.5 / DT)
    out: list[tuple[float, float]] = []
    for k in range(len(values)):
        near = values[max(k - half, 0) : k + half + 1]
        turning = abs(values[k]) > share * size and values[k] in (near.max(), near.min())
        if turning and (not out or times[k] - out[-1][0] > 0.5):
            out.append((float(times[k]), float(values[k])))
    return out


def print_swing() -> None:
    """Joint 1's swing between the second push's end and the third's onset: its turning points."""
    end, third = spans("pushes")[1][1], spans("pushes")[2][0]
    inside = (end - 1e-9 <= TIMES) & (third > TIMES)
    print("\nJoint 1 less the taught motion between push 2's end and push 3's onset: its turning points (s, deg):")
    for name in ESNS:
        for scenario, label, _ in PUSHES:
            values = np.degrees(sampled(log(scenario, name, "esn"), "q")[:, 0] - taught_at(TIMES)[:, 0])[inside]
            points = extremes(values, TIMES[inside])
            print(f"  {name}, {label}: " + ", ".join(f"{t:.2f} ({v:+.1f})" for t, v in points))


OFFSET_MEASURES: list[tuple[str, Callable[[list[dict[str, Any]]], str]]] = [
    ("Arrive and never leave the goal", lambda runs: f"{sum(holds(r) for r in runs)}"),
    ("Latest arrival (s)", lambda runs: f"{max(r['arrival_time_s'] for r in runs):.2f}"),
    ("Path RMSE, median (mm)", lambda runs: f"{1000 * np.median([r['taught_path_rmse_m'] for r in runs]):.1f}"),
    ("Path RMSE, largest (mm)", lambda runs: f"{1000 * max(r['taught_path_rmse_m'] for r in runs):.1f}"),
    ("Peak torque, median (N m)", lambda runs: f"{np.median([r['peak_torque_nm'] for r in runs]):.1f}"),
    ("Accumulated torque, median (N m s)", lambda runs: f"{np.median([r['task_torque_nms'] for r in runs]):.1f}"),
    ("Accumulated torque, largest (N m s)", lambda runs: f"{max(r['task_torque_nms'] for r in runs):.1f}"),
    (
        "Starts with joints 2 to 4 at a stop over 0.1 s",
        lambda runs: f"{sum(r['at_stop_s'][1:].max() > 0.1 for r in runs)}",
    ),
    ("Starts with the reference over 5° past a limit", lambda runs: f"{sum(r['past_deg'] > 5 for r in runs)}"),
    (
        "Reference to arm in the first second, median (°)",
        lambda runs: f"{np.median([r['first_second_deg'] for r in runs]):.1f}",
    ),
    ("Offset retained, median", lambda runs: f"{np.median([r['offset_retained'] for r in runs]):.2f}"),
]


def offset_arms() -> list[tuple[str, str, str]]:
    return [("ESN #1", "ESN #1", "esn"), ("ESN #2", "ESN #2", "esn"), ("Replay", "ESN #1", "replay")]


def print_offset_table(which: str, sizes: list[tuple[str, float]]) -> None:
    counts = {len(group(scenario, "ESN #1", "esn", which)) for scenario, _ in sizes}
    print(f"\nFrom the {which} offset starts ({counts}); each cell at {', '.join(SIZES[s] for _, s in sizes)} deg:")
    print("| | ESN #1 | ESN #2 | Replay |\n| --- | --- | --- | --- |")
    for label, measure in OFFSET_MEASURES:
        print(
            f"| {label} | "
            + " | ".join(
                " / ".join(measure(group(scenario, esn, arm, which)) for scenario, _ in sizes)
                for _, esn, arm in offset_arms()
            )
            + " |"
        )
    print("  The ESNs' medians against the replay's (%): peak torque; accumulated torque:")
    for name in ESNS:
        texts = []
        for scenario, s in sizes:
            mine, replay = group(scenario, name, "esn", which), group(scenario, name, "replay", which)
            peak = (
                100 * np.median([r["peak_torque_nm"] for r in mine]) / np.median([r["peak_torque_nm"] for r in replay])
            )
            task = (
                100
                * np.median([r["task_torque_nms"] for r in mine])
                / np.median([r["task_torque_nms"] for r in replay])
            )
            texts.append(f"{SIZES[s]}: {peak:.0f}, {task:.0f}")
        print(f"    {name}: {'; '.join(texts)}")
    print("  The reference's furthest past a limit (deg), and the longest time of joints 2 to 4 at a stop (s):")
    for label, esn, arm in offset_arms():
        texts = []
        for scenario, s in sizes:
            runs = group(scenario, esn, arm, which)
            texts.append(
                f"{SIZES[s]}: {max(r['past_deg'] for r in runs):.1f}, {max(r['at_stop_s'][1:].max() for r in runs):.2f}"
            )
        print(f"    {label}: {'; '.join(texts)}")


def print_offsets() -> None:
    print_offset_table("shared", OFFSETS)
    print("  How far the starts move the hand from the taught start (mm), median and largest, over the moved starts:")
    start = taught_hand()[0]
    for scenario, s in OFFSETS:
        runs = [r for r in offset_runs(scenario, "ESN #1", "replay") if r["moved"]]
        distances = [1000 * np.linalg.norm(r["hand"][0] - start) for r in runs]
        shared = [1000 * np.linalg.norm(r["hand"][0] - start) for r in runs if r["offset"][1] >= 0]
        print(
            f"    {SIZES[s]}: all {len(runs)}: {np.median(distances):.1f}, {max(distances):.1f};"
            f" the {len(shared)} shared: {np.median(shared):.1f}, {max(shared):.1f}"
        )
    print("  The start farthest from the target at each size (mm), against the take's start:")
    for scenario, s in OFFSETS:
        runs = offset_runs(scenario, "ESN #1", "replay")
        far = max(runs, key=lambda r: np.linalg.norm(r["hand"][0] - TARGET))
        take = next(r for r in runs if not r["moved"])
        print(
            f"    {SIZES[s]}: {far['origin'].split()[-2]}, {1000 * np.linalg.norm(far['hand'][0] - TARGET):.0f};"
            f" the take's start {1000 * np.linalg.norm(take['hand'][0] - TARGET):.0f}"
        )
    print("  The starts with a joint of 2 to 4 at a stop, or the reference over 5 deg past a limit, at 5 and 2.5 deg:")
    for name in ESNS:
        for scenario, s in OFFSETS[1:]:
            for r in offset_runs(scenario, name, "esn"):
                if r["at_stop_s"][1:].max() > 0 or r["past_deg"] > 5:
                    print(
                        f"    {name}, {SIZES[s]}, {r['origin'].split()[-2]}: past {r['past_deg']:.1f} deg,"
                        f" at a stop {np.round(r['at_stop_s'], 2).tolist()} s, arrival {r['arrival_time_s']:.2f} s,"
                        f" RMSE {1000 * r['taught_path_rmse_m']:.1f} mm, accumulated {r['task_torque_nms']:.1f} N m s"
                    )


def print_lower() -> None:
    print_offset_table("lower", OFFSETS[1:])
    print(
        "  From (0, -s, -s, -s) (the arrival, path RMSE, furthest past a limit, time at the stops, accumulated torque):"
    )
    for scenario, s in OFFSETS[1:]:
        origin = origin_of("+0,-{s},-{s},-{s}", s)
        for label, esn, arm in offset_arms():
            r = next(r for r in offset_runs(scenario, esn, arm) if r["origin"] == origin)
            print(
                f"    {SIZES[s]}, {label}: {r['arrival_time_s']:.2f} s, {1000 * r['taught_path_rmse_m']:.1f} mm,"
                f" {r['past_deg']:.1f} deg, {np.round(r['at_stop_s'], 2).tolist()} s, {r['task_torque_nms']:.1f} N m s"
            )
    print("  The latest arrival from the lower starts but (0, -5, -5, -5) for ESN #1, of any arm (s):", end=" ")
    latest = max(
        r["arrival_time_s"]
        for scenario, _ in OFFSETS[1:]
        for _, esn, arm in offset_arms()
        for r in group(scenario, esn, arm, "lower")
        if not (esn == "ESN #1" and arm == "esn" and r["origin"] == origin_of("+0,-{s},-{s},-{s}", 5.0))
    )
    print(f"{latest:.2f}")


def print_coupling() -> None:
    """ESN #1 from (0, 0, -5, -5) at 1 s: the stiffness part of joint 2's torque, joint by joint."""
    origin = origin_of("+0,+0,-{s},-{s}", 5.0)
    state = log("offsets_5deg", "ESN #1", "esn", origin)
    k = int(np.argmin(np.abs(state.times - 1.0)))
    inertia = mass_matrix_at(Skeleton.from_toml(R021.ARM_CONFIG), R021.taught()[-1])
    stiffness = R021.OMEGA**2 * inertia
    error = state.channel("q_ref")[k] - state.channel("q")[k]
    parts = stiffness[1] * error
    applied = state.channel("tau")[k, 1]
    print(f"\nESN #1 from {origin.split()[1]} at {state.times[k]:.2f} s:")
    print(f"  the reference less the arm (deg): {np.round(np.degrees(error), 2).tolist()}")
    print(f"  the arm (deg): {np.round(np.degrees(state.channel('q')[k]), 2).tolist()}")
    print(
        f"  joint 2's torque: from each joint's error (Kp e), {np.round(parts, 2).tolist()} N m, summing to"
        f" {parts.sum():.2f}; the rest, from the velocities, {applied - parts.sum():.2f}; applied {applied:.2f} N m"
    )
    for scenario, size in OFFSETS:
        run_ = next(
            r for r in offset_runs(scenario, "ESN #1", "esn") if r["origin"] == origin_of("+0,+0,-{s},-{s}", size)
        )
        print(
            f"  From {run_['origin'].split()[1]}: the reference past a limit by {run_['past_deg']:.1f} deg;"
            f" at a stop {np.round(run_['at_stop_s'], 3).tolist()} s"
        )
    for which in ("shared", "lower"):
        longest = max(r["at_stop_s"][1:].max() for name in ESNS for r in group("offsets_2.5deg", name, "esn", which))
        print(f"  At 2.5 deg, the {which} starts: joints 2 to 4's longest time at a stop {longest:.3f} s")


def print_checks() -> None:
    """The replays against the progress replays, and the run records."""
    largest = 0.0
    count = 0
    for scenario in [s for s, _, _ in PUSHES[1:]] + [s for s, _ in OFFSETS[1:]]:
        for name in ESNS:
            for row in (r for r in rows(scenario, name) if r["arm"] == "replay"):
                replay = R021.robot_log(scenario, ESNS[name], "replay", row["origin"], RESULTS)
                progress = R021.robot_log(scenario, ESNS[name], "progress", row["origin"], RESULTS)
                for channel in ("q", "tau", "q_ref"):
                    largest = max(largest, float(np.abs(replay.channel(channel) - progress.channel(channel)).max()))
                count += 1
    print(
        f"\nThe progress replay against the replay in step 6's {count} pairs of runs: largest difference {largest:.1e}"
    )
    records = sorted(RESULTS.glob("*/run.toml"))
    states = set()
    for path in records:
        with path.open("rb") as f:
            record = tomllib.load(f)
        states.add((record["commit"][:7], record["uncommitted_changes"]))
    print(f"Step 6's {len(records)} run records: (commit, uncommitted changes) {sorted(states)}")


if __name__ == "__main__":
    main()
