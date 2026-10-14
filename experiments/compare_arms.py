# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Overlay the arms of several robot runs that met the same forces: joint angles and torques over time (Stage 2).

    uv run python experiments/compare_arms.py experiments/<experiment>/compare_<name>.toml

Each ``[[arm]]`` table names an arm of a run of ``experiments/robot_esn.py``: its
``run`` (a path under the storage root), the ``arm`` in it (``esn`` or ``replay``),
a ``label``, and a ``color``. The arms are drawn from the start posture ``start``
of ``[plot]`` (0 by default) over its ``span`` (s, the whole run by default), with
each tracker setting that every run has, one column each. The runs should share
their disturbance, such as a force recorded in the robot app (a ``[disturbance]``
of ``type = "recorded"``), so that every arm meets the same forces at the same
times: the comparison refuses arms whose tip forces differ. The run directory
receives:

- ``joints.png``: the tip force's magnitude, then each joint's angle over time: the
  taught motion (thick gray), each arm, and each ESN's reference (dashed);
- ``torques.png``: the tip force's magnitude, then each joint's torque over time;
- ``paths.png``: each arm's tip path in the plane, one panel per arm (rows) and
  tracker setting (columns), over the taught path (thick gray), the goal, and the
  arm at its initial posture (faint gray); at the start of each disturbance, an
  arrow from the tip shows the mean force over the disturbance, ``FORCE_SCALE``
  meters per newton.

Over time, the ESNs' lines are drawn over the replay's, and the spans of the
disturbances are shaded.
"""

from __future__ import annotations

import argparse
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrow
from numpy.typing import NDArray
from skelarm import Skeleton, StateLog, Task, compute_forward_kinematics

from arm_esn_ctrl.demonstrations import endpoint_positions, load_joint_angles
from arm_esn_ctrl.disturbances import disturbance_spans
from arm_esn_ctrl.storage import resolve_run_path, start_run

TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
DISTURBANCE_COLOR = "#f0efec"
TAUGHT_COLOR = "#b9b7b0"
FORCE_COLOR = "#52514e"
PUSH_COLOR = "#dc0000"  # the force arrows in the plane, red as the robot app's drag arrow
INITIAL_COLOR = "#a3a29d"
FORCE_SCALE = 0.025  # m/N: the length of a force arrow in the plane, 25 mm per newton
# The layers of the lines over time: the ESNs' arms on top, their references (dashed) under them, the replay below.
ARM_ZORDERS = {"esn": 3.0, "replay": 2.0}
REFERENCE_ZORDER = 2.5


@dataclass(frozen=True)
class Arm:
    """One arm of one run, as an ``[[arm]]`` table names it."""

    label: str
    run: Path  # the run directory
    arm: str  # "esn" or "replay"
    color: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="comparison configuration file (TOML)")
    args = parser.parse_args()

    config, run_dir = start_run(args.config)
    arms = [
        Arm(table["label"], resolve_run_path(table["run"]), table["arm"], table["color"]) for table in config["arm"]
    ]
    options = config.get("plot", {})
    start = int(options.get("start", 0))
    settings = shared_settings(arms)
    logs = {setting: {arm.label: load_log(arm, setting, start) for arm in arms} for setting in settings}
    check_same_forces(logs)
    with (arms[0].run / "config.toml").open("rb") as f:
        run_config = tomllib.load(f)
    spans = disturbance_spans(run_config.get("disturbance"))
    taught = taught_motion(arms[0].run)
    span = options.get("span")
    title = f"{args.config.stem}: from start {start}"
    for channel in ("joints", "torques"):
        figure = plot_arms(logs, arms, channel, taught, spans, span, title)
        figure.savefig(run_dir / f"{channel}.png", dpi=150)
    first = logs[settings[0]][arms[0].label]
    task = Task.from_dict(first.extra["playback"]["task"])
    skeleton = first.build_skeleton()
    taught_hand = endpoint_positions(skeleton, taught[1])
    figure = plot_paths(logs, arms, spans, skeleton, taught_hand, task.require_target(), task.tolerance or 0.0, title)
    figure.savefig(run_dir / "paths.png", dpi=150)
    print(f"{len(arms)} arms, tracker settings {', '.join(settings)}; wrote the results to {run_dir}")


def shared_settings(arms: list[Arm]) -> list[str]:
    """The tracker settings every arm's run has (its ``<law>_w<omega>[_z<damping>]`` directories), in name order.

    Raises
    ------
    ValueError
        If the runs share none.
    """
    shared = None
    for arm in arms:
        names = {path.name for path in arm.run.iterdir() if path.is_dir()}
        shared = names if shared is None else shared & names
    if not shared:
        msg = "the runs share no tracker setting"
        raise ValueError(msg)
    return sorted(shared)


def load_log(arm: Arm, setting: str, start: int) -> StateLog:
    """The log of an arm with a tracker setting, from a start posture."""
    return StateLog.load(arm.run / setting / f"{arm.arm}_{start:02d}.sklog.npz")


def check_same_forces(logs: dict[str, dict[str, StateLog]]) -> None:
    """Refuse arms that did not meet the same tip force at the same times.

    Raises
    ------
    ValueError
        If two arms' times or ``ext_force`` differ, the final frame excepted (its force acts on no step).
    """
    for setting, arms in logs.items():
        (first, reference), *others = arms.items()
        for label, log in others:
            same_times = np.array_equal(log.times, reference.times)
            same_force = same_times and np.array_equal(_force(log)[:-1], _force(reference)[:-1])
            if not same_force:
                msg = f"{label} and {first} met different forces with {setting}: compare runs of the same disturbance"
                raise ValueError(msg)


def taught_motion(run: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The take the run's ESN was trained on, which its replay replays: times and joint angles (rad)."""
    with (run / "config.toml").open("rb") as f:
        model = resolve_run_path(tomllib.load(f)["esn"]["model"])
    with (model.parent / "config.toml").open("rb") as f:
        esn_config = tomllib.load(f)
    demonstrations = esn_config["demonstrations"]
    take = resolve_run_path(demonstrations["run"]) / demonstrations["train"][0]
    return load_joint_angles(take, esn_config["esn"]["dt"])


def plot_arms(
    logs: dict[str, dict[str, StateLog]],
    arms: list[Arm],
    channel: str,
    taught: tuple[NDArray[np.float64], NDArray[np.float64]],
    spans: list[tuple[float, float]],
    span: list[float] | None,
    title: str,
) -> Figure:
    """The tip force, then each joint's angle (``channel = "joints"``) or torque (``"torques"``), per setting."""
    settings = list(logs)
    n_joints = logs[settings[0]][arms[0].label].channel("q").shape[1]
    fig = Figure(
        figsize=(4.8 * len(settings) + 1.0, 2.2 + 2.6 * n_joints), facecolor=SURFACE_COLOR, layout="constrained"
    )
    what = "joint angles" if channel == "joints" else "joint torques"
    fig.suptitle(f"{title}: the tip force and the {what}", color="#0b0b0b")
    axes = np.asarray(
        fig.subplots(1 + n_joints, len(settings), sharex=True, height_ratios=[0.6] + [1.0] * n_joints)
    ).reshape(1 + n_joints, len(settings))
    for col, setting in enumerate(settings):
        reference = logs[setting][arms[0].label]
        for row in range(1 + n_joints):
            ax = axes[row, col]
            _style(ax)
            for onset, end in spans:
                ax.axvspan(onset, end, color=DISTURBANCE_COLOR, zorder=0)
        axes[0, col].plot(reference.times, np.linalg.norm(_force(reference), axis=1), color=FORCE_COLOR, linewidth=1.0)
        axes[0, col].set_title(setting, color="#0b0b0b", fontsize=10)
        for joint in range(n_joints):
            ax = axes[1 + joint, col]
            if channel == "joints":
                times, q = taught
                ax.plot(times, np.degrees(q[:, joint]), color=TAUGHT_COLOR, linewidth=4.0, zorder=1)
            for arm in arms:
                _draw_arm(ax, logs[setting][arm.label], arm, channel, joint)
            if span is not None:
                ax.set_xlim(*span)
        axes[-1, col].set_xlabel("time (s)", color=TEXT_COLOR)
    axes[0, 0].set_ylabel("tip force (N)", color=TEXT_COLOR)
    unit = "deg" if channel == "joints" else "N m"
    for joint in range(n_joints):
        axes[1 + joint, 0].set_ylabel(f"joint {joint + 1} ({unit})", color=TEXT_COLOR)
    handles = [Line2D([], [], color=arm.color, linewidth=1.4, label=arm.label) for arm in arms]
    if channel == "joints":
        handles += [
            Line2D([], [], color=TEXT_COLOR, linewidth=1.0, linestyle="--", label="an ESN's reference"),
            Line2D([], [], color=TAUGHT_COLOR, linewidth=4.0, label="the taught motion"),
        ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_paths(
    logs: dict[str, dict[str, StateLog]],
    arms: list[Arm],
    spans: list[tuple[float, float]],
    skeleton: Skeleton,
    taught_hand: NDArray[np.float64],
    target: NDArray[np.float64],
    radius: float,
    title: str,
) -> Figure:
    """Each arm's tip path in the plane, one panel per arm and tracker setting, with the forces of ``spans``.

    Each panel draws the arm at its initial posture, the taught path, the goal, the
    tip's path, and, for each disturbance, an arrow from where the tip is at its
    onset along the mean force over it.
    """
    settings = list(logs)
    # The plane is about twice as wide as it is tall, as the reach goes across it.
    size = (4.6 * len(settings) + 0.6, 2.7 * len(arms) + 1.2)
    fig = Figure(figsize=size, facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(f"{title}: the tip's path, and the force at the start of each push", color="#0b0b0b")
    axes = np.asarray(fig.subplots(len(arms), len(settings), sharex=True, sharey=True)).reshape(
        len(arms), len(settings)
    )
    for row, arm in enumerate(arms):
        for col, setting in enumerate(settings):
            ax = axes[row, col]
            _style(ax)
            log = logs[setting][arm.label]
            q = log.channel("q").reshape(len(log.times), -1)
            joints = _arm_points(skeleton, q[0])
            ax.plot(joints[:, 0], joints[:, 1], color=INITIAL_COLOR, linewidth=5.0, alpha=0.6, solid_capstyle="round")
            ax.plot(joints[:, 0], joints[:, 1], "o", color=INITIAL_COLOR, markersize=4)
            ax.plot(taught_hand[:, 0], taught_hand[:, 1], color=TAUGHT_COLOR, linewidth=4.0, zorder=1)
            ax.add_patch(Circle(tuple(target), radius, fill=False, color=TEXT_COLOR, linewidth=1.0, zorder=2))
            hand = endpoint_positions(skeleton, q)
            ax.plot(hand[:, 0], hand[:, 1], color=arm.color, linewidth=1.2, zorder=3)
            for tip, force in push_arrows(log, hand, spans):
                ax.add_patch(
                    FancyArrow(
                        *tip,
                        *(FORCE_SCALE * force),
                        width=0.006,
                        length_includes_head=True,
                        head_width=0.03,
                        head_length=0.03,
                        color=PUSH_COLOR,
                        zorder=4,
                    )
                )
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(setting, color="#0b0b0b", fontsize=10)
            if row == len(arms) - 1:
                ax.set_xlabel("x (m)", color=TEXT_COLOR)
            if col == 0:
                ax.set_ylabel(f"{arm.label}\ny (m)", color=TEXT_COLOR)
    handles = [Line2D([], [], color=arm.color, linewidth=1.4, label=arm.label) for arm in arms]
    handles += [
        Line2D([], [], color=TAUGHT_COLOR, linewidth=4.0, label="the taught path"),
        Line2D([], [], color=INITIAL_COLOR, linewidth=5.0, alpha=0.6, label="the initial posture"),
        Line2D([], [], color=PUSH_COLOR, linewidth=2.0, label=f"mean force of a push, {1000 * FORCE_SCALE:g} mm per N"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def push_arrows(
    log: StateLog, hand: NDArray[np.float64], spans: list[tuple[float, float]]
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """For each span of ``spans``: where the tip is at its onset, and the mean tip force over it."""
    times = np.asarray(log.times)
    force = _force(log)
    arrows = []
    for onset, end in spans:
        acting = (times > onset - 1e-9) & (times < end - 1e-9)
        if acting.any():
            first = int(np.flatnonzero(acting)[0])
            arrows.append((hand[first], force[acting].mean(axis=0)))
    return arrows


def _arm_points(skeleton: Skeleton, q: NDArray[np.float64]) -> NDArray[np.float64]:
    """The base, every joint, and the tip of the arm at ``q``."""
    arm = skeleton.clone()
    for link, angle in zip(arm.links[1:], q, strict=True):
        link.q = float(angle)
    compute_forward_kinematics(arm)
    return np.array([[0.0, 0.0]] + [[link.xe, link.ye] for link in arm.links[1:]])


def _draw_arm(ax: Axes, log: StateLog, arm: Arm, channel: str, joint: int) -> None:
    n = len(log.times)
    zorder = ARM_ZORDERS.get(arm.arm, 2.0)
    if channel == "joints":
        q = np.degrees(log.channel("q").reshape(n, -1)[:, joint])
        ax.plot(log.times, q, color=arm.color, linewidth=1.2, zorder=zorder)
        if arm.arm == "esn" and "q_ref" in log.channel_names:
            q_ref = np.degrees(log.channel("q_ref").reshape(n, -1)[:, joint])
            ax.plot(log.times, q_ref, color=arm.color, linewidth=1.0, linestyle="--", zorder=REFERENCE_ZORDER)
    else:
        ax.plot(log.times, log.channel("tau").reshape(n, -1)[:, joint], color=arm.color, linewidth=1.0, zorder=zorder)


def _force(log: StateLog) -> NDArray[np.float64]:
    """The tip force of every frame (zero for a run without one)."""
    if "ext_force" in log.channel_names:
        return log.channel("ext_force")
    return np.zeros((len(log.times), 2))


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


if __name__ == "__main__":
    main()
