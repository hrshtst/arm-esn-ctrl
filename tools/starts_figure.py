# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Draw the start postures of a demonstration configuration, to judge them before recording.

    uv run python tools/starts_figure.py experiments/demonstrations/reach_manual_v2_starts.toml --output starts.png

The left panel shows the workspace: the arm at each start posture of
``[demonstrations]`` ``start_q``, numbered as its take (``demo_00``, ...), the
target and its goal radius, and the ring the hand can reach. The right panel shows
the same starts in joint space, within the joint limits, with the postures that put
the hand at the target (stars). ``--starts`` adds the start postures no take
starts from that another configuration lists, such as an ESN's
``[evaluation.extra_starts]``, as hollow markers. A task without a target, such as
a periodic motion, shows its curve instead.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from skelarm import Skeleton, Task, compute_inverse_kinematics

from arm_esn_ctrl.demonstrations import endpoint_positions, task_path
from arm_esn_ctrl.drawing import draw_postures
from arm_esn_ctrl.postures import joint_limits_deg

TAUGHT_COLOR = "#2a78d6"
UNTAUGHT_COLOR = "#eb6834"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="demonstration configuration file (TOML)")
    parser.add_argument("--output", type=Path, required=True, help="the figure to write, such as starts.png")
    parser.add_argument(
        "--starts", type=Path, help="a configuration whose [evaluation.extra_starts] are drawn too, hollow"
    )
    args = parser.parse_args()
    with args.config.open("rb") as f:
        config = tomllib.load(f)
    extra: dict[str, list[list[float]]] = {}
    if args.starts is not None:
        with args.starts.open("rb") as f:
            extra = tomllib.load(f).get("evaluation", {}).get("extra_starts", {})
    skeleton = Skeleton.from_toml(args.config)
    figure = plot_starts(skeleton, config["task"], config["demonstrations"]["start_q"], extra, args.config.stem)
    figure.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")


def target_postures(skeleton: Skeleton, target: np.ndarray) -> np.ndarray:
    """The postures that put the hand at ``target`` (deg), found from seeds all over the joint space, each once."""
    lower, upper = joint_limits_deg(skeleton)
    found: list[np.ndarray] = []
    rng = np.random.default_rng(0)
    for _ in range(60):
        arm = skeleton.clone()
        arm.q = np.radians(rng.uniform(lower, upper))
        try:
            q = np.degrees(compute_inverse_kinematics(arm, tuple(target)).q)
        except (ValueError, RuntimeError):
            continue
        reached = np.linalg.norm(endpoint_positions(skeleton, np.radians(q)[np.newaxis])[0] - target) < 1e-4
        if reached and not any(np.abs(q - other).max() < 1.0 for other in found):
            found.append(q)
    return np.array(found)


def plot_starts(
    skeleton: Skeleton,
    task_table: dict,
    starts_deg: list[list[float]],
    extra: dict[str, list[list[float]]],
    title: str,
) -> Figure:
    """The start postures in the workspace and in joint space (see the module)."""
    task = Task.from_dict(task_table)
    starts = np.radians(starts_deg)
    hands = endpoint_positions(skeleton, starts)
    lengths = [link.prop.length for link in skeleton.links[1:]]
    reach, inner = sum(lengths), max(2 * max(lengths) - sum(lengths), 0.0)
    lower, upper = joint_limits_deg(skeleton)

    fig = Figure(figsize=(13, 6.5), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(f"Start postures: {title}", color="#0b0b0b")
    ax_work, ax_joint = fig.subplots(1, 2)
    for ax in (ax_work, ax_joint):
        ax.set_facecolor(SURFACE_COLOR)
        ax.grid(color=GRID_COLOR, linewidth=0.8)
        ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)

    ring = np.linspace(0.0, 2 * np.pi, 200)
    for radius in (reach, inner):
        ax_work.plot(radius * np.cos(ring), radius * np.sin(ring), color=TEXT_COLOR, linewidth=0.8, linestyle=":")
    draw_postures(ax_work, skeleton, starts, alpha=0.45)
    ax_work.scatter(hands[:, 0], hands[:, 1], color=TAUGHT_COLOR, s=45, zorder=3, label="taught starts (hand)")
    for i, hand in enumerate(hands):
        ax_work.annotate(f"{i:02d}", hand, textcoords="offset points", xytext=(6, 6), color=TAUGHT_COLOR, fontsize=9)
    for group, postures in extra.items():
        untaught = endpoint_positions(skeleton, np.radians(postures))
        ax_work.scatter(
            untaught[:, 0], untaught[:, 1], facecolors="none", edgecolors=UNTAUGHT_COLOR, s=30, zorder=2, label=group
        )
    path = task_path(task_table)
    if path is not None:
        ax_work.plot(path[:, 0], path[:, 1], color="#0b0b0b", linewidth=1.2, label="the task's curve")
    if task.target is not None:
        ax_work.plot(*task.target, marker="+", markersize=13, color="#0b0b0b", markeredgewidth=1.6, linestyle="none")
        if task.tolerance is not None:
            ax_work.plot(
                task.target[0] + task.tolerance * np.cos(ring),
                task.target[1] + task.tolerance * np.sin(ring),
                color="#0b0b0b",
                linewidth=0.9,
            )
    ax_work.set(
        title="Workspace (cross: target; dotted: the hand's reach)", xlabel="x (m)", ylabel="y (m)", aspect="equal"
    )
    ax_work.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=TEXT_COLOR)

    taught = np.asarray(starts_deg, dtype=np.float64)
    ax_joint.scatter(taught[:, 0], taught[:, 1], color=TAUGHT_COLOR, s=45, zorder=3)
    for i, q in enumerate(taught):
        ax_joint.annotate(f"{i:02d}", q[:2], textcoords="offset points", xytext=(6, 6), color=TAUGHT_COLOR, fontsize=9)
    for postures in extra.values():
        q = np.asarray(postures, dtype=np.float64)
        ax_joint.scatter(q[:, 0], q[:, 1], facecolors="none", edgecolors=UNTAUGHT_COLOR, s=30, zorder=2)
    if task.target is not None:
        at_target = target_postures(skeleton, task.target)
        if len(at_target):
            ax_joint.plot(
                at_target[:, 0], at_target[:, 1], marker="*", markersize=14, color="#0b0b0b", linestyle="none"
            )
    ax_joint.set(
        title="Joint space (stars: the hand at the target)",
        xlabel="joint 1 (deg)",
        ylabel="joint 2 (deg)",
        xlim=(lower[0] - 5, upper[0] + 5),
        ylim=(lower[1] - 5, upper[1] + 5),
    )
    return fig


if __name__ == "__main__":
    main()
