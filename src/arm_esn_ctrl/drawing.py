# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Drawing the robot in figures: its postures over a hand path, in the colors of skelarm's apps.

A figure of hand paths shows where the arm was with a few faint postures: the
first, the last, and some between them, spread along the path by its length
(:func:`spread_along_path`) and drawn by :func:`draw_postures`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray
    from skelarm import Skeleton

# The colors of skelarm's canvas and of its draw_skeleton: the movable links, the fixed base link, the joints,
# and the origin.
LINK_COLOR = "#0064c8"
BASE_LINK_COLOR = "#969696"
JOINT_COLOR = "#00aa00"
ORIGIN_COLOR = "black"
POSTURE_ALPHA = 0.3  # faint, so that the postures stay behind the paths


def spread_along_path(hand: NDArray[np.float64], count: int = 5) -> list[int]:
    """The samples of ``count`` postures spread along the hand path ``hand`` (shape ``(n, 2)``).

    The first and the last sample, and between them the first samples at equal
    distances along the path, by its length rather than by time, so that a pause
    adds none. A path that never moves gives its first and last sample.
    """
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(hand, axis=0), axis=1))])
    if along[-1] == 0:
        return [0, len(hand) - 1]
    indices = [int(np.searchsorted(along, mark - 1e-12)) for mark in np.linspace(0.0, along[-1], count)]
    indices[-1] = len(hand) - 1
    return list(dict.fromkeys(indices))


def draw_postures(
    ax: Axes, skeleton: Skeleton, postures: NDArray[np.float64], alpha: float = POSTURE_ALPHA, linewidth: float = 2.5
) -> None:
    """Draw the arm at each row of joint angles ``postures`` (rad), faintly, in the colors of skelarm's apps.

    The links are drawn beneath the hand paths (``zorder`` 0.5) and add no legend
    entries.
    """
    arm = skeleton.clone()
    for q in postures:
        arm.q = q  # the setter refreshes the forward kinematics
        for i, link in enumerate(arm.links):
            color = BASE_LINK_COLOR if i == 0 else LINK_COLOR
            ax.plot([link.x, link.xe], [link.y, link.ye], color=color, linewidth=linewidth, alpha=alpha, zorder=0.5)
        base = arm.links[0]
        ax.plot(
            [base.x], [base.y], linestyle="none", marker="o", markersize=5, color=ORIGIN_COLOR, alpha=alpha, zorder=0.6
        )
        joints_x, joints_y = [link.xe for link in arm.links], [link.ye for link in arm.links]
        ax.plot(
            joints_x, joints_y, linestyle="none", marker="o", markersize=5, color=JOINT_COLOR, alpha=alpha, zorder=0.6
        )
