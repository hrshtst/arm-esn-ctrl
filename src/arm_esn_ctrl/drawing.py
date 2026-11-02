# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Drawing the robot in figures: its postures over a hand path, in the colors of skelarm's apps; offset maps.

A figure of hand paths shows where the arm was with a few faint postures: the
first, the last, and some between them, spread along the path by its length
(:func:`spread_along_path`) and drawn by :func:`draw_postures`. A map over the start
offsets of any number of joints places each offset in a cell (:class:`OffsetMap`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray
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


@dataclass(frozen=True)
class OffsetMap:
    """A map over start offsets in any number of joints (deg): one cell per combination of each joint's offsets.

    With two joints, joint 1's offset runs along x and joint 2's along y, in degrees.
    With more, joints 1, 3, ... share x and joints 2, 4, ... share y as nested grids:
    along each axis, the first of its joints changes fastest, within blocks of the
    next, and the cells are counted from 0. Draw a map with ``imshow(grid,
    origin="lower", extent=offset_map.extent)``, then :meth:`decorate`.
    """

    values: tuple[NDArray[np.float64], ...]  # each joint's offsets (deg), sorted

    @classmethod
    def of(cls, offsets: ArrayLike) -> OffsetMap:
        """The map of the offsets ``offsets`` (deg), one row per start: every joint's values among them."""
        rounded = np.round(np.asarray(offsets, dtype=np.float64), 6)
        return cls(tuple(np.unique(rounded[:, k]) for k in range(rounded.shape[1])))

    @property
    def nested(self) -> bool:
        """Whether more than two joints share the axes."""
        return len(self.values) > 2

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """The map's extent for ``imshow``: in degrees with two joints, else in cells."""
        if not self.nested:
            (xs, ys), half = self.values, [_half_step(v) for v in self.values]
            return (xs[0] - half[0], xs[-1] + half[0], ys[0] - half[1], ys[-1] + half[1])
        return (-0.5, self._size(0) - 0.5, -0.5, self._size(1) - 0.5)

    @property
    def zero(self) -> tuple[float, float]:
        """Where the zero offset, the demonstrated start, lies in the map's coordinates."""
        if not self.nested:
            return 0.0, 0.0
        row, column = self.cell([0.0] * len(self.values))
        return float(column), float(row)

    def labels(self) -> tuple[str, str]:
        """The x and y axes' labels."""
        if not self.nested:
            return "joint 1 offset (deg)", "joint 2 offset (deg)"
        x, y = (", in blocks of ".join(f"joint {j + 1}" for j in self._joints(axis)) for axis in (0, 1))
        return f"{x} offsets (deg)", f"{y} offsets (deg)"

    def cell(self, offset: Sequence[float]) -> tuple[int, int]:
        """The row and the column of the cell of ``offset`` (deg), as a grid's indices."""
        rounded = np.round(np.asarray(offset, dtype=np.float64), 6)
        index = [int(np.flatnonzero(values == value)[0]) for values, value in zip(self.values, rounded, strict=True)]
        if not self.nested:
            return index[1], index[0]
        return self._position(1, index), self._position(0, index)

    def grid(self, cells: Mapping[tuple[float, ...], float]) -> NDArray[np.float64]:
        """The map's values, rows along y: ``cells`` maps an offset to its value; NaN where none is given."""
        data = np.full((self._size(1), self._size(0)), np.nan)
        for offset, value in cells.items():
            data[self.cell(offset)] = value
        return data

    def decorate(self, ax: Axes, fontsize: float = 7.0) -> None:
        """Label a nested map's cells by the fastest joint's offset, and divide its blocks; nothing with two joints."""
        if not self.nested:
            return
        for axis in (0, 1):
            fastest = self.values[self._joints(axis)[0]]
            size = self._size(axis)
            ticks, labels = np.arange(size), [f"{fastest[i % len(fastest)]:+g}" for i in range(size)]
            if axis == 0:
                ax.set_xticks(ticks, labels, fontsize=fontsize, rotation=90)
            else:
                ax.set_yticks(ticks, labels, fontsize=fontsize)
            for edge in np.arange(len(fastest), size, len(fastest)) - 0.5:
                line = ax.axvline if axis == 0 else ax.axhline
                line(edge, color="white", linewidth=1.5)

    def _joints(self, axis: int) -> list[int]:
        return list(range(axis, len(self.values), 2))

    def _size(self, axis: int) -> int:
        return int(np.prod([len(self.values[j]) for j in self._joints(axis)]))

    def _position(self, axis: int, index: Sequence[int]) -> int:
        position, scale = 0, 1
        for j in self._joints(axis):
            position += index[j] * scale
            scale *= len(self.values[j])
        return position


def _half_step(values: NDArray[np.float64]) -> float:
    """Half the step between ``values``, or 0.5 for a single one."""
    return float(values[1] - values[0]) / 2 if len(values) > 1 else 0.5
