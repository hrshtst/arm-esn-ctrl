# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""The postures of an arm of any number of joints: its joint limits, grids over them, and their CSV columns.

The experiments read their arm from the take, so the same code serves the two-link arm
of reports 005 to 020, whose joints turn from -180 to 180 deg, and the four-link arm of
skelarm's ``examples/reach_four_dof_robot.toml``, whose limits are far narrower:

- :func:`joint_limits_deg`: each joint's limits;
- :func:`map_grid`: each joint's angles from its lower limit to its upper one, both
  included, as a map over the postures needs them;
- :func:`start_grid`: the same for start postures, where a joint that turns a whole
  turn leaves out its upper limit, its lower one's posture;
- :func:`joint_columns`, :func:`joint_values`, and :func:`format_angles`: one value
  per joint in a CSV row (``q1_deg``, ``q2_deg``, ...) and in a printed line.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from numpy.typing import NDArray
    from skelarm import Skeleton

WHOLE_TURN_DEG = 360.0


def joint_limits_deg(skeleton: Skeleton) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Each joint's lower and upper limits (deg), rounded to 1e-9 deg: the skeleton keeps them in radians."""
    lower = [round(math.degrees(link.prop.qmin), 9) for link in skeleton.links[1:]]
    upper = [round(math.degrees(link.prop.qmax), 9) for link in skeleton.links[1:]]
    return np.array(lower), np.array(upper)


def map_grid(skeleton: Skeleton, step_deg: float) -> list[NDArray[np.float64]]:
    """Each joint's angles (deg) from its lower limit to its upper one in steps of ``step_deg``, both included.

    The step must divide every joint's range, so that the grid reaches each upper limit.
    """
    grids = []
    for low, high in zip(*joint_limits_deg(skeleton), strict=True):
        steps = (high - low) / step_deg
        if abs(steps - round(steps)) > 1e-9:
            msg = f"a grid every {step_deg:g} deg misses the upper limit of a joint from {low:g} to {high:g} deg"
            raise ValueError(msg)
        grids.append(np.arange(low, high + 1e-9, step_deg))
    return grids


def start_grid(skeleton: Skeleton, step_deg: float) -> list[NDArray[np.float64]]:
    """Each joint's angles (deg) from its lower limit in steps of ``step_deg``, up to its upper limit.

    A joint that turns a whole turn, as from -180 to 180 deg, leaves out its upper limit:
    there, the arm has the posture of its lower one.
    """
    grids = []
    for low, high in zip(*joint_limits_deg(skeleton), strict=True):
        end = high - step_deg / 2 if high - low >= WHOLE_TURN_DEG - 1e-9 else high + 1e-9
        grids.append(np.arange(low, end, step_deg))
    return grids


def joint_columns(template: str, values: Iterable[float]) -> dict[str, float]:
    """One column per joint, named by ``template`` with the joint's number from 1, such as ``"q{j}_deg"``."""
    return {template.format(j=j): float(value) for j, value in enumerate(values, start=1)}


def joint_values(row: Mapping[str, Any], template: str) -> NDArray[np.float64]:
    """The values of the columns :func:`joint_columns` names by ``template`` in ``row``, as many as it has."""
    values: list[float] = []
    while (name := template.format(j=len(values) + 1)) in row:
        values.append(float(row[name]))
    if not values:
        msg = f"the row has no column {template.format(j=1)}"
        raise KeyError(msg)
    return np.array(values)


def format_angles(values: Iterable[float], digits: int = 2) -> str:
    """Joint angles in parentheses, such as ``(30.00, 45.00)``."""
    return "(" + ", ".join(f"{value:.{digits}f}" for value in values) + ")"
