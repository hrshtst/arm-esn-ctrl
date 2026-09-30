# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Reaching demonstrations: generate them with skelarm and load them as joint angles.

A demonstration is a skelarm state log (``*.sklog.npz``). It is either

- **scripted**: simulated with one of skelarm's reaching controllers, from each
  start posture listed in a configuration file (:func:`simulate_reaches`); or
- **taught**: recorded by dragging the arm tip with the mouse in skelarm's
  ``tools/trajectory_recorder.py``.

Both kinds replay in skelarm's ``tools/player.py`` and load with
:func:`load_joint_angles`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skelarm import Skeleton, StateLog, compute_forward_kinematics, run_scenario, scenario_from_config

# The configuration tables that make up a skelarm scenario (see skelarm's
# "Run Controlled Scenarios" guide). [initial] is set per start posture.
SCENARIO_TABLES = ("skeleton", "task", "simulator", "controller")


def simulate_reaches(config: dict[str, Any]) -> list[StateLog]:
    """Run the configured reaching controller once from every start posture.

    ``config`` holds the skelarm scenario tables plus a ``[demonstrations]``
    table whose ``start_q`` lists the start postures (joint angles in degrees,
    as in skelarm's ``[initial]`` table).
    """
    logs = []
    for start_q in config["demonstrations"]["start_q"]:
        scenario_config = {name: config[name] for name in SCENARIO_TABLES}
        scenario_config["initial"] = {"q": start_q}
        logs.append(run_scenario(scenario_from_config(scenario_config)))
    return logs


def load_joint_angles(path: str | Path, dt: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load the joint angles of a demonstration, resampled at a fixed period ``dt``.

    Returns the times (starting at zero) and the joint angles in radians, shaped
    ``(n_samples, n_joints)``. Taught demonstrations are not sampled evenly, so
    the angles are linearly interpolated onto the new time grid.
    """
    log = StateLog.load(path)
    times = log.times - log.times[0]
    q = log.channel("q").reshape(len(times), -1)
    new_times = np.arange(0.0, times[-1] + 0.5 * dt, dt, dtype=np.float64)
    new_q = np.column_stack([np.interp(new_times, times, q[:, j]) for j in range(q.shape[1])])
    return new_times, new_q


def endpoint_positions(skeleton: Skeleton, q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the endpoint (hand) position ``(x, y)`` for every row of joint angles ``q``."""
    positions = np.empty((len(q), 2))
    for i, qi in enumerate(q):
        skeleton.q = qi
        compute_forward_kinematics(skeleton)
        tip = skeleton.links[-1]
        positions[i] = tip.xe, tip.ye
    return positions
