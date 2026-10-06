# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Reaching demonstrations: generate them with skelarm and load them as joint angles.

A demonstration is a skelarm state log (``*.sklog.npz``). It is either

- **scripted**: simulated with one of skelarm's reaching controllers, from each
  start posture listed in a configuration file (:func:`simulate_reaches`); or
- **taught**: recorded by dragging the arm tip with the mouse in skelarm's
  ``tools/trajectory_recorder.py``.

Both kinds replay in skelarm's ``tools/player.py`` and load with
:func:`load_joint_angles`, which resamples them at a fixed period. A taught take is
checked with :func:`check_take` before it becomes a demonstration.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from skelarm import (
    Skeleton,
    StateLog,
    compute_forward_kinematics,
    run_scenario,
    scenario_from_config,
    simulate_controlled,
)

from arm_esn_ctrl.metrics import hold_metrics

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


def simulate_disturbed_reach(
    config: dict[str, Any],
    start_q: NDArray[np.float64],
    duration: float,
    external_force: Callable[[float, Skeleton], NDArray[np.float64]],
) -> StateLog:
    """Run the configured reaching controller from ``start_q`` (radians) under a tip force.

    Like :func:`simulate_reaches` for one start posture, but ``external_force`` (see
    :mod:`arm_esn_ctrl.disturbances`) acts on the arm's tip, recorded as ``ext_force``.
    """
    scenario_config = {name: config[name] for name in SCENARIO_TABLES}
    scenario_config["initial"] = {"q": np.degrees(start_q).tolist()}
    scenario = scenario_from_config(scenario_config)
    return simulate_controlled(
        scenario.skeleton,
        scenario.controller,
        duration=duration,
        dt=scenario.simulator.dt,
        enforce_limits=scenario.simulator.enforce_limits,
        extra={"playback": {"task": config["task"]}},
        external_force=external_force,
    )


def load_joint_angles(path: str | Path, dt: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load the joint angles of a demonstration file, resampled at a fixed period ``dt``."""
    return resample_joint_angles(StateLog.load(path), dt)


def resample_joint_angles(log: StateLog, dt: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the joint angles of a log, resampled at a fixed period ``dt``.

    Returns the times (starting at zero) and the joint angles in radians, shaped
    ``(n_samples, n_joints)``. Taught demonstrations are not sampled evenly, so
    the angles are linearly interpolated onto the new time grid.
    """
    times = log.times - log.times[0]
    q = log.channel("q").reshape(len(times), -1)
    new_times = np.arange(0.0, times[-1] + 0.5 * dt, dt, dtype=np.float64)
    new_q = np.column_stack([np.interp(new_times, times, q[:, j]) for j in range(q.shape[1])])
    return new_times, new_q


def endpoint_positions(skeleton: Skeleton, q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the endpoint (hand) position ``(x, y)`` for every row of joint angles ``q``.

    The angles are used as given, even beyond the joint limits. The ``Skeleton.q``
    setter would clamp them, but an ESN's output is not bounded by the limits, and
    clamping would misreport where it sends the hand.
    """
    positions = np.empty((len(q), 2))
    for i, qi in enumerate(q):
        for link, angle in zip(skeleton.links[1:], qi, strict=True):
            link.q = angle
        compute_forward_kinematics(skeleton)
        tip = skeleton.links[-1]
        positions[i] = tip.xe, tip.ye
    return positions


def joint_trajectory_log(
    skeleton: Skeleton, times: NDArray[np.float64], q: NDArray[np.float64], task: dict[str, Any], producer: str
) -> StateLog:
    """Make a skelarm log of a joint-angle trajectory, so that skelarm's player can replay it.

    ``task`` is a skelarm ``[task]`` table; the player draws its target.
    """
    log = StateLog(skeleton, producer=producer, extra={"playback": {"task": task}})
    dq = np.gradient(q, times, axis=0)
    for t, qi, dqi in zip(times, q, dq, strict=True):
        log.record(float(t), q=qi, dq=dqi)
    return log


def check_take(
    log: StateLog,
    skeleton: Skeleton,
    *,
    start_q: ArrayLike,
    target: ArrayLike,
    radius: float,
    hold: float,
    start_tolerance: float,
    max_tip_speed: float,
) -> list[str]:
    """Return why a take taught by hand cannot be a demonstration, or nothing if it can.

    A take can be a demonstration when

    - it was recorded with the arm ``skeleton`` (the same link properties);
    - it begins at the start posture ``start_q`` (rad), within ``start_tolerance`` in every joint;
    - the hand arrives within ``radius`` of the ``target`` and stays there for
      ``hold`` seconds, before the take ends;
    - the hand never moves faster than ``max_tip_speed`` (m/s) between two samples,
      which would be a jump rather than a reach.
    """
    problems = []
    recorded = [dataclasses.astuple(link.prop) for link in log.build_skeleton().links]
    expected = [dataclasses.astuple(link.prop) for link in skeleton.links]
    if len(recorded) != len(expected) or not np.allclose(recorded, expected):
        problems.append("the take was recorded with another arm than the configuration's [skeleton]")
        return problems
    times = log.times
    q = log.channel("q").reshape(len(times), -1)
    offset = np.degrees(np.abs(q[0] - np.asarray(start_q, dtype=np.float64)))
    if offset.max() > np.degrees(start_tolerance):
        problems.append(f"the take begins {offset.max():.3g} deg away from the start posture")
    hand = endpoint_positions(skeleton, q)
    held = hold_metrics(times, hand, target, radius, hold)
    if not held["arrived"]:
        problems.append(f"the hand never comes within {radius:g} m of the target")
    elif held["left_goal"]:
        problems.append(f"the hand leaves the goal within {hold:g} s of arriving")
    elif times[-1] - held["arrival_time_s"] < hold:  # the take must run past the whole hold window
        problems.append(
            f"the take ends {times[-1] - held['arrival_time_s']:.2f} s after the hand arrives,"
            f" before {hold:g} s of holding"
        )
    speed = np.linalg.norm(np.diff(hand, axis=0), axis=1) / np.diff(times)
    if speed.max() > max_tip_speed:
        k = int(np.argmax(speed))
        problems.append(
            f"the hand jumps at {times[k]:.2f} s, moving at {speed[k]:.1f} m/s (more than {max_tip_speed:g})"
        )
    return problems
