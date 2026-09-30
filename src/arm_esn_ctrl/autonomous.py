# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Autonomous runs of a trained ESN, compared with the demonstrator (Stage 1).

An experiment configuration names the training demonstrations (``[demonstrations]``)
and the start postures to run from (``[evaluation]``). :func:`load_setup` loads the
demonstrations and simulates the demonstrator's own reach from every start posture,
once. :func:`run_autonomously` then runs a trained ESN from the same postures, and
:func:`run_metrics` compares each run with the demonstrator's reach.
"""

from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skelarm import Skeleton, StateLog, Task

from arm_esn_ctrl.demonstrations import endpoint_positions, resample_joint_angles, simulate_reaches
from arm_esn_ctrl.esn import ReachingEsn
from arm_esn_ctrl.metrics import path_distance, reach_metrics
from arm_esn_ctrl.storage import storage_root


@dataclass(frozen=True)
class Start:
    """A start posture of an autonomous run."""

    origin: str  # where it comes from: "demo_07", "demo_07 +3,-3 deg", or "new"
    q: NDArray[np.float64]  # joint angles (rad)
    demonstrated: bool  # whether a training demonstration starts exactly here


@dataclass(frozen=True)
class Setup:
    """What every autonomous run is compared with, loaded once per experiment."""

    demos: dict[str, NDArray[np.float64]]  # training demonstrations: joint angles at the ESN's period
    starts: list[Start]
    demonstrator_logs: list[StateLog]  # the demonstrator's reach from each start posture
    q_refs: list[NDArray[np.float64]]  # the same reaches: joint angles at the ESN's period
    hand_refs: list[NDArray[np.float64]]  # the same reaches: hand positions
    skeleton: Skeleton
    task: dict[str, Any]  # skelarm's [task] table of the demonstrations
    target: NDArray[np.float64]
    times: NDArray[np.float64]  # time of each sample of an autonomous run, from 0


@dataclass(frozen=True)
class Run:
    """The ESN's autonomous run from one start posture, and the demonstrator's reach from it."""

    start: Start
    q_esn: NDArray[np.float64]
    q_ref: NDArray[np.float64]
    hand_esn: NDArray[np.float64]
    hand_ref: NDArray[np.float64]


def load_setup(config: dict[str, Any]) -> Setup:
    """Load the demonstrations and simulate the demonstrator from every start posture."""
    dt = config["esn"]["dt"]
    evaluation = config["evaluation"]
    demo_dir = storage_root() / config["demonstrations"]["run"]
    with (demo_dir / "config.toml").open("rb") as f:
        demo_config = tomllib.load(f)
    logs = {name.split(".")[0]: StateLog.load(demo_dir / name) for name in config["demonstrations"]["train"]}
    demos = {name: resample_joint_angles(log, dt)[1] for name, log in logs.items()}
    skeleton = next(iter(logs.values())).build_skeleton()

    starts = start_postures(evaluation, demos)
    demonstrator_logs = simulate_reaches(demonstrator_config(demo_config, starts, evaluation["duration"]))
    q_refs = [resample_joint_angles(log, dt)[1] for log in demonstrator_logs]
    n_steps = round(evaluation["duration"] / dt)
    return Setup(
        demos=demos,
        starts=starts,
        demonstrator_logs=demonstrator_logs,
        q_refs=q_refs,
        hand_refs=[endpoint_positions(skeleton, q) for q in q_refs],
        skeleton=skeleton,
        task=demo_config["task"],
        target=Task.from_dict(demo_config["task"]).require_target(),
        times=dt * np.arange(n_steps + 1),
    )


def start_postures(evaluation: dict[str, Any], demos: dict[str, NDArray[np.float64]]) -> list[Start]:
    """The start postures of the autonomous runs: offsets around each demonstration's start, then new ones."""
    starts = []
    for name, q in demos.items():
        for offset in evaluation["start_offsets_deg"]:
            demonstrated = not any(offset)
            origin = name if demonstrated else f"{name} {offset[0]:+g},{offset[1]:+g} deg"
            starts.append(Start(origin, q[0] + np.radians(offset), demonstrated))
    for q_deg in evaluation.get("extra_start_q_deg", []):
        starts.append(Start("new", np.radians(q_deg), demonstrated=False))
    return starts


def demonstrator_config(demo_config: dict[str, Any], starts: list[Start], duration: float) -> dict[str, Any]:
    """The demonstration configuration, changed to reach from ``starts`` for ``duration`` seconds."""
    reference = copy.deepcopy(demo_config)
    reference["demonstrations"]["start_q"] = [np.degrees(start.q).tolist() for start in starts]
    reference["task"]["duration"] = duration
    return reference


def run_autonomously(esn: ReachingEsn, setup: Setup) -> list[Run]:
    """Run a trained ESN from every start posture of ``setup``."""
    runs = []
    for start, q_ref, hand_ref in zip(setup.starts, setup.q_refs, setup.hand_refs, strict=True):
        q_esn = esn.generate(start.q, len(setup.times) - 1)
        runs.append(Run(start, q_esn, q_ref, endpoint_positions(setup.skeleton, q_esn), hand_ref))
    return runs


def run_metrics(run: Run, setup: Setup) -> dict[str, float]:
    """Compare an autonomous run with the demonstrator's reach from the same start posture.

    Returns
    -------
    dict[str, float]
        - ``first_step_m``: how far the hand moves in the ESN's first step; a jump
          shows that the ESN snaps back to a trajectory it learned.
        - ``joint_rms_error_deg``: RMS joint-angle difference from the demonstrator, over time.
        - ``path_distance_m``: how far the hand path strays from the demonstrator's,
          regardless of timing.
        - ``final_error_m``: the distance from the hand to the target at the end.
        - ``peak_timing`` and ``speed_profile_error``: the shape of the hand speed
          profile (see :func:`arm_esn_ctrl.metrics.reach_metrics`).
    """
    shape = reach_metrics(setup.times, run.hand_esn, setup.target)
    return {
        "first_step_m": float(np.linalg.norm(run.hand_esn[1] - run.hand_esn[0])),
        "joint_rms_error_deg": rms_degrees(run.q_esn - run.q_ref),
        "path_distance_m": path_distance(run.hand_esn, run.hand_ref),
        "final_error_m": float(np.linalg.norm(run.hand_esn[-1] - setup.target)),
        "peak_timing": shape["peak_timing"],
        "speed_profile_error": shape["speed_profile_error"],
    }


def rms_degrees(error: NDArray[np.float64]) -> float:
    """Root mean square of an angle error given in radians, in degrees."""
    return float(np.degrees(np.sqrt(np.mean(error**2))))
