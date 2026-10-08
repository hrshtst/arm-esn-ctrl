# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Autonomous runs of a trained ESN, compared with the demonstrator (Stage 1).

An experiment configuration names the training demonstrations (``[demonstrations]``)
and the start postures to run from (``[evaluation]``). :func:`load_setup` loads the
demonstrations and simulates the demonstrator's own reach from every start posture,
once. :func:`run_autonomously` then runs a trained ESN from the same postures, and
:func:`run_metrics` measures each run's reach against the taught motion and the
demonstrator's, and its hold at the target (see :mod:`arm_esn_ctrl.metrics` for the
two phases).

A demonstration taught by hand has no demonstrator to simulate: its run's
configuration has no ``[controller]``. The setup then holds no demonstrator's
reaches, and the runs are measured against the taught motion and the target only.
"""

from __future__ import annotations

import copy
import itertools
import tomllib
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skelarm import Skeleton, StateLog, Task

from arm_esn_ctrl.demonstrations import endpoint_positions, resample_joint_angles, simulate_reaches
from arm_esn_ctrl.esn import ReachingEsn
from arm_esn_ctrl.metrics import (
    arrival_index,
    count_speed_peaks,
    distances_to_path,
    hold_metrics,
    jitter,
    onset_index,
    path_distance,
    path_rmse,
)
from arm_esn_ctrl.storage import resolve_run_path
from arm_esn_ctrl.tracking import nearest_demonstration

# The keys allowed in an [evaluation] table.
EVALUATION_KEYS = {"duration", "hold", "start_offsets_deg", "start_grid", "extra_starts", "effort_window"}
# A demonstrator's reach this close to the training path, on average, has no course of its own to
# compare with: from a start along the demonstrated path, the ratio would divide by almost nothing (rad).
# The same holds for a start this close to it, which has no offset to retain.
_ON_TRAINING_PATH = float(np.radians(0.25))
# The jitter is the deviation from a moving average over this long (s); it is measured from this
# long after the start, past a first step that jumps.
_JITTER_WINDOW = 0.05


@dataclass(frozen=True)
class Start:
    """A start posture of an autonomous run."""

    origin: str  # where it comes from: "demo_07", "demo_07 +3,-3 deg", or "<group> <index>"
    q: NDArray[np.float64]  # joint angles (rad)
    group: str  # "demonstrated", "offset", or the name of a group of extra start postures

    @property
    def demonstrated(self) -> bool:
        """Whether a training demonstration starts exactly here."""
        return self.group == "demonstrated"


@dataclass(frozen=True)
class Reference:
    """The metrics that compare a run with its reference, and how to name them.

    The reference is the demonstrator's reach from the same start posture, or the
    taught motion when there is no demonstrator (a demonstration taught by hand).
    """

    name: str  # what the runs are compared with, for labels
    path_distance: str  # the metric of the hand path's distance from the reference
    joint_error: str  # the metric of the joint error from the reference, at equal times
    arrival_delay: str  # the metric of the arrival time after the reference's
    course: str  # the metric of whether the run returns onto the training path
    course_label: str  # how to read that metric


DEMONSTRATOR = Reference(
    "demonstrator",
    "reach_path_distance_m",
    "reach_joint_error_deg",
    "arrival_delay_s",
    "training_path_ratio",
    "Training path ratio: 0 returns onto the demonstration, 1 reaches as the demonstrator",
)
TAUGHT_MOTION = Reference(
    "taught motion",
    "taught_path_distance_m",
    "taught_joint_error_deg",
    "taught_arrival_delay_s",
    "offset_retained",
    "Offset retained: 0 returns onto the taught motion, 1 keeps the start's offset",
)


@dataclass(frozen=True)
class Setup:
    """What every autonomous run is compared with, loaded once per experiment."""

    demos: dict[str, NDArray[np.float64]]  # training demonstrations: joint angles at the ESN's period
    demo_hands: dict[str, NDArray[np.float64]]  # the same demonstrations: hand positions
    starts: list[Start]
    demonstrator_logs: list[StateLog]  # the demonstrator's reach from each start posture (none without one)
    q_refs: list[NDArray[np.float64]] | None  # the same reaches: joint angles at the ESN's period
    hand_refs: list[NDArray[np.float64]] | None  # the same reaches: hand positions
    skeleton: Skeleton
    task: dict[str, Any]  # skelarm's [task] table of the demonstrations
    target: NDArray[np.float64]
    radius: float  # goal radius: the target tolerance of the demonstrations' task (m)
    hold: float  # hold duration T_h after arrival (s)
    times: NDArray[np.float64]  # time of each sample of an autonomous run, from 0

    @property
    def reference(self) -> Reference:
        """What the runs are compared with: the demonstrator, or the taught motion without one."""
        return TAUGHT_MOTION if self.q_refs is None else DEMONSTRATOR


@dataclass(frozen=True)
class Run:
    """A run from one start posture, sampled at the ESN's period, and the demonstrator's reach from it.

    In Stage 1, the run is the ESN's autonomous trajectory; in Stage 2, the robot arm's.
    Without a demonstrator, as for a demonstration taught by hand, the references are None.
    """

    start: Start
    q: NDArray[np.float64]  # joint angles of the run
    q_ref: NDArray[np.float64] | None  # the demonstrator's joint angles
    hand: NDArray[np.float64]  # hand positions of the run
    hand_ref: NDArray[np.float64] | None  # the demonstrator's hand positions


def load_setup(config: dict[str, Any]) -> Setup:
    """Load the demonstrations and simulate the demonstrator from every start posture.

    Demonstrations whose configuration has no ``[controller]``, as one taught by
    hand, have no demonstrator: nothing is simulated.
    """
    dt = config["esn"]["dt"]
    evaluation = config["evaluation"]
    demo_dir = resolve_run_path(config["demonstrations"]["run"])
    with (demo_dir / "config.toml").open("rb") as f:
        demo_config = tomllib.load(f)
    logs = {name.split(".")[0]: StateLog.load(demo_dir / name) for name in config["demonstrations"]["train"]}
    demos = {name: resample_joint_angles(log, dt)[1] for name, log in logs.items()}
    skeleton = next(iter(logs.values())).build_skeleton()

    starts = start_postures(evaluation, demos)
    demonstrator_logs: list[StateLog] = []
    q_refs = hand_refs = None
    if "controller" in demo_config:
        demonstrator_logs = simulate_reaches(demonstrator_config(demo_config, starts, evaluation["duration"]))
        q_refs = [resample_joint_angles(log, dt)[1] for log in demonstrator_logs]
        hand_refs = [endpoint_positions(skeleton, q) for q in q_refs]
    n_steps = round(evaluation["duration"] / dt)
    task = Task.from_dict(demo_config["task"])
    if task.tolerance is None:
        msg = "the demonstrations' [task] target needs a tolerance, which is the goal radius"
        raise ValueError(msg)
    return Setup(
        demos=demos,
        demo_hands={name: endpoint_positions(skeleton, q) for name, q in demos.items()},
        starts=starts,
        demonstrator_logs=demonstrator_logs,
        q_refs=q_refs,
        hand_refs=hand_refs,
        skeleton=skeleton,
        task=demo_config["task"],
        target=task.require_target(),
        radius=task.tolerance,
        hold=evaluation["hold"],
        times=dt * np.arange(n_steps + 1),
    )


def start_postures(evaluation: dict[str, Any], demos: dict[str, NDArray[np.float64]]) -> list[Start]:
    """The start postures of the autonomous runs.

    First, each demonstration's start plus every offset in ``start_offsets_deg``
    and in the ``start_grid`` table (a zero offset is the demonstrated start
    itself). The grid offsets each joint from ``-span_deg`` to ``span_deg`` in steps
    of ``step_deg``, in every combination. Then the extra start postures in the
    ``extra_starts`` table, which maps a group name to a list of joint angles in
    degrees, such as ``between = [[40.5, 71.4], ...]``.
    """
    unknown = sorted(set(evaluation) - EVALUATION_KEYS)
    if unknown:
        msg = f"unknown keys in [evaluation]: {', '.join(unknown)}"
        raise ValueError(msg)
    offsets = [tuple(offset) for offset in evaluation.get("start_offsets_deg", [])]
    if "start_grid" in evaluation:
        grid = evaluation["start_grid"]
        span, step = grid["span_deg"], grid["step_deg"]
        values = [round(value, 9) for value in np.arange(-span, span + step / 2, step)]
        n_joints = len(next(iter(demos.values()))[0])
        offsets += list(itertools.product(values, repeat=n_joints))
    offsets = list(dict.fromkeys(offsets))  # each offset once, in the order given
    starts = []
    for name, q in demos.items():
        for offset in offsets:
            if any(offset):
                starts.append(Start(f"{name} {offset[0]:+g},{offset[1]:+g} deg", q[0] + np.radians(offset), "offset"))
            else:
                starts.append(Start(name, q[0] + np.radians(offset), "demonstrated"))
    for group, postures in evaluation.get("extra_starts", {}).items():
        if group in ("demonstrated", "offset"):
            msg = f"[evaluation.extra_starts] cannot use the reserved group name {group!r}"
            raise ValueError(msg)
        starts.extend(Start(f"{group} {i}", np.radians(q_deg), group) for i, q_deg in enumerate(postures))
    if not starts:
        msg = "[evaluation] lists no start postures: give start_offsets_deg, [evaluation.start_grid], or extra_starts"
        raise ValueError(msg)
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
    for i, start in enumerate(setup.starts):
        q_esn = esn.generate(start.q, len(setup.times) - 1)
        q_ref = None if setup.q_refs is None else setup.q_refs[i]
        hand_ref = None if setup.hand_refs is None else setup.hand_refs[i]
        runs.append(Run(start, q_esn, q_ref, endpoint_positions(setup.skeleton, q_esn), hand_ref))
    return runs


def run_metrics(run: Run, setup: Setup) -> dict[str, float | bool]:
    """Measure a run's reach against the taught motion and the demonstrator's, and its hold at the target.

    The reach phase lasts until the arrival time t_a, when the hand first comes
    within the goal radius of the target; the hold window is [t_a, t_a + T_h].

    Returns
    -------
    dict[str, float | bool]
        Reach, compared with the demonstrator's reach from the same start posture
        (left out without a demonstrator):

        - ``reach_path_distance_m``: how far the hand path strays from the
          demonstrator's until t_a (the whole run if the hand never arrives),
          regardless of timing.
        - ``reach_joint_error_deg``: RMS joint-angle difference from the demonstrator,
          compared at equal times, until the demonstrator arrives.
        - ``arrival_delay_s``: t_a minus the demonstrator's arrival time (NaN if the
          hand never arrives).

        ``first_step_m``: how far the hand moves in the run's first step; a jump
        shows that the ESN snaps back to a trajectory it learned.

        Hold, which needs no reference: ``arrived``, ``arrival_time_s``,
        ``left_goal``, ``hold_error_m``, ``hold_observed_s``, and ``success``
        (see :func:`arm_esn_ctrl.metrics.hold_metrics`), and ``final_distance_m``,
        the hand's distance to the target at the end of the run.

        And whether the reach is of its own or returns to what was learned, in joint space:

        - ``training_path_distance_deg``: the mean distance of the run's joint
          angles from the nearest training demonstration's path, during the reach;
        - ``demonstrator_training_path_distance_deg``: the same for the
          demonstrator's reach from the start posture, until it arrives (left out
          without a demonstrator);
        - ``training_path_ratio``: the first over the second. Near 0, the run
          returns onto the training path; near 1, it keeps as far from it as the
          demonstrator's own reach. NaN if the demonstrator's reach keeps within
          0.25 deg of the training path on average, as from a start along it (left
          out without a demonstrator);
        - ``offset_retained``: the first over the start posture's own distance from
          the training paths. Near 0, the run returns onto the taught motion; near 1,
          it keeps its offset. NaN for a start within 0.25 deg of the training paths.

        Against the taught motion: the training demonstration whose start is nearest
        (by its timing, from t = 0):

        - ``taught_joint_error_deg``: RMS joint-angle difference, compared at equal
          times, over the shorter of the run and the demonstration;
        - ``taught_tip_error_m``: RMS distance between the hand and the taught hand,
          compared at equal times over the same span;
        - ``taught_path_distance_m``: how far the hand path strays from the taught
          hand path until t_a, regardless of timing;
        - ``taught_path_rmse_m``: RMS distance between the whole hand path and the
          whole taught hand path, regardless of timing (see
          :func:`arm_esn_ctrl.metrics.path_rmse`): waiting or moving slowly adds
          nothing;
        - ``onset_delay_s`` and ``taught_arrival_delay_s``: when the hand starts to
          move (5 mm from where it started) and when it arrives, minus the same for
          the taught motion (NaN if either never happens);
        - ``jitter_deg`` and ``taught_jitter_deg``: the RMS deviation of the joint
          angles from their 50 ms moving average, from 50 ms (past a first jump) until
          t_a, for the run and the taught motion (NaN if too short);
        - ``speed_peaks`` and ``taught_speed_peaks``: the peaks in the hand speed over
          the same span (see :func:`arm_esn_ctrl.metrics.count_speed_peaks`).
    """
    hold = hold_metrics(setup.times, run.hand, setup.target, setup.radius, setup.hold)
    arrival = arrival_index(run.hand, setup.target, setup.radius)
    reach_end = len(run.hand) if arrival is None else arrival + 1
    training = list(setup.demos.values())
    from_training = training_path_distance(run.q[:reach_end], training)
    start_distance = training_path_distance(run.q[:1], training)
    reach: dict[str, float | bool] = {}  # against the demonstrator, when there is one
    course: dict[str, float | bool] = {}
    if run.q_ref is not None and run.hand_ref is not None:
        arrival_ref = arrival_index(run.hand_ref, setup.target, setup.radius)
        reach_end_ref = len(run.q_ref) if arrival_ref is None else arrival_ref + 1
        delay = float("nan")
        if arrival is not None and arrival_ref is not None:
            delay = float(setup.times[arrival] - setup.times[arrival_ref])
        demonstrator_from_training = training_path_distance(run.q_ref[:reach_end_ref], training)
        ratio = float("nan")
        if demonstrator_from_training >= _ON_TRAINING_PATH:
            ratio = from_training / demonstrator_from_training
        reach = {
            "reach_path_distance_m": path_distance(run.hand[:reach_end], run.hand_ref),
            "reach_joint_error_deg": rms_degrees(run.q[:reach_end_ref] - run.q_ref[:reach_end_ref]),
            "arrival_delay_s": delay,
        }
        course = {
            "demonstrator_training_path_distance_deg": float(np.degrees(demonstrator_from_training)),
            "training_path_ratio": ratio,
        }
    retained = from_training / start_distance if start_distance >= _ON_TRAINING_PATH else float("nan")
    return (
        {"first_step_m": float(np.linalg.norm(run.hand[1] - run.hand[0]))}
        | reach
        | hold
        | {"final_distance_m": float(np.linalg.norm(run.hand[-1] - setup.target))}
        | {"training_path_distance_deg": float(np.degrees(from_training))}
        | course
        | {"offset_retained": retained}
        | taught_metrics(run, setup, arrival)
    )


def taught_metrics(run: Run, setup: Setup, arrival: int | None) -> dict[str, float]:
    """The metrics of a run against the taught motion (see :func:`run_metrics`), given its ``arrival`` sample."""
    name = nearest_demonstration(run.start.q, setup.demos)
    taught, taught_hand = setup.demos[name], setup.demo_hands[name]
    taught_arrival = arrival_index(taught_hand, setup.target, setup.radius)
    taught_end = len(taught_hand) if taught_arrival is None else taught_arrival + 1
    dt = float(setup.times[1] - setup.times[0])
    n = min(len(run.q), len(taught))

    def time_of(index: int | None) -> float:
        return float("nan") if index is None else index * dt

    reach_end = len(run.hand) if arrival is None else arrival + 1
    window = 2 * round(_JITTER_WINDOW / dt / 2) + 1  # an odd number of samples
    first = round(_JITTER_WINDOW / dt)
    return {
        "taught_joint_error_deg": rms_degrees(run.q[:n] - taught[:n]),
        "taught_tip_error_m": float(np.sqrt(np.mean(np.sum((run.hand[:n] - taught_hand[:n]) ** 2, axis=1)))),
        "taught_path_distance_m": path_distance(run.hand[:reach_end], taught_hand),
        "taught_path_rmse_m": path_rmse(run.hand, taught_hand),
        "onset_delay_s": time_of(onset_index(run.hand)) - time_of(onset_index(taught_hand)),
        "taught_arrival_delay_s": time_of(arrival) - time_of(taught_arrival),
        "jitter_deg": jitter_degrees(run.q[first:reach_end], window),
        "taught_jitter_deg": jitter_degrees(taught[first:taught_end], window),
        "speed_peaks": peaks(setup.times[first:reach_end], run.hand[first:reach_end]),
        "taught_speed_peaks": peaks(setup.times[first:taught_end], taught_hand[first:taught_end]),
    }


def jitter_degrees(q: NDArray[np.float64], window: int) -> float:
    """The jitter of joint angles ``q`` (rad) in degrees, or NaN if they are shorter than two windows."""
    return float(np.degrees(jitter(q, window))) if len(q) >= 2 * window else float("nan")


def peaks(times: NDArray[np.float64], hand: NDArray[np.float64]) -> float:
    """The speed peaks of a hand path, or NaN if it is too short to have any."""
    return float(count_speed_peaks(times, hand)) if len(hand) >= 3 else float("nan")


def training_path_distance(q: NDArray[np.float64], training: list[NDArray[np.float64]]) -> float:
    """The mean distance (rad) of the joint angles ``q`` from the nearest path among the ``training`` demonstrations."""
    return float(np.mean(np.min([distances_to_path(q, demo) for demo in training], axis=0)))


def rms_degrees(error: NDArray[np.float64]) -> float:
    """Root mean square of an angle error given in radians, in degrees."""
    return float(np.degrees(np.sqrt(np.mean(error**2))))
