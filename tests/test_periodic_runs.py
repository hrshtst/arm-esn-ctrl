# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Runs of a periodic task: what they are measured against, how its take is replayed, and where its pushes go."""

import sys
import tomllib

import numpy as np
import pytest
from skelarm import Skeleton

from arm_esn_ctrl.autonomous import Setup, Start, cycle_run_metrics, task_cycle
from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.disturbances import make_disturbance
from arm_esn_ctrl.metrics import taught_loop, taught_path
from arm_esn_ctrl.storage import REPO_ROOT
from arm_esn_ctrl.tracking import ReplaySource

# The runners import each other as modules of their own directory, as they run.
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from robot_esn import progress_source, push_aim

ELLIPSE = REPO_ROOT / "experiments/demonstrations/ellipse_scripted_v2_no_limits.toml"
EVALUATION = {"duration": 40.0, "path_tolerance": 0.02, "cycle_periods": 5, "cycle_turns": [4.0, 6.0]}


def ellipse_setup(loops=3.5, pause=1.0):
    """The setup of a periodic take: still for ``pause`` seconds, then ``loops`` times around the ellipse, 4 s a loop.

    The take is sampled every 10 ms, the ESN's period, with joint 2 positive, and the
    runs last 40 s.
    """
    with ELLIPSE.open("rb") as f:
        task = tomllib.load(f)["task"]
    skeleton = Skeleton.from_toml(ELLIPSE)
    times = np.arange(0.0, pause + 4.0 * loops + 0.005, 0.01)
    angle = 2 * np.pi * np.clip(times - pause, 0.0, None) / 4.0
    hand = np.array(task["center"]) + np.column_stack([task["a"] * np.cos(angle), task["b"] * np.sin(angle)])
    # The two-link arm's inverse kinematics, joint 2 positive.
    upper, forearm = 0.5, 0.4
    q2 = np.arccos((np.sum(hand**2, axis=1) - upper**2 - forearm**2) / (2 * upper * forearm))
    q1 = np.arctan2(hand[:, 1], hand[:, 0]) - np.arctan2(forearm * np.sin(q2), upper + forearm * np.cos(q2))
    q = np.column_stack([q1, q2])
    return Setup(
        demos={"demo_00": q},
        demo_hands={"demo_00": endpoint_positions(skeleton, q)},
        starts=[Start("demo_00", q[0], "demonstrated")],
        demonstrator_logs=[],
        q_refs=None,
        hand_refs=None,
        skeleton=skeleton,
        task=task,
        target=np.full(2, np.nan),
        radius=float("nan"),
        hold=float("nan"),
        times=0.01 * np.arange(4001, dtype=np.float64),
    )


def test_a_periodic_task_is_measured_against_its_takes_taught_path_period_and_loop():
    setup = ellipse_setup()
    q, hand = setup.demos["demo_00"], setup.demo_hands["demo_00"]

    cycle = task_cycle(setup, EVALUATION)

    assert cycle is not None
    assert cycle.center == pytest.approx([-0.1, 0.55])
    assert cycle.path == pytest.approx(taught_path(hand, cycle.center))
    assert cycle.path == pytest.approx(hand[500:1301])  # from 5 s, the end of the first loop, to 13 s
    assert cycle.period == pytest.approx(4.0, abs=1e-3)
    assert cycle.loop == taught_loop(q, hand, cycle.center)
    assert (cycle.tolerance, cycle.periods, cycle.turns) == (0.02, 5, (4.0, 6.0))
    assert cycle.after == 2.0  # the default after_window
    longer = task_cycle(setup, EVALUATION | {"after_window": 1.5})
    assert longer is not None and longer.after == 1.5


def test_a_reach_has_no_cycle():
    setup = ellipse_setup()
    reach = Setup(**{**setup.__dict__, "task": {"type": "reaching"}})

    assert task_cycle(reach, EVALUATION) is None


def test_a_periodic_run_cycles_or_not_and_has_no_arrival_or_hold():
    setup = ellipse_setup()
    cycle = task_cycle(setup, EVALUATION)
    assert cycle is not None
    # The hand going around the curve from its start, at the taught pace.
    angle = 2 * np.pi * setup.times / 4.0
    hand = cycle.center + np.column_stack([0.2 * np.cos(angle), 0.12 * np.sin(angle)])

    m = cycle_run_metrics(hand, setup, cycle)

    assert m["cycles"]
    assert m["period_ratio"] == pytest.approx(1.0, abs=1e-3)
    assert (m["arrived"], m["left_goal"], m["success"]) == ("", "", "")
    assert np.isnan(m["hold_observed_s"])
    assert not cycle_run_metrics(np.tile(hand[0], (len(hand), 1)), setup, cycle)["cycles"]  # at rest


def test_the_replays_of_a_periodic_take_go_around_its_taught_loop_without_end():
    # Still for 0.5 s, less than the progress replay's window of 1 s, within which it looks for where the arm is.
    setup = ellipse_setup(pause=0.5)
    q = setup.demos["demo_00"]
    cycle = task_cycle(setup, EVALUATION)
    assert cycle is not None
    first, end = cycle.loop
    replay = ReplaySource(q, cycle.loop)
    progress = progress_source({"lead_deg": 2.5, "window": 1.0}, q, 0.01, cycle.loop)
    progress.reset(q[0])

    steps = range(-1, 4000)
    given = np.array([replay.next_posture(k, q[0]) for k in steps])
    arm = q[0]
    for k in steps:  # an arm that keeps up with the progress replay
        posture = progress.next_posture(k, arm)
        assert posture == pytest.approx(given[k + 1])
        arm = posture

    # The posture for each step from the task's start: the take until its loop's end, then the loop again.
    assert given[:end] == pytest.approx(q[:end])
    assert given[end : 2 * end - first] == pytest.approx(q[first:end])
    assert given[2 * end - first : 3 * end - 2 * first] == pytest.approx(q[first:end])
    assert progress.loop == cycle.loop


def test_a_periodic_tasks_push_is_turned_from_the_direction_toward_the_curves_center():
    setup = ellipse_setup()
    cycle = task_cycle(setup, EVALUATION)
    start_hand = setup.demo_hands["demo_00"][0]  # (0.1, 0.55) m: the center is straight to the left
    push = {"type": "push", "force": 3.0, "onset": 8.0, "duration": 0.5, "angle_deg": 10.0}

    aim = push_aim(setup, cycle)
    disturbance = make_disturbance(push, start_hand, aim)
    assert disturbance is not None
    force = disturbance(8.2, setup.skeleton)

    assert aim == pytest.approx([-0.1, 0.55])
    # 3 N toward the center, turned 10 deg counterclockwise: to the left and a little down.
    assert force == pytest.approx(3.0 * np.array([-np.cos(np.radians(10.0)), -np.sin(np.radians(10.0))]), abs=1e-3)


def test_a_reachs_push_is_turned_from_the_direction_toward_its_target():
    setup = ellipse_setup()
    reach = Setup(**{**setup.__dict__, "task": {"type": "reaching"}, "target": np.array([-0.6, 0.4])})

    assert push_aim(reach, None) == pytest.approx([-0.6, 0.4])
