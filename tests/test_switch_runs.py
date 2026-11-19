# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Runs whose task changes between a reach and a periodic motion: what is measured of each part, on made-up logs."""

import math
import sys
import tomllib

import numpy as np
import pytest
from skelarm import Skeleton, StateLog

from arm_esn_ctrl.autonomous import CURVE_SIDES, Cycle, Setup, Start
from arm_esn_ctrl.demonstrations import task_path
from arm_esn_ctrl.storage import REPO_ROOT

# The runners import each other as modules of their own directory, as they run.
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from robot_esn import switch_metrics

ELLIPSE = REPO_ROOT / "experiments/demonstrations/ellipse_scripted_v2_no_limits.toml"
with ELLIPSE.open("rb") as f:
    TASK = tomllib.load(f)["task"]  # centered at (-0.1, 0.55) m, 0.2 m by 0.12 m
SKELETON = Skeleton.from_toml(ELLIPSE)
CENTER = np.array(TASK["center"])
TARGET = np.array([-0.6, 0.4])  # the reach's, 0.33 m from the ellipse
PERIOD = 4.0  # s, a loop
CURVE = task_path(TASK, CURVE_SIDES)
assert CURVE is not None
# A cycle over three periods of 4 s, of two to four turns, within 5 mm of the curve in RMS: a hand 0.33 m off the
# curve for one sample of a period's window makes that window's RMS 16 mm, above it.
CYCLE = Cycle(
    center=CENTER,
    curve=CURVE,
    period=PERIOD,
    loop=(0, 400),
    tolerance=0.005,
    periods=3,
    turns=(2.0, 4.0),
    after=2.0,
)
REACH, LOOP = 0, 1  # the tasks' indices


def setup(task, target, radius, hold):
    """A task's setup, of which the measures of a change use the arm, the target, the goal radius, and the hold."""
    q = np.zeros((2, 2))
    return Setup(
        demos={"demo_00": q},
        demo_hands={"demo_00": q},
        starts=[Start("demo_00", q[0], "demonstrated")],
        demonstrator_logs=[],
        q_refs=None,
        hand_refs=None,
        skeleton=SKELETON,
        task=task,
        target=target,
        radius=radius,
        hold=hold,
        times=0.01 * np.arange(2, dtype=np.float64),
    )


TASKS = (
    [setup({"type": "reaching"}, TARGET, 0.02, 2.0), setup(TASK, np.full(2, np.nan), math.nan, math.nan)],
    [None, CYCLE],
)


def around(t, since):
    """The hand on the ellipse at times ``t``, from its angle 0 at ``since`` on, a loop every 4 s, counterclockwise."""
    angle = 2 * np.pi * (t - since) / PERIOD
    return CENTER + np.column_stack([TASK["a"] * np.cos(angle), TASK["b"] * np.sin(angle)])


def joint_angles(hand):
    """The two-link arm's inverse kinematics (links of 0.5 m and 0.4 m), joint 2 positive."""
    upper, forearm = 0.5, 0.4
    q2 = np.arccos((np.sum(hand**2, axis=1) - upper**2 - forearm**2) / (2 * upper * forearm))
    q1 = np.arctan2(hand[:, 1], hand[:, 0]) - np.arctan2(forearm * np.sin(q2), upper + forearm * np.cos(q2))
    return np.column_stack([q1, q2])


def run_log(times, hand, q_ref=None, tau=None):
    """A run's log every 10 ms, its hand at ``times``, after a warm-up of 0.5 s with the hand held at its start.

    The reference and the torque are ``q_ref`` and ``tau`` (rad and N m, one row a sample of ``times``), or zero.
    """
    q = joint_angles(hand)
    q_ref = np.zeros_like(q) if q_ref is None else q_ref
    tau = np.zeros_like(q) if tau is None else tau
    log = StateLog(SKELETON)
    for k in range(-50, 0):
        log.record(0.01 * k, q=q[0], dq=np.zeros(2), tau=np.zeros(2), q_ref=q_ref[0])
    for k in range(len(times)):
        log.record(0.01 * k, q=q[k], dq=np.zeros(2), tau=tau[k], q_ref=q_ref[k])
    return log


def times_to(end):
    """The run's times every 10 ms from 0 to ``end`` (s)."""
    return 0.01 * np.arange(round(end / 0.01) + 1)


def test_after_a_change_to_the_reach_the_hand_arrives_and_holds_and_before_it_the_hand_cycled():
    times = times_to(30.0)
    # Around the ellipse until the change at 16 s, at its angle 0 then, held there 1.5 s more, then at the target.
    hand = np.where((times < 16.0)[:, np.newaxis], around(times, 0.0), around(np.array([16.0]), 0.0))
    hand[times >= 17.5 - 1e-9] = TARGET
    # The reference moves joint 1 at 100 deg/s from 5 s to 5.5 s, and at 30 deg/s from 16.5 s to 17 s.
    speed = 100.0 * np.clip(times - 5.0, 0.0, 0.5) + 30.0 * np.clip(times - 16.5, 0.0, 0.5)
    q_ref = np.column_stack([np.radians(speed), np.zeros_like(times)])
    # A torque of 7 N m at 17 s, within 2 s of the change, and of 9 N m at 19 s, after them.
    tau = np.ones((len(times), 2))
    tau[np.isclose(times, 17.0), 1] = -7.0
    tau[np.isclose(times, 19.0), 0] = 9.0
    schedule = [(0.0, LOOP), (16.0, REACH)]

    m = switch_metrics(run_log(times, hand, q_ref, tau), times, schedule, TASKS, 0.01, 2.0)

    assert list(m) == [
        "switch_arrived",
        "switch_arrival_time_s",
        "switch_left_goal",
        "switch_hold_error_m",
        "switch_hold_observed_s",
        "switch_success",
        "switch_final_distance_m",
        "cycles_before_switch_1",
        "switch_peak_reference_speed_dps_1",
        "switch_peak_torque_nm_1",
        "switch_arrival_time_s_1",
    ]
    assert (m["switch_arrived"], m["switch_success"], m["switch_left_goal"]) == (True, True, False)
    assert m["switch_arrival_time_s"] == pytest.approx(1.5)  # from the change
    assert m["switch_hold_observed_s"] == pytest.approx(2.0)
    assert m["switch_final_distance_m"] == pytest.approx(0.0, abs=1e-9)
    assert m["cycles_before_switch_1"] is True  # three turns over the 12 s before the change
    assert m["switch_peak_reference_speed_dps_1"] == pytest.approx(30.0)
    assert m["switch_peak_torque_nm_1"] == pytest.approx(7.0)
    assert m["switch_arrival_time_s_1"] == pytest.approx(1.5)


def test_after_a_change_to_the_ellipse_the_hand_cycles_at_the_end_and_joins_the_curve_from_the_change():
    times = times_to(40.0)
    # At the target until a second after the change at 20 s, then around the ellipse from its angle 0.
    hand = np.where((times < 21.0 - 1e-9)[:, np.newaxis], TARGET, around(times, 21.0))
    schedule = [(0.0, REACH), (20.0, LOOP)]

    m = switch_metrics(run_log(times, hand), times, schedule, TASKS, 0.01, 2.0)

    assert list(m) == [
        "switch_cycles",
        "switch_turns",
        "switch_curve_rms_m",
        "switch_joining_time_s",
        "cycles_before_switch_1",
        "switch_peak_reference_speed_dps_1",
        "switch_peak_torque_nm_1",
        "switch_joining_time_s_1",
    ]
    assert m["switch_cycles"] is True
    assert m["switch_turns"] == pytest.approx(3.0)  # over the run's last three periods
    assert m["switch_curve_rms_m"] < 1e-4
    assert m["switch_joining_time_s"] == pytest.approx(1.0)  # from the change
    assert m["cycles_before_switch_1"] == ""  # the task left is a reach
    assert m["switch_joining_time_s_1"] == pytest.approx(1.0)
    assert m["switch_peak_reference_speed_dps_1"] == 0.0


def test_back_and_forth_each_change_is_measured_until_the_next_and_the_hand_cycles_before_each_one_from_the_ellipse():
    times = times_to(60.0)
    # Around the ellipse until 16 s; at the target from 17.5 s; still there at the change to the ellipse at 30 s,
    # and around it from 31 s; at the change to the reach at 50 s, held where it is, and at the target from 52 s.
    hand = around(times, 0.0)
    hand[times >= 16.0 - 1e-9] = around(np.array([16.0]), 0.0)
    hand[times >= 17.5 - 1e-9] = TARGET
    hand[times >= 31.0 - 1e-9] = around(times[times >= 31.0 - 1e-9], 31.0)
    hand[times >= 50.0 - 1e-9] = around(np.array([50.0]), 31.0)
    hand[times >= 52.0 - 1e-9] = TARGET
    schedule = [(0.0, LOOP), (16.0, REACH), (30.0, LOOP), (50.0, REACH)]

    m = switch_metrics(run_log(times, hand), times, schedule, TASKS, 0.01, 2.0)

    assert (m["cycles_before_switch_1"], m["cycles_before_switch_2"], m["cycles_before_switch_3"]) == (True, "", True)
    assert m["switch_arrival_time_s_1"] == pytest.approx(1.5)
    assert m["switch_joining_time_s_2"] == pytest.approx(1.0)  # over the part until the next change
    assert m["switch_arrival_time_s_3"] == pytest.approx(2.0)
    assert m["switch_arrival_time_s"] == pytest.approx(2.0)
    assert m["switch_success"] is True
    assert [name for name in m if name.endswith("_3")] == [
        "cycles_before_switch_3",
        "switch_peak_reference_speed_dps_3",
        "switch_peak_torque_nm_3",
        "switch_arrival_time_s_3",
    ]


def test_a_change_whose_task_is_not_done_before_the_next_change_has_no_time_and_a_hand_still_does_not_cycle():
    times = times_to(60.0)
    # Around the ellipse until 16 s, then still there until the hand reaches the target at 31 s, after the change
    # back to the ellipse at 30 s; it never joins the curve, and is still at the target at 50 s.
    hand = around(times, 0.0)
    hand[times >= 16.0 - 1e-9] = around(np.array([16.0]), 0.0)
    hand[times >= 31.0 - 1e-9] = TARGET
    schedule = [(0.0, LOOP), (16.0, REACH), (30.0, LOOP), (50.0, REACH)]

    m = switch_metrics(run_log(times, hand), times, schedule, TASKS, 0.01, 2.0)

    assert math.isnan(m["switch_arrival_time_s_1"])  # it arrives, but only after the next change
    assert math.isnan(m["switch_joining_time_s_2"])
    assert m["cycles_before_switch_3"] is False
    assert m["switch_arrival_time_s_3"] == 0.0  # at the target already


def test_a_replay_keeps_to_its_take_so_its_switch_columns_are_all_empty():
    schedule = [(0.0, LOOP), (16.0, REACH), (36.0, LOOP)]
    times = times_to(60.0)

    m = switch_metrics(None, times, schedule, TASKS, 0.01, 2.0)

    hand = np.where((times < 16.0)[:, np.newaxis], around(times, 0.0), TARGET)
    assert list(m) == list(switch_metrics(run_log(times, hand), times, schedule, TASKS, 0.01, 2.0))
    assert set(m.values()) == {""}
