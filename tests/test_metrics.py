# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the reach metrics, using reaches whose answers are known."""

import numpy as np
import pytest

from arm_esn_ctrl.metrics import arrival_index, hold_metrics, path_distance, path_progress, reach_metrics

START = np.array([0.5, 1.2])
TARGET = np.array([0.0, 1.2])


def minimum_jerk_reach(lateral: float = 0.0, duration: float = 1.0, hold: float = 0.5):
    """A minimum-jerk reach from START to TARGET, then a hold; ``lateral`` bows the path sideways."""
    times = np.arange(0.0, duration + hold, 0.001)
    tau = np.clip(times / duration, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    direction = TARGET - START
    normal = np.array([-direction[1], direction[0]])
    hand = START + np.outer(s, direction) + np.outer(lateral * np.sin(np.pi * s), normal)
    return times, hand


def test_minimum_jerk_reach_scores_as_ideal():
    times, hand = minimum_jerk_reach()

    m = reach_metrics(times, hand, TARGET)

    assert m["peak_timing"] == pytest.approx(0.5, abs=0.01)
    assert m["speed_peaks"] == 1
    assert m["path_deviation"] == pytest.approx(0.0, abs=1e-9)
    assert m["overshoot"] == 0.0
    assert m["final_error"] == pytest.approx(0.0, abs=1e-9)
    assert m["speed_profile_error"] < 0.005


def test_path_deviation_is_relative_to_reach_distance():
    times, hand = minimum_jerk_reach(lateral=0.1)

    m = reach_metrics(times, hand, TARGET)

    assert m["path_deviation"] == pytest.approx(0.1, rel=1e-3)


def test_movement_time_excludes_the_hold():
    times, hand = minimum_jerk_reach(duration=0.8, hold=1.0)

    m = reach_metrics(times, hand, TARGET)

    # Minimum-jerk speed exceeds 5 % of its peak for about 88 % of the duration.
    assert 0.8 * 0.8 < m["movement_time"] < 0.8


def test_path_distance_ignores_timing():
    line = np.column_stack([np.linspace(0.0, 1.0, 11), np.zeros(11)])
    slower_on_the_same_line = np.column_stack([np.linspace(0.0, 1.0, 101) ** 2, np.zeros(101)])

    assert path_distance(slower_on_the_same_line, line) == pytest.approx(0.0, abs=1e-12)


def test_path_distance_is_the_largest_distance_to_the_nearest_point():
    line = np.array([[0.0, 0.0], [1.0, 0.0]])
    bowed = np.array([[0.0, 0.0], [0.5, 0.2], [1.0, 0.05], [1.3, 0.0]])

    assert path_distance(bowed, line) == pytest.approx(0.3)


def hand_path(distances):
    """A hand moving along the x axis toward the target at the origin, at the given distances (one per 0.1 s)."""
    times = np.asarray(0.1 * np.arange(len(distances)), dtype=np.float64)
    return times, np.column_stack([distances, np.zeros(len(distances))])


def test_path_progress_is_the_fraction_of_the_path_up_to_the_nearest_point():
    # An L-shaped path, 3 long, that rests at its start and at its end.
    path = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 1.0]])
    points = np.array([[0.0, 0.0], [0.5, 0.3], [2.4, 0.5], [2.0, 1.0], [5.0, 5.0], [-1.0, 0.0]])

    progress = path_progress(points, path)

    assert progress == pytest.approx([0.0, 0.5 / 3, 2.5 / 3, 1.0, 1.0, 0.0])


def test_arrival_is_the_first_sample_within_the_radius():
    _, hand = hand_path([0.5, 0.3, 0.03, 0.02, 0.01])

    assert arrival_index(hand, [0.0, 0.0], radius=0.02) == 3
    assert arrival_index(hand, [0.0, 0.0], radius=0.001) is None


def test_hold_succeeds_when_the_hand_arrives_and_stays():
    times, hand = hand_path([0.3, 0.1, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0, 0.0])

    m = hold_metrics(times, hand, [0.0, 0.0], radius=0.02, hold=0.4)

    assert m["arrived"] and m["success"] and not m["left_goal"]
    assert m["arrival_time_s"] == pytest.approx(0.2)
    assert m["hold_error_m"] == pytest.approx(0.001)  # at t_a + 0.4 s = 0.6 s
    assert m["hold_observed_s"] == pytest.approx(0.4)


def test_hold_fails_when_the_hand_leaves_the_goal_within_the_window():
    times, hand = hand_path([0.3, 0.02, 0.01, 0.03, 0.01, 0.0, 0.0])

    m = hold_metrics(times, hand, [0.0, 0.0], radius=0.02, hold=0.3)

    assert m["arrived"] and m["left_goal"] and not m["success"]


def test_leaving_after_the_window_does_not_count():
    times, hand = hand_path([0.3, 0.02, 0.01, 0.01, 0.05])

    m = hold_metrics(times, hand, [0.0, 0.0], radius=0.02, hold=0.2)

    assert m["success"]
    assert m["hold_error_m"] == pytest.approx(0.01)


def test_hold_fails_when_the_hand_never_arrives():
    times, hand = hand_path([0.3, 0.2, 0.1, 0.05])

    m = hold_metrics(times, hand, [0.0, 0.0], radius=0.02, hold=0.2)

    assert not m["arrived"] and not m["success"]
    assert np.isnan(m["hold_error_m"]) and np.isnan(m["arrival_time_s"])
    assert m["hold_observed_s"] == 0.0


def test_a_late_arrival_is_judged_on_the_part_of_the_window_that_fits():
    times, hand = hand_path([0.3, 0.2, 0.1, 0.02, 0.01])

    m = hold_metrics(times, hand, [0.0, 0.0], radius=0.02, hold=1.0)

    assert m["success"]
    assert m["hold_observed_s"] == pytest.approx(0.1)
    assert m["hold_error_m"] == pytest.approx(0.01)  # at the end of the trajectory
