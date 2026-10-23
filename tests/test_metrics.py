# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the reach metrics, using reaches whose answers are known."""

import numpy as np
import pytest

from arm_esn_ctrl.metrics import (
    arrival_index,
    count_speed_peaks,
    disturbance_following,
    disturbance_returns,
    disturbance_ringing,
    hold_metrics,
    jitter,
    join_index,
    motor_effort,
    onset_index,
    path_distance,
    path_progress,
    path_rmse,
    reach_metrics,
    route_spread,
)

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


LINE = np.array([[0.0, 0.0], [1.0, 0.0]])


def test_path_rmse_ignores_timing_and_pauses():
    """A path traversed slowly, with a long pause at the start, is the same path."""
    paused = np.zeros((300, 2))  # 300 samples standing still at the start
    slow = np.column_stack([np.linspace(0.0, 1.0, 101) ** 3, np.zeros(101)])

    assert path_rmse(np.vstack([paused, slow]), LINE) == pytest.approx(0.0, abs=1e-12)


def test_path_rmse_of_a_parallel_path_is_their_distance():
    shifted = LINE + np.array([0.0, 0.02])

    assert path_rmse(shifted, LINE) == pytest.approx(0.02)


def test_path_rmse_counts_the_part_of_the_reference_a_path_skips_by_its_length():
    """Stopping halfway: the skipped half, 0.5 m long, lies 0 to 0.5 m from the path, over 1.5 m of both paths."""
    half = np.array([[0.0, 0.0], [0.5, 0.0]])

    assert path_rmse(half, LINE) == pytest.approx(np.sqrt((0.5**3 / 3) / 1.5), rel=1e-2)


def test_path_rmse_of_an_arm_that_never_moves_is_how_far_the_reference_goes():
    still = np.zeros((50, 2))

    assert path_rmse(still, LINE) == pytest.approx(np.sqrt(1 / 3), rel=1e-2)


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


def test_the_onset_is_the_first_sample_the_hand_is_away_from_its_start():
    rest = np.zeros((20, 2))
    moving = np.column_stack([0.001 * np.arange(1, 30), np.zeros(29)])  # 1 mm per sample
    hand = np.vstack([rest, moving]).astype(np.float64)

    assert onset_index(hand, 0.005) == 20 + 5  # first beyond 5 mm: 6 mm, the sixth moving sample
    assert onset_index(rest, 0.005) is None


def test_jitter_is_zero_for_a_ramp_and_measures_alternating_noise():
    ramp = np.column_stack([np.linspace(0.0, 1.0, 50), np.linspace(1.0, 0.0, 50)])
    noise = 0.01 * (-1.0) ** np.arange(50)

    assert jitter(ramp, 5) == pytest.approx(0.0, abs=1e-12)
    # A 5-sample average of alternating +-e is +-e/5, leaving +-0.8 e in the one noisy joint.
    assert jitter(ramp + np.column_stack([noise, np.zeros(50)]), 5) == pytest.approx(0.008)


def test_speed_peaks_count_the_bells_of_the_speed_profile():
    times = np.linspace(0.0, 2.0, 401)
    one = np.column_stack([np.cumsum(np.sin(np.pi * times / 2.0) ** 2), np.zeros_like(times)])
    two = np.column_stack([np.cumsum(np.sin(np.pi * times) ** 2), np.zeros_like(times)])

    assert count_speed_peaks(times, one) == 1
    assert count_speed_peaks(times, two) == 2


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


def test_runs_on_one_route_have_no_spread_whatever_their_timing():
    route = np.column_stack([np.linspace(0.0, 1.0, 101), np.zeros(101)])
    late = np.column_stack([np.clip((np.arange(101) - 30) / 70, 0.0, 1.0), np.zeros(101)])  # waits, then hurries
    spread = route_spread([route, late, route[::-1]])  # the last one runs it backward
    np.testing.assert_allclose(spread, 0.0, atol=1e-12)


def test_a_run_that_gathers_onto_the_others_route_loses_its_spread():
    x = np.linspace(0.0, 1.0, 101)
    on_route = [np.column_stack([x, np.zeros(101)]) for _ in range(4)]
    joining = np.column_stack([x, 0.5 * np.clip(1.0 - x / 0.4, 0.0, None)])  # 0.5 off, on the route from x = 0.4
    spread = route_spread([*on_route, joining])
    assert spread[4, 0] == pytest.approx(0.5)
    assert spread[4, 40:].max() == pytest.approx(0.0, abs=1e-12)
    assert spread[:4].max() == pytest.approx(0.0, abs=1e-12)  # the median ignores the one that is off
    assert join_index(spread[4], 0.12) == 31  # 0.5 (1 - x / 0.4) < 0.12 from x > 0.304
    assert join_index(np.array([0.5, 0.0, 0.5, 0.0]), 0.1) == 3  # it must stay
    assert join_index(np.array([0.5, 0.5]), 0.1) is None


def test_each_disturbance_is_measured_until_the_next_one_starts():
    times = np.arange(0.0, 10.0, 0.5)
    distances = np.zeros_like(times)
    distances[(times >= 2.0) & (times < 3.5)] = [0.01, 0.03, 0.02]  # the first push sends the hand 30 mm away
    distances[times >= 8.5] = 0.04  # the second one, at 8 s, leaves it away

    (first, second) = disturbance_returns(times, distances, [(2.0, 2.5), (8.0, 8.5)], 0.005)

    assert first == (pytest.approx(0.03), pytest.approx(1.0))  # back from 3.5 s, 1 s after the push ends
    assert second[0] == pytest.approx(0.04)
    assert np.isnan(second[1])  # not back by the end of the run


def test_a_disturbance_the_hand_never_strays_from_returns_at_once():
    times = np.arange(0.0, 6.0, 0.5)
    distances = np.where((times >= 1.0) & (times < 2.0), 0.002, 0.0)  # within the tolerance throughout

    ((farthest, back),) = disturbance_returns(times, distances, [(1.0, 2.0)], 0.005)

    assert farthest == pytest.approx(0.002)
    assert back == 0.0


def test_a_disturbance_that_lasts_past_the_run_has_no_return():
    times = np.arange(0.0, 3.0, 0.5)

    ((farthest, back),) = disturbance_returns(times, np.zeros_like(times), [(2.0, 4.0)], 0.005)

    assert farthest == 0.0
    assert np.isnan(back)


def ringing_after(times, end, amplitude=0.02, frequency=3.2, damping=0.1):
    """A damped oscillation at ``frequency`` (Hz) along x that starts at ``end``, as a tracker rings."""
    t = np.clip(times - end, 0.0, None)
    omega = 2 * np.pi * frequency
    return np.where(times >= end, amplitude * np.exp(-damping * omega * t) * np.sin(omega * t), 0.0)


def test_slow_motion_does_not_ring():
    times = np.arange(0.0, 20.0, 0.01)
    points = np.column_stack([0.3 * (1 - np.cos(2 * np.pi * 0.2 * times)), 0.1 * np.sin(2 * np.pi * 0.1 * times)])

    (ringing,) = disturbance_ringing(times, points, [(8.0, 8.2)], cutoff_hz=1.5, window=2.0)

    assert ringing < 1e-4  # under 0.1 mm, against a motion 0.6 m wide


def test_ringing_is_the_rms_of_the_oscillation_after_the_disturbance():
    times = np.arange(0.0, 20.0, 0.01)
    slow = 0.3 * (1 - np.cos(2 * np.pi * 0.2 * times))
    oscillation = ringing_after(times, 8.2)
    points = np.column_stack([slow + oscillation, np.zeros_like(times)])

    (ringing,) = disturbance_ringing(times, points, [(8.0, 8.2)], cutoff_hz=1.5, window=2.0)

    after = (times >= 8.2) & (times < 10.2)
    expected = np.sqrt(np.mean(oscillation[after] ** 2))
    assert ringing == pytest.approx(expected, rel=0.15)  # the filter passes about 90% of a decaying 3.2 Hz


def test_ringing_is_measured_until_the_next_disturbance_starts():
    times = np.arange(0.0, 20.0, 0.01)
    points = np.column_stack([ringing_after(times, 9.0), np.zeros_like(times)])  # rings only from 9 s

    first, _ = disturbance_ringing(times, points, [(8.0, 8.2), (9.0, 9.2)], cutoff_hz=1.5, window=2.0)
    (alone,) = disturbance_ringing(times, points, [(8.0, 8.2)], cutoff_hz=1.5, window=2.0)
    (late,) = disturbance_ringing(times, points, [(19.0, 21.0)], cutoff_hz=1.5, window=2.0)

    # Its window ends at 9 s, when the ringing starts; alone, it would run to 10.2 s. The zero-phase filter
    # spreads the ringing's start back by a few tenths of a second, which the shortened window still catches.
    assert first < 0.3 * alone
    assert np.isnan(late)  # no sample after the disturbance ends


def pushed(times, onset=3.0, end=3.5, size=0.1):
    """A displacement along y that rises during a push from ``onset`` to ``end`` and decays after it."""
    rise = np.clip((times - onset) / (end - onset), 0.0, 1.0)
    return np.column_stack([np.zeros_like(times), size * rise * np.exp(-np.clip(times - end, 0.0, None))])


def test_a_replayed_reference_does_not_follow_and_one_at_the_arm_follows_fully():
    times = np.arange(0.0, 10.0, 0.01)
    taught = np.column_stack([0.1 * times, np.zeros_like(times)])  # the taught motion moves on along x
    arm = taught + pushed(times)

    (replayed,) = disturbance_following(times, arm, taught, taught, [(3.0, 3.5)], after=0.5)
    (at_the_arm,) = disturbance_following(times, arm, arm, taught, [(3.0, 3.5)], after=0.5)
    (halfway,) = disturbance_following(times, arm, taught + 0.5 * pushed(times), taught, [(3.0, 3.5)], after=0.5)

    assert replayed == 0.0
    assert at_the_arm == pytest.approx(1.0)
    assert halfway == pytest.approx(0.5)


def test_a_reference_that_waits_for_a_held_arm_follows_it():
    """Under a block, the arm stays put while the taught motion moves on; a reference that stays with it follows."""
    times = np.arange(0.0, 10.0, 0.01)
    taught = np.column_stack([0.1 * times, np.zeros_like(times)])
    held = taught.copy()
    held[(times >= 4.5) & (times < 5.5), 0] = 0.45  # held where it was at 4.5 s

    (waiting,) = disturbance_following(times, held, held, taught, [(4.5, 5.5)], after=0.0)
    (moving_on,) = disturbance_following(times, held, taught, taught, [(4.5, 5.5)], after=0.0)

    assert waiting == pytest.approx(1.0)
    assert moving_on == 0.0


def test_following_ignores_a_lag_from_before_the_disturbance():
    times = np.arange(0.0, 10.0, 0.01)
    taught = np.column_stack([0.1 * times, np.zeros_like(times)])
    lag = np.array([-0.02, 0.01])  # both the arm and its reference are behind and aside since an earlier push
    arm = taught + lag + pushed(times)

    (gain,) = disturbance_following(times, arm, taught + lag + 0.5 * pushed(times), taught, [(3.0, 3.5)], after=0.5)

    assert gain == pytest.approx(0.5)


def test_following_is_measured_until_the_next_disturbance_starts():
    times = np.arange(0.0, 10.0, 0.01)
    taught = np.zeros((len(times), 2))
    arm = pushed(times) + pushed(times, onset=3.8, end=4.3)
    reference = pushed(times)  # follows the first push only

    first, second = disturbance_following(times, arm, reference, taught, [(3.0, 3.5), (3.8, 4.3)], after=0.5)
    (alone,) = disturbance_following(times, arm, reference, taught, [(3.0, 3.5)], after=0.5)

    assert first == pytest.approx(1.0)  # its window stops at 3.8 s, before the second push moves the arm
    assert alone < 1.0  # alone, its window runs to 4 s, into the second push
    assert second < 0.1


def test_following_is_undefined_without_a_displacement_or_a_sample():
    times = np.arange(0.0, 5.0, 0.01)
    still = np.zeros((len(times), 2))

    unmoved, late = disturbance_following(times, still, still, still, [(1.0, 1.5), (6.0, 6.5)], after=0.5)

    assert np.isnan(unmoved)
    assert np.isnan(late)


def test_the_motors_effort_sums_the_joints_and_counts_only_the_work_they_do():
    times = np.arange(0.0, 2.0, 0.002)
    torques = np.column_stack([np.full(len(times), 3.0), np.full(len(times), -0.5)])
    velocities = np.column_stack([np.full(len(times), 2.0), np.full(len(times), 4.0)])  # joint 2 is driven backward
    duration = len(times) * 0.002

    torque, squared, work = motor_effort(times, torques, velocities)

    assert torque == pytest.approx((3.0 + 0.5) * duration)
    assert squared == pytest.approx((9.0 + 0.25) * duration)
    assert work == pytest.approx(6.0 * duration)  # joint 2's negative power is not recovered


def test_pressing_without_moving_costs_torque_but_no_work():
    times = np.arange(0.0, 1.0, 0.002)
    torques = np.full((len(times), 2), 4.0)

    torque, squared, work = motor_effort(times, torques, np.zeros_like(torques))

    assert torque > 0.0
    assert squared > 0.0
    assert work == 0.0
