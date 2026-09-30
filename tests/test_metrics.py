# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the reach metrics, using reaches whose answers are known."""

import numpy as np
import pytest

from arm_esn_ctrl.metrics import reach_metrics

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
