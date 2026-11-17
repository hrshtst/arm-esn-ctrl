# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the figure of the ESN's autonomous runs."""

import sys

import numpy as np
import pytest

from arm_esn_ctrl.autonomous import Run, Start
from arm_esn_ctrl.storage import REPO_ROOT

# The runners import each other as modules of their own directory, as they run.
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from autonomous_esn import plot_runs


def reach(n_samples):
    """Joint angles and hand positions of ``n_samples`` samples that move over the first 50 and then hold."""
    progress = np.clip(np.arange(n_samples) / 50.0, 0.0, 1.0)[:, np.newaxis]
    return progress * np.array([1.0, 0.5]), progress * np.array([0.3, 0.2])


def test_a_training_take_longer_than_the_runs_is_drawn_over_its_own_length():
    times = 0.01 * np.arange(101)  # runs of 1 s
    q, hand = reach(len(times))
    run = Run(start=Start(origin="demo_00", q=q[0], group="demonstrated"), q=q, q_ref=None, hand=hand, hand_ref=None)
    training = [reach(151)]  # a take of 1.5 s, without a demonstrator to compare the runs with

    figure = plot_runs(times, [run], training, np.array([0.3, 0.2]), "test")

    speed = figure.axes[1]
    assert max(float(np.asarray(line.get_xdata())[-1]) for line in speed.lines) == 1.5
    # The speed axis stops above the take's reach: 0.36 m in 0.5 s.
    assert speed.get_ylim()[1] == pytest.approx(1.6 * np.hypot(0.3, 0.2) / 0.5)
