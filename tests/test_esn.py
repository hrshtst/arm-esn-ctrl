# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the ESN trained on one trajectory and run autonomously."""

import numpy as np
import pytest
from numpy.typing import NDArray

from arm_esn_ctrl.esn import EsnConfig, ReachingEsn

DT = 0.01
CONFIG = EsnConfig(
    dt=DT,
    warmup=1.0,
    n_neurons=300,
    spectral_radius=0.5,
    sparsity=0.1,
    leak_rate=0.3,
    input_scaling=1.0,
    bias=True,
    ridge=1.0,
    seed=0,
)


def joint_reach(start=(0.3, 2.0), goal=(0.8, 1.6)) -> NDArray[np.float64]:
    """A two-joint minimum-jerk reach lasting 1 s, then 1 s of holding still."""
    tau = np.clip(np.arange(0.0, 2.0 + DT / 2, DT) / 1.0, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    start_q = np.asarray(start, dtype=np.float64)
    return start_q + np.outer(s, np.asarray(goal, dtype=np.float64) - start_q)


def test_warmup_is_converted_to_samples():
    assert CONFIG.warmup_steps == 100


def test_autonomous_run_reproduces_the_training_trajectory():
    q = joint_reach()
    esn = ReachingEsn(CONFIG)
    esn.fit([q])

    generated = esn.generate(q[0], len(q) - 1)

    assert generated.shape == q.shape
    assert generated[0] == pytest.approx(q[0])
    assert np.degrees(np.sqrt(np.mean((generated - q) ** 2))) < 2.0
    assert np.degrees(np.abs(generated[-1] - q[-1])).max() < 0.5


def test_autonomous_run_stays_at_the_goal_after_the_trajectory_ends():
    q = joint_reach()
    esn = ReachingEsn(CONFIG)
    esn.fit([q])

    generated = esn.generate(q[0], 3 * (len(q) - 1))

    assert np.degrees(np.abs(generated[len(q) :] - q[-1])).max() < 0.5


def test_runs_restart_from_a_reset_reservoir():
    q = joint_reach()
    esn = ReachingEsn(CONFIG)
    esn.fit([q])

    assert esn.generate(q[0], 50) == pytest.approx(esn.generate(q[0], 50))


def test_each_training_trajectory_is_reproduced_from_its_own_start():
    trajectories = [joint_reach(start) for start in ((0.3, 2.0), (1.0, 1.3), (0.2, 1.4))]
    esn = ReachingEsn(CONFIG)
    esn.fit(trajectories)

    for q in trajectories:
        generated = esn.generate(q[0], len(q) - 1)
        assert np.degrees(np.sqrt(np.mean((generated - q) ** 2))) < 2.0
        assert np.degrees(np.abs(generated[-1] - q[-1])).max() < 0.5


def test_normalization_does_not_depend_on_how_long_the_hold_lasts():
    q = joint_reach()
    longer_hold = np.vstack([q, np.repeat(q[-1:], 200, axis=0)])
    short, long = ReachingEsn(CONFIG), ReachingEsn(CONFIG)
    short.fit([q])
    long.fit([longer_hold])

    assert long.center == pytest.approx(short.center)
    assert long.half_range == pytest.approx(short.half_range)
    normalized = (q - short.center) / short.half_range
    assert normalized.min(axis=0) == pytest.approx([-1.0, -1.0])
    assert normalized.max(axis=0) == pytest.approx([1.0, 1.0])
