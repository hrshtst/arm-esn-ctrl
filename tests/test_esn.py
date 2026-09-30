# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the ESN trained on one trajectory and run autonomously."""

import numpy as np
import pytest

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


def joint_reach():
    """A two-joint minimum-jerk reach lasting 1 s, then 1 s of holding still."""
    tau = np.clip(np.arange(0.0, 2.0 + DT / 2, DT) / 1.0, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    return np.array([0.3, 2.0]) + np.outer(s, [0.5, -0.4])


def test_warmup_is_converted_to_samples():
    assert CONFIG.warmup_steps == 100


def test_autonomous_run_reproduces_the_training_trajectory():
    q = joint_reach()
    esn = ReachingEsn(CONFIG)
    esn.fit(q)

    generated = esn.generate(q[0], len(q) - 1)

    assert generated.shape == q.shape
    assert generated[0] == pytest.approx(q[0])
    assert np.degrees(np.sqrt(np.mean((generated - q) ** 2))) < 2.0
    assert np.degrees(np.abs(generated[-1] - q[-1])).max() < 0.5


def test_autonomous_run_stays_at_the_goal_after_the_trajectory_ends():
    q = joint_reach()
    esn = ReachingEsn(CONFIG)
    esn.fit(q)

    generated = esn.generate(q[0], 3 * (len(q) - 1))

    assert np.degrees(np.abs(generated[len(q) :] - q[-1])).max() < 0.5


def test_runs_restart_from_a_reset_reservoir():
    q = joint_reach()
    esn = ReachingEsn(CONFIG)
    esn.fit(q)

    assert esn.generate(q[0], 50) == pytest.approx(esn.generate(q[0], 50))
