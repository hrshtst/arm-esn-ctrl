# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the autonomous runs from start postures all over the joint space."""

import sys
import tomllib

import numpy as np
import pytest

from arm_esn_ctrl.demonstrations import resample_joint_angles, simulate_reaches
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.storage import REPO_ROOT

# The runners import each other as modules of their own directory, as they run.
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from global_starts import closed_loop_run, grid_postures, held_steps, nearest_states, outcome

CONFIG = REPO_ROOT / "experiments/demonstrations/reach_tvs.toml"
ESN_CONFIG = EsnConfig(
    dt=0.01,
    warmup=0.25,
    n_neurons=100,
    spectral_radius=0.9,
    sparsity=0.1,
    leak_rate=0.1,
    input_scaling=1.0,
    bias=True,
    ridge=1e-6,
    seed=0,
)


@pytest.fixture(scope="module")
def esn():
    """An ESN trained on demonstration 6, sampled every 10 ms."""
    with CONFIG.open("rb") as f:
        config = tomllib.load(f)
    config["demonstrations"]["start_q"] = config["demonstrations"]["start_q"][6:7]
    config["task"]["duration"] = 1.0
    trained = ReachingEsn(ESN_CONFIG)
    trained.fit([resample_joint_angles(simulate_reaches(config)[0], ESN_CONFIG.dt)[1]])
    return trained


def test_the_grid_covers_both_joints_whole_range_the_first_fastest():
    grid = grid_postures(90.0)

    np.testing.assert_array_equal(grid[:5], [[-180, -180], [-90, -180], [0, -180], [90, -180], [-180, -90]])
    assert len(grid) == 16
    assert grid.max() == 90.0


def test_the_run_is_the_generated_one_and_its_states_follow_each_input(esn):
    start = np.array([0.3, 0.8])
    q, states = closed_loop_run(esn, start, 30)

    np.testing.assert_allclose(q, esn.generate(start, 30), rtol=0.0, atol=1e-12)
    assert states.shape == (31, ESN_CONFIG.n_neurons)
    # The state at time 0 has seen the warm-up and the start posture once more, as when the start is held.
    np.testing.assert_allclose(states[0], held_steps(esn, start, ESN_CONFIG.warmup_steps + 1)[-1], atol=1e-12)


def test_the_nearest_states_are_found_by_distance():
    rng = np.random.default_rng(0)
    states, reference = rng.normal(size=(7, 5)), rng.normal(size=(11, 5))
    distance, index = nearest_states(states, reference)

    brute = np.linalg.norm(states[:, None, :] - reference[None, :, :], axis=2)
    np.testing.assert_array_equal(index, np.argmin(brute, axis=1))
    np.testing.assert_allclose(distance, brute.min(axis=1), atol=1e-12)


def test_a_run_ends_at_the_target_at_rest_elsewhere_moving_or_out_of_range():
    dt, target = 0.01, np.array([0.5, 0.0])
    still = np.zeros((300, 2))
    at_target = np.tile(target, (300, 1))
    creeping = np.column_stack([np.radians(np.linspace(0.0, 3.0, 300)), np.zeros(300)])  # 1 deg/s

    assert outcome(still, at_target, target, 0.02, dt, 0.5) == "target"
    assert outcome(still, at_target + 0.1, target, 0.02, dt, 0.5) == "rest elsewhere"
    assert outcome(creeping, at_target, target, 0.02, dt, 0.5) == "moving"
    assert outcome(still + np.radians(181.0), at_target, target, 0.02, dt, 0.5) == "out of range"
