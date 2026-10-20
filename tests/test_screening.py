# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for screening a trained ESN by where it can rest, from the held reservoir's fixed points."""

import tomllib

import numpy as np
import pytest

from arm_esn_ctrl.demonstrations import resample_joint_angles, simulate_reaches
from arm_esn_ctrl.dynamics import ClosedLoop
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.screening import autonomous_runs, find_fixed_points, held_fixed_points, posture_map, spectrum
from arm_esn_ctrl.storage import REPO_ROOT

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
def demo():
    """Demonstration 6, sampled every 10 ms."""
    with CONFIG.open("rb") as f:
        config = tomllib.load(f)
    config["demonstrations"]["start_q"] = config["demonstrations"]["start_q"][6:7]
    config["task"]["duration"] = 1.5
    return resample_joint_angles(simulate_reaches(config)[0], ESN_CONFIG.dt)[1]


@pytest.fixture(scope="module")
def esn(demo):
    trained = ReachingEsn(ESN_CONFIG)
    trained.fit([demo])
    return trained


@pytest.fixture(scope="module")
def loop(esn):
    return ClosedLoop.from_esn(esn)


def test_the_held_fixed_points_are_where_the_held_reservoir_rests(loop, demo):
    q = demo[::30]
    x, converged = held_fixed_points(loop, q)

    assert converged.all()
    u = loop.normalize(q)
    np.testing.assert_allclose(loop.held(x, u), x, atol=1e-12)
    rested = np.zeros_like(x)
    for _ in range(3000):
        rested = loop.held(rested, u)
    np.testing.assert_allclose(x, rested, atol=1e-10)
    # From other states, the same.
    other, converged = held_fixed_points(loop, q, np.random.default_rng(0).uniform(-1.0, 1.0, x.shape))
    assert converged.all()
    np.testing.assert_allclose(other, x, atol=1e-10)


def test_the_runs_on_its_own_are_the_esns(loop, esn, demo):
    starts = np.array([demo[0], demo[-1], demo[0] + 0.3])
    q = autonomous_runs(loop, starts, ESN_CONFIG.warmup_steps, 120)

    assert q.shape == (3, 121, 2)
    for start, run in zip(starts, q, strict=True):
        np.testing.assert_allclose(run, esn.generate(start, 120), atol=1e-12)


def test_the_fixed_points_are_the_zeros_of_the_posture_map(loop, esn, demo):
    grid = np.degrees(np.linspace(demo.min() - 0.5, demo.max() + 0.5, 15))
    found = find_fixed_points(loop, posture_map(loop, grid))

    assert found.candidates >= len(found.states) >= 1
    for x in found.states:
        np.testing.assert_allclose(loop.advance(x), x, atol=1e-12)
    # Among them, where the run from the demonstration's start comes to rest.
    rest, converged = loop.fixed_point(esn_rest_state(esn, demo[0]))
    assert converged
    assert min(np.abs(x - rest).max() for x in found.states) < 1e-8


def test_a_fixed_point_is_the_held_reservoirs_at_its_own_posture(loop, esn, demo):
    rest, _ = loop.fixed_point(esn_rest_state(esn, demo[0]))
    held, converged = held_fixed_points(loop, loop.posture(rest)[np.newaxis])

    assert converged.all()
    np.testing.assert_allclose(held[0], rest, atol=1e-10)
    pmap = posture_map(loop, np.degrees(loop.posture(rest)[0]) + np.array([-1.0, 0.0, 1.0]))
    assert pmap.residual.shape == (3, 3, 2)


def test_the_spectrum_is_sorted_by_modulus(loop, esn, demo):
    rest, _ = loop.fixed_point(esn_rest_state(esn, demo[0]))
    eigenvalues = spectrum(loop, rest)

    assert eigenvalues.shape == (ESN_CONFIG.n_neurons,)
    assert np.all(np.diff(np.abs(eigenvalues)) <= 1e-12)
    np.testing.assert_allclose(np.abs(eigenvalues[0]), np.abs(np.linalg.eigvals(loop.jacobian(rest))).max())


def esn_rest_state(esn, start_q):
    """The state at the end of a long run of ``esn`` on its own from ``start_q``."""
    esn.reset()
    for _ in range(esn.config.warmup_steps):
        esn.step(start_q)
    output = esn.step(start_q)
    for _ in range(3000):
        output = esn.step(output)
    return esn.state()
