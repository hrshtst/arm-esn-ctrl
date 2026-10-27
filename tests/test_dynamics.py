# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for a trained ESN's update, evaluated with numpy anywhere in its state space."""

import tomllib

import numpy as np
import pytest

from arm_esn_ctrl.demonstrations import resample_joint_angles, simulate_reaches
from arm_esn_ctrl.dynamics import ClosedLoop, local_rates
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.states import teacher_forced_states
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


def run_states(esn, start_q, n_steps):
    """The joint angles and the states of an autonomous run of ``esn``, as generate() runs it."""
    esn.reset()
    for _ in range(esn.config.warmup_steps):
        esn.step(start_q)
    output = esn.step(start_q)
    q, states = [start_q], [esn.state()]
    for _ in range(n_steps):
        q.append(output)
        output = esn.step(output)
        states.append(esn.state())
    return np.array(q), np.array(states)


def test_the_map_reproduces_the_held_warmup_and_the_autonomous_run(esn, loop, demo):
    q, states = run_states(esn, demo[0], 80)

    np.testing.assert_allclose(q, esn.generate(demo[0], 80), atol=1e-12)
    # Fed its own output, each state is the image of the one before, and gives the next posture.
    np.testing.assert_allclose(loop.advance(states[:-1]), states[1:], atol=1e-12)
    np.testing.assert_allclose(loop.posture(states[:-1]), q[1:], atol=1e-12)
    # Held at the start posture from the reset, the reservoir goes through the warm-up's states.
    warmup = teacher_forced_states(esn, demo)[: ESN_CONFIG.warmup_steps]
    x, u = np.zeros(ESN_CONFIG.n_neurons), loop.normalize(demo[0])
    for expected in warmup:
        x = loop.held(x, u)
        np.testing.assert_allclose(x, expected, atol=1e-12)


def test_the_jacobians_are_the_derivatives_of_the_maps(loop, esn, demo):
    _, states = run_states(esn, demo[0], 30)
    x, u = states[-1], loop.normalize(demo[5])
    rng = np.random.default_rng(0)
    change = 1e-6 * rng.normal(size=loop.n_neurons)

    linear = loop.jacobian(x) @ change
    np.testing.assert_allclose(loop.advance(x + change) - loop.advance(x - change), 2 * linear, atol=1e-12)
    linear = loop.held_jacobian(x, u) @ change
    np.testing.assert_allclose(loop.held(x + change, u) - loop.held(x - change, u), 2 * linear, atol=1e-12)


def test_newtons_method_finds_where_the_run_comes_to_rest(loop, esn, demo):
    q, states = run_states(esn, demo[0], 3000)
    fixed, converged = loop.fixed_point(states[-1])

    assert converged
    np.testing.assert_allclose(loop.advance(fixed), fixed, atol=1e-12)
    np.testing.assert_allclose(loop.posture(fixed), q[-1], atol=1e-6)
    held, converged = loop.fixed_point(states[0], loop.normalize(demo[0]))
    assert converged
    np.testing.assert_allclose(loop.held(held, loop.normalize(demo[0])), held, atol=1e-12)


def test_a_mode_decays_by_e_in_its_time_constant(loop):
    assert loop.time_constant(np.exp(-loop.dt / 0.5)) == pytest.approx(0.5)
    assert loop.time_constant(1.0 + 0.0j) == float("inf")


def test_the_rates_along_a_run_are_those_of_its_jacobians(loop, esn, demo):
    _, states = run_states(esn, demo[0], 60)
    rates = local_rates(loop, states, window=20, stride=10, n_rates=3)

    np.testing.assert_array_equal(rates.starts, [0, 10, 20, 30])
    phi = np.eye(loop.n_neurons)
    for x in states[10:30]:
        phi = loop.jacobian(x) @ phi
    singular = np.linalg.svd(phi, compute_uv=False)
    np.testing.assert_allclose(rates.exponents[1], np.log(singular[:3]) / 0.2, rtol=1e-9)
    assert rates.across.shape == (4, 3)
    assert np.all(rates.across[:, 0] <= rates.exponents[:, 0] + 1e-9)  # a projection never grows a change more
    assert rates.posture.shape == (4, 2)
    with pytest.raises(ValueError, match="whole number of strides"):
        local_rates(loop, states, window=25, stride=10, n_rates=3)


def test_at_rest_to_the_last_bit_every_change_is_across_the_run(loop, esn, demo):
    _, states = run_states(esn, demo[0], 3000)
    fixed, _ = loop.fixed_point(states[-1])
    rates = local_rates(loop, np.repeat(fixed[np.newaxis], 41, axis=0), window=20, stride=10, n_rates=3)

    assert np.all(np.isfinite(rates.across))
    np.testing.assert_allclose(rates.across, rates.exponents, rtol=1e-12)
