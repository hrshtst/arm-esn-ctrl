# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the robot arm tracking a reference generated from its measured state."""

import tomllib

import numpy as np
import pytest
from skelarm import Skeleton

from arm_esn_ctrl.demonstrations import resample_joint_angles, simulate_reaches
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.storage import REPO_ROOT
from arm_esn_ctrl.tracking import (
    EsnSource,
    ReplaySource,
    TrackerConfig,
    nearest_demonstration,
    task_joint_angles,
    track,
    tracking_gains,
)

CONFIG = REPO_ROOT / "configs/demonstrations/reach_tvs.toml"
ESN_CONFIG = EsnConfig(
    dt=0.01,
    warmup=0.25,
    n_neurons=200,
    spectral_radius=1.3,
    sparsity=0.1,
    leak_rate=0.05,
    input_scaling=0.1,
    bias=True,
    ridge=1e-6,
    seed=0,
)
W = ESN_CONFIG.warmup_steps
DT = 0.002  # simulation step: 5 per reference period
COMPUTED_TORQUE = TrackerConfig("computed_torque", omega=40.0, acceleration_filter=0.02)


@pytest.fixture(scope="module")
def demos():
    """Demonstrations 6 and 7, sampled every reference period."""
    with CONFIG.open("rb") as f:
        config = tomllib.load(f)
    config["demonstrations"]["start_q"] = config["demonstrations"]["start_q"][6:]
    config["task"]["duration"] = 2.5
    return [resample_joint_angles(log, ESN_CONFIG.dt)[1] for log in simulate_reaches(config)]


@pytest.fixture(scope="module")
def esn(demos):
    trained = ReachingEsn(ESN_CONFIG)
    trained.fit(demos)
    return trained


def run(source, start_q, config=COMPUTED_TORQUE, duration=2.0):
    skeleton = Skeleton.from_toml(CONFIG)
    skeleton.q = start_q
    gains = tracking_gains(config, skeleton, start_q)
    return track(skeleton, source, config, gains, period=ESN_CONFIG.dt, warmup_steps=W, duration=duration, dt=DT)


class RecordingSource(ReplaySource):
    """A replay that remembers what it was given."""

    def __init__(self, q_demo):
        super().__init__(q_demo)
        self.given = []

    def next_posture(self, k, q):
        self.given.append((k, q.copy()))
        return super().next_posture(k, q)


def test_the_arm_holds_its_start_posture_through_the_warmup_and_the_task_starts_at_0(demos):
    demo = demos[1]

    log = run(ReplaySource(demo), demo[0])

    times = log.times
    assert times[0] == pytest.approx(-ESN_CONFIG.warmup)
    assert np.min(np.abs(times)) < 1e-9  # a frame at t = 0
    assert times[-1] == pytest.approx(2.0)
    warmup = times < 1e-9
    held = np.repeat(demo[:1], int(warmup.sum()), axis=0)
    assert log.channel("q")[warmup] == pytest.approx(held, abs=1e-12)
    assert log.channel("q_ref")[warmup] == pytest.approx(held, abs=1e-12)


def test_the_source_is_given_the_measured_posture_every_reference_period(demos):
    source = RecordingSource(demos[1])

    log = run(source, demos[1][0], duration=0.5)

    steps_per_period = round(ESN_CONFIG.dt / DT)
    assert [k for k, _ in source.given] == list(range(-W, 50))
    q = log.channel("q")
    for k, given in source.given:
        assert given == pytest.approx(q[(k + W) * steps_per_period], abs=1e-15)


def test_the_reference_moves_straight_from_one_posture_to_the_next():
    start = np.radians([30.0, 90.0])
    step = np.radians([0.1, -0.05])
    ramp = start + step * np.arange(1, 400)[:, np.newaxis]  # one step further every period

    log = run(ReplaySource(np.vstack([start, ramp]).astype(np.float64)), start, duration=1.0)

    task = log.times > -1e-9
    task[-1] = False  # skelarm records the last frame without advancing the controller
    expected = start + np.outer(log.times[task] / ESN_CONFIG.dt, step)
    assert log.channel("q_ref")[task] == pytest.approx(expected, abs=1e-12)


def test_replaying_a_demonstration_reproduces_it(demos):
    demo = demos[1]

    log = run(ReplaySource(demo), demo[0])

    assert np.degrees(task_joint_angles(log, ESN_CONFIG.dt, 2.0) - demo[:201]) == pytest.approx(0.0, abs=0.1)


def test_the_esn_on_the_measured_posture_generates_its_autonomous_run_when_tracked_closely(demos, esn):
    start = demos[1][0]

    log = run(EsnSource(esn), start)

    q = task_joint_angles(log, ESN_CONFIG.dt, 2.0)
    assert np.degrees(q - esn.generate(start, 200)) == pytest.approx(0.0, abs=0.1)


def test_joint_pd_also_tracks_with_gains_scaled_by_the_inertia(demos):
    demo = demos[1]
    config = TrackerConfig("pd", omega=40.0, acceleration_filter=0.02)

    log = run(ReplaySource(demo), demo[0], config=config)

    assert np.degrees(task_joint_angles(log, ESN_CONFIG.dt, 2.0) - demo[:201]) == pytest.approx(0.0, abs=0.2)


def test_gains_set_the_natural_frequency_of_the_tracking_error():
    skeleton = Skeleton.from_toml(CONFIG)
    posture = np.radians([48.6, 97.2])
    omega = 10.0

    kp, kd = tracking_gains(TrackerConfig("computed_torque", omega, 0.02), skeleton, posture)
    kp_pd, kd_pd = tracking_gains(TrackerConfig("pd", omega, 0.02), skeleton, posture)

    assert kp == pytest.approx([100.0, 100.0]) and kd == pytest.approx([20.0, 20.0])
    assert kd_pd / kp_pd == pytest.approx(kd / kp)  # critically damped, joint by joint
    assert kp_pd[0] > kp_pd[1]  # the shoulder moves the whole arm


def test_an_unknown_tracking_law_is_rejected():
    with pytest.raises(ValueError, match="unknown tracking law"):
        tracking_gains(TrackerConfig("bang_bang", 10.0, 0.02), Skeleton.from_toml(CONFIG), np.zeros(2))


def test_the_reference_period_must_be_a_whole_number_of_simulation_steps(demos):
    skeleton = Skeleton.from_toml(CONFIG)
    skeleton.q = demos[1][0]
    gains = tracking_gains(COMPUTED_TORQUE, skeleton, demos[1][0])

    with pytest.raises(ValueError, match="whole number"):
        track(
            skeleton,
            ReplaySource(demos[1]),
            COMPUTED_TORQUE,
            gains,
            period=0.01,
            warmup_steps=W,
            duration=0.1,
            dt=0.003,
        )


def test_the_nearest_demonstration_is_found_by_its_start_posture():
    demos = {"demo_00": np.radians([[29.4, 88.2]]), "demo_07": np.radians([[18.2, 119.9]])}

    assert nearest_demonstration(np.radians([20.0, 115.0]), demos) == "demo_07"
