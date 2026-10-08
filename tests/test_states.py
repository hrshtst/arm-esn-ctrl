# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for recomputing an ESN's reservoir states rather than storing them."""

import dataclasses
import tomllib

import numpy as np
import pytest
from skelarm import Skeleton

from arm_esn_ctrl.demonstrations import resample_joint_angles, simulate_reaches
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.states import run_states, teacher_forced_states
from arm_esn_ctrl.storage import REPO_ROOT
from arm_esn_ctrl.tracking import EsnSource, TrackerConfig, track, tracking_gains

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
W = ESN_CONFIG.warmup_steps
DT = 0.002  # simulation step: 5 per reference period
TRACKER = TrackerConfig("pd", omega=20.0, acceleration_filter=0.02, damping=0.1, reference_velocity=False)


@pytest.fixture(scope="module")
def demo():
    """Demonstration 6, sampled every reference period."""
    with CONFIG.open("rb") as f:
        config = tomllib.load(f)
    config["demonstrations"]["start_q"] = config["demonstrations"]["start_q"][6:7]
    config["task"]["duration"] = 1.0
    return resample_joint_angles(simulate_reaches(config)[0], ESN_CONFIG.dt)[1]


def trained(demo, seed=0):
    esn = ReachingEsn(dataclasses.replace(ESN_CONFIG, seed=seed))
    esn.fit([demo])
    return esn


class RecordingSource(EsnSource):
    """The ESN on the measured posture, remembering its state after every step."""

    def __init__(self, esn):
        super().__init__(esn)
        self.states = []

    def next_posture(self, k, q):
        posture = super().next_posture(k, q)
        self.states.append(self.esn.state())
        return posture


def run(esn, start_q, duration=0.5):
    """A tracked run of ``esn`` from ``start_q``, and the states it went through."""
    skeleton = Skeleton.from_toml(CONFIG)
    skeleton.q = start_q
    source = RecordingSource(esn)
    gains = tracking_gains(TRACKER, skeleton, start_q)
    log = track(skeleton, source, TRACKER, gains, period=ESN_CONFIG.dt, warmup_steps=W, duration=duration, dt=DT)
    return log, np.array(source.states)


def test_the_states_recomputed_from_a_run_log_are_the_runs_own(demo):
    esn = trained(demo)
    log, recorded = run(esn, demo[0] + np.radians([5.0, -5.0]))

    times, states = run_states(esn, log)

    assert states.shape == recorded.shape == (W + 50, 100)
    assert states == pytest.approx(recorded, abs=1e-12)
    assert times == pytest.approx(ESN_CONFIG.dt * np.arange(-W, 50))  # the warm-up at negative times


def test_a_log_the_esn_did_not_drive_is_refused(demo):
    log, _ = run(trained(demo), demo[0])

    with pytest.raises(ValueError, match="logged reference"):
        run_states(trained(demo, seed=1), log)


def test_the_take_fed_in_gives_a_state_per_input_after_the_held_start_posture(demo):
    esn = trained(demo)

    states = teacher_forced_states(esn, demo)

    assert states.shape == (W + len(demo), 100)
    assert teacher_forced_states(esn, demo) == pytest.approx(states, abs=0.0)  # from the reset state each time
