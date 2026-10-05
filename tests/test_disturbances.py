# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the disturbances of Stage 2 and the arms that meet them."""

import tomllib

import numpy as np
import pytest
from skelarm import Skeleton

from arm_esn_ctrl.demonstrations import endpoint_positions, simulate_disturbed_reach, simulate_reaches
from arm_esn_ctrl.disturbances import Block, make_disturbance
from arm_esn_ctrl.storage import REPO_ROOT
from arm_esn_ctrl.tracking import ReplaySource, TrackerConfig, track, tracking_gains

CONFIG = REPO_ROOT / "configs/demonstrations/reach_tvs.toml"
TARGET = np.array([0.0, 1.2])
PUSH = {"type": "push", "force": 5.0, "onset": 0.4, "duration": 0.1}
BLOCK = {"type": "block", "onset": 0.1, "release": 0.4, "stiffness": 20000.0, "damping": 100.0}


def tip(skeleton):
    return np.array([skeleton.links[-1].xe, skeleton.links[-1].ye])


def ramp_run(external_force, duration=0.6):
    """The arm tracking a steady joint ramp with computed torque, from a reset after a 0.05 s warm-up."""
    start = np.radians([30.0, 90.0])
    ramp = start + np.radians([0.2, -0.1]) * np.arange(200)[:, np.newaxis]
    skeleton = Skeleton.from_toml(CONFIG)
    skeleton.q = start
    config = TrackerConfig("computed_torque", 10.0, 0.02)
    gains = tracking_gains(config, skeleton, start)
    log = track(
        skeleton,
        ReplaySource(ramp.astype(np.float64)),
        config,
        gains,
        period=0.01,
        warmup_steps=5,
        duration=duration,
        dt=0.002,
        external_force=external_force,
    )
    return log, endpoint_positions(skeleton, log.channel("q"))


def test_a_push_acts_sideways_to_the_reach_for_its_duration():
    start_hand = np.array([0.5, 1.2])
    push = make_disturbance(PUSH, start_hand, TARGET)
    arm = Skeleton.from_toml(CONFIG)

    assert push is not None
    force = push(0.45, arm)
    assert np.linalg.norm(force) == pytest.approx(5.0)
    assert force @ (TARGET - start_hand) == pytest.approx(0.0)
    assert force == pytest.approx([0.0, -5.0])  # the reach goes in -x; counterclockwise from it is -y
    assert push(0.39, arm) == pytest.approx([0.0, 0.0])
    assert push(0.5, arm) == pytest.approx([0.0, 0.0])


def test_forces_act_on_the_task_clock():
    times = []

    def watch(t, skeleton):
        times.append(t)
        return np.zeros(2)

    ramp_run(watch, duration=0.1)

    assert times[0] == pytest.approx(-0.05)
    assert min(abs(t) for t in times) < 1e-9


def test_a_block_holds_the_tip_where_it_was_and_then_lets_go():
    block = make_disturbance(BLOCK, np.zeros(2), TARGET)
    _, free_hand = ramp_run(None)

    log, hand = ramp_run(block)

    assert isinstance(block, Block) and block.held is not None
    blocked = (log.times >= 0.1 + 0.002) & (log.times < 0.4)
    assert np.linalg.norm(hand[blocked] - block.held, axis=1).max() < 0.003
    assert np.linalg.norm(free_hand[blocked] - block.held, axis=1).max() > 0.05  # it would have moved on
    after = log.times >= 0.4
    assert np.abs(log.channel("ext_force")[after]).max() == 0.0


def test_unknown_disturbances_and_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown disturbance type"):
        make_disturbance({"type": "shake"}, np.zeros(2), TARGET)
    with pytest.raises(ValueError, match="exactly the keys"):
        make_disturbance(PUSH | {"direction": "up"}, np.zeros(2), TARGET)


def test_without_a_force_the_disturbed_demonstrator_reaches_as_the_demonstrations():
    with CONFIG.open("rb") as f:
        config = tomllib.load(f)
    config["demonstrations"]["start_q"] = config["demonstrations"]["start_q"][7:]
    config["task"]["duration"] = 1.0
    undisturbed = simulate_reaches(config)[0]

    log = simulate_disturbed_reach(
        config, np.radians(config["demonstrations"]["start_q"][0]), 1.0, lambda t, s: np.zeros(2)
    )

    assert log.channel("q") == pytest.approx(undisturbed.channel("q"), abs=1e-12)
    assert log.channel("ext_force") == pytest.approx(0.0)
