# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for generating and loading reaching demonstrations."""

import tomllib
from pathlib import Path

import numpy as np
import pytest
from skelarm import Skeleton, StateLog

from arm_esn_ctrl.demonstrations import endpoint_positions, joint_trajectory_log, load_joint_angles, simulate_reaches
from arm_esn_ctrl.storage import REPO_ROOT

CONFIG = REPO_ROOT / "configs/demonstrations/reach_tvs.toml"


def short_config():
    """The time-varying-stiffness demonstrations, cut down to two short reaches."""
    with CONFIG.open("rb") as f:
        config = tomllib.load(f)
    config["demonstrations"]["start_q"] = config["demonstrations"]["start_q"][:2]
    config["task"]["duration"] = 1.6
    return config


def test_every_scripted_reach_ends_at_the_target():
    config = short_config()
    target = np.asarray(config["task"]["target"]["pos"])

    logs = simulate_reaches(config)

    assert len(logs) == 2
    for log, start_q in zip(logs, config["demonstrations"]["start_q"], strict=True):
        q = log.channel("q")
        assert np.degrees(q[0]) == pytest.approx(start_q)
        hand = endpoint_positions(log.build_skeleton(), q)
        assert np.linalg.norm(hand[-1] - target) < 0.01


def test_joint_angles_are_resampled_at_a_fixed_period(tmp_path: Path):
    log = simulate_reaches(short_config())[0]
    path = tmp_path / "demo.sklog.npz"
    log.save(path)

    times, q = load_joint_angles(path, dt=0.01)

    assert times[0] == 0.0
    assert np.diff(times) == pytest.approx(0.01)
    assert times[-1] == pytest.approx(log.times[-1] - log.times[0])
    # The simulation step is 0.002 s, so every fifth sample is kept unchanged.
    assert q == pytest.approx(log.channel("q")[::5])


def test_endpoint_positions_follow_forward_kinematics():
    skeleton = Skeleton.from_toml(CONFIG)
    q = np.radians([[0.0, 0.0], [90.0, 0.0], [0.0, 90.0]])

    hand = endpoint_positions(skeleton, q)

    assert hand == pytest.approx(np.array([[1.8, 0.0], [0.0, 1.8], [1.0, 0.8]]), abs=1e-12)


def test_endpoint_positions_use_angles_beyond_the_joint_limits(recwarn: pytest.WarningsRecorder):
    skeleton = Skeleton.from_toml(CONFIG)  # joint limits are +-180 deg
    q = np.radians([[0.0, 270.0]])

    hand = endpoint_positions(skeleton, q)

    assert hand == pytest.approx(np.array([[1.0, -0.8]]), abs=1e-12)
    assert len(recwarn) == 0


def test_joint_trajectory_log_replays_the_trajectory(tmp_path: Path):
    skeleton = Skeleton.from_toml(CONFIG)
    times = np.linspace(0.0, 1.0, 11)
    q = np.column_stack([np.linspace(0.3, 0.8, 11), np.linspace(2.0, 1.6, 11)])
    task = {"type": "reaching", "target": {"pos": [0.0, 1.2], "tolerance": 0.02}}
    path = tmp_path / "trajectory.sklog.npz"

    joint_trajectory_log(skeleton, times, q, task, producer="test").save(path)
    log = StateLog.load(path)

    assert log.times == pytest.approx(times)
    assert log.channel("q") == pytest.approx(q)
    assert log.channel("dq")[:, 0] == pytest.approx(0.5)
    assert log.extra["playback"]["task"] == task
