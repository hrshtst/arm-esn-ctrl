# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for generating and loading reaching demonstrations."""

import tomllib
from pathlib import Path

import numpy as np
import pytest
from skelarm import Skeleton, StateLog

from arm_esn_ctrl.demonstrations import (
    check_take,
    endpoint_positions,
    joint_trajectory_log,
    load_joint_angles,
    simulate_reaches,
    smooth_take,
)
from arm_esn_ctrl.storage import REPO_ROOT

CONFIG = REPO_ROOT / "experiments/demonstrations/reach_tvs.toml"


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


START = np.radians([18.2, 119.9])
END = np.radians([48.6, 97.2])


def taught_take(rest=0.5, reach=1.0, hold=3.0, start=START, jump_at=None, skeleton=None, step_deg=None):
    """A take as the recorder saves it: at rest, a minimum-jerk reach to END, then still; sampled near 100 Hz.

    The intervals vary a little, as the recorder's clock does. ``jump_at`` moves the
    arm by 5 deg in joint 1 at that time, and back 50 ms later. ``step_deg`` rounds
    the joint angles' change from the start to its multiples, as the recorder's
    whole-pixel cursor does.
    """
    skeleton = skeleton or Skeleton.from_toml(CONFIG)
    rng = np.random.default_rng(0)
    times = np.cumsum(np.concatenate([[0.0], rng.uniform(0.009, 0.012, 2000)]))
    times = times[times <= rest + reach + hold]
    tau = np.clip((times - rest) / reach, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    q = start + np.outer(s, END - start)
    if step_deg is not None:
        q = start + np.radians(step_deg) * np.round((q - start) / np.radians(step_deg))
    if jump_at is not None:  # out and back within 50 ms
        q[(times >= jump_at) & (times < jump_at + 0.05)] += np.radians([5.0, 0.0])
    log = StateLog(skeleton)
    for t, posture in zip(times, q, strict=True):
        skeleton.q = posture
        log.record_skeleton(skeleton, float(t))
    return log


def check(log, **overrides):
    skeleton = Skeleton.from_toml(CONFIG)
    target = endpoint_positions(skeleton, END[np.newaxis])[0]
    options = {"start_q": START, "target": target, "radius": 0.02, "hold": 2.0}
    options |= {"start_tolerance": np.radians(0.01), "max_tip_speed": 3.0}
    return check_take(log, skeleton, **(options | overrides))


def test_a_take_that_starts_arrives_and_holds_is_a_demonstration():
    assert check(taught_take()) == []


def test_a_take_must_begin_at_the_start_posture():
    (problem,) = check(taught_take(start=START + np.radians([1.0, 0.0])))
    assert "begins" in problem


def test_a_take_must_arrive_and_hold():
    (never,) = check(taught_take(), target=np.array([0.0, 0.0]))
    assert "never" in never
    (short,) = check(taught_take(hold=1.0))
    assert "before" in short


def test_a_take_must_not_jump():
    (problem,) = check(taught_take(jump_at=0.2))
    assert "m/s" in problem


def test_a_take_must_be_recorded_with_the_same_arm():
    other = Skeleton.from_toml(CONFIG)
    other.links[1].prop.length = 1.1
    problems = check(taught_take(skeleton=other))
    assert any("arm" in problem for problem in problems)


SMOOTHING = {"kind": "butterworth", "cutoff_hz": 8.0, "order": 4}


def task_table():
    with CONFIG.open("rb") as f:
        return tomllib.load(f)["task"]


def test_smoothing_keeps_the_times_and_a_still_posture():
    take = taught_take(start=END)  # from the end posture: still all along
    smoothed = smooth_take(take, Skeleton.from_toml(CONFIG), task_table(), SMOOTHING)
    np.testing.assert_array_equal(smoothed.times, take.times)
    np.testing.assert_allclose(smoothed.channel("q"), take.channel("q"), atol=1e-12)


def test_smoothing_removes_the_pixel_steps_and_keeps_a_demonstration():
    """A slow take whose joints move in 0.3 deg steps, as a whole-pixel cursor makes them, stalls between steps."""
    exact, stepped = taught_take(reach=3.0), taught_take(reach=3.0, step_deg=0.3)
    smoothed = smooth_take(stepped, Skeleton.from_toml(CONFIG), task_table(), SMOOTHING)

    def stalls(log):  # samples of the reach where no joint moves
        q = log.channel("q")
        reach = (log.times[1:] > 0.8) & (log.times[1:] < 3.2)
        return int(np.count_nonzero(np.all(np.abs(np.diff(q, axis=0)) < 1e-6, axis=1) & reach))

    assert stalls(exact) == 0
    assert stalls(stepped) > 20
    assert stalls(smoothed) == 0
    error = np.degrees(np.abs(smoothed.channel("q") - exact.channel("q"))).max()
    assert error < 0.3  # within one step of the exact reach
    assert check(smoothed) == []


def test_the_configured_filter_never_takes_the_hand_out_of_where_it_held():
    """A correction inside the goal, out toward its edge and back, as a hand makes it.

    The filter of reach_manual_single.toml only averages, so the filtered hand never
    goes farther from the target than the recorded hand did; a 4th-order Butterworth
    filter overshoots such a correction.
    """
    skeleton = Skeleton.from_toml(CONFIG)
    take = taught_take(reach=3.0)
    times, q = take.times, take.channel("q")
    # After arriving, move joint 1 out by 0.9 deg (about 16 mm of hand) within 0.1 s, then back.
    q[(times > 4.0) & (times < 4.1)] += np.radians([0.9, 0.0])
    corrected = joint_trajectory_log(skeleton, times, q, task_table(), producer="test")
    target = endpoint_positions(skeleton, END[np.newaxis])[0]

    def farthest(log):
        hand = endpoint_positions(skeleton, log.channel("q"))
        return np.linalg.norm(hand - target, axis=1)[times > 3.6].max()

    with (REPO_ROOT / "experiments/demonstrations/reach_manual_single.toml").open("rb") as f:
        smoothing = tomllib.load(f)["recording"]["filter"]
    assert farthest(smooth_take(corrected, skeleton, task_table(), smoothing)) <= farthest(corrected) + 1e-5
    assert farthest(smooth_take(corrected, skeleton, task_table(), SMOOTHING)) > farthest(corrected) + 1e-3
