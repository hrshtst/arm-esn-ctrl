# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the robot arm tracking a reference generated from its measured state."""

import tomllib
from dataclasses import replace

import numpy as np
import pytest
from skelarm import Skeleton

from arm_esn_ctrl.demonstrations import joint_trajectory_log, resample_joint_angles, simulate_reaches
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.storage import REPO_ROOT
from arm_esn_ctrl.tracking import (
    EsnSource,
    ProgressReplaySource,
    ReplaySource,
    TrackerConfig,
    error_dynamics,
    gain_posture,
    gains_posture,
    nearest_demonstration,
    task_joint_angles,
    track,
    tracking_gains,
)

CONFIG = REPO_ROOT / "experiments/demonstrations/reach_tvs.toml"
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


@pytest.mark.parametrize("given", [True, False])
def test_without_the_reference_velocity_joint_pd_lags_a_ramp_by_kd_over_kp_times_its_speed(given):
    start = np.radians([30.0, 90.0])
    step = np.radians([0.1, -0.05])
    ramp = start + step * np.arange(1, 400)[:, np.newaxis]  # a constant reference velocity
    config = TrackerConfig("pd", omega=40.0, acceleration_filter=0.02, reference_velocity=given)

    log = run(ReplaySource(np.vstack([start, ramp]).astype(np.float64)), start, config, duration=1.0)

    # Once settled, the arm moves with the reference; without its velocity, the
    # derivative term resists that motion, which the proportional term balances.
    kp, kd = tracking_gains(config, Skeleton.from_toml(CONFIG), start)
    lag = 0.0 if given else kd / kp * step / ESN_CONFIG.dt
    settled = (log.times > 0.5) & (log.times < 0.9)
    assert np.degrees(log.channel("error")[settled] - lag) == pytest.approx(0.0, abs=0.01)


def test_the_reference_is_sampled_on_the_task_clock_too(demos):
    demo = demos[1]

    log = run(ReplaySource(demo), demo[0])

    # The log's times are sums of simulation steps, a little off the period's multiples.
    assert task_joint_angles(log, ESN_CONFIG.dt, 2.0, "q_ref") == pytest.approx(demo[:201], abs=1e-6)


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


@pytest.mark.parametrize("law", ["computed_torque", "pd"])
def test_the_error_dynamics_of_gains_from_omega_are_critically_damped_at_omega(law):
    skeleton = Skeleton.from_toml(CONFIG)
    posture = np.radians([48.6, 97.2])

    gains = tracking_gains(TrackerConfig(law, 10.0, 0.02), skeleton, posture)
    omega, zeta = error_dynamics(law, gains, skeleton, posture)

    assert omega == pytest.approx([10.0, 10.0]) and zeta == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize("law", ["computed_torque", "pd"])
def test_a_damping_ratio_sets_the_derivative_gain(law):
    skeleton = Skeleton.from_toml(CONFIG)
    posture = np.radians([48.6, 97.2])

    gains = tracking_gains(TrackerConfig(law, 10.0, 0.02, damping=0.3), skeleton, posture)
    critical = tracking_gains(TrackerConfig(law, 10.0, 0.02), skeleton, posture)
    omega, zeta = error_dynamics(law, gains, skeleton, posture)

    assert gains[0] == pytest.approx(critical[0]) and gains[1] == pytest.approx(0.3 * critical[1])
    assert omega == pytest.approx([10.0, 10.0]) and zeta == pytest.approx([0.3, 0.3])


def test_a_smaller_derivative_gain_lowers_the_damping_ratio():
    skeleton = Skeleton.from_toml(CONFIG)
    gains = (np.array([100.0, 100.0]), np.array([4.0, 4.0]))

    omega, zeta = error_dynamics("computed_torque", gains, skeleton, np.zeros(2))

    assert omega == pytest.approx([10.0, 10.0]) and zeta == pytest.approx([0.2, 0.2])


def test_gains_from_omega_need_an_omega():
    with pytest.raises(ValueError, match="given directly"):
        tracking_gains(TrackerConfig("pd", None, 0.02), Skeleton.from_toml(CONFIG), np.zeros(2))


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


CONTROL_PERIOD = 0.01  # 5 simulation steps
HELD = TrackerConfig("pd", omega=20.0, acceleration_filter=0.02, damping=0.1, control_period=CONTROL_PERIOD)


def test_with_a_control_period_the_torque_is_computed_at_its_instants_and_held_between(demos):
    demo = demos[1]

    log = run(ReplaySource(demo), demo[0], HELD, duration=1.0)

    tau = log.channel("tau")[:-1]  # skelarm records the last frame without advancing the controller
    blocks = tau.reshape(-1, round(CONTROL_PERIOD / DT), tau.shape[1])
    assert blocks == pytest.approx(np.repeat(blocks[:, :1], blocks.shape[1], axis=1), abs=0.0)
    assert np.abs(np.diff(blocks[:, 0], axis=0)).max() > 0.0


def test_a_control_period_of_one_simulation_step_changes_nothing(demos):
    demo = demos[1]
    every_step = TrackerConfig("pd", omega=20.0, acceleration_filter=0.02, damping=0.1)

    held = run(ReplaySource(demo), demo[0], replace(HELD, control_period=DT), duration=1.0)
    default = run(ReplaySource(demo), demo[0], every_step, duration=1.0)

    for channel in ("q", "tau", "q_ref", "error"):
        assert held.channel(channel) == pytest.approx(default.channel(channel), abs=0.0)


def test_with_a_control_period_the_reference_and_the_error_are_still_logged_at_every_step(demos):
    demo = demos[1]

    held = run(ReplaySource(demo), demo[0], HELD, duration=1.0)
    every_step = run(ReplaySource(demo), demo[0], replace(HELD, control_period=None), duration=1.0)

    # A replay's reference does not depend on the arm, so it is the same, step by step.
    assert held.channel("q_ref") == pytest.approx(every_step.channel("q_ref"), abs=1e-12)
    assert held.channel("error") == pytest.approx(held.channel("q_ref") - held.channel("q"), abs=1e-12)


def test_a_control_period_must_be_a_whole_number_of_simulation_steps(demos):
    config = replace(HELD, control_period=0.005)

    with pytest.raises(ValueError, match="control period"):
        run(ReplaySource(demos[1]), demos[1][0], config, duration=0.1)


# A demonstration for the progress-based replay: at rest for 5 samples, then along joint 1 by 0.01 rad a sample.
RESTING_LINE = np.vstack([np.zeros((5, 2)), np.column_stack([0.01 * np.arange(1, 51), np.zeros(50)])]).astype(
    np.float64
)


def test_the_progress_replay_is_the_replay_for_an_arm_that_keeps_up():
    progress = ProgressReplaySource(RESTING_LINE, lead=0.02, window=10)
    replay = ReplaySource(RESTING_LINE)
    progress.reset(RESTING_LINE[0])

    for k in range(-3, len(RESTING_LINE) + 5):
        q = RESTING_LINE[min(max(k, 0), len(RESTING_LINE) - 1)]
        assert progress.next_posture(k, q) == pytest.approx(replay.next_posture(k, q))


def test_the_progress_replay_waits_for_a_held_arm_at_its_lead():
    source = ProgressReplaySource(RESTING_LINE, lead=0.025, window=10)
    source.reset(RESTING_LINE[0])
    held = RESTING_LINE[20]  # 0.16 rad along the path

    given = [source.next_posture(k, held) for k in range(40)]

    # The last sample at most 0.025 rad past the arm: 0.18 rad along the path, sample 22.
    assert given[-1] == pytest.approx(RESTING_LINE[22])
    assert given[-1][0] - held[0] <= 0.025 + 1e-12


def test_the_progress_replay_falls_back_with_an_arm_pushed_back():
    source = ProgressReplaySource(RESTING_LINE, lead=0.025, window=10)
    source.reset(RESTING_LINE[0])
    for k in range(30):
        source.next_posture(k, RESTING_LINE[k])

    given = np.vstack([source.next_posture(k, RESTING_LINE[25]) for k in range(30, 35)])

    # Back to 0.025 rad past the arm (sample 27), one sample a period from sample 30.
    assert given == pytest.approx(RESTING_LINE[[29, 28, 27, 27, 27]])


def test_the_progress_replay_searches_the_arm_only_within_its_window():
    there_and_back = np.vstack([RESTING_LINE, RESTING_LINE[::-1]]).astype(np.float64)  # every posture twice
    source = ProgressReplaySource(there_and_back, lead=0.02, window=5)
    source.reset(there_and_back[0])

    for k in range(9):
        source.next_posture(k, there_and_back[k])
    given = source.next_posture(9, there_and_back[9])

    # The arm at sample 9 is also sample 100 on the way back, out of the window.
    assert given == pytest.approx(there_and_back[10])


@pytest.mark.parametrize(("lead", "window"), [(0.0, 10), (0.02, 0)])
def test_the_progress_replay_needs_a_positive_lead_and_window(lead, window):
    with pytest.raises(ValueError, match="must be positive"):
        ProgressReplaySource(RESTING_LINE, lead=lead, window=window)


@pytest.fixture(scope="module")
def task_esn(demos):
    """An ESN of two tasks: demonstration 6 as the first, demonstration 7 as the second."""
    trained = ReachingEsn(ESN_CONFIG)
    trained.fit(demos, contexts=[0, 1], names=["six", "seven"])
    return trained


def test_a_schedule_gives_the_esn_its_first_task_through_the_warmup(task_esn):
    source = EsnSource(task_esn, [(0.0, "seven")])
    task_esn.set_context("six")

    source.reset(np.zeros(2))
    source.next_posture(-W, np.zeros(2))

    assert task_esn.context == pytest.approx([0.0, 1.0])


def test_a_schedule_changes_the_task_at_its_times(task_esn):
    source = EsnSource(task_esn, [(0.0, "six"), (0.5, "seven"), (1.0, [0.5, 0.5])])

    assert [source.task_at(time) for time in (-0.25, 0.0, 0.49, 0.5, 0.99)] == ["six", "six", "six", "seven", "seven"]
    assert source.task_at(1.0) == [0.5, 0.5]
    source.reset(np.zeros(2))
    source.next_posture(round(0.5 / ESN_CONFIG.dt), np.zeros(2))  # the step at 0.5 s is already the second task's
    assert task_esn.context == pytest.approx([0.0, 1.0])


def test_without_a_schedule_the_esn_keeps_its_context(demos, task_esn):
    start = demos[1][0]
    task_esn.set_context("seven")

    log = run(EsnSource(task_esn), start)

    task_esn.set_context("seven")
    q = task_joint_angles(log, ESN_CONFIG.dt, 2.0)
    assert np.degrees(q - task_esn.generate(start, 200)) == pytest.approx(0.0, abs=0.1)


def test_a_schedule_must_be_in_the_order_of_its_times(task_esn):
    with pytest.raises(ValueError, match="order of its times"):
        EsnSource(task_esn, [(1.0, "six"), (0.5, "seven")])


# A periodic demonstration: 10 samples of approach along joint 1, then a circle in joint space of 40 samples a turn,
# gone around twice and a little more.
_TURN = 0.2 * np.column_stack([np.cos(2 * np.pi * np.arange(90) / 40), np.sin(2 * np.pi * np.arange(90) / 40)])
PERIODIC = np.vstack([np.column_stack([np.linspace(0.0, 0.18, 10), np.zeros(10)]), _TURN]).astype(np.float64)
LOOP = (10, 50)  # one turn: its first sample, and one past its last


def test_a_periodic_replay_goes_around_its_loop_without_end():
    source = ReplaySource(PERIODIC, loop=LOOP)

    given = np.vstack([source.next_posture(k, PERIODIC[0]) for k in range(-2, 200)])

    assert given[:2] == pytest.approx(PERIODIC[[0, 0]])  # the warm-up
    assert given[2:51] == pytest.approx(PERIODIC[1:50])  # the approach and the first turn
    assert given[51:91] == pytest.approx(PERIODIC[10:50])  # the loop again
    assert given[171:202] == pytest.approx(PERIODIC[10:41])  # and again


def test_the_periodic_progress_replay_is_the_periodic_replay_for_an_arm_that_keeps_up():
    progress = ProgressReplaySource(PERIODIC, lead=0.1, window=10, loop=LOOP)
    replay = ReplaySource(PERIODIC, loop=LOOP)
    progress.reset(PERIODIC[0])

    q = PERIODIC[0]
    for k in range(-3, 200):
        given = progress.next_posture(k, q)
        assert given == pytest.approx(replay.next_posture(k, q))
        q = given  # the arm is where the reference was


def test_the_periodic_progress_replay_waits_for_a_held_arm_in_a_later_turn():
    source = ProgressReplaySource(PERIODIC, lead=0.07, window=10, loop=LOOP)
    source.reset(PERIODIC[0])
    q = PERIODIC[0]
    for k in range(70):  # into the second turn, the arm keeping up
        q = source.next_posture(k, q)
    held = q

    given = [source.next_posture(k, held) for k in range(70, 120)]

    # A sample is 0.2 * 2 pi / 40 = 0.0314 rad of the path: the reference stops two samples past the arm.
    ahead = np.linalg.norm(given[-1] - held)
    assert 0.05 < ahead < 0.07
    assert given[-1] == pytest.approx(given[-2])


def test_the_loops_path_goes_on_from_its_last_sample_to_its_first():
    source = ProgressReplaySource(PERIODIC, lead=0.07, window=10, loop=LOOP)

    turn = source._length_at(50, LOOP) - source._length_at(10, LOOP)

    assert turn == pytest.approx(40 * np.linalg.norm(PERIODIC[11] - PERIODIC[10]))
    assert source._length_at(95, LOOP) == pytest.approx(source._length_at(15, LOOP) + 2 * turn)


def test_a_given_take_sets_where_the_gains_are_set(demos, tmp_path, monkeypatch):
    monkeypatch.setenv("ARM_ESN_CTRL_STORAGE_ROOT", str(tmp_path))
    skeleton = Skeleton.from_toml(CONFIG)
    times = (ESN_CONFIG.dt * np.arange(len(demos[0]))).astype(np.float64)
    joint_trajectory_log(skeleton, times, demos[0], {"type": "reaching"}, producer="test").save(
        tmp_path / "take.sklog.npz"
    )

    given = gains_posture({"gains_from": "take.sklog.npz"}, [demos[1]], ESN_CONFIG.dt)

    assert given == pytest.approx(demos[0][-1])
    assert gains_posture({}, demos, ESN_CONFIG.dt) == pytest.approx(gain_posture(demos))
