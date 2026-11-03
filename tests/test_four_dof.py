# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""The experiments on skelarm's four-link arm: its start offsets, tracker, stops, and ESN as a dynamical system."""

import itertools

import numpy as np
import pytest
from matplotlib.figure import Figure
from skelarm import JointPD, Skeleton, compute_mass_matrix
from skelarm.dynamics import StopModel

from arm_esn_ctrl.autonomous import start_postures
from arm_esn_ctrl.drawing import OffsetMap
from arm_esn_ctrl.dynamics import ClosedLoop, RobotLoop, StateRecorder
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.postures import joint_limits_deg, map_grid, start_grid
from arm_esn_ctrl.screening import find_fixed_points, posture_map
from arm_esn_ctrl.storage import REPO_ROOT
from arm_esn_ctrl.tracking import (
    MatrixPD,
    ReplaySource,
    TrackerConfig,
    error_dynamics,
    mass_matrix_at,
    task_joint_angles,
    track,
    tracking_gains,
)

ARM = REPO_ROOT / "experiments/demonstrations/reach_manual_four_dof.toml"
START = np.radians([-20.0, 10.0, 60.0, 50.0])  # the file's start: joint 1 at its lower limit
END = np.radians([11.0, 39.0, 79.0, 53.0])  # where a reach to the target (0, 0.4) m ends
DT = 0.01  # the ESN's period (s)
SIMULATION_DT = 0.002
ESN_CONFIG = EsnConfig(
    dt=DT,
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
# The search's tracker, joint PD at 20 rad/s and a damping ratio of 0.4, its torque held 10 ms: with each joint's
# own inertia (v2's), and with the whole mass matrix (the four-link arm's).
PER_JOINT = TrackerConfig("pd", 20.0, 0.02, damping=0.4, control_period=DT)
MASS_MATRIX = TrackerConfig("pd", 20.0, 0.02, damping=0.4, control_period=DT, mass_matrix=True)
STOPS: StopModel = "locked"  # the arm's [simulator] stops


@pytest.fixture(scope="module")
def skeleton():
    return Skeleton.from_toml(ARM)


@pytest.fixture(scope="module")
def reach():
    """A reach in joint space of minimum jerk, 2 s long after a 0.5 s pause, then held 1 s, every 10 ms."""
    s = np.clip((np.arange(round(3.5 / DT) + 1) * DT - 0.5) / 2.0, 0.0, 1.0)[:, np.newaxis]
    return START + (10 * s**3 - 15 * s**4 + 6 * s**5) * (END - START)


def test_the_offset_starts_beyond_the_joint_limits_are_left_out(skeleton, reach):
    evaluation = {"start_offsets_deg": [[0.0]], "start_grid": {"span_deg": 15.0, "step_deg": 15.0}}
    starts = start_postures(evaluation, {"demo_00": reach}, joint_limits_deg(skeleton))

    # Joint 1 can only go up (at its lower limit) and joint 2 at most 10 deg down: 2 x 2 x 3 x 3.
    assert len(starts) == 36
    assert starts[0].origin == "demo_00"
    np.testing.assert_array_equal(starts[0].q, START)
    assert "demo_00 +0,+15,-15,+15 deg" in [start.origin for start in starts]
    assert len(start_postures(evaluation, {"demo_00": reach})) == 81  # without the limits, every combination
    with pytest.raises(ValueError, match="one angle per joint"):
        start_postures({"start_offsets_deg": [[0.0, 0.0]]}, {"demo_00": reach})


def test_the_grids_cover_each_joints_limits(skeleton):
    grids = map_grid(skeleton, 10.0)

    assert [len(g) for g in grids] == [6, 13, 16, 12]
    assert [g[-1] for g in grids] == [30.0, 120.0, 150.0, 90.0]
    assert [len(g) for g in start_grid(skeleton, 30.0)] == [2, 5, 6, 4]  # a joint's upper limit only on the grid
    with pytest.raises(ValueError, match="misses the upper limit"):
        map_grid(skeleton, 30.0)


def test_an_offset_map_nests_joints_3_and_4_around_joints_1_and_2():
    offsets = np.array(list(itertools.product([0.0, 15.0], [0.0, 15.0], [-15.0, 0.0, 15.0], [-15.0, 0.0, 15.0])))
    layout = OffsetMap.of(offsets)

    assert layout.nested
    assert layout.extent == (-0.5, 5.5, -0.5, 5.5)  # 2 x 3 cells along x (joints 1, 3), 2 x 3 along y (2, 4)
    assert layout.cell([15.0, 0.0, -15.0, 15.0]) == (4, 1)  # row: joint 2's 0 in joint 4's third block
    assert layout.zero == (2.0, 2.0)
    grid = layout.grid({tuple(offset): float(i) for i, offset in enumerate(offsets)})
    assert grid.shape == (6, 6)
    assert np.isfinite(grid).all()
    ax = Figure().subplots()
    layout.decorate(ax)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["+0", "+15"] * 3  # joint 1, in each block of joint 3
    top, right = ax.child_axes  # the blocks: joint 3's offsets on the top, joint 4's on the right
    assert [t.get_text() for t in top.get_xticklabels()] == ["-15", "+0", "+15"]
    assert right.get_ylabel() == "joint 4 offset (deg)"
    two = OffsetMap.of(np.array([[-5.0, 0.0], [5.0, 0.0], [0.0, 5.0]]))
    assert not two.nested
    assert two.extent == (-7.5, 7.5, -2.5, 7.5)


def test_the_mass_matrix_gains_give_every_mode_omega_and_zeta(skeleton):
    kp, kd = tracking_gains(MASS_MATRIX, skeleton, END)
    inertia = mass_matrix_at(skeleton, END)

    np.testing.assert_allclose(kp, 400.0 * inertia)
    np.testing.assert_allclose(kd, 16.0 * inertia)
    omega, zeta = error_dynamics("pd", (kp, kd), skeleton, END)
    np.testing.assert_allclose(omega, 20.0, rtol=1e-9)
    np.testing.assert_allclose(zeta, 0.4, rtol=1e-9)
    # Each joint's own inertia is far above what joints 1 and 2 feel with the others free.
    assert np.all(np.diag(inertia)[:2] > 10.0 / np.diag(np.linalg.inv(inertia))[:2])


def test_matrix_pd_with_diagonal_gains_is_joint_pd(skeleton):
    kp, kd = tracking_gains(PER_JOINT, skeleton, END)

    class Reference:
        def sample(self, t):
            return END, np.full(4, 0.3), np.zeros(4)

    arm = skeleton.clone()
    arm.q, arm.dq = START, np.full(4, -0.2)
    matrix = MatrixPD(Reference(), np.diag(kp), np.diag(kd)).control(0.0, arm)
    np.testing.assert_allclose(matrix, JointPD(Reference(), kp, kd).control(0.0, arm))


def replay(skeleton, config, reach, stops: StopModel = STOPS):
    """The arm replaying ``reach`` from the start by time with the tracker ``config``: its log."""
    posed = skeleton.clone()
    posed.q = START
    gains = tracking_gains(config, skeleton, END)
    return track(
        posed,
        ReplaySource(reach),
        config,
        gains,
        period=DT,
        warmup_steps=25,
        duration=3.5,
        dt=SIMULATION_DT,
        stops=stops,
    )


def test_with_each_joints_own_inertia_the_held_torque_throws_the_arm_off(skeleton, reach):
    q = task_joint_angles(replay(skeleton, PER_JOINT, reach), DT, 3.5)

    assert np.degrees(np.abs(q - reach).max()) > 20.0


def test_with_the_mass_matrix_the_arm_follows_the_reach(skeleton, reach):
    q = task_joint_angles(replay(skeleton, MASS_MATRIX, reach), DT, 3.5)

    assert np.degrees(np.abs(q - reach).max()) < 1.0


def energy_from_the_stops(skeleton, log):
    """The kinetic energy the arm gained over a run, less the motors' work on it (J): what the stops gave it."""
    q, dq, tau = (log.channel(name).reshape(len(log.times), -1) for name in ("q", "dq", "tau"))
    arm = skeleton.clone()

    def energy(k):
        arm.set_state(q=q[k], dq=dq[k])
        return 0.5 * dq[k] @ compute_mass_matrix(arm) @ dq[k]

    return energy(len(q) - 1) - energy(0) - float(np.sum(tau[:-1] * np.diff(q, axis=0)))


def test_a_joint_pressed_into_its_stop_gains_the_arm_energy_unless_the_stop_locks_it(skeleton, reach):
    pressed = reach.copy()
    pressed[:, 0] = np.radians(-25.0)  # joint 1 asked to stay 5 deg below its stop, where it starts
    projected = replay(skeleton, MASS_MATRIX, pressed, stops="projected")
    locked = replay(skeleton, MASS_MATRIX, pressed, stops="locked")

    for log in (projected, locked):
        assert np.mean(log.channel("q")[:, 0] == np.radians(-20.0)) > 0.9  # joint 1 at its stop, nearly throughout
    assert energy_from_the_stops(skeleton, projected) > 1.0  # the motors take out what the stop puts in
    assert energy_from_the_stops(skeleton, locked) < 0.01  # it does no work, and its impacts take some out


@pytest.fixture(scope="module")
def esn(reach):
    trained = ReachingEsn(ESN_CONFIG)
    trained.fit([reach])
    return trained


@pytest.fixture(scope="module")
def loop(esn):
    return ClosedLoop.from_esn(esn)


@pytest.mark.parametrize("stops", ["projected", "locked"])
def test_the_robot_map_with_gain_matrices_reproduces_the_tracked_arm(esn, loop, skeleton, reach, stops):
    gains = tracking_gains(MASS_MATRIX, skeleton, reach[-1])
    robot = RobotLoop.from_tracker(loop, skeleton, MASS_MATRIX, gains, SIMULATION_DT, stops=stops)
    posed = skeleton.clone()
    posed.q = reach[0]
    recorder = StateRecorder(esn)
    log = track(
        posed,
        recorder,
        MASS_MATRIX,
        gains,
        period=DT,
        warmup_steps=ESN_CONFIG.warmup_steps,
        duration=1.5,
        dt=SIMULATION_DT,
        stops=stops,
    )
    states = robot.run_states(log, recorder, ESN_CONFIG.warmup_steps)

    assert robot.n_joints == 4
    assert states.shape == (150, ESN_CONFIG.n_neurons + 8)
    assert np.sum(log.channel("q")[log.times >= 0, 0] == START[0]) > 5  # the ESN first holds joint 1 on its stop
    for z, z_next in itertools.pairwise(states[1:]):
        np.testing.assert_allclose(robot.advance(z), z_next, atol=1e-12)
    z = states[60]
    change = 1e-6 * np.random.default_rng(0).normal(size=robot.dimension)
    linear = robot.jacobian(z) @ change
    np.testing.assert_allclose(robot.advance(z + change) - robot.advance(z - change), 2 * linear, atol=1e-11)


def test_the_fixed_points_are_found_on_a_grid_of_four_joints(esn, loop, reach):
    esn.reset()
    for _ in range(ESN_CONFIG.warmup_steps):
        esn.step(reach[0])
    output = esn.step(reach[0])
    for _ in range(3000):
        output = esn.step(output)
    rest, converged = loop.fixed_point(esn.state())
    assert converged

    around = np.degrees(loop.posture(rest))[:, np.newaxis] + np.array([-4.0, -1.0, 2.0])
    pmap = posture_map(loop, around)
    found = find_fixed_points(loop, pmap)

    assert pmap.residual.shape == (3, 3, 3, 3, 4)
    assert len(found.states) >= 1
    assert min(np.abs(x - rest).max() for x in found.states) < 1e-8
