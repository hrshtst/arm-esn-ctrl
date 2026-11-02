# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""A trained ESN as a dynamical system: its update, evaluated anywhere in its state space.

Run on its own (:meth:`arm_esn_ctrl.esn.ReachingEsn.generate`), the ESN is fed its
own output, so that its reservoir state x, a row of one value per neuron, follows the map

    x' = G(x) = (1 - a) x + a tanh(u(x) W_in + b + x W),  u(x) = x W_out + b_out,

where u(x) is the ESN's output, the next joint angles in normalized units, W the
recurrent weights, W_in the input weights, b the bias, and a the leak rate. Nothing
but the state carries over from one step to the next: the history of the run is
in the state, and G is the whole system. Its fixed points are the postures the ESN
can rest at, and its Jacobian tells how a small change of the state grows or
decays over a step. Held at a posture u, as during the warm-up, the reservoir
follows x' = F(x, u) = (1 - a) x + a tanh(u W_in + b + x W) instead.

:class:`ClosedLoop` reads the weights of a trained ESN from rclib and evaluates both
maps and their Jacobians with numpy, at any state. :func:`local_rates` measures,
along a run, how fast small changes of the state grow or decay, those across the
run, which tell whether it attracts the states near it, among them.

On the robot (:mod:`arm_esn_ctrl.tracking`), the ESN is fed the arm's measured joint
angles rather than its own output, and joint PD, its torque computed every period and
held, makes the arm follow the ESN's output. Sampled at the ESN's period, the ESN and
the arm are then one system, whose state z = (x, q, q̇) holds the reservoir's state and
the arm's joint angles and velocities, and which follows the map

    x' = F(x, u(q)),  τ = K_p (P(x) - q) + K_d ((P(x') - P(x)) / Δ - q̇),
    (q', q̇') = the arm's motion over the period Δ under the torque τ,

where u(q) normalizes the joint angles as the ESN's input and P(x) is the posture the
ESN gives from x, the reference's segment over the period running from P(x) to P(x').
Nothing else carries over: the reference's last point is P(x), the arm moves in a
plane without gravity, and joint PD keeps no state. At rest, the arm needs no torque,
so it rests at P(x), fed to the ESN as its own output: the system's fixed points are
the ESN's own, the arm at rest at their postures, and only how they attract changes.
:class:`RobotLoop` evaluates this map and its Jacobian, which :func:`local_rates`
measures along the robot's run as along the ESN's own.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from skelarm import compute_forward_kinematics, compute_jacobian, integrate_with_limits

from arm_esn_ctrl.tracking import EsnSource

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray
    from skelarm import Skeleton, StateLog

    from arm_esn_ctrl.esn import ReachingEsn
    from arm_esn_ctrl.tracking import TrackerConfig


@dataclass(frozen=True)
class ClosedLoop:
    """The update of a trained ESN fed its own output, and of its reservoir held at a posture (see the module)."""

    recurrent: NDArray[np.float64]  # W, (n, n): the entry (i, j) is the weight from neuron i to neuron j
    inputs: NDArray[np.float64]  # W_in, (n_joints, n)
    bias: NDArray[np.float64]  # b, (n,)
    leak_rate: float  # a
    readout: NDArray[np.float64]  # W_out, (n, n_joints)
    readout_bias: NDArray[np.float64]  # b_out, (n_joints,)
    center: NDArray[np.float64]  # the joint angles of the normalized posture 0 (rad)
    half_range: NDArray[np.float64]  # the joint angles of a normalized unit (rad)
    dt: float  # the ESN's period (s)

    @classmethod
    def from_esn(cls, esn: ReachingEsn) -> ClosedLoop:
        """The update of ``esn``, a trained ESN of one reservoir and a ridge readout with a bias, read from rclib."""
        reservoir = esn.model.get_reservoir(0)
        n = reservoir.getNNeurons()
        rows, cols, values = reservoir.getRecurrentWeights()
        recurrent = np.zeros((n, n))
        recurrent[rows, cols] = values
        readout = esn.model.get_readout()
        if not readout.getIncludeBias():
            msg = "the ESN's readout must have a bias"
            raise ValueError(msg)
        weights = readout.getWeights()
        if weights.shape[0] != n + 1:
            msg = "the ESN's readout must read one reservoir's state"
            raise ValueError(msg)
        return cls(
            recurrent,
            reservoir.getInputWeights(),
            reservoir.getBias(),
            reservoir.getLeakRate(),
            weights[:-1],
            weights[-1],
            esn.center,
            esn.half_range,
            esn.config.dt,
        )

    @property
    def n_neurons(self) -> int:
        """The number of reservoir neurons."""
        return len(self.bias)

    @property
    def dimension(self) -> int:
        """The number of values in the state: one per neuron."""
        return self.n_neurons

    @property
    def scale(self) -> NDArray[np.float64] | None:
        """How :func:`local_rates` weighs the state's values: alike, all of them reservoir values."""
        return None

    @cached_property
    def posture_readout(self) -> NDArray[np.float64]:
        """How a change of the state moves the posture the ESN gives (rad), (n_joints, n)."""
        return self.half_range[:, np.newaxis] * self.readout.T

    def output(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """The ESN's output read from the state ``x`` (one row per state): the next posture, normalized."""
        return x @ self.readout + self.readout_bias

    def posture(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """The joint angles (rad) the ESN gives from the state ``x``."""
        return self.output(x) * self.half_range + self.center

    def normalize(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """The joint angles ``q`` (rad), normalized as the ESN's input."""
        return (q - self.center) / self.half_range

    def held(self, x: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
        """F(x, u): the state after ``x`` given the normalized posture ``u``."""
        return (1.0 - self.leak_rate) * x + self.leak_rate * np.tanh(u @ self.inputs + self.bias + x @ self.recurrent)

    def advance(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """G(x): the state after ``x`` when the ESN is fed its own output."""
        return self.held(x, self.output(x))

    @cached_property
    def coupled(self) -> NDArray[np.float64]:
        """How the state reaches each neuron fed the ESN's own output: directly and through the output, (n, n)."""
        return self.readout @ self.inputs + self.recurrent

    def jacobian(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """The Jacobian of G at the state ``x``: the entry (i, j) is dG_i/dx_j."""
        return self._jacobian(x, self.output(x), self.coupled)

    def held_jacobian(self, x: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
        """The Jacobian of F(., u) at the state ``x``: the entry (i, j) is dF_i/dx_j."""
        return self._jacobian(x, u, self.recurrent)

    def input_jacobian(self, x: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
        """The Jacobian of F(``x``, .) at the normalized posture ``u``: the entry (i, j) is dF_i/du_j."""
        slope = 1.0 - np.tanh(u @ self.inputs + self.bias + x @ self.recurrent) ** 2
        return self.leak_rate * slope[:, np.newaxis] * self.inputs.T

    def fixed_point(
        self, x: NDArray[np.float64], u: NDArray[np.float64] | None = None, tolerance: float = 1e-12, steps: int = 50
    ) -> tuple[NDArray[np.float64], bool]:
        """A fixed point of G, or of F(., ``u``) if ``u`` is given, by Newton's method from the state ``x``.

        Returns the last iterate and whether it converged: whether a step moved no
        entry of the state by more than ``tolerance`` within ``steps`` steps.
        """
        identity = np.eye(self.n_neurons)
        for _ in range(steps):
            if u is None:
                residual, jacobian = self.advance(x) - x, self.jacobian(x)
            else:
                residual, jacobian = self.held(x, u) - x, self.held_jacobian(x, u)
            try:
                step = np.linalg.solve(jacobian - identity, -residual)
            except np.linalg.LinAlgError:
                return x, False
            x = x + step
            if not np.all(np.isfinite(x)):
                return x, False
            if np.abs(step).max() <= tolerance:
                return x, True
        return x, False

    def time_constant(self, eigenvalue: complex) -> float:
        """How long a mode of ``eigenvalue`` takes to decay by e (s); infinite if it does not decay."""
        modulus = abs(eigenvalue)
        return float(-self.dt / np.log(modulus)) if modulus < 1.0 else float("inf")

    def _jacobian(
        self, x: NDArray[np.float64], u: NDArray[np.float64], coupled: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        slope = 1.0 - np.tanh(u @ self.inputs + self.bias + x @ self.recurrent) ** 2
        return (1.0 - self.leak_rate) * np.eye(self.n_neurons) + self.leak_rate * slope[:, np.newaxis] * coupled.T


@dataclass(frozen=True)
class RobotLoop:
    """The ESN driving the arm through joint PD, as one map of the state z = (x, q, q̇) every period (see the module).

    The tracker is :class:`arm_esn_ctrl.tracking.ReferenceTracker`'s joint PD, its
    torque computed at each reference instant and held for the period, given the
    reference's velocity or not; the arm is simulated as skelarm's
    ``simulate_controlled`` simulates it, by ``substeps`` steps of semi-implicit Euler,
    its joints stopped at their limits. A tip force (N) can act over a period, as a
    pulse pushes the arm: held, or one force per simulation step
    (:func:`simulated_forces` gives a run's).
    """

    loop: ClosedLoop
    arm: Skeleton  # the arm's model; every evaluation overwrites its state
    kp: NDArray[np.float64]  # joint PD's gains, per joint
    kd: NDArray[np.float64]
    substeps: int  # simulation steps per period
    reference_velocity: bool = True  # False gives joint PD a zero reference velocity
    lower: NDArray[np.float64] | None = None  # the joint limits (rad), hard stops; None: none
    upper: NDArray[np.float64] | None = None
    weights: tuple[float, float] = (1.0, 1.0)  # how much more local_rates weighs the joint angles and velocities

    @classmethod
    def from_tracker(
        cls,
        loop: ClosedLoop,
        skeleton: Skeleton,
        config: TrackerConfig,
        gains: tuple[NDArray[np.float64], NDArray[np.float64]],
        dt: float,
        *,
        enforce_limits: bool = True,
    ) -> RobotLoop:
        """The ESN of ``loop`` driving ``skeleton`` by the tracker ``config`` and ``gains``, simulated every ``dt``."""
        held = config.control_period is not None and np.isclose(config.control_period, loop.dt)
        if config.law != "pd" or not held:
            msg = "the system is joint PD's, its torque computed every ESN period and held"
            raise ValueError(msg)
        substeps = round(loop.dt / dt)
        if substeps < 1 or not np.isclose(substeps * dt, loop.dt):
            msg = f"the ESN's period {loop.dt} s must be a whole number of simulation steps of {dt} s"
            raise ValueError(msg)
        arm = skeleton.clone()
        lower = upper = None
        if enforce_limits:
            lower = np.array([link.prop.qmin for link in arm.links[1:]], dtype=np.float64)
            upper = np.array([link.prop.qmax for link in arm.links[1:]], dtype=np.float64)
        return cls(loop, arm, gains[0], gains[1], substeps, config.reference_velocity, lower, upper)

    @property
    def n_joints(self) -> int:
        return len(self.kp)

    @property
    def dimension(self) -> int:
        """The number of values in the state: one per neuron, then each joint's angle, then its velocity."""
        return self.loop.n_neurons + 2 * self.n_joints

    @property
    def dt(self) -> float:
        """The period (s)."""
        return self.loop.dt

    @cached_property
    def scale(self) -> NDArray[np.float64]:
        """How :func:`local_rates` weighs the state's values: the joint angles as the ESN's input, their velocities
        by how far they move them in a period, each times its ``weights``."""
        n, half_range = self.loop.n_neurons, self.loop.half_range
        angles, velocities = self.weights
        return np.concatenate([np.ones(n), angles / half_range, velocities * self.dt / half_range])

    @cached_property
    def posture_readout(self) -> NDArray[np.float64]:
        """How a change of the state moves the arm's posture (rad), (n_joints, dimension)."""
        n, j = self.loop.n_neurons, self.n_joints
        return np.hstack([np.zeros((j, n)), np.eye(j), np.zeros((j, j))])

    def split(self, z: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """The reservoir's state, the joint angles, and the joint velocities in ``z``."""
        n, j = self.loop.n_neurons, self.n_joints
        return z[:n], z[n : n + j], z[n + j :]

    def at_rest(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """The state of the reservoir at ``x`` and the arm at rest at the posture the ESN gives from it."""
        return np.concatenate([x, self.loop.posture(x), np.zeros(self.n_joints)])

    def advance(self, z: NDArray[np.float64], force: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
        """The state a period after ``z``, under the tip ``force`` (N) if given: held, or a row per simulation step."""
        x, q, dq = self.split(z)
        x_next = self.loop.held(x, self.loop.normalize(q))
        q_next, dq_next = self._arm_motion(q, dq, self._torque(x, x_next, q, dq), force)
        return np.concatenate([x_next, q_next, dq_next])

    def jacobian(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """The Jacobian of the map at ``z``: the ESN's analytically, the arm's motion's by central differences."""
        loop, j = self.loop, self.n_joints
        x, q, dq = self.split(z)
        u = loop.normalize(q)
        x_next = loop.held(x, u)
        fx = loop.held_jacobian(x, u)
        fq = loop.input_jacobian(x, u) / loop.half_range
        readout = loop.posture_readout
        gain = self.kd / self.dt if self.reference_velocity else np.zeros(j)
        torque_x = self.kp[:, np.newaxis] * readout + gain[:, np.newaxis] * (readout @ fx - readout)
        torque_q = -np.diag(self.kp) + gain[:, np.newaxis] * (readout @ fq)
        motion = self._motion_jacobian(q, dq, self._torque(x, x_next, q, dq))
        by_state, by_torque = motion[:, : 2 * j], motion[:, 2 * j :]
        top = np.hstack([fx, fq, np.zeros((loop.n_neurons, j))])
        bottom = np.hstack(
            [
                by_torque @ torque_x,
                by_state[:, :j] + by_torque @ torque_q,
                by_state[:, j:] - by_torque * self.kd[np.newaxis, :],
            ]
        )
        return np.vstack([top, bottom])

    def force_jacobian(self, z: NDArray[np.float64], step: float = 1e-6) -> NDArray[np.float64]:
        """How a small tip force held over the period moves the next state, per newton, (dimension, 2)."""
        x, q, dq = self.split(z)
        torque = self._torque(x, self.loop.held(x, self.loop.normalize(q)), q, dq)
        columns = []
        for k in range(2):
            push = np.zeros(2)
            push[k] = step
            ahead = np.concatenate(self._arm_motion(q, dq, torque, push))
            behind = np.concatenate(self._arm_motion(q, dq, torque, -push))
            columns.append((ahead - behind) / (2 * step))
        return np.vstack([np.zeros((self.loop.n_neurons, 2)), np.column_stack(columns)])

    def run_states(self, log: StateLog, recorder: StateRecorder, warmup_steps: int) -> NDArray[np.float64]:
        """The states of a run of :func:`arm_esn_ctrl.tracking.track`, at every reference instant of its task.

        ``recorder`` drove the run's reference, ``warmup_steps`` its warm-up. The state
        at the task's k-th instant holds the reservoir's state before the ESN's step
        there and the arm's at that instant. From the second instant on, each is the
        image of the one before: over the first period, the reference runs from the
        start posture rather than from the ESN's output.
        """
        q = log.channel("q").reshape(len(log.times), -1)
        dq = log.channel("dq").reshape(len(log.times), -1)
        states = []
        for k, x in recorder.states[warmup_steps:]:
            step = (k + warmup_steps) * self.substeps
            states.append(np.concatenate([x, q[step], dq[step]]))
        return np.array(states)

    def _torque(
        self, x: NDArray[np.float64], x_next: NDArray[np.float64], q: NDArray[np.float64], dq: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Joint PD's torque at an instant, the reference running from the posture at ``x`` to that at ``x_next``."""
        start, end = self.loop.posture(x), self.loop.posture(x_next)
        velocity = (end - start) / self.dt if self.reference_velocity else np.zeros_like(q)
        return self.kp * (start - q) + self.kd * (velocity - dq)

    def _arm_motion(
        self,
        q: NDArray[np.float64],
        dq: NDArray[np.float64],
        torque: NDArray[np.float64],
        force: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The arm's joint angles and velocities a period after ``q`` and ``dq``, ``torque`` held, under ``force``."""
        for link, angle, velocity in zip(self.arm.links[1:], q, dq, strict=True):
            link.q, link.dq = float(angle), float(velocity)
        compute_forward_kinematics(self.arm)
        h = self.dt / self.substeps
        forces = None if force is None else np.broadcast_to(force, (self.substeps, 2))
        for i in range(self.substeps):
            applied = torque if forces is None else torque + compute_jacobian(self.arm).T @ forces[i]
            integrate_with_limits(self.arm, applied, h, self.lower, self.upper)
        return self.arm.q.copy(), self.arm.dq.copy()

    def _motion_jacobian(
        self, q: NDArray[np.float64], dq: NDArray[np.float64], torque: NDArray[np.float64], step: float = 1e-6
    ) -> NDArray[np.float64]:
        """The derivatives of the arm's motion over a period by its joint angles, velocities, and torque, (2j, 3j)."""
        point = np.concatenate([q, dq, torque])
        j = self.n_joints
        columns = []
        for k in range(len(point)):
            change = np.zeros(len(point))
            change[k] = step
            ahead, behind = point + change, point - change
            columns.append(
                (
                    np.concatenate(self._arm_motion(ahead[:j], ahead[j : 2 * j], ahead[2 * j :]))
                    - np.concatenate(self._arm_motion(behind[:j], behind[j : 2 * j], behind[2 * j :]))
                )
                / (2 * step)
            )
        return np.column_stack(columns)


def simulated_forces(
    external_force: Callable[..., NDArray[np.float64]], warmup: float, dt: float, n_steps: int
) -> NDArray[np.float64]:
    """The tip force at each of the first ``n_steps`` simulation steps of a run of :func:`arm_esn_ctrl.tracking.track`.

    ``external_force`` is called on the task clock, as ``track`` calls it, at the times
    skelarm's clock gives the steps: from the warm-up's start, ``dt`` added step by
    step, less the ``warmup`` (s). A force that ignores the arm, such as a push, is
    then the run's to the bit: rounded so, a pulse can start or end within a period.
    """
    forces, t = [], 0.0
    for _ in range(n_steps):
        forces.append(np.asarray(external_force(t - warmup, None), dtype=np.float64))
        t += dt
    return np.array(forces)


class StateRecorder(EsnSource):
    """The ESN driving the arm, as :class:`arm_esn_ctrl.tracking.EsnSource`, recording its reservoir's state.

    ``states`` holds, for every reference instant from the warm-up's start, its index
    k (negative in the warm-up) and the state before the ESN's step there.
    """

    def __init__(self, esn: ReachingEsn) -> None:
        super().__init__(esn)
        self.states: list[tuple[int, NDArray[np.float64]]] = []

    def reset(self, start_q: NDArray[np.float64]) -> None:
        super().reset(start_q)
        self.states = []

    def next_posture(self, k: int, q: NDArray[np.float64]) -> NDArray[np.float64]:
        self.states.append((k, self.esn.state().copy()))
        return super().next_posture(k, q)


@dataclass(frozen=True)
class LocalRates:
    """How fast small changes of the state grow or decay along a run, over windows (see :func:`local_rates`)."""

    starts: NDArray[np.intp]  # the step each window starts at
    exponents: NDArray[np.float64]  # (n_windows, n_rates): the largest growth rates of any change (1/s)
    across: NDArray[np.float64]  # (n_windows, n_rates): the largest growth rates of changes across the run (1/s)
    posture: NDArray[np.float64]  # (n_windows, n_joints): how much a change of the output posture is scaled


def local_rates(
    loop: ClosedLoop | RobotLoop, states: NDArray[np.float64], window: int, stride: int, n_rates: int
) -> LocalRates:
    """How fast small changes of the state grow or decay along a run of ``loop``, over windows of ``window`` steps.

    ``states`` are the run's states, one per step, each the image of the one before
    under the map (G, for the ESN on its own). For each window, starting every ``stride`` steps (``window`` must be a
    multiple of ``stride``), Phi is the product of the Jacobians over it, which maps
    a small change of the state at its start to the change at its end. A rate r
    means a change grows as exp(r t): negative rates decay.

    - ``exponents``: the logarithms of Phi's largest singular values, per second;
    - ``across``: those of Phi's largest singular values between the changes
      orthogonal to the run's step at the window's start and at its end. A shift
      of the run in time, along its step, is never corrected nor amplified by an
      autonomous system, beyond the run's own change of speed; the run attracts
      the states near it if the changes across it decay;
    - ``posture``: the singular values of the map from a change of the output
      posture at the window's start, in degrees, made by the smallest change of
      the state, to the change of the output posture at its end, in degrees:
      below 1, a deviation of the posture has shrunk by the window's end. On the
      robot, the posture is the arm's.

    A state whose values are of different kinds, as the robot's, is weighed by the
    loop's ``scale``: sizes and directions are those of the scaled state. The
    eigenvalues, and the rates over long windows, do not depend on it.
    """
    if window % stride:
        msg = "the window must be a whole number of strides"
        raise ValueError(msg)
    n = loop.dimension
    blocks = []  # the product of the Jacobians over each stride
    for begin in range(0, len(states) - stride, stride):
        product = np.eye(n)
        for x in states[begin : begin + stride]:
            product = loop.jacobian(x) @ product
        blocks.append(product)
    scale = loop.scale
    if scale is not None:  # in the scaled state
        blocks = [scale[:, np.newaxis] * block / scale[np.newaxis, :] for block in blocks]
        states = states * scale
    per_window = window // stride
    readout = loop.posture_readout  # how the state moves the output posture (rad)
    if scale is not None:
        readout = readout / scale[np.newaxis, :]
    smallest = np.linalg.pinv(readout)  # the smallest change of the state that moves the output posture
    seconds = window * loop.dt
    starts, exponents, across, posture = [], [], [], []
    for b in range(len(blocks) - per_window + 1):
        phi = np.eye(n)
        for block in blocks[b : b + per_window]:
            phi = block @ phi
        begin, end = b * stride, b * stride + window
        if end + 1 >= len(states):  # the step at the window's end needs the state after it
            break
        step_begin, step_end = states[begin + 1] - states[begin], states[end + 1] - states[end]
        exponents.append(np.log(np.linalg.svd(phi, compute_uv=False)[:n_rates]) / seconds)
        transverse = _without(step_end) @ phi @ _without(step_begin)
        across.append(np.log(np.linalg.svd(transverse, compute_uv=False)[:n_rates]) / seconds)
        posture.append(np.linalg.svd(readout @ phi @ smallest, compute_uv=False))
        starts.append(begin)
    return LocalRates(np.array(starts), np.array(exponents), np.array(across), np.array(posture))


def _without(direction: NDArray[np.float64]) -> NDArray[np.float64]:
    """The projection onto the changes orthogonal to ``direction``: every change, if the run has come to rest."""
    norm = np.linalg.norm(direction)
    if norm == 0.0:  # at a fixed point to the last bit, the run has no direction to leave out
        return np.eye(len(direction))
    unit = direction / norm
    return np.eye(len(unit)) - np.outer(unit, unit)
