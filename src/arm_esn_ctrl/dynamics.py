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
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from arm_esn_ctrl.esn import ReachingEsn


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
class LocalRates:
    """How fast small changes of the state grow or decay along a run, over windows (see :func:`local_rates`)."""

    starts: NDArray[np.intp]  # the step each window starts at
    exponents: NDArray[np.float64]  # (n_windows, n_rates): the largest growth rates of any change (1/s)
    across: NDArray[np.float64]  # (n_windows, n_rates): the largest growth rates of changes across the run (1/s)
    posture: NDArray[np.float64]  # (n_windows, n_joints): how much a change of the output posture is scaled


def local_rates(loop: ClosedLoop, states: NDArray[np.float64], window: int, stride: int, n_rates: int) -> LocalRates:
    """How fast small changes of the state grow or decay along a run of ``loop``, over windows of ``window`` steps.

    ``states`` are the run's states, one per step, each the image of the one before
    under G. For each window, starting every ``stride`` steps (``window`` must be a
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
      below 1, a deviation of the posture has shrunk by the window's end.
    """
    if window % stride:
        msg = "the window must be a whole number of strides"
        raise ValueError(msg)
    n = loop.n_neurons
    blocks = []  # the product of the Jacobians over each stride
    for begin in range(0, len(states) - stride, stride):
        product = np.eye(n)
        for x in states[begin : begin + stride]:
            product = loop.jacobian(x) @ product
        blocks.append(product)
    per_window = window // stride
    readout = loop.half_range[:, np.newaxis] * loop.readout.T  # how the state moves the output posture (rad)
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
