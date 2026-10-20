# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""A trained ESN screened by where it can rest and how it sets off, without the robot.

Fed its own output, a trained ESN rests at the fixed points x* of its update G (see
:mod:`arm_esn_ctrl.dynamics`): x* = F(x*, u(x*)), with u(x*) its output. Such an x*
is also the fixed point of the reservoir held at its own output. Where the held
reservoir has one fixed point x_h(q) at each posture q, the fixed points of G are
the postures at which the held reservoir's output is that posture itself, the zeros
of

    R(q) = posture(x_h(q)) - q,

a map of the joint angles alone, which a grid over both joints' range covers. R(q)
is also how far the ESN's output leads a posture held at q once its reservoir has
settled there: on the robot, how far its reference leads an arm held still at q.

- :func:`held_fixed_points`: x_h at many postures at once;
- :func:`posture_map`: x_h and R over a grid of postures;
- :func:`find_fixed_points`: the zeros of R, each refined by Newton's method on G;
- :func:`fixed_points_at_rest`: the fixed points where runs come to rest, which
  the grid misses where R is nearly flat;
- :func:`autonomous_runs`: the ESN's runs on its own from many start postures at
  once, as :meth:`arm_esn_ctrl.esn.ReachingEsn.generate` runs each.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from arm_esn_ctrl.dynamics import ClosedLoop

SETTLE_STEPS = 200  # the held reservoir runs this long from the reset before Newton's method refines it
CHUNK = 64  # postures whose Newton steps are solved together


def held_fixed_points(
    loop: ClosedLoop,
    q: NDArray[np.float64],
    x: NDArray[np.float64] | None = None,
    tolerance: float = 1e-12,
    steps: int = 30,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """The fixed point of the reservoir held at each posture of ``q`` (rows, rad), and whether it converged.

    From ``x`` (one state per posture; by default the reset state, zero), the held
    reservoir first runs :data:`SETTLE_STEPS` steps, then Newton's method on F(., u)
    refines each state until no entry moves by more than ``tolerance`` within
    ``steps`` steps.
    """
    u = loop.normalize(q)
    x = np.zeros((len(q), loop.n_neurons)) if x is None else np.array(x, dtype=np.float64)
    for _ in range(SETTLE_STEPS):
        x = loop.held(x, u)
    converged = np.zeros(len(q), dtype=bool)
    for begin in range(0, len(q), CHUNK):
        rows = slice(begin, begin + CHUNK)
        x[rows], converged[rows] = _held_newton(loop, x[rows], u[rows], tolerance, steps)
    return x, converged


@dataclass(frozen=True)
class PostureMap:
    """The held reservoir's fixed point and the step R(q) of the ESN from rest, over a grid of postures."""

    grid_deg: NDArray[np.float64]  # (m,): the grid's joint angles, the same for each joint (deg)
    held: NDArray[np.float64]  # (m, m, n): x_h at each posture; the entry (i, j) is at (grid[i], grid[j])
    residual: NDArray[np.float64]  # (m, m, n_joints): R(q) (rad)
    converged: NDArray[np.bool_]  # (m, m): whether Newton's method converged at each posture

    def postures(self) -> NDArray[np.float64]:
        """The grid's postures (rad), one row per posture, in the order of ``held`` flattened."""
        q1, q2 = np.meshgrid(np.radians(self.grid_deg), np.radians(self.grid_deg), indexing="ij")
        return np.column_stack([q1.ravel(), q2.ravel()])


def posture_map(loop: ClosedLoop, grid_deg: NDArray[np.float64]) -> PostureMap:
    """x_h and R over the grid of both joints at the angles ``grid_deg`` (a two-joint arm)."""
    m = len(grid_deg)
    q1, q2 = np.meshgrid(np.radians(grid_deg), np.radians(grid_deg), indexing="ij")
    q = np.column_stack([q1.ravel(), q2.ravel()])
    held, converged = held_fixed_points(loop, q)
    residual = loop.posture(held) - q
    return PostureMap(
        np.asarray(grid_deg, dtype=np.float64),
        held.reshape(m, m, -1),
        residual.reshape(m, m, -1),
        converged.reshape(m, m),
    )


@dataclass(frozen=True)
class FixedPoints:
    """The fixed points of G found from a posture map, and what the search tried."""

    states: list[NDArray[np.float64]]  # each fixed point's state, in the order found
    candidates: int  # the grid cells over whose corners both components of R change sign
    empty: int  # the candidates near which Newton's method on R found no zero


def find_fixed_points(loop: ClosedLoop, posture_map: PostureMap, distinct: float = 1e-8) -> FixedPoints:
    """The fixed points of G: the zeros of R near the cells of ``posture_map`` where both its components change sign.

    From the zero of the plane fitted to R at each such cell's corners, Newton's
    method on R (:func:`map_zero`) finds the zero, if it stays within a cell's width
    of the cell; the held reservoir's fixed point there is a fixed point of G, and
    Newton's method on G checks it. Fixed points whose states differ by no more than
    ``distinct`` in every entry are one.
    """
    r = posture_map.residual
    grid = np.radians(posture_map.grid_deg)
    states: list[NDArray[np.float64]] = []
    candidates = empty = 0
    for i in range(len(grid) - 1):
        for j in range(len(grid) - 1):
            corners = r[i : i + 2, j : j + 2].reshape(4, -1)
            if not all(corners[:, k].min() <= 0.0 <= corners[:, k].max() for k in range(corners.shape[1])):
                continue
            candidates += 1
            q1, q2 = grid[i : i + 2], grid[j : j + 2]
            nearest = np.unravel_index(int(np.argmin(np.linalg.norm(corners, axis=1))), (2, 2))
            width = np.array([q1[1] - q1[0], q2[1] - q2[0]])
            low, high = np.array([q1[0], q2[0]]) - width, np.array([q1[1], q2[1]]) + width
            x = map_zero(loop, _cell_zero(q1, q2, corners), posture_map.held[i + nearest[0], j + nearest[1]], low, high)
            if x is None:
                empty += 1
                continue
            x, converged = loop.fixed_point(x)
            if not converged:
                empty += 1
                continue
            if all(np.abs(x - found).max() > distinct for found in states):
                states.append(x)
    return FixedPoints(states, candidates, empty)


def map_zero(
    loop: ClosedLoop,
    q: NDArray[np.float64],
    x: NDArray[np.float64],
    low: NDArray[np.float64],
    high: NDArray[np.float64],
    tolerance: float = 1e-12,
    steps: int = 30,
) -> NDArray[np.float64] | None:
    """The held reservoir's fixed point at a zero of R, by Newton's method on R from the posture ``q``.

    ``x`` is a state near the held reservoir's fixed point at ``q``. R's Jacobian is
    exact: holding at u = (q - c) / h, the fixed point moves as
    dx/du = (I - dF/dx)^-1 dF/du, and the output posture as h W_out^T dx/du. Returns
    None if a step leaves the box from ``low`` to ``high`` (rad) or Newton's method on
    R or on the held reservoir does not converge.
    """
    a = loop.leak_rate
    for _ in range(steps):
        u = loop.normalize(q)
        x, converged = loop.fixed_point(x, u)
        if not converged:
            return None
        residual = loop.posture(x) - q
        slope = 1.0 - np.tanh(u @ loop.inputs + loop.bias + x @ loop.recurrent) ** 2
        held = (1.0 - a) * np.eye(loop.n_neurons) + a * slope[:, np.newaxis] * loop.recurrent.T  # dF/dx
        moves = np.linalg.solve(np.eye(loop.n_neurons) - held, a * slope[:, np.newaxis] * loop.inputs.T)  # dx/du
        jacobian = loop.half_range[:, np.newaxis] * (loop.readout.T @ moves) / loop.half_range - np.eye(len(q))
        step = np.linalg.solve(jacobian, -residual)
        q = q + step
        if np.any(q < low) or np.any(q > high):
            return None
        if np.abs(step).max() <= tolerance:
            return loop.fixed_point(x, loop.normalize(q))[0]
    return None


def spectrum(loop: ClosedLoop, x: NDArray[np.float64]) -> NDArray[np.complex128]:
    """The eigenvalues of G's Jacobian at the state ``x``, the largest modulus first."""
    eigenvalues = np.linalg.eigvals(loop.jacobian(x))
    return np.asarray(eigenvalues[np.argsort(-np.abs(eigenvalues))], dtype=np.complex128)


def fixed_points_at_rest(
    loop: ClosedLoop,
    q: NDArray[np.float64],
    x: NDArray[np.float64],
    known: list[NDArray[np.float64]],
    still: float = np.radians(0.5),
    window: int = 100,
    distinct: float = 1e-8,
) -> list[NDArray[np.float64]]:
    """The fixed points of G where runs come to rest, other than those in ``known``.

    ``q`` and ``x`` are runs' joint angles and their states at the end, as
    :func:`autonomous_runs` returns them. A run comes to rest if no joint moves by
    more than ``still`` (rad) over its last ``window`` steps; Newton's method on G
    starts from its state at the end, unless its end posture lies within ``still``
    of a fixed point's already found. Near an attracting fixed point where R is
    nearly flat, as where the ESN learned to hold still, Newton's method on R can
    step far from the zero, and these find it.
    """
    found: list[NDArray[np.float64]] = []
    postures = [loop.posture(f) for f in known]
    at_rest = np.ptp(q[:, -window:], axis=1).max(axis=1) <= still
    for state, end in zip(x[at_rest], q[at_rest, -1], strict=True):
        if any(np.abs(end - p).max() <= still for p in postures):
            continue
        fixed, converged = loop.fixed_point(state)
        if converged and all(np.abs(fixed - f).max() > distinct for f in known + found):
            found.append(fixed)
            postures.append(loop.posture(fixed))
    return found


def autonomous_runs(
    loop: ClosedLoop, start_q: NDArray[np.float64], warmup_steps: int, n_steps: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The joint angles of the ESN's runs on its own from each start posture of ``start_q`` (rows, rad).

    As :meth:`arm_esn_ctrl.esn.ReachingEsn.generate` runs each: from the reset state,
    the start posture held for ``warmup_steps`` steps and one more, then the ESN fed
    its own output. Returns the joint angles, ``(len(start_q), n_steps + 1,
    n_joints)``, the start posture at time 0, and the states at the end, from which
    the next output would come.
    """
    u = loop.normalize(start_q)
    x = np.zeros((len(start_q), loop.n_neurons))
    for _ in range(warmup_steps + 1):
        x = loop.held(x, u)
    q = np.empty((len(start_q), n_steps + 1, start_q.shape[1]))
    q[:, 0] = start_q
    for k in range(1, n_steps + 1):
        q[:, k] = loop.posture(x)
        x = loop.advance(x)
    return q, x


def _held_newton(
    loop: ClosedLoop, x: NDArray[np.float64], u: NDArray[np.float64], tolerance: float, steps: int
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Newton's method on F(., u) for each row of ``x`` and ``u`` at once."""
    a, identity = loop.leak_rate, np.eye(loop.n_neurons)
    converged = np.zeros(len(x), dtype=bool)
    for _ in range(steps):
        active = ~converged
        if not active.any():
            break
        xa, ua = x[active], u[active]
        drive = np.tanh(ua @ loop.inputs + loop.bias + xa @ loop.recurrent)
        residual = a * (drive - xa)  # F(x, u) - x
        # dF/dx - I = a (diag(1 - drive^2) W^T - I), one matrix per row
        jacobian = a * ((1.0 - drive**2)[:, :, np.newaxis] * loop.recurrent.T[np.newaxis] - identity)
        try:
            step = np.linalg.solve(jacobian, -residual[:, :, np.newaxis])[:, :, 0]
        except np.linalg.LinAlgError:
            break
        x[active] = xa + step
        converged[np.flatnonzero(active)[np.abs(step).max(axis=1) <= tolerance]] = True
    return x, converged & np.all(np.isfinite(x), axis=1)


def _cell_zero(q1: NDArray[np.float64], q2: NDArray[np.float64], corners: NDArray[np.float64]) -> NDArray[np.float64]:
    """The zero of the plane fitted to R at a cell's four corners, kept inside the cell."""
    points = np.array([[a, b] for a in q1 for b in q2])  # the corners in the order of ``corners``
    design = np.column_stack([np.ones(4), points - points.mean(axis=0)])
    coefficients, *_ = np.linalg.lstsq(design, corners, rcond=None)  # R ~ c0 + (q - mean) C
    try:
        zero = points.mean(axis=0) + np.linalg.solve(coefficients[1:].T, -coefficients[0])
    except np.linalg.LinAlgError:
        zero = points.mean(axis=0)
    return np.clip(zero, [q1[0], q2[0]], [q1[1], q2[1]])
