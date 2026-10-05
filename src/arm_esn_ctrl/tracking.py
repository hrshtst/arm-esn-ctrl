# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""A robot arm tracking a reference generated step by step from its measured state (Stage 2).

Every reference period Δ (the ESN's sampling period), a reference source receives
the arm's measured joint angles q(t_k) and gives the posture the arm should have
one period later, q̂_{k+1}. Between two such instants, the reference moves in a
straight line from q̂_k to q̂_{k+1}, so its velocity is (q̂_{k+1} - q̂_k)/Δ; its
acceleration is the change of that velocity per period, low-pass filtered. A
tracking law of skelarm, computed torque or joint PD, turns the reference into
joint torques at every simulation step.

There are two sources:

- :class:`EsnSource`: the ESN, driven by the measured joint angles. If the arm
  tracked its reference perfectly, the ESN would generate its autonomous run of
  Stage 1; otherwise it generates from wherever the arm actually is.
- :class:`ReplaySource`: a demonstration replayed by time (the time-indexed
  baseline), which ignores the measured state.

Both start alike. The arm holds its start posture for the ESN's warm-up, during
which the ESN is driven by the measured (held) posture, and the task starts at
t = 0, when the warm-up ends. :func:`track` returns logs on this task clock, so
the warm-up is at negative times.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from skelarm import (
    ComputedTorque,
    Controller,
    JointPD,
    Skeleton,
    StateLog,
    compute_forward_kinematics,
    compute_mass_matrix,
    simulate_controlled,
)

from arm_esn_ctrl.esn import ReachingEsn

# The tracking laws of skelarm that a tracker can use, by their configuration name.
LAWS = {"computed_torque": ComputedTorque, "pd": JointPD}


@dataclass(frozen=True)
class TrackerConfig:
    """How the arm tracks its reference, read from the ``[tracker]`` table of a configuration file."""

    law: str  # "computed_torque" or "pd"
    omega: float | None  # natural frequency of the tracking error (rad/s), critically damped; None if gains are given
    acceleration_filter: float  # time constant of the low-pass filter on the reference acceleration (s)


def tracking_gains(
    config: TrackerConfig, skeleton: Skeleton, posture: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The gains ``(kp, kd)`` that give the tracking error the natural frequency ``config.omega``.

    Computed torque cancels the arm's dynamics, so the error obeys
    ``ë + kd ė + kp e = 0``: ``kp = ω²`` and ``kd = 2ω`` make it critically damped.
    Joint PD does not, so each joint's gains are scaled by its inertia, the diagonal
    ``M_ii`` of the mass matrix at ``posture`` (such as the target posture):
    ``kp = M_ii ω²`` and ``kd = 2 M_ii ω``. That is only approximate, since the
    inertia changes with the posture and couples the joints.
    """
    if config.omega is None:
        msg = "the tracker has no natural frequency omega; its gains must be given directly"
        raise ValueError(msg)
    scale = gain_scale(config.law, skeleton, posture)
    return scale * config.omega**2, 2 * scale * config.omega


def error_dynamics(
    law: str, gains: tuple[NDArray[np.float64], NDArray[np.float64]], skeleton: Skeleton, posture: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The natural frequency (rad/s) and damping ratio of each joint's tracking error under ``gains``.

    The inverse of :func:`tracking_gains`: with ``s`` the joint's scale (1 for
    computed torque, ``M_ii`` at ``posture`` for joint PD), the error obeys
    ``s ë + kd ė + kp e = 0``, so ``ω = sqrt(kp / s)`` and ``ζ = kd / (2 sqrt(kp s))``.
    A damping ratio below 1 makes the error oscillate as it decays.
    """
    kp, kd = gains
    scale = gain_scale(law, skeleton, posture)
    return np.sqrt(kp / scale), kd / (2 * np.sqrt(kp * scale))


def gain_scale(law: str, skeleton: Skeleton, posture: NDArray[np.float64]) -> NDArray[np.float64]:
    """Each joint's gain scale: 1 for computed torque, its inertia ``M_ii`` at ``posture`` for joint PD."""
    if law == "computed_torque":
        return np.ones(skeleton.num_joints)
    if law == "pd":
        model = skeleton.clone()
        for link, angle in zip(model.links[1:], posture, strict=True):
            link.q = float(angle)
        compute_forward_kinematics(model)
        return np.diag(compute_mass_matrix(model)).copy()
    msg = f"unknown tracking law {law!r}; choose from {', '.join(LAWS)}"
    raise ValueError(msg)


class ReferenceSource(Protocol):
    """Gives the posture the arm should have one reference period later."""

    def reset(self, start_q: NDArray[np.float64]) -> None:
        """Prepare a run from the start posture ``start_q``."""
        ...

    def next_posture(self, k: int, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """The posture for time ``(k + 1) Δ``, given the measured joint angles ``q`` at time ``k Δ``.

        ``k`` counts reference periods from the start of the task; it is negative
        during the warm-up, when the tracker ignores the result.
        """
        ...


class EsnSource:
    """The ESN, driven by the arm's measured joint angles."""

    def __init__(self, esn: ReachingEsn) -> None:
        self.esn = esn

    def reset(self, start_q: NDArray[np.float64]) -> None:
        self.esn.reset()

    def next_posture(self, k: int, q: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.esn.step(q)


class ReplaySource:
    """A demonstration ``q_demo``, sampled every reference period, replayed by time.

    After its last sample, the demonstration's final posture is held.
    """

    def __init__(self, q_demo: NDArray[np.float64]) -> None:
        self.q_demo = q_demo

    def reset(self, start_q: NDArray[np.float64]) -> None:
        pass

    def next_posture(self, k: int, q: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.q_demo[min(max(k + 1, 0), len(self.q_demo) - 1)]


class _CurrentReference:
    """The reference at the current simulation step, as skelarm's tracking laws sample it."""

    def __init__(self, q: NDArray[np.float64]) -> None:
        self.q = q
        self.dq = np.zeros_like(q)
        self.ddq = np.zeros_like(q)

    def sample(self, t: float) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        return self.q, self.dq, self.ddq


class ReferenceTracker(Controller):
    """A skelarm controller that tracks the reference a source generates from the measured state.

    Parameters
    ----------
    source : ReferenceSource
        Gives the next posture every reference period.
    config : TrackerConfig
        The tracking law and its filter.
    kp, kd : NDArray[np.float64]
        Per-joint gains of the tracking law (see :func:`tracking_gains`).
    period : float
        Reference period Δ (s); a whole number of simulation steps.
    warmup_steps : int
        Reference periods of the warm-up, during which the arm holds its start posture.
    """

    def __init__(
        self,
        source: ReferenceSource,
        config: TrackerConfig,
        kp: NDArray[np.float64],
        kd: NDArray[np.float64],
        *,
        period: float,
        warmup_steps: int,
    ) -> None:
        if config.law not in LAWS:
            msg = f"unknown tracking law {config.law!r}; choose from {', '.join(LAWS)}"
            raise ValueError(msg)
        self.source = source
        self.config = config
        self.period = period
        self.warmup_steps = warmup_steps
        self.reference = _CurrentReference(np.zeros(len(kp)))
        self.law = LAWS[config.law](self.reference, kp, kd)
        self._filter_gain = period / (config.acceleration_filter + period)
        self._step = 0  # simulation steps since the start of the warm-up
        self._start = np.zeros(len(kp))
        self._q_from = self._q_to = self._start  # the current segment of the reference
        self._velocity = np.zeros(len(kp))

    def reset(self, skeleton: Skeleton) -> None:
        """Start a run from the arm's current posture."""
        self._start = skeleton.q.copy()
        self.source.reset(self._start)
        self._step = 0
        self._q_from = self._q_to = self._start
        self._velocity = np.zeros_like(self._start)
        self.reference.q = self._start
        self.reference.dq = np.zeros_like(self._start)
        self.reference.ddq = np.zeros_like(self._start)

    def update(self, t: float, skeleton: Skeleton, dt: float) -> None:
        """Ask the source for the next posture at each reference instant, and move along the segment."""
        steps_per_period = round(self.period / dt)
        if steps_per_period < 1 or not np.isclose(steps_per_period * dt, self.period):
            msg = f"the reference period {self.period} s must be a whole number of simulation steps of {dt} s"
            raise ValueError(msg)
        within = self._step % steps_per_period
        if within == 0:
            k = self._step // steps_per_period - self.warmup_steps
            q_next = self.source.next_posture(k, skeleton.q.copy())
            if k < 0:
                q_next = self._start  # the arm holds its start posture through the warm-up
            self._q_from, self._q_to = self._q_to, np.asarray(q_next, dtype=np.float64)
            velocity = (self._q_to - self._q_from) / self.period
            acceleration = (velocity - self._velocity) / self.period
            self.reference.ddq = self.reference.ddq + self._filter_gain * (acceleration - self.reference.ddq)
            self.reference.dq = self._velocity = velocity
        self.reference.q = self._q_from + (within / steps_per_period) * (self._q_to - self._q_from)
        self._step += 1

    def control(self, t: float, skeleton: Skeleton) -> NDArray[np.float64]:
        """The tracking law's torque for the current reference."""
        return self.law.control(t, skeleton)

    def log_channels(self) -> dict[str, Any]:
        """The reference ``q_ref`` and the tracking error ``error = q_ref - q``."""
        return self.law.log_channels()


def track(
    skeleton: Skeleton,
    source: ReferenceSource,
    config: TrackerConfig,
    gains: tuple[NDArray[np.float64], NDArray[np.float64]],
    *,
    period: float,
    warmup_steps: int,
    duration: float,
    dt: float,
    enforce_limits: bool = True,
    extra: dict[str, Any] | None = None,
    external_force: Callable[[float, Skeleton], NDArray[np.float64]] | None = None,
) -> StateLog:
    """Simulate the arm tracking ``source`` from its current posture: the warm-up, then ``duration`` of the task.

    Returns the log on the task clock: the warm-up at negative times, the task from 0.
    Besides skelarm's ``q``, ``dq``, and ``tau``, it records the reference ``q_ref``
    and the tracking error ``error`` at every simulation step of ``dt`` seconds.
    An ``external_force`` at the tip (see :mod:`arm_esn_ctrl.disturbances`) is
    called with the time on the task clock and recorded as ``ext_force``.
    """
    tracker = ReferenceTracker(source, config, *gains, period=period, warmup_steps=warmup_steps)
    warmup = warmup_steps * period
    force = None if external_force is None else _delayed(external_force, warmup)
    log = simulate_controlled(
        skeleton,
        tracker,
        duration=warmup + duration,
        dt=dt,
        enforce_limits=enforce_limits,
        extra=extra,
        external_force=force,
    )
    return shift_times(log, -warmup)


def _delayed(
    external_force: Callable[[float, Skeleton], NDArray[np.float64]], delay: float
) -> Callable[[float, Skeleton], NDArray[np.float64]]:
    """``external_force`` on skelarm's clock, which starts ``delay`` seconds before the task."""
    return lambda t, state: external_force(t - delay, state)


def shift_times(log: StateLog, offset: float) -> StateLog:
    """A copy of ``log`` with every time shifted by ``offset``."""
    shifted = StateLog(log.build_skeleton(), producer=log.producer, channel_meta=log.channel_meta, extra=log.extra)
    channels = {name: log.channel(name) for name in log.channel_names}
    for i, t in enumerate(log.times):
        shifted.record(float(t) + offset, **{name: values[i] for name, values in channels.items()})
    return shifted


def task_joint_angles(log: StateLog, period: float, duration: float) -> NDArray[np.float64]:
    """The joint angles of a log on the task clock, sampled every ``period`` from 0 to ``duration``."""
    times = period * np.arange(round(duration / period) + 1)
    q = log.channel("q").reshape(len(log.times), -1)
    return np.column_stack([np.interp(times, log.times, q[:, j]) for j in range(q.shape[1])])


def nearest_demonstration(start_q: NDArray[np.float64], demos: dict[str, NDArray[np.float64]]) -> str:
    """The name of the demonstration whose start posture is nearest ``start_q`` (in joint space)."""
    return min(demos, key=lambda name: float(np.linalg.norm(demos[name][0] - start_q)))
