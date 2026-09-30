# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Metrics of reaching motions.

Human point-to-point reaching has a roughly straight hand path and a smooth,
single-peaked (bell-shaped) hand speed profile, well described by the minimum-jerk
model (Flash and Hogan, 1985). :func:`reach_metrics` measures how close a reach
comes to that description.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# The movement lasts while the hand speed exceeds this fraction of its peak.
MOVING_THRESHOLD = 0.05
# Speed peaks below this fraction of the highest peak are ignored.
PEAK_THRESHOLD = 0.1
# Normalized time at which a minimum-jerk speed profile first exceeds
# MOVING_THRESHOLD: the solution of 16 tau^2 (1 - tau)^2 = MOVING_THRESHOLD.
_MINIMUM_JERK_ONSET = 0.5 * (1.0 - np.sqrt(1.0 - np.sqrt(MOVING_THRESHOLD)))


def minimum_jerk_profile(tau: NDArray[np.float64]) -> NDArray[np.float64]:
    """Speed profile of a minimum-jerk reach at normalized time ``tau`` in ``[0, 1]``, with peak 1."""
    return 16.0 * tau**2 * (1.0 - tau) ** 2


def hand_speed(times: NDArray[np.float64], hand: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the hand speed from hand positions ``hand`` (shape ``(n, 2)``) over ``times``."""
    return np.linalg.norm(np.gradient(hand, times, axis=0), axis=1)


def movement_interval(speed: NDArray[np.float64]) -> tuple[int, int]:
    """Return the first and last sample indices where the hand moves (speed above 5 % of peak)."""
    moving = np.nonzero(speed > MOVING_THRESHOLD * speed.max())[0]
    return int(moving[0]), int(moving[-1])


def speed_profile(
    times: NDArray[np.float64], speed: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the speed profile of a movement against normalized time, ready to compare with minimum jerk.

    The profile is the speed during the movement divided by its peak. The movement
    starts and ends where the speed crosses 5 % of its peak, and a minimum-jerk
    reach crosses that level at a normalized time slightly after 0 and before 1.
    The movement is therefore mapped onto that part of the normalized time, so a
    perfect minimum-jerk reach lies exactly on :func:`minimum_jerk_profile`.
    """
    start, end = movement_interval(speed)
    fraction = (times[start : end + 1] - times[start]) / (times[end] - times[start])
    tau = _MINIMUM_JERK_ONSET + (1.0 - 2.0 * _MINIMUM_JERK_ONSET) * fraction
    return tau, speed[start : end + 1] / speed.max()


def reach_metrics(times: NDArray[np.float64], hand: NDArray[np.float64], target: ArrayLike) -> dict[str, float]:
    """Measure a reach from its hand positions ``hand`` (shape ``(n, 2)``) over ``times``.

    Returns
    -------
    dict[str, float]
        - ``movement_time``: how long the hand speed stays above 5 % of its peak (s).
        - ``peak_speed``: the highest hand speed (m/s).
        - ``peak_timing``: when the speed peaks, in the normalized time of
          :func:`speed_profile`; 0.5 is a symmetric bell.
        - ``speed_profile_error``: RMS difference between the speed profile and
          the minimum-jerk profile (both with peak 1); 0 is a perfect match.
        - ``speed_peaks``: the number of peaks in the speed profile; 1 is smooth.
        - ``path_deviation``: the largest distance of the hand from the straight
          line between start and target, relative to the reach distance.
        - ``overshoot``: how far the hand passes the target along that line,
          relative to the reach distance.
        - ``final_error``: the distance from the hand to the target at the end (m).
    """
    target = np.asarray(target, dtype=np.float64)
    speed = hand_speed(times, hand)
    start, end = movement_interval(speed)
    tau, profile = speed_profile(times, speed)

    distance = float(np.linalg.norm(target - hand[0]))
    direction = (target - hand[0]) / distance
    along = (hand - hand[0]) @ direction
    across = (hand - hand[0]) @ np.array([-direction[1], direction[0]])

    inner = speed[1:-1]
    is_peak = (inner > speed[:-2]) & (inner >= speed[2:]) & (inner > PEAK_THRESHOLD * speed.max())

    return {
        "movement_time": float(times[end] - times[start]),
        "peak_speed": float(speed.max()),
        "peak_timing": float(tau[np.argmax(profile)]),
        "speed_profile_error": float(np.sqrt(np.mean((profile - minimum_jerk_profile(tau)) ** 2))),
        "speed_peaks": float(np.count_nonzero(is_peak)),
        "path_deviation": float(np.max(np.abs(across)) / distance),
        "overshoot": float(max(0.0, along.max() - distance) / distance),
        "final_error": float(np.linalg.norm(hand[-1] - target)),
    }
