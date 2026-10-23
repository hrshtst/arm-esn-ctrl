# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Metrics of reaching motions.

Human point-to-point reaching has a roughly straight hand path and a smooth,
single-peaked (bell-shaped) hand speed profile, well described by the minimum-jerk
model (Flash and Hogan, 1985). :func:`reach_metrics` measures how close a reach
comes to that description.

A reach is split into two phases at the **arrival time** t_a, the first time the
hand comes within the goal radius r of the target (:func:`arrival_index`): the
reach phase [0, t_a] and the hold window [t_a, t_a + T_h], where T_h is the hold
duration. :func:`hold_metrics` measures whether the hand dwells at the target
during the hold window; it needs no reference trajectory.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from skelarm import butterworth_lowpass

# The movement lasts while the hand speed exceeds this fraction of its peak.
MOVING_THRESHOLD = 0.05
# Speed peaks below this fraction of the highest peak are ignored.
PEAK_THRESHOLD = 0.1
# The hand has started to move once it is this far from where it started (m).
MOVED = 0.005
# Paths are compared at points this far apart along their length (m): a polyline through
# them strays from a bend of 5 cm radius by under 0.1 mm, and comparing paths stays fast.
PATH_SPACING = 0.005
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

    return {
        "movement_time": float(times[end] - times[start]),
        "peak_speed": float(speed.max()),
        "peak_timing": float(tau[np.argmax(profile)]),
        "speed_profile_error": float(np.sqrt(np.mean((profile - minimum_jerk_profile(tau)) ** 2))),
        "speed_peaks": float(count_speed_peaks(times, hand)),
        "path_deviation": float(np.max(np.abs(across)) / distance),
        "overshoot": float(max(0.0, along.max() - distance) / distance),
        "final_error": float(np.linalg.norm(hand[-1] - target)),
    }


def count_speed_peaks(times: NDArray[np.float64], hand: NDArray[np.float64]) -> int:
    """Return the number of peaks in the hand speed, ignoring those below 10 % of the highest; 1 is smooth."""
    speed = hand_speed(times, hand)
    inner = speed[1:-1]
    is_peak = (inner > speed[:-2]) & (inner >= speed[2:]) & (inner > PEAK_THRESHOLD * speed.max())
    return int(np.count_nonzero(is_peak))


def onset_index(hand: NDArray[np.float64], distance: float = MOVED) -> int | None:
    """Return the first sample where the hand is more than ``distance`` from where it started, or None if never.

    It marks when a movement begins, whether it starts smoothly or with a jump.
    """
    moved = np.linalg.norm(hand - hand[0], axis=1) > distance
    return int(np.argmax(moved)) if moved.any() else None


def jitter(values: NDArray[np.float64], window: int) -> float:
    """Return the RMS deviation of ``values`` from their centered moving average over ``window`` samples.

    ``values`` is a sequence of vectors, shaped ``(n, d)``, such as joint angles, and
    ``window`` is odd. The deviation is the vector's length, averaged over the samples
    whose window fits: a straight ramp has none, and noise faster than the window
    shows almost in full. It is in the units of ``values``.
    """
    kernel = np.ones(window) / window
    smooth = np.column_stack([np.convolve(values[:, j], kernel, mode="valid") for j in range(values.shape[1])])
    half = window // 2
    residual = values[half : len(values) - half] - smooth
    return float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))


def path_distance(path: NDArray[np.float64], reference: NDArray[np.float64]) -> float:
    """Return how far ``path`` strays from the ``reference`` path, regardless of timing.

    Both are sequences of points, shaped ``(n, 2)``. The result is the largest
    distance from a point of ``path`` to the nearest point on the polyline through
    ``reference``.
    """
    return float(np.max(distances_to_path(path, reference)))


def path_rmse(path: NDArray[np.float64], reference: NDArray[np.float64], spacing: float = PATH_SPACING) -> float:
    """Return the RMS distance between the ``path`` and the ``reference`` path, regardless of timing.

    Both are sequences of points, shaped ``(n, d)``. Each is resampled every
    ``spacing`` along its length, so that its points weigh by length rather than by
    time: a slow stretch or a pause adds nothing. The result is the root mean square,
    over the points of both, of each point's distance to the other path: a path that
    strays counts by how far it strays, and one that skips part of the reference
    counts by how far that part lies from it.
    """
    a, b = resample_by_length(path, spacing), resample_by_length(reference, spacing)
    distances = np.concatenate([distances_to_path(a, b), distances_to_path(b, a)])
    return float(np.sqrt(np.mean(distances**2)))


def resample_by_length(points: NDArray[np.float64], spacing: float) -> NDArray[np.float64]:
    """Return points every ``spacing`` along the polyline through ``points``, from its start to its end.

    A polyline of no length (points that never move) gives its start twice, so that
    the result is always a polyline.
    """
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    moving = steps > 0
    points = points[np.concatenate([[True], moving])]  # without repeated points, so that the length increases
    length = np.concatenate([[0.0], np.cumsum(steps[moving])])
    if length[-1] == 0:
        return np.vstack([points[:1], points[:1]])
    along = np.append(np.arange(0.0, length[-1], spacing), length[-1])
    return np.column_stack([np.interp(along, length, points[:, j]) for j in range(points.shape[1])])


def distances_to_path(points: NDArray[np.float64], reference: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the distance from each of ``points`` to the nearest point on the polyline through ``reference``.

    Both are sequences of points of any dimension, shaped ``(n, d)``, such as hand
    positions or joint angles.
    """
    return _nearest_on_path(points, reference)[2]


def route_spread(paths: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Return how far each point of each path is from the routes the other paths take, regardless of timing.

    ``paths`` are sequences of points of any dimension, each shaped ``(n, d)``. For
    each point, the result is the median over the other paths of its distance to
    the nearest point on their polylines, shaped ``(len(paths), n)``. Runs that
    gather onto a common route lose their spread from where they join it, however
    late or early they get there; the median ignores the few that do not.
    """
    spread = np.empty((len(paths), len(paths[0])))
    for i, path in enumerate(paths):
        others = [distances_to_path(path, other) for j, other in enumerate(paths) if j != i]
        spread[i] = np.median(others, axis=0)
    return spread


def join_index(distances: NDArray[np.float64], threshold: float) -> int | None:
    """Return the first index from which ``distances`` stay below ``threshold``, or None if they end above it."""
    above = np.flatnonzero(distances >= threshold)
    if above.size == 0:
        return 0
    return None if above[-1] == len(distances) - 1 else int(above[-1]) + 1


def disturbance_returns(
    times: NDArray[np.float64], distances: NDArray[np.float64], spans: list[tuple[float, float]], tolerance: float
) -> list[tuple[float, float]]:
    """Return how far each disturbance sends the hand from a path, and how soon the hand comes back.

    ``distances`` are the hand's distances from the path at ``times``, and ``spans``
    when each disturbance acts, as ``(onset, end)``. A disturbance's window lasts
    from its onset until the next disturbance starts, or to the last sample. For
    each span, the result is the largest distance in its window, and the time from
    its end until the hand is back within ``tolerance`` of the path for the rest of
    the window: 0 if it is never farther after the end, NaN if it is not back by
    the end of the window.
    """
    nan = float("nan")
    returns = []
    for onset, end in spans:
        later = [other for other, _ in spans if other > onset]
        stop = min(later) if later else np.inf
        window = (times > onset - 1e-9) & (times < stop - 1e-9)
        after = window & (times > end - 1e-9)
        farthest = float(distances[window].max()) if window.any() else nan
        outside = np.flatnonzero(distances[after] > tolerance)
        if not after.any() or (outside.size and outside[-1] == np.count_nonzero(after) - 1):
            back = nan  # the window ends before the disturbance does, or with the hand still away
        else:
            back = float(times[after][outside[-1] + 1] - end) if outside.size else 0.0
        returns.append((farthest, back))
    return returns


def disturbance_ringing(
    times: NDArray[np.float64],
    points: NDArray[np.float64],
    spans: list[tuple[float, float]],
    cutoff_hz: float,
    window: float,
) -> list[float]:
    """Return how much the ``points``, such as the hand, ring after each disturbance.

    The ringing is their motion above ``cutoff_hz``: the points, sampled at
    ``times`` and shaped ``(n, d)``, minus their zero-phase Butterworth low-pass of
    order 2 (skelarm's, applied forward and backward). Pick a cutoff between the
    frequencies of the motion itself and of a tracker's ringing. For each span
    ``(onset, end)`` of ``spans``, when a disturbance acts, the result is the RMS
    size of the ringing over ``window`` seconds from its end, or until the next
    disturbance starts if that is sooner: NaN if no sample is left. The zero-phase
    filter spreads a sudden motion over about ``1 / cutoff_hz`` on either side, so
    a window that ends within that much of the next disturbance catches the start of
    the response to it.
    """
    period = float(np.median(np.diff(times)))
    ringing = np.linalg.norm(points - butterworth_lowpass(points, period, cutoff_hz=cutoff_hz), axis=1)
    onsets = [onset for onset, _ in spans]
    result = []
    for onset, end in spans:
        stop = min([end + window] + [other for other in onsets if other > onset])
        after = (times > end - 1e-9) & (times < stop - 1e-9)
        result.append(float(np.sqrt(np.mean(ringing[after] ** 2))) if after.any() else float("nan"))
    return result


def disturbance_following(
    times: NDArray[np.float64],
    points: NDArray[np.float64],
    reference: NDArray[np.float64],
    taught: NDArray[np.float64],
    spans: list[tuple[float, float]],
    after: float,
) -> list[float]:
    """Return how much of each disturbance's displacement of the ``points``, such as the hand, the ``reference`` shares.

    ``points`` are where the arm is, ``reference`` where its reference would put it,
    and ``taught`` where the taught motion is at the same ``times``, as a replay of
    it by time would put the reference; all are shaped ``(n, d)``. For each span
    ``(onset, end)`` of ``spans``, over the window from its onset until ``after``
    seconds past its end, or until the next disturbance starts if that is sooner,
    the arm's and the reference's displacements are their differences from the
    taught points, less those at the onset, so that a lag from before cancels. The
    result is the least-squares gain of the reference's displacement on the arm's:
    0 for a reference that keeps to the taught motion whatever the arm does, as a
    replay's, and 1 for one that goes wherever the arm goes. Under a block it
    measures how far the reference waits for the held arm; under a push, how far it
    yields to the push. NaN if the arm is not displaced or no sample is left.
    """
    arm, ref = points - taught, reference - taught
    onsets = [onset for onset, _ in spans]
    result = []
    for onset, end in spans:
        stop = min([end + after] + [other for other in onsets if other > onset])
        window = np.flatnonzero((times > onset - 1e-9) & (times < stop - 1e-9))
        if window.size == 0:
            result.append(float("nan"))
            continue
        arm_moved, ref_moved = arm[window] - arm[window[0]], ref[window] - ref[window[0]]
        size = float(np.sum(arm_moved**2))
        result.append(float(np.sum(ref_moved * arm_moved)) / size if size > 0 else float("nan"))
    return result


def motor_effort(
    times: NDArray[np.float64], torques: NDArray[np.float64], velocities: NDArray[np.float64]
) -> tuple[float, float, float]:
    """Return how hard the motors work: the accumulated torque, the squared torque, and the positive work.

    ``torques`` (N m) and ``velocities`` (rad/s) are the joints' at ``times``, evenly
    spaced, shaped ``(n, joints)``. Each is an integral over ``times`` (a sum over
    the samples, times their spacing) of a sum over the joints:

    - the accumulated torque, of |τ_i| (N m s);
    - the squared torque, of τ_i² (N² m² s): a motor's current is proportional to
      its torque, and its copper losses to the square of the current;
    - the positive work, of max(τ_i q̇_i, 0) (J): the mechanical work the motors
      do, each joint on its own, none of it recovered while a joint is driven
      backward. Pressing against a block without moving costs no work, while the
      squared torque grows.
    """
    spacing = float(np.diff(times).mean())
    power = torques * velocities
    return (
        float(np.abs(torques).sum() * spacing),
        float(np.sum(torques**2) * spacing),
        float(np.clip(power, 0.0, None).sum() * spacing),
    )


def path_progress(points: NDArray[np.float64], reference: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return how far along the ``reference`` path the nearest point to each of ``points`` lies.

    The progress is the length of the polyline through ``reference`` up to that
    nearest point, as a fraction of its whole length: 0 at its first point, 1 at its
    last. Both are sequences of points of any dimension, shaped ``(n, d)``; where
    the reference rests, it adds no length.
    """
    k, fraction, _ = _nearest_on_path(points, reference)
    lengths = np.linalg.norm(np.diff(reference, axis=0), axis=1)
    before = np.concatenate([[0.0], np.cumsum(lengths)])  # the length up to each reference point
    return (before[k] + fraction * lengths[k]) / before[-1]


def _nearest_on_path(
    points: NDArray[np.float64], reference: NDArray[np.float64]
) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.float64]]:
    """For each of ``points``, the nearest point on the polyline through ``reference``.

    Returns the index of the segment it lies on, how far along that segment (0 to 1),
    and the distance to it. Of several equally near points, the first is taken.
    """
    start, segment = reference[:-1], np.diff(reference, axis=0)
    length2 = np.maximum(np.sum(segment**2, axis=1), np.finfo(float).tiny)
    offset = points[:, np.newaxis, :] - start[np.newaxis, :, :]
    fraction = np.clip(np.sum(offset * segment, axis=2) / length2, 0.0, 1.0)
    nearest = start + fraction[..., np.newaxis] * segment
    distance = np.linalg.norm(points[:, np.newaxis, :] - nearest, axis=2)
    k = np.argmin(distance, axis=1)
    rows = np.arange(len(points))
    return k, fraction[rows, k], distance[rows, k]


def arrival_index(hand: NDArray[np.float64], target: ArrayLike, radius: float) -> int | None:
    """Return the first sample index where the hand is within ``radius`` of the target, or None if never."""
    inside = np.linalg.norm(hand - np.asarray(target, dtype=np.float64), axis=1) <= radius
    return int(np.argmax(inside)) if inside.any() else None


def hold_metrics(
    times: NDArray[np.float64], hand: NDArray[np.float64], target: ArrayLike, radius: float, hold: float
) -> dict[str, float | bool]:
    """Measure whether the hand dwells within ``radius`` of the target after it arrives.

    The hold window is [t_a, t_a + ``hold``], cut at the end of the trajectory if
    it does not fit.

    Returns
    -------
    dict[str, float | bool]
        - ``arrived``: whether the hand ever comes within ``radius`` of the target.
        - ``arrival_time_s``: t_a (NaN if the hand never arrives).
        - ``left_goal``: whether the hand leaves the goal radius during the hold window.
        - ``hold_error_m``: the distance from the hand to the target at the end of the
          hold window (NaN if the hand never arrives).
        - ``hold_observed_s``: how much of the hold window the trajectory covers, which
          is less than ``hold`` when the trajectory ends early (0 if the hand never arrives).
        - ``success``: arrived and did not leave the goal radius during the hold window.
    """
    arrival = arrival_index(hand, target, radius)
    if arrival is None:
        nan = float("nan")
        return {
            "arrived": False,
            "arrival_time_s": nan,
            "left_goal": False,
            "hold_error_m": nan,
            "hold_observed_s": 0.0,
            "success": False,
        }
    distance = np.linalg.norm(hand - np.asarray(target, dtype=np.float64), axis=1)
    # The tolerance keeps the last sample of the window when t_a + hold rounds just below it.
    window = (times >= times[arrival]) & (times <= times[arrival] + hold + 1e-9)
    left_goal = bool(np.any(distance[window] > radius))
    return {
        "arrived": True,
        "arrival_time_s": float(times[arrival]),
        "left_goal": left_goal,
        "hold_error_m": float(distance[window][-1]),
        "hold_observed_s": float(times[window][-1] - times[arrival]),
        "success": not left_goal,
    }
