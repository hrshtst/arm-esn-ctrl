# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Disturbances of Stage 2: forces at the arm's tip, scripted on the task clock.

A disturbance is configured by a ``[disturbance]`` table, and several by
``[[disturbance]]`` tables, whose forces add up:

- ``type = "push"``: a constant force of ``force`` newtons from ``onset`` for
  ``duration`` seconds, in a direction relative to the reach (the straight line
  from the start hand position to the target, fixed for the run): ``angle_deg``
  degrees counterclockwise from the direction toward the target, or a
  ``direction``: ``"forward"`` along the reach toward the target (0 deg),
  ``"across"`` it (90 deg, the default), or ``"backward"`` away from the target
  (180 deg).
- ``type = "block"``: from ``onset`` until ``release``, a stiff spring-damper
  (``stiffness`` in N/m, ``damping`` in N s/m) holds the tip where it was at
  ``onset``, like a hand gripping the arm; then it lets go.

Each is a skelarm external force ``f(t, skeleton) -> (fx, fy)`` with ``t`` on the
task clock (0 when the task starts). From the same start posture, every arm meets
the same pushes: the same force, in the same direction, at the same times. The initial offset needs no force: it is a
start posture away from the demonstrated ones (``start_offsets_deg`` in
``[evaluation]``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skelarm import Skeleton, compute_jacobian

# The keys each type of [disturbance] table needs, and those it may have.
DISTURBANCE_KEYS = {
    "push": {"type", "force", "onset", "duration"},
    "block": {"type", "onset", "release", "stiffness", "damping"},
}
OPTIONAL_KEYS = {"push": {"direction", "angle_deg"}, "block": set()}
# The push directions by name, as angles counterclockwise from the direction toward the target (deg).
PUSH_DIRECTIONS = {"forward": 0.0, "across": 90.0, "backward": 180.0}


@dataclass
class Push:
    """A constant tip force ``force`` (N, base frame) during ``[onset, onset + duration)``."""

    force: NDArray[np.float64]
    onset: float
    duration: float

    def __call__(self, t: float, skeleton: Skeleton) -> NDArray[np.float64]:
        if self.onset <= t < self.onset + self.duration:
            return self.force
        return np.zeros(2)


@dataclass
class Block:
    """A spring-damper that holds the tip where it was at ``onset``, until ``release``."""

    onset: float
    release: float
    stiffness: float  # N/m
    damping: float  # N s/m
    held: NDArray[np.float64] | None = field(default=None, init=False)  # where the tip is held

    def __call__(self, t: float, skeleton: Skeleton) -> NDArray[np.float64]:
        if not self.onset <= t < self.release:
            return np.zeros(2)
        tip = np.array([skeleton.links[-1].xe, skeleton.links[-1].ye])
        if self.held is None:
            self.held = tip
        velocity = compute_jacobian(skeleton) @ skeleton.dq
        return -self.stiffness * (tip - self.held) - self.damping * velocity


@dataclass
class Disturbances:
    """Several disturbances in one run: their tip forces add up."""

    parts: list[Push | Block]

    def __call__(self, t: float, skeleton: Skeleton) -> NDArray[np.float64]:
        return np.sum([part(t, skeleton) for part in self.parts], axis=0)


def make_disturbance(
    config: dict[str, Any] | list[dict[str, Any]] | None, start_hand: NDArray[np.float64], target: NDArray[np.float64]
) -> Push | Block | Disturbances | None:
    """A fresh disturbance for one run from the hand position ``start_hand``, or None without one.

    ``config`` is a ``[disturbance]`` table, or a list of them (``[[disturbance]]``),
    whose forces add up. Make one per run: a block remembers where it holds the tip.
    """
    if config is None:
        return None
    if isinstance(config, dict):
        return _make_one(config, start_hand, target)
    parts = []
    for index, table in enumerate(config, start=1):
        try:
            parts.append(_make_one(table, start_hand, target))
        except ValueError as error:
            msg = f"disturbance {index} of {len(config)}: {error}"
            raise ValueError(msg) from error
    return Disturbances(parts) if parts else None


def disturbance_spans(config: dict[str, Any] | list[dict[str, Any]] | None) -> list[tuple[float, float]]:
    """When each disturbance of ``config`` acts on the task clock, in the order given (none without one)."""
    tables = [] if config is None else [config] if isinstance(config, dict) else config
    return [
        (table["onset"], table["onset"] + table["duration"])
        if table["type"] == "push"
        else (table["onset"], table["release"])
        for table in tables
    ]


def _make_one(config: dict[str, Any], start_hand: NDArray[np.float64], target: NDArray[np.float64]) -> Push | Block:
    """The disturbance of one ``[disturbance]`` table."""
    kind = config.get("type")
    if kind not in DISTURBANCE_KEYS:
        msg = f"unknown disturbance type {kind!r}; choose from {', '.join(DISTURBANCE_KEYS)}"
        raise ValueError(msg)
    keys = set(config)
    if not DISTURBANCE_KEYS[kind] <= keys <= DISTURBANCE_KEYS[kind] | OPTIONAL_KEYS[kind]:
        optional = f", and may have {', '.join(sorted(OPTIONAL_KEYS[kind]))}" if OPTIONAL_KEYS[kind] else ""
        msg = f"a {kind} [disturbance] needs the keys {', '.join(sorted(DISTURBANCE_KEYS[kind]))}{optional}"
        raise ValueError(msg)
    if kind == "push":
        if "direction" in config and "angle_deg" in config:
            msg = "a push takes a direction or an angle_deg, not both"
            raise ValueError(msg)
        direction = config.get("direction", "across")
        if direction not in PUSH_DIRECTIONS:
            msg = f"unknown push direction {direction!r}; choose from {', '.join(PUSH_DIRECTIONS)}"
            raise ValueError(msg)
        angle = np.radians(config.get("angle_deg", PUSH_DIRECTIONS[direction]))
        forward = (target - start_hand) / np.linalg.norm(target - start_hand)
        turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])  # counterclockwise
        return Push(config["force"] * (turn @ forward), config["onset"], config["duration"])
    return Block(config["onset"], config["release"], config["stiffness"], config["damping"])
