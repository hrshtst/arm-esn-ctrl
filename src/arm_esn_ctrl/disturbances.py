# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Disturbances of Stage 2: forces at the arm's tip, scripted on the task clock.

A disturbance is configured by a ``[disturbance]`` table:

- ``type = "push"``: a constant force of ``force`` newtons from ``onset`` for
  ``duration`` seconds, perpendicular to the reach (the straight line from the
  start hand position to the target), turned counterclockwise.
- ``type = "block"``: from ``onset`` until ``release``, a stiff spring-damper
  (``stiffness`` in N/m, ``damping`` in N s/m) holds the tip where it was at
  ``onset``, like a hand gripping the arm; then it lets go.

Each is a skelarm external force ``f(t, skeleton) -> (fx, fy)`` with ``t`` on the
task clock (0 when the task starts). The initial offset needs no force: it is a
start posture away from the demonstrated ones (``start_offsets_deg`` in
``[evaluation]``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skelarm import Skeleton, compute_jacobian

# The keys of each type of [disturbance] table.
DISTURBANCE_KEYS = {
    "push": {"type", "force", "onset", "duration"},
    "block": {"type", "onset", "release", "stiffness", "damping"},
}


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


def make_disturbance(
    config: dict[str, Any] | None, start_hand: NDArray[np.float64], target: NDArray[np.float64]
) -> Push | Block | None:
    """A fresh disturbance for one run from the hand position ``start_hand``, or None without one.

    Make one per run: a block remembers where it holds the tip.
    """
    if config is None:
        return None
    kind = config.get("type")
    if kind not in DISTURBANCE_KEYS:
        msg = f"unknown disturbance type {kind!r}; choose from {', '.join(DISTURBANCE_KEYS)}"
        raise ValueError(msg)
    if set(config) != DISTURBANCE_KEYS[kind]:
        msg = f"a {kind} [disturbance] needs exactly the keys {', '.join(sorted(DISTURBANCE_KEYS[kind]))}"
        raise ValueError(msg)
    if kind == "push":
        reach = target - start_hand
        sideways = np.array([-reach[1], reach[0]]) / np.linalg.norm(reach)  # counterclockwise
        return Push(config["force"] * sideways, config["onset"], config["duration"])
    return Block(config["onset"], config["release"], config["stiffness"], config["damping"])
