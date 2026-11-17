# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Start postures of a two-link arm whose hands are spread evenly over the workspace.

    uv run python tools/tip_starts.py experiments/demonstrations/reach_manual_v2_no_limits.toml \
        --counts 6 3 --first-angle 43.3

Start postures a step apart in joint space put their hands unevenly in the
workspace: crowded where the arm is nearly straight or folded, and thin between.
These are laid out by the hand instead. The hand reaches the ring between the two
links' difference and their sum from the base. ``--counts`` splits it into rings
around the base, from the outermost inward, whose areas are as their numbers of
hands, so that every hand stands for the same area; a ring's hands lie on the
circle that halves its area, equally spaced. The outermost ring's first hand is at
``--first-angle`` from the x axis, and each ring inward is turned by half the
spacing of the ring outside it, so that its hands fall between that ring's.

Each hand position has two postures, one with the elbow on either side: ``--elbow``
is the sign of joint 2. The postures are rounded to whole degrees, to type them as
the recorder's ``--pose``, and printed as rows of a configuration's
``[demonstrations]`` ``start_q`` or ``[evaluation.extra_starts]``, each with the hand
position its rounded posture gives.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from skelarm import Skeleton

from arm_esn_ctrl.demonstrations import endpoint_positions

N_LINKS = 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="a configuration file with the arm (TOML)")
    parser.add_argument(
        "--counts", type=int, nargs="+", required=True, help="hands on each ring, from the outermost inward"
    )
    parser.add_argument(
        "--first-angle", type=float, default=0.0, help="direction of the outermost ring's first hand (deg)"
    )
    parser.add_argument("--elbow", type=int, choices=(1, -1), default=1, help="the sign of joint 2")
    args = parser.parse_args()
    skeleton = Skeleton.from_toml(args.config)
    hands = ring_hands(skeleton, args.counts, args.first_angle)
    postures = hand_postures(skeleton, hands, args.elbow)
    for posture, hand in zip(postures, endpoint_positions(skeleton, np.radians(postures)), strict=True):
        print(f"  [{posture[0]:.1f}, {posture[1]:.1f}],  # hand ({hand[0]:.3f}, {hand[1]:.3f}) m")


def link_lengths(skeleton: Skeleton) -> tuple[float, float]:
    """The two links' lengths (m); an arm of another number of links has no one posture for a hand position."""
    lengths = [link.prop.length for link in skeleton.links[1:]]
    if len(lengths) != N_LINKS:
        msg = f"the hand's rings give start postures for a two-link arm only, not one of {len(lengths)} links"
        raise ValueError(msg)
    return lengths[0], lengths[1]


def ring_hands(skeleton: Skeleton, counts: list[int], first_angle_deg: float) -> NDArray[np.float64]:
    """Hand positions (m) on rings of equal area for each hand, from the outermost ring inward (see the module)."""
    upper, forearm = link_lengths(skeleton)
    inner, outer = abs(upper - forearm), upper + forearm
    hands = []
    outside = 0  # hands on the rings outside this one
    angle = first_angle_deg
    for ring, count in enumerate(counts):
        if ring > 0:
            angle += 180.0 / counts[ring - 1]
        within = 1.0 - (outside + count / 2) / sum(counts)  # the part of the area within the ring's circle
        radius = np.sqrt(inner**2 + (outer**2 - inner**2) * within)
        directions = np.radians(angle + 360.0 / count * np.arange(count))
        hands.append(radius * np.column_stack((np.cos(directions), np.sin(directions))))
        outside += count
    return np.vstack(hands)


def hand_postures(skeleton: Skeleton, hands: NDArray[np.float64], elbow: int) -> NDArray[np.float64]:
    """The postures (whole deg) that put the hand at ``hands``, joint 2 of the sign of ``elbow``, joint 1 from -180."""
    upper, forearm = link_lengths(skeleton)
    cosine = ((hands**2).sum(axis=1) - upper**2 - forearm**2) / (2 * upper * forearm)
    q2 = elbow * np.arccos(cosine)
    q1 = np.arctan2(hands[:, 1], hands[:, 0]) - np.arctan2(forearm * np.sin(q2), upper + forearm * np.cos(q2))
    postures = np.round(np.degrees(np.column_stack((q1, q2))))
    postures[:, 0] = (postures[:, 0] + 180.0) % 360.0 - 180.0
    return postures


if __name__ == "__main__":
    main()
