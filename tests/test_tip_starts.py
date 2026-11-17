# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the start postures laid out by the hand over the workspace."""

import sys

import numpy as np
import pytest
from skelarm import Skeleton

from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.storage import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))
from tools.tip_starts import hand_postures, ring_hands

V2 = REPO_ROOT / "experiments/demonstrations/reach_manual_v2.toml"  # links of 0.5 and 0.4 m
FOUR_DOF = REPO_ROOT / "experiments/demonstrations/reach_manual_four_dof.toml"


def test_every_hand_stands_for_the_same_area_of_the_ring_the_hand_reaches():
    hands = ring_hands(Skeleton.from_toml(V2), [18, 12, 6], 0.0)

    radii = np.linalg.norm(hands, axis=1)
    reach = 0.9**2 - 0.1**2  # the area the hand reaches, over pi
    # Each ring's circle halves the ring's area, and the rings' areas are as their hands.
    np.testing.assert_allclose(radii[:18] ** 2, 0.1**2 + reach * (1 - 9 / 36))
    np.testing.assert_allclose(radii[18:30] ** 2, 0.1**2 + reach * (1 - 24 / 36))
    np.testing.assert_allclose(radii[30:] ** 2, 0.1**2 + reach * (1 - 33 / 36))
    # So the hands are about as far apart along each ring as the rings are from each other.
    along = [2 * np.pi * radius / count for radius, count in ((radii[0], 18), (radii[18], 12), (radii[30], 6))]
    between = [radii[0] - radii[18], radii[18] - radii[30]]
    assert max(along + between) / min(along + between) < 1.2


def test_a_rings_hands_are_equally_spaced_and_fall_between_those_of_the_ring_outside():
    hands = ring_hands(Skeleton.from_toml(V2), [6, 3], 43.3)

    directions = np.degrees(np.arctan2(hands[:, 1], hands[:, 0])) % 360.0
    np.testing.assert_allclose(directions[:6], 43.3 + 60.0 * np.arange(6))
    np.testing.assert_allclose(directions[6:], 73.3 + 120.0 * np.arange(3))


@pytest.mark.parametrize("elbow", [1, -1])
def test_the_postures_put_the_hand_there_in_whole_degrees_with_the_elbow_on_the_side_asked(elbow):
    skeleton = Skeleton.from_toml(V2)
    hands = ring_hands(skeleton, [18, 12, 6], 53.3)

    postures = hand_postures(skeleton, hands, elbow)

    np.testing.assert_array_equal(postures, np.round(postures))
    assert (np.sign(postures[:, 1]) == elbow).all()
    assert postures[:, 0].min() >= -180.0
    assert postures[:, 0].max() < 180.0
    reached = endpoint_positions(skeleton, np.radians(postures))
    assert np.linalg.norm(reached - hands, axis=1).max() < 0.01


def test_an_arm_of_more_links_has_no_one_posture_for_a_hand_position():
    with pytest.raises(ValueError, match="two-link arm only"):
        ring_hands(Skeleton.from_toml(FOUR_DOF), [6, 3], 0.0)
