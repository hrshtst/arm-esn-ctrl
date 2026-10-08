# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for drawing the robot's postures in figures."""

import numpy as np
import pytest
from matplotlib.figure import Figure
from skelarm import Skeleton

from arm_esn_ctrl.drawing import JOINT_COLOR, LINK_COLOR, draw_postures, spread_along_path
from arm_esn_ctrl.storage import REPO_ROOT

CONFIG = REPO_ROOT / "experiments/demonstrations/reach_manual_v2.toml"


def test_postures_spread_along_the_path_by_its_length_not_by_time():
    """A pause at the start adds samples but no length: the postures still spread evenly along the path."""
    paused = np.zeros((100, 2))
    moving = np.column_stack([np.linspace(0.0, 1.0, 101), np.zeros(101)])
    hand = np.vstack([paused, moving])

    indices = spread_along_path(hand, 5)

    assert indices[0] == 0
    assert indices[-1] == len(hand) - 1
    assert hand[indices, 0] == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_a_path_that_never_moves_has_its_first_and_last_postures():
    assert spread_along_path(np.zeros((10, 2)), 5) == [0, 9]


def test_each_posture_draws_its_links_and_joints_faintly_without_a_legend():
    skeleton = Skeleton.from_toml(CONFIG)
    ax = Figure().subplots()
    postures = np.radians([[30.0, 30.0], [60.0, 90.0]])

    draw_postures(ax, skeleton, postures, alpha=0.3)

    links = [line for line in ax.lines if line.get_color() == LINK_COLOR]
    joints = [line for line in ax.lines if line.get_color() == JOINT_COLOR]
    assert len(links) == 2 * 2  # two movable links per posture
    assert len(joints) == 2
    assert all(line.get_alpha() == pytest.approx(0.3) for line in ax.lines)
    assert all(str(line.get_label()).startswith("_") for line in ax.lines)  # no legend entries
    tip = np.asarray(joints[1].get_xydata(), dtype=np.float64)[-1]
    skeleton.q = postures[1]
    assert tip == pytest.approx([skeleton.links[-1].xe, skeleton.links[-1].ye])
