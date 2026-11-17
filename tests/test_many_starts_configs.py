# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""The many-starts experiment's configurations agree with each other, with the takes, and with v2's."""

import sys
import tomllib

import numpy as np
import pytest
from skelarm import Skeleton

from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.disturbances import same_arm
from arm_esn_ctrl.esn import EsnConfig
from arm_esn_ctrl.storage import REPO_ROOT

# The runners import each other as modules of their own directory, as they run.
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from robot_esn import tracker_settings

sys.path.insert(0, str(REPO_ROOT))
from tools.starts_figure import joint_1_wall, plot_starts, target_postures
from tools.tip_starts import hand_postures, ring_hands

TAKES = REPO_ROOT / "experiments/demonstrations/reach_manual_v2_starts.toml"
V2 = REPO_ROOT / "experiments/demonstrations/reach_manual_v2.toml"
EXPERIMENT = REPO_ROOT / "experiments/manual_demonstration_v2_many_starts"
V2_PUSHES = (
    REPO_ROOT / "experiments/manual_demonstration_v2_consistent_search/pushes_robot_esn1_lr0.03_sr1.2_is1.5_seed8.toml"
)


def load(path):
    with path.open("rb") as f:
        return tomllib.load(f)


def test_the_takes_are_of_v2s_arm_and_target_their_starts_laid_out_by_the_hand():
    takes, v2 = load(TAKES), load(V2)

    assert same_arm(Skeleton.from_toml(TAKES), Skeleton.from_toml(V2))
    assert takes["task"] == v2["task"]
    assert takes["simulator"] == v2["simulator"]
    assert takes["recording"]["filter"] == v2["recording"]["filter"]
    starts = takes["demonstrations"]["start_q"]
    assert len(takes["recording"]["takes"]) == len(starts) == 9
    assert takes["recording"]["takes"][0] == v2["recording"]["take"]  # the take of reports 005 to 020
    assert starts[0] == v2["demonstrations"]["start_q"][0]
    # An outer ring of six hands and an inner one of three; the take of reports 005 to 020
    # stands in the outer ring's first place, where its hand's direction begins the ring.
    skeleton = Skeleton.from_toml(TAKES)
    rings = hand_postures(skeleton, ring_hands(skeleton, [6, 3], 43.3), 1)
    assert starts[1:] == rings[1:].tolist()
    first_hand = endpoint_positions(skeleton, np.radians(starts[:1]))[0]
    assert np.degrees(np.arctan2(first_hand[1], first_hand[0])) == pytest.approx(43.3, abs=0.05)
    assert all(start[1] > 0 for start in starts)  # the elbow on one side


def test_the_hand_esn_trains_on_every_take_with_800_neurons():
    config = load(EXPERIMENT / "hand_esn.toml")

    assert config["demonstrations"]["train"] == [f"demo_{i:02d}_filtered.sklog.npz" for i in range(9)]
    esn = EsnConfig(**config["esn"])
    assert esn.n_neurons == 800
    assert (esn.dt, esn.warmup, esn.sparsity, esn.bias) == (0.01, 1.0, 0.05, True)


def test_the_untaught_starts_are_36_hands_over_the_workspace_with_the_elbow_on_either_side():
    skeleton = Skeleton.from_toml(TAKES)
    taught = endpoint_positions(skeleton, np.radians(load(TAKES)["demonstrations"]["start_q"]))
    hands = ring_hands(skeleton, [18, 12, 6], 53.3)
    extra = load(EXPERIMENT / "hand_esn.toml")["evaluation"]["extra_starts"]

    assert list(extra) == ["untaught", "untaught_other_elbow"]
    assert extra["untaught"] == hand_postures(skeleton, hands, 1).tolist()
    assert extra["untaught_other_elbow"] == hand_postures(skeleton, hands, -1).tolist()
    for postures in extra.values():
        assert np.abs(postures).max() <= 180.0
        untaught = endpoint_positions(skeleton, np.radians(postures))
        # Whole degrees move a hand less than 1 cm from its place on the ring.
        assert np.linalg.norm(untaught - hands, axis=1).max() < 0.01
        # No hand starts where a take does, nor at the target.
        assert np.linalg.norm(untaught[:, np.newaxis] - taught[np.newaxis], axis=2).min() >= 0.10
        assert np.linalg.norm(untaught - load(TAKES)["task"]["target"]["pos"], axis=1).min() >= 0.10
    assert load(EXPERIMENT / "hand_robot.toml")["evaluation"]["extra_starts"] == extra


@pytest.mark.parametrize("name", ["hand_robot.toml", "hand_pushes_robot.toml"])
def test_the_robot_runs_use_the_consistent_searchs_tracker_with_its_takes_gains(name):
    tracker, v2 = load(EXPERIMENT / name)["tracker"], load(V2_PUSHES)["tracker"]

    assert (
        tracker.pop("gains_from") == "results/demonstrations/20261008-170034-reach_manual_v2/demo_00_filtered.sklog.npz"
    )
    assert tracker == v2
    assert [setting.name for setting in tracker_settings(tracker)] == ["pd_w20_z0.4"]


def test_the_pushes_are_report_007s_the_first_two_placed_along_each_take():
    pushes, v2 = load(EXPERIMENT / "hand_pushes_robot.toml")["disturbance"], load(V2_PUSHES)["disturbance"]

    assert [push["way"] for push in pushes[:2]] == [0.18, 0.62]
    assert pushes[2]["onset"] == v2[2]["onset"] == 15.0
    for push, theirs in zip(pushes, v2, strict=True):
        for key in ("type", "force", "duration", "angle_deg"):
            assert push[key] == theirs[key]


def test_the_apps_get_the_consistent_searchs_tracker_and_a_take_in_the_repository():
    arguments = load(EXPERIMENT / "hand_esn.toml")["apps"]["robot_app"].split()

    assert arguments[:7] == ["--law", "pd", "--omega", "20", "--damping", "0.4", "--control-period"]
    take = arguments[arguments.index("--gains-from") + 1]
    assert (REPO_ROOT / take).exists()


def test_the_starts_figure_draws_the_takes_starts_and_the_untaught_ones():
    takes = load(TAKES)
    skeleton = Skeleton.from_toml(TAKES)
    extra = load(EXPERIMENT / "hand_esn.toml")["evaluation"]["extra_starts"]

    figure = plot_starts(skeleton, takes["task"], takes["demonstrations"]["start_q"], extra, "test")

    workspace, joint_space = figure.axes
    assert len(workspace.collections) == 3  # the taught starts' hands, and the untaught of either elbow
    assert len(joint_space.collections) == 3
    assert "joint 1 at its limit" in [line.get_label() for line in workspace.lines]
    # The hand is at the target with the elbow on either side.
    postures = target_postures(skeleton, np.array(takes["task"]["target"]["pos"]))
    assert len(postures) == 2
    assert sorted(np.round(postures[:, 1])) == [-74.0, 74.0]


def test_joint_1s_limit_crosses_the_workspace_below_the_base_on_the_takes_elbow_side():
    skeleton = Skeleton.from_toml(TAKES)
    starts = np.radians(load(TAKES)["demonstrations"]["start_q"])

    wall = joint_1_wall(skeleton, starts)

    assert wall is not None
    np.testing.assert_allclose(wall[0], [-0.9, 0.0], atol=1e-9)  # straight
    np.testing.assert_allclose(wall[-1], [-0.1, 0.0], atol=1e-9)  # folded
    np.testing.assert_allclose(np.linalg.norm(wall - [-0.5, 0.0], axis=1), 0.4)
    assert wall[1:-1, 1].max() < 0.0
    # With the elbow on the other side, it crosses above the base.
    other = joint_1_wall(skeleton, -starts)
    assert other is not None
    assert other[1:-1, 1].min() > 0.0
