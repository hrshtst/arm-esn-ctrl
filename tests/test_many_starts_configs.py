# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""The many-starts experiment's configurations agree with each other, with the takes, and with v2's."""

import sys
import tomllib

import numpy as np
import pytest
from skelarm import Skeleton

from arm_esn_ctrl.disturbances import same_arm
from arm_esn_ctrl.esn import EsnConfig
from arm_esn_ctrl.storage import REPO_ROOT

# The runners import each other as modules of their own directory, as they run.
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from robot_esn import tracker_settings

sys.path.insert(0, str(REPO_ROOT))
from tools.starts_figure import plot_starts, target_postures

TAKES = REPO_ROOT / "experiments/demonstrations/reach_manual_v2_starts.toml"
V2 = REPO_ROOT / "experiments/demonstrations/reach_manual_v2.toml"
EXPERIMENT = REPO_ROOT / "experiments/manual_demonstration_v2_many_starts"
V2_PUSHES = (
    REPO_ROOT / "experiments/manual_demonstration_v2_consistent_search/pushes_robot_esn1_lr0.03_sr1.2_is1.5_seed8.toml"
)


def load(path):
    with path.open("rb") as f:
        return tomllib.load(f)


def test_the_takes_are_of_v2s_arm_and_target_one_per_start_of_the_lattice():
    takes, v2 = load(TAKES), load(V2)

    assert same_arm(Skeleton.from_toml(TAKES), Skeleton.from_toml(V2))
    assert takes["task"] == v2["task"]
    assert takes["simulator"] == v2["simulator"]
    assert takes["recording"]["filter"] == v2["recording"]["filter"]
    starts = takes["demonstrations"]["start_q"]
    assert len(takes["recording"]["takes"]) == len(starts) == 9
    assert takes["recording"]["takes"][0] == v2["recording"]["take"]  # the take of reports 005 to 020
    assert starts[0] == v2["demonstrations"]["start_q"][0]
    lattice = {(q1, q2) for q1 in (-60.0, 30.0, 120.0) for q2 in (30.0, 90.0, 150.0)}
    assert {tuple(start) for start in starts} == lattice


def test_the_hand_esn_trains_on_every_take_with_800_neurons():
    config = load(EXPERIMENT / "hand_esn.toml")

    assert config["demonstrations"]["train"] == [f"demo_{i:02d}_filtered.sklog.npz" for i in range(9)]
    esn = EsnConfig(**config["esn"])
    assert esn.n_neurons == 800
    assert (esn.dt, esn.warmup, esn.sparsity, esn.bias) == (0.01, 1.0, 0.05, True)


def test_no_untaught_start_is_a_taught_one_and_all_lie_within_the_joint_limits():
    taught = np.array(load(TAKES)["demonstrations"]["start_q"])
    for name in ("hand_esn.toml", "hand_robot.toml"):
        untaught = np.array(load(EXPERIMENT / name)["evaluation"]["extra_starts"]["untaught"])
        assert untaught.shape == (36, 2)
        assert len({tuple(q) for q in untaught}) == 36
        distance = np.abs(untaught[:, np.newaxis, :] - taught[np.newaxis, :, :]).max(axis=2)
        assert distance.min() >= 15.0
        assert np.abs(untaught).max() <= 180.0
    assert (
        load(EXPERIMENT / "hand_esn.toml")["evaluation"]["extra_starts"]
        == (load(EXPERIMENT / "hand_robot.toml")["evaluation"]["extra_starts"])
    )


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
    assert len(workspace.collections) == 2  # the taught starts' hands, and the untaught
    assert len(joint_space.collections) == 2
    # The hand is at the target with the elbow on either side.
    postures = target_postures(skeleton, np.array(takes["task"]["target"]["pos"]))
    assert len(postures) == 2
    assert sorted(np.round(postures[:, 1])) == [-74.0, 74.0]
