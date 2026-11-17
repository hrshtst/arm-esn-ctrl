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

TAKES = REPO_ROOT / "experiments/demonstrations/reach_manual_v2_no_limits.toml"
V2 = REPO_ROOT / "experiments/demonstrations/reach_manual_v2.toml"
EXPERIMENT = REPO_ROOT / "experiments/manual_demonstration_v2_many_starts"
V2_PUSHES = (
    REPO_ROOT / "experiments/manual_demonstration_v2_consistent_search/pushes_robot_esn1_lr0.03_sr1.2_is1.5_seed8.toml"
)


def load(path):
    with path.open("rb") as f:
        return tomllib.load(f)


def test_the_takes_are_of_v2s_arm_without_joint_limits_and_of_its_target():
    takes, v2 = load(TAKES), load(V2)

    links, v2_links = takes["skeleton"]["link"], v2["skeleton"]["link"]
    assert [link.pop("limits") for link in links] == [[-3600.0, 3600.0]] * 2
    assert [link.pop("limits") for link in v2_links] == [[-180.0, 180.0]] * 2
    assert takes["skeleton"] == v2["skeleton"]  # the same links otherwise
    assert not same_arm(Skeleton.from_toml(TAKES), Skeleton.from_toml(V2))
    assert takes["task"] == v2["task"]
    assert takes["simulator"] == v2["simulator"]
    assert takes["recording"]["filter"] == v2["recording"]["filter"]


def test_the_nine_takes_start_with_their_hands_spread_evenly_over_the_workspace():
    takes = load(TAKES)
    skeleton = Skeleton.from_toml(TAKES)

    starts = takes["demonstrations"]["start_q"]
    assert takes["recording"]["takes"] == [f"data/reach_manual_v2_no_limits/take_{i:02d}.sklog.npz" for i in range(9)]
    # An outer ring of six hands and an inner one of three, the elbow on one side.
    assert starts == hand_postures(skeleton, ring_hands(skeleton, [6, 3], 43.3), 1).tolist()
    assert all(start[1] > 0 for start in starts)
    assert takes["initial"]["q"] == starts[0]
    # The first hand is in the direction of the start of reports 005 to 020's take.
    v2_hand = endpoint_positions(skeleton, np.radians(load(V2)["demonstrations"]["start_q"]))[0]
    assert np.degrees(np.arctan2(v2_hand[1], v2_hand[0])) == pytest.approx(43.3, abs=0.05)


def test_the_hand_esn_trains_on_every_take_with_800_neurons():
    config = load(EXPERIMENT / "hand_esn.toml")

    assert config["demonstrations"]["train"] == [f"demo_{i:02d}_filtered.sklog.npz" for i in range(9)]
    esn = EsnConfig(**config["esn"])
    assert esn.n_neurons == 800
    assert (esn.dt, esn.warmup, esn.sparsity, esn.bias) == (0.01, 1.0, 0.05, True)


def test_the_untaught_starts_are_36_hands_over_the_workspace_with_the_takes_elbow():
    skeleton = Skeleton.from_toml(TAKES)
    taught = endpoint_positions(skeleton, np.radians(load(TAKES)["demonstrations"]["start_q"]))
    hands = ring_hands(skeleton, [18, 12, 6], 53.3)
    extra = load(EXPERIMENT / "hand_esn.toml")["evaluation"]["extra_starts"]

    assert list(extra) == ["untaught"]
    assert extra["untaught"] == hand_postures(skeleton, hands, 1).tolist()
    untaught = endpoint_positions(skeleton, np.radians(extra["untaught"]))
    # Whole degrees move a hand less than 1 cm from its place on the ring.
    assert np.linalg.norm(untaught - hands, axis=1).max() < 0.01
    # No hand starts where a take does, nor at the target.
    assert np.linalg.norm(untaught[:, np.newaxis] - taught[np.newaxis], axis=2).min() >= 0.10
    assert np.linalg.norm(untaught - load(TAKES)["task"]["target"]["pos"], axis=1).min() >= 0.10
    assert load(EXPERIMENT / "hand_robot.toml")["evaluation"]["extra_starts"] == extra


def test_the_coarse_search_runs_108_settings_of_one_seed_from_each_takes_start():
    coarse, hand = load(EXPERIMENT / "coarse_search.toml"), load(EXPERIMENT / "hand_esn.toml")

    assert coarse["demonstrations"]["train"] == hand["demonstrations"]["train"]
    assert coarse["sweep"] == {
        "leak_rate": [0.01, 0.03, 0.1, 0.3],
        "spectral_radius": [0.9, 1.2, 1.5],
        "input_scaling": [0.1, 0.5, 2.0],
        "ridge": [1e-4, 1e-2, 1.0],
    }
    esn = EsnConfig(**coarse["esn"])
    assert (esn.n_neurons, esn.seed) == (800, 0)
    assert (esn.dt, esn.warmup, esn.sparsity, esn.bias) == (0.01, 1.0, 0.05, True)
    assert coarse["ranking"] == {"by": "detour_ratio"}
    # Undisturbed, from each take's own start and no other.
    assert coarse["evaluation"] == {"duration": 24.0, "hold": 2.0, "start_offsets_deg": [[0.0, 0.0]]}
    assert "disturbance" not in coarse
    assert coarse["tracker"] == load(EXPERIMENT / "hand_robot.toml")["tracker"]


def test_the_fine_search_runs_81_settings_around_the_center_with_seeds_1_to_5():
    coarse, fine = load(EXPERIMENT / "coarse_search.toml"), load(EXPERIMENT / "fine_search.toml")
    center = {"leak_rate": 0.1, "spectral_radius": 1.2, "input_scaling": 0.5, "ridge": 1.0}

    sweep = dict(fine["sweep"])
    assert sweep.pop("seed") == [1, 2, 3, 4, 5]
    for name, values in sweep.items():
        grid = coarse["sweep"][name]
        below, middle, above = values
        assert middle == center[name] == fine["esn"][name]
        k = grid.index(middle)
        if name == "spectral_radius":  # halfway in difference
            assert (below, above) == pytest.approx(((grid[k - 1] + middle) / 2, (middle + grid[k + 1]) / 2))
        else:  # halfway in ratio, to two digits; beyond the grid's edge, as far as on the other side
            assert below == pytest.approx(np.sqrt(grid[k - 1] * middle), rel=0.03)
            beyond = grid[k + 1] if k + 1 < len(grid) else middle**2 / grid[k - 1]
            assert above == pytest.approx(np.sqrt(middle * beyond), rel=0.03)
    # The same takes, ESN, tracker, runs, and score as the coarse search's.
    for table in ("demonstrations", "tracker", "evaluation", "ranking"):
        assert fine[table] == coarse[table]
    swept = {*fine["sweep"]}
    assert {k: v for k, v in fine["esn"].items() if k not in swept} == {
        k: v for k, v in coarse["esn"].items() if k not in swept
    }


def test_the_untaught_search_runs_20_of_the_fine_searchs_settings_from_the_untaught_starts_too():
    again, fine = load(EXPERIMENT / "untaught_search.toml"), load(EXPERIMENT / "fine_search.toml")
    names = ["leak_rate", "spectral_radius", "input_scaling", "ridge"]

    settings = [tuple(setting[name] for name in names) for setting in again["settings"]]
    assert len(set(settings)) == len(settings) == 20
    for setting in again["settings"]:
        assert list(setting) == names
        assert all(setting[name] in fine["sweep"][name] for name in names)
    assert settings[:2] == [(0.1, 1.2, 1.0, 1.0), (0.055, 1.2, 1.0, 0.1)]  # the first pair's
    assert again["sweep"] == {"seed": fine["sweep"]["seed"]}
    # The same takes, ESN, tracker, and score as the fine search's, with the untaught starts added.
    for table in ("demonstrations", "tracker", "ranking"):
        assert again[table] == fine[table]
    assert {k: v for k, v in again["esn"].items() if k not in names} == {
        k: v for k, v in fine["esn"].items() if k not in names
    }
    extra = again["evaluation"].pop("extra_starts")
    assert again["evaluation"] == fine["evaluation"]
    assert extra == load(EXPERIMENT / "hand_esn.toml")["evaluation"]["extra_starts"]


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("esn1_lr0.055_sr1.2_is1_ridge1_seed4.toml", (0.055, 1.2, 1.0, 1.0, 4)),
        ("esn2_lr0.1_sr1.05_is0.5_ridge10_seed3.toml", (0.1, 1.05, 0.5, 10.0, 3)),
        ("taught1_lr0.1_sr1.2_is1_ridge1_seed2.toml", (0.1, 1.2, 1.0, 1.0, 2)),
        ("taught2_lr0.055_sr1.2_is1_ridge0.1_seed4.toml", (0.055, 1.2, 1.0, 0.1, 4)),
    ],
)
def test_the_chosen_esns_are_settings_and_seeds_of_the_fine_search_run_as_the_hand_tuned_one(name, values):
    chosen, fine, hand = (
        load(EXPERIMENT / name),
        load(EXPERIMENT / "fine_search.toml"),
        load(EXPERIMENT / "hand_esn.toml"),
    )
    swept = ("leak_rate", "spectral_radius", "input_scaling", "ridge", "seed")

    assert tuple(chosen["esn"][key] for key in swept) == values
    for key, value in zip(swept, values, strict=True):
        assert value in fine["sweep"][key]
    assert {k: v for k, v in chosen["esn"].items() if k not in swept} == {
        k: v for k, v in fine["esn"].items() if k not in swept
    }
    assert chosen["demonstrations"] == fine["demonstrations"]
    assert chosen["evaluation"] == hand["evaluation"]
    assert chosen["apps"] == hand["apps"]


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
    # No joint has a limit to show, and the joint space shows one turn.
    assert "joint 1 at its limit" not in [line.get_label() for line in workspace.lines]
    assert joint_1_wall(skeleton, np.radians(takes["demonstrations"]["start_q"])) is None
    assert joint_space.get_xlim() == joint_space.get_ylim() == (-185.0, 185.0)
    # The hand is at the target with the elbow on either side, once each within the turn.
    postures = target_postures(skeleton, np.array(takes["task"]["target"]["pos"]))
    assert len(postures) == 2
    assert sorted(np.round(postures[:, 1])) == [-74.0, 74.0]
    assert np.abs(postures).max() <= 180.0


def test_v2s_joint_1_limit_crosses_the_workspace_below_the_base_on_the_takes_elbow_side():
    skeleton = Skeleton.from_toml(V2)
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
