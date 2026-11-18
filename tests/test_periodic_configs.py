# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""The periodic motion's configurations agree with each other, with the ellipse's takes, and with stage 1's."""

import tomllib

import numpy as np
import pytest
from skelarm import Skeleton, build_curve

from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.storage import REPO_ROOT

EXPERIMENT = REPO_ROOT / "experiments/manual_demonstration_v2_periodic"
STAGE_1 = REPO_ROOT / "experiments/manual_demonstration_v2_many_starts"
TAKES = REPO_ROOT / "experiments/demonstrations/ellipse_manual_v2_no_limits.toml"
SEARCHES = ["coarse_search.toml", "scripted_coarse_search.toml"]
CYCLE = {"curve_tolerance": 0.02, "cycle_periods": 5, "cycle_turns": [4.0, 6.0]}


def load(path):
    with path.open("rb") as f:
        return tomllib.load(f)


@pytest.mark.parametrize("name", SEARCHES)
def test_the_coarse_searches_run_stage_1s_108_settings_with_its_esn_and_tracker(name):
    search, stage_1 = load(EXPERIMENT / name), load(STAGE_1 / "coarse_search.toml")

    assert search["esn"] == stage_1["esn"]  # 800 neurons, seed 0, and the rest of stage 1's
    assert search["sweep"] == stage_1["sweep"]
    assert search["tracker"] == stage_1["tracker"]
    assert search["ranking"] == {"by": "curve_rms_m"}
    # Undisturbed, for 40 s, from the eight starts on the curve, the take's own and seven more.
    assert search["evaluation"] == {
        "duration": 40.0,
        **CYCLE,
        "start_offsets_deg": [[0.0, 0.0]],
        "extra_starts": {"on_curve": load(EXPERIMENT / "hand_esn.toml")["evaluation"]["extra_starts"]["on_curve"]},
    }
    assert "disturbance" not in search


def test_the_two_coarse_searches_differ_only_in_the_take():
    by_hand, scripted = (load(EXPERIMENT / name) for name in SEARCHES)

    assert by_hand.pop("demonstrations") == {
        "run": "results/demonstrations/YYYYMMDD-HHMMSS-ellipse_manual_v2_no_limits",  # until the take is recorded
        "train": ["demo_00_filtered.sklog.npz"],
    }
    assert scripted.pop("demonstrations") == {
        "run": "results/demonstrations/20261118-171132-ellipse_scripted_v2_no_limits",
        "train": ["demo_00.sklog.npz"],  # not filtered
    }
    assert by_hand == scripted


def test_the_eight_starts_put_the_hand_on_the_curve_an_eighth_of_a_loop_apart_with_joint_2_positive():
    takes = load(TAKES)
    skeleton = Skeleton.from_toml(TAKES)
    curve = build_curve("ellipse", takes["task"])
    on_curve = load(EXPERIMENT / "hand_esn.toml")["evaluation"]["extra_starts"]["on_curve"]

    starts = takes["demonstrations"]["start_q"] + on_curve  # the take's own start first
    hands = endpoint_positions(skeleton, np.radians(starts))

    assert len(starts) == 8
    for k, (start, hand) in enumerate(zip(starts, hands, strict=True)):
        assert np.linalg.norm(hand - curve(np.radians(45.0 * k))) < 0.001
        assert start[1] > 0
        assert start == pytest.approx(np.round(start, 1).tolist())  # to 0.1 deg


def test_the_untaught_starts_are_stage_1s():
    stage_1 = load(STAGE_1 / "hand_esn.toml")["evaluation"]["extra_starts"]["untaught"]

    for name in ("hand_esn.toml", "hand_robot.toml"):
        assert load(EXPERIMENT / name)["evaluation"]["extra_starts"]["untaught"] == stage_1


def test_the_hand_esn_starts_from_report_023s_esn_1_and_runs_from_the_search_starts_and_the_untaught_ones():
    hand, esn1 = load(EXPERIMENT / "hand_esn.toml"), load(STAGE_1 / "esn1_lr0.055_sr1.2_is1_ridge1_seed4.toml")
    search = load(EXPERIMENT / "coarse_search.toml")

    assert hand["esn"] == esn1["esn"]
    assert hand["demonstrations"] == search["demonstrations"]
    assert hand["apps"] == load(STAGE_1 / "hand_esn.toml")["apps"]
    assert list(hand["evaluation"].pop("extra_starts")) == ["on_curve", "untaught"]
    search["evaluation"].pop("extra_starts")
    assert hand["evaluation"] == search["evaluation"]


def test_the_hand_robot_runs_the_hand_esn_with_the_searchs_tracker_against_the_replay_alone():
    robot, hand = load(EXPERIMENT / "hand_robot.toml"), load(EXPERIMENT / "hand_esn.toml")

    assert robot["esn"]["model"].endswith("-hand_esn/esn.toml")
    assert robot["tracker"] == load(EXPERIMENT / "coarse_search.toml")["tracker"]
    assert "progress" not in robot  # no progress replay in this stage
    assert robot["evaluation"] == hand["evaluation"]
    assert "disturbance" not in robot


# The scripted coarse search's best setting (run 20261118-191114-scripted_coarse_search): the fine search's center.
SCRIPTED_CENTER = {"leak_rate": 0.03, "spectral_radius": 1.5, "input_scaling": 2.0, "ridge": 0.01}


def test_the_scripted_fine_search_is_the_scripted_coarse_search_but_for_its_values_and_seeds():
    fine, coarse = load(EXPERIMENT / "scripted_fine_search.toml"), load(EXPERIMENT / "scripted_coarse_search.toml")

    fine_sweep, coarse_sweep = fine.pop("sweep"), coarse.pop("sweep")
    assert fine_sweep.pop("seed") == [1, 2, 3, 4, 5]
    assert list(fine_sweep) == list(coarse_sweep)
    fine_esn, coarse_esn = fine.pop("esn"), coarse.pop("esn")
    assert fine_esn == coarse_esn | SCRIPTED_CENTER | {"seed": 1}  # the center's values, and the first seed
    assert fine == coarse  # the same take, tracker, runs, and score


def test_the_scripted_fine_search_takes_the_center_and_the_values_halfway_to_its_coarse_neighbors():
    fine, coarse = load(EXPERIMENT / "scripted_fine_search.toml"), load(EXPERIMENT / "scripted_coarse_search.toml")

    for name, center in SCRIPTED_CENTER.items():
        grid = coarse["sweep"][name]
        k = grid.index(center)
        if name == "spectral_radius":  # halfway in difference
            # Beyond an edge of the grid, as far from the center as the neighbor on the other side.
            lower = grid[k - 1] if k > 0 else 2 * center - grid[k + 1]
            upper = grid[k + 1] if k + 1 < len(grid) else 2 * center - grid[k - 1]
            expected = [round((lower + center) / 2, 2), center, round((center + upper) / 2, 2)]
        else:  # halfway in ratio, to two digits
            lower = grid[k - 1] if k > 0 else center**2 / grid[k + 1]
            upper = grid[k + 1] if k + 1 < len(grid) else center**2 / grid[k - 1]
            expected = [float(f"{np.sqrt(lower * center):.2g}"), center, float(f"{np.sqrt(center * upper):.2g}")]
        assert fine["sweep"][name] == expected, name


ESN_NAMES = ["leak_rate", "spectral_radius", "input_scaling", "ridge"]


def test_the_scripted_untaught_search_is_the_scripted_fine_search_but_for_its_settings_seeds_and_untaught_starts():
    again, fine = load(EXPERIMENT / "scripted_untaught_search.toml"), load(EXPERIMENT / "scripted_fine_search.toml")

    settings, sweep, fine_sweep = again.pop("settings"), again.pop("sweep"), fine.pop("sweep")
    assert len(settings) == 20
    assert sweep == {"seed": fine_sweep["seed"]}  # seeds 1 to 5, the other values listed in settings
    starts = again["evaluation"]["extra_starts"]
    assert list(starts) == ["on_curve", "untaught"]
    starts.pop("untaught")
    assert again == fine  # the same take, ESN, tracker, runs, and score


def test_the_scripted_untaught_searchs_settings_are_distinct_and_on_the_fine_searchs_grid():
    again, fine = load(EXPERIMENT / "scripted_untaught_search.toml"), load(EXPERIMENT / "scripted_fine_search.toml")

    settings = [tuple(setting[name] for name in ESN_NAMES) for setting in again["settings"]]
    assert len(set(settings)) == len(settings) == 20
    for setting in again["settings"]:
        assert list(setting) == ESN_NAMES
        assert all(setting[name] in fine["sweep"][name] for name in ESN_NAMES)
    assert settings[:2] == [(0.017, 1.35, 4.0, 1e-3), (0.017, 1.5, 4.0, 1e-3)]  # the fine search's two best


def test_the_scripted_untaught_search_starts_on_the_fine_searchs_curve_and_from_stage_1s_untaught_starts():
    starts = load(EXPERIMENT / "scripted_untaught_search.toml")["evaluation"]["extra_starts"]
    fine = load(EXPERIMENT / "scripted_fine_search.toml")["evaluation"]["extra_starts"]

    assert starts["on_curve"] == fine["on_curve"]
    assert starts["untaught"] == load(EXPERIMENT / "hand_esn.toml")["evaluation"]["extra_starts"]["untaught"]
    assert starts["untaught"] == load(STAGE_1 / "hand_esn.toml")["evaluation"]["extra_starts"]["untaught"]
    assert len(starts["untaught"]) == 36
