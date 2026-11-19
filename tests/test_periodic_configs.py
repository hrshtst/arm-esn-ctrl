# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""The periodic motion's configurations agree with each other, with the ellipse's takes, and with stage 1's."""

import re
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
        "run": "results/demonstrations/20261119-093746-ellipse_manual_v2_no_limits",
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
# The coarse search's best setting on the take by hand (run 20261119-093837-coarse_search): its fine search's center.
HAND_CENTER = {"leak_rate": 0.03, "spectral_radius": 0.9, "input_scaling": 0.5, "ridge": 1.0}
# Each take's fine search, with its coarse search and its center.
FINE_SEARCHES = {
    "fine_search.toml": ("coarse_search.toml", HAND_CENTER),
    "scripted_fine_search.toml": ("scripted_coarse_search.toml", SCRIPTED_CENTER),
}


@pytest.mark.parametrize("search", FINE_SEARCHES)
def test_each_fine_search_is_its_coarse_search_but_for_its_values_and_seeds(search):
    coarse_name, center = FINE_SEARCHES[search]
    fine, coarse = load(EXPERIMENT / search), load(EXPERIMENT / coarse_name)

    fine_sweep, coarse_sweep = fine.pop("sweep"), coarse.pop("sweep")
    assert fine_sweep.pop("seed") == [1, 2, 3, 4, 5]
    assert list(fine_sweep) == list(coarse_sweep)
    fine_esn, coarse_esn = fine.pop("esn"), coarse.pop("esn")
    assert fine_esn == coarse_esn | center | {"seed": 1}  # the center's values, and the first seed
    assert fine == coarse  # the same take, tracker, runs, and score


@pytest.mark.parametrize("search", FINE_SEARCHES)
def test_each_fine_search_takes_the_center_and_the_values_halfway_to_its_coarse_neighbors(search):
    coarse_name, setting = FINE_SEARCHES[search]
    fine, coarse = load(EXPERIMENT / search), load(EXPERIMENT / coarse_name)

    for name, center in setting.items():
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


# The scripted take's chosen ESNs, the typical seeds of the two best settings of the scoring with the untaught starts
# (run 20261118-205822-scripted_untaught_search): the stems of their configurations.
SCRIPTED_ESNS = {
    1: "scripted_esn1_lr0.017_sr1.35_is4_ridge0.001_seed5",
    2: "scripted_esn2_lr0.017_sr1.5_is4_ridge0.001_seed5",
}


@pytest.mark.parametrize("n", [1, 2])
def test_the_scripted_takes_esn_n_is_trained_as_the_untaught_searchs_esns_with_its_nth_setting_and_seed_5(n):
    esn, search = load(EXPERIMENT / f"{SCRIPTED_ESNS[n]}.toml"), load(EXPERIMENT / "scripted_untaught_search.toml")

    assert esn["demonstrations"] == search["demonstrations"]  # the scripted take, as it is made
    assert esn["esn"] == search["esn"] | search["settings"][n - 1] | {"seed": 5}


def test_the_scripted_takes_two_esns_differ_only_in_the_spectral_radius():
    one, two = (load(EXPERIMENT / f"{stem}.toml") for stem in SCRIPTED_ESNS.values())

    assert (one["esn"].pop("spectral_radius"), two["esn"].pop("spectral_radius")) == (1.35, 1.5)
    assert one == two


@pytest.mark.parametrize("n", [1, 2])
def test_the_scripted_takes_esns_run_on_their_own_as_the_hand_esn(n):
    esn, hand = load(EXPERIMENT / f"{SCRIPTED_ESNS[n]}.toml"), load(EXPERIMENT / "hand_esn.toml")

    # For 40 s, from the eight starts on the curve and the 36 untaught ones, with the same apps' commands.
    assert esn["evaluation"] == hand["evaluation"]
    assert esn["apps"] == hand["apps"]


# The analysis of the scripted take's two ESNs: each kind of run, with a configuration for ESN #1 and one for ESN #2.
CLAIM_RUNS = [
    "scripted_robot",  # claims 1 and 2
    "scripted_untaught_robot",  # claims 2 and 3
    "scripted_pushes1x_robot",  # claim 4
    "scripted_pushes2x_robot",
    "scripted_pushes4x_robot",
    "scripted_block_robot",  # claim 5
]
# Report 017's runs of report 007's pushes and of its block.
CONSISTENT = REPO_ROOT / "experiments/manual_demonstration_v2_consistent_search"
REPORT_017_PUSHES = CONSISTENT / "pushes_robot_esn1_lr0.03_sr1.2_is1.5_seed8.toml"
REPORT_017_BLOCK = CONSISTENT / "block_robot_esn1_lr0.03_sr1.2_is1.5_seed8.toml"
# The time stamp of an ESN's run: a placeholder until the ESN is trained, then the run's.
STAMP = r"(YYYYMMDD-HHMMSS|\d{8}-\d{6})"


@pytest.mark.parametrize("name", CLAIM_RUNS)
def test_each_claims_run_takes_each_esn_from_the_run_of_its_configuration_and_is_alike_for_both(name):
    one, two = load(EXPERIMENT / f"{name}_esn1.toml"), load(EXPERIMENT / f"{name}_esn2.toml")

    for n, config in ((1, one), (2, two)):
        esn = config.pop("esn")
        assert list(esn) == ["model"]  # esn.toml in the run of the ESN's configuration
        run = rf"results/manual_demonstration_v2_periodic/{STAMP}-{re.escape(SCRIPTED_ESNS[n])}"
        assert re.fullmatch(rf"{run}/esn\.toml", esn["model"])
    assert one == two


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("name", CLAIM_RUNS)
def test_each_claims_run_has_the_hand_robots_tracker_and_cycle_against_the_replay_alone(name, n):
    config, hand = load(EXPERIMENT / f"{name}_esn{n}.toml"), load(EXPERIMENT / "hand_robot.toml")

    assert config["tracker"] == hand["tracker"]
    assert {key: config["evaluation"][key] for key in CYCLE} == {key: hand["evaluation"][key] for key in CYCLE} == CYCLE
    assert "progress" not in config  # no progress replay in this stage


def test_the_undisturbed_runs_on_the_curve_last_120_s_from_the_eight_starts_on_the_curve():
    config, search = load(EXPERIMENT / "scripted_robot_esn1.toml"), load(EXPERIMENT / "scripted_fine_search.toml")

    # The take's own start and the seven others on the curve, as in the search, for 30 taught loops.
    assert config["evaluation"] == search["evaluation"] | {"duration": 120.0}
    assert "disturbance" not in config


def test_the_untaught_runs_last_60_s_from_stage_1s_36_untaught_starts_alone():
    config = load(EXPERIMENT / "scripted_untaught_robot_esn1.toml")
    untaught = load(STAGE_1 / "hand_esn.toml")["evaluation"]["extra_starts"]["untaught"]

    assert len(untaught) == 36
    assert config["evaluation"] == {
        "duration": 60.0,
        **CYCLE,
        "start_offsets_deg": [],  # not from the take's own start
        "extra_starts": {"untaught": untaught},
    }
    assert "disturbance" not in config


@pytest.mark.parametrize("times", [1, 2, 4])
def test_the_pushes_are_report_007s_three_at_8_20_5_and_33_s_at_one_two_and_four_times_their_forces(times):
    pushes = load(EXPERIMENT / f"scripted_pushes{times}x_robot_esn1.toml")["disturbance"]
    theirs = load(REPORT_017_PUSHES)["disturbance"]

    assert [(push["force"], push["angle_deg"], push["duration"]) for push in pushes] == [
        (3.0 * times, 10.0, 0.5),
        (5.0 * times, 80.0, 0.5),
        (4.0 * times, -80.0, 0.5),
    ]
    assert [push["onset"] for push in pushes] == [8.0, 20.5, 33.0]  # 12.5 s apart, three loops and an eighth
    for push, their in zip(pushes, theirs, strict=True):
        assert push.pop("force") == times * their.pop("force")
        push.pop("onset")
        their.pop("onset")
        assert push == their  # a push of the same type, duration, and angle


@pytest.mark.parametrize("n", [1, 2])
def test_the_three_sizes_of_pushes_differ_only_in_their_forces(n):
    sizes = [load(EXPERIMENT / f"scripted_pushes{times}x_robot_esn{n}.toml") for times in (1, 2, 4)]

    for config in sizes:
        for push in config["disturbance"]:
            push.pop("force")
    assert sizes[0] == sizes[1] == sizes[2]


@pytest.mark.parametrize(("name", "duration"), [("scripted_pushes1x_robot", 80.0), ("scripted_block_robot", 60.0)])
def test_the_pushed_and_blocked_runs_come_back_within_20_mm_of_the_curve_from_the_eight_starts_on_it(name, duration):
    config, robot = load(EXPERIMENT / f"{name}_esn1.toml"), load(EXPERIMENT / "scripted_robot_esn1.toml")
    stage_1 = load(STAGE_1 / "pushes1x_60s_robot_esn1.toml")["evaluation"]

    evaluation = config["evaluation"]
    assert evaluation.pop("return_tolerance") == 0.02  # m: back within 20 mm of the curve
    assert evaluation.pop("after_window") == 2.0  # s: the largest torque and off the curve after a disturbance
    for key in ("ringing_cutoff_hz", "ringing_window", "following_window"):  # stage 1's other measures after one
        assert evaluation.pop(key) == stage_1[key]
    assert evaluation == robot["evaluation"] | {"duration": duration}


def test_the_block_holds_the_tip_from_10_s_to_13_s_with_report_017s_stiffness_and_damping():
    block = load(EXPERIMENT / "scripted_block_robot_esn1.toml")["disturbance"]
    theirs = load(REPORT_017_BLOCK)["disturbance"]

    assert (theirs["stiffness"], theirs["damping"]) == (20000.0, 100.0)
    # The same block, held 3 s, three quarters of a loop, from 10 s.
    assert block == theirs | {"onset": 10.0, "release": 13.0}
