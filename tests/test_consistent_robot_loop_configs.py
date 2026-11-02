# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Report 020's configurations read report 019's ESNs and runs with the robot (see the experiment's README.md)."""

import tomllib
from pathlib import Path

from arm_esn_ctrl.storage import REPO_ROOT

DYNAMICS = REPO_ROOT / "experiments/manual_demonstration_v2_consistent_dynamics"
ROBOT_LOOP = REPO_ROOT / "experiments/manual_demonstration_v2_consistent_robot_loop"
STUDY = "results/manual_demonstration_v2_consistent_dynamics"


def load(directory: Path, name: str) -> dict:
    with (directory / f"{name}.toml").open("rb") as f:
        return tomllib.load(f)


def test_from_rest_anywhere_is_report_019s_step_1_on_the_robot_with_the_searchs_tracker():
    config, earlier = load(ROBOT_LOOP, "robot_starts"), load(DYNAMICS, "global_starts")

    assert config["esn"] == earlier["esn"]
    assert config["grid"] == earlier["grid"]
    assert config["evaluation"] == earlier["evaluation"]
    assert config["analysis"].items() <= earlier["analysis"].items()
    assert config["tracker"] == load(DYNAMICS, "push_pulses")["tracker"]


def test_the_chosen_esns_with_the_arm_read_report_019s_pulses_and_fixed_points():
    config, earlier = load(ROBOT_LOOP, "robot_fixed_points"), load(DYNAMICS, "fixed_points")

    assert config["runs"]["pulses"] == f"{STUDY}/20261027-180441-push_pulses"
    assert config["runs"]["fixed_points"] == f"{STUDY}/20261027-183137-fixed_points"
    assert config["rates"] == {key: earlier["analysis"][key] for key in ("windows", "stride", "n_rates")}
    assert config["linear"]["window"] == load(DYNAMICS, "push_pulses")["pulses"]["response_window"]


def test_the_screens_with_the_arm_read_report_019s_screens_outcomes_and_releases():
    config = load(ROBOT_LOOP, "screen_robot_loop")
    against, heldout = load(DYNAMICS, "against_robot"), load(DYNAMICS, "against_robot_seeds10to19")
    release = load(DYNAMICS, "release_robot")

    assert config["runs"]["screens"] == [against["runs"]["screen"], heldout["runs"]["screen"]]
    assert config["runs"]["outcomes"] == [
        release["runs"]["outcomes"],
        f"{STUDY}/20261028-115552-against_robot_seeds10to19",
    ]
    assert config["runs"]["release"] == f"{STUDY}/20261028-044731-release_robot"
    assert config["tracker"] == release["tracker"]


def test_the_nudged_releases_are_report_019s_releases():
    config = load(ROBOT_LOOP, "nudge_releases")

    assert config["runs"]["release"] == load(ROBOT_LOOP, "screen_robot_loop")["runs"]["release"]
    assert 0.0 < config["nudge"]["deg"] <= 1e-6


def test_the_weighed_rates_are_step_2s_over_its_claimed_windows():
    config, step2 = load(ROBOT_LOOP, "robot_rates_weighed"), load(ROBOT_LOOP, "robot_fixed_points")

    assert config["runs"]["pulses"] == step2["runs"]["pulses"]
    assert config["rates"] == {"windows": [2.0, 4.0], "stride": step2["rates"]["stride"]}
    assert [1.0, 1.0] in config["weighings"]["factors"]
