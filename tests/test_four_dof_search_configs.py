# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""The four-link consistent search's configurations are v2's, with the changes its README.md fixes."""

import tomllib

import numpy as np
import pytest
from skelarm import Skeleton

from arm_esn_ctrl.autonomous import start_postures
from arm_esn_ctrl.postures import joint_limits_deg
from arm_esn_ctrl.storage import REPO_ROOT

SEARCH = REPO_ROOT / "experiments/manual_demonstration_four_dof_consistent_search"
V2 = REPO_ROOT / "experiments/manual_demonstration_v2_consistent_search"
ARM = REPO_ROOT / "experiments/demonstrations/reach_manual_four_dof.toml"

DEMONSTRATIONS = {
    "run": "results/demonstrations/20261103-134628-reach_manual_four_dof",
    "train": ["demo_00_filtered.sklog.npz"],
}
LEAKS = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
SCENARIOS = ["nominal", "block", "pushes", "offsets"]
GRID_NAMES = [
    f"sweep_lr{leak:g}_{scenario}{tag}" for leak in LEAKS for scenario in SCENARIOS for tag in ["", "_seeds10to19"]
]

# The take's times by the README's rule: the take (demo_00_filtered) arrives at 9.51 s
# and lasts 21.43 s, and its hand is 18%, 32%, and 62% of the way to the target at
# 2.64, 4.02, and 6.69 s.
TAKE_LENGTH = 21.43
BLOCK = {"onset": 4.0, "release": 5.0}
PUSH_ONSETS = [2.6, 6.7, 14.9]
DURATION = 23.9


def load(path):
    with path.open("rb") as f:
        return tomllib.load(f)


def test_the_grid_has_one_configuration_per_leak_rate_scenario_and_seeds():
    names = {path.stem for path in SEARCH.glob("sweep_lr*.toml")}

    assert names == set(GRID_NAMES)
    assert names == {path.stem for path in V2.glob("sweep_lr*.toml")}


@pytest.mark.parametrize("name", GRID_NAMES)
def test_a_grid_configuration_is_v2s_on_the_four_link_take(name):
    config, v2 = load(SEARCH / f"{name}.toml"), load(V2 / f"{name}.toml")

    assert config["demonstrations"] == DEMONSTRATIONS
    assert config["esn"] == v2["esn"]
    assert config["sweep"] == v2["sweep"]
    assert config["tracker"] == v2["tracker"] | {"mass_matrix": True}
    changed = {"duration": DURATION, "start_offsets_deg": [[0.0, 0.0, 0.0, 0.0]]}
    if "effort_window" in v2["evaluation"]:
        changed["effort_window"] = [BLOCK["onset"], BLOCK["release"] + 0.5]
    if "start_grid" in v2["evaluation"]:
        changed["start_grid"] = {"span_deg": 15.0, "step_deg": 15.0}
    assert config["evaluation"] == v2["evaluation"] | changed
    disturbance = v2.get("disturbance")
    if isinstance(disturbance, dict):
        disturbance = disturbance | BLOCK
    elif disturbance is not None:
        disturbance = [push | {"onset": onset} for push, onset in zip(disturbance, PUSH_ONSETS, strict=True)]
    assert config.get("disturbance") == disturbance


@pytest.mark.parametrize("leak", LEAKS)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_the_extended_grid_is_the_grid_below_its_input_scalings(leak, scenario):
    config = load(SEARCH / f"extend_lr{leak:g}_{scenario}.toml")
    grid = load(SEARCH / f"sweep_lr{leak:g}_{scenario}.toml")

    assert max(config["sweep"]["input_scaling"]) < min(grid["sweep"]["input_scaling"])
    assert config == grid | {"sweep": grid["sweep"] | {"input_scaling": [0.02, 0.05]}}


def test_the_extended_grid_has_one_configuration_per_leak_rate_and_scenario():
    names = {path.stem for path in SEARCH.glob("extend_*.toml")}

    assert names == {f"extend_lr{leak:g}_{scenario}" for leak in LEAKS for scenario in SCENARIOS}


def test_the_times_follow_the_take():
    assert BLOCK["release"] == BLOCK["onset"] + 1.0
    assert PUSH_ONSETS[0] < BLOCK["onset"] < PUSH_ONSETS[1]
    assert np.isclose(PUSH_ONSETS[2], round(9.51 + 5.4, 1))  # 5.4 s after the take arrives
    assert np.isclose(DURATION, max(PUSH_ONSETS[2] + 9.0, TAKE_LENGTH))


def test_the_offset_starts_are_the_36_within_the_joint_limits():
    evaluation = load(SEARCH / "sweep_lr0.03_offsets.toml")["evaluation"]
    start = np.radians(load(ARM)["initial"]["q"])
    starts = start_postures(evaluation, {"demo_00": start[np.newaxis]}, joint_limits_deg(Skeleton.from_toml(ARM)))

    assert len(starts) == 36
    np.testing.assert_array_equal(starts[0].q, start)


def test_the_guards_esn_is_v2s_esn_1_setting_on_this_take():
    config = load(SEARCH / "guard_esn_lr0.03_sr1.2_is1.5_seed8.toml")
    v2 = load(V2 / "esn1_lr0.03_sr1.2_is1.5_seed8.toml")

    assert config["demonstrations"] == DEMONSTRATIONS
    assert config["esn"] == v2["esn"]
    assert config["evaluation"] == {
        "duration": DURATION,
        "hold": 2.0,
        "start_grid": {"span_deg": 15.0, "step_deg": 15.0},
    }


def run_names(runs):
    """The runs' configurations: their directories but for the date and time."""
    directory = "results/manual_demonstration_four_dof_consistent_search/"
    assert all(run.startswith(directory) for run in runs)
    return [run.removeprefix(directory).split("-", 2)[2] for run in runs]


@pytest.mark.parametrize(("name", "grids"), [("select", ["sweep"]), ("select_extended", ["sweep", "extend"])])
def test_the_choice_is_v2s_rule_over_the_grids_with_seeds_0_to_9_and_the_guards_runs(name, grids):
    config, v2 = load(SEARCH / f"{name}.toml"), load(V2 / "select.toml")

    assert config["rule"] == v2["rule"]
    assert config["runs"].keys() == v2["runs"].keys()
    for scenario in SCENARIOS:
        assert run_names(config["runs"][scenario]) == [
            f"{grid}_lr{leak:g}_{scenario}" for grid in grids for leak in LEAKS
        ]
    assert run_names(config["runs"]["replay"]) == ["guard_block_robot", "guard_pushes_robot"]


def test_the_following_rule_is_over_the_same_runs_with_the_three_changes_of_step_5():
    config, extended = load(SEARCH / "select_following.toml"), load(SEARCH / "select_extended.toml")

    assert config["runs"] == extended["runs"]
    assert config["rule"] == {
        "chosen": 2,
        "fallback_seeds": 8,
        "qualifying_seeds": 8,
        "excuse_leaving": ["pushes", "offsets"],
        "max_block_gap": 0.5,
        "rank_by": "block_gap",
    }


CHOSEN = {  # select run 20261108-051827-select_extended's ESN #1 and ESN #2
    "esn1_lr0.01_sr1.5_is0.02_seed7": {"leak_rate": 0.01, "spectral_radius": 1.5, "input_scaling": 0.02, "seed": 7},
    "esn2_lr0.02_sr1.5_is0.02_seed8": {"leak_rate": 0.02, "spectral_radius": 1.5, "input_scaling": 0.02, "seed": 8},
}


FOLLOWING = {  # select run 20261109-202923-select_following's ESN #1 and ESN #2
    "following_esn1_lr0.015_sr1.2_is0.05_seed0": {
        "leak_rate": 0.015,
        "spectral_radius": 1.2,
        "input_scaling": 0.05,
        "seed": 0,
    },
    "following_esn2_lr0.02_sr1.35_is0.05_seed7": {
        "leak_rate": 0.02,
        "spectral_radius": 1.35,
        "input_scaling": 0.05,
        "seed": 7,
    },
}
QUALIFYING = [  # the 8 settings that qualify in that run: (leak rate, spectral radius, input scaling)
    (0.015, 1.2, 0.05),
    (0.02, 1.35, 0.05),
    (0.02, 1.2, 0.02),
    (0.015, 1.35, 0.05),
    (0.01, 1.2, 0.05),
    (0.015, 1.2, 0.02),
    (0.01, 0.99, 0.02),
    (0.02, 1.35, 0.02),
]


@pytest.mark.parametrize("name", [*CHOSEN, *FOLLOWING])
def test_the_chosen_esns_are_the_guards_esn_at_their_settings(name):
    config = load(SEARCH / f"{name}.toml")
    guard = load(SEARCH / "guard_esn_lr0.03_sr1.2_is1.5_seed8.toml")

    assert config == guard | {"esn": guard["esn"] | (CHOSEN | FOLLOWING)[name]}


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize(("leak", "radius", "scaling"), QUALIFYING)
def test_the_held_out_seeds_run_each_qualifying_setting_as_the_extension_did(leak, radius, scaling, scenario):
    config = load(SEARCH / f"heldout_lr{leak:g}_sr{radius:g}_is{scaling:g}_{scenario}.toml")
    extension = load(SEARCH / f"extend_lr{leak:g}_{scenario}.toml")
    sweep = {"spectral_radius": [radius], "input_scaling": [scaling], "seed": list(range(10, 20))}

    assert config == extension | {"sweep": sweep}


def test_the_held_out_configurations_are_those_of_the_qualifying_settings():
    expected = {
        f"heldout_lr{leak:g}_sr{radius:g}_is{scaling:g}_{scenario}"
        for leak, radius, scaling in QUALIFYING
        for scenario in SCENARIOS
    }
    assert {path.stem for path in SEARCH.glob("heldout_*.toml")} == expected


@pytest.mark.parametrize("name", [*CHOSEN, *FOLLOWING])
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_the_chosen_esns_run_on_the_robot_as_in_v2s_step_4_with_the_grids_tracker_and_scenarios(scenario, name):
    config = load(SEARCH / f"{scenario}_robot_{name}.toml")
    grid = load(SEARCH / f"sweep_lr0.01_{scenario}.toml")
    v2 = load(V2 / f"{scenario}_robot_esn1_lr0.03_sr1.2_is1.5_seed8.toml")

    assert config["esn"]["model"].startswith("results/manual_demonstration_four_dof_consistent_search/")
    assert config["esn"]["model"].endswith(f"-{name}/esn.toml")
    assert config["tracker"] == grid["tracker"]
    assert config["progress"] == v2["progress"]
    assert config["evaluation"] == grid["evaluation"]
    assert config.get("disturbance") == grid.get("disturbance")
    if scenario == "offsets":
        assert config["plots"] == {"offsets_deg": [[15.0, 15.0, 15.0, 15.0], [0.0, 0.0, -15.0, -15.0]]}
    assert config.keys() == v2.keys()


@pytest.mark.parametrize("scenario", ["block", "pushes"])
def test_the_guards_runs_meet_the_grids_tracker_and_disturbances_beside_the_guards_esn(scenario):
    config, grid = load(SEARCH / f"guard_{scenario}_robot.toml"), load(SEARCH / f"sweep_lr0.03_{scenario}.toml")

    assert config["esn"]["model"].endswith("-guard_esn_lr0.03_sr1.2_is1.5_seed8/esn.toml")
    assert config["tracker"] == grid["tracker"]
    assert config["evaluation"] == grid["evaluation"]
    assert config["disturbance"] == grid["disturbance"]
