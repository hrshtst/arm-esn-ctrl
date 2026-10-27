# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Part 2's configurations run the search's chosen ESNs with its protocol's trackers (see the search's README.md)."""

import tomllib
from pathlib import Path

import pytest

from arm_esn_ctrl.storage import REPO_ROOT

SEARCH = REPO_ROOT / "experiments/manual_demonstration_v2_consistent_search"
TRACKERS = REPO_ROOT / "experiments/manual_demonstration_v2_consistent_trackers"

ESNS = ["esn1_lr0.03_sr1.2_is1.5_seed8", "esn2_lr0.03_sr1.2_is1_seed2"]
SCENARIOS = ["nominal", "block", "pushes", "offsets"]
RECORDED = ["recorded000", "recorded001", "recorded002"]
# Each reference velocity's laws: computed torque is given the reference's velocity only.
VELOCITIES = {"reference": (True, ["pd", "computed_torque"]), "zero": (False, ["pd"])}
NAMES = {
    f"{scenario}_robot_{esn}_{velocity}_velocity": (scenario, esn, velocity)
    for scenario in SCENARIOS + RECORDED
    for esn in ESNS
    for velocity in VELOCITIES
}


def load(directory: Path, name: str) -> dict:
    with (directory / f"{name}.toml").open("rb") as f:
        return tomllib.load(f)


def without(config: dict, *keys: str) -> dict:
    """A configuration but for some of its tables."""
    return {key: value for key, value in config.items() if key not in keys}


def test_there_is_one_configuration_per_scenario_esn_and_reference_velocity():
    names = {path.stem for path in TRACKERS.glob("*.toml")}

    assert names == set(NAMES)


@pytest.mark.parametrize("name", sorted(NAMES))
def test_the_trackers_are_the_searchs_but_for_the_laws_damping_ratios_and_reference_velocity(name):
    _, esn, velocity = NAMES[name]
    reference_velocity, laws = VELOCITIES[velocity]
    config, search = load(TRACKERS, name), load(SEARCH, f"nominal_robot_{esn}")

    assert config["tracker"] == search["tracker"] | {
        "laws": laws,
        "dampings": [0.2, 0.4, 0.8],
        "reference_velocity": reference_velocity,
    }


@pytest.mark.parametrize("name", sorted(NAMES))
def test_the_arms_are_the_search_choices_and_its_replays(name):
    _, esn, _ = NAMES[name]
    config, search = load(TRACKERS, name), load(SEARCH, f"nominal_robot_{esn}")

    assert config["esn"] == search["esn"]
    assert config["progress"] == search["progress"]


@pytest.mark.parametrize("name", [name for name, (scenario, _, _) in NAMES.items() if scenario in SCENARIOS])
def test_the_searchs_scenarios_are_run_as_in_the_search(name):
    scenario, esn, _ = NAMES[name]
    config, search = load(TRACKERS, name), load(SEARCH, f"{scenario}_robot_{esn}")

    assert without(config, "tracker") == without(search, "tracker")


@pytest.mark.parametrize("name", [name for name, (scenario, _, _) in NAMES.items() if scenario in RECORDED])
def test_the_recorded_pushes_replace_the_scripted_ones_from_the_logs_start(name):
    scenario, esn, _ = NAMES[name]
    config, search = load(TRACKERS, name), load(SEARCH, f"pushes_robot_{esn}")

    assert config["disturbance"] == {
        "type": "recorded",
        "log": f"data/recorded_forces/recorded_forces_{scenario.removeprefix('recorded')}.sklog.npz",
    }
    assert config["evaluation"] == without(search["evaluation"], "start_offsets_deg")
    assert "plots" not in config
