# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""The consistent search's configurations follow its protocol (see its README.md)."""

import tomllib

import pytest

from arm_esn_ctrl.storage import REPO_ROOT

SEARCH = REPO_ROOT / "experiments/manual_demonstration_v2_consistent_search"

DEMONSTRATIONS = {
    "run": "results/demonstrations/20261008-170034-reach_manual_v2",
    "train": ["demo_00_filtered.sklog.npz"],
}
FIXED_ESN = {"dt": 0.01, "warmup": 1.0, "n_neurons": 400, "sparsity": 0.05, "bias": True, "ridge": 1e-2}
LEAKS = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
GRID = {"spectral_radius": [0.7, 0.9, 0.99, 1.2, 1.35, 1.5, 1.75], "input_scaling": [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]}
SEEDS = {"": list(range(10)), "_seeds10to19": list(range(10, 20))}
TRACKER = {
    "laws": ["pd"],
    "omegas": [20.0],
    "dampings": [0.4],
    "acceleration_filter": 0.02,
    "reference_velocity": True,
    "control_period": 0.01,
}
EVALUATION = {
    "duration": 24.0,
    "hold": 2.0,
    "start_offsets_deg": [[0.0, 0.0]],
    "return_tolerance": 0.005,
    "ringing_cutoff_hz": 1.5,
    "ringing_window": 2.0,
    "following_window": 0.5,
}
BLOCK = {"type": "block", "onset": 4.5, "release": 5.5, "stiffness": 20000.0, "damping": 100.0}
PUSHES = [
    {"type": "push", "force": 3.0, "onset": 3.0, "duration": 0.5, "angle_deg": 10.0},
    {"type": "push", "force": 5.0, "onset": 7.0, "duration": 0.5, "angle_deg": 80.0},
    {"type": "push", "force": 4.0, "onset": 15.0, "duration": 0.5, "angle_deg": -80.0},
]
# Each scenario's [evaluation] beyond EVALUATION, and its disturbance.
SCENARIOS = {
    "nominal": ({}, None),
    "block": ({"effort_window": [4.5, 6.0]}, BLOCK),
    "pushes": ({}, PUSHES),
    "offsets": ({"start_grid": {"span_deg": 15.0, "step_deg": 5.0}}, None),
}
GRID_NAMES = {
    f"sweep_lr{leak:g}_{scenario}{tag}": (leak, scenario, seeds)
    for leak in LEAKS
    for scenario in SCENARIOS
    for tag, seeds in SEEDS.items()
}


def load(name: str) -> dict:
    with (SEARCH / f"{name}.toml").open("rb") as f:
        return tomllib.load(f)


def test_the_grid_has_one_configuration_per_leak_rate_scenario_and_seeds():
    names = {path.stem for path in SEARCH.glob("sweep_lr*.toml")}

    assert names == set(GRID_NAMES)


@pytest.mark.parametrize("name", sorted(GRID_NAMES))
def test_a_grid_configuration_follows_the_protocol(name):
    leak, scenario, seeds = GRID_NAMES[name]
    extra, disturbance = SCENARIOS[scenario]
    config = load(name)

    assert config["demonstrations"] == DEMONSTRATIONS
    assert {key: config["esn"][key] for key in FIXED_ESN} == FIXED_ESN
    assert config["esn"]["leak_rate"] == leak
    assert config["sweep"] == GRID | {"seed": seeds}
    assert config["tracker"] == TRACKER
    assert config["evaluation"] == EVALUATION | extra
    assert config.get("disturbance") == disturbance


@pytest.mark.parametrize("scenario", ["block", "pushes"])
def test_the_guards_runs_meet_the_grids_tracker_and_disturbances(scenario):
    config, grid = load(f"guard_{scenario}_robot"), load(f"sweep_lr0.03_{scenario}")

    assert config["tracker"] == grid["tracker"]
    assert config["evaluation"] == grid["evaluation"]
    assert config["disturbance"] == grid["disturbance"]


def run_names(paths: list[str]) -> list[str]:
    """The configuration stems of runs, their directories' names but for the time stamps."""
    return [path.rsplit("-", 1)[1] for path in paths]


@pytest.mark.parametrize("tag", sorted(SEEDS))
def test_the_choice_reads_one_run_per_leak_rate_and_scenario(tag):
    config = load(f"select{tag}")

    assert set(config["runs"]) == {*SCENARIOS, "replay"}
    for scenario in SCENARIOS:
        assert run_names(config["runs"][scenario]) == [f"sweep_lr{leak:g}_{scenario}{tag}" for leak in LEAKS]
    assert run_names(config["runs"]["replay"]) == ["guard_block_robot", "guard_pushes_robot"]
    assert config["rule"] == {"chosen": 2, "fallback_seeds": 9}


CHECKS = {"size": {"n_neurons": [100, 200, 400, 800]}, "ridge": {"ridge": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]}}


@pytest.mark.parametrize("check", sorted(CHECKS))
def test_the_size_and_ridge_checks_are_measured_by_the_choices_rule(check):
    config, choice = load(f"select_{check}"), load("select")

    assert {scenario: run_names(config["runs"][scenario]) for scenario in SCENARIOS} == {
        scenario: [f"{check}_{scenario}"] for scenario in SCENARIOS
    }
    assert config["runs"]["replay"] == choice["runs"]["replay"]
    assert config["rule"] == choice["rule"]


@pytest.mark.parametrize("check", sorted(CHECKS))
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_the_size_and_ridge_checks_are_the_grid_at_esn_1s_setting(check, scenario):
    config, grid = load(f"{check}_{scenario}"), load(f"sweep_lr0.03_{scenario}")

    assert config["sweep"] == CHECKS[check] | {"seed": SEEDS[""]}
    assert {key: value for key, value in config.items() if key != "sweep"} == {
        key: value for key, value in grid.items() if key != "sweep"
    }
    assert (config["esn"]["spectral_radius"], config["esn"]["input_scaling"]) == (1.2, 1.5)


def without(config: dict, *keys: str) -> dict:
    """A configuration but for some of its tables."""
    return {key: value for key, value in config.items() if key not in keys}


@pytest.mark.parametrize(("esn", "scaling"), [("esn1", 1.5), ("esn2", 1.0)])
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_the_bias_check_at_a_chosen_esns_setting_is_the_grid_with_a_scaled_bias(esn, scaling, scenario):
    config, grid = load(f"bias_{esn}_{scenario}"), load(f"sweep_lr0.03_{scenario}")

    assert config["sweep"] == {"bias_scaling": [0.0, 0.25, 0.5, 1.0, 2.0], "seed": SEEDS[""]}
    assert config["esn"] == grid["esn"] | {"input_scaling": scaling, "bias_scaling": 1.0}
    assert without(config, "sweep", "esn") == without(grid, "sweep", "esn")


BIAS_REGION = {"spectral_radius": [0.99, 1.2, 1.35, 1.5], "input_scaling": [0.5, 1.0, 1.5, 2.0]}


@pytest.mark.parametrize("bias", ["0", "0.5", "2"])
@pytest.mark.parametrize("leak", [0.015, 0.02, 0.03, 0.04])
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_the_bias_check_around_the_qualifying_region_is_the_grid_with_a_scaled_bias(bias, leak, scenario):
    config, grid = load(f"bias{bias}_lr{leak:g}_{scenario}"), load(f"sweep_lr{leak:g}_{scenario}")

    assert config["sweep"] == BIAS_REGION | {"seed": SEEDS[""]}
    assert config["esn"] == grid["esn"] | {"bias_scaling": float(bias)}
    assert without(config, "sweep", "esn") == without(grid, "sweep", "esn")


def test_there_are_no_other_bias_checks():
    names = {path.stem for path in SEARCH.glob("bias*.toml")}

    assert len(names) == 8 + 48


CHOSEN = {"esn1_lr0.03_sr1.2_is1.5_seed8": (1.5, 8), "esn2_lr0.03_sr1.2_is1_seed2": (1.0, 2)}


@pytest.mark.parametrize("esn", sorted(CHOSEN))
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_the_chosen_esns_run_on_the_robot_as_in_the_grid(esn, scenario):
    config, grid = load(f"{scenario}_robot_{esn}"), load(f"sweep_lr0.03_{scenario}")

    assert config["esn"]["model"].startswith("results/manual_demonstration_v2_consistent_search/")
    assert config["esn"]["model"].endswith(f"-{esn}/esn.toml")
    assert config["progress"] == {"lead_deg": 2.5, "window": 1.0}
    assert config["tracker"] == grid["tracker"]
    assert config["evaluation"] == grid["evaluation"]
    assert config.get("disturbance") == grid.get("disturbance")


@pytest.mark.parametrize(("name", "scaling", "seed"), [(name, *chosen) for name, chosen in CHOSEN.items()])
def test_the_chosen_esns_are_trained_with_the_protocols_settings(name, scaling, seed):
    config = load(name)

    assert config["demonstrations"] == DEMONSTRATIONS
    chosen = {"leak_rate": 0.03, "spectral_radius": 1.2, "input_scaling": scaling, "seed": seed}
    assert config["esn"] == FIXED_ESN | chosen
