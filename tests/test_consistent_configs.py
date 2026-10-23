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
