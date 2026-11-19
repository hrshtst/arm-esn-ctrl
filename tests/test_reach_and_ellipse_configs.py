# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""The reach and ellipse experiment's configurations agree with stage 1's, stage 2's, and stage 3a's."""

import itertools
import tomllib

import numpy as np
import pytest

from arm_esn_ctrl.autonomous import task_evaluations, task_names
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.storage import REPO_ROOT
from arm_esn_ctrl.sweeps import sweep_combinations

EXPERIMENT = REPO_ROOT / "experiments/manual_demonstration_v2_reach_and_ellipse"
MANY_STARTS = REPO_ROOT / "experiments/manual_demonstration_v2_many_starts"  # stage 1's
TARGETS = REPO_ROOT / "experiments/manual_demonstration_v2_targets"  # stage 2's
PERIODIC = REPO_ROOT / "experiments/manual_demonstration_v2_periodic"  # stage 3a's
CONTEXT = ("context_scaling", "context_times_input_scaling")  # the [esn] keys of the context


def load(path):
    with path.open("rb") as f:
        return tomllib.load(f)


def header_command(path):
    """The command a configuration's header comment gives, its continued lines joined, as its words."""
    header = itertools.takewhile(lambda line: line.startswith("#"), path.read_text().splitlines())
    lines = [line.lstrip("#").strip() for line in header]
    first = next(i for i, line in enumerate(lines) if line.startswith("uv run "))
    words = []
    for line in lines[first:]:
        words += line.removesuffix("\\").split()
        if not line.endswith("\\"):
            break
    return words


def test_the_coarse_search_trains_on_stage_1s_nine_takes_as_reach_and_stage_3as_take_by_hand_as_ellipse():
    reach, ellipse = load(EXPERIMENT / "coarse_search.toml")["demonstrations"]

    assert reach.pop("context") == "reach"
    assert reach == load(MANY_STARTS / "coarse_search.toml")["demonstrations"]
    assert reach["run"] == "results/demonstrations/20261117-164658-reach_manual_v2_no_limits"
    assert reach["train"] == [f"demo_{i:02d}_filtered.sklog.npz" for i in range(9)]
    assert ellipse.pop("context") == "ellipse"
    assert ellipse == load(PERIODIC / "coarse_search.toml")["demonstrations"]
    assert ellipse == {
        "run": "results/demonstrations/20261119-093746-ellipse_manual_v2_no_limits",
        "train": ["demo_00_filtered.sklog.npz"],
    }


def test_the_coarse_search_parses_as_a_sweep_of_stage_1s_108_settings_of_one_esn_of_two_tasks_seed_0():
    config = load(EXPERIMENT / "coarse_search.toml")

    assert task_names(config["demonstrations"]) == ("reach", "ellipse")
    assert config["sweep"] == load(MANY_STARTS / "coarse_search.toml")["sweep"]
    assert config["sweep"] == load(PERIODIC / "coarse_search.toml")["sweep"]
    combinations = sweep_combinations(config["sweep"], config.get("settings"))
    assert len(combinations) == 108
    for combination in combinations:
        EsnConfig(**config["esn"] | combination)
    esn = EsnConfig(**config["esn"])
    assert (esn.n_neurons, esn.seed) == (800, 0)


def test_the_coarse_searchs_esn_is_stage_1s_and_3as_with_stage_2s_context():
    config = load(EXPERIMENT / "coarse_search.toml")
    targets = load(TARGETS / "coarse_search.toml")

    esn = {key: value for key, value in config["esn"].items() if key not in CONTEXT}
    assert esn == load(MANY_STARTS / "coarse_search.toml")["esn"]
    assert esn == load(PERIODIC / "coarse_search.toml")["esn"]
    assert {key: config["esn"][key] for key in CONTEXT} == {key: targets["esn"][key] for key in CONTEXT}
    assert (config["esn"]["context_scaling"], config["esn"]["context_times_input_scaling"]) == (1.0, False)
    assert not set(CONTEXT) & set(config["sweep"])
    # Whatever the input scaling, the context's input weights, times it, have the scale 1.
    for input_scaling in config["sweep"]["input_scaling"]:
        model = ReachingEsn(EsnConfig(**config["esn"] | {"input_scaling": input_scaling}))
        assert model.context_gain * input_scaling == pytest.approx(1.0)


def test_the_coarse_search_tracks_as_stage_1s_and_3as_and_ranks_by_the_worse_of_the_two_tasks_ranks():
    config = load(EXPERIMENT / "coarse_search.toml")

    assert config["tracker"] == load(MANY_STARTS / "coarse_search.toml")["tracker"]
    assert config["tracker"] == load(PERIODIC / "coarse_search.toml")["tracker"]
    assert config["ranking"] == {"by": "worse_rank"}
    assert "disturbance" not in config


def test_each_tasks_runs_are_those_of_the_coarse_search_of_the_stage_it_comes_from():
    config = load(EXPERIMENT / "coarse_search.toml")

    reach, ellipse = task_evaluations(config["evaluation"], task_names(config["demonstrations"]))

    # The reach from the nine takes' starts for 24 s with stage 1's hold; the ellipse from the eight starts on the
    # curve for 40 s with 3a's cycle: 17 runs.
    assert reach == load(MANY_STARTS / "coarse_search.toml")["evaluation"]
    assert reach == {"duration": 24.0, "hold": 2.0, "start_offsets_deg": [[0.0, 0.0]]}
    assert ellipse == load(PERIODIC / "coarse_search.toml")["evaluation"]
    assert ellipse["duration"] == 40.0
    assert (ellipse["curve_tolerance"], ellipse["cycle_periods"], ellipse["cycle_turns"]) == (0.02, 5, [4.0, 6.0])
    assert len(ellipse["start_offsets_deg"]) + len(ellipse["extra_starts"]["on_curve"]) == 8


def test_the_coarse_searchs_header_runs_it_with_sweep_robot_esn_and_a_number_of_workers():
    words = header_command(EXPERIMENT / "coarse_search.toml")

    assert words[:-1] == [
        "uv",
        "run",
        "python",
        "experiments/sweep_robot_esn.py",
        "experiments/manual_demonstration_v2_reach_and_ellipse/coarse_search.toml",
        "--workers",
    ]
    assert int(words[-1]) > 0


# The coarse search's best setting (run 20261119-190537-coarse_search): the fine search's center.
CENTER = {"leak_rate": 0.1, "spectral_radius": 1.5, "input_scaling": 0.5, "ridge": 1.0}


def test_the_fine_search_is_the_coarse_search_but_for_its_values_and_seeds():
    fine, coarse = load(EXPERIMENT / "fine_search.toml"), load(EXPERIMENT / "coarse_search.toml")

    fine_sweep, coarse_sweep = fine.pop("sweep"), coarse.pop("sweep")
    assert fine_sweep.pop("seed") == [1, 2, 3, 4, 5]
    assert list(fine_sweep) == list(coarse_sweep)
    fine_esn, coarse_esn = fine.pop("esn"), coarse.pop("esn")
    assert fine_esn == coarse_esn | CENTER | {"seed": 1}  # the center's values, the first seed, and the same context
    assert fine == coarse  # the same takes and contexts, tracker, runs, and score


def test_the_fine_search_takes_the_center_and_the_values_halfway_to_its_coarse_neighbors():
    fine, coarse = load(EXPERIMENT / "fine_search.toml"), load(EXPERIMENT / "coarse_search.toml")

    for name, center in CENTER.items():
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
