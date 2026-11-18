# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""The targets experiment's configurations agree with each other, with the takes, and with the many-starts ones."""

import itertools
import tomllib

import pytest
from skelarm import Task, multi_target_specs

from arm_esn_ctrl.autonomous import task_names
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.storage import REPO_ROOT
from arm_esn_ctrl.sweeps import sweep_combinations

TARGETS = REPO_ROOT / "experiments/demonstrations/reach_manual_v2_targets.toml"
EXPERIMENT = REPO_ROOT / "experiments/manual_demonstration_v2_targets"
MANY_STARTS = REPO_ROOT / "experiments/manual_demonstration_v2_many_starts"  # stage 1's
CONTEXT = ("context_scaling", "context_times_input_scaling")  # the [esn] keys of the context, which stage 1 has not
TAKES = [f"demo_{i:02d}_filtered.sklog.npz" for i in range(9)]  # each target's nine takes, filtered


def load(path):
    with path.open("rb") as f:
        return tomllib.load(f)


def without(table, keys):
    """``table`` without ``keys``."""
    return {k: v for k, v in table.items() if k not in keys}


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


def test_the_coarse_search_parses_as_a_sweep_of_108_settings_of_one_esn_of_three_tasks():
    config = load(EXPERIMENT / "coarse_search.toml")

    assert task_names(config["demonstrations"]) == ("A", "B", "C")
    esn = EsnConfig(**config["esn"])
    assert (esn.n_neurons, esn.seed) == (800, 0)
    combinations = sweep_combinations(config["sweep"], config.get("settings"))
    assert len(combinations) == 108
    for combination in combinations:
        EsnConfig(**config["esn"] | combination)


def test_the_coarse_search_trains_on_the_nine_filtered_takes_of_each_target_a_b_and_c_in_that_order():
    tasks = load(EXPERIMENT / "coarse_search.toml")["demonstrations"]
    labels = [label for _, label, _, _ in multi_target_specs(Task.from_dict(load(TARGETS)["task"]))]

    assert [table["context"] for table in tasks] == labels == ["A", "B", "C"]
    for table in tasks:
        assert list(table) == ["context", "run", "train"]
        # The demonstration run the import writes for the target its context names.
        assert table["run"].startswith("results/demonstrations/")
        assert table["run"].endswith(f"-reach_manual_v2_targets_{table['context'].lower()}")
        assert table["train"] == TAKES


@pytest.mark.parametrize("table", ["sweep", "ranking", "tracker", "evaluation"])
def test_the_coarse_search_sweeps_scores_tracks_and_runs_as_stage_1s_coarse_search(table):
    targets, many_starts = load(EXPERIMENT / "coarse_search.toml"), load(MANY_STARTS / "coarse_search.toml")

    assert targets[table] == many_starts[table]


def test_the_coarse_searchs_esn_is_stage_1s_coarse_searchs_with_a_context():
    targets, many_starts = load(EXPERIMENT / "coarse_search.toml"), load(MANY_STARTS / "coarse_search.toml")

    assert not set(CONTEXT) & set(many_starts["esn"])
    assert without(targets["esn"], CONTEXT) == many_starts["esn"]
    assert "disturbance" not in targets


def test_the_context_is_fed_in_divided_by_the_input_scaling_with_a_scale_of_1_and_is_not_swept():
    config = load(EXPERIMENT / "coarse_search.toml")

    assert (config["esn"]["context_scaling"], config["esn"]["context_times_input_scaling"]) == (1.0, False)
    assert not set(CONTEXT) & set(config["sweep"])
    # Whatever the input scaling, the context's input weights, times it, have the scale 1.
    for input_scaling in config["sweep"]["input_scaling"]:
        esn = ReachingEsn(EsnConfig(**config["esn"] | {"input_scaling": input_scaling}))
        assert esn.context_gain * input_scaling == pytest.approx(1.0)


CENTER = {"leak_rate": 0.1, "spectral_radius": 1.2, "input_scaling": 2.0, "ridge": 1.0}  # the coarse search's best


def two_digits(x):
    """``x`` rounded to two significant digits."""
    return float(f"{x:.2g}")


def test_the_fine_search_parses_as_a_sweep_of_81_settings_of_five_seeds_1_to_5_of_three_tasks():
    config = load(EXPERIMENT / "fine_search.toml")

    assert task_names(config["demonstrations"]) == ("A", "B", "C")
    sweep = dict(config["sweep"])
    assert sweep.pop("seed") == [1, 2, 3, 4, 5]
    assert len(sweep_combinations(sweep)) == 81
    combinations = sweep_combinations(config["sweep"], config.get("settings"))
    assert len(combinations) == 81 * 5
    for combination in combinations:
        EsnConfig(**config["esn"] | combination)


def test_the_fine_searchs_middle_values_are_the_coarse_searchs_best_setting():
    coarse, fine = load(EXPERIMENT / "coarse_search.toml"), load(EXPERIMENT / "fine_search.toml")

    assert list(fine["sweep"]) == [*CENTER, "seed"]
    for name, center in CENTER.items():
        assert fine["sweep"][name][1] == fine["esn"][name] == center
        assert center in coarse["sweep"][name]


def test_the_fine_searchs_outer_values_are_halfway_to_the_coarse_neighbors_and_as_far_beyond_an_edge():
    coarse, fine = load(EXPERIMENT / "coarse_search.toml"), load(EXPERIMENT / "fine_search.toml")

    # The center's input scaling and ridge are the coarse grid's largest.
    assert [name for name, center in CENTER.items() if center == coarse["sweep"][name][-1]] == [
        "input_scaling",
        "ridge",
    ]
    for name, center in CENTER.items():
        grid = coarse["sweep"][name]
        below, _, above = fine["sweep"][name]
        k = grid.index(center)
        # Beyond the grid's edge, as far from the center as the coarse neighbor on the other side.
        if name == "spectral_radius":  # halfway in difference
            lower = grid[k - 1] if k > 0 else 2 * center - grid[k + 1]
            upper = grid[k + 1] if k + 1 < len(grid) else 2 * center - grid[k - 1]
            assert (below, above) == pytest.approx(((lower + center) / 2, (center + upper) / 2))
        else:  # halfway in ratio, rounded to two digits
            lower = grid[k - 1] if k > 0 else center**2 / grid[k + 1]
            upper = grid[k + 1] if k + 1 < len(grid) else center**2 / grid[k - 1]
            assert (below, above) == (two_digits((lower * center) ** 0.5), two_digits((center * upper) ** 0.5))


@pytest.mark.parametrize("table", ["demonstrations", "ranking", "tracker", "evaluation"])
def test_the_fine_search_trains_on_the_same_takes_and_scores_tracks_and_runs_as_the_coarse_search(table):
    coarse, fine = load(EXPERIMENT / "coarse_search.toml"), load(EXPERIMENT / "fine_search.toml")

    assert fine[table] == coarse[table]


def test_the_fine_searchs_esn_is_the_coarse_searchs_with_its_context_but_for_the_swept_values():
    coarse, fine = load(EXPERIMENT / "coarse_search.toml"), load(EXPERIMENT / "fine_search.toml")

    swept = {*fine["sweep"]}
    assert without(fine["esn"], swept) == without(coarse["esn"], swept)
    assert set(CONTEXT) <= set(fine["esn"])
    assert list(fine) == list(coarse)  # the same tables, and no disturbance


@pytest.mark.parametrize("name", ["coarse_search.toml", "fine_search.toml"])
def test_the_searchs_header_runs_it_with_15_workers_as_the_machine_is_shared(name):
    assert header_command(EXPERIMENT / name) == [
        "uv",
        "run",
        "python",
        "experiments/sweep_robot_esn.py",
        f"experiments/manual_demonstration_v2_targets/{name}",
        "--workers",
        "15",
    ]
