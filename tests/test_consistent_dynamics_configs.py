# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Part 3's configurations run reports 015's and 016's analyses on the consistent search's ESNs (see its README.md)."""

import tomllib
from pathlib import Path

import pytest

from arm_esn_ctrl.storage import REPO_ROOT

SEARCH = REPO_ROOT / "experiments/manual_demonstration_v2_consistent_search"
DYNAMICS = REPO_ROOT / "experiments/manual_demonstration_v2_consistent_dynamics"
REPORT_015 = REPO_ROOT / "experiments/manual_demonstration_v2_esn_dynamics"
REPORT_016 = REPO_ROOT / "experiments/manual_demonstration_v2_esn_screening"

# The chosen ESNs: their tag in the search, and in report 015's configurations.
CHOSEN = {
    "esn1_lr0.03_sr1.2_is1.5_seed8": "lr0.03_sr1.2_is1.5_seed8",
    "esn2_lr0.03_sr1.2_is1_seed2": "lr0.03_sr1.2_is1_seed2",
}


def load(directory: Path, name: str) -> dict:
    with (directory / f"{name}.toml").open("rb") as f:
        return tomllib.load(f)


def without(table: dict, *keys: str) -> dict:
    return {key: value for key, value in table.items() if key not in keys}


def models() -> list[str]:
    """The chosen ESNs as the search trained them, ESN #1's first, as its robot runs read them."""
    return [load(SEARCH, f"nominal_robot_{tag}")["esn"]["model"] for tag in CHOSEN]


@pytest.mark.parametrize("name", ["global_starts", "push_pulses"])
def test_the_chosen_esns_are_the_searchs(name):
    esns = load(DYNAMICS, name)["esn"]

    assert [e["model"] for e in esns] == models()
    assert [(e["key"], e["label"]) for e in esns] == [("esn1", "ESN #1"), ("esn2", "ESN #2")]


def test_from_rest_anywhere_is_report_015s_run_for_60_s():
    config, earlier = load(DYNAMICS, "global_starts"), load(REPORT_015, "global_starts")

    assert config["evaluation"] == earlier["evaluation"] | {"duration": 60.0}
    assert without(config, "esn", "evaluation") == without(earlier, "esn", "evaluation")


def test_the_pulses_are_report_015s_with_the_searchs_tracker_in_runs_as_long_as_its():
    config, earlier = load(DYNAMICS, "push_pulses"), load(REPORT_015, "push_pulses")
    search = load(SEARCH, f"pushes_robot_{next(iter(CHOSEN))}")

    assert config["tracker"] == search["tracker"]
    assert config["progress"] == search["progress"]
    assert config["evaluation"] == earlier["evaluation"] | {"duration": search["evaluation"]["duration"]}
    assert without(config, "esn", "evaluation") == without(earlier, "esn", "evaluation")


@pytest.mark.parametrize(("tag", "earlier_tag"), CHOSEN.items())
def test_the_warmups_are_report_015s_in_runs_as_long_as_the_searchs(tag, earlier_tag):
    config, earlier = load(DYNAMICS, f"warmup_{tag}"), load(REPORT_015, f"warmup_{earlier_tag}")

    assert config["esn"]["model"] == models()[list(CHOSEN).index(tag)]
    assert config["evaluation"] == earlier["evaluation"] | {"duration": 24.0}
    assert without(config, "esn", "evaluation") == without(earlier, "esn", "evaluation")


SWEEP = {
    "leak_rate": [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5],
    "spectral_radius": [0.7, 0.9, 0.99, 1.2, 1.35, 1.5, 1.75],
    "input_scaling": [0.1, 0.2, 0.5, 1.0, 1.5, 2.0],
}


@pytest.mark.parametrize(("name", "seeds"), [("screen", range(10)), ("screen_seeds10to19", range(10, 20))])
def test_the_screens_are_report_016s_over_the_searchs_esns(name, seeds):
    config, earlier = load(DYNAMICS, name), load(REPORT_016, "screen")
    grid = load(SEARCH, "sweep_lr0.03_nominal")

    assert config["demonstrations"] == grid["demonstrations"]
    assert without(config["esn"], *SWEEP, "seed") == without(grid["esn"], *SWEEP, "seed")
    assert config["sweep"] == SWEEP | {"seed": list(seeds)}
    assert config["screen"] == earlier["screen"]


def test_where_they_rest_is_report_015s_from_step_1s_run():
    config, earlier = load(DYNAMICS, "fixed_points"), load(REPORT_015, "fixed_points")

    assert [e["model"] for e in config["esn"]] == models()
    assert config["global_starts"]["run"].endswith("-global_starts")
    assert config["global_starts"]["run"].startswith("results/manual_demonstration_v2_consistent_dynamics/")
    assert without(config, "esn", "global_starts") == without(earlier, "esn", "global_starts")


def test_what_the_screen_predicts_is_scored_against_the_searchs_sweeps_with_report_016s_rules():
    config, earlier = load(DYNAMICS, "against_robot"), load(REPORT_016, "against_robot")
    runs = load(SEARCH, "select")["runs"]

    assert config["runs"]["screen"].startswith("results/manual_demonstration_v2_consistent_dynamics/")
    assert config["runs"]["screen"].endswith("-screen")
    assert sorted(config["runs"]["robot"]) == sorted(runs["nominal"] + runs["block"] + runs["pushes"] + runs["offsets"])
    assert config["rules"] == earlier["rules"]
    # The search's 11 qualifying settings, ESN #1's and ESN #2's first.
    assert len(config["qualifying"]["settings"]) == 11
    assert config["qualifying"]["settings"][:2] == [[0.03, 1.2, 1.5], [0.03, 1.2, 1.0]]
    assert config["named"] == earlier["named"]
