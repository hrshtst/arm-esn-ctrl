# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the score of a search's settings over their undisturbed and their pushed runs."""

import math
import sys

import pytest

from arm_esn_ctrl.storage import REPO_ROOT

# The runners import each other as modules of their own directory, as they run.
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from score_sweeps import read_runs, score_runs

RADIUS = 0.02


def undisturbed(leak_rate, seed, ratio, *, failed=False):
    return {"leak_rate": leak_rate, "seed": seed, "detour_ratio": ratio, "failed": failed}


def pushed(leak_rate, seed, final_distance, *, arrived=True, failed=False, ratio=5.0):
    """A pushed run; ``failed`` is the sweep's, which also fails a run that leaves the goal during the hold."""
    return {
        "leak_rate": leak_rate,
        "seed": seed,
        "detour_ratio": ratio,
        "failed": failed,
        "arrived": arrived,
        "final_distance_m": final_distance,
    }


def test_a_setting_is_scored_by_its_passing_runs_undisturbed_and_pushed_then_by_its_undisturbed_detour():
    scores = score_runs(
        [
            undisturbed(0.03, 1, 1.20),
            undisturbed(0.1, 1, 1.01),
            undisturbed(0.3, 1, 1.10),
        ],
        [
            pushed(0.03, 1, 0.005),
            pushed(0.1, 1, 0.3),  # ends away from the target
            pushed(0.3, 1, 0.005),
        ],
        ["leak_rate", "seed"],
        RADIUS,
    )

    assert [score["leak_rate"] for score in scores] == [0.3, 0.03, 0.1]
    assert [score["passing_runs"] for score in scores] == [2, 2, 1]
    assert [(score["holding_runs"], score["passing_pushed_runs"]) for score in scores] == [(1, 1), (1, 1), (1, 0)]
    assert [(score["runs"], score["undisturbed_runs"], score["pushed_runs"]) for score in scores] == [(2, 1, 1)] * 3


def test_a_pushed_run_passes_if_it_arrives_and_ends_within_the_goal_whatever_it_does_during_the_hold():
    (score,) = score_runs(
        [undisturbed(0.03, 1, 1.1)],
        [
            pushed(0.03, 1, 0.005, failed=True),  # left the goal during the hold: excused
            pushed(0.03, 1, 0.02),  # ends on the goal's edge
            pushed(0.03, 1, 0.021),
            pushed(0.03, 1, 0.005, arrived=False),
        ],
        ["leak_rate", "seed"],
        RADIUS,
    )

    assert (score["pushed_runs"], score["passing_pushed_runs"]) == (4, 2)


def test_the_detour_ratio_is_the_undisturbed_runs_and_the_typical_seed_is_among_those_passing_most():
    scores = score_runs(
        [
            undisturbed(0.03, 1, 1.10),
            undisturbed(0.03, 2, 1.30),
            undisturbed(0.03, 3, 1.20),
        ],
        [
            pushed(0.03, 1, 0.005, ratio=9.0),
            pushed(0.03, 2, 0.005, ratio=9.0),
            pushed(0.03, 3, 0.4, ratio=9.0),  # the seed nearest the median fails the pushes
        ],
        ["leak_rate", "seed"],
        RADIUS,
    )

    (score,) = scores
    assert score["median_detour_ratio"] == pytest.approx(1.20)
    assert score["seeds"] == 3
    assert score["typical_seed"] in (1, 2)
    assert score["passing_runs"] == 5


def test_the_pushed_runs_other_esns_are_left_out_and_a_missing_one_is_an_error():
    rows = [undisturbed(0.03, 1, 1.1)]

    (score,) = score_runs(rows, [pushed(0.03, 1, 0.005), pushed(0.1, 1, 0.005)], ["leak_rate", "seed"], RADIUS)
    assert score["pushed_runs"] == 1

    with pytest.raises(ValueError, match="not in the pushed run"):
        score_runs(rows, [pushed(0.03, 2, 0.005)], ["leak_rate", "seed"], RADIUS)


def test_runs_are_read_with_booleans_whole_seeds_and_numbers(tmp_path):
    path = tmp_path / "runs.csv"
    path.write_text("leak_rate,seed,origin,arrived,detour_ratio\n0.03,4,demo_00_filtered,True,nan\n")

    (row,) = read_runs(path)

    assert row["leak_rate"] == 0.03
    assert row["seed"] == 4
    assert isinstance(row["seed"], int)
    assert row["origin"] == "demo_00_filtered"
    assert row["arrived"] is True
    assert math.isnan(row["detour_ratio"])
