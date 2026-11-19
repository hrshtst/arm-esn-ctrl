# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the summaries of ESN sweeps over seeds, and for the settings' scores."""

import math

import numpy as np
import pytest

from arm_esn_ctrl.sweeps import (
    competition_ranks,
    score_cycles,
    score_settings,
    score_tasks,
    summarize_seeds,
    sweep_combinations,
)


def row(scaling, radius, seed, rmse, *, failures=0, arrival=10.0, final=0.001):
    return {
        "input_scaling": scaling,
        "spectral_radius": radius,
        "seed": seed,
        "failures": failures,
        "worst_path_rmse_m": rmse,
        "latest_arrival_s": arrival,
        "worst_final_distance_m": final,
    }


def test_each_setting_is_summarized_over_its_seeds_in_the_order_of_the_sweep():
    rows = [
        row(2.0, 0.99, 0, 0.001),
        row(2.0, 0.99, 1, 0.003, final=0.004),
        row(2.0, 0.99, 2, 0.002),
        row(1.5, 1.1, 0, 0.010, failures=1, arrival=math.nan),
        row(1.5, 1.1, 1, 0.020, failures=2, arrival=12.0),
        row(1.5, 1.1, 2, 0.030, arrival=11.0, final=0.05),
    ]

    first, second = summarize_seeds(rows, ["input_scaling", "spectral_radius"])

    assert (first["input_scaling"], first["spectral_radius"]) == (2.0, 0.99)
    assert (second["input_scaling"], second["spectral_radius"]) == (1.5, 1.1)
    assert "seed" not in first
    assert first["seeds"] == second["seeds"] == 3
    assert (first["failing_seeds"], second["failing_seeds"]) == (0, 2)
    assert first["median_worst_path_rmse_m"] == pytest.approx(0.002)
    assert first["mean_worst_path_rmse_m"] == pytest.approx(0.002)
    assert first["sd_worst_path_rmse_m"] == pytest.approx(np.std([0.001, 0.003, 0.002], ddof=1))
    assert (first["min_worst_path_rmse_m"], first["max_worst_path_rmse_m"]) == (0.001, 0.003)
    assert first["worst_final_distance_m"] == 0.004
    assert second["median_latest_arrival_s"] == pytest.approx(11.5)  # the seed that never arrives is left out
    assert second["worst_final_distance_m"] == 0.05


def test_a_setting_whose_seeds_never_arrive_has_no_arrival():
    rows = [row(1.0, 1.5, seed, 0.5, failures=2, arrival=math.nan) for seed in range(3)]

    (summary,) = summarize_seeds(rows, ["input_scaling", "spectral_radius"])

    assert math.isnan(summary["median_latest_arrival_s"])
    assert summary["failing_seeds"] == 3


def test_a_single_seed_has_no_spread():
    (summary,) = summarize_seeds([row(2.0, 0.99, 0, 0.001)], ["input_scaling", "spectral_radius"])

    assert summary["sd_worst_path_rmse_m"] == 0.0


def test_a_sweep_with_disturbances_also_summarizes_the_returns_to_the_path():
    rows = [
        row(2.0, 0.99, 0, 0.001) | {"unreturned": 0, "latest_return_s": 3.0, "worst_off_path_m": 0.04},
        row(2.0, 0.99, 1, 0.001) | {"unreturned": 1, "latest_return_s": math.nan, "worst_off_path_m": 0.20},
        row(2.0, 0.99, 2, 0.001) | {"unreturned": 0, "latest_return_s": 2.0, "worst_off_path_m": 0.05},
    ]

    (summary,) = summarize_seeds(rows, ["input_scaling", "spectral_radius"])

    assert summary["unreturned_seeds"] == 1
    assert summary["median_latest_return_s"] == pytest.approx(2.5)  # the seed whose run is not back is left out
    assert summary["median_worst_off_path_m"] == pytest.approx(0.05)


def test_an_undisturbed_sweep_has_no_returns():
    (summary,) = summarize_seeds([row(2.0, 0.99, 0, 0.001)], ["input_scaling", "spectral_radius"])

    assert "unreturned_seeds" not in summary


def run(leak, seed, ratio, *, failed=False):
    return {"leak_rate": leak, "seed": seed, "failed": failed, "detour_ratio": ratio}


def test_settings_are_scored_by_their_holding_runs_then_by_their_median_detour_ratio():
    runs = [
        run(0.01, 0, 1.30),
        run(0.01, 0, 1.10),
        run(0.03, 0, 1.05),
        run(0.03, 0, math.nan, failed=True),  # never arrives
        run(0.1, 0, 1.02),
        run(0.1, 0, 1.08, failed=True),  # arrives, and leaves the goal: its detour ratio counts
        run(0.3, 0, 1.04),
        run(0.3, 0, 1.06),
    ]

    scores = score_settings(runs, ["leak_rate"])

    assert [score["leak_rate"] for score in scores] == [0.3, 0.01, 0.03, 0.1]
    assert [score["holding_runs"] for score in scores] == [2, 2, 1, 1]
    assert [score["runs"] for score in scores] == [2] * 4
    assert [score["median_detour_ratio"] for score in scores] == pytest.approx([1.05, 1.20, 1.05, 1.05])


def test_a_setting_whose_runs_never_arrive_comes_last_and_equal_settings_keep_the_sweeps_order():
    runs = [
        run(0.01, 0, math.nan, failed=True),
        run(0.03, 0, 1.5, failed=True),
        run(0.1, 0, 1.5, failed=True),
    ]

    scores = score_settings(runs, ["leak_rate"])

    assert [score["leak_rate"] for score in scores] == [0.03, 0.1, 0.01]
    assert math.isnan(scores[-1]["median_detour_ratio"])


def test_a_settings_runs_are_those_of_all_its_seeds_and_its_typical_seed_is_nearest_its_median():
    runs = [
        *(run(0.03, 1, ratio) for ratio in (1.00, 1.02, 1.04)),
        *(run(0.03, 2, ratio) for ratio in (1.10, 1.12, 1.14)),
        *(run(0.03, 3, ratio) for ratio in (1.30, 1.32, 1.34)),
        # The seed nearest the median does not count: one of its runs does not hold.
        run(0.03, 4, 1.11),
        run(0.03, 4, 1.11),
        run(0.03, 4, 1.11, failed=True),
    ]

    (score,) = score_settings(runs, ["leak_rate"])

    assert (score["seeds"], score["runs"], score["holding_runs"]) == (4, 12, 11)
    assert score["median_detour_ratio"] == pytest.approx(1.11)
    assert score["typical_seed"] == 2


def test_without_seeds_a_setting_has_no_typical_seed():
    runs = [{"leak_rate": 0.03, "failed": False, "detour_ratio": 1.1}]

    (score,) = score_settings(runs, ["leak_rate"])

    assert "typical_seed" not in score
    assert "seeds" not in score


def test_a_settings_runs_are_those_of_all_its_tasks_and_each_tasks_holding_runs_are_given():
    runs = [
        run(0.01, 0, 1.10) | {"task": "near"},
        run(0.01, 0, 1.20) | {"task": "near"},
        run(0.01, 0, 1.30) | {"task": "far"},
        run(0.01, 0, 1.40) | {"task": "far"},
        # Fails one target in both runs: fewer holding runs, whatever its detour ratio.
        run(0.03, 0, 1.01) | {"task": "near"},
        run(0.03, 0, 1.02) | {"task": "near"},
        run(0.03, 0, math.nan, failed=True) | {"task": "far"},
        run(0.03, 0, math.nan, failed=True) | {"task": "far"},
    ]

    scores = score_settings(runs, ["leak_rate"])

    assert [score["leak_rate"] for score in scores] == [0.01, 0.03]
    assert [(score["runs"], score["holding_runs"]) for score in scores] == [(4, 4), (4, 2)]
    assert [(score["holding_runs_near"], score["holding_runs_far"]) for score in scores] == [(2, 2), (2, 0)]
    assert scores[0]["median_detour_ratio"] == pytest.approx(1.25)
    assert list(scores[0]) == [
        "leak_rate",
        "runs",
        "holding_runs",
        "median_detour_ratio",
        "holding_runs_near",
        "holding_runs_far",
        "seeds",
        "typical_seed",
    ]


def test_a_setting_of_one_task_has_no_holding_runs_of_each_task():
    (score,) = score_settings([run(0.03, 0, 1.1), run(0.03, 1, 1.2)], ["leak_rate"])

    assert list(score) == ["leak_rate", "runs", "holding_runs", "median_detour_ratio", "seeds", "typical_seed"]


def test_a_sweep_tries_every_combination_of_its_values_the_last_varying_fastest():
    combinations = sweep_combinations({"leak_rate": [0.01, 0.03], "seed": [1, 2]})

    assert combinations == [
        {"leak_rate": 0.01, "seed": 1},
        {"leak_rate": 0.01, "seed": 2},
        {"leak_rate": 0.03, "seed": 1},
        {"leak_rate": 0.03, "seed": 2},
    ]


def test_listed_settings_are_each_tried_with_every_combination_of_the_sweep():
    settings = [{"leak_rate": 0.1, "ridge": 1.0}, {"leak_rate": 0.055, "ridge": 0.1}]

    combinations = sweep_combinations({"seed": [1, 2]}, settings)

    assert combinations == [
        {"leak_rate": 0.1, "ridge": 1.0, "seed": 1},
        {"leak_rate": 0.1, "ridge": 1.0, "seed": 2},
        {"leak_rate": 0.055, "ridge": 0.1, "seed": 1},
        {"leak_rate": 0.055, "ridge": 0.1, "seed": 2},
    ]
    assert sweep_combinations({}, settings) == settings


def test_listed_settings_must_give_the_same_hyperparameters_and_none_that_is_swept():
    with pytest.raises(ValueError, match="same hyperparameters"):
        sweep_combinations({"seed": [1]}, [{"leak_rate": 0.1}, {"ridge": 1.0}])
    with pytest.raises(ValueError, match="both give"):
        sweep_combinations({"leak_rate": [0.1]}, [{"leak_rate": 0.1}])


def loop_run(leak, seed, rms, *, cycles=True):
    return {"leak_rate": leak, "seed": seed, "cycles": cycles, "curve_rms_m": rms}


def test_settings_are_scored_by_their_runs_that_cycle_then_by_the_median_curve_rms_of_those_runs():
    runs = [
        loop_run(0.01, 0, 0.004),
        loop_run(0.01, 0, 0.006),
        loop_run(0.03, 0, 0.002),
        loop_run(0.03, 0, 0.001, cycles=False),  # close to the curve, at the wrong pace: its curve RMS does not count
        loop_run(0.1, 0, 0.003),
        loop_run(0.1, 0, 0.300, cycles=False),
        loop_run(0.3, 0, 0.003),
        loop_run(0.3, 0, 0.004),
    ]

    scores = score_cycles(runs, ["leak_rate"])

    assert [score["leak_rate"] for score in scores] == [0.3, 0.01, 0.03, 0.1]
    assert [score["cycling_runs"] for score in scores] == [2, 2, 1, 1]
    assert [score["runs"] for score in scores] == [2] * 4
    assert [score["median_curve_rms_m"] for score in scores] == pytest.approx([0.0035, 0.005, 0.002, 0.003])


def test_a_setting_whose_runs_never_cycle_comes_last_and_equal_settings_keep_the_sweeps_order():
    runs = [
        loop_run(0.01, 0, 0.001, cycles=False),
        loop_run(0.03, 0, 0.005),
        loop_run(0.1, 0, 0.005),
    ]

    scores = score_cycles(runs, ["leak_rate"])

    assert [score["leak_rate"] for score in scores] == [0.03, 0.1, 0.01]
    assert math.isnan(scores[-1]["median_curve_rms_m"])


def test_a_settings_typical_seed_cycles_the_most_and_its_median_curve_rms_is_nearest_the_settings():
    runs = [
        *(loop_run(0.03, 1, rms) for rms in (0.001, 0.002, 0.003)),
        *(loop_run(0.03, 2, rms) for rms in (0.004, 0.005, 0.006)),
        *(loop_run(0.03, 3, rms) for rms in (0.010, 0.011, 0.012)),
        # The seed nearest the median does not count: one of its runs does not cycle.
        loop_run(0.03, 4, 0.0045),
        loop_run(0.03, 4, 0.0045),
        loop_run(0.03, 4, 0.0045, cycles=False),
    ]

    (score,) = score_cycles(runs, ["leak_rate"])

    assert (score["seeds"], score["runs"], score["cycling_runs"]) == (4, 12, 11)
    assert score["median_curve_rms_m"] == pytest.approx(0.0045)
    assert score["typical_seed"] == 2


def reach_run(leak, seed, ratio, *, failed=False):
    """A run of the reach: no cycles, its detour ratio NaN if it never arrives."""
    return {"leak_rate": leak, "seed": seed, "task": "reach", "failed": failed, "detour_ratio": ratio}


def ellipse_run(leak, seed, rms, *, cycles=True):
    """A run of the periodic task, which passes if it cycles."""
    return {
        "leak_rate": leak,
        "seed": seed,
        "task": "ellipse",
        "failed": not cycles,
        "detour_ratio": math.nan,
        "cycles": cycles,
        "curve_rms_m": rms,
    }


def test_equal_values_share_the_best_of_their_places_and_every_nan_shares_the_last():
    assert competition_ranks([1.1, 1.0, 1.1, math.nan, 1.2, math.nan]) == [2, 1, 2, 5, 4, 5]
    assert competition_ranks([math.nan, math.nan]) == [1, 1]


def test_settings_are_scored_by_their_runs_that_pass_then_by_the_worse_of_two_ranks_then_by_the_better():
    runs = [
        # The best median detour ratio and the third curve RMS: ranks 1 and 3.
        reach_run(0.01, 0, 1.05),
        ellipse_run(0.01, 0, 0.006),
        # The second of both: ranks 2 and 2, so the worse rank, 2, puts it first.
        reach_run(0.03, 0, 1.10),
        ellipse_run(0.03, 0, 0.005),
        # The third detour ratio and the best curve RMS: ranks 3 and 1, as the first setting, which comes before.
        reach_run(0.1, 0, 1.20),
        ellipse_run(0.1, 0, 0.004),
        # One run that does not pass: after every setting whose runs all pass, whatever its measures.
        reach_run(0.3, 0, 1.01),
        ellipse_run(0.3, 0, 0.001, cycles=False),
    ]

    scores = score_tasks(runs, ["leak_rate"])

    assert [score["leak_rate"] for score in scores] == [0.03, 0.01, 0.1, 0.3]
    assert [score["passing_runs"] for score in scores] == [2, 2, 2, 1]
    assert [(score["detour_ratio_rank"], score["curve_rms_rank"]) for score in scores] == [
        (2, 2),
        (1, 3),
        (3, 1),
        (1, 1),  # ranked among the settings with one run that passes: itself alone
    ]
    assert [score["median_detour_ratio"] for score in scores] == pytest.approx([1.10, 1.05, 1.20, 1.01])
    # The run that does not cycle does not count, close to the curve as it is.
    assert math.isnan(scores[-1]["median_curve_rms_m"])


def test_settings_with_equal_medians_share_a_rank_and_are_then_ordered_by_the_other():
    runs = [
        reach_run(0.01, 0, 1.10),
        ellipse_run(0.01, 0, 0.006),
        reach_run(0.03, 0, 1.10),
        ellipse_run(0.03, 0, 0.004),
        reach_run(0.1, 0, 1.30),
        ellipse_run(0.1, 0, 0.005),
    ]

    scores = score_tasks(runs, ["leak_rate"])

    # The first two share the first rank of the detour ratio; the third is third, not second.
    assert [(score["leak_rate"], score["detour_ratio_rank"], score["curve_rms_rank"]) for score in scores] == [
        (0.03, 1, 1),
        (0.01, 1, 3),
        (0.1, 3, 2),  # worse rank 3, as the second's, and better rank 2, worse than its 1
    ]


def test_a_setting_without_a_reach_run_that_arrives_or_an_ellipse_run_that_cycles_is_last_in_that_rank():
    runs = [
        # No reach run arrives: no detour ratio, last in its rank, however well the ellipse's runs cycle.
        reach_run(0.01, 0, math.nan, failed=True),
        reach_run(0.01, 0, math.nan, failed=True),
        ellipse_run(0.01, 0, 0.002),
        ellipse_run(0.01, 0, 0.002),
        # One reach run arrives and leaves the goal: its detour ratio counts, and it does not pass.
        reach_run(0.03, 0, 1.2, failed=True),
        reach_run(0.03, 0, 1.3),
        ellipse_run(0.03, 0, 0.003),
        ellipse_run(0.03, 0, 0.001, cycles=False),  # close to the curve, at the wrong pace: it does not count
        # No ellipse run cycles: no curve RMS, last in its rank, however directly the reaches arrive.
        reach_run(0.1, 0, 1.1),
        reach_run(0.1, 0, 1.0),
        ellipse_run(0.1, 0, 0.001, cycles=False),
        ellipse_run(0.1, 0, 0.001, cycles=False),
    ]

    scores = score_tasks(runs, ["leak_rate"])

    assert [score["passing_runs"] for score in scores] == [2, 2, 2]
    # The second setting is second in both ranks; the others are first in one and last in the other, in the
    # sweep's order.
    assert [score["leak_rate"] for score in scores] == [0.03, 0.01, 0.1]
    assert [(score["detour_ratio_rank"], score["curve_rms_rank"]) for score in scores] == [(2, 2), (3, 1), (1, 3)]
    assert scores[0]["median_detour_ratio"] == pytest.approx(1.25)
    assert scores[0]["median_curve_rms_m"] == pytest.approx(0.003)
    assert math.isnan(scores[1]["median_detour_ratio"])
    assert math.isnan(scores[2]["median_curve_rms_m"])
    assert [(score["passing_runs_reach"], score["passing_runs_ellipse"]) for score in scores] == [
        (1, 1),
        (0, 2),
        (2, 0),
    ]


def test_a_settings_typical_seed_passes_the_most_and_its_median_curve_rms_is_nearest_the_settings():
    runs = [
        *(ellipse_run(0.03, 1, rms) for rms in (0.001, 0.002, 0.003)),
        reach_run(0.03, 1, 1.0),
        *(ellipse_run(0.03, 2, rms) for rms in (0.004, 0.005, 0.006)),
        reach_run(0.03, 2, 1.5),  # its detour ratio does not choose the typical seed
        *(ellipse_run(0.03, 3, rms) for rms in (0.010, 0.011, 0.012)),
        reach_run(0.03, 3, 1.1),
        # The seed nearest the median does not count: its reach does not hold.
        *(ellipse_run(0.03, 4, 0.0045) for _ in range(3)),
        reach_run(0.03, 4, 1.1, failed=True),
    ]

    (score,) = score_tasks(runs, ["leak_rate"])

    assert (score["seeds"], score["runs"], score["passing_runs"]) == (4, 16, 15)
    assert score["median_curve_rms_m"] == pytest.approx(0.0045)
    assert score["typical_seed"] == 2
    assert list(score)[-2:] == ["seeds", "typical_seed"]
