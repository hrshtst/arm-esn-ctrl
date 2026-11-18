# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the summaries of ESN sweeps over seeds, and for the settings' scores."""

import math

import numpy as np
import pytest

from arm_esn_ctrl.sweeps import score_cycles, score_settings, summarize_seeds, sweep_combinations


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
