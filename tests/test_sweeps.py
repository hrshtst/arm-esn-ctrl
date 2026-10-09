# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the summaries of ESN sweeps over seeds."""

import math

import numpy as np
import pytest

from arm_esn_ctrl.sweeps import summarize_seeds


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
