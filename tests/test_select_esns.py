# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the rule that chooses a robot search's ESNs."""

import sys

import numpy as np
import pytest

from arm_esn_ctrl.storage import REPO_ROOT

# The runners import each other as modules of their own directory, as they run.
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from select_esns import disturbance_windows, esn_key, esn_measures, failure, rank_settings, typical_seed

RADIUS = 0.02
BLOCK = {"type": "block", "onset": 4.5, "release": 5.5}
PUSHES = [{"type": "push", "onset": t, "duration": 0.5} for t in (3.0, 7.0, 15.0)]
WINDOWS = [
    *disturbance_windows({"disturbance": BLOCK, "evaluation": {"duration": 24.0}}, "block"),
    *disturbance_windows({"disturbance": PUSHES, "evaluation": {"duration": 24.0}}, "push"),
]


def run(**changes):
    """A sweep's run that arrives, holds, and ends at the target, and follows each disturbance by 0.5."""
    row = {
        "arrival_time_s": 9.5,
        "final_distance_m": 0.003,
        "left_goal": False,
        "failed": False,
        "taught_path_rmse_m": 0.001,
        "reference_lead": 0.1,
        "peak_torque_nm": 2.0,
        "first_step_m": 0.01,
        "origin": "demo_00_filtered",
    }
    for suffix in ("", "_1", "_2", "_3"):
        row |= {
            f"return_time_s{suffix}": 1.0,
            f"following_gain{suffix}": 0.5,
            f"ringing_m{suffix}": 0.005,
            f"off_path_m{suffix}": 0.1,
        }
    return row | changes


def test_a_disturbances_window_lasts_until_the_next_one_starts_or_the_run_ends():
    names = [(name, suffix) for name, suffix, _ in WINDOWS]
    windows = [window for _, _, window in WINDOWS]

    assert names == [("block", ""), ("push1", "_1"), ("push2", "_2"), ("push3", "_3")]
    assert windows == pytest.approx([24.0 - 5.5, 7.0 - 3.5, 15.0 - 7.5, 24.0 - 15.5])


@pytest.mark.parametrize(
    ("changes", "scenario", "how"),
    [
        ({}, "nominal", ""),
        ({"arrival_time_s": float("nan")}, "nominal", "never arrives"),
        ({"final_distance_m": 0.03}, "pushes", "ends away"),
        ({"left_goal": True}, "block", "leaves the goal during the hold"),
        ({"left_goal": True, "failed": True}, "pushes", ""),  # excused under the pushes
        ({"failed": True}, "offsets", "hold cut short"),
    ],
)
def test_a_run_fails_by_the_rule(changes, scenario, how):
    assert failure(run(**changes), scenario, RADIUS) == how


def test_a_hand_not_back_counts_as_back_at_the_end_of_its_window():
    measures = esn_measures(run(), run(), run(return_time_s_2=float("nan")), [run()], WINDOWS, RADIUS)

    assert measures["return_s_push2"] == 7.5
    assert not measures["back_push2"]
    assert measures["return_s"] == pytest.approx((1.0 + 1.0 + 7.5 + 1.0) / 4)
    assert measures["following"] == pytest.approx(0.5)


def test_an_esn_not_run_from_the_offsets_or_failing_from_one_does_not_pass():
    assert not esn_measures(run(), run(), run(), None, WINDOWS, RADIUS)["passes"]
    away = run(origin="demo_00_filtered +5,-10 deg", final_distance_m=0.05)
    measures = esn_measures(run(), run(), run(), [run(), away], WINDOWS, RADIUS)
    assert not measures["passes"]
    assert measures["failures"] == "demo_00_filtered +5,-10 deg: ends away"


def test_the_typical_seed_is_the_passing_one_nearest_the_medians():
    seeds = [
        (0, {"path_rmse_mm": 1.0, "following": 0.7, "ringing_mm": 5.0, "passes": True}),
        (1, {"path_rmse_mm": 1.1, "following": 0.6, "ringing_mm": 5.5, "passes": False}),  # nearest, but fails
        (2, {"path_rmse_mm": 1.4, "following": 0.5, "ringing_mm": 6.0, "passes": True}),
    ]

    assert typical_seed(seeds) == 0  # 2.5 interquartile ranges from the medians, where seed 2 is 3.5


def esns(settings):
    """ESNs of settings given as (spectral radius, following, ringing, passing seeds of 3)."""
    out = {}
    for radius, following, ringing, passing in settings:
        for seed in range(3):
            measures = {
                "passes": seed < passing,
                "path_rmse_mm": 1.0,
                "following": following + 0.01 * seed,
                "ringing_mm": ringing,
            }
            out[esn_key({"spectral_radius": radius, "seed": seed})] = measures
    return out


def test_the_qualifying_settings_within_the_guard_are_ranked_by_their_following():
    settings = rank_settings(esns([(1.0, 0.6, 5.0, 3), (1.2, 0.8, 5.0, 3), (1.35, 0.9, 12.0, 3)]), 10.0, 2, 2)
    row = {dict(k)["spectral_radius"]: v for k, v in settings.items()}

    assert row[1.2]["rank"] == 1 and row[1.2]["chosen"] == 1
    assert row[1.0]["rank"] == 2 and row[1.0]["chosen"] == 2
    assert row[1.35]["qualifies"] and not row[1.35]["within_guard"] and row[1.35]["rank"] is None
    assert row[1.2]["typical_seed"] == 1  # the median following


def test_too_few_qualifying_settings_are_made_up_from_those_with_enough_passing_seeds():
    settings = rank_settings(esns([(1.0, 0.6, 5.0, 3), (1.2, 0.8, 5.0, 2), (1.35, 0.9, 5.0, 1)]), 10.0, 2, 2)
    row = {dict(k)["spectral_radius"]: v for k, v in settings.items()}

    assert row[1.0]["chosen"] == 1  # the one qualifying, first
    assert row[1.2]["chosen"] == 2 and row[1.2]["rank"] is None
    assert row[1.2]["typical_seed"] in (0, 1)  # a passing seed
    assert row[1.35]["chosen"] is None
    assert np.isclose(row[1.2]["following"], 0.81)
