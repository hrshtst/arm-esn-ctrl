# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""The push search's configurations are the many-starts experiment's, but for the pushes and the ESNs."""

import tomllib

import pytest

from arm_esn_ctrl.storage import REPO_ROOT

MANY_STARTS = REPO_ROOT / "experiments/manual_demonstration_v2_many_starts"
EXPERIMENT = REPO_ROOT / "experiments/manual_demonstration_v2_many_starts_push_search"
SWEPT = ("leak_rate", "spectral_radius", "input_scaling", "ridge")


def load(path):
    with path.open("rb") as f:
        return tomllib.load(f)


def assert_pushed_as(pushed, undisturbed):
    """``pushed`` is the sweep ``undisturbed`` under the pushes of the many-starts analysis at their forces."""
    pushes = load(MANY_STARTS / "pushes1x_robot_esn1.toml")
    for table in ("demonstrations", "esn", "sweep", "tracker"):
        assert pushed[table] == undisturbed[table]
    assert pushed.get("settings") == undisturbed.get("settings")
    assert "ranking" not in pushed  # score_sweeps.py scores the pushed runs with the undisturbed ones
    assert pushed["disturbance"] == pushes["disturbance"]
    assert [push["force"] for push in pushed["disturbance"]] == [3.0, 5.0, 4.0]
    measures = ("return_tolerance", "ringing_cutoff_hz", "ringing_window", "following_window")
    assert pushed["evaluation"] == undisturbed["evaluation"] | {name: pushes["evaluation"][name] for name in measures}


@pytest.mark.parametrize("prefix", ["", "one_take_"])
def test_the_coarse_searchs_pushed_runs_are_the_many_starts_coarse_search_under_the_pushes(prefix):
    pushed = load(EXPERIMENT / f"{prefix}coarse_pushes.toml")
    undisturbed = load(MANY_STARTS / f"{prefix}coarse_search.toml")

    assert_pushed_as(pushed, undisturbed)
    assert pushed["esn"]["seed"] == 0
    assert len(pushed["demonstrations"]["train"]) == (1 if prefix else 9)
