# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for comparing the arms of robot runs that met the same forces."""

import sys

import numpy as np
import pytest
from skelarm import Skeleton, StateLog

from arm_esn_ctrl.storage import REPO_ROOT

# The runners import each other as modules of their own directory, as they run.
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from compare_arms import Arm, check_same_forces, shared_settings

CONFIG = REPO_ROOT / "experiments/demonstrations/reach_tvs.toml"


def run_log(forces):
    """A log of a still arm meeting ``forces``, one per frame, every 2 ms."""
    log = StateLog(Skeleton.from_toml(CONFIG), producer="test")
    for k, force in enumerate(forces):
        log.record(0.002 * k, q=np.zeros(2), dq=np.zeros(2), tau=np.zeros(2), ext_force=np.asarray(force, float))
    return log


def test_arms_that_met_the_same_forces_are_compared():
    pushed = [[0.0, 0.0], [3.0, 1.0], [3.0, 1.0], [0.0, 0.0]]
    final = [[0.0, 0.0], [3.0, 1.0], [3.0, 1.0], [5.0, 5.0]]  # the final frame's force acts on no step

    check_same_forces({"pd_w20_z0.4": {"ESN #1": run_log(pushed), "replay": run_log(final)}})


def test_arms_that_met_different_forces_are_refused():
    pushed = [[0.0, 0.0], [3.0, 1.0], [3.0, 1.0], [0.0, 0.0]]
    other = [[0.0, 0.0], [3.0, 1.0], [2.0, 1.0], [0.0, 0.0]]

    with pytest.raises(ValueError, match="different forces"):
        check_same_forces({"pd_w20_z0.4": {"ESN #1": run_log(pushed), "ESN #2": run_log(other)}})


def test_only_the_tracker_settings_every_run_has_are_compared(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    for run, settings in ((first, ("pd_w20_z0.2", "pd_w20_z0.4")), (second, ("pd_w20_z0.4", "pd_w20_z0.8"))):
        for setting in settings:
            (run / setting).mkdir(parents=True)

    arms = [Arm("ESN #1", first, "esn", "#2a78d6"), Arm("ESN #2", second, "esn", "#5b2a86")]

    assert shared_settings(arms) == ["pd_w20_z0.4"]
    with pytest.raises(ValueError, match="share no tracker setting"):
        shared_settings([arms[0], Arm("other", tmp_path, "esn", "#000000")])
