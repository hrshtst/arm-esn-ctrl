# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the short pushes all along the reach."""

import sys
from typing import Any

import numpy as np

from arm_esn_ctrl.disturbances import disturbance_spans
from arm_esn_ctrl.storage import REPO_ROOT

# The runners import each other as modules of their own directory, as they run.
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from push_pulses import Pulse, against_undisturbed, pulses


def test_the_pulses_are_every_force_onset_and_angle():
    table = {"forces": [3.0, 6.0], "onsets": [1.0, 2.0], "angles_deg": [0.0, 90.0, 180.0], "duration": 0.2}
    every = pulses(table)

    assert len(every) == 12
    assert every[0] == Pulse(1.0, 0.0, 3.0, 0.2)
    assert every[1] == Pulse(1.0, 90.0, 3.0, 0.2)
    assert every[3] == Pulse(2.0, 0.0, 3.0, 0.2)
    assert every[6] == Pulse(1.0, 0.0, 6.0, 0.2)
    assert disturbance_spans(every[3].disturbance()) == [(2.0, 2.2)]


def test_a_pulsed_run_is_measured_against_its_arms_undisturbed_run():
    times = np.linspace(0.0, 5.0, 501)
    nan = float("nan")
    base_progress = np.clip(times / 4.0, 0.0, 1.0)
    pushed_progress = base_progress.copy()
    pushed_progress[220:240] = 0.4  # pushed back from 0.5 at the onset, at 2 s, to 0.4
    base_tau, pushed_tau = np.ones((501, 2)), np.ones((501, 2))
    pushed_tau[230] = [-4.0, 2.0]
    rows: list[dict[str, Any]] = [
        {"arm": "esn1", "onset": nan, "arrival_time_s": 4.0},
        {"arm": "esn1", "onset": 2.0, "arrival_time_s": 4.5},
        {"arm": "replay", "onset": nan, "arrival_time_s": 4.0},
    ]
    channels = [
        {"progress": base_progress, "tau": base_tau},
        {"progress": pushed_progress, "tau": pushed_tau},
        {"progress": base_progress, "tau": base_tau},
    ]
    against_undisturbed(rows, channels, times, 1.0)

    assert np.isnan(rows[0]["setback"])
    assert rows[1]["arrival_shift_s"] == 0.5
    np.testing.assert_allclose(rows[1]["setback"], 0.1, atol=1e-3)
    assert rows[1]["peak_torque_rise_nm"] == 3.0
