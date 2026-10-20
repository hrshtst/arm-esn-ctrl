# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the tracker settings of the robot runs."""

import sys

import numpy as np
import pytest
from skelarm import Skeleton

from arm_esn_ctrl.esn import EsnConfig
from arm_esn_ctrl.storage import REPO_ROOT
from arm_esn_ctrl.tracking import TrackerConfig, tracking_gains

# The runners import each other as modules of their own directory, as they run.
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from robot_esn import progress_source, tracker_settings, warmup_steps

CONFIG = REPO_ROOT / "experiments/demonstrations/reach_tvs.toml"


def test_the_natural_frequencies_and_damping_ratios_make_one_setting_each():
    settings = tracker_settings({"laws": ["pd"], "omegas": [20.0], "dampings": [0.2, 1.0]})

    assert [s.name for s in settings] == ["pd_w20_z0.2", "pd_w20"]
    assert settings[0].columns == {"law": "pd", "omega": 20.0, "damping": 0.2}
    skeleton = Skeleton.from_toml(CONFIG)
    posture = np.array([1.0, 1.2])
    config = TrackerConfig("pd", 20.0, 0.02, damping=0.2)
    for got, expected in zip(
        settings[0].gains(config, skeleton, posture), tracking_gains(config, skeleton, posture), strict=True
    ):
        np.testing.assert_array_equal(got, expected)


def test_given_gains_apply_to_every_joint_or_one_per_joint():
    skeleton = Skeleton.from_toml(CONFIG)
    posture = np.array([1.0, 1.2])
    (shared,) = tracker_settings({"laws": ["pd"], "kp": 80.0, "kd": 3.0})
    (per_joint,) = tracker_settings({"laws": ["pd"], "kp": [80.0, 9.0], "kd": [3.0, 0.4]})

    assert shared.name == "pd_kp80_kd3"
    assert shared.label == "pd, kp = 80, kd = 3"
    assert shared.columns == {"law": "pd", "omega": "", "damping": ""}
    assert per_joint.name == "pd_kp80-9_kd3-0.4"
    config = TrackerConfig("pd", None, 0.02)
    kp, kd = shared.gains(config, skeleton, posture)
    np.testing.assert_array_equal(kp, [80.0, 80.0])
    np.testing.assert_array_equal(kd, [3.0, 3.0])
    kp, kd = per_joint.gains(config, skeleton, posture)
    np.testing.assert_array_equal(kp, [80.0, 9.0])
    np.testing.assert_array_equal(kd, [3.0, 0.4])


@pytest.mark.parametrize(
    ("tracker", "message"),
    [
        ({"laws": ["pd"], "kp": 80.0}, "go together"),
        ({"laws": ["pd"], "omegas": [20.0], "kp": 80.0, "kd": 3.0}, "not both"),
        ({"laws": ["pd", "computed_torque"], "kp": 80.0, "kd": 3.0}, "one law"),
        ({"laws": ["pd"], "kp": 0.0, "kd": 3.0}, "positive"),
    ],
)
def test_given_gains_are_checked(tracker, message):
    with pytest.raises(ValueError, match=message):
        tracker_settings(tracker)


def test_given_gains_need_one_value_or_one_per_joint():
    (setting,) = tracker_settings({"laws": ["pd"], "kp": [80.0, 9.0, 1.0], "kd": 3.0})

    with pytest.raises(ValueError, match="one per joint"):
        setting.gains(TrackerConfig("pd", None, 0.02), Skeleton.from_toml(CONFIG), np.array([1.0, 1.2]))


def test_the_progress_table_gives_the_lead_in_degrees_and_the_window_in_seconds():
    q_demo = np.zeros((100, 2))

    source = progress_source({"lead_deg": 2.0, "window": 1.0}, q_demo, 0.01)

    assert source.lead == pytest.approx(np.radians(2.0))
    assert source.window == 100
    with pytest.raises(ValueError, match="not lag"):
        progress_source({"lead_deg": 2.0, "window": 1.0, "lag": 1.0}, q_demo, 0.01)


def test_the_warmup_is_the_esns_unless_the_evaluation_gives_one():
    config = EsnConfig(
        dt=0.01,
        warmup=1.0,
        n_neurons=10,
        spectral_radius=0.9,
        sparsity=0.1,
        leak_rate=0.1,
        input_scaling=1.0,
        bias=True,
        ridge=1e-6,
        seed=0,
    )

    assert warmup_steps({}, config) == config.warmup_steps == 100
    assert warmup_steps({"warmup": 4.0}, config) == 400
