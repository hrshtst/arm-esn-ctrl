# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""The ellipse takes' configurations agree with each other and with the many-starts takes'."""

import tomllib

import numpy as np
import pytest
from skelarm import Skeleton, build_curve, compute_inverse_kinematics

from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.disturbances import same_arm
from arm_esn_ctrl.storage import REPO_ROOT

MANUAL = REPO_ROOT / "experiments/demonstrations/ellipse_manual_v2_no_limits.toml"
SCRIPTED = REPO_ROOT / "experiments/demonstrations/ellipse_scripted_v2_no_limits.toml"
MANY_STARTS = REPO_ROOT / "experiments/demonstrations/reach_manual_v2_no_limits.toml"
CHECKS = ("start_tolerance_deg", "max_tip_speed", "min_loops")


def load(path):
    with path.open("rb") as f:
        return tomllib.load(f)


@pytest.mark.parametrize("path", [MANUAL, SCRIPTED])
def test_the_ellipse_takes_are_of_the_many_starts_arm(path):
    config, many_starts = load(path), load(MANY_STARTS)

    assert config["skeleton"] == many_starts["skeleton"]
    assert config["simulator"] == many_starts["simulator"]
    assert same_arm(Skeleton.from_toml(path), Skeleton.from_toml(MANY_STARTS))


def test_the_hand_take_is_checked_and_filtered_as_the_many_starts_takes():
    recording, many_starts = load(MANUAL)["recording"], load(MANY_STARTS)["recording"]

    assert recording["takes"] == ["data/ellipse_manual_v2_no_limits/take_00.sklog.npz"]
    assert recording["start_tolerance_deg"] == many_starts["start_tolerance_deg"]
    assert recording["max_tip_speed"] == many_starts["max_tip_speed"]
    assert recording["filter"] == many_starts["filter"] == {"kind": "lowpass", "cutoff_hz": 2.0}
    assert recording["min_loops"] == 8


def test_the_hand_and_the_script_teach_the_same_ellipse_from_the_same_start():
    manual, scripted = load(MANUAL), load(SCRIPTED)

    assert manual["task"] == scripted["task"]
    assert manual["task"] == {
        "type": "periodic_curve",
        "curve": "ellipse",
        "center": [-0.1, 0.55],
        "a": 0.2,
        "b": 0.12,
        "period": 4.0,
        "duration": 40.0,
    }
    assert manual["demonstrations"] == scripted["demonstrations"]
    assert manual["initial"] == scripted["initial"] == {"q": manual["demonstrations"]["start_q"][0]}
    assert {key: scripted["recording"][key] for key in CHECKS} == {key: manual["recording"][key] for key in CHECKS}
    assert scripted["recording"]["takes"] == ["data/ellipse_scripted_v2_no_limits/take_00.sklog.npz"]
    assert scripted["script"] == {"dt": 0.01}  # the recorder's 100 Hz


def test_the_start_posture_puts_the_hand_on_the_curve_at_angle_0_with_joint_2_positive():
    config = load(MANUAL)
    skeleton = Skeleton.from_toml(MANUAL)
    (start,) = config["demonstrations"]["start_q"]
    point = build_curve("ellipse", config["task"])(0.0)

    hand = endpoint_positions(skeleton, np.radians([start]))[0]

    assert point == pytest.approx([0.1, 0.55])
    assert np.linalg.norm(hand - point) < 0.001
    assert start[1] > 0
    # The posture that puts the hand there exactly, rounded to 0.1 deg.
    skeleton.q = np.radians(start)
    compute_inverse_kinematics(skeleton, point)
    assert start == pytest.approx(np.round(np.degrees(skeleton.q), 1).tolist())
