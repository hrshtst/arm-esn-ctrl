# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""The four-link arm's configurations take skelarm's examples/reach_four_dof_robot.toml unchanged."""

import tomllib

import numpy as np
from skelarm import Skeleton

from arm_esn_ctrl.postures import joint_limits_deg
from arm_esn_ctrl.storage import REPO_ROOT

EXAMPLE = REPO_ROOT / "third_party/skelarm/examples/reach_four_dof_robot.toml"
DEMONSTRATION = REPO_ROOT / "experiments/demonstrations/reach_manual_four_dof.toml"
V2 = REPO_ROOT / "experiments/demonstrations/reach_manual_v2.toml"


def load(path):
    with path.open("rb") as f:
        return tomllib.load(f)


def test_the_demonstration_takes_the_examples_arm_start_task_and_simulator():
    example, config = load(EXAMPLE), load(DEMONSTRATION)

    for table in ("skeleton", "initial", "task", "simulator"):
        assert config[table] == example[table], table
    assert "controller" not in config  # taught by hand: no demonstrator
    assert config["demonstrations"]["start_q"] == [example["initial"]["q"]]


def test_the_demonstration_is_recorded_and_checked_as_the_v2_take():
    recording, v2 = load(DEMONSTRATION)["recording"], load(V2)["recording"]

    assert recording["take"] == "data/reach_manual_four_dof/reach_manual_four_dof.sklog.npz"
    assert {k: v for k, v in recording.items() if k != "take"} == {k: v for k, v in v2.items() if k != "take"}


def test_the_arm_has_four_joints_within_their_limits_at_the_start():
    skeleton = Skeleton.from_toml(DEMONSTRATION)
    lower, upper = joint_limits_deg(skeleton)

    assert skeleton.num_joints == 4
    np.testing.assert_array_equal(lower, [-20.0, 0.0, 0.0, -20.0])
    np.testing.assert_array_equal(upper, [30.0, 120.0, 150.0, 90.0])
    start = np.array(load(DEMONSTRATION)["initial"]["q"])
    assert np.all(start >= lower)
    assert np.all(start <= upper)
