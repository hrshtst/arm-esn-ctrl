# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the start postures and the metrics of autonomous runs."""

import numpy as np
import pytest
from skelarm import Skeleton

from arm_esn_ctrl.autonomous import Run, Setup, Start, run_metrics, start_postures
from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.metrics import path_distance
from arm_esn_ctrl.storage import REPO_ROOT

CONFIG = REPO_ROOT / "configs/demonstrations/reach_tvs.toml"


def test_start_postures_are_offsets_around_each_demonstration_then_new_ones():
    demos = {"demo_00": np.radians([[10.0, 20.0], [30.0, 40.0]]), "demo_01": np.radians([[50.0, 60.0]])}
    evaluation = {"start_offsets_deg": [[0.0, 0.0], [3.0, -3.0]], "extra_start_q_deg": [[70.0, 80.0]]}

    starts = start_postures(evaluation, demos)

    assert [s.origin for s in starts] == ["demo_00", "demo_00 +3,-3 deg", "demo_01", "demo_01 +3,-3 deg", "new"]
    assert [s.demonstrated for s in starts] == [True, False, True, False, False]
    assert np.degrees(starts[1].q) == pytest.approx([13.0, 17.0])
    assert np.degrees(starts[4].q) == pytest.approx([70.0, 80.0])


def test_extra_start_postures_are_optional():
    demos = {"demo_07": np.radians([[18.2, 119.9]])}

    starts = start_postures({"start_offsets_deg": [[0.0, 0.0]]}, demos)

    assert [s.origin for s in starts] == ["demo_07"]


def reach_setup(q_ref):
    """A setup whose single demonstrator reach is ``q_ref``, sampled every 0.01 s, ending at its target."""
    skeleton = Skeleton.from_toml(CONFIG)
    times = np.asarray(0.01 * np.arange(len(q_ref)), dtype=np.float64)
    hand_ref = endpoint_positions(skeleton, q_ref)
    start = Start("demo_00", q_ref[0], demonstrated=True)
    return Setup(
        demos={"demo_00": q_ref},
        starts=[start],
        demonstrator_logs=[],
        q_refs=[q_ref],
        hand_refs=[hand_ref],
        skeleton=skeleton,
        task={},
        target=hand_ref[-1],
        radius=0.02,
        hold=0.5,
        times=times,
    )


def joint_reach(start, goal, n=200):
    tau = np.clip(np.linspace(0.0, 2.0, n), 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    return np.radians(start) + np.outer(s, np.radians(goal) - np.radians(start))


def test_a_run_identical_to_the_demonstrator_has_no_reach_error_and_holds():
    q_ref = joint_reach([18.2, 119.9], [48.6, 97.2])
    setup = reach_setup(q_ref)
    run = Run(setup.starts[0], q_ref, q_ref, setup.hand_refs[0], setup.hand_refs[0])

    m = run_metrics(run, setup)

    assert m["reach_path_distance_m"] == pytest.approx(0.0, abs=1e-12)
    assert m["reach_joint_error_deg"] == pytest.approx(0.0, abs=1e-12)
    assert m["arrival_delay_s"] == 0.0
    assert m["success"] and m["hold_observed_s"] == pytest.approx(0.5)


def test_a_run_that_never_arrives_fails_and_is_measured_over_the_whole_run():
    q_ref = joint_reach([18.2, 119.9], [48.6, 97.2])
    q_esn = joint_reach([18.2, 119.9], [30.0, 110.0])  # stops short of the target
    setup = reach_setup(q_ref)
    hand_esn = endpoint_positions(setup.skeleton, q_esn)
    run = Run(setup.starts[0], q_esn, q_ref, hand_esn, setup.hand_refs[0])

    m = run_metrics(run, setup)

    assert not m["arrived"] and not m["success"]
    assert np.isnan(m["arrival_delay_s"])
    assert m["reach_path_distance_m"] == pytest.approx(path_distance(hand_esn, setup.hand_refs[0]))
