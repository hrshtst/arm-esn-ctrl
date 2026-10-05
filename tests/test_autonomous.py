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

CONFIG = REPO_ROOT / "experiments/demonstrations/reach_tvs.toml"


def test_start_postures_are_offsets_around_each_demonstration_then_extra_groups():
    demos = {"demo_00": np.radians([[10.0, 20.0], [30.0, 40.0]]), "demo_01": np.radians([[50.0, 60.0]])}
    evaluation = {
        "start_offsets_deg": [[0.0, 0.0], [3.0, -3.0]],
        "extra_starts": {"between": [[70.0, 80.0], [75.0, 85.0]], "farther": [[5.0, 120.0]]},
    }

    starts = start_postures(evaluation, demos)

    assert [s.origin for s in starts] == [
        "demo_00",
        "demo_00 +3,-3 deg",
        "demo_01",
        "demo_01 +3,-3 deg",
        "between 0",
        "between 1",
        "farther 0",
    ]
    assert [s.group for s in starts] == [
        "demonstrated",
        "offset",
        "demonstrated",
        "offset",
        "between",
        "between",
        "farther",
    ]
    assert [s.demonstrated for s in starts] == [True, False, True, False, False, False, False]
    assert np.degrees(starts[1].q) == pytest.approx([13.0, 17.0])
    assert np.degrees(starts[6].q) == pytest.approx([5.0, 120.0])


def test_unknown_evaluation_keys_are_rejected():
    demos = {"demo_07": np.radians([[18.2, 119.9]])}
    evaluation = {"start_offsets_deg": [[0.0, 0.0]], "extra_start_q_deg": [[40.5, 71.4]]}

    with pytest.raises(ValueError, match="extra_start_q_deg"):
        start_postures(evaluation, demos)


def test_extra_start_groups_cannot_take_reserved_names():
    demos = {"demo_07": np.radians([[18.2, 119.9]])}
    evaluation = {"start_offsets_deg": [[0.0, 0.0]], "extra_starts": {"demonstrated": [[40.5, 71.4]]}}

    with pytest.raises(ValueError, match="reserved"):
        start_postures(evaluation, demos)


def test_extra_start_postures_are_optional():
    demos = {"demo_07": np.radians([[18.2, 119.9]])}

    starts = start_postures({"start_offsets_deg": [[0.0, 0.0]]}, demos)

    assert [s.origin for s in starts] == ["demo_07"]


def reach_setup(q_ref, training=None):
    """A setup whose single demonstrator reach is ``q_ref``, sampled every 0.01 s, ending at its target.

    ``training`` is the training demonstration (by default, ``q_ref`` itself).
    """
    skeleton = Skeleton.from_toml(CONFIG)
    times = np.asarray(0.01 * np.arange(len(q_ref)), dtype=np.float64)
    hand_ref = endpoint_positions(skeleton, q_ref)
    start = Start("demo_00", q_ref[0], "demonstrated")
    return Setup(
        demos={"demo_00": q_ref if training is None else training},
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
    q_run = joint_reach([18.2, 119.9], [30.0, 110.0])  # stops short of the target
    setup = reach_setup(q_ref)
    hand_run = endpoint_positions(setup.skeleton, q_run)
    run = Run(setup.starts[0], q_run, q_ref, hand_run, setup.hand_refs[0])

    m = run_metrics(run, setup)

    assert not m["arrived"] and not m["success"]
    assert np.isnan(m["arrival_delay_s"])
    assert m["reach_path_distance_m"] == pytest.approx(path_distance(hand_run, setup.hand_refs[0]))


def test_a_start_grid_surrounds_each_demonstrated_start():
    demos = {"demo_00": np.radians([[10.0, 20.0], [30.0, 40.0]])}
    evaluation = {"start_offsets_deg": [[0.0, 0.0], [2.5, 0.0]], "start_grid": {"span_deg": 5.0, "step_deg": 2.5}}

    starts = start_postures(evaluation, demos)

    assert len(starts) == 25  # 5 x 5 offsets, the two listed ones included once
    assert [s.group for s in starts].count("demonstrated") == 1
    offsets = {tuple(np.round(np.degrees(s.q) - [10.0, 20.0], 6)) for s in starts}
    assert offsets == {(a, b) for a in (-5.0, -2.5, 0.0, 2.5, 5.0) for b in (-5.0, -2.5, 0.0, 2.5, 5.0)}
    assert "demo_00 -5,+2.5 deg" in [s.origin for s in starts]


def test_an_evaluation_needs_start_postures():
    with pytest.raises(ValueError, match="start postures"):
        start_postures({}, {"demo_07": np.radians([[18.2, 119.9]])})


def test_the_training_path_ratio_tells_a_reach_of_its_own_from_a_jump_back():
    training = joint_reach([18.2, 119.9], [48.6, 97.2])
    from_offset = joint_reach([28.2, 129.9], [48.6, 97.2])  # the demonstrator, 10 deg away in both joints
    setup = reach_setup(from_offset, training=training)
    jump_back = np.vstack([from_offset[:1], training[1:]])  # back onto the demonstration in one step
    hand = {name: endpoint_positions(setup.skeleton, q) for name, q in (("own", from_offset), ("back", jump_back))}

    like_demonstrator = run_metrics(Run(setup.starts[0], from_offset, from_offset, hand["own"], hand["own"]), setup)
    jumping_back = run_metrics(Run(setup.starts[0], jump_back, from_offset, hand["back"], hand["own"]), setup)
    on_training = run_metrics(
        Run(setup.starts[0], training, training, hand["back"], hand["back"]), reach_setup(training)
    )

    assert like_demonstrator["training_path_ratio"] == pytest.approx(1.0)
    assert like_demonstrator["demonstrator_training_path_distance_deg"] > 1.0
    assert jumping_back["training_path_ratio"] < 0.05
    assert np.isnan(on_training["training_path_ratio"])  # nothing to compare: the start is on the training path
