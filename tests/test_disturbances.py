# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the disturbances of Stage 2 and the arms that meet them."""

import tomllib

import numpy as np
import pytest
from skelarm import Skeleton

from arm_esn_ctrl.demonstrations import endpoint_positions, simulate_disturbed_reach, simulate_reaches
from arm_esn_ctrl.disturbances import (
    Block,
    Disturbances,
    check_recorded_arm,
    disturbance_spans,
    keep_recorded_logs,
    make_disturbance,
    recorded_evaluation,
    same_arm,
)
from arm_esn_ctrl.storage import REPO_ROOT
from arm_esn_ctrl.tracking import ReplaySource, TrackerConfig, track, tracking_gains

CONFIG = REPO_ROOT / "experiments/demonstrations/reach_tvs.toml"
TARGET = np.array([0.0, 1.2])
PUSH = {"type": "push", "force": 5.0, "onset": 0.4, "duration": 0.1}
BLOCK = {"type": "block", "onset": 0.1, "release": 0.4, "stiffness": 20000.0, "damping": 100.0}


def tip(skeleton):
    return np.array([skeleton.links[-1].xe, skeleton.links[-1].ye])


def ramp_run(external_force, duration=0.6):
    """The arm tracking a steady joint ramp with computed torque, from a reset after a 0.05 s warm-up."""
    start = np.radians([30.0, 90.0])
    ramp = start + np.radians([0.2, -0.1]) * np.arange(200)[:, np.newaxis]
    skeleton = Skeleton.from_toml(CONFIG)
    skeleton.q = start
    config = TrackerConfig("computed_torque", 10.0, 0.02)
    gains = tracking_gains(config, skeleton, start)
    log = track(
        skeleton,
        ReplaySource(ramp.astype(np.float64)),
        config,
        gains,
        period=0.01,
        warmup_steps=5,
        duration=duration,
        dt=0.002,
        external_force=external_force,
    )
    return log, endpoint_positions(skeleton, log.channel("q"))


def test_a_push_acts_sideways_to_the_reach_for_its_duration():
    start_hand = np.array([0.5, 1.2])
    push = make_disturbance(PUSH, start_hand, TARGET)
    arm = Skeleton.from_toml(CONFIG)

    assert push is not None
    force = push(0.45, arm)
    assert np.linalg.norm(force) == pytest.approx(5.0)
    assert force @ (TARGET - start_hand) == pytest.approx(0.0)
    assert force == pytest.approx([0.0, -5.0])  # the reach goes in -x; counterclockwise from it is -y
    assert push(0.39, arm) == pytest.approx([0.0, 0.0])
    assert push(0.5, arm) == pytest.approx([0.0, 0.0])


def test_a_push_acts_along_the_reach_forward_or_backward():
    start_hand = np.array([0.5, 1.2])
    arm = Skeleton.from_toml(CONFIG)
    forward = make_disturbance(PUSH | {"direction": "forward"}, start_hand, TARGET)
    backward = make_disturbance(PUSH | {"direction": "backward"}, start_hand, TARGET)
    across = make_disturbance(PUSH | {"direction": "across"}, start_hand, TARGET)
    default = make_disturbance(PUSH, start_hand, TARGET)

    assert forward is not None and backward is not None and across is not None and default is not None
    assert forward(0.45, arm) == pytest.approx([-5.0, 0.0])  # toward the target
    assert backward(0.45, arm) == pytest.approx([5.0, 0.0])
    assert across(0.45, arm) == pytest.approx(default(0.45, arm))


def test_an_angle_turns_the_push_counterclockwise_from_the_reach():
    start_hand = np.array([0.5, 1.2])  # the reach goes in -x
    arm = Skeleton.from_toml(CONFIG)
    for angle, word in ((0.0, "forward"), (90.0, "across"), (180.0, "backward")):
        by_angle = make_disturbance(PUSH | {"angle_deg": angle}, start_hand, TARGET)
        by_word = make_disturbance(PUSH | {"direction": word}, start_hand, TARGET)
        assert by_angle is not None and by_word is not None
        assert by_angle(0.45, arm) == pytest.approx(by_word(0.45, arm))

    thirty = make_disturbance(PUSH | {"angle_deg": 30.0}, start_hand, TARGET)

    assert thirty is not None
    assert thirty(0.45, arm) == pytest.approx(5.0 * np.array([-np.cos(np.radians(30.0)), -np.sin(np.radians(30.0))]))


def test_a_push_takes_a_direction_or_an_angle_not_both():
    with pytest.raises(ValueError, match="not both"):
        make_disturbance(PUSH | {"direction": "forward", "angle_deg": 0.0}, np.array([0.5, 1.2]), TARGET)


def test_several_disturbances_act_each_at_its_time_and_add_up():
    start_hand = np.array([0.5, 1.2])  # the reach goes in -x
    arm = Skeleton.from_toml(CONFIG)
    pushes = [
        PUSH | {"onset": 0.2, "angle_deg": 0.0},
        PUSH | {"onset": 0.4, "angle_deg": 90.0},
        PUSH | {"onset": 0.45, "force": 2.0, "angle_deg": 0.0},
    ]

    force = make_disturbance(pushes, start_hand, TARGET)

    assert isinstance(force, Disturbances)
    assert force(0.25, arm) == pytest.approx([-5.0, 0.0])
    assert force(0.42, arm) == pytest.approx([0.0, -5.0])
    assert force(0.47, arm) == pytest.approx([-2.0, -5.0])  # two at once: they add up
    assert force(0.6, arm) == pytest.approx([0.0, 0.0])


def test_several_blocks_each_hold_the_tip_where_it_was():
    windows = [(0.05, 0.15), (0.3, 0.4)]
    blocks = make_disturbance(
        [BLOCK | {"onset": onset, "release": release} for onset, release in windows], np.zeros(2), TARGET
    )

    log, hand = ramp_run(blocks)

    assert isinstance(blocks, Disturbances)
    held = [part.held for part in blocks.parts if isinstance(part, Block) and part.held is not None]
    assert len(held) == 2
    assert np.linalg.norm(held[1] - held[0]) > 0.05  # the arm moved on between the two
    for (onset, release), point in zip(windows, held, strict=True):
        window = (log.times >= onset + 0.002) & (log.times < release)
        assert np.linalg.norm(hand[window] - point, axis=1).max() < 0.003


def test_a_bad_disturbance_in_a_list_is_rejected_by_its_place():
    with pytest.raises(ValueError, match="disturbance 2 of 2"):
        make_disturbance([PUSH, {"type": "shake"}], np.zeros(2), TARGET)


def test_the_spans_are_when_each_disturbance_acts():
    assert disturbance_spans(None) == []
    assert disturbance_spans(PUSH) == [pytest.approx((0.4, 0.5))]
    assert disturbance_spans([PUSH, BLOCK]) == [pytest.approx((0.4, 0.5)), pytest.approx((0.1, 0.4))]


def test_forces_act_on_the_task_clock():
    times = []

    def watch(t, skeleton):
        times.append(t)
        return np.zeros(2)

    ramp_run(watch, duration=0.1)

    assert times[0] == pytest.approx(-0.05)
    assert min(abs(t) for t in times) < 1e-9


def test_a_block_holds_the_tip_where_it_was_and_then_lets_go():
    block = make_disturbance(BLOCK, np.zeros(2), TARGET)
    _, free_hand = ramp_run(None)

    log, hand = ramp_run(block)

    assert isinstance(block, Block) and block.held is not None
    blocked = (log.times >= 0.1 + 0.002) & (log.times < 0.4)
    assert np.linalg.norm(hand[blocked] - block.held, axis=1).max() < 0.003
    assert np.linalg.norm(free_hand[blocked] - block.held, axis=1).max() > 0.05  # it would have moved on
    after = log.times >= 0.4
    assert np.abs(log.channel("ext_force")[after]).max() == 0.0


def test_unknown_disturbances_and_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown disturbance type"):
        make_disturbance({"type": "shake"}, np.zeros(2), TARGET)
    with pytest.raises(ValueError, match="needs the keys"):
        make_disturbance(PUSH | {"angle": 90.0}, np.zeros(2), TARGET)
    with pytest.raises(ValueError, match="needs the keys"):
        make_disturbance({"type": "push", "force": 5.0, "onset": 0.4}, np.zeros(2), TARGET)
    with pytest.raises(ValueError, match="unknown push direction"):
        make_disturbance(PUSH | {"direction": "up"}, np.zeros(2), TARGET)
    with pytest.raises(ValueError, match="needs the keys"):
        make_disturbance(BLOCK | {"direction": "forward"}, np.zeros(2), TARGET)


def test_without_a_force_the_disturbed_demonstrator_reaches_as_the_demonstrations():
    with CONFIG.open("rb") as f:
        config = tomllib.load(f)
    config["demonstrations"]["start_q"] = config["demonstrations"]["start_q"][7:]
    config["task"]["duration"] = 1.0
    undisturbed = simulate_reaches(config)[0]

    log = simulate_disturbed_reach(
        config, np.radians(config["demonstrations"]["start_q"][0]), 1.0, lambda t, s: np.zeros(2)
    )

    assert log.channel("q") == pytest.approx(undisturbed.channel("q"), abs=1e-12)
    assert log.channel("ext_force") == pytest.approx(0.0)


TWO_PUSHES = [
    {"type": "push", "force": 4.0, "onset": 0.1, "duration": 0.1},
    {"type": "push", "force": 3.0, "onset": 0.35, "duration": 0.05, "direction": "forward"},
]


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    """A run pushed twice, saved under data/recorded_forces/ in a storage root of its own; its table and its log."""
    monkeypatch.setenv("ARM_ESN_CTRL_STORAGE_ROOT", str(tmp_path))
    log, _ = ramp_run(make_disturbance(TWO_PUSHES, np.array([0.5, 1.2]), TARGET), duration=0.6)
    path = tmp_path / "data" / "recorded_forces" / "pushed.sklog.npz"
    path.parent.mkdir(parents=True)
    log.save(path)
    return {"type": "recorded", "log": "data/recorded_forces/pushed.sklog.npz"}, log


def test_a_recorded_disturbance_replays_the_logs_force_on_the_task_clock(recorded):
    table, log = recorded
    replay = make_disturbance(table, np.array([0.5, 1.2]), TARGET)
    arm = Skeleton.from_toml(CONFIG)

    assert replay is not None
    forces = np.array([replay(float(t), arm) for t in log.times[:-1]])
    assert np.array_equal(forces, log.channel("ext_force")[:-1])


def test_a_recorded_disturbance_replays_the_same_run_exactly(recorded):
    """The arm that recorded the force, meeting it again, retraces its run."""
    table, log = recorded
    again, _ = ramp_run(make_disturbance(table, np.array([0.5, 1.2]), TARGET, dt=0.002), duration=0.6)

    assert np.array_equal(again.channel("q"), log.channel("q"))


def test_with_another_step_a_recorded_force_keeps_its_impulse(recorded):
    table, log = recorded
    replay = make_disturbance(table, np.array([0.5, 1.2]), TARGET, dt=0.001)
    arm = Skeleton.from_toml(CONFIG)

    assert replay is not None
    impulse = sum(replay(-0.05 + k * 0.001, arm) * 0.001 for k in range(650))
    assert impulse == pytest.approx((log.channel("ext_force")[:-1] * 0.002).sum(axis=0))


def test_the_spans_of_a_recorded_disturbance_are_its_pushes(recorded):
    table, _ = recorded

    apart = table | {"min_gap": 0.1}

    assert disturbance_spans(table) == [pytest.approx((0.1, 0.4))]  # 0.15 s apart: one push by the default gap
    assert disturbance_spans(apart) == [pytest.approx((0.1, 0.2)), pytest.approx((0.35, 0.4))]
    assert disturbance_spans(apart | {"threshold": 3.5}) == [pytest.approx((0.1, 0.2))]
    assert disturbance_spans(apart | {"time_offset": 1.0}) == [pytest.approx((1.1, 1.2)), pytest.approx((1.35, 1.4))]


def test_runs_under_a_recorded_disturbance_start_from_its_logs_start(recorded):
    table, log = recorded

    evaluation = recorded_evaluation({"duration": 0.6, "hold": 0.1}, table)

    assert evaluation["extra_starts"]["recorded"] == [pytest.approx(np.degrees(log.channel("q")[0]))]
    assert recorded_evaluation({"duration": 0.6}, PUSH) == {"duration": 0.6}  # nothing recorded, nothing changed


def test_a_recorded_disturbance_refuses_other_starts_a_short_run_and_a_second_log(recorded):
    table, _ = recorded

    with pytest.raises(ValueError, match="start posture"):
        recorded_evaluation({"duration": 0.6, "start_offsets_deg": [[0.0, 0.0]]}, table)
    with pytest.raises(ValueError, match=r"until 0\.40 s"):
        recorded_evaluation({"duration": 0.3}, table)
    with pytest.raises(ValueError, match="one recorded"):
        recorded_evaluation({"duration": 0.6}, [table, table])


def test_a_recorded_disturbance_with_scripted_ones_is_allowed_with_a_warning(recorded):
    table, _ = recorded
    with pytest.warns(UserWarning, match="add up"):
        recorded_evaluation({"duration": 0.6}, [table, PUSH])


def test_a_recorded_disturbance_of_another_arm_is_refused(recorded):
    table, _ = recorded
    other = Skeleton.from_toml(CONFIG)
    other.links[2].prop.m *= 2.0

    assert same_arm(Skeleton.from_toml(CONFIG), Skeleton.from_toml(CONFIG))
    assert not same_arm(other, Skeleton.from_toml(CONFIG))
    check_recorded_arm(table, Skeleton.from_toml(CONFIG))
    with pytest.raises(ValueError, match="another arm"):
        check_recorded_arm(table, other)


def test_a_run_keeps_a_copy_of_its_recorded_log_and_its_checksum(recorded, tmp_path):
    table, _ = recorded
    run_dir = tmp_path / "results" / "experiment" / "run"
    run_dir.mkdir(parents=True)

    keep_recorded_logs(table, run_dir)

    copy = run_dir / "pushed.sklog.npz"
    assert copy.read_bytes() == (tmp_path / table["log"]).read_bytes()
    with (run_dir / "recorded_forces.toml").open("rb") as f:
        kept = tomllib.load(f)["log"]
    assert kept[0]["log"] == table["log"]
    assert len(kept[0]["sha256"]) == 64


def test_a_recorded_disturbance_needs_a_log_with_a_force(tmp_path, monkeypatch):
    monkeypatch.setenv("ARM_ESN_CTRL_STORAGE_ROOT", str(tmp_path))
    log, _ = ramp_run(None, duration=0.1)
    (tmp_path / "data").mkdir()
    log.save(tmp_path / "data" / "quiet.sklog.npz")

    with pytest.raises(ValueError, match="ext_force"):
        make_disturbance({"type": "recorded", "log": "data/quiet.sklog.npz"}, np.array([0.5, 1.2]), TARGET)
