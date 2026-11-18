# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Takes of a periodic motion: the check of their loops, the take a script makes, and their import."""

import csv
import sys
import tomllib

import numpy as np
import pytest
from skelarm import Skeleton, StateLog, build_curve

from arm_esn_ctrl.demonstrations import check_take, endpoint_positions
from arm_esn_ctrl.metrics import loop_phase
from arm_esn_ctrl.storage import REPO_ROOT

# The runners import each other as modules of their own directory, as they run.
sys.path.insert(0, str(REPO_ROOT / "experiments"))
import import_demonstrations
import make_periodic_take

MANUAL = REPO_ROOT / "experiments/demonstrations/ellipse_manual_v2_no_limits.toml"
SCRIPTED = REPO_ROOT / "experiments/demonstrations/ellipse_scripted_v2_no_limits.toml"


def load(path):
    with path.open("rb") as f:
        return tomllib.load(f)


def ellipse_take(loops, pause=1.0, clockwise=False, rest=0.0):
    """A take as the recorder saves one: still for ``pause`` seconds, then the hand around the task's ellipse.

    The hand goes ``loops`` times around, 4 s a loop, from the curve's point at angle
    0, with the elbow on joint 2's positive side, and then rests for ``rest`` seconds
    before the take ends; the samples are near 100 Hz, their intervals varying a
    little, as the recorder's clock does.
    """
    skeleton = Skeleton.from_toml(MANUAL)
    task = load(MANUAL)["task"]
    rng = np.random.default_rng(0)
    times = np.cumsum(np.concatenate([[0.0], rng.uniform(0.009, 0.012, 6000)]))
    times = times[times <= pause + 4.0 * loops + rest]
    angle = (-1 if clockwise else 1) * 2 * np.pi * np.clip(times - pause, 0.0, 4.0 * loops) / 4.0
    curve = build_curve("ellipse", task)
    hand = np.array([curve(theta) for theta in angle])
    # The two-link arm's inverse kinematics, joint 2 positive.
    upper, forearm = 0.5, 0.4
    q2 = np.arccos((np.sum(hand**2, axis=1) - upper**2 - forearm**2) / (2 * upper * forearm))
    q1 = np.arctan2(hand[:, 1], hand[:, 0]) - np.arctan2(forearm * np.sin(q2), upper + forearm * np.cos(q2))
    log = StateLog(skeleton)
    for t, posture in zip(times, np.column_stack([q1, q2]), strict=True):
        skeleton.q = posture
        log.record_skeleton(skeleton, float(t))
    return log


def check(log, min_loops=8):
    config = load(MANUAL)
    recording = config["recording"]
    return check_take(
        log,
        Skeleton.from_toml(MANUAL),
        start_q=np.radians(config["demonstrations"]["start_q"][0]),
        target=None,
        radius=0.0,
        hold=0.0,
        start_tolerance=np.radians(recording["start_tolerance_deg"]),
        max_tip_speed=recording["max_tip_speed"],
        center=config["task"]["center"],
        min_loops=min_loops,
        end_window=recording["end_window"],
        end_speed_ratio=recording["end_speed_ratio"],
    )


def test_a_take_that_goes_around_the_curve_enough_times_counterclockwise_is_a_demonstration():
    assert check(ellipse_take(8.5)) == []


def test_a_take_of_fewer_loops_than_the_least_is_not_a_demonstration():
    (problem,) = check(ellipse_take(7.5))

    assert "7 whole times, fewer than 8" in problem


def test_a_take_that_goes_around_clockwise_is_not_a_demonstration():
    (problem,) = check(ellipse_take(8.5, clockwise=True))

    assert "clockwise" in problem


def test_a_take_whose_hand_rests_for_a_second_at_its_end_is_not_a_demonstration():
    (problem,) = check(ellipse_take(8.5, rest=1.0))

    assert "end the take while the hand moves" in problem


def test_a_take_that_ends_while_the_hand_goes_around_is_a_demonstration():
    assert check(ellipse_take(8.3)) == []  # ended near the top of the ellipse, where the hand is fastest


def test_the_scripted_take_puts_the_hand_on_the_curve_ten_times_around_from_the_start_posture(tmp_path, monkeypatch):
    monkeypatch.setenv("ARM_ESN_CTRL_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["make_periodic_take.py", str(SCRIPTED)])
    config = load(SCRIPTED)
    task = config["task"]

    make_periodic_take.main()

    log = StateLog.load(tmp_path / config["recording"]["takes"][0])
    times, q = log.times, log.channel("q")
    assert times == pytest.approx(0.01 * np.arange(4001))  # 40 s every 10 ms
    hand = endpoint_positions(Skeleton.from_toml(SCRIPTED), q)
    curve = build_curve("ellipse", task)
    on_curve = np.array([curve(2 * np.pi * t / task["period"]) for t in times])
    assert np.linalg.norm(hand - on_curve, axis=1).max() < 1e-6
    turned = loop_phase(hand, task["center"])
    assert turned[-1] - turned[0] == pytest.approx(10 * 2 * np.pi, abs=1e-5)
    # Continuous, on the start posture's elbow side, and from it.
    assert np.degrees(np.abs(np.diff(q, axis=0))).max() < 0.5
    assert q[:, 1].min() > 0
    start = np.radians(config["demonstrations"]["start_q"][0])
    assert np.degrees(np.abs(q[0] - start)).max() < 0.1
    assert log.extra["acquisition"]["requested_rate_hz"] == pytest.approx(100.0)
    assert log.extra["playback"]["task"] == task


def test_the_script_never_overwrites_a_take(tmp_path, monkeypatch):
    monkeypatch.setenv("ARM_ESN_CTRL_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["make_periodic_take.py", str(SCRIPTED)])
    take = tmp_path / load(SCRIPTED)["recording"]["takes"][0]
    take.parent.mkdir(parents=True)
    take.write_bytes(b"a take")

    with pytest.raises(SystemExit, match="exists"):
        make_periodic_take.main()

    assert take.read_bytes() == b"a take"


def save_take(storage, log):
    take = storage / load(MANUAL)["recording"]["takes"][0]
    take.parent.mkdir(parents=True)
    log.save(take)


def test_a_periodic_take_is_imported_with_its_loops_and_its_taught_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("ARM_ESN_CTRL_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["import_demonstrations.py", str(MANUAL)])
    save_take(tmp_path, ellipse_take(8.5))

    import_demonstrations.main()

    (run,) = (tmp_path / "results/demonstrations").iterdir()
    assert {"demo_00.sklog.npz", "demo_00_filtered.sklog.npz", "demonstrations.png"} <= {p.name for p in run.iterdir()}
    with (run / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [row["demo"] for row in rows] == ["demo_00", "demo_00_filtered"]
    for row in rows:
        assert "arrival_time_s" not in row
        assert "movement_time" not in row
        assert int(row["loops"]) == 8
        # The taught loops, after the first, which also holds the pause before moving.
        assert float(row["median_period_s"]) == pytest.approx(4.0, abs=0.02)
        assert float(row["shortest_period_s"]) == pytest.approx(4.0, abs=0.02)
        assert float(row["longest_period_s"]) == pytest.approx(4.0, abs=0.02)
        assert float(row["taught_loop_start_s"]) == pytest.approx(5.0, abs=0.02)
        assert float(row["taught_loop_mismatch_deg"]) < 1.0
        assert float(row["onset_time_s"]) == pytest.approx(1.0, abs=0.1)
        assert float(row["samples"]) > 3000
    recorded = rows[0]
    assert float(recorded["curve_max_m"]) < 1e-5  # on the curve, as the polygon it is measured against
    assert float(recorded["loop_spread_m"]) < 1e-3
    assert float(recorded["taught_loop_mismatch_deg"]) < 0.5


def test_a_periodic_take_of_too_few_loops_is_not_imported(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ARM_ESN_CTRL_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["import_demonstrations.py", str(MANUAL)])
    save_take(tmp_path, ellipse_take(5.5))

    with pytest.raises(SystemExit):
        import_demonstrations.main()

    assert "fewer than 8" in capsys.readouterr().out
    assert not (tmp_path / "results").exists()


def test_the_scripted_take_is_imported_as_configured_with_its_filtered_copy_the_take_itself(tmp_path, monkeypatch):
    monkeypatch.setenv("ARM_ESN_CTRL_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["make_periodic_take.py", str(SCRIPTED)])
    make_periodic_take.main()
    monkeypatch.setattr(sys, "argv", ["import_demonstrations.py", str(SCRIPTED)])

    import_demonstrations.main()

    (run,) = (tmp_path / "results/demonstrations").iterdir()
    made, filtered = (StateLog.load(run / f"{name}.sklog.npz") for name in ("demo_00", "demo_00_filtered"))
    assert filtered.channel("q") == pytest.approx(made.channel("q"), abs=1e-12)
    with (run / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        assert int(row["loops"]) >= 9  # the tenth ends at the last sample
        assert float(row["median_period_s"]) == pytest.approx(4.0, abs=1e-6)
        assert float(row["curve_max_m"]) < 1e-5
        assert float(row["taught_loop_mismatch_deg"]) < 1e-3
        assert float(row["achieved_rate_hz"]) == pytest.approx(100.0)
