# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the interactive robot reaching app, run headless."""

import os
import tomllib

import numpy as np
import pytest

# Importing the tool pulls in PyQt6; run headless.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication
from skelarm import Skeleton

from arm_esn_ctrl.demonstrations import resample_joint_angles, simulate_reaches
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.storage import REPO_ROOT
from arm_esn_ctrl.tracking import EsnSource, TrackerConfig, track
from tools.robot_app import RobotApp, build_parser, check_arguments

CONFIG = REPO_ROOT / "configs/demonstrations/reach_tvs.toml"
ESN_CONFIG = EsnConfig(
    dt=0.01,
    warmup=0.25,
    n_neurons=200,
    spectral_radius=1.3,
    sparsity=0.1,
    leak_rate=0.05,
    input_scaling=0.1,
    bias=True,
    ridge=1e-6,
    seed=0,
)
COMPUTED_TORQUE = TrackerConfig("computed_torque", 10.0, 0.02)
DT = 0.002  # the configuration's simulation step


@pytest.fixture(scope="module")
def qapp():
    """Provide a single QApplication instance for the GUI tests."""
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def demo_config():
    """The time-varying-stiffness demonstrations, cut down to two reaches of 2.5 s."""
    with CONFIG.open("rb") as f:
        config = tomllib.load(f)
    config["demonstrations"]["start_q"] = config["demonstrations"]["start_q"][6:]
    config["task"]["duration"] = 2.5
    return config


@pytest.fixture(scope="module")
def esn(demo_config):
    """An ESN trained on the two demonstrations."""
    trained = ReachingEsn(ESN_CONFIG)
    trained.fit([resample_joint_angles(log, ESN_CONFIG.dt)[1] for log in simulate_reaches(demo_config)])
    return trained


def demo7_start(demo_config):
    return np.radians(demo_config["demonstrations"]["start_q"][1])


def make_app(demo_config, **options):
    """The app with the arm at demo 7's start posture, tracking with computed torque unless the demonstrator drives."""
    skeleton = Skeleton.from_toml(CONFIG)
    skeleton.q = demo7_start(demo_config)
    if not options.get("demonstrator"):
        options.setdefault("tracker", COMPUTED_TORQUE)
    return RobotApp(skeleton, demo_config, hold=1.0, **options)


def steps(app, n):
    """Advance a paused run by ``n`` presses of Step (10 ms each)."""
    for _ in range(n):
        app.step()


def left_click(x, y):
    point = QPointF(x, y)
    return QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        point,
        point,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def test_the_launch_options_choose_the_mode(qapp, demo_config, esn):
    given = np.radians([25.0, 115.0])

    modes = {
        "esn": make_app(demo_config, esn=esn),
        "replay_from_start": make_app(demo_config),
        "replay_from_given": make_app(demo_config, given_q=given),
        "demonstrator": make_app(demo_config, demonstrator=True),
    }

    for mode, app in modes.items():
        assert app.mode == mode
    assert "ESN" in modes["esn"].mode_label.text()
    assert "(25.0, 115.0) deg" in modes["replay_from_given"].mode_label.text()
    assert "computed torque, ω = 10 rad/s" in modes["replay_from_start"].mode_label.text()
    assert "demonstrator's own (time varying stiffness)" in modes["demonstrator"].mode_label.text()


def test_the_esn_mode_simulates_the_arm_as_the_experiment_does(qapp, demo_config, esn):
    app = make_app(demo_config, esn=esn)
    start = app.skeleton.q.copy()

    app.play()
    app.pause()
    assert app.run is not None and app.run.time == pytest.approx(0.0, abs=1e-9)  # the warm-up is consumed at once
    steps(app, 100)

    skeleton = Skeleton.from_toml(CONFIG)
    skeleton.q = start
    assert app.gains is not None
    log = track(
        skeleton,
        EsnSource(esn),
        COMPUTED_TORQUE,
        app.gains,
        period=ESN_CONFIG.dt,
        warmup_steps=ESN_CONFIG.warmup_steps,
        duration=1.0,
        dt=DT,
    )
    assert app.run.time == pytest.approx(1.0)
    assert app.skeleton.q == pytest.approx(log.channel("q")[-1], abs=1e-12)


def test_forces_never_act_during_the_warmup(qapp, demo_config, esn):
    app = make_app(demo_config, esn=esn)
    start = app.skeleton.q.copy()
    app.canvas.drag_point = (1.0, 0.0)  # already pulling when the run starts

    app.play()
    app.pause()

    assert app.run is not None
    assert app.skeleton.q == pytest.approx(start, abs=1e-15)  # held still through the warm-up
    steps(app, 1)
    assert np.linalg.norm(app.run.force) > 1.0 and app.run.peak_force > 1.0


def test_a_drag_poses_the_arm_without_a_run_and_pushes_it_during_one(qapp, demo_config):
    app = make_app(demo_config)
    before = app.skeleton.q.copy()

    app.canvas.mousePressEvent(left_click(10.0, 10.0))
    posed = app.skeleton.q.copy()
    app.step()  # start a run, at time 0
    app.canvas.mousePressEvent(left_click(10.0, 10.0))

    assert not np.allclose(posed, before)  # the click posed the arm by inverse kinematics
    assert not app.canvas.posing
    assert app.canvas.drag_point is not None  # the second click grabbed the tip instead
    assert app.skeleton.q == pytest.approx(posed)  # and moved nothing by itself
    steps(app, 5)
    assert app.run is not None and np.linalg.norm(app.run.force) > 0.0


def test_replaying_from_the_start_posture_follows_the_demonstrator_without_a_jump(qapp, demo_config):
    app = make_app(demo_config)
    app.canvas.solve_to_world(0.3, 0.9)  # a posture no demonstration starts from

    app.step()
    steps(app, 200)

    run = app.run
    assert run is not None and run.tracker is not None
    assert np.degrees(run.peak_reference_speed) < 200.0  # the reference starts where the arm is
    assert run.arrival_time is not None and run.tracking_rms is not None
    assert np.degrees(run.tracking_rms) < 0.2


def test_a_given_posture_is_the_reference_origin_and_posing_away_makes_an_offset(qapp, demo_config):
    given = demo7_start(demo_config)
    app = make_app(demo_config, given_q=given)
    app.canvas.solve_to_world(0.3, 0.9)
    posed = app.skeleton.q.copy()

    app.step()
    app.step()

    run = app.run
    assert run is not None and run.tracker is not None
    assert run.start_q == pytest.approx(posed)
    reach = app._given_reference
    assert reach is not None and reach[0] == pytest.approx(given)
    jump = np.abs(reach[1] - posed).max() / ESN_CONFIG.dt  # from the posed arm to the given posture's reach
    assert run.peak_reference_speed == pytest.approx(jump)
    assert app.canvas.static_trails[0].points == pytest.approx(app._arm_points(given))  # the faint initial posture


def test_reset_returns_to_the_last_start_posture_and_keeps_the_given_one_as_the_initial_posture(qapp, demo_config):
    given = demo7_start(demo_config)
    app = make_app(demo_config, given_q=given)
    app.canvas.solve_to_world(0.3, 0.9)
    posed = app.skeleton.q.copy()
    app.play()
    app.pause()
    steps(app, 30)

    app.reset()

    assert app.run is None and app.canvas.posing and not app.is_playing
    assert app.skeleton.q == pytest.approx(posed) and app.skeleton.dq == pytest.approx([0.0, 0.0])
    assert app.initial_posture == pytest.approx(given)


def test_without_a_given_posture_the_initial_posture_is_the_last_start(qapp, demo_config):
    app = make_app(demo_config)
    assert app.initial_posture is None
    app.canvas.solve_to_world(0.3, 0.9)
    posed = app.skeleton.q.copy()

    app.step()
    app.reset()

    assert app.initial_posture == pytest.approx(posed)


def test_the_demonstrator_mode_reaches_as_the_demonstrations(qapp, demo_config):
    app = make_app(demo_config, demonstrator=True)

    app.step()
    steps(app, 100)

    config = dict(demo_config)
    config["demonstrations"] = {"start_q": demo_config["demonstrations"]["start_q"][1:]}
    demonstration = simulate_reaches(config)[0]
    assert app.run is not None and app.run.tracker is None
    assert app.skeleton.q == pytest.approx(demonstration.channel("q")[500], abs=1e-12)  # at 1.0 s


@pytest.mark.parametrize(
    ("arguments", "complaint"),
    [
        (["--model", "esn.toml", "--demonstrator"], "exclude each other"),
        (["--demonstrator", "--law", "pd", "--omega", "10"], "does not use"),
        (["--law", "pd"], "required unless --demonstrator"),
        (["--law", "pd", "--omega", "10", "--model", "esn.toml", "--period", "0.02"], "only without"),
    ],
)
def test_the_command_line_rejects_contradictions(arguments, complaint, capsys):
    parser = build_parser()
    args = parser.parse_args([str(CONFIG), *arguments])

    with pytest.raises(SystemExit):
        check_arguments(parser, args)
    assert complaint in capsys.readouterr().err


def test_ct_is_short_for_computed_torque():
    args = build_parser().parse_args([str(CONFIG), "--law", "ct", "--omega", "10"])

    assert args.law == "computed_torque"


def test_the_command_line_accepts_each_mode():
    parser = build_parser()
    for arguments in (
        ["--law", "computed_torque", "--omega", "10", "--model", "esn.toml"],
        ["--law", "ct", "--omega", "10", "--model", "esn.toml"],
        ["--law", "pd", "--omega", "20"],
        ["--law", "pd", "--omega", "20", "--pose", "29.4,88.2"],
        ["--demonstrator", "--pose", "29.4,88.2"],
    ):
        check_arguments(parser, parser.parse_args([str(CONFIG), *arguments]))
