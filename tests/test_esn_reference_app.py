# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the interactive ESN reference app, run headless."""

import os
import time
import tomllib

import numpy as np
import pytest

# Importing the tool pulls in PyQt6; run headless.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication
from skelarm import LinkProp, Skeleton, Task

from arm_esn_ctrl.demonstrations import resample_joint_angles, simulate_reaches
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.storage import REPO_ROOT
from tools.esn_reference_app import EsnReferenceApp, build_parser

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


def make_app(demo_config, esn, *, demonstrator=True):
    skeleton = Skeleton.from_toml(CONFIG)
    skeleton.q = np.radians(demo_config["demonstrations"]["start_q"][1])  # demo 7's start
    return EsnReferenceApp(
        skeleton,
        Task.from_dict(demo_config["task"]),
        esn,
        hold=1.0,
        demonstrator_config=demo_config if demonstrator else None,
    )


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


def test_the_esn_runs_from_the_posed_start_posture(qapp, demo_config, esn):
    app = make_app(demo_config, esn, demonstrator=False)
    app.canvas.solve_to_world(0.3, 0.9)  # pose the arm by dragging its tip
    start = app.skeleton.q.copy()
    expected = esn.generate(start, 100)

    app.play()
    app.pause()
    app.advance(0.01 * (ESN_CONFIG.warmup_steps + 101))

    assert app.run is not None
    assert app.run.times[0] == pytest.approx(-ESN_CONFIG.warmup)
    assert np.array(app.run.q[: ESN_CONFIG.warmup_steps]) == pytest.approx(np.repeat(start[np.newaxis], 25, axis=0))
    assert np.array(app.run.q[ESN_CONFIG.warmup_steps :]) == pytest.approx(expected, abs=1e-12)
    assert app.skeleton.q == pytest.approx(expected[-1])


def test_dragging_does_not_move_the_arm_while_a_run_exists(qapp, demo_config, esn):
    app = make_app(demo_config, esn, demonstrator=False)
    app.step()
    app.advance(0.5)
    posture = app.skeleton.q.copy()

    app.canvas.mousePressEvent(left_click(10.0, 10.0))

    assert not app.canvas.drag_to_pose
    assert app.skeleton.q == pytest.approx(posture)


def test_reset_returns_to_the_start_posture_ready_for_posing(qapp, demo_config, esn):
    app = make_app(demo_config, esn, demonstrator=False)
    start = app.skeleton.q.copy()
    app.play()
    app.advance(1.0)

    app.reset()

    assert app.run is None and not app.is_playing
    assert app.skeleton.q == pytest.approx(start)
    assert app.canvas.drag_to_pose
    assert len(app.canvas.static_trails) == 1  # the last run's tip path, drawn faintly


def test_step_takes_one_step_at_a_time(qapp, demo_config, esn):
    app = make_app(demo_config, esn, demonstrator=False)

    app.step()
    app.step()

    assert app.run is not None
    assert len(app.run.times) == 2
    assert app.run.phase == "warming up"


def test_metrics_follow_the_reach_and_the_hold(qapp, demo_config, esn):
    app = make_app(demo_config, esn)
    app.step()
    app.advance(ESN_CONFIG.warmup + 3.0)

    run = app.run
    assert run is not None and run.demonstrator is not None
    assert run.first_step is not None and run.first_step < 0.01
    assert run.arrival_time is not None
    assert run.arrival_time == pytest.approx(run.demonstrator.arrival_time, abs=0.05)
    assert run.reach_path_distance < 0.01
    assert run.reach_joint_error is not None and run.reach_joint_error < 1.0
    assert not run.left_goal and run.hold_observed == pytest.approx(1.0)
    assert run.phase == "held for the whole hold window"
    assert len(app.canvas.trails) == 2  # the demonstrator's path, then the ESN's on top


def test_a_model_for_another_robot_is_rejected(qapp, demo_config, esn):
    link = LinkProp(length=0.6, m=1.0, i=0.1, rgx=0.3, rgy=0.0, qmin=-np.pi, qmax=np.pi)

    with pytest.raises(ValueError, match="joint"):
        EsnReferenceApp(Skeleton([link, link, link]), Task.from_dict(demo_config["task"]), esn, hold=1.0)


def test_the_command_line_needs_a_model():
    args = build_parser().parse_args([str(CONFIG), "--model", "run/esn.toml", "--speed", "0.5"])

    assert args.model.name == "esn.toml" and args.speed == 0.5 and args.hold == 2.0


def test_playing_runs_the_esn_on_its_own(qapp, demo_config, esn):
    app = make_app(demo_config, esn, demonstrator=False)
    app.speed_spin.setValue(2.0)

    app.play()
    deadline = time.monotonic() + 0.2
    while time.monotonic() < deadline:  # let the playback clock tick
        qapp.processEvents()
        time.sleep(0.002)
    app.pause()

    assert app.speed == 2.0
    assert app.run is not None
    assert len(app.run.times) >= 10  # about 200 ms x 2 / 10 ms per step, less timer jitter
    assert not app.is_playing
