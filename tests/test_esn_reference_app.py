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
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from skelarm import LinkProp, Skeleton, Task

from arm_esn_ctrl.demonstrations import resample_joint_angles, simulate_reaches
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.storage import REPO_ROOT
from tools.esn_reference_app import EsnReferenceApp, build_parser

CONFIG = REPO_ROOT / "experiments/demonstrations/reach_tvs.toml"
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


def make_app(demo_config, esn, *, demonstrator=True, compare=True, show_warmup=False):
    skeleton = Skeleton.from_toml(CONFIG)
    skeleton.q = np.radians(demo_config["demonstrations"]["start_q"][1])  # demo 7's start
    return EsnReferenceApp(
        skeleton,
        Task.from_dict(demo_config["task"]),
        esn,
        hold=1.0,
        demonstrator_config=demo_config if demonstrator else None,
        compare=compare,
        show_warmup=show_warmup,
    )


def wait_until(qapp, condition, timeout=10.0):
    """Process Qt events until ``condition()`` holds, so background results can arrive."""
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        qapp.processEvents()
        time.sleep(0.005)


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


W = ESN_CONFIG.warmup_steps


def test_play_consumes_the_warmup_at_once_and_runs_from_time_0(qapp, demo_config, esn):
    app = make_app(demo_config, esn, demonstrator=False)
    app.canvas.solve_to_world(0.3, 0.9)  # pose the arm by dragging its tip
    start = app.skeleton.q.copy()
    expected = esn.generate(start, 100)

    app.play()
    app.pause()

    assert app.run is not None
    assert app.run.time == 0.0 and len(app.run.times) == W + 1
    assert np.array(app.run.q) == pytest.approx(np.repeat(start[np.newaxis], W + 1, axis=0))

    app.advance(1.0)

    assert np.array(app.run.q[W:]) == pytest.approx(expected, abs=1e-12)
    assert app.skeleton.q == pytest.approx(expected[-1])


def test_the_warmup_can_play_in_real_time(qapp, demo_config, esn):
    app = make_app(demo_config, esn, demonstrator=False, show_warmup=True)

    app.play()
    app.pause()
    assert app.run is not None and not app.run.times
    app.advance(ESN_CONFIG.warmup)

    assert len(app.run.times) == W
    assert app.run.times[0] == pytest.approx(-ESN_CONFIG.warmup)
    assert app.run.phase == "warming up"


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


def test_step_starts_at_time_0_then_takes_one_step_at_a_time(qapp, demo_config, esn):
    app = make_app(demo_config, esn, demonstrator=False)

    app.step()
    assert app.run is not None and app.run.time == 0.0
    app.step()

    assert len(app.run.times) == W + 2
    assert app.run.time == pytest.approx(ESN_CONFIG.dt)
    assert app.run.phase == "reaching"


def test_step_goes_through_the_warmup_when_it_plays_in_real_time(qapp, demo_config, esn):
    app = make_app(demo_config, esn, demonstrator=False, show_warmup=True)

    app.step()
    app.step()

    assert app.run is not None
    assert len(app.run.times) == 2
    assert app.run.phase == "warming up"


def test_play_does_not_wait_for_the_demonstrator(qapp, demo_config, esn):
    app = make_app(demo_config, esn)

    app.play()
    app.pause()

    run = app.run
    assert run is not None and run.time == 0.0  # the ESN has started
    assert run.demonstrator is None and run.demonstrator_pending  # its comparison comes later
    wait_until(qapp, lambda: run.demonstrator is not None)
    assert not run.demonstrator_pending


def test_metrics_follow_the_reach_and_the_hold(qapp, demo_config, esn):
    app = make_app(demo_config, esn)
    app.step()
    app.advance(3.0)
    run = app.run
    assert run is not None
    wait_until(qapp, lambda: run.demonstrator is not None)

    assert run.demonstrator is not None
    assert run.first_step is not None and run.first_step < 0.01
    assert run.arrival_time is not None
    assert run.arrival_time == pytest.approx(run.demonstrator.arrival_time, abs=0.05)
    assert run.reach_path_distance < 0.01
    assert run.reach_joint_error is not None and run.reach_joint_error < 1.0
    assert not run.left_goal and run.hold_observed == pytest.approx(1.0)
    assert run.phase == "held for the whole hold window"
    assert len(app.canvas.trails) == 2  # the demonstrator's path, then the ESN's on top


def test_the_demonstrator_is_simulated_only_while_comparing(qapp, demo_config, esn):
    late = make_app(demo_config, esn, compare=False)
    late.play()
    late.pause()
    late.advance(2.0)
    late_run = late.run
    assert late_run is not None
    time.sleep(0.2)
    qapp.processEvents()
    assert late_run.demonstrator is None and not late_run.demonstrator_pending
    assert "demonstrator" not in late.metrics_label.text()

    late.demonstrator_checkbox.setChecked(True)  # compare from now on, catching up on the steps taken
    assert late_run.demonstrator_pending
    wait_until(qapp, lambda: late_run.demonstrator is not None)

    early = make_app(demo_config, esn)  # compares as it goes, from the start
    early.play()
    early.pause()
    early_run = early.run
    assert early_run is not None
    wait_until(qapp, lambda: early_run.demonstrator is not None)
    early.advance(2.0)

    assert late_run.reach_path_distance == early_run.reach_path_distance
    assert late_run.reach_joint_error == early_run.reach_joint_error
    assert "demonstrator" in late.metrics_label.text()


def test_a_demonstrator_simulated_for_a_reset_run_is_discarded(qapp, demo_config, esn):
    app = make_app(demo_config, esn)
    app.play()
    app.reset()
    app.canvas.solve_to_world(0.3, 0.9)  # a different start posture
    app.play()
    app.pause()
    run = app.run
    assert run is not None

    wait_until(qapp, lambda: run.demonstrator is not None)

    assert run.demonstrator is not None
    assert run.demonstrator.q[0] == pytest.approx(run.start_q)


def test_a_model_for_another_robot_is_rejected(qapp, demo_config, esn):
    link = LinkProp(length=0.6, m=1.0, i=0.1, rgx=0.3, rgy=0.0, qmin=-np.pi, qmax=np.pi)

    with pytest.raises(ValueError, match="joint"):
        EsnReferenceApp(Skeleton([link, link, link]), Task.from_dict(demo_config["task"]), esn, hold=1.0)


def test_the_command_line_needs_a_model():
    args = build_parser().parse_args([str(CONFIG), "--model", "run/esn.toml", "--speed", "0.5"])

    assert args.model.name == "esn.toml" and args.speed == 0.5 and args.hold == 2.0
    assert not args.show_warmup and not args.no_demonstrator


def test_playing_runs_the_esn_on_its_own(qapp, demo_config, esn):
    app = make_app(demo_config, esn, demonstrator=False)
    app.speed_spin.setValue(2.0)

    app.play()
    steps_at_start = len(app.run.times) if app.run is not None else 0
    deadline = time.monotonic() + 0.2
    while time.monotonic() < deadline:  # let the playback clock tick
        qapp.processEvents()
        time.sleep(0.002)
    app.pause()

    assert app.speed == 2.0
    assert app.run is not None
    assert len(app.run.times) - steps_at_start >= 10  # about 200 ms x 2 / 10 ms per step, less timer jitter
    assert not app.is_playing


def test_space_plays_while_the_speed_box_has_the_focus(qapp, demo_config, esn):
    app = make_app(demo_config, esn, demonstrator=False)
    app.show()
    app.activateWindow()
    app.speed_spin.setFocus()
    qapp.processEvents()

    QTest.keyClick(app.speed_spin, Qt.Key.Key_Space)  # type: ignore[call-overload]  # PyQt6 stubs type QTest methods as bound

    assert app.is_playing
    app.pause()
    app.close()


@pytest.fixture(scope="module")
def task_esn(demo_config):
    """An ESN of two tasks: demonstration 6 as the first, demonstration 7 as the second."""
    trained = ReachingEsn(ESN_CONFIG)
    demos = [resample_joint_angles(log, ESN_CONFIG.dt)[1] for log in simulate_reaches(demo_config)]
    trained.fit(demos, contexts=[0, 1], names=["six", "seven"])
    return trained


def two_tasks(demo_config):
    """The demonstrations' task, and the same with another target."""
    other = demo_config["task"] | {"target": {"pos": [0.5, 0.9], "tolerance": 0.02}}
    return [Task.from_dict(demo_config["task"]), Task.from_dict(other)]


def make_task_app(demo_config, task_esn, tasks=None):
    skeleton = Skeleton.from_toml(CONFIG)
    skeleton.q = np.radians(demo_config["demonstrations"]["start_q"][1])
    tasks = two_tasks(demo_config) if tasks is None else tasks
    return EsnReferenceApp(skeleton, tasks[0], task_esn, hold=1.0, tasks=tasks)


def test_an_esn_of_one_task_has_no_task_buttons(qapp, demo_config, esn):
    app = make_app(demo_config, esn, demonstrator=False)

    assert app.task_buttons.buttons() == []
    with pytest.raises(ValueError, match="one task"):
        app.set_task(0)


def test_an_esn_of_several_tasks_has_a_button_and_a_target_for_each(qapp, demo_config, task_esn):
    app = make_task_app(demo_config, task_esn)

    assert [button.text() for button in app.task_buttons.buttons()] == ["1: six", "2: seven"]
    assert task_esn.context == pytest.approx([1.0, 0.0])  # the first task's, at the start
    targets = app.canvas.overlay_targets
    assert [bool(active) for *_, active in targets] == [True, False]
    assert targets[1][0] == pytest.approx([0.5, 0.9])


def test_a_task_button_changes_the_context_and_the_target_while_the_esn_runs(qapp, demo_config, task_esn):
    app = make_task_app(demo_config, task_esn)
    start = app.skeleton.q.copy()
    app.play()
    app.pause()
    for _ in range(40):
        app.step()
    assert app.run is not None
    state = task_esn.state()

    button = app.task_buttons.button(1)
    assert button is not None
    button.click()

    assert task_esn.context == pytest.approx([0.0, 1.0])
    assert task_esn.state() == pytest.approx(state)  # the reservoir keeps its state
    assert app.target == pytest.approx([0.5, 0.9])
    assert app.run.target == pytest.approx([0.5, 0.9])
    assert app.run.arrival_time is None
    assert [bool(active) for *_, active in app.canvas.overlay_targets] == [False, True]
    assert "Task: seven" in app.metrics_label.text()
    # The run goes on from where it was, now with the second task's context.
    for _ in range(10):
        app.step()
    task_esn.set_context(0)
    kept = task_esn.generate(start, 50)
    assert np.degrees(np.abs(np.array(app.run.q)[-1] - kept[-1]).max()) > 0.01
    app.close()


def test_the_arrival_is_measured_anew_after_a_change_of_task(qapp, demo_config, task_esn):
    app = make_task_app(demo_config, task_esn)
    app.play()
    app.pause()
    assert app.run is not None
    app.run.arrival_time, app.run.left_goal, app.run.hold_error, app.run.hold_observed = 0.5, True, 0.1, 0.3

    app.set_task(1)

    assert (app.run.arrival_time, app.run.left_goal, app.run.hold_error, app.run.hold_observed) == (
        None,
        False,
        None,
        0,
    )
    app.close()


def test_a_periodic_task_has_a_curve_and_no_target(qapp, demo_config, task_esn):
    ellipse = Task.from_dict({"type": "periodic_curve", "curve": "ellipse", "center": [0.4, 0.9], "a": 0.2, "b": 0.1})
    app = make_task_app(demo_config, task_esn, [two_tasks(demo_config)[0], ellipse])

    assert len(app.canvas.overlay_targets) == 1
    assert app.canvas.overlay_path is not None
    assert app.canvas.overlay_path[0] == pytest.approx([0.6, 0.9])
    app.set_task(1)
    assert np.isnan(app.radius)
    app.play()
    app.pause()
    for _ in range(5):
        app.step()
    assert app.run is not None and app.run.arrival_time is None
    assert "the task has no target" in app.metrics_label.text()
    app.close()


def test_the_tasks_must_be_the_esns(qapp, demo_config, task_esn):
    with pytest.raises(ValueError, match="2 tasks"):
        make_task_app(demo_config, task_esn, two_tasks(demo_config)[:1])
