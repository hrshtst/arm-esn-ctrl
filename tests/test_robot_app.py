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
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from skelarm import RecordedForce, Skeleton, StateLog

from arm_esn_ctrl.demonstrations import resample_joint_angles, simulate_reaches
from arm_esn_ctrl.disturbances import same_arm
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.storage import REPO_ROOT
from arm_esn_ctrl.tracking import EsnSource, ReplaySource, TrackerConfig, gain_posture, track, tracking_gains
from tools.robot_app import (
    RECORDED_FORCES,
    RobotApp,
    build_parser,
    check_arguments,
    joint_values,
    reference_demonstrations,
)

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


@pytest.fixture(scope="module")
def take(demo_config):
    """A recorded demonstration to replay: demonstration 7's reach, sampled every reference period."""
    return resample_joint_angles(simulate_reaches(demo_config)[1], ESN_CONFIG.dt)[1]


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


def test_the_launch_options_choose_the_mode(qapp, demo_config, esn, take):
    given = np.radians([25.0, 115.0])

    modes = {
        "esn": make_app(demo_config, esn=esn),
        "replay_from_start": make_app(demo_config),
        "replay_from_given": make_app(demo_config, given_q=given),
        "replay_take": make_app(demo_config, given_q=given, take=take, name="take.sklog.npz"),
        "demonstrator": make_app(demo_config, demonstrator=True),
    }

    for mode, app in modes.items():
        assert app.mode == mode
    assert "ESN" in modes["esn"].mode_label.text()
    assert "(25.0, 115.0) deg" in modes["replay_from_given"].mode_label.text()
    assert "computed torque, ω = 10 rad/s" in modes["replay_from_start"].mode_label.text()
    assert "demonstrator's own (time varying stiffness)" in modes["demonstrator"].mode_label.text()
    assert "the take take.sklog.npz, replayed by time" in modes["replay_take"].mode_label.text()


@pytest.mark.parametrize(
    "tracker",
    [
        COMPUTED_TORQUE,
        TrackerConfig("pd", 20.0, 0.02, damping=0.1, reference_velocity=False),
        TrackerConfig("pd", 20.0, 0.02, damping=0.4, control_period=0.01),
    ],
    ids=[
        "computed torque",
        "joint PD underdamped without the reference velocity",
        "joint PD with the torque held for 10 ms",
    ],
)
def test_a_take_is_replayed_as_the_experiment_replays_it_without_a_demonstrator(qapp, demo_config, take, tracker):
    config = {name: table for name, table in demo_config.items() if name != "controller"}  # no demonstrator
    app = make_app(config, take=take, tracker=tracker)
    start = app.skeleton.q.copy()

    app.play()
    app.pause()
    steps(app, 100)

    skeleton = Skeleton.from_toml(CONFIG)
    skeleton.q = start
    assert app.gains is not None
    log = track(
        skeleton, ReplaySource(take), tracker, app.gains, period=ESN_CONFIG.dt, warmup_steps=0, duration=1.0, dt=DT
    )
    assert app.run is not None and app.run.time == pytest.approx(1.0)
    assert app.skeleton.q == pytest.approx(log.channel("q")[-1], abs=1e-12)
    if not tracker.reference_velocity:
        assert "Reference velocity: zero" in app.mode_label.text()
    if tracker.control_period is not None:
        assert "Torque: computed every 10 ms, held between" in app.mode_label.text()


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


def test_the_gains_can_be_given_directly(qapp, demo_config):
    gains = (np.array([100.0, 20.0]), np.array([5.0, 1.0]))
    app = make_app(demo_config, tracker=TrackerConfig("pd", None, 0.02), gains=gains)

    app.step()

    assert app.run is not None and app.run.tracker is not None
    assert app.run.tracker.law.kp == pytest.approx(gains[0]) and app.run.tracker.law.kd == pytest.approx(gains[1])
    text = app.mode_label.text()
    assert "joint PD, kp = 100, 20; kd = 5, 1" in text and "damping ratio" in text


def test_the_panel_shows_how_damped_the_tracking_error_is(qapp, demo_config):
    underdamped = (np.array([100.0, 100.0]), np.array([4.0, 4.0]))  # computed torque: omega 10 rad/s, zeta 0.2

    app = make_app(demo_config, tracker=TrackerConfig("computed_torque", None, 0.02), gains=underdamped)

    assert "natural frequency 10, 10 rad/s, damping ratio 0.20, 0.20" in app.mode_label.text()
    assert "damping ratio 1.00, 1.00" in make_app(demo_config).mode_label.text()  # from omega: critically damped


def test_gains_must_be_positive_and_one_per_joint(qapp, demo_config):
    tracker = TrackerConfig("pd", None, 0.02)

    with pytest.raises(ValueError, match="positive"):
        make_app(demo_config, tracker=tracker, gains=(np.array([0.0, 1.0]), np.array([1.0, 1.0])))
    with pytest.raises(ValueError, match="per joint"):
        make_app(demo_config, tracker=tracker, gains=(np.array([1.0]), np.array([1.0])))


def test_the_control_period_must_be_a_whole_number_of_steps(qapp, demo_config):
    with pytest.raises(ValueError, match="control period"):
        make_app(demo_config, tracker=TrackerConfig("pd", 20.0, 0.02, control_period=0.005))  # 2.5 steps of 2 ms


def test_one_gain_applies_to_every_joint():
    parser = build_parser()

    assert joint_values(parser, "--kp", "50", 2) == pytest.approx([50.0, 50.0])
    assert joint_values(parser, "--kp", "50,7", 2) == pytest.approx([50.0, 7.0])
    with pytest.raises(SystemExit):
        joint_values(parser, "--kp", "1,2,3", 2)


@pytest.mark.parametrize("box", ["speed_spin", "stiffness_spin"])
def test_space_plays_while_a_number_box_has_the_focus(qapp, demo_config, box):
    app = make_app(demo_config)
    app.show()
    app.activateWindow()
    getattr(app, box).setFocus()
    qapp.processEvents()

    QTest.keyClick(getattr(app, box), Qt.Key.Key_Space)  # type: ignore[call-overload]  # PyQt6 stubs type QTest methods as bound

    assert app.is_playing
    app.pause()
    app.close()


@pytest.mark.parametrize(
    ("arguments", "complaint"),
    [
        (["--model", "esn.toml", "--demonstrator"], "exclude each other"),
        (["--demonstrator", "--law", "pd", "--omega", "10"], "does not use"),
        (["--law", "pd"], "required unless --demonstrator"),
        (["--law", "pd", "--omega", "10", "--model", "esn.toml", "--period", "0.02"], "only without"),
        (["--law", "pd", "--omega", "10", "--kp", "100", "--kd", "10"], "either --omega or --kp and --kd"),
        (["--law", "pd", "--kp", "100"], "go together"),
        (["--demonstrator", "--kp", "100", "--kd", "10"], "does not use"),
        (["--demonstrator", "--zero-reference-velocity"], "does not use"),
        (["--demonstrator", "--control-period", "0.01"], "does not use"),
        (["--law", "pd", "--kp", "100", "--kd", "10", "--damping", "0.1"], "with --omega"),
        (["--law", "pd", "--omega", "20", "--replay", "take.npz", "--model", "esn.toml"], "excludes"),
        (["--replay", "take.npz", "--demonstrator"], "excludes"),
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
        ["--law", "pd", "--kp", "100,20", "--kd", "5,1"],
        ["--law", "pd", "--omega", "20", "--damping", "0.1", "--zero-reference-velocity", "--replay", "take.npz"],
        ["--law", "pd", "--omega", "20", "--damping", "0.1", "--zero-reference-velocity", "--model", "esn.toml"],
        ["--law", "pd", "--omega", "20", "--damping", "0.4", "--control-period", "0.01", "--model", "esn.toml"],
    ):
        check_arguments(parser, parser.parse_args([str(CONFIG), *arguments]))


def pushed_run(app, n=120):
    """Run the app for ``n`` steps of 10 ms, pulling the tip around between 0.3 s and 0.8 s."""
    app.play()
    app.pause()
    for k in range(n):
        if k == 30:
            app.canvas.drag_point = (0.5, 1.6)
        elif k == 50:
            app.canvas.drag_point = (0.2, 1.9)
        elif k == 80:
            app.canvas.drag_point = None
        app.step()


def test_a_saved_log_is_the_experiments_log_of_the_run(qapp, demo_config, esn, tmp_path):
    """A saved log is on the task clock, with every step's channels, as robot_esn.py logs a run, and a final frame."""
    app = make_app(demo_config, esn=esn, provenance={"model": "a/esn.toml", "name": "a"})
    pushed_run(app)

    log = StateLog.load(app.save_log(tmp_path / "run.sklog.npz"))

    assert app.run is not None
    warmup = ESN_CONFIG.warmup_steps * ESN_CONFIG.dt
    assert log.times[0] == pytest.approx(-warmup)
    assert log.times[-1] == pytest.approx(app.run.time)
    assert np.diff(log.times) == pytest.approx(DT)
    assert {"q", "dq", "tau", "ext_force", "q_ref", "error"} <= set(log.channel_names)
    force = log.channel("ext_force")
    assert not np.any(force[log.times < 0])  # no force in the warm-up
    assert np.max(np.linalg.norm(force, axis=1)) > 1.0
    assert log.extra["tracking"]["law"] == "computed_torque"
    assert log.extra["tracking"]["reference"] == "esn"
    assert log.extra["robot_app"]["model"] == "a/esn.toml"
    assert log.extra["robot_app"]["dt"] == DT
    assert log.channel("q")[-1] == pytest.approx(app.skeleton.q)  # the final frame: where the arm is now


def test_logs_are_saved_by_default_where_experiments_share_them(qapp, demo_config, esn, tmp_path, monkeypatch):
    monkeypatch.setenv("ARM_ESN_CTRL_STORAGE_ROOT", str(tmp_path))
    app = make_app(demo_config, esn=esn, provenance={"name": "esn7"})

    path = app.default_log_path()

    assert path.parent == tmp_path / RECORDED_FORCES
    assert path.name.endswith("-esn7_computed_torque_w10_z1.sklog.npz")


def test_a_run_cannot_be_saved_before_it_starts(qapp, demo_config, esn):
    with pytest.raises(ValueError, match="no run to save"):
        make_app(demo_config, esn=esn).saved_log()


def test_a_headless_run_meeting_the_recorded_force_reproduces_the_app_run(qapp, demo_config, esn, tmp_path):
    """The experiments' tracking, given the app's recorded force, retraces the run the app showed."""
    app = make_app(demo_config, esn=esn)
    start = app.skeleton.q.copy()
    pushed_run(app)
    recorded = StateLog.load(app.save_log(tmp_path / "run.sklog.npz"))

    skeleton = Skeleton.from_toml(CONFIG)
    skeleton.q = start
    assert app.gains is not None
    again = track(
        skeleton,
        EsnSource(esn),
        COMPUTED_TORQUE,
        app.gains,
        period=ESN_CONFIG.dt,
        warmup_steps=ESN_CONFIG.warmup_steps,
        duration=1.2,
        dt=DT,
        external_force=RecordedForce.from_log(recorded).stepped(DT),
    )

    assert again.times == pytest.approx(recorded.times)
    assert np.array_equal(again.channel("ext_force")[:-1], recorded.channel("ext_force")[:-1])
    assert np.array_equal(again.channel("q"), recorded.channel("q"))


def test_recorded_forces_are_replayed_from_the_log_start_without_posing_or_dragging(qapp, demo_config, esn, tmp_path):
    """With a saved log's forces, a run starts where the log did and meets its forces, whatever the mouse does."""
    recording = make_app(demo_config, esn=esn)
    pushed_run(recording)
    recorded = StateLog.load(recording.save_log(tmp_path / "run.sklog.npz"))

    skeleton = Skeleton.from_toml(CONFIG)
    skeleton.q = demo7_start(demo_config) + np.radians([10.0, -10.0])  # elsewhere
    app = RobotApp(skeleton, demo_config, hold=1.0, tracker=COMPUTED_TORQUE, esn=esn, forces=recorded)
    assert app.skeleton.q == pytest.approx(recorded.channel("q")[0])
    before = app.skeleton.q.copy()
    app.canvas.mousePressEvent(left_click(10.0, 10.0))
    assert app.skeleton.q == pytest.approx(before)  # no posing

    app.play()
    app.pause()
    for _ in range(120):
        app.canvas.mousePressEvent(left_click(10.0, 10.0))  # no dragging either
        app.step()
    replayed = app.saved_log()

    assert app.canvas.drag_point is None
    assert np.array_equal(replayed.channel("ext_force")[:-1], recorded.channel("ext_force")[:-1])
    assert np.array_equal(replayed.channel("q"), recorded.channel("q"))
    assert "Forces: replayed" in app.mode_label.text()


def test_forces_recorded_with_another_arm_are_refused(qapp, demo_config, esn, tmp_path):
    recording = make_app(demo_config, esn=esn)
    pushed_run(recording, n=40)
    recorded = recording.saved_log()
    other = Skeleton.from_toml(CONFIG)
    other.links[1].prop.length *= 1.1
    assert not same_arm(other, Skeleton.from_toml(CONFIG))
    log = StateLog(other, producer="test", channel_meta=recorded.channel_meta)
    for i, time in enumerate(recorded.times):
        log.record(float(time), **{name: recorded.channel(name)[i] for name in recorded.channel_names})

    with pytest.raises(ValueError, match="another arm"):
        RobotApp(Skeleton.from_toml(CONFIG), demo_config, hold=1.0, tracker=COMPUTED_TORQUE, esn=esn, forces=log)


def test_the_command_line_takes_recorded_forces():
    args = build_parser().parse_args([str(CONFIG), "--law", "pd", "--omega", "20", "--forces", "pushes.sklog.npz"])
    assert args.forces.name == "pushes.sklog.npz"


JOINT_PD = TrackerConfig("pd", 20.0, 0.02, damping=0.4, control_period=0.01)


def training_takes(demo_config):
    """The joint angles the test ESN was trained on, every ESN period."""
    return [resample_joint_angles(log, ESN_CONFIG.dt)[1] for log in simulate_reaches(demo_config)]


def test_joint_pd_with_the_experiments_gains_retraces_the_app_run(qapp, demo_config, esn, tmp_path):
    """With joint PD's gains set where the training takes end, the experiments' tracking retraces an app run."""
    posture = gain_posture(training_takes(demo_config))
    app = make_app(demo_config, esn=esn, tracker=JOINT_PD, gains_at=posture)
    start = app.skeleton.q.copy()
    pushed_run(app)
    recorded = StateLog.load(app.save_log(tmp_path / "run.sklog.npz"))

    skeleton = Skeleton.from_toml(CONFIG)
    skeleton.q = start
    gains = tracking_gains(JOINT_PD, skeleton, posture)
    again = track(
        skeleton,
        EsnSource(esn),
        JOINT_PD,
        gains,
        period=ESN_CONFIG.dt,
        warmup_steps=ESN_CONFIG.warmup_steps,
        duration=1.2,
        dt=DT,
        external_force=RecordedForce.from_log(recorded).stepped(DT),
    )

    assert app.gains is not None
    assert np.array_equal(app.gains[0], gains[0]) and np.array_equal(app.gains[1], gains[1])
    assert np.array_equal(again.channel("q"), recorded.channel("q"))
    assert recorded.extra["tracking"]["kp"] == pytest.approx(gains[0].tolist())


def test_the_reference_demonstrations_are_the_esns_training_takes(demo_config, esn, tmp_path, monkeypatch):
    """With a model, the takes its run was trained on, as named in its run's configuration and read from the storage."""
    monkeypatch.setenv("ARM_ESN_CTRL_STORAGE_ROOT", str(tmp_path))
    run = tmp_path / "results" / "demonstrations" / "20261001-000000-demo"
    run.mkdir(parents=True)
    names = []
    for i, log in enumerate(simulate_reaches(demo_config)):
        log.save(run / f"demo_{i:02d}.sklog.npz")
        names.append(f"demo_{i:02d}.sklog.npz")
    model = tmp_path / "results" / "experiment" / "20261001-000001-esn" / "esn.toml"
    model.parent.mkdir(parents=True)
    train = ", ".join(f'"{name}"' for name in names)
    (model.parent / "config.toml").write_text(
        f'[demonstrations]\nrun = "results/demonstrations/20261001-000000-demo"\ntrain = [{train}]\n'
    )

    takes = reference_demonstrations(demo_config, model, esn, None, ESN_CONFIG.dt)

    assert len(takes) == len(names)
    for take, expected in zip(takes, training_takes(demo_config), strict=True):
        assert take == pytest.approx(expected, abs=1e-12)


def test_the_reference_demonstrations_of_the_other_modes(demo_config, take):
    """A replayed take is its own demonstration; the demonstrator's reaches are those of the configuration."""
    assert reference_demonstrations(demo_config, None, None, take, ESN_CONFIG.dt)[0] is take
    reaches = reference_demonstrations(demo_config, None, None, None, ESN_CONFIG.dt)
    assert gain_posture(reaches) == pytest.approx(gain_posture(training_takes(demo_config)))
    assert reference_demonstrations({"task": demo_config["task"]}, None, None, None, ESN_CONFIG.dt) == []


def test_a_log_names_the_commit_of_the_code_launched(qapp, demo_config, esn, monkeypatch):
    """The repository's state is taken at launch, the code that runs, not when a run starts."""
    monkeypatch.setattr("tools.robot_app.git_state", lambda: ("launched", False))
    app = make_app(demo_config, esn=esn)
    monkeypatch.setattr("tools.robot_app.git_state", lambda: ("edited since", True))
    app.play()
    app.pause()
    app.step()

    extra = app.saved_log().extra["robot_app"]
    assert (extra["commit"], extra["uncommitted_changes"]) == ("launched", False)
